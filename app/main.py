from fastapi import FastAPI, HTTPException, Depends, Header, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
from pydantic import BaseModel, Field
from river import linear_model, preprocessing, metrics
import threading
import uvicorn
import os
import pickle
from typing import Dict, Optional
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import tempfile
import shutil


class TrainRequest(BaseModel):
    features: Dict[str, float] = Field(..., example={"x": 1.0})
    target: float = Field(..., example=2.0)


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., example={"x": 1.0})


def get_auth_token():
    # Read token from environment for simple auth. Default is 'devtoken' for local dev.
    return os.environ.get("AUTH_TOKEN", "devtoken")


def require_auth(authorization: Optional[str] = Header(None)):
    token = get_auth_token()
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    # Expect header like: "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


app = FastAPI(title="Celsius")

# Simple online model using River
model = preprocessing.StandardScaler() | linear_model.LinearRegression()
train_count = 0
metric_mse = metrics.MSE()
prom_train_count = Gauge('celsius_train_count', 'Number of training examples')
prom_mse = Gauge('celsius_mse', 'Mean squared error')
model_lock = threading.Lock()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "celsius_model.pkl")
SESSIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "assistant_sessions.pkl")

# simple in-memory session store with persistence
assistant_sessions = {}


def load_sessions():
    global assistant_sessions
    try:
        if os.path.exists(SESSIONS_PATH):
            with open(SESSIONS_PATH, "rb") as f:
                assistant_sessions = pickle.load(f)
    except Exception:
        assistant_sessions = {}


def save_sessions():
    try:
        with open(SESSIONS_PATH, "wb") as f:
            pickle.dump(assistant_sessions, f)
    except Exception:
        pass


@app.on_event("startup")
def load_model():
    # Attempt to load persisted model
    try:
        path = os.path.abspath(MODEL_PATH)
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            # replace the global model in-place
            global model
            model = loaded
    except Exception:
        # ignore load errors for now
        pass
    # load assistant sessions
    load_sessions()


@app.on_event("shutdown")
def save_model():
    try:
        path = os.path.abspath(MODEL_PATH)
        # atomic save: write to temp then move
        dirn = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dirn)
        os.close(fd)
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
        shutil.move(tmp, path)
    except Exception:
        pass
    # persist sessions
    save_sessions()


@app.get("/status")
def status():
    return {"status": "ok", "model": str(model)}


@app.post("/train")
def train(req: TrainRequest, _auth=Depends(require_auth)):
    try:
        with model_lock:
            model.learn_one(req.features, req.target)
            # update metrics
            global train_count
            train_count += 1
            pred = model.predict_one(req.features)
            metric_mse.update(req.target, pred or 0.0)
            # update prometheus gauges
            prom_train_count.set(train_count)
            prom_mse.set(metric_mse.get())
        return {"status": "trained"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest, _auth=Depends(require_auth)):
    try:
        with model_lock:
            y = model.predict_one(req.features)
        return {"prediction": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/save")
def save(_auth=Depends(require_auth)):
    try:
        save_model()
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics_endpoint():
    return {"train_count": train_count, "mse": metric_mse.get()}


@app.get("/metrics/prometheus")
def prometheus_metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# Mount UI static files (simple assistant web UI)
ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui'))
if os.path.isdir(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")


class AssistantRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


@app.post("/assistant")
def assistant(req: AssistantRequest, _auth=Depends(require_auth)):
    """Simple rule-based assistant that can call model train/predict/status commands.

    Commands:
    - train x=1 y=2 target=3
    - predict x=1 y=2
    - status
    - help
    """
    sid = req.session_id or "default"
    sess = assistant_sessions.setdefault(sid, {"history": []})
    msg = req.message.strip()
    sess['history'].append({"role": "user", "text": msg})

    # simple parsing for train/predict
    def parse_features(text: str):
        items = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)=([-+]?[0-9]*\.?[0-9]+)", text)
        return {k: float(v) for k, v, *_ in items}

    reply = "I didn't understand. Try 'help' for commands."
    low = msg.lower()
    if low.startswith("help"):
        reply = "Commands: train x=1 y=2 target=3 | predict x=1 y=2 | status | metrics"
    elif low.startswith("status"):
        reply = f"status: train_count={train_count}, mse={metric_mse.get()}"
    elif low.startswith("metrics"):
        reply = f"metrics: train_count={train_count}, mse={metric_mse.get()}"
    elif low.startswith("train"):
        feats = parse_features(msg)
        if 'target' not in feats:
            reply = "train requires a 'target' value. Example: train x=1 y=2 target=3"
        else:
            target = feats.pop('target')
            with model_lock:
                model.learn_one(feats, target)
                global train_count
                train_count += 1
                pred = model.predict_one(feats)
                metric_mse.update(target, pred or 0.0)
                prom_train_count.set(train_count)
                prom_mse.set(metric_mse.get())
            reply = f"Trained on features={feats} target={target}."
    elif low.startswith("predict"):
        feats = parse_features(msg)
        if not feats:
            reply = "predict requires features. Example: predict x=1 y=2"
        else:
            with model_lock:
                p = model.predict_one(feats)
            reply = f"prediction: {p}"

    sess['history'].append({"role": "assistant", "text": reply})
    save_sessions()
    return {"session_id": sid, "reply": reply}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
