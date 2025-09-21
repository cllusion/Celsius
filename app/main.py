from fastapi import FastAPI, HTTPException, Depends, Header, Response
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
