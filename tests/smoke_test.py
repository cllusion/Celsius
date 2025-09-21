import requests
import time

BASE = "http://127.0.0.1:8000"
HEADERS = {"Authorization": "Bearer devtoken"}

# Wait briefly for server
for i in range(10):
    try:
        r = requests.get(BASE + "/status", timeout=1)
        if r.status_code == 200:
            break
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("Server did not start")

# Train with a simple linear example
train_data = {"features": {"x": 1.0}, "target": 2.0}
resp = requests.post(BASE + "/train", json=train_data, headers=HEADERS)
print("train", resp.status_code, resp.json())

# Predict
pred = requests.post(BASE + "/predict", json={"features": {"x": 1.0}}, headers=HEADERS)
print("predict", pred.status_code, pred.json())

# Save model
sv = requests.post(BASE + "/save", headers=HEADERS)
print("save", sv.status_code, sv.json())
