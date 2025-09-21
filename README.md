# Celsius — Self-learning local AI (FastAPI online learner)

Quick start:

1. Run the PowerShell setup (from this folder) to create a venv and install deps:

   .\setup.ps1

2. Activate the virtual environment in PowerShell:

   . .venv\Scripts\Activate.ps1

3. Start the server:

   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. From your phone (on same LAN), open `http://<desktop-ip>:8000/docs` to explore the API.

Notes:
- The server uses River (online ML) so calls to `/train` update the model incrementally.
# Project name: Celsius
- If accessing from outside your LAN, use ngrok or configure your router/firewall. Opening ports has security implications.
- If you see a placeholder `python.exe` in WindowsApps, use the full path to the real interpreter shown in the earlier command: `C:\Users\micro\AppData\Local\Programs\Python\Python313\python.exe`.

Authentication & persistence
- The server expects a Bearer token via the `Authorization` header. By default the token is `devtoken` (for local use). For production, set the `AUTH_TOKEN` environment variable before starting uvicorn, e.g.:

   ```powershell
   $env:AUTH_TOKEN = "mysecret"
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

- The model is saved to `model.pkl` in the project root on shutdown and loaded on startup. Call `/save` to persist manually.

Accessing from your phone (LAN)
- Ensure your phone and desktop are on the same Wi-Fi.
- Find your desktop IP with `ipconfig` and use `http://<desktop-ip>:8000/docs` from the phone browser.

Remote access (ngrok)
- Install ngrok and run `ngrok http 8000`. Use the generated public URL to make requests from your phone outside the LAN. Be sure to set a strong `AUTH_TOKEN` when exposing the service.

Firewall
- If the server is not reachable from other devices on your LAN, ensure Windows Firewall allows incoming connections on port 8000 for Python or explicitly add a rule for port 8000.


Files:
- `app/main.py` - FastAPI server
- `requirements.txt` - python deps
- `setup.ps1` - helper to create venv and install deps
- `tests/smoke_test.py` - simple train/predict smoke test

Next steps:
- Add authentication (API token) before exposing server publicly
- Persist model state to disk between restarts
- Add more advanced online learners and data validation
