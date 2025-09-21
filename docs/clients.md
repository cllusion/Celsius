Celsius client examples

1) curl (training)

curl -X POST "http://<DESKTOP_IP>:8000/train" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"features":{"x":1.0},"target":2.0}'

2) curl (predict)

curl -X POST "http://<DESKTOP_IP>:8000/predict" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"features":{"x":1.0}}'

3) Android (Termux) example

pkg install curl
curl -X POST "http://<DESKTOP_IP>:8000/predict" -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"features":{"x":1.0}}'

4) iOS Shortcuts

- Create a new Shortcut that does a POST to the above URL with JSON body and adds the Authorization header.
 
Assistant UI

- A simple assistant web UI is available at `http://<DESKTOP_IP>:8000/ui/assistant.html`.
- Default token is `devtoken`; change `AUTH_TOKEN` env var for production.
- The assistant accepts commands: `help`, `status`, `metrics`, `train x=1 y=2 target=3`, `predict x=1 y=2`.

Mobile access notes

- If your phone is on the same Wi-Fi, open the assistant UI in the phone browser using `http://<DESKTOP_IP>:8000/ui/assistant.html`.
- For remote access, use `ngrok http 8000` and open the provided ngrok URL in your phone browser. Remember to set a strong `AUTH_TOKEN` before exposing the service.
