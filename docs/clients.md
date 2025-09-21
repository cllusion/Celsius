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
