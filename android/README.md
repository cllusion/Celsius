Android wrapper (WebView) for Celsius assistant

This folder contains a minimal Android Studio project skeleton that wraps the Celsius assistant UI in a WebView. It's intended as a starting point — open in Android Studio and build an APK or generate a signed bundle.

Steps to build (Android Studio):
1. Open Android Studio -> Open an existing project -> select this `android/` folder.
2. Edit `app/src/main/java/.../MainActivity.java` to set the default URL if needed (e.g., https://your-ngrok-url/ui/assistant.html or http://192.168.1.100:8000/ui/assistant.html).
3. Build -> Build Bundle(s) / APK(s) -> Build APK(s).
4. Install the generated APK on your phone (enable 'Install from unknown sources' if sideloading).

Notes:
- For local LAN usage, ensure your phone and desktop are on the same network and use the desktop IP.
- For remote usage, run `ngrok http 8000` and use the ngrok public URL in the app.
