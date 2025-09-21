package com.celsius.assistant;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private WebView webView;
    private SharedPreferences prefs;
    private static final String PREFS = "celsius_prefs";
    private static final String KEY_URL = "server_url";
    private static final String KEY_TOKEN = "auth_token";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        webView = new WebView(this);
        setContentView(webView);
        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webView.setWebViewClient(new WebViewClient());

        String url = prefs.getString(KEY_URL, null);
        String token = prefs.getString(KEY_TOKEN, null);

        if (url == null) {
            promptForConfig();
        } else {
            loadWithToken(url, token);
        }
    }

    private void promptForConfig() {
        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Configure Celsius");
        final android.widget.LinearLayout layout = new android.widget.LinearLayout(this);
        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        final android.widget.EditText urlInput = new android.widget.EditText(this);
        urlInput.setHint("Server URL, e.g. http://192.168.1.100:8000/ui/assistant.html");
        final android.widget.EditText tokenInput = new android.widget.EditText(this);
        tokenInput.setHint("Auth token");
        layout.addView(urlInput);
        layout.addView(tokenInput);
        builder.setView(layout);
        builder.setPositiveButton("Save", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                String url = urlInput.getText().toString().trim();
                String token = tokenInput.getText().toString().trim();
                prefs.edit().putString(KEY_URL, url).putString(KEY_TOKEN, token).apply();
                loadWithToken(url, token);
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });
        builder.show();
    }

    private void loadWithToken(String url, String token) {
        webView.loadUrl(url);
        if (token != null && token.length() > 0) {
            // inject token into JS context
            webView.evaluateJavascript("window.__AUTH_TOKEN = '" + token.replace("'", "\\'") + "';", null);
        }
    }
}
