import threading
import time
import webbrowser

def launch_flask():
    from app import app
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  PredictNasdaq Desktop Window Launcher")
    print("=" * 65 + "\n")

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=launch_flask, daemon=True)
    flask_thread.start()
    time.sleep(1.5)

    try:
        import webview
        print("Launching native desktop window using pywebview...")
        webview.create_window(
            title="🔮 PredictNasdaq Engine",
            url="http://127.0.0.1:5000",
            width=1280,
            height=850,
            resizable=True
        )
        webview.start()
    except ImportError:
        print("Notice: 'pywebview' library is not installed.")
        print("Opening app in your default web browser (http://127.0.0.1:5000)...")
        print("Tip: You can run 'python gui_app.py' for the pure Tkinter Python GUI desktop app!")
        webbrowser.open("http://127.0.0.1:5000")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down desktop app server.")
