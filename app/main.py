import uvicorn
import webbrowser
import threading
import time
from web_server import app

def open_browser():
    # Wait a moment for server to initialize
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("EcoMind AI: High-Performance Waste Detection")
    print("="*50 + "\n")
    
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
