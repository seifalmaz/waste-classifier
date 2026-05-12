import cv2
import time
import json
import asyncio
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from detection.yolo_detector import YOLODetector
from inference.classifier import WasteClassifier
from inference.smoothing import PredictionSmoother

from pathlib import Path

# =====================================================
# PATHS & CONFIGURATION
# =====================================================

# Build absolute paths relative to this script's location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR.parent / "models" / "efficientnet_b3_finetuned_best.pth").resolve()

# Startup verification
print(f"\n[SYSTEM] Initializing AI Waste Detection System...")
print(f"[SYSTEM] Resolving Model Path: {MODEL_PATH}")

if not MODEL_PATH.exists():
    error_msg = f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. Please ensure the 'models' directory exists in the project root."
    print(f"\n[ERROR] {error_msg}\n")
    raise FileNotFoundError(error_msg)

app = FastAPI(title="Waste Detection AI Dashboard")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (using absolute path)
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =====================================================
# SYSTEM STATE
# =====================================================

class GlobalState:
    def __init__(self):
        self.detector = YOLODetector(confidence_threshold=0.35)
        self.classifier = WasteClassifier(model_path=MODEL_PATH, mode="realtime")
        self.upload_classifier = WasteClassifier(model_path=MODEL_PATH, mode="upload")
        self.smoother = PredictionSmoother(history_size=10)
        self.cap = None
        self.is_running = False
        self.last_results = {"detections": [], "fps": 0}
        self.session_stats = {
            "Plastic": 0,
            "Metal": 0,
            "Glass": 0,
            "Paper": 0,
            "Organic": 0,
            "Unknown": 0
        }

state = GlobalState()

# =====================================================
# HELPERS
# =====================================================

def get_camera():
    if state.cap is None or not state.cap.isOpened():
        state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return state.cap

async def generate_frames():
    cap = get_camera()
    prev_time = time.time()
    frame_count = 0
    
    YOLO_SKIP_FRAMES = 2
    CLASSIFY_EVERY = 6
    
    cached_detections = []
    cached_prediction = "Unknown"
    cached_confidence = 0.0
    previous_roi = None

    while state.is_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO Detection logic (mirrored from realtime_demo.py)
        if frame_count % YOLO_SKIP_FRAMES == 0:
            detections = state.detector.detect(rgb)
            cached_detections = []
            
            for detection in detections:
                x, y, w, h = detection["bbox"]
                roi = detection["roi"]
                if roi.size == 0: continue

                should_classify = False
                if previous_roi is None:
                    should_classify = True
                else:
                    try:
                        prev_small = cv2.resize(previous_roi, (64, 64))
                        curr_small = cv2.resize(roi, (64, 64))
                        diff = np.mean(cv2.absdiff(prev_small, curr_small))
                        if diff > 25: should_classify = True
                    except:
                        should_classify = True

                if frame_count % CLASSIFY_EVERY == 0:
                    should_classify = True

                if should_classify:
                    result = state.classifier.predict(roi)
                    pred_class = result["class"]
                    conf = result["confidence"]
                    
                    state.smoother.update(pred_class, conf)
                    pred_class, conf = state.smoother.get_smoothed_prediction()
                    
                    if conf < 0.60: pred_class = "Unknown"
                    
                    if pred_class != cached_prediction and pred_class != "Unknown":
                        state.session_stats[pred_class] += 1
                        
                    cached_prediction = pred_class
                    cached_confidence = conf
                    previous_roi = roi.copy()

                cached_detections.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "class": cached_prediction,
                    "confidence": float(cached_confidence)
                })

        # Draw overlays for the stream
        for det in cached_detections:
            x, y, w, h = det["bbox"]
            color = (0, 255, 0) # Default green
            if det["class"] == "Plastic": color = (0, 0, 255)
            elif det["class"] == "Metal": color = (180, 180, 180)
            elif det["class"] == "Glass": color = (255, 200, 0)
            elif det["class"] == "Paper": color = (0, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{det['class']} {det['confidence']*100:.0f}%", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        state.last_results = {
            "detections": cached_detections,
            "fps": int(fps),
            "stats": state.session_stats
        }

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0.01)

# =====================================================
# ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Error: static/index.html not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    state.is_running = True
    return StreamingResponse(generate_frames(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stop_feed")
async def stop_feed():
    state.is_running = False
    if state.cap:
        state.cap.release()
        state.cap = None
    return {"status": "stopped"}

@app.get("/results")
async def get_results():
    return state.last_results

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detections = state.detector.detect(rgb)
    results = []
    
    for det in detections:
        roi = det["roi"]
        if roi.size == 0: continue
        
        class_res = state.upload_classifier.predict(roi)
        results.append({
            "bbox": [int(c) for c in det["bbox"]],
            "class": class_res["class"],
            "confidence": class_res["confidence"],
            "probabilities": class_res["probabilities"]
        })
    
    # Encode original image with boxes for preview
    vis = img.copy()
    for res in results:
        x, y, w, h = res["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(vis, f"{res['class']}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    _, buffer = cv2.imencode('.jpg', vis)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "results": results,
        "image": f"data:image/jpeg;base64,{img_base64}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
