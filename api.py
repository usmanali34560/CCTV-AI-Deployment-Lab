import os, time
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI(title="CCTV AI Inference API", version="1.0.0")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
model = YOLO(MODEL_NAME)
started = time.time()

@app.get("/")
def root():
    return {"service": "CCTV AI Inference API", "status": "online"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - started, 1),
        "device": str(model.device)
    }

@app.get("/metrics")
def metrics():
    return {
        "model": MODEL_NAME,
        "device": str(model.device),
        "cuda_available": str(model.device).startswith("cuda")
    }
