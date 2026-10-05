# CCTV AI Deployment Lab

Selection-oriented prototype for a Computer Vision & AI Deployment Engineer role.

Pipeline:
Camera / Video / RTSP -> OpenCV -> YOLO -> Tracking -> Events -> FastAPI -> Metrics

## Features
- Webcam, local video, or RTSP input
- YOLO/Ultralytics object detection
- Lightweight centroid tracking with persistent IDs
- Person/vehicle event counting
- FPS and latency monitoring
- Camera reconnect handling
- FastAPI health/metrics endpoints
- Structured logging
- Docker + systemd examples
- GPU-aware configuration
- Production deployment notes

## Run
Python 3.11+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

For API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs

## Input
Set VIDEO_SOURCE in `.env`:
- `0` for webcam
- `sample.mp4` for local video
- `rtsp://user:password@CAMERA_IP:554/stream` for RTSP

Never commit real camera credentials.

## GPU
Install a CUDA-compatible PyTorch build separately when using NVIDIA GPU. The application automatically selects CUDA when available.

## Important
This is a portfolio/technical-assessment prototype. It demonstrates the architecture and engineering approach; it does not claim previous customer production deployment.
