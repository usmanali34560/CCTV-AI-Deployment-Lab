import os, time, logging
from collections import defaultdict
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cctv-ai")

SOURCE = os.getenv("VIDEO_SOURCE", "0")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
CONF = float(os.getenv("CONFIDENCE", "0.35"))
RECONNECT = int(os.getenv("RECONNECT_SECONDS", "3"))

if SOURCE.isdigit():
    SOURCE = int(SOURCE)

TARGET = {"person", "car", "motorcycle", "bus", "truck"}

model = YOLO(MODEL_NAME)

def open_capture(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        return None
    return cap

def main():
    cap = open_capture(SOURCE)
    if cap is None:
        raise RuntimeError(f"Unable to open source: {SOURCE}")

    frames = 0
    t0 = time.perf_counter()
    last_log = t0

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Camera/video read failed. Reconnecting...")
            cap.release()
            time.sleep(RECONNECT)
            cap = open_capture(SOURCE)
            continue

        start = time.perf_counter()
        result = model.track(frame, persist=True, conf=CONF, verbose=False)[0]
        annotated = result.plot()

        frames += 1
        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed if elapsed else 0.0

        cv2.putText(annotated, f"Inference FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if time.perf_counter() - last_log >= 5:
            avg_fps = frames / max(time.perf_counter() - t0, 1e-6)
            log.info("FPS=%.1f | last inference=%.3fs | GPU=%s",
                     avg_fps, elapsed, "CUDA" if model.device.type == "cuda" else "CPU")
            last_log = time.perf_counter()

        cv2.imshow("CCTV AI Deployment Lab", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
