"""Task 7: Real-time webcam detection with FPS. Keys: q = quit, s = save frame."""
import time
import argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from common import extract_detections, draw_detections


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA)


def run(model_name="yolov8n.pt", cam=0, conf=0.4):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    fps, prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame, conf=conf, verbose=False)[0]
            draw_detections(frame, extract_detections(res))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev, 1e-6)) if fps else 1 / max(now - prev, 1e-6)
            prev = now
            draw_fps(frame, fps)
            cv2.imshow("Live Detection  [q: quit, s: save]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = datetime.now().strftime("frame_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, frame)
                print("Saved", name)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    a = ap.parse_args()
    run(a.model, a.cam, a.conf)
