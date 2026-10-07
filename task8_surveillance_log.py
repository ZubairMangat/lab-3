"""Task 8: Event-triggered surveillance. Saves alerts/YYYYMMDD_HHMMSS.jpg and appends to events.log.
Keys: q = quit.  Default targets: person (0) and cell phone (67), conf > 0.65.
"""
import os
import time
import argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from common import extract_detections, draw_detections

TARGETS = {0, 67}
CONF_TRIGGER = 0.65
COOLDOWN_S = 2.0            # avoid logging every single frame
LOG_FILE = "events.log"


def main(model_name, cam):
    os.makedirs("alerts", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,class,confidence,x1,y1,x2,y2\n")

    model = YOLO(model_name)
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    last_log, flash_until = 0.0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame, conf=0.3, verbose=False)[0]
            dets = extract_detections(res)
            draw_detections(frame, dets)
            hits = [d for d in dets if d["cls_id"] in TARGETS and d["conf"] > CONF_TRIGGER]
            now = time.time()

            if hits and now - last_log >= COOLDOWN_S:
                ts = datetime.now()
                stamp = ts.strftime("%Y%m%d_%H%M%S")
                clean = frame.copy()
                cv2.imwrite(os.path.join("alerts", f"{stamp}.jpg"), clean)
                with open(LOG_FILE, "a") as f:
                    for d in hits:
                        x1, y1, x2, y2 = d["box"]
                        f.write(f"{ts.isoformat(sep=' ', timespec='seconds')},{d['label']},"
                                f"{d['conf']:.3f},{x1},{y1},{x2},{y2}\n")
                print(f"[{stamp}] logged {len(hits)} event(s)")
                last_log, flash_until = now, now + 1.0

            if now < flash_until:          # visual indicator while logging
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)
                cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "REC / EVENT LOGGED", (52, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Surveillance [q: quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    main(a.model, a.cam)
