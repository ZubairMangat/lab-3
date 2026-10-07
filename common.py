"""Shared helpers for YOLO tasks."""
import cv2
import numpy as np


def extract_detections(result):
    """Return list of dicts: box (x1,y1,x2,y2), conf, cls_id, label."""
    out = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for b, c, k in zip(xyxy, conf, cls):
        out.append({"box": tuple(int(v) for v in b), "conf": float(c),
                    "cls_id": int(k), "label": result.names[int(k)]})
    return out


def draw_detections(img, dets, color=(0, 255, 0)):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def banner(img, text, color=(0, 0, 0), fg=(255, 255, 255), scale=0.9):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], th + 20), color, -1)
    cv2.putText(img, text, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)
    return img
