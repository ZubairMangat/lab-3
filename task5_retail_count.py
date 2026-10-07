"""Task 5: Class-filtered counting. Usage: python task5_retail_count.py shelf.jpg [class_id ...]
Default target: COCO 39 (bottle). 47 = apple.
"""
import sys
import cv2
from ultralytics import YOLO
from common import extract_detections, draw_detections, banner

CONF = 0.50


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    targets = [int(x) for x in sys.argv[2:]] or [39]
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit("Cannot read image.")
    model = YOLO("yolov8m.pt")
    res = model(img, conf=CONF, verbose=False)[0]
    dets = [d for d in extract_detections(res) if d["cls_id"] in targets and d["conf"] > CONF]
    names = sorted({d["label"] for d in dets}) or [model.names[t] for t in targets]
    out = draw_detections(img.copy(), dets)
    banner(out, f"Target Item Count: {len(dets)}  ({', '.join(names)})")
    cv2.imwrite("task5_output.jpg", out)
    print(f"Target Item Count: {len(dets)} -> task5_output.jpg")


if __name__ == "__main__":
    main()
