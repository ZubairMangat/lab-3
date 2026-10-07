"""Task 6: ROI intrusion alert. Usage: python task6_roi_intrusion.py scene.jpg
Edit ROI_FRACTIONS to match your camera view (fractions of width/height).
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from common import extract_detections, draw_detections, banner

ROI_FRACTIONS = [(0.30, 0.40), (0.75, 0.40), (0.85, 0.95), (0.20, 0.95)]
CONF = 0.4


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit("Cannot read image.")
    h, w = img.shape[:2]
    roi = np.array([(int(x * w), int(y * h)) for x, y in ROI_FRACTIONS], np.int32)

    model = YOLO("yolov8n.pt")
    res = model(img, conf=CONF, classes=[0], verbose=False)[0]      # class 0 = person
    people = extract_detections(res)

    intruders = 0
    out = img.copy()
    for d in people:
        x1, y1, x2, y2 = d["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        inside = cv2.pointPolygonTest(roi.reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0
        intruders += inside
        col = (0, 0, 255) if inside else (255, 128, 0)
        draw_detections(out, [d], col)
        cv2.circle(out, (cx, cy), 5, col, -1)

    color = (0, 0, 255) if intruders else (0, 255, 0)
    cv2.polylines(out, [roi], True, color, 3)
    if intruders:
        banner(out, f"INTRUSION DETECTED ({intruders})", color=(0, 0, 200))
    else:
        banner(out, "Area clear", color=(0, 120, 0))
    cv2.imwrite("task6_output.jpg", out)
    print(f"People: {len(people)}, inside ROI: {intruders} -> task6_output.jpg")


if __name__ == "__main__":
    main()
