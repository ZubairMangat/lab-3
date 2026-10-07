"""Task 4: YOLOv8n vs YOLOv8m benchmark.
Usage: python task4_yolo_benchmark.py image.jpg [--runs 10]
"""
import sys, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import Counter
from common import extract_detections, draw_detections

MODELS = ["yolov8n.pt", "yolov8m.pt"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 10
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit("Cannot read image.")

    rows, shown = [], []
    for name in MODELS:
        model = YOLO(name)
        model(img, verbose=False)                      # warm-up
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            res = model(img, verbose=False)[0]
            times.append((time.perf_counter() - t0) * 1000)
        dets = extract_detections(res)
        counts = Counter(d["label"] for d in dets)
        print(f"\n{name}:")
        for d in dets:
            print(f"  {d['label']:<15} conf={d['conf']:.3f} box={d['box']}")
        rows.append({"Model": name, "Latency ms (mean)": round(np.mean(times), 1),
                     "Latency ms (std)": round(np.std(times), 1),
                     "Detections": len(dets), "Distinct classes": len(counts),
                     "Avg confidence": round(float(np.mean([d['conf'] for d in dets])) if dets else 0, 3),
                     "Class counts": dict(counts)})
        shown.append((name, draw_detections(img.copy(), dets)))

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    df.to_csv("task4_comparison.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for a, (name, im) in zip(ax, shown):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        r = df[df.Model == name].iloc[0]
        a.set_title(f"{name} | {r['Latency ms (mean)']} ms | {r['Detections']} objs")
        a.axis("off")
    plt.tight_layout()
    plt.savefig("task4_output.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
