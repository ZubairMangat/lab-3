# AI-316 Lab 03 - CV Prototyping & Object Detection
Setup: `pip install -r requirements.txt`

| Task | Run |
|---|---|
| 1 | `python task1_preprocessing.py traffic.jpg` |
| 2 | `python task2_defect_hsv.py part.jpg` (tune LOWER/UPPER HSV) |
| 3 | `python task3_medical_edges.py scan.png --noise 20` |
| 4 | `python task4_yolo_benchmark.py image.jpg` |
| 5 | `python task5_retail_count.py shelf.jpg 39` |
| 6 | `python task6_roi_intrusion.py scene.jpg` (edit ROI_FRACTIONS) |
| 7 | `python live_detection.py` |
| 8 | `python task8_surveillance_log.py` |

Tasks 1-3 fall back to a synthetic image if no path is given.
Keep `common.py` in the same folder. Tasks 7-8 need a local webcam.
