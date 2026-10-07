"""Task 2: Defect/colour isolation with HSV + LAB, inRange, threshold, bitwise_and.
Usage: python task2_defect_hsv.py [image.jpg]
Default HSV range targets rust (orange-brown). Tune LOWER/UPPER for your image.
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOWER = np.array([5, 80, 50])     # H, S, V  (OpenCV: H 0-179)
UPPER = np.array([25, 255, 255])


def synthetic_part():
    img = np.full((400, 600, 3), (170, 170, 170), np.uint8)       # grey metal
    img = cv2.GaussianBlur(img, (0, 0), 1)
    for (x, y, r) in [(150, 120, 40), (400, 250, 55), (480, 90, 25)]:
        cv2.circle(img, (x, y), r, (30, 85, 150), -1)              # rust (BGR)
    return img


def main():
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("Using synthetic metal part with rust patches.")
        img = synthetic_part()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    mask = cv2.inRange(hsv, LOWER, UPPER)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)   # remove speckles
    isolated = cv2.bitwise_and(img, img, mask=mask_bin)

    pct = 100 * np.count_nonzero(mask_bin) / mask_bin.size
    print(f"Defect area: {pct:.2f}% of image")

    h, s, v = cv2.split(hsv)
    L, A, B = cv2.split(lab)
    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original (BGR->RGB)", None),
        (h, "HSV - Hue", "hsv"), (s, "HSV - Saturation", "gray"), (v, "HSV - Value", "gray"),
        (L, "LAB - L", "gray"), (A, "LAB - a", "gray"), (B, "LAB - b", "gray"),
        (mask_bin, "Binary mask", "gray"),
        (cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB), "Isolated defect (bitwise AND)", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (im, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("task2_output.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
