"""Task 1: Traffic camera preprocessing (letterbox 640x640 + [0,1] normalisation).
Usage: python task1_preprocessing.py [path/to/traffic.jpg]
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def letterbox(img, size=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad the rest to size x size."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, (left, top)


def synthetic_traffic(w=1920, h=1080):
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.rectangle(img, (0, h // 3), (w, h), (60, 60, 60), -1)
    for x in range(0, w, 200):
        cv2.rectangle(img, (x, int(h * 0.66)), (x + 100, int(h * 0.68)), (230, 230, 230), -1)
    for (x, y, c) in [(300, 600, (0, 0, 200)), (900, 700, (200, 100, 0)), (1400, 620, (0, 180, 180))]:
        cv2.rectangle(img, (x, y), (x + 220, y + 110), c, -1)
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = cv2.imread(path) if path else None
    if img is None:
        print("No valid image given -> using synthetic traffic scene.")
        img = synthetic_traffic()

    lb, scale, (padx, pady) = letterbox(img, 640)
    norm = lb.astype(np.float32) / 255.0
    rgb_norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)

    print(f"Original : shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    print(f"Letterbox: shape={lb.shape}, dtype={lb.dtype}, min={lb.min()}, max={lb.max()}, scale={scale:.4f}, pad=({padx},{pady})")
    print(f"Normalised: shape={norm.shape}, dtype={norm.dtype}, min={norm.min():.3f}, max={norm.max():.3f}, mean={norm.mean():.3f}, std={norm.std():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Original {img.shape[1]}x{img.shape[0]}")
    ax[1].imshow(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Letterboxed 640x640 (uint8)")
    ax[2].imshow(rgb_norm)
    ax[2].set_title(f"Normalised [0,1] float32\nmin={norm.min():.2f} max={norm.max():.2f}")
    for a in ax:
        a.set_xlabel("x (pixels)")
        a.set_ylabel("y (pixels)")
    plt.tight_layout()
    plt.savefig("task1_output.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
