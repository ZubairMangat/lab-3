"""Task 3: Noise reduction + edge detection on a grayscale medical scan.
Usage: python task3_medical_edges.py [scan.png] [--noise SIGMA]
Quantitative metric: edge maps are compared against a reference edge map
(Canny on the clean image). Reports edge density, precision, recall, F1
(3px tolerance) for: no filter, Gaussian, Median.
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def phantom(size=400):
    img = np.zeros((size, size), np.uint8)
    cv2.ellipse(img, (200, 200), (150, 180), 0, 0, 360, 90, -1)
    cv2.circle(img, (150, 170), 45, 170, -1)
    cv2.circle(img, (260, 230), 35, 200, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def sobel_mag(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, np.uint8(255 * mag / (mag.max() + 1e-9))


def prf(edge, ref, tol=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ref_d, edge_d = cv2.dilate(ref, k), cv2.dilate(edge, k)
    tp_p = np.count_nonzero((edge > 0) & (ref_d > 0))
    tp_r = np.count_nonzero((ref > 0) & (edge_d > 0))
    p = tp_p / max(np.count_nonzero(edge), 1)
    r = tp_r / max(np.count_nonzero(ref), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    args = [a for a in sys.argv[1:]]
    sigma = 20.0
    if "--noise" in args:
        i = args.index("--noise"); sigma = float(args[i + 1]); del args[i:i + 2]
    clean = cv2.imread(args[0], cv2.IMREAD_GRAYSCALE) if args else None
    if clean is None:
        print("Using synthetic phantom.")
        clean = phantom()

    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)

    variants = {
        "Noisy (no filter)": noisy,
        "Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
        "Median k=5": cv2.medianBlur(noisy, 5),
    }
    T_LOW, T_HIGH = 50, 150
    ref = cv2.Canny(cv2.GaussianBlur(clean, (5, 5), 0), T_LOW, T_HIGH)

    rows, results = [], []
    for name, im in variants.items():
        gx, gy, mag = sobel_mag(im)
        canny = cv2.Canny(im, T_LOW, T_HIGH)
        p, r, f1 = prf(canny, ref)
        psnr = cv2.PSNR(clean, im)
        results.append((name, im, mag, canny))
        rows.append((name, psnr, 100 * np.count_nonzero(canny) / canny.size, p, r, f1))

    print(f"\nNoise sigma={sigma}, Canny T_lower={T_LOW}, T_upper={T_HIGH}")
    print(f"{'Filter':<20}{'PSNR(dB)':>10}{'Edge%':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for n, ps, d, p, r, f in rows:
        print(f"{n:<20}{ps:>10.2f}{d:>9.2f}{p:>8.3f}{r:>8.3f}{f:>8.3f}")

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    ax[0, 0].imshow(clean, cmap="gray"); ax[0, 0].set_title("Clean reference")
    ax[1, 0].imshow(ref, cmap="gray"); ax[1, 0].set_title("Reference Canny edges")
    ax[2, 0].axis("off")
    for j, (name, im, mag, canny) in enumerate(results):
        pass
    for i, (name, im, mag, canny) in enumerate(results):
        ax[i, 1].imshow(im, cmap="gray"); ax[i, 1].set_title(name)
        ax[i, 2].imshow(mag, cmap="gray"); ax[i, 2].set_title(f"Sobel magnitude - {name}")
        ax[i, 3].imshow(canny, cmap="gray"); ax[i, 3].set_title(f"Canny - {name}")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig("task3_output.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
