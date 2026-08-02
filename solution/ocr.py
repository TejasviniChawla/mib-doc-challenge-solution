"""Rendering + OCR with image cleanup for scanned/damaged pages.

Rendering a page rasterizes only what a human sees, so OCR output is
inherently injection-resistant: white-on-white and off-crop text never
survives rasterization.
"""

import cv2
import fitz
import numpy as np
import pytesseract

RENDER_DPI = 200


def render_page(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


def estimate_skew(img: np.ndarray) -> float:
    """Estimate rotation angle in degrees from near-horizontal text lines."""
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=img.shape[1] // 4, maxLineGap=8)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= 25:  # near-horizontal text lines only
            angles.append(ang)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def enhance(img: np.ndarray) -> np.ndarray:
    """Contrast-normalize washed-out scans."""
    lo, hi = np.percentile(img, (2, 98))
    if hi - lo < 10:
        return img
    stretched = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return stretched


def osd_rotation(img: np.ndarray) -> int:
    """Detect 90/180/270 page rotation via Tesseract OSD. Returns degrees."""
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


def rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return img
    k = (degrees // 90) % 4
    return np.rot90(img, k=4 - k).copy() if k else img


def ocr_image(img: np.ndarray, psm: int = 6) -> tuple[str, float]:
    """OCR an image. Returns (text, mean word confidence 0-100)."""
    data = pytesseract.image_to_data(
        img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        words.append(word)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return text, mean_conf


def ocr_page(page: fitz.Page) -> tuple[str, float]:
    """Full pipeline: render → orientation fix → deskew → enhance → OCR."""
    img = render_page(page)
    rot = osd_rotation(img)
    if rot:
        img = rotate(img, rot)
    img = deskew(img, estimate_skew(img))
    img = enhance(img)
    return ocr_image(img)
