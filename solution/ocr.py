"""Rendering + OCR for scanned/damaged pages.

Empirically (train recon): raw 300dpi grayscale + psm 6 beats aggressive
preprocessing on these synthetic scans; binarization/unsharp destroy the
ink-degraded bold text. Preprocessing variants are used only as retries
when the raw pass has low confidence. Rasterization is inherently
injection-resistant: hidden text never survives rendering.
"""

import cv2
import fitz
import numpy as np
import pytesseract

RENDER_DPI = 300
GOOD_CONF = 60.0


def render_page(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


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


def rotate90(img: np.ndarray, degrees: int) -> np.ndarray:
    k = (degrees // 90) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=4 - k))


def estimate_skew(img: np.ndarray) -> float:
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=img.shape[1] // 4, maxLineGap=8,
    )
    if lines is None:
        return 0.0
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
        if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) <= 25
    ]
    return float(np.median(angles)) if angles else 0.0


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.4:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def stretch(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (2, 98))
    if hi - lo < 10 or (lo < 30 and hi > 225):
        return img
    return np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _ocr(img: np.ndarray, psm: int) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple, list[str]] = {}
    confs = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text, (float(np.mean(confs)) if confs else 0.0)


def ocr_page(page: fitz.Page) -> tuple[str, float]:
    text, conf, _ = ocr_page_full(page)
    return text, conf


def ocr_page_full(page: fitz.Page) -> tuple[str, float, np.ndarray]:
    """OCR one page, escalating through variants until confidence is good.

    Bounded work: at most 3 tesseract passes plus one OSD probe, keeping the
    worst case within the per-PDF runtime budget. Returns the image variant
    that produced the best text so callers can run targeted re-OCR on it.
    """
    img = render_page(page)

    best_text, best_conf = _ocr(img, 6)
    best_img = img
    if best_conf >= GOOD_CONF:
        return best_text, best_conf, best_img

    # Low confidence: probe for 90/180/270 rotation, retry once if rotated.
    rot = osd_rotation(img)
    if rot:
        img = rotate90(img, rot)
        text, conf = _ocr(img, 6)
        if conf > best_conf:
            best_text, best_conf, best_img = text, conf, img
        if best_conf >= GOOD_CONF:
            return best_text, best_conf, best_img

    # One more retry: deskew if tilted, else contrast stretch.
    angle = estimate_skew(img)
    variant = deskew(img, angle) if abs(angle) >= 0.4 else stretch(img)
    text, conf = _ocr(variant, 6)
    if conf > best_conf:
        best_text, best_conf, best_img = text, conf, variant
    return best_text, best_conf, best_img


ID_WORD = None  # lazy-compiled regex


def precise_ids(img: np.ndarray) -> list[str]:
    """Re-OCR SPN-####/MIB-###### tokens from word crops with a digit
    whitelist at 2x scale — digits in degraded scans are the main source of
    exact-match extraction misses."""
    global ID_WORD
    import re

    if ID_WORD is None:
        ID_WORD = re.compile(r"(SPN|MIB|SPH|SPM|5PN)", re.I)
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    out = []
    h, w = img.shape[:2]
    for i, word in enumerate(data["text"]):
        if not word.strip() or not ID_WORD.search(word):
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # Include the following word too (value may be split, e.g. "SPN-" "2913").
        x2 = min(w, x + bw + int(bw * 1.8))
        pad = max(4, bh // 3)
        crop = img[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):x2]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        txt = pytesseract.image_to_string(
            big,
            config="--psm 7 -c tessedit_char_whitelist=SPNMIB-0123456789",
        ).strip()
        if txt:
            out.append(txt)
    return out
