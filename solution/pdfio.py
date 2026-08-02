"""PDF text acquisition with trust classification.

Implements the FIELD_MANUAL evidence rule: visible document evidence wins;
hidden text (white-on-white, outside the page crop, invisible render mode)
is untrusted and must never supply field values.
"""

from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class Span:
    text: str
    bbox: tuple  # (x0, y0, x1, y1)
    page: int
    color: int  # sRGB int
    size: float
    flags: int
    hidden_reasons: list = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        return not self.hidden_reasons


def _luminance(srgb: int) -> float:
    r, g, b = (srgb >> 16) & 255, (srgb >> 8) & 255, srgb & 255
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def classify_spans(page: fitz.Page, page_index: int) -> list[Span]:
    """Extract text-layer spans and mark each with hidden/untrusted reasons."""
    spans: list[Span] = []
    crop = page.rect  # visible area (CropBox)
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                text = s.get("text", "")
                if not text.strip():
                    continue
                sp = Span(
                    text=text,
                    bbox=tuple(s["bbox"]),
                    page=page_index,
                    color=s.get("color", 0),
                    size=s.get("size", 0.0),
                    flags=s.get("flags", 0),
                )
                bbox = fitz.Rect(sp.bbox)
                # Outside the visible crop (fully or mostly).
                inter = bbox & crop
                if inter.is_empty or inter.get_area() < 0.3 * max(bbox.get_area(), 1e-6):
                    sp.hidden_reasons.append("off_crop")
                # Near-white text (white-on-white trick). Background is
                # near-white in these scanned packets; verified during recon.
                if _luminance(sp.color) > 0.92:
                    sp.hidden_reasons.append("white_text")
                # Microscopic font — invisible to a human reader.
                if sp.size and sp.size < 2.0:
                    sp.hidden_reasons.append("tiny_font")
                spans.append(sp)
    return spans


def page_visible_text(spans: list[Span]) -> str:
    return "\n".join(s.text for s in spans if s.trusted)


def page_hidden_text(spans: list[Span]) -> str:
    return "\n".join(s.text for s in spans if not s.trusted)


def open_pdf(path: str) -> fitz.Document:
    return fitz.open(path)
