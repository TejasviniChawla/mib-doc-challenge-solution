"""Field extraction helpers: regexes, normalizers, and OCR-tolerant fuzzy
snapping onto the closed vocabularies. Layout anchors live in pipeline.py."""

import re

from rapidfuzz import fuzz, process

from solution import vocab

CASE_ID_RE = re.compile(r"MIB[-–—:\s]{0,2}(\d{6})")
SPONSOR_RE = re.compile(r"SPN[-–—:\s]{0,2}(\d{4})")
DATE_ISO_RE = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")

# Common OCR confusions for digit restoration inside IDs.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def find_case_ids(text: str) -> list[str]:
    fixed = _fix_id_digits(text)
    return [f"MIB-{m}" for m in CASE_ID_RE.findall(fixed)]


def find_sponsor_ids(text: str) -> list[str]:
    fixed = _fix_id_digits(text)
    return [f"SPN-{m}" for m in SPONSOR_RE.findall(fixed)]


def _fix_id_digits(text: str) -> str:
    # Only repair characters that appear inside ID-like tokens.
    def repair(m: re.Match) -> str:
        return m.group(0).translate(_DIGIT_FIX)

    return re.sub(r"(?:MIB|SPN)[-–—:\s]{0,2}[0-9OoIlSBZG]{4,6}", repair, text)


def normalize_date(text: str) -> str | None:
    m = DATE_ISO_RE.search(text)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None


def snap(value: str, choices: list[str], min_score: float = 72.0) -> str | None:
    """Fuzzy-snap a noisy OCR value onto a closed vocabulary."""
    if not value or not value.strip():
        return None
    result = process.extractOne(value.strip(), choices, scorer=fuzz.WRatio)
    if result and result[1] >= min_score:
        return result[0]
    return None


def snap_species(value: str) -> str | None:
    return snap(value.upper().replace(" ", "_"), vocab.SPECIES_CODES)


def snap_world(value: str) -> str | None:
    return snap(value, vocab.HOME_WORLDS)


def snap_purpose(value: str) -> str | None:
    return snap(value.lower(), vocab.DECLARED_PURPOSES)


def snap_visa(value: str) -> str | None:
    v = value.upper().strip()
    # Visa classes are short; enforce structure before fuzzy matching.
    v = v.replace(" ", "").replace("–", "-").replace("—", "-")
    for cls in vocab.VISA_CLASSES:
        if cls.replace("-", "") in v.replace("-", ""):
            return cls
    return snap(v, vocab.VISA_CLASSES, min_score=80)


def snap_fee(value: str) -> str | None:
    return snap(value.lower(), vocab.FEE_STATUSES, min_score=75)


def snap_flag(value: str) -> str | None:
    return snap(value.lower().replace(" ", "_"), vocab.ALL_RISK_FLAGS, min_score=80)
