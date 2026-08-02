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
    # Only repair the digit portion of ID-like tokens, never the prefix.
    def repair(m: re.Match) -> str:
        return m.group(1) + m.group(2).translate(_DIGIT_FIX)

    return re.sub(r"((?:MIB|SPN)[-–—:\s]{0,2})([0-9OoIlSBZG]{4,6})", repair, text)


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
    v = value.lower().strip()
    if not v:
        return None
    # OCR often truncates these short words; first letters are distinctive
    # (paid/pac/pai, waived/waivec, unpaid/unpaic, unknown).
    result = process.extractOne(v, vocab.FEE_STATUSES, scorer=fuzz.WRatio)
    if result and result[1] >= 70:
        return result[0]
    if v.startswith("pa"):
        return "paid"
    if v.startswith("wa"):
        return "waived"
    if v.startswith("unp"):
        return "unpaid"
    if v.startswith("unk"):
        return "unknown"
    return None


def snap_name(value: str) -> str | None:
    """Normalize an applicant name onto the closed two-token vocabulary.

    Returns None for cut-out markers; falls back to the raw first two
    alpha tokens when snapping fails.
    """
    if not value:
        return None
    if fuzz.partial_ratio("CUT OUT", value.upper()) >= 80:
        return None
    toks = [t for t in re.split(r"[^A-Za-z]+", value) if len(t) >= 3]
    snapped = []
    for t in toks:
        r = process.extractOne(t.capitalize(), vocab.NAME_TOKENS, scorer=fuzz.ratio)
        if r and r[1] >= 70:
            snapped.append(r[0])
        elif len(snapped) < 2:
            snapped.append(t)
        if len(snapped) == 2:
            break
    if not snapped:
        return None
    return " ".join(snapped[:2])


def snap_flag(value: str) -> str | None:
    v = value.lower().strip().replace(" ", "_").strip("_")
    if len(v) < 5:
        return None
    # Flags appear in labeled contexts only, so a lower threshold is safe;
    # heavy OCR damage ("bichmzsrd_yed" -> biohazard_red) needs it.
    result = process.extractOne(v, vocab.ALL_RISK_FLAGS, scorer=fuzz.ratio)
    if result and result[1] >= 62:
        return result[0]
    return None
