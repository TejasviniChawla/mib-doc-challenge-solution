"""Closed vocabularies observed in the public training labels, plus policy constants
from the public FIELD_MANUAL. Used to snap noisy OCR output onto valid values."""

SPECIES_CODES = [
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
]

HOME_WORLDS = [
    "Barnard-c",
    "Eris Relay",
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "TRAPPIST-1e",
    "Titan Freeport",
    "Wolf-1061c",
    "Zeta Reticuli",
]

DECLARED_PURPOSES = [
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
]

VISA_CLASSES = ["XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"]

FEE_STATUSES = ["paid", "waived", "unpaid", "unknown"]

DISQUALIFYING_FLAGS = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}

REVIEW_FLAGS = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}

ALL_RISK_FLAGS = sorted(DISQUALIFYING_FLAGS | REVIEW_FLAGS)

# Revoked sponsors listed in the public field manual; more may be learned from train data.
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040"}

ADJUDICATIONS = ["APPROVED", "DENIED", "NEEDS_REVIEW"]
