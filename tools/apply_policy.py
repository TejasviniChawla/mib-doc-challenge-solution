"""Re-derive adjudication + confidence from cached features without re-running
OCR. Lets the decision layer iterate in seconds instead of ~13 minutes.

Usage:
  python tools/apply_policy.py --features F.jsonl --predictions P.jsonl --out NEW.jsonl
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import solution.calib as calib

    importlib.reload(calib)
    from solution.pipeline import confidence_for

    feats = {}
    for line in open(args.features):
        f = json.loads(line)
        feats[f["case_id"]] = f

    with open(args.out, "w") as out:
        for line in open(args.predictions):
            r = json.loads(line)
            f = feats.get(r["case_id"])
            if f:
                rule = f.get("rule", "?")
                new_decision = calib.POLICY.get(rule) if hasattr(calib, "POLICY") else None
                if new_decision:
                    r["adjudication"] = new_decision
                    f = {**f, "decision": new_decision}
                r["confidence"] = confidence_for(f)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
