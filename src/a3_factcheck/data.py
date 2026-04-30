import json
from collections import Counter
from pathlib import Path


LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def majority_label(claims):
    counts = Counter(
        claim["claim_label"]
        for claim in claims.values()
        if claim.get("claim_label") in LABELS
    )
    if not counts:
        return "NOT_ENOUGH_INFO"
    return counts.most_common(1)[0][0]

