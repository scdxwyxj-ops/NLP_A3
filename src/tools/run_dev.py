import json
import sys
from pathlib import Path

# fix import path
sys.path.append("src")

from a3_factcheck.classifier.pipeline import Pipeline

# load data
dev_data = json.load(open("data/dev-claims.json"))
evidence = json.load(open("data/evidence.json"))

# TEMP retrieval (replace later with BM25)
def retrieve_fn(claim):
    return list(evidence.keys())[:3]

# build pipeline
pipeline = Pipeline(retrieve_fn, evidence)

predictions = {}

for claim_id, data in dev_data.items():
    claim_text = data["claim_text"]

    label, ev_ids = pipeline.run(claim_text)

    predictions[claim_id] = {
        "claim_label": label,
        "evidences": ev_ids
    }

output_path = Path("data/outputs/dev-output.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# save output
with output_path.open("w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=2)

print(f"{output_path} created")
