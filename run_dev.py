import json
import sys

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

# save output
with open("dev-output.json", "w") as f:
    json.dump(predictions, f, indent=2)

print(" dev-output.json created")