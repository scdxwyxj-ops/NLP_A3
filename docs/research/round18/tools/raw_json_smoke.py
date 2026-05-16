from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import (
    ALLOWED_DATA_FILES,
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


def describe_claims(claims: dict) -> dict:
    labelled = sum(1 for item in claims.values() if "claim_label" in item)
    with_evidence = sum(1 for item in claims.values() if "evidences" in item)
    labels: dict[str, int] = {}
    for item in claims.values():
        label = item.get("claim_label")
        if label:
            labels[label] = labels.get(label, 0) + 1
    return {
        "claims": len(claims),
        "labelled_claims": labelled,
        "claims_with_evidence": with_evidence,
        "label_counts": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Round18 raw JSON smoke run with manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("round18/outputs/batch1_smoke"))
    parser.add_argument("--manifest", type=Path, default=Path("round18/manifests/batch1_smoke/run_manifest.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    data_dir = args.data_dir
    summaries = {}
    input_files = []
    scanned_paths = []

    for filename, split in ALLOWED_DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise SystemExit(f"Missing required raw JSON file: {path}")
        scanned_paths.append(str(path))
        payload = load_json(path)
        if filename == "evidence.json":
            summaries[split] = {"evidence_items": len(payload)}
        else:
            summaries[split] = describe_claims(payload)
        input_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "split": split,
                "labels_used": split in {"train", "dev"},
            }
        )

    guard_hits = find_forbidden_tokens(scanned_paths)
    output_summary = args.output_dir / "raw_json_summary.json"
    write_json(output_summary, summaries)

    manifest = manifest_base(
        run_id="batch1_raw_json_smoke",
        status="strict-candidate",
        mode="STRICT",
        stage="batch1_smoke",
        command="python -m round18.tools.raw_json_smoke",
        working_directory=Path.cwd(),
    )
    manifest["input_files"] = input_files
    manifest["forbidden_input_scan"] = {
        "passed": not bool(guard_hits),
        "notes": "Only raw data file paths were scanned." if not guard_hits else str(guard_hits),
    }
    manifest["output_files"] = [str(output_summary), str(args.manifest)]
    manifest["metrics"] = summaries
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["data_flow_summary"] = "Loaded raw course JSON files and wrote count/hash summary only."
    manifest["split_isolation_summary"] = (
        "No training, validation, model selection, or test content inspection was performed. "
        "Labels were counted for train/dev only as a smoke summary."
    )
    manifest["leakage_risk"] = "low"
    manifest["reproducibility_risk"] = "low"
    write_json(args.manifest, manifest)
    print(f"Wrote {output_summary}")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()

