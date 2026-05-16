from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_json


REQUIRED_TOP_LEVEL = {
    "round",
    "run_id",
    "status",
    "mode",
    "stage",
    "agent_id",
    "agent_role",
    "timestamp_utc",
    "git_commit",
    "git_status_short",
    "command",
    "working_directory",
    "input_files",
    "forbidden_input_scan",
    "output_files",
    "metrics",
    "runtime",
    "data_flow_summary",
    "split_isolation_summary",
    "leakage_risk",
    "reproducibility_risk",
}

VALID_STATUS = {"strict-candidate", "diagnostic-only", "rejected"}
VALID_MODE = {"STRICT", "DIAGNOSTIC"}
VALID_RISK = {"low", "medium", "high"}


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    payload = load_json(path)
    missing = sorted(REQUIRED_TOP_LEVEL - set(payload))
    if missing:
        errors.append(f"missing top-level fields: {', '.join(missing)}")

    if payload.get("round") != "round18":
        errors.append("round must be round18")
    if payload.get("status") not in VALID_STATUS:
        errors.append("status must be strict-candidate, diagnostic-only, or rejected")
    if payload.get("mode") not in VALID_MODE:
        errors.append("mode must be STRICT or DIAGNOSTIC")
    if payload.get("leakage_risk") not in VALID_RISK:
        errors.append("leakage_risk must be low, medium, or high")
    if payload.get("reproducibility_risk") not in VALID_RISK:
        errors.append("reproducibility_risk must be low, medium, or high")

    scan = payload.get("forbidden_input_scan")
    if not isinstance(scan, dict) or "passed" not in scan:
        errors.append("forbidden_input_scan must contain passed")
    elif payload.get("mode") == "STRICT" and scan.get("passed") is not True:
        errors.append("STRICT manifest must have forbidden_input_scan.passed=true")

    input_files = payload.get("input_files")
    if not isinstance(input_files, list):
        errors.append("input_files must be a list")
    else:
        for idx, item in enumerate(input_files):
            if not isinstance(item, dict):
                errors.append(f"input_files[{idx}] must be an object")
                continue
            for field in ("path", "sha256", "split", "labels_used"):
                if field not in item:
                    errors.append(f"input_files[{idx}] missing {field}")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
    elif "wall_seconds" not in runtime:
        errors.append("runtime missing wall_seconds")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Round18 run manifest shape.")
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for path in args.manifests:
        errors = validate_manifest(path)
        if errors:
            failed = True
            print(f"{path}: INVALID")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{path}: OK")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

