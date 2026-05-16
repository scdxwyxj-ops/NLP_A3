from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_TOKENS = (
    "outputs/round",
    "teacher",
    "pseudo",
    "checkpoint",
    "best_model",
    "label_source",
    "drive.mount",
    "old_prediction",
    "cached_prediction",
    "round15_selector_rank",
    "claim_key_selector_rank",
)

ALLOWED_DATA_FILES = {
    "train-claims.json": "train",
    "dev-claims.json": "dev",
    "test-claims-unlabelled.json": "test",
    "evidence.json": "evidence",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def find_forbidden_tokens(values: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for value in values:
        normalized = value.replace("\\", "/").lower()
        for token in FORBIDDEN_TOKENS:
            if token.lower() in normalized:
                hits.setdefault(token, []).append(value)
    return hits


def manifest_base(
    *,
    run_id: str,
    status: str,
    mode: str,
    stage: str,
    command: str,
    working_directory: Path,
    config_path: str = "",
    config_hash: str = "",
    random_seed: int | None = None,
    cv_seed: int | None = None,
) -> dict[str, Any]:
    return {
        "round": "round18",
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "stage": stage,
        "agent_id": "master",
        "agent_role": "orchestrator",
        "timestamp_utc": utc_now(),
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "command": command,
        "working_directory": str(working_directory),
        "config_path": config_path,
        "config_hash": config_hash,
        "random_seed": random_seed,
        "cv_seed": cv_seed,
        "fold_policy": "",
        "model_names": [],
        "input_files": [],
        "forbidden_input_scan": {"passed": False, "notes": ""},
        "output_files": [],
        "metrics": {},
        "runtime": {
            "wall_seconds": None,
            "peak_memory_mb": None,
            "device": "",
        },
        "data_flow_summary": "",
        "split_isolation_summary": "",
        "leakage_risk": "medium",
        "reproducibility_risk": "medium",
        "notes": "",
    }

