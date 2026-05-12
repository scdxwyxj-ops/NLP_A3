from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "o_classifier"
    / "o_c3_strict_context"
    / "run_o_c3_strict_context.py"
)


def _run_script(
    tmp_path: Path,
    args: list[str],
    expect_success: bool = True,
    *,
    train_pool: Path | None = None,
    dev_pool: Path | None = None,
) -> subprocess.CompletedProcess:
    has_train_pool = "--train-pool" in args
    has_dev_pool = "--dev-pool" in args
    cmd = [
        sys.executable,
        str(SCRIPT),
        *args,
    ]
    if not has_train_pool:
        pool = train_pool or (tmp_path / "train_full_o_a2_calibrated_rank_top500_candidates.json")
        cmd.extend(["--train-pool", str(pool)])
    if not has_dev_pool:
        pool = dev_pool or (tmp_path / "dev_full_o_a2_calibrated_rank_top500_candidates.json")
        cmd.extend(["--dev-pool", str(pool)])
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    if expect_success:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    else:
        assert proc.returncode != 0
    return proc


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_inputs(tmp_path: Path, family_train: str = "o_a2_calibrated_rank", family_dev: str = "o_a2_calibrated_rank") -> tuple[Path, Path, Path]:
    train_claims = {
        "t1": {
            "claim_text": "CO2 is not a pollutant.",
            "claim_label": "DISPUTED",
            "evidences": ["e1"],
        },
        "t2": {
            "claim_text": "Trees increase oxygen.",
            "claim_label": "SUPPORTS",
            "evidences": ["e2"],
        },
    }
    dev_claims = {
        "d1": {
            "claim_text": "CO2 is good for plants.",
            "claim_label": "SUPPORTS",
            "evidences": ["e3"],
        },
        "d2": {
            "claim_text": "This claim lacks support.",
            "claim_label": "NOT_ENOUGH_INFO",
            "evidences": ["e4"],
        },
    }
    evidence = {
        "e1": "Some evidence about CO2 pollution.",
        "e2": "Evidence about trees and oxygen.",
        "e3": "Research on CO2 use by plants.",
        "e4": "No relevant evidence found.",
    }
    pool_train = {
        "t1": [{"evidence_id": "e1", "rank": 1, "score": 0.9}],
        "t2": [{"evidence_id": "e2", "rank": 1, "score": 0.8}],
    }
    pool_dev = {
        "d1": [{"evidence_id": "e1", "rank": 1, "score": 0.95}],
        "d2": [{"evidence_id": "e3", "rank": 1, "score": 0.75}],
    }
    train_claim_path = tmp_path / "train-claims.json"
    dev_claim_path = tmp_path / "dev-claims.json"
    evidence_path = tmp_path / "evidence.json"
    _write(train_claim_path, train_claims)
    _write(dev_claim_path, dev_claims)
    _write(evidence_path, evidence)
    # keep family token in filename to pass strict family check
    train_pool_path = tmp_path / f"train_full_{family_train}_top500_candidates.json"
    dev_pool_path = tmp_path / f"dev_full_{family_dev}_top500_candidates.json"
    _write(train_pool_path, pool_train)
    _write(dev_pool_path, pool_dev)
    return train_claim_path, dev_claim_path, evidence_path


def test_family_mismatch_blocks_without_diagnostic(tmp_path: Path) -> None:
    train_claim_path, dev_claim_path, evidence_path = _base_inputs(
        tmp_path, family_train="o_family_a", family_dev="o_family_b"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-claims",
            str(train_claim_path),
            "--dev-claims",
            str(dev_claim_path),
            "--evidence",
            str(evidence_path),
            "--train-pool",
            str(tmp_path / "train_full_o_family_a_top500_candidates.json"),
            "--dev-pool",
            str(tmp_path / "dev_full_o_family_b_top500_candidates.json"),
            "--output-dir",
            str(out_dir),
            "--run-id",
            "tc3",
            "--run-logreg",
            "--clf-max-features-grid",
            "200",
            "--clf-ngram-grid",
            "1-1",
            "--logreg-c-grid",
            "1.0",
            "--train-context-k",
            "1",
            "--dev-context-k",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "context source mismatch" in proc.stderr + proc.stdout


def test_family_mismatch_diagnostic_is_allowed(tmp_path: Path) -> None:
    train_claim_path, dev_claim_path, evidence_path = _base_inputs(
        tmp_path, family_train="o_family_a", family_dev="o_family_b"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "run_manifest.json"
    record_path = out_dir / "run_record.json"

    _run_script(
        tmp_path,
        [
            "--train-claims",
            str(train_claim_path),
            "--dev-claims",
            str(dev_claim_path),
            "--evidence",
            str(evidence_path),
            "--output-dir",
            str(out_dir),
            "--run-id",
            "tc3",
            "--manifest",
            str(manifest_path),
            "--record",
            str(record_path),
            "--run-logreg",
            "--enable-diagnostic",
            "--clf-max-features-grid",
            "200",
            "--clf-ngram-grid",
            "1-1",
            "--logreg-c-grid",
            "1.0",
            "--train-context-k",
            "1",
            "--dev-context-k",
            "1",
        ],
        train_pool=tmp_path / "train_full_o_family_a_top500_candidates.json",
        dev_pool=tmp_path / "dev_full_o_family_b_top500_candidates.json",
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "diagnostic-only"
    assert record["train_context_source"] == "o_family_a"
    assert record["dev_context_source"] == "o_family_b"


def test_script_writes_required_artifacts(tmp_path: Path) -> None:
    train_claim_path, dev_claim_path, evidence_path = _base_inputs(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    manifest_path = out_dir / "run_manifest.json"
    record_path = out_dir / "run_record.json"

    _run_script(
        tmp_path,
        [
            "--train-claims",
            str(train_claim_path),
            "--dev-claims",
            str(dev_claim_path),
            "--evidence",
            str(evidence_path),
            "--output-dir",
            str(out_dir),
            "--run-id",
            "tc3",
            "--manifest",
            str(manifest_path),
            "--record",
            str(record_path),
            "--run-logreg",
            "--run-svm",
            "--clf-max-features-grid",
            "200",
            "--clf-ngram-grid",
            "1-1",
            "--logreg-c-grid",
            "1.0",
            "--svm-c-grid",
            "1.0",
            "--train-context-k",
            "1",
            "--dev-context-k",
            "1",
        ],
        train_pool=tmp_path / "train_full_o_a2_calibrated_rank_top500_candidates.json",
        dev_pool=tmp_path / "dev_full_o_a2_calibrated_rank_top500_candidates.json",
    )

    assert manifest_path.exists()
    assert record_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "strict-candidate"
    assert manifest["metrics"]["strict_family_match"] is True

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "strict-candidate"
    assert set(record["model_results"].keys()) == {"tfidf_logreg", "linear_svm"}
    for model, bundle in record["model_results"].items():
        for suffix in ("predictions", "metrics", "report", "confusion_matrix", "grid_search"):
            assert Path(bundle[suffix]).exists(), f"{model} missing {suffix}"
        metric = json.loads(Path(bundle["metrics"]).read_text(encoding="utf-8"))
        assert "per_class_recall" in metric
        assert "prediction_histogram" in metric
        assert "collapse_gate" in metric
        assert metric["collapse_gate"]["status"] in {"passed", "failed"}


def test_short_pool_context_rejected_without_fallback(tmp_path: Path) -> None:
    # one candidate while context_k=2 should be rejected because no hidden fallback.
    train_claim_path, dev_claim_path, evidence_path = _base_inputs(tmp_path)
    _write(
        tmp_path / "train_full_o_a2_calibrated_rank_top500_candidates.json",
        {
            "t1": [{"evidence_id": "e1", "rank": 1, "score": 0.9}],
            "t2": [{"evidence_id": "e2", "rank": 1, "score": 0.8}],
        },
    )
    _write(
        tmp_path / "dev_full_o_a2_calibrated_rank_top500_candidates.json",
        {
            "d1": [{"evidence_id": "e1", "rank": 1, "score": 0.9}],
            "d2": [{"evidence_id": "e3", "rank": 1, "score": 0.8}],
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-claims",
            str(train_claim_path),
            "--dev-claims",
            str(dev_claim_path),
            "--evidence",
            str(evidence_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "tc3",
            "--train-pool",
            str(tmp_path / "train_full_o_a2_calibrated_rank_top500_candidates.json"),
            "--dev-pool",
            str(tmp_path / "dev_full_o_a2_calibrated_rank_top500_candidates.json"),
            "--run-logreg",
            "--clf-max-features-grid",
            "200",
            "--clf-ngram-grid",
            "1-1",
            "--logreg-c-grid",
            "1.0",
            "--train-context-k",
            "2",
            "--dev-context-k",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    text = proc.stdout + proc.stderr
    assert "context too short" in text
