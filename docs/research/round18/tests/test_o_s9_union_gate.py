from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import importlib.util


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = Path(__file__).resolve().parents[1] / "experiments/o_sparse/o_s9_union_gate/run_o_s9_union_gate.py"
    spec = importlib.util.spec_from_file_location("round18_o_s9_union_gate", module_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    import sys as _sys
    _sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


mod = _load_module()


def test_stable_deduplicate_evidence_rows_is_first_seen_and_ordered() -> None:
    rows = [
        {"evidence_id": "e2", "rank": 2},
        {"evidence_id": "e1", "rank": 1},
        {"evidence_id": "e2", "rank": 3},
        {"evidence_id": "e3", "rank": 4},
    ]
    deduped = mod.stable_deduplicate_evidence_rows(rows)
    assert [row["evidence_id"] for row in deduped] == ["e2", "e1", "e3"]


def test_source_provenance_preserved_in_output_rows() -> None:
    claim_rows: dict[str, list[dict[str, Any]]] = {
        "bm25": [
            {"evidence_id": "e1", "rank": 2, "score": 5.0},
            {"evidence_id": "e2", "rank": 1, "score": 4.0},
        ],
        "char": [
            {"evidence_id": "e1", "rank": 1, "score": 3.0},
            {"evidence_id": "e3", "rank": 2, "score": 6.0},
        ],
        "structured": [],
        "prf": [],
    }
    rows = mod.build_policy_candidates(
        claim_source_rows=claim_rows,
        policy="rrf",
        candidate_k=5,
        source_weights={"bm25": 1.0, "char": 1.0, "structured": 1.0, "prf": 1.0},
        rrf_k=60,
    )
    hit = {row["evidence_id"]: row for row in rows}["e1"]
    assert hit["source_count"] == 2
    sources = {entry["source"] for entry in hit["source_breakdown"]}
    assert sources == {"bm25", "char"}
    assert {entry["source_rank"] for entry in hit["source_breakdown"]} == {1, 2}
    assert {entry["source_score"] for entry in hit["source_breakdown"]} == {3.0, 5.0}


def test_max_top_k_is_enforced_for_policy_and_source_caps() -> None:
    policy_rows = {
        "bm25": [
            {"evidence_id": f"e{i}", "rank": i, "score": 1.0 / (i + 1)} for i in range(1, 6)
        ],
        "char": [
            {"evidence_id": f"f{i}", "rank": i, "score": 0.1 * i} for i in range(1, 6)
        ],
        "structured": [],
        "prf": [],
    }
    weighted = {"bm25": 1.0, "char": 1.0, "structured": 1.0, "prf": 1.0}

    ranked = mod.build_policy_candidates(
        claim_source_rows=policy_rows,
        policy="priority",
        candidate_k=3,
        source_weights=weighted,
        rrf_k=60,
    )
    assert len(ranked) == 3

    limited_sources = mod.limit_evidence_count(policy_rows["bm25"], 2)
    assert len(limited_sources) == 2
    assert [row["evidence_id"] for row in limited_sources] == ["e1", "e2"]
