from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.experiments.o_dense.o_d3b_cross_encoder_s7 import (  # noqa: E402
    run_o_d3b_cross_encoder_s7 as base,
)


def _enforce_diagnostic_inputs(claims: Path, evidence: Path, sparse_pool: Path) -> None:
    if claims != Path("data/dev-claims.json"):
        raise SystemExit("O-D3x diagnostic reranker is dev-only: claims must be data/dev-claims.json.")
    if evidence != Path("data/evidence.json"):
        raise SystemExit("O-D3x diagnostic reranker requires data/evidence.json.")
    if not sparse_pool.exists():
        raise SystemExit(f"Missing sparse pool file: {sparse_pool}")
    if not str(sparse_pool).startswith("round18/outputs/"):
        raise SystemExit("O-D3x diagnostic reranker only accepts Round18 output artifacts.")


def main() -> None:
    base._enforce_contract = _enforce_diagnostic_inputs
    base.main()


if __name__ == "__main__":
    main()
