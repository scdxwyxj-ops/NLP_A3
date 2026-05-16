from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from round18.experiments.o_aggregate.o_a2c_plain_leaf_rank_s8 import (  # noqa: E402
    run_o_a2c_plain_leaf_rank_s8 as base,
)


def _enforce_diagnostic_inputs(
    claims: Path,
    evidence: Path,
    sparse_pool: Path,
    ce_pool: Path,
) -> None:
    if claims != Path("data/dev-claims.json"):
        raise SystemExit("O-A2x diagnostic aggregate is dev-only: claims must be data/dev-claims.json.")
    if evidence != Path("data/evidence.json"):
        raise SystemExit("O-A2x diagnostic aggregate requires data/evidence.json.")
    for label, path in {"sparse_pool": sparse_pool, "ce_pool": ce_pool}.items():
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")
        if not str(path).startswith("round18/outputs/"):
            raise SystemExit(f"O-A2x diagnostic aggregate only accepts Round18 output artifacts for {label}.")


def main() -> None:
    base._enforce_strict_inputs = _enforce_diagnostic_inputs
    base.main()


if __name__ == "__main__":
    main()
