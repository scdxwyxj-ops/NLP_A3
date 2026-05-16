"""Backward-compatible wrapper for the TF-IDF retrieval experiment.

Prefer running:
    PYTHONPATH=src python src/experiments/retrieval/tfidf_baseline.py
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.retrieval.tfidf_baseline import main


if __name__ == "__main__":
    main()
