#!/usr/bin/env python3
"""Fast sweep over context formatting for the current Colab top64 cache."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier

from run_current_colab_classifier_grid import LABELS, OUT_DIR, SEED, fit_predict, make_rows


CACHE_PATH = OUT_DIR / "current_colab_top64_cache.json"


def main() -> None:
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    train_claims = data["train_claims"]
    target_claims = data["target_claims"]
    train_top64 = data["train_top64"]
    target_top64 = data["target_top64"]
    evidence = data["evidence"]
    y_train = [c["claim_label"] for c in train_claims.values()]
    y_dev = [c["claim_label"] for c in target_claims.values()]
    groups = []
    for context_mode in ("plain", "rank_aware", "top5_weighted", "rank_weighted", "enhanced"):
        for context_k in (5, 8, 12, 20, 32):
            for top_budget, tail_budget in ((120, 50), (160, 50)):
                groups.append(
                    {
                        "model": "logreg",
                        "solver": "liblinear",
                        "context_mode": context_mode,
                        "context_k": context_k,
                        "top_budget": top_budget,
                        "tail_budget": tail_budget,
                        "max_features": 60000,
                        "ngram_range": (1, 2),
                        "stop_words": True,
                        "min_df": 1,
                        "sublinear_tf": False,
                    }
                )
    results = []
    for gi, base in enumerate(groups, 1):
        train_rows = make_rows(train_claims, train_top64, evidence, base, True)
        dev_rows = make_rows(target_claims, target_top64, evidence, base, False)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=60000,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform([r["text"] for r in train_rows])
        x_dev = vectorizer.transform([r["text"] for r in dev_rows])
        for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            for class_weight in (None, "balanced"):
                clf = OneVsRestClassifier(
                    LogisticRegression(
                        C=c,
                        solver="liblinear",
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=SEED,
                    )
                )
                clf.fit(x_train, y_train)
                preds = clf.predict(x_dev).astype(str).tolist()
                results.append(
                    {
                        **base,
                        "C": c,
                        "class_weight": class_weight,
                        "accuracy": float(accuracy_score(y_dev, preds)),
                        "macro_f1": float(f1_score(y_dev, preds, labels=LABELS, average="macro", zero_division=0)),
                        "prediction_histogram": dict(Counter(preds)),
                    }
                )
        best = max(results, key=lambda r: (r["accuracy"], r["macro_f1"]))
        print(
            "group",
            gi,
            "/",
            len(groups),
            "best",
            {k: best[k] for k in ("accuracy", "macro_f1", "context_mode", "context_k", "top_budget", "tail_budget", "C", "class_weight")},
            flush=True,
        )
    results.sort(key=lambda r: (r["accuracy",] if False else r["accuracy"], r["macro_f1"]), reverse=True)
    best = results[0]
    preds = fit_predict(
        make_rows(train_claims, train_top64, evidence, best, True),
        make_rows(target_claims, target_top64, evidence, best, False),
        best,
    )
    summary = {
        "best": best,
        "top50": results[:50],
        "classification_report": classification_report(y_dev, preds, labels=LABELS, zero_division=0),
        "confusion_matrix": confusion_matrix(y_dev, preds, labels=LABELS).tolist(),
    }
    out = OUT_DIR / "fast_context_format_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2), flush=True)
    print(summary["classification_report"], flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
