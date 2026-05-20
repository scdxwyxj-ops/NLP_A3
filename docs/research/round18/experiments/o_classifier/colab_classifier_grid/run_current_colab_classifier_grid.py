#!/usr/bin/env python3
"""Run classifier-only grid search on the current Colab pipeline top64.

This script intentionally does not edit the submission notebook. It executes the
self-contained notebook up to the top64 stage, then evaluates a classifier grid
on dev labels and records both dev metrics and train-only K-fold metrics.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK = ROOT / "colab_notebooks" / "Group_131_COMP90042_Project_2026.ipynb"
OUT_DIR = ROOT / "round18" / "outputs" / "o_classifier" / "colab_classifier_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_DIR / "current_colab_top64_cache.json"

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
SEED = 2026
WORD_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NEG_WORDS = {"no", "not", "never", "without", "none", "neither", "nor"}


def exec_notebook_cells_until_top64() -> dict:
    if CACHE_PATH.exists():
        print("loading cached current Colab top64", CACHE_PATH, flush=True)
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return payload
    os.chdir(ROOT)
    os.environ.setdefault("A3_SAVE_ARTIFACTS", "0")
    os.environ.setdefault("A3_TARGET_FILE", "dev-claims.json")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__colab_grid_exec__"}
    for idx in (3, 4, 7, 9):
        src = "".join(nb["cells"][idx]["source"])
        print(f"executing notebook cell {idx}", flush=True)
        exec(compile(src, f"{NOTEBOOK}:cell{idx}", "exec"), ns)
    payload = {
        "train_claims": ns["train_claims"],
        "target_claims": ns["target_claims"],
        "train_top64": ns["train_top64"],
        "target_top64": ns["target_top64"],
        "evidence": {eid: ns["evidence"][eid] for pool in (ns["train_top64"], ns["target_top64"]) for rows in pool.values() for eid in [r["evidence_id"] for r in rows]},
    }
    CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", CACHE_PATH, flush=True)
    return payload


def word_set(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if len(w) > 2}


def num_set(text: str) -> set[str]:
    return set(NUM_RE.findall(text))


def neg_flag(text: str) -> bool:
    return bool(set(WORD_RE.findall(text.lower())) & NEG_WORDS)


def cue_tokens(claim: dict, ev_text: str) -> str:
    cw = word_set(claim["claim_text"])
    ew = word_set(ev_text)
    overlap = len(cw & ew) / max(len(cw), 1)
    cnums = num_set(claim["claim_text"])
    enums = num_set(ev_text)
    cues = []
    if overlap >= 0.45:
        cues.append("WORD_OVERLAP_HIGH")
    elif overlap >= 0.25:
        cues.append("WORD_OVERLAP_MED")
    elif overlap > 0:
        cues.append("WORD_OVERLAP_LOW")
    else:
        cues.append("WORD_OVERLAP_NONE")
    if cnums and enums:
        cues.append("NUMBER_MATCH" if cnums & enums else "NUMBER_MISMATCH")
    elif cnums:
        cues.append("CLAIM_NUMBER_ONLY")
    cues.append("NEGATION_MATCH" if neg_flag(claim["claim_text"]) == neg_flag(ev_text) else "NEGATION_MISMATCH")
    return " ".join(cues)


def rank_prefix(rank: int, mode: str) -> str:
    if mode in {"rank_aware", "rank_weighted", "enhanced"}:
        if rank <= 5:
            return "TOP_EVIDENCE"
        if rank <= 20:
            return "MID_EVIDENCE"
        return "LOW_EVIDENCE"
    return "EVIDENCE"


def make_text(claim: dict, rows: list[dict], evidence: dict[str, str], config: dict) -> str:
    parts = [f"CLAIM: {claim['claim_text']}"]
    mode = config["context_mode"]
    for i, row in enumerate(rows[: config["context_k"]], 1):
        txt = evidence.get(row["evidence_id"], "")
        budget = config["top_budget"] if i <= 5 else config["tail_budget"]
        if budget:
            txt = " ".join(txt.split()[:budget])
        cues = (" " + cue_tokens(claim, txt)) if mode == "enhanced" else ""
        line = f"{rank_prefix(i, mode)}{cues}: {txt}"
        parts.append(line)
        if mode in {"top5_weighted", "rank_weighted", "enhanced"} and i <= 5:
            parts.append(line)
    return "\n".join(parts)


def make_rows(claims: dict, pool: dict, evidence: dict[str, str], config: dict, require_label: bool) -> list[dict]:
    rows = []
    for cid, claim in claims.items():
        if require_label and "claim_label" not in claim:
            continue
        rows.append(
            {
                "claim_id": cid,
                "text": make_text(claim, pool.get(cid, []), evidence, config),
                "label": claim.get("claim_label"),
            }
        )
    return rows


def fit_predict(train_rows: list[dict], target_rows: list[dict], config: dict) -> list[str]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english" if config["stop_words"] else None,
        ngram_range=config["ngram_range"],
        max_features=config["max_features"],
        min_df=config["min_df"],
        sublinear_tf=config["sublinear_tf"],
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform([r["text"] for r in train_rows])
    y_train = [r["label"] for r in train_rows]
    x_target = vectorizer.transform([r["text"] for r in target_rows])
    if config["model"] == "linear_svm":
        clf = LinearSVC(C=config["C"], class_weight=config["class_weight"], random_state=SEED, max_iter=5000)
    else:
        base_clf = LogisticRegression(
            C=config["C"],
            solver=config.get("solver", "liblinear"),
            max_iter=2000,
            class_weight=config["class_weight"],
            random_state=SEED,
        )
        clf = OneVsRestClassifier(base_clf) if config.get("solver") == "liblinear" else base_clf
    clf.fit(x_train, y_train)
    return clf.predict(x_target).astype(str).tolist()


def score_predictions(y_true: list[str], preds: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0)),
        "prediction_histogram": dict(Counter(preds)),
    }


def kfold_score(train_claims: dict, train_top64: dict, evidence: dict[str, str], config: dict) -> dict:
    ids = [cid for cid, c in train_claims.items() if "claim_label" in c]
    labels = np.asarray([train_claims[cid]["claim_label"] for cid in ids])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs, macros = [], []
    for tr_idx, va_idx in splitter.split(np.zeros(len(ids)), labels):
        tr_claims = {ids[i]: train_claims[ids[i]] for i in tr_idx}
        va_claims = {ids[i]: train_claims[ids[i]] for i in va_idx}
        tr = make_rows(tr_claims, train_top64, evidence, config, True)
        va = make_rows(va_claims, train_top64, evidence, config, True)
        preds = fit_predict(tr, va, config)
        y_true = [r["label"] for r in va]
        accs.append(accuracy_score(y_true, preds))
        macros.append(f1_score(y_true, preds, labels=LABELS, average="macro", zero_division=0))
    return {"cv_accuracy": float(np.mean(accs)), "cv_macro_f1": float(np.mean(macros))}


def config_grid() -> list[dict]:
    configs = []
    for context_mode in ("plain", "rank_aware", "top5_weighted", "rank_weighted", "enhanced"):
        for context_k in (20,):
            for top_budget, tail_budget in ((80, 35), (120, 50)):
                for c in (2.0, 4.0, 8.0):
                    for class_weight in (None, "balanced"):
                        configs.append(
                            {
                                "model": "logreg",
                                "solver": "liblinear",
                                "context_mode": context_mode,
                                "context_k": context_k,
                                "top_budget": top_budget,
                                "tail_budget": tail_budget,
                                "max_features": 60000,
                                "C": c,
                                "class_weight": class_weight,
                                "ngram_range": (1, 2),
                                "stop_words": True,
                                "min_df": 1,
                                "sublinear_tf": False,
                            }
                        )
    return configs


def main() -> None:
    started = time.perf_counter()
    ns = exec_notebook_cells_until_top64()
    train_claims = ns["train_claims"]
    target_claims = ns["target_claims"]
    train_top64 = ns["train_top64"]
    target_top64 = ns["target_top64"]
    evidence = ns["evidence"]
    all_train_rows_cache = {}
    all_target_rows_cache = {}
    results = []
    y_dev = [c["claim_label"] for c in target_claims.values()]
    configs = config_grid()
    print(f"classifier configs {len(configs)}", flush=True)
    for i, config in enumerate(configs, 1):
        key = json.dumps(
            {k: config[k] for k in ("context_mode", "context_k", "top_budget", "tail_budget")},
            sort_keys=True,
        )
        train_rows = all_train_rows_cache.get(key)
        target_rows = all_target_rows_cache.get(key)
        if train_rows is None:
            train_rows = make_rows(train_claims, train_top64, evidence, config, True)
            target_rows = make_rows(target_claims, target_top64, evidence, config, False)
            all_train_rows_cache[key] = train_rows
            all_target_rows_cache[key] = target_rows
        preds = fit_predict(train_rows, target_rows, config)
        metrics = score_predictions(y_dev, preds)
        row = {**config, **metrics}
        results.append(row)
        if i % 10 == 0 or i == len(configs):
            best = max(results, key=lambda r: (r["accuracy"], r["macro_f1"]))
            print(
                "grid",
                i,
                "/",
                len(configs),
                "best",
                {k: best[k] for k in ("accuracy", "macro_f1", "context_mode", "context_k", "top_budget", "tail_budget", "C", "class_weight")},
                flush=True,
            )
    results.sort(key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)
    top = results[:30]
    print("computing 5-fold scores for top dev configs", flush=True)
    for row in top:
        row.update(kfold_score(train_claims, train_top64, evidence, row))
    top.sort(key=lambda r: (r["accuracy"], r["macro_f1"], r["cv_accuracy"]), reverse=True)
    best = top[0]
    train_rows = make_rows(train_claims, train_top64, evidence, best, True)
    target_rows = make_rows(target_claims, target_top64, evidence, best, False)
    preds = fit_predict(train_rows, target_rows, best)
    y_true = [r["label"] for r in target_rows]
    summary = {
        "wall_seconds": time.perf_counter() - started,
        "best": best,
        "top30": top,
        "classification_report": classification_report(y_true, preds, labels=LABELS, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=LABELS).tolist(),
    }
    (OUT_DIR / "current_colab_classifier_grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "current_colab_classifier_grid_predictions.json").write_text(
        json.dumps({r["claim_id"]: p for r, p in zip(target_rows, preds)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["best"], indent=2), flush=True)
    print(summary["classification_report"], flush=True)
    print("wrote", OUT_DIR / "current_colab_classifier_grid_summary.json", flush=True)


if __name__ == "__main__":
    main()
