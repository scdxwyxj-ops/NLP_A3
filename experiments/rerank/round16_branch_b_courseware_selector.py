import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from a3_factcheck.data import majority_label  # noqa: E402
from a3_factcheck.metrics import (  # noqa: E402
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
NEGATIONS = {"no", "not", "never", "none", "without", "false", "fake", "myth"}
UNITS = {
    "c",
    "celsius",
    "co2",
    "carbon",
    "cm",
    "degrees",
    "dioxide",
    "dollars",
    "f",
    "fahrenheit",
    "feet",
    "gt",
    "gw",
    "hectares",
    "inches",
    "km",
    "m",
    "metres",
    "meters",
    "miles",
    "million",
    "ppm",
    "percent",
    "percentage",
    "tonnes",
    "tons",
    "w",
    "watts",
    "years",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['.-][a-z0-9]+)?", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
NUMBER_RE = re.compile(r"(?<![a-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", re.I)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


def content_tokens(text):
    return [token for token in tokens(text) if token not in STOPWORDS and len(token) > 1]


def numbers(text):
    values = []
    for match in NUMBER_RE.findall(text):
        raw = match.replace(",", "")
        is_percent = raw.endswith("%")
        raw = raw.rstrip("%")
        try:
            value = float(raw)
        except ValueError:
            continue
        values.append(("pct" if is_percent else "num", round(value, 4)))
    return set(values)


def years(text):
    return set(YEAR_RE.findall(text))


def unit_tokens(text):
    return set(content_tokens(text)) & UNITS


def split_sentences(text):
    parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return parts if parts else [text.strip()]


def sentence_windows(text):
    sentences = split_sentences(text)
    windows = list(sentences)
    windows.extend(
        f"{sentences[i]} {sentences[i + 1]}" for i in range(len(sentences) - 1)
    )
    if text.strip() not in windows:
        windows.append(text.strip())
    return windows


def overlap_stats(claim_set, evidence_set):
    if not claim_set or not evidence_set:
        return 0.0, 0.0, 0.0, 0
    inter = claim_set & evidence_set
    precision = len(inter) / len(evidence_set)
    recall = len(inter) / len(claim_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1, len(inter)


def max_window_stats(claim_set, windows):
    best = (0.0, 0.0, 0.0, 0, 0)
    for window in windows:
        window_set = set(content_tokens(window))
        precision, recall, f1, count = overlap_stats(claim_set, window_set)
        if f1 > best[2]:
            best = (precision, recall, f1, count, len(window_set))
    return best


def number_features(claim_text, evidence_text):
    claim_nums = numbers(claim_text)
    ev_nums = numbers(evidence_text)
    claim_years = years(claim_text)
    ev_years = years(evidence_text)
    claim_units = unit_tokens(claim_text)
    ev_units = unit_tokens(evidence_text)

    num_overlap = len(claim_nums & ev_nums)
    year_overlap = len(claim_years & ev_years)
    unit_overlap = len(claim_units & ev_units)
    num_mismatch = int(bool(claim_nums) and bool(ev_nums) and num_overlap == 0)
    year_mismatch = int(bool(claim_years) and bool(ev_years) and year_overlap == 0)
    unit_mismatch = int(bool(claim_units) and bool(ev_units) and unit_overlap == 0)
    claim_num_missing = int(bool(claim_nums) and not ev_nums)
    claim_year_missing = int(bool(claim_years) and not ev_years)
    return [
        len(claim_nums),
        len(ev_nums),
        num_overlap,
        num_overlap / len(claim_nums) if claim_nums else 0.0,
        num_mismatch,
        claim_num_missing,
        len(claim_years),
        len(ev_years),
        year_overlap,
        year_overlap / len(claim_years) if claim_years else 0.0,
        year_mismatch,
        claim_year_missing,
        len(claim_units),
        len(ev_units),
        unit_overlap,
        unit_overlap / len(claim_units) if claim_units else 0.0,
        unit_mismatch,
    ]


def lexical_features(claim_text, evidence_text, rank, sparse_score):
    claim_tokens = content_tokens(claim_text)
    evidence_tokens = content_tokens(evidence_text)
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    precision, recall, f1, count = overlap_stats(claim_set, evidence_set)
    win_precision, win_recall, win_f1, win_count, win_len = max_window_stats(
        claim_set, sentence_windows(evidence_text)
    )
    neg_claim = set(tokens(claim_text)) & NEGATIONS
    neg_ev = set(tokens(evidence_text)) & NEGATIONS
    num_feats = number_features(claim_text, evidence_text)
    return [
        float(sparse_score),
        1.0 / max(rank, 1),
        math.log1p(rank),
        rank / 4500.0,
        len(claim_tokens),
        len(evidence_tokens),
        precision,
        recall,
        f1,
        count,
        win_precision,
        win_recall,
        win_f1,
        win_count,
        win_len,
        int(bool(neg_claim)),
        int(bool(neg_ev)),
        len(neg_claim & neg_ev),
        int(bool(neg_claim) and not bool(neg_ev)),
        *num_feats,
    ]


def feature_names():
    base = [
        "sparse_score",
        "reciprocal_rank",
        "log_rank",
        "rank_over_4500",
        "claim_token_count",
        "evidence_token_count",
        "whole_precision",
        "whole_recall",
        "whole_f1",
        "whole_overlap_count",
        "window_precision",
        "window_recall",
        "window_f1",
        "window_overlap_count",
        "best_window_token_count",
        "claim_has_negation",
        "evidence_has_negation",
        "negation_overlap",
        "claim_negation_missing",
    ]
    numeric = [
        "claim_num_count",
        "evidence_num_count",
        "num_overlap",
        "num_overlap_recall",
        "num_mismatch",
        "claim_num_missing",
        "claim_year_count",
        "evidence_year_count",
        "year_overlap",
        "year_overlap_recall",
        "year_mismatch",
        "claim_year_missing",
        "claim_unit_count",
        "evidence_unit_count",
        "unit_overlap",
        "unit_overlap_recall",
        "unit_mismatch",
    ]
    return base + numeric


def build_rows(claims, pool, evidence, max_candidates, include_gold, collect_text=True):
    rows = []
    y = []
    pair_texts = []
    groups = []
    gold_rank_hits = Counter()
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        candidates = list(pool.get(claim_id, []))[:max_candidates]
        if include_gold:
            seen = {item["evidence_id"] for item in candidates}
            for evidence_id in gold - seen:
                if evidence_id in evidence:
                    candidates.append(
                        {
                            "claim_id": claim_id,
                            "evidence_id": evidence_id,
                            "rank": max_candidates + 1,
                            "score": 0.0,
                        }
                    )
        for item in candidates:
            evidence_id = item["evidence_id"]
            evidence_text = evidence.get(evidence_id, "")
            rows.append(
                lexical_features(
                    claim["claim_text"],
                    evidence_text,
                    int(item.get("rank", max_candidates + 1)),
                    float(item.get("score", 0.0)),
                )
            )
            pair_texts.append(f"{claim['claim_text']} [SEP] {evidence_text}" if collect_text else "")
            y.append(1 if evidence_id in gold else 0)
            groups.append((claim_id, evidence_id, evidence_text, item))
            if evidence_id in gold:
                gold_rank_hits[min(int(item.get("rank", max_candidates + 1)), max_candidates + 1)] += 1
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int8), pair_texts, groups, gold_rank_hits


def take_second_stage(X, pair_texts, groups, scores, per_claim):
    selected_indices = []
    by_claim = {}
    for index, (claim_id, _, _, _) in enumerate(groups):
        by_claim.setdefault(claim_id, []).append(index)
    for indices in by_claim.values():
        indices.sort(key=lambda index: -scores[index])
        selected_indices.extend(indices[:per_claim])
    return (
        X[selected_indices],
        [pair_texts[index] for index in selected_indices],
        [groups[index] for index in selected_indices],
        scores[selected_indices],
    )


def pair_texts_for_groups(claims, groups):
    texts = []
    for claim_id, _, evidence_text, _ in groups:
        texts.append(f"{claims[claim_id]['claim_text']} [SEP] {evidence_text}")
    return texts


def build_vectorizer(max_features):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def train_models(X_train, y_train, pair_texts, max_text_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    feature_model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=16,
        class_weight="balanced",
    )
    feature_model.fit(X_scaled, y_train)

    text_vectorizer = build_vectorizer(max_text_features)
    X_text = text_vectorizer.fit_transform(pair_texts)
    text_model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=16,
    )
    text_model.fit(X_text, y_train)
    return scaler, feature_model, text_vectorizer, text_model


def score_rows(X, pair_texts, scaler, feature_model, text_vectorizer, text_model, alpha):
    feature_scores = feature_model.predict_proba(scaler.transform(X))[:, 1]
    text_scores = text_model.predict_proba(text_vectorizer.transform(pair_texts))[:, 1]
    return alpha * feature_scores + (1.0 - alpha) * text_scores, feature_scores, text_scores


def rows_to_ranked(groups, scores, feature_scores, text_scores):
    ranked = {}
    for (claim_id, evidence_id, evidence_text, item), score, feat_score, text_score in zip(
        groups, scores, feature_scores, text_scores
    ):
        ranked.setdefault(claim_id, []).append(
            {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "score": float(score),
                "feature_score": float(feat_score),
                "text_score": float(text_score),
                "sparse_rank": int(item.get("rank", 0)),
                "sparse_score": float(item.get("score", 0.0)),
                "text": evidence_text,
            }
        )
    for items in ranked.values():
        items.sort(
            key=lambda item: (
                -item["score"],
                item["sparse_rank"] if item["sparse_rank"] else 10**9,
                item["evidence_id"],
            )
        )
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
    return ranked


def jaccard(a, b):
    a_set = set(content_tokens(a))
    b_set = set(content_tokens(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def select_mmr(items, k, lambda_score):
    selected = []
    remaining = list(items)
    while remaining and len(selected) < k:
        best_index = 0
        best_value = -1e9
        for index, item in enumerate(remaining):
            redundancy = max(
                (jaccard(item.get("text", ""), chosen.get("text", "")) for chosen in selected),
                default=0.0,
            )
            value = lambda_score * item["score"] - (1.0 - lambda_score) * redundancy
            value += 0.000001 / max(item.get("sparse_rank", 1), 1)
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def build_predictions(claims, ranked, default_label, top_k, mmr_lambda=None):
    predictions = {}
    for claim_id, claim in claims.items():
        items = ranked.get(claim_id, [])
        if mmr_lambda is not None and top_k <= 10:
            chosen = select_mmr(items[:100], top_k, mmr_lambda)
        else:
            chosen = items[:top_k]
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": default_label,
            "evidences": [item["evidence_id"] for item in chosen],
        }
    return predictions


def metric_row(name, top_k, claims, predictions, path):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
        "path": str(path),
        "top_macro_recall": macro_recall(claims, predictions),
        "top_evidence_f": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
    }
    for label in LABELS:
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(
            claims, predictions, label
        )["macro_recall"]
    return row


def write_summary(rows, csv_path, json_path):
    with Path(csv_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    dump_json(rows, json_path)


def strip_ranked_for_json(ranked):
    stripped = {}
    for claim_id, items in ranked.items():
        stripped[claim_id] = [
            {key: value for key, value in item.items() if key != "text"} for item in items
        ]
    return stripped


def main():
    parser = argparse.ArgumentParser(
        description="Round16 Branch B courseware-hinted evidence selector."
    )
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument(
        "--train-pool",
        default="outputs/round13/core_fusion_train/candidates/rrf_bm25_char_base_baseq_k500_top2000.json",
    )
    parser.add_argument(
        "--dev-pool",
        default="outputs/round14/s20_sparse_pool_mix_base_heavy_rrfk500_fine/weighted_rrf_top5000.json",
    )
    parser.add_argument(
        "--baseline-top3",
        default="outputs/round15/recommended/top3_submission.json",
    )
    parser.add_argument("--output-dir", default="outputs/round16/branch_b_courseware")
    parser.add_argument("--train-candidates", type=int, default=350)
    parser.add_argument("--dev-candidates", type=int, default=4500)
    parser.add_argument("--second-stage-candidates", type=int, default=700)
    parser.add_argument("--text-features", type=int, default=30000)
    parser.add_argument("--score-alpha", type=float, default=0.55)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_json(args.train_pool)
    dev_pool = load_json(args.dev_pool)
    baseline = load_json(args.baseline_top3)
    default_label = majority_label(train_claims)

    X_train, y_train, train_pair_texts, train_groups, gold_rank_hits = build_rows(
        train_claims,
        train_pool,
        evidence,
        args.train_candidates,
        include_gold=False,
    )
    scaler, feature_model, text_vectorizer, text_model = train_models(
        X_train, y_train, train_pair_texts, args.text_features
    )
    train_scores, train_feature_scores, train_text_scores = score_rows(
        X_train,
        train_pair_texts,
        scaler,
        feature_model,
        text_vectorizer,
        text_model,
        args.score_alpha,
    )
    train_ranked = rows_to_ranked(
        train_groups, train_scores, train_feature_scores, train_text_scores
    )

    mmr_candidates = [None, 0.65, 0.75, 0.85, 0.95]
    train_diagnostic_rows = []
    best_mmr = None
    best_train_f = -1.0
    for mmr_lambda in mmr_candidates:
        train_pred = build_predictions(
            train_claims, train_ranked, default_label, 3, mmr_lambda=mmr_lambda
        )
        name = f"train_top3_mmr_{mmr_lambda}" if mmr_lambda is not None else "train_top3_no_mmr"
        row = metric_row(name, 3, train_claims, train_pred, "train_diagnostic")
        train_diagnostic_rows.append(row)
        if row["top_evidence_f"] > best_train_f:
            best_train_f = row["top_evidence_f"]
            best_mmr = mmr_lambda

    X_dev, _, dev_pair_texts, dev_groups, _ = build_rows(
        dev_claims,
        dev_pool,
        evidence,
        args.dev_candidates,
        include_gold=False,
        collect_text=False,
    )
    dev_feature_scores = feature_model.predict_proba(scaler.transform(X_dev))[:, 1]
    X_dev_stage2, _, dev_groups_stage2, dev_feature_scores_stage2 = take_second_stage(
        X_dev,
        dev_pair_texts,
        dev_groups,
        dev_feature_scores,
        args.second_stage_candidates,
    )
    dev_pair_texts_stage2 = pair_texts_for_groups(dev_claims, dev_groups_stage2)
    dev_text_scores = text_model.predict_proba(
        text_vectorizer.transform(dev_pair_texts_stage2)
    )[:, 1]
    dev_scores = (
        args.score_alpha * dev_feature_scores_stage2
        + (1.0 - args.score_alpha) * dev_text_scores
    )
    dev_ranked = rows_to_ranked(
        dev_groups_stage2, dev_scores, dev_feature_scores_stage2, dev_text_scores
    )

    ranked_path = output_dir / "courseware_ranked_top4500.json"
    dump_json(strip_ranked_for_json(dev_ranked), ranked_path)

    top100_path = output_dir / "top100_context_predictions.json"
    top100_predictions = build_predictions(dev_claims, dev_ranked, default_label, 100)
    dump_json(top100_predictions, top100_path)

    top3_path = output_dir / "top3_predictions.json"
    top3_predictions = build_predictions(
        dev_claims, dev_ranked, default_label, 3, mmr_lambda=best_mmr
    )
    dump_json(top3_predictions, top3_path)

    baseline_row = metric_row(
        "round15_recommended_baseline_top3",
        3,
        dev_claims,
        baseline,
        args.baseline_top3,
    )
    top100_row = metric_row(
        "courseware_ranker_top100_context",
        100,
        dev_claims,
        top100_predictions,
        top100_path,
    )
    top3_row = metric_row(
        f"courseware_ranker_top3_mmr_{best_mmr}",
        3,
        dev_claims,
        top3_predictions,
        top3_path,
    )
    summary_rows = [baseline_row, top100_row, top3_row]
    write_summary(
        summary_rows,
        output_dir / "summary.csv",
        output_dir / "summary.json",
    )
    write_summary(
        train_diagnostic_rows,
        output_dir / "train_mmr_selection_diagnostic.csv",
        output_dir / "train_mmr_selection_diagnostic.json",
    )

    diagnostic = {
        "method": "Train-only lightweight claim-evidence selector with sentence/window overlap, pair TF-IDF, numeric/date/unit mismatch features, and train-selected MMR.",
        "score_alpha_feature_model": args.score_alpha,
        "best_mmr_lambda_selected_on_train": best_mmr,
        "train_candidates_per_claim": args.train_candidates,
        "dev_candidates_per_claim": args.dev_candidates,
        "dev_second_stage_candidates_per_claim": args.second_stage_candidates,
        "train_positive_pairs": int(y_train.sum()),
        "train_pairs": int(len(y_train)),
        "train_positive_rate": float(y_train.mean()),
        "feature_names": feature_names(),
        "feature_model_train_auc": float(roc_auc_score(y_train, train_feature_scores)),
        "text_model_train_auc": float(roc_auc_score(y_train, train_text_scores)),
        "blend_train_auc": float(roc_auc_score(y_train, train_scores)),
        "blend_train_average_precision": float(
            average_precision_score(y_train, train_scores)
        ),
        "gold_rank_hits_within_train_candidate_depth": dict(gold_rank_hits),
        "note": "MMR lambda was selected on train labels only. Dev labels are used only for the reported evaluation metrics.",
    }
    dump_json(diagnostic, output_dir / "method_diagnostic.json")

    print(f"Wrote ranked candidates: {ranked_path}")
    print(f"Wrote top100 context predictions: {top100_path}")
    print(f"Wrote top3 predictions: {top3_path}")
    print(f"Wrote summary: {output_dir / 'summary.csv'}")
    print(
        "Top3 dev F="
        f"{top3_row['top_evidence_f']:.6f} macro_recall={top3_row['top_macro_recall']:.6f}"
    )
    print(
        "Top100 dev macro_recall="
        f"{top100_row['top_macro_recall']:.6f}; selected train MMR={best_mmr}"
    )


if __name__ == "__main__":
    main()
