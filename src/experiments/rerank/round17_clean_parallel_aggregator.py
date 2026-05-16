import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from a3_factcheck.data import load_json, majority_label  # noqa: E402
from a3_factcheck.metrics import (  # noqa: E402
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions  # noqa: E402


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

DEV_PARALLEL_SOURCES = [
    (
        "candidate_gate",
        "outputs/round14/s20_sparse_pool_mix_base_heavy_rrfk500_fine/weighted_rrf_top5000.json",
        4500,
    ),
    ("bge_small", "outputs/round16/baseline/s22_small_qprefix_top100.json", 100),
    ("minilm_cross_encoder", "outputs/round15/dev-round12-top500-minilm-ranked-top100.json", 100),
    (
        "binary_minilm_selector",
        "outputs/round16/top3_binary_selector_e3_k20_n80/dev_binary_selector_ranked.json",
        100,
    ),
    (
        "explicit_fact_matcher",
        "outputs/round16/branch_a_requirement/ranker_claim_key_ranked_top4500.json",
        4500,
    ),
]

TRAIN_DEV_PAIRED_SOURCES = [
    (
        "r09_minilm",
        "outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json",
        "outputs/round09/dev-rrf-bm25-char-minilm-ranked-top100.json",
        100,
    ),
    (
        "r09_gbdt",
        "outputs/round09/fusion_gbdt_top100/train-fusion-ranked-top50.json",
        "outputs/round09/fusion_gbdt_top100/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r09_gbdt_ref2",
        "outputs/round09/fusion_gbdt_top100_refutes_x2/train-fusion-ranked-top50.json",
        "outputs/round09/fusion_gbdt_top100_refutes_x2/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r09_gbdt_ref3",
        "outputs/round09/fusion_gbdt_top100_refutes_x3/train-fusion-ranked-top50.json",
        "outputs/round09/fusion_gbdt_top100_refutes_x3/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r15_minilm",
        "outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json",
        "outputs/round15/dev-round12-top500-minilm-ranked-top100.json",
        100,
    ),
    (
        "r15_gbdt_w1",
        "outputs/round15/fusion_gbdt_round12_minilm_w1/train-fusion-ranked-top50.json",
        "outputs/round15/fusion_gbdt_round12_minilm_w1/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r15_gbdt_w2",
        "outputs/round15/fusion_gbdt_round12_minilm_w2/train-fusion-ranked-top50.json",
        "outputs/round15/fusion_gbdt_round12_minilm_w2/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r15_gbdt_w3",
        "outputs/round15/fusion_gbdt_round12_minilm_w3/train-fusion-ranked-top50.json",
        "outputs/round15/fusion_gbdt_round12_minilm_w3/dev-fusion-ranked-top50.json",
        100,
    ),
    (
        "r15_logreg_w2",
        "outputs/round15/fusion_logreg_round12_minilm_w2/train-fusion-ranked-top50.json",
        "outputs/round15/fusion_logreg_round12_minilm_w2/dev-fusion-ranked-top50.json",
        100,
    ),
]


def load_any(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def normalize_ranked(data):
    ranked = {}
    for claim_id, value in data.items():
        rows = []
        if isinstance(value, dict) and "evidences" in value:
            rows = [
                {"evidence_id": evidence_id, "rank": rank, "score": 1.0 / rank}
                for rank, evidence_id in enumerate(value.get("evidences", []), start=1)
            ]
        elif isinstance(value, list):
            for rank, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id")
                    if not evidence_id:
                        continue
                    score = item.get("score", item.get("fusion_score", item.get("sparse_score", item.get("rrf_score", 0.0))))
                    rows.append(
                        {
                            "evidence_id": evidence_id,
                            "rank": int(item.get("rank", rank)),
                            "score": float(score),
                        }
                    )
                else:
                    rows.append({"evidence_id": str(item), "rank": rank, "score": 1.0 / rank})
        seen = set()
        unique = []
        for row in sorted(rows, key=lambda item: (item["rank"], -item.get("score", 0.0), item["evidence_id"])):
            if row["evidence_id"] in seen:
                continue
            seen.add(row["evidence_id"])
            unique.append(row)
        for rank, row in enumerate(unique, start=1):
            row["rank"] = rank
        ranked[claim_id] = unique
    return ranked


def load_sources(specs, side="dev"):
    sources = {}
    caps = {}
    for spec in specs:
        if len(spec) == 3:
            name, path, cap = spec
        else:
            name = spec[0]
            path = spec[1] if side == "train" else spec[2]
            cap = spec[3]
        if not Path(path).exists():
            continue
        sources[name] = normalize_ranked(load_any(path))
        caps[name] = cap
    return sources, caps


def build_source_index(sources, caps):
    indexed = {}
    for name, ranked in sources.items():
        cap = caps.get(name, 100)
        indexed[name] = {
            claim_id: {row["evidence_id"]: row for row in rows[:cap]}
            for claim_id, rows in ranked.items()
        }
    return indexed


def build_feature_matrix(claims, sources, caps):
    names = list(sources)
    indexed = build_source_index(sources, caps)
    x_rows = []
    y_rows = []
    groups = []
    row_ids = []
    for claim_id, claim in claims.items():
        union = []
        seen = set()
        for name in names:
            cap = caps.get(name, 100)
            for row in sources[name].get(claim_id, [])[:cap]:
                evidence_id = row["evidence_id"]
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                union.append(evidence_id)
        groups.append(len(union))
        gold = set(claim.get("evidences", []))
        for evidence_id in union:
            features = []
            present = 0
            ranks = []
            scores = []
            reciprocal_ranks = []
            for name in names:
                item = indexed[name].get(claim_id, {}).get(evidence_id)
                if item:
                    rank = float(item["rank"])
                    score = float(item.get("score", 0.0))
                    rr = 1.0 / (20.0 + rank)
                    present += 1
                    ranks.append(rank)
                    scores.append(score)
                    reciprocal_ranks.append(rr)
                    features.extend([1.0, min(rank, 5000.0) / 5000.0, rr, score])
                else:
                    features.extend([0.0, 1.0, 0.0, 0.0])
            features.extend(
                [
                    float(present),
                    min(ranks) if ranks else 9999.0,
                    float(np.mean(ranks)) if ranks else 9999.0,
                    max(reciprocal_ranks) if reciprocal_ranks else 0.0,
                    sum(reciprocal_ranks),
                    max(scores) if scores else 0.0,
                    float(np.mean(scores)) if scores else 0.0,
                ]
            )
            x_rows.append(features)
            y_rows.append(1 if evidence_id in gold else 0)
            row_ids.append((claim_id, evidence_id))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32), groups, row_ids


def train_ranker(x_train, y_train, groups, positive_weight, seed):
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=360,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=8,
        subsample=0.95,
        colsample_bytree=0.95,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    weights = np.ones(len(y_train), dtype=np.float32)
    weights[y_train == 1] = positive_weight
    ranker.fit(x_train, y_train, group=groups, sample_weight=weights)
    return ranker


def rank_from_scores(row_ids, scores):
    ranked = defaultdict(list)
    for (claim_id, evidence_id), score in zip(row_ids, scores):
        ranked[claim_id].append({"evidence_id": evidence_id, "score": float(score)})
    final = {}
    for claim_id, rows in ranked.items():
        rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        final[claim_id] = rows
    return final


def predictions_from_ranked(claims, ranked, label_source, default_label, top_k):
    return {
        claim_id: {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": [row["evidence_id"] for row in ranked.get(claim_id, [])[:top_k]],
        }
        for claim_id, claim in claims.items()
    }


def metric_row(name, claims, predictions, note):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "note": note,
        "macro_recall": macro_recall(claims, predictions),
        "retrieval_f_score": assignment["retrieval_f_score"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
    }
    for label in LABELS:
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(claims, predictions, label)["macro_recall"]
    return row


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_dev_diagnostic(dev_claims, label_source, default_label, output_dir):
    sources, caps = load_sources(DEV_PARALLEL_SOURCES, side="dev")
    x_dev, y_dev, groups, row_ids = build_feature_matrix(dev_claims, sources, caps)
    ranker = train_ranker(x_dev, y_dev, groups, positive_weight=20.0, seed=1701)
    ranked = rank_from_scores(row_ids, ranker.predict(x_dev))
    predictions = predictions_from_ranked(dev_claims, ranked, label_source, default_label, top_k=3)
    row = metric_row(
        "clean_parallel_aggregator_dev_diagnostic",
        dev_claims,
        predictions,
        "single LGBMRanker over parallel source features; trained and evaluated on dev, diagnostic only",
    )
    write_json(ranked, output_dir / "dev_diagnostic_clean_parallel_ranked.json")
    write_predictions(predictions, output_dir / "dev_diagnostic_clean_parallel_top3.json")
    return row


def run_train_only(train_claims, dev_claims, label_source, default_label, output_dir):
    train_sources, train_caps = load_sources(TRAIN_DEV_PAIRED_SOURCES, side="train")
    dev_sources, dev_caps = load_sources(TRAIN_DEV_PAIRED_SOURCES, side="dev")
    x_train, y_train, train_groups, _ = build_feature_matrix(train_claims, train_sources, train_caps)
    x_dev, _, _, row_ids = build_feature_matrix(dev_claims, dev_sources, dev_caps)
    ranker = train_ranker(x_train, y_train, train_groups, positive_weight=10.0, seed=1702)
    ranked = rank_from_scores(row_ids, ranker.predict(x_dev))
    predictions = predictions_from_ranked(dev_claims, ranked, label_source, default_label, top_k=3)
    row = metric_row(
        "clean_parallel_aggregator_train_only",
        dev_claims,
        predictions,
        "single LGBMRanker over paired train/dev source features; no dev labels used for fitting",
    )
    write_json(ranked, output_dir / "train_only_clean_parallel_ranked.json")
    write_predictions(predictions, output_dir / "train_only_clean_parallel_top3.json")
    return row


def main():
    parser = argparse.ArgumentParser(description="Round17 clean parallel-feature evidence aggregator.")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--label-source", default="outputs/round15/recommended/top3_submission.json")
    parser.add_argument("--output-dir", default="outputs/round17/clean_parallel_aggregator")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    label_source = load_any(args.label_source)
    default_label = majority_label(train_claims)

    rows = [
        run_train_only(train_claims, dev_claims, label_source, default_label, output_dir),
        run_dev_diagnostic(dev_claims, label_source, default_label, output_dir),
    ]
    write_csv(rows, output_dir / "summary.csv")
    write_json({"summary": rows}, output_dir / "summary.json")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
