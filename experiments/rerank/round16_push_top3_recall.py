import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

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


SAFE_SOURCE_PAIRS = [
    {
        "name": "r09_minilm",
        "train": "outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json",
        "dev": "outputs/round09/dev-rrf-bm25-char-minilm-ranked-top100.json",
    },
    {
        "name": "r09_gbdt",
        "train": "outputs/round09/fusion_gbdt_top100/train-fusion-ranked-top50.json",
        "dev": "outputs/round09/fusion_gbdt_top100/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r09_gbdt_ref2",
        "train": "outputs/round09/fusion_gbdt_top100_refutes_x2/train-fusion-ranked-top50.json",
        "dev": "outputs/round09/fusion_gbdt_top100_refutes_x2/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r09_gbdt_ref3",
        "train": "outputs/round09/fusion_gbdt_top100_refutes_x3/train-fusion-ranked-top50.json",
        "dev": "outputs/round09/fusion_gbdt_top100_refutes_x3/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r15_minilm",
        "train": "outputs/round09/train-rrf-bm25-char-minilm-ranked-top100.json",
        "dev": "outputs/round15/dev-round12-top500-minilm-ranked-top100.json",
    },
    {
        "name": "r15_gbdt_w1",
        "train": "outputs/round15/fusion_gbdt_round12_minilm_w1/train-fusion-ranked-top50.json",
        "dev": "outputs/round15/fusion_gbdt_round12_minilm_w1/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r15_gbdt_w2",
        "train": "outputs/round15/fusion_gbdt_round12_minilm_w2/train-fusion-ranked-top50.json",
        "dev": "outputs/round15/fusion_gbdt_round12_minilm_w2/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r15_gbdt_w3",
        "train": "outputs/round15/fusion_gbdt_round12_minilm_w3/train-fusion-ranked-top50.json",
        "dev": "outputs/round15/fusion_gbdt_round12_minilm_w3/dev-fusion-ranked-top50.json",
    },
    {
        "name": "r15_logreg_w2",
        "train": "outputs/round15/fusion_logreg_round12_minilm_w2/train-fusion-ranked-top50.json",
        "dev": "outputs/round15/fusion_logreg_round12_minilm_w2/dev-fusion-ranked-top50.json",
    },
]


DEV_DIAGNOSTIC_SOURCES = [
    ("r15_top3", "outputs/round15/rrf_top3_selector/ranked.json"),
    ("r15_top3_repro", "outputs/round15/rrf_top3_selector_repro/rrf_top3_selector-ranked.json"),
    ("r15_minilm", "outputs/round15/dev-round12-top500-minilm-ranked-top100.json"),
    ("r15_bge_reranker", "outputs/round15/dev-round12-top500-bge-reranker-base-ranked-top100.json"),
    ("r15_gbdt_w1", "outputs/round15/fusion_gbdt_round12_minilm_w1/dev-fusion-ranked-top50.json"),
    ("r15_gbdt_w2", "outputs/round15/fusion_gbdt_round12_minilm_w2/dev-fusion-ranked-top50.json"),
    ("r15_gbdt_w3", "outputs/round15/fusion_gbdt_round12_minilm_w3/dev-fusion-ranked-top50.json"),
    ("r15_logreg_w2", "outputs/round15/fusion_logreg_round12_minilm_w2/dev-fusion-ranked-top50.json"),
    ("r09_alpha04", "outputs/round09/blend_top100_refutes_x2/alpha_0.4_ranked.json"),
    ("r09_alpha07", "outputs/round09/blend_top50/alpha_0.7_ranked.json"),
    ("r09_alpha08", "outputs/round09/blend_top100_refutes_x2/alpha_0.8_ranked.json"),
    ("r09_minilm", "outputs/round09/dev-rrf-bm25-char-minilm-ranked-top100.json"),
    ("r09_gbdt", "outputs/round09/fusion_gbdt_top100/dev-fusion-ranked-top50.json"),
    ("r09_gbdt_ref2", "outputs/round09/fusion_gbdt_top100_refutes_x2/dev-fusion-ranked-top50.json"),
    ("r09_gbdt_ref3", "outputs/round09/fusion_gbdt_top100_refutes_x3/dev-fusion-ranked-top50.json"),
    ("r10_verifier", "outputs/round10/verifier_distilroberta/dev-verifier-ranked.json"),
    ("r10_verifier_e5", "outputs/round10/verifier_distilroberta_e5/dev-verifier-ranked.json"),
    ("r10_hybrid02", "outputs/round10/hybrid_verifier_e5/gamma_0.2_ranked.json"),
    ("r10_hybrid04", "outputs/round10/hybrid_verifier_e5/gamma_0.4_ranked.json"),
    ("r10_hybrid06", "outputs/round10/hybrid_verifier_e5/gamma_0.6_ranked.json"),
    ("r10_hybrid08", "outputs/round10/hybrid_verifier_e5/gamma_0.8_ranked.json"),
    ("r11_dense_bge_minilm", "outputs/round11/dev-rrf-bm25-char-dense-bge-top1000-minilm-ranked-top100.json"),
    ("r11_sparse_minilm", "outputs/round11/dev-rrf-bm25-char-top1000-minilm-ranked-top100.json"),
    ("r16_s22_rrf", "outputs/round16/baseline/s22_rrf_sparsegate_small_top100.json"),
    ("r16_s22_bm25char", "outputs/round16/baseline/s22_rrf_bm25_char_small_top100.json"),
    ("r16_s22_plain", "outputs/round16/baseline/s22_small_plain_top100.json"),
    ("r16_s22_qprefix", "outputs/round16/baseline/s22_small_qprefix_top100.json"),
    ("r16_a_ranker", "outputs/round16/branch_a_requirement/ranker_claim_key_ranked_top4500.json"),
    ("r16_a_fixed", "outputs/round16/branch_a_requirement/fixed_rrf_claim_key_ranked_top4500.json"),
    ("r16_b_courseware", "outputs/round16/branch_b_courseware/courseware_ranked_top4500.json"),
    ("r16_sparse_gate", "outputs/round14/s20_sparse_pool_mix_base_heavy_rrfk500_fine/weighted_rrf_top5000.json"),
]


def load_any(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def normalize_ranked(data):
    ranked = {}
    for claim_id, value in data.items():
        if isinstance(value, dict) and "evidences" in value:
            items = [
                {"evidence_id": evidence_id, "rank": rank, "score": 1.0 / rank}
                for rank, evidence_id in enumerate(value.get("evidences", []), start=1)
            ]
        elif isinstance(value, list):
            items = []
            for rank, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id")
                    if evidence_id is None:
                        continue
                    items.append(
                        {
                            "evidence_id": evidence_id,
                            "rank": int(item.get("rank", rank)),
                            "score": float(
                                item.get(
                                    "score",
                                    item.get("fusion_score", item.get("sparse_score", 0.0)),
                                )
                            ),
                        }
                    )
                else:
                    items.append(
                        {"evidence_id": str(item), "rank": rank, "score": 1.0 / rank}
                    )
        else:
            items = []
        seen = set()
        unique = []
        for item in sorted(items, key=lambda row: (row["rank"], -row["score"], row["evidence_id"])):
            if item["evidence_id"] in seen:
                continue
            seen.add(item["evidence_id"])
            unique.append(item)
        for rank, item in enumerate(unique, start=1):
            item["rank"] = rank
        ranked[claim_id] = unique
    return ranked


def source_from_path(path):
    return normalize_ranked(load_any(path))


def rrf_fuse(sources, weights, rrf_k, caps=None):
    caps = caps or {}
    claim_ids = set()
    for source in sources.values():
        claim_ids.update(source.keys())
    fused = {}
    for claim_id in claim_ids:
        scores = defaultdict(float)
        parts = defaultdict(dict)
        for name, source in sources.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            cap = caps.get(name)
            items = source.get(claim_id, [])
            if cap is not None:
                items = items[:cap]
            for idx, item in enumerate(items, start=1):
                evidence_id = item["evidence_id"]
                rank = int(item.get("rank", idx))
                scores[evidence_id] += weight / (rrf_k + rank)
                parts[evidence_id][f"{name}_rank"] = rank
        rows = [
            {"evidence_id": evidence_id, "score": float(score), **parts[evidence_id]}
            for evidence_id, score in scores.items()
        ]
        rows.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, item in enumerate(rows, start=1):
            item["rank"] = rank
        fused[claim_id] = rows
    return fused


def predictions_from_ranked(claims, ranked, label_source, default_label, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        label = label_source.get(claim_id, {}).get("claim_label", default_label)
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label,
            "evidences": [
                item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]
            ],
        }
    return predictions


def metric_row(name, claims, predictions, top_k, extra=None):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
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
        row[f"{label.lower()}_macro_recall"] = label_subset_metrics(
            claims, predictions, label
        )["macro_recall"]
    if extra:
        row.update(extra)
    return row


def write_rows(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def score_ranked(claims, ranked, label_source, default_label, top_k=3):
    predictions = predictions_from_ranked(claims, ranked, label_source, default_label, top_k)
    return metric_row("candidate", claims, predictions, top_k), predictions


def random_weight_configs(names, rng, trials, values):
    yielded = set()
    for size in range(1, min(2, len(names)) + 1):
        for subset in combinations(names, size):
            weights = {name: 0.0 for name in names}
            for name in subset:
                weights[name] = 1.0
            key = tuple(sorted(weights.items()))
            yielded.add(key)
            yield weights
    for _ in range(trials):
        size = rng.randint(2, min(8, len(names)))
        subset = rng.sample(names, size)
        weights = {name: 0.0 for name in names}
        for name in subset:
            weights[name] = rng.choice(values)
        key = tuple(sorted(weights.items()))
        if key in yielded:
            continue
        yielded.add(key)
        yield weights


def run_search(claims, sources, label_source, default_label, trials, seed, caps=None):
    rng = random.Random(seed)
    names = list(sources)
    values = [0.15, 0.25, 0.4, 0.65, 1.0, 1.5, 2.25, 3.0]
    rows = []
    best = None
    for rrf_k in [5.0, 10.0, 20.0, 45.0, 60.0, 100.0]:
        for weights in random_weight_configs(names, rng, trials, values):
            if sum(1 for value in weights.values() if value > 0) == 0:
                continue
            ranked = rrf_fuse(sources, weights, rrf_k, caps=caps)
            row, predictions = score_ranked(
                claims, ranked, label_source, default_label, top_k=3
            )
            active = {
                name: weight for name, weight in weights.items() if weight > 0
            }
            row.update(
                {
                    "rrf_k": rrf_k,
                    "sources": "|".join(f"{name}:{weight:g}" for name, weight in active.items()),
                }
            )
            rows.append(row)
            key = (
                row["macro_recall"],
                row["retrieval_f_score"],
                row["hit_any"],
                -len(active),
            )
            if best is None or key > best[0]:
                best = (key, row, ranked, predictions, weights)
    rows.sort(key=lambda row: (-row["macro_recall"], -row["retrieval_f_score"]))
    return rows, best


def oracle_union_predictions(claims, sources, label_source, default_label, per_source_k, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        pool = []
        seen = set()
        for source in sources.values():
            for item in source.get(claim_id, [])[:per_source_k]:
                evidence_id = item["evidence_id"]
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                pool.append(evidence_id)
        chosen = [evidence_id for evidence_id in pool if evidence_id in gold][:top_k]
        if len(chosen) < top_k:
            chosen.extend([evidence_id for evidence_id in pool if evidence_id not in chosen][: top_k - len(chosen)])
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label_source.get(claim_id, {}).get("claim_label", default_label),
            "evidences": chosen,
        }
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Round16 push top3 recall experiments.")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--output-dir", default="outputs/round16/top3_push")
    parser.add_argument("--trials", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=1601)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    default_label = majority_label(train_claims)
    round15_top3 = load_any("outputs/round15/recommended/top3_submission.json")

    safe_train_sources = {}
    safe_dev_sources = {}
    for spec in SAFE_SOURCE_PAIRS:
        if Path(spec["train"]).exists() and Path(spec["dev"]).exists():
            safe_train_sources[spec["name"]] = source_from_path(spec["train"])
            safe_dev_sources[spec["name"]] = source_from_path(spec["dev"])

    individual_rows = []
    for name, source in safe_dev_sources.items():
        row, _ = score_ranked(dev_claims, source, round15_top3, default_label)
        row["name"] = f"safe_individual:{name}"
        individual_rows.append(row)

    train_rows, best_train = run_search(
        train_claims,
        safe_train_sources,
        round15_top3,
        default_label,
        trials=args.trials,
        seed=args.seed,
        caps={name: 100 for name in safe_train_sources},
    )
    write_rows(train_rows[:250], output_dir / "train_fixed_search_top250.csv")

    best_train_row = best_train[1]
    best_train_weights = best_train[4]
    best_train_ranked_dev = rrf_fuse(
        safe_dev_sources,
        best_train_weights,
        float(best_train_row["rrf_k"]),
        caps={name: 100 for name in safe_dev_sources},
    )
    best_train_dev_row, best_train_dev_predictions = score_ranked(
        dev_claims, best_train_ranked_dev, round15_top3, default_label
    )
    best_train_dev_row["name"] = "train_fixed_rrf_on_dev"
    best_train_dev_row["rrf_k"] = best_train_row["rrf_k"]
    best_train_dev_row["sources"] = best_train_row["sources"]

    dev_sources = {}
    for name, path in DEV_DIAGNOSTIC_SOURCES:
        if Path(path).exists():
            dev_sources[name] = source_from_path(path)
    for name, source in dev_sources.items():
        row, _ = score_ranked(dev_claims, source, round15_top3, default_label)
        row["name"] = f"dev_individual:{name}"
        individual_rows.append(row)
    individual_rows.sort(key=lambda row: (-row["macro_recall"], -row["retrieval_f_score"]))
    write_rows(individual_rows, output_dir / "individual_sources.csv")

    dev_source_scores = {}
    for row in individual_rows:
        name = row["name"]
        if name.startswith("dev_individual:"):
            dev_source_scores[name.split(":", 1)[1]] = row["macro_recall"]
    dev_search_names = [
        name
        for name, _ in sorted(
            dev_source_scores.items(), key=lambda item: item[1], reverse=True
        )[:16]
    ]
    for required in ["r15_top3", "r16_s22_rrf", "r16_s22_qprefix", "r16_a_fixed"]:
        if required in dev_sources and required not in dev_search_names:
            dev_search_names.append(required)
    dev_search_sources = {name: dev_sources[name] for name in dev_search_names}

    dev_rows, best_dev = run_search(
        dev_claims,
        dev_search_sources,
        round15_top3,
        default_label,
        trials=args.trials,
        seed=args.seed + 17,
        caps={name: 100 for name in dev_search_sources},
    )
    write_rows(dev_rows[:500], output_dir / "dev_diagnostic_search_top500.csv")

    best_dev_row = best_dev[1]
    best_dev_ranked = best_dev[2]
    best_dev_predictions = best_dev[3]

    oracle_rows = []
    for per_source_k in [3, 5, 10, 20]:
        oracle_predictions = oracle_union_predictions(
            dev_claims,
            dev_sources,
            round15_top3,
            default_label,
            per_source_k=per_source_k,
            top_k=3,
        )
        oracle_rows.append(
            metric_row(
                f"oracle_union_per_source_top{per_source_k}",
                dev_claims,
                oracle_predictions,
                3,
            )
        )

    summary_rows = [
        best_train_dev_row,
        {**best_dev_row, "name": "dev_diagnostic_rrf_best"},
        *oracle_rows,
    ]
    write_rows(summary_rows, output_dir / "summary.csv")
    write_predictions(
        best_train_dev_predictions,
        output_dir / "train_fixed_rrf_top3.json",
    )
    write_predictions(best_dev_predictions, output_dir / "dev_diagnostic_rrf_top3.json")
    (output_dir / "dev_diagnostic_rrf_ranked.json").write_text(
        json.dumps(best_dev_ranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_fixed_on_dev": best_train_dev_row,
                "dev_diagnostic_best": best_dev_row,
                "oracle_rows": oracle_rows,
                "safe_sources": list(safe_dev_sources),
                "dev_sources": list(dev_sources),
                "dev_search_sources": list(dev_search_sources),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
