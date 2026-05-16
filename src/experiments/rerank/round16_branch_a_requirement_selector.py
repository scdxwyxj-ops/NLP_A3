import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from a3_factcheck.data import load_json, majority_label
from a3_factcheck.metrics import (
    aggregate_confusion,
    assignment_metrics,
    confusion_rows,
    label_subset_metrics,
    macro_recall,
)
from a3_factcheck.rerank.api import write_predictions
from a3_factcheck.semantic.features import extract_semantic_features


LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}
NEGATION = {"no", "not", "never", "none", "without", "cannot", "can't", "n't"}
RELATION_HINTS = {
    "cause",
    "caused",
    "causes",
    "lead",
    "leads",
    "led",
    "result",
    "results",
    "increase",
    "increases",
    "increased",
    "decrease",
    "decreases",
    "decreased",
    "reduce",
    "reduces",
    "reduced",
    "produce",
    "produces",
    "produced",
    "emit",
    "emits",
    "emitted",
    "warm",
    "warms",
    "warmed",
    "cool",
    "cools",
    "cooled",
    "affect",
    "affects",
    "affected",
}


FEATURES = [
    "source_rank",
    "source_score",
    "source_rr",
    "top100_rank",
    "top3_rank",
    "minilm_rank",
    "bge_rank",
    "gbdt_rank",
    "claim_key_weighted_coverage",
    "claim_key_exact_coverage",
    "claim_key_soft_maxsim",
    "claim_key_min_maxsim",
    "claim_key_top3_mean",
    "content_recall",
    "content_precision",
    "content_jaccard",
    "bigram_recall",
    "trigram_recall",
    "char3_jaccard",
    "char4_jaccard",
    "entity_overlap",
    "entity_recall",
    "entity_jaccard",
    "number_overlap",
    "number_recall",
    "year_overlap",
    "year_recall",
    "negation_match",
    "negation_xor",
    "comparison_overlap",
    "causality_overlap",
    "relation_overlap",
    "claim_len",
    "evidence_len",
    "length_ratio",
]


def load_ranked(path):
    if not path or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def rank_index(ranked):
    indexed = {}
    for claim_id, items in ranked.items():
        current = {}
        for idx, item in enumerate(items, start=1):
            evidence_id = item["evidence_id"] if isinstance(item, dict) else item
            rank = int(item.get("rank", idx)) if isinstance(item, dict) else idx
            score = float(item.get("score", item.get("fusion_score", 0.0))) if isinstance(item, dict) else 0.0
            current[evidence_id] = (rank, score)
        indexed[claim_id] = current
    return indexed


def prediction_rank_index(predictions):
    indexed = {}
    for claim_id, pred in predictions.items():
        indexed[claim_id] = {
            evidence_id: (idx, 1.0 / idx)
            for idx, evidence_id in enumerate(pred.get("evidences", []), start=1)
        }
    return indexed


def tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


def content_tokens(text):
    return [token for token in tokens(text) if token not in STOPWORDS and len(token) > 1]


def ngrams(items, n):
    return set(zip(*(items[i:] for i in range(n)))) if len(items) >= n else set()


def char_ngrams(text, n):
    compact = re.sub(r"\s+", " ", text.lower())
    if len(compact) < n:
        return set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(left, right):
    if not left and not right:
        return 0.0
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def overlap_count(left, right):
    return len({item.lower() for item in left} & {item.lower() for item in right})


def recall(left, right):
    left = set(left)
    if not left:
        return 0.0
    return len(left & set(right)) / len(left)


def precision(left, right):
    right = set(right)
    if not right:
        return 0.0
    return len(set(left) & right) / len(right)


def soft_token_match(claim_token, evidence_tokens):
    if claim_token in evidence_tokens:
        return 1.0
    if len(claim_token) >= 5:
        prefix = claim_token[:5]
        if any(token.startswith(prefix) or prefix.startswith(token[:5]) for token in evidence_tokens if len(token) >= 5):
            return 0.78
    if len(claim_token) >= 4 and any(claim_token in token or token in claim_token for token in evidence_tokens if len(token) >= 4):
        return 0.58
    return 0.0


def claim_key_weights(claim_text):
    raw_tokens = tokens(claim_text)
    feats = extract_semantic_features(claim_text)
    entity_tokens = set()
    for entity in feats["entities"]:
        entity_tokens.update(tokens(entity))
    number_tokens = set(NUMBER_RE.findall(claim_text))
    year_tokens = set(YEAR_RE.findall(claim_text))
    keys = []
    for token in raw_tokens:
        if token in STOPWORDS and token not in NEGATION:
            continue
        weight = 1.0
        if token in entity_tokens:
            weight += 1.2
        if token in number_tokens or token in year_tokens:
            weight += 1.5
        if token in NEGATION:
            weight += 1.2
        if token in RELATION_HINTS:
            weight += 0.8
        if len(token) >= 8:
            weight += 0.3
        keys.append((token, weight))
    return keys


def text_profile(text):
    toks = tokens(text)
    content = [token for token in toks if token not in STOPWORDS and len(token) > 1]
    return {
        "tokens": toks,
        "token_set": set(toks),
        "content": content,
        "content_set": set(content),
        "bigrams": ngrams(content, 2),
        "trigrams": ngrams(content, 3),
        "char3": char_ngrams(text, 3),
        "char4": char_ngrams(text, 4),
        "semantic": extract_semantic_features(text),
        "numbers": set(NUMBER_RE.findall(text)),
        "years": set(YEAR_RE.findall(text)),
    }


def build_idf(claims):
    doc_freq = Counter()
    for claim in claims.values():
        for token in set(content_tokens(claim["claim_text"])):
            doc_freq[token] += 1
    n_docs = max(1, len(claims))
    return {
        token: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def make_features(claim_profile, evidence_profile, source_rank, source_score, extra_ranks):
    key_weights = claim_profile["key_weights"]
    weighted_total = sum(weight for _, weight in key_weights) or 1.0
    maxsims = []
    exact_weight = 0.0
    soft_weight = 0.0
    for token, weight in key_weights:
        sim = soft_token_match(token, evidence_profile["token_set"])
        maxsims.append(sim)
        if sim == 1.0:
            exact_weight += weight
        soft_weight += weight * sim
    sorted_sims = sorted(maxsims, reverse=True)

    c_sem = claim_profile["semantic"]
    e_sem = evidence_profile["semantic"]
    claim_numbers = claim_profile["numbers"]
    evidence_numbers = evidence_profile["numbers"]
    claim_years = claim_profile["years"]
    evidence_years = evidence_profile["years"]
    length_ratio = len(evidence_profile["tokens"]) / max(1, len(claim_profile["tokens"]))

    row = {
        "source_rank": float(source_rank),
        "source_score": float(source_score),
        "source_rr": 1.0 / (60.0 + float(source_rank)),
        "claim_key_weighted_coverage": soft_weight / weighted_total,
        "claim_key_exact_coverage": exact_weight / weighted_total,
        "claim_key_soft_maxsim": float(np.mean(maxsims)) if maxsims else 0.0,
        "claim_key_min_maxsim": min(maxsims) if maxsims else 0.0,
        "claim_key_top3_mean": float(np.mean(sorted_sims[:3])) if sorted_sims else 0.0,
        "content_recall": recall(claim_profile["content_set"], evidence_profile["content_set"]),
        "content_precision": precision(claim_profile["content_set"], evidence_profile["content_set"]),
        "content_jaccard": jaccard(claim_profile["content_set"], evidence_profile["content_set"]),
        "bigram_recall": recall(claim_profile["bigrams"], evidence_profile["bigrams"]),
        "trigram_recall": recall(claim_profile["trigrams"], evidence_profile["trigrams"]),
        "char3_jaccard": jaccard(claim_profile["char3"], evidence_profile["char3"]),
        "char4_jaccard": jaccard(claim_profile["char4"], evidence_profile["char4"]),
        "entity_overlap": overlap_count(c_sem["entities"], e_sem["entities"]),
        "entity_recall": recall([x.lower() for x in c_sem["entities"]], [x.lower() for x in e_sem["entities"]]),
        "entity_jaccard": jaccard([x.lower() for x in c_sem["entities"]], [x.lower() for x in e_sem["entities"]]),
        "number_overlap": len(claim_numbers & evidence_numbers),
        "number_recall": recall(claim_numbers, evidence_numbers),
        "year_overlap": len(claim_years & evidence_years),
        "year_recall": recall(claim_years, evidence_years),
        "negation_match": float(bool(c_sem["negation_cues"]) == bool(e_sem["negation_cues"])),
        "negation_xor": float(bool(c_sem["negation_cues"]) != bool(e_sem["negation_cues"])),
        "comparison_overlap": overlap_count(c_sem["comparison_cues"], e_sem["comparison_cues"]),
        "causality_overlap": overlap_count(c_sem["causality_cues"], e_sem["causality_cues"]),
        "relation_overlap": overlap_count(c_sem["relation_verbs"], e_sem["relation_verbs"]),
        "claim_len": len(claim_profile["tokens"]),
        "evidence_len": len(evidence_profile["tokens"]),
        "length_ratio": length_ratio,
    }
    for name in ["top100_rank", "top3_rank", "minilm_rank", "bge_rank", "gbdt_rank"]:
        row[name] = float(extra_ranks.get(name, 999999))
    return row


def prepare_claim_profiles(claims):
    profiles = {}
    for claim_id, claim in claims.items():
        profile = text_profile(claim["claim_text"])
        profile["key_weights"] = claim_key_weights(claim["claim_text"])
        profiles[claim_id] = profile
    return profiles


def candidate_ids_for_claim(items, limit):
    ids = []
    seen = set()
    for item in items[:limit]:
        evidence_id = item["evidence_id"]
        if evidence_id not in seen:
            seen.add(evidence_id)
            ids.append((evidence_id, int(item.get("rank", len(ids) + 1)), float(item.get("score", 0.0))))
    return ids


def matrix_from_rows(rows):
    return np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float32)


def grouped_rows(claims, evidence, pool, pool_limit, source_indexes, include_gold):
    claim_profiles = prepare_claim_profiles(claims)
    evidence_cache = {}
    rows = []
    labels = []
    groups = []
    row_claims = []
    row_evidence_ids = []
    for claim_id, claim in claims.items():
        candidates = candidate_ids_for_claim(pool.get(claim_id, []), pool_limit)
        if include_gold:
            seen = {evidence_id for evidence_id, _, _ in candidates}
            for evidence_id in claim.get("evidences", []):
                if evidence_id not in seen:
                    candidates.append((evidence_id, 999998, 0.0))
        groups.append(len(candidates))
        gold = set(claim.get("evidences", []))
        for evidence_id, source_rank, source_score in candidates:
            if evidence_id not in evidence_cache:
                evidence_cache[evidence_id] = text_profile(evidence[evidence_id])
            extra_ranks = {}
            for name, indexed in source_indexes.items():
                if evidence_id in indexed.get(claim_id, {}):
                    extra_ranks[name] = indexed[claim_id][evidence_id][0]
            rows.append(
                make_features(
                    claim_profiles[claim_id],
                    evidence_cache[evidence_id],
                    source_rank,
                    source_score,
                    extra_ranks,
                )
            )
            labels.append(1 if evidence_id in gold else 0)
            row_claims.append(claim_id)
            row_evidence_ids.append(evidence_id)
    return rows, np.asarray(labels, dtype=np.int32), groups, row_claims, row_evidence_ids


def stream_ranked_predictions(claims, evidence, pool, pool_limit, source_indexes, ranker):
    claim_profiles = prepare_claim_profiles(claims)
    ranked = {}
    dev_positive_rows = 0
    for claim_id, claim in claims.items():
        candidates = candidate_ids_for_claim(pool.get(claim_id, []), pool_limit)
        rows = []
        evidence_ids = []
        gold = set(claim.get("evidences", []))
        for evidence_id, source_rank, source_score in candidates:
            evidence_profile = text_profile(evidence[evidence_id])
            extra_ranks = {}
            for name, indexed in source_indexes.items():
                if evidence_id in indexed.get(claim_id, {}):
                    extra_ranks[name] = indexed[claim_id][evidence_id][0]
            rows.append(
                make_features(
                    claim_profiles[claim_id],
                    evidence_profile,
                    source_rank,
                    source_score,
                    extra_ranks,
                )
            )
            evidence_ids.append(evidence_id)
            if evidence_id in gold:
                dev_positive_rows += 1
        scores = ranker.predict(matrix_from_rows(rows)) if rows else []
        ranked[claim_id] = ranked_from_scores(
            [claim_id] * len(evidence_ids),
            evidence_ids,
            rows,
            scores,
        ).get(claim_id, [])
    return ranked, dev_positive_rows


def ranked_from_scores(row_claims, row_evidence_ids, rows, scores):
    ranked = defaultdict(list)
    for claim_id, evidence_id, row, score in zip(row_claims, row_evidence_ids, rows, scores):
        item = {
            "evidence_id": evidence_id,
            "score": float(score),
            "source_rank": int(row["source_rank"]),
            "source_score": float(row["source_score"]),
            "claim_key_weighted_coverage": float(row["claim_key_weighted_coverage"]),
            "content_recall": float(row["content_recall"]),
            "entity_recall": float(row["entity_recall"]),
            "number_recall": float(row["number_recall"]),
            "year_recall": float(row["year_recall"]),
        }
        ranked[claim_id].append(item)
    final = {}
    for claim_id, items in ranked.items():
        items.sort(key=lambda item: (-item["score"], item["source_rank"], item["evidence_id"]))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
        final[claim_id] = items
    return final


def rrf_fuse_ranked(sources, weights, rrf_k=60.0):
    claim_ids = set()
    for source in sources.values():
        claim_ids.update(source.keys())
    fused = {}
    for claim_id in claim_ids:
        scores = defaultdict(float)
        parts = defaultdict(dict)
        for name, source in sources.items():
            weight = weights.get(name, 0.0)
            if weight == 0.0:
                continue
            for idx, item in enumerate(source.get(claim_id, []), start=1):
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


def prediction_from_ranked(claims, ranked, label_source, default_label, top_k):
    predictions = {}
    for claim_id, claim in claims.items():
        label = label_source.get(claim_id, {}).get("claim_label", default_label)
        predictions[claim_id] = {
            "claim_text": claim["claim_text"],
            "claim_label": label,
            "evidences": [item["evidence_id"] for item in ranked.get(claim_id, [])[:top_k]],
        }
    return predictions


def summary_row(name, top_k, path, claims, predictions, note):
    assignment = assignment_metrics(claims, predictions)
    aggregate = aggregate_confusion(confusion_rows(claims, predictions))
    row = {
        "name": name,
        "top_k": top_k,
        "path": str(path),
        "note": note,
        "top100_macro_recall": "",
        "top3_macro_recall": macro_recall(claims, predictions) if top_k == 3 else "",
        "top3_evidence_f": assignment["retrieval_f_score"] if top_k == 3 else "",
        "macro_recall": macro_recall(claims, predictions),
        "micro_recall": aggregate["micro_recall"],
        "precision": aggregate["precision"],
        "hit_any": aggregate["hit_any"],
        "all_gold": aggregate["all_gold"],
        "claim_accuracy": assignment["claim_accuracy"],
        "harmonic_mean": assignment["harmonic_mean"],
    }
    if top_k == 100:
        row["top100_macro_recall"] = row["macro_recall"]
    for label in LABELS:
        metric = label_subset_metrics(claims, predictions, label)
        row[f"{label.lower()}_macro_recall"] = metric["macro_recall"]
    return row


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Round16 Branch A claim-key evidence selector.")
    parser.add_argument("--train-claims", default="data/train-claims.json")
    parser.add_argument("--dev-claims", default="data/dev-claims.json")
    parser.add_argument("--evidence", default="data/evidence.json")
    parser.add_argument("--train-pool", default="outputs/round14/s0_small_only_train_rrf/candidates/rrf_bm25_char_small_smallq_k500_top2000.json")
    parser.add_argument("--dev-pool", default="outputs/round14/s20_sparse_pool_mix_base_heavy_rrfk500_fine/weighted_rrf_top5000.json")
    parser.add_argument("--train-pool-limit", type=int, default=200)
    parser.add_argument("--dev-pool-limit", type=int, default=4500)
    parser.add_argument("--output-dir", default="outputs/round16/branch_a_requirement")
    parser.add_argument("--recommended-top3", default="outputs/round15/recommended/top3_submission.json")
    parser.add_argument("--recommended-top100", default="outputs/round15/recommended/top100_classifier_context.json")
    parser.add_argument("--minilm-ranked", default="outputs/round15/dev-round12-top500-minilm-ranked-top100.json")
    parser.add_argument("--bge-ranked", default="outputs/round15/dev-round12-top500-bge-reranker-base-ranked-top100.json")
    parser.add_argument("--gbdt-ranked", default="outputs/round15/fusion_gbdt_round12_minilm_w2/dev-fusion-ranked-top50.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_claims = load_json(args.train_claims)
    dev_claims = load_json(args.dev_claims)
    evidence = load_json(args.evidence)
    train_pool = load_ranked(args.train_pool)
    dev_pool = load_ranked(args.dev_pool)

    recommended_top3 = load_ranked(args.recommended_top3)
    recommended_top100 = load_ranked(args.recommended_top100)
    source_indexes = {
        "top100_rank": prediction_rank_index(recommended_top100),
        "top3_rank": prediction_rank_index(recommended_top3),
        "minilm_rank": rank_index(load_ranked(args.minilm_ranked)),
        "bge_rank": rank_index(load_ranked(args.bge_ranked)),
        "gbdt_rank": rank_index(load_ranked(args.gbdt_ranked)),
    }

    train_rows, train_y, train_groups, _, _ = grouped_rows(
        train_claims,
        evidence,
        train_pool,
        args.train_pool_limit,
        {},
        include_gold=True,
    )
    train_x = matrix_from_rows(train_rows)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=260,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=1601,
        n_jobs=-1,
        verbose=-1,
    )
    weights = np.ones(len(train_y), dtype=np.float32)
    weights[train_y == 1] = 8.0
    ranker.fit(train_x, train_y, group=train_groups, sample_weight=weights)
    ranker_ranked, dev_positive_rows = stream_ranked_predictions(
        dev_claims,
        evidence,
        dev_pool,
        args.dev_pool_limit,
        source_indexes,
        ranker,
    )
    write_json(ranker_ranked, output_dir / "ranker_claim_key_ranked_top4500.json")

    fixed_sources = {
        "ranker": ranker_ranked,
        "top100": {
            claim_id: [{"evidence_id": evidence_id, "rank": i} for i, evidence_id in enumerate(pred["evidences"], start=1)]
            for claim_id, pred in recommended_top100.items()
        },
        "top3": {
            claim_id: [{"evidence_id": evidence_id, "rank": i} for i, evidence_id in enumerate(pred["evidences"], start=1)]
            for claim_id, pred in recommended_top3.items()
        },
        "gate": dev_pool,
    }
    fixed_weights = {
        "ranker": 1.35,
        "top100": 0.70,
        "top3": 0.80,
        "gate": 0.20,
    }
    fused_ranked = rrf_fuse_ranked(fixed_sources, fixed_weights, rrf_k=45.0)
    write_json(fused_ranked, output_dir / "fixed_rrf_claim_key_ranked_top4500.json")

    default_label = majority_label(train_claims)
    label_source = recommended_top3 or recommended_top100
    summary_rows = []
    outputs = [
        ("ranker_claim_key", ranker_ranked, "train-only LightGBM LambdaRank over lexical Claim-Key MaxSim features"),
        ("fixed_rrf_claim_key", fused_ranked, "fixed no-dev-tuned RRF: ranker + round15 recommended + sparse gate"),
    ]
    for name, ranked, note in outputs:
        for top_k in [100, 3]:
            pred = prediction_from_ranked(dev_claims, ranked, label_source, default_label, top_k)
            out_path = output_dir / f"{name}_top{top_k}.json"
            write_predictions(pred, out_path)
            summary_rows.append(summary_row(name, top_k, out_path, dev_claims, pred, note))

    baseline_rows = []
    for name, pred_path, note in [
        ("round15_recommended_top3", Path(args.recommended_top3), "existing best baseline"),
        ("round15_recommended_top100", Path(args.recommended_top100), "existing classifier context baseline"),
    ]:
        pred = load_ranked(pred_path)
        top_k = 3 if "top3" in name else 100
        baseline_rows.append(summary_row(name, top_k, pred_path, dev_claims, pred, note))
    summary_rows = baseline_rows + summary_rows

    write_summary(summary_rows, output_dir / "summary.csv")
    write_json(
        {
            "feature_names": FEATURES,
            "train_rows": len(train_rows),
            "train_positive_rows": int(train_y.sum()),
            "dev_rows": sum(min(len(dev_pool.get(claim_id, [])), args.dev_pool_limit) for claim_id in dev_claims),
            "dev_positive_rows_in_gate": int(dev_positive_rows),
            "train_pool": args.train_pool,
            "train_pool_limit": args.train_pool_limit,
            "dev_pool": args.dev_pool,
            "dev_pool_limit": args.dev_pool_limit,
            "model": "LGBMRanker lambdarank",
            "fixed_rrf_weights": fixed_weights,
            "summary": summary_rows,
            "feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(zip(FEATURES, ranker.feature_importances_), key=lambda x: x[1], reverse=True)
            ],
        },
        output_dir / "summary.json",
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
