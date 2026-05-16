from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.retrieval.run_round13_colab_generator import query_bm25_pool
from round18.tools.common import (
    find_forbidden_tokens,
    load_json,
    manifest_base,
    sha256_file,
    write_json,
)


SMART_QUOTES = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "’": "'",
    "‘": "'",
    "\u00a0": " ",
})

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
DATE_PATTERNS = [
    re.compile(
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{4}\s*(?:ce|ad|bce|ce\.?|ad\.?|bc|b\.c\.)\b",
        re.IGNORECASE,
    ),
]
DATE_SUFFIX_RE = re.compile(r"^\d{4}$")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
NUMBER_UNIT_RE = re.compile(
    rf"""
    \b(?P<number>\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?)
    \s*(?P<unit>
    %|percent|per\s+cent|
    kg|kilogram|kg\s*co2|kgco2|co2|
    g|gram|mg|µg|ug|
    t|ton|tons|tonne|tonnes|
    mm|cm|m|km|ha|hectare|hectares|
    ppm|ppb|ppt|
    wh|kwh|mwh|gwh|twh|
    °\s*c|°\s*f|celsius|fahrenheit|kelvin
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
ENTITY_STOP = {
    "the",
    "a",
    "an",
    "and",
    "as",
    "at",
    "in",
    "on",
    "for",
    "of",
    "to",
    "is",
    "are",
    "was",
    "were",
    "that",
    "with",
    "this",
    "from",
    "its",
    "it",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "than",
    "into",
    "within",
    "across",
    "between",
    "their",
    "there",
    "these",
    "those",
    "also",
    "not",
    "may",
    "can",
}
ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")
ROUTE_IDS = ("baseline", "entity", "number", "date")


@dataclass
class RouteSpec:
    route: str
    query: str
    evidence_hits: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round18 O-S2 structured sparse routing prototype from raw claim/evidence JSON."
        )
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/dev-claims.json"),
        help="Split claim file path.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence.json"),
        help="Evidence corpus path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s2_structured"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-id",
        default="o_s2_structured",
        help="Run identifier used in output filenames.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s2_structured/run_manifest.json"),
        help="Manifest output path.",
    )
    parser.add_argument("--candidate-k", type=int, default=500, help="Top-k candidate output.")
    parser.add_argument(
        "--eval-k",
        type=str,
        default="100,500,1000",
        help="Comma-separated k for recall metrics.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Route-aware RRF-like fusion denominator.",
    )
    parser.add_argument("--max-features", type=int, default=200_000, help="BM25 vocab cap.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke mode.")
    parser.add_argument("--max-claims", type=int, default=0, help="Cap claims (0 = no cap).")
    parser.add_argument("--max-evidence", type=int, default=0, help="Cap evidence (0 = no cap).")
    parser.add_argument(
        "--smoke-claims",
        type=int,
        default=64,
        help="Smoke cap for claims when --smoke is set.",
    )
    parser.add_argument(
        "--smoke-evidence",
        type=int,
        default=50_000,
        help="Smoke cap for evidence docs when --smoke is set.",
    )
    parser.add_argument(
        "--route-baseline-weight",
        type=float,
        default=1.0,
        help="RRF weight for baseline claim wording.",
    )
    parser.add_argument(
        "--route-entity-weight",
        type=float,
        default=1.3,
        help="RRF weight for named-entity-like routing.",
    )
    parser.add_argument(
        "--route-number-weight",
        type=float,
        default=1.4,
        help="RRF weight for number/unit/date-like routing.",
    )
    parser.add_argument(
        "--route-date-weight",
        type=float,
        default=1.2,
        help="RRF weight for date-route.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=Path("round18/outputs/o_sparse/o_s2_structured/run_record.json"),
        help="Run record output path.",
    )
    parser.add_argument("--random-seed", type=int, default=1337, help="Reproducibility seed marker.")
    parser.add_argument(
        "--stage",
        type=str,
        default="o_s2_structured",
        help="Manifest stage label.",
    )
    return parser.parse_args()


def parse_k_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("eval-k must include at least one integer.")
    return sorted(set(values))


def clean_text(text: str) -> str:
    text = text.translate(SMART_QUOTES)
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalise_unit(unit: str) -> str:
    unit = re.sub(r"\s+", " ", unit.strip().lower())
    return unit.replace("  ", " ")


def _is_capitalized_word(word: str) -> bool:
    return bool(word) and word[0].isupper() and not word.isupper()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def extract_entities(text: str) -> list[str]:
    cleaned = clean_text(text)
    entities: list[str] = []
    current: list[str] = []
    for word in WORD_RE.findall(cleaned):
        lower = word.lower()
        if word in (" ", ""):
            continue
        if len(word) < 2:
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
            continue
        if word.lower() in ENTITY_STOP:
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
            continue
        if _is_capitalized_word(word) or bool(ACRONYM_RE.fullmatch(word)):
            current.append(word)
        else:
            if current:
                entities.extend(_finalize_entity_tokens(current))
                current = []
    if current:
        entities.extend(_finalize_entity_tokens(current))
    return _dedupe([e for e in entities if e])


def _finalize_entity_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) >= 3 and not token.lower() in ENTITY_STOP:
            return [token]
        return []
    phrase = " ".join(tokens)
    if len(phrase) <= 3:
        return []
    return [phrase] + tokens


def extract_numbers_and_units(text: str) -> list[str]:
    matched: list[str] = []
    for match in NUMBER_UNIT_RE.finditer(text):
        number = match.group("number")
        unit = _normalise_unit(match.group("unit"))
        matched.append(f"{number} {unit}".strip())
    if matched:
        return _dedupe([item for item in matched if item])

    # fall back to generic number extraction when no explicit unit exists
    return _dedupe([m.group(0) for m in NUMBER_RE.finditer(text)])


def extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if not candidate:
                continue
            values.append(candidate)
    return _dedupe([value.strip() for value in values if value.strip()])


def decompose_claim(claim_text: str) -> dict[str, list[str]]:
    entities = extract_entities(claim_text)
    numbers = extract_numbers_and_units(claim_text)
    dates = extract_dates(claim_text)
    routes = {"baseline": [clean_text(claim_text)]}
    if entities:
        routes["entity"] = [" ".join(entities[:12])]
    if numbers:
        routes["number"] = [" ".join(numbers[:12])]
    if dates:
        routes["date"] = [" ".join(dates[:6])]
    return routes


def limit_dict(data: dict[str, Any], max_items: int) -> dict[str, Any]:
    if max_items <= 0 or len(data) <= max_items:
        return dict(data)
    head = list(data.items())[:max_items]
    return {key: value for key, value in head}


def build_route_claim_sets(
    claims: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, RouteSpec]], dict[str, int]]:
    route_to_claims: dict[str, dict[str, dict[str, str]]] = {route: {} for route in ROUTE_IDS}
    route_specs: dict[str, dict[str, RouteSpec]] = {route: {} for route in ROUTE_IDS}
    stats: Counter[str] = Counter()

    for claim_id, claim in claims.items():
        claim_text = str(claim.get("claim_text", "")).strip()
        if not claim_text:
            continue
        decomposed = decompose_claim(claim_text)
        for route, queries in decomposed.items():
            if route not in route_to_claims or not queries:
                continue
            stats[route] += 1
            route_query = queries[0].strip()
            if not route_query:
                continue
            route_to_claims[route][claim_id] = {"claim_text": route_query}
            route_specs[route][claim_id] = RouteSpec(
                route=route,
                query=route_query,
            )
    return route_to_claims, route_specs, dict(stats)


def collect_route_pools(
    route_to_claims: dict[str, dict[str, dict[str, str]]],
    evidence: dict[str, str],
    top_k: int,
    k1: float,
    b: float,
    max_features: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    route_pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for route, route_claims in route_to_claims.items():
        if route_claims:
            raw_pool = query_bm25_pool(route_claims, evidence, top_k, k1=k1, b=b)
            routed_pool: dict[str, list[dict[str, Any]]] = {}
            for claim_id, candidates in raw_pool.items():
                routed_pool[claim_id] = [
                    {
                        "evidence_id": candidate.evidence_id,
                        "rank": candidate.rank,
                        "score": float(candidate.score),
                    }
                    for candidate in candidates
                ]
            route_pools[route] = routed_pool
        else:
            route_pools[route] = {}
    return route_pools


def merge_route_pools(
    claims: dict[str, Any],
    route_pools: dict[str, dict[str, list[dict[str, Any]]]],
    route_weights: dict[str, float],
    rrf_k: int,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for claim_id in claims:
        score_map: dict[str, float] = {}
        route_map: dict[str, list[dict[str, float | str | int]]] = defaultdict(list)
        for route, pool in route_pools.items():
            weight = route_weights.get(route, 1.0)
            if weight <= 0:
                continue
            for candidate in pool.get(claim_id, []):
                evidence_id = candidate["evidence_id"]
                route_contrib = weight / (rrf_k + float(candidate["rank"]))
                score_map[evidence_id] = score_map.get(evidence_id, 0.0) + route_contrib
                route_map[evidence_id].append(
                    {
                        "route": route,
                        "route_rank": int(candidate["rank"]),
                        "route_score": float(candidate["score"]),
                        "route_weight": float(weight),
                        "route_contribution": float(route_contrib),
                    }
                )

        ranked_items = sorted(
            score_map.items(),
            key=lambda item: (-item[1], item[0]),
        )
        candidates = []
        for rank, (evidence_id, score) in enumerate(ranked_items, start=1):
            candidate = {
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "rank": rank,
                "score": float(score),
                "routes": route_map[evidence_id],
                "source": "|".join(sorted({entry["route"] for entry in route_map[evidence_id]})),
            }
            candidates.append(candidate)
        merged[claim_id] = candidates
    return merged


def evaluate_recall(
    claims: dict[str, Any],
    pool: dict[str, list[dict[str, Any]]],
    ks: list[int],
) -> dict[str, float | int | list[int]]:
    ks = sorted(set(int(value) for value in ks if int(value) > 0))
    label_totals: dict[str, int] = Counter()
    label_hits: dict[int, Counter[str]] = {k: Counter() for k in ks}
    k_lookup = {k: {"tp": 0, "gold": 0, "hits": 0, "recalls": []} for k in ks}
    claim_count = 0

    for claim_id, claim in claims.items():
        gold = set(claim.get("evidences", []))
        if not gold:
            continue
        claim_count += 1
        label = str(claim.get("claim_label", "UNLABELED")).lower()
        label_totals[label] += 1
        predicted_ids = [candidate["evidence_id"] for candidate in pool.get(claim_id, [])]
        for k in ks:
            predicted = set(predicted_ids[:k])
            tp = len(gold & predicted)
            k_lookup[k]["tp"] += tp
            k_lookup[k]["gold"] += len(gold)
            recall = tp / len(gold) if gold else 0.0
            k_lookup[k]["recalls"].append(recall)
            if tp > 0:
                k_lookup[k]["hits"] += 1
                label_hits[k][label] += 1

    rows: dict[str, Any] = {
        "claims_with_evidence": claim_count,
        "evaluated_ks": ks,
    }
    for k in ks:
        recalls = k_lookup[k]["recalls"]
        rows[f"macro_recall_at_{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        total_tp = float(k_lookup[k]["tp"])
        total_gold = float(k_lookup[k]["gold"])
        rows[f"micro_recall_at_{k}"] = total_tp / total_gold if total_gold else 0.0
        rows[f"hit_any_at_{k}"] = k_lookup[k]["hits"] / claim_count if claim_count else 0.0
        for label in sorted(label_totals):
            total = label_totals[label]
            hit = label_hits[k][label]
            rows[f"{label}_hit_any_at_{k}"] = hit / total if total else 0.0
    rows["avg_candidate_counts"] = (
        sum(len(cands) for cands in pool.values()) / claim_count if claim_count else 0.0
    )
    rows["union_candidates"] = len({cand["evidence_id"] for candidates in pool.values() for cand in candidates})
    return rows


def infer_split_name(claims_path: Path) -> str:
    if str(claims_path) == "data/train-claims.json":
        return "train"
    if str(claims_path) == "data/dev-claims.json":
        return "dev"
    if str(claims_path) == "data/test-claims-unlabelled.json":
        return "test"
    return "claims"


def input_records(claims_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    inferred_split = infer_split_name(claims_path)
    return [
        {
            "path": str(claims_path),
            "sha256": sha256_file(claims_path),
            "split": inferred_split,
            "labels_used": inferred_split in {"train", "dev"},
        },
        {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "split": "evidence",
            "labels_used": False,
        },
    ]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    if not args.claims.exists():
        raise SystemExit(f"Missing claims file: {args.claims}")
    if not args.evidence.exists():
        raise SystemExit(f"Missing evidence file: {args.evidence}")
    if args.smoke:
        if args.max_claims <= 0:
            args.max_claims = args.smoke_claims
        if args.max_evidence <= 0:
            args.max_evidence = args.smoke_evidence

    eval_k = parse_k_list(args.eval_k)
    max_eval_k = max(eval_k)
    candidate_k = max(args.candidate_k, max_eval_k)

    claims = load_json(args.claims)
    evidence = load_json(args.evidence)
    if not isinstance(claims, dict) or not isinstance(evidence, dict):
        raise SystemExit("Claims and evidence inputs must be dictionaries keyed by id.")

    claims = limit_dict(claims, args.max_claims)
    evidence = limit_dict(evidence, args.max_evidence)

    route_weights = {
        "baseline": args.route_baseline_weight,
        "entity": args.route_entity_weight,
        "number": args.route_number_weight,
        "date": args.route_date_weight,
    }

    route_claims, route_specs, route_stats = build_route_claim_sets(claims)
    route_pools = collect_route_pools(
        route_to_claims=route_claims,
        evidence=evidence,
        top_k=candidate_k,
        k1=args.k1,
        b=args.b,
        max_features=args.max_features,
    )
    merged_pool = merge_route_pools(
        claims=claims,
        route_pools=route_pools,
        route_weights=route_weights,
        rrf_k=args.rrf_k,
    )
    # Enforce final truncation after merge.
    for claim_id in list(merged_pool.keys()):
        merged_pool[claim_id] = merged_pool[claim_id][:candidate_k]

    route_profile: dict[str, Any] = {}
    for claim_id, claim in claims.items():
        decomposed = decompose_claim(str(claim.get("claim_text", "")))
        route_profile[claim_id] = {
            "claim_text": claim.get("claim_text", ""),
            "routes": {
                route: bool(values)
                for route, values in decomposed.items()
            },
            "selected_routes": [route for route, values in decomposed.items() if values],
            "decomposed_queries": decomposed,
            "evidence_count": len(merged_pool.get(claim_id, [])),
            "top_routes": [route for route in ROUTE_IDS if route in route_specs and claim_id in route_specs[route]],
        }

    metrics = evaluate_recall(claims=claims, pool=merged_pool, ks=eval_k)
    route_coverage = {route: route_stats.get(route, 0) for route in ROUTE_IDS}
    metrics["route_coverage_claims"] = route_coverage
    metrics["route_weights"] = route_weights
    metrics["rrf_k"] = args.rrf_k
    metrics["candidate_k"] = candidate_k
    metrics["claims_input_count"] = len(claims)
    metrics["evidence_input_count"] = len(evidence)
    metrics["max_claims"] = args.max_claims
    metrics["max_evidence"] = args.max_evidence
    metrics["smoke"] = args.smoke

    for route, claims_with_route in route_stats.items():
        metrics[f"{route}_route_claims"] = claims_with_route

    split_name = infer_split_name(args.claims)
    forbidden_hits = find_forbidden_tokens(
        [str(args.claims), str(args.evidence), str(args.output_dir), str(args.manifest)]
    )
    split_file_suffix = split_name if split_name != "claims" else "claims"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = (
        args.output_dir
        / f"{split_file_suffix}_{args.run_id}_decomposed_candidates.json"
    )
    profile_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_decomposition_profile.json"
    metrics_path = args.output_dir / f"{split_file_suffix}_{args.run_id}_metrics.json"
    output_files = [str(candidate_path), str(profile_path), str(metrics_path)]

    write_json(candidate_path, merged_pool)
    write_json(profile_path, route_profile)
    write_json(metrics_path, metrics)

    command = (
        "python round18/experiments/o_sparse/o_s2_structured/run_o_s2_structured.py "
        f"--claims {args.claims} --evidence {args.evidence} --candidate-k {candidate_k} --eval-k "
        f"{','.join(map(str, eval_k))} --max-features {args.max_features} --k1 {args.k1} --b {args.b}"
    )
    if args.smoke:
        command += f" --smoke --smoke-claims {args.max_claims} --smoke-evidence {args.max_evidence}"
    if args.max_claims:
        command += f" --max-claims {args.max_claims}"
    if args.max_evidence:
        command += f" --max-evidence {args.max_evidence}"

    run_record = {
        "run_id": args.run_id,
        "stage": args.stage,
        "mode": "STRICT",
        "command": command,
        "command_args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "split": split_name,
        "smoke": args.smoke,
        "decomposition_routes": ROUTE_IDS,
        "route_weights": route_weights,
        "forbidden_hits": forbidden_hits,
        "split_file_suffix": split_file_suffix,
        "files_written": output_files + [str(args.manifest), str(args.record_path)],
        "route_stats": route_stats,
    }
    if args.max_claims > 0:
        run_record["subset_claim_count"] = len(claims)
    if args.max_evidence > 0:
        run_record["subset_evidence_count"] = len(evidence)
    write_json(args.record_path, run_record)
    output_files.append(str(args.record_path))

    manifest = manifest_base(
        run_id=args.run_id,
        status="strict-candidate",
        mode="STRICT",
        stage=args.stage,
        command=command,
        working_directory=Path.cwd(),
        config_path="",
        config_hash="",
        random_seed=args.random_seed,
        cv_seed=None,
    )
    manifest["input_files"] = input_records(args.claims, args.evidence)
    manifest["forbidden_input_scan"] = {
        "passed": len(forbidden_hits) == 0,
        "notes": "Input path scan against forbidden strict tokens passed."
        if len(forbidden_hits) == 0
        else str(forbidden_hits),
    }
    manifest["output_files"] = output_files + [str(args.manifest)]
    manifest["metrics"] = {
        "mode": "route_decomposition_sparse",
        "recall": metrics,
        "candidate_top_k": candidate_k,
        "route_weights": route_weights,
        "route_coverage": route_coverage,
    }
    manifest["runtime"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    manifest["runtime"]["device"] = "cpu"
    manifest["data_flow_summary"] = (
        "Parsed raw claim/evidence JSON, extracted entity/number/date route queries by regex, "
        "built BM25 pools per route, route-fused candidates with weighted RRF, and wrote metrics."
    )
    manifest["split_isolation_summary"] = (
        "Train labels are used only for optional recall readout. "
        "Dev labels may be used for diagnostic metrics but not for route selection."
    )
    manifest["notes"] = (
        "Prototype only: route weights are fixed, and recall curves are computed for reporting."
    )
    if args.smoke:
        manifest["notes"] += (
            " Smoke mode capped claims/evidence for runtime control."
        )
        manifest["leakage_risk"] = "low"
        manifest["reproducibility_risk"] = "low"
    write_json(args.manifest, manifest)
    output_files.append(str(args.manifest))

    print(f"Wrote candidate pool: {candidate_path}")
    print(f"Wrote decomposition profile: {profile_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote run record: {args.record_path}")
    print(f"Wrote manifest: {args.manifest}")
    for key in sorted(metrics.keys()):
        if key.startswith("macro_recall_at_") or key.startswith("micro_recall_at_") or key.startswith("hit_any_at_"):
            print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
