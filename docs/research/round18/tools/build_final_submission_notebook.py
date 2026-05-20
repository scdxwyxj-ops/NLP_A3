from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_NOTEBOOK_PATH = ROOT / "colab_notebooks" / "Group_131_COMP90042_Project_2026.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (NLP)",
            "language": "python",
            "name": "nlp",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    }

    nb["cells"] = [
        md(
            """
            # 2026 COMP90042 Project

            This notebook is the submission driver for the selected Round18 system. It is intentionally aligned with `group_meetings/second_meeting /tutorial.ipynb` and uses the same promoted stage outputs:

            1. Candidate: four sparse leaf gates fused by fixed reciprocal-rank fusion.
            2. Top64 context: MiniLM embedding inner product plus hand-engineered shallow factual features.
            3. Top3 evidence: train-selected CE plus embedding fusion.
            4. Classifier: Context20 TF-IDF Logistic Regression, `max_features=60000`, word `1-2` grams, `C=4.0`.

            The notebook writes `final_predictions.json` and `run_summary.json`.
            """
        ),
        md(
            """
            # Readme

            Expected local files:

            - `data/dev-claims.json` or an override via `A3_TARGET_CLAIMS_PATH`
            - `data/evidence.json` or an override via `A3_EVIDENCE_PATH`
            - the Round18 selected pipeline artifacts under `round18/outputs/` and `round18/reports/`

            For an unlabelled test run, point `A3_TARGET_CLAIMS_PATH`, `A3_TOP3_POOL_PATH`, and `A3_CLASSIFIER_PREDICTIONS_PATH` at the corresponding test artifacts produced by the same four-stage pipeline.
            """
        ),
        md(
            """
            # 1.DataSet Processing
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import os
            import time
            from collections import Counter

            import matplotlib.pyplot as plt
            import numpy as np
            from IPython.display import display
            from sklearn.metrics import classification_report, confusion_matrix, f1_score

            NOTEBOOK_START = time.perf_counter()

            OUTPUT_DIR = Path(os.environ.get("A3_OUTPUT_DIR", "outputs/final_submission_aligned"))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            TARGET_CLAIMS_PATH = Path(os.environ.get("A3_TARGET_CLAIMS_PATH", "data/dev-claims.json"))
            EVIDENCE_PATH = Path(os.environ.get("A3_EVIDENCE_PATH", "data/evidence.json"))

            CANDIDATE_POOL_PATH = Path(os.environ.get(
                "A3_CANDIDATE_POOL_PATH",
                "round18/outputs/o_sparse/o_s7_plain_leaf_fusion/dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_top500_candidates.json",
            ))
            CANDIDATE_METRICS_PATH = Path(os.environ.get(
                "A3_CANDIDATE_METRICS_PATH",
                "round18/outputs/o_sparse/o_s7_plain_leaf_fusion/dev_full_dev_o_s7_plain_leaf_fusion_strict_rrf_char_heavy_metrics.json",
            ))

            TOP64_POOL_PATH = Path(os.environ.get(
                "A3_TOP64_POOL_PATH",
                "round18/reports/embedding_shallow_feature_kfold/embedding_shallow_feature_kfold_best_dev_top500_candidates.json",
            ))
            TOP64_SUMMARY_PATH = Path(os.environ.get(
                "A3_TOP64_SUMMARY_PATH",
                "round18/reports/embedding_shallow_feature_kfold/embedding_shallow_feature_kfold_summary.json",
            ))

            TOP3_POOL_PATH = Path(os.environ.get(
                "A3_TOP3_POOL_PATH",
                "round18/reports/top3_train_kfold_fusion_with_hand/top3_train_kfold_fusion_with_hand_best_dev_top500_candidates.json",
            ))
            TOP3_SUMMARY_PATH = Path(os.environ.get(
                "A3_TOP3_SUMMARY_PATH",
                "round18/reports/top3_train_kfold_fusion_with_hand/top3_train_kfold_fusion_with_hand_summary.json",
            ))

            CLASSIFIER_PREDICTIONS_PATH = Path(os.environ.get(
                "A3_CLASSIFIER_PREDICTIONS_PATH",
                "round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_dev_predictions.json",
            ))
            CLASSIFIER_METRICS_PATH = Path(os.environ.get(
                "A3_CLASSIFIER_METRICS_PATH",
                "round18/outputs/o_classifier/o_c4_ce_factual_context_k20_trainholdout/o_c4_ce_factual_context_k20_trainholdout_tfidf_logreg_mf60000_ngram1x2_c4p0_metrics.json",
            ))

            SUBMISSION_EVIDENCE_K = 3
            LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
            LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

            def load_json(path):
                path = require(Path(path))
                with path.open(encoding="utf-8") as f:
                    return json.load(f)

            def write_json(path, payload):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

            def require(path):
                candidates = [
                    path,
                    Path.cwd() / path,
                    Path.cwd().parent / path,
                    Path.cwd().parent.parent / path,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                raise FileNotFoundError(f"Missing required aligned pipeline artifact: {path}")

            def has_gold(claims):
                return all("claim_label" in claim and "evidences" in claim for claim in claims.values())

            def top_evidence(pool, claim_id, k):
                rows = pool.get(claim_id, [])
                result = []
                seen = set()
                for row in rows:
                    eid = str(row.get("evidence_id", ""))
                    if eid and eid not in seen:
                        result.append(eid)
                        seen.add(eid)
                    if len(result) >= k:
                        break
                return result

            def evidence_f_score(gold, predicted):
                gold = set(gold)
                predicted = list(predicted)
                if not gold or not predicted:
                    return 0.0
                correct = len(gold & set(predicted))
                if correct == 0:
                    return 0.0
                precision = correct / len(predicted)
                recall = correct / len(gold)
                return 2 * precision * recall / (precision + recall)

            print("TARGET_CLAIMS_PATH =", TARGET_CLAIMS_PATH)
            print("OUTPUT_DIR =", OUTPUT_DIR)
            """
        ),
        code(
            """
            target_claims = load_json(TARGET_CLAIMS_PATH)
            evidence = load_json(EVIDENCE_PATH)
            candidate_pool = load_json(CANDIDATE_POOL_PATH)
            candidate_metrics = load_json(CANDIDATE_METRICS_PATH)
            top64_pool = load_json(TOP64_POOL_PATH)
            top64_summary = load_json(TOP64_SUMMARY_PATH)
            top3_pool = load_json(TOP3_POOL_PATH)
            top3_summary = load_json(TOP3_SUMMARY_PATH)
            classifier_predictions = load_json(CLASSIFIER_PREDICTIONS_PATH)
            classifier_metrics = load_json(CLASSIFIER_METRICS_PATH)

            labelled_target = has_gold(target_claims)
            for name, pool in [
                ("candidate", candidate_pool),
                ("top64", top64_pool),
                ("top3", top3_pool),
                ("classifier", classifier_predictions),
            ]:
                missing = sorted(set(target_claims) - set(pool))
                if missing:
                    raise ValueError(f"{name} artifact does not cover target claims. First missing ids: {missing[:5]}")

            display([
                {"item": "target claims", "count": len(target_claims), "labelled": labelled_target},
                {"item": "evidence corpus", "count": len(evidence)},
                {"item": "candidate pool claims", "count": len(candidate_pool)},
                {"item": "top64 pool claims", "count": len(top64_pool)},
                {"item": "top3 pool claims", "count": len(top3_pool)},
                {"item": "classifier prediction claims", "count": len(classifier_predictions)},
            ])
            """
        ),
        md(
            """
            # 2.Model Implementation
            """
        ),
        md(
            """
            ## Candidate Stage: Four Sparse Gates

            This is the tutorial candidate stage: BM25 word gate, character n-gram gate, structured cue gate, and query-expansion/PRF gate fused with fixed reciprocal-rank fusion. The promoted artifact is `strict_rrf_char_heavy` top500.
            """
        ),
        code(
            """
            candidate_summary = {
                "method": "four sparse gate fixed RRF fusion",
                "sources": ["BM25 word", "character n-gram", "structured cue", "query expansion / PRF"],
                "weights": {"BM25 word": 1.0, "character n-gram": 1.5, "structured cue": 1.0, "query expansion / PRF": 1.0},
                "candidate_k": 500,
                "macro_recall@100": candidate_metrics.get("macro_recall_at_100") or candidate_metrics.get("macro_recall@100"),
                "macro_recall@500": candidate_metrics.get("macro_recall_at_500") or candidate_metrics.get("macro_recall@500"),
                "micro_recall@500": candidate_metrics.get("micro_recall_at_500") or candidate_metrics.get("micro_recall@500"),
                "artifact": str(CANDIDATE_POOL_PATH),
            }
            display([candidate_summary])
            """
        ),
        md(
            """
            ## Top64 Context Stage: Embedding Plus Hand Features

            This is the tutorial top64 context selector. It combines MiniLM embedding inner-product signal with shallow factual hand features, selected by train k-fold evaluation.
            """
        ),
        code(
            """
            top64_best = top64_summary["best_variant"]
            top64_summary_row = {
                "method": "embedding inner product + shallow factual hand features",
                "variant": top64_best.get("variant"),
                "selection_basis": top64_best.get("selection_basis"),
                "candidate_k": top64_best.get("candidate_k"),
                "feature_count": top64_best.get("feature_count"),
                "dev_macro_recall@64": top64_best.get("dev_macro_recall@64"),
                "dev_hit_any@64": top64_best.get("dev_hit_any@64"),
                "artifact": str(TOP64_POOL_PATH),
            }
            display([top64_summary_row])
            """
        ),
        md(
            """
            ## Top3 Evidence Stage: CE Plus Embedding Fusion

            This is the tutorial top3 stage. The selected train-kfold fusion combines CE rank/score and embedding score. It is not the BGE-small fallback and not the archived Round16 multisource shortcut.
            """
        ),
        code(
            """
            top3_best = top3_summary["best"]
            top3_summary_row = {
                "method": "train-selected CE + embedding top3 fusion",
                "selection_basis": top3_best.get("selection_basis"),
                "status": top3_best.get("status"),
                "weights": top3_summary.get("best_weights"),
                "dev_macro_recall@3": top3_best.get("dev_macro_recall@3"),
                "dev_evidence_f@3": top3_best.get("dev_evidence_f@3"),
                "dev_hit_any@3": top3_best.get("dev_hit_any@3"),
                "artifact": str(TOP3_POOL_PATH),
            }
            display([top3_summary_row])
            """
        ),
        md(
            """
            ## Classifier Stage: Context20 TF-IDF Logistic Regression

            This is the tutorial classifier: Context20 evidence text, TF-IDF word `1-2` grams, `max_features=60000`, Logistic Regression with `C=4.0`.
            """
        ),
        code(
            """
            classifier_summary = {
                "method": "Context20 TF-IDF Logistic Regression",
                "context_k": 20,
                "tfidf_max_features": classifier_metrics.get("config", {}).get("tfidf_max_features"),
                "ngram_range": classifier_metrics.get("config", {}).get("tfidf_ngram_range"),
                "C": classifier_metrics.get("config", {}).get("C"),
                "accuracy": classifier_metrics.get("accuracy"),
                "macro_f1": classifier_metrics.get("macro_f1"),
                "macro_recall": classifier_metrics.get("macro_recall"),
                "prediction_histogram": classifier_metrics.get("prediction_histogram"),
                "artifact": str(CLASSIFIER_PREDICTIONS_PATH),
            }
            display([classifier_summary])

            if "confusion_matrix" in classifier_metrics:
                conf = np.asarray(classifier_metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(5.4, 4.6))
                im = ax.imshow(conf, cmap="Blues")
                ax.set_xticks(np.arange(len(LABELS)))
                ax.set_yticks(np.arange(len(LABELS)))
                ax.set_xticklabels(LABELS, rotation=25, ha="right")
                ax.set_yticklabels(LABELS)
                ax.set_xlabel("predicted label")
                ax.set_ylabel("gold label")
                ax.set_title("Context20 logistic confusion matrix")
                for i in range(conf.shape[0]):
                    for j in range(conf.shape[1]):
                        ax.text(j, i, int(conf[i, j]), ha="center", va="center", color="white" if conf[i, j] > conf.max() * 0.55 else "#222222")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                plt.show()
            """
        ),
        md(
            """
            # 3.Testing and Evaluation
            """
        ),
        code(
            """
            final_predictions = {}
            for claim_id in target_claims:
                classifier_row = classifier_predictions[claim_id]
                label = classifier_row.get("claim_label") or classifier_row.get("pred_class")
                final_predictions[claim_id] = {
                    "claim_label": label,
                    "evidences": top_evidence(top3_pool, claim_id, SUBMISSION_EVIDENCE_K),
                }

            write_json(OUTPUT_DIR / "final_predictions.json", final_predictions)

            final_metrics = {}
            if labelled_target:
                f_scores = [
                    evidence_f_score(target_claims[cid].get("evidences", []), final_predictions[cid].get("evidences", []))
                    for cid in target_claims
                ]
                y_true = [LABEL_TO_ID[target_claims[cid]["claim_label"]] for cid in target_claims]
                y_pred = [LABEL_TO_ID[final_predictions[cid]["claim_label"]] for cid in target_claims]
                assignment_f = float(np.mean(f_scores)) if f_scores else 0.0
                accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
                harmonic = 0.0 if assignment_f + accuracy == 0 else 2 * assignment_f * accuracy / (assignment_f + accuracy)
                final_metrics = {
                    "evidence_f": assignment_f,
                    "accuracy": accuracy,
                    "harmonic": harmonic,
                    "top3_macro_recall": top3_best.get("dev_macro_recall@3"),
                    "classification_macro_f1": f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0),
                }
                print("FINAL METRICS")
                print(json.dumps(final_metrics, indent=2))
                print("CLASSIFICATION REPORT")
                print(classification_report(y_true, y_pred, target_names=LABELS, labels=list(range(len(LABELS))), zero_division=0))
                print("CONFUSION MATRIX")
                print(confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))))
            else:
                print("Target claims are unlabelled; evaluation skipped.")

            label_counts = Counter(row["claim_label"] for row in final_predictions.values())
            run_summary = {
                "mode": "tutorial-aligned artifact-backed final pipeline",
                "target_claims_path": str(TARGET_CLAIMS_PATH),
                "output_dir": str(OUTPUT_DIR),
                "candidate_stage": candidate_summary,
                "top64_stage": top64_summary_row,
                "top3_stage": top3_summary_row,
                "classifier_stage": classifier_summary,
                "final_metrics": final_metrics,
                "label_counts": dict(label_counts),
                "total_wall_seconds": time.perf_counter() - NOTEBOOK_START,
            }
            write_json(OUTPUT_DIR / "run_summary.json", run_summary)
            print("Wrote", OUTPUT_DIR / "final_predictions.json")
            print("Wrote", OUTPUT_DIR / "run_summary.json")
            """
        ),
        md(
            """
            ## Object Oriented Programming codes here

            The selected pipeline is artifact-backed for submission consistency. The heavy training and reranking objects are defined in the Round18 experiment scripts that produced the selected artifacts; this notebook keeps the final prediction driver small and auditable.
            """
        ),
    ]
    return nb


def main() -> None:
    raise SystemExit(
        "This legacy artifact-backed notebook generator is disabled. "
        f"The canonical self-contained submission notebook is {CANONICAL_NOTEBOOK_PATH}. "
        "Do not regenerate notebooks/ or submissions/COMP90042_Group_131_resource/ from this script."
    )


if __name__ == "__main__":
    main()
