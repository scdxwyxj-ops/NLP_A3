from round18.experiments.o_sparse.o_s10_wide_hand_feature.run_o_s10_wide_hand_feature import (
    evaluate_recall_at_k,
    parse_k_list,
)


def test_parse_k_list_sorts_and_deduplicates():
    assert parse_k_list("500,64,64,3") == [3, 64, 500]


def test_evaluate_recall_at_k_reports_macro_and_hits():
    claims = {
        "c1": {"claim_label": "SUPPORTS", "evidences": ["e1", "e2"]},
        "c2": {"claim_label": "REFUTES", "evidences": ["e9"]},
    }
    ranked = {
        "c1": [{"evidence_id": "e1"}, {"evidence_id": "x"}],
        "c2": [{"evidence_id": "y"}, {"evidence_id": "e9"}],
    }
    metrics = evaluate_recall_at_k(claims, ranked, [1, 2])
    assert metrics["macro_recall_at_1"] == 0.25
    assert metrics["macro_recall_at_2"] == 0.75
    assert metrics["hit_any_at_1"] == 0.5
    assert metrics["hit_any_at_2"] == 1.0

