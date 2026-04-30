from a3_factcheck.semantic import extract_semantic_features


def test_extract_semantic_features_finds_quantities_and_negation():
    features = extract_semantic_features(
        "Australia produces 1.3% of emissions and no reduction will affect climate."
    )

    assert "1.3%" in features["percentages"]
    assert "no" in features["negation_cues"]
    assert "affect" in features["causality_cues"]
    assert "produces" in features["relation_verbs"]
