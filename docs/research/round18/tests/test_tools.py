from pathlib import Path

from round18.tools.common import find_forbidden_tokens, sha256_text
from round18.tools.strict_guard import collect_values


def test_forbidden_token_scan_detects_historical_outputs():
    hits = find_forbidden_tokens(["outputs/round16/foo.json", "data/train-claims.json"])
    assert "outputs/round" in hits


def test_collect_values_recurses_json_like_objects():
    values = collect_values({"a": ["b", {"c": "d"}]})
    assert {"a", "b", "c", "d"}.issubset(set(values))


def test_sha256_text_is_stable():
    assert sha256_text("round18") == sha256_text("round18")
    assert sha256_text("round18") != sha256_text("round17")


def test_round18_directory_exists():
    assert Path("round18").exists()

