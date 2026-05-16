from __future__ import annotations

import argparse
from pathlib import Path

from .common import FORBIDDEN_TOKENS, write_json


DEFAULT_EXCLUDES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "round18/outputs",
    "round18/manifests",
}


def is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    parts = set(rel.split("/"))
    if parts & DEFAULT_EXCLUDES:
        return True
    return any(rel.startswith(prefix + "/") for prefix in DEFAULT_EXCLUDES if "/" in prefix)


def scan_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    hits = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        lowered = line.lower()
        for token in FORBIDDEN_TOKENS:
            if token.lower() in lowered:
                hits.append({"line": idx, "token": token, "text": line.strip()[:300]})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Round18 static suspicious-token audit.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("round18/reports/batch1/static_audit.json"))
    parser.add_argument("--include-round18-contracts", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    report: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_excluded(path, root):
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("round18/") and not args.include_round18_contracts:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        hits = scan_file(path)
        if hits:
            report[rel] = hits

    write_json(args.out, report)
    print(f"Scanned {root}")
    print(f"Files with suspicious tokens: {len(report)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

