from __future__ import annotations

import argparse
from pathlib import Path

from .common import find_forbidden_tokens, load_json


def collect_values(obj) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.append(str(key))
            values.extend(collect_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(collect_values(value))
    elif obj is not None:
        values.append(str(obj))
    return values


def scan_paths(paths: list[Path]) -> dict[str, list[str]]:
    values: list[str] = []
    for path in paths:
        values.append(str(path))
        if path.suffix.lower() == ".json" and path.exists():
            values.extend(collect_values(load_json(path)))
        elif path.exists() and path.is_file() and path.stat().st_size <= 2_000_000:
            try:
                values.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return find_forbidden_tokens(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round18 strict-mode forbidden input guard.")
    parser.add_argument("paths", nargs="*", type=Path, help="Config, manifest, or path strings to scan.")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    if not args.paths and not args.allow_empty:
        raise SystemExit("No paths supplied. Use --allow-empty for an explicit empty scan.")

    hits = scan_paths(args.paths)
    if hits:
        print("STRICT GUARD FAILED")
        for token, values in sorted(hits.items()):
            print(f"- {token}: {len(values)} hit(s)")
            for value in values[:5]:
                print(f"  {value}")
        raise SystemExit(2)

    print("STRICT GUARD PASSED")


if __name__ == "__main__":
    main()

