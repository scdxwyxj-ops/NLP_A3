from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import load_json, write_json


def prefix_for(path: str) -> str:
    parts = path.split("/")
    if not parts:
        return path
    if parts[0] in {"experiments", "src", "agent_docs", "docs", "notebooks", "outputs", "configs"}:
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Round18 static suspicious-token audit.")
    parser.add_argument("--audit", type=Path, default=Path("round18/reports/batch1/static_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("round18/reports/batch1/static_audit_summary.json"))
    args = parser.parse_args()

    audit = load_json(args.audit)
    token_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    top_files: list[dict] = []

    for path, hits in audit.items():
        prefix_counts[prefix_for(path)] += 1
        for hit in hits:
            token_counts[hit["token"]] += 1
        top_files.append(
            {
                "path": path,
                "hit_count": len(hits),
                "tokens": sorted({hit["token"] for hit in hits}),
            }
        )

    top_files.sort(key=lambda item: (-item["hit_count"], item["path"]))
    summary = {
        "files_with_hits": len(audit),
        "total_hits": sum(len(hits) for hits in audit.values()),
        "token_counts": dict(token_counts.most_common()),
        "prefix_counts": dict(prefix_counts.most_common()),
        "top_files": top_files[:30],
        "interpretation": (
            "This is a repository-wide risk map. Hits in historical rounds and documentation are expected; "
            "strict Round18 outputs must still pass per-run guard scans."
        ),
    }
    write_json(args.out, summary)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

