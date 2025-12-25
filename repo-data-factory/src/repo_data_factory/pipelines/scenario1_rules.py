from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from ..rules.scenario1 import run_rules


def write_jsonl(items, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scenario 1: run deterministic rules to generate draft Q&A samples.")
    parser.add_argument("--repo_dir", type=str, required=True, help="Local repository root directory.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL path for drafts.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    out_jsonl = Path(args.out_jsonl).resolve()

    items = run_rules(repo_dir)
    write_jsonl(items, out_jsonl)

    print(f"[S1_RULES] repo={repo_dir} items={len(items)} out={out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
