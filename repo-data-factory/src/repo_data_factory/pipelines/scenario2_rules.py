# src/repo_data_factory/pipelines/scenario2_rules.py
from __future__ import annotations

import argparse

from ..rules.scenario2 import run_rules


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Scenario 2 draft dataset (rules-only).")
    parser.add_argument("--repo_dir", type=str, required=True, help="Path to the target repository root.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL path for Scenario2 drafts.")

    # Optional knobs (keep defaults aligned with rules/scenario2.py)
    parser.add_argument("--max_items", type=int, default=400)
    parser.add_argument("--max_readme_reqs", type=int, default=30)
    parser.add_argument("--max_cli_reqs", type=int, default=30)
    parser.add_argument("--max_test_reqs", type=int, default=30)
    parser.add_argument("--top_hubs", type=int, default=8)
    parser.add_argument("--max_ext_points", type=int, default=20)
    parser.add_argument("--max_flow_snips", type=int, default=20)

    args = parser.parse_args()

    n = run_rules(
        repo_dir=args.repo_dir,
        out_jsonl=args.out_jsonl,
        max_items=args.max_items,
        max_readme_reqs=args.max_readme_reqs,
        max_cli_reqs=args.max_cli_reqs,
        max_test_reqs=args.max_test_reqs,
        top_hubs=args.top_hubs,
        max_ext_points=args.max_ext_points,
        max_flow_snips=args.max_flow_snips,
    )

    print(f"[PIPE] scenario2 rules done. wrote={n} -> {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())