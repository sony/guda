#!/usr/bin/env python3
"""
Extract a compact paper-style ranking table from advanced_ranking_metrics.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = {
    "Top-1": "top1_overlap_mean",
    "MRR": "mrr_mean",
    "NDCG@3": "ndcg@3_mean",
    "Top-3": "top3_overlap_mean",
    "RBO": "rbo_mean",
    "Spearman": "spearman_score_mean",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract compact ranking table")
    p.add_argument("--metrics_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--reference_label", type=str, default=None, help="Optional reference row label to prepend")
    p.add_argument("--method_order", nargs="*", default=None, help="Optional ordered list of method labels")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.metrics_csv)
    if "Method" not in df.columns:
        raise ValueError(f"'Method' column not found in {args.metrics_csv}")

    missing = [column for column in METRIC_COLUMNS.values() if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns: {missing}")

    table = df[["Method", *METRIC_COLUMNS.values()]].copy()
    table = table.rename(columns={value: key for key, value in METRIC_COLUMNS.items()})

    if args.method_order:
        rank = {name: idx for idx, name in enumerate(args.method_order)}
        table["_order"] = table["Method"].map(rank).fillna(len(rank))
        table = table.sort_values(["_order", "Method"]).drop(columns="_order")

    if args.reference_label:
        reference_row = pd.DataFrame(
            [{"Method": args.reference_label, **{name: "" for name in METRIC_COLUMNS}}]
        )
        table = pd.concat([reference_row, table], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"Saved compact ranking table to {args.output_csv}")


if __name__ == "__main__":
    main()
