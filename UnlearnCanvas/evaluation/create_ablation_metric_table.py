#!/usr/bin/env python3
"""Create a compact metric comparison table for an ablation summary."""
import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    ("top1_agreement", "Top-1"),
    ("mrr_mean", "MRR"),
    ("ndcg3_mean", "NDCG@3"),
    ("top3_agreement", "Top-3"),
    ("rbo_mean", "RBO"),
    ("spearman_mean", "Spearman"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create ablation metric change table")
    parser.add_argument("--summary_csv", required=True, type=str)
    parser.add_argument("--baseline_config", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--output_md", required=True, type=str)
    parser.add_argument("--param_configs", required=False, type=str)
    parser.add_argument("--step", type=int, default=2000)
    return parser.parse_args()


def load_config_order(param_configs_path: str | None) -> list[str]:
    if not param_configs_path:
        return []

    with open(param_configs_path, "r") as f:
        config = json.load(f)

    return [item["id"] for item in config.get("param_grid", [])]


def main():
    args = parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    step_df = summary_df[summary_df["step"] == args.step].copy()

    if step_df.empty:
        raise ValueError(f"No rows found for step={args.step} in {args.summary_csv}")

    baseline_rows = step_df[step_df["config"] == args.baseline_config]
    if baseline_rows.empty:
        raise ValueError(
            f"Baseline config '{args.baseline_config}' not found at step={args.step}"
        )

    baseline = baseline_rows.iloc[0]

    order = load_config_order(args.param_configs)
    if order:
        step_df["config_order"] = step_df["config"].apply(
            lambda name: order.index(name) if name in order else len(order)
        )
        step_df = step_df.sort_values(["config_order", "config"]).drop(columns=["config_order"])
    else:
        step_df = step_df.sort_values("config")

    for metric, _label in METRIC_COLUMNS:
        step_df[f"{metric}_delta"] = step_df[metric] - baseline[metric]

    output_cols = ["config"]
    for metric, _label in METRIC_COLUMNS:
        output_cols.extend([metric, f"{metric}_delta"])

    output_df = step_df[output_cols].copy()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    md_lines = [
        f"# Ablation Metric Changes (step {args.step})",
        "",
        f"Baseline: `{args.baseline_config}`",
        "",
        "| Config | Top-1 | dTop-1 | MRR | dMRR | NDCG@3 | dNDCG@3 | Top-3 | dTop-3 | RBO | dRBO | Spearman | dSpearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in output_df.iterrows():
        md_lines.append(
            "| {config} | {top1:.4f} | {dtop1:+.4f} | {mrr:.4f} | {dmrr:+.4f} | "
            "{ndcg3:.4f} | {dndcg3:+.4f} | {top3:.4f} | {dtop3:+.4f} | "
            "{rbo:.4f} | {drbo:+.4f} | {spearman:.4f} | {dspearman:+.4f} |".format(
                config=row["config"],
                top1=row["top1_agreement"],
                dtop1=row["top1_agreement_delta"],
                mrr=row["mrr_mean"],
                dmrr=row["mrr_mean_delta"],
                ndcg3=row["ndcg3_mean"],
                dndcg3=row["ndcg3_mean_delta"],
                top3=row["top3_agreement"],
                dtop3=row["top3_agreement_delta"],
                rbo=row["rbo_mean"],
                drbo=row["rbo_mean_delta"],
                spearman=row["spearman_mean"],
                dspearman=row["spearman_mean_delta"],
            )
        )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n")

    print(f"Saved CSV: {output_csv}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
