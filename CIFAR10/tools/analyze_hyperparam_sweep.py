#!/usr/bin/env python3
"""
Analyze hyperparameter sweep results by comparing merged CSVs against LOGOA.

Outputs summary CSVs with nDCG and top-k overlap metrics per config/epoch.
"""
from pathlib import Path
import csv
import sys
import numpy as np
import csv as csvlib
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ranking_metrics import compute_all_metrics, aggregate_metrics


REFERENCE = Path(
    "results/cifar10_results_2048/delta_elbo_leave_one_out/"
    "timesteps4000_skip10_min1_epoch_2400/delta_elbo_leave_one_out_sorted.csv"
)
EPOCHS = ["0010", "0020", "0030", "0040", "0050"]
COMMON_CLASSES = list(range(10))


def build_dense_vector(scores_dict, class_ids, label):
    vec = []
    missing = []
    for cid in class_ids:
        if cid not in scores_dict:
            missing.append(cid)
        else:
            vec.append(scores_dict[cid])
    if missing:
        raise ValueError(f"{label}: missing {missing}")
    return np.array(vec, dtype=float)


def spearman_correlation(x, y):
    if len(x) < 2:
        return 0.0
    res = spearmanr(x, y)
    corr = float(res.correlation) if res.correlation is not None else 0.0
    if np.isnan(corr):
        return 0.0
    return corr


def load_csv_data_with_scores(csv_path: str):
    with open(csv_path, 'r') as f:
        csv_reader = csvlib.reader(f)
        headers = next(csv_reader)
        headers = [h.strip() for h in headers]

        class_cols_indices = [i for i, col in enumerate(headers) if col.strip().lower().endswith('_cls')]
        score_cols_indices = [i for i, col in enumerate(headers) if col.strip().lower().endswith('_sc')]

        baseline_format = False
        class_id_headers = {}
        if not class_cols_indices or not score_cols_indices:
            for idx, header in enumerate(headers[1:], start=1):
                try:
                    class_id = int(header)
                    class_id_headers[idx] = class_id
                    baseline_format = True
                except ValueError:
                    pass

        data = []
        for row in csv_reader:
            ranking = {}
            scores = {}

            if baseline_format:
                class_scores = []
                for col_idx, class_id in class_id_headers.items():
                    if col_idx < len(row) and row[col_idx]:
                        try:
                            score_value = float(str(row[col_idx]).strip())
                            class_scores.append((class_id, score_value))
                            scores[class_id] = score_value
                        except (ValueError, IndexError):
                            continue
                class_scores.sort(key=lambda x: x[1], reverse=True)
                for rank_position, (class_id, _) in enumerate(class_scores, 1):
                    ranking[class_id] = rank_position

            elif class_cols_indices and score_cols_indices:
                for rank_position, (cls_idx, score_idx) in enumerate(zip(class_cols_indices, score_cols_indices), 1):
                    if cls_idx < len(row) and score_idx < len(row) and row[cls_idx]:
                        try:
                            class_id = int(str(row[cls_idx]).strip())
                            score_value = float(str(row[score_idx]).strip())
                            ranking[class_id] = rank_position
                            scores[class_id] = score_value
                        except (ValueError, IndexError):
                            continue
            else:
                idx = 1
                rank_position = 1
                n_cols = len(row)
                while idx + 2 < n_cols:
                    cls_raw = str(row[idx]).strip() if row[idx] is not None else ''
                    sc_raw = str(row[idx + 2]).strip() if row[idx + 2] is not None else ''
                    if cls_raw == '' and sc_raw == '':
                        idx += 1
                        continue
                    try:
                        class_id = int(cls_raw)
                        score_value = float(sc_raw)
                        ranking[class_id] = rank_position
                        scores[class_id] = score_value
                        rank_position += 1
                        idx += 3
                    except ValueError:
                        idx += 1
                        continue
            data.append((ranking, scores))
    return data


def evaluate_csv(reference_data, test_csv):
    test_data = load_csv_data_with_scores(str(test_csv))
    if len(reference_data) != len(test_data):
        raise ValueError(f"Row count mismatch: ref={len(reference_data)} test={len(test_data)}")

    image_metrics = []
    simple_spearman = []
    topk_overlaps = {1: [], 3: []}

    for img_idx in range(len(reference_data)):
        _, ref_scores = reference_data[img_idx]
        _, test_scores = test_data[img_idx]
        try:
            g = build_dense_vector(ref_scores, COMMON_CLASSES, f"ref_{img_idx}")
            s = build_dense_vector(test_scores, COMMON_CLASSES, f"test_{img_idx}")

            metrics = compute_all_metrics(s=s, g=g, k_values=[3, 5, 10], wpra_m=3, rbo_p=0.9)
            image_metrics.append(metrics)

            simple_spearman.append(spearman_correlation(s, g))

            ref_order = np.argsort(-g)
            test_order = np.argsort(-s)
            for k in topk_overlaps.keys():
                ref_topk = set(ref_order[:k].tolist())
                test_topk = set(test_order[:k].tolist())
                overlap = len(ref_topk.intersection(test_topk)) / float(k)
                topk_overlaps[k].append(overlap)
        except Exception:
            continue

    if not image_metrics:
        return None

    aggregated = aggregate_metrics(image_metrics)
    if simple_spearman:
        aggregated["spearman_score_mean"] = float(np.mean(simple_spearman))
    for k, vals in topk_overlaps.items():
        if vals:
            aggregated[f"top{k}_overlap_mean"] = float(np.mean(vals))
    return aggregated


def sweep(method: str, output_dir: Path) -> Path:
    sweep_dir = Path("outputs/hyperparam_sweep") / f"{method}_cifar10"
    out_csv = output_dir / f"{method}_logoa_metrics.csv"

    reference_data = load_csv_data_with_scores(str(REFERENCE))
    rows = []

    config_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("lambda")]
    for config_dir in sorted(config_dirs):
        for epoch in EPOCHS:
            merged = config_dir / "merged" / f"delta_elbo_epoch_{epoch}_merged_sorted.csv"
            if not merged.exists():
                continue
            metrics = evaluate_csv(reference_data, merged)
            if metrics is None:
                continue
            row = {
                "method": method,
                "config": config_dir.name,
                "epoch": epoch,
                "ndcg@3_mean": metrics.get("ndcg@3_mean", 0.0),
                "ndcg@5_mean": metrics.get("ndcg@5_mean", 0.0),
                "ndcg@10_mean": metrics.get("ndcg@10_mean", 0.0),
                "mrr_mean": metrics.get("mrr_mean", 0.0),
                "top1_overlap_mean": metrics.get("top1_overlap_mean", 0.0),
                "top3_overlap_mean": metrics.get("top3_overlap_mean", 0.0),
                "rbo_mean": metrics.get("rbo_mean", 0.0),
                "spearman_score_mean": metrics.get("spearman_score_mean", 0.0),
            }
            row["composite_score"] = (
                row["ndcg@3_mean"] + row["top1_overlap_mean"] + row["top3_overlap_mean"]
            )
            rows.append(row)

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return out_csv


def main():
    if not REFERENCE.exists():
        raise SystemExit(f"Reference CSV not found: {REFERENCE}")

    output_dir = Path("results/hyperparam_sweep/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    retrack_csv = sweep("retrack", output_dir)
    esd_csv = sweep("esd", output_dir)

    print(f"Saved: {retrack_csv}")
    print(f"Saved: {esd_csv}")


if __name__ == "__main__":
    main()
