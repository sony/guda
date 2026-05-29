#!/usr/bin/env python3
"""
Plot comprehensive correlation analysis between multiple CSV files with attribution scores.
Shows rank correlations, metric value correlations, and metric distributions across ranks.
Compares multiple test CSV files against a single reference CSV (ground truth).
"""
import matplotlib.pyplot as plt
import csv
import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from pathlib import Path
import hashlib
from ranking_metrics import compute_all_metrics, aggregate_metrics

def spearman_correlation(x, y):
    """Wrapper around scipy.stats.spearmanr with defensive handling.

    Returns (correlation, pvalue). If inputs are too short or constant, returns (0.0, 1.0).
    """
    try:
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 1.0
        # Use scipy implementation which handles ties correctly and returns p-value
        res = spearmanr(x, y)
        corr = float(res.correlation) if res.correlation is not None else 0.0
        pval = float(res.pvalue) if getattr(res, 'pvalue', None) is not None else 1.0
        # scipy may return nan for constant input; handle that
        if np.isnan(corr):
            return 0.0, 1.0
        return corr, pval
    except Exception:
        return 0.0, 1.0

def get_args():
    parser = argparse.ArgumentParser(description="Plot comprehensive correlation analysis between multiple CSV files")
    parser.add_argument("--test_csvs", nargs='+', required=True, 
                       help="List of test CSV file paths to compare against reference")
    parser.add_argument("--reference_csv", required=True, 
                       help="Reference CSV file path (ground truth)")
    parser.add_argument("--num_images", type=int, default=None, 
                       help="Number of images to plot (default: all)")
    parser.add_argument("--start_image", type=int, default=0, 
                       help="Starting image index (default: 0)")
    parser.add_argument("--output_dir", default=None, 
                       help="Output directory (default: same as reference_csv)")
    parser.add_argument("--labels", nargs='*', default=None,
                       help="Custom labels for test CSVs (default: filenames)")
    parser.add_argument("--exp_name", default=None,
                       help="Experiment name or condition to include in output directory name (e.g., job name, config)")
    return parser.parse_args()

def load_csv_data_with_scores(csv_path):
    """Load CSV data and return both rankings and score values for each image"""
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        # Be robust to stray spaces / case in headers
        headers = [h.strip() for h in headers]
        
        # Find class and score columns (like R1_Cls, R1_Sc, etc.)
        class_cols_indices = [i for i, col in enumerate(headers) if col.strip().lower().endswith('_cls')]
        score_cols_indices = [i for i, col in enumerate(headers) if col.strip().lower().endswith('_sc')]

        # Check for baseline format: image_id,0,1,2,...,9 (class IDs as column headers)
        baseline_format = False
        class_id_headers = {}  # column_index -> class_id
        if not class_cols_indices or not score_cols_indices:
            # Try to parse headers as class IDs (skip first column which is image_id)
            for idx, header in enumerate(headers[1:], start=1):
                try:
                    class_id = int(header)
                    class_id_headers[idx] = class_id
                    baseline_format = True
                except ValueError:
                    pass
        
        if baseline_format:
            print(f"[load_csv] Info: Detected baseline format (class IDs as headers) in {os.path.basename(csv_path)}.\n"
                  f"Found {len(class_id_headers)} class columns: {sorted(class_id_headers.values())}")
        elif not class_cols_indices or not score_cols_indices:
            print(f"[load_csv] Info: No *_Cls/_Sc columns detected in {os.path.basename(csv_path)}.\n"
                  f"Headers (first 10): {headers[:10]}. Will try fallback triplet parsing (class_id, class_name, score).")
        if len(class_cols_indices) != len(score_cols_indices):
            print(f"[load_csv] Warning: Mismatched counts of class ({len(class_cols_indices)}) and score ({len(score_cols_indices)}) columns in {os.path.basename(csv_path)}. Using min-length pairing.")
        
        # Read all data and create rankings and scores
        data = []
        for row in csv_reader:
            # For each image, create ranking and score dictionaries: class_id -> rank/score
            ranking = {}
            scores = {}

            if baseline_format:
                # Baseline format: each column (after image_id) contains the score for that class
                # Column index -> class_id mapping from headers
                # Need to sort by score to determine ranks
                class_scores = []  # (class_id, score_value)
                for col_idx, class_id in class_id_headers.items():
                    if col_idx < len(row) and row[col_idx]:
                        try:
                            score_value = float(str(row[col_idx]).strip())
                            class_scores.append((class_id, score_value))
                            scores[class_id] = score_value
                        except (ValueError, IndexError):
                            continue
                
                # Sort by score (descending) to get ranks
                class_scores.sort(key=lambda x: x[1], reverse=True)
                for rank_position, (class_id, _) in enumerate(class_scores, 1):
                    ranking[class_id] = rank_position
                    
            elif class_cols_indices and score_cols_indices:
                # Standard Rk_Cls / Rk_Sc format
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
                # Fallback: NGA-style rows with repeating (class_id, class_name, score) triplets
                # Assume col0 is image path/name, then triplets start at col1
                idx = 1
                rank_position = 1
                n_cols = len(row)
                while idx + 2 < n_cols:
                    cls_raw = str(row[idx]).strip() if row[idx] is not None else ''
                    name_raw = str(row[idx + 1]).strip() if row[idx + 1] is not None else ''
                    sc_raw = str(row[idx + 2]).strip() if row[idx + 2] is not None else ''

                    # Skip empty groups
                    if cls_raw == '' and sc_raw == '':
                        idx += 1  # move forward to try to sync on next boundary
                        continue

                    try:
                        class_id = int(cls_raw)
                        score_value = float(sc_raw)
                        ranking[class_id] = rank_position
                        scores[class_id] = score_value
                        rank_position += 1
                        idx += 3  # move to next triplet
                    except ValueError:
                        # If class_id is like "52" (OK) or name cell accidentally here, try to realign by +1
                        idx += 1
                        continue
            data.append((ranking, scores))
    
    return data

def plot_multi_rank_correlation(reference_data, test_data_list, test_labels, common_classes, 
                               actual_num_images, output_dir):
    """Plot rank correlation for multiple test CSVs against reference showing mean and variance"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_data_list)))
    
    correlations = []
    max_rank = max(len(common_classes), 10)  # Determine max rank for plotting
    any_plotted = False
    
    for test_idx, (test_data, test_label) in enumerate(zip(test_data_list, test_labels)):
        # Collect rank pairs across all images and classes
        ref_ranks = []
        test_ranks = []
        
        for img_idx in range(actual_num_images):
            ref_ranking, _ = reference_data[img_idx]
            test_ranking, _ = test_data[img_idx]
            
            for class_id in common_classes:
                if class_id in ref_ranking and class_id in test_ranking:
                    ref_ranks.append(ref_ranking[class_id])
                    test_ranks.append(test_ranking[class_id])
        
        if len(ref_ranks) > 0:
            # Calculate correlation
            correlation, p_value = spearman_correlation(ref_ranks, test_ranks)
            correlations.append(correlation)
            
            # Calculate mean and std of reference ranks for each test rank
            rank_stats = defaultdict(list)
            for tr, rr in zip(test_ranks, ref_ranks):
                rank_stats[tr].append(rr)
            
            # Extract data for plotting
            test_rank_positions = sorted(rank_stats.keys())
            ref_rank_means = [np.mean(rank_stats[tr]) for tr in test_rank_positions]
            ref_rank_stds = [np.std(rank_stats[tr]) for tr in test_rank_positions]
            
            # Plot with error bars
            plt.errorbar(test_rank_positions, ref_rank_means, yerr=ref_rank_stds,
                        fmt='o-', alpha=0.7, capsize=3, capthick=1,
                        color=colors[test_idx], label=f'{test_label} (r={correlation:.3f})')
            any_plotted = True
        else:
            # Maintain alignment so downstream plots don't crash
            correlations.append(0.0)
            print(f"[rank_correlation] Warning: No overlapping ranks for '{test_label}'. Appending r=0.0")
        
        max_rank = max(max_rank, max(test_ranks) if test_ranks else max_rank)
    
    # Plot diagonal line for perfect correlation
    plt.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.5, label='Perfect correlation')
    
    plt.xlabel('Test Rank', fontsize=12)
    plt.ylabel('Reference Rank (Mean ± Std)', fontsize=12)
    plt.title(f'Rank Correlation Analysis\n({actual_num_images} images, {len(common_classes)} classes)', 
              fontsize=14)
    if any_plotted:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "multi_rank_correlation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, correlations

def plot_multi_metric_correlation_normalized(reference_data, test_data_list, test_labels, common_classes,
                                            actual_num_images, output_dir, mode: str):
    """Plot metric value correlation with specified normalization mode.
    Modes:
      - 'per_image_zscore': per-image mean/std for both ref and test
      - 'per_image_center_global_std': per-image mean centering, divide by global std (ref/test separately)
    """
    assert mode in {"per_image_zscore", "per_image_center_global_std"}

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_data_list)))

    correlations = []

    # Precompute global stds if needed
    if mode == "per_image_center_global_std":
        # Reference global std across all images/classes
        ref_all_vals = []
        for img_idx in range(actual_num_images):
            _, ref_score_dict = reference_data[img_idx]
            for class_id in common_classes:
                if class_id in ref_score_dict:
                    ref_all_vals.append(ref_score_dict[class_id])
        ref_all_vals = np.array(ref_all_vals, dtype=float) if len(ref_all_vals) > 0 else np.array([1.0])
        ref_global_std = float(np.std(ref_all_vals)) if ref_all_vals.size > 0 else 1.0
        if ref_global_std == 0.0:
            ref_global_std = 1.0

    for test_idx, (test_data, test_label) in enumerate(zip(test_data_list, test_labels)):
        # Optional: compute test global std for this method
        if mode == "per_image_center_global_std":
            test_all_vals = []
            for img_idx in range(actual_num_images):
                _, test_score_dict = test_data[img_idx]
                for class_id in common_classes:
                    if class_id in test_score_dict:
                        test_all_vals.append(test_score_dict[class_id])
            test_all_vals = np.array(test_all_vals, dtype=float) if len(test_all_vals) > 0 else np.array([1.0])
            test_global_std = float(np.std(test_all_vals)) if test_all_vals.size > 0 else 1.0
            if test_global_std == 0.0:
                test_global_std = 1.0

        ref_vals_norm = []
        test_vals_norm = []

        for img_idx in range(actual_num_images):
            ref_ranking, ref_score_dict = reference_data[img_idx]
            test_ranking, test_score_dict = test_data[img_idx]

            # Gather aligned scores for common classes present in both
            ref_scores_img = []
            test_scores_img = []
            for class_id in common_classes:
                if class_id in ref_score_dict and class_id in test_score_dict:
                    ref_scores_img.append(ref_score_dict[class_id])
                    test_scores_img.append(test_score_dict[class_id])

            if not ref_scores_img:
                continue

            ref_scores_img = np.array(ref_scores_img, dtype=float)
            test_scores_img = np.array(test_scores_img, dtype=float)

            if mode == "per_image_zscore":
                # Per-image z-score for both ref and test
                r_mean = float(np.mean(ref_scores_img))
                r_std = float(np.std(ref_scores_img))
                t_mean = float(np.mean(test_scores_img))
                t_std = float(np.std(test_scores_img))
                r_std = r_std if r_std > 0 else 1.0
                t_std = t_std if t_std > 0 else 1.0
                ref_norm = (ref_scores_img - r_mean) / r_std
                test_norm = (test_scores_img - t_mean) / t_std
            else:  # per_image_center_global_std
                r_mean = float(np.mean(ref_scores_img))
                t_mean = float(np.mean(test_scores_img))
                ref_norm = (ref_scores_img - r_mean) / ref_global_std
                test_norm = (test_scores_img - t_mean) / test_global_std

            ref_vals_norm.extend(ref_norm.tolist())
            test_vals_norm.extend(test_norm.tolist())

        # Compute Spearman correlation on normalized values
        if len(ref_vals_norm) >= 2:
            corr, _ = spearman_correlation(ref_vals_norm, test_vals_norm)
        else:
            corr = 0.0
        correlations.append(corr)

        # Scatter plot for this method
        alpha = 0.35
        plt.scatter(test_vals_norm, ref_vals_norm, alpha=alpha, s=12,
                    color=colors[test_idx], label=f'{test_label} (r={corr:.3f})', zorder=2)

        # Add regression line (least squares) if enough points
        if len(test_vals_norm) >= 2:
            x_vals = np.array(test_vals_norm, dtype=float)
            y_vals = np.array(ref_vals_norm, dtype=float)
            A = np.vstack([x_vals, np.ones(len(x_vals))]).T
            slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
            x_line = np.linspace(np.percentile(x_vals, 1), np.percentile(x_vals, 99), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, '--', color=colors[test_idx], alpha=0.9,
                     linewidth=2.5, zorder=10, label=f'{test_label} fit (slope={slope:.3f})')

    mode_title = "Per-Image Z-Score" if mode == "per_image_zscore" else "Per-Image Center + Global Std"
    plt.xlabel(f'Normalized Test Score ({mode_title})', fontsize=12)
    plt.ylabel(f'Normalized Reference Score ({mode_title})', fontsize=12)
    plt.title(f'Metric Value Correlation Analysis [{mode_title}]\n({actual_num_images} images, {len(common_classes)} classes)',
              fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = 'per_image_zscore' if mode == 'per_image_zscore' else 'per_imageCenter_globalStd'
    output_path = os.path.join(output_dir, f"multi_metric_correlation_{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path, correlations

def plot_correlation_summary(test_labels, rank_correlations, metric_correlations_normalized, output_dir):
    """Plot a summary of all correlations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_pos = np.arange(len(test_labels))
    width = 0.35

    # Align sizes defensively to avoid shape mismatch
    if len(rank_correlations) != len(test_labels):
        print(f"[summary] Warning: rank_correlations size {len(rank_correlations)} != labels {len(test_labels)}. Padding/truncating.")
        if len(rank_correlations) < len(test_labels):
            rank_correlations = list(rank_correlations) + [0.0] * (len(test_labels) - len(rank_correlations))
        else:
            rank_correlations = list(rank_correlations)[:len(test_labels)]
    if len(metric_correlations_normalized) != len(test_labels):
        print(f"[summary] Warning: metric_correlations_normalized size {len(metric_correlations_normalized)} != labels {len(test_labels)}. Padding/truncating.")
        if len(metric_correlations_normalized) < len(test_labels):
            metric_correlations_normalized = list(metric_correlations_normalized) + [0.0] * (len(test_labels) - len(metric_correlations_normalized))
        else:
            metric_correlations_normalized = list(metric_correlations_normalized)[:len(test_labels)]
    
    # Rank correlations
    bars1 = ax1.bar(x_pos, rank_correlations, width, alpha=0.8, 
                   color=plt.cm.Set1(np.linspace(0, 1, len(test_labels))))
    ax1.set_xlabel('Test CSV', fontsize=12)
    ax1.set_ylabel('Spearman Correlation', fontsize=12)
    ax1.set_title('Rank Correlations vs Reference', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(test_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-1, 1)
    
    # Add value labels on bars
    for bar, corr in zip(bars1, rank_correlations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Metric correlations (Per-Image Center + Global Std)
    bars2 = ax2.bar(x_pos, metric_correlations_normalized, width, alpha=0.8,
                   color=plt.cm.Set1(np.linspace(0, 1, len(test_labels))))
    ax2.set_xlabel('Test CSV', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontsize=12)
    ax2.set_title('Metric Value Correlations vs Reference\n[Per-Image Center + Global Std]', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(test_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-1, 1)
    
    # Add value labels on bars
    for bar, corr in zip(bars2, metric_correlations_normalized):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "correlation_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_rank_vs_metric_values(reference_data, test_data_list, test_labels, common_classes,
                              actual_num_images, output_dir):
    """Plot test rank vs reference metric values showing mean and variance"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_data_list)))
    
    for test_idx, (test_data, test_label) in enumerate(zip(test_data_list, test_labels)):
        # Collect rank and score pairs
        test_ranks = []
        ref_scores = []
        
        for img_idx in range(actual_num_images):
            ref_ranking, ref_score_dict = reference_data[img_idx]
            test_ranking, _ = test_data[img_idx]
            
            for class_id in common_classes:
                if (class_id in ref_ranking and class_id in test_ranking and 
                    class_id in ref_score_dict):
                    test_ranks.append(test_ranking[class_id])
                    ref_scores.append(ref_score_dict[class_id])
        
        if len(test_ranks) > 0:
            # Calculate mean and std of reference scores for each test rank
            rank_stats = defaultdict(list)
            for tr, rs in zip(test_ranks, ref_scores):
                rank_stats[tr].append(rs)
            
            # Extract data for plotting
            test_rank_positions = sorted(rank_stats.keys())
            ref_score_means = [np.mean(rank_stats[tr]) for tr in test_rank_positions]
            ref_score_stds = [np.std(rank_stats[tr]) for tr in test_rank_positions]
            
            # Plot with error bars
            plt.errorbar(test_rank_positions, ref_score_means, yerr=ref_score_stds,
                        fmt='o-', alpha=0.7, capsize=3, capthick=1,
                        color=colors[test_idx], label=f'{test_label}')
    
    plt.xlabel('Test Rank', fontsize=12)
    plt.ylabel('Reference Score (Mean ± Std)', fontsize=12)
    plt.title(f'Test Rank vs Reference Score Analysis\n({actual_num_images} images, {len(common_classes)} classes)', 
              fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rank_vs_metric_values.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_rank_distribution_comparison(reference_data, test_data_list, test_labels, 
                                    common_classes, actual_num_images, output_dir):
    """Plot Top-k overlap comparison across different methods"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Top-k accuracy comparison
    k_values = [1, 3, 5, 10]
    
    for test_idx, (test_data, test_label) in enumerate(zip(test_data_list, test_labels)):
        accuracies = []
        
        for k in k_values:
            correct = 0
            total = 0
            
            for img_idx in range(actual_num_images):
                ref_ranking, _ = reference_data[img_idx]
                test_ranking, _ = test_data[img_idx]
                
                # Get top-k classes from reference
                ref_top_k = {cls for cls, rank in ref_ranking.items() if rank <= k}
                test_top_k = {cls for cls, rank in test_ranking.items() if rank <= k}
                
                if ref_top_k and test_top_k:
                    overlap = len(ref_top_k.intersection(test_top_k))
                    correct += overlap
                    total += k
            
            accuracies.append(correct / total if total > 0 else 0)
        
        ax.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8,
               label=test_label, color=plt.cm.Set1(test_idx / len(test_data_list)))
    
    ax.set_xlabel('k (Top-k)', fontsize=14)
    ax.set_ylabel('Top-k Overlap Ratio', fontsize=14)
    ax.set_title('Top-k Overlap with Reference', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rank_distribution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def compute_advanced_metrics(reference_data, test_data_list, test_labels, common_classes, 
                             actual_num_images, output_dir):
    """
    Compute advanced ranking metrics (NDCG@k, MRR, WPRA, RBO, etc.) for each test method.
    
    Args:
        reference_data: Reference (gold) data
        test_data_list: List of test datasets
        test_labels: Labels for test methods
        common_classes: List of class IDs common across all datasets (should be [0-9] for CIFAR-10)
        actual_num_images: Number of images to process
        output_dir: Directory to save results
    
    Returns:
        dict: Summary of all metrics per method
    """
    # CIFAR-10 validation: enforce exactly 10 classes
    expected_num_classes = 10
    if len(common_classes) != expected_num_classes:
        print(f"[compute_advanced_metrics] Warning: Expected {expected_num_classes} classes for CIFAR-10, "
              f"got {len(common_classes)}. Forcing common_classes to [0-9].")
        common_classes = list(range(expected_num_classes))
    
    # Ensure common_classes is sorted
    common_classes = sorted(common_classes)
    
    def build_dense_vector(scores_dict, class_ids, label):
        """
        Build dense score vector for specified class IDs.
        Raises ValueError if any class is missing.
        
        Args:
            scores_dict: Dict mapping class_id to score
            class_ids: List of class IDs to extract
            label: Label for error reporting
        
        Returns:
            numpy array of scores
        """
        vec = []
        missing = []
        for cid in class_ids:
            if cid not in scores_dict:
                missing.append(cid)
            else:
                vec.append(scores_dict[cid])
        
        if missing:
            raise ValueError(f"{label}: Missing class_ids {missing}. "
                           f"Available: {sorted(scores_dict.keys())}")
        
        return np.array(vec, dtype=float)
    
    results = {}
    
    for test_idx, (test_data, test_label) in enumerate(zip(test_data_list, test_labels)):
        print(f"\n  Processing {test_label}...")
        
        # Collect metrics for each image
        image_metrics = []
        simple_spearman = []  # per-image Spearman on raw scores
        topk_overlaps = {1: [], 3: [], 5: []}  # per-image top-k overlaps
        skipped_images = 0
        
        for img_idx in range(actual_num_images):
            ref_ranking, ref_scores = reference_data[img_idx]
            test_ranking, test_scores = test_data[img_idx]
            
            try:
                # Build dense vectors (K=10 for CIFAR-10)
                g = build_dense_vector(ref_scores, common_classes, f"Image{img_idx}_Gold")
                s = build_dense_vector(test_scores, common_classes, f"Image{img_idx}_{test_label}")
                
                # Validate vector length
                if len(s) != len(g) or len(s) != len(common_classes):
                    raise ValueError(f"Vector length mismatch: s={len(s)}, g={len(g)}, expected={len(common_classes)}")
                
                # Compute all metrics for this image
                metrics = compute_all_metrics(
                    s=s,
                    g=g,
                    k_values=[3, 5, 10],
                    wpra_m=3,
                    rbo_p=0.9
                )
                image_metrics.append(metrics)

                # Legacy/simple metrics: Spearman on score vectors and Top-k overlap
                sp_corr, _ = spearman_correlation(s, g)
                simple_spearman.append(sp_corr)

                # Compute top-k overlap ratios for k in {1, 3, 5}
                # Build rank arrays from scores (highest score rank=1)
                ref_order = np.argsort(-g)  # descending
                test_order = np.argsort(-s)
                for k in topk_overlaps.keys():
                    ref_topk = set(ref_order[:k].tolist())
                    test_topk = set(test_order[:k].tolist())
                    overlap = len(ref_topk.intersection(test_topk)) / float(k)
                    topk_overlaps[k].append(overlap)
                
            except ValueError as e:
                # Missing class data - skip this image
                skipped_images += 1
                if skipped_images <= 3:  # Print first few errors
                    print(f"    Warning: Skipped image {img_idx}: {e}")
                continue
        
        if skipped_images > 0:
            print(f"    Total skipped images: {skipped_images}/{actual_num_images}")
        
        # Aggregate across images
        if image_metrics:
            aggregated = aggregate_metrics(image_metrics)

            # Aggregate legacy/simple metrics
            if simple_spearman:
                aggregated["spearman_score_mean"] = float(np.mean(simple_spearman))
                aggregated["spearman_score_median"] = float(np.median(simple_spearman))
                aggregated["spearman_score_std"] = float(np.std(simple_spearman))
            for k, values in topk_overlaps.items():
                if values:
                    aggregated[f"top{k}_overlap_mean"] = float(np.mean(values))
                    aggregated[f"top{k}_overlap_median"] = float(np.median(values))
                    aggregated[f"top{k}_overlap_std"] = float(np.std(values))

            results[test_label] = aggregated
            print(f"    Successfully computed metrics for {len(image_metrics)}/{actual_num_images} images")
        else:
            results[test_label] = {}
            print(f"    Error: No valid metrics computed for {test_label}")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "advanced_ranking_metrics.csv")
    save_metrics_to_csv(results, csv_path)
    print(f"\nAdvanced metrics saved to: {csv_path}")
    
    # Print summary table
    print_metrics_table(results)
    
    # Generate comparison plots
    plot_metrics_comparison(results, test_labels, output_dir)
    
    return results


def save_metrics_to_csv(results, csv_path):
    """Save metrics results to CSV file."""
    if not results:
        return
    
    # Get all metric names from first method
    first_method = list(results.keys())[0]
    metric_names = sorted(results[first_method].keys())
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Method'] + metric_names)
        
        # Data rows
        for method, metrics in results.items():
            row = [method] + [f"{metrics.get(name, 0.0):.6f}" for name in metric_names]
            writer.writerow(row)


def print_metrics_table(results):
    """Print formatted table of metrics."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("Advanced Ranking Metrics Summary")
    print(f"{'='*80}\n")
    
    # Extract key metrics for display
    key_metrics = [
        'spearman_score_mean',
        'ndcg@3_mean', 'ndcg@5_mean', 'ndcg@10_mean',
        'mrr_mean', 'wpra_top3_mean', 'wpra_all_mean',
        'top1_overlap_mean', 'top3_overlap_mean', 'top5_overlap_mean',
        'calib_mae_mean', 'calib_mse_mean', 'rbo_mean'
    ]
    
    # Print header
    print(f"{'Method':<20}", end='')
    for metric in key_metrics:
        short_name = metric.replace('_mean', '').replace('ndcg@', 'N@').replace('wpra_', 'W')
        print(f"{short_name:>12}", end='')
    print()
    print('-' * (20 + 12 * len(key_metrics)))
    
    # Print data
    for method, metrics in results.items():
        print(f"{method:<20}", end='')
        for metric in key_metrics:
            value = metrics.get(metric, 0.0)
            print(f"{value:12.4f}", end='')
        print()


def plot_metrics_comparison(results, test_labels, output_dir):
    """Generate comparison plots for advanced metrics."""
    if not results:
        return
    
    # Group metrics by type
    metric_groups = {
        'Rank/Overlap': ['spearman_score_mean', 'top1_overlap_mean', 'top3_overlap_mean', 'top5_overlap_mean'],
        'NDCG@k': ['ndcg@3_mean', 'ndcg@5_mean', 'ndcg@10_mean'],
        'WPRA': ['wpra_top3_mean', 'wpra_all_mean'],
        'Calibrated Errors': ['calib_mae_mean', 'calib_mse_mean'],
        'Other': ['mrr_mean', 'rbo_mean']
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(test_labels)))
    
    for idx, (group_name, metric_list) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        x_pos = np.arange(len(metric_list))
        width = 0.8 / len(test_labels)
        
        for method_idx, method in enumerate(test_labels):
            if method not in results:
                continue
            
            values = [results[method].get(metric, 0.0) for metric in metric_list]
            offset = (method_idx - len(test_labels)/2) * width + width/2
            
            ax.bar(x_pos + offset, values, width, 
                  label=method, color=colors[method_idx], alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_mean', '').replace('ndcg@', 'N@') 
                            for m in metric_list], rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide any unused subplots (in case grid > groups)
    for j in range(len(metric_groups), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "advanced_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to: {output_path}")


def plot_comprehensive_multi_analysis(reference_csv_path, test_csv_paths, test_labels=None,
                                    num_images=None, start_image=0, output_dir=None, exp_name=None):
    """Perform comprehensive analysis comparing multiple test CSVs against a reference CSV"""
    
    print(f"Loading reference CSV: {reference_csv_path}")
    reference_data = load_csv_data_with_scores(reference_csv_path)
    
    test_data_list = []
    for csv_path in test_csv_paths:
        print(f"Loading test CSV: {csv_path}")
        test_data_list.append(load_csv_data_with_scores(csv_path))
    
    # Generate labels if not provided
    if test_labels is None:
        test_labels = [os.path.basename(path).replace('.csv', '') for path in test_csv_paths]
    elif len(test_labels) != len(test_csv_paths):
        print(f"Warning: Number of labels ({len(test_labels)}) doesn't match number of test CSVs ({len(test_csv_paths)})")
        test_labels = [os.path.basename(path).replace('.csv', '') for path in test_csv_paths]
    
    # Determine common classes across all datasets (across all images, not just first)
    min_data_length = min(len(reference_data), min(len(data) for data in test_data_list))
    
    if min_data_length == 0:
        raise ValueError("One or more CSV files are empty")
    
    # For CIFAR-10: Force common_classes to [0-9] regardless of CSV contents
    # This ensures metrics are always computed over the full 10-class space
    CIFAR10_CLASSES = list(range(10))
    common_classes = CIFAR10_CLASSES
    num_classes = len(common_classes)
    
    print(f"Fixed common classes to CIFAR-10 classes: {common_classes}")
    print(f"Found {num_classes} common classes across all datasets")
    print(f"Total images available: {min_data_length}")
    
    # Determine actual number of images to process
    if num_images is None:
        actual_num_images = min_data_length - start_image
    else:
        actual_num_images = min(num_images, min_data_length - start_image)
    
    end_image = start_image + actual_num_images
    
    # Slice data
    reference_data = reference_data[start_image:end_image]
    test_data_list = [data[start_image:end_image] for data in test_data_list]
    
    print(f"Processing {actual_num_images} images (indices {start_image} to {end_image-1})")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(reference_csv_path)
    
    reference_name = os.path.basename(reference_csv_path).replace('.csv', '')
    
    # Build directory name components
    dir_parts = ["multi_corr_anal", reference_name]
    
    # Add experiment name if provided
    if exp_name is not None and exp_name.strip():
        dir_parts.append(exp_name.strip())
    
    # Add label information if provided
    if test_labels is not None and len(test_labels) > 0:
        # Create a joined label string but avoid creating extremely long filenames.
        label_part_raw = "_".join([str(label) for label in test_labels])
        # Maximum length allowed for the label portion of the directory name
        max_label_len = 120
        if len(label_part_raw) <= max_label_len:
            label_part_safe = label_part_raw
            write_label_mapping = False
        else:
            # Use a short hash to represent the long label string and record a mapping later
            label_hash = hashlib.md5(label_part_raw.encode('utf-8')).hexdigest()[:10]
            label_part_safe = f"labels_{label_hash}"
            write_label_mapping = True
        dir_parts.append(label_part_safe)
    
    plot_dir_name = "__".join(dir_parts)
    plot_dir = os.path.join(output_dir, plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to {plot_dir}")
    
    # If we used a hash for labels, write the full original labels into a mapping file for traceability
    if 'write_label_mapping' in locals() and write_label_mapping:
        try:
            mapping_path = os.path.join(plot_dir, "labels_mapping.txt")
            with open(mapping_path, 'w', encoding='utf-8') as mf:
                mf.write("Original labels (joined):\n")
                mf.write(label_part_raw + "\n\n")
                mf.write("Individual labels:\n")
                for lbl in test_labels:
                    mf.write(str(lbl) + "\n")
        except Exception:
            pass

    # Generate all plots
    plots_info = []
    
    print("Generating multi-rank correlation plot...")
    plot_path, rank_corrs = plot_multi_rank_correlation(
        reference_data, test_data_list, test_labels, common_classes, 
        actual_num_images, plot_dir
    )
    plots_info.append(("Multi Rank Correlation", plot_path, rank_corrs))
    
    # Additional analyses with alternative normalizations
    print("Generating multi-metric correlation plot (per-image z-score)...")
    plot_path, metric_corrs_img = plot_multi_metric_correlation_normalized(
        reference_data, test_data_list, test_labels, common_classes,
        actual_num_images, plot_dir, mode='per_image_zscore'
    )
    plots_info.append(("Multi Metric Correlation (per-image z)", plot_path, metric_corrs_img))

    print("Generating multi-metric correlation plot (per-image center, global std)...")
    plot_path, metric_corrs_img_gstd = plot_multi_metric_correlation_normalized(
        reference_data, test_data_list, test_labels, common_classes,
        actual_num_images, plot_dir, mode='per_image_center_global_std'
    )
    plots_info.append(("Multi Metric Correlation (per-image center + global std)", plot_path, metric_corrs_img_gstd))
    
    print("Generating correlation summary plot...")
    plot_path = plot_correlation_summary(test_labels, rank_corrs, metric_corrs_img_gstd, plot_dir)
    plots_info.append(("Correlation Summary", plot_path, None))
    
    print("Generating rank vs metric values plot...")
    plot_path = plot_rank_vs_metric_values(
        reference_data, test_data_list, test_labels, common_classes, 
        actual_num_images, plot_dir
    )
    plots_info.append(("Rank vs Metric Values", plot_path, None))
    
    print("Generating rank distribution comparison plot...")
    plot_path = plot_rank_distribution_comparison(
        reference_data, test_data_list, test_labels, common_classes, 
        actual_num_images, plot_dir
    )
    plots_info.append(("Rank Distribution Comparison", plot_path, None))
    
    # Compute advanced ranking metrics
    print("Computing advanced ranking metrics (NDCG, MRR, WPRA, RBO, etc.)...")
    metrics_summary = compute_advanced_metrics(
        reference_data, test_data_list, test_labels, common_classes, 
        actual_num_images, plot_dir
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("Multi-CSV Correlation Analysis Complete!")
    print(f"{'='*80}")
    print(f"Reference: {reference_name}")
    print(f"Test CSVs: {len(test_csv_paths)}")
    for i, label in enumerate(test_labels):
        print(f"  {i+1}. {label}")
    print(f"Analyzed {actual_num_images} images with {num_classes} common classes")
    print(f"Plots saved to: {plot_dir}")
    print("\nGenerated plots and correlations:")
    for plot_name, plot_path, correlations in plots_info:
        if correlations is not None and isinstance(correlations, list):
            print(f"  - {plot_name}: {os.path.basename(plot_path)}")
            for i, corr in enumerate(correlations):
                print(f"    {test_labels[i]}: r={corr:.3f}")
        else:
            print(f"  - {plot_name}: {os.path.basename(plot_path)}")
    
    return plot_dir

def main():
    args = get_args()
    
    # Check if reference CSV exists
    if not os.path.exists(args.reference_csv):
        raise FileNotFoundError(f"Reference CSV file not found: {args.reference_csv}")
    
    # Check if all test CSV files exist
    for csv_path in args.test_csvs:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {csv_path}")
    
    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate comprehensive analysis
    plot_dir = plot_comprehensive_multi_analysis(
        args.reference_csv,
        args.test_csvs,
        test_labels=args.labels,
        num_images=args.num_images,
        start_image=args.start_image,
        output_dir=args.output_dir,
        exp_name=args.exp_name
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
