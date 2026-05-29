#!/usr/bin/env python3
"""
Correlation analysis between UNA (ReTrack) and LOGOA scores.

This script computes per-image correlations across the 16 ReTrack styles.

Note: This is a preliminary analysis with only 16 styles (vs 60 for full LOGOA).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ranking_metrics import ndcg_at_k, mrr_top1, wpra_top_heavy

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="UNA-LOGOA correlation analysis")
    parser.add_argument(
        "--logoa_dir",
        type=str,
        default="outputs/logoa_scores_16styles_step7500",
        help="Directory containing LOGOA scores CSV"
    )
    parser.add_argument(
        "--una_dir",
        type=str,
        default="outputs/una_retrack_scores_16styles_step5000",
        help="Directory containing UNA ReTrack scores CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/una_retrack_correlation",
        help="Output directory for analysis results"
    )
    return parser.parse_args()


def load_and_merge_scores(logoa_dir, una_dir):
    """Load and merge LOGOA and UNA scores."""
    print("Loading score files...")
    
    logoa_csv = Path(logoa_dir) / "logoa_scores.csv"
    una_csv = Path(una_dir) / "una_retrack_scores.csv"
    
    df_logoa = pd.read_csv(logoa_csv)
    df_una = pd.read_csv(una_csv)
    
    print(f"  LOGOA: {len(df_logoa)} rows")
    print(f"  UNA ReTrack: {len(df_una)} rows")
    
    # Check column names
    print(f"\n  LOGOA columns: {list(df_logoa.columns)}")
    print(f"  UNA columns: {list(df_una.columns)}")
    
    # Normalize style names (replace spaces with underscores)
    # This handles inconsistencies between different score files (e.g. "Artist Sketch" vs "Artist_Sketch")
    print("  Normalizing style names (spaces -> underscores)...")
    for df_name, df_temp in [("LOGOA", df_logoa), ("UNA", df_una)]:
        if 'generated_style' in df_temp.columns:
            df_temp['generated_style'] = df_temp['generated_style'].astype(str).str.replace(' ', '_')
        if 'attribution_style' in df_temp.columns:
            df_temp['attribution_style'] = df_temp['attribution_style'].astype(str).str.replace(' ', '_')
    
    # Both should have format: generated_style, object_name, attribution_style, score, image_path
    # Merge on common keys
    merge_keys = ['generated_style', 'object_name', 'attribution_style', 'image_path']
    
    df = df_logoa.merge(
        df_una,
        on=merge_keys,
        how='inner',
        suffixes=('_logoa', '_una')
    )
    
    # Rename score columns to ensure consistent naming
    rename_map = {}
    if 'score_logoa' in df.columns:
        rename_map['score_logoa'] = 'logoa_score'
    if 'score_una' in df.columns:
        rename_map['score_una'] = 'una_score'
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")
    
    print(f"\n  Merged: {len(df)} rows")
    
    # Verify structure
    unique_images = df.groupby(['generated_style', 'object_name', 'image_path']).size()
    print(f"  Unique images: {len(unique_images)}")
    if len(unique_images) > 0:
        print(f"  Styles per image: {unique_images.iloc[0]}")
    
    # Show which attribution styles are available
    attribution_styles = sorted(df['attribution_style'].unique())
    K_expected = len(attribution_styles)
    print(f"\n  Attribution styles available (K={K_expected}): {attribution_styles[:5]}...")
    
    # Check for missing styles per image
    styles_per_image = df.groupby(['generated_style', 'object_name', 'image_path'])['attribution_style'].nunique()
    incomplete_images = (styles_per_image != K_expected).sum()
    if incomplete_images > 0:
        print(f"  WARNING: {incomplete_images} images have incomplete style coverage (expected {K_expected})")
    
    print()
    
    return df, K_expected


def compute_per_image_correlations(df, K_expected):
    """
    Compute correlation per image across attribution styles.
    
    For each image:
    - Get N-dimensional vector of LOGOA scores (one per attribution style)
    - Get N-dimensional vector of UNA scores
    - Compute correlations between these vectors
    
    Args:
        df: Merged dataframe with LOGOA and UNA scores
        K_expected: Expected number of styles per image
    
    Returns:
        DataFrame with per-image correlation coefficients
    """
    print("Computing per-image correlations...")
    
    results = []
    skipped_incomplete = 0
    
    # Group by image
    image_groups = df.groupby(['generated_style', 'object_name', 'image_path'])
    
    for (gen_style, obj_name, img_path), group in tqdm(image_groups, desc="Images"):
        # Sort by attribution style for consistency
        group = group.sort_values('attribution_style')
        
        logoa_scores = group['logoa_score'].values
        una_scores = group['una_score'].values
        
        n_styles = len(logoa_scores)
        
        # CRITICAL: Skip images without complete style coverage
        if n_styles != K_expected:
            skipped_incomplete += 1
            continue
        
        # Skip if too few styles (need at least 3 for meaningful correlation)
        if n_styles < 3:
            continue
        
        # Compute correlations
        try:
            pearson_r, pearson_p = pearsonr(logoa_scores, una_scores)
            spearman_r, spearman_p = spearmanr(logoa_scores, una_scores)
            kendall_tau, kendall_p = kendalltau(logoa_scores, una_scores)
        except Exception as e:
            print(f"Warning: Correlation failed for {gen_style}/{obj_name}/{img_path}: {e}")
            continue
        
        # Convert to numpy arrays for ranking metrics
        logoa_scores_array = np.array(logoa_scores)
        una_scores_array = np.array(una_scores)
        
        # Top-K agreement
        logoa_ranks = np.argsort(-logoa_scores_array)  # Descending order
        una_ranks = np.argsort(-una_scores_array)
        
        top1_match = logoa_ranks[0] == una_ranks[0]
        
        # Top-3 agreement (only if K >= 3)
        if n_styles >= 3:
            top3_match = len(set(logoa_ranks[:3]) & set(una_ranks[:3])) / 3.0
        else:
            top3_match = np.nan
        
        # Top-5 agreement (only if K >= 5)
        if n_styles >= 5:
            top5_match = len(set(logoa_ranks[:5]) & set(una_ranks[:5])) / 5.0
        else:
            top5_match = np.nan
        
        # NDCG@k (using LOGOA as ground truth)
        ndcg3 = ndcg_at_k(una_scores_array, logoa_scores_array, k=3) if n_styles >= 3 else np.nan
        ndcg5 = ndcg_at_k(una_scores_array, logoa_scores_array, k=5) if n_styles >= 5 else np.nan
        ndcg10 = ndcg_at_k(una_scores_array, logoa_scores_array, k=10) if n_styles >= 10 else np.nan
        
        # MRR (Mean Reciprocal Rank)
        mrr = mrr_top1(una_scores_array, logoa_scores_array)
        
        # WPRA_top3 (Weighted Pairwise Ranking Agreement)
        wpra_m = min(3, n_styles)
        wpra = wpra_top_heavy(una_scores_array, logoa_scores_array, m=wpra_m)
        
        result = {
            'generated_style': gen_style,
            'object_name': obj_name,
            'image_path': img_path,
            'n_styles': n_styles,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'top1_agreement': top1_match,
            'top3_agreement': top3_match,
            'top5_agreement': top5_match,
            'ndcg@3': ndcg3,
            'ndcg@5': ndcg5,
            'ndcg@10': ndcg10,
            'mrr': mrr,
            'wpra_top3': wpra
        }
        
        results.append(result)
    
    if skipped_incomplete > 0:
        print(f"  Skipped {skipped_incomplete} images with incomplete style coverage (expected K={K_expected})")
    
    return pd.DataFrame(results)


def compute_top_k_agreement(df, k_values=[1, 3, 5]):
    """
    Compute top-k agreement per image.
    
    For each image, check if the top-k attribution styles match between LOGOA and UNA.
    """
    print(f"Computing top-k agreement (k={k_values})...")
    
    results = []
    
    # Group by image
    image_groups = df.groupby(['generated_style', 'object_name', 'image_path'])
    
    for (gen_style, obj_name, img_path), group in tqdm(image_groups, desc="Images"):
        # Sort by scores (descending)
        logoa_ranked = group.sort_values('logoa_score', ascending=False)['attribution_style'].values
        una_ranked = group.sort_values('una_score', ascending=False)['attribution_style'].values
        
        result = {
            'generated_style': gen_style,
            'object_name': obj_name,
            'image_path': img_path,
            'n_styles': len(group)
        }
        
        # Compute agreement for each k
        for k in k_values:
            if k > len(logoa_ranked):
                continue
            
            logoa_topk = set(logoa_ranked[:k])
            una_topk = set(una_ranked[:k])
            
            overlap = len(logoa_topk & una_topk)
            agreement = overlap / k
            
            result[f'top{k}_agreement'] = agreement
            result[f'top{k}_overlap'] = overlap
        
        results.append(result)
    
    return pd.DataFrame(results)


def plot_correlation_distributions(df_corr, output_dir):
    """Plot distributions of correlation coefficients."""
    print("Plotting correlation distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['pearson_r', 'spearman_r', 'kendall_tau']
    titles = ['Pearson r', 'Spearman ρ', 'Kendall τ']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = df_corr[metric].dropna()
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
        ax.axvline(values.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {values.median():.3f}')
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} Distribution\n(UNA vs LOGOA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: correlation_distributions.png")


def plot_correlation_by_prompt(df_corr, output_dir):
    """Plot correlation by prompt type."""
    print("Plotting correlation by prompt type...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['pearson_r', 'spearman_r', 'kendall_tau']
    titles = ['Pearson r', 'Spearman ρ', 'Kendall τ']
    
    for ax, metric, title in zip(axes, metrics, titles):
        df_corr.boxplot(column=metric, by='prompt_type', ax=ax)
        ax.set_title(f'{title} by Prompt Type')
        ax.set_xlabel('Prompt Type')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('UNA-LOGOA Correlation by Prompt Type', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_by_prompt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: correlation_by_prompt.png")


def plot_topk_agreement(df_topk, output_dir):
    """Plot top-k agreement statistics."""
    print("Plotting top-k agreement...")
    
    # Find k values
    k_columns = [col for col in df_topk.columns if col.startswith('top') and col.endswith('_agreement')]
    k_values = sorted([int(col.replace('top', '').replace('_agreement', '')) for col in k_columns])
    
    # Compute mean agreement for each k
    mean_agreements = []
    for k in k_values:
        mean_agreements.append(df_topk[f'top{k}_agreement'].mean())
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(k_values)), mean_agreements, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylabel('Mean Agreement')
    ax.set_title('Top-K Agreement: UNA vs LOGOA')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, val in enumerate(mean_agreements):
        ax.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: topk_agreement.png")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("UNA-LOGOA Correlation Analysis (ReTrack)")
    print("=" * 80)
    print(f"LOGOA directory: {args.logoa_dir}")
    print(f"UNA directory: {args.una_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load and merge scores
    df, K_expected = load_and_merge_scores(args.logoa_dir, args.una_dir)
    
    if len(df) == 0:
        print("Error: No matching scores found between LOGOA and UNA")
        return
    
    # Save merged data
    merged_csv = output_dir / 'merged_scores.csv'
    df.to_csv(merged_csv, index=False)
    print(f"✓ Saved merged scores to: {merged_csv}")
    print()
    
    # Compute per-image correlations
    df_corr = compute_per_image_correlations(df, K_expected)
    
    # Save correlation results
    corr_csv = output_dir / 'per_image_correlations.csv'
    df_corr.to_csv(corr_csv, index=False)
    print(f"✓ Saved per-image correlations to: {corr_csv}")
    print()
    
    # Print summary statistics
    print("=" * 80)
    print("COMPREHENSIVE RANKING METRICS SUMMARY")
    print("=" * 80)
    print(f"Number of images: {len(df_corr)}")
    
    # Correlation metrics
    print(f"\nCorrelation Metrics:")
    print(f"  Spearman ρ:     {df_corr['spearman_r'].mean():.4f} ± {df_corr['spearman_r'].std():.4f} (median: {df_corr['spearman_r'].median():.4f})")
    print(f"  Pearson r:      {df_corr['pearson_r'].mean():.4f} ± {df_corr['pearson_r'].std():.4f} (median: {df_corr['pearson_r'].median():.4f})")
    print(f"  Kendall τ:      {df_corr['kendall_tau'].mean():.4f} ± {df_corr['kendall_tau'].std():.4f} (median: {df_corr['kendall_tau'].median():.4f})")
    
    # Top-K agreement
    print(f"\nTop-K Agreement:")
    if 'top1_agreement' in df_corr.columns:
        top1_count = df_corr['top1_agreement'].sum()
        print(f"  Top-1:          {df_corr['top1_agreement'].mean():.4f} ({top1_count:.0f}/{len(df_corr)} matches)")
    if 'top3_agreement' in df_corr.columns and df_corr['top3_agreement'].notna().any():
        print(f"  Top-3:          {df_corr['top3_agreement'].mean():.4f} ± {df_corr['top3_agreement'].std():.4f}")
    if 'top5_agreement' in df_corr.columns and df_corr['top5_agreement'].notna().any():
        print(f"  Top-5:          {df_corr['top5_agreement'].mean():.4f} ± {df_corr['top5_agreement'].std():.4f}")
    
    # NDCG metrics
    print(f"\nNDCG (Normalized Discounted Cumulative Gain):")
    if 'ndcg@3' in df_corr.columns and df_corr['ndcg@3'].notna().any():
        print(f"  NDCG@3:         {df_corr['ndcg@3'].mean():.4f} ± {df_corr['ndcg@3'].std():.4f} (median: {df_corr['ndcg@3'].median():.4f})")
    if 'ndcg@5' in df_corr.columns and df_corr['ndcg@5'].notna().any():
        print(f"  NDCG@5:         {df_corr['ndcg@5'].mean():.4f} ± {df_corr['ndcg@5'].std():.4f} (median: {df_corr['ndcg@5'].median():.4f})")
    if 'ndcg@10' in df_corr.columns and df_corr['ndcg@10'].notna().any():
        print(f"  NDCG@10:        {df_corr['ndcg@10'].mean():.4f} ± {df_corr['ndcg@10'].std():.4f} (median: {df_corr['ndcg@10'].median():.4f})")
    
    # Other ranking metrics
    print(f"\nOther Ranking Metrics:")
    if 'mrr' in df_corr.columns and df_corr['mrr'].notna().any():
        print(f"  MRR:            {df_corr['mrr'].mean():.4f} ± {df_corr['mrr'].std():.4f} (median: {df_corr['mrr'].median():.4f})")
    if 'wpra_top3' in df_corr.columns and df_corr['wpra_top3'].notna().any():
        print(f"  WPRA_top3:      {df_corr['wpra_top3'].mean():.4f} ± {df_corr['wpra_top3'].std():.4f} (median: {df_corr['wpra_top3'].median():.4f})")
    print("=" * 80)
    print()
    
    # Correlation by prompt type (if available)
    if 'prompt_type' in df_corr.columns:
        print("Correlation by prompt type:")
        for prompt_type in sorted(df_corr['prompt_type'].unique()):
            subset = df_corr[df_corr['prompt_type'] == prompt_type]
            print(f"\n  {prompt_type}:")
            print(f"    Pearson:  {subset['pearson_r'].mean():.4f}")
            print(f"    Spearman: {subset['spearman_r'].mean():.4f}")
            print(f"    Kendall:  {subset['kendall_tau'].mean():.4f}")
        print()
    else:
        print("Correlation by prompt type: Not available (column 'prompt_type' missing)")
        print()
    
    # Top-K agreement is already computed in per-image correlations
    # Print summary (already included in comprehensive metrics above)
    print("Note: Top-K Agreement metrics are included in the per-image correlations CSV")
    print()
    
    # Generate plots
    print("=" * 80)
    print("Generating Plots")
    print("=" * 80)
    plot_correlation_distributions(df_corr, output_dir)
    
    if 'prompt_type' in df_corr.columns:
        plot_correlation_by_prompt(df_corr, output_dir)
    else:
        print("Skipping plot_correlation_by_prompt (column 'prompt_type' missing)")
    
    # Plot Top-K agreement from per-image correlations
    if 'top1_agreement' in df_corr.columns:
        plot_topk_agreement(df_corr, output_dir)
    else:
        print("Skipping plot_topk_agreement (Top-K metrics not computed)")
    
    # Write summary report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UNA-LOGOA Correlation Analysis (ReTrack)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"LOGOA directory: {args.logoa_dir}\n")
        f.write(f"UNA directory: {args.una_dir}\n")
        f.write(f"Number of attribution styles: {df['attribution_style'].nunique()}\n")
        f.write(f"Number of images: {len(df_corr)}\n\n")
        
        f.write("Correlation Metrics:\n")
        f.write(f"  Spearman ρ: {df_corr['spearman_r'].mean():.4f} ± {df_corr['spearman_r'].std():.4f} (median: {df_corr['spearman_r'].median():.4f})\n")
        f.write(f"  Pearson r:  {df_corr['pearson_r'].mean():.4f} ± {df_corr['pearson_r'].std():.4f} (median: {df_corr['pearson_r'].median():.4f})\n")
        f.write(f"  Kendall τ:  {df_corr['kendall_tau'].mean():.4f} ± {df_corr['kendall_tau'].std():.4f} (median: {df_corr['kendall_tau'].median():.4f})\n\n")
        
        # Top-K agreement
        f.write("Top-K Agreement:\n")
        if 'top1_agreement' in df_corr.columns:
            f.write(f"  Top-1: {df_corr['top1_agreement'].mean():.4f} ({df_corr['top1_agreement'].sum():.0f}/{len(df_corr)} matches)\n")
        if 'top3_agreement' in df_corr.columns and df_corr['top3_agreement'].notna().any():
            f.write(f"  Top-3: {df_corr['top3_agreement'].mean():.4f} ± {df_corr['top3_agreement'].std():.4f}\n")
        if 'top5_agreement' in df_corr.columns and df_corr['top5_agreement'].notna().any():
            f.write(f"  Top-5: {df_corr['top5_agreement'].mean():.4f} ± {df_corr['top5_agreement'].std():.4f}\n")
        f.write("\n")
        
        # NDCG metrics
        f.write("NDCG Metrics:\n")
        if 'ndcg@3' in df_corr.columns and df_corr['ndcg@3'].notna().any():
            f.write(f"  NDCG@3:  {df_corr['ndcg@3'].mean():.4f} ± {df_corr['ndcg@3'].std():.4f} (median: {df_corr['ndcg@3'].median():.4f})\n")
        if 'ndcg@5' in df_corr.columns and df_corr['ndcg@5'].notna().any():
            f.write(f"  NDCG@5:  {df_corr['ndcg@5'].mean():.4f} ± {df_corr['ndcg@5'].std():.4f} (median: {df_corr['ndcg@5'].median():.4f})\n")
        if 'ndcg@10' in df_corr.columns and df_corr['ndcg@10'].notna().any():
            f.write(f"  NDCG@10: {df_corr['ndcg@10'].mean():.4f} ± {df_corr['ndcg@10'].std():.4f} (median: {df_corr['ndcg@10'].median():.4f})\n")
        f.write("\n")
        
        # Other ranking metrics
        f.write("Other Ranking Metrics:\n")
        if 'mrr' in df_corr.columns and df_corr['mrr'].notna().any():
            f.write(f"  MRR:        {df_corr['mrr'].mean():.4f} ± {df_corr['mrr'].std():.4f} (median: {df_corr['mrr'].median():.4f})\n")
        if 'wpra_top3' in df_corr.columns and df_corr['wpra_top3'].notna().any():
            f.write(f"  WPRA_top3:  {df_corr['wpra_top3'].mean():.4f} ± {df_corr['wpra_top3'].std():.4f} (median: {df_corr['wpra_top3'].median():.4f})\n")
        f.write("\n")
    
    print(f"✓ Saved summary report to: {report_path}")
    print()
    print("=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
