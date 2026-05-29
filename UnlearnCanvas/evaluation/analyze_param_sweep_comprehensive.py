#!/usr/bin/env python3
"""
Comprehensive correlation analysis for parameter sweep (baseline and weighted_style_select).

Computes all metrics from ffsd_vr_comprehensive_comparison.md:
- Top-K Agreement (k=1, 3, 5)
- Correlation (Spearman, Pearson, Kendall)
- NDCG@k (k=3, 5, 10)
- Other Ranking (MRR, WPRA_top3, RBO)

Focuses on highest-performing configurations based on key metrics:
- Top-1, Top-3, NDCG@3, MRR, WPRA_top3, Spearman
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import json

# Import ranking metrics
import sys
sys.path.append(str(Path(__file__).parent))
from ranking_metrics import ndcg_at_k, mrr_top1, wpra_top_heavy, rbo_truncated


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive parameter sweep correlation analysis")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["baseline", "weighted", "beta_eta"],
        help="Strategy to analyze: 'baseline' (style_removed), 'weighted' (weighted_style_select), or 'beta_eta' (beta/eta sweep)"
    )
    parser.add_argument(
        "--sweep_dir",
        type=str,
        help="Directory containing sweep results (auto-detected if not specified)"
    )
    parser.add_argument(
        "--logoa_csv",
        type=str,
        default="outputs/logoa_scores_ffsd_very_relaxed/logoa_scores.csv",
        help="LOGOA scores CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--param_configs",
        type=str,
        help="Parameter configuration JSON file (auto-detected if not specified)"
    )
    return parser.parse_args()


def load_logoa_scores(logoa_csv):
    """Load LOGOA ground truth scores."""
    print(f"Loading LOGOA scores from {logoa_csv}")
    df = pd.read_csv(logoa_csv)
    
    # Normalize style names
    df['generated_style'] = df['generated_style'].str.replace(' ', '_')
    df['attribution_style'] = df['attribution_style'].str.replace(' ', '_')
    
    print(f"  Loaded {len(df)} LOGOA entries")
    print(f"  Unique images: {df.groupby(['generated_style', 'object_name', 'image_path']).ngroups}")
    print(f"  Attribution styles: {df['attribution_style'].nunique()}")
    
    return df


def load_una_scores(sweep_dir, config_name, step):
    """Load UNA scores for a specific config and step."""
    all_scores = []
    config_dir = Path(sweep_dir) / config_name
    
    if not config_dir.exists():
        print(f"  Warning: Config directory not found: {config_dir}")
        return None
    
    # Iterate through style directories
    style_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()])
    
    for style_dir in style_dirs:
        csv_file = style_dir / f"una_scores_step{step}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # Normalize style names
            df['generated_style'] = df['generated_style'].str.replace(' ', '_')
            df['attribution_style'] = df['attribution_style'].str.replace(' ', '_')
            all_scores.append(df)
    
    if not all_scores:
        return None
    
    merged = pd.concat(all_scores, ignore_index=True)
    return merged


def compute_comprehensive_metrics(una_df, logoa_df):
    """
    Compute all metrics for comparison with ffsd_vr_comprehensive_comparison.md.
    
    Returns dict with:
    - top1_agreement, top3_agreement, top5_agreement
    - spearman_mean, spearman_std, spearman_median
    - pearson_mean, pearson_std, pearson_median
    - kendall_mean, kendall_std, kendall_median
    - ndcg3_mean, ndcg5_mean, ndcg10_mean
    - mrr_mean, mrr_median
    - wpra_top3_mean, wpra_top3_median
    - rbo_mean (if computed)
    - n_images
    """
    # Merge on common keys
    merge_keys = ['generated_style', 'object_name', 'attribution_style', 'image_path']
    
    df = logoa_df.merge(
        una_df,
        on=merge_keys,
        how='inner',
        suffixes=('_logoa', '_una')
    )
    
    # Handle score column naming
    if 'logoa_score' not in df.columns and 'score_logoa' in df.columns:
        df = df.rename(columns={'score_logoa': 'logoa_score'})
    if 'una_score' not in df.columns and 'score_una' in df.columns:
        df = df.rename(columns={'score_una': 'una_score'})
    
    n_total_rows = len(df)
    n_images = df.groupby(['generated_style', 'object_name', 'image_path']).ngroups
    
    if n_images == 0:
        print("  Error: No images after merge")
        return None
    
    # Per-image metrics
    results = []
    image_groups = df.groupby(['generated_style', 'object_name', 'image_path'])
    
    for (gen_style, obj_name, img_path), group in tqdm(image_groups, desc="  Computing metrics"):
        group = group.sort_values('attribution_style')
        
        logoa_scores = group['logoa_score'].values
        una_scores = group['una_score'].values
        
        n_styles = len(logoa_scores)
        
        if n_styles < 3:
            continue
        
        # Correlations
        try:
            pearson_r, _ = pearsonr(logoa_scores, una_scores)
            spearman_r, _ = spearmanr(logoa_scores, una_scores)
            kendall_tau, _ = kendalltau(logoa_scores, una_scores)
        except:
            continue
        
        # Rankings
        logoa_ranks = np.argsort(-logoa_scores)
        una_ranks = np.argsort(-una_scores)
        
        # Top-K agreement
        top1_match = 1.0 if logoa_ranks[0] == una_ranks[0] else 0.0
        top3_match = len(set(logoa_ranks[:3]) & set(una_ranks[:3])) / 3.0 if n_styles >= 3 else np.nan
        top5_match = len(set(logoa_ranks[:5]) & set(una_ranks[:5])) / 5.0 if n_styles >= 5 else np.nan
        
        # NDCG@k
        ndcg3 = ndcg_at_k(una_scores, logoa_scores, k=3) if n_styles >= 3 else np.nan
        ndcg5 = ndcg_at_k(una_scores, logoa_scores, k=5) if n_styles >= 5 else np.nan
        ndcg10 = ndcg_at_k(una_scores, logoa_scores, k=10) if n_styles >= 10 else np.nan
        
        # MRR
        mrr = mrr_top1(una_scores, logoa_scores)
        
        # WPRA_top3
        wpra_m = min(3, n_styles)
        wpra = wpra_top_heavy(una_scores, logoa_scores, m=wpra_m)
        
        # RBO (optional, can be slow)
        try:
            rbo_val = rbo_truncated(una_scores, logoa_scores, p=0.9)
        except:
            rbo_val = np.nan
        
        results.append({
            'generated_style': gen_style,
            'object_name': obj_name,
            'image_path': img_path,
            'n_styles': n_styles,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'kendall_tau': kendall_tau,
            'top1_agreement': top1_match,
            'top3_agreement': top3_match,
            'top5_agreement': top5_match,
            'ndcg@3': ndcg3,
            'ndcg@5': ndcg5,
            'ndcg@10': ndcg10,
            'mrr': mrr,
            'wpra_top3': wpra,
            'rbo_p09': rbo_val
        })
    
    if not results:
        print("  Error: No valid image-level metrics computed")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Aggregate metrics
    metrics = {
        # Sample size
        'n_images': len(results_df),
        'n_rows': n_total_rows,
        
        # Top-K Agreement (percentage)
        'top1_agreement': results_df['top1_agreement'].mean(),
        'top1_count': int(results_df['top1_agreement'].sum()),
        'top3_agreement': results_df['top3_agreement'].mean(),
        'top3_std': results_df['top3_agreement'].std(),
        'top5_agreement': results_df['top5_agreement'].mean(),
        'top5_std': results_df['top5_agreement'].std(),
        
        # Correlations
        'spearman_mean': results_df['spearman_r'].mean(),
        'spearman_std': results_df['spearman_r'].std(),
        'spearman_median': results_df['spearman_r'].median(),
        'pearson_mean': results_df['pearson_r'].mean(),
        'pearson_std': results_df['pearson_r'].std(),
        'pearson_median': results_df['pearson_r'].median(),
        'kendall_mean': results_df['kendall_tau'].mean(),
        'kendall_std': results_df['kendall_tau'].std(),
        'kendall_median': results_df['kendall_tau'].median(),
        
        # NDCG
        'ndcg3_mean': results_df['ndcg@3'].mean(),
        'ndcg3_std': results_df['ndcg@3'].std(),
        'ndcg3_median': results_df['ndcg@3'].median(),
        'ndcg5_mean': results_df['ndcg@5'].mean(),
        'ndcg5_std': results_df['ndcg@5'].std(),
        'ndcg5_median': results_df['ndcg@5'].median(),
        'ndcg10_mean': results_df['ndcg@10'].mean(),
        'ndcg10_std': results_df['ndcg@10'].std(),
        'ndcg10_median': results_df['ndcg@10'].median(),
        
        # Other ranking
        'mrr_mean': results_df['mrr'].mean(),
        'mrr_std': results_df['mrr'].std(),
        'mrr_median': results_df['mrr'].median(),
        'wpra_top3_mean': results_df['wpra_top3'].mean(),
        'wpra_top3_std': results_df['wpra_top3'].std(),
        'wpra_top3_median': results_df['wpra_top3'].median(),
        'rbo_mean': results_df['rbo_p09'].mean(),
        'rbo_std': results_df['rbo_p09'].std(),
        'rbo_median': results_df['rbo_p09'].median(),
    }
    
    return metrics, results_df


def main():
    args = parse_args()
    
    # Auto-detect directories based on strategy
    if args.strategy == "baseline":
        sweep_dir = args.sweep_dir or "outputs/param_sweep_retrack"
        output_dir = args.output_dir or "outputs/param_sweep_comprehensive"
        param_configs = args.param_configs or "param_configs.json"
        config_prefix = ""
    elif args.strategy == "weighted":
        sweep_dir = args.sweep_dir or "outputs/weighted_select_sweep"
        output_dir = args.output_dir or "outputs/weighted_select_comprehensive"
        param_configs = args.param_configs or "param_configs_weighted_select.json"
        config_prefix = "ws_"
    else:  # beta_eta
        sweep_dir = args.sweep_dir or "outputs/beta_eta_sweep"
        output_dir = args.output_dir or "outputs/beta_eta_comprehensive"
        param_configs = args.param_configs or "param_configs_beta_eta_sweep.json"
        config_prefix = ""
    
    sweep_dir = Path(sweep_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print(f"Comprehensive Correlation Analysis: {args.strategy.upper()}")
    print("="*80)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LOGOA scores: {args.logoa_csv}")
    print()
    
    # Load LOGOA
    logoa_df = load_logoa_scores(args.logoa_csv)
    
    # Load parameter configurations
    with open(param_configs, 'r') as f:
        config_data = json.load(f)
    
    # Extract parameter combinations
    param_combos = []
    if 'base_config' in config_data:
        # New format (weighted_style_select or beta_eta with param_grid)
        base = config_data['base_config']
        # Try both 'param_grid' and 'parameter_combinations' keys
        param_list = config_data.get('param_grid', config_data.get('parameter_combinations', []))
        for param_set in param_list:
            config_name = param_set['id']  # Already has prefix (ws_) or not
            params = param_set.get('params', param_set)  # Handle nested params
            
            # For beta_eta sweep, params only has beta and eta_uniform
            # learning_rate, lambda_stabilize, epsilon are in base_config
            if args.strategy == "beta_eta":
                param_combos.append({
                    'config_name': config_name,
                    'beta': params.get('beta', base.get('beta', 2.0)),
                    'eta_uniform': params.get('eta_uniform', base.get('eta_uniform', 0.3)),
                    'learning_rate': base.get('learning_rate'),
                    'lambda_stabilize': base.get('lambda_stabilize'),
                    'epsilon': base.get('epsilon')
                })
            else:
                param_combos.append({
                    'config_name': config_name,
                    'learning_rate': params['learning_rate'],
                    'lambda_stabilize': params['lambda_stabilize'],
                    'epsilon': params['epsilon']
                })
    else:
        # Old format (baseline with parameter_combinations)
        for param_set in config_data.get('parameter_combinations', []):
            param_combos.append({
                'config_name': param_set['id'],
                'learning_rate': param_set['learning_rate'],
                'lambda_stabilize': param_set['lambda_stabilize'],
                'epsilon': param_set['epsilon']
            })
    
    print(f"Found {len(param_combos)} parameter configurations")
    print()
    
    # Evaluate all configs at all steps
    steps = [1000, 2000, 3000, 4000, 5000]
    all_results = []
    
    for param in tqdm(param_combos, desc="Configurations"):
        config_name = param['config_name']
        
        for step in steps:
            print(f"\nAnalyzing {config_name} at step {step}...")
            
            # Load UNA scores
            una_df = load_una_scores(sweep_dir, config_name, step)
            
            if una_df is None or len(una_df) == 0:
                print(f"  Skipping: No UNA scores found")
                continue
            
            print(f"  Loaded {len(una_df)} UNA rows")
            
            # Compute comprehensive metrics
            result = compute_comprehensive_metrics(una_df, logoa_df)
            
            if result is None:
                print(f"  Skipping: Metrics computation failed")
                continue
            
            metrics, per_image_df = result
            
            # Add configuration info
            metrics['config'] = config_name
            metrics['step'] = step
            metrics['learning_rate'] = param['learning_rate']
            metrics['lambda_stabilize'] = param['lambda_stabilize']
            metrics['epsilon'] = param['epsilon']
            
            # Add beta_eta specific params if applicable
            if args.strategy == "beta_eta":
                metrics['beta'] = param.get('beta')
                metrics['eta_uniform'] = param.get('eta_uniform')
            
            all_results.append(metrics)
            
            # Save per-image results
            per_image_file = output_dir / f"{config_name}_step{step}_per_image.csv"
            per_image_df.to_csv(per_image_file, index=False)
            
            print(f"  Top-1: {metrics['top1_agreement']:.4f} ({metrics['top1_count']}/{metrics['n_images']})")
            print(f"  Spearman: {metrics['spearman_mean']:.4f}")
            print(f"  MRR: {metrics['mrr_mean']:.4f}")
            print(f"  NDCG@3: {metrics['ndcg3_mean']:.4f}")
            print(f"  WPRA_top3: {metrics['wpra_top3_mean']:.4f}")
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = output_dir / "comprehensive_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n{'='*80}")
        print(f"Saved summary to {summary_file}")
        
        # Generate report
        generate_report(summary_df, output_dir, args.strategy)
    else:
        print("\nNo results to save")


def generate_report(summary_df, output_dir, strategy):
    """Generate comprehensive report with focus on key metrics."""
    report_file = output_dir / "comprehensive_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Comprehensive Correlation Analysis Report: {strategy.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Key Metrics (priority order):\n")
        f.write("1. Top-1 Agreement\n")
        f.write("2. Top-3 Agreement\n")
        f.write("3. NDCG@3\n")
        f.write("4. MRR\n")
        f.write("5. WPRA_top3\n")
        f.write("6. Spearman ρ\n\n")
        
        # Best overall across all metrics
        f.write("="*80 + "\n")
        f.write("Best Configurations by Metric\n")
        f.write("="*80 + "\n\n")
        
        key_metrics = [
            ('top1_agreement', 'Top-1 Agreement', 'higher'),
            ('top3_agreement', 'Top-3 Agreement', 'higher'),
            ('ndcg3_mean', 'NDCG@3', 'higher'),
            ('mrr_mean', 'MRR', 'higher'),
            ('wpra_top3_mean', 'WPRA_top3', 'higher'),
            ('spearman_mean', 'Spearman ρ', 'higher'),
        ]
        
        for metric_col, metric_name, direction in key_metrics:
            if direction == 'higher':
                best_row = summary_df.loc[summary_df[metric_col].idxmax()]
            else:
                best_row = summary_df.loc[summary_df[metric_col].idxmin()]
            
            f.write(f"\n--- {metric_name} (Best) ---\n")
            f.write(f"Config: {best_row['config']}\n")
            f.write(f"Step: {best_row['step']}\n")
            f.write(f"Value: {best_row[metric_col]:.4f}\n")
            f.write(f"Learning Rate: {best_row['learning_rate']}\n")
            f.write(f"Lambda: {best_row['lambda_stabilize']}\n")
            f.write(f"Epsilon: {best_row['epsilon']}\n")
        
        # Best configs per step
        f.write("\n" + "="*80 + "\n")
        f.write("Best Configurations by Step (Top-1 metric)\n")
        f.write("="*80 + "\n\n")
        
        for step in sorted(summary_df['step'].unique()):
            step_df = summary_df[summary_df['step'] == step].copy()
            step_df = step_df.sort_values('top1_agreement', ascending=False)
            
            f.write(f"\n--- Step {step} ---\n")
            f.write("Top 3 configs:\n")
            for i, (_, row) in enumerate(step_df.head(3).iterrows(), 1):
                f.write(f"\n{i}. {row['config']}\n")
                f.write(f"   Top-1: {row['top1_agreement']:.4f} ({row['top1_count']}/{int(row['n_images'])})\n")
                f.write(f"   Top-3: {row['top3_agreement']:.4f}\n")
                f.write(f"   NDCG@3: {row['ndcg3_mean']:.4f}\n")
                f.write(f"   MRR: {row['mrr_mean']:.4f}\n")
                f.write(f"   WPRA_top3: {row['wpra_top3_mean']:.4f}\n")
                f.write(f"   Spearman: {row['spearman_mean']:.4f}\n")
        
        # Overall best
        f.write("\n" + "="*80 + "\n")
        f.write("Overall Best Configuration (by Top-1)\n")
        f.write("="*80 + "\n\n")
        
        best = summary_df.loc[summary_df['top1_agreement'].idxmax()]
        f.write(f"Config: {best['config']}\n")
        f.write(f"Step: {best['step']}\n")
        f.write(f"Learning Rate: {best['learning_rate']}\n")
        f.write(f"Lambda: {best['lambda_stabilize']}\n")
        f.write(f"Epsilon: {best['epsilon']}\n\n")
        
        f.write("All Metrics:\n")
        f.write(f"  Top-1: {best['top1_agreement']:.4f} ({best['top1_count']}/{int(best['n_images'])})\n")
        f.write(f"  Top-3: {best['top3_agreement']:.4f} ± {best['top3_std']:.4f}\n")
        f.write(f"  Top-5: {best['top5_agreement']:.4f} ± {best['top5_std']:.4f}\n")
        f.write(f"  Spearman: {best['spearman_mean']:.4f} ± {best['spearman_std']:.4f} (median: {best['spearman_median']:.4f})\n")
        f.write(f"  Pearson: {best['pearson_mean']:.4f} ± {best['pearson_std']:.4f} (median: {best['pearson_median']:.4f})\n")
        f.write(f"  Kendall: {best['kendall_mean']:.4f} ± {best['kendall_std']:.4f} (median: {best['kendall_median']:.4f})\n")
        f.write(f"  NDCG@3: {best['ndcg3_mean']:.4f} ± {best['ndcg3_std']:.4f} (median: {best['ndcg3_median']:.4f})\n")
        f.write(f"  NDCG@5: {best['ndcg5_mean']:.4f} ± {best['ndcg5_std']:.4f} (median: {best['ndcg5_median']:.4f})\n")
        f.write(f"  NDCG@10: {best['ndcg10_mean']:.4f} ± {best['ndcg10_std']:.4f} (median: {best['ndcg10_median']:.4f})\n")
        f.write(f"  MRR: {best['mrr_mean']:.4f} ± {best['mrr_std']:.4f} (median: {best['mrr_median']:.4f})\n")
        f.write(f"  WPRA_top3: {best['wpra_top3_mean']:.4f} ± {best['wpra_top3_std']:.4f} (median: {best['wpra_top3_median']:.4f})\n")
        f.write(f"  RBO(0.9): {best['rbo_mean']:.4f} ± {best['rbo_std']:.4f} (median: {best['rbo_median']:.4f})\n")
    
    print(f"Saved report to {report_file}")


if __name__ == "__main__":
    main()
