"""
Analyze UNA-LOGOA correlation for weighted_style_select parameter sweep.
Similar to baseline param_sweep analysis but for weighted_select_sweep directory.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(os.environ.get("GUDA_UNLEARNCANVAS_ROOT", ".")).resolve()
WEIGHTED_SELECT_DIR = BASE_DIR / "outputs" / "weighted_select_sweep"
LOGOA_FILE = BASE_DIR / "outputs" / "logoa_scores_ffsd_very_relaxed" / "logoa_scores.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "weighted_select_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Parameter configurations
PARAM_CONFIGS = [
    "ws_lr5e7_lam1_eps0.05", "ws_lr5e7_lam2_eps0.1", "ws_lr5e7_lam3_eps0.2",
    "ws_lr1e6_lam1_eps0.05", "ws_lr1e6_lam2_eps0.1", "ws_lr1e6_lam3_eps0.2",
    "ws_lr2e6_lam1_eps0.05", "ws_lr2e6_lam2_eps0.1", "ws_lr2e6_lam3_eps0.2",
]

STEPS = [1000, 2000, 3000, 4000, 5000]

def load_logoa_scores():
    """Load LOGOA ground truth scores."""
    print(f"Loading LOGOA scores from {LOGOA_FILE}")
    df = pd.read_csv(LOGOA_FILE)
    print(f"Loaded {len(df)} LOGOA entries")
    return df

def merge_una_scores(config_dir, step):
    """Merge UNA scores across all styles for a given config and step."""
    all_scores = []
    style_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()])
    
    for style_dir in style_dirs:
        csv_file = style_dir / f"una_scores_step{step}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_scores.append(df)
        else:
            print(f"Warning: Missing {csv_file}")
    
    if all_scores:
        merged = pd.concat(all_scores, ignore_index=True)
        print(f"Merged {len(merged)} UNA scores from {len(all_scores)} styles")
        return merged
    else:
        return None

def compute_correlation(una_df, logoa_df):
    """Compute Spearman and Pearson correlation between UNA and LOGOA."""
    # Merge on (generated_style, object_name, attribution_style)
    merged = una_df.merge(
        logoa_df,
        on=['generated_style', 'object_name', 'attribution_style'],
        how='inner',
        suffixes=('_una', '_logoa')
    )
    
    if len(merged) == 0:
        print("Warning: No matching entries between UNA and LOGOA")
        return None, None, 0
    
    una_scores = merged['una_score'].values
    logoa_scores = merged['logoa_score'].values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(una_scores) | np.isnan(logoa_scores))
    una_scores = una_scores[valid_mask]
    logoa_scores = logoa_scores[valid_mask]
    
    if len(una_scores) < 2:
        return None, None, len(una_scores)
    
    spearman_corr, spearman_pval = spearmanr(una_scores, logoa_scores)
    pearson_corr, pearson_pval = pearsonr(una_scores, logoa_scores)
    
    return (spearman_corr, spearman_pval), (pearson_corr, pearson_pval), len(una_scores)

def main():
    print("="*80)
    print("Weighted Style Select Parameter Sweep Analysis")
    print("="*80)
    
    # Load LOGOA scores
    logoa_df = load_logoa_scores()
    
    # Results storage
    results = []
    
    # Analyze each parameter configuration
    for param_config in PARAM_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Analyzing configuration: {param_config}")
        print(f"{'='*80}")
        
        config_dir = WEIGHTED_SELECT_DIR / param_config
        if not config_dir.exists():
            print(f"Warning: Config directory not found: {config_dir}")
            continue
        
        for step in STEPS:
            print(f"\n--- Step {step} ---")
            
            # Merge UNA scores
            una_df = merge_una_scores(config_dir, step)
            if una_df is None:
                print(f"No UNA scores found for {param_config} at step {step}")
                continue
            
            # Save merged UNA scores
            merged_file = OUTPUT_DIR / f"{param_config}_step{step}_merged.csv"
            una_df.to_csv(merged_file, index=False)
            print(f"Saved merged UNA scores to {merged_file}")
            
            # Compute correlation
            spearman_result, pearson_result, n_samples = compute_correlation(una_df, logoa_df)
            
            if spearman_result is not None:
                spearman_corr, spearman_pval = spearman_result
                pearson_corr, pearson_pval = pearson_result
                
                print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")
                print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_pval:.4e})")
                print(f"Number of samples: {n_samples}")
                
                results.append({
                    'config': param_config,
                    'step': step,
                    'spearman_corr': spearman_corr,
                    'spearman_pval': spearman_pval,
                    'pearson_corr': pearson_corr,
                    'pearson_pval': pearson_pval,
                    'n_samples': n_samples
                })
            else:
                print("Correlation computation failed")
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        summary_file = OUTPUT_DIR / "parameter_sweep_summary.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"\n{'='*80}")
        print(f"Saved summary to {summary_file}")
        
        # Generate report
        generate_report(results_df)
        
        # Create visualization
        create_visualization(results_df)
    else:
        print("\nNo results to save")

def generate_report(results_df):
    """Generate a text report of the analysis."""
    report_file = OUTPUT_DIR / "parameter_sweep_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Weighted Style Select Parameter Sweep - Correlation Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("Strategy: weighted_style_select (CLIP-based probabilistic style selection)\n")
        f.write("Fixed Hyperparameters: beta=2.0, eta_uniform=0.3\n")
        f.write("Swept Hyperparameters: learning_rate, lambda_stabilize, epsilon\n\n")
        
        # Group by step
        for step in STEPS:
            step_df = results_df[results_df['step'] == step].copy()
            if len(step_df) == 0:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Step {step} Results\n")
            f.write(f"{'='*80}\n\n")
            
            # Sort by Spearman correlation
            step_df = step_df.sort_values('spearman_corr', ascending=False)
            
            # Best config
            best = step_df.iloc[0]
            f.write(f"Best Configuration:\n")
            f.write(f"  Config: {best['config']}\n")
            f.write(f"  Spearman: {best['spearman_corr']:.4f} (p={best['spearman_pval']:.4e})\n")
            f.write(f"  Pearson: {best['pearson_corr']:.4f} (p={best['pearson_pval']:.4e})\n")
            f.write(f"  Samples: {best['n_samples']}\n\n")
            
            # All configs
            f.write("All Configurations (sorted by Spearman):\n")
            f.write("-"*80 + "\n")
            for _, row in step_df.iterrows():
                f.write(f"  {row['config']:30s} | Spearman: {row['spearman_corr']:6.4f} | ")
                f.write(f"Pearson: {row['pearson_corr']:6.4f} | Samples: {row['n_samples']}\n")
            
            # Statistics
            f.write(f"\nStatistics:\n")
            f.write(f"  Mean Spearman: {step_df['spearman_corr'].mean():.4f}\n")
            f.write(f"  Std Spearman: {step_df['spearman_corr'].std():.4f}\n")
            f.write(f"  Median Spearman: {step_df['spearman_corr'].median():.4f}\n")
            f.write(f"  Min Spearman: {step_df['spearman_corr'].min():.4f}\n")
            f.write(f"  Max Spearman: {step_df['spearman_corr'].max():.4f}\n")
        
        # Overall best
        f.write(f"\n{'='*80}\n")
        f.write("Overall Best Configuration\n")
        f.write(f"{'='*80}\n")
        best_overall = results_df.loc[results_df['spearman_corr'].idxmax()]
        f.write(f"Config: {best_overall['config']}\n")
        f.write(f"Step: {best_overall['step']}\n")
        f.write(f"Spearman: {best_overall['spearman_corr']:.4f} (p={best_overall['spearman_pval']:.4e})\n")
        f.write(f"Pearson: {best_overall['pearson_corr']:.4f} (p={best_overall['pearson_pval']:.4e})\n")
        f.write(f"Samples: {best_overall['n_samples']}\n")
    
    print(f"Saved report to {report_file}")

def create_visualization(results_df):
    """Create visualization of correlation trends."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Spearman correlation
    ax = axes[0]
    for config in PARAM_CONFIGS:
        config_df = results_df[results_df['config'] == config]
        if len(config_df) > 0:
            ax.plot(config_df['step'], config_df['spearman_corr'], 
                   marker='o', label=config.replace('ws_', ''))
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Weighted Style Select: UNA-LOGOA Spearman Correlation', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot Pearson correlation
    ax = axes[1]
    for config in PARAM_CONFIGS:
        config_df = results_df[results_df['config'] == config]
        if len(config_df) > 0:
            ax.plot(config_df['step'], config_df['pearson_corr'], 
                   marker='o', label=config.replace('ws_', ''))
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Weighted Style Select: UNA-LOGOA Pearson Correlation', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "parameter_sweep_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {plot_file}")

if __name__ == "__main__":
    main()
