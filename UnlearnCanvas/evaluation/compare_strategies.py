"""
Compare correlation performance between baseline (style_removed) and weighted_style_select strategies.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(os.environ.get("GUDA_UNLEARNCANVAS_ROOT", ".")).resolve()
BASELINE_SUMMARY = BASE_DIR / "outputs" / "param_sweep_analysis" / "parameter_sweep_summary.csv"
WEIGHTED_SUMMARY = BASE_DIR / "outputs" / "weighted_select_analysis" / "parameter_sweep_summary.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "strategy_comparison"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_baseline_summary():
    """Load and format baseline summary."""
    df = pd.read_csv(BASELINE_SUMMARY)
    # Extract config name from param_id
    df['config'] = df['param_id']
    df['strategy'] = 'style_removed'
    # Use spearman_mean as the correlation metric
    df['spearman_corr'] = df['spearman_mean']
    df['pearson_corr'] = df['pearson_mean']
    return df[['config', 'step', 'spearman_corr', 'pearson_corr', 'strategy']]

def load_weighted_summary():
    """Load and format weighted_style_select summary."""
    df = pd.read_csv(WEIGHTED_SUMMARY)
    # Remove 'ws_' prefix for consistency
    df['config'] = df['config'].str.replace('ws_', '')
    df['strategy'] = 'weighted_style_select'
    return df[['config', 'step', 'spearman_corr', 'pearson_corr', 'strategy']]

def main():
    print("="*80)
    print("Strategy Comparison: style_removed vs weighted_style_select")
    print("="*80)
    
    # Load both summaries
    baseline_df = load_baseline_summary()
    weighted_df = load_weighted_summary()
    
    print(f"\nBaseline configs: {baseline_df['config'].unique()}")
    print(f"Weighted configs: {weighted_df['config'].unique()}")
    
    # Merge on config and step
    comparison_df = pd.merge(
        baseline_df,
        weighted_df,
        on=['config', 'step'],
        suffixes=('_baseline', '_weighted')
    )
    
    # Compute differences
    comparison_df['spearman_diff'] = comparison_df['spearman_corr_weighted'] - comparison_df['spearman_corr_baseline']
    comparison_df['pearson_diff'] = comparison_df['pearson_corr_weighted'] - comparison_df['pearson_corr_baseline']
    
    # Save comparison
    comparison_file = OUTPUT_DIR / "strategy_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nSaved comparison to {comparison_file}")
    
    # Generate report
    generate_report(baseline_df, weighted_df, comparison_df)
    
    # Create visualizations
    create_visualizations(baseline_df, weighted_df, comparison_df)

def generate_report(baseline_df, weighted_df, comparison_df):
    """Generate comparison report."""
    report_file = OUTPUT_DIR / "strategy_comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Strategy Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("Baseline Strategy: style_removed (anchor uses random object + random retain style)\n")
        f.write("New Strategy: weighted_style_select (anchor uses probabilistic CLIP-based style selection)\n")
        f.write("Fixed Hyperparameters: beta=2.0, eta_uniform=0.3\n\n")
        
        # Overall statistics
        f.write("="*80 + "\n")
        f.write("Overall Statistics (All Configs, All Steps)\n")
        f.write("="*80 + "\n\n")
        
        baseline_mean = baseline_df['spearman_corr'].mean()
        weighted_mean = weighted_df['spearman_corr'].mean()
        
        f.write(f"Baseline:\n")
        f.write(f"  Mean Spearman: {baseline_mean:.4f}\n")
        f.write(f"  Std Spearman: {baseline_df['spearman_corr'].std():.4f}\n")
        f.write(f"  Max Spearman: {baseline_df['spearman_corr'].max():.4f}\n\n")
        
        f.write(f"Weighted Style Select:\n")
        f.write(f"  Mean Spearman: {weighted_mean:.4f}\n")
        f.write(f"  Std Spearman: {weighted_df['spearman_corr'].std():.4f}\n")
        f.write(f"  Max Spearman: {weighted_df['spearman_corr'].max():.4f}\n\n")
        
        f.write(f"Difference (Weighted - Baseline):\n")
        f.write(f"  Mean: {weighted_mean - baseline_mean:.4f}\n")
        f.write(f"  Median: {comparison_df['spearman_diff'].median():.4f}\n\n")
        
        # Best configurations
        f.write("="*80 + "\n")
        f.write("Best Configurations by Step\n")
        f.write("="*80 + "\n\n")
        
        for step in sorted(comparison_df['step'].unique()):
            f.write(f"\n--- Step {step} ---\n")
            step_df = comparison_df[comparison_df['step'] == step]
            
            # Best baseline
            best_baseline = step_df.loc[step_df['spearman_corr_baseline'].idxmax()]
            f.write(f"Best Baseline: {best_baseline['config']}\n")
            f.write(f"  Spearman: {best_baseline['spearman_corr_baseline']:.4f}\n")
            
            # Best weighted
            best_weighted = step_df.loc[step_df['spearman_corr_weighted'].idxmax()]
            f.write(f"Best Weighted: {best_weighted['config']}\n")
            f.write(f"  Spearman: {best_weighted['spearman_corr_weighted']:.4f}\n")
            
            # Improvement
            if best_weighted['config'] == best_baseline['config']:
                improvement = best_weighted['spearman_corr_weighted'] - best_weighted['spearman_corr_baseline']
                f.write(f"Same config - Improvement: {improvement:+.4f}\n")
            else:
                f.write(f"Different best configs\n")
        
        # Pairwise comparison (same config, same step)
        f.write("\n" + "="*80 + "\n")
        f.write("Pairwise Comparison (Same Config, Same Step)\n")
        f.write("="*80 + "\n\n")
        
        wins_weighted = (comparison_df['spearman_diff'] > 0).sum()
        wins_baseline = (comparison_df['spearman_diff'] < 0).sum()
        ties = (comparison_df['spearman_diff'] == 0).sum()
        
        f.write(f"Total comparisons: {len(comparison_df)}\n")
        f.write(f"Weighted wins: {wins_weighted} ({wins_weighted/len(comparison_df)*100:.1f}%)\n")
        f.write(f"Baseline wins: {wins_baseline} ({wins_baseline/len(comparison_df)*100:.1f}%)\n")
        f.write(f"Ties: {ties}\n\n")
        
        # Largest improvements and degradations
        f.write("Largest Improvements (Weighted better):\n")
        top_improvements = comparison_df.nlargest(5, 'spearman_diff')
        for _, row in top_improvements.iterrows():
            f.write(f"  {row['config']} Step {row['step']}: {row['spearman_diff']:+.4f} ")
            f.write(f"({row['spearman_corr_baseline']:.4f} → {row['spearman_corr_weighted']:.4f})\n")
        
        f.write("\nLargest Degradations (Baseline better):\n")
        top_degradations = comparison_df.nsmallest(5, 'spearman_diff')
        for _, row in top_degradations.iterrows():
            f.write(f"  {row['config']} Step {row['step']}: {row['spearman_diff']:+.4f} ")
            f.write(f"({row['spearman_corr_baseline']:.4f} → {row['spearman_corr_weighted']:.4f})\n")
    
    print(f"Saved report to {report_file}")

def create_visualizations(baseline_df, weighted_df, comparison_df):
    """Create comparison visualizations."""
    
    # 1. Side-by-side comparison across steps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spearman correlation
    ax = axes[0]
    configs = sorted(comparison_df['config'].unique())
    for config in configs:
        baseline_data = baseline_df[baseline_df['config'] == config]
        weighted_data = weighted_df[weighted_df['config'] == config]
        
        ax.plot(baseline_data['step'], baseline_data['spearman_corr'], 
               marker='o', linestyle='--', alpha=0.6, label=f'{config} (baseline)')
        ax.plot(weighted_data['step'], weighted_data['spearman_corr'], 
               marker='s', linestyle='-', label=f'{config} (weighted)')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Spearman Correlation: Baseline vs Weighted Style Select', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Difference heatmap
    ax = axes[1]
    pivot = comparison_df.pivot(index='config', columns='step', values='spearman_diff')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
               ax=ax, cbar_kws={'label': 'Spearman Diff (Weighted - Baseline)'})
    ax.set_title('Correlation Difference Heatmap', fontsize=14)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "strategy_comparison_spearman.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {plot_file}")
    
    # 2. Bar plot of best configs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = sorted(comparison_df['step'].unique())
    x = np.arange(len(steps))
    width = 0.35
    
    baseline_best = [comparison_df[comparison_df['step'] == s]['spearman_corr_baseline'].max() 
                     for s in steps]
    weighted_best = [comparison_df[comparison_df['step'] == s]['spearman_corr_weighted'].max() 
                     for s in steps]
    
    ax.bar(x - width/2, baseline_best, width, label='Baseline (best)', alpha=0.8)
    ax.bar(x + width/2, weighted_best, width, label='Weighted (best)', alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Best Spearman Correlation', fontsize=12)
    ax.set_title('Best Configuration Performance by Step', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / "strategy_comparison_best.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {plot_file}")

if __name__ == "__main__":
    main()
