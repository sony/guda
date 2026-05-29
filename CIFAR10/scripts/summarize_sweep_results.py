#!/usr/bin/env python3
"""
Summarize hyperparameter sweep results.

Parse correlation analysis outputs from all configs and epochs,
generate summary tables and heatmaps showing performance across
hyperparameter space.

Usage:
    python scripts/summarize_sweep_results.py --method retrack --job_id 275500
    python scripts/summarize_sweep_results.py --method esd --job_id 275501
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Parameter grids
RETRACK_LAMBDA = [0.003, 0.01, 0.03]
RETRACK_LR = [3e-5, 1e-5, 1e-4, 3e-4]

ESD_LAMBDA = [0.3, 1.0, 3.0]
ESD_LR = [1e-5, 3e-5, 1e-4, 3e-4]


def get_param_grid(method):
    """Get parameter grid for method"""
    if method == "retrack":
        return RETRACK_LAMBDA, RETRACK_LR
    elif method == "esd":
        return ESD_LAMBDA, ESD_LR
    else:
        raise ValueError(f"Unknown method: {method}")


def config_id_to_params(config_id, method):
    """Convert config ID to parameter values"""
    lambda_values, lr_values = get_param_grid(method)
    
    lambda_idx = config_id % len(lambda_values)
    lr_idx = config_id // len(lambda_values)
    
    return lambda_values[lambda_idx], lr_values[lr_idx]


def load_metrics_csv(csv_path):
    """Load advanced_ranking_metrics.csv and extract metrics"""
    try:
        df = pd.read_csv(csv_path)
        
        # Extract first row (assuming single method comparison)
        if len(df) == 0:
            return None
        
        row = df.iloc[0]
        
        metrics = {
            'spearman': row.get('spearman_mean', np.nan),
            'pearson': row.get('pearson_mean', np.nan),
            'ndcg@3': row.get('ndcg@3_mean', np.nan),
            'ndcg@5': row.get('ndcg@5_mean', np.nan),
            'ndcg@10': row.get('ndcg@10_mean', np.nan),
            'mrr': row.get('mrr_mean', np.nan),
            'wpra_top3': row.get('wpra_top3_mean', np.nan),
            'calib_mae': row.get('calib_mae_mean', np.nan),
            'rbo': row.get('rbo_mean', np.nan)
        }
        
        return metrics
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def collect_all_metrics(base_dir, method, num_configs=12, epochs=[10, 20, 30, 40, 50]):
    """Collect metrics from all configs and epochs"""
    results = []
    lambda_values, lr_values = get_param_grid(method)
    
    for config_id in range(num_configs):
        lambda_val, lr_val = config_id_to_params(config_id, method)
        
        for epoch in epochs:
            csv_path = Path(base_dir) / f"epoch_{epoch}" / "advanced_ranking_metrics.csv"
            
            if not csv_path.exists():
                print(f"Warning: Missing {csv_path}")
                continue
            
            metrics = load_metrics_csv(csv_path)
            
            if metrics is not None:
                results.append({
                    'config_id': config_id,
                    'epoch': epoch,
                    'lambda': lambda_val,
                    'lr': lr_val,
                    **metrics
                })
    
    return pd.DataFrame(results)


def create_heatmap(df, metric_name, method, output_path):
    """Create heatmap for a specific metric"""
    lambda_values, lr_values = get_param_grid(method)
    
    # Get final epoch data
    final_epoch = df['epoch'].max()
    df_final = df[df['epoch'] == final_epoch]
    
    # Create pivot table
    pivot = df_final.pivot(index='lr', columns='lambda', values=metric_name)
    
    # Reindex to ensure correct order
    pivot = pivot.reindex(index=lr_values, columns=lambda_values)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': metric_name},
        ax=ax
    )
    
    ax.set_title(f'{method.upper()} Hyperparameter Sweep: {metric_name} (Epoch {final_epoch})')
    ax.set_xlabel('λ_forget')
    ax.set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved heatmap: {output_path}")


def create_learning_curves(df, metric_name, method, output_path):
    """Create learning curves for top configs"""
    # Get best config at final epoch
    final_epoch = df['epoch'].max()
    df_final = df[df['epoch'] == final_epoch].copy()
    
    # Sort by metric (higher is better for most metrics, lower for calib_mae)
    ascending = (metric_name == 'calib_mae')
    df_final = df_final.sort_values(metric_name, ascending=ascending)
    
    # Top 5 configs
    top_configs = df_final.head(5)['config_id'].values
    
    # Plot learning curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for config_id in top_configs:
        df_config = df[df['config_id'] == config_id]
        lambda_val, lr_val = config_id_to_params(config_id, method)
        
        label = f'λ={lambda_val:.3f}, lr={lr_val:.0e}'
        ax.plot(df_config['epoch'], df_config[metric_name], marker='o', label=label)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{method.upper()}: {metric_name} Learning Curves (Top 5 Configs)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved learning curves: {output_path}")


def create_summary_table(df, method, output_path):
    """Create summary table of best configs per epoch"""
    epochs = sorted(df['epoch'].unique())
    
    summary_data = []
    
    for epoch in epochs:
        df_epoch = df[df['epoch'] == epoch].copy()
        
        # Sort by spearman (primary metric)
        df_epoch = df_epoch.sort_values('spearman', ascending=False)
        
        best = df_epoch.iloc[0]
        lambda_val, lr_val = config_id_to_params(int(best['config_id']), method)
        
        summary_data.append({
            'epoch': epoch,
            'config_id': int(best['config_id']),
            'lambda': lambda_val,
            'lr': lr_val,
            'spearman': best['spearman'],
            'ndcg@3': best['ndcg@3'],
            'mrr': best['mrr'],
            'wpra_top3': best['wpra_top3'],
            'calib_mae': best['calib_mae']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved summary table: {output_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Summarize hyperparameter sweep results")
    parser.add_argument("--method", type=str, required=True, choices=["retrack", "esd"])
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/cifar10_results_2048/hyperparam_sweep/summary")
    
    args = parser.parse_args()
    
    # Set base directory
    if args.base_dir is None:
        args.base_dir = f"results/cifar10_results_2048/hyperparam_sweep/{args.method}_job{args.job_id}"
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_path}")
        return
    
    output_path = Path(args.output_dir) / args.method
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Summarizing {args.method.upper()} Hyperparameter Sweep")
    print(f"{'='*80}")
    print(f"Base directory: {base_path}")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")
    
    # Collect all metrics
    print("Collecting metrics from all configs and epochs...")
    df = collect_all_metrics(base_path, args.method)
    
    if len(df) == 0:
        print("Error: No metrics collected")
        return
    
    print(f"✓ Collected {len(df)} data points")
    print(f"  Configs: {df['config_id'].nunique()}")
    print(f"  Epochs: {sorted(df['epoch'].unique())}")
    print()
    
    # Save full results
    full_results_path = output_path / f"{args.method}_full_results.csv"
    df.to_csv(full_results_path, index=False)
    print(f"✓ Saved full results: {full_results_path}\n")
    
    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(df, args.method, output_path / f"{args.method}_best_configs.csv")
    print()
    
    # Create heatmaps for key metrics
    print("Creating heatmaps...")
    key_metrics = ['spearman', 'ndcg@3', 'mrr', 'wpra_top3', 'calib_mae']
    
    for metric in key_metrics:
        heatmap_path = output_path / f"{args.method}_heatmap_{metric}.png"
        create_heatmap(df, metric, args.method, heatmap_path)
    
    print()
    
    # Create learning curves
    print("Creating learning curves...")
    for metric in key_metrics:
        curves_path = output_path / f"{args.method}_curves_{metric}.png"
        create_learning_curves(df, metric, args.method, curves_path)
    
    print()
    
    # Print best config
    print(f"{'='*80}")
    print("Best Configuration (by Spearman correlation)")
    print(f"{'='*80}")
    
    best = summary_df.iloc[-1]  # Last epoch
    print(f"Epoch: {int(best['epoch'])}")
    print(f"Config ID: {int(best['config_id'])}")
    print(f"λ_forget: {best['lambda']}")
    print(f"Learning Rate: {best['lr']:.0e}")
    print(f"\nMetrics:")
    print(f"  Spearman: {best['spearman']:.3f}")
    print(f"  NDCG@3: {best['ndcg@3']:.3f}")
    print(f"  MRR: {best['mrr']:.3f}")
    print(f"  WPRA_top3: {best['wpra_top3']:.3f}")
    print(f"  Calibrated MAE: {best['calib_mae']:.3f}")
    print(f"{'='*80}\n")
    
    print(f"✓ All visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()
