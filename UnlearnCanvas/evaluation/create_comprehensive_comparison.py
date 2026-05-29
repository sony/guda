#!/usr/bin/env python3
"""
Create comparison summary table for baseline vs weighted_style_select.
Formats results in the style of ffsd_vr_comprehensive_comparison.md table.
"""
import os
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(os.environ.get("GUDA_UNLEARNCANVAS_ROOT", ".")).resolve()
BASELINE_CSV = BASE_DIR / "outputs" / "param_sweep_comprehensive" / "comprehensive_summary.csv"
WEIGHTED_CSV = BASE_DIR / "outputs" / "weighted_select_comprehensive" / "comprehensive_summary.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "strategy_comparison"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_and_find_best(csv_file, strategy_name):
    """Load summary and find best configurations for each key metric."""
    df = pd.read_csv(csv_file)
    
    # Key metrics to optimize
    key_metrics = {
        'top1_agreement': 'max',
        'top3_agreement': 'max',
        'ndcg3_mean': 'max',
        'mrr_mean': 'max',
        'wpra_top3_mean': 'max',
        'spearman_mean': 'max'
    }
    
    results = {}
    
    # Find best overall for each metric
    for metric, direction in key_metrics.items():
        if direction == 'max':
            best_row = df.loc[df[metric].idxmax()]
        else:
            best_row = df.loc[df[metric].idxmin()]
        
        results[f'{metric}_best'] = {
            'config': best_row['config'],
            'step': int(best_row['step']),
            'value': best_row[metric],
            'learning_rate': best_row['learning_rate'],
            'lambda': best_row['lambda_stabilize'],
            'epsilon': best_row['epsilon']
        }
    
    # Find overall best config (by top1)
    best_overall = df.loc[df['top1_agreement'].idxmax()]
    
    results['overall_best'] = {
        'config': best_overall['config'],
        'step': int(best_overall['step']),
        'top1_agreement': best_overall['top1_agreement'],
        'top1_count': int(best_overall['top1_count']),
        'top3_agreement': best_overall['top3_agreement'],
        'top3_std': best_overall['top3_std'],
        'top5_agreement': best_overall['top5_agreement'],
        'top5_std': best_overall['top5_std'],
        'spearman_mean': best_overall['spearman_mean'],
        'spearman_std': best_overall['spearman_std'],
        'spearman_median': best_overall['spearman_median'],
        'pearson_mean': best_overall['pearson_mean'],
        'pearson_std': best_overall['pearson_std'],
        'pearson_median': best_overall['pearson_median'],
        'kendall_mean': best_overall['kendall_mean'],
        'kendall_std': best_overall['kendall_std'],
        'kendall_median': best_overall['kendall_median'],
        'ndcg3_mean': best_overall['ndcg3_mean'],
        'ndcg3_std': best_overall['ndcg3_std'],
        'ndcg3_median': best_overall['ndcg3_median'],
        'ndcg5_mean': best_overall['ndcg5_mean'],
        'ndcg5_std': best_overall['ndcg5_std'],
        'ndcg5_median': best_overall['ndcg5_median'],
        'ndcg10_mean': best_overall['ndcg10_mean'],
        'ndcg10_std': best_overall['ndcg10_std'],
        'ndcg10_median': best_overall['ndcg10_median'],
        'mrr_mean': best_overall['mrr_mean'],
        'mrr_std': best_overall['mrr_std'],
        'mrr_median': best_overall['mrr_median'],
        'wpra_top3_mean': best_overall['wpra_top3_mean'],
        'wpra_top3_std': best_overall['wpra_top3_std'],
        'wpra_top3_median': best_overall['wpra_top3_median'],
        'rbo_mean': best_overall['rbo_mean'],
        'rbo_std': best_overall['rbo_std'],
        'rbo_median': best_overall['rbo_median'],
        'n_images': int(best_overall['n_images'])
    }
    
    return results, df

def generate_comparison_table(baseline_results, weighted_results, output_dir):
    """Generate markdown table comparing baseline vs weighted."""
    report_file = output_dir / "comprehensive_comparison_report.md"
    
    baseline_best = baseline_results['overall_best']
    weighted_best = weighted_results['overall_best']
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive Comparison: Baseline vs Weighted Style Select\n\n")
        f.write(f"**Date**: 2026-01-23\n")
        f.write(f"**Dataset**: FFSD Very Relaxed (320 images, 16 styles)\n")
        f.write(f"**LOGOA**: Gold standard for attribution\n\n")
        
        f.write("## Best Overall Configurations (by Top-1 Agreement)\n\n")
        
        # Baseline best
        f.write("### Baseline (style_removed)\n")
        f.write(f"- **Config**: {baseline_best['config']}\n")
        f.write(f"- **Step**: {baseline_best['step']}\n")
        f.write(f"- **Top-1**: {baseline_best['top1_agreement']:.4f} ({baseline_best['top1_count']}/{baseline_best['n_images']})\n\n")
        
        # Weighted best
        f.write("### Weighted Style Select\n")
        f.write(f"- **Config**: {weighted_best['config']}\n")
        f.write(f"- **Step**: {weighted_best['step']}\n")
        f.write(f"- **Top-1**: {weighted_best['top1_agreement']:.4f} ({weighted_best['top1_count']}/{weighted_best['n_images']})\n\n")
        
        f.write("---\n\n")
        f.write("## Detailed Metrics Comparison\n\n")
        f.write("### Table: All Metrics (ffsd_vr_comprehensive_comparison.md format)\n\n")
        
        # Create comparison table
        f.write("| Metric | Baseline (style_removed) | Weighted (weighted_style_select) | Winner | Notes |\n")
        f.write("|--------|--------------------------|----------------------------------|--------|-------|\n")
        
        # Top-K Agreement
        f.write("| **Top-K Agreement** | | | | |\n")
        
        baseline_top1 = baseline_best['top1_agreement']
        weighted_top1 = weighted_best['top1_agreement']
        winner_top1 = "Baseline" if baseline_top1 > weighted_top1 else ("Weighted" if weighted_top1 > baseline_top1 else "Tie")
        f.write(f"| Top-1 | **{baseline_top1:.4f}** | {weighted_top1:.4f} | {winner_top1} | {baseline_best['top1_count']} vs {weighted_best['top1_count']}/320 |\n")
        
        baseline_top3 = baseline_best['top3_agreement']
        weighted_top3 = weighted_best['top3_agreement']
        winner_top3 = "Baseline" if baseline_top3 > weighted_top3 else ("Weighted" if weighted_top3 > baseline_top3 else "Tie")
        f.write(f"| Top-3 | **{baseline_top3:.4f}** | {weighted_top3:.4f} | {winner_top3} | |\n")
        
        baseline_top5 = baseline_best['top5_agreement']
        weighted_top5 = weighted_best['top5_agreement']
        winner_top5 = "Baseline" if baseline_top5 > weighted_top5 else ("Weighted" if weighted_top5 > baseline_top5 else "Tie")
        f.write(f"| Top-5 | **{baseline_top5:.4f}** | {weighted_top5:.4f} | {winner_top5} | |\n")
        
        # Correlation
        f.write("| **Correlation** | | | | |\n")
        
        baseline_spearman = baseline_best['spearman_mean']
        weighted_spearman = weighted_best['spearman_mean']
        winner_spearman = "Baseline" if baseline_spearman > weighted_spearman else ("Weighted" if weighted_spearman > baseline_spearman else "Tie")
        f.write(f"| Spearman ρ | **{baseline_spearman:.4f}** | {weighted_spearman:.4f} | {winner_spearman} | median: {baseline_best['spearman_median']:.4f} vs {weighted_best['spearman_median']:.4f} |\n")
        
        baseline_pearson = baseline_best['pearson_mean']
        weighted_pearson = weighted_best['pearson_mean']
        winner_pearson = "Baseline" if baseline_pearson > weighted_pearson else ("Weighted" if weighted_pearson > baseline_pearson else "Tie")
        f.write(f"| Pearson r | **{baseline_pearson:.4f}** | {weighted_pearson:.4f} | {winner_pearson} | |\n")
        
        baseline_kendall = baseline_best['kendall_mean']
        weighted_kendall = weighted_best['kendall_mean']
        winner_kendall = "Baseline" if baseline_kendall > weighted_kendall else ("Weighted" if weighted_kendall > baseline_kendall else "Tie")
        f.write(f"| Kendall τ | **{baseline_kendall:.4f}** | {weighted_kendall:.4f} | {winner_kendall} | |\n")
        
        # NDCG
        f.write("| **NDCG (Top-weighted)** | | | | |\n")
        
        baseline_ndcg3 = baseline_best['ndcg3_mean']
        weighted_ndcg3 = weighted_best['ndcg3_mean']
        winner_ndcg3 = "Baseline" if baseline_ndcg3 > weighted_ndcg3 else ("Weighted" if weighted_ndcg3 > baseline_ndcg3 else "Tie")
        f.write(f"| NDCG@3 | **{baseline_ndcg3:.4f}** | {weighted_ndcg3:.4f} | {winner_ndcg3} | |\n")
        
        baseline_ndcg5 = baseline_best['ndcg5_mean']
        weighted_ndcg5 = weighted_best['ndcg5_mean']
        winner_ndcg5 = "Baseline" if baseline_ndcg5 > weighted_ndcg5 else ("Weighted" if weighted_ndcg5 > baseline_ndcg5 else "Tie")
        f.write(f"| NDCG@5 | **{baseline_ndcg5:.4f}** | {weighted_ndcg5:.4f} | {winner_ndcg5} | |\n")
        
        baseline_ndcg10 = baseline_best['ndcg10_mean']
        weighted_ndcg10 = weighted_best['ndcg10_mean']
        winner_ndcg10 = "Baseline" if baseline_ndcg10 > weighted_ndcg10 else ("Weighted" if weighted_ndcg10 > baseline_ndcg10 else "Tie")
        f.write(f"| NDCG@10 | **{baseline_ndcg10:.4f}** | {weighted_ndcg10:.4f} | {winner_ndcg10} | |\n")
        
        # Other Ranking
        f.write("| **Other Ranking** | | | | |\n")
        
        baseline_mrr = baseline_best['mrr_mean']
        weighted_mrr = weighted_best['mrr_mean']
        winner_mrr = "Baseline" if baseline_mrr > weighted_mrr else ("Weighted" if weighted_mrr > baseline_mrr else "Tie")
        f.write(f"| MRR (mean) | **{baseline_mrr:.4f}** | {weighted_mrr:.4f} | {winner_mrr} | median: {baseline_best['mrr_median']:.4f} vs {weighted_best['mrr_median']:.4f} |\n")
        
        baseline_wpra = baseline_best['wpra_top3_mean']
        weighted_wpra = weighted_best['wpra_top3_mean']
        winner_wpra = "Baseline" if baseline_wpra > weighted_wpra else ("Weighted" if weighted_wpra > baseline_wpra else "Tie")
        f.write(f"| WPRA_top3 | **{baseline_wpra:.4f}** | {weighted_wpra:.4f} | {winner_wpra} | |\n")
        
        baseline_rbo = baseline_best['rbo_mean']
        weighted_rbo = weighted_best['rbo_mean']
        winner_rbo = "Baseline" if baseline_rbo > weighted_rbo else ("Weighted" if weighted_rbo > baseline_rbo else "Tie")
        f.write(f"| RBO (p=0.9) | **{baseline_rbo:.4f}** | {weighted_rbo:.4f} | {winner_rbo} | |\n")
        
        f.write("\n---\n\n")
        f.write("## Best Configurations by Key Metric\n\n")
        
        # For each key metric, show best config from both strategies
        metrics_info = [
            ('top1_agreement', 'Top-1 Agreement', 'higher'),
            ('top3_agreement', 'Top-3 Agreement', 'higher'),
            ('ndcg3_mean', 'NDCG@3', 'higher'),
            ('mrr_mean', 'MRR', 'higher'),
            ('wpra_top3_mean', 'WPRA_top3', 'higher'),
            ('spearman_mean', 'Spearman ρ', 'higher')
        ]
        
        for metric_key, metric_name, _ in metrics_info:
            f.write(f"### {metric_name}\n\n")
            
            baseline_best_for_metric = baseline_results[f'{metric_key}_best']
            weighted_best_for_metric = weighted_results[f'{metric_key}_best']
            
            f.write(f"**Baseline**: {baseline_best_for_metric['config']} @ Step {baseline_best_for_metric['step']}\n")
            f.write(f"- Value: {baseline_best_for_metric['value']:.4f}\n")
            f.write(f"- LR: {baseline_best_for_metric['learning_rate']}, λ: {baseline_best_for_metric['lambda']}, ε: {baseline_best_for_metric['epsilon']}\n\n")
            
            f.write(f"**Weighted**: {weighted_best_for_metric['config']} @ Step {weighted_best_for_metric['step']}\n")
            f.write(f"- Value: {weighted_best_for_metric['value']:.4f}\n")
            f.write(f"- LR: {weighted_best_for_metric['learning_rate']}, λ: {weighted_best_for_metric['lambda']}, ε: {weighted_best_for_metric['epsilon']}\n\n")
            
            winner_value = "Baseline" if baseline_best_for_metric['value'] > weighted_best_for_metric['value'] else ("Weighted" if weighted_best_for_metric['value'] > baseline_best_for_metric['value'] else "Tie")
            diff = baseline_best_for_metric['value'] - weighted_best_for_metric['value']
            f.write(f"**Winner**: {winner_value} (diff: {diff:+.4f})\n\n")
            f.write("---\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Top-1 Performance**: ")
        if baseline_top1 > weighted_top1:
            f.write(f"Baseline wins by {(baseline_top1-weighted_top1)*100:.2f}%\n")
        elif weighted_top1 > baseline_top1:
            f.write(f"Weighted wins by {(weighted_top1-baseline_top1)*100:.2f}%\n")
        else:
            f.write("Tie\n")
        
        f.write(f"2. **Best Learning Rate**: Both strategies perform best with LR={baseline_best['config'].split('_')[0]}/Weighted={weighted_best['config'].split('_')[1]}\n")
        f.write(f"3. **Best Training Step**: Baseline peaks at step {baseline_best['step']}, Weighted peaks at step {weighted_best['step']}\n")
        f.write(f"4. **Overall Winner**: ")
        if baseline_top1 >= weighted_top1:
            f.write("Baseline (style_removed) achieves slightly better Top-1 accuracy\n")
        else:
            f.write("Weighted (weighted_style_select) achieves slightly better Top-1 accuracy\n")
    
    print(f"Saved comparison report to {report_file}")
    return report_file

def main():
    print("="*80)
    print("Creating Comprehensive Comparison Summary")
    print("="*80)
    
    # Load both summaries
    baseline_results, baseline_df = load_and_find_best(BASELINE_CSV, "baseline")
    weighted_results, weighted_df = load_and_find_best(WEIGHTED_CSV, "weighted")
    
    print(f"\nBaseline best Top-1: {baseline_results['overall_best']['top1_agreement']:.4f}")
    print(f"  Config: {baseline_results['overall_best']['config']} @ Step {baseline_results['overall_best']['step']}")
    
    print(f"\nWeighted best Top-1: {weighted_results['overall_best']['top1_agreement']:.4f}")
    print(f"  Config: {weighted_results['overall_best']['config']} @ Step {weighted_results['overall_best']['step']}")
    
    # Generate comparison report
    report_file = generate_comparison_table(baseline_results, weighted_results, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print(f"Comparison report saved: {report_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
