#!/usr/bin/env python3
"""
Plot comprehensive correlation analysis between two CSV files with attribution scores.
Shows rank correlations, metric value correlations, and metric distributions across ranks.
"""
import matplotlib.pyplot as plt
import csv
import os
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Plot comprehensive correlation analysis between two CSV files")
    parser.add_argument("--csv1", required=True, help="Reference CSV file path")
    parser.add_argument("--csv2", required=True, help="Comparison CSV file path")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to plot (default: all)")
    parser.add_argument("--start_image", type=int, default=0, help="Starting image index (default: 0)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as csv1)")
    return parser.parse_args()

def load_csv_data_with_scores(csv_path):
    """Load CSV data and return both rankings and score values for each image"""
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        headers = next(csv_reader)
        
        # Find class and score columns (like R1_Cls, R1_Sc, etc.)
        class_cols_indices = [i for i, col in enumerate(headers) if col.endswith('_Cls')]
        score_cols_indices = [i for i, col in enumerate(headers) if col.endswith('_Sc')]
        
        # Read all data and create rankings and scores
        data = []
        for row in csv_reader:
            # For each image, create ranking and score dictionaries: class_id -> rank/score
            ranking = {}
            scores = {}
            for rank_position, (cls_idx, score_idx) in enumerate(zip(class_cols_indices, score_cols_indices), 1):
                class_id = int(row[cls_idx])
                score_value = float(row[score_idx])
                ranking[class_id] = rank_position
                scores[class_id] = score_value
            data.append((ranking, scores))
    
    return data

def plot_rank_correlation(data1, data2, common_classes, actual_num_images, output_dir, csv1_name, csv2_name):
    """Plot rank correlation aggregated across multiple images"""
    
    # Collect data for each CSV1 rank position
    rank_data = defaultdict(list)  # csv1_rank -> list of csv2_ranks
    
    for img_idx in range(actual_num_images):
        ranking1, _ = data1[img_idx]
        ranking2, _ = data2[img_idx]
        
        # For each image, go through CSV1 ranks and find corresponding CSV2 ranks
        for csv1_rank in range(1, len(common_classes) + 1):
            # Find which class has this rank in CSV1
            class_at_rank = None
            for class_id, rank in ranking1.items():
                if rank == csv1_rank and class_id in common_classes:
                    class_at_rank = class_id
                    break
            
            # If we found the class, get its rank in CSV2
            if class_at_rank is not None and class_at_rank in ranking2:
                csv2_rank = ranking2[class_at_rank]
                rank_data[csv1_rank].append(csv2_rank)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    csv1_positions = []
    csv2_means = []
    csv2_stds = []
    all_csv1_ranks = []
    all_csv2_ranks = []
    
    for csv1_rank in sorted(rank_data.keys()):
        csv2_ranks = rank_data[csv1_rank]
        if csv2_ranks:
            csv1_positions.append(csv1_rank)
            csv2_means.append(np.mean(csv2_ranks))
            csv2_stds.append(np.std(csv2_ranks))
            
            # For scatter plot - add some jitter to x-axis for visibility
            jitter = np.random.normal(0, 0.1, len(csv2_ranks))
            plt.scatter([csv1_rank + j for j in jitter], csv2_ranks, 
                       alpha=0.4, s=20, color='lightblue', edgecolors='none')
            
            # Collect all data for overall correlation
            all_csv1_ranks.extend([csv1_rank] * len(csv2_ranks))
            all_csv2_ranks.extend(csv2_ranks)
    
    # Plot means with error bars
    plt.errorbar(csv1_positions, csv2_means, yerr=csv2_stds, 
                fmt='o-', color='red', markersize=6, linewidth=2, 
                capsize=5, capthick=2, label='Mean ± Std')
    
    # Add diagonal line (perfect correlation)
    max_rank = len(common_classes)
    plt.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.5, linewidth=1, label='Perfect correlation')
    
    # Set labels and title
    plt.xlabel('Rank Position in CSV1', fontsize=12)
    plt.ylabel('Rank in CSV2', fontsize=12)
    plt.title(f'Rank Correlation: {csv1_name} vs {csv2_name}\n({actual_num_images} images)', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set limits
    plt.xlim(0.5, max_rank + 0.5)
    plt.ylim(0.5, max_rank + 0.5)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "rank_correlation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation
    correlation = np.corrcoef(all_csv1_ranks, all_csv2_ranks)[0, 1] if all_csv1_ranks else 0
    return output_path, correlation

def plot_metric_correlation(data1, data2, common_classes, actual_num_images, output_dir, csv1_name, csv2_name):
    """Plot enhanced scatter plot with smooth density visualization and outlier handling"""
    
    # Collect all metric values
    csv1_metrics = []
    csv2_metrics = []
    
    for img_idx in range(actual_num_images):
        _, scores1 = data1[img_idx]
        _, scores2 = data2[img_idx]
        
        for class_id in common_classes:
            if class_id in scores1 and class_id in scores2:
                csv1_metrics.append(scores1[class_id])
                csv2_metrics.append(scores2[class_id])
    
    csv1_metrics = np.array(csv1_metrics)
    csv2_metrics = np.array(csv2_metrics)
    
    # Remove outliers using IQR method for all plots
    def remove_outliers_iqr(x, y):
        # Calculate IQR for both dimensions
        q75_x, q25_x = np.percentile(x, [75, 25])
        q75_y, q25_y = np.percentile(y, [75, 25])
        iqr_x = q75_x - q25_x
        iqr_y = q75_y - q25_y
        
        # Define outlier bounds
        lower_x, upper_x = q25_x - 1.5 * iqr_x, q75_x + 1.5 * iqr_x
        lower_y, upper_y = q25_y - 1.5 * iqr_y, q75_y + 1.5 * iqr_y
        
        # Filter data
        mask = (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
        return x[mask], y[mask], mask
    
    csv1_clean, csv2_clean, inlier_mask = remove_outliers_iqr(csv1_metrics, csv2_metrics)
    outliers_removed = len(csv1_metrics) - len(csv1_clean)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # 1. Smooth density plot with cleaned data (left)
    ax1 = plt.subplot(1, 2, 1)
    
    # Create smooth 2D density plot for cleaned data
    if len(csv1_clean) > 50:
        # Use contour plots for smooth density visualization
        from scipy.stats import gaussian_kde
        
        # Create a grid for density estimation
        x_min, x_max = csv1_clean.min(), csv1_clean.max()
        y_min, y_max = csv2_clean.min(), csv2_clean.max()
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Create grid
        grid_size = 50
        xx, yy = np.mgrid[x_min:x_max:complex(grid_size), y_min:y_max:complex(grid_size)]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Estimate density
        try:
            kernel = gaussian_kde(np.vstack([csv1_clean, csv2_clean]))
            density = np.reshape(kernel(positions).T, xx.shape)
            
            # Create contour plot
            contour = plt.contourf(xx, yy, density, levels=15, cmap='Blues', alpha=0.7)
            plt.colorbar(contour, ax=ax1, label='Density')
            
            # Add contour lines
            plt.contour(xx, yy, density, levels=8, colors='darkblue', alpha=0.4, linewidths=0.8)
            
        except Exception as e:
            print(f"Warning: Could not create density plot, falling back to scatter: {e}")
            plt.scatter(csv1_clean, csv2_clean, alpha=0.6, s=15, color='blue', edgecolors='darkblue', linewidth=0.3)
    else:
        # Use scatter for smaller datasets
        plt.scatter(csv1_clean, csv2_clean, alpha=0.6, s=20, color='blue', edgecolors='darkblue', linewidth=0.5)
    
    # Add trend line for cleaned data
    if len(csv1_clean) > 1:
        z_clean = np.polyfit(csv1_clean, csv2_clean, 1)
        p_clean = np.poly1d(z_clean)
        x_trend_clean = np.linspace(csv1_clean.min(), csv1_clean.max(), 100)
        plt.plot(x_trend_clean, p_clean(x_trend_clean), "red", alpha=0.9, linewidth=3, 
                label=f'Trend line (slope={z_clean[0]:.3f})')
    
    # Calculate correlation for cleaned data
    correlation_clean = np.corrcoef(csv1_clean, csv2_clean)[0, 1] if len(csv1_clean) > 1 else 0
    
    # Calculate correlation for all data
    correlation_all = np.corrcoef(csv1_metrics, csv2_metrics)[0, 1] if len(csv1_metrics) > 1 else 0
    
    # Add statistics text (only all data correlation)
    plt.text(0.05, 0.95, f'All data correlation: {correlation_all:.3f}\nN = {len(csv1_metrics)}\nOutliers removed: {outliers_removed}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.9),
             verticalalignment='top', fontsize=12, fontweight='bold')
    
    plt.xlabel(f'{csv1_name} Metric Values', fontsize=12)
    plt.ylabel(f'{csv2_name} Metric Values', fontsize=12)
    plt.title(f'Smooth Density Correlation (Outliers Removed): {csv1_name} vs {csv2_name}\n({actual_num_images} images)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    if len(csv1_clean) <= 50:  # Only show legend for scatter plot
        plt.legend()
    
    # 2. All data scatter plot with bounding box (right)
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot all data points
    plt.scatter(csv1_metrics, csv2_metrics, alpha=0.5, s=10, color='blue', 
               edgecolors='darkblue', linewidth=0.2, label='All data')
    
    # Add trend line for all data
    if len(csv1_metrics) > 1:
        z_all = np.polyfit(csv1_metrics, csv2_metrics, 1)
        p_all = np.poly1d(z_all)
        x_trend_all = np.linspace(csv1_metrics.min(), csv1_metrics.max(), 100)
        plt.plot(x_trend_all, p_all(x_trend_all), "red", alpha=0.9, linewidth=2, 
                label=f'Trend line (slope={z_all[0]:.3f})')
    
    # Add bounding box showing the clean data range (left plot range)
    if len(csv1_clean) > 0:
        x_min_clean, x_max_clean = csv1_clean.min(), csv1_clean.max()
        y_min_clean, y_max_clean = csv2_clean.min(), csv2_clean.max()
        
        # Create rectangle for bounding box
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_min_clean, y_min_clean), 
                        x_max_clean - x_min_clean, 
                        y_max_clean - y_min_clean,
                        linewidth=2, edgecolor='green', facecolor='none', 
                        linestyle='--', alpha=0.8, label='Clean data range')
        ax2.add_patch(rect)
    
    plt.text(0.05, 0.95, f'All data correlation: {correlation_all:.3f}\nN = {len(csv1_metrics)}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.9),
             verticalalignment='top', fontsize=12, fontweight='bold')
    
    plt.xlabel(f'{csv1_name} Metric Values', fontsize=12)
    plt.ylabel(f'{csv2_name} Metric Values', fontsize=12)
    plt.title(f'All Data with Clean Data Range', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    output_path = os.path.join(output_dir, "metric_correlation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, correlation_all

def plot_rank_vs_metric_values(data1, data2, common_classes, actual_num_images, output_dir, csv1_name, csv2_name):
    """Plot CSV1 ranks vs CSV2 metric values (not ranks)"""
    
    # Collect data for each CSV1 rank position
    rank_metric_data = defaultdict(list)  # csv1_rank -> list of csv2_metric_values
    
    for img_idx in range(actual_num_images):
        ranking1, _ = data1[img_idx]
        _, scores2 = data2[img_idx]
        
        # For each image, go through CSV1 ranks and find corresponding CSV2 metric values
        for csv1_rank in range(1, len(common_classes) + 1):
            # Find which class has this rank in CSV1
            class_at_rank = None
            for class_id, rank in ranking1.items():
                if rank == csv1_rank and class_id in common_classes:
                    class_at_rank = class_id
                    break
            
            # If we found the class, get its metric value in CSV2
            if class_at_rank is not None and class_at_rank in scores2:
                csv2_metric = scores2[class_at_rank]
                rank_metric_data[csv1_rank].append(csv2_metric)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    csv1_positions = []
    csv2_metric_means = []
    csv2_metric_stds = []
    all_csv1_ranks = []
    all_csv2_metrics = []
    
    for csv1_rank in sorted(rank_metric_data.keys()):
        csv2_metrics = rank_metric_data[csv1_rank]
        if csv2_metrics:
            csv1_positions.append(csv1_rank)
            csv2_metric_means.append(np.mean(csv2_metrics))
            csv2_metric_stds.append(np.std(csv2_metrics))
            
            # For scatter plot - add some jitter to x-axis for visibility
            jitter = np.random.normal(0, 0.1, len(csv2_metrics))
            plt.scatter([csv1_rank + j for j in jitter], csv2_metrics, 
                       alpha=0.4, s=20, color='lightgreen', edgecolors='none')
            
            # Collect all data for overall correlation
            all_csv1_ranks.extend([csv1_rank] * len(csv2_metrics))
            all_csv2_metrics.extend(csv2_metrics)
    
    # Plot means with error bars
    plt.errorbar(csv1_positions, csv2_metric_means, yerr=csv2_metric_stds, 
                fmt='o-', color='darkgreen', markersize=6, linewidth=2, 
                capsize=5, capthick=2, label='Mean ± Std')
    
    # Calculate correlation
    correlation = np.corrcoef(all_csv1_ranks, all_csv2_metrics)[0, 1] if all_csv1_ranks else 0
    
    # Add statistics text
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
             verticalalignment='top', fontsize=12)
    
    plt.xlabel('Rank Position in CSV1', fontsize=12)
    plt.ylabel(f'{csv2_name} Metric Values', fontsize=12)
    plt.title(f'CSV1 Ranks vs CSV2 Metric Values: {csv1_name} vs {csv2_name}\n({actual_num_images} images)', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "rank_vs_metric_values.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, correlation

def plot_comprehensive_analysis(csv1_path, csv2_path, num_images=None, start_image=0, output_dir=None):
    """Plot comprehensive correlation analysis"""
    
    # Load data from both CSV files
    print(f"Loading data from {csv1_path}...")
    data1 = load_csv_data_with_scores(csv1_path)
    
    print(f"Loading data from {csv2_path}...")
    data2 = load_csv_data_with_scores(csv2_path)
    
    # Find common classes between both CSVs
    if not data1 or not data2:
        raise ValueError("No data found in CSV files")
    
    common_classes = set(data1[0][0].keys()) & set(data2[0][0].keys())
    num_classes = len(common_classes)
    
    print(f"Found {num_classes} common classes between CSVs")
    
    if num_classes == 0:
        raise ValueError("No common classes found between the two CSV files")
    
    # Determine number of images to process
    max_images = min(len(data1), len(data2))
    if num_images is None:
        end_image = max_images
        actual_num_images = max_images - start_image
    else:
        end_image = min(start_image + num_images, max_images)
        actual_num_images = end_image - start_image
    
    # Slice data
    data1 = data1[start_image:end_image]
    data2 = data2[start_image:end_image]
    
    print(f"Processing {actual_num_images} images (indices {start_image} to {end_image-1})")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv1_path)
    
    csv1_name = os.path.basename(csv1_path).replace('.csv', '')
    csv2_name = os.path.basename(csv2_path).replace('.csv', '')
    
    # Create subdirectory for plots
    plot_dir = os.path.join(output_dir, f"corr_anal_{csv1_name}_vs_{csv2_name}")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Saving plots to {plot_dir}")
    
    # Generate all plots
    plots_info = []
    
    print("Generating rank correlation plot...")
    plot_path, corr = plot_rank_correlation(data1, data2, common_classes, actual_num_images, plot_dir, csv1_name, csv2_name)
    plots_info.append(("Rank Correlation", plot_path, corr))
    
    print("Generating metric correlation plot...")
    plot_path, corr = plot_metric_correlation(data1, data2, common_classes, actual_num_images, plot_dir, csv1_name, csv2_name)
    plots_info.append(("Metric Correlation", plot_path, corr))
    
    print("Generating rank vs metric values plot...")
    plot_path, corr = plot_rank_vs_metric_values(data1, data2, common_classes, actual_num_images, plot_dir, csv1_name, csv2_name)
    plots_info.append(("Rank vs Metric Values", plot_path, corr))
    
    # Summary
    print(f"\n{'='*60}")
    print("Correlation Analysis Complete!")
    print(f"{'='*60}")
    print(f"Analyzed {actual_num_images} images with {num_classes} common classes")
    print(f"Plots saved to: {plot_dir}")
    print("\nGenerated plots and correlations:")
    for plot_name, plot_path, correlation in plots_info:
        print(f"  - {plot_name}: {os.path.basename(plot_path)} (r={correlation:.3f})")
    
    return plot_dir

def main():
    args = get_args()
    
    # Check if CSV files exist
    if not os.path.exists(args.csv1):
        raise FileNotFoundError(f"CSV1 file not found: {args.csv1}")
    if not os.path.exists(args.csv2):
        raise FileNotFoundError(f"CSV2 file not found: {args.csv2}")
    
    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate comprehensive analysis
    plot_dir = plot_comprehensive_analysis(
        args.csv1, 
        args.csv2, 
        num_images=args.num_images,
        start_image=args.start_image,
        output_dir=args.output_dir
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
