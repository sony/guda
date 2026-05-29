#!/usr/bin/env python3
"""
batch_metrics.py

Recursively find all *_sorted.csv under a base directory,
compute nDCG, Spearman's rho, Kendall's tau at k=3 and k=5,
and write all results into a single CSV file.
"""

import argparse
import csv
import glob
import os
from evaluation.utils import calculate_ndcg_metrics

def find_sorted_csv_files(base_dir):
    """Return sorted list of paths matching *_sorted.csv under base_dir."""
    pattern = os.path.join(base_dir, '**', '*_sorted.csv')
    return sorted(glob.glob(pattern, recursive=True))

def main():
    parser = argparse.ArgumentParser(
        description="Batch compute nDCG, Spearman's rho, Kendall's tau and save to CSV"
    )
    parser.add_argument('search_dir',
                        help='Base directory to search for *_sorted.csv')
    parser.add_argument('reference_csv',
                        help='Reference CSV file for calculate_ndcg_metrics')
    parser.add_argument('output_csv',
                        help='Output CSV filename for aggregated metrics')
    args = parser.parse_args()

    files = find_sorted_csv_files(args.search_dir)
    if not files:
        print("No *_sorted.csv files found.")
        return

    # Prepare CSV header
    header = [
        'sorted_csv_path',
        'k',
        'ndcg',
        'spearman_rho',
        'kendall_tau'
    ]

    with open(args.output_csv, 'w', newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)

        for path in files:
            # metrics for k=3
            res3 = calculate_ndcg_metrics(path, args.reference_csv, k=3)
            if isinstance(res3, dict):
                ndcg3 = res3.get('ndcg@k')
                rho3 = res3.get('spearman_value')
                tau3 = res3.get('kendall_value')
                writer.writerow([path, 3, f"{ndcg3:.4f}", f"{rho3:.4f}", f"{tau3:.4f}"])
            else:
                writer.writerow([path, 3, '', '', ''])

            # metrics for k=5
            res5 = calculate_ndcg_metrics(path, args.reference_csv, k=5)
            if isinstance(res5, dict):
                ndcg5 = res5.get('ndcg@k')
                rho5 = res5.get('spearman_value')
                tau5 = res5.get('kendall_value')
                writer.writerow([path, 5, f"{ndcg5:.4f}", f"{rho5:.4f}", f"{tau5:.4f}"])
            else:
                writer.writerow([path, 5, '', '', ''])

    print(f"All done. Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()