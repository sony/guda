#!/usr/bin/env python3
"""
Merge attribution scores from parallel array jobs.

This script merges CSV files from array jobs into a single output file.
It handles LOGOA, UNA, and CLIPA score files.
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Merge attribution score CSV files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing partial CSV files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output merged CSV file path"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match (e.g., 'logoa_scores_*.csv')"
    )
    parser.add_argument(
        "--sort_by",
        nargs='+',
        default=None,
        help="Columns to sort by (default: generated_style, prompt_type, image_id, attribution_style)"
    )
    return parser.parse_args()


def merge_csv_files(
    input_dir: Path,
    output_file: Path,
    pattern: str = "*.csv",
    sort_by: List[str] = None
) -> None:
    """
    Merge multiple CSV files into one.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to output merged file
        pattern: Glob pattern for input files
        sort_by: Columns to sort by
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    # Find all matching CSV files
    csv_files = sorted(input_dir.glob(pattern))
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {input_dir}")
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()
    
    # Read all CSV files
    dfs = []
    total_rows = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        total_rows += len(df)
        print(f"  {csv_file.name}: {len(df)} rows")
    
    print(f"\nTotal rows before merge: {total_rows}")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(merged_df)}")
    
    # Check for duplicates
    duplicate_cols = [
        col
        for col in merged_df.columns
        if col not in ['score', 'logoa_score', 'una_score', 'clipa_score', 'soup_score']
    ]
    if duplicate_cols:
        duplicates = merged_df.duplicated(subset=duplicate_cols, keep=False)
        if duplicates.any():
            print(f"\nWARNING: Found {duplicates.sum()} duplicate rows!")
            # Show some examples
            dup_df = merged_df[duplicates].head(10)
            print("Example duplicates:")
            print(dup_df)
    
    # Sort if requested
    if sort_by is None:
        # Default sorting for attribution scores
        if 'generated_style' in merged_df.columns:
            sort_by = ['generated_style', 'prompt_type', 'image_id', 'attribution_style']
        else:
            sort_by = list(merged_df.columns[:4])  # Sort by first 4 columns
    
    if sort_by:
        # Only use columns that exist
        sort_cols = [col for col in sort_by if col in merged_df.columns]
        if sort_cols:
            print(f"\nSorting by: {sort_cols}")
            merged_df = merged_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged file
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged CSV saved to: {output_file}")
    print(f"Final row count: {len(merged_df)}")
    print(f"Columns: {list(merged_df.columns)}")
    
    # Show summary statistics
    if 'attribution_style' in merged_df.columns:
        n_attribution_styles = merged_df['attribution_style'].nunique()
        print(f"\nNumber of unique attribution styles: {n_attribution_styles}")
    
    if 'generated_style' in merged_df.columns:
        n_generated_styles = merged_df['generated_style'].nunique()
        print(f"Number of unique generated styles: {n_generated_styles}")
    
    if 'prompt_type' in merged_df.columns:
        n_prompts = merged_df['prompt_type'].nunique()
        print(f"Number of prompt types: {n_prompts}")
    
    if 'image_id' in merged_df.columns:
        images_per_style_prompt = merged_df.groupby(['generated_style', 'prompt_type'])['image_id'].nunique().mean()
        print(f"Average images per style/prompt: {images_per_style_prompt:.1f}")


def main():
    args = parse_args()
    
    merge_csv_files(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file),
        pattern=args.pattern,
        sort_by=args.sort_by
    )


if __name__ == "__main__":
    main()
