import torch
import math
import csv
import os
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau, rankdata

def get_class_names(num_classes: int) -> dict:
    """Dynamically get class names for CIFAR-10 or CIFAR-100."""
    if num_classes == 10:
        return {
            0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
    elif num_classes == 100:
        try:
            from torchvision.datasets import CIFAR100
            # Download once to get class names
            dataset = CIFAR100(root="./data", train=True, download=True)
            return {i: name for i, name in enumerate(dataset.classes)}
        except (ImportError, Exception) as e:
            print(f"Warning: Could not load CIFAR100 class names, falling back to numbers. Error: {e}")
            return {i: str(i) for i in range(num_classes)}
    else:
        # Fallback for other datasets
        return {i: str(i) for i in range(num_classes)}


def export_sorted_attribution_scores_from_rows(rows, output_csv="sorted_attribution_scores.csv", num_classes=10):
    """
    Export attribution scores sorted in descending order for each image to a CSV file
    from the original CSV row data (image, class_id, attribution_score).
    
    Parameters:
    -----------
    rows : List[Tuple]
        List of tuples in the format (image_path, class_id, attribution_score)
    output_csv : str
        Output CSV filename
    num_classes : int
        The number of classes in the dataset (e.g., 10 for CIFAR-10).
    """
    # --- Dynamically get class name mapping ---
    class_names = get_class_names(num_classes)
    
    # --- Aggregate scores by image ---
    image_scores = defaultdict(dict)
    for image_path, class_id, score in rows:
        image_scores[image_path][int(class_id)] = float(score)
    
    # --- Process images and write sorted CSV ---
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image"] + [f"R{i+1}_{s}" for i in range(num_classes) for s in ["Cls", "Name", "Sc"]]
        writer.writerow(header)
        
        for image_path, scores_dict in image_scores.items():
            img_name = os.path.basename(image_path)
            class_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            
            row = [img_name]
            for class_id, score in class_scores:
                # Use scientific notation for very small values, otherwise use appropriate precision
                if abs(score) < 1e-6 and score != 0:
                    score_str = f"{score:.6e}"
                elif abs(score) < 1e-3:
                    score_str = f"{score:.6f}"
                else:
                    score_str = f"{score:.4f}"
                
                row.extend([
                    class_id,
                    f"{class_id}: {class_names.get(class_id, 'Unknown')}",
                    score_str
                ])
            # Ensure row has enough columns for all classes if some are missing
            while len(row) < len(header):
                row.extend(['', '', ''])
            writer.writerow(row)
    
    print(f"Sorted attribution scores saved to {output_csv}")
    return output_csv

def calculate_ndcg_metrics(
    sorted_csv,
    reference_csv,
    k: int = 3,
    num_classes: int = 10,
    # Binary relevance definition (trial): choose how to derive positives from reference
    binary_pos_mode: str = "top_m",  # one of {"top_m", "threshold"}
    binary_m: int | None = None,      # if None, defaults to k
    binary_threshold: float | None = None,  # used when binary_pos_mode == "threshold"
    return_all: bool = True,          # default: return dict of all metrics
):
    """
    Calculate nDCG@k and additional ranking/binary metrics comparing a predicted sorted table against a reference.

    Parameters:
    -----------
    sorted_csv : str
        Path to the CSV file with sorted attribution scores.
    reference_csv : str
        Path to the reference CSV file with ground truth scores.
    k : int
        The k in nDCG@k and Top-k style metrics (default: 3).
    num_classes : int
        Number of classes (e.g., 10 or 100).
    binary_pos_mode : str
        How to define positive classes from the reference per image: "top_m" or "threshold".
    binary_m : Optional[int]
        If using "top_m", number of positives. Defaults to k when None.
    binary_threshold : Optional[float]
        If using "threshold", any class with ref score > threshold is a positive.
    return_all : bool
        If True, return a dict of all aggregated metrics; otherwise return (nDCG@k, Spearman(value), Kendall(value)) for compatibility.

    Returns:
    --------
    - If return_all is False: tuple (avg_ndcg, avg_spearman_value, avg_kendall_value) for backward compatibility.
    - If return_all is True (default): dict with many aggregated metrics.
    """
    import numpy as np
    try:
        from sklearn.metrics import (
            ndcg_score,
            roc_auc_score,
            average_precision_score,
            label_ranking_average_precision_score,
            coverage_error,
            label_ranking_loss,
        )
    except ImportError:
        print("Warning: scikit-learn not found. nDCG calculation will be skipped.")
        return None
    
    if not os.path.exists(reference_csv):
        print(f"Reference CSV file {reference_csv} not found.")
        return None
        
    # --- Load reference data ---
    reference_scores = defaultdict(dict)
    global_min_ref_score = 0.0
    
    print(f"Loading reference scores from {reference_csv} for nDCG calculation...")
    with open(reference_csv, "r", newline="") as ref_file:
        reader = csv.reader(ref_file)
        header = next(reader)
        for row in reader:
            if not row: continue
            img_name = row[0]
            ### MODIFIED ###
            for i in range(num_classes):
                class_col_idx = 1 + i * 3
                score_col_idx = 3 + i * 3
                if score_col_idx < len(row) and row[class_col_idx]:
                    try:
                        class_id = int(row[class_col_idx])
                        score = float(row[score_col_idx])
                        reference_scores[img_name][class_id] = score
                        if score < global_min_ref_score:
                            global_min_ref_score = score
                    except (ValueError, IndexError):
                        continue
    
    # --- Load sorted scores ---
    sorted_scores = defaultdict(dict)
    with open(sorted_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: continue
            img_name = row[0]
            for i in range(num_classes):
                class_col_idx = 1 + i * 3
                score_col_idx = 3 + i * 3
                if score_col_idx < len(row) and row[class_col_idx]:
                    try:
                        class_id = int(row[class_col_idx])
                        score = float(row[score_col_idx])
                        sorted_scores[img_name][class_id] = score
                    except (ValueError, IndexError):
                        continue
    
    # --- Calculate offset to handle negative scores ---
    offset = abs(global_min_ref_score) if global_min_ref_score < 0 else 0.0
    if offset > 0:
        print(f"Found negative reference scores. Applying an offset of {offset:.4f} for nDCG calculation.")
    
    # --- Calculate nDCG scores ---
    ndcg_scores_list = []
    spearman_val_list = []
    kendall_val_list = []
    # Ranking-only metrics (scale-invariant)
    top1_list = []
    topk_overlap_list = []  # size of intersection / k
    prec_at_k_list = []     # equals overlap when both sets size k
    rec_at_k_list = []      # equals overlap when both sets size k
    jacc_at_k_list = []
    spearman_rank_list = []
    kendall_rank_list = []
    # Binary relevance (trial)
    roc_auc_list = []
    ap_list = []  # Average Precision (AP)
    lrap_list = []
    lrloss_list = []
    coverage_list = []
    
    for img_name, ref_img_scores in reference_scores.items():
        if img_name in sorted_scores:
            # Create score vectors ordered by class_id (0 to num_classes-1)
            y_true = np.zeros(num_classes)
            y_score = np.zeros(num_classes)
            
            for i in range(num_classes):
                y_true[i] = ref_img_scores.get(i, 0)
                y_score[i] = sorted_scores[img_name].get(i, 0)
            
            # Apply offset to both true and predicted scores to make them non-negative
            y_true += offset
            y_score += offset
            
            # --- nDCG@k ---
            score = ndcg_score([y_true], [y_score], k=k)
            ndcg_scores_list.append(score)
            # --- Spearman/Kendall on values ---
            rho, _ = spearmanr(y_true, y_score)
            spearman_val_list.append(rho)
            tau, _ = kendalltau(y_true, y_score)
            kendall_val_list.append(tau)

            # --- Ranking-only metrics ---
            # Ranks (descending order: higher score => smaller rank number)
            r_true = np.argsort(-y_true)
            r_pred = np.argsort(-y_score)

            # Top-1 equality
            top1_list.append(int(r_true[0] == r_pred[0]))

            # Top-k set overlap and derived metrics
            set_true_k = set(r_true[:k])
            set_pred_k = set(r_pred[:k])
            inter = len(set_true_k & set_pred_k)
            union = len(set_true_k | set_pred_k)
            overlap = inter / float(k)
            topk_overlap_list.append(overlap)
            prec_at_k_list.append(overlap)  # both sets have size k
            rec_at_k_list.append(overlap)   # both sets have size k
            jacc_at_k_list.append(inter / float(union) if union > 0 else 0.0)

            # Spearman/Kendall computed on rank vectors (tie-handling via rankdata)
            rt = rankdata(-y_true, method="average")
            rp = rankdata(-y_score, method="average")
            rho_r, _ = spearmanr(rt, rp)
            tau_r, _ = kendalltau(rt, rp)
            spearman_rank_list.append(rho_r)
            kendall_rank_list.append(tau_r)

            # --- Binary relevance metrics (trial) ---
            # Define positives from reference per image
            m = binary_m if (binary_m is not None) else k
            if binary_pos_mode == "top_m":
                pos_idx = set(r_true[:max(1, m)])
                y_true_bin = np.zeros(num_classes, dtype=int)
                for idx in pos_idx:
                    y_true_bin[idx] = 1
            elif binary_pos_mode == "threshold":
                thr = 0.0 if (binary_threshold is None) else float(binary_threshold)
                y_true_bin = (y_true > thr).astype(int)
                # Ensure at least one positive (else metrics undefined); fallback to top-1
                if y_true_bin.sum() == 0:
                    y_true_bin[r_true[0]] = 1
            else:
                # Fallback to top-1 positives
                y_true_bin = np.zeros(num_classes, dtype=int)
                y_true_bin[r_true[0]] = 1

            # ROC-AUC (needs both classes)
            try:
                if y_true_bin.min() != y_true_bin.max():
                    roc_auc = roc_auc_score(y_true_bin, y_score)
                    roc_auc_list.append(roc_auc)
            except Exception:
                pass

            # Average Precision (AP)
            try:
                if y_true_bin.sum() > 0:
                    ap = average_precision_score(y_true_bin, y_score)
                    ap_list.append(ap)
            except Exception:
                pass

            # LRAP, label ranking loss, coverage error (sklearn multilabel ranking)
            try:
                Y_true_bin = y_true_bin.reshape(1, -1)
                Y_score = y_score.reshape(1, -1)
                if y_true_bin.sum() > 0:
                    lrap_val = label_ranking_average_precision_score(Y_true_bin, Y_score)
                    lrap_list.append(lrap_val)
                    lrloss_val = label_ranking_loss(Y_true_bin, Y_score)
                    lrloss_list.append(lrloss_val)
                    cov_val = coverage_error(Y_true_bin, Y_score)
                    coverage_list.append(cov_val)
            except Exception:
                pass
    
    if not ndcg_scores_list:
        print("No matching images found for nDCG calculation.")
        return None
    
    avg_ndcg = np.mean(ndcg_scores_list)
    avg_spearman_val = np.nanmean(spearman_val_list)
    avg_kendall_val = np.nanmean(kendall_val_list)

    # Ranking-only aggregates
    top1 = float(np.mean(top1_list)) if top1_list else float("nan")
    topk = float(np.mean(topk_overlap_list)) if topk_overlap_list else float("nan")
    prec_k = float(np.mean(prec_at_k_list)) if prec_at_k_list else float("nan")
    rec_k = float(np.mean(rec_at_k_list)) if rec_at_k_list else float("nan")
    jacc_k = float(np.mean(jacc_at_k_list)) if jacc_at_k_list else float("nan")
    spearman_rank = float(np.nanmean(spearman_rank_list)) if spearman_rank_list else float("nan")
    kendall_rank = float(np.nanmean(kendall_rank_list)) if kendall_rank_list else float("nan")

    # Binary aggregates
    roc_auc = float(np.nanmean(roc_auc_list)) if roc_auc_list else float("nan")
    ap = float(np.nanmean(ap_list)) if ap_list else float("nan")
    lrap = float(np.nanmean(lrap_list)) if lrap_list else float("nan")
    lrloss = float(np.nanmean(lrloss_list)) if lrloss_list else float("nan")
    coverage = float(np.nanmean(coverage_list)) if coverage_list else float("nan")

    print(f"nDCG@{k}: {avg_ndcg:.4f} (calculated on {len(ndcg_scores_list)} images)")
    print(f"Spearman(value): {avg_spearman_val:.4f} | Kendall(value): {avg_kendall_val:.4f}")
    print(f"Top-1: {top1:.4f} | Top-{k} overlap: {topk:.4f} | Jaccard@{k}: {jacc_k:.4f}")
    print(f"Spearman(rank): {spearman_rank:.4f} | Kendall(rank): {kendall_rank:.4f}")
    print(f"ROC-AUC(bin): {roc_auc:.4f} | AP(bin): {ap:.4f} | LRAP: {lrap:.4f} | RankLoss: {lrloss:.4f} | Coverage: {coverage:.4f}")
    
    # --- Save metrics to file ---
    metrics_file = os.path.splitext(sorted_csv)[0] + "_metrics.txt"
    with open(metrics_file, "w") as mf:
        mf.write(f"nDCG@{k}: {avg_ndcg:.4f} (on {len(ndcg_scores_list)} images)\n")
        mf.write(f"Spearman(value): {avg_spearman_val:.4f}\n")
        mf.write(f"Kendall(value): {avg_kendall_val:.4f}\n")
        mf.write(f"Top-1: {top1:.4f}\n")
        mf.write(f"Top-{k} overlap: {topk:.4f}\n")
        mf.write(f"P@{k}: {prec_k:.4f}\n")
        mf.write(f"R@{k}: {rec_k:.4f}\n")
        mf.write(f"Jaccard@{k}: {jacc_k:.4f}\n")
        mf.write(f"Spearman(rank): {spearman_rank:.4f}\n")
        mf.write(f"Kendall(rank): {kendall_rank:.4f}\n")
        mf.write(f"ROC-AUC(bin): {roc_auc:.4f}\n")
        mf.write(f"AP(bin): {ap:.4f}\n")
        mf.write(f"LRAP: {lrap:.4f}\n")
        mf.write(f"LabelRankingLoss: {lrloss:.4f}\n")
        mf.write(f"CoverageError: {coverage:.4f}\n")
    print(f"Metrics saved to {metrics_file}")

    if return_all:
        return {
            "ndcg@k": avg_ndcg,
            "spearman_value": avg_spearman_val,
            "kendall_value": avg_kendall_val,
            "top1": top1,
            "topk_overlap": topk,
            "precision@k": prec_k,
            "recall@k": rec_k,
            "jaccard@k": jacc_k,
            "spearman_rank": spearman_rank,
            "kendall_rank": kendall_rank,
            "roc_auc": roc_auc,
            "average_precision": ap,
            "lrap": lrap,
            "label_ranking_loss": lrloss,
            "coverage_error": coverage,
            "num_images": len(ndcg_scores_list),
            "k": k,
            "binary_pos_mode": binary_pos_mode,
            "binary_m": (binary_m if binary_m is not None else k),
            "binary_threshold": binary_threshold,
        }
    else:
        return avg_ndcg, avg_spearman_val, avg_kendall_val