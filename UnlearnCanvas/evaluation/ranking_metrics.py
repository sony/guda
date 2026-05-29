#!/usr/bin/env python3
"""
Advanced ranking metrics for style attribution evaluation.

Implementation of information retrieval and ranking evaluation metrics
optimized for style attribution tasks where:
- K = number of styles (typically 16 or 60)
- For each image x, we have:
  - Gold vector g(x) ∈ R^K: LOGOA scores per style
  - Predicted vector s(x) ∈ R^K: UNA or CLIPA scores per style

Metrics implemented:
1. NDCG@k (Normalized Discounted Cumulative Gain)
2. MRR (Mean Reciprocal Rank)
3. WPRA (Weighted Pairwise Ranking Agreement)
4. Affine-calibrated MAE/MSE
5. RBO (Rank-Biased Overlap)

References:
- NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
- Kendall tau: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
- RBO: Webber et al. (2010) "A Similarity Measure for Indefinite Rankings"
"""
import numpy as np
from typing import Tuple, Optional


def ndcg_at_k(predicted: np.ndarray, gold: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k using rank-based relevance.
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        k: Cutoff position
        
    Returns:
        NDCG@k score in [0, 1]
        
    Note:
        Uses rank-based relevance to avoid issues with negative or
        small magnitude scores. Relevance = K - rank + 1 for each style.
    """
    K = len(gold)
    k = min(k, K)
    
    # Get rankings (descending order)
    gold_ranks = np.argsort(-gold)  # Indices sorted by gold scores
    pred_ranks = np.argsort(-predicted)  # Indices sorted by predicted scores
    
    # Create rank-based relevance: higher gold rank = higher relevance
    # Gold rank 0 (best) gets relevance K, gold rank K-1 (worst) gets relevance 1
    relevance = np.zeros(K)
    for rank, idx in enumerate(gold_ranks):
        relevance[idx] = K - rank
    
    # Compute DCG@k for predicted ranking
    dcg = 0.0
    for i in range(k):
        idx = pred_ranks[i]
        rel = relevance[idx]
        dcg += rel / np.log2(i + 2)  # i+2 because positions start at 1
    
    # Compute ideal DCG@k (using gold ranking)
    idcg = 0.0
    for i in range(k):
        idx = gold_ranks[i]
        rel = relevance[idx]
        idcg += rel / np.log2(i + 2)
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr_top1(predicted: np.ndarray, gold: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for the top-1 gold style.
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        
    Returns:
        Reciprocal rank in (0, 1]
        
    Note:
        MRR measures how quickly the method finds the gold top-1 style.
        If top-1 gold style is at position r in predicted ranking, MRR = 1/r.
    """
    # Find gold top-1 style
    gold_top1_idx = np.argmax(gold)
    
    # Find its position in predicted ranking (1-indexed)
    pred_ranks = np.argsort(-predicted)  # Descending order
    position = np.where(pred_ranks == gold_top1_idx)[0][0] + 1
    
    return 1.0 / position


def wpra_top_heavy(
    predicted: np.ndarray,
    gold: np.ndarray,
    m: int = 5,
    epsilon: float = 1e-6,
    pred_epsilon: float = 1e-6
) -> float:
    """
    Compute Weighted Pairwise Ranking Agreement (WPRA) with top-heavy focus.
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        m: Head size (number of top styles to focus on)
        epsilon: Tolerance for tie handling in gold scores
        pred_epsilon: Tolerance for tie handling in predicted scores
        
    Returns:
        WPRA score in [0, 1]
        
    Note:
        WPRA emphasizes pairs with clear gold separation.
        Weight w_ab = |g_a - g_b| for each pair (a, b).
        Only considers pairs where at least one style is in top-m.
        Skips pairs that are tied in either gold or predicted (within epsilon).
    """
    K = len(gold)
    m = min(m, K)
    
    # Get top-m indices by gold scores
    gold_ranks = np.argsort(-gold)
    top_m_indices = set(gold_ranks[:m])
    
    total_weight = 0.0
    agreement_weight = 0.0
    
    # Iterate over all pairs
    for a in range(K):
        for b in range(a + 1, K):
            # Skip if neither is in top-m
            if a not in top_m_indices and b not in top_m_indices:
                continue
            
            # Weight by gold difference
            weight = abs(gold[a] - gold[b])
            
            # Skip ties in gold (within epsilon)
            if weight < epsilon:
                continue
            
            # Skip ties in predicted (for stability)
            pred_diff = abs(predicted[a] - predicted[b])
            if pred_diff < pred_epsilon:
                continue
            
            total_weight += weight
            
            # Check if predicted agrees with gold ordering
            gold_order = gold[a] > gold[b]
            pred_order = predicted[a] > predicted[b]
            
            if gold_order == pred_order:
                agreement_weight += weight
    
    # Avoid division by zero
    if total_weight == 0:
        return 0.0
    
    return agreement_weight / total_weight


def affine_calibrated_errors(
    predicted: np.ndarray,
    gold: np.ndarray,
    mode: str = 'per-image'
) -> Tuple[float, float]:
    """
    Compute affine-calibrated MAE and MSE (per-image mode).
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        mode: 'per-image' (kept for API compatibility)
        
    Returns:
        Tuple of (MSE, MAE) after per-image affine calibration
        
    Note:
        Finds optimal affine transformation: s_cal = a * s + b
        to minimize squared error with gold FOR THIS IMAGE.
        This is the "easy" calibration that absorbs all scale/offset.
        For main paper results, use affine_calibrated_errors_global instead.
    """
    # Flatten to 1D arrays
    s = predicted.flatten()
    g = gold.flatten()
    
    # Fit affine transformation: s_cal = a * s + b
    # Minimize: ||g - (a * s + b)||^2
    # Solution: a = cov(s,g) / var(s), b = mean(g) - a * mean(s)
    
    if len(s) < 2:
        return float(np.mean((s - g)**2)), float(np.mean(np.abs(s - g)))
    
    # Compute statistics
    mean_s = np.mean(s)
    mean_g = np.mean(g)
    var_s = np.var(s)
    cov_sg = np.mean((s - mean_s) * (g - mean_g))
    
    # Handle degenerate case (constant predictions)
    if var_s < 1e-10:
        # Just use mean offset
        a = 0.0
        b = mean_g
    else:
        a = cov_sg / var_s
        b = mean_g - a * mean_s
    
    # Apply calibration
    s_cal = a * s + b
    
    # Compute errors
    mse = float(np.mean((s_cal - g)**2))
    mae = float(np.mean(np.abs(s_cal - g)))
    
    return mse, mae


def affine_calibrated_errors_global(
    predicted_batch: np.ndarray,
    gold_batch: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute affine-calibrated MAE and MSE with GLOBAL calibration.
    
    Fits a single (a, b) across ALL images and styles, then computes errors.
    This is the RECOMMENDED metric for paper reporting, as it tests whether
    the method's magnitudes are consistently accurate across the dataset.
    
    Args:
        predicted_batch: Predicted scores [N, K] where N = num images, K = num styles
        gold_batch: Gold standard scores [N, K]
        
    Returns:
        Tuple of (MSE, MAE, a, b) where:
        - MSE: Mean squared error after global calibration
        - MAE: Mean absolute error after global calibration
        - a: Fitted scale parameter
        - b: Fitted offset parameter
        
    Note:
        This is stricter than per-image calibration because it cannot
        "cheat" by fitting different (a,b) for each image. Use this
        to demonstrate that your method's contribution magnitudes are
        systematically accurate.
    """
    # Flatten to 1D arrays (concatenate all images and styles)
    s = predicted_batch.reshape(-1)
    g = gold_batch.reshape(-1)
    
    if len(s) < 2:
        return float(np.mean((s - g)**2)), float(np.mean(np.abs(s - g))), 1.0, 0.0
    
    # Compute statistics over entire dataset
    mean_s = np.mean(s)
    mean_g = np.mean(g)
    var_s = np.var(s)
    cov_sg = np.mean((s - mean_s) * (g - mean_g))
    
    # Fit global affine transformation
    if var_s < 1e-10:
        a = 0.0
        b = mean_g
    else:
        a = cov_sg / var_s
        b = mean_g - a * mean_s
    
    # Apply calibration
    s_cal = a * s + b
    
    # Compute errors
    mse = float(np.mean((s_cal - g)**2))
    mae = float(np.mean(np.abs(s_cal - g)))
    
    return mse, mae, a, b


def rbo_truncated(
    predicted: np.ndarray,
    gold: np.ndarray,
    p: float = 0.9,
    depth: Optional[int] = None
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) with top-heavy weighting.
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        p: Persistence parameter in (0, 1). Higher = less top-heavy.
           Recommended: 0.9 (strongly top-heavy) or 0.95 (slightly less)
        depth: Maximum depth to compute (default: full length)
        
    Returns:
        RBO score in [0, 1]
        
    Note:
        RBO is designed for top-weighted ranked-list similarity.
        Weight at position d: (1-p) * p^(d-1)
        
    Reference:
        Webber et al. (2010) "A Similarity Measure for Indefinite Rankings"
        https://blog.mobile.codalism.com/research/papers/wmz10_tois.pdf
    """
    K = len(gold)
    depth = K if depth is None else min(depth, K)
    
    # Get rankings (indices in order of scores, descending)
    gold_ranking = np.argsort(-gold).tolist()
    pred_ranking = np.argsort(-predicted).tolist()
    
    # Convert to sets for overlap computation
    rbo_score = 0.0
    
    for d in range(1, depth + 1):
        # Get top-d sets
        gold_set = set(gold_ranking[:d])
        pred_set = set(pred_ranking[:d])
        
        # Overlap at depth d
        overlap = len(gold_set & pred_set)
        agreement = overlap / d
        
        # Weight by RBO formula: (1-p) * p^(d-1)
        weight = (1 - p) * (p ** (d - 1))
        rbo_score += weight * agreement
    
    return rbo_score


def compute_all_ranking_metrics(
    predicted: np.ndarray,
    gold: np.ndarray,
    k_values: list = [3, 5, 10],
    m_wpra: int = 5,
    p_rbo: float = 0.9,
    pred_epsilon: float = 0.0
) -> dict:
    """
    Compute all ranking metrics for a single image.
    
    Args:
        predicted: Predicted attribution scores [K]
        gold: Gold standard (LOGOA) scores [K]
        k_values: List of k values for NDCG@k
        m_wpra: Head size for WPRA
        p_rbo: Persistence parameter for RBO
        pred_epsilon: Small epsilon added to predicted scores for tie-breaking
                      (only affects WPRA ranking; default 0.0 means no adjustment)
        
    Returns:
        Dictionary with all metric values
    """
    K = len(gold)
    
    # Adjust k_values to not exceed K
    k_values = [k for k in k_values if k <= K]
    
    # Adjust m_wpra
    m_wpra = min(m_wpra, K)
    
    metrics = {}
    
    # NDCG@k
    for k in k_values:
        metrics[f'ndcg@{k}'] = ndcg_at_k(predicted, gold, k)
    
    # MRR
    metrics['mrr'] = mrr_top1(predicted, gold)
    
    # WPRA (with optional tie-breaking)
    metrics[f'wpra_m{m_wpra}'] = wpra_top_heavy(predicted, gold, m=m_wpra, pred_epsilon=pred_epsilon)
    
    # RBO
    metrics[f'rbo_p{p_rbo}'] = rbo_truncated(predicted, gold, p=p_rbo)
    
    # Affine-calibrated errors
    mse, mae = affine_calibrated_errors(predicted, gold)
    metrics['cal_mse'] = mse
    metrics['cal_mae'] = mae
    
    return metrics


def compute_metrics_batch(
    predicted_batch: np.ndarray,
    gold_batch: np.ndarray,
    k_values: list = [3, 5, 10],
    m_wpra: int = 5,
    p_rbo: float = 0.9
) -> dict:
    """
    Compute metrics for multiple images and return aggregated statistics.
    
    Args:
        predicted_batch: Predicted scores [N, K] where N = num images
        gold_batch: Gold standard scores [N, K]
        k_values: List of k values for NDCG@k
        m_wpra: Head size for WPRA
        p_rbo: Persistence parameter for RBO
        
    Returns:
        Dictionary with mean metric values across all images
    """
    N = len(predicted_batch)
    all_metrics = []
    
    for i in range(N):
        metrics = compute_all_ranking_metrics(
            predicted_batch[i],
            gold_batch[i],
            k_values=k_values,
            m_wpra=m_wpra,
            p_rbo=p_rbo
        )
        all_metrics.append(metrics)
    
    # Aggregate: compute mean for each metric
    aggregated = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    K = 16  # Number of styles
    
    # Simulate gold and predicted scores
    gold = np.random.randn(K)
    predicted = gold + 0.3 * np.random.randn(K)  # Noisy version
    
    print("Example: Single image ranking metrics")
    print("=" * 60)
    print(f"K = {K} styles")
    print()
    
    metrics = compute_all_ranking_metrics(predicted, gold)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key:15s}: {value:.4f}")
    
    print()
    print("For K=16 styles, recommended settings:")
    print("  - NDCG@3, NDCG@5 (main), NDCG@10 (optional)")
    print("  - WPRA with m=5 or m=10")
    print("  - RBO with p=0.9 (strongly top-heavy)")
