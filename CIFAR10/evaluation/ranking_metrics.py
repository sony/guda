#!/usr/bin/env python3
"""
Advanced ranking metrics for attribution quality evaluation.
Implements NDCG@k, MRR, WPRA, Affine-calibrated MAE/MSE, and RBO.
"""
import numpy as np


def argsort_desc(x):
    """Return indices that would sort x in descending order."""
    return np.argsort(-np.array(x))


def rank_of(idx, perm):
    """
    Find 1-based rank of idx in permutation perm.
    perm is list/array of indices sorted by score descending.
    """
    return int(np.where(perm == idx)[0][0]) + 1


def relevance_rank_based(g):
    """
    Convert gold scores to nonnegative rank-based relevance.
    Highest gold score gets relevance K-1, lowest gets 0.
    
    Args:
        g: Gold scores (numpy array or list)
    
    Returns:
        rel: Rank-based relevance vector (numpy array)
    """
    g = np.array(g, dtype=float)
    K = len(g)
    perm = argsort_desc(g)
    rel = np.zeros(K, dtype=float)
    for r, i in enumerate(perm, start=1):
        rel[i] = K - r
    return rel


def relevance_shift_nonneg(g):
    """
    Convert gold scores to nonnegative by shifting minimum to 0.
    
    Args:
        g: Gold scores (numpy array or list)
    
    Returns:
        rel: Shifted relevance vector (numpy array)
    """
    g = np.array(g, dtype=float)
    return g - np.min(g)


def ndcg_at_k(s, g, k, use_rank_rel=True):
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
        k: Cutoff rank
        use_rank_rel: If True, use rank-based relevance; else shift to nonneg
    
    Returns:
        NDCG@k score (float in [0, 1])
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    K = len(s)
    k = min(k, K)
    
    # Convert gold to relevance
    rel = relevance_rank_based(g) if use_rank_rel else relevance_shift_nonneg(g)
    
    # Get permutations (descending order)
    ps = argsort_desc(s)
    pg = argsort_desc(rel)  # ideal ranking uses relevance
    
    def dcg(perm):
        """Compute DCG@k for given permutation."""
        out = 0.0
        for r in range(1, k + 1):
            i = perm[r - 1]
            out += (2.0 ** rel[i] - 1.0) / np.log2(r + 1.0)
        return out
    
    denom = dcg(pg)
    if denom == 0:
        return 0.0
    return dcg(ps) / denom


def mrr_top1(s, g):
    """
    Compute Mean Reciprocal Rank (MRR) for single ranking.
    Measures how well the predicted ranking places the gold Top-1 class.
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
    
    Returns:
        RR score (float in (0, 1])
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    
    # Find gold Top-1 class
    pg = argsort_desc(g)
    c_star = pg[0]
    
    # Find rank of c_star in predicted ranking
    ps = argsort_desc(s)
    rr = 1.0 / rank_of(c_star, ps)
    return rr


def wpra_top_heavy(s, g, m=3, eps=1e-12):
    """
    Compute Weighted Pairwise Ranking Agreement (WPRA).
    Focuses on pairs where at least one class is in gold Top-m.
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
        m: Number of top classes to focus on (default: 3)
        eps: Tolerance for considering scores equal (default: 1e-12)
    
    Returns:
        WPRA score (float in [0, 1])
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    K = len(s)
    
    # Identify gold Top-m classes
    pg = argsort_desc(g)
    H = set(pg[:m].tolist())
    
    num = 0.0
    den = 0.0
    
    for a in range(K):
        for b in range(a + 1, K):
            # Skip if neither class is in Top-m
            if (a not in H) and (b not in H):
                continue
            
            dg = g[a] - g[b]
            
            # Skip ties in gold
            if abs(dg) < eps:
                continue
            
            ds = s[a] - s[b]
            w = abs(dg)
            den += w
            
            # Check if predicted ranking agrees with gold
            if np.sign(dg) == np.sign(ds):
                num += w
    
    return 0.0 if den == 0 else num / den


def affine_calibrated_errors(s, g):
    """
    Compute affine-calibrated MAE and MSE.
    Fits a*s + b to g via least squares, then measures errors.
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
    
    Returns:
        (mse, mae): Calibrated MSE and MAE
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    
    # Check for degenerate case
    if np.var(s) < 1e-12:
        # s is nearly constant; fall back to just offset
        b = np.mean(g)
        g_hat = np.full_like(g, b)
    else:
        # Solve min_{a,b} || a*s + b - g ||_2^2
        X = np.stack([s, np.ones_like(s)], axis=1)  # [K, 2]
        theta, *_ = np.linalg.lstsq(X, g, rcond=None)
        a, b = theta
        g_hat = a * s + b
    
    mse = np.mean((g_hat - g) ** 2)
    mae = np.mean(np.abs(g_hat - g))
    
    return mse, mae


def rbo_truncated(s, g, p=0.9):
    """
    Compute Rank-Biased Overlap (RBO) with geometric decay parameter p.
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
        p: Decay parameter in (0, 1); larger p emphasizes deeper ranks less
    
    Returns:
        RBO score (float)
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    K = len(s)
    
    ps = argsort_desc(s)
    pg = argsort_desc(g)
    
    A = set()
    B = set()
    total = 0.0
    
    for d in range(1, K + 1):
        A.add(int(ps[d - 1]))
        B.add(int(pg[d - 1]))
        Xd = len(A.intersection(B)) / d
        total += (p ** (d - 1)) * Xd
    
    return (1.0 - p) * total


def compute_all_metrics(s, g, k_values=[3, 5, 10], wpra_m=3, rbo_p=0.9):
    """
    Compute all ranking metrics for a single image.
    
    Args:
        s: Predicted scores (numpy array or list)
        g: Gold scores (numpy array or list)
        k_values: List of k values for NDCG@k
        wpra_m: Number of top classes for WPRA
        rbo_p: Decay parameter for RBO
    
    Returns:
        dict: Dictionary with all metric scores
    """
    s = np.array(s, dtype=float)
    g = np.array(g, dtype=float)
    
    metrics = {}
    
    # NDCG@k for each k
    for k in k_values:
        metrics[f'ndcg@{k}'] = ndcg_at_k(s, g, k, use_rank_rel=True)
    
    # MRR
    metrics['mrr'] = mrr_top1(s, g)
    
    # WPRA (Top-m focused)
    metrics[f'wpra_top{wpra_m}'] = wpra_top_heavy(s, g, m=wpra_m)
    
    # WPRA (All pairs)
    K = len(s)
    metrics['wpra_all'] = wpra_top_heavy(s, g, m=K)
    
    # Affine-calibrated errors
    mse, mae = affine_calibrated_errors(s, g)
    metrics['calib_mse'] = mse
    metrics['calib_mae'] = mae
    
    # RBO
    metrics['rbo'] = rbo_truncated(s, g, p=rbo_p)
    
    return metrics


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics across multiple images.
    
    Args:
        metrics_list: List of metric dictionaries (one per image)
    
    Returns:
        dict: Dictionary with mean, std, and median for each metric
    """
    if not metrics_list:
        return {}
    
    # Collect values for each metric
    metric_names = metrics_list[0].keys()
    metric_values = {name: [] for name in metric_names}
    
    for metrics in metrics_list:
        for name, value in metrics.items():
            metric_values[name].append(value)
    
    # Compute statistics
    aggregated = {}
    for name, values in metric_values.items():
        values = np.array(values)
        aggregated[f'{name}_mean'] = np.mean(values)
        aggregated[f'{name}_std'] = np.std(values)
        aggregated[f'{name}_median'] = np.median(values)
    
    return aggregated
