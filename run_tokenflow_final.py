#!/usr/bin/env python3
"""
Robust aggregation methods for Min-K%++ membership inference.
"""

import os
import argparse
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# =============================================================================
# REPRODUCIBILITY: Set all random seeds
# =============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_wikimia_data(dataset_name):
    """Load WikiMIA dataset."""
    dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
    data = [{"input": d["input"], "label": d["label"]} for d in dataset]
    return data


# =============================================================================
# CORE SCORING FUNCTIONS
# =============================================================================

def compute_token_statistics(text, model, tokenizer):
    """
    Compute comprehensive token-level statistics for a text.

    Returns:
        dict: Contains log_probs, z_scores, entropies, and various aggregations
    """
    # Tokenize
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)

    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss.item()
    logits = outputs.logits

    # Token-level computations
    input_ids_shifted = input_ids[0, 1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs_all = F.log_softmax(logits[0, :-1], dim=-1)

    # Get log-prob of actual tokens
    token_log_probs = log_probs_all.gather(dim=-1, index=input_ids_shifted).squeeze(-1)

    # Compute expected log-prob and variance (Min-K++ normalization)
    mu = (probs * log_probs_all).sum(-1)
    sigma_sq = (probs * torch.square(log_probs_all)).sum(-1) - torch.square(mu)
    sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))

    # Z-scores
    z_scores = ((token_log_probs - mu) / sigma).float().cpu().numpy()

    # Entropy at each position
    entropies = -(probs * log_probs_all).sum(-1).float().cpu().numpy()

    # Raw log probs
    log_probs_np = token_log_probs.float().cpu().numpy()

    return {
        'loss': -loss,
        'log_probs': log_probs_np,
        'z_scores': z_scores,
        'entropies': entropies,
        'n_tokens': len(z_scores),
        'text_bytes': len(zlib.compress(bytes(text, 'utf-8')))
    }


def compute_aggregation_scores(stats_dict):
    """
    Compute various aggregation strategies for the token statistics.

    This is the core of our analysis: comparing different aggregation methods.
    """
    z_scores = stats_dict['z_scores']
    log_probs = stats_dict['log_probs']
    n = len(z_scores)

    if n < 3:
        return None

    scores = {}

    # =================================
    # Baseline Methods
    # =================================
    scores['loss'] = stats_dict['loss']
    scores['zlib'] = stats_dict['loss'] / stats_dict['text_bytes']

    # =================================
    # Min-K% Family (Standard Mean)
    # =================================
    sorted_z = np.sort(z_scores)

    for k_pct in [10, 20, 30, 40, 50]:
        k = max(1, int(n * k_pct / 100))
        scores[f'mink_pp_{k_pct}'] = np.mean(sorted_z[:k])

    # =================================
    # Contribution 1: Robust Aggregation
    # Trimmed Mean - removes extreme values before averaging
    # =================================
    for k_pct in [20, 30, 40]:
        k = max(1, int(n * k_pct / 100))
        lowest_k = sorted_z[:k]

        # Standard mean (baseline)
        scores[f'standard_mean_{k_pct}'] = np.mean(lowest_k)

        # Trimmed mean (10% trim from each end)
        if len(lowest_k) > 4:
            trim_n = max(1, int(len(lowest_k) * 0.1))
            trimmed = lowest_k[trim_n:-trim_n]
            scores[f'trimmed_mean_{k_pct}'] = np.mean(trimmed)
        else:
            scores[f'trimmed_mean_{k_pct}'] = np.mean(lowest_k)

        # Winsorized mean (clip to 10th/90th percentile)
        if len(lowest_k) > 2:
            p10, p90 = np.percentile(lowest_k, [10, 90])
            clipped = np.clip(lowest_k, p10, p90)
            scores[f'winsorized_mean_{k_pct}'] = np.mean(clipped)
        else:
            scores[f'winsorized_mean_{k_pct}'] = np.mean(lowest_k)

        # Median (most robust)
        scores[f'median_{k_pct}'] = np.median(lowest_k)

    # =================================
    # Contribution 2: Gradient-Informed Weighting
    # Weight tokens by prediction stability
    # =================================
    gradients = np.abs(np.diff(log_probs))
    gradients = np.concatenate([[gradients[0]], gradients])

    # Normalize gradients to [0, 1]
    g_range = gradients.max() - gradients.min()
    if g_range > 1e-10:
        norm_grad = (gradients - gradients.min()) / g_range
    else:
        norm_grad = np.zeros_like(gradients)

    # Stability weights: higher weight for stable (low gradient) regions
    stability_weights = 1.0 - norm_grad
    stability_weights = stability_weights / (stability_weights.sum() + 1e-10)

    scores['gradient_weighted'] = np.sum(z_scores * stability_weights)

    # Combined: gradient-weighted selection of lowest k%
    for k_pct in [20, 30]:
        k = max(1, int(n * k_pct / 100))
        # Adjust scores by stability, then select
        adjusted = z_scores - 0.3 * norm_grad  # Bonus for stable regions
        selected_idx = np.argsort(adjusted)[:k]
        scores[f'gradient_selected_{k_pct}'] = np.mean(z_scores[selected_idx])

    # =================================
    # Contribution 3: Multi-Quantile Ensemble
    # Combine different k values
    # =================================
    # Fixed-weight ensemble
    scores['ensemble_fixed'] = (
        0.3 * scores['mink_pp_20'] +
        0.4 * scores['mink_pp_30'] +
        0.3 * scores['mink_pp_40']
    )

    # Robust ensemble (using trimmed means)
    scores['ensemble_robust'] = (
        0.3 * scores['trimmed_mean_20'] +
        0.4 * scores['trimmed_mean_30'] +
        0.3 * scores['trimmed_mean_40']
    )

    # =================================
    # NEW: Entropy-Based Methods
    # =================================
    entropies = stats_dict['entropies']

    # Method 1: Raw entropy score (lower entropy = more confident = likely member)
    scores['entropy_mean'] = -np.mean(entropies)

    # Method 2: Min-K% by entropy (select low-entropy tokens, average their z-scores)
    for k_pct in [20, 30, 40]:
        k = max(1, int(n * k_pct / 100))
        # Select tokens with LOWEST entropy (most confident predictions)
        low_entropy_idx = np.argsort(entropies)[:k]
        scores[f'entropy_select_{k_pct}'] = np.mean(z_scores[low_entropy_idx])

    # Method 3: Entropy-weighted z-scores (weight by inverse entropy)
    inv_entropy = 1.0 / (entropies + 1e-6)
    inv_entropy_norm = inv_entropy / inv_entropy.sum()
    scores['entropy_weighted'] = np.sum(z_scores * inv_entropy_norm)

    # Method 4: Entropy-weighted Min-K%
    for k_pct in [20, 30, 40]:
        k = max(1, int(n * k_pct / 100))
        lowest_k_idx = np.argsort(z_scores)[:k]
        lowest_k_z = z_scores[lowest_k_idx]
        lowest_k_entropy = entropies[lowest_k_idx]

        # Weight by inverse entropy
        weights = 1.0 / (lowest_k_entropy + 1e-6)
        weights = weights / weights.sum()
        scores[f'entropy_weighted_mink_{k_pct}'] = np.sum(lowest_k_z * weights)

    # Method 5: Combined score (z-score normalized by entropy)
    scores['z_div_entropy'] = np.mean(z_scores / (entropies + 1e-6))

    # Method 6: Low-entropy tokens only (entropy < median)
    median_entropy = np.median(entropies)
    low_entropy_mask = entropies < median_entropy
    if low_entropy_mask.sum() > 0:
        scores['low_entropy_only'] = np.mean(z_scores[low_entropy_mask])
    else:
        scores['low_entropy_only'] = np.mean(z_scores)

    # TokenFlow = trimmed mean at k=30%
    scores['tokenflow'] = scores['trimmed_mean_30']

    return scores


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_cohens_d(scores, labels):
    """
    Compute Cohen's d effect size.

    Cohen's d measures the standardized difference between two groups.
    |d| > 0.2: small, |d| > 0.5: medium, |d| > 0.8: large
    """
    scores = np.array(scores)
    labels = np.array(labels)

    member = scores[labels == 1]
    non_member = scores[labels == 0]

    mean_diff = np.mean(member) - np.mean(non_member)
    pooled_std = np.sqrt((np.var(member, ddof=1) + np.var(non_member, ddof=1)) / 2)

    if pooled_std < 1e-10:
        return 0.0

    return mean_diff / pooled_std


def compute_metrics(scores, labels):
    """Compute AUROC, FPR@95%TPR, TPR@5%FPR."""
    scores = np.array(scores)
    labels = np.array(labels)

    # Handle NaN/Inf
    mask = np.isfinite(scores)
    if not mask.all():
        median_val = np.median(scores[mask]) if mask.any() else 0.0
        scores = np.where(mask, scores, median_val)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    # FPR at 95% TPR
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx_95[0]] if len(idx_95) > 0 else 1.0

    # TPR at 5% FPR
    idx_05 = np.where(fpr <= 0.05)[0]
    tpr05 = tpr[idx_05[-1]] if len(idx_05) > 0 else 0.0

    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05
    }


def statistical_significance_test(scores1, scores2, labels):
    """Wilcoxon signed-rank test between two score sets."""
    from scipy.stats import wilcoxon

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Simple paired comparison of scores
    try:
        stat, p_value = wilcoxon(scores1, scores2)
    except:
        p_value = 1.0

    return p_value


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(model_name, dataset_name, output_dir, use_half=True):
    """Run complete experiment with all analyses."""

    print("=" * 70)
    print(f"TokenFlow: Rigorous Experiment")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Seed: {SEED}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    kwargs = {"torch_dtype": torch.bfloat16} if use_half else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        device_map='auto',
        **kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load data
    print("[2/4] Loading dataset...")
    data = load_wikimia_data(dataset_name)
    labels = [d['label'] for d in data]
    print(f"  Samples: {len(data)} (members={sum(labels)}, non-members={len(labels)-sum(labels)})")

    # Collect scores
    print("[3/4] Computing scores...")
    all_scores = defaultdict(list)

    for d in tqdm(data, desc="Processing"):
        stats = compute_token_statistics(d['input'], model, tokenizer)
        if stats is None:
            # Handle edge case: very short text
            for key in all_scores.keys():
                all_scores[key].append(np.nan)
            continue

        scores = compute_aggregation_scores(stats)
        if scores is None:
            for key in all_scores.keys():
                all_scores[key].append(np.nan)
            continue

        for key, value in scores.items():
            all_scores[key].append(value)

    # =================================
    # ANALYSIS
    # =================================
    print("\n[4/4] Analyzing results...")

    results = {
        'model': model_name,
        'dataset': dataset_name,
        'n_samples': len(data),
        'seed': SEED,
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }

    # Compute metrics for each method
    print("\n" + "-" * 70)
    print(f"{'Method':<30} {'AUROC':>8} {'FPR@95':>8} {'TPR@05':>8} {'Cohen d':>8}")
    print("-" * 70)

    method_metrics = []

    for method_name, scores in sorted(all_scores.items()):
        metrics = compute_metrics(scores, labels)
        cohens_d = compute_cohens_d(scores, labels)

        results['methods'][method_name] = {
            'auroc': float(metrics['auroc']),
            'fpr95': float(metrics['fpr95']),
            'tpr05': float(metrics['tpr05']),
            'cohens_d': float(cohens_d)
        }

        method_metrics.append({
            'method': method_name,
            'auroc': metrics['auroc'],
            'fpr95': metrics['fpr95'],
            'tpr05': metrics['tpr05'],
            'cohens_d': cohens_d
        })

        # Print results
        marker = "★" if method_name == 'tokenflow' else " "
        print(f"{marker} {method_name:<28} {metrics['auroc']*100:>7.1f}% {metrics['fpr95']*100:>7.1f}% {metrics['tpr05']*100:>7.1f}% {cohens_d:>+7.3f}")

    print("-" * 70)

    # =================================
    # KEY COMPARISONS
    # =================================
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    # Compare standard mean vs robust statistics
    baseline_auroc = results['methods']['mink_pp_30']['auroc']
    baseline_d = results['methods']['mink_pp_30']['cohens_d']

    print(f"\nBaseline (Min-K++ 30%): AUROC={baseline_auroc*100:.1f}%, Cohen's d={baseline_d:.3f}")

    comparisons = [
        ('trimmed_mean_30', 'Trimmed Mean'),
        ('winsorized_mean_30', 'Winsorized Mean'),
        ('median_30', 'Median'),
        ('gradient_weighted', 'Gradient Weighted'),
        ('ensemble_robust', 'Robust Ensemble'),
        ('tokenflow', 'TokenFlow (Ours)'),
        # Entropy methods
        ('entropy_mean', 'Entropy Mean'),
        ('entropy_weighted', 'Entropy Weighted'),
        ('entropy_select_30', 'Entropy Select 30%'),
        ('entropy_weighted_mink_30', 'Entropy-Wtd MinK 30%'),
        ('z_div_entropy', 'Z / Entropy'),
        ('low_entropy_only', 'Low Entropy Only'),
    ]

    print("\nOur Methods vs Baseline:")
    for method_key, method_label in comparisons:
        if method_key in results['methods']:
            m = results['methods'][method_key]
            auroc_diff = (m['auroc'] - baseline_auroc) * 100
            d_diff = m['cohens_d'] - baseline_d

            # Statistical significance test
            p_value = statistical_significance_test(
                all_scores[method_key],
                all_scores['mink_pp_30'],
                labels
            )

            sig = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"  {method_label:<20}: AUROC={m['auroc']*100:.1f}% ({auroc_diff:+.1f}%), "
                  f"d={m['cohens_d']:.3f} ({d_diff:+.3f}) {sig}")

    # =================================
    # SAVE RESULTS
    # =================================
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_name.split('/')[-1]

    # Save detailed JSON
    json_path = f"{output_dir}/{dataset_name}_{model_short}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save CSV summary
    df = pd.DataFrame(method_metrics)
    df['auroc_pct'] = (df['auroc'] * 100).round(1).astype(str) + '%'
    df['fpr95_pct'] = (df['fpr95'] * 100).round(1).astype(str) + '%'
    df['tpr05_pct'] = (df['tpr05'] * 100).round(1).astype(str) + '%'

    csv_path = f"{output_dir}/{dataset_name}_{model_short}_summary.csv"
    df[['method', 'auroc_pct', 'fpr95_pct', 'tpr05_pct', 'cohens_d']].to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # Save raw scores for further analysis
    scores_path = f"{output_dir}/{dataset_name}_{model_short}_scores.npz"
    np.savez(scores_path,
             labels=np.array(labels),
             **{k: np.array(v) for k, v in all_scores.items()})
    print(f"Raw scores saved to: {scores_path}")

    return results, all_scores, labels


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TokenFlow: Robust MIA Detection')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-2.8b',
                        help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='WikiMIA_length32',
                        help='Dataset variant (WikiMIA_length32/64/128)')
    parser.add_argument('--output_dir', type=str, default='results_final',
                        help='Output directory')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    global SEED
    SEED = args.seed
    set_seed(SEED)

    # Run experiment
    results, all_scores, labels = run_experiment(
        args.model,
        args.dataset,
        args.output_dir,
        use_half=args.half
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
