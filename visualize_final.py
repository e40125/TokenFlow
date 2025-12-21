#!/usr/bin/env python3
"""
Generate figures from experimental results.

Usage:
    python visualize_final.py --results_dir results_final
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# Publication-quality settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme
COLORS = {
    'baseline': '#7f8c8d',
    'mink': '#3498db',
    'trimmed': '#27ae60',
    'winsorized': '#2ecc71',
    'gradient': '#e74c3c',
    'ensemble': '#9b59b6',
    'tokenflow': '#f39c12',
    'member': '#27ae60',
    'non_member': '#c0392b',
}


def load_results(results_dir, dataset, model):
    """Load experiment results."""
    model_short = model.split('/')[-1]

    # Load JSON results
    json_path = f"{results_dir}/{dataset}_{model_short}_results.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = None

    # Load raw scores
    scores_path = f"{results_dir}/{dataset}_{model_short}_scores.npz"
    if os.path.exists(scores_path):
        data = np.load(scores_path)
        scores = {key: data[key] for key in data.files}
    else:
        scores = None

    return results, scores


def fig1_method_comparison(results, scores, output_dir):
    """Figure 1: Method Comparison (AUROC bar chart)."""
    if results is None:
        print("No results data available for Figure 1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Select methods to compare
    methods_to_plot = [
        ('loss', 'Loss', COLORS['baseline']),
        ('zlib', 'Zlib', COLORS['baseline']),
        ('mink_pp_20', 'Min-K++ 20%', COLORS['mink']),
        ('mink_pp_30', 'Min-K++ 30%', COLORS['mink']),
        ('trimmed_mean_30', 'Trimmed 30%', COLORS['trimmed']),
        ('winsorized_mean_30', 'Winsor. 30%', COLORS['winsorized']),
        ('gradient_weighted', 'Grad. Weight', COLORS['gradient']),
        ('ensemble_robust', 'Ensemble', COLORS['ensemble']),
        ('tokenflow', 'TokenFlow', COLORS['tokenflow']),
    ]

    # Filter to available methods
    available = [(k, l, c) for k, l, c in methods_to_plot if k in results['methods']]

    # Panel A: AUROC
    ax1 = axes[0]
    x = np.arange(len(available))
    aurocs = [results['methods'][k]['auroc'] * 100 for k, _, _ in available]
    colors = [c for _, _, c in available]
    labels = [l for _, l, _ in available]

    bars = ax1.bar(x, aurocs, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

    # Baseline reference line
    baseline_auroc = results['methods']['mink_pp_30']['auroc'] * 100
    ax1.axhline(y=baseline_auroc, color='navy', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')

    ax1.set_ylabel('AUROC (%)')
    ax1.set_title('(a) AUROC Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(min(aurocs) - 3, max(aurocs) + 2)
    ax1.legend(loc='lower right')

    # Value labels
    for bar, val in zip(bars, aurocs):
        ax1.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel B: Cohen's d
    ax2 = axes[1]
    cohens_d = [results['methods'][k]['cohens_d'] for k, _, _ in available]

    bars2 = ax2.bar(x, cohens_d, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

    baseline_d = results['methods']['mink_pp_30']['cohens_d']
    ax2.axhline(y=baseline_d, color='navy', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')

    ax2.set_ylabel("Cohen's d (Effect Size)")
    ax2.set_title("(b) Discriminative Power (Cohen's d)", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='lower right')

    # Value labels
    for bar, val in zip(bars2, cohens_d):
        ax2.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    fig.savefig(f'{output_dir}/fig1_method_comparison.pdf', bbox_inches='tight')
    fig.savefig(f'{output_dir}/fig1_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig1_method_comparison.pdf/png")


def fig2_roc_curves(results, scores, output_dir):
    """Figure 2: ROC Curves."""
    if scores is None:
        print("No scores data available for Figure 2")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    labels = scores['labels']

    # Methods to plot
    methods = [
        ('loss', 'Loss', COLORS['baseline'], '--'),
        ('mink_pp_30', 'Min-K++ 30%', COLORS['mink'], '-'),
        ('trimmed_mean_30', 'Trimmed Mean 30%', COLORS['trimmed'], '-'),
        ('tokenflow', 'TokenFlow (Ours)', COLORS['tokenflow'], '-'),
    ]

    for method_key, method_label, color, linestyle in methods:
        if method_key not in scores:
            continue

        method_scores = scores[method_key]

        # Handle NaN
        mask = np.isfinite(method_scores)
        if not mask.all():
            method_scores = np.where(mask, method_scores, np.nanmedian(method_scores))

        fpr, tpr, _ = roc_curve(labels, method_scores)
        auroc = auc(fpr, tpr)

        lw = 3 if method_key == 'tokenflow' else 2
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=lw,
                label=f'{method_label} (AUROC={auroc*100:.1f}%)')

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Pre-training Data Detection', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    fig.savefig(f'{output_dir}/fig2_roc_curves.pdf', bbox_inches='tight')
    fig.savefig(f'{output_dir}/fig2_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig2_roc_curves.pdf/png")


def fig3_score_distribution(results, scores, output_dir):
    """Figure 3: Score Distributions."""
    if scores is None:
        print("No scores data available for Figure 3")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    labels = scores['labels']

    methods = [
        ('mink_pp_30', 'Min-K++ 30%'),
        ('trimmed_mean_30', 'Trimmed Mean 30%'),
        ('tokenflow', 'TokenFlow'),
    ]

    for idx, (method_key, method_label) in enumerate(methods):
        ax = axes[idx]

        if method_key not in scores:
            ax.set_title(f'{method_label} (no data)')
            continue

        method_scores = scores[method_key]

        member_scores = method_scores[labels == 1]
        non_member_scores = method_scores[labels == 0]

        # Remove NaN
        member_scores = member_scores[np.isfinite(member_scores)]
        non_member_scores = non_member_scores[np.isfinite(non_member_scores)]

        # Plot histograms
        ax.hist(member_scores, bins=30, alpha=0.6, color=COLORS['member'],
                label='Training Data', density=True, edgecolor='black', linewidth=0.3)
        ax.hist(non_member_scores, bins=30, alpha=0.6, color=COLORS['non_member'],
                label='Non-training Data', density=True, edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'({chr(97+idx)}) {method_label}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        # Add statistics
        sep = np.mean(member_scores) - np.mean(non_member_scores)
        ax.text(0.02, 0.98, f'Mean sep: {sep:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    fig.savefig(f'{output_dir}/fig3_score_distribution.pdf', bbox_inches='tight')
    fig.savefig(f'{output_dir}/fig3_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig3_score_distribution.pdf/png")


def fig4_cohens_d_analysis(results, output_dir):
    """
    Figure 4: Cohen's d Analysis
    Horizontal bar chart showing effect sizes.
    """
    if results is None:
        print("No results data available for Figure 4")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all methods with Cohen's d
    methods_data = []
    for method_name, metrics in results['methods'].items():
        methods_data.append({
            'method': method_name,
            'cohens_d': metrics['cohens_d'],
            'auroc': metrics['auroc']
        })

    # Sort by Cohen's d
    methods_data = sorted(methods_data, key=lambda x: x['cohens_d'], reverse=True)

    # Take top 15
    methods_data = methods_data[:15]

    y = np.arange(len(methods_data))
    cohens_d = [m['cohens_d'] for m in methods_data]
    method_names = [m['method'].replace('_', ' ').title() for m in methods_data]

    # Color by method type
    colors = []
    for m in methods_data:
        name = m['method']
        if 'trimmed' in name or 'winsor' in name or 'median' in name:
            colors.append(COLORS['trimmed'])
        elif 'gradient' in name:
            colors.append(COLORS['gradient'])
        elif 'ensemble' in name:
            colors.append(COLORS['ensemble'])
        elif 'tokenflow' in name:
            colors.append(COLORS['tokenflow'])
        elif 'mink' in name:
            colors.append(COLORS['mink'])
        else:
            colors.append(COLORS['baseline'])

    bars = ax.barh(y, cohens_d, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

    # Baseline reference
    baseline_d = results['methods'].get('mink_pp_30', {}).get('cohens_d', 0)
    ax.axvline(x=baseline_d, color='navy', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_ylabel('Method')
    ax.set_title("Feature Discriminative Power (Cohen's d)", fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(method_names, fontsize=9)

    # Value labels
    for bar, val in zip(bars, cohens_d):
        ax.annotate(f'{val:.3f}',
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['trimmed'], label='Robust Statistics'),
        mpatches.Patch(color=COLORS['gradient'], label='Gradient-based'),
        mpatches.Patch(color=COLORS['mink'], label='Min-K++ Family'),
        mpatches.Patch(color=COLORS['baseline'], label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    fig.savefig(f'{output_dir}/fig4_cohens_d.pdf', bbox_inches='tight')
    fig.savefig(f'{output_dir}/fig4_cohens_d.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig4_cohens_d.pdf/png")


def fig5_k_sensitivity(results_dir, model, output_dir):
    """
    Figure 5: Sensitivity to K selection
    Compare different k values for standard vs trimmed mean.
    """
    # Load all three dataset results
    datasets = ['WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128']
    model_short = model.split('/')[-1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        json_path = f"{results_dir}/{dataset}_{model_short}_results.json"
        if not os.path.exists(json_path):
            ax.set_title(f'{dataset} (no data)')
            continue

        with open(json_path, 'r') as f:
            results = json.load(f)

        k_values = [10, 20, 30, 40, 50]

        # Standard Min-K++
        mink_aurocs = []
        for k in k_values:
            key = f'mink_pp_{k}'
            if key in results['methods']:
                mink_aurocs.append(results['methods'][key]['auroc'] * 100)
            else:
                mink_aurocs.append(np.nan)

        # Trimmed mean
        trimmed_aurocs = []
        for k in [20, 30, 40]:
            key = f'trimmed_mean_{k}'
            if key in results['methods']:
                trimmed_aurocs.append(results['methods'][key]['auroc'] * 100)
            else:
                trimmed_aurocs.append(np.nan)

        ax.plot(k_values, mink_aurocs, 'o-', color=COLORS['mink'],
                linewidth=2, markersize=8, label='Min-K++ (standard)')
        ax.plot([20, 30, 40], trimmed_aurocs, 's-', color=COLORS['trimmed'],
                linewidth=2, markersize=8, label='Trimmed Mean')

        length = dataset.split('_')[1]
        ax.set_xlabel('K (%)')
        ax.set_ylabel('AUROC (%)')
        ax.set_title(f'({chr(97+idx)}) {length}', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(5, 55)

    plt.tight_layout()

    fig.savefig(f'{output_dir}/fig5_k_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(f'{output_dir}/fig5_k_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fig5_k_sensitivity.pdf/png")


def generate_latex_tables(results_dir, models, datasets, output_dir):
    """Generate LaTeX tables for the paper."""

    # Main results table
    table_rows = []

    for model in models:
        model_short = model.split('/')[-1]
        row = [model_short]

        for dataset in datasets:
            json_path = f"{results_dir}/{dataset}_{model_short}_results.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results = json.load(f)

                baseline = results['methods'].get('mink_pp_30', {}).get('auroc', 0) * 100
                tokenflow = results['methods'].get('tokenflow', {}).get('auroc', 0) * 100

                row.append(f"{baseline:.1f}")
                row.append(f"\\textbf{{{tokenflow:.1f}}}")
            else:
                row.append("-")
                row.append("-")

        table_rows.append(" & ".join(row) + " \\\\")

    # Write to file
    with open(f"{output_dir}/table_main_results.tex", 'w') as f:
        f.write("% Auto-generated table\n")
        f.write("\\begin{tabular}{l|cc|cc|cc}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{2}{c|}{length32} & \\multicolumn{2}{c|}{length64} & \\multicolumn{2}{c}{length128} \\\\\n")
        f.write("Model & Base & Ours & Base & Ours & Base & Ours \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            f.write(row + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"  Saved: table_main_results.tex")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_final')
    parser.add_argument('--output_dir', type=str, default='paper/figures')
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-2.8b')
    parser.add_argument('--dataset', type=str, default='WikiMIA_length32')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Visualizations from Real Data")
    print("=" * 60)

    # Load results
    results, scores = load_results(args.results_dir, args.dataset, args.model)

    if results is None:
        print(f"\nNo results found in {args.results_dir}")
        print("Please run experiments first: python run_tokenflow_final.py")
        return

    print(f"\nLoaded results for {args.model} on {args.dataset}")

    # Generate figures
    print("\nGenerating figures...")
    fig1_method_comparison(results, scores, args.output_dir)
    fig2_roc_curves(results, scores, args.output_dir)
    fig3_score_distribution(results, scores, args.output_dir)
    fig4_cohens_d_analysis(results, args.output_dir)
    fig5_k_sensitivity(args.results_dir, args.model, args.output_dir)

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)


if __name__ == '__main__':
    main()
