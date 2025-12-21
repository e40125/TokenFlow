#!/usr/bin/env python3
"""
Run Complete TokenFlow Experiments

This script runs all experiments needed for the paper:
- 6 models x 3 datasets = 18 experiments
- Generates all figures
- Creates summary tables

Usage:
    python run_all_final.py --quick    # Run only pythia-2.8b (fast)
    python run_all_final.py --full     # Run all models (slow)
"""

import subprocess
import sys
import os
import json
import argparse
from datetime import datetime
import pandas as pd


# Model configurations
MODELS_QUICK = [
    'EleutherAI/pythia-2.8b',
]

MODELS_FULL = [
    'EleutherAI/pythia-2.8b',
    'state-spaces/mamba-1.4b-hf',
    'EleutherAI/gpt-neo-2.7B',
    'microsoft/phi-2',
    'facebook/opt-2.7b',
    'meta-llama/Llama-3.2-1B',
]

DATASETS = [
    'WikiMIA_length32',
    'WikiMIA_length64',
    'WikiMIA_length128',
]

OUTPUT_DIR = 'results_final'


def run_experiment(model, dataset, use_half=True):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        'run_tokenflow_final.py',
        '--model', model,
        '--dataset', dataset,
        '--output_dir', OUTPUT_DIR,
        '--seed', '42',
    ]

    if use_half:
        cmd.append('--half')

    print(f"\n{'='*60}")
    print(f"Running: {model.split('/')[-1]} on {dataset}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def generate_summary_table(output_dir):
    """Generate summary table from all results."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY TABLE")
    print("="*60)

    all_results = []

    for model in MODELS_FULL:
        model_short = model.split('/')[-1]
        for dataset in DATASETS:
            json_path = f"{output_dir}/{dataset}_{model_short}_results.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results = json.load(f)

                baseline_auroc = results['methods'].get('mink_pp_30', {}).get('auroc', 0)
                tokenflow_auroc = results['methods'].get('tokenflow', {}).get('auroc', 0)
                baseline_d = results['methods'].get('mink_pp_30', {}).get('cohens_d', 0)
                tokenflow_d = results['methods'].get('tokenflow', {}).get('cohens_d', 0)

                all_results.append({
                    'model': model_short,
                    'dataset': dataset,
                    'baseline_auroc': baseline_auroc * 100,
                    'tokenflow_auroc': tokenflow_auroc * 100,
                    'improvement': (tokenflow_auroc - baseline_auroc) * 100,
                    'baseline_d': baseline_d,
                    'tokenflow_d': tokenflow_d,
                    'd_improvement': tokenflow_d - baseline_d,
                })

    if all_results:
        df = pd.DataFrame(all_results)

        # Print summary
        print("\n" + "-"*80)
        print(f"{'Model':<20} {'Dataset':<18} {'Base':>8} {'Ours':>8} {'Δ':>8} {'d_base':>8} {'d_ours':>8}")
        print("-"*80)

        for _, row in df.iterrows():
            print(f"{row['model']:<20} {row['dataset']:<18} "
                  f"{row['baseline_auroc']:>7.1f}% {row['tokenflow_auroc']:>7.1f}% "
                  f"{row['improvement']:>+7.1f}% "
                  f"{row['baseline_d']:>+7.3f} {row['tokenflow_d']:>+7.3f}")

        print("-"*80)

        # Save to CSV
        df.to_csv(f"{output_dir}/summary_all.csv", index=False)
        print(f"\nSaved to: {output_dir}/summary_all.csv")

        # Calculate overall statistics
        print("\nOVERALL STATISTICS:")
        print(f"  Mean AUROC improvement: {df['improvement'].mean():+.2f}%")
        print(f"  Mean Cohen's d improvement: {df['d_improvement'].mean():+.3f}")
        print(f"  Positive improvements: {(df['improvement'] > 0).sum()}/{len(df)}")

    return all_results


def generate_visualizations(output_dir):
    """Generate all figures."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    cmd = [
        sys.executable,
        'visualize_final.py',
        '--results_dir', output_dir,
        '--output_dir', 'paper/figures',
        '--model', 'EleutherAI/pythia-2.8b',
        '--dataset', 'WikiMIA_length32',
    ]

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick run (pythia only)')
    parser.add_argument('--full', action='store_true', help='Full run (all models)')
    parser.add_argument('--visualize-only', action='store_true', help='Only generate visualizations')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("TokenFlow: Complete Experiment Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if args.visualize_only:
        generate_visualizations(OUTPUT_DIR)
        return

    # Select models
    models = MODELS_QUICK if args.quick else MODELS_FULL

    print(f"\nModels: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"\nDatasets: {len(DATASETS)}")
    for d in DATASETS:
        print(f"  - {d}")

    total = len(models) * len(DATASETS)
    print(f"\nTotal experiments: {total}")

    # Run experiments
    completed = 0
    failed = 0
    results = []

    for model in models:
        for dataset in DATASETS:
            success = run_experiment(model, dataset)
            if success:
                completed += 1
                results.append((model, dataset, 'SUCCESS'))
            else:
                failed += 1
                results.append((model, dataset, 'FAILED'))

            print(f"\nProgress: {completed + failed}/{total} (completed={completed}, failed={failed})")

    # Generate summary
    all_results = generate_summary_table(OUTPUT_DIR)

    # Generate visualizations
    generate_visualizations(OUTPUT_DIR)

    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {completed} completed, {failed} failed")
    print("="*60)


if __name__ == '__main__':
    main()
