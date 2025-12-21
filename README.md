# The Symmetric Influence of Outliers in Membership Inference

Code for reproducing experiments on robust aggregation methods (trimmed mean, winsorized mean) for Min-K%++ membership inference.

## Requirements

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)

## Setup

```bash
conda create -n tokenflow python=3.10 -y
conda activate tokenflow
pip install -r requirements.txt
```

**Note:** Mamba models require `causal-conv1d` and `mamba-ssm` which need CUDA. If you only want to run non-Mamba models, you can comment out those lines in `requirements.txt`.

## Run

```bash
python run_tokenflow_final.py --model EleutherAI/pythia-2.8b --dataset WikiMIA_length32
```

### Run All Models
```bash
python run_all_final.py --quick   # Pythia only (fast)
python run_all_final.py --full    # All 6 models
```

## Supported Models

| Model | HuggingFace ID |
|-------|----------------|
| Pythia-2.8B | `EleutherAI/pythia-2.8b` |
| Mamba-1.4B | `state-spaces/mamba-1.4b-hf` |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` |
| Phi-2 | `microsoft/phi-2` |
| OPT-2.7B | `facebook/opt-2.7b` |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` |

## Results

Results on WikiMIA (length=32) benchmark:

| Model | Method | AUROC (%) | Cohen's d |
|-------|--------|-----------|-----------|
| Pythia-2.8B | Min-K%++ | 63.9 | 0.355 |
| Pythia-2.8B | Trimmed Mean | 63.2 | 0.478 |
| Pythia-2.8B | Winsorized Mean | 63.6 | 0.496 |
| GPT-Neo-2.7B | Min-K%++ | 67.6 | 0.593 |
| GPT-Neo-2.7B | Trimmed Mean | 66.2 | 0.560 |
| GPT-Neo-2.7B | Winsorized Mean | 66.7 | 0.579 |
| Mamba-1.4B | Min-K%++ | 66.4 | 0.546 |
| Mamba-1.4B | Trimmed Mean | 65.9 | 0.554 |
| Mamba-1.4B | Winsorized Mean | 66.2 | 0.569 |
| OPT-2.7B | Min-K%++ | 62.6 | 0.350 |
| OPT-2.7B | Trimmed Mean | 62.4 | 0.393 |
| OPT-2.7B | Winsorized Mean | 63.0 | 0.410 |
| Phi-2 | Min-K%++ | 66.1 | 0.448 |
| Phi-2 | Trimmed Mean | 66.8 | 0.553 |
| Phi-2 | Winsorized Mean | 67.0 | 0.552 |

## Output

Results are saved to `results_final/`:
- `*_summary.csv`: AUROC, FPR@95, TPR@05, Cohen's d
- `*_scores.npz`: Raw scores
- `*_results.json`: Full config
