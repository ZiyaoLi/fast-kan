# FastKAN V1 vs V2 — MNIST Benchmark

## Setup

```bash
cd experiments
python benchmark_mnist.py --epochs 25 --output-dir results
```

## What it does

- Trains **FastKAN V1** (original) and **FastKAN V2** (new split-weight version) on MNIST for 25 epochs.
- Measures:
  - **Accuracy** (train & val per epoch)
  - **Training speed** (per-epoch time, total time)
  - **Memory** (peak memory via `tracemalloc`)
  - **Parameter count**
- Outputs:
  - `results/<model>_results.json` — full per-epoch data
  - `results/accuracy_loss_time.png` — curves
  - `results/summary_bars.png` — bar chart summary
  - `results/REPORT.md` — markdown report with tables and plot references

## Customization

| Flag | Default | Description |
|---|---|---|
| `--layers` | `784 64 10` | Network widths |
| `--num-grids` | `8` | Number of RBF grid points |
| `--epochs` | `25` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--gamma` | `0.8` | Exponential LR scheduler gamma |
| `--seed` | `42` | Random seed |
| `--output-dir` | `experiments/results` | Output directory |