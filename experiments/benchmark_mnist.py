# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script: FastKAN (V1) vs FastKANV2 on MNIST.
Measures accuracy, training speed, memory usage, and parameter count.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import tracemalloc
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_fastkan_v1(layers_hidden, num_grids=8):
    """Create the original FastKAN model."""
    from fastkan import FastKAN as FastKANV1
    return FastKANV1(layers_hidden, num_grids=num_grids)


def make_fastkan_v2(layers_hidden, num_grids=8):
    """Create the new FastKANV2 model."""
    from fastkan.fastkan import FastKANV2
    return FastKANV2(layers_hidden, num_grids=num_grids)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, trainloader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for images, labels in trainloader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def evaluate(model, valloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for images, labels in valloader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        output = model(images)
        total_loss += criterion(output, labels).item()
        total_acc += (output.argmax(dim=1) == labels).float().mean().item()
        num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_name, make_model_fn, layers, num_grids, num_epochs,
                  batch_size, lr, gamma, seed, output_dir):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    model = make_model_fn(layers, num_grids=num_grids)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    num_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Params: {num_params:,}")
    print(f"{'='*50}\n")

    # Measure peak memory via tracemalloc
    tracemalloc.start()
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, trainloader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, valloader, device, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.2f}s")

    # Final memory snapshot
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_epoch_time = np.mean(epoch_times)
    total_train_time = np.sum(epoch_times)

    results = {
        "model_name": model_name,
        "num_parameters": num_params,
        "final_train_acc": float(train_accs[-1]),
        "final_val_acc": float(val_accs[-1]),
        "best_val_acc": float(max(val_accs)),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "avg_epoch_time_s": float(avg_epoch_time),
        "total_train_time_s": float(total_train_time),
        "peak_memory_bytes": int(peak_mem),
        "peak_memory_mb": float(peak_mem / (1024 * 1024)),
        "train_accs": [float(a) for a in train_accs],
        "val_accs": [float(a) for a in val_accs],
        "train_losses": [float(l) for l in train_losses],
        "val_losses": [float(l) for l in val_losses],
        "epoch_times": [float(t) for t in epoch_times],
    }

    # Save results JSON
    results_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results_list, output_dir):
    """Generate comparison plots from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)

    names = [r["model_name"] for r in results_list]
    colors = ["#2196F3", "#FF5722"]

    # --- Accuracy curve ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    ax = axes[0]
    for r, name, color in zip(results_list, names, colors):
        ax.plot(range(1, len(r["val_accs"]) + 1), r["val_accs"],
                label=name, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training Loss
    ax = axes[1]
    for r, name, color in zip(results_list, names, colors):
        ax.plot(range(1, len(r["train_losses"]) + 1), r["train_losses"],
                label=name, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-epoch time
    ax = axes[2]
    for r, name, color in zip(results_list, names, colors):
        ax.plot(range(1, len(r["epoch_times"]) + 1), r["epoch_times"],
                label=name, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title("Epoch Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_loss_time.png"), dpi=150)
    plt.close()

    # --- Bar chart: summary metrics ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    metrics = [
        ("Final Val Acc", "final_val_acc", True),
        ("Best Val Acc", "best_val_acc", True),
        ("Params (K)", "num_parameters", False),
        ("Peak Mem (MB)", "peak_memory_mb", False),
    ]

    for mi, (title, key, higher_better) in enumerate(metrics):
        ax = axes[mi]
        vals = [r[key] for r in results_list]
        bars = ax.bar(names, vals, color=colors[:len(names)])
        ax.set_title(title)
        ax.set_ylabel(title if "Acc" not in title else "Accuracy")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * (1 if higher_better else 100),
                    f"{v:.3f}" if higher_better else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_bars.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_md_report(results_list, output_dir):
    """Generate a Markdown report with tables and references to plots."""
    lines = []
    lines.append("# FastKAN V1 vs V2 — MNIST Benchmark Report\n")
    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append("| Dataset | MNIST |")
    lines.append("| Input | 28×28 = 784 |")
    lines.append("| Epochs | 25 |")
    lines.append("| Batch Size | 64 |")
    lines.append("| Optimizer | AdamW (lr=1e-3, wd=1e-4) |")
    lines.append("| Scheduler | ExponentialLR (gamma=0.8) |")
    lines.append("\n")

    # Results table
    lines.append("## Results Summary\n")
    lines.append("| Metric | " + " | ".join(r["model_name"] for r in results_list) + " |")
    lines.append("|---|---|" + "---|" * (len(results_list) - 1))

    metric_keys = [
        ("Parameters", "num_parameters", "{:,}"),
        ("Final Train Acc", "final_train_acc", "{:.4f}"),
        ("Final Val Acc", "final_val_acc", "{:.4f}"),
        ("Best Val Acc", "best_val_acc", "{:.4f}"),
        ("Final Train Loss", "final_train_loss", "{:.4f}"),
        ("Final Val Loss", "final_val_loss", "{:.4f}"),
        ("Avg Epoch Time (s)", "avg_epoch_time_s", "{:.3f}"),
        ("Total Train Time (s)", "total_train_time_s", "{:.2f}"),
        ("Peak Memory (MB)", "peak_memory_mb", "{:.2f}"),
    ]

    for label, key, fmt in metric_keys:
        row = f"| {label} |"
        for r in results_list:
            row += f" `{fmt.format(r[key])}` |"
        lines.append(row)

    lines.append("\n")

    # Per-epoch table (abbreviated)
    lines.append("## Per-Epoch Validation Accuracy\n")
    lines.append("| Epoch | " + " | ".join(r["model_name"] for r in results_list) + " |")
    lines.append("|---|---|" + "---|" * (len(results_list) - 1))
    for epoch_idx in range(len(results_list[0]["val_accs"])):
        row = f"| {epoch_idx + 1} |"
        for r in results_list:
            row += f" `{r['val_accs'][epoch_idx]:.4f}` |"
        lines.append(row)

    lines.append("\n")
    lines.append("## Plots\n")
    lines.append("- ![Accuracy, Loss & Time curves](accuracy_loss_time.png)\n")
    lines.append("- ![Summary bar charts](summary_bars.png)\n")

    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FastKAN V1 vs V2 MNIST Benchmark")
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 64, 10],
                        help="Layer widths, e.g. 784 64 10")
    parser.add_argument("--num-grids", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    make_fn_v1 = lambda layers, num_grids: make_fastkan_v1(layers, num_grids=num_grids)
    make_fn_v2 = lambda layers, num_grids: make_fastkan_v2(layers, num_grids=num_grids)

    results = []
    results.append(run_benchmark(
        "FastKAN V1", make_fn_v1, args.layers, args.num_grids,
        args.epochs, args.batch_size, args.lr, args.gamma, args.seed, args.output_dir
    ))
    results.append(run_benchmark(
        "FastKAN V2", make_fn_v2, args.layers, args.num_grids,
        args.epochs, args.batch_size, args.lr, args.gamma, args.seed, args.output_dir
    ))

    plot_results(results, args.output_dir)
    generate_md_report(results, args.output_dir)

    print("\n=== DONE ===")
    print(f"Check {args.output_dir}/REPORT.md for the full report.")


if __name__ == "__main__":
    main()