"""
Plotting script for the GRPO MUTAG project.

Run this AFTER your 100-step training finishes. It expects three files
that the patched pipeline writes:
  - grpo_output/reward_log.json     (per-step reward / std / completion_len)
  - grpo_output/eval_baseline.json  (batch eval, no training)
  - grpo_output/eval_sft.json       (batch eval, after SFT warmup)
  - grpo_output/eval_grpo.json      (batch eval, after SFT + GRPO)

Outputs four PNG files into grpo_output/plots/:
  1. reward_curve.png      -- reward vs training step
  2. metric_bars.png       -- per-metric bars for the three model versions
  3. score_distribution.png -- box plot of overall scores across graphs
  4. graph_comparison.png  -- target / baseline / SFT+GRPO renders for 2 graphs

Usage:
  python make_plots.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUT_DIR = Path("grpo_output")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "num_nodes", "num_edges", "is_connected",
    "clustering_coefficient", "diameter", "avg_cycle_length",
]
METRIC_LABELS = {
    "num_nodes": "Nodes",
    "num_edges": "Edges",
    "is_connected": "Connected",
    "clustering_coefficient": "Clustering",
    "diameter": "Diameter",
    "avg_cycle_length": "Cycle len",
}


# -----------------------------------------------------------------------------
# 1. REWARD LEARNING CURVE
# -----------------------------------------------------------------------------
def _rolling_mean(values, window):
    """Centered rolling mean with edge-handling so the line spans the full x-axis."""
    if len(values) == 0:
        return []
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_reward_curve():
    log_path = OUT_DIR / "reward_log.json"
    if not log_path.exists():
        print(f"Skipping reward curve: {log_path} not found.")
        return

    log = json.loads(log_path.read_text())
    if not log:
        print("Skipping reward curve: log is empty.")
        return

    steps = [r["step"] for r in log]
    rewards = [r["reward"] for r in log]
    stds = [r.get("reward_std", 0) for r in log]

    # Smoothed trend line: 10-step rolling mean (or 1/10th of total, whichever
    # is smaller, to keep tiny-runs sensible). Helps the eye see the trend
    # through the noise of per-step GRPO variance.
    window = max(3, min(10, len(rewards) // 10))
    smooth = _rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Raw per-step reward (faint)
    ax.plot(steps, rewards, color="steelblue", linewidth=1.0, alpha=0.4,
            label="Mean reward (per step)")
    # +/- 1 std band
    ax.fill_between(steps,
                    [r - s for r, s in zip(rewards, stds)],
                    [r + s for r, s in zip(rewards, stds)],
                    color="steelblue", alpha=0.15, label="+/- 1 std")
    # Smoothed trend line (bold)
    ax.plot(steps, smooth, color="darkblue", linewidth=2.5,
            label=f"Smoothed (rolling {window}-step mean)")

    # Overlay the SFT batch-eval baseline as a horizontal reference line.
    sft_path = OUT_DIR / "eval_sft.json"
    if sft_path.exists():
        sft = json.loads(sft_path.read_text())
        sft_overall = sft.get("mean_overall")
        if sft_overall is not None:
            ax.axhline(sft_overall, color="gray", linestyle="--", linewidth=1.5,
                       label=f"SFT eval baseline ({sft_overall:.3f})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Training: Reward over Steps")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Auto-zoom the y-axis to the actual reward range so trend is visible,
    # while keeping a little headroom and never cropping the std band.
    lo = max(0.0, min(rewards) - max(stds) - 0.05)
    hi = min(1.05, max(rewards) + max(stds) + 0.05)
    ax.set_ylim(lo, hi)

    out = PLOT_DIR / "reward_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# 2. PER-METRIC BARS
# -----------------------------------------------------------------------------
def plot_metric_bars():
    paths = {
        "Baseline": OUT_DIR / "eval_baseline.json",
        "SFT only": OUT_DIR / "eval_sft.json",
        "SFT + GRPO": OUT_DIR / "eval_grpo.json",
    }
    versions = {k: json.loads(p.read_text()) for k, p in paths.items()
                if p.exists()}
    if not versions:
        print("Skipping metric bars: no eval_*.json files found.")
        return

    x = np.arange(len(METRICS))
    width = 0.8 / max(len(versions), 1)
    colors = {"Baseline": "lightgray", "SFT only": "tan",
              "SFT + GRPO": "tomato"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, data) in enumerate(versions.items()):
        per_metric = data.get("per_metric_mean", {})
        values = [per_metric.get(m, 0) for m in METRICS]
        ax.bar(x + i * width, values, width, label=name,
               color=colors.get(name, None), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width * (len(versions) - 1) / 2)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15)
    ax.set_ylabel("Mean score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-metric Score by Training Stage (averaged over eval graphs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = PLOT_DIR / "metric_bars.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# 3. SCORE DISTRIBUTION
# -----------------------------------------------------------------------------
def plot_score_distribution():
    paths = {
        "Baseline": OUT_DIR / "eval_baseline.json",
        "SFT only": OUT_DIR / "eval_sft.json",
        "SFT + GRPO": OUT_DIR / "eval_grpo.json",
    }
    versions = {k: json.loads(p.read_text()) for k, p in paths.items()
                if p.exists()}
    if not versions:
        print("Skipping score distribution: no eval_*.json files found.")
        return

    data = []
    labels = []
    means = []
    for name, d in versions.items():
        scores = d.get("overall_per_graph", [])
        if scores:
            data.append(scores)
            labels.append(f"{name}\n(n={len(scores)})")
            means.append(sum(scores) / len(scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    # `tick_labels` was renamed from `labels` in matplotlib 3.9; fall back if needed.
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
    for patch, color in zip(bp["boxes"], ["lightgray", "tan", "tomato"]):
        patch.set_facecolor(color)

    # Annotate mean above each box -- helps the reader see small differences
    # that are visually subtle in the boxes themselves.
    for i, (m, scores) in enumerate(zip(means, data), start=1):
        upper = max(scores)
        ax.text(i, min(1.02, upper + 0.04), f"mean={m:.3f}",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="black")

    ax.set_ylabel("Weighted overall score")
    ax.set_ylim(0, 1.10)
    ax.set_title("Distribution of Overall Scores Across Eval Graphs")
    ax.grid(axis="y", alpha=0.3)

    out = PLOT_DIR / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
# 4. GRAPH RENDERS  (target vs baseline vs SFT+GRPO for 2 graphs)
# -----------------------------------------------------------------------------
def plot_graph_renders():
    paths = {
        "Baseline": OUT_DIR / "eval_baseline.json",
        "SFT + GRPO": OUT_DIR / "eval_grpo.json",
    }
    versions = {k: json.loads(p.read_text()) for k, p in paths.items()
                if p.exists()}
    if not versions:
        print("Skipping graph renders: no eval_*.json files found.")
        return

    # Use the first two stored graphs from the GRPO run as examples.
    grpo = versions.get("SFT + GRPO")
    if grpo is None or "samples" not in grpo:
        print("Skipping graph renders: no per-graph 'samples' stored.")
        return
    samples = grpo["samples"][:2]
    if not samples:
        return

    n_examples = len(samples)
    n_cols = 1 + len(versions)  # target + each model
    fig, axes = plt.subplots(n_examples, n_cols,
                             figsize=(4 * n_cols, 4 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for row, sample in enumerate(samples):
        idx = sample["graph_index"]

        # Target
        G_target = nx.Graph()
        G_target.add_nodes_from(sample["target_nodes"])
        G_target.add_edges_from(sample["target_edges"])
        pos = nx.spring_layout(G_target, seed=42)
        nx.draw_networkx(G_target, pos, ax=axes[row][0], with_labels=True,
                         node_color="steelblue", node_size=400,
                         font_color="white", font_size=8)
        axes[row][0].set_title(f"Target (graph #{idx})")
        axes[row][0].set_axis_off()

        for col, (name, data) in enumerate(versions.items(), start=1):
            sample_n = next((s for s in data.get("samples", [])
                             if s["graph_index"] == idx), None)
            ax = axes[row][col]
            if sample_n and sample_n.get("gen_nodes"):
                G = nx.Graph()
                G.add_nodes_from(sample_n["gen_nodes"])
                G.add_edges_from(sample_n["gen_edges"])
                pos2 = nx.spring_layout(G, seed=42)
                color = "tomato" if "GRPO" in name else "lightgray"
                nx.draw_networkx(G, pos2, ax=ax, with_labels=True,
                                 node_color=color, node_size=400,
                                 font_color="white", font_size=8)
                score = sample_n.get("overall", 0)
                ax.set_title(f"{name}\nscore={score:.2f}")
            else:
                ax.text(0.5, 0.5, "(failed to parse)", ha="center", va="center")
                ax.set_title(name)
            ax.set_axis_off()

    out = PLOT_DIR / "graph_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    plot_reward_curve()
    plot_metric_bars()
    plot_score_distribution()
    plot_graph_renders()
    print(f"\nAll plots in: {PLOT_DIR.resolve()}")