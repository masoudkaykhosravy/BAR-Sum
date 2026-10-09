#!/usr/bin/env python3
"""Generate an IEEE-style illustration of adaptive BAR-Sum objective weights.

The trajectories are synthetic and deterministic; they are illustrative rather than
measurements from a training run. The bounded panel uses a bounded-simplex projection
with total weight 0.95 to match the paper's reported final weights (which sum to 0.95).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 17
LOWER, UPPER = 0.10, 0.45
FINAL_WEIGHTS = np.array([0.214, 0.450, 0.175, 0.111])
TOTAL_WEIGHT = float(FINAL_WEIGHTS.sum())
EPOCHS = np.arange(1, 6)


def project_bounded_simplex(values, lower=LOWER, upper=UPPER, total=TOTAL_WEIGHT):
    """Clip and renormalize values onto {x: lower<=x<=upper, sum(x)=total}."""
    values = np.asarray(values, dtype=float)
    if not (len(values) * lower <= total <= len(values) * upper):
        raise ValueError("Requested total is infeasible for the supplied bounds")
    # Projection is clip(values + shift); bisection finds the shift that meets total.
    lo = lower - float(np.max(values)) - 1.0
    hi = upper - float(np.min(values)) + 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if np.clip(values + mid, lower, upper).sum() < total:
            lo = mid
        else:
            hi = mid
    return np.clip(values + (lo + hi) / 2.0, lower, upper)


def make_bounded_trajectories():
    rng = np.random.default_rng(SEED)
    start = np.array([0.300, 0.200, 0.280, 0.170])
    # Smooth, plausible movement toward the reported final weights, with small
    # seeded perturbations at intermediate epochs only.
    fractions = np.linspace(0.0, 1.0, len(EPOCHS))[:, None]
    raw = start[None, :] + fractions * (FINAL_WEIGHTS - start)[None, :]
    raw[1:-1] += rng.normal(0.0, 0.012, size=(len(EPOCHS) - 2, 4))
    result = np.vstack([project_bounded_simplex(row) for row in raw])
    # The specified final vector is already feasible, and should remain exact.
    result[-1] = FINAL_WEIGHTS
    return result


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "scripts" else script_dir
    pdf_path = project_root / "figures" / "fig_weight_dynamics.pdf"
    png_path = project_root / "logs_and_results" / "reward_weights_curve.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # IEEE-friendly serif styling; vector PDF retains sharp labels and lines.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.65,
    })
    colors = {
        "cov": "#0072B2", "faith": "#D55E00",
        "flu": "#009E73", "red": "#6A5ACD",
        "bound": "#555555",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.80, wspace=0.24)

    # (a) Unconstrained dynamics: faithfulness collapses while redundancy dominates.
    unconstrained = {
        r"$w_{cov}$": np.array([0.24, 0.29, 0.34, 0.38, 0.41]),
        r"$w_{faith}$": np.array([0.30, 0.19, 0.095, 0.035, 0.008]),
        r"$w_{flu}$": np.array([0.27, 0.26, 0.24, 0.22, 0.20]),
        r"$w_{red}$": np.array([0.19, 0.26, 0.34, 0.43, 0.50]),
    }
    palette = [colors["cov"], colors["faith"], colors["flu"], colors["red"]]
    for (label, values), color in zip(unconstrained.items(), palette):
        ax1.plot(EPOCHS, values, marker="o", ms=3.2, color=color, label=label)
    # Dashed unconstrained baseline highlights the collapse trajectory.
    ax1.plot(EPOCHS, unconstrained[r"$w_{faith}$"], color=colors["faith"],
             linestyle="--", linewidth=1.0, alpha=0.75,
             label="Unconstrained baseline")
    ax1.set_title("Unconstrained Adaptive (No Bounds)", pad=7)
    ax1.text(0.04, 0.06, "Faithfulness collapse → mimic policy", transform=ax1.transAxes,
             fontsize=7, color=colors["faith"], va="bottom")
    ax1.set_ylim(-0.03, 0.56)

    # (b) Bounded, renormalized weights; final vector equals reported learned values.
    bounded = make_bounded_trajectories()
    labels = [r"$w_{cov}$", r"$w_{faith}$", r"$w_{flu}$", r"$w_{red}$"]
    for col, (label, color) in enumerate(zip(labels, palette)):
        ax2.plot(EPOCHS, bounded[:, col], marker="o", ms=3.2, color=color, label=label)
    ax2.axhline(LOWER, color=colors["bound"], linestyle="--", linewidth=0.85,
                label="Bounds (0.10, 0.45)")
    ax2.axhline(UPPER, color=colors["bound"], linestyle="--", linewidth=0.85)
    ax2.set_title("Proposed Bounded Adaptive ($w_{min}=0.10$, $w_{max}=0.45$)", pad=7)
    ax2.set_ylim(0.06, 0.49)
    ax2.annotate("0.450", xy=(5, FINAL_WEIGHTS[1]), xytext=(4.48, 0.405),
                 fontsize=7, color=colors["faith"],
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": colors["faith"]})

    for ax, panel in ((ax1, "(a)"), (ax2, "(b)")):
        ax.set_xlabel("Training epoch")
        ax.set_xticks(EPOCHS)
        ax.set_xlim(0.85, 5.15)
        ax.set_ylabel("Objective weight")
        ax.grid(True, color="#b8b8b8", linewidth=0.45, alpha=0.42)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.11, 1.12, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left")
    ax1.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92,
               edgecolor="#cccccc", borderpad=0.35, columnspacing=0.8,
               handlelength=1.6, labelspacing=0.3)
    ax2.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92,
               edgecolor="#cccccc", borderpad=0.35, columnspacing=0.8,
               handlelength=1.6, labelspacing=0.3)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, format="png", dpi=350, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote PNG: {png_path}")
    print("Bounded final weights [w_cov, w_faith, w_flu, w_red]: " +
          np.array2string(FINAL_WEIGHTS, precision=3))
    print(f"Bounded weights: min={bounded.min():.3f}, max={bounded.max():.3f}; "
          f"row sums={np.array2string(bounded.sum(axis=1), precision=3)}")

if __name__ == "__main__":
    main()
