from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "convergence_audit"
SOURCE = OUTPUT / "source"

CAMPAIGN_SCREEN = (
    ROOT
    / "outputs"
    / "seaf_architecture_screen_remote"
    / "b717f069fbb1481e52cb9bd537f910810a4e9fd6_seaf_arch_screen30"
    / "screen"
)
CAMPAIGN_CONFIRM = (
    ROOT
    / "outputs"
    / "seaf_lcff_confirmation_remote"
    / "72f0e0ed04bed67c4bd8ba7aed888e91505235770e0acbaeb6d31d278e651b92_seaf_lcff_confirmation"
    / "confirm_validation"
)
CAMPAIGN_H192 = (
    ROOT
    / "outputs"
    / "seaf_h192_confirmation_remote"
    / "f41a17ca1120c04c7c2c7881aff74835c37e0e5e_seaf_h192_confirmation"
    / "confirm_validation"
    / "seaf_h192"
)

MODEL_PATHS = {
    "SEAF": {
        seed: CAMPAIGN_H192 / f"seed_{seed}" / "run_summary.json"
        for seed in (42, 123, 3407)
    },
    "FourCastNet": {
        seed: SOURCE / f"ofb_fourcastnet_anomaly_seed{seed}_run_summary.json"
        for seed in (42, 123, 3407)
    },
    "ClimaX": {
        seed: SOURCE / f"ofb_climax_anomaly_seed{seed}_run_summary.json"
        for seed in (42, 123, 3407)
    },
    "Swin": {
        seed: SOURCE / f"ofb_swin_anomaly_seed{seed}_run_summary.json"
        for seed in (42, 123, 3407)
    },
}

COLORS = {42: "#0F4D92", 123: "#42949E", 3407: "#B64342"}
MARKERS = {42: "o", 123: "s", 3407: "^"}


def load_curves(paths: dict[int, Path]) -> dict[int, np.ndarray]:
    curves = {}
    for seed, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        values = np.asarray(record["validation_selection_losses"], dtype=float)
        if values.shape != (30,) or not np.isfinite(values).all():
            raise ValueError(f"Unexpected validation curve in {path}: {values.shape}")
        curves[seed] = values
    return curves


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_panel(ax: plt.Axes, model: str, curves: dict[int, np.ndarray], show_legend: bool) -> None:
    epochs = np.arange(1, 31)
    matrix = np.vstack([curves[seed] for seed in (42, 123, 3407)])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)

    for seed in (42, 123, 3407):
        curve = curves[seed]
        best_index = int(np.argmin(curve))
        ax.plot(
            epochs,
            curve,
            color=COLORS[seed],
            linewidth=1.35,
            alpha=0.78,
            label=f"Seed {seed}",
        )
        ax.scatter(
            epochs[best_index],
            curve[best_index],
            color=COLORS[seed],
            marker=MARKERS[seed],
            s=35,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )

    ax.fill_between(epochs, mean - std, mean + std, color="#CFCECE", alpha=0.32, linewidth=0)
    ax.plot(epochs, mean, color="#272727", linewidth=2.15, label="Three-seed mean")
    best_epochs = [int(np.argmin(curves[s])) + 1 for s in (42, 123, 3407)]
    ax.set_title(f"{model}  |  best epochs: {best_epochs[0]}, {best_epochs[1]}, {best_epochs[2]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation selection loss")
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.65)
    ax.margins(y=0.08)
    if show_legend:
        ax.legend(loc="upper right", ncol=2, fontsize=8.5, handlelength=2.3)


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.05)


def main() -> None:
    style()
    loaded = {model: load_curves(paths) for model, paths in MODEL_PATHS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    for ax, model in zip(axes.flat, ("SEAF", "FourCastNet", "ClimaX", "Swin")):
        draw_panel(ax, model, loaded[model], show_legend=(model == "SEAF"))
    fig.tight_layout(pad=1.25)
    save(fig, "backbone_validation_curves_4panel")
    plt.close(fig)

    for model, curves in loaded.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.1))
        draw_panel(ax, model, curves, show_legend=True)
        fig.tight_layout(pad=1.2)
        save(fig, f"validation_curve_{model.lower()}")
        plt.close(fig)


if __name__ == "__main__":
    main()
