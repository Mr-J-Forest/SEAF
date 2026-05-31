#!/usr/bin/env python3
"""Generate training/validation loss curves for full vs no_tsc at 50 epochs (TEMP + SALT)."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Publication styling ---
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": True,
    "legend.edgecolor": "#DDD", "legend.fancybox": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.12, "grid.linestyle": "-",
    "lines.linewidth": 1.5, "lines.markersize": 0,
})

COLORS = {
    "full_train": "#264653", "full_val": "#2A9D8F",
    "notsc_train": "#E9C46A", "notsc_val": "#E76F51",
}

PROJECT_ROOT = Path("D:/OceanProjects/7.0")
OUTPUT_DIR = Path(__file__).resolve().parent

# --- Result directories ---
TEMP_FULL_DIR = PROJECT_ROOT / "outputs/results/101_results_20260529_112520_ablation_TSC-Fusion_(full)_TEMP"
TEMP_NOTSC_DIR = PROJECT_ROOT / "outputs/results/102_results_20260529_115057_ablation_TSC-Fusion_w-o_TSC_memory_TEMP"
SALT_FULL_DIR = PROJECT_ROOT / "outputs/results/103_results_20260529_122127_ablation_TSC-Fusion_(full)_SALT"
SALT_NOTSC_DIR = PROJECT_ROOT / "outputs/results/104_results_20260529_124448_ablation_TSC-Fusion_w-o_TSC_memory_SALT"


def load_losses(result_dir):
    ckpt_path = Path(result_dir) / "latest_checkpoint.pth"
    if not ckpt_path.exists():
        ckpt_path = Path(result_dir) / "best_model.pth"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint in {result_dir}")
        return None, None
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get("train_losses", []), ckpt.get("val_losses", [])


def main():
    configs = {}

    temp_full_train, temp_full_val = load_losses(TEMP_FULL_DIR)
    if temp_full_train:
        configs["TEMP_full"] = (temp_full_train, temp_full_val)
        print(f"TEMP full: {len(temp_full_train)} epochs, best_val={min(temp_full_val):.5f}")

    temp_notsc_train, temp_notsc_val = load_losses(TEMP_NOTSC_DIR)
    if temp_notsc_train:
        configs["TEMP_notsc"] = (temp_notsc_train, temp_notsc_val)
        print(f"TEMP notsc: {len(temp_notsc_train)} epochs, best_val={min(temp_notsc_val):.5f}")

    if SALT_FULL_DIR is not None and SALT_NOTSC_DIR is not None:
        salt_full_train, salt_full_val = load_losses(SALT_FULL_DIR)
        if salt_full_train:
            configs["SALT_full"] = (salt_full_train, salt_full_val)
            print(f"SALT full: {len(salt_full_train)} epochs, best_val={min(salt_full_val):.5f}")
        salt_notsc_train, salt_notsc_val = load_losses(SALT_NOTSC_DIR)
        if salt_notsc_train:
            configs["SALT_notsc"] = (salt_notsc_train, salt_notsc_val)
            print(f"SALT notsc: {len(salt_notsc_train)} epochs, best_val={min(salt_notsc_val):.5f}")

    n_panels = 2 if "SALT_full" in configs else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(3.25 * n_panels, 2.6))
    if n_panels == 1:
        axes = [axes]

    panels = [("TEMP", "TEMP_full", "TEMP_notsc", axes[0])]
    if n_panels == 2:
        panels.append(("SALT", "SALT_full", "SALT_notsc", axes[1]))

    for idx, (var_name, full_key, notsc_key, ax) in enumerate(panels):
        if full_key in configs:
            train, val = configs[full_key]
            epochs = np.arange(1, len(train) + 1)
            ax.plot(epochs, train, color=COLORS["full_train"], linewidth=1.5,
                    label="TSC-Fusion (train)")
            ax.plot(epochs, val, color=COLORS["full_val"], linewidth=1.5,
                    label="TSC-Fusion (val)")

        if notsc_key in configs:
            train, val = configs[notsc_key]
            epochs = np.arange(1, len(train) + 1)
            ax.plot(epochs, train, color=COLORS["notsc_train"], linewidth=1.5,
                    linestyle="--", label="w/o TSC (train)")
            ax.plot(epochs, val, color=COLORS["notsc_val"], linewidth=1.5,
                    linestyle="--", label="w/o TSC (val)")

        ax.set_title(f"({chr(97 + idx)}) {var_name}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(loc="upper right", ncol=1)

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "fig_convergence.pdf"
    png_path = OUTPUT_DIR / "fig_convergence.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
