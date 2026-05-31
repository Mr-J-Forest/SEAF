#!/usr/bin/env python3
"""
Generate publication-quality prediction figure for TSC-Fusion.
Academic-plotting skill: Workflow 2 (matplotlib), Modern Minimal style.

Layout: double-column figure, 2 rows x 3 cols
  Row 1: TEMP @ 50 m  —  GT | TSC-Fusion | Absolute Error
  Row 2: SALT @ 50 m  —  GT | TSC-Fusion | Absolute Error

Usage: /c/Users/ysx/anaconda3/envs/common/python.exe gen_fig_prediction.py
"""
import torch
import numpy as np
import json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config import DEFAULT_CONFIG
from data_loader import OceanDataset
from convlstm_model import create_ocean_model
from torch.utils.data import DataLoader

# --- Paths ---
PROJECT = Path(__file__).resolve().parent.parent.parent.parent
TEMP_MODEL = str(PROJECT / "outputs/results/101_results_20260529_112520_ablation_TSC-Fusion_(full)_TEMP")
SALT_MODEL = str(PROJECT / "outputs/results/103_results_20260529_122127_ablation_TSC-Fusion_(full)_SALT")
OUTPUT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT)

DEPTH_IDX = 5       # 50 m
SAMPLE_IDX = 0      # first test sample
LEAD_IDX = 0        # T+1

# --- Academic-plotting skill: Publication styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# --- Ocean Dusk palette (academic-plotting skill) ---
DEEP_TEAL = "#264653"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
SANDY = "#F4A261"
CORAL = "#E76F51"

# --- ICM-style colormaps (built into matplotlib) ---
TEMP_CMAP = "RdYlBu_r"   # warm (red) → cool (blue) for temperature
SALT_CMAP = "viridis"     # perceptually uniform for salinity
ERR_CMAP = "YlOrRd"       # error magnitude


def load_model(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["model_type"] = config.get("model_type", "tsc_fusion")
    model = create_ocean_model(config).to(device)
    ckpt_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "latest_checkpoint.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, config


def predict(model, dataloader, device):
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def denormalize(pred, scaler):
    N, T, C, H, W = pred.shape
    flat = pred.transpose(0, 1, 3, 4, 2).reshape(-1, C)
    denorm = scaler.inverse_transform(flat).reshape(N, T, H, W, C)
    return denorm.transpose(0, 1, 4, 2, 3)


def make_dataset(model_config):
    cfg = DEFAULT_CONFIG.copy()
    for key in ["input_variables", "target_variables", "sequence_length",
                "prediction_length", "enable_positional_encoding",
                "positional_encoding_frequencies", "depth_encoding_frequencies",
                "enable_time_encoding", "time_encoding_frequencies",
                "include_year_trend", "return_additional_info",
                "enable_arima_xgboost", "use_thermohaline_memory"]:
        if key in model_config:
            cfg[key] = model_config[key]
    cfg["return_additional_info"] = False
    cfg["enable_arima_xgboost"] = False
    tgt = list(cfg.get("target_variables", []))
    for v in ["TEMP", "SALT"]:
        if v not in tgt:
            tgt.append(v)
    cfg["target_variables"] = tgt
    return OceanDataset(cfg["data_path"], cfg, mode="test")


def get_ground_truth(dataset):
    pred_len = dataset.prediction_length
    levels = dataset.all_regions_data[0]["coords"]["levels"]
    lats = dataset.all_regions_data[0]["coords"]["lats"]
    lons = dataset.all_regions_data[0]["coords"]["lons"]
    gt_temp, gt_salt = [], []
    for idx in range(len(dataset)):
        start_idx, region_idx = dataset.sequences[idx]
        rd = dataset.all_regions_data[region_idx]["data"]
        ts = start_idx + dataset.sequence_length
        te = ts + pred_len
        gt_temp.append(rd.get("TEMP", np.full((pred_len, len(levels), len(lats), len(lons)), np.nan))[ts:te])
        gt_salt.append(rd.get("SALT", np.full((pred_len, len(levels), len(lats), len(lons)), np.nan))[ts:te])
    return np.stack(gt_temp), np.stack(gt_salt), levels, lats, lons


def draw_map(ax, data, cmap, vmin, vmax, title, cbar_label, lons, lats,
             draw_grid=True):
    """Draw a single pcolormesh panel with clean styling."""
    X, Y = np.meshgrid(lons, lats)
    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="auto", linewidth=0, rasterized=True)

    # Subtle grid lines
    if draw_grid:
        ax.set_xticks(np.arange(130, 165, 5))
        ax.set_yticks(np.arange(5, 30, 5))
        ax.grid(True, alpha=0.2, linewidth=0.3, color="#666")

    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=7.5, labelpad=1)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.outline.set_linewidth(0.4)

    return im


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    temp_model, temp_cfg = load_model(TEMP_MODEL, device)
    salt_model, salt_cfg = load_model(SALT_MODEL, device)

    # TEMP inference
    print("TEMP inference...")
    temp_dataset = make_dataset(temp_cfg)
    temp_loader = DataLoader(temp_dataset, batch_size=8, shuffle=False, num_workers=0)
    gt_temp, gt_salt, levels, lats, lons = get_ground_truth(temp_dataset)
    pred_temp = denormalize(predict(temp_model, temp_loader, device),
                            temp_dataset.scalers["TEMP"])

    # SALT inference
    print("SALT inference...")
    salt_dataset = make_dataset(salt_cfg)
    salt_loader = DataLoader(salt_dataset, batch_size=8, shuffle=False, num_workers=0)
    pred_salt = denormalize(predict(salt_model, salt_loader, device),
                            salt_dataset.scalers["SALT"])

    # Extract 2D slices at 50 m
    gt_t = gt_temp[SAMPLE_IDX, LEAD_IDX, DEPTH_IDX]
    pr_t = pred_temp[SAMPLE_IDX, LEAD_IDX, DEPTH_IDX]
    gt_s = gt_salt[SAMPLE_IDX, LEAD_IDX, DEPTH_IDX]
    pr_s = pred_salt[SAMPLE_IDX, LEAD_IDX, DEPTH_IDX]

    err_t = np.abs(gt_t - pr_t)
    err_s = np.abs(gt_s - pr_s)

    # =============================================================
    # Build the figure — 2 rows x 3 columns
    # =============================================================
    fig = plt.figure(figsize=(7.0, 4.2))
    gs = GridSpec(2, 3, figure=fig,
                  left=0.06, right=0.93, bottom=0.09, top=0.92,
                  hspace=0.35, wspace=0.40)

    vmin_t = min(gt_t.min(), pr_t.min())
    vmax_t = max(gt_t.max(), pr_t.max())
    vmin_s = min(gt_s.min(), pr_s.min())
    vmax_s = max(gt_s.max(), pr_s.max())
    vmax_err_t = max(err_t.max(), 0.01)
    vmax_err_s = max(err_s.max(), 0.01)

    column_titles = ["Ground Truth", "TSC-Fusion", "Absolute Error"]
    row_labels = [("Temperature (°C)", "°C"), ("Salinity (PSU)", "PSU")]

    panels_data = [
        # (data_tuple, vmin, vmax, cmap, cbar_unit)
        {"data": [gt_t, pr_t, err_t], "vmin": [vmin_t, vmin_t, 0],
         "vmax": [vmax_t, vmax_t, vmax_err_t],
         "cmap": [TEMP_CMAP, TEMP_CMAP, ERR_CMAP],
         "unit": "°C"},
        {"data": [gt_s, pr_s, err_s], "vmin": [vmin_s, vmin_s, 0],
         "vmax": [vmax_s, vmax_s, vmax_err_s],
         "cmap": [SALT_CMAP, SALT_CMAP, ERR_CMAP],
         "unit": "PSU"},
    ]

    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            d = panels_data[row]
            X, Y = np.meshgrid(lons, lats)
            im = ax.pcolormesh(X, Y, d["data"][col],
                                cmap=d["cmap"][col],
                                vmin=d["vmin"][col],
                                vmax=d["vmax"][col],
                                shading="auto", linewidth=0, rasterized=True)

            # Grid
            ax.set_xticks(np.arange(130, 165, 5))
            ax.set_yticks(np.arange(5, 30, 5))
            ax.grid(True, alpha=0.15, linewidth=0.3, color="#666")

            # Labels
            if row == 1:
                ax.set_xlabel("Longitude (°E)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel("Latitude (°N)", fontsize=8)
            else:
                ax.tick_params(labelleft=False)

            # Column titles only on top row
            if row == 0:
                ax.set_title(column_titles[col], fontsize=9.5,
                             fontweight="bold", pad=4)

            # Panel label
            idx = row * 3 + col
            ax.text(0.03, 0.97, f"({panel_labels[idx]})",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="6%", pad=0.12)
            cbar = plt.colorbar(im, cax=cax)
            if col < 2:
                cbar.set_label(row_labels[row][1], fontsize=7, labelpad=1)
            else:
                cbar.set_label(f"|Error| ({row_labels[row][1]})",
                               fontsize=7, labelpad=1)
            cbar.ax.tick_params(labelsize=6, length=2)
            cbar.outline.set_linewidth(0.4)

    # Row labels
    fig.text(0.005, 0.70, "Temperature at 50 m",
             rotation=90, fontsize=9, fontweight="bold", va="center")
    fig.text(0.005, 0.30, "Salinity at 50 m",
             rotation=90, fontsize=9, fontweight="bold", va="center")

    fig.savefig(OUTPUT_DIR / "fig_prediction.pdf")
    fig.savefig(OUTPUT_DIR / "fig_prediction.png", dpi=300)
    print(f"Saved: {OUTPUT_DIR / 'fig_prediction.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'fig_prediction.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
