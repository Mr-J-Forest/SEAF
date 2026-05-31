#!/usr/bin/env python3
"""
Generate Fig. 3: Depth-wise RMSE profiles for TSC-Fusion vs. FuXi-Ocean and AxiomOcean.
Loads actual trained model checkpoints and computes per-depth RMSE on the test set.

Usage: /c/Users/ysx/anaconda3/envs/common/bin/python.exe gen_fig_depth_rmse.py
Output: fig_depth_rmse.pdf, fig_depth_rmse.png
"""
import torch
import numpy as np
import json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from config import DEFAULT_CONFIG
from data_loader import OceanDataset
from convlstm_model import create_ocean_model
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT)

# --- Model checkpoint paths (from comparison_20epoch_fair_temp_salt.json) ---
MODEL_PATHS = {
    "TEMP": {
        "TSC-Fusion":  PROJECT / "outputs/results/74_results_20260528_193930_tsc_fusion_TEMP",
        "FuXi-Ocean":   PROJECT / "outputs/results/75_results_20260528_193950_paper_reimplementation_FuXi-Ocean",
        "AxiomOcean":   PROJECT / "outputs/results/77_results_20260528_194024_paper_reimplementation_AxiomOcean",
    },
    "SALT": {
        "TSC-Fusion":  PROJECT / "outputs/results/79_results_20260528_194112_tsc_fusion_SALT",
        "FuXi-Ocean":   PROJECT / "outputs/results/80_results_20260528_194128_paper_reimplementation_FuXi-Ocean",
        "AxiomOcean":   PROJECT / "outputs/results/82_results_20260528_194206_paper_reimplementation_AxiomOcean",
    },
}

# --- Publication styling ---
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

# --- Colors (Ocean Dusk palette) ---
AXIOM_COLOR = "#E76F51"      # coral
FUXI_COLOR = "#2A9D8F"       # teal
TSC_COLOR = "#264653"        # deep teal


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


def get_depth_levels(dataset):
    rd = dataset.all_regions_data[0]
    return np.array(rd["coords"]["levels"])


def make_dataset(model_config, var_name, project_root):
    """Create test dataset matching model config."""
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
    # Ensure target variable is included
    tgt = list(cfg.get("target_variables", []))
    if var_name not in tgt:
        tgt.append(var_name)
    cfg["target_variables"] = tgt
    # Use absolute data path
    data_path = os.path.join(str(project_root), "Data/FullData_preprocessed.nc")
    cfg["data_path"] = data_path
    return OceanDataset(data_path, cfg, mode="test")


def compute_depth_rmse(model, dataloader, dataset, device, var_name):
    """Run inference and compute RMSE per depth level for first lead time."""
    pred = predict(model, dataloader, device)

    levels = get_depth_levels(dataset)
    n_depths = len(levels)

    # Get GT by iterating test samples
    pred_len = dataset.prediction_length
    all_gt = []
    for idx in range(len(dataset)):
        start_idx, region_idx = dataset.sequences[idx]
        region = dataset.all_regions_data[region_idx]
        rd = region["data"]
        n_lats = len(region["coords"]["lats"])
        n_lons = len(region["coords"]["lons"])
        ts = start_idx + dataset.sequence_length
        te = ts + pred_len
        gt_data = rd[var_name][ts:te]
        all_gt.append(gt_data)
    gt = np.stack(all_gt)

    # Extract output channels for this variable
    target_slices = dataset.target_channel_slices
    ch_slice = target_slices.get(var_name, slice(0, n_depths))
    pred_var = pred[:, :, ch_slice, :, :]

    # Denormalize to physical units
    pred_denorm = dataset.inverse_transform(pred_var, var_name)

    # Physical-space RMSE per depth for first lead time
    rmse_t1 = np.sqrt(np.nanmean((pred_denorm[:, 0, :, :, :] - gt[:, 0, :, :, :]) ** 2, axis=(0, 2, 3)))
    valid = ~np.isnan(rmse_t1)
    if not np.all(valid):
        rmse_t1 = np.where(valid, rmse_t1, np.interp(
            np.arange(n_depths), np.where(valid)[0], rmse_t1[valid]
        ))
    return rmse_t1, levels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_profiles = {"TEMP": {}, "SALT": {}}
    depth_levels = None

    for var in ["TEMP", "SALT"]:
        print(f"\n--- {var} ---")
        for model_name, model_dir in MODEL_PATHS[var].items():
            print(f"  {model_name}...", end=" ", flush=True)
            model, cfg = load_model(str(model_dir), device)
            dataset = make_dataset(cfg, var, PROJECT)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            if depth_levels is None:
                depth_levels = get_depth_levels(dataset)
            rmse, levels = compute_depth_rmse(model, dataloader, dataset, device, var)
            all_profiles[var][model_name] = rmse
            print(f"done")

    print(f"\nGenerating figure...")
    depths = depth_levels
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=False)

    # --- Panel (a): TEMP ---
    ax = axes[0]
    ax.plot(all_profiles["TEMP"]["TSC-Fusion"], depths, color=TSC_COLOR, linewidth=1.5,
            linestyle="-", label="TSC-Fusion")
    ax.plot(all_profiles["TEMP"]["FuXi-Ocean"], depths, color=FUXI_COLOR, linewidth=1.5,
            linestyle="--", label="FuXi-Ocean")
    ax.plot(all_profiles["TEMP"]["AxiomOcean"], depths, color=AXIOM_COLOR, linewidth=1.5,
            linestyle="-.", label="AxiomOcean")

    ax.set_xlabel("RMSE (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("(a) TEMP")
    ax.invert_yaxis()
    ax.set_ylim(1050, -50)
    x_max = max(np.nanmax(all_profiles["TEMP"][m]) for m in all_profiles["TEMP"])
    ax.set_xlim(0, x_max * 1.15)
    ax.grid(True, alpha=0.2, linewidth=0.3, color="#666")
    ax.legend(loc="lower right")

    # --- Panel (b): SALT ---
    ax = axes[1]
    ax.plot(all_profiles["SALT"]["TSC-Fusion"], depths, color=TSC_COLOR, linewidth=1.5,
            linestyle="-", label="TSC-Fusion")
    ax.plot(all_profiles["SALT"]["FuXi-Ocean"], depths, color=FUXI_COLOR, linewidth=1.5,
            linestyle="--", label="FuXi-Ocean")
    ax.plot(all_profiles["SALT"]["AxiomOcean"], depths, color=AXIOM_COLOR, linewidth=1.5,
            linestyle="-.", label="AxiomOcean")

    ax.set_xlabel("RMSE (PSU)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("(b) SALT")
    ax.invert_yaxis()
    ax.set_ylim(1050, -50)
    x_max = max(np.nanmax(all_profiles["SALT"][m]) for m in all_profiles["SALT"])
    ax.set_xlim(0, x_max * 1.15)
    ax.grid(True, alpha=0.2, linewidth=0.3, color="#666")
    ax.legend(loc="lower right")

    # --- Tight layout and save ---
    plt.tight_layout(pad=1.0)
    fig.savefig(OUTPUT_DIR / "fig_depth_rmse.pdf")
    fig.savefig(OUTPUT_DIR / "fig_depth_rmse.png", dpi=300)
    print(f"\nSaved: {OUTPUT_DIR / 'fig_depth_rmse.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'fig_depth_rmse.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
