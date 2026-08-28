#!/usr/bin/env python3
"""Legacy figure draft; not an admissible source for current paper evidence.

The script hard-codes obsolete result IDs and external-model labels and uses a
superseded data/metric protocol.  It is intentionally gated so it cannot be
mistaken for the provenance-aware figure pipeline.
"""
import torch, numpy as np, json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_CONFIG
from data_loader import OceanDataset
from model_factory import create_ocean_model

OUTPUT_DIR = Path(__file__).resolve().parent
RESULT_BASE = "D:/OceanProjects/7.0/outputs/results"

MODEL_PAIRS = [
    ("SEAF",       "101", "103"),
    ("TianHai",    "73",  "78"),
    ("FuXi-Ocean", "75",  "80"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "legend.fontsize": 7, "legend.frameon": True,
    "legend.edgecolor": "#DDD", "legend.fancybox": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS = {
    "SEAF": "#264653",
    "TianHai": "#E76F51",
    "FuXi-Ocean": "#2A9D8F",
}


def find_model_dir(model_index: str) -> str:
    dirs = [d for d in os.listdir(RESULT_BASE) if d.startswith(f"{model_index}_results_")]
    if not dirs:
        raise FileNotFoundError(f"No result dir for index {model_index}")
    return os.path.join(RESULT_BASE, sorted(dirs)[-1])


def load_model(model_index, device):
    model_dir = find_model_dir(model_index)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    config["model_type"] = config.get("model_type", "seaf")
    model = create_ocean_model(config).to(device)
    ckpt = torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, config


def predict(model, loader, device):
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, (list, tuple)):
                x = [t.to(device) for t in x]
            else:
                x = x.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def denormalize(pred, scaler):
    N, T, C, H, W = pred.shape
    flat = pred.transpose(0, 1, 3, 4, 2).reshape(-1, C)
    denorm = scaler.inverse_transform(flat).reshape(N, T, H, W, C)
    return denorm.transpose(0, 1, 4, 2, 3)


def rmse_depth_all(gt, pred, lead_idx):
    """gt: (N, T, Level, Lat, Lon), pred: (N, T, C, H, W) with C = n_levels * pred_len"""
    rmses = []
    for d in range(gt.shape[2]):
        ch = d  # for lead=0, channel = depth_idx
        g = gt[:, lead_idx, d, :, :]
        p = pred[:, lead_idx, ch, :, :]
        m = ~np.isnan(g) & ~np.isnan(p)
        if m.any():
            rmses.append(float(np.sqrt(np.mean((g[m] - p[m]) ** 2))))
        else:
            rmses.append(np.nan)
    return np.array(rmses)


def main():
    if os.environ.get('TSC_ALLOW_LEGACY_FIGURES') != '1':
        raise RuntimeError(
            '这是旧版占位绘图脚本，不能用于论文证据。若只为检查历史图形，'
            '请显式设置 TSC_ALLOW_LEGACY_FIGURES=1。'
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load reference dataset ----
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "return_additional_info": False, "enable_arima_xgboost": False,
        "enable_positional_encoding": True, "positional_encoding_frequencies": 8,
        "enable_time_encoding": True,
        "time_encoding_frequencies": 4, "include_year_trend": True,
        "target_variables": ["TEMP", "SALT", "PTEMP", "PDEN", "SPICE"],
    })
    print("Loading dataset...")
    dataset = OceanDataset(cfg["data_path"], cfg, mode="test")

    levels = dataset.all_regions_data[0]["coords"]["levels"]
    lats = dataset.all_regions_data[0]["coords"]["lats"]
    lons = dataset.all_regions_data[0]["coords"]["lons"]
    seq_len = cfg["sequence_length"]
    pred_len = cfg["prediction_length"]

    # Ground truth
    print("Extracting ground truth...")
    gt_temp, gt_salt = [], []
    for idx in range(len(dataset)):
        start_idx, region_idx = dataset.sequences[idx]
        rd = dataset.all_regions_data[region_idx]["data"]
        ts = start_idx + seq_len
        te = ts + pred_len
        gt_temp.append(rd.get("TEMP", np.full((pred_len, len(levels), len(lats), len(lons)), np.nan))[ts:te])
        gt_salt.append(rd.get("SALT", np.full((pred_len, len(levels), len(lats), len(lons)), np.nan))[ts:te])
    gt_temp = np.stack(gt_temp)
    gt_salt = np.stack(gt_salt)
    print(f"GT shapes — TEMP: {gt_temp.shape}, SALT: {gt_salt.shape}")

    # ---- Run inference for each model pair ----
    all_preds = {}
    for name, tidx, sidx in MODEL_PAIRS:
        print(f"\n=== {name} ===")
        # TEMP
        print("  Loading TEMP model...")
        t_model, t_cfg = load_model(tidx, device)
        ds = OceanDataset(t_cfg["data_path"], t_cfg, mode="test")
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        scaler_t = ds.scalers.get("TEMP")
        print("  Running TEMP inference...")
        pt = predict(t_model, dl, device)
        pt = denormalize(pt, scaler_t)  # (N, T, C, H, W), C = n_levels * pred_len

        # SALT
        print("  Loading SALT model...")
        s_model, s_cfg = load_model(sidx, device)
        ds2 = OceanDataset(s_cfg["data_path"], s_cfg, mode="test")
        dl2 = DataLoader(ds2, batch_size=8, shuffle=False, num_workers=0)
        scaler_s = ds2.scalers.get("SALT")
        print("  Running SALT inference...")
        ps = predict(s_model, dl2, device)
        ps = denormalize(ps, scaler_s)

        # Validate: single-variable TEMP models have C = n_levels
        n_levels = len(levels)
        assert pt.shape[2] == n_levels, f"TEMP channels {pt.shape[2]} != n_levels {n_levels}"
        assert ps.shape[2] == n_levels, f"SALT channels {ps.shape[2]} != n_levels {n_levels}"
        all_preds[name] = {"temp": pt, "salt": ps}

    lead_idx = 0

    # =============================================
    # FIGURE 1: Prediction maps (Truth / Prediction / Error + colorbar)
    # =============================================
    print("\n=== Figure 1: Prediction maps ===")
    sel_depths = [0, 5, 10]  # surface, 50m, 200m
    depth_labels = [f"{levels[d]:.0f} m" for d in sel_depths]

    fig1 = plt.figure(figsize=(8.5, 4.5))
    gs = GridSpec(3, 6, figure=fig1, wspace=0.25, hspace=0.35,
                  left=0.05, right=0.82, bottom=0.09, top=0.93)

    seaf = all_preds["SEAF"]
    temp_im, temp_errim = None, None
    salt_im, salt_errim = None, None

    for vi, (vname, gt, unit) in enumerate([("TEMP", gt_temp, "°C"), ("SALT", gt_salt, "PSU")]):
        pred = seaf[vname.lower()]

        # Global color scale across all selected depths
        gt_all = np.concatenate([gt[:, lead_idx, d, :, :].ravel() for d in sel_depths])
        valid = gt_all[~np.isnan(gt_all)]
        vmin_g, vmax_g = float(np.nanpercentile(valid, 2)), float(np.nanpercentile(valid, 98))

        err_all = np.concatenate(
            [(pred[:, lead_idx, d, :, :] - gt[:, lead_idx, d, :, :]).ravel() for d in sel_depths])
        valid_e = err_all[~np.isnan(err_all)]
        emax_g = max(abs(float(np.nanpercentile(valid_e, 1))),
                     abs(float(np.nanpercentile(valid_e, 99))))

        for di, d in enumerate(sel_depths):
            row = di
            co = vi * 3

            gt_slice = np.nanmean(gt[:, lead_idx, d, :, :], axis=0)
            pr_slice = np.nanmean(pred[:, lead_idx, d, :, :], axis=0)
            err = pr_slice - gt_slice

            # Truth
            ax = fig1.add_subplot(gs[row, co])
            im_gt = ax.imshow(gt_slice, aspect="auto", cmap="RdYlBu_r",
                              vmin=vmin_g, vmax=vmax_g,
                              extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                              origin="lower")
            if vi == 0:
                ax.set_ylabel(f"{depth_labels[di]}\nLatitude")
            if di == 0:
                ax.set_title(f"{vname} Truth")
            ax.tick_params(labelsize=6)

            # Prediction
            ax = fig1.add_subplot(gs[row, co + 1])
            ax.imshow(pr_slice, aspect="auto", cmap="RdYlBu_r",
                      vmin=vmin_g, vmax=vmax_g,
                      extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                      origin="lower")
            if di == 0:
                ax.set_title(f"{vname} Prediction")
            ax.tick_params(labelsize=6)

            # Error
            ax = fig1.add_subplot(gs[row, co + 2])
            im_err = ax.imshow(err, aspect="auto", cmap="RdBu_r",
                               vmin=-emax_g, vmax=emax_g,
                               extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                               origin="lower")
            if di == 0:
                ax.set_title(f"{vname} Error")
            if vi == 1 and di == 2:
                ax.set_xlabel("Longitude")
            ax.tick_params(labelsize=6)

        # Save handles for colorbars
        if vi == 0:
            temp_im, temp_errim = im_gt, im_err
        else:
            salt_im, salt_errim = im_gt, im_err

    # Colorbars: 2 columns (TEMP / SALT) x 2 rows (main / error)
    cb_x0, cb_x1 = 0.84, 0.91
    cb_w = 0.008
    cb_y1, cb_y2, cb_h = 0.59, 0.09, 0.28

    for vi, (label, im, err_im) in enumerate([
            ("TEMP (°C)", temp_im, temp_errim),
            ("SALT (PSU)", salt_im, salt_errim)]):
        x = cb_x0 + vi * (cb_x1 - cb_x0)
        cax = fig1.add_axes([x, cb_y1, cb_w, cb_h])
        fig1.colorbar(im, cax=cax)
        cax.set_ylabel(label, fontsize=7)
        cax.tick_params(labelsize=6)

        cax = fig1.add_axes([x, cb_y2, cb_w, cb_h])
        fig1.colorbar(err_im, cax=cax)
        cax.set_ylabel(f"{label.split(' ')[0]} Error", fontsize=7)
        cax.tick_params(labelsize=6)

    fig1.savefig(OUTPUT_DIR / "fig_prediction_maps.pdf")
    fig1.savefig(OUTPUT_DIR / "fig_prediction_maps.png", dpi=300)
    plt.close(fig1)
    print("  Saved fig_prediction_maps.pdf")

    # =============================================
    # FIGURE 2: Depth RMSE (2-panel)
    # =============================================
    print("\n=== Figure 2: Depth RMSE ===")
    fig2 = plt.figure(figsize=(5.0, 3.2))
    gs2 = GridSpec(1, 2, figure=fig2, wspace=0.35,
                   left=0.09, right=0.97, bottom=0.17, top=0.90)

    # Panel a: TEMP depth RMSE
    ax_t = fig2.add_subplot(gs2[0, 0])
    for name in ["TianHai", "FuXi-Ocean", "SEAF"]:
        pred = all_preds[name]["temp"]
        rmse = rmse_depth_all(gt_temp, pred, lead_idx)
        ax_t.plot(rmse, levels, color=COLORS.get(name, "#333"),
                  label=name, linewidth=1.2, marker=".", markersize=3)
    ax_t.set_ylim(levels[0], levels[-1])
    ax_t.invert_yaxis()
    ax_t.set_xlabel("RMSE (°C)")
    ax_t.set_ylabel("Depth (m)")
    ax_t.set_title("(a) TEMP", fontweight="bold")
    ax_t.grid(True, alpha=0.12)

    # Panel b: SALT depth RMSE
    ax_s = fig2.add_subplot(gs2[0, 1])
    for name in ["TianHai", "FuXi-Ocean", "SEAF"]:
        pred = all_preds[name]["salt"]
        rmse = rmse_depth_all(gt_salt, pred, lead_idx)
        ax_s.plot(rmse, levels, color=COLORS.get(name, "#333"),
                  label=name, linewidth=1.2, marker=".", markersize=3)
    ax_s.set_ylim(levels[0], levels[-1])
    ax_s.invert_yaxis()
    ax_s.set_xlabel("RMSE (PSU)")
    ax_s.set_title("(b) SALT", fontweight="bold")
    ax_s.grid(True, alpha=0.12)
    ax_s.legend(loc="lower right", fontsize=6)

    fig2.savefig(OUTPUT_DIR / "fig_depth_rmse.pdf")
    fig2.savefig(OUTPUT_DIR / "fig_depth_rmse.png", dpi=300)
    plt.close(fig2)
    print("  Saved fig_depth_rmse.pdf")

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
