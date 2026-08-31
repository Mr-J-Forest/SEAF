from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:/OceanProjects/TSC-Fusion")
INPUT = ROOT / (
    "outputs/seaf_h192_confirmation_remote/"
    "f41a17ca1120c04c7c2c7881aff74835c37e0e5e_seaf_h192_confirmation/"
    "outputs/prediction_evidence/seaf_h192_final_test/sample_0/SEAF/"
    "regional_evidence.npz"
)
OUTPUT = ROOT / "figures/generated"
LEADS = (0, 2, 4)


def plot_variable(data, variable, ch_slice, depth_idx):
    pred = data["prediction"][:, ch_slice][:, depth_idx]
    target = data["target"][:, ch_slice][:, depth_idx]
    ap = data["ap_physical"][:, ch_slice][:, depth_idx]
    dap = data["dap_physical"][:, ch_slice][:, depth_idx]
    lons = data["lons"]
    lats = data["lats"]
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    error_pred = pred - target
    error_ap = ap - target
    fields = [
        ("ORAS5", target),
        ("SEAF", pred),
        ("AP", ap),
        ("DAP", dap),
    ]
    errors = [
        ("SEAF error", error_pred),
        ("AP error", error_ap),
    ]

    fig, axes = plt.subplots(6, 3, figsize=(11.0, 16.0), constrained_layout=True)
    physical_min = min(float(np.nanmin(x)) for _, x in fields)
    physical_max = max(float(np.nanmax(x)) for _, x in fields)
    error_max = max(float(np.nanmax(np.abs(x))) for _, x in errors)
    unit = r"TEMP ($^\circ$C)" if variable == "TEMP" else "SALT (PSU)"
    depth_m = int(round(float(data["levels"][depth_idx])))

    for row, (label, values) in enumerate(fields):
        vmin, vmax = physical_min, physical_max
        for col, lead in enumerate(LEADS):
            ax = axes[row, col]
            image = ax.imshow(
                values[lead], origin="lower", extent=extent, aspect="auto",
                vmin=vmin, vmax=vmax, cmap="RdYlBu_r",
            )
            ax.set_title(f"{label} lead {lead + 1}", fontsize=10)
            if col == 0:
                ax.set_ylabel("Latitude")
            if row == len(fields) - 1:
                ax.set_xlabel("Longitude")
            fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)

    for row, (label, values) in enumerate(errors, start=len(fields)):
        vmin, vmax = -error_max, error_max
        for col, lead in enumerate(LEADS):
            ax = axes[row, col]
            image = ax.imshow(
                values[lead], origin="lower", extent=extent, aspect="auto",
                vmin=vmin, vmax=vmax, cmap="RdBu_r",
            )
            ax.set_title(f"{label} lead {lead + 1}", fontsize=10)
            if col == 0:
                ax.set_ylabel("Latitude")
            if row == len(fields) + len(errors) - 1:
                ax.set_xlabel("Longitude")
            fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)

    fig.suptitle(f"{unit} surface forecast ({depth_m} m)", fontsize=16)
    path = OUTPUT / f"fig_final_test_{variable.lower()}_surface.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    data = np.load(INPUT)
    plot_variable(data, "TEMP", slice(0, 20), 0)
    plot_variable(data, "SALT", slice(20, 40), 0)


if __name__ == "__main__":
    main()
