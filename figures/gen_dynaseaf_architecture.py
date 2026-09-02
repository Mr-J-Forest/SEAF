"""Publication-grade architecture diagram generator for DynaSEAF.
Fully aligned with official code implementation (seaf_model.py, dynaseaf_model.py, data_loader.py):
1. Backbone: Parallel 3-branch additive encoder (Profile + Tendency + Forcing with learnable scale) -> Spatial Blocks
2. LCFF: Lead-conditioned global router weights G_m(h) per lead h in {1..5}
3. Gate: Pre-activation initial bias b_0 = -1.7346, yielding initial gate prior sigma(b_0) = 0.15
4. Innovation: Residual non-transport anomaly correction (unsupervised residual without overclaiming individual physics)
5. Physical Reconstruction: 2-step process (Inverse target scaling T^-1 un-standardizing to physical anomaly + Future Climatology C_{t+h})
6. Statistical Ribbon: Clear distinction between 3-seed descriptive metrics and targeted origin-paired moving-block bootstrap with BH FDR
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# High quality fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Segoe UI']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# ==================== COLOR PALETTE ====================
NAVY_BORDER   = "#1E3A8A"
NAVY_HEADER   = "#1E40AF"
NAVY_FILL     = "#F0F6FF"

BLUE_BORDER   = "#2563EB"
BLUE_HEADER   = "#1D4ED8"
BLUE_FILL     = "#F8FAFC"
BLUE_CARD     = "#EFF6FF"

PURPLE_BORDER = "#7C3AED"
PURPLE_HEADER = "#6D28D9"
PURPLE_FILL   = "#FAF5FF"
PURPLE_CARD   = "#FFFFFF"

ORANGE_BORDER = "#EA580C"
ORANGE_HEADER = "#C2410C"
ORANGE_FILL   = "#FFF7ED"
ORANGE_CARD   = "#FFFFFF"

AMBER_BORDER  = "#D97706"
AMBER_HEADER  = "#B45309"
AMBER_FILL    = "#FFFBEB"
AMBER_CARD    = "#FEF3C7"

TEAL_BORDER   = "#0D9488"
TEAL_HEADER   = "#0F766E"
TEAL_FILL     = "#F0FDFA"
TEAL_CARD     = "#FFFFFF"

ROSE_BORDER   = "#E11D48"
ROSE_HEADER   = "#BE123C"
ROSE_FILL     = "#FFF1F2"
ROSE_CARD     = "#FFFFFF"

GREEN_BORDER  = "#059669"
GREEN_HEADER  = "#047857"
GREEN_FILL    = "#ECFDF5"
GREEN_CARD    = "#FFFFFF"

GRAY_BORDER   = "#64748B"
GRAY_HEADER   = "#334155"
GRAY_FILL     = "#F8FAFC"
GRAY_CARD     = "#F1F5F9"

INK_DARK  = "#0F172A"
INK_MED   = "#334155"
INK_MUTED = "#64748B"
ARROW_INK = "#1E293B"


# ==================== CONTAINER & CARD PRIMITIVES ====================

def draw_container(ax, xy, w, h, title="", border=NAVY_BORDER, fill=NAVY_FILL,
                   header_bg=NAVY_HEADER, title_color="white", r=0.6, lw=1.4,
                   dashed=False, header_h=3.5, title_size=9.8, zorder=2):
    """Draw a main section container with an integrated top banner header and shadow."""
    x, y = xy
    ls = (0, (4, 3)) if dashed else "-"

    # Shadow
    shadow = FancyBboxPatch(
        (x + 0.25, y - 0.25), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=0, facecolor="#CBD5E1", alpha=0.35, zorder=zorder - 1
    )
    ax.add_patch(shadow)

    # Body
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=lw, edgecolor=border, facecolor=fill, linestyle=ls, zorder=zorder
    )
    ax.add_patch(patch)

    if title:
        header_patch = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            linewidth=0, facecolor=header_bg, zorder=zorder + 1
        )
        ax.add_patch(header_patch)
        rect = Rectangle(
            (x, y + h - header_h), w, header_h / 2.0,
            facecolor=header_bg, linewidth=0, zorder=zorder + 1
        )
        ax.add_patch(rect)
        ax.text(x + w / 2.0, y + h - header_h / 2.0, title,
                ha="center", va="center", color=title_color,
                fontsize=title_size, fontweight="bold", zorder=zorder + 2)


def draw_card(ax, xy, w, h, title="", lines=None, border=BLUE_BORDER,
              fill="white", header_color=BLUE_HEADER, r=0.4, lw=1.0,
              title_size=8.4, body_size=7.4, line_spacing=1.3,
              align="left", pad_top=0.55, pad_left=0.5, zorder=3):
    """Draw a clean subcard with drop shadow and typography."""
    x, y = xy
    shadow = FancyBboxPatch(
        (x + 0.15, y - 0.15), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=0, facecolor="#94A3B8", alpha=0.18, zorder=zorder - 1
    )
    ax.add_patch(shadow)

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=lw, edgecolor=border, facecolor=fill, zorder=zorder
    )
    ax.add_patch(patch)

    curr_y = y + h - pad_top
    if title:
        tx = x + w / 2.0 if align == "center" else x + pad_left
        ha = "center" if align == "center" else "left"
        ax.text(tx, curr_y, title, ha=ha, va="top", color=header_color,
                fontsize=title_size, fontweight="bold", zorder=zorder + 1)
        curr_y -= (title_size * 0.16 + 0.38)

    if lines:
        for item in lines:
            if isinstance(item, tuple):
                text_str, font_sz, color_str, is_bold = item
            else:
                text_str, font_sz, color_str, is_bold = item, body_size, INK_DARK, False

            tx = x + w / 2.0 if align == "center" else x + pad_left
            ha = "center" if align == "center" else "left"
            weight = "bold" if is_bold else "normal"
            ax.text(tx, curr_y, text_str, ha=ha, va="top", color=color_str,
                    fontsize=font_sz, fontweight=weight, zorder=zorder + 1)
            curr_y -= (font_sz * 0.16 + line_spacing * 0.28)


def draw_arrow(ax, start, end, color=ARROW_INK, lw=1.3, dashed=False,
               rad=0.0, label=None, label_pos=(0.5, 0.5), label_size=7.6,
               label_color=INK_DARK, label_box=True, zorder=5):
    """Draw sleek, publication-grade vector arrow."""
    ls = "--" if dashed else "-"
    conn = f"arc3,rad={rad}"
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=conn,
        zorder=zorder
    )
    ax.add_patch(arrow)

    if label:
        lx = start[0] + (end[0] - start[0]) * label_pos[0]
        ly = start[1] + (end[1] - start[1]) * label_pos[1]
        bbox_dict = dict(boxstyle="round,pad=0.18,rounding_size=0.15",
                         facecolor="white", edgecolor=color, linewidth=0.7, alpha=0.96) if label_box else None
        ax.text(lx, ly, label, ha="center", va="center", color=label_color,
                fontsize=label_size, fontweight="bold", bbox=bbox_dict, zorder=zorder + 1)


def draw_orthogonal_arrow(ax, points, color=ARROW_INK, lw=1.3, dashed=False,
                          label=None, label_pt=None, label_size=7.6,
                          label_color=INK_DARK, zorder=5):
    """Draw multi-segment orthogonal arrow."""
    ls = "--" if dashed else "-"
    for i in range(len(points) - 2):
        p1, p2 = points[i], points[i+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw,
                linestyle=ls, zorder=zorder)
    p_penult, p_last = points[-2], points[-1]
    arrow = FancyArrowPatch(
        p_penult, p_last,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=ls,
        zorder=zorder
    )
    ax.add_patch(arrow)

    if label and label_pt:
        bbox_dict = dict(boxstyle="round,pad=0.18,rounding_size=0.15",
                         facecolor="white", edgecolor=color, linewidth=0.7, alpha=0.96)
        ax.text(label_pt[0], label_pt[1], label, ha="center", va="center", color=label_color,
                fontsize=label_size, fontweight="bold", bbox=bbox_dict, zorder=zorder + 1)


def draw_operator_node(ax, xy, symbol=r"$\oplus$", r=1.05, fill="white",
                       border="#334155", text_color="#0F172A", font_size=9.2, zorder=6):
    """Draw circular math operator node with drop shadow."""
    x, y = xy
    shadow = Circle((x + 0.12, y - 0.12), r, facecolor="#94A3B8", alpha=0.3, linewidth=0, zorder=zorder - 1)
    ax.add_patch(shadow)
    circle = Circle((x, y), r, facecolor=fill, edgecolor=border, linewidth=1.2, zorder=zorder)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha="center", va="center", color=text_color,
            fontsize=font_size, fontweight="bold", zorder=zorder + 1)


# ==================== GRAPHICAL VISUALIZATIONS ====================

def draw_mini_ocean_heatmap(ax, xy, w, h, cmap_name="coolwarm", show_coast=True,
                            title="", eddy_sign=1.0, zorder=4, alpha=0.95):
    """Draw a miniature realistic ocean SST/anomaly heatmap with Pacific coastline."""
    x, y = xy
    nx, ny = 28, 20
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    eddy1 = eddy_sign * np.exp(-((X - 0.62)**2 + (Y - 0.55)**2) / 0.06) * 1.3
    eddy2 = -eddy_sign * np.exp(-((X - 0.32)**2 + (Y - 0.72)**2) / 0.05) * 0.95
    eddy3 = 0.4 * np.sin(X * 4.0) * np.cos(Y * 3.0)
    data = eddy1 + eddy2 + eddy3

    extent = [x, x + w, y, y + h]
    im = ax.imshow(data, extent=extent, origin="lower", cmap=cmap_name,
                   aspect="auto", alpha=alpha, zorder=zorder, interpolation="bicubic")

    rect = Rectangle((x, y), w, h, fill=False, edgecolor="#64748B", linewidth=0.75, zorder=zorder + 1)
    ax.add_patch(rect)

    if show_coast:
        coast_pts = np.array([
            [x, y + h],
            [x + w * 0.32, y + h],
            [x + w * 0.24, y + h * 0.65],
            [x + w * 0.36, y + h * 0.40],
            [x + w * 0.16, y + h * 0.20],
            [x + w * 0.19, y],
            [x, y]
        ])
        land = Polygon(coast_pts, closed=True, facecolor="#CBD5E1", edgecolor="#475569",
                       linewidth=0.75, zorder=zorder + 2)
        ax.add_patch(land)

    if title:
        ax.text(x + w / 2.0, y - 0.45, title, ha="center", va="top",
                fontsize=7.0, color=INK_DARK, fontweight="bold", zorder=zorder + 3)


def draw_isometric_map_stack(ax, base_xy, num_layers=4, layer_w=6.5, layer_h=3.2,
                             dx=0.7, dy=0.85, cmaps=None, labels=None, zorder=4):
    """Draw 3D isometric stacked slices of ocean depth levels."""
    bx, by = base_xy
    if cmaps is None:
        cmaps = ["Blues_r", "viridis", "coolwarm", "magma"]
    if labels is None:
        labels = [r"$z_{20}$ (2000m)", r"$z_{15}$", r"$z_5$", r"$z_1$ (0m)"]

    for i in range(num_layers):
        lx = bx + i * dx
        ly = by + i * dy
        shear = 0.6
        pts = np.array([
            [lx, ly],
            [lx + layer_w, ly],
            [lx + layer_w + shear, ly + layer_h],
            [lx + shear, ly + layer_h]
        ])
        cmap = plt.get_cmap(cmaps[i % len(cmaps)])
        face_color = cmap(0.55)
        poly = Polygon(pts, closed=True, facecolor=face_color, edgecolor="#1E293B",
                       linewidth=0.85, alpha=0.90, zorder=zorder + i * 2)
        ax.add_patch(poly)

        if i == num_layers - 1:
            coast = Polygon(np.array([
                [lx + shear, ly + layer_h],
                [lx + shear + layer_w * 0.35, ly + layer_h],
                [lx + shear + layer_w * 0.20, ly + layer_h * 0.50],
                [lx, ly + layer_h * 0.30],
                [lx, ly + layer_h * 0.60]
            ]), closed=True, facecolor="#E2E8F0", edgecolor="#334155",
            linewidth=0.7, zorder=zorder + i * 2 + 1)
            ax.add_patch(coast)

            ax.text(lx + layer_w + shear + 0.3, ly + layer_h * 0.5, labels[-1],
                    fontsize=6.5, color=INK_DARK, va="center", zorder=zorder + i * 2 + 2)
        elif i == 0:
            ax.text(lx + layer_w + shear + 0.3, ly + layer_h * 0.5, labels[0],
                    fontsize=6.5, color=INK_DARK, va="center", zorder=zorder + i * 2 + 2)

        if i < num_layers - 1:
            nxt_lx = bx + (i + 1) * dx
            nxt_ly = by + (i + 1) * dy
            ax.plot([lx, nxt_lx], [ly, nxt_ly], color="#64748B", linestyle=":", lw=0.75, zorder=zorder + i * 2)
            ax.plot([lx + layer_w, nxt_lx + layer_w], [ly, nxt_ly], color="#64748B", linestyle=":", lw=0.75, zorder=zorder + i * 2)


def draw_vertical_profile_icon(ax, xy, w, h, zorder=4):
    """Draw vertical thermohaline profile curves across 20 depth levels."""
    x, y = xy
    rect = Rectangle((x, y), w, h, facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=0.8, zorder=zorder)
    ax.add_patch(rect)

    for gy in np.linspace(y + 0.4, y + h - 0.4, 5):
        ax.plot([x, x + w], [gy, gy], color="#E2E8F0", lw=0.6, zorder=zorder + 1)

    zs = np.linspace(0, 1, 30)
    temp_prof = 1.0 - (1.0 / (1.0 + np.exp(-10 * (zs - 0.28))))
    tx = x + 0.3 + temp_prof * (w - 0.6)
    ty = y + h - 0.3 - zs * (h - 0.6)
    ax.plot(tx, ty, color="#DC2626", lw=1.4, zorder=zorder + 2)

    salt_prof = 0.15 + 0.75 / (1.0 + np.exp(-8 * (zs - 0.38)))
    sx = x + 0.3 + salt_prof * (w - 0.6)
    sy = y + h - 0.3 - zs * (h - 0.6)
    ax.plot(sx, sy, color="#2563EB", lw=1.4, linestyle="--", zorder=zorder + 2)

    ax.text(x + 0.3, y + h - 0.45, "0m", fontsize=5.6, color="#475569", zorder=zorder + 3)
    ax.text(x + 0.3, y + 0.35, "2000m", fontsize=5.6, color="#475569", zorder=zorder + 3)


def draw_spectral_fourier_icon(ax, xy, w, h, zorder=4):
    """Draw 2D Fourier low-frequency mode retention grid."""
    x, y = xy
    rect = Rectangle((x, y), w, h, facecolor="#FAF5FF", edgecolor="#7C3AED", linewidth=0.8, zorder=zorder)
    ax.add_patch(rect)

    nx, ny = 6, 6
    cw, ch = w / nx, h / ny
    for i in range(nx):
        for j in range(ny):
            cx = x + i * cw
            cy = y + (ny - 1 - j) * ch
            if (i < 2 and j < 2) or (i > 4 and j < 2) or (i < 2 and j > 4) or (i > 4 and j > 4):
                cell_color = "#C084FC"
                edge = "#6D28D9"
            else:
                cell_color = "#F3E8FF"
                edge = "#E9D5FF"
            c_rect = Rectangle((cx + 0.08, cy + 0.08), cw - 0.16, ch - 0.16,
                               facecolor=cell_color, edgecolor=edge, linewidth=0.5, zorder=zorder + 1)
            ax.add_patch(c_rect)

    ax.text(x + w / 2.0, y - 0.35, "rFFT2 (8x8 low modes)",
            ha="center", va="top", fontsize=6.6, color=PURPLE_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_conv_encoder_icon(ax, xy, w, h, label="Conv2D + GN", zorder=4):
    """Draw 2D spatial convolution kernel sliding on multi-channel feature tensor."""
    x, y = xy
    rect = Rectangle((x, y + 0.25), w, h - 0.25, facecolor="#F5F3FF", edgecolor=PURPLE_BORDER, linewidth=0.75, zorder=zorder)
    ax.add_patch(rect)

    nx, ny = 5, 3
    xs = np.linspace(x + 0.4, x + w - 0.4, nx)
    ys = np.linspace(y + 0.5, y + h - 0.3, ny)
    for px in xs:
        for py in ys:
            dot = Circle((px, py), 0.11, facecolor="#A78BFA", edgecolor="#6D28D9", lw=0.4, zorder=zorder + 1)
            ax.add_patch(dot)

    k_rect = Rectangle((x + 0.7, y + 0.6), w * 0.52, (h - 0.25) * 0.55,
                       facecolor="#DDD6FE", edgecolor="#7C3AED", linewidth=1.1, alpha=0.7, zorder=zorder + 2)
    ax.add_patch(k_rect)
    ax.text(x + w / 2.0, y - 0.15, label,
            ha="center", va="top", fontsize=6.2, color=PURPLE_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_quiver_streamlines(ax, xy, w, h, zorder=4):
    """Draw miniature ocean flow velocity vectors (UVEL, VVEL) and SSHA eddy."""
    x, y = xy
    rect = Rectangle((x, y + 0.3), w, h - 0.3, facecolor="#FFF7ED", edgecolor="#EA580C", linewidth=0.8, zorder=zorder)
    ax.add_patch(rect)

    nx, ny = 6, 4
    xs = np.linspace(x + 0.4, x + w - 0.4, nx)
    ys = np.linspace(y + 0.6, y + h - 0.3, ny)
    for px in xs:
        for py in ys:
            cx, cy = x + w * 0.5, y + 0.3 + (h - 0.3) * 0.5
            dx, dy = px - cx, py - cy
            u = -dy * 0.38
            v = dx * 0.38
            mag = np.sqrt(u**2 + v**2) + 1e-5
            scale = 0.36
            arrow = FancyArrowPatch((px, py), (px + u/mag * scale, py + v/mag * scale),
                                    arrowstyle="-|>", mutation_scale=4.0, color="#C2410C",
                                    linewidth=0.8, zorder=zorder + 2)
            ax.add_patch(arrow)

    core = Circle((x + w * 0.5, y + 0.3 + (h - 0.3) * 0.5), 0.32, facecolor="#FDBA74", edgecolor="#EA580C",
                  linewidth=0.7, zorder=zorder + 1)
    ax.add_patch(core)
    ax.text(x + w / 2.0, y - 0.15, r"$\widehat{D}_{t+h}$ (Flow & Height)",
            ha="center", va="top", fontsize=6.6, color=ORANGE_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_displacement_vectors_icon(ax, xy, w, h, zorder=4):
    """Draw 2D learned deformation vectors (dx, dy)."""
    x, y = xy
    rect = Rectangle((x, y + 0.3), w, h - 0.3, facecolor="#FFFBEB", edgecolor=AMBER_BORDER, linewidth=0.75, zorder=zorder)
    ax.add_patch(rect)

    nx, ny = 5, 4
    xs = np.linspace(x + 0.4, x + w - 0.4, nx)
    ys = np.linspace(y + 0.6, y + h - 0.3, ny)
    for px in xs:
        for py in ys:
            dx = 0.32 * np.sin((py - y) * 2.0)
            dy = 0.22 * np.cos((px - x) * 2.0)
            arrow = FancyArrowPatch((px, py), (px + dx, py + dy),
                                    arrowstyle="-|>", mutation_scale=3.6, color="#B45309",
                                    linewidth=0.75, zorder=zorder + 2)
            ax.add_patch(arrow)

    ax.text(x + w / 2.0, y - 0.15, r"$\Delta_{t+h} = (\Delta x, \Delta y)$",
            ha="center", va="top", fontsize=6.5, color=AMBER_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_deformed_mesh_icon(ax, xy, w, h, zorder=4):
    """Draw a deformed spatial mesh grid illustrating mask-aware bilinear warp."""
    x, y = xy
    rect = Rectangle((x, y + 0.3), w, h - 0.3, facecolor="#FFFBEB", edgecolor="#D97706", linewidth=0.8, zorder=zorder)
    ax.add_patch(rect)

    nx, ny = 6, 5
    grid_x = np.linspace(x + 0.3, x + w - 0.3, nx)
    grid_y = np.linspace(y + 0.5, y + h - 0.2, ny)

    cx, cy = x + w * 0.5, y + 0.3 + (h - 0.3) * 0.5
    X, Y = np.meshgrid(grid_x, grid_y)
    disp_x = 0.26 * np.sin((Y - cy) * 2.0)
    disp_y = 0.20 * np.cos((X - cx) * 2.0)

    X_def = X + disp_x
    Y_def = Y + disp_y

    for i in range(ny):
        ax.plot(X_def[i, :], Y_def[i, :], color="#D97706", lw=0.7, zorder=zorder + 1)
    for j in range(nx):
        ax.plot(X_def[:, j], Y_def[:, j], color="#D97706", lw=0.7, zorder=zorder + 1)

    ax.text(x + w / 2.0, y - 0.15, r"$\mathcal{W}(A_t, \Delta_{t+h})$ Warp",
            ha="center", va="top", fontsize=6.6, color=AMBER_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_innovation_flux_icon(ax, xy, w, h, zorder=4):
    """Draw residual non-transport anomaly correction heatmap."""
    x, y = xy
    nx, ny = 20, 15
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    flux = 0.8 * np.sin(X * 5.0) * np.cos(Y * 4.0) + 0.4 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
    extent = [x, x + w, y + 0.3, y + h]
    im = ax.imshow(flux, extent=extent, origin="lower", cmap="coolwarm",
                   aspect="auto", alpha=0.92, zorder=zorder)
    rect = Rectangle((x, y + 0.3), w, h - 0.3, fill=False, edgecolor=TEAL_BORDER, linewidth=0.75, zorder=zorder + 1)
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y - 0.15, r"$R_h$ Anomaly Residual",
            ha="center", va="top", fontsize=6.5, color=TEAL_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_gating_heatmap_icon(ax, xy, w, h, zorder=4):
    """Draw miniature 2D spatial gating balance map."""
    x, y = xy
    nx, ny = 20, 15
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    gate_data = 0.15 + 0.70 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.15)
    extent = [x, x + w, y + 0.3, y + h]
    im = ax.imshow(gate_data, extent=extent, origin="lower", cmap="YlGnBu",
                   aspect="auto", alpha=0.92, zorder=zorder, vmin=0, vmax=1)
    rect = Rectangle((x, y + 0.3), w, h - 0.3, fill=False, edgecolor="#0D9488", linewidth=0.75, zorder=zorder + 1)
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y - 0.15, r"$g_h \in [0, 1]$ Gate Map",
            ha="center", va="top", fontsize=6.5, color=TEAL_HEADER, fontweight="bold", zorder=zorder + 3)


def draw_lcff_weights_icon(ax, xy, w, h, zorder=4):
    """Draw stacked softmax weight bars for LCFF lead router."""
    x, y = xy
    rect = Rectangle((x, y + 0.25), w, h - 0.25, facecolor="#FFF7ED", edgecolor=ORANGE_BORDER, linewidth=0.75, zorder=zorder)
    ax.add_patch(rect)

    leads = [1, 2, 3, 4, 5]
    n_leads = len(leads)
    bw = (w - 0.6) / n_leads
    colors = ["#7C3AED", "#2563EB", "#059669", "#D97706"]

    weights = [
        [0.45, 0.25, 0.18, 0.12],
        [0.35, 0.30, 0.20, 0.15],
        [0.25, 0.35, 0.22, 0.18],
        [0.18, 0.28, 0.32, 0.22],
        [0.12, 0.22, 0.36, 0.30],
    ]

    for l_idx, ws in enumerate(weights):
        bx = x + 0.3 + l_idx * bw
        cum_h = 0.0
        for m_idx, wt in enumerate(ws):
            bar_h = wt * (h - 0.55)
            b_patch = Rectangle((bx + 0.05, y + 0.45 + cum_h), bw - 0.10, bar_h,
                                facecolor=colors[m_idx], edgecolor="white", lw=0.3, zorder=zorder + 1)
            ax.add_patch(b_patch)
            cum_h += bar_h

    ax.text(x + w / 2.0, y - 0.15, r"$G_m(h)\ \mathrm{Router\ Weights}$",
            ha="center", va="top", fontsize=6.5, color=ORANGE_HEADER, fontweight="bold", zorder=zorder + 3)


# ==================== MAIN COMPLETE GENERATOR ====================

def generate_publication_diagram():
    fig, ax = plt.subplots(figsize=(21.5, 11.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ==================== TOP TITLE BANNER ====================
    ax.text(50, 98.4, "DynaSEAF: Dynamics-Guided Transport & Innovation for Ocean Anomaly Forecasting",
            ha="center", va="center", fontsize=15.8, fontweight="bold", color=INK_DARK)
    ax.text(50, 96.0,
            "Dual-branch architecture combining inherited direct anomaly forecasting with dynamics-conditioned effective transport, residual innovation, and adaptive gating",
            ha="center", va="center", fontsize=9.2, color=INK_MED, style="italic")

    # ==================== 1. INPUT HISTORY CONTAINER (LEFT) ====================
    draw_container(ax, (1.2, 28.0), 16.5, 65.5,
                   title="12-Month Input History",
                   border=NAVY_BORDER, fill=NAVY_FILL, header_bg=NAVY_HEADER,
                   title_size=10.0, header_h=3.5)

    # 1A. 3D Isometric Map Stack for 12 Historical Months & 20 Depths
    draw_card(ax, (2.2, 69.2), 14.5, 20.0,
              title="3D Anomaly Profile Stack",
              lines=[
                  ("TEMP & SALT (20 depth levels)", 7.6, INK_DARK, False),
                  (r"Input: $X \in \mathbb{R}^{B \times 12 \times 2 \times 20 \times H \times W}$", 7.8, BLUE_HEADER, True),
                  ("Climatology-subtracted: Y_t - C_t", 7.2, INK_MUTED, False),
              ],
              border=BLUE_BORDER, fill="white", header_color=BLUE_HEADER,
              title_size=8.6, pad_top=0.55)

    draw_isometric_map_stack(ax, (3.2, 70.8), num_layers=4, layer_w=6.2, layer_h=3.2,
                             dx=0.7, dy=0.85, cmaps=["Blues", "YlGnBu", "plasma", "coolwarm"])

    # 1B. Causal Tendencies
    draw_card(ax, (2.2, 54.8), 14.5, 13.2,
              title=r"Causal Tendencies ($\Delta V_t$)",
              lines=[
                  (r"$\Delta \mathrm{TEMP}_t = \mathrm{TEMP}_t - \mathrm{TEMP}_{t-1}$", 7.4, INK_DARK, False),
                  (r"$\Delta \mathrm{SALT}_t = \mathrm{SALT}_t - \mathrm{SALT}_{t-1}$", 7.4, INK_DARK, False),
                  ("1-step backward differences", 7.2, INK_MUTED, False),
                  ("(preserves temporal inertia)", 7.2, INK_MUTED, False),
              ],
              border=BLUE_BORDER, fill="white", header_color=BLUE_HEADER,
              title_size=8.6, pad_top=0.55)

    # 1C. External Dynamics
    draw_card(ax, (2.2, 41.8), 14.5, 11.8,
              title="External Dynamics (8 vars)",
              lines=[
                  ("Ocean: UVEL, VVEL, SSHA, MLD", 7.4, INK_DARK, False),
                  ("Atmosphere: TAUX, TAUY, QNET, WFLUX", 7.2, INK_MUTED, False),
              ],
              border=BLUE_BORDER, fill="white", header_color=BLUE_HEADER,
              title_size=8.6, pad_top=0.55)

    # 1D. Latest Anomaly Source (A_t)
    draw_card(ax, (2.2, 29.5), 14.5, 11.2,
              title=r"Latest Anomaly Source ($A_t$)",
              lines=[
                  (r"$A_t = Y_t - C_t$ (20 depth levels)", 7.8, AMBER_HEADER, True),
                  ("Bypasses feature encoder to warp", 7.2, INK_DARK, False),
              ],
              border=AMBER_BORDER, fill=AMBER_CARD, header_color=AMBER_HEADER,
              title_size=8.6, pad_top=0.55)

    # Input feeder arrows into Backbone 3 Parallel Branches
    # Profile -> Profile Mixer
    draw_arrow(ax, (16.7, 78.0), (19.8, 78.0), color=NAVY_BORDER, lw=1.3,
               label=r"$X_{\mathrm{prof}}$", label_pos=(0.5, 0.5), label_size=7.2)
    # Tendency -> Context Encoder
    draw_arrow(ax, (16.7, 61.4), (19.8, 61.4), color=NAVY_BORDER, lw=1.3,
               label=r"$\Delta V$", label_pos=(0.5, 0.5), label_size=7.2)
    # Forcing -> Forcing Encoder
    draw_arrow(ax, (16.7, 47.7), (19.8, 47.7), color=NAVY_BORDER, lw=1.3,
               label=r"$X_{\mathrm{forc}}$", label_pos=(0.5, 0.5), label_size=7.2)

    # ==================== 2. SEAF-v1 BACKBONE CONTAINER (PARALLEL 3-BRANCH ENCODER) ====================
    draw_container(ax, (19.8, 28.0), 17.5, 65.5,
                   title="SEAF-v1 Backbone (d=192)",
                   border=PURPLE_BORDER, fill=PURPLE_FILL, header_bg=PURPLE_HEADER,
                   title_size=10.0, header_h=3.5)

    # 2A. Branch 1: Profile Mixer
    draw_card(ax, (20.5, 73.2), 15.0, 15.5,
              title="1. Schema-Aware Profile Mixer",
              lines=[
                  ("Grouped Depth Conv1D (Z)", 7.1, INK_DARK, False),
                  ("Grouped Time Conv1D (J)", 7.1, INK_DARK, False),
                  (r"$\to \mathbf{F}_{\mathrm{prof}} \in \mathbb{R}^{B \times 192 \times H \times W}$", 7.1, PURPLE_HEADER, True),
              ],
              border=PURPLE_BORDER, fill="white", header_color=PURPLE_HEADER,
              title_size=8.2, pad_top=0.48)
    draw_vertical_profile_icon(ax, (32.2, 74.0), 3.0, 5.8)

    # 2B. Branch 2: Context / Tendency Encoder
    draw_card(ax, (20.5, 60.2), 15.0, 11.8,
              title="2. Tendency Context Encoder",
              lines=[
                  (r"Flatten $(J \times C_{\mathrm{tend}}) \to$ Conv2D + GN", 7.1, INK_DARK, False),
                  (r"$\to \mathbf{F}_{\mathrm{tend}} \in \mathbb{R}^{B \times 192 \times H \times W}$", 7.1, PURPLE_HEADER, True),
              ],
              border=PURPLE_BORDER, fill="white", header_color=PURPLE_HEADER,
              title_size=8.2, pad_top=0.48)
    draw_conv_encoder_icon(ax, (31.4, 60.8), 3.8, 3.2, label="Conv2D + GN")

    # 2C. Branch 3: Forcing Encoder with Learnable Scale
    draw_card(ax, (20.5, 47.2), 15.0, 11.8,
              title="3. Forcing Encoder (Scaled)",
              lines=[
                  (r"Flatten $(J \times C_{\mathrm{forc}}) \to$ ConvGN + 1x1", 7.1, INK_DARK, False),
                  (r"$\to \alpha_{\mathrm{forc}} \mathbf{F}_{\mathrm{forc}}$ ($\alpha$ learnable)", 7.1, PURPLE_HEADER, True),
              ],
              border=PURPLE_BORDER, fill="white", header_color=PURPLE_HEADER,
              title_size=8.2, pad_top=0.48)
    draw_conv_encoder_icon(ax, (31.4, 47.8), 3.8, 3.2, label="Scaled Conv")

    # Clean Parallel Branch Additive Bus
    # Side taps from each card right border to vertical bus line at x=36.0
    ax.plot([35.5, 36.2], [80.5, 80.5], color=PURPLE_BORDER, lw=1.1, zorder=5)
    ax.plot([35.5, 36.2], [66.0, 66.0], color=PURPLE_BORDER, lw=1.1, zorder=5)
    ax.plot([35.5, 36.2], [53.0, 53.0], color=PURPLE_BORDER, lw=1.1, zorder=5)

    # Vertical connecting bus from 80.5 down to 43.5
    ax.plot([36.2, 36.2], [80.5, 43.5], color=PURPLE_BORDER, lw=1.2, zorder=5)

    # Arrow into summation node
    draw_arrow(ax, (36.2, 43.5), (29.2, 43.5), color=PURPLE_BORDER, lw=1.2)

    # Parallel Fusion Summation Node inside Backbone
    draw_operator_node(ax, (28.0, 43.5), symbol=r"$\oplus$", r=0.92,
                       fill="#FAF5FF", border=PURPLE_BORDER, text_color=PURPLE_HEADER)

    # 2D. 2x Local-Global Spatial Blocks
    draw_card(ax, (20.5, 29.5), 16.0, 11.8,
              title="2x Local-Global Spatial Blocks",
              lines=[
                  (r"$\mathbf{F}_0 = \mathbf{F}_{\mathrm{prof}} + \mathbf{F}_{\mathrm{tend}} + \alpha\mathbf{F}_{\mathrm{forc}}$", 7.1, INK_DARK, True),
                  ("Local: 3x3 ResNet CNN + GN", 6.8, INK_DARK, False),
                  ("Global: rFFT2 8x8 low modes", 6.8, INK_DARK, False),
                  (r"$\mathbf{F} \in \mathbb{R}^{B \times 192 \times H \times W}$ Shared Repr.", 7.2, PURPLE_HEADER, True),
              ],
              border=PURPLE_BORDER, fill="white", header_color=PURPLE_HEADER,
              title_size=8.2, pad_top=0.45)
    draw_spectral_fourier_icon(ax, (32.2, 30.2), 3.9, 3.6)

    # Arrow from Summation Node to Spatial Blocks
    draw_arrow(ax, (28.0, 42.5), (28.0, 41.3), color=PURPLE_BORDER, lw=1.2)

    # ==================== 3. DIRECT FORECAST BRANCH (TOP) ====================
    draw_container(ax, (38.8, 71.0), 23.8, 22.5,
                   title="Direct Anomaly Forecast Branch",
                   border=PURPLE_BORDER, fill=PURPLE_FILL, header_bg=PURPLE_HEADER,
                   title_size=9.6, header_h=3.4)

    # 3A. 4 Anomaly Heads with mini heatmap panels
    draw_card(ax, (39.6, 72.5), 10.4, 16.5,
              title="4 Anomaly Heads",
              lines=[
                  (r"$\widehat{A}_1, \widehat{A}_2, \widehat{A}_3, \widehat{A}_4$", 7.6, INK_DARK, True),
                  ("(4 independent", 7.2, INK_MUTED, False),
                  ("hypotheses)", 7.2, INK_MUTED, False),
              ],
              border=PURPLE_BORDER, fill="white", header_color=PURPLE_HEADER,
              title_size=8.2, pad_top=0.5)

    # 2x2 Mini Heatmap Grid for the 4 Member Heads
    draw_mini_ocean_heatmap(ax, (40.2, 73.0), 4.1, 3.2, cmap_name="coolwarm", show_coast=True, title=r"$\widehat{A}_1$", eddy_sign=1.0)
    draw_mini_ocean_heatmap(ax, (45.1, 73.0), 4.1, 3.2, cmap_name="coolwarm", show_coast=True, title=r"$\widehat{A}_2$", eddy_sign=-1.0)
    draw_mini_ocean_heatmap(ax, (40.2, 77.2), 4.1, 3.2, cmap_name="coolwarm", show_coast=True, title=r"$\widehat{A}_3$", eddy_sign=0.8)
    draw_mini_ocean_heatmap(ax, (45.1, 77.2), 4.1, 3.2, cmap_name="coolwarm", show_coast=True, title=r"$\widehat{A}_4$", eddy_sign=-0.7)

    # 3B. LCFF Lead Router (router_type: lead) with Stacked Softmax Weight Bar Chart
    draw_card(ax, (50.8, 72.5), 11.0, 16.5,
              title="Lead-Conditioned Fusion",
              lines=[
                  ("Global Lead Router Logits", 7.4, INK_DARK, False),
                  (r"$G_m(h) = \mathrm{Softmax}(\mathbf{w}_h)_m$", 7.4, ORANGE_HEADER, True),
                  (r"Direct Forecast ($h \in \{1..5\}$):", 7.2, INK_DARK, False),
                  (r"$\mathbf{\widehat{A}_{dir}(h) = \sum_m G_m(h) \widehat{A}_m}$", 7.6, INK_DARK, True),
              ],
              border=ORANGE_BORDER, fill="white", header_color=ORANGE_HEADER,
              title_size=8.2, pad_top=0.5)
    draw_lcff_weights_icon(ax, (51.5, 73.2), 9.6, 4.2)

    draw_arrow(ax, (50.0, 80.5), (50.8, 80.5), color=PURPLE_BORDER, lw=1.2)

    # Arrow from Backbone to Direct Branch
    draw_arrow(ax, (37.3, 68.0), (38.8, 81.5), color=PURPLE_BORDER, lw=1.5,
               rad=-0.10, label=r"$F$", label_pos=(0.45, 0.45), label_size=7.8)

    # ==================== 4. ADAPTIVE BRANCH FUSION & FINAL OUTPUT (TOP RIGHT) ====================
    # Fusion Container
    draw_container(ax, (64.0, 71.0), 20.0, 22.5,
                   title="Adaptive Branch Fusion",
                   border=ROSE_BORDER, fill=ROSE_FILL, header_bg=ROSE_HEADER,
                   title_size=9.6, header_h=3.4)

    draw_card(ax, (64.8, 72.5), 18.4, 16.5,
              title="",
              lines=[
                  (r"$\mathbf{\widehat{A}_{t+h} = (1 - g_h) \odot \widehat{A}_{dir} + g_h \odot \widehat{A}_{trans} + R_h}$", 7.5, ROSE_HEADER, True),
                  ("• Direct Forecast:", 7.4, PURPLE_HEADER, True),
                  ("  Inherited reliable fallback", 7.2, INK_DARK, False),
                  ("• Transport Forecast:", 7.4, AMBER_HEADER, True),
                  ("  Dynamics-advected anomaly", 7.2, INK_DARK, False),
                  ("• Residual Innovation:", 7.4, TEAL_HEADER, True),
                  ("  Non-transport anomaly correction", 7.2, INK_DARK, False),
              ],
              border=ROSE_BORDER, fill="white", header_color=ROSE_HEADER,
              title_size=8.0, pad_top=0.45)

    # Math operator nodes for fusion
    draw_operator_node(ax, (66.5, 74.0), symbol=r"$\odot$", r=0.85, fill="#FDF2F8", border=ROSE_BORDER, text_color=ROSE_HEADER)
    draw_operator_node(ax, (73.0, 74.0), symbol=r"$\oplus$", r=0.85, fill="#FDF2F8", border=ROSE_BORDER, text_color=ROSE_HEADER)

    # Arrow from Direct Branch to Fusion
    draw_arrow(ax, (61.8, 81.5), (64.0, 81.5), color=PURPLE_BORDER, lw=1.5,
               label=r"$\widehat{A}_{\mathrm{dir}}$", label_pos=(0.5, 0.5), label_size=7.8)

    # Final Physical Reconstruction Box (2-Step: Inverse Scaling + Climatology Addition)
    draw_container(ax, (85.2, 71.0), 13.6, 22.5,
                   title="Physical Forecast",
                   border=GREEN_BORDER, fill=GREEN_FILL, header_bg=GREEN_HEADER,
                   title_size=9.4, header_h=3.4)

    draw_card(ax, (86.0, 72.5), 12.0, 16.5,
              title="2-Step Reconstruction",
              lines=[
                  (r"1. $\mathbf{\widehat{A}_{t+h}^{\mathrm{phys}} = \mathcal{T}^{-1}(\widehat{A}_{t+h})}$", 7.4, GREEN_HEADER, True),
                  ("   Inverse Target Scaling", 7.0, INK_DARK, False),
                  (r"2. $\mathbf{\widehat{Y}_{t+h} = C_{t+h} + \widehat{A}_{t+h}^{\mathrm{phys}}}$", 7.4, GREEN_HEADER, True),
                  ("   Climatology Restoration", 7.0, INK_DARK, False),
                  ("Physical Outputs (h=1..5):", 7.0, INK_DARK, True),
                  ("• TEMP (°C) & SALT (PSU)", 7.0, INK_MUTED, False),
              ],
              border=GREEN_BORDER, fill="white", header_color=GREEN_HEADER,
              title_size=8.0, pad_top=0.45)

    # 3D Stacked Forecast Maps on the Right
    draw_isometric_map_stack(ax, (91.0, 73.0), num_layers=3, layer_w=4.2, layer_h=2.2,
                             dx=0.45, dy=0.65, cmaps=["coolwarm", "YlGnBu", "viridis"],
                             labels=[r"$h=5$", r"$h=3$", r"$h=1$"])

    # Arrow from Fusion to Physical Reconstruction
    draw_arrow(ax, (83.2, 81.5), (85.2, 81.5), color=GREEN_BORDER, lw=1.6,
               label=r"$\mathcal{T}^{-1}, +C$", label_pos=(0.5, 0.5), label_size=7.6, label_color=GREEN_HEADER)

    # ==================== 5. DYNAMICS-GUIDED TRANSPORT & INNOVATION (BOTTOM) ====================
    draw_container(ax, (38.8, 28.0), 60.0, 41.0,
                   title="Dynamics-Guided Transport & Innovation Branch (DynaSEAF)",
                   border=ORANGE_BORDER, fill=ORANGE_FILL, header_bg=ORANGE_HEADER,
                   title_size=10.2, header_h=3.5)

    # 5A. Left: Lead Embedding & Future Dynamics Head with Quiver Flow Icon
    draw_card(ax, (39.8, 59.2), 16.5, 5.0,
              title="",
              lines=[(r"$\mathbf{Lead\ Embedding:}\ e_h \in \mathbb{R}^{d_e}\ (h=1..5)$", 7.6, ORANGE_HEADER, True)],
              border=ORANGE_BORDER, fill=ORANGE_CARD, header_color=ORANGE_HEADER,
              pad_top=0.45, pad_left=0.5)

    draw_card(ax, (39.8, 29.5), 16.5, 28.5,
              title="Future Dynamics Head",
              lines=[
                  (r"$\widehat{D}_{t+h} = \mathcal{H}_{\mathrm{dyn}}(F, e_h)$", 7.8, ORANGE_HEADER, True),
                  ("Predicted Physical Dynamics:", 7.4, INK_DARK, True),
                  ("• Ocean Flow: UVEL, VVEL", 7.2, INK_DARK, False),
                  ("• Sea Level & Layer: SSHA, MLD", 7.2, INK_DARK, False),
                  ("Auxiliary Supervision (Train-only):", 7.4, ORANGE_HEADER, True),
                  (r"$\mathcal{L}_{\mathrm{dyn}} = \frac{1}{4} \sum_v \mathcal{L}_v\ (\lambda=0.10)$", 7.4, INK_DARK, False),
              ],
              border=ORANGE_BORDER, fill="white", header_color=ORANGE_HEADER,
              title_size=8.4, pad_top=0.5)
    draw_quiver_streamlines(ax, (49.6, 38.0), 6.0, 4.8)

    # Arrow from Backbone to Future Dynamics Head
    draw_arrow(ax, (37.3, 42.5), (39.8, 42.5), color=ORANGE_BORDER, lw=1.5,
               label=r"$F$", label_pos=(0.5, 0.5), label_size=7.8)
    draw_arrow(ax, (48.0, 59.2), (48.0, 58.0), color=ORANGE_BORDER, lw=1.0)

    # 5B. Middle: Deformation Head (with Displacement Icon) & Mask-Aware Transport Warp (with Deformed Mesh Icon)
    draw_card(ax, (58.3, 47.8), 19.5, 16.5,
              title="Learned Deformation Head",
              lines=[
                  (r"$\Delta_{t+h} = \mathcal{D}_\psi(F, \widehat{D}_{t+h}, e_h)$", 7.8, AMBER_HEADER, True),
                  ("Shared 2D shifts: (dx, dy)", 7.4, INK_DARK, False),
                  (r"$\Delta = \tanh(\cdot) \times \Delta_{\max}$ (bounded)", 7.2, INK_MUTED, False),
              ],
              border=AMBER_BORDER, fill="white", header_color=AMBER_HEADER,
              title_size=8.4, pad_top=0.5)
    draw_displacement_vectors_icon(ax, (71.0, 49.6), 6.0, 4.8)

    draw_card(ax, (58.3, 29.5), 19.5, 17.2,
              title=r"Mask-Aware Transport Warp ($\mathcal{W}$)",
              lines=[
                  (r"$\mathbf{\widehat{A}_{trans}(h) = \mathcal{W}(A_t, \Delta_{t+h})}$", 7.8, AMBER_HEADER, True),
                  ("• Bilinear grid_sample interpolation", 7.2, INK_DARK, False),
                  ("• Valid-value mask renormalization", 7.2, INK_DARK, False),
                  ("• Land mask NaN protection", 7.2, INK_DARK, False),
              ],
              border=AMBER_BORDER, fill=AMBER_CARD, header_color=AMBER_HEADER,
              title_size=8.4, pad_top=0.5)
    draw_deformed_mesh_icon(ax, (71.0, 31.2), 6.0, 4.8)

    # Internal Dynamics feeding arrows:
    # 1. To Deformation Head
    draw_arrow(ax, (56.3, 54.0), (58.3, 54.0), color=ORANGE_BORDER, lw=1.3,
               label=r"$\widehat{D}$", label_pos=(0.4, 0.4), label_size=7.4)
    # 2. Deformation to Warp
    draw_arrow(ax, (68.0, 47.8), (68.0, 46.7), color=AMBER_BORDER, lw=1.3,
               label=r"$\Delta_{t+h}$", label_pos=(0.5, 0.5), label_size=7.4)

    # Long bypass arrow from Latest Anomaly Source (A_t) to Warp Block
    draw_orthogonal_arrow(
        ax,
        points=[(16.7, 35.0), (58.3, 35.0)],
        color=AMBER_BORDER, lw=1.6,
        label=r"$\mathrm{Source\ Anomaly}\ A_t$", label_pt=(37.5, 35.0),
        label_size=8.0, label_color=AMBER_HEADER
    )

    # 5C. Right: Innovation Head & Adaptive Gate Head (with b_0 = -1.7346, prior ~ 0.15)
    draw_card(ax, (79.8, 47.8), 18.0, 16.5,
              title=r"Innovation Head ($R_h$)",
              lines=[
                  (r"$R_h = \mathcal{R}(F, \widehat{D}_{t+h}, e_h)$", 7.8, TEAL_HEADER, True),
                  ("Learned residual anomaly correction:", 7.2, INK_DARK, True),
                  ("• Non-transport dynamics / local terms", 7.0, INK_DARK, False),
                  ("• Zero-init output convolution", 7.0, INK_MUTED, False),
              ],
              border=TEAL_BORDER, fill="white", header_color=TEAL_HEADER,
              title_size=8.4, pad_top=0.5)
    draw_innovation_flux_icon(ax, (91.5, 49.6), 5.5, 4.8)

    draw_card(ax, (79.8, 29.5), 18.0, 17.2,
              title=r"Adaptive Gate ($g_h$)",
              lines=[
                  (r"$g_h = \sigma(\mathcal{G}(F, \widehat{D}_{t+h}, e_h))$", 7.8, TEAL_HEADER, True),
                  ("Target-resolved spatial gate:", 7.2, INK_DARK, True),
                  (r"• Pre-act bias $b_0 \approx -1.7346$", 7.0, INK_DARK, False),
                  (r"• Gate prior $\sigma(b_0) \approx 0.15$ (direct prior)", 7.0, INK_DARK, False),
                  (r"• Dynamic lead/depth balance map", 7.0, INK_DARK, False),
              ],
              border=TEAL_BORDER, fill="white", header_color=TEAL_HEADER,
              title_size=8.4, pad_top=0.5)
    draw_gating_heatmap_icon(ax, (91.5, 31.2), 5.5, 4.8)

    # Horizontal bus for dynamics and features
    draw_orthogonal_arrow(
        ax,
        points=[(56.3, 47.0), (78.3, 47.0), (79.8, 52.0)],
        color=TEAL_BORDER, lw=1.2,
        label=r"$F, \widehat{D}$", label_pt=(78.0, 47.0), label_size=7.4
    )
    draw_orthogonal_arrow(
        ax,
        points=[(78.3, 47.0), (78.3, 38.0), (79.8, 38.0)],
        color=TEAL_BORDER, lw=1.2
    )

    # Outputs rising upward into Adaptive Branch Fusion
    # 1. Transport Anomaly (A_trans) rising into Fusion
    draw_orthogonal_arrow(
        ax,
        points=[(72.8, 46.7), (72.8, 71.0)],
        color=AMBER_BORDER, lw=1.5,
        label=r"$\widehat{A}_{\mathrm{trans}}$", label_pt=(72.8, 68.5),
        label_size=7.8, label_color=AMBER_HEADER
    )
    # 2. Residual Innovation (R_h) and Gate (g_h) rising into Fusion
    draw_orthogonal_arrow(
        ax,
        points=[(83.8, 64.3), (83.8, 68.5), (80.8, 71.0)],
        color=TEAL_BORDER, lw=1.3,
        label=r"$R_h, g_h$", label_pt=(83.8, 67.2),
        label_size=7.6, label_color=TEAL_HEADER
    )

    # Training-only Auxiliary indicator badge
    ax.text(50.0, 25.5, r"$\mathbf{Training\text{-}Only\ Auxiliary\ Supervision:}$ $\mathcal{L}_{\mathrm{dyn}}$ backpropagates through $\mathcal{H}_{\mathrm{dyn}}$ and backbone $\mathbf{F}$ (labels absent during validation/test)",
            ha="center", va="center", color=ORANGE_HEADER, fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=ORANGE_CARD, edgecolor=ORANGE_BORDER, lw=0.85))

    # ==================== 6. BOTTOM STRIP: EXTERNAL EVALUATION PROTOCOL ====================
    ax.plot([1.2, 98.8], [23.0, 23.0], color=GRAY_BORDER, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(50.0, 21.6, "External Persistence-Aware Evaluation Protocol (Independent Baselines - Not Model Inputs)",
            ha="center", va="center", color=GRAY_HEADER, fontsize=8.8, fontweight="bold")

    # Card 1: AP Reference
    draw_card(ax, (1.5, 2.5), 22.5, 17.2,
              title="Anomaly Persistence (AP)",
              lines=[
                  (r"$\mathbf{\widehat{Y}_{t+h}^{\mathrm{AP}} = C_{t+h} + (Y_t - C_t)}$", 7.8, GRAY_HEADER, True),
                  ("• Carries initial anomaly forward without decay", 7.4, INK_DARK, False),
                  ("• Rigorous benchmark for long-memory ocean physics", 7.4, INK_DARK, False),
                  ("• External reference generated only by evaluator", 7.2, INK_MUTED, False),
              ],
              border=GRAY_BORDER, fill=GRAY_CARD, header_color=GRAY_HEADER,
              title_size=8.4, pad_top=0.5)

    # Card 2: DAP Reference
    draw_card(ax, (26.0, 2.5), 22.5, 17.2,
              title="Damped Anomaly Persistence (DAP)",
              lines=[
                  (r"$\mathbf{\widehat{Y}_{t+h}^{\mathrm{DAP}} = C_{t+h} + \rho_h (Y_t - C_t)}$", 7.8, GRAY_HEADER, True),
                  (r"• $\rho_h$: empirical lag-$h$ autocorrelation per depth/var", 7.4, INK_DARK, False),
                  ("• Prevents inflated skill at longer leads (h=4, 5)", 7.4, INK_DARK, False),
                  ("• Computed strictly from training period data", 7.2, INK_MUTED, False),
              ],
              border=GRAY_BORDER, fill=GRAY_CARD, header_color=GRAY_HEADER,
              title_size=8.4, pad_top=0.5)

    # Card 3: Skill Scores
    draw_card(ax, (50.5, 2.5), 22.5, 17.2,
              title=r"Skill Scores ($\mathrm{SS_{AP}}$ & $\mathrm{SS_{DAP}}$)",
              lines=[
                  (r"$\mathbf{\mathrm{SS}_{\mathrm{AP}} = 1 - \frac{\mathrm{MSE}_{\mathrm{model}}}{\mathrm{MSE}_{\mathrm{AP}}}}$", 7.8, GRAY_HEADER, True),
                  (r"$\mathbf{\mathrm{SS}_{\mathrm{DAP}} = 1 - \frac{\mathrm{MSE}_{\mathrm{model}}}{\mathrm{MSE}_{\mathrm{DAP}}}}$", 7.8, GRAY_HEADER, True),
                  ("• Evaluated only on finite/valid ocean points", 7.4, INK_DARK, False),
                  ("• Macro-averaged across all forecast origins", 7.4, INK_DARK, False),
              ],
              border=GRAY_BORDER, fill=GRAY_CARD, header_color=GRAY_HEADER,
              title_size=8.4, pad_top=0.5)

    # Card 4: Statistical Testing Protocol
    draw_card(ax, (75.0, 2.5), 23.5, 17.2,
              title="Statistical Protocol & Contrasts",
              lines=[
                  ("• Descriptive: 3-Seed mean ± sample SD", 7.4, INK_DARK, True),
                  ("• Targeted Origin-Paired Contrasts:", 7.4, INK_DARK, True),
                  ("  10,000 Moving-Block Bootstrap replicates", 7.2, INK_DARK, False),
                  ("  Block length = 5 months (preserving auto-corr)", 7.0, INK_MUTED, False),
                  ("• Multiple Testing: Benjamini-Hochberg FDR q-val", 7.0, INK_MUTED, False),
              ],
              border=GRAY_BORDER, fill=GRAY_CARD, header_color=GRAY_HEADER,
              title_size=8.4, pad_top=0.5)

    # Connecting arrows in evaluation strip
    draw_arrow(ax, (24.0, 11.1), (26.0, 11.1), color=GRAY_BORDER, lw=1.1, dashed=True)
    draw_arrow(ax, (48.5, 11.1), (50.5, 11.1), color=GRAY_BORDER, lw=1.1, dashed=True)
    draw_arrow(ax, (73.0, 11.1), (75.0, 11.1), color=GRAY_BORDER, lw=1.1, dashed=True)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    pdf_path = OUT / "dynaseaf_architecture.pdf"
    png_path = OUT / "dynaseaf_architecture.png"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Successfully generated:\n  - {pdf_path}\n  - {png_path}")

if __name__ == "__main__":
    generate_publication_diagram()
