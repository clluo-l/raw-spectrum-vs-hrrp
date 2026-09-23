#!/usr/bin/env python3
"""Create the offline--online paradigm diagram used as manuscript Fig. 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


INK = "#27313D"
MUTED = "#5D6874"
PANEL_BG = "#FAFBFC"
PANEL_EDGE = "#CBD3DC"

DATA_FILL = "#E6F0F8"
DATA_EDGE = "#4C78A8"
TRANSFORM_FILL = "#FFF1D6"
TRANSFORM_EDGE = "#C88719"
MODEL_FILL = "#ECE7F5"
MODEL_EDGE = "#7563A6"
OUTPUT_FILL = "#E2F2EC"
OUTPUT_EDGE = "#2A7F6E"
CONTROL_FILL = "#F2F4F6"
CONTROL_EDGE = "#66727E"


def rounded_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.0,
    linestyle: str = "-",
    linewidth: float = 0.85,
    zorder: int = 3,
):
    """Draw one restrained rounded box and its centered label."""

    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.009,rounding_size=0.010",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.10,
        zorder=zorder + 1,
    )
    return patch


def panel(ax, xy: tuple[float, float], wh: tuple[float, float], title: str, subtitle: str):
    """Draw a panel with a compact title region and no decorative clutter."""

    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=PANEL_BG,
        edgecolor=PANEL_EDGE,
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(patch)
    title_y = xy[1] + wh[1] - 0.042
    ax.text(
        xy[0] + 0.016,
        title_y,
        title,
        ha="left",
        va="center",
        fontsize=9.3,
        fontweight="semibold",
        color=INK,
        zorder=2,
    )
    ax.text(
        xy[0] + 0.016,
        title_y - 0.045,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.1,
        color=MUTED,
        zorder=2,
    )
    ax.plot(
        [xy[0] + 0.015, xy[0] + wh[0] - 0.015],
        [xy[1] + wh[1] - 0.105, xy[1] + wh[1] - 0.105],
        color=PANEL_EDGE,
        linewidth=0.7,
        zorder=1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 0.95,
    linestyle: str = "-",
    rad: float = 0.0,
    zorder: int = 4,
):
    """Draw a clean arrow whose endpoints are already on box boundaries."""

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.0,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def poly_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    linewidth: float = 0.95,
    linestyle: str = "-",
    zorder: int = 4,
):
    """Draw a routed arrow through explicit waypoints without crossing boxes."""

    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    path = MplPath(points, codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=8.0,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.16, 2.55))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel(
        ax,
        (0.015, 0.055),
        (0.350, 0.90),
        "(a) Offline model development",
        "Simulation, development-only selection, and protocol lock",
    )
    panel(
        ax,
        (0.380, 0.055),
        (0.605, 0.90),
        "(b) Bank-free single-sweep inference",
        "Same coherent sweep; matched interfaces; independently trained weights",
    )

    # Offline lane: one causal direction, no crossed or backward arrows.
    offline_x, offline_w, offline_h = 0.055, 0.305, 0.105
    offline_y = [0.675, 0.510, 0.345, 0.180]
    rounded_box(
        ax,
        (offline_x, offline_y[0]),
        (offline_w, offline_h),
        "20 PEC meshes\n+ 5 x 12 nonpolar grid",
        facecolor=OUTPUT_FILL,
        edgecolor=OUTPUT_EDGE,
    )
    rounded_box(
        ax,
        (offline_x, offline_y[1]),
        (offline_w, offline_h),
        "CST 2025 complex sweeps\n91 bins, 0.10-1.00 GHz",
        facecolor=DATA_FILL,
        edgecolor=DATA_EDGE,
    )
    rounded_box(
        ax,
        (offline_x, offline_y[2]),
        (offline_w, offline_h),
        "Paired inputs + labels\n" r"$x_f,\ x_r,\ \mathbf{p}=(\theta,\phi)$",
        facecolor=DATA_FILL,
        edgecolor=DATA_EDGE,
    )
    rounded_box(
        ax,
        (offline_x, offline_y[3]),
        (offline_w, offline_h),
        "Development-only selection\nfreeze preprocessing + models",
        facecolor=TRANSFORM_FILL,
        edgecolor=TRANSFORM_EDGE,
    )
    for top, bottom in zip(offline_y[:-1], offline_y[1:]):
        arrow(
            ax,
            (offline_x + offline_w / 2, top),
            (offline_x + offline_w / 2, bottom + offline_h),
            color=CONTROL_EDGE,
        )
    ax.text(
        offline_x + offline_w / 2,
        0.115,
        "Frozen before untouched ID / held-mesh tests",
        ha="center",
        va="center",
        fontsize=6.9,
        color=MUTED,
    )

    # Online lane: a single fork followed by two parallel, equally weighted paths.
    query_xy, query_wh = (0.405, 0.505), (0.135, 0.185)
    rounded_box(
        ax,
        query_xy,
        query_wh,
        "One coherent\nquery sweep\n" r"$E_\theta(f_k),\ E_\phi(f_k)$",
        facecolor=DATA_FILL,
        edgecolor=DATA_EDGE,
        fontsize=7.7,
    )
    ax.text(
        query_xy[0] + query_wh[0] / 2,
        query_xy[1] - 0.022,
        "simulated here",
        ha="center",
        va="top",
        fontsize=6.6,
        color=MUTED,
    )

    view_x, view_w, view_h = 0.585, 0.160, 0.115
    freq_y, hrrp_y = 0.660, 0.445
    rounded_box(
        ax,
        (view_x, freq_y),
        (view_w, view_h),
        "Frequency input $x_f$\ncomplex-4: Re/Im",
        facecolor=DATA_FILL,
        edgecolor=DATA_EDGE,
        fontsize=7.4,
    )
    rounded_box(
        ax,
        (view_x, hrrp_y),
        (view_w, view_h),
        "Rect/Hann " r"$\rightarrow$" " IFFT\ncomplex-4 HRRP $x_r$",
        facecolor=TRANSFORM_FILL,
        edgecolor=TRANSFORM_EDGE,
        fontsize=7.4,
    )

    encoder_x, encoder_w, encoder_h = 0.795, 0.170, 0.115
    rounded_box(
        ax,
        (encoder_x, freq_y),
        (encoder_w, encoder_h),
        "Frequency encoder $h_f$\n(separate weights)",
        facecolor=MODEL_FILL,
        edgecolor=MODEL_EDGE,
        fontsize=7.4,
    )
    rounded_box(
        ax,
        (encoder_x, hrrp_y),
        (encoder_w, encoder_h),
        "HRRP encoder $h_r$\n(separate weights)",
        facecolor=MODEL_FILL,
        edgecolor=MODEL_EDGE,
        fontsize=7.4,
    )

    fork_x, fork_y = 0.560, query_xy[1] + query_wh[1] / 2
    ax.plot(
        [query_xy[0] + query_wh[0], fork_x],
        [fork_y, fork_y],
        color=CONTROL_EDGE,
        linewidth=0.95,
        zorder=4,
    )
    ax.scatter([fork_x], [fork_y], s=8, color=CONTROL_EDGE, zorder=5)
    arrow(ax, (fork_x, fork_y), (view_x, freq_y + view_h / 2), color=DATA_EDGE, rad=-0.08)
    arrow(ax, (fork_x, fork_y), (view_x, hrrp_y + view_h / 2), color=TRANSFORM_EDGE, rad=0.08)
    arrow(
        ax,
        (view_x + view_w, freq_y + view_h / 2),
        (encoder_x, freq_y + encoder_h / 2),
        color=DATA_EDGE,
    )
    arrow(
        ax,
        (view_x + view_w, hrrp_y + view_h / 2),
        (encoder_x, hrrp_y + encoder_h / 2),
        color=TRANSFORM_EDGE,
    )

    head_xy, head_wh = (0.745, 0.265), (0.220, 0.110)
    rounded_box(
        ax,
        head_xy,
        head_wh,
        "Factorized heads\n5 " r"$\theta$" " + 12 " r"$\phi$" " logits",
        facecolor=MODEL_FILL,
        edgecolor=MODEL_EDGE,
        fontsize=7.6,
    )
    poly_arrow(
        ax,
        [
            (encoder_x + encoder_w, freq_y + encoder_h / 2),
            (0.975, freq_y + encoder_h / 2),
            (0.975, head_xy[1] + head_wh[1] + 0.020),
            (head_xy[0] + head_wh[0] - 0.035, head_xy[1] + head_wh[1]),
        ],
        color=MODEL_EDGE,
    )
    arrow(
        ax,
        (encoder_x + encoder_w / 2, hrrp_y),
        (head_xy[0] + 0.090, head_xy[1] + head_wh[1]),
        color=MODEL_EDGE,
        rad=0.03,
    )
    ax.text(0.971, 0.600, "$z_f$", fontsize=7.2, color=MODEL_EDGE, ha="right")
    ax.text(0.865, 0.405, "$z_r$", fontsize=7.2, color=MODEL_EDGE, ha="center")

    output_xy, output_wh = (0.745, 0.085), (0.220, 0.125)
    rounded_box(
        ax,
        output_xy,
        output_wh,
        "Grid decode " r"$\hat{\mathbf{p}}=(\hat\theta,\hat\phi)$" "\nbody-frame LOS on $S^2$\n2-DoF, not full $SO(3)$ | GCE",
        facecolor=OUTPUT_FILL,
        edgecolor=OUTPUT_EDGE,
        fontsize=6.9,
    )
    arrow(
        ax,
        (head_xy[0] + head_wh[0] / 2, head_xy[1]),
        (output_xy[0] + output_wh[0] / 2, output_xy[1] + output_wh[1]),
        color=OUTPUT_EDGE,
    )
    # A callout replaces the old disconnected response-bank pseudo-node.
    ax.plot([0.405, 0.405], [0.145, 0.205], color=OUTPUT_EDGE, linewidth=2.0, solid_capstyle="round")
    ax.text(
        0.418,
        0.175,
        "No online target-specific\nresponse bank",
        ha="left",
        va="center",
        fontsize=7.2,
        color=INK,
        linespacing=1.08,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": None, "pad_inches": 0.0}
    if args.out.suffix.lower() == ".png":
        save_kwargs["dpi"] = 300
    fig.savefig(args.out, **save_kwargs)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
