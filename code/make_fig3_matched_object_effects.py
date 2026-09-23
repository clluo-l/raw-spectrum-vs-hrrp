#!/usr/bin/env python3
"""Create manuscript Fig. 3 from strict object-centered held-mesh results.

The figure is deliberately object-centered: each marker is one held object's
mean after averaging all 60 fixed directions and seven training seeds.  A
Hann-IFFT audit summary can be supplied later to add a third paired marker
without changing the figure contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


FOLD_COLORS = {
    1: "#0072B2",
    2: "#D55E00",
    3: "#009E73",
    4: "#AA4499",
}
REPRESENTATION_STYLE = {
    "frequency": {"label": "Frequency", "marker": "o"},
    "rect_ifft": {"label": "Rect-IFFT", "marker": "s"},
    "hann_ifft": {"label": "Hann-IFFT", "marker": "^"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def held_model(summary: dict[str, Any], representation: str) -> dict[str, Any]:
    try:
        model = summary["models"][representation]["held_geometry"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"summary lacks models.{representation}.held_geometry") from exc
    if not isinstance(model, dict):
        raise TypeError(f"models.{representation}.held_geometry is not an object")
    return model


def per_object_map(model: dict[str, Any], representation: str) -> dict[str, dict[str, Any]]:
    rows = model.get("per_object")
    if not isinstance(rows, list):
        raise TypeError(f"{representation} held model lacks per_object list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        object_code = str(row["object_code"])
        if object_code in result:
            raise ValueError(f"duplicate object in {representation}: {object_code}")
        result[object_code] = {
            "object_code": object_code,
            "fold": int(row["fold"]),
            "gce_deg": float(row["great_circle_mae_deg"]),
            "top1": float(row["top1_acc"]),
            "seed_count": int(row["seed_count"]),
            "directions_per_seed": [int(value) for value in row["directions_per_seed"]],
        }
    return result


def validate_strict_inputs(
    summary: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    if summary.get("status") != "ok":
        raise RuntimeError("strict summary is not status=ok")
    if audit.get("status") != "pass":
        raise RuntimeError("strict audit is not status=pass")
    if int(audit.get("expected_task_count", -1)) != 70:
        raise RuntimeError("strict audit does not cover 70 formal tasks")
    if int(audit.get("observed_task_count", -1)) != 70:
        raise RuntimeError("strict audit did not observe all 70 formal tasks")
    if audit.get("errors"):
        raise RuntimeError("strict audit contains errors")

    frequency_model = held_model(summary, "frequency")
    rect_model = held_model(summary, "rect_ifft")
    object_maps = {
        "frequency": per_object_map(frequency_model, "frequency"),
        "rect_ifft": per_object_map(rect_model, "rect_ifft"),
    }
    object_codes = set(object_maps["frequency"])
    if object_codes != set(object_maps["rect_ifft"]) or len(object_codes) != 20:
        raise RuntimeError("strict held arms do not contain the same 20 objects")
    fold_counts: dict[int, int] = {}
    for object_code in object_codes:
        left = object_maps["frequency"][object_code]
        right = object_maps["rect_ifft"][object_code]
        if left["fold"] != right["fold"]:
            raise RuntimeError(f"fold mismatch for {object_code}")
        if left["seed_count"] != 7 or right["seed_count"] != 7:
            raise RuntimeError(f"{object_code} does not contain seven seeds in both strict arms")
        if left["directions_per_seed"] != [60] or right["directions_per_seed"] != [60]:
            raise RuntimeError(f"{object_code} does not contain all 60 held directions")
        fold_counts[left["fold"]] = fold_counts.get(left["fold"], 0) + 1
    if fold_counts != {1: 5, 2: 5, 3: 5, 4: 5}:
        raise RuntimeError(f"unexpected held-fold composition: {fold_counts}")

    paired = summary.get("paired_difference", {}).get("held_geometry")
    if not isinstance(paired, dict):
        raise RuntimeError("strict summary lacks held paired_difference")
    paired_rows = paired.get("per_object")
    if not isinstance(paired_rows, list) or len(paired_rows) != 20:
        raise RuntimeError("strict summary lacks 20 held object effects")
    paired_map = {str(row["object_code"]): row for row in paired_rows}
    if set(paired_map) != object_codes:
        raise RuntimeError("paired object effects do not match strict model objects")
    for object_code in object_codes:
        expected_delta = (
            object_maps["frequency"][object_code]["gce_deg"]
            - object_maps["rect_ifft"][object_code]["gce_deg"]
        )
        observed_delta = float(paired_map[object_code]["mean_gce_difference_deg"])
        if not np.isclose(expected_delta, observed_delta, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"paired delta is inconsistent for {object_code}")
    overall_delta = float(paired["object_centered_gce_difference_deg"])
    object_mean = float(
        np.mean([float(row["mean_gce_difference_deg"]) for row in paired_rows])
    )
    if not np.isclose(overall_delta, object_mean, rtol=0.0, atol=1e-12):
        raise RuntimeError("overall held delta is not the mean of 20 centered object effects")
    interval = paired.get("object_centered_gce_difference_95ci")
    if not isinstance(interval, dict) or not {"low", "high"}.issubset(interval):
        raise RuntimeError("strict summary lacks the object-bootstrap held CI")
    strict_info = {
        "overall": {
            "frequency_gce_deg": float(frequency_model["object_centered_gce_deg"]),
            "rect_ifft_gce_deg": float(rect_model["object_centered_gce_deg"]),
            "frequency_minus_rect_gce_deg": overall_delta,
            "frequency_minus_rect_95ci_deg": [float(interval["low"]), float(interval["high"])],
            "bootstrap_iterations": int(interval["iterations"]),
            "bootstrap_seed": int(interval["bootstrap_seed"]),
        },
        "paired_map": paired_map,
    }
    return strict_info, object_maps


def add_hann(
    strict_maps: dict[str, dict[str, dict[str, Any]]],
    hann_summary: dict[str, Any],
    hann_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    if hann_summary.get("status") != "ok":
        raise RuntimeError("Hann summary is not status=ok")
    if hann_audit is not None and hann_audit.get("status") != "pass":
        raise RuntimeError("Hann audit is not status=pass")
    hann_rect = per_object_map(held_model(hann_summary, "rect_ifft"), "hann-summary rect_ifft")
    hann_map = per_object_map(held_model(hann_summary, "hann_ifft"), "hann_ifft")
    expected_objects = set(strict_maps["rect_ifft"])
    if set(hann_rect) != expected_objects or set(hann_map) != expected_objects:
        raise RuntimeError("Hann summary does not contain the strict 20 held objects")
    for object_code in expected_objects:
        strict_rect = strict_maps["rect_ifft"][object_code]
        extension_rect = hann_rect[object_code]
        if strict_rect["fold"] != extension_rect["fold"]:
            raise RuntimeError(f"Hann summary rect fold mismatch for {object_code}")
        if not np.isclose(
            strict_rect["gce_deg"], extension_rect["gce_deg"], rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"Hann summary does not reuse the strict rect arm for {object_code}")
        if hann_map[object_code]["fold"] != strict_rect["fold"]:
            raise RuntimeError(f"Hann fold mismatch for {object_code}")
        if hann_map[object_code]["seed_count"] != 7 or hann_map[object_code]["directions_per_seed"] != [60]:
            raise RuntimeError(f"Hann result is incomplete for {object_code}")
    strict_maps["hann_ifft"] = hann_map
    hann_held = held_model(hann_summary, "hann_ifft")
    return {
        "hann_ifft_gce_deg": float(hann_held["object_centered_gce_deg"]),
        "audit_status": hann_audit.get("status") if hann_audit is not None else "not supplied",
    }


def lighten(color: str, amount: float = 0.62) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float64)
    return tuple((rgb + (1.0 - rgb) * amount).tolist())


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.8,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_figure(
    object_maps: dict[str, dict[str, dict[str, Any]]],
    strict_info: dict[str, Any],
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> dict[str, Any]:
    configure_style()
    representations = ["frequency", "rect_ifft"]
    if "hann_ifft" in object_maps:
        representations.append("hann_ifft")
    object_codes = sorted(
        object_maps["frequency"],
        key=lambda code: (object_maps["frequency"][code]["fold"], code),
    )
    positions = np.arange(len(object_codes), dtype=np.float64)
    if len(representations) == 2:
        offsets = {"frequency": -0.12, "rect_ifft": 0.12}
    else:
        offsets = {"frequency": -0.18, "rect_ifft": 0.0, "hann_ifft": 0.18}

    figure = plt.figure(figsize=(7.16, 5.15))
    grid = figure.add_gridspec(2, 1, height_ratios=(2.35, 1.0), hspace=0.16)
    top = figure.add_subplot(grid[0])
    bottom = figure.add_subplot(grid[1], sharex=top)

    all_values = [
        object_maps[representation][object_code]["gce_deg"]
        for representation in representations
        for object_code in object_codes
    ]
    ymax = max(100.0, math.ceil(max(all_values) / 10.0) * 10.0 + 10.0)
    for fold in range(1, 5):
        start = (fold - 1) * 5 - 0.5
        stop = fold * 5 - 0.5
        for axis in (top, bottom):
            axis.axvspan(start, stop, color=lighten(FOLD_COLORS[fold], 0.90), zorder=-5)
        if fold > 1:
            top.axvline(start, color="#B8B8B8", linewidth=0.55, zorder=-1)
            bottom.axvline(start, color="#B8B8B8", linewidth=0.55, zorder=-1)
        top.text(
            (start + stop) / 2.0,
            ymax - 2.2,
            f"Fold {fold}",
            color=FOLD_COLORS[fold],
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="top",
        )

    for x, object_code in zip(positions, object_codes, strict=True):
        fold = object_maps["frequency"][object_code]["fold"]
        color = FOLD_COLORS[fold]
        xs = [x + offsets[representation] for representation in representations]
        ys = [object_maps[representation][object_code]["gce_deg"] for representation in representations]
        top.plot(xs, ys, color=color, alpha=0.52, linewidth=0.85, zorder=2)
        for representation, point_x, point_y in zip(representations, xs, ys, strict=True):
            marker = REPRESENTATION_STYLE[representation]["marker"]
            if representation == "frequency":
                facecolor = color
                alpha = 0.95
            elif representation == "rect_ifft":
                facecolor = "white"
                alpha = 1.0
            else:
                facecolor = lighten(color, 0.55)
                alpha = 1.0
            top.scatter(
                point_x,
                point_y,
                marker=marker,
                s=25,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=0.9,
                alpha=alpha,
                zorder=4,
            )

    top.set_ylim(0.0, ymax)
    top.set_ylabel("Object-centered held GCE (°)")
    top.grid(axis="y", color="#D6D6D6", linewidth=0.45, alpha=0.75)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    top.text(
        -0.055,
        1.015,
        "(a)",
        transform=top.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    top.text(
        -0.015,
        1.015,
        "Paired held-object performance",
        transform=top.transAxes,
        fontsize=8.5,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )

    overall = strict_info["overall"]
    representation_handles = []
    for representation in representations:
        if representation == "frequency":
            mean_value = overall["frequency_gce_deg"]
            facecolor = "#4D4D4D"
        elif representation == "rect_ifft":
            mean_value = overall["rect_ifft_gce_deg"]
            facecolor = "white"
        else:
            mean_value = float(
                np.mean([object_maps["hann_ifft"][code]["gce_deg"] for code in object_codes])
            )
            facecolor = "#BDBDBD"
        representation_handles.append(
            Line2D(
                [0],
                [0],
                marker=REPRESENTATION_STYLE[representation]["marker"],
                linestyle="none",
                markerfacecolor=facecolor,
                markeredgecolor="#4D4D4D",
                markeredgewidth=0.85,
                markersize=5.2,
                label=f"{REPRESENTATION_STYLE[representation]['label']} ({mean_value:.2f}°)",
            )
        )
    representation_legend = top.legend(
        handles=representation_handles,
        title="Representation (20-object mean)",
        title_fontsize=6.9,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.965),
        frameon=True,
        framealpha=0.94,
        edgecolor="#D0D0D0",
        ncol=len(representations),
        columnspacing=1.1,
        handletextpad=0.35,
        borderpad=0.45,
    )
    top.add_artist(representation_legend)

    deltas = np.asarray(
        [float(strict_info["paired_map"][code]["mean_gce_difference_deg"]) for code in object_codes],
        dtype=np.float64,
    )
    ci_low, ci_high = overall["frequency_minus_rect_95ci_deg"]
    delta_mean = overall["frequency_minus_rect_gce_deg"]
    bottom.axhspan(
        ci_low,
        ci_high,
        color="#7A7A7A",
        alpha=0.13,
        zorder=-2,
        label="Conditional 95% object interval",
    )
    bottom.axhline(0.0, color="#303030", linewidth=0.75, zorder=1)
    bottom.axhline(delta_mean, color="#555555", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    for x, object_code, delta in zip(positions, object_codes, deltas, strict=True):
        fold = object_maps["frequency"][object_code]["fold"]
        color = FOLD_COLORS[fold]
        bottom.vlines(x, 0.0, delta, color=color, alpha=0.58, linewidth=0.8, zorder=2)
        bottom.scatter(
            x,
            delta,
            s=22,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    delta_padding = max(2.0, 0.12 * float(np.ptp(np.r_[deltas, ci_low, ci_high, 0.0])))
    bottom.set_ylim(
        math.floor((min(float(deltas.min()), ci_low, 0.0) - delta_padding) / 5.0) * 5.0,
        math.ceil((max(float(deltas.max()), ci_high, 0.0) + delta_padding) / 5.0) * 5.0,
    )
    bottom.set_ylabel("F − Rect (°)")
    bottom.grid(axis="y", color="#D6D6D6", linewidth=0.45, alpha=0.75)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(object_codes, rotation=55, ha="right", rotation_mode="anchor")
    bottom.set_xlim(-0.55, len(object_codes) - 0.45)
    bottom.text(
        -0.055,
        1.015,
        "(b)",
        transform=bottom.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    bottom.text(
        -0.015,
        1.015,
        "Paired object differences; negative favors Frequency",
        transform=bottom.transAxes,
        fontsize=8.5,
        fontweight="semibold",
        ha="left",
        va="bottom",
    )
    bottom.text(
        0.995,
        0.94,
        (
            f"Overall Δ = {delta_mean:+.2f}°\n"
            f"95% object interval [{ci_low:+.2f}, {ci_high:+.2f}]°"
        ),
        transform=bottom.transAxes,
        fontsize=7.2,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.94},
    )

    for axis in (top, bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.085, right=0.992, top=0.948, bottom=0.185)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Object-centered strict matched-transform results",
        "Author": "Generated from audited formal predictions",
        "Subject": "Frequency versus length-91 rectangular-IFFT held-object comparison",
        "Creator": Path(__file__).name,
    }
    figure.savefig(output_pdf, metadata=metadata)
    figure.savefig(output_png, dpi=dpi)
    plt.close(figure)
    return {
        "width_inches": 7.16,
        "height_inches": 5.15,
        "dpi": dpi,
        "representations": representations,
        "object_order": object_codes,
        "fold_colors": FOLD_COLORS,
        "delta_definition": "Frequency minus rectangular-IFFT; negative favors Frequency",
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    strict_root = (
        repo_root
        / "progress"
        / "20260828_tgrs_nested_nonpolar_matched_rect91_provenance"
        / "formal"
        / "audit"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-summary", type=Path, default=strict_root / "summary.json")
    parser.add_argument("--strict-audit", type=Path, default=strict_root / "audit.json")
    parser.add_argument("--hann-summary", type=Path)
    parser.add_argument("--hann-audit", type=Path)
    parser.add_argument(
        "--output-pdf", type=Path, default=repo_root / "paper/figs/fig3_matched_object_effects.pdf"
    )
    parser.add_argument(
        "--output-png", type=Path, default=repo_root / "paper/figs/fig3_matched_object_effects.png"
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=repo_root / "progress/fig3_matched_object_effects_provenance.json",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for required in (args.strict_summary, args.strict_audit):
        if not required.is_file():
            raise FileNotFoundError(required)
    if (args.hann_summary is None) != (args.hann_audit is None):
        raise ValueError("provide both --hann-summary and --hann-audit, or neither")
    if args.hann_summary is not None:
        for required in (args.hann_summary, args.hann_audit):
            if not required.is_file():
                raise FileNotFoundError(required)
    if args.dpi < 200:
        raise ValueError("PNG dpi must be at least 200")

    strict_summary = read_json(args.strict_summary)
    strict_audit = read_json(args.strict_audit)
    strict_info, object_maps = validate_strict_inputs(strict_summary, strict_audit)
    optional_hann: dict[str, Any] | None = None
    if args.hann_summary is not None:
        optional_hann = add_hann(
            object_maps,
            read_json(args.hann_summary),
            read_json(args.hann_audit),
        )
    figure_contract = make_figure(
        object_maps,
        strict_info,
        args.output_pdf,
        args.output_png,
        args.dpi,
    )

    script_path = Path(__file__).resolve()
    object_rows = []
    for object_code in figure_contract["object_order"]:
        row = {
            "object_code": object_code,
            "fold": object_maps["frequency"][object_code]["fold"],
            "frequency_gce_deg": object_maps["frequency"][object_code]["gce_deg"],
            "rect_ifft_gce_deg": object_maps["rect_ifft"][object_code]["gce_deg"],
            "frequency_minus_rect_gce_deg": float(
                strict_info["paired_map"][object_code]["mean_gce_difference_deg"]
            ),
        }
        if "hann_ifft" in object_maps:
            row["hann_ifft_gce_deg"] = object_maps["hann_ifft"][object_code]["gce_deg"]
        object_rows.append(row)
    provenance = {
        "status": "ok",
        "schema": "fig3-matched-object-effects-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(script_path), *sys.argv[1:]],
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "sources": {
            "strict_summary": {
                "path": str(args.strict_summary.resolve()),
                "sha256": sha256_file(args.strict_summary),
            },
            "strict_audit": {
                "path": str(args.strict_audit.resolve()),
                "sha256": sha256_file(args.strict_audit),
                "status": strict_audit["status"],
                "formal_runs": strict_audit["observed_task_count"],
                "prediction_rows": strict_audit["sample_row_count"],
            },
            "hann_summary": None,
            "hann_audit": None,
        },
        "estimand": (
            "For each held object, average GCE over all 60 nonpolar directions within seed, "
            "then over seeds 70--76; objects receive equal weight."
        ),
        "uncertainty": (
            "Percentile 95% interval obtained by resampling the 20 centered object effects; "
            "directions and seeds are repeated measurements, not independent target units."
        ),
        "strict_overall": strict_info["overall"],
        "optional_hann": optional_hann,
        "objects": object_rows,
        "figure_contract": figure_contract,
        "outputs": {
            "pdf": {
                "path": str(args.output_pdf.resolve()),
                "bytes": args.output_pdf.stat().st_size,
                "sha256": sha256_file(args.output_pdf),
            },
            "png": {
                "path": str(args.output_png.resolve()),
                "bytes": args.output_png.stat().st_size,
                "sha256": sha256_file(args.output_png),
            },
        },
    }
    if args.hann_summary is not None:
        provenance["sources"]["hann_summary"] = {
            "path": str(args.hann_summary.resolve()),
            "sha256": sha256_file(args.hann_summary),
        }
        provenance["sources"]["hann_audit"] = {
            "path": str(args.hann_audit.resolve()),
            "sha256": sha256_file(args.hann_audit),
            "status": read_json(args.hann_audit)["status"],
        }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "representations": figure_contract["representations"],
                "objects": len(object_rows),
                "pdf": str(args.output_pdf.resolve()),
                "png": str(args.output_png.resolve()),
                "provenance": str(args.provenance.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
