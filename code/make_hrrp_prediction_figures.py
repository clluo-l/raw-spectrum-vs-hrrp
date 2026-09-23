#!/usr/bin/env python3
"""Build supplementary Figs. S1--S2 from frozen tgrs811 predictions.

The accepted protocol is deliberately fail-closed: the global Frequency MLP
must contain 160 untouched ID samples and 4 x 40 held samples for every seed
70--76. Object intervals use a crossed seed-by-direction bootstrap because
the same eight directions are evaluated by every seed for a given object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


EXPECTED_SEEDS = tuple(range(70, 77))
EXPECTED_FOLDS = tuple(range(1, 5))
EXPECTED_DIRECTIONS_PER_OBJECT = 8
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED_ID = 20260828
BOOTSTRAP_SEED_HELD = 20260829
BOOTSTRAP_SEED_PHI_OFFSET = 10_000

FOLD_COLORS = {
    1: "#4e79a7",
    2: "#59a14f",
    3: "#f28e2b",
    4: "#b07aa1",
}

S1_CAPTION = (
    "Untouched tgrs811 ID-test statistics for the global Frequency MLP "
    "(160 samples per seed; seeds 70--76). Left: the theta-MAE and GCE "
    "violins pool all 1,120 predictions, whereas circular phi MAE excludes "
    "polar targets and pools 868 predictions (124 per seed); horizontal ticks "
    "mark pooled medians and the seven offset open circles are seed-wise "
    "means. Right: bars are object-wise mean GCE and filled circles are "
    "object-wise theta MAE over all eight fixed directions; open squares are "
    "circular phi MAE over only that object's 3--8 nonpolar directions. Bar "
    "and square whiskers are percentile 95% intervals from 10,000-replicate "
    "crossed bootstraps that independently resample the seven seeds and, "
    "respectively, all eight or only the eligible nonpolar directions."
)

S2_CAPTION = (
    "Held-geometry statistics for the global Frequency MLP over four outer "
    "folds and seeds 70--76. Each colored bar is a held object's mean GCE and "
    "each filled circle is its theta MAE over all eight untouched directions; "
    "open squares are circular phi MAE over only that object's 3--8 nonpolar "
    "directions. Each fold contains five held objects and 40 total held "
    "directions (30, 33, 31, and 30 nonpolar directions in folds 1--4). Bar "
    "and square whiskers are percentile 95% intervals from 10,000-replicate "
    "crossed bootstraps that independently resample the seven seeds and, "
    "respectively, all eight or only the eligible nonpolar directions."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style_axes(ax) -> None:
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def direction_set(group: pd.DataFrame) -> set[tuple[float, float]]:
    return {
        (float(theta), float(phi))
        for theta, phi in zip(group["theta_true_deg"], group["phi_true_deg"])
    }


def parse_boolean(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        bad = sorted(normalized.loc[mapped.isna()].unique())
        raise ValueError(f"unrecognized {column} values: {bad}")
    return mapped.astype(bool)


def validate_samples(samples: pd.DataFrame, model_tag: str) -> pd.DataFrame:
    required = {
        "model_tag",
        "protocol",
        "fold",
        "seed",
        "index",
        "object_code",
        "theta_true_deg",
        "phi_true_deg",
        "theta_err_deg",
        "phi_err_deg",
        "great_circle_err_deg",
        "pole",
        "top1",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"prediction CSV is missing columns: {missing}")

    selected = samples.loc[samples["model_tag"].astype(str) == model_tag].copy()
    if selected.empty:
        raise ValueError(f"model tag {model_tag!r} is absent from prediction CSV")

    numeric_columns = [
        "fold",
        "seed",
        "index",
        "theta_true_deg",
        "phi_true_deg",
        "theta_err_deg",
        "phi_err_deg",
        "great_circle_err_deg",
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    selected["fold"] = selected["fold"].astype(int)
    selected["seed"] = selected["seed"].astype(int)
    selected["index"] = selected["index"].astype(int)
    selected["protocol"] = selected["protocol"].astype(str)
    selected["object_code"] = selected["object_code"].astype(str)
    selected["_is_pole"] = parse_boolean(selected["pole"], "pole")

    finite_columns = ["theta_err_deg", "phi_err_deg", "great_circle_err_deg"]
    if not np.isfinite(selected[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("prediction errors contain non-finite values")
    for column in finite_columns:
        values = selected[column].to_numpy(dtype=float)
        if np.any((values < -1e-9) | (values > 180.0 + 1e-9)):
            raise ValueError(f"{column} contains values outside [0, 180] degrees")

    protocols = ("id", "outer1", "outer2", "outer3", "outer4")
    observed_protocols = tuple(sorted(selected["protocol"].unique()))
    if set(observed_protocols) != set(protocols):
        raise ValueError(f"expected protocols {protocols}, found {observed_protocols}")
    if tuple(sorted(selected["seed"].unique())) != EXPECTED_SEEDS:
        raise ValueError(
            f"final figures require exactly seeds {EXPECTED_SEEDS}; "
            f"found {tuple(sorted(selected['seed'].unique()))}"
        )
    if selected.duplicated(["protocol", "fold", "seed", "index"]).any():
        raise ValueError("duplicate protocol/fold/seed/index prediction rows found")

    expected_run_counts = {"id": 160, **{f"outer{fold}": 40 for fold in EXPECTED_FOLDS}}
    for protocol, expected_count in expected_run_counts.items():
        protocol_rows = selected.loc[selected["protocol"] == protocol]
        expected_fold = 0 if protocol == "id" else int(protocol.removeprefix("outer"))
        if set(protocol_rows["fold"].unique()) != {expected_fold}:
            raise ValueError(f"{protocol} rows do not carry fold={expected_fold}")
        counts = protocol_rows.groupby("seed").size().to_dict()
        expected = {seed: expected_count for seed in EXPECTED_SEEDS}
        if counts != expected:
            raise ValueError(f"{protocol} expected per-seed counts {expected}, found {counts}")

    id_rows = selected.loc[selected["protocol"] == "id"]
    if len(id_rows) != 160 * len(EXPECTED_SEEDS):
        raise ValueError(f"ID expected 1120 rows, found {len(id_rows)}")
    id_nonpolar_counts = id_rows.loc[~id_rows["_is_pole"]].groupby("seed").size().to_dict()
    expected_id_nonpolar = {seed: 124 for seed in EXPECTED_SEEDS}
    if id_nonpolar_counts != expected_id_nonpolar:
        raise ValueError(
            f"ID expected 124 nonpolar rows per seed, found {id_nonpolar_counts}"
        )
    id_objects = set(id_rows["object_code"].unique())
    if len(id_objects) != 20:
        raise ValueError(f"ID expected 20 objects, found {len(id_objects)}")

    held_rows = selected.loc[selected["protocol"].str.startswith("outer")]
    if len(held_rows) != 4 * 40 * len(EXPECTED_SEEDS):
        raise ValueError(f"held expected 1120 rows, found {len(held_rows)}")
    expected_held_nonpolar_per_fold = {1: 30, 2: 33, 3: 31, 4: 30}
    held_nonpolar_counts = (
        held_rows.loc[~held_rows["_is_pole"]].groupby(["fold", "seed"]).size().to_dict()
    )
    expected_held_nonpolar = {
        (fold, seed): count
        for fold, count in expected_held_nonpolar_per_fold.items()
        for seed in EXPECTED_SEEDS
    }
    if held_nonpolar_counts != expected_held_nonpolar:
        raise ValueError(
            "held nonpolar counts do not match the frozen 30/33/31/30 "
            f"directions-per-fold protocol: {held_nonpolar_counts}"
        )
    held_objects_by_fold: dict[int, set[str]] = {}
    for fold in EXPECTED_FOLDS:
        objects = set(held_rows.loc[held_rows["fold"] == fold, "object_code"].unique())
        if len(objects) != 5:
            raise ValueError(f"fold {fold} expected 5 held objects, found {len(objects)}")
        held_objects_by_fold[fold] = objects
    held_objects = set().union(*held_objects_by_fold.values())
    if sum(len(objects) for objects in held_objects_by_fold.values()) != len(held_objects):
        raise ValueError("a held object appears in more than one outer fold")
    if held_objects != id_objects:
        raise ValueError("ID and held protocols do not cover the same 20 objects")

    for protocol, protocol_rows in selected.groupby("protocol", sort=True):
        for object_code, object_rows in protocol_rows.groupby("object_code", sort=True):
            reference_directions: set[tuple[float, float]] | None = None
            for seed, seed_rows in object_rows.groupby("seed", sort=True):
                if len(seed_rows) != EXPECTED_DIRECTIONS_PER_OBJECT:
                    raise ValueError(
                        f"{protocol}/{object_code}/seed{seed} expected 8 rows, "
                        f"found {len(seed_rows)}"
                    )
                directions = direction_set(seed_rows)
                if len(directions) != EXPECTED_DIRECTIONS_PER_OBJECT:
                    raise ValueError(
                        f"{protocol}/{object_code}/seed{seed} does not contain "
                        "eight unique directions"
                    )
                if reference_directions is None:
                    reference_directions = directions
                elif directions != reference_directions:
                    raise ValueError(
                        f"direction identities differ across seeds for {protocol}/{object_code}"
                    )
            pole_by_direction = (
                object_rows.groupby(["theta_true_deg", "phi_true_deg"], sort=True)["_is_pole"]
                .nunique()
            )
            if not pole_by_direction.eq(1).all():
                raise ValueError(
                    f"polar eligibility differs across seeds for {protocol}/{object_code}"
                )
            nonpolar_counts = object_rows.loc[~object_rows["_is_pole"]].groupby("seed").size()
            if len(nonpolar_counts) != len(EXPECTED_SEEDS) or nonpolar_counts.nunique() != 1:
                raise ValueError(
                    f"nonpolar direction count differs across seeds for {protocol}/{object_code}"
                )
            if not 1 <= int(nonpolar_counts.iloc[0]) <= EXPECTED_DIRECTIONS_PER_OBJECT:
                raise ValueError(f"{protocol}/{object_code} has no eligible nonpolar direction")

    return selected.sort_values(["protocol", "fold", "seed", "index"]).reset_index(drop=True)


def crossed_seed_direction_interval(
    group: pd.DataFrame,
    value_key: str,
    bootstrap_seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
    expected_directions: int | None = EXPECTED_DIRECTIONS_PER_OBJECT,
) -> tuple[float, float]:
    """Resample seeds and the object's matched directions independently."""

    seeds = tuple(sorted(int(seed) for seed in group["seed"].unique()))
    directions = tuple(sorted(direction_set(group)))
    if seeds != EXPECTED_SEEDS:
        raise ValueError("crossed bootstrap requires exactly seeds 70--76")
    if expected_directions is not None and len(directions) != expected_directions:
        raise ValueError(
            f"crossed bootstrap expected {expected_directions} directions, "
            f"found {len(directions)}"
        )
    if not directions:
        raise ValueError("crossed bootstrap received no eligible directions")

    keyed = group.set_index(["seed", "theta_true_deg", "phi_true_deg"])[value_key]
    if keyed.index.duplicated().any():
        raise ValueError("duplicate seed-direction cell in object bootstrap")
    matrix = np.asarray(
        [
            [float(keyed.loc[(seed, theta, phi)]) for theta, phi in directions]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    if matrix.shape != (len(EXPECTED_SEEDS), len(directions)):
        raise ValueError(f"unexpected bootstrap matrix shape {matrix.shape}")

    rng = np.random.default_rng(bootstrap_seed)
    sampled_seeds = rng.integers(0, matrix.shape[0], size=(iterations, matrix.shape[0]))
    sampled_directions = rng.integers(0, matrix.shape[1], size=(iterations, matrix.shape[1]))
    draws = matrix[
        sampled_seeds[:, :, np.newaxis],
        sampled_directions[:, np.newaxis, :],
    ].mean(axis=(1, 2))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_objects(samples: pd.DataFrame, bootstrap_seed: int) -> pd.DataFrame:
    rows = []
    for object_index, (object_code, group) in enumerate(samples.groupby("object_code", sort=True)):
        folds = sorted(int(fold) for fold in group["fold"].unique())
        if len(folds) != 1:
            raise ValueError(f"object {object_code} maps to multiple folds: {folds}")
        gce = float(group["great_circle_err_deg"].mean())
        lower, upper = crossed_seed_direction_interval(
            group,
            "great_circle_err_deg",
            bootstrap_seed + object_index,
        )
        nonpolar = group.loc[~group["_is_pole"]]
        nonpolar_directions = len(direction_set(nonpolar))
        phi = float(nonpolar["phi_err_deg"].mean())
        phi_lower, phi_upper = crossed_seed_direction_interval(
            nonpolar,
            "phi_err_deg",
            bootstrap_seed + BOOTSTRAP_SEED_PHI_OFFSET + object_index,
            expected_directions=nonpolar_directions,
        )
        rows.append(
            {
                "object_code": str(object_code),
                "fold": folds[0],
                "theta_mae_deg": float(group["theta_err_deg"].mean()),
                "phi_mae_deg": phi,
                "phi_ci_low": phi_lower,
                "phi_ci_high": phi_upper,
                "great_circle_mae_deg": gce,
                "gce_ci_low": lower,
                "gce_ci_high": upper,
                "prediction_count": int(len(group)),
                "nonpolar_prediction_count": int(len(nonpolar)),
                "seeds": int(group["seed"].nunique()),
                "directions": int(len(direction_set(group))),
                "nonpolar_directions": int(nonpolar_directions),
            }
        )
    return pd.DataFrame(rows)


def save_both(fig: plt.Figure, out_path: Path) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path, pdf_path


def draw_object_panel(ax, objects: pd.DataFrame, colors: list[str], show_fold_legend: bool) -> None:
    y = np.arange(len(objects))
    theta_y = y - 0.16
    phi_y = y + 0.16
    gce = objects["great_circle_mae_deg"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            gce - objects["gce_ci_low"].to_numpy(dtype=float),
            objects["gce_ci_high"].to_numpy(dtype=float) - gce,
        ]
    )
    ax.barh(
        y,
        gce,
        color=colors,
        alpha=0.86,
        edgecolor="#333333",
        linewidth=0.35,
        zorder=2,
    )
    ax.errorbar(
        gce,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2.0,
        zorder=3,
    )
    phi = objects["phi_mae_deg"].to_numpy(dtype=float)
    phi_xerr = np.vstack(
        [
            phi - objects["phi_ci_low"].to_numpy(dtype=float),
            objects["phi_ci_high"].to_numpy(dtype=float) - phi,
        ]
    )
    ax.errorbar(
        phi,
        phi_y,
        xerr=phi_xerr,
        fmt="none",
        ecolor="#71808d",
        elinewidth=0.65,
        capsize=1.6,
        zorder=3,
    )
    theta_handle = ax.scatter(
        objects["theta_mae_deg"],
        theta_y,
        s=19,
        marker="o",
        color="#15283a",
        zorder=4,
        label=r"$\theta$ MAE",
    )
    phi_handle = ax.scatter(
        phi,
        phi_y,
        s=21,
        marker="s",
        facecolor="white",
        edgecolor="#15283a",
        linewidth=0.8,
        zorder=4,
        label=r"nonpolar circular $\phi$ MAE",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(objects["object_code"], fontsize=7)
    ax.set_xlabel("Angular error (deg)", fontsize=9)
    ax.set_xlim(left=0)
    style_axes(ax)
    handles = [theta_handle, phi_handle]
    if show_fold_legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="#333333",
                markersize=5.5,
                label=f"Fold {fold}",
            )
            for fold, color in FOLD_COLORS.items()
        ] + handles
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(handles),
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )


def make_id_figure(samples: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    selected = samples.loc[samples["protocol"] == "id"].copy()
    objects = summarize_objects(selected, bootstrap_seed=BOOTSTRAP_SEED_ID).sort_values(
        "great_circle_mae_deg", ascending=True
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 4.15),
        gridspec_kw={"width_ratios": [0.80, 1.45]},
    )
    metric_columns = ["theta_err_deg", "phi_err_deg", "great_circle_err_deg"]
    nonpolar = selected.loc[~selected["_is_pole"]]
    values = [
        selected["theta_err_deg"].to_numpy(dtype=float),
        nonpolar["phi_err_deg"].to_numpy(dtype=float),
        selected["great_circle_err_deg"].to_numpy(dtype=float),
    ]
    violins = axes[0].violinplot(
        values,
        positions=[1, 2, 3],
        widths=0.72,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], ("#4e79a7", "#e07b39", "#59a14f")):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.55)
    violins["cmedians"].set_color("#111111")
    violins["cmedians"].set_linewidth(1.0)

    seed_means = pd.DataFrame(
        {
            "theta_err_deg": selected.groupby("seed", sort=True)["theta_err_deg"].mean(),
            "phi_err_deg": nonpolar.groupby("seed", sort=True)["phi_err_deg"].mean(),
            "great_circle_err_deg": selected.groupby("seed", sort=True)[
                "great_circle_err_deg"
            ].mean(),
        }
    )
    jitter = np.linspace(-0.15, 0.15, len(seed_means))
    for column_index, column in enumerate(metric_columns, start=1):
        axes[0].scatter(
            column_index + jitter,
            seed_means[column].to_numpy(dtype=float),
            s=18,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.7,
            zorder=4,
        )
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markersize=4.2,
                label="seed-wise mean",
            )
        ],
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        borderaxespad=0.0,
    )
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(
        [r"$\theta$ MAE", "nonpolar\n" + r"circular $\phi$ MAE", "GCE"],
        fontsize=8,
    )
    axes[0].set_ylabel("Prediction-level error (deg)", fontsize=9)
    axes[0].set_ylim(0, 180)
    axes[0].grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].text(-0.16, 1.03, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold")

    draw_object_panel(axes[1], objects, ["#789ac6"] * len(objects), show_fold_legend=False)
    axes[1].text(-0.10, 1.03, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(pad=0.55, w_pad=1.1)
    return objects, save_both(fig, out_path)


def make_held_figure(samples: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    selected = samples.loc[samples["protocol"].str.startswith("outer")].copy()
    objects = summarize_objects(selected, bootstrap_seed=BOOTSTRAP_SEED_HELD).sort_values(
        "great_circle_mae_deg", ascending=True
    )
    colors = [FOLD_COLORS[int(fold)] for fold in objects["fold"]]
    fig, ax = plt.subplots(figsize=(7.25, 4.45))
    draw_object_panel(ax, objects, colors, show_fold_legend=True)
    fig.tight_layout(pad=0.55)
    return objects, save_both(fig, out_path)


def top1_mean(series: pd.Series) -> float:
    return float(parse_boolean(series, "top1").mean())


def aggregate_record(rows: pd.DataFrame) -> dict[str, float | int]:
    nonpolar = rows.loc[~rows["_is_pole"]]
    if nonpolar.empty:
        raise ValueError("aggregate contains no nonpolar rows for circular phi MAE")
    return {
        "prediction_count": int(len(rows)),
        "nonpolar_prediction_count": int(len(nonpolar)),
        "theta_mae_deg": float(rows["theta_err_deg"].mean()),
        "circular_phi_mae_deg": float(nonpolar["phi_err_deg"].mean()),
        "great_circle_mae_deg": float(rows["great_circle_err_deg"].mean()),
        "top1": top1_mean(rows["top1"]),
    }


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def source_records(samples: pd.DataFrame) -> list[dict]:
    if not {"source_path", "source_sha256"}.issubset(samples.columns):
        return []
    grouped = (
        samples.groupby(
            ["source_path", "source_sha256", "protocol", "fold", "seed"],
            sort=True,
            dropna=False,
        )
        .size()
        .reset_index(name="prediction_count")
    )
    return dataframe_records(grouped)


def grouped_aggregate_records(rows: pd.DataFrame, keys: list[str]) -> list[dict]:
    records = []
    for group_key, group in rows.groupby(keys, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        record = {key: int(value) for key, value in zip(keys, key_values)}
        record.update(aggregate_record(group))
        records.append(record)
    return records


def write_provenance(
    samples: pd.DataFrame,
    samples_path: Path,
    model_tag: str,
    id_objects: pd.DataFrame,
    held_objects: pd.DataFrame,
    figure_paths: dict[str, tuple[Path, Path]],
    out_path: Path,
) -> None:
    id_rows = samples.loc[samples["protocol"] == "id"]
    held_rows = samples.loc[samples["protocol"].str.startswith("outer")]

    output_records = {}
    for name, (png_path, pdf_path) in figure_paths.items():
        output_records[name] = {
            "png": str(png_path.resolve()),
            "png_sha256": sha256_file(png_path),
            "pdf": str(pdf_path.resolve()),
            "pdf_sha256": sha256_file(pdf_path),
        }

    provenance = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "command": sys.argv,
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_csv": str(samples_path.resolve()),
        "input_csv_sha256": sha256_file(samples_path),
        "model_tag": model_tag,
        "protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "id_samples_per_seed": 160,
            "id_nonpolar_samples_per_seed": 124,
            "id_objects": 20,
            "id_directions_per_object": EXPECTED_DIRECTIONS_PER_OBJECT,
            "held_folds": list(EXPECTED_FOLDS),
            "held_objects_per_fold": 5,
            "held_directions_per_object": EXPECTED_DIRECTIONS_PER_OBJECT,
            "held_directions_per_fold": 40,
            "held_nonpolar_directions_per_fold": {"1": 30, "2": 33, "3": 31, "4": 30},
            "nonpolar_directions_per_object_range": [3, 8],
        },
        "bootstrap": {
            "confidence_level": 0.95,
            "interval": "percentile",
            "iterations": BOOTSTRAP_ITERATIONS,
            "gce_unit": (
                "within each object, resample seven seed IDs and eight matched "
                "direction IDs independently with replacement, then average the "
                "resampled seed-by-direction Cartesian cells"
            ),
            "phi_unit": (
                "within each object, discard polar targets, then resample seven "
                "seed IDs and that object's 3--8 matched nonpolar direction IDs "
                "independently with replacement and average the resampled "
                "seed-by-direction Cartesian cells"
            ),
            "id_base_seed": BOOTSTRAP_SEED_ID,
            "held_base_seed": BOOTSTRAP_SEED_HELD,
        },
        "remote_sources": source_records(samples),
        "aggregates": {
            "id": {
                **aggregate_record(id_rows),
                "per_seed": grouped_aggregate_records(id_rows, ["seed"]),
            },
            "held": {
                **aggregate_record(held_rows),
                "per_fold": grouped_aggregate_records(held_rows, ["fold"]),
                "per_fold_seed": grouped_aggregate_records(held_rows, ["fold", "seed"]),
            },
        },
        "per_object": {
            "id": dataframe_records(id_objects.sort_values("object_code")),
            "held": dataframe_records(held_objects.sort_values(["fold", "object_code"])),
        },
        "figure_semantics": {
            "bar": "object-wise mean great-circle error (GCE)",
            "filled_circle": "object-wise mean absolute polar-angle error (theta MAE)",
            "open_square": (
                "object-wise mean circular azimuth error (phi MAE), restricted "
                "to the object's nonpolar target directions"
            ),
            "bar_whisker": (
                "crossed seed-by-direction bootstrap percentile 95% interval "
                "for object-wise mean GCE over all eight directions"
            ),
            "square_whisker": (
                "crossed seed-by-direction bootstrap percentile 95% interval "
                "for object-wise circular phi MAE over only 3--8 nonpolar directions"
            ),
            "s1_violin": (
                "pooled prediction-level distribution: 1120 all-sample theta/GCE "
                "errors and 868 nonpolar-only circular phi errors"
            ),
            "s1_open_circle": (
                "seed-wise metric mean; seven points per metric, horizontally "
                "offset only for visibility; phi excludes polar targets"
            ),
        },
        "suggested_captions": {"figS1": S1_CAPTION, "figS2": S2_CAPTION},
        "outputs": output_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-csv",
        type=Path,
        required=True,
        help="combined frozen tgrs811 prediction CSV with protocol/fold/seed metadata",
    )
    parser.add_argument("--model-tag", default="frequency_mlp")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="machine-readable statistics/provenance (default: <out-dir>/figS1_figS2_statistics.json)",
    )
    args = parser.parse_args()

    samples_path = args.samples_csv.resolve()
    samples = validate_samples(pd.read_csv(samples_path), args.model_tag)
    id_path = args.out_dir / "figS1_id_metric_statistics.png"
    held_path = args.out_dir / "figS2_ood_aircraft_statistics.png"
    summary_path = args.summary_json or (args.out_dir / "figS1_figS2_statistics.json")

    id_objects, id_figure_paths = make_id_figure(samples, id_path)
    held_objects, held_figure_paths = make_held_figure(samples, held_path)
    write_provenance(
        samples,
        samples_path,
        args.model_tag,
        id_objects,
        held_objects,
        {"figS1": id_figure_paths, "figS2": held_figure_paths},
        summary_path,
    )

    print(id_figure_paths[0])
    print(held_figure_paths[0])
    print(summary_path)
    print(
        json.dumps(
            {
                "id": aggregate_record(samples.loc[samples["protocol"] == "id"]),
                "held": aggregate_record(samples.loc[samples["protocol"].str.startswith("outer")]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
