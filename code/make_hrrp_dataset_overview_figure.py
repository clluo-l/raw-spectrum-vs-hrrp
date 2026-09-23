#!/usr/bin/env python3
"""Create the GT-IP dataset overview figure for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: E402


FOLDS = [
    ("Fold 1", ["b871", "bb7d", "b827", "b905", "bbc6"], "#4e79a7"),
    ("Fold 2", ["b80b", "ba0f", "b7c1", "b9e6", "bb7c"], "#59a14f"),
    ("Fold 3", ["b943", "b97b", "b812", "bc2c", "b974"], "#f28e2b"),
    ("Fold 4", ["bb26", "b7fd", "baa9", "b979", "b8ed"], "#b07aa1"),
]

DESCRIPTOR_NAMES = (
    "frequency_cv",
    "frequency_total_variation",
    "phi_energy_fraction",
    "hrrp_circular_90pct_energy_width_m",
    "hrrp_effective_bins",
    "hrrp_total_variation",
    "absolute_range_centroid_m",
    "hrrp_rms_spread_m",
    "hrrp_periodicity_acf_0p35_to_2m",
)
LOG_DESCRIPTOR_INDICES = (0, 1, 3, 4, 5, 6, 7)


def read_ascii_stl(path: Path, max_triangles: int = 0) -> np.ndarray:
    vertices: list[list[float]] = []
    triangles: list[list[list[float]]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(vertices) == 3:
                triangles.append(vertices)
                vertices = []
    tris = np.asarray(triangles, dtype=np.float32)
    if max_triangles > 0 and len(tris) > max_triangles:
        idx = np.linspace(0, len(tris) - 1, max_triangles).astype(int)
        tris = tris[idx]
    return tris


def rotation_matrix(rx_deg: float = 60.0, ry_deg: float = 0.0, rz_deg: float = -35.0) -> np.ndarray:
    rx, ry, rz = [math.radians(v) for v in (rx_deg, ry_deg, rz_deg)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rzm @ rym @ rxm


def project_triangles(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = tris.reshape(-1, 3)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    scale = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    scale = max(scale, 1e-6)
    norm = (tris - center) / scale
    rotated = norm @ rotation_matrix().T
    projected = rotated[:, :, [0, 2]]
    depth = rotated[:, :, 1].mean(axis=1)
    order = np.argsort(depth)
    return projected[order], depth[order]


def mesh_path(geometry_root: Path, object_code: str) -> Path:
    nested = geometry_root / object_code / f"{object_code}_clean_ascii.stl"
    if nested.exists():
        return nested
    return geometry_root / f"{object_code}_clean_ascii.stl"


def project_surface_triangles(tris: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray]:
    """Project a solid mesh and retain Fig. 3-style facet illumination."""

    pts = tris.reshape(-1, 3).astype(np.float64)
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    scale = max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), 1e-12)
    normalized = (tris.astype(np.float64) - center) / scale
    rotated = normalized @ rotation_matrix().astype(np.float64).T
    projected = rotated[:, :, [0, 2]]
    depth = rotated[:, :, 1].mean(axis=1)
    order = np.argsort(depth)

    normals = np.cross(rotated[:, 1] - rotated[:, 0], rotated[:, 2] - rotated[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, -0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)
    illumination = np.clip(normals @ light, 0.0, 1.0)
    base = np.array([int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)], dtype=np.float64)
    rgb = np.clip(base[None, :] * (0.58 + 0.42 * illumination[:, None]), 0.0, 1.0)
    rgba = np.column_stack([rgb, np.ones(rgb.shape[0], dtype=np.float64)])
    return projected[order].astype(np.float32), rgba[order].astype(np.float32)


def fold_objects() -> list[str]:
    return [obj for _, objects, _ in FOLDS for obj in objects]


def build_mesh_cache(geometry_root: Path, cache_path: Path, max_triangles: int) -> None:
    arrays = {}
    colors = {obj: color for _, objects, color in FOLDS for obj in objects}
    for obj in fold_objects():
        stl_path = mesh_path(geometry_root, obj)
        tris = read_ascii_stl(stl_path, max_triangles=max_triangles)
        polys, rgba = project_surface_triangles(tris, colors[obj])
        arrays[f"{obj}__polys"] = polys
        arrays[f"{obj}__rgba"] = rgba
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    print(cache_path)


def load_mesh_cache(cache_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    if not cache_path.exists():
        return {}
    data = np.load(cache_path)
    cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key in data.files:
        if key.endswith("__polys"):
            obj = key.removesuffix("__polys")
            rgba_key = f"{obj}__rgba"
            cache[obj] = (data[key], data[rgba_key] if rgba_key in data.files else None)
        elif "__" not in key:
            cache[key] = (data[key], None)
    return cache


def draw_mesh(
    ax,
    stl_path: Path,
    color: str,
    label: str,
    cached: tuple[np.ndarray, np.ndarray | None] | None,
    max_triangles: int,
) -> None:
    from matplotlib.collections import PolyCollection

    if cached is None:
        tris = read_ascii_stl(stl_path, max_triangles=max_triangles)
        polys, rgba = project_surface_triangles(tris, color)
    else:
        polys, rgba = cached
    collection = PolyCollection(
        polys,
        facecolors=rgba if rgba is not None else color,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        alpha=1.0,
        rasterized=True,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02,
        0.04,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="medium",
        color="#202124",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_energy_width(probability: np.ndarray, fraction: float = 0.90) -> int:
    probability = np.asarray(probability, dtype=np.float64)
    target = fraction * float(probability.sum())
    doubled = np.concatenate([probability, probability])
    right = 0
    accumulated = 0.0
    best = probability.size
    for left in range(probability.size):
        while right < left + probability.size and accumulated < target:
            accumulated += doubled[right]
            right += 1
        if accumulated >= target:
            best = min(best, right - left)
        accumulated -= doubled[left]
    return int(best)


def sample_descriptors(data: np.lib.npyio.NpzFile) -> np.ndarray:
    required = {
        "freq_ghz",
        "complex_etheta_real",
        "complex_etheta_imag",
        "complex_ephi_real",
        "complex_ephi_imag",
        "hrrp_vector_mag",
        "range_m",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"sample is missing required arrays: {sorted(missing)}")

    e_theta = data["complex_etheta_real"].astype(np.float64) + 1j * data[
        "complex_etheta_imag"
    ].astype(np.float64)
    e_phi = data["complex_ephi_real"].astype(np.float64) + 1j * data[
        "complex_ephi_imag"
    ].astype(np.float64)
    frequency_magnitude = np.sqrt(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)
    hrrp = data["hrrp_vector_mag"].astype(np.float64)
    relative_range = data["range_m"].astype(np.float64)
    if not (
        np.all(np.isfinite(frequency_magnitude))
        and np.all(np.isfinite(hrrp))
        and np.all(np.isfinite(relative_range))
    ):
        raise ValueError("non-finite values in representative-sample inputs")

    probability = hrrp**2
    probability /= max(float(probability.sum()), 1e-300)
    spacing_m = float(np.median(np.diff(relative_range)))
    effective_bins = float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))))
    width90_m = circular_energy_width(probability, 0.90) * spacing_m
    hrrp_normalized = hrrp / max(float(hrrp.max()), 1e-300)
    centroid_m = float(np.sum(relative_range * probability))
    rms_spread_m = float(np.sqrt(np.sum((relative_range - centroid_m) ** 2 * probability)))

    central = np.abs(relative_range) <= 3.0
    centered_profile = hrrp_normalized[central] - float(np.mean(hrrp_normalized[central]))
    autocorrelation = np.correlate(centered_profile, centered_profile, mode="full")[
        centered_profile.size - 1 :
    ]
    autocorrelation /= max(float(autocorrelation[0]), 1e-300)
    lag_lo = max(1, int(round(0.35 / spacing_m)))
    lag_hi = min(autocorrelation.size, int(round(2.0 / spacing_m)) + 1)
    periodicity = float(np.max(autocorrelation[lag_lo:lag_hi]))

    theta_energy = float(np.sum(np.abs(e_theta) ** 2))
    phi_energy = float(np.sum(np.abs(e_phi) ** 2))
    return np.asarray(
        [
            float(np.std(frequency_magnitude) / np.mean(frequency_magnitude)),
            float(np.mean(np.abs(np.diff(frequency_magnitude))) / np.mean(frequency_magnitude)),
            phi_energy / max(theta_energy + phi_energy, 1e-300),
            width90_m,
            effective_bins,
            float(np.mean(np.abs(np.diff(hrrp_normalized))) / np.mean(hrrp_normalized)),
            abs(centroid_m),
            rms_spread_m,
            periodicity,
        ],
        dtype=np.float64,
    )


def prominent_peak_diagnostics(data: np.lib.npyio.NpzFile) -> dict[str, float | int | list[float] | None]:
    from scipy.signal import find_peaks

    hrrp = data["hrrp_vector_mag"].astype(np.float64)
    hrrp /= max(float(hrrp.max()), 1e-300)
    relative_range = data["range_m"].astype(np.float64)
    spacing_m = float(np.median(np.diff(relative_range)))
    peaks, _ = find_peaks(
        hrrp,
        prominence=0.08,
        distance=max(1, int(round(0.15 / spacing_m))),
    )
    peak_ranges = relative_range[peaks]
    spacing_cv = None
    if peaks.size >= 3:
        separations = np.diff(peak_ranges)
        spacing_cv = float(np.std(separations) / np.mean(separations))
    return {
        "prominence_threshold": 0.08,
        "minimum_separation_m": 0.15,
        "prominent_peak_count": int(peaks.size),
        "prominent_peak_ranges_m": [float(value) for value in peak_ranges],
        "prominent_peak_spacing_cv": spacing_cv,
    }


def transform_descriptors(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float64).copy()
    transformed[..., list(LOG_DESCRIPTOR_INDICES)] = np.log(
        np.maximum(transformed[..., list(LOG_DESCRIPTOR_INDICES)], 1e-12)
    )
    return transformed


def descriptor_summary(values: np.ndarray) -> dict[str, dict[str, float]]:
    quantiles = (0, 10, 25, 50, 75, 90, 100)
    labels = ("min", "p10", "p25", "median", "p75", "p90", "max")
    return {
        name: {
            label: float(value)
            for label, value in zip(labels, np.percentile(values[:, column], quantiles), strict=True)
        }
        for column, name in enumerate(DESCRIPTOR_NAMES)
    }


def split_membership(protocol_splits_root: Path, sample_relative_path: str) -> list[str]:
    if not protocol_splits_root.exists():
        return []
    normalized = sample_relative_path.replace("\\", "/")
    memberships = []
    for split_path in sorted(protocol_splits_root.rglob("*.txt")):
        rows = {
            line.strip().replace("\\", "/")
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if normalized in rows:
            memberships.append(split_path.relative_to(protocol_splits_root).with_suffix("").as_posix())
    return memberships


def select_representative_sample(
    dataset_root: Path,
    sample_out: Path,
    provenance_out: Path,
    previous_sample: Path | None,
    protocol_splits_root: Path,
) -> None:
    """Select a non-polar sample nearest the population's robust descriptor center."""

    index_path = dataset_root / "indexes" / "official" / "samples_index.jsonl"
    manifest_path = dataset_root / "dataset_manifest.json"
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1680:
        raise ValueError(f"expected 1680 indexed samples, found {len(rows)}")

    descriptors = []
    peak_counts = []
    for row in rows:
        with np.load(dataset_root / row["path"], allow_pickle=False) as data:
            descriptors.append(sample_descriptors(data))
            peak_counts.append(int(prominent_peak_diagnostics(data)["prominent_peak_count"]))
    values = np.asarray(descriptors, dtype=np.float64)
    transformed = transform_descriptors(values)
    population_median = np.median(transformed, axis=0)
    population_mad = np.median(np.abs(transformed - population_median), axis=0)
    robust_scale = np.maximum(1.4826 * population_mad, 1e-9)
    robust_z = (transformed - population_median) / robust_scale
    scores = np.sqrt(np.mean(robust_z**2, axis=1))

    eligible = [
        index
        for index, row in enumerate(rows)
        if 0.0 < float(row["theta_inc_deg"]) < 180.0
    ]
    if len(eligible) != 1200:
        raise ValueError(f"expected 1200 non-polar candidates, found {len(eligible)}")
    selected_index = min(eligible, key=lambda index: (float(scores[index]), str(rows[index]["sample_id"])))
    selected_row = rows[selected_index]
    selected_source = dataset_root / selected_row["path"]
    selected_fold = next(
        fold_index
        for fold_index, (_, objects, _) in enumerate(FOLDS, start=1)
        if selected_row["object_code"] in objects
    )

    sample_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_source, sample_out)
    with np.load(sample_out, allow_pickle=False) as selected_data:
        selected_metadata = json.loads(str(selected_data["metadata_json"]))
        selected_peak_diagnostics = prominent_peak_diagnostics(selected_data)

    overall_order = sorted(range(len(rows)), key=lambda index: (float(scores[index]), rows[index]["sample_id"]))
    eligible_order = sorted(eligible, key=lambda index: (float(scores[index]), rows[index]["sample_id"]))
    selected_values = values[selected_index]
    percentiles = 100.0 * np.mean(values <= selected_values[None, :], axis=0)
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_manifest_path = protocol_splits_root / "manifest.json"

    previous_payload = None
    if previous_sample is not None and previous_sample.exists():
        with np.load(previous_sample, allow_pickle=False) as previous_data:
            previous_metadata = json.loads(str(previous_data["metadata_json"]))
            previous_values = sample_descriptors(previous_data)
            previous_peaks = prominent_peak_diagnostics(previous_data)
        previous_transformed = transform_descriptors(previous_values)
        previous_z = (previous_transformed - population_median) / robust_scale
        previous_sample_id = str(previous_metadata.get("job_id", previous_sample.stem.removesuffix("_hrrp")))
        matching_rows = [row for row in rows if row["sample_id"] == previous_sample_id]
        current_same_path = dataset_root / matching_rows[0]["path"] if matching_rows else None
        previous_payload = {
            "sample_id": previous_sample_id,
            "asset_path": previous_sample.as_posix(),
            "asset_sha256": sha256_file(previous_sample),
            "current_dataset_same_id_sha256": sha256_file(current_same_path) if current_same_path else None,
            "matches_current_dataset_same_id": bool(
                current_same_path and sha256_file(previous_sample) == sha256_file(current_same_path)
            ),
            "robust_rms_z_score": float(np.sqrt(np.mean(previous_z**2))),
            "descriptor_values": {
                name: float(value) for name, value in zip(DESCRIPTOR_NAMES, previous_values, strict=True)
            },
            "descriptor_percentiles": {
                name: float(100.0 * np.mean(values[:, column] <= previous_values[column]))
                for column, name in enumerate(DESCRIPTOR_NAMES)
            },
            "peak_diagnostics": previous_peaks,
            "diagnosis": (
                "The former hard-coded figure input is a stale pre-backfill mirror of a direction listed "
                "in anomaly_fix_backfill. Its eight prominent HRRP peaks create the visible repeated-lobe "
                "pattern; its file hash does not match the current centered-v2 sample with the same ID."
            ),
        }

    provenance = {
        "artifact": "paper/figs/fig2_dataset_overview.png",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_algorithm": {
            "version": "fig2_robust_descriptor_medoid_v1",
            "population": "all 1680 current centered complex-E-field samples",
            "candidate_filter": (
                "non-polar directions only (0 < theta < 180 deg), avoiding azimuth-degenerate pole examples"
            ),
            "candidate_count": len(eligible),
            "descriptors": list(DESCRIPTOR_NAMES),
            "transform": {
                "log_transformed": [DESCRIPTOR_NAMES[index] for index in LOG_DESCRIPTOR_INDICES],
                "untransformed": [
                    name for index, name in enumerate(DESCRIPTOR_NAMES) if index not in LOG_DESCRIPTOR_INDICES
                ],
            },
            "standardization": "population median and 1.4826 * MAD per descriptor",
            "score": "root mean square of the nine robust z-scores; minimum score wins",
            "tie_break": "lexicographically smallest sample_id",
        },
        "dataset": {
            "canonical_root": dataset_manifest.get("output_root"),
            "local_mirror_root": str(dataset_root),
            "version": dataset_manifest.get("version"),
            "sample_count": int(dataset_manifest.get("sample_count", len(rows))),
            "object_count": 20,
            "source_directions_per_object": 84,
            "benchmark_nonpolar_directions_per_object": 60,
            "frequency_grid_ghz": {"start": 0.10, "stop": 1.00, "step": 0.01, "count": 91},
            "nfft": int(dataset_manifest.get("nfft", 256)),
            "anomaly_fix_backfill_count": int(dataset_manifest.get("anomaly_fix_backfill_count", 0)),
            "anomaly_fix_backfill_ts": dataset_manifest.get("anomaly_fix_backfill_ts"),
            "manifest_sha256": sha256_file(manifest_path),
            "official_index_sha256": sha256_file(index_path),
        },
        "nested_nonpolar_protocol": {
            "manifest_path": protocol_manifest_path.as_posix(),
            "manifest_sha256": sha256_file(protocol_manifest_path) if protocol_manifest_path.exists() else None,
            "selected_sample_membership": split_membership(protocol_splits_root, selected_row["path"]),
        },
        "selected_sample": {
            "sample_id": selected_row["sample_id"],
            "object_code": selected_row["object_code"],
            "material_distribution_id": selected_row["material_distribution_id"],
            "theta_inc_deg": float(selected_row["theta_inc_deg"]),
            "phi_inc_deg": float(selected_row["phi_inc_deg"]),
            "fold": selected_fold,
            "source_dataset_path": selected_row["path"],
            "source_index_complex_sample_sha256": selected_row["sha256"],
            "source_complex_sample_sha256_in_metadata": selected_metadata.get("source_sample_sha256"),
            "centered_asset_path": sample_out.as_posix(),
            "centered_asset_sha256": sha256_file(sample_out),
            "centered_asset_bytes": sample_out.stat().st_size,
            "hrrp_version": selected_metadata.get("hrrp_version"),
            "overall_rank": overall_order.index(selected_index) + 1,
            "eligible_nonpolar_rank": eligible_order.index(selected_index) + 1,
            "robust_rms_z_score": float(scores[selected_index]),
            "descriptor_values": {
                name: float(value) for name, value in zip(DESCRIPTOR_NAMES, selected_values, strict=True)
            },
            "descriptor_percentiles": {
                name: float(value) for name, value in zip(DESCRIPTOR_NAMES, percentiles, strict=True)
            },
            "robust_z_scores": {
                name: float(value) for name, value in zip(DESCRIPTOR_NAMES, robust_z[selected_index], strict=True)
            },
            "peak_diagnostics": selected_peak_diagnostics,
        },
        "population_descriptor_summary": descriptor_summary(values),
        "population_prominent_peak_count": {
            label: float(value)
            for label, value in zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"),
                np.percentile(np.asarray(peak_counts), (0, 10, 25, 50, 75, 90, 100)),
                strict=True,
            )
        },
        "top_five_eligible_candidates": [
            {"sample_id": rows[index]["sample_id"], "robust_rms_z_score": float(scores[index])}
            for index in eligible_order[:5]
        ],
        "previous_figure_sample_diagnosis": previous_payload,
    }
    provenance_out.parent.mkdir(parents=True, exist_ok=True)
    provenance_out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(sample_out)
    print(provenance_out)


def load_sample(sample_path: Path, provenance_path: Path) -> tuple[np.lib.npyio.NpzFile, dict, dict]:
    if not sample_path.exists():
        raise FileNotFoundError(
            f"missing representative sample {sample_path}; rerun with --select-from <centered-dataset-root>"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hash = provenance["selected_sample"]["centered_asset_sha256"]
    actual_hash = sha256_file(sample_path)
    if actual_hash != expected_hash:
        raise ValueError(f"representative sample hash mismatch: {actual_hash} != {expected_hash}")
    data = np.load(sample_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return data, metadata, provenance


def style_axis(ax) -> None:
    ax.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#606060")
    ax.spines["bottom"].set_color("#606060")
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="both", labelsize=8.2, width=0.65, length=3.0, pad=1.5)


def make_figure(
    geometry_root: Path,
    sample_path: Path,
    provenance_path: Path,
    out_path: Path,
    pdf_out_path: Path | None,
    mesh_cache_path: Path,
    max_triangles: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, metadata, provenance = load_sample(sample_path, provenance_path)
    freq = data["freq_ghz"]
    etheta = data["complex_etheta_real"] + 1j * data["complex_etheta_imag"]
    ephi = data["complex_ephi_real"] + 1j * data["complex_ephi_imag"]
    hrrp_vec_raw = np.asarray(data["hrrp_vector_mag"], dtype=np.float64)
    hrrp_denom = max(float(np.max(hrrp_vec_raw)), 1e-12)
    hrrp_theta = np.asarray(data["hrrp_theta_mag"], dtype=np.float64) / hrrp_denom
    hrrp_phi = np.asarray(data["hrrp_phi_mag"], dtype=np.float64) / hrrp_denom
    hrrp_vec = hrrp_vec_raw / hrrp_denom
    selected = provenance["selected_sample"]
    if metadata.get("job_id") != selected["sample_id"]:
        raise ValueError(f"sample metadata/provenance mismatch: {metadata.get('job_id')} != {selected['sample_id']}")
    if metadata.get("hrrp_version") != "centered_hann_ifft_v2":
        raise ValueError(f"{sample_path} is not a centered-HRRP v2 sample")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.20, 4.25), facecolor="white")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.105, top=0.925)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.56, 1.0], wspace=0.18)

    mesh_grid = outer[0, 0].subgridspec(4, 5, wspace=0.02, hspace=0.02)
    mesh_cache = load_mesh_cache(mesh_cache_path)
    for row, (_, objects, color) in enumerate(FOLDS):
        for col, obj in enumerate(objects):
            ax = fig.add_subplot(mesh_grid[row, col])
            stl_path = mesh_path(geometry_root, obj)
            draw_mesh(ax, stl_path, color, obj, mesh_cache.get(obj), max_triangles=max_triangles)

    fig.text(0.055, 0.965, "(a) Aircraft folds", fontsize=10.0, fontweight="semibold", ha="left", va="top")
    fold_centers = np.linspace(0.822, 0.208, len(FOLDS))
    for center, (fold_name, _, color) in zip(fold_centers, FOLDS, strict=True):
        fig.text(
            0.018,
            float(center),
            fold_name,
            fontsize=8.2,
            fontweight="semibold",
            color=color,
            rotation=90,
            ha="center",
            va="center",
        )

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.72)

    ax_grid = fig.add_subplot(right[0, 0])
    theta = np.arange(0, 181, 30)
    phi = np.arange(0, 360, 30)
    phis, thetas = np.meshgrid(phi, theta)
    nonpolar = (thetas.ravel() > 0) & (thetas.ravel() < 180)
    ax_grid.scatter(
        phis.ravel()[nonpolar],
        thetas.ravel()[nonpolar],
        s=14,
        color="#425466",
        linewidths=0,
        label="retained",
    )
    ax_grid.scatter(
        phis.ravel()[~nonpolar],
        thetas.ravel()[~nonpolar],
        s=14,
        facecolors="none",
        edgecolors="#A6ADB5",
        linewidths=0.65,
        label="excluded poles",
    )
    ax_grid.set_xlim(-10, 340)
    ax_grid.set_ylim(190, -10)
    ax_grid.set_xticks([0, 90, 180, 270])
    ax_grid.set_yticks([0, 60, 120, 180])
    ax_grid.set_xlabel(r"$\phi$ (deg)", fontsize=9.0, labelpad=1.0)
    ax_grid.set_ylabel(r"$\theta$ (deg)", fontsize=9.0, labelpad=1.0)
    ax_grid.set_title("(b) Look-direction grid", fontsize=9.5, fontweight="semibold", loc="left", pad=2.5)
    ax_grid.legend(loc="upper right", frameon=False, fontsize=6.6, handletextpad=0.3, borderpad=0.1)
    style_axis(ax_grid)

    ax_freq = fig.add_subplot(right[1, 0])
    vector_frequency_magnitude = np.sqrt(np.abs(etheta) ** 2 + np.abs(ephi) ** 2)
    frequency_denom = max(float(np.max(vector_frequency_magnitude)), 1e-12)
    ax_freq.plot(
        freq,
        np.abs(etheta) / frequency_denom,
        color="#3569a8",
        linewidth=1.35,
        label=r"$|E_\theta|$",
    )
    ax_freq.plot(
        freq,
        np.abs(ephi) / frequency_denom,
        color="#c4572f",
        linewidth=1.35,
        label=r"$|E_\phi|$",
    )
    ax_freq.set_xlim(0.10, 1.00)
    ax_freq.set_ylim(0.0, 1.05)
    ax_freq.set_xlabel("Frequency (GHz)", fontsize=9.0, labelpad=1.0)
    ax_freq.set_ylabel("Normalized magnitude", fontsize=9.0, labelpad=1.0)
    ax_freq.set_title("(c) Complex E-field", fontsize=9.5, fontweight="semibold", loc="left", pad=2.5)
    ax_freq.text(
        1.0,
        1.045,
        rf"{selected['object_code']}, $({selected['theta_inc_deg']:.0f}^\circ,\,"
        rf"{selected['phi_inc_deg']:.0f}^\circ)$",
        transform=ax_freq.transAxes,
        fontsize=7.9,
        color="#555555",
        ha="right",
        va="bottom",
    )
    ax_freq.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=8.2,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.35,
        borderpad=0.15,
        borderaxespad=0.2,
    )
    style_axis(ax_freq)

    ax_hrrp = fig.add_subplot(right[2, 0])
    if "range_m" not in data:
        raise ValueError(f"{sample_path} is not a centered-HRRP v2 sample")
    relative_range = np.asarray(data["range_m"], dtype=float)
    ax_hrrp.axvline(0.0, color="#a5a5a5", linewidth=0.55, linestyle=":", zorder=0)
    ax_hrrp.plot(relative_range, hrrp_vec, color="#287a78", linewidth=1.65, label=r"$|h|$")
    ax_hrrp.plot(
        relative_range,
        hrrp_theta,
        color="#3569a8",
        linewidth=1.15,
        linestyle="--",
        label=r"$|h_\theta|$",
    )
    ax_hrrp.plot(
        relative_range,
        hrrp_phi,
        color="#c4572f",
        linewidth=1.15,
        linestyle="-.",
        label=r"$|h_\phi|$",
    )
    # A fixed range window retains the representative target return while suppressing zero-padding silence.
    ax_hrrp.set_xlim(-1.5, 2.5)
    ax_hrrp.set_ylim(0.0, 1.05)
    ax_hrrp.set_xlabel("Relative range (m)", fontsize=9.0, labelpad=1.0)
    ax_hrrp.set_ylabel("Normalized magnitude", fontsize=9.0, labelpad=1.0)
    ax_hrrp.set_title("(d) Hann-IFFT HRRP", fontsize=9.5, fontweight="semibold", loc="left", pad=2.5)
    ax_hrrp.legend(
        loc="upper left",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=8.2,
        columnspacing=0.75,
        handlelength=1.45,
        handletextpad=0.3,
        borderpad=0.15,
        borderaxespad=0.2,
    )
    style_axis(ax_hrrp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=600,
        facecolor="white",
        metadata={
            "Software": "make_hrrp_dataset_overview_figure.py",
            "Description": f"GT-IP dataset overview; representative sample {selected['sample_id']}",
        },
    )
    if pdf_out_path is not None:
        pdf_out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_out_path,
            format="pdf",
            dpi=600,
            facecolor="white",
            metadata={
                "Title": "GT-IP dataset overview",
                "Subject": f"Representative centered complex E-field sample {selected['sample_id']}",
                "Creator": "make_hrrp_dataset_overview_figure.py",
                "Keywords": "GT-IP, complex E-field, HRRP, nested nonpolar protocol",
            },
        )
    rendered_artifacts = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "png": {
            "path": out_path.as_posix(),
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "pixel_dimensions": [4320, 2550],
            "dpi": 600,
        },
    }
    if pdf_out_path is not None:
        rendered_artifacts["pdf"] = {
            "path": pdf_out_path.as_posix(),
            "sha256": sha256_file(pdf_out_path),
            "bytes": pdf_out_path.stat().st_size,
            "page_size_in": [7.20, 4.25],
            "rendering": "vector text, axes, markers, and curves; aircraft surfaces rasterized at 600 dpi",
            "font_embedding": "CID TrueType (Matplotlib pdf.fonttype=42)",
        }
    provenance["rendered_artifacts"] = rendered_artifacts
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    data.close()
    print(out_path)
    if pdf_out_path is not None:
        print(pdf_out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path(r"paper\figs\data\fig3_meshes"),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(r"paper\figs\data\fig2_representative_sample.npz"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(r"paper\figs\data\fig2_representative_sample.json"),
    )
    parser.add_argument(
        "--select-from",
        type=Path,
        help="Current HRRP_centered_v2 root; recomputes and snapshots the robust representative sample.",
    )
    parser.add_argument(
        "--previous-sample",
        type=Path,
        default=Path(
            r"paper\figs\data\fig3_centered_samples\b7c1\pec\b7c1_pec_ti030_pi180_hrrp.npz"
        ),
        help="Former Fig. 2 input retained read-only for provenance diagnostics.",
    )
    parser.add_argument(
        "--protocol-splits-root",
        type=Path,
        default=Path(r"code\review_protocol\splits\tgrs_nested_nonpolar_v2"),
    )
    parser.add_argument("--out", type=Path, default=Path(r"paper\figs\fig2_dataset_overview.png"))
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=Path(r"paper\figs\fig2_dataset_overview.pdf"),
        help="Hybrid-vector PDF output (vector text/curves; 600-dpi rasterized aircraft surfaces).",
    )
    parser.add_argument("--mesh-cache", type=Path, default=Path(r"paper\figs\data\fig2_mesh_surface_cache.npz"))
    parser.add_argument(
        "--max-triangles",
        type=int,
        default=0,
        help="Maximum faces per aircraft; 0 keeps every STL triangle (paper default).",
    )
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args()

    if args.build_cache_only:
        build_mesh_cache(args.geometry_root, args.mesh_cache, max_triangles=args.max_triangles)
    else:
        if args.select_from is not None:
            select_representative_sample(
                args.select_from,
                args.sample,
                args.provenance,
                args.previous_sample,
                args.protocol_splits_root,
            )
        make_figure(
            args.geometry_root,
            args.sample,
            args.provenance,
            args.out,
            args.pdf_out,
            args.mesh_cache,
            max_triangles=args.max_triangles,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
