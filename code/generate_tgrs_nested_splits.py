#!/usr/bin/env python
"""Generate immutable nonpolar ID and nested held-aircraft review splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


OOD_FOLDS = {
    1: ("b871", "bb7d", "b827", "b905", "bbc6"),
    2: ("b80b", "ba0f", "b7c1", "b9e6", "bb7c"),
    3: ("b943", "b97b", "b812", "bc2c", "b974"),
    4: ("bb26", "b7fd", "baa9", "b979", "b8ed"),
}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f"{row['path'].replace(chr(92), '/')}\n" for row in rows), encoding="utf-8")


def _direction_role(theta_deg: float, phi_deg: float, object_offset: int) -> str:
    theta_values = (30.0, 60.0, 90.0, 120.0, 150.0)
    phi_values = tuple(float(value) for value in range(0, 360, 30))
    theta_idx = theta_values.index(float(theta_deg))
    phi_idx = phi_values.index(float(phi_deg))
    shift = int(object_offset) % 12
    test = {(2 * theta_idx + shift) % 12, (2 * theta_idx + 6 + shift) % 12}
    dev = {(2 * theta_idx + 3 + shift) % 12, (2 * theta_idx + 9 + shift) % 12}
    if phi_idx in test:
        return "test"
    if phi_idx in dev:
        return "dev"
    return "train"


def _counts(rows: list[dict]) -> dict:
    by_object = Counter(str(row["object_code"]) for row in rows)
    by_theta = Counter(str(int(round(float(row["theta_inc_deg"])))) for row in rows)
    by_direction = Counter(
        f"{int(round(float(row['theta_inc_deg'])))}_{int(round(float(row['phi_inc_deg'])))}" for row in rows
    )
    return {
        "samples": len(rows),
        "objects": len(by_object),
        "samples_per_object_unique": sorted(set(by_object.values())),
        "by_object": dict(sorted(by_object.items())),
        "by_theta": dict(sorted(by_theta.items(), key=lambda item: int(item[0]))),
        "direction_coverage": {
            "unique_directions": len(by_direction),
            "samples_per_direction_min": min(by_direction.values()) if by_direction else 0,
            "samples_per_direction_max": max(by_direction.values()) if by_direction else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    index_path = args.dataset_root / "indexes" / "official" / "samples_index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if float(row["theta_inc_deg"]) not in (0.0, 180.0)]
    object_values = sorted({str(row["object_code"]) for row in rows})
    object_offsets = {value: idx for idx, value in enumerate(object_values)}
    for row in rows:
        row["direction_role"] = _direction_role(
            float(row["theta_inc_deg"]),
            float(row["phi_inc_deg"]),
            object_offsets[str(row["object_code"])],
        )
    rows.sort(key=lambda row: (str(row["object_code"]), float(row["theta_inc_deg"]), float(row["phi_inc_deg"])))

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "ok",
        "version": "tgrs_nested_nonpolar_v2",
        "source_index": str(index_path),
        "excluded_theta_deg": [0.0, 180.0],
        "directions_per_object": 60,
        "direction_partition_rule": {
            "object_specific_shift": "s = lexicographic object index mod 12",
            "test_phi_indices_per_theta": ["(2*t+s) mod 12", "(2*t+6+s) mod 12"],
            "dev_phi_indices_per_theta": ["(2*t+3+s) mod 12", "(2*t+9+s) mod 12"],
            "id_counts_per_object": {"train": 40, "dev": 10, "test": 10},
            "purpose": "sample-disjoint per object while retaining every direction in other training objects",
        },
        "splits": {},
        "outer_folds": {},
    }

    for role in ("train", "dev", "test"):
        subset = [row for row in rows if row["direction_role"] == role]
        name = f"id_{role}.txt"
        _write(args.output_root / name, subset)
        manifest["splits"][name] = _counts(subset)

    all_objects = sorted({str(row["object_code"]) for row in rows})
    for fold, held_tuple in OOD_FOLDS.items():
        held = set(held_tuple)
        if not held <= set(all_objects):
            raise ValueError(f"fold {fold} has unknown held objects: {sorted(held - set(all_objects))}")
        nonheld = set(all_objects) - held
        fold_train = [row for row in rows if row["object_code"] in nonheld and row["direction_role"] != "dev"]
        fold_dev = [row for row in rows if row["object_code"] in nonheld and row["direction_role"] == "dev"]
        fold_test = [row for row in rows if row["object_code"] in held]
        split_rows = {"train": fold_train, "dev": fold_dev, "test": fold_test}
        fold_record = {"held_objects": list(held_tuple), "splits": {}}
        path_sets = {}
        for role, subset in split_rows.items():
            name = f"outer{fold}_{role}.txt"
            _write(args.output_root / name, subset)
            fold_record["splits"][name] = _counts(subset)
            path_sets[role] = {row["path"].replace("\\", "/") for row in subset}
        fold_record["pairwise_overlap_counts"] = {
            "train_dev": len(path_sets["train"] & path_sets["dev"]),
            "train_test": len(path_sets["train"] & path_sets["test"]),
            "dev_test": len(path_sets["dev"] & path_sets["test"]),
        }
        manifest["outer_folds"][str(fold)] = fold_record

    total_outer_test = sum(record["splits"][f"outer{fold}_test.txt"]["samples"] for fold, record in ((int(key), value) for key, value in manifest["outer_folds"].items()))
    manifest["pooled_outer_test_samples_per_seed"] = total_outer_test
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
