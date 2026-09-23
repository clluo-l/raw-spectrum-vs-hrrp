#!/usr/bin/env python3
"""Export an outcome-blind, redistributable complex-response review subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MANIFEST_SHA256 = "b4e33b6d333e853b7fa6ba5b18ac80bd30c932a21c475b6232695e4354f1f37d"
EXPECTED_INDEX_SHA256 = "ecf61b2be5f783eaedd218952878fde1ba45ac47313ca7b091d78c9f9d5d437c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--theta-deg", type=float, default=90.0)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = args.dataset_root / "dataset_manifest.json"
    index_path = args.dataset_root / "indexes" / "official" / "samples_index.jsonl"
    manifest_sha = sha256_file(manifest_path)
    index_sha = sha256_file(index_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256 or index_sha != EXPECTED_INDEX_SHA256:
        raise RuntimeError("repaired-corpus manifest/index hash mismatch")

    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        row
        for row in rows
        if abs(float(row["theta_inc_deg"]) - args.theta_deg) < 1e-6
        and abs(float(row["phi_inc_deg"]) - args.phi_deg) < 1e-6
    ]
    selected.sort(key=lambda row: str(row["object_code"]))
    object_codes = [str(row["object_code"]) for row in selected]
    if len(selected) != 20 or len(set(object_codes)) != 20:
        raise RuntimeError(f"fixed-direction subset must contain 20 unique objects, got {len(selected)}")

    fieldnames = [
        "object_code",
        "sample_id",
        "theta_deg",
        "phi_deg",
        "frequency_ghz",
        "etheta_real",
        "etheta_imag",
        "ephi_real",
        "ephi_imag",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    source_records: list[dict[str, Any]] = []
    exported_rows = 0
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            relative_path = Path(str(row["path"]).replace("\\", "/"))
            sample_path = args.dataset_root / relative_path
            with np.load(sample_path, allow_pickle=False) as sample:
                frequencies = np.asarray(sample["freq_ghz"], dtype=np.float64)
                etheta_real = np.asarray(sample["complex_etheta_real"], dtype=np.float64)
                etheta_imag = np.asarray(sample["complex_etheta_imag"], dtype=np.float64)
                ephi_real = np.asarray(sample["complex_ephi_real"], dtype=np.float64)
                ephi_imag = np.asarray(sample["complex_ephi_imag"], dtype=np.float64)
            arrays = [frequencies, etheta_real, etheta_imag, ephi_real, ephi_imag]
            if any(array.shape != (91,) for array in arrays):
                raise RuntimeError(f"unexpected sample shape: {relative_path}")
            sample_id = str(row.get("sample_id", relative_path.stem))
            for index in range(91):
                writer.writerow(
                    {
                        "object_code": row["object_code"],
                        "sample_id": sample_id,
                        "theta_deg": f"{args.theta_deg:.1f}",
                        "phi_deg": f"{args.phi_deg:.1f}",
                        "frequency_ghz": f"{frequencies[index]:.8g}",
                        "etheta_real": f"{etheta_real[index]:.9g}",
                        "etheta_imag": f"{etheta_imag[index]:.9g}",
                        "ephi_real": f"{ephi_real[index]:.9g}",
                        "ephi_imag": f"{ephi_imag[index]:.9g}",
                    }
                )
                exported_rows += 1
            source_records.append(
                {
                    "object_code": row["object_code"],
                    "sample_id": sample_id,
                    "source_relative_path": relative_path.as_posix(),
                    "source_npz_sha256": sha256_file(sample_path),
                }
            )
    temporary_csv.replace(args.output_csv)
    if exported_rows != 20 * 91:
        raise RuntimeError(f"unexpected exported row count: {exported_rows}")

    write_json(
        args.output_manifest,
        {
            "status": "ok",
            "selection_rule": "one fixed outcome-blind nonpolar direction (theta=90 deg, phi=0 deg) for every object",
            "dataset_manifest_sha256": manifest_sha,
            "official_index_sha256": index_sha,
            "object_count": 20,
            "frequency_samples_per_object": 91,
            "csv_row_count": exported_rows,
            "csv_path": args.output_csv.name,
            "csv_sha256": sha256_file(args.output_csv),
            "source_records": source_records,
            "scope": "review subset only; not an independent validation domain",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
