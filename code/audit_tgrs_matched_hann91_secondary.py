#!/usr/bin/env python3
"""Audit the 35-run Hann extension against the frozen strict rect-IFFT arm."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from audit_tgrs_matched_rect91_mlp import (
    EXPECTED_ARGS,
    EXPECTED_PARAM_COUNT,
    PROTOCOLS,
    aggregate_representation,
    audit_run_root as audit_strict_run_root,
    canonical_direction,
    load_split_keys,
    locate_prediction_csv,
    object_only_bootstrap,
    object_seed_cells,
    object_then_seed_bootstrap,
    prediction_key,
    read_csv,
    read_json,
    sha256_file,
    split_names,
    standardize_prediction_rows,
    validate_formal_protocol,
    values_equal,
    write_csv,
)


HANN_INPUT_KEY = "hrrp_hann91_complex_4ch"
RECT_INPUT_KEY = "hrrp_rect91_complex_4ch"
RUN_RE = re.compile(
    r"_FORMAL_matched_hann91_secondary_"
    r"(?P<protocol>id|outer[1-4])_seed(?P<seed>\d+)_sgpu(?P<gpu>\d+)$"
)
HANN_EXPECTED_ARGS = dict(EXPECTED_ARGS, hrrp_window="hann")
BOOKKEEPING_ARG_DIFFERENCES = {"out_dir", "run_name", "run_tag"}
SCIENTIFIC_ARG_DIFFERENCES = {"input_key", "hrrp_window"}


def expected_hann_tasks(seeds: Iterable[int]) -> set[tuple[str, int]]:
    return {(protocol, int(seed)) for protocol in PROTOCOLS for seed in seeds}


def one_factor_arg_differences(
    rect_args: dict[str, Any], hann_args: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    ignored = BOOKKEEPING_ARG_DIFFERENCES | SCIENTIFIC_ARG_DIFFERENCES
    return {
        field: {"rect_ifft": rect_args.get(field), "hann_ifft": hann_args.get(field)}
        for field in sorted(set(rect_args) | set(hann_args))
        if field not in ignored and rect_args.get(field) != hann_args.get(field)
    }


def validate_one_factor_pair(
    rect_args: dict[str, Any], hann_args: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    errors: dict[str, dict[str, Any]] = {}
    if rect_args.get("input_key") != RECT_INPUT_KEY or hann_args.get("input_key") != HANN_INPUT_KEY:
        errors["input_key"] = {
            "rect_ifft": rect_args.get("input_key"),
            "hann_ifft": hann_args.get("input_key"),
        }
    if rect_args.get("hrrp_window") != "rect" or hann_args.get("hrrp_window") != "hann":
        errors["hrrp_window"] = {
            "rect_ifft": rect_args.get("hrrp_window"),
            "hann_ifft": hann_args.get("hrrp_window"),
        }
    errors.update(one_factor_arg_differences(rect_args, hann_args))
    return errors


def audit_hann_run_root(
    run_root: Path,
    split_root: Path,
    seeds: list[int],
    lock_path: Path | None = None,
    expected_base_strict_lock_sha256: str | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    errors: list[str] = []
    expected = expected_hann_tasks(seeds)
    discovered: dict[tuple[str, int], list[Path]] = defaultdict(list)
    unrecognized_relevant: list[str] = []
    if not run_root.is_dir():
        raise FileNotFoundError(f"Hann run root does not exist: {run_root}")
    if not split_root.is_dir():
        raise FileNotFoundError(f"split root does not exist: {split_root}")

    lock: dict[str, Any] | None = None
    lock_sha256: str | None = None
    if lock_path is not None:
        lock = read_json(lock_path)
        lock_sha256 = sha256_file(lock_path)
        if lock.get("status") != "locked" or lock.get("protocol") != "tgrs_nested_nonpolar_v2":
            errors.append(f"{lock_path}: invalid lock status/protocol")
        if lock.get("representation", {}).get("input_key") != HANN_INPUT_KEY:
            errors.append(f"{lock_path}: invalid Hann input key")
        if lock.get("representation", {}).get("normalize") != "l2":
            errors.append(f"{lock_path}: invalid normalization")
        if lock.get("training", {}).get("protocols") != list(PROTOCOLS):
            errors.append(f"{lock_path}: invalid protocols")
        locked_seeds = [int(value) for value in lock.get("training", {}).get("seeds", [])]
        if not set(seeds).issubset(locked_seeds):
            errors.append(f"{lock_path}: requested seeds are outside the lock")
        if seeds == locked_seeds and int(lock.get("formal_matrix", {}).get("expected_runs", -1)) != len(expected):
            errors.append(f"{lock_path}: expected run count does not equal {len(expected)}")
        if int(lock.get("model_contract", {}).get("expected_trainable_parameters", -1)) != EXPECTED_PARAM_COUNT:
            errors.append(f"{lock_path}: invalid parameter count")
        for name, expected_hash in lock.get("expected_split_hashes", {}).items():
            path = split_root / name
            if not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"{lock_path}: canonical split hash changed for {name}")
        base_hash = str(lock.get("base_strict_matrix", {}).get("lock_sha256", ""))
        if expected_base_strict_lock_sha256 and base_hash != expected_base_strict_lock_sha256:
            errors.append(f"{lock_path}: base strict lock hash mismatch")

    for args_path in run_root.rglob("args.json"):
        run_dir = args_path.parent
        match = RUN_RE.search(run_dir.name)
        try:
            args = read_json(args_path)
        except Exception as exc:
            errors.append(f"{args_path}: cannot read args: {type(exc).__name__}: {exc}")
            continue
        if match is None:
            if args.get("input_key") == HANN_INPUT_KEY:
                unrecognized_relevant.append(str(run_dir))
            continue
        discovered[(match.group("protocol"), int(match.group("seed")))].append(run_dir)
    for path in unrecognized_relevant:
        errors.append(f"Hann input run violates the formal naming convention: {path}")
    duplicate = {key: paths for key, paths in discovered.items() if len(paths) != 1}
    for key, paths in duplicate.items():
        errors.append(f"duplicate Hann task {key}: {[str(path) for path in paths]}")
    observed = set(discovered)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"missing {len(missing)} Hann tasks: {missing}")
    if extra:
        errors.append(f"unexpected {len(extra)} Hann tasks: {extra}")

    args_by_task: dict[tuple[str, int], dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for protocol, seed in sorted(expected & observed):
        paths = discovered[(protocol, seed)]
        if len(paths) != 1:
            continue
        run_dir = paths[0]
        args_path = run_dir / "args.json"
        args = read_json(args_path)
        args_by_task[(protocol, seed)] = args
        error_count_before = len(errors)
        if args.get("input_key") != HANN_INPUT_KEY:
            errors.append(f"{run_dir}: invalid input_key {args.get('input_key')!r}")
        if int(args.get("seed", -1)) != seed:
            errors.append(f"{run_dir}: args seed does not match directory")
        for field, expected_value in HANN_EXPECTED_ARGS.items():
            if field not in args:
                errors.append(f"{run_dir}: args missing required field {field}")
            elif not values_equal(args[field], expected_value):
                errors.append(
                    f"{run_dir}: args {field}={args[field]!r} != expected {expected_value!r}"
                )
        train_name, dev_name, test_name = split_names(protocol)
        if Path(str(args.get("train_split", ""))).name != train_name:
            errors.append(f"{run_dir}: train_split is not {train_name}")
        if Path(str(args.get("val_split", ""))).name != dev_name:
            errors.append(f"{run_dir}: val_split is not {dev_name}")
        if args.get("run_name") and str(args["run_name"]) != run_dir.name:
            errors.append(f"{run_dir}: args run_name does not equal directory name")

        formal = validate_formal_protocol(
            run_dir / "formal_protocol.json",
            "hann_ifft",
            protocol,
            seed,
            split_root,
            errors,
            expected_lock_sha256=lock_sha256,
        )
        if formal is not None and expected_base_strict_lock_sha256:
            if formal.get("base_strict_lock_sha256") != expected_base_strict_lock_sha256:
                errors.append(f"{run_dir}: formal record has wrong base strict lock hash")
            if formal.get("single_scientific_factor") != "spectral_window_rect_to_hann":
                errors.append(f"{run_dir}: formal record lacks the one-factor declaration")

        summary_path = run_dir / "summary.json"
        summary: dict[str, Any] | None = None
        if not summary_path.is_file():
            errors.append(f"{run_dir}: missing summary.json")
        else:
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                errors.append(f"{summary_path}: invalid JSON: {type(exc).__name__}: {exc}")
        if summary is not None:
            if summary.get("status") != "ok" or int(summary.get("epochs_completed", -1)) != 250:
                errors.append(f"{run_dir}: training did not complete 250 epochs with status=ok")
            for field in ("stopped_by_time", "stopped_by_steps", "stopped_by_early_stop"):
                if bool(summary.get(field, False)):
                    errors.append(f"{run_dir}: {field} is true")
            if int(summary.get("param_count", -1)) != EXPECTED_PARAM_COUNT:
                errors.append(f"{run_dir}: parameter count mismatch")
            config = summary.get("config", {})
            if not isinstance(config, dict) or int(config.get("global_batch_size", -1)) != 48:
                errors.append(f"{run_dir}: summary global batch size is not 48")
            if int(summary.get("train_samples", -1)) != len(load_split_keys(split_root / train_name)):
                errors.append(f"{run_dir}: train sample count mismatch")
            if int(summary.get("val_samples", -1)) != len(load_split_keys(split_root / dev_name)):
                errors.append(f"{run_dir}: development sample count mismatch")

        prediction_path = locate_prediction_csv(run_dir)
        standardized: list[dict[str, Any]] = []
        if prediction_path is None:
            errors.append(f"{run_dir}: missing test prediction export")
        else:
            try:
                standardized = standardize_prediction_rows(
                    read_csv(prediction_path), "hann_ifft", protocol, seed, run_dir
                )
            except Exception as exc:
                errors.append(f"{prediction_path}: invalid predictions: {type(exc).__name__}: {exc}")
        if standardized:
            expected_keys = load_split_keys(split_root / test_name)
            observed_keys = {
                (
                    str(row["object_code"]),
                    *canonical_direction(row["theta_true_deg"], row["phi_true_deg"]),
                )
                for row in standardized
            }
            if len(observed_keys) != len(standardized) or observed_keys != expected_keys:
                errors.append(f"{run_dir}: prediction keys do not exactly equal {test_name}")
            sample_rows.extend(standardized)

        best_val = summary.get("best_val", {}) if summary else {}
        match = RUN_RE.search(run_dir.name)
        run_rows.append(
            {
                "representation": "hann_ifft",
                "protocol": protocol,
                "fold": int(protocol.removeprefix("outer")) if protocol.startswith("outer") else 0,
                "seed": seed,
                "gpu": int(match.group("gpu")) if match else -1,
                "run_dir": str(run_dir),
                "args_sha256": sha256_file(args_path),
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else "",
                "formal_protocol_sha256": sha256_file(run_dir / "formal_protocol.json")
                if (run_dir / "formal_protocol.json").is_file()
                else "",
                "prediction_sha256": sha256_file(prediction_path) if prediction_path else "",
                "param_count": int(summary.get("param_count", -1)) if summary else -1,
                "best_dev_epoch": int(best_val.get("epoch", -1)) if isinstance(best_val, dict) else -1,
                "best_dev_gce_deg": float(best_val.get("great_circle_mae_deg", math.nan))
                if isinstance(best_val, dict)
                else math.nan,
                "prediction_count": len(standardized),
                "run_status": "pass" if len(errors) == error_count_before else "fail",
            }
        )

    keyed = {prediction_key(row): row for row in sample_rows}
    if len(keyed) != len(sample_rows):
        errors.append("Hann prediction exports contain duplicate canonical keys")
    report = {
        "status": "pass" if not errors else "fail",
        "protocol": "tgrs_nested_nonpolar_v2",
        "run_root": str(run_root.resolve()),
        "expected_input_key": HANN_INPUT_KEY,
        "expected_model": "freq_mlp",
        "expected_param_count": EXPECTED_PARAM_COUNT,
        "expected_seeds": seeds,
        "expected_task_count": len(expected),
        "observed_task_count": len(observed),
        "audited_unique_task_count": len(run_rows),
        "sample_row_count": len(sample_rows),
        "missing_tasks": [list(item) for item in missing],
        "extra_tasks": [list(item) for item in extra],
        "duplicate_tasks": {
            "|".join(map(str, key)): [str(path) for path in paths]
            for key, paths in duplicate.items()
        },
        "experiment_lock": {
            "path": str(lock_path.resolve()) if lock_path else None,
            "sha256": lock_sha256,
            "lock_version": lock.get("lock_version") if lock else None,
        },
        "errors": errors,
        "runs": run_rows,
    }
    return report, run_rows, sample_rows, args_by_task


def aggregate_window_pair(
    rect_rows: list[dict[str, Any]],
    hann_rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rect = {prediction_key(row): row for row in rect_rows}
    hann = {prediction_key(row): row for row in hann_rows}
    if set(rect) != set(hann):
        raise RuntimeError("cannot aggregate unmatched rectangular/Hann prediction keys")
    paired_rows = []
    for key in sorted(rect):
        reference, extension = rect[key], hann[key]
        paired_rows.append(
            {
                "protocol": reference["protocol"],
                "fold": reference["fold"],
                "seed": reference["seed"],
                "object_code": reference["object_code"],
                "theta_true_deg": reference["theta_true_deg"],
                "phi_true_deg": reference["phi_true_deg"],
                "great_circle_err_rect_ifft_deg": reference["great_circle_err_deg"],
                "great_circle_err_hann_ifft_deg": extension["great_circle_err_deg"],
                "great_circle_err_hann_minus_rect_deg": extension["great_circle_err_deg"]
                - reference["great_circle_err_deg"],
                "theta_err_hann_minus_rect_deg": extension["theta_err_deg"]
                - reference["theta_err_deg"],
                "phi_err_hann_minus_rect_deg": extension["phi_err_deg"]
                - reference["phi_err_deg"],
                "top1_rect_ifft": reference["top1"],
                "top1_hann_ifft": extension["top1"],
                "top1_hann_minus_rect": extension["top1"] - reference["top1"],
            }
        )
    proxy_rows = [
        {
            "protocol": row["protocol"],
            "fold": row["fold"],
            "seed": row["seed"],
            "object_code": row["object_code"],
            "great_circle_err_deg": row["great_circle_err_hann_minus_rect_deg"],
            "theta_err_deg": row["theta_err_hann_minus_rect_deg"],
            "phi_err_deg": row["phi_err_hann_minus_rect_deg"],
            "top1": row["top1_hann_minus_rect"],
        }
        for row in paired_rows
    ]
    paired_cells = object_seed_cells(proxy_rows, "hann_minus_rect_ifft")
    result: dict[str, Any] = {}
    categories = {
        "id": {"id"},
        "held_geometry": {"outer1", "outer2", "outer3", "outer4"},
    }
    for category_index, (category, protocols) in enumerate(categories.items()):
        selected = [row for row in paired_rows if row["protocol"] in protocols]
        cells = [row for row in paired_cells if row["protocol"] in protocols]
        per_object = []
        for object_code in sorted({str(row["object_code"]) for row in cells}):
            group = [row for row in cells if row["object_code"] == object_code]
            per_object.append(
                {
                    "object_code": object_code,
                    "fold": int(group[0]["fold"]),
                    "mean_gce_hann_minus_rect_deg": float(
                        np.mean([row["great_circle_mae_deg"] for row in group])
                    ),
                    "mean_top1_hann_minus_rect": float(
                        np.mean([row["top1_acc"] for row in group])
                    ),
                }
            )
        run_rows = []
        for protocol in sorted(protocols):
            for seed in sorted({int(row["seed"]) for row in selected if row["protocol"] == protocol}):
                group = [
                    row for row in selected
                    if row["protocol"] == protocol and int(row["seed"]) == seed
                ]
                run_rows.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "mean_gce_hann_minus_rect_deg": float(
                            np.mean([row["great_circle_err_hann_minus_rect_deg"] for row in group])
                        ),
                        "mean_top1_hann_minus_rect": float(
                            np.mean([row["top1_hann_minus_rect"] for row in group])
                        ),
                    }
                )
        category_result: dict[str, Any] = {
            "paired_prediction_count": len(selected),
            "object_count": len(per_object),
            "run_count": len(run_rows),
            "object_centered_gce_hann_minus_rect_deg": float(
                np.mean([row["mean_gce_hann_minus_rect_deg"] for row in per_object])
            ),
            "object_centered_top1_hann_minus_rect": float(
                np.mean([row["mean_top1_hann_minus_rect"] for row in per_object])
            ),
            "hann_lower_error_runs": int(
                sum(row["mean_gce_hann_minus_rect_deg"] < 0.0 for row in run_rows)
            ),
            "object_centered_gce_difference_95ci": object_only_bootstrap(
                per_object,
                "mean_gce_hann_minus_rect_deg",
                bootstrap_iterations,
                bootstrap_seed + category_index * 20,
            ),
            "object_centered_top1_difference_95ci": object_only_bootstrap(
                per_object,
                "mean_top1_hann_minus_rect",
                bootstrap_iterations,
                bootstrap_seed + category_index * 20 + 1,
            ),
            "object_seed_hierarchical_gce_difference_95ci": object_then_seed_bootstrap(
                cells,
                "great_circle_mae_deg",
                bootstrap_iterations,
                bootstrap_seed + category_index * 20 + 2,
            ),
            "object_seed_hierarchical_top1_difference_95ci": object_then_seed_bootstrap(
                cells,
                "top1_acc",
                bootstrap_iterations,
                bootstrap_seed + category_index * 20 + 3,
            ),
            "per_object": per_object,
            "per_run": run_rows,
        }
        if category == "held_geometry":
            category_result["per_fold"] = []
            for fold in range(1, 5):
                group = [row for row in selected if int(row["fold"]) == fold]
                category_result["per_fold"].append(
                    {
                        "fold": fold,
                        "mean_gce_hann_minus_rect_deg": float(
                            np.mean([row["great_circle_err_hann_minus_rect_deg"] for row in group])
                        ),
                        "mean_top1_hann_minus_rect": float(
                            np.mean([row["top1_hann_minus_rect"] for row in group])
                        ),
                    }
                )
        result[category] = category_result
    return result, paired_rows, paired_cells


def audit_extension(
    strict_run_root: Path,
    hann_run_root: Path,
    split_root: Path,
    seeds: list[int],
    strict_lock_path: Path,
    hann_lock_path: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strict_lock_sha256 = sha256_file(strict_lock_path)
    strict_audit, strict_runs, strict_samples = audit_strict_run_root(
        strict_run_root, split_root, seeds, lock_path=strict_lock_path
    )
    hann_audit, hann_runs, hann_samples, hann_args = audit_hann_run_root(
        hann_run_root,
        split_root,
        seeds,
        lock_path=hann_lock_path,
        expected_base_strict_lock_sha256=strict_lock_sha256,
    )
    errors: list[str] = []
    if strict_audit["status"] != "pass":
        errors.append("base strict 70-run matrix audit failed")
    if hann_audit["status"] != "pass":
        errors.append("Hann 35-run extension audit failed")

    rect_args: dict[tuple[str, int], dict[str, Any]] = {}
    for row in strict_runs:
        if row["representation"] == "rect_ifft":
            path = Path(row["run_dir"]) / "args.json"
            rect_args[(str(row["protocol"]), int(row["seed"]))] = read_json(path)
    for key in sorted(expected_hann_tasks(seeds)):
        if key not in rect_args or key not in hann_args:
            continue
        differences = validate_one_factor_pair(rect_args[key], hann_args[key])
        if differences:
            errors.append(
                f"rect/Hann args violate one-factor contract for {key}: "
                f"{json.dumps(differences, sort_keys=True)}"
            )

    rect_samples = [row for row in strict_samples if row["representation"] == "rect_ifft"]
    rect_keys = {prediction_key(row) for row in rect_samples}
    hann_keys = {prediction_key(row) for row in hann_samples}
    if rect_keys != hann_keys:
        errors.append(
            "rectangular and Hann predictions do not share identical keys; "
            f"rect_only={len(rect_keys - hann_keys)}, hann_only={len(hann_keys - rect_keys)}"
        )

    aggregate_summary: dict[str, Any] = {}
    object_cells: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    if not errors:
        rect_summary, rect_cells = aggregate_representation(
            rect_samples, bootstrap_iterations, bootstrap_seed
        )
        hann_summary, hann_cells = aggregate_representation(
            hann_samples, bootstrap_iterations, bootstrap_seed + 100
        )
        paired_summary, paired_rows, paired_cells = aggregate_window_pair(
            rect_samples,
            hann_samples,
            bootstrap_iterations,
            bootstrap_seed + 10_000,
        )
        object_cells.extend(rect_cells)
        object_cells.extend(hann_cells)
        object_cells.extend(paired_cells)
        aggregate_summary = {
            "status": "ok",
            "protocol": "tgrs_nested_nonpolar_v2",
            "definitions": {
                "only_scientific_factor": "spectral window: rectangular versus Hann",
                "paired_difference": "Hann-IFFT minus rectangular-IFFT; positive GCE means Hann is worse",
                "object_centered": "average fixed directions within object-seed, then seeds within object, then 20 objects equally",
                "primary_uncertainty": "percentile bootstrap over 20 centered object effects",
            },
            "models": {"rect_ifft": rect_summary, "hann_ifft": hann_summary},
            "paired_difference": paired_summary,
        }
    report = {
        "status": "pass" if not errors else "fail",
        "protocol": "tgrs_nested_nonpolar_v2",
        "strict_lock_sha256": strict_lock_sha256,
        "hann_lock_sha256": sha256_file(hann_lock_path),
        "strict_matrix": strict_audit,
        "hann_extension": hann_audit,
        "single_factor_contract": {
            "scientific_difference": {
                "input_key": [RECT_INPUT_KEY, HANN_INPUT_KEY],
                "hrrp_window": ["rect", "hann"],
            },
            "bookkeeping_differences_ignored": sorted(BOOKKEEPING_ARG_DIFFERENCES),
            "all_other_args_required_identical": True,
        },
        "paired_key_count": len(rect_keys & hann_keys),
        "aggregate_summary": aggregate_summary,
        "errors": errors,
    }
    return report, strict_runs + hann_runs, strict_samples + hann_samples, object_cells + paired_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-run-root", type=Path, required=True)
    parser.add_argument("--hann-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-lock", type=Path)
    parser.add_argument("--hann-lock", type=Path)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--seeds", default="70,71,72,73,74,75,76")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    strict_lock = args.strict_lock or code_root / "review_protocol/matched_rect91_global_mlp_lock.json"
    hann_lock = args.hann_lock or code_root / "review_protocol/matched_hann91_secondary_lock.json"
    split_root = args.split_root or code_root / "review_protocol/splits/tgrs_nested_nonpolar_v2"
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    report, run_rows, sample_rows, derived_rows = audit_extension(
        args.strict_run_root,
        args.hann_run_root,
        split_root,
        seeds,
        strict_lock,
        hann_lock,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "derived_object_and_paired_rows.csv", derived_rows)
    if report["status"] == "pass":
        # Recompute the compact summary only after all formal gates pass.
        rect_samples = [row for row in sample_rows if row["representation"] == "rect_ifft"]
        hann_samples = [row for row in sample_rows if row["representation"] == "hann_ifft"]
        rect_summary, _ = aggregate_representation(
            rect_samples, args.bootstrap_iterations, args.bootstrap_seed
        )
        hann_summary, _ = aggregate_representation(
            hann_samples, args.bootstrap_iterations, args.bootstrap_seed + 100
        )
        paired_summary, paired_rows, _ = aggregate_window_pair(
            rect_samples,
            hann_samples,
            args.bootstrap_iterations,
            args.bootstrap_seed + 10_000,
        )
        summary = {
            "status": "ok",
            "definitions": {
                "paired_difference": "Hann-IFFT minus rectangular-IFFT; positive GCE means Hann is worse",
                "only_scientific_factor": "spectral window",
            },
            "models": {"rect_ifft": rect_summary, "hann_ifft": hann_summary},
            "paired_difference": paired_summary,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_csv(args.output_dir / "paired_samples.csv", paired_rows)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
