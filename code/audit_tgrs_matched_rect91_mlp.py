#!/usr/bin/env python3
"""Audit and aggregate the strict Frequency/rect-IFFT global-MLP matrix.

The expected experiment is the 70-run ``tgrs_nested_nonpolar_v2`` matrix:
two four-channel, length-91, sample-L2-normalized representations, five
protocols (ID plus four object-disjoint outer folds), and seeds 70--76.  Both
representations must use the identical ``freq_mlp`` code path.  This script is
intended to run after a remote run root has been synchronized locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPRESENTATIONS = {
    "frequency": "freq_complex_4ch",
    "rect_ifft": "hrrp_rect91_complex_4ch",
}
PROTOCOLS = ("id", "outer1", "outer2", "outer3", "outer4")
RUN_RE = re.compile(
    r"_FORMAL_matched_rect91_(?P<representation>frequency|rect_ifft)_"
    r"(?P<protocol>id|outer[1-4])_seed(?P<seed>\d+)_sgpu(?P<gpu>\d+)$"
)
SAMPLE_RE = re.compile(
    r"(?P<object>[A-Za-z0-9]+)_pec_ti(?P<theta>\d{3})_pi(?P<phi>\d{3})_hrrp\.npz$"
)
EXPECTED_PARAM_COUNT = 3_022_993

EXPECTED_ARGS: dict[str, Any] = {
    "model": "freq_mlp",
    "object_code": "all",
    "normalize": "l2",
    "hrrp_window": "rect",
    "hrrp_nfft": 91,
    "hrrp_gate": "full",
    "hrrp_range_align": "none",
    "hrrp_phase_reference": "none",
    "bandwidth_min_ghz": 0.0,
    "bandwidth_max_ghz": 0.0,
    "snr_db": None,
    "train_random_global_phase_max_deg": 0.0,
    "train_random_range_offset_max_m": 0.0,
    "train_random_channel_gain_max_db": 0.0,
    "train_random_channel_phase_max_deg": 0.0,
    "eval_global_phase_deg": 0.0,
    "eval_range_offset_m": 0.0,
    "eval_common_gain_db": 0.0,
    "eval_channel_gain_db": 0.0,
    "eval_channel_phase_deg": 0.0,
    "eval_receive_basis_rotation_deg": 0.0,
    "eval_random_global_phase_max_deg": 0.0,
    "eval_random_range_offset_max_m": 0.0,
    "eval_random_common_gain_max_db": 0.0,
    "eval_random_channel_gain_max_db": 0.0,
    "eval_random_channel_phase_max_deg": 0.0,
    "eval_random_receive_basis_rotation_max_deg": 0.0,
    "eval_zero_etheta": False,
    "eval_zero_ephi": False,
    "train_limit": 0,
    "val_limit": 0,
    "epochs": 250,
    "batch_size": 48,
    "num_workers": 2,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine",
    "min_lr": 1e-6,
    "early_stop_patience": 0,
    "grad_clip_norm": 1.0,
    "hrrp_base_channels": 96,
    "hrrp_transformer_layers": 3,
    "sequence_layers": 3,
    "transolver3_heads": 8,
    "transolver3_slices": 32,
    "hrrp_feature_dim": 384,
    "head_hidden": 384,
    "dropout": 0.1,
    "classifier_head": "factorized",
    "phi_loss_weight": 1.0,
    "phi_loss_mode": "circular_soft",
    "phi_soft_sigma_classes": 0.5,
    "label_smoothing": 0.0,
    "spherical_risk_weight": 0.0,
    "object_conditioning": "none",
    "max_seconds": 0.0,
    "max_train_steps": 0,
    "eval_every": 5,
}

PAIRWISE_IGNORED_ARGS = {"input_key", "run_name", "run_tag"}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def values_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return observed == expected
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return bool(np.isclose(float(observed), float(expected), rtol=1e-10, atol=1e-12))
    return observed == expected


def split_names(protocol: str) -> tuple[str, str, str]:
    if protocol == "id":
        return "id_train.txt", "id_dev.txt", "id_test.txt"
    return f"{protocol}_train.txt", f"{protocol}_dev.txt", f"{protocol}_test.txt"


def canonical_direction(theta: Any, phi: Any) -> tuple[float, float]:
    theta_value = round(float(theta), 3)
    phi_value = round(float(phi) % 360.0, 3)
    if phi_value == 360.0:
        phi_value = 0.0
    return theta_value, phi_value


def split_sample_key(relative_path: str) -> tuple[str, float, float]:
    match = SAMPLE_RE.search(relative_path.replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot parse object/direction from split entry: {relative_path}")
    return (
        match.group("object"),
        float(int(match.group("theta"))),
        float(int(match.group("phi")) % 360),
    )


def load_split_keys(path: Path) -> set[tuple[str, float, float]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    keys = {split_sample_key(line) for line in lines}
    if len(keys) != len(lines):
        raise RuntimeError(f"split contains duplicate object/direction keys: {path}")
    return keys


def truth(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(float(value) >= 0.5)
    return float(str(value).strip().lower() in {"1", "true", "yes", "y"})


def locate_prediction_csv(run_dir: Path) -> Path | None:
    candidates = (
        run_dir / "test_eval" / "samples.csv",
        run_dir / "prediction_eval" / "samples.csv",
        run_dir / "test_eval" / "predictions.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def standardize_prediction_rows(
    source_rows: list[dict[str, str]],
    representation: str,
    protocol: str,
    seed: int,
    run_dir: Path,
) -> list[dict[str, Any]]:
    result = []
    fold = int(protocol.removeprefix("outer")) if protocol.startswith("outer") else 0
    for source in source_rows:
        object_code = str(source.get("object_code", "")).strip()
        if not object_code:
            raise KeyError(f"{run_dir}: prediction row has no object_code")
        theta_true = float(source["theta_true_deg"])
        phi_true = float(source["phi_true_deg"])
        theta_pred = float(source["theta_pred_deg"])
        phi_pred = float(source["phi_pred_deg"])
        theta_err = float(source.get("theta_err_deg") or abs(theta_pred - theta_true))
        phi_delta = abs(phi_pred - phi_true) % 360.0
        phi_err = float(source.get("phi_err_deg") or min(phi_delta, 360.0 - phi_delta))
        great_circle = float(source["great_circle_err_deg"])
        top1 = truth(source["top1"])
        numeric = (theta_true, phi_true, theta_pred, phi_pred, theta_err, phi_err, great_circle, top1)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{run_dir}: non-finite prediction metric in row {source}")
        result.append(
            {
                "representation": representation,
                "protocol": protocol,
                "fold": fold,
                "seed": seed,
                "object_code": object_code,
                "theta_true_deg": theta_true,
                "phi_true_deg": phi_true,
                "theta_pred_deg": theta_pred,
                "phi_pred_deg": phi_pred,
                "theta_err_deg": theta_err,
                "phi_err_deg": phi_err,
                "great_circle_err_deg": great_circle,
                "top1": top1,
                "run_dir": str(run_dir),
            }
        )
    return result


def prediction_key(row: dict[str, Any]) -> tuple[Any, ...]:
    theta, phi = canonical_direction(row["theta_true_deg"], row["phi_true_deg"])
    return (
        str(row["protocol"]),
        int(row["seed"]),
        str(row["object_code"]),
        theta,
        phi,
    )


def expected_task_set(seeds: Iterable[int]) -> set[tuple[str, str, int]]:
    return {
        (representation, protocol, int(seed))
        for representation in REPRESENTATIONS
        for protocol in PROTOCOLS
        for seed in seeds
    }


def normalize_for_pairwise_args(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if key not in PAIRWISE_IGNORED_ARGS}


def validate_formal_protocol(
    path: Path,
    representation: str,
    protocol: str,
    seed: int,
    split_root: Path,
    errors: list[str],
    expected_lock_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"{path.parent}: missing formal_protocol.json")
        return None
    try:
        record = read_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot read formal protocol: {type(exc).__name__}: {exc}")
        return None
    task = record.get("task")
    if not isinstance(task, dict):
        errors.append(f"{path}: missing task object")
    else:
        if str(task.get("protocol")) != protocol or int(task.get("seed", -1)) != seed:
            errors.append(f"{path}: task protocol/seed does not match directory")
        method = str(task.get("method", ""))
        declared_representation = str(task.get("representation", record.get("representation", "")))
        if representation not in method and declared_representation != representation:
            errors.append(
                f"{path}: task does not identify representation {representation!r}: {task}"
            )
    if record.get("evaluation_role") != "untouched-test":
        errors.append(f"{path}: evaluation_role is not untouched-test")
    if record.get("status") != "complete":
        errors.append(f"{path}: formal task status is not complete")
    if not bool(record.get("test_manifest_opened")):
        errors.append(f"{path}: test_manifest_opened is not true")
    if not bool(record.get("test_manifest_opened_after_checkpoint")):
        errors.append(f"{path}: test was not recorded as opened after checkpoint creation")
    if int(record.get("epochs", -1)) != 250:
        errors.append(f"{path}: formal protocol epochs is not 250")
    if expected_lock_sha256 is not None:
        observed_lock_hash = str(record.get("experiment_lock_sha256", ""))
        if observed_lock_hash != expected_lock_sha256:
            errors.append(
                f"{path}: experiment lock hash {observed_lock_hash!r} != {expected_lock_sha256}"
            )

    checkpoint = path.parent / "model_best.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        errors.append(f"{path.parent}: missing/nonpositive model_best.pt")
    else:
        observed_checkpoint_hash = sha256_file(checkpoint)
        if str(record.get("checkpoint_sha256", "")) != observed_checkpoint_hash:
            errors.append(f"{path}: checkpoint SHA-256 does not match synchronized model_best.pt")
    prediction_path = locate_prediction_csv(path.parent)
    if prediction_path is not None:
        observed_prediction_hash = sha256_file(prediction_path)
        if str(record.get("prediction_samples_sha256", "")) != observed_prediction_hash:
            errors.append(f"{path}: prediction SHA-256 does not match synchronized samples.csv")

    train_name, dev_name, test_name = split_names(protocol)
    for role, expected_name in (("train_split", train_name), ("dev_split", dev_name), ("test_split", test_name)):
        canonical_path = split_root / expected_name
        declared_path = Path(str(record.get(role, "")))
        declared_hash = str(record.get(f"{role}_sha256", ""))
        if declared_path.name != expected_name:
            errors.append(f"{path}: {role} basename {declared_path.name!r} != {expected_name!r}")
        if not canonical_path.is_file():
            errors.append(f"canonical split is missing: {canonical_path}")
            continue
        canonical_hash = sha256_file(canonical_path)
        if declared_hash != canonical_hash:
            errors.append(
                f"{path}: {role} hash {declared_hash!r} != canonical SHA-256 {canonical_hash}"
            )
    return record


def audit_run_root(
    run_root: Path,
    split_root: Path,
    seeds: list[int],
    lock_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected = expected_task_set(seeds)
    discovered: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    unrecognized_relevant: list[str] = []

    if not run_root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {run_root}")
    if not split_root.is_dir():
        raise FileNotFoundError(f"split root does not exist: {split_root}")

    lock_record: dict[str, Any] | None = None
    lock_sha256: str | None = None
    if lock_path is not None:
        if not lock_path.is_file():
            raise FileNotFoundError(f"experiment lock does not exist: {lock_path}")
        lock_record = read_json(lock_path)
        lock_sha256 = sha256_file(lock_path)
        if lock_record.get("status") != "locked":
            errors.append(f"{lock_path}: status is not locked")
        if lock_record.get("protocol") != "tgrs_nested_nonpolar_v2":
            errors.append(f"{lock_path}: unexpected protocol {lock_record.get('protocol')!r}")
        locked_seeds = [int(value) for value in lock_record.get("training", {}).get("seeds", [])]
        if not set(seeds).issubset(locked_seeds):
            errors.append(f"{lock_path}: requested seeds are not a subset of the locked seeds")
        if seeds == locked_seeds and int(lock_record.get("formal_matrix", {}).get("expected_runs", -1)) != len(expected):
            errors.append(f"{lock_path}: expected run count does not match the full requested matrix")
        if lock_record.get("training", {}).get("protocols") != list(PROTOCOLS):
            errors.append(f"{lock_path}: locked protocols do not match auditor protocols")
        if int(lock_record.get("model_contract", {}).get("expected_trainable_parameters", -1)) != EXPECTED_PARAM_COUNT:
            errors.append(f"{lock_path}: locked parameter count is not {EXPECTED_PARAM_COUNT}")
        for representation, input_key in REPRESENTATIONS.items():
            locked = lock_record.get("representations", {}).get(representation, {})
            if locked.get("model") != "freq_mlp" or locked.get("input_key") != input_key or locked.get("normalize") != "l2":
                errors.append(f"{lock_path}: invalid locked representation contract for {representation}")
        for name, expected_hash in lock_record.get("expected_split_hashes", {}).items():
            canonical_path = split_root / name
            if not canonical_path.is_file():
                errors.append(f"{lock_path}: locked split is missing locally: {canonical_path}")
            elif sha256_file(canonical_path) != expected_hash:
                errors.append(f"{lock_path}: canonical split hash changed for {name}")

    for args_path in run_root.rglob("args.json"):
        run_dir = args_path.parent
        match = RUN_RE.search(run_dir.name)
        try:
            args = read_json(args_path)
        except Exception as exc:
            errors.append(f"{args_path}: cannot read args: {type(exc).__name__}: {exc}")
            continue
        if match is None:
            if args.get("input_key") in REPRESENTATIONS.values():
                unrecognized_relevant.append(str(run_dir))
            continue
        key = (
            match.group("representation"),
            match.group("protocol"),
            int(match.group("seed")),
        )
        discovered[key].append(run_dir)

    for run_dir in unrecognized_relevant:
        errors.append(f"matched input run does not follow the formal naming convention: {run_dir}")
    duplicate_tasks = {key: paths for key, paths in discovered.items() if len(paths) != 1}
    for key, paths in duplicate_tasks.items():
        errors.append(f"duplicate task {key}: {[str(path) for path in paths]}")
    observed = set(discovered)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"missing {len(missing)} formal tasks: {missing}")
    if extra:
        errors.append(f"unexpected {len(extra)} formal tasks: {extra}")

    run_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    args_by_task: dict[tuple[str, str, int], dict[str, Any]] = {}
    split_hashes = {
        path.name: sha256_file(path)
        for path in split_root.glob("*.txt")
        if path.is_file()
    }

    for task in sorted(expected & observed):
        representation, protocol, seed = task
        paths = discovered[task]
        if len(paths) != 1:
            continue
        run_dir = paths[0]
        args_path = run_dir / "args.json"
        args = read_json(args_path)
        args_by_task[task] = args
        run_errors_before = len(errors)
        expected_input_key = REPRESENTATIONS[representation]
        if args.get("input_key") != expected_input_key:
            errors.append(
                f"{run_dir}: input_key {args.get('input_key')!r} != {expected_input_key!r}"
            )
        if int(args.get("seed", -1)) != seed:
            errors.append(f"{run_dir}: args seed {args.get('seed')} != directory seed {seed}")
        for field, expected_value in EXPECTED_ARGS.items():
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
            representation,
            protocol,
            seed,
            split_root,
            errors,
            expected_lock_sha256=lock_sha256,
        )

        summary_path = run_dir / "summary.json"
        summary: dict[str, Any] | None = None
        if not summary_path.is_file():
            errors.append(f"{run_dir}: missing summary.json")
        else:
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                errors.append(f"{summary_path}: cannot read summary: {type(exc).__name__}: {exc}")
        if summary is not None:
            if summary.get("status") != "ok":
                errors.append(f"{run_dir}: training status is not ok")
            if int(summary.get("epochs_completed", -1)) != 250:
                errors.append(f"{run_dir}: epochs_completed is not 250")
            for stopped_field in ("stopped_by_time", "stopped_by_steps", "stopped_by_early_stop"):
                if bool(summary.get(stopped_field, False)):
                    errors.append(f"{run_dir}: {stopped_field} is true")
            if int(summary.get("param_count", -1)) != EXPECTED_PARAM_COUNT:
                errors.append(
                    f"{run_dir}: param_count {summary.get('param_count')} != {EXPECTED_PARAM_COUNT}"
                )
            config = summary.get("config", {})
            if not isinstance(config, dict):
                errors.append(f"{run_dir}: summary config is not an object")
                config = {}
            if int(config.get("global_batch_size", -1)) != 48:
                errors.append(f"{run_dir}: summary global_batch_size is not 48")
            expected_train_count = len(load_split_keys(split_root / train_name))
            expected_dev_count = len(load_split_keys(split_root / dev_name))
            if int(summary.get("train_samples", -1)) != expected_train_count:
                errors.append(
                    f"{run_dir}: train_samples {summary.get('train_samples')} != {expected_train_count}"
                )
            if int(summary.get("val_samples", -1)) != expected_dev_count:
                errors.append(
                    f"{run_dir}: val_samples {summary.get('val_samples')} != {expected_dev_count}"
                )

        prediction_path = locate_prediction_csv(run_dir)
        standardized: list[dict[str, Any]] = []
        if prediction_path is None:
            errors.append(f"{run_dir}: missing test prediction samples.csv")
        else:
            try:
                standardized = standardize_prediction_rows(
                    read_csv(prediction_path), representation, protocol, seed, run_dir
                )
            except Exception as exc:
                errors.append(
                    f"{prediction_path}: invalid prediction export: {type(exc).__name__}: {exc}"
                )
                standardized = []
        if standardized:
            expected_keys = load_split_keys(split_root / test_name)
            observed_keys = {
                (
                    str(row["object_code"]),
                    *canonical_direction(row["theta_true_deg"], row["phi_true_deg"]),
                )
                for row in standardized
            }
            if len(observed_keys) != len(standardized):
                errors.append(f"{run_dir}: duplicate object/direction keys in prediction export")
            if observed_keys != expected_keys:
                missing_keys = sorted(expected_keys - observed_keys)
                extra_keys = sorted(observed_keys - expected_keys)
                errors.append(
                    f"{run_dir}: prediction keys differ from {test_name}; "
                    f"missing={missing_keys[:10]}, extra={extra_keys[:10]}"
                )
            sample_rows.extend(standardized)

        best_val = summary.get("best_val", {}) if summary else {}
        run_rows.append(
            {
                "representation": representation,
                "protocol": protocol,
                "fold": int(protocol.removeprefix("outer")) if protocol.startswith("outer") else 0,
                "seed": seed,
                "gpu": int(RUN_RE.search(run_dir.name).group("gpu")),  # type: ignore[union-attr]
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
                "run_status": "pass" if len(errors) == run_errors_before else "fail",
                "formal_protocol_present": formal is not None,
            }
        )

    for protocol in PROTOCOLS:
        for seed in seeds:
            left_key = ("frequency", protocol, seed)
            right_key = ("rect_ifft", protocol, seed)
            if left_key not in args_by_task or right_key not in args_by_task:
                continue
            left = normalize_for_pairwise_args(args_by_task[left_key])
            right = normalize_for_pairwise_args(args_by_task[right_key])
            if left != right:
                differing = {
                    field: {"frequency": left.get(field), "rect_ifft": right.get(field)}
                    for field in sorted(set(left) | set(right))
                    if left.get(field) != right.get(field)
                }
                errors.append(
                    f"paired args mismatch for {protocol} seed {seed}: {json.dumps(differing, sort_keys=True)}"
                )

    by_representation: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for representation in REPRESENTATIONS:
        selected = [row for row in sample_rows if row["representation"] == representation]
        keyed = {prediction_key(row): row for row in selected}
        if len(keyed) != len(selected):
            errors.append(f"{representation}: duplicate canonical prediction keys across runs")
        by_representation[representation] = keyed
    if set(by_representation["frequency"]) != set(by_representation["rect_ifft"]):
        left_keys = set(by_representation["frequency"])
        right_keys = set(by_representation["rect_ifft"])
        errors.append(
            "Frequency and rect-IFFT prediction exports do not share identical paired keys; "
            f"frequency_only={len(left_keys - right_keys)}, rect_ifft_only={len(right_keys - left_keys)}"
        )

    report = {
        "status": "pass" if not errors else "fail",
        "protocol": "tgrs_nested_nonpolar_v2",
        "run_root": str(run_root.resolve()),
        "split_root": str(split_root.resolve()),
        "expected_representations": REPRESENTATIONS,
        "expected_model": "freq_mlp for both representations",
        "expected_param_count": EXPECTED_PARAM_COUNT,
        "experiment_lock": {
            "path": str(lock_path.resolve()) if lock_path is not None else None,
            "sha256": lock_sha256,
            "lock_version": lock_record.get("lock_version") if lock_record else None,
        },
        "expected_seeds": seeds,
        "expected_task_count": len(expected),
        "observed_task_count": len(observed),
        "audited_unique_task_count": len(run_rows),
        "sample_row_count": len(sample_rows),
        "missing_tasks": [list(item) for item in missing],
        "extra_tasks": [list(item) for item in extra],
        "duplicate_tasks": {
            "|".join(map(str, key)): [str(path) for path in paths]
            for key, paths in duplicate_tasks.items()
        },
        "split_sha256": split_hashes,
        "warnings": warnings,
        "errors": errors,
        "runs": run_rows,
    }
    return report, run_rows, sample_rows


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gce = np.asarray([float(row["great_circle_err_deg"]) for row in rows], dtype=np.float64)
    top1 = np.asarray([float(row["top1"]) for row in rows], dtype=np.float64)
    theta = np.asarray([float(row["theta_err_deg"]) for row in rows], dtype=np.float64)
    phi = np.asarray([float(row["phi_err_deg"]) for row in rows], dtype=np.float64)
    return {
        "prediction_count": int(gce.size),
        "great_circle_mae_deg": float(gce.mean()),
        "great_circle_median_deg": float(np.median(gce)),
        "great_circle_p90_deg": float(np.quantile(gce, 0.90, method="higher")),
        "great_circle_p95_deg": float(np.quantile(gce, 0.95, method="higher")),
        "theta_mae_deg": float(theta.mean()),
        "phi_mae_deg": float(phi.mean()),
        "top1_acc": float(top1.mean()),
        "failure_rate_gt_30_deg": float(np.mean(gce > 30.0)),
        "failure_rate_gt_60_deg": float(np.mean(gce > 60.0)),
        "failure_rate_gt_90_deg": float(np.mean(gce > 90.0)),
    }


def object_seed_cells(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), int(row["seed"]), str(row["object_code"]))].append(row)
    result = []
    for (protocol, seed, object_code), group in sorted(groups.items()):
        result.append(
            {
                "label": label,
                "protocol": protocol,
                "fold": int(protocol.removeprefix("outer")) if protocol.startswith("outer") else 0,
                "seed": seed,
                "object_code": object_code,
                "direction_count": len(group),
                "great_circle_mae_deg": float(
                    np.mean([float(row["great_circle_err_deg"]) for row in group])
                ),
                "theta_mae_deg": float(np.mean([float(row["theta_err_deg"]) for row in group])),
                "phi_mae_deg": float(np.mean([float(row["phi_err_deg"]) for row in group])),
                "top1_acc": float(np.mean([float(row["top1"]) for row in group])),
            }
        )
    return result


def object_then_seed_bootstrap(
    cells: list[dict[str, Any]],
    value_key: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    objects = sorted({str(row["object_code"]) for row in cells})
    seeds = sorted({int(row["seed"]) for row in cells})
    lookup = {
        (str(row["object_code"]), int(row["seed"])): float(row[value_key])
        for row in cells
    }
    missing = [(object_code, seed_value) for object_code in objects for seed_value in seeds if (object_code, seed_value) not in lookup]
    if missing:
        raise RuntimeError(f"object-seed matrix is incomplete for {value_key}: {missing[:10]}")
    matrix = np.asarray(
        [[lookup[(object_code, seed_value)] for seed_value in seeds] for object_code in objects],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        object_indices = rng.integers(0, len(objects), size=len(objects))
        seed_indices = rng.integers(0, len(seeds), size=(len(objects), len(seeds)))
        estimates[index] = float(matrix[object_indices[:, None], seed_indices].mean())
    return {
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "resampling_hierarchy": "object_then_seed; directions fixed within each object-seed cell",
    }


def object_only_bootstrap(
    object_rows: list[dict[str, Any]],
    value_key: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap already-centered object effects, the confirmatory primary CI."""
    values = np.asarray([float(row[value_key]) for row in object_rows], dtype=np.float64)
    if values.size == 0:
        raise RuntimeError(f"cannot bootstrap an empty object vector for {value_key}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    estimates = values[draws].mean(axis=1)
    return {
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
        "iterations": iterations,
        "bootstrap_seed": seed,
        "resampling_unit": "centered object mean after averaging fixed directions and all training seeds",
    }


def aggregate_representation(
    rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {}
    all_cells: list[dict[str, Any]] = []
    categories = {
        "id": {"id"},
        "held_geometry": {"outer1", "outer2", "outer3", "outer4"},
    }
    for category_index, (category, protocols) in enumerate(categories.items()):
        selected = [row for row in rows if row["protocol"] in protocols]
        cells = object_seed_cells(selected, str(rows[0]["representation"]))
        all_cells.extend(cells)
        per_object = []
        for object_code in sorted({str(row["object_code"]) for row in cells}):
            group = [row for row in cells if str(row["object_code"]) == object_code]
            per_object.append(
                {
                    "object_code": object_code,
                    "fold": int(group[0]["fold"]),
                    "great_circle_mae_deg": float(np.mean([row["great_circle_mae_deg"] for row in group])),
                    "theta_mae_deg": float(np.mean([row["theta_mae_deg"] for row in group])),
                    "phi_mae_deg": float(np.mean([row["phi_mae_deg"] for row in group])),
                    "top1_acc": float(np.mean([row["top1_acc"] for row in group])),
                    "seed_count": len(group),
                    "directions_per_seed": sorted({int(row["direction_count"]) for row in group}),
                }
            )
        per_seed = []
        for seed_value in sorted({int(row["seed"]) for row in cells}):
            group = [row for row in cells if int(row["seed"]) == seed_value]
            per_seed.append(
                {
                    "seed": seed_value,
                    "object_centered_gce_deg": float(
                        np.mean([row["great_circle_mae_deg"] for row in group])
                    ),
                    "object_centered_top1": float(np.mean([row["top1_acc"] for row in group])),
                }
            )
        summary = metric_summary(selected)
        summary.update(
            {
                "run_count": len({(row["protocol"], int(row["seed"])) for row in selected}),
                "object_count": len(per_object),
                "seed_count": len(per_seed),
                "object_centered_gce_deg": float(
                    np.mean([row["great_circle_mae_deg"] for row in per_object])
                ),
                "object_centered_theta_mae_deg": float(
                    np.mean([row["theta_mae_deg"] for row in per_object])
                ),
                "object_centered_phi_mae_deg": float(
                    np.mean([row["phi_mae_deg"] for row in per_object])
                ),
                "object_centered_top1": float(np.mean([row["top1_acc"] for row in per_object])),
                "object_centered_gce_95ci": object_only_bootstrap(
                    per_object,
                    "great_circle_mae_deg",
                    bootstrap_iterations,
                    bootstrap_seed + category_index * 20,
                ),
                "object_centered_top1_95ci": object_only_bootstrap(
                    per_object,
                    "top1_acc",
                    bootstrap_iterations,
                    bootstrap_seed + category_index * 20 + 1,
                ),
                "object_seed_hierarchical_gce_95ci": object_then_seed_bootstrap(
                    cells,
                    "great_circle_mae_deg",
                    bootstrap_iterations,
                    bootstrap_seed + category_index * 20 + 2,
                ),
                "object_seed_hierarchical_top1_95ci": object_then_seed_bootstrap(
                    cells,
                    "top1_acc",
                    bootstrap_iterations,
                    bootstrap_seed + category_index * 20 + 3,
                ),
                "per_object": per_object,
                "per_seed": per_seed,
            }
        )
        if category == "held_geometry":
            summary["per_fold"] = []
            for fold in range(1, 5):
                fold_rows = [row for row in selected if int(row["fold"]) == fold]
                fold_cells = [row for row in cells if int(row["fold"]) == fold]
                fold_summary = metric_summary(fold_rows)
                fold_summary["fold"] = fold
                fold_summary["object_centered_gce_deg"] = float(
                    np.mean([row["great_circle_mae_deg"] for row in fold_cells])
                )
                fold_summary["object_centered_top1"] = float(
                    np.mean([row["top1_acc"] for row in fold_cells])
                )
                summary["per_fold"].append(fold_summary)
            summary["leave_one_fold_out"] = []
            for omitted_fold in range(1, 5):
                retained = [row for row in per_object if int(row["fold"]) != omitted_fold]
                summary["leave_one_fold_out"].append(
                    {
                        "omitted_fold": omitted_fold,
                        "object_count": len(retained),
                        "object_centered_gce_deg": float(
                            np.mean([row["great_circle_mae_deg"] for row in retained])
                        ),
                        "object_centered_top1": float(
                            np.mean([row["top1_acc"] for row in retained])
                        ),
                    }
                )
        result[category] = summary
    return result, all_cells


def aggregate_paired(
    frequency_rows: list[dict[str, Any]],
    rect_ifft_rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    frequency = {prediction_key(row): row for row in frequency_rows}
    rect_ifft = {prediction_key(row): row for row in rect_ifft_rows}
    if set(frequency) != set(rect_ifft):
        raise RuntimeError("cannot aggregate unmatched Frequency/rect-IFFT prediction keys")
    paired_rows = []
    for key in sorted(frequency):
        left, right = frequency[key], rect_ifft[key]
        paired_rows.append(
            {
                "protocol": left["protocol"],
                "fold": left["fold"],
                "seed": left["seed"],
                "object_code": left["object_code"],
                "theta_true_deg": left["theta_true_deg"],
                "phi_true_deg": left["phi_true_deg"],
                "great_circle_err_frequency_deg": left["great_circle_err_deg"],
                "great_circle_err_rect_ifft_deg": right["great_circle_err_deg"],
                "great_circle_err_difference_deg": left["great_circle_err_deg"]
                - right["great_circle_err_deg"],
                "theta_err_difference_deg": left["theta_err_deg"] - right["theta_err_deg"],
                "phi_err_difference_deg": left["phi_err_deg"] - right["phi_err_deg"],
                "top1_frequency": left["top1"],
                "top1_rect_ifft": right["top1"],
                "top1_difference": left["top1"] - right["top1"],
            }
        )

    proxy_rows = [
        {
            "protocol": row["protocol"],
            "fold": row["fold"],
            "seed": row["seed"],
            "object_code": row["object_code"],
            "great_circle_err_deg": row["great_circle_err_difference_deg"],
            "theta_err_deg": row["theta_err_difference_deg"],
            "phi_err_deg": row["phi_err_difference_deg"],
            "top1": row["top1_difference"],
        }
        for row in paired_rows
    ]
    paired_cells = object_seed_cells(proxy_rows, "frequency_minus_rect_ifft")
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
                    "mean_gce_difference_deg": float(
                        np.mean([row["great_circle_mae_deg"] for row in group])
                    ),
                    "mean_top1_difference": float(np.mean([row["top1_acc"] for row in group])),
                }
            )
        run_differences = []
        for protocol in sorted(protocols):
            for seed in sorted({int(row["seed"]) for row in selected if row["protocol"] == protocol}):
                group = [
                    row for row in selected if row["protocol"] == protocol and int(row["seed"]) == seed
                ]
                run_differences.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "mean_gce_difference_deg": float(
                            np.mean([row["great_circle_err_difference_deg"] for row in group])
                        ),
                        "mean_top1_difference": float(
                            np.mean([row["top1_difference"] for row in group])
                        ),
                    }
                )
        category_result = {
            "paired_prediction_count": len(selected),
            "object_count": len(per_object),
            "run_count": len(run_differences),
            "mean_gce_difference_deg": float(
                np.mean([row["great_circle_err_difference_deg"] for row in selected])
            ),
            "object_centered_gce_difference_deg": float(
                np.mean([row["mean_gce_difference_deg"] for row in per_object])
            ),
            "object_centered_top1_difference": float(
                np.mean([row["mean_top1_difference"] for row in per_object])
            ),
            "frequency_lower_prediction_fraction": float(
                np.mean([row["great_circle_err_difference_deg"] < 0.0 for row in selected])
            ),
            "frequency_lower_object_seed_cells": int(
                sum(row["great_circle_mae_deg"] < 0.0 for row in cells)
            ),
            "object_seed_cell_count": len(cells),
            "frequency_lower_runs": int(
                sum(row["mean_gce_difference_deg"] < 0.0 for row in run_differences)
            ),
            "object_centered_gce_difference_95ci": object_only_bootstrap(
                per_object,
                "mean_gce_difference_deg",
                bootstrap_iterations,
                bootstrap_seed + category_index * 20,
            ),
            "object_centered_top1_difference_95ci": object_only_bootstrap(
                per_object,
                "mean_top1_difference",
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
            "per_run": run_differences,
        }
        if category == "held_geometry":
            category_result["per_fold"] = []
            for fold in range(1, 5):
                group = [row for row in selected if int(row["fold"]) == fold]
                category_result["per_fold"].append(
                    {
                        "fold": fold,
                        "mean_gce_difference_deg": float(
                            np.mean([row["great_circle_err_difference_deg"] for row in group])
                        ),
                        "mean_top1_difference": float(
                            np.mean([row["top1_difference"] for row in group])
                        ),
                    }
                )
            category_result["leave_one_fold_out"] = []
            for omitted_fold in range(1, 5):
                retained = [row for row in per_object if int(row["fold"]) != omitted_fold]
                category_result["leave_one_fold_out"].append(
                    {
                        "omitted_fold": omitted_fold,
                        "object_count": len(retained),
                        "object_centered_gce_difference_deg": float(
                            np.mean([row["mean_gce_difference_deg"] for row in retained])
                        ),
                        "object_centered_top1_difference": float(
                            np.mean([row["mean_top1_difference"] for row in retained])
                        ),
                    }
                )
        result[category] = category_result
    return result, paired_rows, paired_cells


def aggregate_all(
    sample_rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models = {}
    object_cells: list[dict[str, Any]] = []
    by_representation = {
        representation: [row for row in sample_rows if row["representation"] == representation]
        for representation in REPRESENTATIONS
    }
    for index, representation in enumerate(REPRESENTATIONS):
        if not by_representation[representation]:
            continue
        models[representation], cells = aggregate_representation(
            by_representation[representation],
            bootstrap_iterations,
            bootstrap_seed + index * 100,
        )
        object_cells.extend(cells)
    paired = {}
    paired_rows: list[dict[str, Any]] = []
    if all(by_representation.values()):
        paired, paired_rows, paired_cells = aggregate_paired(
            by_representation["frequency"],
            by_representation["rect_ifft"],
            bootstrap_iterations,
            bootstrap_seed + 10_000,
        )
        object_cells.extend(paired_cells)
    return (
        {
            "status": "ok",
            "protocol": "tgrs_nested_nonpolar_v2",
            "definitions": {
                "object_centered": (
                    "average the fixed-direction mean within each object-seed cell, then average seeds "
                    "within object and objects with equal weight"
                ),
                "uncertainty": (
                    "primary percentile CI resamples the 20 centered object means after averaging fixed "
                    "directions and all seeds; an object-then-seed hierarchical CI is reported as sensitivity"
                ),
                "paired_difference": "Frequency minus rectangular-IFFT; negative favors Frequency",
            },
            "models": models,
            "paired_difference": paired,
        },
        object_cells,
        paired_rows,
        [row for representation in REPRESENTATIONS for row in by_representation[representation]],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "review_protocol"
        / "splits"
        / "tgrs_nested_nonpolar_v2",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "review_protocol"
        / "matched_rect91_global_mlp_lock.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(70, 77)))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial audit/aggregate and exit zero while a synchronized matrix is still incomplete.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit, run_rows, sample_rows = audit_run_root(
        args.run_root,
        args.split_root,
        sorted(set(args.seeds)),
        lock_path=args.lock,
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "samples.csv", sample_rows)

    aggregate = {
        "status": "unavailable",
        "reason": "no complete paired prediction exports were found",
    }
    object_cells: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    try:
        if sample_rows:
            aggregate, object_cells, paired_rows, _ = aggregate_all(
                sample_rows,
                args.bootstrap_iterations,
                args.bootstrap_seed,
            )
    except Exception as exc:
        audit["errors"].append(f"aggregation failed: {type(exc).__name__}: {exc}")
        audit["status"] = "fail"
        (args.output_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        aggregate = {
            "status": "fail",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "object_seed_cells.csv", object_cells)
    write_csv(args.output_dir / "paired_samples.csv", paired_rows)

    print(
        json.dumps(
            {
                "audit_status": audit["status"],
                "expected_task_count": audit["expected_task_count"],
                "observed_task_count": audit["observed_task_count"],
                "error_count": len(audit["errors"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if audit["status"] == "pass" or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
