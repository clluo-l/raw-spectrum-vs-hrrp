#!/usr/bin/env python3
"""CPU audit of function-matched Frequency/rect-IFFT MLP checkpoints.

For globally L2-normalized complex-four inputs, define ``x_rect = Q x_freq``,
where Q is the real-block representation of the unitary length-91
``fftshift(ifft(..., norm='ortho'))`` transform applied independently to the
two receive components.  A Frequency first layer ``W_freq`` is therefore
function-matched in the rect-IFFT basis by ``W_rect = W_freq Q^{-1} =
W_freq Q.T``.  Biases and every later parameter remain unchanged.

This script never trains a model and is CPU-only.  It evaluates all 35 strict
Frequency checkpoints on their Frequency test inputs and the conjugated
models on the corresponding rect-IFFT inputs, reporting input, logit,
posterior, and hard-decision discrepancies for every evaluated sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from train_hrrp_class_ddp import FrequencySequenceFactorizedDirectionClassifier  # noqa: E402
from train_hrrp_cpu_smoke import HRRPSplitDataset  # noqa: E402


RUN_RE = re.compile(
    r"_FORMAL_matched_rect91_frequency_(?P<protocol>id|outer[1-4])_"
    r"seed(?P<seed>\d+)_sgpu(?P<gpu>\d+)$"
)
PROTOCOLS = ("id", "outer1", "outer2", "outer3", "outer4")
SEEDS = tuple(range(70, 77))
FIRST_WEIGHT_KEY = "encoder.network.0.weight"
EXPECTED_PARAM_COUNT = 3_022_993
THETA_VALUES = np.asarray([30.0, 60.0, 90.0, 120.0, 150.0], dtype=np.float32)
PHI_VALUES = np.arange(0.0, 360.0, 30.0, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complex_unitary_rect_ifft_matrix(length: int = 91) -> np.ndarray:
    """Return T such that h = T e is unitary IFFT followed by fftshift."""
    identity = np.eye(length, dtype=np.complex128)
    return np.fft.fftshift(np.fft.ifft(identity, axis=0, norm="ortho"), axes=0)


def real_block_frequency_to_rect_matrix(length: int = 91, components: int = 2) -> np.ndarray:
    """Return channel-major Q mapping normalized complex-four Frequency to rect-IFFT."""
    transform = complex_unitary_rect_ifft_matrix(length)
    real = transform.real
    imag = transform.imag
    one_component = np.block([[real, -imag], [imag, real]])
    blocks = np.zeros(
        (components * one_component.shape[0], components * one_component.shape[1]),
        dtype=np.float64,
    )
    block_size = one_component.shape[0]
    for component in range(components):
        start = component * block_size
        blocks[start : start + block_size, start : start + block_size] = one_component
    return blocks


def matrix_audit(q: np.ndarray) -> dict[str, Any]:
    identity = np.eye(q.shape[0], dtype=np.float64)
    orthogonality = q.T @ q - identity
    singular_values = np.linalg.svd(q, compute_uv=False)
    sign, logabsdet = np.linalg.slogdet(q)
    return {
        "shape": list(q.shape),
        "dtype": str(q.dtype),
        "sha256": sha256_array(q),
        "max_abs_qtq_minus_i": float(np.max(np.abs(orthogonality))),
        "fro_qtq_minus_i": float(np.linalg.norm(orthogonality)),
        "singular_value_min": float(singular_values.min()),
        "singular_value_max": float(singular_values.max()),
        "determinant_sign": float(sign),
        "log_abs_determinant": float(logabsdet),
        "definition": (
            "x_rect = Q x_frequency for channel-major [Re(Etheta), Im(Etheta), "
            "Re(Ephi), Im(Ephi)] after global sample-L2 normalization"
        ),
    }


def build_model() -> FrequencySequenceFactorizedDirectionClassifier:
    return FrequencySequenceFactorizedDirectionClassifier(
        domain="frequency",
        family="mlp",
        input_channels=4,
        num_theta_classes=5,
        num_phi_classes=12,
        feature_dim=384,
        base_channels=96,
        sequence_layers=3,
        heads=8,
        head_hidden=384,
        dropout=0.1,
    )


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is not a state dictionary: {path}")
    state = {}
    for key, value in payload.items():
        normalized_key = str(key).removeprefix("module.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint entry {key!r} is not a tensor")
        state[normalized_key] = value.detach().cpu()
    expected_keys = set(build_model().state_dict())
    if set(state) != expected_keys:
        raise RuntimeError(
            f"checkpoint state keys differ from the strict MLP; "
            f"missing={sorted(expected_keys - set(state))}, extra={sorted(set(state) - expected_keys)}"
        )
    return state


def conjugate_first_layer_state(
    frequency_state: dict[str, torch.Tensor],
    q_frequency_to_rect: np.ndarray,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if FIRST_WEIGHT_KEY not in frequency_state:
        raise KeyError(FIRST_WEIGHT_KEY)
    weight = frequency_state[FIRST_WEIGHT_KEY]
    if tuple(weight.shape) != (1024, 4 * 91):
        raise RuntimeError(f"unexpected first-layer shape: {tuple(weight.shape)}")
    q_tensor = torch.as_tensor(q_frequency_to_rect, dtype=weight.dtype, device="cpu")
    rect_state = {key: value.clone() for key, value in frequency_state.items()}
    rect_weight = weight @ q_tensor.T
    rect_state[FIRST_WEIGHT_KEY] = rect_weight
    changed_keys = [
        key for key in frequency_state if not torch.equal(frequency_state[key], rect_state[key])
    ]
    reconstructed = rect_weight @ q_tensor
    reconstruction_error = float(torch.max(torch.abs(reconstructed - weight)).item())
    return rect_state, {
        "formula": "W_rect = W_frequency Q^{-1} = W_frequency Q.T",
        "first_weight_key": FIRST_WEIGHT_KEY,
        "changed_keys": changed_keys,
        "unchanged_key_count": len(frequency_state) - len(changed_keys),
        "weight_dtype": str(weight.dtype),
        "frequency_weight_sha256": sha256_array(weight.numpy()),
        "rect_weight_sha256": sha256_array(rect_weight.numpy()),
        "max_abs_inverse_reconstruction_error": reconstruction_error,
    }


def split_path(split_root: Path, protocol: str, role: str) -> Path:
    return split_root / f"{protocol}_{role}.txt"


def expected_task_set() -> set[tuple[str, int]]:
    return {(protocol, seed) for protocol in PROTOCOLS for seed in SEEDS}


def validate_base_strict_lock(
    function_lock: dict[str, Any],
    function_lock_path: Path,
    dataset_root: Path,
    split_root: Path,
) -> dict[str, Any]:
    base_path = function_lock_path.parent / "matched_rect91_global_mlp_lock.json"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    observed_base_hash = sha256_file(base_path)
    expected_base_hash = str(function_lock.get("base_strict_matrix", {}).get("lock_sha256", ""))
    if observed_base_hash != expected_base_hash:
        raise RuntimeError("base strict lock hash differs from the function-matched lock")
    base = read_json(base_path)
    if base.get("status") != "locked" or base.get("protocol") != "tgrs_nested_nonpolar_v2":
        raise RuntimeError("base strict lock status/protocol is invalid")
    if int(base.get("model_contract", {}).get("expected_trainable_parameters", -1)) != EXPECTED_PARAM_COUNT:
        raise RuntimeError("base strict model contract changed")
    data_hashes = {}
    for relative, expected_hash in base.get("expected_data_hashes", {}).items():
        path = dataset_root / relative
        observed = sha256_file(path)
        if observed != expected_hash:
            raise RuntimeError(f"dataset content hash changed: {relative}")
        data_hashes[relative] = observed
    split_hashes = {}
    for name, expected_hash in base.get("expected_split_hashes", {}).items():
        path = split_root / name
        observed = sha256_file(path)
        if observed != expected_hash:
            raise RuntimeError(f"canonical split hash changed: {name}")
        split_hashes[name] = observed
    return {
        "path": str(base_path.resolve()),
        "sha256": observed_base_hash,
        "lock_version": base.get("lock_version"),
        "data_sha256": data_hashes,
        "split_sha256": split_hashes,
    }


def discover_frequency_runs(run_root: Path) -> dict[tuple[str, int], Path]:
    candidates: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for checkpoint in run_root.rglob("model_best.pt"):
        run_dir = checkpoint.parent
        match = RUN_RE.search(run_dir.name)
        if match is not None:
            candidates[(match.group("protocol"), int(match.group("seed")))].append(run_dir)
    duplicate = {key: paths for key, paths in candidates.items() if len(paths) != 1}
    if duplicate:
        raise RuntimeError(
            "duplicate strict Frequency checkpoints: "
            + json.dumps({str(key): [str(path) for path in paths] for key, paths in duplicate.items()})
        )
    observed = set(candidates)
    expected = expected_task_set()
    if observed != expected:
        raise RuntimeError(
            f"strict Frequency checkpoint matrix is incomplete; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return {key: paths[0] for key, paths in candidates.items()}


def validate_run_contract(
    run_dir: Path,
    protocol: str,
    seed: int,
    checkpoint: Path,
) -> dict[str, Any]:
    args = read_json(run_dir / "args.json")
    required = {
        "model": "freq_mlp",
        "input_key": "freq_complex_4ch",
        "normalize": "l2",
        "hrrp_window": "rect",
        "hrrp_nfft": 91,
        "hrrp_gate": "full",
        "hrrp_range_align": "none",
        "hrrp_phase_reference": "none",
        "classifier_head": "factorized",
        "hrrp_feature_dim": 384,
        "head_hidden": 384,
        "sequence_layers": 3,
        "dropout": 0.1,
        "seed": seed,
    }
    for field, expected in required.items():
        if args.get(field) != expected:
            raise RuntimeError(
                f"{run_dir}: {field}={args.get(field)!r} does not match {expected!r}"
            )
    if Path(str(args.get("train_split", ""))).name != f"{protocol}_train.txt":
        raise RuntimeError(f"{run_dir}: wrong train split")
    if Path(str(args.get("val_split", ""))).name != f"{protocol}_dev.txt":
        raise RuntimeError(f"{run_dir}: wrong development split")
    summary = read_json(run_dir / "summary.json")
    if summary.get("status") != "ok" or int(summary.get("epochs_completed", -1)) != 250:
        raise RuntimeError(f"{run_dir}: checkpoint training did not complete the strict protocol")
    if int(summary.get("param_count", -1)) != EXPECTED_PARAM_COUNT:
        raise RuntimeError(f"{run_dir}: unexpected parameter count")
    formal = read_json(run_dir / "formal_protocol.json")
    if formal.get("status") != "complete":
        raise RuntimeError(f"{run_dir}: formal record is not complete")
    if str(formal.get("checkpoint_sha256", "")) != sha256_file(checkpoint):
        raise RuntimeError(f"{run_dir}: checkpoint hash differs from formal record")
    return {
        "args_sha256": sha256_file(run_dir / "args.json"),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "formal_protocol_sha256": sha256_file(run_dir / "formal_protocol.json"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "best_dev_epoch": int(summary.get("best_val", {}).get("epoch", -1)),
    }


def load_protocol_inputs(
    dataset_root: Path,
    split_root: Path,
    protocol: str,
    q: np.ndarray,
) -> dict[str, Any]:
    test_split = split_path(split_root, protocol, "test")
    common = {
        "dataset_root": dataset_root,
        "split_name": str(test_split),
        "object_code": "all",
        "normalize": "l2",
        "hrrp_window": "rect",
        "hrrp_nfft": 91,
        "hrrp_gate": "full",
        "hrrp_range_align": "none",
        "hrrp_phase_reference": "none",
    }
    frequency_dataset = HRRPSplitDataset(key="freq_complex_4ch", **common)
    rect_dataset = HRRPSplitDataset(key="hrrp_rect91_complex_4ch", **common)
    if len(frequency_dataset) != len(rect_dataset):
        raise RuntimeError(f"{protocol}: Frequency and rect datasets have different lengths")
    frequency_values = []
    rect_values = []
    metas = []
    input_errors = []
    q_transpose = q.T.astype(np.float32, copy=False)
    for index in range(len(frequency_dataset)):
        frequency_tensor, _, frequency_meta = frequency_dataset[index]
        rect_tensor, _, rect_meta = rect_dataset[index]
        identity = (
            str(frequency_meta["object_code"]),
            float(frequency_meta["theta_inc_deg"]),
            float(frequency_meta["phi_inc_deg"]),
        )
        rect_identity = (
            str(rect_meta["object_code"]),
            float(rect_meta["theta_inc_deg"]),
            float(rect_meta["phi_inc_deg"]),
        )
        if identity != rect_identity:
            raise RuntimeError(f"{protocol}: paired dataset identity mismatch at row {index}")
        frequency_flat = frequency_tensor.numpy().reshape(-1).astype(np.float32, copy=False)
        rect_flat = rect_tensor.numpy().reshape(-1).astype(np.float32, copy=False)
        predicted_rect = frequency_flat @ q_transpose
        difference = predicted_rect - rect_flat
        input_errors.append(
            {
                "input_max_abs": float(np.max(np.abs(difference))),
                "input_rms": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
                "input_relative_inf": float(
                    np.max(np.abs(difference)) / max(float(np.max(np.abs(rect_flat))), np.finfo(float).tiny)
                ),
            }
        )
        frequency_values.append(frequency_tensor.numpy())
        rect_values.append(rect_tensor.numpy())
        metas.append(
            {
                "sample_index": index,
                "object_code": identity[0],
                "theta_true_deg": identity[1],
                "phi_true_deg": identity[2],
            }
        )
    return {
        "frequency": torch.from_numpy(np.stack(frequency_values).astype(np.float32)),
        "rect": torch.from_numpy(np.stack(rect_values).astype(np.float32)),
        "metas": metas,
        "input_errors": input_errors,
        "test_split": str(test_split),
        "test_split_sha256": sha256_file(test_split),
    }


def posterior_quantities(
    theta_logits: torch.Tensor,
    phi_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    theta_prob = torch.softmax(theta_logits, dim=1)
    phi_prob = torch.softmax(phi_logits, dim=1)
    joint = theta_prob.unsqueeze(2) * phi_prob.unsqueeze(1)
    theta_grid = torch.deg2rad(torch.as_tensor(THETA_VALUES, dtype=theta_logits.dtype)).view(1, -1, 1)
    phi_grid = torch.deg2rad(torch.as_tensor(PHI_VALUES, dtype=theta_logits.dtype)).view(1, 1, -1)
    x = (joint * torch.sin(theta_grid) * torch.cos(phi_grid)).sum(dim=(1, 2))
    y = (joint * torch.sin(theta_grid) * torch.sin(phi_grid)).sum(dim=(1, 2))
    z = (joint * torch.cos(theta_grid)).sum(dim=(1, 2))
    vector = torch.stack([x, y, z], dim=1)
    vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-8)
    return {
        "theta_prob": theta_prob,
        "phi_prob": phi_prob,
        "posterior_unit_xyz": vector,
        "theta_index": theta_logits.argmax(dim=1),
        "phi_index": phi_logits.argmax(dim=1),
    }


def row_max_abs(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return torch.amax(torch.abs(left - right).reshape(left.shape[0], -1), dim=1).cpu().numpy()


def evaluate_checkpoint(
    run_dir: Path,
    protocol: str,
    seed: int,
    inputs: dict[str, Any],
    q: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = run_dir / "model_best.pt"
    contract = validate_run_contract(run_dir, protocol, seed, checkpoint)
    frequency_state = load_checkpoint_state(checkpoint)
    rect_state, conjugation = conjugate_first_layer_state(frequency_state, q)
    if conjugation["changed_keys"] != [FIRST_WEIGHT_KEY]:
        raise RuntimeError(f"{run_dir}: conjugation changed more than the first layer")

    frequency_model = build_model().cpu().eval()
    rect_model = build_model().cpu().eval()
    frequency_model.load_state_dict(frequency_state, strict=True)
    rect_model.load_state_dict(rect_state, strict=True)
    sample_rows: list[dict[str, Any]] = []
    same_numeric_prediction_mismatch = 0
    conjugated_prediction_mismatch = 0

    frequency_values = inputs["frequency"]
    rect_values = inputs["rect"]
    with torch.inference_mode():
        for start in range(0, len(frequency_values), batch_size):
            stop = min(len(frequency_values), start + batch_size)
            frequency_batch = frequency_values[start:stop]
            rect_batch = rect_values[start:stop]
            frequency_theta, frequency_phi = frequency_model(frequency_batch)
            same_numeric_theta, same_numeric_phi = frequency_model(rect_batch)
            conjugated_theta, conjugated_phi = rect_model(rect_batch)
            frequency_posterior = posterior_quantities(frequency_theta, frequency_phi)
            conjugated_posterior = posterior_quantities(conjugated_theta, conjugated_phi)
            same_numeric_posterior = posterior_quantities(same_numeric_theta, same_numeric_phi)

            theta_logit_diff = row_max_abs(frequency_theta, conjugated_theta)
            phi_logit_diff = row_max_abs(frequency_phi, conjugated_phi)
            theta_prob_diff = row_max_abs(
                frequency_posterior["theta_prob"], conjugated_posterior["theta_prob"]
            )
            phi_prob_diff = row_max_abs(
                frequency_posterior["phi_prob"], conjugated_posterior["phi_prob"]
            )
            posterior_xyz_diff = row_max_abs(
                frequency_posterior["posterior_unit_xyz"],
                conjugated_posterior["posterior_unit_xyz"],
            )
            same_numeric_logit_diff = np.maximum(
                row_max_abs(frequency_theta, same_numeric_theta),
                row_max_abs(frequency_phi, same_numeric_phi),
            )
            conjugated_equal = (
                (frequency_posterior["theta_index"] == conjugated_posterior["theta_index"])
                & (frequency_posterior["phi_index"] == conjugated_posterior["phi_index"])
            ).cpu().numpy()
            same_numeric_equal = (
                (frequency_posterior["theta_index"] == same_numeric_posterior["theta_index"])
                & (frequency_posterior["phi_index"] == same_numeric_posterior["phi_index"])
            ).cpu().numpy()
            conjugated_prediction_mismatch += int(np.sum(~conjugated_equal))
            same_numeric_prediction_mismatch += int(np.sum(~same_numeric_equal))
            logit_scale = torch.maximum(
                torch.ones(stop - start),
                torch.maximum(
                    torch.amax(torch.abs(frequency_theta), dim=1),
                    torch.amax(torch.abs(frequency_phi), dim=1),
                ).cpu(),
            ).numpy()

            for local_index, global_index in enumerate(range(start, stop)):
                meta = inputs["metas"][global_index]
                input_error = inputs["input_errors"][global_index]
                sample_rows.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        **meta,
                        **input_error,
                        "theta_logit_max_abs": float(theta_logit_diff[local_index]),
                        "phi_logit_max_abs": float(phi_logit_diff[local_index]),
                        "logit_max_abs": float(
                            max(theta_logit_diff[local_index], phi_logit_diff[local_index])
                        ),
                        "logit_relative_inf": float(
                            max(theta_logit_diff[local_index], phi_logit_diff[local_index])
                            / logit_scale[local_index]
                        ),
                        "theta_probability_max_abs": float(theta_prob_diff[local_index]),
                        "phi_probability_max_abs": float(phi_prob_diff[local_index]),
                        "probability_max_abs": float(
                            max(theta_prob_diff[local_index], phi_prob_diff[local_index])
                        ),
                        "posterior_unit_xyz_max_abs": float(posterior_xyz_diff[local_index]),
                        "prediction_equal_after_conjugation": bool(conjugated_equal[local_index]),
                        "same_numeric_weight_logit_max_abs": float(
                            same_numeric_logit_diff[local_index]
                        ),
                        "prediction_equal_with_same_numeric_weight": bool(
                            same_numeric_equal[local_index]
                        ),
                    }
                )
    run_metrics = {
        "protocol": protocol,
        "seed": seed,
        "run_dir": str(run_dir),
        **contract,
        **conjugation,
        "evaluated_samples": len(sample_rows),
        "input_max_abs": max(row["input_max_abs"] for row in sample_rows),
        "input_relative_inf_max": max(row["input_relative_inf"] for row in sample_rows),
        "logit_max_abs": max(row["logit_max_abs"] for row in sample_rows),
        "logit_relative_inf_max": max(row["logit_relative_inf"] for row in sample_rows),
        "probability_max_abs": max(row["probability_max_abs"] for row in sample_rows),
        "posterior_unit_xyz_max_abs": max(
            row["posterior_unit_xyz_max_abs"] for row in sample_rows
        ),
        "prediction_mismatch_after_conjugation": conjugated_prediction_mismatch,
        "same_numeric_weight_logit_max_abs": max(
            row["same_numeric_weight_logit_max_abs"] for row in sample_rows
        ),
        "prediction_mismatch_with_same_numeric_weight": same_numeric_prediction_mismatch,
    }
    return run_metrics, sample_rows


def aggregate_maxima(run_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_max_fields = (
        "input_max_abs",
        "input_relative_inf",
        "logit_max_abs",
        "logit_relative_inf",
        "probability_max_abs",
        "posterior_unit_xyz_max_abs",
        "same_numeric_weight_logit_max_abs",
    )
    result = {
        field: float(max(float(row[field]) for row in sample_rows)) for field in numeric_max_fields
    }
    result.update(
        {
            "checkpoint_count": len(run_rows),
            "evaluated_sample_checkpoint_pairs": len(sample_rows),
            "prediction_mismatch_after_conjugation": int(
                sum(int(row["prediction_mismatch_after_conjugation"]) for row in run_rows)
            ),
            "prediction_mismatch_with_same_numeric_weight": int(
                sum(int(row["prediction_mismatch_with_same_numeric_weight"]) for row in run_rows)
            ),
            "same_numeric_weight_prediction_mismatch_fraction": float(
                np.mean(
                    [not bool(row["prediction_equal_with_same_numeric_weight"]) for row in sample_rows]
                )
            ),
            "first_weight_inverse_reconstruction_max_abs": float(
                max(float(row["max_abs_inverse_reconstruction_error"]) for row in run_rows)
            ),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=repo_root / "code/review_protocol/splits/tgrs_nested_nonpolar_v2",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=repo_root / "code/review_protocol/function_matched_rect91_conjugation_lock.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--save-conjugated-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for required in (args.run_root, args.dataset_root, args.split_root, args.lock):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("batch size and CPU thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    lock = read_json(args.lock)
    if lock.get("status") != "locked" or lock.get("protocol") != "tgrs_nested_nonpolar_v2":
        raise RuntimeError("unexpected function-matched lock")
    base_lock_audit = validate_base_strict_lock(
        lock, args.lock, args.dataset_root, args.split_root
    )
    q = real_block_frequency_to_rect_matrix(length=91, components=2)
    q_audit = matrix_audit(q)
    if q_audit["sha256"] != lock.get("q_contract", {}).get("expected_sha256"):
        raise RuntimeError("constructed Q differs from the frozen lock")
    runs = discover_frequency_runs(args.run_root)
    input_cache = {
        protocol: load_protocol_inputs(args.dataset_root, args.split_root, protocol, q)
        for protocol in PROTOCOLS
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "Q_frequency_to_rect.npy", q, allow_pickle=False)

    run_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for protocol, seed in sorted(runs):
        run_metrics, rows = evaluate_checkpoint(
            runs[(protocol, seed)],
            protocol,
            seed,
            input_cache[protocol],
            q,
            args.batch_size,
        )
        run_rows.append(run_metrics)
        sample_rows.extend(rows)
        if args.save_conjugated_root is not None:
            source_state = load_checkpoint_state(runs[(protocol, seed)] / "model_best.pt")
            rect_state, _ = conjugate_first_layer_state(source_state, q)
            destination = args.save_conjugated_root / runs[(protocol, seed)].name / "model_best.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(rect_state, destination)

    aggregate = aggregate_maxima(run_rows, sample_rows)
    tolerances = lock["numeric_tolerances"]
    gates = {
        "q_orthogonality": q_audit["max_abs_qtq_minus_i"]
        <= float(tolerances["q_max_abs_qtq_minus_i"]),
        "input_mapping": aggregate["input_max_abs"] <= float(tolerances["input_max_abs"]),
        "input_mapping_relative": aggregate["input_relative_inf"]
        <= float(tolerances["input_relative_inf"]),
        "first_weight_inverse": aggregate["first_weight_inverse_reconstruction_max_abs"]
        <= float(tolerances["first_weight_inverse_max_abs"]),
        "logits": aggregate["logit_max_abs"] <= float(tolerances["logit_max_abs"]),
        "logits_relative": aggregate["logit_relative_inf"]
        <= float(tolerances["logit_relative_inf"]),
        "probabilities": aggregate["probability_max_abs"]
        <= float(tolerances["probability_max_abs"]),
        "posterior_direction": aggregate["posterior_unit_xyz_max_abs"]
        <= float(tolerances["posterior_unit_xyz_max_abs"]),
        "hard_predictions": aggregate["prediction_mismatch_after_conjugation"] == 0,
    }
    status = "pass" if all(gates.values()) else "fail"
    summary = {
        "status": status,
        "protocol": "tgrs_nested_nonpolar_v2",
        "device": "cpu",
        "theory": {
            "input_relation": "x_rect = Q x_frequency",
            "function_matched_first_layer": "W_rect = W_frequency Q^{-1} = W_frequency Q.T",
            "unchanged_parameters": "first-layer bias and every subsequent parameter",
            "same_numerical_initialization": (
                "Using the same numerical first-layer tensor W in both bases is parameter-sample "
                "matched but not function matched: W Q x generally differs from W x."
            ),
            "scope": (
                "The conjugation proves exact representational closure of the flattened MLP in "
                "exact arithmetic. It does not make AdamW coordinate invariant during training."
            ),
        },
        "q_audit": q_audit,
        "base_strict_lock_audit": base_lock_audit,
        "numeric_tolerances": tolerances,
        "gates": gates,
        "aggregate": aggregate,
        "per_protocol_inputs": {
            protocol: {
                "samples": len(input_cache[protocol]["metas"]),
                "test_split": input_cache[protocol]["test_split"],
                "test_split_sha256": input_cache[protocol]["test_split_sha256"],
                "input_max_abs": max(
                    row["input_max_abs"] for row in input_cache[protocol]["input_errors"]
                ),
                "input_relative_inf_max": max(
                    row["input_relative_inf"] for row in input_cache[protocol]["input_errors"]
                ),
            }
            for protocol in PROTOCOLS
        },
        "runs": run_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "samples.csv", sample_rows)

    script_path = Path(__file__).resolve()
    provenance = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(script_path), *sys.argv[1:]],
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "lock": {"path": str(args.lock.resolve()), "sha256": sha256_file(args.lock)},
        "base_strict_lock_audit": base_lock_audit,
        "strict_run_root": str(args.run_root.resolve()),
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_root / "dataset_manifest.json"),
            "official_index_sha256": sha256_file(
                args.dataset_root / "indexes/official/samples_index.jsonl"
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": args.threads,
            "cpu_only": True,
        },
        "outputs": {},
    }
    for name in ("Q_frequency_to_rect.npy", "summary.json", "runs.csv", "samples.csv"):
        path = args.output_dir / name
        provenance["outputs"][name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "checkpoints": aggregate["checkpoint_count"],
                "sample_checkpoint_pairs": aggregate["evaluated_sample_checkpoint_pairs"],
                "logit_max_abs": aggregate["logit_max_abs"],
                "probability_max_abs": aggregate["probability_max_abs"],
                "posterior_unit_xyz_max_abs": aggregate["posterior_unit_xyz_max_abs"],
                "prediction_mismatch_after_conjugation": aggregate[
                    "prediction_mismatch_after_conjugation"
                ],
                "same_numeric_weight_prediction_mismatch_fraction": aggregate[
                    "same_numeric_weight_prediction_mismatch_fraction"
                ],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
