#!/usr/bin/env python3
"""Export per-sample and per-object metrics for a completed HRRP class run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_CODE = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from profile_hrrp_class_flops import DEFAULTS, PATH_KEYS, build_model_and_batch  # noqa: E402
from train_hrrp_class_ddp import (  # noqa: E402
    GeoConditionalDirectionClassifier,
    GeoDirectionClassifier,
    GeoFactorizedDirectionClassifier,
    GeoResidualFactorizedDirectionClassifier,
    HRRPClassDataset,
    HRRPConditionalDirectionClassifier,
    HRRPConditionalObjectDirectionClassifier,
    HRRPDirectionClassifier,
    HRRPFactorizedClassDataset,
    HRRPFactorizedDirectionClassifier,
    HRRPFactorizedObjectDirectionClassifier,
    build_angle_classes,
    build_factorized_angle_classes,
    circular_phi_error_deg,
    collate_geo_class_batch,
    infer_input_channels,
    is_geometry_model,
    make_base_datasets,
    make_geometry_wrapper,
    move_observation_to_device,
)


def load_run_args(args_json: Path) -> SimpleNamespace:
    values = dict(DEFAULTS)
    values.update(json.loads(args_json.read_text(encoding="utf-8")))
    if values.get("normalize") == "none":
        values["normalize"] = ""
    for key in PATH_KEYS:
        values[key] = Path(values[key])
    values["batch_size"] = 1
    values["num_workers"] = 0
    return SimpleNamespace(**values)


def apply_path_overrides(
    run_args: SimpleNamespace,
    *,
    dataset_root: Path | None = None,
    geometry_root: Path | None = None,
    eval_split: Path | None = None,
) -> None:
    """Apply relocated evaluation paths without undoing ``load_run_args`` typing."""

    if dataset_root is not None:
        run_args.dataset_root = dataset_root.resolve()
    if geometry_root is not None:
        run_args.geometry_root = geometry_root.resolve()
    if eval_split is not None:
        run_args.val_split = eval_split.resolve()
        run_args.val_include_objects = ""
        run_args.val_exclude_objects = ""
        run_args.holdout_objects = ""


def find_run_dir(path: Path) -> Path:
    if (path / "args.json").exists() and (path / "model_best.pt").exists():
        return path
    matches = sorted(path.glob("*/summary.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one completed child under {path}, found {len(matches)}")
    return matches[0].parent


def make_model(run_args: SimpleNamespace, num_theta: int, num_phi: int, num_joint: int, num_objects: int | None):
    input_channels = infer_input_channels(run_args.input_key)
    common = dict(
        input_channels=input_channels,
        feature_dim=run_args.hrrp_feature_dim,
        base_channels=run_args.hrrp_base_channels,
        transformer_layers=run_args.hrrp_transformer_layers,
        head_hidden=run_args.head_hidden,
        dropout=run_args.dropout,
    )
    if run_args.classifier_head in {"factorized", "conditional"}:
        if run_args.model == "geo":
            if run_args.fusion_type == "residual":
                return GeoResidualFactorizedDirectionClassifier(
                    num_theta_classes=num_theta,
                    num_phi_classes=num_phi,
                    geo_attn_depth=run_args.geo_attn_depth,
                    **common,
                )
            model_cls = GeoConditionalDirectionClassifier if run_args.classifier_head == "conditional" else GeoFactorizedDirectionClassifier
            return model_cls(
                num_theta_classes=num_theta,
                num_phi_classes=num_phi,
                geo_attn_depth=run_args.geo_attn_depth,
                fusion_type=run_args.fusion_type,
                **common,
            )
        if run_args.object_conditioning == "embedding":
            if num_objects is None:
                raise ValueError("num_objects is required for object-conditioned HRRP model")
            model_cls = HRRPConditionalObjectDirectionClassifier if run_args.classifier_head == "conditional" else HRRPFactorizedObjectDirectionClassifier
            return model_cls(
                num_theta_classes=num_theta,
                num_phi_classes=num_phi,
                num_objects=num_objects,
                object_embed_dim=run_args.object_embed_dim,
                **common,
            )
        model_cls = HRRPConditionalDirectionClassifier if run_args.classifier_head == "conditional" else HRRPFactorizedDirectionClassifier
        return model_cls(num_theta_classes=num_theta, num_phi_classes=num_phi, **common)

    if run_args.model == "geo":
        return GeoDirectionClassifier(
            num_classes=num_joint,
            geo_attn_depth=run_args.geo_attn_depth,
            fusion_type=run_args.fusion_type,
            **common,
        )
    return HRRPDirectionClassifier(num_classes=num_joint, **common)


def build_eval_objects(run_args: SimpleNamespace):
    train_base, val_base = make_base_datasets(run_args)
    theta_values, phi_values, theta_to_idx, phi_to_idx = build_factorized_angle_classes(train_base, val_base)
    angle_classes, class_to_idx, class_theta, class_phi = build_angle_classes(train_base, val_base)
    object_to_idx = None
    object_values = None
    if run_args.object_conditioning == "embedding":
        object_values = sorted({str(r["object_code"]) for r in train_base.rows + val_base.rows})
        object_to_idx = {value: idx for idx, value in enumerate(object_values)}

    if run_args.classifier_head in {"factorized", "conditional"}:
        val_ds = HRRPFactorizedClassDataset(val_base, theta_to_idx, phi_to_idx, object_to_idx=object_to_idx)
    else:
        val_ds = HRRPClassDataset(val_base, class_to_idx)

    if is_geometry_model(run_args.model):
        _, val_ds = make_geometry_wrapper(run_args, train_base, val_base, val_ds, val_ds)
        collate_fn = collate_geo_class_batch
    else:
        collate_fn = None

    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # Keep evaluation construction in lock-step with the formal training and
    # FLOPs profiler, including all Transolver-3 single/dual/geometry variants.
    model, _unused_profile_batch = build_model_and_batch(run_args, torch.device("cpu"))
    return loader, model, theta_values, phi_values, angle_classes, class_theta, class_phi


def decode_output(output, y: torch.Tensor, run_args: SimpleNamespace, theta_values, phi_values, class_theta, class_phi):
    if run_args.classifier_head in {"factorized", "conditional"}:
        theta_logits, phi_logits = output
        pred_theta_idx = theta_logits.argmax(dim=1)
        if phi_logits.dim() == 3:
            pred_phi_logits = phi_logits[torch.arange(phi_logits.shape[0], device=phi_logits.device), pred_theta_idx]
        else:
            pred_phi_logits = phi_logits
        pred_phi_idx = pred_phi_logits.argmax(dim=1)
        true_theta_idx = y[:, 0]
        true_phi_idx = y[:, 1]
        pole = y[:, 2].bool()
        theta_tensor = torch.tensor(theta_values, dtype=torch.float32, device=theta_logits.device)
        phi_tensor = torch.tensor(phi_values, dtype=torch.float32, device=theta_logits.device)
        pred_theta = theta_tensor[pred_theta_idx]
        true_theta = theta_tensor[true_theta_idx]
        pred_phi = phi_tensor[pred_phi_idx]
        true_phi = phi_tensor[true_phi_idx]
        theta_ok = pred_theta_idx == true_theta_idx
        phi_ok = pred_phi_idx == true_phi_idx
        theta_grid = torch.deg2rad(theta_tensor).view(1, -1, 1)
        phi_grid = torch.deg2rad(phi_tensor).view(1, 1, -1)
        theta_prob = torch.softmax(theta_logits, dim=1)
        if phi_logits.dim() == 3:
            joint_prob = theta_prob.unsqueeze(2) * torch.softmax(phi_logits, dim=2)
        else:
            joint_prob = theta_prob.unsqueeze(2) * torch.softmax(phi_logits, dim=1).unsqueeze(1)
    else:
        logits = output
        pred_idx = logits.argmax(dim=1)
        true_idx = y
        pred_theta = class_theta.to(logits.device)[pred_idx]
        pred_phi = class_phi.to(logits.device)[pred_idx]
        true_theta = class_theta.to(logits.device)[true_idx]
        true_phi = class_phi.to(logits.device)[true_idx]
        pole = (true_theta.abs() < 1e-6) | ((true_theta - 180.0).abs() < 1e-6)
        theta_ok = pred_theta == true_theta
        phi_ok = pred_phi == true_phi
        joint_prob = torch.softmax(logits, dim=1).unsqueeze(2)
        theta_grid = torch.deg2rad(class_theta.to(logits.device)).view(1, -1, 1)
        phi_grid = torch.deg2rad(class_phi.to(logits.device)).view(1, -1, 1)

    theta_err = (pred_theta - true_theta).abs()
    phi_err = circular_phi_error_deg(pred_phi, true_phi)
    top1 = theta_ok & (pole | phi_ok)
    mean_err = 0.5 * (theta_err + torch.where(pole, torch.zeros_like(phi_err), phi_err))
    report_mean_err = 0.5 * (theta_err + phi_err)
    pred_theta_rad = torch.deg2rad(pred_theta)
    true_theta_rad = torch.deg2rad(true_theta)
    delta_phi_rad = torch.deg2rad(pred_phi - true_phi)
    direction_cosine = (
        torch.cos(pred_theta_rad) * torch.cos(true_theta_rad)
        + torch.sin(pred_theta_rad) * torch.sin(true_theta_rad) * torch.cos(delta_phi_rad)
    ).clamp(-1.0, 1.0)
    great_circle_err = torch.rad2deg(torch.acos(direction_cosine))
    # A continuous posterior-mean direction is a readout of the same grid
    # probabilities, not a separately trained regression head.
    posterior_x = (joint_prob * torch.sin(theta_grid) * torch.cos(phi_grid)).sum(dim=(1, 2))
    posterior_y = (joint_prob * torch.sin(theta_grid) * torch.sin(phi_grid)).sum(dim=(1, 2))
    posterior_z = (joint_prob * torch.cos(theta_grid)).sum(dim=(1, 2))
    posterior_norm = torch.sqrt(posterior_x.square() + posterior_y.square() + posterior_z.square()).clamp_min(1e-8)
    posterior_theta = torch.rad2deg(torch.acos((posterior_z / posterior_norm).clamp(-1.0, 1.0)))
    posterior_phi = torch.rad2deg(torch.atan2(posterior_y, posterior_x)) % 360.0
    posterior_theta_err = (posterior_theta - true_theta).abs()
    posterior_phi_err = circular_phi_error_deg(posterior_phi, true_phi)
    posterior_direction_cosine = (
        torch.cos(torch.deg2rad(posterior_theta)) * torch.cos(true_theta_rad)
        + torch.sin(torch.deg2rad(posterior_theta))
        * torch.sin(true_theta_rad)
        * torch.cos(torch.deg2rad(posterior_phi - true_phi))
    ).clamp(-1.0, 1.0)
    posterior_great_circle_err = torch.rad2deg(torch.acos(posterior_direction_cosine))
    return {
        "pred_theta": pred_theta,
        "pred_phi": pred_phi,
        "true_theta": true_theta,
        "true_phi": true_phi,
        "theta_err": theta_err,
        "phi_err": phi_err,
        "top1": top1,
        "pole": pole,
        "mean_err": mean_err,
        "report_mean_err": report_mean_err,
        "great_circle_err": great_circle_err,
        "posterior_theta": posterior_theta,
        "posterior_phi": posterior_phi,
        "posterior_theta_err": posterior_theta_err,
        "posterior_phi_err": posterior_phi_err,
        "posterior_great_circle_err": posterior_great_circle_err,
    }


def summarize_rows(rows: list[dict]) -> dict:
    count = len(rows)
    nonpole = [row for row in rows if not row["pole"]]
    theta_mae = sum(row["theta_err_deg"] for row in rows) / max(count, 1)
    phi_mae = sum(row["phi_err_deg"] for row in nonpole) / max(len(nonpole), 1)
    mean_mae = 0.5 * (theta_mae + phi_mae)
    great_circle = np.asarray([row["great_circle_err_deg"] for row in rows], dtype=np.float64)
    great_circle_mean = float(great_circle.mean()) if count else 0.0
    great_circle_median = float(np.median(great_circle)) if count else 0.0
    great_circle_p90 = float(np.quantile(great_circle, 0.90, method="higher")) if count else 0.0
    great_circle_p95 = float(np.quantile(great_circle, 0.95, method="higher")) if count else 0.0
    posterior_great_circle = np.asarray(
        [row["posterior_great_circle_err_deg"] for row in rows], dtype=np.float64
    )
    return {
        "count": count,
        "nonpole_count": len(nonpole),
        "theta_mae_deg": theta_mae,
        "theta_score_pct": 100.0 * (1.0 - theta_mae / 180.0),
        "phi_mae_deg": phi_mae,
        "phi_score_pct": 100.0 * (1.0 - phi_mae / 360.0),
        "mean_mae_deg": mean_mae,
        "mean_score_pct": 0.5 * (100.0 * (1.0 - theta_mae / 180.0) + 100.0 * (1.0 - phi_mae / 360.0)),
        "top1_acc": sum(1 for row in rows if row["top1"]) / max(count, 1),
        "great_circle_mae_deg": great_circle_mean,
        "great_circle_median_deg": great_circle_median,
        "great_circle_p90_deg": great_circle_p90,
        "great_circle_p95_deg": great_circle_p95,
        "failure_rate_gt_30_deg": float(np.mean(great_circle > 30.0)) if count else 0.0,
        "failure_rate_gt_60_deg": float(np.mean(great_circle > 60.0)) if count else 0.0,
        "failure_rate_gt_90_deg": float(np.mean(great_circle > 90.0)) if count else 0.0,
        "posterior_great_circle_mae_deg": float(posterior_great_circle.mean()) if count else 0.0,
        "posterior_great_circle_median_deg": float(np.median(posterior_great_circle)) if count else 0.0,
        "posterior_great_circle_p90_deg": float(
            np.quantile(posterior_great_circle, 0.90, method="higher")
        )
        if count
        else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def first_meta(metas) -> dict:
    if isinstance(metas, (list, tuple)):
        return dict(metas[0])
    if isinstance(metas, dict):
        result = {}
        for key, value in metas.items():
            item = value[0] if isinstance(value, (list, tuple, torch.Tensor)) else value
            if isinstance(item, torch.Tensor) and item.numel() == 1:
                item = item.item()
            result[key] = item
        return result
    raise TypeError(f"unsupported metadata batch type: {type(metas)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval-split", type=Path, default=None, help="Optional untouched split manifest used only for this evaluation.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional relocated dataset root for evaluating a checkpoint copied from another host.",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=None,
        help="Optional relocated geometry root for evaluating a checkpoint copied from another host.",
    )
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--definition", default="model_best validation-set predictions")
    parser.add_argument("--eval-global-phase-deg", type=float, default=0.0)
    parser.add_argument("--eval-range-offset-m", type=float, default=0.0)
    parser.add_argument("--eval-common-gain-db", type=float, default=0.0)
    parser.add_argument("--eval-channel-gain-db", type=float, default=0.0)
    parser.add_argument("--eval-channel-phase-deg", type=float, default=0.0)
    parser.add_argument("--eval-receive-basis-rotation-deg", type=float, default=0.0)
    parser.add_argument("--eval-random-global-phase-max-deg", type=float, default=0.0)
    parser.add_argument("--eval-random-range-offset-max-m", type=float, default=0.0)
    parser.add_argument("--eval-random-common-gain-max-db", type=float, default=0.0)
    parser.add_argument("--eval-random-channel-gain-max-db", type=float, default=0.0)
    parser.add_argument("--eval-random-channel-phase-max-deg", type=float, default=0.0)
    parser.add_argument("--eval-random-receive-basis-rotation-max-deg", type=float, default=0.0)
    parser.add_argument("--eval-augment-seed", type=int, default=100070)
    parser.add_argument("--bandwidth-min-ghz", type=float, default=0.0)
    parser.add_argument("--bandwidth-max-ghz", type=float, default=0.0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--zero-etheta", action="store_true")
    parser.add_argument("--zero-ephi", action="store_true")
    parser.add_argument("--geometry-mode", choices=("true", "shuffle", "fixed"), default=None)
    parser.add_argument("--fixed-geometry-object", default=None)
    args = parser.parse_args()

    run_dir = find_run_dir(args.run_dir)
    run_args = load_run_args(run_dir / "args.json")
    apply_path_overrides(
        run_args,
        dataset_root=args.dataset_root,
        geometry_root=args.geometry_root,
        eval_split=args.eval_split,
    )
    run_args.eval_global_phase_deg = args.eval_global_phase_deg
    run_args.eval_range_offset_m = args.eval_range_offset_m
    run_args.eval_common_gain_db = args.eval_common_gain_db
    run_args.eval_channel_gain_db = args.eval_channel_gain_db
    run_args.eval_channel_phase_deg = args.eval_channel_phase_deg
    run_args.eval_receive_basis_rotation_deg = args.eval_receive_basis_rotation_deg
    run_args.eval_random_global_phase_max_deg = args.eval_random_global_phase_max_deg
    run_args.eval_random_range_offset_max_m = args.eval_random_range_offset_max_m
    run_args.eval_random_common_gain_max_db = args.eval_random_common_gain_max_db
    run_args.eval_random_channel_gain_max_db = args.eval_random_channel_gain_max_db
    run_args.eval_random_channel_phase_max_deg = args.eval_random_channel_phase_max_deg
    run_args.eval_random_receive_basis_rotation_max_deg = args.eval_random_receive_basis_rotation_max_deg
    run_args.bandwidth_min_ghz = args.bandwidth_min_ghz
    run_args.bandwidth_max_ghz = args.bandwidth_max_ghz
    run_args.snr_db = args.snr_db
    run_args.eval_zero_etheta = args.zero_etheta
    run_args.eval_zero_ephi = args.zero_ephi
    if args.geometry_mode is not None:
        run_args.geometry_mode = args.geometry_mode
        if args.geometry_mode == "shuffle":
            run_args.geometry_shuffle_scope = "val"
    if args.fixed_geometry_object is not None:
        run_args.fixed_geometry_object = args.fixed_geometry_object
    run_args.seed = args.eval_augment_seed - 100000
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    loader, model, theta_values, phi_values, _angle_classes, class_theta, class_phi = build_eval_objects(run_args)
    model.load_state_dict(torch.load(run_dir / args.checkpoint, map_location=device))
    model.to(device).eval()

    rows = []
    theta_logit_rows: list[np.ndarray] = []
    phi_logit_rows: list[np.ndarray] = []
    joint_logit_rows: list[np.ndarray] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if is_geometry_model(run_args.model):
                x, y, metas, verts, faces, face_edges, face_mask = batch
                output = model(
                    move_observation_to_device(x, device),
                    verts.to(device),
                    faces.to(device),
                    face_edges.to(device),
                    face_mask=face_mask.to(device),
                )
            else:
                x, y, metas = batch
                x = move_observation_to_device(x, device)
                y = y.to(device)
                if getattr(model, "uses_object_conditioning", False):
                    output = model(x, y[:, 3])
                else:
                    output = model(x)
            if isinstance(output, tuple):
                theta_logits_raw, phi_logits_raw = output
                theta_logit_rows.append(theta_logits_raw[0].detach().cpu().numpy().astype(np.float32))
                phi_logit_rows.append(phi_logits_raw[0].detach().cpu().numpy().reshape(-1).astype(np.float32))
            else:
                joint_logit_rows.append(output[0].detach().cpu().numpy().astype(np.float32))
            decoded = decode_output(output, y.to(device), run_args, theta_values, phi_values, class_theta, class_phi)
            meta = first_meta(metas)
            rows.append(
                {
                    "index": idx,
                    "object_code": str(meta.get("object_code", "")),
                    "mesh_object_code": str(meta.get("mesh_object_code", meta.get("object_code", ""))),
                    "theta_true_deg": float(decoded["true_theta"].cpu().item()),
                    "phi_true_deg": float(decoded["true_phi"].cpu().item()),
                    "theta_pred_deg": float(decoded["pred_theta"].cpu().item()),
                    "phi_pred_deg": float(decoded["pred_phi"].cpu().item()),
                    "theta_err_deg": float(decoded["theta_err"].cpu().item()),
                    "phi_err_deg": float(decoded["phi_err"].cpu().item()),
                    "mean_err_deg": float(decoded["report_mean_err"].cpu().item()),
                    "great_circle_err_deg": float(decoded["great_circle_err"].cpu().item()),
                    "posterior_theta_pred_deg": float(decoded["posterior_theta"].cpu().item()),
                    "posterior_phi_pred_deg": float(decoded["posterior_phi"].cpu().item()),
                    "posterior_theta_err_deg": float(decoded["posterior_theta_err"].cpu().item()),
                    "posterior_phi_err_deg": float(decoded["posterior_phi_err"].cpu().item()),
                    "posterior_great_circle_err_deg": float(decoded["posterior_great_circle_err"].cpu().item()),
                    "pole": bool(decoded["pole"].cpu().item()),
                    "top1": bool(decoded["top1"].cpu().item()),
                }
            )

    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object_code"]].append(row)
    object_rows = []
    for object_code, group in sorted(by_object.items()):
        summary = summarize_rows(group)
        object_rows.append({"object_code": object_code, **summary})
    summary = {
        "run_dir": str(run_dir),
        "definition": args.definition,
        "checkpoint": args.checkpoint,
        "eval_split": str(args.eval_split.resolve()) if args.eval_split is not None else str(run_args.val_split),
        "overall": summarize_rows(rows),
        "objects": object_rows,
    }

    out_dir = args.out_dir or (run_dir / "prediction_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "samples.csv", rows)
    write_csv(out_dir / "objects.csv", object_rows)
    logit_payload: dict[str, np.ndarray] = {
        "object_code": np.asarray([row["object_code"] for row in rows], dtype=str),
        "theta_true_deg": np.asarray([row["theta_true_deg"] for row in rows], dtype=np.float32),
        "phi_true_deg": np.asarray([row["phi_true_deg"] for row in rows], dtype=np.float32),
    }
    if theta_logit_rows:
        logit_payload["theta_logits"] = np.stack(theta_logit_rows)
        logit_payload["phi_logits"] = np.stack(phi_logit_rows)
    else:
        logit_payload["joint_logits"] = np.stack(joint_logit_rows)
    np.savez_compressed(out_dir / "logits.npz", **logit_payload)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, sort_keys=True), flush=True)
    print(out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
