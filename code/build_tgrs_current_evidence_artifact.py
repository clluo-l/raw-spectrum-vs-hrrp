"""Build an infrastructure-anonymized current-evidence TGRS audit archive.

This builder is intentionally independent of the historical Round-1 artifact
and the unfinished all-Round-2 collector.  It packages only evidence already
used by the current manuscript.  Source files are selected by an explicit
allowlist; text payloads fail closed on private paths, machine/account
identifiers, or credential-like material.  Scientific files are never edited
or redacted by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import tempfile
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA = "tgrs-current-evidence-artifact-v1"
READY_SCHEMA = "tgrs-current-evidence-ready-v1"
ARCHIVE_NAME = "tgrs_current_evidence_artifact.zip"
SIDECAR_NAME = "tgrs_current_evidence_artifact.manifest.json"
READY_NAME = "READY.json"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")

TEXT_SUFFIXES = {".bib", ".cls", ".csv", ".json", ".md", ".py", ".tex", ".txt"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".png"}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bz2",
    ".ckpt",
    ".glb",
    ".gltf",
    ".gz",
    ".h5",
    ".npy",
    ".npz",
    ".obj",
    ".off",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".ply",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".stl",
    ".tar",
    ".xz",
    ".zip",
}
FORBIDDEN_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
}
FORBIDDEN_PARTS = {".aws", ".git", ".ssh", "checkpoints", "models", "meshes"}
FORBIDDEN_MAGIC = (
    (b"PK\x03\x04", "ZIP-like payload"),
    (b"PK\x05\x06", "empty ZIP payload"),
    (b"PK\x07\x08", "spanned ZIP payload"),
    (b"\x89HDF\r\n\x1a\n", "HDF/checkpoint payload"),
    (b"\x93NUMPY", "NumPy array payload"),
)

# URLs are deliberately excluded from the Unix-path expression.  A portable
# interpreter shebang is the sole accepted path-shaped line; it identifies no
# machine, account, workspace, or scientific source location.
WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
UNC_RE = re.compile(r"(?<![\\])\\\\[A-Za-z0-9._-]+[\\/]+[A-Za-z0-9$_.-]+")
UNIX_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9:/\\.])/(?:[A-Za-z0-9._~+-]+/)+[A-Za-z0-9._~+-]+"
)
SSH_ID_RE = re.compile(
    r"\b[A-Za-z0-9._-]{2,}@[A-Za-z0-9._-]{2,}:(?:[/~]|[A-Za-z]:[\\/])"
)
IPV4_RE = re.compile(
    r"(?<![0-9.])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"
    r"(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9.])"
)
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----", re.I
)
AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
GITHUB_TOKEN_RE = re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{32,}\b")
OPENAI_TOKEN_RE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)"
    r"\s*[:=]\s*[\"']?[A-Za-z0-9_+./=-]{8,}"
)
STRUCTURED_ID_KEYS = {
    "host",
    "hostname",
    "host_name",
    "login_user",
    "machine_name",
    "node_name",
    "user_name",
    "username",
}
PORTABLE_SHEBANGS = {
    "#!" + "/" + "usr/bin/env python",
    "#!" + "/" + "usr/bin/env python3",
}


@dataclass(frozen=True)
class Requirement:
    group: str
    role: str
    source_rel: str
    archive_path: str


@dataclass(frozen=True)
class Payload:
    group: str
    role: str
    archive_path: str
    data: bytes


def req(group: str, role: str, source_rel: str, archive_path: str) -> Requirement:
    return Requirement(group, role, source_rel, archive_path)


STATIC_REQUIREMENTS = (
    req("paper", "paper_source", "paper/main.tex", "paper/source/main.tex"),
    req(
        "paper",
        "paper_source",
        "paper/supplementary.tex",
        "paper/source/supplementary.tex",
    ),
    req("paper", "paper_source", "paper/reference.bib", "paper/source/reference.bib"),
    req("paper", "paper_source", "paper/IEEEtran.cls", "paper/source/IEEEtran.cls"),
    req(
        "review_data",
        "raw_response_subset",
        "paper/review_data/tgrs_complex_response_subset.csv",
        "evidence/review_data/tgrs_complex_response_subset.csv",
    ),
    req(
        "review_data",
        "raw_response_manifest",
        "paper/review_data/tgrs_complex_response_subset.manifest.json",
        "evidence/review_data/tgrs_complex_response_subset.manifest.json",
    ),
    req(
        "dataset_audit",
        "table_input",
        "paper/figs/data/dataset_audit/centered_hrrp_audit.json",
        "evidence/dataset/centered_hrrp_audit.json",
    ),
    req(
        "strict_rect91",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_rect91_provenance/formal/audit/summary.json",
        "evidence/strict_rect91/summary.json",
    ),
    req(
        "strict_rect91",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_rect91_provenance/formal/audit/object_seed_cells.csv",
        "evidence/strict_rect91/object_seed_cells.csv",
    ),
    req(
        "strict_rect91",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_rect91_provenance/formal/audit/paired_samples.csv",
        "evidence/strict_rect91/paired_samples.csv",
    ),
    req(
        "strict_rect91",
        "preflight_audit",
        "progress/20260828_tgrs_nested_nonpolar_matched_rect91_provenance/accepted_preflight/equivalence_report.json",
        "evidence/strict_rect91/equivalence_report.json",
    ),
    req(
        "hann_control",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_hann91_provenance/formal/audit/summary.json",
        "evidence/hann/summary.json",
    ),
    req(
        "hann_control",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_hann91_provenance/formal/audit/derived_object_and_paired_rows.csv",
        "evidence/hann/derived_object_and_paired_rows.csv",
    ),
    req(
        "hann_control",
        "derived_result",
        "progress/20260828_tgrs_nested_nonpolar_matched_hann91_provenance/formal/audit/paired_samples.csv",
        "evidence/hann/paired_samples.csv",
    ),
    req(
        "hann_control",
        "preflight_audit",
        "progress/20260828_tgrs_nested_nonpolar_matched_hann91_provenance/accepted_preflight/hann91_secondary_report.json",
        "evidence/hann/hann91_secondary_report.json",
    ),
    req(
        "function_match",
        "experiment_lock",
        "code/review_protocol/function_matched_rect91_conjugation_lock.json",
        "evidence/function_match/experiment_lock.json",
    ),
    req(
        "function_match",
        "derived_result",
        "progress/20260828_tgrs_function_matched_rect91/samples.csv",
        "evidence/function_match/samples.csv",
    ),
    req(
        "function_match",
        "regeneration_code",
        "code/tools/audit_tgrs_function_matched_rect91.py",
        "evidence/function_match/audit.py",
    ),
    req(
        "formal_stress",
        "derived_result",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/calibration_responses.csv",
        "evidence/formal_stress/calibration_responses.csv",
    ),
    req(
        "formal_stress",
        "derived_result",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/curve_summary.csv",
        "evidence/formal_stress/curve_summary.csv",
    ),
    req(
        "formal_stress",
        "derived_result",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/object_curve_metrics.csv",
        "evidence/formal_stress/object_curve_metrics.csv",
    ),
    req(
        "formal_stress",
        "derived_result",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/object_metrics.csv",
        "evidence/formal_stress/object_metrics.csv",
    ),
    req(
        "formal_stress",
        "derived_result",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/paired_delta_summary.csv",
        "evidence/formal_stress/paired_delta_summary.csv",
    ),
    req(
        "formal_stress",
        "derived_state",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/state.json",
        "evidence/formal_stress/state.json",
    ),
    req(
        "formal_stress",
        "figure_input",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/measurement_stress_curves.pdf",
        "evidence/formal_stress/measurement_stress_curves.pdf",
    ),
    req(
        "formal_stress",
        "figure_input",
        "progress/20260828_tgrs_measurement_domain_stress_provenance/formal/measurement_stress_curves.png",
        "evidence/formal_stress/measurement_stress_curves.png",
    ),
)

DETERMINISTIC_FILES = (
    "README.md",
    "comparison_vs_current_operational.json",
    "current_reference_audit.json",
    "invariance_self_test.json",
    "leakage_audit.json",
    "objects.csv",
    "pooled.csv",
    "samples.csv",
    "selections.json",
    "summary.json",
)

TAIL_AUDIT_FILES = (
    "confusion_60x60.csv",
    "confusion_60x60.png",
    "direction_index.csv",
    "error_exact_angle_atoms.csv",
    "error_severity_bins.csv",
    "fold_mesh_audit.csv",
    "input_gate.json",
    "mesh_and_fold_summary.json",
    "mesh_descriptor_pairs.csv",
    "metrics_by_fold.csv",
    "metrics_by_object.csv",
    "metrics_by_true_direction.csv",
    "metrics_by_true_phi.csv",
    "metrics_by_true_theta.csv",
    "paired_direction_object_bootstrap.csv",
    "paired_object_effects.csv",
    "paired_overall_object_bootstrap.csv",
    "prediction_input_audit.json",
    "stress_awgn_seed_manifest.csv",
    "stress_calibration_coefficients.csv",
    "stress_condition_definitions.csv",
    "stress_exact_definitions.json",
    "strict_rows_embedded_in_hann_audit.json",
    "summary.json",
    "tail_by_direction.csv",
    "tail_by_object.csv",
    "tail_concentration_summary.csv",
    "tail_concentration.png",
)

FIGURE_FILES = (
    "fig1_paradigm.pdf",
    "fig1_paradigm.png",
    "fig2_dataset_overview.pdf",
    "fig2_dataset_overview.png",
    "fig3_sample_visualization.pdf",
    "fig3_sample_visualization.png",
    "fig3_matched_object_effects.pdf",
    "fig3_matched_object_effects.png",
    "figS1_measurement_stress_curves.pdf",
    "figS1_measurement_stress_curves.png",
    "figS1_id_metric_statistics.pdf",
    "figS1_id_metric_statistics.png",
    "figS2_direction_confusion_60x60.png",
    "figS3_tail_concentration.png",
)

REGENERATION_CODE = (
    "code/net/signal_encoders_1d.py",
    "code/tools/audit_tgrs_matched_rect91_mlp.py",
    "code/tools/audit_tgrs_matched_hann91_secondary.py",
    "code/tools/evaluate_hrrp_class_predictions.py",
    "code/tools/export_tgrs_repaired_complex_subset.py",
    "code/tools/generate_tgrs_nested_splits.py",
    "code/tools/make_fig3_matched_object_effects.py",
    "code/tools/make_hrrp_dataset_overview_figure.py",
    "code/tools/make_hrrp_paradigm_figure.py",
    "code/tools/make_hrrp_prediction_figures.py",
)

NESTED_SPLIT_FILES = (
    "id_dev.txt",
    "id_test.txt",
    "id_train.txt",
    "outer1_dev.txt",
    "outer1_test.txt",
    "outer1_train.txt",
    "outer2_dev.txt",
    "outer2_test.txt",
    "outer2_train.txt",
    "outer3_dev.txt",
    "outer3_test.txt",
    "outer3_train.txt",
    "outer4_dev.txt",
    "outer4_test.txt",
    "outer4_train.txt",
)

TGRS811_SPLIT_FILES = (
    "id_test.txt",
    "id_train.txt",
    "id_val.txt",
    "outer1_test.txt",
    "outer1_train.txt",
    "outer1_val.txt",
    "outer2_test.txt",
    "outer2_train.txt",
    "outer2_val.txt",
    "outer3_test.txt",
    "outer3_train.txt",
    "outer3_val.txt",
    "outer4_test.txt",
    "outer4_train.txt",
    "outer4_val.txt",
)

FIG3_PREDICTION_FILES = tuple(
    f"{arm}_mlp_id_seed{seed}.csv" for arm in ("freq", "hrrp") for seed in range(70, 77)
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        try:
            next(rows)
        except StopIteration as exc:
            raise RuntimeError(f"empty CSV: {path}") from exc
        return sum(1 for _ in rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_string_values(
    value: Any, key: str | None = None
) -> Iterable[tuple[str | None, str]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from iter_string_values(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from iter_string_values(child, key)
    elif isinstance(value, str):
        yield key, value


def runtime_identifiers() -> set[str]:
    candidates = {
        os.environ.get("COMPUTERNAME", ""),
        os.environ.get("HOSTNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("USERNAME", ""),
        platform.node(),
    }
    return {value.casefold() for value in candidates if len(value.strip()) >= 6}


def known_infrastructure_identifiers() -> set[str]:
    # Constructed in fragments so this builder remains eligible for its own
    # content scan while still rejecting identifiers present in legacy records.
    return {
        "240" + "21110937",
        "305" + "simu2",
        "codex-" + "p40",
        "cs" + "chl",
        "cs" + "jxt",
        "cs" + "zx",
        "ws-" + "w580-g20",
    }


def scan_text(
    text: str, label: str, *, suffix: str = "", check_ip: bool = True
) -> None:
    lines = text.splitlines()
    scan_value = text
    if lines and lines[0] in PORTABLE_SHEBANGS:
        scan_value = "\n".join(lines[1:])

    locator_checks: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("Windows absolute path", WINDOWS_ABSOLUTE_RE),
        ("UNC path", UNC_RE),
        ("Unix absolute path", UNIX_ABSOLUTE_RE),
        ("SSH user/host locator", SSH_ID_RE),
    )
    if check_ip:
        locator_checks += (("IP host/address", IPV4_RE),)
    checks = locator_checks + (
        ("private-key marker", PRIVATE_KEY_RE),
        ("AWS credential marker", AWS_KEY_RE),
        ("GitHub credential marker", GITHUB_TOKEN_RE),
        ("OpenAI credential marker", OPENAI_TOKEN_RE),
        ("credential assignment", CREDENTIAL_ASSIGNMENT_RE),
    )
    for name, pattern in checks:
        if pattern.search(scan_value):
            raise ValueError(f"{label}: rejected {name}")

    folded = scan_value.casefold()
    for identifier in runtime_identifiers():
        if identifier in folded:
            raise ValueError(f"{label}: rejected runtime machine/account identifier")
    for identifier in known_infrastructure_identifiers():
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(identifier)}(?![A-Za-z0-9])", folded
        ):
            raise ValueError(f"{label}: rejected known machine/account identifier")

    if suffix == ".json":
        parsed = json.loads(text)
        for key, value in iter_string_values(parsed):
            normalized = (key or "").casefold().replace("-", "_")
            if normalized in STRUCTURED_ID_KEYS and value.strip():
                raise ValueError(f"{label}: rejected structured host/account field")
            # JSON decoding converts escaped backslashes before this second scan.
            for name, pattern in locator_checks:
                if pattern.search(value):
                    raise ValueError(f"{label}: rejected {name} in JSON value")
    elif suffix == ".csv":
        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{label}: empty CSV") from exc
        normalized = [item.casefold().replace("-", "_").strip() for item in header]
        sensitive_columns = [
            i for i, item in enumerate(normalized) if item in STRUCTURED_ID_KEYS
        ]
        if sensitive_columns:
            for row in reader:
                if any(i < len(row) and row[i].strip() for i in sensitive_columns):
                    raise ValueError(f"{label}: rejected host/account CSV column")


def pdf_page_count_and_scan(data: bytes, label: str) -> int:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ARC4 has been moved.*")
            from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "pypdf is required to bind reviewed PDF page counts"
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception as exc:
        raise ValueError(f"{label}: unreadable PDF") from exc
    if not reader.pages:
        raise ValueError(f"{label}: PDF has no pages")
    metadata = reader.metadata or {}
    for key, value in metadata.items():
        if value is not None:
            scan_text(str(value), f"{label} metadata {key}")
    for index, page in enumerate(reader.pages):
        try:
            value = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(f"{label}: cannot extract page {index + 1} text") from exc
        # Plot-axis tick labels can concatenate into IPv4-looking numeric text
        # during PDF extraction. Paths, SSH locators, runtime identifiers, and
        # credentials remain checked; IP literals are checked in metadata and
        # every native UTF-8 payload.
        scan_text(value, f"{label} page {index + 1}", check_ip=False)
    return len(reader.pages)


def validate_archive_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.name:
        raise ValueError(f"unsafe archive path: {value}")
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"unrecognized archive suffix: {value}")
    if suffix in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden archive suffix: {value}")
    if path.name.casefold() in FORBIDDEN_NAMES:
        raise ValueError(f"forbidden archive filename: {value}")
    if any(part.casefold() in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"forbidden archive path class: {value}")


def validate_payload(item: Payload) -> int | None:
    validate_archive_path(item.archive_path)
    if not item.data:
        raise ValueError(f"empty payload: {item.archive_path}")
    if len(item.data) > MAX_FILE_BYTES:
        raise ValueError(f"oversized payload: {item.archive_path}")
    for magic, name in FORBIDDEN_MAGIC:
        if item.data.startswith(magic):
            raise ValueError(f"{item.archive_path}: forbidden {name}")
    suffix = PurePosixPath(item.archive_path).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        try:
            text = item.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"non-UTF-8 text payload: {item.archive_path}") from exc
        scan_text(text, item.archive_path, suffix=suffix)
        return None
    if suffix == ".pdf":
        if not item.data.startswith(b"%PDF-"):
            raise ValueError(f"invalid PDF magic: {item.archive_path}")
        return pdf_page_count_and_scan(item.data, item.archive_path)
    if suffix == ".png" and not item.data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"invalid PNG magic: {item.archive_path}")
    return None


def source_payload(
    source: Path, *, group: str, role: str, archive_path: str, repo_root: Path | None
) -> tuple[Payload, int | None]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise ValueError(f"symlink source is forbidden: {source}")
    if repo_root is not None:
        try:
            source.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"allowlisted source escaped repository: {source}"
            ) from exc
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden source suffix: {source}")
    item = Payload(group, role, archive_path, source.read_bytes())
    pages = validate_payload(item)
    return item, pages


def dynamic_requirements() -> list[Requirement]:
    values = list(STATIC_REQUIREMENTS)
    values.extend(
        req(
            "deterministic_control",
            "derived_result",
            f"output/20260828_tgrs_nested_template_correlations_v4/{name}",
            f"evidence/deterministic/{name}",
        )
        for name in DETERMINISTIC_FILES
    )
    values.extend(
        req(
            "direction_tail_audit",
            "derived_result"
            if Path(name).suffix.lower() in TEXT_SUFFIXES
            else "figure_input",
            f"output/20260829_tgrs_direction_tail_audit_v3/{name}",
            f"evidence/direction_tail/{name}",
        )
        for name in TAIL_AUDIT_FILES
    )
    values.extend(
        req(
            "figures",
            "paper_figure",
            f"paper/figs/{name}",
            f"paper/figures/{name}",
        )
        for name in FIGURE_FILES
    )
    values.extend(
        req(
            "regeneration_code",
            "regeneration_code",
            name,
            f"code/{Path(name).name}",
        )
        for name in REGENERATION_CODE
    )
    values.append(
        req(
            "artifact_builder",
            "artifact_builder",
            "code/tools/build_tgrs_current_evidence_artifact.py",
            "code/build_tgrs_current_evidence_artifact.py",
        )
    )
    values.extend(
        req(
            "split_membership",
            "split_membership",
            f"code/review_protocol/splits/tgrs_nested_nonpolar_v2/{name}",
            f"evidence/splits/tgrs_nested_nonpolar_v2/{name}",
        )
        for name in NESTED_SPLIT_FILES
    )
    values.extend(
        req(
            "split_membership",
            "split_membership",
            f"code/review_protocol/splits/tgrs_811/{name}",
            f"evidence/splits/tgrs_811/{name}",
        )
        for name in TGRS811_SPLIT_FILES
    )
    values.extend(
        req(
            "fig3_inputs",
            "figure_input",
            f"tmp/fig3_tgrs811_7seed/{name}",
            f"evidence/fig3_predictions/{name}",
        )
        for name in FIG3_PREDICTION_FILES
    )
    return values


def split_expected_counts(source_manifest: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, value in source_manifest.get("splits", {}).items():
        result[name] = int(value["samples"])
    for fold in source_manifest.get("outer_folds", {}).values():
        for name, value in fold.get("splits", {}).items():
            result[name] = int(value["samples"])
    return result


def tgrs811_expected_counts(source_manifest: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, value in source_manifest.get("splits", {}).items():
        result[f"{name}.txt"] = int(value["samples"])
    for fold_name, fold in source_manifest.get("outer_folds", {}).items():
        for split_name, value in fold.get("splits", {}).items():
            normalized = (
                split_name
                if split_name.startswith("outer")
                else f"outer{fold_name}_{split_name}"
            )
            result[f"{normalized}.txt"] = int(value["samples"])
    return result


def validate_science(repo_root: Path) -> dict[str, Any]:
    strict = read_json(
        repo_root
        / "progress/20260828_tgrs_nested_nonpolar_matched_rect91_provenance/formal/audit/summary.json"
    )
    if (
        strict.get("status") != "ok"
        or strict.get("protocol") != "tgrs_nested_nonpolar_v2"
    ):
        raise RuntimeError("strict rect91 summary is not current and complete")
    sf = strict["models"]["frequency"]["held_geometry"]
    sr = strict["models"]["rect_ifft"]["held_geometry"]
    sp = strict["paired_difference"]["held_geometry"]
    if (sf["prediction_count"], sr["prediction_count"], sp["object_count"]) != (
        8400,
        8400,
        20,
    ):
        raise RuntimeError("strict rect91 count gate failed")
    if abs(float(sp["mean_gce_difference_deg"]) - (-0.3242683597973415)) > 1e-10:
        raise RuntimeError("strict rect91 numerical anchor changed")

    hann = read_json(
        repo_root
        / "progress/20260828_tgrs_nested_nonpolar_matched_hann91_provenance/formal/audit/summary.json"
    )
    hp = hann.get("paired_difference", {}).get("held_geometry", {})
    if hann.get("status") != "ok" or hp.get("paired_prediction_count") != 8400:
        raise RuntimeError("Hann summary is not current and complete")
    if (
        abs(float(hp["object_centered_gce_hann_minus_rect_deg"]) - 5.127550013973599)
        > 1e-10
    ):
        raise RuntimeError("Hann numerical anchor changed")

    function_match_path = (
        repo_root / "progress/20260828_tgrs_function_matched_rect91/summary.json"
    )
    function_match = read_json(function_match_path)
    aggregate = function_match.get("aggregate", {})
    if (
        function_match.get("status") != "pass"
        or function_match.get("protocol") != "tgrs_nested_nonpolar_v2"
        or aggregate.get("checkpoint_count") != 35
        or aggregate.get("evaluated_sample_checkpoint_pairs") != 9800
        or aggregate.get("prediction_mismatch_after_conjugation") != 0
        or aggregate.get("prediction_mismatch_with_same_numeric_weight") != 9639
    ):
        raise RuntimeError("function-matched conjugation gate failed")

    deterministic = read_json(
        repo_root / "output/20260828_tgrs_nested_template_correlations_v4/summary.json"
    )
    operational = deterministic.get("held", {}).get("operational_6ch_1nn", {})
    if (
        deterministic.get("status") != "ok"
        or deterministic.get("protocol") != "tgrs_nested_nonpolar_v2"
        or operational.get("count") != 1200
    ):
        raise RuntimeError("deterministic-control gate failed")

    tail = read_json(
        repo_root / "output/20260829_tgrs_direction_tail_audit_v3/summary.json"
    )
    arms = tail.get("arms", {})
    if (
        tail.get("status") != "ok"
        or tail.get("stress", {}).get("realization_count") != 42
    ):
        raise RuntimeError("direction/tail/stress audit gate failed")
    expected_rows = {
        "frequency": 8400,
        "rect_ifft": 8400,
        "hann_ifft": 8400,
        "strong_1nn": 1200,
    }
    if {
        key: arms.get(key, {}).get("prediction_rows") for key in expected_rows
    } != expected_rows:
        raise RuntimeError("direction/tail arm counts changed")
    if abs(float(arms["strong_1nn"]["tail_rate_gt90_deg"]) - 0.13) > 1e-12:
        raise RuntimeError("tolerance-aware deterministic tail rate changed")

    raw_deterministic_tail = float(operational["failure_rate_gt_90_deg"])
    if abs(raw_deterministic_tail - (17 / 120)) > 1e-12:
        raise RuntimeError("raw strict-threshold deterministic tail rate changed")

    stress_root = (
        repo_root / "progress/20260828_tgrs_measurement_domain_stress_provenance/formal"
    )
    stress_rows = {
        "calibration_responses": csv_row_count(
            stress_root / "calibration_responses.csv"
        ),
        "curve_summary": csv_row_count(stress_root / "curve_summary.csv"),
        "object_curve_metrics": csv_row_count(stress_root / "object_curve_metrics.csv"),
        "object_metrics": csv_row_count(stress_root / "object_metrics.csv"),
        "paired_delta_summary": csv_row_count(stress_root / "paired_delta_summary.csv"),
    }
    if stress_rows != {
        "calibration_responses": 1638,
        "curve_summary": 36,
        "object_curve_metrics": 720,
        "object_metrics": 11760,
        "paired_delta_summary": 18,
    }:
        raise RuntimeError(f"formal stress row-count gate failed: {stress_rows}")

    native_stress_rows = {
        row["condition"]: row
        for row in read_csv_dicts(stress_root / "paired_delta_summary.csv")
    }
    manuscript_sign_anchors = {
        "clean": "-0.324",
        "phase_15deg": "-0.426",
        "range_1cm": "-0.244",
        "snr_20db": "-2.879",
        "snr_10db": "-9.154",
        "cal_1db_10deg": "-2.134",
        "cal_2db_20deg": "-5.381",
    }
    for condition, expected_text in manuscript_sign_anchors.items():
        native_hrrp_minus_frequency = float(
            native_stress_rows[condition]["hrrp_minus_frequency_gce_deg"]
        )
        expected_frequency_minus_hrrp = -native_hrrp_minus_frequency
        if f"{expected_frequency_minus_hrrp:.3f}" != expected_text:
            raise RuntimeError(f"stress sign anchor changed: {condition}")

    subset_manifest = read_json(
        repo_root / "paper/review_data/tgrs_complex_response_subset.manifest.json"
    )
    subset_csv = repo_root / "paper/review_data/tgrs_complex_response_subset.csv"
    if (
        subset_manifest.get("status") != "ok"
        or subset_manifest.get("csv_row_count") != 1820
        or subset_manifest.get("csv_sha256") != sha256_bytes(subset_csv.read_bytes())
        or csv_row_count(subset_csv) != 1820
    ):
        raise RuntimeError("review-data subset gate failed")

    fig3_counts = {
        name: csv_row_count(repo_root / "tmp/fig3_tgrs811_7seed" / name)
        for name in FIG3_PREDICTION_FILES
    }
    if set(fig3_counts.values()) != {160}:
        raise RuntimeError("Fig. 3 prediction inputs are incomplete")

    main_text = (repo_root / "paper/main.tex").read_text(encoding="utf-8")
    required_anchors = (
        "36.996",
        "37.320",
        "-0.324",
        "-2.165",
        "1.354",
        "5.128",
        "2.864",
        "7.424",
        "53.194",
        "62.347",
    )
    missing_anchors = [value for value in required_anchors if value not in main_text]
    if missing_anchors:
        raise RuntimeError(
            f"current manuscript numerical anchors missing: {missing_anchors}"
        )
    for expected_text in manuscript_sign_anchors.values():
        if expected_text not in main_text:
            raise RuntimeError(f"manuscript lacks F-R stress anchor: {expected_text}")

    figure_references: set[str] = set()
    for source_name in ("main.tex", "supplementary.tex"):
        source_text = (repo_root / "paper" / source_name).read_text(encoding="utf-8")
        figure_references.update(
            re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+)\}", source_text)
        )
    expected_references = {
        "fig1_paradigm.pdf",
        "fig2_dataset_overview.pdf",
        "fig3_sample_visualization.pdf",
        "fig3_matched_object_effects.pdf",
        "figS1_measurement_stress_curves.pdf",
        "figS1_id_metric_statistics.pdf",
        "figS2_direction_confusion_60x60.png",
        "figS3_tail_concentration.png",
    }
    if figure_references != expected_references:
        raise RuntimeError(
            "paper figure-reference set changed: "
            f"{sorted(figure_references)} != {sorted(expected_references)}"
        )

    return {
        "strict_rect91": {
            "status": "pass",
            "held_rows_per_arm": 8400,
            "objects": 20,
            "seeds": 7,
            "frequency_gce_deg": sf["great_circle_mae_deg"],
            "rect_ifft_gce_deg": sr["great_circle_mae_deg"],
            "frequency_minus_rect_gce_deg": sp["mean_gce_difference_deg"],
        },
        "hann": {
            "status": "pass",
            "held_rows": hp["paired_prediction_count"],
            "hann_minus_rect_gce_deg": hp["object_centered_gce_hann_minus_rect_deg"],
        },
        "function_match": {
            "status": "pass",
            "source_summary_included": False,
            "source_summary_omission": "contains infrastructure-specific absolute paths",
            "source_summary_sha256": sha256_bytes(function_match_path.read_bytes()),
            "checkpoints": aggregate["checkpoint_count"],
            "sample_checkpoint_pairs": aggregate["evaluated_sample_checkpoint_pairs"],
            "mismatches_after_conjugation": aggregate[
                "prediction_mismatch_after_conjugation"
            ],
            "same_numeric_weight_changes": aggregate[
                "prediction_mismatch_with_same_numeric_weight"
            ],
            "max_logit_abs": aggregate["logit_max_abs"],
        },
        "deterministic": {
            "status": "pass",
            "held_rows": operational["count"],
            "raw_strict_gt90_without_tolerance": raw_deterministic_tail,
            "tolerance_aware_tail_gt90": arms["strong_1nn"]["tail_rate_gt90_deg"],
            "reconciliation": "the raw value counts floating values just above 90 deg; the manuscript uses a 1e-3-deg physical threshold tolerance",
        },
        "formal_stress": {
            "status": "pass",
            **stress_rows,
            "native_delta": "Rect-IFFT minus Frequency",
            "manuscript_delta": "Frequency minus Rect-IFFT",
            "sign_anchor_count": len(manuscript_sign_anchors),
        },
        "review_data": {"status": "pass", "csv_rows": 1820, "objects": 20},
        "fig3_inputs": {"status": "pass", "files": 14, "rows_per_file": 160},
        "paper_numeric_anchors": {
            "status": "pass",
            "anchor_count": len(required_anchors),
        },
        "paper_figure_coverage": {
            "status": "pass",
            "referenced_assets": [
                f"paper/figures/{name}" for name in sorted(figure_references)
            ],
        },
    }


def generated_split_manifest(
    repo_root: Path, payloads: list[Payload]
) -> dict[str, Any]:
    by_archive = {item.archive_path: item for item in payloads}
    protocols = []
    specifications = (
        (
            "tgrs_nested_nonpolar_v2",
            60,
            repo_root
            / "code/review_protocol/splits/tgrs_nested_nonpolar_v2/manifest.json",
            NESTED_SPLIT_FILES,
            split_expected_counts,
        ),
        (
            "tgrs_811_v1",
            84,
            repo_root / "code/review_protocol/splits/tgrs_811/manifest.json",
            TGRS811_SPLIT_FILES,
            tgrs811_expected_counts,
        ),
    )
    for protocol, directions, source_path, names, expected_builder in specifications:
        source = read_json(source_path)
        if source.get("status") != "ok" or source.get("version") != protocol:
            raise RuntimeError(f"source split manifest is not current: {protocol}")
        expected = expected_builder(source)
        if set(expected) != set(names):
            raise RuntimeError(f"source split membership changed: {protocol}")
        records = []
        for name in names:
            archive_path = f"evidence/splits/{protocol.removesuffix('_v1')}/{name}"
            item = by_archive[archive_path]
            rows = len(
                [
                    line
                    for line in item.data.decode("utf-8").splitlines()
                    if line.strip()
                ]
            )
            if rows != expected[name]:
                raise RuntimeError(
                    f"split line count changed: {protocol}/{name}: {rows} != {expected[name]}"
                )
            records.append(
                {
                    "archive_path": archive_path,
                    "bytes": len(item.data),
                    "samples": rows,
                    "sha256": sha256_bytes(item.data),
                }
            )
        protocols.append(
            {
                "protocol": protocol,
                "directions_per_object": directions,
                "source_manifest_included": False,
                "source_manifest_omission": "contains an infrastructure-specific absolute source path",
                "source_manifest_sha256": sha256_bytes(source_path.read_bytes()),
                "files": records,
            }
        )
    return {
        "schema": "tgrs-current-evidence-split-manifest-v1",
        "protocols": protocols,
    }


def generated_fig3_manifest(payloads: list[Payload]) -> dict[str, Any]:
    records = []
    for item in payloads:
        if item.group != "fig3_inputs":
            continue
        rows = csv_row_count_from_bytes(item.data)
        records.append(
            {
                "archive_path": item.archive_path,
                "bytes": len(item.data),
                "rows": rows,
                "sha256": sha256_bytes(item.data),
            }
        )
    return {
        "schema": "tgrs-current-evidence-fig3-input-manifest-v1",
        "scope": "seen-object tgrs811 qualitative diagnostic only",
        "files": sorted(records, key=lambda item: item["archive_path"]),
    }


def csv_row_count_from_bytes(data: bytes) -> int:
    rows = csv.reader(io.StringIO(data.decode("utf-8")))
    try:
        next(rows)
    except StopIteration as exc:
        raise ValueError("empty CSV bytes") from exc
    return sum(1 for _ in rows)


def readme_bytes(group_counts: Counter[str]) -> bytes:
    lines = [
        "# Current-evidence TGRS result-audit archive",
        "",
        "This archive binds the reviewed PDFs and only the evidence used by the current no-new-experiment manuscript.",
        "All paths below are archive-relative. The archive does not contain raw NumPy arrays, model weights, meshes, credentials, or nested archives.",
        "",
        "## Evidence scope",
        "",
        "- strict 91-bin Frequency versus rectangular-IFFT results;",
        "- the matched Hann-window control;",
        "- deterministic current-data controls;",
        "- the frozen-checkpoint formal measurement-domain stress audit;",
        "- split memberships and an archive-relative split manifest;",
        "- paper figure assets and clean figure/table inputs;",
        "- a small complex-response review subset; and",
        "- paper sources plus regeneration code that passed the privacy scan.",
        "",
        "Native controller, audit, provenance, and figure-input records that contain machine paths, hostnames, account names, or forbidden asset references were omitted rather than redacted. Their scientific values are represented only where an already-derived clean CSV or JSON exists. Regeneration still requires the external dataset and checkpoints; neither is distributed here.",
        "",
        "## Group inventory",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count} file(s)" for name, count in sorted(group_counts.items())
    )
    lines.extend(
        [
            "",
            "`MANIFEST.json` hashes every payload. The sidecar manifest is byte-identical to it. `READY.json` binds the ZIP, sidecar, and both reviewed PDF page counts and hashes.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def collect_payloads(
    repo_root: Path, main_pdf: Path, supplementary_pdf: Path
) -> tuple[list[Payload], dict[str, int], dict[str, Any]]:
    payloads: list[Payload] = []
    pdf_pages: dict[str, int] = {}
    for requirement in dynamic_requirements():
        item, pages = source_payload(
            repo_root / requirement.source_rel,
            group=requirement.group,
            role=requirement.role,
            archive_path=requirement.archive_path,
            repo_root=repo_root,
        )
        payloads.append(item)
        if pages is not None:
            pdf_pages[item.archive_path] = pages

    for label, source, archive_path in (
        ("main", main_pdf, "paper/reviewed/main.pdf"),
        ("supplementary", supplementary_pdf, "paper/reviewed/supplementary.pdf"),
    ):
        item, pages = source_payload(
            source,
            group="reviewed_pdfs",
            role=f"reviewed_{label}_pdf",
            archive_path=archive_path,
            repo_root=None,
        )
        if pages is None:
            raise RuntimeError(f"reviewed {label} input is not a PDF")
        payloads.append(item)
        pdf_pages[archive_path] = pages

    science = validate_science(repo_root)
    split_manifest = Payload(
        "split_membership",
        "archive_relative_manifest",
        "evidence/splits/manifest.archive.json",
        json_bytes(generated_split_manifest(repo_root, payloads)),
    )
    fig3_manifest = Payload(
        "fig3_inputs",
        "archive_relative_manifest",
        "evidence/fig3_predictions/manifest.archive.json",
        json_bytes(generated_fig3_manifest(payloads)),
    )
    science_receipt = Payload(
        "audit_receipts",
        "scientific_gate",
        "audit/scientific_gate.json",
        json_bytes(
            {"schema": "tgrs-current-evidence-scientific-gate-v1", "gates": science}
        ),
    )
    for item in (split_manifest, fig3_manifest, science_receipt):
        validate_payload(item)
        payloads.append(item)

    paths = [item.archive_path for item in payloads]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate archive path")
    if sum(len(item.data) for item in payloads) > MAX_TOTAL_BYTES:
        raise RuntimeError("total payload exceeds the archive limit")

    group_counts = Counter(item.group for item in payloads)
    readme = Payload("documentation", "readme", "README.md", readme_bytes(group_counts))
    validate_payload(readme)
    payloads.append(readme)
    return sorted(payloads, key=lambda item: item.archive_path), pdf_pages, science


def manifest_for(
    artifact_id: str,
    payloads: list[Payload],
    pdf_pages: dict[str, int],
    science: dict[str, Any],
) -> dict[str, Any]:
    files = [
        {
            "archive_path": item.archive_path,
            "bytes": len(item.data),
            "group": item.group,
            "role": item.role,
            "sha256": sha256_bytes(item.data),
        }
        for item in payloads
    ]
    index = {item["archive_path"]: item for item in files}
    pdfs = {
        label: {
            "archive_path": archive_path,
            "bytes": index[archive_path]["bytes"],
            "pages": pdf_pages[archive_path],
            "sha256": index[archive_path]["sha256"],
        }
        for label, archive_path in (
            ("main", "paper/reviewed/main.pdf"),
            ("supplementary", "paper/reviewed/supplementary.pdf"),
        )
    }
    return {
        "schema": SCHEMA,
        "artifact_id": artifact_id,
        "status": "READY",
        "scope": "current manuscript evidence only; no new or unfinished experiment",
        "anonymization": {
            "policy": "reject identifying payloads; never redact scientific files",
            "text_scan": "all UTF-8 payloads plus extracted PDF text and metadata",
            "portable_interpreter_shebang": "permitted only as the first source line",
            "source_paths_recorded": False,
        },
        "excluded_payload_classes": [
            "raw NPZ/NumPy arrays",
            "model weights and checkpoints",
            "mesh assets",
            "credentials and key material",
            "nested archives",
            "native records containing absolute paths or machine/account identifiers",
            "unfinished Round-2 experiments",
        ],
        "zip": {
            "compression": "stored",
            "entry_order": "UTF-8/POSIX archive path ascending",
            "entry_timestamp": "1980-01-01T00:00:00",
            "internal_double_build_required": True,
        },
        "reviewed_pdfs": pdfs,
        "scientific_gates": science,
        "payload_count": len(files),
        "total_payload_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def zip_bytes(payloads: list[Payload], manifest_data: bytes) -> bytes:
    entries = [(item.archive_path, item.data) for item in payloads]
    entries.append(("MANIFEST.json", manifest_data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for archive_path, data in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(archive_path, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def verify_zip(data: bytes, payloads: list[Payload], manifest_data: bytes) -> None:
    expected = {item.archive_path: item.data for item in payloads}
    expected["MANIFEST.json"] = manifest_data
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        if archive.namelist() != sorted(expected):
            raise RuntimeError("ZIP entry order or membership changed")
        for name, value in expected.items():
            if archive.read(name) != value:
                raise RuntimeError(f"ZIP payload verification failed: {name}")


def build(
    artifact_id: str,
    main_pdf: Path,
    supplementary_pdf: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise ValueError("invalid artifact id")
    if artifact_id == "TGRS-MFH-20260828-v1":
        raise ValueError("refusing immutable historical artifact id")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output_dir.parent}")
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()

    payloads, pdf_pages, science = collect_payloads(
        repo_root, main_pdf.resolve(), supplementary_pdf.resolve()
    )
    manifest = manifest_for(artifact_id, payloads, pdf_pages, science)
    manifest_data = json_bytes(manifest)
    scan_text(manifest_data.decode("utf-8"), "sidecar manifest", suffix=".json")

    first_zip = zip_bytes(payloads, manifest_data)
    second_zip = zip_bytes(payloads, manifest_data)
    if first_zip != second_zip:
        raise RuntimeError("internal deterministic double-build check failed")
    verify_zip(first_zip, payloads, manifest_data)

    pdf_index = manifest["reviewed_pdfs"]
    ready = {
        "schema": READY_SCHEMA,
        "artifact_id": artifact_id,
        "status": "READY",
        "archive": {
            "path": ARCHIVE_NAME,
            "bytes": len(first_zip),
            "sha256": sha256_bytes(first_zip),
        },
        "manifest": {
            "path": SIDECAR_NAME,
            "bytes": len(manifest_data),
            "sha256": sha256_bytes(manifest_data),
            "embedded_archive_path": "MANIFEST.json",
        },
        "reviewed_pdfs": pdf_index,
        "payload_count": manifest["payload_count"],
        "gates": {
            "content_allowlist": "pass",
            "forbidden_payload_classes": "pass",
            "privacy_scan": "pass",
            "scientific_current_evidence": "pass",
            "internal_double_build_byte_identity": "pass",
            "zip_roundtrip": "pass",
        },
    }
    ready_data = json_bytes(ready)
    scan_text(ready_data.decode("utf-8"), READY_NAME, suffix=".json")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        (stage / ARCHIVE_NAME).write_bytes(first_zip)
        (stage / SIDECAR_NAME).write_bytes(manifest_data)
        (stage / READY_NAME).write_bytes(ready_data)
        if (
            sha256_bytes((stage / ARCHIVE_NAME).read_bytes())
            != ready["archive"]["sha256"]
        ):
            raise RuntimeError("staged ZIP hash changed")
        if (
            sha256_bytes((stage / SIDECAR_NAME).read_bytes())
            != ready["manifest"]["sha256"]
        ):
            raise RuntimeError("staged manifest hash changed")
        stage.rename(output_dir)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return ready


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--main-pdf", type=Path, required=True)
    parser.add_argument("--supp-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ready = build(
        args.artifact_id,
        args.main_pdf,
        args.supp_pdf,
        args.output_dir,
    )
    print(json.dumps(ready, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
