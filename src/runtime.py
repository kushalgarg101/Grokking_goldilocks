"""Runtime helpers: filesystem, config parsing, W&B setup, and argument checks."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

from .data import TaskData
from .probes import EpochMetrics


def save_metrics_csv(history: list[EpochMetrics], path: Path) -> None:
    """Persist epoch metrics as a flat CSV table."""
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def resolve_device(device_arg: str) -> torch.device:
    """Resolve explicit device or auto-select CUDA/MPS/CPU."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def can_write_probe(path: Path) -> tuple[bool, str | None]:
    """Check write permission by creating and deleting a small probe file."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / ".probe_write_check"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        return True, None
    except Exception as exc:
        return False, f"{path}: {exc}"


def prepare_wandb_paths(wandb_root: Path) -> dict[str, Path]:
    """Create W&B working directories and pin temporary paths to a writable root."""
    wandb_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "run": wandb_root / "run",
        "cache": wandb_root / "cache",
        "config": wandb_root / "config",
        "data": wandb_root / "data",
        "tmp": wandb_root / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    run_dir = str(paths["run"].resolve())
    cache_dir = str(paths["cache"].resolve())
    config_dir = str(paths["config"].resolve())
    data_dir = str(paths["data"].resolve())
    tmp_dir = str(paths["tmp"].resolve())

    os.environ["WANDB_DIR"] = run_dir
    os.environ["WANDB_CACHE_DIR"] = cache_dir
    os.environ["WANDB_CONFIG_DIR"] = config_dir
    os.environ["WANDB_DATA_DIR"] = data_dir
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    os.environ.setdefault("WANDB__DISABLE_SERVICE", "true")
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    tempfile.tempdir = tmp_dir
    return paths


def log_figure_to_wandb(
    wandb_run: Any,
    wandb: Any,
    key: str,
    fig: plt.Figure,
    step: int,
    media_dir: Path,
    stem: str,
) -> None:
    """Save a matplotlib figure to disk and log it as a W&B image artifact."""
    media_dir.mkdir(parents=True, exist_ok=True)
    figure_path = media_dir / f"{stem}_epoch_{int(step):06d}.png"
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    wandb_run.log({key: wandb.Image(str(figure_path))}, step=step)


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    """Initialize W&B safely, falling back to local-only mode on permission issues."""
    if args.wandb_mode == "disabled":
        return None, None
    wandb_paths = prepare_wandb_paths(Path(args.wandb_dir))
    can_write_tmp, tmp_reason = can_write_probe(wandb_paths["tmp"])
    if not can_write_tmp:
        print(f"wandb skipped: cannot write to temporary directory ({tmp_reason}).")
        return None, None

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        can_write_logs, logs_reason = can_write_probe(Path(local_appdata) / "wandb" / "logs")
        if not can_write_logs:
            print(
                "wandb skipped: cannot write to required local wandb logs directory "
                f"({logs_reason})."
            )
            return None, None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb is not installed. Continuing without wandb logging.")
        return None, None

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config=config,
            mode=args.wandb_mode,
            dir=str(wandb_paths["run"]),
        )
        return wandb, run
    except Exception as exc:
        print(
            "wandb initialization failed: "
            f"{exc.__class__.__name__}: {exc}. Continuing without wandb logging."
        )
        return None, None


def parse_simple_yaml_config(path: str) -> dict[str, Any]:
    """Parse a flat key:value YAML file into parser-compatible defaults."""
    config: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line {lineno}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            key = key.strip().replace("-", "_")
            value = value.strip()
            if value == "":
                raise ValueError(f"Missing value at config line {lineno}: {raw_line.rstrip()}")

            low = value.lower()
            if low in ("true", "false"):
                parsed: Any = (low == "true")
            elif low in ("null", "none"):
                parsed = None
            else:
                parsed = None
                try:
                    parsed = ast.literal_eval(value)
                except Exception:
                    try:
                        if any(ch in low for ch in (".", "e")):
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except Exception:
                        parsed = value.strip("'\"")
            config[key] = parsed
    return config


def sample_from_dataset(
    dataset: TensorDataset,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a deterministic subset from a TensorDataset."""
    x, y = dataset.tensors
    n = min(sample_size, len(dataset))
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[:n]
    return x[idx], y[idx]


def build_token_labels(sample_tokens: np.ndarray, task_metadata: dict[str, Any]) -> list[str]:
    """Generate human-readable token labels for probe visualizations."""
    task_name = str(task_metadata.get("task", ""))
    if task_name == "modular_division" and len(sample_tokens) == 4:
        return ["a", "/", "b", "="]

    labels: list[str] = []
    special = task_metadata.get("special_tokens", {})
    slash_tok = special.get("slash")
    equals_tok = special.get("equals")
    for t in sample_tokens.tolist():
        if slash_tok is not None and t == slash_tok:
            labels.append("/")
        elif equals_tok is not None and t == equals_tok:
            labels.append("=")
        else:
            labels.append(str(int(t)))
    return labels


def build_token_context(sample_tokens: np.ndarray, task_metadata: dict[str, Any]) -> str | None:
    """Build a compact textual rendering of a probe sequence."""
    special = task_metadata.get("special_tokens", {})
    slash_tok = special.get("slash")
    equals_tok = special.get("equals")
    rendered: list[str] = []
    for t in sample_tokens.tolist():
        if slash_tok is not None and t == slash_tok:
            rendered.append("/")
        elif equals_tok is not None and t == equals_tok:
            rendered.append("=")
        else:
            rendered.append(str(int(t)))
    return "Example: [" + ", ".join(rendered) + "]"


def remove_legacy_artifacts(output_dir: Path) -> None:
    """Delete obsolete plot artifacts that can confuse result interpretation."""
    for name in ("attention_cityscape.png", "pca_donut.png"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def validate_args(args: argparse.Namespace) -> None:
    """Validate runtime arguments that have strict constraints."""
    if args.full_batch and args.shock_enabled:
        raise ValueError("`--full-batch` and `--shock-enabled` are mutually exclusive.")
    if args.hessian_every <= 0:
        raise ValueError("--hessian-every must be > 0")
    if args.plot_every <= 0:
        raise ValueError("--plot-every must be > 0")
    if args.wandb_log_every <= 0:
        raise ValueError("--wandb-log-every must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.shock_batch_size <= 0:
        raise ValueError("--shock-batch-size must be > 0")
    if args.shock_duration_epochs <= 0:
        raise ValueError("--shock-duration-epochs must be > 0")
    if args.hessian_probe_size <= 0:
        raise ValueError("--hessian-probe-size must be > 0")
    if args.probe_size <= 0:
        raise ValueError("--probe-size must be > 0")
    if args.spectrum_before_epoch <= 0:
        raise ValueError("--spectrum-before-epoch must be > 0")
    if args.spectrum_during_epoch <= 0:
        raise ValueError("--spectrum-during-epoch must be > 0")


def build_run_config(
    args: argparse.Namespace,
    task_data: TaskData,
    base_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build run config dictionary for W&B metadata and summaries."""
    return {
        "prime": args.prime,
        "train_fraction": args.train_fraction,
        "layers": args.layers,
        "heads": args.heads,
        "d_model": args.d_model,
        "mlp_mult": args.mlp_mult,
        "task": args.task,
        "task_metadata": task_data.metadata,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": base_batch_size,
        "full_batch": args.full_batch,
        "shock_enabled": args.shock_enabled,
        "shock_batch_size": args.shock_batch_size,
        "shock_duration_epochs": args.shock_duration_epochs,
        "hessian_every": args.hessian_every,
        "hessian_iters": args.hessian_iters,
        "spectrum_before_epoch": args.spectrum_before_epoch,
        "spectrum_during_epoch": args.spectrum_during_epoch,
        "extra_media": args.extra_media,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
    }
