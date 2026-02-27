"""Training orchestration and CLI parsing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_task_data, set_seed
from .model import TinyDecoderTransformer
from .plotting import (
    align_landscape_basis_to_progress,
    init_landscape_basis,
    make_attention_snapshot_figure,
    make_dashboard_figure,
    make_loss_landscape_figure,
    make_spectrum_figure,
    project_to_landscape_coordinates,
)
from .probes import (
    EpochMetrics,
    attention_gradient_norms,
    attention_probe_metrics,
    embedding_fourier_metrics,
    estimate_hessian_top_eigenvalue,
    evaluate,
    get_layer0_q_spectrum,
    infer_stage,
    lr_scale_for_epoch,
    model_svd_stats,
    predict_grokking_epoch,
)
from .runtime import (
    build_run_config,
    build_token_context,
    build_token_labels,
    log_figure_to_wandb,
    maybe_init_wandb,
    parse_simple_yaml_config,
    remove_legacy_artifacts,
    resolve_device,
    sample_from_dataset,
    save_metrics_csv,
    validate_args,
)


def train(args: argparse.Namespace) -> None:
    """Run one full experiment: train, evaluate, plot artifacts, and write summaries."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_artifacts(output_dir)

    task_data = build_task_data(args)
    train_ds = task_data.train_ds
    test_ds = task_data.test_ds
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Dataset split produced empty train or test set. Adjust task size/split.")
    base_batch_size = len(train_ds) if args.full_batch else args.batch_size
    eval_batch_size = min(args.batch_size, len(test_ds))
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyDecoderTransformer(
        vocab_size=task_data.vocab_size,
        num_classes=task_data.num_classes,
        seq_len=task_data.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
        mlp_mult=args.mlp_mult,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss()

    config = build_run_config(
        args=args,
        task_data=task_data,
        base_batch_size=base_batch_size,
        device=device,
    )
    wandb, wandb_run = maybe_init_wandb(args, config)

    probe_tokens, _ = sample_from_dataset(
        test_ds,
        sample_size=min(args.probe_size, len(test_ds)),
        seed=args.seed + 17,
    )
    hessian_tokens, hessian_targets = sample_from_dataset(
        train_ds,
        sample_size=min(args.hessian_probe_size, len(train_ds)),
        seed=args.seed + 31,
    )
    probe_batch = probe_tokens
    probe_labels = build_token_labels(probe_batch[0].cpu().numpy(), task_data.metadata)
    probe_context = build_token_context(probe_batch[0].cpu().numpy(), task_data.metadata)
    history: list[EpochMetrics] = []
    epoch_grad_norm_history: list[float] = []
    max_rank_seen = -float("inf")
    last_hessian_lambda = float("nan")
    landscape_enabled = args.extra_media != "none"
    landscape_theta0: list[torch.Tensor] = []
    landscape_dir1: list[torch.Tensor] = []
    landscape_dir2: list[torch.Tensor] = []
    landscape_basis_aligned = False
    landscape_trajectory: list[tuple[int, float, float]] = []
    wandb_media_dir = output_dir / "wandb_media"
    if landscape_enabled:
        landscape_theta0, landscape_dir1, landscape_dir2 = init_landscape_basis(model)
    spectrum_snapshots: dict[str, tuple[int, np.ndarray]] = {}
    spectrum_before_target = min(args.spectrum_before_epoch, args.epochs)
    spectrum_during_target = min(args.spectrum_during_epoch, args.epochs)

    shock_start_epoch: int | None = None
    shock_end_epoch: int | None = None
    shock_trigger_reason = "disabled"
    if args.shock_enabled and args.shock_start_epoch > 0:
        shock_start_epoch = args.shock_start_epoch
        shock_end_epoch = shock_start_epoch + args.shock_duration_epochs - 1
        shock_trigger_reason = "manual_epoch"
    elif args.shock_enabled:
        shock_trigger_reason = "armed_auto"

    def make_train_loader(epoch_batch_size: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            train_ds,
            batch_size=epoch_batch_size,
            shuffle=(not args.full_batch),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    epoch_iter = range(1, args.epochs + 1)
    if not args.no_tqdm:
        epoch_iter = tqdm(epoch_iter, desc="Training", ncols=120)

    for epoch in epoch_iter:
        shock_active = bool(
            args.shock_enabled
            and shock_start_epoch is not None
            and shock_end_epoch is not None
            and shock_start_epoch <= epoch <= shock_end_epoch
        )
        epoch_batch_size = args.shock_batch_size if shock_active else base_batch_size
        train_loader = make_train_loader(epoch_batch_size)

        current_lr = args.lr * lr_scale_for_epoch(epoch=epoch, warmup_epochs=args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        grad_norms: list[float] = []
        q_grad_norms: list[float] = []
        k_grad_norms: list[float] = []

        for tokens, targets in train_loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(tokens, return_attn=False)
            loss = loss_fn(logits, targets)
            loss.backward()

            grad_norm, q_grad_norm, k_grad_norm = attention_gradient_norms(model)
            grad_norms.append(grad_norm)
            q_grad_norms.append(q_grad_norm)
            k_grad_norms.append(k_grad_norm)

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            batch_count = targets.shape[0]
            total_count += batch_count
            total_loss += loss.item() * batch_count
            total_correct += int((logits.argmax(dim=-1) == targets).sum().item())

        train_loss = total_loss / total_count if total_count else 0.0
        train_acc = total_correct / total_count if total_count else 0.0

        test_loss, test_acc, test_conf = evaluate(model, test_loader, loss_fn, device=device)
        svd_stats = model_svd_stats(model)
        attn_stats = attention_probe_metrics(model, probe_batch, device=device)
        fourier_stats = embedding_fourier_metrics(
            model,
            num_symbol_tokens=task_data.metadata.get("prime", task_data.vocab_size),
            low_freq_k=args.fourier_low_k,
        )

        grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else 0.0
        grad_norm_max = float(np.max(grad_norms)) if grad_norms else 0.0
        q_grad_norm_mean = float(np.mean(q_grad_norms)) if q_grad_norms else 0.0
        k_grad_norm_mean = float(np.mean(k_grad_norms)) if k_grad_norms else 0.0
        epoch_grad_norm_history.append(grad_norm_mean)

        grad_eps = args.cv_eps
        if len(grad_norms) > 1:
            grad_arr = np.array(grad_norms, dtype=np.float64)
            c_v_raw = float(np.var(grad_arr))
            c_v = float(c_v_raw / (float(np.mean(grad_arr) ** 2) + grad_eps))
        else:
            window = np.array(
                epoch_grad_norm_history[-args.cv_window_epochs :],
                dtype=np.float64,
            )
            c_v_raw = float(np.var(window)) if len(window) > 1 else 0.0
            c_v = float(c_v_raw / (float(np.mean(window) ** 2) + grad_eps)) if len(window) else 0.0

        if epoch == 1 or epoch % args.hessian_every == 0:
            h_tokens = hessian_tokens.to(device)
            h_targets = hessian_targets.to(device)
            last_hessian_lambda = estimate_hessian_top_eigenvalue(
                model,
                loss_fn,
                probe_tokens=h_tokens,
                probe_targets=h_targets,
                iters=args.hessian_iters,
                eps=args.hessian_eps,
            )

        cv_baseline = (
            float(np.median([m.c_v for m in history])) if history else max(c_v, 1e-9)
        )
        stage = infer_stage(train_acc=train_acc, test_acc=test_acc, c_v=c_v, cv_baseline=cv_baseline)
        inverse_temperature_beta = float(epoch_batch_size / (current_lr + 1e-12))

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            test_confidence=test_conf,
            generalization_gap=train_acc - test_acc,
            c_v=c_v,
            c_v_raw=c_v_raw,
            grad_norm_mean=grad_norm_mean,
            q_grad_norm_mean=q_grad_norm_mean,
            k_grad_norm_mean=k_grad_norm_mean,
            grad_norm_max=grad_norm_max,
            s_svd=svd_stats["s_svd"],
            top3_mass=svd_stats["top3_mass"],
            effective_rank=svd_stats["effective_rank"],
            order_parameter=svd_stats["order_parameter"],
            attention_entropy=attn_stats["attention_entropy"],
            attention_distance=attn_stats["attention_distance"],
            operand_focus=attn_stats["operand_focus"],
            hessian_lambda_max=last_hessian_lambda,
            fourier_amp1=fourier_stats["amp1"],
            fourier_ratio1=fourier_stats["ratio1"],
            fourier_lowfreq_ratio=fourier_stats["lowfreq_ratio"],
            inverse_temperature_beta=inverse_temperature_beta,
            shock_active=shock_active,
            shock_batch_size=epoch_batch_size,
            lr=current_lr,
            stage=stage,
        )
        history.append(metrics)
        if landscape_enabled:
            if not landscape_basis_aligned:
                aligned_dir1, aligned_dir2 = align_landscape_basis_to_progress(model, landscape_theta0)
                if aligned_dir1 is not None and aligned_dir2 is not None:
                    landscape_dir1 = aligned_dir1
                    landscape_dir2 = aligned_dir2
                    landscape_basis_aligned = True
            alpha_t, beta_t = project_to_landscape_coordinates(model, landscape_theta0, landscape_dir1, landscape_dir2)
            landscape_trajectory.append((epoch, alpha_t, beta_t))

        if "before" not in spectrum_snapshots and epoch >= spectrum_before_target:
            spectrum_snapshots["before"] = (epoch, get_layer0_q_spectrum(model))
        if "during" not in spectrum_snapshots:
            if shock_start_epoch is not None and epoch >= shock_start_epoch:
                spectrum_snapshots["during"] = (epoch, get_layer0_q_spectrum(model))
            elif epoch >= spectrum_during_target:
                spectrum_snapshots["during"] = (epoch, get_layer0_q_spectrum(model))
        if epoch == args.epochs:
            spectrum_snapshots["after"] = (epoch, get_layer0_q_spectrum(model))

        max_rank_seen = max(max_rank_seen, metrics.effective_rank)

        if (
            args.shock_enabled
            and shock_start_epoch is None
            and epoch >= args.shock_min_epoch
            and (max_rank_seen - metrics.effective_rank) >= args.shock_rank_drop_threshold
        ):
            shock_start_epoch = epoch + 1
            shock_end_epoch = shock_start_epoch + args.shock_duration_epochs - 1
            shock_trigger_reason = "rank_drop_trigger"

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                "epoch=%d train_acc=%.4f test_acc=%.4f c_v=%.6f lambda_max=%.4f batch=%d stage=%s"
                % (
                    metrics.epoch,
                    metrics.train_acc,
                    metrics.test_acc,
                    metrics.c_v,
                    metrics.hessian_lambda_max,
                    metrics.shock_batch_size,
                    metrics.stage,
                )
            )

        if wandb_run is not None and wandb is not None and epoch % args.wandb_log_every == 0:
            log_data = {
                "epoch": metrics.epoch,
                "metrics/train_loss": metrics.train_loss,
                "metrics/train_accuracy": metrics.train_acc,
                "metrics/test_loss": metrics.test_loss,
                "metrics/test_accuracy": metrics.test_acc,
                "metrics/generalization_gap": metrics.generalization_gap,
                "physics/heat_capacity_Cv": metrics.c_v,
                "physics/hessian_lambda_max": metrics.hessian_lambda_max,
                "physics/inverse_temperature": metrics.inverse_temperature_beta,
                "compression/svd_entropy": metrics.s_svd,
                "compression/effective_rank": metrics.effective_rank,
                "compression/top3_sv_mass": metrics.top3_mass,
                "compression/order_parameter": metrics.order_parameter,
                "mechanistic/fourier_ratio_k1": metrics.fourier_ratio1,
                "mechanistic/attention_entropy": metrics.attention_entropy,
                "mechanistic/operand_focus": metrics.operand_focus,
                "grads/norm_mean": metrics.grad_norm_mean,
                "optim/lr": metrics.lr,
                "optim/stage": metrics.stage,
            }
            wandb_run.log(log_data, step=metrics.epoch)

            if epoch == 1 or epoch % args.plot_every == 0 or epoch == args.epochs:
                current_prediction = predict_grokking_epoch(history, acc_threshold=args.acc_threshold)
                dashboard_fig = make_dashboard_figure(history, prediction=current_prediction)
                log_figure_to_wandb(
                    wandb_run=wandb_run,
                    wandb=wandb,
                    key="plots/phase_dashboard",
                    fig=dashboard_fig,
                    step=metrics.epoch,
                    media_dir=wandb_media_dir,
                    stem="phase_dashboard",
                )
                plt.close(dashboard_fig)

                extra_now = (
                    args.extra_media == "always"
                    or (args.extra_media == "final" and epoch == args.epochs)
                )
                if extra_now:
                    spec_fig = make_spectrum_figure(spectrum_snapshots)
                    log_figure_to_wandb(
                        wandb_run=wandb_run,
                        wandb=wandb,
                        key="plots/singular_spectrum",
                        fig=spec_fig,
                        step=metrics.epoch,
                        media_dir=wandb_media_dir,
                        stem="singular_spectrum",
                    )
                    plt.close(spec_fig)

                    attn_fig = make_attention_snapshot_figure(
                        attn_stats["snapshot"],
                        token_labels=probe_labels,
                        title_context=probe_context,
                    )
                    if attn_fig is not None:
                        log_figure_to_wandb(
                            wandb_run=wandb_run,
                            wandb=wandb,
                            key="plots/attention_matrix_annotated",
                            fig=attn_fig,
                            step=metrics.epoch,
                            media_dir=wandb_media_dir,
                            stem="attention_matrix_annotated",
                        )
                        plt.close(attn_fig)

    prediction = predict_grokking_epoch(history, acc_threshold=args.acc_threshold)
    if args.shock_enabled and shock_start_epoch is None and shock_trigger_reason == "armed_auto":
        shock_trigger_reason = "armed_auto_not_triggered"

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(history, metrics_path)

    dashboard_path = output_dir / "dashboard.png"
    dashboard_fig = make_dashboard_figure(history, prediction=prediction)
    dashboard_fig.savefig(dashboard_path, dpi=180, bbox_inches="tight")
    plt.close(dashboard_fig)

    spectrum_path: str | None = None
    attention_path: str | None = None
    landscape_path: str | None = None
    if "after" not in spectrum_snapshots:
        spectrum_snapshots["after"] = (args.epochs, get_layer0_q_spectrum(model))
    if "before" not in spectrum_snapshots and "after" in spectrum_snapshots:
        spectrum_snapshots["before"] = spectrum_snapshots["after"]
    if "during" not in spectrum_snapshots and "after" in spectrum_snapshots:
        spectrum_snapshots["during"] = spectrum_snapshots["after"]

    if args.extra_media != "none":
        spec_file = output_dir / "singular_spectrum.png"
        spec_fig = make_spectrum_figure(spectrum_snapshots)
        spec_fig.savefig(spec_file, dpi=180, bbox_inches="tight")
        plt.close(spec_fig)
        spectrum_path = str(spec_file)

        landscape_file = output_dir / "loss_landscape.png"
        landscape_fig = make_loss_landscape_figure(
            model,
            loss_fn,
            hessian_tokens,
            hessian_targets,
            device,
            theta_0=landscape_theta0,
            dir1=landscape_dir1,
            dir2=landscape_dir2,
            trajectory=landscape_trajectory,
            shock_epoch=shock_start_epoch,
        )
        landscape_fig.savefig(landscape_file, dpi=180, bbox_inches="tight")
        plt.close(landscape_fig)
        landscape_path = str(landscape_file)

        final_snapshot = attention_probe_metrics(model, probe_batch, device=device)["snapshot"]
        attention_file = output_dir / "attention_snapshot.png"
        attn_fig = make_attention_snapshot_figure(
            final_snapshot,
            token_labels=probe_labels,
            title_context=probe_context,
        )
        if attn_fig is not None:
            attn_fig.savefig(attention_file, dpi=180, bbox_inches="tight")
            plt.close(attn_fig)
            attention_path = str(attention_file)

    summary = {
        "config": config,
        "task_metadata": task_data.metadata,
        "shock": {
            "enabled": args.shock_enabled,
            "trigger_reason": shock_trigger_reason,
            "trigger_start_epoch": shock_start_epoch,
            "trigger_end_epoch": shock_end_epoch,
        },
        "prediction": asdict(prediction),
        "final_metrics": asdict(history[-1]) if history else {},
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "dashboard_png": str(dashboard_path),
            "spectrum_png": spectrum_path,
            "attention_png": attention_path,
            "loss_landscape_png": landscape_path,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if wandb_run is not None:
        if wandb is not None:
            wandb_run.summary["prediction/predicted_epoch"] = prediction.predicted_epoch
            wandb_run.summary["prediction/observed_epoch"] = prediction.observed_epoch
            wandb_run.summary["prediction/cv_spike_value"] = prediction.cv_spike_value
            wandb_run.summary["artifacts/metrics_csv"] = str(metrics_path)
            wandb_run.summary["artifacts/dashboard_png"] = str(dashboard_path)
            if spectrum_path is not None:
                wandb_run.summary["artifacts/spectrum_png"] = spectrum_path
            if attention_path is not None:
                wandb_run.summary["artifacts/attention_png"] = attention_path
            if landscape_path is not None:
                wandb_run.summary["artifacts/loss_landscape_png"] = landscape_path

            if landscape_path is not None:
                wandb_run.log({"plots/final_loss_landscape_contour": wandb.Image(landscape_path)})

        wandb_run.finish()

    print(f"Run complete. Results written to: {output_dir.resolve()}")
    print(f"Predicted grokking epoch: {prediction.predicted_epoch}")
    print(f"Observed grokking epoch:  {prediction.observed_epoch}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args with optional flat-YAML defaults from --config."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Thermodynamics-of-grokking experiment with W&B logging.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to flat YAML config file.")
    parser.add_argument(
        "--task",
        type=str,
        choices=("modular_division", "sparse_parity", "boolean_logic"),
        default="modular_division",
    )
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--cv-window-epochs", type=int, default=32)
    parser.add_argument("--cv-eps", type=float, default=1e-12)
    parser.add_argument("--hessian-every", type=int, default=50)
    parser.add_argument("--hessian-iters", type=int, default=10)
    parser.add_argument("--hessian-eps", type=float, default=1e-12)
    parser.add_argument("--hessian-probe-size", type=int, default=512)
    parser.add_argument("--fourier-low-k", type=int, default=5)
    parser.add_argument("--spectrum-before-epoch", type=int, default=100)
    parser.add_argument("--spectrum-during-epoch", type=int, default=801)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--shock-enabled", action="store_true")
    parser.add_argument("--shock-start-epoch", type=int, default=-1)
    parser.add_argument("--shock-min-epoch", type=int, default=800)
    parser.add_argument("--shock-duration-epochs", type=int, default=50)
    parser.add_argument("--shock-batch-size", type=int, default=16)
    parser.add_argument("--shock-rank-drop-threshold", type=float, default=0.2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--acc-threshold", type=float, default=0.99)
    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--parity-bits", type=int, default=16)
    parser.add_argument("--parity-k", type=int, default=5)
    parser.add_argument("--parity-dataset-size", type=int, default=16384)
    parser.add_argument("--logic-input-bits", type=int, default=8)
    parser.add_argument("--logic-num-gates", type=int, default=5)
    parser.add_argument("--logic-dataset-size", type=int, default=0)

    parser.add_argument("--wandb-project", type=str, default="thermodynamics-of-grokking")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=".wandb")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="offline",
        help="offline logs locally and can be synced later.",
    )
    parser.add_argument("--wandb-log-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=200)
    parser.add_argument(
        "--extra-media",
        type=str,
        choices=("none", "final", "always"),
        default="final",
        help="Controls logging/saving of auxiliary images (spectrum, landscape, attention).",
    )

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-tqdm", action="store_true")
    if pre_args.config:
        loaded = parse_simple_yaml_config(pre_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown = sorted(k for k in loaded.keys() if k not in valid_keys)
        if unknown:
            raise ValueError(
                "Unknown keys in config: "
                + ", ".join(unknown)
            )
        parser.set_defaults(**loaded)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    train(parse_args())
