"""Metric probes and phase-transition helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import TinyDecoderTransformer


@dataclass
class EpochMetrics:
    """All scalar metrics recorded for one training epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    test_confidence: float
    generalization_gap: float
    c_v: float
    c_v_raw: float
    grad_norm_mean: float
    q_grad_norm_mean: float
    k_grad_norm_mean: float
    grad_norm_max: float
    s_svd: float
    top3_mass: float
    effective_rank: float
    order_parameter: float
    attention_entropy: float
    attention_distance: float
    operand_focus: float
    hessian_lambda_max: float
    fourier_amp1: float
    fourier_ratio1: float
    fourier_lowfreq_ratio: float
    inverse_temperature_beta: float
    shock_active: bool
    shock_batch_size: int
    lr: float
    stage: str


@dataclass
class GrokkingPrediction:
    """Predicted and observed grokking transition metadata."""

    predicted_epoch: int | None
    observed_epoch: int | None
    cv_baseline: float | None
    cv_spike_value: float | None
    score_at_prediction: float | None


def attention_gradient_norms(model: TinyDecoderTransformer) -> tuple[float, float, float]:
    """Return total, Q-only, and K-only gradient norms across blocks."""
    total_sq = 0.0
    q_sq = 0.0
    k_sq = 0.0
    found = False

    for block in model.blocks:
        q_grad = block.attn.q_proj.weight.grad
        k_grad = block.attn.k_proj.weight.grad

        if q_grad is not None:
            val = float(torch.sum(q_grad.detach().float() ** 2).item())
            q_sq += val
            total_sq += val
            found = True
        if k_grad is not None:
            val = float(torch.sum(k_grad.detach().float() ** 2).item())
            k_sq += val
            total_sq += val
            found = True

    if not found:
        return 0.0, 0.0, 0.0
    return math.sqrt(total_sq), math.sqrt(q_sq), math.sqrt(k_sq)


@torch.no_grad()
def embedding_fourier_metrics(
    model: TinyDecoderTransformer,
    num_symbol_tokens: int,
    low_freq_k: int = 5,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute low-frequency Fourier statistics for token embeddings."""
    emb = model.token_emb.weight[:num_symbol_tokens].detach().float()
    if emb.shape[0] <= 1:
        return {"amp1": 0.0, "ratio1": 0.0, "lowfreq_ratio": 0.0}

    spectrum = torch.fft.rfft(emb, dim=0)
    amp = spectrum.abs().mean(dim=1)
    if amp.shape[0] <= 1:
        return {"amp1": 0.0, "ratio1": 0.0, "lowfreq_ratio": 0.0}

    non_dc = amp[1:]
    non_dc_total = float(non_dc.sum().item())
    amp1 = float(amp[1].item())
    ratio1 = amp1 / (non_dc_total + eps)
    k = min(low_freq_k, non_dc.shape[0])
    lowfreq_ratio = float(non_dc[:k].sum().item()) / (non_dc_total + eps)
    return {"amp1": amp1, "ratio1": ratio1, "lowfreq_ratio": lowfreq_ratio}


def _normalize_tensors(vecs: list[torch.Tensor], eps: float = 1e-12) -> list[torch.Tensor]:
    """Normalize a list of tensors as one concatenated vector."""
    norm_sq = torch.zeros((), device=vecs[0].device, dtype=vecs[0].dtype)
    for v in vecs:
        norm_sq = norm_sq + torch.sum(v * v)
    norm = torch.sqrt(norm_sq)
    return [v / (norm + eps) for v in vecs]


def estimate_hessian_top_eigenvalue(
    model: TinyDecoderTransformer,
    loss_fn: nn.Module,
    probe_tokens: torch.Tensor,
    probe_targets: torch.Tensor,
    iters: int = 10,
    eps: float = 1e-12,
) -> float:
    """Estimate top Hessian eigenvalue with power iteration + Hessian-vector products."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or iters <= 0:
        return float("nan")

    was_training = model.training
    model.eval()
    v = [torch.randn_like(p) for p in params]
    v = _normalize_tensors(v, eps=eps)

    lambda_est = 0.0
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        logits, _ = model(probe_tokens, return_attn=False)
        loss = loss_fn(logits, probe_targets)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        hv_seed = sum((g * vi).sum() for g, vi in zip(grads, v))
        hv = torch.autograd.grad(hv_seed, params, retain_graph=False)

        lambda_est = float(
            sum((vi.detach() * hi.detach()).sum().item() for vi, hi in zip(v, hv))
        )
        hv_detached = [h.detach() for h in hv]
        hv_norm_sq = sum(torch.sum(h * h) for h in hv_detached)
        hv_norm = float(torch.sqrt(hv_norm_sq).item())
        if hv_norm < eps:
            lambda_est = 0.0
            break
        v = [h / (hv_norm + eps) for h in hv_detached]

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return lambda_est


def matrix_svd_stats(weight: torch.Tensor, eps: float = 1e-12) -> tuple[float, float, float, float]:
    """Return entropy, top-3 mass, effective rank, and order parameter for singular values."""
    s = torch.linalg.svdvals(weight.detach().float().cpu())
    total = float(s.sum().item())
    if total <= eps:
        return 0.0, 0.0, 0.0, 0.0
    p = s / (total + eps)
    entropy = float((-(p * torch.log(p + eps))).sum().item())
    top3_mass = float(p[:3].sum().item())
    effective_rank = float(math.exp(entropy))
    order_parameter = float((p[0] - p[1:].mean()).item() if len(p) > 1 else p[0].item())
    return entropy, top3_mass, effective_rank, order_parameter


@torch.no_grad()
def model_svd_stats(model: TinyDecoderTransformer) -> dict[str, float]:
    """Aggregate spectrum statistics over all Q/K projection matrices."""
    entropies: list[float] = []
    top3_masses: list[float] = []
    effective_ranks: list[float] = []
    order_params: list[float] = []

    for block in model.blocks:
        for weight in (block.attn.q_proj.weight, block.attn.k_proj.weight):
            entropy, top3_mass, eff_rank, order_param = matrix_svd_stats(weight)
            entropies.append(entropy)
            top3_masses.append(top3_mass)
            effective_ranks.append(eff_rank)
            order_params.append(order_param)

    return {
        "s_svd": float(np.mean(entropies)) if entropies else 0.0,
        "top3_mass": float(np.mean(top3_masses)) if top3_masses else 0.0,
        "effective_rank": float(np.mean(effective_ranks)) if effective_ranks else 0.0,
        "order_parameter": float(np.mean(order_params)) if order_params else 0.0,
    }


@torch.no_grad()
def get_layer0_q_spectrum(model: TinyDecoderTransformer) -> np.ndarray:
    """Extract layer-0 Q-projection singular values."""
    if len(model.blocks) == 0:
        return np.array([], dtype=np.float64)
    return torch.linalg.svdvals(model.blocks[0].attn.q_proj.weight.detach().float().cpu()).numpy()


@torch.no_grad()
def evaluate(
    model: TinyDecoderTransformer,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate average loss, accuracy, and confidence on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_confidence = 0.0

    for tokens, targets in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        logits, _ = model(tokens, return_attn=False)
        loss = loss_fn(logits, targets)

        probs = torch.softmax(logits, dim=-1)
        confidence, preds = probs.max(dim=-1)

        batch_size = targets.shape[0]
        total_count += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((preds == targets).sum().item())
        total_confidence += float(confidence.sum().item())

    if total_count == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / total_count,
        total_correct / total_count,
        total_confidence / total_count,
    )


@torch.no_grad()
def attention_probe_metrics(
    model: TinyDecoderTransformer,
    probe_tokens: torch.Tensor,
    device: torch.device,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute attention entropy/distance/focus and one snapshot heatmap."""
    model.eval()
    probe_tokens = probe_tokens.to(device)
    _, attn_maps = model(probe_tokens, return_attn=True)
    if not attn_maps:
        return {
            "attention_entropy": 0.0,
            "attention_distance": 0.0,
            "operand_focus": 0.0,
            "snapshot": None,
        }

    all_attn = torch.stack(attn_maps, dim=0)  # [L, B, H, T, T]
    attention_entropy = float((-(all_attn * torch.log(all_attn + eps))).sum(dim=-1).mean().item())

    seq_len = all_attn.shape[-1]
    idx = torch.arange(seq_len, device=all_attn.device, dtype=all_attn.dtype)
    distance = torch.abs(idx[:, None] - idx[None, :])  # [T, T]
    attention_distance = float((all_attn * distance).sum(dim=-1).mean().item())

    last_query = all_attn[..., -1, :]  # [L, B, H, T]
    left_idx = 0
    right_idx = max(seq_len - 2, 0)
    operand_focus = float((last_query[..., left_idx] + last_query[..., right_idx]).mean().item())

    snapshot = all_attn[0, 0, 0].detach().float().cpu().numpy()
    return {
        "attention_entropy": attention_entropy,
        "attention_distance": attention_distance,
        "operand_focus": operand_focus,
        "snapshot": snapshot,
    }


def infer_stage(train_acc: float, test_acc: float, c_v: float, cv_baseline: float) -> str:
    """Assign coarse training phase label from accuracy and heat-capacity dynamics."""
    spike_threshold = max(cv_baseline * 3.0, 1e-9)
    if test_acc >= 0.99:
        return "stage_3_aha"
    if train_acc >= 0.95 and test_acc <= 0.2 and c_v <= spike_threshold:
        return "stage_1_memorization"
    if c_v > spike_threshold and test_acc < 0.8:
        return "stage_2_critical"
    return "transition"


def lr_scale_for_epoch(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup schedule multiplier."""
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= warmup_epochs:
        return max(epoch / warmup_epochs, 1e-8)
    return 1.0


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD."""
    if values.size == 0:
        return values
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def predict_grokking_epoch(
    history: list[EpochMetrics],
    acc_threshold: float = 0.99,
) -> GrokkingPrediction:
    """Predict grokking by combining C_v spikes with entropy-collapse slope."""
    if not history:
        return GrokkingPrediction(None, None, None, None, None)

    epochs = np.array([m.epoch for m in history], dtype=np.int64)
    test_acc = np.array([m.test_acc for m in history], dtype=np.float64)
    c_v = np.array([m.c_v for m in history], dtype=np.float64)
    s_svd = np.array([m.s_svd for m in history], dtype=np.float64)

    hits = np.where(test_acc >= acc_threshold)[0]
    observed_idx = int(hits[0]) if hits.size else None
    cutoff_idx = observed_idx if observed_idx is not None else len(history) - 1

    cv_window = c_v[: cutoff_idx + 1]
    entropy_window = s_svd[: cutoff_idx + 1]
    if cv_window.size == 0:
        return GrokkingPrediction(
            None,
            int(epochs[observed_idx]) if observed_idx is not None else None,
            None,
            None,
            None,
        )

    cv_score = robust_zscore(cv_window)
    entropy_drop = -np.diff(entropy_window, prepend=entropy_window[0])
    entropy_score = robust_zscore(entropy_drop)
    score = cv_score + 0.5 * entropy_score

    pred_local_idx = int(np.argmax(score))
    pred_epoch = int(epochs[pred_local_idx])
    cv_baseline = float(np.median(cv_window))
    cv_spike = float(cv_window[pred_local_idx])
    observed_epoch = int(epochs[observed_idx]) if observed_idx is not None else None

    return GrokkingPrediction(
        predicted_epoch=pred_epoch,
        observed_epoch=observed_epoch,
        cv_baseline=cv_baseline,
        cv_spike_value=cv_spike,
        score_at_prediction=float(score[pred_local_idx]),
    )


def ema(values: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Simple exponential moving average for smoother plots."""
    if values.size == 0:
        return values
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def shock_windows(epochs: np.ndarray, shock_flag: np.ndarray) -> list[tuple[int, int]]:
    """Convert a per-epoch shock flag series into contiguous epoch windows."""
    windows: list[tuple[int, int]] = []
    in_window = False
    start = 0
    for idx, active in enumerate(shock_flag > 0.5):
        if active and not in_window:
            in_window = True
            start = int(epochs[idx])
        if in_window and (not active):
            windows.append((start, int(epochs[idx - 1])))
            in_window = False
    if in_window and epochs.size:
        windows.append((start, int(epochs[-1])))
    return windows

