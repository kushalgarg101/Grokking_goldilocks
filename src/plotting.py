"""Plotting utilities for dashboards and interpretability artifacts."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from .model import TinyDecoderTransformer
from .probes import EpochMetrics, GrokkingPrediction, ema, shock_windows


def make_dashboard_figure(
    history: list[EpochMetrics],
    prediction: GrokkingPrediction | None = None,
) -> plt.Figure:
    """ dashboard."""
    epochs = np.array([m.epoch for m in history], dtype=np.int64)
    train_acc = np.array([m.train_acc for m in history], dtype=np.float64)
    test_acc = np.array([m.test_acc for m in history], dtype=np.float64)
    train_loss = np.array([m.train_loss for m in history], dtype=np.float64)
    test_loss = np.array([m.test_loss for m in history], dtype=np.float64)
    c_v = np.array([m.c_v for m in history], dtype=np.float64)
    hessian = np.array([m.hessian_lambda_max for m in history], dtype=np.float64)
    s_svd = np.array([m.s_svd for m in history], dtype=np.float64)
    top3_mass = np.array([m.top3_mass for m in history], dtype=np.float64)
    eff_rank = np.array([m.effective_rank for m in history], dtype=np.float64)
    attn_entropy = np.array([m.attention_entropy for m in history], dtype=np.float64)
    operand_focus = np.array([m.operand_focus for m in history], dtype=np.float64)
    fourier_ratio1 = np.array([m.fourier_ratio1 for m in history], dtype=np.float64)
    grad_norm = np.array([m.grad_norm_mean for m in history], dtype=np.float64)
    shock_flag = np.array([1.0 if m.shock_active else 0.0 for m in history], dtype=np.float64)

    _alpha = 0.12
    test_acc_s = ema(test_acc, alpha=_alpha)
    c_v_s = ema(c_v, alpha=_alpha)
    hessian_s = ema(np.nan_to_num(hessian, nan=0.0), alpha=_alpha)
    eff_rank_s = ema(eff_rank, alpha=_alpha)
    s_svd_s = ema(s_svd, alpha=_alpha)
    fourier_s = ema(fourier_ratio1, alpha=_alpha)
    grad_s = ema(grad_norm, alpha=_alpha)
    shock_regions = shock_windows(epochs, shock_flag)

    BG = "#0d1117"
    GRID_C = "#21262d"
    TEXT_C = "#c9d1d9"
    ACCENT1 = "#58a6ff"
    ACCENT2 = "#3fb950"
    ACCENT3 = "#f85149"
    ACCENT4 = "#bc8cff"
    ACCENT5 = "#f0883e"
    ACCENT6 = "#79c0ff"
    ACCENT7 = "#d2a8ff"

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID_C,
            "axes.labelcolor": TEXT_C,
            "text.color": TEXT_C,
            "xtick.color": TEXT_C,
            "ytick.color": TEXT_C,
            "legend.facecolor": "#161b22",
            "legend.edgecolor": GRID_C,
            "legend.labelcolor": TEXT_C,
            "grid.color": GRID_C,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def _add_pred_lines(ax_: plt.Axes) -> None:
        if prediction is None:
            return
        if prediction.predicted_epoch is not None:
            ax_.axvline(
                prediction.predicted_epoch,
                ls="--",
                color=ACCENT3,
                alpha=0.5,
                label="Predicted Grok" if ax_ is axes[0, 0] else "",
            )
        if (
            prediction.observed_epoch is not None
            and prediction.observed_epoch != prediction.predicted_epoch
        ):
            ax_.axvline(
                prediction.observed_epoch,
                ls="--",
                color=ACCENT2,
                alpha=0.5,
                label="Observed Grok" if ax_ is axes[0, 0] else "",
            )

    l_kwargs = dict(fontsize=7, framealpha=0.3, loc="upper right", borderaxespad=1.0)

    ax = axes[0, 0]
    ax.plot(epochs, train_acc, color=ACCENT1, alpha=0.30, lw=1.0, label="Train Accuracy")
    ax.plot(epochs, test_acc_s, color=ACCENT2, lw=2.2, label="Test Accuracy (smoothed)")
    ax.fill_between(epochs, 0, test_acc_s, color=ACCENT2, alpha=0.06)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(epochs, train_loss, color=ACCENT1, alpha=0.25, ls="--", lw=1.0, label="Train Loss")
    ax2.plot(epochs, test_loss, color=ACCENT5, alpha=0.45, ls="--", lw=1.0, label="Test Loss")
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=9, color=ACCENT5)
    ax2.tick_params(axis="y", colors=ACCENT5)
    _add_pred_lines(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, **l_kwargs)
    ax.set_title("Learning Progress", fontsize=11, fontweight="bold", pad=8)

    ax = axes[0, 1]
    ax.plot(epochs, c_v_s, color=ACCENT3, lw=2.2, label="Heat Capacity Cᵥ (smoothed)")
    ax.fill_between(epochs, 0, c_v_s, color=ACCENT3, alpha=0.06)
    ax.set_ylabel("Normalized Heat Capacity  Cᵥ", fontsize=9, color=ACCENT3)
    ax.tick_params(axis="y", colors=ACCENT3)
    ax.grid(alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(epochs, hessian_s, color=ACCENT4, lw=2.0, label="Hessian λ_max (smoothed)")
    ax2.set_ylabel("Top Hessian Eigenvalue  λ_max", fontsize=9, color=ACCENT4)
    ax2.tick_params(axis="y", colors=ACCENT4)
    _add_pred_lines(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, **l_kwargs)
    ax.set_title("Phase-Transition Signals", fontsize=11, fontweight="bold", pad=8)

    ax = axes[1, 0]
    ax.plot(epochs, s_svd_s, color=ACCENT5, lw=2.2, label="SVD Entropy  S_SVD (smoothed)")
    ax.fill_between(epochs, 0, s_svd_s, color=ACCENT5, alpha=0.06)
    ax.set_ylabel("SVD Entropy  S_SVD", fontsize=9, color=ACCENT5)
    ax.tick_params(axis="y", colors=ACCENT5)
    ax.grid(alpha=0.15)
    ax_r = ax.twinx()
    ax_r.plot(epochs, eff_rank_s, color=ACCENT6, lw=2.0, label="Effective Rank (smoothed)")
    ax_r.plot(epochs, top3_mass, color=ACCENT7, lw=1.6, alpha=0.8, label="Top-3 SV Mass")
    ax_r.set_ylabel("Rank / Top-3 Mass", fontsize=9, color=ACCENT6)
    ax_r.tick_params(axis="y", colors=ACCENT6)
    _add_pred_lines(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, **l_kwargs)
    ax.set_title("Representation Compression (RG View)", fontsize=11, fontweight="bold", pad=8)

    ax = axes[1, 1]
    ax.plot(epochs, fourier_s, color=ACCENT2, lw=2.0, label="Fourier k=1 Ratio (smoothed)")
    ax.plot(epochs, operand_focus, color="#d2a8ff", lw=1.6, alpha=0.85, label="Operand Focus")
    ax.set_ylabel("Fourier / Focus", fontsize=9)
    ax.grid(alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(epochs, grad_s, color=ACCENT3, alpha=0.7, lw=1.8, label="Gradient Norm (smoothed)")
    ax2.set_ylabel("Gradient ‖∇θ‖", fontsize=9, color=ACCENT3)
    ax2.tick_params(axis="y", colors=ACCENT3)
    for start, end in shock_regions:
        ax.axvspan(
            start,
            end,
            color="#ffd166",
            alpha=0.12,
            label="Shock Window" if start == shock_regions[0][0] else "",
        )
    _add_pred_lines(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, **l_kwargs)
    ax.set_title("Mechanistic & Intervention Signals", fontsize=11, fontweight="bold", pad=8)

    for row in axes:
        for a in row:
            a.set_xlabel("Epoch", fontsize=9)

    fig.suptitle(
        "Thermodynamics of Grokking — Training Dashboard",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def make_attention_snapshot_figure(
    snapshot: np.ndarray | None,
    token_labels: list[str] | None = None,
    title_context: str | None = None,
) -> plt.Figure | None:
    """Render an annotated 2D attention heatmap for one sample."""
    if snapshot is None:
        return None
    seq_len = snapshot.shape[0]
    labels = token_labels if token_labels is not None and len(token_labels) == seq_len else [str(i) for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(snapshot, cmap="inferno", aspect="equal", interpolation="nearest", vmin=0.0, vmax=1.0)
    title = "Annotated Attention Matrix (Last Query-Token Routing)"
    if title_context:
        title = f"{title}\n{title_context}"
    ax.set_title(title, fontsize=11, fontweight="bold", color="#c9d1d9", pad=10)
    ax.set_xlabel("Key Token", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Query Token", fontsize=10, color="#c9d1d9")
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(labels, fontsize=9, color="#c9d1d9")
    ax.set_yticklabels(labels, fontsize=9, color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9", labelsize=9)

    thresh = float(snapshot.max() + snapshot.min()) / 2.0
    for i in range(seq_len):
        for j in range(seq_len):
            val = float(snapshot[i, j])
            txt_color = "#111111" if val > thresh else "#e5e7eb"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#c9d1d9")
    cb.set_label("Attention Weight", color="#c9d1d9", fontsize=9)
    fig.tight_layout()
    return fig


def make_spectrum_figure(
    spectrum_snapshots: dict[str, tuple[int, np.ndarray]],
) -> plt.Figure:
    """Plot before/during/after singular spectra on a shared log-scale axis."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    style_map = {
        "before": {"color": "#58a6ff", "ls": "-", "lw": 2.0, "label": "Before (Memorization)"},
        "during": {"color": "#f0883e", "ls": ":", "lw": 2.2, "label": "During Shock/Transition"},
        "after": {"color": "#3fb950", "ls": "-", "lw": 2.0, "label": "After (Late Phase)"},
    }

    plotted_any = False
    for key in ("before", "during", "after"):
        if key not in spectrum_snapshots:
            continue
        epoch_id, sv = spectrum_snapshots[key]
        style = style_map[key]
        ax.plot(
            sv,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=f"{style['label']} · epoch {epoch_id}",
        )
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No spectrum snapshots captured",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#c9d1d9",
        )

    ax.set_yscale("log")
    ax.set_title("Layer-0 Q Singular Spectrum Across Time", fontsize=11, fontweight="bold", color="#c9d1d9", pad=8)
    ax.set_xlabel("Singular Value Index", fontsize=9, color="#c9d1d9")
    ax.set_ylabel("Magnitude (log scale)", fontsize=9, color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    ax.grid(alpha=0.12, color="#21262d")
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", loc="upper right", framealpha=0.4)
    fig.tight_layout()
    return fig


def init_landscape_basis(
    model: TinyDecoderTransformer,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Initialize random filter-normalized 2D basis around current parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    theta_0 = [p.detach().clone() for p in params]

    def _rand_direction() -> list[torch.Tensor]:
        d = [torch.randn_like(p) for p in theta_0]
        for di, pi in zip(d, theta_0):
            if di.dim() >= 2:
                for idx in range(di.shape[0]):
                    di[idx].mul_(pi[idx].norm() / (di[idx].norm() + 1e-10))
            else:
                di.mul_(pi.norm() / (di.norm() + 1e-10))
        return d

    dir1 = _rand_direction()
    dir2 = _rand_direction()
    dot = sum((d1 * d2).sum() for d1, d2 in zip(dir1, dir2))
    norm1_sq = sum((d1 * d1).sum() for d1 in dir1)
    for d1, d2 in zip(dir1, dir2):
        d2.sub_(d1 * (dot / (norm1_sq + 1e-10)))
    return theta_0, dir1, dir2


@torch.no_grad()
def align_landscape_basis_to_progress(
    model: TinyDecoderTransformer,
    theta_0: list[torch.Tensor],
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """Align landscape basis with observed optimization direction."""
    params = [p for p in model.parameters() if p.requires_grad]
    delta = [p.detach() - t0 for p, t0 in zip(params, theta_0)]
    norm_sq = sum(torch.sum(d * d) for d in delta)
    delta_norm = float(torch.sqrt(norm_sq).item())
    if delta_norm <= eps:
        return None, None

    dir1 = [d.clone() for d in delta]
    dir2 = [torch.randn_like(d) for d in dir1]
    for d2, d1 in zip(dir2, dir1):
        if d2.dim() >= 2:
            for idx in range(d2.shape[0]):
                d2[idx].mul_(d1[idx].norm() / (d2[idx].norm() + eps))
        else:
            d2.mul_(d1.norm() / (d2.norm() + eps))

    dot = sum((d1 * d2).sum() for d1, d2 in zip(dir1, dir2))
    norm1_sq = sum((d1 * d1).sum() for d1 in dir1)
    for d1, d2 in zip(dir1, dir2):
        d2.sub_(d1 * (dot / (norm1_sq + eps)))
    norm2_sq = sum(torch.sum(d2 * d2) for d2 in dir2)
    norm2 = float(torch.sqrt(norm2_sq).item())
    if norm2 <= eps:
        return None, None
    norm1 = float(torch.sqrt(norm1_sq).item())
    scale = norm1 / (norm2 + eps)
    dir2 = [d2 * scale for d2 in dir2]
    return dir1, dir2


@torch.no_grad()
def project_to_landscape_coordinates(
    model: TinyDecoderTransformer,
    theta_0: list[torch.Tensor],
    dir1: list[torch.Tensor],
    dir2: list[torch.Tensor],
) -> tuple[float, float]:
    """Project current weights onto a fixed 2D basis centered at theta_0."""
    params = [p for p in model.parameters() if p.requires_grad]
    delta = [p.detach() - t0 for p, t0 in zip(params, theta_0)]

    a_num = sum(float((d * u).sum().item()) for d, u in zip(delta, dir1))
    a_den = sum(float((u * u).sum().item()) for u in dir1) + 1e-12
    b_num = sum(float((d * v).sum().item()) for d, v in zip(delta, dir2))
    b_den = sum(float((v * v).sum().item()) for v in dir2) + 1e-12
    return a_num / a_den, b_num / b_den


@torch.no_grad()
def make_loss_landscape_figure(
    model: TinyDecoderTransformer,
    loss_fn: nn.Module,
    data_tokens: torch.Tensor,
    data_targets: torch.Tensor,
    device: torch.device,
    theta_0: list[torch.Tensor],
    dir1: list[torch.Tensor],
    dir2: list[torch.Tensor],
    trajectory: list[tuple[int, float, float]],
    shock_epoch: int | None = None,
    grid_points: int = 60,
    radius: float = 1.0,
) -> plt.Figure:
    """Draw topographic loss contours and overlay the epoch-colored weight trajectory."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    theta_curr = [p.detach().clone() for p in params]

    center_a = 0.0
    center_b = 0.0
    radius_eff = radius
    if trajectory:
        traj_arr = np.array(trajectory, dtype=np.float64)
        a_min, a_max = float(np.min(traj_arr[:, 1])), float(np.max(traj_arr[:, 1]))
        b_min, b_max = float(np.min(traj_arr[:, 2])), float(np.max(traj_arr[:, 2]))
        center_a = 0.5 * (a_min + a_max)
        center_b = 0.5 * (b_min + b_max)
        span = max(a_max - a_min, b_max - b_min)
        radius_eff = max(radius, 0.55 * span + 1e-4)

    alphas = np.linspace(center_a - radius_eff, center_a + radius_eff, grid_points)
    betas = np.linspace(center_b - radius_eff, center_b + radius_eff, grid_points)
    A, B = np.meshgrid(alphas, betas)
    Z = np.zeros_like(A)

    tokens_dev = data_tokens.to(device)
    targets_dev = data_targets.to(device)

    for i in range(grid_points):
        for j in range(grid_points):
            a, b = float(A[i, j]), float(B[i, j])
            for p, t0, d1, d2 in zip(params, theta_0, dir1, dir2):
                p.data.copy_(t0 + a * d1.to(t0.device) + b * d2.to(t0.device))
            logits, _ = model(tokens_dev, return_attn=False)
            Z[i, j] = float(loss_fn(logits, targets_dev).item())

    for p, t_cur in zip(params, theta_curr):
        p.data.copy_(t_cur)

    z_cap = np.percentile(Z, 97)
    Z_clipped = np.clip(Z, None, z_cap)

    fig, ax = plt.subplots(figsize=(9.8, 8.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    cf = ax.contourf(A, B, Z_clipped, levels=40, cmap="magma")
    ax.contour(A, B, Z_clipped, levels=20, colors="#ffffff25", linewidths=0.45)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#c9d1d9")
    cb.set_label("Loss", color="#c9d1d9", fontsize=9)

    if trajectory:
        traj_arr = np.array(trajectory, dtype=np.float64)
        traj_epochs = traj_arr[:, 0]
        traj_a = traj_arr[:, 1]
        traj_b = traj_arr[:, 2]

        if shock_epoch is not None and traj_epochs.max() > traj_epochs.min():
            values = np.zeros_like(traj_epochs, dtype=np.float64)
            for i, ep in enumerate(traj_epochs):
                if ep <= shock_epoch:
                    values[i] = 0.5 * (ep - traj_epochs.min()) / max(shock_epoch - traj_epochs.min(), 1.0)
                else:
                    values[i] = 0.5 + 0.5 * (ep - shock_epoch) / max(traj_epochs.max() - shock_epoch, 1.0)
        else:
            values = (traj_epochs - traj_epochs.min()) / max(traj_epochs.max() - traj_epochs.min(), 1.0)

        traj_cmap = LinearSegmentedColormap.from_list(
            "time_byr",
            ["#2563eb", "#facc15", "#dc2626"],
            N=256,
        )
        points = np.column_stack([traj_a, traj_b]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap=traj_cmap,
            norm=plt.Normalize(0.0, 1.0),
            linewidths=2.0,
            alpha=0.95,
            zorder=4,
        )
        lc.set_array(values[:-1] if len(values) > 1 else values)
        ax.add_collection(lc)
        sc = ax.scatter(
            traj_a,
            traj_b,
            c=values,
            cmap=traj_cmap,
            s=16,
            edgecolors="#111111",
            linewidths=0.2,
            zorder=5,
        )
        ax.scatter(traj_a[0], traj_b[0], marker="o", color="#58a6ff", s=50, zorder=5, label="Start")
        ax.scatter(traj_a[-1], traj_b[-1], marker="X", color="#f85149", s=60, zorder=5, label="End")
        if shock_epoch is not None:
            shock_idx = int(np.argmin(np.abs(traj_epochs - float(shock_epoch))))
            ax.scatter(
                traj_a[shock_idx],
                traj_b[shock_idx],
                marker="*",
                color="#f6e05e",
                s=120,
                zorder=6,
                label=f"Shock (~epoch {int(traj_epochs[shock_idx])})",
            )

        cbt = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.09)
        cbt.ax.tick_params(colors="#c9d1d9")
        cbt.set_label("Trajectory Time (Blue→Yellow→Red)", color="#c9d1d9", fontsize=9)

    ax.set_title("Loss Landscape (Topographic Contour) + Weight Trajectory", fontsize=12, fontweight="bold", color="#c9d1d9", pad=10)
    ax.set_xlabel("Direction 1 (α)", fontsize=9, color="#8b949e")
    ax.set_ylabel("Direction 2 (β)", fontsize=9, color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.grid(alpha=0.12, color="#21262d")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.35)
    fig.tight_layout()
    return fig

