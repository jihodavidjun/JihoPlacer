"""Area-overlap bin density objective for PinePlace v2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from pine_place.v2.placement_state import PlacementState


GridSize = Union[int, Tuple[int, int]]


def normalize_grid_size(grid_size: GridSize) -> Tuple[int, int]:
    if isinstance(grid_size, int):
        return int(grid_size), int(grid_size)
    rows, cols = grid_size
    return int(rows), int(cols)


def bin_edges(
    state: PlacementState,
    grid_size: GridSize,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = normalize_grid_size(grid_size)
    x_edges = torch.linspace(0.0, state.canvas_width, cols + 1, dtype=state.dtype, device=state.device)
    y_edges = torch.linspace(0.0, state.canvas_height, rows + 1, dtype=state.dtype, device=state.device)
    return x_edges[:-1], x_edges[1:], y_edges[:-1], y_edges[1:]


def axis_overlap(
    centers: torch.Tensor,
    half_sizes: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    low = centers[:, None] - half_sizes[:, None]
    high = centers[:, None] + half_sizes[:, None]
    left = torch.maximum(low, starts[None, :])
    right = torch.minimum(high, ends[None, :])
    return torch.relu(right - left)


def macro_bin_utilization(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    grid_size: GridSize = (32, 32),
) -> torch.Tensor:
    macro_pos = state.positions if positions is None else positions
    rows, cols = normalize_grid_size(grid_size)
    x0, x1, y0, y1 = bin_edges(state, (rows, cols))
    ox = axis_overlap(macro_pos[:, 0], state.sizes[:, 0] * 0.5, x0, x1)
    oy = axis_overlap(macro_pos[:, 1], state.sizes[:, 1] * 0.5, y0, y1)
    area = oy.transpose(0, 1).matmul(ox)
    bin_area = max(float(state.canvas_width) / cols * float(state.canvas_height) / rows, 1.0e-6)
    return area / bin_area


def density_overflow(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    grid_size: GridSize = (32, 32),
    target_util: float = 0.85,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    util = macro_bin_utilization(state, positions, grid_size)
    overflow = torch.relu(util - float(target_util))
    loss = overflow.pow(2).mean()
    with torch.no_grad():
        stats = {
            "density_max_util": float(util.max().detach().cpu().item()) if util.numel() else 0.0,
            "density_mean_overflow": float(overflow.mean().detach().cpu().item()) if overflow.numel() else 0.0,
            "density_overflow_bins": int((overflow > 0.0).sum().detach().cpu().item()) if overflow.numel() else 0,
        }
    return loss, stats


def gaussian_density_penalty(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    grid_size: Optional[GridSize] = None,
    target_density: Optional[float] = None,
    sigma_scale: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable bell-curve macro density over a regular grid."""

    macro_pos = state.positions if positions is None else positions
    if grid_size is None:
        side = 64 if str(state.metadata.get("benchmark_name", "")).lower().startswith("ibm") else 32
        rows = cols = side
    else:
        rows, cols = normalize_grid_size(grid_size)
    rows = max(1, rows)
    cols = max(1, cols)

    x_centers = torch.linspace(
        float(state.canvas_width) / (2.0 * cols),
        float(state.canvas_width) * (1.0 - 1.0 / (2.0 * cols)),
        cols,
        dtype=state.dtype,
        device=state.device,
    )
    y_centers = torch.linspace(
        float(state.canvas_height) / (2.0 * rows),
        float(state.canvas_height) * (1.0 - 1.0 / (2.0 * rows)),
        rows,
        dtype=state.dtype,
        device=state.device,
    )
    cell_w = float(state.canvas_width) / cols
    cell_h = float(state.canvas_height) / rows
    sigma_x = max(float(sigma_scale) * cell_w, 1.0e-6)
    sigma_y = max(float(sigma_scale) * cell_h, 1.0e-6)

    dx = (macro_pos[:, 0:1] - x_centers[None, :]) / sigma_x
    dy = (macro_pos[:, 1:2] - y_centers[None, :]) / sigma_y
    wx = torch.exp(-0.5 * dx.pow(2))
    wy = torch.exp(-0.5 * dy.pow(2))
    wx = wx / torch.clamp(wx.sum(dim=1, keepdim=True), min=1.0e-12)
    wy = wy / torch.clamp(wy.sum(dim=1, keepdim=True), min=1.0e-12)

    area = state.sizes[:, 0] * state.sizes[:, 1]
    density_area = wy.transpose(0, 1).matmul(area[:, None] * wx)
    cell_area = max(cell_w * cell_h, 1.0e-12)
    density = density_area / cell_area
    if target_density is None:
        canvas_area = max(float(state.canvas_width) * float(state.canvas_height), 1.0e-12)
        target = float(area.detach().sum().cpu().item()) / (canvas_area * 0.8)
    else:
        target = float(target_density)
    overflow = torch.relu(density - target)
    loss = overflow.pow(2).sum() / float(rows * cols)
    with torch.no_grad():
        stats = {
            "gaussian_density_max": float(density.max().detach().cpu().item()) if density.numel() else 0.0,
            "gaussian_density_mean_overflow": float(overflow.mean().detach().cpu().item()) if overflow.numel() else 0.0,
            "gaussian_density_hot_bins": int((overflow > 0.0).sum().detach().cpu().item()) if overflow.numel() else 0,
            "gaussian_density_target": float(target),
        }
    return loss, stats
