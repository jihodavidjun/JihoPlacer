"""Coarse route-demand congestion objective for JihoPlace v2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from jiho_place.v2.density import GridSize, bin_edges, macro_bin_utilization, normalize_grid_size
from jiho_place.v2.placement_state import PlacementState


def route_demand_overflow(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    grid_size: GridSize = (32, 32),
    target: float = 1.0,
    gamma: Optional[float] = None,
    blockage_factor: float = 0.65,
    min_capacity: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Estimate smooth route overflow from directional net bounding boxes."""

    macro_pos = state.positions if positions is None else positions
    rows, cols = normalize_grid_size(grid_size)
    if not state.nets:
        zero = macro_pos.sum() * 0.0
        return zero, _empty_stats()

    x0, x1, y0, y1 = bin_edges(state, (rows, cols))
    all_pos = state.owner_positions(macro_pos)
    temp = float(gamma) if gamma is not None else max(state.span * 0.020, 1.0e-3)
    h_demand = torch.zeros((rows, cols), dtype=state.dtype, device=state.device)
    v_demand = torch.zeros((rows, cols), dtype=state.dtype, device=state.device)
    active_nets = 0

    for net_id, owners in enumerate(state.nets):
        if int(owners.numel()) < 2:
            continue
        pts = all_pos.index_select(0, owners) + state.net_pin_offsets[net_id]
        xmax = temp * torch.logsumexp(pts[:, 0] / temp, dim=0)
        xmin = -temp * torch.logsumexp(-pts[:, 0] / temp, dim=0)
        ymax = temp * torch.logsumexp(pts[:, 1] / temp, dim=0)
        ymin = -temp * torch.logsumexp(-pts[:, 1] / temp, dim=0)
        width = torch.clamp(xmax - xmin, min=temp)
        height = torch.clamp(ymax - ymin, min=temp)

        ox = torch.relu(torch.minimum(xmax, x1) - torch.maximum(xmin, x0))
        oy = torch.relu(torch.minimum(ymax, y1) - torch.maximum(ymin, y0))
        x_cover = ox / torch.clamp(width, min=1.0e-6)
        y_cover = oy / torch.clamp(height, min=1.0e-6)
        bbox_cover = y_cover[:, None] * x_cover[None, :]

        degree_scale = max(float(owners.numel()) - 1.0, 1.0) ** 0.5
        weight = float(state.net_weights[net_id].item()) if net_id < int(state.net_weights.numel()) else 1.0
        net_weight = abs(weight) * degree_scale

        # Horizontal routes span the x direction through rows in the bbox;
        # vertical routes span y through columns in the bbox. The small bbox
        # cover term keeps gradients smooth near bin boundaries.
        h_len = width / max(float(state.canvas_width), 1.0e-6)
        v_len = height / max(float(state.canvas_height), 1.0e-6)
        h_demand = h_demand + net_weight * h_len * (0.75 * y_cover[:, None] + 0.25 * bbox_cover)
        v_demand = v_demand + net_weight * v_len * (0.75 * x_cover[None, :] + 0.25 * bbox_cover)
        active_nets += 1

    if active_nets == 0:
        zero = macro_pos.sum() * 0.0
        return zero, _empty_stats()

    macro_util = macro_bin_utilization(state, macro_pos, (rows, cols))
    blockage = torch.clamp(float(blockage_factor) * torch.clamp(macro_util, min=0.0, max=1.25), min=0.0, max=0.92)
    capacity_scale = torch.clamp(1.0 - blockage, min=float(min_capacity))

    h_active = h_demand[h_demand > 0.0]
    v_active = v_demand[v_demand > 0.0]
    h_base = h_active.mean().detach() if h_active.numel() else h_demand.mean().detach() + 1.0
    v_base = v_active.mean().detach() if v_active.numel() else v_demand.mean().detach() + 1.0
    h_capacity = torch.clamp(h_base * capacity_scale, min=1.0e-6)
    v_capacity = torch.clamp(v_base * capacity_scale, min=1.0e-6)

    h_ratio = h_demand / h_capacity
    v_ratio = v_demand / v_capacity
    ratio = torch.maximum(h_ratio, v_ratio)
    raw_overflow = ratio - float(target)
    smooth_overflow = torch.nn.functional.softplus(raw_overflow, beta=2.0)
    loss = torch.log1p(smooth_overflow).pow(2).mean()

    with torch.no_grad():
        demand = h_demand + v_demand
        capacity = torch.minimum(h_capacity, v_capacity)
        overflow = torch.relu(ratio - float(target))
        stats = {
            "demand_max": _max_or_zero(demand),
            "capacity_min": _min_or_zero(capacity),
            "overflow_max": _max_or_zero(overflow),
            "overflow_mean": _mean_or_zero(overflow),
            "hot_bin_count": int((overflow > 0.0).sum().detach().cpu().item()) if overflow.numel() else 0,
            "blockage_max": _max_or_zero(blockage),
            "h_demand_max": _max_or_zero(h_demand),
            "v_demand_max": _max_or_zero(v_demand),
            "congestion_max_overflow": _max_or_zero(overflow),
            "congestion_mean_overflow": _mean_or_zero(overflow),
            "congestion_hot_bins": int((overflow > 0.0).sum().detach().cpu().item()) if overflow.numel() else 0,
        }
    return loss, stats


def _empty_stats() -> Dict[str, float]:
    return {
        "demand_max": 0.0,
        "capacity_min": 0.0,
        "overflow_max": 0.0,
        "overflow_mean": 0.0,
        "hot_bin_count": 0,
        "blockage_max": 0.0,
        "h_demand_max": 0.0,
        "v_demand_max": 0.0,
        "congestion_max_overflow": 0.0,
        "congestion_mean_overflow": 0.0,
        "congestion_hot_bins": 0,
    }


def _max_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.max().detach().cpu().item()) if tensor.numel() else 0.0


def _min_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.min().detach().cpu().item()) if tensor.numel() else 0.0


def _mean_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.mean().detach().cpu().item()) if tensor.numel() else 0.0
