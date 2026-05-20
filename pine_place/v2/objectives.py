"""Differentiable objective terms for PinePlace v2."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from pine_place.v2.placement_state import PlacementState


def smooth_hpwl(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Log-sum-exp HPWL over variable-size nets.

    Pin-level offsets are applied when `PlacementState` was built from
    `Benchmark.net_pin_nodes`.
    """

    macro_pos = state.positions if positions is None else positions
    if not state.nets:
        return macro_pos.sum() * 0.0

    net_node_idx, net_mask, net_offsets, net_weights = _batched_net_data(state)
    if int(net_node_idx.shape[0]) == 0:
        return macro_pos.sum() * 0.0
    all_pos = state.owner_positions(macro_pos)
    temp = float(gamma) if gamma is not None else max(state.span * 0.015, 1.0e-3)
    pts = all_pos.index_select(0, net_node_idx.reshape(-1)).reshape(net_node_idx.shape[0], net_node_idx.shape[1], 2)
    pts = pts + net_offsets
    neg_inf = torch.tensor(float("-inf"), dtype=pts.dtype, device=pts.device)
    x = pts[:, :, 0] / temp
    y = pts[:, :, 1] / temp
    hpwl = temp * (
        torch.logsumexp(torch.where(net_mask, x, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, -x, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, y, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, -y, neg_inf), dim=1)
    )
    weights = torch.abs(net_weights.to(device=pts.device, dtype=pts.dtype))
    weight_total = torch.clamp(weights.sum(), min=1.0e-12)
    return (weights * hpwl).sum() / (weight_total * state.span)


def default_lse_gamma(state: PlacementState) -> float:
    """Inverse-temperature gamma for refined log-sum-exp HPWL."""

    if hasattr(state, "net_mask"):
        active_nets = max(int(getattr(state, "net_mask").shape[0]), 1)
    else:
        active_nets = max(sum(1 for owners in state.nets if int(owners.numel()) >= 2), 1)
    chip_dimension = math.sqrt(max(float(state.canvas_width) * float(state.canvas_height), 1.0e-12))
    return 1.0 / max(0.1 * chip_dimension / math.sqrt(float(active_nets)), 1.0e-6)


def smooth_hpwl_lse(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Log-sum-exp HPWL using gamma as the inverse smoothing temperature."""

    macro_pos = state.positions if positions is None else positions
    if not state.nets:
        return macro_pos.sum() * 0.0

    net_node_idx, net_mask, net_offsets, net_weights = _batched_net_data(state)
    if int(net_node_idx.shape[0]) == 0:
        return macro_pos.sum() * 0.0
    all_pos = state.owner_positions(macro_pos)
    inv_temp = float(gamma) if gamma is not None else default_lse_gamma(state)
    inv_temp = max(inv_temp, 1.0e-9)
    pts = all_pos.index_select(0, net_node_idx.reshape(-1)).reshape(net_node_idx.shape[0], net_node_idx.shape[1], 2)
    pts = pts + net_offsets
    neg_inf = torch.tensor(float("-inf"), dtype=pts.dtype, device=pts.device)
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    hpwl = (
        torch.logsumexp(torch.where(net_mask, inv_temp * x, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, -inv_temp * x, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, inv_temp * y, neg_inf), dim=1)
        + torch.logsumexp(torch.where(net_mask, -inv_temp * y, neg_inf), dim=1)
    ) / inv_temp
    weights = torch.abs(net_weights.to(device=pts.device, dtype=pts.dtype))
    weight_total = torch.clamp(weights.sum(), min=1.0e-12)
    return (weights * hpwl).sum() / (weight_total * state.span)


def _batched_net_data(
    state: PlacementState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(
        hasattr(state, name)
        for name in ("net_node_idx", "net_mask", "net_pin_offset_tensor", "net_weight_tensor")
    ):
        _build_batched_net_data(state)
    return (
        getattr(state, "net_node_idx"),
        getattr(state, "net_mask"),
        getattr(state, "net_pin_offset_tensor"),
        getattr(state, "net_weight_tensor"),
    )


def _build_batched_net_data(state: PlacementState) -> None:
    owner_count = state.num_macros + int(state.port_positions.shape[0])
    usable = []
    for net_id, owners in enumerate(state.nets):
        if int(owners.numel()) < 2:
            continue
        owners = owners.to(device=state.device, dtype=torch.long).flatten()
        offsets = state.net_pin_offsets[net_id].to(device=state.device, dtype=state.dtype)
        valid = (owners >= 0) & (owners < owner_count)
        owners = owners[valid]
        offsets = offsets[valid]
        if int(owners.numel()) < 2:
            continue
        weight = state.net_weights[net_id] if net_id < int(state.net_weights.numel()) else torch.tensor(
            1.0, dtype=state.dtype, device=state.device
        )
        usable.append((owners, offsets, weight.to(device=state.device, dtype=state.dtype)))

    if not usable:
        setattr(state, "net_node_idx", torch.zeros((0, 1), dtype=torch.long, device=state.device))
        setattr(state, "net_mask", torch.zeros((0, 1), dtype=torch.bool, device=state.device))
        setattr(state, "net_pin_offset_tensor", torch.zeros((0, 1, 2), dtype=state.dtype, device=state.device))
        setattr(state, "net_weight_tensor", torch.zeros((0,), dtype=state.dtype, device=state.device))
        return

    max_degree = max(int(owners.numel()) for owners, _offsets, _weight in usable)
    count = len(usable)
    net_node_idx = torch.zeros((count, max_degree), dtype=torch.long, device=state.device)
    net_mask = torch.zeros((count, max_degree), dtype=torch.bool, device=state.device)
    net_offsets = torch.zeros((count, max_degree, 2), dtype=state.dtype, device=state.device)
    net_weights = torch.empty((count,), dtype=state.dtype, device=state.device)
    for row, (owners, offsets, weight) in enumerate(usable):
        degree = int(owners.numel())
        net_node_idx[row, :degree] = owners
        net_mask[row, :degree] = True
        net_offsets[row, :degree] = offsets
        net_weights[row] = weight

    setattr(state, "net_node_idx", net_node_idx)
    setattr(state, "net_mask", net_mask)
    setattr(state, "net_pin_offset_tensor", net_offsets)
    setattr(state, "net_weight_tensor", net_weights)


def soft_boundary_penalty(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft out-of-bounds penalty for center-coordinate placements."""

    macro_pos = state.positions if positions is None else positions
    half = state.sizes * 0.5
    x_low = torch.relu(half[:, 0] - macro_pos[:, 0])
    x_high = torch.relu(macro_pos[:, 0] + half[:, 0] - float(state.canvas_width))
    y_low = torch.relu(half[:, 1] - macro_pos[:, 1])
    y_high = torch.relu(macro_pos[:, 1] + half[:, 1] - float(state.canvas_height))
    violation = torch.stack((x_low, x_high, y_low, y_high), dim=1)
    if bool(state.movable_mask.any()):
        violation = violation[state.movable_mask]
    return (violation / state.span).pow(2).sum()


def boundary_penalty(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> torch.Tensor:
    macro_pos = state.positions if positions is None else positions
    if not bool(state.movable_mask.any()):
        return macro_pos.sum() * 0.0

    half = state.sizes * 0.5
    low = half + float(margin)
    high = state.canvas_size.view(1, 2) - half - float(margin)
    violation = torch.relu(low - macro_pos) + torch.relu(macro_pos - high)
    violation = violation[state.movable_mask]
    return (violation / state.span).pow(2).mean()


def hard_macro_overlap_penalty(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Pairwise differentiable hard-macro overlap penalty with chunking."""

    macro_pos = state.positions if positions is None else positions
    hard_idx = torch.nonzero(state.hard_mask, as_tuple=False).flatten()
    if int(hard_idx.numel()) <= 1:
        return macro_pos.sum() * 0.0

    hard_pos = macro_pos.index_select(0, hard_idx)
    hard_sizes = state.sizes.index_select(0, hard_idx)
    hard_movable = state.movable_mask.index_select(0, hard_idx)
    mean_area = torch.clamp((hard_sizes[:, 0] * hard_sizes[:, 1]).mean().detach(), min=1.0e-6)

    total = macro_pos.sum() * 0.0
    count = 0
    n = int(hard_idx.numel())
    step = max(1, int(chunk_size))

    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        pi = hard_pos[i0:i1]
        si = hard_sizes[i0:i1]
        mi = hard_movable[i0:i1]
        for j0 in range(i0, n, step):
            j1 = min(n, j0 + step)
            pj = hard_pos[j0:j1]
            sj = hard_sizes[j0:j1]
            mj = hard_movable[j0:j1]

            ox = torch.relu((si[:, None, 0] + sj[None, :, 0]) * 0.5 - torch.abs(pi[:, None, 0] - pj[None, :, 0]))
            oy = torch.relu((si[:, None, 1] + sj[None, :, 1]) * 0.5 - torch.abs(pi[:, None, 1] - pj[None, :, 1]))
            overlap_area = ox * oy

            pair_mask = mi[:, None] | mj[None, :]
            if i0 == j0:
                upper = torch.triu(
                    torch.ones((i1 - i0, j1 - j0), dtype=torch.bool, device=state.device),
                    diagonal=1,
                )
                pair_mask = pair_mask & upper
            overlap_area = overlap_area[pair_mask]
            if overlap_area.numel() == 0:
                continue
            total = total + (overlap_area / mean_area).pow(2).sum()
            count += int(overlap_area.numel())

    if count == 0:
        return macro_pos.sum() * 0.0
    return total / float(count)


def objective_snapshot(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    gamma: Optional[float] = None,
) -> Dict[str, float]:
    macro_pos = state.positions if positions is None else positions
    with torch.no_grad():
        return {
            "smooth_hpwl": float(smooth_hpwl(state, macro_pos, gamma).detach().cpu().item()),
            "boundary": float(boundary_penalty(state, macro_pos).detach().cpu().item()),
            "hard_overlap": float(hard_macro_overlap_penalty(state, macro_pos).detach().cpu().item()),
        }


def boundary_and_overlap(
    state: PlacementState,
    positions: Optional[torch.Tensor] = None,
    overlap_chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    macro_pos = state.positions if positions is None else positions
    return (
        boundary_penalty(state, macro_pos),
        hard_macro_overlap_penalty(state, macro_pos, chunk_size=overlap_chunk_size),
    )
