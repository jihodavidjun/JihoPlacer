"""Differentiable objective terms for JihoPlace v2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from jiho_place.v2.placement_state import PlacementState


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

    all_pos = state.owner_positions(macro_pos)
    temp = float(gamma) if gamma is not None else max(state.span * 0.015, 1.0e-3)
    total = macro_pos.sum() * 0.0
    weight_total = 0.0

    for net_id, owners in enumerate(state.nets):
        if int(owners.numel()) < 2:
            continue
        pts = all_pos.index_select(0, owners) + state.net_pin_offsets[net_id]
        x = pts[:, 0] / temp
        y = pts[:, 1] / temp
        hpwl = temp * (
            torch.logsumexp(x, dim=0)
            + torch.logsumexp(-x, dim=0)
            + torch.logsumexp(y, dim=0)
            + torch.logsumexp(-y, dim=0)
        )
        weight = float(state.net_weights[net_id].item()) if net_id < int(state.net_weights.numel()) else 1.0
        total = total + weight * hpwl
        weight_total += abs(weight)

    if weight_total <= 0.0:
        return macro_pos.sum() * 0.0
    return total / (weight_total * state.span)


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
