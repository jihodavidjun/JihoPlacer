"""Hard-macro legalization for PinePlace v2 candidate placements."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

from pine_place.v2.placement_state import PlacementState


def legalize_placement(
    state: PlacementState,
    positions: torch.Tensor,
    max_iters: int = 200,
    padding: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Repair hard-macro overlaps while preserving fixed macros and bounds."""

    pos = positions.detach().clone().to(device=state.device, dtype=state.dtype)
    pos = _clip_bounds(state, pos)
    pos[state.fixed_mask] = state.positions[state.fixed_mask]
    start_pos = pos.clone()

    hard_idx = torch.nonzero(state.hard_mask, as_tuple=False).flatten().tolist()
    initial_overlaps = _collect_overlaps(state, pos, hard_idx, padding)
    repairs = 0
    spiral_repairs = 0
    stalled_rounds = 0

    for _iter in range(max(0, int(max_iters))):
        overlaps = _collect_overlaps(state, pos, hard_idx, padding)
        if not overlaps:
            break
        before_count = len(overlaps)
        moved_this_iter = 0
        moved_targets = set()
        for item in overlaps:
            i = int(item["i"])
            j = int(item["j"])
            if not (bool(state.movable_mask[i]) or bool(state.movable_mask[j])):
                continue
            if _separate_pair(state, pos, i, j, float(item["overlap_x"]), float(item["overlap_y"])):
                moved_this_iter += 1
                repairs += 1
                moved_targets.update(_movable_repair_targets(state, i, j))
                pos = _clip_bounds(state, pos)
                pos[state.fixed_mask] = state.positions[state.fixed_mask]

        after_push = _collect_overlaps(state, pos, hard_idx, padding)
        if after_push and (moved_this_iter == 0 or len(after_push) >= before_count):
            for macro_idx in _offending_macros(after_push, state):
                if macro_idx in moved_targets or not bool(state.movable_mask[macro_idx]):
                    continue
                if _spiral_relocate(state, pos, macro_idx, hard_idx, padding):
                    spiral_repairs += 1
                    moved_this_iter += 1
                    pos = _clip_bounds(state, pos)
                    pos[state.fixed_mask] = state.positions[state.fixed_mask]
                    break

        new_count = len(_collect_overlaps(state, pos, hard_idx, padding))
        if new_count >= before_count and moved_this_iter == 0:
            stalled_rounds += 1
        else:
            stalled_rounds = 0
        if stalled_rounds >= 3:
            break

    final_overlaps = _collect_overlaps(state, pos, hard_idx, padding)
    disp = torch.linalg.norm((pos - start_pos).detach(), dim=1)
    moved = disp > 1.0e-6
    failed_pairs = [
        f"{int(item['i'])}-{int(item['j'])}:{float(item['area']):.6g}"
        for item in final_overlaps[:8]
    ]
    max_overlap_area = max((float(item["area"]) for item in final_overlaps), default=0.0)

    stats = {
        "initial_overlap_count": float(len(initial_overlaps)),
        "final_overlap_count": float(len(final_overlaps)),
        "max_overlap_area": float(max_overlap_area),
        "moved_macro_count": float(moved.sum().detach().cpu().item()),
        "mean_displacement": float(disp[moved].mean().detach().cpu().item()) if bool(moved.any()) else 0.0,
        "max_displacement": float(disp.max().detach().cpu().item()) if disp.numel() else 0.0,
        "failed_pairs_sample": ",".join(failed_pairs),
        "legalization_repairs": float(repairs),
        "legalization_spiral_repairs": float(spiral_repairs),
        "legalization_remaining_overlaps": float(len(final_overlaps)),
        "legalization_max_overlap_area": float(max_overlap_area),
    }
    return pos, stats


def _clip_bounds(state: PlacementState, positions: torch.Tensor, epsilon: float = 1.0e-4) -> torch.Tensor:
    half = state.sizes * 0.5
    low = half + float(epsilon)
    high = state.canvas_size.view(1, 2) - half - float(epsilon)
    feasible = high >= low
    midpoint = state.canvas_size.view(1, 2) * 0.5
    clipped = torch.minimum(torch.maximum(positions, torch.minimum(low, high)), torch.maximum(low, high))
    return torch.where(feasible, clipped, midpoint.expand_as(clipped))


def _overlap_amount(
    state: PlacementState,
    positions: torch.Tensor,
    i: int,
    j: int,
    padding: float,
) -> Tuple[float, float]:
    dx = abs(float(positions[i, 0].item() - positions[j, 0].item()))
    dy = abs(float(positions[i, 1].item() - positions[j, 1].item()))
    sep_x = float((state.sizes[i, 0] + state.sizes[j, 0]).item()) * 0.5 + float(padding)
    sep_y = float((state.sizes[i, 1] + state.sizes[j, 1]).item()) * 0.5 + float(padding)
    return max(0.0, sep_x - dx), max(0.0, sep_y - dy)


def _collect_overlaps(
    state: PlacementState,
    positions: torch.Tensor,
    hard_idx: Sequence[int],
    padding: float,
) -> List[Dict[str, float]]:
    overlaps: List[Dict[str, float]] = []
    for a_pos, i in enumerate(hard_idx):
        for j in hard_idx[a_pos + 1 :]:
            overlap_x, overlap_y = _overlap_amount(state, positions, i, j, padding)
            if overlap_x <= 0.0 or overlap_y <= 0.0:
                continue
            area = overlap_x * overlap_y
            overlaps.append(
                {
                    "i": float(i),
                    "j": float(j),
                    "overlap_x": float(overlap_x),
                    "overlap_y": float(overlap_y),
                    "area": float(area),
                }
            )
    overlaps.sort(key=lambda item: float(item["area"]), reverse=True)
    return overlaps


def _separate_pair(
    state: PlacementState,
    positions: torch.Tensor,
    i: int,
    j: int,
    overlap_x: float,
    overlap_y: float,
) -> bool:
    axis = 0 if overlap_x <= overlap_y else 1
    amount = (overlap_x if axis == 0 else overlap_y) + 1.0e-3
    delta = float(positions[i, axis].item() - positions[j, axis].item())
    if abs(delta) < 1.0e-9:
        delta = -1.0 if i < j else 1.0
    direction = 1.0 if delta > 0.0 else -1.0

    i_movable = bool(state.movable_mask[i])
    j_movable = bool(state.movable_mask[j])
    if not (i_movable or j_movable):
        return False

    i_area = float((state.sizes[i, 0] * state.sizes[i, 1]).item())
    j_area = float((state.sizes[j, 0] * state.sizes[j, 1]).item())
    total_area = max(i_area + j_area, 1.0e-12)

    if i_movable and j_movable and max(i_area, j_area) > 2.5 * max(min(i_area, j_area), 1.0e-12):
        if i_area <= j_area:
            positions[i, axis] += direction * amount
        else:
            positions[j, axis] -= direction * amount
    elif i_movable and j_movable:
        positions[i, axis] += direction * amount * (j_area / total_area)
        positions[j, axis] -= direction * amount * (i_area / total_area)
    elif i_movable:
        positions[i, axis] += direction * amount
    elif j_movable:
        positions[j, axis] -= direction * amount
    return True


def _movable_repair_targets(state: PlacementState, i: int, j: int) -> List[int]:
    targets = []
    if bool(state.movable_mask[i]):
        targets.append(i)
    if bool(state.movable_mask[j]):
        targets.append(j)
    return targets


def _offending_macros(overlaps: Sequence[Dict[str, float]], state: PlacementState) -> List[int]:
    severity: Dict[int, float] = {}
    for item in overlaps:
        for key in ("i", "j"):
            idx = int(item[key])
            if bool(state.movable_mask[idx]):
                area = float((state.sizes[idx, 0] * state.sizes[idx, 1]).item())
                severity[idx] = severity.get(idx, 0.0) + float(item["area"]) / max(area, 1.0e-12)
    return [idx for idx, _score in sorted(severity.items(), key=lambda pair: pair[1], reverse=True)]


def _spiral_relocate(
    state: PlacementState,
    positions: torch.Tensor,
    macro_idx: int,
    hard_idx: Sequence[int],
    padding: float,
) -> bool:
    if not bool(state.movable_mask[macro_idx]):
        return False

    original = positions[macro_idx].clone()
    span = max(float(state.canvas_width), float(state.canvas_height), 1.0e-6)
    macro_step = max(float(state.sizes[macro_idx].min().item()) * 0.35, span * 0.004, 1.0e-4)
    best = None
    best_score = float("inf")

    for ring in range(1, 9):
        radius = macro_step * ring
        samples = max(12, ring * 10)
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            candidate = original + torch.tensor(
                [math.cos(angle) * radius, math.sin(angle) * radius],
                dtype=positions.dtype,
                device=positions.device,
            )
            trial = positions.clone()
            trial[macro_idx] = candidate
            trial = _clip_bounds(state, trial)
            trial[state.fixed_mask] = state.positions[state.fixed_mask]
            overlap_area, overlap_count = _macro_overlap_score(state, trial, macro_idx, hard_idx, padding)
            disp = float(torch.linalg.norm((trial[macro_idx] - original).detach()).cpu().item())
            score = overlap_area * 1.0e6 + overlap_count * 1.0e3 + disp
            if score < best_score:
                best_score = score
                best = trial[macro_idx].clone()
            if overlap_count == 0:
                positions[macro_idx] = trial[macro_idx]
                return True

    if best is not None:
        old_area, old_count = _macro_overlap_score(state, positions, macro_idx, hard_idx, padding)
        trial = positions.clone()
        trial[macro_idx] = best
        new_area, new_count = _macro_overlap_score(state, trial, macro_idx, hard_idx, padding)
        if (new_count, new_area) < (old_count, old_area):
            positions[macro_idx] = best
            return True
    return False


def _macro_overlap_score(
    state: PlacementState,
    positions: torch.Tensor,
    macro_idx: int,
    hard_idx: Sequence[int],
    padding: float,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    for other in hard_idx:
        if other == macro_idx:
            continue
        overlap_x, overlap_y = _overlap_amount(state, positions, macro_idx, other, padding)
        if overlap_x > 0.0 and overlap_y > 0.0:
            count += 1
            total += overlap_x * overlap_y
    return total, count
