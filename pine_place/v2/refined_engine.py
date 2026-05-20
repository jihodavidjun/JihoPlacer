"""Warm-started v2 analytical refinement for v1 placements."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from pine_place.v2.congestion import differentiable_routing_congestion
from pine_place.v2.density import gaussian_density_penalty
from pine_place.v2.legalization import legalize_placement
from pine_place.v2.objectives import default_lse_gamma, smooth_hpwl_lse, soft_boundary_penalty
from pine_place.v2.placement_state import PlacementState


DeviceLike = Optional[Union[str, torch.device]]


@dataclass(frozen=True)
class _StageWeights:
    density: float
    congestion: float
    lr_scale: float


class V2RefinedEngine:
    """Conservative DREAMPlace-style post-processor for a legal v1 placement."""

    def __init__(
        self,
        device: DeviceLike = None,
        iterations: int = 1200,
        log_every: int = 50,
        max_nets: Optional[int] = None,
        hpwl_weight: float = 1.0,
    ) -> None:
        self.device = self._resolve_device(device)
        self.iterations = max(0, int(iterations))
        self.log_every = max(1, int(log_every))
        self.max_nets = max_nets
        self.hpwl_weight = float(hpwl_weight)
        self.logs: List[Dict[str, float]] = []
        self.legalization_stats: Dict[str, Any] = {}
        self.proxy_stats: Dict[str, float] = {}
        self._state_cache: Dict[int, PlacementState] = {}

    def refine(self, v1_placement: torch.Tensor, benchmark: Any, plc: Any) -> torch.Tensor:
        if plc is None:
            self.logs = [{"stage": -1.0, "step": 0.0, "skipped_no_plc": 1.0}]
            return v1_placement.detach().cpu().clone()

        from macro_place.objective import compute_proxy_cost

        state = self._cached_state(benchmark)

        initial = v1_placement.detach().to(device=state.device, dtype=state.dtype).clone()
        if tuple(initial.shape) != tuple(state.positions.shape):
            self.logs = [{"stage": -1.0, "step": 0.0, "invalid_shape": 1.0}]
            return v1_placement.detach().cpu().clone()

        state = state.with_positions(initial.clone())
        self._build_net_data(state)
        initial = self._clip_to_bounds(state, initial)
        initial[state.fixed_mask] = state.positions[state.fixed_mask]
        param = torch.nn.Parameter(initial.clone())
        base_lr = self._base_lr(state)
        optimizer = torch.optim.SGD([param], lr=base_lr, momentum=0.9, nesterov=True)
        gamma = default_lse_gamma(state)
        grid_size = self._density_grid_size(state)
        congestion_grid = self._congestion_grid_size(state)
        self.logs = []

        with torch.no_grad():
            param.data = self._clip_to_bounds(state, param.data)
            param.data[state.fixed_mask] = state.positions[state.fixed_mask]

        for step in range(self.iterations):
            stage_id, weights = self._stage_weights(step)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * weights.lr_scale

            iter_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            pos = torch.where(state.movable_mask[:, None], param, state.positions)
            hpwl = smooth_hpwl_lse(state, pos, gamma=gamma)
            density, density_stats = gaussian_density_penalty(state, pos, grid_size=grid_size)
            if weights.congestion > 0.0:
                congestion, congestion_stats = differentiable_routing_congestion(
                    state,
                    pos,
                    grid_size=congestion_grid,
                    capacity=1.5,
                    gamma=gamma,
                )
            else:
                congestion = pos.sum() * 0.0
                congestion_stats = {}
            boundary = soft_boundary_penalty(state, pos)
            loss = (
                self.hpwl_weight * hpwl
                + weights.density * density
                + weights.congestion * congestion
                + (10.0 * weights.density) * boundary
            )
            forward_s = time.time() - iter_t0

            if not torch.isfinite(loss):
                self._append_log(step, stage_id, loss, hpwl, density, congestion, boundary, weights, base_lr, param, initial, state)
                break

            backward_t0 = time.time()
            loss.backward()
            grad_norm = self._masked_norm(param.grad, state.movable_mask) if param.grad is not None else 0.0
            backward_s = time.time() - backward_t0
            before = param.detach().clone()
            step_t0 = time.time()
            optimizer.step()
            with torch.no_grad():
                param.data = self._clip_to_bounds(state, param.data)
                param.data[state.fixed_mask] = state.positions[state.fixed_mask]
            step_norm = self._masked_norm(param.detach() - before, state.movable_mask)
            step_s = time.time() - step_t0

            repair_s = 0.0
            if self._should_soft_repair(step):
                repair_t0 = time.time()
                with torch.no_grad():
                    repaired = self._soft_overlap_repair(state, param.detach(), gamma=gamma)
                    param.data.copy_(repaired)
                    param.data = self._clip_to_bounds(state, param.data)
                    param.data[state.fixed_mask] = state.positions[state.fixed_mask]
                    self._clear_momentum(optimizer)
                repair_s = time.time() - repair_t0

            if step == 0 or step == self.iterations - 1 or step % self.log_every == 0:
                self._append_log(
                    step,
                    stage_id,
                    loss,
                    hpwl,
                    density,
                    congestion,
                    boundary,
                    weights,
                    base_lr,
                    param,
                    initial,
                    state,
                    grad_norm=grad_norm,
                    step_norm=step_norm,
                    extra={
                        **density_stats,
                        **congestion_stats,
                        "forward_s": forward_s,
                        "backward_s": backward_s,
                        "step_s": step_s,
                        "repair_s": repair_s,
                    },
                )

        continuous = torch.where(state.movable_mask[:, None], param.detach(), state.positions)
        continuous = self._clip_to_bounds(state, continuous)
        continuous[state.fixed_mask] = state.positions[state.fixed_mask]
        legalized, legal_stats = legalize_placement(state, continuous)
        self.legalization_stats = legal_stats

        try:
            continuous_cost = compute_proxy_cost(continuous.detach().cpu(), benchmark, plc)
            legalized_cost = compute_proxy_cost(legalized.detach().cpu(), benchmark, plc)
        except Exception:
            self.proxy_stats = {"proxy_eval_failed": 1.0}
            return v1_placement.detach().cpu().clone()

        continuous_proxy = float(continuous_cost.get("proxy_cost", float("inf")))
        legalized_proxy = float(legalized_cost.get("proxy_cost", float("inf")))
        self.proxy_stats = {
            "continuous_proxy": continuous_proxy,
            "legalized_proxy": legalized_proxy,
            "legalization_ratio": legalized_proxy / max(continuous_proxy, 1.0e-12),
            "legalized_overlaps": float(legalized_cost.get("overlap_count", 0.0)),
        }
        if int(legalized_cost.get("overlap_count", 0)) > 0:
            self.proxy_stats["returned_v1_due_to_overlap"] = 1.0
            return v1_placement.detach().cpu().clone()
        if not math.isfinite(legalized_proxy) or legalized_proxy > 1.15 * continuous_proxy:
            self.proxy_stats["returned_v1_due_to_legalization"] = 1.0
            return v1_placement.detach().cpu().clone()
        return legalized.detach().cpu().to(dtype=v1_placement.detach().cpu().dtype)

    def _stage_weights(self, step: int) -> Tuple[int, _StageWeights]:
        if step < 200:
            return 1, _StageWeights(density=1.0e-3, congestion=0.0, lr_scale=1.0)
        if step < 600:
            frac = (step - 200) / 400.0
            return 2, _StageWeights(density=1.0e-3 + frac * (0.1 - 1.0e-3), congestion=0.0, lr_scale=1.0)
        if step < 1000:
            frac = (step - 600) / 400.0
            return 3, _StageWeights(density=0.1, congestion=frac * 0.05, lr_scale=1.0)
        return 4, _StageWeights(density=0.1, congestion=0.05, lr_scale=0.1)

    def _cached_state(self, benchmark: Any) -> PlacementState:
        key = id(benchmark)
        state = self._state_cache.get(key)
        if state is None:
            state = PlacementState.from_benchmark(benchmark, device=self.device)
            state.metadata["grid_rows"] = int(getattr(benchmark, "grid_rows", 32))
            state.metadata["grid_cols"] = int(getattr(benchmark, "grid_cols", getattr(benchmark, "grid_rows", 32)))
            if self.max_nets is not None and int(self.max_nets) > 0:
                state = state.limited_nets(int(self.max_nets))
            self._state_cache[key] = state
        return state

    def _build_net_data(self, state: PlacementState) -> None:
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
            state.net_node_idx = torch.zeros((0, 1), dtype=torch.long, device=state.device)
            state.net_mask = torch.zeros((0, 1), dtype=torch.bool, device=state.device)
            state.net_pin_offset_tensor = torch.zeros((0, 1, 2), dtype=state.dtype, device=state.device)
            state.net_weight_tensor = torch.zeros((0,), dtype=state.dtype, device=state.device)
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

        state.net_node_idx = net_node_idx
        state.net_mask = net_mask
        state.net_pin_offset_tensor = net_offsets
        state.net_weight_tensor = net_weights

    def _should_soft_repair(self, step: int) -> bool:
        return 200 <= step < 1000 and (step + 1) % 100 == 0

    def _soft_overlap_repair(self, state: PlacementState, positions: torch.Tensor, gamma: float) -> torch.Tensor:
        pos = positions.detach().clone()
        hard_idx = torch.nonzero(state.hard_mask, as_tuple=False).flatten().tolist()
        half = state.sizes * 0.5
        for _pass in range(3):
            moved = False
            for a, i in enumerate(hard_idx):
                for j in hard_idx[a + 1 :]:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    ox = (half[i, 0] + half[j, 0]) - torch.abs(dx)
                    oy = (half[i, 1] + half[j, 1]) - torch.abs(dy)
                    if float(ox.item()) <= 0.0 or float(oy.item()) <= 0.0:
                        continue
                    if not (bool(state.movable_mask[i]) or bool(state.movable_mask[j])):
                        continue
                    axis = 0 if float(ox.item()) <= float(oy.item()) else 1
                    amount = float((ox if axis == 0 else oy).item()) * 0.5 + 1.0e-4
                    candidates = []
                    for sign in (-1.0, 1.0):
                        trial = pos.clone()
                        self._apply_pair_shift(state, trial, i, j, axis, amount, sign)
                        trial = self._clip_to_bounds(state, trial)
                        trial[state.fixed_mask] = state.positions[state.fixed_mask]
                        hpwl = smooth_hpwl_lse(state, trial, gamma=gamma)
                        candidates.append((float(hpwl.detach().cpu().item()), trial))
                    best_trial = min(candidates, key=lambda item: item[0])[1]
                    if not torch.allclose(best_trial, pos):
                        pos = best_trial
                        moved = True
            if not moved:
                break
        return pos

    def _apply_pair_shift(
        self,
        state: PlacementState,
        positions: torch.Tensor,
        i: int,
        j: int,
        axis: int,
        amount: float,
        sign: float,
    ) -> None:
        i_movable = bool(state.movable_mask[i])
        j_movable = bool(state.movable_mask[j])
        if i_movable and j_movable:
            positions[i, axis] += sign * amount * 0.5
            positions[j, axis] -= sign * amount * 0.5
        elif i_movable:
            positions[i, axis] += sign * amount
        elif j_movable:
            positions[j, axis] -= sign * amount

    def _append_log(
        self,
        step: int,
        stage_id: int,
        loss: torch.Tensor,
        hpwl: torch.Tensor,
        density: torch.Tensor,
        congestion: torch.Tensor,
        boundary: torch.Tensor,
        weights: _StageWeights,
        base_lr: float,
        param: torch.Tensor,
        initial: torch.Tensor,
        state: PlacementState,
        grad_norm: float = 0.0,
        step_norm: float = 0.0,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        pos = param.detach()
        disp = torch.linalg.norm(pos - initial, dim=1)
        hard = disp[state.hard_mask]
        soft = disp[state.soft_mask]
        log = {
            "step": float(step),
            "stage": float(stage_id),
            "loss": float(loss.detach().cpu().item()) if torch.isfinite(loss.detach()).all() else float("nan"),
            "hpwl": float(hpwl.detach().cpu().item()),
            "density": float(density.detach().cpu().item()),
            "congestion": float(congestion.detach().cpu().item()),
            "boundary": float(boundary.detach().cpu().item()),
            "density_weight": float(weights.density),
            "congestion_weight": float(weights.congestion),
            "lr": float(base_lr * weights.lr_scale),
            "grad_norm": float(grad_norm),
            "step_norm": float(step_norm),
            "total_mean_disp": float(disp.mean().detach().cpu().item()) if disp.numel() else 0.0,
            "total_max_disp": float(disp.max().detach().cpu().item()) if disp.numel() else 0.0,
            "hard_mean_disp": float(hard.mean().detach().cpu().item()) if hard.numel() else 0.0,
            "hard_max_disp": float(hard.max().detach().cpu().item()) if hard.numel() else 0.0,
            "soft_mean_disp": float(soft.mean().detach().cpu().item()) if soft.numel() else 0.0,
            "soft_max_disp": float(soft.max().detach().cpu().item()) if soft.numel() else 0.0,
        }
        if extra:
            log.update({key: float(value) for key, value in extra.items() if isinstance(value, (int, float))})
        self.logs.append(log)

    def _base_lr(self, state: PlacementState) -> float:
        chip_area = max(float(state.canvas_width) * float(state.canvas_height), 1.0e-12)
        n_hard = max(int(state.hard_mask.sum().detach().cpu().item()), 1)
        return 5.0 * chip_area / float(n_hard)

    def _density_grid_size(self, state: PlacementState) -> Tuple[int, int]:
        name = str(state.metadata.get("benchmark_name", "")).lower()
        side = 64 if name.startswith("ibm") else 32
        return side, side

    def _congestion_grid_size(self, state: PlacementState) -> Tuple[int, int]:
        rows = int(state.metadata.get("grid_rows", 0) or 0)
        cols = int(state.metadata.get("grid_cols", 0) or 0)
        if rows <= 0 or cols <= 0:
            return 32, 32
        return rows, cols

    def _clip_to_bounds(self, state: PlacementState, positions: torch.Tensor, epsilon: float = 1.0e-4) -> torch.Tensor:
        half = state.sizes * 0.5
        low = half + float(epsilon)
        high = state.canvas_size.view(1, 2) - half - float(epsilon)
        feasible = high >= low
        midpoint = state.canvas_size.view(1, 2) * 0.5
        clipped = torch.minimum(torch.maximum(positions, torch.minimum(low, high)), torch.maximum(low, high))
        return torch.where(feasible, clipped, midpoint.expand_as(clipped))

    def _masked_norm(self, tensor: torch.Tensor, mask: torch.Tensor) -> float:
        if tensor is None or tensor.numel() == 0 or not bool(mask.any()):
            return 0.0
        return float(torch.linalg.norm(tensor.detach()[mask]).detach().cpu().item())

    def _clear_momentum(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param)
                if state and "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()

    def _resolve_device(self, device: DeviceLike) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
