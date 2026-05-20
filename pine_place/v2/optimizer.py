"""Nesterov optimization loop for PinePlace v2."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import torch

from pine_place.v2.congestion import route_demand_overflow
from pine_place.v2.density import GridSize, density_overflow
from pine_place.v2.objectives import boundary_penalty, hard_macro_overlap_penalty, smooth_hpwl
from pine_place.v2.placement_state import PlacementState


@dataclass(frozen=True)
class ObjectiveWeights:
    wl_weight: float = 1.0
    density_weight: float = 0.35
    congestion_weight: float = 0.20
    overlap_weight: float = 1.0
    boundary_weight: float = 1.0


@dataclass(frozen=True)
class StageConfig:
    name: str
    iterations: int
    weights: ObjectiveWeights
    lr_scale: float = 1.0
    gamma_scale: float = 1.0


def default_stage_schedule(iterations: int = 240) -> Tuple[StageConfig, ...]:
    early = max(1, int(iterations * 0.35))
    middle = max(1, int(iterations * 0.40))
    late = max(1, int(iterations) - early - middle)
    return (
        StageConfig(
            name="early",
            iterations=early,
            weights=ObjectiveWeights(
                wl_weight=1.20,
                density_weight=0.08,
                congestion_weight=0.03,
                overlap_weight=0.20,
                boundary_weight=1.00,
            ),
            lr_scale=1.00,
            gamma_scale=1.35,
        ),
        StageConfig(
            name="middle",
            iterations=middle,
            weights=ObjectiveWeights(
                wl_weight=0.85,
                density_weight=0.55,
                congestion_weight=0.12,
                overlap_weight=0.85,
                boundary_weight=1.00,
            ),
            lr_scale=0.70,
            gamma_scale=1.00,
        ),
        StageConfig(
            name="late",
            iterations=late,
            weights=ObjectiveWeights(
                wl_weight=0.70,
                density_weight=0.60,
                congestion_weight=0.35,
                overlap_weight=1.25,
                boundary_weight=1.20,
            ),
            lr_scale=0.40,
            gamma_scale=0.75,
        ),
    )


@dataclass
class OptimizerConfig:
    iterations: int = 240
    learning_rate: float = 0.025
    momentum: float = 0.9
    grid_size: GridSize = (32, 32)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    gamma: Optional[float] = None
    density_target: float = 0.85
    congestion_target: float = 1.0
    stage_schedule: Optional[Tuple[StageConfig, ...]] = None
    logging_interval: int = 20
    overlap_chunk_size: int = 512
    init_mode: str = "balanced"
    jitter_scale: float = 0.012
    init_scale: float = 1.0
    seed: int = 42
    device: Optional[Union[str, torch.device]] = None
    max_nets: Optional[int] = None

    def stages(self) -> Tuple[StageConfig, ...]:
        if self.stage_schedule is not None:
            return self.stage_schedule
        return default_stage_schedule(self.iterations)


def optimize_placement(
    state: PlacementState,
    config: OptimizerConfig,
    initial_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[Dict[str, float]], Dict[str, float]]:
    """Optimize movable macro centers and return final positions plus logs."""

    init = state.positions if initial_positions is None else initial_positions.to(device=state.device, dtype=state.dtype)
    param = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.SGD(
        [param],
        lr=float(config.learning_rate),
        momentum=float(config.momentum),
        nesterov=True,
    )

    logs: List[Dict[str, float]] = []
    last_stats: Dict[str, float] = {}
    global_step = 0

    with torch.no_grad():
        param.data = _clip_to_bounds(state, param.data)
        param.data[state.fixed_mask] = state.positions[state.fixed_mask]

    initial_clipped = param.detach().clone()

    for stage_id, stage in enumerate(config.stages()):
        if stage.iterations <= 0:
            continue
        stage_start = param.detach().clone()
        for group in optimizer.param_groups:
            group["lr"] = float(config.learning_rate) * float(stage.lr_scale)
        gamma = (float(config.gamma) if config.gamma is not None else max(state.span * 0.015, 1.0e-3)) * float(stage.gamma_scale)
        weights = stage.weights or config.objective_weights

        for local_step in range(int(stage.iterations)):
            optimizer.zero_grad(set_to_none=True)
            pos = torch.where(state.movable_mask[:, None], param, state.positions)

            wl = smooth_hpwl(state, pos, gamma=gamma)
            density, density_stats = density_overflow(
                state,
                pos,
                grid_size=config.grid_size,
                target_util=config.density_target,
            )
            congestion, congestion_stats = route_demand_overflow(
                state,
                pos,
                grid_size=config.grid_size,
                target=config.congestion_target,
                gamma=gamma,
            )
            overlap = hard_macro_overlap_penalty(state, pos, chunk_size=config.overlap_chunk_size)
            boundary = boundary_penalty(state, pos)
            loss = (
                float(weights.wl_weight) * wl
                + float(weights.density_weight) * density
                + float(weights.congestion_weight) * congestion
                + float(weights.overlap_weight) * overlap
                + float(weights.boundary_weight) * boundary
            )

            if not torch.isfinite(loss):
                logs.append(
                    {
                        "step": float(global_step),
                        "stage": float(stage_id),
                        "loss": float("nan"),
                        "nonfinite": 1.0,
                    }
                )
                break

            loss.backward()
            grad_norm = _masked_norm(param.grad, state.movable_mask) if param.grad is not None else 0.0
            before_step = param.detach().clone()
            optimizer.step()
            with torch.no_grad():
                param.data = _clip_to_bounds(state, param.data)
                param.data[state.fixed_mask] = state.positions[state.fixed_mask]
            after_step = param.detach()
            step_norm = _masked_norm(after_step - before_step, state.movable_mask)

            should_log = (
                global_step == 0
                or local_step == int(stage.iterations) - 1
                or (config.logging_interval > 0 and global_step % int(config.logging_interval) == 0)
            )
            if should_log:
                movement = movement_diagnostics(state, after_step, initial_clipped, prefix="")
                stage_movement = movement_diagnostics(state, after_step, stage_start, prefix="stage_")
                last_stats = {
                    "step": float(global_step),
                    "stage": float(stage_id),
                    "loss": float(loss.detach().cpu().item()),
                    "wl": float(wl.detach().cpu().item()),
                    "density": float(density.detach().cpu().item()),
                    "congestion": float(congestion.detach().cpu().item()),
                    "overlap": float(overlap.detach().cpu().item()),
                    "boundary": float(boundary.detach().cpu().item()),
                    "lr": float(config.learning_rate) * float(stage.lr_scale),
                    "gamma": float(gamma),
                    "grad_norm": float(grad_norm),
                    "step_norm": float(step_norm),
                    **movement,
                    **stage_movement,
                    **density_stats,
                    **congestion_stats,
                }
                logs.append(last_stats)
            global_step += 1

    final_pos = torch.where(state.movable_mask[:, None], param.detach(), state.positions)
    final_pos = _clip_to_bounds(state, final_pos)
    final_pos[state.fixed_mask] = state.positions[state.fixed_mask]
    if not last_stats:
        last_stats = {"loss": 0.0, "step": 0.0}
    last_stats.update(movement_diagnostics(state, final_pos, initial_clipped, prefix="final_"))
    return final_pos.detach(), logs, last_stats


def config_with_weights(config: OptimizerConfig, weights: ObjectiveWeights) -> OptimizerConfig:
    stages = tuple(replace(stage, weights=weights) for stage in config.stages())
    return replace(config, objective_weights=weights, stage_schedule=stages)


def _clip_to_bounds(state: PlacementState, positions: torch.Tensor, epsilon: float = 1.0e-4) -> torch.Tensor:
    half = state.sizes * 0.5
    low = half + float(epsilon)
    high = state.canvas_size.view(1, 2) - half - float(epsilon)
    feasible = high >= low
    midpoint = state.canvas_size.view(1, 2) * 0.5
    clipped = torch.minimum(torch.maximum(positions, torch.minimum(low, high)), torch.maximum(low, high))
    return torch.where(feasible, clipped, midpoint.expand_as(clipped))


def movement_diagnostics(
    state: PlacementState,
    positions: torch.Tensor,
    reference: torch.Tensor,
    prefix: str = "",
) -> Dict[str, float]:
    disp = torch.linalg.norm((positions - reference).detach(), dim=1)
    hard = disp[state.hard_mask]
    soft = disp[state.soft_mask]
    return {
        f"{prefix}hard_mean_disp": _mean_or_zero(hard),
        f"{prefix}hard_max_disp": _max_or_zero(hard),
        f"{prefix}soft_mean_disp": _mean_or_zero(soft),
        f"{prefix}soft_max_disp": _max_or_zero(soft),
        f"{prefix}total_mean_disp": _mean_or_zero(disp),
        f"{prefix}total_max_disp": _max_or_zero(disp),
    }


def _masked_norm(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if tensor.numel() == 0 or not bool(mask.any()):
        return 0.0
    selected = tensor.detach()[mask]
    return float(torch.linalg.norm(selected).detach().cpu().item())


def _mean_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.mean().detach().cpu().item()) if tensor.numel() else 0.0


def _max_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.max().detach().cpu().item()) if tensor.numel() else 0.0
