"""Public PinePlace v2 analytical placement engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from pine_place.v2.congestion import route_demand_overflow
from pine_place.v2.density import density_overflow
from pine_place.v2.legalization import legalize_placement
from pine_place.v2.objectives import boundary_penalty, hard_macro_overlap_penalty, smooth_hpwl
from pine_place.v2.optimizer import ObjectiveWeights, OptimizerConfig, StageConfig, default_stage_schedule, optimize_placement
from pine_place.v2.placement_state import PlacementState


ConfigLike = Optional[Union[str, OptimizerConfig, Dict[str, Any]]]


class AnalyticalPlacementEngine:
    """Macro-only DREAMPlace/RePlAce-inspired v2 candidate generator."""

    def run(self, benchmark: Any, config: ConfigLike = None) -> Dict[str, Any]:
        opt_config = make_v2_config(config)
        state = PlacementState.from_benchmark(benchmark, device=opt_config.device).limited_nets(opt_config.max_nets)
        initial = apply_initialization_transform(state, opt_config)
        optimized, logs, opt_stats = optimize_placement(state, opt_config, initial_positions=initial)
        clipped = _clip_to_bounds(state, optimized)
        clipped[state.fixed_mask] = state.positions[state.fixed_mask]
        legalized, legalization_stats = legalize_placement(state, clipped)
        final_stats = evaluate_objectives(state, legalized, opt_config)
        return {
            "placement": legalized.detach().cpu(),
            "logs": logs,
            "objective_stats": final_stats,
            "optimizer_stats": opt_stats,
            "legalization_stats": legalization_stats,
            "valid_summary": placement_valid_summary(state, legalized, legalization_stats),
            "config": opt_config,
            "metadata": dict(state.metadata),
        }


def run_v2_multistart_candidates(benchmark: Any, device: Optional[Union[str, torch.device]] = None) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    engine = AnalyticalPlacementEngine()
    for preset in (
        "legal_balanced",
        "legal_congestion",
        "legal_density",
        "x_spread",
        "y_spread",
        "center_evac",
    ):
        config = make_v2_config(preset, device=device)
        result = engine.run(benchmark, config)
        result["label"] = preset
        result["preset"] = preset
        candidates.append(result)
    return candidates


def make_v2_config(config: ConfigLike = None, device: Optional[Union[str, torch.device]] = None) -> OptimizerConfig:
    if isinstance(config, OptimizerConfig):
        return config
    if isinstance(config, dict):
        preset = str(config.get("preset", config.get("name", "balanced")))
        base = make_v2_config(preset, device=device)
        allowed = {field.name for field in OptimizerConfig.__dataclass_fields__.values()}
        overrides = {key: value for key, value in config.items() if key in allowed}
        return replace(base, **overrides)
    preset = "balanced" if config is None else str(config)
    base = OptimizerConfig(init_mode=_init_mode_for_preset(preset), device=device)

    if preset == "density_heavy":
        return replace(
            base,
            density_target=0.80,
            stage_schedule=_scale_schedule(base.stages(), density=1.65, overlap=1.20, wl=0.90),
        )
    if preset == "congestion_heavy":
        return replace(
            base,
            congestion_target=0.92,
            stage_schedule=_scale_schedule(base.stages(), congestion=1.90, density=1.10, wl=0.90),
        )
    if preset == "legal_balanced":
        return replace(
            base,
            learning_rate=0.018,
            density_target=0.84,
            congestion_target=0.96,
            init_scale=1.0,
            stage_schedule=_legal_schedule(base.iterations, mode="balanced"),
        )
    if preset == "legal_congestion":
        return replace(
            base,
            learning_rate=0.022,
            density_target=0.86,
            congestion_target=0.88,
            init_mode="center_evac",
            init_scale=1.10,
            stage_schedule=_legal_schedule(base.iterations, mode="congestion"),
        )
    if preset == "legal_density":
        return replace(
            base,
            learning_rate=0.020,
            density_target=0.78,
            congestion_target=0.96,
            init_mode="balanced",
            init_scale=1.02,
            stage_schedule=_legal_schedule(base.iterations, mode="density"),
        )
    if preset == "legal_moderate_aggressive":
        return replace(
            base,
            learning_rate=0.034,
            density_target=0.78,
            congestion_target=0.90,
            init_mode="center_evac",
            init_scale=1.25,
            stage_schedule=_legal_schedule(base.iterations, mode="moderate_aggressive"),
        )
    if preset == "aggressive_density":
        stages = default_stage_schedule(base.iterations)
        aggressive = (
            replace(
                stages[0],
                weights=ObjectiveWeights(
                    wl_weight=0.22,
                    density_weight=1.35,
                    congestion_weight=0.70,
                    overlap_weight=0.35,
                    boundary_weight=1.00,
                ),
                lr_scale=1.55,
                gamma_scale=1.15,
            ),
            replace(
                stages[1],
                weights=ObjectiveWeights(
                    wl_weight=0.35,
                    density_weight=1.80,
                    congestion_weight=1.00,
                    overlap_weight=0.80,
                    boundary_weight=1.10,
                ),
                lr_scale=1.10,
                gamma_scale=0.95,
            ),
            replace(
                stages[2],
                weights=ObjectiveWeights(
                    wl_weight=0.45,
                    density_weight=1.45,
                    congestion_weight=1.20,
                    overlap_weight=1.20,
                    boundary_weight=1.25,
                ),
                lr_scale=0.75,
                gamma_scale=0.75,
            ),
        )
        return replace(
            base,
            learning_rate=0.065,
            density_target=0.72,
            congestion_target=0.82,
            init_mode="center_evac",
            init_scale=1.65,
            stage_schedule=aggressive,
        )
    if preset == "jitter":
        return replace(base, init_mode="jitter", seed=77, jitter_scale=0.018)
    if preset in {"x_spread", "y_spread", "center_evac", "balanced"}:
        return base
    return replace(base, init_mode="balanced")


def apply_initialization_transform(state: PlacementState, config: OptimizerConfig) -> torch.Tensor:
    pos = state.positions.clone()
    mode = str(config.init_mode)
    intensity = max(float(config.init_scale), 0.0)
    center = state.canvas_size.view(1, 2) * 0.5
    movable = state.movable_mask

    if mode == "x_spread":
        pos[movable, 0] = center[0, 0] + (pos[movable, 0] - center[0, 0]) * (1.0 + 0.14 * intensity)
        pos[movable, 1] = center[0, 1] + (pos[movable, 1] - center[0, 1]) * (1.0 - 0.02 * intensity)
    elif mode == "y_spread":
        pos[movable, 0] = center[0, 0] + (pos[movable, 0] - center[0, 0]) * (1.0 - 0.02 * intensity)
        pos[movable, 1] = center[0, 1] + (pos[movable, 1] - center[0, 1]) * (1.0 + 0.14 * intensity)
    elif mode == "center_evac":
        vec = pos - center
        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        if bool((norm < 1.0e-6).any()):
            angles = torch.arange(state.num_macros, dtype=state.dtype, device=state.device).view(-1, 1) * 2.39996323
            fallback = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
            vec = torch.where(norm < 1.0e-6, fallback, vec)
            norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        strength = torch.relu(1.0 - norm / max(state.span * 0.45, 1.0e-6)) * (state.span * 0.040 * intensity)
        pos[movable] = pos[movable] + vec[movable] / torch.clamp(norm[movable], min=1.0e-6) * strength[movable]
    elif mode == "jitter":
        generator = torch.Generator(device=state.device)
        generator.manual_seed(int(config.seed))
        noise = torch.randn(pos.shape, generator=generator, dtype=state.dtype, device=state.device)
        pos[movable] = pos[movable] + noise[movable] * (float(config.jitter_scale) * intensity * state.span)
    elif abs(intensity - 1.0) > 1.0e-9:
        pos[movable] = center + (pos[movable] - center) * intensity

    pos = _clip_to_bounds(state, pos)
    pos[state.fixed_mask] = state.positions[state.fixed_mask]
    return pos


def evaluate_objectives(
    state: PlacementState,
    positions: torch.Tensor,
    config: OptimizerConfig,
) -> Dict[str, float]:
    gamma = float(config.gamma) if config.gamma is not None else max(state.span * 0.015, 1.0e-3)
    with torch.no_grad():
        wl = smooth_hpwl(state, positions, gamma=gamma)
        density, density_stats = density_overflow(state, positions, config.grid_size, config.density_target)
        congestion, congestion_stats = route_demand_overflow(
            state,
            positions,
            config.grid_size,
            config.congestion_target,
            gamma=gamma,
        )
        overlap = hard_macro_overlap_penalty(state, positions, config.overlap_chunk_size)
        boundary = boundary_penalty(state, positions)
    stats = {
        "smooth_hpwl": float(wl.detach().cpu().item()),
        "density": float(density.detach().cpu().item()),
        "congestion": float(congestion.detach().cpu().item()),
        "hard_overlap": float(overlap.detach().cpu().item()),
        "boundary": float(boundary.detach().cpu().item()),
    }
    stats.update(density_stats)
    stats.update(congestion_stats)
    return stats


def placement_valid_summary(
    state: PlacementState,
    positions: torch.Tensor,
    legalization_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    placement = positions.detach().to(device=state.device, dtype=state.dtype)
    shape_ok = placement.shape == state.positions.shape
    half = state.sizes * 0.5
    low_slack = placement - half
    high_slack = state.canvas_size.view(1, 2) - (placement + half)
    slack = torch.minimum(low_slack, high_slack)
    violation = torch.relu(-slack)
    fixed_ok = True
    if bool(state.fixed_mask.any()):
        fixed_ok = bool(torch.allclose(placement[state.fixed_mask], state.positions[state.fixed_mask], atol=1.0e-4))
    final_overlap = 0.0
    if legalization_stats is not None:
        final_overlap = float(legalization_stats.get("final_overlap_count", legalization_stats.get("legalization_remaining_overlaps", 0.0)))
    return {
        "valid_shape": float(bool(shape_ok)),
        "valid_bounds": float(bool((violation <= 1.0e-5).all())),
        "valid_fixed": float(fixed_ok),
        "hard_overlap_count": float(final_overlap),
        "bounds_max_violation": float(violation.max().detach().cpu().item()) if violation.numel() else 0.0,
    }


def _scale_schedule(
    stages: Iterable[StageConfig],
    wl: float = 1.0,
    density: float = 1.0,
    congestion: float = 1.0,
    overlap: float = 1.0,
    boundary: float = 1.0,
) -> tuple[StageConfig, ...]:
    scaled = []
    for stage in stages:
        weights = ObjectiveWeights(
            wl_weight=stage.weights.wl_weight * float(wl),
            density_weight=stage.weights.density_weight * float(density),
            congestion_weight=stage.weights.congestion_weight * float(congestion),
            overlap_weight=stage.weights.overlap_weight * float(overlap),
            boundary_weight=stage.weights.boundary_weight * float(boundary),
        )
        scaled.append(replace(stage, weights=weights))
    return tuple(scaled)


def _legal_schedule(iterations: int, mode: str) -> tuple[StageConfig, ...]:
    stages = default_stage_schedule(iterations)
    if mode == "congestion":
        weights = (
            ObjectiveWeights(0.70, 0.18, 0.35, 0.75, 1.20),
            ObjectiveWeights(0.62, 0.42, 0.70, 1.30, 1.35),
            ObjectiveWeights(0.58, 0.48, 0.82, 1.75, 1.50),
        )
        lr_scales = (0.85, 0.65, 0.42)
    elif mode == "density":
        weights = (
            ObjectiveWeights(0.78, 0.42, 0.18, 0.85, 1.20),
            ObjectiveWeights(0.66, 0.95, 0.28, 1.35, 1.35),
            ObjectiveWeights(0.62, 0.90, 0.36, 1.80, 1.50),
        )
        lr_scales = (0.85, 0.62, 0.40)
    elif mode == "moderate_aggressive":
        weights = (
            ObjectiveWeights(0.52, 0.55, 0.42, 0.95, 1.25),
            ObjectiveWeights(0.50, 0.92, 0.72, 1.55, 1.45),
            ObjectiveWeights(0.56, 0.78, 0.82, 2.10, 1.65),
        )
        lr_scales = (1.00, 0.72, 0.45)
    else:
        weights = (
            ObjectiveWeights(0.88, 0.24, 0.18, 0.80, 1.20),
            ObjectiveWeights(0.72, 0.58, 0.36, 1.30, 1.35),
            ObjectiveWeights(0.66, 0.56, 0.45, 1.80, 1.50),
        )
        lr_scales = (0.82, 0.60, 0.38)

    return tuple(
        replace(stage, weights=weights[idx], lr_scale=lr_scales[idx], gamma_scale=max(stage.gamma_scale, 0.85))
        for idx, stage in enumerate(stages)
    )


def _init_mode_for_preset(preset: str) -> str:
    if preset in {"x_spread", "y_spread", "center_evac", "jitter"}:
        return preset
    return "balanced"


def _clip_to_bounds(state: PlacementState, positions: torch.Tensor, epsilon: float = 1.0e-4) -> torch.Tensor:
    half = state.sizes * 0.5
    low = half + float(epsilon)
    high = state.canvas_size.view(1, 2) - half - float(epsilon)
    feasible = high >= low
    midpoint = state.canvas_size.view(1, 2) * 0.5
    clipped = torch.minimum(torch.maximum(positions, torch.minimum(low, high)), torch.maximum(low, high))
    return torch.where(feasible, clipped, midpoint.expand_as(clipped))
