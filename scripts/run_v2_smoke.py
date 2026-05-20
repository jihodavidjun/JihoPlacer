#!/usr/bin/env python3
"""Tiny smoke runner for PinePlace v2 analytical placement."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = REPO_ROOT / "third_party" / "macro-place-challenge-2026"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CHALLENGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CHALLENGE_ROOT))

from pine_place.v2.analytical_engine import AnalyticalPlacementEngine, make_v2_config
from pine_place.v2.optimizer import ObjectiveWeights, StageConfig
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir


NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def main() -> int:
    args = parse_args()
    requested_device = normalize_device(args.device)
    active_device = resolve_active_device(requested_device)

    print("PinePlace v2 smoke")
    print(f"repo_root={REPO_ROOT}")
    print(f"challenge_root={CHALLENGE_ROOT}")
    print(f"benchmark={args.benchmark} preset={args.preset} iters={args.iters}")
    print(f"device_requested={args.device} device_active={active_device} cuda_available={torch.cuda.is_available()}")

    benchmark, plc, load_message = load_smoke_benchmark(args.benchmark)
    print(load_message)
    print(
        "benchmark_info "
        f"name={getattr(benchmark, 'name', args.benchmark)} "
        f"macros={getattr(benchmark, 'num_macros', '?')} "
        f"hard={getattr(benchmark, 'num_hard_macros', '?')} "
        f"soft={getattr(benchmark, 'num_soft_macros', '?')} "
        f"nets={getattr(benchmark, 'num_nets', '?')} "
        f"canvas={float(getattr(benchmark, 'canvas_width', 0.0)):.3f}x"
        f"{float(getattr(benchmark, 'canvas_height', 0.0)):.3f}"
    )

    config = build_config(args, active_device)
    print(
        "smoke_config "
        f"grid_size={config.grid_size} max_nets={config.max_nets} "
        f"lr={config.learning_rate} target_util={config.density_target} "
        f"gamma={config.gamma} init_scale={config.init_scale} log_every={config.logging_interval}"
    )

    start = time.time()
    result = AnalyticalPlacementEngine().run(benchmark, config)
    elapsed = time.time() - start
    placement = result["placement"]

    print()
    print("optimization_logs")
    print_stage_logs(result.get("logs", []), limit=12)

    print()
    print("validation")
    print(f"runtime_sec={elapsed:.3f}")
    print(f"placement_shape={tuple(placement.shape)} expected={(int(benchmark.num_macros), 2)}")
    print(f"placement_dtype={placement.dtype} placement_device={placement.device}")
    print(f"has_nan={bool(torch.isnan(placement).any().item())} has_inf={bool(torch.isinf(placement).any().item())}")
    print_stats("bounds", bounds_stats(placement, benchmark))
    print_stats("displacement", displacement_stats(placement, benchmark))
    print_stats("overlap", overlap_stats(placement, benchmark))
    legalization_stats = result.get("legalization_stats", {})
    print_stats("legalization", legalization_stats)
    print(
        "legalization_summary "
        f"initial_overlap_count={float(legalization_stats.get('initial_overlap_count', 0.0)):.6g} "
        f"final_overlap_count={float(legalization_stats.get('final_overlap_count', 0.0)):.6g} "
        f"moved_macro_count={float(legalization_stats.get('moved_macro_count', 0.0)):.6g} "
        f"max_overlap_area={float(legalization_stats.get('max_overlap_area', 0.0)):.6g}"
    )
    objective_stats = result.get("objective_stats", {})
    print_stats("objectives", objective_stats)
    print(
        "route_demand_summary "
        f"demand_max={float(objective_stats.get('demand_max', 0.0)):.6g} "
        f"capacity_min={float(objective_stats.get('capacity_min', 0.0)):.6g} "
        f"overflow_max={float(objective_stats.get('overflow_max', objective_stats.get('congestion_max_overflow', 0.0))):.6g} "
        f"overflow_mean={float(objective_stats.get('overflow_mean', objective_stats.get('congestion_mean_overflow', 0.0))):.6g} "
        f"hot_bin_count={float(objective_stats.get('hot_bin_count', objective_stats.get('congestion_hot_bins', 0.0))):.6g} "
        f"blockage_max={float(objective_stats.get('blockage_max', 0.0)):.6g}"
    )
    print(
        "hotspot_summary "
        f"max_density_util={float(objective_stats.get('density_max_util', 0.0)):.6g} "
        f"max_congestion_overflow={float(objective_stats.get('congestion_max_overflow', 0.0)):.6g}"
    )

    if args.enable_exact_check:
        print()
        print("exact_proxy_check")
        if plc is None:
            print("skipped: PlacementCost was not available for this benchmark load path")
        else:
            run_exact_check(placement, benchmark, plc)

    if args.save_placement:
        save_path = Path(args.save_placement)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(placement, save_path)
        print(f"saved_placement={save_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny PinePlace v2 smoke test.")
    parser.add_argument("--benchmark", default="ibm01", help="Benchmark name or benchmark directory path.")
    parser.add_argument("--preset", default="balanced", help="v2 preset name.")
    parser.add_argument("--iters", type=int, default=20, help="Override total v2 iterations.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lr", type=float, default=None, help="Override base optimizer learning rate.")
    parser.add_argument("--grid-size", default="8", help="Density/congestion grid as N or ROWSxCOLS.")
    parser.add_argument("--max-nets", type=int, default=64, help="Cap v2 nets for smoke speed; use 0 for no cap.")
    parser.add_argument("--wl-weight", type=float, default=None, help="Override wirelength weight in all stages.")
    parser.add_argument("--density-weight", type=float, default=None, help="Override density weight in all stages.")
    parser.add_argument("--congestion-weight", type=float, default=None, help="Override congestion weight in all stages.")
    parser.add_argument("--overlap-weight", type=float, default=None, help="Override hard-overlap weight in all stages.")
    parser.add_argument("--boundary-weight", type=float, default=None, help="Override boundary weight in all stages.")
    parser.add_argument("--target-util", type=float, default=None, help="Override density target utilization.")
    parser.add_argument("--gamma", type=float, default=None, help="Override smooth HPWL/congestion gamma.")
    parser.add_argument("--init-scale", type=float, default=None, help="Scale initialization transform intensity.")
    parser.add_argument("--log-every", type=int, default=None, help="Optimizer log interval.")
    parser.add_argument("--save-placement", default=None, help="Optional .pt path for final placement tensor.")
    parser.add_argument("--enable-exact-check", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def build_config(args: argparse.Namespace, active_device: str):
    config = make_v2_config(args.preset, device=active_device)
    stages = resize_stage_iterations(config.stages(), int(args.iters))
    stages = apply_weight_overrides(stages, args)
    max_nets = None if int(args.max_nets) <= 0 else int(args.max_nets)
    overrides = {
        "iterations": int(args.iters),
        "stage_schedule": stages,
        "logging_interval": max(1, int(args.log_every)) if args.log_every is not None else max(1, int(args.iters) // 5),
        "grid_size": parse_grid_size(args.grid_size),
        "max_nets": max_nets,
    }
    if args.lr is not None:
        overrides["learning_rate"] = float(args.lr)
    if args.target_util is not None:
        overrides["density_target"] = float(args.target_util)
    if args.gamma is not None:
        overrides["gamma"] = float(args.gamma)
    if args.init_scale is not None:
        overrides["init_scale"] = float(args.init_scale)
    return replace(config, **overrides)


def parse_grid_size(value: str):
    text = str(value).lower().replace(",", "x")
    if "x" in text:
        rows, cols = text.split("x", 1)
        return int(rows), int(cols)
    size = int(text)
    return size, size


def resize_stage_iterations(stages: Tuple[StageConfig, ...], iterations: int) -> Tuple[StageConfig, ...]:
    if not stages:
        return tuple()
    total_old = sum(max(0, int(stage.iterations)) for stage in stages)
    if total_old <= 0:
        return stages
    remaining = max(0, int(iterations))
    resized = []
    for idx, stage in enumerate(stages):
        if idx == len(stages) - 1:
            count = remaining
        else:
            count = max(1, int(round(iterations * max(0, int(stage.iterations)) / total_old)))
            count = min(count, max(0, remaining - (len(stages) - idx - 1)))
        remaining -= count
        resized.append(replace(stage, iterations=count))
    return tuple(resized)


def apply_weight_overrides(stages: Tuple[StageConfig, ...], args: argparse.Namespace) -> Tuple[StageConfig, ...]:
    if not any(
        value is not None
        for value in (
            args.wl_weight,
            args.density_weight,
            args.congestion_weight,
            args.overlap_weight,
            args.boundary_weight,
        )
    ):
        return stages
    updated = []
    for stage in stages:
        weights = ObjectiveWeights(
            wl_weight=stage.weights.wl_weight if args.wl_weight is None else float(args.wl_weight),
            density_weight=stage.weights.density_weight if args.density_weight is None else float(args.density_weight),
            congestion_weight=stage.weights.congestion_weight
            if args.congestion_weight is None
            else float(args.congestion_weight),
            overlap_weight=stage.weights.overlap_weight if args.overlap_weight is None else float(args.overlap_weight),
            boundary_weight=stage.weights.boundary_weight if args.boundary_weight is None else float(args.boundary_weight),
        )
        updated.append(replace(stage, weights=weights))
    return tuple(updated)


def normalize_device(value: str) -> Optional[str]:
    if value == "auto":
        return None
    return value


def resolve_active_device(requested: Optional[str]) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # This smoke runner favors quick startup over GPU throughput. Force CUDA
    # explicitly with --device cuda when profiling the GPU path.
    return "cpu"


def load_smoke_benchmark(name_or_path: str) -> Tuple[Benchmark, Optional[Any], str]:
    candidate = Path(name_or_path)
    if candidate.exists():
        bench_dir = candidate if candidate.is_dir() else candidate.parent
        benchmark, plc = load_benchmark_from_dir(str(bench_dir))
        return benchmark, plc, f"load_path=directory:{bench_dir}"

    ibm_dir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name_or_path
    if ibm_dir.exists():
        benchmark, plc = load_benchmark_from_dir(str(ibm_dir))
        return benchmark, plc, f"load_path=iccad04:{ibm_dir}"

    if name_or_path in NG45_BENCHMARKS:
        ng45_dir = REPO_ROOT / NG45_BENCHMARKS[name_or_path]
        if ng45_dir.exists():
            benchmark, plc = load_benchmark(
                str(ng45_dir / "netlist.pb.txt"),
                str(ng45_dir / "initial.plc"),
                name=name_or_path,
            )
            return benchmark, plc, f"load_path=ng45:{ng45_dir}"

    processed = CHALLENGE_ROOT / "benchmarks" / "processed" / "public" / f"{name_or_path}.pt"
    if processed.exists():
        benchmark = Benchmark.load(str(processed))
        return benchmark, None, f"load_path=processed_pt:{processed} exact_check_available=0"

    raise FileNotFoundError(
        f"Could not resolve benchmark '{name_or_path}'. Checked direct path, ICCAD04, NG45 map, and processed/public .pt."
    )


def print_stage_logs(logs: Any, limit: int) -> None:
    if not logs:
        print("no optimizer logs emitted")
        return
    if len(logs) > limit:
        keep = logs[: limit // 2] + logs[-(limit - limit // 2) :]
        print(f"showing {len(keep)} of {len(logs)} log rows")
    else:
        keep = logs
    for row in keep:
        print(
            "log "
            f"step={int(row.get('step', 0))} "
            f"stage={int(row.get('stage', 0))} "
            f"loss={float(row.get('loss', 0.0)):.6g} "
            f"wl={float(row.get('wl', 0.0)):.6g} "
            f"density={float(row.get('density', 0.0)):.6g} "
            f"congestion={float(row.get('congestion', 0.0)):.6g} "
            f"overlap={float(row.get('overlap', 0.0)):.6g} "
            f"boundary={float(row.get('boundary', 0.0)):.6g}"
            f" grad_norm={float(row.get('grad_norm', 0.0)):.6g}"
            f" step_norm={float(row.get('step_norm', 0.0)):.6g}"
            f" hard_mean_disp={float(row.get('hard_mean_disp', 0.0)):.6g}"
            f" hard_max_disp={float(row.get('hard_max_disp', 0.0)):.6g}"
            f" soft_mean_disp={float(row.get('soft_mean_disp', 0.0)):.6g}"
            f" soft_max_disp={float(row.get('soft_max_disp', 0.0)):.6g}"
            f" total_mean_disp={float(row.get('total_mean_disp', 0.0)):.6g}"
            f" total_max_disp={float(row.get('total_max_disp', 0.0)):.6g}"
            f" stage_hard_mean_disp={float(row.get('stage_hard_mean_disp', 0.0)):.6g}"
            f" stage_hard_max_disp={float(row.get('stage_hard_max_disp', 0.0)):.6g}"
            f" stage_soft_mean_disp={float(row.get('stage_soft_mean_disp', 0.0)):.6g}"
            f" stage_soft_max_disp={float(row.get('stage_soft_max_disp', 0.0)):.6g}"
            f" stage_total_mean_disp={float(row.get('stage_total_mean_disp', 0.0)):.6g}"
            f" stage_total_max_disp={float(row.get('stage_total_max_disp', 0.0)):.6g}"
        )


def bounds_stats(placement: torch.Tensor, benchmark: Benchmark) -> Dict[str, float]:
    pos = placement.detach().cpu()
    sizes = benchmark.macro_sizes.detach().cpu().to(dtype=pos.dtype)
    half = sizes * 0.5
    low_slack = pos - half
    high_slack = torch.tensor([benchmark.canvas_width, benchmark.canvas_height], dtype=pos.dtype) - (pos + half)
    slack = torch.minimum(low_slack, high_slack)
    out = torch.relu(-slack)
    return {
        "min_slack": float(slack.min().item()) if slack.numel() else 0.0,
        "max_violation": float(out.max().item()) if out.numel() else 0.0,
        "violating_macros": float((out.max(dim=1).values > 0).sum().item()) if out.numel() else 0.0,
    }


def displacement_stats(placement: torch.Tensor, benchmark: Benchmark) -> Dict[str, float]:
    init = benchmark.macro_positions.detach().cpu().to(dtype=placement.dtype)
    disp = torch.linalg.norm(placement.detach().cpu() - init, dim=1)
    hard_count = int(getattr(benchmark, "num_hard_macros", placement.shape[0]))
    hard_disp = disp[:hard_count]
    soft_disp = disp[hard_count:]
    return {
        "hard_mean_disp": float(hard_disp.mean().item()) if hard_disp.numel() else 0.0,
        "hard_max_disp": float(hard_disp.max().item()) if hard_disp.numel() else 0.0,
        "soft_mean_disp": float(soft_disp.mean().item()) if soft_disp.numel() else 0.0,
        "soft_max_disp": float(soft_disp.max().item()) if soft_disp.numel() else 0.0,
        "total_mean_disp": float(disp.mean().item()) if disp.numel() else 0.0,
        "total_max_disp": float(disp.max().item()) if disp.numel() else 0.0,
    }


def overlap_stats(placement: torch.Tensor, benchmark: Benchmark) -> Dict[str, float]:
    pos = placement.detach().cpu()
    sizes = benchmark.macro_sizes.detach().cpu().to(dtype=pos.dtype)
    num_hard = int(getattr(benchmark, "num_hard_macros", pos.shape[0]))
    count = 0
    total_area = 0.0
    max_area = 0.0
    touched = set()
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            ox = max(0.0, float((sizes[i, 0] + sizes[j, 0]).item()) * 0.5 - abs(float(pos[i, 0] - pos[j, 0])))
            oy = max(0.0, float((sizes[i, 1] + sizes[j, 1]).item()) * 0.5 - abs(float(pos[i, 1] - pos[j, 1])))
            if ox > 0.0 and oy > 0.0:
                area = ox * oy
                count += 1
                total_area += area
                max_area = max(max_area, area)
                touched.add(i)
                touched.add(j)
    return {
        "hard_overlap_count": float(count),
        "hard_overlap_area": float(total_area),
        "hard_max_overlap_area": float(max_area),
        "hard_macros_touched": float(len(touched)),
    }


def print_stats(label: str, stats: Dict[str, Any]) -> None:
    if not stats:
        print(f"{label}_stats empty")
        return
    parts = []
    for key in sorted(stats):
        value = stats[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    print(f"{label}_stats " + " ".join(parts))


def run_exact_check(placement: torch.Tensor, benchmark: Benchmark, plc: Any) -> None:
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    is_valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"valid={is_valid} violations={'; '.join(violations[:3]) if violations else 'none'}")
    print(
        "exact "
        f"proxy={float(costs.get('proxy_cost', 0.0)):.6g} "
        f"wirelength={float(costs.get('wirelength_cost', 0.0)):.6g} "
        f"density={float(costs.get('density_cost', 0.0)):.6g} "
        f"congestion={float(costs.get('congestion_cost', 0.0)):.6g} "
        f"overlaps={int(costs.get('overlap_count', 0))}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
