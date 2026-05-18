#!/usr/bin/env python3
"""Manual smoke test for V2RefinedEngine on ibm01."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = REPO_ROOT / "third_party" / "macro-place-challenge-2026"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
for path in (REPO_ROOT, CHALLENGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jiho_place.v2.refined_engine import V2RefinedEngine
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement
from submissions.jiho.placer import JihoPlacer


NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def main() -> int:
    benchmark_name = os.environ.get("JIHO_V2_REFINE_SMOKE_BENCH", "ibm01")
    iters = int(os.environ.get("JIHO_V2_REFINE_ITERS", "1200"))
    log_every = int(os.environ.get("JIHO_V2_REFINE_LOG_EVERY", "50"))
    max_nets_raw = os.environ.get("JIHO_V2_REFINE_MAX_NETS", "")
    max_nets = int(max_nets_raw) if max_nets_raw.strip() else None

    benchmark, plc, load_message = load_smoke_benchmark(benchmark_name)
    print("JihoPlace v2 refined smoke")
    print(f"repo_root={REPO_ROOT}")
    print(load_message)
    print(
        "benchmark "
        f"name={benchmark.name} macros={benchmark.num_macros} hard={benchmark.num_hard_macros} "
        f"soft={benchmark.num_soft_macros} nets={benchmark.num_nets} "
        f"canvas={float(benchmark.canvas_width):.3f}x{float(benchmark.canvas_height):.3f}"
    )

    old_refine = os.environ.get("JIHO_V2_REFINE")
    os.environ["JIHO_V2_REFINE"] = "0"
    try:
        start = time.time()
        v1_placement = JihoPlacer().place(benchmark)
        v1_elapsed = time.time() - start
    finally:
        if old_refine is None:
            os.environ.pop("JIHO_V2_REFINE", None)
        else:
            os.environ["JIHO_V2_REFINE"] = old_refine

    v1_cost = compute_proxy_cost(v1_placement, benchmark, plc)
    print_cost("v1", v1_cost, v1_elapsed)

    engine = V2RefinedEngine(iterations=iters, log_every=log_every, max_nets=max_nets)
    start = time.time()
    refined = engine.refine(v1_placement, benchmark, plc)
    refine_elapsed = time.time() - start
    refined_cost = compute_proxy_cost(refined, benchmark, plc)
    print_stage_logs(engine.logs)
    print_cost("v2_refined", refined_cost, refine_elapsed)

    valid, violations = validate_placement(refined, benchmark)
    delta = float(v1_cost["proxy_cost"]) - float(refined_cost["proxy_cost"])
    print(
        "summary "
        f"improved={delta > 0.0} delta={delta:.6g} "
        f"v1_proxy={float(v1_cost['proxy_cost']):.6g} refined_proxy={float(refined_cost['proxy_cost']):.6g} "
        f"valid={valid} violations={';'.join(violations[:4]) if violations else 'none'}"
    )
    if engine.proxy_stats:
        print("engine_proxy_stats " + " ".join(f"{k}={v:.6g}" for k, v in sorted(engine.proxy_stats.items())))
    if engine.legalization_stats:
        print(
            "legalization_stats "
            + " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(engine.legalization_stats.items())
            )
        )

    out_path = REPO_ROOT / "experiments" / "results" / "v2_refine_ibm01.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_placement(refined, benchmark, save_path=str(out_path), plc=plc)
    print(f"visualization={out_path}")
    return 0


def load_smoke_benchmark(name_or_path: str) -> Tuple[Benchmark, Any, str]:
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

    raise FileNotFoundError(f"Could not resolve benchmark '{name_or_path}'")


def print_cost(label: str, costs: dict, elapsed: float) -> None:
    print(
        f"{label}_cost "
        f"proxy={float(costs.get('proxy_cost', 0.0)):.6g} "
        f"wirelength={float(costs.get('wirelength_cost', 0.0)):.6g} "
        f"density={float(costs.get('density_cost', 0.0)):.6g} "
        f"congestion={float(costs.get('congestion_cost', 0.0)):.6g} "
        f"overlaps={int(costs.get('overlap_count', 0))} "
        f"elapsed_s={elapsed:.3f}"
    )


def print_stage_logs(logs: list[dict]) -> None:
    if not logs:
        print("stage_logs none")
        return
    print("stage_logs")
    for row in logs:
        print(
            "log "
            f"step={int(row.get('step', 0))} stage={int(row.get('stage', 0))} "
            f"loss={float(row.get('loss', 0.0)):.6g} hpwl={float(row.get('hpwl', 0.0)):.6g} "
            f"density={float(row.get('density', 0.0)):.6g} congestion={float(row.get('congestion', 0.0)):.6g} "
            f"boundary={float(row.get('boundary', 0.0)):.6g} lr={float(row.get('lr', 0.0)):.6g} "
            f"grad_norm={float(row.get('grad_norm', 0.0)):.6g} step_norm={float(row.get('step_norm', 0.0)):.6g} "
            f"forward_s={float(row.get('forward_s', 0.0)):.3f} "
            f"backward_s={float(row.get('backward_s', 0.0)):.3f} "
            f"step_s={float(row.get('step_s', 0.0)):.3f} "
            f"repair_s={float(row.get('repair_s', 0.0)):.3f} "
            f"hard_mean_disp={float(row.get('hard_mean_disp', 0.0)):.6g} "
            f"hard_max_disp={float(row.get('hard_max_disp', 0.0)):.6g} "
            f"total_mean_disp={float(row.get('total_mean_disp', 0.0)):.6g} "
            f"total_max_disp={float(row.get('total_max_disp', 0.0)):.6g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
