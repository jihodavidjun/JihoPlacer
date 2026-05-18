#!/usr/bin/env python3
"""Manual smoke test for JihoPlace v3 global placement."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = REPO_ROOT / "third_party" / "macro-place-challenge-2026"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
for path in (REPO_ROOT, CHALLENGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jiho_place.v3.global_engine import V3GlobalEngine
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement


NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def main() -> int:
    benchmark_name = os.environ.get("JIHO_V3_SMOKE_BENCH", "ibm01")
    iterations = int(os.environ.get("JIHO_V3_ITERS", "1200"))
    starts = int(os.environ.get("JIHO_V3_NUM_STARTS", "3"))
    max_time = float(os.environ.get("JIHO_V3_MAX_TIME_SECONDS", "300"))

    benchmark, plc, load_message = load_smoke_benchmark(benchmark_name)
    print("JihoPlace v3 global smoke")
    print(f"repo_root={REPO_ROOT}")
    print(load_message)
    print(
        "benchmark "
        f"name={benchmark.name} macros={benchmark.num_macros} hard={benchmark.num_hard_macros} "
        f"soft={benchmark.num_soft_macros} nets={benchmark.num_nets} "
        f"canvas={float(benchmark.canvas_width):.3f}x{float(benchmark.canvas_height):.3f}"
    )

    engine = V3GlobalEngine(iterations=iterations, num_starts=starts, max_time_seconds=max_time)
    start = time.time()
    placement = engine.place(benchmark, plc)
    elapsed = time.time() - start
    costs = compute_proxy_cost(placement, benchmark, plc)
    valid, violations = validate_placement(placement, benchmark)

    print_stage_logs(engine.logs)
    print(
        "timing "
        f"init_s={float(engine.timing.get('init_s', 0.0)):.3f} "
        f"stage1_s={float(engine.timing.get('stage1_s', 0.0)):.3f} "
        f"stage2_s={float(engine.timing.get('stage2_s', 0.0)):.3f} "
        f"stage3_s={float(engine.timing.get('stage3_s', 0.0)):.3f} "
        f"stage4_s={float(engine.timing.get('stage4_s', 0.0)):.3f} "
        f"legalize_s={float(engine.timing.get('legalize_s', 0.0)):.3f} "
        f"total_s={elapsed:.3f}"
    )
    print(
        "final "
        f"proxy={float(costs.get('proxy_cost', 0.0)):.6g} "
        f"wirelength={float(costs.get('wirelength_cost', 0.0)):.6g} "
        f"density={float(costs.get('density_cost', 0.0)):.6g} "
        f"congestion={float(costs.get('congestion_cost', 0.0)):.6g} "
        f"overlaps={int(costs.get('overlap_count', 0))} "
        f"replace_baseline=0.9976 "
        f"delta_vs_replace={float(costs.get('proxy_cost', 0.0)) - 0.9976:.6g} "
        f"valid={valid} violations={';'.join(violations[:4]) if violations else 'none'}"
    )

    out_path = REPO_ROOT / "experiments" / "results" / "v3_global_ibm01.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_placement(placement, benchmark, save_path=str(out_path), plc=plc)
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


def print_stage_logs(logs: list[dict]) -> None:
    if not logs:
        print("stage_logs none")
        return
    print("stage_logs")
    for row in logs:
        if "final_proxy" in row:
            print(
                "candidate "
                f"seed={int(row.get('seed', 0))} proxy={float(row.get('final_proxy', 0.0)):.6g} "
                f"wirelength={float(row.get('final_wirelength', 0.0)):.6g} "
                f"density={float(row.get('final_density', 0.0)):.6g} "
                f"congestion={float(row.get('final_congestion', 0.0)):.6g} "
                f"overlaps={int(row.get('final_overlaps', 0))}"
            )
            continue
        print(
            "log "
            f"seed={int(row.get('seed', 0))} step={int(row.get('step', 0))} stage={int(row.get('stage', 0))} "
            f"loss={float(row.get('loss', 0.0)):.6g} hpwl={float(row.get('hpwl', 0.0)):.6g} "
            f"density={float(row.get('density', 0.0)):.6g} congestion={float(row.get('congestion', 0.0)):.6g} "
            f"density_w={float(row.get('density_w', 0.0)):.6g} "
            f"congestion_w={float(row.get('congestion_w', 0.0)):.6g} lr={float(row.get('lr', 0.0)):.6g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
