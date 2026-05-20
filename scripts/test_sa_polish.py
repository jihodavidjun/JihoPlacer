#!/usr/bin/env python3
"""Manual smoke test for SAPolisher on ibm01."""

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

from pine_place.v1.sa_polish import SAPolisher
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement
from submissions.pineplace.placer import PinePlace


NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def main() -> int:
    benchmark_name = os.environ.get("PINE_SA_SMOKE_BENCH", "ibm01")
    time_budget = int(os.environ.get("PINE_SA_TIME", "60"))

    benchmark, plc, load_message = load_smoke_benchmark(benchmark_name)
    print("PinePlace SA polish smoke")
    print(f"repo_root={REPO_ROOT}")
    print(load_message)
    print(
        "benchmark "
        f"name={benchmark.name} macros={benchmark.num_macros} hard={benchmark.num_hard_macros} "
        f"soft={benchmark.num_soft_macros} nets={benchmark.num_nets} "
        f"canvas={float(benchmark.canvas_width):.3f}x{float(benchmark.canvas_height):.3f}"
    )

    saved_env = {key: os.environ.get(key) for key in ("PINE_SA_POLISH", "PINE_V2_REFINE", "PINE_V3_GLOBAL")}
    os.environ["PINE_SA_POLISH"] = "0"
    os.environ["PINE_V2_REFINE"] = "0"
    os.environ["PINE_V3_GLOBAL"] = "0"
    try:
        start = time.time()
        initial = PinePlace().place(benchmark)
        v1_elapsed = time.time() - start
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    initial_cost = compute_proxy_cost(initial, benchmark, plc)
    print_cost("initial", initial_cost, v1_elapsed)

    polisher = SAPolisher(device=os.environ.get("PINE_SA_DEVICE", "cuda"))
    start = time.time()
    polished = polisher.polish(initial, benchmark, plc, time_budget_s=time_budget)
    polish_elapsed = time.time() - start
    polished_cost = compute_proxy_cost(polished, benchmark, plc)
    valid, violations = validate_placement(polished, benchmark)

    initial_proxy = float(initial_cost["proxy_cost"])
    polished_proxy = float(polished_cost["proxy_cost"])
    improvement = initial_proxy - polished_proxy
    improvement_pct = 100.0 * improvement / max(initial_proxy, 1.0e-12)
    acceptance_rate = polisher.moves_accepted / max(polisher.moves_tried, 1)

    print_cost("polished", polished_cost, polish_elapsed)
    print(
        "sa_stats "
        f"initial_proxy={initial_proxy:.6g} best_proxy={polisher.best_proxy:.6g} "
        f"returned_proxy={polished_proxy:.6g} improvement_pct={improvement_pct:.4f} "
        f"moves_tried={polisher.moves_tried} moves_accepted={polisher.moves_accepted} "
        f"rejected={polisher.rejected_moves} overlap_rejects={polisher.overlap_rejects} "
        f"proxy_evals={polisher.proxy_evals} acceptance_rate={acceptance_rate:.6g} "
        f"temperature={polisher.temperature:.6g} elapsed_s={polisher.elapsed_s:.3f} "
        f"valid={valid} violations={';'.join(violations[:4]) if violations else 'none'}"
    )
    for row in polisher.logs:
        print(
            "sa_log "
            f"iter={int(row.get('iter', 0))} accepted={int(row.get('accepted_moves', 0))} "
            f"rejected={int(row.get('rejected_moves', 0))} current_proxy={float(row.get('current_proxy', 0.0)):.6g} "
            f"best_proxy={float(row.get('best_proxy', 0.0)):.6g} "
            f"temperature={float(row.get('temperature', 0.0)):.6g} elapsed_s={float(row.get('elapsed_s', 0.0)):.3f}"
        )

    out_path = REPO_ROOT / "experiments" / "results" / "sa_polish_ibm01.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_placement(polished, benchmark, save_path=str(out_path), plc=plc)
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


if __name__ == "__main__":
    raise SystemExit(main())
