#!/usr/bin/env python3
"""Manual smoke test for the opt-in hotspot micro-CD candidate family."""

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

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
from submissions.jiho.placer import JihoPlacer


NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def main() -> int:
    benchmark_name = os.environ.get("JIHO_CD_SMOKE_BENCH", "ibm01")
    tuned_default = os.environ.get("JIHO_SUBMISSION_TUNED", "1").strip().lower() not in {"", "0", "false", "no", "off"}
    if not tuned_default:
        os.environ.setdefault("JIHO_HOTSPOT_CD", "1")
    os.environ.setdefault("JIHO_CD_TIME", "60")

    benchmark, plc, load_message = load_smoke_benchmark(benchmark_name)
    print("JihoPlace hotspot micro-CD smoke")
    print(f"repo_root={REPO_ROOT}")
    print(load_message)
    print(
        "benchmark "
        f"name={benchmark.name} macros={benchmark.num_macros} hard={benchmark.num_hard_macros} "
        f"soft={benchmark.num_soft_macros} nets={benchmark.num_nets} "
        f"canvas={float(benchmark.canvas_width):.3f}x{float(benchmark.canvas_height):.3f}"
    )

    saved_env = {key: os.environ.get(key) for key in ("JIHO_SA_POLISH", "JIHO_V2_REFINE", "JIHO_V3_GLOBAL")}
    os.environ["JIHO_SA_POLISH"] = "0"
    os.environ["JIHO_V2_REFINE"] = "0"
    os.environ["JIHO_V3_GLOBAL"] = "0"
    try:
        placer = JihoPlacer()
        start = time.time()
        placement = placer.place(benchmark)
        elapsed = time.time() - start
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    costs = compute_proxy_cost(placement, benchmark, plc)
    valid, violations = validate_placement(placement, benchmark)
    print_cost("selected", costs, elapsed)
    print(
        "hotspot_cd_stats "
        f"{getattr(placer, 'hotspot_micro_cd_log', 'missing')} "
        f"selected_candidate={getattr(placer, 'selected_candidate', '')} "
        f"valid={valid} violations={';'.join(violations[:4]) if violations else 'none'}"
    )
    print(
        "heuristic_search_stats "
        f"{getattr(placer, 'heuristic_search_log', 'missing')} "
        f"selected_candidate={getattr(placer, 'selected_candidate', '')}"
    )
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
