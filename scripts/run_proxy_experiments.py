#!/usr/bin/env python3
"""Run proxy-placement experiments and append CSV results.

This is a lightweight harness around the repo's existing evaluator. It does not
change scoring behavior; it just records results in a reproducible table.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.evaluate import IBM_BENCHMARKS, _load_placer, evaluate_benchmark
from macro_place.utils import validate_placement


FIELDNAMES = [
    "timestamp_utc",
    "git_sha",
    "placer_path",
    "placer_class",
    "config_summary",
    "selected_candidate",
    "selected_profile_params",
    "candidate_exact_scores",
    "congestion_alignment_summary",
    "exact_style_hotspot_summary",
    "topology_parent_delta_summary",
    "profile_sweep_scores",
    "num_profiles_evaluated",
    "runtime_breakdown",
    "execution_mode_used",
    "skipped_soft_global_reason",
    "torch_cuda_available",
    "torch_cuda_device_name",
    "soft_global_device_used",
    "soft_global_stage_log",
    "soft_global_checkpoint_log",
    "soft_global_candidate_scores",
    "soft_global_legalization_log",
    "basin_escape_preselection_log",
    "flow_drag_log",
    "random_basin_log",
    "random_basin_preselection_log",
    "num_soft_global_candidates_generated",
    "num_soft_global_candidates_kept",
    "soft_global_dropped_candidates",
    "exact_polish_start_proxy",
    "exact_polish_end_proxy",
    "exact_polish_accepted_moves",
    "num_hard_macros",
    "num_soft_macros",
    "macro_area_utilization",
    "num_edges",
    "avg_degree",
    "max_degree",
    "benchmark",
    "proxy_cost",
    "wirelength",
    "density",
    "congestion",
    "overlaps",
    "valid",
    "validation_reason",
    "validation_details",
    "runtime_sec",
    "sa_baseline",
    "replace_baseline",
    "notes",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_rows(path: Path, rows: Iterable[Dict[str, object]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _classify_validation(result: Dict[str, object]) -> Dict[str, str]:
    """Return concise validation failure reason/details for CSV logs."""
    placement = result["placement"]
    benchmark = result["benchmark"]
    is_valid, violations = validate_placement(placement, benchmark)
    if is_valid:
        return {"validation_reason": "ok", "validation_details": ""}

    reasons = []
    details = []

    if placement.shape != (benchmark.num_macros, 2):
        reasons.append("shape")
        details.append(f"shape={tuple(placement.shape)} expected={(benchmark.num_macros, 2)}")
        return {
            "validation_reason": "+".join(reasons),
            "validation_details": "; ".join(details + violations[:3])[:500],
        }

    if torch.isnan(placement).any():
        reasons.append("nan")
        details.append(f"nan_count={int(torch.isnan(placement).sum().item())}")
    if torch.isinf(placement).any():
        reasons.append("inf")
        details.append(f"inf_count={int(torch.isinf(placement).sum().item())}")

    sizes = benchmark.macro_sizes
    x_min = placement[:, 0] - sizes[:, 0] / 2
    x_max = placement[:, 0] + sizes[:, 0] / 2
    y_min = placement[:, 1] - sizes[:, 1] / 2
    y_max = placement[:, 1] + sizes[:, 1] / 2
    hard_mask = benchmark.get_hard_macro_mask()
    soft_mask = benchmark.get_soft_macro_mask()

    x_oob = (x_min < 0) | (x_max > benchmark.canvas_width)
    y_oob = (y_min < 0) | (y_max > benchmark.canvas_height)
    oob = x_oob | y_oob
    if oob.any():
        if (oob & hard_mask).any():
            reasons.append("hard_oob")
        if (oob & soft_mask).any():
            reasons.append("soft_oob")
        details.append(
            "oob="
            f"total:{int(oob.sum().item())},"
            f"hard:{int((oob & hard_mask).sum().item())},"
            f"soft:{int((oob & soft_mask).sum().item())},"
            f"x:{int(x_oob.sum().item())},"
            f"y:{int(y_oob.sum().item())}"
        )

    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        moved = torch.norm(placement[fixed_mask] - benchmark.macro_positions[fixed_mask], dim=1) > 1e-3
        if moved.any():
            fixed_indices = torch.where(fixed_mask)[0]
            moved_indices = fixed_indices[moved]
            moved_hard = int((moved_indices < benchmark.num_hard_macros).sum().item())
            moved_soft = int((moved_indices >= benchmark.num_hard_macros).sum().item())
            reasons.append("fixed_moved")
            details.append(
                f"fixed_moved=total:{int(moved.sum().item())},hard:{moved_hard},soft:{moved_soft}"
            )

    overlap_violations = [v for v in violations if "overlap" in v.lower()]
    if overlap_violations:
        overlaps = int(result.get("overlaps", 0))
        reasons.append("hard_overlap_epsilon" if overlaps == 0 else "hard_overlap")
        details.append(f"overlap_metric={overlaps}")

    if not reasons:
        reasons.append("other")

    return {
        "validation_reason": "+".join(dict.fromkeys(reasons)),
        "validation_details": "; ".join(details + violations[:3])[:500],
    }


def _benchmark_structure(benchmark) -> Dict[str, object]:
    """Compact structural features for grouping result regressions."""
    n_hard = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    canvas_area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1.0e-12)
    macro_area = float((benchmark.macro_sizes[: benchmark.num_macros, 0] * benchmark.macro_sizes[: benchmark.num_macros, 1]).sum().item())

    edge_set = set()
    degrees = [0] * n_hard
    if benchmark.net_pin_nodes:
        nets = (pins[:, 0] for pins in benchmark.net_pin_nodes if pins.numel() > 0)
    else:
        nets = benchmark.net_nodes

    for owners_tensor in nets:
        owners = sorted(set(int(x) for x in owners_tensor.tolist()))
        hard = [o for o in owners if 0 <= o < n_hard]
        for i, a in enumerate(hard):
            for b in owners:
                if b == a:
                    continue
                key = (a, b) if a < b else (b, a)
                edge_set.add(key)

    for a, b in edge_set:
        if 0 <= a < n_hard:
            degrees[a] += 1
        if 0 <= b < n_hard:
            degrees[b] += 1

    return {
        "num_hard_macros": n_hard,
        "num_soft_macros": n_soft,
        "macro_area_utilization": f"{macro_area / canvas_area:.6f}",
        "num_edges": len(edge_set),
        "avg_degree": f"{(sum(degrees) / n_hard) if n_hard else 0.0:.3f}",
        "max_degree": max(degrees) if degrees else 0,
    }


def run(args: argparse.Namespace) -> int:
    benchmarks = IBM_BENCHMARKS if args.all else args.benchmarks
    testcase_root = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
    git_sha = _git_sha()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    all_rows: List[Dict[str, object]] = []
    for placer_arg in args.placers:
        placer_path = Path(placer_arg)
        if not placer_path.is_absolute():
            placer_path = ROOT / placer_path
        placer = _load_placer(placer_path)
        placer_class = type(placer).__name__

        print(f"\n== {placer_class} ({placer_path.relative_to(ROOT)}) ==")
        for benchmark in benchmarks:
            print(f"  {benchmark}...", end=" ", flush=True)
            result = evaluate_benchmark(placer, benchmark, str(testcase_root))
            validation = _classify_validation(result)
            structure = _benchmark_structure(result["benchmark"])
            row = {
                "timestamp_utc": timestamp,
                "git_sha": git_sha,
                "placer_path": str(placer_path.relative_to(ROOT)),
                "placer_class": placer_class,
                "config_summary": getattr(placer, "config_summary", ""),
                "selected_candidate": getattr(placer, "selected_candidate", ""),
                "selected_profile_params": getattr(placer, "selected_profile_params", ""),
                "candidate_exact_scores": getattr(placer, "candidate_exact_scores", ""),
                "congestion_alignment_summary": getattr(placer, "congestion_alignment_summary", ""),
                "exact_style_hotspot_summary": getattr(placer, "exact_style_hotspot_summary", ""),
                "topology_parent_delta_summary": getattr(placer, "topology_parent_delta_summary", ""),
                "profile_sweep_scores": getattr(placer, "profile_sweep_scores", ""),
                "num_profiles_evaluated": getattr(placer, "num_profiles_evaluated", ""),
                "runtime_breakdown": getattr(placer, "runtime_breakdown", ""),
                "execution_mode_used": getattr(placer, "execution_mode_used", ""),
                "skipped_soft_global_reason": getattr(placer, "skipped_soft_global_reason", ""),
                "torch_cuda_available": getattr(placer, "torch_cuda_available", ""),
                "torch_cuda_device_name": getattr(placer, "torch_cuda_device_name", ""),
                "soft_global_device_used": getattr(placer, "soft_global_device_used", ""),
                "soft_global_stage_log": getattr(placer, "soft_global_stage_log", ""),
                "soft_global_checkpoint_log": getattr(placer, "soft_global_checkpoint_log", ""),
                "soft_global_candidate_scores": getattr(placer, "soft_global_candidate_scores", ""),
                "soft_global_legalization_log": getattr(placer, "soft_global_legalization_log", ""),
                "basin_escape_preselection_log": getattr(placer, "basin_escape_preselection_log", ""),
                "flow_drag_log": getattr(placer, "flow_drag_log", ""),
                "random_basin_log": getattr(placer, "random_basin_log", ""),
                "random_basin_preselection_log": getattr(placer, "random_basin_preselection_log", ""),
                "num_soft_global_candidates_generated": getattr(placer, "num_soft_global_candidates_generated", ""),
                "num_soft_global_candidates_kept": getattr(placer, "num_soft_global_candidates_kept", ""),
                "soft_global_dropped_candidates": getattr(placer, "soft_global_dropped_candidates", ""),
                "exact_polish_start_proxy": getattr(placer, "exact_polish_start_proxy", ""),
                "exact_polish_end_proxy": getattr(placer, "exact_polish_end_proxy", ""),
                "exact_polish_accepted_moves": getattr(placer, "exact_polish_accepted_moves", ""),
                **structure,
                "benchmark": benchmark,
                "proxy_cost": f"{result['proxy_cost']:.6f}",
                "wirelength": f"{result['wirelength']:.6f}",
                "density": f"{result['density']:.6f}",
                "congestion": f"{result['congestion']:.6f}",
                "overlaps": int(result["overlaps"]),
                "valid": bool(result["valid"]),
                "validation_reason": validation["validation_reason"],
                "validation_details": validation["validation_details"],
                "runtime_sec": f"{result['runtime']:.3f}",
                "sa_baseline": "" if result["sa_baseline"] is None else f"{result['sa_baseline']:.6f}",
                "replace_baseline": ""
                if result["replace_baseline"] is None
                else f"{result['replace_baseline']:.6f}",
                "notes": args.notes,
            }
            all_rows.append(row)
            status = "valid" if result["overlaps"] == 0 and result["valid"] else "invalid"
            reason = "" if validation["validation_reason"] == "ok" else f" reason={validation['validation_reason']}"
            selected = getattr(placer, "selected_candidate", "")
            selected_text = f" selected={selected}" if selected else ""
            print(
                f"proxy={result['proxy_cost']:.4f} "
                f"wl={result['wirelength']:.3f} den={result['density']:.3f} "
                f"cong={result['congestion']:.3f} overlaps={result['overlaps']} "
                f"{status}{reason} runtime={result['runtime']:.2f}s"
                f"{selected_text}"
            )

    _write_rows(args.out, all_rows, append=args.append)
    print(f"\nWrote {len(all_rows)} rows to {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placers",
        nargs="+",
        required=True,
        help="One or more placer .py files.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["ibm01"],
        help="IBM benchmarks to run when --all is not set.",
    )
    parser.add_argument("--all", action="store_true", help="Run all 17 IBM benchmarks.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/proxy_experiments.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV.")
    parser.add_argument("--notes", default="", help="Free-form experiment label.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
