"""
JihoPlacer v1 - density-aware multi-start local search.

This placer is intentionally self-contained for submission use. It starts from
the provided initial placement, legalizes hard macros, then runs a local search
with cheap surrogate deltas for wirelength, density, and coarse congestion.

Key choices:
- hard macros must be overlap-free;
- soft macros are moved conservatively after hard placement;
- exact proxy cost is used only for final candidate selection, not inner loops.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


Owner = int
Edge = Tuple[int, int, float]
Candidate = Tuple[float, np.ndarray, bool, str, Optional[torch.Tensor]]


def _load_plc_for_exact(name: str):
    """Load a PlacementCost object for exact final scoring when available."""
    try:
        from macro_place.loader import load_benchmark, load_benchmark_from_dir
    except Exception:
        return None

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    design = ng45.get(name)
    if design:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / design
            / "netlist"
            / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"), name=name)
            return plc
    return None


def _owner_positions(benchmark: Benchmark) -> np.ndarray:
    """Return positions for hard+soft macros plus ports in one owner array."""
    macro_pos = benchmark.macro_positions.numpy().astype(np.float64)
    if benchmark.port_positions.shape[0] == 0:
        return macro_pos
    ports = benchmark.port_positions.numpy().astype(np.float64)
    return np.vstack([macro_pos, ports])


def _extract_weighted_edges(benchmark: Benchmark) -> Tuple[List[Edge], List[List[int]], List[List[Tuple[int, float]]]]:
    """Build pairwise weighted edges from net ownership.

    Edges are generated from each net's unique owners. Ports are represented as
    owner indices >= benchmark.num_macros. The returned incident list is indexed
    by hard macro id and stores edge ids touching that macro.
    """
    n_hard = benchmark.num_hard_macros
    edge_weights: Dict[Tuple[int, int], float] = {}
    soft_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(benchmark.num_soft_macros)]

    nets: Iterable[torch.Tensor]
    if benchmark.net_pin_nodes:
        nets = (pins[:, 0] for pins in benchmark.net_pin_nodes if pins.numel() > 0)
    else:
        nets = benchmark.net_nodes

    for owners_tensor in nets:
        owners = sorted(set(int(x) for x in owners_tensor.tolist()))
        if len(owners) < 2:
            continue

        hard_owners = [o for o in owners if 0 <= o < n_hard]
        if not hard_owners:
            continue

        weight = 1.0 / max(1, len(owners) - 1)
        for a_i, a in enumerate(owners):
            if benchmark.num_hard_macros <= a < benchmark.num_macros:
                soft_idx = a - benchmark.num_hard_macros
                for b in owners:
                    if b != a:
                        soft_neighbors[soft_idx].append((b, weight))

            if a >= n_hard:
                continue
            for b in owners[a_i + 1 :]:
                if b == a:
                    continue
                key = (a, b) if a < b else (b, a)
                edge_weights[key] = edge_weights.get(key, 0.0) + weight

    edges: List[Edge] = [(a, b, w) for (a, b), w in edge_weights.items()]
    incident: List[List[int]] = [[] for _ in range(n_hard)]
    for edge_id, (a, b, _w) in enumerate(edges):
        if a < n_hard:
            incident[a].append(edge_id)
        if b < n_hard:
            incident[b].append(edge_id)

    return edges, incident, soft_neighbors


class JihoPlacer:
    """Competition placer discovered by evaluate.py via the place() method."""

    def __init__(self):
        self.base_seed = 42
        self.max_exact_candidates = 4
        self.use_soft_motion = False
        self.use_density = True
        self.use_congestion = False
        self.use_cluster_shift = False
        self.use_soft_bounds_repair = True
        self.num_seeds: Optional[int] = 1
        self.exact_final_select = True
        self.execution_mode = os.environ.get("JIHO_EXECUTION_MODE", "auto")
        self.use_soft_global_gpu = True
        self.soft_global_iters = (300, 500, 200)
        self.soft_global_iters_debug = (10, 10, 0)
        self.soft_global_lr = 0.02
        self.soft_global_device = "auto"
        self.soft_global_exact_select = True
        self.soft_global_use_diff_congestion = True
        self.soft_global_max_nets_debug = 256
        self.soft_global_schedules = (
            "spread_only",
            "spread_cong",
            "balanced_full",
            "congestion_guarded",
        )
        self.soft_global_checkpoint_every = 40
        self.soft_global_checkpoint_top_k = 4
        self.soft_global_density_target_scale = 1.04
        self.soft_global_soft_disp_weight = 0.45
        self.soft_global_congestion_target_scale = 1.25
        self.soft_global_legalized_min_disp = 0.08
        self.use_soft_global_partition_refined = os.environ.get("JIHO_USE_PARTITION_REFINED", "0") == "1"
        self.use_hotspot_cd = os.environ.get("JIHO_HOTSPOT_CD", "0") == "1"
        self.use_old_meta_fallback = True
        self.use_analytical_global_place = False
        self.use_profile_sweep = False
        self.profile_sweep_count = 12
        self.profile_sweep_top_k = 6
        self.profile_sweep_polish_top_k = 2
        self.use_exact_proxy_polish = False
        self.exact_polish_max_moves = 10
        self.exact_polish_candidate_macros = 16
        self.exact_polish_step_scales = (0.003, 0.006, 0.012)
        self.high_congestion_exact_polish = os.environ.get("JIHO_HIGH_CONG_EXACT_POLISH", "0") != "0"
        self.high_congestion_exact_polish_max_moves = int(os.environ.get("JIHO_HIGH_CONG_EXACT_POLISH_MAX_MOVES", "80"))
        self.high_congestion_exact_polish_candidate_macros = int(
            os.environ.get("JIHO_HIGH_CONG_EXACT_POLISH_CANDIDATE_MACROS", "32")
        )
        self.topology_disp_scale = float(os.environ.get("JIHO_TOPO_DISP_SCALE", "1.0"))
        self.topology_max_hard_disp = float(os.environ.get("JIHO_TOPO_MAX_HARD_DISP", "0.34"))
        self.topology_enable_mirror = os.environ.get("JIHO_TOPO_ENABLE_MIRROR", "0") == "1"
        self.analytical_stage_iters = (18, 24, 12)
        self.analytical_attraction_weight = 0.026
        self.analytical_repulsion_weight = 0.040
        self.analytical_density_weight = 0.030
        self.analytical_boundary_weight = 0.040
        self.analytical_momentum = 0.45
        self.analytical_repair_every = 64
        self.analytical_congestion_weight = 0.018
        self.analytical_congestion_grid_size = 14
        self.analytical_congestion_target_scale = 1.15
        self.use_force_directed_candidate = False
        self.fd_iters = 28
        self.fd_attraction_weight = 0.035
        self.fd_repulsion_weight = 0.075
        self.fd_density_weight = 0.025
        self._selected_use_soft = True
        self._selected_repair_soft_bounds = True
        self.selected_candidate = ""
        self.candidate_exact_scores = ""
        self.selected_profile_params = ""
        self.profile_sweep_scores = ""
        self.num_profiles_evaluated = 0
        self.runtime_breakdown = ""
        self.exact_polish_start_proxy = ""
        self.exact_polish_end_proxy = ""
        self.exact_polish_accepted_moves = 0
        self.execution_mode_used = ""
        self.skipped_soft_global_reason = ""
        self.torch_cuda_available = False
        self.torch_cuda_device_name = ""
        self.soft_global_device_used = ""
        self.soft_global_stage_log = ""
        self.soft_global_candidate_scores = ""
        self.soft_global_checkpoint_log = ""
        self.soft_global_legalization_log = ""
        self.congestion_alignment_summary = ""
        self.exact_style_hotspot_summary = ""
        self.topology_parent_delta_summary = ""
        self.num_soft_global_candidates_generated = 0
        self.num_soft_global_candidates_kept = 0
        self.soft_global_dropped_candidates = ""
        self.basin_escape_preselection_log = ""
        self.flow_drag_log = ""
        self.hotspot_micro_cd_log = ""
        self.random_basin_log = ""
        self.random_basin_preselection_log = ""
        self._candidate_parent_label: Dict[str, str] = {}
        self._candidate_parent_metrics: Dict[str, Dict[str, float]] = {}
        self._profile_params_by_label: Dict[str, str] = {}
        self._macro_alloc_cache: Dict[str, Tuple[float, float]] = {}

    def _config_summary(self) -> str:
        seed_text = "all" if self.num_seeds is None else str(self.num_seeds)
        return (
            "JihoPlacer("
            f"use_soft_motion={self.use_soft_motion}, "
            f"use_density={self.use_density}, "
            f"use_congestion={self.use_congestion}, "
            f"use_cluster_shift={self.use_cluster_shift}, "
            f"use_soft_bounds_repair={self.use_soft_bounds_repair}, "
            f"num_seeds={seed_text}, "
            f"exact_final_select={self.exact_final_select}, "
            f"execution_mode={self.execution_mode}, "
            f"use_soft_global_gpu={self.use_soft_global_gpu}, "
            f"soft_global_iters={self.soft_global_iters}, "
            f"soft_global_iters_debug={self.soft_global_iters_debug}, "
            f"soft_global_lr={self.soft_global_lr}, "
            f"soft_global_device={self.soft_global_device}, "
            f"soft_global_exact_select={self.soft_global_exact_select}, "
            f"soft_global_use_diff_congestion={self.soft_global_use_diff_congestion}, "
            f"soft_global_max_nets_debug={self.soft_global_max_nets_debug}, "
            f"soft_global_schedules={self.soft_global_schedules}, "
            f"soft_global_checkpoint_every={self.soft_global_checkpoint_every}, "
            f"soft_global_checkpoint_top_k={self.soft_global_checkpoint_top_k}, "
            f"soft_global_density_target_scale={self.soft_global_density_target_scale}, "
            f"soft_global_soft_disp_weight={self.soft_global_soft_disp_weight}, "
            f"soft_global_congestion_target_scale={self.soft_global_congestion_target_scale}, "
            f"use_hotspot_cd={self.use_hotspot_cd}, "
            f"use_soft_global_partition_refined={self.use_soft_global_partition_refined}, "
            f"use_old_meta_fallback={self.use_old_meta_fallback}, "
            f"use_analytical_global_place={self.use_analytical_global_place}, "
            f"use_profile_sweep={self.use_profile_sweep}, "
            f"profile_sweep_count={self.profile_sweep_count}, "
            f"profile_sweep_top_k={self.profile_sweep_top_k}, "
            f"profile_sweep_polish_top_k={self.profile_sweep_polish_top_k}, "
            f"use_exact_proxy_polish={self.use_exact_proxy_polish}, "
            f"exact_polish_max_moves={self.exact_polish_max_moves}, "
            f"exact_polish_candidate_macros={self.exact_polish_candidate_macros}, "
            f"exact_polish_step_scales={self.exact_polish_step_scales}, "
            f"high_congestion_exact_polish={self.high_congestion_exact_polish}, "
            f"high_congestion_exact_polish_max_moves={self.high_congestion_exact_polish_max_moves}, "
            f"high_congestion_exact_polish_candidate_macros={self.high_congestion_exact_polish_candidate_macros}, "
            f"topology_disp_scale={self.topology_disp_scale}, "
            f"topology_max_hard_disp={self.topology_max_hard_disp}, "
            f"topology_enable_mirror={self.topology_enable_mirror}, "
            f"analytical_stage_iters={self.analytical_stage_iters}, "
            f"analytical_attraction_weight={self.analytical_attraction_weight}, "
            f"analytical_repulsion_weight={self.analytical_repulsion_weight}, "
            f"analytical_density_weight={self.analytical_density_weight}, "
            f"analytical_boundary_weight={self.analytical_boundary_weight}, "
            f"analytical_momentum={self.analytical_momentum}, "
            f"analytical_repair_every={self.analytical_repair_every}, "
            f"analytical_congestion_weight={self.analytical_congestion_weight}, "
            f"analytical_congestion_grid_size={self.analytical_congestion_grid_size}, "
            f"analytical_congestion_target_scale={self.analytical_congestion_target_scale}, "
            f"use_force_directed_candidate={self.use_force_directed_candidate}, "
            f"fd_iters={self.fd_iters}, "
            f"fd_attraction_weight={self.fd_attraction_weight}, "
            f"fd_repulsion_weight={self.fd_repulsion_weight}, "
            f"fd_density_weight={self.fd_density_weight}"
            ")"
        )

    @property
    def config_summary(self) -> str:
        """Human-readable toggle summary for experiment logs."""
        return self._config_summary()

    def _cuda_device_name(self) -> str:
        if not torch.cuda.is_available():
            return ""
        try:
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            return "cuda"

    def _sync_timing_device(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _component_timer_start(self, device: torch.device) -> float:
        self._sync_timing_device(device)
        return time.perf_counter()

    def _component_timer_add(
        self, timings: Dict[str, float], name: str, device: torch.device, started_at: float
    ) -> None:
        self._sync_timing_device(device)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    def _maybe_component_timer_start(self, device: torch.device, enabled: bool) -> float:
        if not enabled:
            return 0.0
        return self._component_timer_start(device)

    def _maybe_component_timer_add(
        self, timings: Dict[str, float], name: str, device: torch.device, started_at: float, enabled: bool
    ) -> None:
        if enabled:
            self._component_timer_add(timings, name, device, started_at)

    def _compute_proxy_cost_timed(self, compute_proxy_cost, placement, benchmark, plc, label: str):
        print(f"[JihoPlacer][exact] start label={label}", flush=True)
        t0 = time.perf_counter()
        costs = compute_proxy_cost(placement, benchmark, plc)
        elapsed = time.perf_counter() - t0
        print(
            "[JihoPlacer][exact] done "
            f"label={label} elapsed={elapsed:.3f}s "
            f"proxy={float(costs.get('proxy_cost', float('nan'))):.6f} "
            f"wl={float(costs.get('wirelength_cost', float('nan'))):.6f} "
            f"density={float(costs.get('density_cost', float('nan'))):.6f} "
            f"congestion={float(costs.get('congestion_cost', float('nan'))):.6f} "
            f"overlap={int(costs.get('overlap_count', 999999))}",
            flush=True,
        )
        return costs, elapsed

    def _is_large_soft_global_design(self, benchmark: Benchmark) -> bool:
        return int(benchmark.num_hard_macros) >= 650 or int(benchmark.num_macros) >= 1800

    def _benchmark_profile(self, benchmark: Benchmark, edges: Optional[List[Edge]] = None) -> Dict[str, object]:
        edge_count = (
            len(edges)
            if edges is not None
            else int(getattr(self, "_profile_edge_count_override", 0) or getattr(benchmark, "num_edges", 0) or 0)
        )
        n_hard = int(benchmark.num_hard_macros)
        num_macros = int(benchmark.num_macros)
        num_soft = int(benchmark.num_soft_macros)
        degree = np.zeros(max(num_macros, 1), dtype=np.float64)
        if edges is not None:
            for edge in edges:
                a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= a < num_macros:
                    degree[a] += weight
                if 0 <= b < num_macros:
                    degree[b] += weight
        denom = max(1, n_hard)
        avg_degree = float(np.mean(degree[:num_macros])) if edges is not None and num_macros > 0 else (
            (2.0 * float(edge_count) / denom) if edge_count else 0.0
        )
        max_degree = float(np.max(degree[:num_macros])) if edges is not None and num_macros > 0 else 0.0
        macro_area = float(torch.prod(benchmark.macro_sizes[:num_macros], dim=1).sum().item()) if num_macros else 0.0
        canvas_area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1.0e-9)
        macro_area_util = macro_area / canvas_area
        large_high = (
            edge_count > 3500
            or avg_degree > 8.0
            or max_degree > 35.0
            or num_macros >= 1700
            or num_soft >= 1500
        )
        small_good = (not large_high) and edge_count < 1400 and num_macros < 900
        return {
            "large_high_congestion": bool(large_high),
            "high_congestion_risk": bool(large_high),
            "dense_axis": bool(large_high and num_soft >= n_hard),
            "small_good": bool(small_good),
            "num_hard_macros": n_hard,
            "num_soft_macros": num_soft,
            "macro_area_utilization": float(macro_area_util),
            "num_edges": int(edge_count),
            "avg_degree": float(avg_degree),
            "max_degree": float(max_degree),
        }

    def _resolve_execution_mode(self, benchmark: Benchmark) -> str:
        requested = str(getattr(self, "execution_mode", "auto")).lower()
        allowed = {"auto", "local_dev", "small_cpu_soft_global", "cuda_debug", "cuda_competition"}
        if requested not in allowed:
            requested = "auto"
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda_debug"
        if not self._is_large_soft_global_design(benchmark):
            return "small_cpu_soft_global"
        return "local_dev"

    def _execution_uses_cuda(self) -> bool:
        return self.execution_mode_used in {"cuda_debug", "cuda_competition"} and torch.cuda.is_available()

    def _soft_global_uses_diff_congestion(self) -> bool:
        if self.execution_mode_used == "cuda_debug":
            return False
        if self.execution_mode_used != "cuda_competition":
            return False
        return bool(self.soft_global_use_diff_congestion)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        start = time.time()
        self.selected_candidate = ""
        self.candidate_exact_scores = ""
        self.selected_profile_params = ""
        self.profile_sweep_scores = ""
        self.num_profiles_evaluated = 0
        self.runtime_breakdown = ""
        self.exact_polish_start_proxy = ""
        self.exact_polish_end_proxy = ""
        self.exact_polish_accepted_moves = 0
        self.execution_mode_used = ""
        self.skipped_soft_global_reason = ""
        self.torch_cuda_available = bool(torch.cuda.is_available())
        self.torch_cuda_device_name = self._cuda_device_name()
        self.soft_global_device_used = ""
        self.soft_global_stage_log = ""
        self.soft_global_candidate_scores = ""
        self.soft_global_checkpoint_log = ""
        self.soft_global_legalization_log = ""
        self.congestion_alignment_summary = ""
        self.exact_style_hotspot_summary = ""
        self.topology_parent_delta_summary = ""
        self.num_soft_global_candidates_generated = 0
        self.num_soft_global_candidates_kept = 0
        self._candidate_parent_label = {}
        self._candidate_parent_metrics = {}
        self.soft_global_dropped_candidates = ""
        self.basin_escape_preselection_log = ""
        self.flow_drag_log = ""
        self.hotspot_micro_cd_log = ""
        self.random_basin_log = ""
        self.random_basin_preselection_log = ""
        self._profile_params_by_label = {}
        runtime_parts: Dict[str, float] = {}

        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        initial_hard = benchmark.macro_positions[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy()
        owner_pos = _owner_positions(benchmark)
        edges, incident, soft_neighbors = _extract_weighted_edges(benchmark)

        if n == 0:
            return benchmark.macro_positions.clone()

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        self.execution_mode_used = self._resolve_execution_mode(benchmark)
        if self.execution_mode_used == "cuda_debug":
            self.soft_global_use_diff_congestion = False
        print(
            "[JihoPlacer] "
            f"execution_mode={self.execution_mode_used} "
            f"cuda_available={self.torch_cuda_available} "
            f"cuda_device={self.torch_cuda_device_name or 'none'} "
            f"diff_congestion={self._soft_global_uses_diff_congestion()}",
            flush=True,
        )

        seeds = self._seed_schedule(n)
        if self.num_seeds is not None:
            seeds = seeds[: max(0, int(self.num_seeds))]
        iterations = self._iteration_budget(n, len(edges))
        time_budget = self._time_budget_seconds(n)

        candidates: List[Candidate] = []
        run_soft_global = bool(self.use_soft_global_gpu)
        if self.execution_mode_used == "local_dev" and self._is_large_soft_global_design(benchmark):
            run_soft_global = False
            self.skipped_soft_global_reason = "local_dev_large_benchmark"
        elif self.execution_mode_used in {"cuda_debug", "cuda_competition"} and not torch.cuda.is_available():
            run_soft_global = False
            self.skipped_soft_global_reason = f"{self.execution_mode_used}_without_cuda"
        if run_soft_global:
            t0 = time.time()
            candidates.extend(
                self._soft_global_candidates(
                    benchmark=benchmark,
                    initial_hard=initial_hard,
                    movable=movable,
                    sizes=sizes,
                    half_w=half_w,
                    half_h=half_h,
                    cw=cw,
                    ch=ch,
                    edges=edges,
                    incident=incident,
                    owner_pos=owner_pos,
                )
            )
            runtime_parts["soft_global"] = time.time() - t0
        elif not self.skipped_soft_global_reason:
            self.skipped_soft_global_reason = "soft_global_disabled"

        legacy_edges = self._extract_hard_clique_edges(benchmark)
        if self.use_old_meta_fallback and legacy_edges[0].size > 0:
            t0 = time.time()
            rng = random.Random(self.base_seed)
            pos = self._legalize(initial_hard.copy(), movable, sizes, half_w, half_h, cw, ch, n)
            pos = self._legacy_sa_refine(
                pos,
                legacy_edges[0],
                legacy_edges[1],
                movable,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                rng,
            )
            pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
            candidates.append((self._surrogate_cost(pos, edges, owner_pos, benchmark, sizes), pos, True, "legacy_sa", None))
            runtime_parts["legacy"] = time.time() - t0

        if self.use_old_meta_fallback and self.use_profile_sweep and self.use_analytical_global_place:
            t0 = time.time()
            sweep_candidates, polished_candidates = self._run_profile_sweep(
                benchmark=benchmark,
                initial_hard=initial_hard,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                iterations=iterations,
                soft_neighbors=soft_neighbors,
            )
            candidates.extend(sweep_candidates)
            candidates.extend(polished_candidates)
            runtime_parts["profile_sweep"] = time.time() - t0

        elif self.use_old_meta_fallback and self.use_analytical_global_place:
            t0 = time.time()
            analytical_variants = [
                ("analytical_density_guarded_default", "density_guarded_default"),
                ("analytical_density_guarded_congestion", "density_guarded_congestion"),
                ("analytical_density_guarded_gentle_large", "density_guarded_gentle_large"),
            ]
            analytical_for_refine: Optional[np.ndarray] = None
            for label, profile in analytical_variants:
                analytical = self._legalize(initial_hard.copy(), movable, sizes, half_w, half_h, cw, ch, n)
                analytical = self._analytical_global_place(
                    pos=analytical,
                    movable=movable,
                    sizes=sizes,
                    half_w=half_w,
                    half_h=half_h,
                    cw=cw,
                    ch=ch,
                    edges=edges,
                    owner_pos=owner_pos,
                    benchmark=benchmark,
                    profile=profile,
                )
                analytical = self._repair_all_overlaps(analytical, movable, sizes, half_w, half_h, cw, ch)
                analytical = self._clip_hard_np(analytical, benchmark)
                candidates.append(
                    (
                        self._surrogate_cost(analytical, edges, owner_pos, benchmark, sizes),
                        analytical,
                        True,
                        label,
                        None,
                    )
                )
                if profile == "density_guarded_default":
                    analytical_for_refine = analytical

            analytical_refined_start = analytical_for_refine if analytical_for_refine is not None else candidates[-1][1]
            analytical_refined_iters = max(900, int(iterations * 0.35))
            analytical_refined = self._refine(
                pos=analytical_refined_start.copy(),
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                benchmark=benchmark,
                iterations=analytical_refined_iters,
                rng=random.Random(self.base_seed + 101),
                np_rng=np.random.default_rng(self.base_seed + 101),
            )
            analytical_refined = self._repair_all_overlaps(
                analytical_refined, movable, sizes, half_w, half_h, cw, ch
            )
            analytical_refined = self._clip_hard_np(analytical_refined, benchmark)
            candidates.append(
                (
                    self._surrogate_cost(analytical_refined, edges, owner_pos, benchmark, sizes),
                    analytical_refined,
                    True,
                    "analytical_refined",
                    None,
                )
            )
            runtime_parts["analytical_fixed"] = time.time() - t0

        if self.use_old_meta_fallback and self.use_force_directed_candidate and edges:
            t0 = time.time()
            fd_pos = self._legalize(initial_hard.copy(), movable, sizes, half_w, half_h, cw, ch, n)
            fd_pos = self._force_directed_candidate(
                pos=fd_pos,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                owner_pos=owner_pos,
                benchmark=benchmark,
            )
            fd_pos = self._repair_all_overlaps(fd_pos, movable, sizes, half_w, half_h, cw, ch)
            fd_pos = self._clip_hard_np(fd_pos, benchmark)
            candidates.append((self._surrogate_cost(fd_pos, edges, owner_pos, benchmark, sizes), fd_pos, True, "force_directed", None))

            refined_iters = max(1200, iterations // 2)
            refined = self._refine(
                pos=fd_pos.copy(),
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                benchmark=benchmark,
                iterations=refined_iters,
                rng=random.Random(self.base_seed + 17),
                np_rng=np.random.default_rng(self.base_seed + 17),
            )
            refined = self._repair_all_overlaps(refined, movable, sizes, half_w, half_h, cw, ch)
            refined = self._clip_hard_np(refined, benchmark)
            candidates.append(
                (
                    self._surrogate_cost(refined, edges, owner_pos, benchmark, sizes),
                    refined,
                    True,
                    "force_directed_refined",
                    None,
                )
            )
            runtime_parts["force_directed"] = time.time() - t0

        t0 = time.time()
        for seed_offset, seed in enumerate(seeds if self.use_old_meta_fallback else []):
            if not self.use_profile_sweep and time.time() - start > time_budget and candidates:
                break
            rng = random.Random(seed)
            np_rng = np.random.default_rng(seed)
            pos = initial_hard.copy()
            pos = self._legalize(pos, movable, sizes, half_w, half_h, cw, ch, n)
            pos = self._refine(
                pos=pos,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                benchmark=benchmark,
                iterations=iterations + seed_offset * max(250, iterations // 8),
                rng=rng,
                np_rng=np_rng,
            )
            pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
            surrogate = self._surrogate_cost(pos, edges, owner_pos, benchmark, sizes)
            candidates.append((surrogate, pos, False, f"local_search_s{seed}", None))

            # Also test a lightly center-relaxed version on some seeds. This is
            # often better for WL but can hurt density, so exact selection decides.
            if seed_offset == 0 or n < 450:
                relaxed = self._barycenter_polish(
                    pos.copy(), movable, edges, incident, owner_pos, sizes, half_w, half_h, cw, ch
                )
                relaxed = self._repair_all_overlaps(relaxed, movable, sizes, half_w, half_h, cw, ch)
                candidates.append((self._surrogate_cost(relaxed, edges, owner_pos, benchmark, sizes), relaxed, False, f"barycenter_s{seed}", None))
        runtime_parts["local"] = time.time() - t0

        if os.environ.get("JIHO_RANDOM_BASIN_PROBE", "0") == "1":
            t0 = time.time()
            random_candidates = self._random_basin_probe_candidates(
                candidates=candidates,
                benchmark=benchmark,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
            )
            candidates.extend(random_candidates)
            runtime_parts["random_basin"] = time.time() - t0
        else:
            self.random_basin_log = "disabled"
            self.random_basin_preselection_log = ""

        if not candidates:
            pos = self._legalize(initial_hard.copy(), movable, sizes, half_w, half_h, cw, ch, n)
            candidates.append((0.0, pos, False, "fallback_legalized", None))

        candidates = self._dedupe_candidates(candidates, benchmark)
        candidates.sort(key=lambda row: row[0])
        cpu_large_smoke = self.execution_mode_used == "local_dev" and self._is_large_soft_global_design(benchmark)
        if os.environ.get("JIHO_BASIN_ESCAPE_DEDUPE", "0") == "1" and not cpu_large_smoke:
            shortlist = self._basin_escape_exact_preselect(candidates, benchmark, edges)
        elif cpu_large_smoke:
            legal_soft = [c for c in candidates if "soft_global" in c[3] and "_legalized" in c[3]]
            shortlist = legal_soft[:1] if legal_soft else candidates[:1]
            legacy = next((c for c in candidates if c[3] == "legacy_sa"), None)
            if legacy is not None and all(legacy is not existing for existing in shortlist):
                shortlist.append(legacy)
        elif n > 400 or self._benchmark_profile(benchmark, edges)["large_high_congestion"]:
            shortlist = self._large_design_exact_shortlist(candidates, benchmark)
        else:
            shortlist = candidates[: self.max_exact_candidates]
            for candidate in candidates:
                if candidate[2] and all(candidate is not existing for existing in shortlist):
                    shortlist.append(candidate)
        shortlist = self._dedupe_candidates(shortlist, benchmark)

        if self.exact_final_select:
            t0 = time.time()
            selected, use_soft, repair_soft_bounds, selected_label, selected_full = self._select_by_exact_proxy(
                shortlist, benchmark, soft_neighbors
            )
            self._selected_use_soft = self.use_soft_motion and use_soft
            self._selected_repair_soft_bounds = self.use_soft_bounds_repair and repair_soft_bounds
            self.selected_candidate = selected_label
            runtime_parts["final_exact"] = time.time() - t0
        else:
            selected = shortlist[0][1]
            selected_full = shortlist[0][4]
            self._selected_use_soft = self.use_soft_motion
            self._selected_repair_soft_bounds = self.use_soft_bounds_repair
            suffix = "+soft_bounds" if self._selected_repair_soft_bounds else ""
            self.selected_candidate = f"{shortlist[0][3]}{suffix}"

        if self.use_exact_proxy_polish and selected_full is None:
            t0 = time.time()
            selected = self._exact_proxy_polish(
                selected=selected,
                benchmark=benchmark,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                soft_neighbors=soft_neighbors,
            )
            runtime_parts["exact_polish"] = time.time() - t0

        self.selected_profile_params = self._selected_profile_param_text(self.selected_candidate)
        runtime_parts["total"] = time.time() - start
        self.runtime_breakdown = ";".join(f"{k}={v:.3f}" for k, v in runtime_parts.items())

        selected = self._clip_hard_np(selected.copy(), benchmark)
        if selected_full is not None:
            placement = selected_full.clone()
            placement[:n] = torch.tensor(selected, dtype=torch.float32)
        else:
            placement = benchmark.macro_positions.clone()
            placement[:n] = torch.tensor(selected, dtype=torch.float32)
            if self.use_soft_motion and self._selected_use_soft:
                placement = self._place_soft_macros(placement, benchmark, soft_neighbors)
            if self.use_soft_bounds_repair and self._selected_repair_soft_bounds:
                placement = self._repair_soft_bounds_tensor(placement, benchmark)
        placement = self._repair_hard_bounds_tensor(placement, benchmark)
        fixed_mask = benchmark.macro_fixed
        if fixed_mask.any():
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return placement

    def _hotspot_cd_start_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
    ) -> Tuple[Optional[Candidate], Optional[float], Optional[float], str]:
        if not candidates:
            return None, None, None, "hotspot_cd_start=none"
        cheap_best = min(candidates, key=lambda row: float(row[0]))
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return cheap_best, None, None, f"hotspot_cd_start=cheap|label={cheap_best[3]}|reason=no_exact_import"

        plc = _load_plc_for_exact(benchmark.name)
        if plc is None:
            return cheap_best, None, None, f"hotspot_cd_start=cheap|label={cheap_best[3]}|reason=no_plc"

        ordered = [cheap_best] + [candidate for candidate in candidates if candidate is not cheap_best]
        best_candidate = cheap_best
        best_proxy = float("inf")
        exact_eval_s: Optional[float] = None
        exact_count = 0
        for index, candidate in enumerate(ordered):
            _surrogate, hard_pos, _force_include, label, full_placement = candidate
            hard_pos = self._clip_hard_np(hard_pos.copy(), benchmark)
            if full_placement is not None:
                placement = full_placement.clone()
                placement[: benchmark.num_hard_macros] = torch.tensor(hard_pos, dtype=torch.float32)
                placement = self._repair_hard_bounds_tensor(placement, benchmark)
                if benchmark.macro_fixed.any():
                    placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            else:
                placement = benchmark.macro_positions.clone()
                placement[: benchmark.num_hard_macros] = torch.tensor(hard_pos, dtype=torch.float32)
                placement = self._repair_hard_bounds_tensor(placement, benchmark)
                if benchmark.macro_fixed.any():
                    placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            try:
                costs, elapsed = self._compute_proxy_cost_timed(
                    compute_proxy_cost, placement, benchmark, plc, f"hotspot_cd_start_{label}"
                )
            except Exception as exc:
                if index == 0:
                    return cheap_best, None, None, (
                        f"hotspot_cd_start=cheap|label={cheap_best[3]}|reason=exact_failed:{type(exc).__name__}"
                    )
                continue
            exact_count += 1
            if exact_eval_s is None:
                exact_eval_s = float(elapsed)
                if exact_eval_s > 8.0:
                    return cheap_best, None, exact_eval_s, (
                        f"hotspot_cd_start=cheap|label={cheap_best[3]}|"
                        f"exact_eval_s={exact_eval_s:.3f}|reason=slow_exact"
                    )
            overlaps = int(costs.get("overlap_count", 999999))
            proxy = float(costs.get("proxy_cost", float("inf"))) + overlaps * 1.0e6
            if proxy < best_proxy:
                best_proxy = proxy
                best_candidate = candidate
        if math.isfinite(best_proxy):
            return best_candidate, best_proxy, exact_eval_s, (
                f"hotspot_cd_start=exact|label={best_candidate[3]}|proxy={best_proxy:.6f}|"
                f"exact_eval_s={float(exact_eval_s or float('nan')):.3f}|exact_evals={exact_count}"
            )
        return cheap_best, None, exact_eval_s, f"hotspot_cd_start=cheap|label={cheap_best[3]}|reason=no_finite_exact"

    def _soft_global_candidates(
        self,
        benchmark: Benchmark,
        initial_hard: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
    ) -> List[Candidate]:
        n_hard = benchmark.num_hard_macros
        if benchmark.num_macros == 0:
            return []

        candidates: List[Candidate] = []
        legal_logs = []
        stage_logs = []
        checkpoint_logs = []
        self._profile_edge_count_override = len(edges)
        schedule_specs = self._soft_global_schedule_specs(benchmark)
        if not schedule_specs:
            return []

        checkpoint_offset = 0
        refined_count = 0
        generated_records = []
        max_refined = 2
        if n_hard >= 650 or benchmark.num_macros >= 1800:
            max_refined = 0
        elif n_hard >= 450 or benchmark.num_macros >= 1200:
            max_refined = 1
        profile = self._benchmark_profile(benchmark, edges)
        large_design = self._is_large_soft_global_design(benchmark) or bool(profile["large_high_congestion"])
        legal_logs.append(
            "benchmark_class|"
            f"large_high_congestion={int(bool(profile['large_high_congestion']))}|"
            f"dense_axis={int(bool(profile['dense_axis']))}|"
            f"small_good={int(bool(profile['small_good']))}|"
            f"hard={int(profile['num_hard_macros'])}|"
            f"soft={int(profile['num_soft_macros'])}|"
            f"util={float(profile['macro_area_utilization']):.4f}|"
            f"edges={int(profile['num_edges'])}|"
            f"avg_degree={float(profile['avg_degree']):.3f}|"
            f"max_degree={float(profile['max_degree']):.3f}"
        )
        total_schedules = len(schedule_specs)
        for schedule_index, schedule in enumerate(schedule_specs, start=1):
            schedule_name = str(schedule["name"])
            stage_iter_count = sum(int(stage[0]) for stage in schedule.get("stages", ()))
            schedule_start = time.perf_counter()
            print(
                "[JihoPlacer][soft_global] schedule "
                f"{schedule_index}/{total_schedules} start name={schedule_name} "
                f"stages={len(schedule.get('stages', ()))} iters={stage_iter_count} "
                f"device={self._soft_global_device()}",
                flush=True,
            )
            try:
                checkpoints, schedule_stage_log, schedule_checkpoint_log = self._soft_global_optimize(
                    benchmark, schedule
                )
            except Exception as exc:
                elapsed = time.perf_counter() - schedule_start
                print(
                    "[JihoPlacer][soft_global] schedule "
                    f"{schedule_index}/{total_schedules} failed name={schedule_name} "
                    f"elapsed={elapsed:.3f}s error={type(exc).__name__}",
                    flush=True,
                )
                stage_logs.append(f"{schedule_name}:failed={type(exc).__name__}")
                continue
            elapsed = time.perf_counter() - schedule_start
            print(
                "[JihoPlacer][soft_global] schedule "
                f"{schedule_index}/{total_schedules} done name={schedule_name} "
                f"checkpoints={len(checkpoints)} elapsed={elapsed:.3f}s",
                flush=True,
            )
            stage_logs.append(f"{schedule_name}[{schedule_stage_log}]")
            checkpoint_logs.append(f"{schedule_name}[{schedule_checkpoint_log}]")
            for local_rank, (checkpoint_label, soft_pos, objective_score) in enumerate(checkpoints):
                checkpoint_rank = checkpoint_offset + local_rank
                hard = soft_pos[:n_hard].copy()
                hard = self._clip_hard_np(hard, benchmark)
                soft = soft_pos[n_hard : benchmark.num_macros]
                label_prefix = f"soft_global_{schedule_name}_{checkpoint_label}"

                raw_full = self._full_candidate_from_parts(benchmark, hard, soft)
                raw_owner_pos = self._owner_positions_from_placement(raw_full, benchmark)
                raw_candidate = (
                    self._surrogate_cost(hard, edges, raw_owner_pos, benchmark, sizes),
                    hard,
                    not large_design,
                    f"{label_prefix}_raw",
                    raw_full,
                )
                candidates.append(raw_candidate)
                generated_records.append(
                    self._soft_global_preselect_record(
                        raw_candidate, raw_owner_pos, edges, benchmark, objective_score, legal_disp=0.0
                    )
                )

                legal_hard = self._repair_all_overlaps(hard.copy(), movable, sizes, half_w, half_h, cw, ch)
                legal_hard = self._clip_hard_np(legal_hard, benchmark)
                legal_disp = float(np.linalg.norm(legal_hard - hard, axis=1).mean() / max(max(cw, ch), 1.0e-6))
                legal_logs.append(f"{schedule_name}:{checkpoint_label}|obj={objective_score:.4f}|hard_disp={legal_disp:.5f}")
                legal_full = self._full_candidate_from_parts(benchmark, legal_hard, soft)
                legal_owner_pos = self._owner_positions_from_placement(legal_full, benchmark)
                legal_candidate = (
                    self._surrogate_cost(legal_hard, edges, legal_owner_pos, benchmark, sizes),
                    legal_hard,
                    True,
                    f"{label_prefix}_legalized",
                    legal_full,
                )
                candidates.append(legal_candidate)
                generated_records.append(
                    self._soft_global_preselect_record(
                        legal_candidate, legal_owner_pos, edges, benchmark, objective_score, legal_disp=legal_disp
                    )
                )

                if checkpoint_rank < max_refined:
                    refine_iters = 350 if n_hard < 450 else 220
                    if n_hard >= 650 or len(edges) >= 4500:
                        refine_iters = 120
                    refined_hard = self._refine(
                        pos=legal_hard.copy(),
                        movable=movable,
                        sizes=sizes,
                        half_w=half_w,
                        half_h=half_h,
                        cw=cw,
                        ch=ch,
                        edges=edges,
                        incident=incident,
                        owner_pos=legal_owner_pos,
                        benchmark=benchmark,
                        iterations=refine_iters,
                        rng=random.Random(self.base_seed + 707 + refined_count),
                        np_rng=np.random.default_rng(self.base_seed + 707 + refined_count),
                    )
                    refined_count += 1
                    refined_hard = self._repair_all_overlaps(refined_hard, movable, sizes, half_w, half_h, cw, ch)
                    refined_hard = self._clip_hard_np(refined_hard, benchmark)
                    refined_full = self._full_candidate_from_parts(benchmark, refined_hard, soft)
                    refined_owner_pos = self._owner_positions_from_placement(refined_full, benchmark)
                    refined_candidate = (
                        self._surrogate_cost(refined_hard, edges, refined_owner_pos, benchmark, sizes),
                        refined_hard,
                        True,
                        f"{label_prefix}_refined",
                        refined_full,
                    )
                    candidates.append(refined_candidate)
                    generated_records.append(
                        self._soft_global_preselect_record(
                            refined_candidate,
                            refined_owner_pos,
                            edges,
                            benchmark,
                            objective_score,
                            legal_disp=legal_disp,
                        )
                    )
            checkpoint_offset += len(checkpoints)
        density_spread = self._soft_global_density_spread_candidate(
            candidates=candidates,
            benchmark=benchmark,
            sizes=sizes,
            edges=edges,
        )
        if density_spread is not None:
            spread_candidate, spread_record, spread_log = density_spread
            candidates.append(spread_candidate)
            generated_records.append(spread_record)
            legal_logs.append(spread_log)
        axis_spread = self._soft_global_axis_spread_candidate(
            candidates=candidates,
            benchmark=benchmark,
            sizes=sizes,
            edges=edges,
        )
        if axis_spread is not None:
            axis_candidate, axis_record, axis_log = axis_spread
            candidates.append(axis_candidate)
            generated_records.append(axis_record)
            legal_logs.append(axis_log)
        corridor_v2 = self._soft_global_corridor_v2_candidates(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
            incident=incident,
        )
        for corridor_candidate, corridor_record, corridor_log in corridor_v2:
            candidates.append(corridor_candidate)
            generated_records.append(corridor_record)
            legal_logs.append(corridor_log)
        if self.use_soft_global_partition_refined:
            partition_refined = self._soft_global_partition_refined_candidate(
                candidates=candidates,
                benchmark=benchmark,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
            )
            if partition_refined is not None:
                partition_candidate, partition_record, partition_log = partition_refined
                candidates.append(partition_candidate)
                generated_records.append(partition_record)
                legal_logs.append(partition_log)
        hotspot_refined = self._soft_global_hotspot_refined_candidate(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
        )
        if hotspot_refined is not None:
            hotspot_candidate, hotspot_record, hotspot_log = hotspot_refined
            candidates.append(hotspot_candidate)
            generated_records.append(hotspot_record)
            legal_logs.append(hotspot_log)
        channel_refined = self._soft_global_channel_refined_candidate(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
        )
        if channel_refined is not None:
            channel_candidate, channel_record, channel_log = channel_refined
            candidates.append(channel_candidate)
            generated_records.append(channel_record)
            legal_logs.append(channel_log)
        congestion_refined = self._soft_global_congestion_refined_candidate(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
        )
        if congestion_refined is not None:
            refined_candidate, refined_record, refined_log = congestion_refined
            candidates.append(refined_candidate)
            generated_records.append(refined_record)
            legal_logs.append(refined_log)
        flow_drag_candidates = self._soft_global_congestion_flow_drag_candidates(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
        )
        for flow_candidate, flow_record, flow_log in flow_drag_candidates:
            candidates.append(flow_candidate)
            generated_records.append(flow_record)
            legal_logs.append(flow_log)
        topology_portfolio = self._soft_global_topology_portfolio_candidates(
            candidates=candidates,
            benchmark=benchmark,
            movable=movable,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=cw,
            ch=ch,
            edges=edges,
            incident=incident,
        )
        if topology_portfolio:
            legal_logs.append(f"topology_portfolio_enabled|generated={len(topology_portfolio)}")
        for topo_candidate, topo_record, topo_log in topology_portfolio:
            candidates.append(topo_candidate)
            generated_records.append(topo_record)
            legal_logs.append(topo_log)
        if self.use_hotspot_cd:
            try:
                from jiho_place.v1.hotspot_micro_cd import HotspotMicroCDGenerator

                cd_budget = float(os.environ.get("JIHO_CD_TIME", "180"))
                cd_start, cd_start_proxy, cd_exact_eval_s, cd_start_log = self._hotspot_cd_start_candidate(
                    candidates, benchmark
                )
                cd_generator = HotspotMicroCDGenerator(device=self._soft_global_device(), seed=self.base_seed + 911)
                hotspot_cd = cd_generator.generate(
                    engine=self,
                    candidates=candidates,
                    benchmark=benchmark,
                    movable=movable,
                    sizes=sizes,
                    half_w=half_w,
                    half_h=half_h,
                    cw=cw,
                    ch=ch,
                    edges=edges,
                    time_budget_s=min(180.0, max(1.0, cd_budget)),
                    start_candidate=cd_start,
                    start_proxy=cd_start_proxy,
                    exact_eval_time_s=cd_exact_eval_s,
                )
                for cd_candidate, cd_record, cd_log in hotspot_cd:
                    candidates.append(cd_candidate)
                    generated_records.append(cd_record)
                    legal_logs.append(cd_log)
                self.hotspot_micro_cd_log = ";".join([cd_start_log] + cd_generator.logs)
            except Exception as exc:
                self.hotspot_micro_cd_log = f"failed={type(exc).__name__}:{exc}"
                legal_logs.append(f"hotspot_micro_cd_failed={type(exc).__name__}")
        else:
            self.hotspot_micro_cd_log = "disabled"
        self.num_soft_global_candidates_generated = len(candidates)
        candidates, dropped = self._preselect_soft_global_candidates(candidates, generated_records, large_design)
        self.num_soft_global_candidates_kept = len(candidates)
        self.soft_global_dropped_candidates = ";".join(dropped)
        self.soft_global_stage_log = ";".join(stage_logs)
        self.soft_global_checkpoint_log = ";".join(checkpoint_logs)
        self.soft_global_legalization_log = ";".join(legal_logs)
        self.soft_global_candidate_scores = ";".join(
            f"{label}|sur={surrogate:.6f}" for surrogate, _hard, _force, label, _full in candidates
        )
        return candidates

    def _congestion_grid_np(self, owner_pos: np.ndarray, edges: List[Edge], benchmark: Benchmark, rows: int, cols: int):
        placement_np = owner_pos[: benchmark.num_macros] if len(owner_pos) >= benchmark.num_macros else owner_pos
        h_grid, v_grid = self._exact_style_congestion_arrays_np(placement_np, benchmark)
        exact = h_grid + v_grid
        exact_rows, exact_cols = exact.shape
        if rows == exact_rows and cols == exact_cols:
            return exact

        row_bins = np.minimum((np.arange(exact_rows) * rows) // max(exact_rows, 1), rows - 1)
        col_bins = np.minimum((np.arange(exact_cols) * cols) // max(exact_cols, 1), cols - 1)
        grid = np.zeros((rows, cols), dtype=np.float64)
        counts = np.zeros((rows, cols), dtype=np.float64)
        for r in range(exact_rows):
            br = int(row_bins[r])
            for c in range(exact_cols):
                bc = int(col_bins[c])
                grid[br, bc] += exact[r, c]
                counts[br, bc] += 1.0
        return grid / np.maximum(counts, 1.0)

    def _partition_regions(self, variant: str, cw: float, ch: float, grid: np.ndarray) -> List[Dict[str, object]]:
        if variant == "partition_quad":
            nx, ny = 2, 2
        elif variant == "partition_stripe_x":
            nx, ny = 4, 1
        elif variant == "partition_stripe_y":
            nx, ny = 1, 4
        else:
            nx, ny = (3, 2) if cw >= ch else (2, 3)

        rows, cols = grid.shape
        regions: List[Dict[str, object]] = []
        for y_id in range(ny):
            y0 = ch * float(y_id) / float(ny)
            y1 = ch * float(y_id + 1) / float(ny)
            r0 = int(np.clip(math.floor(y0 / max(ch, 1.0e-9) * rows), 0, rows - 1))
            r1 = int(np.clip(math.ceil(y1 / max(ch, 1.0e-9) * rows), r0 + 1, rows))
            for x_id in range(nx):
                x0 = cw * float(x_id) / float(nx)
                x1 = cw * float(x_id + 1) / float(nx)
                c0 = int(np.clip(math.floor(x0 / max(cw, 1.0e-9) * cols), 0, cols - 1))
                c1 = int(np.clip(math.ceil(x1 / max(cw, 1.0e-9) * cols), c0 + 1, cols))
                window = grid[r0:r1, c0:c1]
                risk = float(np.mean(window) + 0.35 * np.max(window)) if window.size else 0.0
                area = max((x1 - x0) * (y1 - y0), 1.0e-9)
                regions.append(
                    {
                        "name": f"r{x_id}_{y_id}",
                        "center": np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64),
                        "bounds": (x0, y0, x1, y1),
                        "area": area,
                        "capacity": area * 0.82,
                        "risk": risk,
                    }
                )
        max_risk = max(float(r["risk"]) for r in regions) if regions else 1.0
        for region in regions:
            region["risk_norm"] = float(region["risk"]) / max(max_risk, 1.0e-9)
        return regions

    def _region_utilization_summary(
        self,
        placement_np: np.ndarray,
        regions: List[Dict[str, object]],
        all_sizes: np.ndarray,
    ) -> str:
        if not regions:
            return "none"
        loads = np.zeros(len(regions), dtype=np.float64)
        centers = np.stack([np.asarray(r["center"], dtype=np.float64) for r in regions], axis=0)
        for idx, point in enumerate(placement_np):
            rid = int(np.argmin(np.linalg.norm(centers - point, axis=1)))
            loads[rid] += float(all_sizes[idx, 0] * all_sizes[idx, 1])
        utils = np.array(
            [loads[i] / max(float(regions[i]["capacity"]), 1.0e-9) for i in range(len(regions))], dtype=np.float64
        )
        return f"max={float(np.max(utils)):.3f},mean={float(np.mean(utils)):.3f}"

    def _build_partition_clusters(
        self,
        placement_np: np.ndarray,
        benchmark: Benchmark,
        edges: List[Edge],
        max_cluster_size: int = 42,
    ) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, int], float]]:
        num_macros = int(benchmark.num_macros)
        all_sizes = benchmark.macro_sizes[:num_macros].numpy().astype(np.float64)
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_macros)]
        degree = np.zeros(num_macros, dtype=np.float64)
        for edge in edges:
            a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= a < num_macros:
                degree[a] += weight
            if 0 <= b < num_macros:
                degree[b] += weight
            if 0 <= a < num_macros and 0 <= b < num_macros:
                adjacency[a].append((b, weight))
                adjacency[b].append((a, weight))
        for neighbors in adjacency:
            neighbors.sort(key=lambda item: item[1], reverse=True)

        used = np.zeros(num_macros, dtype=bool)
        clusters: List[Dict[str, object]] = []
        seeds = sorted(range(num_macros), key=lambda idx: (degree[idx], all_sizes[idx, 0] * all_sizes[idx, 1]), reverse=True)
        for seed in seeds:
            if used[seed]:
                continue
            members: List[int] = []
            frontier = [seed]
            while frontier and len(members) < max_cluster_size:
                cur = frontier.pop()
                if used[cur]:
                    continue
                used[cur] = True
                members.append(cur)
                for nbr, _weight in adjacency[cur][:10]:
                    if not used[nbr] and len(members) + len(frontier) < max_cluster_size:
                        frontier.append(nbr)
            if not members:
                continue
            member_arr = np.array(members, dtype=np.int64)
            areas = all_sizes[member_arr, 0] * all_sizes[member_arr, 1]
            area = float(np.sum(areas))
            centroid = np.average(placement_np[member_arr], axis=0, weights=np.maximum(areas, 1.0e-9))
            member_set = set(members)
            external = 0.0
            for idx in members:
                for nbr, weight in adjacency[idx]:
                    if nbr not in member_set:
                        external += float(weight)
            clusters.append(
                {
                    "id": len(clusters),
                    "members": members,
                    "area": area,
                    "centroid": centroid.astype(np.float64),
                    "degree": float(np.sum(degree[member_arr])),
                    "external": float(external),
                }
            )

        macro_to_cluster = {}
        for cluster in clusters:
            for member in cluster["members"]:
                macro_to_cluster[int(member)] = int(cluster["id"])
        cluster_edges: Dict[Tuple[int, int], float] = {}
        for edge in edges:
            a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
            ca = macro_to_cluster.get(a)
            cb = macro_to_cluster.get(b)
            if ca is None or cb is None or ca == cb:
                continue
            key = (min(ca, cb), max(ca, cb))
            cluster_edges[key] = cluster_edges.get(key, 0.0) + weight
        max_degree = max((float(c["degree"]) for c in clusters), default=1.0)
        max_external = max((float(c["external"]) for c in clusters), default=1.0)
        for cluster in clusters:
            cluster["degree_norm"] = float(cluster["degree"]) / max(max_degree, 1.0e-9)
            cluster["external_norm"] = float(cluster["external"]) / max(max_external, 1.0e-9)
        return clusters, cluster_edges

    def _assign_partition_clusters(
        self,
        clusters: List[Dict[str, object]],
        cluster_edges: Dict[Tuple[int, int], float],
        regions: List[Dict[str, object]],
        span: float,
        variant: str,
    ) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        loads = np.zeros(len(regions), dtype=np.float64)
        high_degree_load = np.zeros(len(regions), dtype=np.float64)
        neighbors: Dict[int, List[Tuple[int, float]]] = {int(c["id"]): [] for c in clusters}
        for (a, b), weight in cluster_edges.items():
            neighbors.setdefault(a, []).append((b, weight))
            neighbors.setdefault(b, []).append((a, weight))
        max_edge = max(cluster_edges.values(), default=1.0)
        ordered = sorted(
            clusters,
            key=lambda c: (float(c["degree"]) + 0.45 * float(c["external"]), float(c["area"])),
            reverse=True,
        )
        for cluster in ordered:
            cid = int(cluster["id"])
            best_region = 0
            best_cost = float("inf")
            for rid, region in enumerate(regions):
                fill = (loads[rid] + float(cluster["area"])) / max(float(region["capacity"]), 1.0e-9)
                over = max(0.0, fill - 1.0)
                dist = float(np.linalg.norm(np.asarray(region["center"]) - np.asarray(cluster["centroid"]))) / span
                conn = 0.0
                for nbr, weight in neighbors.get(cid, []):
                    nbr_region = assignments.get(nbr)
                    if nbr_region is None:
                        continue
                    conn += (weight / max_edge) * float(
                        np.linalg.norm(np.asarray(region["center"]) - np.asarray(regions[nbr_region]["center"]))
                    ) / span
                degree_norm = float(cluster.get("degree_norm", 0.0))
                cost = (
                    0.55 * dist
                    + 3.2 * over * over
                    + 0.42 * degree_norm * float(region.get("risk_norm", 0.0))
                    + 0.18 * float(cluster.get("external_norm", 0.0)) * dist
                    + 0.16 * conn
                )
                if variant == "partition_degree_spread":
                    cost += 0.55 * degree_norm * high_degree_load[rid]
                if cost < best_cost:
                    best_cost = cost
                    best_region = rid
            assignments[cid] = best_region
            loads[best_region] += float(cluster["area"])
            high_degree_load[best_region] += float(cluster.get("degree_norm", 0.0))
        return assignments

    def _soft_global_partition_refined_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
    ):
        profile = self._benchmark_profile(benchmark, edges)
        if not bool(profile["large_high_congestion"]) or not edges:
            return None
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3]
            and ("_legalized" in c[3] or "_refined" in c[3])
            and "_raw" not in c[3]
            and "channel_refined" not in c[3]
            and "congestion_refined" not in c[3]
            and c[4] is not None
        ]
        if not pool:
            return None
        base = min(pool, key=lambda c: c[0])
        _base_surrogate, _base_hard, _force, _base_label, base_full = base
        assert base_full is not None

        base_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[: benchmark.num_macros].numpy().astype(bool)
        span = max(float(cw), float(ch), 1.0e-9)
        base_owner = self._owner_positions_from_placement(base_full, benchmark)
        cong_grid = self._congestion_grid_np(
            base_owner,
            edges,
            benchmark,
            max(6, min(18, int(benchmark.grid_rows))),
            max(6, min(18, int(benchmark.grid_cols))),
        )
        clusters, cluster_edges = self._build_partition_clusters(base_np, benchmark, edges)
        if not clusters:
            return None

        best = None
        variants = ("partition_quad", "partition_stripe_x", "partition_stripe_y", "partition_degree_spread")
        for variant in variants:
            regions = self._partition_regions(variant, float(cw), float(ch), cong_grid)
            assignments = self._assign_partition_clusters(clusters, cluster_edges, regions, span, variant)
            placement_np = base_np.copy()
            disp_frac = 0.18
            if variant == "partition_degree_spread":
                disp_frac = 0.23
            elif variant in {"partition_stripe_x", "partition_stripe_y"}:
                disp_frac = 0.16
            moved = 0
            for cluster in clusters:
                rid = assignments.get(int(cluster["id"]))
                if rid is None:
                    continue
                target = np.asarray(regions[rid]["center"], dtype=np.float64)
                delta = target - np.asarray(cluster["centroid"], dtype=np.float64)
                norm = float(np.linalg.norm(delta))
                max_disp = disp_frac * span
                if norm > max_disp:
                    delta *= max_disp / max(norm, 1.0e-9)
                if not np.any(np.abs(delta) > 1.0e-9):
                    continue
                for idx in cluster["members"]:
                    idx = int(idx)
                    if fixed[idx]:
                        continue
                    factor = 0.52 if idx < benchmark.num_hard_macros else 0.72
                    if variant == "partition_degree_spread" and float(cluster.get("degree_norm", 0.0)) > 0.65:
                        factor += 0.08
                    half = all_sizes[idx] * 0.5
                    new_pos = placement_np[idx] + delta * factor
                    new_pos[0] = np.clip(new_pos[0], half[0], float(cw) - half[0])
                    new_pos[1] = np.clip(new_pos[1], half[1], float(ch) - half[1])
                    if float(np.linalg.norm(new_pos - placement_np[idx])) > 1.0e-4:
                        moved += 1
                    placement_np[idx] = new_pos
            if moved == 0:
                continue

            hard = placement_np[: benchmark.num_hard_macros].copy()
            hard = self._repair_all_overlaps(hard, movable, sizes, half_w, half_h, cw, ch)
            hard = self._clip_hard_np(hard, benchmark)
            trial_full = base_full.clone()
            trial_full[: benchmark.num_hard_macros] = torch.tensor(hard, dtype=torch.float32)
            trial_full[benchmark.num_hard_macros : benchmark.num_macros] = torch.tensor(
                placement_np[benchmark.num_hard_macros : benchmark.num_macros], dtype=torch.float32
            )
            trial_full = self._repair_soft_bounds_tensor(trial_full, benchmark)
            trial_full = self._repair_hard_bounds_tensor(trial_full, benchmark)
            if benchmark.macro_fixed.any():
                trial_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            trial_owner = self._owner_positions_from_placement(trial_full, benchmark)
            refine_iters = 70 if int(benchmark.num_hard_macros) < 650 else 45
            refined_hard = self._refine(
                pos=trial_full[: benchmark.num_hard_macros].numpy().astype(np.float64),
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=trial_owner,
                benchmark=benchmark,
                iterations=refine_iters,
                rng=random.Random(self.base_seed + 1301 + len(variant)),
                np_rng=np.random.default_rng(self.base_seed + 1301 + len(variant)),
            )
            refined_hard = self._repair_all_overlaps(refined_hard, movable, sizes, half_w, half_h, cw, ch)
            refined_hard = self._clip_hard_np(refined_hard, benchmark)
            trial_full[: benchmark.num_hard_macros] = torch.tensor(refined_hard, dtype=torch.float32)
            trial_full = self._repair_hard_bounds_tensor(trial_full, benchmark)
            if benchmark.macro_fixed.any():
                trial_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            trial_np = trial_full[: benchmark.num_macros].numpy().astype(np.float64)
            trial_owner = self._owner_positions_from_placement(trial_full, benchmark)
            score = self._estimate_congestion_overflow_np(trial_owner, edges, benchmark) + 0.20 * self._estimate_density_overflow_np(
                trial_np, benchmark
            )
            displacement = float(np.linalg.norm(trial_np - base_np, axis=1).mean()) / span
            score += 0.035 * displacement
            util_before = self._region_utilization_summary(base_np, regions, all_sizes)
            util_after = self._region_utilization_summary(trial_np, regions, all_sizes)
            assignment_counts: Dict[int, int] = {}
            for rid in assignments.values():
                assignment_counts[int(rid)] = assignment_counts.get(int(rid), 0) + 1
            assignment_summary = ",".join(
                f"{regions[rid]['name']}:{assignment_counts[rid]}" for rid in sorted(assignment_counts)
            )
            current = (
                score,
                variant,
                trial_full,
                refined_hard,
                moved,
                util_before,
                util_after,
                assignment_summary,
            )
            if best is None or current[0] < best[0]:
                best = current

        if best is None:
            return None
        score, variant, full, hard, moved, util_before, util_after, assignment_summary = best
        owner = self._owner_positions_from_placement(full, benchmark)
        candidate = (
            self._surrogate_cost(hard, edges, owner, benchmark, sizes),
            hard,
            True,
            "soft_global_partition_refined",
            full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            owner,
            edges,
            benchmark,
            objective_score=float(score),
            legal_disp=0.0,
        )
        return (
            candidate,
            record,
            "partition_refined|"
            f"variant={variant}|clusters={len(clusters)}|moved={moved}|assign={assignment_summary}|"
            f"util_before={util_before}|util_after={util_after}|score={float(score):.4f}",
        )

    def _soft_global_channel_refined_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
    ):
        profile = self._benchmark_profile(benchmark, edges)
        if not profile["large_high_congestion"]:
            return None
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3] and ("_legalized" in c[3] or "_refined" in c[3]) and c[4] is not None
        ]
        if not pool or not edges:
            return None

        escape_pool = [c for c in pool if "congestion_escape" in c[3]]
        base = min(escape_pool if escape_pool else pool, key=lambda c: c[0])
        _base_surrogate, base_hard, _force, _base_label, base_full = base
        assert base_full is not None
        pos = base_hard.copy()
        placement_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        rows = int(benchmark.grid_rows)
        cols = int(benchmark.grid_cols)
        cell_w = float(cw) / max(cols, 1)
        cell_h = float(ch) / max(rows, 1)
        span = max(float(cw), float(ch), 1.0e-9)
        pair_sep_x = (sizes[:, None, 0] + sizes[None, :, 0]) / 2.0
        pair_sep_y = (sizes[:, None, 1] + sizes[None, :, 1]) / 2.0

        owner_pos = (
            placement_np
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        )
        placement_np = owner_pos[: benchmark.num_macros] if len(owner_pos) >= benchmark.num_macros else hard_pos
        h_grid, v_grid = self._exact_style_congestion_arrays_np(placement_np, benchmark)
        grid = h_grid + v_grid
        row_score = grid.mean(axis=1)
        col_score = grid.mean(axis=0)
        if not np.any(row_score) and not np.any(col_score):
            return None
        hot_row_cut = float(np.percentile(row_score, 88))
        hot_col_cut = float(np.percentile(col_score, 88))
        cold_rows = np.argsort(row_score)[: max(2, rows // 5)]
        cold_cols = np.argsort(col_score)[: max(2, cols // 5)]
        hot_rows = set(int(x) for x in np.argwhere(row_score >= hot_row_cut).reshape(-1))
        hot_cols = set(int(x) for x in np.argwhere(col_score >= hot_col_cut).reshape(-1))
        if not hot_rows and not hot_cols:
            return None

        def owner_array(candidate_placement: np.ndarray) -> np.ndarray:
            if benchmark.port_positions.shape[0] == 0:
                return candidate_placement
            return np.vstack([candidate_placement, benchmark.port_positions.numpy().astype(np.float64)])

        def full_score(candidate_placement: np.ndarray) -> float:
            owner = owner_array(candidate_placement)
            return self._estimate_congestion_overflow_np(owner, edges, benchmark) + 0.18 * self._estimate_density_overflow_np(
                candidate_placement, benchmark
            )

        def channel_delta(point: np.ndarray, new_point: np.ndarray) -> float:
            old_r = int(np.clip(point[1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            old_c = int(np.clip(point[0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            new_r = int(np.clip(new_point[1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            new_c = int(np.clip(new_point[0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            old = float(row_score[old_r] + col_score[old_c])
            new = float(row_score[new_r] + col_score[new_c])
            return new - old + 0.015 * float(np.linalg.norm(new_point - point)) / span

        def target_delta(point: np.ndarray, idx: int) -> np.ndarray:
            r = int(np.clip(point[1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            c = int(np.clip(point[0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            dx = 0.0
            dy = 0.0
            if c in hot_cols:
                best_c = min(cold_cols, key=lambda cc: abs(int(cc) - c))
                dx = ((float(best_c) + 0.5) * cell_w - float(point[0]))
                dx = float(np.clip(dx, -0.035 * span, 0.035 * span))
            if r in hot_rows:
                best_r = min(cold_rows, key=lambda rr: abs(int(rr) - r))
                dy = ((float(best_r) + 0.5) * cell_h - float(point[1]))
                dy = float(np.clip(dy, -0.035 * span, 0.035 * span))
            if dx == 0.0 and dy == 0.0:
                hot_center = np.array(
                    [
                        (np.mean(list(hot_cols)) + 0.5) * cell_w if hot_cols else cw * 0.5,
                        (np.mean(list(hot_rows)) + 0.5) * cell_h if hot_rows else ch * 0.5,
                    ],
                    dtype=np.float64,
                )
                away = point - hot_center
                norm = float(np.linalg.norm(away))
                if norm > 1.0e-9:
                    away = away / norm
                    dx = away[0] * 0.015 * span
                    dy = away[1] * 0.015 * span
            return np.array([dx, dy], dtype=np.float64)

        hard_scores = []
        for idx in range(benchmark.num_hard_macros):
            if not movable[idx]:
                continue
            r = int(np.clip(pos[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            c = int(np.clip(pos[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            if r in hot_rows or c in hot_cols:
                hard_scores.append((float(row_score[r] + col_score[c]), idx))
        hard_scores.sort(reverse=True)

        accepted = 0
        for _score, idx in hard_scores[: min(30, len(hard_scores))]:
            delta = target_delta(pos[idx], idx)
            if not np.any(np.abs(delta) > 1.0e-9):
                continue
            best_pos = pos[idx].copy()
            best_delta = 0.0
            for scale in (0.30, 0.55, 0.75):
                for mask in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                    trial_point = pos[idx] + delta * np.array(mask) * scale
                    trial_point[0] = np.clip(trial_point[0], half_w[idx], cw - half_w[idx])
                    trial_point[1] = np.clip(trial_point[1], half_h[idx], ch - half_h[idx])
                    trial = pos.copy()
                    trial[idx] = trial_point
                    if self._any_overlap(trial, [idx], pair_sep_x, pair_sep_y, gap=0.025):
                        continue
                    d = channel_delta(pos[idx], trial_point)
                    if d < best_delta - 1.0e-9:
                        best_delta = d
                        best_pos = trial_point.copy()
            if best_delta < -1.0e-9:
                pos[idx] = best_pos
                placement_np[idx] = best_pos
                accepted += 1

        soft_scores = []
        for idx in range(benchmark.num_hard_macros, benchmark.num_macros):
            if bool(benchmark.macro_fixed[idx]):
                continue
            point = placement_np[idx]
            r = int(np.clip(point[1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            c = int(np.clip(point[0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            if r in hot_rows or c in hot_cols:
                soft_scores.append((float(row_score[r] + col_score[c]), idx))
        soft_scores.sort(reverse=True)
        soft_limit = 90 if benchmark.num_hard_macros < 400 else 180
        for _score, idx in soft_scores[: min(soft_limit, len(soft_scores))]:
            point = placement_np[idx]
            delta = target_delta(point, idx)
            if not np.any(np.abs(delta) > 1.0e-9):
                continue
            best_pos = point.copy()
            best_delta = 0.0
            half = all_sizes[idx] * 0.5
            for scale in (0.25, 0.45, 0.65):
                for mask in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                    trial_point = point + delta * np.array(mask) * scale
                    trial_point[0] = np.clip(trial_point[0], half[0], cw - half[0])
                    trial_point[1] = np.clip(trial_point[1], half[1], ch - half[1])
                    d = channel_delta(point, trial_point)
                    if d < best_delta - 1.0e-9:
                        best_delta = d
                        best_pos = trial_point.copy()
            if best_delta < -1.0e-9:
                placement_np[idx] = best_pos
                accepted += 1

        # One connected-neighborhood pass: nearby hot soft clusters move together
        # when their average channel direction points to a colder row/column.
        hot_soft = [idx for _score, idx in soft_scores[: min(180, len(soft_scores))]]
        hot_set = set(hot_soft)
        adjacency: Dict[int, List[int]] = {idx: [] for idx in hot_soft}
        for a, b, _w in edges:
            if a in hot_set and b in hot_set:
                adjacency[int(a)].append(int(b))
                adjacency[int(b)].append(int(a))
        visited = set()
        components: List[List[int]] = []
        for idx in hot_soft:
            if idx in visited:
                continue
            stack = [idx]
            visited.add(idx)
            comp = []
            while stack and len(comp) < 40:
                cur = stack.pop()
                comp.append(cur)
                for nxt in adjacency.get(cur, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            if len(comp) >= 3:
                components.append(comp)
        components.sort(key=len, reverse=True)
        for comp in components[:4]:
            delta = np.mean([target_delta(placement_np[idx], idx) for idx in comp], axis=0) * 0.18
            if not np.any(np.abs(delta) > 1.0e-9):
                continue
            for idx in comp:
                half = all_sizes[idx] * 0.5
                new_pos = placement_np[idx] + delta
                new_pos[0] = np.clip(new_pos[0], half[0], cw - half[0])
                new_pos[1] = np.clip(new_pos[1], half[1], ch - half[1])
                if channel_delta(placement_np[idx], new_pos) < 0.0:
                    placement_np[idx] = new_pos
                    accepted += 1

        if accepted == 0:
            return None
        pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
        pos = self._clip_hard_np(pos, benchmark)
        channel_full = base_full.clone()
        channel_full[: benchmark.num_hard_macros] = torch.tensor(pos, dtype=torch.float32)
        channel_full[benchmark.num_hard_macros : benchmark.num_macros] = torch.tensor(
            placement_np[benchmark.num_hard_macros : benchmark.num_macros], dtype=torch.float32
        )
        channel_full = self._repair_soft_bounds_tensor(channel_full, benchmark)
        channel_full = self._repair_hard_bounds_tensor(channel_full, benchmark)
        if benchmark.macro_fixed.any():
            channel_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        channel_owner = self._owner_positions_from_placement(channel_full, benchmark)
        score = full_score(channel_full[: benchmark.num_macros].numpy().astype(np.float64))
        candidate = (
            self._surrogate_cost(pos, edges, channel_owner, benchmark, sizes),
            pos,
            True,
            "soft_global_channel_refined",
            channel_full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            channel_owner,
            edges,
            benchmark,
            objective_score=score,
            legal_disp=0.0,
        )
        return candidate, record, f"channel_refined|accepted={accepted}|rows={len(hot_rows)}|cols={len(hot_cols)}|score={score:.4f}"

    def _soft_global_topology_portfolio_candidates(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
    ) -> List[Tuple[Candidate, Dict[str, object], str]]:
        profile = self._benchmark_profile(benchmark, edges)
        if not bool(profile["large_high_congestion"]) or not edges:
            return []
        num_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        if num_macros <= 0 or n_hard <= 0:
            return []

        parent_pool = [
            c
            for c in candidates
            if c[4] is not None
            and ("_legalized" in c[3] or "_refined" in c[3])
            and "_raw" not in c[3]
            and "partition_refined" not in c[3]
            and "corridor_" not in c[3]
            and "channel_refined" not in c[3]
            and "hotspot_refined" not in c[3]
            and "congestion_refined" not in c[3]
            and "soft_global_topo_" not in c[3]
        ]
        if not parent_pool:
            return []

        def parent_rank(candidate: Candidate) -> Tuple[int, float]:
            label = candidate[3]
            priority = 4
            if "spread_cong" in label and "_refined" in label:
                priority = 0
            elif "spread_cong" in label and "_legalized" in label:
                priority = 1
            elif "density_spread" in label:
                priority = 2
            elif "density_axis" in label:
                priority = 3
            return priority, float(candidate[0])

        ordered_parents = sorted(parent_pool, key=parent_rank)
        parents: List[Candidate] = []
        seen_labels = set()
        for candidate in ordered_parents:
            family = "spread" if "spread_cong" in candidate[3] else candidate[3]
            if family in seen_labels:
                continue
            seen_labels.add(family)
            parents.append(candidate)
            if len(parents) >= 3:
                break
        primary = parents[0]
        assert primary[4] is not None
        primary_np = primary[4][:num_macros].numpy().astype(np.float64)
        primary_owner = self._owner_positions_from_placement(primary[4], benchmark)
        primary_density = self._estimate_density_overflow_np(primary_np, benchmark)
        primary_congestion = self._estimate_exact_style_congestion_overflow_np(primary_np, benchmark)
        primary_score = primary_congestion + 0.52 * primary_density

        importance, hot_rows, hot_cols = self._topology_macro_importance(
            primary_np[:n_hard], benchmark, sizes, edges, primary_owner
        )
        low_targets = self._topology_low_density_targets(primary_np, benchmark, count=18)
        fixed = benchmark.macro_fixed[:num_macros].numpy().astype(bool)
        movable_all = ~fixed
        span = max(float(cw), float(ch), 1.0e-9)
        topo_scale = max(0.20, float(self.topology_disp_scale))
        default_max_disp = max(0.06, float(self.topology_max_hard_disp))
        hot_pressure_before = self._topology_exact_style_pressure(primary_np, benchmark)

        def clip_all(pos: np.ndarray) -> np.ndarray:
            all_sizes = benchmark.macro_sizes[:num_macros].numpy().astype(np.float64)
            out = pos.copy()
            out[:, 0] = np.clip(out[:, 0], all_sizes[:, 0] * 0.5, cw - all_sizes[:, 0] * 0.5)
            out[:, 1] = np.clip(out[:, 1], all_sizes[:, 1] * 0.5, ch - all_sizes[:, 1] * 0.5)
            return out

        def mirror(axis: int, parent: Candidate) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            limit = cw if axis == 0 else ch
            pos[movable_all, axis] = limit - pos[movable_all, axis]
            pos[fixed] = benchmark.macro_positions[:num_macros].numpy().astype(np.float64)[fixed]
            return clip_all(pos)

        def axis_spread(axis: int, parent: Candidate, scale: float) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            center = (cw if axis == 0 else ch) * 0.5
            pos[movable_all, axis] = center + (pos[movable_all, axis] - center) * scale
            pos[fixed] = benchmark.macro_positions[:num_macros].numpy().astype(np.float64)[fixed]
            return clip_all(pos)

        def center_evac(parent: Candidate) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
            rel = pos[:n_hard] - center
            norm = np.linalg.norm(rel, axis=1).clip(min=1.0e-9)
            rows = max(1, int(benchmark.grid_rows))
            cols = max(1, int(benchmark.grid_cols))
            hot_row_set = set(hot_rows)
            hot_col_set = set(hot_cols)
            weights = 0.075 + 0.205 * topo_scale * self._safe_norm_np(importance)
            for idx in range(n_hard):
                if not movable[idx]:
                    continue
                r = int(np.clip(math.floor(float(pos[idx, 1]) / max(ch, 1.0e-9) * rows), 0, rows - 1))
                c = int(np.clip(math.floor(float(pos[idx, 0]) / max(cw, 1.0e-9) * cols), 0, cols - 1))
                central = abs(rel[idx, 0]) < cw * 0.30 and abs(rel[idx, 1]) < ch * 0.30
                if not (central or r in hot_row_set or c in hot_col_set):
                    continue
                away = rel[idx] / norm[idx]
                if r in hot_row_set:
                    hot_y = (float(r) + 0.5) * ch / rows
                    away[1] += 0.85 if pos[idx, 1] >= hot_y else -0.85
                if c in hot_col_set:
                    hot_x = (float(c) + 0.5) * cw / cols
                    away[0] += 0.85 if pos[idx, 0] >= hot_x else -0.85
                away = away / max(float(np.linalg.norm(away)), 1.0e-9)
                pos[idx] = pos[idx] + away * span * weights[idx]
            return clip_all(pos)

        def hot_region_evac(parent: Candidate) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            rows = max(1, int(benchmark.grid_rows))
            cols = max(1, int(benchmark.grid_cols))
            priority = self._safe_norm_np(importance)
            for idx in range(n_hard):
                if not movable[idx]:
                    continue
                r = int(np.clip(math.floor(float(pos[idx, 1]) / max(ch, 1.0e-9) * rows), 0, rows - 1))
                c = int(np.clip(math.floor(float(pos[idx, 0]) / max(cw, 1.0e-9) * cols), 0, cols - 1))
                row_dist = min((abs(r - hr) for hr in hot_rows), default=rows)
                col_dist = min((abs(c - hc) for hc in hot_cols), default=cols)
                if row_dist > 1 and col_dist > 1:
                    continue
                dx = dy = 0.0
                if hot_cols:
                    nearest_col = min(hot_cols, key=lambda hc0: abs(c - hc0))
                    hot_x = (float(nearest_col) + 0.5) * cw / cols
                    dx = 1.0 if pos[idx, 0] >= hot_x else -1.0
                if hot_rows:
                    nearest_row = min(hot_rows, key=lambda hr0: abs(r - hr0))
                    hot_y = (float(nearest_row) + 0.5) * ch / rows
                    dy = 1.0 if pos[idx, 1] >= hot_y else -1.0
                direction = np.array([dx, dy], dtype=np.float64)
                if float(np.linalg.norm(direction)) <= 1.0e-9:
                    continue
                direction = direction / max(float(np.linalg.norm(direction)), 1.0e-9)
                step = span * topo_scale * (0.055 + 0.145 * priority[idx])
                pos[idx] = pos[idx] + direction * step
            return clip_all(pos)

        def edge_bias(parent: Candidate) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            priority = self._safe_norm_np(importance)
            order = [int(i) for i in np.argsort(priority)[::-1] if movable[i]][: max(12, min(48, n_hard // 4))]
            center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
            for rank, idx in enumerate(order):
                rel = pos[idx] - center
                axis = 0 if abs(rel[0]) >= abs(rel[1]) else 1
                direction = 1.0 if rel[axis] >= 0.0 else -1.0
                step = span * (0.045 + 0.075 * priority[idx]) * (0.75 if rank > len(order) * 0.5 else 1.0)
                pos[idx, axis] += direction * step
            return clip_all(pos)

        def bridge_scatter(parent: Candidate) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            priority = self._safe_norm_np(importance)
            bridge_ids = [int(i) for i in np.argsort(priority)[::-1] if movable[i]][: min(24, max(8, n_hard // 10))]
            if not low_targets:
                return clip_all(pos)
            for rank, idx in enumerate(bridge_ids):
                target = low_targets[rank % len(low_targets)]
                alpha = 0.30 + 0.18 * priority[idx]
                pos[idx] = pos[idx] + alpha * (target - pos[idx])
            return clip_all(pos)

        def cluster_quadrant(parent: Candidate) -> Tuple[np.ndarray, str]:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            seeds = [int(i) for i in np.argsort(importance)[::-1] if movable[i]][:4]
            if len(seeds) < 2:
                return clip_all(pos), "clusters=0"
            seed_slot = {
                seeds[i]: np.array([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)][i], dtype=np.float64)
                * np.array([cw, ch], dtype=np.float64)
                for i in range(len(seeds))
            }
            weights_to_seed = np.zeros((n_hard, len(seeds)), dtype=np.float64)
            seed_index = {seed: idx for idx, seed in enumerate(seeds)}
            for a, b, w in edges:
                if a < n_hard and b in seed_index:
                    weights_to_seed[a, seed_index[b]] += w
                if b < n_hard and a in seed_index:
                    weights_to_seed[b, seed_index[a]] += w
            groups = [[] for _ in seeds]
            for idx in range(n_hard):
                if not movable[idx]:
                    continue
                if weights_to_seed[idx].max() > 0.0:
                    group_id = int(np.argmax(weights_to_seed[idx]))
                else:
                    group_id = int((pos[idx, 0] >= cw * 0.5) + 2 * (pos[idx, 1] >= ch * 0.5)) % len(seeds)
                groups[group_id].append(idx)
            for group_id, members in enumerate(groups):
                if not members:
                    continue
                seed = seeds[group_id]
                target = seed_slot[seed]
                centroid = np.mean(pos[members], axis=0)
                for idx in members:
                    internal = pos[idx] - centroid
                    jitter = np.array([((idx * 37) % 11 - 5) * 0.004 * cw, ((idx * 53) % 11 - 5) * 0.004 * ch])
                    pos[idx] = 0.56 * pos[idx] + 0.44 * (target + 0.58 * internal + jitter)
            return clip_all(pos), "clusters=" + ",".join(str(len(g)) for g in groups)

        def group_migrate(parent: Candidate, axis: Optional[int] = None) -> Tuple[np.ndarray, str]:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            groups = self._topology_macro_groups(n_hard, edges, movable, importance, max_groups=8)
            rows = max(1, int(benchmark.grid_rows))
            cols = max(1, int(benchmark.grid_cols))
            low = low_targets if low_targets else [np.array([cw * 0.25, ch * 0.25]), np.array([cw * 0.75, ch * 0.75])]
            moved = 0
            summaries = []
            for group_id, members in enumerate(groups):
                if not members:
                    continue
                centroid = np.mean(pos[members], axis=0)
                r = int(np.clip(math.floor(float(centroid[1]) / max(ch, 1.0e-9) * rows), 0, rows - 1))
                c = int(np.clip(math.floor(float(centroid[0]) / max(cw, 1.0e-9) * cols), 0, cols - 1))
                row_dist = min((abs(r - hr) for hr in hot_rows), default=rows)
                col_dist = min((abs(c - hc) for hc in hot_cols), default=cols)
                if row_dist > 2 and col_dist > 2 and group_id >= 3:
                    continue
                target = min(low, key=lambda t: float(np.linalg.norm(t - centroid)))
                delta = target - centroid
                if axis == 0:
                    delta[1] = 0.0
                elif axis == 1:
                    delta[0] = 0.0
                norm = max(float(np.linalg.norm(delta)), 1.0e-9)
                max_step = span * topo_scale * (0.085 + 0.035 * min(len(members), 12) / 12.0)
                shift = delta / norm * min(norm, max_step)
                for idx in members:
                    pos[idx] = pos[idx] + shift
                moved += len(members)
                summaries.append(f"{len(members)}@{shift[0]/span:.3f},{shift[1]/span:.3f}")
                if moved >= max(28, n_hard // 3):
                    break
            return clip_all(pos), f"groups={len(groups)}|moved={moved}|moves={','.join(summaries[:4])}"

        def frontier_slide(parent: Candidate) -> Tuple[np.ndarray, str]:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            rows = max(1, int(benchmark.grid_rows))
            cols = max(1, int(benchmark.grid_cols))
            priority = self._safe_norm_np(importance)
            moved = 0
            for idx in np.argsort(priority)[::-1]:
                idx = int(idx)
                if not movable[idx]:
                    continue
                r = int(np.clip(math.floor(float(pos[idx, 1]) / max(ch, 1.0e-9) * rows), 0, rows - 1))
                c = int(np.clip(math.floor(float(pos[idx, 0]) / max(cw, 1.0e-9) * cols), 0, cols - 1))
                near_hot = min((abs(r - hr) for hr in hot_rows), default=rows) <= 1 or min(
                    (abs(c - hc) for hc in hot_cols), default=cols
                ) <= 1
                if not near_hot:
                    continue
                candidates_xy = []
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                    nr = int(np.clip(r + dy, 0, rows - 1))
                    nc = int(np.clip(c + dx, 0, cols - 1))
                    if nr in hot_rows or nc in hot_cols:
                        continue
                    candidates_xy.append(np.array([(nc + 0.5) * cw / cols, (nr + 0.5) * ch / rows], dtype=np.float64))
                if not candidates_xy:
                    continue
                target = min(candidates_xy, key=lambda t: float(np.linalg.norm(t - pos[idx])))
                alpha = topo_scale * (0.18 + 0.22 * priority[idx])
                pos[idx] = pos[idx] + alpha * (target - pos[idx])
                moved += 1
                if moved >= min(30, max(10, n_hard // 8)):
                    break
            return clip_all(pos), f"frontier_slide={moved}"

        def frontier_swap(parent: Candidate) -> Tuple[np.ndarray, str]:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            rows = max(1, int(benchmark.grid_rows))
            cols = max(1, int(benchmark.grid_cols))
            priority = self._safe_norm_np(importance)
            hot_ids = []
            cool_ids = []
            for idx in range(n_hard):
                if not movable[idx]:
                    continue
                r = int(np.clip(math.floor(float(pos[idx, 1]) / max(ch, 1.0e-9) * rows), 0, rows - 1))
                c = int(np.clip(math.floor(float(pos[idx, 0]) / max(cw, 1.0e-9) * cols), 0, cols - 1))
                near_hot = min((abs(r - hr) for hr in hot_rows), default=rows) <= 1 or min(
                    (abs(c - hc) for hc in hot_cols), default=cols
                ) <= 1
                if near_hot and priority[idx] > 0.25:
                    hot_ids.append(idx)
                elif not near_hot and priority[idx] < 0.55:
                    cool_ids.append(idx)
            swaps = 0
            used = set()
            for idx in hot_ids[:18]:
                area_i = sizes[idx, 0] * sizes[idx, 1]
                candidates_j = [
                    j
                    for j in cool_ids
                    if j not in used and abs(float(sizes[j, 0] * sizes[j, 1] - area_i)) / max(area_i, 1.0e-9) < 0.70
                ]
                if not candidates_j:
                    continue
                j = min(candidates_j, key=lambda jj: float(np.linalg.norm(pos[jj] - pos[idx])))
                old_i = pos[idx].copy()
                pos[idx] = pos[idx] + 0.70 * (pos[j] - pos[idx])
                pos[j] = pos[j] + 0.45 * (old_i - pos[j])
                used.add(j)
                swaps += 1
                if swaps >= 8:
                    break
            return clip_all(pos), f"frontier_swaps={swaps}"

        def soft_reflow(parent: Candidate, anchor: bool) -> np.ndarray:
            assert parent[4] is not None
            pos = parent[4][:num_macros].numpy().astype(np.float64)
            if anchor:
                priority = self._safe_norm_np(importance)
                low_ids = [int(i) for i in np.argsort(priority) if movable[i]][: max(10, n_hard // 5)]
                center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
                for idx in low_ids:
                    rel = pos[idx] - center
                    norm = max(float(np.linalg.norm(rel)), 1.0e-9)
                    pos[idx] = pos[idx] + rel / norm * span * 0.030
            pos = self._topology_reflow_soft_positions(pos, benchmark, alpha=0.70 if not anchor else 0.82)
            return clip_all(pos)

        cluster_pos, cluster_log = cluster_quadrant(primary)
        variants: List[Tuple[str, Candidate, object, str, float]] = [
            ("topo_hot_region_evac", primary, hot_region_evac, f"hot_rows={len(hot_rows)}|hot_cols={len(hot_cols)}", default_max_disp),
            ("topo_center_evac", primary, center_evac, f"hot_rows={len(hot_rows)}|hot_cols={len(hot_cols)}", default_max_disp * 0.90),
            ("topo_group_migrate", primary, lambda p: group_migrate(p, None), "group_axis=both", default_max_disp * 1.10),
            ("topo_group_migrate_x", primary, lambda p: group_migrate(p, 0), "group_axis=x", default_max_disp),
            ("topo_group_migrate_y", primary, lambda p: group_migrate(p, 1), "group_axis=y", default_max_disp),
            ("topo_cluster_quadrant", primary, lambda _p, cp=cluster_pos: cp, cluster_log, default_max_disp * 1.15),
            ("topo_frontier_slide", primary, frontier_slide, "frontier=slide", default_max_disp * 0.80),
            ("topo_frontier_swap", primary, frontier_swap, "frontier=swap", default_max_disp * 0.95),
            ("topo_edge_bias", primary, edge_bias, "edge=nearest_low_pressure", default_max_disp * 0.85),
            ("topo_bridge_scatter", primary, bridge_scatter, f"targets={len(low_targets)}", default_max_disp * 1.05),
            ("topo_axis_spread_x", primary, lambda p: axis_spread(0, p, 1.18 + 0.10 * topo_scale), "axis=x|scale=medium", default_max_disp * 0.75),
            ("topo_axis_spread_y", primary, lambda p: axis_spread(1, p, 1.18 + 0.10 * topo_scale), "axis=y|scale=medium", default_max_disp * 0.75),
            ("topo_soft_reflow", primary, lambda p: soft_reflow(p, False), "soft_alpha=0.70", default_max_disp * 0.50),
            ("topo_anchor_soft_redistribute", primary, lambda p: soft_reflow(p, True), "soft_alpha=0.82|anchor=high_importance", default_max_disp * 0.60),
        ]
        if self.topology_enable_mirror:
            variants.extend(
                [
                    ("topo_mirror_x", primary, lambda p: mirror(0, p), "mirror_axis=x|enabled=1", default_max_disp * 1.40),
                    ("topo_mirror_y", primary, lambda p: mirror(1, p), "mirror_axis=y|enabled=1", default_max_disp * 1.40),
                ]
            )
        if len(parents) > 1:
            variants.append(("topo_parent_density_spread", parents[1], lambda p: axis_spread(0, p, 1.14 + 0.08 * topo_scale), "parent_variant=axis_x", default_max_disp * 0.75))
        if len(parents) > 2:
            variants.append(("topo_parent_density_axis", parents[2], lambda p: axis_spread(1, p, 1.14 + 0.08 * topo_scale), "parent_variant=axis_y", default_max_disp * 0.75))

        results: List[Tuple[Candidate, Dict[str, object], str]] = []
        kept_positions: List[np.ndarray] = []
        for name, parent, builder, extra_log, max_disp_cap in variants:
            assert parent[4] is not None
            try:
                trial_np = builder(parent)  # type: ignore[misc]
            except Exception:
                continue
            if isinstance(trial_np, tuple):
                trial_np, generated_log = trial_np
                extra_log = f"{extra_log}|{generated_log}"
            built = self._soft_global_topology_build_candidate(
                name=name,
                trial_np=trial_np,
                parent=parent,
                parent_score=primary_score,
                parent_pressure=hot_pressure_before,
                max_disp_cap=max_disp_cap,
                benchmark=benchmark,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                extra_log=extra_log,
            )
            if built is None:
                continue
            candidate, record, log = built
            pos = self._candidate_positions_np(candidate, benchmark)
            if any(existing.shape == pos.shape and float(np.max(np.abs(existing - pos))) < 1.0e-5 for existing in kept_positions):
                continue
            kept_positions.append(pos)
            results.append((candidate, record, log))
            if len(results) >= 10:
                break
        return results

    def _soft_global_topology_build_candidate(
        self,
        name: str,
        trial_np: np.ndarray,
        parent: Candidate,
        parent_score: float,
        parent_pressure: float,
        max_disp_cap: float,
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        extra_log: str,
    ) -> Optional[Tuple[Candidate, Dict[str, object], str]]:
        num_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        parent_label = str(parent[3])
        parent_full = parent[4]
        if parent_full is None or trial_np.shape[0] < num_macros:
            return None
        parent_np = parent_full[:num_macros].numpy().astype(np.float64)
        hard = trial_np[:n_hard].copy()
        hard = self._repair_all_overlaps(hard, movable, sizes, half_w, half_h, cw, ch)
        hard = self._clip_hard_np(hard, benchmark)
        if benchmark.macro_fixed[:n_hard].any():
            fixed = benchmark.macro_fixed[:n_hard].numpy().astype(bool)
            hard[fixed] = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)[fixed]
        if self._any_overlap(hard, range(len(hard)), (sizes[:, 0:1] + sizes[:, 0:1].T) * 0.5, (sizes[:, 1:2] + sizes[:, 1:2].T) * 0.5, gap=0.025):
            return None
        full = parent_full.clone()
        full[:n_hard] = torch.tensor(hard, dtype=torch.float32)
        if num_macros > n_hard:
            full[n_hard:num_macros] = torch.tensor(trial_np[n_hard:num_macros], dtype=torch.float32)
        full = self._repair_soft_bounds_tensor(full, benchmark)
        full = self._repair_hard_bounds_tensor(full, benchmark)
        if benchmark.macro_fixed.any():
            full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        final_np = full[:num_macros].numpy().astype(np.float64)
        owner = self._owner_positions_from_placement(full, benchmark)
        density = self._estimate_density_overflow_np(final_np, benchmark)
        legacy_congestion = self._estimate_congestion_overflow_np(owner, edges, benchmark) if edges else 0.0
        congestion = self._estimate_exact_style_congestion_overflow_np(final_np, benchmark)
        blockage = self._cheap_corridor_blockage_np(final_np, benchmark)
        cheap_score = congestion + 0.52 * density + 0.16 * blockage
        hot_pressure = self._topology_exact_style_pressure(final_np, benchmark)
        parent_hard = parent_np[:n_hard]
        hard_disp = float(np.linalg.norm(hard - parent_hard, axis=1).mean() / max(max(cw, ch), 1.0e-9))
        hard_max_disp = float(np.linalg.norm(hard - parent_hard, axis=1).max() / max(max(cw, ch), 1.0e-9))
        if density > 0.42 or density > max(0.35, 2.6 * max(0.02, parent_score)):
            return None
        if hard_disp < 0.004 and cheap_score >= parent_score * 0.996 and hot_pressure >= parent_pressure * 0.997:
            return None
        if hard_disp > max_disp_cap and cheap_score >= parent_score * 0.972 and hot_pressure >= parent_pressure * 0.965:
            return None
        candidate = (
            self._surrogate_cost(hard, edges, owner, benchmark, sizes),
            hard,
            True,
            f"soft_global_{name}",
            full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            owner,
            edges,
            benchmark,
            objective_score=cheap_score,
            legal_disp=hard_disp,
        )
        log = (
            f"topology|name={name}|parent={parent_label}|{extra_log}|hard_disp={hard_disp:.4f}|"
            f"hard_max_disp={hard_max_disp:.4f}|moved={int(np.sum(np.linalg.norm(hard - parent_hard, axis=1) > max(max(cw, ch), 1.0e-9) * 0.004))}|"
                f"hot_pressure_before={parent_pressure:.4f}|hot_pressure_after={hot_pressure:.4f}|"
            f"cheap_d={density:.4f}|cheap_c={legacy_congestion:.4f}|style_c={congestion:.4f}|"
            f"cheap_b={blockage:.4f}|cheap_like={cheap_score:.4f}"
        )
        candidate_label = f"soft_global_{name}"
        parent_density = self._estimate_density_overflow_np(parent_np, benchmark)
        parent_style_c = self._estimate_exact_style_congestion_overflow_np(parent_np, benchmark)
        parent_blockage = self._cheap_corridor_blockage_np(parent_np, benchmark)
        self._candidate_parent_label[candidate_label] = parent_label
        self._candidate_parent_metrics[candidate_label] = {
            "density": float(parent_density),
            "style_c": float(parent_style_c),
            "blockage": float(parent_blockage),
            "style_hot": float(parent_pressure),
        }
        return candidate, record, log

    def _topology_macro_importance(
        self,
        hard_pos: np.ndarray,
        benchmark: Benchmark,
        sizes: np.ndarray,
        edges: List[Edge],
        owner_pos: np.ndarray,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        n = int(benchmark.num_hard_macros)
        degree = np.zeros(n, dtype=np.float64)
        span_score = np.zeros(n, dtype=np.float64)
        hot_score = np.zeros(n, dtype=np.float64)
        h_grid, v_grid = self._routing_congestion_arrays_np(owner_pos, edges, benchmark)
        row_score = h_grid.sum(axis=1) + v_grid.sum(axis=1)
        col_score = h_grid.sum(axis=0) + v_grid.sum(axis=0)
        hot_rows = [int(x) for x in np.argsort(row_score)[-max(1, min(5, len(row_score) // 5 or 1)):]] if row_score.size else []
        hot_cols = [int(x) for x in np.argsort(col_score)[-max(1, min(7, len(col_score) // 5 or 1)):]] if col_score.size else []
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cell_w = float(benchmark.canvas_width) / cols
        cell_h = float(benchmark.canvas_height) / rows
        hot_row_set = set(hot_rows)
        hot_col_set = set(hot_cols)

        def cell(point: np.ndarray) -> Tuple[int, int]:
            r = int(np.clip(math.floor(float(point[1]) / max(cell_h, 1.0e-9)), 0, rows - 1))
            c = int(np.clip(math.floor(float(point[0]) / max(cell_w, 1.0e-9)), 0, cols - 1))
            return r, c

        for a, b, w in edges:
            if a >= len(owner_pos) or b >= len(owner_pos):
                continue
            pa = owner_pos[a]
            pb = owner_pos[b]
            dist = abs(float(pa[0] - pb[0])) + abs(float(pa[1] - pb[1]))
            ra, ca = cell(pa)
            rb, cb = cell(pb)
            r0, r1 = sorted((ra, rb))
            c0, c1 = sorted((ca, cb))
            crosses_hot = any(r0 <= r <= r1 for r in hot_row_set) or any(c0 <= c <= c1 for c in hot_col_set)
            if a < n:
                degree[a] += w
                span_score[a] += w * dist
                hot_score[a] += w * float(crosses_hot)
            if b < n:
                degree[b] += w
                span_score[b] += w * dist
                hot_score[b] += w * float(crosses_hot)
        area = sizes[:, 0] * sizes[:, 1]
        dense = self._macro_dense_bin_scores(hard_pos, benchmark, sizes)
        importance = (
            1.20 * self._safe_norm_np(degree)
            + 0.85 * self._safe_norm_np(span_score)
            + 1.05 * self._safe_norm_np(hot_score)
            + 0.60 * self._safe_norm_np(area)
            + 0.55 * self._safe_norm_np(dense)
        )
        return importance, hot_rows, hot_cols

    def _topology_macro_groups(
        self,
        n_hard: int,
        edges: List[Edge],
        movable: np.ndarray,
        importance: np.ndarray,
        max_groups: int,
    ) -> List[List[int]]:
        adjacency: List[Dict[int, float]] = [dict() for _ in range(n_hard)]
        for a, b, w in edges:
            if 0 <= a < n_hard and 0 <= b < n_hard and movable[a] and movable[b]:
                adjacency[a][b] = adjacency[a].get(b, 0.0) + float(w)
                adjacency[b][a] = adjacency[b].get(a, 0.0) + float(w)
        seeds = [int(i) for i in np.argsort(importance)[::-1] if movable[i]][: max(2, int(max_groups))]
        groups: List[List[int]] = []
        assigned = set()
        for seed in seeds:
            if seed in assigned:
                continue
            members = [seed]
            assigned.add(seed)
            frontier = [seed]
            while frontier and len(members) < 24:
                current = frontier.pop(0)
                neighbors = sorted(adjacency[current], key=adjacency[current].get, reverse=True)
                for nb in neighbors[:6]:
                    if nb in assigned or not movable[nb]:
                        continue
                    members.append(nb)
                    assigned.add(nb)
                    frontier.append(nb)
                    if len(members) >= 24:
                        break
            if 2 <= len(members) <= 36:
                groups.append(members)
            if len(groups) >= max_groups:
                break
        return groups

    def _topology_hot_pressure(self, placement_np: np.ndarray, benchmark: Benchmark, edges: List[Edge]) -> float:
        if not edges:
            return 0.0
        owner = (
            placement_np
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        )
        h_grid, v_grid = self._routing_congestion_arrays_np(owner, edges, benchmark)
        values = np.concatenate([h_grid.reshape(-1), v_grid.reshape(-1)])
        if values.size == 0:
            return 0.0
        count = max(1, int(math.ceil(values.size * 0.08)))
        return float(np.mean(np.sort(values)[-count:]))

    def _topology_low_density_targets(self, placement_np: np.ndarray, benchmark: Benchmark, count: int) -> List[np.ndarray]:
        rows = max(4, min(10, int(benchmark.grid_rows)))
        cols = max(4, min(10, int(benchmark.grid_cols)))
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cell_w = cw / cols
        cell_h = ch / rows
        grid = np.zeros((rows, cols), dtype=np.float64)
        sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        for pos, size in zip(placement_np, sizes):
            r = int(np.clip(math.floor(float(pos[1]) / max(cell_h, 1.0e-9)), 0, rows - 1))
            c = int(np.clip(math.floor(float(pos[0]) / max(cell_w, 1.0e-9)), 0, cols - 1))
            grid[r, c] += float(size[0] * size[1]) / max(cell_w * cell_h, 1.0e-9)
        order = np.argsort(grid.reshape(-1))
        targets: List[np.ndarray] = []
        for flat in order:
            r, c = divmod(int(flat), cols)
            if 0 < r < rows - 1 and 0 < c < cols - 1 and len(targets) < count // 3:
                continue
            targets.append(np.array([(c + 0.5) * cell_w, (r + 0.5) * cell_h], dtype=np.float64))
            if len(targets) >= count:
                break
        return targets

    def _topology_reflow_soft_positions(self, placement_np: np.ndarray, benchmark: Benchmark, alpha: float) -> np.ndarray:
        out = placement_np.copy()
        n_hard = int(benchmark.num_hard_macros)
        num_macros = int(benchmark.num_macros)
        if num_macros <= n_hard:
            return out
        ports = benchmark.port_positions.numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:num_macros].numpy().astype(bool)
        owner_pos = out if ports.shape[0] == 0 else np.vstack([out, ports])
        raw_nets = benchmark.net_pin_nodes if benchmark.net_pin_nodes else benchmark.net_nodes
        weights = benchmark.net_weights.tolist() if getattr(benchmark, "net_weights", None) is not None else []
        accum = np.zeros((num_macros - n_hard, 2), dtype=np.float64)
        total = np.zeros(num_macros - n_hard, dtype=np.float64)
        for net_id, owners_tensor in enumerate(raw_nets):
            owners_raw = owners_tensor[:, 0] if getattr(owners_tensor, "ndim", 1) == 2 else owners_tensor
            owners = sorted(set(int(x) for x in owners_raw.tolist()))
            owners = [o for o in owners if 0 <= o < len(owner_pos)]
            softs = [o for o in owners if n_hard <= o < num_macros and not fixed[o]]
            if not softs or len(owners) < 2:
                continue
            weight = float(weights[net_id]) if net_id < len(weights) else 1.0
            for soft_owner in softs:
                others = [o for o in owners if o != soft_owner]
                if not others:
                    continue
                center = np.mean(owner_pos[others], axis=0)
                idx = soft_owner - n_hard
                accum[idx] += weight * center
                total[idx] += weight
        for idx in range(num_macros - n_hard):
            owner = n_hard + idx
            if fixed[owner] or total[idx] <= 0.0:
                continue
            target = accum[idx] / total[idx]
            out[owner] = (1.0 - alpha) * out[owner] + alpha * target
        return out

    def _safe_norm_np(self, values: np.ndarray) -> np.ndarray:
        mx = float(np.max(values)) if values.size else 0.0
        return values / mx if mx > 1.0e-12 else values

    def _soft_global_corridor_v2_candidates(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
    ) -> List[Tuple[Candidate, Dict[str, object], str]]:
        profile = self._benchmark_profile(benchmark, edges)
        if not profile["large_high_congestion"] or not edges:
            return []
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3]
            and ("_legalized" in c[3] or "_refined" in c[3])
            and "_raw" not in c[3]
            and c[4] is not None
            and "partition_refined" not in c[3]
            and "channel_refined" not in c[3]
            and "hotspot_refined" not in c[3]
            and "congestion_refined" not in c[3]
            and "corridor_" not in c[3]
        ]
        if not pool:
            return []

        def parent_score(candidate: Candidate) -> float:
            surrogate, _hard, _force, label, full = candidate
            assert full is not None
            placement = full[: benchmark.num_macros].numpy().astype(np.float64)
            owner = self._owner_positions_from_placement(full, benchmark)
            priority = 0.0
            if "density_axis" in label:
                priority -= 0.055
            elif "density_spread" in label:
                priority -= 0.045
            return (
                0.50 * self._estimate_density_overflow_np(placement, benchmark)
                + 0.70 * self._estimate_congestion_overflow_np(owner, edges, benchmark)
                + 1.0e-6 * float(surrogate)
                + priority
            )

        parent = min(pool, key=parent_score)
        _parent_surrogate, _parent_hard, _force, parent_label, parent_full = parent
        assert parent_full is not None
        parent_np = parent_full[: benchmark.num_macros].numpy().astype(np.float64)
        parent_owner = self._owner_positions_from_placement(parent_full, benchmark)
        parent_density = self._estimate_density_overflow_np(parent_np, benchmark)
        parent_congestion = self._estimate_congestion_overflow_np(parent_owner, edges, benchmark)

        variants = [
            {
                "name": "corridor_narrow",
                "width_scale": 0.70,
                "corridor_weight": 1.55,
                "density_weight": 2.25,
                "hpwl_weight": 0.035,
                "anchor_weight": 0.34,
                "bridge_boost": 0.82,
                "iters": 14,
                "max_disp": 0.030,
            },
            {
                "name": "corridor_wide",
                "width_scale": 1.55,
                "corridor_weight": 1.20,
                "density_weight": 2.55,
                "hpwl_weight": 0.030,
                "anchor_weight": 0.42,
                "bridge_boost": 0.72,
                "iters": 14,
                "max_disp": 0.026,
            },
            {
                "name": "corridor_bridge",
                "width_scale": 1.05,
                "corridor_weight": 1.85,
                "density_weight": 2.30,
                "hpwl_weight": 0.032,
                "anchor_weight": 0.36,
                "bridge_boost": 1.30,
                "iters": 16,
                "max_disp": 0.034,
            },
            {
                "name": "corridor_density_guarded",
                "width_scale": 1.10,
                "corridor_weight": 1.30,
                "density_weight": 3.30,
                "hpwl_weight": 0.030,
                "anchor_weight": 0.48,
                "bridge_boost": 0.85,
                "iters": 14,
                "max_disp": 0.024,
            },
        ]
        if self.execution_mode_used != "cuda_debug":
            for variant in variants:
                variant["iters"] = int(variant["iters"]) * 2

        results: List[Tuple[Candidate, Dict[str, object], str]] = []
        for variant in variants:
            built = self._soft_global_corridor_v2_optimize(
                parent_full=parent_full,
                parent_label=parent_label,
                parent_density=parent_density,
                parent_congestion=parent_congestion,
                benchmark=benchmark,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                variant=variant,
            )
            if built is not None:
                results.append(built)
        return results

    def _soft_global_corridor_v2_optimize(
        self,
        parent_full: torch.Tensor,
        parent_label: str,
        parent_density: float,
        parent_congestion: float,
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        variant: Dict[str, float],
    ):
        device = self._soft_global_device()
        dtype = torch.float32
        num_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        span = max(float(cw), float(ch), 1.0e-9)
        parent_init = parent_full[:num_macros].to(device=device, dtype=dtype)
        all_sizes_t = benchmark.macro_sizes[:num_macros].to(device=device, dtype=dtype)
        fixed_t = benchmark.macro_fixed[:num_macros].to(device=device)
        movable_t = (~fixed_t).to(device=device)
        half_t = all_sizes_t * 0.5
        low = half_t + 1.0e-4
        high = torch.tensor([cw, ch], device=device, dtype=dtype).view(1, 2) - half_t - 1.0e-4
        corridor_plan = self._soft_global_predict_corridors(
            benchmark,
            device,
            dtype,
            base_positions=parent_full[:num_macros],
            width_scale=float(variant["width_scale"]),
            weight_scale=float(variant["corridor_weight"]),
            bridge_boost=float(variant["bridge_boost"]),
            max_corridors=3,
        )
        if corridor_plan is None:
            return None

        nets = self._soft_global_nets(benchmark, device)
        rows = min(max(12, int(benchmark.grid_rows)), 18)
        cols = min(max(12, int(benchmark.grid_cols)), 18)
        density_edges = self._soft_global_bin_edges(cw, ch, rows, cols, device, dtype)
        var = torch.nn.Parameter(parent_init.clone())
        optimizer = torch.optim.AdamW([var], lr=float(self.soft_global_lr) * 0.45, weight_decay=0.0)
        max_disp = float(variant["max_disp"]) * span
        best_pos = parent_init.detach().clone()
        best_score = float("inf")
        last_parts: Dict[str, float] = {}
        for _step in range(int(variant["iters"])):
            optimizer.zero_grad(set_to_none=True)
            pos = torch.where(movable_t[:, None], var, parent_init)
            density = self._soft_global_density(pos, all_sizes_t, cw, ch, rows, cols, density_edges, 1.02)
            corridor = self._soft_global_corridor_blockage(pos, all_sizes_t, movable_t, corridor_plan, cw, ch)
            wl = self._soft_global_hpwl(torch.cat([pos, benchmark.port_positions.to(device=device, dtype=dtype)], dim=0), nets, span)
            disp = torch.linalg.norm((pos - parent_init) / span, dim=1)
            macro_weight = corridor_plan["macro_weight"].to(device=device, dtype=dtype)
            anchor = (disp.pow(2) * torch.clamp(macro_weight, min=0.35)).mean()
            loss = (
                float(variant["density_weight"]) * density
                + corridor
                + float(variant["hpwl_weight"]) * wl
                + float(variant["anchor_weight"]) * anchor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([var], max_norm=span * 0.12)
            optimizer.step()
            with torch.no_grad():
                var.data = torch.minimum(torch.maximum(var.data, low), high)
                delta = var.data - parent_init
                norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1.0e-9)
                scale = torch.clamp(max_disp / norm, max=1.0)
                var.data = parent_init + delta * scale
                var.data[fixed_t] = parent_init[fixed_t]
            score = float(loss.detach().cpu())
            if score < best_score:
                best_score = score
                best_pos = torch.where(movable_t[:, None], var, parent_init).detach().clone()
                last_parts = {
                    "density_loss": float(density.detach().cpu()),
                    "corridor_loss": float(corridor.detach().cpu()),
                    "wl_loss": float(wl.detach().cpu()),
                    "anchor_loss": float(anchor.detach().cpu()),
                }

        trial_np = best_pos.detach().cpu().numpy().astype(np.float64)
        hard = trial_np[:n_hard].copy()
        hard = self._repair_all_overlaps(hard, movable, sizes, half_w, half_h, cw, ch)
        hard = self._clip_hard_np(hard, benchmark)
        full = parent_full.clone()
        full[:n_hard] = torch.tensor(hard, dtype=torch.float32)
        full[n_hard:num_macros] = torch.tensor(trial_np[n_hard:num_macros], dtype=torch.float32)
        full = self._repair_soft_bounds_tensor(full, benchmark)
        full = self._repair_hard_bounds_tensor(full, benchmark)
        if benchmark.macro_fixed.any():
            full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        owner = self._owner_positions_from_placement(full, benchmark)
        final_np = full[:num_macros].numpy().astype(np.float64)
        final_density = self._estimate_density_overflow_np(final_np, benchmark)
        final_congestion = self._estimate_congestion_overflow_np(owner, edges, benchmark)
        final_blockage = float(
            self._soft_global_corridor_blockage(
                torch.tensor(final_np, dtype=dtype, device=device),
                all_sizes_t,
                movable_t,
                corridor_plan,
                cw,
                ch,
            )
            .detach()
            .cpu()
        )
        parent_blockage = float(
            self._soft_global_corridor_blockage(parent_init, all_sizes_t, movable_t, corridor_plan, cw, ch).detach().cpu()
        )
        density_delta = final_density - float(parent_density)
        congestion_delta = final_congestion - float(parent_congestion)
        blockage_delta = final_blockage - parent_blockage
        parent_exact_like = float(parent_congestion) + 0.52 * float(parent_density) + 0.16 * float(parent_blockage)
        final_exact_like = float(final_congestion) + 0.52 * float(final_density) + 0.16 * float(final_blockage)
        required_score_drop = max(0.006, 0.006 * max(parent_exact_like, 1.0e-9))
        required_cong_drop = max(0.0035, 0.004 * max(float(parent_congestion), 1.0e-9))
        density_guard = max(0.010, 0.055 * max(float(parent_density), 1.0e-9))
        if final_exact_like > parent_exact_like - required_score_drop:
            return None
        if final_congestion > float(parent_congestion) - required_cong_drop:
            return None
        if final_density > float(parent_density) + density_guard:
            return None
        candidate = (
            self._surrogate_cost(hard, edges, owner, benchmark, sizes),
            hard,
            True,
            f"soft_global_{variant['name']}_refined",
            full,
        )
        cheap_score = final_congestion + 0.35 * final_density + 0.18 * final_blockage
        record = self._soft_global_preselect_record(
            candidate,
            owner,
            edges,
            benchmark,
            objective_score=cheap_score,
            legal_disp=0.0,
        )
        log = (
            f"{variant['name']}|parent={parent_label}|class=high_congestion|corridors={corridor_plan['log']}|"
            f"density_delta={density_delta:.4f}|congestion_delta={congestion_delta:.4f}|"
            f"blockage_before={parent_blockage:.4f}|blockage_after={final_blockage:.4f}|"
            f"blockage_delta={blockage_delta:.4f}|exact_like_parent={parent_exact_like:.4f}|"
            f"exact_like_after={final_exact_like:.4f}|"
            f"macro_stats={corridor_plan.get('macro_stats', '')}|"
            f"loss_den={last_parts.get('density_loss', 0.0):.4f}|loss_corr={last_parts.get('corridor_loss', 0.0):.4f}"
        )
        return candidate, record, log

    def _soft_global_hotspot_refined_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
    ):
        profile = self._benchmark_profile(benchmark, edges)
        if not profile["large_high_congestion"] or not edges:
            return None
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3]
            and ("_legalized" in c[3] or "_refined" in c[3])
            and "_raw" not in c[3]
            and "partition_refined" not in c[3]
            and "channel_refined" not in c[3]
            and "congestion_refined" not in c[3]
            and "hotspot_refined" not in c[3]
            and c[4] is not None
        ]
        if not pool:
            return None

        base = min(pool, key=lambda c: c[0])
        _base_surrogate, base_hard, _force, _base_label, base_full = base
        assert base_full is not None
        hard = base_hard.copy()
        placement_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[: benchmark.num_macros].numpy().astype(bool)
        ports = benchmark.port_positions.numpy().astype(np.float64)
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cell_w = float(cw) / cols
        cell_h = float(ch) / rows
        span = max(float(cw), float(ch), 1.0e-9)
        pair_sep_x = (sizes[:, None, 0] + sizes[None, :, 0]) / 2.0
        pair_sep_y = (sizes[:, None, 1] + sizes[None, :, 1]) / 2.0

        def owner_array(candidate_placement: np.ndarray) -> np.ndarray:
            if ports.shape[0] == 0:
                return candidate_placement
            return np.vstack([candidate_placement, ports])

        def grid_loc(point: np.ndarray) -> Tuple[int, int]:
            r = int(np.clip(float(point[1]) / max(ch, 1.0e-9) * rows, 0, rows - 1))
            c = int(np.clip(float(point[0]) / max(cw, 1.0e-9) * cols, 0, cols - 1))
            return r, c

        def density_overflow_grid(candidate_placement: np.ndarray) -> np.ndarray:
            d_rows = min(max(10, int(benchmark.grid_rows)), 18)
            d_cols = min(max(10, int(benchmark.grid_cols)), 18)
            d_cell_w = float(cw) / d_cols
            d_cell_h = float(ch) / d_rows
            d_cell_area = max(d_cell_w * d_cell_h, 1.0e-9)
            grid = np.zeros((d_rows, d_cols), dtype=np.float64)
            xs = np.clip((candidate_placement[:, 0] / max(cw, 1.0e-9) * d_cols).astype(int), 0, d_cols - 1)
            ys = np.clip((candidate_placement[:, 1] / max(ch, 1.0e-9) * d_rows).astype(int), 0, d_rows - 1)
            area = all_sizes[:, 0] * all_sizes[:, 1]
            np.add.at(grid, (ys, xs), area)
            util = grid / d_cell_area
            target = float(
                np.clip(area.sum() / max(float(cw) * float(ch), 1.0e-9) * self.soft_global_density_target_scale, 0.58, 0.92)
            )
            return np.maximum(util - target, 0.0) ** 2

        owner_pos = owner_array(placement_np)
        h_grid, v_grid = self._routing_congestion_arrays_np(owner_pos, edges, benchmark)
        row_score = h_grid.mean(axis=1) + v_grid.mean(axis=1)
        col_score = h_grid.mean(axis=0) + v_grid.mean(axis=0)
        if not np.any(row_score) and not np.any(col_score):
            return None

        hot_row_count = max(1, min(4, rows // 10))
        hot_col_count = max(1, min(4, cols // 10))
        cold_row_count = max(2, min(6, rows // 6))
        cold_col_count = max(2, min(6, cols // 6))
        hot_rows = set(int(x) for x in np.argsort(row_score)[-hot_row_count:])
        hot_cols = set(int(x) for x in np.argsort(col_score)[-hot_col_count:])
        cold_rows = [int(x) for x in np.argsort(row_score)[:cold_row_count]]
        cold_cols = [int(x) for x in np.argsort(col_score)[:cold_col_count]]

        contrib = np.zeros(benchmark.num_macros, dtype=np.float64)
        for edge in edges:
            a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
            if a >= len(owner_pos) or b >= len(owner_pos):
                continue
            r0, c0 = grid_loc(owner_pos[a])
            r1, c1 = grid_loc(owner_pos[b])
            r_lo, r_hi = sorted((r0, r1))
            c_lo, c_hi = sorted((c0, c1))
            row_hit = any(r in hot_rows for r in range(r_lo, r_hi + 1))
            col_hit = any(c in hot_cols for c in range(c_lo, c_hi + 1))
            if not row_hit and not col_hit:
                continue
            signal = weight
            if row_hit:
                signal += 0.25 * max(float(row_score[r]) for r in range(r_lo, r_hi + 1) if r in hot_rows)
            if col_hit:
                signal += 0.25 * max(float(col_score[c]) for c in range(c_lo, c_hi + 1) if c in hot_cols)
            if 0 <= a < benchmark.num_macros and not fixed[a]:
                contrib[a] += signal
            if 0 <= b < benchmark.num_macros and not fixed[b]:
                contrib[b] += signal

        movable_ids = np.flatnonzero(contrib > 0.0)
        if movable_ids.size == 0:
            return None
        ranked = sorted((float(contrib[idx]), int(idx)) for idx in movable_ids)
        ranked.reverse()
        max_moves = max(8, min(48, int(math.ceil(0.035 * float(len(ranked))))))
        selected = ranked[:max_moves]

        current_cong = self._estimate_congestion_overflow_np(owner_pos, edges, benchmark)
        current_density = self._estimate_density_overflow_np(placement_np, benchmark)
        local_density = density_overflow_grid(placement_np)
        d_rows, d_cols = local_density.shape
        accepted = 0

        def local_density_at(point: np.ndarray, density_grid: np.ndarray) -> float:
            r = int(np.clip(float(point[1]) / max(ch, 1.0e-9) * density_grid.shape[0], 0, density_grid.shape[0] - 1))
            c = int(np.clip(float(point[0]) / max(cw, 1.0e-9) * density_grid.shape[1], 0, density_grid.shape[1] - 1))
            return float(density_grid[r, c])

        def target_point(idx: int) -> np.ndarray:
            point = placement_np[idx]
            r, c = grid_loc(point)
            target = point.copy()
            max_disp = (0.010 if idx < benchmark.num_hard_macros else 0.018) * span
            if r in hot_rows:
                best_r = min(cold_rows, key=lambda rr: abs(rr - r))
                target[1] = (float(best_r) + 0.5) * cell_h
            if c in hot_cols:
                best_c = min(cold_cols, key=lambda cc: abs(cc - c))
                target[0] = (float(best_c) + 0.5) * cell_w
            delta = target - point
            norm = float(np.linalg.norm(delta))
            if norm > max_disp:
                delta *= max_disp / max(norm, 1.0e-9)
            if norm <= 1.0e-9:
                hot_center = np.array(
                    [
                        (np.mean(list(hot_cols)) + 0.5) * cell_w if hot_cols else point[0],
                        (np.mean(list(hot_rows)) + 0.5) * cell_h if hot_rows else point[1],
                    ],
                    dtype=np.float64,
                )
                away = point - hot_center
                away_norm = float(np.linalg.norm(away))
                if away_norm > 1.0e-9:
                    delta = away / away_norm * max_disp
            return point + delta

        for _score, idx in selected:
            if idx < benchmark.num_hard_macros and not movable[idx]:
                continue
            if idx >= benchmark.num_hard_macros and fixed[idx]:
                continue
            point = placement_np[idx]
            target = target_point(idx)
            half = all_sizes[idx] * 0.5
            best_trial = None
            best_cong = current_cong
            best_density = current_density
            for scale in (0.35, 0.60, 0.85):
                trial_point = point + (target - point) * scale
                trial_point[0] = np.clip(trial_point[0], half[0], float(cw) - half[0])
                trial_point[1] = np.clip(trial_point[1], half[1], float(ch) - half[1])
                if float(np.linalg.norm(trial_point - point)) < 1.0e-7:
                    continue
                if idx < benchmark.num_hard_macros:
                    trial_hard = hard.copy()
                    trial_hard[idx] = trial_point
                    if self._any_overlap(trial_hard, [idx], pair_sep_x, pair_sep_y, gap=0.025):
                        continue
                old_local = local_density_at(point, local_density)
                new_local = local_density_at(trial_point, local_density)
                if new_local > old_local + max(0.0015, 0.04 * max(old_local, 1.0e-9)):
                    continue
                trial_np = placement_np.copy()
                trial_np[idx] = trial_point
                trial_density = self._estimate_density_overflow_np(trial_np, benchmark)
                if trial_density > current_density + max(0.0015, 0.025 * max(current_density, 1.0e-9)):
                    continue
                trial_cong = self._estimate_congestion_overflow_np(owner_array(trial_np), edges, benchmark)
                if trial_cong < best_cong - 1.0e-5:
                    best_trial = trial_point.copy()
                    best_cong = trial_cong
                    best_density = trial_density
            if best_trial is None:
                continue
            placement_np[idx] = best_trial
            if idx < benchmark.num_hard_macros:
                hard[idx] = best_trial
            current_cong = best_cong
            current_density = best_density
            local_density = density_overflow_grid(placement_np)
            accepted += 1

        if accepted == 0:
            return None
        hard = self._repair_all_overlaps(hard, movable, sizes, half_w, half_h, cw, ch)
        hard = self._clip_hard_np(hard, benchmark)
        hotspot_full = base_full.clone()
        hotspot_full[: benchmark.num_hard_macros] = torch.tensor(hard, dtype=torch.float32)
        hotspot_full[benchmark.num_hard_macros : benchmark.num_macros] = torch.tensor(
            placement_np[benchmark.num_hard_macros : benchmark.num_macros], dtype=torch.float32
        )
        hotspot_full = self._repair_soft_bounds_tensor(hotspot_full, benchmark)
        hotspot_full = self._repair_hard_bounds_tensor(hotspot_full, benchmark)
        if benchmark.macro_fixed.any():
            hotspot_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        hotspot_owner = self._owner_positions_from_placement(hotspot_full, benchmark)
        final_np = hotspot_full[: benchmark.num_macros].numpy().astype(np.float64)
        score = self._estimate_congestion_overflow_np(hotspot_owner, edges, benchmark) + 0.20 * self._estimate_density_overflow_np(
            final_np, benchmark
        )
        candidate = (
            self._surrogate_cost(hard, edges, hotspot_owner, benchmark, sizes),
            hard,
            True,
            "soft_global_hotspot_refined",
            hotspot_full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            hotspot_owner,
            edges,
            benchmark,
            objective_score=score,
            legal_disp=0.0,
        )
        return (
            candidate,
            record,
            f"hotspot_refined|accepted={accepted}|rows={len(hot_rows)}|cols={len(hot_cols)}|"
            f"cong={current_cong:.4f}|density={current_density:.4f}|score={score:.4f}",
        )

    def _soft_global_congestion_refined_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
    ):
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3] and ("_legalized" in c[3] or "_refined" in c[3]) and c[4] is not None
        ]
        if not pool or not edges:
            return None

        base = min(pool, key=lambda c: c[0])
        _base_surrogate, base_hard, _force, _base_label, base_full = base
        assert base_full is not None
        pos = base_hard.copy()
        full = base_full.clone()
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        rows = int(benchmark.grid_rows)
        cols = int(benchmark.grid_cols)
        span = max(float(cw), float(ch), 1.0e-9)
        pair_sep_x = (sizes[:, None, 0] + sizes[None, :, 0]) / 2.0
        pair_sep_y = (sizes[:, None, 1] + sizes[None, :, 1]) / 2.0

        placement_np = full[: benchmark.num_macros].numpy().astype(np.float64)

        def score_placement(candidate_placement: np.ndarray) -> float:
            if benchmark.port_positions.shape[0] == 0:
                owner = candidate_placement
            else:
                owner = np.vstack([candidate_placement, benchmark.port_positions.numpy().astype(np.float64)])
            return self._estimate_congestion_overflow_np(owner, edges, benchmark) + 0.20 * self._estimate_density_overflow_np(
                candidate_placement, benchmark
            )

        def score(candidate_hard: np.ndarray) -> float:
            trial = placement_np.copy()
            trial[: benchmark.num_hard_macros] = candidate_hard
            return score_placement(trial)

        current_score = score_placement(placement_np)
        owner_pos = (
            placement_np
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        )
        grid = self._congestion_grid_np(owner_pos, edges, benchmark, rows, cols)
        active = grid[grid > 0.0]
        if active.size == 0:
            return None
        threshold = float(np.percentile(active, 88))
        cell_w = float(cw) / cols
        cell_h = float(ch) / rows
        hot_cells = np.argwhere(grid >= threshold)
        if hot_cells.size == 0:
            return None
        hot_centers = np.stack(
            [(hot_cells[:, 1] + 0.5) * cell_w, (hot_cells[:, 0] + 0.5) * cell_h],
            axis=1,
        )
        hot_weights = grid[hot_cells[:, 0], hot_cells[:, 1]]
        hot_center = np.average(hot_centers, axis=0, weights=np.maximum(hot_weights, 1.0e-9))

        macro_scores = []
        for idx in range(benchmark.num_hard_macros):
            if not movable[idx]:
                continue
            c = int(np.clip(pos[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            r = int(np.clip(pos[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            if grid[r, c] >= threshold:
                macro_scores.append((float(grid[r, c]), idx))
        macro_scores.sort(reverse=True)
        selected = [idx for _value, idx in macro_scores[: min(18, len(macro_scores))]]
        if not selected:
            return None

        base_directions = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / math.sqrt(2.0),
            np.array([1.0, -1.0]) / math.sqrt(2.0),
            np.array([-1.0, 1.0]) / math.sqrt(2.0),
            np.array([-1.0, -1.0]) / math.sqrt(2.0),
        ]

        accepted = 0
        for idx in selected:
            old_c = int(np.clip(pos[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            old_r = int(np.clip(pos[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            old_hot = float(grid[old_r, old_c])
            away = pos[idx] - hot_center
            norm = float(np.linalg.norm(away))
            directions = list(base_directions)
            if norm > 1.0e-9:
                directions = [away / norm] + directions
            best_pos = pos[idx].copy()
            best_delta = 0.0
            for step_scale in (0.010, 0.022):
                step = span * step_scale
                for direction in directions:
                    trial = pos.copy()
                    trial[idx, 0] = np.clip(pos[idx, 0] + direction[0] * step, half_w[idx], cw - half_w[idx])
                    trial[idx, 1] = np.clip(pos[idx, 1] + direction[1] * step, half_h[idx], ch - half_h[idx])
                    if self._any_overlap(trial, [idx], pair_sep_x, pair_sep_y, gap=0.025):
                        continue
                    new_c = int(np.clip(trial[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
                    new_r = int(np.clip(trial[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
                    move_cost = 0.04 * float(np.linalg.norm(trial[idx] - pos[idx])) / span
                    delta = float(grid[new_r, new_c]) - old_hot + move_cost
                    if delta < best_delta - 1.0e-9:
                        best_delta = delta
                        best_pos = trial[idx].copy()
            if best_delta < -1.0e-9:
                pos[idx] = best_pos
                current_score += best_delta
                accepted += 1

        placement_np[: benchmark.num_hard_macros] = pos
        soft_scores = []
        for idx in range(benchmark.num_hard_macros, benchmark.num_macros):
            if bool(benchmark.macro_fixed[idx]):
                continue
            c = int(np.clip(placement_np[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            r = int(np.clip(placement_np[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            if grid[r, c] >= threshold:
                soft_scores.append((float(grid[r, c]), idx))
        soft_scores.sort(reverse=True)
        soft_selected = [idx for _value, idx in soft_scores[: min(48, len(soft_scores))]]
        for idx in soft_selected:
            old_c = int(np.clip(placement_np[idx, 0] / max(cw, 1.0e-9) * cols, 0, cols - 1))
            old_r = int(np.clip(placement_np[idx, 1] / max(ch, 1.0e-9) * rows, 0, rows - 1))
            old_hot = float(grid[old_r, old_c])
            away = placement_np[idx] - hot_center
            norm = float(np.linalg.norm(away))
            directions = list(base_directions)
            if norm > 1.0e-9:
                directions = [away / norm] + directions
            best_pos = placement_np[idx].copy()
            best_delta = 0.0
            for step_scale in (0.025, 0.055):
                step = span * step_scale
                for direction in directions[:5]:
                    half = all_sizes[idx]
                    new_x = np.clip(
                        placement_np[idx, 0] + direction[0] * step,
                        half[0] / 2.0,
                        cw - half[0] / 2.0,
                    )
                    new_y = np.clip(
                        placement_np[idx, 1] + direction[1] * step,
                        half[1] / 2.0,
                        ch - half[1] / 2.0,
                    )
                    new_c = int(np.clip(new_x / max(cw, 1.0e-9) * cols, 0, cols - 1))
                    new_r = int(np.clip(new_y / max(ch, 1.0e-9) * rows, 0, rows - 1))
                    move_cost = 0.02 * math.hypot(float(new_x - placement_np[idx, 0]), float(new_y - placement_np[idx, 1])) / span
                    delta = float(grid[new_r, new_c]) - old_hot + move_cost
                    if delta < best_delta - 1.0e-9:
                        best_delta = delta
                        best_pos = np.array([new_x, new_y], dtype=np.float64)
            if best_delta < -1.0e-9:
                placement_np[idx] = best_pos
                current_score += best_delta
                accepted += 1

        if accepted == 0:
            return None
        current_score = score_placement(placement_np)
        pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
        pos = self._clip_hard_np(pos, benchmark)
        refined_full = base_full.clone()
        refined_full[: benchmark.num_hard_macros] = torch.tensor(pos, dtype=torch.float32)
        refined_full[benchmark.num_hard_macros : benchmark.num_macros] = torch.tensor(
            placement_np[benchmark.num_hard_macros : benchmark.num_macros], dtype=torch.float32
        )
        refined_full = self._repair_hard_bounds_tensor(refined_full, benchmark)
        refined_owner = self._owner_positions_from_placement(refined_full, benchmark)
        candidate = (
            self._surrogate_cost(pos, edges, refined_owner, benchmark, sizes),
            pos,
            True,
            "soft_global_congestion_refined",
            refined_full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            refined_owner,
            edges,
            benchmark,
            objective_score=current_score,
            legal_disp=0.0,
        )
        return candidate, record, f"congestion_refined|accepted={accepted}|score={current_score:.4f}"

    def _soft_global_congestion_flow_drag_candidates(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
    ) -> List[Tuple[Candidate, Dict[str, object], str]]:
        if os.environ.get("JIHO_BASIN_ESCAPE_FLOW_DRAG", "0") != "1":
            self.flow_drag_log = "disabled"
            return []
        profile = self._benchmark_profile(benchmark, edges)
        if not bool(profile["large_high_congestion"]) and not self._is_large_soft_global_design(benchmark):
            self.flow_drag_log = "skipped=not_large_or_high_congestion"
            return []
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3] and ("_legalized" in c[3] or "_refined" in c[3]) and c[4] is not None
        ]
        if not pool or not edges:
            self.flow_drag_log = "skipped=no_parent_or_edges"
            return []

        def parent_rank(candidate: Candidate) -> Tuple[int, float]:
            label = candidate[3]
            if "congestion_refined" in label:
                priority = 0
            elif "hotspot_refined" in label:
                priority = 1
            elif "spread_cong" in label and "_refined" in label:
                priority = 2
            elif "spread_cong" in label:
                priority = 3
            else:
                priority = 4
            return priority, float(candidate[0])

        base = min(pool, key=parent_rank)
        _base_surrogate, base_hard, _force, base_label, base_full = base
        assert base_full is not None
        base_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        base_density = self._estimate_density_overflow_np(base_np, benchmark)
        base_owner = self._owner_positions_from_placement(base_full, benchmark)
        base_cheap_cong = self._estimate_congestion_overflow_np(base_owner, edges, benchmark)
        base_style_cong = self._estimate_exact_style_congestion_overflow_np(base_np, benchmark)

        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cell_w = float(cw) / max(cols, 1)
        cell_h = float(ch) / max(rows, 1)
        span = max(float(cw), float(ch), 1.0e-9)
        grid = self._exact_style_congestion_map_np(base_np, benchmark)
        if grid.size == 0 or not np.any(grid > 0.0):
            self.flow_drag_log = "skipped=empty_congestion_map"
            return []
        active = grid[grid > 0.0]
        hot_threshold = float(np.percentile(active, 82)) if active.size else float(np.max(grid))
        hot_cells = np.argwhere(grid >= hot_threshold)
        hot_center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
        if hot_cells.size:
            hot_weights = np.maximum(grid[hot_cells[:, 0], hot_cells[:, 1]], 1.0e-9)
            hot_centers = np.stack(
                [(hot_cells[:, 1] + 0.5) * cell_w, (hot_cells[:, 0] + 0.5) * cell_h],
                axis=1,
            )
            hot_center = np.average(hot_centers, axis=0, weights=hot_weights)

        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        all_sizes = benchmark.macro_sizes[:n_macros].numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_macros].numpy().astype(bool)
        degree = np.zeros(n_macros, dtype=np.float64)
        adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_macros)]
        for a_raw, b_raw, weight_raw in edges:
            a, b, weight = int(a_raw), int(b_raw), float(weight_raw)
            if 0 <= a < n_macros:
                degree[a] += weight
            if 0 <= b < n_macros:
                degree[b] += weight
            if 0 <= a < n_macros and 0 <= b < n_macros and a != b:
                adjacency[a].append((b, weight))
                adjacency[b].append((a, weight))
        for nbrs in adjacency:
            nbrs.sort(key=lambda item: item[1], reverse=True)

        areas = np.maximum(all_sizes[:n_hard, 0] * all_sizes[:n_hard, 1], 1.0e-9)
        area_norm = areas / max(float(np.mean(areas)), 1.0e-9)
        degree_norm = degree[:n_hard] / max(float(np.percentile(degree[:n_hard][degree[:n_hard] > 0.0], 90)) if np.any(degree[:n_hard] > 0.0) else 1.0, 1.0e-9)
        hot_scores: List[Tuple[float, int]] = []
        for idx in range(n_hard):
            if not bool(movable[idx]) or bool(fixed[idx]):
                continue
            r, c = self._flow_drag_cell(base_np[idx], cw, ch, rows, cols)
            local_hot = float(grid[r, c])
            if local_hot <= 0.0:
                continue
            score = local_hot * math.sqrt(float(area_norm[idx])) * (0.75 + 0.35 * min(float(degree_norm[idx]), 3.0))
            hot_scores.append((score, idx))
        hot_scores.sort(reverse=True)
        topk = max(1, int(os.environ.get("JIHO_FLOW_DRAG_TOPK", "8")))
        hot_ids = [idx for _score, idx in hot_scores[:topk]]
        if not hot_ids:
            self.flow_drag_log = "skipped=no_hot_macros"
            return []

        steps = self._parse_float_list_env("JIHO_FLOW_DRAG_STEPS", (0.04, 0.08, 0.12))
        max_disp = max(0.0, float(os.environ.get("JIHO_FLOW_DRAG_MAX_DISP", "0.20"))) * span
        neighbor_count = max(0, int(os.environ.get("JIHO_FLOW_DRAG_NEIGHBORS", "3")))
        neighbor_beta = float(os.environ.get("JIHO_FLOW_DRAG_NEIGHBOR_BETA", "0.45"))
        max_candidates = max(1, int(os.environ.get("JIHO_FLOW_DRAG_MAX_CANDIDATES", "6")))
        require_cong_improve = os.environ.get("JIHO_FLOW_DRAG_REQUIRE_CHEAP_CONG_IMPROVE", "0") == "1"
        max_density_worsen = float(os.environ.get("JIHO_FLOW_DRAG_MAX_DENSITY_WORSEN", "0.10"))
        min_mean_disp = float(os.environ.get("JIHO_FLOW_DRAG_MIN_MEAN_DISP", "0.002"))
        pair_sep_x = (sizes[:, None, 0] + sizes[None, :, 0]) / 2.0
        pair_sep_y = (sizes[:, None, 1] + sizes[None, :, 1]) / 2.0

        group_specs: List[Tuple[str, List[int], float]] = []
        if hot_ids:
            group_specs.append(("single", [hot_ids[0]], steps[0]))
            if len(steps) > 1:
                group_specs.append(("single_mid", [hot_ids[0]], steps[1]))
        if len(hot_ids) >= 2:
            group_specs.append(("top2", hot_ids[:2], steps[min(1, len(steps) - 1)]))
        if len(hot_ids) >= 3:
            group_specs.append(("top3", hot_ids[:3], steps[min(2, len(steps) - 1)]))
        if len(hot_ids) >= 4:
            group_specs.append(("all_hot", hot_ids, steps[0]))
        for extra_id in hot_ids[1:]:
            if len(group_specs) >= max_candidates:
                break
            group_specs.append((f"single_{extra_id}", [extra_id], steps[0]))
        group_specs = group_specs[:max_candidates]

        results: List[Tuple[Candidate, Dict[str, object], str]] = []
        hot_log = ",".join(f"{idx}:{score:.4f}" for score, idx in hot_scores[: min(16, len(hot_scores))])
        logs: List[str] = [
            "enabled|"
            f"parent={base_label}|base_density={base_density:.4f}|base_cheap_cong={base_cheap_cong:.4f}|"
            f"base_style_cong={base_style_cong:.4f}|hot={hot_log}"
        ]
        for spec_id, (variant, group, step_frac) in enumerate(group_specs):
            trial_np = base_np.copy()
            trial_hard = base_hard.copy()
            moved: Dict[int, np.ndarray] = {}
            for hot_idx in group:
                direction = self._flow_drag_vector(base_np[hot_idx], grid, hot_center, cw, ch, rows, cols)
                step = min(float(step_frac) * span, max_disp)
                if step <= 0.0:
                    continue
                moved[hot_idx] = moved.get(hot_idx, np.zeros(2, dtype=np.float64)) + direction * step
                nbrs = [item for item in adjacency[hot_idx] if 0 <= int(item[0]) < n_macros and not fixed[int(item[0])]]
                if neighbor_count > 0 and nbrs:
                    max_weight = max(float(w) for _nbr, w in nbrs[:neighbor_count])
                    for nbr, weight in nbrs[:neighbor_count]:
                        nbr = int(nbr)
                        if nbr < n_hard and not bool(movable[nbr]):
                            continue
                        weight_scale = float(weight) / max(max_weight, 1.0e-9)
                        moved[nbr] = moved.get(nbr, np.zeros(2, dtype=np.float64)) + direction * step * neighbor_beta * weight_scale
            if not moved:
                logs.append(f"{variant}|drop=zero_move")
                continue
            for idx, delta in moved.items():
                disp_norm = float(np.linalg.norm(delta))
                if disp_norm > max_disp:
                    delta = delta * (max_disp / max(disp_norm, 1.0e-9))
                half = all_sizes[idx] * 0.5
                new_pos = trial_np[idx] + delta
                new_pos[0] = np.clip(new_pos[0], half[0], cw - half[0])
                new_pos[1] = np.clip(new_pos[1], half[1], ch - half[1])
                trial_np[idx] = new_pos
                if idx < n_hard:
                    trial_hard[idx] = new_pos
            pre_repair_np = trial_np.copy()
            pre_disp = np.linalg.norm(pre_repair_np[:n_macros] - base_np[:n_macros], axis=1)
            pre_mean_disp_norm = float(np.mean(pre_disp) / span) if pre_disp.size else 0.0
            touched_hard = [idx for idx in moved if idx < n_hard]
            if touched_hard and self._any_overlap(trial_hard, touched_hard, pair_sep_x, pair_sep_y, gap=0.0):
                trial_hard = self._repair_all_overlaps(trial_hard, movable, sizes, half_w, half_h, cw, ch)
                trial_hard = self._clip_hard_np(trial_hard, benchmark)
                trial_np[:n_hard] = trial_hard
            disp = np.linalg.norm(trial_np[:n_macros] - base_np[:n_macros], axis=1)
            mean_disp_norm = float(np.mean(disp) / span) if disp.size else 0.0
            hard_disp = disp[:n_hard]
            hard_mean_disp_norm = float(np.mean(hard_disp) / span) if hard_disp.size else 0.0
            max_disp_norm = float(np.max(disp) / span) if disp.size else 0.0
            variant_prefix = (
                f"{variant}|group={','.join(str(x) for x in group)}|step={float(step_frac):.4f}|moved={len(moved)}|"
                f"pre_mean={pre_mean_disp_norm:.5f}|post_mean={mean_disp_norm:.5f}|"
                f"hard_mean={hard_mean_disp_norm:.5f}|post_max={max_disp_norm:.5f}|"
                f"density_before={base_density:.4f}|style_before={base_style_cong:.4f}|cheap_before={base_cheap_cong:.4f}"
            )
            if mean_disp_norm < min_mean_disp:
                logs.append(f"{variant_prefix}|drop=near_zero")
                continue
            density = self._estimate_density_overflow_np(trial_np, benchmark)
            density_worsen = density - base_density
            if density_worsen > max_density_worsen:
                logs.append(f"{variant_prefix}|density_after={density:.4f}|drop=density|dd={density_worsen:.4f}")
                continue
            style_congestion = self._estimate_exact_style_congestion_overflow_np(trial_np, benchmark)
            trial_owner_pre = (
                trial_np
                if benchmark.port_positions.shape[0] == 0
                else np.vstack([trial_np, benchmark.port_positions.numpy().astype(np.float64)])
            )
            cheap_congestion = self._estimate_congestion_overflow_np(trial_owner_pre, edges, benchmark)
            if require_cong_improve and style_congestion > base_style_cong - 1.0e-6:
                logs.append(
                    f"{variant_prefix}|density_after={density:.4f}|style_after={style_congestion:.4f}|"
                    f"cheap_after={cheap_congestion:.4f}|drop=cong"
                )
                continue
            trial_full = base_full.clone()
            trial_full[:n_macros] = torch.tensor(trial_np, dtype=torch.float32)
            trial_full = self._repair_soft_bounds_tensor(trial_full, benchmark)
            trial_full = self._repair_hard_bounds_tensor(trial_full, benchmark)
            if benchmark.macro_fixed.any():
                trial_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            final_np = trial_full[:n_macros].numpy().astype(np.float64)
            final_hard = final_np[:n_hard].copy()
            owner = self._owner_positions_from_placement(trial_full, benchmark)
            label = "soft_global_congestion_flow_drag_refined" if not results else f"soft_global_congestion_flow_drag_refined_{spec_id}"
            final_density = self._estimate_density_overflow_np(final_np, benchmark)
            final_style_congestion = self._estimate_exact_style_congestion_overflow_np(final_np, benchmark)
            final_cheap_congestion = self._estimate_congestion_overflow_np(owner, edges, benchmark)
            final_disp = np.linalg.norm(final_np[:n_macros] - base_np[:n_macros], axis=1)
            final_mean_disp_norm = float(np.mean(final_disp) / span) if final_disp.size else 0.0
            final_hard_disp = final_disp[:n_hard]
            final_hard_mean_disp_norm = float(np.mean(final_hard_disp) / span) if final_hard_disp.size else 0.0
            final_max_disp_norm = float(np.max(final_disp) / span) if final_disp.size else 0.0
            cheap_score = final_style_congestion + 0.20 * final_density
            candidate = (
                self._surrogate_cost(final_hard, edges, owner, benchmark, sizes),
                final_hard,
                True,
                label,
                trial_full,
            )
            record = self._soft_global_preselect_record(
                candidate,
                owner,
                edges,
                benchmark,
                objective_score=cheap_score,
                legal_disp=final_mean_disp_norm,
            )
            results.append(
                (
                    candidate,
                    record,
                    "flow_drag|"
                    f"variant={variant}|group={','.join(str(x) for x in group)}|step={float(step_frac):.4f}|"
                    f"moved={len(moved)}|pre_mean={pre_mean_disp_norm:.5f}|post_mean={final_mean_disp_norm:.5f}|"
                    f"hard_mean={final_hard_mean_disp_norm:.5f}|post_max={final_max_disp_norm:.5f}|"
                    f"density_before={base_density:.4f}|density_after={final_density:.4f}|"
                    f"style_before={base_style_cong:.4f}|style_after={final_style_congestion:.4f}|"
                    f"cheap_before={base_cheap_cong:.4f}|cheap_after={final_cheap_congestion:.4f}|score={cheap_score:.4f}",
                )
            )
            logs.append(
                f"{variant_prefix}|density_after={final_density:.4f}|style_after={final_style_congestion:.4f}|"
                f"cheap_after={final_cheap_congestion:.4f}|final_mean={final_mean_disp_norm:.5f}|"
                f"final_hard_mean={final_hard_mean_disp_norm:.5f}|final_max={final_max_disp_norm:.5f}|keep"
            )
            if len(results) >= max_candidates:
                break
        self.flow_drag_log = ";".join(logs[:24])
        return results

    def _parse_float_list_env(self, name: str, default: Sequence[float]) -> Tuple[float, ...]:
        raw = os.environ.get(name)
        if not raw:
            return tuple(float(x) for x in default)
        values: List[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                continue
        return tuple(values) if values else tuple(float(x) for x in default)

    def _flow_drag_cell(
        self,
        point: np.ndarray,
        cw: float,
        ch: float,
        rows: int,
        cols: int,
    ) -> Tuple[int, int]:
        r = int(np.clip(float(point[1]) / max(float(ch), 1.0e-9) * int(rows), 0, int(rows) - 1))
        c = int(np.clip(float(point[0]) / max(float(cw), 1.0e-9) * int(cols), 0, int(cols) - 1))
        return r, c

    def _flow_drag_vector(
        self,
        point: np.ndarray,
        grid: np.ndarray,
        hot_center: np.ndarray,
        cw: float,
        ch: float,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        r, c = self._flow_drag_cell(point, cw, ch, rows, cols)
        left = float(grid[r, max(0, c - 1)])
        right = float(grid[r, min(cols - 1, c + 1)])
        down = float(grid[max(0, r - 1), c])
        up = float(grid[min(rows - 1, r + 1), c])
        grad = np.array([(right - left) / max(float(cw) / max(cols, 1), 1.0e-9), (up - down) / max(float(ch) / max(rows, 1), 1.0e-9)], dtype=np.float64)
        direction = -grad
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.asarray(point, dtype=np.float64) - np.asarray(hot_center, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        return direction / norm

    def _random_basin_probe_candidates(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
    ) -> List[Candidate]:
        if os.environ.get("JIHO_RANDOM_BASIN_PROBE", "0") != "1":
            self.random_basin_log = "disabled"
            return []
        if not candidates or not edges:
            self.random_basin_log = "skipped=no_candidates_or_edges"
            return []

        pool = [c for c in candidates if c[4] is not None]
        if not pool:
            pool = candidates

        def parent_rank(candidate: Candidate) -> Tuple[int, float]:
            label = candidate[3]
            if "congestion_refined" in label:
                priority = 0
            elif "hotspot_refined" in label:
                priority = 1
            elif "spread_cong" in label and "_refined" in label:
                priority = 2
            elif "soft_global_topo_" in label:
                priority = 3
            elif "soft_global" in label:
                priority = 4
            else:
                priority = 5
            return priority, float(candidate[0])

        base = min(pool, key=parent_rank)
        _base_surrogate, base_hard, _force, base_label, base_full = base
        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        span = max(float(cw), float(ch), 1.0e-9)

        if base_full is None:
            base_full = benchmark.macro_positions.clone()
            base_full[:n_hard] = torch.tensor(base_hard, dtype=torch.float32)
            base_full = self._repair_hard_bounds_tensor(base_full, benchmark)
            base_full = self._repair_soft_bounds_tensor(base_full, benchmark)
            if benchmark.macro_fixed.any():
                base_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        base_np = base_full[:n_macros].detach().cpu().numpy().astype(np.float64)
        base_density = self._estimate_density_overflow_np(base_np, benchmark)
        base_owner = self._owner_positions_from_placement(base_full, benchmark)
        base_cheap_cong = self._estimate_congestion_overflow_np(base_owner, edges, benchmark)
        base_style_cong = self._estimate_exact_style_congestion_overflow_np(base_np, benchmark)

        grid = self._exact_style_congestion_map_np(base_np, benchmark)
        if grid.size == 0 or not np.any(grid > 0.0):
            self.random_basin_log = f"skipped=empty_congestion_map|parent={base_label}"
            return []
        rows, cols = grid.shape
        cell_w = float(cw) / max(int(cols), 1)
        cell_h = float(ch) / max(int(rows), 1)
        hot_r, hot_c = np.unravel_index(int(np.argmax(grid)), grid.shape)
        hot_center = np.array([(hot_c + 0.5) * cell_w, (hot_r + 0.5) * cell_h], dtype=np.float64)

        fixed = benchmark.macro_fixed[:n_macros].detach().cpu().numpy().astype(bool)
        all_sizes = benchmark.macro_sizes[:n_macros].detach().cpu().numpy().astype(np.float64)
        degree = np.zeros(n_macros, dtype=np.float64)
        for a_raw, b_raw, weight_raw in edges:
            a, b, weight = int(a_raw), int(b_raw), float(weight_raw)
            if 0 <= a < n_macros:
                degree[a] += weight
            if 0 <= b < n_macros:
                degree[b] += weight

        areas = np.maximum(all_sizes[:n_hard, 0] * all_sizes[:n_hard, 1], 1.0e-9)
        area_norm = areas / max(float(np.mean(areas)), 1.0e-9)
        positive_degree = degree[:n_hard][degree[:n_hard] > 0.0]
        degree_scale = max(float(np.percentile(positive_degree, 90)) if positive_degree.size else 1.0, 1.0e-9)
        hot_scores: List[Tuple[float, int, int, int]] = []
        for idx in range(n_hard):
            if not bool(movable[idx]) or bool(fixed[idx]):
                continue
            r, c = self._flow_drag_cell(base_np[idx], cw, ch, rows, cols)
            local = float(grid[r, c])
            if local <= 0.0:
                continue
            cell_dist = math.hypot(float(r - hot_r), float(c - hot_c))
            degree_boost = 0.75 + 0.35 * min(float(degree[idx]) / degree_scale, 3.0)
            score = local * math.sqrt(float(area_norm[idx])) * degree_boost / (1.0 + 0.08 * cell_dist)
            hot_scores.append((score, idx, r, c))
        hot_scores.sort(reverse=True)

        macro_count = max(1, int(os.environ.get("JIHO_RANDOM_BASIN_MACROS", "6")))
        count = max(1, int(os.environ.get("JIHO_RANDOM_BASIN_COUNT", "12")))
        radius = max(0.0, float(os.environ.get("JIHO_RANDOM_BASIN_RADIUS", "0.18"))) * span
        if not hot_scores or radius <= 0.0:
            self.random_basin_log = f"skipped=no_hot_macros|parent={base_label}|radius={radius / span:.4f}"
            return []

        hot_pool = [idx for _score, idx, _r, _c in hot_scores[: max(macro_count * 3, macro_count)]]
        rng = np.random.default_rng(int(self.base_seed) + 73021)
        pair_sep_x = (sizes[:, None, 0] + sizes[None, :, 0]) / 2.0
        pair_sep_y = (sizes[:, None, 1] + sizes[None, :, 1]) / 2.0
        logs: List[str] = [
            "enabled|"
            f"parent={base_label}|base_density={base_density:.4f}|base_cheap_cong={base_cheap_cong:.4f}|"
            f"base_style_cong={base_style_cong:.4f}|hot_cell={hot_r},{hot_c}:{float(grid[hot_r, hot_c]):.4f}|"
            f"selected={','.join(str(idx) for idx in hot_pool[:macro_count])}|"
            f"hot_scores={','.join(f'{idx}:{score:.4f}' for score, idx, _r, _c in hot_scores[:min(16, len(hot_scores))])}"
        ]
        results: List[Candidate] = []

        for probe_id in range(count):
            if len(hot_pool) <= macro_count:
                group = list(hot_pool[:macro_count])
            else:
                group = list(rng.choice(hot_pool, size=macro_count, replace=False).astype(int))
                if probe_id == 0 and hot_pool[0] not in group:
                    group[0] = hot_pool[0]
            group = sorted(set(int(x) for x in group if 0 <= int(x) < n_hard and bool(movable[int(x)]) and not bool(fixed[int(x)])))
            if not group:
                logs.append(f"probe={probe_id}|drop=empty_group")
                continue

            trial_np = base_np.copy()
            mode = "shuffle" if len(group) >= 2 and probe_id % 3 == 0 else "nearby"
            original_positions = trial_np[group].copy()
            if mode == "shuffle":
                perm = rng.permutation(len(group))
                shuffled = original_positions[perm]
                for idx, new_pos in zip(group, shuffled):
                    trial_np[idx, 0] = np.clip(float(new_pos[0]), float(half_w[idx]), float(cw - half_w[idx]))
                    trial_np[idx, 1] = np.clip(float(new_pos[1]), float(half_h[idx]), float(ch - half_h[idx]))
            else:
                for idx in group:
                    current = trial_np[idx].copy()
                    best_pos = current.copy()
                    best_heat = float("inf")
                    for _attempt in range(10):
                        theta = float(rng.uniform(0.0, math.tau))
                        dist = float(rng.uniform(0.25, 1.0)) * radius
                        if rng.random() < 0.45:
                            away = current - hot_center
                            norm = float(np.linalg.norm(away))
                            if norm > 1.0e-9:
                                direction = away / norm
                            else:
                                direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
                        else:
                            direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
                        candidate_pos = current + direction * dist
                        candidate_pos[0] = np.clip(candidate_pos[0], half_w[idx], cw - half_w[idx])
                        candidate_pos[1] = np.clip(candidate_pos[1], half_h[idx], ch - half_h[idx])
                        rr, cc = self._flow_drag_cell(candidate_pos, cw, ch, rows, cols)
                        heat = float(grid[rr, cc]) + 0.015 * float(np.linalg.norm(candidate_pos - current)) / span
                        if heat < best_heat:
                            best_heat = heat
                            best_pos = candidate_pos
                    trial_np[idx] = best_pos

            pre_repair_np = trial_np.copy()
            pre_disp = np.linalg.norm(pre_repair_np[:n_macros] - base_np[:n_macros], axis=1)
            pre_mean = float(np.mean(pre_disp) / span) if pre_disp.size else 0.0
            trial_hard = trial_np[:n_hard].copy()
            if self._any_overlap(trial_hard, group, pair_sep_x, pair_sep_y, gap=0.0):
                trial_hard = self._repair_all_overlaps(trial_hard, movable, sizes, half_w, half_h, cw, ch)
                trial_hard = self._clip_hard_np(trial_hard, benchmark)
                trial_np[:n_hard] = trial_hard

            trial_full = base_full.clone()
            trial_full[:n_macros] = torch.tensor(trial_np, dtype=torch.float32)
            trial_full = self._repair_soft_bounds_tensor(trial_full, benchmark)
            trial_full = self._repair_hard_bounds_tensor(trial_full, benchmark)
            if benchmark.macro_fixed.any():
                trial_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            final_np = trial_full[:n_macros].detach().cpu().numpy().astype(np.float64)
            final_hard = final_np[:n_hard].copy()
            final_disp = np.linalg.norm(final_np[:n_macros] - base_np[:n_macros], axis=1)
            final_mean = float(np.mean(final_disp) / span) if final_disp.size else 0.0
            hard_mean = float(np.mean(final_disp[:n_hard]) / span) if n_hard else 0.0
            final_max = float(np.max(final_disp) / span) if final_disp.size else 0.0
            legal_disp = float(np.mean(np.linalg.norm(final_np[:n_macros] - pre_repair_np[:n_macros], axis=1)) / span)
            density_after = self._estimate_density_overflow_np(final_np, benchmark)
            style_after = self._estimate_exact_style_congestion_overflow_np(final_np, benchmark)
            owner = self._owner_positions_from_placement(trial_full, benchmark)
            cheap_after = self._estimate_congestion_overflow_np(owner, edges, benchmark)
            label = "random_basin_probe" if not results else f"random_basin_probe_{probe_id:02d}"
            surrogate = self._surrogate_cost(final_hard, edges, owner, benchmark, sizes)
            candidate = (surrogate, final_hard, True, label, trial_full)
            results.append(candidate)

            orig_token = "/".join(f"{idx}:{original_positions[k,0]:.3f},{original_positions[k,1]:.3f}" for k, idx in enumerate(group[:8]))
            final_token = "/".join(f"{idx}:{final_np[idx,0]:.3f},{final_np[idx,1]:.3f}" for idx in group[:8])
            logs.append(
                f"{label}|mode={mode}|group={','.join(str(x) for x in group)}|radius={radius / span:.4f}|"
                f"orig={orig_token}|pert={final_token}|pre_mean={pre_mean:.5f}|post_mean={final_mean:.5f}|"
                f"hard_mean={hard_mean:.5f}|post_max={final_max:.5f}|legal_disp={legal_disp:.5f}|"
                f"density_before={base_density:.4f}|density_after={density_after:.4f}|"
                f"style_before={base_style_cong:.4f}|style_after={style_after:.4f}|"
                f"cheap_before={base_cheap_cong:.4f}|cheap_after={cheap_after:.4f}|keep=generated"
            )

        if not results:
            logs.append("skipped=no_generated_candidates")
        self.random_basin_log = ";".join(logs[: max(24, min(48, count + 4))])
        return results

    def _soft_global_density_spread_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        sizes: np.ndarray,
        edges: List[Edge],
    ):
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3] and ("_legalized" in c[3] or "_refined" in c[3]) and c[4] is not None
        ]
        if not pool:
            return None
        base = min(pool, key=lambda c: c[0])
        _base_surrogate, base_hard, _force, _base_label, base_full = base
        assert base_full is not None

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
        base_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)

        def score_placement(candidate_placement: np.ndarray) -> float:
            if benchmark.port_positions.shape[0] == 0:
                owner = candidate_placement
            else:
                owner = np.vstack([candidate_placement, benchmark.port_positions.numpy().astype(np.float64)])
            return (
                0.55 * self._estimate_density_overflow_np(candidate_placement, benchmark)
                + 0.50 * self._estimate_congestion_overflow_np(owner, edges, benchmark)
            )

        best_np = base_np.copy()
        best_score = score_placement(best_np)
        fallback_np = None
        fallback_density = float("inf")
        fallback_scale = 1.0
        best_scale = 1.0
        for scale in (0.94, 1.06, 1.12, 1.20, 1.30, 1.42):
            trial = base_np.copy()
            soft = trial[benchmark.num_hard_macros : benchmark.num_macros]
            if len(soft):
                moved = center + (soft - center) * scale
                soft_sizes = all_sizes[benchmark.num_hard_macros : benchmark.num_macros]
                moved[:, 0] = np.clip(moved[:, 0], soft_sizes[:, 0] / 2.0, cw - soft_sizes[:, 0] / 2.0)
                moved[:, 1] = np.clip(moved[:, 1], soft_sizes[:, 1] / 2.0, ch - soft_sizes[:, 1] / 2.0)
                trial[benchmark.num_hard_macros : benchmark.num_macros] = moved
            trial_score = score_placement(trial)
            trial_density = self._estimate_density_overflow_np(trial, benchmark)
            if scale != 1.0 and trial_density < fallback_density - 1.0e-9:
                fallback_density = trial_density
                fallback_np = trial.copy()
                fallback_scale = scale
            if trial_score < best_score - 1.0e-7:
                best_score = trial_score
                best_np = trial
                best_scale = scale

        if np.allclose(best_np, base_np, rtol=0.0, atol=1.0e-9):
            if (not self._benchmark_profile(benchmark, edges)["large_high_congestion"]) or fallback_np is None:
                return None
            best_np = fallback_np
            best_score = score_placement(best_np)
            best_scale = fallback_scale
        spread_full = base_full.clone()
        spread_full[: benchmark.num_macros] = torch.tensor(best_np, dtype=torch.float32)
        spread_full = self._repair_soft_bounds_tensor(spread_full, benchmark)
        spread_full = self._repair_hard_bounds_tensor(spread_full, benchmark)
        if benchmark.macro_fixed.any():
            spread_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        spread_owner = self._owner_positions_from_placement(spread_full, benchmark)
        hard = spread_full[: benchmark.num_hard_macros].numpy().astype(np.float64)
        candidate = (
            self._surrogate_cost(hard, edges, spread_owner, benchmark, sizes),
            hard,
            True,
            "soft_global_density_spread_refined",
            spread_full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            spread_owner,
            edges,
            benchmark,
            objective_score=best_score,
            legal_disp=0.0,
        )
        return candidate, record, f"density_spread|scale={best_scale:.2f}|score={best_score:.4f}"

    def _soft_global_axis_spread_candidate(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        sizes: np.ndarray,
        edges: List[Edge],
    ):
        if not self._benchmark_profile(benchmark, edges)["large_high_congestion"]:
            return None
        pool = [
            c
            for c in candidates
            if "soft_global" in c[3] and ("_legalized" in c[3] or "_refined" in c[3]) and c[4] is not None
        ]
        if not pool:
            return None
        base = min(pool, key=lambda c: c[0])
        _base_surrogate, _base_hard, _force, _base_label, base_full = base
        assert base_full is not None

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        center = np.array([cw * 0.5, ch * 0.5], dtype=np.float64)
        base_np = base_full[: benchmark.num_macros].numpy().astype(np.float64)
        all_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)

        def score_placement(candidate_placement: np.ndarray) -> float:
            if benchmark.port_positions.shape[0] == 0:
                owner = candidate_placement
            else:
                owner = np.vstack([candidate_placement, benchmark.port_positions.numpy().astype(np.float64)])
            return (
                0.45 * self._estimate_density_overflow_np(candidate_placement, benchmark)
                + 0.60 * self._estimate_congestion_overflow_np(owner, edges, benchmark)
            )

        best_np = base_np.copy()
        best_score = score_placement(best_np)
        best_scales = (1.0, 1.0)
        fallback_np = None
        fallback_density = float("inf")
        fallback_scales = (1.0, 1.0)
        soft_sizes = all_sizes[benchmark.num_hard_macros : benchmark.num_macros]
        for sx, sy in ((1.03, 1.01), (1.01, 1.03), (1.04, 1.00), (1.00, 1.04), (1.02, 0.99), (0.99, 1.02)):
            trial = base_np.copy()
            soft = trial[benchmark.num_hard_macros : benchmark.num_macros]
            if not len(soft):
                continue
            moved = soft.copy()
            moved[:, 0] = center[0] + (soft[:, 0] - center[0]) * sx
            moved[:, 1] = center[1] + (soft[:, 1] - center[1]) * sy
            moved[:, 0] = np.clip(moved[:, 0], soft_sizes[:, 0] / 2.0, cw - soft_sizes[:, 0] / 2.0)
            moved[:, 1] = np.clip(moved[:, 1], soft_sizes[:, 1] / 2.0, ch - soft_sizes[:, 1] / 2.0)
            trial[benchmark.num_hard_macros : benchmark.num_macros] = moved
            trial_score = score_placement(trial)
            trial_density = self._estimate_density_overflow_np(trial, benchmark)
            if trial_density < fallback_density - 1.0e-9:
                fallback_density = trial_density
                fallback_np = trial.copy()
                fallback_scales = (sx, sy)
            if trial_score < best_score - 1.0e-7:
                best_score = trial_score
                best_np = trial
                best_scales = (sx, sy)

        if np.allclose(best_np, base_np, rtol=0.0, atol=1.0e-9):
            if fallback_np is None:
                return None
            best_np = fallback_np
            best_score = score_placement(best_np)
            best_scales = fallback_scales
        spread_full = base_full.clone()
        spread_full[: benchmark.num_macros] = torch.tensor(best_np, dtype=torch.float32)
        spread_full = self._repair_soft_bounds_tensor(spread_full, benchmark)
        spread_full = self._repair_hard_bounds_tensor(spread_full, benchmark)
        if benchmark.macro_fixed.any():
            spread_full[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        spread_owner = self._owner_positions_from_placement(spread_full, benchmark)
        hard = spread_full[: benchmark.num_hard_macros].numpy().astype(np.float64)
        candidate = (
            self._surrogate_cost(hard, edges, spread_owner, benchmark, sizes),
            hard,
            True,
            "soft_global_density_axis_refined",
            spread_full,
        )
        record = self._soft_global_preselect_record(
            candidate,
            spread_owner,
            edges,
            benchmark,
            objective_score=best_score,
            legal_disp=0.0,
        )
        return candidate, record, f"density_axis|sx={best_scales[0]:.2f}|sy={best_scales[1]:.2f}|score={best_score:.4f}"

    def _soft_global_preselect_record(
        self,
        candidate: Candidate,
        owner_pos: np.ndarray,
        edges: List[Edge],
        benchmark: Benchmark,
        objective_score: float,
        legal_disp: float,
    ) -> Dict[str, object]:
        surrogate, _hard, _force, label, full = candidate
        if full is None:
            density = 0.0
            congestion = 0.0
        else:
            placement_np = full[: benchmark.num_macros].numpy().astype(np.float64)
            density = self._estimate_density_overflow_np(placement_np, benchmark)
            profile = self._benchmark_profile(benchmark, edges)
            if bool(profile["large_high_congestion"]):
                congestion = self._estimate_exact_style_congestion_overflow_np(placement_np, benchmark)
            else:
                congestion = self._estimate_congestion_overflow_np(owner_pos, edges, benchmark)
        raw_penalty = 0.08 if "_raw" in label else 0.0
        profile = self._benchmark_profile(benchmark, edges)
        if bool(profile["large_high_congestion"]):
            norm_surrogate = float(surrogate) / max(float(len(edges)) * 14.0, 1.0)
            corridor_penalty = 0.035 if "corridor_" in label or "corridor_planned" in label else 0.0
            topology_bonus = -0.020 if "soft_global_topo_" in label else 0.0
            rank = (
                0.35 * norm_surrogate
                + 0.20 * float(objective_score)
                + 0.65 * float(density)
                + 1.20 * float(congestion)
                + 4.0 * float(legal_disp)
                + raw_penalty
                + corridor_penalty
                + topology_bonus
            )
        else:
            rank = (
                float(surrogate)
                + 0.05 * float(objective_score)
                + 0.75 * float(density)
                + 0.35 * float(congestion)
                + 4.0 * float(legal_disp)
                + raw_penalty
            )
        return {
            "label": label,
            "candidate": candidate,
            "rank": rank,
            "density": density,
            "congestion": congestion,
            "objective": float(objective_score),
            "legal_disp": float(legal_disp),
        }

    def _preselect_soft_global_candidates(
        self,
        candidates: List[Candidate],
        records: List[Dict[str, object]],
        large_design: bool,
    ) -> Tuple[List[Candidate], List[str]]:
        if not large_design or not records:
            return candidates, []

        k = 2 if self._execution_uses_cuda() else 1
        if self._execution_uses_cuda() and len(candidates) > 10:
            k = 3

        protected_records = [
            r
            for r in records
            if ("spread_cong" in str(r["label"]) and ("_legalized" in str(r["label"]) or "_refined" in str(r["label"])))
            or "density_spread" in str(r["label"])
            or "density_axis" in str(r["label"])
            or "hotspot_refined" in str(r["label"])
            or "congestion_refined" in str(r["label"])
            or "hotspot_micro_cd" in str(r["label"])
        ]
        corridor_records = [
            r
            for r in records
            if ("corridor_planned" in str(r["label"]) or "corridor_" in str(r["label"]))
            and protected_records
            and float(r["rank"]) <= min(float(p["rank"]) for p in protected_records) * 0.995
        ]
        if not protected_records:
            corridor_records = [
                r
                for r in records
                if "corridor_planned" in str(r["label"]) or "corridor_" in str(r["label"])
            ][:1]
        topology_records = [r for r in records if "soft_global_topo_" in str(r["label"])]
        flow_records = [r for r in records if "congestion_flow_drag" in str(r["label"])]
        force_records = (
            protected_records
            + sorted(corridor_records, key=lambda r: float(r["rank"]))[:1]
            + sorted(topology_records, key=lambda r: float(r["rank"]))[:8]
        )
        if os.environ.get("JIHO_FLOW_DRAG_FORCE_EXACT", "0") == "1" and flow_records:
            best_flow = sorted(flow_records, key=lambda r: float(r["rank"]))[0]
            if all(id(best_flow["candidate"]) != id(existing["candidate"]) for existing in force_records):
                force_records.append(best_flow)
            self.flow_drag_log = (
                f"{self.flow_drag_log};flow_drag_force_soft_preselect=1|"
                f"kept={best_flow['label']}|rank={float(best_flow['rank']):.4f}|reason=debug_force"
            )
        legal_records = [r for r in records if "_legalized" in str(r["label"])]
        pool = records if any("hotspot_refined" in str(r["label"]) for r in records) else (legal_records if legal_records else records)
        kept_records = sorted(pool, key=lambda r: float(r["rank"]))[:k]
        for record in force_records:
            if all(id(record["candidate"]) != id(existing["candidate"]) for existing in kept_records):
                kept_records.append(record)
        kept_ids = {id(r["candidate"]) for r in kept_records}
        kept = [c for c in candidates if id(c) in kept_ids]
        dropped = [
            f"{r['label']}|rank={float(r['rank']):.4f}|den={float(r['density']):.4f}|"
            f"cong={float(r['congestion']):.4f}|disp={float(r['legal_disp']):.5f}|drop=soft_global_preselect"
            for r in records
            if id(r["candidate"]) not in kept_ids
        ]
        return kept, dropped

    def _estimate_density_overflow_np(self, placement: np.ndarray, benchmark: Benchmark) -> float:
        rows = min(max(10, int(benchmark.grid_rows)), 18)
        cols = min(max(10, int(benchmark.grid_cols)), 18)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        cell_w = cw / cols
        cell_h = ch / rows
        cell_area = max(cell_w * cell_h, 1.0e-9)
        grid = np.zeros((rows, cols), dtype=np.float64)
        xs = np.clip((placement[:, 0] / max(cw, 1.0e-9) * cols).astype(int), 0, cols - 1)
        ys = np.clip((placement[:, 1] / max(ch, 1.0e-9) * rows).astype(int), 0, rows - 1)
        area = sizes[:, 0] * sizes[:, 1]
        np.add.at(grid, (ys, xs), area)
        util = grid / cell_area
        target = float(np.clip(area.sum() / max(cw * ch, 1.0e-9) * self.soft_global_density_target_scale, 0.58, 0.92))
        return float(np.mean(np.maximum(util - target, 0.0) ** 2))

    def _estimate_congestion_overflow_np(
        self, owner_pos: np.ndarray, edges: List[Edge], benchmark: Benchmark
    ) -> float:
        h_grid, v_grid = self._routing_congestion_arrays_np(owner_pos, edges, benchmark)
        values = np.sort(np.concatenate([h_grid.reshape(-1), v_grid.reshape(-1)]))[::-1]
        if values.size == 0:
            return 0.0
        cnt = max(1, int(math.floor(values.size * 0.05)))
        return float(np.sum(values[:cnt]) / cnt)

    def _macro_routing_allocations(self, benchmark: Benchmark) -> Tuple[float, float]:
        cached = self._macro_alloc_cache.get(benchmark.name)
        if cached is not None:
            return cached

        h_alloc = float(benchmark.hroutes_per_micron) * 0.46
        v_alloc = float(benchmark.vroutes_per_micron) * 0.67
        plc_path = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name / "initial.plc"
        if plc_path.exists():
            try:
                for line in plc_path.read_text().splitlines():
                    if "Routes used by macros" not in line:
                        continue
                    parts = line.replace(":", " ").split()
                    h_idx = parts.index("hor") + 1
                    v_idx = parts.index("ver") + 1
                    h_alloc = float(parts[h_idx])
                    v_alloc = float(parts[v_idx])
                    break
            except Exception:
                pass
        self._macro_alloc_cache[benchmark.name] = (h_alloc, v_alloc)
        return h_alloc, v_alloc

    def _routing_congestion_arrays_np(
        self, owner_pos: np.ndarray, edges: List[Edge], benchmark: Benchmark
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cell_w = cw / cols
        cell_h = ch / rows
        h_grid = np.zeros((rows, cols), dtype=np.float64)
        v_grid = np.zeros((rows, cols), dtype=np.float64)

        def grid_loc(point: np.ndarray) -> Tuple[int, int]:
            r = int(math.floor(float(point[1]) / max(cell_h, 1.0e-9)))
            c = int(math.floor(float(point[0]) / max(cell_w, 1.0e-9)))
            return int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1))

        for edge in edges:
            if hasattr(edge, "a"):
                a, b, weight = int(edge.a), int(edge.b), float(edge.weight)
            else:
                a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
            if a >= len(owner_pos) or b >= len(owner_pos):
                continue
            r0, c0 = grid_loc(owner_pos[a])
            r1, c1 = grid_loc(owner_pos[b])
            c_lo, c_hi = sorted((c0, c1))
            r_lo, r_hi = sorted((r0, r1))
            if c_hi > c_lo:
                h_grid[r0, c_lo:c_hi] += weight
            if r_hi > r_lo:
                v_grid[r_lo:r_hi, c1] += weight

        sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].numpy().astype(np.float64)
        h_alloc, v_alloc = self._macro_routing_allocations(benchmark)
        hard_pos = owner_pos[: benchmark.num_hard_macros]
        for (x, y), (w, h) in zip(hard_pos, sizes):
            x0 = max(0.0, float(x) - float(w) * 0.5)
            x1 = min(cw, float(x) + float(w) * 0.5)
            y0 = max(0.0, float(y) - float(h) * 0.5)
            y1 = min(ch, float(y) + float(h) * 0.5)
            if x1 <= x0 or y1 <= y0:
                continue
            c0 = int(np.clip(math.floor(x0 / max(cell_w, 1.0e-9)), 0, cols - 1))
            c1 = int(np.clip(math.floor(x1 / max(cell_w, 1.0e-9)), 0, cols - 1))
            r0 = int(np.clip(math.floor(y0 / max(cell_h, 1.0e-9)), 0, rows - 1))
            r1 = int(np.clip(math.floor(y1 / max(cell_h, 1.0e-9)), 0, rows - 1))
            for r in range(r0, r1 + 1):
                gy0 = r * cell_h
                gy1 = (r + 1) * cell_h
                y_ov = max(0.0, min(y1, gy1) - max(y0, gy0))
                if y_ov <= 0.0:
                    continue
                for c in range(c0, c1 + 1):
                    gx0 = c * cell_w
                    gx1 = (c + 1) * cell_w
                    x_ov = max(0.0, min(x1, gx1) - max(x0, gx0))
                    if x_ov <= 0.0:
                        continue
                    v_grid[r, c] += x_ov * v_alloc
                    h_grid[r, c] += y_ov * h_alloc

        h_grid /= max(cell_h * float(benchmark.hroutes_per_micron), 1.0e-9)
        v_grid /= max(cell_w * float(benchmark.vroutes_per_micron), 1.0e-9)

        smooth = 2
        if smooth > 0:
            smoothed_v = np.zeros_like(v_grid)
            for c in range(cols):
                lo = max(0, c - smooth)
                hi = min(cols - 1, c + smooth)
                smoothed_v[:, lo : hi + 1] += v_grid[:, c : c + 1] / (hi - lo + 1)
            v_grid = smoothed_v

            smoothed_h = np.zeros_like(h_grid)
            for r in range(rows):
                lo = max(0, r - smooth)
                hi = min(rows - 1, r + smooth)
                smoothed_h[lo : hi + 1, :] += h_grid[r : r + 1, :] / (hi - lo + 1)
            h_grid = smoothed_h

        return h_grid, v_grid

    def _pin_points_for_net_np(
        self, pins_tensor, owner_pos: np.ndarray, benchmark: Benchmark
    ) -> Tuple[np.ndarray, List[int]]:
        if getattr(pins_tensor, "numel", lambda: 0)() == 0:
            return np.zeros((0, 2), dtype=np.float64), []
        if getattr(pins_tensor, "ndim", 1) == 2:
            pins = pins_tensor.detach().cpu().numpy().astype(np.int64)
            owners = pins[:, 0].astype(np.int64)
            pin_ids = pins[:, 1].astype(np.int64)
        else:
            owners = pins_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
            pin_ids = np.zeros_like(owners)
        points: List[np.ndarray] = []
        kept_owners: List[int] = []
        offsets = getattr(benchmark, "macro_pin_offsets", [])
        n_hard = int(benchmark.num_hard_macros)
        for owner, pin_id in zip(owners.tolist(), pin_ids.tolist()):
            if owner < 0 or owner >= len(owner_pos):
                continue
            point = owner_pos[owner].astype(np.float64).copy()
            if owner < n_hard and owner < len(offsets):
                try:
                    owner_offsets = offsets[owner]
                    if int(pin_id) < int(owner_offsets.shape[0]):
                        point += owner_offsets[int(pin_id)].detach().cpu().numpy().astype(np.float64)
                except Exception:
                    pass
            points.append(point)
            kept_owners.append(int(owner))
        if not points:
            return np.zeros((0, 2), dtype=np.float64), []
        return np.vstack(points), kept_owners

    def _smooth_grid_np(self, grid: np.ndarray) -> np.ndarray:
        if grid.size == 0 or min(grid.shape) <= 1:
            return grid
        padded = np.pad(grid, ((1, 1), (1, 1)), mode="edge")
        out = np.zeros_like(grid)
        for dr in range(3):
            for dc in range(3):
                out += padded[dr : dr + grid.shape[0], dc : dc + grid.shape[1]]
        return out / 9.0

    def _exact_style_congestion_arrays_np(
        self, placement_np: np.ndarray, benchmark: Benchmark
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cell_w = cw / cols
        cell_h = ch / rows
        h_grid = np.zeros((rows, cols), dtype=np.float64)
        v_grid = np.zeros((rows, cols), dtype=np.float64)
        owner_pos = (
            placement_np
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        )
        raw_nets = benchmark.net_pin_nodes if benchmark.net_pin_nodes else benchmark.net_nodes
        weights = benchmark.net_weights.tolist() if getattr(benchmark, "net_weights", None) is not None else []
        span = max(cw, ch, 1.0e-9)
        for net_id, pins_tensor in enumerate(raw_nets):
            pts, owners = self._pin_points_for_net_np(pins_tensor, owner_pos, benchmark)
            if pts.shape[0] < 2:
                continue
            unique_owners = sorted(set(owners))
            fanout = max(2, len(unique_owners))
            xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
            ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
            x0 = int(np.clip(math.floor(xmin / max(cell_w, 1.0e-9)), 0, cols - 1))
            x1 = int(np.clip(math.floor(xmax / max(cell_w, 1.0e-9)), 0, cols - 1))
            y0 = int(np.clip(math.floor(ymin / max(cell_h, 1.0e-9)), 0, rows - 1))
            y1 = int(np.clip(math.floor(ymax / max(cell_h, 1.0e-9)), 0, rows - 1))
            if x1 == x0 and y1 == y0:
                continue
            weight = float(weights[net_id]) if net_id < len(weights) else 1.0
            weight /= math.sqrt(max(1, fanout - 1))
            x_span = max(xmax - xmin, cell_w)
            y_span = max(ymax - ymin, cell_h)
            if x1 > x0:
                row_count = max(1, y1 - y0 + 1)
                route_weight = weight * (1.0 + 0.22 * y_span / span) / row_count
                h_grid[y0 : y1 + 1, x0:x1] += route_weight
            if y1 > y0:
                col_count = max(1, x1 - x0 + 1)
                route_weight = weight * (1.0 + 0.22 * x_span / span) / col_count
                v_grid[y0:y1, x0 : x1 + 1] += route_weight

        sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        h_alloc, v_alloc = self._macro_routing_allocations(benchmark)
        for idx, ((x, y), (w, h)) in enumerate(zip(placement_np, sizes)):
            x0 = max(0.0, float(x) - float(w) * 0.5)
            x1 = min(cw, float(x) + float(w) * 0.5)
            y0 = max(0.0, float(y) - float(h) * 0.5)
            y1 = min(ch, float(y) + float(h) * 0.5)
            if x1 <= x0 or y1 <= y0:
                continue
            factor = 1.0 if idx < int(benchmark.num_hard_macros) else 0.22
            c0 = int(np.clip(math.floor(x0 / max(cell_w, 1.0e-9)), 0, cols - 1))
            c1 = int(np.clip(math.floor(x1 / max(cell_w, 1.0e-9)), 0, cols - 1))
            r0 = int(np.clip(math.floor(y0 / max(cell_h, 1.0e-9)), 0, rows - 1))
            r1 = int(np.clip(math.floor(y1 / max(cell_h, 1.0e-9)), 0, rows - 1))
            for r in range(r0, r1 + 1):
                gy0 = r * cell_h
                gy1 = (r + 1) * cell_h
                y_ov = max(0.0, min(y1, gy1) - max(y0, gy0))
                if y_ov <= 0.0:
                    continue
                for c in range(c0, c1 + 1):
                    gx0 = c * cell_w
                    gx1 = (c + 1) * cell_w
                    x_ov = max(0.0, min(x1, gx1) - max(x0, gx0))
                    if x_ov <= 0.0:
                        continue
                    h_grid[r, c] += factor * y_ov * h_alloc
                    v_grid[r, c] += factor * x_ov * v_alloc

        h_grid /= max(cell_h * float(benchmark.hroutes_per_micron), 1.0e-9)
        v_grid /= max(cell_w * float(benchmark.vroutes_per_micron), 1.0e-9)
        return self._smooth_grid_np(h_grid), self._smooth_grid_np(v_grid)

    def _exact_style_congestion_map_np(self, placement_np: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        h_grid, v_grid = self._exact_style_congestion_arrays_np(placement_np, benchmark)
        return h_grid + v_grid

    def _tail_mean_np(self, values: np.ndarray, frac: float) -> float:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return 0.0
        count = max(1, int(math.ceil(flat.size * frac)))
        return float(np.mean(np.sort(flat)[-count:]))

    def _estimate_exact_style_congestion_overflow_np(self, placement_np: np.ndarray, benchmark: Benchmark) -> float:
        h_grid, v_grid = self._exact_style_congestion_arrays_np(placement_np, benchmark)
        return self._tail_mean_np(np.concatenate([h_grid.reshape(-1), v_grid.reshape(-1)]), 0.05)

    def _topology_exact_style_pressure(self, placement_np: np.ndarray, benchmark: Benchmark) -> float:
        return self._tail_mean_np(self._exact_style_congestion_map_np(placement_np, benchmark), 0.08)

    def _extract_exact_congestion_map(self, plc) -> Optional[np.ndarray]:
        try:
            rows = int(getattr(plc, "grid_row", 0) or getattr(plc, "num_rows", 0) or 0)
            cols = int(getattr(plc, "grid_col", 0) or getattr(plc, "num_cols", 0) or 0)
            h_raw = getattr(plc, "H_routing_cong", None)
            v_raw = getattr(plc, "V_routing_cong", None)
            if h_raw is None and hasattr(plc, "get_horizontal_routing_congestion"):
                h_raw = plc.get_horizontal_routing_congestion()
            if v_raw is None and hasattr(plc, "get_vertical_routing_congestion"):
                v_raw = plc.get_vertical_routing_congestion()
            if h_raw is None or v_raw is None:
                return None
            h_grid = np.asarray(h_raw, dtype=np.float64)
            v_grid = np.asarray(v_raw, dtype=np.float64)
            if rows <= 0 or cols <= 0:
                size = int(math.sqrt(max(h_grid.size, v_grid.size)))
                rows = cols = size
            if rows * cols <= 0 or h_grid.size != rows * cols or v_grid.size != rows * cols:
                return None
            return h_grid.reshape(rows, cols) + v_grid.reshape(rows, cols)
        except Exception:
            return None

    def _resample_grid_np(self, grid: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        if grid.shape == shape:
            return grid.astype(np.float64)
        rows, cols = shape
        if rows <= 0 or cols <= 0 or grid.size == 0:
            return np.zeros(shape, dtype=np.float64)
        r_idx = np.clip(((np.arange(rows) + 0.5) * grid.shape[0] / rows).astype(int), 0, grid.shape[0] - 1)
        c_idx = np.clip(((np.arange(cols) + 0.5) * grid.shape[1] / cols).astype(int), 0, grid.shape[1] - 1)
        return grid[np.ix_(r_idx, c_idx)].astype(np.float64)

    def _hot_cell_tokens(self, grid: np.ndarray, limit: int = 8) -> str:
        if grid.size == 0:
            return "none"
        flat = grid.reshape(-1)
        count = min(limit, flat.size)
        order = np.argsort(flat)[-count:][::-1]
        tokens = []
        for flat_idx in order:
            r, c = divmod(int(flat_idx), grid.shape[1])
            tokens.append(f"{r}x{c}:{float(flat[flat_idx]):.3f}")
        return "/".join(tokens)

    def _grid_alignment_summary(self, predicted: np.ndarray, exact: Optional[np.ndarray]) -> Tuple[str, float, float]:
        if exact is None or predicted.size == 0 or exact.size == 0:
            return "exact_map=missing", float("nan"), float("nan")
        pred = self._resample_grid_np(predicted, exact.shape).reshape(-1)
        ex = exact.astype(np.float64).reshape(-1)
        pred_std = float(np.std(pred))
        ex_std = float(np.std(ex))
        corr = 0.0 if pred_std <= 1.0e-12 or ex_std <= 1.0e-12 else float(np.corrcoef(pred, ex)[0, 1])
        top_n = max(1, int(math.ceil(ex.size * 0.06)))
        pred_top = set(int(i) for i in np.argsort(pred)[-top_n:])
        exact_top = set(int(i) for i in np.argsort(ex)[-top_n:])
        overlap = len(pred_top & exact_top) / max(1, top_n)
        return f"corr={corr:.3f}|top_overlap={overlap:.3f}", corr, overlap

    def _soft_global_schedule_specs(self, benchmark: Benchmark) -> List[Dict[str, object]]:
        requested = getattr(self, "soft_global_schedules", ())
        if isinstance(requested, str):
            requested = (requested,)
        wanted = {str(name) for name in requested}
        if self.execution_mode_used == "cuda_debug":
            profile = self._benchmark_profile(benchmark)
            wanted = {"spread_cong", "congestion_escape", "corridor_planned"} if profile["large_high_congestion"] else {"spread_cong"}
        n_hard = int(benchmark.num_hard_macros)
        num_macros = int(benchmark.num_macros)
        large = n_hard >= 650 or num_macros >= 1800
        if large and self.execution_mode_used == "local_dev":
            wanted &= {"balanced_full"}
        base_source = self.soft_global_iters_debug if self.execution_mode_used == "cuda_debug" else self.soft_global_iters
        base = tuple(int(x) for x in base_source)
        if len(base) != 3:
            base = (30, 50, 20) if self.execution_mode_used == "cuda_debug" else (300, 500, 200)

        def spec(
            name,
            stages,
            lr_scale=1.0,
            density_target=None,
            congestion_target=None,
            soft_disp=None,
            checkpoint_top_k=None,
            force_diff_congestion=False,
            corridor_weight=0.0,
        ):
            if self.execution_mode_used == "cuda_debug":
                checkpoint_top_k = 1
            return {
                "name": name,
                "stages": stages,
                "lr_scale": float(lr_scale),
                "density_target_scale": float(
                    self.soft_global_density_target_scale if density_target is None else density_target
                ),
                "congestion_target_scale": float(
                    self.soft_global_congestion_target_scale if congestion_target is None else congestion_target
                ),
                "soft_disp_weight": float(self.soft_global_soft_disp_weight if soft_disp is None else soft_disp),
                "checkpoint_top_k": checkpoint_top_k,
                "force_diff_congestion": bool(force_diff_congestion),
                "corridor_weight": float(corridor_weight),
            }

        specs = []
        if "spread_only" in wanted:
            specs.append(
                spec(
                    "spread_only",
                    ((max(base[0] + base[1] // 4, base[0]), 0.025, 5.4, 0.18, 5.0, 1.0, 0.04, 1.00),),
                    density_target=1.03,
                    congestion_target=1.35,
                    soft_disp=0.34,
                    checkpoint_top_k=1 if large else 3,
                )
            )
        if "spread_cong" in wanted:
            specs.append(
                spec(
                    "spread_cong",
                    (
                        (base[0], 0.025, 4.9, 0.28, 4.6, 1.0, 0.05, 1.00),
                        (
                            base[1] if self.execution_mode_used == "cuda_debug" else max(base[1] // 3, 60),
                            0.055,
                            2.4,
                            1.15,
                            2.8,
                            0.9,
                            0.18,
                            0.45,
                        ),
                    ),
                    density_target=1.04,
                    congestion_target=1.18,
                    soft_disp=0.46,
                    checkpoint_top_k=1 if large else 3,
                )
            )
        if "balanced_full" in wanted:
            specs.append(
                spec(
                    "balanced_full",
                    (
                        (base[0], 0.025, 4.8, 0.45, 4.5, 1.0, 0.06, 1.00),
                        (base[1], 0.190, 1.7, 1.20, 2.4, 0.8, 0.24, 0.55),
                        (base[2], 0.145, 2.2, 0.95, 2.9, 1.0, 0.62, 0.25),
                    ),
                    checkpoint_top_k=1 if large else 4,
                )
            )
        if "congestion_guarded" in wanted:
            specs.append(
                spec(
                    "congestion_guarded",
                    (
                        (base[0], 0.018, 3.4, 1.30, 4.2, 1.1, 0.16, 0.58),
                        (
                            base[2] if self.execution_mode_used == "cuda_debug" else max(base[1] // 2, 80),
                            0.050,
                            1.2,
                            2.40,
                            2.0,
                            1.0,
                            0.70,
                            0.25,
                        ),
                    ),
                    lr_scale=0.78,
                    density_target=1.08,
                    congestion_target=1.06,
                    soft_disp=0.70,
                    checkpoint_top_k=1 if large else 3,
                )
            )
        if "congestion_escape" in wanted:
            specs.append(
                spec(
                    "congestion_escape",
                    (
                        (base[0], 0.020, 5.8, 0.95, 4.6, 1.0, 0.06, 1.00),
                        (
                            base[1] if self.execution_mode_used == "cuda_debug" else max(base[1] // 3, 60),
                            0.030,
                            4.2,
                            1.65,
                            2.8,
                            0.9,
                            0.20,
                            0.45,
                        ),
                    ),
                    density_target=1.00,
                    congestion_target=1.05,
                    soft_disp=0.55,
                    checkpoint_top_k=1 if large else 3,
                    force_diff_congestion=True,
                )
            )
        if "corridor_planned" in wanted:
            specs.append(
                spec(
                    "corridor_planned",
                    (
                        (base[0], 0.026, 4.8, 0.0, 4.4, 1.0, 0.08, 1.00, 0.55),
                        (
                            base[1] if self.execution_mode_used == "cuda_debug" else max(base[1] // 3, 60),
                            0.040,
                            3.1,
                            0.0,
                            2.9,
                            0.9,
                            0.20,
                            0.48,
                            0.85,
                        ),
                    ),
                    density_target=1.03,
                    congestion_target=1.18,
                    soft_disp=0.52,
                    checkpoint_top_k=1 if large else 3,
                    corridor_weight=1.0,
                )
            )
        return specs

    def _soft_global_optimize(self, benchmark: Benchmark, schedule: Dict[str, object]) -> Tuple[List[Tuple[str, np.ndarray, float]], str, str]:
        device = self._soft_global_device()
        self.soft_global_device_used = str(device)
        dtype = torch.float32
        num_macros = benchmark.num_macros
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        span = max(cw, ch)

        init = benchmark.macro_positions[:num_macros].to(device=device, dtype=dtype)
        sizes = benchmark.macro_sizes[:num_macros].to(device=device, dtype=dtype)
        fixed = benchmark.macro_fixed[:num_macros].to(device=device)
        movable = (~fixed).to(device=device)
        soft_mask = torch.zeros(num_macros, dtype=torch.bool, device=device)
        soft_mask[n_hard:num_macros] = True
        soft_movable = soft_mask & movable
        ports = benchmark.port_positions.to(device=device, dtype=dtype)
        half = sizes / 2.0
        low = half + 1.0e-4
        high = torch.tensor([cw, ch], device=device, dtype=dtype).view(1, 2) - half - 1.0e-4

        var = torch.nn.Parameter(init.clone())
        base_lr = float(self.soft_global_lr) * float(schedule.get("lr_scale", 1.0))
        optimizer = torch.optim.AdamW([var], lr=base_lr, weight_decay=0.0)
        nets = self._soft_global_nets(benchmark, device)
        raw_stage_weights = tuple(schedule.get("stages", ()))
        if not raw_stage_weights:
            raw_stage_weights = (
                (300, 0.025, 4.8, 0.45, 4.5, 1.0, 0.06, 1.00),
                (500, 0.190, 1.7, 1.20, 2.4, 0.8, 0.24, 0.55),
                (200, 0.145, 2.2, 0.95, 2.9, 1.0, 0.62, 0.25),
            )
        iter_scale = 1.0
        if device.type != "cuda":
            # Local CPU fallback is for correctness smoke; judge machine gets full GPU schedule.
            iter_scale = 0.08 if num_macros < 1800 else 0.04

        rows = max(12, min(40, int(benchmark.grid_rows)))
        cols = max(12, min(40, int(benchmark.grid_cols)))
        if self.execution_mode_used == "cuda_debug":
            rows = min(rows, 16)
            cols = min(cols, 16)
        if device.type != "cuda":
            rows = min(rows, 24)
            cols = min(cols, 24)
        density_edges = self._soft_global_bin_edges(cw, ch, rows, cols, device, dtype)
        cong_rows = max(10, min(rows, 28 if device.type == "cuda" else 18))
        cong_cols = max(10, min(cols, 28 if device.type == "cuda" else 18))
        congestion_edges = self._soft_global_bin_edges(cw, ch, cong_rows, cong_cols, device, dtype)

        stage_weights = []
        for raw in raw_stage_weights:
            if len(raw) >= 9:
                iters, w_wl, w_den, w_cong, w_ov, w_bound, w_soft_disp, lr_scale, w_corridor = raw[:9]
            else:
                iters, w_wl, w_den, w_cong, w_ov, w_bound, w_soft_disp, lr_scale = raw
                w_corridor = 0.0
            scaled_iters = int(iters)
            if iter_scale != 1.0:
                scaled_iters = max(6, int(scaled_iters * iter_scale))
            stage_weights.append((scaled_iters, w_wl, w_den, w_cong, w_ov, w_bound, w_soft_disp, lr_scale, w_corridor))
        logs = []
        checkpoints: List[Tuple[str, np.ndarray, float, Dict[str, float]]] = []
        best_objective = float("inf")
        checkpoint_every = max(0, int(self.soft_global_checkpoint_every))
        density_target_scale = float(schedule.get("density_target_scale", self.soft_global_density_target_scale))
        congestion_target_scale = float(
            schedule.get("congestion_target_scale", self.soft_global_congestion_target_scale)
        )
        soft_disp_weight = float(schedule.get("soft_disp_weight", self.soft_global_soft_disp_weight))
        schedule_name = str(schedule.get("name", "unknown"))
        corridor_weight_scale = float(schedule.get("corridor_weight", 0.0))
        corridor_plan = (
            self._soft_global_predict_corridors(benchmark, device, dtype) if corridor_weight_scale > 0.0 else None
        )
        total_stage_count = len(stage_weights)
        use_diff_congestion = self._soft_global_uses_diff_congestion() or (
            self.execution_mode_used == "cuda_debug" and bool(schedule.get("force_diff_congestion", False))
        )
        for stage_id, (iters, w_wl, w_den, w_cong, w_ov, w_bound, w_soft_disp, lr_scale, w_corridor) in enumerate(stage_weights):
            if iters <= 0:
                continue
            if not use_diff_congestion:
                w_cong = 0.0
            for group in optimizer.param_groups:
                group["lr"] = base_lr * float(lr_scale)
            last = {}
            stage_timings: Dict[str, float] = {}
            stage_start = time.perf_counter()
            print(
                "[JihoPlacer][soft_global] stage "
                f"{stage_id + 1}/{total_stage_count} start schedule={schedule_name} "
                f"iters={iters} device={device} lr={base_lr * float(lr_scale):.6f}",
                flush=True,
            )
            for step in range(iters):
                profile_components = self.execution_mode_used != "cuda_debug" or step == 0 or step == iters - 1
                optimizer.zero_grad(set_to_none=True)
                pos = torch.where(movable[:, None], var, init)
                all_pos = torch.cat([pos, ports], dim=0) if ports.numel() else pos
                t_part = self._maybe_component_timer_start(device, profile_components)
                wl = self._soft_global_hpwl(all_pos, nets, span)
                self._maybe_component_timer_add(stage_timings, "hpwl", device, t_part, profile_components)
                t_part = self._maybe_component_timer_start(device, profile_components)
                density = self._soft_global_density(
                    pos, sizes, cw, ch, rows, cols, density_edges, density_target_scale
                )
                self._maybe_component_timer_add(stage_timings, "density", device, t_part, profile_components)
                t_part = self._maybe_component_timer_start(device, profile_components)
                if use_diff_congestion and w_cong != 0.0:
                    cong = self._soft_global_congestion(
                        all_pos, nets, cw, ch, cong_rows, cong_cols, congestion_edges, congestion_target_scale
                    )
                else:
                    cong = all_pos.sum() * 0.0
                self._maybe_component_timer_add(stage_timings, "congestion", device, t_part, profile_components)
                t_part = self._maybe_component_timer_start(device, profile_components)
                overlap = self._soft_global_hard_overlap(pos[:n_hard], sizes[:n_hard], span)
                self._maybe_component_timer_add(stage_timings, "overlap", device, t_part, profile_components)
                t_part = self._maybe_component_timer_start(device, profile_components)
                if corridor_plan is not None and w_corridor != 0.0:
                    corridor = self._soft_global_corridor_blockage(pos, sizes, movable, corridor_plan, cw, ch)
                else:
                    corridor = pos.sum() * 0.0
                self._maybe_component_timer_add(stage_timings, "corridor", device, t_part, profile_components)
                boundary = self._soft_global_boundary(pos, half, cw, ch, span)
                soft_disp = self._soft_global_soft_displacement(pos, init, sizes, soft_movable, span)
                loss = (
                    w_wl * wl
                    + w_den * density
                    + w_cong * cong
                    + w_ov * overlap
                    + w_bound * boundary
                    + soft_disp_weight * w_soft_disp * soft_disp
                    + corridor_weight_scale * w_corridor * corridor
                )
                t_part = self._maybe_component_timer_start(device, profile_components)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([var], max_norm=span * 0.25)
                optimizer.step()
                with torch.no_grad():
                    var.data = torch.minimum(torch.maximum(var.data, low), high)
                    var.data[fixed] = init[fixed]
                self._maybe_component_timer_add(stage_timings, "backward_step", device, t_part, profile_components)
                objective = float(loss.detach().cpu())
                last = {
                    "loss": objective,
                    "wl": float(wl.detach().cpu()),
                    "den": float(density.detach().cpu()),
                    "cong": float(cong.detach().cpu()),
                    "ov": float(overlap.detach().cpu()),
                    "corridor": float(corridor.detach().cpu()),
                    "soft_disp": float(soft_disp.detach().cpu()),
                }
                if checkpoint_every and objective < best_objective and (step + 1) % checkpoint_every == 0:
                    best_objective = objective
                    out = torch.where(movable[:, None], var, init).detach().cpu().numpy().astype(np.float64)
                    checkpoints.append((f"best_s{stage_id}_{step + 1}", out, objective, dict(last)))
            self._sync_timing_device(device)
            stage_elapsed = time.perf_counter() - stage_start
            out = torch.where(movable[:, None], var, init).detach().cpu().numpy().astype(np.float64)
            checkpoints.append((f"stage{stage_id}", out, float(last.get("loss", float("inf"))), dict(last)))
            timing_text = (
                f"hpwl_s={stage_timings.get('hpwl', 0.0):.3f},"
                f"density_s={stage_timings.get('density', 0.0):.3f},"
                f"congestion_s={stage_timings.get('congestion', 0.0):.3f},"
                f"corridor_s={stage_timings.get('corridor', 0.0):.3f},"
                f"overlap_s={stage_timings.get('overlap', 0.0):.3f},"
                f"backward_step_s={stage_timings.get('backward_step', 0.0):.3f}"
            )
            print(
                "[JihoPlacer][soft_global] stage "
                f"{stage_id + 1}/{total_stage_count} done schedule={schedule_name} "
                f"iters={iters} device={device} elapsed={stage_elapsed:.3f}s "
                f"loss={last.get('loss', 0.0):.4f} {timing_text}",
                flush=True,
            )
            logs.append(
                "stage"
                f"{stage_id}:iters={iters},loss={last.get('loss', 0.0):.4f},"
                f"wl={last.get('wl', 0.0):.4f},den={last.get('den', 0.0):.4f},"
                f"cong={last.get('cong', 0.0):.4f},ov={last.get('ov', 0.0):.4f},"
                f"corridor={last.get('corridor', 0.0):.4f},soft_disp={last.get('soft_disp', 0.0):.4f},"
                f"elapsed_s={stage_elapsed:.3f},"
                f"{timing_text}"
            )
        final_out = torch.where(movable[:, None], var, init).detach().cpu().numpy().astype(np.float64)
        if not checkpoints or not np.allclose(checkpoints[-1][1], final_out, rtol=0.0, atol=1.0e-9):
            checkpoints.append(("final", final_out, float(logs and last.get("loss", float("inf")) or 0.0), dict(last)))

        deduped: List[Tuple[str, np.ndarray, float, Dict[str, float]]] = []
        seen = set()
        effective_top_k = max(1, int(schedule.get("checkpoint_top_k") or self.soft_global_checkpoint_top_k))
        if self.execution_mode_used == "cuda_debug":
            effective_top_k = 1
        large_design = n_hard >= 650 or num_macros >= 1800
        if large_design:
            effective_top_k = min(effective_top_k, 2)
        elif n_hard >= 450 or num_macros >= 1200:
            effective_top_k = min(effective_top_k, 3)
        for label, pos, score, parts in sorted(checkpoints, key=lambda row: row[2]):
            key = label.split("_", 1)[0] if label.startswith("stage") else label
            if label in seen:
                continue
            seen.add(label)
            deduped.append((label, pos, score, parts))
            if len(deduped) >= effective_top_k:
                break
        if not large_design and not any(label == "final" for label, _pos, _score, _parts in deduped):
            deduped.append(("final", final_out, float(last.get("loss", float("inf"))), dict(last)))

        init_np = init.detach().cpu().numpy()
        soft_slice = slice(n_hard, num_macros)
        cp_logs = []
        for label, pos, score, parts in deduped:
            if num_macros > n_hard:
                disp = np.linalg.norm(pos[soft_slice] - init_np[soft_slice], axis=1)
                avg_soft_disp = float(disp.mean() / max(span, 1.0e-6)) if len(disp) else 0.0
                max_soft_disp = float(disp.max() / max(span, 1.0e-6)) if len(disp) else 0.0
            else:
                avg_soft_disp = max_soft_disp = 0.0
            cp_logs.append(
                f"{label}|obj={score:.4f}|wl={parts.get('wl', 0.0):.4f}|den={parts.get('den', 0.0):.4f}|"
                f"cong={parts.get('cong', 0.0):.4f}|corridor={parts.get('corridor', 0.0):.4f}|"
                f"avg_soft_disp={avg_soft_disp:.5f}|max_soft_disp={max_soft_disp:.5f}"
            )
        if corridor_plan is not None:
            final_tensor = torch.as_tensor(final_out, device=device, dtype=dtype)
            init_block = float(self._soft_global_corridor_blockage(init, sizes, movable, corridor_plan, cw, ch).detach().cpu())
            final_block = float(
                self._soft_global_corridor_blockage(final_tensor, sizes, movable, corridor_plan, cw, ch).detach().cpu()
            )
            logs.insert(
                0,
                f"corridors={corridor_plan['log']}|blockage_before={init_block:.4f}|blockage_after={final_block:.4f}",
            )
        return [(label, pos, score) for label, pos, score, _parts in deduped], ";".join(logs), ";".join(cp_logs)

    def _soft_global_predict_corridors(
        self,
        benchmark: Benchmark,
        device: torch.device,
        dtype: torch.dtype,
        base_positions: Optional[torch.Tensor] = None,
        width_scale: float = 1.0,
        weight_scale: float = 1.0,
        bridge_boost: float = 0.45,
        max_corridors: int = 3,
    ):
        num_macros = int(benchmark.num_macros)
        if num_macros <= 0:
            return None
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        span = max(cw, ch, 1.0e-9)
        if base_positions is None:
            macro_pos = benchmark.macro_positions[:num_macros].numpy().astype(np.float64)
        else:
            macro_pos = base_positions.detach().cpu().numpy().astype(np.float64)
        ports = benchmark.port_positions.numpy().astype(np.float64)
        owner_pos = macro_pos if ports.shape[0] == 0 else np.vstack([macro_pos, ports])
        cut_cols = max(8, min(32, int(benchmark.grid_cols)))
        cut_rows = max(8, min(32, int(benchmark.grid_rows)))
        x_demand = np.zeros(cut_cols, dtype=np.float64)
        y_demand = np.zeros(cut_rows, dtype=np.float64)
        degree = np.zeros(num_macros, dtype=np.float64)

        raw_nets = benchmark.net_pin_nodes if benchmark.net_pin_nodes else benchmark.net_nodes
        weights = benchmark.net_weights.tolist() if getattr(benchmark, "net_weights", None) is not None else []
        for net_id, owners_tensor in enumerate(raw_nets):
            owners_raw = owners_tensor[:, 0] if getattr(owners_tensor, "ndim", 1) == 2 else owners_tensor
            owners = sorted(set(int(x) for x in owners_raw.tolist()))
            owners = [o for o in owners if 0 <= o < len(owner_pos)]
            if len(owners) < 2:
                continue
            pts = owner_pos[owners]
            xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
            ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
            if xmax <= xmin and ymax <= ymin:
                continue
            weight = float(weights[net_id]) if net_id < len(weights) else 1.0
            weight /= math.sqrt(max(1, len(owners) - 1))
            for owner in owners:
                if owner < num_macros:
                    degree[owner] += weight

            x0 = int(np.clip(math.floor(xmin / max(cw, 1.0e-9) * cut_cols), 0, cut_cols - 1))
            x1 = int(np.clip(math.floor(xmax / max(cw, 1.0e-9) * cut_cols), 0, cut_cols - 1))
            y0 = int(np.clip(math.floor(ymin / max(ch, 1.0e-9) * cut_rows), 0, cut_rows - 1))
            y1 = int(np.clip(math.floor(ymax / max(ch, 1.0e-9) * cut_rows), 0, cut_rows - 1))
            if x1 > x0:
                x_demand[x0 : x1 + 1] += weight * (1.0 + (ymax - ymin) / span)
            if y1 > y0:
                y_demand[y0 : y1 + 1] += weight * (1.0 + (xmax - xmin) / span)

        corridors: List[Tuple[str, float, float, float]] = []

        def pick(demand: np.ndarray, axis: str, extent: float, limit: int) -> None:
            active = demand[demand > 0.0]
            if active.size == 0:
                return
            chosen: List[int] = []
            order = list(np.argsort(demand)[::-1])
            median = float(np.median(active))
            peak = float(np.max(active))
            for idx in order:
                idx = int(idx)
                if demand[idx] <= max(median * 1.12, peak * 0.35):
                    break
                if any(abs(idx - existing) <= 1 for existing in chosen):
                    continue
                chosen.append(idx)
                importance = float(demand[idx] / max(peak, 1.0e-9)) * float(weight_scale)
                width = max(extent / len(demand) * 1.15, extent * 0.028) * float(width_scale)
                center = (float(idx) + 0.5) * extent / len(demand)
                corridors.append((axis, center, width, importance))
                if len(chosen) >= limit:
                    break

        per_axis = max(1, min(2, int(max_corridors)))
        pick(x_demand, "v", cw, per_axis)
        pick(y_demand, "h", ch, per_axis)
        if not corridors:
            return None
        corridors.sort(key=lambda item: item[3], reverse=True)
        corridors = corridors[: max(1, int(max_corridors))]

        x_centers = [center for axis, center, _width, _importance in corridors if axis == "v"]
        x_widths = [width for axis, _center, width, _importance in corridors if axis == "v"]
        x_weights = [importance for axis, _center, _width, importance in corridors if axis == "v"]
        y_centers = [center for axis, center, _width, _importance in corridors if axis == "h"]
        y_widths = [width for axis, _center, width, _importance in corridors if axis == "h"]
        y_weights = [importance for axis, _center, _width, importance in corridors if axis == "h"]

        area = benchmark.macro_sizes[:num_macros].numpy().astype(np.float64)
        macro_area = np.maximum(area[:, 0] * area[:, 1], 1.0e-9)
        area_norm = macro_area / max(float(np.mean(macro_area)), 1.0e-9)
        degree_norm = degree / max(float(np.percentile(degree[degree > 0.0], 90)) if np.any(degree > 0.0) else 1.0, 1.0e-9)
        macro_weight = np.clip(np.sqrt(area_norm) * (0.70 + float(bridge_boost) * np.clip(degree_norm, 0.0, 2.0)), 0.25, 3.8)
        macro_weight[benchmark.macro_fixed[:num_macros].numpy().astype(bool)] = 0.0

        def tensor(values: List[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=dtype, device=device)

        log = ",".join(
            f"{axis}@{center / (cw if axis == 'v' else ch):.3f}:w={width / span:.3f}:imp={importance:.2f}"
            for axis, center, width, importance in corridors
        )
        active_degree = degree[degree > 0.0]
        macro_stats = (
            f"degree_p90={float(np.percentile(active_degree, 90)) if active_degree.size else 0.0:.3f},"
            f"degree_max={float(np.max(active_degree)) if active_degree.size else 0.0:.3f},"
            f"weight_mean={float(np.mean(macro_weight)):.3f},"
            f"weight_p95={float(np.percentile(macro_weight, 95)):.3f}"
        )
        return {
            "x_centers": tensor(x_centers),
            "x_widths": tensor(x_widths),
            "x_weights": tensor(x_weights),
            "y_centers": tensor(y_centers),
            "y_widths": tensor(y_widths),
            "y_weights": tensor(y_weights),
            "macro_weight": torch.tensor(macro_weight, dtype=dtype, device=device),
            "log": log,
            "macro_stats": macro_stats,
        }

    def _soft_global_corridor_blockage(
        self,
        pos: torch.Tensor,
        sizes: torch.Tensor,
        movable: torch.Tensor,
        corridor_plan,
        cw: float,
        ch: float,
    ) -> torch.Tensor:
        if corridor_plan is None:
            return pos.sum() * 0.0
        macro_weight = corridor_plan["macro_weight"].to(device=pos.device, dtype=pos.dtype)
        movable_weight = movable.to(dtype=pos.dtype) * macro_weight
        total = pos.sum() * 0.0
        norm = torch.clamp((sizes[:, 0] * sizes[:, 1] * movable_weight).sum(), min=1.0e-6)
        if corridor_plan["x_centers"].numel():
            centers = corridor_plan["x_centers"].to(device=pos.device, dtype=pos.dtype)
            widths = corridor_plan["x_widths"].to(device=pos.device, dtype=pos.dtype)
            weights = corridor_plan["x_weights"].to(device=pos.device, dtype=pos.dtype)
            starts = centers - widths * 0.5
            ends = centers + widths * 0.5
            ox = self._soft_global_axis_overlap(pos[:, 0], sizes[:, 0] * 0.5, starts, ends)
            blocked = ox * sizes[:, 1:2]
            total = total + (blocked * movable_weight[:, None] * weights[None, :]).sum()
        if corridor_plan["y_centers"].numel():
            centers = corridor_plan["y_centers"].to(device=pos.device, dtype=pos.dtype)
            widths = corridor_plan["y_widths"].to(device=pos.device, dtype=pos.dtype)
            weights = corridor_plan["y_weights"].to(device=pos.device, dtype=pos.dtype)
            starts = centers - widths * 0.5
            ends = centers + widths * 0.5
            oy = self._soft_global_axis_overlap(pos[:, 1], sizes[:, 1] * 0.5, starts, ends)
            blocked = oy * sizes[:, 0:1]
            total = total + (blocked * movable_weight[:, None] * weights[None, :]).sum()
        return total / norm

    def _soft_global_device(self) -> torch.device:
        if self.execution_mode_used in {"cuda_debug", "cuda_competition"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.execution_mode_used in {"local_dev", "small_cpu_soft_global"}:
            return torch.device("cpu")
        requested = str(self.soft_global_device).lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _soft_global_nets(self, benchmark: Benchmark, device: torch.device) -> List[Tuple[torch.Tensor, float]]:
        raw_nets = benchmark.net_pin_nodes if benchmark.net_pin_nodes else benchmark.net_nodes
        max_nets = 12000 if device.type == "cuda" else 1800
        if self.execution_mode_used == "cuda_debug":
            max_nets = max(1, int(self.soft_global_max_nets_debug))
        max_degree = 80 if device.type == "cuda" else 48
        nets: List[Tuple[torch.Tensor, float]] = []
        weights = benchmark.net_weights.tolist() if getattr(benchmark, "net_weights", None) is not None else []
        for net_id, owners_tensor in enumerate(raw_nets):
            owners = owners_tensor[:, 0] if owners_tensor.ndim == 2 else owners_tensor
            unique = sorted(set(int(x) for x in owners.tolist()))
            if len(unique) < 2:
                continue
            if len(unique) > max_degree:
                hard_soft = [x for x in unique if x < benchmark.num_macros]
                ports = [x for x in unique if x >= benchmark.num_macros]
                unique = (hard_soft[: max_degree - min(len(ports), 8)] + ports[:8])[:max_degree]
                if len(unique) < 2:
                    continue
            weight = float(weights[net_id]) if net_id < len(weights) else 1.0
            weight /= math.sqrt(max(1, len(unique) - 1))
            nets.append((torch.tensor(unique, dtype=torch.long, device=device), weight))
            if len(nets) >= max_nets:
                break
        return nets

    def _soft_global_hpwl(self, all_pos: torch.Tensor, nets, span: float) -> torch.Tensor:
        if not nets:
            return all_pos.sum() * 0.0
        gamma = max(span * 0.015, 1.0e-3)
        total = all_pos.sum() * 0.0
        for owners, weight in nets:
            pts = all_pos.index_select(0, owners)
            x = pts[:, 0] / gamma
            y = pts[:, 1] / gamma
            hpwl = gamma * (
                torch.logsumexp(x, dim=0)
                + torch.logsumexp(-x, dim=0)
                + torch.logsumexp(y, dim=0)
                + torch.logsumexp(-y, dim=0)
            )
            total = total + float(weight) * hpwl
        return total / (len(nets) * max(span, 1.0e-6))

    def _soft_global_bin_edges(self, cw, ch, rows, cols, device, dtype):
        x_edges = torch.linspace(0.0, float(cw), int(cols) + 1, device=device, dtype=dtype)
        y_edges = torch.linspace(0.0, float(ch), int(rows) + 1, device=device, dtype=dtype)
        return x_edges[:-1], x_edges[1:], y_edges[:-1], y_edges[1:]

    def _soft_global_axis_overlap(self, center, half, starts, ends):
        lo = center[:, None] - half[:, None]
        hi = center[:, None] + half[:, None]
        left = torch.maximum(lo, starts[None, :])
        right = torch.minimum(hi, ends[None, :])
        return torch.relu(right - left)

    def _soft_global_density(self, pos, sizes, cw, ch, rows, cols, bin_edges, target_scale):
        x0, x1, y0, y1 = bin_edges
        ox = self._soft_global_axis_overlap(pos[:, 0], sizes[:, 0] * 0.5, x0, x1)
        oy = self._soft_global_axis_overlap(pos[:, 1], sizes[:, 1] * 0.5, y0, y1)
        grid_area = oy.transpose(0, 1).matmul(ox)
        cell_area = (float(cw) / int(cols)) * (float(ch) / int(rows))
        util = grid_area / max(cell_area, 1.0e-6)
        total_area = (sizes[:, 0] * sizes[:, 1]).sum()
        target = torch.clamp(
            total_area / max(float(cw) * float(ch), 1.0e-6) * float(target_scale),
            min=0.58,
            max=0.92,
        )
        overflow = torch.relu(util - target)
        return overflow.pow(2).mean()

    def _soft_global_congestion(self, all_pos, nets, cw, ch, rows, cols, bin_edges, target_scale):
        if not nets:
            return all_pos.sum() * 0.0
        x0, x1, y0, y1 = bin_edges
        grid = torch.zeros(int(rows), int(cols), dtype=all_pos.dtype, device=all_pos.device)
        gamma = max(max(cw, ch) * 0.020, 1.0e-3)
        if all_pos.device.type == "cuda":
            max_nets = min(len(nets), 5000)
        else:
            max_nets = min(len(nets), 120 if all_pos.shape[0] >= 1800 else 1000)
        for owners, weight in nets[:max_nets]:
            pts = all_pos.index_select(0, owners)
            xmax = gamma * torch.logsumexp(pts[:, 0] / gamma, dim=0)
            xmin = -gamma * torch.logsumexp(-pts[:, 0] / gamma, dim=0)
            ymax = gamma * torch.logsumexp(pts[:, 1] / gamma, dim=0)
            ymin = -gamma * torch.logsumexp(-pts[:, 1] / gamma, dim=0)
            width = torch.clamp(xmax - xmin, min=gamma)
            height = torch.clamp(ymax - ymin, min=gamma)
            bbox_area = torch.clamp(width * height, min=gamma * gamma)
            ox = torch.relu(torch.minimum(xmax, x1) - torch.maximum(xmin, x0))
            oy = torch.relu(torch.minimum(ymax, y1) - torch.maximum(ymin, y0))
            route_len = width + height
            demand = float(weight) * route_len / max(max(cw, ch), 1.0e-6)
            grid = grid + oy[:, None] * ox[None, :] / bbox_area * demand
        active = grid[grid > 0]
        target = active.mean() * float(target_scale) if active.numel() else grid.sum() * 0.0 + 1.0
        return torch.relu(grid - target).pow(2).mean() / (target.detach().pow(2) + 1.0e-6)

    def _soft_global_soft_displacement(self, pos, init, sizes, soft_movable, span):
        if not bool(soft_movable.any()):
            return pos.sum() * 0.0
        disp = torch.linalg.norm(pos[soft_movable] - init[soft_movable], dim=1) / max(float(span), 1.0e-6)
        area = sizes[soft_movable, 0] * sizes[soft_movable, 1]
        area_weight = torch.sqrt(area / torch.clamp(area.mean().detach(), min=1.0e-9))
        return (disp.pow(2) * area_weight).mean()

    def _soft_splat(self, points, values, cw, ch, rows, cols):
        x = torch.clamp(points[:, 0] / max(cw, 1.0e-6) * (cols - 1), 0.0, cols - 1.0001)
        y = torch.clamp(points[:, 1] / max(ch, 1.0e-6) * (rows - 1), 0.0, rows - 1.0001)
        c0 = torch.floor(x).long()
        r0 = torch.floor(y).long()
        c1 = torch.clamp(c0 + 1, max=cols - 1)
        r1 = torch.clamp(r0 + 1, max=rows - 1)
        tx = x - c0.to(x.dtype)
        ty = y - r0.to(y.dtype)
        weights = (
            ((1 - tx) * (1 - ty), r0, c0),
            (tx * (1 - ty), r0, c1),
            ((1 - tx) * ty, r1, c0),
            (tx * ty, r1, c1),
        )
        flat = torch.zeros(rows * cols, dtype=points.dtype, device=points.device)
        for w, r, c in weights:
            flat.scatter_add_(0, r * cols + c, values * w)
        return flat.view(rows, cols)

    def _soft_global_hard_overlap(self, hard_pos, hard_sizes, span):
        n = hard_pos.shape[0]
        if n <= 1:
            return hard_pos.sum() * 0.0
        dx = torch.abs(hard_pos[:, None, 0] - hard_pos[None, :, 0])
        dy = torch.abs(hard_pos[:, None, 1] - hard_pos[None, :, 1])
        sep_x = (hard_sizes[:, None, 0] + hard_sizes[None, :, 0]) / 2.0
        sep_y = (hard_sizes[:, None, 1] + hard_sizes[None, :, 1]) / 2.0
        ox = torch.relu(sep_x - dx)
        oy = torch.relu(sep_y - dy)
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=hard_pos.device), diagonal=1)
        overlap = (ox * oy)[mask]
        return overlap.pow(2).mean() / max(span**4, 1.0e-6)

    def _soft_global_boundary(self, pos, half, cw, ch, span):
        left = torch.relu(half[:, 0] - pos[:, 0])
        right = torch.relu(pos[:, 0] - (cw - half[:, 0]))
        bottom = torch.relu(half[:, 1] - pos[:, 1])
        top = torch.relu(pos[:, 1] - (ch - half[:, 1]))
        return (left.pow(2) + right.pow(2) + bottom.pow(2) + top.pow(2)).mean() / max(span**2, 1.0e-6)

    def _full_candidate_from_parts(self, benchmark: Benchmark, hard: np.ndarray, soft: np.ndarray) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        n_hard = benchmark.num_hard_macros
        placement[:n_hard] = torch.tensor(hard, dtype=torch.float32)
        if benchmark.num_macros > n_hard and len(soft):
            placement[n_hard : benchmark.num_macros] = torch.tensor(soft, dtype=torch.float32)
        placement = self._repair_soft_bounds_tensor(placement, benchmark)
        placement = self._repair_hard_bounds_tensor(placement, benchmark)
        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return placement

    def _owner_positions_from_placement(self, placement: torch.Tensor, benchmark: Benchmark) -> np.ndarray:
        macro_pos = placement[: benchmark.num_macros].numpy().astype(np.float64)
        if benchmark.port_positions.shape[0] == 0:
            return macro_pos
        return np.vstack([macro_pos, benchmark.port_positions.numpy().astype(np.float64)])

    def _run_profile_sweep(
        self,
        benchmark: Benchmark,
        initial_hard: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
        iterations: int,
        soft_neighbors,
    ) -> Tuple[List[Candidate], List[Candidate]]:
        specs = self._generate_profile_sweep_specs(benchmark, edges)
        self.num_profiles_evaluated = len(specs)
        if not specs:
            return [], []

        plc = _load_plc_for_exact(benchmark.name) if self.exact_final_select else None
        scored = []
        score_log = []
        for spec in specs:
            label = str(spec["label"])
            params = spec["params"]
            self._profile_params_by_label[label] = self._profile_spec_text(params)

            pos = self._legalize(initial_hard.copy(), movable, sizes, half_w, half_h, cw, ch, benchmark.num_hard_macros)
            pos = self._analytical_global_place(
                pos=pos,
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                owner_pos=owner_pos,
                benchmark=benchmark,
                profile=params,
            )
            pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
            pos = self._clip_hard_np(pos, benchmark)
            surrogate = self._surrogate_cost(pos, edges, owner_pos, benchmark, sizes)
            exact = self._score_exact_hard_position(pos, label, benchmark, soft_neighbors, plc)
            rank_score = exact["score"] if exact is not None else surrogate
            scored.append((rank_score, surrogate, pos, label, exact, params))
            if exact is not None:
                score_log.append(f"{label}|p={exact['raw_proxy']:.6f}|ov={int(exact['overlaps'])}|{self._profile_spec_text(params)}")
            else:
                score_log.append(f"{label}|sur={surrogate:.6f}|{self._profile_spec_text(params)}")

        scored.sort(key=lambda row: row[0])
        if benchmark.num_hard_macros >= 650 or len(edges) >= 4500:
            top_k = min(max(1, int(self.profile_sweep_top_k)), 4)
        elif benchmark.num_hard_macros >= 450:
            top_k = min(max(1, int(self.profile_sweep_top_k)), 5)
        else:
            top_k = max(1, int(self.profile_sweep_top_k))
        kept = scored[:top_k]
        candidates = [(surrogate, pos, True, label, None) for _score, surrogate, pos, label, _exact, _params in kept]

        polished = []
        polish_k = max(0, int(self.profile_sweep_polish_top_k))
        if benchmark.num_hard_macros >= 650 or len(edges) >= 4500:
            polish_k = 0
        refine_scale = 0.24 if benchmark.num_hard_macros < 450 else 0.16
        if benchmark.num_hard_macros >= 650:
            refine_scale = 0.055
        polish_iters = max(300, int(iterations * refine_scale))
        for _score, _surrogate, pos, label, _exact, params in kept[:polish_k]:
            polished_label = f"{label}_polished"
            self._profile_params_by_label[polished_label] = (
                self._profile_spec_text(params) + f",polish_iters={polish_iters}"
            )
            refined = self._refine(
                pos=pos.copy(),
                movable=movable,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                benchmark=benchmark,
                iterations=polish_iters,
                rng=random.Random(self.base_seed + 501 + len(polished)),
                np_rng=np.random.default_rng(self.base_seed + 501 + len(polished)),
            )
            refined = self._repair_all_overlaps(refined, movable, sizes, half_w, half_h, cw, ch)
            refined = self._clip_hard_np(refined, benchmark)
            surrogate = self._surrogate_cost(refined, edges, owner_pos, benchmark, sizes)
            exact = self._score_exact_hard_position(refined, polished_label, benchmark, soft_neighbors, plc)
            polished.append((surrogate, refined, True, polished_label, None))
            if exact is not None:
                score_log.append(
                    f"{polished_label}|p={exact['raw_proxy']:.6f}|ov={int(exact['overlaps'])}|"
                    f"{self._profile_params_by_label[polished_label]}"
                )
            else:
                score_log.append(f"{polished_label}|sur={surrogate:.6f}|{self._profile_params_by_label[polished_label]}")

        self.profile_sweep_scores = ";".join(score_log)
        return candidates, polished

    def _generate_profile_sweep_specs(self, benchmark: Benchmark, edges: List[Edge]) -> List[Dict[str, object]]:
        n = max(1, int(benchmark.num_hard_macros))
        canvas_area = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1.0e-12)
        macro_area = float(
            (benchmark.macro_sizes[: benchmark.num_macros, 0] * benchmark.macro_sizes[: benchmark.num_macros, 1])
            .sum()
            .item()
        )
        util = macro_area / canvas_area
        avg_degree = (2.0 * len(edges) / n) if n else 0.0
        dense_bias = float(np.clip((util - 0.74) / 0.10, 0.0, 1.0))
        edge_bias = float(np.clip((avg_degree - 5.0) / 9.0, 0.0, 1.0))
        large_bias = float(np.clip((n - 350.0) / 450.0, 0.0, 1.0))

        base_a = 0.48 + 0.16 * edge_bias
        base_r = 0.48 + 0.06 * dense_bias
        base_d = 1.02 + 0.22 * dense_bias
        base_c = 0.15 + 0.72 * edge_bias
        base_step = 0.58 - 0.14 * large_bias - 0.08 * dense_bias
        base_target = 1.00 - 0.040 * dense_bias
        base_iters = self.analytical_stage_iters
        if n >= 650:
            base_iters = (7, 10, 5)
        elif n >= 450:
            base_iters = (10, 14, 6)

        fixed_specs = [
            (
                "known_default",
                {
                    "a_scale": 0.55,
                    "r_scale": 0.55,
                    "d_scale": 1.15,
                    "c_scale": 0.00,
                    "step_scale": 0.55,
                    "target_scale": 0.96,
                    "b_scale": 0.85,
                    "stage_iters": tuple(int(x) for x in base_iters),
                    "repair_every": int(64 + 24 * large_bias),
                    "momentum": 0.45,
                },
            ),
            (
                "known_congestion",
                {
                    "a_scale": 0.50,
                    "r_scale": 0.50,
                    "d_scale": 1.10,
                    "c_scale": 0.80,
                    "step_scale": 0.50,
                    "target_scale": 0.96,
                    "b_scale": 0.85,
                    "stage_iters": tuple(int(x) for x in base_iters),
                    "repair_every": int(64 + 24 * large_bias),
                    "momentum": 0.45,
                },
            ),
            (
                "known_gentle_large",
                {
                    "a_scale": 0.42,
                    "r_scale": 0.42,
                    "d_scale": 1.05,
                    "c_scale": 0.45,
                    "step_scale": 0.42,
                    "target_scale": 0.98,
                    "b_scale": 0.80,
                    "stage_iters": tuple(int(x) for x in base_iters),
                    "repair_every": int(64 + 24 * large_bias),
                    "momentum": 0.45,
                },
            ),
        ]

        variants = [
            ("balanced", 1.00, 1.00, 1.00, 0.20, 1.00, 1.00, 1.00, 64, 0.42),
            ("dense", 0.92, 1.03, 1.16, 0.25, 0.88, 0.96, 1.00, 48, 0.36),
            ("dense_lowstep", 0.86, 1.00, 1.24, 0.20, 0.74, 0.94, 1.05, 40, 0.32),
            ("cong", 1.02, 0.95, 1.05, 0.85, 0.86, 0.98, 1.00, 56, 0.36),
            ("cong_dense", 0.96, 0.98, 1.18, 1.05, 0.78, 0.95, 1.05, 48, 0.34),
            ("cong_gentle", 0.88, 0.90, 1.08, 0.75, 0.66, 0.98, 0.95, 80, 0.30),
            ("attract", 1.18, 0.90, 0.96, 0.35, 0.86, 1.02, 0.92, 72, 0.40),
            ("attract_cong", 1.22, 0.88, 1.00, 0.90, 0.76, 1.00, 0.92, 72, 0.36),
            ("gentle", 0.82, 0.82, 1.08, 0.35, 0.62, 0.99, 0.88, 96, 0.28),
            ("gentle_dense", 0.78, 0.82, 1.20, 0.45, 0.58, 0.96, 0.92, 80, 0.26),
            ("spread", 0.72, 1.18, 1.12, 0.20, 0.70, 0.98, 1.12, 40, 0.30),
            ("spread_cong", 0.78, 1.08, 1.10, 0.95, 0.66, 0.98, 1.02, 48, 0.30),
            ("target_lo", 0.90, 0.95, 1.22, 0.55, 0.72, 0.92, 1.00, 56, 0.32),
            ("target_hi", 1.00, 0.88, 0.95, 0.35, 0.82, 1.06, 0.90, 80, 0.38),
            ("aggressive_small", 1.10, 1.05, 1.06, 0.60, 1.05, 0.98, 1.00, 48, 0.46),
            ("large_stable", 0.76, 0.78, 1.14, 0.55, 0.50, 0.98, 0.84, 112, 0.22),
            ("edge_heavy", 1.28, 0.78, 1.02, 1.10, 0.64, 1.00, 0.86, 80, 0.30),
            ("density_wall", 0.70, 1.12, 1.34, 0.30, 0.54, 0.90, 1.10, 40, 0.24),
        ]

        count = max(12, min(20, int(self.profile_sweep_count)))
        if n >= 650 or len(edges) >= 4500:
            count = 6
        elif n >= 450:
            count = 8
        specs = []
        for idx, (name, params) in enumerate(fixed_specs[:count]):
            specs.append({"label": f"profile_sweep_p{idx:02d}_{name}", "params": params})

        for raw_idx, (name, a_m, r_m, d_m, c_m, s_m, t_m, b_m, repair, momentum) in enumerate(
            variants[: max(0, count - len(specs))]
        ):
            idx = len(specs)
            step = float(np.clip(base_step * s_m, 0.24, 0.82))
            if n >= 650:
                step = min(step, 0.42)
            params = {
                "a_scale": float(np.clip(base_a * a_m, 0.25, 1.10)),
                "r_scale": float(np.clip(base_r * r_m, 0.20, 1.30)),
                "d_scale": float(np.clip(base_d * d_m, 0.00, 1.65)),
                "c_scale": float(np.clip(base_c * c_m, 0.00, 1.25)),
                "step_scale": step,
                "target_scale": float(np.clip(base_target * t_m, 0.86, 1.08)),
                "b_scale": float(np.clip(0.85 * b_m, 0.55, 1.30)),
                "stage_iters": tuple(int(x) for x in base_iters),
                "repair_every": int(max(24, min(128, repair + int(24 * large_bias)))),
                "momentum": float(np.clip(momentum - 0.06 * large_bias, 0.18, 0.48)),
            }
            specs.append({"label": f"profile_sweep_p{idx:02d}_{name}", "params": params})
        return specs

    def _profile_spec_text(self, params: object) -> str:
        if not isinstance(params, dict):
            return str(params)
        iters = params.get("stage_iters", self.analytical_stage_iters)
        return (
            f"a={float(params.get('a_scale', 0.0)):.3f},"
            f"r={float(params.get('r_scale', 0.0)):.3f},"
            f"d={float(params.get('d_scale', 0.0)):.3f},"
            f"c={float(params.get('c_scale', 0.0)):.3f},"
            f"s={float(params.get('step_scale', 0.0)):.3f},"
            f"t={float(params.get('target_scale', 0.0)):.3f},"
            f"b={float(params.get('b_scale', 0.0)):.3f},"
            f"it={'/'.join(str(int(x)) for x in iters)},"
            f"rep={int(params.get('repair_every', self.analytical_repair_every))},"
            f"mom={float(params.get('momentum', self.analytical_momentum)):.2f}"
        )

    def _selected_profile_param_text(self, selected_label: str) -> str:
        base = selected_label.split("+", 1)[0]
        return self._profile_params_by_label.get(base, "")

    def _score_exact_hard_position(self, hard_pos, label, benchmark, soft_neighbors, plc):
        if plc is None:
            return None
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return None

        best = None
        hard_pos = self._clip_hard_np(hard_pos.copy(), benchmark)
        soft_options = (False, True) if self.use_soft_motion else (False,)
        bounds_options = (True,) if self.use_soft_bounds_repair else (False,)
        for use_soft in soft_options:
            for repair_soft_bounds in bounds_options:
                placement = benchmark.macro_positions.clone()
                placement[: benchmark.num_hard_macros] = torch.tensor(hard_pos, dtype=torch.float32)
                if use_soft:
                    placement = self._place_soft_macros(placement, benchmark, soft_neighbors)
                if repair_soft_bounds:
                    placement = self._repair_soft_bounds_tensor(placement, benchmark)
                placement = self._repair_hard_bounds_tensor(placement, benchmark)
                try:
                    costs, exact_elapsed = self._compute_proxy_cost_timed(
                        compute_proxy_cost, placement, benchmark, plc, label
                    )
                except Exception:
                    continue
                overlaps = int(costs.get("overlap_count", 999999))
                raw_proxy = float(costs["proxy_cost"])
                score = raw_proxy + overlaps * 1.0e6
                suffixes = []
                if use_soft:
                    suffixes.append("soft")
                if repair_soft_bounds:
                    suffixes.append("soft_bounds")
                scored_label = label + (f"+{'+'.join(suffixes)}" if suffixes else "")
                row = {
                    "score": score,
                    "raw_proxy": raw_proxy,
                    "wirelength": float(costs.get("wirelength_cost", float("nan"))),
                    "density": float(costs.get("density_cost", float("nan"))),
                    "congestion": float(costs.get("congestion_cost", float("nan"))),
                    "overlaps": overlaps,
                    "label": scored_label,
                    "exact_elapsed": exact_elapsed,
                }
                if best is None or row["score"] < best["score"]:
                    best = row
        return best

    def _exact_proxy_polish(
        self,
        selected: np.ndarray,
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
        soft_neighbors,
    ) -> np.ndarray:
        plc = _load_plc_for_exact(benchmark.name)
        if plc is None or benchmark.num_hard_macros == 0:
            return selected
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return selected

        n = benchmark.num_hard_macros
        pos = self._clip_hard_np(selected.copy(), benchmark)
        fixed_hard = benchmark.macro_fixed[:n].numpy()
        if fixed_hard.any():
            pos[fixed_hard] = benchmark.macro_positions[:n].numpy().astype(np.float64)[fixed_hard]

        def exact_score(hard_pos):
            placement = benchmark.macro_positions.clone()
            placement[:n] = torch.tensor(hard_pos, dtype=torch.float32)
            if self.use_soft_motion and self._selected_use_soft:
                placement = self._place_soft_macros(placement, benchmark, soft_neighbors)
            if self.use_soft_bounds_repair:
                placement = self._repair_soft_bounds_tensor(placement, benchmark)
            placement = self._repair_hard_bounds_tensor(placement, benchmark)
            if benchmark.macro_fixed.any():
                placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            costs, _exact_elapsed = self._compute_proxy_cost_timed(
                compute_proxy_cost, placement, benchmark, plc, "exact_polish"
            )
            overlaps = int(costs.get("overlap_count", 999999))
            raw_proxy = float(costs["proxy_cost"])
            return raw_proxy + overlaps * 1.0e6, raw_proxy, overlaps

        try:
            current_score, current_proxy, current_overlaps = exact_score(pos)
        except Exception:
            return selected
        if current_overlaps:
            return selected

        self.exact_polish_start_proxy = f"{current_proxy:.6f}"
        self.exact_polish_end_proxy = f"{current_proxy:.6f}"

        candidate_ids = self._exact_polish_candidate_ids(pos, benchmark, movable, sizes, edges, incident, owner_pos)
        if not candidate_ids:
            return pos

        max_moves = max(0, int(self.exact_polish_max_moves))
        if n >= 650 or len(edges) >= 4500:
            max_moves = min(max_moves, 4)
        elif n >= 450:
            max_moves = min(max_moves, 6)
        if max_moves <= 0:
            return pos

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        tried = 0
        accepted = 0
        span = max(cw, ch)
        step_scales = tuple(float(x) for x in self.exact_polish_step_scales)
        hard_neighbor_weights = self._hard_neighbor_weights(edges, n)

        for i in candidate_ids:
            if tried >= max_moves:
                break
            proposals = self._exact_polish_proposals(
                i=i,
                pos=pos,
                movable=movable,
                half_w=half_w,
                half_h=half_h,
                cw=cw,
                ch=ch,
                edges=edges,
                incident=incident,
                owner_pos=owner_pos,
                hard_neighbor_weights=hard_neighbor_weights,
                step_scales=step_scales,
                span=span,
            )
            for updates in proposals:
                if tried >= max_moves:
                    break
                changed_ids = [idx for idx, _new in updates]
                old = [(idx, pos[idx].copy()) for idx in changed_ids]
                for idx, new_xy in updates:
                    pos[idx] = new_xy
                pos = self._clip_hard_np(pos, benchmark)
                if fixed_hard.any():
                    pos[fixed_hard] = benchmark.macro_positions[:n].numpy().astype(np.float64)[fixed_hard]
                if self._any_overlap(pos, changed_ids, sep_x, sep_y, gap=0.025):
                    for idx, old_xy in old:
                        pos[idx] = old_xy
                    continue

                tried += 1
                try:
                    score, raw_proxy, overlaps = exact_score(pos)
                except Exception:
                    for idx, old_xy in old:
                        pos[idx] = old_xy
                    continue
                if overlaps == 0 and score < current_score - 1.0e-7:
                    current_score = score
                    current_proxy = raw_proxy
                    accepted += 1
                else:
                    for idx, old_xy in old:
                        pos[idx] = old_xy

        self.exact_polish_accepted_moves = accepted
        self.exact_polish_end_proxy = f"{current_proxy:.6f}"
        if accepted:
            self.selected_candidate = f"{self.selected_candidate}+exact_polish{accepted}"
        return pos

    def _exact_polish_candidate_ids(
        self,
        pos: np.ndarray,
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
    ) -> List[int]:
        n = benchmark.num_hard_macros
        movable_ids = np.where(movable[:n])[0]
        if len(movable_ids) == 0:
            return []

        degree_score = np.zeros(n, dtype=np.float64)
        wire_score = np.zeros(n, dtype=np.float64)
        for edge_id, (a, b, w) in enumerate(edges):
            pa = pos[a] if a < n else owner_pos[a]
            pb = pos[b] if b < n else owner_pos[b]
            dist = abs(float(pa[0] - pb[0])) + abs(float(pa[1] - pb[1]))
            if a < n:
                degree_score[a] += w
                wire_score[a] += w * dist
            if b < n:
                degree_score[b] += w
                wire_score[b] += w * dist

        dense_score = self._macro_dense_bin_scores(pos, benchmark, sizes)

        def norm(x):
            mx = float(np.max(x)) if x.size else 0.0
            return x / mx if mx > 1.0e-12 else x

        priority = 1.10 * norm(degree_score) + 0.95 * norm(wire_score) + 1.25 * norm(dense_score)
        ordered = sorted((int(i) for i in movable_ids), key=lambda idx: priority[idx], reverse=True)
        cap = max(1, int(self.exact_polish_candidate_macros))
        if n >= 650 or len(edges) >= 4500:
            cap = min(cap, 8)
        elif n >= 450:
            cap = min(cap, 12)
        return ordered[:cap]

    def _macro_dense_bin_scores(self, pos: np.ndarray, benchmark: Benchmark, hard_sizes: np.ndarray) -> np.ndarray:
        n = benchmark.num_hard_macros
        rows = max(8, min(24, int(benchmark.grid_rows)))
        cols = max(8, min(24, int(benchmark.grid_cols)))
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cell_area = (cw / cols) * (ch / rows)
        grid = np.zeros((rows, cols), dtype=np.float64)

        macro_pos = benchmark.macro_positions[: benchmark.num_macros].numpy().astype(np.float64)
        macro_sizes = benchmark.macro_sizes[: benchmark.num_macros].numpy().astype(np.float64)
        macro_pos[:n] = pos
        for p, s in zip(macro_pos, macro_sizes):
            r = int(np.clip(math.floor(float(p[1]) / ch * rows), 0, rows - 1))
            c = int(np.clip(math.floor(float(p[0]) / cw * cols), 0, cols - 1))
            grid[r, c] += (float(s[0]) * float(s[1])) / max(cell_area, 1.0e-12)

        target = max(0.72, float(np.mean(grid)) * 1.15)
        overflow = np.maximum(0.0, grid - target)
        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            r = int(np.clip(math.floor(float(pos[i, 1]) / ch * rows), 0, rows - 1))
            c = int(np.clip(math.floor(float(pos[i, 0]) / cw * cols), 0, cols - 1))
            scores[i] = overflow[r, c] * (hard_sizes[i, 0] * hard_sizes[i, 1])
        return scores

    def _exact_polish_proposals(
        self,
        i: int,
        pos: np.ndarray,
        movable: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
        hard_neighbor_weights: List[List[int]],
        step_scales: Tuple[float, ...],
        span: float,
    ) -> List[List[Tuple[int, np.ndarray]]]:
        proposals: List[List[Tuple[int, np.ndarray]]] = []
        for scale in step_scales:
            step = span * scale
            for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                new = np.array(
                    [
                        np.clip(pos[i, 0] + dx, half_w[i], cw - half_w[i]),
                        np.clip(pos[i, 1] + dy, half_h[i], ch - half_h[i]),
                    ],
                    dtype=np.float64,
                )
                if np.linalg.norm(new - pos[i]) > 1.0e-9:
                    proposals.append([(i, new)])

        if i < len(incident) and incident[i]:
            sx = sy = sw = 0.0
            for edge_id in incident[i]:
                a, b, w = edges[edge_id]
                other = b if a == i else a
                p = pos[other] if other < len(pos) else owner_pos[other]
                sx += w * p[0]
                sy += w * p[1]
                sw += w
            if sw > 0.0:
                target = np.array([sx / sw, sy / sw], dtype=np.float64)
                for alpha in (0.05, 0.10, 0.16):
                    new = pos[i] + alpha * (target - pos[i])
                    new[0] = np.clip(new[0], half_w[i], cw - half_w[i])
                    new[1] = np.clip(new[1], half_h[i], ch - half_h[i])
                    if np.linalg.norm(new - pos[i]) > 1.0e-9:
                        proposals.append([(i, new)])

        swap_candidates = []
        for j in hard_neighbor_weights[i][:6] if i < len(hard_neighbor_weights) else []:
            if 0 <= j < len(pos) and movable[j] and j != i:
                swap_candidates.append(int(j))
        if not swap_candidates:
            d = np.linalg.norm(pos - pos[i], axis=1)
            for j in np.argsort(d)[:8]:
                if int(j) != i and movable[int(j)]:
                    swap_candidates.append(int(j))
                    break
        for j in swap_candidates[:2]:
            new_i = np.array(
                [np.clip(pos[j, 0], half_w[i], cw - half_w[i]), np.clip(pos[j, 1], half_h[i], ch - half_h[i])],
                dtype=np.float64,
            )
            new_j = np.array(
                [np.clip(pos[i, 0], half_w[j], cw - half_w[j]), np.clip(pos[i, 1], half_h[j], ch - half_h[j])],
                dtype=np.float64,
            )
            proposals.append([(i, new_i), (j, new_j)])
        return proposals

    def _clip_hard_np(self, pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        """Clip hard macro centers just inside canvas bounds."""
        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return pos
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        eps = 1.0e-4
        low_x = sizes[:, 0] / 2.0 + eps
        high_x = float(benchmark.canvas_width) - sizes[:, 0] / 2.0 - eps
        low_y = sizes[:, 1] / 2.0 + eps
        high_y = float(benchmark.canvas_height) - sizes[:, 1] / 2.0 - eps
        pos[:n_hard, 0] = np.minimum(np.maximum(pos[:n_hard, 0], low_x), high_x)
        pos[:n_hard, 1] = np.minimum(np.maximum(pos[:n_hard, 1], low_y), high_y)
        return pos

    def _repair_hard_bounds_tensor(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Final hard-macro bounds guard after float32 conversion."""
        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return placement
        out = placement.clone()
        sizes = benchmark.macro_sizes[:n_hard]
        eps = torch.tensor(1.0e-4, dtype=out.dtype, device=out.device)
        low_x = sizes[:, 0].to(out.device, out.dtype) / 2 + eps
        high_x = torch.tensor(benchmark.canvas_width, dtype=out.dtype, device=out.device) - sizes[:, 0].to(out.device, out.dtype) / 2 - eps
        low_y = sizes[:, 1].to(out.device, out.dtype) / 2 + eps
        high_y = torch.tensor(benchmark.canvas_height, dtype=out.dtype, device=out.device) - sizes[:, 1].to(out.device, out.dtype) / 2 - eps
        out[:n_hard, 0] = torch.minimum(torch.maximum(out[:n_hard, 0], low_x), high_x)
        out[:n_hard, 1] = torch.minimum(torch.maximum(out[:n_hard, 1], low_y), high_y)
        return out

    def _repair_soft_bounds_tensor(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """Clip movable soft macro centers into canvas bounds without connectivity motion."""
        n_hard = benchmark.num_hard_macros
        n_macros = benchmark.num_macros
        if n_hard >= n_macros:
            return placement

        out = placement.clone()
        soft_slice = slice(n_hard, n_macros)
        sizes = benchmark.macro_sizes[soft_slice].to(out.device, out.dtype)
        eps = torch.tensor(1.0e-4, dtype=out.dtype, device=out.device)
        low_x = sizes[:, 0] / 2 + eps
        high_x = torch.tensor(benchmark.canvas_width, dtype=out.dtype, device=out.device) - sizes[:, 0] / 2 - eps
        low_y = sizes[:, 1] / 2 + eps
        high_y = torch.tensor(benchmark.canvas_height, dtype=out.dtype, device=out.device) - sizes[:, 1] / 2 - eps

        soft_fixed = benchmark.macro_fixed[soft_slice].to(out.device)
        clipped_x = torch.minimum(torch.maximum(out[soft_slice, 0], low_x), high_x)
        clipped_y = torch.minimum(torch.maximum(out[soft_slice, 1], low_y), high_y)
        movable_soft = ~soft_fixed
        out[soft_slice, 0] = torch.where(movable_soft, clipped_x, out[soft_slice, 0])
        out[soft_slice, 1] = torch.where(movable_soft, clipped_y, out[soft_slice, 1])
        return out

    def _extract_hard_clique_edges(self, benchmark: Benchmark) -> Tuple[np.ndarray, np.ndarray]:
        n_hard = benchmark.num_hard_macros
        edge_dict: Dict[Tuple[int, int], float] = {}
        nets: Iterable[torch.Tensor]
        if benchmark.net_pin_nodes:
            nets = (pins[:, 0] for pins in benchmark.net_pin_nodes if pins.numel() > 0)
        else:
            nets = benchmark.net_nodes
        for owners_tensor in nets:
            hard = sorted(set(int(x) for x in owners_tensor.tolist() if 0 <= int(x) < n_hard))
            if len(hard) < 2:
                continue
            weight = 1.0 / max(1, len(hard) - 1)
            for i, a in enumerate(hard):
                for b in hard[i + 1 :]:
                    key = (a, b)
                    edge_dict[key] = edge_dict.get(key, 0.0) + weight
        if not edge_dict:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)
        edge_items = list(edge_dict.items())
        return (
            np.array([k for k, _v in edge_items], dtype=np.int64),
            np.array([v for _k, v in edge_items], dtype=np.float64),
        )

    def _legacy_sa_refine(
        self,
        pos,
        edges,
        edge_weights,
        movable,
        sizes,
        half_w,
        half_h,
        cw,
        ch,
        rng,
    ):
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0 or len(edges) == 0:
            return pos

        pos = pos.copy()
        n = len(pos)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        neighbors = [[] for _ in range(n)]
        for i, j in edges:
            neighbors[int(i)].append(int(j))
            neighbors[int(j)].append(int(i))

        def wl_cost():
            dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
            dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
            return float((edge_weights * (dx + dy)).sum())

        def check_single_overlap(idx):
            gap = 0.05
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            overlaps[idx] = False
            return bool(overlaps.any())

        current_cost = wl_cost()
        best_pos = pos.copy()
        best_cost = current_cost
        iterations = 3000
        t_start = max(cw, ch) * 0.15
        t_end = max(cw, ch) * 0.001

        for step in range(iterations):
            frac = step / iterations
            temp = t_start * (t_end / t_start) ** frac
            i = int(rng.choice(movable_idx))
            old_x, old_y = pos[i, 0], pos[i, 1]
            move = rng.random()

            if move < 0.5:
                shift = temp * (0.3 + 0.7 * (1.0 - frac))
                pos[i, 0] = np.clip(pos[i, 0] + rng.gauss(0.0, shift), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + rng.gauss(0.0, shift), half_h[i], ch - half_h[i])
                changed = [i]
                old_j = None
            elif move < 0.8:
                cands = [j for j in neighbors[i] if movable[j]] if neighbors[i] and rng.random() < 0.7 else []
                j = int(rng.choice(cands)) if cands else int(rng.choice(movable_idx))
                if i == j:
                    continue
                old_j = (j, pos[j, 0], pos[j, 1])
                pos[i, 0] = np.clip(pos[j, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[j, 1], half_h[i], ch - half_h[i])
                pos[j, 0] = np.clip(old_x, half_w[j], cw - half_w[j])
                pos[j, 1] = np.clip(old_y, half_h[j], ch - half_h[j])
                changed = [i, j]
            else:
                if not neighbors[i]:
                    continue
                j = int(rng.choice(neighbors[i]))
                alpha = rng.uniform(0.05, 0.3)
                pos[i, 0] = np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]), half_h[i], ch - half_h[i])
                changed = [i]
                old_j = None

            if any(check_single_overlap(idx) for idx in changed):
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                if old_j is not None:
                    j, jx, jy = old_j
                    pos[j, 0] = jx
                    pos[j, 1] = jy
                continue

            new_cost = wl_cost()
            delta = new_cost - current_cost
            if delta < 0.0 or rng.random() < math.exp(-delta / max(temp, 1.0e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_x
                pos[i, 1] = old_y
                if old_j is not None:
                    j, jx, jy = old_j
                    pos[j, 0] = jx
                    pos[j, 1] = jy

        return best_pos

    def _analytical_global_place(
        self,
        pos: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        owner_pos: np.ndarray,
        benchmark: Benchmark,
        profile: str = "light",
    ) -> np.ndarray:
        """Multi-stage continuous hard-macro placement with smooth spreading forces."""
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        pos = pos.copy()
        n = len(pos)
        velocity = np.zeros_like(pos)
        area = sizes[:, 0] * sizes[:, 1]
        span = max(cw, ch)

        if edges:
            edge_src = np.array([a for a, _b, _w in edges], dtype=np.int64)
            edge_dst = np.array([b for _a, b, _w in edges], dtype=np.int64)
            edge_w = np.array([w for _a, _b, w in edges], dtype=np.float64)
            src_hard = edge_src < n
            dst_hard = edge_dst < n
        else:
            edge_src = edge_dst = np.zeros(0, dtype=np.int64)
            edge_w = np.zeros(0, dtype=np.float64)
            src_hard = dst_hard = np.zeros(0, dtype=bool)

        hard_degree = np.ones(n, dtype=np.float64)
        if edge_w.size:
            np.add.at(hard_degree, edge_src[src_hard], edge_w[src_hard])
            np.add.at(hard_degree, edge_dst[dst_hard], edge_w[dst_hard])

        congestion_edge_ids = np.zeros(0, dtype=np.int64)
        congestion_incident: List[List[int]] = [[] for _ in range(n)]
        if edge_w.size and self.analytical_congestion_weight > 0.0:
            max_congestion_edges = min(edge_w.size, max(350, min(2200, 5 * n)))
            if edge_w.size > max_congestion_edges:
                congestion_edge_ids = np.argpartition(edge_w, -max_congestion_edges)[-max_congestion_edges:]
            else:
                congestion_edge_ids = np.arange(edge_w.size, dtype=np.int64)
            for compact_id, edge_id in enumerate(congestion_edge_ids):
                a = int(edge_src[edge_id])
                b = int(edge_dst[edge_id])
                if a < n:
                    congestion_incident[a].append(compact_id)
                if b < n:
                    congestion_incident[b].append(compact_id)

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        eye = np.eye(n, dtype=bool)

        d_rows = max(10, min(28, int(benchmark.grid_rows)))
        d_cols = max(10, min(28, int(benchmark.grid_cols)))
        cell_area = (cw / d_cols) * (ch / d_rows)

        def density_contrib(x, y, w, h):
            x0 = max(0.0, x - w / 2.0)
            x1 = min(cw, x + w / 2.0)
            y0 = max(0.0, y - h / 2.0)
            y1 = min(ch, y + h / 2.0)
            col0 = int(np.clip(math.floor(x0 / cw * d_cols), 0, d_cols - 1))
            col1 = int(np.clip(math.floor(max(x1 - 1.0e-9, 0.0) / cw * d_cols), 0, d_cols - 1))
            row0 = int(np.clip(math.floor(y0 / ch * d_rows), 0, d_rows - 1))
            row1 = int(np.clip(math.floor(max(y1 - 1.0e-9, 0.0) / ch * d_rows), 0, d_rows - 1))
            cell_w = cw / d_cols
            cell_h = ch / d_rows
            contrib = []
            for r in range(row0, row1 + 1):
                cy0 = r * cell_h
                cy1 = cy0 + cell_h
                oy = max(0.0, min(y1, cy1) - max(y0, cy0))
                if oy <= 0.0:
                    continue
                for c in range(col0, col1 + 1):
                    cx0 = c * cell_w
                    cx1 = cx0 + cell_w
                    ox = max(0.0, min(x1, cx1) - max(x0, cx0))
                    if ox > 0.0:
                        contrib.append((r, c, (ox * oy) / cell_area))
            return contrib

        soft_grid = np.zeros((d_rows, d_cols), dtype=np.float64)
        soft_pos = benchmark.macro_positions[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        soft_sizes = benchmark.macro_sizes[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        for p, s in zip(soft_pos, soft_sizes):
            for r, c, val in density_contrib(float(p[0]), float(p[1]), float(s[0]), float(s[1])):
                soft_grid[r, c] += val

        stage_iters = tuple(int(x) for x in self.analytical_stage_iters)
        if len(stage_iters) != 3:
            stage_iters = (18, 24, 12)
        repair_every_value = int(self.analytical_repair_every)
        momentum = float(self.analytical_momentum)
        if isinstance(profile, dict):
            profile_scale = (
                float(profile.get("a_scale", 0.55)),
                float(profile.get("r_scale", 0.55)),
                float(profile.get("d_scale", 1.15)),
                float(profile.get("b_scale", 0.85)),
                float(profile.get("step_scale", 0.55)),
                float(profile.get("target_scale", 0.96)),
                float(profile.get("c_scale", 0.00)),
            )
            custom_iters = profile.get("stage_iters")
            if custom_iters is not None:
                stage_iters = tuple(int(x) for x in custom_iters)
                if len(stage_iters) != 3:
                    stage_iters = (18, 24, 12)
            repair_every_value = int(profile.get("repair_every", repair_every_value))
            momentum = float(profile.get("momentum", momentum))
        else:
            profile_scale = {
                "density_guarded_default": (0.55, 0.55, 1.15, 0.85, 0.55, 0.96, 0.00),
                "density_guarded_congestion": (0.50, 0.50, 1.10, 0.85, 0.50, 0.96, 0.80),
                "density_guarded_gentle_large": (0.42, 0.42, 1.05, 0.80, 0.42, 0.98, 0.45),
                # Backward-compatible aliases for older ablation wrappers.
                "density_guarded": (0.55, 0.55, 1.15, 0.85, 0.55, 0.96, 0.00),
                "light": (0.75, 0.65, 0.70, 0.75, 0.70, 1.00, 0.00),
                "wl_only_light": (0.85, 0.35, 0.00, 0.70, 0.50, 1.04, 0.00),
            }.get(profile, (0.55, 0.55, 1.15, 0.85, 0.55, 0.96, 0.00))
        a_scale, r_scale, d_scale, b_scale, step_scale, target_scale, c_scale = profile_scale
        stages = (
            (stage_iters[0], 0.25 * a_scale, 0.85 * r_scale, 0.95 * d_scale, 0.95 * b_scale, 0.25 * c_scale, 0.010 * step_scale, 0.0040 * step_scale, 0.10),
            (stage_iters[1], 0.90 * a_scale, 0.45 * r_scale, 0.70 * d_scale, 0.70 * b_scale, 0.85 * c_scale, 0.0065 * step_scale, 0.0022 * step_scale, 0.07),
            (stage_iters[2], 0.55 * a_scale, 0.30 * r_scale, 0.90 * d_scale, 1.10 * b_scale, 0.70 * c_scale, 0.0032 * step_scale, 0.0012 * step_scale, 0.045),
        )
        repair_every = max(1, repair_every_value)
        c_grid_size = max(6, min(32, int(self.analytical_congestion_grid_size)))
        c_rows = c_cols = c_grid_size
        c_cell_w = cw / c_cols
        c_cell_h = ch / c_rows

        global_step = 0
        for iters, a_mul, r_mul, d_mul, b_mul, c_mul, step0, step1, halo_mul in stages:
            if iters <= 0:
                continue
            for step in range(iters):
                frac = step / max(1, iters - 1)
                forces = np.zeros_like(pos)
                owner_points = None

                if edge_w.size:
                    owner_points = owner_pos.copy()
                    owner_points[:n] = pos
                    pa = owner_points[edge_src]
                    pb = owner_points[edge_dst]
                    spring = (pb - pa) * edge_w[:, None]
                    np.add.at(forces, edge_src[src_hard], spring[src_hard])
                    np.add.at(forces, edge_dst[dst_hard], -spring[dst_hard])
                    forces /= hard_degree[:, None]
                    forces *= self.analytical_attraction_weight * a_mul

                dx = pos[:, 0:1] - pos[:, 0:1].T
                dy = pos[:, 1:2] - pos[:, 1:2].T
                abs_dx = np.abs(dx)
                abs_dy = np.abs(dy)
                halo = halo_mul * np.maximum(
                    np.maximum(sizes[:, 0:1], sizes[:, 1:2]),
                    np.maximum(sizes[:, 0:1].T, sizes[:, 1:2].T),
                )
                ox = sep_x + halo - abs_dx
                oy = sep_y + halo - abs_dy
                near = (ox > 0.0) & (oy > 0.0) & ~eye
                sign_x = np.where(dx >= 0.0, 1.0, -1.0)
                sign_y = np.where(dy >= 0.0, 1.0, -1.0)
                denom_x = np.maximum(sep_x + halo, 1.0e-9)
                denom_y = np.maximum(sep_y + halo, 1.0e-9)
                smooth = np.where(near, (ox / denom_x) * (oy / denom_y), 0.0)
                push_x = sign_x * smooth * ox
                push_y = sign_y * smooth * oy
                forces[:, 0] += self.analytical_repulsion_weight * r_mul * push_x.sum(axis=1)
                forces[:, 1] += self.analytical_repulsion_weight * r_mul * push_y.sum(axis=1)

                if d_mul > 0.0:
                    hard_grid = soft_grid.copy()
                    hard_contribs = []
                    for i in range(n):
                        contrib = density_contrib(pos[i, 0], pos[i, 1], sizes[i, 0], sizes[i, 1])
                        hard_contribs.append(contrib)
                        for r, c, val in contrib:
                            hard_grid[r, c] += val
                    target = max(0.74, float(np.mean(hard_grid)) * 1.10 * target_scale)
                    overflow = np.maximum(0.0, hard_grid - target)
                    cell_w = cw / d_cols
                    cell_h = ch / d_rows
                    density_force = np.zeros_like(pos)
                    for i, contrib in enumerate(hard_contribs):
                        fx = fy = total = 0.0
                        for r, c, val in contrib:
                            ov = overflow[r, c]
                            if ov <= 0.0:
                                continue
                            cx = (c + 0.5) * cell_w
                            cy = (r + 0.5) * cell_h
                            vx = pos[i, 0] - cx
                            vy = pos[i, 1] - cy
                            norm = max(math.hypot(vx, vy), 1.0)
                            weight = val * ov
                            fx += weight * vx / norm
                            fy += weight * vy / norm
                            total += weight
                        if total > 0.0:
                            density_force[i, 0] = fx / total * min(span, max(sizes[i, 0], sizes[i, 1]) * 3.0)
                            density_force[i, 1] = fy / total * min(span, max(sizes[i, 0], sizes[i, 1]) * 3.0)
                    forces += self.analytical_density_weight * d_mul * density_force

                if c_mul > 0.0 and congestion_edge_ids.size and owner_points is not None:
                    cong_grid = np.zeros((c_rows, c_cols), dtype=np.float64)
                    edge_cells: List[List[Tuple[int, int]]] = []
                    for edge_id in congestion_edge_ids:
                        pa = owner_points[int(edge_src[edge_id])]
                        pb = owner_points[int(edge_dst[edge_id])]
                        x0 = max(0.0, min(float(pa[0]), float(pb[0])))
                        x1 = min(cw, max(float(pa[0]), float(pb[0])))
                        y0 = max(0.0, min(float(pa[1]), float(pb[1])))
                        y1 = min(ch, max(float(pa[1]), float(pb[1])))
                        col0 = int(np.clip(math.floor(x0 / cw * c_cols), 0, c_cols - 1))
                        col1 = int(np.clip(math.floor(max(x1 - 1.0e-9, 0.0) / cw * c_cols), 0, c_cols - 1))
                        row0 = int(np.clip(math.floor(y0 / ch * c_rows), 0, c_rows - 1))
                        row1 = int(np.clip(math.floor(max(y1 - 1.0e-9, 0.0) / ch * c_rows), 0, c_rows - 1))
                        rows = range(row0, row1 + 1)
                        cols = range(col0, col1 + 1)
                        cell_count = (row1 - row0 + 1) * (col1 - col0 + 1)
                        if cell_count <= 32:
                            cells = [(r, c) for r in rows for c in cols]
                        else:
                            mid_r = (row0 + row1) // 2
                            mid_c = (col0 + col1) // 2
                            cells = [(mid_r, c) for c in cols]
                            cells.extend((r, mid_c) for r in rows)
                            if len(cells) > 32:
                                stride = max(1, math.ceil(len(cells) / 32))
                                cells = cells[::stride]
                            cells = sorted(set(cells))
                        if not cells:
                            cells = [(row0, col0)]
                        demand = float(edge_w[edge_id]) / max(1, len(cells))
                        for r, c in cells:
                            cong_grid[r, c] += demand
                        edge_cells.append(cells)

                    active = cong_grid[cong_grid > 0.0]
                    if active.size:
                        target = max(
                            float(np.mean(active)) * float(self.analytical_congestion_target_scale),
                            float(np.percentile(active, 72)),
                            1.0e-9,
                        )
                        overflow = np.maximum(0.0, (cong_grid - target) / target)
                        congestion_force = np.zeros_like(pos)
                        for i, compact_ids in enumerate(congestion_incident):
                            fx = fy = total = 0.0
                            for compact_id in compact_ids:
                                edge_id = int(congestion_edge_ids[compact_id])
                                for r, c in edge_cells[compact_id]:
                                    ov = overflow[r, c]
                                    if ov <= 0.0:
                                        continue
                                    cx = (c + 0.5) * c_cell_w
                                    cy = (r + 0.5) * c_cell_h
                                    vx = pos[i, 0] - cx
                                    vy = pos[i, 1] - cy
                                    norm = max(math.hypot(vx, vy), 1.0)
                                    weight = ov * float(edge_w[edge_id])
                                    fx += weight * vx / norm
                                    fy += weight * vy / norm
                                    total += weight
                            if total > 0.0:
                                limit = min(span, max(sizes[i, 0], sizes[i, 1]) * 2.0)
                                congestion_force[i, 0] = fx / total * limit
                                congestion_force[i, 1] = fy / total * limit
                        forces += self.analytical_congestion_weight * c_mul * congestion_force

                margin = span * 0.035
                left = pos[:, 0] - half_w
                right = cw - half_w - pos[:, 0]
                bottom = pos[:, 1] - half_h
                top = ch - half_h - pos[:, 1]
                boundary = np.zeros_like(pos)
                boundary[:, 0] += np.maximum(0.0, margin - left) / margin
                boundary[:, 0] -= np.maximum(0.0, margin - right) / margin
                boundary[:, 1] += np.maximum(0.0, margin - bottom) / margin
                boundary[:, 1] -= np.maximum(0.0, margin - top) / margin
                forces += self.analytical_boundary_weight * b_mul * boundary * span

                forces[~movable] = 0.0
                velocity = momentum * velocity + (1.0 - momentum) * forces
                velocity[~movable] = 0.0

                max_step = span * (step0 * (step1 / step0) ** frac)
                norms = np.linalg.norm(velocity, axis=1)
                scale = np.minimum(1.0, max_step / np.maximum(norms, 1.0e-12))
                pos[movable, 0] += velocity[movable, 0] * scale[movable]
                pos[movable, 1] += velocity[movable, 1] * scale[movable]
                pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
                pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

                global_step += 1
                if global_step % repair_every == 0:
                    pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)
                    velocity *= 0.10

        return pos

    def _force_directed_candidate(
        self,
        pos: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        owner_pos: np.ndarray,
        benchmark: Benchmark,
    ) -> np.ndarray:
        """Cheap analytical candidate: attractive net springs plus spreading forces."""
        movable_idx = np.where(movable)[0]
        iters = max(0, int(self.fd_iters))
        if len(movable_idx) == 0 or iters == 0 or not edges:
            return pos

        pos = pos.copy()
        n = len(pos)
        edge_src = np.array([a for a, _b, _w in edges], dtype=np.int64)
        edge_dst = np.array([b for _a, b, _w in edges], dtype=np.int64)
        edge_w = np.array([w for _a, _b, w in edges], dtype=np.float64)
        src_hard = edge_src < n
        dst_hard = edge_dst < n

        hard_degree = np.ones(n, dtype=np.float64)
        np.add.at(hard_degree, edge_src[src_hard], edge_w[src_hard])
        np.add.at(hard_degree, edge_dst[dst_hard], edge_w[dst_hard])

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        eye = np.eye(n, dtype=bool)
        area = sizes[:, 0] * sizes[:, 1]
        max_step0 = max(cw, ch) * 0.020
        max_step1 = max(cw, ch) * 0.003
        repair_every = max(10, iters // 2)

        d_rows = max(8, min(20, int(benchmark.grid_rows)))
        d_cols = max(8, min(20, int(benchmark.grid_cols)))
        cell_area = (cw / d_cols) * (ch / d_rows)
        soft_pos = benchmark.macro_positions[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        soft_sizes = benchmark.macro_sizes[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        soft_grid = np.zeros((d_rows, d_cols), dtype=np.float64)
        for p, s in zip(soft_pos, soft_sizes):
            r = int(np.clip(math.floor(float(p[1]) / ch * d_rows), 0, d_rows - 1))
            c = int(np.clip(math.floor(float(p[0]) / cw * d_cols), 0, d_cols - 1))
            soft_grid[r, c] += (float(s[0]) * float(s[1])) / cell_area

        for step in range(iters):
            frac = step / max(1, iters - 1)
            owner_points = owner_pos.copy()
            owner_points[:n] = pos

            forces = np.zeros_like(pos)

            pa = owner_points[edge_src]
            pb = owner_points[edge_dst]
            spring = (pb - pa) * edge_w[:, None]
            np.add.at(forces, edge_src[src_hard], spring[src_hard])
            np.add.at(forces, edge_dst[dst_hard], -spring[dst_hard])
            forces /= hard_degree[:, None]
            forces *= self.fd_attraction_weight

            dx = pos[:, 0:1] - pos[:, 0:1].T
            dy = pos[:, 1:2] - pos[:, 1:2].T
            abs_dx = np.abs(dx)
            abs_dy = np.abs(dy)
            ox = sep_x + 0.12 - abs_dx
            oy = sep_y + 0.12 - abs_dy
            overlap = (ox > 0.0) & (oy > 0.0) & ~eye
            sign_x = np.where(dx >= 0.0, 1.0, -1.0)
            sign_y = np.where(dy >= 0.0, 1.0, -1.0)
            push_x = np.where(overlap & (ox <= oy), sign_x * ox, 0.0)
            push_y = np.where(overlap & (oy < ox), sign_y * oy, 0.0)
            forces[:, 0] += self.fd_repulsion_weight * push_x.sum(axis=1)
            forces[:, 1] += self.fd_repulsion_weight * push_y.sum(axis=1)

            if self.fd_density_weight > 0.0:
                hard_grid = soft_grid.copy()
                rows = np.clip(np.floor(pos[:, 1] / ch * d_rows).astype(np.int64), 0, d_rows - 1)
                cols = np.clip(np.floor(pos[:, 0] / cw * d_cols).astype(np.int64), 0, d_cols - 1)
                np.add.at(hard_grid, (rows, cols), area / cell_area)
                target = max(0.80, float(np.mean(hard_grid)) * 1.20)
                overflow = np.maximum(0.0, hard_grid - target)
                bin_overflow = overflow[rows, cols] / max(target, 1.0e-9)
                bin_centers = np.column_stack(
                    [
                        (cols.astype(np.float64) + 0.5) * cw / d_cols,
                        (rows.astype(np.float64) + 0.5) * ch / d_rows,
                    ]
                )
                away = pos - bin_centers
                away_norm = np.linalg.norm(away, axis=1)
                away_norm = np.maximum(away_norm, 1.0)
                forces += self.fd_density_weight * bin_overflow[:, None] * away / away_norm[:, None] * max(cw, ch)

            forces[~movable] = 0.0
            norms = np.linalg.norm(forces, axis=1)
            max_step = max_step0 * (max_step1 / max_step0) ** frac
            scale = np.minimum(1.0, max_step / np.maximum(norms, 1.0e-12))
            pos[movable, 0] += forces[movable, 0] * scale[movable]
            pos[movable, 1] += forces[movable, 1] * scale[movable]
            pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

            if (step + 1) % repair_every == 0:
                pos = self._repair_all_overlaps(pos, movable, sizes, half_w, half_h, cw, ch)

        return pos

    def _seed_schedule(self, n_hard: int) -> List[int]:
        if n_hard >= 650:
            return [42, 314]
        if n_hard >= 400:
            return [42, 314, 2718]
        return [42, 314, 2718, 9001]

    def _iteration_budget(self, n_hard: int, num_edges: int) -> int:
        edge_factor = 1.0 if num_edges < 8000 else 0.8
        if n_hard >= 650:
            return int(6500 * edge_factor)
        if n_hard >= 400:
            return int(5500 * edge_factor)
        return int(4500 * edge_factor)

    def _time_budget_seconds(self, n_hard: int) -> float:
        if n_hard >= 650:
            return 180.0
        if n_hard >= 400:
            return 130.0
        return 90.0

    def _refine(
        self,
        pos: np.ndarray,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
        benchmark: Benchmark,
        iterations: int,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> np.ndarray:
        movable_idx = np.where(movable)[0]
        if len(movable_idx) == 0:
            return pos

        state = _SearchState(
            pos,
            sizes,
            benchmark,
            edges,
            incident,
            owner_pos,
            use_density=self.use_density,
            use_congestion=self.use_congestion,
        )
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        hard_neighbor_weights = self._hard_neighbor_weights(edges, benchmark.num_hard_macros)

        current = state.total_cost()
        best_cost = current
        best_pos = pos.copy()
        t_start = max(cw, ch) * 0.10
        t_end = max(cw, ch) * 0.0008

        for step in range(iterations):
            frac = step / max(1, iterations - 1)
            temp = t_start * (t_end / t_start) ** frac
            move_kind = rng.random()

            if move_kind < 0.56:
                changed = self._propose_shift(state, movable_idx, half_w, half_h, cw, ch, temp, frac, rng)
            elif move_kind < 0.74:
                changed = self._propose_pull(state, movable_idx, incident, edges, owner_pos, half_w, half_h, cw, ch, rng)
            elif move_kind < 0.90 or not self.use_cluster_shift:
                changed = self._propose_swap(state, movable_idx, hard_neighbor_weights, half_w, half_h, cw, ch, rng)
            else:
                changed = self._propose_cluster_shift(
                    state, movable, movable_idx, hard_neighbor_weights, half_w, half_h, cw, ch, temp, frac, rng, np_rng
                )

            if not changed:
                continue

            changed_ids = [idx for idx, _old, _new in changed]
            if self._any_overlap(state.pos, changed_ids, sep_x, sep_y, gap=0.025):
                state.rollback(changed)
                continue

            new_cost = state.total_cost()
            delta = new_cost - current
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1.0e-9)):
                current = new_cost
                if current < best_cost:
                    best_cost = current
                    best_pos = state.pos.copy()
            else:
                state.rollback(changed)

        return best_pos

    def _propose_shift(self, state, movable_idx, half_w, half_h, cw, ch, temp, frac, rng):
        i = int(rng.choice(movable_idx))
        old = state.pos[i].copy()
        scale = temp * (0.25 + 0.75 * (1.0 - frac))
        new = np.array(
            [
                np.clip(old[0] + rng.gauss(0.0, scale), half_w[i], cw - half_w[i]),
                np.clip(old[1] + rng.gauss(0.0, scale), half_h[i], ch - half_h[i]),
            ],
            dtype=np.float64,
        )
        return state.apply([(i, new)])

    def _propose_pull(self, state, movable_idx, incident, edges, owner_pos, half_w, half_h, cw, ch, rng):
        i = int(rng.choice(movable_idx))
        if not incident[i]:
            return None
        sx = sy = sw = 0.0
        for edge_id in incident[i]:
            a, b, w = edges[edge_id]
            other = b if a == i else a
            p = state.pos[other] if other < state.n_hard else owner_pos[other]
            sx += w * p[0]
            sy += w * p[1]
            sw += w
        if sw <= 0.0:
            return None
        target = np.array([sx / sw, sy / sw], dtype=np.float64)
        alpha = rng.uniform(0.04, 0.22)
        old = state.pos[i].copy()
        new = old + alpha * (target - old)
        new[0] = np.clip(new[0], half_w[i], cw - half_w[i])
        new[1] = np.clip(new[1], half_h[i], ch - half_h[i])
        return state.apply([(i, new)])

    def _propose_swap(self, state, movable_idx, hard_neighbor_weights, half_w, half_h, cw, ch, rng):
        i = int(rng.choice(movable_idx))
        if hard_neighbor_weights[i] and rng.random() < 0.65:
            j = int(rng.choice(hard_neighbor_weights[i][: min(8, len(hard_neighbor_weights[i]))]))
            if j not in set(movable_idx.tolist()):
                j = int(rng.choice(movable_idx))
        else:
            j = int(rng.choice(movable_idx))
        if i == j:
            return None
        old_i = state.pos[i].copy()
        old_j = state.pos[j].copy()
        new_i = np.array([np.clip(old_j[0], half_w[i], cw - half_w[i]), np.clip(old_j[1], half_h[i], ch - half_h[i])])
        new_j = np.array([np.clip(old_i[0], half_w[j], cw - half_w[j]), np.clip(old_i[1], half_h[j], ch - half_h[j])])
        return state.apply([(i, new_i), (j, new_j)])

    def _propose_cluster_shift(
        self,
        state,
        movable,
        movable_idx,
        hard_neighbor_weights,
        half_w,
        half_h,
        cw,
        ch,
        temp,
        frac,
        rng,
        np_rng,
    ):
        root = int(rng.choice(movable_idx))
        cluster = [root]
        for nb in hard_neighbor_weights[root][:3]:
            if movable[nb] and rng.random() < 0.65:
                cluster.append(int(nb))
        cluster = sorted(set(cluster))
        scale = temp * (0.18 + 0.50 * (1.0 - frac))
        delta = np_rng.normal(0.0, scale, size=2)
        updates = []
        for i in cluster:
            old = state.pos[i]
            new = np.array(
                [
                    np.clip(old[0] + delta[0], half_w[i], cw - half_w[i]),
                    np.clip(old[1] + delta[1], half_h[i], ch - half_h[i]),
                ],
                dtype=np.float64,
            )
            updates.append((i, new))
        return state.apply(updates)

    def _hard_neighbor_weights(self, edges: List[Edge], n_hard: int) -> List[List[int]]:
        acc: List[Dict[int, float]] = [dict() for _ in range(n_hard)]
        for a, b, w in edges:
            if a < n_hard and b < n_hard:
                acc[a][b] = acc[a].get(b, 0.0) + w
                acc[b][a] = acc[b].get(a, 0.0) + w
        return [sorted(d, key=d.get, reverse=True) for d in acc]

    def _any_overlap(self, pos: np.ndarray, changed_ids: Sequence[int], sep_x, sep_y, gap: float) -> bool:
        for idx in changed_ids:
            dx = np.abs(pos[idx, 0] - pos[:, 0])
            dy = np.abs(pos[idx, 1] - pos[:, 1])
            overlaps = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
            overlaps[idx] = False
            if overlaps.any():
                return True
        return False

    def _barycenter_polish(self, pos, movable, edges, incident, owner_pos, sizes, half_w, half_h, cw, ch):
        for _ in range(2):
            order = sorted(np.where(movable)[0].tolist(), key=lambda i: -len(incident[i]))
            for i in order:
                if not incident[i]:
                    continue
                sx = sy = sw = 0.0
                for edge_id in incident[i]:
                    a, b, w = edges[edge_id]
                    other = b if a == i else a
                    p = pos[other] if other < len(pos) else owner_pos[other]
                    sx += w * p[0]
                    sy += w * p[1]
                    sw += w
                if sw <= 0.0:
                    continue
                old = pos[i].copy()
                new = old + 0.08 * (np.array([sx / sw, sy / sw]) - old)
                new[0] = np.clip(new[0], half_w[i], cw - half_w[i])
                new[1] = np.clip(new[1], half_h[i], ch - half_h[i])
                pos[i] = new
                if self._overlaps_one(pos, i, sizes, gap=0.025):
                    pos[i] = old
        return pos

    def _overlaps_one(self, pos, idx, sizes, gap):
        sep_x = (sizes[idx, 0] + sizes[:, 0]) / 2.0
        sep_y = (sizes[idx, 1] + sizes[:, 1]) / 2.0
        dx = np.abs(pos[idx, 0] - pos[:, 0])
        dy = np.abs(pos[idx, 1] - pos[:, 1])
        overlaps = (dx < sep_x + gap) & (dy < sep_y + gap)
        overlaps[idx] = False
        return bool(overlaps.any())

    def _repair_all_overlaps(self, pos, movable, sizes, half_w, half_h, cw, ch):
        n = len(pos)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        for _ in range(80):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    ox = sep_x[i, j] + 0.035 - abs(dx)
                    oy = sep_y[i, j] + 0.035 - abs(dy)
                    if ox <= 0.0 or oy <= 0.0:
                        continue
                    if not movable[i] and not movable[j]:
                        continue
                    axis_x = ox < oy
                    sign = 1.0 if (dx if axis_x else dy) >= 0.0 else -1.0
                    if movable[i] and movable[j]:
                        share_i = share_j = 0.5
                    elif movable[i]:
                        share_i, share_j = 1.0, 0.0
                    else:
                        share_i, share_j = 0.0, 1.0
                    shift = (ox if axis_x else oy) + 0.02
                    if axis_x:
                        pos[i, 0] = np.clip(pos[i, 0] + sign * share_i * shift, half_w[i], cw - half_w[i])
                        pos[j, 0] = np.clip(pos[j, 0] - sign * share_j * shift, half_w[j], cw - half_w[j])
                    else:
                        pos[i, 1] = np.clip(pos[i, 1] + sign * share_i * shift, half_h[i], ch - half_h[i])
                        pos[j, 1] = np.clip(pos[j, 1] - sign * share_j * shift, half_h[j], ch - half_h[j])
                    moved = True
            if not moved:
                break
        return self._legalize(pos, movable, sizes, half_w, half_h, cw, ch, n)

    def _legalize(self, pos, movable, sizes, half_w, half_h, cw, ch, n):
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()
        for idx in order:
            legal[idx, 0] = np.clip(legal[idx, 0], half_w[idx], cw - half_w[idx])
            legal[idx, 1] = np.clip(legal[idx, 1], half_h[idx], ch - half_h[idx])
            if not movable[idx]:
                placed[idx] = True
                continue
            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                conflict = (dx < sep_x[idx] + 0.035) & (dy < sep_y[idx] + 0.035) & placed
                conflict[idx] = False
                if not conflict.any():
                    placed[idx] = True
                    continue
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.20
            best_p = legal[idx].copy()
            best_d = float("inf")
            for radius in range(1, 180):
                found = False
                for dxm in range(-radius, radius + 1):
                    for dym in range(-radius, radius + 1):
                        if abs(dxm) != radius and abs(dym) != radius:
                            continue
                        cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                        cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                        if placed.any():
                            dx = np.abs(cx - legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            conflict = (dx < sep_x[idx] + 0.035) & (dy < sep_y[idx] + 0.035) & placed
                            conflict[idx] = False
                            if conflict.any():
                                continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy], dtype=np.float64)
                            found = True
                if found:
                    break
            legal[idx] = best_p
            placed[idx] = True
        return legal

    def _surrogate_cost(self, pos, edges, owner_pos, benchmark, sizes) -> float:
        state = _SearchState(
            pos.copy(),
            sizes,
            benchmark,
            edges,
            [[] for _ in range(len(pos))],
            owner_pos,
            use_density=self.use_density,
            use_congestion=self.use_congestion,
        )
        return state.total_cost()

    def _candidate_positions_np(self, candidate: Candidate, benchmark: Benchmark) -> np.ndarray:
        _surrogate, hard, _force, _label, full = candidate
        if full is not None:
            return full[: benchmark.num_macros].numpy().astype(np.float64)
        return hard.astype(np.float64)

    def _dedupe_candidates(self, candidates: List[Candidate], benchmark: Benchmark, tol: float = 1.0e-5) -> List[Candidate]:
        kept: List[Candidate] = []
        kept_pos: List[np.ndarray] = []
        for candidate in candidates:
            pos = self._candidate_positions_np(candidate, benchmark)
            duplicate = False
            for existing in kept_pos:
                if pos.shape == existing.shape and float(np.max(np.abs(pos - existing))) < tol:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
                kept_pos.append(pos)
        return kept

    def _candidate_family(self, label: str) -> str:
        base = label.split("+")[0]
        if "soft_global_topo_" in base:
            return "topology_portfolio"
        if "corridor_narrow" in base or "corridor_wide" in base or "corridor_bridge" in base:
            return "corridor_v2"
        if "corridor_density_guarded" in base or "corridor_planned" in base:
            return "corridor_v2"
        if "density_spread" in base:
            return "density_spread"
        if "density_axis" in base:
            return "density_axis"
        if "hotspot_refined" in base:
            return "hotspot_refined"
        if "congestion_refined" in base:
            return "congestion_refined"
        if "congestion_flow_drag" in base:
            return "congestion_flow_drag"
        if "hotspot_micro_cd" in base:
            return "hotspot_micro_cd"
        if "random_basin_probe" in base:
            return "random_basin_probe"
        if "exact_polished" in base:
            return "exact_polished"
        if "spread_cong" in base:
            return "spread_cong"
        if "congestion_escape" in base:
            return "congestion_escape"
        if base.startswith("local_search"):
            return "local_search"
        if base == "legacy_sa":
            return "legacy_sa"
        if "soft_global" in base:
            return "soft_global_other"
        return base

    def _cheap_corridor_blockage_np(self, placement_np: np.ndarray, benchmark: Benchmark) -> float:
        try:
            device = torch.device("cpu")
            dtype = torch.float32
            num_macros = int(benchmark.num_macros)
            if num_macros <= 0:
                return 0.0
            plan = self._soft_global_predict_corridors(
                benchmark,
                device,
                dtype,
                base_positions=benchmark.macro_positions[:num_macros],
                width_scale=1.0,
                weight_scale=1.0,
                bridge_boost=0.85,
                max_corridors=3,
            )
            if plan is None:
                return 0.0
            pos = torch.tensor(placement_np[:num_macros], dtype=dtype, device=device)
            sizes = benchmark.macro_sizes[:num_macros].to(device=device, dtype=dtype)
            movable = (~benchmark.macro_fixed[:num_macros].to(device=device))
            return float(
                self._soft_global_corridor_blockage(
                    pos,
                    sizes,
                    movable,
                    plan,
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                )
                .detach()
                .cpu()
            )
        except Exception:
            return float("nan")

    def _cheap_candidate_diagnostics(
        self, placement: torch.Tensor, benchmark: Benchmark, edges: List[Edge], label: str
    ) -> Tuple[str, float, float, float, float]:
        family = self._candidate_family(label)
        placement_np = placement[: benchmark.num_macros].detach().cpu().numpy().astype(np.float64)
        if benchmark.port_positions.shape[0] == 0:
            owner = placement_np
        else:
            owner = np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        cheap_density = self._estimate_density_overflow_np(placement_np, benchmark)
        cheap_congestion = self._estimate_congestion_overflow_np(owner, edges, benchmark) if edges else 0.0
        cheap_blockage = self._cheap_corridor_blockage_np(placement_np, benchmark)
        cheap_like = cheap_congestion + 0.52 * cheap_density + 0.16 * cheap_blockage
        return family, cheap_density, cheap_congestion, cheap_blockage, cheap_like

    def _candidate_exact_diagnostics(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        edges: List[Edge],
        label: str,
        plc=None,
    ) -> Dict[str, object]:
        family = self._candidate_family(label)
        placement_np = placement[: benchmark.num_macros].detach().cpu().numpy().astype(np.float64)
        owner = (
            placement_np
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([placement_np, benchmark.port_positions.numpy().astype(np.float64)])
        )
        cheap_density = self._estimate_density_overflow_np(placement_np, benchmark)
        cheap_h, cheap_v = self._routing_congestion_arrays_np(owner, edges, benchmark) if edges else (
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
        )
        cheap_map = cheap_h + cheap_v
        cheap_congestion = self._tail_mean_np(np.concatenate([cheap_h.reshape(-1), cheap_v.reshape(-1)]), 0.05)
        cheap_hot = self._tail_mean_np(cheap_map, 0.08)
        cheap_blockage = self._cheap_corridor_blockage_np(placement_np, benchmark)
        style_h, style_v = self._exact_style_congestion_arrays_np(placement_np, benchmark)
        style_map = style_h + style_v
        style_congestion = self._tail_mean_np(np.concatenate([style_h.reshape(-1), style_v.reshape(-1)]), 0.05)
        style_hot = self._tail_mean_np(style_map, 0.08)
        exact_map = self._extract_exact_congestion_map(plc) if plc is not None else None
        cheap_align, cheap_corr, cheap_overlap = self._grid_alignment_summary(cheap_map, exact_map)
        style_align, style_corr, style_overlap = self._grid_alignment_summary(style_map, exact_map)
        cheap_like = cheap_congestion + 0.52 * cheap_density + 0.16 * cheap_blockage
        style_like = style_congestion + 0.52 * cheap_density + 0.16 * cheap_blockage
        exact_top = self._hot_cell_tokens(exact_map, limit=6) if exact_map is not None else "missing"
        return {
            "family": family,
            "cheap_density": float(cheap_density),
            "cheap_congestion": float(cheap_congestion),
            "cheap_blockage": float(cheap_blockage),
            "cheap_hot": float(cheap_hot),
            "cheap_like": float(cheap_like),
            "style_congestion": float(style_congestion),
            "style_hot": float(style_hot),
            "style_like": float(style_like),
            "cheap_top": self._hot_cell_tokens(cheap_map, limit=6),
            "style_top": self._hot_cell_tokens(style_map, limit=6),
            "exact_top": exact_top,
            "cheap_align": cheap_align,
            "style_align": style_align,
            "cheap_corr": float(cheap_corr) if math.isfinite(cheap_corr) else float("nan"),
            "style_corr": float(style_corr) if math.isfinite(style_corr) else float("nan"),
            "cheap_overlap": float(cheap_overlap) if math.isfinite(cheap_overlap) else float("nan"),
            "style_overlap": float(style_overlap) if math.isfinite(style_overlap) else float("nan"),
        }

    def _large_design_exact_shortlist(self, candidates: List[Candidate], benchmark: Benchmark) -> List[Candidate]:
        shortlist: List[Candidate] = []

        def add(candidate: Optional[Candidate]) -> None:
            if candidate is not None and all(candidate is not existing for existing in shortlist):
                shortlist.append(candidate)

        exact_edges, _diag_incident, _diag_soft_neighbors = _extract_weighted_edges(benchmark)
        profile = self._benchmark_profile(benchmark, exact_edges)

        def cheap_shortlist_rank(candidate: Candidate) -> float:
            surrogate, _hard, _force, label, _full = candidate
            try:
                pos = self._candidate_positions_np(candidate, benchmark)
                if pos.shape[0] != benchmark.num_macros:
                    return float(surrogate)
                owner = pos if benchmark.port_positions.shape[0] == 0 else np.vstack(
                    [pos, benchmark.port_positions.numpy().astype(np.float64)]
                )
                density = self._estimate_density_overflow_np(pos, benchmark)
                if bool(profile["large_high_congestion"]):
                    congestion = self._estimate_exact_style_congestion_overflow_np(pos, benchmark)
                else:
                    congestion = self._estimate_congestion_overflow_np(owner, exact_edges, benchmark) if exact_edges else 0.0
                norm_surrogate = float(surrogate) / max(float(len(exact_edges)) * 14.0, 1.0)
                corridor_penalty = 0.035 if "corridor_" in label or "corridor_planned" in label else 0.0
                topology_bonus = -0.020 if "soft_global_topo_" in label else 0.0
                return 0.35 * norm_surrogate + 1.20 * congestion + 0.65 * density + corridor_penalty + topology_bonus
            except Exception:
                return float(surrogate)

        hotspot_refined = next((c for c in candidates if c[3] == "soft_global_hotspot_refined"), None)
        congestion_refined = next((c for c in candidates if c[3] == "soft_global_congestion_refined"), None)
        corridor_labels = ("corridor_narrow", "corridor_wide", "corridor_bridge", "corridor_density_guarded")
        corridor_representative = min(
            (
                c
                for c in candidates
                if any(name in c[3] for name in corridor_labels)
                and ("_legalized" in c[3] or "_refined" in c[3])
                and "_raw" not in c[3]
            ),
            key=cheap_shortlist_rank,
            default=None,
        )
        if corridor_representative is None:
            corridor_representative = min(
                (
                    c
                    for c in candidates
                    if "soft_global_corridor_planned" in c[3]
                    and ("_legalized" in c[3] or "_refined" in c[3])
                    and "_raw" not in c[3]
                ),
                key=cheap_shortlist_rank,
                default=None,
            )
        density_spread = next((c for c in candidates if c[3] == "soft_global_density_spread_refined"), None)
        density_axis = next((c for c in candidates if c[3] == "soft_global_density_axis_refined"), None)
        topology_pool = sorted(
            (c for c in candidates if "soft_global_topo_" in c[3]),
            key=cheap_shortlist_rank,
        )
        hotspot_cd_pool = sorted(
            (c for c in candidates if "hotspot_micro_cd" in c[3]),
            key=cheap_shortlist_rank,
        )
        regular_soft_pool = [
            c
            for c in candidates
            if "soft_global" in c[3]
            and ("_legalized" in c[3] or "_refined" in c[3])
            and "_raw" not in c[3]
            and "channel_refined" not in c[3]
            and "congestion_refined" not in c[3]
            and "density_spread" not in c[3]
            and "density_axis" not in c[3]
            and "partition_refined" not in c[3]
            and "hotspot_refined" not in c[3]
            and "corridor_" not in c[3]
            and "soft_global_topo_" not in c[3]
        ]
        spread_pool = [c for c in regular_soft_pool if "spread_cong" in c[3]]
        best_regular = min(spread_pool or regular_soft_pool, key=cheap_shortlist_rank) if regular_soft_pool else None
        legacy = next((c for c in candidates if c[3] == "legacy_sa"), None)
        local = min((c for c in candidates if c[3].startswith("local_search")), key=lambda c: c[0], default=None)
        if bool(profile["large_high_congestion"]):
            add(best_regular)
            add(density_spread)
            add(density_axis)
            add(hotspot_refined)
            add(legacy)
            add(local)
            add(congestion_refined)
            add(corridor_representative)
            for cd_candidate in hotspot_cd_pool[:2]:
                add(cd_candidate)
            for topo_candidate in topology_pool[:6]:
                add(topo_candidate)
            if not shortlist and candidates:
                add(candidates[0])
            return self._dedupe_candidates(shortlist[:14], benchmark)

        for cd_candidate in hotspot_cd_pool[:1]:
            add(cd_candidate)
        add(hotspot_refined)
        add(corridor_representative)
        add(congestion_refined)
        add(density_spread)
        add(density_axis)
        add(best_regular)
        if not shortlist:
            soft_pool = [c for c in candidates if "soft_global" in c[3]]
            add(min(soft_pool, key=lambda c: c[0]) if soft_pool else None)
        add(legacy)
        if not shortlist and candidates:
            add(candidates[0])
        return self._dedupe_candidates(shortlist[:5], benchmark)

    def _basin_escape_exact_preselect(
        self,
        candidates: List[Candidate],
        benchmark: Benchmark,
        edges: List[Edge],
    ) -> List[Candidate]:
        before = len(candidates)
        if not candidates:
            self.basin_escape_preselection_log = "before=0|after_distance=0|after_budget=0"
            return []
        profile = self._benchmark_profile(benchmark, edges)
        budget = int(os.environ.get("JIHO_EXACT_BUDGET_DEFAULT", "6"))
        if self._is_large_soft_global_design(benchmark):
            budget = int(os.environ.get("JIHO_EXACT_BUDGET_LARGE", "8"))
        if bool(profile["large_high_congestion"]):
            budget = int(os.environ.get("JIHO_EXACT_BUDGET_HIGH_CONG", "10"))
        budget = max(1, budget)
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0e-9)
        mean_thresh = max(0.0, float(os.environ.get("JIHO_DEDUPE_MEAN_DISP_THRESH", "0.003")))
        hard_mean_thresh = max(0.0, float(os.environ.get("JIHO_DEDUPE_HARD_MEAN_DISP_THRESH", "0.002")))

        records = []
        dropped: List[str] = []
        for candidate in candidates:
            try:
                macro_pos = self._candidate_macro_positions_np(candidate, benchmark)
                if macro_pos.shape != (benchmark.num_macros, 2):
                    dropped.append(f"{candidate[3]}|invalid_shape")
                    continue
                cheap = self._basin_escape_cheap_record(candidate, macro_pos, benchmark, edges, profile)
                records.append(cheap)
            except Exception as exc:
                dropped.append(f"{candidate[3]}|invalid_shape:{type(exc).__name__}")

        records.sort(key=lambda r: float(r["cheap_score"]))
        distance_kept: List[Dict[str, object]] = []
        for record in records:
            duplicate_of = None
            for existing in distance_kept:
                dist = self._basin_escape_distance(record["macro_pos"], existing["macro_pos"], benchmark, span)
                same_family = record["family"] == existing["family"]
                if dist["mean"] < mean_thresh and dist["hard_mean"] < hard_mean_thresh:
                    duplicate_of = existing
                    break
                if same_family and dist["hard_mean"] < hard_mean_thresh * 1.35:
                    duplicate_of = existing
                    break
            if duplicate_of is not None:
                dropped.append(self._basin_escape_drop_text(record, "near_duplicate", f"dup={duplicate_of['label']}"))
                continue
            distance_kept.append(record)

        protected_records = [r for r in distance_kept if bool(r["protected"])]
        regular_records = [r for r in distance_kept if not bool(r["protected"])]
        protected_records.sort(key=lambda r: (self._protected_family_priority(str(r["family"])), float(r["cheap_score"])))
        regular_records.sort(key=lambda r: float(r["cheap_score"]))

        selected: List[Dict[str, object]] = []
        selected_families = set()
        for record in protected_records:
            if len(selected) >= budget:
                dropped.append(self._basin_escape_drop_text(record, "cheap_rank_budget"))
                continue
            selected.append(record)
            selected_families.add(str(record["family"]))

        for record in regular_records:
            if len(selected) >= budget:
                dropped.append(self._basin_escape_drop_text(record, "cheap_rank_budget"))
                continue
            selected.append(record)

        if os.environ.get("JIHO_BASIN_ESCAPE_FLOW_DRAG", "0") == "1" and "congestion_flow_drag" not in selected_families:
            flow = min(
                (r for r in distance_kept if str(r["family"]) == "congestion_flow_drag"),
                key=lambda r: float(r["cheap_score"]),
                default=None,
            )
            if flow is not None:
                if len(selected) >= budget:
                    non_protected = [r for r in selected if not bool(r["protected"])]
                    if non_protected:
                        victim = max(non_protected, key=lambda r: float(r["cheap_score"]))
                        selected.remove(victim)
                        dropped.append(self._basin_escape_drop_text(victim, "cheap_rank_budget", "replaced_by_flow_drag"))
                if all(id(flow["candidate"]) != id(r["candidate"]) for r in selected):
                    selected.append(flow)

        if os.environ.get("JIHO_FLOW_DRAG_FORCE_EXACT", "0") == "1":
            flow = min(
                (r for r in records if str(r["family"]) == "congestion_flow_drag"),
                key=lambda r: float(r["cheap_score"]),
                default=None,
            )
            if flow is not None:
                if all(id(flow["candidate"]) != id(r["candidate"]) for r in selected):
                    selected.append(flow)
                self.flow_drag_log = (
                    f"{self.flow_drag_log};flow_drag_force_exact=1|kept={flow['label']}|reason=debug_force"
                )

        random_force_text = ""
        if (
            os.environ.get("JIHO_RANDOM_BASIN_PROBE", "0") == "1"
            and os.environ.get("JIHO_RANDOM_BASIN_FORCE_EXACT", "1") == "1"
        ):
            random_probe = min(
                (r for r in records if str(r["family"]) == "random_basin_probe"),
                key=lambda r: float(r["cheap_score"]),
                default=None,
            )
            if random_probe is not None:
                if all(id(random_probe["candidate"]) != id(r["candidate"]) for r in selected):
                    selected.append(random_probe)
                random_force_text = (
                    f"random_basin_force_exact=1|kept={random_probe['label']}|"
                    f"rank={float(random_probe['cheap_score']):.6f}|reason=debug_force"
                )
                self.random_basin_log = f"{self.random_basin_log};{random_force_text}"

        if not selected and distance_kept:
            selected.append(distance_kept[0])
        selected_ids = {id(r["candidate"]) for r in selected}
        for record in distance_kept:
            if id(record["candidate"]) not in selected_ids and not any(str(record["label"]) in item for item in dropped):
                dropped.append(self._basin_escape_drop_text(record, "cheap_rank_budget"))

        kept_rows = [
            f"{r['label']}|fam={r['family']}|score={float(r['cheap_score']):.6f}|nov={float(r['novelty']):.5f}|prot={int(bool(r['protected']))}"
            for r in selected
        ]
        self.basin_escape_preselection_log = (
            f"before={before}|after_distance={len(distance_kept)}|after_budget={len(selected)}|budget={budget}|"
            f"kept={','.join(str(r['label']) for r in selected)}|"
            f"kept_meta={'/'.join(kept_rows[:18])}|"
            f"dropped={'/'.join(dropped[:36])}"
        )
        if os.environ.get("JIHO_RANDOM_BASIN_PROBE", "0") == "1":
            random_kept = [
                f"{r['label']}|rank={float(r['cheap_score']):.6f}|den={float(r.get('density', 0.0)):.4f}|"
                f"cong={float(r.get('congestion', 0.0)):.4f}|style_cong={float(r.get('style_congestion', 0.0)):.4f}|"
                f"disp={float(r.get('novelty', 0.0)):.5f}"
                for r in selected
                if str(r["family"]) == "random_basin_probe"
            ]
            random_dropped = [item for item in dropped if "random_basin_probe" in item]
            self.random_basin_preselection_log = (
                f"{random_force_text or 'random_basin_force_exact=0'}|"
                f"kept={'/'.join(random_kept[:12])}|"
                f"dropped={'/'.join(random_dropped[:24])}"
            )
        return self._dedupe_candidates([r["candidate"] for r in selected], benchmark)

    def _candidate_macro_positions_np(self, candidate: Candidate, benchmark: Benchmark) -> np.ndarray:
        _surrogate, hard, _force, _label, full = candidate
        if full is not None:
            return full[: benchmark.num_macros].numpy().astype(np.float64)
        macro_pos = benchmark.macro_positions[: benchmark.num_macros].numpy().astype(np.float64)
        macro_pos[: benchmark.num_hard_macros] = hard.astype(np.float64)
        return macro_pos

    def _basin_escape_cheap_record(
        self,
        candidate: Candidate,
        macro_pos: np.ndarray,
        benchmark: Benchmark,
        edges: List[Edge],
        profile: Dict[str, object],
    ) -> Dict[str, object]:
        surrogate, _hard, _force, label, _full = candidate
        family = self._candidate_family(label)
        density = self._estimate_density_overflow_np(macro_pos, benchmark)
        owner = (
            macro_pos
            if benchmark.port_positions.shape[0] == 0
            else np.vstack([macro_pos, benchmark.port_positions.numpy().astype(np.float64)])
        )
        if bool(profile["large_high_congestion"]):
            congestion = self._estimate_exact_style_congestion_overflow_np(macro_pos, benchmark)
        else:
            congestion = self._estimate_congestion_overflow_np(owner, edges, benchmark) if edges else 0.0
        if bool(profile["large_high_congestion"]) or family in {"congestion_flow_drag", "random_basin_probe"}:
            style_congestion = self._estimate_exact_style_congestion_overflow_np(macro_pos, benchmark)
        else:
            style_congestion = float(congestion)
        overlap_risk = self._hard_overlap_risk_np(macro_pos[: benchmark.num_hard_macros], benchmark)
        norm_wl = float(surrogate) / max(float(len(edges)) * 14.0, 1.0)
        novelty = self._candidate_novelty_from_initial(macro_pos, benchmark)
        protected = self._is_protected_candidate_family(family, label)
        cheap_score = (
            0.35 * norm_wl
            + 0.65 * float(density)
            + 1.15 * float(congestion)
            + 2.25 * float(overlap_risk)
            - 0.18 * float(novelty)
        )
        if protected:
            cheap_score -= 0.035
        return {
            "candidate": candidate,
            "macro_pos": macro_pos,
            "label": label,
            "family": family,
            "cheap_score": float(cheap_score),
            "density": float(density),
            "congestion": float(congestion),
            "style_congestion": float(style_congestion),
            "overlap_risk": float(overlap_risk),
            "novelty": float(novelty),
            "protected": bool(protected),
        }

    def _basin_escape_drop_text(self, record: Dict[str, object], reason: str, extra: str = "") -> str:
        text = (
            f"{record['label']}|rank={float(record['cheap_score']):.6f}|den={float(record.get('density', 0.0)):.4f}|"
            f"cong={float(record.get('congestion', 0.0)):.4f}|style_cong={float(record.get('style_congestion', 0.0)):.4f}|"
            f"disp={float(record.get('novelty', 0.0)):.5f}|drop={reason}"
        )
        return f"{text}|{extra}" if extra else text

    def _candidate_novelty_from_initial(self, macro_pos: np.ndarray, benchmark: Benchmark) -> float:
        base = benchmark.macro_positions[: benchmark.num_macros].numpy().astype(np.float64)
        if base.shape != macro_pos.shape:
            return 0.0
        span = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0e-9)
        disp = np.linalg.norm(macro_pos - base, axis=1)
        return float(np.mean(disp) / span) if disp.size else 0.0

    def _basin_escape_distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        benchmark: Benchmark,
        span: float,
    ) -> Dict[str, float]:
        if a.shape != b.shape:
            return {"mean": float("inf"), "max": float("inf"), "hard_mean": float("inf")}
        disp = np.linalg.norm(a - b, axis=1) / span
        hard_n = int(benchmark.num_hard_macros)
        hard_disp = disp[:hard_n]
        return {
            "mean": float(np.mean(disp)) if disp.size else 0.0,
            "max": float(np.max(disp)) if disp.size else 0.0,
            "hard_mean": float(np.mean(hard_disp)) if hard_disp.size else 0.0,
        }

    def _hard_overlap_risk_np(self, hard_pos: np.ndarray, benchmark: Benchmark) -> float:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return 0.0
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        area_mean = max(float(np.mean(sizes[:, 0] * sizes[:, 1])), 1.0e-9)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                ox = max(0.0, (sizes[i, 0] + sizes[j, 0]) * 0.5 - abs(float(hard_pos[i, 0] - hard_pos[j, 0])))
                oy = max(0.0, (sizes[i, 1] + sizes[j, 1]) * 0.5 - abs(float(hard_pos[i, 1] - hard_pos[j, 1])))
                if ox > 0.0 and oy > 0.0:
                    total += (ox * oy) / area_mean
                    count += 1
        return float(total + 0.01 * count)

    def _is_protected_candidate_family(self, family: str, label: str) -> bool:
        if family in {
            "spread_cong",
            "density_spread",
            "density_axis",
            "hotspot_refined",
            "congestion_refined",
            "legacy_sa",
            "corridor_v2",
            "congestion_flow_drag",
        }:
            return True
        return "corridor_" in label or "corridor_planned" in label

    def _protected_family_priority(self, family: str) -> int:
        order = {
            "spread_cong": 0,
            "density_spread": 1,
            "density_axis": 2,
            "hotspot_refined": 3,
            "congestion_refined": 4,
            "congestion_flow_drag": 5,
            "legacy_sa": 6,
            "corridor_v2": 7,
        }
        return order.get(family, 20)

    def _select_by_exact_proxy(self, shortlist, benchmark, soft_neighbors):
        self.candidate_exact_scores = ""
        plc = _load_plc_for_exact(benchmark.name)
        if plc is None:
            return shortlist[0][1], self.use_soft_motion, self.use_soft_bounds_repair, shortlist[0][3], shortlist[0][4]

        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return shortlist[0][1], self.use_soft_motion, self.use_soft_bounds_repair, shortlist[0][3], shortlist[0][4]

        exact_edges, _diag_incident, _diag_soft_neighbors = _extract_weighted_edges(benchmark)
        best_cost = float("inf")
        best_raw_proxy = float("inf")
        best_scored_placement: Optional[torch.Tensor] = None
        best_pos = shortlist[0][1]
        best_use_soft = self.use_soft_motion
        best_repair_soft_bounds = self.use_soft_bounds_repair
        best_label = shortlist[0][3]
        best_full = shortlist[0][4]
        score_rows = []
        alignment_rows: List[str] = []
        hotspot_rows: List[str] = []
        parent_delta_rows: List[str] = []
        exact_by_base_label: Dict[str, Dict[str, float]] = {}

        def base_label(scored_label: str) -> str:
            return scored_label.split("+", 1)[0]

        def append_exact_row(
            scored_label: str,
            placement: torch.Tensor,
            raw_proxy: float,
            wirelength: float,
            density: float,
            congestion: float,
            overlaps: int,
            exact_elapsed: float,
        ) -> None:
            diag = self._candidate_exact_diagnostics(placement, benchmark, exact_edges, scored_label, plc)
            base = base_label(scored_label)
            parent_base = self._candidate_parent_label.get(base, "")
            parent_text = ""
            if parent_base:
                parent_text = f"|parent={parent_base}"
                parent_exact = exact_by_base_label.get(parent_base)
                if parent_exact is not None and math.isfinite(parent_exact.get("proxy", float("nan"))):
                    dp = raw_proxy - parent_exact["proxy"]
                    dd = density - parent_exact["density"]
                    dc = congestion - parent_exact["congestion"]
                    dw = wirelength - parent_exact["wirelength"]
                    parent_text += f"|dp={dp:.6f}|dd={dd:.6f}|dc={dc:.6f}|dwl={dw:.6f}"
                    parent_delta_rows.append(
                        f"{base}|parent={parent_base}|dp={dp:.6f}|dd={dd:.6f}|dc={dc:.6f}|dwl={dw:.6f}"
                    )
                parent_metrics = self._candidate_parent_metrics.get(base)
                if parent_metrics is not None:
                    parent_text += (
                        f"|dstyle_c={float(diag['style_congestion']) - parent_metrics.get('style_c', 0.0):.6f}"
                        f"|dcheap_d={float(diag['cheap_density']) - parent_metrics.get('density', 0.0):.6f}"
                        f"|dstyle_hot={float(diag['style_hot']) - parent_metrics.get('style_hot', 0.0):.6f}"
                    )
            score_rows.append(
                f"{scored_label}|fam={diag['family']}|p={raw_proxy:.6f}|wl={wirelength:.6f}|d={density:.6f}|"
                f"c={congestion:.6f}|cheap_d={float(diag['cheap_density']):.6f}|"
                f"cheap_c={float(diag['cheap_congestion']):.6f}|cheap_b={float(diag['cheap_blockage']):.6f}|"
                f"cheap_hot={float(diag['cheap_hot']):.6f}|cheap_like={float(diag['cheap_like']):.6f}|"
                f"style_c={float(diag['style_congestion']):.6f}|style_hot={float(diag['style_hot']):.6f}|"
                f"style_like={float(diag['style_like']):.6f}{parent_text}|ov={overlaps}|exact_s={exact_elapsed:.3f}"
            )
            if str(diag["family"]) == "random_basin_probe":
                self.random_basin_log = (
                    f"{self.random_basin_log};exact|label={scored_label}|proxy={raw_proxy:.6f}|"
                    f"wl={wirelength:.6f}|density={density:.6f}|congestion={congestion:.6f}|"
                    f"overlaps={overlaps}|exact_s={exact_elapsed:.3f}"
                )
            alignment_rows.append(
                f"{base}|cheap_{diag['cheap_align']}|style_{diag['style_align']}"
            )
            hotspot_rows.append(
                f"{base}|cheap_top={diag['cheap_top']}|style_top={diag['style_top']}|exact_top={diag['exact_top']}"
            )
            exact_by_base_label[base] = {
                "proxy": float(raw_proxy),
                "wirelength": float(wirelength),
                "density": float(density),
                "congestion": float(congestion),
            }

        for _surrogate, hard_pos, _force_include, label, full_placement in shortlist:
            hard_pos = self._clip_hard_np(hard_pos.copy(), benchmark)
            if full_placement is not None:
                placement = full_placement.clone()
                placement[: benchmark.num_hard_macros] = torch.tensor(hard_pos, dtype=torch.float32)
                placement = self._repair_hard_bounds_tensor(placement, benchmark)
                if benchmark.macro_fixed.any():
                    placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
                try:
                    costs, exact_elapsed = self._compute_proxy_cost_timed(
                        compute_proxy_cost, placement, benchmark, plc, f"{label}+full_soft"
                    )
                    overlaps = int(costs.get("overlap_count", 999999))
                    raw_proxy = float(costs["proxy_cost"])
                    proxy = raw_proxy + overlaps * 1.0e6
                    wirelength = float(costs.get("wirelength_cost", float("nan")))
                    density = float(costs.get("density_cost", float("nan")))
                    congestion = float(costs.get("congestion_cost", float("nan")))
                except Exception:
                    overlaps = 999999
                    raw_proxy = float(_surrogate)
                    proxy = raw_proxy + overlaps * 1.0e6
                    wirelength = density = congestion = float("nan")
                    exact_elapsed = float("nan")
                scored_label = f"{label}+full_soft"
                append_exact_row(scored_label, placement, raw_proxy, wirelength, density, congestion, overlaps, exact_elapsed)
                if proxy < best_cost:
                    best_cost = proxy
                    best_raw_proxy = raw_proxy
                    best_scored_placement = placement.clone()
                    best_pos = hard_pos
                    best_use_soft = False
                    best_repair_soft_bounds = True
                    best_label = scored_label
                    best_full = placement
                continue
            soft_options = (False, True) if self.use_soft_motion else (False,)
            bounds_options = (True,) if self.use_soft_bounds_repair else (False,)
            for use_soft in soft_options:
                for repair_soft_bounds in bounds_options:
                    placement = benchmark.macro_positions.clone()
                    placement[: benchmark.num_hard_macros] = torch.tensor(hard_pos, dtype=torch.float32)
                    if use_soft:
                        placement = self._place_soft_macros(placement, benchmark, soft_neighbors)
                    if repair_soft_bounds:
                        placement = self._repair_soft_bounds_tensor(placement, benchmark)
                    placement = self._repair_hard_bounds_tensor(placement, benchmark)
                    suffixes = []
                    if use_soft:
                        suffixes.append("soft")
                    if repair_soft_bounds:
                        suffixes.append("soft_bounds")
                    scored_label = label + (f"+{'+'.join(suffixes)}" if suffixes else "")
                    try:
                        costs, exact_elapsed = self._compute_proxy_cost_timed(
                            compute_proxy_cost, placement, benchmark, plc, scored_label
                        )
                        overlaps = int(costs.get("overlap_count", 999999))
                        raw_proxy = float(costs["proxy_cost"])
                        proxy = raw_proxy + overlaps * 1.0e6
                        wirelength = float(costs.get("wirelength_cost", float("nan")))
                        density = float(costs.get("density_cost", float("nan")))
                        congestion = float(costs.get("congestion_cost", float("nan")))
                    except Exception:
                        overlaps = 999999
                        raw_proxy = float(_surrogate)
                        proxy = raw_proxy + overlaps * 1.0e6
                        wirelength = density = congestion = float("nan")
                        exact_elapsed = float("nan")
                    append_exact_row(scored_label, placement, raw_proxy, wirelength, density, congestion, overlaps, exact_elapsed)
                    if proxy < best_cost:
                        best_cost = proxy
                        best_raw_proxy = raw_proxy
                        best_scored_placement = placement.clone()
                        best_pos = hard_pos
                        best_use_soft = use_soft
                        best_repair_soft_bounds = repair_soft_bounds
                        best_label = scored_label
                        best_full = None
        if best_scored_placement is not None and math.isfinite(best_cost) and math.isfinite(best_raw_proxy):
            polished = self._run_high_congestion_exact_polish(
                best_scored_placement,
                best_cost,
                best_raw_proxy,
                best_label,
                compute_proxy_cost,
                benchmark,
                plc,
                exact_edges,
                _diag_incident,
            )
            if polished is not None:
                scored_label = "soft_global_exact_polished+full_soft"
                score_rows.append(
                    f"{scored_label}|fam={polished['family']}|source={polished['source_label']}|"
                    f"p={float(polished['raw_proxy']):.6f}|wl={float(polished['wirelength']):.6f}|"
                    f"d={float(polished['density']):.6f}|c={float(polished['congestion']):.6f}|"
                    f"cheap_d={float(polished['cheap_density']):.6f}|cheap_c={float(polished['cheap_congestion']):.6f}|"
                    f"cheap_b={float(polished['cheap_blockage']):.6f}|cheap_like={float(polished['cheap_like']):.6f}|"
                    f"accepted={int(polished['accepted'])}|tried={int(polished['tried'])}|"
                    f"moves={polished['log']}|ov={int(polished['overlaps'])}|exact_s={float(polished['elapsed']):.3f}"
                )
                best_cost = float(polished["score"])
                best_raw_proxy = float(polished["raw_proxy"])
                best_pos = polished["hard"]
                best_use_soft = False
                best_repair_soft_bounds = True
                best_label = scored_label
                best_full = polished["full"]
        self.candidate_exact_scores = ";".join(score_rows)
        self.congestion_alignment_summary = ";".join(alignment_rows[:18])
        self.exact_style_hotspot_summary = ";".join(hotspot_rows[:18])
        self.topology_parent_delta_summary = ";".join(parent_delta_rows[:18])
        return best_pos, best_use_soft, best_repair_soft_bounds, best_label, best_full

    def _run_high_congestion_exact_polish(
        self,
        base_placement: torch.Tensor,
        base_score: float,
        base_proxy: float,
        source_label: str,
        compute_proxy_cost,
        benchmark: Benchmark,
        plc,
        edges: List[Edge],
        incident: List[List[int]],
    ) -> Optional[Dict[str, object]]:
        if not self.high_congestion_exact_polish:
            return None
        profile = self._benchmark_profile(benchmark, edges)
        if not bool(profile["large_high_congestion"]):
            return None
        n = int(benchmark.num_hard_macros)
        if n <= 0 or not edges:
            return None

        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        half_w = sizes[:, 0] * 0.5
        half_h = sizes[:, 1] * 0.5
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
        fixed_hard = benchmark.macro_fixed[:n].numpy().astype(bool)
        fixed_pos = benchmark.macro_positions[:n].numpy().astype(np.float64)
        hard = base_placement[:n].detach().cpu().numpy().astype(np.float64)
        hard = self._clip_hard_np(hard, benchmark)
        if fixed_hard.any():
            hard[fixed_hard] = fixed_pos[fixed_hard]

        def build_placement(hard_pos: np.ndarray) -> torch.Tensor:
            placement = base_placement.clone()
            placement[:n] = torch.tensor(hard_pos, dtype=torch.float32)
            placement = self._repair_soft_bounds_tensor(placement, benchmark)
            placement = self._repair_hard_bounds_tensor(placement, benchmark)
            if benchmark.macro_fixed.any():
                placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
            return placement

        def exact_score(placement: torch.Tensor):
            costs, elapsed = self._compute_proxy_cost_timed(
                compute_proxy_cost, placement, benchmark, plc, "soft_global_exact_polished"
            )
            overlaps = int(costs.get("overlap_count", 999999))
            raw_proxy = float(costs["proxy_cost"])
            score = raw_proxy + overlaps * 1.0e6
            return {
                "score": score,
                "raw_proxy": raw_proxy,
                "wirelength": float(costs.get("wirelength_cost", float("nan"))),
                "density": float(costs.get("density_cost", float("nan"))),
                "congestion": float(costs.get("congestion_cost", float("nan"))),
                "overlaps": overlaps,
                "elapsed": float(elapsed),
            }

        owner = self._owner_positions_from_placement(base_placement, benchmark)
        candidate_ids, hot_rows, hot_cols = self._high_congestion_exact_polish_candidate_ids(
            hard, benchmark, movable, sizes, edges, incident, owner
        )
        if not candidate_ids:
            return None

        max_moves = max(0, int(self.high_congestion_exact_polish_max_moves))
        if max_moves <= 0:
            return None
        max_macros = max(1, int(self.high_congestion_exact_polish_candidate_macros))
        candidate_ids = candidate_ids[:max_macros]
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) * 0.5
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) * 0.5
        hard_neighbor_weights = self._hard_neighbor_weights(edges, n)
        span = max(cw, ch, 1.0e-9)
        step_scales = (0.0025, 0.005, 0.010, 0.018)

        current_hard = hard.copy()
        current_full = build_placement(current_hard)
        current_score = float(base_score)
        current_proxy = float(base_proxy)
        current_cheap_density = self._estimate_density_overflow_np(
            current_full[: benchmark.num_macros].numpy().astype(np.float64), benchmark
        )
        best_costs: Optional[Dict[str, object]] = None
        accepted = 0
        tried = 0
        accepted_log: List[str] = []

        for _pass in range(2):
            accepted_this_pass = 0
            owner = self._owner_positions_from_placement(current_full, benchmark)
            for i in candidate_ids:
                if tried >= max_moves:
                    break
                proposals = self._high_congestion_exact_polish_proposals(
                    i,
                    current_hard,
                    movable,
                    half_w,
                    half_h,
                    cw,
                    ch,
                    edges,
                    incident,
                    owner,
                    hard_neighbor_weights,
                    step_scales,
                    span,
                    hot_rows,
                    hot_cols,
                    benchmark,
                )
                for updates in proposals:
                    if tried >= max_moves:
                        break
                    changed_ids = [idx for idx, _xy in updates]
                    trial_hard = current_hard.copy()
                    for idx, xy in updates:
                        trial_hard[idx] = xy
                    trial_hard = self._clip_hard_np(trial_hard, benchmark)
                    if fixed_hard.any():
                        trial_hard[fixed_hard] = fixed_pos[fixed_hard]
                    trial_hard = self._repair_all_overlaps(trial_hard, movable, sizes, half_w, half_h, cw, ch)
                    if fixed_hard.any():
                        trial_hard[fixed_hard] = fixed_pos[fixed_hard]
                    if self._any_overlap(trial_hard, range(len(trial_hard)), sep_x, sep_y, gap=0.025):
                        continue
                    trial_full = build_placement(trial_hard)
                    trial_np = trial_full[: benchmark.num_macros].numpy().astype(np.float64)
                    trial_cheap_density = self._estimate_density_overflow_np(trial_np, benchmark)
                    if trial_cheap_density > current_cheap_density + 0.018:
                        continue

                    tried += 1
                    try:
                        costs = exact_score(trial_full)
                    except Exception:
                        continue
                    if int(costs["overlaps"]) == 0 and float(costs["score"]) < current_score - 1.0e-7:
                        improvement = current_proxy - float(costs["raw_proxy"])
                        current_score = float(costs["score"])
                        current_proxy = float(costs["raw_proxy"])
                        current_hard = trial_hard
                        current_full = trial_full
                        current_cheap_density = trial_cheap_density
                        best_costs = costs
                        accepted += 1
                        accepted_this_pass += 1
                        accepted_log.append(f"m{int(i)}:d={improvement:.6f}:p={current_proxy:.6f}")
                        break
                if tried >= max_moves:
                    break
            if accepted_this_pass == 0 or tried >= max_moves:
                break

        self.exact_polish_start_proxy = f"{base_proxy:.6f}"
        self.exact_polish_end_proxy = f"{current_proxy:.6f}"
        self.exact_polish_accepted_moves = accepted
        if accepted == 0 or best_costs is None:
            return None

        family, cheap_density, cheap_congestion, cheap_blockage, cheap_like = self._cheap_candidate_diagnostics(
            current_full, benchmark, edges, "soft_global_exact_polished+full_soft"
        )
        log = ",".join(accepted_log[:12])
        return {
            "hard": current_hard,
            "full": current_full,
            "score": current_score,
            "raw_proxy": current_proxy,
            "wirelength": float(best_costs["wirelength"]),
            "density": float(best_costs["density"]),
            "congestion": float(best_costs["congestion"]),
            "overlaps": int(best_costs["overlaps"]),
            "elapsed": float(best_costs["elapsed"]),
            "family": family,
            "cheap_density": cheap_density,
            "cheap_congestion": cheap_congestion,
            "cheap_blockage": cheap_blockage,
            "cheap_like": cheap_like,
            "accepted": accepted,
            "tried": tried,
            "source_label": source_label,
            "log": log,
        }

    def _high_congestion_exact_polish_candidate_ids(
        self,
        pos: np.ndarray,
        benchmark: Benchmark,
        movable: np.ndarray,
        sizes: np.ndarray,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
    ) -> Tuple[List[int], List[int], List[int]]:
        n = int(benchmark.num_hard_macros)
        movable_ids = np.where(movable[:n])[0]
        if len(movable_ids) == 0:
            return [], [], []
        h_grid, v_grid = self._routing_congestion_arrays_np(owner_pos, edges, benchmark)
        row_score = h_grid.sum(axis=1) + v_grid.sum(axis=1)
        col_score = h_grid.sum(axis=0) + v_grid.sum(axis=0)
        hot_row_count = max(1, min(5, int(math.ceil(len(row_score) * 0.18)))) if row_score.size else 0
        hot_col_count = max(1, min(7, int(math.ceil(len(col_score) * 0.18)))) if col_score.size else 0
        hot_rows = [int(x) for x in np.argsort(row_score)[-hot_row_count:]] if hot_row_count else []
        hot_cols = [int(x) for x in np.argsort(col_score)[-hot_col_count:]] if hot_col_count else []

        degree_score = np.zeros(n, dtype=np.float64)
        wire_score = np.zeros(n, dtype=np.float64)
        channel_score = np.zeros(n, dtype=np.float64)
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        cell_w = cw / cols
        cell_h = ch / rows

        def cell(point: np.ndarray) -> Tuple[int, int]:
            r = int(np.clip(math.floor(float(point[1]) / max(cell_h, 1.0e-9)), 0, rows - 1))
            c = int(np.clip(math.floor(float(point[0]) / max(cell_w, 1.0e-9)), 0, cols - 1))
            return r, c

        hot_row_set = set(hot_rows)
        hot_col_set = set(hot_cols)
        for a, b, w in edges:
            if a >= len(owner_pos) or b >= len(owner_pos):
                continue
            pa = owner_pos[a]
            pb = owner_pos[b]
            ra, ca = cell(pa)
            rb, cb = cell(pb)
            r0, r1 = sorted((ra, rb))
            c0, c1 = sorted((ca, cb))
            crosses_hot = any(r0 <= r <= r1 for r in hot_row_set) or any(c0 <= c <= c1 for c in hot_col_set)
            dist = abs(float(pa[0] - pb[0])) + abs(float(pa[1] - pb[1]))
            if a < n:
                degree_score[a] += w
                wire_score[a] += w * dist
                if crosses_hot:
                    channel_score[a] += w * (1.0 + dist / max(cw + ch, 1.0e-9))
            if b < n:
                degree_score[b] += w
                wire_score[b] += w * dist
                if crosses_hot:
                    channel_score[b] += w * (1.0 + dist / max(cw + ch, 1.0e-9))

        macro_area = sizes[:, 0] * sizes[:, 1]
        dense_score = self._macro_dense_bin_scores(pos, benchmark, sizes)
        near_score = np.zeros(n, dtype=np.float64)
        for i in range(n):
            r, c = cell(pos[i])
            row_dist = min((abs(r - hr) for hr in hot_rows), default=rows)
            col_dist = min((abs(c - hc) for hc in hot_cols), default=cols)
            near_score[i] = max(0.0, 1.0 - min(row_dist, col_dist) / 3.0)

        def norm(values: np.ndarray) -> np.ndarray:
            mx = float(np.max(values)) if values.size else 0.0
            return values / mx if mx > 1.0e-12 else values

        priority = (
            1.20 * norm(degree_score)
            + 1.05 * norm(channel_score)
            + 0.80 * norm(wire_score)
            + 0.55 * norm(macro_area)
            + 0.70 * norm(dense_score)
            + 0.60 * near_score
        )
        ordered = sorted((int(i) for i in movable_ids), key=lambda idx: priority[idx], reverse=True)
        cap = max(1, int(self.high_congestion_exact_polish_candidate_macros))
        return ordered[:cap], hot_rows, hot_cols

    def _high_congestion_exact_polish_proposals(
        self,
        i: int,
        pos: np.ndarray,
        movable: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: List[Edge],
        incident: List[List[int]],
        owner_pos: np.ndarray,
        hard_neighbor_weights: List[List[int]],
        step_scales: Tuple[float, ...],
        span: float,
        hot_rows: List[int],
        hot_cols: List[int],
        benchmark: Benchmark,
    ) -> List[List[Tuple[int, np.ndarray]]]:
        proposals = self._exact_polish_proposals(
            i,
            pos,
            movable,
            half_w,
            half_h,
            cw,
            ch,
            edges,
            incident,
            owner_pos,
            hard_neighbor_weights,
            step_scales,
            span,
        )
        grid_rows = max(1, int(benchmark.grid_rows))
        grid_cols = max(1, int(benchmark.grid_cols))
        cell_h = ch / max(float(grid_rows), 1.0)
        cell_w = cw / max(float(grid_cols), 1.0)
        for scale in (0.006, 0.012, 0.020):
            step = span * scale
            if hot_cols:
                centers = [(c + 0.5) * cell_w for c in hot_cols]
                nearest = min(centers, key=lambda x: abs(float(pos[i, 0]) - x))
                direction = 1.0 if float(pos[i, 0]) >= nearest else -1.0
                new = np.array(
                    [np.clip(pos[i, 0] + direction * step, half_w[i], cw - half_w[i]), pos[i, 1]],
                    dtype=np.float64,
                )
                if np.linalg.norm(new - pos[i]) > 1.0e-9:
                    proposals.insert(0, [(i, new)])
            if hot_rows:
                centers = [(r + 0.5) * cell_h for r in hot_rows]
                nearest = min(centers, key=lambda y: abs(float(pos[i, 1]) - y))
                direction = 1.0 if float(pos[i, 1]) >= nearest else -1.0
                new = np.array(
                    [pos[i, 0], np.clip(pos[i, 1] + direction * step, half_h[i], ch - half_h[i])],
                    dtype=np.float64,
                )
                if np.linalg.norm(new - pos[i]) > 1.0e-9:
                    proposals.insert(0, [(i, new)])
        deduped: List[List[Tuple[int, np.ndarray]]] = []
        seen = set()
        for proposal in proposals:
            key = tuple((idx, round(float(xy[0]), 5), round(float(xy[1]), 5)) for idx, xy in proposal)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(proposal)
        return deduped[:18]

    def _place_soft_macros(self, placement: torch.Tensor, benchmark: Benchmark, soft_neighbors):
        n_hard = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        if n_soft == 0 or not soft_neighbors:
            return placement

        out = placement.clone()
        all_pos = torch.cat([out[: benchmark.num_macros], benchmark.port_positions], dim=0)
        alpha = 0.22
        for soft_idx, nbrs in enumerate(soft_neighbors):
            owner = n_hard + soft_idx
            if owner >= benchmark.num_macros or benchmark.macro_fixed[owner] or not nbrs:
                continue
            sx = sy = sw = 0.0
            for other, weight in nbrs:
                if 0 <= other < all_pos.shape[0]:
                    sx += weight * float(all_pos[other, 0])
                    sy += weight * float(all_pos[other, 1])
                    sw += weight
            if sw <= 0.0:
                continue
            target_x = sx / sw
            target_y = sy / sw
            old = out[owner].clone()
            w, h = benchmark.macro_sizes[owner]
            x = (1.0 - alpha) * float(old[0]) + alpha * target_x
            y = (1.0 - alpha) * float(old[1]) + alpha * target_y
            out[owner, 0] = torch.clamp(torch.tensor(x, dtype=out.dtype), w / 2, benchmark.canvas_width - w / 2)
            out[owner, 1] = torch.clamp(torch.tensor(y, dtype=out.dtype), h / 2, benchmark.canvas_height - h / 2)
        return out


class _SearchState:
    """Mutable local-search state with incremental density/congestion grids."""

    def __init__(
        self,
        pos,
        sizes,
        benchmark,
        edges,
        incident,
        owner_pos,
        use_density: bool = True,
        use_congestion: bool = True,
    ):
        self.pos = pos
        self.sizes = sizes
        self.benchmark = benchmark
        self.edges = edges
        self.incident = incident
        self.owner_pos = owner_pos
        self.n_hard = benchmark.num_hard_macros
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)

        self.d_rows = max(8, min(24, int(benchmark.grid_rows)))
        self.d_cols = max(8, min(24, int(benchmark.grid_cols)))
        self.c_rows = max(8, min(16, int(benchmark.grid_rows)))
        self.c_cols = max(8, min(16, int(benchmark.grid_cols)))
        self.cell_area = (self.cw / self.d_cols) * (self.ch / self.d_rows)
        self.density_grid = np.zeros((self.d_rows, self.d_cols), dtype=np.float64)
        self.hard_density: List[Dict[Tuple[int, int], float]] = []

        soft_pos = benchmark.macro_positions[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        soft_sizes = benchmark.macro_sizes[benchmark.num_hard_macros : benchmark.num_macros].numpy().astype(np.float64)
        for p, s in zip(soft_pos, soft_sizes):
            for key, val in self._density_contrib(p[0], p[1], s[0], s[1]).items():
                self.density_grid[key] += val

        for i in range(self.n_hard):
            contrib = self._density_contrib(self.pos[i, 0], self.pos[i, 1], self.sizes[i, 0], self.sizes[i, 1])
            self.hard_density.append(contrib)
            for key, val in contrib.items():
                self.density_grid[key] += val

        self.density_target = max(0.72, float(np.mean(self.density_grid)) * 1.25)

        self.cong_grid = np.zeros((self.c_rows, self.c_cols), dtype=np.float64)
        self.edge_cong: List[Dict[Tuple[int, int], float]] = []
        for edge_id in range(len(edges)):
            contrib = self._edge_contrib(edge_id)
            self.edge_cong.append(contrib)
            for key, val in contrib.items():
                self.cong_grid[key] += val
        pos_cong = self.cong_grid[self.cong_grid > 0.0]
        self.cong_target = float(np.percentile(pos_cong, 70)) if pos_cong.size else 1.0
        self.cong_target = max(self.cong_target, 1.0e-6)

        self.w_wire = 1.0
        self.w_den = 0.10 if use_density else 0.0
        self.w_cong = 0.035 if use_congestion else 0.0
        self._wire = self._wire_cost_all()
        self._density = self._density_penalty_all()
        self._cong = self._cong_penalty_all()

    def total_cost(self) -> float:
        return self.w_wire * self._wire + self.w_den * self._density + self.w_cong * self._cong

    def apply(self, updates: List[Tuple[int, np.ndarray]]):
        changed = []
        old_wire = self._wire
        old_density = self._density
        old_cong = self._cong

        affected_edges = set()
        affected_density = set()
        affected_cong = set()
        old_edge_contribs = {}
        old_density_contribs = {}

        for idx, new_pos in updates:
            old_pos = self.pos[idx].copy()
            changed.append((idx, old_pos, new_pos.copy()))
            old_density_contribs[idx] = self.hard_density[idx]
            affected_density.update(self.hard_density[idx])
            for edge_id in self.incident[idx] if idx < len(self.incident) else []:
                affected_edges.add(edge_id)
                old_edge_contribs[edge_id] = self.edge_cong[edge_id]
                affected_cong.update(self.edge_cong[edge_id])

        for idx, _old, new_pos in changed:
            self.pos[idx] = new_pos

        # Recompute affected wire exactly from incident edges.
        self._wire += self._wire_delta(changed, old_positions={idx: old for idx, old, _new in changed})

        # Density update.
        for idx, old_contrib in old_density_contribs.items():
            for key, val in old_contrib.items():
                self.density_grid[key] -= val
            new_contrib = self._density_contrib(
                self.pos[idx, 0], self.pos[idx, 1], self.sizes[idx, 0], self.sizes[idx, 1]
            )
            self.hard_density[idx] = new_contrib
            for key, val in new_contrib.items():
                self.density_grid[key] += val
            affected_density.update(new_contrib)
        self._density += self._density_delta_for_cells(affected_density, before=None)

        # The density delta above needs before/after cell penalties, so recompute
        # the scalar. The grid is small and this avoids subtle bookkeeping bugs.
        self._density = self._density_penalty_all()

        # Congestion update.
        for edge_id, old_contrib in old_edge_contribs.items():
            for key, val in old_contrib.items():
                self.cong_grid[key] -= val
            new_contrib = self._edge_contrib(edge_id)
            self.edge_cong[edge_id] = new_contrib
            for key, val in new_contrib.items():
                self.cong_grid[key] += val
            affected_cong.update(new_contrib)
        self._cong = self._cong_penalty_all()

        for idx, old, new in changed:
            # Store rollback scalars on each record only once via tuple expansion
            pass
        self._last_old = (old_wire, old_density, old_cong, old_density_contribs, old_edge_contribs)
        return changed

    def rollback(self, changed):
        old_wire, old_density, old_cong, old_density_contribs, old_edge_contribs = self._last_old

        for edge_id, current_contrib in list(old_edge_contribs.items()):
            for key, val in self.edge_cong[edge_id].items():
                self.cong_grid[key] -= val
            self.edge_cong[edge_id] = current_contrib
            for key, val in current_contrib.items():
                self.cong_grid[key] += val

        for idx, old_contrib in old_density_contribs.items():
            for key, val in self.hard_density[idx].items():
                self.density_grid[key] -= val
            self.hard_density[idx] = old_contrib
            for key, val in old_contrib.items():
                self.density_grid[key] += val

        for idx, old, _new in changed:
            self.pos[idx] = old

        self._wire = old_wire
        self._density = old_density
        self._cong = old_cong

    def _point(self, owner: Owner):
        if owner < self.n_hard:
            return self.pos[owner]
        return self.owner_pos[owner]

    def _edge_wl(self, edge_id: int) -> float:
        a, b, w = self.edges[edge_id]
        pa = self._point(a)
        pb = self._point(b)
        return w * (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]))

    def _wire_cost_all(self) -> float:
        return sum(self._edge_wl(i) for i in range(len(self.edges)))

    def _wire_delta(self, changed, old_positions):
        affected = set()
        for idx, _old, _new in changed:
            if idx < len(self.incident):
                affected.update(self.incident[idx])
        if not affected:
            return 0.0
        new_sum = sum(self._edge_wl(edge_id) for edge_id in affected)
        old_sum = 0.0
        for edge_id in affected:
            a, b, w = self.edges[edge_id]
            pa = old_positions[a] if a in old_positions else self._point(a)
            pb = old_positions[b] if b in old_positions else self._point(b)
            old_sum += w * (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]))
        return new_sum - old_sum

    def _density_contrib(self, x, y, w, h) -> Dict[Tuple[int, int], float]:
        x0 = max(0.0, x - w / 2.0)
        x1 = min(self.cw, x + w / 2.0)
        y0 = max(0.0, y - h / 2.0)
        y1 = min(self.ch, y + h / 2.0)
        col0 = int(np.clip(math.floor(x0 / self.cw * self.d_cols), 0, self.d_cols - 1))
        col1 = int(np.clip(math.floor(max(x1 - 1.0e-9, 0.0) / self.cw * self.d_cols), 0, self.d_cols - 1))
        row0 = int(np.clip(math.floor(y0 / self.ch * self.d_rows), 0, self.d_rows - 1))
        row1 = int(np.clip(math.floor(max(y1 - 1.0e-9, 0.0) / self.ch * self.d_rows), 0, self.d_rows - 1))
        contrib: Dict[Tuple[int, int], float] = {}
        cell_w = self.cw / self.d_cols
        cell_h = self.ch / self.d_rows
        for r in range(row0, row1 + 1):
            cy0 = r * cell_h
            cy1 = cy0 + cell_h
            oy = max(0.0, min(y1, cy1) - max(y0, cy0))
            if oy <= 0.0:
                continue
            for c in range(col0, col1 + 1):
                cx0 = c * cell_w
                cx1 = cx0 + cell_w
                ox = max(0.0, min(x1, cx1) - max(x0, cx0))
                if ox > 0.0:
                    contrib[(r, c)] = (ox * oy) / self.cell_area
        return contrib

    def _density_penalty_all(self) -> float:
        overflow = np.maximum(0.0, self.density_grid - self.density_target)
        return float(np.sum(overflow * overflow))

    def _density_delta_for_cells(self, _cells, before):
        return 0.0

    def _edge_contrib(self, edge_id: int) -> Dict[Tuple[int, int], float]:
        a, b, w = self.edges[edge_id]
        pa = self._point(a)
        pb = self._point(b)
        x0, x1 = sorted((float(pa[0]), float(pb[0])))
        y0, y1 = sorted((float(pa[1]), float(pb[1])))
        col0 = int(np.clip(math.floor(x0 / self.cw * self.c_cols), 0, self.c_cols - 1))
        col1 = int(np.clip(math.floor(x1 / self.cw * self.c_cols), 0, self.c_cols - 1))
        row0 = int(np.clip(math.floor(y0 / self.ch * self.c_rows), 0, self.c_rows - 1))
        row1 = int(np.clip(math.floor(y1 / self.ch * self.c_rows), 0, self.c_rows - 1))
        cells = max(1, (row1 - row0 + 1) * (col1 - col0 + 1))
        val = w / cells
        return {(r, c): val for r in range(row0, row1 + 1) for c in range(col0, col1 + 1)}

    def _cong_penalty_all(self) -> float:
        overflow = np.maximum(0.0, self.cong_grid - self.cong_target)
        return float(np.sum(overflow * overflow))
