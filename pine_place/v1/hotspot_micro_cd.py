"""Hotspot-focused micro coordinate descent candidate generation."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pine_place.v1.sa_polish import SAPolisher, _SANetData


Candidate = Tuple[float, np.ndarray, bool, str, Optional[torch.Tensor]]


@dataclass
class HotspotCDStats:
    hotspot_count: int = 0
    moves_tried: int = 0
    moves_accepted: int = 0
    cheap_evals: int = 0
    exact_evals: int = 0
    exact_eval_time_s: float = float("inf")
    proxy_before: float = float("nan")
    proxy_after: float = float("nan")
    elapsed_s: float = 0.0


class HotspotMicroCDGenerator:
    """Small exact-gated CD/LNS candidate generator for hard macro hotspots."""

    _NET_CACHE: Dict[Tuple[str, str], _SANetData] = {}

    def __init__(self, device: Optional[str] = None, seed: int = 271828) -> None:
        self.device = self._resolve_device(device)
        self.seed = int(seed)
        self.stats = HotspotCDStats()
        self.hotspot_ids: List[int] = []
        self.logs: List[str] = []
        self._net_helper = SAPolisher(device=self.device, seed=seed)

    def generate(
        self,
        engine: Any,
        candidates: Sequence[Candidate],
        benchmark: Any,
        movable: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        edges: Sequence[Tuple[int, int, float]],
        time_budget_s: float = 180.0,
        start_candidate: Optional[Candidate] = None,
        start_proxy: Optional[float] = None,
        exact_eval_time_s: Optional[float] = None,
    ) -> List[Tuple[Candidate, Dict[str, object], str]]:
        start = time.time()
        self.stats = HotspotCDStats()
        self.logs = []
        self.hotspot_ids = []
        budget = max(1.0, float(time_budget_s))
        exact_per_macro = max(1, int(os.environ.get("PINE_HOTSPOT_CD_EXACT_PER_MACRO", "2")))
        if not candidates or int(getattr(benchmark, "num_hard_macros", 0)) <= 0:
            self.logs.append("skipped=no_candidates")
            return []

        base = start_candidate if start_candidate is not None else min(candidates, key=lambda row: float(row[0]))
        base_full = self._candidate_full(base, benchmark)
        base_full = base_full.detach().cpu().clone()
        n_hard = int(benchmark.num_hard_macros)
        hard = base_full[:n_hard].detach().cpu().numpy().astype(np.float64)
        movable_idx = np.nonzero(np.asarray(movable, dtype=bool))[0].astype(np.int64)
        if movable_idx.size == 0:
            self.logs.append("skipped=no_movable_hard")
            return []

        plc, compute_proxy_cost = self._load_exact_tools(benchmark)
        probe_score = float(start_proxy) if start_proxy is not None else float("inf")
        if exact_eval_time_s is not None:
            self.stats.exact_eval_time_s = float(exact_eval_time_s)
        if (
            start_proxy is None
            and exact_eval_time_s is None
            and plc is not None
            and compute_proxy_cost is not None
        ):
            t0 = time.perf_counter()
            try:
                probe = compute_proxy_cost(base_full, benchmark, plc)
                self.stats.exact_eval_time_s = time.perf_counter() - t0
                self.stats.exact_evals += 1
                probe_score = float(probe.get("proxy_cost", float("inf")))
            except Exception as exc:
                self.stats.exact_eval_time_s = float("inf")
                self.logs.append(f"exact_probe_failed={type(exc).__name__}")
        use_exact = self.stats.exact_eval_time_s <= 8.0 and plc is not None and compute_proxy_cost is not None

        net_data = self._net_data_for(benchmark, plc, base_full.shape[0])
        positions = base_full.to(device=self.device, dtype=torch.float32).clone()
        net_hpwl = self._net_helper._compute_all_net_hpwl(net_data, positions)
        current_hpwl = float(net_hpwl.sum().detach().cpu().item())
        current_density = float(engine._estimate_density_overflow_np(base_full[: benchmark.num_macros].numpy(), benchmark))
        current_congestion = self._cheap_congestion(engine, base_full, benchmark)
        current_score = (
            probe_score
            if use_exact and math.isfinite(probe_score)
            else self._cheap_score(current_hpwl, current_density, current_congestion)
        )
        self.stats.proxy_before = float(probe_score if math.isfinite(probe_score) else current_score)

        hotspots = self._identify_hotspots(engine, hard, benchmark, movable_idx, net_data)
        self.hotspot_ids = hotspots
        self.stats.hotspot_count = len(hotspots)
        print(
            "[hotspot_micro_cd] "
            f"hotspot_macros={len(hotspots)} exact_eval_time_s={self.stats.exact_eval_time_s:.3f} "
            f"mode={'exact' if use_exact else 'cheap'} budget_s={budget:.1f}",
            flush=True,
        )
        if not hotspots:
            self.logs.append("skipped=no_hotspots")
            return []

        outputs: List[Tuple[Candidate, Dict[str, object], str]] = []
        best_full = base_full.clone()
        best_hard = hard.copy()
        best_score = current_score
        best_density = current_density
        best_congestion = current_congestion
        best_hpwl = current_hpwl
        best_net_hpwl = net_hpwl.clone()
        best_positions = positions.clone()

        for label in ("hotspot_micro_cd_step1", "hotspot_micro_cd_step2"):
            if time.time() - start >= budget:
                break
            before_accepts = self.stats.moves_accepted
            state = self._run_micro_cd(
                engine=engine,
                benchmark=benchmark,
                full=best_full,
                hard=best_hard,
                macro_ids=hotspots,
                net_data=net_data,
                positions=best_positions,
                net_hpwl=best_net_hpwl,
                current_hpwl=best_hpwl,
                current_density=best_density,
                current_congestion=best_congestion,
                current_score=best_score,
                use_exact=use_exact,
                exact_per_macro=exact_per_macro,
                compute_proxy_cost=compute_proxy_cost,
                plc=plc,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=float(cw),
                ch=float(ch),
                start_time=start,
                budget_s=budget,
            )
            best_full, best_hard, best_positions, best_net_hpwl, best_hpwl, best_density, best_congestion, best_score = state
            if self.stats.moves_accepted > before_accepts:
                outputs.append(self._candidate_output(engine, benchmark, best_full, best_hard, edges, label, best_score))

        lns_full, lns_hard, lns_positions, lns_net_hpwl, lns_hpwl, lns_density, lns_congestion, lns_score = (
            best_full.clone(),
            best_hard.copy(),
            best_positions.clone(),
            best_net_hpwl.clone(),
            best_hpwl,
            best_density,
            best_congestion,
            best_score,
        )
        lns_best: Optional[Tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, float, float, float, float]] = None
        rng = np.random.default_rng(self.seed + 17)
        subset_size = max(1, int(round(0.20 * len(movable_idx))))
        if budget - (time.time() - start) >= 0.25 * budget:
            lns_iterations = 3
        else:
            lns_iterations = 0
        for _iteration in range(lns_iterations):
            if time.time() - start >= budget:
                break
            trial_full = lns_full.clone()
            trial_hard = lns_hard.copy()
            subset = self._lns_subset(movable_idx, hotspots, net_data, subset_size, rng)
            trial_hard, trial_full = self._perturb_subset(
                trial_hard, trial_full, subset, sizes, half_w, half_h, float(cw), float(ch), rng
            )
            trial_positions = trial_full.to(device=self.device, dtype=torch.float32)
            trial_net_hpwl = self._net_helper._compute_all_net_hpwl(net_data, trial_positions)
            trial_hpwl = float(trial_net_hpwl.sum().detach().cpu().item())
            trial_density = float(engine._estimate_density_overflow_np(trial_full[: benchmark.num_macros].numpy(), benchmark))
            trial_congestion = self._cheap_congestion(engine, trial_full, benchmark)
            trial_score = self._cheap_score(trial_hpwl, trial_density, trial_congestion)
            state = self._run_micro_cd(
                engine=engine,
                benchmark=benchmark,
                full=trial_full,
                hard=trial_hard,
                macro_ids=subset.tolist(),
                net_data=net_data,
                positions=trial_positions,
                net_hpwl=trial_net_hpwl,
                current_hpwl=trial_hpwl,
                current_density=trial_density,
                current_congestion=trial_congestion,
                current_score=trial_score,
                use_exact=False,
                exact_per_macro=exact_per_macro,
                compute_proxy_cost=compute_proxy_cost,
                plc=plc,
                sizes=sizes,
                half_w=half_w,
                half_h=half_h,
                cw=float(cw),
                ch=float(ch),
                start_time=start,
                budget_s=budget,
            )
            trial_full, trial_hard, trial_positions, trial_net_hpwl, trial_hpwl, trial_density, trial_congestion, trial_score = state
            trial_score = self._score_full(
                full=trial_full,
                benchmark=benchmark,
                compute_proxy_cost=compute_proxy_cost,
                plc=plc,
                use_exact=use_exact,
                hpwl=trial_hpwl,
                density=trial_density,
                congestion=trial_congestion,
            )
            if trial_score < lns_score:
                lns_full, lns_hard, lns_positions, lns_net_hpwl, lns_hpwl, lns_density, lns_congestion = (
                    trial_full,
                    trial_hard,
                    trial_positions,
                    trial_net_hpwl,
                    trial_hpwl,
                    trial_density,
                    trial_congestion,
                )
                lns_score = trial_score
                lns_best = (
                    trial_full,
                    trial_hard,
                    trial_positions,
                    trial_net_hpwl,
                    trial_hpwl,
                    trial_density,
                    trial_congestion,
                    trial_score,
                )

        if lns_best is not None:
            best_full, best_hard, best_positions, best_net_hpwl, best_hpwl, best_density, best_congestion, best_score = lns_best
            outputs.append(self._candidate_output(engine, benchmark, best_full, best_hard, edges, "hotspot_micro_cd_lns", best_score))

        self.stats.proxy_after = float(best_score)
        self.stats.elapsed_s = time.time() - start
        summary = (
            f"hotspots={self.stats.hotspot_count}|moves_tried={self.stats.moves_tried}|"
            f"moves_accepted={self.stats.moves_accepted}|cheap_evals={self.stats.cheap_evals}|"
            f"exact_evals={self.stats.exact_evals}|exact_eval_s={self.stats.exact_eval_time_s:.3f}|"
            f"proxy_before={self.stats.proxy_before:.6f}|proxy_after={self.stats.proxy_after:.6f}|"
            f"elapsed_s={self.stats.elapsed_s:.3f}"
        )
        self.logs.append(summary)
        print(f"[hotspot_micro_cd] {summary}", flush=True)
        return outputs[:3]

    def _run_micro_cd(
        self,
        engine: Any,
        benchmark: Any,
        full: torch.Tensor,
        hard: np.ndarray,
        macro_ids: Sequence[int],
        net_data: _SANetData,
        positions: torch.Tensor,
        net_hpwl: torch.Tensor,
        current_hpwl: float,
        current_density: float,
        current_congestion: float,
        current_score: float,
        use_exact: bool,
        exact_per_macro: int,
        compute_proxy_cost: Any,
        plc: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        start_time: float,
        budget_s: float,
    ) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, float, float, float, float]:
        span = math.sqrt(max(cw * ch, 1.0e-12))
        n_hard = int(benchmark.num_hard_macros)
        for order, macro_id in enumerate(macro_ids):
            if order % 5 == 0 and time.time() - start_time >= budget_s:
                break
            i = int(macro_id)
            if i < 0 or i >= n_hard:
                continue
            best = None
            affected = net_data.macro_to_nets[i] if i < len(net_data.macro_to_nets) else []
            old_pos = positions[i].clone()
            proposals = []
            for new_xy in self._proposal_positions(engine, full, hard, i, sizes, half_w, half_h, cw, ch, benchmark, span):
                if time.time() - start_time >= budget_s:
                    break
                self.stats.moves_tried += 1
                if np.allclose(new_xy, hard[i], atol=1.0e-12):
                    continue
                if self._single_overlap(i, new_xy, hard, sizes):
                    continue
                proposal_positions = positions.clone()
                proposal_positions[i] = torch.tensor(new_xy, dtype=positions.dtype, device=positions.device)
                rows, new_hpwl_values = self._net_helper._compute_affected_net_hpwl(net_data, proposal_positions, affected)
                if int(rows.numel()) > 0:
                    old_contribution = net_hpwl.index_select(0, rows).sum()
                    delta_hpwl = float((new_hpwl_values.sum() - old_contribution).detach().cpu().item())
                else:
                    old_contribution = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
                    new_hpwl_values = torch.zeros((0,), dtype=positions.dtype, device=positions.device)
                    delta_hpwl = 0.0
                if delta_hpwl > 0.015 * max(current_hpwl, 1.0e-12):
                    continue

                proposal_full = full.clone()
                proposal_full[i] = torch.tensor(new_xy, dtype=proposal_full.dtype)
                new_hpwl = current_hpwl + delta_hpwl
                new_density = float(engine._estimate_density_overflow_np(proposal_full[: benchmark.num_macros].numpy(), benchmark))
                new_congestion = self._cheap_congestion(engine, proposal_full, benchmark)
                cheap_delta = self._cheap_score(new_hpwl, new_density, new_congestion) - self._cheap_score(
                    current_hpwl, current_density, current_congestion
                )
                proposals.append(
                    (cheap_delta, new_xy.copy(), rows, new_hpwl_values, old_contribution, new_hpwl, new_density, new_congestion, proposal_full)
                )
            if use_exact:
                eval_pool = sorted(proposals, key=lambda row: float(row[0]))[: max(1, int(exact_per_macro))]
            else:
                eval_pool = proposals
            for cheap_delta, new_xy, rows, new_hpwl_values, old_contribution, new_hpwl, new_density, new_congestion, proposal_full in eval_pool:
                if time.time() - start_time >= budget_s:
                    break
                if use_exact:
                    score = self._score_full(
                        full=proposal_full,
                        benchmark=benchmark,
                        compute_proxy_cost=compute_proxy_cost,
                        plc=plc,
                        use_exact=True,
                        hpwl=new_hpwl,
                        density=new_density,
                        congestion=new_congestion,
                    )
                    delta_score = score - current_score
                else:
                    self.stats.cheap_evals += 1
                    delta_score = cheap_delta
                    score = current_score + delta_score
                if delta_score < -1.0e-12 and (best is None or score < best[0]):
                    best = (score, new_xy.copy(), rows, new_hpwl_values, old_contribution, new_hpwl, new_density, new_congestion)
            if best is None:
                positions[i] = old_pos
                continue
            score, new_xy, rows, new_hpwl_values, _old_contribution, current_hpwl, current_density, current_congestion = best
            hard[i] = new_xy
            full[i] = torch.tensor(new_xy, dtype=full.dtype)
            positions[i] = torch.tensor(new_xy, dtype=positions.dtype, device=positions.device)
            if int(rows.numel()) > 0:
                net_hpwl[rows] = new_hpwl_values
            current_score = score
            self.stats.moves_accepted += 1
        return full, hard, positions, net_hpwl, current_hpwl, current_density, current_congestion, current_score

    def _identify_hotspots(
        self,
        engine: Any,
        hard: np.ndarray,
        benchmark: Any,
        movable_idx: np.ndarray,
        net_data: _SANetData,
    ) -> List[int]:
        placement_np = self._candidate_macro_positions(hard, benchmark)
        h_grid, v_grid = engine._exact_style_congestion_arrays_np(placement_np, benchmark)
        grid = h_grid + v_grid
        rows, cols = grid.shape
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].detach().cpu().numpy().astype(np.float64)
        cong_scores = np.zeros((benchmark.num_hard_macros,), dtype=np.float64)
        for i in movable_idx.tolist():
            x0 = max(0, int(math.floor((hard[i, 0] - 0.5 * sizes[i, 0]) / max(cw, 1.0e-12) * cols)))
            x1 = min(cols - 1, int(math.floor((hard[i, 0] + 0.5 * sizes[i, 0]) / max(cw, 1.0e-12) * cols)))
            y0 = max(0, int(math.floor((hard[i, 1] - 0.5 * sizes[i, 1]) / max(ch, 1.0e-12) * rows)))
            y1 = min(rows - 1, int(math.floor((hard[i, 1] + 0.5 * sizes[i, 1]) / max(ch, 1.0e-12) * rows)))
            if x1 >= x0 and y1 >= y0:
                cong_scores[i] = float(grid[y0 : y1 + 1, x0 : x1 + 1].sum())
        degree = np.array([len(nets) for nets in net_data.macro_to_nets[: benchmark.num_hard_macros]], dtype=np.float64)
        max_cong = float(np.max(cong_scores[movable_idx])) if movable_idx.size else 0.0
        max_degree = float(np.max(degree[movable_idx])) if movable_idx.size else 0.0
        cong_norm = cong_scores / max(max_cong, 1.0e-12)
        degree_norm = degree / max(max_degree, 1.0e-12)
        combined = 0.6 * cong_norm + 0.4 * degree_norm
        ordered = movable_idx[np.argsort(-combined[movable_idx])]
        return [int(i) for i in ordered[: min(48, len(ordered))]]

    def _candidate_output(
        self,
        engine: Any,
        benchmark: Any,
        full: torch.Tensor,
        hard: np.ndarray,
        edges: Sequence[Tuple[int, int, float]],
        label: str,
        objective_score: float,
    ) -> Tuple[Candidate, Dict[str, object], str]:
        hard = hard.copy()
        owner_pos = engine._owner_positions_from_placement(full, benchmark)
        surrogate = engine._surrogate_cost(hard, list(edges), owner_pos, benchmark, benchmark.macro_sizes[: benchmark.num_hard_macros].numpy().astype(np.float64))
        candidate: Candidate = (surrogate, hard, True, label, full.detach().cpu().clone())
        record = engine._soft_global_preselect_record(candidate, owner_pos, list(edges), benchmark, float(objective_score), 0.0)
        log = (
            f"{label}|score={float(objective_score):.6f}|moves={self.stats.moves_accepted}|"
            f"exact_eval_s={self.stats.exact_eval_time_s:.3f}"
        )
        return candidate, record, log

    def _proposal_positions(
        self,
        engine: Any,
        full: torch.Tensor,
        hard: np.ndarray,
        i: int,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        benchmark: Any,
        span: float,
    ) -> List[np.ndarray]:
        old = hard[i].astype(np.float64, copy=True)
        proposals: List[np.ndarray] = []

        def add(xy: np.ndarray) -> None:
            clipped = np.asarray(xy, dtype=np.float64).copy()
            clipped[0] = min(max(clipped[0], float(half_w[i])), float(cw) - float(half_w[i]))
            clipped[1] = min(max(clipped[1], float(half_h[i])), float(ch) - float(half_h[i]))
            key = (round(float(clipped[0]), 7), round(float(clipped[1]), 7))
            if key not in seen:
                seen.add(key)
                proposals.append(clipped)

        seen: set[Tuple[float, float]] = set()
        for direction in self._directions():
            for scale in (0.003, 0.008, 0.018):
                add(old + direction * (scale * span))

        try:
            placement_np = self._candidate_macro_positions(hard, benchmark)
            h_grid, v_grid = engine._exact_style_congestion_arrays_np(placement_np, benchmark)
            grid = np.asarray(h_grid, dtype=np.float64) + np.asarray(v_grid, dtype=np.float64)
            rows, cols = grid.shape
            if rows > 0 and cols > 0:
                col = int(np.clip(math.floor(old[0] / max(float(cw), 1.0e-12) * cols), 0, cols - 1))
                row = int(np.clip(math.floor(old[1] / max(float(ch), 1.0e-12) * rows), 0, rows - 1))
                radius = 5
                r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
                c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
                window = grid[r0:r1, c0:c1]
                if window.size:
                    local_min = float(np.min(window))
                    weights = np.maximum(window - local_min, 0.0)
                    yy, xx = np.mgrid[r0:r1, c0:c1]
                    cell_x = (xx + 0.5) / max(cols, 1) * float(cw)
                    cell_y = (yy + 0.5) / max(rows, 1) * float(ch)
                    weight_sum = float(np.sum(weights))
                    if weight_sum > 1.0e-12:
                        centroid = np.array(
                            [
                                float(np.sum(cell_x * weights) / weight_sum),
                                float(np.sum(cell_y * weights) / weight_sum),
                            ],
                            dtype=np.float64,
                        )
                        away = old - centroid
                        away_norm = float(np.linalg.norm(away))
                        if away_norm > 1.0e-12:
                            add(old + away / away_norm * (0.012 * span))

                    low_local = np.unravel_index(int(np.argmin(window)), window.shape)
                    low_row = r0 + int(low_local[0])
                    low_col = c0 + int(low_local[1])
                    low_xy = np.array(
                        [
                            (low_col + 0.5) / max(cols, 1) * float(cw),
                            (low_row + 0.5) / max(rows, 1) * float(ch),
                        ],
                        dtype=np.float64,
                    )
                    to_low = low_xy - old
                    dist = float(np.linalg.norm(to_low))
                    if dist > 1.0e-12:
                        unit = to_low / dist
                        add(old + unit * min(dist, 0.012 * span))
                        add(old + unit * min(dist, 0.030 * span))
                        add(low_xy)
        except Exception:
            pass

        return proposals

    def _cheap_score(self, hpwl: float, density: float, congestion: float) -> float:
        return float(hpwl) + 0.5 * float(density) + float(congestion)

    def _cheap_congestion(self, engine: Any, full: torch.Tensor, benchmark: Any) -> float:
        try:
            placement_np = full[: benchmark.num_macros].detach().cpu().numpy().astype(np.float64)
            return float(engine._estimate_exact_style_congestion_overflow_np(placement_np, benchmark))
        except Exception:
            return 0.0

    def _lns_subset(
        self,
        movable_idx: np.ndarray,
        hotspots: Sequence[int],
        net_data: _SANetData,
        subset_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        movable_idx = np.asarray(movable_idx, dtype=np.int64)
        if movable_idx.size == 0:
            return movable_idx
        target = min(max(1, int(subset_size)), int(movable_idx.size))
        movable_set = {int(i) for i in movable_idx.tolist()}
        degrees = np.array([len(nets) for nets in net_data.macro_to_nets], dtype=np.float64)
        degree_order = [int(i) for i in movable_idx[np.argsort(-degrees[movable_idx])].tolist()]

        priority: List[int] = []
        for i in list(hotspots) + degree_order:
            ii = int(i)
            if ii in movable_set and ii not in priority:
                priority.append(ii)

        priority_target = min(target, max(1, target // 2), len(priority))
        chosen = priority[:priority_target]
        remaining = np.array([int(i) for i in movable_idx.tolist() if int(i) not in set(chosen)], dtype=np.int64)
        random_needed = target - len(chosen)
        if random_needed > 0 and remaining.size > 0:
            random_pick = rng.choice(remaining, size=min(random_needed, int(remaining.size)), replace=False)
            chosen.extend(int(i) for i in random_pick.tolist())
        return np.asarray(chosen[:target], dtype=np.int64)

    def _score_full(
        self,
        full: torch.Tensor,
        benchmark: Any,
        compute_proxy_cost: Any,
        plc: Any,
        use_exact: bool,
        hpwl: float,
        density: float,
        congestion: float,
    ) -> float:
        if use_exact and compute_proxy_cost is not None and plc is not None:
            try:
                costs = compute_proxy_cost(full.detach().cpu(), benchmark, plc)
                self.stats.exact_evals += 1
                return float(costs.get("proxy_cost", float("inf"))) + int(costs.get("overlap_count", 0)) * 1.0e6
            except Exception:
                return float("inf")
        self.stats.cheap_evals += 1
        return self._cheap_score(hpwl, density, congestion)

    def _perturb_subset(
        self,
        hard: np.ndarray,
        full: torch.Tensor,
        subset: Sequence[int],
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        sigma = np.array([0.05 * cw, 0.05 * ch], dtype=np.float64)
        for i_raw in subset:
            i = int(i_raw)
            old = hard[i].copy()
            proposed = old + rng.normal(0.0, sigma, size=2)
            proposed[0] = min(max(proposed[0], half_w[i]), cw - half_w[i])
            proposed[1] = min(max(proposed[1], half_h[i]), ch - half_h[i])
            if self._single_overlap(i, proposed, hard, sizes):
                continue
            hard[i] = proposed
            full[i] = torch.tensor(proposed, dtype=full.dtype)
        return hard, full

    def _single_overlap(self, i: int, new_xy: np.ndarray, hard: np.ndarray, sizes: np.ndarray) -> bool:
        hard_t = torch.as_tensor(hard, dtype=torch.float32, device=self.device)
        new_t = torch.tensor(new_xy, dtype=torch.float32, device=self.device)
        sizes_t = torch.as_tensor(sizes, dtype=torch.float32, device=self.device)
        dx = torch.abs(new_t[0] - hard_t[:, 0])
        dy = torch.abs(new_t[1] - hard_t[:, 1])
        min_x = 0.5 * (sizes_t[i, 0] + sizes_t[:, 0])
        min_y = 0.5 * (sizes_t[i, 1] + sizes_t[:, 1])
        overlap = (dx < min_x) & (dy < min_y)
        overlap[i] = False
        return bool(overlap.any().detach().cpu().item())

    def _candidate_full(self, candidate: Candidate, benchmark: Any) -> torch.Tensor:
        _surrogate, hard, _force, _label, full = candidate
        if full is not None:
            return full.clone()
        placement = benchmark.macro_positions.clone()
        placement[: benchmark.num_hard_macros] = torch.tensor(hard, dtype=torch.float32)
        return placement

    def _candidate_macro_positions(self, hard: np.ndarray, benchmark: Any) -> np.ndarray:
        placement = benchmark.macro_positions[: benchmark.num_macros].detach().cpu().numpy().astype(np.float64)
        placement[: benchmark.num_hard_macros] = hard
        return placement

    def _net_data_for(self, benchmark: Any, plc: Any, n_macros: int) -> _SANetData:
        key = (str(getattr(benchmark, "name", "benchmark")), str(self.device))
        cached = self._NET_CACHE.get(key)
        if cached is not None:
            return cached
        net_data = self._net_helper._net_data_for(benchmark, plc, self.device, torch.float32, n_macros)
        self._NET_CACHE[key] = net_data
        return net_data

    def _load_exact_tools(self, benchmark: Any) -> Tuple[Any, Any]:
        try:
            from macro_place.objective import compute_proxy_cost
            from pine_place.v1.current_engine import _load_plc_for_exact

            return _load_plc_for_exact(str(getattr(benchmark, "name", ""))), compute_proxy_cost
        except Exception:
            return None, None

    def _directions(self) -> List[np.ndarray]:
        inv = 1.0 / math.sqrt(2.0)
        return [
            np.array([1.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, -1.0], dtype=np.float64),
            np.array([inv, inv], dtype=np.float64),
            np.array([-inv, -inv], dtype=np.float64),
            np.array([inv, -inv], dtype=np.float64),
            np.array([-inv, inv], dtype=np.float64),
        ]

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
