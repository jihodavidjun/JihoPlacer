"""Heuristic search candidate generator for v1 placements."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from jiho_place.v1.hotspot_micro_cd import Candidate, HotspotMicroCDGenerator


class HeuristicSearchGenerator(HotspotMicroCDGenerator):
    """The generator is deliberately candidate-only: it never mutates final
    selection semantics, and all exact decisions remain gated by runtime.
    """

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
        time_budget_s: float = 900.0,
        start_candidate: Optional[Candidate] = None,
        start_proxy: Optional[float] = None,
        exact_eval_time_s: Optional[float] = None,
    ) -> List[Tuple[Candidate, Dict[str, object], str]]:
        start = time.time()
        budget = max(30.0, float(time_budget_s))
        self.logs = []
        self.hotspot_ids = []
        self.stats.exact_evals = 0
        self.stats.cheap_evals = 0
        if not candidates or int(getattr(benchmark, "num_hard_macros", 0)) <= 0:
            self.logs.append("heuristic_search=skipped|reason=no_candidates")
            return []

        base = start_candidate if start_candidate is not None else min(candidates, key=lambda row: float(row[0]))
        base_full = self._candidate_full(base, benchmark).detach().cpu().clone()
        n_hard = int(benchmark.num_hard_macros)
        movable_idx = np.nonzero(np.asarray(movable, dtype=bool))[0].astype(np.int64)
        if movable_idx.size == 0:
            self.logs.append("heuristic_search=skipped|reason=no_movable")
            return []

        plc, compute_proxy_cost = self._load_exact_tools(benchmark)
        if exact_eval_time_s is not None:
            self.stats.exact_eval_time_s = float(exact_eval_time_s)
        else:
            self.stats.exact_eval_time_s = float("inf")
        exact_fast = (
            plc is not None
            and compute_proxy_cost is not None
            and self.stats.exact_eval_time_s <= 3.0
            and n_hard < 350
        )

        net_data = self._net_data_for(benchmark, plc, base_full.shape[0])
        hard = base_full[:n_hard].detach().cpu().numpy().astype(np.float64)
        state = self._state_from_full(engine, benchmark, base_full, hard, net_data)
        if exact_fast and start_proxy is not None:
            state = (*state[:-1], float(start_proxy))
        elif exact_fast:
            state = self._score_state_exact(state, benchmark, compute_proxy_cost, plc)
        hotspots = self._identify_hotspots(engine, hard, benchmark, movable_idx, net_data)
        degree_order = self._degree_order(movable_idx, net_data)
        boundary_order = self._boundary_order(movable_idx, hard, cw, ch)
        self.hotspot_ids = hotspots

        outputs: List[Tuple[Candidate, Dict[str, object], str]] = []
        best = state
        best_hard = hard.copy()
        exact_per_macro = max(1, int(os.environ.get("JIHO_HEURISTIC_EXACT_PER_MACRO", "6")))
        print(
            "[heuristic_search] "
            f"mode={'exact' if exact_fast else 'cheap'} budget_s={budget:.1f} "
            f"exact_eval_s={self.stats.exact_eval_time_s:.3f} hard={n_hard}",
            flush=True,
        )

        # Stage A: exact-gated CD on fast small/medium cases.
        if exact_fast and self._remaining(start, budget) > 5.0:
            ids = self._unique_ids(hotspots + degree_order[:24])
            stage_budget = min(self._remaining(start, budget), max(60.0, 0.35 * budget))
            stage_deadline = time.time() + stage_budget
            for label in ("heuristic_exact_cd_step1", "heuristic_exact_cd_step2"):
                if time.time() >= stage_deadline:
                    break
                before = best[-1]
                best = self._run_cd_state(
                    engine,
                    benchmark,
                    best,
                    ids,
                    net_data,
                    compute_proxy_cost,
                    plc,
                    sizes,
                    half_w,
                    half_h,
                    cw,
                    ch,
                    use_exact=True,
                    exact_per_macro=exact_per_macro,
                    start_time=start,
                    budget_s=min(budget, time.time() - start + max(1.0, stage_deadline - time.time())),
                )
                if best[-1] < before - 1.0e-12:
                    best_hard = best[1].copy()
                    outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, label, best[-1]))

        # Stage B: cheap-guided LNS families for medium/large and as backup for small.
        rng = np.random.default_rng(self.seed + 101)
        lns_variants = (
            ("heuristic_lns_hotspot", hotspots + degree_order[:16]),
            ("heuristic_lns_degree", degree_order),
            ("heuristic_lns_random", []),
        )
        lns_rounds = 8 if self._remaining(start, budget) > 300.0 else 4
        for label, priority in lns_variants:
            if self._remaining(start, budget) < 10.0:
                break
            candidate_state = self._best_lns_variant(
                engine,
                benchmark,
                best,
                movable_idx,
                priority,
                hotspots,
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                rounds=lns_rounds,
                rng=rng,
                start_time=start,
                budget_s=budget,
                exact_final=exact_fast,
            )
            if candidate_state is not None and candidate_state[-1] < best[-1] - 1.0e-12:
                best = candidate_state
                best_hard = best[1].copy()
                outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, label, best[-1]))

        # Stage C: legalization-aware push chains from hotspot macros.
        if self._remaining(start, budget) > 8.0:
            push_state = self._push_chain_candidate(
                engine,
                benchmark,
                best,
                hotspots[:24],
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                start,
                budget,
                exact_fast,
            )
            if push_state is not None and push_state[-1] < best[-1] - 1.0e-12:
                best = push_state
                best_hard = best[1].copy()
                outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, "heuristic_push_chain", best[-1]))

        # Stage D: structured ILS wrapper.
        if self._remaining(start, budget) > 12.0:
            ils_state = self._ils_candidate(
                engine,
                benchmark,
                best,
                movable_idx,
                hotspots,
                degree_order,
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                rng,
                start,
                budget,
                exact_fast,
            )
            if ils_state is not None and ils_state[-1] < best[-1] - 1.0e-12:
                best = ils_state
                outputs.append(self._candidate_output(engine, benchmark, best[0], best[1], edges, "heuristic_ils_best", best[-1]))

        elapsed = time.time() - start
        summary = (
            f"heuristic_search=done|mode={'exact' if exact_fast else 'cheap'}|"
            f"outputs={len(outputs)}|best={best[-1]:.6f}|exact_evals={self.stats.exact_evals}|"
            f"cheap_evals={self.stats.cheap_evals}|elapsed_s={elapsed:.3f}"
        )
        self.logs.append(summary)
        print(f"[heuristic_search] {summary}", flush=True)
        return outputs[:7]

    def _state_from_full(self, engine: Any, benchmark: Any, full: torch.Tensor, hard: np.ndarray, net_data: Any):
        positions = full.to(device=self.device, dtype=torch.float32).clone()
        net_hpwl = self._net_helper._compute_all_net_hpwl(net_data, positions)
        hpwl = float(net_hpwl.sum().detach().cpu().item())
        density = float(engine._estimate_density_overflow_np(full[: benchmark.num_macros].numpy(), benchmark))
        congestion = self._cheap_congestion(engine, full, benchmark)
        score = self._cheap_score(hpwl, density, congestion)
        return full, hard, positions, net_hpwl, hpwl, density, congestion, score

    def _score_state_exact(self, state, benchmark: Any, compute_proxy_cost: Any, plc: Any):
        full, hard, positions, net_hpwl, hpwl, density, congestion, score = state
        if compute_proxy_cost is None or plc is None:
            return state
        try:
            costs = compute_proxy_cost(full.detach().cpu(), benchmark, plc)
            self.stats.exact_evals += 1
            score = float(costs.get("proxy_cost", float("inf"))) + int(costs.get("overlap_count", 0)) * 1.0e6
        except Exception:
            pass
        return full, hard, positions, net_hpwl, hpwl, density, congestion, score

    def _run_cd_state(
        self,
        engine: Any,
        benchmark: Any,
        state,
        macro_ids: Sequence[int],
        net_data: Any,
        compute_proxy_cost: Any,
        plc: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        use_exact: bool,
        exact_per_macro: int,
        start_time: float,
        budget_s: float,
    ):
        full, hard, positions, net_hpwl, hpwl, density, congestion, score = state
        return self._run_micro_cd(
            engine=engine,
            benchmark=benchmark,
            full=full.clone(),
            hard=hard.copy(),
            macro_ids=list(macro_ids),
            net_data=net_data,
            positions=positions.clone(),
            net_hpwl=net_hpwl.clone(),
            current_hpwl=hpwl,
            current_density=density,
            current_congestion=congestion,
            current_score=score,
            use_exact=use_exact,
            exact_per_macro=exact_per_macro,
            compute_proxy_cost=compute_proxy_cost,
            plc=plc,
            sizes=sizes,
            half_w=half_w,
            half_h=half_h,
            cw=float(cw),
            ch=float(ch),
            start_time=start_time,
            budget_s=budget_s,
        )

    def _best_lns_variant(
        self,
        engine: Any,
        benchmark: Any,
        base_state,
        movable_idx: np.ndarray,
        priority: Sequence[int],
        hotspots: Sequence[int],
        net_data: Any,
        compute_proxy_cost: Any,
        plc: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        rounds: int,
        rng: np.random.Generator,
        start_time: float,
        budget_s: float,
        exact_final: bool,
    ):
        best = None
        subset_base = max(1, int(round(0.16 * len(movable_idx))))
        for round_idx in range(max(1, int(rounds))):
            if self._remaining(start_time, budget_s) < 5.0:
                break
            frac = 0.10 + 0.03 * (round_idx % 6)
            subset_size = min(len(movable_idx), max(subset_base, int(round(frac * len(movable_idx)))))
            subset = self._mixed_subset(movable_idx, priority, subset_size, rng)
            trial = self._pocket_perturb(engine, benchmark, base_state, subset, net_data, sizes, half_w, half_h, cw, ch, rng)
            repair_ids = self._unique_ids(list(subset) + list(hotspots[:32]))
            trial = self._run_cd_state(
                engine,
                benchmark,
                trial,
                repair_ids,
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                use_exact=False,
                exact_per_macro=1,
                start_time=start_time,
                budget_s=budget_s,
            )
            if exact_final and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
                trial = self._score_state_exact(trial, benchmark, compute_proxy_cost, plc)
            if trial[-1] < base_state[-1] - 1.0e-12 and (best is None or trial[-1] < best[-1]):
                best = trial
        return best

    def _pocket_perturb(
        self,
        engine: Any,
        benchmark: Any,
        base_state,
        subset: Sequence[int],
        net_data: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        rng: np.random.Generator,
    ):
        full, hard, _positions, _net_hpwl, _hpwl, _density, _congestion, _score = base_state
        trial_full = full.clone()
        trial_hard = hard.copy()
        span = math.sqrt(max(float(cw) * float(ch), 1.0e-12))
        for raw_i in subset:
            i = int(raw_i)
            old = trial_hard[i].copy()
            pocket = self._low_pocket(engine, benchmark, trial_hard, old, cw, ch, radius=7)
            alpha = float(rng.uniform(0.35, 0.85))
            proposed = old + (pocket - old) * alpha + rng.normal(0.0, 0.006 * span, size=2)
            proposed[0] = min(max(proposed[0], half_w[i]), cw - half_w[i])
            proposed[1] = min(max(proposed[1], half_h[i]), ch - half_h[i])
            if not self._single_overlap(i, proposed, trial_hard, sizes):
                trial_hard[i] = proposed
                trial_full[i] = torch.tensor(proposed, dtype=trial_full.dtype)
        return self._state_from_full(engine, benchmark, trial_full, trial_hard, net_data)

    def _push_chain_candidate(
        self,
        engine: Any,
        benchmark: Any,
        base_state,
        macro_ids: Sequence[int],
        net_data: Any,
        compute_proxy_cost: Any,
        plc: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        start_time: float,
        budget_s: float,
        exact_final: bool,
    ):
        best = None
        for raw_i in macro_ids:
            if self._remaining(start_time, budget_s) < 4.0:
                break
            full, hard, _positions, _net_hpwl, _hpwl, _density, _congestion, _score = base_state
            trial_full = full.clone()
            trial_hard = hard.copy()
            i = int(raw_i)
            target = self._low_pocket(engine, benchmark, trial_hard, trial_hard[i], cw, ch, radius=8)
            moved = self._try_push_chain(i, target, trial_hard, trial_full, sizes, half_w, half_h, cw, ch)
            if not moved:
                continue
            if self._any_overlap(trial_hard, sizes):
                continue
            trial = self._state_from_full(engine, benchmark, trial_full, trial_hard, net_data)
            if exact_final and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
                trial = self._score_state_exact(trial, benchmark, compute_proxy_cost, plc)
            if trial[-1] < base_state[-1] - 1.0e-12 and (best is None or trial[-1] < best[-1]):
                best = trial
        return best

    def _ils_candidate(
        self,
        engine: Any,
        benchmark: Any,
        base_state,
        movable_idx: np.ndarray,
        hotspots: Sequence[int],
        degree_order: Sequence[int],
        net_data: Any,
        compute_proxy_cost: Any,
        plc: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        rng: np.random.Generator,
        start_time: float,
        budget_s: float,
        exact_final: bool,
    ):
        best = None
        attempts = 5 if self._remaining(start_time, budget_s) > 180.0 else 3
        priority = self._unique_ids(list(hotspots) + list(degree_order[:48]))
        for attempt in range(attempts):
            if self._remaining(start_time, budget_s) < 8.0:
                break
            if attempt % 3 == 0:
                subset = np.asarray(priority[: max(8, min(len(priority), len(movable_idx) // 6))], dtype=np.int64)
            else:
                subset = self._mixed_subset(movable_idx, priority, max(8, len(movable_idx) // 5), rng)
            trial = self._group_escape(engine, benchmark, base_state, subset, net_data, sizes, half_w, half_h, cw, ch, rng, attempt)
            repair_ids = self._unique_ids(list(subset) + list(hotspots[:32]))
            trial = self._run_cd_state(
                engine,
                benchmark,
                trial,
                repair_ids,
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                use_exact=False,
                exact_per_macro=1,
                start_time=start_time,
                budget_s=budget_s,
            )
            if exact_final and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
                trial = self._score_state_exact(trial, benchmark, compute_proxy_cost, plc)
            if trial[-1] < base_state[-1] - 1.0e-12 and (best is None or trial[-1] < best[-1]):
                best = trial
        return best

    def _group_escape(
        self,
        engine: Any,
        benchmark: Any,
        base_state,
        subset: Sequence[int],
        net_data: Any,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        rng: np.random.Generator,
        attempt: int,
    ):
        full, hard, _positions, _net_hpwl, _hpwl, _density, _congestion, _score = base_state
        trial_full = full.clone()
        trial_hard = hard.copy()
        center = np.array([0.5 * cw, 0.5 * ch], dtype=np.float64)
        span = math.sqrt(max(float(cw) * float(ch), 1.0e-12))
        for raw_i in subset:
            i = int(raw_i)
            old = trial_hard[i].copy()
            if attempt % 3 == 0:
                direction = old - center
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1.0e-12 else rng.normal(0.0, 1.0, size=2)
                proposed = old + direction * (0.025 * span)
            elif attempt % 3 == 1:
                pocket = self._low_pocket(engine, benchmark, trial_hard, old, cw, ch, radius=10)
                proposed = old + (pocket - old) * 0.65
            else:
                proposed = old + rng.normal(0.0, [0.035 * cw, 0.035 * ch], size=2)
            proposed[0] = min(max(proposed[0], half_w[i]), cw - half_w[i])
            proposed[1] = min(max(proposed[1], half_h[i]), ch - half_h[i])
            if not self._single_overlap(i, proposed, trial_hard, sizes):
                trial_hard[i] = proposed
                trial_full[i] = torch.tensor(proposed, dtype=trial_full.dtype)
        return self._state_from_full(engine, benchmark, trial_full, trial_hard, net_data)

    def _try_push_chain(
        self,
        i: int,
        target: np.ndarray,
        hard: np.ndarray,
        full: torch.Tensor,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
    ) -> bool:
        current = int(i)
        desired = np.asarray(target, dtype=np.float64)
        direction = desired - hard[current]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            return False
        direction = direction / norm
        moved_any = False
        for _depth in range(3):
            desired[0] = min(max(desired[0], half_w[current]), cw - half_w[current])
            desired[1] = min(max(desired[1], half_h[current]), ch - half_h[current])
            collided = self._overlap_indices(current, desired, hard, sizes)
            hard[current] = desired.copy()
            full[current] = torch.tensor(desired, dtype=full.dtype)
            moved_any = True
            if not collided:
                return True
            current = int(collided[0])
            step = 0.65 * max(float(sizes[current, 0]), float(sizes[current, 1]))
            desired = hard[current] + direction * max(step, 0.01 * math.sqrt(max(cw * ch, 1.0e-12)))
        return moved_any

    def _overlap_indices(self, i: int, new_xy: np.ndarray, hard: np.ndarray, sizes: np.ndarray) -> List[int]:
        dx = np.abs(float(new_xy[0]) - hard[:, 0])
        dy = np.abs(float(new_xy[1]) - hard[:, 1])
        min_x = 0.5 * (float(sizes[i, 0]) + sizes[:, 0])
        min_y = 0.5 * (float(sizes[i, 1]) + sizes[:, 1])
        overlap = (dx < min_x) & (dy < min_y)
        overlap[i] = False
        return [int(j) for j in np.nonzero(overlap)[0].tolist()]

    def _any_overlap(self, hard: np.ndarray, sizes: np.ndarray) -> bool:
        n = int(hard.shape[0])
        for i in range(n):
            if self._overlap_indices(i, hard[i], hard, sizes):
                return True
        return False

    def _low_pocket(
        self,
        engine: Any,
        benchmark: Any,
        hard: np.ndarray,
        origin: np.ndarray,
        cw: float,
        ch: float,
        radius: int,
    ) -> np.ndarray:
        try:
            placement_np = self._candidate_macro_positions(hard, benchmark)
            h_grid, v_grid = engine._exact_style_congestion_arrays_np(placement_np, benchmark)
            grid = np.asarray(h_grid, dtype=np.float64) + np.asarray(v_grid, dtype=np.float64)
            rows, cols = grid.shape
            col = int(np.clip(math.floor(float(origin[0]) / max(cw, 1.0e-12) * cols), 0, cols - 1))
            row = int(np.clip(math.floor(float(origin[1]) / max(ch, 1.0e-12) * rows), 0, rows - 1))
            r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
            c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
            window = grid[r0:r1, c0:c1]
            rr, cc = np.unravel_index(int(np.argmin(window)), window.shape)
            low_r, low_c = r0 + int(rr), c0 + int(cc)
            return np.array([(low_c + 0.5) / max(cols, 1) * cw, (low_r + 0.5) / max(rows, 1) * ch], dtype=np.float64)
        except Exception:
            return np.asarray(origin, dtype=np.float64)

    def _mixed_subset(
        self,
        movable_idx: np.ndarray,
        priority: Sequence[int],
        subset_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        movable = [int(i) for i in np.asarray(movable_idx, dtype=np.int64).tolist()]
        movable_set = set(movable)
        target = min(max(1, int(subset_size)), len(movable))
        chosen: List[int] = []
        for raw_i in priority:
            i = int(raw_i)
            if i in movable_set and i not in chosen:
                chosen.append(i)
            if len(chosen) >= max(1, target // 2):
                break
        rest = np.array([i for i in movable if i not in set(chosen)], dtype=np.int64)
        if len(chosen) < target and rest.size:
            pick = rng.choice(rest, size=min(target - len(chosen), int(rest.size)), replace=False)
            chosen.extend(int(i) for i in pick.tolist())
        return np.asarray(chosen[:target], dtype=np.int64)

    def _degree_order(self, movable_idx: np.ndarray, net_data: Any) -> List[int]:
        degrees = np.array([len(nets) for nets in net_data.macro_to_nets], dtype=np.float64)
        return [int(i) for i in movable_idx[np.argsort(-degrees[movable_idx])].tolist()]

    def _boundary_order(self, movable_idx: np.ndarray, hard: np.ndarray, cw: float, ch: float) -> List[int]:
        center = np.array([0.5 * cw, 0.5 * ch], dtype=np.float64)
        norm = np.array([max(0.5 * cw, 1.0e-9), max(0.5 * ch, 1.0e-9)], dtype=np.float64)
        dist = np.max(np.abs((hard[movable_idx] - center) / norm), axis=1)
        return [int(i) for i in movable_idx[np.argsort(-dist)].tolist()]

    def _unique_ids(self, ids: Sequence[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for raw_i in ids:
            i = int(raw_i)
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _remaining(self, start_time: float, budget_s: float) -> float:
        return float(budget_s) - (time.time() - start_time)
