"""Budgeted heuristic search candidate generator for v1 placements."""

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
        budget = min(max(30.0, float(time_budget_s)), float(os.environ.get("JIHO_HEURISTIC_MAX_TIME", "3300")))
        self.logs = []
        self.hotspot_ids = []
        self.stats.exact_evals = 0
        self.stats.cheap_evals = 0
        self._heuristic_exact_cap = 0
        self._stage_log_every_s = max(10.0, float(os.environ.get("JIHO_HEURISTIC_STAGE_LOG_EVERY", "30")))
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
        exact_affordable = (
            plc is not None
            and compute_proxy_cost is not None
            and self.stats.exact_eval_time_s <= 8.0
            and n_hard < 550
        )
        if exact_fast:
            self._heuristic_exact_cap = max(
                8,
                int(os.environ.get("JIHO_HEURISTIC_EXACT_CAP", str(max(20, min(140, int(0.28 * budget / max(self.stats.exact_eval_time_s, 1.0e-6))))))),
            )
        elif exact_affordable:
            self._heuristic_exact_cap = max(
                2,
                int(os.environ.get("JIHO_HEURISTIC_EXACT_CAP", str(max(2, min(24, int(0.08 * budget / max(self.stats.exact_eval_time_s, 1.0e-6))))))),
            )
        else:
            self._heuristic_exact_cap = max(0, int(os.environ.get("JIHO_HEURISTIC_EXACT_CAP", "0")))

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
        cong_bin_order = self._congestion_bin_order(engine, benchmark, hard, movable_idx)
        self.hotspot_ids = hotspots

        outputs: List[Tuple[Candidate, Dict[str, object], str]] = []
        best = state
        best_hard = hard.copy()
        exact_per_macro = max(1, int(os.environ.get("JIHO_HEURISTIC_EXACT_PER_MACRO", "6")))
        cd_patience = max(8, int(os.environ.get("JIHO_HEURISTIC_CD_PATIENCE_MACROS", "64")))
        stage_budgets = self._stage_budgets(budget, exact_fast, exact_affordable, n_hard)
        print(
            "[heuristic_search] "
            f"mode={'exact' if exact_fast else 'cheap'} budget_s={budget:.1f} "
            f"exact_eval_s={self.stats.exact_eval_time_s:.3f} hard={n_hard} "
            f"exact_cap={self._heuristic_exact_cap}",
            flush=True,
        )

        # Stage A: exact-gated CD on fast small/medium cases.
        if exact_affordable and self._remaining(start, budget) > 5.0 and stage_budgets["exact_cd"] > 1.0:
            stage_start = time.time()
            self._stage_log("exact_cd", "start", start, budget, best[-1], moves=0)
            ids = self._unique_ids(hotspots + degree_order[:24])
            stage_budget = min(self._remaining(start, budget), stage_budgets["exact_cd"])
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
                    use_exact=exact_fast,
                    exact_per_macro=exact_per_macro,
                    start_time=start,
                    budget_s=min(budget, time.time() - start + max(1.0, stage_deadline - time.time())),
                    stage_name="exact_cd",
                    patience_macros=cd_patience,
                )
                if best[-1] < before - 1.0e-12:
                    best_hard = best[1].copy()
                    outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, label, best[-1]))
                self._stage_log("exact_cd", "heartbeat", start, budget, best[-1], moves=self.stats.moves_accepted)
            self._stage_log("exact_cd", "end", start, budget, best[-1], moves=self.stats.moves_accepted, stage_elapsed=time.time() - stage_start)

        # Stage B: cheap-guided LNS families for medium/large and as backup for small.
        rng = np.random.default_rng(self.seed + 101)
        lns_variants = (
            ("heuristic_lns_hotspot", hotspots + degree_order[:16]),
            ("heuristic_lns_degree", degree_order),
            ("heuristic_lns_cong_bin", cong_bin_order),
            ("heuristic_lns_window", []),
            ("heuristic_lns_boundary", boundary_order),
            ("heuristic_lns_v23", self._unique_ids(boundary_order[:24] + hotspots[:24] + degree_order[:16])),
            ("heuristic_lns_random", []),
        )
        self._stage_log("lns", "start", start, budget, best[-1], moves=self.stats.moves_accepted)
        lns_rounds = 8 if stage_budgets["lns"] > 240.0 else 4
        lns_stage_end = time.time() + min(stage_budgets["lns"], self._remaining(start, budget))
        for label, priority in lns_variants:
            if self._remaining(start, budget) < 10.0 or time.time() >= lns_stage_end:
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
                budget_s=min(budget, time.time() - start + max(1.0, lns_stage_end - time.time())),
                exact_final=exact_affordable,
                variant=label,
            )
            if candidate_state is not None and candidate_state[-1] < best[-1] - 1.0e-12:
                best = candidate_state
                best_hard = best[1].copy()
                outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, label, best[-1]))
            self._stage_log("lns", "heartbeat", start, budget, best[-1], moves=self.stats.moves_accepted)
        self._stage_log("lns", "end", start, budget, best[-1], moves=self.stats.moves_accepted)

        # Stage C: legalization-aware push chains from hotspot macros.
        if self._remaining(start, budget) > 8.0 and stage_budgets["push"] > 1.0:
            self._stage_log("push_chain", "start", start, budget, best[-1], moves=self.stats.moves_accepted)
            push_budget = min(stage_budgets["push"], self._remaining(start, budget))
            push_state = self._push_chain_candidate(
                engine,
                benchmark,
                best,
                self._unique_ids(hotspots[:32] + cong_bin_order[:24] + degree_order[:12]),
                net_data,
                compute_proxy_cost,
                plc,
                sizes,
                half_w,
                half_h,
                cw,
                ch,
                start,
                min(budget, time.time() - start + push_budget),
                exact_affordable,
            )
            if push_state is not None and push_state[-1] < best[-1] - 1.0e-12:
                best = push_state
                best_hard = best[1].copy()
                outputs.append(self._candidate_output(engine, benchmark, best[0], best_hard, edges, "heuristic_push_chain", best[-1]))
            self._stage_log("push_chain", "end", start, budget, best[-1], moves=self.stats.moves_accepted)

        # Stage D: structured ILS wrapper.
        if self._remaining(start, budget) > 12.0 and stage_budgets["basin"] > 1.0:
            self._stage_log("basin", "start", start, budget, best[-1], moves=self.stats.moves_accepted)
            basin_budget = min(stage_budgets["basin"], self._remaining(start, budget))
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
                min(budget, time.time() - start + basin_budget),
                exact_affordable,
            )
            if ils_state is not None and ils_state[-1] < best[-1] - 1.0e-12:
                best = ils_state
                outputs.append(self._candidate_output(engine, benchmark, best[0], best[1], edges, "heuristic_basin_best", best[-1]))
            self._stage_log("basin", "end", start, budget, best[-1], moves=self.stats.moves_accepted)

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
        if compute_proxy_cost is None or plc is None or not self._can_exact_eval():
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
        stage_name: str = "cd",
        patience_macros: int = 64,
    ):
        full, hard, positions, net_hpwl, hpwl, density, congestion, score = state
        return self._run_tabu_cd(
            engine,
            benchmark,
            full.clone(),
            hard.copy(),
            list(macro_ids),
            net_data,
            positions.clone(),
            net_hpwl.clone(),
            hpwl,
            density,
            congestion,
            score,
            bool(use_exact),
            int(exact_per_macro),
            compute_proxy_cost,
            plc,
            sizes,
            half_w,
            half_h,
            float(cw),
            float(ch),
            start_time,
            budget_s,
            stage_name,
            patience_macros,
        )

    def _run_tabu_cd(
        self,
        engine: Any,
        benchmark: Any,
        full: torch.Tensor,
        hard: np.ndarray,
        macro_ids: Sequence[int],
        net_data: Any,
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
        stage_name: str,
        patience_macros: int,
    ):
        span = math.sqrt(max(cw * ch, 1.0e-12))
        n_hard = int(benchmark.num_hard_macros)
        tabu_ttl = max(4, int(os.environ.get("JIHO_HEURISTIC_TABU_TTL", "18")))
        tabu: Dict[Tuple[int, int, int], int] = {}
        no_improve = 0
        last_log = time.time()
        current_cheap = self._cheap_score(current_hpwl, current_density, current_congestion)
        ordered = self._unique_ids(macro_ids)
        for order, macro_id in enumerate(ordered):
            now = time.time()
            if now - start_time >= budget_s:
                break
            if no_improve >= max(1, int(patience_macros)):
                self._stage_log(stage_name, "patience_stop", start_time, budget_s, current_score, moves=self.stats.moves_accepted)
                break
            if now - last_log >= self._stage_log_every_s:
                self._stage_log(stage_name, "heartbeat", start_time, budget_s, current_score, moves=self.stats.moves_accepted)
                last_log = now
            i = int(macro_id)
            if i < 0 or i >= n_hard:
                continue
            best = None
            affected = net_data.macro_to_nets[i] if i < len(net_data.macro_to_nets) else []
            old_pos = positions[i].clone()
            old_key = self._tabu_key(i, hard[i])
            proposals = []
            for new_xy in self._proposal_positions(engine, full, hard, i, sizes, half_w, half_h, cw, ch, benchmark, span):
                if time.time() - start_time >= budget_s:
                    break
                self.stats.moves_tried += 1
                if np.allclose(new_xy, hard[i], atol=1.0e-12):
                    continue
                key = self._tabu_key(i, new_xy)
                if tabu.get(key, -1) > order:
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
                if delta_hpwl > 0.020 * max(current_hpwl, 1.0e-12):
                    continue
                proposal_full = full.clone()
                proposal_full[i] = torch.tensor(new_xy, dtype=proposal_full.dtype)
                new_hpwl = current_hpwl + delta_hpwl
                new_density = float(engine._estimate_density_overflow_np(proposal_full[: benchmark.num_macros].numpy(), benchmark))
                new_congestion = self._cheap_congestion(engine, proposal_full, benchmark)
                cheap_score = self._cheap_score(new_hpwl, new_density, new_congestion)
                cheap_delta = cheap_score - current_cheap
                if not use_exact and cheap_delta > 0.010 * max(abs(current_cheap), 1.0):
                    continue
                proposals.append(
                    (
                        cheap_delta,
                        new_xy.copy(),
                        rows,
                        new_hpwl_values,
                        old_contribution,
                        new_hpwl,
                        new_density,
                        new_congestion,
                        proposal_full,
                        cheap_score,
                    )
                )
            eval_limit = max(1, int(exact_per_macro)) if use_exact else max(1, min(10, len(proposals)))
            eval_pool = sorted(proposals, key=lambda row: float(row[0]))[:eval_limit]
            for cheap_delta, new_xy, rows, new_hpwl_values, _old_contribution, new_hpwl, new_density, new_congestion, proposal_full, cheap_score in eval_pool:
                if time.time() - start_time >= budget_s:
                    break
                if use_exact:
                    if not self._can_exact_eval():
                        continue
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
                    score = cheap_score
                    delta_score = cheap_delta
                if delta_score < -1.0e-12 and (best is None or score < best[0]):
                    best = (score, new_xy.copy(), rows, new_hpwl_values, new_hpwl, new_density, new_congestion, cheap_score)
            if best is None:
                positions[i] = old_pos
                no_improve += 1
                continue
            score, new_xy, rows, new_hpwl_values, current_hpwl, current_density, current_congestion, current_cheap = best
            hard[i] = new_xy
            full[i] = torch.tensor(new_xy, dtype=full.dtype)
            positions[i] = torch.tensor(new_xy, dtype=positions.dtype, device=positions.device)
            if int(rows.numel()) > 0:
                net_hpwl[rows] = new_hpwl_values
            tabu[old_key] = order + tabu_ttl
            current_score = score
            self.stats.moves_accepted += 1
            no_improve = 0
        return full, hard, positions, net_hpwl, current_hpwl, current_density, current_congestion, current_score

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
        variant: str = "heuristic_lns_random",
    ):
        best = None
        subset_base = max(1, int(round(0.16 * len(movable_idx))))
        for round_idx in range(max(1, int(rounds))):
            if self._remaining(start_time, budget_s) < 5.0:
                break
            frac = 0.10 + 0.03 * (round_idx % 6)
            subset_size = min(len(movable_idx), max(subset_base, int(round(frac * len(movable_idx)))))
            subset = self._lns_neighborhood_subset(movable_idx, priority, base_state[1], cw, ch, subset_size, rng, variant)
            trial = self._structured_perturb(engine, benchmark, base_state, subset, net_data, sizes, half_w, half_h, cw, ch, rng, variant, round_idx)
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
                stage_name="lns_repair",
                patience_macros=max(24, len(repair_ids) // 2),
            )
            if exact_final and self._can_exact_eval() and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
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
            plan = self._plan_push_chain(i, target, trial_hard, sizes, half_w, half_h, cw, ch, max_depth=3, beam=3)
            if not plan:
                continue
            self._apply_chain_plan(plan, trial_hard, trial_full)
            if self._any_overlap(trial_hard, sizes):
                continue
            trial = self._state_from_full(engine, benchmark, trial_full, trial_hard, net_data)
            if exact_final and self._can_exact_eval() and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
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
        attempts = 8 if self._remaining(start_time, budget_s) > 240.0 else 4
        priority = self._unique_ids(list(hotspots) + list(degree_order[:48]))
        boundary = self._boundary_order(movable_idx, base_state[1], cw, ch)
        cong_bin = self._congestion_bin_order(engine, benchmark, base_state[1], movable_idx)
        variants = ("basin_hotspot", "basin_pocket", "basin_window", "basin_boundary", "basin_v23", "basin_push")
        for attempt in range(attempts):
            if self._remaining(start_time, budget_s) < 8.0:
                break
            variant = variants[attempt % len(variants)]
            variant_priority = priority
            if "boundary" in variant:
                variant_priority = boundary
            elif "v23" in variant:
                variant_priority = self._unique_ids(boundary[:32] + cong_bin[:32] + priority[:24])
            elif "window" in variant:
                variant_priority = []
            if attempt % 3 == 0:
                subset = np.asarray(priority[: max(8, min(len(priority), len(movable_idx) // 6))], dtype=np.int64)
            else:
                subset = self._lns_neighborhood_subset(movable_idx, variant_priority, base_state[1], cw, ch, max(8, len(movable_idx) // 5), rng, variant)
            if "push" in variant:
                pushed = self._push_chain_candidate(
                    engine,
                    benchmark,
                    base_state,
                    self._unique_ids(list(subset[:16]) + priority[:16]),
                    net_data,
                    compute_proxy_cost,
                    plc,
                    sizes,
                    half_w,
                    half_h,
                    cw,
                    ch,
                    start_time,
                    budget_s,
                    False,
                )
                trial = pushed if pushed is not None else self._group_escape(engine, benchmark, base_state, subset, net_data, sizes, half_w, half_h, cw, ch, rng, attempt)
            else:
                trial = self._structured_perturb(engine, benchmark, base_state, subset, net_data, sizes, half_w, half_h, cw, ch, rng, variant, attempt)
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
                stage_name="basin_repair",
                patience_macros=max(24, len(repair_ids) // 2),
            )
            if exact_final and self._can_exact_eval() and self._remaining(start_time, budget_s) > max(4.0, self.stats.exact_eval_time_s + 1.0):
                trial = self._score_state_exact(trial, benchmark, compute_proxy_cost, plc)
            accept_margin = 1.0e-12 if exact_final else -0.002 * max(abs(base_state[-1]), 1.0)
            if trial[-1] < base_state[-1] - accept_margin and (best is None or trial[-1] < best[-1]):
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

    def _lns_neighborhood_subset(
        self,
        movable_idx: np.ndarray,
        priority: Sequence[int],
        hard: np.ndarray,
        cw: float,
        ch: float,
        subset_size: int,
        rng: np.random.Generator,
        variant: str,
    ) -> np.ndarray:
        movable = np.asarray(movable_idx, dtype=np.int64)
        if movable.size == 0:
            return movable
        target = min(max(1, int(subset_size)), int(movable.size))
        if "window" in variant:
            anchor = hard[int(rng.choice(movable))]
            dx = np.abs(hard[movable, 0] - float(anchor[0])) / max(float(cw), 1.0e-12)
            dy = np.abs(hard[movable, 1] - float(anchor[1])) / max(float(ch), 1.0e-12)
            return movable[np.argsort(dx + dy)[:target]].astype(np.int64)
        if "boundary" in variant:
            priority = self._boundary_order(movable, hard, cw, ch)
        chosen: List[int] = []
        movable_set = {int(i) for i in movable.tolist()}
        for raw_i in priority:
            i = int(raw_i)
            if i in movable_set and i not in chosen:
                chosen.append(i)
            if len(chosen) >= max(1, target // 2):
                break
        rest = np.asarray([int(i) for i in movable.tolist() if int(i) not in set(chosen)], dtype=np.int64)
        if rest.size and len(chosen) < target:
            pick = rng.choice(rest, size=min(target - len(chosen), int(rest.size)), replace=False)
            chosen.extend(int(i) for i in pick.tolist())
        return np.asarray(chosen[:target], dtype=np.int64)

    def _structured_perturb(
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
        variant: str,
        round_idx: int,
    ):
        full, hard, _positions, _net_hpwl, _hpwl, _density, _congestion, _score = base_state
        trial_full = full.clone()
        trial_hard = hard.copy()
        center = np.array([0.5 * cw, 0.5 * ch], dtype=np.float64)
        span = math.sqrt(max(float(cw) * float(ch), 1.0e-12))
        for raw_i in subset:
            i = int(raw_i)
            old = trial_hard[i].copy()
            if "boundary" in variant:
                direction = old - center
                norm = float(np.linalg.norm(direction))
                direction = direction / norm if norm > 1.0e-12 else rng.normal(0.0, 1.0, size=2)
                proposed = old - direction * (0.020 * span)
            elif "v23" in variant:
                pocket = self._low_pocket(engine, benchmark, trial_hard, old, cw, ch, radius=12)
                flow = pocket - old
                swirl = np.array([-(old[1] - center[1]) / max(ch, 1.0e-12), (old[0] - center[0]) / max(cw, 1.0e-12)])
                proposed = old + 0.55 * flow + swirl * (0.015 * span)
            elif "cong_bin" in variant or "hotspot" in variant or "pocket" in variant:
                pocket = self._low_pocket(engine, benchmark, trial_hard, old, cw, ch, radius=9 + (round_idx % 4))
                proposed = old + (pocket - old) * float(rng.uniform(0.45, 0.90))
            elif "window" in variant:
                proposed = old + rng.normal(0.0, [0.018 * cw, 0.018 * ch], size=2)
            else:
                proposed = old + rng.normal(0.0, [0.035 * cw, 0.035 * ch], size=2)
            proposed[0] = min(max(proposed[0], half_w[i]), cw - half_w[i])
            proposed[1] = min(max(proposed[1], half_h[i]), ch - half_h[i])
            if not self._single_overlap(i, proposed, trial_hard, sizes):
                trial_hard[i] = proposed
                trial_full[i] = torch.tensor(proposed, dtype=trial_full.dtype)
        return self._state_from_full(engine, benchmark, trial_full, trial_hard, net_data)

    def _plan_push_chain(
        self,
        i: int,
        target: np.ndarray,
        hard: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
        max_depth: int,
        beam: int,
    ) -> Optional[List[Tuple[int, np.ndarray]]]:
        i = int(i)
        direction = np.asarray(target, dtype=np.float64) - hard[i]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            return None
        direction = direction / norm
        frontier: List[List[Tuple[int, np.ndarray]]] = [[(i, self._clip_xy(i, target, half_w, half_h, cw, ch))]]
        for _depth in range(max(1, int(max_depth))):
            next_frontier: List[List[Tuple[int, np.ndarray]]] = []
            for plan in frontier:
                trial = hard.copy()
                for mid, xy in plan:
                    trial[int(mid)] = np.asarray(xy, dtype=np.float64)
                unresolved = self._first_plan_collision(plan, trial, sizes)
                if unresolved is None:
                    return plan
                _mover, collided = unresolved
                if any(int(mid) == int(collided) for mid, _xy in plan):
                    continue
                step = max(0.006 * math.sqrt(max(cw * ch, 1.0e-12)), 0.65 * max(float(sizes[collided, 0]), float(sizes[collided, 1])))
                options = (
                    hard[collided] + direction * step,
                    hard[collided] - direction * step,
                    self._low_chain_pocket(collided, trial, sizes, half_w, half_h, cw, ch),
                )
                for opt in options:
                    xy = self._clip_xy(collided, opt, half_w, half_h, cw, ch)
                    if not np.allclose(xy, hard[collided], atol=1.0e-12):
                        next_frontier.append(plan + [(int(collided), xy)])
            next_frontier.sort(key=lambda plan: sum(float(np.linalg.norm(xy - hard[mid])) for mid, xy in plan))
            frontier = next_frontier[: max(1, int(beam))]
            if not frontier:
                return None
        for plan in frontier:
            trial = hard.copy()
            for mid, xy in plan:
                trial[int(mid)] = np.asarray(xy, dtype=np.float64)
            if self._first_plan_collision(plan, trial, sizes) is None:
                return plan
        return None

    def _apply_chain_plan(self, plan: Sequence[Tuple[int, np.ndarray]], hard: np.ndarray, full: torch.Tensor) -> None:
        for mid, xy in plan:
            hard[int(mid)] = np.asarray(xy, dtype=np.float64)
            full[int(mid)] = torch.tensor(xy, dtype=full.dtype)

    def _first_plan_collision(
        self,
        plan: Sequence[Tuple[int, np.ndarray]],
        trial_hard: np.ndarray,
        sizes: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        moved = {int(mid) for mid, _xy in plan}
        for mid, xy in plan:
            collided = self._overlap_indices(int(mid), np.asarray(xy, dtype=np.float64), trial_hard, sizes)
            collided = [int(j) for j in collided if int(j) not in moved]
            if collided:
                return int(mid), int(collided[0])
        return None

    def _low_chain_pocket(
        self,
        i: int,
        hard: np.ndarray,
        sizes: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
    ) -> np.ndarray:
        old = hard[int(i)]
        span = math.sqrt(max(cw * ch, 1.0e-12))
        candidates = []
        for direction in self._directions():
            xy = self._clip_xy(i, old + direction * (0.018 * span), half_w, half_h, cw, ch)
            if not self._single_overlap(i, xy, hard, sizes):
                candidates.append(xy)
        if candidates:
            center = np.array([0.5 * cw, 0.5 * ch], dtype=np.float64)
            candidates.sort(key=lambda xy: float(np.linalg.norm(xy - center)))
            return candidates[0]
        return old

    def _clip_xy(
        self,
        i: int,
        xy: np.ndarray,
        half_w: np.ndarray,
        half_h: np.ndarray,
        cw: float,
        ch: float,
    ) -> np.ndarray:
        out = np.asarray(xy, dtype=np.float64).copy()
        out[0] = min(max(out[0], float(half_w[i])), float(cw) - float(half_w[i]))
        out[1] = min(max(out[1], float(half_h[i])), float(ch) - float(half_h[i]))
        return out

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

    def _congestion_bin_order(self, engine: Any, benchmark: Any, hard: np.ndarray, movable_idx: np.ndarray) -> List[int]:
        try:
            placement_np = self._candidate_macro_positions(hard, benchmark)
            h_grid, v_grid = engine._exact_style_congestion_arrays_np(placement_np, benchmark)
            grid = np.asarray(h_grid, dtype=np.float64) + np.asarray(v_grid, dtype=np.float64)
            rows, cols = grid.shape
            cw = float(benchmark.canvas_width)
            ch = float(benchmark.canvas_height)
            sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].detach().cpu().numpy().astype(np.float64)
            scores = np.zeros((benchmark.num_hard_macros,), dtype=np.float64)
            for i in np.asarray(movable_idx, dtype=np.int64).tolist():
                x0 = max(0, int(math.floor((hard[i, 0] - 0.5 * sizes[i, 0]) / max(cw, 1.0e-12) * cols)))
                x1 = min(cols - 1, int(math.floor((hard[i, 0] + 0.5 * sizes[i, 0]) / max(cw, 1.0e-12) * cols)))
                y0 = max(0, int(math.floor((hard[i, 1] - 0.5 * sizes[i, 1]) / max(ch, 1.0e-12) * rows)))
                y1 = min(rows - 1, int(math.floor((hard[i, 1] + 0.5 * sizes[i, 1]) / max(ch, 1.0e-12) * rows)))
                if x1 >= x0 and y1 >= y0:
                    window = grid[y0 : y1 + 1, x0 : x1 + 1]
                    scores[i] = float(np.max(window) + 0.25 * np.mean(window))
            return [int(i) for i in np.asarray(movable_idx, dtype=np.int64)[np.argsort(-scores[np.asarray(movable_idx, dtype=np.int64)])].tolist()]
        except Exception:
            return []

    def _stage_budgets(self, budget: float, exact_fast: bool, exact_affordable: bool, n_hard: int) -> Dict[str, float]:
        if exact_fast:
            exact_cd = min(0.34 * budget, 480.0)
            lns = min(0.34 * budget, 480.0)
            push = min(0.14 * budget, 240.0)
        elif exact_affordable:
            exact_cd = min(0.12 * budget, 120.0)
            lns = min(0.42 * budget, 540.0)
            push = min(0.18 * budget, 240.0)
        else:
            exact_cd = 0.0
            lns = min(0.48 * budget, 420.0 if n_hard < 700 else 300.0)
            push = min(0.18 * budget, 180.0)
        basin = max(0.0, budget - exact_cd - lns - push - 8.0)
        return {"exact_cd": exact_cd, "lns": lns, "push": push, "basin": basin}

    def _stage_log(
        self,
        stage: str,
        event: str,
        start_time: float,
        budget_s: float,
        best: float,
        moves: int = 0,
        stage_elapsed: Optional[float] = None,
    ) -> None:
        elapsed = time.time() - start_time
        fields = [
            f"stage={stage}",
            f"event={event}",
            f"evals={self.stats.exact_evals}",
            f"cheap={self.stats.cheap_evals}",
            f"moves={moves}",
            f"tried={self.stats.moves_tried}",
            f"best={float(best):.6f}",
            f"elapsed={elapsed:.1f}",
            f"remaining={max(0.0, budget_s - elapsed):.1f}",
        ]
        if stage_elapsed is not None:
            fields.append(f"stage_elapsed={float(stage_elapsed):.1f}")
        line = "|".join(fields)
        self.logs.append(line)
        print(f"[heuristic_search] {line}", flush=True)

    def _can_exact_eval(self) -> bool:
        return int(self.stats.exact_evals) < int(getattr(self, "_heuristic_exact_cap", 0))

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
        if use_exact and compute_proxy_cost is not None and plc is not None and self._can_exact_eval():
            try:
                costs = compute_proxy_cost(full.detach().cpu(), benchmark, plc)
                self.stats.exact_evals += 1
                return float(costs.get("proxy_cost", float("inf"))) + int(costs.get("overlap_count", 0)) * 1.0e6
            except Exception:
                return float("inf")
        self.stats.cheap_evals += 1
        return self._cheap_score(hpwl, density, congestion)

    def _tabu_key(self, macro_id: int, xy: np.ndarray) -> Tuple[int, int, int]:
        return (int(macro_id), int(round(float(xy[0]) * 2000.0)), int(round(float(xy[1]) * 2000.0)))

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
