"""Global analytical macro placement engine for JihoPlace v3."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from jiho_place.v2.objectives import smooth_hpwl_lse


DeviceLike = Optional[Union[str, torch.device]]


@dataclass
class _V3State:
    positions: torch.Tensor
    sizes: torch.Tensor
    movable_mask: torch.Tensor
    fixed_mask: torch.Tensor
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    hard_movable_indices: torch.Tensor
    canvas_width: float
    canvas_height: float
    port_positions: torch.Tensor
    nets: List[torch.Tensor]
    net_node_idx: torch.Tensor
    net_mask: torch.Tensor
    net_pin_offset_tensor: torch.Tensor
    net_weight_tensor: torch.Tensor
    net_weights: torch.Tensor
    density_x: torch.Tensor
    density_y: torch.Tensor
    congestion_x0: torch.Tensor
    congestion_x1: torch.Tensor
    congestion_y0: torch.Tensor
    congestion_y1: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    @property
    def num_macros(self) -> int:
        return int(self.positions.shape[0])

    @property
    def span(self) -> float:
        return max(float(self.canvas_width), float(self.canvas_height), 1.0e-6)

    def owner_positions(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        macro_pos = self.positions if positions is None else positions
        if self.port_positions.numel() == 0:
            return macro_pos
        return torch.cat([macro_pos, self.port_positions], dim=0)


class V3GlobalEngine:
    """Pure-PyTorch global analytical hard-macro placer."""

    def __init__(
        self,
        device: DeviceLike = None,
        iterations: int = 1200,
        init_iterations: int = 50,
        num_starts: int = 3,
        max_time_seconds: float = 300.0,
        log_every: int = 100,
    ) -> None:
        self.device = self._resolve_device(device)
        self.iterations = max(0, int(iterations))
        self.init_iterations = max(0, int(init_iterations))
        self.num_starts = max(1, int(num_starts))
        self.max_time_seconds = max(1.0, float(max_time_seconds))
        self.log_every = max(1, int(log_every))
        self.logs: List[Dict[str, float]] = []
        self.timing: Dict[str, float] = {}
        self.best_costs: Dict[str, float] = {}
        self._cache: Dict[str, _V3State] = {}

    def place(self, benchmark: Any, plc: Any) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        start_time = time.time()
        state = self._state_for(benchmark, plc)
        print(
            "[V3] index_map "
            f"movable_hard={int(state.hard_movable_indices.numel())} "
            f"hard={int(state.hard_mask.sum().detach().cpu().item())} "
            f"macros={state.num_macros}"
        )
        best_placement: Optional[torch.Tensor] = None
        best_proxy = float("inf")
        best_any_placement: Optional[torch.Tensor] = None
        best_any_proxy = float("inf")
        best_any_overlaps = 999999
        self.logs = []
        self.timing = {"init_s": 0.0, "stage1_s": 0.0, "stage2_s": 0.0, "stage3_s": 0.0, "stage4_s": 0.0, "legalize_s": 0.0}
        self.best_costs = {}

        for seed in range(self.num_starts):
            if time.time() - start_time > self.max_time_seconds * 0.92:
                break
            placement, timing = self._run_one_start(state, seed, start_time)
            for key, value in timing.items():
                self.timing[key] = self.timing.get(key, 0.0) + float(value)
            placement_cpu = placement.detach().cpu()
            try:
                costs = compute_proxy_cost(placement_cpu, benchmark, plc)
            except Exception:
                continue
            proxy = float(costs.get("proxy_cost", float("inf")))
            overlaps = int(costs.get("overlap_count", 999999))
            row = {
                "seed": float(seed),
                "final_proxy": proxy,
                "final_wirelength": float(costs.get("wirelength_cost", float("nan"))),
                "final_density": float(costs.get("density_cost", float("nan"))),
                "final_congestion": float(costs.get("congestion_cost", float("nan"))),
                "final_overlaps": float(overlaps),
            }
            self.logs.append(row)
            if math.isfinite(proxy) and (overlaps < best_any_overlaps or (overlaps == best_any_overlaps and proxy < best_any_proxy)):
                best_any_proxy = proxy
                best_any_overlaps = overlaps
                best_any_placement = placement_cpu
            if overlaps == 0 and proxy < best_proxy:
                best_proxy = proxy
                best_placement = placement_cpu
                self.best_costs = row

        if best_placement is None:
            best_placement = best_any_placement
        if best_placement is None:
            best_placement = self._initial_full_placement(state, seed=0).detach().cpu()
        return best_placement

    def _run_one_start(self, state: _V3State, seed: int, global_start: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        timing = {"init_s": 0.0, "stage1_s": 0.0, "stage2_s": 0.0, "stage3_s": 0.0, "stage4_s": 0.0, "legalize_s": 0.0}
        pos = self._initial_full_placement(state, seed)
        chip = math.sqrt(max(float(state.canvas_width) * float(state.canvas_height), 1.0e-12))
        n_hard = max(int(state.hard_mask.sum().detach().cpu().item()), 1)

        t0 = time.time()
        init_param = torch.nn.Parameter(pos.clone())
        init_opt = torch.optim.Adam([init_param], lr=2.0 * chip / float(n_hard))
        gamma = self._hpwl_gamma(state)
        for _ in range(self.init_iterations):
            init_opt.zero_grad(set_to_none=True)
            hpwl = smooth_hpwl_lse(state, init_param, gamma=gamma)
            hpwl.backward()
            init_opt.step()
            with torch.no_grad():
                init_param.data = self._clip_and_restore(state, init_param.data)
        with torch.no_grad():
            pos = self._spread_overlaps(state, init_param.detach(), max_iters=20)
            pos = self._clip_and_restore(state, pos)
        timing["init_s"] += time.time() - t0

        param = torch.nn.Parameter(pos.clone())
        base_lr = 5.0 * chip / float(n_hard)
        optimizer = torch.optim.Adam([param], lr=base_lr)
        stage_times = {1: "stage1_s", 2: "stage2_s", 3: "stage3_s", 4: "stage4_s"}
        stage_start = time.time()
        last_stage = 1

        for step in range(self.iterations):
            if time.time() - global_start > self.max_time_seconds * 0.97:
                break
            stage, density_w, congestion_w, lr_scale = self._stage_params(step)
            if stage != last_stage:
                timing[stage_times[last_stage]] += time.time() - stage_start
                stage_start = time.time()
                last_stage = stage
                if stage == 2:
                    optimizer = torch.optim.Adam([param], lr=base_lr * lr_scale)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

            optimizer.zero_grad(set_to_none=True)
            cur = torch.where(state.movable_mask[:, None], param, state.positions)
            hpwl = smooth_hpwl_lse(state, cur, gamma=gamma)
            density = self._density_penalty(state, cur)
            if congestion_w > 0.0:
                congestion = self._congestion_penalty(state, cur, gamma)
            else:
                congestion = cur.sum() * 0.0
            loss = hpwl + density_w * density + congestion_w * congestion
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                param.data = self._clip_and_restore(state, param.data)

            if step == 0 or step == self.iterations - 1 or step % self.log_every == 0:
                self.logs.append(
                    {
                        "seed": float(seed),
                        "step": float(step),
                        "stage": float(stage),
                        "loss": float(loss.detach().cpu().item()),
                        "hpwl": float(hpwl.detach().cpu().item()),
                        "density": float(density.detach().cpu().item()),
                        "congestion": float(congestion.detach().cpu().item()),
                        "density_w": float(density_w),
                        "congestion_w": float(congestion_w),
                        "lr": float(base_lr * lr_scale),
                    }
                )
        timing[stage_times[last_stage]] += time.time() - stage_start

        t0 = time.time()
        with torch.no_grad():
            final = self._fast_legalize(state, param.detach())
        timing["legalize_s"] = time.time() - t0
        return final, timing

    def _initial_full_placement(self, state: _V3State, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=state.device)
        generator.manual_seed(1729 + int(seed))
        pos = state.positions.clone()
        movable_hard_idx = state.hard_movable_indices
        center = torch.tensor([state.canvas_width * 0.5, state.canvas_height * 0.5], dtype=state.dtype, device=state.device)
        jitter_scale = torch.tensor([state.canvas_width, state.canvas_height], dtype=state.dtype, device=state.device) * 0.01
        jitter = torch.randn((int(movable_hard_idx.numel()), 2), generator=generator, dtype=state.dtype, device=state.device)
        pos[movable_hard_idx] = center.view(1, 2) + jitter * jitter_scale.view(1, 2)
        return self._clip_and_restore(state, pos)

    def _density_penalty(self, state: _V3State, positions: torch.Tensor) -> torch.Tensor:
        hard_pos = positions[state.hard_mask]
        hard_sizes = state.sizes[state.hard_mask]
        if int(hard_pos.shape[0]) == 0:
            return positions.sum() * 0.0
        rows = int(state.density_y.numel())
        cols = int(state.density_x.numel())
        cell_w = float(state.canvas_width) / max(cols, 1)
        cell_h = float(state.canvas_height) / max(rows, 1)
        ux = torch.abs(hard_pos[:, 0:1] - state.density_x.view(1, -1)) / max(cell_w * 3.0, 1.0e-6)
        uy = torch.abs(hard_pos[:, 1:2] - state.density_y.view(1, -1)) / max(cell_h * 3.0, 1.0e-6)
        wx = torch.where(ux <= 1.0, 1.0 - ux.pow(2), torch.zeros_like(ux))
        wy = torch.where(uy <= 1.0, 1.0 - uy.pow(2), torch.zeros_like(uy))
        wx = wx / torch.clamp(wx.sum(dim=1, keepdim=True), min=1.0e-12)
        wy = wy / torch.clamp(wy.sum(dim=1, keepdim=True), min=1.0e-12)
        areas = hard_sizes[:, 0] * hard_sizes[:, 1]
        # Each cell receives bell_x * bell_y * macro_area / cell_area.
        density = wy.transpose(0, 1).matmul((areas / max(cell_w * cell_h, 1.0e-12))[:, None] * wx)
        hard_area = float(areas.detach().sum().cpu().item())
        target = hard_area / max(float(state.canvas_width) * float(state.canvas_height) * 0.85, 1.0e-12)
        return torch.relu(density - target).pow(2).mean()

    def _congestion_penalty(self, state: _V3State, positions: torch.Tensor, gamma: float) -> torch.Tensor:
        net_count = int(state.net_node_idx.shape[0])
        if net_count == 0:
            return positions.sum() * 0.0
        rows = int(state.congestion_y0.numel())
        cols = int(state.congestion_x0.numel())
        grid_cells = rows * cols
        if net_count * grid_cells > 20_000_000:
            demand = torch.zeros((rows, cols), dtype=state.dtype, device=state.device)
            for start in range(0, net_count, 4096):
                end = min(net_count, start + 4096)
                demand = demand + self._congestion_demand_chunk(state, positions, gamma, start, end)
        else:
            demand = self._congestion_demand_chunk(state, positions, gamma, 0, net_count)
        return torch.relu(demand - 1.5).pow(2).mean()

    def _congestion_demand_chunk(
        self,
        state: _V3State,
        positions: torch.Tensor,
        gamma: float,
        start: int,
        end: int,
    ) -> torch.Tensor:
        idx = state.net_node_idx[start:end]
        mask = state.net_mask[start:end]
        offsets = state.net_pin_offset_tensor[start:end]
        all_pos = state.owner_positions(positions)
        pts = all_pos.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], 2) + offsets
        neg_inf = torch.tensor(float("-inf"), dtype=state.dtype, device=state.device)
        x = pts[:, :, 0]
        y = pts[:, :, 1]
        xmax = torch.logsumexp(torch.where(mask, gamma * x, neg_inf), dim=1) / gamma
        xmin = -torch.logsumexp(torch.where(mask, -gamma * x, neg_inf), dim=1) / gamma
        ymax = torch.logsumexp(torch.where(mask, gamma * y, neg_inf), dim=1) / gamma
        ymin = -torch.logsumexp(torch.where(mask, -gamma * y, neg_inf), dim=1) / gamma
        pin_count = torch.clamp(mask.sum(dim=1).to(dtype=state.dtype), min=1.0)
        net_demand = 1.0 / torch.sqrt(pin_count)

        sx = max(float(state.canvas_width) / max(int(state.congestion_x0.numel()), 1) * 0.35, 1.0e-6)
        sy = max(float(state.canvas_height) / max(int(state.congestion_y0.numel()), 1) * 0.35, 1.0e-6)
        x0 = state.congestion_x0.view(1, 1, -1)
        x1 = state.congestion_x1.view(1, 1, -1)
        y0 = state.congestion_y0.view(1, -1, 1)
        y1 = state.congestion_y1.view(1, -1, 1)
        x_cover = torch.sigmoid((x1 - xmin[:, None, None]) / sx) * torch.sigmoid((xmax[:, None, None] - x0) / sx)
        y_cover = torch.sigmoid((y1 - ymin[:, None, None]) / sy) * torch.sigmoid((ymax[:, None, None] - y0) / sy)
        cover = y_cover * x_cover
        cover = cover / torch.clamp(cover.sum(dim=(1, 2), keepdim=True), min=1.0e-12)
        return (cover * net_demand.view(-1, 1, 1)).sum(dim=0)

    def _spread_overlaps(self, state: _V3State, positions: torch.Tensor, max_iters: int) -> torch.Tensor:
        pos = positions.clone()
        for _ in range(max_iters):
            delta, max_overlap = self._overlap_repulsion(state, pos, strength=1.0, margin=0.10)
            if max_overlap < 1.0e-4:
                break
            pos = self._clip_and_restore(state, pos + delta)
        return pos

    def _fast_legalize(self, state: _V3State, positions: torch.Tensor) -> torch.Tensor:
        pos = self._clip_and_restore(state, positions.clone())
        for _ in range(100):
            delta, max_overlap = self._overlap_repulsion(state, pos, strength=1.05, margin=0.0)
            if max_overlap < 1.0e-4:
                break
            pos = self._clip_and_restore(state, pos + delta)
        pos = self._greedy_overlap_sweep(state, pos, max_iters=50)
        return pos

    def _greedy_overlap_sweep(self, state: _V3State, positions: torch.Tensor, max_iters: int) -> torch.Tensor:
        pos = self._clip_and_restore(state, positions.clone())
        hard_idx = torch.nonzero(state.hard_mask, as_tuple=False).flatten()
        n = int(hard_idx.numel())
        if n <= 1:
            return pos
        hard_sizes = state.sizes.index_select(0, hard_idx)
        canvas = torch.tensor([state.canvas_width, state.canvas_height], dtype=state.dtype, device=state.device)

        for _ in range(max_iters):
            _delta, max_overlap = self._overlap_repulsion(state, pos, strength=0.0, margin=0.0)
            if max_overlap < 1.0e-4:
                break

            moved_any = False
            order = torch.randperm(n, device=state.device)
            hard_pos = pos.index_select(0, hard_idx)
            for local_raw in order.tolist():
                local = int(local_raw)
                idx = hard_idx[local]
                if not bool(state.movable_mask[idx].detach().cpu().item()):
                    continue

                hard_pos = pos.index_select(0, hard_idx)
                diff = pos[idx].view(1, 2) - hard_pos
                sep = (state.sizes[idx].view(1, 2) + hard_sizes) * 0.5
                overlap = sep - torch.abs(diff)
                pair = (overlap[:, 0] > 0.0) & (overlap[:, 1] > 0.0)
                pair[local] = False
                if not bool(pair.any()):
                    continue

                candidate_idx = torch.nonzero(pair, as_tuple=False).flatten()
                depth = torch.minimum(overlap.index_select(0, candidate_idx)[:, 0], overlap.index_select(0, candidate_idx)[:, 1])
                other_local = int(candidate_idx[int(torch.argmax(depth).detach().cpu().item())].detach().cpu().item())
                pair_overlap = overlap[other_local]
                pair_diff = diff[other_local]
                axis = 0 if bool((pair_overlap[0] <= pair_overlap[1]).detach().cpu().item()) else 1
                sign = 1.0 if float(pair_diff[axis].detach().cpu().item()) >= 0.0 else -1.0
                step = (pair_overlap[axis] + torch.tensor(1.0e-4, dtype=state.dtype, device=state.device)) * sign
                pos[idx, axis] = pos[idx, axis] + step
                half = state.sizes[idx] * 0.5
                low = half
                high = canvas - half
                pos[idx] = torch.minimum(torch.maximum(pos[idx], torch.minimum(low, high)), torch.maximum(low, high))
                moved_any = True

            pos = self._clip_and_restore(state, pos)
            if not moved_any:
                break
        return pos

    def _overlap_repulsion(
        self,
        state: _V3State,
        positions: torch.Tensor,
        strength: float,
        margin: float,
    ) -> Tuple[torch.Tensor, float]:
        hard_idx = torch.nonzero(state.hard_mask, as_tuple=False).flatten()
        order = torch.argsort(positions.index_select(0, hard_idx)[:, 0])
        hard_idx = hard_idx.index_select(0, order)
        hard_pos = positions.index_select(0, hard_idx)
        hard_sizes = state.sizes.index_select(0, hard_idx)
        n = int(hard_pos.shape[0])
        if n <= 1:
            return torch.zeros_like(positions), 0.0

        # Broadcasted [N, N, 2] pair deltas; no torch.combinations or pair list.
        diff = hard_pos.unsqueeze(1) - hard_pos.unsqueeze(0)
        sep = (hard_sizes.unsqueeze(1) + hard_sizes.unsqueeze(0)) * 0.5
        overlap = sep - torch.abs(diff)
        pair = (overlap[:, :, 0] > 0.0) & (overlap[:, :, 1] > 0.0)
        pair = torch.triu(pair, diagonal=1)
        if not bool(pair.any()):
            return torch.zeros_like(positions), 0.0

        axis_x = overlap[:, :, 0] <= overlap[:, :, 1]
        sign = torch.where(diff >= 0.0, 1.0, -1.0)
        zero = torch.zeros_like(diff)
        move = zero.clone()
        move[:, :, 0] = torch.where(axis_x, sign[:, :, 0] * overlap[:, :, 0] * (1.0 + margin), zero[:, :, 0])
        move[:, :, 1] = torch.where(~axis_x, sign[:, :, 1] * overlap[:, :, 1] * (1.0 + margin), zero[:, :, 1])
        move = torch.where(pair[:, :, None], move, zero)

        hard_movable = state.movable_mask.index_select(0, hard_idx)
        i_can = hard_movable[:, None]
        j_can = hard_movable[None, :]
        both = i_can & j_can
        only_i = i_can & ~j_can
        only_j = ~i_can & j_can
        i_move = torch.where(both[:, :, None], move * 0.5, torch.where(only_i[:, :, None], move, zero))
        j_move = torch.where(both[:, :, None], -move * 0.5, torch.where(only_j[:, :, None], -move, zero))
        accum = i_move.sum(dim=1) + j_move.sum(dim=0)
        counts = ((i_move.abs().sum(dim=2) > 0.0).sum(dim=1) + (j_move.abs().sum(dim=2) > 0.0).sum(dim=0)).clamp(min=1)
        accum = accum / counts.to(dtype=state.dtype).view(-1, 1)
        full_delta = torch.zeros_like(positions)
        full_delta.index_add_(0, hard_idx, accum * float(strength))
        max_overlap = float(torch.where(pair, torch.minimum(overlap[:, :, 0], overlap[:, :, 1]), torch.zeros_like(overlap[:, :, 0])).max().detach().cpu().item())
        return full_delta, max_overlap

    def _clip_and_restore(self, state: _V3State, positions: torch.Tensor) -> torch.Tensor:
        half = state.sizes * 0.5
        low = half
        high = torch.tensor([state.canvas_width, state.canvas_height], dtype=state.dtype, device=state.device).view(1, 2) - half
        clipped = torch.minimum(torch.maximum(positions, torch.minimum(low, high)), torch.maximum(low, high))
        out = state.positions.clone()
        out[state.hard_movable_indices] = clipped[state.hard_movable_indices]
        out[state.fixed_mask] = state.positions[state.fixed_mask]
        return out

    def _stage_params(self, step: int) -> Tuple[int, float, float, float]:
        if step < 300:
            return 1, 1.0e-2, 0.0, 1.0
        if step < 700:
            frac = (step - 300) / 400.0
            return 2, 1.0e-2 + frac * (0.5 - 1.0e-2), 0.0, 0.5
        if step < 1000:
            frac = (step - 700) / 300.0
            return 3, 0.5, frac * 0.3, 0.2
        return 4, 0.5, 0.3, 0.05

    def _hpwl_gamma(self, state: _V3State) -> float:
        num_nets = max(int(state.net_node_idx.shape[0]), 1)
        chip = math.sqrt(max(float(state.canvas_width) * float(state.canvas_height), 1.0e-12))
        return float(min(20.0, max(5.0, 1.0 / max(0.1 * chip / math.sqrt(float(num_nets)), 1.0e-6))))

    def _state_for(self, benchmark: Any, plc: Any) -> _V3State:
        name = str(getattr(benchmark, "name", "benchmark"))
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        device = self.device
        dtype = torch.float32
        positions = torch.as_tensor(benchmark.macro_positions, dtype=dtype, device=device).clone()
        sizes = torch.as_tensor(benchmark.macro_sizes, dtype=dtype, device=device).clone()
        fixed = torch.as_tensor(benchmark.macro_fixed, dtype=torch.bool, device=device).clone()
        n_macros = int(getattr(benchmark, "num_macros", positions.shape[0]))
        n_hard = int(getattr(benchmark, "num_hard_macros", n_macros))
        arange = torch.arange(n_macros, device=device)
        hard = arange < n_hard
        soft = ~hard
        movable = (~fixed) & hard
        hard_movable_indices = torch.nonzero(movable & hard, as_tuple=False).flatten()
        port_positions = torch.as_tensor(getattr(benchmark, "port_positions", torch.zeros(0, 2)), dtype=dtype, device=device).reshape(-1, 2)
        net_idx, net_mask, offsets, weights, nets = self._extract_nets(benchmark, plc, device, dtype, n_macros, port_positions.shape[0])

        density_side = 64 if n_macros > 300 else 32
        density_x = torch.linspace(
            float(benchmark.canvas_width) / (2.0 * density_side),
            float(benchmark.canvas_width) * (1.0 - 1.0 / (2.0 * density_side)),
            density_side,
            dtype=dtype,
            device=device,
        )
        density_y = torch.linspace(
            float(benchmark.canvas_height) / (2.0 * density_side),
            float(benchmark.canvas_height) * (1.0 - 1.0 / (2.0 * density_side)),
            density_side,
            dtype=dtype,
            device=device,
        )
        rows = max(1, int(getattr(benchmark, "grid_rows", 32)))
        cols = max(1, int(getattr(benchmark, "grid_cols", rows)))
        x_edges = torch.linspace(0.0, float(benchmark.canvas_width), cols + 1, dtype=dtype, device=device)
        y_edges = torch.linspace(0.0, float(benchmark.canvas_height), rows + 1, dtype=dtype, device=device)
        state = _V3State(
            positions=positions,
            sizes=sizes,
            movable_mask=movable,
            fixed_mask=fixed,
            hard_mask=hard,
            soft_mask=soft,
            hard_movable_indices=hard_movable_indices,
            canvas_width=float(benchmark.canvas_width),
            canvas_height=float(benchmark.canvas_height),
            port_positions=port_positions,
            nets=nets,
            net_node_idx=net_idx,
            net_mask=net_mask,
            net_pin_offset_tensor=offsets,
            net_weight_tensor=weights,
            net_weights=weights,
            density_x=density_x,
            density_y=density_y,
            congestion_x0=x_edges[:-1],
            congestion_x1=x_edges[1:],
            congestion_y0=y_edges[:-1],
            congestion_y1=y_edges[1:],
            metadata={"benchmark_name": name},
        )
        self._cache[name] = state
        return state

    def _extract_nets(
        self,
        benchmark: Any,
        plc: Any,
        device: torch.device,
        dtype: torch.dtype,
        n_macros: int,
        num_ports: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        usable: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pin_nets = getattr(benchmark, "net_pin_nodes", None)
        if pin_nets:
            for pins in pin_nets:
                pins_cpu = torch.as_tensor(pins, dtype=torch.long).cpu()
                if pins_cpu.numel() == 0:
                    continue
                owners = pins_cpu[:, 0].flatten() if pins_cpu.ndim > 1 else pins_cpu.flatten()
                pin_ids = pins_cpu[:, 1].flatten() if pins_cpu.ndim > 1 else torch.zeros_like(owners)
                offsets = self._pin_offsets(benchmark, owners, pin_ids, dtype)
                usable.append((owners, offsets))
        elif getattr(benchmark, "net_nodes", None):
            for owners_raw in benchmark.net_nodes:
                owners = torch.as_tensor(owners_raw, dtype=torch.long).flatten().cpu()
                offsets = torch.zeros((int(owners.numel()), 2), dtype=dtype)
                usable.append((owners, offsets))
        elif plc is not None and hasattr(plc, "nets"):
            usable.extend(self._nets_from_plc(benchmark, plc, dtype))

        owner_count = n_macros + num_ports
        cleaned: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for owners, offsets in usable:
            valid = (owners >= 0) & (owners < owner_count)
            owners = owners[valid]
            offsets = offsets[valid]
            if int(torch.unique(owners).numel()) >= 2:
                cleaned.append((owners, offsets))

        if not cleaned:
            return (
                torch.zeros((0, 1), dtype=torch.long, device=device),
                torch.zeros((0, 1), dtype=torch.bool, device=device),
                torch.zeros((0, 1, 2), dtype=dtype, device=device),
                torch.zeros((0,), dtype=dtype, device=device),
                [],
            )
        max_degree = max(int(owners.numel()) for owners, _offsets in cleaned)
        net_idx = torch.zeros((len(cleaned), max_degree), dtype=torch.long, device=device)
        net_mask = torch.zeros((len(cleaned), max_degree), dtype=torch.bool, device=device)
        offsets = torch.zeros((len(cleaned), max_degree, 2), dtype=dtype, device=device)
        nets: List[torch.Tensor] = []
        for row, (owners, off) in enumerate(cleaned):
            degree = int(owners.numel())
            net_idx[row, :degree] = owners.to(device=device)
            net_mask[row, :degree] = True
            offsets[row, :degree] = off.to(device=device, dtype=dtype)
            nets.append(owners.to(device=device))
        weights = torch.ones((len(cleaned),), dtype=dtype, device=device)
        return net_idx, net_mask, offsets, weights, nets

    def _pin_offsets(self, benchmark: Any, owners: torch.Tensor, pin_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        macro_offsets: Sequence[Any] = getattr(benchmark, "macro_pin_offsets", [])
        n_hard = int(getattr(benchmark, "num_hard_macros", getattr(benchmark, "num_macros", 0)))
        out: List[Tuple[float, float]] = []
        for owner_raw, pin_raw in zip(owners.tolist(), pin_ids.tolist()):
            owner = int(owner_raw)
            pin_id = int(pin_raw)
            if 0 <= owner < n_hard and owner < len(macro_offsets):
                offsets = torch.as_tensor(macro_offsets[owner], dtype=dtype)
                if offsets.ndim == 2 and 0 <= pin_id < int(offsets.shape[0]):
                    x, y = offsets[pin_id].tolist()
                    out.append((float(x), float(y)))
                    continue
            out.append((0.0, 0.0))
        return torch.tensor(out, dtype=dtype)

    def _nets_from_plc(self, benchmark: Any, plc: Any, dtype: torch.dtype) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        name_to_owner = {str(name): idx for idx, name in enumerate(getattr(benchmark, "macro_names", []))}
        for port_offset, plc_idx in enumerate(getattr(plc, "port_indices", [])):
            try:
                name_to_owner[str(plc.modules_w_pins[plc_idx].get_name())] = int(benchmark.num_macros) + port_offset
            except Exception:
                continue
        out = []
        for driver, sinks in getattr(plc, "nets", {}).items():
            owners = []
            for pin_name in [driver] + list(sinks):
                text = str(pin_name)
                owner = name_to_owner.get(text)
                if owner is None:
                    owner = name_to_owner.get(text.split("/")[0])
                if owner is not None:
                    owners.append(int(owner))
            if len(set(owners)) >= 2:
                owner_tensor = torch.tensor(owners, dtype=torch.long)
                out.append((owner_tensor, torch.zeros((len(owners), 2), dtype=dtype)))
        return out

    def _resolve_device(self, device: DeviceLike) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
