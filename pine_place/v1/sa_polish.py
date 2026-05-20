"""HPWL-differential simulated annealing polish for v1 placements."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


@dataclass
class _SANetData:
    net_node_idx: torch.Tensor
    net_mask: torch.Tensor
    net_pin_offsets: torch.Tensor
    port_positions: torch.Tensor
    macro_to_nets: List[List[int]]


class SAPolisher:
    """Small SA pass scored by differential HPWL with periodic exact recalibration."""

    def __init__(self, device: DeviceLike = None, seed: int = 1729) -> None:
        self.device = self._resolve_device(device)
        self.seed = int(seed)
        self.logs: List[Dict[str, float]] = []
        self.moves_tried = 0
        self.moves_accepted = 0
        self.rejected_moves = 0
        self.overlap_rejects = 0
        self.proxy_evals = 0
        self.initial_proxy = float("inf")
        self.current_proxy = float("inf")
        self.best_proxy = float("inf")
        self.temperature = 0.0
        self.elapsed_s = 0.0
        self.running_hpwl_total = 0.0
        self._net_cache: Dict[Tuple[str, str], _SANetData] = {}

    def polish(self, placement: torch.Tensor, benchmark: Any, plc: Any, time_budget_s: int = 180) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost

        rng = random.Random(self.seed)
        start_time = time.time()
        budget = max(1.0, float(time_budget_s))
        self.logs = []
        self.moves_tried = 0
        self.moves_accepted = 0
        self.rejected_moves = 0
        self.overlap_rejects = 0
        self.proxy_evals = 0

        dtype = torch.float32
        device = self.device
        original_cpu = torch.as_tensor(placement, dtype=dtype).detach().cpu().clone()
        positions = original_cpu.to(device=device).clone()
        sizes = torch.as_tensor(benchmark.macro_sizes, dtype=dtype, device=device).clone()
        fixed = torch.as_tensor(benchmark.macro_fixed, dtype=torch.bool, device=device)
        n_macros = int(positions.shape[0])
        n_hard = int(getattr(benchmark, "num_hard_macros", n_macros))
        hard_movable_mask = (~fixed[:n_hard]).clone()
        hard_movable_idx = torch.nonzero(hard_movable_mask, as_tuple=False).flatten()
        if int(hard_movable_idx.numel()) == 0:
            return original_cpu
        net_data = self._net_data_for(benchmark, plc, device, dtype, n_macros)

        canvas = torch.tensor(
            [float(benchmark.canvas_width), float(benchmark.canvas_height)],
            dtype=dtype,
            device=device,
        )
        half = sizes[:n_hard] * 0.5
        low = half
        high = canvas.view(1, 2) - half
        min_sep_x = (sizes[:n_hard, 0:1] + sizes[:n_hard, 0:1].T) * 0.5
        min_sep_y = (sizes[:n_hard, 1:2] + sizes[:n_hard, 1:2].T) * 0.5

        initial_cost = compute_proxy_cost(original_cpu, benchmark, plc)
        self.proxy_evals += 1
        current_proxy = float(initial_cost["proxy_cost"])
        best_proxy = current_proxy
        self.initial_proxy = current_proxy
        self.current_proxy = current_proxy
        self.best_proxy = current_proxy
        best_positions = positions.clone()
        current_positions = positions.clone()
        net_hpwl = self._compute_all_net_hpwl(net_data, current_positions)
        running_hpwl_total = net_hpwl.sum()
        initial_hpwl_total = float(running_hpwl_total.detach().cpu().item())
        hpwl_scale = self.initial_proxy / max(initial_hpwl_total, 1.0e-12)
        self.running_hpwl_total = initial_hpwl_total

        t_start = max(0.10 * current_proxy, 1.0e-9)
        t_end = max(0.001 * current_proxy, 1.0e-12)
        temperature = t_start
        self.temperature = temperature
        canvas_sigma = canvas.view(1, 2)
        last_exact_time = start_time
        last_exact_accepts = 0

        iteration = 0
        while True:
            iteration += 1
            self.moves_tried += 1
            if iteration % 500 == 0 and time.time() - start_time >= budget:
                break

            if rng.random() < 0.70 or int(hard_movable_idx.numel()) < 2:
                local = int(hard_movable_idx[rng.randrange(int(hard_movable_idx.numel()))].detach().cpu().item())
                proposal = current_positions.clone()
                sigma = max(temperature, t_end) * canvas_sigma
                delta = torch.randn((1, 2), dtype=dtype, device=device) * sigma
                new_xy = current_positions[local : local + 1] + delta
                new_xy = torch.minimum(torch.maximum(new_xy, low[local : local + 1]), high[local : local + 1])
                if self._single_hard_overlap(local, new_xy.view(2), current_positions, min_sep_x, min_sep_y):
                    self.rejected_moves += 1
                    self.overlap_rejects += 1
                    continue
                proposal[local] = new_xy.view(2)
                affected = net_data.macro_to_nets[local]
            else:
                first = rng.randrange(int(hard_movable_idx.numel()))
                second = rng.randrange(int(hard_movable_idx.numel()) - 1)
                if second >= first:
                    second += 1
                local_i = int(hard_movable_idx[first].detach().cpu().item())
                local_j = int(hard_movable_idx[second].detach().cpu().item())
                proposal = current_positions.clone()
                pi = current_positions[local_i].clone()
                pj = current_positions[local_j].clone()
                if not self._inside_bounds(local_i, pj, low, high) or not self._inside_bounds(local_j, pi, low, high):
                    self.rejected_moves += 1
                    continue
                proposal[local_i] = pj
                proposal[local_j] = pi
                if self._single_hard_overlap(local_i, proposal[local_i], proposal, min_sep_x, min_sep_y) or self._single_hard_overlap(
                    local_j, proposal[local_j], proposal, min_sep_x, min_sep_y
                ):
                    self.rejected_moves += 1
                    self.overlap_rejects += 1
                    continue
                affected = sorted(set(net_data.macro_to_nets[local_i]) | set(net_data.macro_to_nets[local_j]))

            affected_rows, new_hpwl_values = self._compute_affected_net_hpwl(net_data, proposal, affected)
            if int(affected_rows.numel()) == 0:
                delta_proxy = 0.0
            else:
                old_contribution = net_hpwl.index_select(0, affected_rows).sum()
                new_contribution = new_hpwl_values.sum()
                delta_hpwl = float((new_contribution - old_contribution).detach().cpu().item())
                delta_proxy = delta_hpwl * hpwl_scale
            accept = delta_proxy < 0.0 or rng.random() < math.exp(-delta_proxy / max(temperature, 1.0e-12))
            if not accept:
                self.rejected_moves += 1
                continue

            current_positions = proposal
            if int(affected_rows.numel()) > 0:
                net_hpwl[affected_rows] = new_hpwl_values
                running_hpwl_total = running_hpwl_total + (new_hpwl_values.sum() - old_contribution)
            running_hpwl_estimate = float(running_hpwl_total.detach().cpu().item())
            current_proxy = self.initial_proxy + (running_hpwl_estimate - initial_hpwl_total) * hpwl_scale
            self.moves_accepted += 1
            temperature = max(t_end, temperature * 0.9999)
            self.current_proxy = current_proxy
            self.temperature = temperature

            need_exact = (self.moves_accepted - last_exact_accepts) >= 2000 or (time.time() - last_exact_time) >= 30.0
            if need_exact:
                true_cost = compute_proxy_cost(current_positions.detach().cpu(), benchmark, plc)
                self.proxy_evals += 1
                true_proxy = float(true_cost["proxy_cost"])
                last_exact_time = time.time()
                last_exact_accepts = self.moves_accepted
                self.current_proxy = true_proxy
                if true_proxy < best_proxy:
                    best_proxy = true_proxy
                    best_positions = current_positions.clone()
                    self.best_proxy = best_proxy
                row = {
                    "iter": float(iteration),
                    "accepted_moves": float(self.moves_accepted),
                    "rejected_moves": float(self.rejected_moves),
                    "current_proxy": float(true_proxy),
                    "true_proxy": float(true_proxy),
                    "best_proxy": float(best_proxy),
                    "running_hpwl_estimate": float(running_hpwl_estimate),
                    "temperature": float(temperature),
                    "elapsed_s": float(time.time() - start_time),
                }
                self.logs.append(row)
                print(
                    "[SA] "
                    f"iter={iteration} accepted={self.moves_accepted} rejected={self.rejected_moves} "
                    f"true={true_proxy:.6f} best={best_proxy:.6f} hpwl={running_hpwl_estimate:.6f} T={temperature:.6g}"
                )

        self.elapsed_s = time.time() - start_time
        final_cost = compute_proxy_cost(current_positions.detach().cpu(), benchmark, plc)
        self.proxy_evals += 1
        final_proxy = float(final_cost["proxy_cost"])
        self.current_proxy = final_proxy
        self.running_hpwl_total = float(running_hpwl_total.detach().cpu().item())
        if final_proxy < best_proxy:
            best_proxy = final_proxy
            best_positions = current_positions.clone()
            self.best_proxy = best_proxy
        if best_proxy < self.initial_proxy:
            return best_positions.detach().cpu()
        return original_cpu

    def _compute_all_net_hpwl(self, net_data: _SANetData, positions: torch.Tensor) -> torch.Tensor:
        if int(net_data.net_node_idx.shape[0]) == 0:
            return torch.zeros((0,), dtype=positions.dtype, device=positions.device)
        return self._hpwl_rows(net_data, positions, torch.arange(net_data.net_node_idx.shape[0], dtype=torch.long, device=positions.device))

    def _compute_affected_net_hpwl(
        self,
        net_data: _SANetData,
        positions: torch.Tensor,
        affected: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not affected:
            empty_rows = torch.zeros((0,), dtype=torch.long, device=positions.device)
            empty_vals = torch.zeros((0,), dtype=positions.dtype, device=positions.device)
            return empty_rows, empty_vals
        rows = torch.tensor(list(affected), dtype=torch.long, device=positions.device)
        return rows, self._hpwl_rows(net_data, positions, rows)

    def _hpwl_rows(self, net_data: _SANetData, positions: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        idx = net_data.net_node_idx.index_select(0, rows)
        mask = net_data.net_mask.index_select(0, rows)
        offsets = net_data.net_pin_offsets.index_select(0, rows)
        all_pos = positions if net_data.port_positions.numel() == 0 else torch.cat([positions, net_data.port_positions], dim=0)
        pts = all_pos.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], 2) + offsets
        inf = torch.tensor(float("inf"), dtype=positions.dtype, device=positions.device)
        neg_inf = torch.tensor(float("-inf"), dtype=positions.dtype, device=positions.device)
        x = pts[:, :, 0]
        y = pts[:, :, 1]
        xmax = torch.where(mask, x, neg_inf).max(dim=1).values
        xmin = torch.where(mask, x, inf).min(dim=1).values
        ymax = torch.where(mask, y, neg_inf).max(dim=1).values
        ymin = torch.where(mask, y, inf).min(dim=1).values
        return (xmax - xmin) + (ymax - ymin)

    def _single_hard_overlap(
        self,
        local_idx: int,
        new_pos: torch.Tensor,
        positions: torch.Tensor,
        min_sep_x: torch.Tensor,
        min_sep_y: torch.Tensor,
    ) -> bool:
        hard_pos = positions[: min_sep_x.shape[0]]
        dx = torch.abs(new_pos[0] - hard_pos[:, 0])
        dy = torch.abs(new_pos[1] - hard_pos[:, 1])
        overlap = (dx < min_sep_x[local_idx]) & (dy < min_sep_y[local_idx])
        overlap[local_idx] = False
        return bool(overlap.any().detach().cpu().item())

    def _inside_bounds(self, local_idx: int, pos: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> bool:
        ok = (pos >= torch.minimum(low[local_idx], high[local_idx])) & (pos <= torch.maximum(low[local_idx], high[local_idx]))
        return bool(ok.all().detach().cpu().item())

    def _net_data_for(self, benchmark: Any, plc: Any, device: torch.device, dtype: torch.dtype, n_macros: int) -> _SANetData:
        key = (str(getattr(benchmark, "name", "benchmark")), str(device))
        cached = self._net_cache.get(key)
        if cached is not None:
            return cached

        port_positions = torch.as_tensor(getattr(benchmark, "port_positions", torch.zeros(0, 2)), dtype=dtype, device=device).reshape(-1, 2)
        usable: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pin_nets = getattr(benchmark, "net_pin_nodes", None)
        if pin_nets is not None and len(pin_nets) > 0:
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
                usable.append((owners, torch.zeros((int(owners.numel()), 2), dtype=dtype)))
        elif plc is not None and hasattr(plc, "nets"):
            usable.extend(self._nets_from_plc(benchmark, plc, dtype))

        owner_count = n_macros + int(port_positions.shape[0])
        cleaned: List[Tuple[torch.Tensor, torch.Tensor]] = []
        macro_to_nets: List[List[int]] = [[] for _ in range(n_macros)]
        for owners, offsets in usable:
            valid = (owners >= 0) & (owners < owner_count)
            owners = owners[valid]
            offsets = offsets[valid]
            if int(torch.unique(owners).numel()) < 2:
                continue
            net_id = len(cleaned)
            cleaned.append((owners, offsets))
            for owner in torch.unique(owners).tolist():
                idx = int(owner)
                if 0 <= idx < n_macros:
                    macro_to_nets[idx].append(net_id)

        if not cleaned:
            net_data = _SANetData(
                net_node_idx=torch.zeros((0, 1), dtype=torch.long, device=device),
                net_mask=torch.zeros((0, 1), dtype=torch.bool, device=device),
                net_pin_offsets=torch.zeros((0, 1, 2), dtype=dtype, device=device),
                port_positions=port_positions,
                macro_to_nets=macro_to_nets,
            )
            self._net_cache[key] = net_data
            return net_data

        max_degree = max(int(owners.numel()) for owners, _offsets in cleaned)
        net_node_idx = torch.zeros((len(cleaned), max_degree), dtype=torch.long, device=device)
        net_mask = torch.zeros((len(cleaned), max_degree), dtype=torch.bool, device=device)
        net_pin_offsets = torch.zeros((len(cleaned), max_degree, 2), dtype=dtype, device=device)
        for row, (owners, offsets) in enumerate(cleaned):
            degree = int(owners.numel())
            net_node_idx[row, :degree] = owners.to(device=device)
            net_mask[row, :degree] = True
            net_pin_offsets[row, :degree] = offsets.to(device=device, dtype=dtype)

        for nets in macro_to_nets:
            nets.sort()
        net_data = _SANetData(
            net_node_idx=net_node_idx,
            net_mask=net_mask,
            net_pin_offsets=net_pin_offsets,
            port_positions=port_positions,
            macro_to_nets=macro_to_nets,
        )
        self._net_cache[key] = net_data
        return net_data

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
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
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
