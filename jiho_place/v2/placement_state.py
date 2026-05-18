"""Tensor state for JihoPlace v2 analytical placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


def _resolve_device(device: DeviceLike = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_bool_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool, device=device)


@dataclass
class PlacementState:
    """Continuous macro placement state consumed by v2 objectives.

    Macro coordinates are centers in microns. Owner indices in nets follow the
    challenge convention: macros first, then ports.
    """

    positions: torch.Tensor
    sizes: torch.Tensor
    movable_mask: torch.Tensor
    fixed_mask: torch.Tensor
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    canvas_width: float
    canvas_height: float
    nets: List[torch.Tensor]
    net_pin_offsets: List[torch.Tensor]
    net_weights: torch.Tensor
    port_positions: torch.Tensor
    macro_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    def num_hard_macros(self) -> int:
        return int(self.hard_mask.sum().item())

    @property
    def num_soft_macros(self) -> int:
        return int(self.soft_mask.sum().item())

    @property
    def span(self) -> float:
        return max(float(self.canvas_width), float(self.canvas_height), 1.0e-6)

    @property
    def canvas_size(self) -> torch.Tensor:
        return torch.tensor(
            [self.canvas_width, self.canvas_height],
            dtype=self.dtype,
            device=self.device,
        )

    def owner_positions(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        macro_pos = self.positions if positions is None else positions
        if self.port_positions.numel() == 0:
            return macro_pos
        return torch.cat([macro_pos, self.port_positions], dim=0)

    def with_positions(self, positions: torch.Tensor) -> "PlacementState":
        return PlacementState(
            positions=positions,
            sizes=self.sizes,
            movable_mask=self.movable_mask,
            fixed_mask=self.fixed_mask,
            hard_mask=self.hard_mask,
            soft_mask=self.soft_mask,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            nets=self.nets,
            net_pin_offsets=self.net_pin_offsets,
            net_weights=self.net_weights,
            port_positions=self.port_positions,
            macro_names=self.macro_names,
            metadata=dict(self.metadata),
        )

    def limited_nets(self, max_nets: Optional[int]) -> "PlacementState":
        if max_nets is None or int(max_nets) <= 0 or len(self.nets) <= int(max_nets):
            return self
        limit = int(max_nets)
        metadata = dict(self.metadata)
        metadata["limited_nets"] = limit
        metadata["original_v2_nets"] = len(self.nets)
        return PlacementState(
            positions=self.positions,
            sizes=self.sizes,
            movable_mask=self.movable_mask,
            fixed_mask=self.fixed_mask,
            hard_mask=self.hard_mask,
            soft_mask=self.soft_mask,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            nets=self.nets[:limit],
            net_pin_offsets=self.net_pin_offsets[:limit],
            net_weights=self.net_weights[:limit],
            port_positions=self.port_positions,
            macro_names=self.macro_names,
            metadata=metadata,
        )

    @classmethod
    def from_benchmark(cls, benchmark: Any, device: DeviceLike = None) -> "PlacementState":
        resolved = _resolve_device(device)
        positions = _as_float_tensor(getattr(benchmark, "macro_positions"), resolved).clone()
        sizes = _as_float_tensor(getattr(benchmark, "macro_sizes"), resolved).clone()
        num_macros = int(getattr(benchmark, "num_macros", positions.shape[0]))
        num_hard = int(getattr(benchmark, "num_hard_macros", num_macros))
        num_soft = int(getattr(benchmark, "num_soft_macros", max(0, num_macros - num_hard)))

        fixed_value = getattr(
            benchmark,
            "macro_fixed",
            torch.zeros(num_macros, dtype=torch.bool),
        )
        fixed_mask = _as_bool_tensor(fixed_value, resolved).clone()
        movable_mask = ~fixed_mask

        arange = torch.arange(num_macros, device=resolved)
        hard_mask = arange < num_hard
        soft_mask = (arange >= num_hard) & (arange < num_hard + num_soft)

        port_positions = _as_float_tensor(
            getattr(benchmark, "port_positions", torch.zeros(0, 2)),
            resolved,
        ).reshape(-1, 2)

        raw_weights = getattr(benchmark, "net_weights", None)
        weights_cpu = (
            torch.as_tensor(raw_weights, dtype=torch.float32).flatten().cpu()
            if raw_weights is not None
            else torch.ones(int(getattr(benchmark, "num_nets", 0)), dtype=torch.float32)
        )

        nets: List[torch.Tensor] = []
        net_pin_offsets: List[torch.Tensor] = []
        selected_weights: List[float] = []

        pin_nets = getattr(benchmark, "net_pin_nodes", None)
        if pin_nets:
            for net_id, pins in enumerate(pin_nets):
                pins_cpu = torch.as_tensor(pins, dtype=torch.long).cpu()
                if pins_cpu.numel() == 0:
                    continue
                if pins_cpu.ndim == 1:
                    owners_cpu = pins_cpu.flatten()
                    pin_ids_cpu = torch.zeros_like(owners_cpu)
                else:
                    owners_cpu = pins_cpu[:, 0].flatten()
                    pin_ids_cpu = pins_cpu[:, 1].flatten()
                nets.append(owners_cpu.to(device=resolved, dtype=torch.long))
                net_pin_offsets.append(
                    _net_offsets_from_pin_nodes(benchmark, owners_cpu, pin_ids_cpu, resolved)
                )
                selected_weights.append(_weight_at(weights_cpu, net_id))
        else:
            for net_id, owners in enumerate(getattr(benchmark, "net_nodes", [])):
                owners_cpu = torch.as_tensor(owners, dtype=torch.long).flatten().cpu()
                if owners_cpu.numel() == 0:
                    continue
                nets.append(owners_cpu.to(device=resolved, dtype=torch.long))
                net_pin_offsets.append(
                    torch.zeros((int(owners_cpu.numel()), 2), dtype=torch.float32, device=resolved)
                )
                selected_weights.append(_weight_at(weights_cpu, net_id))

        net_weights = torch.tensor(selected_weights, dtype=torch.float32, device=resolved)
        macro_names = list(getattr(benchmark, "macro_names", []))
        metadata = {
            "benchmark_name": getattr(benchmark, "name", ""),
            "num_ports": int(port_positions.shape[0]),
            "source_num_nets": int(getattr(benchmark, "num_nets", len(nets))),
            "used_pin_level_nets": bool(pin_nets),
        }

        return cls(
            positions=positions,
            sizes=sizes,
            movable_mask=movable_mask,
            fixed_mask=fixed_mask,
            hard_mask=hard_mask,
            soft_mask=soft_mask,
            canvas_width=float(getattr(benchmark, "canvas_width")),
            canvas_height=float(getattr(benchmark, "canvas_height")),
            nets=nets,
            net_pin_offsets=net_pin_offsets,
            net_weights=net_weights,
            port_positions=port_positions,
            macro_names=macro_names,
            metadata=metadata,
        )


def _weight_at(weights: torch.Tensor, index: int) -> float:
    if 0 <= index < int(weights.numel()):
        return float(weights[index].item())
    return 1.0


def _net_offsets_from_pin_nodes(
    benchmark: Any,
    owners: torch.Tensor,
    pin_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    num_hard = int(getattr(benchmark, "num_hard_macros", getattr(benchmark, "num_macros", 0)))
    macro_pin_offsets: Sequence[Any] = getattr(benchmark, "macro_pin_offsets", [])
    offsets: List[Tuple[float, float]] = []
    for owner_raw, pin_raw in zip(owners.tolist(), pin_ids.tolist()):
        owner = int(owner_raw)
        pin_id = int(pin_raw)
        if 0 <= owner < num_hard and owner < len(macro_pin_offsets):
            owner_offsets = torch.as_tensor(macro_pin_offsets[owner], dtype=torch.float32)
            if owner_offsets.ndim == 2 and 0 <= pin_id < int(owner_offsets.shape[0]):
                x_off, y_off = owner_offsets[pin_id].tolist()
                offsets.append((float(x_off), float(y_off)))
                continue
        offsets.append((0.0, 0.0))
    return torch.tensor(offsets, dtype=torch.float32, device=device)
