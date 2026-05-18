#!/usr/bin/env python3
"""Print PyTorch CUDA availability and run a tiny CUDA tensor smoke test."""

from __future__ import annotations

import torch


def main() -> int:
    print(f"torch version: {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    print(f"CUDA device count: {torch.cuda.device_count() if available else 0}")

    if not available:
        print("current device name: none")
        print("CUDA tensor smoke: skipped")
        return 1

    device_id = torch.cuda.current_device()
    print(f"current device name: {torch.cuda.get_device_name(device_id)}")
    try:
        x = torch.arange(8, device="cuda", dtype=torch.float32)
        y = (x * x).sum()
        torch.cuda.synchronize()
        print(f"CUDA tensor smoke: ok result={float(y.cpu()):.1f}")
        return 0
    except Exception as exc:
        print(f"CUDA tensor smoke: failed {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
