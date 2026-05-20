# PinePlace

PinePlace is a macro-placement framework for fast proxy-aware macro layout optimization. It combines GPU-assisted analytical placement, candidate selection, and targeted local refinement to produce legal macro placements on standard academic benchmarks. The primary entrypoint is:

```text
submissions/pineplace/placer.py
```

PinePlace has been verified on the full IBM benchmark suite with zero reported overlaps in the measured runs below.

## Method

PinePlace starts from global soft/hard macro placement candidates and then applies proxy-aware local refinement. The default runtime path uses:

- global candidate generation for soft and hard macro layouts,
- hotspot micro coordinate descent for local congestion/density pressure,
- a tuned 600 second heuristic search budget,
- coordinate descent, LNS-style repair, push-chain moves, and basin repair,
- adaptive exact-vs-cheap proxy evaluation so larger benchmarks remain tractable.

The project also contains experimental SA, v2, and v3 refinement hooks, but the default production path is the tuned PinePlace v1 engine.

PinePlace is designed to be useful as a macro initialization or post-processing layer around modern industrial placers. Engines such as DREAMPlace or Xplace can provide strong continuous/global placement, while PinePlace can add proxy-aware macro polishing, legality repair, and targeted congestion/density escape moves.

## Strengths And Limitations

Strengths:

- Produces legal IBM placements with zero reported overlaps in the measured full-suite run.
- Combines global analytical placement with local, exact proxy-aware refinement.
- Uses adaptive cheap screening and exact evaluation to spend expensive proxy calls where they matter most.
- Works well as a refinement layer on top of stronger global placers or learned placement proposals.

Limitations:

- Runtime is intentionally search-heavy; full-suite runs are measured in hours, not seconds.
- Exact proxy evaluation becomes expensive on large designs, so some stages rely on cheaper surrogate scoring.
- The current NG45/WNS/Area validation path depends on a working OpenROAD-flow-scripts environment.
- Results are not uniformly better on every benchmark; the strongest gains come from benchmarks where the heuristic search path is selected.

## Setup

From a fresh checkout:

```bash
git submodule update --init --recursive
uv sync
```

The repository expects the benchmark/evaluation package under `third_party/macro-place-challenge-2026` and uses `external/MacroPlacement` as the benchmark path.

## IBM Proxy Evaluation

Run all 17 IBM benchmarks:

```bash
cd /path/to/PinePlace

PYTHONPATH=$PWD:$PWD/third_party/macro-place-challenge-2026:$PWD/external/MacroPlacement/CodeElements/Plc_client:$PYTHONPATH \
UV_CACHE_DIR=/tmp/uv-cache-pineplace \
PINE_SUBMISSION_TUNED=1 \
PINE_HEURISTIC_TIME=600 \
uv run python scripts/run_proxy_experiments.py \
  --placers submissions/pineplace/placer.py \
  --all \
  --out experiments/results/pineplace_proxy_all.csv \
  --notes pineplace_all
```

Run one benchmark:

```bash
cd /path/to/PinePlace

PYTHONPATH=$PWD:$PWD/third_party/macro-place-challenge-2026:$PWD/external/MacroPlacement/CodeElements/Plc_client:$PYTHONPATH \
UV_CACHE_DIR=/tmp/uv-cache-pineplace \
PINE_SUBMISSION_TUNED=1 \
PINE_HEURISTIC_TIME=600 \
uv run python scripts/run_proxy_experiments.py \
  --placers submissions/pineplace/placer.py \
  --benchmarks ibm01 \
  --out experiments/results/pineplace_proxy_ibm01.csv \
  --notes pineplace_ibm01
```

The benchmark package also exposes an evaluator console script when it is installed as a project, but the commands above are preferred because they run from this repository directly.

## IBM Results

| Benchmark | Proxy | Runtime s | Selected candidate |
|---|---:|---:|---|
| ibm01 | 1.05186 | 829.900 | heuristic_exact_cd_step1+full_soft |
| ibm02 | 1.54313 | 938.320 | heuristic_push_chain+full_soft |
| ibm03 | 1.30651 | 891.660 | hotspot_micro_cd_step1+full_soft |
| ibm04 | 1.33286 | 974.003 | heuristic_push_chain+full_soft |
| ibm06 | 1.66482 | 903.425 | heuristic_push_chain+full_soft |
| ibm07 | 1.42084 | 320.495 | soft_global_hotspot_refined+full_soft |
| ibm08 | 1.48981 | 995.819 | soft_global_density_axis_refined+full_soft |
| ibm09 | 1.09392 | 965.988 | heuristic_push_chain+full_soft |
| ibm10 | 1.31993 | 1200.908 | soft_global_density_axis_refined+full_soft |
| ibm11 | 1.19391 | 1049.744 | soft_global_spread_cong_stage1_refined+full_soft |
| ibm12 | 1.62477 | 943.673 | soft_global_hotspot_refined+full_soft |
| ibm13 | 1.35790 | 1105.474 | soft_global_density_axis_refined+full_soft |
| ibm14 | 1.58089 | 1424.319 | soft_global_density_spread_refined+full_soft |
| ibm15 | 1.57617 | 529.269 | soft_global_density_spread_refined+full_soft |
| ibm16 | 1.44320 | 1004.010 | soft_global_density_axis_refined+full_soft |
| ibm17 | 1.69601 | 2518.498 | soft_global_congestion_refined+full_soft |
| ibm18 | 1.76632 | 917.103 | soft_global_topo_group_migrate+full_soft |

Summary:

- Average IBM proxy: `1.43899`
- Average IBM runtime: `1030.153 s`
- Total IBM runtime: `17512.608 s`
- RePlAce average proxy baseline: `1.45784`
- PinePlace improves over the RePlAce average by `1.29%` proxy cost across the full IBM suite.
- On the 5 benchmarks where the tuned heuristic path is selected, PinePlace improves over the RePlAce average by `2.74%`.
- All listed rows reported `overlaps=0`.

Individual benchmark behavior varies. PinePlace is best understood as a proxy-aware framework with strong local-search wins, not as a claim of uniform dominance on every benchmark.

## NG45 / WNS / Area Evaluation

Full-flow NG45 validation uses OpenROAD-flow-scripts. To run the public `ariane133_ng45` flow locally, first generate a placement tensor:

```bash
cd /path/to/PinePlace

PYTHONPATH=$PWD:$PWD/third_party/macro-place-challenge-2026:$PYTHONPATH \
UV_CACHE_DIR=/tmp/uv-cache-pineplace \
PINE_SUBMISSION_TUNED=1 \
uv run python - <<'PY'
from pathlib import Path
import torch
from macro_place.benchmark import Benchmark
from submissions.pineplace.placer import PinePlace

benchmark = Benchmark.load(
    "third_party/macro-place-challenge-2026/benchmarks/processed/public/ariane133_ng45.pt"
)
placement = PinePlace().place(benchmark)

out = Path("experiments/results/pineplace_ariane133_ng45.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(placement, out)
print(f"saved {out}")
PY
```

Then run the ORFS wrapper:

```bash
cd /path/to/PinePlace/third_party/macro-place-challenge-2026

PYTHONPATH=/path/to/PinePlace:$PWD:$PYTHONPATH \
UV_CACHE_DIR=/tmp/uv-cache-pineplace \
uv run python scripts/evaluate_with_orfs.py \
  --benchmark ariane133_ng45 \
  --placement /path/to/PinePlace/experiments/results/pineplace_ariane133_ng45.pt \
  --orfs-root /path/to/OpenROAD-flow-scripts \
  --no-docker
```

Remove `--no-docker` to use ORFS Docker mode. Native mode requires `yosys` and `openroad` on `PATH`; Docker mode requires permission to access the Docker daemon.

Local ORFS status:

- The wrapper successfully loaded `ariane133_ng45`, generated macro placement TCL, and computed placement proxy `0.754263`.
- WNS, TNS, and Area are not claimed from local results.
- Native ORFS failed locally because `yosys` and `openroad` were unavailable.
- Docker ORFS failed locally because the user did not have Docker socket permission.

The command above follows the provided ORFS path. It should produce WNS/TNS/Area in an evaluation environment or on any machine with a working OpenROAD-flow-scripts setup.

## Docker Runtime

This repository includes a `Dockerfile` for containerized environments that build the project image directly. It starts from:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

The image copies `pine_place/` and `submissions/` so the primary placer can import the full PinePlace framework even if an evaluator invokes only `submissions/pineplace/placer.py`.

## Future Work

- Tighter integration with modern global placement frameworks.
- Learned candidate ranking for local search moves.
- Stronger routability-aware gradients and faster proxy approximations.
