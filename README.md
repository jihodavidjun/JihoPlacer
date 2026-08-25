# PinePlace

PinePlace is a macro placement framework for fast proxy-aware macro layout optimization. It combines GPU-assisted analytical placement, candidate selection, and targeted local refinement to produce legal macro placements on standard academic benchmarks. 

PinePlace was originally developed for the **Partcl × Hudson River Trading (HRT) Macro Placement Challenge 2026** and was subsequently refined and evaluated across the IBM benchmark suite.

The primary entry point is:

`submissions/pineplace/placer.py`

## Method

PinePlace generates multiple global macro-placement candidates and then selectively refines promising solutions using local search.

The default flow combines:

- global candidate generation for soft and hard macro layouts,
- density- and congestion-aware refinement,
- hotspot-focused coordinate descent,
- push-chain moves for local placement repair,
- cheap candidate screening followed by more expensive proxy evaluation,
- final candidate selection based on the placement proxy.

The main idea is to avoid applying expensive refinement uniformly. PinePlace first explores a broader set of candidate placements, identifies promising regions of the search space, and then applies targeted local optimization where additional proxy evaluations are most useful.

The framework is implemented directly in PyTorch and NumPy rather than wrapping an external analytical placer or relying on custom CUDA extensions.

Experimental refinement hooks are also present in the repository, but the reported IBM results use the tuned PinePlace path.

## Setup

From a fresh checkout:

```bash
git submodule update --init --recursive
uv sync
```

The repository expects the benchmark/evaluation package under `third_party/macro-place-challenge-2026` and uses `external/MacroPlacement` as the benchmark path for the IBM evaluation flow.

## Running PinePlace

### Run all 17 IBM benchmarks

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

### Run one benchmark

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

## IBM Benchmark Evaluation

PinePlace is evaluated across the full 17-design IBM benchmark suite available through the project evaluation environment.

### Results

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

### Aggregate Results

- PinePlace average proxy: **1.43899**
- RePlAce average proxy baseline: **1.45784**
- Average improvement over RePlAce: **1.29%**
- PinePlace achieves a lower placement objective than RePlAce on **11 of 17** benchmarks.
- All listed PinePlace placements report `overlaps=0`.
- Average runtime: **1030.153 s** per benchmark
- Total measured runtime: **17512.608 s**

The strongest gains are benchmark-dependent. PinePlace should therefore be viewed as a macro-placement framework that combines global exploration with targeted local refinement rather than as a method that dominates the baseline on every design.

## Design Tradeoffs and Limitations

PinePlace intentionally uses a search-heavy refinement strategy. This enables targeted exploration of alternative macro configurations, but it also makes the current implementation substantially slower than a single-pass placement method.

Exact proxy evaluation becomes expensive on larger designs, so PinePlace uses cheaper candidate screening before committing additional evaluation effort.

The measured runtime varies considerably by benchmark. The slowest listed run is `ibm17` at approximately **2518 s**.

Results are also benchmark-dependent: although PinePlace improves the average proxy and outperforms RePlAce on 11 of 17 designs, it does not improve the objective on every benchmark.

The current results should therefore be interpreted as **placement-proxy results on the IBM benchmark suite**, not as post-route timing, power, or area claims.

## Docker Runtime

The repository includes a `Dockerfile` based on:

`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

The image includes the PinePlace framework and submission code so that:

`submissions/pineplace/placer.py`

can import the full implementation in containerized evaluation environments.

## Future Work

Potential extensions include:

- tighter integration with modern global-placement frameworks,
- faster density, congestion, and candidate scoring,
- learned candidate ranking for local-search moves,
- stronger routability-aware optimization,
- faster approximations for expensive placement-proxy evaluation.
