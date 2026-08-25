# Reference environment

These are the authors' measurements. `results_full.csv` / `results_full.json` were
produced by `./run_full.sh` on this machine:

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, 81920 MiB, 535.183.01 |
| CPU | 2x Intel Xeon Gold 6326 @ 2.90GHz (32 cores / 64 threads total) |
| RAM | 1.0 TiB |
| OS | Ubuntu 22.04.5 LTS, Linux 6.8 |
| CUDA Toolkit | 12.5.82 |
| Host compiler | GCC 11.4.0 |
| CMake / Python | 3.22.1 / 3.10.12 |
| GPU architectures built | sm_70, sm_75, sm_80, sm_90 |
| nptest commit | 47a45b69cbc90d0d0fd36d7c9ac8edaf041a6f05 (sequential, PARALLEL_RUN=OFF) |
| Baseline cap | 1800 s per task set |

Note the paper reports an A100 40GB; the machine available for this artifact has an
A100 80GB PCIe. The workload configuration shipped here fits comfortably in both,
and the memory-pressure test below shows response times do not change with the
memory cap -- only wall time does.

## Memory-pressure check (simulating smaller GPUs)

Same task set (`taskset_000`), same binary, VRAM artificially capped:

| configuration | wall time | response times |
|---|---|---|
| full 80 GiB | 97.99 ms | baseline |
| `SAG_VRAM_LIMIT_MB=8000` (8 GiB card) | 107.57 ms | identical |
| `SAG_VRAM_LIMIT_MB=4000` + `SAG_V5_HOST_SPILL_GB=16` | 121.72 ms | identical |

Memory pressure changes speed, not results.

## Determinism

The solver is deterministic: repeated runs of the same task set reproduce the
same reported wall time to four decimal places and byte-identical response-time
output.
