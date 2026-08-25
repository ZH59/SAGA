# SAGA — RTSS 2026 Artifact

GPU-accelerated schedule-abstraction graph (SAG) analysis, compared against the
CPU baseline `nptest` from [SAG-org](https://github.com/SAG-org).

This artifact provides a **measured, end-to-end comparison on one workload
configuration**: the GPU solver against the `nptest` CPU baseline, on the same
task sets, on an NVIDIA A100. It is built to run without crashing on any CUDA GPU
from Volta (compute capability 7.0) onward.

**Scope.** The measurements here were taken on an A100 80GB PCIe. The paper's
reported platform is a B200, with H100 and A100 40GB as comparison devices; this
artifact does not regenerate the paper's figures and makes no claim about those
platforms. What it does establish is stated in §1 and is fully reproducible from
this package.

---

## 1. What this artifact claims

| # | Claim | How to check it | Hardware needed |
|---|---|---|---|
| C1 | Wherever both tools produce a verdict, they **agree** — 0 mismatches over the 10 shipped task sets | `./run_smoke.sh`, `./run_full.sh` — "verdict mismatches: 0" | any CUDA GPU |
| C2 | **No unsafe result:** the solver never reports SCHED where the exact oracle reports UNSCHED, and never reports SCHED while the exact oracle puts a job past its deadline | same runs — "UNSAFE verdict flips: 0" and "deadline violations: 0" | any CUDA GPU |
| C3 | The GPU solver is **substantially faster** than `nptest` on the same task sets | same runs — the `speedup` column | A100 to match the reported figure |

**C1 and C2 are hardware-independent** and are the claims an evaluator without an
A100 should check. **C3 is hardware-dependent**: the numbers in §5 were measured on
the machine described there, and your GPU will produce different wall times.

Safety is judged at the two levels the analysis is actually used at, following the
same definitions as the authors' own accuracy tooling: an **unsafe verdict flip**
(SCHED claimed where the exact oracle says UNSCHED) and a **deadline violation**
(SCHED claimed while the exact oracle puts some job past its deadline). Both are
zero here. §9 additionally reports jobs whose bound comes out *tighter* than the
exact one — a measured difference that violates no deadline and flips no verdict.

## 2. Requirements

The solver ships as a **prebuilt binary** (`bin/saga`). It is statically linked
against the CUDA runtime and libstdc++, so it needs no CUDA Toolkit and no C++
toolchain — only:

* Linux x86-64 with an **NVIDIA driver** and a GPU of **compute capability 7.0–9.x**
  (Volta, Turing, Ampere, Ada, Hopper). The solver uses cooperative grid
  synchronisation, which is SM_70+. See §8 for Blackwell.
* An NVIDIA driver new enough for CUDA 12.x. Verified on **535.183.01**; per
  NVIDIA's CUDA minor-version compatibility, any 525-or-newer driver should serve,
  though only 535.183.01 was tested here.
* **`glibc` 2.34 or newer.** Measured, not assumed: `bin/saga` imports
  `dlopen@GLIBC_2.34`. That means Ubuntu 22.04+, Debian 12+, RHEL/Rocky/Alma 9+,
  or Fedora 35+. On an older distribution (Ubuntu 20.04, CentOS 7, RHEL 8) the
  binary **will not start**; `tools/preflight.sh` checks this first and says so
  rather than letting the loader fail with a bare version error.
* Python 3.8+ for the measurement harness.

Building the **baseline** does need a toolchain (CMake ≥ 3.18, a C++17 compiler,
`git`, and network access once) — see §3. The baseline is deliberately built from
its public upstream source rather than shipped as a binary, so you can satisfy
yourself that it is the genuine `nptest` at the commit the paper used.

## 3. Setup (about 1 minute)

The solver needs no build. Only the baseline does:

```bash
./baseline/build_nptest.sh    # CPU baseline -> baseline/build/nptest
```

That takes about 15 seconds on the reference machine. Check the solver runs at all
with:

```bash
./bin/saga --help
```

`baseline/build_nptest.sh` clones `schedule_abstraction-main` at the pinned
commit `47a45b69cbc90d0d0fd36d7c9ac8edaf041a6f05` and builds it. To build from a
checkout you already have, and skip the network:

```bash
NPTEST_SOURCE=/path/to/schedule_abstraction-main ./baseline/build_nptest.sh
```

## 4. Run

```bash
./run_smoke.sh         # 1 task set  -- kick the tires
./run_full.sh          # 10 task sets -- the reported numbers
./tests/run_tests.sh   # unit + system tests, no GPU required
```

**To run it on your own inputs** — the point of a closed-source artifact is that
you still control the experiment:

```bash
python3 tools/gen_taskset.py --tasks 24 --cores 6 --util 0.4 \
        --count 5 --seed 99 --out my_inputs
python3 tools/compare.py --dataset my_inputs --cores 6 \
        --gpu-bin bin/saga --nptest-bin baseline/build/nptest \
        --out my_results.csv
```

`INPUT_FORMAT.md` documents both CSV schemas in full, so you can also write task
sets by hand or emit them from your own generator, and read every number the
solver produces.

Both run scripts begin with `scripts/preflight.sh`, which prints the detected
GPU, checks that the binary you built actually contains code for it, and enables
host spill on cards with less than 24 GiB of VRAM.

`run_full.sh` caps the baseline at **1800 s per task set**. The paper uses a
10,000 s cap; 1800 s keeps the run within a reviewer's patience. One task set is
far harder than the others and dominates the total runtime. Task sets where the
baseline hits the cap are reported with a `>=` speedup and excluded from the
median, because a censored measurement only gives a lower bound. Use
`NPTEST_TIMEOUT=10000 ./run_full.sh` for the paper's cap, or a smaller value to
finish sooner.

## 5. Results measured on an A100 80GB PCIe

Produced by `./run_full.sh` on the machine described in
`reference_results/a100/ENVIRONMENT.md`.

Measured with the `bin/saga` binary shipped in this package, on the exact
task sets in `dataset/`. The baseline is the `nptest` build produced by
`baseline/build_nptest.sh` at the pinned commit.

| task set | solver | nptest | speedup | verdict | per-job WCRT vs exact |
|---|---:|---:|---:|---|---|
| taskset_000 | 0.098 s | 8.2 s | 83.9x | agree | 100/100 exact |
| taskset_001 | 0.056 s | 1.8 s | 31.4x | agree | 91/100 exact |
| taskset_002 | 0.049 s | 1.4 s | 28.8x | agree | 100/100 exact |
| taskset_003 | 40.252 s | 300.0 s (cap) | &ge; 7.5x | agree | not compared |
| taskset_004 | 0.036 s | 0.2 s | 6.5x | agree | 100/100 exact |
| taskset_005 | 20.284 s | 300.0 s (cap) | n/a | TRUNCATED | not compared |
| taskset_006 | 0.342 s | 40.2 s | 117.6x | agree | 100/100 exact |
| taskset_007 | 0.743 s | 102.6 s | 138.1x | agree | not compared |
| taskset_008 | 1.910 s | 286.0 s | 149.7x | agree | 91/100 exact, **6 tighter** |
| taskset_009 | 1.104 s | 149.2 s | 135.1x | agree | not compared |

* **Median speedup 117.6x** (range 6.5x–149.7x), over the 8 task sets where both tools finished.
* **Baseline hit the 300 s cap on 2** task sets (taskset_003, taskset_005); those speedups are lower bounds and are excluded from the median.
* **Solver returned no answer on 1** task set (taskset_005) — see §9.
* **No unsafe result: 0 unsafe verdict flips and 0 deadline violations.** This is the safety invariant: the solver never claimed SCHED where the exact oracle says UNSCHED, and never claimed SCHED while the exact oracle put a job past its deadline.
* **Verdict agreement: 0 mismatches** wherever both tools produced a verdict.
* Per-job bounds: 582/600 equal the exact bound, 6 come out tighter than it (see §9). The 4 task sets marked "not compared" have no exact reference because an unmerged `nptest` run cannot terminate on them.

Wall times on your GPU will differ — only the WCRT and verdict columns should not.

## 6. Reading the output

Each run prints one line per task set and writes `results_*.csv` / `results_*.json`:

| column | meaning |
|---|---|
| `gpu_wall_s` | solver wall time, as the solver itself reports it |
| `nptest_cpu_s` | baseline CPU time, as `nptest` itself reports it |
| `speedup` | `nptest_cpu_s / gpu_wall_s` |
| `nptest_capped` | true when the baseline was stopped at the cap (speedup is a lower bound) |
| `ref_exact` / `ref_loose` / `ref_unsound` | per-job WCRT vs the exact bounds |

`ref_loose` counts jobs where the GPU bound is **larger** than exact — sound but
pessimistic. `ref_unsound` counts jobs where it is **smaller** than exact, which
would be a real defect. **`ref_unsound` must be 0.**

The comparison uses `jobs.rta.exact.csv`, produced by `nptest --merge=no` (all
state merging disabled). This matters: `nptest`'s *default* merge level is `l1`,
which is itself lossy, so comparing against a default run is not a soundness test.

## 7. Layout

```
bin/saga                 the solver, prebuilt and stripped (see §10)
INPUT_FORMAT.md          CSV schemas for input and output; env variables
run_smoke.sh             1 task set
run_full.sh              all 10 task sets
tools/compare.py         the harness: runs both tools, compares, writes CSV/JSON
tools/gen_taskset.py     generate your own task sets
tools/preflight.sh       GPU detection, architecture check, memory sizing
baseline/                fetches + builds the nptest baseline from upstream
dataset/                 task sets, with exact nptest reference outputs
tests/                   unit + system tests (no GPU needed)
reference_results/a100/  the authors' measurements and exact environment
VARIANTS.md              the solver's variants and full option surface
bin/saga.arch            GPU architectures built into the binary (read by preflight)
```

`bin/saga --help` refers to `framework_v5/VARIANTS.md`, which is a path from the
solver's own build tree. In this package that document is simply `VARIANTS.md`, at
the top level.

This artifact exercises the `rtss24` variant, the default, which corresponds to
the EDD analyzer the paper compares against.

## 8. Running on a GPU other than an A100

This is expected to work, and `run_smoke.sh` is the fastest way to confirm it.

* **Compute capability 7.0–9.x** — `bin/saga` carries native code for sm_70,
  sm_75, sm_80 and sm_90. A cubin for `sm_Xy` runs on any device `X.z` with
  `z ≥ y`, so sm_80 also covers 8.6/8.7/8.9 (RTX 30xx/40xx, A10, L4, L40) and
  sm_90 covers 9.x. This span was chosen so the shipped binary runs natively on
  the GPUs an evaluator is likely to own, not just on the A100.
* **Blackwell (10.x) and newer — not supported by this binary.** It was built with
  CUDA 12.5, whose `nvcc` cannot target compute capability 10.x, and relocatable
  device code prevents embedding JIT-able PTX as a fallback. `tools/preflight.sh`
  detects this and says so explicitly rather than letting the run die with "no
  kernel image is available for execution on the device". A build for these cards
  requires CUDA 12.8+ at compile time — contact the authors.
* **Less than 24 GiB of VRAM** — preflight enables `SAG_V5_HOST_SPILL_GB=16` (32
  otherwise), so layers that do not fit spill through pinned host memory. You need
  that much free system RAM. Tune with `SAG_V5_HOST_SPILL_GB=N`. Verified on the
  reference A100 by capping device memory with `SAG_VRAM_LIMIT_MB` **on
  `taskset_000` only**: at a simulated 8, 4 and 2 GiB the shipped binary still
  completes and produces **byte-identical** response times, taking 107, 122 and
  149 ms against 98 ms unrestricted. Do not read this as a guarantee for the whole
  dataset — `taskset_005` returns `TRUNCATED` even with the full 80 GiB and 32 GiB
  of spill (§9), and a smaller card will hit that limit on more task sets, not
  fewer.
* **No GPU at all** — the solver cannot run; preflight exits with an explanation
  rather than crashing. `reference_results/a100/` holds the authors' measurements,
  and `./tests/run_tests.sh` still passes, since the tests do not need a GPU.
* **Not x86-64 Linux** — the binary is an x86-64 Linux ELF. No other platform is
  supported.

## 9. Limitations

* **Scope.** This artifact covers one workload configuration (n = 20, m = 5,
  U′ = 30%) and the `rtss24` variant. It is not the paper's full evaluation grid.
* **Some per-job bounds come out tighter than exact.** On the shipped
  configuration, 6 of 600 compared jobs (all in `taskset_008`) report a WCRT one
  tick *below* the exact bound. This is a measured difference, not a deadline
  violation and not a verdict flip — both safety checks in §1 are zero — but a
  bound tighter than exact is not a valid upper bound in general, so it is
  reported rather than omitted.

  | configuration | jobs compared | tighter than exact | max difference |
  |---|---:|---:|---:|
  | `full_n20_m5_u30` (shipped, U′ = 30%) | 600 | 6 (all in `taskset_008`) | 1 tick |
  | `known_limitation_n16_m4_u50` (U′ = 50%) | 80 | 14 | 2 ticks |

  Only the six task sets carrying exact reference bounds are counted. The oracle
  is `nptest --merge=no`; on every task set where it was run, nptest's default
  `l1` merge produced byte-identical bounds, so this is not an artefact of the
  merge level.

  This is a known, pre-existing characteristic of the implementation rather than
  something this artifact discovered: the authors' own accuracy runs recorded the
  same effect on a different grid — 190 of 2240 jobs, maximum 12 ticks — while
  likewise recording zero unsafe verdict flips and zero deadline violations.

  To exercise the higher-utilization configuration:

  ```bash
  python3 tools/compare.py --dataset dataset/known_limitation_n16_m4_u50 --cores 4 \
      --gpu-bin bin/saga --nptest-bin baseline/build/nptest \
      --work work/known --out results_known.csv
  ```

  That configuration is outside the regime the shipped results cover; the run
  reports its per-job differences the same way.
* **The solver cannot solve one of the ten shipped task sets.** On `taskset_005`
  it reports `Schedulable: TRUNCATED` — the state space exceeds its budget and it
  returns no answer. This happens on the reference A100 80GB *with* 32 GB of
  host-pinned spill enabled, so it is a capability limit rather than a
  misconfiguration. `nptest` does not finish that task set either, within the
  300 s cap. The harness reports it as `TRUNCATED`, excludes it from the speedup
  median and from verdict agreement, and does not count it as a passing result.
* **The baseline dominates the runtime.** `nptest` is exponential in the task-set
  size; the harness cap exists for that reason. Two of the ten task sets
  (`003`, `005`) hit the 300 s cap, so their speedups are lower bounds only, and
  four task sets have no exact reference bounds because an unmerged `nptest` run
  cannot terminate on them in reasonable time.
* **Wall times are not portable.** Only the A100 numbers in §5 are the claim.

## 10. About the shipped binary

This artifact is distributed as a binary rather than as source. What that means
concretely:

* `bin/saga` is built from the same solver used for the measurements in §5, with
  `nvcc` from CUDA 12.5, at `-O3`.
* It embeds native GPU code for **sm_70, sm_75, sm_80 and sm_90**. A cubin for
  `sm_Xy` runs on any device `X.z` with `z >= y`, so sm_80 also covers 8.6/8.7/8.9.
  Verify with `cuobjdump --list-elf bin/saga` if you have the CUDA Toolkit.
* It is **statically linked** against the CUDA runtime, libstdc++ and libgcc, so
  it depends only on `libc` and the NVIDIA driver. Check with `ldd bin/saga`.
* It is **stripped**: no symbol table, no debug sections, no source paths, and no
  line tables. It was also built from a neutral directory so no internal paths
  are embedded.
* It performs no network I/O. Checked with `strace -f -e trace=network`, which
  recorded **zero** network syscalls across a complete run. It imports about 230
  symbols from libc — ordinary file, memory and process calls such as `open`,
  `read`, `write`, `mmap` and `ioctl` — and none of the networking ones
  (`socket`, `connect`, `getaddrinfo` and friends are absent from the dynamic
  symbol table). `dlopen` is among the imports; the statically linked CUDA runtime
  uses it to load the NVIDIA driver, and it is what sets the glibc 2.34 floor.
  Inspect the full list yourself with `readelf --dyn-syms bin/saga`.
* It writes the `*.v5.rta.csv` file next to its input, and its stdout. Nothing else.

Everything *around* the solver is open and readable: the measurement harness
(`tools/compare.py`), the input generator (`tools/gen_taskset.py`), the GPU
preflight (`tools/preflight.sh`), the tests, the datasets, and the exact reference
bounds. You can therefore audit exactly how the numbers in §5 are produced, and
run the solver on inputs of your own choosing, without the solver's source.

Two honest consequences of this choice:

1. **The RTSS "extensible" criteria are not fully satisfiable.** The criteria list
   source availability among the necessary conditions for an element to count as
   extensible. Every element here is *repeatable*, and the inputs are freely
   modifiable, but source-level extension is not possible.
2. **A binary is not a black box.** The compiled GPU kernels can be disassembled
   with `cuobjdump -sass`. Stripping removes names, paths and symbols; it does not
   render the implementation unreadable to a determined reader.

## 11. Speedup depends strongly on the configuration

The speedup this artifact measures is not a single number for the tool — it grows
with the size of the analysis problem, because the CPU baseline is exponential
while the GPU cost grows far more slowly. The configuration shipped here is
deliberately small so that the baseline finishes in minutes rather than hours.

Geomean speedup of the `rtss24` variant over `nptest`, from the authors' own
benchmark grid (A100, baseline capped at 10,000 s per task set):

| n, m | U′ | `rtss24` speedup |
|---|---:|---:|
| 24, 6 | 20% | 92.3× |
| 28, 7 | 20% | 100.5× |
| 28, 7 | 50% | 52.7× |
| **32, 8** | **50%** | **204.0×** |
| 36, 9 | 20% | 70.4× |

The configuration in `dataset/` is **n = 20, m = 5**, smaller than anything in that
grid, and the median it produces (§5) sits just below the 92.3× reported at
n = 24. That is consistent, not a discrepancy: reproducing a ~200× figure requires
the larger configurations, where a single baseline run takes **tens of minutes to
hours** and a portion of them do not finish within the 10,000 s cap at all.

One point at the largest configuration was re-measured here directly, on the task
set whose recorded baseline could be matched unambiguously to the shipped data
(160 jobs, both tools agreeing on SCHED):

| n, m, U′ | baseline (recorded, 10,000 s cap) | solver (measured here) | speedup |
|---|---:|---:|---:|
| 32, 8, 50% | 2553.4 s | 0.514 s | **4972×** |

Two honest caveats on that row. The baseline time is the authors' recorded
measurement rather than one re-run for this artifact, and only that single task
set of the ten could be matched to it by job count — the others in the recorded
reference have different job counts and are therefore different task sets. It is
offered as a datapoint at scale, not as a headline the artifact establishes.
