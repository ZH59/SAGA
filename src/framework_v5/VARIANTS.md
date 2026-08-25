# framework_v5 SAG Variants

The V5 GPU SAG framework supports four SAG-paper analyses, selected by
either a CLI flag or env var:

```
expand_test_v5 --variant {rtss24|rtss17|ecrts19|ecrts22} ...
SAG_V5_VARIANT={rtss24|rtss17|ecrts19|ecrts22}    (fallback if no --variant)
```

CLI flag takes precedence. Default is `rtss24`. Aliases listed in
`include/v5_variant.h`. Run `expand_test_v5 --help` for the full list of
env vars and variant model citations.

All internal optimization toggles (IJP, F-cache hoist, merge level, etc.) are
auto-selected per variant. Users do not need to set `SAG_V5_IJP` etc. — those
become deprecated back-doors that take effect only when `SAG_V5_VARIANT=rtss24`.

## Variant table (current as of Phase 4)

| Variant | Paper | Status | Model summary |
|---|---|---|---|
| `rtss24` (default) | Srinivasan, Gunzel, Nelissen — RTSS 2024 | **Implemented** | Limited-preemptive self-suspending + event-driven delay-induced. Multi-core global. 20/20 nptest match on n16_m4_u20+u50 default gate. |
| `rtss17` | Nasri, Brandenburg — RTSS 2017 | **Implemented** | Exact uniprocessor non-preemptive. K1+K2 forward to V5 default with m=1; new V5MergeKernelRTSS17 enforces conservative-only merge. **Byte-identical** to nptest -m 1 on small uniprocessor inputs. V5 K2 STEP 7's Lemma 6 clamp had an m=1 bug (also affecting rtss24 -m 1) -- now fixed. See PHASE2_STATUS.md. |
| `ecrts19` | Nasri, Nelissen, Brandenburg — ECRTS 2019 | **Implemented** | Limited-preemptive parallel DAG via segment expansion (each task's segments are V5 jobs connected by zero-delay edges). On n16_m4_u20/taskset_000: byte-identical to nptest reference for all 80 jobs. See PHASE4_STATUS.md. |
| `ecrts22` | Nelissen, Marce-i-Igual, Nasri — ECRTS 2022 | **Implemented** | Non-preemptive moldable gang. Parser auto-detects `{p:cmin:cmax;...}` cost format. K1 has parallelism-aware `A_pair[p-1]` eligibility + greedy-earliest p selection (single-emit). K2 commits p cores via STEP 7 multi-core sort. Rigid p=1 inputs: routed to rtss24 path (byte-identical, skips 111MB side-array alloc). Rigid p>=2 + moldable: ECRTS22-specific path with single-emit. SCHED verdicts match nptest 100%. p>=2 WCRTs loose vs nptest in some cases (gang-aware rho_hp pending). See PHASE3_STATUS.md. |

## Hard invariants (apply to every variant)

These are part of the project contract; every implementation must honor them:

1. **Strict layer-by-layer.** No inter-layer overlap. Layer L expansion + merge
   completes before any work at L+1 begins. Enforced by host
   `cudaStreamSynchronize(0)` barriers (`framework_v5_main.cu:4169` after K1+K2,
   `:2151` after merge).
2. **Within a layer, ordering does not matter.** K1 / K2 may run any order
   over states or pairs.
3. **No truncation cheats.** Every post-merge state at layer L MUST appear as a
   parent in layer L+1. The spill ring round-trips every state. Overflow sets
   `d_trunc_flag` and aborts loudly.
4. **Paper math is authoritative.** Every body header cites the equations it
   implements; deviations require a comment.
5. **Merge is optional.** Skipping merges never drops a state; unmerged states
   pass forward to the next layer's K1 input. Variant-specific compatibility
   predicates trade compute for downstream IO.

## Operator surface

| Operator | File | Per-variant override |
|---|---|---|
| K1 (eligibility) | `include/k1_body_<variant>.cuh` | `K1Body<variant>` template (warp per state) |
| K2 (successor)   | `include/k2_body_<variant>.cuh` | `K2Body<variant>` (warp per (state,job)) |
| Merge            | `include/merge_body_<variant>.cuh` | `dev_check_compat_<variant>()` + V5MergeKernel wrapper |

## Auto-tricks table

Hidden behind the variant choice; documented here for reproducibility. Source
of truth: `include/v5_variant.h::v5_default_tricks()`.

| Variant | IJP | Merge level | F_MAX_PER_STATE | m forced to 1 | Notes |
|---|---|---|---|---|---|
| `rtss24` | off (override via `SAG_V5_IJP=1`) | lossy `l1` | 32 | no | Current path; matches existing benchmarks |
| `rtss17` | off (always) | conservative `c2` | 32 | yes | Lossy merge would void exactness |
| `ecrts19` | off (always) | lossy `l1` | 48 | no | Segment expansion makes IJP redundant; bigger F-set per state |
| `ecrts22` | off (always) | lossy `l1` w/ parallelism check | 32 | no | Parallelism asymmetry breaks IJP equivalence |

## Per-variant implementation notes

### `rtss17` — Phase 2 (next implementation)

- **State layout**: existing `SAGStateLayoutV5` with `m=1`. `A_pair[1]` holds
  the single core's `[A_min, A_max]`.
- **K1**: same two-pass structure as `K1BodyV5` (sub-phase 1 = `rho_max`,
  sub-phase 2 = `rho_min` + eligibility). Single-core path drops the multi-
  core priority dominance scan.
- **K2**: drop IJP path entirely (would void exactness). Drop the multi-core
  sort/replacement on `A_pair`; collapse to a single update.
- **Merge**: conservative c2 (sub-interval containment) for exactness. Lossy
  l1 is not exposed (would silently over-approximate).
- **CSV input**: same as current. `-m` is honored as `1` (whatever the user
  passes is overwritten with `1`); this is logged at startup.
- **nptest oracle**: `nptest -m 1 jobs.csv` (or `nptest jobs.csv`).

### `ecrts22` — Phase 3

- **Input format extension**: optional 9th column on `jobs.csv` rows = gang
  cost map `{p_min:p_max | C_min^p1:C_max^p1 ; ... }` (mirroring nptest's
  gang format, README §"Gang jobs").
- **Constant-memory tables**: extend `const_job_arrays_v5.cuh` with
  `c_p_min[N]`, `c_p_max[N]`, `c_cost_min[N][P_MAX]`, `c_cost_max[N][P_MAX]`.
- **ValidPair extension**: add a parallelism field. Decision in Phase 3:
  either widen `ValidPair` to a `ValidPairGang` struct (9 bytes -> 12 bytes
  padded) or pack `p` into the high bits of `f_min` (lossless for `p<=16`).
- **K1**: enumerate parallelism levels; emit one pair per (state, j, p).
- **K2**: commit `p` cores from sorted `A_pair` prefix; cost depends on `p`.
- **Merge**: same as V5 + per-job parallelism equality check on the
  intersection of F-masks.
- **nptest oracle**: `nptest -m M jobs.csv` with gang-format jobs.

### `ecrts19` — Phase 4

- **Strategy**: SEGMENT EXPANSION (plan option (a)). Each task becomes a chain
  of segment-jobs connected by zero-delay finish-to-start edges. K1/K2/Merge
  bodies are nearly identical to NPG24; the delta lives in the host parser.
- **Input format extension**: a separate segment file (CSV) or inline
  `{c_min:c_max; c_min:c_max; ... }` per task. Phase 4 picks the simpler one.
- **Limited-preemption semantics**: emerges automatically from segment-level
  precedence — a higher-prio task's segment becomes K1-eligible at any
  segment boundary in the SAG.
- **Bound**: `n` grows by avg-segments-per-task. The `n<=40, m<=20` industrial
  envelope must be honored on segment-expanded inputs (a `n=20` task set with
  3 segments each is `n=60` post-expansion — at the limit; document this).
- **nptest oracle**: `nptest -m M jobs.csv -p prec.csv` with the segment-
  expanded inputs.

## Phase 1 (this session) deliverable

- `include/v5_variant.h` with the enum + parser + tricks table.
- 9 scaffold body headers (`k1_body_<variant>.cuh`, `k2_body_<variant>.cuh`,
  `merge_body_<variant>.cuh` for each of `rtss17`, `ecrts19`, `ecrts22`).
- `controller.cu` adds stub kernel wrappers + accessors so that Phase 2-4
  only fills bodies, not dispatch.
- `framework_v5_main.cu` parses `SAG_V5_VARIANT`, validates, and for
  unimplemented variants prints a friendly message + exits non-zero before
  any GPU work.
- `VARIANTS.md` (this file).

The default `rtss24` path remains byte-identical to the prior V5 build.

## Memory budget and host spill (applies to all variants)

V5 wave sizing is bounded by VRAM (`SAG_V5_VRAM_FRAC`, default 0.85). When a
layer's projected scratch + post-state footprint exceeds the available
budget, V5 spills to host pinned RAM via the spill ring; once spilled,
subsequent layers stream child states back through PCIe.

**`SAG_V5_HOST_SPILL_GB` is opt-in (default 0).** With the default of 0,
spill-required workloads emit
`[V5 WARN] spill required but SAG_V5_HOST_SPILL_GB=0` and TRUNCATE the
analysis loudly (this is the safe behavior so we never silently approximate).
Set `SAG_V5_HOST_SPILL_GB=N` to allow up to N gibibytes of host pinned
memory for spill.

Empirical sizing on this branch (chain-DAG `n24_m6_u20/taskset_000`):

```
SAG_V5_HOST_SPILL_GB=200 expand_test_v5 -m 6 jobs.csv jobsprec.csv
  Schedulable:            YES
  Total layers:           121
  Scratch spill chunks:   96 (states 1958072411)
  Controller wall:        326486.78 ms
  Spill: total_spilled=289667110 states (131.65 GB), total_xferd=3576.25 GB
```

Rule of thumb for industrial deployment:
- `n <= 20`: typically fits in VRAM, no spill needed.
- `n = 24`: budget ~150 GB host RAM (verified above).
- `n = 28+`: budget several hundred GB; the iter620 dominance pruning
  collapses the merge heavy-tail but does not collapse parent-state
  retention.

The `--variant` choice does not change this budget envelope; rtss17 (uni)
typically uses much less than rtss24/ecrts19/ecrts22 because m=1 limits the
A_pair fan-out. ecrts22 moldable inputs in particular grow ValidPair
emissions per layer linearly in the average parallelism range.
