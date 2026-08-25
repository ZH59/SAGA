# Input and output formats

Everything the solver reads and writes is plain CSV, so you can supply your own
task sets and inspect every number it produces. `tools/gen_taskset.py` generates
valid inputs for you; this document describes the format so you can also write
them by hand or from your own generator.

## Invoking the solver directly

```
bin/saga --variant rtss24 -m <CORES> <JOBS.csv> [<PRECEDENCE.csv>]
```

* `--variant` — `rtss24` (default), `rtss17`, `ecrts19`, or `ecrts22`.
  `rtss24` is the limited-preemptive self-suspending model used by this artifact.
  It can also be set with `SAG_V5_VARIANT=<variant>`.
* `-m <CORES>` — number of processors.
* The precedence file is optional; without it, jobs are independent.

The solver writes `<JOBS>.v5.rta.csv` next to the input and prints a verdict and
timing to stdout.

## `jobs.csv` — one row per job

Header row required. All values are non-negative integers in an abstract discrete
time unit ("ticks"); there is no implied wall-clock scale.

| Column | Meaning |
|---|---|
| `Task ID` | Task this job belongs to. Any integer; need not be contiguous. |
| `Job ID` | **Globally unique, dense index.** The solver indexes internal arrays by this column, so it must run 0..N-1 across the whole file — not restart per task. |
| `Arrival min` | Earliest release time. |
| `Arrival max` | Latest release time. `Arrival max >= Arrival min`; equal values mean a known release. |
| `Cost min` | Best-case execution time (BCET), `>= 0`. |
| `Cost max` | Worst-case execution time (WCET). Must be `>= Cost min`. |
| `Deadline` | Absolute deadline of this job. |
| `Priority` | Smaller value = higher priority. |

```csv
Task ID, Job ID, Arrival min, Arrival max, Cost min, Cost max, Deadline, Priority
0,0,0,0,1,1,119,11
0,1,0,0,1,1,119,11
1,5,0,0,1,1,37,4
```

The `Job ID` requirement is the one that most often trips people up: a file whose
per-task job ids restart at 0 will be parsed, but the results will be wrong.
`tools/gen_taskset.py` always emits a dense global range.

## `jobsprec.csv` — one row per precedence edge

| Column | Meaning |
|---|---|
| `Predecessor TID` / `Predecessor JID` | The job that must complete first. |
| `Successor TID` / `Successor JID` | The job released by that completion. |
| `Sus min` | Minimum suspension between them. |
| `Sus max` | Maximum suspension. `>= Sus min`. |

```csv
Predecessor TID, Predecessor JID, Successor TID, Successor JID, Sus min, Sus max
0,0,0,1,0,0
7,35,7,36,2,3
```

Both endpoints must exist in `jobs.csv`. A non-zero `Sus max` models a
self-suspension: the successor becomes ready somewhere in
`[completion + Sus min, completion + Sus max]`. The shipped task sets chain each
task's segments this way.

## `*.rta.csv` — per-job response-time output

Written by both the solver (`.v5.rta.csv`) and by `nptest` (`.rta.csv`), with
identical columns, which is what makes them directly comparable.

| Column | Meaning |
|---|---|
| `Task ID`, `Job ID` | Identify the job. |
| `BCCT` / `WCCT` | Best- and worst-case **completion** time (absolute). |
| `BCRT` / `WCRT` | Best- and worst-case **response** time (relative to release). |

`WCRT` is the number the accuracy comparison uses. A sound analysis must report a
`WCRT` greater than or equal to the exact value; see README §9 for where this
solver does not.

## Solver verdicts on stdout

| Line | Meaning |
|---|---|
| `Schedulable: YES` | Every job provably meets its deadline. |
| `Schedulable: NO` | A deadline miss is reachable. |
| `Schedulable: TRUNCATED` | **No answer.** The state space exceeded the budget; the run stopped early and wrote no response times. Not a verdict — do not read it as `NO`. |
| `Controller wall: <ms>` | Solver-reported wall time, used as the timing measurement. |

`TRUNCATED` happens on `taskset_005` of the shipped dataset even on an A100 80GB
with 32 GB of host spill; see README §9.

## Environment variables

| Variable | Effect |
|---|---|
| `SAG_V5_VARIANT` | Selects the variant when `--variant` is not passed. |
| `SAG_V5_HOST_SPILL_GB` | Host-pinned spill budget in GB. **Defaults to 0 (off)**, in which case a layer that does not fit causes `TRUNCATED`. `scripts/preflight.sh` sets 32 (16 on small cards). |
| `SAG_VRAM_LIMIT_MB` | Caps device memory the solver will use. Useful for reproducing small-GPU behaviour on a large card. |
| `SAG_MAX_VRAM_PCT` | Caps device memory as a percentage of free VRAM. |
