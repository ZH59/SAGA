# Solver variants

`bin/saga` implements four schedule-abstraction-graph analyses from the SAG
literature, selected with `--variant` (or `SAG_V5_VARIANT`):

| `--variant` | Task model | Source of the model |
|---|---|---|
| `rtss24` **(default)** | Limited-preemptive self-suspending, event-driven delay-induced; global multiprocessor | Srinivasan, Günzel, Nelissen — RTSS 2024 |
| `rtss17` | Exact uniprocessor non-preemptive | Nasri, Brandenburg — RTSS 2017 |
| `ecrts19` | Non-preemptive tasks with precedence constraints (DAG) | Nasri, Nelissen, Brandenburg — ECRTS 2019 |
| `ecrts22` | Non-preemptive moldable gang | Nelissen, Marcè i Igual, Nasri — ECRTS 2022 |

## What this artifact covers

**Only `rtss24`.** Every measurement in `README.md` §5, every task set in
`dataset/`, and every reference bound in `dataset/*/jobs.rta.exact.csv` uses the
`rtss24` variant, because that is the analysis the paper compares against — the
EDD analyzer of Srinivasan et al., the most computationally demanding member of
the SAG lineage.

The other three variants are present in the binary and selectable, but this
artifact makes **no claim** about their accuracy or performance and ships no
reference data for them. If you run them, treat the output as unvalidated: the
input conventions differ per model (for example `rtss17` is uniprocessor, so it
expects `-m 1`, and `ecrts22` expects a gang cost-map format in the cost columns),
and a mismatch between the model and the input will produce a wrong answer rather
than an error.

## Input conventions

`rtss24` — the configuration this artifact uses — expects the CSV layout described
in `INPUT_FORMAT.md`: one row per job with a dense global `Job ID`, and precedence
edges carrying suspension intervals. Run it as:

```bash
bin/saga --variant rtss24 -m <CORES> jobs.csv jobsprec.csv
```
