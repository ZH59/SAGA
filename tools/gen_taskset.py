#!/usr/bin/env python3
"""Generate task sets for the SAGA solver, so you can run it on your own inputs.

The solver takes two CSV files per task set (see INPUT_FORMAT.md for the full
schema). This script produces valid ones for a chosen size, core count, target
utilization, segment count and seed, letting you probe the solver beyond the task
sets shipped in dataset/.

Task model (matching the shipped task sets): every task is a chain of S segments
released together at time 0, with a suspension interval between consecutive
segments; deadlines are implicit (D = T) and priorities are deadline-monotonic.

    python3 tools/gen_taskset.py --tasks 20 --cores 5 --util 0.3 \
        --count 3 --seed 42 --out my_inputs

Then run the solver over them:

    python3 tools/compare.py --dataset my_inputs --cores 5 \
        --gpu-bin bin/saga --nptest-bin baseline/build/nptest --out my_results.csv

NOTE ON FIDELITY: this generator is provided so you can vary the input. It is
*not* the generator that produced dataset/ -- that one is not part of this
artifact, and because integer segment costs round upward, the nominal utilization
label of a task set does not translate directly into a measured one. Use this to
explore the solver's behaviour, not to reproduce the shipped task sets byte for
byte (those are shipped precisely so they need not be regenerated).
"""

import argparse
import csv
import json
import random
from pathlib import Path

# Periods are drawn from this ladder rather than a continuous range, so that
# hyperperiods stay manageable and deadlines spread over a useful dynamic range.
PERIOD_LADDER = [10, 15, 20, 25, 40, 50, 80, 100, 120, 160, 200, 250,
                 400, 500, 800, 1000]


def uunifast(n, total, rng):
    """Classic UUniFast: n utilizations, uniformly distributed, summing to total.

    Returns a list of n floats in (0, total). This is the standard algorithm from
    Bini & Buttazzo, "Measuring the performance of schedulability tests" (2005).
    """
    utils = []
    remaining = total
    for i in range(1, n):
        nxt = remaining * (rng.random() ** (1.0 / (n - i)))
        utils.append(remaining - nxt)
        remaining = nxt
    utils.append(remaining)
    return utils


def make_taskset(n, m, per_core_util, segments, beta, max_suspension, rng):
    """Build one task set as a list of per-task dicts.

    Each task gets a period from the ladder, a utilization from UUniFast, and a
    WCET split across `segments` sub-jobs of at least one tick each. Because each
    segment must cost >= 1 tick, small-utilization tasks are inflated upward --
    this is inherent to a discrete-time model, not a bug.
    """
    total = per_core_util * m
    utils = uunifast(n, total, rng)

    tasks = []
    for i in range(n):
        period = rng.choice(PERIOD_LADDER)
        wcet = max(segments, int(round(utils[i] * period)))  # >= 1 tick/segment

        # Split the WCET across segments: start with an even split, then move a
        # few ticks around so segments are not all identical.
        base = wcet // segments
        costs = [max(1, base)] * segments
        for _ in range(wcet - sum(costs)):
            costs[rng.randrange(segments)] += 1

        segs = []
        for c_max in costs:
            c_min = max(1, int(round(beta * c_max)))
            segs.append((c_min, c_max))

        tasks.append({
            "period": period,
            "deadline": period,            # implicit deadline
            "segments": segs,
            "suspensions": [(0, rng.randint(0, max_suspension))
                            for _ in range(segments - 1)],
        })

    # Deadline-monotonic priorities: 1 is highest. Ties broken by task index so
    # the assignment is deterministic for a given seed.
    order = sorted(range(n), key=lambda i: (tasks[i]["deadline"], i))
    for rank, i in enumerate(order, start=1):
        tasks[i]["priority"] = rank
    return tasks


def write_taskset(out_dir, tasks):
    """Write jobs.csv and jobsprec.csv. Job IDs are dense and globally unique."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = 0
    ids = []           # ids[task][segment] = global job id
    with open(out_dir / "jobs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Task ID", "Job ID", "Arrival min", "Arrival max",
                    "Cost min", "Cost max", "Deadline", "Priority"])
        for t, task in enumerate(tasks):
            row_ids = []
            for (c_min, c_max) in task["segments"]:
                w.writerow([t, job_id, 0, 0, c_min, c_max,
                            task["deadline"], task["priority"]])
                row_ids.append(job_id)
                job_id += 1
            ids.append(row_ids)

    with open(out_dir / "jobsprec.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Predecessor TID", "Predecessor JID",
                    "Successor TID", "Successor JID", "Sus min", "Sus max"])
        for t, task in enumerate(tasks):
            for k, (s_min, s_max) in enumerate(task["suspensions"]):
                w.writerow([t, ids[t][k], t, ids[t][k + 1], s_min, s_max])
    return job_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, default=20, help="tasks per set (n)")
    ap.add_argument("--cores", type=int, default=5, help="cores (m)")
    ap.add_argument("--util", type=float, default=0.3,
                    help="target utilization PER CORE (0.3 = 30%%)")
    ap.add_argument("--segments", type=int, default=5, help="segments per task (S)")
    ap.add_argument("--beta", type=float, default=0.5, help="BCET/WCET ratio")
    ap.add_argument("--max-suspension", type=int, default=5,
                    help="upper bound on the suspension between segments")
    ap.add_argument("--count", type=int, default=1, help="how many task sets")
    ap.add_argument("--seed", type=int, default=42, help="base random seed")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()

    if args.segments < 1:
        raise SystemExit("--segments must be >= 1")
    if args.tasks < 1 or args.cores < 1:
        raise SystemExit("--tasks and --cores must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(args.count):
        # One fresh, reproducible seed per task set.
        rng = random.Random(args.seed + k)
        tasks = make_taskset(args.tasks, args.cores, args.util, args.segments,
                             args.beta, args.max_suspension, rng)
        ts_dir = args.out / f"taskset_{k:03d}"
        njobs = write_taskset(ts_dir, tasks)
        (ts_dir / "manifest.json").write_text(json.dumps({
            "tasks": args.tasks, "cores": args.cores,
            "target_util_per_core": args.util, "segments": args.segments,
            "beta": args.beta, "max_suspension": args.max_suspension,
            "seed": args.seed + k, "jobs": njobs,
            "generator": "tools/gen_taskset.py",
        }, indent=2))
        print(f"wrote {ts_dir} ({args.tasks} tasks, {njobs} jobs)")

    print(f"\nRun the solver on them with:\n"
          f"  python3 tools/compare.py --dataset {args.out} --cores {args.cores} \\\n"
          f"      --gpu-bin bin/saga --nptest-bin baseline/build/nptest \\\n"
          f"      --out my_results.csv\n"
          f"(no exact reference ships with generated inputs, so the WCRT column\n"
          f" will read \"not compared\"; the verdict and timing columns still apply)")


if __name__ == "__main__":
    main()
