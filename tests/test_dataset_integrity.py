#!/usr/bin/env python3
"""System-level test: the shipped task sets are well-formed and self-consistent.

This is a known-solution test in the sense the RTSS criteria describe: the
expected structure of every dataset file is stated here, so a failure points at
a concrete corrupted or missing input rather than at the solver.

Run:  python3 tests/test_dataset_integrity.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_HEADER = ["Task ID", "Job ID", "Arrival min", "Arrival max",
               "Cost min", "Cost max", "Deadline", "Priority"]
PREC_HEADER = ["Predecessor TID", "Predecessor JID",
               "Successor TID", "Successor JID"]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL {msg}")


def check_taskset(ts):
    jobs_p, prec_p = ts / "jobs.csv", ts / "jobsprec.csv"
    if not jobs_p.exists():
        return fail(f"{ts}: jobs.csv missing")

    rows = list(csv.reader(open(jobs_p)))
    header = [h.strip() for h in rows[0]]
    if header != JOBS_HEADER:
        return fail(f"{ts}: unexpected jobs.csv header {header}")

    ids = set()
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != 8:
            return fail(f"{ts}: jobs.csv line {i} has {len(r)} fields, expected 8")
        try:
            tid, jid = int(r[0]), int(r[1])
            amin, amax, cmin, cmax, dl = (int(r[2]), int(r[3]), int(r[4]),
                                          int(r[5]), int(r[6]))
        except ValueError:
            return fail(f"{ts}: jobs.csv line {i} has non-integer fields")
        if (tid, jid) in ids:
            return fail(f"{ts}: duplicate job id ({tid},{jid})")
        ids.add((tid, jid))
        # Invariants any valid job must satisfy.
        if amin > amax:
            return fail(f"{ts}: job ({tid},{jid}) arrival min > max")
        if cmin > cmax:
            return fail(f"{ts}: job ({tid},{jid}) cost min > max")
        if dl <= 0:
            return fail(f"{ts}: job ({tid},{jid}) non-positive deadline")

    if prec_p.exists():
        prows = list(csv.reader(open(prec_p)))
        ph = [h.strip() for h in prows[0]][:4]
        if ph != PREC_HEADER:
            return fail(f"{ts}: unexpected jobsprec.csv header {ph}")
        for i, r in enumerate(prows[1:], start=2):
            if len(r) < 4:
                return fail(f"{ts}: jobsprec.csv line {i} has {len(r)} fields")
            try:
                pred = (int(r[0]), int(r[1]))
                succ = (int(r[2]), int(r[3]))
            except ValueError:
                return fail(f"{ts}: jobsprec.csv line {i} non-integer")
            # Every edge endpoint must be a job that exists in jobs.csv.
            for e, what in ((pred, "predecessor"), (succ, "successor")):
                if e not in ids:
                    return fail(f"{ts}: edge {what} {e} not present in jobs.csv")

    # Reference outputs must cover exactly the jobs in the input.
    for ref_name in ("jobs.rta.exact.csv", "jobs.rta.nptest.csv"):
        ref = ts / ref_name
        if not ref.exists():
            continue
        got = set()
        for r in csv.reader(open(ref)):
            if len(r) < 6:
                continue
            try:
                got.add((int(r[0]), int(r[1])))
            except ValueError:
                continue
        if got != ids:
            return fail(f"{ts}: {ref_name} covers {len(got)} jobs, input has {len(ids)}")
    return None


def main():
    tasksets = sorted(ROOT.glob("dataset/*/taskset_*"))
    if not tasksets:
        print("FAIL: no task sets found under dataset/")
        return 1
    print(f"checking {len(tasksets)} task sets...")
    for ts in tasksets:
        check_taskset(ts)
    if failures:
        print(f"\n{len(failures)} dataset problem(s)")
        return 1
    print(f"  ok   all {len(tasksets)} task sets well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
