#!/usr/bin/env python3
"""Run the SAGA GPU solver and the nptest CPU baseline on the same task sets and
report speedup plus per-job response-time agreement.

For each task set directory the harness:

  1. copies ``jobs.csv`` / ``jobsprec.csv`` into a scratch directory, so the
     shipped dataset is never modified by either tool;
  2. runs the GPU solver, recording its self-reported controller wall time;
  3. runs nptest, recording its self-reported CPU time;
  4. checks every per-job WCRT against ``jobs.rta.exact.csv``, the EXACT bounds
     shipped with the dataset (produced by ``nptest --merge=no``, i.e. with all
     state merging disabled).

The soundness gate is the hardware-independent claim and must hold on ANY CUDA
GPU: a response-time analysis may report a bound equal to or LARGER than the
exact one (sound, possibly pessimistic), but never SMALLER. Note that nptest's
default merge level is ``l1`` (lossy), so comparing against a default nptest run
is not a soundness test -- that is why the exact bounds are shipped separately.

The speedup is hardware-dependent and only matches the paper on an NVIDIA A100.

Both tools write their response-time output next to the copied jobs.csv, under
distinct names (``jobs.v5.rta.csv`` for the GPU, ``jobs.rta.csv`` for nptest),
so they never collide.

Exit status is 0 when every task set produced a matching verdict and no unsafe
WCRT deviation, 1 otherwise.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The GPU solver prints these two lines; we parse rather than wall-clock the
# process so CUDA context setup is attributed the same way the paper does.
SCHED_TOKEN = "Schedulable:"
WALL_TOKEN = "Controller wall:"


def read_rta(path):
    """Read a response-time CSV into {(task_id, job_id): (bcct, wcct, bcrt, wcrt)}.

    Both tools emit the same 6-column layout:
        Task ID, Job ID, BCCT, WCCT, BCRT, WCRT
    The header row is skipped by the int() conversion failing.
    """
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            try:
                key = (int(row[0]), int(row[1]))
                out[key] = tuple(int(row[i]) for i in range(2, 6))
            except ValueError:
                continue  # header line
    return out


def compare_rta(gpu, ref):
    """Bucket per-job WCRT differences between the GPU result and a reference.

    exact   : identical bound
    loose   : GPU bound is larger -- sound, merely pessimistic
    unsound : GPU bound is SMALLER than the exact bound -- a real defect
    """
    keys = sorted(set(gpu) & set(ref))
    exact = safe = unsafe = 0
    maxdev = 0
    for k in keys:
        g, r = gpu[k][3], ref[k][3]  # index 3 = WCRT
        if g == r:
            exact += 1
        elif g > r:
            safe += 1
        else:
            unsafe += 1
        maxdev = max(maxdev, abs(g - r))
    return {
        "jobs": len(keys),
        "exact": exact,
        "loose": safe,
        "unsound": unsafe,
        "maxdev": maxdev,
        "only_gpu": len(set(gpu) - set(ref)),
        "only_ref": len(set(ref) - set(gpu)),
    }


def run_gpu(binary, variant, cores, jobs, prec, wd, timeout, env):
    cmd = [str(binary), "-m", str(cores), str(jobs)] + ([str(prec)] if prec else [])
    env = dict(env, SAG_V5_VARIANT=variant)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=wd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"verdict": "TIMEOUT", "wall_ms": None, "rta": {}}

    (wd / "gpu_stdout.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)

    verdict, wall = "ERR", None
    for line in proc.stdout.splitlines():
        if SCHED_TOKEN in line:
            # The solver prints YES, NO, or TRUNCATED. TRUNCATED means it ran out
            # of state-space budget and produced NO ANSWER -- it must never be
            # collapsed into "NO", which would silently turn a non-result into a
            # definitive "unschedulable" verdict and a meaningless speedup.
            tail = line.split(SCHED_TOKEN, 1)[1].strip().upper()
            if tail.startswith("Y"):
                verdict = "YES"
            elif tail.startswith("N"):
                verdict = "NO"
            elif tail.startswith("TRUNC"):
                verdict = "TRUNCATED"
            else:
                verdict = tail.split()[0] if tail else "ERR"
        elif WALL_TOKEN in line:
            try:
                wall = float(line.split(WALL_TOKEN, 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    if wall is None:
        wall = (time.time() - t0) * 1000.0
    if proc.returncode != 0 and verdict == "ERR":
        sys.stderr.write(f"  GPU exited {proc.returncode}: {proc.stderr.strip()[:300]}\n")
    return {"verdict": verdict, "wall_ms": wall,
            "rta": read_rta(wd / (Path(jobs).stem + ".v5.rta.csv"))}


def run_nptest(binary, cores, jobs, prec, wd, timeout, env):
    # -r emits per-job response times, -c writes them to <stem>.rta.csv.
    cmd = [str(binary), "-m", str(cores), str(jobs)]
    if prec:
        cmd += ["-p", str(prec)]
    cmd += ["-r", "-c"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=wd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # The baseline was stopped at the harness cap. This is a CENSORED
        # measurement: the true nptest time is >= the cap, so the speedup
        # computed from it is a LOWER BOUND, not an equality.
        return {"verdict": "CAP", "cpu_s": float(timeout), "rta": {}, "capped": True}

    (wd / "nptest_stdout.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)

    # nptest stdout is one CSV line: file, sched(0/1), #jobs, ..., CPU time(col 7)
    verdict, cpu = "ERR", time.time() - t0
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1] in ("0", "1"):
            verdict = "YES" if parts[1] == "1" else "NO"
            try:
                cpu = float(parts[7])
            except (IndexError, ValueError):
                pass
            break
    return {"verdict": verdict, "cpu_s": cpu, "capped": False,
            "rta": read_rta(wd / (Path(jobs).stem + ".rta.csv"))}


def _write_csv(path, rows):
    """Write results after every task set.

    A full run can take hours because the baseline is exponential; flushing
    incrementally means an interrupted run still leaves usable measurements.
    """
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="directory containing taskset_* subdirectories")
    ap.add_argument("--cores", required=True, type=int, help="number of cores (m)")
    ap.add_argument("--gpu-bin", type=Path, help="path to expand_test_v5")
    ap.add_argument("--nptest-bin", type=Path, help="path to nptest")
    ap.add_argument("--variant", default="rtss24",
                    choices=["rtss24", "rtss17", "ecrts19", "ecrts22"],
                    help="SAG variant; the paper's baseline is rtss24 (default)")
    ap.add_argument("--work", type=Path, default=Path("work"),
                    help="scratch directory (default: ./work)")
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    ap.add_argument("--gpu-timeout", type=int, default=600)
    ap.add_argument("--nptest-timeout", type=int, default=10000,
                    help="harness cap in seconds (paper uses 10000)")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N task sets (0 = all)")
    ap.add_argument("--reference", type=Path,
                    help="reference_results JSON to print alongside this run")
    args = ap.parse_args()

    # Resolve every path to absolute. Both tools are launched with cwd set to the
    # per-task-set scratch directory, so a relative path (such as the default
    # --work "work") would be interpreted relative to THAT directory and the
    # tools would fail with "cannot open jobs file". Regression-tested in
    # tests/test_compare_units.py::TestPathResolution.
    args.dataset = args.dataset.resolve()
    args.work = args.work.resolve()
    args.out = args.out.resolve()
    if args.gpu_bin:
        args.gpu_bin = args.gpu_bin.resolve()
    if args.nptest_bin:
        args.nptest_bin = args.nptest_bin.resolve()
    if args.reference:
        args.reference = args.reference.resolve()

    tasksets = sorted(d for d in args.dataset.glob("taskset_*") if (d / "jobs.csv").exists())
    if args.limit:
        tasksets = tasksets[:args.limit]
    if not tasksets:
        sys.exit(f"ERROR: no taskset_*/jobs.csv found under {args.dataset}")

    for name, path in (("--gpu-bin", args.gpu_bin), ("--nptest-bin", args.nptest_bin)):
        if not path or not Path(path).exists():
            sys.exit(f"ERROR: {name} not found: {path}\n"
                     f"Build it first (scripts/build.sh, baseline/build_nptest.sh).")

    env = dict(os.environ)
    args.work.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'taskset':<14}{'GPU':>10}{'nptest':>12}{'speedup':>10}  verdict  WCRT vs exact bounds")
    print("-" * 78, flush=True)
    for ts in tasksets:
        wd = args.work / ts.name
        wd.mkdir(parents=True, exist_ok=True)
        jobs = wd / "jobs.csv"
        shutil.copy(ts / "jobs.csv", jobs)
        prec = None
        if (ts / "jobsprec.csv").exists():
            prec = wd / "jobsprec.csv"
            shutil.copy(ts / "jobsprec.csv", prec)

        # Carriage return only makes sense on a terminal; when the output is
        # redirected to a file the placeholder would be left in the log.
        if sys.stdout.isatty():
            print(f"  ... running {ts.name}", end="\r", flush=True)
        gpu = run_gpu(args.gpu_bin, args.variant, args.cores, jobs, prec, wd,
                      args.gpu_timeout, env)
        npt = run_nptest(args.nptest_bin, args.cores, jobs, prec, wd,
                         args.nptest_timeout, env)

        # Only the exact (unmerged) bounds generated by this artifact's own
        # pinned nptest build are trusted as the oracle. Task sets without one
        # are reported as "not compared" rather than silently checked against a
        # weaker or unknown-provenance reference.
        shipped = read_rta(ts / "jobs.rta.exact.csv")
        vs_ref = compare_rta(gpu["rta"], shipped) if (gpu["rta"] and shipped) else None
        vs_run = compare_rta(gpu["rta"], npt["rta"]) if (gpu["rta"] and npt["rta"]) else None

        # A GPU run that did not conclude carries no timing or verdict meaning.
        gpu_ok = gpu["verdict"] in ("YES", "NO")

        speedup = None
        if gpu_ok and gpu["wall_ms"] and npt["cpu_s"]:
            speedup = npt["cpu_s"] * 1000.0 / gpu["wall_ms"]

        capped = npt.get("capped", False)
        if not gpu_ok:
            match = False          # nothing to agree with
        elif capped:
            match = True           # baseline never finished; GPU did conclude
        else:
            match = gpu["verdict"] == npt["verdict"]
        if vs_ref:
            refstr = f"{vs_ref['exact']}/{vs_ref['jobs']} exact, {vs_ref['unsound']} unsound"
        elif not (ts / "jobs.rta.exact.csv").exists():
            refstr = "not compared (no exact reference)"
        else:
            refstr = "not compared (solver produced no bounds)"
        print(f"{ts.name:<14}{gpu['wall_ms']/1000.0 if gpu['wall_ms'] else float('nan'):>9.3f}s"
              f"{npt['cpu_s'] if npt['cpu_s'] else float('nan'):>11.3f}s"
              f"{('>=' if capped else '') + format(speedup, '.1f') if speedup else 'nan':>9}x"
              f"  {gpu['verdict'] if not gpu_ok else ('ok' if match else 'MISMATCH'):<10} {refstr}", flush=True)

        rows.append({
            "taskset": ts.name, "variant": args.variant, "cores": args.cores,
            "gpu_verdict": gpu["verdict"], "nptest_verdict": npt["verdict"],
            "verdict_match": match,
            "gpu_wall_s": round(gpu["wall_ms"] / 1000.0, 4) if gpu["wall_ms"] else None,
            "nptest_cpu_s": round(npt["cpu_s"], 4) if npt["cpu_s"] else None,
            "speedup": round(speedup, 2) if speedup else None,
            "nptest_capped": capped,
            "ref_jobs": vs_ref["jobs"] if vs_ref else 0,
            "ref_exact": vs_ref["exact"] if vs_ref else 0,
            "ref_loose": vs_ref["loose"] if vs_ref else 0,
            "ref_unsound": vs_ref["unsound"] if vs_ref else 0,
            "ref_maxdev": vs_ref["maxdev"] if vs_ref else 0,
            "run_exact": vs_run["exact"] if vs_run else 0,
            "run_unsound": vs_run["unsound"] if vs_run else 0,
        })

        _write_csv(args.out, rows)

    # ---- summary -------------------------------------------------------------
    ok = [r for r in rows if r["speedup"] and not r["nptest_capped"]]
    capped_rows = [r for r in rows if r["nptest_capped"]]
    speedups = sorted(r["speedup"] for r in ok)
    med = speedups[len(speedups) // 2] if speedups else None
    tot_jobs = sum(r["ref_jobs"] for r in rows)
    tot_exact = sum(r["ref_exact"] for r in rows)
    tot_unsound = sum(r["ref_unsound"] for r in rows)
    incomplete = [r["taskset"] for r in rows if r["gpu_verdict"] not in ("YES", "NO")]
    mismatches = [r["taskset"] for r in rows
                  if not r["verdict_match"] and r["taskset"] not in incomplete]

    print("-" * 78)
    print(f"task sets            : {len(rows)}")
    if capped_rows:
        print(f"baseline hit the cap : {len(capped_rows)} "
              f"({', '.join(r['taskset'] for r in capped_rows)}) -- "
              f"their speedups are LOWER BOUNDS and are excluded from the median")
    if speedups:
        print(f"speedup              : median {med:.1f}x   range {speedups[0]:.1f}x-{speedups[-1]:.1f}x")
    if incomplete:
        print(f"solver did not finish: {len(incomplete)} ({', '.join(incomplete)}) "
              f"-- excluded from speedup and verdict agreement")
    print(f"verdict mismatches   : {len(mismatches)}" + (f" {mismatches}" if mismatches else ""))
    print(f"WCRT vs exact bounds : {tot_exact}/{tot_jobs} exact, {tot_unsound} UNSOUND")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        # Record the dataset by name, not by absolute path: the summary is
        # shipped as reference data and must not carry the authors' filesystem.
        "dataset": args.dataset.name, "variant": args.variant, "cores": args.cores,
        "tasksets": len(rows), "tasksets_capped": len(capped_rows),
        "speedup_median": med,
        "speedup_min": speedups[0] if speedups else None,
        "speedup_max": speedups[-1] if speedups else None,
        "wcrt_jobs": tot_jobs, "wcrt_exact": tot_exact, "wcrt_unsound": tot_unsound,
        "verdict_mismatches": mismatches, "incomplete": incomplete,
    }
    Path(str(args.out).replace(".csv", ".json")).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} and {str(args.out).replace('.csv', '.json')}")

    if args.reference and args.reference.exists():
        ref = json.loads(args.reference.read_text())
        print("\n--- reference (authors' A100) vs this run ---")
        print(f"  speedup median : {ref.get('speedup_median')}x (ref)  vs  {med}x (yours)")
        print(f"  WCRT exact     : {ref.get('wcrt_exact')}/{ref.get('wcrt_jobs')} (ref)"
              f"  vs  {tot_exact}/{tot_jobs} (yours)")
        print("  Wall times depend on your GPU; the WCRT column should match exactly.")

    sys.exit(0 if (not mismatches and not incomplete and tot_unsound == 0) else 1)


if __name__ == "__main__":
    main()
