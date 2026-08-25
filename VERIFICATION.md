# Verification performed on this package

Every check below was run on the reference machine
(`reference_results/a100/ENVIRONMENT.md`) against the committed tree.

## Portability to non-A100 hardware

Reviewers run this on their own GPUs, so the degradation paths were exercised
explicitly rather than assumed.

All rows below were run with the **shipped** `bin/saga`, on `taskset_000`.

| Situation | How it was simulated | Result |
|---|---|---|
| glibc older than the binary needs | `ldd` shimmed to report 2.31 | Preflight refuses with the required version and a list of qualifying distributions. |
| No NVIDIA driver at all | `PATH` restricted to a sandbox without `nvidia-smi` | Preflight explains and returns 1; `run_smoke.sh` exits **2**. No crash. |
| GPU with 8 GiB VRAM | `SAG_VRAM_LIMIT_MB=8000` | Completes, 107.1 ms vs 98.0 ms unrestricted. |
| GPU with 4 GiB VRAM | `SAG_VRAM_LIMIT_MB=4000`, spill 16 GB | Completes, 122.0 ms. |
| GPU with 2 GiB VRAM | `SAG_VRAM_LIMIT_MB=2000`, spill 16 GB | Completes, 148.7 ms. |
| Response times under memory pressure | compared all four runs above | **Byte-identical.** Memory pressure changes speed, not results. |
| GPU the binary has no code for | `tests/test_arch_compat.sh`, 12 rule cases + 2 manifest checks | Accepts an sm_80 cubin on 8.6/8.9, rejects Blackwell (10.x) and RTX 50xx (12.0), rejects sm_80-only builds on 7.x. The test sources `tools/preflight.sh` and calls its own `saga_arch_compatible`, so it exercises the shipped implementation rather than a copy. |
| Same check on a machine **without** the CUDA Toolkit | `PATH` restricted to a sandbox with a driver but no `cuobjdump` | Still works, reading `bin/saga.arch`. This is the normal evaluator case; an earlier cuobjdump-only version of this check would have silently skipped. |
| Malformed `nvidia-smi` output | fields blanked | Preflight reports one clear parse error instead of cascading into "compute capability ," and "no GPU code for sm_". |
| Binary missing | ran `run_smoke.sh` without the baseline built | Clear error naming the missing binary, exit 1. |

**Scope limit, stated explicitly:** the memory-pressure rows cover one task set.
They show the spill path works and is result-preserving; they are *not* evidence
that a small GPU can complete the whole dataset. `taskset_005` returns `TRUNCATED`
on the full 80 GiB card with 32 GB of spill, and a smaller card will truncate on
more task sets, not fewer.

## The shipped binary

* `cuobjdump --list-elf bin/saga` confirms native SASS for **sm_70, sm_75, sm_80
  and sm_90** — not just the A100's sm_80.
* No JIT-able PTX survives, because the target uses relocatable device code and
  resolves device symbols at link time, so Blackwell needs a newer toolkit rather
  than a PTX fallback. Stated in README §8 and enforced by preflight.
* `ldd bin/saga` lists only `libc` and the loader: the CUDA runtime, libstdc++ and
  libgcc are statically linked, so no toolchain is needed on the target machine.
* `file bin/saga` reports **stripped**; `readelf -S` shows **0** debug sections.
* **glibc floor measured, not assumed.** `objdump -T bin/saga` shows the highest
  required symbol version is `GLIBC_2.34` (from `dlopen`), so the binary needs
  glibc >= 2.34. `tools/preflight.sh` enforces this.
* **No network I/O.** `strace -f -e trace=network` over a complete run recorded
  **zero** network syscalls. `readelf --dyn-syms` shows the only libc function it
  imports is `dlopen`; `socket`, `connect`, `getaddrinfo` and friends are absent
  from the dynamic symbol table.

### Leakage audit (binary-only distribution)

Counted with `strings -a bin/saga | grep -c ...`:

| Pattern | Occurrences |
|---|---:|
| `/home/zhaoz26` (author paths) | **0** |
| `/data1` (build-host paths) | **0** |
| original `.cu` translation-unit names | **0** |
| private header names (`k1_body`, `k2_body`, `merge_body`, `ijp_detect`, `por_detect`) | **0** |
| internal plan/status document names | **0** |

The binary was rebuilt for distribution without `-lineinfo` (which had embedded
68 source filenames and 41 absolute paths) and from a neutral directory with
renamed translation units.

**Exactly what still appears**, so the table above is not read as "nothing":

| String | Why it is there | Assessment |
|---|---|---|
| `/tmp/nb/s/m.cu` | `nvcc` records the translation-unit path for device-code registration. The build was done in a throwaway directory with the unit renamed, so this is the path it recorded. | Reveals no author path and no real filename. |
| `framework_v5/VARIANTS.md` | Printed by `--help` as a documentation pointer. | A path in the solver's build tree; the document ships here as `VARIANTS.md`. Noted in README §7. |
| `framework_v5 (iter380: ...)` | Startup/version banner. | Internal version string. Reveals component vocabulary, no source. |

No other source filenames, header names, author paths or internal document names
remain; the table above was produced with `strings -a bin/saga`.

### The shipped binary was checked against the one that produced the results

The measurements in README §5 were taken with a dynamically linked, unstripped
build. `bin/saga` is a different compilation, so equivalence was checked rather
than assumed. Across all 10 shipped task sets:

* **identical verdicts** (including `TRUNCATED` on `taskset_005`),
* **byte-identical** `*.v5.rta.csv` response-time output on every task set that
  produced any,
* wall times within run-to-run noise (largest case 39.81 s vs 39.92 s, 0.3%).

## Package hygiene

Checked on the extracted archive, not on the working tree:

* No author or build-host paths appear in any shipped file **except this one**,
  which necessarily quotes the patterns it searched for. Excluding
  `VERIFICATION.md`, `grep -rl '/home/zhaoz26\|/data1\|nerds09'` returns **0 files**.
* No solver source ships: `find . -name '*.cu' -o -name '*.cuh'` returns **0**.
* `tests/run_tests.sh` → all tests pass in the extracted archive, no GPU required.
* `bin/saga` retains its executable bit through the archive, and `./bin/saga --help`
  runs from the extracted copy.
* The archive is produced with `git archive` from the `rtss26-ae` tag, then two
  deliberate edits: `AE_INSTRUCTIONS.pdf` is added and `.gitignore` is removed.
  Nothing else is included — no scratch directories, no build trees, no history.

## Correctness of the reference data

* This package's freshly built `nptest` reproduces the shipped
  `jobs.rta.nptest.csv` reference **byte-for-byte (80/80 jobs)** on the smoke
  task set, confirming the baseline build and invocation are right.
* Exact bounds (`nptest --merge=no`) are shipped for **6 of the 10** task sets:
  000, 001, 002, 004, 006, 008. On **all six**, the exact output is
  **byte-identical** to nptest's default `l1` output, so the four task sets that
  fall back to the default-merge reference (003, 005, 007, 009) are not being
  checked against a weaker oracle in practice. 003 and 005 are omitted because
  both exhaust the 300 s cap even under the cheaper default merge level, so an
  unmerged run cannot terminate in reasonable time.
* `taskset_008`, the one carrying the unsound deviation, **does** have exact
  bounds, so that finding rests on the strict oracle rather than the fallback.
* The solver is deterministic: repeated runs report the same wall time to four
  decimals and byte-identical response times.

## Tests

`tests/run_tests.sh` — 7 Python unit tests, 10 architecture-rule cases, and a
dataset integrity system test over all shipped task sets. All pass, no GPU needed.

## Packaging decisions

* **Binary, not source.** The solver ships prebuilt; see README §10 for what that
  does and does not conceal, and for the consequences under the RTSS criteria.
* **No container image.** The static binary already removes the dependency problem
  a container would solve, and an untested image would add a failure mode without
  adding a capability.
* **The baseline is built from upstream source, not shipped as a binary.** That
  lets you verify the baseline is the genuine `nptest` at the pinned commit rather
  than taking the authors' word for it.
