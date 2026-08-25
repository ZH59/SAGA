# This is the SOURCE variant of the artifact

Read this before `README.md`.

`README.md` was written for the binary-only distribution, where the solver ships
prebuilt as `bin/saga`. **This package ships the solver's source instead**, so two
things differ:

1. **You must build the solver first.** There is no `bin/saga` here.

   ```bash
   ./scripts/build.sh          # -> build/expand_test_v5
   ./baseline/build_nptest.sh  # -> baseline/build/nptest
   ```

   `scripts/build.sh` finds a CUDA 12.x toolkit automatically (override with
   `CUDACXX=/path/to/nvcc`) and prints the GPU architectures it embedded. It emits
   native code for sm_70/75/80/90 — and sm_100 as well if your toolkit is CUDA
   12.8 or newer, which is how a Blackwell GPU is supported here but not in the
   binary distribution.

2. **Point the run scripts at what you built:**

   ```bash
   GPU_BIN=build/expand_test_v5 ./run_smoke.sh
   GPU_BIN=build/expand_test_v5 ./run_full.sh
   ```

   Or copy it into place once: `mkdir -p bin && cp build/expand_test_v5 bin/saga`,
   after which `README.md` applies verbatim.

Because you are building from source, the glibc floor discussed in `README.md`
section 2 and the architecture manifest (`bin/saga.arch`) described in section 10
do not apply: your own toolchain determines both.

Everything else in `README.md` — the claims, the measured results, the input
format, the limitations, the safety definitions — applies unchanged, since the
source here is the source the shipped binary was built from.

## Layout added by this variant

```
src/framework_v5/   the GPU solver (CUDA C++17): src/, include/, CMakeLists.txt
src/include/        shared SAG headers the solver includes
scripts/build.sh    multi-architecture build with CUDA autodetection
```
