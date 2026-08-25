# Third-party components

## nptest (CPU baseline) -- not redistributed here

The baseline is `nptest` from the SAG-org `schedule_abstraction-main` repository.
It is **not** bundled with this artifact; `baseline/build_nptest.sh` clones it at
the pinned commit `47a45b69cbc90d0d0fd36d7c9ac8edaf041a6f05` and builds it locally.

* Upstream: https://github.com/SAG-org/schedule_abstraction-main
* License: BSD 3-Clause, Copyright (c) 2017-2021 Björn Brandenburg and contributors
* Transitively pulls in `yaml-cpp` (MIT) as a git submodule, used only when
  `yaml-cpp` is not already installed system-wide.

## Task-set data

`dataset/` contains synthetic task sets generated with the UUniFast-derived
generator described in the paper. The per-task-set `jobs.rta.*.csv` files are
reference outputs produced by `nptest` at the pinned commit above.

## GPU solver

`bin/saga` is part of this work and is covered by the top-level LICENSE. It is
distributed as a binary; its source is not included.

It statically links the NVIDIA CUDA runtime (`cudart_static`) and uses the CUB
primitives that ship inside the CUDA Toolkit, both under the NVIDIA CUDA Toolkit
End User License Agreement, which permits redistribution of the runtime as part of
an application. It also statically links GNU `libstdc++` and `libgcc`, distributed
under GPLv3 with the GCC Runtime Library Exception, which permits distribution of
the linked result under the licence of your choice.
