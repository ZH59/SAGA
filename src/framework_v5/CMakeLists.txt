# framework_v5/CMakeLists.txt -- bootstrap build for the GPU-resident SAG
# solver introduced in iter300.
#
# Side-by-side with the legacy build:
#   - Legacy: cmake --build build -j16 produces build/expand_test
#   - V5:     cmake -S framework_v5 -B build_v5 ... && cmake --build build_v5 -j16
#             produces build_v5/expand_test_v5
#
# Generic across A/H/B-series NV GPUs: emit native SASS for sm_80 (Ampere/A100),
# sm_90 (Hopper/H100), and sm_100 (Blackwell/B200) when the toolchain supports
# them. Falls back gracefully to whatever subset the current CUDA version can
# build, with PTX for the highest emitted arch as a JIT-fallback for newer GPUs.
# The cooperative-launch + this_grid().sync() APIs used here are SM_70+.
# No Hopper-only features (TMA, wgmma, DSM/cluster-shared-mem, FP8) are used.

cmake_minimum_required(VERSION 3.18)
project(framework_v5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Multi-arch: probe what nvcc supports and assemble the strongest possible
# native SASS list. Industrial users get zero JIT cost on first run when the
# build host's CUDA toolkit can target their deployment GPU.
#
# RTSS 2026 AE note: the candidate list starts at 70, not 80, so the artifact
# also runs natively on the GPUs an evaluator is most likely to own (V100 =
# sm_70, T4 / RTX 20xx = sm_75). PTX only JITs *forward*, so an sm_80-and-up
# build would fail on those cards with "no kernel image is available for
# execution on the device". sm_60 is deliberately excluded: the cooperative
# this_grid().sync() in controller.cu requires SM_70+. Cubins are compatible
# within a major generation (z >= y for sm_Xy), so sm_80 covers 8.6/8.7/8.9
# and sm_90 covers 9.x -- 70/75/80/90/100 spans every current NVIDIA GPU.
#
# CUDA 12.5 supports up to compute_90 (Hopper). compute_100 (Blackwell) is
# added in CUDA 12.8+. We probe each candidate via nvcc --list-gpu-arch and
# include only the supported ones. The final list is converted to CMake's
# CUDA_ARCHITECTURES form so that PTX for the highest arch is also emitted
# (CMake appends the virtual arch automatically when given e.g. "90", which
# is sufficient for forward-compat JIT to unsupported newer GPUs).
execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
    OUTPUT_VARIABLE _v5_nvcc_arch_list
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(_v5_arch_candidates "70" "75" "80" "90" "100")
set(_v5_arch_emitted "")
foreach(_arch IN LISTS _v5_arch_candidates)
    if(_v5_nvcc_arch_list MATCHES "compute_${_arch}")
        list(APPEND _v5_arch_emitted "${_arch}")
    endif()
endforeach()

# Safety net: if the probe somehow returns nothing (e.g. older CMake/nvcc
# combination without --list-gpu-arch), fall back to sm_80 alone.
if(NOT _v5_arch_emitted)
    set(_v5_arch_emitted "80")
endif()

# NOTE on PTX / forward compatibility: this target uses relocatable device code
# (CUDA_SEPARABLE_COMPILATION below, required so cg::this_grid() resolves across
# translation units). nvlink resolves device symbols at build time and emits
# cubins only, so no JIT-able PTX survives into the executable -- adding a
# "-virtual" architecture here has no effect. Verified with cuobjdump --list-ptx.
#
# That is fine for artifact evaluation because the artifact is built FROM SOURCE
# on the evaluator's own machine: the probe above selects the architectures that
# THEIR toolkit supports. An evaluator on Blackwell simply needs CUDA 12.8+, at
# which point "100" is picked up automatically. scripts/preflight.sh checks the
# built binary against the detected GPU and reports this explicitly rather than
# letting the run die with "no kernel image is available for execution".
set(CMAKE_CUDA_ARCHITECTURES ${_v5_arch_emitted})
message(STATUS "framework_v5: emitting native SASS for sm_${_v5_arch_emitted}")

find_package(CUDAToolkit REQUIRED)

# Pull in the legacy include/ directory (sag_config.h, sag_types.h,
# k1_body.cuh) plus our own framework_v5/include/ (k2_body_v5.cuh).
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(expand_test_v5
    src/controller.cu
    src/framework_v5_main.cu
)

target_link_libraries(expand_test_v5 PRIVATE CUDA::cudart_static)
target_link_options(expand_test_v5 PRIVATE -static-libstdc++ -static-libgcc)

target_compile_options(expand_test_v5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-extended-lambda --default-stream=per-thread>
)

# Cooperative launch requires relocatable device code so cg::this_grid() can
# resolve across TUs (controller.cu invokes K1Body inline; relocatable lets
# the linker fold the cross-TU __device__ references safely).
set_target_properties(expand_test_v5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
