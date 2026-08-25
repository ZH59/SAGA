// sag_api.h -- Public C++ library API for the GPU SAG analyzer.
//
// This header is intentionally pure-C++17 (no CUDA types in the public surface)
// so external tools can link against the library without depending on the
// CUDA toolkit at compile time. Implementation lives in src/sag_api.cu.
//
// Usage (host code):
//
//   #include "sag_api.h"
//   sag::AnalyzeOptions opts;
//   opts.m = 4;
//   opts.nvtx_trace = true;
//   sag::AnalyzeResult r = sag::analyze(jobs, prec, opts);
//   if (r.status == sag::Status::SCHED) { ... }
//
// Iter129 (Phase D, industrial-readiness scaffolding).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sag {

// One job (single instance / dispatch).
//   id        : opaque identifier (will become the row index in result vectors)
//   r_min/r_max: release-time interval
//   c_min/c_max: WCET / BCET interval
//   deadline  : absolute deadline
//   priority  : higher value = higher priority (matches the existing CSV
//               convention used by parse_jobs_csv)
struct Job {
    int     id;
    int32_t r_min;
    int32_t r_max;
    int32_t c_min;
    int32_t c_max;
    int32_t deadline;
    int32_t priority;
};

// One precedence edge (src must finish before dst can be released).
//   sus_min/sus_max: optional finish-to-release suspension interval
struct PrecEdge {
    int     src;
    int     dst;
    int32_t sus_min;
    int32_t sus_max;
};

// Caller-tunable solver knobs. Defaults reproduce the current CLI behavior.
struct AnalyzeOptions {
    // Fraction of free GPU memory to use as the working set (0,1].
    // Industrial deployments can lower this when the GPU is shared. 1.0
    // matches the post-iter112 behavior.
    double max_vram_pct  = 1.0;

    // Reserved for a future deterministic-merge mode. Currently a no-op.
    bool   deterministic = false;

    // Emit NVTX ranges around K1 / K2 / Merge / sort-stream / layer
    // transitions when the binary was compiled with SAG_NVTX_TRACE.
    bool   nvtx_trace    = false;

    // Number of cores in the target system (must be >= 1, <= MAX_CORES).
    int    m             = 1;

    // Number of OpenMP workers for the parallel batch path. Set to 1 for
    // a single-taskset call.
    int    par_workers   = 1;
};

// Top-level outcome. Distinguishes a clean SCHED/UNSCHED verdict from
// out-of-resource conditions where the analyzer couldn't finish.
enum class Status {
    SCHED,              // Found schedulable; bcrt/wcrt are sound bounds
    UNSCHED,            // Found an UNSCHED witness during state expansion
    INCOMPLETE_OOM,     // Ran out of GPU memory before completion
    INCOMPLETE_TRUNC,   // Buffer overflow truncated the SAG (e.g. wave too small)
    INCOMPLETE_TIMEOUT, // External timeout fired (reserved; not yet detectable)
    INTERNAL_ERROR      // Unexpected internal failure (CUDA error, bad input, ...)
};

// Per-call result. bcrt/wcrt are populated only for successful runs
// (Status::SCHED or Status::UNSCHED — for UNSCHED they are partial bounds
// covering jobs reached before the witness was hit).
struct AnalyzeResult {
    Status               status        = Status::INTERNAL_ERROR;
    std::string          error_msg;

    std::vector<int32_t> bcrt;          // size N
    std::vector<int32_t> wcrt;          // size N

    double               wall_seconds   = 0.0;
    long long            states_explored = 0;  // total SAG nodes visited
    long long            states_merged   = 0;  // post-merge state count
    int                  layers_processed = 0;
};

// Synchronous solve. Blocks until the analysis terminates. Thread-safe to
// call concurrently across CPU threads provided each call gets its own
// AnalyzeOptions instance and they share at most max_vram_pct of the GPU
// budget across the par_workers axis (the implementation handles partition).
AnalyzeResult analyze(const std::vector<Job>&      jobs,
                      const std::vector<PrecEdge>& prec,
                      const AnalyzeOptions&        opts);

} // namespace sag
