// sag_internal.h -- shared internal types between expand_test_main.cu and
// the library wrapper sag_api.cu. NOT a public header (do not install).
//
// Iter129 (Phase D): exposes SolveResult so sag_api.cu can pass it to
// sag_run_one_taskset_export and translate fields into the public
// AnalyzeResult. Also exposes the global VRAM-fraction state so the library
// path can scope a temporary override.
#pragma once

#include "sag_api.h"
#include <cstdint>
#include <string>
#include <vector>

// Mirror of the SolveResult struct defined in expand_test_main.cu. Must stay
// byte-for-byte identical; the wrapper in sag_api.cu only reads it.
struct SolveResult {
    bool        ok = false;
    bool        schedulable = false;
    double      wall_time_s = 0.0;
    double      gpu_time_ms = 0.0;
    long long   total_nodes = 0;
    long long   total_expanded = 0;
    long long   total_merged = 0;
    int         jobs_tracked = 0;
    int         N = 0;
    int         W = 0;
    int         M = 0;
    std::string error;
    sag::Status status = sag::Status::INTERNAL_ERROR;
    std::vector<int32_t> bcrt;
    std::vector<int32_t> wcrt;
    int layers_processed = 0;
};

// Bridge from sag_api.cu to the static run_one_taskset in expand_test_main.cu.
// Defined in expand_test_main.cu.
extern "C" int sag_run_one_taskset_export(const char* jobs_csv_path,
                                          const char* prec_csv_path,
                                          const char* states_bin_path,
                                          int M_in,
                                          SolveResult* result_out);

// Globals declared in expand_test_main.cu. The library API sets them via
// guarded scoped overrides so caller-supplied AnalyzeOptions take effect.
namespace sag {
extern bool g_nvtx_runtime_enabled;
} // namespace sag
extern double g_max_vram_pct;
extern int    g_batch_par_n;
