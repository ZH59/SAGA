// sag_nvtx.h -- compile-time-gated NVTX hooks.
//
// Define SAG_NVTX_TRACE in the build flags to inline nvtxRangePush/Pop
// at every SAG_NVTX_PUSH / SAG_NVTX_POP site. Without the define, every
// call expands to a no-op so release builds carry zero overhead.
//
// The runtime g_nvtx_runtime_enabled flag adds a second gate (set by the
// CLI --nvtx flag or AnalyzeOptions.nvtx_trace) so a single binary can
// be deployed with NVTX instrumentation built in but off-by-default.
#pragma once

#ifdef SAG_NVTX_TRACE
  #include <nvToolsExt.h>
#endif

namespace sag {
// Runtime gate. Lives in src/expand_test_main.cu (one definition).
// The compile-time gate above eliminates the load entirely when
// instrumentation is disabled at build time.
extern bool g_nvtx_runtime_enabled;
} // namespace sag

#ifdef SAG_NVTX_TRACE
  #define SAG_NVTX_PUSH(name) \
      do { if (sag::g_nvtx_runtime_enabled) nvtxRangePushA(name); } while (0)
  #define SAG_NVTX_POP() \
      do { if (sag::g_nvtx_runtime_enabled) nvtxRangePop(); } while (0)
#else
  #define SAG_NVTX_PUSH(name) ((void)0)
  #define SAG_NVTX_POP()      ((void)0)
#endif

// Scope guard so we don't leak a push when the function early-returns.
namespace sag {
struct NvtxScope {
    bool active;
    explicit NvtxScope(const char* name) : active(false) {
        (void)name;
#ifdef SAG_NVTX_TRACE
        if (g_nvtx_runtime_enabled) {
            nvtxRangePushA(name);
            active = true;
        }
#endif
    }
    ~NvtxScope() {
#ifdef SAG_NVTX_TRACE
        if (active) nvtxRangePop();
#endif
    }
    NvtxScope(const NvtxScope&)            = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;
};
} // namespace sag

#define SAG_NVTX_SCOPE(name) ::sag::NvtxScope _sag_nvtx_scope_##__LINE__(name)
