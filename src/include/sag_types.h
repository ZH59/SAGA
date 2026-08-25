#pragma once
#include "sag_config.h"
#include <cstdint>
#include <type_traits>

namespace sag {

// Compute the number of uint64_t words needed for n bits
inline __host__ __device__ int bitset_words(int n) {
    return (n + 63) / 64;
}

// ---------------------------------------------------------------------------
// Iter44: F_min and F_max are stored INTERLEAVED as int2 pairs
//   [F_min[0], F_max[0], F_min[1], F_max[1], ...]
// keeping (F_min[j], F_max[j]) in 8 contiguous bytes (one cache-line span).
// Same total bytes as before (n × int2 = 2*n × int32). The merge kernel hot
// loops read the pair in a single LDG.E.64 via F_pair() instead of two
// scattered LDG.E.32 across 224-byte-apart cache lines (n=28 case).
//
// FStridedView<IsConst> is a backward-compatible view returned by F_min /
// F_max accessors so existing call sites that did
//   const int32_t* sFmin = layout.F_min(buf, idx);
//   ... sFmin[J] ...
// keep working with one tweak: change `const int32_t*` to `auto`. The
// wrapper's operator[] does the stride-2 hop into the int2 storage. Hot loops
// in merge_kernels.cu were further rewritten to use F_pair() directly so
// NVCC fuses the (F_min[J], F_max[J]) read into a single int2 load.
// ---------------------------------------------------------------------------

template<bool IsConst>
struct FStridedView {
    using PtrT = typename std::conditional<IsConst, const int32_t*, int32_t*>::type;
    using RefT = typename std::conditional<IsConst, int32_t, int32_t&>::type;
    PtrT base;  // points to first int32 of the field (x or y of pair 0)

    __host__ __device__ FStridedView(PtrT p) : base(p) {}
    __host__ __device__ RefT operator[](int j) const { return base[2 * j]; }
};

using FViewMut   = FStridedView<false>;
using FViewConst = FStridedView<true>;

// Layout descriptor for dynamically-sized SAG states in flat memory.
//
// Each state occupies a contiguous block of bytes:
//   D[W]  | X[W]  | F_mask[W]  | A_min[m] | A_max[m] | F_pair[n] | ovf (int32)
//
// where W = bitset_words(n) = ceil(n/64) and F_pair[j] = int2{ x = F_min[j],
// y = F_max[j] }.
struct SAGStateLayout {
    int W;  // bitset words = ceil(n/64)
    int n;  // number of jobs
    int m;  // number of cores

    __host__ __device__ int bytes_per_state() const {
        int raw = 3 * W * (int)sizeof(uint64_t)  // D, X, F_mask
                + 2 * m * (int)sizeof(int32_t)    // A_min, A_max
                + 2 * n * (int)sizeof(int32_t)    // F_pair (n × int2)
                + (int)sizeof(int32_t);           // ovf
        // Round up to 8-byte alignment so uint64_t D[] of the next state is aligned
        return (raw + 7) & ~7;
    }

    __host__ __device__ uint64_t* D(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state());
    }
    __host__ __device__ const uint64_t* D(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state());
    }

    __host__ __device__ uint64_t* X(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* X(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }

    __host__ __device__ uint64_t* F_mask(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* F_mask(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }

    // Iter45: A_min and A_max are stored INTERLEAVED as int2 pairs in A_pair[m]
    // (same pattern as F_pair from iter44). dev_check_range_compatible cond 2
    // reads (a_Amin[i], a_Amax[i]) and (b_Amin[i], b_Amax[i]) within each
    // state for the i-loop over m cores; interleaving fuses each pair's read
    // into one LDG.E.64. Total bytes per state UNCHANGED (m × int2 still =
    // 2*m × int32).
    __host__ __device__ int2* A_pair(char* base, int state_idx) const {
        return (int2*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }
    __host__ __device__ const int2* A_pair(const char* base, int state_idx) const {
        return (const int2*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }

    __host__ __device__ FViewMut A_min(char* base, int state_idx) const {
        int2* p = A_pair(base, state_idx);
        return FViewMut(reinterpret_cast<int32_t*>(p));
    }
    __host__ __device__ FViewConst A_min(const char* base, int state_idx) const {
        const int2* p = A_pair(base, state_idx);
        return FViewConst(reinterpret_cast<const int32_t*>(p));
    }
    __host__ __device__ FViewMut A_max(char* base, int state_idx) const {
        int2* p = A_pair(base, state_idx);
        return FViewMut(reinterpret_cast<int32_t*>(p) + 1);
    }
    __host__ __device__ FViewConst A_max(const char* base, int state_idx) const {
        const int2* p = A_pair(base, state_idx);
        return FViewConst(reinterpret_cast<const int32_t*>(p) + 1);
    }

    // Raw int2-typed access to the F_pair array (use this for vectorized
    // 8-byte loads/stores covering both F_min and F_max in one transaction).
    __host__ __device__ int2* F_pair(char* base, int state_idx) const {
        return (int2*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t));
    }
    __host__ __device__ const int2* F_pair(const char* base, int state_idx) const {
        return (const int2*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t));
    }

    // Strided int32 views over F_pair: F_min(buf, idx)[j] reads x of pair j,
    // F_max reads y. Returns a wrapper struct (use `auto` at call sites).
    __host__ __device__ FViewMut F_min(char* base, int state_idx) const {
        int2* p = F_pair(base, state_idx);
        return FViewMut(reinterpret_cast<int32_t*>(p));
    }
    __host__ __device__ FViewConst F_min(const char* base, int state_idx) const {
        const int2* p = F_pair(base, state_idx);
        return FViewConst(reinterpret_cast<const int32_t*>(p));
    }
    __host__ __device__ FViewMut F_max(char* base, int state_idx) const {
        int2* p = F_pair(base, state_idx);
        return FViewMut(reinterpret_cast<int32_t*>(p) + 1);
    }
    __host__ __device__ FViewConst F_max(const char* base, int state_idx) const {
        const int2* p = F_pair(base, state_idx);
        return FViewConst(reinterpret_cast<const int32_t*>(p) + 1);
    }

    __host__ __device__ int32_t* ovf(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t)
               + 2 * n * (int)sizeof(int32_t));
    }
    __host__ __device__ const int32_t* ovf(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t)
               + 2 * n * (int)sizeof(int32_t));
    }
};

struct ValidPair {
    int     state_idx;
    int     job_j;
    int32_t s_min;
    int32_t s_max;
};

// iter180: per-state IJP detection result. Populated by IJPDetectKernel
// after K1 and read by K2 (via state_idx) to decide whether to emit a
// multi-job IJP successor or a single-job successor.
//
// has_ijp: 1 if a non-trivial IJP set (k ≥ 2) was detected for this state.
// k:       size of J' (≥ 2 when has_ijp).
// mask:    bitmask of J' (job j ∈ J' iff (mask >> j) & 1). Limited to N ≤ 64.
// t_disp:  worst-case dispatch time = max over J' of s_max(j; slot k-1).
// leader:  the smallest job index in J' (the "leader" pair in the K1 ValidPair
//          stream; only the leader executes the IJP path in K2).
struct IJPInfo {
    uint64_t mask;
    int32_t  t_disp;
    int32_t  leader;
    int32_t  k;          // |J'|
    int32_t  has_ijp;    // 0 or 1
};

} // namespace sag
