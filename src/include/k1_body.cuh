// k1_body.cuh -- K1 (FusedEligibility) body extracted as a __device__ inline
// function so it can be called both from the standalone FusedEligibilityKernel
// in fused_eligibility.cu and from the fused K1+K2 cooperative kernel in
// successor_creation.cu (iter210). The body is templated on warps-per-block
// to support both K1's native 4-warp blocks and the fused kernel's 12-warp
// blocks (matching K2's K2_WARPS_PER_BLOCK).
//
// The shared-memory layout grows linearly with WPB:
//   s_rho_max       [WPB * n] int32
//   s_in_ready      [WPB * n] bool
//   s_priority      [n]       int32
//   s_prio_order    [n]       int32
//   s_rho_min_below [WPB * n] int32
//
// Caller must pass:
//   - smem_base: base of dynamic shmem (extern __shared__)
//   - threadIdx_x within [0..32)  (lane)
//   - warpInBlock within [0..WPB) (warp index in block, the state slot)
//   - tid_in_block in [0..WPB*32) for the cooperative priority load
//   - stateIdx: global state index this warp owns (or -1 to skip per-warp work
//     while still participating in the block-wide priority load + barrier)

#pragma once
#include "sag_config.h"
#include "sag_types.h"

namespace sag {
namespace k1 {

using namespace sag::config;

__device__ __forceinline__ int ctz64_k1(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ bool bit_test_k1(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

__device__ __forceinline__ bool is_subset_k1(const uint64_t* mask,
                                             const uint64_t* set, int W) {
    for (int w = 0; w < W; w++) {
        if (mask[w] & ~set[w]) return false;
    }
    return true;
}

__device__ __forceinline__ int32_t warp_min_k1(int32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = min(val, other);
    }
    return val;
}

// Templated K1 body: WPB = warps per block (state slots). The shared memory
// layout is parametrized on WPB and n via the smem_base pointer.
//
// Pre-condition: caller supplied smem_base points at a region of at least
//   2*WPB*n*int32 + WPB*n*bool + 2*n*int32 bytes. (rho_max, rho_min_below,
//   priority, prio_order, in_ready)
template<int WPB>
__device__ __forceinline__ void K1Body(
    char* smem_base,
    const char* __restrict__     d_input,
    SAGStateLayout               layout,
    const int* __restrict__      d_candidates,
    int num_states,
    int num_candidates,
    const uint64_t* __restrict__ d_TC,
    const uint64_t* __restrict__ d_PO,
    const uint64_t* __restrict__ d_Pred,
    const int32_t* __restrict__  d_r_min,
    const int32_t* __restrict__  d_r_max,
    const int32_t* __restrict__  d_priority,
    const int32_t* __restrict__  d_prio_order,
    const int32_t* __restrict__  d_sus_min,
    const int32_t* __restrict__  d_sus_max,
    int n, int W,
    ValidPair* __restrict__      d_valid_pairs,
    int* __restrict__            d_valid_count,
    int* __restrict__            d_unschedulable_flag,
    int max_valid_pairs,
    int* __restrict__            d_trunc_flag,
    int                          stateIdx,        // global state index (or -1)
    int                          lane,            // 0..31
    int                          warpInBlock,     // 0..WPB-1
    int                          tid_in_block)    // 0..WPB*32-1
{
    // Shared memory layout (matches fused_eligibility.cu's STATES_PER_BLOCK
    // version but with WPB substituted in for state-count slots).
    int32_t* s_rho_max       = (int32_t*)smem_base;
    bool*    s_in_ready      = (bool*)(smem_base + WPB * n * sizeof(int32_t));
    int32_t* s_priority      = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool));
    int32_t* s_prio_order    = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t));
    int32_t* s_rho_min_below = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + n * sizeof(int32_t));

    // Block-level cooperative priority + prio_order load.
    for (int t = tid_in_block; t < n; t += WPB * WARP_SIZE) {
        s_priority[t]   = d_priority[t];
        s_prio_order[t] = d_prio_order[t];
    }
    // Per-warp init of s_rho_max / s_in_ready.
    for (int j = lane; j < n; j += WARP_SIZE) {
        s_rho_max[warpInBlock * n + j]  = INF_TIME;
        s_in_ready[warpInBlock * n + j] = false;
    }
    __syncthreads();

    if (stateIdx < 0 || stateIdx >= num_states) return;

    const uint64_t* st_D      = layout.D(d_input, stateIdx);
    const uint64_t* st_F_mask = layout.F_mask(d_input, stateIdx);
    auto st_A_min  = layout.A_min(d_input, stateIdx);
    auto st_A_max  = layout.A_max(d_input, stateIdx);
    auto st_F_min  = layout.F_min(d_input, stateIdx);
    auto st_F_max  = layout.F_max(d_input, stateIdx);

    // Sub-phase 1
    #pragma unroll 2
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int c = d_candidates[ci];
        if (bit_test_k1(st_D, c)) continue;
        const uint64_t* preds = &d_Pred[c * W];
        if (!is_subset_k1(preds, st_D, W)) continue;
        int32_t rho_max_c = d_r_max[c];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds[w] & st_F_mask[w];
            while (pbits) {
                int bit = ctz64_k1(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                rho_max_c = max(rho_max_c, st_F_max[i] + d_sus_max[c * n + i]);
            }
        }
        s_rho_max[warpInBlock * n + c]  = rho_max_c;
        s_in_ready[warpInBlock * n + c] = true;
    }
    __syncwarp(0xFFFFFFFF);

    // Iter114 prefix-min over priority order
    if (lane == 0) {
        int32_t running_min = INF_TIME;
        int32_t pending_min = INF_TIME;
        bool have_prio = false;
        int last_prio = 0;
        for (int k = 0; k < n; k++) {
            int j = s_prio_order[k];
            int p = s_priority[j];
            if (!have_prio || p != last_prio) {
                running_min = min(running_min, pending_min);
                pending_min = INF_TIME;
                last_prio = p;
                have_prio = true;
            }
            s_rho_min_below[warpInBlock * n + j] = running_min;
            pending_min = min(pending_min, s_rho_max[warpInBlock * n + j]);
        }
    }
    __syncwarp(0xFFFFFFFF);

    int32_t local_rho_any = INF_TIME;
    for (int j = lane; j < n; j += WARP_SIZE) {
        if (s_in_ready[warpInBlock * n + j])
            local_rho_any = min(local_rho_any, s_rho_max[warpInBlock * n + j]);
    }
    int32_t rho_any = warp_min_k1(local_rho_any);

    // Sub-phase 2
    #pragma unroll 2
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int j = d_candidates[ci];
        if (!s_in_ready[warpInBlock * n + j]) continue;

        const uint64_t* tc_j = &d_TC[j * W];
        const uint64_t* po_j = &d_PO[j * W];
        bool guard_ok = true;
        for (int w = 0; w < W; w++) {
            if (((tc_j[w] | po_j[w]) & ~st_D[w]) != 0) { guard_ok = false; break; }
        }
        if (!guard_ok) continue;

        int32_t rho_min_j = d_r_min[j];
        const uint64_t* preds_j = &d_Pred[j * W];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds_j[w] & st_F_mask[w];
            while (pbits) {
                int bit = ctz64_k1(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                rho_min_j = max(rho_min_j, st_F_min[i] + d_sus_min[j * n + i]);
            }
        }
        int32_t rho_hp_j = s_rho_min_below[warpInBlock * n + j];
        int32_t s_min_j = max(rho_min_j, st_A_min[0]);
        int32_t s_max_j = min(rho_hp_j - 1, max(st_A_max[0], rho_any));

        if (s_min_j <= s_max_j) {
            int idx = atomicAdd(d_valid_count, 1);
            if (idx < max_valid_pairs) {
                d_valid_pairs[idx].state_idx = stateIdx;
                d_valid_pairs[idx].job_j     = j;
                d_valid_pairs[idx].s_min     = s_min_j;
                d_valid_pairs[idx].s_max     = s_max_j;
            } else {
                atomicExch(d_trunc_flag, 1);
            }
        }
    }
}

} // namespace k1
} // namespace sag
