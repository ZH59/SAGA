// por_detect_v5.cuh -- Partial-Order Reduction (POR) detection for V5.
//
// This is the Phase B SCAFFOLD per `framework_v5/PHASE_POR_PLAN.md`.
//
// Phase A (already shipped): SAG_V5_POR=1 env var, status banner, plan doc.
// Phase B (THIS FILE): PORInfoV5 struct + skeleton PORDetectBodyV5 that
//   evaluates the conservative P1+P3'+P4+P5+P6 predicate from RTAS 2022
//   (Ranjha/Nelissen/Nasri). Per memory `iter150_por_basic_zero_fires.md`,
//   the basic predicate fires 0% on hyperperiod-expanded periodic
//   (chain-DAG) workloads — Phase B is plumbing-only verification.
// Phase C (FUTURE): drop P6 cap + Algorithm 4 greedy interferer
//   absorption. THIS is where the 40-70% state-count cut lands.
// Phase D (FUTURE): multicore EFT/LFT envelope.
// Phase E (FUTURE): IJP↔POR layering (POR fills states where IJP didn't).
//
// === Current scaffold semantics ===
// PORDetectBodyV5 sets d_por_info[state_idx].has_por = 0 unconditionally.
// Verdicts byte-identical to no-POR (predicate fires 0%). The full
// predicate body is left as a TODO comment block to be filled in by
// Phase C.
//
// Mirrors `ijp_detect_v5.cuh` structure. Reuses the IJP-style
// per-state detection result struct + lane-0 sequential scan idiom.
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include <cstdio>
#include "sag_config.h"
#include "sag_types_v5.h"

namespace sag {
namespace v5 {
namespace por {

// Phase C POR telemetry. Defined in controller.cu inside this same
// namespace. Gated by g_v5_por_telemetry_enabled (default 0); when 1,
// atomicAdd to g_v5_por_fire_count per state where the predicate sets
// has_por=1.
extern __device__ int g_v5_por_telemetry_enabled;
extern __device__ unsigned long long g_v5_por_fire_count;

using namespace sag::config;

// Compile-time bounds. POR_V5_MAX_K bounds the cluster J_S size; in
// Phase B (P6 = |ES| <= m), bounded by m at runtime, so 32 covers up
// to m=32 cleanly.
constexpr int POR_V5_MAX_K    = 32;
constexpr int POR_V5_INF_TIME = INT32_MAX / 2;

// Per-state POR detection result. Mirrors IJPInfoV5 layout to allow K2
// to dispatch through the same multi-job emit path.
//
// has_por: 0 = no POR firing, normal per-job K1 emission
//          1 = POR fired; leader pair emits multi-job successor
// k:        |J_S| (cluster size)
// leader:   smallest job_idx in J_S (lowest bit of mask)
// t_disp:   max LST over J_S members (worst-case dispatch time)
// mask:     bitmap of J_S members
// member_s_max[t]: per-member tentative s_max for the t-th set bit of mask
// member_s_min[t]: per-member tentative s_min for the t-th set bit of mask
// es_max_smax: max s_max across ES (used for paper Lemma 5 closure check)
struct PORInfoV5 {
    int32_t  has_por;
    int32_t  k;
    int32_t  leader;
    int32_t  t_disp;
    int32_t  es_max_smax;
    uint64_t mask[MAX_BITSET_WORDS];
    int32_t  member_s_max[POR_V5_MAX_K];
    int32_t  member_s_min[POR_V5_MAX_K];
};

// Phase B SCAFFOLD body. Per `iter150_por_basic_zero_fires.md`, the
// conservative P1+P3'+P4+P5+P6 predicate fires 0% on chain-DAG
// hyperperiod workloads. Phase C will drop P6 + add greedy absorption.
//
// Current implementation: set has_por = 0 unconditionally. Verdicts
// remain byte-identical to no-POR.
//
// === Phase C TODO (16-20h) ===
// Implement the actual predicate:
//
// P1: |ES| >= 2 (multiple eligible candidates; otherwise POR is vacuous)
// P6: |ES| <= m (Phase C drops this; Phase B keeps it conservative)
// P4: s_max_j == max(A_pair[0].y, rho_any) for all j ∈ ES
//     (all eligible jobs share the same worst-case start time)
// P5: f_max_j = s_max_j + C_max(j) <= d_j (each member is schedulable)
// P3' (paper Lemma 5): strict closure -- for every j' ∉ ES with
//     pred(j') ⊆ D ∪ ES, j' must also be in ES (Phase C drops by
//     greedy absorption: keep growing J_S as long as Eq. 5 holds).
//
// Algorithm 4 (Phase C):
//     J_S := ES
//     env_top := max f_max over J_S
//     for each c with r_min(c) <= env_top:
//         if absorbing c into J_S preserves L1 (predec-ready) and
//            L2 (no-starve in extended LFT envelope):
//             J_S := J_S ∪ {c}; recompute env_top
//     if |J_S| >= 2: emit POR-marker
//
// (Telemetry symbols are defined at global scope in controller.cu;
// declared via the file-scope extern below.)

// Phase C predicate body — DESIGN ATTEMPT.
//
// First implementation walked d_valid_pairs[0..num_valid) per state to
// recover the eligible set (ES). Per-warp cost: O(num_valid) reads.
// With wave_count states each running this, total work is
// O(num_states × num_valid) = quadratic. On n24/m6/u20 with peak
// wave_count ~100K and num_valid ~500K, single launch wall blew up to
// >200s (kernel timed out).
//
// === Required design change ===
// PORDetectBody MUST have per-state ES indexing. Two options:
//  (A) K1 emits d_valid_pairs sorted by state_idx + per-state
//      offset/count array → POR scans only its state's range
//      (O(|ES|) per state; total O(num_valid)).
//  (B) Move POR detection INTO K1 sub-phase 2 (where ES is in shmem
//      via s_in_ready[]; per-candidate s_max/f_max already computed
//      per-pair in K1). Phase A scaffold did this for P1 only via
//      `g_por_dbg[]`; full predicate would extend that path.
//
// (B) is more invasive (K1 modification) but follows the iter150
// worktree's working POR detection pattern. (A) is simpler at the
// cost of a sort.
//
// For now, the body remains the SKELETON setting has_por=0 (Phase B).
// Phase C predicate body is deferred until we implement option (A)
// or (B). The plumbing (kernel wrapper, kargs, dispatch) is fully
// wired; only the predicate body is still scaffold.
//
// Mirrors `IJPDetectBodyV5` in `ijp_detect_v5.cuh:105+`.
__device__ __forceinline__ void PORDetectBodyV5(
    const char* __restrict__     /*d_input*/,
    SAGStateLayoutV5             /*layout*/,
    const int* __restrict__      /*d_candidates*/,
    int                          num_states,
    int                          /*num_candidates*/,
    const uint64_t* __restrict__ /*d_TC*/,
    const uint64_t* __restrict__ /*d_PO*/,
    const uint64_t* __restrict__ /*d_Pred*/,
    const int32_t*  __restrict__ /*d_r_min*/,
    const int32_t*  __restrict__ /*d_r_max*/,
    const int32_t*  __restrict__ /*d_priority*/,
    const int32_t*  __restrict__ /*d_C_max*/,
    const int32_t*  __restrict__ /*d_deadline*/,
    const int32_t*  __restrict__ /*d_sus_min*/,
    const int32_t*  __restrict__ /*d_sus_max*/,
    int /*n*/, int /*m*/, int /*W*/,
    const ValidPair* __restrict__ /*d_valid_pairs*/,
    int                          /*num_valid*/,
    PORInfoV5* __restrict__      d_por_info,
    int                          stateIdx,
    int                          lane)
{
    if (stateIdx < 0 || stateIdx >= num_states) return;

    // Skeleton: has_por=0 unconditionally. Verdicts byte-identical to
    // no-POR. See header design notes for predicate-body deferral.
    if (lane == 0) {
        d_por_info[stateIdx].has_por = 0;
    }
}

} // namespace por
} // namespace v5
} // namespace sag
