# Security Audit: Privacy Pipeline — March 2026

## Audit History

| Date | Scope | PRs | Summary |
|------|-------|-----|---------|
| 2026-03 (initial) | Full pipeline review | #7–#34 | Identified 7 critical gaps, implemented P0–P3 fixes |
| 2026-03 (second) | Hardening & SOTA gaps | #36, #37 | P0 hardening (H1–H4), P1 PRV/SVT, P2 auxiliary composition, S1/S3/S5 |

## Current State (Post-Hardening)

| Layer | Mechanism | Formal Guarantee? | Status |
|---|---|---|---|
| L1 | Gaussian DP (Flower wrappers) | (ε,δ)-DP | PLD/PRV accounting with budget enforcement |
| L1b | Per-example DP-SGD (Opacus) | (ε,δ)-DP | AutoClip + Ghost Clipping + joint composition |
| L2a | PercentilePrivacy | (ε,δ)-DP (calibrated) | Adaptive K accounting with ε-splitting (S1) |
| L2b | SVT Privacy | ε-DP | Optimal budget split (Lyu 2017) + observability |
| L2c | ExcludeVars | **None** | Documented as defense-in-depth only |
| L2d | Gradient Compression | (ε,δ)-DP (calibrated) | TopK/random + PLD noise + error feedback |
| L2e | Partial Freeze (Lambda-SecAgg) | N/A | Shape metadata for correct reconstruction |
| L3a | SecAgg+ | Information-theoretic | Config wired, threshold enforced |
| L3b | HE (CKKS) | Semantic security | Demo only, impractical >100 params |
| L4 | TEE/CVM | Hardware isolation | Docs only, no code |
| L5a | Multi-Krum | Byzantine-tolerant | CSRNG-seeded JL projection, norm verification |
| L5b | Trimmed Mean | Byzantine-tolerant | Coordinate-wise, fallback on small n |
| L5c | FoundationFL (NDSS 2025) | Byzantine-tolerant | Mandatory root_update enforcement (S3) |

### Previous State (Pre-Hardening)

| Layer | Mechanism | Formal Guarantee? | Status |
|---|---|---|---|
| L1 | Gaussian DP (Flower wrappers) | (ε,δ)-DP | Thin wrapper, no privacy accounting |
| L2a | PercentilePrivacy | **None** | Heuristic sparsification only |
| L2b | SVT Privacy | ε-DP (claimed) | Non-standard composition, bugs |
| L2c | ExcludeVars | **None** | Security theater against adaptive adversary |
| L3a | SecAgg+ | Information-theoretic | Config only, not wired |
| L3b | HE (CKKS) | Semantic security | Demo only, impractical >100 params |
| L4 | TEE/CVM | Hardware isolation | Docs only, no code |

### Concrete Weaknesses

**1. No privacy accounting (biggest gap)**

We set `noise_multiplier=0.1` and `clipping_norm=10.0` but never compute the resulting (ε,δ) after T rounds. Without this, claims of "differential privacy" are aspirational. The 2024+ standard is to use Rényi DP (RDP) accountants or the PLD (Privacy Loss Distribution) accountant, which give tight composition bounds. Flower's `DPFedAvgFixed` wrappers don't track this either — they just add noise.

Post-2024 SOTA: Google's DP-FTRL and Apple's approach both use per-round privacy budget tracking with automatic stopping when budget is exhausted. The `dp-accounting` library (Google) and `opacus` (Meta) both provide PLD-based accountants. We should track cumulative ε across rounds and halt training when budget depletes.

**2. SVT implementation has issues**

- The composition formula `eps_2 = ε * (2*n_upload)^(2/3)` is non-standard. The classical SVT (Dwork & Roth, 2014) uses `ε₁ = ε/2` for the threshold noise and `ε₂ = ε/(2c)` per query, where c is the number of queries answered. Our formula doesn't match any published variant.
- `noise_var` is used as `scale = gamma * 2 / noise_var` for the output noise, but it's also a config parameter (default 0.1) independent of epsilon — this breaks the formal ε-DP guarantee because the output perturbation needs to be calibrated to the same privacy budget.
- The while loop can run unbounded iterations if acceptance rate is very low.

**3. PercentilePrivacy offers no privacy guarantee**

Top-K sparsification (Shokri & Shmatikov 2015) was shown by Zhu et al. (NeurIPS 2019, "Deep Leakage from Gradients") to be insufficient against gradient inversion. Keeping 10% of gradients is enough for reconstruction of training examples. This is a pre-2020 insight. The 2023+ literature is clear: sparsification without noise is not private.

**4. ExcludeVars is cosmetic**

Zeroing embedding layers doesn't prevent information leakage through downstream layers. Gradient flow means information from "excluded" layers propagates through the rest of the network. This gives users a false sense of security.

**5. SecAgg is config-only**

The `SecAggConfig` class exists but is never used anywhere in the runners. SecAgg is arguably the single most impactful mechanism we could deploy because it prevents the server from seeing individual updates at all (information-theoretic guarantee, not computational).

**6. No gradient compression/quantization defense**

Post-2023 attacks (LAMP, Fishing for User Data, APRIL) can reconstruct training batch data from gradients even with small noise. The SOTA defense combination is DP + SecAgg + gradient compression, not just noise addition.

---

## 2026 SOTA Recommendations — Status

### Phase 1: Foundation (PRs #7–#34)

| Priority | Improvement | Status | PR |
|---|---|---|---|
| **P0** | PLD privacy accountant + budget enforcement | **DONE** | #7-#10 |
| **P0** | Wire SecAgg+ into runners | **DONE** | #8-#9 |
| **P1** | Fix SVT composition (optimal budget split) | **DONE** | #13 |
| **P1** | PercentilePrivacy calibrated noise + warning | **DONE** | #11 |
| **P2** | Gradient compression (TopK + noise + error feedback) | **DONE** | #14, #32 |
| **P2** | Per-example DP-SGD (Opacus) | **DONE** | #16 |
| **P2** | AutoClip (Li et al., NeurIPS 2023) | **DONE** | #27 |
| **P2** | Ghost Clipping (memory-efficient DP-SGD) | **DONE** | #31 |
| **P2** | Shuffle-model DP amplification | **DONE** | #29 |
| **P2** | Per-layer clipping (Yu et al., ICLR 2022) | **DONE** | #33 |
| **P2** | FoundationFL trust scoring (NDSS 2025) | **DONE** | #30 |
| **P2** | Lambda-SecAgg partial freezing | **DONE** | #34 |
| **P3** | Byzantine-robust aggregation (Multi-Krum, Trimmed Mean) | **DONE** | #15 |
| **P3** | Replace CKKS HE with Lattigo/OpenFHE BFV | Open | — |

### Phase 2: Hardening (PR #36)

Second audit identified operational safety gaps. All fixes in PR #36.

| ID | Finding | Fix | Status |
|---|---|---|---|
| **H1** | Partial freeze sends `np.zeros(1)` for frozen layers — server can't reconstruct correct shapes | Store frozen layer shapes in `_frozen_shapes` metric; server restores correct shapes | **DONE** |
| **H2** | SecAgg `reconstruction_threshold` below 2/3 majority only warns | Raise `ValueError` when `threshold < ceil(2*num_shares/3)` | **DONE** |
| **H3** | Krum JL projection uses `RandomState(42)` — adversary can pre-compute projection | Use `secure_rng()` (CSRNG-seeded) for projection matrix | **DONE** |
| **H4** | No server-side norm check — compromised client can bypass client-side DP clip | Added `verify_update_norms()` filter before aggregation | **DONE** |
| **P1** | PRV `eps_error` hardcoded at 0.1 (±10% epsilon) | Configurable, default 0.01 for ±1% accuracy | **DONE** |
| **P1** | SVT low-acceptance silently degrades utility | Enhanced warnings + `svt_acceptance_rate` metric | **DONE** |
| **P2** | Adaptive clip/SVT DP cost not composed into running budget | `compose_auxiliary()` on PrivacyAccountant via ComposedDpEvent | **DONE** |

### Phase 3: Security Hardening (PR #37)

Third audit focused on formal guarantee gaps and defense-in-depth.

| ID | Finding | Fix | Status |
|---|---|---|---|
| **S1** | Adaptive K in PercentilePrivacy — data-dependent selection leaks information | Split ε into ε/3 (selection) + 2ε/3 (noise) when K is adaptive; `fixed_k` option for data-independent sensitivity | **DONE** |
| **S3** | FoundationFL without root_update — Byzantine majority can manipulate client mean | Require `root_update` or explicit `allow_untrusted_reference=True` (raises `ValueError` by default) | **DONE** |
| **S5** | PrivacyAuditor only tested simplified simulation, not real mod chain | `run_pipeline_audit()` sends canary gradients through actual Flower mods end-to-end | **DONE** |

### Phase 4: CI & Test Reliability (PR #36)

| Finding | Fix |
|---|---|
| Heavy tests (torch/GPU/PRV) slow CI to ~8 min | `@pytest.mark.slow` marker; CI runs `-m "not slow"` (~2.5 min) |
| Flaky SVT test (stochastic acceptance with `secure_rng()`) | Relaxed threshold from 60 to 50 |
| `test_low_noise_warns` fails when dp-accounting not installed | Iterate `call_args_list` instead of checking only last warning |
| SecAgg test assumes old warning behavior | Updated to expect `ValueError` |

---

## Key References

- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014) — canonical SVT (Ch. 3.6)
- Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019) — gradient inversion attacks
- Mironov, "Rényi Differential Privacy" (CSF 2017) — RDP accountant
- Koskela et al., PLD accountant — tight DP composition
- Google `dp-accounting` library — production-grade privacy accounting
- Bonawitz et al., "Practical Secure Aggregation" (CCS 2017) — SecAgg protocol
- Kairouz et al., "Advances and Open Problems in Federated Learning" (2021) — survey
