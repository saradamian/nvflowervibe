# Security Audit: Privacy Pipeline — March 2026

## Current State vs. 2026 SOTA

### What We Have (Honest Assessment)

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

## 2026 SOTA Recommendations (Ranked by Impact/Effort)

| Priority | Improvement | Effort | Impact |
|---|---|---|---|
| **P0** | Add RDP/PLD privacy accountant | Medium | Turns hand-wavy DP into provable DP |
| **P0** | Wire SecAgg+ into runners | Low | Server can't see individual updates |
| **P1** | Fix SVT composition | Medium | Current impl may not be ε-DP at all |
| **P1** | Remove or warn about PercentilePrivacy | Low | Prevents false sense of security |
| **P2** | Add gradient compression (TopK + noise) | Medium | Reduces attack surface + bandwidth |
| **P2** | User-level DP (DP-SGD per-example) | High | Strongest formal guarantee per user |
| **P3** | Replace CKKS HE with Lattigo/OpenFHE BFV for integer aggregation | High | More practical than CKKS for FL |
| **P3** | Byzantine-robust aggregation (Multi-Krum, Trimmed Mean) | Medium | Defends against poisoning attacks |

---

## Key References

- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014) — canonical SVT (Ch. 3.6)
- Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019) — gradient inversion attacks
- Mironov, "Rényi Differential Privacy" (CSF 2017) — RDP accountant
- Koskela et al., PLD accountant — tight DP composition
- Google `dp-accounting` library — production-grade privacy accounting
- Bonawitz et al., "Practical Secure Aggregation" (CCS 2017) — SecAgg protocol
- Kairouz et al., "Advances and Open Problems in Federated Learning" (2021) — survey
