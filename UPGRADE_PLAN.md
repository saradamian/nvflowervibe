# Privacy Pipeline Upgrade Plan
## Goal: Bring nvflowervibe to 2026 SOTA Federated Learning Privacy

---

## Phase 1 — Privacy Accountant (P0, highest impact)
**Branch:** `feature/privacy-accountant`

### Problem
`wrap_strategy_with_dp()` in `src/sfl/privacy/dp.py` adds Gaussian noise via Flower's
`DifferentialPrivacyServerSideFixedClipping` / `DifferentialPrivacyClientSideFixedClipping`
but never computes the resulting (ε,δ) after T rounds. Users have no idea what privacy
level they're actually getting.

### What to Build

#### 1.1 New file: `src/sfl/privacy/accountant.py`
- Add `dp-accounting` (Google) as an optional dependency in `pyproject.toml`:
  ```
  [project.optional-dependencies]
  privacy = ["dp-accounting>=0.4.0"]
  ```
- Implement `PrivacyAccountant` class:
  ```python
  class PrivacyAccountant:
      def __init__(self, noise_multiplier: float, sample_rate: float,
                   delta: float = 1e-5, max_epsilon: float = 10.0):
          ...
      def step(self) -> float:
          """Called after each round. Returns current cumulative epsilon."""
      @property
      def epsilon(self) -> float:
          """Current (ε,δ)-DP guarantee."""
      @property
      def budget_exhausted(self) -> bool:
          """True if epsilon >= max_epsilon."""
  ```
- Use `dp_accounting.pld.PLDAccountant` (Privacy Loss Distribution) for tight composition.
  PLD is strictly tighter than RDP for Gaussian mechanism (Koskela et al.).
- `sample_rate = num_sampled_clients / total_clients` (Poisson subsampling amplification).

#### 1.2 Integrate into `DPConfig`
In `src/sfl/privacy/dp.py`, add fields to `DPConfig`:
```python
@dataclass
class DPConfig:
    ...
    target_delta: float = 1e-5      # δ in (ε,δ)-DP
    max_epsilon: float = 10.0       # Stop training when ε exceeds this
    num_total_clients: int = 2      # For subsampling amplification
```

#### 1.3 Wire into `wrap_strategy_with_dp()`
- Create `PrivacyAccountant` inside the wrapper.
- Return both the wrapped strategy and the accountant as a tuple:
  ```python
  def wrap_strategy_with_dp(strategy, dp_config) -> tuple[Strategy, PrivacyAccountant]:
  ```
- Alternatively, make the accountant accessible as an attribute on the returned strategy
  wrapper (less API breakage).

#### 1.4 Per-round logging
In `src/sfl/server/strategy.py` or via a Flower server-side callback (if Flower 1.17
supports it), call `accountant.step()` after each aggregation round and log:
```
Round 3/10: ε = 1.42 (δ = 1e-5) | budget remaining: 8.58
```
If `budget_exhausted`, log a WARNING and optionally halt training early.

#### 1.5 CLI flags
Both runners (`flower_runner.py`, `esm2_runner.py`):
```
--dp-delta 1e-5         # target delta
--dp-max-epsilon 10.0   # budget cap (stop training if exceeded)
```

#### 1.6 Tests
- `tests/test_accountant.py`:
  - Test that epsilon increases monotonically with rounds.
  - Test that higher noise_multiplier → lower epsilon for same rounds.
  - Test budget_exhausted triggers at max_epsilon.
  - Test subsampling amplification (sample_rate < 1.0 gives lower ε).
  - Verify PLD accountant matches known analytical bounds for Gaussian mechanism.

#### 1.7 Acceptance criteria
- Running `python jobs/esm2_runner.py --dp --dp-noise 0.5 --num-rounds 10` prints
  per-round ε values.
- Running with `--dp-max-epsilon 2.0` stops training early when budget is exhausted.
- `PrivacyAccountant` is independently usable (not coupled to runner).

---

## Phase 2 — Wire SecAgg+ into Runners (P0)
**Branch:** `feature/secagg-integration`

### Problem
`SecAggConfig` and `build_secagg_config()` exist in `src/sfl/privacy/secagg.py` but are
never used. SecAgg is the single most impactful mechanism for preventing the server from
inspecting individual updates (information-theoretic guarantee).

### What to Build

#### 2.1 CLI flags
Both runners:
```
--secagg                         # Enable SecAgg+
--secagg-shares 3                # num_shares (default: 3)
--secagg-threshold 2             # reconstruction_threshold (default: 2)
--secagg-clip 8.0                # clipping_range (default: 8.0)
```

#### 2.2 Server-side integration
Flower SecAgg+ requires `SecAggPlusWorkflow` on the server side. In the runners:
```python
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

if args.secagg:
    from sfl.privacy.secagg import SecAggConfig, build_secagg_config
    secagg_cfg = SecAggConfig(
        num_shares=args.secagg_shares,
        reconstruction_threshold=args.secagg_threshold,
        clipping_range=args.secagg_clip,
    )
    secagg_kwargs = build_secagg_config(secagg_cfg)
    workflow = SecAggPlusWorkflow(**secagg_kwargs)
else:
    workflow = DefaultWorkflow()
```
Then pass `workflow` into `ServerApp` or `run_simulation`.

**Note:** Verify Flower 1.17 API for `SecAggPlusWorkflow`. The exact import and
integration point may have changed. Check:
```python
from flwr.server.workflow import SecAggPlusWorkflow
```
If the API uses `run_config` or `ServerAppComponents`, adapt accordingly.

#### 2.3 Client-side integration
SecAgg+ requires `secaggplus_mod` on the client:
```python
from flwr.client.mod import secaggplus_mod
client_mods.append(secaggplus_mod)
```
This must be added to the client mod chain in both runners when `--secagg` is enabled.

#### 2.4 Validate quantization_range
The current default `quantization_range = 2^22 = 4194304` may cause precision issues
for float values in the range [-1, 1] (typical for normalized gradients). Verify by:
1. Running a training round with SecAgg enabled.
2. Comparing aggregated weights with and without SecAgg.
3. If MSE > 1e-4, increase `quantization_range` to 2^26 or 2^30.

Add a test that quantizes → dequantizes a known float vector and checks roundtrip error.

#### 2.5 Tests
- `tests/test_secagg_integration.py`:
  - Test that `build_secagg_config` returns correct kwargs.
  - Test that CLI flags produce correct `SecAggConfig`.
  - Integration test: run 1-round simulation with `--secagg` and verify convergence.

#### 2.6 Acceptance criteria
- `python jobs/esm2_runner.py --secagg --num-rounds 1` runs successfully.
- SecAgg can be combined with DP: `--dp --secagg` works.
- Aggregated weights match non-SecAgg within quantization error tolerance.

---

## Phase 3 — Fix SVT Implementation (P1)
**Branch:** `feature/fix-svt`

### Problem
The SVT implementation in `filters.py` has three bugs that break its formal ε-DP guarantee:

1. **Non-standard composition**: `eps_2 = ε * (2*n_upload)^(2/3)` doesn't match any
   published SVT variant. The canonical SVT (Dwork & Roth 2014, Theorem 3.25) splits
   the budget as `ε₁ = ε/2` for the threshold and `ε₂ = ε/(2c)` per query.

2. **Decoupled output noise**: `noise_var` is an independent config parameter (default 0.1)
   used as `scale = gamma * 2 / noise_var`. This is disconnected from epsilon, breaking
   the guarantee — the output perturbation scale must be calibrated to the same ε budget.

3. **Unbounded while loop**: If the acceptance rate is very low (high privacy), the loop
   can spin indefinitely.

### What to Fix

#### 3.1 Rewrite SVT to match Dwork & Roth Ch. 3.6
Replace the SVT algorithm in `make_svt_privacy_mod()` with the canonical "Above Threshold"
algorithm:

```python
# Split privacy budget
eps_1 = epsilon / 2.0           # for threshold noise
eps_2 = epsilon / (2.0 * c)     # for each of c queries (c = n_upload)

# Noisy threshold
threshold = tau + np.random.laplace(scale=(sensitivity / eps_1))

# For each parameter (query):
#   noisy_value = |param| + Laplace(scale=sensitivity / eps_2)
#   if noisy_value >= threshold: accept and add output noise
#   else: reject (zero out)
```

Where `sensitivity = gamma` (the L1 sensitivity of clipped params).

#### 3.2 Remove `noise_var` parameter
The output noise must be derived from `epsilon`, not from an independent parameter.
Remove `noise_var` from `SVTPrivacyConfig`. The output noise scale is:
```python
output_noise_scale = gamma / eps_2
```

This is a **breaking API change**. Update:
- `SVTPrivacyConfig` dataclass: remove `noise_var`
- `make_svt_privacy_mod()` signature: remove `noise_var`
- Both runners: no changes needed (neither passes `noise_var` via CLI)
- Tests: update any test that sets `noise_var`

#### 3.3 Add iteration cap to the selection loop
```python
MAX_ITERATIONS = 100
iteration = 0
while len(accepted) < n_upload and candidate_idx.size > 0 and iteration < MAX_ITERATIONS:
    iteration += 1
    ...
if iteration >= MAX_ITERATIONS:
    log(WARNING, "SVT: hit iteration cap, selected only %d/%d", len(accepted), n_upload)
```

#### 3.4 Add formal ε citation
Add a docstring comment referencing:
```
Reference: Dwork & Roth, "The Algorithmic Foundations of DP" (2014),
           Theorem 3.25 (AboveThreshold), Section 3.6 (Sparse Vector Technique)
Total privacy cost: ε (split as ε/2 for threshold + ε/(2c) per query × c queries)
```

#### 3.5 Tests
- `tests/test_svt_fixed.py`:
  - Test that output noise scale is derived from epsilon (not noise_var).
  - Test that iteration cap prevents infinite loop.
  - Test that accepted count ≤ n_upload = fraction × total_params.
  - Statistical test: with very high epsilon (ε=1000), nearly all params accepted;
    with very low epsilon (ε=0.001), nearly all rejected (high sparsity).

#### 3.6 Acceptance criteria
- SVT matches canonical Dwork & Roth formulation.
- No independent noise parameter — everything derived from ε.
- The while loop always terminates.

---

## Phase 4 — PercentilePrivacy Honesty + Optional Noise (P1)
**Branch:** `feature/percentile-privacy-fix`

### Problem
`PercentilePrivacy` (Shokri & Shmatikov 2015) provides **no formal privacy guarantee**.
Zhu et al. (NeurIPS 2019) showed gradient inversion works even on top-10% sparsified
gradients. Users using `--percentile-privacy 10` may believe they have privacy; they don't.

### Options
1. **Remove it entirely** — cleanest, but loses utility benefit (bandwidth reduction).
2. **Add noise to make it a proper randomized mechanism** — adds Gaussian or Laplace noise
   to the top-K values after selection, giving a formal (ε,δ)-DP or ε-DP guarantee.
3. **Keep it but add loud warnings** — mark as "utility optimization only, NOT private".

### Recommended: Option 2 + 3 (noise + warnings)

#### 4.1 Add optional noise to PercentilePrivacy
After the top-K selection and clipping step, add calibrated Gaussian noise:
```python
# After clipping to [-gamma, gamma]:
if noise_scale > 0:
    noise = np.random.normal(0, noise_scale * gamma, size=arr.shape)
    arr = arr + noise
    arr = np.clip(arr, -gamma, gamma)  # re-clip after noise
```

New config field:
```python
@dataclass
class PercentilePrivacyConfig:
    percentile: int = 10
    gamma: float = 0.01
    noise_scale: float = 0.0  # 0 = no noise (legacy), >0 = Gaussian noise std/gamma
```

New CLI flag:
```
--percentile-noise 0.1   # Add Gaussian noise (default: 0, no noise)
```

#### 4.2 Add runtime warning when noise_scale == 0
```python
if cfg.noise_scale == 0:
    log(WARNING,
        "PercentilePrivacy with noise_scale=0 provides NO formal privacy guarantee. "
        "It reduces bandwidth but does NOT prevent gradient inversion attacks. "
        "Use --percentile-noise > 0 for actual privacy, or use --svt-privacy instead.")
```

#### 4.3 Update PRIVACY.md
Add a warning box to the PercentilePrivacy section explaining the limitation.

#### 4.4 Tests
- Test that noise_scale=0 produces deterministic output (given same input).
- Test that noise_scale>0 adds measurable noise (output differs from input).
- Test that the warning is logged when noise_scale=0.

---

## Phase 5 — ExcludeVars Honesty (P1, small)
**Branch:** can be combined with Phase 4 branch

### Problem
Zeroing embedding layers doesn't prevent leakage through downstream layers. Gradient flow
propagates information from "excluded" layers through the rest of the network.

### Fix
This is documentation/warning only — the mechanism itself is fine for reducing what's
explicitly shared.

#### 5.1 Add runtime log
```python
log(WARNING,
    "ExcludeVars zeros %d layers but does NOT guarantee those layers' information "
    "won't leak through other shared layers. Combine with DP for formal guarantees.",
    n_excluded)
```

#### 5.2 Update PRIVACY.md
Add note that ExcludeVars is defense-in-depth, not a standalone privacy mechanism.

---

## Phase 6 — Gradient Compression Defense (P2)
**Branch:** `feature/gradient-compression`

### Problem
Post-2023 attacks (LAMP, Fishing for User Data, APRIL) can reconstruct training data from
gradients even with moderate DP noise. Gradient compression (TopK + random sparsification)
is a complementary defense that also reduces communication bandwidth.

### What to Build

#### 6.1 New mod: `make_gradient_compression_mod()`
In `src/sfl/privacy/filters.py`, add a new filter:
```python
@dataclass
class GradientCompressionConfig:
    compression_ratio: float = 0.1   # Keep top 10% of gradients
    noise_scale: float = 0.01        # Add noise to compressed gradients
    use_random_mask: bool = True     # Random masking (vs deterministic TopK)

def make_gradient_compression_mod(
    compression_ratio: float = 0.1,
    noise_scale: float = 0.01,
    use_random_mask: bool = True,
) -> Callable:
```

Algorithm:
1. Flatten gradients → compute TopK or random mask selecting `compression_ratio * N` values
2. Add Gaussian noise with scale `noise_scale * ||selected||_2 / sqrt(K)`
3. Zero out non-selected values
4. Optionally apply error feedback (accumulate residuals for next round)

**Why random masking**: deterministic TopK leaks which parameters are large, which is
itself information an adversary can exploit. Random masking with probability proportional
to magnitude gives better privacy (Lin et al., "Don't Use Large Mini-Batches", 2020).

#### 6.2 CLI flag
```
--compress 0.1              # Keep 10% of gradients
--compress-noise 0.01       # Noise scale for compressed gradients
--compress-random            # Use random mask (default: True)
```

#### 6.3 Tests
- Test compression ratio is respected (output has ~90% zeros when ratio=0.1).
- Test noise is applied to surviving values.
- Test random mask produces different selections across calls.

---

## Phase 7 — Byzantine-Robust Aggregation (P3)
**Branch:** `feature/robust-aggregation`

### Problem
FedAvg is vulnerable to poisoning attacks — a single malicious client can send adversarial
updates that corrupt the global model. This is orthogonal to privacy (privacy protects
clients from the server; robustness protects the model from clients).

### What to Build

#### 7.1 New strategies in `src/sfl/server/robust.py`

**Multi-Krum** (Blanchard et al., NeurIPS 2017):
```python
class MultiKrumFedAvg(FedAvg):
    """Selects the K most 'typical' client updates, discarding outliers."""
    def __init__(self, num_to_select: int = None, num_byzantine: int = 0, **kwargs):
        ...
    def aggregate_fit(self, server_round, results, failures):
        # 1. Compute pairwise distances between client updates
        # 2. For each client, sum distances to nearest (n - f - 2) neighbors
        # 3. Select K clients with smallest scores
        # 4. Average only selected clients
```

**Trimmed Mean** (Yin et al., ICML 2018):
```python
class TrimmedMeanFedAvg(FedAvg):
    """Coordinate-wise trimmed mean — removes top/bottom β fraction."""
    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        ...
    def aggregate_fit(self, server_round, results, failures):
        # For each parameter coordinate:
        #   Sort client values, remove top/bottom trim_ratio fraction
        #   Average remaining values
```

#### 7.2 CLI flags
```
--aggregation fedavg|krum|trimmed-mean
--krum-byzantine 1           # Expected number of byzantine clients
--trim-ratio 0.1             # Fraction to trim (each side)
```

#### 7.3 Integration
In both runners, select strategy based on `--aggregation`:
```python
if args.aggregation == "krum":
    from sfl.server.robust import MultiKrumFedAvg
    strategy = MultiKrumFedAvg(num_byzantine=args.krum_byzantine, ...)
elif args.aggregation == "trimmed-mean":
    from sfl.server.robust import TrimmedMeanFedAvg
    strategy = TrimmedMeanFedAvg(trim_ratio=args.trim_ratio, ...)
else:
    strategy = SumFedAvg(...)  # or FedAvg for ESM2
```

#### 7.4 Tests
- Test Multi-Krum: inject one outlier update (100x normal magnitude), verify it's excluded.
- Test Trimmed Mean: inject extreme values at specific coordinates, verify they're trimmed.
- Test that with no adversaries, outputs match standard FedAvg within tolerance.

---

## Phase 8 — Documentation Update (all phases)
**Branch:** each phase updates PRIVACY.md in its own PR

### For each phase, update:
1. **docs/PRIVACY.md** — add/update the relevant section
2. **README.md** — update the privacy feature list
3. **config/default.yaml** / **config/esm2.yaml** — add new config keys with comments

### Final PRIVACY.md structure:
```
1. Overview & Threat Model
2. Layer 1: Differential Privacy (with accountant section)
3. Layer 2: Privacy Filters
   2a. SVT (with formal ε citation)
   2b. PercentilePrivacy (with "NOT formal" warning)
   2c. ExcludeVars (with leakage caveat)
   2d. Gradient Compression (new)
4. Layer 3: Secure Aggregation (now wired!)
5. Layer 4: Homomorphic Encryption (demo-only note)
6. Layer 5: Confidential Computing (TEE)
7. Robustness: Byzantine-Robust Aggregation (new)
8. Privacy Accounting (new — how to read ε output)
9. Recommended Configurations (new — cookbook)
10. References
```


---

## Execution Order & Dependencies

```
Phase 1 (Accountant) ──┐
                        ├── Phase 8 (Docs) ──→ Final PR
Phase 2 (SecAgg)   ─────┤
                        │
Phase 3 (Fix SVT)  ─────┤
                        │
Phase 4 (Percentile) ───┤
Phase 5 (ExcludeVars) ──┘   (can merge with Phase 4)
                        
Phase 6 (Compression) ──┐
                        ├── independent, after core fixes
Phase 7 (Byzantine)  ───┘
```

Phases 1-5 are independent and can be developed in parallel.
Phases 6-7 depend on having a stable base (Phases 1-5 merged).
Phase 8 is incremental (each phase updates docs in its own PR).

---

## New Dependencies

| Package | Version | Phase | Optional? |
|---|---|---|---|
| `dp-accounting` | >=0.4.0 | Phase 1 | Yes (`[privacy]` extra) |

No other new dependencies needed — Flower 1.17 already includes SecAgg+, and
compression/robustness use only numpy.

---

## Test Coverage Targets

| Phase | New Tests | Total |
|---|---|---|
| Phase 1 (Accountant) | ~5 | 81 |
| Phase 2 (SecAgg) | ~3 | 84 |
| Phase 3 (SVT fix) | ~4 | 88 |
| Phase 4 (Percentile) | ~3 | 91 |
| Phase 5 (ExcludeVars) | ~1 | 92 |
| Phase 6 (Compression) | ~3 | 95 |
| Phase 7 (Byzantine) | ~4 | 99 |

---

## Risk Assessment

| Phase | Risk | Mitigation |
|---|---|---|
| 1 | `dp-accounting` API may differ from expected | Pin version, write adapter |
| 2 | Flower 1.17 SecAgg+ API may have changed since docs | Check Flower source/changelog first |
| 3 | SVT fix changes output behavior | Keep old behavior behind flag for 1 release |
| 4 | PercentilePrivacy warning may confuse users | Clear docs, default noise_scale to 0 for backward compat |
| 7 | Krum/TrimmedMean slow with many clients or large models | Benchmark first, add warning for >10 clients |
