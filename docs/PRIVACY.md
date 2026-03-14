# Privacy & Security Guide

This document covers all privacy-preserving and security hardening features
available in the SFL framework, from Flower-level DP to NVFlare-native
encryption and infrastructure-level confidential computing.

---

## Overview

SFL supports a layered privacy architecture. Each layer can be used
independently or combined for defense in depth:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              Privacy & Robustness Layers                          │
├──────────────┬───────────────┬──────────────┬──────────────┬────────────────────┤
│ Layer 1      │ Layer 2       │ Layer 3      │ Layer 4      │ Layer 5            │
│ DP           │ Privacy       │ Secure Agg & │ Confidential │ Byzantine          │
│ (server +    │ Filters       │ HE           │ Computing    │ Robustness         │
│ client DPSGD)│               │              │              │                    │
├──────────────┼───────────────┼──────────────┼──────────────┼────────────────────┤
│ Noise added  │ Sparsification│ Server never │ Hardware     │ Outlier-resistant  │
│ to aggregate │ + compression │ sees indiv.  │ enclaves     │ aggregation        │
│ or per-sample│ at client     │ contributions│ (TEEs)       │                    │
├──────────────┼───────────────┼──────────────┼──────────────┼────────────────────┤
│ Flower       │ Flower mods   │ TenSEAL /    │ Azure CVM /  │ Multi-Krum /       │
│ strategy +   │ (ported from  │ Flower       │ NVFlare      │ Trimmed Mean       │
│ Opacus       │ NVFlare)      │ SecAgg+      │ provisioning │ (Flower strategy)  │
└──────────────┴───────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## Layer 1: Differential Privacy (DP)

Adds calibrated Gaussian noise to model updates, providing formal
(ε,δ)-differential privacy guarantees. A model trained with DP
cannot memorize individual training samples.

### Server-Side DP

The server clips each client's update to a fixed L2 norm, then adds
noise to the aggregate. Simpler setup — no client-side modification.

```bash
# Sum demo
python jobs/flower_runner.py --dp --dp-noise 0.5 --dp-clip 5.0

# ESM2
python jobs/esm2_runner.py --dp --dp-noise 0.1 --dp-clip 10.0
```

### Client-Side DP

Each client clips its own update before sending. The server adds
noise to the clipped aggregate. Stronger privacy — the server never
sees unclipped updates.

```bash
python jobs/esm2_runner.py --dp --dp-mode client --dp-noise 0.1
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dp` | off | Enable differential privacy |
| `--dp-noise` | 0.1 | Noise multiplier (σ/C). Higher = more private |
| `--dp-clip` | 10.0 | Max L2 norm for clipping. For ESM2, 10.0 is good |
| `--dp-mode` | server | `server` or `client` |
| `--dp-delta` | 1e-5 | Target δ in (ε,δ)-DP |
| `--dp-max-epsilon` | 10.0 | Budget cap — training auto-stops when ε exceeds this |
| `--dp-adaptive-clip` | off | Enable adaptive clipping (Andrew et al. 2021) |
| `--dp-target-quantile` | 0.5 | Target fraction of unclipped updates (0–1) |
| `--dp-clip-lr` | 0.2 | Learning rate for geometric clip norm update |

### Privacy Accounting

SFL automatically tracks the cumulative (ε,δ)-DP guarantee across rounds
using one of two selectable backends:

- **PLD** (default) — Google's `dp-accounting` library with the Privacy Loss
  Distribution accountant. Tighter composition bounds than RDP for Gaussian
  mechanisms.
- **PRV** — Microsoft's `prv_accountant` library using the Privacy Random
  Variable framework with FFT-based composition. Returns error bounds
  (ε_low, ε_estimate, ε_high) in addition to the point estimate.

```bash
# Use default PLD backend
python jobs/esm2_runner.py --dp --dp-noise 0.5

# Use PRV backend (Microsoft) for error bounds
python jobs/esm2_runner.py --dp --dp-noise 0.5 --dp-accounting-backend prv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dp-accounting-backend` | pld | `pld` (Google) or `prv` (Microsoft) |

**Features:**

- **Automatic budget enforcement** — when cumulative ε reaches
  `--dp-max-epsilon`, `aggregate_fit` returns `(None, {})` to stop
  training. The `_AccountingWrapper` intercepts every round to check
  and step the accountant.
- **Subsampling amplification** — when fewer clients participate than
  the total pool, Poisson subsampling amplification yields tighter ε
  (no extra noise needed — pure accounting win).
- **Per-round participation tracking** — the accountant tracks actual
  `num_participants` per round (not just the sample rate), so ε reflects
  the true participation pattern.
- **Predictive planning** — `compute_epsilon_for_rounds(n)` previews
  what ε would be after N more rounds without advancing state.
- **BudgetExhaustedError** — raised when `enforce_budget=True` and
  the budget is exceeded, allowing callers to handle gracefully.
- **PRV error bounds** — when using the PRV backend, access ε error
  bounds via `accountant.epsilon_bounds` → `(ε_low, ε_est, ε_high)`.

```python
from sfl.privacy.accountant import PrivacyAccountant

# PLD backend (default)
acc = PrivacyAccountant(noise_multiplier=1.0, delta=1e-5, max_epsilon=10.0)

# PRV backend with error bounds
acc = PrivacyAccountant(
    noise_multiplier=1.0, delta=1e-5, max_epsilon=10.0, backend="prv",
)
for r in range(num_rounds):
    eps = acc.step()
    print(f"Round {r+1}: ε = {eps:.4f}")
    if acc.backend == "prv":
        low, est, high = acc.epsilon_bounds
        print(f"  bounds: [{low:.4f}, {high:.4f}]")
    if acc.budget_exhausted:
        break
```

### Adaptive Clipping (Andrew et al. 2021)

Fixed clipping norms require manual tuning. Adaptive clipping
automatically adjusts the clip bound based on the distribution of
client update norms, targeting a configurable quantile.

```bash
python jobs/esm2_runner.py --dp --dp-adaptive-clip --dp-target-quantile 0.5
```

When `quantile_noise_multiplier > 0` (set via `DPConfig`), Gaussian noise
is added to the binary "clipped/not-clipped" indicator before averaging,
making the quantile estimate itself differentially private.

### Per-Layer Clipping (Yu et al., ICLR 2022)

A single global clipping norm applies the same bound to every parameter
tensor. For models like ESM2 where embedding/head layers have much larger
norms than attention layers, this wastes privacy budget on small layers
and under-clips large layers. Per-layer clipping applies independent L2
clips to each parameter tensor, significantly improving utility.

```bash
# Default clip of 1.0 per layer
python jobs/esm2_runner.py --per-layer-clip 1.0

# Custom clip norms per layer index (JSON map)
python jobs/esm2_runner.py --per-layer-clip 1.0 --per-layer-clip-map '{"0": 5.0, "1": 2.0}'
```

| Flag                    | Default | Description                              |
|-------------------------|---------|------------------------------------------|
| `--per-layer-clip`      | off     | Default per-layer L2 clip norm           |
| `--per-layer-clip-map`  | none    | JSON mapping of layer index to clip norm |

The overall L2 sensitivity is `sqrt(sum of clip_i^2)`, so per-layer
clipping gives a tighter bound than applying `max(clip_i)` globally.

### Shuffle-Model DP Amplification

When clients send updates through an anonymous channel (shuffler),
the central (ε,δ)-DP guarantee is significantly tighter than the
local ε₀ each client applies. This is based on the analytical bound
from Feldman, McMillan & Talwar (2021, Theorem 3.1).

For small ε₀ and n clients, the central ε scales as
approximately ε₀ · √(8·log(4/δ)/n) — a √n improvement over
the local guarantee. This is a pure accounting win with no
algorithmic changes needed.

```bash
# Enable shuffle-model amplification
python jobs/esm2_runner.py --dp --dp-noise 0.5 --dp-shuffle

# Works with any DP mode
python jobs/flower_runner.py --dp --dp-noise 0.5 --dp-shuffle
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dp-shuffle` | off | Enable shuffle-model DP amplification |

When enabled, each round's fit metrics include:

- `dp_shuffle_epsilon`: Central ε after amplification
- `dp_shuffle_amplification`: Amplification factor (local ε / central ε)

**Prerequisite**: Requires an anonymous communication channel between
clients and server (e.g., a mixnet, onion routing, or a trusted shuffler
service). Without this infrastructure, the amplification bound does not hold.

```python
from sfl.privacy.accountant import shuffle_amplification_epsilon

# 100 clients, each with local ε₀ = 2.0
central_eps = shuffle_amplification_epsilon(
    local_epsilon=2.0, num_clients=100, delta=1e-5,
)
print(f"Central ε = {central_eps:.4f}")  # Much tighter than 2.0
```

### Noise Calibration

`calibrate_gaussian_sigma()` computes the minimal Gaussian noise σ needed
to satisfy a target (ε,δ)-DP guarantee using binary search on the PLD.
This is used by the PercentilePrivacy and GradientCompression filters when
`epsilon` is specified.

### Joint DP Composition

When clients run per-example DP-SGD (via `--dpsgd`) and the server applies
server-side DP, the `_AccountingWrapper` automatically composes both
guarantees using sequential composition:

```
total_ε = server_ε + max(client_ε_i)
total_δ = server_δ + client_δ
```

The composed metrics (`dp_total_epsilon`, `dp_total_delta`,
`dpsgd_epsilon_max`) are reported in the aggregate fit metrics.

### Choosing ε

The privacy budget ε is determined by `noise_multiplier`, `clipping_norm`,
`num_clients`, and `num_rounds`. Lower ε = stronger privacy:

| ε range | Privacy level | Typical noise_multiplier |
|---------|--------------|-------------------------|
| ε < 1 | Strong | 1.0–10.0 |
| 1 < ε < 10 | Moderate | 0.1–1.0 |
| ε > 10 | Weak (utility-focused) | 0.01–0.1 |

---

### Per-Example DP-SGD (Client-Side)

Per-example DP-SGD via Opacus clips and noises *each sample's gradient*
during local training, providing a formal (ε,δ)-DP guarantee at the
individual sample level — stronger than server-side DP which only
protects the aggregate.

```bash
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0 --dpsgd-clip 1.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dpsgd` | off | Enable per-example DP-SGD |
| `--dpsgd-clip` | 1.0 | Max per-sample gradient L2 norm |
| `--dpsgd-noise` | 1.0 | Noise multiplier for DP-SGD |
| `--dpsgd-delta` | 1e-5 | Target δ for DP-SGD |
| `--dpsgd-autoclip` | off | Enable AutoClip (Li et al., NeurIPS 2023) |
| `--dpsgd-ghost` | off | Enable Ghost Clipping (memory-efficient two-pass DP-SGD) |

Each client reports `dpsgd_epsilon` in its fit metrics, which is
automatically composed with server-side ε by the `_AccountingWrapper`.

#### AutoClip (Automatic Clipping)

Standard DP-SGD requires a clipping norm hyperparameter `C` that
trades off bias (too small) vs. noise magnitude (too large). AutoClip
(Li et al., NeurIPS 2023) eliminates this tradeoff by normalizing
each per-example gradient to unit L2 norm before noise addition.

With AutoClip enabled:

- Each gradient is scaled to ‖g‖ = 1 via a backward hook
- Opacus clips at C = 1, which is a no-op on unit-norm gradients
- Noise scale becomes σ · 1 = σ, independent of gradient magnitude
- The `--dpsgd-clip` flag is ignored

```bash
# AutoClip: no need to tune --dpsgd-clip
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0 --dpsgd-autoclip
```

#### Ghost Clipping (Memory-Efficient DP-SGD)

Standard Opacus materializes per-sample gradients, requiring O(B×P)
memory (batch size × parameters). Ghost Clipping (Li et al., 2022;
Opacus `grad_sample_mode="ghost"`) uses two backward passes instead:

1. First pass computes per-sample gradient norms (O(B) memory)
2. Second pass computes clipped aggregated gradients (O(P) memory)

Total memory: O(B+P) instead of O(B×P) — critical for large models
like ESM2.

```bash
# Ghost clipping for memory-efficient DP-SGD
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0 --dpsgd-ghost

# Can combine with AutoClip
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0 --dpsgd-ghost --dpsgd-autoclip
```

**Requires**: `pip install opacus>=1.5`

---

## Layer 2: Privacy Filters

Ported from NVFlare's server-enforced DXO filters, reimplemented as
Flower client mods. These intercept client updates after local training
and transform them before they reach the server.

### PercentilePrivacy

Only shares the top N% of weight diffs by absolute magnitude. Everything
below the cutoff is zeroed. Based on Shokri & Shmatikov, CCS '15.

```bash
# Keep only top 10% of weight diffs
python jobs/esm2_runner.py --percentile-privacy 10

# With custom clipping bound
python jobs/esm2_runner.py --percentile-privacy 10 --percentile-gamma 0.05
```

| Flag | Default | Description |
|------|---------|-------------|
| `--percentile-privacy` | off | Percentile threshold (0–100) |
| `--percentile-gamma` | 0.01 | Max abs value clip for surviving diffs |
| `--percentile-noise` | 0.0 | Uncalibrated noise scale (manual) |
| `--percentile-epsilon` | 0.0 | Target ε for calibrated Gaussian noise |
| `--percentile-delta` | 1e-5 | Target δ (used with `--percentile-epsilon`) |

When `--percentile-epsilon` is set, noise is auto-calibrated via PLD
binary search (`calibrate_gaussian_sigma`) to provide a formal
(ε,δ)-DP guarantee. When only `--percentile-noise` is set, uncalibrated
noise is added with a warning.

**Effect**: At `--percentile-privacy 10`, 90% of weight diffs are zeroed.
This dramatically reduces information leakage and network bandwidth while
preserving the most significant updates.

### SVT (Sparse Vector Technique) Privacy

Provides formal ε-differential privacy using the Sparse Vector Technique.
Selects parameters via a noisy threshold, adds Laplace noise to
accepted values, zeros the rest.

```bash
# Default SVT (ε=0.1, 10% of params shared)
python jobs/esm2_runner.py --svt-privacy

# Stronger privacy (smaller ε)
python jobs/esm2_runner.py --svt-privacy --svt-epsilon 0.01

# Share more parameters (less sparse)
python jobs/esm2_runner.py --svt-privacy --svt-fraction 0.3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--svt-privacy` | off | Enable SVT differential privacy |
| `--svt-epsilon` | 0.1 | Privacy budget (lower = more private) |
| `--svt-fraction` | 0.1 | Fraction of params to upload (0–1) |
| `--svt-no-optimal` | off | Disable optimal budget allocation (Lyu et al. 2017) |
| `--svt-prescreen` | 1.0 | Pre-screen ratio — only SVT-score the top fraction by magnitude |

By default, SVT uses the numerically-optimal budget split from Lyu
et al. 2017, which allocates more ε to the threshold noise than to the
value noise for better utility. Use `--svt-no-optimal` for the classic
even split. Pre-screening (`--svt-prescreen 0.5`) reduces computation
by only scoring the top half of parameters by absolute value.

### ExcludeVars (Layer Exclusion)

Zeros out entire model layers before sending to the server. Useful for
keeping sensitive components (embeddings, classifier heads) private.

```bash
# Exclude ESM2 embedding layers (indices 0 and 1)
python jobs/esm2_runner.py --exclude-layers 0,1

# Exclude embeddings + LM head
python jobs/esm2_runner.py --exclude-layers 0,1,-2,-1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--exclude-layers` | none | Comma-separated parameter indices to zero |

**Determining ESM2 layer indices**: Run to inspect parameter names:
```python
from sfl.esm2.model import load_model, get_parameters
model = load_model("facebook/esm2_t6_8M_UR50D")
for i, (name, _) in enumerate(model.named_parameters()):
    print(f"  {i}: {name}")
```

### Gradient Compression

Reduces the number of transmitted parameters via TopK or random
masking, with optional (ε,δ)-DP calibrated noise. Useful as both a
communication-efficiency and privacy defense.

```bash
# Keep top 10% of parameters by magnitude
python jobs/esm2_runner.py --compress 0.1 --compress-topk

# Random masking (keep a random 10%)
python jobs/esm2_runner.py --compress 0.1

# With manual noise
python jobs/esm2_runner.py --compress 0.1 --compress-noise 0.05
```

| Flag | Default | Description |
|------|---------|-------------|
| `--compress` | off | Compression ratio (fraction of params to keep, 0–1) |
| `--compress-topk` | off | Use TopK selection (default: random mask) |
| `--compress-noise` | 0.01 | Heuristic noise scale for masked values |
| `--compress-error-feedback` | off | Accumulate compression residuals across rounds (FedSparQ-style) |

When `epsilon` and `delta` are set via `GradientCompressionConfig`
programmatically, noise is auto-calibrated via PLD instead of using
the heuristic `noise_scale`.

### Combining Filters

Multiple filters can be stacked. They are applied in order:
1. Client-side DP clipping (if `--dp --dp-mode client`)
2. PercentilePrivacy (if `--percentile-privacy`)
3. SVT Privacy (if `--svt-privacy`)
4. ExcludeVars (if `--exclude-layers`)

```bash
# Full privacy stack: DP + percentile filter + layer exclusion
python jobs/esm2_runner.py \
    --dp --dp-mode client --dp-noise 0.5 \
    --percentile-privacy 10 \
    --exclude-layers 0,1
```

---

## Layer 3: Secure Aggregation & Homomorphic Encryption

### Secure Aggregation (SecAgg+)

Flower's SecAgg+ protocol ensures the server only sees the aggregate
of client updates — it cannot inspect individual contributions.

**How it works:**
1. Each client splits its update into secret shares
2. Shares are exchanged between clients (not the server)
3. Server receives only the sum of shares
4. If a client drops, remaining shares can reconstruct the aggregate

SecAgg+ is configured via `SecAggConfig` in `src/sfl/privacy/secagg.py`.
It requires the `SecAggPlusWorkflow` on the server and `secaggplus_mod`
on clients. See the [Flower SecAgg documentation](https://flower.ai/docs/framework/how-to-use-secagg.html)
for deployment details.

### Partial Freezing (Lambda-SecAgg)

When fine-tuning large models like ESM2, most layers are frozen and only
a subset are trained. The `--freeze-layers` flag strips frozen layers
from client updates before they pass through SecAgg, reducing the volume
of data that needs secret sharing and encryption.

```bash
# Only layers 4, 5, 6 are trainable — freeze everything else
python jobs/esm2_runner.py --secagg --freeze-layers 4,5,6

# Combine with DP and SecAgg for full protection
python jobs/esm2_runner.py --dp --secagg --freeze-layers 4,5,6
```

| Flag              | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| `--freeze-layers` | off     | Comma-separated trainable layer indices to keep|

This reduces SecAgg overhead from O(P) to O(lambda * P), where lambda is
the fraction of trainable parameters. For ESM2 with only the last few
layers unfrozen, this can be a 90%+ reduction in SecAgg cost.

### Homomorphic Encryption (HE)

HE allows the server to aggregate encrypted model updates without ever
seeing plaintext values. SFL provides a TenSEAL CKKS implementation.

```python
from sfl.privacy.he import HEContext
import numpy as np

# Create encryption context
he = HEContext()

# Client encrypts parameter
encrypted = he.encrypt_parameters([np.array([7.5])])

# Server aggregates encrypted values (homomorphic addition)
# No decryption needed during aggregation!
agg = he.add_encrypted(encrypted_client_1, encrypted_client_2)

# Only the aggregate is decrypted
result = he.decrypt_parameters(agg, [(1,)])
```

#### Installation

```bash
pip install tenseal
# or
pip install sfl[he]
```

#### Limitations

HE has fundamental practical limitations that restrict it to small-model
demonstrations:

| Constraint | Impact |
|-----------|--------|
| **Ciphertext expansion** | A single float32 (4 bytes) becomes ~160KB encrypted. For ESM2 (8M params = 32MB), encrypted parameters would be ~1.3TB per client per round. |
| **Computation overhead** | CKKS arithmetic is ~1000x slower than plaintext. Aggregating encrypted ESM2 updates would take hours per round. |
| **Limited operations** | Only addition and scalar multiplication work on ciphertext. FedAvg's weighted averaging is feasible, but more complex aggregation is not. |
| **Key management** | In simulation, client and server share the same TenSEAL context. Production HE requires a trusted key authority. |
| **Precision** | CKKS is approximate — decrypted values have small numerical errors (~1e-4). Fine for aggregation, but errors accumulate over many rounds. |

**Recommendation**: Use HE for the sum demo or other scalar-parameter tasks.
For ESM2 and other deep learning models, use DP + SecAgg instead — they
provide strong privacy with practical performance.

#### NVFlare Native HE

NVFlare provides production-grade HE via its native `FedJob` pipeline:

- `HEModelEncryptor` — client-side DXO filter that encrypts model weights
- `HEModelDecryptor` — server-side filter that decrypts aggregated result
- `InTimeAccumulateModelAggregator` — aggregates encrypted weights via CKKS

This pipeline requires **native NVFlare jobs** (not the FlowerRecipe
integration used by this project). It is designed for NVFlare's own
workflow system where DXO filters can intercept Shareable objects.

The FlowerRecipe integration runs Flower as an external process — NVFlare
does not intercept Flower's weight exchanges. Therefore, **NVFlare's
HE filters cannot be used with the Flower runner path**. Our TenSEAL
`HEContext` provides equivalent functionality at the application level.

---

## Layer 4: Confidential Computing

Confidential computing provides hardware-level protection where even the
server operator cannot inspect client data or model updates in memory.

### What It Is

Trusted Execution Environments (TEEs) like Intel SGX, AMD SEV, and ARM
TrustZone create isolated memory enclaves. Code running inside a TEE:

- Cannot be inspected or modified by the host OS or hypervisor
- Has memory encrypted by the CPU, inaccessible to other processes
- Can prove its integrity via remote attestation (the client
  can verify the server is running the expected code)

### NVFlare Support

NVFlare 2.7 includes provisioning for confidential computing:

- **Azure Confidential VMs** — `nvflare.lighter.cc_provision.impl.azure`
  provisions startup kits for Azure DCsv3/DCdsv3 VMs with Intel SGX
- **On-prem CVMs** — `nvflare.lighter.cc_provision.impl.onprem_cvm`
  supports on-premise confidential VM deployments
- **Attestation** — clients can verify the server enclave before
  sending updates, ensuring the aggregation code hasn't been tampered with

### When to Use It

Confidential computing is the strongest server-side guarantee, but it
requires infrastructure investment:

| Aspect | Details |
|--------|---------|
| **Protection** | Even a compromised server host cannot read model updates |
| **Requirements** | TEE-capable hardware (Intel SGX, AMD SEV-SNP) |
| **Cloud** | Azure DCsv3 VMs, AWS Nitro Enclaves, GCP Confidential VMs |
| **Performance** | 5-15% overhead for memory encryption |
| **Complexity** | Requires NVFlare provisioning with CC builders |

Confidential computing is orthogonal to DP and HE — it protects the
*execution environment* rather than the *data format*. In a high-security
deployment, you would combine all layers:

```
TEE enclave (Layer 4)
  └── NVFlare server with HE aggregation (Layer 3)
        └── Clients apply DP noise + privacy filters (Layers 1-2)
```

### Setting Up (Azure Example)

```bash
# 1. Generate provisioning with CC support
nvflare provision -p project.yml -w workspace

# 2. project.yml includes CC builder:
# builders:
#   - path: nvflare.lighter.cc_provision.impl.azure.AzureCCProvisioner
#     args:
#       attestation_provider: "sharedweu"

# 3. Deploy to Azure DCsv3 VMs
# The startup kits include attestation verification
```

This is an infrastructure/deployment concern and is not implemented
in the SFL application code. See [NVFlare CC documentation](https://nvflare.readthedocs.io/en/main/programming_guide/confidential_computing.html)
for detailed deployment guides.

---

## Layer 5: Byzantine-Robust Aggregation

Standard FedAvg is vulnerable to Byzantine (malicious or faulty)
clients that send poisoned updates. SFL provides three robust
aggregation strategies as drop-in replacements.

### Multi-Krum (Blanchard et al., NeurIPS 2017)

Scores each client update by its summed distance to the nearest
`n − f − 2` other updates, then selects the `k` updates with the
lowest scores. Outliers (adversarial or faulty clients) are discarded.

```bash
python jobs/esm2_runner.py --aggregation krum --krum-byzantine 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--aggregation` | fedavg | Aggregation strategy: `fedavg`, `krum`, `trimmed-mean`, or `foundation-fl` |
| `--krum-byzantine` | 1 | Max number of Byzantine clients to tolerate |

### Trimmed Mean (Yin et al., ICML 2018)

Computes the coordinate-wise mean after removing the top and bottom
`trim_ratio` fraction of values at each coordinate. Robust to a
minority of corrupted updates.

```bash
python jobs/esm2_runner.py --aggregation trimmed-mean --trim-ratio 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--trim-ratio` | 0.1 | Fraction of values to trim from each end (0–0.5) |

### FoundationFL (NDSS 2025)

Trust-scoring defense using a small server root dataset. The server
computes a reference update from root data, then scores each client
update by cosine similarity to the reference. Clients with similarity
below the trust threshold are excluded; remaining updates are weighted
by their similarity score (trust-weighted averaging).

More robust than spectral or distance-based defenses against adaptive
model-poisoning attacks (Shejwalkar & Houmansadr, USENIX 2021).

```bash
# FoundationFL with default threshold
python jobs/esm2_runner.py --aggregation foundation-fl

# Stricter filtering
python jobs/esm2_runner.py --aggregation foundation-fl --ffl-threshold 0.3

# Binary filtering (no trust weighting)
python jobs/esm2_runner.py --aggregation foundation-fl --ffl-no-weighted
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ffl-threshold` | 0.1 | Min cosine similarity to keep a client update (-1 to 1) |
| `--ffl-no-weighted` | off | Use binary filtering instead of trust-weighted averaging |

**Note**: For strongest defense, provide a root dataset so the server
can compute its own reference update. Without a root dataset, the
strategy falls back to using the client mean as reference (less robust).

---

## Architecture: Why Flower Mods Instead of NVFlare DXO Filters?

NVFlare has excellent built-in privacy filters (`SVTPrivacy`,
`PercentilePrivacy`, `ExcludeVars`) that operate on DXO objects in
the native NVFlare pipeline. However, this project uses NVFlare's
**FlowerRecipe** integration, which runs Flower as an external process.

```
                  NVFlare Native Path              FlowerRecipe Path
                  ─────────────────               ──────────────────
                  Client                          Client
                    │                               │
                    ▼                               ▼
                  DXO Filter ◄── SVTPrivacy       Flower Mod ◄── our mods
                    │                               │
                    ▼                               ▼
                  NVFlare RPC                     Flower gRPC
                    │                               │
                    ▼                               ▼
                  Server DXO Filter               Flower Strategy
                    │                               │
                    ▼                               ▼
                  Aggregator                      FedAvg + DP wrapper
```

In the FlowerRecipe path, NVFlare does not intercept Flower's weight
exchanges — the DXO filter pipeline is bypassed. To provide equivalent
privacy, we port the NVFlare filter algorithms as Flower client mods:

| NVFlare Filter | Flower Mod Equivalent | Status |
|---------------|----------------------|--------|
| `PercentilePrivacy` | `make_percentile_privacy_mod()` | ✅ Implemented |
| `SVTPrivacy` | `make_svt_privacy_mod()` | ✅ Implemented |
| `ExcludeVars` | `make_exclude_vars_mod()` | ✅ Implemented |
| `HEModelEncryptor` | `HEContext` (application-level) | ✅ Demo |
| `SecAggPlusWorkflow` | Flower `SecAggPlusWorkflow` | 📋 Config only |

The Flower mod equivalents implement the same algorithms (percentile
sparsification, SVT with Laplace noise, layer exclusion) and are
applied in the same position (client-side, before upload to server).

---

## Quick Reference

### Sum Demo — All Privacy Options

```bash
# DP only
python jobs/flower_runner.py --dp --dp-noise 0.5

# DP + percentile filter
python jobs/flower_runner.py --dp --percentile-privacy 10

# DP + SVT
python jobs/flower_runner.py --dp --svt-privacy --svt-epsilon 0.05
```

### ESM2 — All Privacy Options

```bash
# DP only (server-side)
python jobs/esm2_runner.py --dp --dp-noise 0.1

# Adaptive clipping
python jobs/esm2_runner.py --dp --dp-adaptive-clip

# DP + percentile filter + layer exclusion
python jobs/esm2_runner.py \
    --dp --dp-mode client \
    --percentile-privacy 10 \
    --exclude-layers 0,1

# Per-example DP-SGD
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0

# SVT privacy (strongest sparsification + formal ε-DP)
python jobs/esm2_runner.py --svt-privacy --svt-epsilon 0.05 --svt-fraction 0.2

# Gradient compression
python jobs/esm2_runner.py --compress 0.1 --compress-topk

# Byzantine-robust aggregation
python jobs/esm2_runner.py --aggregation krum --krum-byzantine 1

# Kitchen sink (all layers except HE)
python jobs/esm2_runner.py \
    --dp --dp-mode client --dp-noise 0.5 --dp-clip 5.0 \
    --dp-adaptive-clip \
    --percentile-privacy 10 --percentile-gamma 0.01 \
    --exclude-layers 0,1 \
    --aggregation trimmed-mean
```

### Privacy vs. Utility Trade-offs

| Configuration | Privacy | Utility | Best for |
|--------------|---------|---------|----------|
| No privacy | None | Maximum | Trusted environments |
| `--dp --dp-noise 0.1` | Moderate | High | Internal consortia |
| `--dp --dp-noise 1.0` | Strong | Moderate | Sensitive data |
| `--dpsgd --dpsgd-noise 1.0` | Strong (sample-level) | Moderate | Per-sample guarantees |
| `--percentile-privacy 10` | Moderate | High | Bandwidth + privacy |
| `--svt-privacy --svt-epsilon 0.1` | Strong (ε-DP) | Moderate | Formal guarantees |
| `--compress 0.1 --compress-topk` | Low-moderate | High | Communication efficiency |
| `--aggregation krum` | Byzantine-robust | High | Adversarial settings |
| Full stack (DP + filters + exclude) | Very strong | Lower | Regulatory compliance |
