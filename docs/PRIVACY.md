# Privacy & Security Guide

This document covers all privacy-preserving and security hardening features
available in the SFL framework, from Flower-level DP to NVFlare-native
encryption and infrastructure-level confidential computing.

---

## Overview

SFL supports a layered privacy architecture. Each layer can be used
independently or combined for defense in depth:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Privacy Layers                               │
├──────────────┬───────────────┬──────────────┬───────────────────┤
│ Layer 1      │ Layer 2       │ Layer 3      │ Layer 4           │
│ DP           │ Privacy       │ Secure Agg & │ Confidential      │
│ (Flower)     │ Filters       │ HE           │ Computing         │
├──────────────┼───────────────┼──────────────┼───────────────────┤
│ Noise added  │ Sparsification│ Server never │ Hardware          │
│ to aggregate │ at client     │ sees indiv.  │ enclaves          │
│ or updates   │ before upload │ contributions│ (TEEs)            │
├──────────────┼───────────────┼──────────────┼───────────────────┤
│ Flower       │ Flower mods   │ TenSEAL /    │ Azure CVM /       │
│ strategy     │ (ported from  │ Flower       │ NVFlare           │
│ wrappers     │ NVFlare)      │ SecAgg+      │ provisioning      │
└──────────────┴───────────────┴──────────────┴───────────────────┘
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

### Choosing ε

The privacy budget ε is determined by `noise_multiplier`, `clipping_norm`,
`num_clients`, and `num_rounds`. Lower ε = stronger privacy:

| ε range | Privacy level | Typical noise_multiplier |
|---------|--------------|-------------------------|
| ε < 1 | Strong | 1.0–10.0 |
| 1 < ε < 10 | Moderate | 0.1–1.0 |
| ε > 10 | Weak (utility-focused) | 0.01–0.1 |

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

# DP + percentile filter + layer exclusion
python jobs/esm2_runner.py \
    --dp --dp-mode client \
    --percentile-privacy 10 \
    --exclude-layers 0,1

# SVT privacy (strongest sparsification + formal ε-DP)
python jobs/esm2_runner.py --svt-privacy --svt-epsilon 0.05 --svt-fraction 0.2

# Kitchen sink (all layers except HE)
python jobs/esm2_runner.py \
    --dp --dp-mode client --dp-noise 0.5 --dp-clip 5.0 \
    --percentile-privacy 10 --percentile-gamma 0.01 \
    --exclude-layers 0,1
```

### Privacy vs. Utility Trade-offs

| Configuration | Privacy | Utility | Best for |
|--------------|---------|---------|----------|
| No privacy | None | Maximum | Trusted environments |
| `--dp --dp-noise 0.1` | Moderate | High | Internal consortia |
| `--dp --dp-noise 1.0` | Strong | Moderate | Sensitive data |
| `--percentile-privacy 10` | Moderate | High | Bandwidth + privacy |
| `--svt-privacy --svt-epsilon 0.1` | Strong (ε-DP) | Moderate | Formal guarantees |
| Full stack (DP + filters + exclude) | Very strong | Lower | Regulatory compliance |
