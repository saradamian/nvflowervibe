# CLAUDE.md — Agent & Developer Guide

## Quick Orientation

SFL is a federated learning framework (~5,600 lines source) with 14 composable privacy layers and 4 aggregation strategies. It runs on Flower + NVFlare.

**The codebase is 75% model-agnostic.** ESM2 (protein model) and LLM (causal language model) are example applications — the privacy, aggregation, and client infrastructure work for any PyTorch model.

## Architecture Map

```
src/sfl/
├── types.py              # Type aliases: Parameters, Metrics, ClientUpdate
├── client/
│   ├── base.py           # BaseFederatedClient — THE extension point for training
│   ├── inference.py      # BaseInferenceClient — extension point for inference
│   ├── dp_client.py      # Per-example DP-SGD (Opacus wrapper)
│   └── sum_client.py     # Demo client (scalar sum)
├── server/
│   ├── strategy.py       # SumFedAvg (demo only)
│   ├── robust.py         # Multi-Krum, TrimmedMean, FoundationFL
│   └── app.py            # Server factory (reads SFL_* env vars)
├── privacy/
│   ├── filters.py        # ★ Main extension point: make_*_mod() factories
│   ├── dp.py             # DPConfig, wrap_strategy_with_dp(), calibration
│   ├── accountant.py     # PLD/PRV privacy accounting + budget
│   ├── adaptive_clip.py  # Adaptive + per-layer clipping
│   ├── audit.py          # PrivacyAuditor (canary-based validation)
│   ├── runner_utils.py   # ★ Shared CLI args + mod builder for runners
│   ├── secagg.py         # SecAgg+ configuration
│   └── he.py             # HE (demo only, impractical for real models)
├── esm2/                 # Example app: protein MLM fine-tuning
│   ├── config.py         # ESM2RunConfig (module-level singleton)
│   ├── model.py          # HuggingFace AutoModelForMaskedLM
│   ├── dataset.py        # Protein sequences, MLM masking
│   ├── client.py         # ESM2Client extends BaseFederatedClient
│   └── server.py         # FedAvg + pretrained weights
├── llm/                  # Example app: causal LM fine-tuning (GPT-2)
│   ├── config.py         # LLMRunConfig (same pattern as ESM2)
│   ├── model.py          # AutoModelForCausalLM, LoRA support
│   ├── dataset.py        # Text dataset, causal LM tokenization
│   ├── client.py         # LLMClient with optional LoRA
│   └── server.py         # FedAvg + pretrained weights
└── utils/
    ├── config.py          # YAML + env + CLI config loading
    ├── logging.py         # Rich/simple/JSON logging
    ├── rng.py             # secure_rng() — CSRNG-seeded RNG
    └── params.py          # downcast/upcast for mixed-precision
```

## How to Add a New Training Use Case

1. Create `src/sfl/your_app/` with: `config.py`, `model.py`, `dataset.py`, `client.py`, `server.py`, `__init__.py`
2. Your client extends `BaseFederatedClient` — implement `compute_update(parameters, config) -> (params, n, metrics)`
3. Create `jobs/your_runner.py` — use `add_privacy_args(parser)` and `build_privacy_mods(args)` from `sfl.privacy.runner_utils` to get all privacy flags for free
4. All privacy mods (DP, SVT, percentile, compression) work automatically — they operate on numpy arrays, not model objects

## Key Patterns

- **Privacy mods are model-agnostic**: They intercept Flower `FitRes` messages and transform numpy parameter arrays. No model knowledge needed.
- **Module-level config singleton**: Each app uses `_run_config` set by the runner, read by `client_fn()`. See `esm2/config.py`.
- **Client factory**: `client_fn(context: Context)` reads `partition_id` from `context.node_config` to assign data splits.
- **Server factory**: `server_fn(context: Context)` creates strategy + config. DP wrapping happens here via `wrap_strategy_with_dp()`.
- **Env var config**: Server reads `SFL_*` env vars (set by runner). Use `validate_env_vars()` at startup to catch typos early.

## Complexity Hotspots

- `privacy/filters.py` (828 lines) — 6 filter implementations with DP accounting. The `make_percentile_privacy_mod()` adaptive K logic (lines 81-252) is the most complex.
- `privacy/accountant.py` (588 lines) — PLD/PRV dual-backend with lazy imports. The `compose_auxiliary()` method only works with PLD.
- `privacy/dp.py` (471 lines) — `_AccountingWrapper` intercepts every aggregation round. Diamond imports from accountant + adaptive_clip.
- `server/robust.py` (429 lines) — Three Byzantine strategies. FoundationFL requires `root_update` or explicit opt-out.

## Common Gotchas

1. **CSRNG only**: All noise must use `sfl.utils.rng.secure_rng()`, never `np.random`. This prevents adversaries from predicting noise.
2. **FoundationFL requires root_update**: Instantiating `FoundationFLFedAvg()` without `root_update` raises `ValueError` unless `allow_untrusted_reference=True`.
3. **SecAgg threshold is enforced**: `reconstruction_threshold < ceil(2*num_shares/3)` raises `ValueError`.
4. **ExcludeVars is NOT private**: It zeroes layers but gradient flow leaks information through shared layers. It's a communication optimization, not a privacy mechanism.
5. **HE is demo-only**: TenSEAL CKKS works for the sum demo but is impractical for real models (160KB per float32 → 1.3TB for ESM2).
6. **Per-layer clipping supports name patterns**: Use `clip_norms={"embed": 5.0, "attention": 1.0}` for regex matching against parameter names, not just integer indices.

## Testing

```bash
# Fast tests only (CI, ~2.5 min)
python -m pytest tests/ -v -m "not slow"

# Full suite (requires torch/opacus/dp-accounting)
python -m pytest tests/ -v
```

Test-to-source mapping: see `tests/README.md`.

## Security Model

- Privacy guarantees: formal (ε,δ)-DP via PLD/PRV accounting
- All DP costs composed (server DP + client DP-SGD + adaptive clip + SVT)
- Byzantine robustness: Multi-Krum, Trimmed Mean, FoundationFL
- Server-side norm verification prevents bypassing client-side clips
- No authentication/rate-limiting on Flower gRPC (simulation only; add for production)
