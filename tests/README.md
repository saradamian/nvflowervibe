# Test Guide

## Source-to-Test Mapping

| Source Module | Test File | Notes |
|---------------|-----------|-------|
| `client/base.py` | `test_client.py` | Base client contract |
| `client/inference.py` | `test_extensibility.py` | BaseInferenceClient contract |
| `client/dp_client.py` | `test_dpsgd.py` | Requires torch + opacus (slow) |
| `server/strategy.py` | `test_integration.py` | Tested via end-to-end pipeline |
| `server/robust.py` | `test_robust.py` | Multi-Krum, TrimmedMean, FoundationFL |
| `privacy/filters.py` | `test_filters.py` | Percentile, SVT, Compression, Freeze, Adapter |
| `privacy/dp.py` | `test_privacy.py` | DPConfig, wrapping, noise calibration |
| `privacy/accountant.py` | `test_accountant.py` | PLD/PRV, composition, budget |
| `privacy/adaptive_clip.py` | `test_privacy.py` | Adaptive clip + per-layer clip |
| `privacy/audit.py` | `test_privacy.py` | PrivacyAuditor, pipeline audit |
| `privacy/secagg.py` | `test_privacy.py` | SecAgg config validation |
| `privacy/he.py` | `test_filters.py` | HE encrypt/decrypt (requires tenseal) |
| `privacy/runner_utils.py` | `test_runner_utils.py` | CLI args, mod building, env validation |
| `esm2/*` | `test_esm2_*.py` | 5 files, all require torch (slow) |
| `llm/*` | `test_llm_*.py` | 5 files, all require torch (slow) |
| `utils/config.py` | `test_config.py` | YAML + env + CLI config |
| `utils/params.py` | `test_extensibility.py` | downcast/upcast utilities |
| `utils/checkpoint.py` | `test_hpc_utils.py` | CheckpointManager, save/load/cleanup |
| `utils/metrics.py` | `test_hpc_utils.py` | MetricsCollector, CSV/JSON export |
| `utils/resources.py` | `test_hpc_utils.py` | ClientResources, auto-detect, backend config |
| `utils/grpc_auth.py` | `test_hpc_utils.py` | TLS config, token auth, env helpers |

## Test Markers

- **`@pytest.mark.slow`**: Tests requiring torch, opacus, or dp-accounting. Excluded from CI.
- **`pytest.mark.skipif`**: Tests skipped when optional dependencies aren't installed.

## Running Tests

```bash
# CI default (fast tests, ~2.5 min)
python -m pytest tests/ -v -m "not slow"

# Full suite (~8 min, needs torch + opacus + dp-accounting)
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_filters.py -v

# With coverage
python -m pytest tests/ --cov=src/sfl --cov-report=term-missing
```

## Test Patterns

- Mock Flower internals with `MagicMock(spec=ClientProxy)` etc.
- Use `_make_train_message()` / `_extract_params()` from `test_filters.py` for mod tests.
- Use `_make_result()` from `test_robust.py` for aggregation strategy tests.
- Privacy mods are tested by passing through a mock Flower message chain.
