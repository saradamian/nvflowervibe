# Contributing to SFL

Thank you for your interest in contributing to the SFL federated learning framework.

## Development Setup

```bash
# Clone and set up
git clone https://github.com/saradamian/nvflowervibe.git
cd nvflowervibe
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt

# Optional dependencies
pip install opacus>=1.5          # Per-example DP-SGD
pip install dp-accounting>=0.4   # PLD privacy accounting
pip install tenseal              # Homomorphic encryption
pip install prv-accountant       # PRV accounting (Microsoft)
```

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_filters.py -v

# With coverage
python -m pytest tests/ --cov=src/sfl --cov-report=term-missing
```

All PRs must pass the full test suite. CI runs automatically on every PR to `main`.

## Branch and PR Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feat/your-feature
   ```

2. Implement your change with tests.

3. Run the full test suite locally:
   ```bash
   python -m pytest tests/ -v
   ```

4. Push and create a PR:
   ```bash
   git push -u origin feat/your-feature
   gh pr create --base main
   ```

5. After review, merge with `--merge` (not squash) to preserve commit history.

### Branch naming

- `feat/<name>` — new features
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation only
- `refactor/<name>` — code restructuring without behavior change

### PR guidelines

- Each PR should be self-contained and independently mergeable
- Update relevant docs (README.md, docs/PRIVACY.md) in the same PR
- Tests must pass before merge
- No force-pushes to `main`

## Code Organization

```
src/sfl/
├── client/          # Federated client implementations
├── server/          # Server strategies and aggregation
├── esm2/            # ESM2 protein model application
├── privacy/         # All privacy mechanisms
│   ├── accountant.py    # PLD/PRV privacy accounting + auxiliary composition
│   ├── adaptive_clip.py # Adaptive + per-layer clipping
│   ├── audit.py         # PrivacyAuditor — empirical DP validation
│   ├── dp.py            # DP wrappers, noise calibration
│   ├── filters.py       # Client mods (Percentile, SVT, Compression, Partial Freeze)
│   ├── he.py            # Homomorphic encryption
│   └── secagg.py        # SecAgg+ configuration
└── utils/           # Shared utilities (config, logging, RNG)
```

## Adding a New Privacy Mechanism

Privacy mechanisms follow a consistent pattern:

1. **Flower client mod** — intercepts `FitRes` after local training, transforms parameters:
   ```python
   def make_my_mod(**config) -> Callable:
       def my_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
           if msg.metadata.message_type != MessageType.TRAIN:
               return call_next(msg, ctxt)
           out_msg = call_next(msg, ctxt)
           # Transform parameters here
           return out_msg
       return my_mod
   ```

2. **Export** from `sfl.privacy.__init__` and add to `__all__`.

3. **CLI flags** in both `jobs/flower_runner.py` and `jobs/esm2_runner.py`.

4. **Tests** in the appropriate test file (e.g., `tests/test_filters.py`).

5. **Documentation** in `docs/PRIVACY.md` with usage examples and a flag reference table.

## Adding a New Aggregation Strategy

1. Extend `FedAvg` in `src/sfl/server/robust.py`.
2. Add a CLI `--aggregation` choice in both runners.
3. Wire through env vars in `src/sfl/server/app.py`.
4. Add tests in `tests/test_robust.py`.

## Adding a New FL Application

Follow the ESM2 module pattern:

1. Create `src/sfl/your_app/` with `config.py`, `model.py`, `dataset.py`, `client.py`, `server.py`.
2. Extend `BaseFederatedClient` — implement `compute_update()`.
3. Create a runner in `jobs/your_app_runner.py`.
4. Add tests in `tests/test_your_app_*.py`.

## Testing Guidelines

- Use `pytest` with descriptive test names and docstrings.
- Mock Flower internals (Context, Message) using `unittest.mock.MagicMock(spec=...)`.
- For optional dependencies (opacus, tenseal, dp-accounting), use `pytest.mark.skipif`.
- Test both the happy path and edge cases (empty inputs, invalid configs).
- Use `_make_train_message()` / `_extract_params()` helpers from `tests/test_filters.py` for mod tests.

### Slow Test Marker

Tests that require heavy dependencies (torch, opacus, dp-accounting PRV) or
GPU resources must be marked with `@pytest.mark.slow`:

```python
import pytest

@pytest.mark.slow
class TestMyHeavyFeature:
    def test_something(self):
        ...

# Or for an entire file:
pytestmark = pytest.mark.slow
```

CI runs only fast tests (`-m "not slow"`) to keep PR feedback under ~2.5 min.
The full suite (238 tests: 176 fast + 62 slow) should be run locally before
merging changes that touch privacy or model code.

### Privacy Auditing in Tests

Use `PrivacyAuditor.run_pipeline_audit()` to validate that new privacy mods
actually reduce information leakage through the real Flower mod chain:

```python
from sfl.privacy import PrivacyAuditor

def test_my_mod_reduces_leakage():
    auditor = PrivacyAuditor(param_shapes=[(1000,)])
    result = auditor.run_pipeline_audit(
        params=[np.zeros(1000, dtype=np.float32)],
        mods=[my_new_mod],
        num_trials=200,
    )
    assert result.passed
```

## Privacy and Security

When contributing privacy-related code:

- All noise must be generated via `sfl.utils.rng.secure_rng()` (CSRNG-seeded), never `np.random`.
- Any new DP mechanism must have its privacy cost accounted for in the total budget.
- Runtime warnings must be logged when a mechanism provides no formal guarantee.
- Document the threat model and any limitations in `docs/PRIVACY.md`.
- Validate all config parameters in `__post_init__` with clear error messages.

## Style

- Follow existing code conventions (no strict linter enforced).
- Keep functions focused and well-documented.
- Prefer composition over inheritance for privacy mechanisms.
- Use dataclasses for configuration.
- Log important events at INFO level, warnings at WARNING level.
