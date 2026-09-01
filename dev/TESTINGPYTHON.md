# Python Testing Guide

Run the complete Felix Python test suite from the repository root with:

```bash
python -m pytest
```

When using the repository virtual environment, run:

```bash
.venv/bin/python -m pytest
```

The Python tests live in `src/felix/pytests`. Use standard pytest selectors to run a targeted subset:

```bash
.venv/bin/python -m pytest src/felix/pytests/test_sensorsim.py
.venv/bin/python -m pytest src/felix/pytests/test_sensorsim.py -k vector_2d
.venv/bin/python -m pytest src/felix/pytests/test_sensorsim_examples.py
```

---

## Adding Python Tests

- Place focused unit and regression tests in `src/felix/pytests` using a `test_<area>.py` naming scheme.
- Name test functions descriptively after the physical behaviour or feature being verified.
- Keep tests deterministic: seed pseudo-random number generators explicitly and avoid order dependencies between tests.
- Use direct numpy array assertions (`np.testing.assert_allclose`) with appropriate relative and absolute tolerances.
- Include reusable test fixtures in `conftest.py`.

---

## Testing Examples & PyVista Isolation

Supported examples are executed by `src/felix/pytests/test_sensorsim_examples.py`. Add new examples to the `BASIC_EXAMPLES` or `EXTENDED_EXAMPLES` parameter lists.

- Example tests run non-interactively in headless mode using Matplotlib's `Agg` backend and PyVista off-screen rendering.
- All figures and plotters are closed after each test in pytest's temporary directory.
- VTK-sensitive modules execute serially in disposable child processes with timeouts to prevent native VTK teardown issues.

---

## Gold Regression Data

Gold files are committed reference data for verifying end-to-end regression consistency.

All gold generators reside in `scripts/`:
- `gengold_sensorsim.py`: Refresh simulation gold results.

```bash
.venv/bin/python scripts/gengold_sensorsim.py --write
.venv/bin/python scripts/gengold_sensorsim.py --write --case <case-tag>
```

---

## Required Pre-Commit Checks

1. Run the targeted tests during feature development:
   ```bash
   .venv/bin/python -m pytest src/felix/pytests/test_<feature>.py
   ```
2. Run the complete pytest test suite:
   ```bash
   .venv/bin/python -m pytest
   ```
3. Ensure no temporary test output or leftover graphical windows were created.
