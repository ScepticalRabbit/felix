# Testing Guide

Run the complete Felix Python test suite from the repository root with:

```bash
python -m pytest
```

When using the repository virtual environment, run:

```bash
.venv/bin/python -m pytest
```

The Python tests currently live in `src/felix/pytests`. Use ordinary pytest
selectors to run a smaller target, for example:

```bash
python -m pytest src/felix/pytests/test_sensorsim.py
python -m pytest src/felix/pytests/test_sensorsim.py -k vector_2d
python -m pytest src/felix/pytests/test_sensorsim_examples.py
```

## Adding tests

- Put focused unit and regression tests in `src/felix/pytests` using a
  `test_<area>.py` filename.
- Name test functions after the behaviour being checked, not the internal
  implementation.
- Keep tests deterministic. Seed random generators and avoid dependence on
  test order or state left by another test.
- Prefer small, direct assertions. Include enough failure context to identify
  the case and largest discrepancy when checking arrays.
- Add reusable test setup to `conftest.py` rather than copying it between test
  modules.
- Run the focused test while developing, then run the full suite before
  completion.

## Testing examples

Supported examples are executed by
`src/felix/pytests/test_sensorsim_examples.py`. Add a new example to the
appropriate `BASIC_EXAMPLES` or `EXTENDED_EXAMPLES` parameter list.

Example tests must run non-interactively and must not leave files or GUI state
for the next test. The suite selects Matplotlib's `Agg` backend, enables
PyVista off-screen rendering, closes all Matplotlib figures and PyVista
plotters after every test, and runs examples from pytest's temporary directory.
Do not weaken this cleanup when adding an example. If an example creates a
different external resource, add equivalent cleanup that also runs when the
example fails.

Tests in VTK-sensitive modules run serially in disposable child processes.
Each child has a 120 second timeout and is terminated as a complete process
group if it exceeds that limit. The child exits immediately after its test call
report so unsafe VTK interpreter teardown cannot corrupt the main pytest
process. Use `@pytest.mark.pyvista` for a VTK-sensitive test outside one of the
existing marked modules. Use `--no-isolate-vtk` only when diagnosing the native
failure locally; it is not the supported full-suite mode.

Example tests replace PyVista and Matplotlib plotting entry points with inert
test doubles. They validate the calculations and example control flow without
creating native rendering contexts. Check interactive visual behaviour manually
when an example's plotting code changes.

Test normal interactive example behaviour manually outside pytest when a
change affects plotting or visual output.

## Gold regression data

Gold files are committed reference results, not ordinary test output. Generate
or update them only after deliberately reviewing an independently known-good
result. Never regenerate gold merely to make a failing test pass.

All gold-generation scripts belong in `scripts/`, including generators used by
only one test module. Name them `gengold_<area>.py`, require an explicit write
option where practical, and make their output location clear. Generated gold
belongs beside the tests that consume it.

Sensor-simulation gold is stored in `src/felix/pytests/gold`. Refresh all of it
with:

```bash
python scripts/gengold_sensorsim.py --write
```

Refresh a single reviewed case with:

```bash
python scripts/gengold_sensorsim.py --write --case <exact-case-tag>
```

After generating gold, inspect the changed files, explain why the expected
result changed, and rerun the focused regression test followed by the complete
suite.

## Required checks

Before completing a Python change:

1. Run the affected focused tests.
2. Run `python -m pytest`.
3. Confirm example tests did not create repository output or leave windows
   open.
4. Inspect the final diff and verify that no unrelated generated artefacts were
   added.
