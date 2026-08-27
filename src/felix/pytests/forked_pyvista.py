"""Pytest plugin to run pyvista tests in isolated subprocesses.

This avoids VTK/PyVista memory leaks by executing marked test modules
in separate Python processes.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


PYVISTA_TEST_MODULES = {
    "test_sensors_ray.py",
    "test_sensors_spatial_ray.py",
    "test_sensor_library.py",
    "test_sensorsim_examples.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-fork-pyvista",
        action="store_true",
        default=False,
        help="Disable forking for pyvista tests (run in main process)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark pyvista test modules for forked execution."""
    if config.getoption("--no-fork-pyvista"):
        return

    pyvista_items = []
    other_items = []

    for item in items:
        module_name = Path(item.fspath).name
        if module_name in PYVISTA_TEST_MODULES:
            item.add_marker(pytest.mark.forked)
            pyvista_items.append(item)
        else:
            other_items.append(item)

    # Reorder: run non-pyvista tests first, then pyvista tests
    items[:] = other_items + pyvista_items


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> bool:
    """Run pyvista tests in a forked subprocess."""
    if not item.get_closest_marker("forked"):
        return None  # Let pytest handle normally

    if item.config.getoption("--no-fork-pyvista"):
        return None

    module_path = item.fspath
    test_name = item.name

    # Run this specific test in a subprocess
    env = os.environ.copy()
    env["PYTEST_CURRENT_TEST"] = f"{module_path}::{test_name}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{module_path}::{test_name}",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    # Replay output
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    # Report result to pytest
    if result.returncode == 0:
        item.session.testscollected += 1
        item._forked_result = "passed"
    else:
        item._forked_result = "failed"

    # Create a fake report for pytest's internal tracking
    from _pytest.reports import TestReport

    rep = TestReport.from_item_and_call(
        item=item,
        call=pytest.CallInfo(
            lambda: None,  # dummy
            sys.exc_info() if result.returncode != 0 else (None, None, None),
        ),
    )
    rep.outcome = "passed" if result.returncode == 0 else "failed"
    rep.longrepr = result.stderr if result.returncode != 0 else ""
    item.session.testscollected -= 1  # We'll report manually
    item.ihook.pytest_runtest_logreport(report=rep)

    return True  # Tell pytest we handled this test


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Add summary of forked tests."""
    forked_passed = 0
    forked_failed = 0
    for rep in terminalreporter.stats.get("passed", []):
        if getattr(rep, "_forked_result", None) == "passed":
            forked_passed += 1
    for rep in terminalreporter.stats.get("failed", []):
        if getattr(rep, "_forked_result", None) == "failed":
            forked_failed += 1

    if forked_passed or forked_failed:
        terminalreporter.write_sep("=", "Forked PyVista Tests")
        terminalreporter.write_line(f"  Passed: {forked_passed}")
        terminalreporter.write_line(f"  Failed: {forked_failed}")