"""Run VTK-sensitive pytest items in bounded, disposable processes."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _pytest.reports import TestReport


VTK_CHILD_ENV = "FELIX_PYTEST_VTK_CHILD"
VTK_TEST_MODULES = {
    "test_sensor_library.py",
    "test_sensors_ray.py",
    "test_sensors_spatial_ray.py",
    "test_sensorsim_examples.py",
}
DEFAULT_TIMEOUT_SECONDS = 120.0
TERMINATE_GRACE_SECONDS = 5.0


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("felix vtk isolation")
    group.addoption(
        "--no-isolate-vtk",
        action="store_true",
        default=False,
        help="run VTK-sensitive tests in the current pytest process",
    )
    group.addoption(
        "--vtk-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="maximum runtime for one isolated VTK test",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Mark known VTK modules and place them after ordinary tests."""
    if _is_child_process():
        return

    ordinary_items: list[pytest.Item] = []
    vtk_items: list[pytest.Item] = []
    for item in items:
        if Path(str(item.path)).name in VTK_TEST_MODULES:
            item.add_marker(pytest.mark.pyvista)

        if item.get_closest_marker("pyvista") is None:
            ordinary_items.append(item)
        else:
            vtk_items.append(item)

    items[:] = ordinary_items + vtk_items


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item,
    nextitem: pytest.Item | None,
) -> bool | None:
    """Execute one marked item without loading VTK into the parent."""
    if item.get_closest_marker("pyvista") is None:
        return None
    if item.config.getoption("--no-isolate-vtk") or _is_child_process():
        return None

    item.ihook.pytest_runtest_logstart(
        nodeid=item.nodeid,
        location=item.location,
    )
    start_time = time.time()
    outcome, longrepr, output = _run_isolated_item(item)
    stop_time = time.time()

    if output and outcome != "passed":
        terminal = item.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write(output)

    report = TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords={name: 1 for name in item.keywords},
        outcome=outcome,
        longrepr=longrepr,
        when="call",
        sections=[],
        duration=stop_time - start_time,
        start=start_time,
        stop=stop_time,
        user_properties=[],
    )
    report.isolated_vtk = True
    item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(
        nodeid=item.nodeid,
        location=item.location,
    )
    return True


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report: TestReport) -> None:
    """Exit a one-test child before unsafe VTK teardown can execute."""
    if not _is_child_process() or report.when != "call":
        return

    if report.failed:
        sys.stderr.write(f"\n{report.longreprtext}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if report.failed else 0)


def _run_isolated_item(
    item: pytest.Item,
) -> tuple[str, str | None, str]:
    timeout = item.config.getoption("--vtk-timeout")
    command = [
        sys.executable,
        "-m",
        "felix.pytests.isolated_pytest",
        item.nodeid,
    ]
    environment = os.environ.copy()
    environment[VTK_CHILD_ENV] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        preexec_fn=_set_parent_death_signal if os.name == "posix" else None,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        output = _stop_process_group(process)
        message = (
            f"isolated VTK test exceeded {timeout:.1f} seconds; "
            "its process group was terminated"
        )
        return "failed", message, output

    if process.returncode == 0:
        return "passed", None, output

    message = f"isolated VTK test exited with status {process.returncode}"
    return "failed", message, output


def _stop_process_group(process: subprocess.Popen[str]) -> str:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        output, _ = process.communicate(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
    return output


def _is_child_process() -> bool:
    return os.environ.get(VTK_CHILD_ENV) == "1"


def _set_parent_death_signal() -> None:
    """Ask Linux to kill a child if its pytest parent disappears."""
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGKILL)
    except (AttributeError, OSError):
        return

    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)
