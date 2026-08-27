"""Run one pytest item and bypass unsafe VTK interpreter teardown."""

from __future__ import annotations

import os
import sys

import pytest


def main() -> None:
    args = [*sys.argv[1:], "--no-isolate-vtk", "-q"]
    exit_code = int(pytest.main(args))
    # The isolation plugin normally exits after the single call report. This
    # fallback handles collection and setup failures that never reach it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
