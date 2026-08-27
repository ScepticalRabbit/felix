# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import os
from collections.abc import Generator

# Set up headless rendering BEFORE any imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_OPENGL_SOFTWARE_RENDERING"] = "1"

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pyvista for headless testing before any tests run."""
    try:
        import pyvista
        pyvista.OFF_SCREEN = True
        pyvista.global_theme.off_screen = True
        pyvista.global_theme.interactive = False
    except Exception:
        pass

    # Register forked_pyvista plugin
    config.pluginmanager.load_setuptools_entrypoints("pytest11")


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Generator[None, None, None]:
    """Close matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
