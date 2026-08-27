# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import os
import sys
from collections.abc import Generator

# Set up headless rendering BEFORE any imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_OPENGL_SOFTWARE_RENDERING"] = "1"

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


def pytest_configure(config: pytest.Config) -> None:
    """Configure PyVista only inside a process that already imported it."""
    pyvista = sys.modules.get("pyvista")
    if pyvista is None:
        return
    pyvista.OFF_SCREEN = True


@pytest.fixture(autouse=True)
def close_visualisation_windows() -> Generator[None, None, None]:
    """Close figures and plotters without importing optional VTK modules."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")

    pyvista = sys.modules.get("pyvista")
    if pyvista is not None:
        pyvista.close_all()
