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

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_visualisation_windows() -> Generator[None, None, None]:
    """Close figures and plotters created by each test, including failures."""
    yield

    import matplotlib.pyplot as plt

    plt.close("all")

    try:
        import pyvista
    except ImportError:
        return

    pyvista.close_all()
