# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import sys
from bench.bench_felix_only import main as main_felix_only
from bench.bench_comp_pyvale import main as main_comp_pyvale


def main() -> None:
    if "--felix-only" in sys.argv or "-f" in sys.argv:
        main_felix_only()
    else:
        main_comp_pyvale()


if __name__ == "__main__":
    main()
