# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Convert Exodus simulation files from Pyvale into CSV format for Felix.

Generates structured simulation directories in ./data/ containing:
- coords.csv: (N, 3) nodal coordinates [x, y, z]
- connectivity.csv: (M, K) element connectivity (0-indexed node IDs)
- field_<var>.csv: (N, T) nodal field variables per coordinate across time
"""

from pathlib import Path
import numpy as np

import pyvale.mooseherder as mh


DATA_DIR = Path(__file__).resolve().parent
PYVALE_EXODUS_DIR = Path(
    "/home/lloydf/pyvale/src/pyvale/data/simulation/exodus"
)

SIM_CASES = {
    # 3D Single Element / Cube Benchmark Meshes
    "cube_hex8": "case00_HEX8_out.e",
    "cube_hex20": "case00_HEX20_out.e",
    "tet4": "case00_TET4_out.e",
    # 2D Single Element Benchmark Meshes
    "tri3": "case00_tri3_out.e",
    "tri6": "case00_tri6_out.e",
    "quad4": "case00_quad4_out.e",
    "quad8": "case00_quad8_out.e",
    # Complex Physics Simulations
    "monoblock_3d": "case16_out.e",
    "plate_2d_mech": "case17_out.e",
    "plate_2d_tm": "case18_out.e",
    "cylinder_3d_mech": "case26_out.e",
}


def export_exodus_to_csv(
    exodus_file_path: Path,
    output_dir_path: Path,
) -> None:
    """Read Exodus file using pyvale and save coords, connect, fields to CSV."""
    if not exodus_file_path.is_file():
        print(f"Skipping {exodus_file_path.name}: file not found.")
        return

    output_dir_path.mkdir(parents=True, exist_ok=True)
    loader = mh.ExodusLoader(exodus_file_path)
    sim_data = loader.load_all_sim_data()

    # 1. Coords: (num_nodes, 3)
    coords = sim_data.coords
    if coords is not None:
        if coords.shape[1] == 2:
            coords_3d = np.zeros((coords.shape[0], 3), dtype=np.float64)
            coords_3d[:, :2] = coords
            coords = coords_3d
        np.savetxt(
            output_dir_path / "coords.csv",
            coords,
            delimiter=",",
            fmt="%.12e",
        )

    # 2. Connectivity: (num_elements, nodes_per_elem)
    if sim_data.connect is not None:
        all_connect = []
        for block_key in sorted(sim_data.connect.keys()):
            conn = sim_data.connect[block_key]
            # In pyvale, connect is (nodes_per_elem, num_elems)
            if conn.ndim == 2:
                if conn.shape[0] < conn.shape[1] or conn.shape[0] in (
                    3,
                    4,
                    6,
                    8,
                    9,
                    10,
                    20,
                ):
                    conn = conn.T
            all_connect.append(conn)
        if all_connect:
            merged_connect = np.vstack(all_connect)
            np.savetxt(
                output_dir_path / "connectivity.csv",
                merged_connect,
                delimiter=",",
                fmt="%d",
            )

    # 3. Nodal Fields: (num_nodes, num_timesteps)
    if sim_data.node_vars is not None:
        for var_name, var_data in sim_data.node_vars.items():
            clean_name = str(var_name).replace(" ", "_")
            field_file = output_dir_path / f"field_{clean_name}.csv"
            np.savetxt(
                field_file,
                var_data,
                delimiter=",",
                fmt="%.12e",
            )

    print(
        f"Exported {exodus_file_path.name} -> "
        f"{output_dir_path.relative_to(DATA_DIR.parent)}"
    )


def main() -> None:
    print(f"Converting Exodus simulation files to CSV in {DATA_DIR}...")
    for case_tag, exo_name in SIM_CASES.items():
        exo_path = PYVALE_EXODUS_DIR / exo_name
        out_dir = DATA_DIR / case_tag
        export_exodus_to_csv(exo_path, out_dir)
    print("Exodus conversion complete.")


if __name__ == "__main__":
    main()
