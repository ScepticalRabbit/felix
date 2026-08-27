# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import copy
import dataclasses
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.dataio.simdata import SimData
from felix.sensorsim.enums import EDim
from felix.cython import felix as fc


def print_dataclass_fields(in_data: Any) -> None:
    print(f"Data class fields for: {type(in_data)}")
    for f in dataclasses.fields(in_data):
        if not f.name.startswith("__"):
            print(f"    {f.name}: {f.type}")
    print()


def print_sim_data(sim_data: SimData) -> None:
    print()
    if sim_data.time is not None:
        print(f"{sim_data.time.shape=}")
    print()
    if sim_data.coords is not None:
        print(f"{sim_data.coords.shape=}")
    print()

    def print_dict(in_dict: dict | None) -> None:
        if in_dict is None:
            print("    None\n")
            return
        for k, v in in_dict.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: {v.shape}")
            else:
                print(f"    {k}: {type(v)}")
        print()

    print("Connect:")
    print_dict(sim_data.connect)
    print("Node vars:")
    print_dict(sim_data.node_vars)
    print("Elem vars:")
    print_dict(sim_data.elem_vars)
    print("Glob vars:")
    print_dict(sim_data.glob_vars)


def print_dimensions(sim_data: SimData) -> None:
    print(80 * "-")
    print("SimData Dimensions:")
    print(
        f"x [min,max] = [{np.min(sim_data.coords[:, 0])},{np.max(sim_data.coords[:, 0])}]"
    )
    print(
        f"y [min,max] = [{np.min(sim_data.coords[:, 1])},{np.max(sim_data.coords[:, 1])}]"
    )
    print(
        f"z [min,max] = [{np.min(sim_data.coords[:, 2])},{np.max(sim_data.coords[:, 2])}]"
    )
    print(
        f"t [min,max] = [{np.min(sim_data.time)},{np.max(sim_data.time)}]"
    )
    print(80 * "-")


def get_sim_dims(sim_data: SimData) -> dict[str, tuple[float, float]]:
    dims = {}
    dims["x"] = (
        float(np.min(sim_data.coords[:, 0])),
        float(np.max(sim_data.coords[:, 0])),
    )
    dims["y"] = (
        float(np.min(sim_data.coords[:, 1])),
        float(np.max(sim_data.coords[:, 1])),
    )
    dims["z"] = (
        float(np.min(sim_data.coords[:, 2])),
        float(np.max(sim_data.coords[:, 2])),
    )
    if sim_data.time is not None and sim_data.time.size > 0:
        dims["t"] = (
            float(np.min(sim_data.time)),
            float(np.max(sim_data.time)),
        )
    else:
        dims["t"] = (0.0, 0.0)
    return dims


def scale_length_units(
    scale: float,
    sim_data: SimData,
    disp_keys: tuple[str, ...] | None = None,
) -> SimData:
    sim_copy = copy.deepcopy(sim_data)
    if sim_copy.coords is not None:
        sim_copy.coords = sim_copy.coords * scale
    if disp_keys is not None and sim_copy.node_vars is not None:
        for k in disp_keys:
            if k in sim_copy.node_vars:
                sim_copy.node_vars[k] = sim_copy.node_vars[k] * scale
    return sim_copy


def infer_elem_type_and_connect(
    sim_data: SimData, spatial_dims: object
) -> tuple[np.ndarray, int]:
    if sim_data.connect is None:
        raise ValueError("SimData has no connectivity table")

    if isinstance(sim_data.connect, dict):
        blocks = list(sim_data.connect.values())
        arr = np.vstack(blocks)
    elif isinstance(sim_data.connect, np.ndarray):
        arr = sim_data.connect
    else:
        raise TypeError("Unknown connect format in SimData")

    is_2d = (
        spatial_dims == EDim.TWOD
        or getattr(spatial_dims, "value", None) == 2
        or spatial_dims == 2
    )

    cols = arr.shape[1]
    if is_2d:
        if cols in (8, 9):
            return arr[:, :8], 5  # quad8
        elif cols == 6:
            return arr[:, :3], 0  # tri3
        elif cols == 4:
            return arr, 1  # quad4
        elif cols == 3:
            return arr, 0  # tri3
        return arr, 0
    else:
        if cols in (20, 27):
            return arr[:, :20], 6  # hex20
        elif cols == 10:
            return arr[:, :4], 2  # tet4
        elif cols == 8:
            return arr, 3  # hex8
        elif cols == 4:
            return arr, 2  # tet4
        return arr, 2


def sample_simdata_field(
    sim_data: SimData,
    comp_keys: tuple[str, ...],
    spatial_dims: EDim,
    points: np.ndarray,
    times: np.ndarray | None = None,
    angles: tuple[Rotation, ...] | None = None,
    is_tensor: bool = False,
    error_specs: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    is_2d = (
        spatial_dims == EDim.TWOD
        or getattr(spatial_dims, "value", None) == 2
        or spatial_dims == 2
    )
    connect_arr, elem_type_id = infer_elem_type_and_connect(
        sim_data, spatial_dims
    )

    if times is None:
        times_arr = (
            sim_data.time
            if sim_data.time is not None
            else np.array([0.0], dtype=np.float64)
        )
    else:
        times_arr = times

    coords_arr = np.ascontiguousarray(sim_data.coords, dtype=np.float64)
    connect_arr = np.ascontiguousarray(connect_arr, dtype=np.uint64)
    points_arr = np.ascontiguousarray(points, dtype=np.float64)
    times_arr = np.ascontiguousarray(times_arr, dtype=np.float64)
    sim_times_arr = (
        np.ascontiguousarray(sim_data.time, dtype=np.float64)
        if sim_data.time is not None
        else np.array([0.0], dtype=np.float64)
    )

    n_nodes = coords_arr.shape[0]
    n_sim_times = sim_times_arr.shape[0]
    n_comps = len(comp_keys)

    field_data = np.zeros(
        (n_nodes, n_comps, n_sim_times), dtype=np.float64, order="C"
    )
    for cc, key in enumerate(comp_keys):
        field_data[:, cc, :] = sim_data.node_vars[key]

    if angles is not None:
        rots_arr = np.array([r.as_matrix().T for r in angles], dtype=np.float64)
    else:
        rots_arr = np.repeat(
            np.eye(3, dtype=np.float64)[None, :, :], points.shape[0], axis=0
        )
    rots_arr = np.ascontiguousarray(rots_arr, dtype=np.float64)

    return fc.simulate_point_sensors(
        coords=coords_arr,
        connect=connect_arr,
        elem_type=elem_type_id,
        nodal_fields=field_data,
        sim_times=sim_times_arr,
        positions=points_arr,
        sample_times=times_arr,
        rot_matrices=rots_arr,
        spatial_dims=2 if is_2d else 3,
        is_tensor=is_tensor,
        error_specs_list=error_specs,
    )
