# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import copy
import dataclasses
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.dataio.simdata import SimData


def print_dataclass_fields(in_data: object) -> None:
    print(f"Data class fields for: {type(in_data)}")
    for data_field in dataclasses.fields(in_data):
        if not data_field.name.startswith("__"):
            print(f"    {data_field.name}: {data_field.type}")
    print()


def print_sim_data(sim_data: SimData) -> None:
    print(f"coords: {None if sim_data.coords is None else sim_data.coords.shape}")
    print(f"time: {None if sim_data.time is None else sim_data.time.shape}")
    print(f"connect: {type(sim_data.connect)}")
    print(f"node_vars: {None if sim_data.node_vars is None else tuple(sim_data.node_vars)}")


def print_dimensions(sim_data: SimData) -> None:
    for name, limits in get_sim_dims(sim_data).items():
        print(f"{name} [min,max] = [{limits[0]},{limits[1]}]")


def get_sim_dims(sim_data: SimData) -> dict[str, tuple[float, float]]:
    coords = sim_data.coords
    if coords is None:
        raise ValueError("SimData has no coords")

    dims = {
        "x": (float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))),
        "y": (float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))),
        "z": (float(np.min(coords[:, 2])), float(np.max(coords[:, 2]))),
    }
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
    scaled_sim = copy.deepcopy(sim_data)
    if scaled_sim.coords is not None:
        scaled_sim.coords = scaled_sim.coords * scale

    if disp_keys is not None and scaled_sim.node_vars is not None:
        for k in disp_keys:
            if k in scaled_sim.node_vars:
                scaled_sim.node_vars[k] = scaled_sim.node_vars[k] * scale

    return scaled_sim


def gen_pos_grid_inside(
    num_sensors: tuple[int, int, int],
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    z_lims: tuple[float, float],
) -> np.ndarray:
    coords_list = []
    lims = (x_lims, y_lims, z_lims)

    for ii, nn in enumerate(num_sensors):
        if nn <= 0:
            coords_list.append(np.array([0.0]))
        elif nn == 1:
            coords_list.append(np.array([0.5 * (lims[ii][0] + lims[ii][1])]))
        else:
            dx = (lims[ii][1] - lims[ii][0]) / (nn + 1)
            coords_list.append(
                np.linspace(lims[ii][0] + dx, lims[ii][1] - dx, nn)
            )

    xx, yy, zz = np.meshgrid(
        coords_list[0], coords_list[1], coords_list[2], indexing="ij"
    )
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def gen_pos_grid_boundary(
    num_sensors: tuple[int, int, int],
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    z_lims: tuple[float, float],
) -> np.ndarray:
    coords_list = []
    lims = (x_lims, y_lims, z_lims)

    for ii, nn in enumerate(num_sensors):
        if nn <= 0:
            coords_list.append(np.array([0.0]))
        elif nn == 1:
            coords_list.append(np.array([0.5 * (lims[ii][0] + lims[ii][1])]))
        else:
            coords_list.append(np.linspace(lims[ii][0], lims[ii][1], nn))

    xx, yy, zz = np.meshgrid(
        coords_list[0], coords_list[1], coords_list[2], indexing="ij"
    )
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def gen_pos_cylinder(
    num_theta: int,
    num_z: int,
    radius: float,
    z_lims: tuple[float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    endpoint_theta: bool = False,
    endpoint_z: bool = True,
) -> np.ndarray:
    thetas = np.linspace(
        0.0, 2.0 * np.pi, num_theta, endpoint=endpoint_theta, dtype=np.float64
    )
    if num_z <= 1:
        z_vals = np.array([(z_lims[0] + z_lims[1]) / 2.0], dtype=np.float64)
    else:
        z_vals = np.linspace(
            z_lims[0], z_lims[1], num_z, endpoint=endpoint_z, dtype=np.float64
        )

    grid_theta, grid_z = np.meshgrid(thetas, z_vals)
    theta_flat = grid_theta.flatten()
    z_flat = grid_z.flatten()

    x_coords = center[0] + radius * np.cos(theta_flat)
    y_coords = center[1] + radius * np.sin(theta_flat)
    z_coords = center[2] + z_flat

    return np.vstack((x_coords, y_coords, z_coords)).T


def gen_pos_sphere(
    num_sensors: int,
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    if num_sensors <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if num_sensors == 1:
        return np.array(
            [[center[0], center[1], center[2] + radius]], dtype=np.float64
        )

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(num_sensors, dtype=np.float64)

    y_unit = 1.0 - (indices / float(num_sensors - 1)) * 2.0
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - y_unit * y_unit))
    theta = golden_angle * indices

    x_coords = center[0] + radius * np.cos(theta) * radius_xy
    y_coords = center[1] + radius * y_unit
    z_coords = center[2] + radius * np.sin(theta) * radius_xy

    return np.vstack((x_coords, y_coords, z_coords)).T


def orient_from_direction(direction: list | np.ndarray) -> Rotation:
    v = np.array(direction, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return Rotation.from_euler("zyx", [0, 0, 0], degrees=True)
    v_norm = v / norm

    e1 = np.array([1.0, 0.0, 0.0])
    dot_val = float(np.dot(e1, v_norm))

    if np.isclose(dot_val, 1.0):
        return Rotation.from_euler("zyx", [0, 0, 0], degrees=True)
    if np.isclose(dot_val, -1.0):
        return Rotation.from_euler("y", 180.0, degrees=True)

    axis = np.cross(e1, v_norm)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0.0:
        axis = axis / axis_norm
    angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
    return Rotation.from_rotvec(axis * angle)


def orient_from_normal(normal: tuple | list | np.ndarray) -> Rotation:
    n = np.array(normal, dtype=float).ravel()
    norm = np.linalg.norm(n)
    if norm == 0.0:
        return Rotation.identity()
    n = n / norm

    e3 = np.array([0.0, 0.0, 1.0])
    dot = np.dot(e3, n)

    if np.isclose(dot, 1.0):
        return Rotation.identity()
    if np.isclose(dot, -1.0):
        return Rotation.from_euler("x", 180.0, degrees=True)

    axis = np.cross(e3, n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0.0:
        axis = axis / axis_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return Rotation.from_rotvec(angle * axis)


def orient_from_normal_and_tangent(
    normal: tuple | list | np.ndarray,
    tangent: tuple | list | np.ndarray,
) -> Rotation:
    n = np.array(normal, dtype=float).ravel()
    t = np.array(tangent, dtype=float).ravel()

    n_norm = np.linalg.norm(n)
    if n_norm > 0.0:
        e3 = n / n_norm
    else:
        e3 = np.array([0.0, 0.0, 1.0])

    t_proj = t - np.dot(t, e3) * e3
    t_norm = np.linalg.norm(t_proj)
    if t_norm > 0.0:
        e1 = t_proj / t_norm
    else:
        fallback = (
            np.array([1.0, 0.0, 0.0])
            if abs(e3[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        t_proj = fallback - np.dot(fallback, e3) * e3
        e1 = t_proj / np.linalg.norm(t_proj)

    e2 = np.cross(e3, e1)
    rot_matrix = np.column_stack((e1, e2, e3))
    return Rotation.from_matrix(rot_matrix)


def print_measurements(
    sens_array: object,
    sensors: int | slice,
    components: int | slice,
    time_steps: int | slice,
) -> None:
    measurement = sens_array.get_measurements()
    truth = sens_array.get_truth()
    rand_errs = sens_array.get_errors_random()
    sys_errs = sens_array.get_errors_systematic()
    tot_errs = sens_array.get_errors_total()

    meas_slice = measurement[sensors, components, time_steps]
    print(f"measurement.shape = \n    {measurement.shape}")
    print(f"measurement = \n    {meas_slice}")
    print(f"truth = \n    {truth[sensors, components, time_steps]}")

    if rand_errs is not None:
        r_slice = rand_errs[sensors, components, time_steps]
        print(f"random errors = \n    {r_slice}")

    if sys_errs is not None:
        s_slice = sys_errs[sensors, components, time_steps]
        print(f"systematic errors = \n    {s_slice}")

    if tot_errs is not None:
        t_slice = tot_errs[sensors, components, time_steps]
        print(f"total errors = \n    {t_slice}")
