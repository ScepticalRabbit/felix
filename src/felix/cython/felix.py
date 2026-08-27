# --------------------------------------------------------------------------
# Felix: A High Performance Sensor Simulation Core
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
import cython
import numpy as np
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.felix.cython import felix as cf


# --------------------------------------------------------------------------
# Public Entry Points
# --------------------------------------------------------------------------

def get_last_error() -> str:
    """Return the last error string from the Felix Zig core."""
    buf_len: cython.size_t = 512
    out: cython.uchar[512]
    msg_len: cython.size_t = cf.felixGetLastError(out, buf_len)
    return bytes(out[:msg_len]).decode("utf-8", errors="replace")


def print_sensor_data(
    positions: np.ndarray,
    sample_times: np.ndarray | None,
    spatial_dims: np.ndarray | None,
) -> None:
    """Forward a pyvale SensorData to the Zig core for printing.

    Parameters
    ----------
    positions:
        Sensor positions, shape (n_sensors, 3), dtype float64,
        C-contiguous row-major.
    sample_times:
        Sample times shape (n_times,), dtype float64, or None.
    spatial_dims:
        Spatial dimensions of sensor, shape (3,), dtype float64, or None.
    """
    # Ensure positions is a contiguous 2-D f64 array, then get a memoryview.
    pos_c = np.ascontiguousarray(positions, dtype=np.float64)
    pos_view: cython.double[:, :] = pos_c
    pos_ptr: cython.p_double = cython.address(pos_view[0, 0])
    pos_len: cython.size_t = pos_c.size

    # sample_times – optional
    st_ptr: cython.p_double = cython.NULL
    st_len: cython.size_t = 0
    st_view: cython.double[:]
    st_c: np.ndarray
    if sample_times is not None:
        st_c = np.ascontiguousarray(sample_times, dtype=np.float64)
        st_view = st_c
        st_ptr = cython.address(st_view[0])
        st_len = st_c.size

    # spatial_dims – optional
    sd_ptr: cython.p_double = cython.NULL
    sd_len: cython.size_t = 0
    sd_view: cython.double[:]
    sd_c: np.ndarray
    if spatial_dims is not None:
        sd_c = np.ascontiguousarray(spatial_dims, dtype=np.float64)
        sd_view = sd_c
        sd_ptr = cython.address(sd_view[0])
        sd_len = sd_c.size

    cf.felixPrintSensorData(
        pos_ptr,
        pos_len,
        st_ptr,
        st_len,
        sd_ptr,
        sd_len,
    )
