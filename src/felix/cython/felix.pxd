# --------------------------------------------------------------------------
# Felix: A High Performance Sensor Simulation Core
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
# Cython declaration file (.pxd) for the Felix C ABI.
# Import this with:
#   from felix.cython cimport felix as cf
# --------------------------------------------------------------------------
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

cdef extern from "felix.h":

    size_t felixGetLastError(
        uint8_t *out_buf,
        size_t out_buf_len,
    )

    void felixPrintSensorData(
        const double *positions_ptr,
        size_t        positions_len,
        const double *sample_times_ptr,
        size_t        sample_times_len,
        const double *spatial_dims_ptr,
        size_t        spatial_dims_len,
    )
