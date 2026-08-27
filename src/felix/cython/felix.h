#ifndef FELIX_H
#define FELIX_H

/*
 * Felix: A High Performance Sensor Simulation Core
 *
 * Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
 * Licensed under the MIT License (see LICENSE file for details)
 *
 * Public Felix C ABI
 * Fixed to f64 precision for ABI stability.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copy the last error message into out_buf (null-terminated).
 * Returns the number of bytes written (excluding the null terminator).
 */
size_t felixGetLastError(uint8_t *out_buf, size_t out_buf_len);

/*
 * Receive a pyvale SensorData from Python and print a summary to stderr.
 *
 *   positions_ptr      flat row-major f64 array, shape (n_sensors, 3)
 *   positions_len      total elements = n_sensors * 3
 *   sample_times_ptr   flat f64 array, shape (n_times,); may be NULL / 0
 *   sample_times_len   number of sample-time elements (0 if not supplied)
 *   spatial_dims_ptr   flat f64 array, shape (3,); may be NULL / 0
 *   spatial_dims_len   number of spatial-dim elements (0 or 3)
 */
void felixPrintSensorData(
    const double *positions_ptr,
    size_t        positions_len,
    const double *sample_times_ptr,
    size_t        sample_times_len,
    const double *spatial_dims_ptr,
    size_t        spatial_dims_len
);

#ifdef __cplusplus
}
#endif

#endif /* FELIX_H */
