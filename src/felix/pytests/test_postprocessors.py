# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Unit and analytic verification tests for post-measurement signal processors,
filters, temporal/spatial differentiation and integration, and DAG execution.
"""

import numpy as np
import pytest

import felix as fx
import pyvale.verif as verif


def test_filter_smoothers_analytic() -> None:
    """Verifies that all pre-smoothing filters significantly reduce Gaussian
    noise on an analytic harmonic wave.
    """
    times = np.linspace(0.0, 2.0 * np.pi, 200)
    truth = np.sin(times)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.15, size=truth.shape)
    noisy = truth + noise

    # Shape: (1, 1, 200)
    raw_data = fx.MeasurementData(
        values=noisy[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("u",),
        units="mm",
    )

    noise_err = np.linalg.norm(noisy - truth)

    # 1. Savitzky-Golay
    savgol = fx.ProcessFilterSavitzkyGolay(
        source="disp", window_length=15, polyorder=2
    )
    savgol_res = savgol.process({"disp": raw_data}).values[0, 0]
    savgol_err = np.linalg.norm(savgol_res - truth)
    assert savgol_err < 0.5 * noise_err

    # 2. Gaussian
    gauss = fx.ProcessFilterGaussian(source="disp", sigma=3.0)
    gauss_res = gauss.process({"disp": raw_data}).values[0, 0]
    gauss_err = np.linalg.norm(gauss_res - truth)
    assert gauss_err < 0.6 * noise_err

    # 3. Moving Average
    mavg = fx.ProcessFilterMovingAverage(source="disp", window_length=9)
    mavg_res = mavg.process({"disp": raw_data}).values[0, 0]
    mavg_err = np.linalg.norm(mavg_res - truth)
    assert mavg_err < 0.6 * noise_err

    # 4. Median
    med = fx.ProcessFilterMedian(source="disp", size=7)
    med_res = med.process({"disp": raw_data}).values[0, 0]
    med_err = np.linalg.norm(med_res - truth)
    assert med_err < 0.7 * noise_err

    # 5. Butterworth
    bw = fx.ProcessFilterButterworth(
        source="disp", cutoff=1.0, order=3, btype="lowpass"
    )
    bw_res = bw.process({"disp": raw_data}).values[0, 0]
    bw_err = np.linalg.norm(bw_res - truth)
    assert bw_err < 0.6 * noise_err


def test_temporal_differentiation_analytic() -> None:
    """Verifies time differentiation against analytic derivatives:
    u(t) = A*sin(w*t) -> v(t) = A*w*cos(w*t) -> a(t) = -A*w^2*sin(w*t).
    """
    omega = 2.0 * np.pi * 2.0  # 2 Hz
    amp = 5.0
    times = np.linspace(0.0, 1.0, 500)

    u_exact = amp * np.sin(omega * times)
    v_exact = amp * omega * np.cos(omega * times)
    a_exact = -amp * (omega**2) * np.sin(omega * times)

    raw_data = fx.MeasurementData(
        values=u_exact[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("u",),
        units="mm",
    )

    # 1. Finite differences 1st derivative (velocity)
    diff_fd1 = fx.ProcessDifferentiateTime(
        source="u", order=1, method="finite_diff"
    )
    v_fd = diff_fd1.process({"u": raw_data}).values[0, 0]
    interior = slice(5, -5)
    np.testing.assert_allclose(v_fd[interior], v_exact[interior], rtol=1e-3)

    # 2. Spline 1st derivative
    diff_spl1 = fx.ProcessDifferentiateTime(
        source="u", order=1, method="spline"
    )
    v_spl = diff_spl1.process({"u": raw_data}).values[0, 0]
    np.testing.assert_allclose(v_spl[interior], v_exact[interior], rtol=1e-3)

    # 3. Finite differences 2nd derivative (acceleration)
    diff_fd2 = fx.ProcessDifferentiateTime(
        source="u", order=2, method="finite_diff"
    )
    a_fd = diff_fd2.process({"u": raw_data}).values[0, 0]
    np.testing.assert_allclose(a_fd[interior], a_exact[interior], rtol=2e-2)


def test_temporal_integration_analytic() -> None:
    """Verifies cumulative time integration:
    v(t) = A*w*cos(w*t) -> u(t) = A*sin(w*t) + u_0.
    """
    omega = 2.0 * np.pi * 3.0
    amp = 2.5
    u0 = 10.0
    times = np.linspace(0.0, 1.0, 600)

    v_exact = amp * omega * np.cos(omega * times)
    u_exact = amp * np.sin(omega * times) + u0

    raw_data = fx.MeasurementData(
        values=v_exact[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("v",),
        units="mm/s",
    )

    integrator = fx.ProcessIntegrateTime(
        source="v", initial_value=u0, label="displacement", units="mm"
    )
    res = integrator.process({"v": raw_data})
    u_calc = res.values[0, 0]

    np.testing.assert_allclose(u_calc, u_exact, rtol=1e-3, atol=1e-3)
    assert res.components == ("displacement",)
    assert res.units == "mm"


def test_spatial_slope_integration_analytic() -> None:
    """Verifies spatial integration of surface slope theta(x) = c*x
    to recover quadratic beam deflection u(x) = 0.5*c*x^2 + u0.
    """
    c = 0.05
    u0 = 2.0
    x_coords = np.linspace(0.0, 100.0, 50)
    pos = np.column_stack(
        [x_coords, np.zeros_like(x_coords), np.zeros_like(x_coords)]
    )

    theta_exact = c * x_coords
    u_exact = 0.5 * c * (x_coords**2) + u0

    raw_data = fx.MeasurementData(
        values=theta_exact[:, np.newaxis, np.newaxis],
        sample_times=np.array([0.0]),
        positions=pos,
        components=("slope",),
        units="rad",
    )

    integrator = fx.ProcessIntegrateSpatial(
        source="slope", initial_value=u0, label="deflection", units="mm"
    )
    res = integrator.process({"slope": raw_data})
    u_calc = res.values[:, 0, 0]

    np.testing.assert_allclose(u_calc, u_exact, rtol=1e-3, atol=1e-3)
    assert res.components == ("deflection",)


def test_spatial_strain_reconstruction_analytic() -> None:
    """Verifies spatial strain reconstruction from a 2D array of displacement
    sensors under a known quadratic displacement field:
    u_x = 0.002*x^2 + 0.001*x*y
    u_y = 0.003*y^2 + 0.004*x*y
    Exact Strains:
    eps_xx = du_x/dx = 0.004*x + 0.001*y
    eps_yy = du_y/dy = 0.006*y + 0.004*x
    eps_xy = 0.5*(du_x/dy + du_y/dx) = 0.5*(0.001*x + 0.004*y)
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 10.0, 5), np.linspace(0.0, 10.0, 5)
    )
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    z_flat = np.zeros_like(x_flat)
    pos = np.column_stack([x_flat, y_flat, z_flat])

    u_x = 0.002 * (x_flat**2) + 0.001 * x_flat * y_flat
    u_y = 0.003 * (y_flat**2) + 0.004 * x_flat * y_flat

    disp_vals = np.stack([u_x, u_y], axis=1)[:, :, np.newaxis]

    raw_data = fx.MeasurementData(
        values=disp_vals,
        sample_times=np.array([0.0]),
        positions=pos,
        components=("u_x", "u_y"),
        units="mm",
    )

    eval_pts = np.array([
        [2.0, 3.0, 0.0],
        [5.0, 5.0, 0.0],
        [8.0, 4.0, 0.0],
    ])

    strain_proc = fx.ProcessSpatialStrain(
        source="disp",
        poly_degree=2,
        eval_positions=eval_pts,
        spatial_dims="2D",
    )
    res = strain_proc.process({"disp": raw_data})
    strains = res.values[:, :, 0]

    x_e = eval_pts[:, 0]
    y_e = eval_pts[:, 1]
    eps_xx_exact = 0.004 * x_e + 0.001 * y_e
    eps_yy_exact = 0.006 * y_e + 0.004 * x_e
    eps_xy_exact = 0.5 * (0.001 * x_e + 0.004 * y_e)

    np.testing.assert_allclose(
        strains[:, 0], eps_xx_exact, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        strains[:, 1], eps_yy_exact, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        strains[:, 2], eps_xy_exact, rtol=1e-4, atol=1e-6
    )


def test_multisensor_stiffness_and_work_analytic() -> None:
    """Verifies stiffness and work calculations for a linear elastic spring
    with stiffness k = 250 N/mm.
    """
    k_spring = 250.0
    times = np.linspace(0.0, 2.0, 200)
    u_vals = 0.5 * times**2
    f_vals = k_spring * u_vals

    work_exact = 0.5 * k_spring * (u_vals**2)

    force_data = fx.MeasurementData(
        values=f_vals[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("force",),
        units="N",
    )

    disp_data = fx.MeasurementData(
        values=u_vals[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("disp",),
        units="mm",
    )

    # 1. Stiffness
    stiff_proc = fx.ProcessStiffness(force="F", disp="u", eps=1e-6)
    stiff_res = stiff_proc.process({"F": force_data, "u": disp_data})
    np.testing.assert_allclose(
        stiff_res.values[0, 0, 10:], k_spring, rtol=1e-3
    )

    # 2. Work
    work_proc = fx.ProcessWork(force="F", disp="u", units="mJ")
    work_res = work_proc.process({"F": force_data, "u": disp_data})
    np.testing.assert_allclose(
        work_res.values[0, 0], work_exact, rtol=1e-3, atol=1e-3
    )


def test_postprocess_dag_and_custom() -> None:
    """Verifies DAG compilation, topological execution, and custom user-defined
    processors.
    """
    times = np.linspace(0.0, 1.0, 100)
    u_vals = np.sin(2.0 * np.pi * times)[np.newaxis, np.newaxis, :]
    f_vals = (100.0 * np.sin(2.0 * np.pi * times))[np.newaxis, np.newaxis, :]
    positions = np.zeros((1, 3))

    u_data = fx.MeasurementData(
        values=u_vals,
        sample_times=times,
        positions=positions,
        components=("u",),
        units="mm",
    )
    f_data = fx.MeasurementData(
        values=f_vals,
        sample_times=times,
        positions=positions,
        components=("f",),
        units="N",
    )

    def custom_energy(
        inputs: dict[str, fx.MeasurementData],
        factor: float = 1.0,
    ) -> fx.MeasurementData:
        u_arr = inputs["disp_smooth"].values
        f_arr = inputs["force_raw"].values
        e_arr = factor * u_arr * f_arr
        return fx.MeasurementData(
            values=e_arr,
            sample_times=inputs["disp_smooth"].sample_times,
            positions=inputs["disp_smooth"].positions,
            components=("energy",),
            units="mJ",
        )

    dag = fx.PostProcessGraph()
    dag.add_processor(
        "disp_smooth",
        fx.ProcessFilterSavitzkyGolay(
            source="disp_raw", window_length=9, polyorder=2
        ),
    )
    dag.add_processor(
        "velocity",
        fx.ProcessDifferentiateTime(source="disp_smooth", order=1, label="vel"),
    )
    dag.add_processor(
        "custom_e",
        fx.ProcessCustom(
            sources={"disp": "disp_smooth", "force": "force_raw"},
            func=custom_energy,
            factor=0.5,
            output_components=("energy",),
        ),
    )

    assert dag.get_execution_order() == ("disp_smooth", "velocity", "custom_e")

    results = dag.execute({"disp_raw": u_data, "force_raw": f_data})

    assert "disp_smooth" in results
    assert "velocity" in results
    assert "custom_e" in results
    assert results["custom_e"].components == ("energy",)
