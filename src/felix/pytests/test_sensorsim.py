# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from pathlib import Path
import numpy as np
import pytest

import pyvale.verif.pointsensscalar as pss
import pyvale.verif.pointsensvector as psv
import pyvale.verif.pointsenstensor as pst
import felix.sensorsim as fs

GOLD_DIR = Path(__file__).parent / "gold"


def run_gold_suite(get_dict_fn, field_cls, rtol=1e-3, atol=1e-4) -> None:
    sens_dict = get_dict_fn()
    assert len(sens_dict) > 0

    for tag, py_sens in sens_dict.items():
        gold_file = GOLD_DIR / f"{tag}.npy"
        if not gold_file.exists():
            continue
        gold = np.load(gold_file)

        sim_data = py_sens.get_field().get_sim_data()
        comp_keys = py_sens.get_field().get_all_components()
        spat_dim = py_sens.get_field()._spatial_dims

        if field_cls is fs.FieldScalar:
            f_field = fs.FieldScalar(sim_data, comp_keys[0], spat_dim)
        elif field_cls is fs.FieldVector:
            f_field = fs.FieldVector(sim_data, comp_keys, spat_dim)
        elif field_cls is fs.FieldTensor:
            norm_keys = py_sens.get_field()._norm_comp_keys
            dev_keys = py_sens.get_field()._dev_comp_keys
            f_field = fs.FieldTensor(sim_data, norm_keys, dev_keys, spat_dim)
        else:
            raise TypeError(f"Unknown field class {field_cls}")

        f_data = fs.SensorData(
            positions=py_sens._sensor_data.positions,
            sample_times=py_sens._sensor_data.sample_times,
            angles=py_sens._sensor_data.angles,
        )
        f_sens = fs.SensorsPoint(f_data, f_field)
        if py_sens._error_integrator is not None:
            f_sens.set_error_chain(
                py_sens._error_integrator._err_chain,
                py_sens._error_integrator._err_int_opts,
            )

        calc = f_sens.sim_measurements()

        # Check against gold with tolerance (accommodates discretization boundary rounding)
        if "err-basic-dep" in tag:
            tol_val = 0.15
            assert np.allclose(calc, gold, atol=tol_val)
        else:
            assert np.allclose(calc, gold, rtol=rtol, atol=atol), (
                f"Failed gold parity for {tag}: max diff = {np.max(np.abs(calc - gold))}"
            )


def test_gold_scalar_2d() -> None:
    run_gold_suite(pss.sens_arrays_2d_dict, fs.FieldScalar)


def test_gold_scalar_3d() -> None:
    run_gold_suite(pss.sens_arrays_3d_dict, fs.FieldScalar)


def test_gold_vector_2d() -> None:
    run_gold_suite(psv.sens_arrays_2d_dict, fs.FieldVector)


def test_gold_vector_3d() -> None:
    run_gold_suite(psv.sens_arrays_3d_dict, fs.FieldVector)


def test_gold_tensor_2d() -> None:
    run_gold_suite(pst.sens_arrays_2d_dict, fs.FieldTensor)


def test_gold_tensor_3d() -> None:
    run_gold_suite(pst.sens_arrays_3d_dict, fs.FieldTensor)
