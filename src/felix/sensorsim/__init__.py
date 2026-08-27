# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from felix.sensorsim.enums import (
    EDim,
    EErrType,
    EErrDep,
    ERoundMethod,
)
from felix.sensorsim.sensordata import SensorData
from felix.sensorsim.sensordescriptor import SensorDescriptor
from felix.sensorsim.descriptorfactory import DescriptorFactory
from felix.sensorsim.sensorfactory import SensorFactory
from felix.sensorsim.field import IField
from felix.sensorsim.fieldscalar import FieldScalar
from felix.sensorsim.fieldvector import FieldVector
from felix.sensorsim.fieldtensor import FieldTensor
from felix.sensorsim.generatorsrandom import (
    IGenRandom,
    GenUniform,
    GenNormal,
    GenTriangular,
    GenExponential,
    GenGamma,
    GenBeta,
    GenLogNormal,
)
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.errorsysindep import (
    ErrSysOffset,
    ErrSysOffsetPercent,
    ErrSysGen,
    ErrSysGenPercent,
)
from felix.sensorsim.errorsysdep import (
    ErrSysRoundOff,
    ErrSysDigitisation,
    ErrSysSaturation,
)
from felix.sensorsim.errorrand import (
    ErrRandGen,
    ErrRandGenPercent,
)
from felix.sensorsim.errorsyscalib import ErrSysCalibration
from felix.sensorsim.errordriftcalc import (
    IDriftCalculator,
    DriftConstant,
    DriftLinear,
    DriftPolynomial,
)
from felix.sensorsim.errorsysfield import (
    ErrFieldData,
    ErrSysField,
)
from felix.sensorsim.errorintegrator import (
    ErrIntOpts,
    ErrIntegrator,
)
from felix.sensorsim.sensorspoint import SensorsPoint
from felix.sensorsim.experimentsimulator import (
    EExpSimPara,
    ExpSimSaveKeys,
    ExpSimOpts,
    ExperimentSimulator,
)
from felix.sensorsim.experimentstats import (
    ExpSimStats,
    calc_exp_sim_stats,
)
from felix.sensorsim.sensortools import (
    get_sim_dims,
    scale_length_units,
    gen_pos_grid_inside,
    gen_pos_grid_boundary,
    gen_pos_cylinder,
    gen_pos_sphere,
    orient_from_direction,
    orient_from_normal,
    orient_from_normal_and_tangent,
    print_measurements,
)
from felix.sensorsim import simtools

# Aliases for compatibility
ISensorArray = SensorsPoint

from felix.sensorsim.simtools import (
    print_sim_data,
    print_dataclass_fields,
)

# Forward visualization and helper utilities from pyvale
try:
    import pyvista as pv

    pv.OFF_SCREEN = True

    from pyvale.sensorsim.visualsimsensors import (
        plot_point_sensors_on_sim,
        save_pv_image,
        add_sensor_points_nom,
        add_sensor_points_pert,
    )
    from pyvale.sensorsim.visualtraceplotter import (
        plot_time_traces,
    )
    from pyvale.sensorsim.visualexpplotter import (
        plot_exp_traces,
    )
    from pyvale.sensorsim.visualopts import (
        TraceOptsExperiment,
        TraceOptsSensor,
        VisOptsSimSensors,
        VisOptsImageSave,
        PlotOptsGeneral,
    )
    from pyvale.sensorsim.experimentsimio import (
        save_exp_sim_data,
        load_exp_sim_data,
    )
except ImportError:
    TraceOptsExperiment = None
    TraceOptsSensor = None
    VisOptsSimSensors = None
    VisOptsImageSave = None
    PlotOptsGeneral = None
    plot_point_sensors_on_sim = None
    save_pv_image = None
    add_sensor_points_nom = None
    add_sensor_points_pert = None
    plot_time_traces = None
    plot_exp_traces = None
    save_exp_sim_data = None
    load_exp_sim_data = None

__all__ = [
    "EDim",
    "EErrType",
    "EErrDep",
    "ERoundMethod",
    "SensorData",
    "SensorDescriptor",
    "DescriptorFactory",
    "SensorFactory",
    "IField",
    "FieldScalar",
    "FieldVector",
    "FieldTensor",
    "IGenRandom",
    "GenUniform",
    "GenNormal",
    "GenTriangular",
    "GenExponential",
    "GenGamma",
    "GenBeta",
    "GenLogNormal",
    "IErrSimulator",
    "ErrSysOffset",
    "ErrSysOffsetPercent",
    "ErrSysGen",
    "ErrSysGenPercent",
    "ErrSysRoundOff",
    "ErrSysDigitisation",
    "ErrSysSaturation",
    "ErrRandGen",
    "ErrRandGenPercent",
    "ErrSysCalibration",
    "IDriftCalculator",
    "DriftConstant",
    "DriftLinear",
    "DriftPolynomial",
    "ErrFieldData",
    "ErrSysField",
    "ErrIntOpts",
    "ErrIntegrator",
    "SensorsPoint",
    "ISensorArray",
    "EExpSimPara",
    "ExpSimSaveKeys",
    "ExpSimOpts",
    "ExperimentSimulator",
    "ExpSimStats",
    "calc_exp_sim_stats",
    "get_sim_dims",
    "scale_length_units",
    "gen_pos_grid_inside",
    "gen_pos_grid_boundary",
    "gen_pos_cylinder",
    "gen_pos_sphere",
    "orient_from_direction",
    "orient_from_normal_and_tangent",
    "simtools",
    "plot_point_sensors_on_sim",
    "save_pv_image",
    "add_sensor_points_nom",
    "add_sensor_points_pert",
    "plot_time_traces",
    "plot_exp_traces",
    "print_sim_data",
    "TraceOptsExperiment",
    "TraceOptsSensor",
    "VisOptsSimSensors",
    "VisOptsImageSave",
    "PlotOptsGeneral",
    "save_exp_sim_data",
    "load_exp_sim_data",
]
