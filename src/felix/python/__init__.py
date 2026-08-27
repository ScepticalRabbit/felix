# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from felix.python.enums import (
    EDim,
    EDifferentialMode,
    EErrDep,
    EErrType,
    EIntegrationMode,
    ERayMode,
    ERoundMethod,
)
from felix.python.errspecs import (
    DriftConstant,
    DriftLinear,
    DriftPolynomial,
    ErrFieldData,
    ErrIntOpts,
    ErrRandGen,
    ErrRandGenPercent,
    ErrSysCalibration,
    ErrSysDigitisation,
    ErrSysField,
    ErrSysGen,
    ErrSysGenPercent,
    ErrSysOffset,
    ErrSysOffsetPercent,
    ErrSysRoundOff,
    ErrSysSaturation,
    GenBeta,
    GenExponential,
    GenGamma,
    GenLogNormal,
    GenNormal,
    GenTriangular,
    GenUniform,
    IDriftCalculator,
    IErrSimulator,
    IGenRandom,
)
from felix.python.errgraph import (
    EErrOp,
    ErrGraph,
    ErrGraphBuilder,
    ErrGraphOpts,
    ErrNode,
    SignalState,
    err_chain_to_graph,
)
from felix.python.experimentstats import ExpSimStats, calc_exp_sim_stats
from felix.python.experimentsimulator import (
    EExpSimPara,
    ExpSimOpts,
    ExpSimSaveKeys,
    ExperimentSimulator,
)
from felix.python.factories import DescriptorFactory, SensorFactory
from felix.python.fieldspecs import (
    FieldScalar,
    FieldTensor,
    FieldVector,
    IField,
)
from felix.python.sensordata import SensorData
from felix.python.sensordescriptor import SensorDescriptor
from felix.python.sensorspoint import ErrIntegrator, ISensorArray, SensorsPoint
from felix.python.sensorsspatial import SensorsSpatial
from felix.python.sensorsray import SensorsRay
from felix.python.sensorsdifferential import SensorsDifferential
from felix.python.spatialwindows import (
    ISpatialWindow,
    SpatialWindowBox,
    SpatialWindowBox3D,
    SpatialWindowCircle,
    SpatialWindowCircle2D,
    SpatialWindowLine,
    SpatialWindowLine1D,
    SpatialWindowPoint,
    SpatialWindowRectangle,
    SpatialWindowRect2D,
    SpatialWindowSphere,
    SpatialWindowSphere3D,
)
from felix.python.temporalwindows import (
    ITemporalKernel,
    ITemporalWindow,
    TemporalWindowExponential,
    TemporalWindowGaussian,
    TemporalWindowInstant,
    TemporalWindowRectangular,
)
from felix.python.spatialkernels import (
    ISpatialKernel,
    SpatialKernelCosine,
    SpatialKernelCustom,
    SpatialKernelEpanechnikov,
    SpatialKernelGaussian,
    SpatialKernelTriangular,
    SpatialKernelUniform,
)
from felix.python.integrationrules import (
    IIntegrationRule,
    IntegrationGaussLegendre,
    IntegrationMidpoint,
    IntegrationMonteCarlo,
    IntegrationSimpson,
    IntegrationTrapezoidal,
)
from felix.python.sensortools import (
    gen_pos_cylinder,
    gen_pos_grid_boundary,
    gen_pos_grid_inside,
    gen_pos_sphere,
    get_sim_dims,
    orient_from_direction,
    orient_from_normal,
    orient_from_normal_and_tangent,
    print_dataclass_fields,
    print_dimensions,
    print_measurements,
    print_sim_data,
    scale_length_units,
)
from felix.python.visualisation import *


__all__ = [name for name in globals() if not name.startswith("_")]
