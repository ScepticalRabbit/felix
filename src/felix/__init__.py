# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from felix import sensorsim
from felix.sensorsim import (
    SensorsPoint,
    SensorData,
    SensorDescriptor,
    FieldScalar,
    FieldVector,
    FieldTensor,
    ErrIntegrator,
    ExperimentSimulator,
)

__version__ = "2026.8.0"

__all__ = [
    "sensorsim",
    "SensorsPoint",
    "SensorData",
    "SensorDescriptor",
    "FieldScalar",
    "FieldVector",
    "FieldTensor",
    "ErrIntegrator",
    "ExperimentSimulator",
]
