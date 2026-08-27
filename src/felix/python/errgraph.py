# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Directed Acyclic Graph (DAG) error integration engine in Felix."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import enum
from graphlib import CycleError, TopologicalSorter
from typing import Callable, Sequence

import numpy as np

import felix.cython.felix as fc
from felix.python.enums import EErrDep, EErrType
from felix.python.errspecs import IErrSimulator
from felix.python.sensordata import SensorData


class EErrOp(enum.Enum):
    """Operation used to combine node outputs with incoming dependencies."""

    ADD = enum.auto()
    MULTIPLY = enum.auto()
    REPLACE = enum.auto()
    CUSTOM = enum.auto()


@dataclass(slots=True)
class SignalState:
    """Carries measurement values and sensor geometric parameters."""

    values: np.ndarray
    sensor_data: SensorData


@dataclass(slots=True)
class ErrNode:
    """A single computation node in the sensor error DAG."""

    name: str
    simulator: IErrSimulator
    inputs: tuple[str, ...] = ()
    op: EErrOp = EErrOp.ADD
    custom_op: (
        Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
    ) = None


@dataclass(slots=True)
class ErrGraphOpts:
    """Options controlling error graph execution and caching."""

    store_node_outputs: bool = False
    force_dependence: EErrDep | None = None


class ErrGraph:
    """Directed Acyclic Graph (DAG) sensor error integration engine."""

    __slots__ = (
        "_nodes",
        "_execution_order",
        "_meas_shape",
        "_sens_data_initial",
        "_sens_data_accumulated",
        "_opts",
        "_errs_systematic",
        "_errs_random",
        "_errs_total",
        "_node_outputs",
        "_node_errors",
    )

    def __init__(
        self,
        nodes: Sequence[ErrNode],
        meas_shape: tuple[int, int, int],
        sensor_data_initial: SensorData,
        opts: ErrGraphOpts | None = None,
    ) -> None:
        self._nodes: dict[str, ErrNode] = {n.name: n for n in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError("ErrNode names in ErrGraph must be unique.")

        self._meas_shape = meas_shape
        self._sens_data_initial = copy.deepcopy(sensor_data_initial)
        self._sens_data_accumulated = copy.deepcopy(sensor_data_initial)
        self._opts = opts if opts is not None else ErrGraphOpts()

        if self._opts.force_dependence is not None:
            for node in self._nodes.values():
                node.simulator.set_error_dep(self._opts.force_dependence)

        self._execution_order = self._compile_graph()

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)

        self._node_outputs: dict[str, SignalState] | None = (
            {} if self._opts.store_node_outputs else None
        )
        self._node_errors: dict[str, np.ndarray] | None = (
            {} if self._opts.store_node_outputs else None
        )

    def _compile_graph(self) -> tuple[str, ...]:
        for name, node in self._nodes.items():
            for inp in node.inputs:
                if inp not in self._nodes:
                    raise KeyError(
                        f"ErrNode '{name}' references non-existent "
                        f"parent '{inp}'."
                    )

        graph_dict = {
            name: set(node.inputs) for name, node in self._nodes.items()
        }
        try:
            sorter = TopologicalSorter(graph_dict)
            return tuple(sorter.static_order())
        except CycleError as exc:
            raise ValueError(f"Error graph contains cycle: {exc}") from exc

    def to_spec_dict(self) -> dict:
        """Converts graph definition to contiguous format for Zig."""
        name_to_idx = {name: ii for ii, name in enumerate(self._nodes.keys())}
        node_dicts = []
        for name, node in self._nodes.items():
            op_code = 0
            if node.op == EErrOp.MULTIPLY:
                op_code = 1
            elif node.op == EErrOp.REPLACE:
                op_code = 2

            input_indices = [name_to_idx[inp] for inp in node.inputs]
            spec_dict = node.simulator.to_spec_dict()
            node_dicts.append(
                {
                    "name": name,
                    "op": op_code,
                    "inputs": input_indices,
                    "spec": spec_dict,
                }
            )

        exec_indices = [name_to_idx[name] for name in self._execution_order]
        leaf_names = [
            n
            for n in self._nodes
            if not any(n in other.inputs for other in self._nodes.values())
        ]
        leaf_indices = [name_to_idx[name] for name in leaf_names]

        return {
            "nodes": node_dicts,
            "execution_order": exec_indices,
            "leaf_indices": leaf_indices,
            "store_node_outputs": self._opts.store_node_outputs,
        }

    def reseed_error_graph(self, seed: int | None = None) -> None:
        for node in self._nodes.values():
            node.simulator.reseed(seed)

    def reseed(self, seed: int | None = None) -> None:
        self.reseed_error_graph(seed)

    def reseed_error_chain(self, seed: int | None = None) -> None:
        self.reseed_error_graph(seed)

    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        return self.calc_errors_from_graph(truth)

    def set_sensor_data_initial(self, sensor_data: SensorData) -> None:
        self._sens_data_initial = copy.deepcopy(sensor_data)
        self._sens_data_accumulated = copy.deepcopy(sensor_data)

    def calc_errors_from_graph(self, truth: np.ndarray) -> np.ndarray:
        self._sens_data_accumulated = copy.deepcopy(self._sens_data_initial)
        graph_dict = self.to_spec_dict()

        num_sensors = truth.shape[0]
        num_comps = truth.shape[1]
        num_times = truth.shape[2]

        coords = np.zeros((3, 3), dtype=np.float64)
        connect = np.array([[0, 1, 2]], dtype=np.uint64)
        nodal_fields = np.zeros((3, num_comps, 1), dtype=np.float64)
        sim_times = (
            self._sens_data_initial.sample_times
            if self._sens_data_initial.sample_times is not None
            else np.linspace(0.0, 1.0, num_times)
        )
        positions = (
            self._sens_data_initial.positions
            if self._sens_data_initial.positions is not None
            else np.zeros((num_sensors, 3), dtype=np.float64)
        )

        res = fc.simulate_err_graph(
            coords=coords,
            connect=connect,
            elem_type=0,
            nodal_fields=nodal_fields,
            sim_times=sim_times,
            positions=positions,
            sample_times=self._sens_data_initial.sample_times,
            rot_matrices=None,
            spatial_dims=2,
            is_tensor=False,
            graph_dict=graph_dict,
            truth=truth,
            num_experiments=1,
        )

        _, _, errs_sys, errs_rand, errs_total, node_outputs = res
        self._errs_systematic = errs_sys
        self._errs_random = errs_rand
        self._errs_total = errs_total

        if self._opts.store_node_outputs and node_outputs is not None:
            self._node_outputs = {}
            for name, node_idx in zip(
                self._nodes.keys(), range(len(self._nodes))
            ):
                self._node_outputs[name] = SignalState(
                    values=node_outputs[node_idx],
                    sensor_data=copy.deepcopy(self._sens_data_initial),
                )

        return self._errs_total

    def _combine_parent_states(
        self,
        parents: Sequence[SignalState],
        truth: np.ndarray,
    ) -> SignalState:
        combined_values = truth.copy()
        for p in parents:
            combined_values += p.values - truth
        combined_sens = copy.deepcopy(parents[-1].sensor_data)
        return SignalState(values=combined_values, sensor_data=combined_sens)

    def _apply_operator(
        self,
        node: ErrNode,
        in_state: SignalState,
        error_array: np.ndarray,
        sens_perturbed: SensorData,
    ) -> SignalState:
        if node.op == EErrOp.ADD:
            out_values = in_state.values + error_array
            return SignalState(
                values=out_values,
                sensor_data=copy.deepcopy(sens_perturbed),
            )
        if node.op == EErrOp.MULTIPLY:
            out_values = in_state.values * (1.0 + error_array)
            return SignalState(
                values=out_values,
                sensor_data=copy.deepcopy(sens_perturbed),
            )
        if node.op == EErrOp.REPLACE:
            out_values = error_array.copy()
            return SignalState(
                values=out_values,
                sensor_data=copy.deepcopy(sens_perturbed),
            )
        if node.op == EErrOp.CUSTOM:
            if node.custom_op is None:
                raise ValueError("Custom op callable required for CUSTOM op")
            return node.custom_op(in_state, error_array, sens_perturbed)

        raise ValueError(f"Unsupported EErrOp: {node.op}")

    @property
    def nodes(self) -> dict[str, ErrNode]:
        return self._nodes

    @property
    def execution_order(self) -> tuple[str, ...]:
        return self._execution_order

    def get_errs_systematic(self) -> np.ndarray:
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        return self._errs_total

    def get_sens_data_accumulated(self) -> SensorData:
        return self._sens_data_accumulated

    def get_node_output(self, name: str) -> SignalState:
        if self._node_outputs is None:
            raise RuntimeError("store_node_outputs was False.")
        return self._node_outputs[name]

    def get_node_outputs(self) -> dict[str, SignalState] | None:
        return self._node_outputs

    def get_node_error(self, name: str) -> np.ndarray:
        if self._node_errors is None:
            raise RuntimeError("store_node_outputs was False.")
        return self._node_errors[name]

    def get_node_errors(self) -> dict[str, np.ndarray] | None:
        return self._node_errors


class ErrGraphBuilder:
    """Fluent builder for constructing ErrGraph instances."""

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: list[ErrNode] = []

    def add_root(
        self,
        name: str,
        simulator: IErrSimulator,
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=(),
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def add_child(
        self,
        name: str,
        simulator: IErrSimulator,
        parent: str,
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=(parent,),
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def add_node(
        self,
        name: str,
        simulator: IErrSimulator,
        inputs: tuple[str, ...],
        op: EErrOp = EErrOp.ADD,
        custom_op: (
            Callable[[SignalState, np.ndarray, SensorData], SignalState] | None
        ) = None,
    ) -> ErrGraphBuilder:
        self._nodes.append(
            ErrNode(
                name=name,
                simulator=simulator,
                inputs=inputs,
                op=op,
                custom_op=custom_op,
            )
        )
        return self

    def build(
        self,
        meas_shape: tuple[int, int, int],
        sensor_data_initial: SensorData,
        opts: ErrGraphOpts | None = None,
    ) -> ErrGraph:
        return ErrGraph(
            nodes=self._nodes,
            meas_shape=meas_shape,
            sensor_data_initial=sensor_data_initial,
            opts=opts,
        )


def err_chain_to_graph(
    err_chain: Sequence[IErrSimulator],
    meas_shape: tuple[int, int, int],
    sensor_data_initial: SensorData,
    opts: ErrGraphOpts | None = None,
) -> ErrGraph:
    """Converts a sequential error chain list to a linear ErrGraph."""
    nodes: list[ErrNode] = []
    for ii, sim in enumerate(err_chain):
        parent = () if ii == 0 else (f"node_{ii-1}",)
        nodes.append(
            ErrNode(
                name=f"node_{ii}",
                simulator=sim,
                inputs=parent,
                op=EErrOp.ADD,
            )
        )
    return ErrGraph(
        nodes=nodes,
        meas_shape=meas_shape,
        sensor_data_initial=sensor_data_initial,
        opts=opts,
    )
