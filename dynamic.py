import random
from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Any

import numpy as np
import networkx as nx
import time

import sympy
import torch
import math

from Helper import (
    ms,
    GraphCharacter,
    Primes,
    size_of_cell,
    np_array_to_tensor_mapping,
    mapping_row_columns,
    show_adjacency_tensor,
)
from count_graph import CountGraph
from gpu_checker import requires_gpu
from optimized_multiplication import (
    pow_number,
    apply_single_column_update,
    apply_single_row_update,
    multiply_sparse_cols_C,
    multiply_sparse_rows_D,
    modular_inverse_matrix_gpu,
)


class Dynamic(CountGraph, ABC):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "32",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)
        self._col_row_to_change_flag = -1
        self._m_matrix = self._create_matrix_from_scratch(self._adj_tensor.shape[0], 0)
        self._max_changes = int(math.sqrt(self._graph.number_of_nodes()))
        self._p = Primes[type_of_data]
        self._graph_n_adj = self._pow_matrix(
            self._graph.number_of_nodes(), self._adj_tensor
        )
        self._tol = 1e-8

    def _zeros(self, size: int) -> torch.Tensor:
        return torch.zeros(
            size, device="cuda", dtype=np_array_to_tensor_mapping(self._type_of_data)
        )

    def _eye(self, size: int) -> torch.Tensor:
        return torch.eye(
            size,
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )

    def _set_numbers_in_matrix(self):
        n = self._adj_tensor.shape[0]
        for i in range(n):
            for j in range(n):
                if self._adj_tensor[i][j] != 0:
                    self._adj_tensor[i][j] = random.randint(1, self._p - 1)

    def _pow_matrix(self, n: int, tensor: torch.tensor) -> torch.tensor:
        curr_state = 1
        eye = self._eye(tensor.shape[0])
        tensor = tensor + eye
        base = tensor.clone()
        result = tensor.clone()
        while curr_state < n:
            result = torch.matmul(result, (eye + base) % self._p) % self._p
            base = torch.matrix_power(base, 2) % self._p
            curr_state = curr_state * 2
        return result

    def _create_matrix_from_scratch(self, n: int, value: int):
        return torch.full(
            (n, n),
            value,
            dtype=np_array_to_tensor_mapping(self._type_of_data),
            device="cuda",
        )

    def _check_axis(self, axis: int) -> int:
        invalid_axis = -1
        if isinstance(axis, str):
            invalid_axis = mapping_row_columns.get(axis)
            if axis is None:
                invalid_axis = -1
        elif isinstance(axis, int):
            invalid_axis = axis if axis in {0, 1} else -1
        else:
            invalid_axis = -1
        return invalid_axis

    def _add_primes_to_vector(self, data: torch.Tensor) -> None:
        return
        # for i in range(data.size(0)):
        #     if data[i].item() != 0:
        #         data[i] = random.randint(1, self._p - 1)

    def _powik(self, base: torch.Tensor, power: int) -> torch.Tensor:
        res = self._zeros(1)
        res[0] = 1
        while power > 0:
            if power % 2:
                res = res * base
                res %= self._p
            power //= 2
            base = base * base
            base = base % self._p
        return res

    @abstractmethod
    def find_path(self, s: int, t: int) -> bool:
        pass

    @abstractmethod
    def update_one_cell(self, i : int, j : int, value : Any):
        pass
    def begin(self,i: int, j: int, value: Any):
        tensor = self._zeros(self._adj_tensor.shape[0])
        tensor[i] = value
        binary_tensor = self._zeros(self._adj_tensor.shape[0])
        binary_tensor[j] = 1
        self._graph_n_adj = self._begin_helper(
            self._graph_n_adj,
            tensor.clone(),
            binary_tensor,
        )
        self._adj_tensor[:, j] += tensor
        self._adj_tensor[:, j] %= self._p


    def _begin_helper(self, graph_n: torch.Tensor, u: torch.Tensor, v_t: torch.Tensor
    ):
        if u.dim() == 1:
            u = u.view(-1, 1)
        if v_t.dim() == 1:
            v_t = v_t.view(1, -1)
        Au = torch.matmul(graph_n, u) % self._p
        vA = torch.matmul(v_t, graph_n) % self._p
        denom = 1.0 + torch.matmul(v_t, Au).squeeze().item()
        denom %= self._p
        denom = pow_number(denom, self._p - 2, self._p)
        outer = torch.matmul(Au, vA) % self._p
        outer = (graph_n - ((outer * denom) % self._p) + self._p) % self._p
        return outer

