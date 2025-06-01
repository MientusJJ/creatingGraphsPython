import random
from typing import Tuple, List, Dict

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
    mapping_row_columns, show_adjacency_tensor,
)
from count_graph import CountGraph
from gpu_checker import requires_gpu
from optimized_multiplication import (
    pow_number,
    optimized_matmul_single_left,
    optimized_matmul_single_right, optimized_sparse_matmul_C, optimized_sparse_matmul_sparse_D,
)


class DynamicReachAbility(CountGraph):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "32",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)
        self._symbols_mapping: Dict[sympy.Basic, int] = {}
        self._col_to_change: List[torch.Tensor] = []
        self._to_change_index_set: set[Tuple[float, float]] = set()
        self._col_row_to_change_flag = -1
        self._row_to_change: List[torch.Tensor] = []
        self._m_matrix = self._create_matrix_from_scratch(self._adj_tensor.shape[0], 0)
        # self._max_changes = int(math.sqrt(self._graph.number_of_nodes()))
        self._max_changes = 2
        self._p = Primes[type_of_data]
        self._set_numbers_in_matrix()
        self._graph_n_adj = self._pow_matrix(self._graph.number_of_nodes())
        self._tol = 1e-8
        # print(self._graph_n_adj)

    def _set_numbers_in_matrix(self):
        n = self._adj_tensor.shape[0]
        for i in range(n):
            for j in range(n):
                if self._adj_tensor[i][j] != 0:
                    self._adj_tensor[i][j] = random.randint(1, self._p - 1)

    def _pow_matrix(self, n: int) -> torch.tensor:
        curr_state = 1

        eye = torch.eye(
            self._adj_tensor.shape[0],
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
        base = self._adj_tensor.clone()
        result = self._adj_tensor.clone()
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
            if axis not in {0, 1}:
                invalid_axis = -1
        else:
            invalid_axis = -1
        return invalid_axis

    def update_one_row_or_col(
        self, new_data: torch.Tensor, indx: int, axis: str | int = "row"
    ):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
        axis = self._check_axis(axis)
        if axis == -1:
            raise ValueError('axis must be either "row" or "col"')
        self._add_primes_to_vector(new_data)
        self._update_matrix()
        bx = self._inverse_one_vector(new_data, indx, axis, False)
        if axis == 0:
            self._graph_n_adj = optimized_matmul_single_right(
                self._graph_n_adj, bx, indx, self._p, False
            )
        elif axis == 1:
            self._graph_n_adj = optimized_matmul_single_left(
                bx, indx,self._graph_n_adj, self._p, True
            )

    def _inverse_one_vector(
        self, new_data: torch.Tensor, indx: int, axis: int, cell=True
    ):
        bi = new_data[indx].item() + 1
        denom = pow_number(bi, self._p - 2, self._p)
        new_data = new_data.to(np_array_to_tensor_mapping(self._type_of_data))
        bx = torch.empty_like(new_data)
        for j in range(len(new_data)):
            if j == indx:
                bx[j] = denom
            else:
                val = (-new_data[j].item() * denom) % self._p
                val = (val + self._p) % self._p
                bx[j] = val
        if axis == 0:
            bx = bx.view(1, -1)
            if cell:
                self._col_row_to_change_flag = 0
        else:
            bx = bx.view(-1, 1)
            if cell:
                self._col_row_to_change_flag = 1
        return bx

    def _update_matrix(self):
        self._graph_n_adj = self._multiplicationN1M()
        self._col_row_to_change_flag = -1
        self._m_matrix = self._create_matrix_from_scratch(self._adj_tensor.shape[0], 0)

    def update_one_cell_row_or_col(
        self, new_data: torch.Tensor, indx: int, axis: str | int = "row"
    ):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
        if len(self._row_to_change) > 0 and self._col_row_to_change_flag != axis:
            self._update_matrix()

        axis = self._check_axis(axis)
        if axis == -1:
            raise ValueError('axis must be either "row" or "col"')
        self._add_primes_to_vector(new_data)
        bx = self._inverse_one_vector(new_data, indx, axis, True)

        if axis == 1:
            self._m_matrix = optimized_matmul_single_left(
                bx, indx, self._m_matrix, self._p, True
            )
        elif axis == 0:
            self._m_matrix = optimized_matmul_single_right(
                self._m_matrix, bx, indx, self._p, False
            )
        if axis == 0:
            bx[0, indx] = (bx[0, indx] - 1) % self._p
        else:
            bx[indx, 0] = (bx[indx, 0] - 1) % self._p
        if axis == 0:
            #print(bx.shape, bx.dim())
            #print(self._m_matrix.shape,self._m_matrix.dim())
            self._m_matrix[indx, :] = (self._m_matrix[indx, :] + bx[0, :]) % self._p
        else:
            self._m_matrix[:, indx] = (self._m_matrix[:, indx] + bx[:, 0]) % self._p
        nonzero_rows = (self._m_matrix.abs().sum(dim=1) > self._tol).sum().item()

        nonzero_cols = (self._m_matrix.abs().sum(dim=0) > self._tol).sum().item()
        if nonzero_cols > self._max_changes or nonzero_rows > self._max_changes:
            self._update_matrix()

    def _add_primes_to_vector(self, data: torch.Tensor) -> None:
        for i in range(data.size(0)):
            if data[i].item() != 0:
                data[i] = random.randint(1, self._p - 1)

    def _powik(self, base: torch.Tensor, power: int) -> torch.Tensor:
        res = torch.zeros(
            1, device="cuda", dtype=np_array_to_tensor_mapping(self._type_of_data)
        )
        res[0] = 1
        while power > 0:
            if power % 2:
                res = res * base
                res %= self._p
            power //= 2
            base = base * base
            base = base % self._p
        return res

    def find_path(self, s: int, t: int) -> bool:
        return self._multiplicationN1M()[s, t].item() != 0

    def _multiplicationN1M(self) -> torch.Tensor:
        adj_matrix = self._m_matrix.clone()
        eye = torch.eye(
            self._adj_tensor.shape[0],
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
        adj_matrix.add_(eye)
        if  self._col_row_to_change_flag == 1:
            adj_matrix = optimized_sparse_matmul_C(adj_matrix,eye, self._graph_n_adj + eye, self._p)
        else:
            adj_matrix = optimized_sparse_matmul_sparse_D(self._graph_n_adj,eye, adj_matrix + eye, self._p)
        show_adjacency_tensor(adj_matrix)
        return adj_matrix
