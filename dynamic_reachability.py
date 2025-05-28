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
    mapping_row_columns,
    modular_inverse_matrix_gpu,
)
from count_graph import CountGraph
from gpu_checker import requires_gpu


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
        self._max_changes = int(math.sqrt(self._graph.number_of_nodes()))
        self._p = Primes[type_of_data]
        self._symbol_matrix = self._create_sym_matrix()
        self._set_numbers_in_matrix()
        self._graph_n_adj = self._pow_matrix(self._graph.number_of_nodes())
        print(self._graph_n_adj)

    def _symbol_string(self, i: int, j: int) -> str:
        return f"X_{i}_{j}"

    def _symbol_basic(self, i: int, j: int) -> sympy.Basic:
        return sympy.Symbol(self._symbol_string(i, j))

    def _set_numbers_in_matrix(self):
        n = self._adj_tensor.shape[0]
        for i in range(n):
            for j in range(n):
                if self._symbol_matrix[i][j] != 0:
                    self._adj_tensor[i][j] = random.randint(1, self._p - 1)

    def _create_sym_matrix(self) -> List[List[sympy.Basic]]:
        n = self._adj_tensor.shape[0]
        symbol_matrix = [[None for _ in range(n)] for _ in range(n)]
        self._symbols_mapping[sympy.S(0)] = 0
        for i in range(n):
            for j in range(n):
                if self._adj_tensor[i, j] == 1:
                    symbol_matrix[i][j] = sympy.Symbol(self._symbol_string(i, j))
                    self._symbols_mapping[symbol_matrix[i][j]] = 1
                else:
                    symbol_matrix[i][j] = sympy.S(0)  # 0 jako symboliczna liczba zero

        return symbol_matrix

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

    def update_row_or_col(
        self, new_data: torch.Tensor, indx: int, axis: str | int = "row"
    ):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
        if len(self._row_to_change) > 0 and self._col_row_to_change_flag != axis:
            self._update_matrix()

        invalid_axis = False

        if isinstance(axis, str):
            axis = mapping_row_columns.get(axis)
            if axis is None:
                invalid_axis = True
        elif isinstance(axis, int):
            if axis not in {0, 1}:
                invalid_axis = True
        else:
            invalid_axis = True

        if invalid_axis:
            raise ValueError('axis must be either "row" or "col"')
        mask = (new_data != 0).to(np_array_to_tensor_mapping(self._type_of_data))
        self._col_row_to_change_flag = axis
        if axis == 0:  # row
            for i in range(mask.size(0)):
                if mask[i, 0] != 0:
                    self._add_to_index_set(new_data, i, indx, i)
            new_data = new_data.T
            self._add_to_changes(mask, new_data)
        elif axis == 1:
            for i in range(mask.size(0)):
                if mask[i, 0] != 0:
                    self._add_to_index_set(new_data, indx, i, i)
            mask = mask.T
            self._add_to_changes(new_data, mask)
        if len(self._row_to_change) > self._max_changes:
            self._update_matrix()

    def _add_to_index_set(self, new_data: torch.Tensor, i: int, k: int, l: int) -> None:
        if (k, i) not in self._to_change_index_set:
            self._to_change_index_set.add((k, i))
            new_data[l, 0].sub_(self._graph_n_adj[k][i]).remainder(self._p)

    def _add_to_changes(self, row: torch.Tensor, col: torch.Tensor):
        self._row_to_change.append(row)
        self._col_to_change.append(col)

    def _create_matrixes_from_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix_col = torch.cat(self._col_to_change, dim=1)
        matrix_row = torch.cat(self._row_to_change, dim=0)
        return matrix_col, matrix_row

    def _clear_to_change_lists(self):
        self._row_to_change.clear()
        self._col_to_change.clear()
        self._to_change_index_set.clear()

    def _update_matrix(self):
        matrix_col, matrix_row = self._create_matrixes_from_vectors()
        t1 = torch.matmul(self._graph_n_adj, matrix_col).remainder(self._p)
        t2 = torch.matmul(matrix_row, t1).remainder(self._p)
        eye = torch.eye(
            self._adj_tensor.shape[0],
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
        t3 = (eye + t2).remainder(self._p)
        t3 = modular_inverse_matrix_gpu(t3, self._p)
        t4 = torch.matmul(t1, t3).remainder(self._p)
        t5 = torch.matmul(t4, matrix_row).remainder(self._p)
        self._graph_n_adj.sub_(t5).remainder(self._p)
        self._clear_to_change_lists()
        self._col_row_to_change_flag = -1

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

    #
    # def _sherman_morrison(self,) -> torch.Tensor:
    #     if vector is None:
    #         raise ValueError("vector cannot be None.")

    def find_path(self, s: int, t: int) -> bool:
        adj_matrix = self._graph_n_adj.clone()
        if self._col_to_change is not None:
            matrix_col, matrix_row = self._create_matrixes_from_vectors()
            result = torch.matmul(matrix_col, matrix_row).remainder(self._p)
            adj_matrix.add_(result).remainder(self._p)

        return adj_matrix[s, t].item() != 0
