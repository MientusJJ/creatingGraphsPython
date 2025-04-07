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
)
from count_graph import CountGraph
from gpu_checker import requires_gpu


class DynamicReachAbility(CountGraph):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "64",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)
        self._symbols_mapping: Dict[sympy.Basic, int] = {}
        self._col_row_to_change: List[Tuple[torch.Tensor, int, int]] = []
        self._max_changes = 0 #int(math.sqrt(self._graph.number_of_nodes()))
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
        self, new_data: torch.Tensor, index: int, axis: str | int = "row"
    ):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
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

        self._col_row_to_change.append((new_data, index, axis))
        if len(self._col_row_to_change) > self._max_changes:
            self._update_matrix()

    def _update_matrix(self):
        for i in range(len(self._col_row_to_change)):
                index = self._col_row_to_change[i][1]
                axis = self._col_row_to_change[i][2]
                data = self._col_row_to_change[i][0]
                for j in range(len(data)):
                    sym =  self._symbol_basic(j,index) if axis == 1 else self._symbol_string(index,j)
                    if data[j].item() == 0 and  sym in self._symbols_mapping:
                        self._symbols_mapping.pop(sym, None)
                    elif data[j].item() != 0:
                        self._symbols_mapping[sym] = 1
                self._add_primes_to_vector(data)
                if axis == 0:
                    self._adj_tensor[index] = data
                else:
                    self._adj_tensor[:, index] = data
        self._col_row_to_change.clear()
        self._graph_n_adj = self._pow_matrix(self._graph.number_of_nodes())
    def _add_primes_to_vector(self, data: torch.Tensor) -> None:
        for i in range(data.size(0)):
            if data[i].item() != 0:
                data[i] = random.randint(1, self._p - 1)

    def _powik(self, base: torch.Tensor, power: int) -> torch.Tensor:
        res = torch.zeros(1, device="cuda", dtype=torch.float64)
        res[0] = 1
        while power > 0:
            if power % 2:
                res = res * base
                res %= self._p
            power //= 2
            base = base * base
            base = base % self._p
        return res

    def _sherman_morrison(self, vector: Tuple[torch.Tensor, int, int]) -> torch.Tensor:
        if vector is None:
            raise ValueError("vector cannot be None.")

        data_vector, index, axis = vector
        data = data_vector.clone()
        self._add_primes_to_vector(data)
        changed_column = torch.zeros(
            self._adj_tensor.shape[0], device="cuda", dtype=torch.float64
        )
        changed_column[index] = 1
        if axis == 0:
            changed_column = changed_column.view(-1, 1)
        else:
            data = data.view(-1, 1)
        A_u = torch.matmul(self._graph_n_adj, data) % self._p
        v_A = torch.matmul(changed_column, self._graph_n_adj) % self._p

        print(A_u)
        print(v_A)
        vAu_scalar = (1 + (torch.matmul(v_A, data) % self._p)) % self._p
        vAu_scalar = self._powik(vAu_scalar, self._p - 2)
        result = ((torch.matmul(A_u, v_A.view(1, -1)) % self._p) * vAu_scalar) % self._p
        return result

    def find_path(self, s: int, t: int) -> bool:
        adj_matrix = self._graph_n_adj.clone()
        if self._col_row_to_change is not None:
            for vector in self._col_row_to_change:
                adj_matrix = (adj_matrix - self._sherman_morrison(vector)) % self._p

        return adj_matrix[s, t].item() != 0
