import random
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
from dynamic import Dynamic
from gpu_checker import requires_gpu
from optimized_multiplication import (
    pow_number,
    apply_single_column_update,
    apply_single_row_update,
    multiply_sparse_cols_C,
    multiply_sparse_rows_D,
    modular_inverse_matrix_gpu,
    dense_to_sparse,
)


class DynamicReachAbility(Dynamic):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "32",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)

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
        print(bx.shape)
        if axis == 0:
            self._graph_n_adj = apply_single_row_update(
                self._graph_n_adj, indx, bx, self._p
            )
        elif axis == 1:

            self._graph_n_adj = apply_single_column_update(
                self._graph_n_adj, indx, bx, self._p
            )

    def _prepare_new_new_data(
        self, new_data: torch.Tensor, indx: int, axis: int
    ) -> torch.Tensor | None:
        if axis == 0:
            a_i = self._adj_tensor[indx, :]
            a_i = new_data - a_i
            new_data2 = a_i.view(1, -1).matmul(self._graph_n_adj) % self._p
            self._adj_tensor[indx, :] = (self._adj_tensor[indx, :] + new_data).clamp_(
                0, 1
            )
        elif axis == 1:
            a_i = self._adj_tensor[:, indx]
            print(a_i.shape)
            a_i = new_data - a_i
            new_data2 = self._graph_n_adj.matmul(a_i.view(-1, 1)) % self._p
            self._adj_tensor[:, indx] = (self._adj_tensor[:, indx] + new_data).clamp_(
                0, 1
            )
        return new_data2.squeeze()

    def _prepare_new_data_one_cell(
        self, new_data: torch.Tensor, indx: int, i: int
    ) -> torch.Tensor | None:
        a_i = self._adj_tensor[:, indx]
        a_i[i] = (new_data - a_i[i] + self._p) % self._p
        a_i = dense_to_sparse(a_i.view(-1, 1))
        new_data2 = multiply_sparse_rows_D(self._graph_n_adj, a_i, self._p)
        adj_matrix = self._m_matrix.clone()
        eye = self._eye(self._adj_tensor.shape[0])
        adj_matrix.add_(eye)
        new_data2 = torch.matmul(adj_matrix, new_data2) % self._p
        self._adj_tensor[i, indx] = (self._adj_tensor[i, indx] + new_data).clamp_(0, 1)
        return new_data2.squeeze()

    def _inverse_one_vector(
        self, new_data: torch.Tensor |int, indx: int, axis: int, cell : int | bool = False
    ) -> torch.Tensor:
        if cell is False:
            new_data = self._prepare_new_new_data(new_data, indx, axis)
        else:
            new_data = self._prepare_new_data_one_cell(new_data, indx,cell)
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
            if cell:
                self._col_row_to_change_flag = 0
        else:
            if cell:
                self._col_row_to_change_flag = 1
        return bx

    def _update_matrix(self):
        self._graph_n_adj = self._multiplicationN1M()
        self._col_row_to_change_flag = -1
        self._m_matrix = self._create_matrix_from_scratch(self._adj_tensor.shape[0], 0)

    def update_one_cell(self, i: int, j: int, value: Any):
        new_data = self._zeros(self._adj_tensor.shape[0])
        new_data[i] = value
        bx = self._inverse_one_vector(value, j, 1, i)
        adj_matrix = self._m_matrix.clone()
        eye = self._eye(self._adj_tensor.shape[0])
        adj_matrix.add_(eye)
        self._m_matrix = apply_single_column_update(
            adj_matrix,
            j,
            bx,
            self._p,
        )
        self._m_matrix.sub_(eye)
        self._m_matrix = (self._m_matrix + self._p) % self._p
        bx[j] = (bx[j] - 1) % self._p
        self._m_matrix[:, j] = (self._m_matrix[:, j] + bx) % self._p
        if self._check_non_zeros() == 2:
            self._update_matrix()

    def _check_non_zeros(self) -> int:
        nonzero_rows = (self._m_matrix.abs().sum(dim=1) > self._tol).sum().item()
        nonzero_cols = (self._m_matrix.abs().sum(dim=0) > self._tol).sum().item()

        if nonzero_cols > self._max_changes or nonzero_rows > self._max_changes:
            return 2
        if nonzero_rows > 0 or nonzero_cols > 0:
            return 1
        return 0

    def find_path(self, s: int, t: int) -> bool:
        return self._multiplicationN1M()[s, t].item() != 0

    def _multiplicationN1M(self) -> torch.Tensor:
        adj_matrix = self._m_matrix.clone()
        eye = self._eye(self._adj_tensor.shape[0])
        adj_matrix.add_(eye)

        if self._col_row_to_change_flag == 1:
            adj_matrix = multiply_sparse_cols_C(adj_matrix, self._graph_n_adj, self._p)
        else:

            adj_matrix = multiply_sparse_rows_D(adj_matrix, self._graph_n_adj, self._p)
        return adj_matrix
