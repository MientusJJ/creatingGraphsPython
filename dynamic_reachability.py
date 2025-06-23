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
        self._max_changes = int(math.sqrt(self._graph.number_of_nodes()))
        self._p = Primes[type_of_data]
        #self._set_numbers_in_matrix()
        self._graph_n_adj = self._pow_matrix(self._graph.number_of_nodes(),self._adj_tensor)
        self._sherman_map_of_changes: dict[Tuple[int, int] : float] = dict()
        self._sherman_flag = -1
        self._tol = 1e-8
        # print(self._graph_n_adj)

    def _set_numbers_in_matrix(self):
        n = self._adj_tensor.shape[0]
        for i in range(n):
            for j in range(n):
                if self._adj_tensor[i][j] != 0:
                    self._adj_tensor[i][j] = random.randint(1, self._p - 1)

    def _pow_matrix(self, n: int, tensor : torch.tensor) -> torch.tensor:
        curr_state = 1

        eye = torch.eye(
            tensor.shape[0],
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
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
    def _prepare_new_new_data(self,new_data: torch.Tensor, indx: int, axis: int) -> torch.Tensor | None:
        if axis == 0:
            a_i = self._adj_tensor[indx, :]
            a_i = a_i - new_data
            new_data = a_i.view(1,-1).matmul(self._graph_n_adj) % self._p
            new_data = new_data.squeeze()
            self._adj_tensor[indx, :] = self._adj_tensor[indx, :] - new_data
            self._adj_tensor[indx, :] = (self._adj_tensor[indx, :] + self._p) % self._p
        elif axis == 1:
           a_i = self._adj_tensor[:, indx]
           print(a_i.shape)
           a_i = a_i - new_data
           new_data = self._graph_n_adj.matmul(a_i.view(-1, 1)) % self._p
           new_data = new_data.squeeze()
           self._adj_tensor[:, indx] = self._adj_tensor[:, indx] - new_data
           self._adj_tensor[:, indx] = (self._adj_tensor[:, indx]+self._p) % self._p
        return new_data.squeeze()
    def _inverse_one_vector(
        self, new_data: torch.Tensor, indx: int, axis: int, cell=True
    ):
        new_data = self._prepare_new_new_data(new_data, indx, axis)

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

    def update_one_cell_row_or_col(
        self, new_data: torch.Tensor, indx: int, axis: str | int = "row"
    ):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
        if (new_data > 0).sum() > 1:
            raise ValueError("Więcej niż jedna wartość dodatnia w tensorze!")  # to rzuci błąd
        if self._check_non_zeros() > 0 and self._col_row_to_change_flag != axis:
            self._update_matrix()

        axis = self._check_axis(axis)
        if axis == -1:
            raise ValueError('axis must be either "row" or "col"')
        self._add_primes_to_vector(new_data)
        bx = self._inverse_one_vector(new_data, indx, axis, True)

        if axis == 1:
            self._m_matrix = apply_single_column_update(
                self._m_matrix,
                indx,
                bx,
                self._p,
            )
        elif axis == 0:
            self._m_matrix = apply_single_row_update(self._m_matrix, indx, bx, self._p)
        bx[indx] = (bx[indx] - 1) % self._p
        if axis == 0:
            # print(bx.shape, bx.dim())
            # print(self._m_matrix.shape,self._m_matrix.dim())
            self._m_matrix[indx, :] = (self._m_matrix[indx, :] + bx) % self._p
        else:
            self._m_matrix[:, indx] = (self._m_matrix[:, indx] + bx) % self._p
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

    def _add_primes_to_vector(self, data: torch.Tensor) -> None:
        return
        #TEST
        # for i in range(data.size(0)):
        #     if data[i].item() != 0:
        #         data[i] = random.randint(1, self._p - 1)

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

        if self._col_row_to_change_flag == 1:
            adj_matrix = multiply_sparse_cols_C(adj_matrix, self._graph_n_adj, self._p)
        else:

            adj_matrix = multiply_sparse_rows_D(adj_matrix, self._graph_n_adj, self._p)
        return adj_matrix

    def sherman_method_add_one_cell(self, i: int, j: int, value: float):
        if (i, j) in self._sherman_map_of_changes:
            self._sherman_map_of_changes[(i, j)] += value % self._p
        else:
            self._sherman_map_of_changes[(i, j)] = value

    def update_matrix_adj_tensor_I_minus_A_pow_minus_1(self):
        for key, val in self._sherman_map_of_changes.items():
            i, j = key
            if val == 0:
                self._adj_tensor[i, j] = 0
            else:
                self._adj_tensor[i, j] = random.randint(1, self._p - 1)

        self._graph_n_adj = self._pow_matrix(self._graph.number_of_nodes(),self._adj_tensor)
        self._sherman_map_of_changes = dict()

    def sparse_dict_to_sparse_tensor(self, device: str = "cuda") -> torch.Tensor:
        indices = torch.tensor(
            list(self._sherman_map_of_changes.keys()), dtype=torch.long
        ).T
        values = torch.tensor(
            list(self._sherman_map_of_changes.values()),
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
        sparse_tensor = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self._adj_tensor.shape[0], self._adj_tensor.shape[0]),
            device=device,
        )
        return sparse_tensor

    def sparse_transposed_indicator_tensor(self, device: str = "cuda") -> torch.Tensor:
        keys = list(self._sherman_map_of_changes.keys())
        indices = torch.tensor(keys, dtype=torch.long).T
        transposed_indices = indices[[1, 0], :]  # (j, i)
        values = torch.ones(
            transposed_indices.shape[1],
            dtype=np_array_to_tensor_mapping(self._type_of_data),
            device=device,
        )
        n = self._adj_tensor.shape[0]

        return torch.sparse_coo_tensor(
            transposed_indices,
            values,
            size=(self._adj_tensor.shape[0], self._adj_tensor.shape[0]),
            device=device,
        ).coalesce()

    def sherman_morisson(self):
        for (i, j), val in self._sherman_map_of_changes.items():
            self._sherman_map_of_changes[(i, j)] = (
                val - self._graph_n_adj[i, j].item() + self._p
            ) % self._p
        eye = torch.eye(
            self._adj_tensor.shape[0],
            device="cuda",
            dtype=np_array_to_tensor_mapping(self._type_of_data),
        )
        U = self.sparse_transposed_indicator_tensor()
        V = self.sparse_dict_to_sparse_tensor()
        t1 = multiply_sparse_rows_D(self._graph_n_adj, U, self._p)
        t2 = multiply_sparse_cols_C(V, t1, self._p)
        t3 = modular_inverse_matrix_gpu((eye + t2) % self._p, self._p)
        t4 = multiply_sparse_cols_C(V, self._graph_n_adj, self._p)
        t2 = multiply_sparse_rows_D(t3, t4, self._p)
        t4 = multiply_sparse_cols_C(t1, t2, self._p)
        self._graph_n_adj = ((self._graph_n_adj - t4) + self._p) % self._p
        #     t1 = torch.matmul(self._graph_n_adj, matrix_col)
        #     t1 = (t1 % self._p + self._p) % self._p
        #
        #     t2 = torch.matmul(matrix_row, t1)
        #     t2 = (t2 % self._p + self._p) % self._p
        #
        #     eye = torch.eye(
        #         self._adj_tensor.shape[0],
        #         device="cuda",
        #         dtype=np_array_to_tensor_mapping(self._type_of_data),
        #     )
        #
        #     t3 = eye + t2
        #     t3 = (t3 % self._p + self._p) % self._p
        #     t3 = modular_inverse_matrix_gpu(t3, self._p)
        #
        #     t4 = torch.matmul(t1, t3)
        #     t4 = (t4 % self._p + self._p) % self._p
        #
        #     t5 = torch.matmul(t4, matrix_row)
        #     t5 = (t5 % self._p + self._p) % self._p
        #
        #     self._graph_n_adj.sub_(t5)
        #     self._graph_n_adj = (self._graph_n_adj % self._p + self._p) % self._p
