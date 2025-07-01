from random import random
from typing import Dict

import networkx as nx
import torch

from Helper import np_array_to_tensor_mapping, GraphCharacter
from dynamic import Dynamic
from gpu_checker import requires_gpu
from optimized_multiplication import (
    multiply_sparse_rows_D,
    multiply_sparse_cols_C,
    modular_inverse_matrix_gpu,
    pow_number,
)


class DynamicSherman(Dynamic):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "32",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)
        self._map_of_vectors: Dict[int, torch.Tensor] = dict()
        self._map_of_vectors_transpose: Dict[int, torch.Tensor] = dict()

    def add_one_vector(self, new_data: torch.Tensor, index: int, axis: int | str):
        if not isinstance(new_data, torch.Tensor):
            raise TypeError("new_data must be a torch.Tensor.")
        axis = self._check_axis(axis)
        if axis == -1:
            raise ValueError('axis must be either "row" or "col"')
        if self._check_if_axis_is_the_same(axis) or self._check_number_of_vectors():
            self._update_matrix()
        binary_tensor = self._zeros(self._adj_tensor.shape[0])
        binary_tensor[index] = 1
        if axis == 0:
            self._col_row_to_change_flag = 0
            if index in self._map_of_vectors_transpose:
                self._map_of_vectors_transpose[index] += new_data
                self._map_of_vectors[index] += binary_tensor
            else:
                self._map_of_vectors_transpose[index] = new_data.clone()
                self._map_of_vectors[index] = binary_tensor
        else:
            self._col_row_to_change_flag = 1
            if index in self._map_of_vectors:
                self._map_of_vectors[index] += new_data
                self._map_of_vectors_transpose[index] += binary_tensor
            else:
                self._map_of_vectors[index] = new_data.clone()
                self._map_of_vectors_transpose[index] = binary_tensor

    def _update_matrix(self):
        for key, tensor in self._map_of_vectors.items():
            self._graph_n_adj = self._sherman_morrison(
                self._graph_n_adj,
                tensor.clone(),
                self._map_of_vectors_transpose[key].clone(),
            )
            if self._col_row_to_change_flag == 1:
                self._adj_tensor[:, key] += tensor
                self._adj_tensor[:, key] %= self._p
            elif self._col_row_to_change_flag == 0:
                self._adj_tensor[key, :] += self._map_of_vectors_transpose[key]
                self._adj_tensor[key, :] %= self._p
        self._col_row_to_change_flag = -1

        self._map_of_vectors: Dict[int, torch.Tensor] = dict()
        self._map_of_vectors_transpose: Dict[int, torch.Tensor] = dict()

    def _check_number_of_vectors(self) -> bool:
        return (
            len(self._map_of_vectors) >= self._max_changes
            or len(self._map_of_vectors_transpose) >= self._max_changes
        )

    def _check_if_axis_is_the_same(self, axis: int) -> bool:
        if self._col_row_to_change_flag != -1 and self._col_row_to_change_flag != axis:
            return True
        return False

    def _sherman_morrison(
        self, graph_n: torch.Tensor, u: torch.Tensor, v_t: torch.Tensor
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

    def find_path(self, s: int, t: int) -> bool:
        if max(len(self._map_of_vectors_transpose), len(self._map_of_vectors)) > 0:
            # return self._find_path_multiplication(s, t)
            self._update_matrix()
        return self._graph_n_adj[s, t].item() != 0

    def _find_path_multiplication(self, s: int, t: int) -> bool:
        adj_matrix = self._create_matrix_from_scratch(self._adj_tensor.shape[0], 0)
        if self._col_row_to_change_flag == 0:
            for key, tensor in self._map_of_vectors_transpose.items():
                adj_matrix[key, :] = tensor
        elif self._col_row_to_change_flag == 1:
            for key, tensor in self._map_of_vectors.items():
                adj_matrix[:, key] = tensor
