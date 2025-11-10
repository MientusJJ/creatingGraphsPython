from random import random
from typing import Dict, Any

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
    apply_single_column_update,
    dense_to_sparse,
    apply_single_row_update,
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

    def update_one_cell(self, i: int, j: int, value: Any):
        tensor = self._zeros(self._adj_tensor.shape[0])
        tensor[i] = value
        binary_tensor = self._zeros(self._adj_tensor.shape[0])
        binary_tensor[j] = 1
        self._update_matrix(tensor, binary_tensor, j)

    def _update_matrix(
        self, tensor: torch.Tensor, binary_tensor: torch.Tensor, index: int
    ):
        self._graph_n_adj = self._sherman_morrison(
            self._graph_n_adj,
            tensor.clone(),
            binary_tensor,
        )
        self._adj_tensor[:, index] += tensor
        self._adj_tensor[:, index] %= self._p

    def _check_number_of_vectors(self) -> bool:
        return (
            len(self._map_of_vectors) >= self._max_changes
            or len(self._map_of_vectors_transpose) >= self._max_changes
        )

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
        return self._graph_n_adj[s, t].item() != 0
