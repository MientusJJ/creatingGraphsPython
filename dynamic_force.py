from typing import Any

import networkx as nx

from Helper import GraphCharacter
from dynamic import Dynamic
from gpu_checker import requires_gpu


class DynamicForce(Dynamic):
    @requires_gpu
    def __init__(
        self,
        graph: nx.Graph,
        enum: GraphCharacter = GraphCharacter.Everything,
        type_of_data: str = "32",
    ):
        super().__init__(graph=graph, enum=enum, type_of_data=type_of_data)

    def update_one_cell(self, i : int, j : int, value : Any):
        self._adj_tensor[i,j] += value
        self._adj_matrix[i,j] %= self._p
        self._graph_n_adj = self._pow_matrix(
            self._graph.number_of_nodes(), self._adj_tensor
        )

    def find_path(self, s: int, t: int) -> bool:
        return self._graph_n_adj[s,t].item() != 0

