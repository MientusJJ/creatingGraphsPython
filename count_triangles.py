import numpy as np
import tensorflow as tf
import networkx as nx
import time

import torch

from Helper import GraphCharacter, ms
from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountTriangles(CountGraph):
    def __init__(
        self, graph: nx.Graph, enum: GraphCharacter = GraphCharacter.Everything
    ):
        super().__init__(graph, enum)

    def count_triangles_cpu(self) -> tuple[str, str]:
        start_time = time.time()
        triangles = np.trace(np.linalg.matrix_power(self._adj_matrix, 3)) / 6
        end_time = (time.time() - start_time) * ms
        return "cpu", f"{end_time:.3f} ms"

    @requires_gpu
    def count_triangles_gpu(self) -> tuple[str, str]:

        start_time_2 = time.time()
        adj_matrix_cubed = torch.matmul(
            torch.matmul(self._adj_tensor, self._adj_tensor), self._adj_tensor
        )
        triangles = torch.trace(adj_matrix_cubed) / 6
        end_time = (time.time() - start_time_2) * ms
        return "gpu", f"{end_time:.3f} ms"
