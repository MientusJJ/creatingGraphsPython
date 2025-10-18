import numpy as np
import tensorflow as tf
import networkx as nx
import time

import torch
from matplotlib import pyplot as plt

from Helper import GraphCharacter, ms
from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountTriangles(CountGraph):
    def __init__(
        self, graph: nx.Graph, enum: GraphCharacter = GraphCharacter.Everything
    ):
        super().__init__(graph, enum)

    def count_triangles_cpu(self) -> tuple[str,int, str]:
        start_time = time.time()
        triangles = int(round(np.trace(np.linalg.matrix_power(self._adj_matrix, 3)) / 6))
        end_time = (time.time() - start_time) * ms
        return "cpu",triangles, f"{end_time:.3f}"

    @requires_gpu
    def count_triangles_gpu(self) -> tuple[str,int, str]:
        start_time_2 = time.time()
        adj_matrix_cubed = torch.matmul(
            torch.matmul(self._adj_tensor, self._adj_tensor), self._adj_tensor
        )
        triangles = int((torch.trace(adj_matrix_cubed) / 6).item())
        end_time = (time.time() - start_time_2) * ms
        return "gpu",triangles, f"{end_time:.3f}"
