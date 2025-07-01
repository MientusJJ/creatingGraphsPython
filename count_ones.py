import numpy as np
import tensorflow as tf
import networkx as nx
import time

import torch

from Helper import ms, GraphCharacter
from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountOnes(CountGraph):
    def __init__(
        self, graph: nx.Graph, enum: GraphCharacter = GraphCharacter.Everything
    ):
        super().__init__(graph, enum)

    def multiply_until_filled_cpu(self) -> tuple[str, str, int, int]:
        current_matrix = self._adj_matrix.astype(np.float16).copy()
        steps = 0
        start_time = time.time()
        while np.any(current_matrix != 1):
            current_matrix = np.dot(current_matrix, self._adj_matrix)
            np.minimum(current_matrix, 1, out=current_matrix)
            steps += 1
        end_time = (time.time() - start_time) * ms
        return "cpu", f"{end_time:.3f} ms", steps + 1, steps

    @requires_gpu
    def multiply_until_filled_gpu(self) -> tuple[str, str, int, int]:

        adj_matrix = self._adj_tensor.to(torch.float16)
        current_matrix = adj_matrix.clone()
        one_tensor = torch.tensor(
            1.0, dtype=current_matrix.dtype, device=current_matrix.device
        )

        steps = 0
        start_time = time.time()
        while torch.any(current_matrix != 1):
            current_matrix = torch.matmul(current_matrix, adj_matrix)
            current_matrix = torch.minimum(current_matrix, one_tensor)
            steps += 1

        elapsed = (time.time() - start_time) * ms
        print(steps + 1)
        return "gpu (torch)", f"{elapsed:.3f} ms", steps + 1, steps
