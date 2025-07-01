from typing import Any

import numpy as np
import tensorflow as tf
import networkx as nx
import time

import torch

from Helper import GraphCharacter, ms
from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountOnesBinSearch(CountGraph):
    def __init__(
        self, graph: nx.Graph, enum: GraphCharacter = GraphCharacter.Everything
    ):
        super().__init__(graph, enum)

    def multiply_until_filled_cpu(self) -> tuple[str, str, int, int]:
        def is_filled(matrix):
            return not np.any(matrix != 1)

        matrices = dict()
        current_matrix = self._adj_matrix.copy()
        steps = 1

        start_time = time.time()
        mults = 0
        while not is_filled(current_matrix):
            next_matrix = np.minimum(np.dot(current_matrix, current_matrix), 1)
            mults += 1
            matrices.update({steps: current_matrix})
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2

        minSteps = steps
        result = steps * 2
        steps = steps // 2
        while steps > 0:
            temp_matrix = np.minimum(np.dot(current_matrix, matrices[steps]), 1)
            mults += 1
            if is_filled(temp_matrix):
                result = minSteps + steps
            else:
                minSteps += steps
                current_matrix = temp_matrix
            steps = steps // 2

        end_time = ms * (time.time() - start_time)
        return (
            "cpu",
            f"{end_time:.3f} ms",
            result,
            mults,
        )

    @requires_gpu
    def multiply_until_filled_gpu(self) -> tuple[str, str, int, int]:
        adj_matrix = self._adj_tensor

        def is_filled(matrix: torch.tensor) -> bool:
            return not torch.any(matrix != 1).item()

        device = "cuda"
        matrices = dict()
        current_matrix = adj_matrix
        steps = 1
        start_time = time.time()
        mults = 0
        while not is_filled(current_matrix):
            next_matrix = torch.minimum(
                torch.matmul(current_matrix, current_matrix),
                torch.tensor(1.0, device=device),
            )
            mults += 1
            matrices.update({steps: current_matrix})
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2
        minSteps = steps
        result = steps * 2
        steps = steps // 2
        while steps > 0:
            temp_matrix = torch.minimum(
                torch.matmul(current_matrix, matrices[steps]),
                torch.tensor(1.0, device=device),
            )
            mults += 1
            if is_filled(temp_matrix):
                result = minSteps + steps
            else:
                minSteps += steps
                current_matrix = temp_matrix
            steps = steps // 2

        end_time = time.time()
        gpu_time = (end_time - start_time) * ms

        return "gpu", f"{gpu_time:.3f} ms", result, mults
