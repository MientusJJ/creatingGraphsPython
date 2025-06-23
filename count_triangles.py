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

    def count_triangles_cpu(self) -> tuple[str, str]:
        start_time = time.time()
        triangles = np.trace(np.linalg.matrix_power(self._adj_matrix, 3)) / 6
        # adj_matrix_pow3 = np.linalg.matrix_power(self._adj_matrix, 3)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(adj_matrix_pow3, cmap="Blues", interpolation="none")
        #
        # plt.title(r"Macierz $A^3$ – liczba ścieżek długości 3", fontsize=14)
        # plt.xlabel("Wierzchołki")
        # plt.ylabel("Wierzchołki")
        # plt.colorbar(label="Liczba ścieżek")
        #
        # # Dodaj liczby do każdej komórki
        # rows, cols = adj_matrix_pow3.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         plt.text(j, i, str(adj_matrix_pow3[i, j]),
        #                  ha='center', va='center', color='black', fontsize=8)
        #
        # plt.tight_layout()
        # plt.show()
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
