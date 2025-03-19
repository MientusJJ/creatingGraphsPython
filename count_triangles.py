import numpy as np
import tensorflow as tf
import networkx as nx
import time

from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountTriangles(CountGraph):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
    def count_triangles_cpu(self) -> tuple[str, str]:
        start_time = time.time()
        triangles = np.trace(np.linalg.matrix_power(self._adj_matrix, 3)) / 6
        end_time = time.time() - start_time
        return "cpu",  f"{end_time:.3f} ms"

    @requires_gpu
    def count_triangles_gpu(self) -> tuple[str, str]:

        adj_matrix_tf = tf.convert_to_tensor(self._adj_matrix, dtype=tf.float32)
        start_time_2 = time.time()
        adj_matrix_cubed = tf.linalg.matmul(tf.linalg.matmul(adj_matrix_tf, adj_matrix_tf), adj_matrix_tf)
        triangles = tf.linalg.trace(adj_matrix_cubed) / 6
        end_time = time.time()
        end_time = end_time - start_time_2
        return "gpu",  f"{end_time:.3f} ms"


