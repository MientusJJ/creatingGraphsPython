import numpy as np
import tensorflow as tf
import networkx as nx
import time

from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountOnes(CountGraph):
    def __init__(self, graph : nx.Graph):
        super().__init__(graph)

    def multiply_until_filled_cpu(self) -> tuple[str, str,int]:
        current_matrix = self._adj_matrix.copy()
        steps = 0
        start_time = time.time()
        while not np.all(current_matrix == 1):
            current_matrix = np.minimum(np.dot(current_matrix, self._adj_matrix), 1)
            steps += 1
        end_time = time.time() - start_time
        return "cpu",  f"{end_time:.3f} ms", steps+1

    @requires_gpu
    def multiply_until_filled_gpu(self) -> tuple[str, str,int]:
        adj_matrix_tf = tf.convert_to_tensor(self._adj_matrix, dtype=tf.float32)
        current_matrix = adj_matrix_tf
        steps = 0
        start_time_1 = time.time()
        while not tf.reduce_all(current_matrix == 1):
            current_matrix = tf.minimum(tf.linalg.matmul(current_matrix, adj_matrix_tf), 1)
            steps += 1
        end_time = time.time()
        gpu_time = end_time - start_time_1
        return "gpu",  f"{gpu_time:.3f} ms",steps+1


