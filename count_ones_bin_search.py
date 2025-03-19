import numpy as np
import tensorflow as tf
import networkx as nx
import time

from count_graph import CountGraph
from gpu_checker import requires_gpu


class CountOnesBinSearch(CountGraph):
    def __init__(self, graph : nx.Graph):
        super().__init__(graph)

    def multiply_until_filled_cpu(self) -> tuple[str, str, int]:
        def is_filled(matrix):
            return np.all(matrix == 1)

        current_matrix = self._adj_matrix.copy()
        steps = 1
        start_time = time.time()

        while not is_filled(current_matrix):
            next_matrix = np.minimum(np.dot(current_matrix, self._adj_matrix), 1)
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2

        low, high = steps // 2, steps
        while low < high:
            mid = (low + high) // 2
            temp_matrix = self._adj_matrix.copy()
            for _ in range(mid):
                temp_matrix = np.minimum(np.dot(temp_matrix, self._adj_matrix), 1)

            if is_filled(temp_matrix):
                high = mid
            else:
                low = mid + 1

        end_time = time.time() - start_time
        return "cpu", f"{end_time:.3f} ms", low+1



    @requires_gpu
    def multiply_until_filled_gpu(self) -> tuple[str, str, int]:
        adj_matrix = nx.to_numpy_array(self._graph)
        adj_matrix_tf = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        def is_filled(matrix):
            return tf.reduce_all(matrix == 1).numpy()

        current_matrix = adj_matrix_tf
        steps = 1
        start_time = time.time()

        while not is_filled(current_matrix):
            next_matrix = tf.minimum(tf.linalg.matmul(current_matrix, adj_matrix_tf), 1)
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2

        low, high = steps // 2, steps
        while low < high:
            mid = (low + high) // 2
            temp_matrix = adj_matrix_tf
            for _ in range(mid):
                temp_matrix = tf.minimum(tf.linalg.matmul(temp_matrix, adj_matrix_tf), 1)

            if is_filled(temp_matrix):
                high = mid
            else:
                low = mid + 1

        end_time = time.time()
        gpu_time = end_time - start_time

        return "gpu", f"{gpu_time:.3f} ms", low+1



