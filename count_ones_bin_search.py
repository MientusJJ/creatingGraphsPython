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
        matrices = dict()
        current_matrix = self._adj_matrix.copy()
        steps = 1

        start_time = time.time()

        while not is_filled(current_matrix):
            next_matrix = np.minimum(np.dot(current_matrix,current_matrix), 1)
            matrices.update({steps: current_matrix})
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2

        minSteps = steps
        result = steps * 2
        steps = steps // 2
        while steps > 0:
            temp_matrix =  np.minimum(np.dot(current_matrix,matrices[steps]), 1)
            if is_filled(temp_matrix):
                result = minSteps + steps
            else:
                minSteps += steps
                current_matrix = temp_matrix
            steps = steps // 2

        end_time = time.time() - start_time
        return "cpu", f"{end_time:.3f} ms", result



    @requires_gpu
    def multiply_until_filled_gpu(self) -> tuple[str, str, int]:
        adj_matrix = nx.to_numpy_array(self._graph)
        adj_matrix_tf = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        def is_filled(matrix):
            return tf.reduce_all(matrix == 1).numpy()

        matrices = dict()
        current_matrix = adj_matrix_tf
        steps = 1
        start_time = time.time()

        while not is_filled(current_matrix):
            next_matrix = tf.minimum(tf.linalg.matmul(current_matrix, current_matrix), 1)
            matrices.update({steps: current_matrix})
            if is_filled(next_matrix):
                break
            current_matrix = next_matrix
            steps *= 2
        minSteps = steps
        result = steps * 2
        steps = steps // 2
        while steps > 0:
            temp_matrix = tf.minimum(tf.linalg.matmul(current_matrix,matrices[steps]), 1)
            if is_filled(temp_matrix):
                result = minSteps + steps
            else:
                minSteps += steps
                current_matrix = temp_matrix
            steps = steps // 2

        end_time = time.time()
        gpu_time = end_time - start_time

        return "gpu", f"{gpu_time:.3f} ms", result



