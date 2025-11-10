import networkx as nx
import numpy as np
import torch

from Helper import GraphCharacter, size_of_cell, np_array_to_tensor_mapping


class CountGraph:
    def __init__(self, graph: nx.Graph, enum: GraphCharacter = GraphCharacter.Everything, type_of_data: str = "32"):
        self._graph = graph
        self._type_of_data = size_of_cell(type_of_data)
        self._adj_matrix = nx.to_numpy_array(graph, dtype=self._type_of_data)
        self._adj_tensor = None

        if enum in {GraphCharacter.Everything, GraphCharacter.GPU}:
            self._adj_tensor = torch.tensor(
                self._adj_matrix,
                dtype=np_array_to_tensor_mapping(self._type_of_data),
                device="cuda",
            )
            print("Tensor is on GPU:", self._adj_tensor.is_cuda)
