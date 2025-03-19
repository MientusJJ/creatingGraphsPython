import networkx as nx


class CountGraph:
    def __init__(self,graph : nx.Graph):
        self._graph =  graph
        self._adj_matrix = nx.to_numpy_array(self._graph)