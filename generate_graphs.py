from __future__ import annotations

import json
import os

import networkx as nx
import matplotlib.pyplot as plt
import random

from tensorflow.python.ops.summary_ops_v2 import graph


class GraphGenerator:
    def __init__(self, nodes: int = 10, edge_prob: float = 0.3, seed: int = 42,graph_type: str = "random", new : bool = True):
        self._graph_type = graph_type.lower()
        self._nodes = nodes
        self._edge_prob = edge_prob if edge_prob is not None else 0.3
        self._seed = seed

        random.seed(self._seed)
        if new:
            self._G = self._generate_graph()
        else:
            self._G = nx.Graph()
        self._add_self_loops()
    def _generate_graph(self) -> nx.Graph:

        match self._graph_type:
            case "random":
                G = nx.erdos_renyi_graph(self._nodes, self._edge_prob, seed=self._seed)
            case "grid":
                side = int(self._nodes ** 0.5)
                G = nx.grid_2d_graph(side, side)
            case "tree":
                G = nx.balanced_tree(r=2, h=int(self._nodes ** 0.5))
            case "complete":
                G = nx.complete_graph(self._nodes)
            case _:
                raise ValueError("Unknown graph type. Choose from: random, grid, tree, complete.")
        components = list(nx.connected_components(G))
        num_components = len(components)
        if num_components > 1:
            representative_nodes = [list(comp)[0] for comp in components]

            for i in range(1, num_components):
                G.add_edge(representative_nodes[i - 1], representative_nodes[i])

            print("✅ All components merged into a single connected graph!")
        return G
    def _add_self_loops(self) -> None:
        for node in self._G.nodes():
            if not self._G.has_edge(node, node):
                self._G.add_edge(node, node)
    def draw_graph(self, title: str = None) -> None:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self._G, seed=self._seed)

        nx.draw(self._G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
                node_size=700, font_size=12, font_weight="bold")

        plt.title(title or f"Graph Type: {self._graph_type.capitalize()}", fontsize=14)
        plt.show()

    def get_graph(self) -> nx.Graph:
        return self._G

    def save_graph(self,path : str | None = None) -> None:
        base_dir = "D:/Program Files/Magisterka/Grafy"
        graph_dir = os.path.join(base_dir, self._graph_type)

        os.makedirs(graph_dir, exist_ok=True)


        if path:
            filename = f"graph_{self._graph_type}_nodes={self._nodes}_p={self._edge_prob:.2f}_seed={self._seed}_{path}.json"
        else:
            filename = f"graph_{self._graph_type}_nodes={self._nodes}_p={self._edge_prob:.2f}_seed={self._seed}.json"
        filepath = os.path.join(graph_dir, filename)

        data = {
            "nodes": list(self._G.nodes),
            "edges": list(self._G.edges),
            "graph_type": self._graph_type,
            "nodes_count": self._nodes,
            "edge_prob": self._edge_prob,
            "seed": self._seed
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Graph saved to {filepath}")

    @staticmethod
    def load_graph(filename: str) -> GraphGenerator:
        with open(filename, "r") as f:
            data = json.load(f)

        graph = GraphGenerator(
            nodes=data["nodes_count"],
            edge_prob=data["edge_prob"],
            seed=data["seed"],
            graph_type=data["graph_type"],
            new=False
        )

        graph._G.add_nodes_from(data["nodes"])
        graph._G.add_edges_from(data["edges"])

        print(f"Graph loaded from {filename}")
        return graph

    @staticmethod
    def load_graph_stanford(filename: str) -> GraphGenerator:
        """Loads a graph from a Stanford Large Network Dataset Collection formatted TXT file."""

        G = nx.Graph()

        with open(filename, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue  # Ignorujemy linie komentarza
                parts = line.strip().split()
                if len(parts) == 2:
                    node1, node2 = map(int, parts)
                    G.add_edge(node1, node2)

        # Informacja o wczytanym grafie
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"✅ Loaded Stanford graph with {num_nodes} nodes and {num_edges} edges.")

        # Tworzymy obiekt GraphGenerator z wczytanym grafem
        graph = GraphGenerator(
            nodes=num_nodes,
            edge_prob=1.,
            seed=-1,
            graph_type="random",
            new=False
        )

        graph._G = G
        return graph
