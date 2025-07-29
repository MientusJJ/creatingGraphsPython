from __future__ import annotations

import json
import os

import networkx as nx
import matplotlib.pyplot as plt
import random

from tensorflow.python.ops.summary_ops_v2 import graph

from Helper import DIR, not_allowed_self_loops_graphs


class GraphGenerator:
    def __init__(
        self,
        nodes: int = 10,
        edge_prob: float = 0.3,
        seed: int = 42,
        graph_type: str = "random",
        new: bool = True,
        self_loops: bool = False,
    ):
        self._graph_type = graph_type.lower()
        self._nodes = nodes
        self._edge_prob = edge_prob if edge_prob is not None else 0.3
        self._seed = seed

        random.seed(self._seed)
        if new:
            self._G = self._generate_graph()
        else:
            self._G = (
                nx.DiGraph()
                if self._graph_type in not_allowed_self_loops_graphs
                else nx.Graph()
            )
        if self._graph_type == "random" or self_loops == True:
            self._add_self_loops()

    def show_adjacency_matrix(self) -> None:
        adj_matrix = nx.to_numpy_array(self._G, dtype=int)
        plt.figure(figsize=(10, 8))
        plt.imshow(adj_matrix, cmap="Blues", interpolation="none")
        plt.title("Macierz sąsiedztwa", fontsize=14)
        plt.xlabel("Wierzchołki")
        plt.ylabel("Wierzchołki")
        plt.colorbar(label="Połączenie")
        plt.tight_layout()
        plt.show()

    def _generate_graph(self) -> nx.Graph:

        match self._graph_type:
            case "random":
                G = nx.erdos_renyi_graph(self._nodes, self._edge_prob, seed=self._seed)
            case "grid":
                side = int(self._nodes**0.5)
                G = nx.grid_2d_graph(side, side)
            case "tree":
                G = nx.balanced_tree(r=2, h=int(self._nodes**0.5))
            case "complete":
                G = nx.complete_graph(self._nodes)
            case "path":
                G = nx.path_graph(self._nodes)
            case "dag":
                G = self._generate_dag()
            case "path_dag":
                G = nx.DiGraph()
                G.add_nodes_from(range(self._nodes))
                for i in range(self._nodes - 1):
                    G.add_edge(i, i + 1)
            case _:
                raise ValueError(
                    "Unknown graph type. Choose from: random, grid, tree, complete, dag,, path_dag."
                )
        if self._graph_type not in not_allowed_self_loops_graphs:
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

    def _generate_dag(self) -> nx.DiGraph:
        G = nx.DiGraph()
        nodes = list(range(self._nodes))
        G.add_nodes_from(nodes)

        for i in range(self._nodes):
            from_node = nodes[i]
            candidates = nodes[i + 1 :]
            num_edges = (
                max(1, int(len(candidates) * self._edge_prob)) if candidates else 0
            )
            targets = random.sample(candidates, min(num_edges, len(candidates)))
            for to_node in targets:
                G.add_edge(from_node, to_node)
        for node in nodes:
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                candidates = [n for n in nodes if n != node]
                if candidates:
                    target = random.choice(candidates)
                    if nodes.index(node) < nodes.index(target):
                        G.add_edge(node, target)
                    else:
                        G.add_edge(target, node)

        return G

    def draw_graph(self, title: str = None) -> None:
        plt.figure(figsize=(12, 10))

        if nx.is_directed(self._G):
            try:
                pos = nx.nx_pydot.graphviz_layout(self._G, prog="dot")
            except:
                pos = nx.spring_layout(self._G, seed=self._seed)
        else:
            pos = nx.spring_layout(self._G, seed=self._seed)

        nx.draw_networkx_nodes(
            self._G,
            pos,
            node_color="lightblue",
            node_size=800,
            edgecolors="black",
        )
        nx.draw_networkx_labels(self._G, pos, font_size=10, font_weight="bold")
        nx.draw_networkx_edges(
            self._G,
            pos,
            arrows=nx.is_directed(self._G),
            connectionstyle="arc3,rad=0.15",
            edge_color="gray",
        )

        plt.title(title or f"Graph Type: {self._graph_type.capitalize()}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def get_graph(self) -> nx.Graph:
        return self._G

    def save_graph(self, test: bool = True, path: str | None = None) -> None:
        base_dir = DIR + "/tests" if test else DIR
        if path:
            filename = f"graph_{self._graph_type}_nodes={self._nodes}_p={self._edge_prob:.2f}_seed={self._seed}_{path}.json"
        else:
            filename = f"graph_{self._graph_type}_nodes={self._nodes}_p={self._edge_prob:.2f}_seed={self._seed}.json"
        filepath = os.path.join(base_dir, filename)

        data = {
            "nodes": list(self._G.nodes),
            "edges": list(self._G.edges),
            "graph_type": self._graph_type,
            "nodes_count": self._nodes,
            "edge_prob": self._edge_prob,
            "seed": self._seed,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Graph saved to {filepath}")

    @staticmethod
    def load_graph(filename: str) -> GraphGenerator:
        filename = os.path.join(DIR, filename)
        with open(filename, "r") as f:
            data = json.load(f)

        graph = GraphGenerator(
            nodes=data["nodes_count"],
            edge_prob=data["edge_prob"],
            seed=data["seed"],
            graph_type=data["graph_type"],
            new=False,
        )

        graph._G.add_nodes_from(data["nodes"])
        graph._G.add_edges_from(data["edges"])

        print(f"Graph loaded from {filename}")
        return graph

    @staticmethod
    def load_graph_stanford(filename: str) -> GraphGenerator:

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
            nodes=num_nodes, edge_prob=1.0, seed=-1, graph_type="random", new=False
        )

        graph._G = G
        return graph
