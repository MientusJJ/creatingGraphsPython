from enum import Enum
from typing import TypeAlias

import numpy as np
import torch
from google.protobuf.text_format import PrintMessage
from matplotlib import pyplot as plt

DIR = "/mnt/e//Magisterka/Grafy"
ms = 1000


class GraphCharacter(Enum):
    Everything = 1
    GPU = 2
    CPU = 3


Graphs = {
    0: "graph_random_nodes=10000_p=0.00_seed=52.json",
    1: "graph_random_nodes=30000_p=0.02_seed=42.json",
    2: "graph_random_nodes=50000_p=0.00_seed=42.json",
    3: "graph_random_nodes=1000_p=0.00_seed=42.json",
    4: "graph_path_nodes=10000_p=0.30_seed=42.json",
    5: "graph_path_nodes=1000_p=0.30_seed=42.json",
    6: "graph_path_nodes=5000_p=0.30_seed=42.json",
    7: "graph_dag_nodes=1000_p=0.50_seed=69.json",
    8: "graph_dag_nodes=10000_p=0.30_seed=42.json",
    9: "graph_path_dag_nodes=10_p=0.30_seed=42.json",
    10: "graph_dag_nodes=10_p=0.10_seed=48.json",
    11: "graph_dag_nodes=10_p=0.50_seed=54.json",
    12: "testgraph.json",
}

Primes = {
    "16": 43,
    "32": 4093,
    "64": 4251749,
    "int32": int(10**9 + 7),
    "int64": int(10**9 + 7),
}
mapping_row_columns = {
    0: "row",
    1: "col",
    "row": 0,
    "col": 1,
}


def size_of_cell(s: str) -> TypeAlias:
    if s == "16":
        return np.float16
    elif s == "32":
        return np.float32
    elif s == "64":
        return np.float64
    elif s == "int32":
        return np.int32
    elif s == "int64":
        return np.int64
    else:
        raise RuntimeError(f"Bad value for size of float: {s}")


def np_array_to_tensor_mapping(s: TypeAlias) -> TypeAlias:
    if s == np.float16:
        return torch.float16
    elif s == np.float32:
        return torch.float32
    elif s == np.float64:
        return torch.float64
    elif s == np.int32:
        return torch.int32
    elif s == np.int64:
        return torch.int64
    else:
        raise RuntimeError(f"Bad value for size of float: {s}")


not_allowed_self_loops_graphs = {"dag", "path_dag"}


def show_adjacency_tensor(
    adj_tensor: torch.Tensor, title: str = "Macierz sąsiedztwa (GPU)"
) -> None:
    if adj_tensor.is_cuda:
        adj_matrix = adj_tensor.detach().cpu().numpy()
    else:
        adj_matrix = adj_tensor.numpy()
    adj_matrix = (adj_matrix != 0).astype(int)

    n = adj_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(adj_matrix, cmap="Blues", interpolation="none", zorder=0)

    # Ustawienia osi
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    # Siatka
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, zorder=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Opisy
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Wierzchołki")
    ax.set_ylabel("Wierzchołki")

    fig.colorbar(im, ax=ax, label="Połączenie")
    plt.tight_layout()
    plt.show()
