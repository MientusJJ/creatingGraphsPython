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
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap="Blues", interpolation="none")
    plt.title(title, fontsize=14)
    plt.xlabel("Wierzchołki")
    plt.ylabel("Wierzchołki")
    plt.colorbar(label="Połączenie")
    plt.tight_layout()
    plt.show()


import torch


def modular_inverse_scalar(a: int, p: int) -> int:
    if a == 0:
        raise ValueError("Inverse does not exist for 0")
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError("a is not invertible")
    return t + p if t < 0 else t


def modular_inverse_matrix_gpu(matrix: torch.Tensor, p: int) -> torch.Tensor:
    assert matrix.device.type == "cuda"
    if matrix.dtype.is_floating_point:
        matrix = matrix.round().to(torch.int64)

    assert matrix.dtype in (torch.int32, torch.int64), "Matrix must be integer type"
    n = matrix.size(0)
    A = matrix.clone()
    I = torch.eye(n, dtype=matrix.dtype, device=matrix.device)

    for i in range(n):
        inv = modular_inverse_scalar(int(A[i, i].item()), p)
        A[i] = (A[i] * inv) % p
        I[i] = (I[i] * inv) % p

        for j in range(n):
            if i != j:
                factor = A[j, i].clone()
                A[j] = (A[j] - factor * A[i]) % p
                I[j] = (I[j] - factor * I[i]) % p

    return I
