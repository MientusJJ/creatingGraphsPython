import logging

from Helper import DIR, Graphs, show_adjacency_tensor
from count_ones import CountOnes
from count_ones_bin_search import CountOnesBinSearch
from count_triangles import CountTriangles
from dynamic_reachability import DynamicReachAbility
from generate_graphs import GraphGenerator
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
import os
import psutil

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")
    # A = torch.tensor(
    #     [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=torch.float64
    # ).cuda()
    # print(A)
    # graphs = GraphGenerator(10,0.5,54,"dag")
    # graphs.save_graph()
    # TODO
    # PRZECZYTAJ:
    # file: // / C: / Users / HardPC / Downloads / dynamic - transitive - closure - via - dynamic - matrix - inverse - extended - a.pdf
    # strona 4
    graphIndex = 11
    graph = GraphGenerator.load_graph(Graphs[graphIndex])
    graph_dynamic = DynamicReachAbility(graph.get_graph())
    # for i in range(0,10):
    #     for j in range(0,10):
    #         print(f"{(i, j)}: {str(graph_dynamic.find_path(i, j)).lower()}")
    i = 0
    j = 4

    print(f"{(i, j)}: {str(graph_dynamic.find_path(i, j)).lower()}")
    show_adjacency_tensor(graph_dynamic._graph_n_adj)
    tensor = torch.zeros(10, device="cuda", dtype=torch.float64)
    tensor[0:1] = 1
    print(tensor)
    graph_dynamic.update_one_row_or_col(tensor, 4, "col")
    print(f"{(i, j)}: {str(graph_dynamic.find_path(i, j)).lower()}")

    # graph_dynamic._update_matrix()
    # print(f"{(i, j)}: {str(graph_dynamic.find_path(i, j)).lower()}")
    # graph_dynamic.update_row_or_col(tensor, 2)
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # # tensor2 = torch.zeros(10, device="cuda", dtype=torch.float64)
    # # tensor2[5:7] = 1
    # graph_dynamic.update_row_or_col(tensor2, 4, 1)
    # #graph_dynamic.find_path(0, 9)
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # for i in range(0,10):
    #     for j in range(0,i):
    #         print(graph_dynamic.find_path(i,j))
    # plt.show()
