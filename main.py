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
    graphIndex = 9
    graph = GraphGenerator.load_graph(Graphs[graphIndex])
    # graph.show_adjacency_matrix()
    graph_dynamic = DynamicReachAbility(graph.get_graph())
    tensor = torch.zeros(10, device="cuda", dtype=torch.float64)
    tensor[3:10] = 1
    graph_dynamic.update_row_or_col(tensor, 2)
    show_adjacency_tensor(graph_dynamic._adj_tensor)
    # tensor2 = torch.zeros(10, device="cuda", dtype=torch.float64)
    # tensor2[5:7] = 1
    graph_dynamic.update_row_or_col(tensor2, 4, 1)
    #graph_dynamic.find_path(0, 9)
    show_adjacency_tensor(graph_dynamic._adj_tensor)
    # for i in range(0,10):
    #     for j in range(0,i):
    #         print(graph_dynamic.find_path(i,j))
    # plt.show()
