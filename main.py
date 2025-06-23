import logging

import numpy as np

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

from optimized_multiplication import multiply_sparse_cols_C

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")
    # # A = torch.tensor(
    # #     [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=torch.float64
    # # ).cuda()
    # # print(A)
    # # graphs = GraphGenerator(10,0.5,54,"dag")
    # # graphs.save_graph()
    # # TODO
    # # PRZECZYTAJ:
    # # file: // / C: / Users / HardPC / Downloads / dynamic - transitive - closure - via - dynamic - matrix - inverse - extended - a.pdf
    # # strona 4
    tensor = torch.zeros(10, device="cuda")
    #tensor[0] = 1
    #tensor[2:5] = 1
    tensor[8:10] = 1
    graphIndex = 11
    print(tensor)
    graph = GraphGenerator.load_graph(Graphs[graphIndex])
    graph_dynamic = DynamicReachAbility(graph.get_graph())
    show_adjacency_tensor(graph_dynamic._adj_tensor)
    print(graph_dynamic.find_path(2,5))
    print(graph_dynamic.find_path(2,8))
    print(graph_dynamic.find_path(2,9))
    show_adjacency_tensor(graph_dynamic._graph_n_adj)
    graph_dynamic.update_one_row_or_col(tensor,5,"row")

    tensor = torch.zeros(10, device="cuda")
    tensor[2:5] = 1
    graph_dynamic.update_one_row_or_col(tensor, 5, "col")
    print(graph_dynamic.find_path(2,5))
    print(graph_dynamic.find_path(2,8))
    print(graph_dynamic.find_path(2,9))
    show_adjacency_tensor(graph_dynamic._graph_n_adj)
    #show_adjacency_tensor(graph_dynamic._graph_n_adj)

    # # GPU
    # try:
    #     gpu_time = float(cog.count_triangles_gpu()[1].replace(" ms", ""))
    # except RuntimeError as e:
    #     print(f"GPU not available: {e}")
    #     gpu_time = None
    # gpu_times.append(gpu_time)
    #
    # labels = ['Graph ' + str(graphIndex)]
    # cpu_times = [cpu_time]
    # gpu_times = [gpu_time]
    #
    # x = np.arange(len(labels))  # pozycje na osi X
    # width = 0.35  # szerokość słupków
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # rects1 = ax.bar(x - width / 2, cpu_times, width, label='CPU')
    # rects2 = ax.bar(x + width / 2, gpu_times, width, label='GPU')
    #
    # # Opis osi i tytuł
    # ax.set_ylabel('Czas działania (ms)')
    # ax.set_title('Porównanie czasów: CPU vs GPU - triangles')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.set_yscale("log")  # skala logarytmiczna
    # ax.legend()
    #
    #
    # # Dodanie wartości nad słupkami
    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate(f'{height:.2f}',
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 5),
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    #
    # autolabel(rects1)
    # autolabel(rects2)
    #
    # plt.tight_layout()
    # plt.savefig("triangles-5000.png", dpi=300)
    # plt.show()