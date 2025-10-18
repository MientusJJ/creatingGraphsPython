import logging

import numpy as np
import time
from tensorboard.notebook import start
from torch._dynamo.polyfills import sys

from Helper import DIR, Graphs, show_adjacency_tensor, ms
from count_ones import CountOnes
from count_ones_bin_search import CountOnesBinSearch
from count_triangles import CountTriangles
from dynamic import Dynamic
from dynamic_reachability import DynamicReachAbility
from dynamic_sherman import DynamicSherman
from generate_graphs import GraphGenerator
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
import os
import psutil

from optimized_multiplication import multiply_sparse_cols_C
from PIL import Image
import os

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")

    graph = GraphGenerator(nodes=2500, graph_type="empty")
    graph.save_graph()

    # Posortuj pliki numerycznie według nazw np. 1.png, 2.png, ..., 15.png

    # Posortuj pliki numerycznie np. 1.jpg, 2.jpg, ..., 15.jpg
    # nodes = [
    #     #50,
    #     #1000,
    #     #10000,
    #     #5000,
    #     #400,
    #     #200,
    #     #2500,
    #     #100,
    #     8000
    # ]
    # prob = [0.001,0.05,0.1,0.95,0.3]
    # for v in nodes:
    #     for p in prob:
    #         graph = GraphGenerator(nodes=v, edge_prob=p)
    #         graph.save_graph()
    # graph = GraphGenerator(nodes=8000,graph_type="path",self_loops=True)
    # graph.save_graph(test=True)
    # # A = torch.tensor(
    # #     [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=torch.float64
    # # ).cuda()
    # # print(A)
    # # graphs = GraphGenerator(10,0.5,54,"dag")
    # # graphs.save_graph()
    # # PRZECZYTAJ:
    # # file: // / C: / Users / HardPC / Downloads / dynamic - transitive - closure - via - dynamic - matrix - inverse - extended - a.pdf
    # # strona 4
    # row_0 = torch.zeros(10, device="cuda")
    # col_3 = torch.zeros(10, device="cuda")
    # row_1 = torch.zeros(10, device="cuda")
    # col_6 = torch.zeros(10, device="cuda")
    # col_5 = torch.zeros(10, device="cuda")
    # row_3 = torch.zeros(10, device="cuda")
    # col_8 = torch.zeros(10, device="cuda")
    # row_8 = torch.zeros(10, device="cuda")
    # row_0[2] = row_0[3] = row_0[5] = row_0[9] = 1
    # col_3[0] = col_3[1] = 1
    # row_1[8] = row_1[9] = 1
    # col_6[2] = col_6[4] = 1
    # col_5[3] = col_5[4] = 1
    # row_3[7] = row_3[9] = 1
    # col_8[1] = col_8[7] = 1
    # row_8[9] = 1
    # tensor = torch.zeros(10, device="cuda")
    # tensor[8:10] = 1
    # tensor[8:10] = 1
    # graphIndex = 16
    # print(tensor)
    # graph = GraphGenerator.load_graph(Graphs[graphIndex])
    # graph_dynamic = DynamicReachAbility(graph.get_graph())
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # graph_dynamic.update_one_row_or_col(tensor, 5, "row")
    #
    # graph_dynamic.update_one_row_or_col(row_0, 0, "row")
    # graph_dynamic.update_one_row_or_col(col_3, 3, "col")
    # graph_dynamic.update_one_row_or_col(row_1, 1, "row")
    # graph_dynamic.update_one_row_or_col(col_6, 6, "col")
    # graph_dynamic.update_one_row_or_col(col_5, 5, "col")
    # graph_dynamic.update_one_row_or_col(row_3, 3, "row")
    # graph_dynamic.update_one_row_or_col(col_8, 8, "col")
    # graph_dynamic.update_one_row_or_col(row_8, 8, "row")
    # # # print(graph_dynamic.find_path(2,5))
    # # # print(graph_dynamic.find_path(2,8))
    # # # print(graph_dynamic.find_path(2,9))
    # # #show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # # # graph_dynamic.update_one_row_or_col(tensor,2,"row")
    # # # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # # # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # tensor = torch.zeros(10, device="cuda")
    # tensor[2:5] = 1
    # graph_dynamic.update_one_row_or_col(tensor, 5, "col")
    # for i in range(10):
    #     for j in range(i+1,10):
    #         print(i, j,graph_dynamic.find_path(i, j) )
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # show_adjacency_tensor(graph_dynamic._adj_tensor)

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
    # graphIndex = 15
    #
    # graph = GraphGenerator.load_graph(Graphs[graphIndex])
    # # TODO
    # graph_dynamic = DynamicReachAbility(graph.get_graph())
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # start = time.time()
    # graph_dynamic.update_one_cell(0, 2, 1)
    # graph_dynamic.update_one_cell(0, 3, 1)
    # graph_dynamic.update_one_cell(1, 3, 1)
    # graph_dynamic.update_one_cell(0, 5, 1)
    # graph_dynamic.update_one_cell(0, 9, 1)
    # graph_dynamic.update_one_cell(1, 8, 1)
    # graph_dynamic.update_one_cell(1, 9, 1)
    # graph_dynamic.update_one_cell(2, 5, 1)
    # graph_dynamic.update_one_cell(2, 6, 1)
    # graph_dynamic.update_one_cell(3, 5, 1)
    # graph_dynamic.update_one_cell(3, 7, 1)
    # graph_dynamic.update_one_cell(3, 9, 1)
    # graph_dynamic.update_one_cell(4, 5, 1)
    # graph_dynamic.update_one_cell(4, 6, 1)
    # graph_dynamic.update_one_cell(5, 8, 1)
    # graph_dynamic.update_one_cell(5, 9, 1)
    # graph_dynamic.update_one_cell(7, 8, 1)
    # graph_dynamic.update_one_cell(8, 9, 1)
    # endtime = ms * (time.time() - start)
    # print(endtime)
    # graph_dynamic._update_matrix()
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)

    # row_0 = torch.zeros(10, device="cuda")
    # col_3 = torch.zeros(10, device="cuda")
    # row_1 = torch.zeros(10, device="cuda")
    # col_6 = torch.zeros(10, device="cuda")
    # col_5 = torch.zeros(10, device="cuda")
    # row_3 = torch.zeros(10, device="cuda")
    # col_8 = torch.zeros(10, device="cuda")
    # row_8 = torch.zeros(10, device="cuda")
    # row_0[2] = row_0[3] = row_0[5] = row_0[9] = 1
    # col_3[0] = col_3[1] = 1
    # row_1[8] = row_1[9] = 1
    # col_6[2] = col_6[4] = 1
    # col_5[3] = col_5[4] = 1
    # row_3[7] = row_3[9] = 1
    # col_8[1] = col_8[7] = 1
    # row_8[9] = 1
    # tensor = torch.zeros(10, device="cuda")
    # tensor[8:10] = 1
    # #tensor[8:10] = 1
    # graphIndex = 15
    # #print(tensor)
    # graph = GraphGenerator.load_graph(Graphs[graphIndex])
    # graph_dynamic = DynamicReachAbility(graph.get_graph())
    # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # start_time = time.time()
    # graph_dynamic.update_one_row_or_col(tensor, 5, "row")
    # row_0[2] = row_0[3] = row_0[5] = row_0[9] = 1
    # graph_dynamic.update_one_cell(0, 2, 1)
    # graph_dynamic.update_one_cell(0, 3, 1)
    # graph_dynamic.update_one_cell(0, 9, 1)
    # graph_dynamic.update_one_cell(0, 5, 1)
    # graph_dynamic.update_one_row_or_col(col_3, 3, "col")
    # graph_dynamic.update_one_row_or_col(row_1, 1, "row")
    # graph_dynamic.update_one_row_or_col(col_6, 6, "col")
    # col_5[3] = col_5[4] = 1
    # graph_dynamic.update_one_cell(4, 5, 1)
    # graph_dynamic.update_one_cell(3, 5, 1)
    # graph_dynamic.update_one_row_or_col(row_3, 3, "row")
    # graph_dynamic.update_one_row_or_col(col_8, 8, "col")
    # graph_dynamic.update_one_row_or_col(row_8, 8, "row")
    # print(graph_dynamic.find_path(2,5))
    # print(graph_dynamic.find_path(2,8))
    # print(graph_dynamic.find_path(2,9))
    # # graph_dynamic.update_one_row_or_col(tensor,2,"row")
    # # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # # show_adjacency_tensor(graph_dynamic._adj_tensor)
    # tensor = torch.zeros(10, device="cuda")
    # tensor[2:5] = 1
    # graph_dynamic.update_one_row_or_col(tensor, 5, "col")
    # end_time = (time.time() - start_time) * ms
    # print(end_time, "ms")
    # print(graph_dynamic.find_path(2, 5))
    # print(graph_dynamic.find_path(2, 8))
    # print(graph_dynamic.find_path(2, 9))
    # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # show_adjacency_tensor(graph_dynamic._adj_tensor)

    #
    # row_0 = torch.zeros(10, device="cuda")
    # col_3 = torch.zeros(10, device="cuda")
    # row_1 = torch.zeros(10, device="cuda")
    # col_6 = torch.zeros(10, device="cuda")
    # col_5 = torch.zeros(10, device="cuda")
    # row_3 = torch.zeros(10, device="cuda")
    # col_8 = torch.zeros(10, device="cuda")
    # row_8 = torch.zeros(10, device="cuda")
    # row_0[2] = row_0[3] = row_0[5] = row_0[9] = 1
    # col_3[0] = col_3[1] = 1
    # row_1[8] = row_1[9] = 1
    # col_6[2] = col_6[4] = 1
    # col_5[3] = col_5[4] = 1
    # row_3[7] = row_3[9] = 1
    # col_8[1] = col_8[7] = 1
    # row_8[9] = 1
    # tensor = torch.zeros(10, device="cuda")
    # tensor[8:10] = 1
    # # tensor[8:10] = 1
    # graphIndex = 15
    # # print(tensor)
    # graph = GraphGenerator.load_graph(Graphs[graphIndex])
    # graph_sherman = DynamicSherman(graph.get_graph())
    # show_adjacency_tensor(graph_sherman._adj_tensor)
    # show_adjacency_tensor(graph_sherman._graph_n_adj)
    # start = time.time()
    # graph_sherman.add_one_vector(tensor, 5, "row")
    #
    # graph_sherman.add_one_vector(row_0, 0, "row")
    # graph_sherman.add_one_vector(col_3, 3, "col")
    # graph_sherman.add_one_vector(row_1, 1, "row")
    # graph_sherman.add_one_vector(col_6, 6, "col")
    # graph_sherman.add_one_vector(col_5, 5, "col")
    # graph_sherman.add_one_vector(row_3, 3, "row")
    # graph_sherman.add_one_vector(col_8, 8, "col")
    # graph_sherman.add_one_vector(row_8, 8, "row")
    # # print(graph_sherman.find_path(2,5))
    # # print(graph_sherman.find_path(2,8))
    # # print(graph_sherman.find_path(2,9))
    # # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # # show_adjacency_tensor(graph_dynamic._graph_n_adj)
    # # show_adjacency_(graph_dynamic._adj_tensor)
    # tensor = torch.zeros(10, device="cuda")
    # tensor[2:5] = 1
    # graph_sherman.add_one_vector(tensor, 5, "col")
    # # print(graph_sherman.find_path(0, 2))
    # # print(graph_sherman.find_path(0, 3))
    # # print(graph_sherman.find_path(0, 5))
    # # print(graph_sherman.find_path(0, 7))
    # # print(graph_sherman.find_path(0, 8))
    # # print(graph_sherman.find_path(0, 9))
    # # print(graph_sherman.find_path(1, 8))
    # # print(graph_sherman.find_path(1, 9))
    # # print(graph_sherman.find_path(3, 7))
    # # print(graph_sherman.find_path(3, 9))
    # # print(graph_sherman.find_path(5, 8))
    # # print(graph_sherman.find_path(5, 9))
    # # print(graph_sherman.find_path(8, 9))
    # #graph_sherman._update_matrix()
    # # for i in range(10):
    # #     for j in range(i + 1, 10):
    # #         print(i, j, graph_sherman.find_path(i, j))
    # end = (time.time() - start) * ms
    # print(end, "ms")
    # graph_sherman._update_matrix()
    #
    # show_adjacency_tensor(graph_sherman._graph_n_adj)
    # show_adjacency_tensor(graph_sherman._adj_tensor)
