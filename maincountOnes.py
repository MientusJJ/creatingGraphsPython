import logging
import os
from datetime import datetime
import re

import numpy as np
from psutil import cpu_times

from Helper import Graphs, comparison_and_plot, GraphsTriangles, GraphsOnes, DIR
from count_ones import CountOnes
from count_ones_bin_search import CountOnesBinSearch
from count_triangles import CountTriangles

from generate_graphs import GraphGenerator
import matplotlib.pyplot as plt

import torch

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")
    # graphs = {
    #     0: 50,
    #     1: 1000,
    #     2: 10000,
    #     3: 5000,
    #     4: 400,
    #     5: 200,
    #     6: 2500,
    #     7: 100,
    # }
    # size = 8
    # prob = {
    #     0: 0.3,
    #     2: 0.95,
    #     3: 0.1,
    #     5: 0.05,
    #     6: 0.001,
    #     7: 0.001,
    # }
    # s = set()
    folder_path = DIR + "/tests"
    json_filenames = [
        filename for filename in os.listdir(folder_path) if filename.endswith(".json")
    ]

    # Wypisz każdy jako string
    for name in json_filenames:
        match = re.search(r"graph_(\w+)_nodes=(\d+)_p=([\d.]+)", name)
        last_dir = os.path.basename(folder_path.rstrip("/\\"))
        name = os.path.basename(name)
        name = os.path.join(last_dir, name)
        if match:
            graph_type = match.group(1)
            nodes = int(match.group(2))
            p_value = float(match.group(3))
            print(graph_type, nodes, p_value)
            graph = GraphGenerator.load_graph(name)
            graphTriangles = CountOnesBinSearch(graph.get_graph())
            print("OK")
            DIR = "countOnesBinSearchPlots"
            filename = f"countOnesBinSearch{graph_type} {nodes}_prob_{p_value}.png"
            print(
                "▶ Czas:",
                datetime.now().strftime("%H:%M:%S"),
                "nodes:",
                nodes,
                " prob:",
                p_value,
            )
            comparison_and_plot(
                nodes,
                p_value,
                "countOnesBinSearch",
                DIR,
                filename,
                graphTriangles.multiply_until_filled_cpu,
                graphTriangles.multiply_until_filled_gpu,
            )
    # for index, p in prob.items():
    #     path = ""
    #     if index in s:
    #         path = "Path"
    #     s.add(index)
    #     for key, values in graphs.items():
    #         graph = GraphGenerator.load_graph(GraphsOnes[key + size * index])
    #         graphTriangles = CountOnes(graph.get_graph())
    #         print("OK")
    #         DIR = "countOnesPlots"
    #         filename = f"countOnes{path}{values}_prob_{p}.png"
    #         print("▶ Czas:", datetime.now().strftime("%H:%M:%S"), "nodes:", values, " prob:",p)
    #         comparison_and_plot(
    #             graph._nodes,
    #             graph._edge_prob,
    #             "countOnes",
    #             DIR,
    #             filename,
    #             graphTriangles.multiply_until_filled_cpu,
    #             graphTriangles.multiply_until_filled_gpu,
    #         )
