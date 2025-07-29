import logging
import os

import numpy as np
from psutil import cpu_times

from Helper import Graphs, comparison_and_plot, GraphsTriangles
from count_triangles import CountTriangles

from generate_graphs import GraphGenerator
import matplotlib.pyplot as plt

import torch

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")
    graphs = {
        0: 50,
        1: 1000,
        2: 10000,
        3: 5000,
        4: 400,
        5: 200,
        6: 2500,
        7: 100,
        8: 15000,
    }
    for key, value in graphs.items():
        graph = GraphGenerator.load_graph(GraphsTriangles[key])
        graphTriangles = CountTriangles(graph.get_graph())

        DIR = "trianglesPlots"
        filename = f"triangles_{value}_prob_0_3.png"
        comparison_and_plot(
            graph._nodes,
            graph._edge_prob,
            "triangles",
            DIR,
            filename,
            graphTriangles.count_triangles_cpu,
            graphTriangles.count_triangles_gpu,
        )

    plt.show()
