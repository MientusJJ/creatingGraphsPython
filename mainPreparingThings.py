import ast
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
import time

from matplotlib import pyplot as plt

from Helper import (
    DIR_non_graphs,
    Graphs,
    show_adjacency_tensor,
    ms,
    DIR,
    extract_nodes_and_p_preparing_graphs, extract_nodes_and_p_reading, DIR_tests, DIR_stanford,
)
import os
from collections import defaultdict

from count_ones import CountOnes
from count_ones_bin_search import CountOnesBinSearch
from count_triangles import CountTriangles
from dynamic_reachability import DynamicReachAbility
from dynamic_sherman import DynamicSherman
from generate_graphs import GraphGenerator

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ PyTorch nie wykrył GPU! Program może nie działać poprawnie.")
    else:
        print(f"✅ PyTorch wykrył GPU: {torch.cuda.get_device_name(0)}")
    mapa = {
        "sherman",

    }
    for value in mapa:

            folder = os.path.join(os.path.join(DIR_stanford,"directed"))
            print(folder)
            files = glob.glob(os.path.join(folder, "*.txt"))
            for filename in files:
                max_value = float('-inf')  # wartość początkowa – najmniejsza możliwa

                with open(filename, "r") as f:
                    for line in f:
                        # Wyciągnij wszystkie liczby z linii
                        numbers = list(map(int, re.findall(r'\d+', line)))
                        if numbers:
                            max_in_line = max(numbers)
                            max_value = max(max_value, max_in_line)
                max_value += 1
                print("Najwyższa wartość:", max_value)
                graph = DynamicReachAbility(GraphGenerator(nodes=max_value,graph_type="empty").get_graph()) if value == "sankowski" else DynamicSherman(GraphGenerator(nodes=max_value,graph_type="empty").get_graph())
                with open(filename, "r") as infile:
                    counter = 0
                    starttime = time.time()
                    for line in infile:
                        line = line.strip()
                        if line:
                            try:
                                # Rozbij na część z src i dst
                                src_part, dst_part = line.split(", dst:")
                                src = int(src_part.replace("src:", "").strip())
                                row = torch.zeros(graph._adj_tensor.shape[0], device="cuda")
                                row[ ast.literal_eval(dst_part)] = 1
                                graph.update_one_row_or_col(row, src, "row") if value == "sankowski" else graph.add_one_vector(row,src,"row")
                                print("OK",counter)
                                counter += 1
                            except Exception as e:
                                print(f"Błąd w linii: {line}")
                                raise e
                endtime = ms * (time.time() - starttime)
                with open(os.path.join(os.path.join(DIR_stanford,"undirected"), "results.txt"), "a",
                          encoding="utf-8") as f:
                    filepath = Path(filename)
                    name = filepath.name
                    f.write(f"{value} {name}: {endtime}\n" if "DAG" not in filename else f"DAG {value} {name}: {endtime}\n")
                print(endtime)
    # def parse_line(line, default_graph_type="plain"):
    #     pattern = r'(?:(\w+)_)?(sankowski|sherman)_(\d+)_(\d+(?:\.\d+)?):\s*([\d.]+)'
    #     match = re.match(pattern, line.strip())
    #     if not match:
    #         return None
    #
    #     graph_type, method, nodes, prob, time_ms = match.groups()
    #     graph_type = graph_type if graph_type else default_graph_type
    #     return graph_type, method, int(nodes), float(prob), float(time_ms)
    #
    #
    # def load_data(folder,*filenames):
    #     data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    #     for filename in filenames:
    #         with open(os.path.join(folder,filename), "r") as f:
    #             for line in f:
    #                 parsed = parse_line(line)
    #                 if parsed:
    #                     graph_type, method, nodes, prob, time = parsed
    #                     data[graph_type][prob][method][nodes] = time
    #     return data
    #
    #
    # def plot_data(folder, data):
    #     for graph_type, prob_dict in data.items():
    #         for prob, method_dict in sorted(prob_dict.items()):
    #             plt.figure()
    #             methods = sorted(method_dict.keys())
    #             node_sets = set()
    #             for method in methods:
    #                 node_sets.update(method_dict[method].keys())
    #             nodes_sorted = sorted(node_sets)
    #
    #             width = 0.35  # szerokość słupków
    #             x = np.arange(len(nodes_sorted))  # pozycje X
    #
    #             for idx, method in enumerate(methods):
    #                 times = [method_dict[method].get(node, np.nan) for node in nodes_sorted]
    #                 offset = width * idx - (width * (len(methods) - 1) / 2)
    #                 plt.bar(x + offset, times, width=width, label=method)
    #
    #             plt.title(f"{graph_type.upper()} | Prawdopodobieństwo: {prob}")
    #             plt.xlabel("Liczba wierzchołków")
    #             plt.ylabel("Czas (ms, skala log)")
    #             plt.xticks(x, nodes_sorted)
    #             plt.yscale("log")
    #             plt.legend()
    #             plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #             plt.tight_layout()
    #
    #             filename = os.path.join(folder, f"{graph_type.upper()}_prob_{prob}.png")
    #             plt.savefig(filename)
    #             plt.close()
    #
    #
    # folder = os.path.join(DIR_stanford,"undirected")
    # graf = "facebook_combined_raw.txt"
    # method = "CountOnes"
    # graph = CountOnesBinSearch(GraphGenerator.load_graph_stanford(os.path.join(folder,graf)).get_graph())
    # result, time,_,_ = graph.multiply_until_filled_gpu()
    # result2,time2,_,_ = graph.multiply_until_filled_cpu()
    #
    # output_path = os.path.join(folder,"results.txt")
    # with open(output_path, "a") as f:
    #         graf = graf.replace("_raw.txt", "")
    #
    #         f.write(f"{method} {graf}, gpu : {time} ms\n")
    #         f.write(f"{method} {graf}, cpu : {time2} ms\n")
    # graph = CountOnes(GraphGenerator.load_graph_stanford(os.path.join(DIR_non_graphs,"facebook_combined.txt" ),True).get_graph())
    # _, time,res,_ = graph.multiply_until_filled_gpu()
    # # with open(os.path.join(DIR_tests, "results_Stanford.txt"), "a",
    # #                 encoding="utf-8") as f:
    # print(time,res)
    # _, time,_,_ = graph.multiply_until_filled_gpu()
    # print(time)
