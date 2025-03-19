from count_ones import CountOnes
from count_ones_bin_search import CountOnesBinSearch
from count_triangles import CountTriangles
from generate_graphs import GraphGenerator

if __name__ == "__main__":
    nodes = 100
    edge_prob = 0.003

    graph = GraphGenerator.load_graph_stanford("D:\\Program Files\\Magisterka\\Grafy\\stanford\\test.txt")
    ones = CountOnes(graph.get_graph())
    print(ones.multiply_until_filled_cpu())
    onesBS  = CountOnesBinSearch(graph.get_graph())
    print(onesBS.multiply_until_filled_cpu())
    graph.save_graph()