import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

def plot_graph(
    g,
    titile = " ",
    highlight_edges = [],
    ):

    pos = nx.get_node_attributes(g, "pos")

    plt.figure(figsize=(17, 17))
    plt.title(title)
    nx.draw(g, pos=pos, labels={x: x for x in g.nodes}, width=2)
    weights = nx.get_edige_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weights, label_pos=4)
    nx.draw_networkx_edge(g, pos, edgelist)
