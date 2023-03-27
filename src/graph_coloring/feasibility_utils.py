from dwave_networkx.algorithms.coloring import is_vertex_coloring
import numpy as np
import networkx as nx


def check_coloring(G, coloring):
    # check that the feasibility of the coloring
    feasible_coloring = is_vertex_coloring(G, coloring)
    print('Found vertex coloring? {}'.format(feasible_coloring))   


def _check_maximal(A, x):
    not_selected_nodes = np.where(x == 0)[0]
    maximal_set = True
    for node in not_selected_nodes:
        x_copy = x.copy()
        x_copy[node]=1
        if x_copy.T@A@x_copy==0:
            maximal_set = False
            break
    return maximal_set


def is_mIS(x,G):
    A = nx.to_numpy_matrix(G)
    num_conflicts = int(x.T@A@x)
    maximal_set = _check_maximal(A, x)
    is_MIS = (num_conflicts == 0 and maximal_set)
    return is_MIS