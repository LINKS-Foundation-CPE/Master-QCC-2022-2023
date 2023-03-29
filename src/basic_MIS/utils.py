import networkx as nx
from pulser.devices import Chadoq2
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def to_bitstring(s): 
    str1 = "" 
    for ele in s: 
        str1 += str(ele)  
    return str1 

def pos_to_graph(pos):
    rb = Chadoq2.rydberg_blockade_radius(1.)
    g = nx.Graph()
    N = len(pos)
    edges = [(m,n) for m,n in combinations(range(N), r=2) if np.linalg.norm(pos[m] - pos[n]) < rb]
    for i in range(len(pos)):
        g.add_node(i,pos=pos[i])
    g.add_edges_from(edges)
    return g

def nx_mis(G):
    MIS_list = []
    max = 0
    for i in range(20):
        tmp_max = 0
        bitlist = [0 for _ in range(len(G.nodes()))]
        tmp = nx.maximal_independent_set(G, seed=i)
        for node in tmp:
            bitlist[node] = 1
            tmp_max += 1
        if tmp_max >= max:
            max = tmp_max
        MIS_list.append(to_bitstring(bitlist))

    MIS_list_max = []
    for item in MIS_list:
        ones = 0
        for c in item:
            if c == '1':
                ones += 1
        if ones == max:
            MIS_list_max.append(item)
    
    MIS_set = set(MIS_list_max)
    return MIS_set

def get_mis_string(count_dict):
    return sorted(count_dict.items(), key=lambda item: item[1], reverse=True)[0][0]

def plot_distribution(C, mis_ref):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = list(mis_ref)  # MIS indexes
    color_dict = {key:'coral' if key in indexes else 'royalblue' for key in C}
    plt.figure(figsize=(12,6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color = color_dict.values())
    plt.xticks(rotation='vertical')
    plt.show()