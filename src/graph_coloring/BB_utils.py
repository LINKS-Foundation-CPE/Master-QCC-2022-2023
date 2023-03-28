import networkx as nx
import numpy as np
from numpy.linalg import eigh
import math
import matplotlib.pyplot as plt


def compute_subgraph(x, G):
    MIS_set = []
    node_set = list(G.nodes())
    for node in range(len(x)):
        if x[node] == 1:
            MIS_set.append(node_set[node])
    remaining_nodes = set(node_set).difference(set(MIS_set))
    H = G.subgraph(remaining_nodes)    
    return H, MIS_set

def compute_LB(x, G):
    H, MIS_set = compute_subgraph(x, G)
    num_edges = len(H.edges())
    A = nx.to_numpy_matrix(H)    
    if num_edges>0: 
        remaining_nodes = len(H.nodes())            
        # this method is aware that matrix A is symmetric
        eigs, _ = eigh(A)
        eigs=np.sort(eigs)
        lambda_1=eigs[-1]
        lambda_n=eigs[0]
        n_plus=sum(eigs>0)
        n_minus=sum(eigs<0)        
        # compute lower bound  
        HoffmanLB = math.floor(1-lambda_1/lambda_n)
        ElphicLB = math.floor(1+max(n_minus/n_plus, n_plus/n_minus))
        EdwardsLB = math.floor(remaining_nodes/(remaining_nodes-lambda_1))
        LB=max([HoffmanLB, EdwardsLB, ElphicLB])
    else:
        if remaining_nodes==0:
            LB=0
        else:
            LB=1
    return H, LB, MIS_set

def compute_UB(H):
    A = nx.to_numpy_matrix(H)
    degrees = sum(A)
    degrees=-np.sort(-degrees)
    max_degree = degrees[0,0]
    UB_array = np.zeros((degrees.shape[1]))
    for i in range(degrees.shape[1]):
        UB_array[i] = min(degrees[0,i]+1,i)
    UB_chr_number = np.max(UB_array)
    UB = int(min(max_degree+1, UB_chr_number))
    return UB

def fingerprint(vertexes):
    fp = -1
    for i in vertexes:
        fp+=2**i
    return fp

def print_BB_history(story_dict):
    hist_str='Best solution history:\n'
    for step, idx in story_dict.items():
        if step !=0:
            hist_str+='\t Step {} -> selected MIS solution at position {}\n'.format(step, idx)
    print(hist_str)
    
def print_mIS(coloring_dict):
    MIS_dict={}
    #initializing an empty dictionary for inverting
    for node, color in coloring_dict.items():
    #iterative step for each key value pair in books_copies dictionary
        if color in MIS_dict:
            MIS_dict[color].append(node)
        else:
            MIS_dict[color] = [node]
    for color, node_set in sorted(MIS_dict.items()):
        print('Step {} -> nodes in MIS solution {}'.format(color, node_set))
    
def plot_sol(orig_G, coloring, num_colors):
        cmap=plt.get_cmap('tab10')
        new_cmap = [(1,1,1)]+list(cmap.colors)
        # represent the iterative coloring
        for col in range(1, num_colors+1):
            coloring_copy = {}
            for key, val in coloring.items():
                if val <= col:
                    coloring_copy[key]=new_cmap[val]
                else:
                    coloring_copy[key]=new_cmap[0]
        
        nx.draw(orig_G, pos=dict(orig_G.nodes(data='pos')), node_color=list(coloring_copy.values()), with_labels=True, node_size=500,
                font_weight="bold", node_shape="o")            
        plt.show()
        plt.close()
