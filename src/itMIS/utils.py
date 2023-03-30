from pulser.devices import Chadoq2
import networkx as nx
from scipy.spatial import distance_matrix
import numpy as np
from numpy.linalg import eigh
import math

def compute_rydberg(G):
    pos_list =[]
    for n in G.nodes():
        pos_list.append(G._node[n]['pos'])
    pos=np.array(pos_list) 
    # find the rydberg blockade radius
    dist_matrix = distance_matrix(pos, pos)
    A = nx.to_numpy_matrix(G)
    blockade_radius = dist_matrix[A==1].max() 
    rabi_freq = Chadoq2.rabi_from_blockade(blockade_radius)
    return rabi_freq, blockade_radius

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

def is_MIS(x,G):
    A = nx.to_numpy_matrix(G)
    num_conflicts = int(x.T@A@x)
    maximal_set = _check_maximal(A, x)
    is_MIS = (num_conflicts == 0 and maximal_set)
    return is_MIS

def compute_subgraph(x, G):
    MIS_set = []
    node_set = list(G.nodes())
    for node in range(len(x)):
        if x[node] == 1:
            MIS_set.append(node_set[node])
    remaining_nodes = set(node_set).difference(set(MIS_set))
    H = G.subgraph(remaining_nodes)    
    return H, MIS_set

def NetworkxGC(G):
    strategies = ['largest_first', 'random_sequential', 'smallest_last', 'independent_set', 'connected_sequential_bfs', 'connected_sequential_dfs', 'saturation_largest_first']
    for strategy in strategies:
        coloring = nx.coloring.greedy_color(G, strategy=strategy)
        num_colors=max(coloring.values())+1
        print('Networkx solution with {} strategy uses {} colors'.format(strategy, num_colors))

def compute_obj(G, colors_used, previous_coloring):
    if len(G.edges())>0: 
        # coloring starts with index 0
        obj = colors_used+len(G.nodes())
        # for greedy_color_node in coloring.keys():
        greedy_color = colors_used
        for remaining_node in G.nodes():
            greedy_color += 1
            previous_coloring[remaining_node]=greedy_color
    else:
        for node in G.nodes():
            previous_coloring[node]=colors_used+1
        obj = colors_used+1
    return obj, previous_coloring

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
            hist_str+='\t Step {} -> MIS solution at position {}\n'.format(step, idx)
    print(hist_str)

def print_itMIS(coloring_dict):
    MIS_dict={}
    #initializing an empty dictionary for inverting
    for node, color in coloring_dict.items():
    #iterative step for each key value pair in books_copies dictionary
        if color in MIS_dict:
            MIS_dict[color].append(node)
        else:
            MIS_dict[color] = [node]
    for color, node_set in MIS_dict.items():
        print('Step {} -> nodes in MIS solution {}'.format(color+1, node_set))
