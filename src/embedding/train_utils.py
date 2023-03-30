import torch
import numpy as np
from itertools import combinations
import networkx as nx
import random
import os
import matplotlib.pyplot as plt

def compute_adj_tensor(A):
    """Compute the tensor that describe the adjacency pattern among all vertexes' pairs.

    Args:
        A ( numpy.array ): adjacency matrix.

    Returns:
         torch.Tensor : adjacency tensor.
    """
    adj_list = []
    num_nodes = A.shape[0]
    all_pairs = list(combinations(range(num_nodes), 2))
    for pair in all_pairs:
        (i,j) = pair
        adj_list.append(A[i,j])            
    adj_tensor = torch.Tensor(adj_list)
    adj_tensor.requires_grad=True
    return adj_tensor


def set_seed(seed):
    """Initialize seed

    Args:
        seed ( int ): seed for pseudo-random generation
    """    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHEDSEED']=str(seed)
    torch.use_deterministic_algorithms(True)
    
    
def positions_dict(G, init_pos, new_pos):
    node_idx = 0
    init_pos_dict={}
    feasible_pos_dict={}
    for n in G.nodes():
        init_pos_dict[n] = init_pos[node_idx,:]
        feasible_pos_dict[n] = new_pos[node_idx,:]
        node_idx+=1
    return init_pos_dict, feasible_pos_dict


def visualize_3d(G, pos):
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w")
    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")
    ax.set_xlabel('x ($\mu m$)')
    ax.set_ylabel('y ($\mu m$)')
    ax.set_zlabel('z ($\mu m$)')
    fig.tight_layout()
    plt.show()
    plt.close()
    
def visualize_2d(G, pos_dict):
    fig, ax = plt.subplots()
    nx.drawing.nx_pylab.draw(G, pos=pos_dict,  with_labels=True, ax=ax)
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('x ($\mu m$)')
    plt.ylabel('y ($\mu m$)')
    plt.show()
    plt.close()