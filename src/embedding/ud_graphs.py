import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
from itertools import combinations



def load_graph(filename, max_dist, set_init_pos, dim, equilibrium_dist=7):
    """ Upload graph `G` from saved as gpickle object and compute initial positions either by 
        scaling original positions or by applying Fruchterman-Reingold algorithm.

    Args:
        filename ( string ): path of the gpickle object to upload.
        max_dist ( float ): maximum distance allowed between the origin and the vertexes.
        set_init_pos ( string ): method for computing initial positions (allowed 'scaling' or 'FR').
        dim ( int ): number of dimensions for the coordinates (allowed 2 or 3).
        equilibrium_dist ( float , optional): equilibrium distance for Fruchterman-Reingold algorithm. Defaults to 7.

    Returns:
         networkx.Graph : A undirected graph.
         numpy.array : initial coordinates
    """
    G = nx.read_gpickle(filename)
    A = nx.to_numpy_matrix(G)
    num_vertexes = A.shape[0]
    if set_init_pos=='scaling':
        # scale the original coordinates to a circular domain with radius `max_dist` 
        pos = nx.get_node_attributes(G,'pos')
        pos = np.array(list(pos.values()))    
        dist_matrix = distance_matrix(pos, pos)
        max_dist_n1 = dist_matrix.argmax()//num_vertexes
        max_dist_n2 = dist_matrix.argmax()%num_vertexes
        center = (pos[max_dist_n1]+pos[max_dist_n2])/2
        distances_from_center = np.linalg.norm(pos-center, axis=1)
        max_dist_center = distances_from_center.max()
        init_pos = (pos-center)/max_dist_center*max_dist
        if dim==3:
            # the z-coordinates are initialized to zero
            init_pos=np.concatenate((init_pos, np.zeros((len(G.nodes()),1))), axis=1)
    elif set_init_pos=='FR':
        # initialize coordinates with Fruchterman-Reingold algorithm, the domain is the square/cube inscribed in the 
        # circle/sphere
        scale=max_dist*np.sqrt(2)/2
        pos_dict = nx.spring_layout(G, scale=scale, iterations=1000, k=equilibrium_dist, dim=dim, seed=42)
        init_pos = np.array(list(pos_dict.values()))
    # update vertexes positions in G
    for n in G.nodes():
        G.add_node(n,pos=init_pos[n,:])
    return G, init_pos
    
    
def compute_adj_mtx(G):
    """ Compute adjacency matrix of the graph `G` and of the complement graph of `G`.

    Args:
        G ( networkx.Graph ): A undirected graph.

    Returns:
         numpy.array : adjacency matrix of the graph `G`.
         numpy.array : adjacency matrix of the complement graph of `G`.
    """
    A = nx.to_numpy_matrix(G)
    not_A = np.ones(A.shape)-A
    np.fill_diagonal(not_A,0)  
    return A, not_A


def compute_dists_minmax(D, A, not_A):
    """ Compute minima and maxima distances according to the adjacency pattern of matrix `A`.

    Args:
        D ( numpy.array ): Distance matrix for all pairs of vertexes in a graph G
        A ( numpy.array ): Adjacency matrix of a graph G
        not_A ( numpy.array ): Adjacency matrix of the complement graph of G

    Returns:
         float : minimum pair distance.
         float : maximum pair distance.
         float : maximum pair distance of adjacent vertexes in G.
         float : minimum pair distance of not adjacent vertexes in G.
    """
    if not_A.sum()==0:
        # the graph is completed: there are no not adjacent vertexes
        min_dist_not_adj = None
    else:
        min_dist_not_adj= D[not_A==1].min()
    max_dist_adj = D[A==1].max()
    max_dist = D.max()
    min_dist = D[A==1].min()
    return min_dist, max_dist, max_dist_adj, min_dist_not_adj

def compute_pairs_dict(vertexes):
    """ Compute a dictionary from all unordered pairs of vertexes.

    Returns:
            dict : dictionary with incremental keys and tuples of vertexes pairs as values.
    """
    pairs_dict = {}
    all_pairs = list(combinations(vertexes, 2))
    idx=0
    for pair in all_pairs:
        pairs_dict[idx]=pair
        idx+=1
    return pairs_dict

def check_feasible_embedding(D, A, not_A, min_dist_value, max_dist_value, max_adj_value):
    """ Check if the pair distances corresponding to the embedding are feasible.

    Args:
        D ( numpy.array ): matrix of distances for all the pairs.
        A (numpy.array ): adjacency matrix of a graph `G`.
        not_A ( numpy array ): adjacency matrix of the complement graph of `G`.
        min_dist_value ( float ): minimum allowed distance among vertexes.
        max_dist_value ( float ): maximum allowed distance among vertexes.
        max_adj_value ( float ): maximum allowed distance among adjacent vertexes.
        
    Returns:
         boolean : *True* if the embedding is feasible.
         float : adjacency gap.
    """
    min_dist, max_dist, max_dist_adj, min_dist_not_adj = compute_dists_minmax(D, A, not_A) 
    gap = min_dist_not_adj - max_dist_adj
    if min_dist >= min_dist_value and max_dist<=max_dist_value and max_dist_adj<=max_adj_value and min_dist_not_adj>max_adj_value:
        return True, gap
    else:
        return False, gap  