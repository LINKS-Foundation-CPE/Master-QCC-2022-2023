from pulser import Register
import networkx as nx
import matplotlib.pyplot as plt


def plot_initial_graph_nx(G, num_vertexes):
    mapping = dict(zip(G, range(0, num_vertexes)))
    G = nx.relabel_nodes(G, mapping)
    nx.drawing.nx_pylab.draw(G, pos=dict(G.nodes(data='pos')), with_labels=True)  
    plt.show()
    return G

def plot_initial_graph_pulser(G, rydberg_blockade_radius):
    qubits={}
    for n in G.nodes():
        qubits[n]=G._node[n]['pos']
    reg = Register(qubits)
    reg.draw(blockade_radius=rydberg_blockade_radius, draw_graph=True,  draw_half_radius=True)
    
def plot_sol(coloring, orig_G, num_colors):
    cmap=plt.get_cmap('tab10')
    new_cmap = [(1,1,1)]+list(cmap.colors)
    if num_colors==-1:
        coloring_copy =  dict.fromkeys(orig_G.nodes(), new_cmap[0])
    else:
        # represent the iterative coloring
        for col in range(1, num_colors+1):
            coloring_copy = {}
            for key, val in coloring.items():
                if val <= col and val!=-1:
                    coloring_copy[key]=new_cmap[val]
                else:
                    coloring_copy[key]=new_cmap[0]
    f = plt.figure()
    nx.draw(orig_G, pos=dict(orig_G.nodes(data='pos')), node_color=list(coloring_copy.values()), with_labels=True, node_size=500,
            font_weight="bold", node_shape="o", ax=f.add_subplot(111))            
    ax= plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    plt.tight_layout()
    plt.show()