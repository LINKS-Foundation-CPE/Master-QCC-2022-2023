import osmnx as ox
import pickle
import sys

def draw_graph(G, route_list):
    plot_routes = []
    colors = []
    for index, route in enumerate(route_list):
        #generate a set from the node list
        node_list = []
        node_list.append(route[0][0])
        node_list.extend([nodes[1] for nodes in route])
        plot_routes.append(node_list)
        #colors.append('blue')
    colors = ['blue', 'red', 'green']
    fig, ax = ox.plot_graph_routes(G, plot_routes, route_colors = colors, orig_dest_size = 10, show = True)
    fig.savefig("nodes.png")

def main():
    map_file = sys.argv[1]
    G = ox.io.load_graphml(map_file)
    file_list = sys.argv[2:]
    route_list = []
    for file in file_list:
        with open(file, "rb") as f:
            n_list = pickle.load(f)
            route_list.extend(n_list)
    draw_graph(G, route_list)

if __name__ == "__main__":
    main()