import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import glob
import os
from collections import Counter, OrderedDict
from datetime import datetime

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def read_graph(edgelist_loc, coord_dict_loc):
    ''' load graph and dict containing coords
    of graph nodes

    :param edgelist_loc: path on disk to the
    graph's edgelist
    :param coord_dict_loc: path on disk to
    the coord_dict_loc
    :return G: rebuilt nx.DiGraph from edgelist
    :return: coord_dict: dictionary with node
    coordinates
    '''

    G = nx.read_edgelist(edgelist_loc, data=True, create_using=nx.DiGraph())
    coord_dict = np.load(coord_dict_loc, allow_pickle=True).item()

    return G, coord_dict


def sink_source_analysis(graph):
    ''' analysze how many sources
    and sinks a graph has

    :param graph: an nx.DiGraph
    :return null: only prints the number of
    sources and sinks respectively
    '''
    sinks = 0
    sources = 0
    degree = graph.degree()
    degree_list = []
    for (n, d) in degree:
        degree_list.append(d)
        if graph.in_degree(n) == 0:
            sources += 1
        elif graph.out_degree(n) == 0:
            sinks += 1

    print("sources: {}".format(sources))
    print("sinks: {}".format(sinks))
    # count = Counter(degree_list)
    # count_sorted = sorted(count, reverse=True)
    # print("subgraph_sizes: {0}".format(count))


def connected_comp_analysis(graph):
    ''' print number of connected components
    and their respective sizes '''
    graph = graph.to_undirected()
    nodes = []
    edges = []
    node_size = []
    edge_size = []
    components = [graph.subgraph(p).copy() for p in nx.connected_components(graph)]
    # print(components, type(components))
    for comp in components:
        nodes.append(comp.nodes())
        edges.append(comp.edges())
    for c in nx.connected_components(graph):
        node_size.append(len(c))
    # comp_sizes = Counter(node_size)
    # od = OrderedDict(sorted(comp_sizes.items()))
    # print(f'number of connected components is: {len(node_size)}')
    # print(f'their sizes are: {comp_sizes}')
    # for i in edges:
    #     edge_size.append(len(i))
    # print(f'they have {edge_size} edges')
    return len(node_size)


def network_density(graph):
    '''calculate network density of
    graph.

    :param graph: an nx.DiGraph
    :return null: only prints network
    density.
    '''
    # number of existing nodes
    num_nodes = nx.number_of_nodes(graph)
    # number of existing edges
    e_exist = nx.number_of_edges(graph)
    # number of potential edges
    e_pot = 3/2 * (num_nodes+1)
    # density
    dens = e_exist / e_pot
    # print(f"num_nodes is: \n\t{num_nodes}")
    # print(f"e_exist is: \n\t{e_exist}")
    # print(f"e_pot is: \n\t{e_pot}")
    # print(f"Absolute network density is: \n\t{dens}")
    return round(dens, 4)


def betweenness_centrality(graph):
    '''calculate average betweenness centrality
    for all edges.

    :param graph: an nx.DiGraph
    :return null: only prints average
    betweenness centrality.
    '''
    bet_cent = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    print(f"Average betweenness centrality is: \n\t{np.mean(list(bet_cent.values()))}\n "
          f"min: {np.min(list(bet_cent.values()))}; max: {np.max(list(bet_cent.values()))}")


def shortest_path_lengths_not_connected(graph):
    ''' iterate through list of all connected
    components in a graph to get some analysis
    insights.

    :param graph: an nx.DiGraph
    :return null: only prints average shortest path
    length, the network diameter (longest shortest
    path length), and the number of connected
    components.
    '''
    avg_short_path_length = []
    diameter = []
    for c in nx.connected_components(nx.to_undirected(graph)):
        G_sub = graph.subgraph(c)
        short_path_length = []
        for i in nx.shortest_path_length(G_sub, weight='weight'):
            for key, val in i[1].items():
                if val != 0:
                    short_path_length.append(val*50)
        avg_short_path_length.append(np.mean(short_path_length))
        if len(short_path_length) > 0:
            diameter.append(np.max(short_path_length))
    # print(f"Average shortest path lengths per component (median={np.median(avg_short_path_length)}):\n\t{sorted(avg_short_path_length, reverse=True)} m")
    # print(f"Diameter of each connected component (median={np.median(diameter)}):\n\t{sorted(diameter, reverse=True)} m")
    # print("Number of connected components in the graph:\n\t{}".format(len(avg_short_path_length)))
    max_diam = np.max(diameter)
    return round(max_diam/1000, 2)


def get_total_channel_length(graph):
    ''' calculate the total length of
     all troughs within the study area.

    :param graph:
    :return: : an nx.Graph / nx/DiGraph with true
    length of the edge as weight 'weight'.
    :return null: only prints total length of
    all channels combined.
    '''
    total_length = 0
    for (s, e) in graph.edges:
        total_length += graph[s][e]['weight']
    total_length_km = round((total_length*50)/1000, 2)
    # print("The total length of all channels in the network of the study area is:\n\t{} km".format(round((total_length*50)/1000, 2)))
    return total_length_km


def meandering_factor(graph, coord_dict):
    for (s, e) in graph.edges():
        print(s, e)
        # get coords of start and end node for each edge
        s_coord = np.array(coord_dict[str(s)])
        e_coord = np.array(coord_dict[str(e)])
        print(s_coord, e_coord)
        print(coord_dict[s])
        print(coord_dict[e])
        # calculate euclidean distance
        dist = np.linalg.norm(s_coord - e_coord)

        ps = graph[s][e]['pts']
        print('points: ', ps)

        print('channel length: ', graph[s][e]['weight']*50)
        print('direct lengths: ', dist*50)
        print('___')


def do_analysis(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    # # get sinks and sources
    # sink_source_analysis(graph)
    # number of connected components
    num_cc = connected_comp_analysis(graph)
    # delta diameter
    max_diam = shortest_path_lengths_not_connected(graph)
    # # betweenness centrality
    # betweenness_centrality(graph)
    # network density
    dens = network_density(graph)
    # length of all channels in the network
    total_channel_km = get_total_channel_length(graph)
    return num_nodes, num_edges, num_cc, dens, max_diam, total_channel_km


if __name__ == '__main__':
    startTime = datetime.now()
    # remove csv from previous runs
    if os.path.exists('./delta_metrics.csv'):
        os.remove('./delta_metrics.csv')
    # create new empty csv with header
    with open('./delta_metrics.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        header = 'filename', 'num_nodes', 'num_edges', 'num_cc', 'dens', 'max_diam', 'l_channels_km'
        writer.writerow(header)

    # iterate over delta graphs in directory
    for filename in glob.iglob(f'./graphs/*.npy'):
        # print(filename)
        fn_base = filename[9:-22]
        print(fn_base)
        # read in graph data
        G, coord_dict = read_graph(edgelist_loc=f'./graphs/{fn_base}_graph.edgelist',
                                   coord_dict_loc=f'./graphs/{fn_base}_graph_node-coords.npy')
        # meandering_factor(G, coord_dict)

        # write all metrics into csv file
        with open('./delta_metrics.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            # combine filename and unpacked metrics
            row = fn_base, *do_analysis(G)
            writer.writerow(row)

    print(datetime.now() - startTime)
    plt.show()
