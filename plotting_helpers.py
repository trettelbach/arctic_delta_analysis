import pickle
import scipy
import scipy.ndimage
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from b_extract_trough_transects import read_graph
from a_delta_to_graph import read_data
from datetime import datetime
import matplotlib
# import seaborn as sns
# import pandas as pd
from collections import Counter


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def add_params_graph(G, edge_param_dict):
    ''' take entire transect dictionary and
    the original graph G and add the mean/median
    parameter values to the graph edges.

    :param G: trough network graph created
    from skeleton
    :param edge_param_dict: dictionary with
    - key: edge (s, e) and
    - value: list with
        - mean width [m]
        - median width [m]
        - mean depth [m]
        - median depth [m]
        - mean r2
        - median r2
        - ratio of considered transects/trough
        - ratio of water-filled troughs
    :return : graph with added edge_param_dict
    parameters added as edge weights.
    '''
    num_emp = 0
    num_full = 0

    # iterate through all graph edges
    for (s, e) in G.edges():
        # and retrieve information on the corresponding edges from the dictionary
        if (s, e) in edge_param_dict:  # TODO: apparently some (xx) edges aren't in the edge_param_dict. check out why
            G[s][e]['mean_width'] = edge_param_dict[(s, e)][0]
            G[s][e]['median_width'] = edge_param_dict[(s, e)][1]
            G[s][e]['mean_depth'] = edge_param_dict[(s, e)][2]
            G[s][e]['median_depth'] = edge_param_dict[(s, e)][3]
            G[s][e]['mean_r2'] = edge_param_dict[(s, e)][4]
            G[s][e]['median_r2'] = edge_param_dict[(s, e)][5]
            G[s][e]['considered_trans'] = edge_param_dict[(s, e)][6]
            G[s][e]['water_filled'] = edge_param_dict[(s, e)][7]
            num_full += 1
        else:
            # print("{} was empty... (border case maybe?)".format(str((s, e))))
            num_emp += 1
    print(num_emp, num_full)


def plot_graph_by_weight(graph, col_parameter, coord_dict, bg):
    micr = bg

    G_no_borders = graph.copy()
    for (s, e) in graph.edges():
        # print(graph[s][e])
        if col_parameter in graph[s][e]:
            # print(G_no_borders[s][e]['mean_depth'])
            continue
        else:
            G_no_borders.remove_edge(s, e)

    f = plt.figure(1, figsize=(3.75, 2.45), dpi=300)  # 900
    ax = f.add_subplot(1, 1, 1)

    colors = [G_no_borders[s][e][col_parameter] for s, e in G_no_borders.edges()]

    colors_nonnan = []
    for i in colors:
        if i > 0:
            colors_nonnan.append(i)

    colors = np.nan_to_num(colors, nan=np.mean(colors_nonnan), copy=True)

    if col_parameter == 'mean_width':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        # draw edge by weight
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors, edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0, edge_vmax=10)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        # nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=10))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('width [m]')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == 'mean_depth':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors, edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0, edge_vmax=0.5)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        # plt.clim(np.min(colors), np.max(colors))
        # plt.colorbar(edges)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.5))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('depth [m]')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == 'mean_r2':
        ax.imshow(micr, cmap='gray', alpha=0)
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors,
                                       edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0.8, edge_vmax=1)
        # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        # plt.clim(np.min(colors), np.max(colors))
        # plt.colorbar(edges)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.8, vmax=1))
        # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_ticks([0.80, 0.85, 0.90, 0.95, 1.00])
        cbar.set_ticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'])
        cbar.set_label('r2')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == "centrality":
        # micr = np.fliplr(micr)
        ax.imshow(micr, cmap='gray', alpha=0)
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        # draw directionality
        # (and plot betweenness centrality of edges via color)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=matplotlib.colors.LogNorm(
                                       vmin=3.285928340474751e-07,
                                       vmax=0.0025840540469493443))  # this one's static
                                   # norm=matplotlib.colors.LogNorm(vmin=np.min(list(nx.edge_betweenness_centrality(graph).values())),
                                   #                    vmax=np.max(list(nx.edge_betweenness_centrality(graph).values()))))  # this one's dynamic
        nx.draw(graph, pos=tmp_coord_dict, arrowstyle='->', arrowsize=3.5, width=1, with_labels=False, node_size=0.005,
                edge_color=np.log(np.array(list(nx.edge_betweenness_centrality(graph).values()))), node_color='black',
                edge_cmap=cmap)
        print(np.min(np.array(list(nx.edge_betweenness_centrality(graph).values()))))
        print(np.max(np.array(list(nx.edge_betweenness_centrality(graph).values()))))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('betweenness centrality')
        plt.margins(0.0)
        plt.axis('off')
        plt.tight_layout()
    elif col_parameter == 'directionality':
        # draw directionality
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]

        # without bg
        ax.imshow(micr, cmap='terrain', alpha=0.7)
        nx.draw(graph, pos=tmp_coord_dict, arrowstyle='-', arrowsize=3.5, width=0.8, with_labels=False, node_size=0,
                node_color='seagreen', edge_color='black')
        # with bg
        # ax.imshow(micr, cmap='Greens', alpha=0)
        # nx.draw(graph, pos=tmp_coord_dict, arrowstyle='->', arrowsize=3.5, width=0.8, with_labels=False, node_size=0.65,
        #         node_color='white', edge_color='black')
        # plt.title("directionality")
        plt.margins(0.0)
        plt.tight_layout()
    elif col_parameter == 'water_filled':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        # draw edge by weight
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color="blue", edge_cmap=plt.cm.viridis,
                                       width=colors*2, ax=ax, edge_vmin=0, edge_vmax=1)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        # nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.05, node_color='blue')
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('fraction')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
    else:
        print("please choose one of the following col_parameters: 'mean_width', 'mean_depth', 'mean_r2', 'centrality', 'directionality'")


def scatter_depth_width(edge_param_dict, type):
    # print(edge_param_dict)
    width_mean = []
    width_med = []
    depth_mean = []
    depth_med = []
    r2_mean = []
    r2_med = []
    water = []
    for params in edge_param_dict.values():
        if not np.isnan(params[0]):
            width_mean.append(params[0])
            width_med.append(params[1])
            depth_mean.append(params[2])
            depth_med.append(params[3])
            r2_mean.append(params[4])
            r2_med.append(params[5])
            water.append(params[7])
            if params[7] != 0:
                print(params)

    if type == 'median':
        width = width_med
        depth = depth_med
        r2 = r2_med
    elif type == 'mean':
        width = width_mean
        depth = depth_mean
        r2 = r2_mean
    else:
        print("please provide either the keyword 'mean' or 'median'.")

    plt.figure()
    fig = plt.scatter(width, depth, s=3, c=water, cmap='viridis_r')
    plt.xlabel('Width [m]')
    plt.ylabel('Depth [m]')
    plt.title('Trough %s parameters by water presence' % type)
    plt.colorbar(fig)

    # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\11_trough_analysis\scatter_depth_width_r2_%s_orig' % type, dpi=600)


def density_depth_width(edge_param_dict, type):
    """
    makes a density plot.
    has nothing to do with graph density
    :param edge_param_dict:
    :param type:
    :return:
    """
    width_mean = []
    width_med = []
    depth_mean = []
    depth_med = []
    r2_mean = []
    r2_med = []
    for params in edge_param_dict.values():
        if not np.isnan(params[0]):
            # print(params)
            width_mean.append(params[0])
            width_med.append(params[1])
            depth_mean.append(params[2])
            depth_med.append(params[3])
            r2_mean.append(params[4])
            r2_med.append(params[5])

    if type == 'median':
        width = width_med
        depth = depth_med
        r2 = r2_med
    elif type == 'mean':
        width = width_mean
        depth = depth_mean
        r2 = r2_mean
    else:
        print("please provide either the keyword 'mean' or 'median'.")

    plt.figure()
    plt.hist2d(width, depth, (50, 50), cmap=plt.cm.viridis)
    plt.colorbar()
    plt.xlabel('Width [m]')
    plt.ylabel('Depth [m]')
    plt.title('Trough %s parameter density' % type)
    width_sorted = np.sort(width)
    depth_sorted = np.sort(depth)

    print(np.median(width_sorted[-960:]), np.median(depth_sorted[-960:]))
    # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\11_trough_analysis\density_depth_width_%s_detr' % type, dpi=600)


def plot_connected_components_size(graph):
    '''make histogram of number of nodes in each
    connected component of entire, disconnected graph.'''
    G_u = nx.to_undirected(graph)
    subgraph_sizes = []
    for sub in list(nx.connected_components(G_u)):
        print(type(sub))
        subgraph_sizes.append(len(sub))
    # subgraph_sizes = sorted(subgraph_sizes, reverse=True)
    # print(subgraph_sizes)
    count = Counter(subgraph_sizes)
    # count_sorted = sorted(count, reverse=True)
    print("subgraph_sizes: {0}".format(subgraph_sizes))
    print(count)

    # plt.figure()
    # plt.hist(subgraph_sizes, bins=np.max(subgraph_sizes))
    # plt.title("sub-component sizes")
    # plt.xlabel("number of nodes")
    # plt.ylabel("frequency")


def node_degree_hist(graph1, graph2):
    ''' plot histogram of node degrees
    with degree average.'''
    # first 2009
    degree1 = graph1.degree()
    degree_list1 = []
    for (n, d) in degree1:
        if d != 2:
            degree_list1.append(d)

    av_degree1 = sum(degree_list1) / len(degree_list1)

    print('The average degree for 2009 is {}'.format(np.round(av_degree1, 2)))

    # now 2019
    degree2 = graph2.degree()
    degree_list2 = []
    for (n, d) in degree2:
        if d != 2:
            degree_list2.append(d)

    av_degree2 = sum(degree_list2) / len(degree_list2)
    print(np.max(degree_list1), np.max(degree_list2))
    print('The average degree for 2019 is {}'.format(np.round(av_degree2, 2)))
    # plt.bar(np.array(x1) - 0.15, y1, width=0.3)
    # plt.bar(np.array(x2) + 0.15, y2, width=0.3)
    x_multi = [degree_list1, degree_list2]
    plt.figure(figsize=(4.5, 4.5))
    # plt.style.use('bmh')
    plt.grid(color='gray', linestyle='--', linewidth=0.2, which='both')
    plt.hist(x_multi, bins=range(1, np.max(degree_list2)+2), density=True, rwidth=0.5, color=['salmon', 'teal'],
             label=['2009', '2019'])
    plt.axvline(av_degree1, linestyle='--', color='salmon', linewidth=.9)
    plt.axvline(av_degree2, linestyle='--', color='teal', linewidth=.9)
    plt.xticks([1.5, 3.5, 4.5, 5.5, 6.5], [1, 3, 4, 5, 6])
    plt.text(av_degree1 - 0.175, 0.025, 'mean 2009 = {}'.format(np.round(av_degree1, 2)), rotation=90, fontsize=8)
    plt.text(av_degree2 - 0.175, 0.025, 'mean 2019 = {}'.format(np.round(av_degree2, 2)), rotation=90, fontsize=8)
    plt.text(5.3, 0.015, np.round(x_multi[0].count(5)/len(x_multi[0]), 4), rotation=90, fontsize=8)
    plt.text(5.55, 0.035, np.round(x_multi[1].count(5)/len(x_multi[1]), 4), rotation=90, fontsize=8)
    plt.text(6.3, 0.015, np.round(x_multi[0].count(6)/len(x_multi[0]), 4), rotation=90, fontsize=8)
    plt.text(6.55, 0.015, np.round(x_multi[1].count(6)/len(x_multi[1]), 4), rotation=90, fontsize=8)
    plt.legend(frameon=False)
    plt.ylabel('nodes frequency')
    plt.xlabel('degree')
    plt.savefig('./figures/node_degree_hist.png')

    print(f'{np.round(x_multi[0].count(3)/len(x_multi[0]), 4)} have 3 edges in 2009')
    print(f'{np.round(x_multi[1].count(3)/len(x_multi[1]), 4)} have 3 edges in 2019')
    print(f'{np.round(x_multi[0].count(1)/len(x_multi[0]), 4)} have 1 edge in 2009 -- total: {x_multi[0].count(1)}')
    print(f'{np.round(x_multi[1].count(1)/len(x_multi[1]), 4)} have 1 edge in 2019 -- total: {x_multi[1].count(1)}')


if __name__ == '__main__':
    startTime = datetime.now()

    # read in data
    img = read_data('./data/config_1a4_bedelevation.tif')
    G, coord_dict = read_graph(edgelist_loc='./data/config_1a4_graph.edgelist',
                               coord_dict_loc='./data/config_1a4_graph_node-coords.npy')

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(np.rot90(img, k=0), cmap='gray', alpha=1)
    nx.draw_networkx(G, pos=coord_dict, arrows=False, edge_color='red', width=1, node_size=2, node_color='black',
                     with_labels=False)
    # plt.savefig("./figures/config_1a4_graph_on_bedelevation.png", bbox_inches='tight')

    # transect_dict_fitted_2009 = load_obj('./data/a_2009/arf_transect_dict_avg_2009')
    #
    # add_params_graph(G_09, transect_dict_fitted_2009)

    # dem2009 = Image.open('./data/a_2009/arf_dtm_2009.tif')
    # img_det2009 = read_data('./data/a_2009/arf_microtopo_2009.tif')
    # dem2009 = np.array(dem2009)

    print(datetime.now() - startTime)
    plt.show()
