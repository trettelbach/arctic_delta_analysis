import cv2
import numpy as np
from PIL import Image
import sys
import scipy
import scipy.ndimage
from skimage.morphology import skeletonize, medial_axis, skeletonize_3d
from scipy.ndimage.morphology import generate_binary_structure
import sknw
import matplotlib.pyplot as plt
import networkx as nx
import glob
from scipy import ndimage
from datetime import datetime

startTime = datetime.now()

np.set_printoptions(threshold=sys.maxsize)

def read_data(img):
    ''' helper function to make reading in DEMs easier '''
    img1 = Image.open(img)
    img1 = np.array(img1)
    img1[img1 == 255] = 1
    # print(img1)
    return img1


def scale_data(img):
    ''' scale the image to be between 0 and 255 '''
    img_orig = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_orig


def int_conversion(img):
    '''convert to 8bit int'''
    img = img.astype(int)
    img = img.astype('uint8')
    return img


def make_directed(graph, dem):
    """ convert graph from nx.Graph()
    to nx.DiGraph() - for each edge (u, v)
    an edge (v, u) is generated.
    Then remove all self-looping edges
    (don't exist in nature) and upward
    sloped edges (water only runs downhill)

    :param graph: graph generated from
    skeleton of class nx.Graph
    :param dem: original digital elevation
    model (same extent as detrended image,
    as we're working with pixel indices,
    not spatial coordinates
    :return G_d: a directed graph of
    class nx.DiGraph() with only even or
    downward slope directed edges.
    """
    # we need a helper graph, because we cant remove edges from
    # the graph while iterating through it
    G_help = graph.to_directed()
    G_d = graph.to_directed()
    same_elev = 0
    diff_elev = 0
    for (s, e) in G_help.edges():
        # remove all self-looping edges. they don't make any real world sense...
        if s == e:
            G_d.remove_edge(s, e)

        # now remove all (s, e) edges, where s downslope of e
        # (so remove directed edges that would flow upwards...)
        elev_start = dem[int(G_help.nodes()[s]['o'][0]), int(G_help.nodes()[s]['o'][1])]
        elev_end = dem[int(G_help.nodes()[e]['o'][0]), int(G_help.nodes()[e]['o'][1])]
        if elev_end == elev_start:
            same_elev += 1
        else:
            diff_elev += 1
        if elev_start < elev_end:
            G_d.remove_edge(s, e)
    print(same_elev, diff_elev)
    return G_d


def remove_selfloops(graph):
    """ remove all self-looping edges
    (don't exist in nature)

    :param graph: graph generated from
    skeleton of class nx.Graph
    :return G_d: a directed graph of
    class nx.DiGraph() without self-
    looping edges.
    """
    # we need a helper graph, because we cant remove edges from
    # the graph while iterating through it
    G_help = graph.copy()
    for (s, e) in graph.edges():
        # remove all self-looping edges. they don't make any real world sense...
        if s == e:
            G_help.remove_edge(s, e)
    return G_help


def get_node_coord_dict(graph):
    ''' create dictionary with node ID as key
    and node coordinates as values

    :param graph: nx.DiGraph
    :return dictionary:
    dictionary with Node IDs as keys,
    and pixel coordinates of the nodes as values.
    '''
    nodes = graph.nodes()
    # print(nodes)
    # get pixel coordinates of nodes --> ps
    ps = np.array([nodes[i]['o'] for i in nodes])
    # for i in nodes:
    #     print(i)
    #     print(nodes[i])
    #     print('___')
    # get node ID --> keys
    keys = list(range(len(nodes)))
    keys_str = []
    for i in keys:
        keys_str.append(str(i))
    keys = keys_str
    values = ps.tolist()
    dictionary = dict(zip(keys, values))
    return dictionary


def save_graph_with_coords(graph, dict, location):
    ''' save graph as edgelist to disk
    and coords for nodes as dictionary

    :param graph: nx.DiGraph representing the
    trough network
    :param dict:
    :return NA: function just for saving
    '''
    # save and write Graph as list of edges
    # edge weight 'weight' stores the actual length of the trough in meter
    nx.write_edgelist(graph, location + '.edgelist', data=True)

    # and save coordinates of graph as npy dict to disk
    fname = location + '_node-coords'
    np.save(fname, dict)


def get_depths(H, bed_el):
    print(bed_el.shape)
    all_depths = []
    for (s, e) in H.edges():
        all_i_depths_edge = []
        for i in H[s][e]['pts']:
            depth_i = bed_el[i[0]][i[1]]
            all_i_depths_edge.append(depth_i)
            all_depths.append(depth_i)
        H[s][e]['depth_mean'] = np.mean(all_i_depths_edge)
    return H, all_depths


def do_analysis(filename):
    img = read_data(filename)
    fn_base = filename[7:-25]

    fn_base_name = filename[:-25] + '.tif'
    bed_el = read_data(fn_base_name)
    bed_el = np.fliplr(np.rot90(bed_el, 3))
    print(img.shape)

    # prepare for both possible skeletonization algorithms
    zhang = skeletonize(img)
    # lee = skeletonize_3d(img)

    img_skel = zhang

    # make a transparent raster with only trough pixels in red.
    skel_transp = np.zeros((img_skel.shape[0], img_skel.shape[1], 4))
    for i in range(img_skel.shape[0]):
        for j in range(img_skel.shape[1]):
            if img_skel[i, j] == 1:
                skel_transp[i, j, 0] = 255
                skel_transp[i, j, 1] = 0
                skel_transp[i, j, 2] = 0
                skel_transp[i, j, 3] = 150

    # build graph from skeletonized image
    G = sknw.build_sknw(img_skel, multi=False)

    # # need to avoid np.arrays - so we convert it to a list
    for (s, e) in G.edges():
        G[s][e]['pts'] = G[s][e]['pts'].tolist()
        # print(s, e)
        # print(type(G[s][e]['pts']))
        # y = G[s][e]['pts'][0][1]-1
        # new_s_coord = [G[s][e]['pts'][0][0], y]
        # # print(new_s_coord)
        # G[s][e]['pts'] = np.insert(G[s][e]['pts'], 0, new_s_coord)
        # print(type(G[s][e]['pts']))
        # print('___')

    # and make it a directed graph, since water only flows downslope
    # flow direction is based on distance from root channel
    # dem = Image.open('./data/config_3a_bedelevation.tif')
    # dem = np.rot90(np.array(dem))
    # H = make_directed(G, dem)
    H = remove_selfloops(G)

    # save graph and node coordinates
    dictio = get_node_coord_dict(H)
    # print(dictio)

    H, depths = get_depths(H, bed_el)

    save_graph_with_coords(H, dictio, f'./graphs/{fn_base}_graph')

    plt.figure()  # figsize=(2.5, 2), dpi=600
    plt.imshow(img, cmap='Greens_r', alpha=100)
    plt.imshow(skel_transp)
    plt.axis('off')
    plt.savefig(f"./outputs/{fn_base_name[19:-4]}_skel_transp_on_img_oceanmasked.png", bbox_inches='tight')
    # return H, dictio


if __name__ == '__main__':
    # iterate over delta images
    for filename in glob.iglob(f'./data/config_full/*_channels_oceanmasked.png'):
        print(filename)
        do_analysis(filename)

    # filename = './data/config_full/config_full_1_bedelevation_channels_oceanmasked.png'
    # do_analysis(filename)

    # print time needed for script execution
    print(datetime.now() - startTime)  # ca. 0.6 sec per image
    plt.show()
