import pickle
import scipy
import scipy.ndimage
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from b_delta_network_analysis import read_graph
from a_delta_to_graph import read_data
from datetime import datetime
import glob
import matplotlib
# import seaborn as sns
# import pandas as pd
from collections import Counter


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    startTime = datetime.now()

    for filename in glob.iglob(f'./graphs/config_full/*.edgelist'):
        fn_base_name = filename[21:-15]
        print(fn_base_name)

        # read in data
        img = read_data(F'./data/config_full/{fn_base_name}.tif')
        G, coord_dict = read_graph(edgelist_loc=f'./graphs/config_full/{fn_base_name}_graph.edgelist',
                                   coord_dict_loc=f'./graphs/config_full/{fn_base_name}_graph_node-coords.npy')

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(img, k=0), cmap='gray', alpha=1)
        nx.draw_networkx(G, pos=coord_dict, arrows=False, edge_color='red', width=1, node_size=2, node_color='black',
                         with_labels=False)
        plt.savefig(f"./outputs/graph_on_bedelevation/{fn_base_name}_graph_on_bedelevation.png", bbox_inches='tight')


    print(datetime.now() - startTime)
    # plt.show()
