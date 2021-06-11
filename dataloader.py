from torch_geometric.datasets import Planetoid, Reddit, WebKB, Actor
from torch_geometric.transforms import NormalizeFeatures

from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler, Data
import torch_geometric
import os

import networkx as nx
import numpy as np
import torch as th
from sklearn.model_selection import ShuffleSplit
import utils



def load_data_from_pygeo(config):
    """
    load data method,
    if split is planetoid, load data with fixed train and test index like GMNN
    if split  is random , load data using random split.
    """
    dataset_name = config['dataset_name']
    root = 'tmp/'+dataset_name
    transform = None

    if config['feature_row_normalization']:
        transform = NormalizeFeatures()
        
    if dataset_name == 'Reddit':
        dataset = Reddit(root, transform=transform)

        return  dataset


    else:
        if config['split'] == "planetoid": return Planetoid(root, dataset_name, split = 'public', transform = transform)
        elif config['split'] == "webkb": return WebKB(root, dataset_name)
        elif config['split'] == "actor": return Actor(root)
        elif config['split'] == "random": return Planetoid(root,
                                                     dataset_name,
                                                     split = 'random',
                                                     num_train_per_class=config['num_train_per_class'],
                                                     num_val=config['num_val'],
                                                     num_test=config['num_test'],
                                                     transform=transform)

def load_data_from_directory():return 0

def load_dataset(configs):
    if configs['datasource'] == 'pygeo': return load_data_from_pygeo(configs)
    if configs['datasource'] == 'directory': return load_data_from_directory()


def load_GMNN_dataset():return 0
def load_GCN_dataset(): return 0


 # MIT License
#  #
#  # Copyright (c) 2019 Geom-GCN Authors
#  #
#  # Permission is hereby granted, free of charge, to any person obtaining a copy
#  # of this software and associated documentation files (the "Software"), to deal
#  # in the Software without restriction, including without limitation the rights
#  # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  # copies of the Software, and to permit persons to whom the Software is
#  # furnished to do so, subject to the following conditions:
#  #
#  # The above copyright notice and this permission notice shall be included in all
#  # copies or substantial portions of the Software.
#  #
#  # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  # SOFTWARE.




def load_data_GEOM(dataset_name, splits_file_path=None, train_percentage=None, val_percentage=None, embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = utils.load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:

        path = "../geom-gcn/new_data"
        graph_adjacency_list_file_path = os.path.join(path, dataset_name,'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(path, dataset_name,f'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = utils.preprocess_features(features)


    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

        if dataset_name in {'cora', 'citeseer'}:

            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1

        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    """convert to pygeo Data"""
    edge_index, edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(adj)

    data = Data(edge_index=edge_index, test_mask=test_mask, train_mask=train_mask,
                  val_mask=val_mask, x=features, y=labels)
    dataset = Dataset(data, features.shape[1], len(th.unique(labels)))
    return dataset

class Dataset(object):
    def __init__(self, data, num_node_features, num_classes):
        self.data = data
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.num_features = num_node_features
