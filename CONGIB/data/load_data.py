import os.path as osp

import torch
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Amazon, Planetoid, Coauthor

from torch_geometric.data import Data
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_node_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets')
    if name in {'Cora', 'citeseer', 'pubmed'}:
        dataset = Planetoid(path, name)
    elif name in {'photo', 'computers'}:
        dataset = Amazon(path, name)
    elif name in {'cs', 'physics'}:
        dataset = Coauthor(path, name)
    else:
        dataset = Coauthor(path, name)
    data = dataset[0]

    data.num_classes = dataset.num_classes

    data = train_val_test_split(data, 5, train_ratio=0.6, val_ratio=0.2)

    return data


def load_real_datasets(datasets):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets')
    # path = "C:\\Users\\95194\\Desktop\\CONGIB\\Datasets\\GraphData"
    edge_path = path + "/{}/out1_graph_edges.txt".format(datasets)
    node_feature_path = path + "/{}/out1_node_feature_label.txt".format(datasets)
    with open(edge_path) as edge_file:
        edge_file_lines = edge_file.readlines()
        edge_list = [(num.split()) for num in edge_file_lines]
        edge_list = [[int(num) for num in sublist] for sublist in edge_list[1:]]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    with open(node_feature_path) as node_feature_file:
        node_lines = node_feature_file.readlines()[1:]
        feature_list = []
        labels = []
        max_len = 0
        for node_line in node_lines:
            node_id, feature, label = node_line.split("\t")
            labels.append(int(label))
            features = feature.split(",")
            max_len = max(len(features), max_len)
            feature_list.append([float(feature) for feature in features])
        feature_pad_list = []
        for features in feature_list:
            features += [0] * (max_len - len(features))
            feature_pad_list.append(features)
        feature_array = np.array(feature_pad_list)
        features = torch.from_numpy(feature_array)
        features = features.type(torch.float)

        labels = np.array(labels)
        labels = torch.FloatTensor(labels)
        labels = labels.long()

        dataset = Data(x=features, edge_index=edge_index, y=labels)

        dataset = train_val_test_split(dataset, 5, train_ratio=0.6, val_ratio=0.2)

        num_classes = int(max(labels) + 1)
        dataset.num_classes = num_classes

    return dataset


def load_dataset(name, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets')
    dataset = TUDataset(path, name, cleaned=cleaned, use_node_attr=True, use_edge_attr=True)
    # dataset.data.edge_attr = None

    if name == 'PROTEINS':
        dataset.data.x = torch.index_select(dataset.data.x, 1, torch.tensor([1, 2, 3]))

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


def train_val_test_split(data, run, train_ratio=0.1, val_ratio=0.1):
    """
    Train-Test split for node classification data (Aligning with baseline split)
    """
    num_nodes = data.y.shape[0]

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    for i in range(run):

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        torch.manual_seed(i)
        shuffle_idx = torch.randperm(num_nodes)
        train_idx = shuffle_idx[:num_train]
        val_idx = shuffle_idx[num_train:(num_train + num_val)]
        test_idx = shuffle_idx[(num_train + num_val):]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_mask = train_mask.reshape(1, -1)
        val_mask = val_mask.reshape(1, -1)
        test_mask = test_mask.reshape(1, -1)

        if i == 0:
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        else:
            data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
            data.val_mask = torch.cat((data.val_mask, val_mask), dim=0)
            data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
