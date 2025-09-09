import torch
import copy
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_remaining_self_loops
from torch_sparse import coalesce


def cosine_similarity(x):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = x.div(x_norm + 5e-10)
    cos_adj = torch.mm(x_norm, x_norm.transpose(0, 1))
    return cos_adj


def build_knn_graph(x, k):
    """generating knn graph from data x
    Args:
        x: input data (n, m)
        k: number of nearst neighbors
    returns:
        knn_edge_index, knn_edge_weight
    """
    cos_adj = cosine_similarity(x)
    topk = min(k + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_dilated_knn_graph(x, k1, k2):
    """generating dilated knn graph from data x
    Args:
        x: input data (n, m)
        k1: number of nearst neighbors
        k2: number of dilations
    returns:
        knn_edge_index, knn_edge_weight
    """
    cos_adj = cosine_similarity(x)
    topk = min(k1 + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    knn_val = knn_val[:, k2:]  #
    knn_ind = knn_ind[:, k2:]  #
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_ppr_graph(edge_index, alpha=0.1):
    """generating PageRank graph from adj
    Args:
        edge_index: input adj
        alpha: hyper parameter
    returns:
        ppr_edge_index, ppr_edge_weight
    """
    edge_weight = torch.ones(edge_index.size(-1), dtype=torch.long)
    edge_weight = (alpha - 1) * edge_weight
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)
    adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze()
    ppr_adj = alpha * torch.inverse(adj)
    ppr_edge_index, ppr_edge_weight = dense_to_sparse(ppr_adj)
    return ppr_edge_index, ppr_edge_weight


def view_generator(dataset, view='adj'):
    temp_dataset = []
    assert view in ['PPR', 'KNN', 'DKNN', 'adj']

    if view == 'PPR':
        for data in dataset:
            data.edge_index1, data.edge_weight1 = build_ppr_graph(
                data.edge_index)
            temp_dataset.append(data)
    elif view == 'KNN':
        for data in dataset:
            data.edge_index1, _ = build_knn_graph(data.x, k=5)
            temp_dataset.append(data)
    elif view == 'DKNN':
        for data in dataset:
            data.edge_index1, _ = build_dilated_knn_graph(data.x, k1=10, k2=5)
            temp_dataset.append(data)
    elif view == 'adj':
        for data in dataset:
            data.edge_index1 = data.edge_index
            temp_dataset.append(data)

    return temp_dataset


def node_view_generator(data, view='adj'):
    assert view in ['PPR', 'KNN', 'DKNN', 'adj']

    dataset = copy.deepcopy(data)

    if view == 'PPR':
        dataset.edge_index1, dataset.edge_weight1 = build_ppr_graph(
            data.edge_index)
    elif view == 'KNN':
        dataset.edge_index1, _ = build_knn_graph(data.x, k=5)
    elif view == 'DKNN':
        dataset.edge_index1, _ = build_dilated_knn_graph(data.x, k1=10, k2=5)
    elif view == 'adj':
        dataset.edge_index1 = data.edge_index

    return dataset


def graph_resampling(data, sampling_ratio):

    graph = to_networkx(data)

    # 随机游走获取新的节点集合

    sampled_nodes, num_nodes = random_walk_sampling(graph, sampling_ratio)

    # 构建新的子图
    subgraph = graph.subgraph(sampled_nodes)

    # 将重采样后的NetworkX图形对象转换回Data对象
    subgraph = from_networkx(subgraph)
    edge_is_dummy = data.edge_is_dummy
    edge_is_dummy = torch.tensor(edge_is_dummy)
    subgraph.edge_is_dummy = edge_is_dummy[subgraph.edge_index[0]]
    subgraph.edge_attr = data.edge_attr[subgraph.edge_index[0], :]

    subgraph.x = data.x[sampled_nodes]
    # if data.edge_attr is not None:
    #     row, col = data.edge_index
    #     mask = (row < num_nodes) & (col < num_nodes)
    #     resampled_data.edge_index = torch.stack([row[mask], col[mask]], dim=0)
    #     resampled_data.edge_attr = data.edge_attr[mask]
    #     if data.edge_is_dummy is not None:
    #         edge_is_dummy = data.edge_is_dummy
    #         edge_is_dummy = torch.tensor(edge_is_dummy)
    #         resampled_data.edge_is_dummy = edge_is_dummy[mask]

    return subgraph


def random_walk_sampling(graph, sampling_ratio):
    num_total_nodes = graph.number_of_nodes()

    num_nodes = int(num_total_nodes * sampling_ratio)

    # 生成随机索引
    perm = torch.randperm(num_total_nodes)[:num_nodes]

    return perm.tolist(), num_nodes


def sample_subgraph(data, sampling_ratio):
    num_total_nodes = data.num_nodes
    num_nodes = int(num_total_nodes * sampling_ratio)
    node_idx = torch.randperm(data.num_nodes)[:num_nodes].to("cuda:0")  # 随机采样节点
    subgraph = data.subgraph(node_idx)  # 使用subgraph方法生成子图

    edge_is_dummy = data.edge_is_dummy
    edge_is_dummy = torch.tensor(edge_is_dummy).to("cuda:0")
    subgraph.edge_is_dummy = edge_is_dummy[subgraph.edge_index[0]]
    subgraph.edge_attr = data.edge_attr[subgraph.edge_index[0], :]

    # subgraph.x = data.x[num_nodes]
    return subgraph
