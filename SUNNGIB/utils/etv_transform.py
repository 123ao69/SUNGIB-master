import igraph as ig
import torch

from torch_geometric.data import Data
from torch_geometric.utils import degree


def igraph_to_data(graph, y):
    edge_attr = torch.stack(graph.es["ATTR"])
    if "ATTR" in graph.vertex_attributes():
        x = graph.vertex_attributes()
    else:
        x = None

    edge = graph.get_edgelist()
    edge_index = torch.tensor(edge).permute(1, 0)
    num_nodes = graph.vcount()
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, edge_attr=edge_attr, y=y)
    data.edge_is_dummy = graph.es["ID"]

    if x is None:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        one_hot_deg = torch.zeros(data.num_nodes, int(deg.max()) + 1)
        one_hot_deg[torch.arange(data.num_nodes), deg] = 1
        data.x = one_hot_deg

    return data


def connective_node_generation(data):
    graph = ig.Graph(directed=True)
    n = data.num_nodes
    graph.add_vertices(n)
    src, dst = data.edge_index1
    edges = zip(src, dst)
    graph.add_edges(edges)

    in_degrees = graph.degree(mode="in")
    out_degrees = graph.degree(mode="out")

    if 0 in in_degrees or 0 in out_degrees:
        x = data.x.tolist()
        for i, value in enumerate(x):
            graph.vs[i]["ATTR"] = torch.tensor(value)
        if data.edge_attr is not None:
            edge_attr = data.edge_attr.tolist()
            for i, value in enumerate(edge_attr):
                graph.es[i]["ATTR"] = torch.tensor(value)
        graph.add_vertices(1)
        edges = list()

        graph.vs["IS_DUMMY"] = [0] * n + [1]

        for node in graph.vs:
            if node.index == graph.vcount() - 1:
                break
            if in_degrees[node.index] == 0:
                edges.append((n, node.index))
            if out_degrees[node.index] == 0:
                edges.append((node.index, n))

        e = graph.ecount()
        graph.add_edges(edges)
        Edge_Is_Dummy = [0] * e + len(edges) * [1]

        graph.es["IS_DUMMY"] = Edge_Is_Dummy
    else:
        graph.vs["IS_DUMMY"] = [0] * n
        x = data.x.tolist()
        for i, value in enumerate(x):
            graph.vs[i]["ATTR"] = torch.tensor(value)
        if data.edge_attr is not None:
            edge_attr = data.edge_attr.tolist()
            for i, value in enumerate(edge_attr):
                graph.es[i]["ATTR"] = torch.tensor(value)
        graph.es["IS_DUMMY"] = [0] * graph.ecount()

    graph.vs["ID"] = list(range(graph.vcount()))
    graph.es["ID"] = list(range(graph.ecount()))

    graph = edge_to_vertex_transform(graph)

    graph = igraph_to_data(graph, data.y)

    return graph


def connective_graph_generation(Databatch):
    graphs = []

    for data in Databatch:
        graph = ig.Graph(directed=True)
        n = data.num_nodes
        graph.add_vertices(n)
        src, dst = data.edge_index1
        edges = zip(src, dst)
        graph.add_edges(edges)

        in_degrees = graph.degree(mode="in")
        out_degrees = graph.degree(mode="out")

        if 0 in in_degrees or 0 in out_degrees:
            x = data.x.tolist()
            for i, value in enumerate(x):
                graph.vs[i]["ATTR"] = torch.tensor(value)
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.tolist()
                for i, value in enumerate(edge_attr):
                    graph.es[i]["ATTR"] = torch.tensor(value)
            graph.add_vertices(1)
            edges = list()

            graph.vs["IS_DUMMY"] = [0] * n + [1]

            for node in graph.vs:
                if node.index == graph.vcount() - 1:
                    break
                if in_degrees[node.index] == 0:
                    edges.append((n, node.index))
                if out_degrees[node.index] == 0:
                    edges.append((node.index, n))

            e = graph.ecount()
            graph.add_edges(edges)
            Edge_Is_Dummy = [0] * e + len(edges) * [1]

            graph.es["IS_DUMMY"] = Edge_Is_Dummy
        else:
            graph.vs["IS_DUMMY"] = [0] * n
            x = data.x.tolist()
            for i, value in enumerate(x):
                graph.vs[i]["ATTR"] = torch.tensor(value)
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.tolist()
                for i, value in enumerate(edge_attr):
                    graph.es[i]["ATTR"] = torch.tensor(value)
            graph.es["IS_DUMMY"] = [0] * graph.ecount()

        graph.vs["ID"] = list(range(graph.vcount()))
        graph.es["ID"] = list(range(graph.ecount()))

        graph = edge_to_vertex_transform(graph)

        graph = igraph_to_data(graph, data.y)
        graphs.append(graph)

    return graphs


def edge_to_vertex_transform(graph):
    conj_graph = ig.Graph(directed=True)

    if "ID" in graph.edge_attributes() and graph.ecount() > 0:
        eids = graph.es["ID"]
        num_edges = max(eids) + 1
        id2vertex = [None] * num_edges
        for e, eid in enumerate(eids):
            if id2vertex[eid] is None:
                id2vertex[eid] = e
            else:
                id2vertex[eid] = min(id2vertex[eid], e)
        conj_graph.add_vertices(num_edges)
        for k in graph.edge_attributes():
            v = graph.es[k]
            conj_graph.vs[k] = [
                v[id2vertex[e]] if id2vertex[e] is not None else v[0].__class__() for e in range(num_edges)
            ]
    else:
        num_edges = graph.ecount()
        id2vertex = list(range(num_edges))
        conj_graph.add_vertices(num_edges)
        for k in graph.edge_attributes():
            v = graph.es[k]
            conj_graph.vs[k] = v
        if "ID" not in graph.edge_attributes():
            conj_graph.vs["ID"] = list(range(graph.ecount()))

    edges = list()
    edge_indices = list()
    prev = -1

    if "ID" in graph.edge_attributes() and graph.ecount() > 0 and "LABEL" in graph.vertex_attributes():
        used_keys = set()
        for e in range(graph.ecount()):
            source = graph.es[e].source
            vid = eids[e]
            elabel = graph.vs[source]["LABEL"]
            if prev != source:
                incident_edges = sorted(graph.incident(source, "in"))  # 对入射边排序
            for incident_e in incident_edges:
                uid = eids[incident_e]
                key = (uid, elabel, vid)
                if key not in used_keys:
                    used_keys.add(key)
                    edges.append((uid, vid))
                    edge_indices.append(source)
            prev = source
    else:
        for e in range(graph.ecount()):
            source = graph.es[e].source
            if prev != source:
                incident_edges = sorted(graph.incident(source, "in"))
            for incident_e in incident_edges:
                edges.append((incident_e, e))
                edge_indices.append(source)
            prev = source

    if "IS_DUMMY" in graph.edge_attributes():
        dummy_eids = list()
        new_edges = list()
        new_edge_indices = list()
        if "ID" in graph.edge_attributes():
            for e, flag in zip(eids, graph.es["IS_DUMMY"]):
                if flag:
                    dummy_eids.append(e)
        else:
            for e, flag in enumerate(graph.es["IS_DUMMY"]):
                if flag:
                    dummy_eids.append(e)
        if len(dummy_eids) > 0:
            for e in dummy_eids[1:]:
                id2vertex[e] = None
            prev = dummy_eids[0]
            dummy_eids = set(dummy_eids)
            used_keys = set([(prev, prev)])
            for e in range(len(edges)):
                uid, vid = edges[e]
                if uid in dummy_eids:
                    uid = prev
                if vid in dummy_eids:
                    vid = prev
                key = (uid, vid)
                if key not in used_keys:
                    used_keys.add(key)
                    new_edges.append(key)
                    new_edge_indices.append(edge_indices[e])
            edges, edge_indices = new_edges, new_edge_indices

    if len(edges) > 0:
        conj_graph.add_edges(edges)
        for k in graph.vertex_attributes():
            v = graph.vs[k]
            conj_graph.es[k] = [v[i] for i in edge_indices]
        if "ID" not in graph.vertex_attributes():
            conj_graph.es["ID"] = edge_indices
    else:
        for k in graph.vertex_attributes():
            conj_graph.es[k] = list()
        if "ID" not in graph.vertex_attributes():
            conj_graph.es["ID"] = list()

    for v in id2vertex:
        if v is None:
            conj_graph.delete_vertices([v for v in range(len(id2vertex)) if id2vertex[v] is None])
            break

    return conj_graph

