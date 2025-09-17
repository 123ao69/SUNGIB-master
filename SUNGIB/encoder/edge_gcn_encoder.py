import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv



class EdgeGCNEncoder(torch.nn.Module):
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, hidden_dim, AttnNodeFlag):
        super(EdgeGCNEncoder, self).__init__()

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.Linear = nn.Linear(num_edge_in_embeddings, hidden_dim)

        self.AttnNodeFlag = AttnNodeFlag  # boolean (for ablation studies)

        self.node_attentionND = nn.Linear(num_node_in_embeddings,
                                          num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings // 2 * 2,
                                                  num_edge_in_embeddings // 2) if self.AttnNodeFlag else None
        # self.reset_parameters( )

    @staticmethod
    def concate_NodeIndicator_for_edges(node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)

        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]
        object_idx_list = edge_index_list[:, 1]

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]  # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    # def reset_parameters(self):
    #     self.Linear.reset_parameters()

    def forward(self, node_data, edge_feats):
        # prepare node_feats & edge_feats in the following formats
        # node_feats: (1, num_nodes,  num_embeddings)
        # edge_feats: (1, num_edges,  num_embeddings)
        # (num_embeddings = num_node_in_embeddings = num_edge_in_embeddings) = 2 * num_mid_channels

        node_feats, edge_index = node_data.x, node_data.edge_index

        #### Deriving Node Attention
        if self.AttnNodeFlag and node_feats is not None:
            node_indicator = F.relu(
                self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)  # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        #### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.unsqueeze(0)
        edge_feats = edge_feats.permute(0, 2, 1)  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)  # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, p=0.2, training=self.training) * agg_node_indicator  # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)

        edge_feats = edge_feats.squeeze(0)

        edge_feats = F.relu(self.Linear(edge_feats))

        return edge_feats
