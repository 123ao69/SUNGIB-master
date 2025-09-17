import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
import math


class GCNEncoder(torch.nn.Module):
    def __init__(self, args):
        super(GCNEncoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(args.num_features, args.hidden, cached=False, add_self_loops=True))
        for _ in range(args.num_layers - 2):
            self.convs.append(
                GCNConv(args.hidden, args.hidden, cached=False, add_self_loops=True))
        self.convs.append(GCNConv(args.hidden, args.hidden, cached=False, add_self_loops=True))

        self.dropout = args.dropout_ratio

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, edge_index)
        xx.append(F.relu(x))
        return x

    def outEmb(self, data):
        x = data.x
        edge_index = data.edge_index
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, edge_index)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

# class GCNEncoder(nn.Module):
#     def __init__(self, args, encoding=True):
#         super(GCNEncoder, self).__init__()
#         self.out_dim = args.hidden
#         self.num_layers = args.num_layers
#         self.gcn_layers = nn.ModuleList()
#         self.dropout = args.dropout_ratio
#
#         last_activation = nn.ReLU()
#         residual = args.residual
#         last_residual = encoding and residual
#         in_dim = args.num_features
#         num_hidden = args.hidden
#         num_layers = args.num_layers
#         out_dim = args.hidden
#
#         if num_layers == 1:
#             self.gcn_layers.append(GraphConv(in_dim, out_dim, residual=last_residual, activation=last_activation))
#         else:
#             # input projection (no residual)
#             self.gcn_layers.append(GraphConv(in_dim, num_hidden, residual=last_residual, activation=last_activation))
#             # hidden layers
#             for l in range(1, num_layers - 1):
#                 self.gcn_layers.append(GraphConv(num_hidden, num_hidden, residual=last_residual, activation=last_activation))
#             # output projection
#             self.gcn_layers.append(GraphConv(num_hidden, out_dim, residual=last_residual, activation=last_activation))
#
#         self.norms = None
#         self.head = nn.Identity()
#
#     def forward(self, data):
#
#         x, edge_index = data.x, data.edge_index
#         for l in range(self.num_layers):
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = self.gcn_layers[l](x, edge_index)
#         return self.head(x)
#
#     def reset_classifier(self, num_classes):
#         self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(MessagePassing):
    def __init__(self, in_dim, out_dim, activation=None, residual=True):
        super(GraphConv, self).__init__()

        self._in_feats = in_dim
        self._out_feats = out_dim
        self.fc = nn.Linear(in_dim, out_dim)
        self.residual = residual
        self.activation = activation

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x, edge_index):
        h = self.propagate(edge_index, x=x)
        h = self.fc(h)
        if self.residual:
            if self.res_fc is not None:
                res = self.res_fc(x)
            else:
                res = x
            h += res
        if self.activation is not None:
            h = self.activation(h)
        return h

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GCN_concat_readout(torch.nn.Module):
    def __init__(self, args):
        super(GCN_concat_readout, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        # dummy edge weight
        if args.dummy_weight > 0:
            self.dummy_weight = torch.tensor(args.dummy_weight, requires_grad=True, device=self.args.device)
            self.use_edge_weight = True
        else:
            self.use_edge_weight = False

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.lin1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.lin3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.use_edge_weight:
            # when trainable dummy edge weight is enabled
            edge_attr = torch.ones(data.is_dummy_edge.size()).to(self.args.device)
            edge_attr[data.is_dummy_edge.to(self.args.device)] = self.dummy_weight

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(VGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, hidden_dim)
        self.conv_logvar = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.conv1(h, edge_index))
        z_mu = self.conv_mu(h, edge_index)
        z_logvar = self.conv_logvar(h, edge_index)
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = x.to("cuda:0")
        z_mu, z_logvar = self.encode(x, edge_index)
        return z_mu, z_logvar
