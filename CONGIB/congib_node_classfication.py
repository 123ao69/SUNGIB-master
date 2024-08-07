import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch_geometric import seed_everything

from torch_geometric.utils import (add_self_loops, negative_sampling, remove_self_loops)

from data import load_node_dataset, load_real_datasets
from encoder import FNN, GCNEncoder, EdgeGCNEncoder, VGCNEncoder
from utils import node_view_generator, eval_node, sample_subgraph, connective_node_generation


class LinearPred(nn.Module):

    def __init__(self, emb_dim, nb_classes):
        super(LinearPred, self).__init__()

        self.fc = nn.Linear(emb_dim, nb_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class CLUB(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, h):
        mu, logvar = self.encoder(x, edge_index)

        positive = -(mu - h) ** 2 / 2. / (logvar.exp() + 1e-7)

        prediction_1 = mu.unsqueeze(0)
        h_samples_1 = h.unsqueeze(0)

        negative = -((h_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / (
                logvar.exp() + 1e-7)

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x, edge_index, h):
        mu, logvar = self.encoder(x, edge_index)
        return (-(mu - h) ** 2 / logvar.exp() - logvar +
                1e-7).sum(dim=1).mean(dim=0)

    def learning_loss(self, x, edge_index, h):
        return -self.loglikeli(x, edge_index, h)


class MVGIB(torch.nn.Module):
    def __init__(self, encoder_c1, encoder_h1, encoder_c2, encoder_h2,
                 encoder_f, club):
        super().__init__()

        self.encoder_c1 = encoder_c1
        self.encoder_h1 = encoder_h1
        self.encoder_c2 = encoder_c2
        self.encoder_h2 = encoder_h2

        self.encoder_f = encoder_f
        self.club = club

    @staticmethod
    def decoder(z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    @staticmethod
    def edge_decoder(z, args):
        deconv1 = nn.ConvTranspose1d(args.hidden, args.num_features, 1).to(args.device)
        deconv2 = nn.ConvTranspose1d(args.num_features, args.num_features, 1).to(args.device)
        h = z.unsqueeze(0)
        h = h.permute(0, 2, 1)
        value = F.relu(deconv1(h))
        value = F.dropout(value, p=args.dropout_ratio)
        value = deconv2(value).permute(0, 2, 1)
        value = F.relu(value)
        value = value.squeeze(0)
        return torch.sigmoid(value)

    @staticmethod
    def Delete_duplicate_ID(conj_g, node_feat, edge_feat):
        ids = conj_g.edge_is_dummy
        ids = ids.clone().detach()
        unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
        new_edge_feat = torch.zeros(len(node_feat), edge_feat.shape[1], device="cuda:0")

        counts = torch.bincount(inverse_indices).to("cuda:0")
        sums = torch.zeros(len(unique_ids), edge_feat[:len(ids)].shape[1], dtype=edge_feat.dtype).to(
            "cuda:0")
        index = inverse_indices.unsqueeze(1).repeat(1, edge_feat[:len(ids)].shape[1]).to("cuda:0")
        sums.scatter_add_(dim=0, index=index, src=edge_feat[:len(ids)])

        mean = sums / counts.unsqueeze(-1)
        new_edge_feat[:len(unique_ids)] = mean

        return new_edge_feat

    def forward(self, data, conj_load):
        edge_x = conj_load.edge_attr

        data.to("cuda:0")
        conj_load.to("cuda:0")

        c1 = self.encoder_c1(data)
        c2 = self.encoder_c2(conj_load, edge_x)

        h1 = self.encoder_h1(data)
        h2 = self.encoder_h2(conj_load, edge_x)

        c2 = self.Delete_duplicate_ID(conj_load, c1, c2)
        h2 = self.Delete_duplicate_ID(conj_load, h1, h2)

        edge_feature = self.Delete_duplicate_ID(conj_load, data.x, edge_x)

        if self.training:
            return c1, c2, h1, h2, edge_feature
        else:
            return torch.cat([c1, h1], dim=-1)

    @staticmethod
    def sce_loss(x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        y = y.to("cuda:0")
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def recon_f_loss(self, x, c1, h1):
        z = torch.cat([c1, h1], dim=-1)
        re_f = self.encoder_f(z)
        f_loss = self.sce_loss(re_f, x, alpha=3)
        return f_loss

    def recon_edge_loss(self, z, edge_attr, args):
        h = self.edge_decoder(z, args)
        recon_loss = F.mse_loss(h, edge_attr)
        return recon_loss

    def recon_loss(self, z, edge_index):

        pos_loss = -torch.log(self.decoder(z, edge_index) + 1e-7).mean()

        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(edge_index)

        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) + 1e-7).mean()

        return pos_loss + neg_loss

    def common_loss(self, c1, c2, edge_index1, edge_feature, args):
        loss_t = self.recon_loss(c1, edge_index1) + self.recon_edge_loss(c2, edge_feature, args)
        return loss_t

    def specific_loss(self, x, h1, h2, edge_index1, edge_feature, args):
        loss_t = self.recon_loss(h1, edge_index1) + self.recon_edge_loss(h2, edge_feature, args)

        loss_t_p = self.club(x, edge_index1, h2) + self.club.learning_loss(x, edge_index1, h2)

        return loss_t + loss_t_p

    def loss(self, x, c1, h1, c2, h2, edge_feature, edge_index1, args):
        c_loss = self.common_loss(c1, c2, edge_index1, edge_feature, args)
        v_loss = self.specific_loss(x, h1, h2, edge_index1, edge_feature, args)
        f_loss = self.recon_f_loss(x, c1, h1)
        return c_loss + 0.5 * v_loss + f_loss


def train(dataset, conj_dataset, model, weight_decay, device, args):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        dataset = dataset.to(device)
        conj_dataset = conj_dataset.to(device)

        conj_data = sample_subgraph(conj_dataset, sampling_ratio=args.sampling_ratio)

        out = model(dataset, conj_data)
        c1, c2, h1, h2, edge_feature = out[0], out[1], out[2], out[3], out[4]
        loss = model.loss(x=data.x, c1=c1, h1=h1, c2=c2, h2=h2, edge_feature=edge_feature,
                          edge_index1=dataset.edge_index1, args=args)
        loss.backward()

        optimizer.step()
        print(f"Epoch: {epoch}, Train Loss: {loss.item():.4f}")

        if epoch % args.lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_decay_factor * param_group['lr']

        if epoch % 10 == 0:
            model.eval()
            out = model(dataset, conj_data)
            eval_node(dataset, out, args)
    return model


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--down_epoch', type=int, default=300)
    parser.add_argument('--down_run', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--down_lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=10)
    parser.add_argument('--dropout_ratio', type=float, default=0.25, help='dropout ratio')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--view1', type=str, default='adj')
    parser.add_argument('--view2', type=str, default='KNN')
    parser.add_argument('--AttnNodeFlag', type=bool, default=True)
    parser.add_argument('--FNN_layers', type=int, default=2)
    parser.add_argument('--sampling_ratio', type=int, default=0.01)
    parser.add_argument(
        '--dummy_weight',
        type=float,
        default=0.1,
        help='whether or not to add additional trainable weight on dummy edges. when set to 0.0 this part is disabled.'
    )
    args = parser.parse_args()

    print(args.dataset)

    device = args.device if args.device >= 0 else "cpu"
    seed_everything(42)

    if args.dataset in ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'CS']:
        data = load_node_dataset(args.dataset)
    else:
        data = load_real_datasets(args.dataset)

    args.num_features = data.num_features

    dataset = node_view_generator(data, view=args.view1)

    args.num_classes = data.num_classes

    conj_dataset = node_view_generator(data, view=args.view2)

    conj_dataset = connective_node_generation(conj_dataset)

    print("convert success!")

    model = MVGIB(encoder_c1=GCNEncoder(args),
                  encoder_h1=GCNEncoder(args),
                  encoder_c2=EdgeGCNEncoder(num_node_in_embeddings=conj_dataset.num_features,
                                            num_edge_in_embeddings=data.num_features,
                                            hidden_dim=args.hidden, AttnNodeFlag=args.AttnNodeFlag),
                  encoder_h2=EdgeGCNEncoder(num_node_in_embeddings=conj_dataset.num_features,
                                            num_edge_in_embeddings=data.num_features,
                                            hidden_dim=args.hidden, AttnNodeFlag=args.AttnNodeFlag),
                  encoder_f=FNN(in_dim=2 * args.hidden, hidden_dim=args.hidden, out_dim=data.num_features,
                                num_layers=args.FNN_layers),
                  club=CLUB(encoder=VGCNEncoder(in_dim=data.num_features, hidden_dim=args.hidden,
                                                dropout=args.dropout_ratio))).to(device)

    model = train(dataset=dataset, conj_dataset=conj_dataset, model=model, weight_decay=args.weight_decay,
                  device=device, args=args)
