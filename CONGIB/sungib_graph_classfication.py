import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import (add_self_loops, negative_sampling, remove_self_loops)

from data import load_dataset
from encoder import FNN, GINEncoder, VGINEncoder, EdgeGCNEncoder
from utils import evaluate_embedding, logger, view_generator, connective_graph_generation


class CLUB(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, batch, h):
        mu, logvar = self.encoder(x, edge_index, batch)

        positive = -(mu - h) ** 2 / 2. / (logvar.exp() + 1e-7)

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        h_samples_1 = h.unsqueeze(0)  # shape [1,nsample,dim]

        negative = -((h_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / (
                logvar.exp() + 1e-7)

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x, edge_index, batch, h):
        mu, logvar = self.encoder(x, edge_index, batch)
        return (-(mu - h) ** 2 / logvar.exp() - logvar +
                1e-7).sum(dim=1).mean(dim=0)

    def learning_loss(self, x, edge_index, batch, h):
        return -self.loglikeli(x, edge_index, batch, h)


class CONGIB(torch.nn.Module):
    def __init__(self, encoder_c1, encoder_h1, encoder_c2, encoder_h2,
                 encoder_f, club):
        super().__init__()
        self._replace_rate = 0.1
        self._mask_token_rate = 1 - self._replace_rate

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
    def Delete_duplicate_ID(conj_g, edge_feat):
        new_edge_feat = torch.zeros(len(conj_g.batch), edge_feat.shape[1], device="cuda:0")
        pre = 0
        j = 0
        for i in range(conj_g.num_graphs):
            ids = conj_g[i].edge_is_dummy
            ids = torch.tensor(ids)
            unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)
            counts = torch.bincount(inverse_indices).to("cuda:0")
            sums = torch.zeros(len(unique_ids), edge_feat[pre:(pre + len(ids))].shape[1], dtype=edge_feat.dtype).to(
                "cuda:0")
            index = inverse_indices.unsqueeze(1).repeat(1, edge_feat[pre:(pre + len(ids))].shape[1]).to("cuda:0")
            sums.scatter_add_(dim=0, index=index, src=edge_feat[pre:(pre + len(ids))])

            mean = sums / counts.unsqueeze(-1)
            new_edge_feat[j:(j + len(unique_ids))] = mean
            pre += len(ids)
            j += len(unique_ids)
        return new_edge_feat

    def forward(self, data, conj_load):
        x, batch = data.x, data.batch
        edge_index1 = data.edge_index1
        edge_x, edge_batch = conj_load.edge_attr, conj_load.batch

        data.to("cuda:0")
        conj_load.to("cuda:0")

        c1 = self.encoder_c1(x, edge_index1, batch)
        c2 = self.encoder_c2(conj_load, edge_x)

        h1 = self.encoder_h1(x, edge_index1, batch)
        h2 = self.encoder_h2(conj_load, edge_x)

        c2 = self.Delete_duplicate_ID(conj_load, c2)
        h2 = self.Delete_duplicate_ID(conj_load, h2)

        c2 = global_mean_pool(c2, conj_load.batch)
        h2 = global_mean_pool(h2, conj_load.batch)

        if self.training:
            return c1, c2, h1, h2
        else:
            return torch.cat([c1, c2, h1, h2], dim=-1)

    @staticmethod
    def sce_loss(x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def recon_f_loss(self, x, h1, h2, c1, c2, batch):
        z = torch.cat([h1, h2, c1, c2], dim=-1)
        z = z[batch]
        re_f = self.encoder_f(z)
        f_loss = self.sce_loss(re_f, x, alpha=1)
        return f_loss

    def recon_loss(self, z, edge_index, batch):
        z = z[batch]
        pos_loss = -torch.log(self.decoder(z, edge_index) + 1e-7).mean()

        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index)

        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) + 1e-7).mean()
        return pos_loss + neg_loss

    def common_loss(self, c1, c2, edge_index1, edge_index2, batch, conj_batch):
        loss_t = self.recon_loss(c1, edge_index1, batch) + self.recon_loss(c2, edge_index2, conj_batch)
        return loss_t

    def specific_loss(self, x, h1, h2, edge_index1, edge_index2, batch, conj_batch):
        loss_t = self.recon_loss(h1, edge_index1, batch) + self.recon_loss(h2, edge_index2, conj_batch)

        loss_t_p = self.club(x, edge_index1, batch, h2) + self.club.learning_loss(x, edge_index1, batch, h2)

        return loss_t + 0.1 * loss_t_p

    def loss(self, x, c1, h1, c2, h2, edge_index1, edge_index2, batch, conj_batch):
        c_loss = self.common_loss(c1, c2, edge_index1, edge_index2, batch, conj_batch)
        v_loss = self.specific_loss(x, h1, h2, edge_index1, edge_index2, batch, conj_batch)
        f_loss = self.recon_f_loss(x, h1, h2, c1, c2, batch)
        return c_loss + v_loss + f_loss


def train(model, optimizer, loader, conj_loader, device):
    model.train()

    total_loss = []
    for data, conj in zip(loader, conj_loader):
        optimizer.zero_grad()
        data = data.to(device)
        conj = conj.to(device)

        out = model(data, conj)
        c1, c2, h1, h2 = out[0], out[1], out[2], out[3]
        loss = model.loss(x=data.x, c1=c1, h1=h1, c2=c2, h2=h2, edge_index1=data.edge_index1,
                          edge_index2=conj.edge_index,
                          batch=data.batch, conj_batch=conj.batch)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()
    train_loss = np.mean(total_loss)
    return train_loss


def test(model, loader, conj_loader, device):
    model.eval()
    z, y = [], []

    for data, conj in zip(loader, conj_loader):
        data = data.to(device)
        conj = conj.to(device)
        with torch.no_grad():
            out = model(data, conj)
        z.append(out.cpu().numpy())
        y.append(data.y.cpu().numpy())

    z = np.concatenate(z, 0)
    y = np.concatenate(y, 0)
    test_acc, test_std = evaluate_embedding(z, y)

    return test_acc, test_std


def cross_validation(dataset, conj_dataset, model, epochs, batch_size, lr, lr_decay_factor,
                     lr_decay_step_size, weight_decay, device):
    model.to(device)

    loader = DataLoader(dataset, batch_size, shuffle=True)
    conj_loader = DataLoader(conj_dataset, batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    eval_loader = DataLoader(dataset, batch_size, shuffle=False)
    eval_conj_loader = DataLoader(conj_dataset, batch_size, shuffle=False)

    best_acc, best_std = 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model=model,
                           optimizer=optimizer,
                           loader=loader,
                           conj_loader=conj_loader,
                           device=device)

        test_acc, test_std = test(model=model, loader=eval_loader, conj_loader=eval_conj_loader, device=device)

        if test_acc >= best_acc:
            best_acc = test_acc

        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'best_acc': best_acc
        }

        logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    return model


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=10)
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--view1', type=str, default='adj')
    parser.add_argument('--view2', type=str, default='KNN')
    args = parser.parse_args()

    device = args.device if args.device >= 0 else "cpu"
    seed_everything(42)

    data = load_dataset(args.dataset, cleaned=False)

    args.num_classes = data.num_classes
    args.num_features = data.num_features

    dataset = view_generator(data, view=args.view1)

    dataset1 = view_generator(data, view=args.view2)

    conj_dataset = connective_graph_generation(dataset1)
    print("convert success!")

    model = CONGIB(encoder_c1=GINEncoder(in_dim=data.num_features, hidden_dim=args.hidden, num_layers=args.num_layers),
                  encoder_h1=GINEncoder(in_dim=data.num_features, hidden_dim=args.hidden, num_layers=args.num_layers),
                  encoder_c2=EdgeGCNEncoder(num_node_in_embeddings=data.num_edge_features,
                                            num_edge_in_embeddings=data.num_features,
                                            hidden_dim=args.hidden, AttnNodeFlag=False),
                  encoder_h2=EdgeGCNEncoder(num_node_in_embeddings=data.num_edge_features,
                                            num_edge_in_embeddings=data.num_features,
                                            hidden_dim=args.hidden, AttnNodeFlag=False),
                  encoder_f=FNN(in_dim=4 * args.hidden, hidden_dim=args.hidden, out_dim=data.num_features,
                                num_layers=2),
                  club=CLUB(encoder=VGINEncoder(in_dim=data.num_features, hidden_dim=args.hidden,
                                                num_layers=2))).to(device)

    model = cross_validation(dataset=dataset,
                             conj_dataset=conj_dataset,
                             model=model,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             lr=args.lr,
                             lr_decay_factor=args.lr_decay_factor,
                             lr_decay_step_size=args.lr_decay_step_size,
                             weight_decay=0,
                             device=device)
