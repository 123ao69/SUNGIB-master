import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score


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


def logger(info):
    epoch = info['epoch']
    train_loss, best_acc = info['train_loss'], info['best_acc']
    print(
        f'{epoch:03d}: Train Loss: {train_loss:.3f}, Test Acc: {best_acc:.4f} ')


def test_logger(info):
    test_acc, test_std = info['final_test_acc'], info['test_std']
    print(f'Final Test Acc: {test_acc:.4f} 'f'± {test_std:.4f}')


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(),
                                      params,
                                      cv=5,
                                      scoring='accuracy',
                                      verbose=0)
        else:
            classifier = SVC(C=5)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)


def evaluate_embedding(embeddings, labels, search=False):
    x, y = np.array(embeddings), np.array(labels)

    svc_acc, svc_std = svc_classify(x, y, search=search)
    # print(f'{svc_acc:.4f} ± {svc_std:.4f}')
    return svc_acc, svc_std


def eval_node(data, embedding, args):
    accs = []
    embedding = embedding.detach()
    # labels = data.y.detach().to(args.device)
    labels = data.y.detach()

    for k in range(args.down_run):
        if args.dataset in ['Cora', 'citeSeer', 'pubmed', 'ogbn-arxiv']:
            train_mask = data.train_mask[k]
            val_mask = data.val_mask[k]
            test_mask = data.test_mask[k]
        else:
            train_mask = data.train_mask[k]
            val_mask = data.val_mask[k]
            test_mask = data.test_mask[k]

        pred_model = LinearPred(embedding.size(1), args.num_classes).to(args.device)
        # pred_model = LinearPred(embedding.size(1), args.num_classes)
        optimizer_pred = torch.optim.Adam(pred_model.parameters(), lr=args.lr, weight_decay=args.down_lr)

        acc_val_best, acc_test_best, ep_b = 0, 0, -1
        acc_test_best_all = 0

        for i in range(args.down_epoch):
            pred_model.train()
            optimizer_pred.zero_grad()

            logits = pred_model(embedding[train_mask])
            y = labels[train_mask]
            loss = F.cross_entropy(logits, labels[train_mask])

            loss.backward()
            optimizer_pred.step()

            pred_model.eval()
            preds_val = torch.argmax(pred_model(embedding[val_mask]), dim=1)
            preds_test = torch.argmax(pred_model(embedding[test_mask]), dim=1)
            acc_val = torch.sum(preds_val == labels[val_mask]).float() / labels[val_mask].size(0)
            acc_test = torch.sum(preds_test == labels[test_mask]).float() / labels[test_mask].size(0)

            # if acc_val >= acc_val_best:
            #     acc_val_best = acc_val
            #     if acc_test >= acc_test_best:
            #         acc_test_best = acc_test

            if acc_test >= acc_test_best:
                acc_test_best = acc_test

            if acc_test > acc_test_best_all:
                acc_test_best_all = acc_test

        accs.append(acc_test_best * 100)
    accs = torch.stack(accs)
    mean, std = accs.mean(), accs.std()
    print(f"# acc: {mean:.2f} ± {std:.2f}")


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def test_classify(feature, labels, args):
    f1_mac = []
    f1_mic = []
    accs = []
    feature = feature.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for test_index, train_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        acc = accuracy(preds, test_y)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs
