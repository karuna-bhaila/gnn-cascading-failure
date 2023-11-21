import sys
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss


class Trainer:
    def __init__(self, _args, gnn=None, classifier=None, classifier_e=None, net=None, optimizer='adam'):
        self.max_epochs = _args.epochs
        self.device = _args.device
        self.lr = _args.lr
        self.weight_decay = _args.weight_decay
        self.gnn = gnn
        self.classifier = classifier
        self.classifier_e = classifier_e
        self.net = net
        self.optimizer_name = optimizer

        self.beta = _args.beta

        self.criterion = CrossEntropyLoss(reduction='mean', weight=_args.loss_weight.to(self.device))

    def configure_optimizers(self):
        models = [self.gnn, self.classifier, self.classifier_e]
        params = []

        for model in models:
            if model is not None:
                for param in model.parameters():
                    if param.requires_grad:
                        params.append(param)

        if self.optimizer_name == 'sgd':
            return SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            print("optimizer not configured")
            sys.exit(1)

    def forward(self, batch):
        logits_e, logits_v = None, None

        if self.gnn is not None:
            if self.gnn.name in ['GCN', 'GraphSAGE', 'GIN', 'GAT']:
                embeds_v = self.gnn(batch.x, batch.edge_index)
                logits_v = self.classifier(embeds_v)
                source_v = embeds_v[batch.edge_index[0]]
                target_v = embeds_v[batch.edge_index[1]]
                prob = (source_v * target_v).sum(dim=1).view(-1, 1)
                logits_e = torch.cat((torch.sub(1, prob), prob), dim=1)

            elif self.gnn.name in ['GATE']:
                embeds_v = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
                logits_v = self.classifier(embeds_v)
                source_v = embeds_v[batch.edge_index[0]]
                target_v = embeds_v[batch.edge_index[1]]
                prob = (source_v * target_v).sum(dim=1).view(-1, 1)
                logits_e = torch.cat((torch.sub(1, prob), prob), dim=1)

            elif self.gnn.name in ['NEA_GNN']:
                embeds_e, embeds_v = self.gnn(batch.x, batch.edge_attr, batch.edge_index, batch.node_edge_index)
                logits_v = self.classifier(embeds_v)
                logits_e = self.classifier_e(embeds_e)

        return logits_e, logits_v

    def train(self, dataloader, optimizer):
        if self.gnn is not None:
            self.gnn.train()

        self.classifier.train()

        if self.classifier_e is not None:
            self.classifier_e.train()

        total_loss = 0.
        edge_y_true, edge_y_pred = [], []

        for index, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(self.device)

            logits_e, logits_v = self.forward(batch)

            # Edge label loss
            mask = batch.edge_attr[:, 0] > 0
            loss = self.criterion(input=logits_e, target=batch.edge_label)

            # Store true and predicted labels from each batch
            edge_y_true.extend(batch.edge_label.argmax(dim=1).cpu().detach().numpy().tolist())
            edge_y_pred.extend(logits_e.argmax(dim=1).cpu().detach().numpy().tolist())

            # Node label loss
            v_loss = self.criterion(input=logits_v, target=batch.y)
            loss = loss + self.beta * v_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss = total_loss / len(dataloader)

        acc = accuracy_score(edge_y_true, edge_y_pred) * 100
        b_acc = balanced_accuracy_score(edge_y_true, edge_y_pred) * 100

        return loss, acc, b_acc

    def fit(self, train_loader, val_loader, test_loader=None, verbose=True):
        if self.gnn is not None:
            self.gnn = self.gnn.to(self.device)

        self.classifier = self.classifier.to(self.device)

        if self.classifier_e is not None:
            self.classifier_e = self.classifier_e.to(self.device)

        optimizer = self.configure_optimizers()

        best_val_acc = 0.
        best_test_acc = 0.
        res = {}

        for epoch in range(1, self.max_epochs+1):
            train_loss, train_acc, train_bacc = self.train(train_loader, optimizer)
            val_loss, val_acc, val_bacc, _, _ = self.test(val_loader)
            test_loss, test_acc, test_bacc, node_acc, node_bacc = self.test(test_loader)

            if node_acc is None:
                node_acc = node_bacc = 0.

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                res = {
                    "best_test_acc": best_test_acc,
                }

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_at_best_val = test_acc
                test_bacc_at_best_val = test_bacc
                test_node_at_best_val = node_acc
                test_node_bacc_at_best_val = node_bacc
                res["best_val_acc"] = best_val_acc
                res["test_acc_at_best_val_acc"] = test_at_best_val
                res["test_bacc_at_best_val_acc"] = test_bacc_at_best_val
                res["test_node_acc_at_best_val_acc"] = test_node_at_best_val
                res["test_node_bacc_at_best_val_acc"] = test_node_bacc_at_best_val

            if verbose:
                print("Epoch [{}/{}], Train Loss: {:.2f}, Train Acc: {:.2f}, Train Bacc: {:.2f} "
                      "Val Loss: {:.2f}, Val Acc: {:.2f}, Val Bacc: {:.2f} "
                      "Test Loss: {:.2f},\n Edge Acc: {:.2f}, Edge Bacc: {:.2f}, "
                      "Node Acc: {:.2f}, Node Bacc: {:.2f}".
                      format(epoch, self.max_epochs, train_loss, train_acc, train_bacc,
                             val_loss, val_acc, val_bacc,
                             test_loss, test_acc, test_bacc, node_acc, node_bacc))

        return res

    @torch.no_grad()
    def test(self, dataloader, verbose=False):
        if self.gnn is not None:
            self.gnn.eval()

        self.classifier.eval()

        if self.classifier_e is not None:
            self.classifier_e.eval()

        total_loss = 0.
        edge_y_true = np.array([])
        edge_y_pred = np.array([])
        node_y_true = np.array([])
        node_y_pred = np.array([])
        test_mask = np.array([])

        for i, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            logits_e, logits_v = self.forward(batch)

            mask = batch.edge_attr[:, 0] > 0
            loss = self.criterion(input=logits_e, target=batch.edge_label)

            # Assign true labels to failed branches (failure status already known)
            logits_e[~mask] = batch.edge_label[~mask]

            # Node label loss
            v_loss = self.criterion(input=logits_v, target=batch.y)
            loss += self.beta * v_loss

            edge_y_true = np.append(edge_y_true, batch.edge_label.argmax(dim=1).cpu().detach().numpy())
            edge_y_pred = np.append(edge_y_pred, logits_e.argmax(dim=1).cpu().detach().numpy())
            node_y_true = np.append(node_y_true, batch.y.argmax(dim=1).cpu().detach().numpy())
            node_y_pred = np.append(node_y_pred, logits_v.argmax(dim=1).cpu().detach().numpy())
            test_mask = np.append(test_mask, mask.cpu().detach().numpy())

            total_loss += loss.item()

        loss = float(total_loss) / len(dataloader)

        if verbose:
            print(torch.tensor(edge_y_pred).unique(return_counts=True))

        # unmasked perf
        acc = accuracy_score(edge_y_true.tolist(), edge_y_pred.tolist()) * 100
        bacc = balanced_accuracy_score(edge_y_true.tolist(), edge_y_pred.tolist()) * 100
        node_acc = accuracy_score(node_y_true.tolist(), node_y_pred.tolist()) * 100
        node_bacc = balanced_accuracy_score(node_y_true.tolist(), node_y_pred.tolist()) * 100

        # masked perf
        test_mask = np.where(test_mask)[0].tolist()
        mask_edge_acc = accuracy_score(edge_y_true[test_mask].tolist(), edge_y_pred[test_mask].tolist()) * 100
        mask_edge_bacc = balanced_accuracy_score(edge_y_true[test_mask].tolist(), edge_y_pred[test_mask].tolist()) * 100

        return loss, acc, bacc, node_acc, node_bacc


