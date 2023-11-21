import sys
import argparse
import numpy as np
import random

import torch
from torch_geometric.loader import DataLoader

from data import load_dataset
from model import NEA_GNN, Classification
from model import GCN, SAGE, GAT, GIN, GATE
from train import Trainer
from utils import data_split


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')

    # Dataset
    parser.add_argument('--dataset', type=str, default='case39')
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.60)
    parser.add_argument('--val_ratio', type=float, default=0.20)

    # GNN
    parser.add_argument('--model', type=str, default='nea_gnn',
                        help='Architecture:sage,gcn,gat')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=16,
                        help='Hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for node label loss')
    parser.add_argument('--class_weight', type=float, default=1.,
                        help='Weight for majority class')

    parser.add_argument('--gpu_id', type=int, default=1)

    return parser.parse_args()


def pprint(args):
    for k, v in args.__dict__.items():
        print("\t- {}: {}".format(k, v))


if __name__ == '__main__':
    args = get_arguments()

    args.model = args.model.upper()
    args.loss_weight = torch.tensor([1, args.class_weight])
    pprint(args)
    print()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data = load_dataset(args.dataset)

    initial_profile = data[0]

    train_dataset, val_dataset, test_dataset = data_split(data[1:], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        gpu_id = int(args.gpu_id) if args.gpu_id is not None else 1
        torch.cuda.set_device(gpu_id)
        args.device = torch.device('cuda:{}'.format(gpu_id))
        print('Using GPU with ID:{}'.format(torch.cuda.current_device()))
    else:
        print("CUDA not available!")

    num_node_features = initial_profile.x.shape[1]
    num_edge_features = initial_profile.edge_attr.shape[1]

    hidden_node = args.hidden
    hidden_edge = args.hidden

    clf_layers = 2

    if args.model == 'GCN':
        gnn = GCN(args, num_node_features, args.hidden, args.hidden)

    elif args.model == 'SAGE':
        gnn = SAGE(args, num_node_features, args.hidden, args.hidden)

    elif args.model == 'GAT':  # 4 attention heads
        gnn = GAT(args, num_node_features, args.hidden, args.hidden)

    elif args.model == 'GATE':
        gnn = GATE(args, num_node_features, num_edge_features, args.hidden, args.hidden)

    elif args.model == 'GIN':
        gnn = GIN(args, num_node_features, args.hidden, args.hidden)

    elif args.model == 'NEA_GNN':
        gnn = NEA_GNN(args.num_layers,
                      args.dropout,
                      in_channels=[num_node_features, num_edge_features],
                      hidden_channels=[args.hidden, 1, args.hidden, 1],
                      out_channels=[args.hidden, 1, args.hidden, 1])
        if args.num_layers % 2 == 0:
            hidden_node = args.hidden
            hidden_edge = args.hidden + 1
        else:
            hidden_node = args.hidden + 1
            hidden_edge = args.hidden

    else:
        print('Error: model not implemented')
        sys.exit()

    node_classifier = Classification(num_layers=clf_layers,
                                     in_channels=hidden_node,
                                     hidden_channels=128,
                                     out_channels=2)
    edge_classifier = None
    if args.model in ['NEA_GNN']:
        edge_classifier = Classification(num_layers=clf_layers,
                                         in_channels=hidden_edge,
                                         hidden_channels=128,
                                         out_channels=2)

    trainer = Trainer(args, gnn, node_classifier, edge_classifier)

    trainer.profile = initial_profile.to(args.device)
    results = trainer.fit(train_loader, val_loader, test_loader)

    # Display best metrics
    print("Best metrics: Val Acc: {:.2f}, Edge Acc: {:.2f}, Edge Bacc: {:.2f}, "
          "Node Acc: {:.2f}, Node Bacc: {:.2f}\n".
          format(results['best_val_acc'], results['test_acc_at_best_val_acc'], results['test_bacc_at_best_val_acc'],
                 results['test_node_acc_at_best_val_acc'], results['test_node_bacc_at_best_val_acc']))

