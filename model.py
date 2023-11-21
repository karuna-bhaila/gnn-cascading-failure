import torch
import torch.nn.functional as F
from torch.nn import Dropout, ReLU

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, MLP
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_dense_adj,
    to_undirected,
)

from layer import NEA_GNNConv


class Classification(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers

        for i in range(self.num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == self.num_layers - 1 else hidden_channels
            self.convs.append(torch.nn.Linear(in_channels, hidden_channels))

        self.activation = ReLU(inplace=True)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != self.num_layers - 1:
                x = self.activation(x)

        logits = F.softmax(x, dim=1)
        return logits


class NEA_GNN(torch.nn.Module):
    def __init__(self, num_layers, dropout, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.name = 'NEA_GNN'

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(p=dropout)
        self.activation = ReLU(inplace=True)

        for i in range(0, self.num_layers):
            if i == 0:
                in_channels = in_channels
            else:
                if i % 2 == 0:
                    in_channels = [hidden_channels[0], hidden_channels[2]+hidden_channels[3]]
                else:
                    in_channels = [hidden_channels[0]+hidden_channels[1], hidden_channels[2]]

            hidden_channels = out_channels if i == self.num_layers-1 else hidden_channels
            self.convs.append(NEA_GNNConv(i, in_channels, hidden_channels, normalize=False, root_weight=True))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x_v, x_e, edge_index, node_edge_index):
        for i, conv in enumerate(self.convs):
            x_v, x_e = conv(x_v, x_e, edge_index, node_edge_index)
            # if i != self.num_layers - 1:
            x_v = self.activation(x_v)
            x_e = self.activation(x_e)
            x_v = self.dropout(x_v)
            x_e = self.dropout(x_e)

        return x_e, x_v


class GNN(torch.nn.Module):
    def __init__(self, _args):
        super().__init__()
        self.num_layers = _args.num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(p=_args.dropout)
        self.activation = ReLU(inplace=True)

    def init_weights(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_uniform_(param)


class GCN(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GCN'

        for i in range(0, self.num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == self.num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_channels, hidden_channels, dropout=_args.dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


class SAGE(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GraphSAGE'

        for i in range(0, self.num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == self.num_layers-1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=False, root_weight=True))

    def forward(self, x, edge_index):
        edge_index = to_undirected(edge_index)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


class GATE(GNN):
    def __init__(self, _args, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GATE'

        heads = 4
        self.convs.append(GATConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=True))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)

        return x


class GAT(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GAT'

        heads = 4
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)

        return x


class GIN(GNN):
    def __init__(self, _args, in_channels, hidden_channels, out_channels):
        super().__init__(_args)
        self.name = 'GIN'

        for _ in range(self.num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        return self.mlp(x)

