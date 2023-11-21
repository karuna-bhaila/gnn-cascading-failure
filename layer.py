import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size


class NEA_GNNConv(MessagePassing):
    def __init__(self, layer, in_channels, out_channels, root_weight=True, bias=True, aggr='mean', **kwargs):
        super(NEA_GNNConv, self).__init__(aggr=aggr, **kwargs)

        self.layer = layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.bias = bias

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)

        self.lin_w1 = Linear(in_channels[0], out_channels[0], bias=bias)
        if self.root_weight:
            self.lin_r_1 = Linear(in_channels[0], out_channels[0], bias=False)
        # Edge-to-node
        if self.layer % 2 == 0:
            self.lin_w2 = Linear(out_channels[2], out_channels[1], bias=bias)

        self.lin_w3 = Linear(in_channels[1], out_channels[2], bias=bias)
        # Node-to-edge
        if self.layer % 2 == 1:
            self.lin_w4 = Linear(out_channels[0], out_channels[3], bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_w1.reset_parameters()
        if self.layer % 2 == 0:
            self.lin_w2.reset_parameters()
        self.lin_w3.reset_parameters()
        if self.layer % 2 == 1:
            self.lin_w4.reset_parameters()
        if self.root_weight:
            self.lin_r_1.reset_parameters()

    def forward(self, x_v, x_e, edge_index, node_edge_index):
        if isinstance(x_v, Tensor):
            x_v = (x_v, x_v)

        if isinstance(x_e, Tensor):
            x_e = (x_e, x_e)

        # Propagate in G
        out_v = self.propagate(edge_index, x=x_v[0])
        out_v = self.lin_w1(out_v)

        if self.root_weight:
            out_v = out_v + self.lin_r_1(x_v[1])

        # Linear transformation of x_e
        out_e = self.lin_w3(x_e[0])

        # Edge-to-node message passing
        if self.layer % 2 == 0:
            out_ev = self.propagate(node_edge_index.flip([0]), x=out_e, size=(x_e[0].shape[0], x_v[0].shape[0]))
            out_ev = self.lin_w2(out_ev)
            out_v = torch.cat((out_v, out_ev), dim=1)

        # Node-to-edge message passing
        if self.layer % 2 == 1:
            out_ve = self.propagate(node_edge_index, x=out_v, size=(x_v[0].shape[0], x_e[0].shape[0]))
            out_ve = self.lin_w4(out_ve)
            out_e = torch.cat((out_e, out_ve), dim=1)

        return out_v, out_e

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor):  # noqa
        adj_t = adj_t.set_value(None, layout=None)
        out = matmul(adj_t, x, reduce=self.aggr)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
