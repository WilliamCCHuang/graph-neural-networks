import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        
        self.dropout = dropout

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GATConv(MessagePassing):
    """
    Modify the origin implementation of GAT so that attention weights can be saved.
    
    source code: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/gat_conv.py
    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, save_alpha=False, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.alpha = None
        self.save_alpha = save_alpha

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Save alpha
        if self.save_alpha:
            self.alpha = alpha.cpu()

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads_1=8, heads_2=1, att_dropout=0.6, input_dropout=0.6, save_alpha=False):
        super(GAT, self).__init__()
        assert hidden_dim % heads_1 == 0

        self.att_dropout = att_dropout
        self.input_dropout = input_dropout

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=hidden_dim // heads_1,
                             heads=heads_1,
                             concat=True,
                             dropout=att_dropout,
                             save_alpha=save_alpha)
        self.conv2 = GATConv(in_channels=hidden_dim,
                             out_channels=output_dim,
                             heads=heads_2,
                             concat=False,
                             dropout=att_dropout,
                             save_alpha=save_alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.input_dropout, training =self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GATBlock(nn.Module):
    def __init__(self, input_dim, output_dim, heads, concat, residual, att_dropout, input_dropout, save_alpha=False):
        super(GATBlock, self).__init__()

        self.residual = residual

        self.dropout = nn.Dropout(input_dropout)
        self.conv = GATConv(in_channels=input_dim,
                            out_channels=output_dim,
                            heads=heads,
                            concat=concat,
                            dropout=att_dropout,
                            save_alpha=save_alpha)
        self.elu = nn.ELU(inplace=True)

    def forward(self, input):
        x, edge_index = input
        identity = x

        out = self.dropout(x)
        out = self.conv(out, edge_index)
        out = self.elu(out)
        
        if self.residual:
            out = out + identity

        return out, edge_index


class MultiGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, heads=8, residual=False, att_dropout=0.6, input_dropout=0.6, save_alpha=False):
        super(MultiGAT, self).__init__()
        if isinstance(heads, int):
            heads = [heads] * (num_layer - 1) + [1]
        
        for i, head in enumerate(heads):
            if i < len(heads)-1:
                if hidden_dim % head != 0:
                    raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
                                     `heads`.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.heads = heads
        self.residual = residual
        self.att_dropout = att_dropout
        self.input_dropout = input_dropout
        self.save_alpha = save_alpha

        self.layers, self.dropout, self.conv = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(self.num_layer-1):
            input_dim_ = self.input_dim if i == 0 else self.hidden_dim
            residual_ = self.residual and (input_dim_ == self.hidden_dim)
            layers.append(GATBlock(input_dim=input_dim_,
                                   output_dim=self.hidden_dim // self.heads[i],
                                   heads=self.heads[i],
                                   concat=True,
                                   residual=residual_,
                                   att_dropout=self.att_dropout,
                                   input_dropout=self.input_dropout,
                                   save_alpha=self.save_alpha))

        dropout = nn.Dropout(self.input_dropout)

        input_dim_ = self.input_dim if self.num_layer == 1 else self.hidden_dim
        conv = GATConv(in_channels=input_dim_,
                       out_channels=self.output_dim,
                       heads=self.heads[-1],
                       concat=False,
                       dropout=self.att_dropout,
                       save_alpha=self.save_alpha)
        
        return nn.Sequential(*layers), dropout, conv

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x, edge_index = self.layers((x, edge_index))
        x = self.dropout(x)
        x = self.conv(x, edge_index)

        return x