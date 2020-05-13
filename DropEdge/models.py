import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
from torch_geometric.utils.dropout import dropout_adj


class GCNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout):
        super(GCNBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.conv = GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout(x)

        return x

    def __repr__(self):
        return (f'GCNBlock(input_dim={self.input_dim}, '
                f'output_dim={self.output_dim}, '
                f'dropout={self.dropout_rate})')


class ResGCNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout):
        super(ResGCNBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.conv = GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.residual = (input_dim == output_dim)

    def forward(self, x, edge_index):
        identity = x

        x = self.conv(x, edge_index)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout(x)

        if self.residual:
            x = x + identity

        return x

    def __repr__(self):
        return (f'ResGCNBlock(input_dim={self.input_dim}, '
                f'output_dim={self.output_dim}, '
                f'dropout={self.dropout_rate} '
                f'residual={self.residual})')


class IncepGCNBlock(nn.Module):

    def __init__(self, input_dim, output_dim, dropout):
        super(IncepGCNBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.branch_1 = nn.ModuleList([
            GCNConv(input_dim, output_dim),
        ])
        self.branch_2 = nn.ModuleList([
            GCNConv(input_dim, output_dim),
            GCNConv(output_dim, output_dim),
        ])
        self.branch_3 = nn.ModuleList([
            GCNConv(input_dim, output_dim),
            GCNConv(output_dim, output_dim),
            GCNConv(output_dim, output_dim),
        ])
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.last_conv = GCNConv(output_dim * 3, output_dim)

    def branch_forward(self, branch, x, edge_index):
        for layer in branch:
            x = layer(x, edge_index)
            x = self.relu(x)

            if self.dropout:
                x = self.dropout(x)

        return x

    def forward(self, x, edge_index):
        outputs = []

        for branch in [self.branch_1, self.branch_2, self.branch_3]:
            outputs.append(self.branch_forward(branch, x, edge_index))

        x = torch.cat(outputs, dim=1)  # [#nodes, out_dim * 3]
        x = self.last_conv(x, edge_index)

        return x

    def __repr__(self):
        return (f'IncepGCNBlock(input_dim={self.input_dim}, '
                f'output_dim={self.output_dim}, '
                f'dropout={self.dropout_rate})')


class MultiGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, block, dropout, edge_dropout, layer_wise_dropedge):
        super(MultiGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.layer_wise_dropedge = layer_wise_dropedge

        self.conv_layers = self._make_layer(block)
    
    def _make_layer(self, block):
        layers = []

        if block in [GCNBlock, ResGCNBlock]:
            for i in range(self.n_layers - 1):
                input_dim = self.input_dim if i == 0 else self.hidden_dim
                output_dim = self.output_dim if i == self.n_layers - 1 else self.hidden_dim
                layers.append(block(input_dim, output_dim, self.dropout))

            input_dim_ = self.input_dim if self.n_layers == 1 else self.hidden_dim
            layers.append(GCNConv(input_dim_, self.output_dim))
        elif block is IncepGCNBlock:
            assert self.n_layers % 4 == 0

            for i in range(self.n_layers // 4):
                input_dim = self.input_dim if i == 0 else self.hidden_dim
                output_dim = self.output_dim if i == self.n_layers // 4 - 1 else self.hidden_dim
                layers.append(block(input_dim, output_dim, self.dropout))

        return nn.ModuleList(layers)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        dropedge_index = edge_index

        if self.edge_dropout:
            dropedge_index, _ = dropout_adj(edge_index,
                                            p=self.edge_dropout,
                                            force_undirected=True,
                                            training=self.training)
        
        for layer in self.conv_layers:
            if self.layer_wise_dropedge and self.edge_dropout:
                dropedge_index, _ = dropout_adj(edge_index,
                                                p=self.edge_dropout,
                                                force_undirected=True,
                                                training=self.training)

            x = layer(x, dropedge_index)

        return x

    def __str__(self):
        block = self.conv_layers[0]

        if isinstance(block, GCNBlock):
            model_name = 'GCN'
        elif isinstance(block, ResGCNBlock):
            model_name = 'ResGCN'
        elif isinstance(block, IncepGCNBlock):
            model_name = 'IncepGCN'
        else:
            raise ValueError('Wrong block')

        if self.edge_dropout:
            return f'{model_name}-{self.n_layers}+DropEdge'
        else:
            return f'{model_name}-{self.n_layers}'

    def __repr__(self):
        return 'conv layers: ' + self.conv_layers.__repr__()


def GCN(input_dim, hidden_dim, output_dim, n_layers, dropout=0.5, edge_dropout=0.5, layer_wise_dropedge=False):
    return MultiGCN(input_dim, hidden_dim, output_dim, n_layers, GCNBlock, dropout, edge_dropout, layer_wise_dropedge)


def ResGCN(input_dim, hidden_dim, output_dim, n_layers, dropout=0.5, edge_dropout=0.5, layer_wise_dropedge=False):
    return MultiGCN(input_dim, hidden_dim, output_dim, n_layers, ResGCNBlock, dropout, edge_dropout, layer_wise_dropedge)


def IncepGCN(input_dim, hidden_dim, output_dim, n_layers, dropout=0.5, edge_dropout=0.5):
    return MultiGCN(input_dim, hidden_dim, output_dim, n_layers, IncepGCNBlock, dropout, edge_dropout, layer_wise_dropedge=False)


class JKNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.5, edge_dropout=0.5):
        super(JKNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout

        self.conv_layers = self._make_layer()
        self.last_conv = GCNConv((n_layers - 1) * hidden_dim, output_dim)

    def _make_layer(self):
        layers = []
        for i in range(self.n_layers - 1):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            output_dim = self.hidden_dim
            layers.append(GCNBlock(input_dim, output_dim, self.dropout))

        return nn.ModuleList(layers)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        dropedge_index = edge_index

        if self.edge_dropout:
            dropedge_index, _ = dropout_adj(edge_index,
                                            p=self.edge_dropout,
                                            force_undirected=True,
                                            training=self.training)
        
        outputs = []

        for layer in self.conv_layers:
            x = layer(x, dropedge_index)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.last_conv(x, dropedge_index)

        return x

    def __str__(self):
        if self.edge_dropout:
            return f'JKNet-{self.n_layers}+DropEdge'
        else:
            return f'JKNet-{self.n_layers}'

    def __repr__(self):
        output = 'conv layers: ' + self.conv_layers.__repr__()
        output += '\n'
        output += 'last conv: ' + self.last_conv.__repr__()

        return output


if __name__ == "__main__":
    from torch_geometric.data import Data

    input_dim = 2
    hidden_dim = 3
    output_dim = 4
    dropout = 0.5
    n_layers = 5

    x = torch.zeros((3, input_dim), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2],
                               [0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.long)

    data = Data(x, edge_index)

    # for model_calss in [GCN, ResGCN, JKNet, IncepGCN]:
    for model_calss in [GCN]:
        for edge_dropout in [None, 0.5]:
            if model_calss is IncepGCN:
                n_layers = 8

            model = model_calss(input_dim, hidden_dim, output_dim, n_layers, dropout, edge_dropout)
            print(model)
            print(repr(model))
            print()

            model(data)
