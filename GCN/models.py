import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


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

        return F.log_softmax(x, dim=1)


class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, residual, dropout=None):
        super(GCNBlock, self).__init__()

        self.residual = residual

        self.conv = GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        x, edge_index = input
        identity = x

        out = self.conv(x, edge_index)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)
            
        if self.residual:
            out = out + identity

        return out, edge_index


class MultiGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, residual=False, dropout=0.5):
        super(MultiGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.residual = False if num_layer == 1 else residual
        self.dropout = dropout

        self.layers, self.conv = self._make_layer()
    
    def _make_layer(self):
        layers = []
        for i in range(self.num_layer - 1):
            input_dim_ = self.input_dim if i == 0 else self.hidden_dim
            dropout_ = self.dropout if (i == 0) or (i == self.num_layer - 2) else None
            residual_ = self.residual and (input_dim_ == self.hidden_dim)
            layers.append(GCNBlock(input_dim_, self.hidden_dim, residual_, dropout_))

        input_dim_ = self.input_dim if self.num_layer == 1 else self.hidden_dim
        conv = GCNConv(input_dim_, self.output_dim)

        return nn.Sequential(*layers), conv
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x, edge_index = self.layers((x, edge_index))

        if self.conv:
            x = self.conv(x, edge_index)

        return F.log_softmax(x, dim=1)
