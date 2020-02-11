import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv


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


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_head=8, att_dropout=0.6, input_dropout=0.6):
        super(GAT, self).__init__()
        assert hidden_dim % num_head == 0

        self.att_dropout = att_dropout
        self.input_dropout = input_dropout

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=hidden_dim // num_head,
                             heads=num_head,
                             concat=True, dropout=att_dropout)
        self.conv2 = GATConv(in_channels=hidden_dim,
                             out_channels=output_dim,
                             heads=1, concat=False, dropout=att_dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.input_dropout, training =self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GATBlock(nn.Module):
    def __init__(self, input_dim, output_dim, heads, concat, residual, att_dropout, input_dropout):
        super(GATBlock, self).__init__()

        self.residual = residual

        self.dropout = nn.Dropout(input_dropout)
        self.conv = GATConv(in_channels=input_dim,
                            out_channels=output_dim,
                            heads=heads, concat=concat, dropout=att_dropout)
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, num_head=8, residual=False, att_dropout=0.6, input_dropout=0.6):
        super(MultiGAT, self).__init__()
        assert hidden_dim % num_head == 0

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.heads = num_head
        self.residual = residual
        self.att_dropout = att_dropout
        self.input_dropout = input_dropout

        self.layers, self.dropout, self.conv = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(self.num_layer-1):
            input_dim_ = self.input_dim if i == 0 else self.hidden_dim
            residual_ = self.residual and (input_dim_ == self.hidden_dim)
            layers.append(GATBlock(input_dim=input_dim_,
                                   output_dim=self.hidden_dim // self.heads,
                                   heads=self.heads, concat=True,
                                   residual=residual_,
                                   att_dropout=self.att_dropout,
                                   input_dropout=self.input_dropout))

        dropout = nn.Dropout(self.input_dropout)

        input_dim_ = self.input_dim if self.num_layer == 1 else self.hidden_dim
        conv = GATConv(in_channels=input_dim_,
                       out_channels=self.output_dim,
                       heads=1, concat=False, dropout=self.att_dropout)
        
        return nn.Sequential(*layers), dropout, conv

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x, edge_index = self.layers((x, edge_index))
        x = self.dropout(x)
        x = self.conv(x, edge_index)

        return F.log_softmax(x, dim=1)