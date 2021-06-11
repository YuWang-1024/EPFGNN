import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv
from torch.nn import LayerNorm


class GCNNet(torch.nn.Module):
    def __init__(self,in_channels, num_classes, configs):
        super(GCNNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = GCNConv(in_channels, configs['hidden_size'])
        self.conv2 = GCNConv(configs['hidden_size'], num_classes)
        self.configs = configs


    def reset_parameters(selfs):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x,self.configs['dropout_prob'], training=self.training)
        x = self.conv1(x, edge_index)
        # x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x,self.configs['dropout_prob'], training=self.training)
        x = self.conv2(x, edge_index)

        return x