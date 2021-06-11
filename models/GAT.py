import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, configs):
        super(GATNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = configs['dropout_prob']
        self.conv1 = GATConv(
            in_channels,
            configs['hidden_size'],
            heads=configs['num_heads'],
            dropout=configs['dropout_prob'])
        self.conv2 = GATConv(
            configs['hidden_size'] * configs['num_heads'],
            num_classes,
            heads=configs['num_output_heads'],
            concat=False,
            dropout=configs['dropout_prob'])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)