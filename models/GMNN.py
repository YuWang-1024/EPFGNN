import torch
from .GCN import GCNNet

class GMNNNet(torch.nn.Module):
    def __init__(self, backbone_name, in_channels, out_channels, configs):
        super(GMNNNet, self).__init__()
        self.GNN_q = backbone_name(in_channels, out_channels, configs)
        self.GNN_p = backbone_name(out_channels, out_channels, configs)

    def reset_parameters(self):
        self.GNN_p.reset_parameters()
        self.GNN_q.reset_parameters()

    def forward(self, procedure, data):
        if procedure=="E":
            return self.GNN_q(data)
        elif procedure=="M":
            return self.GNN_p(data)
        elif procedure=="Pre":
            return self.GNN_q(data)
        else: assert "procedure not clear"