import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv
import torch.nn as nn

""" modules that used to construct PWGNN"""
class BPLeafToRoot(MessagePassing):
    def __init__(self):
        super(BPLeafToRoot, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed):
        # here to solve the problem of overcounting, we forward distributed unary and binary energy
        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed)

    def message(self, x_j, binary):

        N, C = x_j.shape
        messages = torch.logsumexp(x_j.view(N, -1, 1) + binary, axis=1)
        return messages

    def update(self, aggr_out,x):

        log_z = torch.logsumexp((x + aggr_out),axis=1)
        # normalizer for every piece, i.e. for every node.
        return log_z

class BPLeafToRoot_edge_rezero(MessagePassing):
    # average redistribution + edge rezero
    def __init__(self):
        # super(BPLeafToRoot_edge_rezero, self).__init__(aggr='add', flow="source_to_target")
        super(BPLeafToRoot_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed, rezero):
        # here to solve the problem of overcounting, we forward distributed unary and binary energy
        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, rezero=rezero)

    def message(self, x_j, binary, rezero):

        N, C = x_j.shape
        messages = torch.logsumexp( (x_j.view(N, -1, 1) + rezero.view(-1, 1, 1) * binary), axis=1)
        return messages

    def update(self, aggr_out,x):

        log_z = torch.logsumexp((x + aggr_out),axis=1)
        # normalizer for every piece, i.e. for every node.
        return log_z

class BPLeafToRoot_center(MessagePassing):
    def __init__(self):
        super(BPLeafToRoot_center, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, unary, edge_index, binary_redistributed):
        return self.propagate(edge_index, size=(unary.size(0), unary.size(0)), x=unary, binary=binary_redistributed)

    def message(self, x_j, binary):

        N, C = x_j.shape
        x_j = x_j*0
        messages = torch.logsumexp(x_j.view(N, -1, 1)+binary, axis=1)
        return messages

    def update(self, aggr_out,x):

        log_z = torch.logsumexp((x + aggr_out),axis=1)
        # normalizer for every piece, i.e. for every node.
        return log_z


class BPLeafToRoot_center_edge_rezero(MessagePassing):
    def __init__(self):
        # super(BPLeafToRoot_center_edge_rezero, self).__init__(aggr='add', flow="source_to_target")
        super(BPLeafToRoot_center_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, unary, edge_index, binary_redistributed, rezero):
        return self.propagate(edge_index, size=(unary.size(0), unary.size(0)), x=unary, binary=binary_redistributed, rezero=rezero)

    def message(self, x_j, binary, rezero):

        N, C = x_j.shape
        x_j = x_j*0
        messages = torch.logsumexp(rezero.view(-1, 1, 1) * (x_j.view(N, -1, 1) + binary), axis=1)
        return messages

    def update(self, aggr_out,x):

        log_z = torch.logsumexp((x + aggr_out),axis=1)
        # normalizer for every piece, i.e. for every node.
        return log_z


class PWLoss(MessagePassing):
    # average redistribution loss
    def __init__(self):
        super(PWLoss, self).__init__(aggr='add', flow="target_to_source")
        # super(PWLoss, self).__init__(aggr='add', flow="source_to_target")

    def forward(self, x_redistributed, edge_index, binary_redistributed, log_z_redistributed, q):
        # forward params needed to construct message and update

        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index)

    def message(self, x_j, edge_index_params, binary, q):

        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(x_j*q_j,axis=1) + torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q):
        # return the loss for every piece, and final loss need the summation
        result = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        return result

class PWLoss_edge_rezero(MessagePassing):
    # average redistribution version + edge_rezero
    def __init__(self):
        # super(PWLoss_edge_rezero, self).__init__(aggr='add', flow="source_to_target")
        super(PWLoss_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed, log_z_redistributed, q, rezero):
        # forward params needed to construct message and update
        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, edge_index_params, binary, q, rezero):

        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(x_j*q_j,axis=1) + rezero * torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q):
        # return the loss for every piece, and final loss need the summation
        result = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        return result


class PWLoss_redis2(MessagePassing):
    # center redistribution
    def __init__(self):
        super(PWLoss_redis2, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, unary, edge_index, binary_redistributed, log_z_redistributed, q):
        # forward params needed to construct message and update

        return self.propagate(edge_index, size=(unary.size(0), unary.size(0)), x=unary, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index)

    def message(self, x_j, edge_index_params, binary, q):
        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q):
        # return the loss for every piece, and final loss need the summation
        result = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        return result

class PWLoss_redis2_edge_rezero(MessagePassing):
    # center redistribution + edge rezero
    def __init__(self):
        super(PWLoss_redis2_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, unary, edge_index, binary_redistributed, log_z_redistributed, q, rezero):
        # forward params needed to construct message and update

        return self.propagate(edge_index, size=(unary.size(0), unary.size(0)), x=unary, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, edge_index_params, binary, q, rezero):
        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = rezero* torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q):
        # return the loss for every piece, and final loss need the summation
        result = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        return result

class PWLoss_average_mixture(MessagePassing):
    # average redistribution loss
    def __init__(self):
        super(PWLoss_average_mixture, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, x_redistributed, edge_index, binary_redistributed, log_z_redistributed, q, unary, rezero):

        return self.propagate(edge_index, size=(x_redistributed.size(0), x_redistributed.size(0)), x=x_redistributed, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index, unary=unary, rezero=rezero)

    def message(self, x_j, edge_index_params, binary, q):

        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(x_j*q_j,axis=1) + torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q, unary, rezero):
        piecewise_loss = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z

        logit = torch.log_softmax(unary, dim=-1)
        entropy_loss = torch.sum(q*logit, dim = -1)
        gamma = torch.nn.functional.sigmoid(rezero)
        result= (1-gamma)*entropy_loss + gamma * piecewise_loss
        return result


class PWLoss_center_mixture(MessagePassing):
    # center redistribution
    def __init__(self):
        super(PWLoss_center_mixture, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, unary, edge_index, binary_redistributed, log_z_redistributed, q, rezero):
        # forward params needed to construct message and update

        return self.propagate(edge_index, size=(unary.size(0), unary.size(0)), x=unary, binary=binary_redistributed, log_z=log_z_redistributed, q=q,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, edge_index_params, binary, q):

        i,j = edge_index_params
        q_j,q_i =q[j],q[i] # q_j and q_i are of shape E*C
        messages = torch.sum(torch.mm(q_i,binary)*q_j, axis=1)
        return messages.view(-1,1)

    def update(self, aggr_out, x, log_z, q, rezero):
        # return the loss for every piece, and final loss need the summation
        piecewise_loss = torch.sum(x*q,axis =1) + aggr_out.squeeze() - log_z
        logit = torch.log_softmax(x, dim=-1)
        entropy_loss = torch.sum(q * logit, dim=-1)
        gamma = torch.nn.functional.sigmoid(rezero)
        result = (1 - gamma) * entropy_loss + gamma * piecewise_loss
        return result

class MFUpdate(MessagePassing):
    # update q
    def __init__(self):
        super(MFUpdate, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, q, edge_index, binary, unary):
        # here x represent the q
        return self.propagate(edge_index, size=(q.size(0), q.size(0)), x=q, binary=binary, unary=unary,
                              edge_index_params=edge_index)

    def message(self, x_j, binary):
        messages = torch.mm(x_j, binary)
        return messages

    def update(self, aggr_out, unary):
        # return the loss for every piece, and final loss need the summation
        return F.softmax(unary + aggr_out, dim=1)

class MFUpdate_edge_rezero(MessagePassing):
    # edge rezero inference
    # update q
    def __init__(self):
        super(MFUpdate_edge_rezero, self).__init__(aggr='add', flow="target_to_source")

    def forward(self, q, edge_index, binary, unary, rezero):
        # here x represent the q
        return self.propagate(edge_index, size=(q.size(0), q.size(0)), x=q, binary=binary, unary=unary,
                              edge_index_params=edge_index, rezero=rezero)

    def message(self, x_j, binary, rezero):
        messages = rezero.view(-1,1) * torch.mm(x_j, binary)
        return messages

    def update(self, aggr_out, unary):
        # return the loss for every piece, and final loss need the summation
        return F.softmax(unary + aggr_out, dim=1)


class PWGNN(nn.Module):
    def __init__(self, gnnbackbone, configs):
        super(PWGNN, self).__init__()

        # # init backbone
        self.gnnbackbone = gnnbackbone
        num_classes = gnnbackbone.num_classes
        self.configs = configs
        self.inf = MFUpdate()

        if configs['redistribution'] =="center": self.up_BP = BPLeafToRoot_center()
        elif configs['redistribution'] == "average": self.up_BP = BPLeafToRoot()
        elif configs['redistribution'] == "average_edge_rezero":
            self.up_BP = BPLeafToRoot_edge_rezero()
            self.inf = MFUpdate_edge_rezero()
        elif configs['redistribution'] == "center_edge_rezero":
            self.up_BP = BPLeafToRoot_center_edge_rezero()
            self.inf = MFUpdate_edge_rezero()

        self.Binary = torch.nn.Parameter((torch.randn(num_classes, num_classes) + torch.eye(num_classes))/num_classes,requires_grad=True)  # identity between every piece

        if self.configs['rezero']:
            self.rezero_coefficients = torch.nn.Parameter(torch.zeros(self.configs['rezero_size']), requires_grad= self.configs['rezero_require_grad'])

    def forward(self, data, deg):
        x, edge_index = data.x, data.edge_index
        Unary = self.gnnbackbone(data)
        Binary = (self.Binary + self.Binary.T)/2
        if self.configs['rezero']:
            if self.configs['rezero_change'] == 'auto':
                rezero = self.rezero_coefficients
            elif self.configs['rezero_change'] == 'linear':
                rezero = self.configs['rezero_value'] + 0.01
                self.configs['rezero_value'] = rezero
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                rezero = rezero.to(device)
            elif self.configs['rezero_change'] == 'pairwise':
                rezero = self.rezero_coefficients
                Binary = rezero * Binary



        if self.configs['redistribution']=='average':
            log_z_reditributed = self.up_BP(Unary/(deg), edge_index, Binary/2)
            return Unary, Unary / (deg), Binary, Binary / 2, log_z_reditributed

        elif self.configs['redistribution'] =='average_edge_rezero':
            log_z_reditributed = self.up_BP(Unary/(deg), edge_index, Binary/2, rezero)
            return Unary, Unary/(deg), Binary, Binary/2, log_z_reditributed, rezero

        elif self.configs['redistribution']=='center':
            log_z_reditributed = self.up_BP(Unary, edge_index, Binary/2)
            return Unary, Binary, log_z_reditributed

        elif self.configs['redistribution'] == 'center_edge_rezero':
            log_z_reditributed = self.up_BP(Unary, edge_index, Binary/2, rezero)
            return  Unary,  Binary, log_z_reditributed, rezero

        else: assert 'no such redistribution version'

    def inference(self,data, deg):
        self.eval()
        with torch.no_grad():
            # need to check whether B need to force to be simatry
            if self.configs['redistribution'] == 'center':
                U, B, _ = self.forward(data, deg)
                q = self.inf(data.y, data.edge_index, B, U)
            elif self.configs['redistribution'] == 'center_edge_rezero':
                U, B, Z, R = self.forward(data, deg)
                q = self.inf(data.y, data.edge_index, B, U, R)
            elif self.configs['redistribution'] == 'average_edge_rezero':
                U, U_redistributed, B, B_redistributed, Z, R = self.forward(data, deg)
                q = self.inf(data.y, data.edge_index, B, U, R)
            else:
                U,U_redistributed, B, B_redistributed, _ = self.forward(data,deg) # we don't want log z since it is only sorrugate approximation.
                q = self.inf(data.y, data.edge_index, B, U)

        return q