import torch
from .trainer import Trainer


class Trainer_GMNN(Trainer):
    def __init__(self, model, config):
        super(Trainer_GMNN, self).__init__()


    def update(self,data):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_soft(self, data):

        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)
        logit = torch.log_softmax(out, dim = -1)

        loss = -torch.mean(torch.sum(data.y[data.train_mask]*logit[data.train_mask], dim = -1))
        loss.backward()
        self.optimizer.step()

        return loss


    def evaluate(self,data, configs):
        self.model.eval()
        return self.metric(self.model(data), data.gold, data.val_mask, verbose=configs['verbose'])

