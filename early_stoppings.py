import numpy as np
import torch

class EarlyStoppingCriterion(object):
    def __init__(self, patience, path, init_best_val_acc):
        self.patience = patience
        self.path = path
        self.best_val_acc = init_best_val_acc
    def should_stop(self, epoch, val_loss, val_accuracy, model):
        raise NotImplementedError

    def after_stopping_ops(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError



class GCNEarlyStoppingCriterion(EarlyStoppingCriterion):

    def __init__(self, patience, path, init_best_val_acc):
        super().__init__(patience, path, init_best_val_acc)
        self.val_acces = []
        self.best_epoch = 0
        self.count = 0

    def should_stop(self, epoch, val_accuracy, model, optimizer):

        self.val_acces.append(val_accuracy)
        if val_accuracy > self.best_val_acc:
            # store the model, optimizer and q of the highest accuracy
            self.best_epoch = epoch
            self.best_val_acc = val_accuracy
            torch.save(model.state_dict(), self.path)
            torch.save(optimizer.state_dict(), self.path + '.opt')
            self.count = 0
        else: self.count = self.count + 1
        return (self.count >= self.patience and self.val_acces[-1]<= np.mean(self.val_acces[-(self.patience + 1):-1])) or self.count>100

    def after_stopping_ops(self, model, optimizer):
        # after stop, reload the corresponding best model, optimizer and q
        return model.load_state_dict(torch.load(self.path)), optimizer.load_state_dict(torch.load(self.path + '.opt'))

    def reset(self):
        self.val_acces = []

class PWEMEarlyStoppingCriterion(EarlyStoppingCriterion):

    def __init__(self, patience, path, init_best_val_acc):
        super().__init__(patience, path, init_best_val_acc)
        self.val_acces = []
        self.best_epoch = 0
        self.count = 0
        print(self.patience)
    def should_stop(self, epoch, val_accuracy, model, optimizer, q):
        # if evaluate accuracy smaller than mean of previous patience times' evaluate accuracy and the count is bigger than the patience, return true

        self.val_acces.append(val_accuracy)
        if val_accuracy > self.best_val_acc:
            # store the model, optimizer and q of the highest accuracy
            self.best_epoch = epoch
            self.best_val_acc = val_accuracy
            torch.save(model.state_dict(), self.path)
            torch.save(optimizer.state_dict(), self.path + '.opt')
            torch.save(q, self.path + 'q.tensor')
            print(self.path + 'q.tensor')
            self.count = 0
        else: self.count = self.count + 1
        return (self.count >= self.patience and self.val_acces[-1]<= torch.mean(torch.tensor(self.val_acces[-(self.patience + 1):-1]))) or self.count>100

    def after_stopping_ops(self):
        # we only need to load q for test
        return torch.load(self.path + 'q.tensor')

    def reset(self):
        self.val_acces = []

class PWEMEarlyStoppingCriterionForM(EarlyStoppingCriterion):

    def __init__(self, patience, path, init_best_val_acc):
        super().__init__(patience, path, init_best_val_acc)
        self.val_acces = []
        self.best_epoch = 0
        self.count = 0

    def should_stop(self, epoch, val_accuracy, model, optimizer, q):

        self.val_acces.append(val_accuracy)
        if val_accuracy > self.best_val_acc:
            # store the model, optimizer and q of the highest accuracy
            self.best_epoch = epoch
            self.best_val_acc = val_accuracy
            torch.save(model.state_dict(), self.path)
            torch.save(optimizer.state_dict(), self.path + '.opt')
            print(self.path)
            self.count = 0
        else: self.count = self.count + 1
        return (self.count >= self.patience and self.val_acces[-1] <= torch.mean(
            torch.tensor(self.val_acces[-(self.patience + 1):-1]))) or self.count > 50



    def after_stopping_ops(self, model, optimizer):
        # we only need to load q for test
        self.count = 0
        model.load_state_dict(torch.load(self.path))
        optimizer.load_state_dict(torch.load(self.path + '.opt'))

        return torch.load(self.path + 'q.tensor')

    def reset(self):
        self.val_acces = []

class PWEMEarlyStoppingCriterionForE(EarlyStoppingCriterion):

    def __init__(self, patience, path, init_best_val_acc):
        super().__init__(patience, path, init_best_val_acc)
        self.val_acces = []
        self.best_epoch = 0
        self.count = 0

    def should_stop(self, epoch, val_accuracy, q):
        # if evaluate accuracy smaller than mean of previous patience times' evaluate accuracy and the count is bigger than the patience, return true

        self.val_acces.append(val_accuracy)
        if val_accuracy > self.best_val_acc:
            # store the model, optimizer and q of the highest accuracy
            self.best_epoch = epoch
            self.best_val_acc = val_accuracy
            torch.save(q, self.path + 'q.tensor')
            print(self.path)
            self.count = 0
        else: self.count = self.count + 1
        return (self.count >= self.patience and self.val_acces[-1]<= np.mean(self.val_acces[-(self.patience + 1):-1])) or self.count>50


    def after_stopping_ops(self):
        # we only need to load q for test
        self.count = 0
        return torch.load(self.path + 'q.tensor')

    def reset(self):
        self.val_acces = []
