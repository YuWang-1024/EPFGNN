import torch
import torch.nn as nn
from metrics import accuracy, accuracy_soft
from early_stoppings import GCNEarlyStoppingCriterion, PWEMEarlyStoppingCriterion, PWEMEarlyStoppingCriterionForM,PWEMEarlyStoppingCriterionForE
from models.PWEM import PWLoss, PWLoss_redis2, PWLoss_redis2_edge_rezero, PWLoss_edge_rezero, PWLoss_average_mixture, PWLoss_center_mixture
from models.PWEM_J import PWLoss_J
import json
from torch.utils.tensorboard import SummaryWriter

def get_optimizer(optimizer_name, parameters, lr, weight_decay):
    if optimizer_name == "adam": return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def get_criterion(criterion_name):
    if criterion_name == "crossEntropy": return nn.CrossEntropyLoss()
    if criterion_name == "pwLoss": return PWLoss()
    if criterion_name == "pwLoss_J": return PWLoss_J()
    if criterion_name == "PWLoss_redis2": return PWLoss_redis2()
    if criterion_name == "PWLoss_redist2_edge_rezero": return PWLoss_redis2_edge_rezero()
    if criterion_name == "PWLoss_average_edge_rezero": return PWLoss_edge_rezero()
    if criterion_name == "PWLoss_average_mixture": return PWLoss_average_mixture()
    if criterion_name == "PWLoss_center_mixture": return PWLoss_center_mixture()


def get_metric(metric_name, soft):
    if metric_name == 'accuracy':
        if soft: return accuracy_soft
        else: return accuracy

def get_early_stopping_criterion(stopping_name,patience, path, init_best_val_acc):
    if stopping_name == 'gcnStoppingCriterion': return GCNEarlyStoppingCriterion(patience=patience, path=path, init_best_val_acc=init_best_val_acc)
    elif stopping_name == 'pwemStoppingCriterion': return PWEMEarlyStoppingCriterion(patience=patience, path=path, init_best_val_acc=init_best_val_acc)
    elif stopping_name == 'pwemStoppingCriterionForEM': return PWEMEarlyStoppingCriterionForM(patience=patience, path=path, init_best_val_acc=init_best_val_acc)

class Trainer(object):
    torch.autograd.set_detect_anomaly(True)
    def __init__(self, model, configs, path):

        optimizezr_name = configs['optimizer_name']
        lr = configs['learning_rate']
        weight_decay = configs['weight_decay']
        parameters = [p for p in model.parameters() if p.requires_grad]
        stopping_name = configs['stopping_name']
        patience = configs['patience']
        self.path = path
        self.optimizer = get_optimizer(optimizezr_name, parameters, lr, weight_decay)
        self.model = model
        self.criterion = get_criterion(configs['criterion_name'])
        self.configs = configs
        self.metric = get_metric(configs['metric'],configs['soft'])
        self.early_stopping_criterion = get_early_stopping_criterion(stopping_name, patience, path, configs['init_best_val_acc'])
        
        self.records = {
            "eval_acc": torch.tensor([]),
            "test_acc": torch.tensor([]),
            "train_loss": torch.tensor([]),
            "eval_loss": torch.tensor([]),
            "configs": configs}

        if configs['cuda']: self.criterion.cuda()
        self.writer = SummaryWriter()




    def run(self, data, mask, target):
        if self.configs['soft']:
            for i in range(self.configs['num_run']):
                loss = self.update_soft(data, mask)
                val_loss = self.evaluate_loss_soft(data, target)
                val_acc = self.evaluate_acc(data, target, self.configs['verbose'])
                test_acc = self.test_acc(data, target, self.configs['verbose'])
                if self.configs['verbose']:print(f"train_loss:{loss}, eval_acc:{val_acc}, test_acc:{test_acc}")
                self.records['eval_acc'] = torch.cat((self.records['eval_acc'],torch.tensor([val_acc])),dim=0)
                self.records['test_acc'] = torch.cat((self.records['test_acc'],torch.tensor([test_acc])),dim=0)
                self.records['train_loss'] = torch.cat((self.records['train_loss'],torch.tensor([loss])),dim=0)
                self.records['eval_loss'] = torch.cat((self.records['eval_loss'],torch.tensor([val_loss])),dim=0)

                if self.early_stopping_criterion.should_stop(epoch=i, val_accuracy=val_acc, model=self.model, optimizer=self.optimizer):
                    return self.early_stopping_criterion.after_stopping_ops(self.model, self.optimizer)

        else:
            for i in range(self.configs['num_run']):
                loss = self.update(data, mask)
                val_loss = self.evaluate_loss(data, target)
                val_acc = self.evaluate_acc(data, target, self.configs['verbose'])
                test_acc = self.test_acc(data, target, self.configs['verbose'])
                if self.configs['verbose']:print(f"train_loss:{loss}, eval_acc:{val_acc}, test_acc:{test_acc}")
                self.records['eval_acc'] = torch.cat((self.records['eval_acc'], torch.tensor([val_acc])), dim=0)
                self.records['test_acc'] = torch.cat((self.records['test_acc'], torch.tensor([test_acc])), dim=0)
                self.records['train_loss'] = torch.cat((self.records['train_loss'], torch.tensor([loss])), dim=0)
                self.records['eval_loss'] = torch.cat((self.records['eval_loss'], torch.tensor([val_loss])), dim=0)

                if self.early_stopping_criterion.should_stop(epoch=i, val_accuracy=val_acc, model=self.model, optimizer=self.optimizer):
                    return self.early_stopping_criterion.after_stopping_ops(self.model, self.optimizer)



    def update(self,data, mask):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)
        loss = self.criterion(out[mask], data.y[mask])
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def update_soft(self, data, mask):

        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)
        logit = torch.log_softmax(out, dim = -1)
        loss = -torch.mean(torch.sum(data.y[mask]*logit[mask], dim = -1))
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            probs = torch.softmax(out, dim=-1)
        return probs

    def evaluate_acc(self,data, target, verbose):
        self.model.eval()
        with torch.no_grad():
            result = self.metric(self.model(data), target, data.val_mask, verbose=verbose)
        return result

    def evaluate_loss_soft(self, data, target):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            logit = torch.log_softmax(out, dim=-1)
            loss = -torch.mean(torch.sum(target[data.val_mask] * logit[data.val_mask], dim=-1))
        return loss.item()

    def evaluate_loss(self, data, target):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            loss = self.criterion(out[data.val_mask], target[data.val_mask])
        return loss.item()


    def test_acc(self, data, target, verbose):
        self.model.eval()
        with torch.no_grad():
            result = self.metric(self.model(data), target, data.test_mask, verbose=verbose)
        return result


    def reset(self): return 0

    def save(self, path_for_results):
        # should append current records to the end of file
        for k, v in self.records.items():
            if isinstance(v, torch.Tensor):
                self.records[k] = v.cpu().tolist()
        with open(path_for_results, 'w') as f:
            json.dump(self.records, f)


    def give_best_test(self):
        # similar as gmnn original code
        best_eval_acc =0
        best_epoch = 0
        for i, acc in enumerate(self.records['eval_acc']):
            if acc>best_eval_acc:
                best_eval_acc=acc
                best_epoch=i
        self.records.update({'best_acc_gmnn_criterion':self.records['test_acc'][best_epoch]})
        return self.records['test_acc'][best_epoch]