
import torch

def accuracy(input, target, idx, verbose=True):
    _,pred = input.max(dim=1)
    correct = float(pred[idx].eq(target[idx]).sum().item())
    acc = correct / idx.sum().item()
    if verbose: print('Accuracy: {:.4f}'.format(acc))
    return acc


def accuracy_soft(input, target, idx, verbose = True):
    _, pred = input.max(dim=1)
    pred_one_hot = torch.nn.functional.one_hot(pred, target.shape[-1]).float()
    correct = float((pred_one_hot[idx]*target[idx]).sum().item())
    acc = correct/idx.sum().item()
    if verbose: print(f"Accuracy: {acc}")
    return acc

def f1(): return 0