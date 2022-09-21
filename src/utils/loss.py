import torch.nn as nn
import torch.nn.functional as F
import torch


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class HardNegativeMining(nn.Module):
    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, targets):
        # inputs should be B, H, W
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        P = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * P))
        loss = tk[0].mean()
        return loss


def weight_train_loss(losses):
    """Function that weights losses over train round, taking only last loss for each user"""
    fin_losses = {}
    c = list(losses.keys())[0]
    loss_names = list(losses[c]['loss'].keys())
    for l_name in loss_names:
        tot_loss = 0
        weights = 0
        for _, d in losses.items():
            tot_loss += d['loss'][l_name][-1] * d['num_samples']
            weights += d['num_samples']
        fin_losses[l_name] = tot_loss / weights
    return fin_losses


def weight_test_loss(losses):
    tot_loss = 0
    weights = 0
    for k, v in losses.items():
        tot_loss = tot_loss + v['loss'] * v['num_samples']
        weights = weights + v['num_samples']
    return tot_loss / weights
