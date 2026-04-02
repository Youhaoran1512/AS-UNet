import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torch.autograd import Variable
import numpy as np

__all__ = ['BCEDiceLoss','FocalLoss', 'PolyCEDiceLoss','PolyGHMDiceLoss','PolyCrossEntroPy']

'''
DICE:衡量两个集合重叠程度的指标，DICE=预测目标像素和真实目标像素重叠在一起的数值✖2，再除以预测目标像素和真实目标像素的总和
DICELoss = 1-DICE,用来最大化预测与真实目标像素的重叠程度，它对样本分布不感兴趣，所关心的只有目标区域(即对背景不敏感)
但是当目标预测和真实目标无交集的时候，会导致当前损失直接为零，但实际上却并非如此，所以会使梯度消失
使用中通常引入一个极小的常量(smooth)，来保证损失不会直接为1而导致梯度消失

单独使用DICELoss的时候，在进行小目标预测的时候极为不利，小目标一旦有一部分预测错误会导致DICE系数大幅度变化，不利于梯度的更新
'''
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # 使用交叉熵损失函数计算基础损失
        #print(f"ce_loss shape: {ce_loss.shape}")
        alpha_t = targets*self.alpha + (1-targets)*(1-self.alpha)
        #print(f"alpha_t shape = {alpha_t.shape}")
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        #print(f"pt shape: {pt.shape}")
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        #print(f"focal loss shape: {focal_loss.shape}")
        return focal_loss.mean()



class PolyCrossEntroPy(nn.Module):
    def __init__(self, epsilon: float=2.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        p = torch.sigmoid(input)
        pt = target * p + (1 - p) * (1 - target)
        CE = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        loss = (CE + self.epsilon*(1-pt)).mean()
        return loss

class PolyCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.PolyCE = PolyCrossEntroPy(epsilon=2.0)
    def forward(self, input, target):
        PolyLoss = self.PolyCE(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5*PolyLoss + dice
    

class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        # the target should be binary class label
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

class PolyGHMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.GHMC = GHMC(bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0)
        self.epsilon = 2.0
    def forward(self, input, target):
        p = torch.sigmoid(input)
        pt = target * p + (1 - p) * (1 - target)
        label_weight = torch.ones_like(target)
        GHMloss = self.GHMC(input, target, label_weight)
        polyloss = (GHMloss + self.epsilon*(1-pt)).mean()
        return polyloss

class PolyGHMDiceLoss(nn.Module):                 #Poly-harmonized Gradient Dice Loss
    def __init__(self):
        super().__init__()
        self.PolyGHMLoss = PolyGHMLoss()
    def forward(self, input, target):

        bce = F.binary_cross_entropy_with_logits(input, target)

        Ploss = self.PolyGHMLoss(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5*Ploss + bce + dice
