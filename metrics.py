import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def precision_score(output, target, smooth=1e-5):
    """
    Compute Precision = TP / (TP + FP)
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred = output > 0.5
    true = target > 0.5
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision


def recall_score(output, target, smooth=1e-5):
    """
    Compute Recall = TP / (TP + FN)
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred = output > 0.5
    true = target > 0.5
    tp = (pred & true).sum()
    fn = (~pred & true).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def f1_score(output, target, smooth=1e-5):
    """
    Compute F1 Score = 2 * Precision * Recall / (Precision + Recall)
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred = output > 0.5
    true = target > 0.5
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    return f1


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
