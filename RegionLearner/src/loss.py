import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

def compute_weights(y, total_regions, beta=0.999):
    samples_per_class = []
    for i in range(total_regions):
        samples_per_class.append(0)    
    for i in y:
        samples_per_class[i] += 1
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * total_regions
    return weights

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction="none"):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, input, target):
        cross_entropy = F.cross_entropy(input, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-cross_entropy)
        focal_loss = (1 - pt) ** self.gamma * cross_entropy
        return torch.mean(focal_loss) if self.reduction == "mean" else torch.sum(focal_loss) if self.reduction == "sum" else focal_loss