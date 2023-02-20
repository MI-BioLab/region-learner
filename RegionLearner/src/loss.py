import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

def compute_weights(y, total_regions, beta=0.999):
    """Weights the different classes in order to deal with imbalanced classification. 
    This weighting mechanism is described in https://arxiv.org/pdf/1901.05555.pdf.
    
    The effective number Eₙ = (1-βⁿ) where n is the number of samples per each class.
    The weights w = (1-β) / Eₙ are then normalized and multiplied for the total number of classes (regions).

    Args:
        y (list(int)): list of the regions groundtruth.
        total_regions (int): total number of regions.
        beta (float, optional): parameter to tune the weighting algorithm. Defaults to 0.999.

    Returns:
        ndarray: the weights for the classes.
    """    
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
    """Focal loss with weighting factor α.
    
                𝑖=𝑛
                ⎲
    FocalLoss = ⎳αᵢ(𝑖-𝑝ᵢ)ᵞlog(𝑝ᵢ)   
                𝑖=1
    """
    
    def __init__(self, gamma, alpha=None, reduction="none"):
        """Constructor for the FocalLoss class.

        Args:
            gamma (float): the gamma parameter of the focal loss.
            alpha (ndarray, optional): the weights for the classes. Defaults to None.
            reduction (str, optional): the reduction to apply to the focal loss. It can be "mean", "sum" or "none". Defaults to "none".
        """
        
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        cross_entropy = F.cross_entropy(input, target, reduction="none", weight=self.alpha) # weighted cross-entropy, α is applied here.
        pt = torch.exp(-cross_entropy)
        focal_loss = (1 - pt) ** self.gamma * cross_entropy
        return torch.mean(focal_loss) if self.reduction == "mean" else torch.sum(focal_loss) if self.reduction == "sum" else focal_loss