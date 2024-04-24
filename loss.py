

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalityLoss(nn.Module):
    def __init__(self, gamma=1):
        super(OrthogonalityLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum())

        loss = torch.abs(neg_pairs_mean)

        return loss