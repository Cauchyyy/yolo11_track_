# reid/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        """
        embeddings: (N, D)
        labels: (N,)
        batch assumed to be PK
        compute pairwise distance
        for each anchor: hardest positive (max dist among same id), hardest negative (min dist among other ids)
        """
        n = embeddings.size(0)
        # pairwise distance
        dist = torch.cdist(embeddings, embeddings, p=2)  # (n,n)
        labels = labels.view(-1,1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()
        # For each anchor
        dist_pos = dist * mask_pos + (-1e12) * (1 - mask_pos)  # keep positives
        hardest_pos, _ = dist_pos.max(dim=1)
        dist_neg = dist * mask_neg + (1e12) * (1 - mask_neg)
        hardest_neg, _ = dist_neg.min(dim=1)
        y = torch.ones_like(hardest_pos)
        loss = F.relu(hardest_pos - hardest_neg + self.margin).mean()
        return loss
