import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anch, pos, neg):
        pos_dist = torch.dist(anch, pos)
        neg_dist = torch.dist(anch, neg)

        dist = pos_dist - neg_dist + self.margin

        loss = torch.max(
            torch.tensor([0, dist])
        )

        return loss
