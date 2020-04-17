import torch
import torch.nn as nn


class LiSiamese(nn.Module):

    def __init__(self, args):
        super(LiSiamese, self).__init__()

        self.extra_layer = args.extra_layer
        # self.layer = nn.Sequential(nn.Linear(25088, 512))
        if self.extra_layer:
            self.layer1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        if self.extra_layer:
            x = self.layer1(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out
