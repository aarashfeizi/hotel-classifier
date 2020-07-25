import torch
import torch.nn as nn


class LiSiamese(nn.Module):

    def __init__(self, args):
        super(LiSiamese, self).__init__()

        if args.feat_extractor == 'resnet50':
            self.input_shape = 2048
        else:
            self.input_shape = 512

        self.extra_layer = args.extra_layer
        # self.layer = nn.Sequential(nn.Linear(25088, 512))
        if self.extra_layer > 0:
            layers = []
            for i in range(self.extra_layer):
                layers.append(nn.Linear(self.input_shape, self.input_shape))
                layers.append(nn.ReLU())
                if args.normalize:
                    layers.append(nn.BatchNorm1d(self.input_shape))

            self.layer1 = nn.Sequential(*layers)

            # self.layer2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(self.input_shape, 1))  # no sigmoid!!!!

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        if self.extra_layer > 0:
            x = self.layer1(x)
        return x

    def forward(self, x1, x2, single=False, feats=False):
        out1 = self.forward_one(x1)
        if single:
            return out1

        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        if feats:
            return out, out1, out2
        else:
            return out
