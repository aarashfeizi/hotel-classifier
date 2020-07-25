from models.linear_siamese import *
from models.resnet import *


class TopModel(nn.Module):

    def __init__(self, ft_net, sm_net):
        super(TopModel, self).__init__()
        self.ft_net = ft_net
        self.sm_net = sm_net

        # print('FEATURE NET')
        # print(self.ft_net)
        # print('SIAMESE NET')
        # print(self.sm_net)

    def forward(self, x1, x2, single=False, feats=False):
        # print('model input:', x1[-1].size())

        x1_f, x1_l = self.ft_net(x1, is_feat=True)
        out1, out2 = None, None

        if single and feats:
            raise Exception('Both single and feats cannot be True')

        if not single:
            x2_f, x2_l = self.ft_net(x2, is_feat=True)
            if feats:
                output, out1, out2 = self.sm_net(x1_f[-1], x2_f[-1], feats=feats)
            else:
                output = self.sm_net(x1_f[-1], x2_f[-1], feats=False)
        else:
            output = self.sm_net(x1_f[-1], None, single)  # single is true

        # print('features:', x2_f[-1].size())
        # print('output:', output.size())

        if feats:
            return output, out1, out2
        else:
            return output


def top_module(args, trained_feat_net=None, trained_sm_net=None, num_classes=1):
    if trained_sm_net is None:
        sm_net = LiSiamese(args)
    else:
        sm_net = trained_sm_net

    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
    }

    if trained_feat_net is None:
        print('Using pretrained model')
        ft_net = model_dict[args.feat_extractor](pretrained=True, num_classes=num_classes)
    else:
        print('Using recently trained model')
        ft_net = trained_feat_net

    if not args.freeze_ext:
        print("Unfreezing Resnet")
        for param in ft_net.parameters():
            param.requires_grad = True
    else:
        print("Freezing Resnet")
        for param in ft_net.parameters():
            param.requires_grad = False

    return TopModel(ft_net=ft_net, sm_net=sm_net)
