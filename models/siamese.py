import torch
import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self, args):
        super(Siamese, self).__init__()
        if args.dataset_name != 'hotels':
            if args.dataset_name == 'cub':  # 84 * 84
                print('CUB MODE')
                input_channel = 3

            elif args.dataset_name == 'omniglot':
                print('OMNIGLOT MODE')
                input_channel = 1
            else:
                raise Exception('Dataset not supported')

            self.conv = nn.Sequential(
                nn.Conv2d(input_channel, 64, args.first_conv_filter),  # 64@491*491
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64@246*246

                nn.Conv2d(64, 128, args.second_conv_filter),
                nn.ReLU(),  # 128@240*240
                nn.MaxPool2d(2),  # 128@120*120

                nn.Conv2d(128, 128, args.third_conv_filter),  # 128@117*117
                nn.ReLU(),
                nn.MaxPool2d(2),  # 128@59*59

                nn.Conv2d(128, 256, args.fourth_conv_filter),
                nn.ReLU(),  # 256@56*56
                # nn.MaxPool2d(2),  # 256@28*28

                # nn.Conv2d(256, 512, 4),
                # nn.ReLU(),  # 512@25*25
                # nn.MaxPool2d(2),  # 512@13*13
                #
                # nn.Conv2d(512, 512, 4),
                # nn.ReLU(),  # 512@10*10
                # nn.MaxPool2d(2),  # 512@5*5
            )
            self.linear = nn.Sequential(nn.Linear(args.conv_output, args.last_layer),
                                        nn.Sigmoid())  # conv_output: cub: 2304 omniglot: 9216
            self.out = nn.Linear(args.last_layer, 1)

        elif args.dataset_name == 'hotels':
            print('HOTELS MODE')

            input_channel = 3
            self.conv = nn.Sequential(
                nn.Conv2d(input_channel, 64, args.first_conv_filter),  # 64@491*491
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64@246*246

                nn.Conv2d(64, 128, args.second_conv_filter),
                nn.ReLU(),  # 128@240*240
                nn.MaxPool2d(2),  # 128@120*120

                nn.Conv2d(128, 128, args.third_conv_filter),  # 128@117*117
                nn.ReLU(),
                nn.MaxPool2d(2),  # 128@59*59

                nn.Conv2d(128, 256, args.fourth_conv_filter),
                nn.ReLU(),  # 256@56*56
                nn.MaxPool2d(2),  # 256@28*28

                nn.Conv2d(256, 256, args.fifth_conv_filter),
                nn.ReLU(),  # 256@56*56
                nn.MaxPool2d(2),  # 256@28*28
                #
                # nn.Conv2d(256, 256, args.sixth_conv_filter),
                # nn.ReLU(),  # 256@56*56
                # nn.MaxPool2d(2),  # 256@28*28

                # nn.Conv2d(256, 512, 4),
                # nn.ReLU(),  # 512@25*25
                # nn.MaxPool2d(2),  # 512@13*13
                #
                # nn.Conv2d(512, 512, 4),
                # nn.ReLU(),  # 512@10*10
                # nn.MaxPool2d(2),  # 512@5*5
            )
            self.linear = nn.Sequential(nn.Linear(args.conv_output, args.last_layer),
                                        nn.Sigmoid())  # conv_output: cub: 2304 omniglot: 9216
            self.out = nn.Linear(args.last_layer, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out
