import argparse
import os
import pickle
import time
from collections import deque

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torchvision.transforms import transforms
from tqdm import tqdm

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class TransformLoader:
    def __init__(self, image_size, rotate=0,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.rotate = rotate

    def parse_transform(self, transform_type):
        # if transform_type == 'ImageJitter':
        #     method = add_transforms.ImageJitter(self.jitter_param)
        #     return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif not aug and self.rotate == 0:
            transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']
        elif not aug and self.rotate != 0:
            transform_list = ['Scale', 'RandomRotation', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class Metric:

    def __init__(self):
        self.rights = 0
        self.wrongs = 0

    def update_acc(self, output, label):
        pred = (output >= 0)
        print(output.size())
        print(label.size())
        batch_rights = sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy()[0]

        self.rights += batch_rights
        self.wrongs += (label.shape[0] - batch_rights)

    def get_acc(self):
        # print('rights: ', self.rights)
        # print('wrongs: ', self.wrongs)
        return ((self.rights) / (self.rights + self.wrongs)) * 100

    def get_right_wrong(self):
        return {'right': self.rights, 'wrong': self.wrongs}

    def reset_acc(self):
        self.rights = 0
        self.wrongs = 0


# '../../dataset/omniglot/python/images_background'
# '../../dataset/omniglot/python/images_evaluation'
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"

    parser.add_argument('-dsn', '--dataset_name', default='omniglot', choices=['omniglot', 'cub', 'hotels'])
    parser.add_argument('-dsp', '--dataset_path', default='CUB/')
    parser.add_argument('-sdp', '--subdir_path', default='images/')
    parser.add_argument('-trp', '--train_path', default='./omniglot/python/images_background')
    parser.add_argument('-tsp', '--test_path', default='./omniglot/python/images_evaluation')
    parser.add_argument('-is', '--image_size', default=0, type=int, help="Image Size")
    parser.add_argument('-sp', '--save_path', default='models/', help="path to store model")
    parser.add_argument('-a', '--aug', default=False, action='store_true')
    parser.add_argument('-r', '--rotate', default=0.0, type=float)
    parser.add_argument('-mn', '--model_name', default='')
    parser.add_argument('-ev', '--eval_mode', default='fewshot', choices=['fewshot', 'simple'])
    parser.add_argument('-fe', '--feat_extractor', default='resnet18', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('-sg', '--sigmoid', default=False, action='store_true')
    parser.add_argument('-fr', '--freeze_ext', default=False, action='store_true')

    parser.add_argument('-s', '--seed', default=402, type=int, help="random seed")
    parser.add_argument('-w', '--way', default=20, type=int, help="how much way one-shot learning")
    parser.add_argument('-t', '--times', default=400, type=int, help="number of samples to test accuracy")
    parser.add_argument('-wr', '--workers', default=4, type=int, help="number of dataLoader workers")
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="number of batch size")
    parser.add_argument('-lr', '--lr', default=0.00006, type=float, help="learning rate")
    parser.add_argument('-lf', '--log_freq', default=10, type=int, help="show result after each show_every iter.")
    parser.add_argument('-sf', '--save_freq', default=100, type=int, help="save model after each save_every iter.")
    parser.add_argument('-tf', '--test_freq', default=100, type=int, help="test model after each test_every iter.")
    # parser.add_argument('-ms', '--max_steps', default=50000, type=int, help="number of steps before stopping")
    parser.add_argument('-ep', '--epochs', default=1, type=int, help="number of epochs before stopping")

    parser.add_argument('-1cf', '--first_conv_filter', default=10, type=int, help="")
    parser.add_argument('-2cf', '--second_conv_filter', default=7, type=int, help="")
    parser.add_argument('-3cf', '--third_conv_filter', default=4, type=int, help="")
    parser.add_argument('-4cf', '--fourth_conv_filter', default=4, type=int, help="")
    parser.add_argument('-5cf', '--fifth_conv_filter', default=0, type=int, help="")
    parser.add_argument('-6cf', '--sixth_conv_filter', default=0, type=int, help="")
    parser.add_argument('-7cf', '--seventh_conv_filter', default=0, type=int, help="")
    parser.add_argument('-co', '--conv_output', default=2304, type=int, help="")
    parser.add_argument('-ll', '--last_layer', default=4096, type=int, help="number of last layer neurons.")

    args = parser.parse_args()

    return args


class ModelMethods:

    def __init__(self, args, logger):
        id_str = str(time.time())
        id_str = '-time_' + id_str[:id_str.find('.')]

        self.model_name = self._parse_args(args)
        self.save_path = os.path.join(args.save_path, self.model_name + id_str)
        self.logger = logger

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            print(f'Created save directory {self.save_path}')
        else:
            print(f'Save directory {self.save_path} already exists')  # almost impossible

    def _parse_args(self, args):
        name = 'model'

        important_args = ['dataset_name',
                          'aug',
                          'rotate',
                          'batch_size',
                          'lr',
                          'ep',
                          'sigmoid',
                          'freeze_ext',
                          'feat_extractor']
        for arg in vars(args):
            if str(arg) in important_args:
                name += '-' + str(arg) + '_' + str(getattr(args, arg))

        return name

    def train(self, net, loss_fn, args, trainLoader, valLoader):
        net.train()

        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        opt.zero_grad()

        train_losses = []
        time_start = time.time()
        queue = deque(maxlen=20)

        # print('steps:', args.max_steps)

        # epochs = int(np.ceil(args.max_steps / len(trainLoader)))
        epochs = args.epochs
        print('epochs: ', epochs)

        total_batch_id = 0
        metric = Metric()

        max_val_acc = 0
        best_model = ''

        for epoch in range(epochs):

            train_loss = 0
            metric.reset_acc()

            with tqdm(total=len(trainLoader), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
                for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):

                    print('input: ', img1.size())

                    if args.cuda:
                        img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
                    else:
                        img1, img2, label = Variable(img1), Variable(img2), Variable(label)

                    net.train()
                    opt.zero_grad()

                    output = net.forward(img1, img2)
                    metric.update_acc(output, label)
                    loss = loss_fn(output, label)
                    print('loss: ', loss.item())
                    train_loss += loss.item()
                    loss.backward()

                    opt.step()
                    total_batch_id += 1
                    t.set_postfix(loss=f'{train_loss / batch_id:.4f}', train_acc=f'{metric.get_acc():.4f}')

                    # if total_batch_id % args.log_freq == 0:
                    #     logger.info('epoch: %d, batch: [%d]\tacc:\t%.5f\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                    #         epoch, batch_id, metric.get_acc(), train_loss / args.log_freq, time.time() - time_start))
                    #     train_loss = 0
                    #     metric.reset_acc()
                    #     time_start = time.time()

                    if total_batch_id % args.test_freq == 0:
                        net.eval()

                        if args.eval_mode == 'fewshot':
                            val_rgt, val_err, val_acc = self.test_fewshot(args, net, valLoader, val=True)
                        elif args.eval_mode == 'simple':
                            val_rgt, val_err, val_acc = self.test_simple(args, net, valLoader, val=True)
                        else:
                            raise Exception('Unsupporeted eval mode')
                        # right, error = 0, 0
                        # val_label = np.zeros(shape=args.way, dtype=np.float32)
                        # val_label[0] = 1
                        # val_label = torch.from_numpy(val_label).reshape((args.way, 1))
                        #
                        # if args.cuda:
                        #     val_label = Variable(val_label.cuda())
                        # else:
                        #     val_label = Variable(val_label)
                        #
                        # for _, (test1, test2) in enumerate(valLoader, 1):
                        #     if args.cuda:
                        #         test1, test2 = test1.cuda(), test2.cuda()
                        #     test1, test2 = Variable(test1), Variable(test2)
                        #     output = net.forward(test1, test2)
                        #     val_loss = loss_fn(output, val_label)
                        #     output = output.data.cpu().numpy()
                        #     pred = np.argmax(output)
                        #     if pred == 0:
                        #         right += 1
                        #     else:
                        #         error += 1
                        #
                        # val_acc = right * 1.0 / (right + error)
                        # self.logger.info('*' * 70)
                        #
                        # self.logger.info(
                        #     'epoch: %d, batch: [%d]\tVal set\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:\t%f' % (
                        #         epoch, batch_id, right, error, val_acc, val_loss))
                        # self.logger.info('*' * 70)

                        if val_acc > max_val_acc:
                            self.logger.info(
                                'saving model... current val acc: [%f], previous val acc [%f]' % (val_acc, max_val_acc))
                            best_model = self.save_model(args, net, total_batch_id, epoch, val_acc)
                            max_val_acc = val_acc

                        else:
                            self.logger.info('Not saving, best val [%f], current was [%f]' % (max_val_acc, val_acc))

                        queue.append(val_rgt * 1.0 / (val_rgt + val_err))
                    train_losses.append(train_loss)

                    t.update()

        with open('train_losses', 'wb') as f:
            pickle.dump(train_losses, f)

        acc = 0.0
        for d in queue:
            acc += d
        print("#" * 70)
        print('queue len: ', len(queue))
        print("final accuracy with train_losses: ", acc / len(queue))

        return net, best_model

    def test_simple(self, args, net, data_loader, loss_fn, val=False):
        net.eval()

        if val:
            prompt_text = 'VAL SIMPLE:\tVal set\tcorrect:\t%d\terror:\t%d\tval_loss:%f\tval_acc:%f\tval_rec:%f\tval_negacc:%f\t'
        else:
            prompt_text = 'TEST SIMPLE:\tTest set\tcorrect:\t%d\terror:\t%d\ttest_loss:%f\ttest_acc:%f\ttest_rec:%f\ttest_negacc:%f\t'

        tests_right, tests_error = 0, 0

        fn = 0
        fp = 0
        tn = 0
        tp = 0

        for label, (test1, test2) in enumerate(data_loader, 1):
            if args.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)

            output = net.forward(test1, test2)
            test_loss = loss_fn(output, label)
            output = output.data.cpu().numpy()
            pred = np.rint(output)

            tn_t, fp_t, fn_t, tp_t = confusion_matrix(label, pred).ravel()

            fn += fn_t
            tn += tn_t
            fp += fp_t
            tp += tp_t

        test_acc = ((tp + tn) * 1.0) / (tp + tn + fn + fp)
        test_recall = (tp * 0.1) / (tp + fn)
        test_negacc = (tn * 0.1) / (tn + fp)
        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_loss, test_acc, test_recall, test_negacc))
        self.logger.info('$' * 70)

        return tests_right, tests_error, test_acc

    def test_fewshot(self, args, net, data_loader, loss_fn, val=False):
        net.eval()

        if val:
            prompt_text = 'VAL FEW SHOT:\tVal set\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:%f\t'
        else:
            prompt_text = 'TEST FEW SHOT:\tTest set\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\ttest_loss:%f\t'

        tests_right, tests_error = 0, 0

        test_label = np.zeros(shape=args.way, dtype=np.float32)
        test_label[0] = 1
        test_label = torch.from_numpy(test_label).reshape((args.way, 1))

        for _, (test1, test2) in enumerate(data_loader, 1):
            if args.cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2)
            test_loss = loss_fn(output, test_label)
            output = output.data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                tests_right += 1
            else:
                tests_error += 1

        test_acc = tests_right * 1.0 / (tests_right + tests_error)
        self.logger.info('$' * 70)
        self.logger.info(prompt_text % (tests_right, tests_error, test_acc, test_loss))
        self.logger.info('$' * 70)

        return tests_right, tests_error, test_acc

    def load_model(self, args, net, best_model):
        checkpoint = torch.load(os.path.join(args.save_path, best_model))
        self.logger.info('Loading model %s from epoch [%d]' % (best_model, checkpoint['epoch']))
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def save_model(self, args, net, total_batch_id, epoch, val_acc):
        best_model = 'model-inter-' + str(total_batch_id + 1) + '-epoch-' + str(
            epoch + 1) + '-val-acc-' + str(val_acc) + '.pt'
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict()},
                   self.save_path + '/' + best_model)
        return best_model
