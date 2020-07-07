import argparse
import multiprocessing
import time

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

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
        # print(output.size())
        # print(label.size())
        # print('output: ', output)
        # print(label)
        batch_rights = sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy()[0]

        # print(f'batch_rights: {batch_rights}')

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


class Percision_At_K():

    def __init__(self):
        self.k1 = 0
        self.k5 = 0
        self.k10 = 0
        self.k100 = 0

        self.r1 = 0
        self.r5 = 0
        self.r10 = 0
        self.r100 = 0

        self.n = 0

    def update(self, lbl, ret_lbls):
        # all_lbl = sum(ret_lbls == lbl)
        if lbl == ret_lbls[0]:
            self.k1 += 1
            # self.r1 += (1 / all_lbl)
        if lbl in ret_lbls[:5]:
            self.k5 += 1
        if lbl in ret_lbls[:10]:
            self.k10 += 1
        if lbl in ret_lbls[:100]:
            self.k100 += 1

        # self.r5 += (sum(ret_lbls[:5] == lbl) / all_lbl)
        # self.r10 += (sum(ret_lbls[:10] == lbl) / all_lbl)
        # self.r100 += (sum(ret_lbls[:100] == lbl) / all_lbl)

        self.n += 1

    def __str__(self):
        k1, k5, k10, k100 = self.get_metrics()

        return f'k@1 = {k1}\n' \
               f'k@5 = {k5}\n' \
               f'k@10 = {k10}\n' \
               f'k@100 = {k100}\n'
        # f'recall@1 = {r1}\n' \
        # f'recall@5 = {r5}\n' \
        # f'recall@10 = {r10}\n' \
        # f'recall@100 = {r100}\n'

    def get_metrics(self):
        return (self.k1 / self.n), \
               (self.k5 / self.n), \
               (self.k10 / self.n), \
               (self.k100 / self.n)
        # self.r1, self.r5, self.r10, self.r100


# '../../dataset/omniglot/python/images_background'
# '../../dataset/omniglot/python/images_evaluation'
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"

    parser.add_argument('-dsn', '--dataset_name', default='omniglot', choices=['omniglot', 'cub', 'hotels'])
    parser.add_argument('-dsp', '--dataset_path', default='CUB/')
    parser.add_argument('-por', '--portion', default=0, type=int)
    parser.add_argument('-dst', '--dataset_split_type', default='new', choices=['original', 'new'])
    parser.add_argument('-sfn', '--splits_file_name', default='splits_50k')
    parser.add_argument('-ls', '--limit_samples', default=0, type=int, help="Limit samples per class for val and test")
    parser.add_argument('-nor', '--number_of_runs', default=1, type=int, help="Number of times to sample for k@n")
    parser.add_argument('-sdp', '--subdir_path', default='images/')
    parser.add_argument('-trp', '--train_path', default='./omniglot/python/images_background')
    parser.add_argument('-tsp', '--test_path', default='./omniglot/python/images_evaluation')
    parser.add_argument('-is', '--image_size', default=0, type=int, help="Image Size")
    parser.add_argument('-sp', '--save_path', default='models/', help="path to store model")
    parser.add_argument('-lp', '--log_path', default='logs/', help="path to log")
    parser.add_argument('-tbp', '--tb_path', default='tensorboard/', help="path for tensorboard")
    parser.add_argument('-a', '--aug', default=False, action='store_true')
    parser.add_argument('-r', '--rotate', default=0.0, type=float)
    parser.add_argument('-mn', '--model_name', default='')
    parser.add_argument('-pmd', '--pretrained_model_dir', default='')
    parser.add_argument('-ev', '--eval_mode', default='fewshot', choices=['fewshot', 'simple'])
    parser.add_argument('-fe', '--feat_extractor', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('-fr', '--freeze_ext', default=False, action='store_true')
    parser.add_argument('-el', '--extra_layer', default=0, type=int,
                        help="Number of 512 extra layers in the Li-Siamese")

    parser.add_argument('-s', '--seed', default=402, type=int, help="random seed")
    parser.add_argument('-w', '--way', default=20, type=int, help="how much way one-shot learning")
    parser.add_argument('-t', '--times', default=400, type=int, help="number of samples to test accuracy")
    parser.add_argument('-wr', '--workers', default=4, type=int, help="number of dataLoader workers")
    parser.add_argument('-pim', '--pin_memory', default=False, action='store_true')
    parser.add_argument('-fbw', '--find_best_workers', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="number of batch size")
    parser.add_argument('-dbb', '--db_batch', default=128, type=int, help="number of batch size for db")
    parser.add_argument('-lrs', '--lr_siamese', default=1e-3, type=float, help="siamese learning rate")
    parser.add_argument('-lrr', '--lr_resnet', default=1e-6, type=float, help="resnet learning rate")
    parser.add_argument('-lf', '--log_freq', default=10, type=int, help="show result after each show_every iter.")
    parser.add_argument('-sf', '--save_freq', default=100, type=int, help="save model after each save_every iter.")
    parser.add_argument('-tf', '--test_freq', default=100, type=int, help="test model after each test_every iter.")
    # parser.add_argument('-ms', '--max_steps', default=50000, type=int, help="number of steps before stopping")
    parser.add_argument('-ep', '--epochs', default=1, type=int, help="number of epochs before stopping")
    parser.add_argument('-es', '--early_stopping', default=10, type=int, help="number of tol for validation acc")
    parser.add_argument('-tst', '--test', default=False, action='store_true')
    parser.add_argument('-katn', '--katn', default=False, action='store_true')
    parser.add_argument('-cbir', '--cbir', default=False, action='store_true')
    parser.add_argument('-ptb', '--project_tb', default=False, action='store_true')

    parser.add_argument('-1cf', '--first_conv_filter', default=10, type=int, help="")
    parser.add_argument('-2cf', '--second_conv_filter', default=7, type=int, help="")
    parser.add_argument('-3cf', '--third_conv_filter', default=4, type=int, help="")
    parser.add_argument('-4cf', '--fourth_conv_filter', default=4, type=int, help="")
    parser.add_argument('-5cf', '--fifth_conv_filter', default=0, type=int, help="")
    parser.add_argument('-6cf', '--sixth_conv_filter', default=0, type=int, help="")
    parser.add_argument('-7cf', '--seventh_conv_filter', default=0, type=int, help="")
    parser.add_argument('-co', '--conv_output', default=2304, type=int, help="")
    parser.add_argument('-ll', '--last_layer', default=4096, type=int, help="number of last layer neurons.")
    parser.add_argument('-n', '--normalize', default=False, action='store_true')

    args = parser.parse_args()

    return args


def loading_time(args, train_set, use_cuda, num_workers, pin_memory):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    start = time.time()
    for epoch in range(4):
        for batch_idx, (_, _, _) in enumerate(train_loader):
            if batch_idx == 15:
                break
            pass
    end = time.time()
    print("  Used {} second with num_workers = {}".format(end - start, num_workers))
    return end - start


def get_best_workers_pinmemory(args, train_set, pin_memories=[False, True], starting_from=0):
    use_cuda = torch.cuda.is_available()
    core_number = multiprocessing.cpu_count()
    batch_size = 64
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    print('cpu_count =', core_number)

    for pin_memory in pin_memories:
        print("While pin_memory =", pin_memory)
        for num_workers in range(starting_from, core_number * 2 + 1, 4):
            current_time = loading_time(args, train_set, use_cuda, num_workers, pin_memory)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else:  # assuming its a convex function
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                for num_workers in (
                        the_range + list(range(best_num_worker[pin_memory] + 1, best_num_worker[pin_memory] + 4))):
                    current_time = loading_time(args, train_set, use_cuda, num_workers, pin_memory)
                    if current_time < best_time[pin_memory]:
                        best_time[pin_memory] = current_time
                        best_num_worker[pin_memory] = num_workers
                break
    if best_time[0] < best_time[1]:
        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
        workers = best_num_worker[0]
        pin_memory = False
    else:
        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")
        workers = best_num_worker[1]
        pin_memory = True

    return workers, pin_memory


def get_val_loaders(args, val_set, val_set_known, val_set_unknown, workers, pin_memory):
    val_loaders = []
    if (val_set is not None) or (val_set_known is not None):

        if args.dataset_split_type == 'original':
            val_loaders.append(DataLoader(val_set, batch_size=args.way, shuffle=False, num_workers=workers,
                                          pin_memory=pin_memory))

        elif args.dataset_split_type == 'new':
            val_loaders.append(
                DataLoader(val_set_known, batch_size=args.way, shuffle=False, num_workers=workers,
                           pin_memory=pin_memory))
            val_loaders.append(
                DataLoader(val_set_unknown, batch_size=args.way, shuffle=False, num_workers=workers,
                           pin_memory=pin_memory))
    else:
        val_loaders = None
        raise Exception('No validation data is set!')

    return val_loaders


def save_h5(data_description, data, data_type, path):
    h5_feats = h5py.File(path, 'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()


def load_h5(data_description, path):
    data = None
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data


def get_distance(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0):
    all_lbls = np.unique(img_lbls)

    k1s = []
    k5s = []
    k10s = []
    k100s = []

    k1s_s = []
    k5s_s = []
    k10s_s = []
    k100s_s = []

    k1s_u = []
    k5s_u = []
    k10s_u = []
    k100s_u = []

    for run in range(run_number):
        logger.info('### Run ' + str(run) + "...")
        chosen_img_feats = []
        chosen_img_lbls = []
        chosen_seen_list = []
        chosen_indices = []
        for lbl in all_lbls:
            sample_num = sum(img_lbls == lbl)
            lbl_indices = np.where(img_lbls == lbl)[0]

            if sample_num > limit:
                random_idx = np.random.choice(sample_num, size=limit, replace=False)

                chosen_indices.extend(lbl_indices[random_idx])
            else:
                chosen_indices.extend(lbl_indices)


        chosen_img_lbls = np.array(img_lbls[chosen_indices])
        chosen_indices = np.array(chosen_indices)

        data = {'label': chosen_img_lbls, 'index': chosen_indices}
        df = pd.DataFrame(data=data)
        df.to_csv(str(run) + '_sample_index_por' + str(args.portion) + '.csv', header=True, index=False)
        print('saved', run)

    import pdb
    pdb.set_trace()
        #
        # sim_mat = cosine_similarity(chosen_img_feats)
        # metric_total = Percision_At_K()
        # metric_seen = Percision_At_K()
        # metric_unseen = Percision_At_K()
        #
        # for idx, (row, lbl, seen) in enumerate(zip(sim_mat, chosen_img_lbls, chosen_seen_list)):
        #     ret_scores = np.delete(row, idx)
        #     ret_lbls = np.delete(chosen_img_lbls, idx)
        #     ret_seens = np.delete(chosen_seen_list, idx)
        #
        #     ret_lbls = [x for _, x in sorted(zip(ret_scores, ret_lbls), reverse=True)]
        #     ret_lbls = np.array(ret_lbls)
        #
        #     metric_total.update(lbl, ret_lbls)
        #
        #     if seen == 1:
        #         metric_seen.update(lbl, ret_lbls[ret_seens == 1])
        #     else:
        #         metric_unseen.update(lbl, ret_lbls[ret_seens == 0])
        #
        # logger.info('Total: ' + str(metric_total.n))
        # logger.info(metric_total)
        # k1, k5, k10, k100 = metric_total.get_metrics()
        # k1s.append(k1)
        # k5s.append(k5)
        # k10s.append(k10)
        # k100s.append(k100)
        # logger.info("*" * 50)
        #
        # logger.info('Seen: ' + str(metric_seen.n))
        # logger.info(metric_seen)
        # k1, k5, k10, k100 = metric_seen.get_metrics()
        # k1s_s.append(k1)
        # k5s_s.append(k5)
        # k10s_s.append(k10)
        # k100s_s.append(k100)
        # logger.info("*" * 50)
        #
        # logger.info('Unseen: ' + str(metric_unseen.n))
        # logger.info(metric_unseen)
        # k1, k5, k10, k100 = metric_unseen.get_metrics()
        # k1s_u.append(k1)
        # k5s_u.append(k5)
        # k10s_u.append(k10)
        # k100s_u.append(k100)
        # logger.info("*" * 50)
        #
        # logger.info('Avg Total: ' + str(metric_total.n))
        # logger.info('k@1: ', np.array(k1s).mean())
        # logger.info('k@5: ', np.array(k5s).mean())
        # logger.info('k@10: ', np.array(k10s).mean())
        # logger.info('k@100: ', np.array(k100s).mean())
        # logger.info("*" * 50)
        #
        # logger.info('Avg Seen: ' + str(metric_seen.n))
        # logger.info('k@1: ', np.array(k1s_s).mean())
        # logger.info('k@5: ', np.array(k5s_s).mean())
        # logger.info('k@10: ', np.array(k10s_s).mean())
        # logger.info('k@100: ', np.array(k100s_s).mean())
        # logger.info("*" * 50)
        #
        # logger.info('Avg Unseen: ' + str(metric_unseen.n))
        # logger.info('k@1: ', np.array(k1s_u).mean())
        # logger.info('k@5: ', np.array(k5s_u).mean())
        # logger.info('k@10: ', np.array(k10s_u).mean())
        # logger.info('k@100: ', np.array(k100s_u).mean())
        # logger.info("*" * 50)
