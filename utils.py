import argparse
import json
import multiprocessing
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

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
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        else:
            return method()

    def get_composed_transform(self, aug=False, random_crop=False):
        transform_list = []
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip']
        elif not aug and self.rotate == 0:
            transform_list = ['Resize']
        elif not aug and self.rotate != 0:
            transform_list = ['Resize', 'RandomRotation']

        if random_crop:
            transform_list.extend(['RandomResizedCrop'])
        else:
            transform_list.extend(['CenterCrop'])

        transform_list.extend(['ToTensor', 'Normalize'])

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform, transform_list


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
        import pdb

        # pdb.set_trace()
        # print('pox', sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy())
        batch_rights = sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy()

        # print(f'batch_rights: {batch_rights}')

        self.rights += batch_rights
        self.wrongs += (label.shape[0] - batch_rights)

    def get_acc(self):
        # print('rights: ', self.rights)
        # print('wrongs: ', self.wrongs)
        return (self.rights / (self.rights + self.wrongs)) * 100

    def get_right_wrong(self):
        return {'right': self.rights, 'wrong': self.wrongs}

    def reset_acc(self):
        self.rights = 0
        self.wrongs = 0


class Percision_At_K():

    def __init__(self, classes=np.array([])):
        self.k1 = 0
        self.k5 = 0
        self.k10 = 0
        self.k100 = 0

        self.r1 = 0
        self.r5 = 0
        self.r10 = 0
        self.r100 = 0

        self.n = 0

        self.classes = classes
        self.class_tot = len(self.classes)
        self.lbl2idx = {c: i for i, c in enumerate(self.classes)}
        self.per_class_k1 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k5 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k10 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k100 = np.zeros(shape=self.class_tot)  # col1: kns

        self.per_class_n = np.zeros(shape=self.class_tot)

    def update(self, lbl, ret_lbls):
        # all_lbl = sum(ret_lbls == lbl)
        if lbl == ret_lbls[0]:
            self.k1 += 1
            self.per_class_k1[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:5]:
            self.k5 += 1
            self.per_class_k5[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:10]:
            self.k10 += 1
            self.per_class_k10[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:100]:
            self.k100 += 1
            self.per_class_k100[self.lbl2idx[lbl]] += 1

        # self.r5 += (sum(ret_lbls[:5] == lbl) / all_lbl)
        # self.r10 += (sum(ret_lbls[:10] == lbl) / all_lbl)
        # self.r100 += (sum(ret_lbls[:100] == lbl) / all_lbl)

        self.n += 1
        self.per_class_n[self.lbl2idx[lbl]] += 1

    def __str__(self):
        k1, k5, k10, k100 = self.get_tot_metrics()

        return f'k@1 = {k1}\n' \
               f'k@5 = {k5}\n' \
               f'k@10 = {k10}\n' \
               f'k@100 = {k100}\n'
        # f'recall@1 = {r1}\n' \
        # f'recall@5 = {r5}\n' \
        # f'recall@10 = {r10}\n' \
        # f'recall@100 = {r100}\n'

    def get_tot_metrics(self):

        return (self.k1 / max(self.n, 1)), \
               (self.k5 / max(self.n, 1)), \
               (self.k10 / max(self.n, 1)), \
               (self.k100 / max(self.n, 1))

    def get_per_class_metrics(self):

        assert sum(self.per_class_n) == self.n
        assert sum(self.per_class_k1) == self.k1
        assert sum(self.per_class_k5) == self.k5
        assert sum(self.per_class_k10) == self.k10
        assert sum(self.per_class_k100) == self.k100

        if self.n == 0:
            denom = [1 for _ in range(len(self.per_class_n))]
        else:
            denom = self.per_class_n

        k1s, k5s, k10s, k100s = (self.per_class_k1 / denom), \
                                (self.per_class_k5 / denom), \
                                (self.per_class_k10 / denom), \
                                (self.per_class_k100 / denom)

        d = {'label': self.classes,
             'n': self.per_class_n,
             'k@1': k1s,
             'k@5': k5s,
             'k@10': k10s,
             'k@100': k100s}

        df = pd.DataFrame(data=d)

        return df

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
    parser.add_argument('-nn', '--no_negative', default=1, type=int)

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
    parser.add_argument('-sr', '--sampled_results', default=True, action='store_true')
    parser.add_argument('-pcr', '--per_class_results', default=True, action='store_true')
    parser.add_argument('-ptb', '--project_tb', default=False, action='store_true')

    parser.add_argument('-mtlr', '--metric_learning', default=False, action='store_true')
    parser.add_argument('-mg', '--margin', default=0.0, type=float, help="margin for triplet loss")
    parser.add_argument('-lss', '--loss', default='bce', choices=['bce', 'trpl'])


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


def calculate_k_at_n(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0, sampled=True,
                     per_class=False, save_path='', mode=''):
    if per_class:
        total, seen, unseen = _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode)
        total.to_csv(os.path.join(save_path, f'{mode}_per_class_total_avg_k@n.csv'), header=True, index=False)
        seen.to_csv(os.path.join(save_path, f'{mode}_per_class_seen_avg_k@n.csv'), header=True, index=False)
        unseen.to_csv(os.path.join(save_path, f'{mode}_per_class_unseen_avg_k@n.csv'), header=True, index=False)

    if sampled:
        kavg, kruns, total, seen, unseen = _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit,
                                                                 run_number, mode)
        kavg.to_csv(os.path.join(save_path, f'{mode}_sampled_avg_k@n.csv'), header=True, index=False)
        kruns.to_csv(os.path.join(save_path, f'{mode}_sampled_runs_k@n.csv'), header=True, index=False)
        total.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_total_avg_k@n.csv'), header=True, index=False)
        seen.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_seen_avg_k@n.csv'), header=True, index=False)
        unseen.to_csv(os.path.join(save_path, f'{mode}_sampled_per_class_unseen_avg_k@n.csv'), header=True, index=False)

    return True


def _get_per_class_distance(args, img_feats, img_lbls, seen_list, logger, mode):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])

    sim_mat = cosine_similarity(img_feats)

    metric_total = Percision_At_K(classes=np.array(all_lbls))
    metric_seen = Percision_At_K(classes=np.array(seen_lbls))
    metric_unseen = Percision_At_K(classes=np.array(unseen_lbls))

    for idx, (row, lbl, seen) in enumerate(zip(sim_mat, img_lbls, seen_list)):
        ret_scores = np.delete(row, idx)
        ret_lbls = np.delete(img_lbls, idx)
        ret_seens = np.delete(seen_list, idx)

        ret_lbls = [x for _, x in sorted(zip(ret_scores, ret_lbls), reverse=True)]
        ret_seens = [x for _, x in sorted(zip(ret_scores, ret_seens), reverse=True)]

        ret_lbls = np.array(ret_lbls)
        ret_seens = np.array(ret_seens)

        metric_total.update(lbl, ret_lbls)

        if seen == 1:
            metric_seen.update(lbl, ret_lbls[ret_seens == 1])
        else:
            metric_unseen.update(lbl, ret_lbls[ret_seens == 0])

    total = metric_total.get_per_class_metrics()
    seen = metric_seen.get_per_class_metrics()
    unseen = metric_unseen.get_per_class_metrics()

    logger.info(f'{mode}')
    logger.info('Without sampling Total: ' + str(metric_total.n))
    logger.info(metric_total)

    logger.info(f'{mode}')
    _log_per_class(logger, total, split_kind='Total')

    logger.info(f'{mode}')
    logger.info('Without sampling Seen: ' + str(metric_seen.n))
    logger.info(metric_seen)

    logger.info(f'{mode}')
    _log_per_class(logger, seen, split_kind='Seen')

    logger.info(f'{mode}')
    logger.info('Without sampling Unseen: ' + str(metric_unseen.n))
    logger.info(metric_unseen)

    logger.info(f'{mode}')
    _log_per_class(logger, unseen, split_kind='Unseen')

    return total, seen, unseen


def _log_per_class(logger, df, split_kind=''):
    logger.info(f'Per class {split_kind}: {np.array(df["n"]).sum()}')
    logger.info(f'Average per class {split_kind}: {np.array(df["n"]).mean()}')
    logger.info(f'k@1 per class average: {np.array(df["k@1"]).mean()}')
    logger.info(f'k@5 per class average: {np.array(df["k@5"]).mean()}')
    logger.info(f'k@10 per class average: {np.array(df["k@10"]).mean()}')
    logger.info(f'k@100 per class average: {np.array(df["k@100"]).mean()}\n')


def _get_sampled_distance(args, img_feats, img_lbls, seen_list, logger, limit=0, run_number=0, mode=''):
    all_lbls = np.unique(img_lbls)
    seen_lbls = np.unique(img_lbls[seen_list == 1])
    unseen_lbls = np.unique(img_lbls[seen_list == 0])

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

    sampled_indices_all = pd.read_csv('sample_index_por' + str(args.portion) + '.csv')
    sampled_label_all = pd.read_csv('sample_label_por' + str(args.portion) + '.csv')

    for run in range(run_number):
        column_name = f'run{run}'
        sampled_indices = np.array(sampled_indices_all[column_name]).astype(int)
        sampled_labels = np.array(sampled_label_all[column_name]).astype(int)

        logger.info(f'{mode}')
        logger.info('### Run ' + str(run) + "...")
        chosen_img_feats = img_feats[sampled_indices]
        chosen_img_lbls = img_lbls[sampled_indices]
        chosen_seen_list = seen_list[sampled_indices]

        assert np.array_equal(sampled_labels, chosen_img_lbls)

        sim_mat = cosine_similarity(chosen_img_feats)
        metric_total = Percision_At_K(classes=all_lbls)
        metric_seen = Percision_At_K(classes=seen_lbls)
        metric_unseen = Percision_At_K(classes=unseen_lbls)

        for idx, (row, lbl, seen) in enumerate(zip(sim_mat, chosen_img_lbls, chosen_seen_list)):
            ret_scores = np.delete(row, idx)
            ret_lbls = np.delete(chosen_img_lbls, idx)
            ret_seens = np.delete(chosen_seen_list, idx)

            ret_lbls = [x for _, x in sorted(zip(ret_scores, ret_lbls), reverse=True)]
            ret_lbls = np.array(ret_lbls)

            metric_total.update(lbl, ret_lbls)

            if seen == 1:
                metric_seen.update(lbl, ret_lbls[ret_seens == 1])
            else:
                metric_unseen.update(lbl, ret_lbls[ret_seens == 0])

        total = metric_total.get_per_class_metrics()
        seen = metric_seen.get_per_class_metrics()
        unseen = metric_unseen.get_per_class_metrics()

        logger.info('Total: ' + str(metric_total.n))
        logger.info(metric_total)
        k1, k5, k10, k100 = metric_total.get_tot_metrics()
        k1s.append(k1)
        k5s.append(k5)
        k10s.append(k10)
        k100s.append(k100)
        logger.info("*" * 50)

        logger.info('Seen: ' + str(metric_seen.n))
        logger.info(metric_seen)
        k1, k5, k10, k100 = metric_seen.get_tot_metrics()
        k1s_s.append(k1)
        k5s_s.append(k5)
        k10s_s.append(k10)
        k100s_s.append(k100)
        logger.info("*" * 50)

        logger.info('Unseen: ' + str(metric_unseen.n))
        logger.info(metric_unseen)
        k1, k5, k10, k100 = metric_unseen.get_tot_metrics()
        k1s_u.append(k1)
        k5s_u.append(k5)
        k10s_u.append(k10)
        k100s_u.append(k100)
        logger.info("*" * 50)

        _log_per_class(logger, total, split_kind='Total')
        _log_per_class(logger, seen, split_kind='Seen')
        _log_per_class(logger, unseen, split_kind='Unseen')

    total = metric_total.get_per_class_metrics()
    seen = metric_seen.get_per_class_metrics()
    unseen = metric_unseen.get_per_class_metrics()

    logger.info('Avg Total: ' + str(metric_total.n))
    logger.info('k@1: ' + str(np.array(k1s).mean()))
    logger.info('k@5: ' + str(np.array(k5s).mean()))
    logger.info('k@10: ' + str(np.array(k10s).mean()))
    logger.info('k@100: ' + str(np.array(k100s).mean()))
    logger.info("*" * 50)

    logger.info('Avg Seen: ' + str(metric_seen.n))
    logger.info('k@1: ' + str(np.array(k1s_s).mean()))
    logger.info('k@5: ' + str(np.array(k5s_s).mean()))
    logger.info('k@10: ' + str(np.array(k10s_s).mean()))
    logger.info('k@100: ' + str(np.array(k100s_s).mean()))
    logger.info("*" * 50)

    logger.info('Avg Unseen: ' + str(metric_unseen.n))
    logger.info('k@1: ' + str(np.array(k1s_u).mean()))
    logger.info('k@5: ' + str(np.array(k5s_u).mean()))
    logger.info('k@10: ' + str(np.array(k10s_u).mean()))
    logger.info('k@100: ' + str(np.array(k100s_u).mean()))
    logger.info("*" * 50)

    d = {'run': [i for i in range(run_number)],
         'kAT1': k1s,
         'kAT5': k5s,
         'kAT10': k10s,
         'kAT100': k100s,
         'kAT1_seen': k1s_s,
         'kAT5_seen': k5s_s,
         'kAT10_seen': k10s_s,
         'kAT100_seen': k100s_s,
         'kAT1_unseen': k1s_u,
         'kAT5_unseen': k5s_u,
         'kAT10_unseen': k10s_u,
         'kAT100_unseen': k100s_u}

    average_tot = pd.DataFrame(data={'avg_kAT1': [np.array(k1s).mean()],
                                     'avg_kAT5': [np.array(k5s).mean()],
                                     'avg_kAT10': [np.array(k10s).mean()],
                                     'avg_kAT100': [np.array(k100s).mean()],
                                     'avg_kAT1_seen': [np.array(k1s_s).mean()],
                                     'avg_kAT5_seen': [np.array(k5s_s).mean()],
                                     'avg_kAT10_seen': [np.array(k10s_s).mean()],
                                     'avg_kAT100_seen': [np.array(k100s_s).mean()],
                                     'avg_kAT1_unseen': [np.array(k1s_u).mean()],
                                     'avg_kAT5_unseen': [np.array(k5s_u).mean()],
                                     'avg_kAT10_unseen': [np.array(k10s_u).mean()],
                                     'avg_kAT100_unseen': [np.array(k100s_u).mean()]})

    return average_tot, pd.DataFrame(data=d), total, seen, unseen


def get_shuffled_data(datas, seed=0, one_hot=True, both_seen_unseen=False, shuffle=True):  # for sequential labels only

    labels = sorted(datas.keys())

    if one_hot:
        lbl2idx = {labels[idx]: idx for idx in range(len(labels))}
        one_hot_labels = np.eye(len(np.unique(labels)))
    # print(one_hot_labels)

    np.random.seed(seed)

    data = []
    for key, value_list in datas.items():
        if one_hot:
            lbl = one_hot_labels[lbl2idx[key]]
        else:
            lbl = key

        if both_seen_unseen:
            ls = [(lbl, value, bl) for value, bl in value_list]  # todo to be able to separate seen and unseen in k@n
        else:
            ls = [(lbl, value) for value in value_list]

        data.extend(ls)

    if shuffle:
        np.random.shuffle(data)

    return data


def _read_org_split(dataset_path, mode):
    image_labels = []
    image_path = []

    if mode == 'train':  # train

        with open(os.path.join(dataset_path, 'base.json'), 'r') as f:
            base_dict = json.load(f)
        image_labels.extend(base_dict['image_labels'])
        image_path.extend(base_dict['image_names'])

    elif mode == 'val':  # val

        with open(os.path.join(dataset_path, 'val.json'), 'r') as f:
            val_dict = json.load(f)
        image_labels.extend(val_dict['image_labels'])
        image_path.extend(val_dict['image_names'])

    elif mode == 'test':  # novel classes

        with open(os.path.join(dataset_path, 'novel.json'), 'r') as f:
            novel_dict = json.load(f)
        image_labels.extend(novel_dict['image_labels'])
        image_path.extend(novel_dict['image_names'])

    return image_path, image_labels


def _read_new_split(dataset_path, mode,
                    dataset_name='cub'):  # mode = [test_seen, val_seen, train, test_unseen, test_unseen]

    file_name = f'{dataset_name}_' + mode + '.csv'

    file = pd.read_csv(os.path.join(dataset_path, file_name))
    image_labels = np.array(file.label)
    image_path = np.array(file.image)

    return image_path, image_labels


def loadDataToMem(dataPath, dataset_name, split_type, mode='train', split_file_name='final_newsplits0_1',
                  portion=0, return_paths=False):
    print(split_file_name, '!!!!!!!!')
    if dataset_name == 'cub':
        dataset_path = os.path.join(dataPath, 'CUB')
    elif dataset_name == 'hotels':
        dataset_path = os.path.join(dataPath, 'hotels_trainval')

    background_datasets = {'val_seen': 'val_unseen',
                           'val_unseen': 'val_seen',
                           'test_seen': 'test_unseen',
                           'test_unseen': 'test_seen',
                           'train_seen': 'train_seen'}

    print("begin loading dataset to memory")
    datas = {}
    datas_bg = {}  # in case of mode == val/test_seen/unseen

    if split_type == 'original':
        image_path, image_labels = _read_org_split(dataset_path, mode)
    elif split_type == 'new':
        image_path, image_labels = _read_new_split(os.path.join(dataset_path, split_file_name), mode, dataset_name)
        if mode != 'train':
            image_path_bg, image_labels_bg = _read_new_split(os.path.join(dataset_path, split_file_name),
                                                             background_datasets[mode], dataset_name)

    if portion > 0:
        image_path = image_path[image_labels < portion]
        image_labels = image_labels[image_labels < portion]

        if mode != 'train':
            image_path_bg = image_path_bg[image_labels_bg < portion]
            image_labels_bg = image_labels_bg[image_labels_bg < portion]

    print(f'{mode} number of imgs:', len(image_labels))
    print(f'{mode} number of labels:', len(np.unique(image_labels)))

    if mode != 'train':
        print(f'{mode} number of bg imgs:', len(image_labels_bg))
        print(f'{mode} number of bg lbls:', len(np.unique(image_labels_bg)))

    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []
            if mode != 'train':
                datas_bg[idx] = []

        datas[idx].append(os.path.join(dataset_path, path))
        if mode != 'train':
            datas_bg[idx].append((os.path.join(dataset_path, path), True))

    if mode != 'train':
        for idx, path in zip(image_labels_bg, image_path_bg):
            if idx not in datas_bg.keys():
                datas_bg[idx] = []
            if (os.path.join(dataset_path, path), False) not in datas_bg[idx] and \
                    (os.path.join(dataset_path, path), True) not in datas_bg[idx]:
                datas_bg[idx].append((os.path.join(dataset_path, path), False))

    labels = np.unique(image_labels)
    print(f'Number of labels in {mode}: ', len(labels))

    if mode != 'train':
        all_labels = np.unique(np.concatenate((image_labels, image_labels_bg)))
        print(f'Number of all labels (bg + fg) in {mode} and {background_datasets[mode]}: ', len(all_labels))

    print(f'finish loading {mode} dataset to memory')
    return datas, num_classes, num_instances, labels, datas_bg


def project_2d(features, labels, title):
    pca = PCA(n_components=2)
    pca_feats = pca.fit_transform(features)
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    plt.scatter(pca_feats[:, 0], pca_feats[:, 1], c=labels, cmap=cmap, alpha=0.2)
    plt.colorbar()
    plt.title(title)

    return plt


def choose_n_from_all(df, n=4):
    chosen_labels = []
    chosen_images = []

    lbls = np.array(df.label)
    images = np.array(df.image)

    lbls_unique = np.unique(lbls)

    for lbl in lbls_unique:
        mask = lbls == lbl
        single_lbl_paths = images[mask]

        if len(single_lbl_paths) > n:
            temp = np.random.choice(single_lbl_paths, size=n, replace=False)
            chosen_images.extend(temp)
            chosen_labels.extend([lbl for _ in range(n)])
        else:
            chosen_images.extend(single_lbl_paths)
            chosen_labels.extend([lbl for _ in range(len(single_lbl_paths))])

    data = {'label': chosen_labels, 'image': chosen_images}

    return pd.DataFrame(data=data)


def create_save_path(path, id_str, logger):
    if not os.path.exists(path):
        os.mkdir(path)
        logger.info(
            f'Created save and tensorboard directories:\n{path}\n')
    else:
        logger.info(f'Save directory {path} already exists, but how?? {id_str}')  # almost impossible
