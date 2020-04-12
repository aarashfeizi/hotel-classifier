import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def loadCUBToMem(dataPath, dataset_name, mode='train'):
    if dataset_name == 'cub':
        dataset_path = os.path.join(dataPath, 'CUB')
    print("begin loading dataset to memory")
    datas = {}

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

    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []
        datas[idx].append(os.path.join(dataPath, path))

    labels = np.unique(image_labels)

    print("finish loading dataset to memory")
    return datas, num_classes, num_instances, labels


class CUBTrain(Dataset):

    def __init__(self, args, transform=None):
        super(CUBTrain, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes, self.length, self.labels = loadCUBToMem(args.dataset_path, args.dataset_name,
                                                                              mode='train')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            class1 = self.labels[idx1]
            image1 = Image.open(random.choice(self.datas[class1]))
            image2 = Image.open(random.choice(self.datas[class1]))
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)

            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)

            class1 = self.labels[idx1]
            class2 = self.labels[idx2]

            image1 = Image.open(random.choice(self.datas[class1]))
            image2 = Image.open(random.choice(self.datas[class2]))

        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class CUBTest_Fewshot(Dataset):

    def __init__(self, args, transform=None, mode='test'):
        np.random.seed(args.seed)
        super(CUBTest_Fewshot, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes, _, self.labels = loadCUBToMem(args.dataset_path, args.dataset_name, mode=mode)

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = self.labels[random.randint(0, self.num_classes - 1)]
            self.img1 = Image.open(random.choice(self.datas[self.c1])).convert('RGB')
            img2 = Image.open(random.choice(self.datas[self.c1])).convert('RGB')
        # generate image pair from different class
        else:
            c2 = self.labels[random.randint(0, self.num_classes - 1)]
            while self.c1 == c2:
                c2 = self.labels[random.randint(0, self.num_classes - 1)]
            img2 = Image.open(random.choice(self.datas[c2])).convert('RGB')

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


class OmniglotTrain(Dataset):

    def __init__(self, args, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(args.train_path)
        # import pdb
        # pdb.set_trace()

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                    datas[idx] = []
                    for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                        filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):

    def __init__(self, args, transform=None):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(args.test_path)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


def loadHotels(dataset_path, dataset_name, mode='train'):
    if dataset_name == 'hotels':
        dataset_path = os.path.join(dataset_path, 'hotels')

    with open(os.path.join(dataset_path, 'hotel50-image_label.csv')) as f:
        hotels = pd.read_csv(f)
    with open(os.path.join(dataset_path, 'background_or_novel.csv')) as f:
        b_or_n = pd.read_csv(f)

    train = (mode == 'train')

    if train:
        label_list = list(b_or_n[b_or_n['background'] == 1]['label'])  # for background classes
    else:
        label_list = list(b_or_n[b_or_n['background'] == 0]['label'])  # for novel classses

    datas = {}
    length = 0
    for idx, row in hotels.iterrows():
        if row['hotel_label'] in label_list:
            lbl = row['hotel_label']
            if lbl not in datas.keys():
                datas[lbl] = []
            datas[lbl].append(os.path.join(dataset_path, row['image']))

    for _, value in datas.items():
        length += len(value)

    return datas, len(label_list), length, label_list


class HotelTrain(Dataset):
    def __init__(self, args, transform=None):
        super(HotelTrain, self).__init__()
        np.random.seed(args.seed)
        self.transform = transform
        self.datas, self.num_classes, self.length, self.labels = loadHotels(args.dataset_path, args.dataset_name,
                                                                            mode='train')
        print('hotel train classes: ', self.num_classes)
        print('hotel train length: ', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            class1 = self.labels[idx1]
            image1 = Image.open(random.choice(self.datas[class1]))
            image2 = Image.open(random.choice(self.datas[class1]))
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)

            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)

            class1 = self.labels[idx1]
            class2 = self.labels[idx2]

            image1 = Image.open(random.choice(self.datas[class1]))
            image2 = Image.open(random.choice(self.datas[class2]))

        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class HotelTest(Dataset):

    def __init__(self, args, transform=None):
        np.random.seed(args.seed)
        super(HotelTest, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes, _, self.labels = loadHotels(args.dataset_path, args.dataset_name, mode='test')

        print('hotel test classes: ', self.num_classes)
        print('hotel test length: ', self.__len__())

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = self.labels[random.randint(0, self.num_classes - 1)]
            self.img1 = Image.open(random.choice(self.datas[self.c1])).convert('RGB')
            img2 = Image.open(random.choice(self.datas[self.c1])).convert('RGB')
        # generate image pair from different class
        else:
            c2 = self.labels[random.randint(0, self.num_classes - 1)]
            while self.c1 == c2:
                c2 = self.labels[random.randint(0, self.num_classes - 1)]
            img2 = Image.open(random.choice(self.datas[c2])).convert('RGB')

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


class ImageNet(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'
        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class MetaImageNet(ImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.ways
        # self.n_shots = args.n_shots
        self.n_shots = 1 # for now
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.times
        self.n_aug_support_samples = args.aug
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs
