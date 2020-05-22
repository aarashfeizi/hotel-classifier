import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
    image_labels = list(file.label)
    image_path = list(file.image)

    return image_path, image_labels


def loadDataToMem(dataPath, dataset_name, split_type, mode='train', split_file_name='final_newsplits0_1'):
    print(split_file_name, '!!!!!!!!')
    if dataset_name == 'cub':
        dataset_path = os.path.join(dataPath, 'CUB')
    elif dataset_name == 'hotels':
        dataset_path = os.path.join(dataPath, 'hotels')

    print("begin loading dataset to memory")
    datas = {}

    if split_type == 'original':
        image_path, image_labels = _read_org_split(dataset_path, mode)
    elif split_type == 'new':
        image_path, image_labels = _read_new_split(os.path.join(dataset_path, split_file_name), mode, dataset_name)

    num_instances = len(image_labels)

    num_classes = len(np.unique(image_labels))

    for idx, path in zip(image_labels, image_path):
        if idx not in datas.keys():
            datas[idx] = []
        datas[idx].append(os.path.join(dataset_path, path))

    labels = np.unique(image_labels)

    print("finish loading dataset to memory")
    return datas, num_classes, num_instances, labels


def get_shuffled_data(datas, seed=0):  # for sequential labels only

    labels = sorted(datas.keys())

    lbl2idx = {labels[idx]: idx for idx in range(len(labels))}

    one_hot_labels = np.eye(len(np.unique(labels)))
    # print(one_hot_labels)

    np.random.seed(seed)
    data = []
    for key, value_list in datas.items():
        ls = [(one_hot_labels[lbl2idx[key]], value) for value in value_list]
        data.extend(ls)

    np.random.shuffle(data)

    return data


class CUBTrain_Top(Dataset):

    def __init__(self, args, transform=None, mode='train'):
        super(CUBTrain_Top, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes, self.length, self.labels = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                               args.dataset_split_type,
                                                                               mode=mode)

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)

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

    def _get_single_item(self, index):
        label, image_path = self.shuffled_data[index]

        image = Image.open(image_path)

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(label, dtype=np.float32))

    def get_k_samples(self, k=100):
        ks = np.random.randint(len(self.shuffled_data), size=k)
        imgs = []
        lbls = []
        for i in ks:
            img, lbl = self._get_single_item(i)
            imgs.append(img)
            lbls.append(lbl)

        return imgs, lbls


class CUBTest_Fewshot(Dataset):

    def __init__(self, args, transform=None, mode='test'):
        np.random.seed(args.seed)
        super(CUBTest_Fewshot, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes, _, self.labels = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                     args.dataset_split_type, mode=mode)

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


class CUBClassification(Dataset):

    def __init__(self, args, transform=None, mode='train'):  # train or val
        super(CUBClassification, self).__init__()
        np.random.seed(args.seed)
        self.transform = transform
        self.datas, self.num_classes, self.length, self.labels = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                               args.dataset_split_type,
                                                                               mode=mode)
        self.shuffled_data = get_shuffled_data(self.datas, seed=args.seed)
        # import pdb
        # pdb.set_trace()

    def __getitem__(self, index):
        label, image_path = self.shuffled_data[index]

        image = Image.open(image_path)

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(label, dtype=np.float32))

    def __len__(self):
        return len(self.shuffled_data)


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
    def __init__(self, args, transform=None, mode='train'):
        super(HotelTrain, self).__init__()
        np.random.seed(args.seed)
        self.transform = transform

        self.datas, self.num_classes, self.length, self.labels = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                               args.dataset_split_type,
                                                                               mode=mode, split_file_name='splits_50k')

        self.shuffled_data = get_shuffled_data(datas=self.datas, seed=args.seed)

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

    def __init__(self, args, transform=None, mode='test_seen'):
        np.random.seed(args.seed)
        super(HotelTest, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None

        self.datas, self.num_classes, _, self.labels = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                     args.dataset_split_type, mode=mode,
                                                                     split_file_name='splits_50k')

        print(f'hotel {mode} classes: ', self.num_classes)
        print(f'hotel {mode} length: ', self.__len__())

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
