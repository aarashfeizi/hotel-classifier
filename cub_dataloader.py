import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import get_shuffled_data, loadDataToMem


class CUBTrain_Top(Dataset):

    def __init__(self, args, transform=None, mode='train'):
        super(CUBTrain_Top, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
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
        self.datas, self.num_classes, _, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
                                                                        args.dataset_split_type,
                                                                        mode=mode)  # todo not updated

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
        self.datas, self.num_classes, self.length, self.labels, _ = loadDataToMem(args.dataset_path, args.dataset_name,
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
