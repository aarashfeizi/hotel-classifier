import pandas as pd
import numpy as np
import os
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp', '--data_path', default='../../dataset/hotels')
    parser.add_argument('-sp', '--split_path', default='splits_50k')
    parser.add_argument('-svp', '--save_path', default='hotels_trainval')

    args = parser.parse_args()

    data_path = args.data_path
    split_path = args.split_path
    save_path = args.save_path

    train = pd.read_csv(os.path.join(split_path, 'hotels_train.csv'))
    val_seen = pd.read_csv(os.path.join(split_path, 'hotels_val_seen.csv'))
    val_unseen = pd.read_csv(os.path.join(split_path, 'hotels_val_unseen.csv'))

    # os.mkdir(save_path)

    # os.mkdir(os.path.join(save_path, 'train'))
    # os.mkdir(os.path.join(save_path, 'train/travel_website'))
    # os.mkdir(os.path.join(save_path, 'train/trafficcam'))
    #
    # os.mkdir(os.path.join(save_path, 'val_seen'))
    # os.mkdir(os.path.join(save_path, 'val_seen/travel_website'))
    # os.mkdir(os.path.join(save_path, 'val_seen/trafficcam'))
    #
    # os.mkdir(os.path.join(save_path, 'val_unseen'))
    # os.mkdir(os.path.join(save_path, 'val_unseen/travel_website'))
    # os.mkdir(os.path.join(save_path, 'val_unseen/trafficcam'))

    train_paths = train.image
    # train_lbls = train.label

    val_seen_paths = val_seen.image
    # val_seen_lbls = val_seen.label

    val_unseen_paths = val_unseen.image
    # val_unseen_lbls = val_unseen.label

    all_paths = [train_paths, val_seen_paths, val_unseen_paths]
    # all_lbls = [train_lbls, val_seen_lbls, val_unseen_lbls]
    all_paths_names = ['train', 'val_seen', 'val_unseen']

    for datasplit_name, datasplit_path in zip(all_paths_names, all_paths):
        length = len(datasplit_path)
        for i, p in enumerate(datasplit_path):
            print(datasplit_name, ':', ((i + 1) / length))
            img_dirs = p.split('/')[:-1]
            if os.path.exists(os.path.join(save_path, p)):
                print('cont:', p)
                continue
            if img_dirs[1] == 'train':
                print('Bad:', p)
                raise Exception('Should have all of train!')
            img = Image.open(os.path.join(data_path, p))  # should not get exception
            if not os.path.exists(os.path.join(save_path, *img_dirs)):
                os.makedirs(os.path.join(save_path, *img_dirs))
            img.save(os.path.join(save_path, p))


if __name__ == '__main__':
    main()
