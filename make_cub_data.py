import argparse
import os
import random

import pandas as pd


def _check_dir(path):
    return os.path.isdir(path)


def make_dataset(dir_path, split=0.0, label=0, three_split=False):
    dirs = os.listdir(dir_path)
    dir_path = dir_path[14:]
    to_add = len(dirs)
    paths = []
    senond_split = []
    if not three_split:
        for dir in dirs:
            if random.random() >= split:
                paths.append((os.path.join(dir_path, dir), label))
            else:
                senond_split.append((os.path.join(dir_path, dir), label))

        return to_add, paths, senond_split
    else:
        third_split = []
        for dir in dirs:
            rand = random.random()
            if rand >= split:
                paths.append((os.path.join(dir_path, dir), label))
            elif rand >= (split / 2) and rand < split:
                senond_split.append((os.path.join(dir_path, dir), label))
            else:
                third_split.append((os.path.join(dir_path, dir), label))
        return to_add, paths, senond_split, third_split


def save_dataset(ls, data_path, name):
    df = pd.DataFrame({'image': list(list(zip(*ls))[0]), 'label': list(list(zip(*ls))[1])})
    df.to_csv(os.path.join(data_path, f'cub_{name}.csv'), index=False, header=True)


def load_cub_data(args):
    split_path = args.split_path
    data_path = args.data_path

    with open(os.path.join(split_path, 'testclasses2.txt')) as f:
        test_cls = f.read().split('\n')[:-1]
    with open(os.path.join(split_path, 'trainclasses2.txt')) as f:
        train_cls = f.read().split('\n')[:-1]
    with open(os.path.join(split_path, 'valclasses2.txt')) as f:
        val_cls = f.read().split('\n')[:-1]

    tr = []

    kwn_vl = []
    ukwn_vl = []

    kwn_ts = []
    ukwn_ts = []

    total_train = 0
    total_val = 0
    total_test = 0

    for train in train_cls:
        lbl = int(train[:3])
        ln, tr_r, kwn_vl_r, kwn_ts_r = make_dataset(os.path.join(data_path, train), label=lbl, split=0.3, three_split=True)
        total_train += ln
        tr.extend(tr_r)
        kwn_vl.extend(kwn_vl_r)
        kwn_ts.extend(kwn_ts_r)

    for val in val_cls:
        lbl = int(val[:3])
        ln, ukwn_vl_r, kwn_ts_r = make_dataset(os.path.join(data_path, val), label=lbl, split=0.2)
        total_val += ln
        ukwn_vl.extend(ukwn_vl_r)
        kwn_ts.extend(kwn_ts_r)

    for test in test_cls:
        lbl = int(test[:3])
        ln, ukwn_ts_r, temp = make_dataset(os.path.join(data_path, test), label=lbl, split=0)
        assert len(temp) == 0
        total_test += ln
        ukwn_ts.extend(ukwn_ts_r)


    print("real total test: ", total_test)
    print("unknown test: ", len(ukwn_ts))
    print("known test:", len(kwn_ts))
    print("Total test set:", len(ukwn_ts) + len(kwn_ts))
    print('*' * 30)
    print("real total val: ", total_val)
    print("unknown val: ", len(ukwn_vl))
    print("known val:", len(kwn_vl))
    print("Total val set:", len(kwn_vl) + len(ukwn_vl))
    print('*' * 30)
    print("real train val: ", total_train)
    print("Total train set:", len(tr))
    print('*' * 30)
    print("Total trainval set:", len(tr) + len(kwn_vl) + len(ukwn_vl))

    input()


    if not os.path.exists(os.path.join(data_path, 'newsplits')):
        data_path = os.path.join(data_path, 'newsplits')
        os.mkdir(data_path)
    else:
        data_path = os.path.join(data_path, 'newsplits')

    save_dataset(tr, data_path, 'train')
    save_dataset(kwn_vl, data_path, 'knwn_cls_val')
    save_dataset(ukwn_vl, data_path, 'uknwn_cls_val')
    save_dataset(kwn_ts, data_path, 'knwn_cls_test')
    save_dataset(ukwn_ts, data_path, 'uknwn_cls_test')




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-sp', '--split_path', help="split path")
    parser.add_argument('-dp', '--data_path', help="data path")

    args = parser.parse_args()

    load_cub_data(args)



# https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html

if __name__ == '__main__':
    main()
