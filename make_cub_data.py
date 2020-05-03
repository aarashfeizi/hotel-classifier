import argparse
import os
import random

import numpy as np
import pandas as pd


def _check_dir(path):
    return os.path.isdir(path)


def make_dataset(dir_path, split=0.0, label=0, second_split=-1.0):
    dirs = os.listdir(dir_path)
    dir_path = dir_path[14:]
    to_add = len(dirs)
    paths = []
    senond_split = []
    if second_split < 0:
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
            if rand >= (split + second_split):
                paths.append((os.path.join(dir_path, dir), label))  # train
            elif split <= rand < (second_split + split):
                senond_split.append((os.path.join(dir_path, dir), label))  # val
            else:
                third_split.append((os.path.join(dir_path, dir), label))  # test
        return to_add, paths, senond_split, third_split


def save_dataset(ls, data_path, name):
    df = pd.DataFrame({'image': list(list(zip(*ls))[0]), 'label': list(list(zip(*ls))[1])})
    df.to_csv(os.path.join(data_path, f'cub_{name}.csv'), index=False, header=True)


def load_cub_data(args):
    split_path = args.split_path
    data_path = args.data_path

    version = args.version

    with open(os.path.join(split_path, f'testclasses{version}.txt')) as f:
        test_cls = f.read().split('\n')[:-1]
    with open(os.path.join(split_path, f'trainclasses{version}.txt')) as f:
        train_cls = f.read().split('\n')[:-1]
    with open(os.path.join(split_path, f'valclasses{version}.txt')) as f:
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
        ln, tr_r, kwn_vl_r, kwn_ts_r = make_dataset(os.path.join(data_path, train),
                                                    label=lbl,
                                                    split=0.33,  # test
                                                    second_split=0.12)  # val
        total_train += ln
        tr.extend(tr_r)
        kwn_vl.extend(kwn_vl_r)
        kwn_ts.extend(kwn_ts_r)

    for val in val_cls:
        lbl = int(val[:3])
        ln, ukwn_vl_r, kwn_ts_r = make_dataset(os.path.join(data_path, val), label=lbl, split=0.1)
        total_val += ln
        ukwn_vl.extend(ukwn_vl_r)
        kwn_ts.extend(kwn_ts_r)

    for test in test_cls:
        lbl = int(test[:3])
        ln, ukwn_ts_r, temp = make_dataset(os.path.join(data_path, test), label=lbl, split=0)
        assert len(temp) == 0
        total_test += ln
        ukwn_ts.extend(ukwn_ts_r)

    output_string = ''
    print("real total test: ", total_test)
    output_string += "real total test: " + str(total_test) + '\n'

    print("unknown test: ", len(ukwn_ts))
    output_string += "unknown test: " + str(len(ukwn_ts)) + '\n'

    print("known test:", len(kwn_ts))
    output_string += "known test: " + str(len(kwn_ts)) + '\n'

    print("Total test set:", len(ukwn_ts) + len(kwn_ts))
    output_string += "Total test set: " + str((len(ukwn_ts) + len(kwn_ts))) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("real total val: ", total_val)
    output_string += "real total val: " + str(total_val) + '\n'

    print("unknown val: ", len(ukwn_vl))
    output_string += "unknown val: " + str(len(ukwn_vl)) + '\n'

    print("known val:", len(kwn_vl))
    output_string += "known val: " + str(len(kwn_vl)) + '\n'

    print("Total val set:", len(kwn_vl) + len(ukwn_vl))
    output_string += "Total val set: " + str(len(kwn_vl) + len(ukwn_vl)) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("real train val: ", total_train)
    output_string += "real train val: " + str(total_train) + '\n'

    print("Total train set:", len(tr))
    output_string += "Total train set: " + str(len(tr)) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("Total trainval set:", len(tr) + len(kwn_vl) + len(ukwn_vl))
    output_string += "Total trainval set: " + str(len(tr) + len(kwn_vl) + len(ukwn_vl)) + '\n'

    input()

    if not os.path.exists(os.path.join(data_path, f'newsplits{version}_{args.save_version}')):
        data_path = os.path.join(data_path, f'newsplits{version}_{args.save_version}')
        os.mkdir(data_path)
    else:
        data_path = os.path.join(data_path, f'newsplits{version}_{args.save_version}')

    with open(os.path.join(data_path, 'config.txt'), 'w') as f:
        f.write(output_string)

    save_dataset(tr, data_path, 'train')
    save_dataset(kwn_vl, data_path, 'knwn_cls_val')
    save_dataset(ukwn_vl, data_path, 'uknwn_cls_val')
    save_dataset(kwn_ts, data_path, 'knwn_cls_test')
    save_dataset(ukwn_ts, data_path, 'uknwn_cls_test')

    _number_of_classes(data_path)


def _number_of_classes(datapath):
    print('data split path', datapath)
    dirs = os.listdir(datapath)
    csvs = []
    names = []

    for dir in dirs:
        if dir.endswith('.csv'):
            df = pd.read_csv(os.path.join(datapath, dir))
            csvs.append(df)
            names.append(dir[:-4])

    # cub_train = pd.read_csv(os.path.join(datapath, 'cub_train' + '.csv'))
    # cub_knwn_cls_test = pd.read_csv(os.path.join(datapath, 'cub_knwn_cls_test' + '.csv'))
    # cub_knwn_cls_val = pd.read_csv(os.path.join(datapath, 'cub_knwn_cls_val' + '.csv'))
    # cub_uknwn_cls_test = pd.read_csv(os.path.join(datapath, 'cub_uknwn_cls_test' + '.csv'))
    # cub_uknwn_cls_val = pd.read_csv(os.path.join(datapath, 'cub_uknwn_cls_val' + '.csv'))


    for name, csv in zip(names, csvs):
        print(name, "number of classes:", len(np.unique(list(csv.label))))

def make_from_new_split(args):

    data_path = args.data_path

    splits = pd.read_csv(os.path.join(args.split_path, 'correct_split.csv'))

    trainval_lbls = np.array(splits.label[splits.split == 'trainval'])
    test_seen_lbls = np.array(splits.label[splits.split == 'test_seen'])
    test_unseen_lbls = np.array(splits.label[splits.split == 'test_unseen'])

    trainval_pths = np.array(splits.path[splits.split == 'trainval'])
    test_seen_pths = np.array(splits.path[splits.split == 'test_seen'])
    test_unseen_pths = np.array(splits.path[splits.split == 'test_unseen'])

    test_seen = list(zip(test_seen_pths, test_seen_lbls))
    test_unseen = list(zip(test_unseen_pths, test_unseen_lbls))

    trainval = list(zip(trainval_pths, trainval_lbls))

    assert len(np.unique(trainval_lbls)) == 150
    assert len(np.unique(test_seen_lbls)) == 150
    assert len(np.unique(test_unseen_lbls)) == 50

    val_unseen_lbls = np.random.choice(np.unique(trainval_lbls), args.val_unseen, replace=False)
    val_unseen = []
    rest = []

    for (path, lbl) in trainval:
        if lbl in val_unseen_lbls:
            val_unseen.append((path, lbl))
        else:
            rest.append((path, lbl))


    assert len(np.unique(list(list(zip(*rest))[1]))) == 150 - (args.val_unseen)
    assert len(np.unique(list(list(zip(*val_unseen))[1]))) == (args.val_unseen)

    val_seen_idx = np.random.choice([i for i in range(len(rest))], int(np.floor(len(rest) * args.trainval_portion)), replace=False)
    val_seen_paths = np.array(list(zip(*rest))[0])[val_seen_idx]
    val_seen_lbls = np.array(list(zip(*rest))[1])[val_seen_idx]

    val_seen = list(zip(val_seen_paths, val_seen_lbls))

    print(val_seen[0])

    train = []
    for pair in rest:
        if pair not in val_seen:
            train.append(pair)

    print('number of labels in val_seen:', len(np.unique(list(list(zip(*val_seen))[1]))))
    assert len(np.unique(list(list(zip(*val_seen))[1]))) == 150 - (args.val_unseen)
    assert len(np.unique(list(list(zip(*train))[1]))) == 150 - (args.val_unseen)

    output_string = ''

    print("unseen test:", len(test_unseen))
    output_string += "known test: " + str(len(test_unseen)) + '\n'

    print("seen test: ", len(test_seen))
    output_string += "unknown test: " + str(len(test_seen)) + '\n'

    print("Total test set:", len(test_seen) + len(test_unseen))
    output_string += "Total test set: " + str((len(test_seen) + len(test_unseen))) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("unseen val: ", len(val_unseen))
    output_string += "unknown val: " + str(len(val_unseen)) + '\n'

    print("seen val:", len(val_seen))
    output_string += "known val: " + str(len(val_seen)) + '\n'

    print("Total val set:", len(val_unseen) + len(val_seen))
    output_string += "Total val set: " + str(len(val_unseen) + len(val_seen)) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("Train set:", len(train))
    output_string += "Total train set: " + str(len(train)) + '\n'

    print('*' * 30)
    output_string += ('*' * 30) + '\n'

    print("Total trainval set:", len(train) + len(val_seen) + len(val_unseen))
    output_string += "Total trainval set: " + str(len(train) + len(val_seen) + len(val_unseen)) + '\n'

    input()

    if not os.path.exists(os.path.join(data_path, f'final_newsplits{args.version}_{args.save_version}')):
        data_path = os.path.join(data_path, f'final_newsplits{args.version}_{args.save_version}')
        os.mkdir(data_path)
    else:
        data_path = os.path.join(data_path, f'final_newsplits{args.version}_{args.save_version}')

    with open(os.path.join(data_path, 'config.txt'), 'w') as f:
        f.write(output_string)

    save_dataset(train, data_path, 'train')
    save_dataset(val_seen, data_path, 'val_seen')
    save_dataset(val_unseen, data_path, 'val_unseen')
    save_dataset(test_seen, data_path, 'test_seen')
    save_dataset(test_unseen, data_path, 'test_unseen')

    _number_of_classes(data_path)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-sp', '--split_path', default='CUB_splits/',help="split path")
    parser.add_argument('-dp', '--data_path', default='../../dataset/CUB/images/', help="data path")
    parser.add_argument('-v', '--version', type=int, default=0, help="version")
    parser.add_argument('-sv', '--save_version', type=int, default=1, help="save_version")
    parser.add_argument('-tvp', '--trainval_portion', type=float, default=0.2, help="percentage of seen val to trainval")
    parser.add_argument('-vu', '--val_unseen', type=int, default=25, help="number of unseen val classes")

    args = parser.parse_args()

    # load_cub_data(args)
    # make_from_new_split(args)

    _number_of_classes('../../dataset/CUB/images/final_newsplits0_1')



# https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html

if __name__ == '__main__':
    main()
