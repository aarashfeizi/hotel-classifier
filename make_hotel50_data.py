import argparse
import os
import statistics

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
import random

def _check_dir(path):
    return os.path.isdir(path)


def get_stats(l):
    length = len(l)
    mean = l.mean()
    median = statistics.median(l)
    max = l.max()
    min = l.min()
    u_v = len(np.unique(l))

    return {'length': length, 'mean': mean, 'median': median, 'max': max, 'min': min, 'unique_values': u_v}


def load_hotels_data(path, directories=['train', 'test'],
                     path_directories={'train': 'train', 'test': 'test/unoccluded'}, maps=None, name='train'):
    hotel_label_list = []
    cam_web_list = []
    image_list = []
    super_class_list = []
    is_trainval_list = []

    org_path = path

    if os.path.exists(os.path.join(path, f'hotel50-image_label_{name}.csv')):
        print('hehe')
        print('Found csv!')

        dataset = pd.read_csv(os.path.join(path, f'hotel50-image_label_{name}.csv'))

    else:
        print('File not found, creating csv...')

        if maps is None:
            print('No maps found :(')
            hotels_chain_branch2lbl = {}
            hotels_chain2lbl = {}
        else:
            print('Maps found!!')
            hotels_chain_branch2lbl = json.load(open(os.path.join(path, maps[0])))
            hotels_chain2lbl = json.load(open(os.path.join(path, maps[1])))

            # hotels_chain_branch2lbl = {int(x): y for x, y in hotels_chain_branch2lbl.items()}
            # hotels_chain2lbl = {int(x): y for x, y in hotels_chain2lbl.items()}

        label = 0
        super_class = 0

        for dir in directories:

            if dir == 'train':
                is_trainval = 1
            else:
                is_trainval = 0

            path = os.path.join(org_path, f'images/{path_directories[dir]}/')

            fst_l_d = os.listdir(path)  # e.g. 1 10 11 12
            length = len(fst_l_d)
            for i, f_dir in enumerate(fst_l_d):

                percent = (1.0 * (i + 1)) / length

                scd_path = os.path.join(path, f_dir)
                print(percent, scd_path)

                if not _check_dir(scd_path):
                    continue

                if f_dir not in hotels_chain_branch2lbl.keys():
                    if dir == 'test':
                        raise Exception(f"{f_dir} not in hotels_chain_branch2lbl keys!")
                    hotels_chain_branch2lbl[f_dir] = {}
                    hotels_chain2lbl[f_dir] = super_class
                    super_class += 1

                scd_l_d = os.listdir(scd_path)  # e.g. 9645 20303 3291 35913

                for s_dir in scd_l_d:  # All same super_class

                    thd_path = os.path.join(scd_path, s_dir)

                    if not _check_dir(thd_path):
                        continue

                    if s_dir not in hotels_chain_branch2lbl[f_dir].keys():
                        if dir == 'test':
                            raise Exception(f"{s_dir} not in in hotels_chain_branch2lbl[{f_dir}] keys!")
                        hotels_chain_branch2lbl[f_dir][s_dir] = label
                        label += 1

                    thd_l_d = os.listdir(thd_path)  # e.g. traffickcam travel_website

                    for t_dir in thd_l_d:  # all same labels

                        imagedir_path = os.path.join(thd_path, t_dir)

                        if not _check_dir(imagedir_path):
                            continue

                        if t_dir == 'travel_website':
                            is_website = 1
                        elif t_dir == 'traffickcam':
                            is_website = 0
                        else:
                            print(imagedir_path)
                            raise Exception('FUCK')

                        if not _check_dir(imagedir_path):
                            continue

                        images = os.listdir(imagedir_path)  # e.g. *.jpg

                        for image in images:
                            image_path = os.path.join(imagedir_path, image)
                            image_list.append(image_path[image_path.find('images'):])
                            hotel_label_list.append(hotels_chain_branch2lbl[f_dir][s_dir])
                            cam_web_list.append(is_website)
                            super_class_list.append(hotels_chain2lbl[f_dir])
                            is_trainval_list.append(is_trainval)

        dataset = pd.DataFrame({'image': image_list, 'hotel_label': hotel_label_list, 'super_class': super_class_list,
                                'is_website': cam_web_list, 'is_trianval': is_trainval_list})
        # dataset.to_csv(os.path.join(org_path, 'hotel50-image_label_train_test_merged.csv'), index=False, header=True)
        dataset.to_csv(os.path.join('.', f'hotel50-image_label_{name}.csv'), index=False, header=True)
        print(os.path.join(path, 'hotels_chain_branch2lbl.json'))
        if name == 'train':
            json.dump(hotels_chain_branch2lbl, open(os.path.join(org_path, 'hotels_chain_branch2lbl.json'), 'w'))
            json.dump(hotels_chain2lbl, open(os.path.join(org_path, 'hotels_branch2lbl.json'), 'w'))

    return dataset


def get_size(df, path):
    sizes = []
    types = []
    for idx, row in df.iterrows():
        img = Image.open(os.path.join(path, row[0]))
        sizes.append(img.size)
        types.append(img.mode)
        if idx % 1000 == 0:
            print(get_stats(np.array(list(zip(*sizes))[1])))

    return sizes, types


def plot_sizes(args, df):
    sizes, types = get_size(df, args.path)

    df['shape0'] = np.array(list(zip(*sizes))[0])
    df['shape1'] = np.array(list(zip(*sizes))[1])
    df['channel'] = np.array(types)

    df.to_csv(os.path.join(args.path, 'hotel50-image_label.csv'), index=False, header=True)

    print('shape 0')
    print(get_stats(np.array(list(zip(*sizes))[0])))
    print('*' * 70)
    print('shape 1')
    print(get_stats(np.array(list(zip(*sizes))[1])))

    sizes_str = list(map(lambda x: str(x), sizes))
    sizes_unq, sizes_cnt = np.unique(sizes_str, return_counts=True)
    sizes_cnt_n = sizes_cnt / sizes_cnt.max()
    sizes_dict = {size: count for size, count in zip(sizes_unq, sizes_cnt)}
    sizes_dict_n = {size: count for size, count in zip(sizes_unq, sizes_cnt_n)}

    areas = [sizes_dict_n[str(p)] * 300 for p in sizes]
    cmaps = [sizes_dict[str(p)] for p in sizes]

    plt.scatter(x=np.array(list(zip(*sizes))[0]), y=np.array(list(zip(*sizes))[1]), s=areas, alpha=0.4,
                cmap='viridis',
                c=cmaps)
    plt.title('Image Size Dist')
    plt.xlabel('Shape 0')
    plt.ylabel('Shape 1')
    plt.colorbar(label='Count')

    plt.savefig('final_2.png')


def sample_pairs(pairs, portion=0.33):
    pairs_idx = np.random.choice([i for i in range(len(pairs))], int(np.floor(len(pairs) * portion)), replace=False)
    pairs_paths = np.array(list(zip(*pairs))[0])[pairs_idx]
    pairs_lbls = np.array(list(zip(*pairs))[1])[pairs_idx]

    return list(zip(pairs_paths, pairs_lbls))


def lbl_in_pairs(pairs):
    pairs_lbls = list(list(zip(*pairs))[1])
    pairs_lbls_u, pairs_lbls_c = np.unique(pairs_lbls, return_counts=True)

    return pairs_lbls_u, pairs_lbls_c


def create_splits(args, df):
    lbls = df.hotel_label
    paths = df.image

    all_pairs = list(zip(paths, lbls))

    lbls_unique, lbls_count = np.unique(lbls, return_counts=True)

    lbls_unique_filtered = lbls_unique[lbls_count >= args.threshold]

    lbls_unique_less_than_thresh = np.array(list(set(lbls_unique) - set(lbls_unique_filtered)), dtype=np.int64)

    val_to_add = []
    test_to_add = []

    for l in lbls_unique_less_than_thresh:
        if random.random() < 0.6:
            test_to_add.append(l)
        else:
            val_to_add.append(l)

    pairs = []
    filtered_out_pairs = []

    for p, l in all_pairs:
        if l in lbls_unique_filtered:
            pairs.append((p, l))
        else:
            filtered_out_pairs.append((p, l))

    test_unseen_lbls_u = np.random.choice(lbls_unique_filtered, args.test_unseen, replace=False)

    test_unseen_lbls_u = np.concatenate([test_to_add, test_unseen_lbls_u])

    trainval_lbls = np.array(list(set(lbls_unique_filtered) - set(test_unseen_lbls_u)), dtype=np.int64)

    test_unseen = []
    for pair in pairs:
        if pair[1] in test_unseen_lbls_u:
            test_unseen.append(pair)

    for pair in filtered_out_pairs:
        if pair[1] in test_unseen_lbls_u:
            test_unseen.append(pair)

    old_len = len(pairs)
    pairs = list(set(pairs) - set(test_unseen))

    assert old_len > len(pairs)

    pairs_dict = {}

    for p, l in pairs:
        if l not in pairs_dict.keys():
            pairs_dict[l] = []
        pairs_dict[l].append(p)

    test_seen = []

    for l, paths in pairs_dict.items():
        chosen = np.random.choice(paths, 5, replace=False)
        test_seen.extend([(p, l) for p in chosen])
        paths = list(set(paths) - set(chosen))
        pairs_dict[l] = paths

    trainval_pairs = []

    for l, ps in pairs_dict.items():
        for p in ps:
            trainval_pairs.append((p, l))

    # test_unseen_lbls_u, test_unseen_lbls_c = lbl_in_pairs(test_unseen)

    val_unseen_lbls_u = np.random.choice(trainval_lbls, args.val_unseen, replace=False)

    val_unseen_lbls_u = np.concatenate([val_to_add, val_unseen_lbls_u])

    val_unseen = []

    for pair in trainval_pairs:
        if pair[1] in val_unseen_lbls_u:
            val_unseen.append(pair)

    for pair in filtered_out_pairs:
        if pair[1] in val_unseen_lbls_u:
            val_unseen.append(pair)

    trainval_pairs = list(set(trainval_pairs) - set(val_unseen))

    pairs_dict = {}

    for p, l in trainval_pairs:
        if l not in pairs_dict.keys():
            pairs_dict[l] = []
        pairs_dict[l].append(p)

    val_seen = []

    for l, paths in pairs_dict.items():
        chosen = np.random.choice(paths, 5, replace=False)
        val_seen.extend([(p, l) for p in chosen])
        paths = list(set(paths) - set(chosen))
        pairs_dict[l] = paths

    train = []

    for l, ps in pairs_dict.items():
        for p in ps:
            train.append((p, l))

    output_string = ''

    output_string += "unseen test: " + str(len(test_unseen)) + '\n'
    _, c = lbl_in_pairs(test_unseen)
    output_string += "unseen test min in classes: " + str(min(c)) + '\n'
    output_string += "unseen test max in classes: " + str(max(c)) + '\n'

    output_string += "seen test: " + str(len(test_seen)) + '\n'
    _, c = lbl_in_pairs(test_seen)
    output_string += "seen test min in classes: " + str(min(c)) + '\n'
    output_string += "seen test max in classes: " + str(max(c)) + '\n'

    output_string += "Total test set: " + str((len(test_seen) + len(test_unseen))) + '\n'

    output_string += ('*' * 30) + '\n'

    output_string += "unseen val: " + str(len(val_unseen)) + '\n'
    _, c = lbl_in_pairs(val_unseen)
    output_string += "unseen val min in classes: " + str(min(c)) + '\n'
    output_string += "unseen val max in classes: " + str(max(c)) + '\n'

    output_string += "seen val: " + str(len(val_seen)) + '\n'
    _, c = lbl_in_pairs(val_seen)
    output_string += "seen val min in classes: " + str(min(c)) + '\n'
    output_string += "seen val max in classes: " + str(max(c)) + '\n'

    output_string += "Total val set: " + str(len(val_unseen) + len(val_seen)) + '\n'

    output_string += ('*' * 30) + '\n'

    output_string += "Total train set: " + str(len(train)) + '\n'
    _, c = lbl_in_pairs(train)
    output_string += "train min in classes: " + str(min(c)) + '\n'
    output_string += "train max in classes: " + str(max(c)) + '\n'

    output_string += ('*' * 30) + '\n'

    output_string += "Total trainval set: " + str(len(train) + len(val_seen) + len(val_unseen)) + '\n'

    print(output_string)

    input()

    data_path = '.'

    if not os.path.exists(os.path.join(data_path, f'splits{args.version}_{args.save_version}')):
        data_path = os.path.join(data_path, f'splits{args.version}_{args.save_version}')
        os.mkdir(data_path)
    else:
        data_path = os.path.join(data_path, f'splits{args.version}_{args.save_version}')

    with open(os.path.join(data_path, 'config.txt'), 'w') as f:
        f.write(output_string)

    save_dataset(train, data_path, 'train')
    save_dataset(val_seen, data_path, 'val_seen')
    save_dataset(val_unseen, data_path, 'val_unseen')
    save_dataset(test_seen, data_path, 'test_seen')
    save_dataset(test_unseen, data_path, 'test_unseen')

    class_config = _number_of_classes(data_path)

    with open(os.path.join(data_path, 'class_config.txt'), 'w') as f:
        f.write(class_config)


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


    s = ''

    for name, csv in zip(names, csvs):
        print(name, "number of classes:", len(np.unique(list(csv.label))))
        s += name + " number of classes: " + str(len(np.unique(list(csv.label)))) + '\n'
    return s


def save_dataset(ls, data_path, name):
    df = pd.DataFrame({'image': list(list(zip(*ls))[0]), 'label': list(list(zip(*ls))[1])})
    df.to_csv(os.path.join(data_path, f'hotels_{name}.csv'), index=False, header=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', default='../../dataset/hotels/', help="path")
    parser.add_argument('-th', '--threshold', type=int, default=15, help="threshold")
    parser.add_argument('-tu', '--test_unseen', type=int, default=4000, help="unseen_test")
    parser.add_argument('-vu', '--val_unseen', type=int, default=4000, help="unseen_val")
    parser.add_argument('-v', '--version', type=int, default=0, help="version")
    parser.add_argument('-hcb2l', '--hotels_chain_branch2lbl', default='hotels_chain_branch2lbl.json', help="hotels_chain_branch2lbl path")
    parser.add_argument('-hb2l', '--hotels_branch2lbl', default='hotels_branch2lbl.json', help="hotels_branch2lbl path")
    parser.add_argument('-sv', '--save_version', type=int, default=1, help="save_version")

    args = parser.parse_args()

    train_df = load_hotels_data(args.path, directories=['train'], name='train')
    test_df = load_hotels_data(args.path, directories=['test'], maps=(args.hotels_chain_branch2lbl,
                                                                      args.hotels_branch2lbl), name='test')

    all = [train_df, test_df]
    df = pd.concat(all)

    # print(len(df))

    create_splits(args, df)

    # plot_sizes(args, df)


# https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html

if __name__ == '__main__':
    main()

###
# plt.figure(figsize=(100, 100))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 100}
# matplotlib.rc('font', **font)
# sizes_str = list(map(lambda x: str(x), sizes))
# sizes_unq, sizes_cnt = np.unique(sizes_str, return_counts=True)
# sizes_cnt_n = sizes_cnt / sizes_cnt.max()
# sizes_dict = {size: count for size, count in zip(sizes_unq, sizes_cnt)}
# sizes_dict_n = {size: count for size, count in zip(sizes_unq, sizes_cnt_n)}
# areas = [sizes_dict_n[str(p)] * 50000 for p in sizes]
# cmaps = [sizes_dict[str(p)] for p in sizes]
# plt.scatter(x=np.array(list(zip(*sizes))[0]), y=np.array(list(zip(*sizes))[1]), s=areas, alpha=0.4,
#             cmap='viridis',
#             c=cmaps)
# plt.title('Image Size Dist')
# plt.xlabel('Shape 0')
# plt.ylabel('Shape 1')
# plt.colorbar(label='Count')
# plt.savefig('final_2.png')
###
