import argparse
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


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


def load_hotels_data(path):
    hotel_label_list = []
    cam_web_list = []
    image_list = []
    super_class_list = []

    org_path = path

    if os.path.exists(os.path.join(path, 'hotel50-image_label.csv')):
        print('hehe')
        print('Found csv!')

        dataset = pd.read_csv(os.path.join(path, 'hotel50-image_label.csv'))

    else:
        print('File not found, creating csv...')

        path = os.path.join(path, 'images/train/')

        fst_l_d = os.listdir(path)  # e.g. 1 10 11 12

        label = 0
        super_class = 0

        for f_dir in fst_l_d:
            scd_path = os.path.join(path, f_dir)
            print(scd_path)

            if not _check_dir(scd_path):
                continue

            scd_l_d = os.listdir(scd_path)  # e.g. 9645 20303 3291 35913

            for s_dir in scd_l_d:  # All same super_class
                thd_path = os.path.join(scd_path, s_dir)

                if not _check_dir(thd_path):
                    continue

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
                        hotel_label_list.append(label)
                        cam_web_list.append(is_website)
                        super_class_list.append(super_class)

                label += 1
            super_class += 1

        dataset = pd.DataFrame({'image': image_list, 'hotel_label': hotel_label_list, 'super_class': super_class_list,
                                'is_website': cam_web_list})
        dataset.to_csv(os.path.join(org_path, 'hotel50-image_label.csv'), index=False, header=True)

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help="path")

    args = parser.parse_args()

    df = load_hotels_data(args.path)

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