# -*- coding: utf-8 -*-
# @Time      :   2020/8/9 12:29
# @Author    :   nicahead@gmail.com
# @File      :   data_utils.py
# @Desc      :   数据增强
import math
import os
import pickle
import random

import pandas as pd

from tqdm import tqdm


def get_data_with_windows(name='train'):
    """
    根据字典将数据编码，并保存至train.pkl
    :param name: 训练数据/测试数据，保存至不同文件夹
    :return:
    """
    # 加载字典
    with open(f'data/prepare/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    # 内部函数，将传入的data（列表）编码为id列表
    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('data/prepare', name)
    files = os.listdir(root)

    for file in tqdm(files):
        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=',')
        num_samples = len(samples)
        sep_index = [-1] + samples[samples['word'] == 'sep'].index.tolist() + [num_samples]  # 所有使用sep分割开的下标
        # 获取所有句子并将句子全都转换成id
        for i in range(len(sep_index) - 1):
            start = sep_index[i] + 1
            end = sep_index[i + 1]
            data = []
            # 对于所有的特征都执行这个操作
            for feature in samples.columns:
                data.append(item2id(list(samples[feature])[start:end], map_dict[feature][1]))
            result.append(data)
        # 去掉有些文件最后一句话是换行的句子
        if len(result[-1][0]) == 1:
            result = result[:-1]

        # 数据增强，两个句子组合/三个句子组合
        two = []
        for i in range(len(result) - 1):
            first = result[i]
            second = result[i + 1]
            two.append([first[k] + second[k] for k in range(len(first))])

        three = []
        for i in range(len(result) - 2):
            first = result[i]
            second = result[i + 1]
            third = result[i + 2]
            three.append([first[k] + second[k] + third[k] for k in range(len(first))])

        results.extend(result + two + three)

        with open(f'data/prepare/' + name + '.pkl', 'wb') as f:
            pickle.dump(results, f)


class BatchManager(object):
    def __init__(self, batch_size, name='train'):
        # 初始化数据，从本地读取
        with open(f'data/prepare/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    # 构造batch数据
    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))  # 总共有多少个batch
        sorted_data = sorted(data, key=lambda x: len(x[0]))  # 按照句子长度排序
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * int(batch_size): (i + 1) * int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        chars = []
        bounds = []
        flags = []
        radicals = []
        pinyins = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])  # len(data[-1][0])
        for line in data:
            char, bound, flag, target, radical, pinyin = line
            padding = [0] * (max_length - len(char))  # 不满max_length填充0
            chars.append(char + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            targets.append(target + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)
        return [chars, bounds, flags, radicals, pinyins, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


if __name__ == '__main__':
    pass
    # get_data_with_windows('train')
    # get_data_with_windows('test')
    # batchManager = BatchManager(10,'train')
