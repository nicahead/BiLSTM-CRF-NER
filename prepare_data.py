# -*- coding: utf-8 -*-
# @Time      :     2020/8/7 15:27
# @Author    :     nicahead@gmail.com
# @File      :     prepare_data.py
# @Desc      :     为模型准备数据

import os
import pickle
import pandas as pd
from collections import Counter
from data_process import split_text
import jieba.posseg as psg
from cnradical import Radical, RunOption
import shutil
from random import shuffle

init_dir = 'data/init'


def process_text(idx, split_method=None, split_name='init'):
    """
    读取文本，切割，打上标记，并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx:文件名，不含拓展名
    :param split_method: 切割文本的函数
    :param split_name: 最终保存的文件夹名字
    :return:
    """
    data = {}  # 保存提取到的特征

    # ------------------获取句子-----------------------
    if split_method is None:
        with open(f'{init_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'{init_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_method(texts)
    data['word'] = texts

    # ------------------获取标签-----------------------
    tag_list = ['O' for s in texts for w in s]  # 双重循环，初始情况下将每个字都标为O
    tag = pd.read_csv(f'{init_dir}/{idx}.ann', header=None, sep='\t')
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ')  # 获取的实体类别以及起始位置
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])  # 转换成对应的类型
        tag_list[start] = 'B-' + cls  # 起始位置写入B-实体类别
        for j in range(start + 1, end):  # 后面的位置写I-实体类别
            tag_list[j] = 'I-' + cls
    assert len([x for s in texts for x in s]) == len(tag_list)  # 保证两个序列长度一致

    # ------------------获取词性和词边界特征-----------------------
    word_bounds = ['M' for item in tag_list]  # 词边界 BMES标注 首先给所有的字都表上B标记
    word_flags = []  # 用来保存每个字的词性
    for sentenc in texts:
        for word, flag in psg.cut(sentenc):
            # 单个字作为词
            if len(word) == 1:
                start = len(word_flags)  # 拿到起始下标
                word_bounds[start] = 'S'  # 标记修改为S
                word_flags.append(flag)  # 将当前词的词性名加入到wordflags列表
            # 多个字作为词
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'  # 第一个字打上B
                word_flags += [flag] * len(word)  # 将这个词的每个字都加上词性标记
                end = len(word_flags) - 1
                word_bounds[end] = 'E'  # 将最后一个字打上E标记

    # ------------------获取拼音、偏旁特征-----------------------
    radical = Radical(RunOption.Radical)  # 提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)  # 用来提取拼音
    # 提取偏旁部首特征  对于没有偏旁部首的字标上PAD
    data['radical'] = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # 提取拼音特征  对于没有拼音的字标上PAD
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]

    # 统一处理，将标签、词边界特征、词性与句子序列对应切分
    tags = []
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        l = len(s)
        end += l
        tags.append(tag_list[start:end])
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += l
    data['label'] = tags
    data['bounds'] = bounds
    data['flags'] = flags

    # ------------------将处理的单个文件的数据保存下来-----------------------
    num_samples = len(texts)  # 获取有多少句话  等于是有多少个样本
    num_col = len(data.keys())  # 获取特征的个数 也就是列数

    dataset = []  # 保存处理后的结果

    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))  # 解压
        dataset += records + [['sep'] * num_col]  # 每存完一个句子需要一行sep进行隔离
    dataset = dataset[:-1]  # 最后一行sep不要
    dataset = pd.DataFrame(dataset, columns=data.keys())  # 转换成dataframe
    save_path = f'data/prepare/{split_name}/{idx}.csv'

    def clean_word(w):
        if w == '\n':
            return 'LB'
        if w in [' ', '\t', '\u2003']:
            return 'SPACE'
        if w.isdigit():  # 将所有的数字都变成一种符号
            return 'num'
        return w

    dataset['word'] = dataset['word'].apply(clean_word)
    dataset.to_csv(save_path, index=False, encoding='utf-8')


def multi_process(split_method=None, train_ratio=0.8):
    """
    将所有的初始标注文件处理为需要的样本格式
    :param split_method: 切割文本的函数
    :param train_ratio: 训练集比例
    :return:
    """
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')
    if not os.path.exists('data/prepare/train/'):
        os.makedirs('data/prepare/train')
        os.makedirs('data/prepare/test')
    idxs = list(set([file.split('.')[0] for file in os.listdir(init_dir)]))  # 获取所有文件的名字
    shuffle(idxs)  # 打乱顺序
    index = int(len(idxs) * train_ratio)  # 拿到训练集的截止下标
    train_ids = idxs[:index]  # 训练集文件名集合
    test_ids = idxs[index:]  # 测试集文件名集合

    import multiprocessing as mp
    num_cpus = mp.cpu_count()  # 获取机器cpu的个数
    pool = mp.Pool(num_cpus)
    results = []
    for idx in train_ids:
        result = pool.apply_async(process_text, args=(idx, split_method, 'train'))
        results.append(result)
    for idx in test_ids:
        result = pool.apply_async(process_text, args=(idx, split_method, 'test'))
        results.append(result)
    pool.close()
    pool.join()


def mapping(data, threshold=10, is_word=False, sep='sep', is_label=False):
    """
    根据传入的列表得到两个映射关系，id2item和item2id
    :param data: 数据列表
    :param threshold: 如果需要，则去掉频率小于threshold的元素
    :param is_word: 传入的是否为字的列表
    :param sep: 分隔符
    :param is_label: 传入的是否为实体标记的列表
    :return: 列表id2item,字典item2id
    """
    count = Counter(data)
    if sep is not None:
        count.pop(sep)
    if is_word:
        count['PAD'] = 100000001
        count['UNK'] = 100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data if x[1] >= threshold]  # 去掉频率小于threshold的元素  未登录词
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item, item2id


def get_dict():
    """
    获取样本数据的字典，保存到本地
    :return:
    """
    map_dict = {}
    from glob import glob
    all_w, all_bounds, all_flags, all_label, all_radical, all_pinyin = [], [], [], [], [], []
    for file in glob('data/prepare/train/*.csv') + glob('data/prepare/test/*.csv'):
        df = pd.read_csv(file, sep=',')
        all_w += df['word'].tolist()
        all_bounds += df['bounds'].tolist()
        all_flags += df['flags'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    map_dict['word'] = mapping(all_w, threshold=20, is_word=True)
    map_dict['bounds'] = mapping(all_bounds)
    map_dict['flags'] = mapping(all_flags)
    map_dict['label'] = mapping(all_label, is_label=True)
    map_dict['radical'] = mapping(all_radical)
    map_dict['pinyin'] = mapping(all_pinyin)

    # 写入字典
    with open(f'data/prepare/dict.pkl', 'wb') as f:
        pickle.dump(map_dict, f)


if __name__ == '__main__':
    multi_process(split_method=split_text)
    get_dict()
    # with open(f'data/prepare/dict.pkl', 'rb') as f:
    #     dic = pickle.load(f)
    # print(dic['label'])
