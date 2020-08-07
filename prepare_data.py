# -*- coding: utf-8 -*-
# @Time : 2020/8/7 15:27
# @Author : nicahead@gmail.com
# @File : prepare_data.py
# 为模型准备数据

import os
import pandas as pd
from collections import Counter
from data_process import split_text
import jieba.posseg as psg
from cnradical import Radical, RunOption

train_dir = 'data/train'
test_dir = 'data/test'


def process_text(idx, split_method=None):
    """
    读取文本，切割，打上标记，并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx:文件名，不含拓展名
    :param split_method: 切割文本的函数
    :return:
    """
    data = {}  # 保存提取到的特征

    # ------------------获取句子-----------------------
    if split_method is None:
        with open(f'{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_method(texts)
    data['word'] = texts

    # ------------------获取标签-----------------------
    tag_list = ['O' for s in texts for w in s]  # 双重循环，初始情况下将每个字都标为O
    tag = pd.read_csv(f'{train_dir}/{idx}.ann', header=None, sep='\t')
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ')
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        tag_list[start] = 'B-' + cls
        for j in range(start + 1, end):
            tag_list[j] = 'I-' + cls

    # ------------------获取词性和词边界特征-----------------------
    word_bounds = ['M' for item in tag_list]  # 词边界 BMES标注
    word_flags = []  # 每个字的词性
    for sentenc in texts:
        for word, flag in psg.cut(sentenc):
            # 单个字作为词
            if len(word) == 1:
                start = len(word_flags)
                word_bounds[start] = 'S'
                word_flags.append(flag)
            # 多个字作为词
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)
                end = len(word_flags) - 1
                word_bounds[end] = 'E'

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

    return texts[0], tags[0], bounds[0], flags[0], data['pinyin'][0]


if __name__ == '__main__':
    print(process_text('0', split_method=split_text))
