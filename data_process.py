# -*- coding: utf-8 -*-
# @Time : 2020/8/7 14:15
# @Author : nicahead@gmail.com
# @File : data_process.py
# 数据预处理

import os
import re

train_dir = 'data/train'
test_dir = 'data/test'


def get_entities(dir):
    """
    统计初始语料各实体数量
    :param dir: 初始语料目录
    :return: 实体统计字典
    """
    entities = {}  # 存储所有的实体名
    files = os.listdir(dir)  # 所有文件名
    files = list(set([file.split('.')[0] for file in files]))  # 所有不重复的文件名
    for file in files:
        path = os.path.join(train_dir, file + '.ann')
        # 读取文件的每一行 T1	Disease 1845 1850	1型糖尿病
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                name = line.split('\t')[1].split(' ')[0]
                # 统计实体个数
                if name in entities:
                    entities[name] += 1
                else:
                    entities[name] = 1
    return entities


def get_label_encoder(entities):
    """
    标签编码，训练的时候需要将字符串的BIO标签转为id
    :param entities: 实体统计字典
    :return: id到实体名的映射、实体名到id的映射
    """
    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)  # 根据实体数量排序
    entities = [entity[0] for entity in entities]  # 实体名
    id2label = []  # id到实体名的映射
    # BIO标注模式
    id2label.append('O')
    for entity in entities:
        id2label.append('B-' + entity)
        id2label.append('I-' + entity)
    label2id = {id2label[i]: i for i in range(len(id2label))}  # 实体名到id的映射
    return id2label, label2id


def ischinese(char):
    """
    判断一个字是否为中文
    :param char:
    :return:
    """
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def split_text(text):
    """
    将文本切分成序列
    :param text:
    :return:
    """
    split_index = []
    pattern1 = '。|，|,|;|；|\.|\?'
    for m in re.finditer(pattern1, text):
        idx = m.span()[0]
        if text[idx - 1] == '\n':
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():  # 前后是数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx + 2].isdigit():  # 前数字 后空格 后后数字
            continue
        if text[idx - 1].islower() and text[idx + 1].islower():  # 前小写字母后小写字母
            continue
        if text[idx - 1].islower() and text[idx + 1].isdigit():  # 前小写字母后数字
            continue
        if text[idx - 1].isupper() and text[idx + 1].isdigit():  # 前大写字母后数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():  # 前数字后小写字母
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():  # 前数字后大写字母
            continue
        if text[idx + 1] in set('.。;；,，'):  # 前句号后句号
            continue
        if text[idx - 1].isspace() and text[idx - 2].isspace() and text[idx - 3] == 'C':  # HBA1C的问题
            continue
        if text[idx - 1].isspace() and text[idx - 2] == 'C':
            continue
        if text[idx - 1].isupper() and text[idx + 1].isupper():  # 前大些后大写
            continue
        if text[idx] == '.' and text[idx + 1:idx + 4] == 'com':  # 域名
            continue
        # 以上情况都不满足，则切开
        split_index.append(idx + 1)
    pattern2 = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern2 += '注:|附录 |表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d|'
    pattern2 += '\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    pattern2 += 'and |or |with |by |because of |as well as '
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with') \
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
            continue
        split_index.append(idx)

    pattern3 = '\n\d\.'  # 匹配1.  2.  这些序号
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if ischinese(text[idx + 3]):
            split_index.append(idx + 1)

    for m in re.finditer('\n\(\d\)', text):  # 匹配(1) (2)这样的序号
        idx = m.span()[0]
        split_index.append(idx + 1)
    split_index = list(sorted(set([0, len(text)] + split_index)))

    other_index = []
    for i in range(len(split_index) - 1):
        begin = split_index[i]
        end = split_index[i + 1]
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin] == '(' and text[begin + 1] in '一二三四五六七八九零十'):  # 如果是一、和(一)这样的标号
            for j in range(begin, end):
                if text[j] == '\n':
                    other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    other_index = []
    for i in range(len(split_index) - 1):  # 对长句子进行拆分
        b = split_index[i]
        e = split_index[i + 1]
        other_index.append(b)
        if e - b > 150:
            for j in range(b, e):
                if (j + 1 - other_index[-1]) > 15:  # 保证句子长度在15以上
                    if text[j] == '\n':
                        other_index.append(j + 1)
                    if text[j] == ' ' and text[j - 1].isnumeric() and text[j + 1].isnumeric():
                        other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    for i in range(1, len(split_index) - 1):  # 10   20  干掉全部是空格的句子
        idx = split_index[i]
        while idx > split_index[i - 1] - 1 and text[idx - 1].isspace():
            idx -= 1
        split_index[i] = idx
    split_index = list(sorted(set([0, len(text)] + split_index)))

    # 处理短句子
    temp_idx = []
    i = 0
    while i < len(split_index) - 1:  # 0 10 20 30 45
        b = split_index[i]
        e = split_index[i + 1]

        num_ch = 0
        num_en = 0
        if e - b < 15:
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch += 1
                elif ch.islower() or ch.isupper():
                    num_en += 1
                if num_ch + 0.5 * num_en > 5:  # 如果汉字加英文超过5个  则单独成为句子
                    temp_idx.append(b)
                    i += 1
                    break
            if num_ch + 0.5 * num_en <= 5:  # 如果汉字加英文不到5个  和后面一个句子合并
                temp_idx.append(b)
                i += 2
        else:
            temp_idx.append(b)
            i += 1
    split_index = list(sorted(set([0, len(text)] + temp_idx)))
    result = []
    for i in range(len(split_index) - 1):
        result.append(text[split_index[i]:split_index[i + 1]])

    # 做一个检查
    s = ''
    for r in result:
        s += r
    assert len(s) == len(text)
    return result


if __name__ == '__main__':
    files = os.listdir(train_dir)
    files = list(set([file.split('.')[0] for file in files]))
    path = os.path.join(train_dir, files[1] + '.txt')
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()
        print(split_text(text))
