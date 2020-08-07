# BiLSTM-CRF-NER
b站项目https://www.bilibili.com/video/BV1Z5411477j，pytorch实现

## 数据集
[![aW3gUg.png](https://s1.ax1x.com/2020/08/07/aW3gUg.png)](https://imgchr.com/i/aW3gUg)

.txt文件，通过文字识别软件得到的关于疾病的描述

[![aW8pqK.png](https://s1.ax1x.com/2020/08/07/aW8pqK.png)](https://imgchr.com/i/aW8pqK)

.ann文件，标注的语料，实体的类别、在文本中的起始位置、实体内容

## 预处理
### 数据集处理
这种标注方式显然不适合我们使用，ner任务我们通常用BIO和BIOES等模式进行序列标注。
所以第一步我们先处理数据集。

**统计实体类别**
统计一共有多少种实体，每一种实体有多少个。得到所有BIO的标签

**构造实体字典**
得到实体标签和id的映射

## 文本预处理

**按标点符号断开**
lstm是一个序列一个序列进行处理的，所以需要对文档进行切分。

**过滤**

**长短句子处理**

## 数据准备

**获取标签**