# `.\SentEval\senteval\utils.py`

```py
# 引入未来版本的特性允许 Python 2/3 兼容
from __future__ import absolute_import, division, unicode_literals

# 引入需要的库
import numpy as np  # 导入数值计算库 numpy
import re  # 导入正则表达式库 re
import inspect  # 导入检查模块信息的 inspect
from torch import optim  # 从 torch 库中导入优化器模块 optim

# 定义函数 create_dictionary，接受一个句子列表作为参数
def create_dictionary(sentences):
    words = {}  # 创建空字典 words
    # 遍历句子列表
    for s in sentences:
        # 遍历句子中的单词
        for word in s:
            # 如果单词已经在字典中，增加其出现次数；否则将其添加到字典并置计数为1
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    # 将特殊符号添加到字典中，赋予它们较高的计数以确保排序时排在前面
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1  # 可选的未知单词标记
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # 对字典按值（计数）进行逆排序
    id2word = []  # 创建空列表 id2word，用于存储单词索引到单词的映射关系
    word2id = {}  # 创建空字典 word2id，用于存储单词到索引的映射关系
    # 遍历排序后的单词列表
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)  # 将单词添加到 id2word 列表中
        word2id[w] = i  # 将单词映射到其索引的字典中
    # 返回单词索引到单词的列表和单词到索引的字典
    return id2word, word2id


# 定义函数 cosine，计算两个向量的余弦相似度
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# 定义 dotdict 类，继承自 dict，实现点号访问字典属性
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# 定义函数 get_optimizer，解析优化器参数并返回相应的优化器函数和参数字典
def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    # 如果参数字符串包含逗号
    if "," in s:
        method = s[:s.find(',')]  # 提取优化方法名
        optim_params = {}  # 创建空字典，用于存储优化器参数
        # 遍历以逗号分隔的参数部分
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')  # 根据等号分隔参数名和参数值
            assert len(split) == 2  # 确保分隔结果为参数名和参数值两部分
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None  # 使用正则表达式检查参数值格式
            optim_params[split[0]] = float(split[1])  # 将参数名和参数值添加到参数字典中
    else:
        method = s  # 如果参数字符串不包含逗号，则整个字符串为优化方法名
        optim_params = {}  # 空字典

    # 根据优化方法名选择相应的优化器函数
    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params  # 确保 SGD 方法中包含学习率参数
    else:
        raise Exception('Unknown optimization method: "%s"' % method)  # 抛出异常，未知的优化方法名

    # 检查优化器函数初始化时的参数
    expected_args = inspect.getargspec(optim_fn.__init__)[0]  # 获取优化器函数初始化方法的参数列表
    assert expected_args[:2] == ['self', 'params']  # 确保参数列表的前两个参数为 self 和 params
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))  # 检查传入的参数是否符合预期

    return optim_fn, optim_params  # 返回选择的优化器函数和参数字典
```