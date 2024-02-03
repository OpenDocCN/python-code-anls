# `jieba\jieba\finalseg\__init__.py`

```py
# 导入必要的模块
from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

# 定义最小浮点数
MIN_FLOAT = -3.14e100

# 定义概率文件名
PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

# 定义前一个状态
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

# 定义需要强制拆分的词语集合
Force_Split_Words = set([])

# 加载模型
def load_model():
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p

# 根据不同平台加载模型
if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P

# 维特比算法
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in xrange(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    return (prob, path[state])

# 分词函数
def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]
# 编译正则表达式，匹配中文字符
re_han = re.compile("([\u4E00-\u9FD5]+)")
# 编译正则表达式，匹配英文字母、数字和百分号
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")

# 添加强制分词的词语到全局变量 Force_Split_Words 中
def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)

# 对输入的句子进行分词处理
def cut(sentence):
    # 将句子转换为字符串编码
    sentence = strdecode(sentence)
    # 使用正则表达式对句子进行分块处理
    blocks = re_han.split(sentence)
    # 遍历每个分块
    for blk in blocks:
        # 如果分块中包含中文字符
        if re_han.match(blk):
            # 对中文分块进行分词处理
            for word in __cut(blk):
                # 如果分词结果不在强制分词词语列表中，则返回分词结果
                if word not in Force_Split_Words:
                    yield word
                # 如果分词结果在强制分词词语列表中，则逐个返回分词结果的每个字符
                else:
                    for c in word:
                        yield c
        # 如果分块中不包含中文字符
        else:
            # 使用正则表达式对非中文分块进行分割处理
            tmp = re_skip.split(blk)
            # 遍历分割后的结果
            for x in tmp:
                # 如果分割结果不为空，则返回该结果
                if x:
                    yield x
```