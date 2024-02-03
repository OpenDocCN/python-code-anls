# `jieba\jieba\analyse\textrank.py`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from .tfidf import KeywordExtractor
from .._compat import *

# 定义一个无向加权图类
class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        # 初始化图的数据结构为 defaultdict(list)
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # 使用元组 (start, end, weight) 代替 Edge 对象
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        # 初始化节点权重字典和出度字典
        ws = defaultdict(float)
        outSum = defaultdict(float)

        # 计算默认节点权重
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # 为了构建稳定的迭代，对图的键进行排序
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 迭代10次
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        # 初始化最小和最大权重值
        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # 统一权重值，不乘以100
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws

# 定义 TextRank 类，继承自 KeywordExtractor 类
class TextRank(KeywordExtractor):

    def __init__(self):
        # 初始化分词器和词性标注器
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
        self.span = 5

    def pairfilter(self, wp):
        # 过滤词性和长度符合条件的词对
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)
    # 使用TextRank算法从句子中提取关键词
    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        # 将allowPOS转换为不可变集合
        self.pos_filt = frozenset(allowPOS)
        # 创建无向加权图对象
        g = UndirectWeightedGraph()
        # 创建默认字典
        cm = defaultdict(int)
        # 使用分词器对句子进行分词
        words = tuple(self.tokenizer.cut(sentence))
        # 遍历分词结果
        for i, wp in enumerate(words):
            # 过滤词对
            if self.pairfilter(wp):
                # 根据span范围处理词对
                for j in xrange(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    # 根据allowPOS和withFlag条件处理词对
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1

        # 遍历词对及其权重，添加到图中
        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        # 对节点进行排名
        nodes_rank = g.rank()
        # 根据条件对关键词进行排序
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        # 返回前topK个关键词
        if topK:
            return tags[:topK]
        else:
            return tags

    # 将textrank方法赋值给extract_tags，用于提取关键词
    extract_tags = textrank
```