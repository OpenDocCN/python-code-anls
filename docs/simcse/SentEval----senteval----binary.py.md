# `.\SentEval\senteval\binary.py`

```py
'''
Binary classifier and corresponding datasets : MR, CR, SUBJ, MPQA
'''
# 引入必要的库和模块
from __future__ import absolute_import, division, unicode_literals  # 允许绝对导入、整数除法和Unicode字面值

import io  # 输入输出流操作
import os  # 提供与操作系统交互的功能
import numpy as np  # 提供对多维数组对象的支持
import logging  # 提供日志记录功能

from senteval.tools.validation import InnerKFoldClassifier  # 从senteval工具包中导入InnerKFoldClassifier类


class BinaryClassifierEval(object):
    def __init__(self, pos, neg, seed=1111):
        self.seed = seed  # 初始化随机种子
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)  # 合并样本和标签列表
        self.n_samples = len(self.samples)  # 计算样本数量

    def do_prepare(self, params, prepare):
        # prepare函数用于处理整个文本
        return prepare(params, self.samples)
        # prepare函数将其输出放入"params"中：params.word2id等
        # 这些输出将被"batcher"进一步使用。

    def loadFile(self, fpath):
        # 加载文件并返回每行文本的分词列表
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def run(self, params, batcher):
        enc_input = []  # 初始化编码输入列表
        # 按长度和标签对数据进行排序，以减少填充
        sorted_corpus = sorted(zip(self.samples, self.labels),
                               key=lambda z: (len(z[0]), z[1]))
        sorted_samples = [x for (x, y) in sorted_corpus]  # 提取排序后的样本
        sorted_labels = [y for (x, y) in sorted_corpus]  # 提取排序后的标签
        logging.info('Generating sentence embeddings')  # 记录日志：生成句子嵌入
        for ii in range(0, self.n_samples, params.batch_size):
            batch = sorted_samples[ii:ii + params.batch_size]  # 获取当前批次的样本
            embeddings = batcher(params, batch)  # 生成批次的嵌入表示
            enc_input.append(embeddings)  # 将嵌入表示添加到编码输入列表中
        enc_input = np.vstack(enc_input)  # 将编码输入列表堆叠成一个大的numpy数组
        logging.info('Generated sentence embeddings')  # 记录日志：已生成句子嵌入

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}  # 配置分类器参数
        clf = InnerKFoldClassifier(enc_input, np.array(sorted_labels), config)  # 创建交叉验证分类器
        devacc, testacc = clf.run()  # 运行分类器并获得开发集和测试集准确率
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))  # 记录详细日志：开发集和测试集准确率
        return {'devacc': devacc, 'acc': testacc, 'ndev': self.n_samples,
                'ntest': self.n_samples}  # 返回性能指标字典


class CREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : CR *****\n\n')  # 记录调试日志：传输任务为CR
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))  # 加载正面评价文件并解析
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))  # 加载负面评价文件并解析
        super(self.__class__, self).__init__(pos, neg, seed)  # 调用父类构造函数进行初始化


class MREval(BinaryClassifierEval):
    pass  # MR任务的评估类，继承自BinaryClassifierEval，暂无额外实现
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, task_path, seed=1111):
        # 输出调试信息，标记开始一个新的转移任务 MR
        logging.debug('***** Transfer task : MR *****\n\n')
        # 加载正面情感文本文件，并将其内容赋给 pos 变量
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        # 加载负面情感文本文件，并将其内容赋给 neg 变量
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        # 调用父类的初始化方法，传入正面和负面情感文本数据以及随机种子
        super(self.__class__, self).__init__(pos, neg, seed)
# 定义 SUBJ 数据集评估类，继承自 BinaryClassifierEval 类
class SUBJEval(BinaryClassifierEval):
    # 初始化方法，接受任务路径和种子值作为参数
    def __init__(self, task_path, seed=1111):
        # 输出调试信息，表示正在处理 SUBJ 数据集的转移任务
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        # 载入并处理目标文件，生成对象类别数据
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        # 载入并处理主观文件，生成主观类别数据
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        # 调用父类的初始化方法，传递处理后的数据和种子值
        super(self.__class__, self).__init__(obj, subj, seed)


# 定义 MPQA 数据集评估类，继承自 BinaryClassifierEval 类
class MPQAEval(BinaryClassifierEval):
    # 初始化方法，接受任务路径和种子值作为参数
    def __init__(self, task_path, seed=1111):
        # 输出调试信息，表示正在处理 MPQA 数据集的转移任务
        logging.debug('***** Transfer task : MPQA *****\n\n')
        # 载入并处理正面文件，生成正面情感类别数据
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        # 载入并处理负面文件，生成负面情感类别数据
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        # 调用父类的初始化方法，传递处理后的数据和种子值
        super(self.__class__, self).__init__(pos, neg, seed)
```