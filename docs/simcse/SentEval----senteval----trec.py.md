# `.\SentEval\senteval\trec.py`

```py
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TREC question-type classification
'''

from __future__ import absolute_import, division, unicode_literals

import os                  # 导入操作系统功能模块
import io                  # 导入输入输出流模块
import logging             # 导入日志记录模块
import numpy as np         # 导入数值计算库NumPy

from senteval.tools.validation import KFoldClassifier   # 从senteval.tools.validation模块导入KFoldClassifier类


class TRECEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : TREC *****\n\n')   # 记录信息到日志，表示开始TREC任务
        self.seed = seed   # 初始化种子值
        self.train = self.loadFile(os.path.join(task_path, 'train_5500.label'))   # 加载训练数据集
        self.test = self.loadFile(os.path.join(task_path, 'TREC_10.label'))       # 加载测试数据集

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']   # 将训练集和测试集的输入样本合并
        return prepare(params, samples)              # 使用给定的prepare函数准备数据集

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}               # 初始化数据字典，包含输入数据X和标签y
        tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,   # 标签到索引的映射
                   'HUM': 3, 'LOC': 4, 'NUM': 5}
        with io.open(fpath, 'r', encoding='latin-1') as f:   # 使用latin-1编码打开文件
            for line in f:                            # 遍历文件的每一行
                target, sample = line.strip().split(':', 1)   # 拆分目标和样本内容
                sample = sample.split(' ', 1)[1].split()       # 处理样本内容，去除开头标识符并按空格拆分为单词列表
                assert target in tgt2idx, target         # 确保目标在tgt2idx字典中
                trec_data['X'].append(sample)            # 将处理后的样本添加到X中
                trec_data['y'].append(tgt2idx[target])   # 将目标对应的索引添加到y中
        return trec_data   # 返回处理后的数据字典
    def run(self, params, batcher):
        # 初始化训练集和测试集的嵌入列表
        train_embeddings, test_embeddings = [], []

        # 对训练集按样本长度和标签排序，以减少填充
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        # 分离排序后的训练样本和标签
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]

        # 对测试集按样本长度和标签排序
        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y']),
                                    key=lambda z: (len(z[0]), z[1]))
        # 分离排序后的测试样本和标签
        test_samples = [x for (x, y) in sorted_corpus_test]
        test_labels = [y for (x, y) in sorted_corpus_test]

        # 获取训练集嵌入向量
        for ii in range(0, len(train_labels), params.batch_size):
            # 按批次获取训练样本
            batch = train_samples[ii:ii + params.batch_size]
            # 使用提供的批处理器获取嵌入向量
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)
        # 将列表中的嵌入向量垂直堆叠为数组
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')  # 记录信息：已计算训练集嵌入向量

        # 获取测试集嵌入向量
        for ii in range(0, len(test_labels), params.batch_size):
            # 按批次获取测试样本
            batch = test_samples[ii:ii + params.batch_size]
            # 使用提供的批处理器获取嵌入向量
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
        # 将列表中的嵌入向量垂直堆叠为数组
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')  # 记录信息：已计算测试集嵌入向量

        # 配置分类器的参数字典
        config_classifier = {'nclasses': 6, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}
        # 使用 KFoldClassifier 初始化分类器对象
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': np.array(test_labels)},
                              config_classifier)
        # 运行分类器，并获取开发集准确率、测试集准确率及其它结果
        devacc, testacc, _ = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} \
            for TREC\n'.format(devacc, testacc))  # 调试信息：开发集和测试集准确率
        # 返回包含评估结果和数据集大小信息的字典
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}
```