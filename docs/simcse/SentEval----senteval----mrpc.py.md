# `.\SentEval\senteval\mrpc.py`

```
# 版权声明和许可信息
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入未来的功能模块，确保代码向后兼容性
from __future__ import absolute_import, division, unicode_literals

# 导入标准库和第三方库
import os             # 处理操作系统相关功能的模块
import logging        # 记录日志信息的模块
import numpy as np    # 处理数值数据的模块
import io             # 处理流输入输出的模块

# 导入自定义的工具函数和类
from senteval.tools.validation import KFoldClassifier   # 导入自定义的交叉验证分类器

# 导入第三方库的功能
from sklearn.metrics import f1_score   # 导入第三方库的 F1 分数计算函数

# 定义 MRPC 评估类
class MRPCEval(object):
    def __init__(self, task_path, seed=1111):
        # 输出日志信息，标明当前任务为 MRPC
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        # 加载训练和测试数据集
        train = self.loadFile(os.path.join(task_path, 'msr_paraphrase_train.txt'))
        test = self.loadFile(os.path.join(task_path, 'msr_paraphrase_test.txt'))
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        # 合并训练集和测试集中的样本
        samples = self.mrpc_data['train']['X_A'] + \
                  self.mrpc_data['train']['X_B'] + \
                  self.mrpc_data['test']['X_A'] + self.mrpc_data['test']['X_B']
        # 调用准备函数处理数据
        return prepare(params, samples)

    def loadFile(self, fpath):
        # 加载文件中的数据到 MRPC 数据结构中
        mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                mrpc_data['X_A'].append(text[3].split())   # 添加第一句文本
                mrpc_data['X_B'].append(text[4].split())   # 添加第二句文本
                mrpc_data['y'].append(text[0])             # 添加标签

        # 移除头部的标签和空数据
        mrpc_data['X_A'] = mrpc_data['X_A'][1:]
        mrpc_data['X_B'] = mrpc_data['X_B'][1:]
        mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]   # 将标签转换为整数类型
        return mrpc_data
    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        for key in self.mrpc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            # 将数据按照长度和标签进行排序，以减少填充
            text_data = {}
            sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
                                       self.mrpc_data[key]['X_B'],
                                       self.mrpc_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                mrpc_embed[key][txt_type] = []
                # 按批次处理文本数据，生成对应的嵌入向量
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    mrpc_embed[key][txt_type].append(embeddings)
                mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
            mrpc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = mrpc_embed['train']['A']
        trainB = mrpc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = mrpc_embed['train']['y']

        # Test
        testA = mrpc_embed['test']['A']
        testB = mrpc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = mrpc_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        # 使用 K 折交叉验证分类器进行训练和测试
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        # 计算测试 F1 分数，并四舍五入保留两位小数
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for MRPC.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
```