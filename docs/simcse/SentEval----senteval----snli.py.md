# `.\SentEval\senteval\snli.py`

```
'''
SNLI - Entailment
'''
# 导入必要的库和模块
from __future__ import absolute_import, division, unicode_literals

import codecs  # 用于处理文件编码
import os  # 提供了与操作系统交互的功能
import io  # 提供了对I/O流的核心工具
import copy  # 提供了对象的深拷贝和浅拷贝操作
import logging  # 提供了灵活的日志记录功能
import numpy as np  # 提供了对数组和矩阵操作的支持

from senteval.tools.validation import SplitClassifier  # 从senteval工具中导入SplitClassifier类


class SNLIEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : SNLI Entailment*****\n\n')
        self.seed = seed
        # 加载训练数据集的句子1和句子2
        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))

        # 加载训练数据集的标签
        trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                              encoding='utf-8').read().splitlines()

        # 加载验证数据集的句子1和句子2
        valid1 = self.loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = self.loadFile(os.path.join(taskpath, 's2.dev'))

        # 加载验证数据集的标签
        validlabels = io.open(os.path.join(taskpath, 'labels.dev'),
                              encoding='utf-8').read().splitlines()

        # 加载测试数据集的句子1和句子2
        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))

        # 加载测试数据集的标签
        testlabels = io.open(os.path.join(taskpath, 'labels.test'),
                             encoding='utf-8').read().splitlines()

        # 按照句子2的长度、句子1的长度和标签的顺序对训练数据进行排序，以减少填充
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        # 按照句子2的长度、句子1的长度和标签的顺序对验证数据进行排序，以减少填充
        sorted_valid = sorted(zip(valid2, valid1, validlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        # 按照句子2的长度、句子1的长度和标签的顺序对测试数据进行排序，以减少填充
        sorted_test = sorted(zip(test2, test1, testlabels),
                             key=lambda z: (len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))

        # 将所有样本合并为一个列表
        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2

        # 构建数据字典，包含训练、验证和测试数据
        self.data = {'train': (train1, train2, trainlabels),
                     'valid': (valid1, valid2, validlabels),
                     'test': (test1, test2, testlabels)
                     }

    def do_prepare(self, params, prepare):
        # 调用预处理函数，处理所有样本
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        # 使用指定的编码方式打开文件，并按行读取内容，每行按空格分割为列表
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.split() for line in
                    f.read().splitlines()]
    # 定义一个方法，接受参数 self, params, batcher
    def run(self, params, batcher):
        # 初始化实例变量 self.X 和 self.y 为空字典
        self.X, self.y = {}, {}
        # 定义一个包含标签的字典
        dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
        # 遍历 self.data 中的每个键值对
        for key in self.data:
            # 如果键不在 self.X 中，则初始化为一个空列表
            if key not in self.X:
                self.X[key] = []
            # 如果键不在 self.y 中，则初始化为一个空列表
            if key not in self.y:
                self.y[key] = []

            # 从 self.data[key] 中获取 input1, input2 和 mylabels
            input1, input2, mylabels = self.data[key]
            # 初始化 enc_input 为空列表
            enc_input = []
            # 计算标签数量
            n_labels = len(mylabels)
            # 以 batch_size 为步长，遍历标签
            for ii in range(0, n_labels, params.batch_size):
                # 获取 batch1 和 batch2
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # 如果 batch1 和 batch2 长度相等且大于0
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    # 使用 batcher 对 batch1 和 batch2 进行处理并拼接结果，添加到 enc_input
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                # 如果满足条件
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    # 打印进度日志
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            # 将 enc_input 转换为 numpy 数组，并赋值给 self.X[key]
            self.X[key] = np.vstack(enc_input)
            # 将 mylabels 转换为数字标签并赋值给 self.y[key]
            self.y[key] = [dico_label[y] for y in mylabels]

        # 配置字典参数
        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        # 深度复制分类器参数
        config_classifier = copy.deepcopy(params.classifier)
        # 设置分类器的最大周期和周期大小
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        # 将分类器配置加入总配置中
        config['classifier'] = config_classifier

        # 创建一个 SplitClassifier 对象，传入数据、标签和配置
        clf = SplitClassifier(self.X, self.y, config)
        # 运行分类器并获取开发集和测试集的准确率
        devacc, testacc = clf.run()
        # 记录开发集和测试集准确率的日志
        logging.debug('Dev acc : {0} Test acc : {1} for SNLI\n'
                      .format(devacc, testacc))
        # 返回字典，包含开发集准确率、测试集准确率、开发集大小和测试集大小
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
```