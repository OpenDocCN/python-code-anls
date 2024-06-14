# `.\SentEval\senteval\sst.py`

```
# 引入必要的模块和库
from __future__ import absolute_import, division, unicode_literals
import os  # 引入操作系统相关功能的模块
import io  # 引入输入输出相关功能的模块
import logging  # 引入日志记录功能的模块
import numpy as np  # 引入数值计算相关功能的模块
from senteval.tools.validation import SplitClassifier  # 从sent_eval.tools.validation模块中导入SplitClassifier类


class SSTEval(object):
    def __init__(self, task_path, nclasses=2, seed=1111):
        self.seed = seed  # 初始化对象的随机种子属性

        # 检查分类数是否为2或5
        assert nclasses in [2, 5]
        self.nclasses = nclasses  # 初始化对象的分类数属性
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'  # 根据分类数确定任务名称
        logging.debug('***** Transfer task : SST %s classification *****\n\n', self.task_name)  # 记录调试信息

        # 载入训练、开发和测试数据集
        train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_path, 'sentiment-test'))
        self.sst_data = {'train': train, 'dev': dev, 'test': test}  # 组织数据集的字典

    # 准备数据
    def do_prepare(self, params, prepare):
        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                  self.sst_data['test']['X']  # 合并训练、开发和测试数据集的输入样本
        return prepare(params, samples)  # 调用准备函数进行数据准备

    # 加载数据文件
    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}  # 初始化数据字典
        with io.open(fpath, 'r', encoding='utf-8') as f:  # 打开文件进行读取
            for line in f:  # 遍历文件的每一行
                if self.nclasses == 2:  # 如果分类数为2
                    sample = line.strip().split('\t')  # 对行进行分割
                    sst_data['y'].append(int(sample[1]))  # 提取标签并转换为整数类型，添加到y列表中
                    sst_data['X'].append(sample[0].split())  # 提取文本内容并进行分词，添加到X列表中
                elif self.nclasses == 5:  # 如果分类数为5
                    sample = line.strip().split(' ', 1)  # 对行进行分割
                    sst_data['y'].append(int(sample[0]))  # 提取标签并转换为整数类型，添加到y列表中
                    sst_data['X'].append(sample[1].split())  # 提取文本内容并进行分词，添加到X列表中
        assert max(sst_data['y']) == self.nclasses - 1  # 断言最大标签值等于分类数减1
        return sst_data  # 返回数据字典
    def run(self, params, batcher):
        # 初始化一个空的嵌入字典，包含训练集、开发集和测试集
        sst_embed = {'train': {}, 'dev': {}, 'test': {}}
        # 批处理大小设定为参数中的批处理大小
        bsize = params.batch_size

        # 遍历self.sst_data中的每个键（train, dev, test）
        for key in self.sst_data:
            # 记录日志，指示正在计算特定数据集的嵌入
            logging.info('Computing embedding for {0}'.format(key))
            # 对数据进行排序以减少填充
            sorted_data = sorted(zip(self.sst_data[key]['X'],
                                     self.sst_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            # 更新数据集，将排序后的数据重新分配回self.sst_data[key]['X']和self.sst_data[key]['y']
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))

            # 初始化一个空列表，用于存储当前数据集的嵌入
            sst_embed[key]['X'] = []
            # 对数据集进行批处理
            for ii in range(0, len(self.sst_data[key]['y']), bsize):
                batch = self.sst_data[key]['X'][ii:ii + bsize]
                # 调用批处理器batcher来计算当前批次的嵌入
                embeddings = batcher(params, batch)
                # 将计算得到的嵌入添加到sset_embed[key]['X']中
                sst_embed[key]['X'].append(embeddings)
            # 将列表转换为NumPy数组，以便后续处理
            sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            # 将当前数据集的标签转换为NumPy数组
            sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
            # 记录日志，指示已计算完成特定数据集的嵌入
            logging.info('Computed {0} embeddings'.format(key))

        # 配置分类器的参数，包括类别数、种子、是否使用PyTorch、分类器类型等
        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        # 使用嵌入数据和分类器配置初始化SplitClassifier对象
        clf = SplitClassifier(X={'train': sst_embed['train']['X'],
                                 'valid': sst_embed['dev']['X'],
                                 'test': sst_embed['test']['X']},
                              y={'train': sst_embed['train']['y'],
                                 'valid': sst_embed['dev']['y'],
                                 'test': sst_embed['test']['y']},
                              config=config_classifier)

        # 运行分类器，并获取开发集和测试集的准确率
        devacc, testacc = clf.run()
        # 记录调试信息，包括开发集和测试集的准确率以及任务名称
        logging.debug('\nDev acc : {0} Test acc : {1} for \
            SST {2} classification\n'.format(devacc, testacc, self.task_name))

        # 返回包含开发集和测试集准确率以及相关统计信息的字典
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(sst_embed['dev']['X']),
                'ntest': len(sst_embed['test']['X'])}
```