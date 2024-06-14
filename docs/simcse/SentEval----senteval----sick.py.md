# `.\SentEval\senteval\sick.py`

```
'''
SICK Relatedness and Entailment
'''
# 导入必要的库和模块
from __future__ import absolute_import, division, unicode_literals

import os  # 导入处理操作系统相关功能的库
import io  # 导入处理文件输入输出流的库
import logging  # 导入记录日志的库
import numpy as np  # 导入处理数值计算的库

from sklearn.metrics import mean_squared_error  # 导入均方误差评估函数
from scipy.stats import pearsonr, spearmanr  # 导入皮尔逊相关系数和斯皮尔曼相关系数评估函数

from senteval.tools.relatedness import RelatednessPytorch  # 从senteval工具包中导入相关性PyTorch类
from senteval.tools.validation import SplitClassifier  # 从senteval工具包中导入分裂分类器类

# 定义SICK评估类，处理相关度任务
class SICKEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')  # 记录调试信息：转移任务为SICK相关度
        self.seed = seed  # 设置随机种子
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))  # 加载训练数据集
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))  # 加载开发数据集
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))  # 加载测试数据集
        self.sick_data = {'train': train, 'dev': dev, 'test': test}  # 存储数据集到对象属性中

    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + \
                  self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + \
                  self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)  # 准备数据样本，调用传入的准备函数

    def loadFile(self, fpath):
        skipFirstLine = True  # 初始化是否跳过第一行的标志
        sick_data = {'X_A': [], 'X_B': [], 'y': []}  # 初始化SICK数据字典
        with io.open(fpath, 'r', encoding='utf-8') as f:  # 打开文件进行读取
            for line in f:  # 遍历文件的每一行
                if skipFirstLine:
                    skipFirstLine = False  # 如果是第一行，则标记为不跳过
                else:
                    text = line.strip().split('\t')  # 去除首尾空白并以制表符分割文本行
                    sick_data['X_A'].append(text[1].split())  # 将第二列文本按空格分割后加入X_A列表
                    sick_data['X_B'].append(text[2].split())  # 将第三列文本按空格分割后加入X_B列表
                    sick_data['y'].append(text[3])  # 将第四列标签加入y列表

        sick_data['y'] = [float(s) for s in sick_data['y']]  # 将标签y列表转换为浮点数类型
        return sick_data  # 返回处理后的SICK数据字典

    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')  # 初始化标签编码矩阵Y
        for j, y in enumerate(labels):  # 遍历标签列表
            for i in range(nclass):  # 遍历类别数
                if i+1 == np.floor(y) + 1:  # 如果当前类别与向下取整后的y相等
                    Y[j, i] = y - np.floor(y)  # 计算编码值
                if i+1 == np.floor(y):  # 如果当前类别与向下取整后的y相等
                    Y[j, i] = np.floor(y) - y + 1  # 计算编码值
        return Y  # 返回编码后的标签矩阵


# 定义SICK蕴含评估类，继承自SICKEval基类
class SICKEntailmentEval(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Entailment*****\n\n')  # 记录调试信息：转移任务为SICK蕴含
        self.seed = seed  # 设置随机种子
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))  # 加载训练数据集
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))  # 加载开发数据集
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))  # 加载测试数据集
        self.sick_data = {'train': train, 'dev': dev, 'test': test}  # 存储数据集到对象属性中
    # 定义一个方法用于加载文件内容并解析为数据结构
    def loadFile(self, fpath):
        # 将标签映射到数字编号的字典
        label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        
        # 标记是否跳过第一行，初始化为 True，表示初始状态下需要跳过第一行
        skipFirstLine = True
        
        # 初始化用于存储数据的字典
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        
        # 打开指定路径的文件，使用 utf-8 编码方式读取
        with io.open(fpath, 'r', encoding='utf-8') as f:
            # 逐行读取文件内容
            for line in f:
                # 如果 skipFirstLine 为 True，则跳过当前行，将其置为 False
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    # 去除行两端的空白字符，并按制表符分割成列表
                    text = line.strip().split('\t')
                    
                    # 将第二列文本按空格分割后作为 X_A 的元素添加到列表中
                    sick_data['X_A'].append(text[1].split())
                    
                    # 将第三列文本按空格分割后作为 X_B 的元素添加到列表中
                    sick_data['X_B'].append(text[2].split())
                    
                    # 将第五列标签直接作为 y 的元素添加到列表中
                    sick_data['y'].append(text[4])
        
        # 将标签列表 sick_data['y'] 转换为对应的数字标签列表
        sick_data['y'] = [label2id[s] for s in sick_data['y']]
        
        # 返回整理好的数据字典
        return sick_data
    def run(self, params, batcher):
        # 初始化一个空字典用于存储不同数据集的嵌入向量
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        # 获取批处理大小
        bsize = params.batch_size

        # 遍历不同数据集（train, dev, test）
        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # 对数据进行排序以减少填充
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            # 更新排序后的数据到原始数据结构中
            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            # 对每种文本类型（'X_A', 'X_B'）计算嵌入向量
            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                # 分批次处理数据，计算每个批次的嵌入向量
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                # 将每个文本类型的所有批次嵌入向量堆叠成一个大的矩阵
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # 训练集嵌入向量和标签
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.sick_data['train']['y'])

        # 验证集嵌入向量和标签
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.sick_data['dev']['y'])

        # 测试集嵌入向量和标签
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.sick_data['test']['y'])

        # 设置分类器的配置参数
        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid}
        # 创建一个分割分类器对象
        clf = SplitClassifier(X={'train': trainF, 'valid': devF, 'test': testF},
                              y={'train': trainY, 'valid': devY, 'test': testY},
                              config=config)

        # 运行分类器，得到验证集和测试集的准确率
        devacc, testacc = clf.run()
        # 打印调试信息，显示验证集和测试集的准确率
        logging.debug('\nDev acc : {0} Test acc : {1} for \
                       SICK entailment\n'.format(devacc, testacc))
        # 返回结果字典，包括验证集准确率、测试集准确率以及数据集大小
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(devA), 'ntest': len(testA)}
```