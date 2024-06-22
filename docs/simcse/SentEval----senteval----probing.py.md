# `.\SentEval\senteval\probing.py`

```py
# 引入必要的库和模块
from __future__ import absolute_import, division, unicode_literals  # 导入未来版本兼容模块

import os  # 导入操作系统功能模块
import io  # 导入输入输出流模块
import copy  # 导入对象复制模块
import logging  # 导入日志记录模块
import numpy as np  # 导入数值计算模块

from senteval.tools.validation import SplitClassifier  # 从senteval工具中导入SplitClassifier类


class PROBINGEval(object):
    def __init__(self, task, task_path, seed=1111):
        self.seed = seed  # 设定随机种子
        self.task = task  # 任务名称
        logging.debug('***** (Probing) Transfer task : %s classification *****', self.task.upper())
        # 记录调试信息，显示任务名称（转移任务分类）

        # 初始化任务数据结构，包括训练、开发和测试集
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}
        
        # 载入指定路径的文件数据
        self.loadFile(task_path)
        
        # 记录信息，显示载入的训练、开发和测试集大小及任务名称
        logging.info('Loaded %s train - %s dev - %s test for %s' %
                     (len(self.task_data['train']['y']), len(self.task_data['dev']['y']),
                      len(self.task_data['test']['y']), self.task))

    def do_prepare(self, params, prepare):
        # 将训练、开发和测试集的数据合并为一个样本列表
        samples = self.task_data['train']['X'] + self.task_data['dev']['X'] + \
                  self.task_data['test']['X']
        # 调用准备函数，处理样本数据
        return prepare(params, samples)

    def loadFile(self, fpath):
        # 定义分割符映射到数据集名称的字典
        self.tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        
        # 打开文件并逐行读取
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')  # 去除行尾换行符并按制表符分割
                # 将样本数据和标签分别加入对应的数据集中
                self.task_data[self.tok2split[line[0]]]['X'].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]['y'].append(line[1])

        # 对训练集标签进行排序并建立标签到索引的映射
        labels = sorted(np.unique(self.task_data['train']['y']))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)  # 记录类别数量

        # 将数据集中的标签转换为对应的索引
        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]['y']):
                self.task_data[split]['y'][i] = self.tok2label[y]
    def run(self, params, batcher):
        # 初始化一个空字典用于存储各个任务的嵌入向量
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        # 获取批处理大小
        bsize = params.batch_size
        # 记录日志，指示正在计算训练集、开发集和测试集的嵌入向量
        logging.info('Computing embeddings for train/dev/test')
        # 遍历每个任务的数据集
        for key in self.task_data:
            # 对数据进行排序，以减少填充
            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            # 更新排序后的数据到任务数据中
            self.task_data[key]['X'], self.task_data[key]['y'] = map(list, zip(*sorted_data))

            # 初始化当前任务的嵌入向量存储空间
            task_embed[key]['X'] = []
            # 分批计算嵌入向量
            for ii in range(0, len(self.task_data[key]['y']), bsize):
                batch = self.task_data[key]['X'][ii:ii + bsize]
                # 调用批处理器计算当前批次的嵌入向量
                embeddings = batcher(params, batch)
                # 将计算得到的嵌入向量添加到当前任务的嵌入向量列表中
                task_embed[key]['X'].append(embeddings)
            # 将当前任务的所有嵌入向量堆叠成一个 numpy 数组
            task_embed[key]['X'] = np.vstack(task_embed[key]['X'])
            # 将任务的标签转换为 numpy 数组
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
        # 计算嵌入向量完成后记录日志
        logging.info('Computed embeddings')

        # 配置分类器参数
        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        # 如果任务是"WordContent"且分类器的隐藏层大小大于0，则调整分类器的配置
        if self.task == "WordContent" and params.classifier['nhid'] > 0:
            # 深拷贝分类器配置
            config_classifier = copy.deepcopy(config_classifier)
            # 将分类器的隐藏层大小设置为0
            config_classifier['classifier']['nhid'] = 0
            # 打印分类器的隐藏层大小（调试用）
            print(params.classifier['nhid'])

        # 创建分类器对象
        clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['dev']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['dev']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

        # 运行分类器，获取开发集和测试集的准确率
        devacc, testacc = clf.run()
        # 记录详细信息，包括开发集和测试集的准确率
        logging.debug('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, self.task.upper()))

        # 返回包含开发集准确率、测试集准确率、开发集样本数量和测试集样本数量的字典
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(task_embed['dev']['X']),
                'ntest': len(task_embed['test']['X'])}
"""
Surface Information
"""

# 定义一个用于评估长度的类，继承自PROBINGEval类
class LengthEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'sentence_length.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'Length', task_path, seed)

# 定义一个用于评估单词内容的类，继承自PROBINGEval类
class WordContentEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'word_content.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'WordContent', task_path, seed)

"""
Latent Structural Information
"""

# 定义一个用于评估深度的类，继承自PROBINGEval类
class DepthEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'tree_depth.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'Depth', task_path, seed)

# 定义一个用于评估顶层成分的类，继承自PROBINGEval类
class TopConstituentsEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'top_constituents.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'TopConstituents', task_path, seed)

# 定义一个用于评估双词移位的类，继承自PROBINGEval类
class BigramShiftEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'bigram_shift.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'BigramShift', task_path, seed)

# TODO: Voice?

"""
Latent Semantic Information
"""

# 定义一个用于评估时态的类，继承自PROBINGEval类
class TenseEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'past_present.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'Tense', task_path, seed)

# 定义一个用于评估主语数的类，继承自PROBINGEval类
class SubjNumberEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'subj_number.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'SubjNumber', task_path, seed)

# 定义一个用于评估宾语数的类，继承自PROBINGEval类
class ObjNumberEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'obj_number.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'ObjNumber', task_path, seed)

# 定义一个用于评估不合常规的成分的类，继承自PROBINGEval类
class OddManOutEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'odd_man_out.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'OddManOut', task_path, seed)

# 定义一个用于评估协调倒装的类，继承自PROBINGEval类
class CoordinationInversionEval(PROBINGEval):
    # 初始化方法
    def __init__(self, task_path, seed=1111):
        # 合并任务路径和文件名
        task_path = os.path.join(task_path, 'coordination_inversion.txt')
        # 初始化父类PROBINGEval，传入标签名和任务路径
        PROBINGEval.__init__(self, 'CoordinationInversion', task_path, seed)
```