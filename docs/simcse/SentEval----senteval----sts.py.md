# `.\SentEval\senteval\sts.py`

```
# 版权声明及引用
# 版权声明，版权归 Facebook, Inc. 所有
# 本源代码在源代码根目录下的 LICENSE 文件中可找到许可证

'''
STS-{2012,2013,2014,2015,2016} (无监督) 和
STS-benchmark (监督) 任务
'''

from __future__ import absolute_import, division, unicode_literals
# 从 __future__ 模块导入绝对导入、除法、Unicode 字符串字面值

import os
import io
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr
# 从 scipy.stats 模块导入 Spearman 相关系数、Pearson 相关系数

from senteval.utils import cosine
from senteval.sick import SICKEval
# 从 senteval.utils 模块导入余弦相似度函数，从 senteval.sick 模块导入 SICK 评估类


class STSEval(object):
    def loadFile(self, fpath):
        # 初始化数据和样本列表
        self.data = {}
        self.samples = []

        # 遍历数据集
        for dataset in self.datasets:
            # 读取句子对和相关度分数
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            # 读取原始分数并转换为浮点数数组
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            # 找到非空分数的索引
            not_empty_idx = raw_scores != ''

            # 提取非空分数和句子对
            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            
            # 按长度排序数据以最小化批处理中的填充
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            # 将数据存储在字典中
            self.data[dataset] = (sent1, sent2, gs_scores)
            # 更新样本列表
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        # 检查参数中是否包含相似度函数，如果没有，则使用默认的余弦相似度函数
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # 默认相似度函数为余弦相似度
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        # 调用准备函数，传入参数和样本列表
        return prepare(params, self.samples)
    def run(self, params, batcher):
        results = {}  # 初始化结果字典
        all_sys_scores = []  # 存储所有系统得分的列表
        all_gs_scores = []  # 存储所有金标准得分的列表
        
        # 遍历每个数据集
        for dataset in self.datasets:
            sys_scores = []  # 存储当前数据集的系统得分列表
            input1, input2, gs_scores = self.data[dataset]  # 获取当前数据集的输入和金标准得分
            
            # 按照批次大小迭代处理数据
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]  # 提取第一个输入批次
                batch2 = input2[ii:ii + params.batch_size]  # 提取第二个输入批次
                
                # 假设 get_batch 已经去除了错误的样本
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)  # 对第一个批次进行编码
                    enc2 = batcher(params, batch2)  # 对第二个批次进行编码
                    
                    # 计算每个样本对的系统得分
                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)  # 将系统得分添加到列表中
            
            all_sys_scores.extend(sys_scores)  # 将当前数据集的系统得分合并到总系统得分列表中
            all_gs_scores.extend(gs_scores)  # 将当前数据集的金标准得分合并到总金标准得分列表中
            
            # 计算当前数据集的评估指标，并将其添加到结果字典中
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            
            # 记录调试信息，显示当前数据集的 Pearson 和 Spearman 相关系数
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        # 计算每个数据集样本数的权重
        weights = [results[dset]['nsamples'] for dset in results.keys()]
        
        # 计算所有数据集的平均 Pearson 相关系数
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        
        # 计算所有数据集的平均 Spearman 相关系数
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        # 计算所有数据集的平均 Pearson 相关系数
        avg_pearson = np.average(list_prs)
        
        # 计算所有数据集的平均 Spearman 相关系数
        avg_spearman = np.average(list_spr)
        
        # 计算加权平均 Pearson 相关系数
        wavg_pearson = np.average(list_prs, weights=weights)
        
        # 计算加权平均 Spearman 相关系数
        wavg_spearman = np.average(list_spr, weights=weights)
        
        # 计算所有系统得分和所有金标准得分的 Pearson 和 Spearman 相关系数
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        
        # 将整体评估指标添加到结果字典中
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        
        # 记录调试信息，显示整体 Pearson 和 Spearman 相关系数
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results  # 返回结果字典
class STS12Eval(STSEval):
    # STS12Eval 类，用于评估 STS12 任务
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        # 调用父类方法，加载任务路径中的文件
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13Eval 类，用于评估 STS13 任务（不包含 "SMT" 子任务，因为许可问题）
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        # 调用父类方法，加载任务路径中的文件
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    # STS14Eval 类，用于评估 STS14 任务
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        # 调用父类方法，加载任务路径中的文件
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    # STS15Eval 类，用于评估 STS15 任务
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        # 调用父类方法，加载任务路径中的文件
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    # STS16Eval 类，用于评估 STS16 任务
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        # 调用父类方法，加载任务路径中的文件
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    # STSBenchmarkEval 类，用于评估 STS Benchmark 任务
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        # 加载训练集、验证集和测试集文件，并存储在对应的数据结构中
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        # 加载指定路径的文件并解析其中的文本数据和标签，存储在数据字典中
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        # 将样本文本数据添加到 self.samples 中
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    # STSBenchmarkFinetune 类，用于 STS Benchmark 的微调任务
    pass
    # 初始化方法，用于创建一个新的STSBenchmark任务实例
    def __init__(self, task_path, seed=1111):
        # 调试日志，记录任务初始化的开始
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        # 设置随机种子
        self.seed = seed
        # 加载训练集数据
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        # 加载开发集数据
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        # 加载测试集数据
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        # 存储所有数据集的字典
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    # 加载给定文件的数据并返回格式化后的数据字典
    def loadFile(self, fpath):
        # 初始化一个空的数据字典，用于存储数据
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        # 使用 utf-8 编码打开文件流
        with io.open(fpath, 'r', encoding='utf-8') as f:
            # 遍历文件的每一行
            for line in f:
                # 去除首尾空白字符并按制表符分割文本行
                text = line.strip().split('\t')
                # 将第6列数据按空格分割并添加到 X_A 列表中
                sick_data['X_A'].append(text[5].split())
                # 将第7列数据按空格分割并添加到 X_B 列表中
                sick_data['X_B'].append(text[6].split())
                # 将第5列数据（标签）添加到 y 列表中
                sick_data['y'].append(text[4])

        # 将 y 列表中的每个元素转换为浮点数
        sick_data['y'] = [float(s) for s in sick_data['y']]
        # 返回格式化后的数据字典
        return sick_data
# 继承自 STSEval 类，用于评估 SICK 相关性任务
class SICKRelatednessEval(STSEval):
    
    # 初始化方法，接受任务路径和种子参数
    def __init__(self, task_path, seed=1111):
        # 输出调试信息，标识当前转移任务为 SICK 相关性评估
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        # 设置种子参数
        self.seed = seed
        # 初始化样本列表
        self.samples = []
        
        # 加载训练集、开发集和测试集数据
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        
        # 设置数据集名称列表
        self.datasets = ['train', 'dev', 'test']
        # 构建数据字典，包含训练集、开发集和测试集数据
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    # 加载指定文件的数据，返回元组包含 X_A, X_B 和 y
    def loadFile(self, fpath):
        # 初始化跳过第一行的标志
        skipFirstLine = True
        # 初始化数据字典，存储 'X_A', 'X_B', 'y' 对应的数据列表
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        
        # 使用 UTF-8 编码打开文件流
        with io.open(fpath, 'r', encoding='utf-8') as f:
            # 逐行读取文件内容
            for line in f:
                # 如果是第一行，则跳过
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    # 分割行内容，以制表符 '\t' 分隔，去除首尾空白字符
                    text = line.strip().split('\t')
                    # 将第二列文本按空格分割后加入 'X_A' 对应的列表
                    sick_data['X_A'].append(text[1].split())
                    # 将第三列文本按空格分割后加入 'X_B' 对应的列表
                    sick_data['X_B'].append(text[2].split())
                    # 将第四列标签转换为浮点数后加入 'y' 对应的列表
                    sick_data['y'].append(text[3])

        # 将 'y' 列表中的内容转换为浮点数
        sick_data['y'] = [float(s) for s in sick_data['y']]
        # 将 'X_A' 和 'X_B' 列表的内容合并到样本列表中
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        # 返回包含 'X_A', 'X_B', 'y' 的元组
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
```