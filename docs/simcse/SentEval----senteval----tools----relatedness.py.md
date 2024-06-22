# `.\SentEval\senteval\tools\relatedness.py`

```py
"""
Semantic Relatedness (supervised) with Pytorch
"""
# 导入必要的库
from __future__ import absolute_import, division, unicode_literals

import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim

from scipy.stats import pearsonr, spearmanr

# 创建一个类用于计算语义相关性
class RelatednessPytorch(object):
    # 初始化方法，接受训练、验证、测试数据，以及相关配置
    # 可用于SICK-Relatedness和STS14
    def __init__(self, train, valid, test, devscores, config):
        # 固定随机种子
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        torch.cuda.manual_seed(config['seed'])

        # 初始化训练、验证、测试数据和开发分数
        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        # 确定输入维度和类别数
        self.inputdim = train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.l2reg = 0.
        self.batch_size = 64
        self.maxepoch = 1000
        self.early_stop = True

        # 创建模型，包括线性层和Softmax层
        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
            nn.Softmax(dim=-1),
        )
        # 定义损失函数为均方误差损失函数
        self.loss_fn = nn.MSELoss()

        # 如果CUDA可用，将模型和损失函数移到GPU上
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        # 设置损失函数的平均方式为False
        self.loss_fn.size_average = False
        # 使用Adam优化器，设置权重衰减参数
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)

    # 数据准备方法，将输入数据转换为PyTorch张量，并移到GPU上
    def prepare_data(self, trainX, trainy, devX, devy, testX, testy):
        # 将训练数据转换为PyTorch张量，并移到GPU上
        trainX = torch.from_numpy(trainX).float().cuda()
        trainy = torch.from_numpy(trainy).float().cuda()
        # 将验证数据转换为PyTorch张量，并移到GPU上
        devX = torch.from_numpy(devX).float().cuda()
        devy = torch.from_numpy(devy).float().cuda()
        # 将测试数据转换为PyTorch张量，并移到GPU上
        testX = torch.from_numpy(testX).float().cuda()
        testY = torch.from_numpy(testy).float().cuda()

        return trainX, trainy, devX, devy, testX, testy
    `
        # 定义一个方法来运行训练过程
        def run(self):
            # 初始化训练周期数为0
            self.nepoch = 0
            # 初始化最佳的Pearson相关系数为-1
            bestpr = -1
            # 初始化早停计数器为0
            early_stop_count = 0
            # 创建一个包含1到5的一维数组
            r = np.arange(1, 6)
            # 初始化停止训练标志为False
            stop_train = False
    
            # 准备数据
            # 调用prepare_data方法准备训练、验证和测试数据集
            trainX, trainy, devX, devy, testX, testy = self.prepare_data(
                self.train['X'], self.train['y'],
                self.valid['X'], self.valid['y'],
                self.test['X'], self.test['y'])
    
            # 训练
            # 当未达到停止训练条件且训练周期数未达到最大周期数时，执行循环
            while not stop_train and self.nepoch <= self.maxepoch:
                # 进行50个周期的训练
                self.trainepoch(trainX, trainy, nepoches=50)
                # 对验证集进行预测，计算得分
                yhat = np.dot(self.predict_proba(devX), r)
                # 计算Pearson相关系数
                pr = spearmanr(yhat, self.devscores)[0]
                # 如果Pearson相关系数为NaN，则置为0（当标准差为0时）
                pr = 0 if pr != pr else pr  
                # 当Pearson相关系数大于最佳相关系数时
                if pr > bestpr:
                    # 更新最佳相关系数和最佳模型
                    bestpr = pr
                    bestmodel = copy.deepcopy(self.model)
                # 如果启用了早停机制
                elif self.early_stop:
                    # 如果早停计数器大于等于3，则停止训练
                    if early_stop_count >= 3:
                        stop_train = True
                    # 否则早停计数器加1
                    early_stop_count += 1
            # 将模型更新为最佳模型
            self.model = bestmodel
    
            # 对测试集进行预测
            yhat = np.dot(self.predict_proba(testX), r)
    
            # 返回最佳的Pearson相关系数和测试集预测结果
            return bestpr, yhat
    
        # 训练一个epoch
        def trainepoch(self, X, y, nepoches=1):
            # 将模型设置为训练模式
            self.model.train()
            # 循环执行指定次数的epoch
            for _ in range(self.nepoch, self.nepoch + nepoches):
                # 对数据集进行随机排列
                permutation = np.random.permutation(len(X))
                all_costs = []
                # 遍历数据集，按照批次大小进行训练
                for i in range(0, len(X), self.batch_size):
                    # 正向传播
                    idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
                    Xbatch = X[idx]
                    yoches
    
        # 预测样本的概率
        def predict_proba(self, devX):
            # 设置模型为评估模式
            self.model.eval()
            probas = []
            with torch.no_grad():
                for i in range(0, len(devX), self.batch_size):
                    Xbatch = devX[i:i + self.batch_size]
                    if len(probas) == 0:
                        probas = self.model(Xbatch).data.cpu().numpy()
                    else:
                        probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
            return probas
```