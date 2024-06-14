# `.\SentEval\senteval\tools\classifier.py`

```
# 版权声明和许可信息
# 所有权利保留
# 此源代码在根目录下的LICENSE文件中找到的许可下获得许可
"""

# 导入所需的库和模块
from __future__ import absolute_import, division, unicode_literals
import numpy as np  # 导入NumPy库
import copy  # 导入拷贝模块
from senteval import utils  # 从senteval库中导入utils模块

import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
import torch.nn.functional as F  # 从PyTorch中导入函数式模块

# 定义PyTorchClassifier类
class PyTorchClassifier(object):
    # 初始化方法，设置输入维度、类别数、L2正则化参数、批量大小、随机种子、是否使用CUDA加速
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False):
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 初始化属性
        self.inputdim = inputdim  # 输入维度
        self.nclasses = nclasses  # 类别数
        self.l2reg = l2reg  # L2正则化参数
        self.batch_size = batch_size  # 批量大小
        self.cudaEfficient = cudaEfficient  # 是否使用CUDA加速

    # 准备数据集拆分的方法，包括训练集、验证集、数据类型转换等
    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # 检查是否提供了验证数据
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))  # 生成随机排列的索引
            trainidx = permutation[int(validation_split * len(X)):]  # 训练集索引
            devidx = permutation[0:int(validation_split * len(X))]  # 验证集索引
            trainX, trainy = X[trainidx], y[trainidx]  # 训练集数据和标签
            devX, devy = X[devidx], y[devidx]  # 验证集数据和标签

        device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')  # 选择设备

        # 数据类型转换为PyTorch张量并发送到指定设备
        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy  # 返回处理后的数据集

    # 拟合模型的方法，包括训练过程、早停策略等
    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0  # 初始化训练轮数
        bestaccuracy = -1  # 初始化最佳准确率
        stop_train = False  # 是否停止训练的标志
        early_stop_count = 0  # 早停计数器

        # 准备训练集和验证集数据
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # 开始训练循环
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)  # 训练一个epoch
            accuracy = self.score(devX, devy)  # 在验证集上计算准确率
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy  # 更新最佳准确率
                bestmodel = copy.deepcopy(self.model)  # 深拷贝当前最佳模型
            elif early_stop:
                if early_stop_count >= self.tenacity:  # 判断是否达到早停条件
                    stop_train = True  # 停止训练
                early_stop_count += 1  # 更新早停计数器
        self.model = bestmodel  # 更新模型为最佳模型
        return bestaccuracy  # 返回最佳准确率
    # 训练一个 epoch 的模型
    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()  # 设置模型为训练模式
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))  # 随机排列数据索引
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)  # 根据索引选择批次数据
                Xbatch = X[idx]  # 根据索引获取输入数据批次
                ybatch = y[idx]  # 根据索引获取标签数据批次

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()  # 将输入数据移到 GPU 上
                    ybatch = ybatch.cuda()  # 将标签数据移到 GPU 上
                output = self.model(Xbatch)  # 前向传播计算输出
                # loss
                loss = self.loss_fn(output, ybatch)  # 计算损失值
                all_costs.append(loss.data.item())  # 记录损失值
                # backward
                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播计算梯度
                # Update parameters
                self.optimizer.step()  # 更新模型参数
        self.nepoch += epoch_size  # 更新当前 epoch 数

    # 计算模型在验证集上的准确率
    def score(self, devX, devy):
        self.model.eval()  # 设置模型为评估模式
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()  # 将验证集输入数据移到 GPU 上
            devy = torch.LongTensor(devy).cuda()  # 将验证集标签数据移到 GPU 上
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]  # 获取验证集输入数据批次
                ybatch = devy[i:i + self.batch_size]  # 获取验证集标签数据批次
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()  # 将输入数据移到 GPU 上
                    ybatch = ybatch.cuda()  # 将标签数据移到 GPU 上
                output = self.model(Xbatch)  # 模型前向传播计算输出
                pred = output.data.max(1)[1]  # 获取预测类别
                correct += pred.long().eq(ybatch.data.long()).sum().item()  # 计算正确预测数量
            accuracy = 1.0 * correct / len(devX)  # 计算准确率
        return accuracy

    # 使用模型预测数据类别
    def predict(self, devX):
        self.model.eval()  # 设置模型为评估模式
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()  # 将输入数据移到 GPU 上
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]  # 获取输入数据批次
                output = self.model(Xbatch)  # 模型前向传播计算输出
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())  # 获取预测类别并追加到 yhat 中
        yhat = np.vstack(yhat)  # 垂直堆叠 yhat 数组
        return yhat

    # 使用模型预测数据类别的概率分布
    def predict_proba(self, devX):
        self.model.eval()  # 设置模型为评估模式
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]  # 获取输入数据批次
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())  # 计算输出的 softmax 概率分布
                if not probas:
                    probas = vals  # 初始化 probas
                else:
                    probas = np.concatenate((probas, vals), axis=0)  # 将新计算的概率分布追加到 probas 中
        return probas  # 返回预测的概率分布
"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

# 定义一个MLP类，继承自PyTorchClassifier
class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        # 调用父类构造函数初始化
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        # 初始化各参数，若参数不在params中则使用默认值
        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        # 根据是否为Logistic Regression选择不同的模型结构
        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),  # 输入维度到类别数的线性层
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),  # 输入维度到隐藏单元数的线性层
                nn.Dropout(p=self.dropout),  # Dropout层
                nn.Sigmoid(),  # Sigmoid激活函数
                nn.Linear(params["nhid"], self.nclasses),  # 隐藏单元数到类别数的线性层
            ).cuda()

        # 定义交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        # 获取优化器和其参数
        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg  # 设置权重衰减
```