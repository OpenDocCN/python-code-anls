# `.\cogview3-finetune\sat\sgm\modules\autoencoding\regularizers\__init__.py`

```
# 导入抽象方法的库
from abc import abstractmethod
# 导入任意类型和元组类型
from typing import Any, Tuple

# 导入 PyTorch 库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入功能模块
import torch.nn.functional as F

# 导入对角高斯分布类
from ....modules.distributions.distributions import DiagonalGaussianDistribution


# 定义抽象正则化器类，继承自 nn.Module
class AbstractRegularizer(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()  # 调用父类构造方法

    # 前向传播方法，接受一个张量，返回一个张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()  # 抛出未实现错误

    # 抽象方法，获取可训练的参数
    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()  # 抛出未实现错误


# 定义对角高斯正则化器类，继承自 AbstractRegularizer
class DiagonalGaussianRegularizer(AbstractRegularizer):
    # 初始化方法，接受一个布尔参数，默认值为 True
    def __init__(self, sample: bool = True):
        super().__init__()  # 调用父类构造方法
        self.sample = sample  # 存储是否采样的参数

    # 获取可训练参数的方法，返回一个生成器
    def get_trainable_parameters(self) -> Any:
        yield from ()  # 生成器为空，表示没有可训练参数

    # 前向传播方法，接受一个张量，返回一个张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()  # 创建一个空字典，用于记录日志
        posterior = DiagonalGaussianDistribution(z)  # 创建对角高斯分布实例
        if self.sample:  # 如果需要采样
            z = posterior.sample()  # 从后验分布中采样
        else:  # 如果不需要采样
            z = posterior.mode()  # 取后验分布的众数
        kl_loss = posterior.kl()  # 计算 Kullback-Leibler 散度损失
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]  # 计算均值损失
        log["kl_loss"] = kl_loss  # 将 KL 损失记录到日志中
        return z, log  # 返回采样或众数以及日志


# 定义测量困惑度的函数，接受预测的索引和质心数量
def measure_perplexity(predicted_indices, num_centroids):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # 评估聚类的困惑度。当困惑度等于 num_embeddings 时，所有聚类被完全均匀使用
    encodings = (
        F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)  # 将预测索引转换为独热编码
    )
    avg_probs = encodings.mean(0)  # 计算每个质心的平均概率
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()  # 计算困惑度
    cluster_use = torch.sum(avg_probs > 0)  # 计算被使用的聚类数量
    return perplexity, cluster_use  # 返回困惑度和聚类使用情况
```