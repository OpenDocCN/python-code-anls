# `.\cogvideo-finetune\sat\vae_modules\regularizers.py`

```py
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 导入 Any 和 Tuple 类型注解
from typing import Any, Tuple

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能函数
import torch.nn.functional as F
# 从 PyTorch 导入神经网络模块
from torch import nn


# 定义对角高斯分布类
class DiagonalGaussianDistribution(object):
    # 初始化方法，接收参数和是否为确定性
    def __init__(self, parameters, deterministic=False):
        # 存储传入的参数
        self.parameters = parameters
        # 将参数分为均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 将对数方差限制在-30到20的范围内
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # 记录是否为确定性
        self.deterministic = deterministic
        # 计算标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 计算方差
        self.var = torch.exp(self.logvar)
        # 如果是确定性，则标准差和方差设置为0
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # 生成样本的方法
    def sample(self):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(
        #     device=self.parameters.device
        # )
        # 从均值和标准差生成样本
        x = self.mean + self.std * torch.randn_like(self.mean)
        # 返回生成的样本
        return x

    # 计算KL散度的方法
    def kl(self, other=None):
        # 如果是确定性，则返回0
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            # 如果没有提供其他分布，则计算与标准正态分布的KL散度
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 否则计算与另一个分布的KL散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 计算负对数似然的方法
    def nll(self, sample, dims=[1, 2, 3]):
        # 如果是确定性，则返回0
        if self.deterministic:
            return torch.Tensor([0.0])
        # 计算2π的对数
        logtwopi = np.log(2.0 * np.pi)
        # 计算负对数似然
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 返回模式（均值）的方法
    def mode(self):
        return self.mean


# 定义抽象正则化器类，继承自nn.Module
class AbstractRegularizer(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()

    # 前向传播的方法，需实现
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    # 获取可训练参数的抽象方法
    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


# 定义身份正则化器类，继承自抽象正则化器
class IdentityRegularizer(AbstractRegularizer):
    # 前向传播方法，返回输入和空字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    # 获取可训练参数的方法，返回空生成器
    def get_trainable_parameters(self) -> Any:
        yield from ()


# 定义测量困惑度的函数
def measure_perplexity(predicted_indices: torch.Tensor, num_centroids: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # 评估聚类困惑度。当困惑度 == num_embeddings 时，所有聚类被完全均匀使用
    # 对预测索引进行独热编码并重塑为二维张量
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    # 计算平均概率
    avg_probs = encodings.mean(0)
    # 计算困惑度
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # 计算使用的聚类数量
    cluster_use = torch.sum(avg_probs > 0)
    # 返回困惑度和聚类使用情况
    return perplexity, cluster_use
# 定义一个对角高斯正则化器类，继承自抽象正则化器
class DiagonalGaussianRegularizer(AbstractRegularizer):
    # 初始化方法，接受一个布尔参数 sample，默认为 True
    def __init__(self, sample: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 保存 sample 参数
        self.sample = sample

    # 获取可训练参数的方法，返回生成器
    def get_trainable_parameters(self) -> Any:
        # 生成一个空的生成器
        yield from ()

    # 前向传播方法，接收一个张量 z，返回一个张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 初始化一个空字典用于存储日志信息
        log = dict()
        # 创建一个对角高斯分布对象，基于输入张量 z
        posterior = DiagonalGaussianDistribution(z)
        # 根据 sample 参数决定如何获取样本
        if self.sample:
            # 从后验分布中采样
            z = posterior.sample()
        else:
            # 获取后验分布的众数
            z = posterior.mode()
        # 计算 KL 散度损失
        kl_loss = posterior.kl()
        # 对 KL 散度损失取平均
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # 将 KL 损失存储在日志字典中
        log["kl_loss"] = kl_loss
        # 返回处理后的张量和日志字典
        return z, log
```