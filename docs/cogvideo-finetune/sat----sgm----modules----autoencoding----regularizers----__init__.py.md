# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\__init__.py`

```py
# 导入抽象方法装饰器和类型注解
from abc import abstractmethod
# 导入任意类型和元组类型注解
from typing import Any, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从自定义模块导入对角高斯分布
from ....modules.distributions.distributions import DiagonalGaussianDistribution
# 从基类模块导入抽象正则化器
from .base import AbstractRegularizer


# 定义对角高斯正则化器类，继承自抽象正则化器
class DiagonalGaussianRegularizer(AbstractRegularizer):
    # 初始化方法，接收一个布尔值参数，默认值为 True
    def __init__(self, sample: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 设置实例属性 sample
        self.sample = sample

    # 定义获取可训练参数的方法，返回任意类型
    def get_trainable_parameters(self) -> Any:
        # 生成一个空的可迭代对象
        yield from ()

    # 定义前向传播方法，接收一个张量，返回一个元组
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 创建一个空字典，用于存储日志信息
        log = dict()
        # 创建一个对角高斯分布实例，基于输入张量 z
        posterior = DiagonalGaussianDistribution(z)
        # 如果 sample 为 True，进行采样
        if self.sample:
            z = posterior.sample()
        # 否则，使用模式值
        else:
            z = posterior.mode()
        # 计算 KL 散度损失
        kl_loss = posterior.kl()
        # 对 KL 损失求和并平均
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # 将 KL 损失添加到日志字典中
        log["kl_loss"] = kl_loss
        # 返回处理后的张量和日志字典
        return z, log
```