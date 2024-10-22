# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\base.py`

```py
# 导入抽象方法和类型注解
from abc import abstractmethod
# 导入任意类型和元组类型
from typing import Any, Tuple

# 导入 PyTorch 和功能模块
import torch
import torch.nn.functional as F
# 导入神经网络模块
from torch import nn


# 定义一个抽象正则化器类，继承自 nn.Module
class AbstractRegularizer(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 定义前向传播方法，接受一个张量并返回张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 抛出未实现错误，强制子类实现该方法
        raise NotImplementedError()

    # 定义获取可训练参数的抽象方法
    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        # 抛出未实现错误，强制子类实现该方法
        raise NotImplementedError()


# 定义身份正则化器类，继承自 AbstractRegularizer
class IdentityRegularizer(AbstractRegularizer):
    # 实现前向传播方法，接受一个张量并返回该张量和空字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    # 实现获取可训练参数的方法，返回一个生成器
    def get_trainable_parameters(self) -> Any:
        # 生成器不返回任何值
        yield from ()


# 定义测量困惑度的函数，接受预测索引和质心数量作为参数
def measure_perplexity(predicted_indices: torch.Tensor, num_centroids: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # 评估集群的困惑度，当困惑度等于嵌入数量时，所有集群的使用是完全均匀的
    # 将预测索引转化为独热编码并重塑为二维张量
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    # 计算每个质心的平均概率
    avg_probs = encodings.mean(0)
    # 计算困惑度
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # 计算使用的集群数量
    cluster_use = torch.sum(avg_probs > 0)
    # 返回困惑度和集群使用数量
    return perplexity, cluster_use
```