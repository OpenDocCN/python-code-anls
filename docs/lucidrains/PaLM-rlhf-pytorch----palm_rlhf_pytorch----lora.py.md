# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\lora.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# LoRA - https://arxiv.org/abs/2106.09685

# 定义 LoRA 类，继承自 nn.Module 类
class LoRA(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_out,
        r = 8,
        alpha = None
    ):
        super().__init__()
        # 如果 alpha 不存在，则使用 r 作为默认值
        alpha = default(alpha, r)
        # 计算缩放因子
        self.scale = alpha / r

        # 定义 A 和 B 为可学习参数
        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))

    # 定义 weight 属性，返回 A 和 B 的乘积再乘以缩放因子
    @property
    def weight(self):
        return (self.A @ self.B) * self.scale

    # 前向传播函数，返回输入 x 与权重 weight 的乘积
    def forward(self, x):
        return x @ self.weight
```