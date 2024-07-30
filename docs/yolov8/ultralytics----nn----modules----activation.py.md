# `.\yolov8\ultralytics\nn\modules\activation.py`

```py
# 导入 PyTorch 库中需要的模块
import torch
import torch.nn as nn

# 定义 AGLU 类，继承自 nn.Module 类，表示这是一个 PyTorch 模型模块
class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """初始化统一激活函数模块。"""
        super().__init__()
        # 使用 Softplus 激活函数初始化 self.act，beta 参数设为 -1.0
        self.act = nn.Softplus(beta=-1.0)
        # 使用均匀分布初始化 lambda 参数，作为可学习参数 nn.Parameter
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        # 使用均匀分布初始化 kappa 参数，作为可学习参数 nn.Parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算统一激活函数的前向传播。"""
        # 限制 lambda 参数的最小值为 0.0001
        lam = torch.clamp(self.lambd, min=0.0001)
        # 计算统一激活函数的输出
        y = torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
        # 返回统一激活函数的输出结果
        return y  # for AGLU simply return y * input
```