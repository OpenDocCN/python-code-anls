# `.\pytorch\torch\ao\quantization\experimental\adaround_loss.py`

```
from typing import Tuple  # 导入类型提示模块

import numpy as np  # 导入NumPy库

import torch  # 导入PyTorch库
from torch.nn import functional as F  # 导入PyTorch的函数模块

ADAROUND_ZETA: float = 1.1  # 定义ADAROUND_ZETA常量
ADAROUND_GAMMA: float = -0.1  # 定义ADAROUND_GAMMA常量


class AdaptiveRoundingLoss(torch.nn.Module):
    """
    自适应舍入损失函数，详见 https://arxiv.org/pdf/2004.10568.pdf
    舍入正则化在公式 [24] 中描述
    重构损失在公式 [25] 中描述，不包括正则化项
    """

    def __init__(
        self,
        max_iter: int,
        warm_start: float = 0.2,
        beta_range: Tuple[int, int] = (20, 2),
        reg_param: float = 0.001,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter  # 最大迭代次数
        self.warm_start = warm_start  # 温暖启动阶段所占比例
        self.beta_range = beta_range  # beta范围元组
        self.reg_param = reg_param  # 正则化参数

    def rounding_regularization(
        self,
        V: torch.Tensor,
        curr_iter: int,
    ) -> torch.Tensor:
        """
        从官方Adaround实现中复制的主要逻辑。
        对输入张量V应用舍入正则化。
        """
        assert (
            curr_iter < self.max_iter
        ), "当前迭代次数严格小于最大迭代次数"
        if curr_iter < self.warm_start * self.max_iter:
            return torch.tensor(0.0)  # 在温暖启动阶段返回0张量
        else:
            start_beta, end_beta = self.beta_range
            warm_start_end_iter = self.warm_start * self.max_iter

            # 计算当前迭代的相对迭代次数
            rel_iter = (curr_iter - warm_start_end_iter) / (
                self.max_iter - warm_start_end_iter
            )
            beta = end_beta + 0.5 * (start_beta - end_beta) * (
                1 + np.cos(rel_iter * np.pi)
            )

            # 用于软量化的修正Sigmoid，如https://arxiv.org/pdf/2004.10568.pdf中第23条公式所述
            h_alpha = torch.clamp(
                torch.sigmoid(V) * (ADAROUND_ZETA - ADAROUND_GAMMA) + ADAROUND_GAMMA,
                min=0,
                max=1,
            )

            # 应用舍入正则化
            # 此正则化项有助于在优化结束时使项收敛为二进制解（0或1）
            inner_term = torch.add(2 * h_alpha, -1).abs().pow(beta)
            regularization_term = torch.add(1, -inner_term).sum()
            return regularization_term * self.reg_param

    def reconstruction_loss(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算软量化输出与原始输出之间的重构损失。
        """
        return F.mse_loss(
            soft_quantized_output, original_output, reduction="none"
        ).mean()

    def forward(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
        V: torch.Tensor,
        curr_iter: int,
    ) -> torch.Tensor:
        """
        前向传播函数，计算整体损失。
        """
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the asymmetric reconstruction formulation as eq [25]
        """
        # 计算不对称重构形式，参考方程 [25]
        
        # 计算规则化项，使用当前迭代次数和V作为参数
        regularization_term = self.rounding_regularization(V, curr_iter)
        
        # 计算重构损失，使用软量化输出和原始输出作为参数
        reconstruction_term = self.reconstruction_loss(
            soft_quantized_output, original_output
        )
        
        # 返回计算得到的规则化项和重构项
        return regularization_term, reconstruction_term
```