# `bitsandbytes\bitsandbytes\optim\adam.py`

```py
# 导入所需的库
import math
import os

import torch
import torch.distributed as dist

# 导入自定义的功能模块和优化器模块
import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import Optimizer2State

# 定义 Adam 优化器类，继承自 Optimizer2State 类
class Adam(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# 定义 Adam8bit 优化器类，继承自 Optimizer2State 类
class Adam8bit(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数，设置优化位数为 8
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# 定义 Adam32bit 优化器类，继承自 Optimizer2State 类
class Adam32bit(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数，设置优化位数为 32
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

# 定义 PagedAdam 优化器类，继承自 Optimizer2State 类
class PagedAdam(Optimizer2State):
    # 初始化方法，接受一系列参数
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 调用父类的初始化方法，传入参数，设置为分页模式
        super().__init__( "adam", params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)
class PagedAdam8bit(Optimizer2State):
    # 定义一个名为 PagedAdam8bit 的类，继承自 Optimizer2State
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 初始化方法，接受一系列参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)

class PagedAdam32bit(Optimizer2State):
    # 定义一个名为 PagedAdam32bit 的类，继承自 Optimizer2State
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, optim_bits=32,
                       args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        # 初始化方法，接受一系列参数
        super().__init__( "adam", params, lr, betas, eps, weight_decay, 32, args, min_8bit_size, percentile_clipping, block_wise, is_paged=True)

class AnalysisAdam(torch.optim.Optimizer):
    # 定义一个名为 AnalysisAdam 的类，继承自 torch.optim.Optimizer
    """Adam that performs 8-bit vs 32-bit error analysis.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """
    # 一个带有详细说明的 Adam 优化器，用于执行 8 位与 32 位误差分析
    # 定义一个链接，指向 "On the Convergence of Adam and Beyond" 的网页
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    # 初始化 AdamW 优化器类
    def __init__(
        self,
        params,  # 参数列表
        lr=1e-3,  # 学习率，默认值为 0.001
        betas=(0.9, 0.999),  # Adam 优化器的 beta 参数，默认为 (0.9, 0.999)
        eps=1e-8,  # 用于数值稳定性的小值，默认为 1e-8
        weight_decay=0,  # 权重衰减，默认为 0
        amsgrad=False,  # 是否使用 AMSGrad，默认为 False
        bnb_analysis="dynamic-blockwise",  # BNB 分析方法，默认为 "dynamic-blockwise"
        savedir=None,  # 保存目录，默认为 None
    ):
        # 初始化默认参数字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        # 调用父类的初始化方法，传入参数列表和默认参数字典
        super().__init__(params, defaults)
        # 设置分析方法
        self.analysis = bnb_analysis
        # 设置保存目录
        self.savedir = savedir

    # 返回是否支持内存高效的 FP16
    @property
    def supports_memory_efficient_fp16(self):
        return True

    # 返回是否支持扁平参数
    @property
    def supports_flat_params(self):
        return True
```