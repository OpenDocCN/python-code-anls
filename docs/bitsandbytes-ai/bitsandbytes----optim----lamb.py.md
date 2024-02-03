# `bitsandbytes\bitsandbytes\optim\lamb.py`

```py
# 导入必要的模块
from bitsandbytes.optim.optimizer import Optimizer2State

# 定义 LAMB 类，继承自 Optimizer2State 类
class LAMB(Optimizer2State):
    # 初始化方法
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adam_w_mode=True,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=False,
        max_unorm=1.0,
    ):
        # 调用父类的初始化方法
        super().__init__(
            "lamb",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            max_unorm=1.0,
        )

# 定义 LAMB8bit 类，继承自 Optimizer2State 类
class LAMB8bit(Optimizer2State):
    # 初始化方法
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adam_w_mode=True,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=False,
        max_unorm=1.0,
    ):
        # 调用父类的初始化方法
        super().__init__(
            "lamb",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            max_unorm=1.0,
        )

# 定义 LAMB32bit 类，继承自 Optimizer2State 类
class LAMB32bit(Optimizer2State):
    # 初始化函数，设置各种参数和默认数值
    def __init__(
        self,
        params,  # 参数列表
        lr=1e-3,  # 学习率，默认值为 0.001
        bias_correction=True,  # 是否进行偏置校正，默认为 True
        betas=(0.9, 0.999),  # Adam 优化器的 beta 参数，默认为 (0.9, 0.999)
        eps=1e-8,  # 用于数值稳定性的小值，默认为 1e-8
        weight_decay=0,  # 权重衰减，默认为 0
        amsgrad=False,  # 是否使用 AMSGrad，默认为 False
        adam_w_mode=True,  # 是否使用 AdamW 模式，默认为 True
        args=None,  # 其他参数，默认为 None
        min_8bit_size=4096,  # 最小 8 位大小，默认为 4096
        percentile_clipping=100,  # 百分位裁剪，默认为 100
        block_wise=False,  # 是否进行块状处理，默认为 False
        max_unorm=1.0,  # 最大无量纲值，默认为 1.0
    ):
        # 调用父类的初始化函数，设置优化器名称为 "lamb"，传入参数列表和其他参数
        super().__init__(
            "lamb",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            32,  # 默认的梯度缩放因子为 32
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            max_unorm=1.0,  # 最大无量纲值为 1.0
        )
```