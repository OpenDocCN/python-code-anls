# `bitsandbytes\bitsandbytes\optim\rmsprop.py`

```py
# 导入必要的模块和类
from bitsandbytes.optim.optimizer import Optimizer1State

# 定义 RMSprop 类，继承自 Optimizer1State 类
class RMSprop(Optimizer1State):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        # 如果 alpha 为 0，则抛出异常
        if alpha == 0:
            raise NotImplementedError(
                "RMSprop with alpha==0.0 is not supported!"
            )
        # 如果 centered 为 True，则抛出异常
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        # 调用父类的初始化函数，传入参数
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )

# 定义 RMSprop8bit 类，继承自 Optimizer1State 类
class RMSprop8bit(Optimizer1State):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        # 如果 alpha 为 0，则抛出异常
        if alpha == 0:
            raise NotImplementedError(
                "RMSprop with alpha==0.0 is not supported!"
            )
        # 如果 centered 为 True，则抛出异常
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        # 调用父类的初始化函数，传入参数
        super().__init__(
            "rmsprop",
            params,
            lr,
            (alpha, momentum),
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )

# 定义 RMSprop32bit 类，继承自 Optimizer1State 类
class RMSprop32bit(Optimizer1State):
    # 初始化 RMSprop 优化器对象，设置各种参数
    def __init__(
        self,
        params,
        lr=1e-2,  # 学习率，默认值为 0.01
        alpha=0.99,  # alpha 参数，默认值为 0.99
        eps=1e-8,  # eps 参数，默认值为 1e-8
        weight_decay=0,  # 权重衰减，默认值为 0
        momentum=0,  # 动量，默认值为 0
        centered=False,  # 是否使用 Centered RMSprop，默认为 False
        args=None,  # 其他参数，默认为 None
        min_8bit_size=4096,  # 最小 8 位大小，默认值为 4096
        percentile_clipping=100,  # 百分位剪裁，默认值为 100
        block_wise=True,  # 是否分块，默认为 True
    ):

        # 如果 alpha 参数为 0，则抛出未实现的错误
        if alpha == 0:
            raise NotImplementedError(
                "RMSprop with alpha==0.0 is not supported!"
            )
        # 如果 centered 参数为 True，则抛出未实现的错误
        if centered:
            raise NotImplementedError("Centered RMSprop is not supported!")
        # 调用父类的初始化方法，设置 RMSprop 优化器的参数
        super().__init__(
            "rmsprop",  # 优化器类型为 rmsprop
            params,  # 待优化的参数
            lr,  # 学习率
            (alpha, momentum),  # alpha 和 momentum 参数的元组
            eps,  # eps 参数
            weight_decay,  # 权重衰减
            32,  # 精度为 32 位
            args,  # 其他参数
            min_8bit_size,  # 最小 8 位大小
            percentile_clipping,  # 百分位剪裁
            block_wise,  # 是否分块
        )
```