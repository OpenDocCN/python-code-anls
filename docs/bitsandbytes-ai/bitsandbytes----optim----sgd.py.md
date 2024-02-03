# `bitsandbytes\bitsandbytes\optim\sgd.py`

```py
# 导入必要的模块
from bitsandbytes.optim.optimizer import Optimizer1State

# 定义 SGD 类，继承自 Optimizer1State 类
class SGD(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        # 如果动量为0，则抛出异常
        if momentum == 0:
            raise NotImplementedError("SGD without momentum is not supported!")
        # 调用父类的初始化函数
        super().__init__(
            "momentum",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )

# 定义 SGD8bit 类，继承自 Optimizer1State 类
class SGD8bit(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        # 如果动量为0，则抛出异常
        if momentum == 0:
            raise NotImplementedError("SGD without momentum is not supported!")
        # 调用父类的初始化函数，设置优化位数为8
        super().__init__(
            "momentum",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )

# 定义 SGD32bit 类，继承自 Optimizer1State 类
class SGD32bit(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        ):
            # 如果动量为0，则抛出未实现错误
            raise NotImplementedError("SGD without momentum is not supported!")
        # 调用父类的构造函数，初始化优化器
        super().__init__(
            "momentum",
            params,
            lr,
            (momentum, dampening),
            0.0,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )
```