# `bitsandbytes\bitsandbytes\optim\adagrad.py`

```py
# 导入必要的模块
from bitsandbytes.optim.optimizer import Optimizer1State

# 定义 Adagrad 类，继承自 Optimizer1State 类
class Adagrad(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    ):
        # 检查学习率是否大于等于0
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查权重衰减值是否大于等于0
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )
        # 检查 epsilon 值是否大于等于0
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查初始累加器值是否为0
        if initial_accumulator_value != 0.0:
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        # 检查学习率衰减值是否为0
        if lr_decay != 0.0:
            raise ValueError("Lr Decay != 0.0 not supported!")
        # 调用父类的初始化函数，传入参数
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )

# 定义 Adagrad8bit 类，继承自 Optimizer1State 类
class Adagrad8bit(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=8,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
    # 检查学习率是否大于等于0
        if not 0.0 <= lr:
            # 如果学习率小于0，则抛出数值错误异常
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查权重衰减是否大于等于0
        if not 0.0 <= weight_decay:
            # 如果权重衰减小于0，则抛出数值错误异常
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )
        # 检查epsilon值是否大于等于0
        if not 0.0 <= eps:
            # 如果epsilon值小于0，则抛出数值错误异常
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查初始累加器值是否为0
        if initial_accumulator_value != 0.0:
            # 如果初始累加器值不为0，则抛出数值错误异常
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        # 检查学习率衰减是否为0
        if lr_decay != 0.0:
            # 如果学习率衰减不为0，则抛出数值错误异常
            raise ValueError("Lr Decay != 0.0 not supported!")
        # 断言块级别是否为真
        assert block_wise
        # 调用父类的初始化方法，传入参数
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )
# 定义 Adagrad32bit 类，继承自 Optimizer1State 类
class Adagrad32bit(Optimizer1State):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        params,
        lr=1e-2,  # 学习率，默认值为 0.01
        lr_decay=0,  # 学习率衰减，默认值为 0
        weight_decay=0,  # 权重衰减，默认值为 0
        initial_accumulator_value=0,  # 初始累加器值，默认值为 0
        eps=1e-10,  # 微小值，默认值为 1e-10
        optim_bits=32,  # 优化位数，默认值为 32
        args=None,  # 参数，默认值为 None
        min_8bit_size=4096,  # 最小 8 位大小，默认值为 4096
        percentile_clipping=100,  # 百分位剪裁，默认值为 100
        block_wise=True,  # 分块方式，默认值为 True
    ):
        # 如果学习率小于等于 0，抛出异常
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 如果权重衰减小于等于 0，抛出异常
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )
        # 如果微小值小于等于 0，抛出异常
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 如果初始累加器值不等于 0，抛出异常
        if initial_accumulator_value != 0.0:
            raise ValueError("Initial accumulator value != 0.0 not supported!")
        # 如果学习率衰减不等于 0，抛出异常
        if lr_decay != 0.0:
            raise ValueError("Lr Decay != 0.0 not supported!")
        # 调用父类的初始化函数，传入参数
        super().__init__(
            "adagrad",
            params,
            lr,
            (0.0, 0.0),
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
        )
```