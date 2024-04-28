# `.\transformers\optimization.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 The Google AI Language Team Authors 和 The HuggingFace Inc. team 所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""PyTorch optimization for BERT model."""

# 导入所需的库
import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

# 导入自定义的模块
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个函数，返回常数 lambda
def _get_constant_lambda(_=None):
    return 1

# 创建一个具有恒定学习率的调度器，使用优化器中设置的学习率
def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)

# 创建一个具有恒定学习率的调度器，当指标停止改善时，学习率会降低
def get_reduce_on_plateau_schedule(optimizer: Optimizer, **kwargs):
    """
    Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        kwargs (`dict`, *optional*):
            Extra parameters to be passed to the scheduler. See `torch.optim.lr_scheduler.ReduceLROnPlateau`
            for possible parameters.

    Return:
        `torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.
    """

    return ReduceLROnPlateau(optimizer, **kwargs)

# 定义一个函数，返回带有预热期的恒定学习率 lambda
def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0

# 创建一个具有预热期的恒定学习率调度器，在预热期间学习率在 0 和优化器中设置的初始学习率之间线性增加
def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    # 使用偏函数 partial 创建一个 lr_lambda 函数，用于设置学习率的调度
    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    # 返回一个 LambdaLR 对象，用于设置学习率的调度
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
# 根据当前步数和预设参数计算带有预热阶段的学习率变化函数
def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    # 如果当前步数小于预热阶段的步数
    if current_step < num_warmup_steps:
        # 返回当前步数除以预热阶段步数的比例，作为学习率
        return float(current_step) / float(max(1, num_warmup_steps))
    # 否则，返回当前步数与总训练步数之差与总训练步数减去预热阶段步数之差的比例，作为学习率
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


# 创建一个具有预热阶段的线性学习率调度器
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # 使用偏函数生成带有预热阶段的学习率变化函数
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    # 返回带有指定学习率变化函数的 LambdaLR 调度器
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 根据当前步数和预设参数计算带有预热阶段的余弦学习率变化函数
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    # 如果当前步数小于预热阶段的步数
    if current_step < num_warmup_steps:
        # 返回当前步数除以预热阶段步数的比例，作为学习率
        return float(current_step) / float(max(1, num_warmup_steps))
    # 否则，计算当前步数处于总训练步数减去预热阶段步数中的进度
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    # 计算余弦学习率变化，其中周期数为预设参数，将进度映射到余弦曲线上
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


# 创建一个具有预热阶段的余弦学习率调度器
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    # 使用偏函数生成带有预热阶段的余弦学习率变化函数
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    # 返回带有指定学习率变化函数的 LambdaLR 调度器
    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # 使用偏函数 partial() 创建一个自定义的学习率衰减函数 lr_lambda
    lr_lambda = partial(
        # 调用 _get_cosine_schedule_with_warmup_lr_lambda 函数，创建一个余弦退火调度的学习率衰减函数
        _get_cosine_schedule_with_warmup_lr_lambda,
        # 设置预热步数参数
        num_warmup_steps=num_warmup_steps,
        # 设置总训练步数参数
        num_training_steps=num_training_steps,
        # 设置余弦退火周期数参数
        num_cycles=num_cycles,
    )
    # 返回一个 LambdaLR 类的实例，使用 optimizer 和自定义的 lr_lambda 函数
    return LambdaLR(optimizer, lr_lambda, last_epoch)
# 定义一个函数，根据当前步数生成带有硬重启和热身的余弦学习率调度
def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    # 如果当前步数小于热身步数，则返回当前步数除以热身步数的比例
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 计算当前进度
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    # 如果进度大于等于1.0，则返回0.0
    if progress >= 1.0:
        return 0.0
    # 返回余弦函数值
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


# 定义一个函数，生成带有热身的余弦学习率调度
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    创建一个学习率调度，根据余弦函数的值从优化器中设置的初始学习率减少到0，带有多个硬重启，在热身期间学习率线性增加从0到优化器中设置的初始学习率。

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            要调度学习率的优化器。
        num_warmup_steps (`int`):
            热身阶段的步数。
        num_training_steps (`int`):
            总的训练步数。
        num_cycles (`int`, *可选*, 默认为1):
            要使用的硬重启次数。
        last_epoch (`int`, *可选*, 默认为-1):
            恢复训练时的最后一个周期的索引。

    Return:
        `torch.optim.lr_scheduler.LambdaLR`，带有适当的调度。
    """

    # 使用偏函数创建学习率函数
    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    # 返回LambdaLR对象
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 定义一个函数，根据当前步数生成带有热身的多项式衰减学习率调度
def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    # 如果当前步数小于热身步数，则返回当前步数除以热身步数的比例
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 如果当前步数大于训练步数，则返回lr_end / lr_init，因为LambdaLR会乘以lr_init
    elif current_step > num_training_steps:
        return lr_end / lr_init
    else:
        # 计算多项式衰减学习率
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # 因为LambdaLR会乘以lr_init


# 定义一个函数，生成带有热身的多项式衰减学习率调度
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    创建一个学习率调度，学习率按照多项式衰减从设置的初始学习率减少。

    Args:
        optimizer:
            要调度学习率的优化器。
        num_warmup_steps:
            热身阶段的步数。
        num_training_steps:
            总的训练步数。
        lr_end:
            最终学习率。
        power:
            多项式衰减的幂。
        last_epoch:
            恢复训练时的最后一个周期的索引。

    Return:
        `torch.optim.lr_scheduler.LambdaLR`，带有适当的调度。
    """
    def get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1):
        """
        返回一个多项式衰减与预热阶段相结合的学习率调度器，该调度器在预热期间从 0 线性增加到优化器中设置的初始学习率，并在预定步数内以指数衰减到最终学习率 *lr_end*。
    
        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                要调度学习率的优化器。
            num_warmup_steps (`int`):
                预热阶段的步数。
            num_training_steps (`int`):
                训练总步数。
            lr_end (`float`, *optional*, 默认值为 1e-7):
                最终学习率。
            power (`float`, *optional*, 默认值为 1.0):
                幂因子。
            last_epoch (`int`, *optional*, 默认值为 -1):
                恢复训练时的最后一个周期的索引。
    
        Note: *power* 默认为 1.0，与 fairseq 实现一致，fairseq 实现又基于原始的 BERT 实现，
        详见 https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    
        Return:
            `torch.optim.lr_scheduler.LambdaLR`，具有适当的调度。
    
        """
    
        # 获取优化器的初始学习率
        lr_init = optimizer.defaults["lr"]
        # 检查最终学习率是否小于初始学习率，若不是，则抛出 ValueError
        if not (lr_init > lr_end):
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")
    
        # 使用 partial 函数构造带参数的 lr_lambda 函数
        lr_lambda = partial(
            _get_polynomial_decay_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            power=power,
            lr_init=lr_init,
        )
        # 返回 LambdaLR 调度器对象
        return LambdaLR(optimizer, lr_lambda, last_epoch)
# 根据当前步数、预热步数以及时间尺度计算逆平方根学习率衰减函数的系数
def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    # 若当前步数小于预热步数，则返回当前步数除以预热步数的比例作为系数，表示线性增长的学习率
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 计算时间偏移
    shift = timescale - num_warmup_steps
    # 计算逆平方根衰减
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    # 返回衰减系数
    return decay


# 创建具有逆平方根学习率调度的函数
def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # 若未提供时间尺度，则默认为预热步数
    if timescale is None:
        timescale = num_warmup_steps

    # 创建一个局部函数，其参数已预先绑定
    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    # 返回一个 LambdaLR 类型的学习率调度器对象
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# 不同调度类型对应的调度器函数字典
TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
}


# 统一获取调度器的函数
def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    Unified API to get any scheduler from its name.
    """
    Args:
        name (`str` or `SchedulerType`):
            要使用的调度器的名称。
        optimizer (`torch.optim.Optimizer`):
            训练期间将使用的优化器。
        num_warmup_steps (`int`, *optional*):
            要执行的热身步数。并非所有调度器都需要此参数（因此参数是可选的），如果未设置并且调度器类型需要它，函数将引发错误。
        num_training_steps (`int`, *optional*):
            要执行的训练步数。并非所有调度器都需要此参数（因此参数是可选的），如果未设置并且调度器类型需要它，函数将引发错误。
        scheduler_specific_kwargs (`dict`, *optional*):
            用于特定调度器的额外参数，例如带重启的余弦调度器。调度器类型和调度器参数不匹配将导致调度器函数引发 TypeError。
    """
    # 将名称转换为 SchedulerType 枚举类型
    name = SchedulerType(name)
    # 获取对应调度器函数
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    
    # 如果调度器类型为 CONSTANT，则直接返回调度器函数应用于优化器的结果
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # 如果未提供 scheduler_specific_kwargs，则将其设置为一个空字典
    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    # 如果调度器类型为 REDUCE_ON_PLATEAU，则返回调度器函数应用于优化器和特定参数的结果
    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # 所有其他调度器都需要 `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # 如果调度器类型为 CONSTANT_WITH_WARMUP 或 INVERSE_SQRT，则返回调度器函数应用于优化器和热身步数的结果
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # 所有其他调度器都需要 `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # 返回调度器函数应用于优化器、热身步数、训练步数和特定参数的结果
    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )
# 定义 AdamW 类，继承自 Optimizer 类
class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    # 初始化方法，接受一系列参数
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        # 如果未设置 no_deprecation_warning 参数，则发出警告
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # 检查 PyTorch 版本是否符合要求
        require_version("torch>=1.5.0")  # add_ with alpha
        # 检查学习率是否为非负数
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        # 检查 beta 参数是否在指定范围内
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        # 检查 epsilon 值是否为非负数
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        # 将参数存储为字典形式
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        # 调用父类的初始化方法
        super().__init__(params, defaults)

    # 使用 torch.no_grad() 修饰的方法
    @torch.no_grad()
    # 定义优化器的单步优化方法
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        # 初始化损失值
        loss = None
        # 如果提供了闭包函数，则重新评估模型并返回损失值
        if closure is not None:
            loss = closure()

        # 遍历优化器中的参数组
        for group in self.param_groups:
            # 遍历当前参数组中的参数
            for p in group["params"]:
                # 如果参数的梯度为None，则跳过此次迭代
                if p.grad is None:
                    continue
                # 获取参数的梯度
                grad = p.grad
                # 如果梯度是稀疏的，则抛出异常，因为Adam不支持稀疏梯度
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # 获取参数的状态信息
                state = self.state[p]

                # 参数状态的初始化
                if len(state) == 0:
                    # 步数初始化为0
                    state["step"] = 0
                    # 梯度值的指数移动平均
                    state["exp_avg"] = torch.zeros_like(p)
                    # 梯度值的平方的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # 获取参数状态中的指数移动平均值和平方的指数移动平均值
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # 获取参数组中的beta1和beta2
                beta1, beta2 = group["betas"]

                # 更新步数
                state["step"] += 1

                # 衰减第一和第二时刻的运行平均系数
                # 使用原地操作同时更新这些平均值
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # 计算分母
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # 获取学习率
                step_size = group["lr"]
                # 如果需要进行偏差校正（例如Bert），则不进行偏差校正
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # 更新参数值
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 正确使用L2正则化/权重衰减的方法
                # 此处添加权重衰减
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        # 返回损失值
        return loss
class Adafactor(Optimizer):
    """
    AdaFactor是一个基于PyTorch的优化器实现，可以作为Adam的替代方案。原始的fairseq代码可参见:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    论文: *Adafactor: 自适应学习率与次线性内存成本* https://arxiv.org/abs/1804.04235 注意，此优化器根据`scale_parameter`、`relative_step`和
    `warmup_init`选项内部调整学习率。要使用手动（外部）学习率调度，您应该设置`scale_parameter=False`和
    `relative_step=False`。

    参数:
        params (`Iterable[nn.parameter.Parameter]`):
            要优化的参数的迭代器或定义参数组的字典。
        lr (`float`, *可选*):
            外部学习率。
        eps (`Tuple[float, float]`, *可选*, 默认为`(1e-30, 0.001)`):
            平方梯度和参数规模的正则化常数
        clip_threshold (`float`, *可选*, 默认为1.0):
            最终梯度更新的均方根的阈值
        decay_rate (`float`, *可选*, 默认为-0.8):
            用于计算平方的运行平均值的系数
        beta1 (`float`, *可选*):
            用于计算梯度的运行平均值的系数
        weight_decay (`float`, *可选*, 默认为0.0):
            权重衰减（L2惩罚）
        scale_parameter (`bool`, *可选*, 默认为`True`):
            如果为True，则学习率将按均方根进行缩放
        relative_step (`bool`, *可选*, 默认为`True`):
            如果为True，则计算时间依赖的学习率，而不是外部学习率
        warmup_init (`bool`, *可选*, 默认为`False`):
            时间依赖的学习率计算取决于是否正在使用预热初始化

    此实现处理低精度（FP16、bfloat）值，但我们尚未进行全面测试。

    推荐的T5微调设置(https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - 不推荐没有LR预热或clip_threshold的训练。

           - 使用调度的LR预热到固定的LR
           - 使用clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - 禁用相对更新
        - 使用scale_parameter=False
        - 不应该与Adafactor一起使用额外的优化器操作，如梯度裁剪

    示例:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```py

    其他人报告以下组合效果很好:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```py

    当使用 `lr=None` 与 [`Trainer`] 时，您很可能需要使用 [`~optimization.AdafactorSchedule`]
    调度器如下:

    ```python
    ```py
    # 从transformers.optimization模块中导入Adafactor和AdafactorSchedule类
    from transformers.optimization import Adafactor, AdafactorSchedule
    
    # 创建Adafactor优化器对象，使用model的参数，设置各种参数选项
    optimizer = Adafactor(
        model.parameters(),  # 使用模型的参数
        lr=1e-3,             # 学习率设为1e-3
        eps=(1e-30, 1e-3),   # epsilon值设为(1e-30, 1e-3)
        clip_threshold=1.0,  # 梯度裁剪阈值设为1.0
        decay_rate=-0.8,     # 衰减率设为-0.8
        beta1=None,          # beta1设为None
        weight_decay=0.0,    # 权重衰减设为0.0
        relative_step=False, # 相对步长设为False
        scale_parameter=False, # 参数缩放设为False
        warmup_init=False,   # 初始热身设为False
    )
    
    # 创建AdafactorSchedule对象，使用之前创建的Adafactor优化器对象
    lr_scheduler = AdafactorSchedule(optimizer)
    
    # 创建Trainer对象，其余参数不变，替换优化器和学习率调度器
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
class AdafactorSchedule(LambdaLR):
    """
    Since [`~optimization.Adafactor`] performs its own scheduling, if the training loop relies on a scheduler (e.g.,
    for logging), this class creates a proxy object that retrieves the current lr values from the optimizer.

    It returns `initial_lr` during startup and the actual `lr` during stepping.
    """

    def __init__(self, optimizer, initial_lr=0.0):
        # 定义一个 lambda 函数，始终返回初始学习率 initial_lr
        def lr_lambda(_):
            return initial_lr

        # 将 initial_lr 添加到每个参数组的字典中
        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        # 调用父类 LambdaLR 的构造函数，传入优化器 optimizer 和定义的 lambda 函数
        super().__init__(optimizer, lr_lambda)
        # 删除每个参数组中的 initial_lr 字段，避免干扰后续步骤
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        # 获取当前优化器对象
        opt = self.optimizer
        # 获取每个参数组的学习率，如果梯度不为 None，则获取当前参数组的学习率
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        # 如果没有可用的学习率值，则返回基础学习率 base_lrs（在执行步进之前调用）
        if len(lrs) == 0:
            lrs = self.base_lrs
        return lrs


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    """
    Get a proxy schedule for [`~optimization.Adafactor`]

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        initial_lr (`float`, *optional*, defaults to 0.0):
            Initial lr

    Return:
        [`~optimization.Adafactor`] proxy schedule object.


    """
    # 返回 AdafactorSchedule 类的实例，传入优化器 optimizer 和初始学习率 initial_lr
    return AdafactorSchedule(optimizer, initial_lr)
```