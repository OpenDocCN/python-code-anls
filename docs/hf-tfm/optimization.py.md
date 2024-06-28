# `.\optimization.py`

```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from .trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version

# 获取logger对象，用于日志记录
logger = logging.get_logger(__name__)


def _get_constant_lambda(_=None):
    # 返回常数学习率调度函数
    return 1


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
    # 使用LambdaLR创建常数学习率调度器
    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


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
    # 使用ReduceLROnPlateau创建学习率调度器，根据指标停止改善时降低学习率
    return ReduceLROnPlateau(optimizer, **kwargs)


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        # 在预热阶段线性增加学习率
        return float(current_step) / float(max(1.0, num_warmup_steps))
    # 达到预热步数后保持恒定学习率
    return 1.0


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
    # 使用LambdaLR创建带预热的常数学习率调度器
    return LambdaLR(optimizer, partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps), last_epoch=last_epoch)
    # 定义一个偏函数，用于生成包含预热学习率调度的 Lambda 函数
    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    # 返回一个 LambdaLR 对象，将给定的优化器和生成的 Lambda 函数作为参数
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    # 如果当前步数小于预热步数，则返回一个线性增长的学习率比例因子
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 否则返回一个线性衰减的学习率比例因子
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


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
    # 使用 partial 函数固定部分参数，创建一个 lambda 函数作为学习率调度器的输入
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    # 返回一个 LambdaLR 类型的学习率调度器对象
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    # 如果当前步数小于预热步数，则返回一个线性增长的学习率比例因子
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 否则计算余弦衰减的学习率比例因子
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


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
    # 使用 partial 函数固定部分参数，创建一个 lambda 函数作为学习率调度器的输入
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    # 返回一个 LambdaLR 类型的学习率调度器对象
    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # 定义一个部分应用的函数 `_get_cosine_schedule_with_warmup_lr_lambda` 的 Lambda 函数，用于生成学习率调度函数
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,  # 设置预热步数参数
        num_training_steps=num_training_steps,  # 设置训练步数参数
        num_cycles=num_cycles,  # 设置余弦退火周期数参数
    )
    # 返回一个 LambdaLR 对象，将这个 Lambda 函数应用于优化器的学习率调度
    return LambdaLR(optimizer, lr_lambda, last_epoch)
# 定义一个函数，根据余弦函数值生成学习率调度表，带有硬重启，并在预热期间逐步增加学习率
def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    # 如果当前步数小于预热步数，返回线性增长的学习率比例
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 计算当前进度，基于训练步数和预热步数，生成一个 0 到 1 的进度值
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    # 如果进度超过或等于 1.0，返回学习率为 0.0
    if progress >= 1.0:
        return 0.0
    # 否则，根据余弦函数生成学习率，带有硬重启的周期性
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


# 创建一个学习率调度对象，基于给定的优化器和参数，返回 LambdaLR 调度器对象
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # 使用偏函数创建 LambdaLR 调度器，使用 _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda 函数
    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    # 返回 LambdaLR 调度器对象
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 定义一个函数，根据多项式衰减生成学习率调度表，带有预热期间的逐步增加学习率
def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    # 如果当前步数小于预热步数，返回线性增长的学习率比例
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 如果当前步数大于训练步数，返回最终学习率与初始学习率的比值
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        # 计算多项式衰减的学习率
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


# 创建一个学习率调度对象，基于给定的优化器和参数，返回 LambdaLR 调度器对象
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to `lr_end` after `num_training_steps`, with a linear warmup over `num_warmup_steps` steps.

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, optional, defaults to 1e-7):
            The final learning rate after the decay.
        power (`float`, optional, defaults to 1.0):
            Power factor for polynomial decay.
        last_epoch (`int`, optional, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # 使用偏函数创建 LambdaLR 调度器，使用 _get_polynomial_decay_schedule_with_warmup_lr_lambda 函数
    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=optimizer.param_groups[0]['lr'],
    )
    # 返回 LambdaLR 调度器对象
    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # 使用多项式衰减调度器生成学习率调度方案，包括预热期和最终学习率结束的定义
    # 根据优化器调度学习率
    # Args:
    #   optimizer ([`~torch.optim.Optimizer`]): 要调度学习率的优化器
    #   num_warmup_steps (`int`): 预热阶段的步数
    #   num_training_steps (`int`): 总训练步数
    #   lr_end (`float`, *optional*, defaults to 1e-7): 最终的学习率
    #   power (`float`, *optional*, defaults to 1.0): 幂因子
    #   last_epoch (`int`, *optional*, defaults to -1): 在恢复训练时的最后一个 epoch 的索引

    # 提取初始学习率
    lr_init = optimizer.defaults["lr"]
    # 检查初始学习率和最终学习率的合理性
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    # 部分函数定义，生成多项式衰减与预热期学习率衰减方案的 lambda 函数
    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    # 返回 LambdaLR 对象，用于优化器的学习率调度
    return LambdaLR(optimizer, lr_lambda, last_epoch)
# 根据当前步数和预热步数计算逆平方根学习率衰减函数的值
def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    # 如果当前步数小于预热步数，则返回线性增长的学习率值
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    # 计算衰减的时间偏移量
    shift = timescale - num_warmup_steps
    # 计算衰减系数，根据当前步数和时间尺度
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    创建一个逆平方根学习率调度，从优化器中设置的初始学习率开始，在一个预热期间内线性增加学习率，从0增加到初始学习率。

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            要调度学习率的优化器。
        num_warmup_steps (`int`):
            预热阶段的步数。
        timescale (`int`, *可选*, 默认为 `num_warmup_steps`):
            时间尺度。
        last_epoch (`int`, *可选*, 默认为 -1):
            恢复训练时的最后一个周期索引。

    Returns:
        `torch.optim.lr_scheduler.LambdaLR`：带有适当调度的对象。
    """
    # 注意：此实现修改自
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    # 如果未指定时间尺度，则使用默认值（预热步数或默认值10000）
    if timescale is None:
        timescale = num_warmup_steps or 10_000

    # 创建一个局部函数，用于 LambdaLR 对象的 lr_lambda 参数
    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,  # 关联逆平方根调度函数
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """
    统一的 API 通过名称获取任何调度器。

    Args:
        name (Union[str, SchedulerType]):
            调度器的名称。
        optimizer (Optimizer):
            要调度学习率的优化器。
        num_warmup_steps (Optional[int], 可选):
            预热阶段的步数。
        num_training_steps (Optional[int], 可选):
            训练总步数。
        scheduler_specific_kwargs (Optional[dict], 可选):
            特定于调度器的其他参数。

    """
    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int`, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    # Convert `name` to SchedulerType enum
    name = SchedulerType(name)
    # Retrieve the scheduler function corresponding to `name`
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # If `optimizer` is a LayerWiseDummyOptimizer, recursively fetch schedulers for each parameter
    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}

        # Iterate over optimizer parameters and fetch corresponding schedulers
        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Define a scheduler hook for each parameter to apply the respective scheduler step
        def scheduler_hook(param):
            if param.grad is not None:
                scheduler_dict[param].step()

        # Register the scheduler hook for each parameter that requires gradients
        for param in optimizer_dict.keys():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        # Return a LayerWiseDummyScheduler instance
        return LayerWiseDummyScheduler()

    # For constant scheduler types, directly apply the scheduler function on `optimizer`
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # Handle REDUCE_ON_PLATEAU scheduler type with specific kwargs if provided
    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}
    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # Raise an error if `num_warmup_steps` is not provided for required scheduler types
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # Apply schedulers requiring `num_warmup_steps` with the provided value
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps` to be provided
    # 如果 `num_training_steps` 参数为 None，则抛出 ValueError 异常，指示需要提供 `num_training_steps` 参数
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    # 调用指定的 `schedule_func` 函数，传入以下参数：
    # - optimizer: 优化器对象
    # - num_warmup_steps: 预热步数
    # - num_training_steps: 训练步数
    # - **scheduler_specific_kwargs: 其他特定调度器的关键字参数，传递给 `schedule_func`
    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )
    # 定义 AdamW 优化器类，继承自 Optimizer 类
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
        # 如果未禁用不推荐使用警告，则发出未来将删除警告
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # 检查学习率是否非负
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        # 检查 beta 参数是否在有效范围内
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        # 检查 epsilon 是否非负
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        # 设置默认参数字典
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        # 调用父类 Optimizer 的构造函数进行初始化
        super().__init__(params, defaults)

    @torch.no_grad()
    # 执行单个优化步骤的方法
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        # 如果提供了闭包函数，重新评估模型并返回损失
        if closure is not None:
            loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历当前参数组中的参数
            for p in group["params"]:
                # 如果参数没有梯度，继续下一个参数
                if p.grad is None:
                    continue
                grad = p.grad
                # 如果梯度是稀疏的，Adam 不支持稀疏梯度，建议使用 SparseAdam
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # 获取或初始化该参数的状态信息
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state["step"] = 0
                    # 梯度值的指数移动平均
                    state["exp_avg"] = torch.zeros_like(p)
                    # 梯度值平方的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # 获取当前参数的 exp_avg 和 exp_avg_sq
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # 更新步数
                state["step"] += 1

                # 更新梯度值的指数移动平均和平方梯度值的指数移动平均
                # 使用原地操作同时更新平均值
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                # 如果需要修正偏差（比如对BERT），不进行偏差修正
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # 更新参数值
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 使用Adam进行权重衰减，与梯度平方移动平均无关
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        # 返回损失值（如果有）
        return loss
class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    ```

    def __init__(self, params, lr=None, eps=(1e-30, 0.001), clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0,
                 scale_parameter=True, relative_step=True, warmup_init=False):
        """
        Initialize Adafactor optimizer

        Args:
            params (Iterable[nn.parameter.Parameter]): Iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): External learning rate (default: None)
            eps (Tuple[float, float], optional): Regularization constants for square gradient and parameter scale (default: (1e-30, 0.001))
            clip_threshold (float, optional): Threshold of root mean square of final gradient update (default: 1.0)
            decay_rate (float, optional): Coefficient used to compute running averages of square (default: -0.8)
            beta1 (float, optional): Coefficient used for computing running averages of gradient (default: None)
            weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0)
            scale_parameter (bool, optional): If True, learning rate is scaled by root mean square (default: True)
            relative_step (bool, optional): If True, time-dependent learning rate is computed instead of external learning rate (default: True)
            warmup_init (bool, optional): Time-dependent learning rate computation depends on whether warm-up initialization is being used (default: False)
        """
        # 调用父类构造函数，初始化优化器
        super(Adafactor, self).__init__(params, defaults=dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                                                             decay_rate=decay_rate, beta1=beta1,
                                                             weight_decay=weight_decay))
        
        # 设置 Adafactor 特有的参数
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            None
        """
        # 获取当前的学习率
        lr = self.defaults['lr']
        if lr is None:
            raise ValueError('Learning rate is required for Adafactor optimizer')

        # 对参数进行更新步骤
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # 参数状态初始化
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.scale_parameter:
                        state['exp_avg_sq_prm'] = torch.zeros_like(p.data)

                # 获取当前状态中的平方梯度估计
                exp_avg_sq = state['exp_avg_sq']
                if self.scale_parameter:
                    exp_avg_sq_prm = state['exp_avg_sq_prm']

                # 更新步骤数
                state['step'] += 1
                bias_correction = 1 - self.decay_rate ** state['step']

                # 计算梯度平方的指数加权平均值
                if self.scale_parameter:
                    grad_sq = grad.pow(2).add_(group['eps'][0])
                    exp_avg_sq.mul_(self.decay_rate).add_(1.0 - self.decay_rate, grad_sq)
                    rms = exp_avg_sq_prm.mul(1 - bias_correction).sqrt().add_(group['eps'][1])
                    p.data.addcdiv_(-lr, grad, rms)
                else:
                    grad_sq = grad.pow(2).add_(group['eps'][0])
                    exp_avg_sq.mul_(self.decay_rate).add_(1.0 - self.decay_rate, grad_sq)
                    rms = exp_avg_sq.sqrt().add_(group['eps'][1])
                    p.data.addcdiv_(-lr, grad, rms)

                # 阈值剪裁
                if group['clip_threshold'] > 0:
                    rms_clipped = rms.clamp(min=group['clip_threshold'])
                    p.data.div_(rms_clipped)

                # L2 正则化
                if group['weight_decay'] > 0:
                    p.data.add_(-group['weight_decay'], p.data)

                # 存储更新后的平方梯度估计
                exp_avg_sq.copy_(exp_avg_sq)

        return None
    # 从transformers.optimization模块导入Adafactor和AdafactorSchedule优化器
    from transformers.optimization import Adafactor, AdafactorSchedule

    # 使用Adafactor优化器初始化，设置相关参数
    optimizer = Adafactor(
        model.parameters(),  # 传入模型的参数
        lr=1e-3,  # 学习率设置为1e-3
        eps=(1e-30, 1e-3),  # epsilon参数设置为(1e-30, 1e-3)
        clip_threshold=1.0,  # 梯度裁剪阈值设置为1.0
        decay_rate=-0.8,  # 衰减率设置为-0.8
        beta1=None,  # beta1设置为None
        weight_decay=0.0,  # 权重衰减设置为0.0
        relative_step=False,  # 相对步长设置为False
        scale_parameter=False,  # 参数缩放设置为False
        warmup_init=False,  # 初始化时的预热设置为False
    )

    # 使用AdafactorSchedule初始化学习率调度器
    lr_scheduler = AdafactorSchedule(optimizer)

    # 使用Trainer初始化训练器，传入相应的参数和优化器
    trainer = Trainer(
        ...,  # 其他训练器的参数，未提供具体内容
        optimizers=(optimizer, lr_scheduler)  # 设置优化器为前面初始化的optimizer和lr_scheduler
    )
class AdafactorSchedule(LambdaLR):
    """
    [`~optimization.Adafactor`] 自行执行调度，如果训练循环依赖于调度器（例如用于日志记录），此类创建代理对象，
    从优化器中检索当前 lr 值。

    在启动期间返回 `initial_lr`，在步进期间返回实际的 `lr`。
    """

    def __init__(self, optimizer, initial_lr=0.0):
        # 定义一个 lambda 函数返回初始 lr
        def lr_lambda(_):
            return initial_lr

        # 为每个参数组设置 "initial_lr" 键值对
        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        # 调用父类 LambdaLR 的初始化方法
        super().__init__(optimizer, lr_lambda)
        # 删除每个参数组的 "initial_lr" 键值对
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        # 获取优化器对象
        opt = self.optimizer
        # 获取每个参数组的学习率列表
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        # 如果没有可用的学习率值，则使用基础学习率列表
        if len(lrs) == 0:
            lrs = self.base_lrs  # 如果在步进之前调用
        return lrs


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    """
    获取 [`~optimization.Adafactor`] 的代理调度对象

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            要调度学习率的优化器。
        initial_lr (`float`, *可选*, 默认为 0.0):
            初始学习率

    Return:
        [`~optimization.Adafactor`] 的代理调度对象。
    """
    return AdafactorSchedule(optimizer, initial_lr)
```