# `.\diffusers\optimization.py`

```py
# coding=utf-8  # 指定源文件的编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 版权声明，标明版权归 HuggingFace Inc. 所有
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 指明该文件遵循 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 说明需遵守许可证使用文件
# You may obtain a copy of the License at  # 提供许可证的获取地址
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的具体链接
#
# Unless required by applicable law or agreed to in writing, software  # 说明软件在许可证下是按“原样”分发
# distributed under the License is distributed on an "AS IS" BASIS,  # 明确不提供任何明示或暗示的担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 声明没有任何类型的担保
# See the License for the specific language governing permissions and  # 参考许可证以了解具体权限
# limitations under the License.  # 以及使用限制
"""PyTorch optimization for diffusion models."""  # 模块说明：用于扩散模型的 PyTorch 优化

import math  # 导入数学模块
from enum import Enum  # 从 enum 模块导入 Enum 类
from typing import Optional, Union  # 导入类型提示：Optional 和 Union

from torch.optim import Optimizer  # 从 torch.optim 导入 Optimizer 类
from torch.optim.lr_scheduler import LambdaLR  # 从 torch.optim.lr_scheduler 导入 LambdaLR 类

from .utils import logging  # 从当前包的 utils 模块导入 logging

logger = logging.get_logger(__name__)  # 创建一个记录器，用于记录当前模块的日志


class SchedulerType(Enum):  # 定义调度器类型的枚举类
    LINEAR = "linear"  # 线性调度类型
    COSINE = "cosine"  # 余弦调度类型
    COSINE_WITH_RESTARTS = "cosine_with_restarts"  # 余弦调度类型（带重启）
    POLYNOMIAL = "polynomial"  # 多项式调度类型
    CONSTANT = "constant"  # 常量调度类型
    CONSTANT_WITH_WARMUP = "constant_with_warmup"  # 带热身的常量调度类型
    PIECEWISE_CONSTANT = "piecewise_constant"  # 分段常量调度类型


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:  # 定义获取常量学习率调度的函数
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.  # 创建一个常量学习率调度，使用优化器中设置的学习率

    Args:  # 参数说明
        optimizer ([`~torch.optim.Optimizer`]):  # 优化器参数类型
            The optimizer for which to schedule the learning rate.  # 用于调度学习率的优化器
        last_epoch (`int`, *optional*, defaults to -1):  # 最后一个 epoch 的索引，默认值为 -1
            The index of the last epoch when resuming training.  # 继续训练时的最后一个 epoch 索引

    Return:  # 返回值说明
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.  # 返回相应的 LambdaLR 调度
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)  # 返回一个常量学习率调度器，学习率始终为 1


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1) -> LambdaLR:  # 定义获取带热身的常量学习率调度的函数
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate  # 创建一个常量学习率调度，前面有热身期，在此期间学习率
    increases linearly between 0 and the initial lr set in the optimizer.  # 从 0 线性增加到优化器中设置的初始学习率

    Args:  # 参数说明
        optimizer ([`~torch.optim.Optimizer`]):  # 优化器参数类型
            The optimizer for which to schedule the learning rate.  # 用于调度学习率的优化器
        num_warmup_steps (`int`):  # 热身步骤的数量
            The number of steps for the warmup phase.  # 热身阶段的步骤数
        last_epoch (`int`, *optional*, defaults to -1):  # 最后一个 epoch 的索引，默认值为 -1
            The index of the last epoch when resuming training.  # 继续训练时的最后一个 epoch 索引

    Return:  # 返回值说明
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.  # 返回相应的 LambdaLR 调度
    """

    def lr_lambda(current_step: int):  # 定义学习率调度的 lambda 函数
        if current_step < num_warmup_steps:  # 如果当前步骤小于热身步骤数
            return float(current_step) / float(max(1.0, num_warmup_steps))  # 返回当前步骤与热身步骤数的比值
        return 1.0  # 否则返回 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)  # 返回带热身的常量学习率调度器


def get_piecewise_constant_schedule(optimizer: Optimizer, step_rules: str, last_epoch: int = -1) -> LambdaLR:  # 定义获取分段常量学习率调度的函数
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.  # 创建一个常量学习率调度，使用优化器中设置的学习率
    # 参数说明
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            # 用于调度学习率的优化器
            The optimizer for which to schedule the learning rate.
        step_rules (`string`):
            # 学习率调整的规则，例如：rule_steps="1:10,0.1:20,0.01:30,0.005"，意味着学习率
            # 在前10步乘以1，接下来20步乘以0.1，再接下来的30步乘以0.01，最后的步骤乘以0.005。
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, multiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.
        last_epoch (`int`, *optional*, defaults to -1):
            # 用于恢复训练时的最后一个epoch索引
            The index of the last epoch when resuming training.

    Return:
        # 返回一个适当调度的 `torch.optim.lr_scheduler.LambdaLR`
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    # 创建一个空字典以存储学习率规则
    rules_dict = {}
    # 根据逗号分隔符将规则字符串分割成列表
    rule_list = step_rules.split(",")
    # 遍历除最后一个规则以外的所有规则字符串
    for rule_str in rule_list[:-1]:
        # 根据冒号分隔符将规则字符串分割成值和步数
        value_str, steps_str = rule_str.split(":")
        # 将步数转换为整数
        steps = int(steps_str)
        # 将值转换为浮点数
        value = float(value_str)
        # 将步数和对应值存入字典
        rules_dict[steps] = value
    # 获取最后一个学习率乘数
    last_lr_multiple = float(rule_list[-1])

    # 创建一个生成学习率的函数
    def create_rules_function(rules_dict, last_lr_multiple):
        # 定义学习率规则函数
        def rule_func(steps: int) -> float:
            # 获取已排序的步数
            sorted_steps = sorted(rules_dict.keys())
            # 遍历已排序的步数
            for i, sorted_step in enumerate(sorted_steps):
                # 如果当前步数小于当前规则的步数，则返回对应的学习率值
                if steps < sorted_step:
                    return rules_dict[sorted_steps[i]]
            # 如果步数超出所有规则，返回最后一个学习率乘数
            return last_lr_multiple

        # 返回生成的规则函数
        return rule_func

    # 调用函数生成学习率规则函数
    rules_func = create_rules_function(rules_dict, last_lr_multiple)

    # 返回配置了调度规则的LambdaLR对象
    return LambdaLR(optimizer, rules_func, last_epoch=last_epoch)
# 创建一个带有预热阶段的学习率调度器，该调度器的学习率会线性减少到0
def get_linear_schedule_with_warmup(
    # 优化器对象，用于设置学习率
    optimizer: Optimizer, 
    # 预热阶段的步数
    num_warmup_steps: int, 
    # 总的训练步数
    num_training_steps: int, 
    # 最近一次训练的轮次，默认为-1
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建一个学习率调度器，学习率在预热期内线性增加至初始学习率，然后线性减少至0。

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            要为其调度学习率的优化器。
        num_warmup_steps (`int`):
            预热阶段的步数。
        num_training_steps (`int`):
            训练的总步数。
        last_epoch (`int`, *可选*, 默认值为 -1):
            恢复训练时最后一轮的索引。

    Return:
        `torch.optim.lr_scheduler.LambdaLR`，具有适当的调度。
    """

    # 定义学习率的计算函数
    def lr_lambda(current_step: int):
        # 如果当前步数小于预热步数
        if current_step < num_warmup_steps:
            # 返回当前步数与预热步数的比值
            return float(current_step) / float(max(1, num_warmup_steps))
        # 返回剩余步数与总步数的比值，确保不小于0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    # 返回一个基于优化器和学习率计算函数的LambdaLR调度器
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 创建一个带有预热阶段的余弦学习率调度器
def get_cosine_schedule_with_warmup(
    # 优化器对象，用于设置学习率
    optimizer: Optimizer, 
    # 预热阶段的步数
    num_warmup_steps: int, 
    # 总的训练步数
    num_training_steps: int, 
    # 余弦函数的周期数，默认为0.5
    num_cycles: float = 0.5, 
    # 最近一次训练的轮次，默认为-1
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建一个学习率调度器，学习率在预热阶段线性增加至初始学习率，然后根据余弦函数递减至0。

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            要为其调度学习率的优化器。
        num_warmup_steps (`int`):
            预热阶段的步数。
        num_training_steps (`int`):
            训练的总步数。
        num_cycles (`float`, *可选*, 默认值为0.5):
            调度中余弦函数的周期数（默认情况下仅从最大值减少到0，遵循半余弦）。
        last_epoch (`int`, *可选*, 默认值为-1):
            恢复训练时最后一轮的索引。

    Return:
        `torch.optim.lr_scheduler.LambdaLR`，具有适当的调度。
    """

    # 定义学习率的计算函数
    def lr_lambda(current_step):
        # 如果当前步数小于预热步数
        if current_step < num_warmup_steps:
            # 返回当前步数与预热步数的比值
            return float(current_step) / float(max(1, num_warmup_steps))
        # 计算经过预热阶段后的进度
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 根据余弦函数计算学习率，确保不小于0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    # 返回一个基于优化器和学习率计算函数的LambdaLR调度器
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 创建一个带有预热阶段和硬重启的余弦学习率调度器
def get_cosine_with_hard_restarts_schedule_with_warmup(
    # 定义一个函数参数，包括优化器、预热步数、训练步数、周期数和最后一个周期的索引
        optimizer: Optimizer,  # 优化器对象，负责更新模型参数
        num_warmup_steps: int,  # 预热步数，指定在训练初期的步数
        num_training_steps: int,  # 总训练步数，指定整个训练过程中的步数
        num_cycles: int = 1,  # 训练周期数，默认为1周期
        last_epoch: int = -1  # 最后一个训练周期的索引，默认为-1表示没有上一个周期
# 返回一个 LambdaLR 的学习率调度器
) -> LambdaLR:
    """
    创建一个学习率调度器，学习率在初始值与 0 之间按余弦函数递减，并有多个硬重启，
    在热身阶段，学习率从 0 线性增加到优化器设定的初始学习率。

    参数：
        optimizer ([`~torch.optim.Optimizer`]):
            需要调度学习率的优化器。
        num_warmup_steps (`int`):
            热身阶段的步数。
        num_training_steps (`int`):
            总的训练步数。
        num_cycles (`int`, *可选*, 默认值为 1):
            使用的硬重启次数。
        last_epoch (`int`, *可选*, 默认值为 -1):
            恢复训练时的最后一个周期索引。

    返回：
        `torch.optim.lr_scheduler.LambdaLR`，具有适当的调度。
    """

    # 定义一个学习率更新函数
    def lr_lambda(current_step):
        # 如果当前步数小于热身步数，则线性增加学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 计算热身后进度
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 如果进度达到或超过 1.0，则学习率为 0
        if progress >= 1.0:
            return 0.0
        # 按余弦函数计算学习率
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    # 返回基于定义的 lr_lambda 函数的 LambdaLR 调度器
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 返回一个带有多项式衰减和热身的学习率调度器
def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    创建一个学习率调度器，学习率从优化器设定的初始值按多项式衰减到由 *lr_end* 定义的结束学习率，
    在热身阶段，学习率从 0 线性增加到优化器设定的初始学习率。

    参数：
        optimizer ([`~torch.optim.Optimizer`]):
            需要调度学习率的优化器。
        num_warmup_steps (`int`):
            热身阶段的步数。
        num_training_steps (`int`):
            总的训练步数。
        lr_end (`float`, *可选*, 默认值为 1e-7):
            结束学习率。
        power (`float`, *可选*, 默认值为 1.0):
            幂因子。
        last_epoch (`int`, *可选*, 默认值为 -1):
            恢复训练时的最后一个周期索引。

    注意：*power* 默认值为 1.0，基于 fairseq 实现，进而基于原始 BERT 实现。
    返回：
        `torch.optim.lr_scheduler.LambdaLR`，具有适当的调度。
    """

    # 获取优化器的初始学习率
    lr_init = optimizer.defaults["lr"]
    # 检查初始学习率是否大于结束学习率
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")
    # 定义一个学习率调度函数，根据当前训练步骤调整学习率
        def lr_lambda(current_step: int):
            # 如果当前步骤在预热步骤内，返回预热比例
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            # 如果当前步骤超过训练总步骤，返回结束学习率与初始学习率的比值
            elif current_step > num_training_steps:
                return lr_end / lr_init  # 因为 LambdaLR 会将结果乘以 lr_init
            else:
                # 计算初始学习率与结束学习率的差值
                lr_range = lr_init - lr_end
                # 计算衰减步骤
                decay_steps = num_training_steps - num_warmup_steps
                # 计算剩余的训练比例
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                # 计算衰减后的学习率
                decay = lr_range * pct_remaining**power + lr_end
                # 返回衰减后的学习率与初始学习率的比值
                return decay / lr_init  # 因为 LambdaLR 会将结果乘以 lr_init
    
        # 返回使用定义的学习率调度函数的 LambdaLR 对象
        return LambdaLR(optimizer, lr_lambda, last_epoch)
# 定义调度器类型与对应的调度函数的映射关系
TYPE_TO_SCHEDULER_FUNCTION = {
    # 线性调度器
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    # 余弦调度器
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    # 带硬重启的余弦调度器
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    # 多项式衰减调度器
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    # 常量调度器
    SchedulerType.CONSTANT: get_constant_schedule,
    # 带热身的常量调度器
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    # 分段常量调度器
    SchedulerType.PIECEWISE_CONSTANT: get_piecewise_constant_schedule,
}

# 获取调度器的统一接口
def get_scheduler(
    # 调度器名称，可以是字符串或 SchedulerType
    name: Union[str, SchedulerType],
    # 使用的优化器
    optimizer: Optimizer,
    # 步骤规则，仅 PIECEWISE_CONSTANT 调度器需要
    step_rules: Optional[str] = None,
    # 热身步骤数，可选参数
    num_warmup_steps: Optional[int] = None,
    # 训练步骤数，可选参数
    num_training_steps: Optional[int] = None,
    # 硬重启的周期数，默认值为 1
    num_cycles: int = 1,
    # 多项式调度器的幂因子，默认值为 1.0
    power: float = 1.0,
    # 恢复训练时的最后一个 epoch 索引，默认值为 -1
    last_epoch: int = -1,
) -> LambdaLR:
    """
    从调度器名称获取相应的调度器。

    Args:
        name (`str` or `SchedulerType`):
            要使用的调度器名称。
        optimizer (`torch.optim.Optimizer`):
            训练中使用的优化器。
        step_rules (`str`, *optional*):
            表示步骤规则的字符串，仅 PIECEWISE_CONSTANT 调度器使用。
        num_warmup_steps (`int`, *optional*):
            热身步骤数。并非所有调度器都需要此参数（因此为可选）。
        num_training_steps (`int`, *optional*):
            训练步骤数。并非所有调度器都需要此参数（因此为可选）。
        num_cycles (`int`, *optional*):
            用于 COSINE_WITH_RESTARTS 调度器的硬重启次数。
        power (`float`, *optional*, defaults to 1.0):
            幂因子。见 POLYNOMIAL 调度器。
        last_epoch (`int`, *optional*, defaults to -1):
            恢复训练时的最后一个 epoch 的索引。
    """
    # 将名称转换为 SchedulerType 类型
    name = SchedulerType(name)
    # 根据名称获取调度函数
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    # 如果是常量调度器，直接返回
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)

    # 如果是分段常量调度器，返回带步骤规则的调度器
    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, step_rules=step_rules, last_epoch=last_epoch)

    # 其他调度器需要提供 num_warmup_steps
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # 如果是带热身的常量调度器，返回带热身步骤的调度器
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)

    # 其他调度器需要提供 num_training_steps
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
    # 检查调度器类型是否为 COSINE_WITH_RESTARTS
        if name == SchedulerType.COSINE_WITH_RESTARTS:
            # 调用调度函数并传入相关参数
            return schedule_func(
                optimizer,
                # 预热步数
                num_warmup_steps=num_warmup_steps,
                # 训练总步数
                num_training_steps=num_training_steps,
                # 周期数
                num_cycles=num_cycles,
                # 最后一个训练轮次
                last_epoch=last_epoch,
            )
    
        # 检查调度器类型是否为 POLYNOMIAL
        if name == SchedulerType.POLYNOMIAL:
            # 调用调度函数并传入相关参数
            return schedule_func(
                optimizer,
                # 预热步数
                num_warmup_steps=num_warmup_steps,
                # 训练总步数
                num_training_steps=num_training_steps,
                # 多项式的幂次
                power=power,
                # 最后一个训练轮次
                last_epoch=last_epoch,
            )
    
        # 调用调度函数作为默认情况，传入相关参数
        return schedule_func(
            optimizer, 
            # 预热步数
            num_warmup_steps=num_warmup_steps, 
            # 训练总步数
            num_training_steps=num_training_steps, 
            # 最后一个训练轮次
            last_epoch=last_epoch
        )
```