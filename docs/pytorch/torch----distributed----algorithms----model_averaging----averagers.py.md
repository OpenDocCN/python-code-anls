# `.\pytorch\torch\distributed\algorithms\model_averaging\averagers.py`

```py
# mypy: allow-untyped-defs
# 导入警告模块，用于显示警告信息
import warnings
# 导入抽象基类（ABC）和抽象方法装饰器
from abc import ABC, abstractmethod
# 导入类型提示模块中的字典（Dict）、可迭代对象（Iterable）和联合类型（Union）
from typing import Dict, Iterable, Union

# 导入PyTorch库
import torch
# 导入PyTorch分布式通信模块
import torch.distributed as dist
# 导入PyTorch分布式算法模型平均化工具模块
import torch.distributed.algorithms.model_averaging.utils as utils

# 声明模块中公开的类列表
__all__ = ["ModelAverager", "PeriodicModelAverager"]

# 定义抽象基类，用于实现模型平均化
class ModelAverager(ABC):
    r"""Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """
    
    # 构造函数，初始化模型平均化器对象
    def __init__(self, process_group=None):
        # 设置使用的进程组，如果未提供则使用全局默认进程组
        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        # 初始化步骤计数器为0
        self.step = 0

    # 定义抽象方法，用于平均化模型参数
    @abstractmethod
    def average_parameters(self, params):
        # 抽象方法，需要在子类中具体实现
        raise NotImplementedError

# 定义周期性模型平均化器类，继承自模型平均化器基类
class PeriodicModelAverager(ModelAverager):
    r"""
    Averages parameters periodically after the warm-up stage.

    This can be used for running `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    by running :class:`~torch.nn.DistributedDataParallel` (DDP)
    using the subgroups created by :meth:`~torch.distributed.new_subgroups`.

    Args:
        period (int): The number of steps per model averaging.
                      Usually the period should be greater than ``1`` to reduce the communication cost.
                      Otherwise, only DDP needs to be used.
        warmup_steps (int): The number of warm-up steps. During this stage,
                            model averaging is skipped.
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """
    
    # 构造函数，初始化周期性模型平均化器对象
    def __init__(self, period, warmup_steps, process_group=None):
        # 调用父类构造函数初始化进程组
        super().__init__(process_group)
        # 设置模型平均化周期
        self.period = period
        # 设置模型热身阶段的步数
        self.warmup_steps = warmup_steps
    def __init__(self, period, warmup_steps=0, process_group=None):
        # 调用父类构造函数并初始化属性
        super().__init__(process_group)
        # 检查并设置 warmup_steps，确保其为非负数
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps
        # 检查并设置 period，确保其为正数；当 period 为 1 时，发出警告
        if period < 1:
            raise ValueError("Arg ``period`` must be a positive value.")
        elif period == 1:
            warnings.warn(
                "When period is 1, no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case."
            )
        self.period = period

    def average_parameters(
        self,
        params: Union[
            Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]
        ],
        # 该方法用于执行模型参数的周期性平均


这段代码定义了一个类 `__init__` 方法和 `average_parameters` 方法。`__init__` 方法用于初始化对象的属性，包括 `period`（周期）、`warmup_steps`（预热步数）和 `process_group`（进程组）。在 `average_parameters` 方法中，它接受模型参数作为输入，并执行周期性的参数平均化操作。
    ):
        """
        如果 ``step`` 不小于 ``warmup_steps``，则对优化器的参数或参数组进行平均。

        可以通过 ``period`` 进行分组，其中 ``step`` 在训练循环中的每次迭代增加1。
        Args:
            params: 模型的参数或优化器的参数组。

        """
        # 检查是否达到平均参数的条件：step 大于等于 warmup_steps，并且 (step - warmup_steps) 可被 period 整除
        if (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        ):
            # 调用工具函数，对参数或参数组进行平均化处理
            utils.average_parameters_or_parameter_groups(params, self.process_group)
        # 增加 step 计数
        self.step += 1
```