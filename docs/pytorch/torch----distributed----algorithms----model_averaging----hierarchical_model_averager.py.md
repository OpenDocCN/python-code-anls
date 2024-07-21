# `.\pytorch\torch\distributed\algorithms\model_averaging\hierarchical_model_averager.py`

```
# 设置 mypy 允许未标注的函数定义
# 版权声明 2022 Cruise LLC
# 导入日志记录模块
import logging
# 导入警告模块
import warnings
# 导入有序字典模块
from collections import OrderedDict
# 导入类型提示模块
from typing import Dict, Iterable, Union

# 导入 PyTorch 深度学习库
import torch
# 导入 PyTorch 分布式训练模块
import torch.distributed as dist
# 导入模型平均化算法的实现模块
import torch.distributed.algorithms.model_averaging.averagers as averagers
# 导入模型平均化算法的工具模块
import torch.distributed.algorithms.model_averaging.utils as utils

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


class HierarchicalModelAverager(averagers.ModelAverager):
    r"""
    运行层次化模型平均化 (`hierarchical SGD <https://arxiv.org/pdf/2010.12998.pdf>`_)。

    不同大小的进程组按层次结构组织，它们在暖身阶段之后并发地使用不同的周期对参数进行平均。
    这是 :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager` 的扩展，
    支持 `post-local SGD <https://arxiv.org/abs/1808.07217>`_，这本质上只支持两级层次结构：
    机器内部层次和全局层次，其中机器内部层次通常嵌入在 :meth:`~torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook` 中。
    同样，该类内部的进程组没有这样的机器内部进程子组，而应该由 post-local SGD 通信钩子来嵌入。
    """
    # 参数说明：
    # period_group_size_dict: 一个有序字典，将模型平均期间的键映射到处理组大小，用于初始化层次结构中不同大小的处理组以并行平均参数。
    #                        在每次迭代中，最多只会有一个处理组执行平均化操作，该组的周期应为当前步骤可以被整除的最大周期。
    #                        例如，如果字典有三个键：2、4 和 8，则意味着总共会创建三个处理组，分别每 2、4 和 8 次迭代执行参数平均化。
    #                        在第 4 次迭代时，只有第二个处理组会执行平均化，因为第一个处理组应该是第二个处理组的子集，无需重复执行第一个处理组。
    #                        另一方面，第三个处理组只能在每 8 次迭代时触发，因此在第 4 次迭代时不会触发它。
    # warmup_steps (int): 热身阶段的步数。在此阶段，跳过模型平均化操作。
    # process_group (ProcessGroup, optional): 包含运行模型平均化的所有进程的整体处理组。
    #                                        如果为 ``None``，将使用默认的处理组，该处理组由 :func:`torch.distributed.init_process_group` 创建。
    #                                        （默认值：``None``）
    >>> # xdoctest: +SKIP('undefined rank')
    >>> from collections import OrderedDict
    >>> import torch
    >>> import torch.distributed as dist
    >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    >>>     PostLocalSGDState,
    >>>     post_localSGD_hook,
    >>> )
    >>> import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
    >>> import torch.nn as nn
    >>>
    >>> # 初始化分布式进程组，使用NCCL后端，设定当前进程的rank和总进程数
    >>> dist.init_process_group("nccl", rank=rank, world_size=16)
    >>> # 设置当前CUDA设备
    >>> torch.cuda.set_device(rank)
    >>> # 创建一个简单的线性模型，没有偏置，放置到指定的rank上
    >>> module = nn.Linear(1, 1, bias=False).to(rank)
    >>> # 使用DistributedDataParallel封装模型，指定设备ID和输出设备
    >>> model = nn.parallel.DistributedDataParallel(
    >>>    module, device_ids=[rank], output_device=rank
    >>> )
    >>> # 注册一个后局部SGD通信钩子
    >>> # 假设每台机器有4个GPU，则每个机器内部的子组大小为4
    >>> subgroup, _ = dist.new_subgroups()
    >>> # 创建后局部SGD状态对象
    >>> state = PostLocalSGDState(process_group=None, subgroup=subgroup, start_localSGD_iter=100)
    >>> # 将状态对象和后局部SGD钩子注册到模型上
    >>> model.register_comm_hook(state, post_localSGD_hook)
    >>>
    >>> # 创建层次模型平均器，设置每组的迭代周期和大小，以及预热步数
    >>> averager = hierarchicalSGD.HierarchicalModelAverager(
    >>>     period_group_size_dict=OrderedDict([(4, 8), (16, 16)]), warmup_steps=100)
    >>> # 注意，warmup_steps必须与PostLocalSGDState中的start_localSGD_iter相同
    >>> # 在前100步中，像正常的DDP一样运行全局梯度平均
    >>> # 在100步后，以两个层次运行模型参数平均
    >>> for step in range(0, 200):
    >>>    optimizer.zero_grad()
    >>>    loss = loss_fn(output, labels)
    >>>    loss.backward()
    >>>    optimizer.step()
    >>>    # 在optimizer.step()之后平均参数
    >>>    # 因此，节点间通信只在warmup_steps之后周期性发生
    >>>    averager.average_parameters(model.parameters())
    def __init__(self, period_group_size_dict=None, warmup_steps=0, process_group=None):
        # 调用父类的构造函数，初始化对象
        super().__init__(process_group)
        
        # 检查是否传入了空的期间-组大小字典，如果是则抛出数值错误异常
        if not period_group_size_dict:
            raise ValueError("Arg ``period_group_size_dict`` must not be empty.")
        
        # 从期间-组大小字典中获取所有期间的列表，并检查最小期间是否大于0，若不是则抛出数值错误异常
        self._periods = list(period_group_size_dict.keys())
        if self._periods[0] <= 0:
            raise ValueError(
                "The minimum period in arg ``period_group_size_dict`` must be a positive value."
            )
        
        # 若最大期间为1，则发出警告，说明模型平均化的使用可能不必要
        elif self._periods[-1] == 1:
            warnings.warn(
                "When the maximum period in arg ``period_group_size_dict`` is 1, "
                "no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case."
            )
        
        # 获取整体组大小
        overall_group_size = dist.get_world_size(group=self.process_group)
        
        # 检查期间-组大小字典中最后一个值是否等于整体组大小，若不等则抛出数值错误异常
        if list(period_group_size_dict.values())[-1] != overall_group_size:
            raise ValueError(
                f"The last value in arg ``period_process_group_dict`` {list(period_group_size_dict.values())[-1]} "
                f"must be equal to the size of arg ``process_group`` {overall_group_size}."
            )

        # 创建有序字典用于存储期间到进程组的映射关系，并记录日志
        self.period_process_group_dict = OrderedDict()
        logger.info("Model averaging hierarchy:")
        
        # 遍历期间-组大小字典，根据不同组大小创建子组或使用整体组，并记录相应日志
        for period, group_size in period_group_size_dict.items():
            logger.info(
                "\tEach group that has %s processes average parameters every %s iterations, "
                "if no higher-level averaging.",
                group_size,
                period,
            )
            if group_size != overall_group_size:
                self.period_process_group_dict[period], _ = dist.new_subgroups(
                    group_size=group_size, group=self.process_group
                )
            else:
                self.period_process_group_dict[period] = self.process_group
        
        # 检查暖身步数是否为非负数，若为负数则抛出数值错误异常
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        
        # 初始化暖身步数
        self.warmup_steps = warmup_steps

    def _find_process_group(self):
        """
        Return a process group as the value of an ``period_process_group_dict`` entry.

        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
        # 反向遍历所有期间，找到能整除当前步数的最大期间对应的进程组，并返回
        for period in reversed(self._periods):
            if self.step % period == 0:
                return self.period_process_group_dict[period]
        
        # 如果找不到对应的进程组，则返回 None
        return None
    `
        # 定义方法 average_parameters，接受参数 params，参数类型为 Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]
        def average_parameters(
            self,
            params: Union[
                Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]
            ],
        ):
            """
            Averages parameters or parameter groups of an optimizer.
    
            Averaging only occurs if ``step`` is no less than ``warmup_steps``
            and it can be divided by a period in the keys of ``period_process_group_dict``,
            where ``step`` is increased by 1 at each iteration in the training loop.
            If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
            only the largest period is used, and the corresponding process group is used for averaging parameters.
            Args:
                params: The parameters of a model or parameter groups of an optimizer.
            """
            # 如果当前 step 大于等于 warmup_steps，则执行平均操作
            if self.step >= self.warmup_steps:
                # 查找用于参数平均的进程组
                group = self._find_process_group()
                # 如果找到进程组，则进行参数平均
                if group is not None:
                    utils.average_parameters_or_parameter_groups(params, group)
            # 每次调用该方法后，step 增加 1
            self.step += 1
```