# `.\pytorch\torch\distributed\optim\post_localSGD_optimizer.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于可能的警告信息
import warnings

# 引入 PyTorch 库
import torch
# 引入模型平均化算法模块
import torch.distributed.algorithms.model_averaging.averagers as averagers

# 自定义优化器类，继承自 torch.optim.Optimizer
class PostLocalSGDOptimizer(torch.optim.Optimizer):
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    This optimizer runs local optimizer at every step.
    After the warm-up stage, it averages parameters periodically afer the local optimizer is applied.

    Args:
        optim: The local optimizer.
        averager: A model averager instance to run post-localSGD algorithm.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import PostLocalSGDOptimizer
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>   PostLocalSGDState,
        >>>   post_localSGD_hook,
        >>> )
        >>>
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>>
        >>> # Register a post-localSGD communication hook.
        >>> state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Create a post-localSGD optimizer that wraps a local optimizer.
        >>> # Note that ``warmup_steps`` used in ``PostLocalSGDOptimizer`` must be the same as
        >>> # ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> local_optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
        >>> opt = PostLocalSGDOptimizer(
        >>>     optim=local_optim,
        >>>     averager=averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> )
        >>>
        >>> # In the first 100 steps, DDP runs global gradient averaging at every step.
        >>> # After 100 steps, DDP runs gradient averaging within each subgroup (intra-node by default),
        >>> # and post-localSGD optimizer runs global model averaging every 4 steps after applying the local optimizer.
        >>> for step in range(0, 200):
        >>>    opt.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    opt.step()
    """

    # 初始化函数，接受一个本地优化器和一个模型平均化器实例作为参数
    def __init__(self, optim: torch.optim.Optimizer, averager: averagers.ModelAverager):
        # 调用父类的初始化函数
        self.optim = optim
        # 获取本地优化器的参数组
        self.param_groups = self.optim.param_groups
        # 设置模型平均化器
        self.averager = averager

    # 属性访问器，返回本地优化器的状态
    @property
    def state(self):
        return self.optim.state

    # 自定义类的字符串表示
    def __repr__(self):
        return self.optim.__repr__()
    # 返回优化器的状态字典，包括模型平均器的步数，以确保加载检查点时不会再次进行不必要的热身
    def state_dict(self):
        """
        This is the same as :class:`torch.optim.Optimizer` :meth:`state_dict`,
        but adds an extra entry to record model averager's step to the checkpoint
        to ensure reload does not cause unnecessary warm up again.
        """
        # 获取优化器的状态字典
        optim_state_dict = self.optim.state_dict()
        # 添加模型平均器的步数到状态字典中
        optim_state_dict["step"] = self.averager.step
        return optim_state_dict

    # 加载给定状态字典，恢复优化器和模型平均器的步数
    def load_state_dict(self, state_dict):
        """
        This is the same as :class:`torch.optim.Optimizer` :meth:`load_state_dict`,
        but also restores model averager's step value to the one
        saved in the provided ``state_dict``.

        If there is no ``"step"`` entry in ``state_dict``,
        it will raise a warning and initialize the model averager's step to 0.
        """
        # 加载优化器的状态字典
        self.optim.load_state_dict(state_dict)
        # 如果状态字典中包含步数信息，则恢复模型平均器的步数
        if "step" in state_dict:
            self.averager.step = state_dict["step"]
        else:
            # 如果状态字典中没有步数信息，则发出警告并将模型平均器的步数初始化为0
            warnings.warn(
                "Loaded state dict does not contain a step counter for an averager. "
                "Setting step counter to 0."
            )
            self.averager.step = 0

    # 执行单个优化步骤（参数更新）
    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        # 执行优化器的优化步骤
        self.optim.step()
        # 对参数进行平均化处理
        self.averager.average_parameters(params=self.param_groups)

    # 将所有参数的梯度清零
    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        # 调用优化器的梯度清零方法
        self.optim.zero_grad(set_to_none=set_to_none)

    # 向优化器添加参数组
    def add_param_group(self, param_group):
        # 调用优化器的添加参数组方法
        self.optim.add_param_group(param_group)
```