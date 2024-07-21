# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\post_localSGD_hook.py`

```
# mypy: allow-untyped-defs
# 引入日志模块
import logging

# 引入PyTorch相关模块
import torch
import torch.distributed as dist

# 从当前目录引入默认钩子
from . import default_hooks as default

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class PostLocalSGDState:
    r"""
    Store state for all-reducing gradients globally until given step, then locally after.

    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """

    # 定义类的__slots__属性，限制实例的动态属性
    __slots__ = [
        "process_group",
        "subgroup",
        "start_localSGD_iter",
        "post_local_gradient_allreduce",
        "iter",
    ]

    def __init__(
        self,
        process_group,
        subgroup,
        start_localSGD_iter,
        post_local_gradient_allreduce=True,
    ):
        """Initialize state object with given parameters and log when localSGD start."""
        # 记录日志，显示在何时开始使用本地SGD
        logger.info(
            "Local SGD will be started after %s iterations", start_localSGD_iter
        )

        # 用于全局梯度全局归约的进程组
        self.process_group = process_group
        # 用于本地梯度全局归约的子组
        self.subgroup = subgroup
        # 开始使用本地SGD的迭代步数
        self.start_localSGD_iter = start_localSGD_iter
        # 控制是否从`start_localSGD_iter`步开始本地梯度全局归约的标志
        self.post_local_gradient_allreduce = post_local_gradient_allreduce
        # 训练循环中的当前迭代步数
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # 当bucket是最后一个要进行全局归约的桶时，增加`iter`计数
        if bucket.is_last():
            self.iter += 1

        # 如果`iter`等于`start_localSGD_iter`，记录日志开始应用本地SGD
        if self.iter == self.start_localSGD_iter:
            logger.info("Start to apply local SGD after %s iterations.", self.iter)


def post_localSGD_hook(
    state: PostLocalSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Run post-localSGD algorithm.

    This DDP communication hook is used for running post-localSGD algorithm,
    by combining with a model averaging component (e.g.,
    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)
    that runs after the optimizer step.
    """
    # 执行post-localSGD算法的钩子函数
    Args:
        state (PostLocalSGDState): 运行 post-localSGD 所需的状态信息。
            用户主要需要调整 `start_localSGD_iter` 来确定何时开始本地SGD。
        bucket (dist.GradBucket): 存储了一个扁平化的1D梯度张量的桶。
            注意，由于DDP通信钩子仅支持单进程单设备模式，
            此桶中仅存储一个张量。

    Returns:
        Future handler of the communication, which updates the gradients in place.
        通信的未来处理程序，用于就地更新梯度。

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,
                                  start_localSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)
        >>> # 还需要建立一个模型平均模块，并在 `optimizer.step()` 后运行模型平均。
        >>> # 请参阅 `torch.distributed.algorithms.model_averaging.averagers` 模块中的示例。
    """
    global_group_to_use = (
        state.process_group if state.process_group is not None else dist.group.WORLD
    )

    # The input tensor is a flattened 1D tensor.
    # 输入张量是一个扁平化的1D张量。
    input_tensor = bucket.buffer()

    # Run allreduce using `global_group_to_use` in the first `start_localSGD_iter` iterations.
    # 在前 `start_localSGD_iter` 次迭代中使用 `global_group_to_use` 运行 allreduce。
    if state.iter < state.start_localSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(global_group_to_use, input_tensor)

    # If `post_local_gradient_allreduce` is not set,
    # then no gradient synchronization after the first `start_localSGD_iter` iterations.
    # 如果未设置 `post_local_gradient_allreduce`，
    # 则在第 `start_localSGD_iter` 次迭代后不进行梯度同步。
    if not state.post_local_gradient_allreduce:
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Run allreduce using `subgroup` after the first `start_localSGD_iter` iterations.
    # 在第 `start_localSGD_iter` 次迭代后使用 `subgroup` 运行 allreduce。
    # 注意，默认情况下，为每个节点创建一个单独的子组，
    # 这会导致在每个训练步骤中执行节点内部的 allreduce。
    # 从这一刻开始，模型平均应在优化器步骤之后运行，
    # 以全局 allreduce 所有参数。
    if state.subgroup is None:
        state.subgroup, _ = dist.new_subgroups()
    return default._allreduce_fut(state.subgroup, input_tensor)
```