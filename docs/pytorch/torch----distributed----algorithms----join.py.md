# `.\pytorch\torch\distributed\algorithms\join.py`

```
# mypy: allow-untyped-defs
# 导入警告模块，用于显示警告信息
import warnings
# 导入抽象基类模块，用于定义抽象基类
from abc import ABC, abstractmethod
# 导入异常追踪类型模块
from types import TracebackType
# 导入类型提示模块，用于类型标注
from typing import Any, List, NamedTuple, Optional, Type

# 导入 PyTorch 库
import torch
# 导入分布式通信模块
import torch.distributed as dist

# 定义公开的模块成员
__all__ = ["JoinHook", "Joinable", "Join"]


class JoinHook:
    r"""
    This defines a join hook, which provides two entry points in the join context manager.

    Entry points : a main hook, which is called repeatedly while there exists a non-joined
    process, and a post-hook, which is called once all processes have joined.

    To implement a join hook for the generic join context manager, define a
    class that inherits from :class:`JoinHook` and override ``main_hook()`` and
    ``post_hook()`` as appropriate.
    """

    # 定义主要钩子函数
    def main_hook(self) -> None:
        r"""Call this hook while there exists a non-joined process to shadow collective communications in a training iteration.

        Training iteration i.e., in one forward pass, backward pass, and optimizer step.
        """
        ...

    # 定义后置钩子函数
    def post_hook(self, is_last_joiner: bool) -> None:
        r"""
        Call hook after all processes have joined.

        It is passed an additional ``bool`` argument ``is_last_joiner``, which indicates if the rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """
        ...


class Joinable(ABC):
    r"""
    This defines an abstract base class for joinable classes.

    A joinable class
    (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
    which returns a :class:`JoinHook` instance, in addition to
    :meth:`join_device` and :meth:`join_process_group` that return device and
    process group information, respectively.
    """

    # 抽象方法，初始化方法
    @abstractmethod
    def __init__(self):
        super().__init__()
        # 初始化加入配置为禁用状态的配置
        self._join_config = _JoinConfig.construct_disabled_join_config()

    # 抽象方法，返回一个 JoinHook 实例
    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Return a :class:`JoinHook` instance for the given :class:`Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        ...

    # 抽象属性，返回用于集体通信的设备
    @property
    @abstractmethod
    def join_device(self) -> torch.device:
        r"""Return the device from which to perform collective communications needed by the join context manager."""
        ...

    # 抽象属性，返回用于集体通信的进程组
    @property
    @abstractmethod
    def join_process_group(self) -> Any:
        r"""Returns the process group for the collective communications needed by the join context manager itself."""
        ...


class _JoinConfig(NamedTuple):
    ...
    r"""This includes all fields needed from a :class:`Joinable` instance for the join context manager side."""
    # 定义一个文档字符串，描述这些字段是从 Joinable 类实例中获取的，用于 join 上下文管理器的一侧。

    enable: bool
    # 表示是否启用 join 相关逻辑的标志。

    throw_on_early_termination: bool
    # 表示在早期终止时是否抛出异常的标志。

    is_first_joinable: bool
    # 表示当前对象是否是第一个可以加入的对象的标志。

    @staticmethod
    def construct_disabled_join_config():
        r"""Return a :class:`_JoinConfig` instance indicating that join-related logic should be disabled.

        e.g. if the caller is not in a join context manager.
        """
        # 静态方法，返回一个 _JoinConfig 类实例，指示禁用与 join 相关的逻辑。
        # 例如，如果调用者不在一个 join 上下文管理器中。

        return _JoinConfig(
            enable=False, throw_on_early_termination=False, is_first_joinable=False
        )
# 定义一个上下文管理器类 Join，用于在进程加入后调用自定义钩子。

r"""
This class defines the generic join context manager, which allows custom hooks to be called after a process joins.

These hooks should shadow the
collective communications of non-joined processes to prevent hanging and
erroring and to ensure algorithmic correctness. Refer to :class:`JoinHook`
for details about the hook definition.

.. warning::
    The context manager requires each participating :class:`Joinable` to
    call the method :meth:`notify_join_context()` before its own per-
    iteration collective communications to ensure correctness.

.. warning::
    The context manager requires that all ``process_group`` attributes in
    the :class:`JoinHook` objects are the same. If there are multiple
    :class:`JoinHook` objects, then the ``device`` of the first is used.
    The process group and device information is used for checking for non-
    joined processes and for notifying processes to throw an exception if
    ``throw_on_early_termination`` is enabled, both of which using an all-
    reduce.

Arguments:
    joinables (List[Joinable]): a list of the participating
        :class:`Joinable` s; their hooks are iterated over in the given
        order.

    enable (bool): a flag enabling uneven input detection; setting to
        ``False`` disables the context manager's functionality and should
        only be set when the user knows the inputs will not be uneven
        (default: ``True``).

    throw_on_early_termination (bool): a flag controlling whether to throw an
        exception upon detecting uneven inputs (default: ``False``).

Example::

    >>> import os
    >>> import torch
    >>> import torch.distributed as dist
    >>> import torch.multiprocessing as mp
    >>> # xdoctest: +SKIP
    >>> import torch.nn.parallel.DistributedDataParallel as DDP
    >>> import torch.distributed.optim.ZeroRedundancyOptimizer as ZeRO
    >>> from torch.distributed.algorithms.join import Join
    >>>
    >>> # On each spawned worker
    >>> def worker(rank):
    >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
    >>>     model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    >>>     optim = ZeRO(model.parameters(), torch.optim.Adam, lr=0.01)
    >>>     # Rank 1 gets one more input than rank 0
    >>>     inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)]
    >>>     with Join([model, optim]):
    >>>         for input in inputs:
    >>>             loss = model(input).sum()
    >>>             loss.backward()
    >>>             optim.step()
    >>>     # All ranks reach here without hanging/erroring
"""

def __init__(
    self,
    joinables: List[Joinable],
    enable: bool = True,
    throw_on_early_termination: bool = False,
    **kwargs,
    ):
        # 如果没有任何可加入的对象，抛出数值错误异常
        if len(joinables) == 0:
            raise ValueError("The join context manager requires at least one joinable")
        
        # 将传入的可加入对象列表存储在实例属性中
        self._joinables = joinables
        
        # 根据每个可加入对象的定义，调用其加入钩子函数，生成钩子函数列表
        self._join_hooks = [
            joinable.join_hook(**kwargs) for joinable in self._joinables
        ]
        
        # 存储是否启用加入管理器的标志
        self._enable = enable
        
        # 存储在提前终止时是否抛出异常的标志
        self._throw_on_early_termination = throw_on_early_termination
        
        # 调用内部方法，为每个可加入对象设置加入配置
        self._set_joinable_configs()
        
        # 调用内部方法，从可加入对象中提取分布信息
        self._extract_dist_info()

    def _set_joinable_configs(self) -> None:
        r"""Set the :class:`_JoinConfig` of each participating :class:`Joinable`."""
        # 断言确保可加入对象列表非空
        assert len(self._joinables) > 0
        is_first_joinable = True
        
        # 遍历每个可加入对象，设置其加入配置
        for joinable in self._joinables:
            joinable._join_config = _JoinConfig(
                enable=self._enable,
                throw_on_early_termination=self._throw_on_early_termination,
                is_first_joinable=is_first_joinable,
            )
            is_first_joinable = False

    def _extract_dist_info(self) -> None:
        r"""
        Extract the process group and device information from the joinables.

        If there are multiple joinables, then the context manager uses the
        first specified device.

        Preconditions:
            ``self._joinables`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``Joinable`` objects.
        """
        process_group = None
        device = None
        
        # 遍历每个可加入对象，提取其进程组和设备信息
        for joinable in self._joinables:
            if process_group is None:
                process_group = joinable.join_process_group
            elif process_group != joinable.join_process_group:
                # 如果存在多个不一致的进程组属性，抛出数值错误异常
                raise ValueError(
                    "Using join context manager with multiple process groups"
                )
            if device is None:
                device = joinable.join_device
        
        # 存储提取到的进程组、设备和设备对应的排名
        self._process_group = process_group
        self._rank = dist.get_rank(self._process_group)
        self._device = device

    def __enter__(self):
        # 实现进入上下文时的行为，暂未定义
        ...

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        r"""
        Repeatedly runs the main hooks until all processes join; then, runs the post-hooks.

        Raises:
            RuntimeError
                If ``throw_on_early_termination=True``.
        """
        # 如果禁用了或者是异常类型，则直接传播异常
        if not self._enable or type:
            return  # 如果有异常被抛出，则直接传播异常

        # 初始设定所有进程都未加入
        all_procs_joined = False
        # 假设当前进程是最后一个加入的进程
        is_last_joiner = True

        i = 0
        WARN_THRESHOLD = 1000
        warnings.simplefilter("once")

        # 当还有进程未加入时循环执行主钩子
        while not all_procs_joined:
            if i > WARN_THRESHOLD:
                # 发出警告，指出输入不均匀的情况
                warnings.warn(
                    "Detected uneven input skew of greater than "
                    f"{WARN_THRESHOLD}. This means that rank "
                    f"{self._rank} has at least {WARN_THRESHOLD} "
                    f"fewer inputs than other currently-active ranks. "
                    "This level of skew could lead to performance "
                    "degradation during training."
                )
            # 在未加入的进程中阴影全归约
            num_nonjoined_procs = self._get_num_nonjoined_procs()
            if num_nonjoined_procs == 0:
                all_procs_joined = True
            else:
                if self._throw_on_early_termination:
                    # 通知进程终止
                    self._notify_procs_to_terminate()

                # 执行主钩子
                for join_hook in self._join_hooks:
                    join_hook.main_hook()

                is_last_joiner = False
                i += 1

        # 执行后置钩子
        for join_hook in self._join_hooks:
            join_hook.post_hook(is_last_joiner)

    def _get_num_nonjoined_procs(self):
        r"""Return the number of non-joined processes by shadowing an all-reduce in the non-joined processes."""
        # 返回未加入进程的数量，通过在未加入进程中阴影全归约
        num_nonjoined_procs = torch.zeros(1, device=self._device)
        dist.all_reduce(num_nonjoined_procs, group=self._process_group)
        return num_nonjoined_procs.item()

    def _notify_procs_to_terminate(self):
        r"""Schedule an all-reduce to notify non-joined processes to terminate.

        Also raise a ``RuntimeError`` indicating that the current process has exhausted its inputs.
        """
        # 发起一个全归约来通知未加入的进程终止
        ones = torch.ones(1, device=self._device)
        dist.all_reduce(ones, group=self._process_group)
        raise RuntimeError(f"Rank {self._rank} exhausted all inputs.")

    @staticmethod
    def notify_join_context(joinable: Joinable):
        r"""
        Notifies the join context manager that the calling process has not yet joined.

        Then, if ``throw_on_early_termination=True``, checks if uneven inputs have been detected
        (i.e. if one process has already joined) and throws an exception if so.

        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
        # Ensure that the joinable object has an _join_config attribute
        assert hasattr(joinable, "_join_config"), (
            f"Check that the {type(joinable)} constructor calls the "
            "``Joinable`` constructor"
        )

        # Retrieve the join configuration from the joinable object
        join_config = joinable._join_config

        # First joinable is responsible for the collective communications
        # If this joinable is not the first one or if join is disabled, return None
        if not join_config.is_first_joinable or not join_config.enable:
            return None

        # Retrieve device and process group from the joinable object
        device = joinable.join_device
        process_group = joinable.join_process_group

        # Schedule an all-reduce operation to indicate that the caller has not yet joined
        ones = torch.ones(1, device=device)
        work = dist.all_reduce(ones, group=process_group, async_op=True)

        # If throw_on_early_termination is enabled, check for uneven inputs
        if join_config.throw_on_early_termination:
            # Create a tensor of zeros on the same device
            zeros = torch.zeros(1, device=device)
            # Perform an all-reduce operation across the process group
            dist.all_reduce(zeros, group=process_group)
            # Check if any rank has exhausted inputs
            should_throw = zeros.item()
            # If uneven inputs detected, raise a RuntimeError
            if should_throw:
                raise RuntimeError(
                    "Detected at least one rank that exhausted inputs. "
                    "Throwing across all ranks."
                )

        # Return the async work handle for the all-reduce operation
        return work
```