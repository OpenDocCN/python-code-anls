# `.\pytorch\torch\distributed\optim\zero_redundancy_optimizer.py`

```
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

r"""Zero Redundancy Optimizer."""
# 导入必要的库和模块
import collections  # 导入collections模块
import copy  # 导入copy模块
import enum  # 导入enum枚举模块
import inspect  # 导入inspect模块
import io  # 导入io模块
import logging  # 导入logging模块
from itertools import chain  # 从itertools模块中导入chain函数
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union  # 导入类型提示相关的内容

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式训练相关模块
from torch.distributed.algorithms.join import Join, Joinable, JoinHook  # 导入PyTorch分布式训练中的Join相关模块
from torch.distributed.optim.utils import functional_optim_map  # 导入PyTorch分布式优化工具中的functional_optim_map函数
from torch.optim import Optimizer  # 从torch.optim中导入Optimizer类


__all__ = ["ZeroRedundancyOptimizer"]  # 定义模块导出的公共接口列表


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if possible.

    Non-tensor values are passed as-is in the result.

    .. note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """
    # 如果value是torch.Tensor类型，则将其复制到指定设备上
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    # 如果value是列表或元组类型，则递归复制其中的每个元素到指定设备上
    if isinstance(value, (list, tuple)):
        values = [
            _recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            for val in value
        ]
        return values if isinstance(value, list) else tuple(values)

    # 如果value是字典类型，则递归复制其中的每个值到指定设备上
    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )
            for key, val in value.items()
        }

    # 对于其他类型的值（例如整数、字符串等），直接返回原始值
    return value


def _is_trainable(param: torch.Tensor) -> bool:
    r"""Return if a parameter is trainable, where trainability is equivalent to requiring a gradient."""
    # 判断参数param是否需要梯度，从而确定其是否可训练
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    device: torch.device = torch.device("cpu"),
) -> Any:
    r"""
    Broadcasts an object to the given group.

    It will be sending the object if called from the source rank and receiving
    the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.group.WORLD``).
        device (``torch.device``, optional): device to send from or receive
            to (default: ``torch.device("cpu")``).

    Returns:
        The broadcasted object.
    """
    # 将对象obj广播到给定的进程组中
    # 如果调用者是源排名(src_rank)，则发送对象；否则接收对象
    return dist.broadcast(obj if dist.get_rank() == src_rank else None, src_rank, group=group, device=device)
    # 检查当前进程在分布式环境中的排名是否与指定发送排名相同
    if dist.get_rank() == src_rank:
        # 如果是发送方，将对象保存到内存缓冲区中
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        # 将缓冲区的数据转换为字节数组
        data = bytearray(buffer.getbuffer())
        # 创建包含数据长度的张量，并移动到指定的设备上
        length_tensor = torch.LongTensor([len(data)]).to(device)
        # 创建包含数据内容的张量，并移动到指定的设备上
        data_send_tensor = torch.ByteTensor(data).to(device)
        # 使用分布式通信将数据长度张量广播给指定的接收排名
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        # 使用分布式通信将数据内容张量广播给指定的接收排名
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # 如果是接收方，创建一个长度为0的张量，表示接收数据长度
        length_tensor = torch.LongTensor([0]).to(device)
        # 使用分布式通信接收广播的数据长度张量
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        # 创建一个空张量，根据接收到的数据长度来存储接收到的数据内容，移动到指定的设备上
        data_recv_tensor = torch.empty(
            [int(length_tensor.item())], dtype=torch.uint8, device=device
        )
        # 使用分布式通信接收广播的数据内容张量
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        # 将接收到的数据内容张量转换为字节流，并加载为对象
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device)
    # 返回接收到或发送的对象
    return obj
class _ZeROJoinHook(JoinHook):
    def __init__(self, zero):
        # 确保传入的 `zero` 参数是 `ZeroRedundancyOptimizer` 的实例
        assert isinstance(zero, ZeroRedundancyOptimizer), (
            "ZeRO join hook requires passing in a ZeroRedundancyOptimizer "
            "instance as the state"
        )
        self.zero = zero  # 将 `zero` 参数保存在实例中
        super().__init__()  # 调用父类的初始化方法

    def main_hook(self):
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()  # 调用 `ZeroRedundancyOptimizer` 实例的 `step` 方法进行优化步骤


class _DDPBucketAssignment:
    r"""
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[torch.Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device (torch.device): device on which the parameters are stored.
        tensor (torch.Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """

    def __init__(
        self,
        bucket_index: int,
        parameters: List[torch.Tensor],
        offset: int,
    ):
        self.bucket_index = bucket_index  # 设置 bucket 的索引
        self.parameters = parameters  # 存储分配给此 bucket 的模型参数列表
        self.offset = offset  # 设置在 `GradBucket` 的 `parameters` 中的偏移量
        if len(self.parameters) == 0:
            raise ValueError("Empty bucket assignment")  # 如果参数列表为空，则抛出异常
        # DDP 保证分配给 bucket 的所有参数都在同一个设备上
        self.device: torch.device = self.parameters[0].device  # 获取参数列表中第一个参数的设备信息
        self.tensor: Optional[torch.Tensor] = None  # 初始化 tensor 属性为 None


class _OverlapStatus(enum.IntEnum):
    r"""
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """

    UNINITIALIZED = 0  # ZeRO 实例未初始化，等待 DDP 完成桶分配
    DDP_HAS_REBUILT_BUCKETS = 1  # DDP 已重新构建桶，桶分配已完成
    INITIALIZED = 2  # ZeRO 实例已完全初始化，可以进行参数优化


class _OverlapInfo:
    r"""
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.
    """
    Attributes:
        shard_buckets (bool):
            # 是否对每个分布式数据并行桶进行分片分配。如果为 True，则每个桶的分配可能跨多个 ZeroRedundancyOptimizer 实例（即可能跨多个进程），以近似均匀性，按照总参数大小除以进程总数的阈值；如果为 False，则每个桶完全分配给单个 ZeroRedundancyOptimizer 实例（即单个进程）；此值应与钩子构造函数中传递的值相同。
        status (_OverlapStatus):
            # 当前的状态；详见 _OverlapStatus 类的更多信息。
        params_per_bucket (List[List[torch.Tensor]]):
            # params_per_bucket[i] 给出第 i 个桶中的模型参数列表。
        params_per_rank (List[List[torch.Tensor]]):
            # params_per_rank[i] 给出分配给第 i 个进程的模型参数列表，其中参数按增加的桶索引分组。
        offsets (Dict[int, int]):
            # 将桶索引映射到 self.params_per_rank[rank] 中给出的第一个参数的偏移量的字典，其中 rank 是此进程自己的排名；此 dict 的键是分配给此进程的桶索引。
        num_bucket_assignments (int):
            # 所有进程中所有桶分配的总数；如果 shard_buckets=False，则这等于 DistributedDataParallel 的梯度桶数；否则可能更大。
        total_size (int, optional):
            # 如果 shard_buckets=True，则所有桶的总大小（即所有桶中所有 param.numel() 的总和）；否则为 None。
        broadcast_handles (List[Work]):
            # 参数广播的异步工作句柄列表。
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            # 将桶索引映射到相应的全局归约 future 的字典。
        bucket_index_to_bucket (Dict[int, dist.GradBucket]):
            # 将桶索引映射到相应的桶对象的字典。
        bucket_indices_seen (List[int]):
            # 在此迭代中看到的桶索引列表。
    def __init__(self, world_size) -> None:
        # 初始化对象的状态为未初始化
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        # 是否使用分片桶
        self.shard_buckets: bool = False

        # 按照桶进行重构后的参数列表
        self.params_per_bucket: List[List[torch.Tensor]] = []
        # 按照每个进程排列的参数列表，初始化为空列表
        self.params_per_rank: List[List[torch.Tensor]] = [[] for _ in range(world_size)]
        # 偏移量字典，用于映射桶的偏移信息
        self.offsets: Dict[int, int] = {}
        # 桶分配的进程组
        self.assigned_ranks_per_bucket: List[Set[int]] = []
        # 桶分配的数量
        self.num_bucket_assignments: int = 0
        # 总大小，可选
        self.total_size: Optional[int] = None

        # 每次迭代修改的数据结构
        self.broadcast_handles: List[Any] = []
        # 已看到的桶索引列表
        self.bucket_indices_seen: List[int] = []
        # 由 `hook_with_zero_step()` 使用的字典，将桶索引映射到 Torch 未来对象
        self.bucket_index_to_future: Dict[int, torch.futures.Future] = {}
        # 由 `hook_with_zero_step()` 使用的字典，将桶索引映射到分布式梯度桶对象
        self.bucket_index_to_bucket: Dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        r"""
        等待所有参数广播完成。

        调用此函数时应当确保所有广播已经被安排，即 `self.broadcast_handles` 已填充。这会清空
        `self.broadcast_handles`，以准备进行下一次迭代。
        """
        # 断言确保每个桶分配都有一个广播句柄
        assert (
            len(self.broadcast_handles) == self.num_bucket_assignments
        ), f"Missing at least one broadcast handle on rank {dist.get_rank()}"
        # 等待所有广播完成
        _ = [x.wait() for x in self.broadcast_handles]
        # 清空广播句柄列表
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        r"""
        清空每次迭代修改的数据结构。

        调用此函数应在迭代结束时。
        """
        # 清空已看到的桶索引列表
        self.bucket_indices_seen.clear()
        # 清空桶索引到未来对象的映射
        self.bucket_index_to_future.clear()
        # 清空桶索引到分布式梯度桶对象的映射
        self.bucket_index_to_bucket.clear()
# 定义一个名为 ZeroRedundancyOptimizer 的类，继承自 Optimizer 和 Joinable
# 用于将任意的 torch.optim.Optimizer 类型对象包装起来，并将其状态在组内的不同进程之间分片存储
class ZeroRedundancyOptimizer(Optimizer, Joinable):
    r"""
    Wrap an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>` and shards its states across ranks in the group.

    The sharing is done as described by ZeRO_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.
    Example::

        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``. To adjust for this, one option
        is to prepend dummy inputs.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    """

    # 定义 ZeroRedundancyOptimizer 类，用于分布式训练优化
    def __init__(
        self,
        params,  # 待优化的参数列表
        optimizer_class: Type[Optimizer],  # 优化器类的类型
        process_group: Optional[Any] = None,  # 进程组对象，可选
        parameters_as_bucket_view: bool = False,  # 参数作为桶视图
        overlap_with_ddp: bool = False,  # 是否与 DistributedDataParallel 重叠
        **defaults: Any,  # 其他参数作为默认设置
    ):
        # 清除缓存的数据结构，包括分区信息的缓存
        def _clear_cache(self) -> None:
            self._partition_parameters_cache.clear()  # 清除分区参数缓存
            self._param_to_rank_cache.clear()  # 清除参数到排名的映射缓存
            self._index_to_param_cache.clear()  # 清除索引到参数的映射缓存
            self._param_to_index_cache.clear()  # 清除参数到索引的映射缓存
            self._device_to_params_per_rank_cache.clear()  # 清除设备到每个排名参数的映射缓存
            self._bucket_assignments_per_rank_cache.clear()  # 清除每个排名的桶分配缓存
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""
        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
        # 如果初始化已经完成且与 DDP 重叠
        if self.initialized and self._overlap_with_ddp:
            # 抛出运行时错误，因为 `overlap_with_ddp=True` 的 ZeroRedundancyOptimizer
            # 仅支持单一的参数组
            raise RuntimeError(
                "ZeroRedundancyOptimizer with `overlap_with_ddp=True` only "
                "supports a single parameter group"
            )

        # 调用父类的 `add_param_group()` 方法，添加新的参数组
        super().add_param_group(param_group)

        # 注意：以下的方法假设调用父类的 `add_param_group()` 方法会将新的参数组追加并保持
        # 先前参数组的顺序不变

        if self.initialized:
            # 强制重新分配参数
            self._clear_cache()
            # 获取分配给当前进程的参数组
            param_groups = self._partition_parameters()[self.rank]

            # 注意：所有旧参数组中的参数应该分配给相同的进程，这样本地优化器不需要重新初始化

            # 如果新参数组的长度等于旧的参数组长度加一，则将新参数组中分配给当前进程的参数
            # 添加到本地优化器中
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # 根据需要更新桶策略
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def _verify_params_per_rank(
        self,
        params_per_rank: List[List[torch.Tensor]],
    ) -> None:
        r"""
        Verify ``params_per_rank`` for :meth:`_partition_parameters`.

        The verification is done by checking that ``params_per_rank`` has length equal
        to the world size and that it does not contain any parameters not passed into the
        :class:`ZeroRedundancyOptimizer` constructor.

        The parameters in ``params_per_rank`` being a strict subset of those
        passed into the constructor is valid since some parameters may be
        frozen.

        Raises:
            ValueError: if ``params_per_rank`` does not have length equal to
                the world size or if it contains a parameter that was not
                passed into the :class:`ZeroRedundancyOptimizer` constructor.
        """
        # Verify the validity of `params_per_rank` for `_partition_parameters` method
        if len(params_per_rank) != self.world_size:
            raise ValueError(
                "`params_per_rank` must have length equal to the world size"
            )
        # Create a set of all parameters passed into the optimizer constructor
        all_params_set = set(self._all_params)
        # Check each set of parameters in `params_per_rank`
        for params in params_per_rank:
            for param in params:
                # Raise error if any parameter in `params_per_rank` is not in `self._all_params`
                if param not in all_params_set:
                    raise ValueError(
                        "Passing a new parameter in `params_per_rank` that "
                        "was not passed into the ZeroRedundancyOptimizer "
                        "constructor"
                    )

    def _partition_param_group(
        self, param_group: Dict[str, Any], params_per_rank: List[List[torch.Tensor]]
    ) -> None:
        r"""
        Partition the parameter group ``param_group`` according to ``params_per_rank``.

        The partition will modify the ``self._partition_parameters_cache``. This method should
        only be used as a subroutine for :meth:`_partition_parameters`.

        Arguments:
            param_group (dict[str, Any]): a parameter group as normally defined
                in an optimizer state.
            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
                length world size containing :class:`list` s of parameters to
                assign to each rank.
        """
        # Iterate over each rank and its corresponding list of parameters
        for rank, params in enumerate(params_per_rank):
            # Create a shallow copy of param_group for the current rank
            rank_param_group = copy.copy(param_group)
            # Assign the current list of parameters to the 'params' key of rank_param_group
            rank_param_group["params"] = params
            # Append the modified param_group to self._partition_parameters_cache for the current rank
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(
        self,
        params_per_rank: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        r"""
        Partition the parameters based on ``params_per_rank``.

        This method partitions the parameters according to ``params_per_rank``,
        where each sublist of `params_per_rank` corresponds to parameters assigned
        to a specific rank.

        Returns:
            List of lists of dictionaries, each containing a parameter group with assigned parameters.
        """
        # Implementation of parameter partitioning based on params_per_rank
        ...

    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        r""":class:`dict` mapping parameters to their assigned data parallel rank in the partition."""
        # If _param_to_rank_cache is empty, populate it
        if len(self._param_to_rank_cache) == 0:
            # Iterate over ranks and their corresponding parameter groups
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    # Assign each parameter in param_group['params'] to its rank in _param_to_rank_cache
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        # Return the parameter to rank mapping
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> Dict[torch.Tensor, int]:
        r"""
        :class:`dict` mapping parameters to their indices in the global optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
        if len(self._param_to_index_cache) == 0:
            # 构建参数到其在全局优化器状态中索引的映射字典
            self._param_to_index_cache = {
                p: i
                for i, p in enumerate(chain(*(g["params"] for g in self.param_groups)))
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> List[torch.Tensor]:
        r"""List mapping parameter indices in the global optimizer scheme to the actual params."""
        if len(self._index_to_param_cache) == 0:
            # 构建全局优化器方案中参数索引到实际参数的列表映射
            self._index_to_param_cache = list(
                chain(*(g["params"] for g in self.param_groups))
            )
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        r"""
        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
        assert not self._overlap_with_ddp, (
            "`_broadcast_params_from_rank()` should not be used if "
            "`overlap_with_ddp=True`; instead, the broadcasting should "
            "happen in the DDP communication hook"
        )
        handles = []
        if self.parameters_as_bucket_view:
            # 对于参数作为桶视图的情况，遍历桶并进行广播
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = dist.distributed_c10d.get_global_rank(
                    self.process_group, rank
                )
                handles.append(
                    dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                )
        else:
            # 对于非桶视图的参数，按照分区将参数广播到其他进程
            param_groups = self._partition_parameters()[rank]
            global_rank = dist.distributed_c10d.get_global_rank(
                self.process_group, rank
            )
            for param_group in param_groups:
                for param in param_group["params"]:
                    handles.append(
                        dist.broadcast(
                            tensor=param.data,
                            src=global_rank,
                            group=self.process_group,
                            async_op=True,
                        )
                    )
        return handles
    def _sync_params(self):
        r"""
        Sync all parameter shards across the ranks.

        This rank sends its shard of the parameters to all other ranks and
        receives a shard from each other rank. This is done using
        ``broadcast()``. Parameters are sent bucket-by-bucket if
        ``parameters_as_bucket_view=True``and sent parameter-by-parameter
        otherwise.
        """
        # 初始化一个空列表，用于存储所有广播操作的句柄
        handles = []
        # 遍历所有进程的排名
        for rank in range(self.world_size):
            # 调用 _broadcast_params_from_rank 方法广播当前进程的参数到指定排名的进程，并将返回的句柄添加到 handles 列表中
            handles.extend(self._broadcast_params_from_rank(rank))
        # 等待所有广播操作完成
        _ = [x.wait() for x in handles]

    @property
    def _device_to_params_per_rank(
        self,
    ) -> Dict[torch.device, List[List[torch.Tensor]]]:
        r"""
        Return device parameters assigned per rank.

        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
        # 断言条件，确保参数以 bucket 视图存储
        assert self.parameters_as_bucket_view, (
            "`_device_to_params_per_rank` should only be used if "
            "`parameters_as_bucket_view=True`"
        )
        # 如果缓存中没有设备到参数列表的映射，则进行初始化
        if len(self._device_to_params_per_rank_cache) == 0:
            # 遍历分区后的参数组
            for rank, param_groups in enumerate(self._partition_parameters()):
                # 遍历每个参数组
                for param_group in param_groups:
                    # 遍历参数组中的每个参数
                    for param in param_group["params"]:
                        # 获取参数的设备信息
                        device = param.device
                        # 如果设备信息尚未在缓存中，初始化一个列表以存储每个排名的参数列表
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [
                                [] for _ in range(self.world_size)
                            ]
                        # 将当前参数添加到对应设备和排名的列表中
                        self._device_to_params_per_rank_cache[device][rank].append(
                            param
                        )
        # 返回设备到参数列表映射的缓存
        return self._device_to_params_per_rank_cache

    def _get_min_index(
        self,
        values: List[int],
        disallowed_indices: Optional[Set[int]] = None,
    ) -> int:
        r"""
        Return ``values.index(min(values))``, except only uses one pass.

        It also excludes any indices in ``disallowed_indices`` if provided.

        Arguments:
            values: (List[int]): :class:`list` of values.
            disallowed_indices (Optional[Set[int]]): indices that are
                disallowed from being the returned min index.
        """
        min_index = -1  # 初始化最小值索引为-1
        min_value = float("inf")  # 初始化最小值为正无穷大
        for i, value in enumerate(values):  # 遍历 values 列表的索引和值
            if disallowed_indices and i in disallowed_indices:  # 如果存在不允许的索引并且当前索引在其中
                continue  # 跳过当前循环
            if value < min_value:  # 如果当前值小于最小值
                min_value = value  # 更新最小值
                min_index = i  # 更新最小值索引
        assert min_index >= 0, "All indices are disallowed"  # 断言最小值索引大于等于0，否则抛出异常
        return min_index  # 返回最小值索引

    def _assign_bucket_subset_to_rank(
        self,
        bucket_index: int,
        bucket_params: List[torch.Tensor],
        bucket_offset: int,
        assigned_rank: int,
        assigned_ranks_per_bucket: List[Set[int]],
    ) -> None:
        r"""
        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.

        The model parameters given by ``bucket_params`` represents a (possibly non-strict)
        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                gradient bucket.
            bucket_params (List[torch.Tensor]): subset of the parameters
                corresponding to the bucket to assign.
            bucket_offset (int): offset giving the index of the first element
                in ``bucket_params`` in the bucket's full parameter list.
            assigned_rank (int): group rank to assign to.
            assigned_ranks_per_bucket (List[Set[int]]): :class:`set` of group ranks
                assigned to each bucket.
        """
        overlap_info = self._overlap_info  # 获取重叠信息
        if len(bucket_params) == 0:  # 如果 bucket_params 为空
            raise ValueError("Empty bucket assignment")  # 抛出数值错误异常
        params_per_rank = overlap_info.params_per_rank  # 获取每个 rank 的参数信息
        offsets = overlap_info.offsets  # 获取偏移信息

        self._bucket_assignments_per_rank_cache[assigned_rank][
            bucket_index
        ] = _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)  # 将 bucket_params 分配给指定的 rank，并缓存相关信息
        if self.global_rank == assigned_rank:  # 如果全局 rank 等于指定的 rank
            offsets[bucket_index] = len(params_per_rank[assigned_rank])  # 更新偏移信息中指定 bucket_index 的值
        params_per_rank[assigned_rank].extend(bucket_params)  # 将 bucket_params 添加到指定 rank 的参数列表中
        assigned_ranks_per_bucket[bucket_index].add(assigned_rank)  # 将指定 rank 添加到指定 bucket_index 的分配 rank 集合中
        self._overlap_info.num_bucket_assignments += 1  # 增加分配的 bucket 数量

    @property
    def _local_step(
        self,
        gradients: Optional[List[Optional[torch.Tensor]]] = None,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ):  # 属性方法 _local_step 的定义
        """
        Placeholder for local step functionality. Returns None.
        """
        return None  # 返回空值

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        **kwargs: Any,
    ):  # step 方法的定义
        """
        Placeholder for step functionality. Returns None.
        """
        return None  # 返回空值
    ) -> Optional[float]:
        r"""
        Perform a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note: Any extra parameters are passed to the base optimizer as-is.
        """
        # 如果开启了与DDP的重叠，给出警告信息并返回空
        if self._overlap_with_ddp:
            logger.warning(
                "`step()` should not be included in the training loop when "
                "`overlap_with_ddp=True`"
            )
            return None

        # 执行本地优化器的一步操作
        loss = self._local_step(closure=closure, **kwargs)

        # 同步所有更新的参数分片到各个进程中
        self._sync_params()

        # 返回损失值
        return loss

    def join_hook(self, **kwargs):
        r"""
        Return the ZeRO join hook.

        It enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        This hook does not support any keyword arguments; i.e. ``kwargs`` is
        unused.
        """
        # 返回 ZeRO 合并钩子对象
        return _ZeROJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        r"""Return default device."""
        # 返回默认设备
        return self._default_device

    @property
    def join_process_group(self) -> Any:
        r"""Return process group."""
        # 返回进程组
        return self.process_group
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""
        从给定的 ``state_dict`` 中加载与给定排名相关的状态，根据需要更新本地优化器。

        Arguments:
            state_dict (dict): 优化器状态；应该是调用 :meth:`state_dict` 返回的对象。

        Raises:
            RuntimeError: 如果 ``overlap_with_ddp=True`` 并且在完全初始化此
                :class:`ZeroRedundancyOptimizer` 实例之前调用此方法，这在
                :class:`DistributedDataParallel` 梯度桶被重建后发生。
        """
        self._check_overlap_initialized()  # 检查重叠初始化是否完成

        for index, value in state_dict["state"].items():
            param = self._index_to_param[index]  # 根据索引获取参数
            if self._param_to_rank[param] != self.rank:
                # 清除与当前排名无关的状态
                state_dict["state"][index] = None
            else:
                # 将参数状态加载到本地优化器中
                self.optim.state[param] = _recursive_copy_to_device(
                    value, non_blocking=True, device=param.device
                )
                # 强制将零维张量（如Adam的“step”）放在CPU上
                for state_name, state_value in self.optim.state[param].items():
                    if torch.is_tensor(state_value) and state_value.dim() == 0:
                        self.optim.state[param][state_name] = state_value.cpu()

        super().load_state_dict(state_dict)  # 调用父类方法加载状态字典

        # 同步输入状态与暴露和本地优化器状态
        self._sync_param_groups(state_dict["param_groups"], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    @staticmethod
    def _sync_param_groups(
        src_param_groups: List[Dict[Any, Any]],
        dst_param_groups: List[Dict[Any, Any]],
    ) -> None:
        r"""
        将源参数组的属性与目标参数组同步。

        示例属性包括学习率或调度器属性。两个参数组应该有相同的长度（即相同数量的参数组）。

        Arguments:
            src_param_groups (list[dict]): 提供要复制的属性设置的参数组。
            dst_param_groups (list[dict]): 提供要设置的属性设置的参数组。
        """
        assert len(src_param_groups) == len(
            dst_param_groups
        ), "源参数组和目标参数组的数量不匹配"
        for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
            # 同步除参数之外的所有属性
            for attr in filter(lambda x: x != "params", src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]
    def _build_param_buckets(self) -> None:
        r"""
        Build parameter buckets if ``parameters_as_bucket_view=True``.

        For each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This method is called in the constructor and any time parameter
        trainability is changed.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
        # 如果不使用参数作为桶视图或者存在与 DDP 重叠，直接返回
        if not self.parameters_as_bucket_view or self._overlap_with_ddp:
            return

        # `self._buckets[i][j]` 表示存储在设备 i 上并分配给 rank j 的参数
        num_devices = len(self._device_to_params_per_rank)
        self._buckets = [[] for _ in range(num_devices)]  # type: ignore[assignment]

        # 遍历每个设备及其对应的参数分配情况
        for dev_i, (device, params_per_rank) in enumerate(
            self._device_to_params_per_rank.items()
        ):
            # 遍历每个参数列表
            for params in params_per_rank:
                bucket_size = 0
                dtype = None
                trainable_params = []
                # 检查每个参数是否可训练，如果不可训练则克隆以避免数据被破坏
                for param in params:
                    if not _is_trainable(param):
                        # 克隆参数以防止数据被破坏
                        param.data = param.data.detach().clone()
                    else:
                        bucket_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype  # 假设所有参数具有相同的数据类型

                # 如果 bucket_size 为 0，则创建一个虚拟桶
                if bucket_size == 0:
                    bucket = torch.zeros(1, device=device)
                else:
                    # 构建桶（假设所有参数都是密集的且具有相同的数据类型）
                    bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next
                self._buckets[dev_i].append(bucket)  # type: ignore[arg-type]
    def _build_ddp_param_buckets(self) -> None:
        r"""
        Build the DDP bucket with parameters assigned to this rank.

        For each DDP bucket with parameters assigned to this rank, flattens the
        data of those parameters into a single tensor and saves the tensor to
        the ``tensor`` attribute in the corresponding
        :class:`_DDPBucketAssignment` instance stored in
        ``self._bucket_assignments_per_rank``.

        :class:`DistributedDataParallel` guarantees that the parameters
        corresponding to a gradient bucket have the same device and the same
        dtype.
        """
        # Iterate over each rank's bucket assignments
        for bucket_assignments in self._bucket_assignments_per_rank:
            # Iterate over each bucket assignment in current rank's bucket assignments
            for bucket_assignment in bucket_assignments.values():
                # Get the parameters assigned to the current bucket assignment
                params = bucket_assignment.parameters
                bucket_size = 0
                dtype = None
                # Iterate over each parameter in the current bucket's parameters
                for param in params:
                    # Check if the parameter requires gradients (should be trainable)
                    assert _is_trainable(param), (
                        "Model parameter "
                        "corresponding to a gradient in a DDP bucket should "
                        "require a gradient"
                    )
                    # Accumulate the total size of the bucket
                    bucket_size += param.numel()
                    # Set the dtype of the bucket (assuming all parameters have the same dtype)
                    dtype = param.dtype  # assumes all same dtype
                # Ensure the bucket is not empty
                assert bucket_size > 0, "Empty bucket"

                # Construct the bucket tensor (assuming all dense and same dtype)
                # Create an empty tensor for the bucket with specified dtype and device
                tensor = torch.empty(
                    bucket_size, dtype=dtype, device=bucket_assignment.device
                )
                offset = 0
                # Flatten and copy each parameter's data into the bucket tensor
                for param in params:
                    offset_next = offset + param.numel()
                    tensor[offset:offset_next].copy_(param.data.flatten())
                    # Update the parameter's data to reference the tensor slice
                    param.data = tensor[offset:offset_next].view_as(param.data)
                    offset = offset_next
                # Store the constructed tensor in the bucket_assignment object
                bucket_assignment.tensor = tensor
    ) -> Union[List[torch.Tensor], List[dict]]:
        r"""
        Verify the type of ``params`` and initializes ``self._all_params`` as a :class:`list` of all parameters.

        The initializagtion will first make sure that provided ``params`` is valid.

        Arguments:
            params (Any): Candidate parameter list or parameter groups to verify.

        Raises:
            TypeError: ``params`` has an invalid type.
            ValueError: ``params`` is empty.

        Returns:
            The persistent form of ``params`` to be passed into the parent
            :class:`Optimizer` constructor -- i.e. returns ``params`` as a
            :class:`list` to ensure that it can be iterated over again.
        """
        # Check if params is a single Tensor, raise error if so
        if isinstance(params, torch.Tensor):
            raise TypeError(
                "`params` argument should be an iterable of "
                f"Tensors, but got {torch.typename(params)}"
            )
        
        try:
            # Attempt to convert params into a list
            all_params = list(params)
        except TypeError as e:
            # Raise a more specific TypeError if params cannot be converted
            raise TypeError(
                "`params` argument should be an iterable of Tensors"
                f" or dicts, but got {torch.typename(params)}"
            ) from e
        
        # Check if the list of params is empty, raise ValueError if so
        if len(all_params) == 0:
            raise ValueError("ZeroRedundancyOptimizer got an empty parameter list")
        
        # Initialize flags to check if all elements are Tensors or all are dicts
        all_tensors = True
        all_dicts = True
        
        # Iterate over all_params to check types
        for param in all_params:
            all_tensors &= isinstance(param, torch.Tensor)
            all_dicts &= isinstance(param, dict)
        
        # If neither all_tensors nor all_dicts is True, raise TypeError
        if not all_tensors and not all_dicts:
            raise TypeError(
                "`params` argument should be an iterable of Tensors or dicts"
            )
        
        # Ensure that `self._all_params` contains a list of all parameters
        if all_tensors:
            self._all_params = all_params
        elif all_dicts:
            self._all_params = []
            # Iterate over param_group in all_params to gather parameters
            for param_group in all_params:
                if "params" not in param_group:
                    raise ValueError(
                        "Each parameter group passed-in via `params` must "
                        "have a 'params' key mapping to the parameters in "
                        "the group"
                    )
                # Extend self._all_params with parameters from param_group
                self._all_params.extend(param_group["params"])
        
        # Return the validated params as a list
        return all_params
    def _verify_same_dense_param_type(self) -> None:
        r"""
        Verify that all parameters are of the same dense type.

        The method assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This method can be removed once support for sparse parameters
        and varying parameter types is added.
        """
        # 获取第一个参数的类型名
        typename = torch.typename(self._all_params[0])
        
        # 检查第一个参数是否为稀疏参数，如果是，则抛出异常
        if self._all_params[0].is_sparse:
            raise ValueError(
                "ZeroRedundancyOptimizer only supports using "
                "the same dense type for all parameters but got "
                f"{typename}"
            )
        
        # 遍历所有参数，检查是否有不同类型的参数
        for param in self._all_params[1:]:
            other_typename = torch.typename(param)
            # 如果有不同类型的参数，则抛出异常
            if other_typename != typename:
                raise ValueError(
                    "ZeroRedundancyOptimizer only supports "
                    "using the same dense type for all "
                    f"parameters but got both {typename} and "
                    f"{other_typename}"
                )

    def _get_is_trainable_mask(self) -> List[bool]:
        r"""Return a boolean mask indicating if each parameter is trainable (``requires_grad``) or not."""
        # 对所有参数应用 `_is_trainable` 函数，返回是否可训练的布尔掩码列表
        return list(map(_is_trainable, self._all_params))

    def _init_zero_for_overlap(self) -> None:
        r"""Perform a delayed initialization of the local optimizer and the supporting data structures."""
        # 断言 `_init_zero_for_overlap()` 方法仅在 `overlap_with_ddp=True` 时调用
        assert self._overlap_with_ddp, (
            "`_init_zero_for_overlap()` should only be called when "
            "`overlap_with_ddp=True`"
        )
        # 将重叠信息状态设置为已初始化，并清除缓存
        self._overlap_info.status = _OverlapStatus.INITIALIZED
        self._clear_cache()
        # 将参数分区，并构建 DDP 参数桶
        self._partition_parameters(self._overlap_info.params_per_rank)
        self._build_ddp_param_buckets()
        # 初始化本地优化器
        self._init_local_optimizer()

    def _get_assigned_rank(self, bucket_index: int) -> int:
        r"""
        Return the single rank assigned to a :class:`DistributedDataParallel` gradient bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                bucket for which to get the assigned rank.
        """
        # 断言不应使用 `_get_assigned_rank` 方法来处理分片桶
        assert not self._overlap_info.shard_buckets, (
            "The bucket assignment requires global bucket information and "
            "will be computed later; there should be no need to use this "
            "method"
        )
        # 返回分配给给定梯度桶的单个排名
        return bucket_index % self.world_size
    # 检查是否已延迟初始化，取决于 overlap_with_ddp 的值

    # 如果 overlap_with_ddp 为 True，则已经发生了延迟初始化（参见 _init_zero_for_overlap 方法）
    # 如果不是，则抛出 RuntimeError。这应该放在那些在延迟初始化之前不应该运行的方法之前。

    # 抛出异常：
    # RuntimeError: 如果 overlap_with_ddp 为 True，但 _init_zero_for_overlap 方法尚未被调用。
    def _check_overlap_initialized(self):
        if (
            self._overlap_with_ddp
            and self._overlap_info.status != _OverlapStatus.INITIALIZED
        ):
            raise RuntimeError(
                "This method should not be called until this "
                "ZeroRedundancyOptimizer instance has been fully "
                "initialized"
            )
    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        r"""
        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.

        Returns:
            - ``optimizer_class`` if ``overlap_with_ddp=False`` and
                ``optimizer_class`` is not a functional optimizer.
            - ``optimizer_class`` if ``overlap_with_ddp=True`` and
                ``optimizer_class`` is already a functional optimizer.
            - The functional equivalent of ``optimizer_class`` if
                ``overlap_with_ddp=True`` and ``optimizer_class`` is not
                already a functional optimizer (assuming the equivalent
                exists).

        Raises:
            ValueError:

                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is
                    neither a functional optimizer nor translatable to a
                    functional optimizer.
                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a
                    functional optimizer.
        """
        functional_optims = functional_optim_map.values()
        
        # Check if `overlap_with_ddp` is False
        if not self._overlap_with_ddp:
            # Check if `optimizer_class` is a functional optimizer
            if optimizer_class in functional_optims:
                # Raise an error since functional optimizers are not supported with `overlap_with_ddp=False`
                raise ValueError(
                    f"Passing in a functional optimizer {optimizer_class} "
                    "when `overlap_with_ddp=False`"
                )
            else:
                # Return the optimizer class directly if not a functional optimizer
                return optimizer_class
        else:
            # Check if `optimizer_class` is already a functional optimizer
            if optimizer_class in functional_optims:
                # Return the optimizer class as it is
                return optimizer_class
            elif optimizer_class in functional_optim_map:
                # Translate `optimizer_class` to its functional equivalent
                optim_constructor = functional_optim_map[optimizer_class]
                # Log the transformation to a functional optimizer
                logger.info(
                    "Using the functional optimizer %s "
                    "instead of %s since "
                    "`overlap_with_ddp=True`",
                    optim_constructor,
                    optimizer_class,
                )
                return optim_constructor
            else:
                # Raise an error if no functional optimizer equivalent exists
                raise ValueError(
                    "Using `ddp_with_overlap=True` requires using a "
                    "functional optimizer, but there is no supported functional "
                    f"optimizer equivalent for {optimizer_class}"
                )
```