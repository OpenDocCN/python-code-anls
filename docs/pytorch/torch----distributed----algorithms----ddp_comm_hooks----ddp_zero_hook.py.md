# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\ddp_zero_hook.py`

```py
# mypy: allow-untyped-defs
# 引入弱引用模块，用于对象的弱引用管理
import weakref
# 引入类型提示模块中的类型定义
from typing import Any, Callable, List, Optional

# 引入 PyTorch 主模块
import torch
# 引入分布式通信模块
import torch.distributed as dist
# 引入 ZeRO 优化器模块
from torch.distributed.optim import ZeroRedundancyOptimizer
# 引入 ZeRO 优化器内部状态模块
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
# 引入分布式数据并行模块
from torch.nn.parallel.distributed import DistributedDataParallel

# 暴露给外部的函数和方法名列表
__all__ = ["hook_with_zero_step", "hook_with_zero_step_interleaved"]

# 定义一个常量，表示没有参数更新
_NO_PARAM_UPDATE: None = None


def _perform_local_step(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
):
    r"""
    使用由 ``bucket`` 提供的梯度执行本地优化器步骤。

    参数:
        bucket (dist.GradBucket): 提供梯度的桶。
        zero (ZeroRedundancyOptimizer): 执行 :meth:`_local_step` 的 :class:`ZeroRedundancyOptimizer` 实例。
        rank (int): 调用进程的等级。

    .. warning::
        此函数假设已经进行了适当的同步，以便使用桶的梯度。
    """
    # 获取重叠信息
    overlap_info = zero._overlap_info
    # 获取桶的索引
    bucket_index = bucket.index()
    # 断言只有一个参数组，因为重叠 DDP 和 ZeRO 只支持单个参数组
    assert (
        len(zero.optim.param_groups) == 1
    ), "Overlapping DDP with ZeRO only supports a single parameter group"

    # 构造本地优化器步骤的 `gradients` 输入，期望在列表位置为 `None` 表示不更新对应的参数
    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
    gradients: List[Optional[torch.Tensor]] = [
        _NO_PARAM_UPDATE for _ in range(num_local_optim_params)
    ]
    # 断言桶索引在重叠信息的偏移量中存在
    assert (
        bucket_index in overlap_info.offsets
    ), f"Bucket index {bucket_index} was not assigned to rank {rank}"
    # 获取梯度偏移量
    gradients_offset = overlap_info.offsets[bucket_index]
    # 获取桶分配
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    # 获取桶的偏移量和长度
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    # 获取桶的梯度
    bucket_gradients = bucket.gradients()[bucket_offset : bucket_offset + length]
    # 将桶的梯度填充到对应的位置
    for i, grad in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad

    # 执行本地优化器步骤
    zero._local_step(gradients)


def _broadcast_bucket(
    bucket_index: int,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    广播桶的参数。

    参数:
        bucket_index (int): 桶索引，对应要广播参数的桶。
        zero (ZeroRedundancyOptimizer): 调用进程的 :class:`ZeroRedundancyOptimizer` 实例。
    """
    # 获取重叠信息
    overlap_info = zero._overlap_info
    # 断言分配给桶的排名数大于桶索引
    assert (
        len(overlap_info.assigned_ranks_per_bucket) > bucket_index

    ), f"Bucket index {bucket_index} was not assigned to any rank"
    ), "`assigned_ranks_per_bucket` is not fully constructed"
    # 确保跨所有排名的相同排序顺序
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    # 断言确保分配给某个桶的排名数量大于0
    assert len(assigned_ranks) > 0, (
        f"Bucket {bucket_index} should be " "assigned to at least one rank"
    )
    # 遍历分配给当前桶的所有排名
    for assigned_rank in assigned_ranks:
        # 获取分配给当前排名的所有桶分配
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        # 如果当前桶在当前排名的桶分配中
        if bucket_index in bucket_assignments:
            # 将当前桶的数据 tensor 进行广播
            overlap_info.broadcast_handles.append(
                dist.broadcast(
                    bucket_assignments[bucket_index].tensor,
                    src=dist.get_global_rank(zero.process_group, assigned_rank),
                    group=zero.process_group,
                    async_op=True,
                )
            )
def _save_ddp_bucket_info(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Save :class:`DistributedDataParallel` gradient bucket information for :class:`ZeroRedundancyOptimizer` instance ``zero``.

    In particular, this function is meant to be called upon seeing each
    gradient bucket to use when overlapping, meaning it does not save or compute any global
    information.

    Arguments:
        bucket (dist.GradBucket): the current gradient bucket.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info  # 获取零冗余优化器实例的重叠信息
    bucket_params = bucket.parameters()  # 获取当前梯度桶中的参数

    assert len(bucket_params) > 0, "Empty bucket"  # 断言：确保梯度桶中至少有一个参数

    # 将参数保存在梯度桶的参数列表中
    overlap_info.params_per_bucket.append(bucket_params)

    if overlap_info.shard_buckets:
        # 另外保存桶的大小，以供分配启发式算法使用
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()  # 计算参数的元素数量并累加
        assert overlap_info.total_size is not None
        overlap_info.total_size += bucket_size  # 累加到重叠信息的总大小中


def _hook_with_zero_step_setup(
    ddp_ref: weakref.ReferenceType,
    zero: ZeroRedundancyOptimizer,
    bucket: dist.GradBucket,
):
    r"""
    Encapsulate the setup logic for :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    This means the logic to run in the
    hook before the backward pass and optimizer step can actually be
    overlapped. This is factored out since it is common to both
    :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    Arguments:
        ddp_ref (weakref.ReferenceType): weak reference to the process's
            :class:`DistributedDataParallel` instance.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        bucket (dist.GradBucket): the current gradient bucket.
    """
    # 直到DDP桶被重建之前，继续正常进行
    if not ddp_ref()._has_rebuilt_buckets:  # type: ignore[union-attr]
        assert zero._overlap_info.status == _OverlapStatus.UNINITIALIZED
        return

    bucket_index = bucket.index()  # 获取梯度桶的索引
    overlap_info = zero._overlap_info  # 获取零冗余优化器实例的重叠信息

    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS  # 将状态设置为DDP已重建桶
    # 检查重叠信息对象的状态是否为DDP_HAS_REBUILT_BUCKETS
    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        # 如果bucket_index为0且overlap_info.params_per_bucket的长度大于0
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            # 这对应于反向传播的第一个桶，在保存所有信息后立即进行，
            # 因此我们可以执行延迟的ZeRO初始化
            zero._init_zero_for_overlap()
        else:
            # 一旦DDP桶已重建但ZeRO尚未正确初始化，保存所需的信息
            _save_ddp_bucket_info(bucket, zero)
# 定义一个函数，用于修改给定的 hook，使其能够将 ZeroRedundancyOptimizer 的优化步骤与 DistributedDataParallel 的反向传播重叠
def hook_with_zero_step(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).
    The optimizer step computation begins after the last gradient bucket computation has finished.

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL/HCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    # 如果 ZeroRedundancyOptimizer 没有与 DistributedDataParallel 重叠，则抛出 ValueError 异常
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    # 创建 DistributedDataParallel 的弱引用
    ddp_ref = weakref.ref(ddp)

    # 注意：使用此重叠方法可能会导致 Gloo 出现问题，因此我们要求
    # 获取当前使用的分布式数据并行（DDP）的后端类型，参考链接用于当前使用的后端
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    # 检查当前后端类型是否为NCCL或"hccl"，如果不是则抛出运行时错误
    if (pg != dist.Backend.NCCL) and (pg != "hccl"):
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL/HCCL backend to avoid hangs"
        )

    # 如果设置了分片桶(shard_buckets)，则更新ZeRO的重叠信息
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    # 定义一个带有ZeRO函数的钩子，接受状态和梯度桶作为参数，返回钩子函数对象
    def hook_with_zero_fn(
        state: Any,
        bucket: dist.GradBucket,
    return hook_with_zero_fn
    # 定义一个函数，修改给定的 hook 函数，使其可以在 ZeroRedundancyOptimizer 的优化步骤与 DistributedDataParallel 的反向传播同时进行交错执行
    def hook_with_zero_step_interleaved(
        hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
        ddp: DistributedDataParallel,
        zero: ZeroRedundancyOptimizer,
        shard_buckets: bool = False,
    ) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
        r"""
        Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass
    
        This approach overlaps the optimizer computation and communication with the
        backward computation and communication. In particular, once a bucket's
        gradients have been computed, the optimizer computation using those
        gradients is launched (though the actual computation must wait for the
        bucket's all-reduce to complete). This yields an interleaving of all-
        reduces and broadcasts in the communication stream.
    
        This approach may be preferred over :meth:`hook_with_zero_step` if
        communication is relatively fast compared to computation.
    
        Arguments:
            hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
                modify.
            ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
                instance to use.
            zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
                instance to use.
            shard_buckets (bool): if ``True``, then the assignment of each
                :class:`DistributedDataParallel` bucket is partitioned across
                possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
                across possibly multiple ranks) to approximate uniformity; if
                ``False``, then each bucket is wholly assigned to a single
                :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).
    
        Returns:
            The modified hook.
    
        Raises:
            ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
            RuntimeError: if using any backend other than NCCL since currently
                Gloo may hang.
    
        .. warning::
            Given the way that overlapping :class:`DistributedDataParallel` with
            :class:`ZeroRedundancyOptimizer` is currently implemented, the first
            two or three training iterations do not perform parameter updates in
            the optimizer step, depending on if ``static_graph=False`` or
            ``static_graph=True``, respectively. This is because it needs
            information about the gradient bucketing strategy used by
            :class:`DistributedDataParallel`, which is not finalized until the
            second forward pass if ``static_graph=False`` or until the third
            forward pass if ``static_graph=True``.
        """
        # 如果 ZeroRedundancyOptimizer 没有使用 overlap_with_ddp=True 构造，则抛出 ValueError 异常
        if not zero._overlap_with_ddp:
            raise ValueError(
                "ZeroRedundancyOptimizer must be constructed with "
                "`overlap_with_ddp=True` to use this hook properly"
            )
        # 使用弱引用引用 DistributedDataParallel 实例，以便稍后访问
        ddp_ref = weakref.ref(ddp)
    
        # 注意: Gloo 可能会在此重叠方法中 hang，因此我们要求
    # 使用 dist 模块获取 DDP 进程组的后端类型，将其赋值给 pg 变量
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    # 如果后端类型不是 NCCL 或 "hccl"，则抛出运行时错误
    if (pg != dist.Backend.NCCL) and (pg != "hccl"):
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL/HCCL backend to avoid hangs"
        )

    # 如果设置了 shard_buckets 标志，更新 ZeRO 的重叠信息
    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    # 定义一个带有 ZeRO 交错功能的钩子函数
    def hook_with_zero_interleaved_fn(
        state,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Return :class:`Future` that gives gradient bucket tensor and performs partial :class:`ZeroRedundancyOptimizer` :meth:`step`.

        This function uses the gradients in gradient in given bucket to perform a partial
        :class:`ZeroRedundancyOptimizer` :meth:`step`

        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        # 调用默认的 hook 函数，并将其返回值赋给 fut 变量
        fut = hook(state, bucket)
        # 调用 _hook_with_zero_step_setup 函数，设置 ZeRO 相关信息
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        # 如果 ZeRO 的重叠信息状态不是 INITIALIZED，则直接返回 fut
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        # 定义实现部分 ZeRORedundancyOptimizer step 的函数 zero_step
        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            r"""
            Perform partial :class:`ZeroRedundancyOptimizer` :meth:`step` using gradients in the :class:`DistributedDataParallel`.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank

            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            # 如果当前 rank 在分配的 ranks 中，则执行本地步骤 _perform_local_step
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)

            # 广播当前 bucket 的更新
            _broadcast_bucket(bucket_index, zero)

            num_buckets = len(overlap_info.params_per_bucket)
            # 如果已经见过的 bucket 数量等于总 bucket 数量，确保所有参数更新完成
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()

            # 返回当前 bucket 的数据内容
            return bucket.buffer()

        # 将 zero_step 函数作为 fut 的后续操作，并返回其结果
        return fut.then(zero_step)

    # 返回定义的 hook_with_zero_interleaved_fn 函数
    return hook_with_zero_interleaved_fn
```