# `.\pytorch\torch\distributed\fsdp\_unshard_param_utils.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理模块
import contextlib
# 引入警告模块
import warnings
# 引入类型提示中的类型转换函数
from typing import cast, Generator

# 引入 PyTorch 库
import torch
# 引入 FSDP 模块下的遍历工具
import torch.distributed.fsdp._traversal_utils as traversal_utils
# 引入 PyTorch 的神经网络模块
import torch.nn as nn
# 引入 FSDP 模块中的共享参数和训练状态相关工具函数
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state,
    _has_fsdp_params,
    _module_handle,
    HandleTrainingState,
    TrainingState,
)
# 引入 FSDP 运行时工具函数
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
    _reshard,
    _reshard_grads,
    _unshard,
    _unshard_grads,
)
# 引入分布式工具模块中的断言函数
from torch.distributed.utils import _p_assert

# 引入本地的扁平参数处理类
from ._flat_param import FlatParamHandle


# 定义全局常量，用于标识扁平参数
FLAT_PARAM = "_flat_param"


# 使用 torch.no_grad() 装饰器定义函数，禁用梯度计算
@torch.no_grad()
def _writeback_to_local_shard(
    handle: FlatParamHandle,
    writeback_grad: bool,
):
    """
    For the handle, writes back the this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: The handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """
    
    # 定义内部函数，用于获取分片数据
    def _get_shard(flat_param_or_grad: torch.Tensor) -> torch.Tensor:
        if handle.uses_sharded_strategy:
            # 对于分片策略，获取未填充的分片而不是填充的分片，
            # 以便持久化用户对填充的更改（尽管 FSDP 并不显式支持这一点）
            shard, _ = FlatParamHandle._get_unpadded_shard(
                flat_param_or_grad,
                handle.rank,
                handle.world_size,
            )
            return shard
        # 对于 `NO_SHARD`，`flat_param` 或其梯度可能已被修改，因此直接写回
        return flat_param_or_grad

    # 获取参数分片
    param_shard = _get_shard(handle.flat_param)
    # 将参数分片的数据复制到本地分片
    handle.flat_param._local_shard[: param_shard.numel()].copy_(param_shard)  # type: ignore[attr-defined]
    # 如果需要写回梯度，则执行以下操作
    if writeback_grad:
        existing_grad = handle.sharded_grad
        if existing_grad is not None:
            # 断言确保参数的梯度存在
            assert handle.flat_param.grad is not None
            # 获取梯度分片
            grad_shard = _get_shard(handle.flat_param.grad)
            # 将梯度分片的数据复制到现有的梯度分片中
            existing_grad[: grad_shard.numel()].copy_(grad_shard)


# 定义函数，用于从封装的模块中取消注册扁平参数
def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    De-registers the flattened parameter from the wrapped module, hiding it
    from ``nn.Module`` methods.

    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
    attribute but dynamically change whether it is visible to ``nn.Module``
    methods.
    """
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        # 从模块中移除 FLAT_PARAM 属性，使其对 nn.Module 方法不可见
        cast(nn.Module, module.module)._parameters.pop(FLAT_PARAM, None)


# 定义函数，用于向封装的模块注册扁平参数
def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.
    """
    # 将 FLAT_PARAM 属性注册到模块中，使其对 nn.Module 方法可见
    cast(nn.Module, module.module)._parameters[FLAT_PARAM] = None
    """
    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
    # 从状态和模块中获取处理后的句柄
    handle = _module_handle(state, module)
    # 检查模块是否包含FSDP参数
    if _has_fsdp_params(state, module):
        # TODO: 解决可组合API的情况。
        # 将 handle.flat_param 赋值给 module.module 的 _parameters[FLAT_PARAM] 属性
        cast(nn.Module, module.module)._parameters[FLAT_PARAM] = handle.flat_param
@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    """
    Assumes that the flattened parameter is unsharded. When in the context,
    de-registers the flattened parameter and unflattens the original
    parameters as ``nn.Parameter`` views into the flattened parameter.
    After the context, re-registers the flattened parameter and restores
    the original parameters as ``Tensor`` views into the flattened
    parameter.
    """
    # 获取模块对应的状态句柄
    handle = _module_handle(state, module)
    # 如果句柄不存在，则直接返回
    if not handle:
        yield
    else:
        # 反注册平坦化参数
        _deregister_flat_param(state, module)
        try:
            # 使用句柄的 unflatten_as_params 方法进入上下文环境
            with handle.unflatten_as_params():
                yield
        finally:
            # 如果不使用原始参数，则重新注册平坦化参数
            if not handle._use_orig_params:
                _register_flat_param(state, module)


def _validate_unshard_params_args(
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
) -> None:
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        # 如果需要梯度并且满足 offload_to_cpu 或者不使用原始参数，则抛出未实现的错误
        raise NotImplementedError(
            f"with_grads={with_grads}, "
            f"use_orig_params={state._use_orig_params}, "
            f"offload_to_cpu={offload_to_cpu} "
            f"is not supported yet"
        )
    if offload_to_cpu and state._handle and (not state._handle.uses_sharded_strategy):
        # 如果 offload_to_cpu=True 并且状态句柄使用 NO_SHARD 策略，则抛出未实现的错误
        raise NotImplementedError(
            "offload_to_cpu=True and NO_SHARD is not supported yet"
        )
    if writeback and rank0_only:
        # 如果 writeback=True 并且 rank0_only=True，则抛出未实现的错误
        # TODO: Rank 0 can broadcast the `FlatParameter` to allow all ranks to
        # persist the changes.
        raise NotImplementedError(
            "writeback=True and rank0_only=True is not supported yet"
        )
    if offload_to_cpu and not rank0_only:
        # 如果 offload_to_cpu=True 并且 rank0_only=False，则发出警告，可能导致冗余的 CPU 内存拷贝
        warnings.warn(
            "offload_to_cpu=True and rank0_only=False may result in the"
            "unsharded parameters being redundantly copied to CPU memory for "
            "GPUs sharing the same CPU memory, which risks CPU OOM. We "
            "recommend using offload_to_cpu=True with rank0_only=True."
        )


@contextlib.contextmanager
def _unshard_fsdp_state_params(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    """
    This unshards the parameters for a single FSDP state ``state`` that
    corresponds to ``module``.
    """
    # 验证参数是否符合要求
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    # 同步设备句柄
    state._device_handle.synchronize()
    # 如果模块句柄已经存在且训练状态不是 SUMMON_FULL_PARAMS，则使用现有句柄
    maybe_handle = _module_handle(state, module)
    handle = None
    if (
        maybe_handle
        and maybe_handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS
    ):
        handle = maybe_handle
    if not handle:
        yield
        return

    # 断言句柄的训练状态是 IDLE
    assert (
        handle._training_state == HandleTrainingState.IDLE
    ), f"Expects the handle training to be IDLE but got {handle._training_state}"
    # 检查训练状态是否为 IDLE，如果不是则抛出异常信息

    handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS
    # 设置训练状态为 SUMMON_FULL_PARAMS，表示正在准备完整参数

    _reset_flat_param_grad_info_if_needed(handle)
    # 重置平坦参数梯度信息（如果需要的话）

    free_unsharded_flat_param = handle.needs_unshard()
    # 检查是否需要进行非分片的平坦参数释放

    # 不需要在计算流中调用 `wait_stream()`，因为我们直接在计算流中进行非分片操作
    computation_stream = state._device_handle.current_stream()
    # 获取当前设备处理的计算流

    _unshard(state, handle, computation_stream, computation_stream)
    # 在计算流中对状态和句柄进行非分片操作

    if with_grads:
        _unshard_grads(handle)
        # 如果需要梯度，对梯度进行非分片操作

    if rank0_only and state.rank != 0:
        # 如果仅在 rank0 上运行并且当前进程不是 rank0
        # 提前释放非分片的平坦参数
        _reshard(state, handle, free_unsharded_flat_param)
        if with_grads:
            _reshard_grads(handle)
        try:
            yield
            # 执行用户代码块
        finally:
            handle._training_state = HandleTrainingState.IDLE
            # 最终将训练状态设置为 IDLE
    else:
        # 如果不是仅在 rank0 上运行或者是 rank0 进程
        # 对非分片的平坦参数进行展开
        with contextlib.ExitStack() as stack:
            # 确保 rank == 0 或者 !rank0_only 这一不变条件
            if offload_to_cpu and handle.uses_sharded_strategy:
                stack.enter_context(handle.to_cpu())
                # 如果需要向 CPU 转移，将句柄内容转移到 CPU 上

                # 注意：由于 PyTorch 要求参数及其梯度的元数据（如设备）匹配，
                # 我们必须在移动参数之后，将梯度移动到 CPU 上
            # 注意：这假设存在一个 `FlatParameter`
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
                # 如果不使用原始参数，则按参数展开为参数

            try:
                yield
                # 执行用户代码块
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                    # 如果需要写回，将结果写回到本地分片
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE
                # 最终将训练状态设置为 IDLE
# 定义一个上下文管理器 `_unshard_params_for_summon`，用于处理分片参数的相关操作
@contextlib.contextmanager
def _unshard_params_for_summon(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    # 验证传入参数的有效性
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    # 惰性初始化 FSDP 状态
    _lazy_init(state, module)
    # 如果处于前向/后向传播状态，则抛出错误
    if state.training_state == TrainingState.FORWARD_BACKWARD:
        raise AssertionError(
            "Cannot manually unshard parameters during forward/backward"
        )
    # 如果已经在解除分片参数状态，则抛出错误
    elif state.training_state == TrainingState.SUMMON_FULL_PARAMS:
        raise AssertionError(
            "Cannot manually unshard parameters when already unsharding parameters"
        )
    # 进入解除分片参数状态的上下文环境
    with _unshard_fsdp_state_params(
        module=module,
        state=state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    ):
        try:
            # 设置训练状态为 SUMMON_FULL_PARAMS，表示正在解除分片参数
            state.training_state = TrainingState.SUMMON_FULL_PARAMS
            # 使用 yield 返回控制权
            yield
        finally:
            # 最终将训练状态设置回 IDLE，表示空闲状态
            state.training_state = TrainingState.IDLE


# 定义另一个上下文管理器 `_unshard_params`，用于处理整个模块树中的分片参数
@contextlib.contextmanager
def _unshard_params(
    module: nn.Module,
    recurse: bool,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    """
    This unshards FSDP-managed parameters for all modules with FSDP applied in
    the module tree rooted at ``module``.
    """
    # 如果不需要递归处理
    if not recurse:
        # 获取模块的 FSDP 状态
        optional_state = _get_module_fsdp_state(module)
        # 如果模块没有 FSDP 状态，则使用空上下文管理器返回
        if optional_state is None:
            with contextlib.nullcontext():
                yield
            return
        # 准备模块状态和模块本身的列表
        states_and_modules = ([optional_state], [module])
    else:
        # 使用遍历工具获取所有包含 FSDP 状态的模块及其状态
        states_and_modules = traversal_utils._get_fsdp_states_with_modules(module)
    # 使用 ExitStack 来管理多个上下文环境
    with contextlib.ExitStack() as stack:
        # 遍历状态和模块列表，并进入上下文管理器 `_unshard_params_for_summon`
        for state, module in zip(*states_and_modules):
            stack.enter_context(
                _unshard_params_for_summon(
                    module=module,
                    state=state,
                    writeback=writeback,
                    rank0_only=rank0_only,
                    offload_to_cpu=offload_to_cpu,
                    with_grads=with_grads,
                )
            )
        # 使用 yield 返回控制权
        yield


# 定义函数 `_deregister_orig_params`，用于注销原始参数并注册 `FlatParameter`
def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the original parameters; registers the ``FlatParameter``.
    """
    # 获取模块句柄
    handle = _module_handle(state, module)
    # 如果句柄不存在，则直接返回
    if not handle:
        return
    # 断言 `_use_orig_params` 一致性，如果不一致则报错
    _p_assert(
        handle._use_orig_params,
        f"Inconsistent `_use_orig_params` -- FSDP: {state._use_orig_params} "
        f"handle: {handle._use_orig_params}",
    )
    # 注销原始参数
    handle._deregister_orig_params()
    # 注册 `FlatParameter`
    _register_flat_param(state, module)


# 定义函数 `_register_orig_params`，用于注册原始参数并注销 `FlatParameter`
def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    """
    Deregisters the ``FlatParameter``; registers the original parameters.
    """
    # 获取模块句柄
    handle = _module_handle(state, module)
    # 如果句柄不存在，则直接返回
    if not handle:
        return
    # 调用函数 _deregister_flat_param，用于取消状态和模块的平坦参数注册
    _deregister_flat_param(state, module)
    # 检查处理器是否使用分片视图处理平坦参数
    if handle.is_sharded(handle.flat_param):
        # 如果是分片的平坦参数，使用分片视图
        handle._use_sharded_views()
        # 使用分片的梯度视图
        handle._use_sharded_grad_views()
    else:
        # 如果不是分片的平坦参数，使用非分片视图，并将其作为参数
        handle._use_unsharded_views(as_params=True)
```