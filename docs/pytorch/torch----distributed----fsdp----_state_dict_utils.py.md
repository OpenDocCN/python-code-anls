# `.\pytorch\torch\distributed\fsdp\_state_dict_utils.py`

```
# mypy: allow-untyped-defs
# 导入所需模块
import contextlib  # 上下文管理工具
import logging  # 日志记录
import math  # 数学函数库
import warnings  # 警告控制
from typing import (  # 类型提示
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterator,
    List,
    no_type_check,
    Tuple,
)

import torch  # PyTorch 深度学习库
import torch.distributed as dist  # 分布式训练模块
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper  # 分布式训练中的检查点包装器
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 神经网络函数库
from torch.distributed._shard.sharded_tensor import (  # 分布式张量的相关操作
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed._tensor import DTensor  # 分布式张量的定义
from torch.distributed.device_mesh import _mesh_resources  # 设备网格资源管理
from torch.distributed.fsdp._common_utils import (  # FSDP 公共工具函数
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _has_fsdp_params,
    _is_composable,
    _module_handle,
    clean_tensor_name,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
)
from torch.distributed.fsdp._debug_utils import SimpleProfiler  # FSDP 调试工具：简单性能分析器
from torch.distributed.fsdp._runtime_utils import (  # FSDP 运行时工具函数
    _cast_buffers_to_dtype_and_device,
    _get_orig_buffer_dtypes,
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (  # FSDP API 接口定义
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.utils import _replace_by_prefix  # 分布式工具函数：替换前缀

from ._fsdp_extensions import (  # 导入 FSDP 扩展功能
    _ext_all_gather_dtensor,
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
    _ext_post_unflatten_transform,
    _ext_pre_load_state_dict_transform,
)
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM  # 参数解聚工具函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _should_unshard_params(fsdp_state: _FSDPState) -> bool:
    # 判断是否需要解聚参数
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD and (
        _is_composable(fsdp_state) or fsdp_state._use_orig_params
    ):
        return False
    else:
        return True


def _convert_to_wrapped_module_name(module_name: str) -> str:
    # 将模块名转换为包装后的模块名
    module_name = module_name.replace(f"{FSDP_PREFIX}", "")
    module_name = module_name.replace(f"{FSDP_WRAPPED_MODULE}", "")
    if module_name:
        module_name = f"{module_name}."
    # `CheckpointWrapper` 添加了一个前缀，需要将其移除
    module_name = module_name.replace(checkpoint_wrapper._CHECKPOINT_PREFIX, "")
    return module_name


def _param_name_infos(
    module: nn.Module, fsdp_state: _FSDPState
) -> Iterator[Tuple[str, str, str]]:
    # 获取参数名信息的迭代器
    if not _has_fsdp_params(fsdp_state, module):
        return
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        fqn = f"{module_name}{param_name}"
        yield fqn, param_name, module_name


def _shared_param_name_infos(
    module: nn.Module, fsdp_state: _FSDPState
) -> Iterator[Tuple[str, str, str]]:
    # 获取共享参数名信息的迭代器
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        yield param_name, module_name
        ).shared_param_module_names():
        # 调用 shared_param_module_names() 方法，返回一个迭代器，遍历所有共享参数模块的名称
        module_name = _convert_to_wrapped_module_name(module_name)
        # 调用 _convert_to_wrapped_module_name() 方法，将模块名转换为包装后的模块名
        fqn = f"{module_name}{param_name}"
        # 组合模块名和参数名，形成完全限定名称（Fully Qualified Name）
        yield fqn, param_name, module_name
        # 返回生成器，生成完全限定名称、参数名和模块名的元组
# 使用装饰器 @no_type_check 来标记该函数不进行类型检查
def _enter_unshard_params_ctx(
    module: nn.Module,
    fsdp_state: _FSDPState,
    writeback: bool = False,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``_unshard_fsdp_state_params``.
    """
    # 断言确保在进入 "_unshard_fsdp_state_params" 上下文之前，fsdp_state._unshard_params_ctx[module] 为 None
    assert module not in fsdp_state._unshard_params_ctx, (
        "Entering the ``_unshard_fsdp_state_params`` context but _unshard_params_ctx[module] "
        "is not None."
    )
    # 调用 _unshard_fsdp_state_params 函数，将其返回的上下文对象存储在 fsdp_state._unshard_params_ctx[module] 中
    fsdp_state._unshard_params_ctx[module] = _unshard_fsdp_state_params(
        module,
        fsdp_state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    )
    # 进入上下文管理器，调用 __enter__ 方法
    fsdp_state._unshard_params_ctx[module].__enter__()


@no_type_check
def _exit_unshard_params_ctx(module: nn.Module, fsdp_state: _FSDPState) -> None:
    """A helper function to exit ``_unshard_fsdp_state_params`` context."""
    # 调用上下文管理器的 __exit__ 方法，退出 "_unshard_fsdp_state_params" 上下文
    fsdp_state._unshard_params_ctx[module].__exit__(None, None, None)
    # 删除 fsdp_state._unshard_params_ctx[module]，清理上下文对象
    fsdp_state._unshard_params_ctx.pop(module)


def _common_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
    # 如果设备句柄可用，则同步设备
    if fsdp_state._device_handle.is_available():
        fsdp_state._device_handle.synchronize()
    # TODO: 需要检查这对于可组合的 FSDP 是否始终正确。
    # 进行懒初始化，即使是根节点也会调用 _lazy_init 函数
    _lazy_init(fsdp_state, module)
    # 如果是根节点，根据需要重置平坦参数梯度信息
    if fsdp_state._is_root:
        _reset_flat_param_grad_info_if_needed(fsdp_state._all_handles)


def _common_unshard_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    offload_to_cpu: bool,
    rank0_only: bool,
) -> None:
    """
    Performs the pre-state_dict tasks shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this hook.
    """
    # 对于可组合的 `fully_shard`，在 `NO_SHARD` 情况下不需要为参数取消分片
    if not _should_unshard_params(fsdp_state):
        return
    # 进入 _unshard_fsdp_state_params 上下文
    _enter_unshard_params_ctx(
        module,
        fsdp_state,
        writeback=False,
        offload_to_cpu=offload_to_cpu,
        rank0_only=rank0_only,
    )


@no_type_check
def _common_unshard_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
    param_hook: Callable,
) -> Dict[str, Any]:
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
    # 替换 state_dict 中以 FSDP_PREFIX 开头的键的前缀为 prefix
    _replace_by_prefix(state_dict, prefix + f"{FSDP_PREFIX}", prefix)
    # 对于简单的情况提前返回
    # 如果状态字典为空或者模块没有FSDP参数，则直接返回状态字典
    if not state_dict or not _has_fsdp_params(fsdp_state, module):
        # 如果应该解除分片参数，则调用_exit_unshard_params_ctx函数进行解除分片参数操作
        if _should_unshard_params(fsdp_state):
            _exit_unshard_params_ctx(module, fsdp_state)
        # 返回状态字典
        return state_dict

    # 如果rank0_only为True且当前rank不是0，则表示该rank只需要参与全聚合操作，不需要保存状态字典
    rank0_only = (
        fsdp_state._state_dict_type == StateDictType.FULL_STATE_DICT
        and cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only
    )
    # 如果no_fsdp_return为True，则表示当前rank返回的状态字典只包含非FSDP控制的参数和缓冲区
    no_fsdp_return = rank0_only and fsdp_state.rank != 0
    if no_fsdp_return and not fsdp_state._use_orig_params:
        # 遍历所有要清理的键名，并根据需要进行处理
        for clean_key in fsdp_state._buffer_names:
            # 这是为了支持激活检查点而进行的一个hack
            clean_key = clean_key.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
            # 从状态字典中移除对应的键
            state_dict.pop(f"{prefix}{clean_key}", None)
        # 如果rank不是0，则在状态字典中移除FLAT_PARAM对应的键
        state_dict.pop(f"{prefix}{FLAT_PARAM}")
        # 调用_exit_unshard_params_ctx函数解除分片参数
        _exit_unshard_params_ctx(module, fsdp_state)
        # 返回更新后的状态字典
        return state_dict

    # 遍历模块中保存的参数，以避免处理缓冲区
    for fqn, param_name, module_name in _param_name_infos(module, fsdp_state):
        fqn = f"{prefix}{fqn}"
        # 如果no_fsdp_return为True，则从状态字典中移除对应的键
        if no_fsdp_return:
            state_dict.pop(fqn)
            continue
        # 断言确保fqn在状态字典中存在
        assert fqn in state_dict, (
            f"FSDP assumes {fqn} is in the state_dict but the state_dict only "
            f"has {state_dict.keys()}. "
            f"prefix={prefix}, module_name={module_name}, "
            f"param_name={param_name} rank={fsdp_state.rank}."
        )

        # 调用param_hook函数处理参数
        param_hook(state_dict, prefix, fqn)

    # 如果应该解除分片参数，则调用_exit_unshard_params_ctx函数进行解除分片参数操作
    if _should_unshard_params(fsdp_state):
        _exit_unshard_params_ctx(module, fsdp_state)

    # 设置CPU设备
    cpu_device = torch.device("cpu")
    # 初始化buffer_clean_fqns和buffers列表
    buffer_clean_fqns = []
    buffers = []
    for clean_key in fsdp_state._buffer_names:
        # 遍历 FSDP 状态对象中的缓冲区名称列表
        # 这是为了支持激活检查点而进行的一个 hack。
        clean_key = clean_tensor_name(clean_key)  # 对清理后的键名进行处理
        fqn = f"{prefix}{clean_key}"  # 构建完全限定名称
        if fqn not in state_dict:
            # 如果完全限定名称不在状态字典中，说明该缓冲区可能未注册为持久化数据，跳过处理。
            continue
        if no_fsdp_return:
            state_dict.pop(fqn)  # 如果设置了 no_fsdp_return，从状态字典中删除该缓冲区
        else:
            buffer = state_dict[fqn]  # 获取状态字典中的缓冲区数据
            if (
                fsdp_state._state_dict_config.offload_to_cpu
                and buffer.device != cpu_device
            ):
                state_dict[fqn] = buffer.to(cpu_device)  # 如果配置要求，将缓冲区数据转移到 CPU 设备上
            # 跳过对被忽略缓冲区的类型提升处理
            if clean_key not in fsdp_state._ignored_buffer_names:
                buffer_clean_fqns.append(clean_key)  # 将清理后的键名添加到缓冲区完全限定名称列表中
                buffers.append(state_dict[fqn])  # 将缓冲区数据添加到缓冲区列表中

    if buffers:
        mixed_precision_enabled_for_buffers = (
            fsdp_state._mixed_precision_enabled_for_buffers()
            if not _is_composable(fsdp_state)
            else (fsdp_state.mixed_precision.buffer_dtype is not None)
        )
        if mixed_precision_enabled_for_buffers:
            buffer_dtypes = _get_orig_buffer_dtypes(fsdp_state, buffer_clean_fqns)
            _cast_buffers_to_dtype_and_device(
                buffers, buffer_dtypes, fsdp_state.compute_device
            )
            for buffer, clean_fqn in zip(buffers, buffer_clean_fqns):
                fqn = f"{prefix}{clean_fqn}"
                logger.info("FSDP is casting the dtype of %s to %s", fqn, buffer.dtype)
                state_dict[fqn] = buffer.clone()  # 将缓冲区克隆并更新到状态字典中
    return state_dict
@no_type_check
def _full_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    在调用 model.state_dict() 前运行的钩子函数。实际上，``nn.Module`` 并不支持 pre-state_dict 钩子。
    因此，此 API 从 ``_full_post_state_dict_hook()`` 中调用以模拟该情况。一旦 ``nn.Module`` 支持 pre-state_dict，
    此钩子将作为 nn.Module 的钩子注册。
    """
    if getattr(fsdp_state, "_device_mesh", False):
        # 获取父级网格资源，基于当前的设备网格
        parent_mesh = _mesh_resources.get_parent_mesh(fsdp_state._device_mesh)

    # 调用通用的 pre-state_dict 钩子
    _common_pre_state_dict_hook(module, fsdp_state)

    # 调用通用的 unshard pre-state_dict 钩子，设置是否将参数转移至 CPU、是否仅在 rank0 上执行
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=fsdp_state._state_dict_config.offload_to_cpu,
        rank0_only=cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only,
    )


@no_type_check
def _full_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    在调用 model.state_dict() 后、在将结果返回给用户前运行的钩子函数。对于 FSDP，由于参数在 _unshard_fsdp_state_params 结束后返回到分片版本，
    我们可能需要克隆 state_dict 中的张量，并移除 ``FSDP_WRAPPED_MODULE`` 前缀。
    """

    def param_hook(
        state_dict: Dict[str, Any],
        prefix: str,
        fqn: str,
    ) -> None:
        # 清理键名，移除前缀以适应缓冲区名称和参数名称
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix):]

        # 在退出 `_unshard_fsdp_state_params()` 上下文前，克隆参数
        if not getattr(state_dict[fqn], "_has_been_cloned", False):
            try:
                state_dict[fqn] = state_dict[fqn].clone().detach()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {fsdp_state.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}"
                )

    # 调用通用的 unshard post-state_dict 钩子
    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


def _full_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    # 惰性初始化 FSDP 状态
    _lazy_init(fsdp_state, module)
    # 如果满足解除分片参数的条件，则执行以下操作
    if _should_unshard_params(fsdp_state):
        # 使用简单分析器记录进入“_enter_unshard_params_ctx”上下文的性能数据
        with SimpleProfiler.profile("_enter_unshard_params_ctx"):
            # 调用函数进入解除分片参数的上下文，并传入模块、状态、写回标志为True
            _enter_unshard_params_ctx(module, fsdp_state, writeback=True)
    
    # 仅对非可组合的FSDP状态执行以下操作：使用FSDP_PREFIX替换state_dict中的prefix
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")
def _full_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    # 如果应该取消分片参数，则执行下面的操作
    if _should_unshard_params(fsdp_state):
        # 使用简单分析器记录退出非分片参数上下文的操作
        with SimpleProfiler.profile("_exit_unshard_params_ctx"):
            _exit_unshard_params_ctx(module, fsdp_state)


def _local_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    在调用 model.state_dict() 之前运行的钩子。目前，PyTorch 核心不支持预状态字典钩子。
    因此，此 API 从 `_local_post_state_dict_hook()` 中调用以模拟情况。
    """
    # 如果当前模块包含 FSDP 参数并且没有使用分片策略，则引发运行时错误
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``local_state_dict`` 只能在参数被展平并且分片时使用。"
        )
    # 调用通用的预状态字典钩子
    _common_pre_state_dict_hook(module, fsdp_state)


@no_type_check
def _local_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    此钩子从本地 flat_param 创建 ShardedTensor，并用 ShardedTensor 替换
    state_dict[f"{prefix}{FLAT_PARAM}]。不会发生复制。底层存储保持不变。
    """
    
    # 使用前缀替换 state_dict 中的条目，确保与 FSDP_PREFIX 开头的键对应的条目被正确替换
    _replace_by_prefix(state_dict, f"{prefix}{FSDP_PREFIX}", prefix)
    # 如果模块不包含 FSDP 参数，则直接返回 state_dict
    if not _has_fsdp_params(fsdp_state, module):
        return state_dict

    # 如果 state_dict[f"{prefix}{FLAT_PARAM}"] 存在且与 flat_param 有相同的张量值，
    # 但它是一个纯张量，因为 nn.Module.state_dict() 会分离参数。
    # 因此，我们需要获取 flat_param 来获取元数据。
    assert _module_handle(fsdp_state, module), "应该已经提前返回"
    flat_param = _module_handle(fsdp_state, module).flat_param
    
    # 从 flat_param 构造一个 ShardedTensor，"不带"填充。
    # 去除填充允许用户在加载本地状态字典时更改排名数量。
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * fsdp_state.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    
    # 如果有效数据大小大于 0
    if valid_data_size > 0:
        # 如果返回 FlatParameter，则 FlatParameter._local_shard 会引发
        # 一个序列化问题（可以使用 torch.save，但不能使用 torch.load）。
        # 由于 state_dict 返回实际 FlatParameter 类别没有好处，
        # 将返回 FlatParameter 的视图（即一个张量）。
        flat_param = flat_param[:valid_data_size].view(valid_data_size)
        local_shards = [
            Shard.from_tensor_and_offsets(flat_param, [shard_offset], fsdp_state.rank)
        ]
    else:
        local_shards = []
    
    # 使用本地分片初始化 ShardedTensor
    sharded_tensor = init_from_local_shards(
        local_shards, full_numel, process_group=fsdp_state.process_group
    )
    )  # type: ignore[assignment]
    # 注释：忽略此处的类型检查警告，标记为对赋值操作的忽略

    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    # 注释：在这里添加对于 LOCAL_STATE_DICT 的 DTensor 状态字典支持的 TODO 提示

    if fsdp_state._state_dict_config.offload_to_cpu:
        # 注释：如果需要将数据转移到 CPU 上，则执行以下操作
        sharded_tensor = sharded_tensor.cpu()

    state_dict[f"{prefix}{FLAT_PARAM}"] = sharded_tensor
    # 注释：将分片张量 sharded_tensor 存入状态字典 state_dict 中，键为 prefix + FLAT_PARAM

    return state_dict
    # 注释：返回最终的状态字典
# 定义一个私有函数，用于在加载模型状态字典前执行特定操作，用于 FSDP 模块
def _local_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    # 此函数暂时为空，用于后续实现加载后的操作
    pass


# 定义一个私有函数，用于在加载模型状态字典前执行特定操作，用于 FSDP 模块
def _local_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    # 初始化 FSDP 状态和模块
    _lazy_init(fsdp_state, module)
    # 替换 state_dict 中的参数名前缀
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
    # 构建 flat_param 的全名
    fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
    # 如果 state_dict 中不存在 flat_param，则返回
    if fqn not in state_dict:
        assert not _has_fsdp_params(fsdp_state, module), (
            "No `FlatParameter` in `state_dict` for this FSDP instance "
            "but it has parameters"
        )
        return
    # 从 state_dict 中加载 flat_param，并验证其类型为 ShardedTensor
    load_tensor = state_dict[fqn]
    assert isinstance(
        load_tensor, ShardedTensor
    ), "Tensors in local_state_dict should be ShardedTensor."

    # 将 ShardedTensor 转换为 Tensor
    flat_param = _module_handle(fsdp_state, module).flat_param
    assert flat_param is not None
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    shards = load_tensor.local_shards()
    if valid_data_size > 0:
        assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
        # 获取第一个 shard 的 tensor
        load_tensor = shards[0].tensor

        # 根据 flat_param 的元数据决定是否需要对加载的 tensor 进行填充
        if flat_param._shard_numel_padded > 0:
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in "
                f"the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    else:
        # 如果没有有效数据大小，则直接使用 flat_param
        load_tensor = flat_param
    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    # 将处理后的 tensor 存入 state_dict
    state_dict[fqn] = load_tensor


# 定义一个私有函数，用于在获取模型状态字典前执行特定操作，用于 FSDP 模块
def _sharded_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
    # 如果模块具有 FSDP 参数但未使用分片策略，则引发运行时错误
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``sharded_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    # 执行通用的模块预状态字典钩子
    _common_pre_state_dict_hook(module, fsdp_state)
    # 在此处设置 offload_to_cpu 不起作用，即使 offload_to_cpu 为 True。
    # 必须先创建 ShardedTensor，然后将其移动到 CPU 上。
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=False,
        rank0_only=False,
    )


# 定义一个不进行类型检查的函数，用于在获取模型状态字典后执行特定操作，用于 FSDP 模块
@no_type_check
def _sharded_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,  # 声明一个变量 fsdp_state，类型为 _FSDPState
    state_dict: Dict[str, Any],  # 声明一个变量 state_dict，类型为字典，键为字符串，值可以是任意类型
    prefix: str,  # 声明一个变量 prefix，类型为字符串
# 定义一个钩子函数，用于处理加载状态字典后的操作，将参数映射为分片张量
def _sharded_post_load_state_dict_hook(
    module: nn.Module,  # 输入参数：神经网络模块
    fsdp_state: _FSDPState,  # 输入参数：FSDP 状态对象
    *args,  # 其他位置参数
    **kwargs  # 其他关键字参数
) -> None:  # 返回类型：无返回值
    # 如果模块包含 FSDP 参数
    if _has_fsdp_params(fsdp_state, module):
        # 使用简单性能分析器记录退出未分片参数上下文
        with SimpleProfiler.profile("_exit_unshard_params_ctx"):
            _exit_unshard_params_ctx(module, fsdp_state)


# 带有类型检查的函数装饰器，定义一个加载状态字典前的钩子函数
@no_type_check
def _sharded_pre_load_state_dict_hook(
    module: nn.Module,  # 输入参数：神经网络模块
    fsdp_state: _FSDPState,  # 输入参数：FSDP 状态对象
    state_dict: Dict[str, Any],  # 输入参数：状态字典
    prefix: str,  # 输入参数：前缀
) -> None:  # 返回类型：无返回值
    """
    钩子函数用于组合未分片的分片参数（ShardedTensor）为新的平坦参数，并将新的平坦参数分片到本地块。
    """
    # 惰性初始化 FSDP 状态
    _lazy_init(fsdp_state, module)
    # 如果 FSDP 不可组合，则替换前缀
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")
    # 如果模块不包含 FSDP 参数，则返回
    if not _has_fsdp_params(fsdp_state, module):
        return

    # 获取模块的处理句柄
    handle = _module_handle(fsdp_state, module)
    # 如果句柄未使用分片策略，则抛出运行时错误
    if not handle.uses_sharded_strategy:
        raise RuntimeError(
            "load_sharded_state_dict can only be called when parameters "
            "are flattened and sharded."
        )
    
    # 将平坦参数的全限定名与参数扩展映射成字典
    fqn_to_param_ext = dict(
        zip(handle.flat_param._fqns, handle.flat_param._param_extensions)
    )

    # 使用简单性能分析器记录进入未分片参数上下文
    with SimpleProfiler.profile("_enter_unshard_params_ctx"):
        _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


# 上下文管理器函数，替换为完整状态字典类型
@contextlib.contextmanager
def _replace_with_full_state_dict_type(fsdp_state: _FSDPState) -> Generator:
    # 保存旧的状态字典配置和类型
    old_state_dict_config = fsdp_state._state_dict_config
    old_state_dict_type = fsdp_state._state_dict_type
    # 设置新的状态字典配置和类型为完整状态字典
    fsdp_state._state_dict_config = FullStateDictConfig()
    fsdp_state._state_dict_type = StateDictType.FULL_STATE_DICT
    yield  # 执行代码块
    # 恢复旧的状态字典配置
    fsdp_state._state_dict_config = old_state_dict_config
    # 设置 fsdp_state 对象的 _state_dict_type 属性为 old_state_dict_type
    fsdp_state._state_dict_type = old_state_dict_type
@no_type_check
@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``fsdp_state._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    # 获取当前模块的FSDP状态
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    
    # 根据分片策略决定上下文处理方式
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        # 替换为完整的状态字典类型
        context = _replace_with_full_state_dict_type(fsdp_state)
        # 发出警告，当使用NO_SHARD时，将返回完整的状态字典
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        # 设置使用DTensor
        _set_use_dtensor(fsdp_state)
        # 使用空上下文
        context = contextlib.nullcontext()

    # 使用上下文管理器处理状态字典后的钩子函数
    with context:
        # 定义不同状态字典类型对应的后处理钩子函数
        _post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
        }
        # 调用相应状态字典类型的后处理钩子函数，处理状态字典
        processed_state_dict = _post_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )

    # 如果是根节点，则记录日志信息
    if fsdp_state._is_root:
        logger.info("FSDP finished processing state_dict(), prefix=%s", prefix)
        # 遍历已处理的状态字典项，输出相应的信息
        for key, tensor in sorted(processed_state_dict.items()):
            if key.startswith(prefix) and isinstance(tensor, torch.Tensor):
                local_shape = tensor.shape
                # 对于ShardedTensor，获取本地形状
                if isinstance(tensor, ShardedTensor):
                    local_shape = None
                    shards = tensor.local_shards()
                    if shards:
                        local_shape = shards[0].tensor.shape
                # 对于DTensor，获取本地形状
                elif isinstance(tensor, DTensor):
                    local_shape = tensor.to_local().shape
                # 记录详细信息到日志中
                logger.info(
                    "FQN=%s: type=%s, shape=%s, local_shape=%s, dtype=%s, device=%s",
                    key,
                    type(tensor),
                    tensor.shape,
                    local_shape,
                    tensor.dtype,
                    tensor.device,
                )

    # 返回处理后的状态字典
    return processed_state_dict


@no_type_check
@torch.no_grad()
def _pre_state_dict_hook(
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    This is called before the core state dict saving logic of ``module``.
    ``fsdp_state._state_dict_type`` is used to decide what postprocessing will
    be done.
    """
    # 获取当前模块的FSDP状态
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    
    # 根据分片策略决定上下文处理方式
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        # 替换为完整的状态字典类型
        context = _replace_with_full_state_dict_type(fsdp_state)
        # 发出警告，当使用NO_SHARD时，将返回完整的状态字典
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        # 设置使用DTensor
        _set_use_dtensor(fsdp_state)
        # 使用空上下文
        context = contextlib.nullcontext()
    # 使用上下文管理器 `context` 执行以下代码块
    with context:
        # 定义一个字典 `_pre_state_dict_hook_fn`，映射不同的 `StateDictType` 到相应的预处理钩子函数
        _pre_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_state_dict_hook,
        }
        # 调用 `_pre_state_dict_hook_fn` 字典中与 `fsdp_state._state_dict_type` 关联的预处理钩子函数
        _pre_state_dict_hook_fn[fsdp_state._state_dict_type](
            fsdp_state,   # 第一个参数是 `fsdp_state`
            module,       # 第二个参数是 `module`
            *args,        # 其余位置参数由 `*args` 传递
            **kwargs,     # 其余关键字参数由 `**kwargs` 传递
        )
# 设置函数装饰器，禁用类型检查
@no_type_check
# 设置函数 _set_use_dtensor，接受一个 _FSDPState 类型参数 fsdp_state，返回 None
def _set_use_dtensor(fsdp_state: _FSDPState) -> None:
    # 如果在初始化 FSDP 时传入了 device_mesh 参数，则自动将 _use_dtensor 标志设置为 True，
    # 这适用于 ShardedStateDictConfig()。
    if getattr(fsdp_state, "_device_mesh", None):
        # 获取 fsdp_state 的 state_dict_type 属性
        state_dict_type = fsdp_state._state_dict_type
        # 如果 state_dict_type 是 LOCAL_STATE_DICT，则抛出 RuntimeError 异常
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError(
                "Found state_dict_type LOCAL_STATE_DICT",
                "DeviceMesh is not compatible with LOCAL_STATE_DICT.",
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.",
            )
        else:
            # 否则将 fsdp_state 的 _use_dtensor 设置为 True
            fsdp_state._state_dict_config._use_dtensor = True


# 设置函数装饰器，禁用类型检查，并且使用 torch.no_grad 上下文
@no_type_check
@torch.no_grad()
# 设置函数 _pre_load_state_dict_hook，接受 nn.Module、Dict[str, Any]、str 和可变参数作为输入，返回 None
def _pre_load_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    This is called before ``module._load_from_state_dict()``.
    ``fsdp_state._state_dict_type`` is used to decide what preprocessing will
    be done.
    """
    # 获取 module 的 FSDP 状态，如果是全分片模块，则返回 fsdp_state，否则返回 None
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    # 如果 sharding_strategy 是 NO_SHARD，则将 fsdp_state 替换为全状态字典类型
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        # 发出警告信息，说明在使用 NO_SHARD 的 ShardingStrategy 时，会返回完整的 state_dict。
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        # 否则调用 _set_use_dtensor 函数，设置 _use_dtensor 标志
        _set_use_dtensor(fsdp_state)
        # 使用 nullcontext() 上下文管理器
        context = contextlib.nullcontext()

    # 惰性初始化 fsdp_state 和 module
    _lazy_init(fsdp_state, module)
    # 如果 fsdp_state 是根节点，则重置 SimpleProfiler
    if fsdp_state._is_root:
        SimpleProfiler.reset()

    # 使用 context 上下文进行下面的操作
    with context:
        # 定义 _pre_load_state_dict_hook_fn 字典，根据 state_dict 类型选择对应的预处理函数
        _pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
        }
        # 对于所有 state_dict 实现都通用的代码
        # 如果 fsdp_state 的 _device_handle 可用，则同步设备
        if fsdp_state._device_handle.is_available():
            fsdp_state._device_handle.synchronize()
        # 调度到特定 state_dict 实现的预加载钩子
        _pre_load_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )


# 设置函数装饰器，禁用类型检查，并且使用 torch.no_grad 上下文
@no_type_check
@torch.no_grad()
# 设置函数 _post_load_state_dict_hook，接受 nn.Module、Tuple[List[str], List[str]] 和可变参数作为输入，返回 None
def _post_load_state_dict_hook(
    module: nn.Module,
    incompatible_keys: Tuple[List[str], List[str]],
    *args: Any,
) -> None:
    # 获取 module 的 FSDP 状态，如果是全分片模块，则返回 fsdp_state，否则返回 None
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    # 如果 sharding_strategy 是 NO_SHARD，则将 fsdp_state 替换为全状态字典类型
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        # 发出警告信息，说明在使用 NO_SHARD 的 ShardingStrategy 时，会返回完整的 state_dict。
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        # 否则使用 nullcontext() 上下文管理器
        context = contextlib.nullcontext()
    # 在上下文环境中执行以下代码块
    with context:
        # 定义一个字典，将不同的 StateDictType 映射到对应的后加载状态字典钩子函数
        _post_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
        }
        # 通用于所有 state_dict 实现的代码
        # 调度到特定类型的 state_dict 实现的后加载钩子函数
        _post_load_state_dict_hook_fn[fsdp_state._state_dict_type](module, fsdp_state)

    # 当报告不兼容的键时，修剪掉 FSDP 前缀
    missing_keys = incompatible_keys[0]
    unexpected_keys = incompatible_keys[1]

    # 清理缺失键中的张量名称
    for i in range(len(missing_keys)):
        missing_keys[i] = clean_tensor_name(missing_keys[i])

    # 清理意外键中的张量名称
    for i in range(len(unexpected_keys)):
        unexpected_keys[i] = clean_tensor_name(unexpected_keys[i])

    # 如果当前节点是根节点
    if fsdp_state._is_root:
        # 简单分析器：输出并重置 "FSDP 模型 load_state_dict 时的性能分析"
        SimpleProfiler.dump_and_reset("FSDP model load_state_dict profiling: ")
# 注册所有状态字典钩子函数，包括预保存、后保存、预加载和后加载状态字典钩子。
def _register_all_state_dict_hooks(state: _FSDPState):
    """
    Registers pre-save, post-save, pre-load, and post-load state dict hooks.
    """
    # 遍历四个钩子注册函数的元组列表
    for hook_registration_fn_str, hook, hook_registration_fn_kwargs in (
        ("register_state_dict_pre_hook", _pre_state_dict_hook, {}),  # 注册预保存状态字典钩子函数
        ("_register_state_dict_hook", _post_state_dict_hook, {}),    # 注册后保存状态字典钩子函数
        (
            "_register_load_state_dict_pre_hook",
            _pre_load_state_dict_hook,
            {"with_module": True},  # 注册预加载状态字典钩子函数，并带有模块参数
        ),
        ("register_load_state_dict_post_hook", _post_load_state_dict_hook, {}),  # 注册后加载状态字典钩子函数
    ):
        # 调用基础的状态字典钩子注册函数来注册具体的钩子
        _register_state_dict_hooks_base(
            state, hook_registration_fn_str, hook, hook_registration_fn_kwargs
        )


@no_type_check
def _register_state_dict_hooks_base(
    state: _FSDPState,
    hook_registration_fn_name: str,
    hook: Callable,
    hook_registration_fn_kwargs: Dict[str, Any],
) -> None:
    """Registers ``hook`` using ``hook_registration_fn``."""
    # 如果状态对象不可组合化，则直接调用其状态字典钩子注册函数
    if not _is_composable(state):
        getattr(state, hook_registration_fn_name)(hook, **hook_registration_fn_kwargs)
    else:
        # 否则，获取状态对象的处理句柄，调用其完全分片模块的状态字典钩子注册函数
        handle = state._handle
        if handle:
            getattr(handle._fully_sharded_module, hook_registration_fn_name)(
                hook, **hook_registration_fn_kwargs
            )
```