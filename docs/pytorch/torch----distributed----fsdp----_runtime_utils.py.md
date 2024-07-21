# `.\pytorch\torch\distributed\fsdp\_runtime_utils.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 库，用于高阶函数操作
import functools
# 导入 logging 库，用于记录日志信息
import logging
# 导入枚举类型相关的 auto 函数和 Enum 类
from enum import auto, Enum
# 导入类型提示相关的 Any, Callable, Dict, List, Optional, Set, Tuple
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple

# 导入 PyTorch 库
import torch
# 导入分布式相关的模块
import torch.distributed as dist
# 导入 FSDP 的遍历工具模块
import torch.distributed.fsdp._traversal_utils as traversal_utils
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络函数模块
import torch.nn.functional as F
# 导入自动求导变量模块
from torch.autograd import Variable
# 导入多重梯度钩子注册模块
from torch.autograd.graph import register_multi_grad_hook
# 导入低精度钩子模块
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
# 导入 FSDP 的通用工具模块
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _FSDPState,
    _get_module_fsdp_state,
    _is_composable,
    _log_post_backward_hook,
    _no_dispatch_record_stream,
    clean_tensor_name,
    TrainingState,
)
# 导入扁平化参数模块
from torch.distributed.fsdp._flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
# 导入 FSDP 的初始化工具模块
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
# 导入反向预取模块
from torch.distributed.fsdp.api import BackwardPrefetch
# 导入分布式工具模块
from torch.distributed.utils import (
    _apply_to_tensors,
    _cast_forward_inputs,
    _p_assert,
    _to_kwargs,
)
# 导入 PyTree 模块
from torch.utils import _pytree as pytree

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义用于启用混合分片和 MoE 情况的属性名称列表，不包括 "process_group"
HOMOGENEOUS_ATTR_NAMES = (
    "_use_orig_params",
    "limit_all_gathers",
    "_use_full_prec_in_eval",
)


class _PrefetchMode(Enum):
    # 定义枚举类型 _PrefetchMode，包含 BACKWARD 和 FORWARD 两个值
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(
    module: nn.Module,
) -> Tuple[List[_FSDPState], List[nn.Module]]:
    """
    返回一个元组，包含以下两部分：
    1. 在以 module 为根的模块树中，所有不重复的顶级 _FSDPState 实例列表，遵循 module.modules() 的遍历顺序（假定为深度优先）。
    2. 对应的顶级模块列表，这些模块拥有第一部分中的状态。

    与 _get_fsdp_states_with_modules 类似，但是必须调用 _is_fsdp_root 来强制延迟初始化以确定 FSDP 根（如果延迟初始化尚未发生）。
    """
    # 存储 FSDP 根状态的列表
    fsdp_root_states: List[_FSDPState] = []
    # 存储拥有 FSDP 根状态的顶级模块列表
    fsdp_root_modules: List[nn.Module] = []
    # 记录已访问的 FSDP 状态，避免重复添加
    visited_fsdp_states: Set[_FSDPState] = set()
    # 注意：此函数假定 `module.modules()` 从上到下进行。
    for submodule in module.modules():
        # 获取子模块的 FSDP 状态
        optional_state = _get_module_fsdp_state(submodule)
        # 如果子模块具有 FSDP 状态且尚未访问过，并且它是 FSDP 根，则添加到相应列表中
        if (
            optional_state is not None
            and optional_state not in visited_fsdp_states
            and _is_fsdp_root(optional_state, submodule)
        ):
            visited_fsdp_states.add(optional_state)
            fsdp_root_states.append(optional_state)
            fsdp_root_modules.append(submodule)
    return fsdp_root_states, fsdp_root_modules


def _get_fsdp_root_states(module: nn.Module) -> List[_FSDPState]:
    # 返回一个列表，包含给定模块中所有顶级 FSDP 状态
    pass  # Placeholder, function implementation needed
    # 调用函数 _get_fsdp_root_states_with_modules，并返回其返回值中的第一个元素 fsdp_root_states
    """See :func:`_get_fsdp_root_states_with_modules`."""
    fsdp_root_states, _ = _get_fsdp_root_states_with_modules(module)
    # 返回获取到的 fsdp_root_states 变量作为结果
    return fsdp_root_states
# 判断给定的状态是否对应于 FSDP 的根节点
def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    """
    Returns if ``state`` corresponds to that of an FSDP root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
    # 强制进行懒初始化以确定是否为 FSDP 的根节点
    _lazy_init(state, module)
    # 断言确保 `_is_root` 不为 `None`，仅用于类型检查
    assert state._is_root is not None  # mypy
    # 返回 `_is_root` 的值，判断是否为 FSDP 的根节点
    return state._is_root


@no_type_check
def _lazy_init(
    state: _FSDPState,
    root_module: nn.Module,
) -> _FSDPState:
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the FSDP instance itself.
    """
    # 如果 `_is_root` 已经不为 `None`，则无需操作，已经进行了懒初始化
    if state._is_root is not None:
        return  # no-op: already lazily initialized
    # 如果当前设备不可用，抛出运行时错误
    if not state._device_handle.is_available():
        # 允许 FSDP 构造函数在没有 CUDA 的情况下运行，但实际执行时会检查
        raise RuntimeError("FSDP does not support CPU only execution")
    # 以下逻辑仅在根 FSDP 实例上运行，因为它将为非根 FSDP 实例设置 `_is_root=False`
    state._is_root = True
    # 断言以确保状态处于训练中的某个空闲状态
    _assert_in_training_states(state, [TrainingState.IDLE])
    # 检查所有 `FlatParameter` 是否在预期设备上，用于懒初始化
    _check_flat_params_on_expected_device(state, root_module)
    # 获取根模块下所有 FSDP 状态
    state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    # 初始化流对象
    _init_streams(state)
    # 获取计算所需的缓冲区和缓冲区数据类型
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(state, root_module)
    # 将缓冲区转换为指定的数据类型和设备
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    # 初始化执行顺序数据
    state._exec_order_data.init(state, root_module, state.process_group)
    # 共享状态并初始化处理属性
    _share_state_and_init_handle_attrs(state, root_module)
    # 返回更新后的状态
    return state


def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
    # 定义 CPU 设备
    cpu_device = torch.device("cpu")
    # 遍历从 traversal_utils 模块中获取的所有 FSDP 句柄（handle）
    for handle in traversal_utils._get_fsdp_handles(module):
        # 检查句柄是否没有离载参数，并且参数所在设备不是当前计算设备
        if (
            not handle._offload_params
            and handle.flat_param.device != state.compute_device
        ):
            # 抛出运行时错误，指示 FSDP 管理的模块意外地具有在指定设备上的参数
            raise RuntimeError(
                "An FSDP-managed module unexpectedly has parameters on "
                f"{handle.flat_param.device}. Make sure to move the module to "
                f"{state.compute_device} before training."
            )
        # 如果句柄启用了离载参数，并且参数所在设备不是 CPU 设备
        elif handle._offload_params and handle.flat_param.device != cpu_device:
            # 抛出运行时错误，指示启用了 CPU 离载的 FSDP 管理的模块具有在非 CPU 设备上的参数
            raise RuntimeError(
                "An FSDP-managed module with parameter CPU offloading enabled "
                f"has parameters on {handle.flat_param.device}. Make sure to "
                f"not move the module from CPU when offloading parameters."
            )
# 声明一个装饰器，用于在类型检查时跳过该函数
@no_type_check
# 定义函数，用于在所有的 FSDP 状态中共享数据结构状态，并初始化处理属性
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,  # 根状态对象，包含要共享的数据结构状态
    root_module: nn.Module,  # 根模块，需要在其模块树中共享状态
) -> None:  # 函数没有返回值
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    # 获取根状态对象的句柄（handle）
    handle = root_state._handle
    # 如果句柄存在，则初始化平坦参数属性
    if handle:
        handle.init_flat_param_attributes()
    # 创建一个空字典，用于存储属性名到值集合的映射
    attr_name_to_values: Dict[str, Set[Any]] = {}
    # 遍历预定义的属性名列表，为每个属性名创建一个空集合
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    # 将根状态对象的所有句柄引用共享给_root_state._all_handles
    root_state._all_handles = root_state._exec_order_data.all_handles  # 共享引用
    # 更新每个句柄的_backward 优化状态
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        # 如果 flat_param 具有 _in_backward_optimizers 属性，则抛出运行时错误
        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError(
                "FSDP optimizer in backward only supported with use_orig_params=True!"
            )
        # 检查并设置句柄的 _has_optim_in_backward 属性
        handle._has_optim_in_backward = (
            flat_param._params is not None and
            any(hasattr(param, "_in_backward_optimizers") for param in flat_param._params)
        )
        # 如果句柄具有优化器在反向传播中的属性，则记录使用情况
        if handle._has_optim_in_backward:
            torch._C._log_api_usage_once("fsdp.optimizer_in_backward")
    # 遍历根状态对象的所有 FSDP 状态
    for fsdp_state in root_state._all_fsdp_states:
        # 遍历预定义的属性名列表，确保每个 FSDP 状态对象具有这些属性
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            # 将属性值添加到对应属性名的集合中
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        # 如果当前状态对象是根状态，则跳过
        if fsdp_state is root_state:
            continue
        # 放宽断言，对于非根 FSDP 实例，允许其嵌套初始化模块在后续被再次包装（例如，在训练后进行推理时）
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been "
            "set yet or should have been set to `False`",
        )
        # 设置非根 FSDP 实例的 _is_root 属性为 False
        fsdp_state._is_root = False
        # 共享流属性给非根 FSDP 实例
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        # 如果 FSDP 扩展不为空，则将其计算流设置为默认流
        if fsdp_state._fsdp_extension is not None:
            fsdp_state._fsdp_extension.compute_stream = root_state._default_stream
        # 获取当前状态对象的句柄，并初始化其平坦参数属性
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
    # 遍历字典 attr_name_to_values，获取属性名 attr_name 和对应的值列表 attr_values
    for attr_name, attr_values in attr_name_to_values.items():
        # 检查属性值列表的长度是否不为1，如果不为1则抛出数值错误异常
        if len(attr_values) != 1:
            raise ValueError(
                f"Expects one homogeneous value for {attr_name} but got {attr_values}"
            )
@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    # 确保当前实例是根节点
    assert state._is_root
    # 确保设备句柄可用
    assert state._device_handle.is_available()
    # 检查是否使用混合分片策略
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    # 对于使用混合分片策略且限制全收集操作，设置较高优先级
    # 否则保持默认优先级为0
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # 设置默认计算流
    state._default_stream = state._device_handle.current_stream()
    if state._fsdp_extension is not None:
        # 将计算流设置为FSDP扩展流
        state._fsdp_extension.compute_stream = state._default_stream

    # 流用于未分片逻辑，包括分配全收集目标张量和全收集本身
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # 流用于与后向传播梯度计算重叠的梯度减少
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # 流用于预未分片逻辑，包括CPU卸载（H2D复制）和混合精度（低精度转换）的分配和写入
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # 流用于作为异步运行HSDP的全收集（如果使用HSDP）
    state._all_reduce_stream = (
        state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    )


@no_type_check
def _unshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle:
        return
    # 在pre_unshard_stream流中执行预未分片操作
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    # 如果运行了预未分片操作，等待pre_unshard_stream流完成
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    # 如果限制了所有收集操作
    if state.limit_all_gathers:
        # 从事件队列中出队一个事件（如果需要），并等待其完成
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function(
                "FullyShardedDataParallel.rate_limiter"
            ):
                event.synchronize()
    # 在unshard_stream流中执行未分片操作
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()


@no_type_check
def _reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    free_unsharded_flat_param: bool,
):
    """
    """
    # 对句柄进行重新分片。`free_unsharded_flat_param` 表示是否释放句柄的填充未分片的平坦参数。
    handle.reshard(free_unsharded_flat_param)
    # 如果 `state.limit_all_gathers` 为真且 `free_unsharded_flat_param` 为真
    if state.limit_all_gathers and free_unsharded_flat_param:
        # 如果不是在 TorchDynamo 编译过程中
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # 我们当前不为释放动作运行偶数队列
            # 但也许我们需要？TODO(voz): 进一步研究此问题
            # 创建一个事件用于释放操作
            free_event = state._device_handle.Event()
            # 记录事件的发生
            free_event.record()
            # 将事件加入到释放事件队列中
            state._free_event_queue.enqueue(free_event)
    # 完成句柄的重新分片后的后续处理
    handle.post_reshard()
    # 无论平坦参数是否被释放，我们始终需要在下次访问时"取消分片"参数
    # 以确保其形状正确
    handle._prefetched = False
# 如果给定有效的参数句柄，则调用其 unshard_grad 方法，用于取消分片梯度
def _unshard_grads(
    handle: Optional[FlatParamHandle],
) -> None:
    if handle:
        handle.unshard_grad()

# 如果给定有效的参数句柄，则调用其 reshard_grad 方法，用于重新分片梯度
def _reshard_grads(
    handle: Optional[FlatParamHandle],
) -> None:
    if handle:
        handle.reshard_grad()

# 这是一个装饰器函数，用于禁用类型检查
@no_type_check
# 执行前向逻辑的函数，包括取消当前分片的参数以及注册后向钩子
# 这个函数还将前向传播的参数 args 和 kwargs 转换为给定的精度
def _pre_forward(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    unshard_fn: Callable,
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
        # 使用 Torch Profiler 记录函数执行时间，标记为 "FullyShardedDataParallel._pre_forward"

        # 对于 `fully_shard` + `checkpoint`，在后向预处理阶段，跳过前向逻辑
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # 如果存在句柄并且处于后向预处理状态，则直接返回输入参数
            return args, kwargs
        
        # 设置训练状态为前向后向
        state.training_state = TrainingState.FORWARD_BACKWARD
        
        # 记录前向操作，包括句柄和模块的训练状态
        state._exec_order_data.record_pre_forward(handle, module.training)
        
        # 如果存在句柄，将其训练状态设置为前向
        if handle:
            handle._training_state = HandleTrainingState.FORWARD
        
        # 如果存在非分片函数，应用该函数到状态和句柄上
        if unshard_fn is not None:
            unshard_fn(state, handle)
        
        # 注册后向钩子函数以重新分片参数并执行梯度的 reduce-scatter 操作
        _register_post_backward_hook(state, handle)
        
        # 如果存在句柄，并且设置了参数卸载，并且 flat_param 的 CPU 梯度为 None，则重新分配 _cpu_grad
        if handle and handle._offload_params and handle.flat_param._cpu_grad is None:
            handle.flat_param._cpu_grad = torch.zeros_like(
                handle.flat_param._local_shard, device=torch.device("cpu")
            ).pin_memory(device=state.compute_device)
        
        # 判断是否需要将前向输入转换为指定精度，根据句柄和混合精度设置
        should_cast_forward_inputs = (
            state._handle and not state._handle._force_full_precision
        )
        
        # 如果需要转换前向输入，并且设置了混合精度的前向输入转换
        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # 递归地将 args 和 kwargs 转换为指定精度
            input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        
        # 注册仅在后向重分片时执行的后向钩子函数
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        
        # 返回处理后的 args 和 kwargs
        return args, kwargs
@no_type_check
def _pre_forward_unshard(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    """在前向传播之前对参数进行去分片处理。"""
    # 如果 handle 不存在，直接返回
    if not handle:
        return
    # 如果参数已经预获取，无需再次调用 `_unshard()`
    if not handle._prefetched:
        _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    # 标记 handle 不再需要前向传播前的去分片处理
    handle._needs_pre_forward_unshard = False
    # 在追踪过程中不进行等待
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        # 获取当前设备上的流对象
        current_stream = state._device_handle.current_stream()
        # 如果存在未分片事件，等待事件完成并将事件置空
        if state._unshard_event is not None:
            current_stream.wait_event(state._unshard_event)
            state._unshard_event = None
        else:
            # 否则等待未分片流完成
            current_stream.wait_stream(state._unshard_stream)
    # 使用 Torch Profiler 记录函数执行时间
    with torch.profiler.record_function(
        "FullyShardedDataParallel._pre_forward_prefetch"
    ):
        # 预取 handle 数据，用于前向传播
        _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


@no_type_check
def _post_forward(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    """
    运行后向传播逻辑。包括对当前未分片参数（如当前前向传播中使用的参数）进行去分片处理的机会，
    并在前向传播输出上注册前向反向钩子。

    Args:
        handles (List[FlatParamHandle]): 当前前向传播中使用的参数的句柄。
        reshard_fn (Optional[Callable]): 用于对当前未分片参数进行去分片（例如从当前前向传播中）的可调用函数，
            或者为 ``None`` 表示不执行任何去分片。
        module (nn.Module): 刚刚运行前向传播的模块，应该是一个完全分片的模块（参见 [Note: Fully Sharded Module]）；由钩子签名预期。
        input (Any): 未使用；由钩子签名预期。
        output (Any): 前向传播输出；在这个输出上注册需要梯度的张量的前向反向钩子。

    Postcondition: 每个 ``FlatParameter`` 的数据指向分片后的扁平参数。
    """
    with torch.profiler.record_function("FullyShardedDataParallel._post_forward"):
        # 使用 torch.profiler.record_function 进行性能分析，记录函数 "FullyShardedDataParallel._post_forward"

        # 对于 `fully_shard` + `checkpoint`，在重新计算的前向传播中，跳过后向传播后处理逻辑
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # 如果存在 handle 并且其训练状态为 BACKWARD_PRE，则直接返回输出结果
            return output

        # 记录执行顺序中的后向传播后处理
        state._exec_order_data.record_post_forward(handle)

        # 如果存在 reshard_fn 函数，则调用它来重新分片状态和 handle
        if reshard_fn is not None:
            reshard_fn(state, handle)

        # 注册预后向传播钩子，用于对梯度计算时展开的参数进行取消分片（如果需要的话）
        output = _register_pre_backward_hooks(state, module, output, handle)

        # 将状态设置为 IDLE，表示训练状态处于空闲
        state.training_state = TrainingState.IDLE

        # 如果存在 handle，则将其训练状态设置为 IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE

        # 返回处理后的输出
        return output
@no_type_check
def _post_forward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> None:
    """Reshards parameters in the post-forward."""
    # 如果 handle 不存在，则直接返回，不执行后续逻辑
    if not handle:
        return
    
    # 如果不是根节点，并且使用的分片策略在 RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES 中，
    # 则不释放根节点的参数，在后向计算之前保持参数可用性（尽管这可能并不总是正确的）
    free_unsharded_flat_param = (
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )
    
    # 调用 _reshard 函数来对参数进行重分片操作
    _reshard(state, handle, free_unsharded_flat_param)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    

@no_type_check
def _root_cast_forward_input(
    state: _FSDPState, module: torch.nn.Module, args, kwargs
) -> Tuple[Any, Any]:
    # 如果状态中有 _handle，则根据 _handle 的状态设置 force_full_precision 变量
    if state._handle:
        force_full_precision = not state._handle._force_full_precision
    else:
        force_full_precision = True

    # 计算是否需要对前向输入进行类型转换的标志
    should_cast_forward_inputs = (
        (module.training or not state._use_full_prec_in_eval) and force_full_precision
    ) and state.mixed_precision.cast_root_forward_inputs

    # 如果需要对前向输入进行类型转换，则根据状态中的 mixed_precision.param_dtype 进行转换
    if should_cast_forward_inputs:
        input_dtype: Optional[torch.dtype] = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)

    return args, kwargs


@no_type_check
def _pre_backward_hook(
    state: _FSDPState,
    module: nn.Module,
    handle: FlatParamHandle,
    grad,
    *unused: Any,
) -> Any:
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
    # 只有在每组参与相同模块前向计算的 handle 上第一次运行前向计算之前的钩子
    if (
        handle
        and hasattr(handle, "_ran_pre_backward_hook")
        and handle._ran_pre_backward_hook
    ):
        # 如果已经运行过 pre-backward hook，则记录调试信息并直接返回 grad
        logger.debug("%s %s", id(state), "Not Running pre backward! Already Ran!")
        return grad
    # 使用 Torch Profiler 记录函数执行时间，标签为 "FullyShardedDataParallel._pre_backward_hook"
    with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
        # 如果是根节点并且尚未排队后向回调，则将后向回调排队
        # 以便将其附加到最外层的后向图任务，确保在所有后向调用完成后执行
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        # 如果存在 handle，则检查状态并确保在允许的训练状态下
        elif handle:
            allowed_states = [TrainingState.IDLE]
            # 如果是可组合的状态，还允许 FORWARD_BACKWARD 状态
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        # 设置当前训练状态为 FORWARD_BACKWARD
        state.training_state = TrainingState.FORWARD_BACKWARD
        # 如果没有 handle，则直接返回梯度 grad
        # 这是 pre-backward 钩子中唯一不依赖 handle 的逻辑
        if not handle:
            return grad
        # 将 handle 的训练状态设置为 BACKWARD_PRE
        handle._training_state = HandleTrainingState.BACKWARD_PRE

        # 如果需要进行 pre-backward 的反解离操作
        if handle._needs_pre_backward_unshard:
            # 如果已经预提取，则无需再次调用 `_unshard()`
            if not handle._prefetched:
                _unshard(
                    state,
                    handle,
                    state._unshard_stream,
                    state._pre_unshard_stream,
                )
            # 在非 TorchDynamo 编译状态下，等待 `_unshard_stream` 完成
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                state._device_handle.current_stream().wait_stream(state._unshard_stream)

        # 将 `_needs_pre_backward_unshard` 设置为 `False`，确保误预提取不会实际反解离这些 handle
        handle._needs_pre_backward_unshard = False
        # 使用 Torch Profiler 记录函数执行时间，标签为 "FullyShardedDataParallel._pre_backward_prefetch"
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_backward_prefetch"
        ):
            # 针对后向操作预提取 handle 的数据
            _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
        # 为后向传播准备梯度
        handle.prepare_gradient_for_backward()
        # 标记已经运行过 pre-backward 钩子
        handle._ran_pre_backward_hook = True
        # 返回梯度 grad
        return grad
@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    flat_param,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of `handle`'s `FlatParameter`.

    Precondition: The `FlatParameter`'s `.grad` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using `NO_SHARD`, then the `.grad` attribute is the reduced
    unsharded gradient.
    - Otherwise, the `_saved_grad_shard` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    # 调用日志记录函数，记录后向传播钩子的信息
    _log_post_backward_hook(state, handle, logger)
    # 获取`handle`中的`flat_param`，并赋值给局部变量`flat_param`
    flat_param = handle.flat_param
    # 设置`_post_backward_called`标志为True，表示后向传播钩子已调用
    flat_param._post_backward_called = True
    # 使用 Torch 的自动求导分析器记录函数执行
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
        # 检查当前状态是否为 FORWARD_BACKWARD，如果不是则抛出异常
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # 对于在共享相同 `FlatParameter` 的子模块之间多次应用可重入 AC，后向钩子可能在一个反向传播中运行多次，
        # 在这种情况下，我们允许状态已经处于 `BACKWARD_POST`。
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        # 将状态设置为 BACKWARD_POST
        handle._training_state = HandleTrainingState.BACKWARD_POST

        # 如果梯度为 None，则直接返回
        if flat_param.grad is None:
            return
        # 如果梯度需要梯度，则抛出异常
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        # 执行后向重分片
        _post_backward_reshard(state, handle)
        # 如果不需要同步梯度，则根据情况返回
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        # 等待当前流中的所有操作（例如梯度计算）完成后再进行梯度的 reduce-scattering
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            state._post_backward_stream.wait_stream(
                state._device_handle.current_stream()
            )

        # 在 post_backward_stream 中执行以下操作
        with state._device_handle.stream(state._post_backward_stream):
            # 获取自动求导计算的梯度
            autograd_computed_grad = flat_param.grad.data
            # 如果低精度钩子未启用，并且梯度的数据类型不是 reduce_dtype
            # （如果我们强制使用全精度但通信梯度，则不会降级梯度）
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                and not handle._force_full_precision
            ):
                # 将梯度数据类型转换为 reduce_dtype
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            # 如果使用了分片策略，则执行梯度 reduce 操作
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                # 如果未使用分片策略，则执行无分片的梯度 reduce 操作
                _reduce_grad_no_shard(state, handle)
            # 由于未分片的梯度是在计算流中生成并在后向流中消耗的，因此在缓存分配器（在作用域结束前）中通知
            _no_dispatch_record_stream(
                autograd_computed_grad, state._post_backward_stream
            )
def _post_backward_reshard_only_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    # 使用 torch.profiler 记录函数调用，用于性能分析，标记后向传播中的重新分片操作
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook_reshard_only"
    ):
        # `_pre_backward_hook` 可能不会执行，如果前向传播输出不需要梯度
        # 为了后向传播预取数据，覆盖 IDLE 状态
        state.training_state = TrainingState.FORWARD_BACKWARD
        handle._training_state = HandleTrainingState.BACKWARD_POST
        # 调用 _post_backward_reshard 函数进行后向传播中的重新分片操作
        _post_backward_reshard(state, handle)


def _post_backward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    # 确定在后向传播中是否应该释放非分片的平坦参数
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    # 调用 _reshard 函数进行重新分片操作
    _reshard(state, handle, free_unsharded_flat_param)

    # TODO: 后向传播预取不支持多个句柄每个模块的情况，因为后向传播钩子按句柄而不是句柄组运行
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_prefetch"
    ):
        # 调用 _prefetch_handle 函数进行后向传播预取数据操作
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


@no_type_check
def _should_free_in_backward(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> bool:
    """
    返回 FSDP 是否应该在后向传播中释放非分片的平坦参数。
    """
    if not handle.uses_sharded_strategy:
        return False
    # 如果不同步梯度，则对于不在前向传播后重新分片的策略，不释放作为一个启发式，以交换内存消耗和吞吐量。
    return (
        state._sync_gradients
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )


@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    对于分片策略，运行梯度减少、分片梯度累积（如果需要）和减少后回调。
    """
    flat_param = handle.flat_param
    # 使用混合分片策略时，这里检查是否是混合分片策略
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # 清除 `.grad` 以允许多次反向传播。这避免了第二次反向传播计算在第一次反向传播减少之前完成，
    # 这是可能的，因为减少是在一个单独的流中发出的，并且是异步的，可能导致减少错误的梯度。
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    # 获取减少散布张量，用于梯度减少
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    # 如果状态对象的通信钩子为None，则使用默认路径
    if state._comm_hook is None:  # default path
        # 如果需要的话，对梯度进行划分
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        # 根据是否使用虚拟Reduce来选择处理组
        pg = (
            handle._fake_process_group
            if handle._use_fake_reduce
            else state.process_group
        )
        # 使用分布式库中的reduce_scatter_tensor函数进行张量的reduce scatter操作
        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=pg,
        )
        # 如果使用混合分片策略
        if uses_hybrid_sharded_strategy:
            # 在跟踪期间不等待
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                # 等待post_backward_stream流完成
                state._all_reduce_stream.wait_stream(state._post_backward_stream)
            # 在all_reduce_stream流上执行下列操作
            with state._device_handle.stream(state._all_reduce_stream):
                # 由于新的分片梯度是在post-backward流中生成并在all-reduce流中消耗的，
                # 通知缓存分配器
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                # 在state._inter_node_pg组中进行全局all_reduce操作
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                # 如果需要的话，对新的分片梯度进行划分
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                # 累积分片梯度并获取要卸载的梯度
                grad_to_offload = _accumulate_sharded_grad(
                    state, handle, new_sharded_grad
                )
                # 执行梯度减少后的回调函数
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                # 返回结果
                return
        # 如果需要的话，对新的分片梯度进行划分
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        # 使用状态对象的通信钩子传递状态和梯度数据
        state._comm_hook(
            state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
        )
        # 注意：HSDP变体不支持通信钩子。
    # 累积分片梯度并获取要卸载的梯度
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    # 执行梯度减少后的回调函数
    _post_reduce_grad_callback(state, handle, grad_to_offload)
@no_type_check
def _get_reduce_scatter_tensors(
    state: _FSDPState, unsharded_grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    # 将未分片的梯度张量切分成与 world_size 相等的块
    chunks = list(unsharded_grad.chunk(state.world_size))
    # 计算需要填充的元素数量，使得梯度张量可以整除 world_size
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    # 如果需要填充，则使用 F.pad 对梯度张量进行填充；否则保持原始梯度张量
    padded_unsharded_grad = (
        F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    # 创建一个与第一个块相同形状的新张量，用于存储填充后的梯度
    new_sharded_grad = torch.empty_like(chunks[0])  # padded
    return padded_unsharded_grad, new_sharded_grad


@no_type_check
def _accumulate_sharded_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    sharded_grad: torch.Tensor,
) -> torch.Tensor:
    """
    Accumulates the reduce-scattered sharded gradient with any existing sharded
    gradient if needed, returning the gradient to offload (if CPU offloading is
    enabled).
    """
    flat_param = handle.flat_param
    # 将分片的梯度张量转换为参数的数据类型
    _cast_grad_to_param_dtype(state, sharded_grad, flat_param)
    # 检查是否已经存在保存的分片梯度，若存在则累加新的分片梯度，否则直接保存新的分片梯度
    accumulate_grad = hasattr(flat_param, "_saved_grad_shard")
    if accumulate_grad:
        _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        flat_param._saved_grad_shard += sharded_grad
    else:
        flat_param._saved_grad_shard = sharded_grad
    # 返回需要卸载的梯度，即累积的梯度
    grad_to_offload = flat_param._saved_grad_shard
    return grad_to_offload


@no_type_check
def _reduce_grad_no_shard(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For no-shard, this runs gradient reduction (which directly covers any
    gradient accumulation implicitly) and the post-reduction callback.
    """
    flat_param = handle.flat_param
    # 如果没有设置通信钩子，则使用默认路径
    if state._comm_hook is None:
        # 根据梯度预除因子进行除法操作
        _div_if_needed(flat_param.grad, state._gradient_predivide_factor)
        # 使用分布式通信方式对梯度进行全局归约
        dist.all_reduce(flat_param.grad, group=state.process_group)
        # 根据梯度后除因子进行除法操作
        _div_if_needed(flat_param.grad, state._gradient_postdivide_factor)
    else:
        # 如果设置了通信钩子，则调用通信钩子函数
        state._comm_hook(state._comm_hook_state, flat_param.grad)
    # 如果不需要保留低精度梯度，则将梯度转换为参数的数据类型
    if not handle._keep_low_precision_grads:
        _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
    # 获取需要卸载的梯度数据
    grad_to_offload = flat_param.grad.data
    # 调用梯度归约后的回调函数
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _post_reduce_grad_callback(
    state: _FSDPState,
    handle: FlatParamHandle,
    # Additional arguments needed for the callback logic
    grad_to_offload: torch.Tensor,
):
    """
    This callback captures any logic to run after the gradient reduction
    finishes. Currently, this offloads the gradient to CPU if CPU offloading is
    enabled and uses sharded gradient views if ``use_orig_params=True``.
    """
    # 在梯度归约后执行回调逻辑，例如根据 CPU 卸载是否启用将梯度卸载到 CPU，并在需要时使用分片梯度视图
    # 调用 _offload_grad 函数，将梯度从指定状态中转移出来，并使用给定的处理句柄和要转移的梯度作为参数
    _offload_grad(state, handle, grad_to_offload)
    
    # 调用 _post_backward_use_sharded_grad_views 函数，用于在反向传播之后使用分片的梯度视图，传递给定的处理句柄作为参数
    _post_backward_use_sharded_grad_views(handle)
# 禁用类型检查的装饰器，标志着这些函数可能包含动态类型的使用
@no_type_check
# 将梯度从分布式状态 `_FSDPState` 中移出，以便与优化器要求的参数和梯度在同一设备上
def _offload_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    grad_to_offload: torch.Tensor,
):
    # 如果没有要移出的参数，则直接返回
    if not handle._offload_params:
        return
    # 根据是否使用分片策略和是否在反向传播中没有优化器，决定是否使用非阻塞的梯度移动
    non_blocking = handle.uses_sharded_strategy and not handle._has_optim_in_backward
    # 将要移出的梯度复制到 CPU 上，以确保参数和梯度在同一设备上
    handle.flat_param._cpu_grad.copy_(
        grad_to_offload.detach(), non_blocking=non_blocking
    )  # 在后向回调中进行同步
    # 由于要移出的梯度可能在计算流中生成，在后向流中消耗，通知缓存分配器
    _no_dispatch_record_stream(grad_to_offload.data, state._post_backward_stream)


@no_type_check
# 在后向传播后使用分片的梯度视图
def _post_backward_use_sharded_grad_views(handle: FlatParamHandle):
    # 如果不使用原始参数，则直接返回
    if not handle._use_orig_params:
        return
    # 重置梯度非空掩码，因为 `FlatParameter` 的梯度计算已完成
    handle._reset_is_grad_none()
    # 推迟使用分片的梯度视图，直到执行了 reduce-scatter 而不是重新分片后立即执行
    handle._use_sharded_grad_views()
    # 如果在反向传播中有优化器，为优化器准备梯度，并执行优化步骤
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            # 检查参数的梯度是否为 `None`，以过滤不在当前进程中的参数
            if orig_param.grad is not None and hasattr(
                orig_param, "_in_backward_optimizers"
            ):
                # 对每个优化器执行优化步骤
                for optim in orig_param._in_backward_optimizers:
                    optim.step()
                # 将梯度置零
                optim.zero_grad(set_to_none=True)
        # 如果需要，重置扁平参数的梯度信息
        handle._reset_flat_param_grad_info_if_needed()
        # 如果需要移出参数，将 `flat_param._cpu_grad` 置为 `None`
        if handle._offload_params:
            handle.flat_param._cpu_grad = None


# 如果除数大于1，则对张量进行除法操作
def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)


@no_type_check
# 将分片的梯度 `_cast_grad_to_param_dtype` 转换为参数的完整数据类型，以便优化步骤使用该数据类型运行
def _cast_grad_to_param_dtype(
    state: _FSDPState,
    sharded_grad: torch.Tensor,
    param: FlatParameter,
):
    """
    将 `sharded_grad` 转换回完整的参数数据类型，以便优化器步骤使用该数据类型运行。如果
    ```
    """
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    确保当前状态在训练过程中，即在前向和反向传播之间

    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
    如果低精度钩子未启用，并且梯度的数据类型与参数的数据类型不同：

        low_prec_grad_data = sharded_grad.data
        将低精度梯度数据保存到变量 low_prec_grad_data 中

        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        将梯度数据类型转换为参数的数据类型

        # Since for `NO_SHARD`, the gradient is produced in the computation
        # stream and consumed here in the post-backward stream, inform the
        # caching allocator; for the sharded strategies, the gradient is
        # produced in the post-backward stream, so this `record_stream()`
        # should be a no-op
        # 因为对于 `NO_SHARD`，梯度是在计算流中产生并在此后向流中消耗的，
        # 通知缓存分配器；对于分片策略，梯度是在后向流中产生的，
        # 因此这里的 `record_stream()` 应该是一个空操作
        _no_dispatch_record_stream(
            low_prec_grad_data, state._device_handle.current_stream()
        )
# 检查并断言累积梯度和新梯度的形状是否匹配
def _check_grad_to_accumulate(
    new_sharded_grad: torch.Tensor,
    accumulated_grad: torch.Tensor,
) -> None:
    _p_assert(
        accumulated_grad.shape == new_sharded_grad.shape,
        "Shape mismatch when accumulating gradients: "
        f"existing gradient shape={accumulated_grad.shape} "
        f"new gradient shape={new_sharded_grad.shape}",
    )
    # 检查并断言累积梯度和新梯度的设备是否一致
    _p_assert(
        accumulated_grad.device == new_sharded_grad.device,
        "Device mismatch when accumulating gradients: "
        f"existing gradient device={accumulated_grad.device} "
        f"new gradient device={new_sharded_grad.device}",
    )


@no_type_check
# 检查状态对象是否启用低精度钩子
def _low_precision_hook_enabled(state: _FSDPState) -> bool:
    return state._comm_hook in LOW_PRECISION_HOOKS


@no_type_check
@torch.no_grad()
# 后向传播最终回调函数，用于等待后向传播完成并进行最终清理工作
# 该函数在整个后向传播结束时运行，只应在根 FSDP 实例上调用
def _post_backward_final_callback(
    state: _FSDPState,
    module: nn.Module,
):
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
    # 断言仅在根 FSDP 实例上调用后向传播回调函数
    _p_assert(
        state._is_root,
        "The post-backward callback should only be called on the root FSDP instance",
    )
    root_state = state

    if root_state._sync_gradients:
        current_stream = state._device_handle.current_stream()
        # TODO (rohan-varma): this also waits for the overlapped optimizer step to finish
        # since it currently runs in the post-backward stream. That can be
        # pushed to the next forward if run in a different stream
        # 等待 post-backward 流完成，同时也等待重叠的优化器步骤完成
        current_stream.wait_stream(root_state._post_backward_stream)
        if root_state._all_reduce_stream is not current_stream:  # uses HSDP
            # 如果使用 HSDP，则等待所有约减流的完成
            current_stream.wait_stream(root_state._all_reduce_stream)
        if root_state.cpu_offload.offload_params:
            # 等待非阻塞 GPU -> CPU 的分片梯度复制完成，
            # 因为 CPU 梯度不会自动与 GPU 同步
            state._device_handle.current_stream().synchronize()
    # 执行下一个迭代的执行顺序数据
    root_state._exec_order_data.next_iter()

    for fsdp_state in state._all_fsdp_states:
        # 对所有 FSDP 状态进行重新分片
        _catch_all_reshard(fsdp_state)
        # 最终化参数
        _finalize_params(fsdp_state)
        # 设置训练状态为空闲
        fsdp_state.training_state = TrainingState.IDLE
        handle = fsdp_state._handle
        if handle:
            # 重置前向钩子的运行状态
            handle._ran_pre_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
    # 对于一次前向和多次后向等情况重置后向回调标志位
    root_state._post_backward_callback_queued = False


@no_type_check
# 对可能未在后向钩子中重新分片的参数进行重新分片
def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    # 将代码放在 try-except 块中，如果出现错误，提供更详细的回溯信息
    try:
        # 检查是否存在状态对象的句柄
        if state._handle:
            # TODO: 这个 already_resharded 检查可能不够健壮：
            # https://github.com/pytorch/pytorch/issues/83956
            # 检查参数是否已经经过重分片
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
                # 如果 FSDP 跳过使用分片视图，则 flat 参数仍指向分片数据，因此我们需要重分片以使用分片视图
                and not state._handle._skipped_use_sharded_views
            )
            # 如果已经经过重分片，则直接返回
            if already_resharded:
                return
            # 检查是否应该在反向传播时释放未分片的 flat 参数
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            # 执行重分片操作
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        # 如果出现异常，在这里进行断言
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        # 抛出异常以中断程序执行
        raise e
@no_type_check
def _finalize_params(
    state: _FSDPState,
) -> None:
    """Finalizes the parameters before the next iteration."""
    # 获取状态对象中的句柄
    handle = state._handle
    # 如果句柄不存在，则返回
    if not handle:
        return
    # 获取句柄中的平坦化参数对象
    flat_param = handle.flat_param
    # 如果当前处于TorchDynamo编译状态
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        # 如果平坦化参数对象有"_post_backward_hook_handle"属性
        if hasattr(flat_param, "_post_backward_hook_handle"):
            # 获取并移除后向传播钩子句柄
            pbhs_handle = flat_param._post_backward_hook_handle
            pbhs_handle.remove()
            del flat_param._post_backward_hook_handle
    else:
        # 如果平坦化参数对象有"_post_backward_hook_state"属性
        if hasattr(flat_param, "_post_backward_hook_state"):
            # 检查后向传播钩子状态列表长度是否正确
            post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
            expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
            _p_assert(
                post_backward_hook_state_len == expected_post_backward_hook_state_len,
                f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
            )
            # 移除最后一个后向传播钩子状态，并删除"_post_backward_hook_state"属性
            flat_param._post_backward_hook_state[-1].remove()
            delattr(flat_param, "_post_backward_hook_state")
    # 如果平坦化参数对象需要梯度
    if flat_param.requires_grad:
        # 如果不需要同步梯度，则保留梯度累积状态
        if not state._sync_gradients:
            # 如果不同步梯度，则直接返回
            # `.grad`保持前一个`no_sync()`迭代的未分片梯度，`_saved_grad_shard`
            # 保留上一个同步迭代的分片梯度
            return
        # 如果句柄没有优化器在反向传播中使用，则准备梯度供优化器使用
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        # 断言"_post_backward_called"属性已设置在FlatParameter上
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        # 将"_post_backward_called"属性设置为False
        flat_param._post_backward_called = False


@no_type_check
def _prefetch_handle(
    state: _FSDPState,
    current_handle: Optional[FlatParamHandle],
    prefetch_mode: _PrefetchMode,
) -> None:
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    # 如果当前句柄不存在，则返回
    if not current_handle:
        return
    # 获取要预取的句柄对象
    handle = _get_handle_to_prefetch(state, current_handle)
    # 如果没有要预取的句柄，则返回
    if not handle:
        return
    # 暂时模拟训练状态，在调用`_unshard`时确保正确的`as_params`用于`_use_unsharded_views()`
    prev_training_state = handle._training_state
    # 根据预取模式设置训练状态
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        handle._training_state = HandleTrainingState.FORWARD
    else:
        # 如果预取模式无效，则抛出错误
        raise ValueError(f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}")
    # 预取下一组句柄，不进行同步以尽可能延迟同步以最大化重叠
    _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    # 恢复之前的训练状态
    handle._training_state = prev_training_state
    # 将对象的 _prefetched 属性设置为 True
    handle._prefetched = True
@no_type_check
def _get_handle_to_prefetch(
    state: _FSDPState,
    current_handle: FlatParamHandle,
) -> FlatParamHandle:
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    # 获取当前模块的训练状态
    training_state = _get_training_state(current_handle)
    # 定义有效的训练状态列表
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    # 断言当前训练状态在有效的训练状态列表中
    _p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    # 获取执行顺序数据
    eod = state._exec_order_data
    # 初始化目标处理句柄为 None
    target_handle: Optional[FlatParamHandle] = None
    # 根据训练状态决定目标处理句柄的赋值逻辑
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        # 获取用于向后预取的处理句柄候选
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        # 如果候选句柄存在且满足预先向后不分片的需求且尚未预取，则选择该候选句柄
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        # 获取用于向前预取的处理句柄候选
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        # 如果候选句柄存在且满足预先向前不分片的需求且尚未预取，则选择该候选句柄
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    return target_handle


def _get_training_state(
    handle: FlatParamHandle,
) -> HandleTrainingState:
    """Returns the training state of the handles in ``handle``."""
    # 断言句柄非空
    _p_assert(handle, "Expects a non-empty handle")
    # 返回句柄的训练状态
    return handle._training_state


@no_type_check
def _register_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers a pre-forward hook on ``module``.
    """
    # 清除之前的所有前向钩子
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    # 获取模块对应的完全分片处理句柄
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    # 创建偏函数钩子
    hook = functools.partial(
        _pre_forward, state, module_param_handle, _pre_forward_unshard
    )
    # 注册前向钩子，并将其添加到前向钩子列表中
    state._pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_post_forward_hook(
    state: _FSDPState,
    module: nn.Module,


    state: _FSDPState,  # 定义变量 state，类型为 _FSDPState，用于存储分布式状态管理信息
    module: nn.Module,  # 定义变量 module，类型为 nn.Module，用于存储神经网络模块对象
def _register_root_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
):
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
    # Remove any existing forward hooks
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    # Clear the list of existing root pre-forward handles
    state._root_pre_forward_handles.clear()
    # Define a partial function hook for root pre-forward operations
    hook = functools.partial(_root_pre_forward, state)
    # Register the hook for the module's forward pass, prepended with kwargs
    state._root_pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )
    # 调用 _apply_to_tensors 函数，对 outputs 中的张量应用 _register_hook 函数
    return _apply_to_tensors(_register_hook, outputs)
def _register_post_backward_hook(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    # 如果没有梯度计算，则不需要后向逻辑
    if not torch.is_grad_enabled():
        return
    
    # 如果没有提供句柄，则返回
    if not handle:
        return
    
    # 获取扁平参数对象
    flat_param = handle.flat_param

    # 如果处于编译状态，则检查是否已经注册过后向钩子或者参数不需要梯度
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
        if already_registered or not flat_param.requires_grad:
            return
        
        # 创建部分函数，用于后向钩子
        hook = functools.partial(_post_backward_hook, state, handle)
        # 注册后向累积梯度钩子
        hook_handle = flat_param.register_post_accumulate_grad_hook(hook)
        # 设置后向钩子句柄
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined]
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            return
        
        # 获取累积梯度对象
        temp_flat_param = flat_param.expand_as(flat_param)
        _p_assert(
            temp_flat_param.grad_fn is not None,
            "The `grad_fn` is needed to access the `AccumulateGrad` and "
            "register the post-backward hook",
        )
        acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
        assert acc_grad is not None
        
        # 注册钩子函数
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )
        # 设置后向钩子状态
        flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]
    # 如果没有梯度计算，则不需要后向逻辑
    if not torch.is_grad_enabled():
        return
    
    # 惰性地构建 `inp_tensors`，以避免在典型情况下每个扁平参数都需要梯度时的 CPU 开销
    inp_tensors: Optional[List[torch.Tensor]] = None
    
    # 如果没有有效的 `handle`，则直接返回
    if not handle:
        return
    
    # 获取扁平参数对象
    flat_param = handle.flat_param
    
    # 检查是否正在 TorchDynamo 编译中，判断是否已经注册过后向钩子
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
    
    # 如果已经注册过或者扁平参数需要梯度，则直接返回
    if already_registered or flat_param.requires_grad:
        return
    
    # 如果 `inp_tensors` 尚未构建，则从参数和关键字参数中提取扁平参数列表
    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [
            obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad
        ]
    
    # 断言确保 `inp_tensors` 不为空，用于类型检查
    assert inp_tensors is not None  # mypy
    
    # 注册多梯度钩子，关联 `_post_backward_reshard_only_hook` 和相关状态与句柄
    hook_handle = register_multi_grad_hook(
        inp_tensors, functools.partial(_post_backward_reshard_only_hook, state, handle)
    )
    
    # 根据 TorchDynamo 编译状态设置后向钩子句柄
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined, assignment]
    else:
        flat_param._post_backward_hook_state = (hook_handle,)  # type: ignore[attr-defined, assignment]
@no_type_check
def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    # 断言只有根 FSDP 实例可以注册后向传播的回调函数
    _p_assert(
        state._is_root,
        "Only the root FSDP instance should register the post-backward callback",
    )
    # 如果已经排队了 post-backward 回调函数，则直接返回
    if state._post_backward_callback_queued:
        return
    # 确保状态处于训练中
    _assert_in_training_states(state, [TrainingState.IDLE])
    # 如果不是在 TorchDynamo 编译模式下，则标记已经排队了 post-backward 回调函数
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        state._post_backward_callback_queued = True
        # 将 _post_backward_final_callback 函数作为回调函数排入执行引擎的队列中
        Variable._execution_engine.queue_callback(
            functools.partial(_post_backward_final_callback, state, module)
        )


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    # 如果是在 TorchDynamo 编译模式下，则直接返回，跳过等待步骤
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        return
    # 让 unshard 流等待 computation 流
    unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]
    # 即使我们不使用 pre-all-gather 流，也让其等待当前流，这样做是可接受的，因为每次迭代只运行一次
    pre_unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]


def _reset_flat_param_grad_info_if_needed(
    handles: List[FlatParamHandle],
):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
    # 如果 handles 不是 list 类型，则转换为 list
    if not isinstance(handles, list):
        handles = [handles]
    # 针对每个 handle，如果使用原始参数，则重置平坦参数的梯度信息
    for handle in handles:
        if handle._use_orig_params:
            handle._reset_flat_param_grad_info_if_needed()


@no_type_check
def _get_buffers_and_dtypes_for_computation(
    state: _FSDPState,
    root_module: nn.Module,
) -> Tuple[List[torch.Tensor], List[Optional[torch.dtype]]]:
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
    # 断言只有根 FSDP 实例可以获取缓冲区和数据类型用于计算
    _p_assert(state._is_root, "Expects the root to cast buffers")
    # 初始化返回结果的列表
    buffers: List[torch.Tensor] = []
    buffer_dtypes: List[Optional[torch.dtype]] = []
    visited_buffers: Set[torch.Tensor] = set()
    # 自底向上遍历 FSDP 状态，以便对每个缓冲区优先使用所属 FSDP 实例的混合精度设置
    # 使用 traversal_utils 模块中的 _get_fsdp_states_with_modules 函数获取 fsdp_states 和 fsdp_modules
    fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(
        root_module
    )
    # 反向遍历 fsdp_states 和 fsdp_modules 列表中的元素
    for fsdp_state, fsdp_module in zip(reversed(fsdp_states), reversed(fsdp_modules)):
        # 遍历 fsdp_module 中所有命名缓冲区的名称和缓冲区对象
        for buffer_name, buffer in fsdp_module.named_buffers():
            # 如果缓冲区已经被访问过，则跳过
            if buffer in visited_buffers:
                continue
            # 将当前缓冲区加入到已访问的缓冲区集合中
            visited_buffers.add(buffer)
            # 如果清理后的缓冲区名称在 fsdp_state._ignored_buffer_names 中，则跳过
            if clean_tensor_name(buffer_name) in fsdp_state._ignored_buffer_names:
                continue
            # 将缓冲区对象添加到 buffers 列表中
            buffers.append(buffer)
            # 将缓冲区对应的数据类型添加到 buffer_dtypes 列表中
            buffer_dtypes.append(fsdp_state.mixed_precision.buffer_dtype)
    # 断言 buffers 列表和 buffer_dtypes 列表的长度相等，如果不相等则抛出异常
    assert len(buffers) == len(buffer_dtypes), f"{len(buffers)} {len(buffer_dtypes)}"
    # 返回 buffers 列表和 buffer_dtypes 列表作为结果
    return buffers, buffer_dtypes
# 声明一个装饰器，用于取消类型检查
@no_type_check
# 定义一个函数，用于获取给定缓冲区名称的原始缓冲区数据类型列表
def _get_orig_buffer_dtypes(
    state: _FSDPState,  # 参数：_FSDPState 类型的对象，表示状态
    buffer_names: List[str],  # 参数：包含字符串的列表，表示缓冲区名称
) -> List[torch.dtype]:  # 返回类型：包含 torch.dtype 元素的列表
    """
    Returns the original buffer types of the given buffer names.
    """
    # 初始化一个空列表，用于存储缓冲区数据类型
    buffer_dtypes: List[torch.dtype] = []
    
    # 遍历每个缓冲区名称
    for buffer_name in buffer_names:
        # 断言：确保 buffer_name 在 state._buffer_name_to_orig_dtype 中存在
        _p_assert(
            buffer_name in state._buffer_name_to_orig_dtype,
            f"{buffer_name} is missing from pre-computed dict on rank "
            f"{state.rank}, which only has keys "
            f"{state._buffer_name_to_orig_dtype.keys()}",
        )
        # 将缓冲区名称对应的原始数据类型添加到 buffer_dtypes 列表中
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    
    # 返回存储了原始缓冲区数据类型的列表
    return buffer_dtypes


# 定义一个函数，用于将缓冲区 tensors 转换成指定的数据类型和设备
def _cast_buffers_to_dtype_and_device(
    buffers: List[torch.Tensor],  # 参数：包含 torch.Tensor 元素的列表，表示缓冲区 tensors
    buffer_dtypes: List[Optional[torch.dtype]],  # 参数：包含可选的 torch.dtype 元素的列表，表示缓冲区数据类型
    device: torch.device,  # 参数：torch.device 对象，表示目标设备
) -> None:  # 返回类型：None
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    # 断言：确保如果 buffer_dtypes 被指定，则它和 buffers 的长度一致
    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    
    # 遍历 buffers 和 buffer_dtypes 列表
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        # 如果 buffer 不是浮点类型或者 buffer_dtype 是 None
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            # 将 buffer 转移到指定设备上（仅移动，不改变数据类型）
            buffer.data = buffer.to(device=device)
        else:
            # 将 buffer 转移到指定设备上，并且转换成指定的数据类型
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)
```