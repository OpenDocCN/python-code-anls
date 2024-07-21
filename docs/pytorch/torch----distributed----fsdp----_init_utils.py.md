# `.\pytorch\torch\distributed\fsdp\_init_utils.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和库
import collections
import itertools
import os
import warnings
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    no_type_check,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _FSDPState,
    _get_module_fsdp_state,
    _is_fsdp_flattened,
    _named_parameters_with_duplicates,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._flat_param import (
    _FSDP_USE_FULL_PREC_IN_EVAL,
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
)
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# 如果在类型检查模式下
if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

# 定义常量，用于控制参数广播时的桶大小
PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)
# 指示是否进行了 FSDP 同步的标记
FSDP_SYNCED = "_fsdp_synced"
# 混合分片策略的进程组类型的说明
HybridShardProcessGroupType = Tuple[dist.ProcessGroup, dist.ProcessGroup]
# 总体进程组的说明
ProcessGroupType = Optional[Union[dist.ProcessGroup, HybridShardProcessGroupType]]

# 映射不同分片策略到对应的处理分片策略
SHARDING_STRATEGY_MAP = {
    ShardingStrategy.NO_SHARD: HandleShardingStrategy.NO_SHARD,
    ShardingStrategy.FULL_SHARD: HandleShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP: HandleShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.HYBRID_SHARD: HandleShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2: HandleShardingStrategy._HYBRID_SHARD_ZERO2,
}
# 定义支持混合分片策略的列表
HYBRID_SHARDING_STRATEGIES = [
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
]
# 定义在前向传播后不重新分片的策略列表
NO_RESHARD_AFTER_FORWARD_STRATEGIES = (
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)

# 注意: 由于不能对非 self 属性进行类型注释，因此有几个属性
# 定义一个函数 `_init_process_group_state`，用于初始化分布式数据并行（FSDP）状态。
@no_type_check
def _init_process_group_state(
    state: _FSDPState,  # 参数state，类型为_FSDPState，表示FSDP的状态对象
    process_group: ProcessGroupType,  # 参数process_group，表示进程组类型
    sharding_strategy: ShardingStrategy,  # 参数sharding_strategy，表示分片策略类型
    policy: Optional[_Policy],  # 参数policy，可选的策略对象
    device_mesh: Optional[DeviceMesh] = None,  # 参数device_mesh，可选的设备网格对象，默认为None
) -> _FSDPState:  # 函数返回类型为_FSDPState

    # 如果同时传入了process_group和device_mesh，则抛出ValueError异常
    if process_group is not None and device_mesh is not None:
        raise ValueError(
            "Cannot pass both process_group and device_mesh at the "
            "same time. Please just pass only one of them."
        )

    # 检查是否是混合策略
    is_hybrid_strategy = sharding_strategy in HYBRID_SHARDING_STRATEGIES

    # 如果是混合策略
    if is_hybrid_strategy:
        # 如果没有传入process_group、policy和device_mesh，则抛出ValueError异常
        if process_group is None and policy is None and device_mesh is None:
            raise ValueError(
                f"Manual wrapping with {sharding_strategy} "
                "requires explicit specification of process group or device_mesh."
            )
        else:
            # 使用 _init_process_group_state_for_hybrid_shard 函数初始化状态
            state = _init_process_group_state_for_hybrid_shard(
                state, process_group, device_mesh
            )
    else:
        # 如果传入了device_mesh
        if device_mesh:
            state._device_mesh = device_mesh
            state.process_group = device_mesh.get_group(mesh_dim=0)
        else:
            # 否则使用传入的process_group，如果没有传入则使用默认的进程组
            state.process_group = (
                process_group if process_group is not None else _get_default_group()
            )

    # 设置状态对象的rank属性为当前进程在进程组中的rank
    state.rank = state.process_group.rank()
    # 设置状态对象的world_size属性为进程组的大小
    state.world_size = state.process_group.size()

    # 计算数据并行世界大小
    data_parallel_world_size = state.world_size
    if is_hybrid_strategy:
        # 如果是混合策略，需要乘以_inter_node_pg的大小
        data_parallel_world_size *= state._inter_node_pg.size()

    # 设置状态对象的_gradient_predivide_factor属性为数据并行世界大小的梯度预分配因子
    state._gradient_predivide_factor = (
        default_hooks.DefaultState._get_gradient_predivide_factor(
            data_parallel_world_size
        )
    )
    # 设置状态对象的_gradient_postdivide_factor属性为数据并行世界大小除以_gradient_predivide_factor
    state._gradient_postdivide_factor = (
        data_parallel_world_size / state._gradient_predivide_factor
    )

    # 返回更新后的状态对象
    return state


# 定义一个函数 `_init_process_group_state_for_hybrid_shard`，用于初始化混合分片的FSDP状态
@no_type_check
def _init_process_group_state_for_hybrid_shard(
    state: _FSDPState,  # 参数state，类型为_FSDPState，表示FSDP的状态对象
    process_group: ProcessGroupType,  # 参数process_group，表示进程组类型
    device_mesh: DeviceMesh,  # 参数device_mesh，表示设备网格类型
) -> _FSDPState:  # 函数返回类型为_FSDPState

    # 如果传入了device_mesh
    if device_mesh:
        # 检查device_mesh是否是有效的混合分片设备网格
        if _is_valid_hybrid_shard_device_mesh(device_mesh):
            state._device_mesh = device_mesh
            # 将_inter_node_pg设为外层维度的进程组，process_group设为内层维度的进程组
            state._inter_node_pg = device_mesh.get_group(mesh_dim=0)
            state.process_group = device_mesh.get_group(mesh_dim=1)
        else:
            # 如果device_mesh不是预期的二维结构，则抛出异常
            raise ValueError(
                f"Expected device_mesh to have ndim=2 but got {device_mesh.ndim}"
            )
    # 如果 process_group 是 None，则使用默认的组进行初始化
    default_group = _get_default_group()
    # 初始化 intra_node_group 和 inter_node_group，根据当前设备数量来确定
    intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
        default_group, state._device_handle.device_count()
    )
    # 在本地节点间进行分片处理
    state.process_group = intra_node_group
    # 将 _inter_node_pg 保存下来用于跨节点的全reduce操作
    state._inter_node_pg = inter_node_group



    # 如果 process_group 不为 None，则进行类型检查并分配 state.process_group 和 state._inter_node_pg
    if _is_valid_hybrid_shard_pg_type(process_group):
        # 假设用户按文档传入 intra node group 和 inter node group
        state.process_group, state._inter_node_pg = process_group
    else:
        # 抛出值错误异常，说明 process_group 的类型不符合预期
        raise ValueError(
            "Expected process_group to be passed in as either None or "
            f"Tuple[dist.ProcessGroup, dist.ProcessGroup] but got {type(process_group)}"
        )



    # 创建用于全reduce的 _inter_node_state 状态
    state._inter_node_state = _get_default_comm_hook_state(
        process_group=state._inter_node_pg,
    )
    # 返回更新后的 state 对象
    return state
# 对给定的 process_group 对象进行类型检查，确保其为一个包含两个 dist.ProcessGroup 实例的元组，并返回布尔值
@no_type_check
def _is_valid_hybrid_shard_pg_type(process_group: Any) -> bool:
    return (
        isinstance(process_group, tuple)
        and len(process_group) == 2
        and all(isinstance(pg, dist.ProcessGroup) for pg in process_group)
    )


# 对给定的 device_mesh 对象进行类型检查，确保其为 DeviceMesh 类型且维度为 2，并返回布尔值
@no_type_check
def _is_valid_hybrid_shard_device_mesh(device_mesh: DeviceMesh) -> bool:
    return isinstance(device_mesh, DeviceMesh) and device_mesh.ndim == 2


# 初始化当前节点内的进程组，并返回该进程组
@no_type_check
def _init_intra_node_process_group(num_devices_per_node: int) -> dist.ProcessGroup:
    """
    Return a process group across the current node.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return an intra-node subgroup across
    [0, 1, ..., 7] or [8, 9, ..., 15] depending on the process's rank.
    For example, rank 3 would get [0, 1, ..., 7].
    """
    # 调用 dist.new_subgroups 方法创建当前节点内的子组，并返回其中的 intra_node_subgroup
    intra_node_subgroup, _ = dist.new_subgroups(num_devices_per_node)
    return intra_node_subgroup


# 初始化跨节点的进程组，并返回该进程组
@no_type_check
def _init_inter_node_process_group(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> dist.ProcessGroup:
    """
    Return an inter-node process group where each contained rank has the same local rank.

    For example, given each row is a distinct node:
    0  1  2  3  4  5  6  7
    8  9 10 11 12 13 14 15
    This API would return inter-node process group [0, 8], [1, 9], [2, 10], and so forth
    depending on the process's rank. For example, rank 1 would get [1, 9], rank 5
    would get [5, 13].
    """
    # 初始化 inter_node_pg 为 None
    inter_node_pg = None
    # 获取全局进程组的后端信息
    sharding_backend = dist.get_backend(global_process_group)
    # 获取全局进程组的总进程数
    world_size = dist.get_world_size(global_process_group)
    # 假设节点完全均匀设置
    num_nodes = world_size // num_devices_per_node
    # 计算当前进程的本地排名
    my_local_rank = dist.get_rank(global_process_group) % num_devices_per_node
    # 遍历每个本地排名
    for local_rank in range(num_devices_per_node):
        # 计算用于 inter-node 组的排名列表
        ranks_for_inter_group = [
            local_rank + (i * num_devices_per_node) for i in range(num_nodes)
        ]
        # 使用 dist.new_group 方法创建新的进程组 grp
        grp = dist.new_group(ranks=ranks_for_inter_group, backend=sharding_backend)
        # 如果当前 local_rank 等于 my_local_rank，则将 grp 赋值给 inter_node_pg
        if local_rank == my_local_rank:
            inter_node_pg = grp

    # 断言确保 inter_node_pg 不为空
    assert (
        inter_node_pg is not None
    ), f"{my_local_rank} expected to assign inter-node pg, but did not"
    return inter_node_pg


# 初始化节点内和跨节点的进程组，并返回对应于当前进程排名的两个进程组
def _init_intra_and_inter_node_groups(
    global_process_group: dist.ProcessGroup,
    num_devices_per_node: int,
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    Initialize intra and inter-node process groups and return the ones corresponding to this process's rank.

    This function can be used to initialize process groups for ``HYBRID_SHARD`` or
    ``_HYBRID_SHARD_ZERO2`` in FSDP.
    This function assumes each node has an equal number of CUDA-enabled devices.
    """
    # 返回两个元素的元组，包含了本地节点内和跨节点间的进程组
    return (
        # 调用函数 _init_intra_node_process_group，创建本地节点内的进程组
        _init_intra_node_process_group(num_devices_per_node),
        # 调用函数 _init_inter_node_process_group，创建跨节点间的进程组
        _init_inter_node_process_group(global_process_group, num_devices_per_node),
    )
@no_type_check
def _init_ignored_module_states(
    state: _FSDPState,
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]],
    ignored_states: Union[
        Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
    ] = None,
) -> _FSDPState:
    # 如果同时传递了 ignored_modules 和 ignored_states，则抛出数值错误
    if ignored_modules is not None and ignored_states is not None:
        raise ValueError(
            "Cannot pass both ignored_modules and ignored_states at the "
            "same time. Please just pass ignored_states."
        )
    ignored_parameters = None
    # 检查是否传递了 ignored_states
    passed_as_ignored_states = ignored_states is not None
    if passed_as_ignored_states:
        # 将 ignored_states 转换为列表并检查其内容
        ignored_states_list = list(ignored_states)
        _check_ignored_states(ignored_states_list, True)
    else:
        # 如果没有传递 ignored_states，则创建一个空列表，并检查 ignored_modules
        ignored_states_list = []
        _check_ignored_states(
            list(ignored_modules) if ignored_modules is not None else [], False
        )
    # 根据 ignored_states_list 的内容确定 ignored_parameters 或 ignored_modules
    if len(ignored_states_list) > 0:
        if isinstance(ignored_states_list[0], nn.Parameter):
            ignored_parameters = ignored_states_list
        else:
            ignored_modules = ignored_states_list
    # 设置状态对象的 _ignored_modules 属性
    state._ignored_modules = _get_ignored_modules(module, ignored_modules)
    # 设置状态对象的 _ignored_params 属性
    state._ignored_params = _get_ignored_params(
        module,
        state._ignored_modules,
        ignored_parameters,
    )
    # 设置状态对象的 _ignored_buffer_names 属性
    state._ignored_buffer_names = _get_ignored_buffer_names(
        module,
        state._ignored_modules,
    )
    # TODO: FSDP 的缓冲区约定尚未明确定义。大多数功能隐式忽略它们，因为它们不分片；
    # 然而，FSDP 对缓冲区仍施加一些语义（例如缓冲区混合精度）。我们应该明确化这一约定，
    # 并决定是否需要计算和存储 `_ignored_buffers`。
    # 返回状态对象
    return state


def _check_ignored_states(
    ignored_states: List[Any], passed_as_ignored_states: bool
) -> None:
    """
    Check that the ignored states are uniformly parameters or uniformly modules.

    We may remove this check in the future if we permit mixing.
    """
    # 如果 ignored_states 列表为空，则直接返回
    if len(ignored_states) == 0:
        return
    # 如果传递了 ignored_states，则检查其元素类型是否统一
    if passed_as_ignored_states:
        all_params = all(isinstance(state, nn.Parameter) for state in ignored_states)
        all_modules = all(isinstance(state, nn.Module) for state in ignored_states)
        if not all_params and not all_modules:
            # 对类型进行排序，以便单元测试正则表达式匹配时保持一致的顺序
            sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
            raise ValueError(
                "ignored_states expects all nn.Parameter or all nn.Module list "
                f"elements but got types {sorted_types}"
            )
    # 如果不是第一个条件满足，则执行以下代码块
    else:
        # 检查 ignored_states 中的每个元素是否都是 nn.Module 类型
        if not all(isinstance(state, nn.Module) for state in ignored_states):
            # 将 ignored_states 中元素的类型去重排序，并按照字符串表示进行排序
            sorted_types = sorted({type(state) for state in ignored_states}, key=repr)
            # 抛出数值错误，说明 ignored_modules 应该包含 nn.Module 类型的列表元素，但实际上给出了哪些类型
            raise ValueError(
                "ignored_modules expects nn.Module list elements but got "
                f"types {sorted_types}"
            )
# 使用 @no_type_check 装饰器来禁用类型检查
@no_type_check
# 初始化设备处理的函数，返回更新后的 _FSDPState 对象
def _init_device_handle(
    state: _FSDPState,  # 参数：当前 FSDP 状态对象
    module: nn.Module,  # 参数：当前神经网络模块
    ignored_params: Set[nn.Parameter],  # 参数：要忽略的参数集合
    device_id: Optional[Union[int, torch.device]],  # 参数：设备 ID 或设备对象的可选类型
) -> _FSDPState:
    """
    Determine device handle used for initializing FSDP.

    If a device is specified by ``device_id``,
    then returns device handle corresponds to that device type. Otherwise, If the
    module is already on a non-CPU device, then the device type is that non-CPU device type.
    If the module is on CPU or meta, then the device type is the current cuda device.

    This method will be called once ignored paramters was determined, as the device handle maybe needed
    for other initialization.
    """
    determined_device = None  # 初始化变量 determined_device 为 None
    if device_id is not None:  # 如果 device_id 不为空
        determined_device = (
            device_id  # 使用给定的 device_id
            if isinstance(device_id, torch.device)  # 如果 device_id 是 torch.device 类型
            else torch.device(device_id)  # 否则创建 torch.device 对象
        )
    if determined_device is None:  # 如果未确定设备类型
        for param in _get_orig_params(module, ignored_params):  # 遍历原始参数
            if param.device.type in {"cpu", "meta"}:  # 如果参数的设备类型是 CPU 或 meta
                continue  # 跳过当前参数
            if determined_device is None:  # 如果尚未确定设备类型
                determined_device = param.device  # 使用当前参数的设备类型
            else:
                if param.device.type != determined_device.type:  # 如果参数设备类型不同于之前确定的设备类型
                    raise RuntimeError(  # 抛出运行时错误
                        f"FSDP does not support modules with different device types "
                        f"but got params on {determined_device.type} and {param.device.type}"
                    )
        determined_device = determined_device or torch.device(  # 如果未设置确定的设备类型，则默认为当前 cuda 设备
            "cuda", torch.cuda.current_device()
        )

    state._device_handle = _FSDPDeviceHandle.from_device(determined_device)  # 将确定的设备类型转换为设备处理对象
    return state  # 返回更新后的状态对象


@no_type_check
# 初始化缓冲区状态的函数，返回更新后的 _FSDPState 对象
def _init_buffer_state(
    state: _FSDPState,  # 参数：当前 FSDP 状态对象
    module: nn.Module,  # 参数：当前神经网络模块
) -> _FSDPState:
    state._buffer_names = _get_buffer_names(module)  # 获取并存储当前模块的缓冲区名称集合

    # 创建一个映射，将干净的完全限定缓冲区名称（从模块开始）映射到其原始数据类型，
    # 以便在启用缓冲区混合精度时，在模型检查点期间恢复其数据类型。
    _buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}  # 初始化映射字典
    for buffer_name, buffer in module.named_buffers():  # 遍历命名缓冲区
        buffer_name = clean_tensor_name(buffer_name)  # 清理缓冲区名称
        _buffer_name_to_orig_dtype[buffer_name] = buffer.dtype  # 将缓冲区名称映射到其原始数据类型
    state._buffer_name_to_orig_dtype = _buffer_name_to_orig_dtype  # 存储映射字典到状态对象
    return state  # 返回更新后的状态对象


@no_type_check
# 初始化核心状态的函数，返回更新后的 _FSDPState 对象
def _init_core_state(
    state: _FSDPState,  # 参数：当前 FSDP 状态对象
    sharding_strategy: Optional[ShardingStrategy],  # 参数：分片策略的可选类型
    mixed_precision: Optional[MixedPrecision],  # 参数：混合精度的可选类型
    cpu_offload: Optional[CPUOffload],  # 参数：CPU 卸载的可选类型
    limit_all_gathers: bool,  # 参数：限制所有 gather 操作的布尔值
    use_orig_params: bool,  # 参数：使用原始参数的布尔值
    backward_prefetch_limit: int,  # 参数：反向预取限制的整数
    forward_prefetch_limit: int,  # 参数：前向预取限制的整数
) -> _FSDPState:
    # We clamp the strategy to `NO_SHARD` for world size of 1 since they are
    # currently functionally equivalent. This may change if/when we integrate
    # 由于世界大小为 1 时它们当前在功能上是等效的，因此我们将策略夹紧到 `NO_SHARD`。
    # 如果/当我们集成时，这可能会更改。
    # 如果只有一个进程，则不进行分片数据并行（FSDP）
    if state.world_size == 1:
        # 如果之前设置了分片策略，发出警告并切换至不分片策略
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            warnings.warn(
                "FSDP is switching to use `NO_SHARD` instead of "
                f"{sharding_strategy or ShardingStrategy.FULL_SHARD} since "
                "the world size is 1."
            )
        # 设置分片策略为不分片
        sharding_strategy = ShardingStrategy.NO_SHARD
    # 如果不是单进程且分片策略为不分片，则发出警告，推荐使用 `DistributedDataParallel`
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        warnings.warn(
            "The `NO_SHARD` sharding strategy is deprecated. If having issues, "
            "please use `DistributedDataParallel` instead.",
            FutureWarning,
            # 警告堆栈层级：1 - 当前处，2 - 来自 `FullyShardedDataParallel`，3 - 真实调用者
            stacklevel=3,
        )
    # 设置状态对象的分片策略为当前或默认的完全分片策略
    state.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
    # 设置混合精度状态为给定的或默认的混合精度对象
    state.mixed_precision = mixed_precision or MixedPrecision()
    # 如果指定了混合精度，则记录 API 使用情况
    if mixed_precision is not None:
        torch._C._log_api_usage_once(
            f"torch.distributed.fsdp.mixed_precision.{str(state.mixed_precision)}"
        )
    # 设置在评估中是否使用完全精度的环境变量状态
    state._use_full_prec_in_eval = (
        os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
    )
    # 设置 CPU 卸载状态为给定的或默认的 CPU 卸载对象
    state.cpu_offload = cpu_offload or CPUOffload()
    # 设置是否限制所有聚集操作的状态
    state.limit_all_gathers = limit_all_gathers
    # 设置是否使用原始参数的状态
    state._use_orig_params = use_orig_params
    # 设置训练状态为空闲
    state.training_state = TrainingState.IDLE
    # 设置是否为根节点的状态
    state._is_root = None
    # 初始化自由事件队列对象
    state._free_event_queue = _FreeEventQueue()
    # 获取分布式调试级别
    state._debug_level = dist.get_debug_level()
    # 初始化执行顺序数据对象
    state._exec_order_data = exec_order_utils._ExecOrderData(
        state._debug_level,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    # 初始化未分片事件为 None
    state._unshard_event = None
    # 映射完全分片模块到其负责的句柄的字典
    _fully_sharded_module_to_handle: Dict[nn.Module, FlatParamHandle] = dict()
    state._fully_sharded_module_to_handle = _fully_sharded_module_to_handle
    # 状态参数列表
    _handle: FlatParamHandle = None
    state._handle = _handle
    # 返回状态对象
    params: List[FlatParameter] = []
    state.params = params
    return state
@no_type_check
def _init_runtime_state(
    state: _FSDPState,
) -> _FSDPState:
    # 初始化一个空列表，用于存储根前向传递处理器的句柄
    _root_pre_forward_handles: List[RemovableHandle] = []
    state._root_pre_forward_handles = _root_pre_forward_handles
    # 初始化一个空列表，用于存储前向传递处理器的句柄
    _pre_forward_handles: List[RemovableHandle] = []
    state._pre_forward_handles = _pre_forward_handles
    # 初始化一个空列表，用于存储后向传递处理器的句柄
    _post_forward_handles: List[RemovableHandle] = []
    state._post_forward_handles = _post_forward_handles
    # 设置状态中的同步梯度标志为True
    state._sync_gradients = True
    # 设置状态中的通信钩子为None
    state._comm_hook = None
    # 设置状态中的通信钩子状态为None
    state._comm_hook_state = None
    # 用于防止多次运行前向传递前钩子的标志
    return state


@no_type_check
def _init_prefetching_state(
    state: _FSDPState,
    backward_prefetch: BackwardPrefetch,
    forward_prefetch: bool,
) -> _FSDPState:
    # 设置状态中的后向预取标志
    state.backward_prefetch = backward_prefetch
    # 设置状态中的前向预取标志
    state.forward_prefetch = forward_prefetch
    # 数据结构使用句柄元组以通用化模块前向传递涉及多个句柄的情况
    return state


@no_type_check
def _init_extension(state: _FSDPState, device_mesh: DeviceMesh = None) -> _FSDPState:
    # TODO: 一旦支持FSDP + PiPPy，我们需要添加额外的检查。
    # 当前这个检查已经足够，因为我们只支持FSDP + TP。
    if device_mesh and _mesh_resources.get_parent_mesh(state._device_mesh) is not None:
        # 如果设备网格存在并且父网格不为空，初始化FSDP扩展对象
        state._fsdp_extension = DTensorExtensions(state._device_handle)
    else:
        # 否则，显式设置_fsd
        # p_extension为None。
        # 否则，在获取属性时会陷入无限递归。
        state._fsdp_extension = None
    return state


@no_type_check
def _init_state_dict_state(state: _FSDPState) -> _FSDPState:
    # 设置状态中的状态字典类型为完整状态字典
    state._state_dict_type = StateDictType.FULL_STATE_DICT
    # 创建一个完整状态字典配置对象
    state_dict_config: StateDictConfig = FullStateDictConfig()
    state._optim_state_dict_config = FullOptimStateDictConfig()
    # 设置状态中的状态字典配置对象
    state._state_dict_config = state_dict_config
    # 创建一个未分片参数上下文的字典，用于存储模块到生成器的映射
    unshard_params_ctx: Dict[nn.Module, Generator] = {}
    state._unshard_params_ctx = unshard_params_ctx

    return state


@no_type_check
def _init_param_handle_from_module(
    state: _FSDPState,
    fully_sharded_module: nn.Module,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _FSDPState:
    """从模块`fully_sharded_module`初始化一个`FlatParamHandle`。"""
    # 检查单设备模块，确保忽略的参数和设备ID一致
    _check_single_device_module(fully_sharded_module, state._ignored_params, device_id)
    # 根据设备ID获取设备
    device_from_device_id = _get_device_from_device_id(device_id, state.rank)
    # 检查是否需要材料化模块
    is_meta_module, is_torchdistX_deferred_init = _need_to_materialize_module(
        fully_sharded_module, state._ignored_params, state._ignored_modules
    )
    # 如果需要材料化模块，并且提供了参数初始化函数，则使用参数初始化函数进行材料化
    if (is_meta_module or is_torchdistX_deferred_init) and param_init_fn is not None:
        _materialize_with_param_init_fn(
            fully_sharded_module, param_init_fn, state._ignored_modules
        )
    # 如果是元模块，则根据指定的元模块进行实例化，忽略状态中标记的模块
    elif is_meta_module:
        _materialize_meta_module(
            fully_sharded_module, device_id, state._ignored_modules
        )
    # 如果是延迟初始化的 TorchDistX 模块，则实例化该模块
    elif is_torchdistX_deferred_init:
        deferred_init.materialize_module(
            fully_sharded_module,
            # 检查子模块是否未被分片，且不在忽略模块列表中
            check_fn=lambda submodule: _get_module_fsdp_state(submodule) is None
            and submodule not in state._ignored_modules,
        )

    # 收集所有忽略模块中的缓冲区对象，构成集合 ignored_buffers
    ignored_buffers = {
        buffer
        for ignored_module in state._ignored_modules
        for buffer in ignored_module.buffers()
    }

    # 将完全分片的模块移动到目标设备，忽略状态中的参数和收集到的缓冲区对象
    _move_module_to_device(
        fully_sharded_module,
        state._ignored_params,
        ignored_buffers,
        device_from_device_id,
    )

    # 获取完全分片模块的计算设备，考虑状态中的参数忽略列表、源设备 ID 和状态中的排名信息
    state.compute_device = _get_compute_device(
        fully_sharded_module,
        state._ignored_params,
        device_from_device_id,
        state.rank,
    )

    # 获取完全分片模块的原始参数列表，并转换为列表形式
    managed_params = list(_get_orig_params(fully_sharded_module, state._ignored_params))

    # 如果需要同步模块状态，则同步模块的参数和缓冲区
    if sync_module_states:
        _sync_module_params_and_buffers(
            fully_sharded_module, managed_params, state.process_group
        )
        # 如果分片策略属于混合分片策略列表，则再次同步模块的参数和缓冲区
        if state.sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            _sync_module_params_and_buffers(
                fully_sharded_module, managed_params, state._inter_node_pg
            )

    # 从参数列表中初始化参数处理句柄，更新到状态对象中
    _init_param_handle_from_params(state, managed_params, fully_sharded_module)

    # 返回更新后的状态对象
    return state
# 根据给定的状态、参数列表和完全分片模块，初始化参数处理器
@no_type_check
def _init_param_handle_from_params(
    state: _FSDPState,
    params: List[nn.Parameter],
    fully_sharded_module: nn.Module,
):
    # 如果参数列表为空，则直接返回
    if len(params) == 0:
        return
    # 创建一个 FlatParamHandle 对象，用于处理参数
    handle = FlatParamHandle(
        params,
        fully_sharded_module,
        state.compute_device,
        SHARDING_STRATEGY_MAP[state.sharding_strategy],
        state.cpu_offload.offload_params,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.mixed_precision.keep_low_precision_grads,
        state.process_group,
        state._use_orig_params,
        fsdp_extension=state._fsdp_extension,
    )
    # 对参数进行分片处理
    handle.shard()
    # 确保状态对象中没有已存在的处理器
    assert not state._handle
    # 将处理后的平坦参数添加到状态对象的参数列表中
    state.params.append(handle.flat_param)
    # 将处理器对象保存到状态对象中
    state._handle = handle
    # 将完全分片模块及其对应的处理器关联保存到状态对象中
    state._fully_sharded_module_to_handle[handle._fully_sharded_module] = handle
    # 如果启用了 CPU 参数卸载，并且处理后的平坦参数不在 CPU 上，则将其转移到 CPU
    cpu_device = torch.device("cpu")
    if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
        handle.flat_param_to(cpu_device)


# 从根模块中获取被忽略的模块集合
def _get_ignored_modules(
    root_module: nn.Module,
    _ignored_modules: Optional[Iterable[torch.nn.Module]],
) -> Set[nn.Module]:
    """
    Check that ``_ignored_modules`` is an iterable of ``nn.Module`` s without any FSDP instances.

    Return the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
    # 错误消息前缀
    msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
    try:
        # 将用户提供的忽略模块转换为集合，若未提供则为空集合
        ignored_root_modules = (
            set(_ignored_modules) if _ignored_modules is not None else set()
        )
    except TypeError as e:
        # 如果无法转换为集合，则抛出类型错误并提示
        raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}") from e
    # 遍历用户提供的忽略模块集合，确保每个元素都是 nn.Module 类型
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(msg_prefix + f"but got an iterable with {type(module)}")
        # 检查忽略模块中是否包含 FSDP 实例，若包含则抛出值错误
        if _get_module_fsdp_state(module):
            # TODO: We may relax this by taking the FSDP instance's wrapped
            # module to provide more flexibility to the user.
            raise ValueError("`ignored_modules` should not include FSDP modules")
    # 对于根模块中的每一个子模块，如果其不可组合，则视其及其子树为被忽略的模块
    for module in root_module.modules():
        if not traversal_utils._composable(module):
            ignored_root_modules.add(module)
    # 注意：即使 ignored_root_modules 为空，也不要提前返回，以便此 FSDP 实例可以获取其子节点中的任何被忽略模块。

    # 包括子模块，并排除嵌套的 FSDP 模块本身
    ignored_modules = {
        child
        for module in ignored_root_modules
        for child in module.modules()
        if not isinstance(child, fsdp_file.FullyShardedDataParallel)
    }
    # 如果根模块在被忽略的模块列表中，则发出警告
    if root_module in ignored_modules:
        warnings.warn(
            "Trying to ignore the top-level module passed into the FSDP "
            "constructor itself will result in all parameters being "
            f"ignored and is not well-supported: {module}"
        )
    
    # 包括嵌套的 FSDP 模块的被忽略模块
    for submodule in root_module.modules():
        # 获取子模块的 FSDP 状态
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            # 确保 FSDP 状态对象具有 "_ignored_modules" 属性
            assert hasattr(optional_fsdp_state, "_ignored_modules")
            # 更新主忽略模块列表以包括当前子模块的忽略模块
            ignored_modules.update(optional_fsdp_state._ignored_modules)
    
    # 返回更新后的忽略模块列表
    return ignored_modules
def _get_ignored_params(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
    ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
) -> Set[torch.nn.Parameter]:
    """
    返回 ``ignored_modules`` 中模块的参数以及 ``ignored_parameters`` 中的参数。

    :class:`FlatParameter` 被排除在结果之外。
    """
    # 初始化一个空集合，用于存储所有被忽略的参数
    all_ignored_params: Set[torch.nn.Parameter] = set()

    # 遍历所有被忽略的模块，获取其参数，并将非平坦的参数添加到集合中
    params_in_ignored_modules = {
        p for m in ignored_modules for p in m.parameters() if not _is_fsdp_flattened(p)
    }

    # 更新总的被忽略参数集合
    all_ignored_params.update(params_in_ignored_modules)

    # 如果存在被忽略的参数列表，将其非平坦的参数加入集合
    if ignored_parameters is not None:
        params_in_ignored_parameters = {
            p for p in ignored_parameters if not _is_fsdp_flattened(p)
        }
        all_ignored_params.update(params_in_ignored_parameters)

    # 总是包含嵌套 FSDP 模块的被忽略参数
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            assert hasattr(optional_fsdp_state, "_ignored_params")
            all_ignored_params.update(optional_fsdp_state._ignored_params)

    # 返回所有被忽略的参数集合
    return all_ignored_params


def _get_ignored_buffer_names(
    root_module: torch.nn.Module,
    ignored_modules: Set[torch.nn.Module],
) -> Set[str]:
    """
    返回 ``ignored_modules`` 中缓冲区的清理后的全限定名。
    """
    # 初始化一个空集合，用于存储所有被忽略的缓冲区名称
    all_ignored_buffer_names: Set[str] = set()

    # 遍历所有被忽略的模块，获取其所有缓冲区，并将缓冲区名称添加到集合中
    buffers_in_ignored_modules = {
        buffer for m in ignored_modules for buffer in m.buffers()
    }

    # 遍历根模块的所有命名缓冲区，将其清理后的名称添加到集合中
    all_ignored_buffer_names.update(
        {
            clean_tensor_name(buffer_name)
            for buffer_name, buffer in root_module.named_buffers()
            if buffer in buffers_in_ignored_modules
        }
    )

    # 总是包含嵌套 FSDP 模块的被忽略缓冲区名称
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            assert hasattr(optional_fsdp_state, "_ignored_buffer_names")
            all_ignored_buffer_names.update(optional_fsdp_state._ignored_buffer_names)

    # 返回所有被忽略的缓冲区名称集合
    return all_ignored_buffer_names


def _get_buffer_names(root_module: nn.Module) -> Set[str]:
    """
    返回 ``root_module`` 及其模块层次结构中所有缓冲区的全限定名称集合。
    """
    # 使用集合推导式遍历根模块的所有命名缓冲区，并将其清理后的名称添加到集合中
    return {
        clean_tensor_name(buffer_name) for buffer_name, _ in root_module.named_buffers()
    }


def _check_single_device_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_id: Optional[Union[int, torch.device]],
) -> None:
    """
    如果 ``module`` 的原始参数分布在多个设备上（忽略 ``ignored_params`` 中的参数），则引发错误。

    因此，在此方法之后，模块必须完全位于 CPU 或非 CPU 设备上。
    """
    # 从 _get_orig_params 函数获取 module 的原始参数，并生成一个集合 devices
    devices = {param.device for param in _get_orig_params(module, ignored_params)}
    
    # 如果 device_id 不为 None，则允许 module 部分在 CPU 上，部分在 GPU 上；
    # device_id 参数会导致 CPU 部分移动到 GPU 上。这在部分模块可能已经并行化到 GPU 的情况下很有用。
    # 我们希望强制要求 device_id 不为 None，否则会导致在混合模块中展平参数，这是不支持的。
    if len(devices) == 2 and torch.device("cpu") in devices:
        if device_id is None:
            # 如果要支持同时存在 CPU 和 GPU 参数的模块，请传入 device_id 参数。
            raise RuntimeError(
                "To support a module with both CPU and GPU params, "
                "please pass in device_id argument."
            )
    
    # 如果 devices 集合中的设备数量大于 1，抛出运行时错误。
    elif len(devices) > 1:
        # FSDP 仅支持单设备模块，但收到了在 {devices} 上的参数。
        raise RuntimeError(
            f"FSDP only supports single device modules but got params on {devices}"
        )
def _get_device_from_device_id(
    device_id: Optional[Union[int, torch.device]],
    rank: int,
) -> Optional[torch.device]:
    """
    Return a ``torch.device`` for the specified ``device_id``.

    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    # 如果 device_id 为 None，则直接返回 None
    if device_id is None:
        return None
    # 如果 device_id 已经是 torch.device 类型，则直接使用它；否则创建一个 torch.device 对象
    device = (
        device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    )
    # 如果设备是 "cuda"，并且没有明确的设备索引，发出警告并使用当前的 CUDA 设备索引
    if device == torch.device("cuda"):
        warnings.warn(
            f"FSDP got the argument `device_id` {device_id} on rank "
            f"{rank}, which does not have an explicit index. "
            f"FSDP will use the current device {torch.cuda.current_device()}. "
            "If this is incorrect, please explicitly call `torch.cuda.set_device()` "
            "before FSDP initialization or pass in the explicit device "
            "index as the `device_id` argument."
        )
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _need_to_materialize_module(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    ignored_modules: Set[nn.Module],
) -> Tuple[bool, bool]:
    """
    Return if ``module`` has parameters on meta device and if ``module`` is using torchdistX deferred initialization.

    At most of the returned bools can
    be ``True``. If either is ``True``, then ``module`` needs to be
    materialized.
    """
    # 获取模块的原始参数列表（考虑忽略的参数）
    managed_params = list(_get_orig_params(module, ignored_params))
    # 判断模块是否包含 meta 参数
    is_meta_module = any(param.is_meta for param in managed_params)
    
    # TODO: We need to establish a contract for FSDP and buffers. For now, we
    # skip checking for meta buffers from ignored modules. We should consider
    # refactoring the initialization holistically to avoid so many traversals.
    
    # 遍历模块的所有子模块，检查是否包含 meta buffers
    for submodule in module.modules():
        if submodule in ignored_modules:
            continue
        for buf in submodule.buffers(recurse=False):
            is_meta_module |= buf.is_meta
    
    # 判断是否使用了 torchdistX 的延迟初始化，并且没有 meta 参数
    is_torchdistX_deferred_init = (
        not is_meta_module
        and _TORCHDISTX_AVAIL
        and any(fake.is_fake(param) for param in managed_params)
    )
    return is_meta_module, is_torchdistX_deferred_init


def _materialize_with_param_init_fn(
    root_module: nn.Module,
    param_init_fn: Callable[[nn.Module], None],
    ignored_modules: Set[nn.Module],
) -> None:
    """
    Materialize modules using a specified parameter initialization function.

    Raises a ``ValueError`` if ``param_init_fn`` is not callable.
    """
    # 检查参数初始化函数是否可调用
    if not callable(param_init_fn):
        raise ValueError(
            f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
        )
    # 获取需要实例化的模块列表
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    # 对每个需要实例化的模块调用参数初始化函数
    for module in modules_to_materialize:
        param_init_fn(module)


def _materialize_meta_module(
    root_module: nn.Module,
    device_from_device_id: Optional[torch.device],
    ignored_modules: Set[nn.Module],
):
    """
    Run default meta device initialization.

    This function is likely to initialize meta modules on the specified device.
    """
    # 运行默认的 meta 设备初始化
    materialization_device = device_from_device_id or torch.device(
        torch.cuda.current_device()
    )
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    try:
        # 假设每个模块的 `reset_parameters()` 方法只会初始化自己的参数，而不是其子模块的参数
        with torch.no_grad():
            # 遍历所有需要实例化的模块
            for module in modules_to_materialize:
                # 根据约定，仅当模块具有直接管理的参数/缓冲区时才调用 `reset_parameters()`
                module_state_iter = itertools.chain(
                    module.parameters(recurse=False), module.buffers(recurse=False)
                )
                # 检查模块是否有状态参数或缓冲区
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    # 将模块移动到指定的设备上，并且不递归处理子模块
                    module.to_empty(device=materialization_device, recurse=False)
                    module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        # 如果出现异常，发出警告并抛出异常
        warnings.warn(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure that your module of"
            f"type {type(module)} implements a `reset_parameters()` method."  # type: ignore[possibly-undefined]
        )
        raise e
# 定义一个函数 `_get_modules_to_materialize`，用于获取需要调用 `reset_parameters()` 的模块列表
def _get_modules_to_materialize(
    root_module: nn.Module, ignored_modules: Set[nn.Module]
) -> List[nn.Module]:
    # 使用广度优先搜索（BFS）收集需要通过 `reset_parameters()` 来初始化的模块，
    # 在遇到已经应用 FSDP 或者被忽略的模块时停止搜索
    modules_to_materialize: List[nn.Module] = []  # 初始化需要初始化参数的模块列表
    queue = collections.deque([root_module])  # 使用双端队列来存储待处理的模块
    visited_modules: Set[nn.Module] = {root_module}  # 用集合存储已访问过的模块，起到去重作用
    while queue:
        module = queue.popleft()  # 取出队列中的第一个模块
        modules_to_materialize.append(module)  # 将该模块添加到需要初始化参数的列表中
        for child_module in module.children():  # 遍历当前模块的子模块
            if (
                child_module not in visited_modules  # 如果子模块还未被访问过
                and _get_module_fsdp_state(child_module) is None  # 并且子模块没有应用 FSDP
                and child_module not in ignored_modules  # 并且子模块不在忽略列表中
            ):
                visited_modules.add(child_module)  # 将子模块添加到已访问集合中
                queue.append(child_module)  # 将子模块加入队列，继续处理
    return modules_to_materialize  # 返回需要初始化参数的模块列表


# 定义一个函数 `_move_module_to_device`，用于将指定的模块移动到指定的设备
def _move_module_to_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    ignored_buffers: Set[torch.Tensor],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    根据 `device_from_device_id` 和当前设备移动 `module`。

    包括移动被忽略模块的参数。

    - 如果 `device_from_device_id` 不为 `None`，则将 `module` 移动到指定设备。
    - 如果 `device_from_device_id` 为 `None`，则不移动 `module`，但如果在 CPU 上则向用户发出警告。

    前提条件： `_check_single_device_module()`。
    """
    cpu_device = torch.device("cpu")  # 创建一个 CPU 设备对象
    # 如果设备 ID 非空：
    if device_from_device_id is not None:
        # 从 `module` 开始进行广度优先搜索（BFS），但不遍历任何嵌套的 FSDP 实例，
        # 收集尚未管理的参数/缓冲区
        queue: Deque[nn.Module] = collections.deque()
        queue.append(module)
        params: List[nn.Parameter] = []
        buffers: List[torch.Tensor] = []
        while queue:
            curr_module = queue.popleft()
            # 注意：我们增加了一个检查，只移动位于 CPU 设备上的参数/缓冲区。
            # 如果它们位于与 `device_id` 不同的 CUDA 设备上，则不进行移动。
            # 这样可以在 `_get_compute_device()` 中引发错误。
            params.extend(
                param
                for param in curr_module.parameters(recurse=False)
                if param.device == cpu_device
            )
            buffers.extend(
                buffer
                for buffer in curr_module.buffers(recurse=False)
                if buffer.device == cpu_device
            )
            for submodule in curr_module.children():
                # 如果子模块不是 `fsdp_file.FullyShardedDataParallel` 的实例，
                # 则将其加入队列继续遍历
                if not isinstance(submodule, fsdp_file.FullyShardedDataParallel):
                    queue.append(submodule)
        # 筛选出未被忽略的参数和缓冲区
        params_to_move = [p for p in params if p not in ignored_params]
        bufs_to_move = [p for p in buffers if p not in ignored_buffers]
        # 将筛选后的参数和缓冲区移动到设备 `device_from_device_id`
        _move_states_to_device(params_to_move, bufs_to_move, device_from_device_id)
        return
    
    # 否则，获取下一个原始参数，并检查其是否位于 CPU 设备上
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device == cpu_device:
        # 如果是，则发出警告，表明正在初始化 CPU 设备
        _warn_cpu_init()
def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device_from_device_id: Optional[torch.device],
) -> None:
    """
    Move states to the specified device.

    Precondition: ``_check_single_device_module()`` and module's parameters and
    buffers have been materialized if needed.
    """
    # 如果参数和缓冲区都为空，则直接返回
    if len(params) == 0 and len(buffers) == 0:
        return
    # 确定当前设备是参数列表中第一个参数的设备或者缓冲区列表中第一个缓冲区的设备
    if len(params) > 0:
        current_device = params[0].device
    elif len(buffers) > 0:
        current_device = buffers[0].device
    # 创建一个 CPU 设备对象
    cpu_device = torch.device("cpu")
    # 如果指定了device_from_device_id
    if device_from_device_id is not None:
        # 将参数和缓冲区移动到指定的设备，类似于`nn.Module.to()`中的`.data`代码路径
        for param in params:
            with torch.no_grad():
                param.data = param.to(device_from_device_id)
                if param.grad is not None:
                    param.grad.data = param.grad.to(device_from_device_id)
        for buffer in buffers:
            buffer.data = buffer.to(device_from_device_id)
    # 如果当前设备是CPU
    elif current_device == cpu_device:  # type: ignore[possibly-undefined]
        _warn_cpu_init()


def _warn_cpu_init():
    """
    Emit a warning about initializing on CPU.

    Warns that initialization with FSDP on a CPU may be slower compared to GPU.
    """
    warnings.warn(
        "The passed-in `module` is on CPU and will thus have FSDP's sharding "
        "initialization run on CPU, which may be slower than on GPU. We "
        "recommend passing in the `device_id` argument for FSDP to move "
        "`module` to GPU for the sharding initialization. `module` must also "
        "be on GPU device to work with the `sync_module_states=True` flag "
        "since that requires GPU communication."
    )


def _get_compute_device(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
    device_from_device_id: Optional[torch.device],
    rank: int,
) -> torch.device:
    """
    Determine and return this FSDP instance's compute device.

    If a device is specified by ``device_id``, then returns that device. Otherwise, 
    if the module is already on a non-CPU device, then the compute device is that non-CPU
    device. If the module is on CPU, then the compute device is the current device.

    Since this method should be called after materializing the module, any
    non-CPU device should not be meta device. For now, the compute device is
    always a CUDA GPU device with its explicit index.

    Precondition: ``_check_single_device_module()`` and ``_move_module_to_device()``.
    """
    # 获取第一个原始参数
    param = next(_get_orig_params(module, ignored_params), None)
    if param is not None and param.device.type != "cpu":
        # 如果参数的设备不是CPU，则计算设备即为参数的设备
        compute_device = param.device  # Determined by model param placement
    else:
        if device_from_device_id is not None and device_from_device_id.type != "cuda":
            # 如果指定了device_from_device_id，并且它不是CUDA设备，则计算设备为device_from_device_id
            compute_device = device_from_device_id  # Determined by custom backend
        else:
            # 否则计算设备为当前CUDA设备
            compute_device = torch.device("cuda", torch.cuda.current_device())
    # 如果 device_from_device_id 不为 None，并且 compute_device 不等于 device_from_device_id，则抛出数值错误异常
    if device_from_device_id is not None and compute_device != device_from_device_id:
        raise ValueError(
            # 抛出的异常信息，指示计算设备和 device_id 在给定的 rank 上不一致
            f"Inconsistent compute device and `device_id` on rank {rank}: "
            # 显示计算设备和 device_id 的具体数值
            f"{compute_device} vs {device_from_device_id}"
        )
    # 返回 compute_device 的值
    return compute_device
# TODO: See how to deprecate!
def _sync_module_params_and_buffers(
    module: nn.Module,
    params: List[nn.Parameter],
    process_group: dist.ProcessGroup,
) -> None:
    """
    Synchronize module states (i.e. parameters ``params`` and all not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

    Precondition: ``sync_module_states == True`` and ``self.process_group`` has
    been set.
    """
    module_states: List[torch.Tensor] = []

    # Iterate through all buffers of the module
    for buffer in module.buffers():
        # Avoid re-synchronizing buffers in case of nested wrapping
        if not getattr(buffer, FSDP_SYNCED, False):
            setattr(buffer, FSDP_SYNCED, True)
            detached_buffer = buffer.detach()

            # Handle buffers that might be nested wrappers
            if is_traceable_wrapper_subclass(detached_buffer):
                # NOTE: Here we assume no nested subclasses, at most one level of subclass
                # in both model's buffers and params
                attrs, _ = detached_buffer.__tensor_flatten__()  # type: ignore[attr-defined]
                inner_buffers = [getattr(detached_buffer, attr) for attr in attrs]
                module_states.extend(inner_buffers)
            else:
                module_states.append(detached_buffer)

    # Iterate through all parameters specified
    for param in params:
        detached_param = param.detach()

        # Handle parameters that might be nested wrappers
        if is_traceable_wrapper_subclass(detached_param):
            attrs, _ = detached_param.__tensor_flatten__()  # type: ignore[attr-defined]
            inner_params = [getattr(detached_param, attr) for attr in attrs]
            module_states.extend(inner_params)
        else:
            module_states.append(detached_param)

    # Ensure all module states are checked before synchronization
    _check_module_states_for_sync_module_states(module_states)

    # Synchronize parameters and buffers across ranks
    _sync_params_and_buffers(
        process_group,
        module_states,
        PARAM_BROADCAST_BUCKET_SIZE,
        src=0,
    )


def _check_module_states_for_sync_module_states(
    module_states: List[torch.Tensor],
) -> None:
    """
    Check if any module state tensors are on CPU when `sync_module_states=True`,
    which requires them to be on GPU for FSDP.

    Raises:
        ValueError: If any module state tensor is found on CPU.
    """
    if module_states and any(
        tensor.device == torch.device("cpu") for tensor in module_states
    ):
        raise ValueError(
            "The module has CPU parameters or buffers when `sync_module_states=True`, "
            "which requires them to be on GPU. Please specify the `device_id` argument "
            "or move the module to GPU before passing it to FSDP."
        )


def _get_orig_params(
    module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Iterator[nn.Parameter]:
    """
    Return an iterator over the original parameters in ``module``.

    The iterator does not return
    the parameters in ``ignored_params``, any ``FlatParameter`` s (which may be
    present due to nested FSDP wrapping), or any original parameters already
    flattened (only relevant when ``use_orig_params=True``).
    """
    param_gen = module.parameters()

    # Iterate through all parameters in the module
    try:
        while True:
            param = next(param_gen)
            if param not in ignored_params and not _is_fsdp_flattened(param):
                yield param
    except StopIteration:
        # 如果迭代器触发 StopIteration 异常，此处使用 pass 语句来忽略异常，继续执行后续代码
        pass
# 检查已被打平的原始参数是否满足条件
def _check_orig_params_flattened(
    fsdp_module,
    ignored_params: Set[nn.Parameter],
) -> None:
    """
    检查 ``fsdp_module`` 中的原始参数是否已经被打平。

    打平的参数对于根为 ``fsdp_module`` 的模块层次结构在 ``named_parameters()`` 中是不可见的。
    这应该在打平包装模块参数后作为一种健全性检查来调用。
    """
    # 对于每个参数名称和参数对象，使用带重复项的命名参数函数来遍历
    for param_name, param in _named_parameters_with_duplicates(fsdp_module):
        # 如果参数不在忽略列表中并且不是 FSDP 打平的，则抛出运行时错误
        if param not in ignored_params and not _is_fsdp_flattened(param):
            raise RuntimeError(
                f"Found an unflattened parameter: {param_name}; "
                f"{param.size()} {param.__class__}"
            )


def _get_default_comm_hook(sharding_strategy: ShardingStrategy):
    """
    根据分片策略获取默认的通信钩子函数。

    如果分片策略为 ShardingStrategy.NO_SHARD，则返回 allreduce 钩子函数，
    否则返回 reduce_scatter 钩子函数。
    """
    return (
        default_hooks.allreduce_hook
        if sharding_strategy == ShardingStrategy.NO_SHARD
        else default_hooks.reduce_scatter_hook
    )


def _get_default_comm_hook_state(
    process_group: dist.ProcessGroup,
) -> default_hooks.DefaultState:
    """
    获取默认通信钩子状态。

    参数 process_group 指定用于通信的进程组。
    返回一个 default_hooks.DefaultState 对象，其 process_group 属性为给定的进程组。
    """
    return default_hooks.DefaultState(process_group=process_group)
```