# `.\pytorch\torch\distributed\fsdp\_optim_utils.py`

```py
# mypy: allow-untyped-defs
# 引入必要的库和模块
import copy  # 导入 copy 模块，用于对象的浅拷贝和深拷贝操作
import functools  # 导入 functools 模块，用于高阶函数的操作
import logging  # 导入 logging 模块，用于日志记录
import warnings  # 导入 warnings 模块，用于警告控制
from contextlib import ExitStack  # 从 contextlib 模块导入 ExitStack，用于管理上下文的堆栈
from dataclasses import dataclass, field  # 从 dataclasses 模块导入 dataclass 和 field 装饰器，用于定义数据类
from typing import (  # 导入 typing 模块，用于类型注解
    Any,  # 通用类型
    cast,  # 强制类型转换
    Dict,  # 字典类型
    Iterable,  # 可迭代类型
    Iterator,  # 迭代器类型
    List,  # 列表类型
    NamedTuple,  # 命名元组类型
    no_type_check,  # 禁用类型检查
    Optional,  # 可选类型
    Sequence,  # 序列类型
    Set,  # 集合类型
    Tuple,  # 元组类型
    TYPE_CHECKING,  # 类型检查标志
    Union,  # 联合类型
)

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式库
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入 FSDP 库的遍历工具模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._state_dict_utils import _gather_state_dict  # 导入分布式状态字典工具函数
from torch.distributed._tensor import DTensor, Replicate  # 导入分布式张量相关类
from torch.distributed.distributed_c10d import _get_pg_default_device  # 导入获取默认设备函数
from torch.distributed.fsdp._common_utils import (  # 导入 FSDP 公共工具函数
    _apply_to_modules,
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _get_param_to_fqns,
    _module_handle,
    _named_parameters_with_duplicates,
    clean_tensor_name,
)
from torch.distributed.fsdp._debug_utils import SimpleProfiler  # 导入简单分析器类
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle  # 导入平坦参数相关类
from torch.distributed.fsdp._fsdp_extensions import (  # 导入 FSDP 扩展功能
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)
from torch.distributed.fsdp._runtime_utils import (  # 导入 FSDP 运行时工具函数
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (  # 导入 FSDP API 接口
    ShardingStrategy,
    StateDictSettings,
    StateDictType,
)
from torch.utils._pytree import tree_map_only  # 导入 PyTorch 工具模块中的树映射函数

if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor import ShardedTensor  # 如果是类型检查，导入分片张量类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass
class FSDPParamInfo:
    """
    Dataclass that holds information related to FSDP parameters.

    Attributes:
        state (_FSDPState): FSDP state information.
        handle (FlatParamHandle): Flat parameter handle.
        param_indices (Dict[str, int]): Mapping of parameter names to their indices.
        param_requires_grad (List[bool]): List indicating whether each parameter requires gradient.
    """
    state: _FSDPState
    handle: FlatParamHandle
    param_indices: Dict[str, int]
    param_requires_grad: List[bool]


def sorted_items(dictionary: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Generator function that yields sorted key-value pairs from a dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary to iterate over.

    Yields:
        Iterator[Tuple[str, Any]]: Key-value pairs sorted by keys.
    """
    keys = sorted(dictionary.keys())  # 对字典的键进行排序
    for k in keys:
        yield k, dictionary[k]  # 生成排序后的键值对


@dataclass
class _ConsolidatedOptimState:
    """
    Dataclass that holds consolidated optimizer state on the target rank.

    This class handles positive-dimension tensor state communication across ranks,
    zero-dimension tensor state, and non-tensor state directly from the target rank.

    Attributes:
        tensor_state (Dict[str, torch.Tensor]): Mapping from positive-dimension tensor state name
            to the unsharded flat tensor representing the state.
        zero_dim_tensor_state (Dict[str, torch.Tensor]): Mapping from zero-dimension tensor state name
            to its value.
        non_tensor_state (Dict[str, Any]): Mapping from non-tensor state name to its value.
    """

    tensor_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    zero_dim_tensor_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    # 定义一个类属性 non_tensor_state，类型为 Dict，键为字符串，值为任意类型，初始值为空字典
    non_tensor_state: Dict[str, Any] = field(default_factory=dict)
class _PosDimTensorInfo(NamedTuple):
    """
    Meatadata for positive-dimension tensors used internally for
    :meth:`scatter_full_optim_state_dict`.

    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size  # 声明一个名为 shape 的属性，类型为 torch.Size，表示张量的形状
    dtype: torch.dtype  # 声明一个名为 dtype 的属性，类型为 torch.dtype，表示张量的数据类型


class _OptimStateKey(NamedTuple):
    """
    This represents an optimizer state key that may be used commonly across
    ranks. It is based on the unflattened parameter names rather than parameter
    IDs to make it independent of each rank's own optimizer construction.
    """

    unflat_param_names: Tuple[str, ...]  # 声明一个名为 unflat_param_names 的元组属性，包含未展平的参数名称
    is_fsdp_managed: bool  # 声明一个名为 is_fsdp_managed 的布尔属性，表示参数是否由FSDP管理


def _unflatten_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: Dict[str, Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool,
) -> List[Dict[str, Any]]:
    """
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): Entry for the flat parameter in the
            "state" part of the optimizer state dict.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter if on the target
        rank or an empty :class:`list` otherwise. The final optimizer state
        dict will need to map these entries using the proper unflattened
        parameter IDs.
    """
    assert (
        not shard_state or to_save
    ), "If ``shard_state`` is True, ``to_save`` has to be True."
    consolidated_state = _communicate_optim_state(
        fsdp_param_info,
        flat_param_state,
    )  # 调用 _communicate_optim_state 函数，整合优化器状态信息

    if to_save:
        unflat_param_state = _unflatten_communicated_optim_state(
            fsdp_param_info,
            consolidated_state,
            shard_state,
        )  # 调用 _unflatten_communicated_optim_state 函数，对优化器状态进行反展开处理
        for optim_state in unflat_param_state:
            # We can't use .items() below cuz we'd run into a concurrent modification error
            if cpu_offload:
                for key in list(optim_state.keys()):
                    state = optim_state[key]
                    if not isinstance(state, torch.Tensor):
                        continue
                    optim_state[key] = state.cpu()  # 如果 cpu_offload 为 True，则将张量状态移到 CPU 上

        return unflat_param_state  # 返回处理后的优化器状态列表
    else:
        # 如果条件不满足，返回一个空列表
        return []
def _is_zero_dim_tensor(x: Any) -> bool:
    # 检查输入是否为 PyTorch 张量并且维度是否为 0
    return torch.is_tensor(x) and x.dim() == 0


def _communicate_optim_state(
    fsdp_param_info: FSDPParamInfo,
    flat_param_state: Dict[str, Any],
) -> _ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flat parameter across ranks. All
    ranks will hold the entire non-sharded optimizer state on GPU.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flat parameter.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for the target
        flat parameter.
    """
    # 从 fsdp_param_info 中获取 FSDP 的状态
    fsdp_state = fsdp_param_info.state
    # 从 fsdp_param_info 中获取扁平参数
    flat_param = fsdp_param_info.handle.flat_param
    # 创建 _ConsolidatedOptimState 对象来存储整合后的优化器状态
    state = _ConsolidatedOptimState()
    # 将状态分为张量状态、零维张量状态和非张量状态
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )
    for state_name, value in sorted_items(flat_param_state):
        # 对于排序后的扁平化参数状态中的每个状态名和值进行迭代

        # 正维度张量状态：在进程之间进行通信
        if torch.is_tensor(value) and value.dim() > 0:
            # 如果参数没有被分片，正维度张量状态也不会被分片，因此无需通信它 --
            # 我们采用目标进程的值
            if (
                fsdp_state.world_size == 1
                or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
            ):
                # 将正维度张量状态设置为当前值
                tensor_state[state_name] = value
                continue
            
            # 断言确保计算设备已经初始化
            assert (
                fsdp_state.compute_device is not None
            ), "compute_device has not been initialized"
            
            # 如果张量的设备类型与计算设备不同，则将其转移到计算设备上
            if value.device.type != fsdp_state.compute_device.type:
                value = value.to(fsdp_state.compute_device)
            
            # 假设正维度张量优化器状态与分片的扁平参数具有相同的形状
            buffer_size = flat_param._full_param_padded.size()  # type: ignore[attr-defined]
            
            # 创建一个与值相同类型的零张量，形状为扩展后的参数的形状
            tensor_buffer = value.new_zeros(*buffer_size)
            
            # 使用分布式通信收集各进程的值到张量缓冲区中
            dist.all_gather_into_tensor(
                tensor_buffer, value, group=fsdp_state.process_group
            )
            
            # 同步设备句柄，确保通信完成
            fsdp_state._device_handle.synchronize()
            
            # 获取未填充部分的元素数量
            unpadded_numel = cast(
                nn.Parameter, flat_param._unpadded_unsharded_size
            ).numel()
            
            # 将张量缓冲区的内容（不包括填充部分）存入正维度张量状态中
            tensor_state[state_name] = tensor_buffer[:unpadded_numel]

        # 零维张量状态和非张量状态：直接使用本进程的值
        else:
            if _is_zero_dim_tensor(value):
                # 将零维张量状态设置为值的分离克隆版本
                zero_dim_tensor_state[state_name] = value.detach().clone()
            else:
                # 将非张量状态设置为当前值
                non_tensor_state[state_name] = value
    
    # 返回最终的状态字典
    return state
# 定义一个函数，用于将通信的优化器状态展开成单个平坦参数的非平坦版本。这应该只在目标排名上调用。

Args:
    fsdp_param_info (FSDPParamInfo): FSDP状态、句柄，以及从FQN到原始参数索引的映射。
    state (_ConsolidatedOptimState): 合并的优化器状态。

Returns:
    List[Dict[str, Any]]: 包含优化器状态字典中与非平坦参数对应的未展开参数条目的列表。
    最终的优化器状态字典将需要使用适当的未展开参数ID来映射这些条目。
    for _ in range(num_unflat_params):
        unflat_state_param = {}
        # 遍历每个未展开的参数的状态，初始化空字典
        for state_name, flat_tensor in sorted_items(tensor_state):
            # 对于每个状态名称和对应的平坦张量，按名称排序后处理
            views_generated = state_name in flat_param_views
            if not views_generated:
                # 如果尚未生成视图，则调用方法获取未展开的视图
                views = handle._get_unflat_views(flat_tensor)
                flat_param_views[state_name] = views
            else:
                # 否则，使用已有的视图
                views = flat_param_views[state_name]
            # 从视图中获取下一个优化状态（可以是张量、分片张量或者分布式张量）
            optim_state: Union[torch.Tensor, ShardedTensor, DTensor] = next(views)
            if shard_state:
                # 如果启用分片状态
                osd_config = fsdp_state._optim_state_dict_config
                if getattr(osd_config, "_use_dtensor", False):
                    # 如果配置使用了分布式张量
                    assert fsdp_state._device_mesh is not None
                    optim_state = _ext_chunk_dtensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state._device_mesh,
                        fsdp_state._fsdp_extension,
                    )
                else:
                    # 否则使用普通张量
                    assert fsdp_state.process_group is not None
                    optim_state = _ext_chunk_tensor(
                        optim_state,
                        fsdp_state.rank,
                        fsdp_state.world_size,
                        fsdp_state._device_handle.device_count(),
                        fsdp_state.process_group,
                        fsdp_state._fsdp_extension,
                    )
            # 将优化状态存入未展开状态参数字典中
            unflat_state_param[state_name] = optim_state

        # 处理零维张量状态：直接获取目标秩的值
        for state_name, zero_dim_tensor in sorted_items(zero_dim_tensor_state):
            unflat_state_param[state_name] = zero_dim_tensor
        # 处理非张量状态：直接获取目标秩的值
        for state_name, non_tensor in sorted_items(non_tensor_state):
            unflat_state_param[state_name] = non_tensor
        # 将当前未展开状态参数字典添加到结果列表中
        unflat_param_state.append(unflat_state_param)
    # 返回所有未展开参数状态的列表
    return unflat_param_state
# 广播处理后的状态，以确保所有进程具有相同的优化器状态
def _broadcast_processed_state(
    fsdp_state: _FSDPState,
    optim_state: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> Dict[str, Any]:
    objects: List[Any] = [None]  # 创建一个包含单个空元素的列表
    if dist.get_rank(group) == 0:  # 如果当前进程是组中的第一个进程
        # 将优化器状态的所有值映射为 torch.Tensor 类型，将标量转移到 CPU 或保留有关张量的信息
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    # 广播对象列表以在所有进程之间同步数据，源地址为 0
    dist.broadcast_object_list(objects, src=0, group=group)
    if dist.get_rank(group) == 0:  # 如果当前进程是组中的第一个进程
        return optim_state  # 返回未更改的优化器状态
    else:
        return objects[0]  # 返回广播后的优化器状态的第一个元素


# 广播状态，以确保所有进程具有相同的状态
def _broadcast_state(
    fsdp_state: _FSDPState, state: Any, group: Optional[dist.ProcessGroup]
) -> Any:
    if dist.get_rank(group) == 0:  # 如果当前进程是组中的第一个进程
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state  # 如果状态不是张量或是标量张量，则返回状态本身
        tensor = state.to(fsdp_state.compute_device)  # 将张量移动到计算设备上
    else:
        if isinstance(state, torch.Tensor):
            assert state.dim() == 0, (
                "对于非零排名，张量状态应具有零维度，"
                "但是状态形状为 {state.shape()}。"
            )
            return state  # 对于非零排名的进程，返回状态本身
        elif not isinstance(state, _PosDimTensorInfo):
            return state  # 如果状态不是 _PosDimTensorInfo 类型，则返回状态本身
        tensor = torch.zeros(
            state.shape, dtype=state.dtype, device=fsdp_state.compute_device
        )  # 创建与状态形状和数据类型匹配的零张量，并将其移动到计算设备上
    # 广播张量以在所有进程之间同步数据，源地址为 0
    dist.broadcast(tensor, src=0, group=group)
    return tensor  # 返回广播后的张量


# 分片原始参数状态的优化器状态
def _shard_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    fqn: str,
    optim_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    分片具有名称 ``fqn`` 的原始参数的优化器状态。
    仅当 ``use_orig_params`` 为 True 时才应使用此 API。
    """
    if not optim_state:  # 如果优化器状态为空，则返回空字典
        return {}
    fsdp_state = fsdp_param_info.state  # 获取 FSDP 参数信息的状态
    flat_param = fsdp_param_info.handle.flat_param  # 获取 FSDP 参数信息的扁平参数句柄
    param_idx = fsdp_param_info.param_indices[fqn]  # 获取参数在参数索引中的索引位置
    shard_param_info = flat_param._shard_param_infos[param_idx]  # type: ignore[attr-defined]
    # 收集优化器状态字典，根据 fsdp_state 的处理组和计算设备进行收集
    optim_state = _gather_state_dict(
        optim_state, pg=fsdp_state.process_group, device=fsdp_state.compute_device
    )
    if not shard_param_info.in_shard:  # 如果不在分片中，则返回空字典
        return {}
    # 新优化器状态的字典，用于保存分片后的状态
    new_optim_state: Dict[str, Any] = {}
    intra_param_start_idx = shard_param_info.intra_param_start_idx  # 获取分片内参数起始索引
    intra_param_end_idx = shard_param_info.intra_param_end_idx  # 获取分片内参数结束索引
    for state_name, value in optim_state.items():  # 遍历优化器状态字典中的每一个键值对
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            # 如果值是张量且维度大于 0，并且分片策略不是 NO_SHARD，则进行展平和分片处理
            value = value.flatten()[intra_param_start_idx : intra_param_end_idx + 1].clone()  # type: ignore[operator]
        new_optim_state[state_name] = value  # 将处理后的值添加到新优化器状态字典中
    return new_optim_state  # 返回分片后的新优化器状态字典


# 扁平化优化器状态字典，以便于分片处理
def _flatten_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: nn.Module,
    use_orig_params: bool = False,
    optim: Optional[torch.optim.Optimizer] = None,
    rank0_only: bool = False,
):
    group: Optional[dist.ProcessGroup] = None,


# 定义一个名为 `group` 的变量，其类型为 `Optional[dist.ProcessGroup]`，默认值为 `None`
def _flatten_optim_state(
    optim_state_dict: Dict[str, Any],
    model: nn.Module,
    group: Optional[Any] = None,
    rank0_only: bool = False,
    use_orig_params: bool = False
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim`` is KeyedOptimizer or
    NamedOptimizer.

    Args:
        optim_state_dict (Dict[str, Any]): The optimizer state dictionary to flatten.
        model (nn.Module): The model for which the optimizer state is being flattened.
        group (Optional[Any], optional): Optional parameter for group broadcasting. Defaults to None.
        rank0_only (bool, optional): Whether to restrict broadcasting to rank 0 only. Defaults to False.
        use_orig_params (bool, optional): Whether to use original parameters. Defaults to False.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    # Reset profiler for profiling purposes
    SimpleProfiler.reset()

    # Make a copy of the unflattened optimizer state dictionary
    unflat_osd = optim_state_dict

    # Check if "state" key exists in unflat_osd, raise error if not present and rank0_only is False
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError(
            '`optim_state_dict` must have the keys "state"'
            "to be a valid optimizer state dict"
        )

    # Retrieve parameter to fully qualified names mapping
    param_to_fqns = _get_param_to_fqns(model)

    # Retrieve fully qualified name to FSDP parameter information mapping
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)

    # Get the FSDP state from the first item in fqn_to_fsdp_param_info values
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True
    if rank0_only:
        unflat_osd = _broadcast_processed_state(fsdp_state, unflat_osd, group=group)

    # Initialize an empty dictionary to store flattened optimizer state "state" part
    flat_osd_state: Dict[Union[_OptimStateKey, str], Any] = {}

    # Get the unflattened optimizer state "state" part
    unflat_osd_state = unflat_osd["state"]

    # Get all keys from the unflattened optimizer state "state" part
    all_state_keys = set(unflat_osd_state.keys())

    # Handle user-defined state, states that are not associated with parameters
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = _broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)

    # Dump and reset profiler information for FSDP _flatten_optim_state_dict() profiling
    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict() profiling: ")

    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}
    fsdp_param_info: FSDPParamInfo,
    # 定义一个变量 fsdp_param_info，类型为 FSDPParamInfo，可能是一个类或数据结构的实例或引用

    unflat_osd_state: Dict[str, Dict[str, Any]],
    # 定义一个变量 unflat_osd_state，类型为字典，键为字符串，值为字典，值字典的值类型为任意类型

    unflat_param_names: List[str],
    # 定义一个变量 unflat_param_names，类型为列表，列表中的元素为字符串类型
    """
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flat parameter in ``fsdp_param_info`` corresponding to the unflattened
    parameter names in ``unflat_param_names``.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flat parameter ``flat_param``.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flat parameter. The sharded optimizer state dict's "state"
        part will map a key to this returned value.
    """
    # Extract necessary information from the FSDPParamInfo object
    fsdp_state = fsdp_param_info.state
    handle = fsdp_param_info.handle
    flat_param = handle.flat_param

    # Ensure consistency between the number of unflattened parameter names and shapes
    num_unflat_params = len(unflat_param_names)
    assert num_unflat_params > 0, (
        "Expects at least one unflattened parameter corresponding to the "
        "flat parameter"
    )
    unflat_param_shapes = flat_param._shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    assert (
        num_unflat_params == num_unflat_param_shapes
    ), f"Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}"

    # Check if any of the unflattened parameters have optimizer state
    has_state = [
        bool(unflat_param_name in unflat_osd_state)
        for unflat_param_name in unflat_param_names
    ]

    # If none of the unflattened parameters have state, return an empty dictionary
    if not any(has_state):
        return {}

    # Gather optimizer state for each unflattened parameter that has state
    unflat_param_states = [
        _gather_state_dict(
            unflat_osd_state[unflat_param_name],
            pg=fsdp_state.process_group,
            device=fsdp_state.compute_device,
        )
        if unflat_param_name in unflat_osd_state
        else None
        for unflat_param_name in unflat_param_names
    ]

    # Ensure all unflattened parameters have the same optimizer state names
    state_names = None
    for unflat_param_state in unflat_param_states:
        if unflat_param_state is None:
            continue
        if state_names is None:
            state_names = set(unflat_param_state.keys())
        else:
            if state_names != set(unflat_param_state.keys()):
                raise ValueError(
                    "Differing optimizer state names for the unflattened "
                    f"parameters: {unflat_param_names}"
                )
    assert state_names is not None

    # Initialize the flattened state dictionary
    flat_state: Dict[str, Any] = {}
    # 遍历状态名称列表，依次处理每个状态名称
    for state_name in state_names:
        # 获取当前状态名称在所有未展平参数状态字典中的对应值列表
        state_values = [
            unflat_param_state[state_name] if unflat_param_state is not None else None
            for unflat_param_state in unflat_param_states
        ]
        
        # 过滤掉状态值列表中的 None 值
        non_none_state_values = [v for v in state_values if v is not None]
        
        # 如果所有的状态值都为 None，则将展平状态字典中对应的值设为 None，并继续下一个状态名称的处理
        if not non_none_state_values:
            flat_state[state_name] = None
            continue
        
        # 初始化类型判断标志
        are_pos_dim_tensors = are_zero_dim_tensors = are_non_tensors = True
        
        # 遍历非 None 状态值列表，判断类型并更新类型判断标志
        for v in non_none_state_values:
            are_pos_dim_tensors &= torch.is_tensor(v) and v.dim() > 0
            are_zero_dim_tensors &= _is_zero_dim_tensor(v)
            are_non_tensors &= not torch.is_tensor(v)
        
        # 获取非 None 状态值列表中的数据类型集合
        types = {type(v) for v in non_none_state_values}
        
        # 如果状态值类型不唯一或者类型与张量属性不符合，则抛出 ValueError 异常
        if len(types) != 1 or not (
            are_pos_dim_tensors or are_zero_dim_tensors or are_non_tensors
        ):
            raise ValueError(
                f"Differing optimizer state types for state {state_name}, "
                f"values {non_none_state_values}, and unflattened parameter "
                f"names {unflat_param_names}"
            )
        
        # 根据不同的状态值类型进行展平处理并更新展平状态字典
        if are_pos_dim_tensors:
            flat_tensor = _flatten_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
                unflat_param_shapes,
                handle,
            )
            
            # 如果分布式参数设置不是单节点，并且分片策略不是 NO_SHARD，则对展平后的张量进行分片处理
            if (
                fsdp_state.world_size != 1
                and fsdp_state.sharding_strategy != ShardingStrategy.NO_SHARD
            ):
                sharded_flat_tensor, _ = FlatParamHandle._get_shard(
                    flat_tensor,
                    fsdp_state.rank,
                    fsdp_state.world_size,
                )
            else:
                sharded_flat_tensor = flat_tensor
            
            # 更新展平状态字典中的状态名称对应值为分片后的展平张量
            flat_state[state_name] = sharded_flat_tensor
        elif are_zero_dim_tensors:
            # 如果状态值为零维张量，则进行零维张量的展平处理
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )
        else:
            # 否则，处理非张量状态值的展平
            assert are_non_tensors
            flat_state[state_name] = _flatten_non_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )
    
    # 返回处理后的展平状态字典
    return flat_state
    """
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flat parameter
    from ``handle`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter's tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam's "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flat parameter.
        handle (FlatParamHandle): The flat parameter's handle.

    Returns:
        torch.Tensor: A flat tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    """
    # Extract the flat parameter from the handle
    flat_param = handle.flat_param
    # Filter out tensors that are not None (i.e., tensors with actual state)
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    # Check that all tensors have the same data type
    dtypes = {t.dtype for t in non_none_tensors}
    if len(dtypes) != 1:
        # Raise an error if tensors have different data types
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have positive-dimension tensor state with the "
            f"same dtype but got dtypes {dtypes} for state {state_name} and "
            f"unflattened parameter names {unflat_param_names}"
        )
    # Get the data type (since all tensors have the same dtype)
    dtype = next(iter(dtypes))
    # Check that each tensor state matches its corresponding parameter's shape
    for tensor, shape in zip(pos_dim_tensors, unflat_param_shapes):
        if tensor is None and len(shape) == 0:
            # Raise an error if trying to flatten a zero-dimension parameter
            raise ValueError("Flattening a zero-dimension parameter is not supported")
        elif tensor is not None and tensor.shape != shape:
            # Raise an error if tensor state shape doesn't match parameter shape
            raise ValueError(
                "Tensor optimizer state does not have same shape as its "
                f"parameter: {tensor.shape} {shape}"
            )
    # Flatten the tensor states into a single tensor, concatenating them
    # Note: No right-hand-side is added here as it is done implicitly
    # 创建一个 CPU 设备对象，用于将数据移到 CPU 上进行处理
    cpu_device = torch.device("cpu")
    
    # 对每个状态值进行扁平化处理，如果状态值不为 None，则将其移到 CPU 上进行处理；
    # 如果状态值为 None，则创建一个与指定形状、数据类型和设备相关的全零张量进行处理
    tensors_to_flatten = [
        torch.flatten(state_value.to(cpu_device))
        if state_value is not None
        else torch.flatten(
            torch.zeros(
                size=shape,
                dtype=dtype,
                device=cpu_device,
            )
        )
        for state_value, shape in zip(pos_dim_tensors, unflat_param_shapes)
    ]
    
    # 使用给定的处理函数 handle.flatten_tensors 对 tensors_to_flatten 中的张量进行扁平化处理，
    # 并按照 handle._aligned_numel 的值对扁平化后的张量进行处理
    flat_tensor = handle.flatten_tensors(tensors_to_flatten, handle._aligned_numel)
    
    # 获取扁平化参数的形状，这里的 flat_param 是一个对象，取其未填充未分片大小的属性值
    flat_param_shape = flat_param._unpadded_unsharded_size  # type: ignore[attr-defined]
    
    # 检查扁平化后的张量形状与扁平化参数的形状是否相等，若不相等则抛出断言错误并输出详细信息
    assert flat_tensor.shape == flat_param_shape, (
        f"tensor optim state: {flat_tensor.shape} "
        f"flat parameter: {flat_param_shape}"
    )
    
    # 返回处理后的扁平化张量作为函数的结果
    return flat_tensor
# 将零维张量优化器状态展平
def _flatten_zero_dim_tensor_optim_state(
    state_name: str,
    zero_dim_tensors: List[torch.Tensor],
    unflat_param_names: List[str],
) -> torch.Tensor:
    """
    根据给定的零维张量 `zero_dim_tensors`，展平优化器状态 `state_name` 对应的单一扁平参数，
    这些参数对应于未展平的参数名称 `unflat_param_names`，通过强制所有张量具有相同的值来实现。

    注意：所有未展平参数都必须具有相同的值和数据类型，以保持 FSDP 与其非分片等效的计算一致性。
    这意味着不能缺少任何未展平的参数状态，因为强加一个值可能与没有值不同。
    例如，对于 Adam 算法的 "step"，没有值意味着最大的偏差修正，而有些正值则表示较少的偏差修正。

    Args:
        state_name (str): 优化器状态名称。
        zero_dim_tensors (List[torch.Tensor]): 未展平参数对应的零维张量优化器状态列表，用于单一扁平参数。
        unflat_param_names (List[str]): 单一扁平参数对应的未展平参数名称列表。

    Returns:
        torch.Tensor: 给出状态 `state_name` 对应于名称 `unflat_param_names` 的所有未展平参数的值的零维张量。
    """
    non_none_tensors = [t for t in zero_dim_tensors if t is not None]
    # 强制所有张量具有相同的值和数据类型
    values_set = {t.item() if t is not None else None for t in zero_dim_tensors}
    dtypes = {t.dtype if t is not None else None for t in zero_dim_tensors}
    if (
        len(non_none_tensors) != len(zero_dim_tensors)
        or len(values_set) != 1
        or len(dtypes) != 1
    ):
        raise ValueError(
            "所有构成单一扁平参数的未展平参数必须具有相同的标量状态，具有相同的值和数据类型，"
            f"但是得到的值为 {values_set}，数据类型为 {dtypes}，状态为 {state_name}，未展平参数名称为 {unflat_param_names}"
        )
    value = next(iter(values_set))
    dtype = next(iter(dtypes))
    return torch.tensor(value, dtype=dtype, device=torch.device("cpu"))


def _flatten_non_tensor_optim_state(
    state_name: str,
    non_tensors: List[Any],
    unflat_param_names: List[str],
) -> Any:
    """
    将由值 `non_tensors` 给出的非张量优化器状态展平，用于状态 `state_name` 对应的单一扁平参数，
    这些参数对应于未展平的参数名称 `unflat_param_names`，通过强制所有值具有相同的值来实现。

    参见 :func:`_flatten_zero_dim_tensor_optim_state` 中的说明。
    """
    Args:
        state_name (str): Optimizer state name.  # 输入参数，优化器状态的名称
        non_tensors (List[Any]): Non-tensor optimizer state for the unflattened
            parameters corresponding to the single flat parameter.  # 输入参数，未展平参数对应的非张量优化器状态列表
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.  # 输入参数，未展平参数的名称列表，对应于单个展平参数

    Returns:
        Any: A non-tensor giving the value of the state ``state_name`` for all
        unflattened parameters corresponding to the names
        ``unflat_param_names``.  # 返回值，返回一个非张量，表示所有未展平参数对应名称为 unflat_param_names 的状态 state_name 的值
    """
    non_none_non_tensors = [nt for nt in non_tensors if nt is not None]
    # Enforce that all have the same value (same type already checked)
    non_tensor_set = set(non_tensors)
    if len(non_none_non_tensors) != len(non_tensors) or len(non_tensor_set) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flat "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {non_tensor_set} for state {state_name} and  "
            f"unflattened parameter names {unflat_param_names}"
        )
    non_tensor = next(iter(non_tensor_set))
    return non_tensor


注释：
- `non_none_non_tensors`: 过滤掉非空值的非张量优化器状态列表。
- `non_tensor_set`: 将 `non_tensors` 转换为集合，用于检查所有状态值是否相同。
- 引发 `ValueError` 异常：如果未展平参数的所有状态值不相同，抛出异常，并显示相关的状态名称和参数名称。
def _rekey_sharded_optim_state_dict(
    sharded_osd: Dict[str, Any],  # 输入参数：分片优化器状态字典
    model: nn.Module,  # 输入参数：神经网络模型
    optim: torch.optim.Optimizer,  # 输入参数：优化器对象
    optim_input: Optional[  # 输入参数：优化器的输入，可以是字典列表或参数可迭代对象的联合类型，可选
        Union[
            List[Dict[str, Any]],  # 可能的类型：字典列表
            Iterable[nn.Parameter],  # 可能的类型：参数可迭代对象
        ]
    ],
    using_optim_input: bool,  # 输入参数：是否使用优化器输入标志
    is_named_optimizer: bool = False,  # 输入参数：是否为命名优化器，默认为False
) -> Dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to flat
    parameter IDs according to the calling rank's ``optim``, which may be
    different across ranks. In particular, the unflattened parameter names are
    represented as :class:`_OptimStateKey` s.
    """
    param_to_fqns = _get_param_to_fqns(model)  # 获得模型参数到完全限定名称的映射
    flat_param_to_fqn = _get_flat_param_to_fqn(model)  # 获得模型参数到平坦参数名称的映射
    param_to_param_key: Dict[nn.Parameter, Union[int, str]] = cast(
        Dict[nn.Parameter, Union[int, str]],
        (
            _get_param_to_param_id_from_optim_input(model, optim_input)
            if using_optim_input
            else _get_param_to_param_key(
                optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
            )
        ),
    )
    # 所有参数键在 `param_to_param_key` 中都应该在 `param_to_fqns` 中 --
    # 当没有将所有参数传递给优化器时，使用严格的不等式
    assert len(param_to_param_key) <= len(param_to_fqns)

    unflat_param_names_to_flat_param_key: Dict[
        Tuple[str, ...], Union[int, str]
    ] = {}  # 用于 "state" 的未平坦参数名称到平坦参数键的映射
    unflat_param_name_to_flat_param_key: Dict[
        str, Union[int, str]
    ] = {}  # 用于 "param_groups" 的未平坦参数名称到平坦参数键的映射
    for param, unflat_param_names in param_to_fqns.items():
        if param not in param_to_param_key:
            # 此参数未传递给优化器
            continue
        flat_param_key = param_to_param_key[param]
        unflat_param_names_to_flat_param_key[tuple(unflat_param_names)] = flat_param_key
        for unflat_param_name in unflat_param_names:
            unflat_param_name_to_flat_param_key[unflat_param_name] = flat_param_key

    sharded_osd_state = sharded_osd["state"]  # 获取分片优化器状态字典中的 "state" 部分
    rekeyed_osd_state: Dict[Union[str, int], Any] = {}  # 重新键入后的优化器状态字典
    for key, param_state in sharded_osd_state.items():
        if isinstance(key, str):
            rekeyed_osd_state[key] = param_state
            continue
        flat_param_key = unflat_param_names_to_flat_param_key.get(
            key.unflat_param_names, key.unflat_param_names
        )
        rekeyed_osd_state[flat_param_key] = param_state

    # 只有在分片优化器状态字典中存在 "param_groups" 时才处理 param_groups
    # 如果在 sharded_osd 中存在 "param_groups" 键
    if "param_groups" in sharded_osd:
        # 初始化重新调整键名后的 OSD 参数组列表
        rekeyed_osd_param_groups: List[Dict[str, Any]] = []
        
        # 遍历 sharded_osd["param_groups"] 中的每个未扁平化的参数组
        for unflat_param_group in sharded_osd["param_groups"]:
            # 深拷贝未扁平化的参数组，以防止原始数据受影响
            flat_param_group = copy.deepcopy(unflat_param_group)
            
            # 获取未扁平化参数组中所有参数的扁平化键名，并排序
            flat_param_keys = sorted(
                {
                    unflat_param_name_to_flat_param_key[unflat_param_name]
                    for unflat_param_name in unflat_param_group["params"]
                }
            )
            
            # 将扁平化后的参数键名列表赋值给 flat_param_group 的 "params" 键
            flat_param_group["params"] = flat_param_keys
            
            # 将重新调整键名后的参数组添加到 rekeyed_osd_param_groups 列表中
            rekeyed_osd_param_groups.append(flat_param_group)
        
        # 返回一个字典，包含重新调整键名后的 OSD 状态和参数组列表
        return {"state": rekeyed_osd_state, "param_groups": rekeyed_osd_param_groups}
    
    # 如果 sharded_osd 中不存在 "param_groups" 键，则返回只包含重新调整键名后的 OSD 状态的字典
    else:
        return {"state": rekeyed_osd_state}
    """
    构建从参数ID到参数的映射。这可以用于既有`FlatParameter`也有没有的模型。

    注意：此方法仅保留用于向后兼容性。推荐使用的代码路径是:meth:`_get_param_key_to_param`，它不依赖于`optim_input`。

    注意：我们重要地假设，无论优化器输入是参数列表还是参数组列表，`torch.optim.Optimizer`都按顺序枚举参数ID。换句话说，对于参数列表输入，参数ID应按列表顺序排列；对于参数组输入，参数ID应在每个参数组内按顺序排列，并在参数组之间也按顺序排列。

    Args:
        model (nn.Module): 将其参数传递给优化器的模型。
        optim_input (Optional[Union[List[Dict[str, Any]], Iterable[nn.Parameter]]]): 传递给优化器的输入，表示参数组的`list`或参数的可迭代对象；如果为`None`，则此方法假定输入为`model.parameters()`。（默认为`None`）

    Returns:
        List[nn.Parameter]: 从参数ID到参数的映射，其中参数ID隐式地是`list`中的索引。
    """
    # 如果未指定`optim_input`，假定通常情况下将`model.parameters()`传递给优化器
    if optim_input is None:
        return dict(enumerate(model.parameters()))

    try:
        params = cast(List[nn.Parameter], list(optim_input))
    except TypeError as e:
        raise TypeError(
            "Optimizer input should be an iterable of Tensors or dicts, "
            f"but got {optim_input}"
        ) from e

    if len(params) == 0:
        raise ValueError("Optimizer input should not be empty")

    # 检查优化器输入表示的是张量还是参数组
    all_tensors = True
    all_dicts = True
    for param in params:
        all_tensors &= isinstance(param, torch.Tensor)
        all_dicts &= isinstance(param, dict)

    if not all_tensors and not all_dicts:
        raise TypeError("Optimizer input should be an iterable of Tensors or dicts")

    if all_tensors:
        return dict(enumerate(params))

    assert all_dicts
    param_id_to_param: List[nn.Parameter] = []
    # 遍历参数组列表，每次迭代取出一个参数组
    for param_group in params:
        # 检查当前参数组是否包含键名为 "params"
        has_params_key = "params" in param_group  # type: ignore[operator]
        # 断言当前参数组必须包含 "params" 键，如果不包含则抛出异常
        assert has_params_key, (
            'A parameter group should map "params" to a list of the '
            "parameters in the group"
        )
        # 将当前参数组中的 "params" 值（即参数列表）添加到 param_id_to_param 列表末尾
        # 这里假设 param_id_to_param 是一个列表，用来存储所有参数
        param_id_to_param.extend(param_group["params"])  # type: ignore[index]
    # 返回一个字典，将 param_id_to_param 列表的索引作为键，列表中的元素作为值
    return dict(enumerate(param_id_to_param))
# 定义一个函数，用于构建从 FlatParameter 到其清理后的完全限定名（FQN）的映射字典
def _get_flat_param_to_fqn(model: torch.nn.Module) -> Dict[FlatParameter, str]:
    """
    Constructs a mapping from ``FlatParameter`` to a cleaned (devoid of prefixes
    from wrappers) fully qualified name (FQN). Note that this FQN is "non-canonical"
    because ``FlatParameter`` s do not come from the original module but are
    registered only after FSDP has been applied. This function returns the FSDP-given
    name for the ``FlatParameter`` (usually module._flat_param) as opposed to the
    canonical FQNs returned for ``FlatParameter`` s in ``_common_utils._get_param_to_fqns(...)``).

    Consequently, this function will only return a non-empty mapping if FSDP was
    applied with ``use_orig_params=False`` as, otherwise, the original parameters
    are used within the module and there would be no ``FlatParameter`` s in the module.

    """

    # 定义一个内部函数，用于遍历模块的参数并生成 FQN 映射
    def module_fn(module, prefix, tree_level, flat_param_to_fqn):
        # 遍历模块中命名的参数，不递归
        for param_name, param in _named_parameters_with_duplicates(
            module, recurse=False
        ):
            # 如果参数不是 FlatParameter 类型，则跳过
            if not isinstance(param, FlatParameter):
                continue
            # 清理后的参数名称生成 FQN
            fqn = clean_tensor_name(prefix + param_name)
            flat_param_to_fqn[param] = fqn

    # 定义一个返回函数，返回 FlatParameter 到 FQN 的映射字典
    def return_fn(flat_param_to_fqn):
        return flat_param_to_fqn

    # 初始化一个空的 FlatParameter 到 FQN 映射字典
    flat_param_to_fqn_ret: Dict[FlatParameter, str] = {}

    # 调用 _apply_to_modules 函数，将 module_fn 和 return_fn 应用于 model
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [fqn for fqn, _ in _named_parameters_with_duplicates(model)],
        flat_param_to_fqn_ret,
    )


# 定义一个函数，用于构建从参数键到参数本身的映射字典，适用于正常优化器和 NamedOptimizer
def _get_param_key_to_param(
    optim: torch.optim.Optimizer,
    model: Optional[nn.Module] = None,
    is_named_optimizer: bool = False,
    param_to_fqns: Optional[Dict[nn.Parameter, List[str]]] = None,
    flat_param_to_fqn: Optional[Dict[FlatParameter, str]] = None,
) -> Dict[Union[int, str], nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """

    # 初始化一个空的清理后 FQN 到当前 FQN 的映射字典
    clean_fqn_to_curr_fqn: Dict[str, str] = {}

    # 如果是 NamedOptimizer，则必须提供 param_to_fqns 和 flat_param_to_fqn 参数
    if is_named_optimizer:
        assert (
            param_to_fqns is not None and flat_param_to_fqn is not None
        ), "The optimizer is a NamedOptimizer, `param_to_fqns` must not be None."
        assert model is not None
        # 遍历模块中命名的参数，并生成清理后 FQN 到当前 FQN 的映射
        for key, _ in _named_parameters_with_duplicates(model):
            clean_fqn_to_curr_fqn[clean_tensor_name(key)] = key

    # 初始化一个空的参数键到参数本身的映射字典
    param_key_to_param: Dict[Union[str, int], nn.Parameter] = {}
    # 初始化参数 ID
    pid = 0
    # 遍历优化器参数组列表
    for param_group in optim.param_groups:
        # 如果优化器参数组有命名优化器的情况
        if is_named_optimizer:
            # 遍历当前参数组的参数列表
            for param in param_group["params"]:
                # 断言确保存在 flat_param_to_fqn 映射关系
                assert flat_param_to_fqn is not None
                # 如果当前参数在 flat_param_to_fqn 中有映射
                if param in flat_param_to_fqn:
                    # FlatParameter 情况，取得其对应的键
                    key = flat_param_to_fqn[param]
                else:
                    # 如果不存在映射，则假设 param_to_fqns 不为空
                    assert param_to_fqns is not None
                    # use_orig_params 情况，断言当前参数只有一个全限定名
                    assert len(param_to_fqns[param]) == 1
                    # 取得唯一的全限定名
                    key = param_to_fqns[param][0]
                # 尝试从 clean_fqn_to_curr_fqn 中获取当前键的最新映射
                try:
                    key = clean_fqn_to_curr_fqn[key]
                # 如果找不到对应的映射，引发 KeyError 异常
                except KeyError as e:
                    raise KeyError(
                        f"Can't find {key} from {list(clean_fqn_to_curr_fqn.keys())}."
                    ) from e
                # 将当前参数与其键对应关系存入 param_key_to_param 字典
                param_key_to_param[key] = param
        else:
            # 如果不是命名优化器，直接遍历当前参数组的参数列表
            for param in param_group["params"]:
                # 将参数与递增的 pid 对应关系存入 param_key_to_param 字典
                param_key_to_param[pid] = param
                pid += 1

    # 返回参数键到参数对象的映射关系字典
    return param_key_to_param
def _get_param_to_param_key(
    optim: torch.optim.Optimizer,
    model: Optional[nn.Module] = None,
    is_named_optimizer: bool = False,
    param_to_fqns: Optional[Dict[nn.Parameter, List[str]]] = None,
    flat_param_to_fqn: Optional[Dict[FlatParameter, str]] = None,
) -> Dict[nn.Parameter, Union[int, str]]:
    """
    Constructs the inverse mapping of :func:`_get_param_key_to_param`. This API
    only supports the case where `optim` is a regular optimizer, not NamedOptimizer.
    So the parameter keys will be parameter ids.
    """
    # 获取参数键到参数对象的映射
    param_id_to_param = _get_param_key_to_param(
        optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
    )
    # 构建参数对象到参数键的反向映射，返回结果字典
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _get_param_to_param_id_from_optim_input(
    model: nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ] = None,
) -> Dict[nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param_from_optim_input`."""
    # 获取参数键到参数对象的映射
    param_id_to_param = _get_param_id_to_param_from_optim_input(model, optim_input)
    # 构建参数对象到参数 id 的反向映射，返回结果字典
    return {param: param_id for param_id, param in param_id_to_param.items()}


def _check_missing_keys_on_rank(
    r0_optim_state_keys: List[_OptimStateKey],
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[str, int]],
    param_key_to_param: Dict[Union[str, int], nn.Parameter],
    group: Optional[dist.ProcessGroup],
) -> None:
    # Ensure that all ranks have at least the optimizer states needed by
    # rank 0's optimizer
    missing_keys: List[_OptimStateKey] = []
    # 遍历 rank 0 的优化器状态键列表
    for r0_optim_state_key in r0_optim_state_keys:
        # 检查当前状态键是否在优化器状态键到参数键的映射中
        if r0_optim_state_key not in optim_state_key_to_param_key:
            # 如果在当前 rank 的优化器状态中找不到 rank 0 的优化器状态键，记录为缺失键
            missing_keys.append(r0_optim_state_key)
            continue
        # 从映射中获取参数键
        param_key = optim_state_key_to_param_key[r0_optim_state_key]
        # 如果参数键是整数类型，确保其在有效范围内
        if isinstance(param_key, int):
            assert param_key >= 0 and param_key < len(
                param_key_to_param
            ), "Check the `param_key_to_param` construction"
    # 获取默认进程组的设备
    device = _get_pg_default_device(group)
    # 创建包含缺失键数量的张量
    num_missing = torch.tensor([len(missing_keys)], dtype=torch.int32, device=device)
    # 在进程组中进行全局缺失键数量的归约操作
    dist.all_reduce(num_missing, group=group)
    # 如果缺失的项目数量大于零，则执行以下逻辑
    obj_list = [None for _ in range(dist.get_world_size(group))]
    # 创建一个与进程数相同长度的空列表 obj_list
    dist.all_gather_object(obj_list, missing_keys, group=group)
    # 使用分布式工具将 missing_keys 收集到 obj_list 中，以确保所有进程都能访问它们
    error_msg = (
        "FSDP currently requires each rank to have at least the "
        "optimizer states needed by rank 0's optimizer but some ranks "
        "are missing some of those states"
    )
    # 设置错误消息的初始内容，描述缺失状态的问题
    for rank, keys in enumerate(obj_list):
        # 遍历 obj_list 中的每个进程及其对应的 keys
        keys = cast(List[_OptimStateKey], keys)
        # 将 keys 强制转换为 _OptimStateKey 类型的列表
        if len(keys) > 0:
            # 如果 keys 列表不为空，则说明存在缺失的参数状态
            error_msg += (
                f"\nRank {rank} is missing states for the parameters: "
                f"{[key.unflat_param_names for key in keys]}"
            )
            # 在错误消息中添加具体描述，指明缺失状态的进程及其对应的参数名称
    # 抛出运行时错误，其中包含详细的错误消息
    raise RuntimeError(error_msg)
def _map_param_key_to_optim_keys(
    optim_state_dict: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
    param_key_to_param: Dict[Union[int, str], nn.Parameter],
    param_to_fqns: Dict[nn.Parameter, List[str]],
    fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
    merge_keys: bool = False,
) -> Tuple[List[_OptimStateKey], Dict[_OptimStateKey, Union[int, str]]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
    # 获取当前进程的rank
    rank = dist.get_rank(group)
    
    # 本地优化状态键到参数键的映射字典
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]] = {}  # local
    
    # 所有优化状态键的列表
    all_optim_state_keys: List[_OptimStateKey] = []

    # 遍历参数键和参数的字典
    for param_key, param in param_key_to_param.items():
        # 如果参数键不在优化器状态字典的"state"部分中，则跳过该参数
        if param_key not in optim_state_dict["state"]:
            continue
        
        # 获取参数对应的全限定名列表
        fqns = param_to_fqns[param]
        
        # 检查参数是否由FSDP管理
        is_fsdp_managed = isinstance(param, FlatParameter)
        if is_fsdp_managed:
            # 断言该参数的第一个全限定名在FSDP参数信息字典中
            assert fqns[0] in fqn_to_fsdp_param_info, (
                fqns[0],
                list(fqn_to_fsdp_param_info.keys()),
            )
        
        # 判断参数是否由FSDP管理
        is_fsdp_managed = fqns[0] in fqn_to_fsdp_param_info
        
        # 创建优化状态键对象
        optim_state_key = _OptimStateKey(
            unflat_param_names=tuple(fqns),
            is_fsdp_managed=is_fsdp_managed,
        )
        
        # 如果当前进程是rank0或者merge_keys为True，则将优化状态键添加到列表中
        if rank == 0 or merge_keys:
            all_optim_state_keys.append(optim_state_key)
        
        # 将优化状态键与参数键建立映射关系
        optim_state_key_to_param_key[optim_state_key] = param_key

    # 如果merge_keys为True，则进行所有进程之间的优化状态键合并
    if merge_keys:
        # 创建用于收集所有优化状态键的列表
        all_keys: List[List[_OptimStateKey]] = [
            [] for _ in range(dist.get_world_size(group))
        ]
        # 在所有进程间收集所有优化状态键
        dist.all_gather_object(all_keys, all_optim_state_keys, group=group)
        # 合并所有优化状态键并去重排序
        merge_all_optim_state_keys = [
            key for local_keys in all_keys for key in local_keys
        ]
        all_optim_state_keys = sorted(set(merge_all_optim_state_keys))
    else:
        # 创建用于广播优化状态键列表的对象列表
        key_obj_list: List[Optional[List[_OptimStateKey]]] = (
            [all_optim_state_keys] if rank == 0 else [None]
        )
        # 广播优化状态键列表
        dist.broadcast_object_list(key_obj_list, src=0, group=group)
        # 断言key_obj_list的第一个元素不为None
        assert key_obj_list[0] is not None
        # 获取广播后的优化状态键列表
        all_optim_state_keys = key_obj_list[0]
        
        # 检查rank上缺失的优化状态键
        _check_missing_keys_on_rank(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            param_key_to_param,
            group,
        )

    # 返回所有优化状态键和优化状态键到参数键的映射字典
    return all_optim_state_keys, optim_state_key_to_param_key


def _unflatten_param_groups(
    state_dict: Dict[str, Any],
    param_key_to_param: Dict[Union[int, str], nn.Parameter],
    param_to_fqns: Dict[nn.Parameter, List[str]],
) -> List[Dict[str, Any]]:
    # 参数组列表
    param_groups: List[Dict[str, Any]] = []
    # 遍历状态字典中的参数组列表
    for flat_param_group in state_dict["param_groups"]:
        # 深拷贝当前参数组，以确保不修改原始数据
        unflat_param_group = copy.deepcopy(flat_param_group)
        
        # 获取当前参数组的参数键列表，并转换为参数对象列表
        param_group_params = [
            param_key_to_param[flat_param_key]
            for flat_param_key in flat_param_group["params"]
        ]
        
        # 根据参数对象列表获取嵌套的非全限定名称列表
        nested_unflat_param_names = [
            param_to_fqns[param] for param in param_group_params
        ]
        
        # 将嵌套的非全限定名称列表展开为一维列表，赋给参数组的params字段
        unflat_param_group["params"] = [
            unflat_param_name
            for unflat_param_names in nested_unflat_param_names
            for unflat_param_name in unflat_param_names
        ]  # flatten the list of lists
        
        # 将处理后的参数组添加到param_groups列表中
        param_groups.append(unflat_param_group)
    
    # 返回最终的参数组列表
    return param_groups
def _is_named_optimizer(optim_state_dict: Dict[str, Any]) -> bool:
    """
    Returns whether the state_dict is from a NamedOptimizer.
    This function checks that the keys in the state_dict['state'] are strings
    (which usually are FQNs) versus integers (which usually refer to param_ids
    from a vanilla torch.optim.Optimizer).
    """
    # 获取优化器状态字典中的 'state' 字段
    state = optim_state_dict.get("state", None)
    if not state:
        # 如果找不到 'state' 字段，假设不是 NamedOptimizer，因为NamedOptimizer具有即时初始化。
        return False
    try:
        # 尝试获取状态字典中的第一个键
        key = next(iter(state.keys()))
    except Exception as e:
        # 如果出现异常，抛出异常并附带 optim_state_dict 信息
        raise Exception(optim_state_dict) from e  # noqa: TRY002
    return isinstance(key, str)


@dataclass
class StateInfo:
    # 这些字典的键是状态名称，例如 `exp_avg`。
    tensors: Dict[str, _PosDimTensorInfo]
    scalar_tensors: Dict[str, torch.Tensor]
    non_tensors: Dict[str, Any]


def _allgather_state_info(
    fsdp_state: _FSDPState,
    input_states: Dict[str, Any],
) -> List[Dict[str, StateInfo]]:
    """
    Given the ``input_states``, allgather StateInfo for each state. The function
    uses all_gather_object to gather StateInfo so no GPU tensors are sent.
    """

    # 创建一个空的字典，用于存储处理过的状态信息
    processed_state_dict: Dict[str, StateInfo] = {}
    # 创建一个列表，用于存储所有进程收集到的状态信息字典
    gathered_state_info: List[Dict[str, StateInfo]] = [
        {} for _ in range(fsdp_state.world_size)
    ]

    # 遍历输入的状态字典
    for fqn, optim_state in input_states.items():
        # 创建一个空的 StateInfo 对象，用于存储处理过的状态信息
        processed_state = StateInfo({}, {}, {})
        # 遍历优化器状态的排序项
        for state_name, value in sorted_items(optim_state):
            if torch.is_tensor(value):
                if value.dim() == 0:
                    # 如果是标量张量，确保将其放在 CPU 上
                    processed_state.scalar_tensors[state_name] = value.cpu()
                else:
                    # 如果是多维张量，存储其形状和数据类型信息
                    processed_state.tensors[state_name] = _PosDimTensorInfo(
                        value.shape, value.dtype
                    )
            else:
                # 如果不是张量，直接存储值
                processed_state.non_tensors[state_name] = value
        processed_state_dict[fqn] = processed_state

    # 使用分布式操作 all_gather_object 收集处理过的状态信息到 gathered_state_info 列表中
    dist.all_gather_object(
        gathered_state_info,
        processed_state_dict,
        group=fsdp_state.process_group,
    )
    return gathered_state_info


def _convert_all_state_info(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: List[Dict[str, StateInfo]],
    input_states: Dict[str, Any],
    output_states: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[torch.dtype], Dict[str, List[Optional[torch.Tensor]]]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API converted
    the StateInfo into the original state if the state is not a non-scalar
    tensor. For a multi-dimensional tensor, the local state will be stored in
    ``state_buffer`` in a correct order for later allgather purpose.
    """

    # 创建一个空字典，用于存储状态缓冲区
    state_buffers: Dict[str, List[Optional[torch.Tensor]]] = {}
    return dtype, state_buffers  # 返回变量 dtype 和 state_buffers，类型可能未定义
def _unflatten_orig_param_states(
    fsdp_param_info: FSDPParamInfo,  # fsdp_param_info 参数信息对象，包含有关参数的扁平化处理和状态信息
    output_states: Dict[str, Dict[str, Any]],  # output_states 输出状态字典，键是参数的完全限定名称（FQN），值是收集到的状态信息
    state_name: str,  # state_name 参数状态的名称
    shard_state: bool,  # shard_state 是否对状态进行分片处理的布尔值
    to_save: bool,  # to_save 是否需要保存状态的布尔值，如果为 False，则直接返回
    cpu_offload: bool,  # cpu_offload 是否对状态进行 CPU 卸载的布尔值
) -> None:
    """
    Given a output state dict, ``output_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parmeter ID), and the values
    are gathered states, unflatten the states to the original dimensions.

    This function performs the unflattening process in-place.
    """
    if not to_save:  # 如果不需要保存状态，则直接返回，不进行后续操作
        return
    flat_param = fsdp_param_info.handle.flat_param  # 获取参数信息对象中的扁平参数
    fsdp_state = fsdp_param_info.state  # 获取参数信息对象中的状态信息
    for fqn, gathered_state in output_states.items():
        # 从 output_states 字典中获取每个 gathered_state 对象，以及对应的 fqn 键名

        value = gathered_state[state_name]
        # 从 gathered_state 中获取指定 state_name 的值

        param_idx = fsdp_param_info.param_indices[fqn]
        # 从 fsdp_param_info.param_indices 字典中获取与 fqn 对应的参数索引

        # TODO: This solution is not general and only apply to PTD TP solution.
        # 提示：此解决方案不是通用的，仅适用于 PTD TP 解决方案。

        if isinstance(value, DTensor):
            # 如果 value 是 DTensor 类型的对象

            placement = value.placements[0]
            # 获取 value 对象的第一个 placement

            # If gathered state is a DTensor and its TP placement is not Replicate(), we need to
            # gather the tensor on its TP dimension before chunking them into DTensor again.
            # 如果 gathered state 是 DTensor 并且其 TP placement 不是 Replicate()，我们需要在将它们再次分块成 DTensor 之前，先在其 TP 维度上收集张量。
            if placement != Replicate():
                placement_dim = placement.dim  # type: ignore[attr-defined]
                # 获取 placement 对象的 dim 属性

                value_local = value.redistribute(placements=(Replicate(),))
                # 重新分发 value 对象，将其放置在 Replicate() 上

                reshape_size = list(flat_param._shapes[param_idx])
                # 根据 param_idx 获取 flat_param._shapes 中对应的形状，并转换为列表

                reshape_size[placement_dim] *= value.device_mesh.size(0)
                # 根据 placement_dim 调整 reshape_size 的尺寸

                reshape_size = torch.Size(reshape_size)
                # 创建一个新的 torch.Size 对象，其尺寸由 reshape_size 定义

                value = value.reshape(reshape_size)
                # 使用 reshape_size 对 value 进行重塑
            else:
                # 如果 gathered state 是一个 replicate DTensor，则直接进行重塑
                value = value.reshape(flat_param._shapes[param_idx])

        else:
            # 如果 gathered state 是一个普通的 tensor，则直接将其重塑为未扁平化状态
            value = value.reshape(flat_param._shapes[param_idx])

        if shard_state:
            # 如果需要分片状态

            osd_config = fsdp_state._optim_state_dict_config
            # 获取 fsdp_state 的 _optim_state_dict_config 属性

            if getattr(osd_config, "_use_dtensor", False):
                # 如果 osd_config 的 _use_dtensor 属性为 True

                assert fsdp_state._device_mesh is not None
                # 断言 fsdp_state 的 _device_mesh 不为 None

                value = _ext_chunk_dtensor(
                    value,
                    fsdp_state.rank,
                    fsdp_state._device_mesh,
                    fsdp_state._fsdp_extension,
                )
                # 使用 _ext_chunk_dtensor 函数对 value 进行分块处理
            else:
                # 如果 _use_dtensor 属性为 False

                assert fsdp_state.process_group is not None
                # 断言 fsdp_state 的 process_group 不为 None

                value = _ext_chunk_tensor(
                    value,
                    fsdp_state.rank,
                    fsdp_state.world_size,
                    fsdp_state._device_handle.device_count(),
                    fsdp_state.process_group,
                    fsdp_state._fsdp_extension,
                )
                # 使用 _ext_chunk_tensor 函数对 value 进行分块处理

        elif not cpu_offload:
            # 如果不需要进行 CPU 卸载

            with SimpleProfiler.profile("clone"):
                # 使用 SimpleProfiler 对象进行 profile

                value = value.detach().clone()
                # 分离并克隆 value 对象

        if cpu_offload:
            # 如果需要进行 CPU 卸载

            with SimpleProfiler.profile(SimpleProfiler.Type.D2H):
                # 使用 SimpleProfiler 对象进行 profile，类型为 SimpleProfiler.Type.D2H

                value = value.cpu()
                # 将 value 对象移动到 CPU 上

        gathered_state[state_name] = value
        # 将处理后的 value 存回 gathered_state 的 state_name 键
def _allgather_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: List[Dict[str, StateInfo]],
    input_states: Dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API allgathers
    all tensor states and restore non-tensor states from ``gathered_state_info``.
    """
    # 获取 FSDP 参数信息中的状态对象
    fsdp_state = fsdp_param_info.state

    # 如果当前进程的排名为 0，并且分布式环境的调试级别为 DETAIL，则记录内存摘要信息
    if fsdp_state.rank == 0 and dist.get_debug_level() == dist.DebugLevel.DETAIL:
        logger.info(
            "Memory Summary before calling to _allgather_orig_param_states %s",
            fsdp_state._device_handle.memory_summary(),
        )

    # 初始化输出状态字典，以输入状态的全限定名为键，空字典为值
    output_states: Dict[str, Dict[str, Any]] = {fqn: {} for fqn in input_states.keys()}

    # 将所有状态信息转换为所需的数据类型和输出状态字典结构
    dtype, state_buffers = _convert_all_state_info(
        fsdp_param_info, gathered_state_info, input_states, output_states
    )

    # 如果状态缓冲区为空，则直接返回空的输出状态字典
    if len(state_buffers) == 0:
        return output_states

    # 检查是否有状态参数，并将结果存储在列表中
    has_state_params: List[bool] = [
        True if fqn in output_states else False
        for fqn, idx in fsdp_param_info.param_indices.items()
    ]

    # 创建一个空的张量，用于收集分布式环境中的参数状态
    flat_param = fsdp_param_info.handle.flat_param
    empty_func = functools.partial(
        torch.empty, dtype=dtype, device=fsdp_state.compute_device
    )
    gathered_tensor = empty_func(flat_param._padded_unsharded_size)

    # 同步设备以确保数据一致性
    fsdp_state._device_handle.synchronize()
    
    # 释放已收集的张量对象，以节省内存
    del gathered_tensor
    
    # 返回最终的输出状态字典
    return output_states


def _gather_all_orig_param_state(
    fsdp_param_info: FSDPParamInfo,
    input_states: Dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> Dict[str, Any]:
    """
    Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parmeter ID), gather all the
    states and unflatten them to the original dimensions. Note that all the
    params referred by the ``input_states`` must be managed by FSDP.
    """
    # 获取 FSDP 参数信息中的状态对象
    fsdp_state = fsdp_param_info.state

    # 如果世界大小为 1 或者分片策略为 NO_SHARD，则根据需求返回输入状态或空字典
    if (
        fsdp_state.world_size == 1
        or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        return input_states if to_save else {}
    # 使用 SimpleProfiler 记录代码块执行时间，类型为 RESHARDING
    with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
        # 使用 SimpleProfiler 记录代码块执行时间，类型为 ALLGATHER_OBJ
        with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER_OBJ):
            # 调用函数 _allgather_state_info，将 fsdp_state 和 input_states 进行全局聚合
            gathered_state_info = _allgather_state_info(fsdp_state, input_states)
        
        # 调用函数 _allgather_orig_param_states，将 fsdp_param_info、gathered_state_info、input_states、
        # shard_state、to_save 和 cpu_offload 作为参数，进行全局聚合，返回输出状态
        output_states = _allgather_orig_param_states(
            fsdp_param_info,
            gathered_state_info,
            input_states,
            shard_state,
            to_save,
            cpu_offload,
        )
    
    # 如果需要保存结果
    if to_save:
        # 遍历 fsdp_param_info.param_indices 字典中的每个键值对
        for key, idx in fsdp_param_info.param_indices.items():
            # 如果键 key 在 output_states 中存在，则继续下一个键值对
            if key in output_states:
                continue
            # 如果 fsdp_param_info.param_requires_grad 中索引 idx 对应的值为 False，则继续下一个键值对
            if not fsdp_param_info.param_requires_grad[idx]:
                continue

            # 抛出运行时错误，指出 key 不在输出状态中的情况
            raise RuntimeError(
                f"{key} is not in the output state. "
                "The FSDPParamInfo has the param keys "
                f"{sorted(fsdp_param_info.param_indices.keys())} while "
                "the output_states has the param keys "
                f"{sorted(output_states.keys())}."
            )
        
        # 返回全局聚合后的输出状态字典
        return output_states
    else:
        # 如果不需要保存结果，则返回空字典
        return {}
def _convert_state_with_orig_params(
    all_optim_state_keys: List[_OptimStateKey],
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]],
    fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
    optim_state_dict: Dict[Union[str, int], Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> Dict[str, Any]:
    # 用于存储 FSDP 状态的字典，键为 FSDPParamInfo 对象的 ID
    fsdp_osd_state: Dict[str, Any] = {}

    # 用于存储所有状态的字典，键为 FSDPParamInfo 对象的 ID，
    # 值为包含参数状态的字典
    all_states: Dict[int, Dict[str, Any]] = {}

    # 按照 rank 0 的平坦参数 ID 的顺序迭代，以确保跨 rank 的对齐全局聚合
    for optim_state_key in all_optim_state_keys:
        # 获取当前优化状态键对应的参数键
        param_key: Union[str, int, None] = optim_state_key_to_param_key.get(
            optim_state_key, None
        )

        # 如果参数键为空并且当前状态键不是 FSDP 管理的，则跳过
        if param_key is None and not optim_state_key.is_fsdp_managed:
            continue

        # 如果当前状态键是 FSDP 管理的
        if optim_state_key.is_fsdp_managed:
            # 获取第一个未扁平化参数名称
            fqn = optim_state_key.unflat_param_names[0]
            # 获取与该参数名称对应的 FSDPParamInfo 对象
            fsdp_param_info = fqn_to_fsdp_param_info.get(fqn, None)
            # 如果找不到对应的 FSDPParamInfo 对象，则跳过
            if fsdp_param_info is None:
                # 这种情况可能发生在不是所有 FSDP 实例都具有所有参数的情况下，
                # 比如在 FSDP + 某些 MPMD 风格的并行中可能会发生。
                # TODO: 不清楚是否需要对非 FSDP 管理的键执行相同检查。
                continue

            # 如果当前 fsdp_param_info 的 ID 尚未在 all_states 中，创建一个空字典
            if id(fsdp_param_info) not in all_states:
                all_states[id(fsdp_param_info)] = {}

            # 将当前参数状态加入到对应的 FSDPParamInfo 的状态字典中
            all_states[id(fsdp_param_info)][fqn] = {} if param_key is None else optim_state_dict[param_key]

        # 如果需要保存状态
        elif to_save:
            # 断言当前状态键的未扁平化参数名称列表长度为 1
            assert len(optim_state_key.unflat_param_names) == 1
            # 获取第一个未扁平化参数名称
            unflat_param_name = optim_state_key.unflat_param_names[0]
            # 使用简单分析器对非 FSDP 管理的拷贝进行性能分析
            with SimpleProfiler.profile("none_fsdp_managed_copy"):
                # 将 param_key 强制转换为 Union[str, int]
                param_key = cast(Union[str, int], param_key)
                # 复制当前参数状态到 fsdp_osd_state 中
                fsdp_osd_state[unflat_param_name] = copy.copy(
                    optim_state_dict[param_key]
                )
                # 如果开启了 CPU 卸载功能
                if cpu_offload:
                    # 对 fsdp_osd_state 中的每个张量进行 CPU 卸载
                    for state_name, value in sorted_items(
                        fsdp_osd_state[unflat_param_name]
                    ):
                        if not torch.is_tensor(value):
                            continue
                        fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    # 替代单独聚合每个参数的状态，一次性执行全局聚合以加快处理速度。
    # 遍历包含在 all_states 字典中的值（每个值都是一个字典）
    for _all_states in all_states.values():
        # 从 _all_states 字典中获取第一个键作为全限定名称（Fully Qualified Name）
        fqn = next(iter(_all_states.keys()))
        # 使用 fqn 查找 fsdp_param_info 字典中对应的参数信息
        fsdp_param_info = fqn_to_fsdp_param_info[fqn]
        # 断言参数信息中的梯度需求列表长度大于零，否则抛出异常
        assert len(fsdp_param_info.param_requires_grad) > 0, (
            "With use_orig_params, FSDPParamInfo should have requires_grad "
            "information. However, the length is zero."
        )
        # 遍历参数索引及其键值对
        for key, idx in fsdp_param_info.param_indices.items():
            # 如果键已经存在于 _all_states 中，则继续下一个键值对
            if key in _all_states:
                continue
            # 如果参数索引对应的梯度不需要计算，则继续下一个索引
            if not fsdp_param_info.param_requires_grad[idx]:
                continue
            # 抛出运行时异常，说明找不到某些参数在优化器状态中
            raise RuntimeError(
                f"{key} is not in the optimizer state. "
                "The FSDPParamInfo has the param keys "
                f"{sorted(fsdp_param_info.param_indices.keys())} while "
                "the optimizer has the param keys "
                f"{sorted(_all_states.keys())}."
            )
        # 更新 fsdp_osd_state 字典，收集所有原始参数的状态信息
        fsdp_osd_state.update(
            _gather_all_orig_param_state(
                fsdp_param_info,
                _all_states,
                shard_state,
                to_save,
                cpu_offload,
            )
        )

    # 返回更新后的 fsdp_osd_state 字典
    return fsdp_osd_state
def _convert_state_with_flat_params(
    all_optim_state_keys: List[_OptimStateKey],
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]],
    fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
    optim_state_dict: Dict[Union[str, int], Any],
    to_save: bool,
    shard_state: bool,
    cpu_offload: bool = True,
) -> Dict[str, Any]:
    # 初始化一个空字典，用于存储 FSDP 管理的状态
    fsdp_osd_state: Dict[str, Any] = {}

    # 按照 rank 0 的扁平化参数 ID 顺序进行迭代，以确保跨 rank 的数据收集对齐
    for optim_state_key in all_optim_state_keys:
        # 获取参数对应的键（可以是字符串或整数），如果没有找到则为 None
        param_key: Union[str, int, None] = optim_state_key_to_param_key.get(
            optim_state_key, None
        )

        # 断言 param_key 不为 None，如果 use_orig_params 为 False，则需要找到对应的参数 ID
        assert param_key is not None, (
            "If use_orig_params is False, we must be able to find the "
            f"corresponding param id. {optim_state_key} {param_key}"
        )

        # 如果当前参数是由 FSDP 管理
        if optim_state_key.is_fsdp_managed:
            # 获取第一个未扁平化参数名，以获取对应的 FSDPParamInfo
            fqn = optim_state_key.unflat_param_names[0]
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            
            # 对优化器状态进行反扁平化操作，获取未扁平化的状态
            unflat_state = _unflatten_optim_state(
                fsdp_param_info,
                optim_state_dict[param_key],
                to_save,
                shard_state,
                cpu_offload,
            )

            # 如果需要保存状态
            if to_save:
                # 断言未扁平化状态的长度与未扁平化参数名列表的长度相同
                assert len(unflat_state) == len(optim_state_key.unflat_param_names)
                # 将每个未扁平化参数名及其状态添加到 fsdp_osd_state 字典中
                for unflat_param_name, unflat_param_state in zip(
                    optim_state_key.unflat_param_names,
                    unflat_state,
                ):
                    fsdp_osd_state[unflat_param_name] = unflat_param_state

        # 如果不是由 FSDP 管理且需要保存状态
        elif to_save:
            # 断言未扁平化参数名列表长度为1
            assert len(optim_state_key.unflat_param_names) == 1
            # 获取第一个未扁平化参数名
            unflat_param_name = optim_state_key.unflat_param_names[0]
            # 复制并存储优化器状态
            fsdp_osd_state[unflat_param_name] = copy.copy(optim_state_dict[param_key])
            
            # 如果启用了 CPU offload，将所有张量移到 CPU 上
            if cpu_offload:
                for state_name, value in sorted_items(
                    fsdp_osd_state[unflat_param_name]
                ):
                    if not torch.is_tensor(value):
                        continue
                    fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    # 返回整理后的 FSDP 管理状态字典
    return fsdp_osd_state


@torch.no_grad()
def _optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ],
    rank0_only: bool,
    shard_state: bool,
    group: Optional[dist.ProcessGroup],
    using_optim_input: bool,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> Dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    """
    """
    Reset the SimpleProfiler and start profiling all operations.
    Reset any flat parameter gradient information associated with FSDP handles.
    Determine whether to save optimizer state on all ranks or just rank 0,
    based on `rank0_only` and `shard_state` flags.

    Args:
        model (nn.Module): The root module, possibly a FullyShardedDataParallel instance.
        optim (torch.optim.Optimizer): The optimizer associated with `model` parameters.
        rank0_only (bool): If True, save optimizer state only on rank 0; otherwise, on all ranks.
        shard_state (bool): If True, shard and distribute non-zero-dimension states.

    Returns:
        Dict[str, Any]: A dictionary containing the optimizer state for the model's original unflattened parameters.
            Includes keys "state" and "param_groups", following torch.optim.Optimizer.state_dict convention.
            Returns an empty dictionary for non-zero ranks if rank0_only is True.
    """
    # Reset the SimpleProfiler statistics
    SimpleProfiler.reset()
    # Create a context manager to profile all operations
    cm = ExitStack()
    cm.enter_context(SimpleProfiler.profile(SimpleProfiler.Type.ALL))
    # Reset any flat parameter gradient information if needed for FSDP handles in the model
    _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    # Determine whether to save optimizer state based on rank0_only and shard_state flags
    to_save = not rank0_only or dist.get_rank(group) == 0 or shard_state
    # 使用简单的性能分析器开始记录"preprocessing"阶段的性能
    with SimpleProfiler.profile("preprocessing"):
        # 获取参数到完全限定名的映射
        param_to_fqns = _get_param_to_fqns(model)
        # 获取扁平化参数到完全限定名的映射
        flat_param_to_fqn = _get_flat_param_to_fqn(model)
        # 检查优化器状态字典中的参数是否命名
        is_named_optimizer = _is_named_optimizer(optim_state_dict)

        # 根据是否使用优化器输入，获取参数键到参数的映射
        param_key_to_param = cast(
            Dict[Union[int, str], nn.Parameter],
            (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_key_to_param(
                    optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
                )
            ),
        )
        # 获取完全限定名到FSDP参数信息的映射
        fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)

    # 使用简单的性能分析器开始记录"preprocessing_with_comm"阶段的性能
    with SimpleProfiler.profile("preprocessing_with_comm"):
        # 将参数键映射到优化器键
        (
            all_optim_state_keys,
            optim_state_key_to_param_key,
        ) = _map_param_key_to_optim_keys(
            optim_state_dict,
            group,
            param_key_to_param,
            param_to_fqns,
            fqn_to_fsdp_param_info,
            merge_keys=use_orig_params,
        )

    # 使用简单的性能分析器开始记录"state_converting"阶段的性能
    with SimpleProfiler.profile("state_converting"):
        # 根据是否使用原始参数，选择转换函数
        convert_fn = (
            _convert_state_with_orig_params
            if use_orig_params
            else _convert_state_with_flat_params
        )
        # 使用选择的转换函数将状态转换为FSDP格式
        fsdp_osd_state = convert_fn(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            fqn_to_fsdp_param_info,
            optim_state_dict["state"],
            to_save,
            shard_state,
            cpu_offload,
        )

    # 到此为止，通信完成，如果不需要保存任何内容，各个等级可以提前返回
    if not to_save:
        return {}

    # 创建一个包含FSDP优化器状态字典的字典
    fsdp_osd: Dict[str, Any] = {"state": fsdp_osd_state}

    # 获取扁平化参数到完全限定名的集合
    flat_param_fqns = set(flat_param_to_fqn.values())
    # 遍历优化器状态字典中的每个条目
    for key, value in optim_state_dict["state"].items():
        # 如果条目已经存在于FSDP优化器状态中，则跳过
        if key in fsdp_osd_state:
            continue
        # 如果条目存在于扁平化参数完全限定名集合中，则跳过
        if key in flat_param_fqns:
            continue
        # 如果条目存在于参数键到参数的映射中，则跳过
        if key in param_key_to_param:
            continue
        # 如果条目未被FSDP识别，可能是用户定义的状态，发出警告
        warnings.warn(
            f"Found a optim state, {key}, that FSDP cannot process. FSDP "
            "will directly copy everything to the returned state_dict. In "
            "most cases, this is a user-defined state that is not "
            "associated with any particular parameter. Another possible "
            "case is this state is managed by TorchRec. Otherwise, there may "
            " be a mismatched assumption of optim_state_dict of this mode."
        )
        # 将未识别的状态添加到FSDP优化器状态字典中
        fsdp_osd_state[key] = value

    # 如果优化器状态字典中包含参数组信息，则解析并添加到FSDP优化器状态字典中
    if "param_groups" in optim_state_dict:
        fsdp_osd["param_groups"] = _unflatten_param_groups(
            optim_state_dict, param_key_to_param, param_to_fqns
        )

    # 关闭性能分析器
    cm.close()
    # 使用 SimpleProfiler 模块记录并输出性能分析信息，标识为 "FSDP _optim_state_dict() profiling: "
    SimpleProfiler.dump_and_reset("FSDP _optim_state_dict() profiling: ")

    # 返回 fsdp_osd 变量作为函数的结果
    return fsdp_osd
# 构建从参数的全限定名（fqn）到其对应的 `FSDPParamInfo` 的映射字典
def _get_fqn_to_fsdp_param_info(model: nn.Module) -> Dict[str, FSDPParamInfo]:
    
    # 定义一个内部函数 `module_fn`，用于处理每个模块，生成相应的映射信息
    def module_fn(module, prefix, tree_level, fqn_to_param_info):
        # 检查当前模块是否由 FSDP 管理，如果不是，则直接返回
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
        if fsdp_state is None:
            return
        
        # 对于属于 FSDP 管理的模块，进行延迟初始化
        _lazy_init(fsdp_state, module)
        
        # 获取模块的句柄
        handle = _module_handle(fsdp_state, module)
        if not handle:
            return
        
        # 获取平坦化参数
        flat_param = handle.flat_param
        
        # 创建 FSDPParamInfo 对象，用于存储参数信息
        fsdp_param_info = FSDPParamInfo(fsdp_state, handle, {}, [])
        
        # 遍历平坦化参数的 `_fqns` 列表，其中存储了局部 fqn
        # NOTE: `idx` 是对数据结构进行索引，不包括填充元素
        for idx, local_fqn in enumerate(flat_param._fqns):
            # 构建全限定名 fqn，并清理张量名
            fqn = clean_tensor_name(prefix + local_fqn)
            
            # 检查 fqn 是否已经在映射字典中，确保其对应的平坦化参数是一致的
            if fqn in fqn_to_param_info:
                assert fqn_to_param_info[fqn].handle.flat_param is flat_param, fqn
            
            # 将 fqn 映射到对应的 FSDPParamInfo 对象
            fqn_to_param_info[fqn] = fsdp_param_info
            
            # 记录 fqn 在参数索引中的位置
            fsdp_param_info.param_indices[fqn] = idx
            
            # 如果平坦化参数包含 `_params` 属性，则记录参数是否需要梯度
            if flat_param._params is not None:
                fsdp_param_info.param_requires_grad.append(
                    flat_param._params[idx].requires_grad
                )
    
    # 定义一个返回函数 `return_fn`，用于返回构建好的 fqn_to_param_info 字典
    def return_fn(fqn_to_param_info):
        return fqn_to_param_info
    
    # 初始化一个空的 fqn_to_param_info 字典，用于存储映射关系
    fqn_to_param_info: Dict[str, FSDPParamInfo] = {}
    
    # 使用 `_apply_to_modules()` 函数遍历模型中的所有模块，应用 `module_fn` 处理函数
    # 使用 `return_fn` 返回函数，处理模型中的命名参数，并将结果存储在 fqn_to_param_info 中
    return _apply_to_modules(
        model,
        module_fn,
        return_fn,
        [fqn for fqn, _ in _named_parameters_with_duplicates(model)],
        fqn_to_param_info,
    )


# 声明一个函数 `_set_optim_use_dtensor`，用于设置优化器使用 `dtensor` 标志
@no_type_check
def _set_optim_use_dtensor(
    fsdp_state: _FSDPState,
    state_dict_settings: StateDictSettings,
) -> None:
    # 如果在初始化 FSDP 时传入了 `device_mesh`，则自动将 `_use_dtensor` 标志设置为 True
    # 对于 `ShardedOptimStateDictConfig()`，如果 `state_dict_type` 必须设置为 `SHARDED_STATE_DICT`
    pass
    # 如果 fsdp_state 对象有属性 "_device_mesh"，则进入条件判断
    if getattr(fsdp_state, "_device_mesh", None):
        # 获取 state_dict_settings 中的 state_dict_type 属性
        state_dict_type = state_dict_settings.state_dict_type
        # 如果 state_dict_type 等于 StateDictType.LOCAL_STATE_DICT，则抛出运行时错误
        if state_dict_type == StateDictType.LOCAL_STATE_DICT:
            raise RuntimeError(
                "Found state_dict_type LOCAL_STATE_DICT.",
                "DeviceMesh is not compatible with LOCAL_STATE_DICT.",
                "Please set state_dict_type to SHARDED_STATE_DICT to get DTensor state_dict.",
            )
        else:
            # 否则，设置 state_dict_settings 中的 optim_state_dict_config._use_dtensor 为 True
            state_dict_settings.optim_state_dict_config._use_dtensor = True
```