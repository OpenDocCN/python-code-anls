# `.\pytorch\torch\distributed\checkpoint\state_dict_loader.py`

```py
# 允许未标记类型的定义在类型检查期间不发出警告
mypy: allow-untyped-defs

# 导入必要的库和模块
import os  # 导入操作系统相关功能
import warnings  # 导入警告模块
from typing import Any, cast, Dict, Optional, Set, Union  # 导入类型相关功能

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner  # 导入默认加载计划类
from torch.distributed.checkpoint.logger import _dcp_method_logger  # 导入分布式检查点日志记录器
from torch.distributed.checkpoint.stateful import Stateful  # 导入有状态对象

from ._storage_utils import _storage_setup  # 导入存储工具函数
from .default_planner import DefaultLoadPlanner  # 导入默认加载计划类
from .planner import LoadPlan, LoadPlanner  # 导入加载计划和加载计划器类
from .storage import StorageReader  # 导入存储读取器类
from .utils import _all_gather_keys, _api_bc_check, _DistWrapper, _profile  # 导入工具函数

__all__ = ["load_state_dict", "load"]  # 指定可导出的模块成员列表


@deprecated(
    "`load_state_dict` is deprecated and will be removed in future versions. "
    "Please use `load` instead.",
    category=FutureWarning,
)
def load_state_dict(
    state_dict: Dict[str, Any],  # 状态字典，键为字符串，值为任意类型
    storage_reader: StorageReader,  # 存储读取器对象
    process_group: Optional[dist.ProcessGroup] = None,  # 进程组对象，可选
    coordinator_rank: int = 0,  # 协调员排名，默认为0
    no_dist: bool = False,  # 是否禁用分布式标志，默认为False
    planner: Optional[LoadPlanner] = None,  # 加载计划器对象，可选
) -> None:
    """
    This method is deprecated. Please switch to 'load'.
    """
    storage_reader.reset()  # 重置存储读取器状态
    with _profile():  # 使用性能分析器上下文管理器
        # TODO: test returning `load` here instead.
        return _load_state_dict(  # 调用内部函数_load_state_dict加载状态字典
            state_dict,
            storage_reader,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)  # 使用日志记录装饰器，记录异常
@_api_bc_check  # 使用API向后兼容性检查装饰器
def load(
    state_dict: Dict[str, Any],  # 状态字典，键为字符串，值为任意类型
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,  # 检查点ID，可以是字符串、路径或空
    storage_reader: Optional[StorageReader] = None,  # 存储读取器对象，可选
    planner: Optional[LoadPlanner] = None,  # 加载计划器对象，可选
    process_group: Optional[dist.ProcessGroup] = None,  # 进程组对象，可选
) -> None:
    """
    Load a distributed ``state_dict`` in SPMD style.

    Each rank will try to read the least amount of data necessary
    to fullfill the requested `state_dict`. When loading :class:`ShardedTensor`
    or :class:`DTensor` instances, each rank only reads data for their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    load will first call ``state_dict`` before attempting deserialization, followed by
    ``load_state_dict`` once the deserialization is complete.

    .. warning::
        All tensors in ``state_dict`` must be allocated on their
        destination device *prior to* calling this function.

        All non-tensor data is loaded using `torch.load()` and modified in place
        on state_dict.

    .. warning::
        Users must call `load_state_dict` on the root module to ensure load
        pos-processing and non-tensor data properly propagates.
    """
    # 检查当前是否没有启用或初始化分布式环境
    no_dist = not (dist.is_available() and dist.is_initialized())
    # 如果没有启用或初始化分布式环境，发出警告提示，假设意图是在单进程中加载检查点。
    if no_dist:
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process."
        )
    # 使用 _profile() 上下文管理器，用于性能分析
    with _profile():
        # 调用 _storage_setup 函数设置存储读取器，将其类型转换为 StorageReader 类型
        storage_reader = cast(
            StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
        )

        # 如果 no_dist 为真，直接获取状态字典的所有键
        if no_dist:
            keys = list(state_dict.keys())
        else:
            # 使用 _all_gather_keys 函数在进程组中收集所有状态字典的键，并排序
            keys = _all_gather_keys(state_dict, process_group)
            # 如果排序后的键列表与状态字典的键列表不匹配，发出警告
            if keys != sorted(state_dict.keys()):
                warnings.warn(
                    "Detected mismatched keys in state dict after all gather!"
                    " This behavior is unsupported and may cause errors may cause errors."
                )

        # 创建一个空字典用于存储有状态元素的状态字典
        statetful_sd = {}
        # 遍历状态字典的键
        for key in keys:
            # 如果状态字典中不包含当前键，继续下一次循环
            if key not in state_dict:
                continue
            # 获取当前键对应的元素
            elem = state_dict[key]
            # 如果元素是 Stateful 类型，则将其状态字典化，否则直接赋值
            statetful_sd[key] = (
                elem.state_dict() if isinstance(elem, Stateful) else elem
            )

        # 调用 _load_state_dict 函数，加载状态字典到模型中
        _load_state_dict(
            state_dict=statetful_sd,
            storage_reader=storage_reader,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )

        # 再次遍历状态字典的键
        for key in keys:
            # 如果状态字典中不包含当前键，继续下一次循环
            if key not in state_dict:
                continue
            # 获取当前键对应的元素
            elem = state_dict[key]
            # 如果元素是 Stateful 类型，加载其状态字典到元素中，并更新状态字典
            if isinstance(elem, Stateful):
                elem.load_state_dict(statetful_sd[key])
            state_dict[key] = statetful_sd[key]
def _load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
) -> None:
    # 记录 API 使用情况，这里是加载状态字典的 API
    torch._C._log_api_usage_once("torch.distributed.checkpoint.load_state_dict")

    # 创建分布式包装器对象
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    
    # 如果未提供加载规划器，则使用默认的加载规划器
    if planner is None:
        planner = DefaultLoadPlanner()

    # 检查是否有 checkpoint_id 属性，若有则传递给加载函数
    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id

    # 用装饰器记录日志，函数用于生成本地加载计划
    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        # 读取元数据并设置加载规划器
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        # 创建本地加载计划并准备本地数据读取计划
        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    # 用装饰器记录日志，函数用于生成全局加载计划
    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        assert planner is not None
        # 创建全局加载计划并准备全局数据读取计划
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    # 通过分布式包装器对象进行 reduce scatter 操作，得到中央加载计划
    central_plan: LoadPlan = distW.reduce_scatter("plan", local_step, global_step)

    # 用装饰器记录日志，函数用于最终数据读取
    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        assert planner is not None
        # 完成加载计划并读取所有数据
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)

        # 等待所有读取操作完成
        all_reads.wait()
        return None

    # 通过分布式包装器对象进行 all gather 操作，读取数据
    _ = distW.all_gather("read", read_data)


def _load_state_dict_from_keys(
    keys: Optional[Union[Set[str], str]] = None,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Load only the specified keys from the checkpoint, if no keys are specified, the entire
    checkpoint will be loaded. Note, this method completely loads the checkpoint into the
    current process and is not distributed.

    .. warning::
        所有非张量数据使用 `torch.load()` 加载

    .. note:
        与通常模式不同，该函数不接受状态字典作为输入，也不在原地加载。而是直接初始化并从文件中读取新的状态字典。

    .. note:
        如果没有初始化进程组，该函数会假定意图是将检查点加载到本地进程中。这在本地推理和使用常规张量（而非 DTensor 或 ShardedTensor）时很有用。

    .. note:
        假定排名 0 是协调员排名。
    """
    # 调用内部函数记录 Torch API 使用情况
    torch._C._log_api_usage_once(
        "torch.distributed.checkpoint._load_state_dict_from_keys"
    )
    
    # 检查当前环境是否支持并初始化了分布式通信
    no_dist = not (dist.is_available() and dist.is_initialized())
    if no_dist:
        # 若分布式通信不可用或未初始化，发出警告假定意图是在单进程中加载
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to load in a single process."
        )
    
    # 设置存储读取器，根据参数设置自动选择或推断读取器
    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )
    
    # 如果 keys 是字符串，则转换为单元素集合
    if isinstance(keys, str):
        keys = {keys}
    
    # 初始化空状态字典
    sd: Dict[str, Any] = {}
    
    # 调用加载状态字典的函数，传入参数
    _load_state_dict(
        state_dict=sd,
        storage_reader=storage_reader,
        process_group=process_group,
        no_dist=no_dist,
        planner=_EmptyStateDictLoadPlanner(keys=keys or set()),
    )
    
    # 返回加载后的状态字典
    return sd
```