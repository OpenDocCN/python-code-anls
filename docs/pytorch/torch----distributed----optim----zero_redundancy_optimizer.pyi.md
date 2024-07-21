# `.\pytorch\torch\distributed\optim\zero_redundancy_optimizer.pyi`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型声明
import enum  # 导入枚举模块
from typing import Any, Callable, overload  # 导入类型声明相关的模块

import torch  # 导入 PyTorch 库
from torch.distributed.algorithms.join import Joinable, JoinHook  # 导入分布式算法相关的模块
from torch.optim import Optimizer  # 导入优化器模块

class _ZeROJoinHook(JoinHook):
    zero: Any = ...
    def __init__(self, zero: Any) -> None: ...
    def main_hook(self) -> None: ...

class _DDPBucketAssignment:
    bucket_index: int  # 存储桶的索引
    parameters: list[torch.Tensor]  # 存储相关张量的列表
    offset: int  # 存储在存储桶中的偏移量
    device: torch.device  # 存储相关设备
    tensor: torch.Tensor | None  # 存储相关张量或者空

class _OverlapStatus(enum.IntEnum):
    UNINITIALIZED: int = ...  # 未初始化状态的枚举值
    DDP_HAS_REBUILT_BUCKETS: int = ...  # DDP 已重建桶的枚举值
    INITIALIZED: int = ...  # 初始化状态的枚举值

class _OverlapInfo:
    status: Any = ...  # 存储状态信息
    params_per_bucket: Any = ...  # 每个存储桶的参数
    params_per_rank: Any = ...  # 每个排名的参数
    offsets: Any = ...  # 存储偏移信息
    broadcast_handles: Any = ...  # 广播处理句柄
    bucket_index_to_future: Any = ...  # 存储存储桶到未来的映射
    bucket_index_to_bucket: Any = ...  # 存储存储桶到存储桶的映射
    bucket_indices_seen: Any = ...  # 存储已见存储桶的索引
    assigned_ranks_per_bucket: list[set[int]] = ...  # 每个存储桶分配的排名集合列表
    total_size: int = ...  # 总大小
    shard_buckets: bool = ...  # 分片存储桶的标志
    def __init__(self) -> None: ...
    def wait_for_broadcasts(self) -> None: ...
    def clear_per_iter_info(self) -> None: ...

class ZeroRedundancyOptimizer(Optimizer, Joinable):
    functional_optim_map: Any = ...  # 存储功能优化映射
    initialized: bool = ...  # 初始化标志
    process_group: Any = ...  # 进程组
    world_size: int = ...  # 世界大小
    rank: int = ...  # 排名
    global_rank: int = ...  # 全局排名
    parameters_as_bucket_view: bool = ...  # 参数作为存储桶视图
    optim: Any = ...  # 优化器
    _device_to_device_index: dict[torch.device, int] = ...  # 设备到设备索引的映射字典
    _overlap_with_ddp: bool = ...  # 是否与DDP重叠的标志
    _overlap_info: _OverlapInfo = ...  # 重叠信息对象
    _buckets: list[list[torch.Tensor]] = ...  # 存储桶列表
    _bucket_assignments_per_rank: list[dict[int, _DDPBucketAssignment]] = ...  # 每个排名的存储桶分配字典列表
    def __init__(
        self,
        params: Any,
        optimizer_class: type[Optimizer],
        process_group: Any | None = ...,
        parameters_as_bucket_view: bool = ...,
        overlap_with_ddp: bool = ...,
        **defaults: Any,
    ) -> None: ...
    def add_param_group(self, param_group: dict[str, Any]) -> None: ...  # 添加参数组
    def consolidate_state_dict(self, to: int = ...) -> None: ...  # 合并状态字典
    @overload
    def step(self, closure: None = ..., **kwargs: Any) -> None: ...  # 步进函数的重载，不返回值
    @overload
    def step(self, closure: Callable[[], float], **kwargs: Any) -> float: ...  # 步进函数的重载，返回损失值
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...  # 加载状态字典
    def state_dict(self) -> dict[str, Any]: ...  # 返回状态字典
    def _local_step(
        self,
        gradients: list[torch.Tensor | None] | None = None,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> float | None: ...  # 本地步进函数
    def _get_assigned_rank(self, bucket_index: int) -> int: ...  # 获取分配的排名
    def _init_zero_for_overlap(self) -> None: ...  # 初始化重叠区域
    def join_hook(self, **kwargs): ...  # 加入钩子函数
    @property
    def join_device(self) -> torch.device: ...  # 加入设备属性
    def join_process_group(self) -> Any: ...  # 加入进程组函数
```