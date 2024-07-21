# `.\pytorch\torch\distributed\checkpoint\planner.py`

```py
import abc
import io
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import reduce
from typing import Any, List, Optional, Tuple, Union

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
    TensorProperties,
)

# 定义模块中公开的类和函数名列表
__all__ = [
    "WriteItemType",
    "LoadItemType",
    "TensorWriteData",
    "WriteItem",
    "ReadItem",
    "SavePlan",
    "LoadPlan",
    "SavePlanner",
    "LoadPlanner",
]

# 枚举类型，表示写入项的不同类型
class WriteItemType(Enum):
    TENSOR = auto()  # 表示写入的是张量数据
    SHARD = auto()   # 表示写入的是分片数据
    BYTE_IO = auto()  # 表示写入的是字节流数据

# 枚举类型，表示读取项的不同类型
class LoadItemType(Enum):
    TENSOR = auto()  # 表示读取的是张量数据
    BYTE_IO = auto()  # 表示读取的是字节流数据

# 数据类，用于表示张量的写入数据
@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata  # 表示数据块的存储元数据
    properties: TensorProperties  # 表示张量的属性
    size: torch.Size  # 表示张量的大小

# 数据类，用于表示写入项
@dataclass(frozen=True)
class WriteItem:
    """Dataclass which holds information about what needs to be written to storage."""

    index: MetadataIndex  # 表示元数据索引
    type: WriteItemType  # 表示写入项的类型

    # 可选字段，如果是张量写入，则存储张量的数据信息
    tensor_data: Optional[TensorWriteData] = None

    def tensor_storage_size(self) -> Optional[int]:
        """
        Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

        Returns:
            Optional[int] storage size, in bytes of underlying tensor if any.
        """
        if self.tensor_data is None:
            return None

        numels = reduce(operator.mul, self.tensor_data.size, 1)  # 计算张量元素总数
        dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)  # 计算张量元素类型的字节大小
        return numels * dtype_size  # 返回张量占用的总字节数量

# 数据类，用于表示读取项
@dataclass(frozen=True)
class ReadItem:
    # Read Item
    type: LoadItemType  # 表示读取项的类型

    # Index into the state_dict
    dest_index: MetadataIndex  # 表示目标张量在状态字典中的索引
    # Offsets into destination tensor
    dest_offsets: torch.Size  # 表示目标张量的偏移量

    # Index into the checkpoint
    storage_index: MetadataIndex  # 表示检查点中数据的索引
    # Offset into the checkpoint data
    storage_offsets: torch.Size  # 表示检查点数据的偏移量

    # Size of the hypercube to copy
    lengths: torch.Size  # 表示要复制的超立方体的大小

# 数据类，用于表示保存计划
@dataclass(frozen=True)
class SavePlan:
    items: List[WriteItem]  # 表示写入项列表
    storage_data: Any = None  # 存储数据，可以是任意类型
    planner_data: Any = None  # 规划数据，可以是任意类型

# 数据类，用于表示加载计划
@dataclass
class LoadPlan:
    items: List[ReadItem]  # 表示读取项列表
    storage_data: Any = None  # 存储数据，可以是任意类型
    planner_data: Any = None  # 规划数据，可以是任意类型

# 抽象类，定义保存计划器的协议
class SavePlanner(abc.ABC):
    """
    Abstract class defining the protocol used by save_state_dict to plan the save process.

    SavePlanners are stateful objects that can be used to customize the whole save process.

    SavePlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during save_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of a checkpoint save.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `SavePlan` that will be sent for global planning.
    """
    # create_global_plan - 仅在协调器(rank)调用。
    # 从所有 rank 的 SavePlan 中获取并制定任何全局决策。
    
    # finish_plan - 在所有 rank 上调用。
    # 这使每个 rank 有机会调整到全局规划决策。
    
    # resolve_data - 在每个 rank 上多次调用。
    # 查找 `state_dict` 中的值，以便写入存储层。
    
    # 用户推荐扩展 DefaultSavePlanner 而不是直接扩展此接口，因为大多数更改可以通过修改单个方法来表达。
    
    # 扩展的三种常见模式：
    
    # 重写 state_dict。这是扩展保存过程的最简单方式，因为它不需要理解 SavePlan 工作的细节：
    
    # >>> # xdoctest: +SKIP("undefined vars")
    # >>> class RenamePlanner(DefaultSavePlanner):
    # >>>     def set_up_planner(
    # >>>         self,
    # >>>         state_dict: STATE_DICT_TYPE,
    # >>>         storage_meta: Optional[StorageMeta],
    # >>>         is_coordinator: bool,
    # >>>     ) -> None:
    # >>>         # 所有键都加上前缀 `foo_`
    # >>>         super().set_up_planner({"foo_" + k: v for k, v in state_dict.items()}, storage_meta, is_coordinator)
    
    # 修改本地计划并并行查找。当需要精细控制数据持久化时很有用。
    
    # >>> # xdoctest: +SKIP("undefined vars")
    # >>> class FP16Planner(DefaultSavePlanner):
    # >>>     def create_local_plan(self):
    # >>>         plan = super().create_local_plan()
    # >>>         for p in plan:
    # >>>             if p.tensor_data is not None:
    # >>>                 p.tensor_data.properties.dtype = torch.float16
    # >>>         return plan
    # >>>
    # >>>     def resolve_data(self, write_item):
    # >>>         item = super().resolve_data(write_item)
    # >>>         return item if write_item.type == WriteItemType.BYTE_IO else item.to(torch.float16)
    
    # 使用全局规划步骤制定中心决策，无法通过每个 rank 单独制定。
    
    # >>> # xdoctest: +SKIP("undefined vars")
    # >>> from itertools import islice
    # >>> from dataclasses import replace
    # >>> class DDPLoadBalancingPlanner(DefaultSavePlanner):
    # >>>     # 使用默认的本地计划行为，在 rank 0 中具有所有非分片写入
    # >>>     # 此示例不处理 ShardedTensors
    # >>>     def create_global_plan(self, all_plans):
    # >>>         def chunk(it, size):
    # >>>             it = iter(it)
    # >>>             return list(iter(lambda: tuple(islice(it, size)), ()))
    # >>>         all_plans = [
    # >>>             replace(plan, items=items) for plan, items in
    # >>>                 zip(all_plans, chunk(all_plans[0].items, len(all_plans)))
    # >>>         ]
    # >>>         return super().create_global_plan(all_plans)
    
    # 最后，一些规划者需要在检查点中保存额外的元数据，每个 rank 通过本地计划贡献他们的数据项来完成这一目标，并
    """
    the global planner aggregate them:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class SaveExtraDataPlanner(DefaultSavePlanner):
    >>>     def create_local_plan(self) -> SavePlan:
    >>>         plan = super().create_local_plan()
    >>>         return replace(plan, planner_data="per-rank-data")
    >>>
    >>>     def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    >>>         global_plan, metadata = super().create_global_plan(all_plans)
    >>>         merged_data = [p.planner_data for p in global_plan]
    >>>         metadata = replace(metadata, planner_data=merged_data)
    >>>         return global_plan, metadata
    """

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: Optional[StorageMeta] = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this planner to save ``state_dict``.

        Implementations should save those values as they won't be provided lated in the save process.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        """
        Compute the save plan for the current rank.

        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        """
        Transform and prepare ``write_item`` from ``state_dict`` for storage, ensuring idempotency and thread-safety.

        Lookup the object associated with ``write_item`` in ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idempotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.
        """
        # 这是一个占位符方法，用于将 write_item 从 state_dict 准备并转换为存储所需的格式。
        # 实现时需要确保方法是幂等的（多次调用返回相同结果）并且线程安全的。
        # 在最终的 SavePlan 中，每个 WriteItem 都会调用这个方法至少一次。
        # 任何需要分配内存的转换应该延迟执行，以减少检查点过程中所需的内存峰值。
        # 当返回张量时，它们可以位于任何设备或格式上，也可以是视图。存储层负责决定如何保存它们。
        pass
# LoadPlanner 类定义了用于计划加载过程的接口协议。
class LoadPlanner:
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    LoadPlanner are stateful objects that can be used to customize the whole load process.

    LoadPlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during load_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of loading a checkpoint.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `LoadPlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the LoadPlan from all ranks and make any global decision.

    4) load_bytes - called multiple times on each rank
        This is called once per non-tensor value in state_dict.

    5) resolve_tensor and commit_tensor - called multiple times on each rank
        They are called in pair for each Tensor value in state_dict.

    Users are recommended to extend DefaultLoadPlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are two usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the load process as it
    doesn't requite understanding the intrincacies of how LoadPlan works. We need
    to keep a reference to the original state_dict as load happens in place so
    we need to be able to perform it in place

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultLoadPlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         metadata: Metadata,
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>>         self.original_state_dict = state_dict
    >>>         state_dict = {"foo_" + k: v for k, v in state_dict.items()}
    >>>
    >>>         if self.flatten_sharded_tensors:
    >>>             state_dict = _flatten_sharded_tensors(state_dict)
    >>>
    >>>         if self.flatten_state_dict:
    >>>             state_dict, self.mappings = flatten_state_dict(state_dict)
    >>>
    >>>         self.state_dict = state_dict
    >>>         self.metadata = metadata
    >>>         self.is_coordinator = is_coordinator
    >>>
    >>>     def load_bytes(self, read_item, value):
    >>>         # Remove the "foo_" prefix
    >>>         self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value)


    Modifying resolve_tensor and commit_tensor to handle load time transformation.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class MetaModelMaterialize(DefaultSavePlanner):
    >>>     def resolve_tensor(self, read_item):
    >>>         tensor = super().resolve_tensor(read_item)
    >>>         return torch.empty_like(tensor, device="cpu")
    >>>
    """
    # LoadPlanner 类作为定义加载过程协议的抽象类。

    def set_up_planner(self, state_dict, metadata, is_coordinator):
        """
        # 在所有 rank 上调用，用于设置加载过程的起始点。
        # state_dict: 当前的状态字典
        # metadata: 元数据
        # is_coordinator: 是否为协调器
        """
        self.original_state_dict = state_dict
        # 在这里可以进行状态字典的重写或转换操作

    def create_local_plan(self, state_dict):
        """
        # 在所有 rank 上调用，处理状态字典并生成一个 LoadPlan，将发送给全局规划。
        # state_dict: 当前的状态字典
        """
        # 在这里生成并返回一个 LoadPlan 对象

    def create_global_plan(self, load_plans):
        """
        # 仅在协调器 rank 上调用，接收来自所有 rank 的 LoadPlan 并进行全局决策。
        # load_plans: 所有 rank 发送的 LoadPlan 列表
        """
        # 在这里执行全局规划并返回结果

    def load_bytes(self, read_item, value):
        """
        # 在每个 rank 上多次调用，用于加载状态字典中每个非张量值。
        # read_item: 读取项的信息
        # value: 要加载的值
        """
        # 在这里执行加载字节内容的操作

    def resolve_tensor(self, read_item):
        """
        # 在每个 rank 上多次调用，用于处理加载时的张量值转换。
        # read_item: 读取项的信息
        """
        # 在这里执行张量解析的操作

    def commit_tensor(self, read_item, tensor):
        """
        # 在每个 rank 上多次调用，用于处理加载时的张量值提交。
        # read_item: 读取项的信息
        # tensor: 要提交的张量
        """
        # 在这里执行张量提交的操作
    """
    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """
        Call once the StorageReader finished loading data into ``tensor``.

        The provided tensor is the same one returned by the call to ``resolve_tensor``.
        This method is only needed if this LoadPlanner needs to post process ``tensor`` prior to
        copying it back to the one in the state_dict.

        The contents of tensor will follow its device synchronization model.
        """
        pass

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this instance to load data into ``state_dict``.

        N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by set_up_planner.

        N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        """
        Compute the global load plan and return plans for each rank.

        N.B. This is called on the coordinator rank only
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        """
        Accept the plan from coordinator and return final LoadPlan.
        """
        pass

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        Load the item described by ``read_item``and ``value``.

        This method is expected to modify in-place the underlying state_dict.

        The contents of ``value`` are defined by the SavePlanner used to produce
        the checkpoint being loaded.
        """
        pass

    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO:
        """
        Return the BytesIO to be used by the StorageReader to load `read_item`.

        The BytesIO should alias with one on the underlying state_dict as StorageReader will replace its contents.
        """
        raise NotImplementedError("LoadPlanner.resolve_bytes is not implemented")

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        """
        Return the tensor described by ``read_item`` to be used by the StorageReader to load `read_item`.

        The tensor should alias with one on the underlying state_dict as StorageReader will replace its contents.
        If, for any reason, that's not possible, the planner can use the ``commit_tensor`` method to copy the data
        back to the one in state_dict.
        """
        pass
```