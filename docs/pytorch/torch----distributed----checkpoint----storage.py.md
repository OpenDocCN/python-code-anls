# `.\pytorch\torch\distributed\checkpoint\storage.py`

```py
import abc  # 导入 abc 模块，用于定义抽象基类
import os  # 导入 os 模块，提供与操作系统交互的功能
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建不可变的数据类
from typing import Any, List, Optional, Union  # 导入类型提示相关的工具

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta  # 导入分布式检查点的元数据相关模块
from torch.distributed.checkpoint.planner import (  # 导入分布式检查点的计划相关模块
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.futures import Future  # 导入 torch 的 Future 类


__all__ = ["WriteResult", "StorageWriter", "StorageReader"]  # 模块公开的接口列表


@dataclass(frozen=True)
class WriteResult:
    """
    Represents the result of a write operation to storage.

    Attributes:
        index (MetadataIndex): Index associated with the write operation.
        size_in_bytes (int): Size of the data written, in bytes.
        storage_data (Any): Additional storage-specific data associated with the write.
    """
    index: MetadataIndex
    size_in_bytes: int
    storage_data: Any


class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """
        Calls to indicates a brand new checkpoint write is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint write. The meaning of the checkpiont_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """
        pass

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """
        pass

    @abc.abstractmethod
    def prepare_global_plan(self) -> LoadPlan:
        """
        Perform storage-specific global planning.

        Returns:
            LoadPlan: The global plan for loading data from storage.
        """
        pass

    @abc.abstractmethod
    def write_data(self, index: MetadataIndex, data: Any) -> WriteResult:
        """
        Write data to storage.

        Args:
            index (MetadataIndex): Index associated with the data.
            data (Any): Data to be written.

        Returns:
            WriteResult: Result object containing metadata about the write operation.
        """
        pass

    @abc.abstractmethod
    def finish(self) -> None:
        """
        Finalize the storage writing process.

        This method is called after all data has been written to storage.
        """
        pass
    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        执行存储的集中规划。

        该方法仅在协调器实例上调用。

        虽然该方法可以生成完全不同的计划，但首选方法是在SavePlan::storage_data中存储特定于存储的数据。

        Args:
            plans: 一个包含每个排名的SavePlan实例列表。

        Returns:
            经存储全局规划转换后的SavePlan列表
        """
        pass

    @abc.abstractmethod
    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[List[WriteResult]]:
        """
        使用planner写入plan中的所有项目以解析数据。

        子类应该对计划中的每个项目调用SavePlanner::resolve_data，以获取要写入的基础对象。

        子类应该懒惰地调用`resolve_data`，因为它可能会分配内存。
        对于张量，做以下假设：
        - 它们可能位于任何设备上，包括不匹配`WriteItem::tensor_data`上的设备。
        - 它们可能是视图或不连续的。只需要保存投影。

        Args:
            plan (SavePlan): 要执行的保存计划。
            planner (SavePlanner): 用于将项目解析为数据的计划器对象。

        Returns:
            完成为WriteResult列表的Future
        """
        pass

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        写入元数据并将当前检查点标记为成功。

        实际用于序列化`metadata`的格式/模式是实现细节。唯一要求是它可以恢复到相同的对象图。

        Args:
            metadata (Metadata): 新检查点的元数据
            results: 所有排名的WriteResult列表。

        Returns:
            None
        """
        pass

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        检查给定的checkpoint_id是否受存储支持。这允许我们启用自动存储选择。
        """
        ...

    def storage_meta(self) -> Optional[StorageMeta]:
        """
        返回特定于存储的元数据。这用于在检查点中存储附加信息，这些信息对于提供请求级可观察性可能非常有用。
        默认情况下返回None。

        TODO: 提供一个示例
        """
        return None
    @abc.abstractmethod
    def prepare_global_plan(self, global_plan: GlobalPlan) -> GlobalPlan:
        """
        Generate a global plan based on coordinator's information.

        This method is called by the coordinator instance to prepare a global
        plan that includes all necessary information for a distributed checkpoint
        across all ranks.

        Args:
            global_plan (GlobalPlan): The initial global plan generated by the
              coordinator.

        Returns:
            A modified ``GlobalPlan`` object that includes necessary information
            for all ranks in the distributed checkpoint.
        """
        pass
    # 准备全局计划的方法，用于存储加载的集中规划

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        使用 planner 解析数据，从 plan 中读取所有项。

        子类应调用 LoadPlanner::load_bytes 将 BytesIO 对象反序列化到正确位置。
        子类应调用 LoadPlanner::resolve_tensor 获取要加载数据的张量。

        存储层需负责适当安排所需的跨设备拷贝操作。

        Args:
            plan (LoadPlan): 要执行的本地计划
            planner (LoadPlanner): 用于解析项的计划器对象

        Returns:
            完成所有读取后的 Future 对象。
        """
        pass

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        检查给定的 checkpoint_id 是否受存储支持，这使我们能够启用自动存储选择。
        """
        ...
```