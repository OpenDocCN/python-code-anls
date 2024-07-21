# `.\pytorch\torch\distributed\checkpoint\_checkpointer.py`

```
# 导入必要的模块和类
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import torch.distributed as dist  # 导入分布式训练相关模块
import torch.distributed.checkpoint.state_dict_loader as loader  # 导入状态字典加载器
import torch.distributed.checkpoint.state_dict_saver as saver  # 导入状态字典保存器
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE  # 导入元数据和状态字典类型
from torch.distributed.checkpoint.storage import (
    LoadPlanner,  # 导入加载计划器
    SavePlanner,  # 导入保存计划器
    StorageReader,  # 导入存储读取器
    StorageWriter,  # 导入存储写入器
)

__all__: List[str] = []  # 初始化一个空的公开接口列表

class _Checkpointer:
    """This base class specefies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        storage_reader: StorageReader,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: Optional[LoadPlanner] = None,
        save_planner: Optional[SavePlanner] = None,
    ):
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite use to perform writes.
                            用于执行写操作的 StorageWrite 实例。
            storage_reader: StorageReader used to load data from.
                            用于加载数据的 StorageReader。
            process_group: ProcessGroup to be used for cross-rank synchronization.
                            用于跨秩同步的 ProcessGroup。
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
                              用于协调检查点的秩。默认使用 rank0。
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
                     如果为 True，则分布式检查点不会以 SPMD 样式加载。默认为 False。
            loader_planner: Instance of LoadPlanner to use when loading.
                            加载时使用的 LoadPlanner 实例。
            save_planner: Instance of SavePlanner to use when saving.
                          保存时使用的 SavePlanner 实例。
        """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Metadata:
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization.
        
        调用 `torch.distributed.state_dict_saver.save` 方法，利用初始化时传递的值进行保存操作。
        """
        return saver.save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
        )

    def async_save(
        self,
        state_dict: STATE_DICT_TYPE,
        callback: Optional[Callable[[Future], None]] = None
    ) -> Future:
        """Initiates asynchronous saving of the state_dict using the provided storage_writer.

        Args:
            state_dict: The state_dict to save asynchronously.
                        要异步保存的状态字典。
            callback: Optional callback function to call upon completion of the saving process.
                      可选的回调函数，在保存过程完成时调用。

        Returns:
            A Future object representing the asynchronous saving operation.
            表示异步保存操作的 Future 对象。
        """
        return saver.async_save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
            callback=callback
        )
    ) -> Future:
        """
        调用 :py:meth: `torch.distributed.state_dict_saver._async_save` 方法，使用初始化时传入的值。

        Returns:
            Future: 返回一个持有从 `save` 方法得到的 Metadata 对象的 Future。
        """
        # 调用 saver 的 async_save 方法来异步保存状态字典
        return saver.async_save(
            state_dict,
            storage_writer=self.storage_writer,
            process_group=self.process_group,
            planner=self.save_planner,
        )

    def load(self, state_dict: Dict[str, Any]) -> None:
        """调用 :py:meth: `torch.distributed.state_dict_loader.load` 方法，使用初始化时传入的值。"""
        # 调用 loader 的 load 方法来加载状态字典
        loader.load(
            state_dict,
            storage_reader=self.storage_reader,
            process_group=self.process_group,
            planner=self.load_planner,
        )
```