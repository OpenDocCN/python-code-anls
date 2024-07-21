# `.\pytorch\torch\distributed\checkpoint\state_dict_saver.py`

```
# 引入需要的模块和依赖项
import inspect  # 用于获取对象信息的模块
import os  # 提供与操作系统交互的功能
import warnings  # 用于警告处理的模块
from concurrent.futures import Future, ThreadPoolExecutor  # 异步执行任务的功能
from typing import cast, Optional, Union  # 类型提示相关的模块和类型

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式训练模块
from torch.distributed._state_dict_utils import _offload_state_dict_to_cpu  # 分布式状态字典的CPU转移工具
from torch.distributed.checkpoint._storage_utils import _storage_setup  # 分布式检查点存储工具
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner  # 默认的保存计划工具
from torch.distributed.checkpoint.logger import _dcp_method_logger  # 分布式检查点方法日志记录器
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE  # 分布式检查点元数据和状态字典类型
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner  # 分布式检查点计划和保存计划工具
from torch.distributed.checkpoint.staging import AsyncStager  # 异步分段器
from torch.distributed.checkpoint.stateful import Stateful  # 有状态对象
from torch.distributed.checkpoint.storage import StorageWriter  # 分布式检查点存储写入工具
from torch.distributed.distributed_c10d import _get_default_group  # 获取默认的分布式组

from .utils import _api_bc_check, _DistWrapper, _profile  # 导入自定义工具函数和装饰器


__all__ = ["save_state_dict", "save", "async_save"]  # 导出的公共接口列表


@deprecated(
    "`save_state_dict` is deprecated and will be removed in future versions."
    "Please use `save` instead.",
    category=FutureWarning,  # 声明警告类型为未来警告
)
def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    """
    Deprecated方法，建议使用`save`代替。

    重置存储写入器的状态。

    Args:
        state_dict: 待保存的状态字典。
        storage_writer: 存储写入器对象。
        process_group: 进程组对象，默认为None。
        coordinator_rank: 协调员排名，默认为0。
        no_dist: 是否禁用分布式，默认为False。
        planner: 保存计划对象，可选。
    
    Returns:
        Metadata: 元数据对象。
    """
    storage_writer.reset()

    # TODO: test returning `save` here instead.
    with _profile():
        return _save_state_dict(
            state_dict,
            storage_writer,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)  # 类型忽略装饰器，用于记录方法调用的异常信息
@_api_bc_check  # API向后兼容性检查装饰器
def save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Metadata:
    """
    以SPMD风格保存分布式模型。

    与`torch.save()`不同，此函数处理`ShardedTensor`和`DTensor`，每个排名仅保存其本地分片。

    对于每个`Stateful`对象（具有`state_dict`和`load_state_dict`），在序列化之前调用`state_dict`。

    Args:
        state_dict: 待保存的状态字典。
        checkpoint_id: 检查点标识，可以是字符串、路径或None。
        storage_writer: 存储写入器对象，可选。
        planner: 保存计划对象，可选。
        process_group: 进程组对象，可选。
    
    Returns:
        Metadata: 元数据对象。
    
    Warnings:
        保存的状态字典在不同PyTorch版本间的向后兼容性不能保证。
        如果使用`process_group`参数，请确保只有其排名调用`save_state_dict`，且所有数据属于它。
    """
    """
    # 记录 API 使用情况，此处为保存检查点操作
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save")

    # 检测是否处于无分布式环境
    no_dist = not (dist.is_available() and dist.is_initialized())
    如果没有分布式环境：
        发出警告，假设意图是在单一进程中保存状态字典。
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to save in a single process."
        )
    """
    # 使用 _profile() 上下文管理器，用于性能分析
    with _profile():
        # 调用 _storage_setup 函数进行存储设置，并强制转换为 StorageWriter 类型
        storage_writer = cast(
            StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
        )
    
        # 调用 _stateful_to_state_dict 将 state_dict 转换为状态字典
        # 调用 _save_state_dict 保存状态字典到存储中，并返回保存结果
        return _save_state_dict(
            state_dict=_stateful_to_state_dict(state_dict),
            storage_writer=storage_writer,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )
@_dcp_method_logger(log_exceptions=True)
# 应用装饰器，用于记录方法调用日志，同时记录异常信息
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Future:
    """Asynchronous version of ``save``. This code first de-stages the state_dict on to the
    staging storage (defaults to CPU memory), and then calls the `save` in a separate thread.

    .. warning::
        This feature is experimental and subject to change.

    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform 'stage' and  'save'. If
            this is not specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specificed, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)

    Returns:
        Future: A future holding the resultant Metadata object from `save`.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter("/checkpoint/1")
        >>> checkpoint_future = torch.distributed.checkpoint.async_save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )
        >>>
        >>> # ... do some work ...
        >>>
        >>> checkpoint_future.result()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint.async_save")

    if dist.is_available() and dist.is_initialized():
        # 获取进程组，若未指定，则使用默认的进程组
        pg = process_group or _get_default_group()
        # 确保在异步保存时启用了 CPU 后端，若未启用则抛出异常
        assert (
            torch.device("cpu") in pg._device_types  # type: ignore[attr-defined]
        ), "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"

    # 设置存储写入器，根据参数设置或推断默认值
    storage_writer = cast(
        StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
    )

    # 转换 state_dict 为可保存状态字典
    state_dict = _stateful_to_state_dict(state_dict)
    
    # 若存储写入器是 AsyncStager 类型，则将 state_dict 分阶段存储
    if isinstance(storage_writer, AsyncStager):
        staged_state_dict = storage_writer.stage(state_dict)
    else:  # 如果不实现 AsyncStager 接口的 storage_writer 提供了 bwc
        # 将 state_dict 数据迁移到 CPU 上
        staged_state_dict = _offload_state_dict_to_cpu(state_dict, type_check=False)

    # 创建一个最大工作线程数为 1 的线程池执行器
    executor = ThreadPoolExecutor(max_workers=1)
    # 提交保存任务到线程池执行器中
    f: Future = executor.submit(
        save,
        staged_state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
    )
    # 添加完成回调函数，用于在任务完成后关闭执行器
    f.add_done_callback(lambda f: executor.shutdown(wait=False))

    # 如果 storage_writer 是 AsyncStager 类型，并且需要在执行后同步
    if (
        isinstance(storage_writer, AsyncStager)
        and storage_writer.should_synchronize_after_execute
    ):
        # 执行 storage_writer 的同步操作
        storage_writer.synchronize_staging()

    # 返回 Future 对象，表示保存任务的状态
    return f
    def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
        # 初始化一个空字典，用于存储转换后的状态字典
        stateful_state_dict = {}
        # 遍历传入的状态字典的每个键值对
        for key, elem in state_dict.items():
            # 如果当前元素是 Stateful 类型，则调用其 state_dict 方法进行浅复制
            stateful_state_dict[key] = (
                elem.state_dict() if isinstance(elem, Stateful) else elem
            )
        # 返回转换后的状态字典
        return stateful_state_dict


    def _save_state_dict(
        state_dict: STATE_DICT_TYPE,
        storage_writer: StorageWriter,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        planner: Optional[SavePlanner] = None,
    ) -> Metadata:
        torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

        # 根据输入的参数初始化一个分布式处理的包装器对象
        distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
        # 如果没有提供保存策略，则使用默认的保存策略
        if planner is None:
            planner = DefaultSavePlanner()
        # 确保保存策略不为空
        assert planner is not None

        # 全局元数据初始化为空
        global_metadata = None

        # 初始化用于检查点的关键字参数字典
        ckpt_kwargs = {}
        # 如果 storage_writer 对象有 checkpoint_id 属性，则将其添加到关键字参数字典中
        if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
            ckpt_kwargs["checkpoint_id"] = ckpt_id

        # 使用装饰器记录方法调用日志
        @_dcp_method_logger(**ckpt_kwargs)
        def local_step():
            # 确保保存策略不为空
            assert planner is not None
            # 获取存储元数据
            storage_meta = storage_writer.storage_meta()
            # 检查保存策略的 set_up_planner 方法是否包含 storage_meta 参数，若无则发出警告
            if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
                warnings.warn(
                    "The function definition for SavePlanner.set_up_planner has been updated"
                    " to include the storage_meta argument. Please update your implementation"
                    " to include this parameter."
                )
                # 调用保存策略的 set_up_planner 方法，传入状态字典和是否为协调器的标志
                planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
            else:
                # 调用保存策略的 set_up_planner 方法，传入状态字典、存储元数据和是否为协调器的标志
                planner.set_up_planner(
                    state_dict=state_dict,
                    storage_meta=storage_meta,
                    is_coordinator=distW.is_coordinator,
                )
            # 设置存储写入器，根据是否为协调器来执行不同的操作
            storage_writer.set_up_storage_writer(distW.is_coordinator)

            # 创建本地检查点计划
            local_plan = planner.create_local_plan()
            # 准备本地检查点计划，根据存储写入器的特定需求进行调整
            local_plan = storage_writer.prepare_local_plan(local_plan)
            return local_plan

        # 使用装饰器记录方法调用日志
        @_dcp_method_logger(**ckpt_kwargs)
        def global_step(all_local_plans):
            nonlocal global_metadata

            # 确保保存策略不为空
            assert planner is not None
            # 创建全局检查点计划，并获取全局元数据
            all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
            # 准备全局检查点计划，根据存储写入器的特定需求进行调整
            all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
            return all_local_plans

        # 使用分布式包装器对象进行 reduce_scatter 操作，获取中央检查点计划
        central_plan: SavePlan = distW.reduce_scatter("plan", local_step, global_step)

        # 使用装饰器记录方法调用日志
        @_dcp_method_logger(**ckpt_kwargs)
        def write_data():
            # 确保保存策略不为空
            assert planner is not None
            # 完成中央检查点计划，获取最终本地计划
            final_local_plan = planner.finish_plan(central_plan)
            # 将数据写入存储，并根据保存策略执行必要的操作
            all_writes = storage_writer.write_data(final_local_plan, planner)

            # 等待所有写入操作完成
            all_writes.wait()
            # 返回所有写入操作的结果
            return all_writes.value()

        # 使用装饰器记录方法调用日志
        @_dcp_method_logger(**ckpt_kwargs)
    # 定义一个函数 `finish_checkpoint`，接受 `all_results` 参数
    def finish_checkpoint(all_results):
        # 断言全局元数据 `global_metadata` 不为 None
        assert global_metadata is not None
        # 调用 `storage_writer` 对象的 `finish` 方法，传入全局元数据和所有结果作为参数
        storage_writer.finish(metadata=global_metadata, results=all_results)
        # 返回全局元数据
        return global_metadata

    # 调用 `distW` 对象的 `all_reduce` 方法，执行 "write" 操作，传入写入数据 `write_data` 和 `finish_checkpoint` 函数作为参数
    return distW.all_reduce("write", write_data, finish_checkpoint)
```