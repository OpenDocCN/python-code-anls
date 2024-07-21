# `.\pytorch\torch\distributed\checkpoint\utils.py`

```
# mypy: allow-untyped-defs
# 引入性能分析工具
import cProfile
# 提供有关对象的信息，如参数列表、函数源代码等
import inspect
# 用于处理输入输出流
import io
# 用于生成迭代工具
import itertools
# 提供与操作系统交互的功能
import os
# 引发警告
import warnings
# 提供上下文管理工具
from contextlib import contextmanager
# 提供装饰器，用于包装函数
from functools import wraps
# 提供性能统计信息
from pstats import Stats
# 提供类型提示
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, TypeVar, Union

# 引入PyTorch库
import torch
# 引入分布式通信模块
import torch.distributed as dist
# 引入分布式张量
from torch.distributed._shard.sharded_tensor import ShardedTensor
# 引入张量分片
from torch.distributed._shard.sharded_tensor.shard import Shard

# 引入本地模块
from .api import (
    _is_wrapped_exception,
    _wrap_exception,
    CheckpointException,
    WRAPPED_EXCEPTION,
)
# 引入元数据索引和状态字典类型
from .metadata import MetadataIndex, STATE_DICT_TYPE


# 导出模块中的公共接口
__all__ = ["find_tensor_shard", "find_state_dict_object"]

# 定义类型变量
T = TypeVar("T")
R = TypeVar("R")


# 函数：根据错误结果列表生成失败字典
def _get_failure_dict(
    results: List[Union[T, WRAPPED_EXCEPTION]]
) -> Dict[int, WRAPPED_EXCEPTION]:
    return cast(
        Dict[int, WRAPPED_EXCEPTION],
        {i: err for i, err in enumerate(results) if _is_wrapped_exception(err)},
    )


# 函数：收集所有键并返回排序后的列表
def _all_gather_keys(
    local_dict: Dict[Any, Any], group: Optional[dist.ProcessGroup] = None
) -> List[Any]:
    """Gathers all keys, and returns them sorted."""
    keys = list(local_dict.keys())
    gathered_keys: List[List[Any]] = [None] * dist.get_world_size()  # type: ignore[list-item]

    dist.all_gather_object(gathered_keys, keys, group=group)
    return sorted(set(itertools.chain.from_iterable(gathered_keys)))


# 类：分布式操作的包装器
class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        use_dist: bool,
        coordinator_rank: int,
    ):
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.rank = 0
            self.is_coordinator = True

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        if self.use_dist:
            return dist.get_world_size(self.group)
        return 1

    def broadcast_object(self, object: Optional[T]) -> T:
        """Implement functionality similar to c10d::broadcast_object_list but without distributed enabled."""
        object_list = [object]
        if self.use_dist:
            dist.broadcast_object_list(
                object_list=object_list,
                group=self.group,
                src=self.coordinator_rank,
            )
        return cast(T, object_list[0])
    def gather_object(self, object: T) -> Optional[List[T]]:
        """Implement functionality similar to c10d::gather_object but without distributed enabled."""
        # 如果启用了分布式，初始化一个与进程数相同的列表，否则为 None
        if self.use_dist:
            gather_objs = (
                cast(List[T], [None] * dist.get_world_size(self.group))
                if self.is_coordinator
                else None
            )

            # 调用分布式函数进行对象的聚集操作
            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.coordinator_rank,
                group=self.group,
            )
            # 将结果设置为聚集对象列表
            result = gather_objs
        else:
            # 如果未启用分布式，则直接返回单个对象组成的列表
            result = [object]
        return result

    def all_gather_object(self, object: T) -> List[T]:
        """Implement functionality similar to c10d::all_gather_object but without distributed enabled."""
        # 如果启用了分布式，初始化一个与进程数相同的列表
        if self.use_dist:
            gather_objs = cast(List[T], [None] * dist.get_world_size(self.group))

            # 调用分布式函数进行对象的全局聚集操作
            dist.all_gather_object(
                object_list=gather_objs, obj=object, group=self.group
            )
        else:
            # 如果未启用分布式，则返回单个对象组成的列表
            gather_objs = [object]
        return gather_objs

    def scatter_object(self, object_list: Optional[List[T]]) -> T:
        """Implement functionality similar to c10d::scatter_object but without distributed enabled."""
        # 如果启用了分布式，初始化一个包含单个元素的列表
        if self.use_dist:
            gather_result = cast(List[T], [None])
            # 调用分布式函数进行对象的分发操作
            dist.scatter_object_list(
                scatter_object_output_list=gather_result,
                scatter_object_input_list=object_list if self.is_coordinator else None,
                src=self.coordinator_rank,
                group=self.group,
            )

            # 取出本地处理后的回复对象
            local_reply = gather_result[0]
        else:
            # 如果未启用分布式，确保输入对象列表不为 None，然后取出第一个元素作为本地回复对象
            assert object_list is not None
            local_reply = object_list[0]
        return local_reply
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a scatter.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Scatter to each rank part of the result.
        """
        local_data: Union[WRAPPED_EXCEPTION, T]
        try:
            local_data = map_fun()  # 调用 map_fun 函数获取本地数据
        except BaseException as e:
            local_data = _wrap_exception(e)  # 如果发生异常，封装异常对象

        all_data = self.gather_object(local_data)  # 聚集所有进程的数据到 rank 0

        all_results: Optional[List[Union[R, CheckpointException]]] = None
        if self.is_coordinator:
            assert all_data is not None
            node_failures = _get_failure_dict(all_data)  # 获取包含失败节点信息的字典

            if len(node_failures) == 0:
                try:
                    # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
                    all_results = cast(
                        List[Union[R, CheckpointException]],
                        reduce_fun(cast(List[T], all_data)),  # 调用 reduce_fun 函数处理所有数据
                    )
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)  # 如果处理过程中发生异常，记录到节点失败信息中

            if len(node_failures) > 0:
                all_results = [
                    CheckpointException(step, node_failures)
                ] * self.get_world_size()  # 如果存在失败节点，创建 CheckpointException 对象列表

        result = self.scatter_object(all_results)  # 将处理后的结果分发到所有进程

        if isinstance(result, CheckpointException):
            raise result  # 如果结果是 CheckpointException，则抛出异常

        return result  # 返回处理后的结果
    ) -> R:
        """
        Compute a value on each rank, then do centralized reduce on a single rank, followed by a broadcast.

        This method operates in the following way:
            Run ``map_fun`` on all ranks
            Gather results on rank 0
            Call ``reduce_fun`` on all those values
            Broadcast the reduced value to all ranks.
        """
        local_data: Union[T, WRAPPED_EXCEPTION]
        try:
            local_data = map_fun()  # 执行 map_fun 函数获取本地数据
        except BaseException as e:
            local_data = _wrap_exception(e)  # 若出现异常，则封装异常为 WRAPPED_EXCEPTION

        all_data = self.gather_object(local_data)  # 收集所有 ranks 的数据到 rank 0
        result: Optional[Union[R, CheckpointException]] = None
        if self.is_coordinator:  # 如果当前 rank 是协调器（rank 0）
            assert all_data is not None  # 断言 all_data 不为空
            node_failures = _get_failure_dict(all_data)  # 获取所有数据中的节点失败情况
            if len(node_failures) == 0:  # 如果没有节点失败
                try:
                    result = reduce_fun(cast(List[T], all_data))  # 调用 reduce_fun 函数对所有数据进行归约
                except BaseException as e:
                    node_failures[self.rank] = _wrap_exception(e)  # 将异常封装后加入节点失败字典

            if len(node_failures) > 0:  # 如果有节点失败
                result = CheckpointException(step, node_failures)  # 创建 CheckpointException 对象

        final_result = self.broadcast_object(result)  # 将最终结果广播到所有 ranks
        if isinstance(final_result, CheckpointException):  # 如果最终结果是 CheckpointException
            raise final_result  # 抛出异常
        return cast(R, final_result)  # 返回最终结果的强制类型转换为 R 类型

    def all_gather(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> List[T]:
        """
        Compute a value on each rank, then all_gather them.

        This method operates in the following way:
            Run ``map_cp`` on all ranks
            all_gather the values to all ranks
        """
        result: Union[T, WRAPPED_EXCEPTION]
        try:
            result = map_fun()  # 执行 map_fun 函数获取结果
        except BaseException as e:
            result = _wrap_exception(e)  # 若出现异常，则封装异常为 WRAPPED_EXCEPTION

        all_results = self.all_gather_object(result)  # 将结果 all_gather 到所有 ranks

        node_failures = _get_failure_dict(all_results)  # 获取所有结果中的节点失败情况
        if len(node_failures) > 0:  # 如果有节点失败
            raise CheckpointException(step, node_failures)  # 抛出 CheckpointException 异常
        return cast(List[T], all_results)  # 返回 all_results 的强制类型转换为 List[T]

    def broadcast(
        self,
        step: str,
        map_fun: Callable[[], T],
    ) -> T:
        """
        Compute a value on rank 0 and broadcast it.

        This method operates in the following way:
            Run ``map_cp`` on rank 0
            broadcast the value
        """
        result: Optional[Union[T, CheckpointException]] = None
        if self.is_coordinator:  # 如果当前 rank 是协调器（rank 0）
            try:
                result = map_fun()  # 执行 map_fun 函数获取结果
            except BaseException as e:
                result = CheckpointException(step, {self.rank: _wrap_exception(e)})  # 封装异常为 CheckpointException

        final_result = self.broadcast_object(result)  # 广播最终结果到所有 ranks
        if isinstance(final_result, CheckpointException):  # 如果最终结果是 CheckpointException
            raise final_result  # 抛出异常
        return cast(T, final_result)  # 返回最终结果的强制类型转换为 T
# 定义一个函数，用于查找分片张量中与给定索引相关的分片对象
def _find_shard(tensor: ShardedTensor, index: MetadataIndex) -> Shard:
    # 如果索引的偏移量为None，抛出数值错误异常
    if index.offset is None:
        raise ValueError(
            f"Cannot lookup {index.fqn} since its a ShardedTensor and no offset was provided"
        )

    # 获取本地分片列表
    shards = tensor.local_shards()
    
    # 索引快速路径
    if index.index is not None:
        # 如果索引小于分片数目并且分片的元数据偏移与索引偏移相同，返回该分片
        if (
            len(shards) > index.index
            and torch.Size(shards[index.index].metadata.shard_offsets) == index.offset
        ):
            return shards[index.index]

    # 遍历所有分片
    for shard in shards:
        # 如果分片的元数据偏移与索引偏移相同，返回该分片
        if torch.Size(shard.metadata.shard_offsets) == index.offset:
            return shard

    # 如果未找到匹配的分片，抛出数值错误异常
    raise ValueError(f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'")


# 定义一个函数，根据给定的索引查找张量的分片
def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor:
    # 如果张量具有属性 "__get_tensor_shard__"
    if hasattr(tensor, "__get_tensor_shard__"):
        # 调用张量的 "__get_tensor_shard__" 方法并返回结果（忽略类型检查）
        return tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
    
    # 如果张量是 ShardedTensor 类型，调用 _find_shard 函数并返回其张量部分
    if isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    
    # 如果索引的偏移量不为None，特殊情况：按原点查找张量
    if index.offset is not None:
        # 如果索引偏移等于零向量，返回张量本身
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        # 否则抛出数值错误异常
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    
    # 如果以上条件都不满足，返回张量本身
    return tensor


# 定义一个函数，根据给定的索引查找状态字典中的对象
def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any:
    # 如果状态字典中不存在给定的全限定名（FQN），抛出数值错误异常
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    
    # 获取状态字典中对应全限定名的对象
    obj = state_dict[index.fqn]

    # 如果对象是张量类型，调用 find_tensor_shard 函数查找其分片
    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    # 如果索引的偏移量不为None，抛出数值错误异常
    elif index.offset is not None:
        raise ValueError(
            f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'"
        )
    
    # 返回状态字典中对应全限定名的对象
    return obj


# 定义一个函数，实现两个整数序列的逐元素加法
def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a, b)]


# 定义一个函数，实现两个整数序列的逐元素减法
def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a - i_b for i_a, i_b in zip(a, b)]


# 定义一个类，实现一个读取视图，继承自 io.IOBase
class _ReaderView(io.IOBase):
    def __init__(self, base_stream: io.IOBase, offset: int, len: int):
        super().__init__()
        self.offset = offset
        self.len = len
        self.base_stream = base_stream
        self.seek(0)

    # 重写 seek 方法，实现基于偏移量的流定位
    def seek(self, __offset: int, __whence: int = os.SEEK_SET) -> int:
        if __whence == os.SEEK_SET:
            # 如果定位方式为从起始位置，计算实际偏移量并进行定位
            __offset = self.offset + __offset
        elif __whence == os.SEEK_END:
            # 如果定位方式为从末尾位置，计算实际偏移量并进行定位
            __whence = os.SEEK_SET
            __offset = (self.offset + self.len) - __offset
        return self.base_stream.seek(__offset, __whence)

    # 重写 tell 方法，返回当前位置相对于起始位置的偏移量
    def tell(self) -> int:
        return self.base_stream.tell() - self.offset

    # 重写 readable 方法，返回基础流的可读性
    def readable(self) -> bool:
        return self.base_stream.readable()

    # 重写 seekable 方法，返回基础流的可定位性
    def seekable(self) -> bool:
        return self.base_stream.seekable()
    # 调用基础流的readinto方法，并返回其结果
    def readinto(self, b):
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    # 调用基础流的read方法，并返回其结果；size参数控制读取的字节数，默认为-1表示读取所有可用数据
    def read(self, size=-1):
        return self.base_stream.read(size)
# 创建一个文件视图，用于读取文件的部分内容
def _create_file_view(file: io.IOBase, offset: int, length: int) -> io.IOBase:
    # FIXME (kumpera) torch.load fails if we wrap with io.BufferedReader
    # 返回一个包含指定偏移和长度的文件视图
    return _ReaderView(file, offset, length)


# 规范化设备信息，将设备类型和设备ID组合成字符串返回
def _normalize_device_info(device_type: str, device_id: int) -> str:
    """Device info normalization."""
    if device_type == "cpu":
        return "cpu"
    return f"{device_type}:{device_id}"


# TODO: integrate with distributed logging flag
# 定义一个全局变量，用于控制是否启用性能分析
ENABLE_PROFILE = False


# 上下文管理器，用于控制性能分析的开关
@contextmanager
def _profile():
    # 只有在启用性能分析且在rank0上或分布式不可用时才记录性能
    if ENABLE_PROFILE and (not dist.is_available() or dist.get_rank() == 0):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            stats = Stats(profiler)
            stats.sort_stats("time").print_stats(10)
    else:
        yield


# API参数检查装饰器
def _api_bc_check(func):
    @wraps(func)
    def inner_func(*args, **kwargs) -> Any:
        if len(args) == 2:
            warnings.warn(
                f"The argument order of {func.__name__} has been changed. "
                "Please check the document to avoid future breakages."
            )
            sig = inspect.signature(func)
            kwonlyargs = [
                p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY
            ]
            if "storage_writer" in kwonlyargs:
                assert "storage_writer" not in kwargs, (args, kwargs)
                kwargs["storage_writer"] = args[1]
            elif "storage_reader" in kwonlyargs:
                assert "storage_reader" not in kwargs, (args, kwargs)
                kwargs["storage_reader"] = args[1]
            else:
                raise RuntimeError(f"Unexpected kwonlyargs = {kwonlyargs}")
            return func(args[0], **kwargs)
        else:
            return func(*args, **kwargs)

    return inner_func
```