# `.\pytorch\torch\distributed\collective_utils.py`

```
"""
A set of primitive functions for performing collective ops.

Each should also handle single rank scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast, Generic, List, Optional, Tuple, TypeVar, Union

import torch.distributed as dist


T = TypeVar("T")


@dataclass
class SyncPayload(Generic[T]):
    stage_name: Optional[str]
    success: bool
    payload: T
    exception: Optional[Exception] = None


def broadcast(
    data_or_fn: Union[T, Callable[[], T]],
    *,
    success: bool = True,
    stage_name: Optional[str] = None,
    rank: int = 0,
    pg: Optional[dist.ProcessGroup] = None,
) -> T:
    """
    Broadcasts the data payload from rank 0 to all other ranks.
    Or if a function is passed, execute it in rank 0 and broadcast result to all other ranks.

    Can be used to broadcast a failure signal to stop all ranks.

    If the function raises an exception, all ranks will raise.

    Args:
        data_or_fn: the data to broadcast or function to execute and broadcast result.
        success: False to stop all ranks.
        stage_name: the name of the logical stage for synchronization and debugging
        rank: rank to broadcast data or execute function and broadcast results.
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        the value after synchronization

    Example usage:
    >> id = broadcast(data_or_fn=allocate_id, rank=0, pg=ext_pg.my_pg)
    """

    # Check if data_or_fn is not None when success is False
    if not success and data_or_fn is not None:
        raise AssertionError(
            "Data or Function is expected to be None if not successful"
        )

    payload: Optional[T] = None
    exception: Optional[Exception] = None

    # Execute on rank 0 if no process group (pg) is passed or if pg exists and rank matches
    if (pg is None and rank == 0) or (pg is not None and pg.rank() == rank):
        # Determine if data_or_fn is a function or data payload
        if callable(data_or_fn):
            try:
                payload = data_or_fn()  # Execute the function and get its result
            except Exception as e:
                success = False
                exception = e
        else:
            payload = data_or_fn  # Use the provided data payload

    # Create a SyncPayload object to synchronize across ranks
    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    # If a process group (pg) is provided, broadcast sync_obj to all ranks in the group
    if pg is not None:
        broadcast_list = [sync_obj]
        dist.broadcast_object_list(broadcast_list, src=rank, group=pg)
        assert len(broadcast_list) == 1  # Ensure only one object is broadcasted
        sync_obj = broadcast_list[0]  # Update sync_obj to the broadcasted object

    # Any failure in any rank will trigger a throw in every rank.
    # 如果同步对象的成功标志为假（即同步失败），执行以下操作
    if not sync_obj.success:
        # 构建错误消息，指示失败的排名和阶段（如果提供）
        error_msg = f"Rank {rank} failed"
        if stage_name is not None:
            error_msg += f": stage {sync_obj.stage_name}"
        # 如果同步对象有异常信息，将异常信息添加到错误消息中
        if sync_obj.exception is not None:
            error_msg += f": exception {sync_obj.exception}"
        # 抛出运行时错误，并关联同步对象的异常信息
        raise RuntimeError(error_msg) from sync_obj.exception

    # 返回同步对象的有效载荷，使用类型 T 进行类型强制转换
    return cast(T, sync_obj.payload)
def all_gather(
    data_or_fn: Union[T, Callable[[], T]],
    stage_name: Optional[str] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> List[T]:
    """
    A simple all_gather primitive with basic synchronization guard logic,
    by checking payload from all ranks has the same stage name.

    Args:
        data_or_fn: the data to be all gathered across ranks or function to be executed
        stage_name: the sync stage name for out-of-sync protection
        pg: the process group for sync
    Throws:
        RuntimeError from original exception trace
    Returns:
        a list of synced data from all ranks

    Example usage:
    >> all_ids = all_gather(data_or_fn=allocate_id, pg=ext_pg.my_pg)
    """
    payload: Optional[T] = None
    exception: Optional[Exception] = None
    success = True

    # determine if it is an executable function or data payload only
    if callable(data_or_fn):
        try:
            payload = data_or_fn()  # Execute the callable to obtain data payload
        except Exception as e:
            success = False
            exception = e
    else:
        payload = data_or_fn  # Set the payload to the provided data

    # Create a synchronization payload object with gathered data
    sync_obj = SyncPayload(
        stage_name=stage_name,
        success=success,
        payload=payload,
        exception=exception,
    )

    if pg is not None:
        # List of success/failure across all ranks.
        total_list = [None] * dist.get_world_size(pg)  # Initialize list for all ranks
        all_gather_object_enforce_type(pg, total_list, sync_obj)
        
        # Determine the expected stage name from the first rank's sync payload
        stage_name = cast(SyncPayload[T], total_list[0]).stage_name
        
        exception_list: List[Tuple[int, Exception]] = []
        ret_list: List[T] = []
        error_msg: str = ""

        # Iterate through all gathered sync payloads
        for i, sp in enumerate(cast(List[SyncPayload[T]], total_list)):
            if sp.stage_name != stage_name:
                error_msg += (
                    f"Unexpected stage name received from rank {i}: {sp.stage_name} "
                )
                continue
            if not sp.success and sp.exception is not None:
                exception_list.append((i, sp.exception))
                continue
            ret_list.append(sp.payload)

        # Raise a RuntimeError if any exceptions were encountered
        if len(exception_list) > 0:
            raise RuntimeError(
                error_msg, exception_list
            ) from exception_list[0]
        
        return ret_list  # Return synchronized data from all ranks
    else:
        # If process group is not provided, check the success of sync operation
        if not sync_obj.success:
            raise RuntimeError(
                f"all_gather failed with exception {sync_obj.exception}",
            ) from sync_obj.exception
        return [sync_obj.payload]  # Return single payload in a list, ignoring type warning


# Note: use Any for typing for now so users can pass in
# either a list of None or target type placeholders
# otherwise pyre would complain
def all_gather_object_enforce_type(
    pg: dist.ProcessGroup,
    object_list: List[Any],  # List to be filled with sync payloads
    obj: Any,  # Sync payload object to be gathered
):
    """
    Enforce type of objects being gathered across ranks in a distributed setting.

    Args:
        pg: Process group for synchronization
        object_list: List to store synchronized objects from ranks
        obj: Sync payload object to be synchronized
    """
    # 声明一个名为 type_checker 的变量，类型为 Callable[[Any, Any], bool]，
    # 这表示 type_checker 是一个可调用对象（函数或者类的实例），接受两个参数，返回布尔值。
    type_checker: Callable[[Any, Any], bool] = lambda x, y: type(x) == type(y),
    # lambda 表达式定义了一个匿名函数，用于检查两个参数 x 和 y 的类型是否相同。
# 定义一个函数，类似于普通的 all_gather_object，但增加了额外的类型检查。
# 在收集完成后进行类型检查，以确保基本的一致性。
# 如果检查不通过，所有的进程都将因异常而失败。

# 这通常用于防止条件逻辑导致接收到意外的消息。这被视为严重的代码错误，
# 但由于逻辑堆栈的原因，实际上可能会隐式发生。

# 默认的检查不检查子类型（认为不同）或协变性（认为相同），但用户可以传递自定义检查器
# 如果需要更复杂的检查。

def all_gather_object_with_type_check(
    object_list: List[Any], obj: Any, pg: Any
) -> None:
    """
    Similar to plain all_gather_object but with additional type checking
    AFTER gather is done to ensure basic consistency.
    If check does not pass, all ranks will fail with exception.

    This is generally to prevent conditional logic leading to
    unexpected messages being received. This is considered fatal code error,
    but due to logic stacks this might happen implicitly in practice.

    The default check does not check sub type (considered different)
    or covariance (considered same) but users can pass in custom checker
    if more complicated check is needed.
    """
    
    # 使用分布式通信库的 all_gather_object 函数，将 obj 收集到 object_list 中，使用指定的进程组 pg
    dist.all_gather_object(object_list, obj, group=pg)

    # 进行保守的类型检查
    list_len = len(object_list)
    if list_len == 0:  # 如果 object_list 为空，则直接返回
        return
    
    # 获取第一个对象
    first_obj = object_list[0]
    
    # 遍历 object_list 中的对象，从第二个对象开始逐一检查类型
    for i in range(1, list_len):
        if not type_checker(first_obj, object_list[i]):  # 使用 type_checker 检查类型是否一致
            # 如果类型不一致，则抛出类型错误异常，指明出错的对象索引和类型信息
            raise TypeError(
                f"Object type at index {i} is {type(object_list[i])}, "
                f"while first object type is {type(first_obj)}"
            )
```