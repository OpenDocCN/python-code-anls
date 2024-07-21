# `.\pytorch\torch\distributed\elastic\utils\store.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from datetime import timedelta
from typing import List

_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"

__all__ = ["store_timeout", "get_all", "synchronize", "barrier"]

@contextmanager
def store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """
    old_timeout = store.timeout  # 保存当前的超时设置
    store.set_timeout(timedelta(seconds=timeout))  # 设置新的超时时间
    yield  # 返回控制权给调用方
    store.set_timeout(old_timeout)  # 恢复之前的超时设置

def get_all(store, rank: int, prefix: str, world_size: int):
    r"""
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    The Rank0 process waits at the end to make sure all other processes
    finished the procedure before exiting.

    Usage

    ::

     values = get_all(store, 'torchelastic/data', 3)
     value1 = values[0] # retrieves the data for key torchelastic/data0
     value2 = values[1] # retrieves the data for key torchelastic/data1
     value3 = values[2] # retrieves the data for key torchelastic/data2

    """
    data_arr = store.multi_get([f"{prefix}{idx}" for idx in range(world_size)])  # 批量获取数据数组
    barrier_key = _barrier_nonblocking(
        store=store,
        world_size=world_size,
        key_prefix=f"{prefix}/finished",
    )  # 获取非阻塞屏障操作的关键字
    if rank == 0:
        # Rank0 runs the TCPStore daemon, as a result it needs to exit last.
        # Otherwise, the barrier may timeout if rank0 process finished the work
        # before other processes finished `get_all` method
        store.get(barrier_key)  # Rank0 进程等待屏障完成
    return data_arr  # 返回数据数组

def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    timeout: float = 300,
) -> List[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.

    Time complexity: O(N) per worker, O(N^2) globally.
    """
    with store_timeout(store, timeout):  # 设置超时时间并执行同步操作
        store.set(f"{key_prefix}{rank}", data)  # 设置当前进程的数据
        agent_data = get_all(store, rank, key_prefix, world_size)  # 获取所有代理的数据
        return agent_data  # 返回所有代理的数据数组

def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns the final key
    that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS  # 创建成员数量的键值对应的键名
    # 构建最后成员检查键，使用给定的键前缀和固定后缀
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN

    # 向存储中添加一个键值对，键为num_members_key，值为1，返回索引值idx
    idx = store.add(num_members_key, 1)
    
    # 如果返回的索引idx等于world_size（世界大小），则将last_member_key设置为"<val_ignored>"
    if idx == world_size:
        store.set(last_member_key, "<val_ignored>")

    # 返回最后成员检查键
    return last_member_key
def barrier(
    store, world_size: int, key_prefix: str, barrier_timeout: float = 300
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    # 使用 store_timeout 上下文管理器，设置超时时间为 barrier_timeout
    with store_timeout(store, barrier_timeout):
        # 调用 _barrier_nonblocking 函数尝试获取最后一个成员的键
        last_member_key = _barrier_nonblocking(
            store=store, world_size=world_size, key_prefix=key_prefix
        )
        # 从 store 中获取最后一个成员的键对应的值
        store.get(last_member_key)
```