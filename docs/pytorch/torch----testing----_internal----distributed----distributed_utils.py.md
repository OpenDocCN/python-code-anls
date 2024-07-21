# `.\pytorch\torch\testing\_internal\distributed\distributed_utils.py`

```
# 忽略 mypy 的错误提示

# 导入所需模块
from contextlib import contextmanager
from datetime import timedelta
from functools import (
    partial,
    wraps,
)

# 导入 Torch 分布式相关模块
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

# 定义 MockProcessGroup 类，继承自 dist.ProcessGroup
class MockProcessGroup(dist.ProcessGroup):

    def __init__(self, rank, world):
        super().__init__(rank, world)

    def getBackendName(self):
        return "mock_process_group"

# 创建 MockProcessGroup 实例的函数
def create_mock_pg(prefix_store, rank, world_size, timeout):
    return MockProcessGroup(rank, world_size)

# 注册 mock_process_group 后端到 dist.Backend
dist.Backend.register_backend('mock_process_group', create_mock_pg)

# 初始化模拟分布式环境的函数
def mock_init_dist(rank, world_size):
    # !!! WARNING !!!
    # Kids don't try this at home, this is a cute pile of hacks that
    # depends on a small mountain of c10d internals
    assert not dist.is_initialized()  # 断言当前尚未初始化分布式环境
    store = dist.HashStore()  # 创建哈希存储对象
    # 伪装 _store_based_barrier 认为其他进程已经完成检查
    # 这里的 0 是组索引
    store.add(f"{c10d.STORE_BASED_BARRIER_PREFIX}:0", world_size - 1)
    # 初始化进程组
    dist.init_process_group(
        backend="mock_process_group",
        rank=rank,
        world_size=world_size,
        store=store,
        group_name="fake",
        timeout=timedelta(seconds=1))

# 定义上下文管理器，初始化 c10d 使用模拟的进程组
@contextmanager
def with_dist(rank=0, world_size=2):
    """
    Context manager that initializer c10d with a fake process group.
    """
    mock_init_dist(rank=rank, world_size=world_size)
    try:
        yield  # 执行被管理代码块
    finally:
        dist.destroy_process_group()  # 清理并销毁进程组

# 装饰函数，用于包装测试函数以使用模拟的通信环境
def with_fake_comms(func=None, rank=0, world_size=2):
    """
    Function wrapper that inits a fake process group designed for testing.
    Right now only querying for world size is available
    """
    if func is None:
        return partial(with_fake_comms, rank=rank, world_size=world_size)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with with_dist(rank, world_size):
            func(self, *args, **kwargs)  # 执行被装饰的函数
    return wrapper
```