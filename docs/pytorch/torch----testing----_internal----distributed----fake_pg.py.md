# `.\pytorch\torch\testing\_internal\distributed\fake_pg.py`

```
# mypy: ignore-errors
# 导入 torch 分布式模块中的 dist
import torch.distributed as dist

# 从 torch._C._distributed_c10d 中导入 FakeProcessGroup 类
from torch._C._distributed_c10d import (
    FakeProcessGroup,
)


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """
    pass


def _create_fake_pg(prefix_store, rank, world_size, timeout):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convinient tool when playing
    with distributed but don't care about the actual data.
    """
    # 创建一个 FakeProcessGroup 对象，模拟一个虚假的进程组
    return FakeProcessGroup(rank, world_size)


# 注册一个名为 "fake" 的后端，用于创建虚假的进程组
# _create_fake_pg 是创建虚假进程组的函数，支持的设备包括 'cpu' 和 'cuda'
dist.Backend.register_backend("fake", _create_fake_pg, devices=['cpu', 'cuda'])
```