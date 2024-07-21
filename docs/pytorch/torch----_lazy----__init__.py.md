# `.\pytorch\torch\_lazy\__init__.py`

```
# mypy: allow-untyped-defs
# 导入线程模块，用于多线程操作
import threading

# 导入 torch._C._lazy 模块，包含了与 lazy tensor 相关的底层操作
import torch._C._lazy
# 导入 tree_flatten 和 tree_unflatten 函数，用于处理树形结构的数据
from torch.utils._pytree import tree_flatten, tree_unflatten

# 导入自定义的闭包函数，用于添加和运行步骤闭包
from .closure import add_step_closure, run_step_closures


def mark_step(device: str = "", wait=False):
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # 调用底层函数 _mark_step，标记一个步骤，处理 lazy tensors
    torch._C._lazy._mark_step(device, [], wait=wait)

    # 运行所有步骤闭包函数
    run_step_closures()


def wait_device_ops(devices=None):
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    # 如果 devices 为 None，则设为空列表
    if devices is None:
        devices = []
    # 等待给定设备上的所有异步操作完成
    torch._C._lazy._wait_device_ops(devices=devices)


def sync_multi(tensors, devices):
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    # 同步 lazy tensors 的数据，以降低其 IR，加快后端激活速度，并缓存编译的计算图
    torch._C._lazy._sync_multi(tensors, devices)


def get_tensor_id(tensor):
    """Return a unique id of the lazy tensor maintained by LTC"""
    # 返回 lazy tensor 的唯一标识符
    return torch._C._lazy._get_tensor_id(tensor)


def to_cpu(tensors, devices=None):
    # 如果 devices 为 None，则默认为 ["lazy"]
    devices = devices or ["lazy"]

    # 将输入 tensors 展平并保存其结构
    flattened, spec = tree_flatten(tensors)
    # 同步展平后的 tensors 数据到 CPU 端
    sync_multi(flattened, devices)
    # 将同步后的数据恢复成原始的树形结构，并将每个 tensor 移动到 CPU
    return tree_unflatten([t.to("cpu") for t in flattened], spec)


def save(tensors, *args, **kwargs):
    # 将输入的 tensors 转移到 CPU，并保存到文件
    torch.save(to_cpu(tensors), *args, **kwargs)
```