# `.\pytorch\torch\_lazy\computation.py`

```py
# mypy: allow-untyped-defs
# 引入 torch._C._lazy 模块，用于访问底层 C/C++ 函数
import torch._C._lazy
# 引入 torch._C._lazy_ts_backend 模块，用于访问懒加载张量的 TorchScript 后端
import torch._C._lazy_ts_backend


# 定义函数：获取 lazy tensors 中的 DeviceData 节点的张量 ID 和急切张量
def get_tensors_ts_device_data_node(tensors):
    """Return tensor ids and eager tensors for DeviceData nodes in the
    IR for the passed in lazy tensors.

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    # 调用 TorchScript 后端函数，获取 DeviceData 节点的张量 ID 和急切张量
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(tensors)


# 定义函数：获取 lazy tensors 的计算图哈希值
def get_graph_hash(tensors):
    """Return the graph hash for the passed in lazy tensors"""
    # 调用 torch._C._lazy 模块的函数，获取传入 lazy tensors 的计算图哈希值
    return torch._C._lazy._get_graph_hash(tensors)


# 定义函数：运行带有给定输入的缓存计算图
def run_cached_graph(hash_str, graph_inputs):
    """Running the cached computation graph with the given inputs

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    # 调用 TorchScript 后端函数，使用给定的哈希值和输入运行缓存的计算图
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, graph_inputs)
```