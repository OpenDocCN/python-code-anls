# `.\pytorch\torch\_C\_lazy.pyi`

```py
# mypy: allow-untyped-defs
# 引入来自 torch 模块的 Tensor 类型
from torch import Tensor

# 下面是一系列用于操作和管理设备、指标和张量的函数定义

# 定义在 torch/csrc/lazy/python/init.cpp 中的函数，用于标记步骤
def _mark_step(device: str, devices: list[str], wait: bool): ...

# 等待设备操作完成的函数
def _wait_device_ops(devices: list[str]): ...

# 重置指标的函数
def _reset_metrics(): ...

# 返回计数器名称列表的函数
def _counter_names() -> list[str]: ...

# 根据名称返回计数器值的函数
def _counter_value(name: str) -> int: ...

# 返回度量报告的字符串表示的函数
def _metrics_report() -> str: ...

# 根据张量列表返回其哈希值的函数
def _get_graph_hash(tensors: list[Tensor]) -> str: ...

# 将张量数据在多设备间同步的函数
def _sync_multi(
    tensors: list[Tensor],
    devices: list[str],
    wait: bool = True,
    sync_ltc_data: bool = True,
): ...

# 返回张量的标识 ID 的函数
def _get_tensor_id(tensor: Tensor) -> int: ...

# 返回张量列表的文本表示的函数
def _get_tensors_text(tensors: list[Tensor]) -> str: ...

# 返回张量列表的 DOT 图形表示的函数
def _get_tensors_dot(tensors: list[Tensor]) -> str: ...

# 返回张量列表的后端名称的函数
def _get_tensors_backend(tensors: list[Tensor]) -> str: ...

# 返回强制回退的状态的函数
def _get_force_fallback() -> str: ...

# 设置强制回退状态的函数
def _set_force_fallback(newval: str): ...

# 清除 IR 缓存的函数
def _clear_ir_cache(): ...

# 将 IR 缓存内容转储到指定文件的函数
def _dump_ir_cache(filename: str): ...

# 设置 IR 复用的函数
def _set_reuse_ir(val: bool): ...

# 返回默认设备类型的函数
def _get_default_device_type(): ...
```