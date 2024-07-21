# `.\pytorch\torch\utils\data\_utils\pin_memory.py`

```py
# 设置 mypy 工具允许未类型化的定义
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to put fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
# 导入必要的模块
import collections
import copy
import queue

import torch
from torch._utils import ExceptionWrapper

# 从当前目录下的 __init__.py 文件中导入 MP_STATUS_CHECK_INTERVAL 常量
from . import MP_STATUS_CHECK_INTERVAL

# 定义一个函数，负责将数据从输入队列取出并放入输出队列的固定内存中
def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    # 设置线程局部变量，限制 pin_memory 函数中的数据拷贝，避免占用所有 CPU 核心
    torch.set_num_threads(1)

    # 设置线程名字为 "pt_data_pin"
    torch.multiprocessing._set_thread_name("pt_data_pin")

    # 根据设备类型设置当前使用的设备
    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    # 定义一个内部函数，执行一个数据处理步骤
    def do_one_step():
        try:
            # 从输入队列获取数据，超时时间为 MP_STATUS_CHECK_INTERVAL
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        # 如果未完成事件未设置，并且数据不是 ExceptionWrapper 类型
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                # 将数据放入固定内存中
                data = pin_memory(data, device)
            except Exception:
                # 若出现异常，则将异常信息封装到 ExceptionWrapper 中
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )
            r = (idx, data)
        # 循环直到完成事件被设置
        while not done_event.is_set():
            try:
                # 将处理后的数据放入输出队列，超时时间为 MP_STATUS_CHECK_INTERVAL
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # 查看详细逻辑请参考文档中的 "Data Loader Multiprocessing Shutdown Logic" 部分
    while not done_event.is_set():
        # 确保不保留任何对象从一个迭代到下一个迭代
        do_one_step()


# 定义一个函数，根据数据类型将数据放入固定内存中
def pin_memory(data, device=None):
    # 如果数据是 torch.Tensor 类型，则调用其 pin_memory 方法将数据放入固定内存
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    # 如果数据是字符串或字节类型，则直接返回数据
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # 如果数据是可变映射类型，则创建其浅复制并更新，
                # 因为序列类型可能有额外的属性，无法直接使用 `type(data)(...)` 创建新序列。
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                # 如果数据是不可变映射类型，则使用其类型创建新的映射对象，并对每个项进行 pin_memory 操作。
                return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # 映射类型可能不支持 `copy()` / `update(mapping)` 或 `__init__(iterable)` 操作。
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        # 如果数据是具名元组，则使用其类型创建新的具名元组，并对每个元素进行 pin_memory 操作。
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        # 如果数据是普通元组，则对每个元素进行 pin_memory 操作，并返回列表（向后兼容）。
        return [
            pin_memory(sample, device) for sample in data
        ]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # 如果数据是可变序列类型，则创建其浅复制，
                # 因为序列类型可能有额外的属性，无法直接使用 `type(data)(...)` 创建新序列。
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            # 如果数据是不可变序列类型，则使用其类型创建新的序列对象，并对每个项进行 pin_memory 操作。
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # 序列类型可能不支持 `copy()` / `__setitem__(index, item)` 或 `__init__(iterable)` 操作（例如 `range`）。
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        # 如果数据具有 `pin_memory` 方法，则调用该方法。
        return data.pin_memory()
    else:
        # 如果数据不属于以上任何类型，则直接返回数据本身。
        return data
```