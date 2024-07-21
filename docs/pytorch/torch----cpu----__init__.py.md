# `.\pytorch\torch\cpu\__init__.py`

```
# mypy: allow-untyped-defs
r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from contextlib import AbstractContextManager  # 导入抽象上下文管理器
from typing import Any, Optional, Union  # 导入类型提示

import torch  # 导入 torch 库

from .. import device as _device  # 导入上层目录中的 device 模块
from . import amp  # 导入当前目录中的 amp 模块

__all__ = [  # 定义模块导出的全部符号列表
    "is_available",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
]

_device_t = Union[_device, str, int, None]  # 定义 _device_t 类型别名，可以是 _device、str、int 或 None 类型的联合类型


def _is_cpu_support_avx2() -> bool:
    r"""Returns a bool indicating if CPU supports AVX2."""
    return torch._C._cpu._is_cpu_support_avx2()  # 调用 torch 底层 API 检查 CPU 是否支持 AVX2


def _is_cpu_support_avx512() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512."""
    return torch._C._cpu._is_cpu_support_avx512()  # 调用 torch 底层 API 检查 CPU 是否支持 AVX512


def _is_cpu_support_vnni() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    # 注意：目前只检查 avx512_vnni，以后将增加对 avx2_vnni 的支持
    return torch._C._cpu._is_cpu_support_avx512_vnni()  # 调用 torch 底层 API 检查 CPU 是否支持 AVX512 VNNI


def _is_cpu_support_amx_tile() -> bool:
    r"""Returns a bool indicating if CPU supports AMX_TILE."""
    return torch._C._cpu._is_cpu_support_amx_tile()  # 调用 torch 底层 API 检查 CPU 是否支持 AMX_TILE


def _init_amx() -> bool:
    r"""Initializes AMX instructions."""
    return torch._C._cpu._init_amx()  # 调用 torch 底层 API 初始化 AMX 指令


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True  # 始终返回 True，表示 CPU 可用


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """
    pass  # 同步函数，不执行任何操作


class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """

    def __init__(self, priority: int = -1) -> None:
        pass  # 初始化函数，不执行任何操作

    def wait_stream(self, stream) -> None:
        pass  # 等待流的方法，不执行任何操作


class Event:
    def query(self) -> bool:
        return True  # 返回 True，表示事件存在

    def record(self, stream=None) -> None:
        pass  # 记录事件的方法，不执行任何操作

    def synchronize(self) -> None:
        pass  # 同步事件的方法，不执行任何操作

    def wait(self, stream=None) -> None:
        pass  # 等待事件的方法，不执行任何操作


_default_cpu_stream = Stream()  # 创建默认的 CPU 流对象
_current_stream = _default_cpu_stream  # 当前流默认为默认的 CPU 流对象


def current_stream(device: _device_t = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return _current_stream  # 返回当前流对象


class StreamContext(AbstractContextManager):
    r"""Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

    cur_stream: Optional[Stream]

    def __init__(self, stream):
        self.stream = stream  # 初始化时传入流对象
        self.prev_stream = _default_cpu_stream  # 记录之前的默认 CPU 流对象
    # 定义上下文管理器进入方法，用于处理流对象
    def __enter__(self):
        # 获取当前流对象
        cur_stream = self.stream
        # 如果当前流对象为空，直接返回，不执行后续操作
        if cur_stream is None:
            return
        
        # 访问全局变量 _current_stream，保存之前的流对象到 prev_stream
        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    # 定义上下文管理器退出方法，处理异常和流对象
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        # 获取当前流对象
        cur_stream = self.stream
        # 如果当前流对象为空，直接返回，不执行后续操作
        if cur_stream is None:
            return
        
        # 访问全局变量 _current_stream，恢复之前保存的流对象
        global _current_stream
        _current_stream = self.prev_stream
# 返回一个抽象上下文管理器，用于包装给定的流对象，选择特定的流进行操作
def stream(stream: Stream) -> AbstractContextManager:
    # 使用 StreamContext 类包装给定的流对象
    return StreamContext(stream)


# 返回当前系统中 CPU 设备的数量，此函数始终返回 1
def device_count() -> int:
    # 返回 CPU 设备的数量，固定为 1
    return 1


# 设置当前设备的函数，对于 CPU 设备，此函数不执行任何操作
def set_device(device: _device_t) -> None:
    # 在 CPU 情况下，不进行任何设备设置操作
    pass


# 返回当前设备的名称，对于 CPU 设备，始终返回字符串 "cpu"
def current_device() -> str:
    # 返回当前设备名称，对于 CPU 设备，始终是 "cpu"
    return "cpu"
```