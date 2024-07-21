# `.\pytorch\torch\utils\dlpack.py`

```py
# 引入所需模块
from typing import Any

# 导入 PyTorch 库
import torch
# 导入枚举模块
import enum

# 从 torch._C 模块中导入两个函数
from torch._C import _from_dlpack
from torch._C import _to_dlpack as to_dlpack

# 定义设备类型枚举
class DLDeviceType(enum.IntEnum):
    # DLPack 规范中的设备枚举值
    kDLCPU = 1,
    kDLGPU = 2,
    kDLCPUPinned = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLExtDev = 12,
    kDLOneAPI = 14,

# 给 to_dlpack 函数添加文档字符串
torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used for anything in Python other than use it as input to
  ``from_dlpack``. The more idiomatic use of DLPack is to call
  ``from_dlpack`` directly on the tensor object - this works when that
  object has a ``__dlpack__`` method, which PyTorch and most other
  libraries indeed have now.

.. warning::
  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.
  Behavior when a capsule is consumed multiple times is undefined.

Args:
    tensor: a tensor to be exported

The DLPack capsule shares the tensor's memory.
""")

# TODO: 添加一个 typing.Protocol 以便告诉 Mypy 只接受具有 __dlpack__ 和 __dlpack_device__ 方法的对象。

def from_dlpack(ext_tensor: Any) -> 'torch.Tensor':
    """from_dlpack(ext_tensor) -> Tensor

    Converts a tensor from an external library into a ``torch.Tensor``.

    The returned PyTorch tensor will share the memory with the input tensor
    (which may have come from another library). Note that in-place operations
    will therefore also affect the data of the input tensor. This may lead to
    unexpected issues (e.g., other libraries may have read-only flags or
    immutable data structures), so the user should only do this if they know
    for sure that this is fine.

    Args:
        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
            The tensor or DLPack capsule to convert.

            If ``ext_tensor`` is a tensor (or ndarray) object, it must support
            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
            an opaque ``PyCapsule`` instance, typically produced by a
            ``to_dlpack`` function or method.
    """
    # 检查扩展张量是否具有 '__dlpack__' 属性
    if hasattr(ext_tensor, '__dlpack__'):
        # 获取扩展张量的设备信息
        device = ext_tensor.__dlpack_device__()
        # 如果设备是 CUDA 或 ROCm，需要传递当前流
        if device[0] in (DLDeviceType.kDLGPU, DLDeviceType.kDLROCM):
            # 获取当前 CUDA 流
            stream = torch.cuda.current_stream(f'cuda:{device[1]}')
            # cuda_stream 是流的指针，是一个公共属性，但未文档化
            # 根据 Array API 规范，CUDA 需要使用默认的传统流，其值为 1
            # 参考：https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_cuda = device[0] == DLDeviceType.kDLGPU
            # 因为 PyTorch 默认不使用 PTDS，直接传递传统流
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            # 获取扩展张量的 DL-Pack 格式，同时传递流指针
            dlpack = ext_tensor.__dlpack__(stream=stream_ptr)
        else:
            # 获取扩展张量的 DL-Pack 格式，不传递流
            dlpack = ext_tensor.__dlpack__()
    else:
        # 对于旧版本的张量，直接调用转换器
        dlpack = ext_tensor
    # 调用内部函数 _from_dlpack，将 DL-Pack 格式的张量转换为 PyTorch 张量
    return _from_dlpack(dlpack)
```