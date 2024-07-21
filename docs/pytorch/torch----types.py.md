# `.\pytorch\torch\types.py`

```py
# mypy: allow-untyped-defs

# 引入内建模块
import builtins

# 在某些情况下，这些基本类型可能会被对应的顶层值所遮盖。下划线变体允许我们引用这些类型。
# 详见 https://github.com/python/mypy/issues/4146，解释为何需要这些解决方案。
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)

# 引入类型检查相关的模块和类型
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

# 引入PyTorch库
import torch

# 如果是类型检查模式，则引入GradientEdge类型
if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge

# 用于常见复合类型的便捷别名，这些类型在PyTorch中经常需要讨论
_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]
_TensorOrTensorsOrGradEdge = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

# 定义PyTorch中常见类型的别名
_dtype = torch.dtype
_device = torch.device
_qscheme = torch.qscheme
_layout = torch.layout
_size = Union[torch.Size, List[builtins.int], Tuple[builtins.int, ...]]
_dispatchkey = Union[builtins.str, torch._C.DispatchKey]

# 定义"数值"类型的元类型，符合我们的文档说明
Number = Union[builtins.int, builtins.float, builtins.bool]

# 定义"类似设备"类型的元类型，注意不要与'device'（实际设备对象）混淆。
# 此命名与PythonArgParser一致。
# None表示使用默认设备（通常是CPU）
Device = Optional[Union[_device, builtins.str, builtins.int]]
del Optional

# 定义Storage类，实现由${Type}StorageBase类实现的存储协议
class Storage:
    _cdata: _int  # 存储底层C数据的整数标识
    device: torch.device  # 存储的设备类型
    dtype: torch.dtype  # 存储的数据类型
    _torch_load_uninitialized: _bool  # 用于Torch加载的未初始化标志

    # 深度复制方法，抛出未实现错误
    def __deepcopy__(self, memo: dict) -> "Storage":
        raise NotImplementedError

    # 创建共享存储的新实例，抛出未实现错误
    def _new_shared(self, size: _int) -> "Storage":
        raise NotImplementedError

    # 将存储内容写入文件，抛出未实现错误
    def _write_file(
        self,
        f: Any,
        is_real_file: _bool,
        save_size: _bool,
        element_size: _int,
    ) -> None:
        raise NotImplementedError

    # 返回元素的大小，抛出未实现错误
    def element_size(self) -> _int:
        raise NotImplementedError

    # 检查是否为共享存储，抛出未实现错误
    def is_shared(self) -> _bool:
        raise NotImplementedError

    # 共享存储到内存，抛出未实现错误
    def share_memory_(self) -> "Storage":
        raise NotImplementedError

    # 返回存储占用的字节数，抛出未实现错误
    def nbytes(self) -> _int:
        raise NotImplementedError

    # 返回在CPU上的存储实例，抛出未实现错误
    def cpu(self) -> "Storage":
        raise NotImplementedError

    # 返回数据指针，抛出未实现错误
    def data_ptr(self) -> _int:
        raise NotImplementedError

    # 从文件中加载存储内容，抛出未实现错误
    def from_file(
        self,
        filename: _str,
        shared: _bool = False,
        nbytes: _int = 0,
    ) -> "Storage":
        raise NotImplementedError

    # 从文件中创建新存储，抛出未实现错误
    def _new_with_file(self, f: Any, element_size: _int) -> "Storage":
        raise NotImplementedError
```