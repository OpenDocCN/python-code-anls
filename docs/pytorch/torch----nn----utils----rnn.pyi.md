# `.\pytorch\torch\nn\utils\rnn.pyi`

```py
# mypy: allow-untyped-defs
# 引入类型相关的模块和声明，包括 Any、Iterable、NamedTuple、overload 和 Sequence
from typing import Any, Iterable, NamedTuple, overload, Sequence
# 从 torch 模块中引入 Tensor 类型
from torch import Tensor
# 从 torch._prims_common 模块中引入 DeviceLikeType 类型
from torch._prims_common import DeviceLikeType
# 从 torch.types 模块中引入 _dtype 类型
from torch.types import _dtype

# 定义一个命名元组 PackedSequence_，包含 data、batch_sizes、sorted_indices 和 unsorted_indices 字段
class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Tensor | None
    unsorted_indices: Tensor | None

# 定义一个函数 bind，接受两个参数，但没有具体实现（Ellipsis 表示未实现）
def bind(optional: Any, fn: Any): ...

# 定义 PackedSequence 类，继承自 PackedSequence_ 类
class PackedSequence(PackedSequence_):
    # 定义构造函数 __new__，接受 data、batch_sizes、sorted_indices 和 unsorted_indices 参数，返回 Self 类型
    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Tensor | None = ...,
        sorted_indices: Tensor | None = ...,
        unsorted_indices: Tensor | None = ...,
    ) -> Self: ...
    
    # 定义 pin_memory 方法，将对象移到 GPU 的内存中
    def pin_memory(self: Self) -> Self: ...
    
    # 定义 cuda 方法，将对象移到 GPU 上，支持不同的参数
    def cuda(self: Self, *args: Any, **kwargs: Any) -> Self: ...
    
    # 定义 cpu 方法，将对象移到 CPU 上
    def cpu(self: Self) -> Self: ...
    
    # 定义 double 方法，将对象转换为双精度浮点数类型
    def double(self: Self) -> Self: ...
    
    # 定义 float 方法，将对象转换为单精度浮点数类型
    def float(self: Self) -> Self: ...
    
    # 定义 half 方法，将对象转换为半精度浮点数类型
    def half(self: Self) -> Self: ...
    
    # 定义 long 方法，将对象转换为长整型
    def long(self: Self) -> Self: ...
    
    # 定义 int 方法，将对象转换为整型
    def int(self: Self) -> Self: ...
    
    # 定义 short 方法，将对象转换为短整型
    def short(self: Self) -> Self: ...
    
    # 定义 char 方法，将对象转换为字符型
    def char(self: Self) -> Self: ...
    
    # 定义 byte 方法，将对象转换为字节型
    def byte(self: Self) -> Self: ...
    
    # 重载 to 方法，支持不同的参数类型，返回 Self 类型
    @overload
    def to(
        self: Self,
        dtype: _dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...
    
    @overload
    def to(
        self: Self,
        device: DeviceLikeType | None = None,
        dtype: _dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...
    
    @overload
    def to(
        self: Self,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...

    # 定义 is_cuda 属性，检查对象是否在 CUDA 上
    @property
    def is_cuda(self) -> bool: ...
    
    # 定义 is_pinned 方法，检查对象是否被固定在内存中
    def is_pinned(self) -> bool: ...

# 定义 invert_permutation 函数，接受一个 Tensor 类型或 None 类型的参数，无返回值
def invert_permutation(permutation: Tensor | None): ...

# 定义 pack_padded_sequence 函数，接受 input、lengths、batch_first 和 enforce_sorted 参数，返回 PackedSequence 类型
def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = ...,
    enforce_sorted: bool = ...,
) -> PackedSequence: ...

# 定义 pad_packed_sequence 函数，接受 sequence、batch_first、padding_value 和 total_length 参数，返回一个包含多个 Tensor 的元组
def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = ...,
    padding_value: float = ...,
    total_length: int | None = ...,
) -> tuple[Tensor, ...]: ...

# 定义 pad_sequence 函数，接受 sequences、batch_first 和 padding_value 参数，返回一个 Tensor 类型的对象
def pad_sequence(
    sequences: Tensor | Iterable[Tensor],
    batch_first: bool = False,
    padding_value: float = ...,
) -> Tensor: ...

# 定义 pack_sequence 函数，接受 sequences 和 enforce_sorted 参数，返回 PackedSequence 类型
def pack_sequence(
    sequences: Sequence[Tensor],
    enforce_sorted: bool = ...,
) -> PackedSequence: ...

# 定义 get_packed_sequence 函数，接受 data、batch_sizes、sorted_indices 和 unsorted_indices 参数，返回 PackedSequence 类型
def get_packed_sequence(
    data: Tensor,
    batch_sizes: Tensor | None,
    sorted_indices: Tensor | None,
    unsorted_indices: Tensor | None,
) -> PackedSequence: ...
```