# `D:\src\scipysrc\pandas\pandas\_libs\sparse.pyi`

```
# 从 typing 模块导入 Sequence 类型
from typing import Sequence

# 导入 numpy 库并使用 np 别名
import numpy as np

# 从 pandas._typing 模块导入 Self 和 npt 类型
from pandas._typing import (
    Self,
    npt,
)

# 定义 SparseIndex 类
class SparseIndex:
    # 类属性：长度和点数
    length: int
    npoints: int
    
    # 初始化方法，无返回值
    def __init__(self) -> None: ...

    # ngaps 属性，返回整数
    @property
    def ngaps(self) -> int: ...

    # nbytes 属性，返回整数
    @property
    def nbytes(self) -> int: ...

    # indices 属性，返回 np.int32 类型的 NumPy 数组
    @property
    def indices(self) -> npt.NDArray[np.int32]: ...

    # equals 方法，接受参数 other，返回布尔值
    def equals(self, other) -> bool: ...

    # lookup 方法，接受整数 index 参数，返回 np.int32 类型
    def lookup(self, index: int) -> np.int32: ...

    # lookup_array 方法，接受 npt.NDArray[np.int32] 类型的 indexer 参数，返回相同类型的数组
    def lookup_array(self, indexer: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]: ...

    # to_int_index 方法，返回 IntIndex 类型对象
    def to_int_index(self) -> IntIndex: ...

    # to_block_index 方法，返回 BlockIndex 类型对象
    def to_block_index(self) -> BlockIndex: ...

    # intersect 方法，接受 SparseIndex 类型参数 y_，返回自身类型
    def intersect(self, y_: SparseIndex) -> Self: ...

    # make_union 方法，接受 SparseIndex 类型参数 y_，返回自身类型
    def make_union(self, y_: SparseIndex) -> Self: ...

# 定义 IntIndex 类，继承自 SparseIndex 类
class IntIndex(SparseIndex):
    # indices 属性，返回 np.int32 类型的 NumPy 数组
    indices: npt.NDArray[np.int32]

    # 初始化方法，接受 length、indices 和可选参数 check_integrity
    def __init__(
        self, length: int, indices: Sequence[int], check_integrity: bool = ...
    ) -> None: ...

# 定义 BlockIndex 类，继承自 SparseIndex 类
class BlockIndex(SparseIndex):
    # nblocks 属性，整数
    nblocks: int
    
    # blocs 属性，np.ndarray 类型的 NumPy 数组
    blocs: np.ndarray
    
    # blengths 属性，np.ndarray 类型的 NumPy 数组
    blengths: np.ndarray

    # 初始化方法，接受 length、blocs 和 blengths 参数
    def __init__(
        self, length: int, blocs: np.ndarray, blengths: np.ndarray
    ) -> None: ...

    # 重写 intersect 方法，接受 SparseIndex 类型参数 other，返回自身类型
    def intersect(self, other: SparseIndex) -> Self: ...

    # 重写 make_union 方法，接受 SparseIndex 类型参数 y，返回自身类型
    def make_union(self, y: SparseIndex) -> Self: ...

# 定义 make_mask_object_ndarray 函数，接受 arr 和 fill_value 参数，返回 np.bool_ 类型的 NumPy 数组
def make_mask_object_ndarray(
    arr: npt.NDArray[np.object_], fill_value
) -> npt.NDArray[np.bool_]: ...

# 定义 get_blocks 函数，接受 indices 参数，返回包含两个 np.int32 类型数组的元组
def get_blocks(
    indices: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
```