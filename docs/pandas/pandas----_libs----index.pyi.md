# `D:\src\scipysrc\pandas\pandas\_libs\index.pyi`

```
# 导入 NumPy 库，用于数值计算和数组操作
import numpy as np

# 导入 pandas 库中的类型定义
from pandas._typing import npt

# 导入 pandas 库中的 Index 和 MultiIndex 类
from pandas import (
    Index,
    MultiIndex,
)

# 导入 pandas 核心数组模块中的 ExtensionArray 类
from pandas.core.arrays import ExtensionArray

# 定义类级变量 multiindex_nulls_shift，表示 MultiIndex 中空值的偏移量
multiindex_nulls_shift: int

# 定义索引引擎类 IndexEngine
class IndexEngine:
    # 类级变量 over_size_threshold，表示是否超过大小阈值
    over_size_threshold: bool

    # 初始化方法，接收一个 NumPy 数组作为参数
    def __init__(self, values: np.ndarray) -> None: ...

    # 成员方法 __contains__，用于判断是否包含特定值，返回布尔值
    def __contains__(self, val: object) -> bool: ...

    # 成员方法 get_loc，接收一个对象作为参数，返回整数、切片或布尔值的 NumPy 数组
    # 返回类型为 int | slice | np.ndarray[bool]
    def get_loc(self, val: object) -> int | slice | np.ndarray: ...

    # 成员方法 sizeof，计算对象的大小，接收一个布尔值参数表示是否深度计算
    def sizeof(self, deep: bool = ...) -> int: ...

    # 成员方法 __sizeof__，返回对象的字节大小
    def __sizeof__(self) -> int: ...

    # 属性方法 is_unique，判断对象是否唯一
    @property
    def is_unique(self) -> bool: ...

    # 属性方法 is_monotonic_increasing，判断对象是否单调递增
    @property
    def is_monotonic_increasing(self) -> bool: ...

    # 属性方法 is_monotonic_decreasing，判断对象是否单调递减
    @property
    def is_monotonic_decreasing(self) -> bool: ...

    # 属性方法 is_mapping_populated，判断映射是否已填充
    @property
    def is_mapping_populated(self) -> bool: ...

    # 成员方法 clear_mapping，清除映射
    def clear_mapping(self): ...

    # 成员方法 get_indexer，接收一个 NumPy 数组作为参数，返回整数数组
    def get_indexer(self, values: np.ndarray) -> npt.NDArray[np.intp]: ...

    # 成员方法 get_indexer_non_unique，接收一个 NumPy 数组作为参数
    # 返回两个整数数组的元组
    def get_indexer_non_unique(
        self,
        targets: np.ndarray,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...

# MaskedIndexEngine 类继承自 IndexEngine 类
class MaskedIndexEngine(IndexEngine):
    # 初始化方法，接收一个对象作为参数
    def __init__(self, values: object) -> None: ...

    # 成员方法 get_indexer_non_unique，接收一个对象作为参数
    # 返回两个整数数组的元组
    def get_indexer_non_unique(
        self, targets: object
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...

# 下面的类都继承自 IndexEngine 类，表示不同数据类型的索引引擎

class Float64Engine(IndexEngine): ...

class Float32Engine(IndexEngine): ...

class Complex128Engine(IndexEngine): ...

class Complex64Engine(IndexEngine): ...

class Int64Engine(IndexEngine): ...

class Int32Engine(IndexEngine): ...

class Int16Engine(IndexEngine): ...

class Int8Engine(IndexEngine): ...

class UInt64Engine(IndexEngine): ...

class UInt32Engine(IndexEngine): ...

class UInt16Engine(IndexEngine): ...

class UInt8Engine(IndexEngine): ...

class ObjectEngine(IndexEngine): ...

class StringEngine(IndexEngine): ...

# DatetimeEngine 类继承自 Int64Engine 类
class DatetimeEngine(Int64Engine): ...

# TimedeltaEngine 类继承自 DatetimeEngine 类
class TimedeltaEngine(DatetimeEngine): ...

# PeriodEngine 类继承自 Int64Engine 类
class PeriodEngine(Int64Engine): ...

# BoolEngine 类继承自 UInt8Engine 类
class BoolEngine(UInt8Engine): ...

# 下面的类都继承自 MaskedIndexEngine 类，表示不同数据类型的掩码索引引擎

class MaskedFloat64Engine(MaskedIndexEngine): ...

class MaskedFloat32Engine(MaskedIndexEngine): ...

class MaskedComplex128Engine(MaskedIndexEngine): ...

class MaskedComplex64Engine(MaskedIndexEngine): ...

class MaskedInt64Engine(MaskedIndexEngine): ...

class MaskedInt32Engine(MaskedIndexEngine): ...

class MaskedInt16Engine(MaskedIndexEngine): ...

class MaskedInt8Engine(MaskedIndexEngine): ...

class MaskedUInt64Engine(MaskedIndexEngine): ...

class MaskedUInt32Engine(MaskedIndexEngine): ...

class MaskedUInt16Engine(MaskedIndexEngine): ...

class MaskedUInt8Engine(MaskedIndexEngine): ...

# MaskedBoolEngine 类继承自 MaskedUInt8Engine 类
class MaskedBoolEngine(MaskedUInt8Engine): ...

# BaseMultiIndexCodesEngine 类
class BaseMultiIndexCodesEngine:
    # levels 属性，表示多级索引的层级
    levels: list[np.ndarray]

    # offsets 属性，表示多级索引的偏移量，是一个 NumPy 数组
    offsets: np.ndarray  # np.ndarray[..., ndim=1]

    # 初始化方法，接收 levels（所有条目可散列）和 labels（所有条目整数类型）作为参数
    # offsets 参数是一个 NumPy 数组，表示偏移量
    def __init__(
        self,
        levels: list[Index],  # all entries hashable
        labels: list[np.ndarray],  # all entries integer-dtyped
        offsets: np.ndarray,  # np.ndarray[..., ndim=1]
    ) -> None: ...

    # 成员方法 get_indexer，接收一个 NumPy 对象数组作为参数，返回整数数组
    def get_indexer(self, target: npt.NDArray[np.object_]) -> npt.NDArray[np.intp]: ...
    # 定义一个私有方法 _extract_level_codes，接受一个 MultiIndex 类型的参数 target，并返回一个 NumPy 数组
    def _extract_level_codes(self, target: MultiIndex) -> np.ndarray:
# 定义 ExtensionEngine 类，用于扩展数组的操作
class ExtensionEngine:
    
    # 初始化方法，接受一个 ExtensionArray 类型的值，并无返回值
    def __init__(self, values: ExtensionArray) -> None:
        ...
    
    # 实现 in 操作符重载，检查对象是否包含某个值，返回布尔类型
    def __contains__(self, val: object) -> bool:
        ...
    
    # 根据值获取其在数组中的位置或索引，返回整数、切片对象或者 ndarray 数组
    def get_loc(self, val: object) -> int | slice | np.ndarray:
        ...
    
    # 获取给定值数组在当前对象中的索引数组，返回一个 ndarray 数组
    def get_indexer(self, values: np.ndarray) -> npt.NDArray[np.intp]:
        ...
    
    # 获取非唯一目标数组的索引器，返回两个 ndarray 数组元组
    def get_indexer_non_unique(
        self,
        targets: np.ndarray,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        ...
    
    # 判断对象是否包含唯一值，返回布尔类型
    @property
    def is_unique(self) -> bool:
        ...
    
    # 判断对象是否单调递增，返回布尔类型
    @property
    def is_monotonic_increasing(self) -> bool:
        ...
    
    # 判断对象是否单调递减，返回布尔类型
    @property
    def is_monotonic_decreasing(self) -> bool:
        ...
    
    # 计算对象占用的内存大小，可选参数 deep 表示是否深度计算
    def sizeof(self, deep: bool = ...) -> int:
        ...
    
    # 清空当前对象的映射或映射缓存
    def clear_mapping(self):
        ...
```