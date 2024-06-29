# `D:\src\scipysrc\pandas\pandas\_libs\hashtable.pyi`

```
# 导入必要的模块和类型定义
from typing import (
    Any,
    Hashable,
    Literal,
    overload,
)

import numpy as np  # 导入 NumPy 库

from pandas._typing import npt  # 导入 Pandas 的类型定义

# 定义一个名为 unique_label_indices 的函数签名，暂未实现具体功能
def unique_label_indices(
    labels: np.ndarray,  # 接受一个 NumPy 数组作为参数，用于存储标签数据
) -> np.ndarray: ...  # 返回一个 NumPy 数组，具体实现未定义

# 定义 Factorizer 类，表示一种因子化器
class Factorizer:
    count: int  # 计数器，存储一个整数值
    uniques: Any  # 存储任意类型的唯一值

    def __init__(self, size_hint: int, uses_mask: bool = False) -> None: ...
    # 初始化方法，接受一个整数类型的 size_hint 参数和一个布尔类型的 uses_mask 参数，无返回值

    def get_count(self) -> int: ...
    # 获取计数器值的方法，返回一个整数类型的值

    def factorize(
        self,
        values: np.ndarray,  # 用于因子化的值，接受一个 NumPy 数组
        na_sentinel=...,  # 标记缺失值的值，具体未定义
        na_value=...,  # 缺失值的具体表示方式，具体未定义
        mask=...,  # 掩码，具体未定义
    ) -> npt.NDArray[np.intp]: ...
    # 因子化方法，接受一个 NumPy 数组作为输入，并返回一个 NumPy 整数类型的数组

    def hash_inner_join(
        self, values: np.ndarray, mask=...
    ) -> tuple[np.ndarray, np.ndarray]: ...
    # 哈希内连接方法，接受一个 NumPy 数组作为输入，并返回两个 NumPy 数组组成的元组

# 定义 ObjectFactorizer 类，继承自 Factorizer 类
class ObjectFactorizer(Factorizer):
    table: PyObjectHashTable  # 存储 PyObject 的哈希表
    uniques: ObjectVector  # 存储 PyObject 的唯一值向量

# 各种类型的 Factorizer 类的定义，均继承自 Factorizer 并指定特定的哈希表和唯一值向量类型

class Int64Factorizer(Factorizer):
    table: Int64HashTable
    uniques: Int64Vector

class UInt64Factorizer(Factorizer):
    table: UInt64HashTable
    uniques: UInt64Vector

class Int32Factorizer(Factorizer):
    table: Int32HashTable
    uniques: Int32Vector

class UInt32Factorizer(Factorizer):
    table: UInt32HashTable
    uniques: UInt32Vector

class Int16Factorizer(Factorizer):
    table: Int16HashTable
    uniques: Int16Vector

class UInt16Factorizer(Factorizer):
    table: UInt16HashTable
    uniques: UInt16Vector

class Int8Factorizer(Factorizer):
    table: Int8HashTable
    uniques: Int8Vector

class UInt8Factorizer(Factorizer):
    table: UInt8HashTable
    uniques: UInt8Vector

class Float64Factorizer(Factorizer):
    table: Float64HashTable
    uniques: Float64Vector

class Float32Factorizer(Factorizer):
    table: Float32HashTable
    uniques: Float32Vector

class Complex64Factorizer(Factorizer):
    table: Complex64HashTable
    uniques: Complex64Vector

class Complex128Factorizer(Factorizer):
    table: Complex128HashTable
    uniques: Complex128Vector

# 各种类型的 Vector 类的定义，用于存储特定数据类型的向量数据

class Int64Vector:
    def __init__(self, *args) -> None: ...  # 初始化方法，参数未定义
    def __len__(self) -> int: ...  # 返回向量长度的方法
    def to_array(self) -> npt.NDArray[np.int64]: ...  # 将向量转换为 NumPy 整数类型数组的方法

class Int32Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.int32]: ...

class Int16Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.int16]: ...

class Int8Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.int8]: ...

class UInt64Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.uint64]: ...

class UInt32Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.uint32]: ...

class UInt16Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    def to_array(self) -> npt.NDArray[np.uint16]: ...

class UInt8Vector:
    def __init__(self, *args) -> None: ...
    def __len__(self) -> int: ...
    # 定义一个方法 `to_array`，返回类型为 `npt.NDArray[np.uint8]`
    def to_array(self) -> npt.NDArray[np.uint8]:
        # 此处暂未实现具体功能，只有占位符 `...`，需要根据具体需求完善功能实现
        pass
class Float64Vector:
    # 64位浮点数向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为64位浮点数的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.float64]: ...

class Float32Vector:
    # 32位浮点数向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为32位浮点数的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.float32]: ...

class Complex128Vector:
    # 128位复数向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为128位复数的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.complex128]: ...

class Complex64Vector:
    # 64位复数向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为64位复数的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.complex64]: ...

class StringVector:
    # 字符串向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为对象类型为字符串的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.object_]: ...

class ObjectVector:
    # 对象向量类
    def __init__(self, *args) -> None: ...
    # 返回向量的长度
    def __len__(self) -> int: ...
    # 将向量转换为对象类型的 NumPy 数组
    def to_array(self) -> npt.NDArray[np.object_]: ...

class HashTable:
    # 注意：基本的 HashTable 类并不实际包含以下这些方法；
    # 我们在这里为了 mypy 的缘故而将它们放在这里，以避免在每个子类中重复定义。
    def __init__(self, size_hint: int = ..., uses_mask: bool = ...) -> None: ...
    # 返回哈希表中元素的数量
    def __len__(self) -> int: ...
    # 检查键是否在哈希表中
    def __contains__(self, key: Hashable) -> bool: ...
    # 返回哈希表的深度大小（即占用内存大小）
    def sizeof(self, deep: bool = ...) -> int: ...
    # 返回哈希表的状态信息
    def get_state(self) -> dict[str, int]: ...
    # TODO: `val/key` 类型是特定于子类的
    # 获取哈希表中的元素值，返回类型需进一步定义
    def get_item(self, val): ...
    # 设置哈希表中的键值对
    def set_item(self, key, val) -> None: ...
    # 获取哈希表的 NA（not available）值，返回类型需进一步定义
    def get_na(self): ...
    # 设置哈希表的 NA（not available）值
    def set_na(self, val) -> None: ...
    # 将值映射到哈希表的位置
    def map_locations(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        mask: npt.NDArray[np.bool_] | None = ...,
    ) -> None: ...
    # 查找值在哈希表中的位置
    def lookup(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        mask: npt.NDArray[np.bool_] | None = ...,
    ) -> npt.NDArray[np.intp]: ...
    # 获取标签对应的哈希表索引
    def get_labels(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        uniques,  # SubclassTypeVector
        count_prior: int = ...,
        na_sentinel: int = ...,
        na_value: object = ...,
        mask=...,
    ) -> npt.NDArray[np.intp]: ...
    @overload
    # 唯一值查找方法的重载定义，返回类型需进一步定义
    def unique(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        *,
        return_inverse: Literal[False] = ...,
        mask: None = ...,
    ) -> np.ndarray: ...
    @overload
    # 唯一值查找方法的重载定义，返回类型需进一步定义
    def unique(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        *,
        return_inverse: Literal[True],
        mask: None = ...,
    ) -> tuple[np.ndarray, npt.NDArray[np.intp]]: ...
    @overload
    # 唯一值查找方法的重载定义，返回类型需进一步定义
    def unique(
        self,
        values: np.ndarray,  # np.ndarray[特定于子类的类型]
        *,
        return_inverse: Literal[False] = ...,
        mask: npt.NDArray[np.bool_],
    def factorize(
        self,
        values: np.ndarray,  # 输入参数 values 是一个 NumPy 数组，用于进行因子化操作
        na_sentinel: int = ...,  # na_sentinel 是一个整数，表示缺失值的哨兵值
        na_value: object = ...,  # na_value 是一个对象，表示缺失值的实际值
        mask=...,  # mask 是一个可选参数，用于指定值的掩码
        ignore_na: bool = True,  # ignore_na 是一个布尔值，表示是否忽略缺失值
    ) -> tuple[np.ndarray, npt.NDArray[np.intp]]:  # 返回一个元组，包含两个 NumPy 数组，分别是因子化后的值和其对应的整数索引
        ...  # 这里是因子化函数的具体实现，具体操作依赖于子类特定的实现方式

    def hash_inner_join(
        self, values: np.ndarray, mask=...  # 输入参数 values 是一个 NumPy 数组，用于哈希内连接操作；mask 是可选的掩码参数
    ) -> tuple[np.ndarray, np.ndarray]:  # 返回一个元组，包含两个 NumPy 数组，分别是内连接后的结果
        ...  # 这里是哈希内连接的具体实现，但具体操作依赖于子类的实现方式
class Complex128HashTable(HashTable):
    # Complex128HashTable 类继承自 HashTable

class Complex64HashTable(HashTable):
    # Complex64HashTable 类继承自 HashTable

class Float64HashTable(HashTable):
    # Float64HashTable 类继承自 HashTable

class Float32HashTable(HashTable):
    # Float32HashTable 类继承自 HashTable

class Int64HashTable(HashTable):
    # Int64HashTable 类继承自 HashTable
    # 仅 Int64HashTable 类有 get_labels_groupby 和 map_keys_to_values 方法

    def get_labels_groupby(
        self,
        values: npt.NDArray[np.int64],  # const int64_t[:]
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
        # 接受一个 int64 类型的数组 values 作为参数，返回一个元组，包含两个数组：索引数组和 int64 类型的数组

    def map_keys_to_values(
        self,
        keys: npt.NDArray[np.int64],
        values: npt.NDArray[np.int64],  # const int64_t[:]
    ) -> None:
        # 接受两个参数，一个是 int64 类型的 keys 数组，另一个是 int64 类型的 values 数组，没有返回值

class Int32HashTable(HashTable):
    # Int32HashTable 类继承自 HashTable

class Int16HashTable(HashTable):
    # Int16HashTable 类继承自 HashTable

class Int8HashTable(HashTable):
    # Int8HashTable 类继承自 HashTable

class UInt64HashTable(HashTable):
    # UInt64HashTable 类继承自 HashTable

class UInt32HashTable(HashTable):
    # UInt32HashTable 类继承自 HashTable

class UInt16HashTable(HashTable):
    # UInt16HashTable 类继承自 HashTable

class UInt8HashTable(HashTable):
    # UInt8HashTable 类继承自 HashTable

class StringHashTable(HashTable):
    # StringHashTable 类继承自 HashTable

class PyObjectHashTable(HashTable):
    # PyObjectHashTable 类继承自 HashTable

class IntpHashTable(HashTable):
    # IntpHashTable 类继承自 HashTable

def duplicated(
    values: np.ndarray,
    keep: Literal["last", "first", False] = ...,
    mask: npt.NDArray[np.bool_] | None = ...,
) -> npt.NDArray[np.bool_]:
    # 接受一个 np.ndarray 参数 values，一个可选的 keep 参数，以及一个可选的 mask 参数，
    # 返回一个布尔类型的数组，表示 values 中是否存在重复的元素

def mode(
    values: np.ndarray,
    dropna: bool,
    mask: npt.NDArray[np.bool_] | None = ...
) -> np.ndarray:
    # 接受一个 np.ndarray 参数 values，一个布尔类型的 dropna 参数，以及一个可选的 mask 参数，
    # 返回一个数组，表示 values 中的众数（出现频率最高的值）

def value_count(
    values: np.ndarray,
    dropna: bool,
    mask: npt.NDArray[np.bool_] | None = ...,
) -> tuple[np.ndarray, npt.NDArray[np.int64], int]:
    # 接受一个 np.ndarray 参数 values，一个布尔类型的 dropna 参数，以及一个可选的 mask 参数，
    # 返回一个元组，包含三个数组：唯一值数组、对应唯一值的计数数组和总计数值

def ismember(
    arr: np.ndarray,
    values: np.ndarray,
) -> npt.NDArray[np.bool_]:
    # 接受两个 np.ndarray 参数 arr 和 values，返回一个布尔类型的数组，
    # 表示 arr 中的每个元素是否在 values 中存在

def object_hash(obj) -> int:
    # 接受一个对象 obj，返回其哈希值的整数表示

def objects_are_equal(a, b) -> bool:
    # 接受两个对象 a 和 b，返回一个布尔值，表示这两个对象是否相等
```