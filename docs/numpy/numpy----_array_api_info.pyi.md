# `D:\src\scipysrc\numpy\numpy\_array_api_info.pyi`

```py
from typing import TypedDict, Optional, Union, Tuple, List
from numpy._typing import DtypeLike

# 定义 TypedDict 类型 Capabilities，表示数组的功能特性
Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,  # 指示是否支持布尔索引
        "data-dependent shapes": bool,  # 指示是否支持依赖数据的形状
    },
)

# 定义 TypedDict 类型 DefaultDataTypes，表示默认数据类型
DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": DtypeLike,  # 实数浮点类型的默认数据类型
        "complex floating": DtypeLike,  # 复数浮点类型的默认数据类型
        "integral": DtypeLike,  # 整数类型的默认数据类型
        "indexing": DtypeLike,  # 索引类型的默认数据类型
    },
)

# 定义 TypedDict 类型 DataTypes，表示各种数据类型的映射
DataTypes = TypedDict(
    "DataTypes",
    {
        "bool": DtypeLike,  # 布尔类型的数据类型
        "float32": DtypeLike,  # 32位浮点数的数据类型
        "float64": DtypeLike,  # 64位浮点数的数据类型
        "complex64": DtypeLike,  # 64位复数的数据类型
        "complex128": DtypeLike,  # 128位复数的数据类型
        "int8": DtypeLike,  # 8位整数的数据类型
        "int16": DtypeLike,  # 16位整数的数据类型
        "int32": DtypeLike,  # 32位整数的数据类型
        "int64": DtypeLike,  # 64位整数的数据类型
        "uint8": DtypeLike,  # 8位无符号整数的数据类型
        "uint16": DtypeLike,  # 16位无符号整数的数据类型
        "uint32": DtypeLike,  # 32位无符号整数的数据类型
        "uint64": DtypeLike,  # 64位无符号整数的数据类型
    },
    total=False,  # 允许额外的未定义键
)

# 定义 __array_namespace_info__ 类
class __array_namespace_info__:
    __module__: str  # 模块名称

    # 返回数组命名空间的功能特性
    def capabilities(self) -> Capabilities: ...

    # 返回默认设备名称
    def default_device(self) -> str: ...

    # 返回默认数据类型的映射
    def default_dtypes(
        self,
        *,
        device: Optional[str] = None,
    ) -> DefaultDataTypes: ...

    # 返回指定设备和类型种类的数据类型映射
    def dtypes(
        self,
        *,
        device: Optional[str] = None,
        kind: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> DataTypes: ...

    # 返回支持的设备列表
    def devices(self) -> List[str]: ...
```