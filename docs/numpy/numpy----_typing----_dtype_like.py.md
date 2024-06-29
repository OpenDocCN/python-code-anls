# `.\numpy\numpy\_typing\_dtype_like.py`

```
# 从 collections.abc 模块导入 Sequence 抽象基类
from collections.abc import Sequence
# 导入类型提示模块中的必要类和函数
from typing import (
    Any,
    Sequence,
    Union,
    TypeVar,
    Protocol,
    TypedDict,
    runtime_checkable,
)
# 导入 NumPy 库并使用 np 别名
import numpy as np

# 导入 _ShapeLike 类
from ._shape import _ShapeLike

# 从 _char_codes 模块导入多个数据类型的编码，包括布尔、整数和浮点数等
from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
    _IntPCodes,
    _IntCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
    _UIntPCodes,
    _UIntCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)

# 定义一个类型变量 _SCT，其类型约束为 np.generic
_SCT = TypeVar("_SCT", bound=np.generic)
# 定义一个协变类型变量 _DType_co，用于 np.dtype 的类型参数
_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype[Any])

# 定义一个任意嵌套的 _DTypeLikeNested 类型的别名，目前没有支持递归类型的支持
_DTypeLikeNested = Any  # TODO: wait for support for recursive types

# 强制存在的键
# 定义一个 TypedDict 类型 _DTypeDictBase，包含 names 和 formats 两个必须键
class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]

# 强制存在和可选存在的键
# 定义一个 TypedDict 类型 _DTypeDict，继承自 _DTypeDictBase，并可选包含 offsets, titles, itemsize 和 aligned 四个键
class _DTypeDict(_DTypeDictBase, total=False):
    # 只有 str 元素可用作索引别名，但 titles 原则上可以接受任何对象
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool

# 一个带有 dtype 属性的任意协议
@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    @property
    def dtype(self) -> _DType_co: ...

# _DTypeLike 的子集，可以关于 np.generic 进行参数化
_DTypeLike = Union[
    np.dtype[_SCT],
    type[_SCT],
    _SupportsDType[np.dtype[_SCT]],
]

# 会创建一个 dtype[np.void]
# _VoidDTypeLike 的类型定义，可以是元组、列表或字典等多种形式
_VoidDTypeLike = Union[
    # (灵活的 dtype, itemsize)
    tuple[_DTypeLikeNested, int],
    # (固定的 dtype, shape)
    tuple[_DTypeLikeNested, _ShapeLike],
    # [(field_name, field_dtype, field_shape), ...]
    #
    # 此处的类型非常广泛，因为 NumPy 接受列表中的多种输入；参见测试用例以获取一些示例。
    list[Any],
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    _DTypeDict,
    # (基础 dtype, 新 dtype)
    tuple[_DTypeLikeNested, _DTypeLikeNested],
]

# 可以被强制转换为 numpy.dtype 的任何类型
# 参考文档：https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike = Union[
    np.dtype[Any],
    # 默认数据类型 (float64)
    None,
    # 数组标量类型和通用类型
    type[Any],  # 注意：由于对象 dtype，我们受限于 `type[Any]`
    # 任何具有 dtype 属性的对象
    _SupportsDType[np.dtype[Any]],
    # 字符串类型，如字符代码、类型字符串或逗号分隔的字段，例如 'float64'
    str,
    _VoidDTypeLike,
]
# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discouraged and
# therefore not included in the Union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.

# 定义布尔类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeBool = Union[
    type[bool],
    type[np.bool],
    np.dtype[np.bool],
    _SupportsDType[np.dtype[np.bool]],
    _BoolCodes,
]

# 定义无符号整数类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeUInt = Union[
    type[np.unsignedinteger],
    np.dtype[np.unsignedinteger],
    _SupportsDType[np.dtype[np.unsignedinteger]],
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _LongCodes,
    _ULongLongCodes,
    _UIntPCodes,
    _UIntCodes,
]

# 定义有符号整数类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeInt = Union[
    type[int],
    type[np.signedinteger],
    np.dtype[np.signedinteger],
    _SupportsDType[np.dtype[np.signedinteger]],
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
    _IntPCodes,
    _IntCodes,
]

# 定义浮点数类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeFloat = Union[
    type[float],
    type[np.floating],
    np.dtype[np.floating],
    _SupportsDType[np.dtype[np.floating]],
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
]

# 定义复数类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeComplex = Union[
    type[complex],
    type[np.complexfloating],
    np.dtype[np.complexfloating],
    _SupportsDType[np.dtype[np.complexfloating]],
    _Complex64Codes,
    _Complex128Codes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
]

# 定义时间日期类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeDT64 = Union[
    type[np.timedelta64],
    np.dtype[np.timedelta64],
    _SupportsDType[np.dtype[np.timedelta64]],
    _TD64Codes,
]

# 定义时间间隔类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeTD64 = Union[
    type[np.datetime64],
    np.dtype[np.datetime64],
    _SupportsDType[np.dtype[np.datetime64]],
    _DT64Codes,
]

# 定义字符串类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeStr = Union[
    type[str],
    type[np.str_],
    np.dtype[np.str_],
    _SupportsDType[np.dtype[np.str_]],
    _StrCodes,
]

# 定义字节串类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeBytes = Union[
    type[bytes],
    type[np.bytes_],
    np.dtype[np.bytes_],
    _SupportsDType[np.dtype[np.bytes_]],
    _BytesCodes,
]

# 定义空类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeVoid = Union[
    type[np.void],
    np.dtype[np.void],
    _SupportsDType[np.dtype[np.void]],
    _VoidCodes,
    _VoidDTypeLike,
]

# 定义对象类型相关的数据类型别名，包括各种可能的类型和代码。
_DTypeLikeObject = Union[
    type,
    np.dtype[np.object_],
    _SupportsDType[np.dtype[np.object_]],
    _ObjectCodes,
]

# 定义复合类型的数据类型别名，包括布尔、整数、浮点数和复数类型。
_DTypeLikeComplex_co = Union[
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
]
```