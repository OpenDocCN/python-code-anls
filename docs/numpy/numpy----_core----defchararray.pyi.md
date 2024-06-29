# `D:\src\scipysrc\numpy\numpy\_core\defchararray.pyi`

```py
# 引入必要的类型提示和函数重载支持
from typing import (
    Literal as L,               # 别名 L 表示 Literal
    overload,                   # 装饰器 overload 支持函数重载
    TypeVar,                    # 泛型变量 TypeVar
    Any,                        # 表示任意类型的 Any
    SupportsIndex,              # 表示支持索引的类型
    SupportsInt,                # 表示支持整数的类型
)

import numpy as np               # 导入 NumPy 库
from numpy import (              # 从 NumPy 中导入以下标识符
    ndarray,                     # 多维数组类型 ndarray
    dtype,                       # 数据类型 dtype
    str_ as _StrDType,           # 字符串类型 str_ 别名为 _StrDType
    bytes_ as _BytesDType,       # 字节类型 bytes_ 别名为 _BytesDType
    int_ as _IntDType,           # 整数类型 int_ 别名为 _IntDType
    object_ as _ObjectDType,     # 对象类型 object_ 别名为 _ObjectDType
    _OrderKACF,                  # 私有类型 _OrderKACF
    _ShapeType,                  # 私有类型 _ShapeType
    _CharDType,                  # 私有类型 _CharDType
    _SupportsBuffer,             # 私有类型 _SupportsBuffer
)

from numpy._typing import (       # 从 NumPy 的类型定义中导入以下标识符
    NDArray,                      # 泛型数组类型 NDArray
    _ShapeLike,                   # 形状类型 _ShapeLike
    _ArrayLikeStr_co as U_co,     # 可变字符串数组类型别名 U_co
    _ArrayLikeBytes_co as S_co,   # 可变字节字符串数组类型别名 S_co
    _ArrayLikeInt_co as i_co,     # 可变整数数组类型别名 i_co
    _ArrayLikeBool_co as b_co,    # 可变布尔数组类型别名 b_co
)

from numpy._core.multiarray import compare_chararrays as compare_chararrays  # 导入字符数组比较函数

_SCT = TypeVar("_SCT", str_, bytes_)      # 定义 _SCT 泛型变量，可以是 str 或 bytes 类型
_CharArray = chararray[Any, dtype[_SCT]]  # 定义 _CharArray 类型为 chararray 类的泛型实例

# chararray 类的定义，继承自 ndarray，支持字符数据类型 _CharDType
class chararray(ndarray[_ShapeType, _CharDType]):
    
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,                 # 形状参数，类型为 _ShapeLike
        itemsize: SupportsIndex | SupportsInt = ...,  # 元素大小，支持索引或整数
        unicode: L[False] = ...,           # 是否是 Unicode 类型的标志，False 表示不是
        buffer: _SupportsBuffer = ...,     # 缓冲区对象，支持 _SupportsBuffer
        offset: SupportsIndex = ...,       # 偏移量，支持索引
        strides: _ShapeLike = ...,         # 步幅，类型为 _ShapeLike
        order: _OrderKACF = ...,          # 排序顺序，类型为 _OrderKACF
    ) -> chararray[Any, dtype[bytes_]]: ...  # 返回 chararray 类型的实例，字符数据类型为 bytes_

    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        itemsize: SupportsIndex | SupportsInt = ...,
        unicode: L[True] = ...,            # 是否是 Unicode 类型的标志，True 表示是
        buffer: _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> chararray[Any, dtype[str_]]: ...  # 返回 chararray 类型的实例，字符数据类型为 str_

    def __array_finalize__(self, obj: object) -> None: ...  # 数组初始化之后的清理方法，无返回值
    def __mul__(self, other: i_co) -> chararray[Any, _CharDType]: ...  # 乘法操作，返回 chararray 类型的实例
    def __rmul__(self, other: i_co) -> chararray[Any, _CharDType]: ...  # 右乘法操作，返回 chararray 类型的实例
    def __mod__(self, i: Any) -> chararray[Any, _CharDType]: ...  # 取模操作，返回 chararray 类型的实例

    @overload
    def __eq__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]

    @overload
    def __eq__(
        self: _CharArray[bytes_],  # 字符数组类型为 bytes_
        other: S_co,               # 另一数组的类型为 S_co
    ) -> NDArray[np.bool]: ...     # 返回布尔数组 NDArray[np.bool]

    @overload
    def __ne__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]

    @overload
    def __ne__(
        self: _CharArray[bytes_],  # 字符数组类型为 bytes_
        other: S_co,               # 另一数组的类型为 S_co
    ) -> NDArray[np.bool]: ...     # 返回布尔数组 NDArray[np.bool]

    @overload
    def __ge__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]

    @overload
    def __ge__(
        self: _CharArray[bytes_],  # 字符数组类型为 bytes_
        other: S_co,               # 另一数组的类型为 S_co
    ) -> NDArray[np.bool]: ...     # 返回布尔数组 NDArray[np.bool]

    @overload
    def __le__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]

    @overload
    def __le__(
        self: _CharArray[bytes_],  # 字符数组类型为 bytes_
        other: S_co,               # 另一数组的类型为 S_co
    ) -> NDArray[np.bool]: ...     # 返回布尔数组 NDArray[np.bool]

    @overload
    def __gt__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]

    @overload
    def __gt__(
        self: _CharArray[bytes_],  # 字符数组类型为 bytes_
        other: S_co,               # 另一数组的类型为 S_co
    ) -> NDArray[np.bool]: ...     # 返回布尔数组 NDArray[np.bool]

    @overload
    def __lt__(
        self: _CharArray[str_],   # 字符数组类型为 str_
        other: U_co,              # 另一数组的类型为 U_co
    ) -> NDArray[np.bool]: ...    # 返回布尔数组 NDArray[np.bool]
    def __lt__(
        self: _CharArray[bytes_],
        other: S_co,
    ) -> NDArray[np.bool]:
    # 定义小于比较方法，返回一个布尔类型的 NumPy 数组
    ...

    @overload
    def __add__(
        self: _CharArray[str_],
        other: U_co,
    ) -> _CharArray[str_]:
    # 定义字符串数组与其他类型相加的重载方法，返回字符串数组
    @overload
    def __add__(
        self: _CharArray[bytes_],
        other: S_co,
    ) -> _CharArray[bytes_]:
    # 定义字节串数组与其他类型相加的重载方法，返回字节串数组
    ...

    @overload
    def __radd__(
        self: _CharArray[str_],
        other: U_co,
    ) -> _CharArray[str_]:
    # 定义其他类型与字符串数组相加的反向重载方法，返回字符串数组
    @overload
    def __radd__(
        self: _CharArray[bytes_],
        other: S_co,
    ) -> _CharArray[bytes_]:
    # 定义其他类型与字节串数组相加的反向重载方法，返回字节串数组
    ...

    @overload
    def center(
        self: _CharArray[str_],
        width: i_co,
        fillchar: U_co = ...,
    ) -> _CharArray[str_]:
    # 定义字符串数组的居中方法，返回字符串数组
    @overload
    def center(
        self: _CharArray[bytes_],
        width: i_co,
        fillchar: S_co = ...,
    ) -> _CharArray[bytes_]:
    # 定义字节串数组的居中方法，返回字节串数组
    ...

    @overload
    def count(
        self: _CharArray[str_],
        sub: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字符串数组的计数方法，返回整型的 NumPy 数组
    @overload
    def count(
        self: _CharArray[bytes_],
        sub: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字节串数组的计数方法，返回整型的 NumPy 数组
    ...

    def decode(
        self: _CharArray[bytes_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[str_]:
    # 定义字节串数组的解码方法，返回字符串数组
    ...

    def encode(
        self: _CharArray[str_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[bytes_]:
    # 定义字符串数组的编码方法，返回字节串数组
    ...

    @overload
    def endswith(
        self: _CharArray[str_],
        suffix: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[np.bool]:
    # 定义字符串数组的后缀判断方法，返回布尔类型的 NumPy 数组
    @overload
    def endswith(
        self: _CharArray[bytes_],
        suffix: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[np.bool]:
    # 定义字节串数组的后缀判断方法，返回布尔类型的 NumPy 数组
    ...

    def expandtabs(
        self,
        tabsize: i_co = ...,
    ) -> chararray[Any, _CharDType]:
    # 定义替换制表符为多个空格的方法，返回字符数组
    ...

    @overload
    def find(
        self: _CharArray[str_],
        sub: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字符串数组的查找方法，返回整型的 NumPy 数组
    @overload
    def find(
        self: _CharArray[bytes_],
        sub: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字节串数组的查找方法，返回整型的 NumPy 数组
    ...

    @overload
    def index(
        self: _CharArray[str_],
        sub: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字符串数组的索引方法，返回整型的 NumPy 数组
    @overload
    def index(
        self: _CharArray[bytes_],
        sub: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]:
    # 定义字节串数组的索引方法，返回整型的 NumPy 数组
    ...

    @overload
    def join(
        self: _CharArray[str_],
        seq: U_co,
    ) -> _CharArray[str_]:
    # 定义字符串数组的连接方法，返回字符串数组
    @overload
    def join(
        self: _CharArray[bytes_],
        seq: S_co,
    ) -> _CharArray[bytes_]:
    # 定义字节串数组的连接方法，返回字节串数组
    ...

    @overload
    def ljust(
        self: _CharArray[str_],
        width: i_co,
        fillchar: U_co = ...,

        width: i_co,
        fillchar: U_co = ...,
    ) -> _CharArray[str_]:
    # 定义字符串数组的左对齐方法，返回字符串数组
    @overload
    def ljust(
        self: _CharArray[bytes_],
        width: i_co,
        fillchar: S_co = ...,
    ) -> _CharArray[bytes_]:
    # 定义字节串数组的左对齐方法，返回字节串数组
    ...
    ) -> _CharArray[str_]: ...
    # 定义 ljust 方法的类型注解和返回类型为字符数组（字符串）的泛型
    @overload
    def ljust(
        self: _CharArray[bytes_],
        width: i_co,
        fillchar: S_co = ...,
    ) -> _CharArray[bytes_]: ...
    # ljust 方法的重载定义，接受一个整数宽度和一个填充字符（字节序列），返回类型为字节序列的字符数组

    @overload
    def lstrip(
        self: _CharArray[str_],
        chars: None | U_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def lstrip(
        self: _CharArray[bytes_],
        chars: None | S_co = ...,
    ) -> _CharArray[bytes_]: ...
    # lstrip 方法的重载定义，接受一个可选的字符集参数，返回类型根据输入的字符数组类型而定

    @overload
    def partition(
        self: _CharArray[str_],
        sep: U_co,
    ) -> _CharArray[str_]: ...
    @overload
    def partition(
        self: _CharArray[bytes_],
        sep: S_co,
    ) -> _CharArray[bytes_]: ...
    # partition 方法的重载定义，接受一个分隔符参数，返回类型根据输入的字符数组类型而定

    @overload
    def replace(
        self: _CharArray[str_],
        old: U_co,
        new: U_co,
        count: None | i_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def replace(
        self: _CharArray[bytes_],
        old: S_co,
        new: S_co,
        count: None | i_co = ...,
    ) -> _CharArray[bytes_]: ...
    # replace 方法的重载定义，接受一个旧值、一个新值和一个可选的计数参数，返回类型根据输入的字符数组类型而定

    @overload
    def rfind(
        self: _CharArray[str_],
        sub: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rfind(
        self: _CharArray[bytes_],
        sub: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]: ...
    # rfind 方法的重载定义，接受一个子字符串、可选的起始位置和结束位置参数，返回类型为整数数组（numpy 数组）

    @overload
    def rindex(
        self: _CharArray[str_],
        sub: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rindex(
        self: _CharArray[bytes_],
        sub: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[int_]: ...
    # rindex 方法的重载定义，接受一个子字符串、可选的起始位置和结束位置参数，返回类型为整数数组（numpy 数组）

    @overload
    def rjust(
        self: _CharArray[str_],
        width: i_co,
        fillchar: U_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rjust(
        self: _CharArray[bytes_],
        width: i_co,
        fillchar: S_co = ...,
    ) -> _CharArray[bytes_]: ...
    # rjust 方法的重载定义，接受一个整数宽度和一个填充字符（字符或字节序列），返回类型根据输入的字符数组类型而定

    @overload
    def rpartition(
        self: _CharArray[str_],
        sep: U_co,
    ) -> _CharArray[str_]: ...
    @overload
    def rpartition(
        self: _CharArray[bytes_],
        sep: S_co,
    ) -> _CharArray[bytes_]: ...
    # rpartition 方法的重载定义，接受一个分隔符参数，返回类型根据输入的字符数组类型而定

    @overload
    def rsplit(
        self: _CharArray[str_],
        sep: None | U_co = ...,
        maxsplit: None | i_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    def rsplit(
        self: _CharArray[bytes_],
        sep: None | S_co = ...,
        maxsplit: None | i_co = ...,
    ) -> NDArray[object_]: ...
    # rsplit 方法的重载定义，接受一个可选的分隔符和最大分割次数参数，返回类型为对象数组（numpy 数组）

    @overload
    def rstrip(
        self: _CharArray[str_],
        chars: None | U_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rstrip(
        self: _CharArray[bytes_],
        chars: None | S_co = ...,
    ) -> _CharArray[bytes_]: ...
    # rstrip 方法的重载定义，接受一个可选的字符集参数，返回类型根据输入的字符数组类型而定

    @overload
    def split(
        self: _CharArray[str_],
        sep: None | U_co = ...,
        maxsplit: None | i_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    # 定义一个方法 `split`，接受一些参数，并返回一个对象数组 NDArray[object_]
    def split(
        self: _CharArray[bytes_],
        sep: None | S_co = ...,
        maxsplit: None | i_co = ...,
    ) -> NDArray[object_]: ...

    # 定义一个方法 `splitlines`，接受一个参数 keepends，并返回一个对象数组 NDArray[object_]
    def splitlines(self, keepends: None | b_co = ...) -> NDArray[object_]: ...

    # 声明一个装饰器函数 `startswith` 的重载，接受不同的参数类型，返回一个布尔类型的数组 NDArray[np.bool]
    @overload
    def startswith(
        self: _CharArray[str_],
        prefix: U_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[np.bool]: ...
    @overload
    def startswith(
        self: _CharArray[bytes_],
        prefix: S_co,
        start: i_co = ...,
        end: None | i_co = ...,
    ) -> NDArray[np.bool]: ...

    # 声明一个装饰器函数 `strip` 的重载，接受不同的参数类型，返回一个 `_CharArray` 类型的对象数组
    @overload
    def strip(
        self: _CharArray[str_],
        chars: None | U_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def strip(
        self: _CharArray[bytes_],
        chars: None | S_co = ...,
    ) -> _CharArray[bytes_]: ...

    # 声明一个装饰器函数 `translate` 的重载，接受不同的参数类型，返回一个 `_CharArray` 类型的对象数组
    @overload
    def translate(
        self: _CharArray[str_],
        table: U_co,
        deletechars: None | U_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def translate(
        self: _CharArray[bytes_],
        table: S_co,
        deletechars: None | S_co = ...,
    ) -> _CharArray[bytes_]: ...

    # 定义一个方法 `zfill`，接受一个参数 width，返回一个特定形状和字符数据类型的对象数组 chararray[Any, _CharDType]
    def zfill(self, width: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...

    # 定义一个方法 `capitalize`，不接受参数，返回一个特定形状和字符数据类型的对象数组 chararray[_ShapeType, _CharDType]
    def capitalize(self) -> chararray[_ShapeType, _CharDType]: ...

    # 定义一个方法 `title`，不接受参数，返回一个特定形状和字符数据类型的对象数组 chararray[_ShapeType, _CharDType]
    def title(self) -> chararray[_ShapeType, _CharDType]: ...

    # 定义一个方法 `swapcase`，不接受参数，返回一个特定形状和字符数据类型的对象数组 chararray[_ShapeType, _CharDType]
    def swapcase(self) -> chararray[_ShapeType, _CharDType]: ...

    # 定义一个方法 `lower`，不接受参数，返回一个特定形状和字符数据类型的对象数组 chararray[_ShapeType, _CharDType]
    def lower(self) -> chararray[_ShapeType, _CharDType]: ...

    # 定义一个方法 `upper`，不接受参数，返回一个特定形状和字符数据类型的对象数组 chararray[_ShapeType, _CharDType]
    def upper(self) -> chararray[_ShapeType, _CharDType]: ...

    # 定义一个方法 `isalnum`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isalnum(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isalpha`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isalpha(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isdigit`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isdigit(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `islower`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def islower(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isspace`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isspace(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `istitle`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def istitle(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isupper`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isupper(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isnumeric`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isnumeric(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...

    # 定义一个方法 `isdecimal`，不接受参数，返回一个特定形状和布尔数据类型的数组 ndarray[_ShapeType, dtype[np.bool]]
    def isdecimal(self) -> ndarray[_ShapeType, dtype[np.bool]]: ...
__all__: list[str]
# 定义模块中所有公开的函数和变量名称列表

# Comparison
@overload
def equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的equal函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

@overload
def not_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def not_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的not_equal函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

@overload
def greater_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的greater_equal函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

@overload
def less_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的less_equal函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

@overload
def greater(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的greater函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

@overload
def less(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...
# 函数重载：定义两个版本的less函数，分别处理不同类型的输入参数，并返回布尔类型的NumPy数组

# String operations
@overload
def add(x1: U_co, x2: U_co) -> NDArray[str_]: ...
@overload
def add(x1: S_co, x2: S_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的add函数，分别处理字符串和字节串的输入参数，并返回字符串或字节串的NumPy数组

@overload
def multiply(a: U_co, i: i_co) -> NDArray[str_]: ...
@overload
def multiply(a: S_co, i: i_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的multiply函数，分别处理字符串和字节串的输入参数，并返回重复字符串或字节串的NumPy数组

@overload
def mod(a: U_co, value: Any) -> NDArray[str_]: ...
@overload
def mod(a: S_co, value: Any) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的mod函数，分别处理字符串和字节串的输入参数，并返回格式化后的字符串或字节串的NumPy数组

@overload
def capitalize(a: U_co) -> NDArray[str_]: ...
@overload
def capitalize(a: S_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的capitalize函数，分别处理字符串和字节串的输入参数，并返回首字母大写的字符串或字节串的NumPy数组

@overload
def center(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[str_]: ...
@overload
def center(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的center函数，分别处理字符串和字节串的输入参数，并返回居中对齐的字符串或字节串的NumPy数组

def decode(
    a: S_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[str_]: ...
# decode函数：解码字节串为字符串，并返回NumPy数组

def encode(
    a: U_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[bytes_]: ...
# encode函数：编码字符串为字节串，并返回NumPy数组

@overload
def expandtabs(a: U_co, tabsize: i_co = ...) -> NDArray[str_]: ...
@overload
def expandtabs(a: S_co, tabsize: i_co = ...) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的expandtabs函数，分别处理字符串和字节串的输入参数，并返回扩展制表符的字符串或字节串的NumPy数组

@overload
def join(sep: U_co, seq: U_co) -> NDArray[str_]: ...
@overload
def join(sep: S_co, seq: S_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的join函数，分别处理字符串和字节串的输入参数，并返回连接序列的字符串或字节串的NumPy数组

@overload
def ljust(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[str_]: ...
@overload
def ljust(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的ljust函数，分别处理字符串和字节串的输入参数，并返回左对齐的字符串或字节串的NumPy数组

@overload
def lower(a: U_co) -> NDArray[str_]: ...
@overload
def lower(a: S_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的lower函数，分别处理字符串和字节串的输入参数，并返回小写化的字符串或字节串的NumPy数组

@overload
def lstrip(a: U_co, chars: None | U_co = ...) -> NDArray[str_]: ...
@overload
def lstrip(a: S_co, chars: None | S_co = ...) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的lstrip函数，分别处理字符串和字节串的输入参数，并返回去除左侧空白字符的字符串或字节串的NumPy数组

@overload
def partition(a: U_co, sep: U_co) -> NDArray[str_]: ...
@overload
def partition(a: S_co, sep: S_co) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的partition函数，分别处理字符串和字节串的输入参数，并返回分割后的字符串或字节串的NumPy数组

@overload
def replace(
    a: U_co,
    old: U_co,
    new: U_co,
    count: None | i_co = ...,
) -> NDArray[str_]: ...
@overload
def replace(
    a: S_co,
    old: S_co,
    new: S_co,
    count: None | i_co = ...,
) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的replace函数，分别处理字符串和字节串的输入参数，并返回替换后的字符串或字节串的NumPy数组

@overload
def rjust(
    a: U_co,
    width: i_co,
    fillchar: U_co = ...,
) -> NDArray[str_]: ...
@overload
def rjust(
    a: S_co,
    width: i_co,
    fillchar: S_co = ...,
) -> NDArray[bytes_]: ...
# 函数重载：定义两个版本的rjust函数，分别处理字符串和字节串的输入参数，并返回右对齐的字符串或字节串的NumPy数组
    a: S_co,            # 定义变量a，类型为S_co
    width: i_co,        # 定义变量width，类型为i_co
    fillchar: S_co = ...,  # 定义变量fillchar，类型为S_co，并初始化为...
# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def rpartition(a: U_co, sep: U_co) -> NDArray[bytes_]: ...
@overload
def rpartition(a: S_co, sep: S_co) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[object_]，表示函数返回一个对象数组
@overload
def rsplit(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[object_]: ...
@overload
def rsplit(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[object_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def rstrip(a: U_co, chars: None | U_co = ...) -> NDArray[bytes_]: ...
@overload
def rstrip(a: S_co, chars: None | S_co = ...) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[object_]，表示函数返回一个对象数组
@overload
def split(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[object_]: ...
@overload
def split(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[object_]: ...

# 返回类型注释为 NDArray[object_]，表示函数返回一个对象数组
@overload
def splitlines(a: U_co, keepends: None | b_co = ...) -> NDArray[object_]: ...
@overload
def splitlines(a: S_co, keepends: None | b_co = ...) -> NDArray[object_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def strip(a: U_co, chars: None | U_co = ...) -> NDArray[bytes_]: ...
@overload
def strip(a: S_co, chars: None | S_co = ...) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def swapcase(a: U_co) -> NDArray[bytes_]: ...
@overload
def swapcase(a: S_co) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def title(a: U_co) -> NDArray[bytes_]: ...
@overload
def title(a: S_co) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def translate(
    a: U_co,
    table: U_co,
    deletechars: None | U_co = ...,
) -> NDArray[bytes_]: ...
@overload
def translate(
    a: S_co,
    table: S_co,
    deletechars: None | S_co = ...,
) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def upper(a: U_co) -> NDArray[bytes_]: ...
@overload
def upper(a: S_co) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[bytes_]，表示函数返回一个字节串数组
@overload
def zfill(a: U_co, width: i_co) -> NDArray[bytes_]: ...
@overload
def zfill(a: S_co, width: i_co) -> NDArray[bytes_]: ...

# 返回类型注释为 NDArray[int_]，表示函数返回一个整数数组
# 字符串信息统计
@overload
def count(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...
@overload
def count(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

# 返回类型注释为 NDArray[np.bool]，表示函数返回一个布尔值数组
@overload
def endswith(
    a: U_co,
    suffix: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.bool]: ...
@overload
def endswith(
    a: S_co,
    suffix: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.bool]: ...

# 返回类型注释为 NDArray[int_]，表示函数返回一个整数数组
@overload
def find(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...
@overload
def find(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

# 返回类型注释为 NDArray[int_]，表示函数返回一个整数数组
@overload
def index(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...
@overload
def index(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

# 返回类型注释为 NDArray[np.bool]，表示函数返回一个布尔值数组
def isalpha(a: U_co | S_co) -> NDArray[np.bool]: ...

# 返回类型注释为 NDArray[np.bool]，表示函数返回一个布尔值数组
def isalnum(a: U_co | S_co) -> NDArray[np.bool]: ...
# 判断参数 a 是否是一个可接受的 Unicode 字符，并返回一个布尔值数组
def isdecimal(a: U_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否是一个可接受的 Unicode 数字，并返回一个布尔值数组
def isdigit(a: U_co | S_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否是小写字母，并返回一个布尔值数组
def islower(a: U_co | S_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否是一个可接受的 Unicode 数字，并返回一个布尔值数组
def isnumeric(a: U_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否是空格，并返回一个布尔值数组
def isspace(a: U_co | S_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否符合标题化格式，并返回一个布尔值数组
def istitle(a: U_co | S_co) -> NDArray[np.bool]: ...

# 判断参数 a 是否是大写字母，并返回一个布尔值数组
def isupper(a: U_co | S_co) -> NDArray[np.bool]: ...

# Overload 1 and 2: 在字符串或字节串中查找子串 sub，从后往前搜索，返回索引数组
# Overload 3 and 4: 在对象中查找子串 sub，从后往前搜索，返回索引数组
@overload
def rfind(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

@overload
def rfind(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

# Overload 1 and 2: 在字符串或字节串中查找子串 sub，从后往前搜索，返回索引数组
# Overload 3 and 4: 在对象中查找子串 sub，从后往前搜索，返回索引数组
@overload
def rindex(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

@overload
def rindex(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[int_]: ...

# Overload 1 and 2: 检查字符串或字节串是否以指定前缀开头，并返回布尔值数组
# Overload 3 and 4: 检查对象是否以指定前缀开头，并返回布尔值数组
@overload
def startswith(
    a: U_co,
    prefix: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.bool]: ...

@overload
def startswith(
    a: S_co,
    prefix: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.bool]: ...

# 返回字符串或字节串的长度作为整数数组
def str_len(A: U_co | S_co) -> NDArray[int_]: ...

# Overload 1 and 2: 根据对象创建字符数组，可以指定是否复制和字节串的大小
# Overload 3: 对于具有 unicode=False 的任意对象，返回字节串数组
# Overload 4: 对于具有 unicode=True 的任意对象，返回字符串数组
@overload
def array(
    obj: U_co,
    itemsize: None | int = ...,
    copy: bool = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[str_]: ...

@overload
def array(
    obj: S_co,
    itemsize: None | int = ...,
    copy: bool = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[bytes_]: ...

@overload
def array(
    obj: object,
    itemsize: None | int = ...,
    copy: bool = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[bytes_]: ...

@overload
def array(
    obj: object,
    itemsize: None | int = ...,
    copy: bool = ...,
    unicode: L[True] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[str_]: ...

# 根据对象创建字符数组，可以指定是否复制和字节串的大小
@overload
def asarray(
    obj: U_co,
    itemsize: None | int = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[str_]: ...

@overload
def asarray(
    obj: S_co,
    itemsize: None | int = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[bytes_]: ...

@overload
def asarray(
    obj: object,
    itemsize: None | int = ...,
    unicode: L[False] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[bytes_]: ...

@overload
def asarray(
    obj: object,
    itemsize: None | int = ...,
    unicode: L[True] = ...,
    order: _OrderKACF = ...,
) -> _CharArray[str_]: ...
```