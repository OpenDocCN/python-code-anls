# `D:\src\scipysrc\numpy\numpy\_core\records.pyi`

```
# 导入标准库 os，用于与操作系统交互
import os
# 从 collections.abc 模块导入 Sequence 和 Iterable 类型，用于声明支持序列和可迭代的抽象基类
from collections.abc import Sequence, Iterable
# 导入 typing 模块中的各种类型，用于类型提示
from typing import (
    Any,
    TypeVar,
    overload,
    Protocol,
    SupportsIndex,
    Literal
)
# 从 numpy 库导入特定的类型和类
from numpy import (
    ndarray,
    dtype,
    generic,
    void,
    _ByteOrder,
    _SupportsBuffer,
    _ShapeType,
    _DType_co,
    _OrderKACF,
)
# 从 numpy._typing 模块导入特定的类型别名
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ShapeLike,
    _ArrayLikeInt_co,
    _ArrayLikeVoid_co,
    _NestedSequence,
)

# 创建一个泛型类型变量 _SCT，其上界为 generic 类型
_SCT = TypeVar("_SCT", bound=generic)

# 定义一个 _RecArray 类型别名，表示元素类型为任意类型的 recarray 类
_RecArray = recarray[Any, dtype[_SCT]]

# 定义一个协议 _SupportsReadInto，规定实现了 seek、tell 和 readinto 方法
class _SupportsReadInto(Protocol):
    def seek(self, offset: int, whence: int, /) -> object: ...
    def tell(self, /) -> int: ...
    def readinto(self, buffer: memoryview, /) -> int: ...

# 定义一个 record 类，继承自 void 类
class record(void):
    # 重载 __getattribute__ 方法，接受一个字符串参数，返回任意类型的对象
    def __getattribute__(self, attr: str) -> Any: ...
    # 重载 __setattr__ 方法，接受一个字符串参数和一个 ArrayLike 类型的值，无返回值
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    # 定义 pprint 方法，返回一个字符串表示对象的打印输出
    def pprint(self) -> str: ...
    # 重载 __getitem__ 方法，接受一个键参数，返回任意类型的对象
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    # 重载 __getitem__ 方法，接受一个列表参数，返回 record 类型的对象
    @overload
    def __getitem__(self, key: list[str]) -> record: ...

# 定义一个 recarray 类，继承自 ndarray 类，元素类型为 _ShapeType 和 _DType_co 类型的泛型
class recarray(ndarray[_ShapeType, _DType_co]):
    # NOTE: While not strictly mandatory, we're demanding here that arguments
    # for the `format_parser`- and `dtype`-based dtype constructors are
    # mutually exclusive
    # 重载 __new__ 方法，根据参数不同进行不同的构造，返回一个 recarray 类型的对象
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: None = ...,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        *,
        formats: DTypeLike,
        names: None | str | Sequence[str] = ...,
        titles: None | str | Sequence[str] = ...,
        byteorder: None | _ByteOrder = ...,
        aligned: bool = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[record]]: ...
    # 重载 __new__ 方法，根据参数不同进行不同的构造，返回一个 recarray 类型的对象
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: DTypeLike,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        formats: None = ...,
        names: None = ...,
        titles: None = ...,
        byteorder: None = ...,
        aligned: Literal[False] = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[Any]]: ...
    # 定义 __array_finalize__ 方法，接受一个对象参数，无返回值
    def __array_finalize__(self, obj: object) -> None: ...
    # 重载 __getattribute__ 方法，接受一个字符串参数，返回任意类型的对象
    def __getattribute__(self, attr: str) -> Any: ...
    # 重载 __setattr__ 方法，接受一个字符串参数和一个 ArrayLike 类型的值，无返回值
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    # 重载 __getitem__ 方法，接受一个索引参数，返回任意类型的对象
    @overload
    def __getitem__(self, indx: (
        SupportsIndex
        | _ArrayLikeInt_co
        | tuple[SupportsIndex | _ArrayLikeInt_co, ...]
    )) -> Any: ...
    # 重载 __getitem__ 方法，返回一个特定类型的 recarray 类对象
    @overload
    def __getitem__(self: recarray[Any, dtype[void]], indx: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> recarray[Any, _DType_co]: ...
    # 重载
    # 定义特殊方法 __getitem__，用于支持多种类型的索引方式，返回值为 ndarray[Any, _DType_co]
    def __getitem__(self, indx: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> ndarray[Any, _DType_co]: ...

    # 定义方法重载 __getitem__，接受字符串 indx 参数，返回值为 NDArray[Any]
    @overload
    def __getitem__(self, indx: str) -> NDArray[Any]: ...

    # 定义方法重载 __getitem__，接受列表参数 indx 包含字符串类型元素，返回值为 recarray[_ShapeType, dtype[record]]
    @overload
    def __getitem__(self, indx: list[str]) -> recarray[_ShapeType, dtype[record]]: ...

    # 定义方法 field，接受整数或字符串类型的 attr 参数，可选的 val 参数默认为 None，返回值为 Any
    @overload
    def field(self, attr: int | str, val: None = ...) -> Any: ...

    # 定义方法重载 field，接受整数或字符串类型的 attr 参数和 ArrayLike 类型的 val 参数，无返回值
    @overload
    def field(self, attr: int | str, val: ArrayLike) -> None: ...
class format_parser:
    # 定义类变量 dtype，默认为 void 类型
    dtype: dtype[void]
    
    # 初始化方法
    def __init__(
        self,
        formats: DTypeLike,        # 数据类型或类似数据类型的参数
        names: None | str | Sequence[str],  # 名称，可以是字符串或字符串序列，或者为 None
        titles: None | str | Sequence[str],  # 标题，可以是字符串或字符串序列，或者为 None
        aligned: bool = ...,       # 对齐标志，默认为省略值（未指定）
        byteorder: None | _ByteOrder = ...,  # 字节顺序，可以是 None 或 _ByteOrder 类型
    ) -> None: ...  # 初始化方法无返回值，注释内容省略

__all__: list[str]  # __all__ 变量，包含导出的符号名称列表

@overload
def fromarrays(
    arrayList: Iterable[ArrayLike],  # 可迭代的数组或类似数组的列表
    dtype: DTypeLike = ...,          # 数据类型或类似数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    formats: None = ...,             # 格式，可以是 None
    names: None = ...,               # 名称，可以是 None
    titles: None = ...,              # 标题，可以是 None
    aligned: bool = ...,             # 对齐标志，默认为省略值（未指定）
    byteorder: None = ...,           # 字节顺序，默认为 None
) -> _RecArray[Any]: ...             # 返回类型为 _RecArray，包含任意类型数据

@overload
def fromarrays(
    arrayList: Iterable[ArrayLike],  # 可迭代的数组或类似数组的列表
    dtype: None = ...,               # 数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    *,
    formats: DTypeLike,              # 格式，数据类型或类似数据类型
    names: None | str | Sequence[str] = ...,  # 名称，可以是 None、字符串或字符串序列
    titles: None | str | Sequence[str] = ...,  # 标题，可以是 None、字符串或字符串序列
    aligned: bool = ...,             # 对齐标志，默认为省略值（未指定）
    byteorder: None | _ByteOrder = ...,  # 字节顺序，可以是 None 或 _ByteOrder 类型
) -> _RecArray[record]: ...         # 返回类型为 _RecArray，记录类型数据

@overload
def fromrecords(
    recList: _ArrayLikeVoid_co | tuple[Any, ...] | _NestedSequence[tuple[Any, ...]],  # 记录列表或类似记录的数据类型
    dtype: DTypeLike = ...,          # 数据类型或类似数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    formats: None = ...,             # 格式，可以是 None
    names: None = ...,               # 名称，可以是 None
    titles: None = ...,              # 标题，可以是 None
    aligned: bool = ...,             # 对齐标志，默认为省略值（未指定）
    byteorder: None = ...,           # 字节顺序，默认为 None
) -> _RecArray[record]: ...          # 返回类型为 _RecArray，记录类型数据

@overload
def fromrecords(
    recList: _ArrayLikeVoid_co | tuple[Any, ...] | _NestedSequence[tuple[Any, ...]],  # 记录列表或类似记录的数据类型
    dtype: None = ...,               # 数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    *,
    formats: DTypeLike = ...,        # 格式，数据类型或类似数据类型
    names: None | str | Sequence[str] = ...,  # 名称，可以是 None、字符串或字符串序列
    titles: None | str | Sequence[str] = ...,  # 标题，可以是 None、字符串或字符串序列
    aligned: bool = ...,             # 对齐标志，默认为省略值（未指定）
    byteorder: None | _ByteOrder = ...,  # 字节顺序，可以是 None 或 _ByteOrder 类型
) -> _RecArray[record]: ...         # 返回类型为 _RecArray，记录类型数据

@overload
def fromstring(
    datastring: _SupportsBuffer,     # 支持缓冲区的数据字符串
    dtype: DTypeLike,               # 数据类型或类似数据类型
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    offset: int = ...,              # 偏移量，默认为省略值（未指定）
    formats: None = ...,            # 格式，可以是 None
    names: None = ...,              # 名称，可以是 None
    titles: None = ...,             # 标题，可以是 None
    aligned: bool = ...,            # 对齐标志，默认为省略值（未指定）
    byteorder: None = ...,          # 字节顺序，默认为 None
) -> _RecArray[record]: ...         # 返回类型为 _RecArray，记录类型数据

@overload
def fromstring(
    datastring: _SupportsBuffer,     # 支持缓冲区的数据字符串
    dtype: None = ...,              # 数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    offset: int = ...,              # 偏移量，默认为省略值（未指定）
    *,
    formats: DTypeLike,             # 格式，数据类型或类似数据类型
    names: None | str | Sequence[str] = ...,  # 名称，可以是 None、字符串或字符串序列
    titles: None | str | Sequence[str] = ...,  # 标题，可以是 None、字符串或字符串序列
    aligned: bool = ...,            # 对齐标志，默认为省略值（未指定）
    byteorder: None | _ByteOrder = ...,  # 字节顺序，可以是 None 或 _ByteOrder 类型
) -> _RecArray[record]: ...         # 返回类型为 _RecArray，记录类型数据

@overload
def fromfile(
    fd: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsReadInto,  # 文件描述符，支持多种类型
    dtype: DTypeLike,               # 数据类型或类似数据类型
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    offset: int = ...,              # 偏移量，默认为省略值（未指定）
    formats: None = ...,            # 格式，可以是 None
    names: None = ...,              # 名称，可以是 None
    titles: None = ...,             # 标题，可以是 None
    aligned: bool = ...,            # 对齐标志，默认为省略值（未指定）
    byteorder: None = ...,          # 字节顺序，默认为 None
) -> _RecArray[Any]: ...            # 返回类型为 _RecArray，包含任意类型数据

@overload
def fromfile(
    fd: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsReadInto,  # 文件描述符，支持多种类型
    dtype: None = ...,              # 数据类型，默认为省略值（未指定）
    shape: None | _ShapeLike = ...,  # 形状，可以是 None 或形状类似数据类型
    offset: int = ...,              # 偏移量，默认为省略值（未指定）
    *,
    formats: DTypeLike,             # 格式，数据类型或类似数据类型
    names: None | str | Sequence[str] = ...,  # 名称，可以是 None、字符串或字符串序列
    titles: None | str | Sequence[str] = ...,  # 标题，可以是 None、字符串或字符串序列
# 定义一个函数签名，用于接受多个参数并返回一个 _RecArray[record] 类型的对象
@overload
def array(
    obj: _SCT | NDArray[_SCT],  # 参数 obj 可以是 _SCT 或 NDArray[_SCT] 类型
    dtype: None = ...,          # 参数 dtype 默认为 None
    shape: None | _ShapeLike = ...,  # 参数 shape 可以是 None 或 _ShapeLike 类型
    offset: int = ...,          # 参数 offset 默认为 int 类型
    formats: None = ...,        # 参数 formats 默认为 None
    names: None = ...,          # 参数 names 默认为 None
    titles: None = ...,         # 参数 titles 默认为 None
    aligned: bool = ...,        # 参数 aligned 默认为 bool 类型
    byteorder: None = ...,      # 参数 byteorder 默认为 None
    copy: bool = ...,           # 参数 copy 默认为 bool 类型
) -> _RecArray[_SCT]: ...       # 函数返回类型为 _RecArray[_SCT]

@overload
def array(
    obj: ArrayLike,             # 参数 obj 可以是 ArrayLike 类型
    dtype: DTypeLike,           # 参数 dtype 可以是 DTypeLike 类型
    shape: None | _ShapeLike = ...,  # 参数 shape 可以是 None 或 _ShapeLike 类型
    offset: int = ...,          # 参数 offset 默认为 int 类型
    formats: None = ...,        # 参数 formats 默认为 None
    names: None = ...,          # 参数 names 默认为 None
    titles: None = ...,         # 参数 titles 默认为 None
    aligned: bool = ...,        # 参数 aligned 默认为 bool 类型
    byteorder: None = ...,      # 参数 byteorder 默认为 None
    copy: bool = ...,           # 参数 copy 默认为 bool 类型
) -> _RecArray[Any]: ...        # 函数返回类型为 _RecArray[Any]

@overload
def array(
    obj: ArrayLike,             # 参数 obj 可以是 ArrayLike 类型
    dtype: None = ...,          # 参数 dtype 默认为 None
    shape: None | _ShapeLike = ...,  # 参数 shape 可以是 None 或 _ShapeLike 类型
    offset: int = ...,          # 参数 offset 默认为 int 类型
    *,
    formats: DTypeLike,         # 参数 formats 必须是 DTypeLike 类型
    names: None | str | Sequence[str] = ...,  # 参数 names 可以是 None、str 或 Sequence[str] 类型
    titles: None | str | Sequence[str] = ..., # 参数 titles 可以是 None、str 或 Sequence[str] 类型
    aligned: bool = ...,        # 参数 aligned 默认为 bool 类型
    byteorder: None | _ByteOrder = ...,  # 参数 byteorder 可以是 None 或 _ByteOrder 类型
    copy: bool = ...,           # 参数 copy 默认为 bool 类型
) -> _RecArray[record]: ...     # 函数返回类型为 _RecArray[record]

@overload
def array(
    obj: None,                  # 参数 obj 必须是 None 类型
    dtype: DTypeLike,           # 参数 dtype 可以是 DTypeLike 类型
    shape: _ShapeLike,          # 参数 shape 必须是 _ShapeLike 类型
    offset: int = ...,          # 参数 offset 默认为 int 类型
    formats: None = ...,        # 参数 formats 默认为 None
    names: None = ...,          # 参数 names 默认为 None
    titles: None = ...,         # 参数 titles 默认为 None
    aligned: bool = ...,        # 参数 aligned 默认为 bool 类型
    byteorder: None = ...,      # 参数 byteorder 默认为 None
    copy: bool = ...,           # 参数 copy 默认为 bool 类型
) -> _RecArray[Any]: ...        # 函数返回类型为 _RecArray[Any]

@overload
def array(
    obj: None,                  # 参数 obj 必须是 None 类型
    dtype: None = ...,          # 参数 dtype 默认为 None
    *,
    shape: _ShapeLike,          # 参数 shape 必须是 _ShapeLike 类型
    formats: DTypeLike,         # 参数 formats 必须是 DTypeLike 类型
    names: None | str | Sequence[str] = ...,  # 参数 names 可以是 None、str 或 Sequence[str] 类型
    titles: None | str | Sequence[str] = ..., # 参数 titles 可以是 None、str 或 Sequence[str] 类型
    aligned: bool = ...,        # 参数 aligned 默认为 bool 类型
    byteorder: None | _ByteOrder = ...,  # 参数 byteorder
```