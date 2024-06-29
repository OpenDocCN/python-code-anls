# `D:\src\scipysrc\numpy\numpy\lib\_npyio_impl.pyi`

```
import os
import sys
import zipfile
import types
from re import Pattern
from collections.abc import Collection, Mapping, Iterator, Sequence, Callable, Iterable
from typing import (
    Literal as L,
    Any,
    TypeVar,
    Generic,
    IO,
    overload,
    Protocol,
)

from numpy import (
    ndarray,
    recarray,
    dtype,
    generic,
    float64,
    void,
    record,
)

from numpy.ma.mrecords import MaskedRecords
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _DTypeLike,
    _SupportsArrayFunc,
)

from numpy._core.multiarray import (
    packbits as packbits,
    unpackbits as unpackbits,
)

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_CharType_co = TypeVar("_CharType_co", str, bytes, covariant=True)
_CharType_contra = TypeVar("_CharType_contra", str, bytes, contravariant=True)

class _SupportsGetItem(Protocol[_T_contra, _T_co]):
    def __getitem__(self, key: _T_contra, /) -> _T_co: ...

class _SupportsRead(Protocol[_CharType_co]):
    def read(self) -> _CharType_co: ...

class _SupportsReadSeek(Protocol[_CharType_co]):
    def read(self, n: int, /) -> _CharType_co: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

class _SupportsWrite(Protocol[_CharType_contra]):
    def write(self, s: _CharType_contra, /) -> object: ...

# 所有公开的接口名称列表
__all__: list[str]

class BagObj(Generic[_T_co]):
    def __init__(self, obj: _SupportsGetItem[str, _T_co]) -> None: ...
    def __getattribute__(self, key: str) -> _T_co: ...
    def __dir__(self) -> list[str]: ...

class NpzFile(Mapping[str, NDArray[Any]]):
    # ZipFile 对象，用于处理 NPZ 文件的 ZIP 压缩包
    zip: zipfile.ZipFile
    # 文件对象，用于指向 NPZ 文件或类似对象的文件句柄
    fid: None | IO[str]
    # NPZ 文件中的文件名列表
    files: list[str]
    # 是否允许使用 pickle 序列化
    allow_pickle: bool
    # pickle 序列化的参数
    pickle_kwargs: None | Mapping[str, Any]
    # 在字符串表示中最大的数组元素数量
    _MAX_REPR_ARRAY_COUNT: int

    # 将 `f` 表示为一个可变属性，以便访问 `self` 的类型
    @property
    def f(self: _T) -> BagObj[_T]: ...

    @f.setter
    def f(self: _T, value: BagObj[_T]) -> None: ...

    # 初始化 NPZ 文件对象
    def __init__(
        self,
        fid: IO[str],
        own_fid: bool = ...,
        allow_pickle: bool = ...,
        pickle_kwargs: None | Mapping[str, Any] = ...,
    ) -> None: ...

    # 进入上下文管理器时调用的方法
    def __enter__(self: _T) -> _T: ...

    # 退出上下文管理器时调用的方法
    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        traceback: None | types.TracebackType,
        /,
    ) -> None: ...

    # 关闭 NPZ 文件对象
    def close(self) -> None: ...

    # 删除 NPZ 文件对象时调用的方法
    def __del__(self) -> None: ...

    # 返回迭代器，遍历 NPZ 文件中的文件名
    def __iter__(self) -> Iterator[str]: ...

    # 返回 NPZ 文件中文件名的数量
    def __len__(self) -> int: ...

    # 获取指定文件名的数组数据
    def __getitem__(self, key: str) -> NDArray[Any]: ...

    # 检查指定文件名是否存在于 NPZ 文件中
    def __contains__(self, key: str) -> bool: ...

    # 返回 NPZ 文件的字符串表示形式
    def __repr__(self) -> str: ...

class DataSource:
    # 初始化数据源对象，指定目标路径
    def __init__(
        self,
        destpath: None | str | os.PathLike[str] = ...,
    ) -> None: ...

    # 删除数据源对象时调用的方法
    def __del__(self) -> None: ...

    # 返回指定路径的绝对路径
    def abspath(self, path: str) -> str: ...

    # 检查指定路径的文件或目录是否存在
    def exists(self, path: str) -> bool: ...
    # 指示函数 `open` 的文档字符串注释说明。该函数用于打开文件对象，
    # 其中 `path` 参数的文件扩展名决定了默认打开文件的模式，是字符串模式还是字节模式。
    # 函数定义了多个参数，包括文件路径 `path`、打开模式 `mode`、编码方式 `encoding` 和换行符 `newline`。
    # 返回类型为 `IO[Any]`，即任意类型的输入输出对象。
    def open(
        self,
        path: str,
        mode: str = ...,
        encoding: None | str = ...,
        newline: None | str = ...,
    ) -> IO[Any]: ...
# 返回一个 `NpzFile` 对象，如果文件是一个 zip 文件；
# 否则返回一个 `ndarray` 或 `memmap` 对象
def load(
    file: str | bytes | os.PathLike[Any] | _SupportsReadSeek[bytes],
    mmap_mode: L[None, "r+", "r", "w+", "c"] = ...,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
    encoding: L["ASCII", "latin1", "bytes"] = ...,
) -> Any: ...

# 将数组 `arr` 保存到文件 `file` 中
def save(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    arr: ArrayLike,
    allow_pickle: bool = ...,
    fix_imports: bool = ...,
) -> None: ...

# 将多个数组 `args` 保存到一个压缩的 `.npz` 文件 `file` 中
def savez(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...

# 将多个数组 `args` 保存到一个压缩的 `.npz` 文件 `file` 中，使用压缩
def savez_compressed(
    file: str | os.PathLike[str] | _SupportsWrite[bytes],
    *args: ArrayLike,
    **kwds: ArrayLike,
) -> None: ...

# 从文本文件 `fname` 中加载数据并返回一个 `NDArray[float64]`
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: None = ...,
    comments: None | str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    quotechar: None | str = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[float64]: ...

# 从文本文件 `fname` 中加载数据并返回一个 `NDArray[_SCT]`
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: _DTypeLike[_SCT],
    comments: None | str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    quotechar: None | str = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[_SCT]: ...

# 从文本文件 `fname` 中加载数据并返回一个 `NDArray[Any]`
@overload
def loadtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: DTypeLike,
    comments: None | str | Sequence[str] = ...,
    delimiter: None | str = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    skiprows: int = ...,
    usecols: int | Sequence[int] = ...,
    unpack: bool = ...,
    ndmin: L[0, 1, 2] = ...,
    encoding: None | str = ...,
    max_rows: None | int = ...,
    *,
    quotechar: None | str = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[Any]: ...

# 将数组 `X` 保存到文本文件 `fname` 中
def savetxt(
    fname: str | os.PathLike[str] | _SupportsWrite[str] | _SupportsWrite[bytes],
    X: ArrayLike,
    fmt: str | Sequence[str] = ...,
    delimiter: str = ...,
    newline: str = ...,
    header: str = ...,
    footer: str = ...,
    comments: str = ...,
    encoding: None | str = ...,
) -> None: ...

# 从文本文件 `file` 中使用正则表达式 `regexp` 读取数据
@overload
def fromregex(
    file: str | os.PathLike[str] | _SupportsRead[str] | _SupportsRead[bytes],
    regexp: str | bytes | Pattern[Any],
    dtype: _DTypeLike[_SCT],
    encoding: None | str = ...


# 定义变量 `dtype`，其类型是 `_DTypeLike[_SCT]`
# 定义变量 `encoding`，其类型可以是 `None` 或者 `str`，默认为 `...`（未指定具体值）
# 根据文件、路径或可读对象中的正则表达式模式从文件中读取数据，并返回一个NDArray对象
@overload
def fromregex(
    file: str | os.PathLike[str] | _SupportsRead[str] | _SupportsRead[bytes],
    regexp: str | bytes | Pattern[Any],
    dtype: DTypeLike,
    encoding: None | str = ...
) -> NDArray[Any]: ...


# 从文件中读取数据生成一个NDArray对象
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: None = ...,
    comments: str = ...,
    delimiter: None | str | int | Iterable[int] = ...,
    skip_header: int = ...,
    skip_footer: int = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    missing_values: Any = ...,
    filling_values: Any = ...,
    usecols: None | Sequence[int] = ...,
    names: L[None, True] | str | Collection[str] = ...,
    excludelist: None | Sequence[str] = ...,
    deletechars: str = ...,
    replace_space: str = ...,
    autostrip: bool = ...,
    case_sensitive: bool | L['upper', 'lower'] = ...,
    defaultfmt: str = ...,
    unpack: None | bool = ...,
    usemask: bool = ...,
    loose: bool = ...,
    invalid_raise: bool = ...,
    max_rows: None | int = ...,
    encoding: str = ...,
    *,
    ndmin: L[0, 1, 2] = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[Any]: ...


# 从文件中读取数据生成一个NDArray对象，带有指定的dtype
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: _DTypeLike[_SCT],
    comments: str = ...,
    delimiter: None | str | int | Iterable[int] = ...,
    skip_header: int = ...,
    skip_footer: int = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    missing_values: Any = ...,
    filling_values: Any = ...,
    usecols: None | Sequence[int] = ...,
    names: L[None, True] | str | Collection[str] = ...,
    excludelist: None | Sequence[str] = ...,
    deletechars: str = ...,
    replace_space: str = ...,
    autostrip: bool = ...,
    case_sensitive: bool | L['upper', 'lower'] = ...,
    defaultfmt: str = ...,
    unpack: None | bool = ...,
    usemask: bool = ...,
    loose: bool = ...,
    invalid_raise: bool = ...,
    max_rows: None | int = ...,
    encoding: str = ...,
    *,
    ndmin: L[0, 1, 2] = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[_SCT]: ...


# 从文件中读取数据生成一个NDArray对象，带有指定的dtype
@overload
def genfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    dtype: DTypeLike,
    comments: str = ...,
    delimiter: None | str | int | Iterable[int] = ...,
    skip_header: int = ...,
    skip_footer: int = ...,
    converters: None | Mapping[int | str, Callable[[str], Any]] = ...,
    missing_values: Any = ...,
    filling_values: Any = ...,
    usecols: None | Sequence[int] = ...,
    names: L[None, True] | str | Collection[str] = ...,
    excludelist: None | Sequence[str] = ...,
    deletechars: str = ...,
    replace_space: str = ...,
    autostrip: bool = ...,
    case_sensitive: bool | L['upper', 'lower'] = ...,
    defaultfmt: str = ...,
    unpack: None | bool = ...,
    usemask: bool = ...,
    loose: bool = ...,
    invalid_raise: bool = ...,
    max_rows: None | int = ...,
    encoding: str = ...,
    *,
    ndmin: L[0, 1, 2] = ...,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[_SCT]: ...
    # 定义变量 max_rows，可以是 None 或者整数类型，默认为 None
    max_rows: None | int = ...,
    # 定义变量 encoding，表示字符串编码方式，默认为 ...
    encoding: str = ...,
    # 使用星号表示后面的参数必须使用关键字参数传入
    *,
    # 定义变量 ndmin，可以是列表 [0, 1, 2] 中的元素，默认为 ...
    ndmin: L[0, 1, 2] = ...,
    # 定义变量 like，可以是 None 或者支持数组函数的类型，默认为 ...
    like: None | _SupportsArrayFunc = ...,
# 定义了一个类型提示，表明函数的返回类型是一个 NumPy 的多维数组（NDArray），包含任意类型的元素
def recfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> NDArray[Any]: ...

# 函数重载：当 usemask 参数为 False 时，返回一个 recarray 类型的 NumPy 数组，元素类型为 dtype[record]
@overload
def recfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[record]]: ...

# 函数重载：当 usemask 参数为 True 时，返回一个 MaskedRecords 类型的 NumPy 数组，元素类型为 dtype[void]
@overload
def recfromtxt(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...

# 定义了一个类型提示，表明函数的返回类型是一个 NumPy 的多维数组（NDArray），包含任意类型的元素
def recfromcsv(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> NDArray[Any]: ...

# 函数重载：当 usemask 参数为 False 时，返回一个 recarray 类型的 NumPy 数组，元素类型为 dtype[record]
@overload
def recfromcsv(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[False] = ...,
    **kwargs: Any,
) -> recarray[Any, dtype[record]]: ...

# 函数重载：当 usemask 参数为 True 时，返回一个 MaskedRecords 类型的 NumPy 数组，元素类型为 dtype[void]
@overload
def recfromcsv(
    fname: str | os.PathLike[str] | Iterable[str] | Iterable[bytes],
    *,
    usemask: L[True],
    **kwargs: Any,
) -> MaskedRecords[Any, dtype[void]]: ...
```