# `D:\src\scipysrc\pandas\pandas\_typing.py`

```
# 从 __future__ 模块导入注解以支持类型注解的标准化
from __future__ import annotations

# 导入集合类型的抽象基类
from collections.abc import (
    Callable,        # 可调用对象的抽象基类
    Hashable,        # 可哈希对象的抽象基类
    Iterator,        # 迭代器对象的抽象基类
    Mapping,         # 映射类型的抽象基类
    MutableMapping,  # 可变映射类型的抽象基类
    Sequence,        # 序列类型的抽象基类
)

# 从 datetime 模块导入多个时间相关的类和函数
from datetime import (
    date,      # 日期对象类
    datetime,  # 日期时间对象类
    timedelta, # 时间间隔对象类
    tzinfo,    # 时区信息对象类
)

# 从 os 模块导入 PathLike 类型
from os import PathLike

# 导入 sys 模块
import sys

# 从 typing 模块导入多个类型和标识符
from typing import (
    TYPE_CHECKING,  # 类型检查时的标识符
    Any,            # 任意类型
    Literal,        # 字面量类型
    Optional,       # 可选类型
    Protocol,       # 协议类型
    Type as type_t, # 类型别名
    TypeVar,        # 类型变量
    Union,          # 联合类型
    overload,       # 函数重载的装饰器
)

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 如果是类型检查状态下，导入以下内容
if TYPE_CHECKING:
    # 导入 numpy.typing 模块的类型
    import numpy.typing as npt

    # 导入 pandas 库内部的一些类型
    from pandas._libs import (
        NaTType,     # 表示缺失时间的类型
        Period,      # 时间段对象类
        Timedelta,   # 时间差对象类
        Timestamp,   # 时间戳对象类
    )
    from pandas._libs.tslibs import BaseOffset  # 时间基准对象类

    # 导入 pandas 库的多个核心类型和类
    from pandas import (
        DatetimeIndex,     # 时间索引对象类
        Interval,          # 区间对象类
        PeriodIndex,       # 时间段索引对象类
        TimedeltaIndex,    # 时间差索引对象类
    )
    from pandas.arrays import (
        DatetimeArray,     # 时间数组类
        TimedeltaArray,    # 时间差数组类
    )
    from pandas.core.arrays.base import ExtensionArray  # 扩展数组基类
    from pandas.core.frame import DataFrame            # 数据帧类
    from pandas.core.generic import NDFrame            # 通用数据结构类
    from pandas.core.groupby.generic import (
        DataFrameGroupBy,  # 数据帧分组类
        GroupBy,           # 分组对象类
        SeriesGroupBy,     # 序列分组类
    )
    from pandas.core.indexes.base import Index             # 索引基类
    from pandas.core.internals import (
        BlockManager,         # 块管理器类
        SingleBlockManager,   # 单块管理器类
    )
    from pandas.core.resample import Resampler        # 重采样器类
    from pandas.core.series import Series            # 序列对象类
    from pandas.core.window.rolling import BaseWindow  # 滚动窗口基类

    from pandas.io.formats.format import EngFormatter           # 工程数值格式化类
    from pandas.tseries.holiday import AbstractHolidayCalendar  # 抽象假期日历类

    # 定义支持标量类型的联合类型
    ScalarLike_co = Union[
        int,          # 整数
        float,        # 浮点数
        complex,      # 复数
        str,          # 字符串
        bytes,        # 字节串
        np.generic,   # NumPy 通用类型
    ]

    # 定义 numpy 兼容的数值数组类型
    NumpyValueArrayLike = Union[ScalarLike_co, npt.ArrayLike]

    # 定义 numpy 兼容的排序数组类型
    NumpySorter = Optional[npt._ArrayLikeInt_co]  # type: ignore[name-defined]

    # 导入类型参数和支持索引的类型
    from typing import (
        ParamSpec,         # 参数规范
        SupportsIndex,     # 支持索引的类型
    )

    # 导入类型别名，对于 pyright 检查忽略未使用导入警告
    from typing import Concatenate  # pyright: ignore[reportUnusedImport]
    from typing import TypeGuard   # pyright: ignore[reportUnusedImport]

    # 如果 Python 版本 >= 3.11，导入 Self 和 Unpack 类型
    if sys.version_info >= (3, 11):
        from typing import Self   # pyright: ignore[reportUnusedImport]
        from typing import Unpack  # pyright: ignore[reportUnusedImport]
    else:
        # 否则，从 typing_extensions 导入 Self 和 Unpack 类型
        from typing_extensions import Self   # pyright: ignore[reportUnusedImport]
        from typing_extensions import Unpack  # pyright: ignore[reportUnusedImport]

# 否则，如果不是类型检查状态，则将这些标识符设置为 None
else:
    npt: Any = None
    ParamSpec: Any = None
    Self: Any = None
    TypeGuard: Any = None
    Concatenate: Any = None
    Unpack: Any = None

# 定义一个类型变量 HashableT，限制为可哈希类型
HashableT = TypeVar("HashableT", bound=Hashable)
# 定义一个泛型类型变量 HashableT2，用于表示可哈希类型
HashableT2 = TypeVar("HashableT2", bound=Hashable)
# 定义一个泛型类型变量 MutableMappingT，用于表示可变映射类型
MutableMappingT = TypeVar("MutableMappingT", bound=MutableMapping)

# array-like 相关定义

# ArrayLike 表示一个数组类型，可以是 ExtensionArray 或者 np.ndarray
ArrayLike = Union["ExtensionArray", np.ndarray]
# ArrayLikeT 是一个泛型类型变量，表示 ArrayLike 类型的子类型
ArrayLikeT = TypeVar("ArrayLikeT", "ExtensionArray", np.ndarray)
# AnyArrayLike 表示任何类似数组的类型，可以是 ArrayLike、Index 或 Series
AnyArrayLike = Union[ArrayLike, "Index", "Series"]
# TimeArrayLike 表示时间数组类型，可以是 DatetimeArray 或 TimedeltaArray
TimeArrayLike = Union["DatetimeArray", "TimedeltaArray"]

# list-like 相关定义

# SequenceNotStr 是一个协议，表示非字符串序列对象的接口
# 定义了 __getitem__、__contains__、__len__、__iter__ 等方法
class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]: ...

    def __contains__(self, value: object, /) -> bool: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[_T_co]: ...

    def index(self, value: Any, start: int = ..., stop: int = ..., /) -> int: ...

    def count(self, value: Any, /) -> int: ...

    def __reversed__(self) -> Iterator[_T_co]: ...

# ListLike 表示列表类型，可以是 AnyArrayLike、SequenceNotStr 或 range
ListLike = Union[AnyArrayLike, SequenceNotStr, range]

# scalars 相关定义

# PythonScalar 表示 Python 中的标量类型，可以是 str、float 或 bool
PythonScalar = Union[str, float, bool]
# DatetimeLikeScalar 表示日期时间相关的标量类型，可以是 Period、Timestamp 或 Timedelta
DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
# PandasScalar 表示 Pandas 支持的标量类型，可以是 Period、Timestamp、Timedelta 或 Interval
PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
# Scalar 表示任意标量类型，包括 PythonScalar、PandasScalar、numpy 的日期时间类型以及 date
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, date]
# IntStrT 是一个泛型类型变量，表示可以是 int 或 str 类型
IntStrT = TypeVar("IntStrT", bound=Union[int, str])

# timestamp 和 timedelta 可转换类型

# TimestampConvertibleTypes 表示可以转换为 Timestamp 的类型，包括 Timestamp、date、np.datetime64、np.int64、float 和 str
TimestampConvertibleTypes = Union[
    "Timestamp", date, np.datetime64, np.int64, float, str
]
# TimestampNonexistent 表示不存在的时间戳类型，可以是特定的文字字面量或 timedelta 类型
TimestampNonexistent = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"], timedelta
]
# TimedeltaConvertibleTypes 表示可以转换为 Timedelta 的类型，包括 Timedelta、timedelta、np.timedelta64、np.int64、float 和 str
TimedeltaConvertibleTypes = Union[
    "Timedelta", timedelta, np.timedelta64, np.int64, float, str
]
# Timezone 表示时区信息，可以是字符串形式的时区名或 tzinfo 对象
Timezone = Union[str, tzinfo]

# ToTimestampHow 表示转换为 Timestamp 的方式，可以是 's'、'e'、'start' 或 'end'
ToTimestampHow = Literal["s", "e", "start", "end"]

# NDFrameT 是一个泛型类型变量，表示继承自 NDFrame 的子类，用于强制指定函数参数和返回类型
# IndexT、FreqIndexT 和 NumpyIndexT 是各自的泛型类型变量，用于指定索引相关的类型限制
NDFrameT = TypeVar("NDFrameT", bound="NDFrame")
IndexT = TypeVar("IndexT", bound="Index")
FreqIndexT = TypeVar("FreqIndexT", "DatetimeIndex", "PeriodIndex", "TimedeltaIndex")
NumpyIndexT = TypeVar("NumpyIndexT", np.ndarray, "Index")

# AxisInt 表示轴的整数编号
AxisInt = int
# Axis 表示轴的类型，可以是整数轴编号或字符串 'index'、'columns'、'rows'
Axis = Union[AxisInt, Literal["index", "columns", "rows"]]
# IndexLabel 表示索引标签，可以是可哈希对象或可哈希对象组成的序列
IndexLabel = Union[Hashable, Sequence[Hashable]]
# Level 表示层级，必须是可哈希的对象
Level = Hashable
# Shape 表示形状，是一个由整数组成的元组
Shape = tuple[int, ...]
# Suffixes 表示后缀字符串的序列，每个元素可以为 None 或字符串
Suffixes = Sequence[Optional[str]]
# Ordered 表示有序性，可以是布尔值或 None
Ordered = Optional[bool]
# JSONSerializable 表示可序列化为 JSON 的对象，可以是 PythonScalar、list 或 dict
JSONSerializable = Optional[Union[PythonScalar, list, dict]]
# Frequency 表示频率信息，可以是字符串形式的频率或 BaseOffset 对象
Frequency = Union[str, "BaseOffset"]
# Axes 表示轴列表，可以是 ListLike 类型
Axes = ListLike

# RandomState 表示随机状态对象，可以是整数、numpy 数组或各种随机状态生成器对象
RandomState = Union[
    int,
    np.ndarray,
    np.random.Generator,
    np.random.BitGenerator,
    np.random.RandomState,
]

# dtypes 相关定义

# NpDtype 表示 numpy 的数据类型，可以是字符串、np.dtype 对象或支持的复杂类型的 type_t 类型
NpDtype = Union[str, np.dtype, type_t[Union[str, complex, bool, object]]]
# Dtype 表示数据类型，可以是 ExtensionDtype 或 NpDtype
Dtype = Union["ExtensionDtype", NpDtype]
# 定义一个类型别名，表示 AstypeArg 可以是 "ExtensionDtype" 或 "npt.DTypeLike" 中的一种
AstypeArg = Union["ExtensionDtype", "npt.DTypeLike"]

# 定义一个类型别名，表示 DtypeArg 可以是 np.dtype 或 Mapping[Hashable, Dtype] 中的一种
DtypeArg = Union[Dtype, Mapping[Hashable, Dtype]]

# 定义一个类型别名，表示 DtypeObj 可以是 np.dtype 或 "ExtensionDtype" 中的一种
DtypeObj = Union[np.dtype, "ExtensionDtype"]

# 定义一个类型别名，表示 ConvertersArg 是一个字典，其键是可哈希的，值是接受 Dtype 并返回 Dtype 的可调用对象
ConvertersArg = dict[Hashable, Callable[[Dtype], Dtype]]

# 定义一个类型别名，表示 ParseDatesArg 可以是 bool、Hashable 的列表、Hashable 列表的列表或者字典
ParseDatesArg = Union[
    bool, list[Hashable], list[list[Hashable]], dict[Hashable, list[Hashable]]
]

# 定义一个类型别名，表示 Renamer 可以是映射 (Mapping[Any, Hashable]) 或接受任意类型并返回 Hashable 的可调用对象
Renamer = Union[Mapping[Any, Hashable], Callable[[Any], Hashable]]

# 定义一个类型变量，用于在泛型函数和参数化中保持类型信息
T = TypeVar("T")

# 定义一个类型别名，表示 FuncType 是一个接受任意参数并返回任意类型的可调用对象
FuncType = Callable[..., Any]

# 定义一个类型变量 F，它是 FuncType 的子类型
F = TypeVar("F", bound=FuncType)

# 定义一个类型变量 TypeT，表示任意的 Python 类型
TypeT = TypeVar("TypeT", bound=type)

# 定义一个类型别名，表示 ValueKeyFunc 是一个可选的函数，接受 Series 并返回 Series 或 AnyArrayLike
ValueKeyFunc = Optional[Callable[["Series"], Union["Series", AnyArrayLike]]]

# 定义一个类型别名，表示 IndexKeyFunc 是一个可选的函数，接受 Index 并返回 Index 或 AnyArrayLike
IndexKeyFunc = Optional[Callable[["Index"], Union["Index", AnyArrayLike]]]

# 定义一个类型别名，表示 AggFuncTypeBase 可以是 Callable 或字符串
AggFuncTypeBase = Union[Callable, str]

# 定义一个类型别名，表示 AggFuncTypeDict 是一个可变映射，其键是可哈希的，值是 AggFuncTypeBase 或 AggFuncTypeBase 的列表
AggFuncTypeDict = MutableMapping[
    Hashable, Union[AggFuncTypeBase, list[AggFuncTypeBase]]
]

# 定义一个类型别名，表示 AggFuncType 可以是 AggFuncTypeBase、AggFuncTypeBase 的列表或 AggFuncTypeDict
AggFuncType = Union[
    AggFuncTypeBase,
    list[AggFuncTypeBase],
    AggFuncTypeDict,
]

# 定义一个类型别名，表示 AggObjType 可以是 Series、DataFrame、GroupBy 等对象的联合类型
AggObjType = Union[
    "Series",
    "DataFrame",
    "GroupBy",
    "SeriesGroupBy",
    "DataFrameGroupBy",
    "BaseWindow",
    "Resampler",
]

# 定义一个类型别名，表示 PythonFuncType 是一个接受任意参数并返回任意类型的 Python 函数
PythonFuncType = Callable[[Any], Any]

# 定义一个类型变量，表示 AnyStr_co 可以是 str 或 bytes，是协变的
AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)

# 定义一个类型变量，表示 AnyStr_contra 可以是 str 或 bytes，是逆变的
AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)

# 定义一个协议，表示 BaseBuffer 类必须实现 mode 属性和 seek、seekable、tell 方法
class BaseBuffer(Protocol):
    @property
    def mode(self) -> str:
        ...

    def seek(self, __offset: int, __whence: int = ...) -> int:
        ...

    def seekable(self) -> bool:
        ...

    def tell(self) -> int:
        ...

# 定义一个协议 ReadBuffer，继承自 BaseBuffer 和 Protocol[AnyStr_co]，并且必须实现 read 方法
class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):
    def read(self, __n: int = ...) -> AnyStr_co:
        ...

# 定义一个协议 WriteBuffer，继承自 BaseBuffer 和 Protocol[AnyStr_contra]，并且必须实现 write 和 flush 方法
class WriteBuffer(BaseBuffer, Protocol[AnyStr_contra]):
    def write(self, __b: AnyStr_contra) -> Any:
        ...

    def flush(self) -> Any:
        ...

# 定义一个协议 ReadPickleBuffer，继承自 ReadBuffer[bytes]，并且必须实现 readline 方法
class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> bytes:
        ...

# 定义一个协议 WriteExcelBuffer，继承自 WriteBuffer[bytes]，并且必须实现 truncate 方法
class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ..., /) -> int:
        ...

# 定义一个协议 ReadCsvBuffer，继承自 ReadBuffer[AnyStr_co]，暂未完整定义，下文未提供完整信息
class ReadCsvBuffer(ReadBuffer[AnyStr_co], Protocol):
    # 可继续根据实际情况完善注释
    `
        # 定义一个迭代器方法，返回一个支持任意字符串类型的迭代器对象
        def __iter__(self) -> Iterator[AnyStr_co]:
            # 标注此方法适用于 engine=python 的情况
            ...
    
        # 定义一个文件描述符方法，返回一个整数表示文件描述符
        def fileno(self) -> int:
            # 标注此方法适用于 _MMapWrapper
            ...
    
        # 定义一个读取行的方法，返回一个支持任意字符串类型的对象
        def readline(self) -> AnyStr_co:
            # 标注此方法适用于 engine=python 的情况
            ...
    
        # 定义一个属性方法，返回一个布尔值表示对象是否已关闭
        @property
        def closed(self) -> bool:
            # 标注此属性适用于 engine=pyarrow
            ...
# 定义文件路径类型，可以是字符串或类似路径的字符串
FilePath = Union[str, "PathLike[str]"]

# 用于传递读写文件时的任意关键字参数选项
StorageOptions = Optional[dict[str, Any]]

# 压缩关键字和压缩选项的字典类型定义
CompressionDict = dict[str, Any]
# 可选的压缩选项，可以是预定义的字符串或自定义的压缩字典
CompressionOptions = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"], CompressionDict]
]

# DataFrameFormatter 中的类型定义
FormattersType = Union[
    list[Callable], tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
# 列空间类型，映射可哈希的键到字符串或整数值的字典类型
ColspaceType = Mapping[Hashable, Union[str, int]]
# 浮点数格式化类型，可以是字符串、可调用对象或者 EngFormatter 类型
FloatFormatType = Union[str, Callable, "EngFormatter"]
# 列空间参数类型，可以是字符串、整数、整数序列或映射类型
ColspaceArgType = Union[
    str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]
]

# fillna() 函数的选项类型，指定填充缺失值的方法
FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]
# 插值选项类型，指定插值方法的字符串常量
InterpolateOptions = Literal[
    "linear",
    "time",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

# 内部管理器类型，可以是 BlockManager 或 SingleBlockManager
Manager = Union["BlockManager", "SingleBlockManager"]

# 索引类型定义
# PositionalIndexer 用于有效的一维位置索引，例如可以传递给 ndarray.__getitem__
# ScalarIndexer 表示单个值作为索引
# SequenceIndexer 表示类似列表或切片的索引（但不包括元组）
# PositionalIndexerTuple 扩展了 PositionalIndexer 用于二维数组的索引
# 这些类型用于各种 __getitem__ 的重载
# 如果是类型检查模式，TakeIndexer 可以是整数或整数数组类型
# 否则，TakeIndexer 可以是任意类型
ScalarIndexer = Union[int, np.integer]
SequenceIndexer = Union[slice, list[int], np.ndarray]
PositionalIndexer = Union[ScalarIndexer, SequenceIndexer]
PositionalIndexerTuple = tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D = Union[PositionalIndexer, PositionalIndexerTuple]
if TYPE_CHECKING:
    TakeIndexer = Union[Sequence[int], Sequence[np.integer], npt.NDArray[np.integer]]
else:
    TakeIndexer = Any

# 由 drop 和 astype 等函数共享的选项，控制是否忽略或引发异常
IgnoreRaise = Literal["ignore", "raise"]

# 窗口排序方法类型
WindowingRankType = Literal["average", "min", "max"]

# read_csv 函数使用的引擎类型
CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]

# read_json 函数使用的引擎类型
JSONEngine = Literal["ujson", "pyarrow"]

# read_xml 函数使用的解析器类型
XMLParsers = Literal["lxml", "etree"]

# read_html 函数使用的解析器类型
HTMLFlavors = Literal["lxml", "html5lib", "bs4"]

# 区间闭合类型
IntervalLeftRight = Literal["left", "right"]
IntervalClosedType = Union[IntervalLeftRight, Literal["both", "neither"]]

# datetime 和 NaTType 类型定义
DatetimeNaTType = Union[datetime, "NaTType"]
DateTimeErrorChoices = Literal["raise", "coerce"]

# sort_index 函数使用的排序方法类型
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition = Literal["first", "last"]
# 指定类型，表示在 NsmallestNlargest 函数中指定的 "first", "last", "all" 三种选项之一
NsmallestNlargestKeep = Literal["first", "last", "all"]

# 指定类型，表示在计算分位数时的插值方法，包括 "linear", "lower", "higher", "midpoint", "nearest" 这几种选项
QuantileInterpolation = Literal["linear", "lower", "higher", "midpoint", "nearest"]

# 指定类型，表示在绘图时的方向选项，可以是 "horizontal" 或 "vertical"
PlottingOrientation = Literal["horizontal", "vertical"]

# 指定类型，表示在处理缺失值时的选项，可以是 "any" 或 "all"
AnyAll = Literal["any", "all"]

# 指定类型，表示在合并数据时的方式，可以是 "left", "right", "inner", "outer", "cross" 之一
MergeHow = Literal["left", "right", "inner", "outer", "cross"]

# 指定类型，表示在合并数据时的验证选项，例如 "one_to_one", "one_to_many", "many_to_one", "many_to_many" 等
MergeValidate = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# 指定类型，表示在连接数据时的方式，可以是 "left", "right", "inner", "outer" 之一
JoinHow = Literal["left", "right", "inner", "outer"]

# 指定类型，表示在连接数据时的验证选项，例如 "one_to_one", "one_to_many", "many_to_one", "many_to_many" 等
JoinValidate = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# 指定类型，表示在重新索引数据时的方法，可以是 FillnaOptions 中的选项或 "nearest"
ReindexMethod = Union[FillnaOptions, Literal["nearest"]]

# 指定类型，表示 Matplotlib 的颜色选项，可以是字符串或浮点数序列
MatplotlibColor = Union[str, Sequence[float]]

# 指定类型，表示时间分组的原点选项，可以是 "Timestamp", "epoch", "start", "start_day", "end", "end_day" 之一
TimeGrouperOrigin = Union[
    "Timestamp", Literal["epoch", "start", "start_day", "end", "end_day"]
]

# 指定类型，表示时间模糊值的选项，可以是 "infer", "NaT", "raise" 或 npt.NDArray[np.bool_] 类型
TimeAmbiguous = Union[Literal["infer", "NaT", "raise"], "npt.NDArray[np.bool_]"]

# 指定类型，表示时间不存在值的选项，可以是 "shift_forward", "shift_backward", "NaT", "raise" 或 timedelta 类型
TimeNonexistent = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"], timedelta
]

# 指定类型，表示在数据对齐时的保留选项，可以是 "first", "last", False 之一
DropKeep = Literal["first", "last", False]

# 指定类型，表示相关性计算的方法，可以是 "pearson", "kendall", "spearman" 或自定义的函数
CorrelationMethod = Union[
    Literal["pearson", "kendall", "spearman"], Callable[[np.ndarray, np.ndarray], float]
]

# 指定类型，表示数据对齐时的连接选项，可以是 "outer", "inner", "left", "right" 之一
AlignJoin = Literal["outer", "inner", "left", "right"]

# 指定类型，表示数据类型后端选项，可以是 "pyarrow" 或 "numpy_nullable"
DtypeBackend = Literal["pyarrow", "numpy_nullable"]

# 指定类型，表示时间单位选项，可以是 "s", "ms", "us", "ns" 之一
TimeUnit = Literal["s", "ms", "us", "ns"]

# 指定类型，表示打开文件时的错误处理选项，包括 "strict", "ignore", "replace", "surrogateescape" 等
OpenFileErrors = Literal[
    "strict",
    "ignore",
    "replace",
    "surrogateescape",
    "xmlcharrefreplace",
    "backslashreplace",
    "namereplace",
]

# 指定类型，表示在更新操作中的连接选项，可以是 "left"
UpdateJoin = Literal["left"]

# 指定类型，表示在 applymap 操作中的 NA 处理选项，可以是 "ignore"
NaAction = Literal["ignore"]

# 指定类型，表示从字典创建对象时的方向选项，可以是 "columns", "index", "tight" 之一
FromDictOrient = Literal["columns", "index", "tight"]

# 指定类型，表示在保存为 Stata 文件时的字节顺序选项，可以是 ">", "<", "little", "big" 之一
ToStataByteorder = Literal[">", "<", "little", "big"]

# 指定类型，表示 ExcelWriter 的选项，用于处理已存在表格时的行为，可以是 "error", "new", "replace", "overlay" 之一
ExcelWriterIfSheetExists = Literal["error", "new", "replace", "overlay"]

# 指定类型，表示 ExcelWriter 的选项，用于处理合并单元格的行为，可以是布尔值或 "columns" 字符串
ExcelWriterMergeCells = Union[bool, Literal["columns"]]

# 指定类型，表示偏移量日历选项，可以是 np.busdaycalendar 类型或 "AbstractHolidayCalendar"
OffsetCalendar = Union[np.busdaycalendar, "AbstractHolidayCalendar"]

# 指定类型，表示 read_csv 函数中的 usecols 参数类型，可以是序列，range，AnyArrayLike，Callable 或 None
UsecolsArgType = Union[
    SequenceNotStr[Hashable],
    range,
    AnyArrayLike,
    Callable[[HashableT], bool],
    None,
]

# 指定类型变量，表示任何可哈希序列的子类型
SequenceT = TypeVar("SequenceT", bound=Sequence[Hashable])
```