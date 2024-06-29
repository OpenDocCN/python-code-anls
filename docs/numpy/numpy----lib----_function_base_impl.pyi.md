# `D:\src\scipysrc\numpy\numpy\lib\_function_base_impl.pyi`

```
# 导入 sys 模块，用于系统相关操作
import sys
# 从 collections.abc 模块中导入 Sequence、Iterator、Callable 和 Iterable 类型
from collections.abc import Sequence, Iterator, Callable, Iterable
# 从 typing 模块中导入多个类型别名和泛型类型
from typing import (
    Literal as L,  # 别名 L 表示 Literal
    Any,            # 任意类型
    TypeVar,        # 泛型类型变量
    overload,       # 函数重载装饰器
    Protocol,       # 协议类型
    SupportsIndex,  # 支持索引操作的类型
    SupportsInt,    # 支持整数操作的类型
)

# 如果 Python 版本大于等于 3.10，导入 TypeGuard 类型
if sys.version_info >= (3, 10):
    from typing import TypeGuard
# 否则，从 typing_extensions 模块导入 TypeGuard 类型
else:
    from typing_extensions import TypeGuard

# 从 numpy 模块中导入多个类型和函数
from numpy import (
    vectorize as vectorize,         # 向量化函数
    ufunc,                          # 通用函数
    generic,                        # 通用类型
    floating,                       # 浮点类型
    complexfloating,                # 复数浮点类型
    intp,                           # 整数指针类型
    float64,                        # 64 位浮点类型
    complex128,                     # 128 位复数类型
    timedelta64,                    # 时间差类型
    datetime64,                     # 日期时间类型
    object_,                        # 对象类型
    _OrderKACF,                     # 私有类型 _OrderKACF
)

# 从 numpy._typing 模块中导入多个类型别名
from numpy._typing import (
    NDArray,                 # Numpy 数组类型
    ArrayLike,               # 类似数组的类型
    DTypeLike,               # 数据类型的类型别名
    _ShapeLike,              # 形状的类型别名
    _ScalarLike_co,          # 协变标量类型别名
    _DTypeLike,              # 数据类型的类型别名
    _ArrayLike,              # 类似数组的类型别名
    _ArrayLikeInt_co,        # 协变整数数组类型别名
    _ArrayLikeFloat_co,      # 协变浮点数组类型别名
    _ArrayLikeComplex_co,    # 协变复数数组类型别名
    _ArrayLikeTD64_co,       # 协变时间差数组类型别名
    _ArrayLikeDT64_co,       # 协变日期时间数组类型别名
    _ArrayLikeObject_co,     # 协变对象数组类型别名
    _FloatLike_co,           # 协变浮点数类型别名
    _ComplexLike_co,         # 协变复数类型别名
)

# 从 numpy._core.multiarray 模块中导入 bincount 函数
from numpy._core.multiarray import (
    bincount as bincount,    # 统计非负整数的出现次数
)

# 定义类型变量 _T 和 _T_co
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
# _SCT 类型变量绑定于 generic 类型
_SCT = TypeVar("_SCT", bound=generic)
# _ArrayType 类型变量绑定于 NDArray 类型
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# 定义 _2Tuple 类型别名，表示两个相同类型的元组
_2Tuple = tuple[_T, _T]

# 定义 _TrimZerosSequence 协议，规定了长度、切片索引和迭代的行为
class _TrimZerosSequence(Protocol[_T_co]):
    def __len__(self) -> int: ...        # 返回序列的长度
    def __getitem__(self, key: slice, /) -> _T_co: ...  # 获取指定切片的元素
    def __iter__(self) -> Iterator[Any]: ...   # 返回迭代器

# 定义 _SupportsWriteFlush 协议，规定了写入和刷新的行为
class _SupportsWriteFlush(Protocol):
    def write(self, s: str, /) -> object: ...   # 写入字符串并返回对象
    def flush(self) -> object: ...              # 刷新并返回对象

# 定义 __all__ 列表，用于声明模块中公开的所有对象
__all__: list[str]

# 函数重载，旋转数组 m 并返回旋转后的结果
@overload
def rot90(
    m: _ArrayLike[_SCT],
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[_SCT]: ...

# 函数重载，旋转数组 m 并返回旋转后的结果
@overload
def rot90(
    m: ArrayLike,
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[Any]: ...

# 函数重载，沿指定轴翻转数组 m 并返回结果
@overload
def flip(m: _SCT, axis: None = ...) -> _SCT: ...

# 函数重载，对标量 m 执行翻转操作并返回结果
@overload
def flip(m: _ScalarLike_co, axis: None = ...) -> Any: ...

# 函数重载，沿指定轴翻转数组 m 并返回结果
@overload
def flip(m: _ArrayLike[_SCT], axis: None | _ShapeLike = ...) -> NDArray[_SCT]: ...

# 函数重载，对数组 m 执行翻转操作并返回结果
@overload
def flip(m: ArrayLike, axis: None | _ShapeLike = ...) -> NDArray[Any]: ...

# 函数定义，判断对象 y 是否可迭代并返回类型保护
def iterable(y: object) -> TypeGuard[Iterable[Any]]: ...

# 函数重载，计算数组 a 的加权平均值并返回结果
@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    weights: None | _ArrayLikeFloat_co= ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> floating[Any]: ...

# 函数重载，计算数组 a 的加权平均值并返回结果
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    weights: None | _ArrayLikeComplex_co = ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> complexfloating[Any, Any]: ...

# 函数重载，计算数组 a 的加权平均值并返回结果
@overload
def average(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    weights: None | Any = ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> Any: ...

# 函数重载，计算数组 a 的加权平均值并返回结果
@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    weights: None | _ArrayLikeFloat_co= ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,
) -> _2Tuple[floating[Any]]: ...

# 函数重载，计算数组 a 的加权平均值并返回结果
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    weights: None | _ArrayLikeComplex_co = ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,


定义一个名为 `keepdims` 的变量，其类型为 `L[False]`，初始值为未定义的占位符 (`...`)。
# 定义了一个函数签名，用于返回复数浮点数类型的二元组
) -> _2Tuple[complexfloating[Any, Any]]: ...

# 重载：计算数组或对象的加权平均值，返回两个元素的元组
@overload
def average(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    weights: None | Any = ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,
) -> _2Tuple[Any]: ...

# 重载：计算数组或复数数组的加权平均值，返回一个元素
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    weights: None | Any = ...,
    returned: L[False] = ...,
    keepdims: bool = ...,
) -> Any: ...

# 重载：计算数组或复数数组的加权平均值，返回两个元素的元组
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    weights: None | Any = ...,
    returned: L[True] = ...,
    keepdims: bool = ...,
) -> _2Tuple[Any]: ...

# 重载：将数组转换为NDArray，检查是否包含有限的值
@overload
def asarray_chkfinite(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[_SCT]: ...

# 重载：将对象转换为NDArray，检查是否包含有限的值
@overload
def asarray_chkfinite(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...

# 重载：将任意数据转换为NDArray，检查是否包含有限的值
@overload
def asarray_chkfinite(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
) -> NDArray[_SCT]: ...

# 重载：将任意数据转换为NDArray，检查是否包含有限的值
@overload
def asarray_chkfinite(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...

# TODO: 一旦mypy支持`Concatenate`，使用PEP 612 `ParamSpec`进行参考
# xref python/mypy#8645
# 重载：根据条件列表和函数列表计算并返回NDArray
@overload
def piecewise(
    x: _ArrayLike[_SCT],
    condlist: ArrayLike,
    funclist: Sequence[Any | Callable[..., Any]],
    *args: Any,
    **kw: Any,
) -> NDArray[_SCT]: ...

# 重载：根据条件列表和函数列表计算并返回NDArray
@overload
def piecewise(
    x: ArrayLike,
    condlist: ArrayLike,
    funclist: Sequence[Any | Callable[..., Any]],
    *args: Any,
    **kw: Any,
) -> NDArray[Any]: ...

# 根据条件列表选择并返回NDArray
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = ...,
) -> NDArray[Any]: ...

# 重载：复制给定数组，并根据指定顺序和可选性选择返回类型
@overload
def copy(
    a: _ArrayType,
    order: _OrderKACF,
    subok: L[True],
) -> _ArrayType: ...

# 重载：复制给定数组，并根据指定顺序和可选性选择返回类型
@overload
def copy(
    a: _ArrayType,
    order: _OrderKACF = ...,
    *,
    subok: L[True],
) -> _ArrayType: ...

# 重载：复制给定数组或对象，并根据指定顺序和可选性选择返回类型
@overload
def copy(
    a: _ArrayLike[_SCT],
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[_SCT]: ...

# 重载：复制给定数组或对象，并根据指定顺序和可选性选择返回类型
@overload
def copy(
    a: ArrayLike,
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[Any]: ...

# 计算给定函数的梯度，并返回结果
def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: None | _ShapeLike = ...,
    edge_order: L[1, 2] = ...,
) -> Any: ...

# 重载：计算数组的差分，返回相同类型的结果
@overload
def diff(
    a: _T,
    n: L[0],
    axis: SupportsIndex = ...,
    prepend: ArrayLike = ...,
    append: ArrayLike = ...,
) -> _T: ...

# 重载：计算数组的差分，返回NDArray
@overload
def diff(
    a: ArrayLike,
    n: int = ...,
    axis: SupportsIndex = ...,
    prepend: ArrayLike = ...,
    append: ArrayLike = ...,
) -> NDArray[Any]: ...

# 重载：根据已知数据点进行一维插值，返回浮点数NDArray
@overload
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeFloat_co,
    left: None | _FloatLike_co = ...,
    right: None | _FloatLike_co = ...,
    period: None | _FloatLike_co = ...,
) -> NDArray[float64]: ...

# 重载：根据已知数据点进行一维插值，返回浮点数NDArray
@overload
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeComplex_co,
    left: None | _ComplexLike_co = ...,
    right: None | _ComplexLike_co = ...,
    period: None | _FloatLike_co = ...,
# 定义函数签名，返回值为复数数组 NDArray[complex128]
def angle(z: _ComplexLike_co, deg: bool = ...) -> NDArray[complex128]: ...

# 重载函数 angle，接受 object_ 类型参数并返回浮点数
@overload
def angle(z: object_, deg: bool = ...) -> floating[Any]: ...

# 重载函数 angle，接受复数数组类型参数并返回浮点数数组 NDArray[floating[Any]]
@overload
def angle(z: _ArrayLikeComplex_co, deg: bool = ...) -> NDArray[floating[Any]]: ...

# 重载函数 angle，接受对象数组类型参数并返回对象数组 NDArray[object_]
@overload
def angle(z: _ArrayLikeObject_co, deg: bool = ...) -> NDArray[object_]: ...

# 定义函数签名，返回值为复数浮点数数组 NDArray[complexfloating[Any, Any]]
def sort_complex(a: ArrayLike) -> NDArray[complexfloating[Any, Any]]: ...

# 定义函数 trim_zeros，接受序列 filt 和可选参数 trim，并返回 _T 类型
def trim_zeros(
    filt: _TrimZerosSequence[_T],
    trim: L["f", "b", "fb", "bf"] = ...,
) -> _T: ...

# 重载函数 extract，接受条件数组和一维数组 arr，并返回符合条件的数组 NDArray[_SCT]
@overload
def extract(condition: ArrayLike, arr: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...

# 重载函数 extract，接受条件数组和一般数组 arr，并返回对象数组 NDArray[Any]
@overload
def extract(condition: ArrayLike, arr: ArrayLike) -> NDArray[Any]: ...

# 定义函数 place，接受数组 arr，掩码 mask 和值 vals，无返回值
def place(arr: NDArray[Any], mask: ArrayLike, vals: Any) -> None: ...

# 定义函数 disp，接受消息对象 mesg，设备 device 和换行符 linefeed，无返回值
def disp(
    mesg: object,
    device: None | _SupportsWriteFlush = ...,
    linefeed: bool = ...,
) -> None: ...

# 重载函数 cov，计算给定数据 m 和 y 的协方差矩阵，接受多个参数，返回浮点数数组 NDArray[floating[Any]]
@overload
def cov(
    m: _ArrayLikeFloat_co,
    y: None | _ArrayLikeFloat_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: None = ...,
) -> NDArray[floating[Any]]: ...

# 重载函数 cov，计算给定数据 m 和 y 的复数协方差矩阵，接受多个参数，返回复数浮点数数组 NDArray[complexfloating[Any, Any]]
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: None = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# 重载函数 cov，计算给定数据 m 和 y 的复数协方差矩阵，接受多个参数，返回指定类型数组 NDArray[_SCT]
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: _DTypeLike[_SCT],
) -> NDArray[_SCT]: ...

# 重载函数 cov，计算给定数据 m 和 y 的复数协方差矩阵，接受多个参数，返回任意类型数组 NDArray[Any]
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: DTypeLike,
) -> NDArray[Any]: ...

# 给出提示，表明 `bias` 和 `ddof` 已经被废弃
@overload
def corrcoef(
    m: _ArrayLikeFloat_co,
    y: None | _ArrayLikeFloat_co = ...,
    rowvar: bool = ...,
    *,
    dtype: None = ...,
) -> NDArray[floating[Any]]: ...

# 重载函数 corrcoef，计算给定数据 m 和 y 的相关系数矩阵，接受多个参数，返回复数浮点数数组 NDArray[complexfloating[Any, Any]]
@overload
def corrcoef(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: None = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: _DTypeLike[_SCT],
# 定义一个函数签名，该函数返回一个 NDArray，使用 SCT 作为类型注释
) -> NDArray[_SCT]: ...

# 重载函数签名，计算两个数组 m 和 y 的相关系数矩阵
@overload
def corrcoef(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: DTypeLike,
) -> NDArray[Any]: ...

# 定义一个函数 blackman，返回一个长度为 M 的黑曼窗口数组
def blackman(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

# 定义一个函数 bartlett，返回一个长度为 M 的巴特利特窗口数组
def bartlett(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

# 定义一个函数 hanning，返回一个长度为 M 的汉宁窗口数组
def hanning(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

# 定义一个函数 hamming，返回一个长度为 M 的哈明窗口数组
def hamming(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

# 定义一个函数 i0，计算修正 Bessel 函数 I_0 在数组 x 上的值
def i0(x: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...

# 定义一个函数 kaiser，返回一个长度为 M 的 Kaiser 窗口数组
def kaiser(
    M: _FloatLike_co,
    beta: _FloatLike_co,
) -> NDArray[floating[Any]]: ...

# 重载函数签名，计算 sinc 函数在浮点数 x 上的值
@overload
def sinc(x: _FloatLike_co) -> floating[Any]: ...
# 重载函数签名，计算 sinc 函数在复数 x 上的值
@overload
def sinc(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...
# 重载函数签名，计算 sinc 函数在数组 x 上每个元素的值
@overload
def sinc(x: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
# 重载函数签名，计算 sinc 函数在复数数组 x 上每个元素的值
@overload
def sinc(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# 重载函数签名，计算数组 a 沿指定轴的中位数
@overload
def median(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> floating[Any]: ...
# 重载函数签名，计算复数数组 a 沿指定轴的中位数
@overload
def median(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> complexfloating[Any, Any]: ...
# 重载函数签名，计算时间戳数组 a 沿指定轴的中位数
@overload
def median(
    a: _ArrayLikeTD64_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> timedelta64: ...
# 重载函数签名，计算对象数组 a 沿指定轴的中位数
@overload
def median(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> Any: ...
# 通用重载函数签名，计算各种类型的数组 a 沿指定轴的中位数
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> Any: ...
# 通用重载函数签名，计算各种类型的数组 a 沿指定轴的中位数并输出到数组 out
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> _ArrayType: ...

# 定义一个类型别名 _MethodKind，表示可能的百分位数计算方法
_MethodKind = L[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]

# 重载函数签名，计算数组 a 沿指定轴的百分位数
@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...,
) -> floating[Any]: ...
# 重载函数签名，计算复数数组 a 沿指定轴的百分位数
@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...,
) -> complexfloating[Any, Any]: ...
def percentile(
    a: _ArrayLikeTD64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> timedelta64:
    ...


# 定义了一个函数 percentile，用于计算时间增量数组（timedelta64）的百分位数
# 参数a是时间增量数组（timedelta64）或类似的数组
# 参数q是指定的百分位数，可以是单个浮点数或浮点数数组
# axis表示要沿着哪个轴计算百分位数，默认为None，表示计算整个数组的百分位数
# out表示结果输出的目标位置，默认为None，即函数内部创建新的数组来存放结果
# overwrite_input表示是否允许覆盖输入数组，默认为False，即不允许
# method是计算百分位数的方法，具体未指定时采用默认值
# keepdims表示是否保持结果的维度信息，默认为False，即不保持
# weights表示百分位数的权重，默认为None，表示所有数据点的权重相同
# 函数返回一个时间增量数组（timedelta64），表示计算得到的百分位数值



@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> datetime64:
    ...


# 使用 @overload 装饰器重载 percentile 函数，处理 datetime64 类型的输入
# 参数和返回类型与前一个函数定义类似，只是处理的数组类型为 datetime64


（以下重载部分省略，均按照相同的注释模式进行解释）


@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> Any:
    ...



@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> NDArray[floating[Any]]:
    ...



@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> NDArray[complexfloating[Any, Any]]:
    ...



@overload
def percentile(
    a: _ArrayLikeTD64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> NDArray[timedelta64]:
    ...



@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> NDArray[datetime64]:
    ...



@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> NDArray[object_]:
    ...



@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> Any:
    ...



@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...
) -> Any:
    ...


# 最后两个重载函数处理多种数组类型的百分位数计算
# 参数a可以是复数数组、时间增量数组、日期时间数组或普通对象数组
# 参数q可以是单个百分位数或百分位数数组
# axis表示要沿着哪个轴计算百分位数，可以是None或具体的轴形状
# out表示结果输出的目标位置，默认为None
# overwrite_input表示是否允许覆盖输入数组，默认为False
# method是计算百分位数的方法，具体未指定时采用默认值
# keepdims表示是否保持结果的维度信息，默认为False
# weights表示百分位数的权重，默认为None
# 这两个函数返回任意类型的数组，根据输入的具体情况可能是单个值、数组或对象
    keepdims: bool = ...,
    *,
    weights: None | _ArrayLikeFloat_co = ...,


# 定义函数参数 keepdims 和 weights
keepdims: bool = ...,
# "*" 表示此处之后的参数必须使用关键字方式传递
# weights 参数可以是 None 或者 float 类型的数组，使用 _ArrayLikeFloat_co 表示可能的类型
weights: None | _ArrayLikeFloat_co = ...,
# 定义一个类型提示注解，指定函数返回的类型为 _ArrayType
) -> _ArrayType: ...

# 注意：虽然不是别名，但它们有相同的签名（可以重用）
# 定义 quantile 函数，它和 percentile 函数有相同的签名和功能
quantile = percentile

# 定义 meshgrid 函数，用于生成坐标网格
def meshgrid(
    *xi: ArrayLike,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: L["xy", "ij"] = ...,
) -> tuple[NDArray[Any], ...]: ...

# 重载 delete 函数的签名，用于从数组中删除指定的元素或切片
@overload
def delete(
    arr: _ArrayLike[_SCT],
    obj: slice | _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def delete(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

# 重载 insert 函数的签名，用于在数组中插入元素或切片
@overload
def insert(
    arr: _ArrayLike[_SCT],
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def insert(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

# 定义 append 函数，用于在数组末尾追加元素
def append(
    arr: ArrayLike,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

# 重载 digitize 函数的签名，用于计算数组中每个元素所属的区间索引
@overload
def digitize(
    x: _FloatLike_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> intp: ...
@overload
def digitize(
    x: _ArrayLikeFloat_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> NDArray[intp]: ...
```