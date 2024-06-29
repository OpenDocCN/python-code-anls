# `D:\src\scipysrc\pandas\pandas\core\arrays\arrow\array.py`

```
# 导入未来版本的注解支持
from __future__ import annotations

# 导入 functools 模块，用于高阶函数操作
import functools
# 导入 operator 模块，用于操作符函数
import operator
# 导入 re 模块，用于正则表达式操作
import re
# 导入 textwrap 模块，用于文本包装和填充
import textwrap
# 导入 typing 模块，用于类型注解
from typing import (
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    Literal,  # 字面量类型
    cast,  # 类型转换函数
    overload,  # 重载函数
)

# 导入 unicodedata 模块，用于 Unicode 数据库操作
import unicodedata

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 从 pandas._libs 中导入 lib 模块
from pandas._libs import lib
# 从 pandas._libs.tslibs 中导入时间间隔和时间戳相关模块
from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
    timezones,
)
# 从 pandas.compat 中导入版本比较函数
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
    pa_version_under13p0,
)
# 从 pandas.util._decorators 中导入文档装饰器
from pandas.util._decorators import doc

# 从 pandas.core.dtypes.cast 中导入类型转换相关函数
from pandas.core.dtypes.cast import (
    can_hold_element,
    infer_dtype_from_scalar,
)
# 从 pandas.core.dtypes.common 中导入常见类型判断函数
from pandas.core.dtypes.common import (
    CategoricalDtype,
    is_array_like,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
# 从 pandas.core.dtypes.dtypes 中导入日期时间类型
from pandas.core.dtypes.dtypes import DatetimeTZDtype
# 从 pandas.core.dtypes.missing 中导入缺失值检测函数
from pandas.core.dtypes.missing import isna

# 从 pandas.core 中导入算法相关模块和缺失值处理模块
from pandas.core import (
    algorithms as algos,
    missing,
    ops,
    roperator,
)
# 从 pandas.core.algorithms 中导入数组映射函数
from pandas.core.algorithms import map_array
# 从 pandas.core.arraylike 中导入数组操作混合类
from pandas.core.arraylike import OpsMixin
# 从 pandas.core.arrays._arrow_string_mixins 中导入 ArrowStringArrayMixin 类
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
# 从 pandas.core.arrays._utils 中导入类型推断函数
from pandas.core.arrays._utils import to_numpy_dtype_inference
# 从 pandas.core.arrays.base 中导入扩展数组相关类
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionArraySupportsAnyAll,
)
# 从 pandas.core.arrays.masked 中导入基础掩码数组类
from pandas.core.arrays.masked import BaseMaskedArray
# 从 pandas.core.arrays.string_ 中导入字符串类型
from pandas.core.arrays.string_ import StringDtype
# 从 pandas.core.common 中导入通用函数
import pandas.core.common as com
# 从 pandas.core.indexers 中导入索引相关函数
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)
# 从 pandas.core.strings.base 中导入基础字符串数组方法类
from pandas.core.strings.base import BaseStringArrayMethods

# 从 pandas.io._util 中导入 Arrow 类型映射函数
from pandas.io._util import _arrow_dtype_mapping
# 从 pandas.tseries.frequencies 中导入偏移量转换函数
from pandas.tseries.frequencies import to_offset

# 如果不是低于 10.1 版本的 pandas，导入 pyarrow 和相关计算模块
if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

    # 从 pandas.core.dtypes.dtypes 中导入 ArrowDtype 类型
    from pandas.core.dtypes.dtypes import ArrowDtype

    # 定义 Arrow 数组的比较函数映射
    ARROW_CMP_FUNCS = {
        "eq": pc.equal,      # 等于
        "ne": pc.not_equal,  # 不等于
        "lt": pc.less,       # 小于
        "gt": pc.greater,    # 大于
        "le": pc.less_equal, # 小于等于
        "ge": pc.greater_equal,  # 大于等于
    }

    # 定义 Arrow 数组的逻辑运算函数映射
    ARROW_LOGICAL_FUNCS = {
        "and_": pc.and_kleene,     # 与
        "rand_": lambda x, y: pc.and_kleene(y, x),  # 翻转与
        "or_": pc.or_kleene,       # 或
        "ror_": lambda x, y: pc.or_kleene(y, x),   # 翻转或
        "xor": pc.xor,             # 异或
        "rxor": lambda x, y: pc.xor(y, x),         # 翻转异或
    }

    # 定义 Arrow 数组的位运算函数映射
    ARROW_BIT_WISE_FUNCS = {
        "and_": pc.bit_wise_and,     # 位与
        "rand_": lambda x, y: pc.bit_wise_and(y, x),  # 翻转位与
        "or_": pc.bit_wise_or,       # 位或
        "ror_": lambda x, y: pc.bit_wise_or(y, x),    # 翻转位或
        "xor": pc.bit_wise_xor,      # 位异或
        "rxor": lambda x, y: pc.bit_wise_xor(y, x),   # 翻转位异或
    }

    # 定义用于真除的 Arrow 数组转换函数
    def cast_for_truediv(
        arrow_array: pa.ChunkedArray, pa_object: pa.Array | pa.Scalar
    ) -> tuple[pa.ChunkedArray, pa.Array | pa.Scalar]:
        # 确保整数 / 整数 -> 浮点数，模仿 Python/Numpy 的行为
        # 因为 pc.divide_checked(int, int) -> int
        if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(
            pa_object.type
        ):
            # GH: 56645.
            # https://github.com/apache/arrow/issues/35563
            # 将 arrow_array 和 pa_object 转换为 float64 类型，不进行安全检查
            return pc.cast(arrow_array, pa.float64(), safe=False), pc.cast(
                pa_object, pa.float64(), safe=False
            )

        return arrow_array, pa_object

    def floordiv_compat(
        left: pa.ChunkedArray | pa.Array | pa.Scalar,
        right: pa.ChunkedArray | pa.Array | pa.Scalar,
    ) -> pa.ChunkedArray:
        # TODO: 替换为 pyarrow 的 floordiv 内核。
        # https://github.com/apache/arrow/issues/39386
        if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
            # 对 left 和 right 进行整数除法，返回结果
            divided = pc.divide_checked(left, right)
            if pa.types.is_signed_integer(divided.type):
                # GH 56676
                # 检查是否有余数
                has_remainder = pc.not_equal(pc.multiply(divided, right), left)
                # 检查是否有一个操作数为负数
                has_one_negative_operand = pc.less(
                    pc.bit_wise_xor(left, right),
                    pa.scalar(0, type=divided.type),
                )
                # 根据条件返回减一或者原始结果
                result = pc.if_else(
                    pc.and_(
                        has_remainder,
                        has_one_negative_operand,
                    ),
                    # GH: 55561
                    pc.subtract(divided, pa.scalar(1, type=divided.type)),
                    divided,
                )
            else:
                result = divided
            # 将结果转换为 left 的类型
            result = result.cast(left.type)
        else:
            # 对 left 和 right 进行普通除法，然后向下取整
            divided = pc.divide(left, right)
            result = pc.floor(divided)
        return result

    ARROW_ARITHMETIC_FUNCS = {
        "add": pc.add_checked,
        "radd": lambda x, y: pc.add_checked(y, x),
        "sub": pc.subtract_checked,
        "rsub": lambda x, y: pc.subtract_checked(y, x),
        "mul": pc.multiply_checked,
        "rmul": lambda x, y: pc.multiply_checked(y, x),
        "truediv": lambda x, y: pc.divide(*cast_for_truediv(x, y)),
        "rtruediv": lambda x, y: pc.divide(*cast_for_truediv(y, x)),
        "floordiv": lambda x, y: floordiv_compat(x, y),
        "rfloordiv": lambda x, y: floordiv_compat(y, x),
        "mod": NotImplemented,
        "rmod": NotImplemented,
        "divmod": NotImplemented,
        "rdivmod": NotImplemented,
        "pow": pc.power_checked,
        "rpow": lambda x, y: pc.power_checked(y, x),
    }
# 如果类型检查开启，则导入必要的类型和模块
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Sequence,
    )

    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        Iterator,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        Scalar,
        Self,
        SortKind,
        TakeIndexer,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Series
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray


def get_unit_from_pa_dtype(pa_dtype) -> str:
    """
    根据给定的类型获取其时间单位。

    Parameters
    ----------
    pa_dtype : ArrowDtype | pa.DataType | Dtype | None
        箭头数据类型、PyArrow 数据类型、Pandas 数据类型或空

    Returns
    -------
    str
        时间单位字符串

    Raises
    ------
    ValueError
        如果时间单位不是 's', 'ms', 'us', 'ns' 中的一种

    Notes
    -----
    此函数用于从 Pandas 中的特定类型中提取时间单位信息。

    See Also
    --------
    to_pyarrow_type : 将 Pandas 数据类型转换为 PyArrow 数据类型的函数
    """
    # 如果 Pandas 版本小于 11.0，则直接从字符串中提取时间单位
    if pa_version_under11p0:
        unit = str(pa_dtype).split("[", 1)[-1][:-1]
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError(pa_dtype)
        return unit
    # 否则返回类型对象的单位信息
    return pa_dtype.unit


def to_pyarrow_type(
    dtype: ArrowDtype | pa.DataType | Dtype | None,
) -> pa.DataType | None:
    """
    将 Pandas 数据类型转换为 PyArrow 数据类型实例。

    Parameters
    ----------
    dtype : ArrowDtype | pa.DataType | Dtype | None
        Pandas 数据类型、PyArrow 数据类型、Numpy 数据类型或空

    Returns
    -------
    pa.DataType | None
        PyArrow 数据类型实例或空

    Notes
    -----
    此函数用于将 Pandas 中的数据类型转换为 PyArrow 中对应的数据类型实例。
    支持 ArrowDtype、pa.DataType 和 DatetimeTZDtype 的转换。
    如果传入的 dtype 不被支持，则返回空值 None。

    See Also
    --------
    get_unit_from_pa_dtype : 从 Pandas 数据类型中获取时间单位的函数
    """
    if isinstance(dtype, ArrowDtype):
        return dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        return dtype
    elif isinstance(dtype, DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)
    elif dtype:
        try:
            # 尝试从 Numpy 数据类型转换为 PyArrow 数据类型
            return pa.from_numpy_dtype(dtype)
        except pa.ArrowNotImplementedError:
            pass
    # 如果无法转换，返回空值
    return None


class ArrowExtensionArray(
    OpsMixin,
    ExtensionArraySupportsAnyAll,
    ArrowStringArrayMixin,
    BaseStringArrayMethods,
):
    """
    使用 PyArrow ChunkedArray 支持的 Pandas ExtensionArray。

    .. warning::

       ArrowExtensionArray 目前属于实验性功能。其实现和部分 API 可能会在无警告的情况下发生更改。

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        PyArrow 数组或分块数组

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    ArrowExtensionArray
        返回一个 ArrowExtensionArray 实例

    Notes
    -----
    大多数方法都使用 `pyarrow compute functions <https://arrow.apache.org/docs/python/api/compute.html>`__ 实现。
    某些方法如果基于安装的 PyArrow 版本不可用，可能会引发异常或 ``PerformanceWarning`` 。

    请安装最新版本的 PyArrow 以启用最佳功能并避免可能在旧版本 PyArrow 中出现的错误。

    Examples
    --------
    使用 :func:`pandas.array` 创建 ArrowExtensionArray：

    >>> pd.array([1, 1, None], dtype="int64[pyarrow]")
    <ArrowExtensionArray>
    [1, 1, <NA>]
    Length: 3, dtype: int64[pyarrow]
    """  # noqa: E501 (http link too long)
    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        # 检查是否满足pyarrow版本>=10.0.1的要求
        if pa_version_under10p1:
            msg = "pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray."
            raise ImportError(msg)
        # 如果传入的值是pa.Array类型，则封装成单一的pa.ChunkedArray
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        # 如果传入的值已经是pa.ChunkedArray类型，则直接使用
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            # 如果传入的值既不是pa.Array也不是pa.ChunkedArray，则抛出异常
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        # 根据_pa_array的类型创建对应的ArrowDtype
        self._dtype = ArrowDtype(self._pa_array.type)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        # 将标量序列转换为pyarrow类型的数组
        pa_type = to_pyarrow_type(dtype)
        pa_array = cls._box_pa_array(scalars, pa_type=pa_type, copy=copy)
        # 使用转换后的pa_array创建新的ExtensionArray对象
        arr = cls(pa_array)
        return arr

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
        # 将字符串序列转换为pyarrow类型的数组，使用指定的ExtensionDtype
        pa_type = to_pyarrow_type(dtype)
        pa_array = cls._box_pa_array(strings, pa_type=pa_type, copy=copy)
        # 使用转换后的pa_array创建新的ExtensionArray对象
        arr = cls(pa_array)
        return arr

    @classmethod
    def _box_pa(
        cls, value, pa_type: pa.DataType | None = None
    ) -> pa.Array | pa.ChunkedArray | pa.Scalar:
        """
        Box value into a pyarrow Array, ChunkedArray or Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray or pa.Scalar
        """
        # 将给定的值转换为pyarrow的Array、ChunkedArray或Scalar
        if isinstance(value, pa.Scalar) or not is_list_like(value):
            return cls._box_pa_scalar(value, pa_type)
        return cls._box_pa_array(value, pa_type)
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        """
        Box value into a pyarrow Scalar.

        Parameters
        ----------
        value : any
            The value to be boxed into a pyarrow Scalar.
        pa_type : pa.DataType | None
            Optional type hint for the pyarrow Scalar.

        Returns
        -------
        pa.Scalar
            The boxed pyarrow Scalar object.
        """
        if isinstance(value, pa.Scalar):
            # If value is already a pyarrow Scalar, use it directly
            pa_scalar = value
        elif isna(value):
            # If value is NA (missing data), create a pyarrow Scalar representing None
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
            # For non-pyarrow Scalars, handle special cases for Timestamp and Timedelta
            if isinstance(value, Timedelta):
                # Convert Timedelta to pyarrow duration scalar
                if pa_type is None:
                    pa_type = pa.duration(value.unit)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value
            elif isinstance(value, Timestamp):
                # Convert Timestamp to pyarrow timestamp scalar
                if pa_type is None:
                    pa_type = pa.timestamp(value.unit, tz=value.tz)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value

            # Box the value into a pyarrow Scalar
            pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)

        # Ensure the returned Scalar matches the specified type if pa_type is provided
        if pa_type is not None and pa_scalar.type != pa_type:
            pa_scalar = pa_scalar.cast(pa_type)

        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array:
        """
        Box value into a pyarrow Array.

        Parameters
        ----------
        value : any
            The value to be boxed into a pyarrow Array.
        pa_type : pa.DataType | None
            Optional type hint for the pyarrow Array.
        copy : bool
            Whether to copy the data.

        Returns
        -------
        pa.Array
            The boxed pyarrow Array object.
        """
        na_value = cls._dtype.na_value
        pa_type = cls._pa_array.type  # Get the type of the pyarrow Array
        box_timestamp = pa.types.is_timestamp(pa_type) and pa_type.unit != "ns"
        box_timedelta = pa.types.is_duration(pa_type) and pa_type.unit != "ns"
        for value in cls._pa_array:
            val = value.as_py()  # Convert pyarrow value to Python object
            if val is None:
                yield na_value  # Yield NA value for missing data
            elif box_timestamp:
                yield Timestamp(val).as_unit(pa_type.unit)  # Yield Timestamp adjusted to pa_type unit
            elif box_timedelta:
                yield Timedelta(val).as_unit(pa_type.unit)  # Yield Timedelta adjusted to pa_type unit
            else:
                yield val  # Yield the original value

    def __arrow_array__(self, type=None) -> pa.ChunkedArray:
        """
        Convert the object to a pyarrow ChunkedArray.

        Parameters
        ----------
        type : any
            Optional type hint.

        Returns
        -------
        pa.ChunkedArray
            The converted pyarrow ChunkedArray object.
        """
        return self._pa_array

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Convert the object to a numpy ndarray.

        Parameters
        ----------
        dtype : np.dtype | None
            Optional numpy data type.
        copy : bool | None
            Whether to copy the data.

        Returns
        -------
        np.ndarray
            The converted numpy ndarray.
        """
        return self.to_numpy(dtype=dtype)
    # 实现按位取反操作符，返回与操作数类型相同的对象
    def __invert__(self) -> Self:
        # 如果操作数是整数类型
        if pa.types.is_integer(self._pa_array.type):
            # 调用位取反操作，返回新的对象
            return type(self)(pc.bit_wise_not(self._pa_array))
        # 如果操作数是字符串类型或大字符串类型
        elif pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            # 抛出类型错误，不支持对字符串类型进行按位取反操作
            raise TypeError("__invert__ is not supported for string dtypes")
        else:
            # 对于其他类型的操作数，调用按位取反操作，返回新的对象
            return type(self)(pc.invert(self._pa_array))

    # 实现取负数操作符，返回与操作数类型相同的对象
    def __neg__(self) -> Self:
        # 调用取负数操作，返回新的对象
        return type(self)(pc.negate_checked(self._pa_array))

    # 实现取正数操作符，返回与操作数类型相同的对象
    def __pos__(self) -> Self:
        # 直接返回当前对象，因为取正数操作不改变数值
        return type(self)(self._pa_array)

    # 实现绝对值操作符，返回与操作数类型相同的对象
    def __abs__(self) -> Self:
        # 调用取绝对值操作，返回新的对象
        return type(self)(pc.abs_checked(self._pa_array))

    # GH 42600: 一旦 https://issues.apache.org/jira/browse/ARROW-10739 得到解决，__getstate__/__setstate__ 将不再需要
    # 获取对象的状态以便序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        # 合并所有块以优化性能
        state["_pa_array"] = self._pa_array.combine_chunks()
        return state

    # 设置对象的状态以便反序列化
    def __setstate__(self, state) -> None:
        # 如果状态中包含 "_data"，则使用 "_data"，否则使用 "_pa_array"
        if "_data" in state:
            data = state.pop("_data")
        else:
            data = state["_pa_array"]
        # 创建新的块化数组对象
        state["_pa_array"] = pa.chunked_array(data)
        # 更新对象的状态
        self.__dict__.update(state)

    # 私有方法，用于比较操作符的实现
    def _cmp_method(self, other, op) -> ArrowExtensionArray:
        # 根据操作符获取对应的处理函数
        pc_func = ARROW_CMP_FUNCS[op.__name__]
        # 如果另一个操作数是 ArrowExtensionArray、np.ndarray、list 或 BaseMaskedArray
        # 或者其 dtype 是 CategoricalDtype
        if isinstance(
            other, (ArrowExtensionArray, np.ndarray, list, BaseMaskedArray)
        ) or isinstance(getattr(other, "dtype", None), CategoricalDtype):
            # 调用对应的处理函数，返回结果
            result = pc_func(self._pa_array, self._box_pa(other))
        # 如果另一个操作数是标量值
        elif is_scalar(other):
            try:
                # 尝试调用对应的处理函数，返回结果
                result = pc_func(self._pa_array, self._box_pa(other))
            except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                # 处理 Arrow 库可能抛出的错误，生成一个掩码用于无效值
                mask = isna(self) | isna(other)
                valid = ~mask
                # 初始化结果数组
                result = np.zeros(len(self), dtype="bool")
                np_array = np.array(self)
                try:
                    # 尝试对有效值执行操作，生成布尔结果
                    result[valid] = op(np_array[valid], other)
                except TypeError:
                    # 如果类型错误，处理无效比较
                    result = ops.invalid_comparison(np_array, other, op)
                # 转换为 Arrow 的布尔数组对象
                result = pa.array(result, type=pa.bool_())
                # 使用条件表达式处理有效值和无效值
                result = pc.if_else(valid, result, None)
        else:
            # 如果操作数类型不支持，则抛出未实现错误
            raise NotImplementedError(
                f"{op.__name__} not implemented for {type(other)}"
            )
        # 返回 Arrow 扩展数组对象
        return ArrowExtensionArray(result)
    def _evaluate_op_method(self, other, op, arrow_funcs) -> Self:
        # 获取当前数组的数据类型
        pa_type = self._pa_array.type
        # 将输入的其他数据进行封装处理
        other = self._box_pa(other)

        # 如果当前数组是字符串、大字符串或二进制类型之一
        if (
            pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
            or pa.types.is_binary(pa_type)
        ):
            # 如果操作是加法或反向加法
            if op in [operator.add, roperator.radd]:
                # 创建一个空的标量以作为字符串连接的分隔符
                sep = pa.scalar("", type=pa_type)
                if op is operator.add:
                    # 对当前数组和其他数据进行逐元素连接操作
                    result = pc.binary_join_element_wise(self._pa_array, other, sep)
                elif op is roperator.radd:
                    # 对其他数据和当前数组进行逐元素连接操作
                    result = pc.binary_join_element_wise(other, self._pa_array, sep)
                # 返回一个新对象，其中包含连接后的结果
                return type(self)(result)
            # 如果操作是乘法或反向乘法
            elif op in [operator.mul, roperator.rmul]:
                binary = self._pa_array
                integral = other
                # 如果乘数不是整数类型，则抛出类型错误异常
                if not pa.types.is_integer(integral.type):
                    raise TypeError("Can only string multiply by an integer.")
                # 对当前数组进行乘法重复操作
                pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
                result = pc.binary_repeat(binary, pa_integral)
                # 返回一个新对象，其中包含重复后的结果
                return type(self)(result)
        
        # 如果其他数据是字符串、二进制或大字符串类型之一，且操作是乘法或反向乘法
        elif (
            pa.types.is_string(other.type)
            or pa.types.is_binary(other.type)
            or pa.types.is_large_string(other.type)
        ) and op in [operator.mul, roperator.rmul]:
            binary = other
            integral = self._pa_array
            # 如果乘数不是整数类型，则抛出类型错误异常
            if not pa.types.is_integer(integral.type):
                raise TypeError("Can only string multiply by an integer.")
            # 对当前数组进行乘法重复操作
            pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
            result = pc.binary_repeat(binary, pa_integral)
            # 返回一个新对象，其中包含重复后的结果
            return type(self)(result)
        
        # 如果其他数据是标量且为空，并且操作是逻辑函数
        if (
            isinstance(other, pa.Scalar)
            and pc.is_null(other).as_py()
            and op.__name__ in ARROW_LOGICAL_FUNCS
        ):
            # 对空标量进行类型转换，以便与当前数组类型匹配
            other = other.cast(pa_type)

        # 获取操作对应的计算函数
        pc_func = arrow_funcs[op.__name__]
        # 如果计算函数未实现，则抛出未实现错误异常
        if pc_func is NotImplemented:
            raise NotImplementedError(f"{op.__name__} not implemented.")

        # 应用计算函数，对当前数组和其他数据进行操作
        result = pc_func(self._pa_array, other)
        # 返回一个新对象，其中包含计算后的结果
        return type(self)(result)

    def _logical_method(self, other, op) -> Self:
        # 对于整数类型，^、|、& 是位运算符并返回整数类型；否则是逻辑运算符
        if pa.types.is_integer(self._pa_array.type):
            return self._evaluate_op_method(other, op, ARROW_BIT_WISE_FUNCS)
        else:
            return self._evaluate_op_method(other, op, ARROW_LOGICAL_FUNCS)

    def _arith_method(self, other, op) -> Self:
        # 调用通用的计算方法，使用算术函数集合
        return self._evaluate_op_method(other, op, ARROW_ARITHMETIC_FUNCS)
    def equals(self, other) -> bool:
        # 检查参数是否为 ArrowExtensionArray 类型的对象，如果不是则返回 False
        if not isinstance(other, ArrowExtensionArray):
            return False
        # 比较当前对象的 _pa_array 属性与另一个对象的 _pa_array 属性是否相等
        # 我被告知 pyarrow 使 __eq__ 的行为类似于 pandas 的 equals 方法；
        # TODO: 这个行为在哪里有文档记录？
        return self._pa_array == other._pa_array

    @property
    def dtype(self) -> ArrowDtype:
        """
        'ExtensionDtype' 的一个实例。
        """
        # 返回当前对象的 _dtype 属性
        return self._dtype

    @property
    def nbytes(self) -> int:
        """
        存储此对象所需的字节数。
        """
        # 返回当前对象的 _pa_array 属性的 nbytes 属性值
        return self._pa_array.nbytes

    def __len__(self) -> int:
        """
        返回此数组的长度。

        Returns
        -------
        length : int
        """
        # 返回当前对象的 _pa_array 属性的长度
        return len(self._pa_array)

    def __contains__(self, key) -> bool:
        # https://github.com/pandas-dev/pandas/pull/51307#issuecomment-1426372604
        # 检查给定的 key 是否是缺失值，并且不等于 dtype 的 na_value
        if isna(key) and key is not self.dtype.na_value:
            # 如果 dtype 的 kind 是 "f"，且 key 是浮点数，则检查 _pa_array 是否包含 NaN
            if self.dtype.kind == "f" and lib.is_float(key):
                return pc.any(pc.is_nan(self._pa_array)).as_py()

            # 对于日期或时间戳类型，不允许 key 为 None，以匹配 pd.NA
            return False
            # TODO: 或许复数？对象？

        # 调用父类的 __contains__ 方法，检查 key 是否存在于当前对象中
        return bool(super().__contains__(key))

    @property
    def _hasna(self) -> bool:
        # 返回当前对象的 _pa_array 属性的 null_count 是否大于 0
        return self._pa_array.null_count > 0

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        返回一个布尔型的 NumPy 数组，指示每个值是否缺失。

        应返回一个与 'self' 长度相同的 1-D 数组。
        """
        # GH51630: 快速路径
        null_count = self._pa_array.null_count
        # 如果 null_count 为 0，则返回一个全为 False 的布尔数组
        if null_count == 0:
            return np.zeros(len(self), dtype=np.bool_)
        # 如果 null_count 等于数组长度，则返回一个全为 True 的布尔数组
        elif null_count == len(self):
            return np.ones(len(self), dtype=np.bool_)

        # 否则，返回 _pa_array 的 is_null() 方法转换为 NumPy 数组的结果
        return self._pa_array.is_null().to_numpy()

    @overload
    def any(self, *, skipna: Literal[True] = ..., **kwargs) -> bool: ...
    @overload
    def any(self, *, skipna: bool, **kwargs) -> bool | NAType: ...
    def any(self, *, skipna: bool = True, **kwargs) -> bool | NAType:
        """
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA's present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        """
        return self._reduce("any", skipna=skipna, **kwargs)
    def all(self, *, skipna: bool = True, **kwargs) -> bool | NAType:
        """
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA's present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        """
        # 调用私有方法 _reduce，执行所有元素的逻辑操作，返回结果
        return self._reduce("all", skipna=skipna, **kwargs)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        # 根据 ascending 参数确定排序顺序，构造排序方式的字符串
        order = "ascending" if ascending else "descending"
        # 根据 na_position 参数确定空值处理方式，映射到对应的枚举值
        null_placement = {"last": "at_end", "first": "at_start"}.get(na_position, None)
        # 如果无效的 na_position 参数，抛出 ValueError 异常
        if null_placement is None:
            raise ValueError(f"invalid na_position: {na_position}")

        # 使用 ArrowExtensionArray 提供的 array_sort_indices 方法进行排序
        result = pc.array_sort_indices(
            self._pa_array, order=order, null_placement=null_placement
        )
        # 将排序结果转换为 NumPy 数组类型 np.ndarray
        np_result = result.to_numpy()
        # 返回排序结果，以 np.intp 类型返回，避免复制数据
        return np_result.astype(np.intp, copy=False)
    def _argmin_max(self, skipna: bool, method: str) -> int:
        # 检查数组长度为0或全部为null，或者存在null且skipna为False时
        if self._pa_array.length() in (0, self._pa_array.null_count) or (
            self._hasna and not skipna
        ):
            # 对于空数组或全部为null，pyarrow返回-1，但pandas期望TypeError
            # 对于skipna=False且数据中存在null，pandas期望NotImplementedError
            # 让ExtensionArray.arg{max|min}抛出异常
            return getattr(super(), f"arg{method}")(skipna=skipna)

        # 获取数据数组
        data = self._pa_array
        # 如果数据类型为持续时间，将其转换为int64类型
        if pa.types.is_duration(data.type):
            data = data.cast(pa.int64())

        # 使用pyarrow.compute库计算最大或最小值的索引
        value = getattr(pc, method)(data, skip_nulls=skipna)
        # 返回索引对应的Python原生整数值
        return pc.index(data, value).as_py()

    def argmin(self, skipna: bool = True) -> int:
        # 调用_argmin_max方法，计算最小值的索引
        return self._argmin_max(skipna, "min")

    def argmax(self, skipna: bool = True) -> int:
        # 调用_argmin_max方法，计算最大值的索引
        return self._argmin_max(skipna, "max")

    def copy(self) -> Self:
        """
        返回数组的浅拷贝。

        Underlying ChunkedArray是不可变的，因此不需要进行深拷贝。

        Returns
        -------
        type(self)
        """
        # 使用相同类型的构造函数创建新对象，参数为当前_pa_array
        return type(self)(self._pa_array)

    def dropna(self) -> Self:
        """
        返回不含NA值的ArrowExtensionArray。

        Returns
        -------
        ArrowExtensionArray
        """
        # 使用pc.drop_null函数创建不含null值的新ArrowExtensionArray对象
        return type(self)(pc.drop_null(self._pa_array))

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # 如果没有null值存在，直接返回当前对象
        if not self._hasna:
            return self

        # 如果limit和limit_area都为None，则根据method执行相应填充操作
        if limit is None and limit_area is None:
            # 清理填充方法
            method = missing.clean_fill_method(method)
            try:
                if method == "pad":
                    # 前向填充null值，创建新对象
                    return type(self)(pc.fill_null_forward(self._pa_array))
                elif method == "backfill":
                    # 后向填充null值，创建新对象
                    return type(self)(pc.fill_null_backward(self._pa_array))
            except pa.ArrowNotImplementedError:
                # 如果填充方法不支持，则捕获异常并忽略
                # ArrowNotImplementedError: Function 'coalesce' has no kernel
                #   matching input types (duration[ns], duration[ns])
                # TODO: 如果pyarrow实现了对持续时间类型的内核，移除try/except包装器
                pass

        # TODO: 为什么不再需要上述情况？
        # TODO(3.0): 在EA.fillna 'method'被弃用后，可以完全删除此方法。
        # 调用父类的_pad_or_backfill方法，进行填充或回填操作
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    @doc(ExtensionArray.fillna)
    def fillna(
        self,
        value: object | ArrayLike,
        limit: int | None = None,
        copy: bool = True,
        ...
    ) -> Self:
        # 如果没有缺失值，返回当前对象的副本
        if not self._hasna:
            return self.copy()

        # 如果指定了限制参数，调用父类的 fillna 方法进行填充
        if limit is not None:
            return super().fillna(value=value, limit=limit, copy=copy)

        # 如果 value 是 np.ndarray 或 ExtensionArray 类型，验证其长度是否与当前对象相同
        if isinstance(value, (np.ndarray, ExtensionArray)):
            # 类似于 check_value_size，但这里不会屏蔽，因为可能会传递给 super() 方法
            if len(value) != len(self):
                raise ValueError(
                    f"Length of 'value' does not match. Got ({len(value)}) "
                    f" expected {len(self)}"
                )

        try:
            # 使用 self._box_pa 方法将 value 转换为适合的类型
            fill_value = self._box_pa(value, pa_type=self._pa_array.type)
        except pa.ArrowTypeError as err:
            # 如果类型转换出错，抛出异常
            msg = f"Invalid value '{value!s}' for dtype {self.dtype}"
            raise TypeError(msg) from err

        try:
            # 使用 pc.fill_null 方法填充缺失值
            return type(self)(pc.fill_null(self._pa_array, fill_value=fill_value))
        except pa.ArrowNotImplementedError:
            # 如果填充操作不被支持，通常处理 ArrowNotImplementedError 异常
            # ArrowNotImplementedError: Function 'coalesce' has no kernel
            #   matching input types (duration[ns], duration[ns])
            # TODO: 如果 pyarrow 实现了 duration 类型的内核，移除 try/except 包装器
            pass

        # 最后调用父类的 fillna 方法进行填充
        return super().fillna(value=value, limit=limit, copy=copy)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # 如果 values 为空数组，则返回长度与 self 相同的全 False 数组
        if not len(values):
            return np.zeros(len(self), dtype=bool)

        # 使用 pc.is_in 方法判断 self._pa_array 中的元素是否在 values 中
        result = pc.is_in(self._pa_array, value_set=pa.array(values, from_pandas=True))
        # pyarrow 2.0.0 返回 nulls，因此我们明确指定 dtype 以将 nulls 转换为 False
        return np.array(result, dtype=np.bool_)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        返回适合因子化的数组和缺失值。

        Returns
        -------
        values : ndarray
        na_value : pd.NA

        Notes
        -----
        此方法返回的值也用于 pandas.util.hash_pandas_object。
        """
        # 将 self._pa_array 转换为 numpy 数组
        values = self._pa_array.to_numpy()
        # 返回 numpy 数组和当前对象的缺失值
        return values, self.dtype.na_value

    @doc(ExtensionArray.factorize)
    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        # 确定空值的编码方式，如果使用 NA 哨兵则为 "mask"，否则为 "encode"
        null_encoding = "mask" if use_na_sentinel else "encode"

        # 获取当前 Arrow 扩展数组的数据
        data = self._pa_array
        # 获取数据的类型
        pa_type = data.type
        # 如果数据的版本低于11.0且数据类型为时间间隔类型，则进行类型转换
        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # 强制将数据类型转换为 int64 类型
            data = data.cast(pa.int64())

        # 如果数据类型是字典类型，则直接使用该数据
        if pa.types.is_dictionary(data.type):
            encoded = data
        else:
            # 否则对数据进行字典编码
            encoded = data.dictionary_encode(null_encoding=null_encoding)
        
        # 如果编码后的数据长度为0，则返回空数组和空的扩展数组
        if encoded.length() == 0:
            indices = np.array([], dtype=np.intp)
            uniques = type(self)(pa.chunked_array([], type=encoded.type.value_type))
        else:
            # 合并编码后的数据块
            combined = encoded.combine_chunks()
            # 获取合并后的数据块的索引
            pa_indices = combined.indices
            # 如果索引中包含空值，则用 -1 填充这些空值
            if pa_indices.null_count > 0:
                pa_indices = pc.fill_null(pa_indices, -1)
            # 将索引转换为 NumPy 数组
            indices = pa_indices.to_numpy(zero_copy_only=False, writable=True).astype(
                np.intp, copy=False
            )
            # 构建包含字典的扩展数组
            uniques = type(self)(combined.dictionary)

        # 如果数据版本低于11.0且数据类型为时间间隔类型，则将返回的唯一值转换为指定的数据类型
        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            uniques = cast(ArrowExtensionArray, uniques.astype(self.dtype))
        # 返回索引数组和唯一值数组的元组
        return indices, uniques

    def reshape(self, *args, **kwargs):
        # 抛出未实现的错误，指示不支持对 1D pyarrow.ChunkedArray 进行 reshape 操作
        raise NotImplementedError(
            f"{type(self)} does not support reshape "
            f"as backed by a 1D pyarrow.ChunkedArray."
        )

    def round(self, decimals: int = 0, *args, **kwargs) -> Self:
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect.

        Returns
        -------
        ArrowExtensionArray
            Rounded values of the ArrowExtensionArray.

        See Also
        --------
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        # 返回当前数组中每个值按照指定小数位数进行四舍五入后的结果
        return type(self)(pc.round(self._pa_array, ndigits=decimals))

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if self._hasna:
            # 如果数组中有缺失值，抛出数值错误，因为带有缺失值的数组无法排序
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        if isinstance(value, ExtensionArray):
            # 如果 value 是 ExtensionArray 类型，则强制转换为对象类型，避免基类 searchsorted 函数的慢速类型转换
            value = value.astype(object)
        # 如果数据类型为 ArrowDtype 类型，则进入条件判断
        dtype = None
        if isinstance(self.dtype, ArrowDtype):
            # 获取 PyArrow 中的数据类型
            pa_dtype = self.dtype.pyarrow_dtype
            if (
                pa.types.is_timestamp(pa_dtype) or pa.types.is_duration(pa_dtype)
            ) and pa_dtype.unit == "ns":
                # 当 PyArrow 数据类型为时间戳或持续时间，并且单位为纳秒时，由于 numpy 类型解析为纳秒时会导致错误，因此设置 dtype 为对象类型
                dtype = object
        # 调用 to_numpy 方法将自身转换为 numpy 数组，并调用其 searchsorted 方法进行搜索
        return self.to_numpy(dtype=dtype).searchsorted(value, side=side, sorter=sorter)

    def take(
        self,
        indices: TakeIndexer,
        allow_fill: bool = False,
        fill_value: Any = None,
    def _maybe_convert_datelike_array(self):
        """Maybe convert to a datelike array."""
        # 获取 PyArrow 数组的数据类型
        pa_type = self._pa_array.type
        # 如果数据类型为时间戳，则调用 _to_datetimearray 方法进行转换为 DatetimeArray
        if pa.types.is_timestamp(pa_type):
            return self._to_datetimearray()
        # 如果数据类型为持续时间，则调用 _to_timedeltaarray 方法进行转换为 TimedeltaArray
        elif pa.types.is_duration(pa_type):
            return self._to_timedeltaarray()
        # 如果不是时间戳或持续时间类型，则返回自身
        return self

    def _to_datetimearray(self) -> DatetimeArray:
        """Convert a pyarrow timestamp typed array to a DatetimeArray."""
        # 导入所需的 DatetimeArray 和 tz_to_dtype 函数
        from pandas.core.arrays.datetimes import (
            DatetimeArray,
            tz_to_dtype,
        )

        # 获取 PyArrow 数组的数据类型
        pa_type = self._pa_array.type
        assert pa.types.is_timestamp(pa_type)
        # 创建 numpy 数据类型，根据 PyArrow 数据类型的单位来确定
        np_dtype = np.dtype(f"M8[{pa_type.unit}]")
        # 根据时区和单位获取 pandas 的数据类型
        dtype = tz_to_dtype(pa_type.tz, pa_type.unit)
        # 将 PyArrow 数组转换为 numpy 数组，并将其转换为指定的 numpy 数据类型
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype)
        # 使用 DatetimeArray 类的 _simple_new 方法创建新的 DatetimeArray 对象
        return DatetimeArray._simple_new(np_array, dtype=dtype)

    def _to_timedeltaarray(self) -> TimedeltaArray:
        """Convert a pyarrow duration typed array to a TimedeltaArray."""
        # 导入 TimedeltaArray 类
        from pandas.core.arrays.timedeltas import TimedeltaArray

        # 获取 PyArrow 数组的数据类型
        pa_type = self._pa_array.type
        assert pa.types.is_duration(pa_type)
        # 创建 numpy 数据类型，根据 PyArrow 数据类型的单位来确定
        np_dtype = np.dtype(f"m8[{pa_type.unit}]")
        # 将 PyArrow 数组转换为 numpy 数组，并将其转换为指定的 numpy 数据类型
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype)
        # 使用 TimedeltaArray 类的 _simple_new 方法创建新的 TimedeltaArray 对象
        return TimedeltaArray._simple_new(np_array, dtype=np_dtype)

    def _values_for_json(self) -> np.ndarray:
        # 如果数据类型是数值类型，则将自身转换为对象类型的 numpy 数组返回
        if is_numeric_dtype(self.dtype):
            return np.asarray(self, dtype=object)
        # 否则调用父类的 _values_for_json 方法
        return super()._values_for_json()

    @doc(ExtensionArray.to_numpy)
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        original_na_value = na_value  # 保存原始的缺失值标记
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, self._hasna)  # 推断出适合的 NumPy 数据类型和缺失值标记
        pa_type = self._pa_array.type  # 获取底层 PyArrow 数组的类型

        if not self._hasna or isna(na_value) or pa.types.is_null(pa_type):
            data = self  # 如果没有缺失值或者缺失值标记为 None，直接使用当前数据
        else:
            data = self.fillna(na_value)  # 否则，用指定的缺失值填充数据
            copy = False  # 设定复制标记为 False

        if pa.types.is_timestamp(pa_type) or pa.types.is_duration(pa_type):
            # 如果数据类型是时间戳或持续时间
            if dtype != object and na_value is self.dtype.na_value:
                na_value = lib.no_default  # 如果数据类型不是 object 并且缺失值与默认的 NA 值相同，则将缺失值置为 lib.no_default
            result = data._maybe_convert_datelike_array().to_numpy(
                dtype=dtype, na_value=na_value
            )  # 将数据转换为 NumPy 数组，处理日期类型的缺失值
        elif pa.types.is_time(pa_type) or pa.types.is_date(pa_type):
            # 如果数据类型是时间或日期
            result = np.array(list(data), dtype=dtype)  # 将数据转换为包含 Python datetime.time 对象的 ndarray
            if data._hasna:
                result[data.isna()] = na_value  # 如果数据含有缺失值，将缺失值替换为指定的 na_value
        elif pa.types.is_null(pa_type):
            # 如果数据类型是空值类型
            if dtype is not None and isna(na_value):
                na_value = None  # 如果指定了数据类型且缺失值标记为 None，则将缺失值置为 None
            result = np.full(len(data), fill_value=na_value, dtype=dtype)  # 创建一个全为指定值的 NumPy 数组
        elif not data._hasna or (
            pa.types.is_floating(pa_type)
            and (
                na_value is np.nan
                or original_na_value is lib.no_default
                and is_float_dtype(dtype)
            )
        ):
            # 如果没有缺失值或数据类型是浮点型并且缺失值为 NaN 或原始缺失值标记为 lib.no_default 且数据类型是浮点型
            result = data._pa_array.to_numpy()  # 将 PyArrow 数组转换为 NumPy 数组
            if dtype is not None:
                result = result.astype(dtype, copy=False)  # 如果指定了数据类型，则将结果转换为指定类型
            if copy:
                result = result.copy()  # 如果需要复制，则进行复制操作
        else:
            # 否则，处理其他情况
            if dtype is None:
                empty = pa.array([], type=pa_type).to_numpy(zero_copy_only=False)  # 创建一个空 PyArrow 数组并转换为 NumPy 数组
                if can_hold_element(empty, na_value):
                    dtype = empty.dtype  # 如果能容纳空数组和缺失值，则使用空数组的数据类型
                else:
                    dtype = np.object_  # 否则，使用对象类型
            result = np.empty(len(data), dtype=dtype)  # 创建一个空的 NumPy 数组
            mask = data.isna()  # 获取数据的缺失值掩码
            result[mask] = na_value  # 将缺失值位置设为指定的 na_value
            result[~mask] = data[~mask]._pa_array.to_numpy()  # 将非缺失值位置填入对应的 PyArrow 数组转换的 NumPy 数组中
        return result  # 返回处理后的 NumPy 数组

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        if is_numeric_dtype(self.dtype):
            return map_array(self.to_numpy(), mapper, na_action=na_action)  # 如果数据类型是数值类型，则调用 map_array 处理映射
        else:
            return super().map(mapper, na_action)  # 否则，调用父类的 map 方法进行处理

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ):
        # 返回重复值检测的结果，可以选择保留第一个、最后一个或不保留
    ) -> npt.NDArray[np.bool_]:
        pa_type = self._pa_array.type  # 获取 Arrow 扩展数组的数据类型
        if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type):
            values = self.to_numpy(na_value=0)  # 将 Arrow 扩展数组转换为 NumPy 数组，处理浮点数或整数类型数据
        elif pa.types.is_boolean(pa_type):
            values = self.to_numpy(na_value=False)  # 将 Arrow 扩展数组转换为 NumPy 数组，处理布尔类型数据
        elif pa.types.is_temporal(pa_type):
            if pa_type.bit_width == 32:
                pa_type = pa.int32()  # 将时间类型数据转换为 int32 类型
            else:
                pa_type = pa.int64()  # 将时间类型数据转换为 int64 类型
            arr = self.astype(ArrowDtype(pa_type))  # 转换 Arrow 扩展数组的数据类型
            values = arr.to_numpy(na_value=0)  # 将 Arrow 扩展数组转换为 NumPy 数组，处理时间类型数据
        else:
            # 因子化值，避免转换为对象数据类型带来的性能损失
            values = self.factorize()[0]  # 对数据进行因子化处理，以避免性能损失

        mask = self.isna() if self._hasna else None  # 如果数据包含缺失值，则生成缺失值掩码
        return algos.duplicated(values, keep=keep, mask=mask)  # 调用算法库，返回重复值的布尔数组

    def unique(self) -> Self:
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
        pa_type = self._pa_array.type  # 获取 Arrow 扩展数组的数据类型

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # 在 Arrow 版本低于 1.1.0 并且数据类型为持续时间时，将数据类型转换为 int64
            data = self._pa_array.cast(pa.int64())
        else:
            data = self._pa_array  # 否则，保持数据类型不变

        pa_result = pc.unique(data)  # 使用 Pandas 计算唯一值

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            pa_result = pa_result.cast(pa_type)  # 如果是持续时间类型数据，将结果转换回原数据类型

        return type(self)(pa_result)  # 返回包含唯一值的新的 ArrowExtensionArray 对象

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        pa_type = self._pa_array.type  # 获取 Arrow 扩展数组的数据类型
        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # 在 Arrow 版本低于 1.1.0 并且数据类型为持续时间时，将数据类型转换为 int64
            data = self._pa_array.cast(pa.int64())
        else:
            data = self._pa_array  # 否则，保持数据类型不变

        from pandas import (
            Index,
            Series,
        )

        vc = data.value_counts()  # 计算每个唯一值的计数

        values = vc.field(0)  # 获取计数结果中的唯一值数组
        counts = vc.field(1)  # 获取计数结果中的计数数组
        if dropna and data.null_count > 0:
            mask = values.is_valid()  # 生成有效值的掩码
            values = values.filter(mask)  # 过滤掉无效值
            counts = counts.filter(mask)  # 过滤掉无效值

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            values = values.cast(pa_type)  # 如果是持续时间类型数据，将唯一值数组转换回原数据类型

        counts = ArrowExtensionArray(counts)  # 使用 Arrow 扩展数组创建计数数组

        index = Index(type(self)(values))  # 使用 Arrow 扩展数组创建索引

        return Series(counts, index=index, name="count", copy=False)  # 返回包含计数结果的 Series 对象

    @classmethod
    def _concat_same_type(cls, to_concat) -> Self:
        """
        Concatenate multiple ArrowExtensionArrays.

        Parameters
        ----------
        to_concat : sequence of ArrowExtensionArrays
            Sequence containing ArrowExtensionArray objects to concatenate.

        Returns
        -------
        ArrowExtensionArray
            Concatenated ArrowExtensionArray object.
        """
        # Iterate through each ArrowExtensionArray in to_concat and retrieve all data chunks
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        
        if to_concat[0].dtype == "string":
            # When dtype is 'string', use large_string() as pyarrow data type
            # since StringDtype has no attribute pyarrow_dtype
            pa_dtype = pa.large_string()
        else:
            # Use the pyarrow_dtype attribute of the first ArrowExtensionArray's dtype
            pa_dtype = to_concat[0].dtype.pyarrow_dtype
        
        # Create a new pyarrow ChunkedArray from the collected data chunks
        arr = pa.chunked_array(chunks, type=pa_dtype)
        
        # Return a new instance of the class (cls) initialized with the resulting ChunkedArray
        return cls(arr)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function specifying the type of accumulation operation.
            Supported values include 'cummin', 'cummax', 'cumsum', 'cumprod'.
        skipna : bool, default True
            Whether to skip NA/null values during accumulation.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, no supported kwargs exist.

        Returns
        -------
        array
            Resulting ExtensionArray from the accumulation operation.

        Raises
        ------
        NotImplementedError
            If the subclass does not define the requested accumulation operation.
        """
        # Map name to the corresponding PyArrow method name
        pyarrow_name = {
            "cummax": "cumulative_max",
            "cummin": "cumulative_min",
            "cumprod": "cumulative_prod_checked",
            "cumsum": "cumulative_sum_checked",
        }.get(name, name)
        
        # Retrieve the PyArrow method from the pa.compute module
        pyarrow_meth = getattr(pc, pyarrow_name, None)
        
        if pyarrow_meth is None:
            # If the PyArrow method is not found, call the superclass's _accumulate method
            return super()._accumulate(name, skipna=skipna, **kwargs)

        # Obtain the underlying PyArrow array from self
        data_to_accum = self._pa_array

        # Determine the data type of the PyArrow array
        pa_dtype = data_to_accum.type

        # Check if conversion to integer types is necessary based on the accumulation operation
        convert_to_int = (
            (pa.types.is_temporal(pa_dtype) and name in ["cummax", "cummin"]) or
            (pa.types.is_duration(pa_dtype) and name == "cumsum")
        )

        # If conversion to int is needed, cast the PyArrow array to int32 or int64
        if convert_to_int:
            if pa_dtype.bit_width == 32:
                data_to_accum = data_to_accum.cast(pa.int32())
            else:
                data_to_accum = data_to_accum.cast(pa.int64())

        # Execute the PyArrow accumulation method on the data, handling nulls as specified
        result = pyarrow_meth(data_to_accum, skip_nulls=skipna, **kwargs)

        # If conversion to int was performed earlier, cast the result back to the original data type
        if convert_to_int:
            result = result.cast(pa_dtype)

        # Return a new instance of the current object type initialized with the resulting array
        return type(self)(result)

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
        """
        Reduce the ExtensionArray by applying the specified reduction operation.

        Parameters
        ----------
        name : str
            Name of the reduction operation. Supported values include 'sum', 'min', 'max', 'mean'.
        skipna : bool, default True
            Whether to skip NA/null values during reduction.
        keepdims : bool, default False
            Whether to retain reduced dimensions.
        **kwargs
            Additional keyword arguments specific to the reduction operation.

        Returns
        -------
        array
            Resulting ExtensionArray after reduction.

        Raises
        ------
        NotImplementedError
            If the subclass does not define the requested reduction operation.
        """
        # Map name to the corresponding PyArrow method name
        pyarrow_name = {
            "sum": "sum",
            "min": "min",
            "max": "max",
            "mean": "mean",
        }.get(name, name)
        
        # Retrieve the PyArrow method from the pa.compute module
        pyarrow_meth = getattr(pc, pyarrow_name, None)
        
        if pyarrow_meth is None:
            # If the PyArrow method is not found, raise NotImplementedError
            raise NotImplementedError(f"{self.__class__.__name__} does not define {name}")
        
        # Obtain the underlying PyArrow array from self
        data_to_reduce = self._pa_array
        
        # Execute the PyArrow reduction method on the data, handling nulls as specified
        result = pyarrow_meth(data_to_reduce, skip_nulls=skipna, keepdims=keepdims, **kwargs)
        
        # Return a new instance of the current object type initialized with the resulting array
        return type(self)(result)
    ):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
            操作的名称，支持的取值包括：
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }。
        skipna : bool, default True
            If True, skip NaN values.
            如果为True，则跳过NaN值。
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.
            传递给约简函数的额外关键字参数。目前只支持 `ddof` 这个关键字参数。

        Returns
        -------
        scalar
        返回标量结果。

        Raises
        ------
        TypeError : subclass does not define reductions
        TypeError：子类没有定义约简操作。

        """
        result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if isinstance(result, pa.Array):
            return type(self)(result)
        else:
            return result

    def _reduce_calc(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        """
        Perform reduction calculation using pyarrow.

        Parameters
        ----------
        name : str
            Name of the reduction operation.
            约简操作的名称。
        skipna : bool, default True
            Whether to exclude NA/null values.
            是否排除NA/null值。
        keepdims : bool, default False
            Whether to retain dimensionality after reduction.
            约简后是否保留维度。
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.
            传递给约简函数的额外关键字参数。目前只支持 `ddof` 这个关键字参数。

        Returns
        -------
        scalar
        返回标量结果。

        """
        pa_result = self._reduce_pyarrow(name, skipna=skipna, **kwargs)

        if keepdims:
            if isinstance(pa_result, pa.Scalar):
                result = pa.array([pa_result.as_py()], type=pa_result.type)
            else:
                result = pa.array(
                    [pa_result],
                    type=to_pyarrow_type(infer_dtype_from_scalar(pa_result)[0]),
                )
            return result

        if pc.is_null(pa_result).as_py():
            return self.dtype.na_value
        elif isinstance(pa_result, pa.Scalar):
            return pa_result.as_py()
        else:
            return pa_result

    def _explode(self):
        """
        See Series.explode.__doc__.
        查看 Series.explode.__doc__。
        """
        # child class explode method supports only list types; return
        # default implementation for non list types.
        # 子类的explode方法只支持列表类型；对于非列表类型，返回默认实现。
        if not pa.types.is_list(self.dtype.pyarrow_dtype):
            return super()._explode()
        values = self
        counts = pa.compute.list_value_length(values._pa_array)
        counts = counts.fill_null(1).to_numpy()
        fill_value = pa.scalar([None], type=self._pa_array.type)
        mask = counts == 0
        if mask.any():
            values = values.copy()
            values[mask] = fill_value
            counts = counts.copy()
            counts[mask] = 1
        values = values.fillna(fill_value)
        values = type(self)(pa.compute.list_flatten(values._pa_array))
        return values, counts
    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # GH50085: unwrap 1D indexers
        # 如果 key 是元组并且长度为 1，则解包为单个元素
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        # 检查并规范化 key，确保其为合法的数组索引
        key = check_array_indexer(self, key)

        # 处理要设置的值，可能进行类型转换或适配
        value = self._maybe_convert_setitem_value(value)

        # 如果 key 表示一个空切片，使用快速路径 (GH50248)
        if com.is_null_slice(key):
            data = self._if_else(True, value, self._pa_array)

        # 如果 key 是整数，也使用快速路径
        elif is_integer(key):
            key = cast(int, key)
            n = len(self)
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {n}"
                )
            # 根据值的类型，生成新的 ChunkedArray
            if isinstance(value, pa.Scalar):
                value = value.as_py()
            elif is_list_like(value):
                raise ValueError("Length of indexer and values mismatch")
            chunks = [
                *self._pa_array[:key].chunks,
                pa.array([value], type=self._pa_array.type, from_pandas=True),
                *self._pa_array[key + 1 :].chunks,
            ]
            data = pa.chunked_array(chunks).combine_chunks()

        # 如果 key 是布尔型数组
        elif is_bool_dtype(key):
            key = np.asarray(key, dtype=np.bool_)
            # 使用掩码替换数据中的部分值
            data = self._replace_with_mask(self._pa_array, key, value)

        # 如果 value 是标量或者单个标量值
        elif is_scalar(value) or isinstance(value, pa.Scalar):
            # 根据布尔掩码替换数据中的部分值
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[key] = True
            data = self._if_else(mask, value, self._pa_array)

        else:
            # 否则，根据索引生成新的 ChunkedArray，并根据给定值进行替换
            indices = np.arange(len(self))[key]
            if len(indices) != len(value):
                raise ValueError("Length of indexer and values mismatch")
            if len(indices) == 0:
                return
            # 解决重复键导致的错误赋值问题
            _, argsort = np.unique(indices, return_index=True)
            indices = indices[argsort]
            value = value.take(argsort)
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[indices] = True
            data = self._replace_with_mask(self._pa_array, mask, value)

        # 如果生成的数据是一个单一的 Array，则转换成 ChunkedArray
        if isinstance(data, pa.Array):
            data = pa.chunked_array([data])

        # 更新对象的内部数据
        self._pa_array = data
    ):
        # 如果 axis 不等于 0，调用父类的 _rank 方法进行排名计算
        if axis != 0:
            ranked = super()._rank(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            # 保持数据类型与以下实现一致
            if method == "average" or pct:
                pa_type = pa.float64()
            else:
                pa_type = pa.uint64()
            # 使用 pyarrow 创建数组对象，并从 pandas 对象转换而来
            result = pa.array(ranked, type=pa_type, from_pandas=True)
            return result

        # 否则，获取组合后的 pyarrow 数组数据
        data = self._pa_array.combine_chunks()
        # 根据 ascending 的值选择排序键
        sort_keys = "ascending" if ascending else "descending"
        # 根据 na_option 的值确定空值放置位置
        null_placement = "at_start" if na_option == "top" else "at_end"
        # 根据 method 的值选择 tiebreaker
        tiebreaker = "min" if method == "average" else method

        # 调用 pyarrow 的 rank 函数进行排名计算
        result = pc.rank(
            data,
            sort_keys=sort_keys,
            null_placement=null_placement,
            tiebreaker=tiebreaker,
        )

        # 如果 na_option 为 "keep"，处理空值
        if na_option == "keep":
            mask = pc.is_null(self._pa_array)
            null = pa.scalar(None, type=result.type)
            result = pc.if_else(mask, null, result)

        # 如果 method 为 "average"，计算平均排名
        if method == "average":
            result_max = pc.rank(
                data,
                sort_keys=sort_keys,
                null_placement=null_placement,
                tiebreaker="max",
            )
            result_max = result_max.cast(pa.float64())
            result_min = result.cast(pa.float64())
            result = pc.divide(pc.add(result_min, result_max), 2)

        # 如果 pct 为 True，计算百分比排名
        if pct:
            if not pa.types.is_floating(result.type):
                result = result.cast(pa.float64())
            if method == "dense":
                divisor = pc.max(result)
            else:
                divisor = pc.count(result)
            result = pc.divide(result, divisor)

        return result

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Self:
        """
        See Series.rank.__doc__.
        """
        # 返回当前对象的类型，调用 _rank_calc 方法进行排名计算
        return type(self)(
            self._rank_calc(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
            Array of quantiles to compute.
        interpolation: str
            Specifies the interpolation method to use.

        Returns
        -------
        same type as self
            Returns an instance of the same type containing computed quantiles.
        """
        pa_dtype = self._pa_array.type  # 获取 self._pa_array 的数据类型

        data = self._pa_array  # 将 self._pa_array 赋值给 data
        if pa.types.is_temporal(pa_dtype):  # 检查数据类型是否为时间类型
            # 处理 Arrow 库 issue 33769 的情况，需要将数据转换为整数类型再转回
            nbits = pa_dtype.bit_width
            if nbits == 32:
                data = data.cast(pa.int32())  # 将数据转换为 int32 类型
            else:
                data = data.cast(pa.int64())  # 将数据转换为 int64 类型

        # 调用 Arrow 库计算分位数
        result = pc.quantile(data, q=qs, interpolation=interpolation)

        if pa.types.is_temporal(pa_dtype):  # 如果数据类型为时间类型
            if pa.types.is_floating(result.type):  # 如果结果类型为浮点数
                result = pc.floor(result)  # 对结果向下取整
            nbits = pa_dtype.bit_width
            if nbits == 32:
                result = result.cast(pa.int32())  # 将结果转换为 int32 类型
            else:
                result = result.cast(pa.int64())  # 将结果转换为 int64 类型
            result = result.cast(pa_dtype)  # 将结果转换为原始数据类型

        return type(self)(result)  # 返回一个新实例，类型与 self 相同

    def _mode(self, dropna: bool = True) -> Self:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            If True, exclude NA/null values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        pa_type = self._pa_array.type  # 获取 self._pa_array 的数据类型
        if pa.types.is_temporal(pa_type):  # 检查数据类型是否为时间类型
            nbits = pa_type.bit_width
            if nbits == 32:
                data = self._pa_array.cast(pa.int32())  # 将数据转换为 int32 类型
            elif nbits == 64:
                data = self._pa_array.cast(pa.int64())  # 将数据转换为 int64 类型
            else:
                raise NotImplementedError(pa_type)  # 抛出未实现的错误，处理未知类型
        else:
            data = self._pa_array  # 否则，直接使用原始数据

        if dropna:
            data = data.drop_null()  # 如果 dropna 为 True，则排除空值

        res = pc.value_counts(data)  # 使用 Arrow 库计算值的出现次数
        most_common = res.field("values").filter(
            pc.equal(res.field("counts"), pc.max(res.field("counts")))
        )  # 找出出现次数最多的值

        if pa.types.is_temporal(pa_type):  # 如果数据类型为时间类型
            most_common = most_common.cast(pa_type)  # 将结果转换为原始数据类型

        most_common = most_common.take(pc.array_sort_indices(most_common))  # 对结果排序
        return type(self)(most_common)  # 返回一个新实例，类型与 self 相同

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        try:
            value = self._box_pa(value, self._pa_array.type)  # 尝试将值转换为 pyarrow 兼容格式
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{value!s}' for dtype {self.dtype}"  # 处理转换错误的异常
            raise TypeError(msg) from err
        return value  # 返回转换后的值
    def interpolate(
        self,
        *,
        method: InterpolateOptions,  # 定义插值方法，指定为 InterpolateOptions 类型
        axis: int,  # 指定插值操作的轴
        index,  # 插值操作的索引
        limit,  # 插值的限制条件
        limit_direction,  # 插值的方向限制
        limit_area,  # 插值的区域限制
        copy: bool,  # 是否复制数据进行插值
        **kwargs,  # 其他可选参数
    ) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        # 注意：即使 copy=False，也返回 type(self)
        if not self.dtype._is_numeric:
            raise ValueError("Values must be numeric.")  # 如果数据不是数值类型，则抛出数值错误异常

        if (
            not pa_version_under13p0  # 如果不是 Pandas 版本小于 13.0
            and method == "linear"  # 并且插值方法是线性插值
            and limit_area is None  # 并且没有限制区域
            and limit is None  # 并且没有限制条件
            and limit_direction == "forward"  # 并且插值方向是向前
        ):
            values = self._pa_array.combine_chunks()  # 组合数据块
            na_value = pa.array([None], type=values.type)  # 创建一个 NA 值数组
            y_diff_2 = pc.fill_null_backward(pc.pairwise_diff_checked(values, period=2))  # 对数据进行差分操作
            prev_values = pa.concat_arrays([na_value, values[:-2], na_value])  # 拼接处理后的数据
            interps = pc.add_checked(prev_values, pc.divide_checked(y_diff_2, 2))  # 执行插值计算
            return type(self)(pc.coalesce(self._pa_array, interps))  # 返回插值后的新对象

        mask = self.isna()  # 获取缺失值的掩码
        if self.dtype.kind == "f":
            data = self._pa_array.to_numpy()  # 转换为 NumPy 数组
        elif self.dtype.kind in "iu":
            data = self.to_numpy(dtype="f8", na_value=0.0)  # 转换为指定类型的 NumPy 数组
        else:
            raise NotImplementedError(
                f"interpolate is not implemented for dtype={self.dtype}"
            )  # 抛出未实现错误，指定的数据类型不支持插值操作

        missing.interpolate_2d_inplace(
            data,
            method=method,  # 指定插值方法
            axis=0,  # 指定插值的轴
            index=index,  # 指定插值的索引
            limit=limit,  # 指定插值的限制条件
            limit_direction=limit_direction,  # 指定插值的方向限制
            limit_area=limit_area,  # 指定插值的区域限制
            mask=mask,  # 指定缺失值的掩码
            **kwargs,  # 其他可选参数
        )
        return type(self)(self._box_pa_array(pa.array(data, mask=mask)))  # 返回插值后的新对象

    @classmethod
    def _if_else(
        cls,
        cond: npt.NDArray[np.bool_] | bool,  # 定义条件数组或布尔值作为参数
        left: ArrayLike | Scalar,  # 定义左侧和右侧的数据类型
        right: ArrayLike | Scalar,  # 定义右侧和左侧的数据类型
    ) -> pa.Array:
        """
        Choose values based on a condition.

        Analogous to pyarrow.compute.if_else, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        cond : npt.NDArray[np.bool_] or bool
            Boolean array or scalar condition for selecting values.
        left : ArrayLike | Scalar
            Values to choose when condition is True.
        right : ArrayLike | Scalar
            Values to choose when condition is False.

        Returns
        -------
        pa.Array
            Resulting array with values selected based on the condition.
        """
        try:
            return pc.if_else(cond, left, right)
        except pa.ArrowNotImplementedError:
            pass

        def _to_numpy_and_type(value) -> tuple[np.ndarray, pa.DataType | None]:
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                pa_type = value.type
            elif isinstance(value, pa.Scalar):
                pa_type = value.type
                value = value.as_py()
            else:
                pa_type = None
            return np.array(value, dtype=object), pa_type

        left, left_type = _to_numpy_and_type(left)
        right, right_type = _to_numpy_and_type(right)
        pa_type = left_type or right_type
        result = np.where(cond, left, right)
        return pa.array(result, type=pa_type, from_pandas=True)

    @classmethod
    def _replace_with_mask(
        cls,
        values: pa.Array | pa.ChunkedArray,
        mask: npt.NDArray[np.bool_] | bool,
        replacements: ArrayLike | Scalar,
    ) -> pa.Array | pa.ChunkedArray:
        """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
            Array or chunked array to apply replacements on.
        mask : npt.NDArray[np.bool_] or bool
            Boolean array or scalar mask indicating positions to replace.
        replacements : ArrayLike or Scalar
            Replacement value(s) to use.

        Returns
        -------
        pa.Array or pa.ChunkedArray
            Resulting array after applying replacements based on the mask.
        """
        if isinstance(replacements, pa.ChunkedArray):
            # replacements must be array or scalar, not ChunkedArray
            replacements = replacements.combine_chunks()
        if isinstance(values, pa.ChunkedArray) and pa.types.is_boolean(values.type):
            # GH#52059 replace_with_mask segfaults for chunked array
            # https://github.com/apache/arrow/issues/34634
            values = values.combine_chunks()
        try:
            return pc.replace_with_mask(values, mask, replacements)
        except pa.ArrowNotImplementedError:
            pass
        if isinstance(replacements, pa.Array):
            replacements = np.array(replacements, dtype=object)
        elif isinstance(replacements, pa.Scalar):
            replacements = replacements.as_py()
        result = np.array(values, dtype=object)
        result[mask] = replacements
        return pa.array(result, type=values.type, from_pandas=True)

    # ------------------------------------------------------------------
    # GroupBy Methods
    def _to_masked(self):
        # 获取当前数组的PyArrow类型
        pa_dtype = self._pa_array.type

        # 根据数据类型判断缺失值的填充值
        if pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype):
            na_value = 1
        elif pa.types.is_boolean(pa_dtype):
            na_value = True
        else:
            # 如果数据类型不支持，则抛出未实现错误
            raise NotImplementedError

        # 获取与PyArrow类型对应的NumPy数据类型
        dtype = _arrow_dtype_mapping()[pa_dtype]

        # 创建一个表示缺失值掩码的布尔数组
        mask = self.isna()

        # 将当前数组转换为NumPy数组，并使用指定的数据类型和缺失值填充值
        arr = self.to_numpy(dtype=dtype.numpy_dtype, na_value=na_value)

        # 根据数据类型构造一个新的数组对象，将NumPy数组和缺失值掩码传入
        return dtype.construct_array_type()(arr, mask)

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        # 如果数据类型为StringDtype，则调用父类的_groupby_op方法处理
        if isinstance(self.dtype, StringDtype):
            return super()._groupby_op(
                how=how,
                has_dropped_na=has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=ids,
                **kwargs,
            )

        # 否则，根据数据类型转换为适合groupby操作的ExtensionArray
        values: ExtensionArray
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            values = self._to_datetimearray()
        elif pa.types.is_duration(pa_type):
            values = self._to_timedeltaarray()
        else:
            values = self._to_masked()

        # 调用ExtensionArray的_groupby_op方法进行分组操作，并返回结果
        result = values._groupby_op(
            how=how,
            has_dropped_na=has_dropped_na,
            min_count=min_count,
            ngroups=ngroups,
            ids=ids,
            **kwargs,
        )

        # 如果结果是NumPy数组，则直接返回
        if isinstance(result, np.ndarray):
            return result
        
        # 否则，根据当前对象的类型，从序列中构造一个新的对象并返回
        return type(self)._from_sequence(result, copy=False)

    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        """Apply a callable to each element while maintaining the chunking structure."""
        # 对当前对象的每个chunk应用给定的函数，保持chunk的结构，并返回结果列表
        return [
            [
                None if val is None else func(val)
                for val in chunk.to_numpy(zero_copy_only=False)
            ]
            for chunk in self._pa_array.iterchunks()
        ]

    def _str_count(self, pat: str, flags: int = 0) -> Self:
        # 如果flags不为0，则抛出未实现的错误，暂不支持带flags的操作
        if flags:
            raise NotImplementedError(f"count not implemented with {flags=}")
        
        # 调用pc库中的count_substring_regex方法，返回包含匹配数量的对象
        return type(self)(pc.count_substring_regex(self._pa_array, pat))

    def _str_contains(
        self, pat, case: bool = True, flags: int = 0, na=None, regex: bool = True
    ) -> Self:
        # 如果flags不为0，则抛出未实现的错误，暂不支持带flags的操作
        if flags:
            raise NotImplementedError(f"contains not implemented with {flags=}")

        # 根据regex参数选择合适的匹配方法
        if regex:
            pa_contains = pc.match_substring_regex
        else:
            pa_contains = pc.match_substring
        
        # 调用合适的匹配方法，返回匹配结果的对象
        result = pa_contains(self._pa_array, pat, ignore_case=not case)
        
        # 如果na参数不是None，则使用fill_null方法填充空值
        if not isna(na):
            result = result.fill_null(na)
        
        # 根据当前对象的类型，返回包含结果的新对象
        return type(self)(result)
    # 定义一个方法用于检查字符串或者字符串元组是否以指定模式开始
    def _str_startswith(self, pat: str | tuple[str, ...], na=None) -> Self:
        # 如果 pat 是字符串，使用 polars 的 starts_with 方法检查数组中元素是否以该模式开始
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        else:
            # 如果 pat 是空元组，使用 pd.StringDtype() 处理缺失值为 null，有效值为 false
            if len(pat) == 0:
                result = pc.if_else(pc.is_null(self._pa_array), None, False)
            else:
                # 否则，对于元组中每个模式，检查数组中元素是否以其中任意一个模式开始
                result = pc.starts_with(self._pa_array, pattern=pat[0])

                for p in pat[1:]:
                    result = pc.or_(result, pc.starts_with(self._pa_array, pattern=p))
        # 如果 na 参数非空，则用给定值填充结果中的 null 值
        if not isna(na):
            result = result.fill_null(na)
        # 返回与 self 类型相同的对象，传递检查结果
        return type(self)(result)

    # 定义一个方法用于检查字符串或者字符串元组是否以指定模式结束
    def _str_endswith(self, pat: str | tuple[str, ...], na=None) -> Self:
        # 如果 pat 是字符串，使用 polars 的 ends_with 方法检查数组中元素是否以该模式结束
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        else:
            # 如果 pat 是空元组，使用 pd.StringDtype() 处理缺失值为 null，有效值为 false
            if len(pat) == 0:
                result = pc.if_else(pc.is_null(self._pa_array), None, False)
            else:
                # 否则，对于元组中每个模式，检查数组中元素是否以其中任意一个模式结束
                result = pc.ends_with(self._pa_array, pattern=pat[0])

                for p in pat[1:]:
                    result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
        # 如果 na 参数非空，则用给定值填充结果中的 null 值
        if not isna(na):
            result = result.fill_null(na)
        # 返回与 self 类型相同的对象，传递检查结果
        return type(self)(result)

    # 定义一个方法用于在字符串中进行替换操作
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self:
        # 如果模式是正则表达式对象，替换函数是不可调用的，或者禁用了大小写敏感性或标志位
        if isinstance(pat, re.Pattern) or callable(repl) or not case or flags:
            raise NotImplementedError(
                "replace is not supported with a re.Pattern, callable repl, "
                "case=False, or flags!=0"
            )

        # 根据 regex 参数选择合适的替换函数，并处理最大替换次数
        func = pc.replace_substring_regex if regex else pc.replace_substring
        pa_max_replacements = None if n < 0 else n
        # 执行替换操作并返回结果
        result = func(
            self._pa_array,
            pattern=pat,
            replacement=repl,
            max_replacements=pa_max_replacements,
        )
        # 返回与 self 类型相同的对象，传递替换后的结果
        return type(self)(result)

    # 定义一个方法用于重复字符串数组中的每个元素
    def _str_repeat(self, repeats: int | Sequence[int]) -> Self:
        # 如果 repeats 不是整数类型，抛出未实现错误
        if not isinstance(repeats, int):
            raise NotImplementedError(
                f"repeat is not implemented when repeats is {type(repeats).__name__}"
            )
        # 使用 polars 的 binary_repeat 方法重复数组中的每个元素，并返回结果
        return type(self)(pc.binary_repeat(self._pa_array, repeats))

    # 定义一个方法用于检查字符串是否以指定模式匹配
    def _str_match(
        self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = None
    ) -> Self:
        # 如果模式不以 "^" 开始，添加 "^" 前缀以确保从开头匹配
        if not pat.startswith("^"):
            pat = f"^{pat}"
        # 调用 _str_contains 方法执行基于正则表达式的包含检查，并传递相关参数
        return self._str_contains(pat, case, flags, na, regex=True)
    def _str_fullmatch(
        self, pat, case: bool = True, flags: int = 0, na: Scalar | None = None
    ) -> Self:
        # 如果模式字符串不是以 "$" 结尾或者以 "\$" 结尾，则添加 "$" 结尾
        if not pat.endswith("$") or pat.endswith("\\$"):
            pat = f"{pat}$"
        # 调用 _str_match 方法，进行全匹配操作，返回匹配结果
        return self._str_match(pat, case, flags, na)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None) -> Self:
        if (start == 0 or start is None) and end is None:
            # 在整个字符串中查找子字符串 sub 的位置
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == "":
                # GH 56792
                # 对每个元素应用查找操作，并返回新的对象类型
                result = self._apply_elementwise(lambda val: val.find(sub, start, end))
                return type(self)(pa.chunked_array(result))
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                # 计算从末尾开始的偏移量
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0, start_offset)
            else:
                start_offset = start
            # 切片操作，获取指定范围内的子字符串
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            # 在切片中查找子字符串的位置
            result = pc.find_substring(slices, sub)
            # 找到子字符串时，计算其在原始字符串中的位置
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        # 返回结果，生成相同类型的对象
        return type(self)(result)

    def _str_join(self, sep: str) -> Self:
        if pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            # 对每个元素应用 list 函数，返回结果作为列表的 chunked_array
            result = self._apply_elementwise(list)
            result = pa.chunked_array(result, type=pa.list_(pa.string()))
        else:
            result = self._pa_array
        # 使用 sep 连接数组中的字符串，并返回结果
        return type(self)(pc.binary_join(result, sep))

    def _str_partition(self, sep: str, expand: bool) -> Self:
        # 对每个元素应用 partition 方法，返回结果作为 chunked_array
        predicate = lambda val: val.partition(sep)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_rpartition(self, sep: str, expand: bool) -> Self:
        # 对每个元素应用 rpartition 方法，返回结果作为 chunked_array
        predicate = lambda val: val.rpartition(sep)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        if start is None:
            start = 0
        if step is None:
            step = 1
        # 对字符串进行切片操作，返回切片后的结果
        return type(self)(
            pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
        )

    def _str_isalnum(self) -> Self:
        # 判断字符串是否只包含字母和数字，并返回结果
        return type(self)(pc.utf8_is_alnum(self._pa_array))

    def _str_isalpha(self) -> Self:
        # 判断字符串是否只包含字母，并返回结果
        return type(self)(pc.utf8_is_alpha(self._pa_array))

    def _str_isdecimal(self) -> Self:
        # 判断字符串是否只包含十进制数字，并返回结果
        return type(self)(pc.utf8_is_decimal(self._pa_array))

    def _str_isdigit(self) -> Self:
        # 判断字符串是否只包含数字，并返回结果
        return type(self)(pc.utf8_is_digit(self._pa_array))
    # 返回一个新的实例，其字符串内容是通过 utf8_is_lower 函数处理后的结果
    def _str_islower(self) -> Self:
        return type(self)(pc.utf8_is_lower(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_is_numeric 函数处理后的结果
    def _str_isnumeric(self) -> Self:
        return type(self)(pc.utf8_is_numeric(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_is_space 函数处理后的结果
    def _str_isspace(self) -> Self:
        return type(self)(pc.utf8_is_space(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_is_title 函数处理后的结果
    def _str_istitle(self) -> Self:
        return type(self)(pc.utf8_is_title(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_is_upper 函数处理后的结果
    def _str_isupper(self) -> Self:
        return type(self)(pc.utf8_is_upper(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_length 函数处理后的结果
    def _str_len(self) -> Self:
        return type(self)(pc.utf8_length(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_lower 函数处理后的结果
    def _str_lower(self) -> Self:
        return type(self)(pc.utf8_lower(self._pa_array))

    # 返回一个新的实例，其字符串内容是通过 utf8_upper 函数处理后的结果
    def _str_upper(self) -> Self:
        return type(self)(pc.utf8_upper(self._pa_array))

    # 返回一个新的实例，其字符串内容经过去除首尾空白字符的处理
    def _str_strip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return type(self)(result)

    # 返回一个新的实例，其字符串内容经过去除左侧空白字符的处理
    def _str_lstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    # 返回一个新的实例，其字符串内容经过去除右侧空白字符的处理
    def _str_rstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    # 根据指定前缀移除字符串内容的开头部分，返回新的实例
    def _str_removeprefix(self, prefix: str):
        if not pa_version_under13p0:
            starts_with = pc.starts_with(self._pa_array, pattern=prefix)
            removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
            result = pc.if_else(starts_with, removed, self._pa_array)
            return type(self)(result)
        predicate = lambda val: val.removeprefix(prefix)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    # 返回一个新的实例，其字符串内容经过 casefold 处理
    def _str_casefold(self) -> Self:
        predicate = lambda val: val.casefold()
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    # 返回一个新的实例，其字符串内容通过指定编码方式编码后的结果
    def _str_encode(self, encoding: str, errors: str = "strict") -> Self:
        predicate = lambda val: val.encode(encoding, errors)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))
    # 定义一个方法用于从字符串数组中提取匹配正则表达式模式的子串
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        # 如果传入了 flags 参数，则抛出未实现的错误
        if flags:
            raise NotImplementedError("Only flags=0 is implemented.")
        # 编译正则表达式并获取其中的命名组名列表
        groups = re.compile(pat).groupindex.keys()
        # 如果没有命名组名，则抛出值错误
        if len(groups) == 0:
            raise ValueError(f"{pat=} must contain a symbolic group name.")
        # 调用外部库函数，从 self._pa_array 中提取与模式 pat 匹配的结果
        result = pc.extract_regex(self._pa_array, pat)
        # 如果 expand 参数为 True，返回一个字典，键为命名组名，值为相应提取的结果
        if expand:
            return {
                col: type(self)(pc.struct_field(result, [i]))
                for col, i in zip(groups, range(result.type.num_fields))
            }
        # 如果 expand 参数为 False，返回提取结果中的第一个元素
        else:
            return type(self)(pc.struct_field(result, [0]))

    # 定义一个方法用于在字符串数组中查找所有与给定模式匹配的子串，并返回结果对象
    def _str_findall(self, pat: str, flags: int = 0) -> Self:
        # 编译正则表达式
        regex = re.compile(pat, flags=flags)
        # 定义谓词函数，返回每个元素中与正则表达式匹配的所有子串
        predicate = lambda val: regex.findall(val)
        # 对数组中的每个元素应用谓词函数，得到结果数组
        result = self._apply_elementwise(predicate)
        # 将结果封装为当前类的对象并返回
        return type(self)(pa.chunked_array(result))

    # 定义一个方法用于在字符串数组中根据分隔符 sep 拆分每个元素，并返回稀疏矩阵形式的结果及唯一的分隔符值列表
    def _str_get_dummies(self, sep: str = "|"):
        # 使用外部库函数，将字符串数组中的每个元素按照 sep 分隔
        split = pc.split_pattern(self._pa_array, sep)
        # 将分隔后的多维数组扁平化
        flattened_values = pc.list_flatten(split)
        # 获取扁平化后的唯一值
        uniques = flattened_values.unique()
        # 对唯一值进行排序
        uniques_sorted = uniques.take(pa.compute.array_sort_indices(uniques))
        # 计算每个子数组的长度并转换为 NumPy 数组
        lengths = pc.list_value_length(split).fill_null(0).to_numpy()
        # 获取字符串数组的行数和唯一值的列数
        n_rows = len(self)
        n_cols = len(uniques)
        # 计算每个值在扁平化数组中的索引
        indices = pc.index_in(flattened_values, uniques_sorted).to_numpy()
        indices = indices + np.arange(n_rows).repeat(lengths) * n_cols
        # 创建一个全零数组，用于表示哑变量
        dummies = np.zeros(n_rows * n_cols, dtype=np.bool_)
        dummies[indices] = True
        # 将哑变量数组重新形状为二维数组
        dummies = dummies.reshape((n_rows, n_cols))
        # 将结果封装为当前类的对象并返回，同时返回排序后的唯一值列表
        result = type(self)(pa.array(list(dummies)))
        return result, uniques_sorted.to_pylist()

    # 定义一个方法用于在字符串数组中查找子串 sub 的第一次出现的索引，并返回结果对象
    def _str_index(self, sub: str, start: int = 0, end: int | None = None) -> Self:
        # 定义谓词函数，返回子串在每个元素中第一次出现的索引
        predicate = lambda val: val.index(sub, start, end)
        # 对数组中的每个元素应用谓词函数，得到结果数组
        result = self._apply_elementwise(predicate)
        # 将结果封装为当前类的对象并返回
        return type(self)(pa.chunked_array(result))

    # 定义一个方法用于在字符串数组中查找子串 sub 最后一次出现的索引，并返回结果对象
    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None) -> Self:
        # 定义谓词函数，返回子串在每个元素中最后一次出现的索引
        predicate = lambda val: val.rindex(sub, start, end)
        # 对数组中的每个元素应用谓词函数，得到结果数组
        result = self._apply_elementwise(predicate)
        # 将结果封装为当前类的对象并返回
        return type(self)(pa.chunked_array(result))

    # 定义一个方法用于对字符串数组中的每个元素应用 Unicode 规范化，并返回结果对象
    def _str_normalize(self, form: str) -> Self:
        # 定义谓词函数，对每个元素应用指定的 Unicode 规范化形式
        predicate = lambda val: unicodedata.normalize(form, val)
        # 对数组中的每个元素应用谓词函数，得到结果数组
        result = self._apply_elementwise(predicate)
        # 将结果封装为当前类的对象并返回
        return type(self)(pa.chunked_array(result))

    # 定义一个方法用于在字符串数组中查找子串 sub 最后一次出现的索引，并返回结果对象
    def _str_rfind(self, sub: str, start: int = 0, end=None) -> Self:
        # 定义谓词函数，返回子串在每个元素中从后往前查找时第一次出现的索引
        predicate = lambda val: val.rfind(sub, start, end)
        # 对数组中的每个元素应用谓词函数，得到结果数组
        result = self._apply_elementwise(predicate)
        # 将结果封装为当前类的对象并返回
        return type(self)(pa.chunked_array(result))

    # 定义一个方法用于在字符串数组中根据指定的分隔符 pat 拆分每个元素，并返回结果对象
    def _str_split(
        self,
        pat: str | None = None,
        n: int | None = -1,
        expand: bool = False,
        regex: bool | None = None,
    ) -> Self:
        if n in {-1, 0}:
            n = None
        # 如果 n 的值为 -1 或 0，将其设为 None
        if pat is None:
            # 如果未指定分隔模式，则使用默认的空白字符分隔函数
            split_func = pc.utf8_split_whitespace
        elif regex:
            # 如果指定了使用正则表达式分隔，则使用指定模式的正则分隔函数
            split_func = functools.partial(pc.split_pattern_regex, pattern=pat)
        else:
            # 否则，使用指定模式的普通分隔函数
            split_func = functools.partial(pc.split_pattern, pattern=pat)
        # 返回一个新的对象，应用上述分隔函数对当前对象进行分割操作，最大分割次数为 n
        return type(self)(split_func(self._pa_array, max_splits=n))

    def _str_rsplit(self, pat: str | None = None, n: int | None = -1) -> Self:
        if n in {-1, 0}:
            n = None
        # 如果 n 的值为 -1 或 0，将其设为 None
        if pat is None:
            # 如果未指定分隔模式，则返回一个新对象，应用默认的空白字符反向分隔操作
            return type(self)(
                pc.utf8_split_whitespace(self._pa_array, max_splits=n, reverse=True)
            )
        # 否则，返回一个新对象，应用指定模式的反向分隔操作
        return type(self)(
            pc.split_pattern(self._pa_array, pat, max_splits=n, reverse=True)
        )

    def _str_translate(self, table: dict[int, str]) -> Self:
        # 定义一个函数，使用给定的映射表对每个字符串元素进行转换
        predicate = lambda val: val.translate(table)
        # 对当前对象的每个元素应用上述转换函数，并构造一个新对象
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_wrap(self, width: int, **kwargs) -> Self:
        kwargs["width"] = width
        # 创建一个文本包装对象，使用指定的宽度和其他关键字参数
        tw = textwrap.TextWrapper(**kwargs)
        # 定义一个函数，将每个字符串按指定宽度包装成多行文本
        predicate = lambda val: "\n".join(tw.wrap(val))
        # 对当前对象的每个元素应用上述包装函数，并构造一个新对象
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    @property
    def _dt_days(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的天数部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.days,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_hours(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的小时部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.hours,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_minutes(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的分钟部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.minutes,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_seconds(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的秒数部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.seconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_milliseconds(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的毫秒部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.milliseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_microseconds(self) -> Self:
        # 返回一个新对象，包含当前时间间隔数组中的微秒部分
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.microseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )
    def _dt_nanoseconds(self) -> Self:
        # 返回一个新的对象，其中包含以纳秒为单位的时间间隔数组
        return type(self)(
            pa.array(
                self._to_timedeltaarray().components.nanoseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    def _dt_to_pytimedelta(self) -> np.ndarray:
        # 获取内部PyArrow数组的Python列表表示
        data = self._pa_array.to_pylist()
        # 如果时间间隔单位为纳秒，则转换为Python datetime.timedelta对象
        if self._dtype.pyarrow_dtype.unit == "ns":
            data = [None if ts is None else ts.to_pytimedelta() for ts in data]
        return np.array(data, dtype=object)

    def _dt_total_seconds(self) -> Self:
        # 返回一个新的对象，其中包含以秒为单位的总秒数的数组
        return type(self)(
            pa.array(self._to_timedeltaarray().total_seconds(), from_pandas=True)
        )

    def _dt_as_unit(self, unit: str) -> Self:
        if pa.types.is_date(self.dtype.pyarrow_dtype):
            # 对于日期类型，不支持单位转换操作
            raise NotImplementedError("as_unit not implemented for date types")
        pd_array = self._maybe_convert_datelike_array()
        # 根据指定的单位将数组转换为新的对象
        # 不直接转换_pa_array，以遵循pandas的单位转换规则
        return type(self)(pa.array(pd_array.as_unit(unit), from_pandas=True))

    @property
    def _dt_year(self) -> Self:
        # 返回一个新的对象，其中包含年份信息的数组
        return type(self)(pc.year(self._pa_array))

    @property
    def _dt_day(self) -> Self:
        # 返回一个新的对象，其中包含天数信息的数组
        return type(self)(pc.day(self._pa_array))

    @property
    def _dt_day_of_week(self) -> Self:
        # 返回一个新的对象，其中包含星期几信息的数组
        return type(self)(pc.day_of_week(self._pa_array))

    _dt_dayofweek = _dt_day_of_week
    _dt_weekday = _dt_day_of_week

    @property
    def _dt_day_of_year(self) -> Self:
        # 返回一个新的对象，其中包含一年中的第几天信息的数组
        return type(self)(pc.day_of_year(self._pa_array))

    _dt_dayofyear = _dt_day_of_year

    @property
    def _dt_hour(self) -> Self:
        # 返回一个新的对象，其中包含小时信息的数组
        return type(self)(pc.hour(self._pa_array))

    def _dt_isocalendar(self) -> Self:
        # 返回一个新的对象，其中包含ISO日历信息的数组
        return type(self)(pc.iso_calendar(self._pa_array))

    @property
    def _dt_is_leap_year(self) -> Self:
        # 返回一个新的对象，其中包含是否为闰年信息的数组
        return type(self)(pc.is_leap_year(self._pa_array))

    @property
    def _dt_is_month_start(self) -> Self:
        # 返回一个新的对象，其中包含是否为月初的布尔数组
        return type(self)(pc.equal(pc.day(self._pa_array), 1))

    @property
    def _dt_is_month_end(self) -> Self:
        # 返回一个新的对象，其中包含是否为月末的布尔数组
        result = pc.equal(
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="month"),
            ),
            1,
        )
        return type(self)(result)

    @property
    def _dt_is_year_start(self) -> Self:
        # 返回一个新的对象，其中包含是否为年初的布尔数组
        return type(self)(
            pc.and_(
                pc.equal(pc.month(self._pa_array), 1),
                pc.equal(pc.day(self._pa_array), 1),
            )
        )

    @property
    def _dt_is_year_end(self) -> Self:
        # 返回一个新的对象，其中包含是否为年末的布尔数组
        return type(self)(
            pc.and_(
                pc.equal(pc.month(self._pa_array), 12),
                pc.equal(pc.day(self._pa_array), 31),
            )
        )
    # 返回一个新对象，其中包含一个布尔数组，指示每个时间戳是否是季度的开始
    def _dt_is_quarter_start(self) -> Self:
        result = pc.equal(
            # 将时间数组向下取整到季度级别
            pc.floor_temporal(self._pa_array, unit="quarter"),
            # 将时间数组向下取整到天级别
            pc.floor_temporal(self._pa_array, unit="day"),
        )
        # 返回类型相同的对象，其中包含布尔数组结果
        return type(self)(result)

    @property
    # 返回一个新对象，其中包含一个布尔数组，指示每个时间戳是否是季度的结束
    def _dt_is_quarter_end(self) -> Self:
        result = pc.equal(
            # 计算每个时间戳的天数与所在季度末的天数之间的差异，结果为1表示是季度末
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="quarter"),
            ),
            1,
        )
        # 返回类型相同的对象，其中包含布尔数组结果
        return type(self)(result)

    @property
    # 返回一个新对象，其中包含一个整数数组，指示每个时间戳所在月份的天数
    def _dt_days_in_month(self) -> Self:
        result = pc.days_between(
            # 将时间数组向下取整到月级别
            pc.floor_temporal(self._pa_array, unit="month"),
            # 将时间数组向上取整到月级别
            pc.ceil_temporal(self._pa_array, unit="month"),
        )
        # 返回类型相同的对象，其中包含整数数组结果
        return type(self)(result)

    # 将 _dt_days_in_month 定义为 _dt_days_in_month 的别名
    _dt_daysinmonth = _dt_days_in_month

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的微秒部分
    def _dt_microsecond(self) -> Self:
        return type(self)(pc.microsecond(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的分钟部分
    def _dt_minute(self) -> Self:
        return type(self)(pc.minute(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的月份部分
    def _dt_month(self) -> Self:
        return type(self)(pc.month(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的纳秒部分
    def _dt_nanosecond(self) -> Self:
        return type(self)(pc.nanosecond(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的季度部分
    def _dt_quarter(self) -> Self:
        return type(self)(pc.quarter(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个整数数组，表示每个时间戳的秒部分
    def _dt_second(self) -> Self:
        return type(self)(pc.second(self._pa_array))

    @property
    # 返回一个新对象，其中包含一个日期数组，将每个时间戳强制转换为日期
    def _dt_date(self) -> Self:
        return type(self)(self._pa_array.cast(pa.date32()))

    @property
    # 返回一个新对象，其中包含一个时间数组，将每个时间戳强制转换为指定单位的时间
    def _dt_time(self) -> Self:
        unit = (
            self.dtype.pyarrow_dtype.unit
            if self.dtype.pyarrow_dtype.unit in {"us", "ns"}
            else "ns"
        )
        return type(self)(self._pa_array.cast(pa.time64(unit)))

    @property
    # 返回一个时区对象，根据数据类型的时区信息（如果有的话）
    def _dt_tz(self):
        return timezones.maybe_get_tz(self.dtype.pyarrow_dtype.tz)

    @property
    # 返回一个时间单位字符串，表示每个时间戳的时间单位
    def _dt_unit(self):
        return self.dtype.pyarrow_dtype.unit

    # 返回一个新对象，其中包含一个日期数组，将每个时间戳向下取整到天级别
    def _dt_normalize(self) -> Self:
        return type(self)(pc.floor_temporal(self._pa_array, 1, "day"))

    # 返回一个新对象，其中包含一个字符串数组，表示每个时间戳按照指定格式格式化后的结果
    def _dt_strftime(self, format: str) -> Self:
        return type(self)(pc.strftime(self._pa_array, format=format))

    # 返回一个新对象，其中包含一个时间数组，根据指定的方法和频率进行时间舍入操作
    def _round_temporally(
        self,
        method: Literal["ceil", "floor", "round"],
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    def _dt_ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 执行时间向上舍入操作，调用内部方法 _round_temporally
        return self._round_temporally("ceil", freq, ambiguous, nonexistent)

    def _dt_floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 执行时间向下舍入操作，调用内部方法 _round_temporally
        return self._round_temporally("floor", freq, ambiguous, nonexistent)

    def _dt_round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 执行时间四舍五入操作，调用内部方法 _round_temporally
        return self._round_temporally("round", freq, ambiguous, nonexistent)

    def _dt_day_name(self, locale: str | None = None) -> Self:
        # 返回当前日期对应的星期几名称，可以指定 locale，默认为 "C"
        if locale is None:
            locale = "C"
        return type(self)(pc.strftime(self._pa_array, format="%A", locale=locale))

    def _dt_month_name(self, locale: str | None = None) -> Self:
        # 返回当前日期对应的月份名称，可以指定 locale，默认为 "C"
        if locale is None:
            locale = "C"
        return type(self)(pc.strftime(self._pa_array, format="%B", locale=locale))

    def _dt_to_pydatetime(self) -> Series:
        # 将日期时间数据转换为 Python 的 datetime 对象数组
        from pandas import Series

        # 如果数据类型是日期类型，抛出异常
        if pa.types.is_date(self.dtype.pyarrow_dtype):
            raise ValueError(
                f"to_pydatetime cannot be called with {self.dtype.pyarrow_dtype} type. "
                "Convert to pyarrow timestamp type."
            )
        # 将数据转换为 Python datetime 对象数组
        data = self._pa_array.to_pylist()
        if self._dtype.pyarrow_dtype.unit == "ns":
            # 如果数据单位是纳秒，转换为 datetime 对象，忽略警告
            data = [None if ts is None else ts.to_pydatetime(warn=False) for ts in data]
        return Series(data, dtype=object)

    def _dt_tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 本地化时间到指定时区，可以指定如何处理歧义和不存在的时间点
        # 调用内部方法 _round_temporally，处理本地化过程
    # 返回类型为 Self 的方法定义
    ) -> Self:
        # 如果 ambiguous 不是 "raise"，则抛出 NotImplementedError 异常
        if ambiguous != "raise":
            raise NotImplementedError(f"{ambiguous=} is not supported")
        # 定义不存在时处理方式的映射字典
        nonexistent_pa = {
            "raise": "raise",
            "shift_backward": "earliest",
            "shift_forward": "latest",
        }.get(
            nonexistent,  # 指定参数类型为忽略类型的注释
            None,  # 如果不存在则返回 None
        )
        # 如果不存在时处理方式为 None，则抛出 NotImplementedError 异常
        if nonexistent_pa is None:
            raise NotImplementedError(f"{nonexistent=} is not supported")
        # 如果时区为 None，则使用当前对象的时区信息来进行类型转换
        if tz is None:
            result = self._pa_array.cast(pa.timestamp(self.dtype.pyarrow_dtype.unit))
        else:
            # 否则，使用 pytz 库中的 assume_timezone 函数来转换时区
            result = pc.assume_timezone(
                self._pa_array, str(tz), ambiguous=ambiguous, nonexistent=nonexistent_pa
            )
        # 返回转换后的结果，类型为当前对象的类型
        return type(self)(result)

    # 将时区转换为指定时区的方法定义
    def _dt_tz_convert(self, tz) -> Self:
        # 如果当前时间戳没有时区信息，则抛出 TypeError 异常
        if self.dtype.pyarrow_dtype.tz is None:
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )
        # 获取当前时间戳的单位
        current_unit = self.dtype.pyarrow_dtype.unit
        # 使用 pyarrow 库的 timestamp 函数进行时区转换
        result = self._pa_array.cast(pa.timestamp(current_unit, tz))
        # 返回转换后的结果，类型为当前对象的类型
        return type(self)(result)
# 定义一个函数，用于将一组 Arrow 扩展数组进行转置操作，但速度更快。
def transpose_homogeneous_pyarrow(
    arrays: Sequence[ArrowExtensionArray],
) -> list[ArrowExtensionArray]:
    """Transpose arrow extension arrays in a list, but faster.

    Input should be a list of arrays of equal length and all have the same
    dtype. The caller is responsible for ensuring validity of input data.
    """
    # 将输入的数组转换为列表形式
    arrays = list(arrays)
    # 获取数组的行数和列数，假设第一个数组的长度即为行数，数组的数量即为列数
    nrows, ncols = len(arrays[0]), len(arrays)
    # 创建一个用于重排索引的数组，确保转置后的数据布局是正确的
    indices = np.arange(nrows * ncols).reshape(ncols, nrows).T.reshape(-1)
    # 将所有数组中的数据合并为一个 chunked array 对象
    arr = pa.chunked_array([chunk for arr in arrays for chunk in arr._pa_array.chunks])
    # 按照指定的索引重新排列数组的数据
    arr = arr.take(indices)
    # 根据每一列的数据创建 ArrowExtensionArray 对象的列表，并返回该列表
    return [ArrowExtensionArray(arr.slice(i * ncols, ncols)) for i in range(nrows)]
```