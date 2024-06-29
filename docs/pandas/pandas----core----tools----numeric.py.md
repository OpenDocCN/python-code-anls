# `D:\src\scipysrc\pandas\pandas\core\tools\numeric.py`

```
# 导入必要的模块和类型声明
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np  # 导入NumPy库，用于数值操作

from pandas._libs import lib  # 导入Pandas的底层库
from pandas.util._validators import check_dtype_backend  # 导入Pandas的数据类型验证函数

from pandas.core.dtypes.cast import maybe_downcast_numeric  # 导入Pandas的数值类型转换函数
from pandas.core.dtypes.common import (  # 导入Pandas的常用数据类型判断函数
    ensure_object,
    is_bool_dtype,
    is_decimal,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype  # 导入Pandas的Arrow数据类型
from pandas.core.dtypes.generic import (  # 导入Pandas的通用数据类型
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays import BaseMaskedArray  # 导入Pandas的基础掩码数组
from pandas.core.arrays.string_ import StringDtype  # 导入Pandas的字符串数据类型

if TYPE_CHECKING:
    from pandas._typing import (  # 类型检查，导入Pandas的类型声明
        DateTimeErrorChoices,
        DtypeBackend,
        npt,
    )


def to_numeric(
    arg,  # 待转换的参数，可以是标量、列表、元组、一维数组或Series
    errors: DateTimeErrorChoices = "raise",  # 错误处理选项，默认为抛出异常
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,  # 下转型选项，限定为特定字符串或None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端，可以是指定类型或无默认值
):
    """
    Convert argument to a numeric type.

    The default return dtype is `float64` or `int64`
    depending on the data supplied. Use the `downcast` parameter
    to obtain other dtypes.

    Please note that precision loss may occur if really large numbers
    are passed in. Due to the internal limitations of `ndarray`, if
    numbers smaller than `-9223372036854775808` (np.iinfo(np.int64).min)
    or larger than `18446744073709551615` (np.iinfo(np.uint64).max) are
    passed in, it is very likely they will be converted to float so that
    they can be stored in an `ndarray`. These warnings apply similarly to
    `Series` since it internally leverages `ndarray`.

    Parameters
    ----------
    arg : scalar, list, tuple, 1-d array, or Series
        Argument to be converted.

    errors : {'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
    """
    downcast : str, default None
        # downcast 参数，用于指定数据类型转换的方式，可选值为 'integer', 'signed', 'unsigned', or 'float'。
        # 如果不为 None，并且数据已成功转换为数值类型（或者本身就是数值类型），则将结果数据按照以下规则进行降级：
        # - 'integer' 或 'signed': 转换为最小的有符号整数类型（最小为 np.int8）
        # - 'unsigned': 转换为最小的无符号整数类型（最小为 np.uint8）
        # - 'float': 转换为最小的浮点数类型（最小为 np.float32）
        
        # 由于这一行为与核心的数值类型转换分开，因此任何在降级过程中引发的错误都会被显示，不受 'errors' 输入值的影响。
        
        # 此外，只有当结果数据的数据类型的大小严格大于要转换为的数据类型时，降级才会发生。
        # 如果检查的所有数据类型都不满足该规范，则不会对数据执行降级操作。

    dtype_backend : {'numpy_nullable', 'pyarrow'}
        # 结果DataFrame应用的后端数据类型（仍处于实验阶段）。
        # 如果未指定，则默认行为是不使用可空数据类型。
        # 如果指定，行为如下：
        # * ``"numpy_nullable"``: 返回带有可空数据类型支持的DataFrame
        # * ``"pyarrow"``: 返回带有pyarrow支持的可空 :class:`ArrowDtype`
        
        # .. versionadded:: 2.0
        # 版本 2.0 添加

    Returns
    -------
    ret
        # 如果解析成功，则返回数值类型。
        # 返回类型取决于输入。如果输入是 Series，则返回 Series，否则返回 ndarray。

    See Also
    --------
    DataFrame.astype : 将参数转换为指定的数据类型。
    to_datetime : 将参数转换为 datetime。
    to_timedelta : 将参数转换为 timedelta。
    numpy.ndarray.astype : 将 numpy 数组转换为指定类型。
    DataFrame.convert_dtypes : 转换数据类型。

    Examples
    --------
    Take separate series and convert to numeric, coercing when told to

    >>> s = pd.Series(["1.0", "2", -3])
    >>> pd.to_numeric(s)
    0    1.0
    1    2.0
    2   -3.0
    dtype: float64
    >>> pd.to_numeric(s, downcast="float")
    0    1.0
    1    2.0
    2   -3.0
    dtype: float32
    >>> pd.to_numeric(s, downcast="signed")
    0    1
    1    2
    2   -3
    dtype: int8
    >>> s = pd.Series(["apple", "1.0", "2", -3])
    >>> pd.to_numeric(s, errors="coerce")
    0    NaN
    1    1.0
    2    2.0
    3   -3.0
    dtype: float64

    Downcasting of nullable integer and floating dtypes is supported:

    >>> s = pd.Series([1, 2, 3], dtype="Int64")
    >>> pd.to_numeric(s, downcast="integer")
    0    1
    1    2
    2    3
    dtype: Int8
    >>> s = pd.Series([1.0, 2.1, 3.0], dtype="Float64")
    >>> pd.to_numeric(s, downcast="float")
    0    1.0
    1    2.1
    2    3.0
    dtype: Float32
    # 检查 downcast 参数是否为预定义的值之一，否则抛出 ValueError 异常
    if downcast not in (None, "integer", "signed", "unsigned", "float"):
        raise ValueError("invalid downcasting method provided")

    # 检查 errors 参数是否为预定义的值之一，否则抛出 ValueError 异常
    if errors not in ("raise", "coerce"):
        raise ValueError("invalid error value specified")

    # 检查并确保 dtype_backend 参数符合要求
    check_dtype_backend(dtype_backend)

    # 初始化标志变量，用于判断参数 arg 的类型
    is_series = False
    is_index = False
    is_scalars = False

    # 根据参数 arg 的类型进行不同的处理
    if isinstance(arg, ABCSeries):
        # 如果 arg 是 Pandas Series 类型，则设置相应标志并获取其值
        is_series = True
        values = arg.values
    elif isinstance(arg, ABCIndex):
        # 如果 arg 是 Pandas Index 类型，则设置相应标志并根据需要进行类型转换
        is_index = True
        if needs_i8_conversion(arg.dtype):
            values = arg.view("i8")
        else:
            values = arg.values
    elif isinstance(arg, (list, tuple)):
        # 如果 arg 是列表或元组，则转换为 NumPy 数组
        values = np.array(arg, dtype="O")
    elif is_scalar(arg):
        # 如果 arg 是标量，则根据类型进行处理
        if is_decimal(arg):
            return float(arg)
        if is_number(arg):
            return arg
        # 如果 arg 是其他标量类型，则转换为包含一个元素的 NumPy 数组
        is_scalars = True
        values = np.array([arg], dtype="O")
    elif getattr(arg, "ndim", 1) > 1:
        # 如果 arg 是多维数组，则抛出 TypeError 异常
        raise TypeError("arg must be a list, tuple, 1-d array, or Series")
    else:
        # 否则直接使用 arg 的值
        values = arg

    # 对于 IntegerArray 和 FloatingArray，提取非空值进行类型转换
    # 保存掩码以便在转换后重建完整数组
    mask: npt.NDArray[np.bool_] | None = None
    if isinstance(values, BaseMaskedArray):
        mask = values._mask
        values = values._data[~mask]

    # 获取 values 的数据类型
    values_dtype = getattr(values, "dtype", None)

    # 如果 values 的数据类型是 ArrowDtype，则处理缺失值
    if isinstance(values_dtype, ArrowDtype):
        mask = values.isna()
        values = values.dropna().to_numpy()

    new_mask: np.ndarray | None = None

    # 如果 values 的数据类型是数值类型，则不进行额外操作
    if is_numeric_dtype(values_dtype):
        pass
    # 如果 values 的数据类型是 "mM" 类型，则将其视为 np.int64 类型
    elif lib.is_np_dtype(values_dtype, "mM"):
        values = values.view(np.int64)
    else:
        # 否则确保 values 是对象类型，并尝试进行数值转换
        values = ensure_object(values)
        coerce_numeric = errors != "raise"
        values, new_mask = lib.maybe_convert_numeric(  # type: ignore[call-overload]
            values,
            set(),
            coerce_numeric=coerce_numeric,
            convert_to_masked_nullable=dtype_backend is not lib.no_default
            or isinstance(values_dtype, StringDtype)
            and not values_dtype.storage == "pyarrow_numpy",
        )

    # 如果存在新的掩码 new_mask，则从 values 中移除不必要的值
    if new_mask is not None:
        values = values[~new_mask]
    # 否则，根据条件生成新的掩码
    elif (
        dtype_backend is not lib.no_default
        and new_mask is None
        or isinstance(values_dtype, StringDtype)
        and not values_dtype.storage == "pyarrow_numpy"
    ):
        new_mask = np.zeros(values.shape, dtype=np.bool_)

    # 如果数据已成功转换为数值类型，并且指定了 downcast 方法，则尝试进行 downcast
    # 注意：此处没有实际的 downcast 操作，仅进行了条件检查
    # 如果 downcast 参数不为 None，并且 values 的数据类型是数值类型
    if downcast is not None and is_numeric_dtype(values.dtype):
        typecodes: str | None = None

        # 根据 downcast 的取值选择合适的类型码
        if downcast in ("integer", "signed"):
            typecodes = np.typecodes["Integer"]
        elif downcast == "unsigned" and (not len(values) or np.min(values) >= 0):
            typecodes = np.typecodes["UnsignedInteger"]
        elif downcast == "float":
            typecodes = np.typecodes["Float"]

            # 由于 pandas 仅支持 np.float32 类型
            # 小于该类型的浮点类型非常罕见且支持不佳
            float_32_char = np.dtype(np.float32).char
            float_32_ind = typecodes.index(float_32_char)
            typecodes = typecodes[float_32_ind:]

        # 如果找到了合适的类型码
        if typecodes is not None:
            # 从小到大尝试转换类型
            for typecode in typecodes:
                dtype = np.dtype(typecode)
                # 如果当前数据类型的字节大小不大于目标类型的字节大小
                if dtype.itemsize <= values.dtype.itemsize:
                    # 尝试进行数值类型转换
                    values = maybe_downcast_numeric(values, dtype)

                    # 如果转换成功
                    if values.dtype == dtype:
                        break

    # GH33013: 对于 IntegerArray、BooleanArray 和 FloatingArray，需要重构掩码数组
    if (mask is not None or new_mask is not None) and not is_string_dtype(values.dtype):
        if mask is None or (new_mask is not None and new_mask.shape == mask.shape):
            # 如果 mask 为 None，或者 new_mask 不为 None 且形状相同
            # 则使用 new_mask 作为当前掩码
            mask = new_mask
        else:
            # 否则复制当前 mask 作为新的掩码
            mask = mask.copy()
        assert isinstance(mask, np.ndarray)
        # 根据掩码创建数据数组
        data = np.zeros(mask.shape, dtype=values.dtype)
        # 将未掩码的数据复制到数据数组中
        data[~mask] = values

        # 导入需要重构的数组类型
        from pandas.core.arrays import (
            ArrowExtensionArray,
            BooleanArray,
            FloatingArray,
            IntegerArray,
        )

        klass: type[IntegerArray | BooleanArray | FloatingArray]
        # 根据数据类型确定需要使用的数组类型
        if is_integer_dtype(data.dtype):
            klass = IntegerArray
        elif is_bool_dtype(data.dtype):
            klass = BooleanArray
        else:
            klass = FloatingArray
        # 使用适当的数组类型和掩码数据创建新的 values
        values = klass(data, mask)

        # 如果 dtype_backend 是 "pyarrow" 或者 values_dtype 是 ArrowDtype 的实例
        if dtype_backend == "pyarrow" or isinstance(values_dtype, ArrowDtype):
            # 将 values 转换为 ArrowExtensionArray
            values = ArrowExtensionArray(values.__arrow_array__())

    # 如果是 Series 对象
    if is_series:
        # 使用 values、index 和 name 构造新的 Series 对象
        return arg._constructor(values, index=arg.index, name=arg.name)
    # 如果是 Index 对象
    elif is_index:
        # 因为希望在可能的情况下强制转换为数值类型，所以不使用 _shallow_copy
        from pandas import Index

        # 使用 values 和 name 构造新的 Index 对象
        return Index(values, name=arg.name)
    # 如果是标量值
    elif is_scalars:
        # 返回 values 的第一个元素作为结果
        return values[0]
    else:
        # 否则直接返回 values
        return values
```