# `D:\src\scipysrc\pandas\pandas\core\dtypes\missing.py`

```
"""
missing types & inference
"""

# 从未来模块导入类型注解支持
from __future__ import annotations

# 导入 Decimal 类型
from decimal import Decimal
# 导入类型检查相关的模块和装饰器
from typing import (
    TYPE_CHECKING,
    overload,
)
# 导入警告模块
import warnings

# 导入 numpy 库并使用 np 别名
import numpy as np

# 导入 pandas 库的内部 C 模块
from pandas._libs import lib
# 导入 pandas 库的缺失值处理模块
import pandas._libs.missing as libmissing
# 导入 pandas 库的时间序列相关模块
from pandas._libs.tslibs import (
    NaT,   # 无效时间
    iNaT,  # 无效时间
)

# 导入 pandas 核心数据类型常用函数
from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    TD64NS_DTYPE,
    ensure_object,
    is_scalar,
    is_string_or_object_np_dtype,
)
# 导入 pandas 核心数据类型
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
# 导入 pandas 核心泛型数据类型
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
# 导入 pandas 核心数据类型推断模块
from pandas.core.dtypes.inference import is_list_like

# 如果是类型检查的情况下，导入特定类型
if TYPE_CHECKING:
    from re import Pattern  # 导入正则表达式模式
    from pandas._libs.missing import NAType  # 导入缺失类型
    from pandas._libs.tslibs import NaTType  # 导入无效时间类型
    from pandas._typing import (  # 导入 pandas 类型定义
        ArrayLike,
        DtypeObj,
        NDFrame,
        NDFrameT,
        Scalar,
        npt,
    )

    from pandas import Series  # 导入 pandas 库的 Series 类型
    from pandas.core.indexes.base import Index  # 导入 pandas 索引基类

# 导入 pandas 内部缺失值处理的标量函数
isposinf_scalar = libmissing.isposinf_scalar
isneginf_scalar = libmissing.isneginf_scalar

# 定义对象和字符串类型的 NumPy 数据类型
_dtype_object = np.dtype("object")
_dtype_str = np.dtype(str)


@overload
# 函数重载：判断标量、正则、缺失值类型、无效时间类型的缺失值情况
def isna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...


@overload
# 函数重载：判断数组、索引、列表等对象的缺失值情况，返回布尔数组
def isna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]: ...


@overload
# 函数重载：判断 NDFrame 类型的对象的缺失值情况，返回相同类型的对象
def isna(obj: NDFrameT) -> NDFrameT: ...


# 处理联合类型的函数重载：可以接受 NDFrameT、ArrayLike、Index、list 中的任何一种类型对象
@overload
def isna(
    obj: NDFrameT | ArrayLike | Index | list,
) -> NDFrameT | npt.NDArray[np.bool_]: ...


@overload
# 函数重载：接受任意对象，返回布尔值、布尔数组或 NDFrame 对象
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...


# 判断对象是否为缺失值的主函数定义
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna("dog")
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    """
    # 创建一个二维的 NumPy 数组，包含浮点数和 NaN（Not a Number，表示缺失值）
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])

    # 使用 pandas 库中的 isna 函数检测数组中的缺失值，返回一个布尔值的二维数组
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    # 对于 DatetimeIndex 类型对象，创建一个包含日期时间值的索引，其中包括 NaT（Not a Time）表示的缺失日期时间
    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None, "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[s]', freq=None)
    
    # 使用 pandas 的 isna 函数检测索引中的缺失值，返回一个布尔值数组
    >>> pd.isna(index)
    array([False, False,  True, False])

    # 对于 DataFrame 和 Series 对象，使用 pandas 的 DataFrame 构造函数创建一个包含字符串的二维数据框
    >>> df = pd.DataFrame([["ant", "bee", "cat"], ["dog", None, "fly"]])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    
    # 使用 pandas 的 isna 函数检测数据框中的缺失值，返回一个布尔值的数据框
    >>> pd.isna(df)
           0      1      2
    0  False  False  False
    1  False   True  False

    # 使用 isna 函数检测数据框中第二列的缺失值，返回一个布尔值的 Series 对象
    >>> pd.isna(df[1])
    0    False
    1     True
    Name: 1, dtype: bool
    """
    # 返回对象 obj 的缺失值检测结果
    return _isna(obj)
# 将 isna 函数赋值给 isnull，以便兼容调用 isna 或 isnull
isnull = isna

# 检测缺失值的函数，将 None、NaN 或 NA 视为 null
def _isna(obj):
    """
    Detect missing values, treating None, NaN or NA as null.

    Parameters
    ----------
    obj: ndarray or object value
        Input array or scalar value.

    Returns
    -------
    boolean ndarray or boolean
    """
    # 如果输入是标量，则调用 libmissing 检查是否为 null
    if is_scalar(obj):
        return libmissing.checknull(obj)
    # 如果输入是 MultiIndex 类型，则抛出未实现异常
    elif isinstance(obj, ABCMultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    # 如果输入是类型对象，则返回 False
    elif isinstance(obj, type):
        return False
    # 如果输入是 ndarray 或 ExtensionArray，则调用 _isna_array 处理
    elif isinstance(obj, (np.ndarray, ABCExtensionArray)):
        return _isna_array(obj)
    # 如果输入是 Index 类型，则尝试使用缓存的 isna，避免材料化 RangeIndex._values
    elif isinstance(obj, ABCIndex):
        if not obj._can_hold_na:
            return obj.isna()
        return _isna_array(obj._values)
    # 如果输入是 Series 类型，则返回其值是否为缺失值的数组
    elif isinstance(obj, ABCSeries):
        result = _isna_array(obj._values)
        # 将结果封装为与原始 Series 类型相同的对象
        result = obj._constructor(result, index=obj.index, name=obj.name, copy=False)
        return result
    # 如果输入是 DataFrame 类型，则返回其是否包含缺失值的 DataFrame
    elif isinstance(obj, ABCDataFrame):
        return obj.isna()
    # 如果输入是列表，则将其转换为 ndarray 后再调用 _isna_array 处理
    elif isinstance(obj, list):
        return _isna_array(np.asarray(obj, dtype=object))
    # 如果输入具有 "__array__" 属性，则将其转换为 ndarray 后再调用 _isna_array 处理
    elif hasattr(obj, "__array__"):
        return _isna_array(np.asarray(obj))
    # 其他情况均返回 False
    else:
        return False


def _isna_array(values: ArrayLike) -> npt.NDArray[np.bool_] | NDFrame:
    """
    Return an array indicating which values of the input array are NaN / NA.

    Parameters
    ----------
    obj: ndarray or ExtensionArray
        The input array whose elements are to be checked.

    Returns
    -------
    array-like
        Array of boolean values denoting the NA status of each element.
    """
    # 获取输入值的数据类型
    dtype = values.dtype
    result: npt.NDArray[np.bool_] | NDFrame

    # 如果 values 不是 ndarray 类型，而是 ExtensionArray，则调用其 isna 方法
    if not isinstance(values, np.ndarray):
        result = values.isna()  # type: ignore[assignment]
    # 如果 values 是 np.rec.recarray 类型，则调用 _isna_recarray_dtype 处理
    elif isinstance(values, np.rec.recarray):
        result = _isna_recarray_dtype(values)
    # 如果 values 的 dtype 是字符串或对象类型，则调用 _isna_string_dtype 处理
    elif is_string_or_object_np_dtype(values.dtype):
        result = _isna_string_dtype(values)
    # 如果 values 的 dtype 类型属于 'mM'，即时间类型，则判断是否为 NaT 模式
    elif dtype.kind in "mM":
        result = values.view("i8") == iNaT
    # 其他情况使用 np.isnan 判断是否为 NaN
    else:
        result = np.isnan(values)

    return result


def _isna_string_dtype(values: np.ndarray) -> npt.NDArray[np.bool_]:
    # 绕过 NumPy ticket 1542 的问题
    dtype = values.dtype

    # 如果 dtype 的 kind 是 'S' 或 'U'，即字符串类型，则初始化一个全为 False 的布尔数组
    if dtype.kind in ("S", "U"):
        result = np.zeros(values.shape, dtype=bool)
    # 否则，根据 values 的维度调用 libmissing.isnaobj 处理
    else:
        if values.ndim in {1, 2}:
            result = libmissing.isnaobj(values)
        else:
            # 对于 0 维的情况，通过例如 mask_missing 达到
            result = libmissing.isnaobj(values.ravel())
            result = result.reshape(values.shape)
    # 返回函数的结果
    return result
# 检测是否为具有缺失值的 np.rec.recarray 类型数据
def _isna_recarray_dtype(values: np.rec.recarray) -> npt.NDArray[np.bool_]:
    # 创建一个与输入数据形状相同的布尔类型的全零数组
    result = np.zeros(values.shape, dtype=bool)
    # 遍历 values 中的每个记录
    for i, record in enumerate(values):
        # 将记录转换为普通的数组
        record_as_array = np.array(record.tolist())
        # 检查该数组中是否存在缺失值，并将结果存入 does_record_contain_nan
        does_record_contain_nan = isna_all(record_as_array)
        # 将该记录的检测结果存入结果数组 result
        result[i] = np.any(does_record_contain_nan)

    # 返回包含每条记录缺失值检测结果的布尔类型数组
    return result


# 函数重载：处理 Scalar、Pattern、NAType 或 NaTType 类型的输入，返回布尔类型值
@overload
def notna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...


# 函数重载：处理 ArrayLike、Index、list 类型的输入，返回布尔类型的 numpy 数组
@overload
def notna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]: ...


# 函数重载：处理 NDFrameT 类型的输入，返回 NDFrameT 类型
@overload
def notna(obj: NDFrameT) -> NDFrameT: ...


# 函数重载：处理 Union[NDFrameT, ArrayLike, Index, list] 类型的输入，返回 Union[NDFrameT, numpy 数组]
@overload
def notna(
    obj: NDFrameT | ArrayLike | Index | list,
) -> NDFrameT | npt.NDArray[np.bool_]: ...


# 函数重载：处理 object 类型的输入，返回 bool、numpy 布尔类型数组或 NDFrame 对象
@overload
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...


# 主函数定义：检测输入对象中的非缺失值
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    检测数组类对象中的非缺失值。

    该函数接受标量或数组类对象，并指示其值是否有效（非缺失，数值数组中为 ``NaN``，对象数组中为 ``None`` 或 ``NaN``，datetimelike 中为 ``NaT``）。

    Parameters
    ----------
    obj : array-like 或对象值
        要检查的非空或非缺失值对象。

    Returns
    -------
    bool 或布尔类型数组
        对于标量输入，返回一个布尔值。
        对于数组输入，返回一个布尔数组，指示每个对应元素是否有效。

    See Also
    --------
    isna : pandas.notna 的布尔逆操作。
    Series.notna : 检测 Series 中的有效值。
    DataFrame.notna : 检测 DataFrame 中的有效值。
    Index.notna : 检测 Index 中的有效值。

    Examples
    --------
    标量参数（包括字符串）会返回一个标量布尔值。

    >>> pd.notna("dog")
    True

    >>> pd.notna(pd.NA)
    False

    >>> pd.notna(np.nan)
    False

    ndarrays 会返回一个布尔类型的数组。

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.notna(array)
    array([[ True, False,  True],
           [ True,  True, False]])

    对于索引，会返回一个布尔类型的数组。

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None, "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[s]', freq=None)
    >>> pd.notna(index)
    array([ True,  True, False,  True])

    对于 Series 和 DataFrame，会返回相同类型的对象，包含布尔类型的值。

    >>> df = pd.DataFrame([["ant", "bee", "cat"], ["dog", None, "fly"]])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    >>> pd.notna(df)
          0      1     2
    0  True   True  True
    1  True  False  True

    >>> pd.notna(df[1])
    0     True
    1    False
    Name: 1, dtype: bool
    """
    # 调用 isna 函数，获取输入对象的反向结果
    res = isna(obj)
    # 如果 res 是布尔类型，则返回其相反值
    if isinstance(res, bool):
        return not res
    # 否则返回 res 的按位取反值（取反操作符）
    return ~res
notnull = notna  # 将 notna 函数赋值给 notnull 变量，使其作为别名使用


def array_equivalent(
    left,
    right,
    strict_nan: bool = False,
    dtype_equal: bool = False,
) -> bool:
    """
    判断两个数组 left 和 right 是否在相应位置具有相等的非 NaN 元素及 NaN 值。如果是则返回 True，否则返回 False。
    假设 left 和 right 是相同 dtype 的 NumPy 数组。如果 dtype 不同，此函数（特别是 NaN 的行为）的行为未定义。

    Parameters
    ----------
    left, right : ndarrays
        要比较的两个 NumPy 数组
    strict_nan : bool, default False
        如果为 True，则将 NaN 和 None 视为不同。
    dtype_equal : bool, default False
        指示 `left` 和 `right` 是否根据 `is_dtype_equal` 具有相同的 dtype。某些方法如 `BlockManager.equals` 要求 dtypes 必须匹配。
        将此设置为 `True` 可以提高性能，但会导致对于相等但不同 dtype 的数组给出不同的结果。

    Returns
    -------
    b : bool
        如果数组等效，则返回 True。

    Examples
    --------
    >>> array_equivalent(np.array([1, 2, np.nan]), np.array([1, 2, np.nan]))
    True
    >>> array_equivalent(np.array([1, np.nan, 2]), np.array([1, 2, np.nan]))
    False
    """
    left, right = np.asarray(left), np.asarray(right)

    # shape compat
    if left.shape != right.shape:
        return False

    if dtype_equal:
        # fastpath when we require that the dtypes match (Block.equals)
        if left.dtype.kind in "fc":
            return _array_equivalent_float(left, right)
        elif left.dtype.kind in "mM":
            return _array_equivalent_datetimelike(left, right)
        elif is_string_or_object_np_dtype(left.dtype):
            # TODO: fastpath for pandas' StringDtype
            return _array_equivalent_object(left, right, strict_nan)
        else:
            return np.array_equal(left, right)

    # Slow path when we allow comparing different dtypes.
    # Object arrays can contain None, NaN and NaT.
    # string dtypes must be come to this path for NumPy 1.7.1 compat
    if left.dtype.kind in "OSU" or right.dtype.kind in "OSU":
        # Note: `in "OSU"` is non-trivially faster than `in ["O", "S", "U"]`
        #  or `in ("O", "S", "U")`
        return _array_equivalent_object(left, right, strict_nan)

    # NaNs can occur in float and complex arrays.
    if left.dtype.kind in "fc":
        if not (left.size and right.size):
            return True
        return ((left == right) | (isna(left) & isna(right))).all()

    elif left.dtype.kind in "mM" or right.dtype.kind in "mM":
        # datetime64, timedelta64, Period
        if left.dtype != right.dtype:
            return False

        left = left.view("i8")
        right = right.view("i8")

    # if we have structured dtypes, compare first
    if (
        left.dtype.type is np.void or right.dtype.type is np.void
    ) and left.dtype != right.dtype:
        return False
    # 检查两个 NumPy 数组是否完全相等，并返回布尔值结果
    return np.array_equal(left, right)
# 检查两个 numpy 数组是否在浮点数方面等价，包括处理 NaN 值的情况
def _array_equivalent_float(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(((left == right) | (np.isnan(left) & np.isnan(right))).all())

# 检查两个 numpy 数组是否在日期时间方面完全等价
def _array_equivalent_datetimelike(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left.view("i8"), right.view("i8"))

# 检查两个 numpy 数组是否在对象方面等价，可以选择是否严格处理 NaN
def _array_equivalent_object(
    left: np.ndarray, right: np.ndarray, strict_nan: bool
) -> bool:
    # 将左右两个数组转换为对象数组
    left = ensure_object(left)
    right = ensure_object(right)

    # 初始化一个掩码变量，用于标记 NaN 值
    mask: npt.NDArray[np.bool_] | None = None
    if strict_nan:
        # 如果需要严格处理 NaN，则生成掩码
        mask = isna(left) & isna(right)
        # 如果没有任何 NaN 值，则重置掩码为空
        if not mask.any():
            mask = None

    try:
        if mask is None:
            # 如果没有掩码，则直接调用库函数比较对象数组的等价性
            return lib.array_equivalent_object(left, right)
        else:
            # 否则，逐个比较未被掩码标记的元素
            if not lib.array_equivalent_object(left[~mask], right[~mask]):
                return False
            # 分别处理剩余的左右数组部分
            left_remaining = left[mask]
            right_remaining = right[mask]
    except ValueError:
        # 如果无法比较左右数组（如嵌套数组），则默认使用整个数组
        left_remaining = left
        right_remaining = right

    # 逐个比较剩余的左右值
    for left_value, right_value in zip(left_remaining, right_remaining):
        if left_value is NaT and right_value is not NaT:
            return False
        elif left_value is libmissing.NA and right_value is not libmissing.NA:
            return False
        elif isinstance(left_value, float) and np.isnan(left_value):
            if not isinstance(right_value, float) or not np.isnan(right_value):
                return False
        else:
            # 使用警告过滤器忽略 numpy 的 "elementwise comparison failed" 警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    # 比较左右值是否有不同
                    if np.any(np.asarray(left_value != right_value)):
                        return False
                except TypeError as err:
                    # 处理某些特定的 TypeError
                    if "boolean value of NA is ambiguous" in str(err):
                        return False
                    raise
                except ValueError:
                    # 如果无法比较左右数组，则默认返回不等价
                    return False
    return True

# 比较两个 ArrayLike 对象是否在数据类型和内容上完全相等
def array_equals(left: ArrayLike, right: ArrayLike) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
    # 检查左右数组的数据类型是否相等
    if left.dtype != right.dtype:
        return False
    # 如果左侧数组属于 ABCExtensionArray 类型，则调用其 equals 方法比较是否相等
    elif isinstance(left, ABCExtensionArray):
        return left.equals(right)
    else:
        # 否则调用 array_equivalent 函数比较两个数组是否相等
        return array_equivalent(left, right, dtype_equal=True)

# 推断给定值的填充值，特别处理 NaN/NaT 值并返回正确的数据类型元素以提供正确的块构造
def infer_fill_value(val):
    """
    infer the fill value for the nan/NaT from the provided
    scalar/ndarray/list-like if we are a NaT, return the correct dtyped
    element to provide proper block construction
    """
    # 如果输入值不是类列表的对象，则转换为列表
    if not is_list_like(val):
        val = [val]
    # 将值转换为 numpy 数组
    val = np.asarray(val)
    # 如果值的数据类型属于日期时间类型（'m' 或 'M'），则返回一个 NaT 类型的数组元素
    if val.dtype.kind in "mM":
        return np.array("NaT", dtype=val.dtype)
    # 如果值的数据类型是 object 类型
    elif val.dtype == object:
        # 推断出值的确切数据类型，跳过 NaN 值
        dtype = lib.infer_dtype(ensure_object(val), skipna=False)
        # 如果推断出的数据类型是日期时间类型
        if dtype in ["datetime", "datetime64"]:
            # 返回一个 datetime64 类型的 NaT 数组
            return np.array("NaT", dtype=DT64NS_DTYPE)
        # 如果推断出的数据类型是时间间隔类型
        elif dtype in ["timedelta", "timedelta64"]:
            # 返回一个 timedelta64 类型的 NaT 数组
            return np.array("NaT", dtype=TD64NS_DTYPE)
        # 否则返回一个 object 类型的 NaN 数组
        return np.array(np.nan, dtype=object)
    
    # 如果值的数据类型的种类是 Unicode 字符串
    elif val.dtype.kind == "U":
        # 返回一个与值相同数据类型的 NaN 数组
        return np.array(np.nan, dtype=val.dtype)
    
    # 如果以上条件都不满足，则返回一个 NaN 值
    return np.nan
# 根据给定值和长度构造一个一维数组
def construct_1d_array_from_inferred_fill_value(
    value: object, length: int
) -> ArrayLike:
    # 导入必要的函数和模块
    from pandas.core.algorithms import take_nd
    from pandas.core.construction import sanitize_array
    from pandas.core.indexes.base import Index
    
    # 使用给定的值构造一个数组，并进行索引化处理，复制设置为False
    arr = sanitize_array(value, Index(range(1)), copy=False)
    # 创建一个长度为`length`，dtype为intp类型的数组，用于索引
    taker = -1 * np.ones(length, dtype=np.intp)
    # 返回通过索引操作后的数组
    return take_nd(arr, taker)


def maybe_fill(arr: np.ndarray) -> np.ndarray:
    """
    用NaN填充numpy.ndarray，除非数组的dtype是整数或布尔类型。
    """
    # 如果数组的dtype的种类不在'iub'中，则填充为NaN
    if arr.dtype.kind not in "iub":
        arr.fill(np.nan)
    return arr


def na_value_for_dtype(dtype: DtypeObj, compat: bool = True):
    """
    返回与dtype兼容的NA值

    Parameters
    ----------
    dtype : string / dtype
        数据类型
    compat : bool, default True
        是否兼容模式

    Returns
    -------
    np.dtype or a pandas dtype
        返回一个numpy或pandas的数据类型

    Examples
    --------
    >>> na_value_for_dtype(np.dtype("int64"))
    0
    >>> na_value_for_dtype(np.dtype("int64"), compat=False)
    nan
    >>> na_value_for_dtype(np.dtype("float64"))
    nan
    >>> na_value_for_dtype(np.dtype("complex128"))
    nan
    >>> na_value_for_dtype(np.dtype("bool"))
    False
    >>> na_value_for_dtype(np.dtype("datetime64[ns]"))
    numpy.datetime64('NaT')
    """

    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        unit = np.datetime_data(dtype)[0]
        return dtype.type("NaT", unit)
    elif dtype.kind in "fc":
        return np.nan
    elif dtype.kind in "iu":
        if compat:
            return 0
        return np.nan
    elif dtype.kind == "b":
        if compat:
            return False
        return np.nan
    return np.nan


def remove_na_arraylike(arr: Series | Index | np.ndarray):
    """
    返回只包含真值或非NaN值的数组，可能为空。

    Parameters
    ----------
    arr : Series | Index | np.ndarray
        一个类似数组，可以是Series、Index或numpy数组

    Returns
    -------
    array-like
        返回只包含真值或非NaN值的数组
    """
    # 如果数组的dtype是ExtensionDtype，则返回不是NaN的值
    if isinstance(arr.dtype, ExtensionDtype):
        return arr[notna(arr)]
    else:
        return arr[notna(np.asarray(arr))]


def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool:
    """
    isna检查，排除不兼容的dtype

    Parameters
    ----------
    obj : object
        对象
    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype
        数据类型

    Returns
    -------
    bool
        返回布尔值
    """
    # 如果obj不是标量或不是NaN，则返回False
    if not lib.is_scalar(obj) or not isna(obj):
        return False
    elif dtype.kind == "M":
        if isinstance(dtype, np.dtype):
            # 即不是时区感知的
            return not isinstance(obj, (np.timedelta64, Decimal))
        # 必须排除tznaive dt64("NaT")
        return not isinstance(obj, (np.timedelta64, np.datetime64, Decimal))
    elif dtype.kind == "m":
        return not isinstance(obj, (np.datetime64, Decimal))
    elif dtype.kind in "iufc":
        # 数字类型
        return obj is not NaT and not isinstance(obj, (np.datetime64, np.timedelta64))
    elif dtype.kind == "b":
        # 对于布尔数组（BooleanArray），允许 pd.NA、None、np.nan（与IntervalDtype相同）
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif dtype == _dtype_str:
        # 对于 numpy 的字符串 dtype，避免处理 float 类型的 np.nan
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal, float))

    elif dtype == _dtype_object:
        # 这对于分类数据（Categorical）是必要的，但有点奇怪
        return True

    elif isinstance(dtype, PeriodDtype):
        # 对于 PeriodDtype 类型的数据，不是 np.datetime64、np.timedelta64、Decimal 类型的对象才返回 True
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))

    elif isinstance(dtype, IntervalDtype):
        # 对于 IntervalDtype 类型，允许 lib.is_float(obj) 为 True，或者 obj 是 None 或 libmissing.NA
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif isinstance(dtype, CategoricalDtype):
        # 对于分类数据类型（CategoricalDtype），检查 obj 是否是有效的 NaN（适合 dtype.categories.dtype）
        return is_valid_na_for_dtype(obj, dtype.categories.dtype)

    # 回退情况，默认允许 NaN、None、NA、NaT
    return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))
`
def isna_all(arr: ArrayLike) -> bool:
    """
    Optimized equivalent to isna(arr).all()
    """
    total_len = len(arr)  # 获取数组的总长度

    # 通常只需检查少数值是否为空即可确定块是否为空，分块有助于此类情况。
    # 参数 1000 和 40 是任意选择的
    chunk_len = max(total_len // 40, 1000)  # 根据总长度计算块的大小，保证块的最小长度为 1000

    dtype = arr.dtype  # 获取数组的数据类型
    if lib.is_np_dtype(dtype, "f"):  # 如果数据类型是浮点类型
        checker = np.isnan  # 使用 numpy 的 isnan 函数

    elif (lib.is_np_dtype(dtype, "mM")) or isinstance(
        dtype, (DatetimeTZDtype, PeriodDtype)
    ):  # 如果数据类型是 datetime 或 period 类型
        checker = lambda x: np.asarray(x.view("i8")) == iNaT  # 将数据视为 64 位整数数组，并判断是否为 iNaT

    else:  # 其他数据类型使用 _isna_array 函数进行检查
        checker = _isna_array  # type: ignore[assignment]

    # 遍历数组，将数组分块检查每一块是否全部为 NaN 值
    return all(
        checker(arr[i : i + chunk_len]).all() for i in range(0, total_len, chunk_len)
    )
```