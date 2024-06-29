# `D:\src\scipysrc\pandas\pandas\core\algorithms.py`

```
"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""

from __future__ import annotations

import decimal
import operator
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    algos,
    hashtable as htable,
    iNaT,
    lib,
)
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    ArrayLikeT,
    AxisInt,
    DtypeObj,
    TakeIndexer,
    npt,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    np_find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_object,
    ensure_platform_int,
    is_bool_dtype,
    is_complex_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_signed_integer_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    CategoricalDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.generic import (
    ABCDatetimeArray,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from pandas._typing import (
        ListLike,
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import (
        Categorical,
        Index,
        Series,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        ExtensionArray,
    )


# --------------- #
# dtype access    #
# --------------- #

# 确保数据的函数，根据输入的类型转换为正确的数据类型
def _ensure_data(values: ArrayLike) -> np.ndarray:
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint8
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    np.ndarray
    """

    # 如果不是多级索引，使用 extract_array 提取数组，转换为 NumPy 数组
    if not isinstance(values, ABCMultiIndex):
        values = extract_array(values, extract_numpy=True)

    # 如果数据类型是对象型，则转换为对象数组
    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values))
    elif isinstance(values.dtype, BaseMaskedDtype):
        # 如果数据类型是基于 BaseMaskedDtype 的子类，例如 BooleanArray、FloatingArray、IntegerArray
        values = cast("BaseMaskedArray", values)
        if not values._hasna:
            # 如果没有缺失值 pd.NA，则可以避免使用对象数据类型转换（和复制），GH#41816
            # 递归调用以避免重新实现例如 bool->uint8 的逻辑
            return _ensure_data(values._data)
        return np.asarray(values)

    elif isinstance(values.dtype, CategoricalDtype):
        # 注意：通过此处的情况不应使用 _reconstruct_data 在后端。
        values = cast("Categorical", values)
        return values.codes

    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            # 即数据类型为 np.dtype("bool")
            return np.asarray(values).view("uint8")
        else:
            # 例如 Sparse[bool, False]  # TODO: 没有测试案例会到达此处
            return np.asarray(values).astype("uint8", copy=False)

    elif is_integer_dtype(values.dtype):
        return np.asarray(values)

    elif is_float_dtype(values.dtype):
        # 注意：检查 `values.dtype == "float128"` 在 Windows 和 32 位系统上会报错
        # 错误：Item "ExtensionDtype" of "Union[Any, ExtensionDtype, dtype[Any]]"
        # 没有属性 "itemsize"
        if values.dtype.itemsize in [2, 12, 16]:  # type: ignore[union-attr]
            # 我们暂时没有 float128 的哈希表支持
            return ensure_float64(values)
        return np.asarray(values)

    elif is_complex_dtype(values.dtype):
        # 如果数据类型是复数类型
        return cast(np.ndarray, values)

    # 如果是类似日期时间的数据类型
    elif needs_i8_conversion(values.dtype):
        npvalues = values.view("i8")
        npvalues = cast(np.ndarray, npvalues)
        return npvalues

    # 如果以上条件都不满足，则返回对象数据类型
    values = np.asarray(values, dtype=object)
    return ensure_object(values)
# 重新构造数据，是 _ensure_data 的逆过程

def _reconstruct_data(
    values: ArrayLikeT, dtype: DtypeObj, original: AnyArrayLike
) -> ArrayLikeT:
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        要处理的数据，可以是 NumPy 数组或扩展数组
    dtype : np.dtype or ExtensionDtype
        数据的目标数据类型，可以是 NumPy 数据类型或扩展数据类型
    original : AnyArrayLike
        原始数据

    Returns
    -------
    ExtensionArray or np.ndarray
        处理后的数组或扩展数组
    """
    if isinstance(values, ABCExtensionArray) and values.dtype == dtype:
        # 捕获 DatetimeArray/TimedeltaArray
        return values

    if not isinstance(dtype, np.dtype):
        # 即 ExtensionDtype；注意上面已经排除了 values.dtype == dtype 的可能性
        cls = dtype.construct_array_type()

        # 错误：在赋值时类型不兼容（表达式类型为 "ExtensionArray"，变量类型为 "ndarray[Any, Any]"）
        values = cls._from_sequence(values, dtype=dtype)  # type: ignore[assignment]

    else:
        values = values.astype(dtype, copy=False)

    return values


def _ensure_arraylike(values, func_name: str) -> ArrayLike:
    """
    ensure that we are arraylike if not already

    Parameters
    ----------
    values : any
        要检查的值
    func_name : str
        调用该函数的函数名

    Returns
    -------
    ArrayLike
        数组或类似数组的值
    """
    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        # GH#52986
        if func_name != "isin-targets":
            # 对于 isin 中的 comps 参数，做出异常处理
            raise TypeError(
                f"{func_name} requires a Series, Index, "
                f"ExtensionArray, or np.ndarray, got {type(values).__name__}."
            )

        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ["mixed", "string", "mixed-integer"]:
            # 为确保不将 ["ss", 42] 强制转换为字符串，将 "mixed-integer" 确保为 str GH#22160
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    "complex128": htable.Complex128HashTable,
    "complex64": htable.Complex64HashTable,
    "float64": htable.Float64HashTable,
    "float32": htable.Float32HashTable,
    "uint64": htable.UInt64HashTable,
    "uint32": htable.UInt32HashTable,
    "uint16": htable.UInt16HashTable,
    "uint8": htable.UInt8HashTable,
    "int64": htable.Int64HashTable,
    "int32": htable.Int32HashTable,
    "int16": htable.Int16HashTable,
    "int8": htable.Int8HashTable,
    "string": htable.StringHashTable,
    "object": htable.PyObjectHashTable,
}


def _get_hashtable_algo(
    values: np.ndarray,
) -> tuple[type[htable.HashTable], np.ndarray]:
    """
    Parameters
    ----------
    values : np.ndarray
        输入的 NumPy 数组

    Returns
    -------
    tuple[type[htable.HashTable], np.ndarray]
        返回哈希表的子类和处理后的数组
    """
    values = _ensure_data(values)

    ndtype = _check_object_for_strings(values)
    hashtable = _hashtables[ndtype]
    return hashtable, values


def _check_object_for_strings(values: np.ndarray) -> str:
    """
    Parameters
    ----------
    values : np.ndarray
        输入的 NumPy 数组

    Returns
    -------
    str
        数组的数据类型，用于获取对应的哈希表
    """
    # 检查是否可以使用字符串哈希表而不是对象哈希表。
    # 
    # 参数：
    # values : ndarray
    #     输入的数组对象
    # 
    # 返回：
    # str
    #     返回值的数据类型名称
    ndtype = values.dtype.name
    
    # 如果数据类型是对象类型
    if ndtype == "object":
        # 通过调用 lib.is_string_array(values, skipna=False) 函数来确定是否可以使用字符串哈希表
        if lib.is_string_array(values, skipna=False):
            # 如果可以使用字符串哈希表，更新数据类型为 "string"
            ndtype = "string"
    
    # 返回最终确定的数据类型名称
    return ndtype
# --------------- #
# top-level algos #
# --------------- #


def unique(values):
    """
    Return unique values based on a hash table.

    Uniques are returned in order of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like
        The input array-like object containing values from which to extract
        unique values.

    Returns
    -------
    numpy.ndarray or ExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray or ExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00'], dtype='datetime64[s]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(np.array(list("baabc"), dtype="O"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique(pd.Series([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]).values)
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """
    # 调用内部函数 unique_with_mask 处理唯一值的逻辑
    return unique_with_mask(values)


def nunique_ints(values: ArrayLike) -> int:
    """
    Placeholder function for computing the number of unique integers.

    Parameters
    ----------
    values : ArrayLike
        Input array-like object (e.g., list, ndarray) containing integers.

    Returns
    -------
    int
        Number of unique integers in the input array.

    Notes
    -----
    This function aims to compute the number of unique integers efficiently.

    Examples
    --------
    >>> nunique_ints([1, 2, 3, 2, 1])
    3

    >>> nunique_ints([1, 1, 1, 1])
    1
    """
    Return the number of unique values for integer array-likes.

    Significantly faster than pandas.unique for long enough sequences.
    No checks are done to ensure input is integral.

    Parameters
    ----------
    values : 1d array-like
        输入参数：一个一维数组，包含整数类型的数据。

    Returns
    -------
    int : The number of unique values in ``values``
        返回值：``values`` 中唯一值的数量。
    """
    # 如果输入数组长度为0，直接返回0
    if len(values) == 0:
        return 0
    # 确保数据是整数类型
    values = _ensure_data(values)
    # 使用 np.bincount 统计整数出现的次数，然后计算非零项的数量，即唯一值的数量
    result = (np.bincount(values.ravel().astype("intp")) != 0).sum()
    # 返回唯一值的数量
    return result
# 确保输入的值转换为可处理的数组形式，如果是扩展类型，则返回其唯一值
def unique_with_mask(values, mask: npt.NDArray[np.bool_] | None = None):
    """See algorithms.unique for docs. Takes a mask for masked arrays."""
    values = _ensure_arraylike(values, func_name="unique")

    # 如果值的数据类型是扩展类型，则调用扩展类型的唯一值方法
    if isinstance(values.dtype, ExtensionDtype):
        return values.unique()

    # 如果值是索引类型，则调用索引的唯一值方法
    if isinstance(values, ABCIndex):
        return values.unique()

    # 将原始值备份，然后获取适合哈希表的算法和处理后的值
    original = values
    hashtable, values = _get_hashtable_algo(values)

    # 创建哈希表，并根据是否存在掩码来进行唯一值的计算
    table = hashtable(len(values))
    if mask is None:
        # 如果没有掩码，计算值的唯一值
        uniques = table.unique(values)
        # 重建数据以匹配原始数据类型，并返回唯一值数组
        uniques = _reconstruct_data(uniques, original.dtype, original)
        return uniques

    else:
        # 如果存在掩码，根据掩码计算唯一值和掩码
        uniques, mask = table.unique(values, mask=mask)
        # 重建数据以匹配原始数据类型，并确保掩码不为空
        uniques = _reconstruct_data(uniques, original.dtype, original)
        assert mask is not None  # for mypy
        # 返回唯一值数组和布尔类型的掩码数组
        return uniques, mask.astype("bool")


# 将 unique1d 定义为 unique 函数的别名
unique1d = unique


# 定义用于比较的数组的最小长度
_MINIMUM_COMP_ARR_LEN = 1_000_000


# 计算是否在指定数组中存在的布尔值数组
def isin(comps: ListLike, values: ListLike) -> npt.NDArray[np.bool_]:
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : list-like
    values : list-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    # 如果 comps 不是类似列表的对象，则抛出类型错误
    if not is_list_like(comps):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(comps).__name__}`"
        )
    # 如果 values 不是类似列表的对象，则抛出类型错误
    if not is_list_like(values):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(values).__name__}`"
        )

    # 如果 values 不是索引、系列、扩展数组或 NumPy 数组类型，则将其转换为数组
    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        orig_values = list(values)
        values = _ensure_arraylike(orig_values, func_name="isin-targets")

        # 如果 values 的长度大于 0，且数据类型适合使用整数、浮点数、布尔或字符，则使用对象数组避免后续上浮到 float64
        if (
            len(values) > 0
            and values.dtype.kind in "iufcb"
            and not is_signed_integer_dtype(comps)
        ):
            values = construct_1d_object_array_from_listlike(orig_values)

    # 如果 values 是多重索引类型，则将其转换为 NumPy 数组
    elif isinstance(values, ABCMultiIndex):
        values = np.array(values)
    else:
        # 否则，使用 extract_array 函数将其转换为 NumPy 数组形式，并确保提取到 NumPy 数组及其范围
        values = extract_array(values, extract_numpy=True, extract_range=True)

    # 确保 comps 被转换为可处理的数组形式，并使用 extract_array 函数将其转换为 NumPy 数组
    comps_array = _ensure_arraylike(comps, func_name="isin")
    comps_array = extract_array(comps_array, extract_numpy=True)

    # 如果 comps_array 不是 NumPy 数组类型，则返回扩展数组的 isin 方法结果
    if not isinstance(comps_array, np.ndarray):
        return comps_array.isin(values)

    # 如果 comps_array 需要转换为 int64 类型，则调用 DatetimeLikeArrayMixin.isin 方法
    elif needs_i8_conversion(comps_array.dtype):
        return pd_array(comps_array).isin(values)
    # 如果需要将values的dtype转换为int64，并且comps_array的dtype不是对象类型
    elif needs_i8_conversion(values.dtype) and not is_object_dtype(comps_array.dtype):
        # 返回一个与comps_array形状相同的全零数组，数据类型为布尔型
        return np.zeros(comps_array.shape, dtype=bool)
        # TODO: 这里还不够准确...稀疏/分类数据
    # 如果需要将values的dtype转换为int64
    elif needs_i8_conversion(values.dtype):
        # 将values转换为对象类型后，使用np.isin检查comps_array中是否存在这些值
        return isin(comps_array, values.astype(object))

    # 如果values的dtype是ExtensionDtype的实例
    elif isinstance(values.dtype, ExtensionDtype):
        # 将comps_array和values都转换为数组后，使用np.isin检查它们的交集
        return isin(np.asarray(comps_array), np.asarray(values))

    # GH16012
    # 确保np.isin不处理对象类型，否则可能会抛出异常
    # 虽然哈希映射的查找时间复杂度为O(1)（与排序数组的O(logn)相比），
    # 但对于小数据量而言，np.isin更快
    if (
        len(comps_array) > _MINIMUM_COMP_ARR_LEN
        and len(values) <= 26
        and comps_array.dtype != object
    ):
        # 如果values包含nan，需要显式检查nan，因为np.nan不等于np.nan
        if isna(values).any():

            def f(c, v):
                # 返回逻辑或结果，包括np.isin的结果和是否为nan的检查
                return np.logical_or(np.isin(c, v).ravel(), np.isnan(c))

        else:
            # 返回np.isin的结果，展平为一维数组
            f = lambda a, b: np.isin(a, b).ravel()

    else:
        # 寻找values和comps_array的公共数据类型
        common = np_find_common_type(values.dtype, comps_array.dtype)
        # 将values和comps_array转换为公共数据类型，不复制数据
        values = values.astype(common, copy=False)
        comps_array = comps_array.astype(common, copy=False)
        # 使用htable.ismember函数来处理comps_array和values的成员关系
        f = htable.ismember

    # 返回函数f对comps_array和values的处理结果
    return f(comps_array, values)
# 将输入的 numpy 数组进行因子化，返回因子化后的代码和唯一值数组

def factorize_array(
    values: np.ndarray,
    use_na_sentinel: bool = True,  # 是否使用 NaN 哨兵值，默认为 True
    size_hint: int | None = None,   # 可选的哈希表大小提示
    na_value: object = None,        # 可选的缺失值标识
    mask: npt.NDArray[np.bool_] | None = None,  # 可选的缺失值掩码
) -> tuple[npt.NDArray[np.intp], np.ndarray]:
    """
    Factorize a numpy array to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
        输入的 numpy 数组
    use_na_sentinel : bool, default True
        是否使用 NaN 哨兵值。若为 True，则使用 -1 表示 NaN 值；若为 False，
        NaN 值将编码为非负整数，并且不会从 uniques 数组中删除 NaN。
    size_hint : int, optional
        传递给哈希表的 'get_labels' 方法的大小提示
    na_value : object, optional
        在 `values` 中表示缺失的值。注意：只有在知道数组中没有任何 pandas 认为
        的缺失值（浮点数据的 NaN、日期时间的 iNaT 等）时才使用此参数。
    mask : ndarray[bool], optional
        如果不是 None，则将掩码用作缺失值的指示器（True = 缺失，False = 有效），而不是
        使用 `na_value` 或条件 "val != val"。

    Returns
    -------
    codes : ndarray[np.intp]
        编码后的整数数组
    uniques : ndarray
        唯一值数组
    """
    original = values
    if values.dtype.kind in "mM":
        # _get_hashtable_algo 将通过 _ensure_data 将 dt64/td64 转换为 i8，因此
        # 我们需要对 na_value 进行相同处理。这里假设传递的 na_value 是适当类型的 NaT。
        # 例如：test_where_datetimelike_categorical
        na_value = iNaT

    # 获取哈希表算法和处理后的值
    hash_klass, values = _get_hashtable_algo(values)

    # 创建哈希表并进行因子化
    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(
        values,
        na_sentinel=-1 if use_na_sentinel else 0,  # 根据 use_na_sentinel 决定 NaN 哨兵值
        na_value=na_value,
        mask=mask,
        ignore_na=use_na_sentinel,
    )

    # 重新转换数据类型，例如 i8->dt64/td64, uint8->bool
    uniques = _reconstruct_data(uniques, original.dtype, original)

    # 确保 codes 适应平台的整数类型
    codes = ensure_platform_int(codes)
    return codes, uniques


@doc(
    values=dedent(
        """\
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
        """
    ),
    sort=dedent(
        """\
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.
        """
    ),
    size_hint=dedent(
        """\
    size_hint : int, optional
        Hint to the hashtable sizer.
        """
    ),
)
def factorize(
    values,
    sort: bool = False,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray | Index]:
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`

    Parameters
    ----------
    values : sequence
        一个一维序列。在进行因子化之前，不是 pandas 对象的序列将被强制转换为 ndarrays。
    sort : bool, default False
        是否对 `uniques` 进行排序，并对 `codes` 进行洗牌以维护二者的关系。
    use_na_sentinel : bool, default True
        是否使用 NaN 哨兵值。默认为 True。
    size_hint : int, optional
        传递给哈希表的大小提示。

    Returns
    -------
    codes : ndarray
        编码后的整数数组
    uniques : ndarray or Index
        唯一值数组或索引
    """
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    {values}{sort}
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.

        .. versionadded:: 1.5.0
    {size_hint}\

    Returns
    -------
    codes : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(codes)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.factorize>` for more examples.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> codes, uniques = pd.factorize(np.array(['b', 'b', 'a', 'c', 'b'], dtype="O"))
    >>> codes
    array([0, 0, 1, 2, 0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `codes` will be
    shuffled so that the relationship is the maintained.

    >>> codes, uniques = pd.factorize(np.array(['b', 'b', 'a', 'c', 'b'], dtype="O"),
    ...                               sort=True)
    >>> codes
    array([1, 1, 0, 2, 1])
    >>> uniques
    array(['a', 'b', 'c'], dtype=object)

    When ``use_na_sentinel=True`` (the default), missing values are indicated in
    the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = pd.factorize(np.array(['b', None, 'a', 'c', 'b'], dtype="O"))
    >>> codes
    array([ 0, -1,  1,  2,  0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = pd.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1])
    >>> uniques
    ['a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.
    """
    # 实现说明：该方法负责三件事情
    # 1.) 将数据强制转换为类数组（ndarray、Index、扩展数组）
    # 2.) 对代码和唯一值进行因子化处理
    # 3.) 可能会将唯一值装箱到一个 Index 中
    #
    # 第二步被分派给扩展类型（如分类）。它们仅负责因子化处理。所有数据强制转换、排序和装箱
    # 都应该在这里处理。
    
    如果 values 是 ABCIndex 或 ABCSeries 的实例，则调用其 factorize 方法处理。
    返回值：codes 和 uniques。
    
    否则，将 values 转换为类数组（通过 _ensure_arraylike 函数），函数名称为 "factorize"。
    
    如果 values 是 ABCDatetimeArray 或 ABCTimedeltaArray 的实例，并且具有 freq 属性（频率不为 None），
    则使用 values 的 factorize 方法进行处理，设置 sort=sort。
    返回值：codes 和 uniques。
    
    否则，如果 values 不是 np.ndarray 类型：
    即扩展数组类型，调用其 factorize 方法处理，设置 use_na_sentinel=use_na_sentinel。
    返回值：codes 和 uniques。
    
    否则，将 values 转换为 np.ndarray 类型。
    如果不使用 NA 标志且 values 的 dtype 是 object 类型：
    factorize 现在可以处理区分各种类型的 null 值。
    这些只能出现在数组具有 object dtype 时。
    但是，为了向后兼容，我们仅对所提供的 dtype 使用 null 值。
    未来可能会重新考虑这个问题，请参见 GH#48476。
    返回值：codes 和 uniques。
    
    如果 sort 为真且 uniques 的长度大于 0：
    对 uniques 和 codes 进行安全排序。
    设置 use_na_sentinel=use_na_sentinel，assume_unique=True，verify=False。
    """
    # 使用 _reconstruct_data 函数重新构造 uniques 对象，传入原始数据类型 original.dtype 和原始数据 original
    uniques = _reconstruct_data(uniques, original.dtype, original)

    # 返回 codes 和重新构造后的 uniques 对象
    return codes, uniques
# 定义一个函数 value_counts_internal，用于计算给定数据的值计数或分组统计
def value_counts_internal(
    values,  # 输入的数据，可以是各种类型，如 Series、Categorical 或数组
    sort: bool = True,  # 是否对结果进行排序，默认为 True
    ascending: bool = False,  # 排序时是否升序，默认为 False
    normalize: bool = False,  # 是否返回相对频率而不是计数，默认为 False
    bins=None,  # 指定分组的区间或数量，用于数值数据的分组统计
    dropna: bool = True,  # 是否忽略缺失值，默认为 True
) -> Series:  # 返回一个 Series 对象

    from pandas import (  # 导入 pandas 库中的一些模块和类
        Index,  # 索引类
        Series,  # Series 类
    )

    index_name = getattr(values, "name", None)  # 获取 values 对象的名称作为索引名称，如果没有则为 None
    name = "proportion" if normalize else "count"  # 如果 normalize 为 True，则使用 "proportion"，否则使用 "count"

    if bins is not None:  # 如果指定了 bins 参数
        from pandas.core.reshape.tile import cut  # 从 pandas 中导入 cut 函数，用于数值的分组切割

        if isinstance(values, Series):  # 如果 values 是一个 Series 类型
            values = values._values  # 获取其底层的数据数组

        try:
            ii = cut(values, bins, include_lowest=True)  # 将数据切割成 bins 指定的区间
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err
            # 捕获可能的 TypeError 异常，并抛出更具体的异常信息

        # 对切割后的结果进行值计数，根据 dropna 参数删除空值索引，设置结果的名称为 name
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]  # 过滤掉空值的索引
        result.index = result.index.astype("interval")  # 将索引转换为 interval 类型
        result = result.sort_index()  # 根据索引排序

        # 如果 dropna 为 True 且结果中所有值均为 0，则返回一个空的结果 Series
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # 根据 ii 的长度生成一个 counts 数组
        counts = np.array([len(ii)])

    else:  # 如果没有指定 bins 参数
        if is_extension_array_dtype(values):  # 如果 values 是扩展数组类型
            # 处理 Categorical 和稀疏类型数据，直接调用 value_counts 方法进行统计
            result = Series(values, copy=False)._values.value_counts(dropna=dropna)
            result.name = name
            result.index.name = index_name
            counts = result._values
            if not isinstance(counts, np.ndarray):
                # 如果 counts 不是 ndarray 类型，则转换为 ndarray 类型
                counts = np.asarray(counts)

        elif isinstance(values, ABCMultiIndex):  # 如果 values 是 ABCMultiIndex 类型
            # 处理多级索引数据，根据每个级别进行分组统计
            levels = list(range(values.nlevels))
            result = (
                Series(index=values, name=name)
                .groupby(level=levels, dropna=dropna)
                .size()
            )
            result.index.names = values.names  # 设置结果的索引名称
            counts = result._values

        else:  # 否则，将 values 转换为数组，并调用 value_counts_arraylike 函数进行处理
            values = _ensure_arraylike(values, func_name="value_counts")
            keys, counts, _ = value_counts_arraylike(values, dropna)
            if keys.dtype == np.float16:
                keys = keys.astype(np.float32)

            # 在 3.0 版本中，不再对构建的 Index 对象执行 dtype 推断
            idx = Index(keys, dtype=keys.dtype, name=index_name)
            result = Series(counts, index=idx, name=name, copy=False)

    if sort:  # 如果 sort 参数为 True，则根据 ascending 参数对结果进行排序
        result = result.sort_values(ascending=ascending)

    if normalize:  # 如果 normalize 参数为 True，则将结果进行归一化处理
        result = result / counts.sum()

    return result  # 返回处理后的结果 Series 对象


# 从 SparseArray 调用一次，否则可能是私有函数
def value_counts_arraylike(
    values: np.ndarray,  # 输入的数据数组
    dropna: bool,  # 是否忽略缺失值
    mask: npt.NDArray[np.bool_] | None = None  # 可选的布尔掩码数组，默认为 None
) -> tuple[ArrayLike, npt.NDArray[np.int64], int]:  # 返回一个元组，包含键、计数和长度

    """
    Parameters
    ----------
    values : np.ndarray  # 输入的数据数组
    dropna : bool  # 是否忽略缺失值

    """
    # 定义一个函数或方法的参数，mask 是一个 NumPy 数组，可以是布尔类型的数组或者 None，默认为 None
    # 返回值说明：返回两个 NumPy 数组，分别是 uniques 和 counts
    """
    # 将 values 复制给 original 变量，作为备份
    original = values
    # 调用 _ensure_data 函数确保 values 变量的数据格式正确
    values = _ensure_data(values)

    # 调用 htable.value_count 函数统计 values 中各元素出现的次数，并返回统计结果
    # keys 是元素值的数组，counts 是对应元素值的计数数组，na_counter 是 NaN 值的计数器
    keys, counts, na_counter = htable.value_count(values, dropna, mask=mask)

    # 如果原始数据类型需要转换为 np.int64 类型（通常是 datetime、timedelta 或 period 类型）
    if needs_i8_conversion(original.dtype):
        # 如果 dropna 为 True，则需要进行 NaN 值的处理
        if dropna:
            # 创建一个掩码数组，用于过滤掉 keys 中的 NaN 值
            mask = keys != iNaT
            keys, counts = keys[mask], counts[mask]

    # 调用 _reconstruct_data 函数重构 keys 数组，以保证其数据类型与 original 一致
    res_keys = _reconstruct_data(keys, original.dtype, original)
    # 返回重构后的 keys 数组、counts 数组以及 NaN 值的计数器
    return res_keys, counts, na_counter
# 返回一个布尔型的 ndarray，指示数组中的重复值
def duplicated(
    values: ArrayLike,
    keep: Literal["first", "last", False] = "first",
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """
    返回一个布尔型的 ndarray，指示数组中的重复值。

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        需要检查重复值的数组。
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : 除第一次出现外，将重复值标记为 ``True``。
        - ``last`` : 除最后一次出现外，将重复值标记为 ``True``。
        - False : 将所有重复值标记为 ``True``。
    mask : ndarray[bool], optional
        指示哪些元素不参与检查的数组。

    Returns
    -------
    duplicated : ndarray[bool]
    """
    # 确保数据类型正确
    values = _ensure_data(values)
    # 调用htable模块的duplicated函数进行重复值检查
    return htable.duplicated(values, keep=keep, mask=mask)


# 返回一个数组的众数（可能有多个）
def mode(
    values: ArrayLike, dropna: bool = True, mask: npt.NDArray[np.bool_] | None = None
) -> ArrayLike:
    """
    返回数组的众数（可能有多个）。

    Parameters
    ----------
    values : array-like
        需要检查众数的数组。
    dropna : bool, default True
        是否忽略 NaN/NaT 值的计数。

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    # 确保数组类型正确
    values = _ensure_arraylike(values, func_name="mode")
    original = values

    # 如果需要将数据类型转换为i8（日期时间类型），则处理特定情况
    if needs_i8_conversion(values.dtype):
        # 如果是ndarray，则转发到DatetimeArray/TimedeltaArray处理
        values = ensure_wrapped_if_datetimelike(values)
        values = cast("ExtensionArray", values)
        return values._mode(dropna=dropna)

    # 确保数据类型正确
    values = _ensure_data(values)

    # 调用htable模块的mode函数获取众数及其掩码（如果有）
    npresult, res_mask = htable.mode(values, dropna=dropna, mask=mask)
    if res_mask is not None:
        return npresult, res_mask  # type: ignore[return-value]

    # 尝试对结果进行排序（如果可能）
    try:
        npresult = np.sort(npresult)
    except TypeError as err:
        # 如果排序失败，则发出警告
        warnings.warn(
            f"Unable to sort modes: {err}",
            stacklevel=find_stack_level(),
        )

    # 重构结果数据
    result = _reconstruct_data(npresult, original.dtype, original)
    return result


# 返回沿指定轴排名后的值
def rank(
    values: ArrayLike,
    axis: AxisInt = 0,
    method: str = "average",
    na_option: str = "keep",
    ascending: bool = True,
    pct: bool = False,
) -> npt.NDArray[np.float64]:
    """
    返回沿指定轴排名后的值。

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        需要排名的数组。该数组的维度不能超过2。
    axis : int, default 0
        执行排名的轴。
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        在排名时用于解决并列的方法。
    na_option : {'keep', 'top'}, default 'keep'
        # NaN 值处理方式选项，缺省为 'keep'
        # - ``keep``: 每个 NaN 值都保持为 NaN 排名
        # - ``top``: 将每个 NaN 替换为 +/- 无穷大，使它们排在顶部
    ascending : bool, default True
        # 元素排名是否按升序排列
    pct : bool, default False
        # 是否以整数形式（如 1, 2, 3）或百分位形式（如 0.333..., 0.666..., 1）显示返回的排名
    """
    # 检查数据是否需要转换为 datetime 类型
    is_datetimelike = needs_i8_conversion(values.dtype)
    # 确保数据格式正确
    values = _ensure_data(values)

    # 如果数据是一维的
    if values.ndim == 1:
        ranks = algos.rank_1d(
            values,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    # 如果数据是二维的
    elif values.ndim == 2:
        ranks = algos.rank_2d(
            values,
            axis=axis,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    else:
        # 抛出异常，不支持超过二维的数组
        raise TypeError("Array with ndim > 2 are not supported.")

    # 返回计算得到的排名
    return ranks
# ---- #
# take #
# ---- #

def take(
    arr,
    indices: TakeIndexer,
    axis: AxisInt = 0,
    allow_fill: bool = False,
    fill_value=None,
):
    """
    Take elements from an array.

    Parameters
    ----------
    arr : array-like or scalar value
        Non array-likes (sequences/scalars without a dtype) are coerced
        to an ndarray.

        .. deprecated:: 2.1.0
            Passing an argument other than a numpy.ndarray, ExtensionArray,
            Index, or Series is deprecated.

    indices : sequence of int or one-dimensional np.ndarray of int
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> import pandas as pd

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> pd.api.extensions.take(
    ...     np.array([10, 20, 30]), [0, 0, -1], allow_fill=True, fill_value=-10
    ... )
    array([ 10,  10, -10])
    """
    # Check if the input array is of an acceptable type
    if not isinstance(arr, (np.ndarray, ABCExtensionArray, ABCIndex, ABCSeries)):
        # Raise TypeError if the input array is not one of the supported types
        # GH#52981
        raise TypeError(
            "pd.api.extensions.take requires a numpy.ndarray, "
            f"ExtensionArray, Index, or Series, got {type(arr).__name__}."
        )

    # Ensure indices are platform integer compatible
    indices = ensure_platform_int(indices)
    # 如果允许填充缺失值
    if allow_fill:
        # 使用 Pandas 风格，-1 表示缺失值
        # 验证索引的有效性，确保 indices 在指定轴上的合法性
        validate_indices(indices, arr.shape[axis])
        # 调用 take_nd 函数处理数组 arr 和 indices，获取处理后的结果
        # 错误: take_nd 的第一个参数类型不兼容，期望类型为 "ndarray[Any, Any]"
        result = take_nd(
            arr,  # type: ignore[arg-type]
            indices,  # indices 是要获取的索引
            axis=axis,  # 指定操作的轴
            allow_fill=True,  # 允许填充缺失值
            fill_value=fill_value,  # 填充缺失值的具体值
        )
    else:
        # 使用 NumPy 风格
        # 错误: ExtensionArray 的 take 方法不支持 "axis" 关键字参数
        # 使用 arr 的 take 方法获取指定索引的元素，返回处理后的结果
        result = arr.take(indices, axis=axis)  # type: ignore[call-arg,assignment]
    # 返回处理后的结果
    return result
# ------------ #
# searchsorted #
# ------------ #

# 定义一个名为 searchsorted 的函数，用于在数组中查找应插入元素以保持顺序的索引位置。

def searchsorted(
    arr: ArrayLike,
    value: NumpyValueArrayLike | ExtensionArray,
    side: Literal["left", "right"] = "left",
    sorter: NumpySorter | None = None,
) -> npt.NDArray[np.intp] | np.intp:
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `arr` (a) such that, if the
    corresponding elements in `value` were inserted before the indices,
    the order of `arr` would be preserved.

    Assuming that `arr` is sorted:

    ======  ================================
    `side`  returned index `i` satisfies
    ======  ================================
    left    ``arr[i-1] < value <= self[i]``
    right   ``arr[i-1] <= value < self[i]``
    ======  ================================

    Parameters
    ----------
    arr: np.ndarray, ExtensionArray, Series
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    value : array-like or scalar
        Values to insert into `arr`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `self`).
    sorter : 1-D array-like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    array of ints or int
        If value is array-like, array of insertion points.
        If value is scalar, a single integer.

    See Also
    --------
    numpy.searchsorted : Similar method from NumPy.
    """
    
    # 如果 `sorter` 不为 None，则确保其为平台整数类型
    if sorter is not None:
        sorter = ensure_platform_int(sorter)

    # 如果 `arr` 是 NumPy 数组，并且其数据类型为无符号整数或有符号整数，并且 `value` 是整数或者整数数据类型
    if (
        isinstance(arr, np.ndarray)
        and arr.dtype.kind in "iu"
        and (is_integer(value) or is_integer_dtype(value))
    ):
        # 如果 `arr` 和 `value` 的数据类型不同，NumPy 会重新转换 `arr`，导致搜索变慢。
        # 因此在进行下面的搜索之前，我们尝试将 `value` 转换为与 `arr` 相同的数据类型，
        # 同时防止整数溢出。
        iinfo = np.iinfo(arr.dtype.type)
        value_arr = np.array([value]) if is_integer(value) else np.array(value)
        if (value_arr >= iinfo.min).all() and (value_arr <= iinfo.max).all():
            # 如果 value 在范围内，没有溢出，可以将 value 的数据类型转换为 arr 的数据类型
            dtype = arr.dtype
        else:
            dtype = value_arr.dtype

        if is_integer(value):
            # 我们知道 value 是整数
            value = cast(int, dtype.type(value))
        else:
            value = pd_array(cast(ArrayLike, value), dtype=dtype)
    else:
        # 如果 `arr` 是一个数据类型为 'datetime64[ns]' 的数组，
        # 而 `value` 是一个 pd.Timestamp 类型，我们可能需要将 `value` 进行转换
        arr = ensure_wrapped_if_datetimelike(arr)

    # 调用数组 arr 的 searchsorted 方法来查找 value 的插入位置
    # 参数 1 为 "ndarray" 类型的对象，但其类型是 "Union[NumpyValueArrayLike, ExtensionArray]"，预期类型是 "NumpyValueArrayLike"
    return arr.searchsorted(value, side=side, sorter=sorter)  # type: ignore[arg-type]
# ---- #
# diff #
# ---- #

# 定义一组特殊的数据类型名称集合，用于特定处理
_diff_special = {"float64", "float32", "int64", "int32", "int16", "int8"}

# 定义了一个差分函数 diff，用于计算数组或扩展数组的差分
def diff(arr, n: int, axis: AxisInt = 0):
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray or ExtensionArray
        输入的数组或扩展数组
    n : int
        差分的周期数
    axis : {0, 1}
        差分操作的轴向，默认为0
    stacklevel : int, default 3
        用于丢失 dtype 警告的堆栈级别

    Returns
    -------
    shifted
    """

    # 对周期数 n 进行整数检查
    # 参考 https://github.com/pandas-dev/pandas/issues/56607
    if not lib.is_integer(n):
        if not (is_float(n) and n.is_integer()):
            raise ValueError("periods must be an integer")
        n = int(n)
    
    na = np.nan  # 定义缺失值为 NaN
    dtype = arr.dtype  # 获取输入数组的数据类型

    is_bool = is_bool_dtype(dtype)  # 检查是否为布尔类型
    if is_bool:
        op = operator.xor  # 如果是布尔类型，使用异或操作符
    else:
        op = operator.sub  # 否则使用减法操作符

    if isinstance(dtype, NumpyEADtype):
        # 如果输入数组是 NumpyExtensionArray 类型，则转换为普通的 numpy 数组
        arr = arr.to_numpy()
        dtype = arr.dtype

    if not isinstance(arr, np.ndarray):
        # 如果输入不是普通的 numpy 数组，则应该是扩展数组
        if hasattr(arr, f"__{op.__name__}__"):
            if axis != 0:
                raise ValueError(f"cannot diff {type(arr).__name__} on axis={axis}")
            return op(arr, arr.shift(n))  # 使用扩展数组的 shift 方法进行差分计算
        else:
            raise TypeError(
                f"{type(arr).__name__} has no 'diff' method. "
                "Convert to a suitable dtype prior to calling 'diff'."
            )

    is_timedelta = False
    if arr.dtype.kind in "mM":
        # 如果数组的数据类型是日期时间类型
        dtype = np.int64
        arr = arr.view("i8")  # 转换为 int64 类型的视图
        na = iNaT  # 设置缺失值为 NaT
        is_timedelta = True

    elif is_bool:
        # 需要强制转换以便能够存储 np.nan
        dtype = np.object_

    elif dtype.kind in "iu":
        # 需要强制转换以便能够存储 np.nan

        # int8、int16 类型与 float64 不兼容
        if arr.dtype.name in ["int8", "int16"]:
            dtype = np.float32
        else:
            dtype = np.float64

    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # 如果输入数组是一维的，则重新塑形为二维数组，便于后续计算
        arr = arr.reshape(-1, 1)
        # TODO: 要求 axis == 0 的处理方式

    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)  # 创建一个空数组，用于存放结果

    na_indexer = [slice(None)] * 2
    na_indexer[axis] = slice(None, n) if n >= 0 else slice(n, None)
    out_arr[tuple(na_indexer)] = na  # 根据指定的轴向和周期数，设置缺失值

    if arr.dtype.name in _diff_special:
        # TODO: 是否可以通过在 diff_2d 中定义 out_arr 来解决 dtype 特化问题
        algos.diff_2d(arr, out_arr, n, axis, datetimelike=is_timedelta)  # 使用特定算法计算差分
    # 如果条件不成立，则为了保持类型检查工具（如mypy）的兼容性，使用列表 _res_indexer 作为索引器，而 res_indexer 则使用元组。
    # 设置切片列表 _res_indexer，初始化为两个 None 切片
    _res_indexer = [slice(None)] * 2
    # 根据轴 axis 的正负决定切片的方向，更新 _res_indexer 中对应轴的切片
    _res_indexer[axis] = slice(n, None) if n >= 0 else slice(None, n)
    # 将 _res_indexer 转换为元组，得到 res_indexer
    res_indexer = tuple(_res_indexer)

    # 设置切片列表 _lag_indexer，初始化为两个 None 切片
    _lag_indexer = [slice(None)] * 2
    # 根据轴 axis 的正负决定切片的方向，更新 _lag_indexer 中对应轴的切片
    _lag_indexer[axis] = slice(None, -n) if n > 0 else slice(-n, None)
    # 将 _lag_indexer 转换为元组，得到 lag_indexer
    lag_indexer = tuple(_lag_indexer)

    # 使用切片操作，将 arr 上的操作 op 应用于 res_indexer 和 lag_indexer 所指定的区域，并存储结果到 out_arr
    out_arr[res_indexer] = op(arr[res_indexer], arr[lag_indexer])

if is_timedelta:
    # 如果结果是时间增量类型，则将 out_arr 视图转换为 timedelta64[ns] 类型
    out_arr = out_arr.view("timedelta64[ns]")

if orig_ndim == 1:
    # 如果原始数组的维度是1，则将 out_arr 调整为只有一列
    out_arr = out_arr[:, 0]
# 返回处理后的结果数组 out_arr
return out_arr
# --------------------------------------------------------------------
# Helper functions

# Note: safe_sort is in algorithms.py instead of sorting.py because it is
#  low-dependency, is used in this module, and used private methods from
#  this module.
def safe_sort(
    values: Index | ArrayLike,
    codes: npt.NDArray[np.intp] | None = None,
    use_na_sentinel: bool = True,
    assume_unique: bool = False,
    verify: bool = True,
) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]:
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : np.ndarray[intp] or None, default None
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``-1``.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to ``-1``. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

    Returns
    -------
    ordered : AnyArrayLike
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    # Check if values parameter is of acceptable types
    if not isinstance(values, (np.ndarray, ABCExtensionArray, ABCIndex)):
        raise TypeError(
            "Only np.ndarray, ExtensionArray, and Index objects are allowed to "
            "be passed to safe_sort as values"
        )

    sorter = None  # Initialize sorter variable

    # Perform special handling for mixed integer and string types
    if (
        not isinstance(values.dtype, ExtensionDtype)
        and lib.infer_dtype(values, skipna=False) == "mixed-integer"
    ):
        # Sort mixed integer types using a specific function
        ordered = _sort_mixed(values)
    else:
        try:
            # 尝试对数值进行排序
            sorter = values.argsort()
            # 根据排序后的索引重新排列数值
            ordered = values.take(sorter)
        except (TypeError, decimal.InvalidOperation):
            # 如果排序失败或不适用，尝试 `_sort_mixed`
            # 对于包含元组的一维数组，_sort_tuples 可能失败
            if values.size and isinstance(values[0], tuple):
                # 错误："_sort_tuples" 的参数 1 具有不兼容类型
                # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; 预期类型
                # "ndarray[Any, Any]"
                ordered = _sort_tuples(values)  # type: ignore[arg-type]
            else:
                ordered = _sort_mixed(values)

    # codes:
    
    if codes is None:
        # 如果没有提供 codes，则返回排序后的结果
        return ordered

    if not is_list_like(codes):
        # 如果 codes 不是类列表对象，则抛出类型错误
        raise TypeError(
            "Only list-like objects or None are allowed to "
            "be passed to safe_sort as codes"
        )
    codes = ensure_platform_int(np.asarray(codes))

    if not assume_unique and not len(unique(values)) == len(values):
        # 如果不假定唯一性，并且 values 不是唯一的，则抛出值错误
        raise ValueError("values should be unique if codes is not None")

    if sorter is None:
        # 混合类型情况下
        # 错误："_get_hashtable_algo" 的参数 1 具有不兼容类型
        # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; 预期类型
        # "ndarray[Any, Any]"
        hash_klass, values = _get_hashtable_algo(values)  # type: ignore[arg-type]
        t = hash_klass(len(values))
        t.map_locations(values)
        # 错误："HashTable" 的 "lookup" 方法的参数 1 具有不兼容类型
        # "ExtensionArray | ndarray[Any, Any] | Index | Series"; 预期类型 "ndarray"
        sorter = ensure_platform_int(t.lookup(ordered))  # type: ignore[arg-type]

    if use_na_sentinel:
        # take_nd 更快，但仅适用于 na_sentinels 为 -1 的情况
        order2 = sorter.argsort()
        if verify:
            mask = (codes < -len(values)) | (codes >= len(values))
            codes[mask] = 0
        else:
            mask = None
        new_codes = take_nd(order2, codes, fill_value=-1)
    else:
        reverse_indexer = np.empty(len(sorter), dtype=int)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        # 超出边界的索引将使用 `mode='wrap'` 掩码，因此在这里我们可以处理它们，而不会损失性能
        new_codes = reverse_indexer.take(codes, mode="wrap")

        if use_na_sentinel:
            mask = codes == -1
            if verify:
                mask = mask | (codes < -len(values)) | (codes >= len(values))

    if use_na_sentinel and mask is not None:
        np.putmask(new_codes, mask, -1)

    return ordered, ensure_platform_int(new_codes)
# 定义一个函数用于对混合类型的数组进行排序，按照整数、字符串、空值的顺序
def _sort_mixed(values) -> AnyArrayLike:
    # 创建一个布尔数组，标记数组中字符串类型的位置
    str_pos = np.array([isinstance(x, str) for x in values], dtype=bool)
    # 创建一个布尔数组，标记数组中空值的位置
    null_pos = np.array([isna(x) for x in values], dtype=bool)
    # 创建一个布尔数组，标记数组中整数类型的位置
    num_pos = ~str_pos & ~null_pos
    # 对字符串类型的元素进行排序，并返回排序后的索引
    str_argsort = np.argsort(values[str_pos])
    # 对整数类型的元素进行排序，并返回排序后的索引
    num_argsort = np.argsort(values[num_pos])
    # 将布尔数组转换为位置索引数组，然后按照对应的值排序
    str_locs = str_pos.nonzero()[0].take(str_argsort)
    num_locs = num_pos.nonzero()[0].take(num_argsort)
    # 找到空值的位置索引
    null_locs = null_pos.nonzero()[0]
    # 按照整数、字符串、空值的顺序连接它们的位置索引
    locs = np.concatenate([num_locs, str_locs, null_locs])
    # 按照排序好的位置索引取出元素并返回
    return values.take(locs)


# 定义一个函数用于对元组数组进行排序，保持列的独立性，因为包含不同类型和空值（nan）
def _sort_tuples(values: np.ndarray) -> np.ndarray:
    # 导入必要的函数
    from pandas.core.internals.construction import to_arrays
    from pandas.core.sorting import lexsort_indexer
    
    # 将元组数组转换为数组数组
    arrays, _ = to_arrays(values, None)
    # 使用lexsort_indexer函数对数组进行排序并返回排序后的索引
    indexer = lexsort_indexer(arrays, orders=True)
    # 返回按照排序后索引排序后的数组
    return values[indexer]


# 定义一个函数用于从lvals和rvals中提取联合值，处理重复项和空值
def union_with_duplicates(
    lvals: ArrayLike | Index, rvals: ArrayLike | Index
) -> ArrayLike | Index:
    """
    从lvals和rvals中提取联合值，处理重复项和空值

    Parameters
    ----------
    lvals: np.ndarray or ExtensionArray
        排在前面的左侧值。
    rvals: np.ndarray or ExtensionArray
        排在后面的右侧值。

    Returns
    -------
    np.ndarray or ExtensionArray
        包含两个数组的未排序联合值。

    Notes
    -----
    调用者负责确保lvals.dtype == rvals.dtype。
    """
    # 导入Series类
    from pandas import Series
    
    # 计算lvals中各值的计数，包括空值
    l_count = value_counts_internal(lvals, dropna=False)
    # 计算rvals中各值的计数，包括空值
    r_count = value_counts_internal(rvals, dropna=False)
    # 将l_count和r_count对齐，并用0填充缺失值
    l_count, r_count = l_count.align(r_count, fill_value=0)
    # 取每个位置上l_count和r_count中较大的值
    final_count = np.maximum(l_count.values, r_count.values)
    # 创建Series对象，使用final_count作为值，l_count的索引，int类型，不复制数据
    final_count = Series(final_count, index=l_count.index, dtype="int", copy=False)
    # 如果lvals和rvals都是多重索引，则将它们连接并找出唯一值
    if isinstance(lvals, ABCMultiIndex) and isinstance(rvals, ABCMultiIndex):
        unique_vals = lvals.append(rvals).unique()
    else:
        # 如果lvals是索引类型，则取其_values属性
        if isinstance(lvals, ABCIndex):
            lvals = lvals._values
        # 如果rvals是索引类型，则取其_values属性
        if isinstance(rvals, ABCIndex):
            rvals = rvals._values
        # 将lvals和rvals连接起来形成combined数组
        combined = concat_compat([lvals, rvals])  # type: ignore[list-item]
        # 找出combined数组中的唯一值
        unique_vals = unique(combined)
        # 如果唯一值是日期时间类型，则确保进行适当的包装
        unique_vals = ensure_wrapped_if_datetimelike(unique_vals)
    # 根据final_count数组中各值的重复次数，重复unique_vals中的元素，并返回结果数组
    repeats = final_count.reindex(unique_vals).values
    return np.repeat(unique_vals, repeats)


# 定义一个函数用于映射数组类型的参数
def map_array(
    arr: ArrayLike,
    mapper,
    na_action: Literal["ignore"] | None = None,


    # 定义函数的参数：mapper 用于映射操作，na_action 控制空值的处理方式，允许取值为 "ignore" 或 None
# 定义函数签名，指定返回类型为 np.ndarray、ExtensionArray 或 Index
def map_values(arr: np.ndarray | ExtensionArray | Index,
               mapper: Callable | dict | Series,
               na_action: Optional[str] = None) -> Union[np.ndarray, Index, ExtensionArray]:
    """
    Map values using an input mapping or function.

    Parameters
    ----------
    mapper : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.

    Returns
    -------
    Union[ndarray, Index, ExtensionArray]
        The output of the mapping function applied to the array.
        If the function returns a tuple with more than one element
        a MultiIndex will be returned.
    """
    # 检查 na_action 参数是否为有效值
    if na_action not in (None, "ignore"):
        msg = f"na_action must either be 'ignore' or None, {na_action} was passed"
        raise ValueError(msg)

    # 对于字典或者 Series 类型的 mapper，进行快速路径处理
    # 因为知道不会返回 Python 原生类型，而是 numpy 数据结构
    if is_dict_like(mapper):
        if isinstance(mapper, dict) and hasattr(mapper, "__missing__"):
            # 如果字典子类定义了默认值方法，将 mapper 转换为查找函数
            dict_with_default = mapper
            mapper = lambda x: dict_with_default[
                np.nan if isinstance(x, float) and np.isnan(x) else x
            ]
        else:
            # 字典没有默认值，可以安全地转换为 Series 以提高效率
            # 在这里指定键处理可能是元组的情况
            # 使用空的 mapper 返回 pd.Series(np.nan, ...) 的预期值
            # 因为 np.nan 是 float64 类型，该方法的返回值应该也是 float64
            from pandas import Series

            if len(mapper) == 0:
                mapper = Series(mapper, dtype=np.float64)
            else:
                mapper = Series(mapper)

    # 如果 mapper 是 ABCSeries 的实例
    if isinstance(mapper, ABCSeries):
        if na_action == "ignore":
            # 如果 na_action 是 'ignore'，则只处理非 NaN 的值
            mapper = mapper[mapper.index.notna()]

        # 由于输入的值来自字典或 Series，mapper 应该是一个索引
        indexer = mapper.index.get_indexer(arr)
        new_values = take_nd(mapper._values, indexer)

        return new_values

    # 如果 arr 长度为 0，直接返回其拷贝
    if not len(arr):
        return arr.copy()

    # 必须将 arr 转换为 Python 类型
    values = arr.astype(object, copy=False)
    if na_action is None:
        # 如果 na_action 是 None，使用 lib.map_infer 处理映射
        return lib.map_infer(values, mapper)
    else:
        # 否则，使用 lib.map_infer_mask 处理映射，同时处理 NaN 值
        return lib.map_infer_mask(values, mapper, mask=isna(values).view(np.uint8))
```