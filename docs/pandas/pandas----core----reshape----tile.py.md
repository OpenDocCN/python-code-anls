# `D:\src\scipysrc\pandas\pandas\core\reshape\tile.py`

```
"""
Quantilization functions and related stuff
"""

from __future__ import annotations  # 允许类型注解引用当前模块中定义的名称

from typing import (  # 引入类型提示的必要模块和类
    TYPE_CHECKING,  # 用于类型检查的特殊标记
    Any,  # 任意类型
    Literal,  # 字面类型
)

import numpy as np  # 导入 NumPy 库并简称为 np

from pandas._libs import (  # 从 pandas._libs 导入特定模块或类
    Timedelta,  # 时间增量类型
    Timestamp,  # 时间戳类型
    lib,  # pandas 库中的低级函数
)

from pandas.core.dtypes.common import (  # 导入 pandas 中的通用数据类型检测函数
    ensure_platform_int,  # 确保是平台整数
    is_bool_dtype,  # 判断是否为布尔类型
    is_integer,  # 判断是否为整数类型
    is_list_like,  # 判断是否为类列表类型
    is_numeric_dtype,  # 判断是否为数值类型
    is_scalar,  # 判断是否为标量
)
from pandas.core.dtypes.dtypes import (  # 导入 pandas 中的具体数据类型
    CategoricalDtype,  # 类别数据类型
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    ExtensionDtype,  # 扩展数据类型
)
from pandas.core.dtypes.generic import ABCSeries  # 导入 pandas 中的抽象基类 Series
from pandas.core.dtypes.missing import isna  # 导入 pandas 中的缺失值检测函数

from pandas import (  # 从 pandas 库中导入特定模块或类
    Categorical,  # 类别类型
    Index,  # 索引类型
    IntervalIndex,  # 区间索引类型
)
import pandas.core.algorithms as algos  # 导入 pandas 中的算法模块
from pandas.core.arrays.datetimelike import dtype_to_unit  # 导入日期时间数组相关的函数

if TYPE_CHECKING:
    from collections.abc import Callable  # 导入 Callable 类型提示

    from pandas._typing import (  # 从 pandas._typing 导入特定类型
        DtypeObj,  # 数据类型对象
        IntervalLeftRight,  # 区间的左右边界
    )


def cut(  # 定义函数 cut，用于将值分成离散区间
    x,  # 输入数组，必须是一维的
    bins,  # 分箱的标准，可以是整数、标量序列或 IntervalIndex
    right: bool = True,  # 指示 bins 是否包含右边界，默认为 True
    labels=None,  # 返回分箱后的标签，默认为 None
    retbins: bool = False,  # 是否返回分箱后的区间，默认为 False
    precision: int = 3,  # 区间边界的精度，默认为 3
    include_lowest: bool = False,  # 是否包含最左边的区间，默认为 False
    duplicates: str = "raise",  # 处理重复值的方式，默认为 "raise"
    ordered: bool = True,  # 返回的标签是否按顺序排列，默认为 True
):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error. When `ordered=False`, labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.

    """
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True,
        the resulting categorical will be ordered. If False, the resulting
        categorical will be unordered (labels must be provided).

    Returns
    -------
    out : Categorical, Series, or ndarray
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`.

        * None (default) : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are Interval dtype.

        * sequence of scalars : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are whatever the type in the sequence is.

        * False : returns an ndarray of integers.

    bins : numpy.ndarray or IntervalIndex.
        The computed or specified bins. Only returned when `retbins=True`.
        For scalar or sequence `bins`, this is an ndarray with the computed
        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For
        an IntervalIndex `bins`, this is equal to `bins`.

    See Also
    --------
    qcut : Discretize variable into equal-sized buckets based on rank
        or based on sample quantiles.
    Categorical : Array type for storing data that come from a
        fixed set of values.
    Series : One-dimensional array with axis labels (including time series).
    IntervalIndex : Immutable Index implementing an ordered, sliceable set.

    Notes
    -----
    Any NA values will be NA in the result. Out of bounds values will be NA in
    the resulting Series or Categorical object.

    Reference :ref:`the user guide <reshaping.tile.cut>` for more examples.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
    ... # doctest: +ELLIPSIS
    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
    ... # doctest: +ELLIPSIS
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...
    array([0.994, 3.   , 5.   , 7.   ]))

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical's categories are `labels` and is ordered.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=["bad", "medium", "good"])
    """
    ['bad', 'good', 'medium', 'medium', 'good', 'bad']
    Categories (3, object): ['bad' < 'medium' < 'good']
    
    ``ordered=False`` will result in unordered categories when labels are passed.
    This parameter can be used to allow non-unique labels:
    
    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=["B", "A", "B"], ordered=False)
    ['B', 'B', 'A', 'A', 'B', 'B']
    Categories (2, object): ['A', 'B']
    
    ``labels=False`` implies you just want the bins back.
    
    >>> pd.cut([0, 1, 1, 2], bins=4, labels=False)
    array([0, 1, 1, 3])
    
    Passing a Series as an input returns a Series with categorical dtype:
    
    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]), index=["a", "b", "c", "d", "e"])
    >>> pd.cut(s, 3)
    ... # doctest: +ELLIPSIS
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64, right]): [(1.992, 4.667] < (4.667, ...
    
    Passing a Series as an input returns a Series with mapping value.
    It is used to map numerically to intervals based on bins.
    
    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]), index=["a", "b", "c", "d", "e"])
    >>> pd.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False)
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    4.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6,  8, 10]))
    
    Use `drop` optional when bins is not unique
    
    >>> pd.cut(
    ...     s,
    ...     [0, 2, 4, 6, 10, 10],
    ...     labels=False,
    ...     retbins=True,
    ...     right=False,
    ...     duplicates="drop",
    ... )
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    3.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6, 10]))
    
    Passing an IntervalIndex for `bins` results in those categories exactly.
    Notice that values not covered by the IntervalIndex are set to NaN. 0
    is to the left of the first bin (which is closed on the right), and 1.5
    falls between two bins.
    
    >>> bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    >>> pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)
    [NaN, (0.0, 1.0], NaN, (2.0, 3.0], (4.0, 5.0]]
    Categories (3, interval[int64, right]): [(0, 1] < (2, 3] < (4, 5]]
    """
    # NOTE: this binning code is changed a bit from histogram for var(x) == 0
    
    # 将原始数据保存到变量 original 中
    original = x
    # 对数据进行预处理以供切分
    x_idx = _preprocess_for_cut(x)
    # 将处理后的数据强制转换为特定类型，获取转换后的数据和未使用的部分
    x_idx, _ = _coerce_to_type(x_idx)
    
    # 如果 bins 不是可迭代对象，则将其转换为适当的区间
    if not np.iterable(bins):
        bins = _nbins_to_bins(x_idx, bins, right)
    
    # 如果 bins 是 IntervalIndex 类型，并且存在重叠区间，则抛出错误
    elif isinstance(bins, IntervalIndex):
        if bins.is_overlapping:
            raise ValueError("Overlapping IntervalIndex is not accepted.")
    
    # 否则，将 bins 转换为 Index 类型，并检查其是否单调递增
    else:
        bins = Index(bins)
        if not bins.is_monotonic_increasing:
            raise ValueError("bins must increase monotonically.")
    # 使用 _bins_to_cuts 函数处理输入的参数，返回分箱后的因子（fac）和分箱边界（bins）
    fac, bins = _bins_to_cuts(
        x_idx,                  # 输入数据的索引或者标签
        bins,                   # 分箱的边界值列表或者分箱数量
        right=right,            # 是否包含右边界的布尔标志
        labels=labels,          # 分箱后每个箱体的标签（可选）
        precision=precision,    # 分箱边界的精度
        include_lowest=include_lowest,  # 是否包含最低边界的布尔标志
        duplicates=duplicates,  # 如何处理重复边界的方式（drop、raise、bins等）
        ordered=ordered         # 是否保持标签的顺序
    )
    
    # 使用 _postprocess_for_cut 函数对分箱结果进行后处理，返回处理后的因子（fac）、边界（bins）、是否返回分箱边界（retbins）以及原始数据
    return _postprocess_for_cut(fac, bins, retbins, original)
# 定义一个用于按分位数离散化数据的函数
def qcut(
    x,
    q,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: str = "raise",
):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    Parameters
    ----------
    x : 1d ndarray or Series
        Input Numpy array or pandas Series object to be discretized.
    q : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    labels : array or False, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins. If True, raises an error.
    retbins : bool, optional
        Whether to return the (bins, labels) or not. Can be useful if bins
        is given as a scalar.
    precision : int, optional
        The precision at which to store and display the bins labels.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical or Series or array of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : ndarray of floats
        Returned only if `retbins` is True.

    See Also
    --------
    cut : Bin values into discrete intervals.
    Series.quantile : Return value at the given quantile.

    Notes
    -----
    Out of bounds values will be NA in the resulting Categorical object

    Examples
    --------
    >>> pd.qcut(range(5), 4)
    ... # doctest: +ELLIPSIS
    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
    Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] ...

    >>> pd.qcut(range(5), 3, labels=["good", "medium", "bad"])
    ... # doctest: +SKIP
    [good, good, medium, bad, bad]
    Categories (3, object): [good < medium < bad]

    >>> pd.qcut(range(5), 4, labels=False)
    array([0, 0, 1, 2, 3])
    """
    # 保存原始输入数据
    original = x
    # 对输入数据进行预处理，确保适用于切割操作
    x_idx = _preprocess_for_cut(x)
    # 将数据类型强制转换为适合操作的类型
    x_idx, _ = _coerce_to_type(x_idx)

    # 根据输入的分位数数量生成等间距的分位数
    quantiles = np.linspace(0, 1, q + 1) if is_integer(q) else q

    # 计算分位数，并且移除其中的NaN值
    bins = x_idx.to_series().dropna().quantile(quantiles)

    # 转换分位数边界到离散化标签
    fac, bins = _bins_to_cuts(
        x_idx,
        Index(bins),
        labels=labels,
        precision=precision,
        include_lowest=True,
        duplicates=duplicates,
    )

    # 返回离散化后的结果，包括离散化后的数据和可能的分位数边界
    return _postprocess_for_cut(fac, bins, retbins, original)


def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    """
    # 如果用户传入的 `nbins` 是一个标量且小于1，抛出值错误异常
    if is_scalar(nbins) and nbins < 1:
        raise ValueError("`bins` should be a positive integer.")
    
    # 如果输入数组 `x_idx` 大小为0，抛出值错误异常
    if x_idx.size == 0:
        raise ValueError("Cannot cut empty array")
    
    # 确定输入数组 `x_idx` 的范围
    rng = (x_idx.min(), x_idx.max())
    mn, mx = rng
    
    # 如果输入数组 `x_idx` 的数据类型是数值型，并且包含无穷大值，抛出值错误异常
    if is_numeric_dtype(x_idx.dtype) and (np.isinf(mn) or np.isinf(mx)):
        # GH#24314
        raise ValueError(
            "cannot specify integer `bins` when input data contains infinity"
        )
    
    # 如果最小值 `mn` 等于最大值 `mx`，调整端点以便进行分箱处理
    if mn == mx:  # adjust end points before binning
        if _is_dt_or_td(x_idx.dtype):
            # 对于日期时间或时间差类型，使用秒数为单位进行调整
            # 错误："dtype_to_unit" 的第1个参数具有不兼容的类型 "dtype[Any] | ExtensionDtype"；
            # 预期的类型是 "DatetimeTZDtype | dtype[Any]"
            unit = dtype_to_unit(x_idx.dtype)  # type: ignore[arg-type]
            td = Timedelta(seconds=1).as_unit(unit)
            # 使用 DatetimeArray/TimedeltaArray 方法生成范围，而不是使用 linspace
            # 错误："ExtensionArray" 或 "ndarray[Any, Any]" 的项目没有 "_generate_range" 属性
            bins = x_idx._values._generate_range(  # type: ignore[union-attr]
                start=mn - td, end=mx + td, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            # 如果不是日期时间或时间差类型，微调端点
            mn -= 0.001 * abs(mn) if mn != 0 else 0.001
            mx += 0.001 * abs(mx) if mx != 0 else 0.001
    
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
    else:  # adjust end points after binning
        if _is_dt_or_td(x_idx.dtype):
            # 使用 DatetimeArray/TimedeltaArray 方法生成范围，而不是使用 linspace
    
            # 错误："dtype_to_unit" 的第1个参数具有不兼容的类型 "dtype[Any] | ExtensionDtype"；
            # 预期的类型是 "DatetimeTZDtype | dtype[Any]"
            unit = dtype_to_unit(x_idx.dtype)  # type: ignore[arg-type]
            # 错误："ExtensionArray" 或 "ndarray[Any, Any]" 的项目没有 "_generate_range" 属性
            bins = x_idx._values._generate_range(  # type: ignore[union-attr]
                start=mn, end=mx, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
        
        # 计算范围的0.1%的微调
        adj = (mx - mn) * 0.001  # 0.1% of the range
        # 根据 `right` 参数调整端点
        if right:
            bins[0] -= adj
        else:
            bins[-1] += adj
    
    # 返回分箱后的索引对象
    return Index(bins)
# 定义一个函数 `_bins_to_cuts`，接受多个参数：
# - `x_idx`：索引对象，用于表示要切割的数据索引
# - `bins`：索引对象，用于指定切割的边界
# - `right`：布尔值，指定是否包括右边界，默认为 True
# - `labels`：可选参数，用于指定切割后的标签
# - `precision`：整数，指定精度，默认为 3
# - `include_lowest`：布尔值，指定是否包括最低边界，默认为 False
# - `duplicates`：字符串，指定处理重复边界的方式，默认为 "raise"
# - `ordered`：布尔值，指示 `bins` 是否有序，默认为 True
def _bins_to_cuts(
    x_idx: Index,
    bins: Index,
    right: bool = True,
    labels=None,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    # 如果 `ordered` 为 False 且未提供 `labels` 参数，则抛出 ValueError 异常
    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")

    # 如果 `duplicates` 参数不是 "raise" 或 "drop"，则抛出 ValueError 异常
    if duplicates not in ["raise", "drop"]:
        raise ValueError(
            "invalid value for 'duplicates' parameter, valid options are: raise, drop"
        )

    # 声明一个结果变量 `result`，可以是 Categorical 或者 np.ndarray 类型
    result: Categorical | np.ndarray

    # 如果 `bins` 是 IntervalIndex 类型，则使用快速路径处理
    if isinstance(bins, IntervalIndex):
        # 获取 `x_idx` 在 `bins` 中的索引
        ids = bins.get_indexer(x_idx)
        # 创建一个 CategoricalDtype 对象，指定有序性
        cat_dtype = CategoricalDtype(bins, ordered=True)
        # 根据索引和数据类型创建 Categorical 对象，并返回结果及 bins
        result = Categorical.from_codes(ids, dtype=cat_dtype, validate=False)
        return result, bins

    # 对 `bins` 进行去重操作，获取唯一的边界值
    unique_bins = algos.unique(bins)
    # 如果去重后的边界值少于原始 `bins` 的长度，并且 `bins` 的长度不为 2
    if len(unique_bins) < len(bins) and len(bins) != 2:
        # 如果 `duplicates` 参数为 "raise"，则抛出 ValueError 异常
        if duplicates == "raise":
            raise ValueError(
                f"Bin edges must be unique: {bins!r}.\n"
                f"You can drop duplicate edges by setting the 'duplicates' kwarg"
            )
        # 否则，使用去重后的 `unique_bins`
        bins = unique_bins

    # 根据 `right` 参数确定搜索方向的值 "left" 或 "right"
    side: Literal["left", "right"] = "left" if right else "right"

    try:
        # 使用 `searchsorted` 方法找到 `x_idx` 在 `bins` 中的插入点索引
        ids = bins.searchsorted(x_idx, side=side)
    except TypeError as err:
        # 如果出现 TypeError 异常，例如当 `bins` 是 DatetimeArray 而 `x_idx` 是整数时
        if x_idx.dtype.kind == "m":
            raise ValueError("bins must be of timedelta64 dtype") from err
        # 如果 `x_idx` 和 `bins` 的 dtype 都是 "M" 类型
        elif x_idx.dtype.kind == bins.dtype.kind == "M":
            raise ValueError(
                "Cannot use timezone-naive bins with timezone-aware values, "
                "or vice-versa"
            ) from err
        # 如果 `x_idx` 是 "M" 类型，而 `bins` 不是
        elif x_idx.dtype.kind == "M":
            raise ValueError("bins must be of datetime64 dtype") from err
        else:
            raise

    # 确保 `ids` 是平台整数类型
    ids = ensure_platform_int(ids)

    # 如果 `include_lowest` 为 True，则将边界值与 `x_idx` 相等的索引设置为 1
    if include_lowest:
        ids[x_idx == bins[0]] = 1

    # 创建一个掩码，标识 `x_idx` 中的缺失值或者在 `bins` 边界之外的值
    na_mask = isna(x_idx) | (ids == len(bins)) | (ids == 0)
    # 检查是否存在缺失值
    has_nas = na_mask.any()
    # 如果 labels 不是 False，则进入条件判断
    if labels is not False:
        # 如果 labels 既不是 None，也不是类列表对象，则抛出数值错误异常
        if not (labels is None or is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a "
                "list-like argument"
            )

        # 如果 labels 是 None，则使用 _format_labels 函数格式化标签
        if labels is None:
            labels = _format_labels(
                bins, precision, right=right, include_lowest=include_lowest
            )
        # 如果 ordered 为 True 并且 labels 中有重复项，则抛出数值错误异常
        elif ordered and len(set(labels)) != len(labels):
            raise ValueError(
                "labels must be unique if ordered=True; pass ordered=False "
                "for duplicate labels"
            )
        # 否则，如果 labels 的长度不等于 bins 的长度减一，则抛出数值错误异常
        else:
            if len(labels) != len(bins) - 1:
                raise ValueError(
                    "Bin labels must be one fewer than the number of bin edges"
                )

        # 如果 labels 不是 CategoricalDtype 类型，则转换为 Categorical 类型
        if not isinstance(getattr(labels, "dtype", None), CategoricalDtype):
            labels = Categorical(
                labels,
                categories=labels if len(set(labels)) == len(labels) else None,
                ordered=ordered,
            )
        # TODO: 处理分类标签顺序与 pandas.cut 的顺序不匹配的情况。

        # 使用 np.putmask 函数处理缺失值标志数组 na_mask
        np.putmask(ids, na_mask, 0)
        # 使用 algos.take_nd 函数根据 ids 数组提取对应 labels 中的值
        result = algos.take_nd(labels, ids - 1)

    else:
        # 如果 labels 是 False，则执行以下操作
        # 直接将 ids 数组减一赋值给 result
        result = ids - 1
        # 如果存在缺失值，则将 result 转换为 np.float64 类型，并使用 np.nan 替换 na_mask 位置的值
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)

    # 返回结果 result 和 bins
    return result, bins
    # 确定数据的类型
    dtype: DtypeObj | None = None

    # 检查是否是日期时间或时间间隔类型
    if _is_dt_or_td(x.dtype):
        dtype = x.dtype
    # 如果是布尔类型，转换为整数类型
    elif is_bool_dtype(x.dtype):
        # GH 20303
        x = x.astype(np.int64)
    
    # 处理扩展数据类型为数值类型的情况，将其转换为浮点型数组
    elif isinstance(x.dtype, ExtensionDtype) and is_numeric_dtype(x.dtype):
        x_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
        x = Index(x_arr)

    # 返回处理后的数据索引和数据类型
    return Index(x), dtype


def _is_dt_or_td(dtype: DtypeObj) -> bool:
    # 注意：此处的 dtype 是从 Index.dtype 获取的，我们知道任何 dt64/td64 类型
    # 都是受支持的单位类型。
    return isinstance(dtype, DatetimeTZDtype) or lib.is_np_dtype(dtype, "mM")


def _format_labels(
    bins: Index,
    precision: int,
    right: bool = True,
    include_lowest: bool = False,
) -> IntervalIndex:
    """根据数据类型返回标签"""

    # 确定闭区间的方向
    closed: IntervalLeftRight = "right" if right else "left"

    # 定义格式化器的类型
    formatter: Callable[[Any], Timestamp] | Callable[[Any], Timedelta]

    # 如果 bins 的类型是日期时间或时间间隔类型
    if _is_dt_or_td(bins.dtype):
        # 获取单位类型
        unit = dtype_to_unit(bins.dtype)  # type: ignore[arg-type]
        # 使用 lambda 函数不做任何调整
        formatter = lambda x: x
        # 调整函数为减去单位类型的一天
        adjust = lambda x: x - Timedelta(1, unit=unit).as_unit(unit)
    else:
        # 推断精度
        precision = _infer_precision(precision, bins)
        # 根据精度四舍五入数字
        formatter = lambda x: _round_frac(x, precision)
        # 调整函数为减去 10 的负精度次方
        adjust = lambda x: x - 10 ** (-precision)

    # 生成标签列表
    breaks = [formatter(b) for b in bins]
    
    # 如果是右闭区间且包含最低值，则调整第一个间隔的左边界
    if right and include_lowest:
        breaks[0] = adjust(breaks[0])

    # 如果 bins 的类型是日期时间或时间间隔类型
    if _is_dt_or_td(bins.dtype):
        # 将 breaks 转换为 bins 类型的索引对象，并按单位类型调整
        breaks = type(bins)(breaks).as_unit(unit)  # type: ignore[attr-defined]

    # 返回间隔索引对象
    return IntervalIndex.from_breaks(breaks, closed=closed)


def _preprocess_for_cut(x) -> Index:
    """
    为 cut 方法处理前的预处理，将传入的输入转换为数组，
    剥离索引信息并单独存储
    """

    # 检查传入数组是否为 Pandas 或 Numpy 对象
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        x = np.asarray(x)
    # 确保输入数组是一维的
    if x.ndim != 1:
        raise ValueError("Input array must be 1 dimensional")

    # 返回索引对象
    return Index(x)


def _postprocess_for_cut(fac, bins, retbins: bool, original):
    """
    处理 cut 方法后的后处理，其中
    # 如果原始数据类型是 Series 类型的实例
    if isinstance(original, ABCSeries):
        # 重新构造一个新的 Series 对象 fac，
        # 使用 fac 数据，保留原始的索引和名称
        fac = original._constructor(fac, index=original.index, name=original.name)

    # 如果不需要返回 bins，则直接返回 fac
    if not retbins:
        return fac

    # 如果 bins 是 Index 类型且其数据类型是数值型
    if isinstance(bins, Index) and is_numeric_dtype(bins.dtype):
        # 将 bins 转换为其内部的值数组
        bins = bins._values

    # 返回 fac 和 bins（如果需要返回）
    return fac, bins
# 对给定的数值 x 的小数部分进行四舍五入，保留指定的精度
def _round_frac(x, precision: int):
    """
    Round the fractional part of the given number
    """
    # 如果 x 不是有限数或者为零，则直接返回 x
    if not np.isfinite(x) or x == 0:
        return x
    else:
        # 将 x 拆分为小数部分 frac 和整数部分 whole
        frac, whole = np.modf(x)
        # 如果整数部分为零，则计算小数部分的有效位数
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            # 否则，使用给定的精度
            digits = precision
        # 对 x 进行四舍五入，保留指定的位数，并返回结果
        return np.around(x, digits)


# 推断适用于 _round_frac 的适当精度
def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    Infer an appropriate precision for _round_frac
    """
    # 从 base_precision 开始逐步增加精度，直到找到适用于所有 bins 的精度
    for precision in range(base_precision, 20):
        # 对 bins 中的每个元素调用 _round_frac 函数，生成一个数组 levels
        levels = np.asarray([_round_frac(b, precision) for b in bins])
        # 如果 levels 中的所有值都是唯一的，则表示找到了适当的精度
        if algos.unique(levels).size == bins.size:
            return precision
    # 如果未找到适当的精度，则返回默认的 base_precision
    return base_precision  # default
```