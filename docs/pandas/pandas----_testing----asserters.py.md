# `D:\src\scipysrc\pandas\pandas\_testing\asserters.py`

```
from __future__ import annotations
# 导入用于类型提示的未来版本特性

import operator
# 导入操作符模块

from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    cast,
)
# 导入类型提示相关模块和类

import numpy as np
# 导入 NumPy 库并使用别名 np

from pandas._libs import lib
# 导入 Pandas 内部库

from pandas._libs.missing import is_matching_na
# 导入 Pandas 缺失值处理相关模块

from pandas._libs.sparse import SparseIndex
# 导入 Pandas 稀疏数据索引模块

import pandas._libs.testing as _testing
# 导入 Pandas 内部测试相关模块，并使用别名 _testing

from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
# 导入 Pandas 时间序列处理相关模块

from pandas.core.dtypes.common import (
    is_bool,
    is_float_dtype,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    needs_i8_conversion,
)
# 导入 Pandas 核心数据类型模块中的数据类型检查函数

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    NumpyEADtype,
)
# 导入 Pandas 核心数据类型模块中的具体数据类型

from pandas.core.dtypes.missing import array_equivalent
# 导入 Pandas 核心数据类型模块中的缺失值处理函数

import pandas as pd
# 导入 Pandas 并使用别名 pd

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalDtype,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
# 从 Pandas 中导入特定类和函数

from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
# 导入 Pandas 核心数组模块中的数组类

from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
# 导入 Pandas 核心数组模块中的日期时间数组混合类

from pandas.core.arrays.string_ import StringDtype
# 导入 Pandas 核心数组模块中的字符串数据类型

from pandas.core.indexes.api import safe_sort_index
# 导入 Pandas 索引 API 模块中的安全排序索引函数

from pandas.io.formats.printing import pprint_thing
# 导入 Pandas 格式打印模块中的打印函数

if TYPE_CHECKING:
    from pandas._typing import DtypeObj
# 如果是类型检查阶段，导入 Pandas 类型提示模块中的 DtypeObj 类型

def assert_almost_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = "equiv",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
        The left object to compare.
    right : object
        The right object to compare against.
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Index` with int64 dtype are also considered
        equivalent when doing type checking.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            exact=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )
    # 如果 left 是索引对象，则调用 assert_index_equal 函数比较

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )
    # 如果 left 是序列对象，则调用 assert_series_equal 函数比较

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )
    # 如果 left 是数据框对象，则调用 assert_frame_equal 函数比较
    else:
        # 处理其他类型的序列。
        if check_dtype:
            # 如果需要检查数据类型
            if is_number(left) and is_number(right):
                # 如果左右两边都是数字类型，如 np.float64 和 float，则跳过
                pass
            elif is_bool(left) and is_bool(right):
                # 如果左右两边都是布尔类型，如 np.bool_ 和 bool，则跳过
                pass
            else:
                # 如果左右至少有一个是 numpy 数组
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    obj = "numpy array"
                else:
                    obj = "Input"
                # 断言左右两边的类别相同
                assert_class_equal(left, right, obj=obj)

        # 如果需要考虑"equiv"，则置为 True
        _testing.assert_almost_equal(
            left, right, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, **kwargs
        )
# 辅助方法用于断言方法，确保比较的两个对象在继续比较之前具有正确的类型。
# left: 要比较的第一个对象。
# right: 要比较的第二个对象。
# cls: 要检查的类类型。
def _check_isinstance(left, right, cls) -> None:
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__

    # 如果 left 不是 cls 类型，则抛出 AssertionError
    if not isinstance(left, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(left)} instead"
        )
    # 如果 right 不是 cls 类型，则抛出 AssertionError
    if not isinstance(right, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(right)} instead"
        )


# 断言两个字典对象相等
# left: 第一个要比较的字典对象。
# right: 第二个要比较的字典对象。
# compare_keys: 是否比较字典键，默认为 True。
def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    _check_isinstance(left, right, dict)
    # 调用 testing 模块中的 assert_dict_equal 方法，比较两个字典对象
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


# 断言两个 Index 对象相等
# left: 第一个要比较的 Index 对象。
# right: 第二个要比较的 Index 对象。
# exact: 是否精确比较，可以是布尔值或字符串 {'equiv'}，默认为 'equiv'。
# check_names: 是否检查名称属性，默认为 True。
# check_exact: 是否精确比较数值，默认为 True。
# check_categorical: 是否精确比较内部分类，默认为 True。
# check_order: 是否比较索引条目的顺序，默认为 True。
# rtol: 相对容差，仅在 check_exact 为 False 时使用，默认为 1e-5。
# atol: 绝对容差，仅在 check_exact 为 False 时使用，默认为 1e-8。
# obj: 要比较的对象名称，默认为 'Index'。
def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | str = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Index",
) -> None:
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
        The first index to compare.
    right : Index
        The second index to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    testing.assert_series_equal : Check that two Series are equal.
    testing.assert_frame_equal : Check that two DataFrames are equal.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = True
    # 定义一个函数 `_check_types`，用于检查并比较两个对象 `left` 和 `right` 的类型和属性
    def _check_types(left, right, obj: str = "Index") -> None:
        # 如果 exact 参数为 False，则直接返回，不执行后续检查
        if not exact:
            return
        
        # 断言左右对象的类别相同
        assert_class_equal(left, right, exact=exact, obj=obj)
        # 断言左右对象的推断类型属性相同
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # 如果左右对象的 dtype 均为 CategoricalDtype 类型，并且 check_categorical 参数为 True
        if isinstance(left.dtype, CategoricalDtype) and isinstance(
            right.dtype, CategoricalDtype
        ):
            if check_categorical:
                # 断言左右对象的 dtype 属性相同
                assert_attr_equal("dtype", left, right, obj=obj)
                # 断言左右对象的 categories 相同
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        # 断言左右对象的 dtype 属性相同
        assert_attr_equal("dtype", left, right, obj=obj)

    # 对左右对象进行 isinstance 检查，确保它们属于 Index 类型
    _check_isinstance(left, right, Index)

    # 检查左右对象的类别和 dtype 属性
    _check_types(left, right, obj=obj)

    # 比较左右对象的层级数目是否相同，若不同则抛出详细的断言错误信息
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # 比较左右对象的长度是否相同，若不同则抛出详细的断言错误信息
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # 如果不需要考虑顺序，则对索引条目进行排序
    if not check_order:
        left = safe_sort_index(left)
        right = safe_sort_index(right)

    # 如果左对象是 MultiIndex 类型，则进行特殊的比较以生成友好的错误消息
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        # 遍历 MultiIndex 的每个层级
        for level in range(left.nlevels):
            lobj = f"MultiIndex level [{level}]"
            try:
                # 尝试比较层级和编码以避免将 MultiIndex 转换为稠密格式
                assert_index_equal(
                    left.levels[level],
                    right.levels[level],
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                # 如果出现断言错误，则获取层级值并比较
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # 检查层级的类型和属性
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # 当 check_categorical 参数为 False 时，跳过对精确索引的检查
    # 如果需要精确比较并且是分类数据
    elif check_exact and check_categorical:
        # 如果左右两边的对象不相等
        if not left.equals(right):
            # 计算不匹配项
            mismatch = left._values != right._values

            # 如果不匹配项不是 NumPy 数组，则转换为 ExtensionArray 类型并填充缺失值为 True
            if not isinstance(mismatch, np.ndarray):
                mismatch = cast("ExtensionArray", mismatch).fillna(True)

            # 计算不匹配的百分比
            diff = np.sum(mismatch.astype(int)) * 100.0 / len(left)
            # 构建错误消息，显示不匹配的百分比
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            # 抛出详细断言错误，包含错误消息和相关对象
            raise_assert_detail(obj, msg, left, right)
    else:
        # 如果设置了 "equiv" 参数，这里将为 True
        exact_bool = bool(exact)
        # 使用 _testing.assert_almost_equal 函数比较左右两边的值，包括相对容差、绝对容差和数据类型检查
        _testing.assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=exact_bool,
            obj=obj,
            lobj=left,
            robj=right,
        )

    # 元数据比较部分
    # 检查是否需要比较名称
    if check_names:
        # 调用 assert_attr_equal 函数，比较左右两边对象的名称属性
        assert_attr_equal("names", left, right, obj=obj)
    # 如果左右任一边是 PeriodIndex 类型
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        # 调用 assert_attr_equal 函数，比较左右两边对象的 dtype 属性
        assert_attr_equal("dtype", left, right, obj=obj)
    # 如果左右任一边是 IntervalIndex 类型
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        # 调用 assert_interval_array_equal 函数，比较左右两边的 IntervalArray 数组
        assert_interval_array_equal(left._values, right._values)

    # 如果需要比较分类数据
    if check_categorical:
        # 如果左边的数据类型是 CategoricalDtype 或者右边的数据类型是 CategoricalDtype
        if isinstance(left.dtype, CategoricalDtype) or isinstance(right.dtype, CategoricalDtype):
            # 调用 assert_categorical_equal 函数，比较左右两边的分类数据
            assert_categorical_equal(left._values, right._values, obj=f"{obj} category")
# 定义一个函数用于断言两个对象的类是否相等
def assert_class_equal(
    left, right, exact: bool | str = True, obj: str = "Input"
) -> None:
    """
    Checks classes are equal.
    """
    # 隐藏异常追溯信息
    __tracebackhide__ = True

    # 定义一个函数，用于返回对象的类名或者 Index 类的对象本身
    def repr_class(x):
        if isinstance(x, Index):
            # 对于 Index 类的对象直接返回，以便在错误消息中包含其值
            return x
        # 返回对象的类名
        return type(x).__name__

    # 完全相等模式或者类等价模式的判断条件
    if type(left) == type(right):
        return

    # 如果是类等价模式，并且 left 和 right 都是 Index 或 RangeIndex 类的实例，则通过检查
    if exact == "equiv":
        if is_class_equiv(left) and is_class_equiv(right):
            return

    # 生成错误消息，指示对象的类不同
    msg = f"{obj} classes are different"
    # 抛出自定义详细断言异常，显示错误消息和 left, right 对象的类名
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


# 函数用于断言两个对象的指定属性相等
def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    # 隐藏异常追溯信息
    __tracebackhide__ = True

    # 获取左右对象的指定属性值
    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    # 如果左右对象的属性相等，或者是匹配的缺失值（如 np.nan, NaT, pd.NA 等）
    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        return None

    try:
        # 尝试比较左右对象的属性值是否相等
        result = left_attr == right_attr
    except TypeError:
        # 如果比较引发 TypeError，例如右边对象是 datetimetz 类型
        result = False

    # 如果左右对象的属性值中有一个是 pd.NA，则结果为 False
    if (left_attr is pd.NA) ^ (right_attr is pd.NA):
        result = False
    # 如果 result 不是布尔值，则使用 .all() 方法检查所有元素是否相等
    elif not isinstance(result, bool):
        result = result.all()

    # 如果比较结果不相等，则生成错误消息，并抛出详细断言异常
    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


# 函数用于断言序列是否按顺序排序
def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    # 如果序列是 Index 或 Series 对象，则获取其值数组
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # 如果序列是 numpy 数组，则比较排序后的结果是否与原数组相等
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        # 否则，比较排序后的结果是否与原序列中按索引排序的结果相等
        assert_extension_array_equal(seq, seq[seq.argsort()])


# 函数用于断言两个 Categorical 对象是否等价
def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    """
    # 隐藏异常追溯信息
    __tracebackhide__ = True
    """
    check_category_order : bool, default True
        是否比较类别的顺序，这意味着比较整数编码是否相同。如果为 False，仅比较结果值。无论如何都会检查 ordered 属性。
    obj : str, default 'Categorical'
        指定正在比较的对象名称，内部用于显示适当的断言消息。
    """
    # 检查 left 和 right 是否都是 Categorical 类型的实例
    _check_isinstance(left, right, Categorical)

    # 确定 exact 变量的值
    exact: bool | str
    if isinstance(left.categories, RangeIndex) or isinstance(
        right.categories, RangeIndex
    ):
        exact = "equiv"  # 如果任一对象的 categories 是 RangeIndex，则要求是等价比较
    else:
        exact = True  # 否则要求精确匹配

    # 如果需要检查类别的顺序
    if check_category_order:
        # 断言 left 和 right 的 categories 属性相等
        assert_index_equal(
            left.categories, right.categories, obj=f"{obj}.categories", exact=exact
        )
        # 断言 left 和 right 的 codes 属性相等
        assert_numpy_array_equal(
            left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes"
        )
    else:
        # 尝试对 left 和 right 的 categories 属性进行排序
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            # 处理排序时可能的类型错误，例如 'int' 和 'str' 之间无法比较
            lc, rc = left.categories, right.categories
        # 断言排序后的 lc 和 rc 相等
        assert_index_equal(lc, rc, obj=f"{obj}.categories", exact=exact)
        # 断言按照 left 和 right 的 codes 属性取相应的值后相等
        assert_index_equal(
            left.categories.take(left.codes),
            right.categories.take(right.codes),
            obj=f"{obj}.values",
            exact=exact,
        )

    # 断言 left 和 right 的 ordered 属性相等
    assert_attr_equal("ordered", left, right, obj=obj)
# 确保两个 IntervalArray 对象相等的断言函数
def assert_interval_array_equal(
    left, right, exact: bool | Literal["equiv"] = "equiv", obj: str = "IntervalArray"
) -> None:
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    # 确保 left 和 right 是 IntervalArray 类型的实例
    _check_isinstance(left, right, IntervalArray)

    kwargs = {}
    # 如果 left._left 的 dtype 是 'm' 或 'M'，表示是 DatetimeArray 或 TimedeltaArray
    if left._left.dtype.kind in "mM":
        # 关闭频率检查
        kwargs["check_freq"] = False

    # 断言左右 IntervalArray 的 _left 数组相等
    assert_equal(left._left, right._left, obj=f"{obj}.left", **kwargs)
    # 断言左右 IntervalArray 的 _right 数组相等
    assert_equal(left._right, right._right, obj=f"{obj}.left", **kwargs)

    # 断言左右 IntervalArray 的 'closed' 属性相等
    assert_attr_equal("closed", left, right, obj=obj)


# 确保两个 PeriodArray 对象相等的断言函数
def assert_period_array_equal(left, right, obj: str = "PeriodArray") -> None:
    # 确保 left 和 right 是 PeriodArray 类型的实例
    _check_isinstance(left, right, PeriodArray)

    # 断言左右 PeriodArray 的 _ndarray 数组相等
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    # 断言左右 PeriodArray 的 'dtype' 属性相等
    assert_attr_equal("dtype", left, right, obj=obj)


# 确保两个 DatetimeArray 对象相等的断言函数
def assert_datetime_array_equal(
    left, right, obj: str = "DatetimeArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    # 确保 left 和 right 是 DatetimeArray 类型的实例
    _check_isinstance(left, right, DatetimeArray)

    # 断言左右 DatetimeArray 的 _ndarray 数组相等
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    # 如果 check_freq 为 True，则断言左右 DatetimeArray 的 'freq' 属性相等
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)
    # 断言左右 DatetimeArray 的 'tz' 属性相等
    assert_attr_equal("tz", left, right, obj=obj)


# 确保两个 TimedeltaArray 对象相等的断言函数
def assert_timedelta_array_equal(
    left, right, obj: str = "TimedeltaArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    # 确保 left 和 right 是 TimedeltaArray 类型的实例
    _check_isinstance(left, right, TimedeltaArray)
    # 断言左右 TimedeltaArray 的 _ndarray 数组相等
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    # 如果 check_freq 为 True，则断言左右 TimedeltaArray 的 'freq' 属性相等
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)


# 抛出详细断言错误信息的函数
def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    # 如果 index_values 是 Index 类型，则转换为 ndarray
    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)

    # 如果 index_values 是 ndarray 类型，则添加到错误信息中
    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    # 格式化并添加左侧的值到错误信息中
    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, NumpyEADtype, StringDtype)):
        left = repr(left)

    msg += f"""
[left]:  {left}
[right]: {right}"""

    # 如果存在 diff 参数，则添加到错误信息中
    if diff is not None:
        msg += f"\n[diff]: {diff}"
    # 如果存在首个不同之处（first_diff 不为 None），将其添加到错误消息中
    msg += f"\n{first_diff}"
    
    # 抛出断言错误，错误消息为 msg，用于指示测试失败的具体原因
    raise AssertionError(msg)
# 定义一个函数，用于断言两个 numpy 数组是否相等
def assert_numpy_array_equal(
    left,
    right,
    strict_nan: bool = False,  # 是否严格对待 NaN 和 None 为不同
    check_dtype: bool | Literal["equiv"] = True,  # 是否检查数组的数据类型
    err_msg=None,  # 错误消息，用于自定义断言失败时的信息
    check_same=None,  # 检查 left 和 right 是否指向同一内存区域
    obj: str = "numpy array",  # 正在比较的对象的名称，默认为 'numpy array'
    index_values=None,  # 可选的索引，用于输出时的索引值显示
) -> None:
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    __tracebackhide__ = True  # 隐藏回溯信息以简化错误显示

    # 实例验证
    # 检查左右两个对象的类是否相同，不同则显示详细的错误消息
    assert_class_equal(left, right, obj=obj)
    # 确保左右两个对象都是 np.ndarray 类型
    _check_isinstance(left, right, np.ndarray)

    # 内部函数，获取对象的基础数组（如果存在）
    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    # 根据 check_same 的设置，判断是否需要确保 left 和 right 引用相同或不同的内存区域
    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{left_base!r} is not {right_base!r}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{left_base!r} is {right_base!r}")

    # 内部函数，用于在断言失败时引发异常并提供详细的错误消息
    def _raise(left, right, err_msg) -> NoReturn:
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(
                    obj, f"{obj} shapes are different", left.shape, right.shape
                )

            diff = 0
            # 遍历比较每个元素，计算不同之处
            for left_arr, right_arr in zip(left, right):
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1

            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    # 比较数组的形状和值是否相等，如果不相等则调用 _raise 函数引发异常
    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)

    # 如果 check_dtype 为 True，则检查左右两个对象的数据类型是否相同
    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal("dtype", left, right, obj=obj)
    # rtol 参数，表示相对容差，用于数值比较，默认为 lib.no_default
    rtol: float | lib.NoDefault = lib.no_default,
    # atol 参数，表示绝对容差，用于数值比较，默认为 lib.no_default
    atol: float | lib.NoDefault = lib.no_default,
    # obj 参数，表示对象类型，默认为 "ExtensionArray"
    obj: str = "ExtensionArray",
# 定义函数 assert_extension_array_equal，用于检查两个 ExtensionArray 是否相等
def assert_extension_array_equal(
    left, right  # left 和 right 分别为要比较的两个 ExtensionArray
    check_dtype: bool = True,  # 是否检查 ExtensionArray 的 dtype 是否相同，默认为 True
    index_values=None,  # 可选参数，用于输出的共享索引
    check_exact: bool = False,  # 是否精确比较数字，默认为 False
    rtol: float = 1e-5,  # 相对容差，当 check_exact 为 False 时使用，默认为 1e-5
    atol: float = 1e-8,  # 绝对容差，当 check_exact 为 False 时使用，默认为 1e-8
    obj: str = 'ExtensionArray'  # 内部使用的对象名称，用于显示适当的断言消息，默认为 'ExtensionArray'
) -> None:
    """
    Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """

    # 如果 check_exact, rtol 和 atol 都未指定，默认对整数 dtype 使用精确比较
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
    elif check_exact is lib.no_default:
        check_exact = False

    # 如果 check_exact 未指定，则设定 rtol 和 atol 的默认值
    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    # 断言 left 和 right 都是 ExtensionArray 类型
    assert isinstance(left, ExtensionArray), "left is not an ExtensionArray"
    assert isinstance(right, ExtensionArray), "right is not an ExtensionArray"

    # 如果 check_dtype 为 True，则断言 left 和 right 的 dtype 相同
    if check_dtype:
        assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    # 如果 left 和 right 都是 DatetimeLikeArrayMixin 的实例且类型相同，则进行特定断言
    if (
        isinstance(left, DatetimeLikeArrayMixin)
        and isinstance(right, DatetimeLikeArrayMixin)
        and type(right) == type(left)
    ):
        # 下面会继续该函数的代码，但因为示例未涵盖完整，这里不作展示
    ):
        # GH 52449: 检查左右对象的数据类型是否为日期时间类型，并处理不同的分辨率情况
        if not check_dtype and left.dtype.kind in "mM":
            # 如果不需要检查数据类型，并且左侧对象的数据类型是日期时间类型
            if not isinstance(left.dtype, np.dtype):
                # 如果左侧数据类型不是 np.dtype，则获取其单位
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                # 否则从 np.datetime_data 中获取单位
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                # 如果右侧数据类型不是 np.dtype，则获取其单位
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                # 否则从 np.datetime_data 中获取单位
                r_unit = np.datetime_data(right.dtype)[0]
            # 如果左右单位不同，并且通过 compare_mismatched_resolutions 函数比较为相等
            if (
                l_unit != r_unit
                and compare_mismatched_resolutions(
                    left._ndarray, right._ndarray, operator.eq
                ).all()
            ):
                return
        # 避免对包含对象数据类型的数组进行缓慢的比较操作
        # 当我们有一个 np.MaskedArray 时，使用 np.asarray 转换
        assert_numpy_array_equal(
            np.asarray(left.asi8),
            np.asarray(right.asi8),
            index_values=index_values,
            obj=obj,
        )
        return

    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    # 比较左右对象的 NA 掩码数组
    assert_numpy_array_equal(
        left_na, right_na, obj=f"{obj} NA mask", index_values=index_values
    )

    # 获取左右对象有效值的 numpy 数组，数据类型为对象
    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        # 如果需要精确比较，则使用 assert_numpy_array_equal 函数比较有效值数组
        assert_numpy_array_equal(
            left_valid, right_valid, obj=obj, index_values=index_values
        )
    else:
        # 否则使用 _testing.assert_almost_equal 函数进行近似比较
        _testing.assert_almost_equal(
            left_valid,
            right_valid,
            check_dtype=bool(check_dtype),
            rtol=rtol,
            atol=atol,
            obj=obj,
            index_values=index_values,
        )
# 定义一个函数用于断言两个 Series 是否相等
def assert_series_equal(
    left,  # 第一个 Series 对象，用于比较
    right,  # 第二个 Series 对象，用于比较
    check_dtype: bool | Literal["equiv"] = True,  # 是否检查 Series 的数据类型是否相同，默认为 True
    check_index_type: bool | Literal["equiv"] = "equiv",  # 是否检查索引的类、数据类型和推断类型是否相同，默认为 'equiv'
    check_series_type: bool = True,  # 是否检查 Series 的类是否相同，默认为 True
    check_names: bool = True,  # 是否检查 Series 和索引的名称属性是否相同，默认为 True
    check_exact: bool | lib.NoDefault = lib.no_default,  # 是否精确比较数值，默认为 lib.no_default
    check_datetimelike_compat: bool = False,  # 是否比较可以忽略数据类型的日期时间类，默认为 False
    check_categorical: bool = True,  # 是否精确比较内部的分类数据，默认为 True
    check_category_order: bool = True,  # 是否比较内部分类的顺序，默认为 True
    check_freq: bool = True,  # 是否检查 DatetimeIndex 或 TimedeltaIndex 上的 `freq` 属性，默认为 True
    check_flags: bool = True,  # 是否检查 `flags` 属性，默认为 True
    rtol: float | lib.NoDefault = lib.no_default,  # 相对容差，仅当 check_exact 为 False 时使用，默认为 lib.no_default
    atol: float | lib.NoDefault = lib.no_default,  # 绝对容差，仅当 check_exact 为 False 时使用，默认为 lib.no_default
    obj: str = "Series",  # 被比较对象的名称，默认为 'Series'
    *,
    check_index: bool = True,  # 是否检查索引的等价性，默认为 True
    check_like: bool = False,  # 是否检查两个 Series 是否相似，默认为 False
) -> None:
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
        First Series to compare.
    right : Series
        Second Series to compare.
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly. This also applies when checking
        Index equivalence.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.
    """
    # 定义一个布尔型参数 check_like，默认为 False
    # 如果为 True，则忽略索引顺序。如果 check_index 为 False，则必须为 False。
    # 注意：相同的标签必须对应相同的数据。

    # 引入自 1.5.0 版本新增功能

    # 参见
    # --------
    # testing.assert_index_equal : 检查两个索引是否相等。
    # testing.assert_frame_equal : 检查两个 DataFrame 是否相等。

    # 示例
    # --------
    # 导入 pandas 的 testing 模块作为 tm
    # 创建两个 Series 对象 a 和 b
    # 使用 tm.assert_series_equal 检查它们是否相等
    """
    __tracebackhide__ = True
    # 如果 check_exact、rtol、atol 都不是 lib.no_default
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        # 检查左右两边的数据类型是否为数值型且不是浮点型
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
        # 如果左边索引的层级为 1，则左索引数据类型为单一类型列表，否则为多层索引数据类型列表
        left_index_dtypes = (
            [left.index.dtype] if left.index.nlevels == 1 else left.index.dtypes
        )
        # 如果右边索引的层级为 1，则右索引数据类型为单一类型列表，否则为多层索引数据类型列表
        right_index_dtypes = (
            [right.index.dtype] if right.index.nlevels == 1 else right.index.dtypes
        )
        # 检查是否需要精确的索引比较
        check_exact_index = all(
            dtype.kind in "iu" for dtype in left_index_dtypes
        ) or all(dtype.kind in "iu" for dtype in right_index_dtypes)
    
    # 如果 check_exact 是 lib.no_default
    elif check_exact is lib.no_default:
        # 将 check_exact 设置为 False
        check_exact = False
        # 将 check_exact_index 设置为 False
        check_exact_index = False
    else:
        # 否则，直接使用 check_exact 的值来设定 check_exact_index
        check_exact_index = check_exact

    # 如果 rtol 不是 lib.no_default，则使用给定的 rtol，否则设为 1.0e-5
    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    # 如果 atol 不是 lib.no_default，则使用给定的 atol，否则设为 1.0e-8
    atol = atol if atol is not lib.no_default else 1.0e-8

    # 如果 check_index 为 False 且 check_like 为 True，则抛出 ValueError
    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # 验证 left 和 right 是否为 Series 类型
    _check_isinstance(left, right, Series)

    # 如果 check_series_type 为 True，则确保 left 和 right 的类相同
    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # 比较左右两边 Series 的长度是否相等，若不等则抛出详细的 AssertionError
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    # 如果 check_flags 为 True，则确保左右两边的 flags 属性相等，否则抛出 AssertionError
    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"

    # 如果 check_index 为 True，则确保左右两边的索引相等
    if check_index:
        # GH #38183
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact_index,
            check_categorical=check_categorical,
            # 如果 check_like 为 True，则不检查索引顺序
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    # 如果 check_like 为 True，则将左边的 Series 重新索引以与右边保持一致
    if check_like:
        left = left.reindex_like(right)

    # 如果 check_freq 为 True 且左边索引是 DatetimeIndex 或 TimedeltaIndex 类型，则验证其频率是否相等
    if check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex)):
        lidx = left.index
        ridx = right.index
        assert lidx.freq == ridx.freq, (lidx.freq, ridx.freq)
    # 如果需要检查数据类型
    if check_dtype:
        # 当 `check_categorical` 为 False 时，跳过精确的数据类型检查。
        # 如果其中一个是 `Categorical`，无论 `check_categorical` 如何，都会触发异常。
        if (
            isinstance(left.dtype, CategoricalDtype)
            and isinstance(right.dtype, CategoricalDtype)
            and not check_categorical
        ):
            # 什么也不做，跳过检查
            pass
        else:
            # 断言左右对象的属性 "dtype" 相等，显示对象为 `{obj}` 的属性。
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    # 如果需要检查精确性
    if check_exact:
        # 获取左右对象的值
        left_values = left._values
        right_values = right._values

        # 只有当数据类型是数值型时才进行精确性检查
        if isinstance(left_values, ExtensionArray) and isinstance(
            right_values, ExtensionArray
        ):
            # 断言两个扩展数组相等
            assert_extension_array_equal(
                left_values,
                right_values,
                check_dtype=check_dtype,
                index_values=left.index,
                obj=str(obj),
            )
        else:
            # 如果不是扩展数组，则转换为 NumPy 数组进行比较（如果之前的检查没有通过会引发异常）
            lv, rv = left_values, right_values
            if isinstance(left_values, ExtensionArray):
                lv = left_values.to_numpy()
            if isinstance(right_values, ExtensionArray):
                rv = right_values.to_numpy()
            # 断言两个 NumPy 数组相等
            assert_numpy_array_equal(
                lv,
                rv,
                check_dtype=check_dtype,
                obj=str(obj),
                index_values=left.index,
            )

    # 如果需要检查日期时间兼容性，并且左右对象的数据类型需要进行整数转换
    elif check_datetimelike_compat and (
        needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype)
    ):
        # 仅在具有兼容数据类型时进行检查
        # 例如，整数和 M|m 不兼容，但在这种情况下我们可以简单地检查值

        # 即使日期时间可能是不同对象（例如 datetime.datetime vs Timestamp），它们可能相等
        if not Index(left._values).equals(Index(right._values)):
            # 抛出异常，说明左值和右值在日期时间兼容性条件下不相等
            msg = (
                f"[datetimelike_compat=True] {left._values} "
                f"is not equal to {right._values}."
            )
            raise AssertionError(msg)

    # 如果左右对象的数据类型是区间类型
    elif isinstance(left.dtype, IntervalDtype) and isinstance(
        right.dtype, IntervalDtype
    ):
        # 断言两个区间数组相等
        assert_interval_array_equal(left.array, right.array)

    # 如果左右对象的数据类型是分类类型之一
    elif isinstance(left.dtype, CategoricalDtype) or isinstance(
        right.dtype, CategoricalDtype
    ):
        # 断言两个分类值相等
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # 如果左右对象的数据类型是扩展类型之一
    elif isinstance(left.dtype, ExtensionDtype) and isinstance(
        right.dtype, ExtensionDtype
    ):
        # 在这里继续添加相应的处理代码（未提供的部分）
    # 如果左右两个扩展数组的值相等，则断言通过
    assert_extension_array_equal(
        left._values,
        right._values,
        rtol=rtol,  # 相对误差容忍度
        atol=atol,  # 绝对误差容忍度
        check_dtype=check_dtype,  # 是否检查数据类型
        index_values=left.index,  # 左侧扩展数组的索引值
        obj=str(obj),  # 对象的字符串表示
    )
    # 如果左右两个扩展数组的数据类型需要转换为int64，则断言通过
    elif is_extension_array_dtype_and_needs_i8_conversion(
        left.dtype, right.dtype
    ) or is_extension_array_dtype_and_needs_i8_conversion(right.dtype, left.dtype):
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,  # 是否检查数据类型
            index_values=left.index,  # 左侧扩展数组的索引值
            obj=str(obj),  # 对象的字符串表示
        )
    # 如果左右两个扩展数组的数据类型需要转换为int64，则断言通过
    elif needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype):
        # DatetimeArray or TimedeltaArray
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,  # 是否检查数据类型
            index_values=left.index,  # 左侧扩展数组的索引值
            obj=str(obj),  # 对象的字符串表示
        )
    else:
        # 如果左右两个扩展数组的值几乎相等，则断言通过
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,  # 相对误差容忍度
            atol=atol,  # 绝对误差容忍度
            check_dtype=bool(check_dtype),  # 是否检查数据类型
            obj=str(obj),  # 对象的字符串表示
            index_values=left.index,  # 左侧扩展数组的索引值
        )

    # 元数据比较
    if check_names:
        # 检查左右两个对象的名称是否相等
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical:
        # 如果需要检查分类数据
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            # 断言左右两个分类数据相等
            assert_categorical_equal(
                left._values,
                right._values,
                obj=f"{obj} category",  # 对象的分类数据字符串表示
                check_category_order=check_category_order,  # 是否检查分类顺序
            )
# 定义一个函数，用于比较两个 DataFrame 是否相等
def assert_frame_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,  # 是否检查 DataFrame 的数据类型是否完全相同
    check_index_type: bool | Literal["equiv"] = "equiv",  # 是否检查索引的类型是否完全相同
    check_column_type: bool | Literal["equiv"] = "equiv",  # 是否检查列的类型是否完全相同
    check_frame_type: bool = True,  # 是否检查 DataFrame 的类是否完全相同
    check_names: bool = True,  # 是否检查 DataFrame 的索引和列的名称是否完全相同
    by_blocks: bool = False,  # 是否按块比较内部数据，False 表示按列比较
    check_exact: bool | lib.NoDefault = lib.no_default,  # 是否精确比较数值
    check_datetimelike_compat: bool = False,  # 是否比较日期时间类数据，忽略其数据类型
    check_categorical: bool = True,  # 是否精确比较分类数据
    check_like: bool = False,  # 是否忽略索引和列的顺序
    check_freq: bool = True,  # 是否检查日期时间索引或时间间隔索引的频率属性
    check_flags: bool = True,  # 是否检查 flags 属性
    rtol: float | lib.NoDefault = lib.no_default,  # 相对误差比较的容忍度
    atol: float | lib.NoDefault = lib.no_default,  # 绝对误差比较的容忍度
    obj: str = "DataFrame",  # 比较对象的名称，默认为 "DataFrame"
) -> None:
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    """
    rtol : float, default 1e-5
        相对容差。仅在 check_exact 为 False 时使用。
    atol : float, default 1e-8
        绝对容差。仅在 check_exact 为 False 时使用。
    obj : str, default 'DataFrame'
        指定正在比较的对象名称，用于内部显示适当的断言消息。

    See Also
    --------
    assert_series_equal : 断言 Series 相等的等效方法。
    DataFrame.equals : 检查 DataFrame 是否相等。

    Examples
    --------
    这个例子展示了比较两个相等的 DataFrame，但列具有不同的数据类型。

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    df1 等于它自身。

    >>> assert_frame_equal(df1, df1)

    df1 与 df2 不同，因为列 'b' 具有不同的数据类型。

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: DataFrame.iloc[:, 1] 的属性 (列名="b") 不同

    属性 "dtype" 不同
    [left]:  int64
    [right]: float64

    使用 check_dtype 忽略列中的不同数据类型。

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True
    _rtol = rtol if rtol is not lib.no_default else 1.0e-5
    _atol = atol if atol is not lib.no_default else 1.0e-8
    _check_exact = check_exact if check_exact is not lib.no_default else False

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(
            obj, f"{obj} shape mismatch", f"{left.shape!r}", f"{right.shape!r}"
        )

    if check_flags:
        assert left.flags == right.flags, f"{left.flags!r} != {right.flags!r}"

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        left = left.reindex_like(right)

    # compare by blocks
    # 如果按块比较
    if by_blocks:
        # 获取右DataFrame的块字典
        rblocks = right._to_dict_of_blocks()
        # 获取左DataFrame的块字典
        lblocks = left._to_dict_of_blocks()
        # 对于所有数据类型，确保左右DataFrame都包含相同的数据类型
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks  # 确保数据类型在左DataFrame的块字典中
            assert dtype in rblocks  # 确保数据类型在右DataFrame的块字典中
            # 使用 assert_frame_equal 函数比较同一数据类型的块
            assert_frame_equal(
                lblocks[dtype],  # 左DataFrame中当前数据类型的块
                rblocks[dtype],  # 右DataFrame中当前数据类型的块
                check_dtype=check_dtype,  # 检查数据类型是否相同
                obj=obj  # 对象名称或标识，用于错误消息
            )

    # 否则按列比较
    else:
        # 对于左DataFrame的每一列进行遍历
        for i, col in enumerate(left.columns):
            # 我们已经检查过列匹配，因此可以使用快速的基于位置的查找
            lcol = left._ixs(i, axis=1)  # 获取左DataFrame第i列
            rcol = right._ixs(i, axis=1)  # 获取右DataFrame第i列

            # GH #38183
            # 使用 check_index=False，因为我们不想为每列运行 assert_index_equal，
            # 因为我们已经在整个DataFrame之前检查过它。
            assert_series_equal(
                lcol,  # 左DataFrame第i列
                rcol,  # 右DataFrame第i列
                check_dtype=check_dtype,  # 检查数据类型是否相同
                check_index_type=check_index_type,  # 检查索引类型是否相同
                check_exact=check_exact,  # 检查是否完全相同
                check_names=check_names,  # 检查名称是否相同
                check_datetimelike_compat=check_datetimelike_compat,  # 检查日期时间兼容性
                check_categorical=check_categorical,  # 检查分类数据类型
                check_freq=check_freq,  # 检查频率
                obj=f'{obj}.iloc[:, {i}] (column name="{col}")',  # 对象名称或标识，用于错误消息
                rtol=rtol,  # 相对误差的容差
                atol=atol,  # 绝对误差的容差
                check_index=False,  # 不检查索引是否相同
                check_flags=False,  # 不检查标志位是否相同
            )
# 定义一个函数 assert_equal，用于比较两个对象的相等性
def assert_equal(left, right, **kwargs) -> None:
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    # 设置特殊的 traceback 隐藏标志
    __tracebackhide__ = True

    # 根据 left 的类型分别进行断言
    if isinstance(left, Index):
        # 若 left 是 Index 类型，则调用 assert_index_equal 函数进行比较
        assert_index_equal(left, right, **kwargs)
        # 如果 left 是 DatetimeIndex 或 TimedeltaIndex 类型，还要检查频率是否相同
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        # 若 left 是 Series 类型，则调用 assert_series_equal 函数进行比较
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        # 若 left 是 DataFrame 类型，则调用 assert_frame_equal 函数进行比较
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        # 若 left 是 IntervalArray 类型，则调用 assert_interval_array_equal 函数进行比较
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        # 若 left 是 PeriodArray 类型，则调用 assert_period_array_equal 函数进行比较
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        # 若 left 是 DatetimeArray 类型，则调用 assert_datetime_array_equal 函数进行比较
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        # 若 left 是 TimedeltaArray 类型，则调用 assert_timedelta_array_equal 函数进行比较
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        # 若 left 是 ExtensionArray 类型，则调用 assert_extension_array_equal 函数进行比较
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        # 若 left 是 np.ndarray 类型，则调用 assert_numpy_array_equal 函数进行比较
        assert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        # 若 left 是 str 类型，则断言 kwargs 为空，并比较 left 和 right 是否相等
        assert kwargs == {}
        assert left == right
    else:
        # 对于其他类型的 left，断言 kwargs 为空，并使用 assert_almost_equal 进行比较
        assert kwargs == {}
        assert_almost_equal(left, right)


# 定义一个函数 assert_sp_array_equal，用于比较 SparseArray 类型的对象
def assert_sp_array_equal(left, right) -> None:
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    # 检查 left 和 right 是否都是 SparseArray 类型的对象
    _check_isinstance(left, right, pd.arrays.SparseArray)

    # 比较 left 和 right 的 sp_values 是否相等
    assert_numpy_array_equal(left.sp_values, right.sp_values)

    # 断言 left 和 right 的 SparseIndex 类型是否为 SparseIndex
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)

    # 获取 left 和 right 的 SparseIndex 对象
    left_index = left.sp_index
    right_index = right.sp_index

    # 如果 left_index 不等于 right_index，则抛出异常
    if not left_index.equals(right_index):
        raise_assert_detail(
            "SparseArray.index", "index are not equal", left_index, right_index
        )
    else:
        # 否则，仅确保通过
        pass

    # 比较 left 和 right 的 fill_value 属性是否相等
    assert_attr_equal("fill_value", left, right)
    # 比较 left 和 right 的 dtype 属性是否相等
    assert_attr_equal("dtype", left, right)
    # 比较 left 和 right 的 to_dense() 方法生成的密集数组是否相等
    assert_numpy_array_equal(left.to_dense(), right.to_dense())


# 定义一个函数 assert_contains_all，用于断言一个可迭代对象中的所有元素都包含在字典中
def assert_contains_all(iterable, dic) -> None:
    for k in iterable:
        # 断言 k 是否在字典 dic 中，若不在则抛出异常
        assert k in dic, f"Did not contain item: {k!r}"


# 定义一个函数 assert_copy，用于比较两个可迭代对象中的元素是否相等，但不是同一个对象
def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    # 遍历两个迭代器（iter1 和 iter2）中的元素，依次取出 elem1 和 elem2
    for elem1, elem2 in zip(iter1, iter2):
        # 使用几乎相等的方式（assert_almost_equal）比较 elem1 和 elem2 是否相等
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        # 构造错误消息，指示 elem1 和 elem2 应为不同对象，但它们却是同一个对象
        msg = (
            f"Expected object {type(elem1)!r} and object {type(elem2)!r} to be "
            "different objects, but they were the same object."
        )
        # 使用 is 运算符检查 elem1 和 elem2 是否不是同一个对象，如果是同一个对象则触发 AssertionError
        assert elem1 is not elem2, msg
# 检查左右数据类型是否满足 ExtensionArraydtype 和需要转换为 int64 的条件
def is_extension_array_dtype_and_needs_i8_conversion(
    left_dtype: DtypeObj, right_dtype: DtypeObj
) -> bool:
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(right_dtype)


# 断言通过位置索引和标签索引获取的 Series 片段等价
def assert_indexing_slices_equivalent(ser: Series, l_slc: slice, i_slc: slice) -> None:
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    # 获取预期的 iloc 结果
    expected = ser.iloc[i_slc]

    # 断言 loc 索引和预期结果相等
    assert_series_equal(ser.loc[l_slc], expected)

    # 如果索引不是整数类型
    if not is_integer_dtype(ser.index):
        # 对于整数索引，.loc 和普通的获取元素操作是基于位置的
        assert_series_equal(ser[l_slc], expected)


# 断言 DataFrame 或 Series 的元数据属性等价
def assert_metadata_equivalent(
    left: DataFrame | Series, right: DataFrame | Series | None = None
) -> None:
    """
    Check that ._metadata attributes are equivalent.
    """
    # 遍历 left 对象的所有 _metadata 属性
    for attr in left._metadata:
        # 获取 left 对象的属性值
        val = getattr(left, attr, None)
        # 如果 right 参数为 None，则期望该属性值也为 None
        if right is None:
            assert val is None
        else:
            # 否则，断言 left 和 right 对象的同名属性值相等
            assert val == getattr(right, attr, None)
```