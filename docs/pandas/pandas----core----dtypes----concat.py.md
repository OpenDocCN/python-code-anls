# `D:\src\scipysrc\pandas\pandas\core\dtypes\concat.py`

```
"""
Utility functions related to concat.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    common_dtype_categorical_compat,
    find_common_type,
    np_find_common_type,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
    )

    from pandas.core.arrays import (
        Categorical,
        ExtensionArray,
    )


def _is_nonempty(x: ArrayLike, axis: AxisInt) -> bool:
    """
    Check if the input array-like object `x` is non-empty along the specified axis.

    Parameters
    ----------
    x : ArrayLike
        Array-like object to check for non-emptiness.
    axis : AxisInt
        Axis along which to check for non-emptiness.

    Returns
    -------
    bool
        True if `x` is non-empty along the specified axis, False otherwise.
    """
    # filter empty arrays
    # 1-d dtypes always are included here
    if x.ndim <= axis:
        return True
    return x.shape[axis] > 0


def concat_compat(
    to_concat: Sequence[ArrayLike], axis: AxisInt = 0, ea_compat_axis: bool = False
) -> ArrayLike:
    """
    Provide concatenation of an array of arrays each of which is a single
    'normalized' dtype (in that for example, if it's object, then it is a
    non-datetimelike and provide a combined dtype for the resulting array that
    preserves the overall dtype if possible).

    Parameters
    ----------
    to_concat : sequence of arrays
        Arrays to concatenate.
    axis : AxisInt, optional
        Axis to provide concatenation (default is 0).
    ea_compat_axis : bool, optional
        For ExtensionArray compat, behave as if axis == 1 when determining
        whether to drop empty arrays.

    Returns
    -------
    ArrayLike
        A single array, preserving the combined dtypes.
    """
    if len(to_concat) and lib.dtypes_all_equal([obj.dtype for obj in to_concat]):
        # fastpath!
        obj = to_concat[0]
        if isinstance(obj, np.ndarray):
            to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
            return np.concatenate(to_concat_arrs, axis=axis)

        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        if ea_compat_axis:
            # We have 1D objects, that don't support axis keyword
            return obj._concat_same_type(to_concat_eas)
        elif axis == 0:
            return obj._concat_same_type(to_concat_eas)
        else:
            # e.g. DatetimeArray
            # NB: We are assuming here that ensure_wrapped_if_arraylike has
            #  been called where relevant.
            return obj._concat_same_type(
                # error: Unexpected keyword argument "axis" for "_concat_same_type"
                # of "ExtensionArray"
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )

    # If all arrays are empty, there's nothing to convert, just short-cut to
    # the concatenation, #3121.
    #
    # Creating an empty array directly is tempting, but the winnings would be
    # marginal given that it would still require shape & dtype calculation and
    # 从待合并的对象列表中筛选出非空的对象，使用_is_nonempty函数进行判断
    non_empties = [x for x in to_concat if _is_nonempty(x, axis)]

    # 调用_get_result_dtype函数获取合并结果的数据类型相关信息：是否包含ExtensionArray，数据种类，目标数据类型
    any_ea, kinds, target_dtype = _get_result_dtype(to_concat, non_empties)

    # 如果存在目标数据类型，则将待合并的对象列表中的每个数组转换为指定的目标数据类型
    if target_dtype is not None:
        to_concat = [astype_array(arr, target_dtype, copy=False) for arr in to_concat]

    # 如果待合并的对象列表中的第一个对象不是np.ndarray类型，而是ExtensionArray类型
    if not isinstance(to_concat[0], np.ndarray):
        # 即 isinstance(to_concat[0], ExtensionArray)

        # 将to_concat列表中的每个ExtensionArray对象转换为Sequence[ExtensionArray]
        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        # 获取第一个ExtensionArray对象的类
        cls = type(to_concat[0])

        # 对于某些类，类方法`_concat_same_type()`可能不支持axis关键字
        # 如果ea_compat_axis为True或axis为0，则调用类方法_cls._concat_same_type(to_concat_eas)
        if ea_compat_axis or axis == 0:
            return cls._concat_same_type(to_concat_eas)
        else:
            # 否则，调用类方法_cls._concat_same_type(to_concat_eas, axis=axis)
            return cls._concat_same_type(
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )
    else:
        # 如果待合并的对象列表中的第一个对象是np.ndarray类型

        # 将to_concat列表中的每个np.ndarray对象转换为Sequence[np.ndarray]
        to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
        # 使用np.concatenate函数将所有np.ndarray对象按指定轴axis进行合并
        result = np.concatenate(to_concat_arrs, axis=axis)

        # 如果没有ExtensionArray对象，并且结果包含"b"类别且结果的dtype种类为"iuf"
        if not any_ea and "b" in kinds and result.dtype.kind in "iuf":
            # 将结果强制转换为object类型，而不是将布尔值转换为数值类型
            result = result.astype(object, copy=False)

    # 返回合并后的结果
    return result
def _get_result_dtype(
    to_concat: Sequence[ArrayLike], non_empties: Sequence[ArrayLike]
) -> tuple[bool, set[str], DtypeObj | None]:
    # 初始化目标数据类型为 None
    target_dtype = None

    # 获取所有待连接对象的数据类型集合
    dtypes = {obj.dtype for obj in to_concat}
    # 获取所有待连接对象的数据类型种类集合
    kinds = {obj.dtype.kind for obj in to_concat}

    # 检查是否存在非 ndarray 类型的对象
    any_ea = any(not isinstance(x, np.ndarray) for x in to_concat)
    if any_ea:
        # 即存在 ExtensionArrays

        # 在这里忽略轴向，因为内部连接 EAs 总是在 axis=0 上进行
        if len(dtypes) != 1:
            # 找到所有待连接对象的公共数据类型
            target_dtype = find_common_type([x.dtype for x in to_concat])
            # 对于分类数据类型的兼容性，进行通用数据类型的确定
            target_dtype = common_dtype_categorical_compat(to_concat, target_dtype)

    elif not len(non_empties):
        # 如果没有非空对象，但可能需要将结果数据类型强制转换为对象类型，
        # 如果存在非数值类型的操作数（否则 numpy 可能将其强制转换为 float）
        if len(kinds) != 1:
            if not len(kinds - {"i", "u", "f"}) or not len(kinds - {"b", "i", "u"}):
                # 让 numpy 自动转换
                pass
            else:
                # 强制转换为对象类型
                target_dtype = np.dtype(object)
                kinds = {"o"}
    else:
        # 错误："np_find_common_type" 的第一个参数的类型 "*Set[Union[ExtensionDtype, Any]]" 不兼容；
        # 预期类型是 "dtype[Any]"
        # 找到所有待连接对象的公共数据类型（忽略类型不匹配的错误）
        target_dtype = np_find_common_type(*dtypes)  # type: ignore[arg-type]

    # 返回三个值：是否存在 ExtensionArrays，数据类型种类集合，目标数据类型
    return any_ea, kinds, target_dtype
    >>> b = pd.Categorical(["a", "b"])
    # 创建一个包含两个类别 'a' 和 'b' 的 Categorical 对象 b
    >>> pd.api.types.union_categoricals([a, b])
    # 将列表中的多个 Categorical 对象合并成一个新的 Categorical 对象
    ['b', 'c', 'a', 'b']
    # 合并后的结果，包含四个值，并按照合并前的顺序排列，即 ['b', 'c', 'a', 'b']
    Categories (3, object): ['b', 'c', 'a']
    # 合并后的 Categorical 对象，包含三个类别 'b', 'c', 'a'

    By default, the resulting categories will be ordered as they appear
    in the `categories` of the data. If you want the categories to be
    lexsorted, use `sort_categories=True` argument.

    >>> pd.api.types.union_categoricals([a, b], sort_categories=True)
    # 使用 sort_categories=True 参数对合并后的类别进行字典序排序
    ['b', 'c', 'a', 'b']
    # 合并后的结果，按字典序排列为 ['a', 'b', 'c', 'b']
    Categories (3, object): ['a', 'b', 'c']
    # 合并后的 Categorical 对象，类别按照字典序排列为 ['a', 'b', 'c']

    `union_categoricals` also works with the case of combining two
    categoricals of the same categories and order information (e.g. what
    you could also `append` for).

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    # 创建一个有序的 Categorical 对象 a，包含类别 'a' 和 'b'
    >>> b = pd.Categorical(["a", "b", "a"], ordered=True)
    # 创建另一个有序的 Categorical 对象 b，包含类别 'a'、'b' 和 'a'
    >>> pd.api.types.union_categoricals([a, b])
    # 合并两个有序的 Categorical 对象 a 和 b
    ['a', 'b', 'a', 'b', 'a']
    # 合并后的结果，按照原始顺序 ['a', 'b', 'a', 'b', 'a'] 排列
    Categories (2, object): ['a' < 'b']
    # 合并后的 Categorical 对象，类别按照 'a' 小于 'b' 的顺序排列

    Raises `TypeError` because the categories are ordered and not identical.

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    # 创建一个有序的 Categorical 对象 a，包含类别 'a' 和 'b'
    >>> b = pd.Categorical(["a", "b", "c"], ordered=True)
    # 创建另一个有序的 Categorical 对象 b，包含类别 'a'、'b' 和 'c'
    >>> pd.api.types.union_categoricals([a, b])
    # 尝试合并两个有序但类别不同的 Categorical 对象 a 和 b
    Traceback (most recent call last):
        ...
    TypeError: to union ordered Categoricals, all categories must be the same
    # 抛出 TypeError 异常，因为要合并的有序 Categorical 对象的类别不一致

    Ordered categoricals with different categories or orderings can be
    combined by using the `ignore_ordered=True` argument.

    >>> a = pd.Categorical(["a", "b", "c"], ordered=True)
    # 创建一个有序的 Categorical 对象 a，包含类别 'a'、'b' 和 'c'
    >>> b = pd.Categorical(["c", "b", "a"], ordered=True)
    # 创建另一个有序的 Categorical 对象 b，包含类别 'c'、'b' 和 'a'
    >>> pd.api.types.union_categoricals([a, b], ignore_order=True)
    # 使用 ignore_order=True 参数合并两个有序但类别不同的 Categorical 对象 a 和 b
    ['a', 'b', 'c', 'c', 'b', 'a']
    # 合并后的结果，忽略了类别的顺序信息，按合并前的顺序排列
    Categories (3, object): ['a', 'b', 'c']
    # 合并后的 Categorical 对象，类别按合并前的顺序排列为 ['a', 'b', 'c']

    `union_categoricals` also works with a `CategoricalIndex`, or `Series`
    containing categorical data, but note that the resulting array will
    always be a plain `Categorical`

    >>> a = pd.Series(["b", "c"], dtype="category")
    # 创建一个包含类别 'b' 和 'c' 的 Series，数据类型为 category
    >>> b = pd.Series(["a", "b"], dtype="category")
    # 创建另一个包含类别 'a' 和 'b' 的 Series，数据类型为 category
    >>> pd.api.types.union_categoricals([a, b])
    # 合并两个包含类别数据的 Series a 和 b
    ['b', 'c', 'a', 'b']
    # 合并后的结果，按照合并前的顺序排列为 ['b', 'c', 'a', 'b']
    Categories (3, object): ['b', 'c', 'a']
    # 合并后的 Categorical 对象，包含三个类别 'b'、'c' 和 'a'
    ```
    from pandas import Categorical
    from pandas.core.arrays.categorical import recode_for_categories

    if len(to_union) == 0:
        raise ValueError("No Categoricals to union")
    # 如果待合并的 Categorical 对象列表为空，则抛出 ValueError 异常

    def _maybe_unwrap(x):
        if isinstance(x, (ABCCategoricalIndex, ABCSeries)):
            return x._values
        elif isinstance(x, Categorical):
            return x
        else:
            raise TypeError("all components to combine must be Categorical")
    # 定义一个内部函数 _maybe_unwrap，用于检查和返回有效的 Categorical 对象或数据

    to_union = [_maybe_unwrap(x) for x in to_union]
    # 对输入列表中的每个元素调用 _maybe_unwrap 函数，获取有效的 Categorical 对象或数据
    first = to_union[0]

    if not lib.dtypes_all_equal([obj.categories.dtype for obj in to_union]):
        raise TypeError("dtype of categories must be the same")
    # 检查待合并的所有 Categorical 对象的类别数据类型是否相同，如果不同则抛出 TypeError 异常

    ordered = False
    # 初始化 ordered 变量为 False，用于标记合并后的 Categorical 对象是否有序
    if all(first._categories_match_up_to_permutation(other) for other in to_union[1:]):
        # 检查所有要合并的分类是否相同，是则快速路径
        categories = first.categories  # 使用第一个分类对象的分类
        ordered = first.ordered  # 使用第一个分类对象的有序属性

        # 对所有要合并的分类对象进行编码，并获取编码后的代码数组
        all_codes = [first._encode_with_my_categories(x)._codes for x in to_union]
        new_codes = np.concatenate(all_codes)  # 将所有编码数组连接起来

        if sort_categories and not ignore_order and ordered:
            raise TypeError("Cannot use sort_categories=True with ordered Categoricals")
        
        if sort_categories and not categories.is_monotonic_increasing:
            # 如果需要对分类进行排序，并且分类不是单调递增的，则对分类进行排序
            categories = categories.sort_values()
            indexer = categories.get_indexer(first.categories)

            from pandas.core.algorithms import take_nd

            new_codes = take_nd(indexer, new_codes, fill_value=-1)
    elif ignore_order or all(not c.ordered for c in to_union):
        # 如果分类不完全相同，则执行并集和重新编码操作
        cats = first.categories.append([c.categories for c in to_union[1:]])
        categories = cats.unique()  # 获取所有唯一的分类值
        if sort_categories:
            categories = categories.sort_values()  # 如果需要排序分类，则对分类进行排序

        # 对每个要合并的分类对象进行重新编码，以适应新的分类
        new_codes = [
            recode_for_categories(c.codes, c.categories, categories) for c in to_union
        ]
        new_codes = np.concatenate(new_codes)  # 将所有重新编码后的代码数组连接起来
    else:
        # 如果分类是有序的，但是未正确处理，抛出适当的错误消息
        if all(c.ordered for c in to_union):
            msg = "to union ordered Categoricals, all categories must be the same"
            raise TypeError(msg)
        raise TypeError("Categorical.ordered must be the same")

    if ignore_order:
        ordered = False  # 如果忽略分类的顺序，将有序标志设置为 False

    # 创建新的分类数据类型对象
    dtype = CategoricalDtype(categories=categories, ordered=ordered)
    return Categorical._simple_new(new_codes, dtype=dtype)
```