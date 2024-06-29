# `D:\src\scipysrc\pandas\pandas\core\sorting.py`

```
"""miscellaneous sorting / groupby utilities"""

from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    algos,
    hashtable,
    lib,
)
from pandas._libs.hashtable import unique_label_indices

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
)
from pandas.core.dtypes.generic import (
    ABCMultiIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.missing import isna

from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        IndexKeyFunc,
        Level,
        NaPosition,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        MultiIndex,
        Series,
    )
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index


def get_indexer_indexer(
    target: Index,
    level: Level | list[Level] | None,
    ascending: list[bool] | bool,
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> npt.NDArray[np.intp] | None:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
        The target index object to be sorted.
    level : int or level name or list of ints or list of level names
        Specifies which level(s) of a MultiIndex to sort.
    ascending : bool or list of bools, default True
        Whether to sort in ascending (True) or descending (False) order.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
        Sorting algorithm to use.
    na_position : {'first', 'last'}
        Specifies where NaNs are placed in the sorted index.
    sort_remaining : bool
        Whether to sort remaining levels if the levels specified by 'level' are sorted.
    key : callable, optional
        Callable function used to map values before sorting.

    Returns
    -------
    Optional[ndarray[intp]]
        The indexer for the new index.
    """

    # error: Incompatible types in assignment (expression has type
    # "Union[ExtensionArray, ndarray[Any, Any], Index, Series]", variable has
    # type "Index")
    # Ensure 'target' is correctly mapped using 'key' function
    target = ensure_key_mapped(target, key, levels=level)  # type: ignore[assignment]

    # Sort 'target' index levels in a monotonic manner
    target = target._sort_levels_monotonic()

    if level is not None:
        # Sort by specified 'level' for MultiIndex
        _, indexer = target.sortlevel(
            level,
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
        )
    elif (np.all(ascending) and target.is_monotonic_increasing) or (
        not np.any(ascending) and target.is_monotonic_decreasing
    ):
        # If already monotonic and ascending or descending, return None
        return None
    elif isinstance(target, ABCMultiIndex):
        # Get codes for sorting and create indexer using lexsort
        codes = [lev.codes for lev in target._get_codes_for_sorting()]
        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )
    else:
        # 如果 ascending 参数不是布尔值，则将其转换为布尔值
        # 对目标 target 进行排序，并返回排序后的索引数组
        indexer = nargsort(
            target,
            kind=kind,
            ascending=cast(bool, ascending),
            na_position=na_position,
        )
    # 返回排序后的索引数组
    return indexer
    # 定义函数 `get_group_index`，用于获取一组标签在完全有序的笛卡尔积中的偏移索引
    # 这些索引能够在int64边界内表示；否则，尽管组索引标识了唯一的标签组合，但无法解构。
    # - 如果 `sort` 为真，则返回的 id 等级保留标签的字典序等级。
    #   即返回的 id 可用于对标签进行字典序排序；
    # - 如果 `xnull` 为真，空值（-1标签）会被传递。

    def _int64_cut_off(shape) -> int:
        # 计算乘积，直到超过int64的最大值
        acc = 1
        for i, mul in enumerate(shape):
            acc *= int(mul)
            # 如果超过了lib.i8max，则返回当前索引i
            if not acc < lib.i8max:
                return i
        # 如果没有超过，则返回shape的长度
        return len(shape)

    def maybe_lift(lab, size: int) -> tuple[np.ndarray, int]:
        # 提升nan值（在lab数组中分配了-1标签），使所有输出值非负
        return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)

    # 将labels中的每个数组确保为int64类型
    labels = [ensure_int64(x) for x in labels]
    # 将shape转换为列表形式
    lshape = list(shape)
    # 如果不允许xnull，则处理labels和shape
    if not xnull:
        for i, (lab, size) in enumerate(zip(labels, shape)):
            # 可能提升lab数组中的标签值
            labels[i], lshape[i] = maybe_lift(lab, size)

    # 迭代处理所有标签，每个块的大小小于lib.i8max，需要的唯一int id将会较少
    # 持续循环直到条件满足退出
    while True:
        # 计算不会溢出的层级数
        nlev = _int64_cut_off(lshape)

        # 计算前 `nlev` 层的扁平化标识
        stride = np.prod(lshape[1:nlev], dtype="i8")
        out = stride * labels[0].astype("i8", subok=False, copy=False)

        # 遍历每一层级，计算累加的标识
        for i in range(1, nlev):
            if lshape[i] == 0:
                stride = np.int64(0)
            else:
                stride //= lshape[i]
            out += labels[i] * stride

        # 如果需要排除空值
        if xnull:
            # 创建空值掩码
            mask = labels[0] == -1
            for lab in labels[1:nlev]:
                mask |= lab == -1
            # 将空值标记为 -1
            out[mask] = -1

        # 如果达到了层级的最大数，结束循环
        if nlev == len(lshape):
            break

        # 压缩已完成的部分，以避免溢出
        # 保持词法顺序，应对观测标识进行排序
        comp_ids, obs_ids = compress_group_index(out, sort=sort)

        # 更新标签和形状以反映压缩后的状态
        labels = [comp_ids] + labels[nlev:]
        lshape = [len(obs_ids)] + lshape[nlev:]

    # 返回处理后的结果标识
    return out
# 定义函数get_compressed_ids，用于从标签和大小的形状创建压缩后的组索引
def get_compressed_ids(
    labels, sizes: Shape
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
        包含标签数组的列表
    sizes : tuple[int] of size of the levels
        级别大小的元组

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids，压缩后的组索引
    np.ndarray[np.int64]
        obs_group_ids，观测组的唯一标签列表
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)


# 判断是否可能发生int64溢出的函数
def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


# 解压组索引的私有函数_decons_group_index
def _decons_group_index(
    comp_labels: npt.NDArray[np.intp], shape: Shape
) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from compressed group indices.

    Parameters
    ----------
    comp_labels : np.ndarray[np.intp]
        压缩的组索引
    shape : tuple[int]
        形状

    Returns
    -------
    list[np.ndarray[np.intp]]
        重构的标签列表
    """
    # reconstruct labels
    if is_int64_overflow_possible(shape):
        # at some point group indices are factorized,
        # and may not be deconstructed here! wrong path!
        raise ValueError("cannot deconstruct factorized group indices!")

    label_list = []
    factor = 1
    y = np.array(0)
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (x - y) % (factor * shape[i]) // factor
        np.putmask(labels, comp_labels < 0, -1)
        label_list.append(labels)
        y = labels * factor
        factor *= shape[i]
    return label_list[::-1]


# 从观测组id解析标签的函数decons_obs_group_ids
def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: Shape,
    labels: Sequence[npt.NDArray[np.signedinteger]],
    xnull: bool,
) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
        压缩的组索引
    obs_ids: np.ndarray[np.intp]
        观测组id
    shape : tuple[int]
        形状
    labels : Sequence[np.ndarray[np.signedinteger]]
        标签序列
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
        如果排除空值；即通过-1标签。

    Returns
    -------
    list[np.ndarray[np.intp]]
        重构的标签列表
    """
    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype=np.intp)
        arr_shape = np.asarray(shape, dtype=np.intp) + lift
        shape = tuple(arr_shape)

    if not is_int64_overflow_possible(shape):
        # obs ids are deconstructable! take the fast route!
        out = _decons_group_index(obs_ids, shape)
        return out if xnull or not lift.any() else [x - y for x, y in zip(out, lift)]

    indexer = unique_label_indices(comp_ids)
    return [lab[indexer].astype(np.intp, subok=False, copy=True) for lab in labels]


# 执行基于键的词法排序的函数lexsort_indexer
def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders=None,
    na_position: str = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> npt.NDArray[np.intp]:
    """
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        要排序的键集合
    orders : None
        排序顺序
    na_position : str, optional
        缺失值位置，默认为"last"
    key : Callable | None, optional
        自定义键值函数，默认为None
    codes_given : bool, optional
        是否已提供编码，默认为False

    Returns
    -------
    np.ndarray[np.intp]
        排序后的索引
    """
    # keys: 要排序的数组序列，可以是数组、索引或者Series
    #       如果key不为None，则序列中应包含Series。
    orders : bool or list of booleans, optional
    # 确定每个keys中元素的排序顺序。如果是一个列表，
    # 其长度必须与keys相同。决定是否按升序（True）或降序（False）排序。
    # 如果是bool类型，则应用于所有元素。如果是None，则默认为True。
    na_position : {'first', 'last'}, default 'last'
    # 确定排序后NA元素在排序列表中的位置，可以是"first"或"last"
    key : Callable, optional
    # 可调用的key函数，对keys中的每个元素进行排序前的映射处理。
    codes_given: bool, False
    # 如果提供了codes，则避免分类材料化。

    Returns
    -------
    np.ndarray[np.intp]
    """
    from pandas.core.arrays import Categorical

    # 检查na_position参数是否有效，只能为"first"或"last"
    if na_position not in ["last", "first"]:
        raise ValueError(f"invalid na_position: {na_position}")

    # 根据orders参数的类型初始化排序顺序的生成器
    if isinstance(orders, bool):
        orders = itertools.repeat(orders, len(keys))
    elif orders is None:
        orders = itertools.repeat(True, len(keys))
    else:
        orders = reversed(orders)

    # 存储每个key的排序结果的标签列表
    labels = []

    # 反向迭代keys和对应的orders
    for k, order in zip(reversed(keys), orders):
        # 确保每个key都经过key函数映射处理
        k = ensure_key_mapped(k, key)
        # 如果提供了codes，则直接转换为np.ndarray
        if codes_given:
            codes = cast(np.ndarray, k)
            n = codes.max() + 1 if len(codes) else 0
        else:
            # 否则创建分类变量并获取其codes
            cat = Categorical(k, ordered=True)
            codes = cat.codes
            n = len(cat.categories)

        # 创建一个标记，表示codes中的NA值
        mask = codes == -1

        # 如果na_position为"last"且存在NA值，则将其转换为n
        if na_position == "last" and mask.any():
            codes = np.where(mask, n, codes)

        # 如果order为False（降序），则对codes进行相应的转换
        if not order:
            codes = np.where(mask, codes, n - codes - 1)

        # 将处理后的codes添加到labels列表中
        labels.append(codes)

    # 使用lexsort函数对labels进行词典序排序，并返回排序后的索引数组
    return np.lexsort(labels)
# 定义一个处理 ExtensionArray 的 argmin 或 argmax 操作的函数
def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
        要处理的扩展数组
    method : {"argmax", "argmin"}
        指定执行 argmax 或 argmin 操作
    axis : int, default 0
        沿着哪个轴执行操作，默认是第一个轴

    Returns
    -------
    int
        返回找到的最小或最大值的索引
    """
    assert method in {"argmax", "argmin"}
    # 根据 method 选择要执行的函数，是 argmax 还是 argmin
    func = np.argmax if method == "argmax" else np.argmin

    # 将缺失值转换为布尔掩码
    mask = np.asarray(isna(values))
    # 调用 `_values_for_argsort()` 方法，返回用于排序的值数组
    arr_values = values._values_for_argsort()

    # 检查数组的维度是否大于 1
    if arr_values.ndim > 1:
        # 如果存在掩码数组中有任何 True 的元素
        if mask.any():
            # 如果轴参数为 1，将值数组和掩码数组压缩成元组序列
            if axis == 1:
                zipped = zip(arr_values, mask)
            # 否则，将值数组和掩码数组的转置压缩成元组序列
            else:
                zipped = zip(arr_values.T, mask.T)
            # 对压缩后的序列中的每个元组调用 `_nanargminmax()` 函数，返回结果数组
            return np.array([_nanargminmax(v, m, func) for v, m in zipped])
        
        # 如果掩码数组中没有 True 的元素，直接对值数组应用函数 `func`
        return func(arr_values, axis=axis)

    # 如果值数组的维度不大于 1，直接调用 `_nanargminmax()` 函数处理
    return _nanargminmax(arr_values, mask, func)
def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    See nanargminmax.__doc__.
    """
    # 创建一个索引数组，范围是values数组的长度
    idx = np.arange(values.shape[0])
    # 从values数组中提取非NaN值
    non_nans = values[~mask]
    # 从索引数组中提取对应的非NaN索引
    non_nan_idx = idx[~mask]

    # 返回非NaN值中根据func函数计算出的索引
    return non_nan_idx[func(non_nans)]


def _ensure_key_mapped_multiindex(
    index: MultiIndex, key: Callable, level=None
) -> MultiIndex:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None). Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        Index to which to apply the key function on the
        specified levels.
    key : Callable
        Function that takes an Index and returns an Index of
        the same shape. This key is applied to each level
        separately. The name of the level can be used to
        distinguish different levels for application.
    level : list-like, int or str, default None
        Level or list of levels to apply the key function to.
        If None, key function is applied to all levels. Other
        levels are left unchanged.

    Returns
    -------
    labels : MultiIndex
        Resulting MultiIndex with modified levels.
    """

    # 如果level不为None，则根据level确定要应用key函数的级别
    if level is not None:
        if isinstance(level, (str, int)):
            level_iter = [level]
        else:
            level_iter = level

        # 获取要排序的级别
        sort_levels: range | set = {index._get_level_number(lev) for lev in level_iter}
    else:
        sort_levels = range(index.nlevels)

    # 对每个级别应用key函数，返回一个新的MultiIndex
    mapped = [
        ensure_key_mapped(index._get_level_values(level), key)
        if level in sort_levels
        else index._get_level_values(level)
        for level in range(index.nlevels)
    ]

    return type(index).from_arrays(mapped)


def ensure_key_mapped(
    values: ArrayLike | Index | Series, key: Callable | None, levels=None
) -> ArrayLike | Index | Series:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    from pandas.core.indexes.api import Index

    # 如果key为None，则直接返回values
    if not key:
        return values

    # 如果values是ABCMultiIndex的实例，则调用_ensure_key_mapped_multiindex函数
    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)

    # 对values应用key函数，并检查结果是否与原始值具有相同的形状
    result = key(values.copy())
    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change the shape of the array."
        )
    try:
        # 检查是否 values 是 Index 类型的实例
        if isinstance(
            values, Index
        ):  # 转换为一个新的 Index 子类，不一定与原来相同
            result = Index(result, tupleize_cols=False)
        else:
            # 否则尝试回到原始类型
            type_of_values = type(values)
            # 使用原始类型构造 result 对象
            result = type_of_values(result)  # type: ignore[call-arg]
    except TypeError as err:
        # 抛出类型错误异常，指明用户提供的 `key` 函数返回了一个无效类型的 result 对象，
        # 无法转换成 values 的类型
        raise TypeError(
            f"User-provided `key` function returned an invalid type {type(result)} \
            which could not be converted to {type(values)}."
        ) from err

    # 返回处理后的 result 对象
    return result
# 定义函数 get_indexer_dict，接收标签列表和索引键列表作为输入，返回标签映射到索引器的字典
def get_indexer_dict(
    label_list: list[np.ndarray], keys: list[Index]
) -> dict[Hashable, npt.NDArray[np.intp]]:
    # 计算 keys 中每个索引键的长度，构成形状元组
    shape = tuple(len(x) for x in keys)

    # 调用 get_group_index 函数获取分组索引，传入标签列表、形状和排序等参数
    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    
    # 如果所有的 group_index 值均为 -1，则直接返回空字典
    if np.all(group_index == -1):
        return {}

    # 计算 ngroups 的值，根据 shape 和是否可能存在 int64 溢出来确定
    ngroups = (
        ((group_index.size and group_index.max()) + 1)
        if is_int64_overflow_possible(shape)
        else np.prod(shape, dtype="i8")
    )

    # 获取用于排序的索引数组
    sorter = get_group_index_sorter(group_index, ngroups)

    # 对标签列表中的每个标签数组按照排序器排序
    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)  # 按照排序器排序 group_index

    # 调用 lib.indices_fast 函数，传入排序器、group_index、keys 和排序后的标签列表，返回结果
    return lib.indices_fast(sorter, group_index, keys, sorted_labels)


# ----------------------------------------------------------------------
# sorting levels...cleverly?


# 定义函数 get_group_index_sorter，接收分组索引和分组数作为输入，返回排序器数组
def get_group_index_sorter(
    group_index: npt.NDArray[np.intp], ngroups: int | None = None
) -> npt.NDArray[np.intp]:
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        signed integer dtype
    ngroups : int or None, default None

    Returns
    -------
    np.ndarray[np.intp]
    """
    # 如果 ngroups 为 None，则设定为 group_index 的最大值加一
    if ngroups is None:
        ngroups = 1 + group_index.max()
    
    # 计算 group_index 的长度
    count = len(group_index)
    
    # 定义 alpha 和 beta，用于决定是否执行 groupsort 的标志
    alpha = 0.0  # taking complexities literally; there may be
    beta = 1.0  # some room for fine-tuning these parameters
    do_groupsort = count > 0 and ((alpha + beta * ngroups) < (count * np.log(count)))
    
    # 根据 do_groupsort 的值选择相应的排序算法
    if do_groupsort:
        # 调用 algos.groupsort_indexer 函数进行计数排序，返回排序器和其他信息
        sorter, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index),
            ngroups,
        )
        # sorter 应该已经是 intp 类型，但是 mypy 目前无法验证

    else:
        # 如果不执行 groupsort，则使用 mergesort 算法对 group_index 进行排序
        sorter = group_index.argsort(kind="mergesort")

    # 返回排序后的排序器数组
    return ensure_platform_int(sorter)


# 定义函数 compress_group_index，接收分组索引和排序标志作为输入，返回压缩后的分组索引和组 ID 元组
def compress_group_index(
    group_index: npt.NDArray[np.int64], sort: bool = True
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
    # 如果 group_index 非空且所有元素递增排序，使用快速路径
    if len(group_index) and np.all(group_index[1:] >= group_index[:-1]):
        # GH 53806: 优化已排序的 group_index 的处理速度

        # 创建唯一性掩码，标识不同组的起始位置
        unique_mask = np.concatenate(
            [group_index[:1] > -1, group_index[1:] != group_index[:-1]]
        )

        # 使用唯一性掩码生成组标识符
        comp_ids = unique_mask.cumsum()
        comp_ids -= 1

        # 根据唯一性掩码提取观察组的索引
        obs_group_ids = group_index[unique_mask]
    else:
        # 计算 group_index 的预期大小
        size_hint = len(group_index)
        
        # 使用哈希表存储 group_index 数据
        table = hashtable.Int64HashTable(size_hint)

        # 确保 group_index 中的数据类型为 int64
        group_index = ensure_int64(group_index)

        # 获取组标签和观察组的标识符
        # 注意：组标签是按升序排列的（例如，1, 2, 3 等）
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

        # 如果需要排序并且观察组标识符列表非空，按照唯一性重新排序
        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    # 确保 comp_ids 和 obs_group_ids 的数据类型为 int64，并返回结果
    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)
# 定义一个函数，用于重新按照唯一值排序标签数组
def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64], labels: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    """
    Parameters
    ----------
    uniques : np.ndarray[np.int64]
        包含唯一值的数组，按此数组重新排序标签数组
    labels : np.ndarray[np.intp]
        待重新排序的标签数组

    Returns
    -------
    np.ndarray[np.int64]
        重新排序后的唯一值数组
    np.ndarray[np.intp]
        重新排序后的标签数组
    """

    # sorter is index where elements ought to go
    # 根据唯一值数组 uniques 的排序顺序，得到标签数组 labels 应该去的索引位置
    sorter = uniques.argsort()

    # reverse_indexer is where elements came from
    # reverse_indexer 是一个数组，表示每个元素原来所在的索引位置
    reverse_indexer = np.empty(len(sorter), dtype=np.intp)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    # mask is True for labels < 0
    # 创建一个布尔掩码，标记所有小于 0 的标签
    mask = labels < 0

    # move labels to right locations (ie, unsort ascending labels)
    # 使用 reverse_indexer 将标签数组 labels 按照排序后的顺序重新排列
    labels = reverse_indexer.take(labels)
    # 将标签数组中小于 0 的位置设置为 -1
    np.putmask(labels, mask, -1)

    # sort observed ids
    # 根据排序后的顺序重新排列唯一值数组 uniques
    uniques = uniques.take(sorter)

    # 返回重新排序后的唯一值数组和标签数组
    return uniques, labels
```