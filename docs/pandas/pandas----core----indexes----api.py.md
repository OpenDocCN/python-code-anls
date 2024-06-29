# `D:\src\scipysrc\pandas\pandas\core\indexes\api.py`

```
# 引入未来的注释类型导入
from __future__ import annotations

# 引入类型检查相关模块
from typing import (
    TYPE_CHECKING,
    cast,
)

# 引入numpy模块，并起别名为np
import numpy as np

# 从pandas._libs中引入NaT和lib对象
from pandas._libs import (
    NaT,
    lib,
)
# 从pandas.errors中引入InvalidIndexError异常类
from pandas.errors import InvalidIndexError

# 从pandas.core.dtypes.cast中引入find_common_type函数
from pandas.core.dtypes.cast import find_common_type

# 从pandas.core.algorithms中引入safe_sort函数
from pandas.core.algorithms import safe_sort
# 从pandas.core.indexes.base中引入多个类和函数
from pandas.core.indexes.base import (
    Index,                          # 索引基类
    _new_Index,                     # 内部索引对象创建函数
    ensure_index,                   # 确保索引函数
    ensure_index_from_sequences,    # 从序列创建索引函数
    get_unanimous_names,            # 获取一致名称函数
    maybe_sequence_to_range,        # 序列转范围函数
)

# 从pandas.core.indexes.category中引入CategoricalIndex类
from pandas.core.indexes.category import CategoricalIndex
# 从pandas.core.indexes.datetimes中引入DatetimeIndex类
from pandas.core.indexes.datetimes import DatetimeIndex
# 从pandas.core.indexes.interval中引入IntervalIndex类
from pandas.core.indexes.interval import IntervalIndex
# 从pandas.core.indexes.multi中引入MultiIndex类
from pandas.core.indexes.multi import MultiIndex
# 从pandas.core.indexes.period中引入PeriodIndex类
from pandas.core.indexes.period import PeriodIndex
# 从pandas.core.indexes.range中引入RangeIndex类
from pandas.core.indexes.range import RangeIndex
# 从pandas.core.indexes.timedeltas中引入TimedeltaIndex类
from pandas.core.indexes.timedeltas import TimedeltaIndex

# 如果处于类型检查环境下，则从pandas._typing中引入Axis类型
if TYPE_CHECKING:
    from pandas._typing import Axis

# 导出的所有公共对象列表
__all__ = [
    "Index",
    "MultiIndex",
    "CategoricalIndex",
    "IntervalIndex",
    "RangeIndex",
    "InvalidIndexError",
    "TimedeltaIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "_new_Index",
    "NaT",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "union_indexes",
    "get_unanimous_names",
    "all_indexes_same",
    "default_index",
    "safe_sort_index",
    "maybe_sequence_to_range",
]


def get_objs_combined_axis(
    objs,
    intersect: bool = False,
    axis: Axis = 0,
    sort: bool = True,
) -> Index:
    """
    提取组合索引：根据给定轴上的索引，返回交集或并集（取决于"intersect"的值），
    如果所有对象都没有索引（例如它们是numpy数组），则返回None。

    Parameters
    ----------
    objs : list
        Series或DataFrame对象的列表，可能是两者的混合。
    intersect : bool, default False
        如果为True，则计算索引的交集。否则，计算并集。
    axis : {0或'index'，1或'outer'}，默认为0
        要从中提取索引的轴。
    sort : bool, 默认为True
        结果索引是否应该排序。

    Returns
    -------
    Index
    """
    # 从每个对象中获取指定轴上的索引列表
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    # 返回组合索引的结果，根据给定参数决定是交集还是并集，并指定是否排序
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    返回一个包含"objs"不同元素（不同的id）的列表。
    保留原始顺序。
    """
    # 使用集合ids来存储已经遍历过的对象id，避免重复
    ids: set[int] = set()
    res = []
    for obj in objs:
        # 如果对象的id不在ids集合中，则将其添加到结果列表中，并将其id加入ids集合
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
    indexes: list[Index],
    intersect: bool = False,
    sort: bool = False,
) -> Index:
    """
    返回索引的并集或交集。

    Parameters
    ----------
    indexes : list of Index or list objects
        当intersect=True时，不接受列表的列表。
    intersect : bool, default False
        如果为True，则返回索引的交集。否则，返回并集。
    sort : bool, 默认为False
        是否对结果索引进行排序。

    Returns
    -------
    Index
    """
    # 根据参数决定返回索引的交集或并集，并指定是否排序
    pass  # 返回值由具体实现决定
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    # 获取唯一的索引对象列表
    indexes = _get_distinct_objs(indexes)
    # 如果索引列表为空，则使用默认的空索引
    if len(indexes) == 0:
        index: Index = default_index(0)
    # 如果只有一个索引对象，则直接使用该索引
    elif len(indexes) == 1:
        index = indexes[0]
    # 如果需要计算交集
    elif intersect:
        index = indexes[0]
        # 对于除第一个索引外的其他索引，计算与第一个索引的交集
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        # 否则计算索引对象的并集，并确保返回的是索引类型
        index = union_indexes(indexes, sort=False)
        index = ensure_index(index)

    # 如果需要对结果索引进行排序，则调用安全排序函数
    if sort:
        index = safe_sort_index(index)
    # 返回最终的索引对象
    return index
def safe_sort_index(index: Index) -> Index:
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    # 如果索引是单调递增的，直接返回该索引对象
    if index.is_monotonic_increasing:
        return index

    try:
        # 尝试安全排序索引
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        # 如果安全排序成功且返回的是 Index 对象，则直接返回该对象
        if isinstance(array_sorted, Index):
            return array_sorted

        # 否则，假设 array_sorted 是 np.ndarray 类型
        array_sorted = cast(np.ndarray, array_sorted)
        # 如果原索引是 MultiIndex 类型，则从元组数组创建新的 MultiIndex
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            # 否则，创建新的 Index 对象，保留原索引的名称和数据类型
            index = Index(array_sorted, name=index.name, dtype=index.dtype)

    return index


def union_indexes(indexes, sort: bool | None = True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    # 如果索引列表为空，则抛出异常
    if len(indexes) == 0:
        raise AssertionError("Must have at least 1 Index to union")
    # 如果索引列表只有一个元素
    if len(indexes) == 1:
        result = indexes[0]
        # 如果结果是列表类型
        if isinstance(result, list):
            # 如果不需要排序，则创建一个新的 Index 对象
            if not sort:
                result = Index(result)
            else:
                # 否则，对列表进行排序后创建一个新的 Index 对象
                result = Index(sorted(result))
        return result

    # 对索引列表进行清理和检查，获取清理后的索引列表和类型信息
    indexes, kind = _sanitize_and_check(indexes)

    # 如果索引类型为 "special"
    if kind == "special":
        result = indexes[0]

        num_dtis = 0
        num_dti_tzs = 0
        for idx in indexes:
            # 统计 DatetimeIndex 的数量及具有时区信息的数量
            if isinstance(idx, DatetimeIndex):
                num_dtis += 1
                if idx.tz is not None:
                    num_dti_tzs += 1
        # 如果有混合时区的 DatetimeIndex，则抛出类型错误
        if num_dti_tzs not in [0, num_dtis]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        # 如果所有索引都是 DatetimeIndex 类型，则强制排序并选择第一个索引
        if num_dtis == len(indexes):
            sort = True
            result = indexes[0]

        elif num_dtis > 1:
            # 如果有混合时区，根据索引的顺序可能会影响类型的转换行为
            sort = False

            # TODO: what about Categorical[dt64]?
            # test_frame_from_dict_with_mixed_indexes
            # 将所有索引强制转换为对象类型
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]

        # 对剩余的索引执行并集操作
        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result
    # 如果 kind 参数为 "array"，执行以下逻辑
    elif kind == "array":
        # 检查索引列表中的所有索引是否相同
        if not all_indexes_same(indexes):
            # 找到索引数组中的共同数据类型
            dtype = find_common_type([idx.dtype for idx in indexes])
            # 将所有索引转换为指定的数据类型
            inds = [ind.astype(dtype, copy=False) for ind in indexes]
            # 获取第一个索引的唯一值作为新的索引
            index = inds[0].unique()
            # 将其余索引合并成一个列表，并移除与新索引重复的部分
            other = inds[1].append(inds[2:])
            diff = other[index.get_indexer_for(other) == -1]
            # 如果有不重复的部分，将其添加到新索引中
            if len(diff):
                index = index.append(diff.unique())
            # 如果需要排序，对新索引进行排序
            if sort:
                index = index.sort_values()
        else:
            # 如果所有索引相同，直接使用第一个索引
            index = indexes[0]

        # 获取所有索引的一致名称，并使用第一个名称作为新的索引名称
        name = get_unanimous_names(*indexes)[0]
        # 如果新索引的名称与一致名称不同，重命名新索引
        if name != index.name:
            index = index.rename(name)
        # 返回处理后的索引对象
        return index
    
    # 如果 kind 参数为 "list"，执行以下逻辑
    elif kind == "list":
        # 找到所有索引中的数据类型列表（仅包括 Index 类型的索引）
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        # 如果存在数据类型，找到它们的共同数据类型
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None
        # 将所有索引转换为列表形式，并将所有非 Index 类型的索引解压缩为列表
        all_lists = (idx.tolist() if isinstance(idx, Index) else idx for idx in indexes)
        # 使用库函数生成一个快速且唯一的多重列表，根据 sort 参数决定是否排序
        return Index(
            lib.fast_unique_multiple_list_gen(all_lists, sort=bool(sort)),
            dtype=dtype,
        )
    
    # 如果 kind 参数既不是 "array" 也不是 "list"，抛出值错误异常
    else:
        raise ValueError(f"{kind=} must be 'special', 'array' or 'list'.")
# 确保索引类型正确，并将列表转换为 Index 对象（如果需要）

def _sanitize_and_check(indexes):
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
        如果 indexes 包含多个列表，则返回原始 indexes 和 'list'
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        如果 indexes 包含列表和 Index 对象，则递归调用以确保所有元素都是 Index 对象
          列表会被排序并转换为 Index 对象
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        如果 indexes 全部是 Index 对象，则返回原始 indexes 和类型（'special' 或 'array'）

    Parameters
    ----------
    indexes : list of Index or list objects
        索引列表，可以包含 Index 对象或列表对象

    Returns
    -------
    sanitized_indexes : list of Index or list objects
        处理后的索引列表，确保所有元素都是 Index 对象
    type : {'list', 'array', 'special'}
        索引列表的类型标识，可以是 'list', 'array' 或 'special'
    """

    kinds = {type(index) for index in indexes}

    if list in kinds:
        if len(kinds) > 1:
            # 如果 kinds 中包含多种类型，将非 Index 对象转换为 Index 对象
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds -= {list}
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


# 检查所有索引对象是否具有相同的元素

def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects
        可迭代的 Index 对象集合

    Returns
    -------
    bool
        如果所有索引对象包含相同元素则返回 True，否则返回 False
    """

    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


# 创建一个默认的 RangeIndex 对象，范围从 0 到 n-1

def default_index(n: int) -> RangeIndex:
    """
    Create a default RangeIndex object.

    Parameters
    ----------
    n : int
        索引的长度

    Returns
    -------
    RangeIndex
        新创建的 RangeIndex 对象，范围从 0 到 n-1
    """

    rng = range(n)
    return RangeIndex._simple_new(rng, name=None)
```