# `D:\src\scipysrc\pandas\pandas\core\arrays\sparse\scipy_sparse.py`

```
# 导入必要的库和模块声明
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas._libs import lib  # 导入 pandas 内部库

from pandas.core.dtypes.missing import notna  # 导入 pandas 缺失数据处理模块

from pandas.core.algorithms import factorize  # 导入 pandas 算法模块
from pandas.core.indexes.api import MultiIndex  # 导入 pandas 多级索引 API
from pandas.core.series import Series  # 导入 pandas 系列数据结构

if TYPE_CHECKING:
    from collections.abc import Iterable  # 导入可迭代类型接口

    import numpy as np  # 导入 numpy 数学运算库
    import scipy.sparse  # 导入 scipy 稀疏矩阵处理库

    from pandas._typing import (  # 导入 pandas 类型标注
        IndexLabel,  # 索引标签类型
        npt,  # numpy 类型
    )


def _check_is_partition(parts: Iterable, whole: Iterable) -> None:
    """
    检查给定的分割是否为集合的一个分区。

    Parameters
    ----------
    parts : Iterable
        要检查的部分集合列表。
    whole : Iterable
        整体集合。

    Raises
    ------
    ValueError
        如果交集不为空或并集不等于整体集合，则引发错误。
    """
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError("Is not a partition because intersection is not null.")
    if set.union(*parts) != whole:
        raise ValueError("Is not a partition because union is not the whole.")


def _levels_to_axis(
    ss,
    levels: tuple[int] | list[int],
    valid_ilocs: npt.NDArray[np.intp],
    sort_labels: bool = False,
) -> tuple[npt.NDArray[np.intp], list[IndexLabel]]:
    """
    对于 MultiIndexed 稀疏系列 `ss`，返回 `ax_coords` 和 `ax_labels`，
    其中 `ax_coords` 是目标稀疏矩阵的两个轴之一的坐标，而 `ax_labels` 是与这些坐标对应的 `ss` 的索引标签。

    Parameters
    ----------
    ss : Series
        输入的稀疏系列。
    levels : tuple/list
        需要处理的索引级别。
    valid_ilocs : numpy.ndarray
        ss 中稀疏矩阵有效值的整数位置数组。
    sort_labels : bool, default False
        是否在形成稀疏矩阵之前对轴标签进行排序。当 `levels` 只引用单个级别时，为了提高执行速度，设置为 True。

    Returns
    -------
    ax_coords : numpy.ndarray
        轴坐标。
    ax_labels : list
        轴标签列表。
    """
    # 如果需要排序标签并且只有一个级别，则可以通过以下更简单、更有效的方法获取所需输出。
    if sort_labels and len(levels) == 1:
        ax_coords = ss.index.codes[levels[0]][valid_ilocs]
        ax_labels = ss.index.levels[levels[0]]

    else:
        levels_values = lib.fast_zip(
            [ss.index.get_level_values(lvl).to_numpy() for lvl in levels]
        )
        codes, ax_labels = factorize(levels_values, sort=sort_labels)
        ax_coords = codes[valid_ilocs]

    ax_labels = ax_labels.tolist()
    return ax_coords, ax_labels


def _to_ijv(
    ss,
    row_levels: tuple[int] | list[int] = (0,),
    column_levels: tuple[int] | list[int] = (1,),
    sort_labels: bool = False,
) -> tuple[
    np.ndarray,
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    list[IndexLabel],
    list[IndexLabel],
]:
    """
    对于任意 MultiIndexed 稀疏系列，返回 (v, i, j, ilabels, jlabels)，其中 (v, (i, j)) 适合传递给 scipy.sparse.coo。

    Parameters
    ----------
    ss : Series
        输入的稀疏系列。
    row_levels : tuple/list, default (0,)
        行索引级别。
    column_levels : tuple/list, default (1,)
        列索引级别。
    sort_labels : bool, default False
        是否在形成稀疏矩阵之前对轴标签进行排序。

    Returns
    -------
    v : numpy.ndarray
        值数组。
    i : numpy.ndarray
        行坐标数组。
    j : numpy.ndarray
        列坐标数组。
    ilabels : list
        行标签列表。
    jlabels : list
        列标签列表。
    """
    ```python`
    # index and column levels must be a partition of the index
    _check_is_partition([row_levels, column_levels], range(ss.index.nlevels))
    # 从稀疏 Series 中获取整数索引和有效稀疏条目的数据。
    sp_vals = ss.array.sp_values
    # 创建一个布尔掩码，标记出有效值的位置。
    na_mask = notna(sp_vals)
    # 提取有效稀疏条目的数值。
    values = sp_vals[na_mask]
    # 获取有效稀疏条目的整数位置索引。
    valid_ilocs = ss.array.sp_index.indices[na_mask]
    
    # 使用 _levels_to_axis 函数将行标签转换为坐标和标签列表。
    i_coords, i_labels = _levels_to_axis(
        ss, row_levels, valid_ilocs, sort_labels=sort_labels
    )
    
    # 使用 _levels_to_axis 函数将列标签转换为坐标和标签列表。
    j_coords, j_labels = _levels_to_axis(
        ss, column_levels, valid_ilocs, sort_labels=sort_labels
    )
    
    # 返回值数组、行坐标、列坐标、行标签和列标签。
    return values, i_coords, j_coords, i_labels, j_labels
# 将稀疏 Series 转换为 scipy.sparse.coo_matrix，并使用指定的行和列层级作为行和列标签。
def sparse_series_to_coo(
    ss: Series,
    row_levels: Iterable[int] = (0,),
    column_levels: Iterable[int] = (1,),
    sort_labels: bool = False,
) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
    import scipy.sparse

    # 如果 Series 的索引层级小于 2，则抛出 ValueError 异常
    if ss.index.nlevels < 2:
        raise ValueError("to_coo requires MultiIndex with nlevels >= 2.")
    # 如果 Series 的索引中存在重复条目，则抛出 ValueError 异常
    if not ss.index.is_unique:
        raise ValueError(
            "Duplicate index entries are not allowed in to_coo transformation."
        )

    # 确定要使用的行和列层级在索引中的位置（层级编号）
    row_levels = [ss.index._get_level_number(x) for x in row_levels]
    column_levels = [ss.index._get_level_number(x) for x in column_levels]

    # 调用内部函数 _to_ijv，获取值 v、行索引 i、列索引 j，以及行和列标签 rows、columns
    v, i, j, rows, columns = _to_ijv(
        ss, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels
    )
    # 创建 scipy.sparse.coo_matrix 稀疏矩阵对象
    sparse_matrix = scipy.sparse.coo_matrix(
        (v, (i, j)), shape=(len(rows), len(columns))
    )
    # 返回稀疏矩阵、行标签和列标签
    return sparse_matrix, rows, columns


# 将 scipy.sparse.coo_matrix 转换为稀疏 Series。
def coo_to_sparse_series(
    A: scipy.sparse.coo_matrix, dense_index: bool = False
) -> Series:
    """
    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
    from pandas import SparseDtype

    try:
        # 使用 scipy.sparse.coo_matrix 的数据创建 Series，使用 MultiIndex 来保存行和列索引
        ser = Series(A.data, MultiIndex.from_arrays((A.row, A.col)), copy=False)
    except AttributeError as err:
        # 如果 A 不是 coo_matrix 类型，则抛出 TypeError 异常
        raise TypeError(
            f"Expected coo_matrix. Got {type(A).__name__} instead."
        ) from err
    # 对 Series 进行索引排序
    ser = ser.sort_index()
    # 将 Series 转换为稀疏类型，使用与原始 Series 相同的数据类型
    ser = ser.astype(SparseDtype(ser.dtype))
    # 如果 dense_index 为 True，则重新索引为密集型 MultiIndex
    if dense_index:
        ind = MultiIndex.from_product([A.row, A.col])
        ser = ser.reindex(ind)
    return ser
```