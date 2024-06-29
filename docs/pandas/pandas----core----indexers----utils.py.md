# `D:\src\scipysrc\pandas\pandas\core\indexers\utils.py`

```
# 导入必要的模块和类
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import AnyArrayLike

    from pandas.core.frame import DataFrame
    from pandas.core.indexes.base import Index

# -----------------------------------------------------------
# 索引器识别


def is_valid_positional_slice(slc: slice) -> bool:
    """
    检查一个切片对象是否可以解释为位置索引器。

    Parameters
    ----------
    slc : slice
        切片对象

    Returns
    -------
    bool
        如果是有效的位置切片，则返回True；否则返回False。

    Notes
    -----
    一个有效的位置切片可能也可以解释为基于标签的切片，具体取决于被切片的索引。
    """
    return (
        lib.is_int_or_none(slc.start)
        and lib.is_int_or_none(slc.stop)
        and lib.is_int_or_none(slc.step)
    )


def is_list_like_indexer(key) -> bool:
    """
    检查是否存在类似列表的索引器，但不是命名元组。

    Parameters
    ----------
    key : object
        索引器对象

    Returns
    -------
    bool
        如果是类似列表的索引器且不是命名元组，则返回True；否则返回False。
    """
    # 允许类似列表的对象，但排除作为索引器的命名元组
    return is_list_like(key) and not (isinstance(key, tuple) and type(key) is not tuple)


def is_scalar_indexer(indexer, ndim: int) -> bool:
    """
    如果所有索引器都是标量，则返回True。

    Parameters
    ----------
    indexer : object
        索引器对象
    ndim : int
        被索引对象的维数。

    Returns
    -------
    bool
        如果所有索引器都是标量，则返回True；否则返回False。
    """
    if ndim == 1 and is_integer(indexer):
        # 对于Series，允许索引器是整数
        return True
    if isinstance(indexer, tuple) and len(indexer) == ndim:
        return all(is_integer(x) for x in indexer)
    return False


def is_empty_indexer(indexer) -> bool:
    """
    检查是否存在空的索引器。

    Parameters
    ----------
    indexer : object
        索引器对象

    Returns
    -------
    bool
        如果存在空的索引器，则返回True；否则返回False。
    """
    if is_list_like(indexer) and not len(indexer):
        return True
    if not isinstance(indexer, tuple):
        indexer = (indexer,)
    return any(isinstance(idx, np.ndarray) and len(idx) == 0 for idx in indexer)


# -----------------------------------------------------------
# 索引器验证


def check_setitem_lengths(indexer, value, values) -> bool:
    """
    验证值和索引器的长度是否相同。

    对于布尔数组索引器，如果True值的数量等于"value"的长度，则允许一种特殊情况。
    在这种情况下，不会引发异常。

    Parameters
    ----------
    indexer : sequence
        设置项的键。

    value : object
        要设置的值。

    values : sequence
        要设置的值的序列。

    Returns
    -------
    bool
        如果值和索引器的长度相同或特殊情况下长度匹配，则返回True；否则返回False。
    """
    # 初始化标志变量为 False，表示不是空的列表设置操作
    no_op = False

    # 检查索引器是否为 ndarray 或 list 类型
    if isinstance(indexer, (np.ndarray, list)):
        # 如果值为列表样式，并且索引器长度与值长度不匹配，并且值是一维的
        if is_list_like(value):
            if len(indexer) != len(value) and values.ndim == 1:
                # 对于布尔索引器，其真值的长度等于值的长度也是可以接受的
                if isinstance(indexer, list):
                    indexer = np.array(indexer)
                if not (
                    isinstance(indexer, np.ndarray)
                    and indexer.dtype == np.bool_
                    and indexer.sum() == len(value)
                ):
                    # 抛出值错误，表明无法使用与值长度不同的列表样式索引器设置
                    raise ValueError(
                        "cannot set using a list-like indexer "
                        "with a different length than the value"
                    )
            # 如果索引器长度为零，则设置标志变量为 True，表示是空的列表设置操作
            if not len(indexer):
                no_op = True

    # 如果索引器是切片类型
    elif isinstance(indexer, slice):
        # 如果值是列表样式，并且值的长度与索引器的长度不匹配，并且值是一维的
        if is_list_like(value):
            if len(value) != length_of_indexer(indexer, values) and values.ndim == 1:
                # 抛出值错误，表明无法使用与值长度不同的切片索引器设置
                raise ValueError(
                    "cannot set using a slice indexer with a "
                    "different length than the value"
                )
            # 如果值的长度为零，则设置标志变量为 True，表示是空的列表设置操作
            if not len(value):
                no_op = True

    # 返回标志变量，表示是否是空的列表设置操作
    return no_op
# 执行索引器的边界检查。
# 允许使用-1表示缺失值。

def validate_indices(indices: np.ndarray, n: int) -> None:
    """
    Perform bounds-checking for an indexer.

    -1 is allowed for indicating missing values.

    Parameters
    ----------
    indices : ndarray
        Array of indices to validate.
    n : int
        Length of the array being indexed.

    Raises
    ------
    ValueError
        If any index in `indices` is less than -1.
    IndexError
        If any index in `indices` is out of bounds (>= n).

    Examples
    --------
    >>> validate_indices(np.array([1, 2]), 3)  # OK

    >>> validate_indices(np.array([1, -2]), 3)
    Traceback (most recent call last):
        ...
    ValueError: 'indices' contains values less than allowed (-2 < -1)

    >>> validate_indices(np.array([1, 2, 3]), 3)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds

    >>> validate_indices(np.array([-1, -1]), 0)  # OK

    >>> validate_indices(np.array([0, 1]), 0)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds
    """
    if len(indices):
        # 获取最小的索引值
        min_idx = indices.min()
        if min_idx < -1:
            msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
            raise ValueError(msg)

        # 获取最大的索引值
        max_idx = indices.max()
        if max_idx >= n:
            raise IndexError("indices are out-of-bounds")


# -----------------------------------------------------------
# Indexer Conversion


def maybe_convert_indices(indices, n: int, verify: bool = True) -> np.ndarray:
    """
    Attempt to convert indices into valid, positive indices.

    If we have negative indices, translate to positive here.
    If we have indices that are out-of-bounds, raise an IndexError.

    Parameters
    ----------
    indices : array-like
        Array of indices that we are to convert.
    n : int
        Number of elements in the array that we are indexing.
    verify : bool, default True
        Check that all entries are between 0 and n - 1, inclusive.

    Returns
    -------
    array-like
        An array-like of positive indices that correspond to the ones
        that were passed in initially to this function.

    Raises
    ------
    IndexError
        One of the converted indices either exceeded the number of,
        elements (specified by `n`), or was still negative.
    """
    if isinstance(indices, list):
        # 将列表类型的indices转换为numpy数组
        indices = np.array(indices)
        if len(indices) == 0:
            # 如果`indices`为空，np.array将返回一个浮点数，
            # 这会导致索引错误。
            return np.empty(0, dtype=np.intp)

    # 创建一个掩码来捕获负索引
    mask = indices < 0
    if mask.any():
        # 复制indices数组，将负索引转换为正索引
        indices = indices.copy()
        indices[mask] += n

    if verify:
        # 创建一个掩码来检查索引是否超出范围
        mask = (indices >= n) | (indices < 0)
        if mask.any():
            raise IndexError("indices are out-of-bounds")
    return indices


# -----------------------------------------------------------
# Unsorted


def length_of_indexer(indexer, target=None) -> int:
    """
    Return the expected length of target[indexer]

    Returns
    -------
    int
        Expected length of the target array after indexing.
    """
    # 如果目标对象不为 None，且索引器为切片对象时执行以下逻辑
    if target is not None and isinstance(indexer, slice):
        # 计算目标对象的长度
        target_len = len(target)
        # 获取切片的起始位置、结束位置和步长
        start = indexer.start
        stop = indexer.stop
        step = indexer.step
        
        # 如果起始位置为 None，则设置为 0
        if start is None:
            start = 0
        # 如果起始位置为负数，则转换为正数索引
        elif start < 0:
            start += target_len
        
        # 如果结束位置为 None 或超出目标对象长度，则设置为目标对象长度
        if stop is None or stop > target_len:
            stop = target_len
        # 如果结束位置为负数，则转换为正数索引
        elif stop < 0:
            stop += target_len
        
        # 如果步长为 None，则设置为 1
        if step is None:
            step = 1
        # 如果步长为负数，则调整起始和结束位置，并取绝对值
        elif step < 0:
            start, stop = stop + 1, start + 1
            step = -step
        
        # 返回经过切片计算后的长度
        return (stop - start + step - 1) // step
    
    # 如果索引器是序列类型（如 pandas 的 Series、Index，numpy 的 ndarray，或 Python 的 list）
    elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
        # 如果索引器是 Python 的 list，则转换为 numpy 的 ndarray 类型
        if isinstance(indexer, list):
            indexer = np.array(indexer)
        
        # 如果索引器的数据类型为布尔型
        if indexer.dtype == bool:
            # 返回布尔值为 True 的数量
            # GH#25774 是 GitHub 上的 issue 编号，表示处理布尔索引的特定问题
            return indexer.sum()
        
        # 返回索引器的长度
        return len(indexer)
    
    # 如果索引器是 range 对象
    elif isinstance(indexer, range):
        # 返回 range 对象的长度
        return (indexer.stop - indexer.start) // indexer.step
    
    # 如果索引器不是类列表对象的索引类型，则返回长度 1
    elif not is_list_like_indexer(indexer):
        return 1
    
    # 如果以上条件均不满足，则抛出断言错误，表示无法确定索引器的长度
    raise AssertionError("cannot find the length of the indexer")
# 禁止对一维 Series/Index 进行多维索引的辅助函数
def disallow_ndim_indexing(result) -> None:
    """
    Helper function to disallow multi-dimensional indexing on 1D Series/Index.

    GH#27125 indexer like idx[:, None] expands dim, but we cannot do that
    and keep an index, so we used to return ndarray, which was deprecated
    in GH#30588.
    """
    # 如果结果的维度大于1，则抛出 ValueError 异常
    if np.ndim(result) > 1:
        raise ValueError(
            "Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer "
            "supported. Convert to a numpy array before indexing instead."
        )


# 解压长度为1的元组/列表中包含的切片的辅助函数
def unpack_1tuple(tup):
    """
    If we have a length-1 tuple/list that contains a slice, unpack to just
    the slice.

    Notes
    -----
    The list case is deprecated.
    """
    # 如果 tup 的长度为1且包含一个切片，则返回该切片
    if len(tup) == 1 and isinstance(tup[0], slice):
        # 如果 tup 是列表，则抛出 ValueError 异常
        if isinstance(tup, list):
            # GH#31299
            raise ValueError(
                "Indexing with a single-item list containing a "
                "slice is not allowed. Pass a tuple instead.",
            )
        return tup[0]
    return tup


# 检查索引键的长度与 DataFrame 列的长度是否一致的辅助函数
def check_key_length(columns: Index, key, value: DataFrame) -> None:
    """
    Checks if a key used as indexer has the same length as the columns it is
    associated with.

    Parameters
    ----------
    columns : Index The columns of the DataFrame to index.
    key : A list-like of keys to index with.
    value : DataFrame The value to set for the keys.

    Raises
    ------
    ValueError: If the length of key is not equal to the number of columns in value
                or if the number of columns referenced by key is not equal to number
                of columns.
    """
    # 如果 columns 是唯一的，则验证 value 的列数与 key 的长度是否一致
    if columns.is_unique:
        if len(value.columns) != len(key):
            raise ValueError("Columns must be same length as key")
    else:
        # 对于非唯一的 columns，未包含在 columns 中的键值使用 -1 表示
        if len(columns.get_indexer_non_unique(key)[0]) != len(value.columns):
            raise ValueError("Columns must be same length as key")


# 解压包含省略号的元组的辅助函数
def unpack_tuple_and_ellipses(item: tuple):
    """
    Possibly unpack arr[..., n] to arr[n]
    """
    # 如果 item 的长度大于1
    if len(item) > 1:
        # 注意：我们假设此索引是在一个类似于 1D 数组的结构上执行的
        if item[0] is Ellipsis:
            # 如果 item 的第一个元素是省略号，则去除省略号
            item = item[1:]
        elif item[-1] is Ellipsis:
            # 如果 item 的最后一个元素是省略号，则去除省略号
            item = item[:-1]

    # 如果 item 的长度大于1，则抛出 IndexError 异常
    if len(item) > 1:
        raise IndexError("too many indices for array.")

    item = item[0]
    return item


# -----------------------------------------------------------
# 公共索引器验证


# 检查 `indexer` 是否是 `array` 的有效数组索引器的函数
def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
    """
    Check if `indexer` is a valid array indexer for `array`.

    For a boolean mask, `array` and `indexer` are checked to have the same
    length. The dtype is validated, and if it is an integer or boolean
    ExtensionArray, it is checked if there are missing values present, and
    """
    from pandas.core.construction import array as pd_array
    # 导入 array 函数，用于将输入转换为 pandas 数组

    # whatever is not an array-like is returned as-is (possible valid array
    # indexers that are not array-like: integer, slice, Ellipsis, None)
    # 非数组样式的索引器（如整数、切片、省略号、None）直接返回原样
    # 如果索引器是类似列表的对象（list-like），但不是元组（tuple），则直接返回索引器本身
    if is_list_like(indexer):
        if isinstance(indexer, tuple):
            return indexer
    else:
        # 如果索引器不是类似列表的对象，则将其转换为数组
        indexer = pd_array(indexer)
        # 如果转换后的数组长度为0，则将其转换为空的整数数组
        if len(indexer) == 0:
            indexer = np.array([], dtype=np.intp)

    # 获取索引器的数据类型
    dtype = indexer.dtype
    # 如果索引器的数据类型是布尔类型
    if is_bool_dtype(dtype):
        # 如果数据类型是扩展数据类型，则将索引器转换为布尔类型的NumPy数组
        if isinstance(dtype, ExtensionDtype):
            indexer = indexer.to_numpy(dtype=bool, na_value=False)
        else:
            # 否则，将索引器转换为布尔类型的NumPy数组
            indexer = np.asarray(indexer, dtype=bool)

        # 检查布尔索引器的长度是否与数组长度相同，如果不同则引发索引错误
        if len(indexer) != len(array):
            raise IndexError(
                f"Boolean index has wrong length: "
                f"{len(indexer)} instead of {len(array)}"
            )
    # 如果索引器的数据类型是整数类型
    elif is_integer_dtype(dtype):
        try:
            # 尝试将索引器转换为整数类型的NumPy数组
            indexer = np.asarray(indexer, dtype=np.intp)
        except ValueError as err:
            # 如果出现值错误，则抛出新的值错误，指示无法使用包含NA值的整数索引器进行索引
            raise ValueError(
                "Cannot index with an integer indexer containing NA values"
            ) from err
    else:
        # 如果索引器不是布尔类型或整数类型，则引发索引错误，要求索引器必须是整数或布尔类型的数组
        raise IndexError("arrays used as indices must be of integer or boolean type")

    # 返回处理后的索引器
    return indexer
```