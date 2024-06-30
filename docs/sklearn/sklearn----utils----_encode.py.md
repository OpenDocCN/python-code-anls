# `D:\src\scipysrc\scikit-learn\sklearn\utils\_encode.py`

```
from collections import Counter
from contextlib import suppress
from typing import NamedTuple

import numpy as np

# 导入自定义模块中的特定函数和变量
from ._array_api import (
    _isin,
    _searchsorted,
    _setdiff1d,
    device,
    get_namespace,
)
# 导入检查是否为标量的函数
from ._missing import is_scalar_nan


def _unique(values, *, return_inverse=False, return_counts=False):
    """Helper function to find unique values with support for python objects.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : ndarray
        Values to check for unknowns.

    return_inverse : bool, default=False
        If True, also return the indices of the unique values.

    return_counts : bool, default=False
        If True, also return the number of times each unique item appears in
        values.

    Returns
    -------
    unique : ndarray
        The sorted unique values.

    unique_inverse : ndarray
        The indices to reconstruct the original array from the unique array.
        Only provided if `return_inverse` is True.

    unique_counts : ndarray
        The number of times each of the unique values comes up in the original
        array. Only provided if `return_counts` is True.
    """
    # 如果数据类型是 object，使用纯 Python 方法查找唯一值
    if values.dtype == object:
        return _unique_python(
            values, return_inverse=return_inverse, return_counts=return_counts
        )
    # 对于数值类型，使用 numpy 方法查找唯一值
    return _unique_np(
        values, return_inverse=return_inverse, return_counts=return_counts
    )


def _unique_np(values, return_inverse=False, return_counts=False):
    """Helper function to find unique values for numpy arrays that correctly
    accounts for nans. See `_unique` documentation for details."""
    # 获取数值操作的命名空间
    xp, _ = get_namespace(values)

    inverse, counts = None, None

    # 根据参数设置，调用不同的 numpy 方法查找唯一值及其相关信息
    if return_inverse and return_counts:
        uniques, _, inverse, counts = xp.unique_all(values)
    elif return_inverse:
        uniques, inverse = xp.unique_inverse(values)
    elif return_counts:
        uniques, counts = xp.unique_counts(values)
    else:
        uniques = xp.unique_values(values)

    # np.unique 会在 `uniques` 的末尾包含重复的缺失值（NaN）
    # 这里修剪掉 NaN 并从 `uniques` 中移除它
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = _searchsorted(xp, uniques, xp.nan)
        uniques = uniques[: nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx

        if return_counts:
            counts[nan_idx] = xp.sum(counts[nan_idx:])
            counts = counts[: nan_idx + 1]

    ret = (uniques,)

    # 根据参数返回相应的结果
    if return_inverse:
        ret += (inverse,)

    if return_counts:
        ret += (counts,)

    return ret[0] if len(ret) == 1 else ret


class MissingValues(NamedTuple):
    """Data class for missing data information"""

    nan: bool
    none: bool
    def to_list(self):
        """Convert tuple to a list where None is always first."""
        # 创建一个空列表，用于存放转换后的结果
        output = []
        # 如果 self.none 为真（即不为 None），将 None 添加到输出列表中
        if self.none:
            output.append(None)
        # 如果 self.nan 为真（即不为 NaN），将 NaN 添加到输出列表中
        if self.nan:
            output.append(np.nan)
        # 返回生成的列表
        return output
def _extract_missing(values):
    """Extract missing values from `values`.

    Parameters
    ----------
    values: set
        Set of values to extract missing from.

    Returns
    -------
    output: set
        Set with missing values extracted.

    missing_values: MissingValues
        Object with missing value information.
    """
    # 创建一个集合，其中包含所有为 None 或是标量 NaN 的值
    missing_values_set = {
        value for value in values if value is None or is_scalar_nan(value)
    }

    # 如果集合为空，表示没有缺失值
    if not missing_values_set:
        return values, MissingValues(nan=False, none=False)

    # 如果集合中包含 None
    if None in missing_values_set:
        # 如果集合中只有一个缺失值，则标记为 none=True, nan=False
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            # 如果集合中有多个缺失值，则标记为 none=True, nan=True
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        # 如果集合中没有 None，但有标量 NaN，则标记为 none=False, nan=True
        output_missing_values = MissingValues(nan=True, none=False)

    # 从原始集合中移除缺失值，得到完整的值集合
    output = values - missing_values_set
    return output, output_missing_values


class _nandict(dict):
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        super().__init__(mapping)
        # 遍历字典的键值对，找到第一个键为标量 NaN 的项，并将对应的值存储在 nan_value 中
        for key, value in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        # 如果在字典中找不到对应的键，且字典中存在 nan_value，则返回 nan_value
        if hasattr(self, "nan_value") and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)


def _map_to_integer(values, uniques):
    """Map values based on its position in uniques."""
    # 获取 values 在 uniques 中的命名空间
    xp, _ = get_namespace(values, uniques)
    # 创建一个 _nandict 对象，其中包含每个值在 uniques 中的索引
    table = _nandict({val: i for i, val in enumerate(uniques)})
    # 将 values 中的每个值映射为其在 uniques 中的索引，并返回
    return xp.asarray([table[v] for v in values], device=device(values))


def _unique_python(values, *, return_inverse, return_counts):
    # 仅在 `_uniques` 中使用，详细信息请参阅那里的文档字符串
    try:
        # 创建 values 的集合表示
        uniques_set = set(values)
        # 提取集合中的缺失值，并返回缺失值集合及其信息
        uniques_set, missing_values = _extract_missing(uniques_set)

        # 对集合中的值进行排序，并将缺失值添加到末尾
        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        # 将结果转换为 numpy 数组，以与 values 相同的数据类型存储
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        # 如果 values 不是统一的字符串或数字类型，则抛出类型错误
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError(
            "Encoders require their input argument must be uniformly "
            f"strings or numbers. Got {types}"
        )
    ret = (uniques,)

    # 如果需要返回 values 在 uniques 中的逆映射，则将其加入返回值中
    if return_inverse:
        ret += (_map_to_integer(values, uniques),)

    # 如果需要返回 values 中每个唯一值的计数，则将其加入返回值中
    if return_counts:
        ret += (_get_counts(values, uniques),)

    return ret[0] if len(ret) == 1 else ret


def _encode(values, *, uniques, check_unknown=True):
    """Helper function to encode values into [0, n_uniques - 1].

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    ```
    xp, _ = get_namespace(values, uniques)
    # 调用 get_namespace 函数，获取 values 和 uniques 的命名空间
    if not xp.isdtype(values.dtype, "numeric"):
        # 如果 xp 不符合 values 的数值类型
        try:
            # 尝试将 values 和 uniques 映射为整数编码
            return _map_to_integer(values, uniques)
        except KeyError as e:
            # 如果出现 KeyError，则抛出 ValueError，提示包含未见过的标签
            raise ValueError(f"y contains previously unseen labels: {str(e)}")
    else:
        # 如果 xp 符合 values 的数值类型
        if check_unknown:
            # 如果 check_unknown 为 True
            # 检查 values 中是否存在 uniques 中没有的标签
            diff = _check_unknown(values, uniques)
            if diff:
                # 如果存在未知标签，则抛出 ValueError，提示包含未见过的标签
                raise ValueError(f"y contains previously unseen labels: {str(diff)}")
        # 使用 _searchsorted 函数在 xp 中搜索 uniques，返回对应的编码值
        return _searchsorted(xp, uniques, values)
# 定义一个辅助函数来检查编码值中的未知值
def _check_unknown(values, known_values, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : array
        Values to check for unknowns.
    known_values : array
        Known values. Must be unique.
    return_mask : bool, default=False
        If True, return a mask of the same shape as `values` indicating
        the valid values.

    Returns
    -------
    diff : list
        The unique values present in `values` and not in `known_values`.
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.

    """
    # 获取适当的命名空间以处理不同的数据类型
    xp, _ = get_namespace(values, known_values)
    valid_mask = None

    # 检查是否为数值类型
    if not xp.isdtype(values.dtype, "numeric"):
        # 将值转换为集合并提取缺失的值
        values_set = set(values)
        values_set, missing_in_values = _extract_missing(values_set)

        # 将已知值转换为集合并提取缺失的值
        uniques_set = set(known_values)
        uniques_set, missing_in_uniques = _extract_missing(uniques_set)

        # 计算差异值集合
        diff = values_set - uniques_set

        # 检查差异集合中的缺失值
        nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
        none_in_diff = missing_in_values.none and not missing_in_uniques.none

        # 定义一个函数来验证值的有效性
        def is_valid(value):
            return (
                value in uniques_set
                or (missing_in_uniques.none and value is None)
                or (missing_in_uniques.nan and is_scalar_nan(value))
            )

        # 如果需要返回掩码，则创建一个布尔数组
        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = xp.array([is_valid(value) for value in values])
            else:
                valid_mask = xp.ones(len(values), dtype=xp.bool)

        # 将差异值转换为列表
        diff = list(diff)
        
        # 如果缺失空值，则添加 None 到差异值列表
        if none_in_diff:
            diff.append(None)
        
        # 如果缺失 NaN 值，则添加 NaN 到差异值列表
        if nan_in_diff:
            diff.append(np.nan)
    else:
        # 对于数值类型，获取唯一值并计算差异
        unique_values = xp.unique_values(values)
        diff = _setdiff1d(unique_values, known_values, xp, assume_unique=True)
        
        # 如果需要返回掩码，则创建一个布尔数组
        if return_mask:
            if diff.size:
                valid_mask = _isin(values, known_values, xp)
            else:
                valid_mask = xp.ones(len(values), dtype=xp.bool)

        # 检查已知值中是否存在 NaN 值
        if xp.any(xp.isnan(known_values)):
            diff_is_nan = xp.isnan(diff)
            if xp.any(diff_is_nan):
                # 如果需要返回掩码并且 diff 非空，则将 NaN 从有效掩码中移除
                if diff.size and return_mask:
                    is_nan = xp.isnan(values)
                    valid_mask[is_nan] = 1

                # 从 diff 中移除 NaN 值
                diff = diff[~diff_is_nan]
        
        # 将差异值转换为列表
        diff = list(diff)

    # 如果需要返回掩码，则返回差异值和有效掩码
    if return_mask:
        return diff, valid_mask
    
    # 否则，只返回差异值
    return diff
    # 定义一个生成器方法，用于生成不包含 NaN 值的项目，并统计 NaN 的数量
    def _generate_items(self, items):
        """Generate items without nans. Stores the nan counts separately."""
        # 遍历输入的items列表
        for item in items:
            # 如果当前项目不是 NaN 值
            if not is_scalar_nan(item):
                # 生成当前项目
                yield item
                # 继续下一个项目
                continue
            # 如果对象没有属性"nan_count"，则初始化为0
            if not hasattr(self, "nan_count"):
                self.nan_count = 0
            # 增加 NaN 计数
            self.nan_count += 1

    # 定义一个特殊方法 __missing__，处理缺失的键
    def __missing__(self, key):
        # 如果对象有"nan_count"属性，并且key是 NaN 值
        if hasattr(self, "nan_count") and is_scalar_nan(key):
            # 返回 NaN 的计数
            return self.nan_count
        # 抛出键错误异常，表示未找到该键
        raise KeyError(key)
def _get_counts(values, uniques):
    """Get the count of each of the `uniques` in `values`.

    The counts will use the order passed in by `uniques`. For non-object dtypes,
    `uniques` is assumed to be sorted and `np.nan` is at the end.
    """
    # 如果 `values` 的数据类型为对象或Unicode字符串
    if values.dtype.kind in "OU":
        # 使用 `_NaNCounter` 类来统计每个 `uniques` 中元素的出现次数
        counter = _NaNCounter(values)
        # 初始化一个全零数组，用于存储每个 `uniques` 中元素的计数结果
        output = np.zeros(len(uniques), dtype=np.int64)
        # 遍历 `uniques` 中的每个元素，尝试从 `counter` 中获取计数值
        for i, item in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        # 返回计数结果数组
        return output

    # 如果 `values` 的数据类型不是对象或Unicode字符串
    # 利用 `_unique_np` 函数获取 `values` 中每个元素的唯一值和对应的计数
    unique_values, counts = _unique_np(values, return_counts=True)

    # 在 `unique_values` 中记录那些存在于 `uniques` 中的唯一值的索引
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    # 如果 `unique_values` 和 `uniques` 的最后一个元素都是 NaN，则标记为匹配
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True

    # 找出 `unique_values` 中存在于 `uniques` 中的元素的索引位置
    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    # 初始化一个与 `uniques` 相同形状的全零数组，用于存储计数结果
    output = np.zeros_like(uniques, dtype=np.int64)
    # 将 `counts` 中对应索引位置的计数值复制到 `output` 中
    output[uniques_in_values] = counts[unique_valid_indices]
    # 返回计数结果数组
    return output
```