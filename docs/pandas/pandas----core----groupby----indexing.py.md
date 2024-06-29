# `D:\src\scipysrc\pandas\pandas\core\groupby\indexing.py`

```
# 导入必要的模块和类型声明
from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import numpy as np  # 导入 NumPy 库

# 导入 Pandas 的函数和装饰器
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

# 导入 Pandas 的数据类型判断函数
from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)

if TYPE_CHECKING:
    from pandas._typing import PositionalIndexer  # 引入类型提示

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.groupby import groupby  # 引入 groupby 函数


class GroupByIndexingMixin:
    """
    Mixin for adding ._positional_selector to GroupBy.
    """

    @cache_readonly
    def _make_mask_from_positional_indexer(
        self,
        arg: PositionalIndexer | tuple,
    ) -> np.ndarray:
        # 根据不同的参数类型生成掩码数组
        if is_list_like(arg):  # 如果参数是类列表的对象
            if all(is_integer(i) for i in cast(Iterable, arg)):  # 如果所有元素都是整数
                mask = self._make_mask_from_list(cast(Iterable[int], arg))  # 生成基于列表的掩码
            else:
                mask = self._make_mask_from_tuple(cast(tuple, arg))  # 生成基于元组的掩码

        elif isinstance(arg, slice):  # 如果参数是切片对象
            mask = self._make_mask_from_slice(arg)  # 生成基于切片的掩码
        elif is_integer(arg):  # 如果参数是整数
            mask = self._make_mask_from_int(cast(int, arg))  # 生成基于整数的掩码
        else:
            # 抛出类型错误，要求参数必须是整数、类列表、切片或整数和切片组成的元组
            raise TypeError(
                f"Invalid index {type(arg)}. "
                "Must be integer, list-like, slice or a tuple of "
                "integers and slices"
            )

        if isinstance(mask, bool):  # 如果掩码是布尔值
            if mask:  # 如果掩码为真
                mask = self._ascending_count >= 0  # 使用升序计数
            else:  # 如果掩码为假
                mask = self._ascending_count < 0  # 使用降序计数

        return cast(np.ndarray, mask)  # 返回转换为 NumPy 数组的掩码

    def _make_mask_from_int(self, arg: int) -> np.ndarray:
        # 根据整数参数生成掩码数组
        if arg >= 0:  # 如果参数是非负数
            return self._ascending_count == arg  # 返回升序计数等于参数的掩码数组
        else:  # 如果参数是负数
            return self._descending_count == (-arg - 1)  # 返回降序计数等于参数的掩码数组

    def _make_mask_from_list(self, args: Iterable[int]) -> bool | np.ndarray:
        # 根据整数列表生成掩码数组或布尔值
        positive = [arg for arg in args if arg >= 0]  # 获取非负整数列表
        negative = [-arg - 1 for arg in args if arg < 0]  # 获取负整数列表

        mask: bool | np.ndarray = False  # 初始化掩码变量为假

        if positive:  # 如果有非负整数
            mask |= np.isin(self._ascending_count, positive)  # 使用升序计数生成掩码

        if negative:  # 如果有负整数
            mask |= np.isin(self._descending_count, negative)  # 使用降序计数生成掩码

        return mask  # 返回掩码数组或布尔值

    def _make_mask_from_tuple(self, args: tuple) -> bool | np.ndarray:
        # 根据整数和切片元组生成掩码数组或布尔值
        mask: bool | np.ndarray = False  # 初始化掩码变量为假

        for arg in args:  # 遍历元组中的每个元素
            if is_integer(arg):  # 如果元素是整数
                mask |= self._make_mask_from_int(cast(int, arg))  # 使用整数生成掩码
            elif isinstance(arg, slice):  # 如果元素是切片对象
                mask |= self._make_mask_from_slice(arg)  # 使用切片生成掩码
            else:
                # 抛出值错误，要求元素必须是整数或切片对象
                raise ValueError(
                    f"Invalid argument {type(arg)}. Should be int or slice."
                )

        return mask  # 返回掩码数组或布尔值
    # 根据给定的切片参数创建一个掩码（布尔数组或者NumPy数组）
    def _make_mask_from_slice(self, arg: slice) -> bool | np.ndarray:
        # 获取切片的起始、结束和步长
        start = arg.start
        stop = arg.stop
        step = arg.step
    
        # 如果步长为负数，抛出异常
        if step is not None and step < 0:
            raise ValueError(f"Invalid step {step}. Must be non-negative")
    
        # 初始设定掩码为True，可以是布尔值或者NumPy数组
        mask: bool | np.ndarray = True
    
        # 如果步长为None，默认设置为1
        if step is None:
            step = 1
    
        # 处理起始索引为None的情况
        if start is None:
            # 如果步长大于1，确保self._ascending_count是step的倍数
            if step > 1:
                mask &= self._ascending_count % step == 0
    
        # 处理起始索引大于等于0的情况
        elif start >= 0:
            # 确保self._ascending_count大于等于起始索引start
            mask &= self._ascending_count >= start
    
            # 如果步长大于1，确保(self._ascending_count - start)是step的倍数
            if step > 1:
                mask &= (self._ascending_count - start) % step == 0
    
        # 处理起始索引小于0的情况
        else:
            # 确保self._descending_count小于-start
            mask &= self._descending_count < -start
    
            # 创建偏移数组，用于处理负索引的情况
            offset_array = self._descending_count + start + 1
            limit_array = (
                self._ascending_count + self._descending_count + (start + 1)
            ) < 0
            offset_array = np.where(limit_array, self._ascending_count, offset_array)
    
            # 确保offset_array是step的倍数
            mask &= offset_array % step == 0
    
        # 处理结束索引不为None的情况
        if stop is not None:
            # 如果结束索引大于等于0，确保self._ascending_count小于stop
            if stop >= 0:
                mask &= self._ascending_count < stop
            # 如果结束索引小于0，确保self._descending_count大于等于-stop
            else:
                mask &= self._descending_count >= -stop
    
        # 返回最终的掩码数组
        return mask
    
    # 缓存装饰器，用于缓存_readonly属性的计算结果
    @cache_readonly
    def _ascending_count(self) -> np.ndarray:
        # 如果类型检查为真，将self强制转换为groupby.GroupBy类型
        if TYPE_CHECKING:
            groupby_self = cast(groupby.GroupBy, self)
        else:
            groupby_self = self
    
        # 返回self的累积计数数组
        return groupby_self._cumcount_array()
    
    # 缓存装饰器，用于缓存_readonly属性的计算结果
    @cache_readonly
    def _descending_count(self) -> np.ndarray:
        # 如果类型检查为真，将self强制转换为groupby.GroupBy类型
        if TYPE_CHECKING:
            groupby_self = cast(groupby.GroupBy, self)
        else:
            groupby_self = self
    
        # 返回self的累积计数数组（降序）
        return groupby_self._cumcount_array(ascending=False)
# 使用 @doc 装饰器将 GroupByIndexingMixin._positional_selector 文档附加到当前类
@doc(GroupByIndexingMixin._positional_selector)
class GroupByPositionalSelector:
    # 初始化方法，接收一个 groupby.GroupBy 对象作为参数
    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    # 实现索引操作符 []，根据位置索引选择每个组的数据
    def __getitem__(self, arg: PositionalIndexer | tuple) -> DataFrame | Series:
        """
        Select by positional index per group.

        Implements GroupBy._positional_selector

        Parameters
        ----------
        arg : PositionalIndexer | tuple
            Allowed values are:
            - int
            - int valued iterable such as list or range
            - slice with step either None or positive
            - tuple of integers and slices

        Returns
        -------
        Series
            The filtered subset of the original groupby Series.
        DataFrame
            The filtered subset of the original groupby DataFrame.

        See Also
        --------
        DataFrame.iloc : Integer-location based indexing for selection by position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy._positional_selector : Return positional selection for each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.
        """
        # 使用 groupby_object 的 _make_mask_from_positional_indexer 方法创建索引掩码
        mask = self.groupby_object._make_mask_from_positional_indexer(arg)
        # 根据掩码选择并返回被选中的对象（DataFrame 或 Series）
        return self.groupby_object._mask_selected_obj(mask)


class GroupByNthSelector:
    """
    Dynamically substituted for GroupBy.nth to enable both call and index
    """

    # 初始化方法，接收一个 groupby.GroupBy 对象作为参数
    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    # 实现调用操作符 ()，调用 groupby_object 的 _nth 方法
    def __call__(
        self,
        n: PositionalIndexer | tuple,
        dropna: Literal["any", "all", None] = None,
    ) -> DataFrame | Series:
        return self.groupby_object._nth(n, dropna)

    # 实现索引操作符 []，调用 groupby_object 的 _nth 方法
    def __getitem__(self, n: PositionalIndexer | tuple) -> DataFrame | Series:
        return self.groupby_object._nth(n)
```