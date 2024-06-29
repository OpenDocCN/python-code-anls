# `D:\src\scipysrc\pandas\pandas\core\indexers\objects.py`

```
"""
Indexer objects for computing start/end window bounds for rolling operations
"""

# 从未来模块导入 annotations 特性，使得类方法能够引用自身类的类型
from __future__ import annotations

# 导入 timedelta 类用于处理时间差
from datetime import timedelta

# 导入 numpy 库，用于数值计算
import numpy as np

# 导入 pandas 库内部的时间序列偏移量类 BaseOffset
from pandas._libs.tslibs import BaseOffset

# 导入 pandas 库内部窗口索引计算函数
from pandas._libs.window.indexers import calculate_variable_window_bounds

# 导入 pandas 库内部的 Appender 装饰器
from pandas.util._decorators import Appender

# 导入 pandas 库内部的平台整数类型检查函数
from pandas.core.dtypes.common import ensure_platform_int

# 导入 pandas 库内部的日期时间索引类 DatetimeIndex
from pandas.core.indexes.datetimes import DatetimeIndex

# 导入 pandas 库内部的纳秒偏移类 Nano
from pandas.tseries.offsets import Nano

# 定义文档字符串，描述了 get_window_bounds 函数的功能和参数
get_window_bounds_doc = """
Computes the bounds of a window.

Parameters
----------
num_values : int, default 0
    number of values that will be aggregated over
window_size : int, default 0
    the number of rows in a window
min_periods : int, default None
    min_periods passed from the top level rolling API
center : bool, default None
    center passed from the top level rolling API
closed : str, default None
    closed passed from the top level rolling API
step : int, default None
    step passed from the top level rolling API
    .. versionadded:: 1.5
win_type : str, default None
    win_type passed from the top level rolling API

Returns
-------
A tuple of ndarray[int64]s, indicating the boundaries of each
window
"""

# 定义一个基础索引器类 BaseIndexer，用于计算滚动操作的窗口起止边界
class BaseIndexer:
    """
    Base class for window bounds calculations.

    Parameters
    ----------
    index_array : np.ndarray, default None
        Array-like structure representing the indices for the data points.
        If None, the default indices are assumed. This can be useful for
        handling non-uniform indices in data, such as in time series
        with irregular timestamps.
    window_size : int, default 0
        Size of the moving window. This is the number of observations used
        for calculating the statistic. The default is to consider all
        observations within the window.
    **kwargs
        Additional keyword arguments passed to the subclass's methods.

    See Also
    --------
    DataFrame.rolling : Provides rolling window calculations on dataframe.
    Series.rolling : Provides rolling window calculations on series.

    Examples
    --------
    >>> from pandas.api.indexers import BaseIndexer
    >>> class CustomIndexer(BaseIndexer):
    ...     def get_window_bounds(self, num_values, min_periods, center, closed, step):
    ...         start = np.arange(num_values, dtype=np.int64)
    ...         end = np.arange(num_values, dtype=np.int64) + self.window_size
    ...         return start, end
    >>> df = pd.DataFrame({"values": range(5)})
    >>> indexer = CustomIndexer(window_size=2)
    >>> df.rolling(indexer).sum()
        values
    0    1.0
    1    3.0
    2    5.0
    3    7.0
    4    4.0
    """

    # 构造函数初始化方法，接受索引数组和窗口大小等参数
    def __init__(
        self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs

        Array-like structure representing the indices for the data points.
        If None, the default indices are assumed. This can be useful for
        handling non-uniform indices in data, such as in time series
        with irregular timestamps.
    window_size : int, default 0
        Size of the moving window. This is the number of observations used
        for calculating the statistic. The default is to consider all
        observations within the window.
    **kwargs
        Additional keyword arguments passed to the subclass's methods.
        
    See Also
    --------
    DataFrame.rolling : Provides rolling window calculations on dataframe.
    Series.rolling : Provides rolling window calculations on series.
    """
    # 初始化方法，设置索引数组和窗口大小作为对象属性
    ) -> None:
        self.index_array = index_array
        self.window_size = window_size
        # 将用户定义的关键字参数作为对象属性，以便在 get_window_bounds 方法中使用

    @Appender(get_window_bounds_doc)
    # 装饰器，将 get_window_bounds 方法与 get_window_bounds_doc 连接
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 抛出未实现错误，子类必须实现这个方法
        raise NotImplementedError
class FixedWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of fixed length."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 根据窗口大小和是否居中计算偏移量
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0

        # 计算窗口的结束位置数组
        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
        # 计算窗口的开始位置数组
        start = end - self.window_size
        # 根据窗口关闭方式调整开始位置
        if closed in ["left", "both"]:
            start -= 1
        # 根据窗口关闭方式调整结束位置
        if closed in ["left", "neither"]:
            end -= 1

        # 确保开始位置和结束位置在合理范围内
        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        # 返回开始位置和结束位置的元组
        return start, end


class VariableWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of variable length, namely for time series."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 调用函数计算基于变量窗口长度的窗口边界
        return calculate_variable_window_bounds(
            num_values,
            self.window_size,
            min_periods,
            center,  # type: ignore[arg-type]
            closed,
            self.index_array,  # type: ignore[arg-type]
        )


class VariableOffsetWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.

    Examples
    --------
    >>> from pandas.api.indexers import VariableOffsetWindowIndexer
    >>> df = pd.DataFrame(range(10), index=pd.date_range("2020", periods=10))
    >>> offset = pd.offsets.BDay(1)
    >>> indexer = VariableOffsetWindowIndexer(index=df.index, offset=offset)
    >>> df
                0
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3
    2020-01-05  4
    2020-01-06  5
    2020-01-07  6
    2020-01-08  7
    2020-01-09  8
    2020-01-10  9
    >>> df.rolling(indexer).sum()
                   0
    2020-01-01   0.0
    2020-01-02   1.0
    2020-01-03   2.0
    2020-01-04   3.0
    2020-01-05   7.0
    2020-01-06  12.0
    2020-01-07   6.0
    2020-01-08   7.0
    2020-01-09   8.0
    2020-01-10   9.0
    """

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs,
    ):
        # 初始化变量偏移窗口索引器，基于给定的索引数组和偏移量
        super().__init__(index_array=index_array, window_size=window_size, **kwargs)
        self.index = index
        self.offset = offset
        ) -> None:
        # 调用父类的初始化方法，设置索引数组和窗口大小等参数
        super().__init__(index_array, window_size, **kwargs)
        # 检查索引是否为 DatetimeIndex 类型，如果不是则抛出数值错误
        if not isinstance(index, DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex.")
        # 设置对象的索引属性
        self.index = index
        # 检查偏移是否为 BaseOffset 类型，如果不是则抛出数值错误
        if not isinstance(offset, BaseOffset):
            raise ValueError("offset must be a DateOffset-like object.")
        # 设置对象的偏移属性
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 如果指定了 step 参数，抛出未实现错误，因为不支持可变偏移窗口的步长
        if step is not None:
            raise NotImplementedError("step not implemented for variable offset window")
        # 如果 num_values 小于等于 0，则返回空的 numpy 数组作为起始和结束点
        if num_values <= 0:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

        # 如果未指定 closed 参数，则根据索引是否存在选择默认值 'right' 或 'both'
        if closed is None:
            closed = "right" if self.index is not None else "both"

        # 根据 closed 参数确定右端点是否闭合
        right_closed = closed in ["right", "both"]
        # 根据 closed 参数确定左端点是否闭合
        left_closed = closed in ["left", "both"]

        # 确定索引的增长方向（正向或负向）
        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        # 计算偏移量的差异，考虑索引的增长方向
        offset_diff = index_growth_sign * self.offset

        # 创建空的起始和结束点数组
        start = np.empty(num_values, dtype="int64")
        start.fill(-1)
        end = np.empty(num_values, dtype="int64")
        end.fill(-1)

        # 第一个窗口的起始点为索引 0
        start[0] = 0

        # 如果右端点是闭合的，则设置第一个窗口的结束点为 1，否则为 0
        if right_closed:
            end[0] = 1
        else:
            end[0] = 0

        # 定义时间间隔零
        zero = timedelta(0)
        # 遍历计算每个窗口的起始和结束点
        # start 是包含的开始切片区间
        # end 是不包含的结束切片区间
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff

            # 如果左端点是闭合的，则调整开始边界
            if left_closed:
                start_bound -= Nano(1)

            # 将开始点推进直到满足约束条件
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break

            # 结束边界为前一个结束或当前索引
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and not right_closed:
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # 如果右端点是开放的，则减去 1
            if not right_closed:
                end[i] -= 1

        # 返回起始点和结束点数组
        return start, end
class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds, mimicking df.expanding()"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 返回一个扩展窗口的起始和结束索引数组，起始索引数组全为0，结束索引数组从1到num_values+1
        return (
            np.zeros(num_values, dtype=np.int64),
            np.arange(1, num_values + 1, dtype=np.int64),
        )


class FixedForwardWindowIndexer(BaseIndexer):
    """
    Creates window boundaries for fixed-length windows that include the current row.

    Parameters
    ----------
    index_array : np.ndarray, default None
        Array-like structure representing the indices for the data points.
        If None, the default indices are assumed. This can be useful for
        handling non-uniform indices in data, such as in time series
        with irregular timestamps.
    window_size : int, default 0
        Size of the moving window. This is the number of observations used
        for calculating the statistic. The default is to consider all
        observations within the window.
    **kwargs
        Additional keyword arguments passed to the subclass's methods.

    See Also
    --------
    DataFrame.rolling : Provides rolling window calculations.
    api.indexers.VariableWindowIndexer : Calculate window bounds based on
        variable-sized windows.

    Examples
    --------
    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0
    """

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 检查是否设置了center=True，对于前向窗口不支持center=True
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        # 检查是否设置了closed参数，前向窗口不支持设置closed参数
        if closed is not None:
            raise ValueError(
                "Forward-looking windows don't support setting the closed argument"
            )
        # 如果步长未指定，默认设置为1
        if step is None:
            step = 1

        # 创建起始索引数组，从0开始，步长为step，数据类型为int64
        start = np.arange(0, num_values, step, dtype="int64")
        # 计算结束索引数组，为起始索引数组加上窗口大小self.window_size
        end = start + self.window_size
        # 如果设置了窗口大小，则对结束索引进行裁剪，确保不超过num_values
        if self.window_size:
            end = np.clip(end, 0, num_values)

        return start, end


class GroupbyIndexer(BaseIndexer):
    """Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()"""
    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int | BaseIndexer = 0,
        groupby_indices: dict | None = None,
        window_indexer: type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        index_array : np.ndarray or None
            np.ndarray of the index of the original object that we are performing
            a chained groupby operation over. This index has been pre-sorted relative to
            the groups
        window_size : int or BaseIndexer
            window size during the windowing operation
        groupby_indices : dict or None
            dict of {group label: [positional index of rows belonging to the group]}
        window_indexer : BaseIndexer
            BaseIndexer class determining the start and end bounds of each group
        indexer_kwargs : dict or None
            Custom kwargs to be passed to window_indexer
        **kwargs :
            keyword arguments that will be available when get_window_bounds is called
        """
        # 初始化方法，设置对象的初始状态和参数
        self.groupby_indices = groupby_indices or {}
        # 设置窗口索引器类
        self.window_indexer = window_indexer
        # 复制索引器参数字典或初始化为空字典
        self.indexer_kwargs = indexer_kwargs.copy() if indexer_kwargs else {}
        # 调用父类的初始化方法，传入指定参数
        super().__init__(
            index_array=index_array,
            window_size=self.indexer_kwargs.pop("window_size", window_size),
            **kwargs,
        )

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 1) 对于每个分组，获取属于该分组的索引
        # 2) 使用这些索引计算窗口的起始和结束边界
        # 3) 按照分组顺序将窗口边界添加到列表中
        start_arrays = []  # 初始化存储窗口起始索引的列表
        end_arrays = []    # 初始化存储窗口结束索引的列表
        window_indices_start = 0  # 窗口索引的起始值
        for indices in self.groupby_indices.values():  # 遍历每个分组的索引集合
            index_array: np.ndarray | None

            if self.index_array is not None:
                index_array = self.index_array.take(ensure_platform_int(indices))  # 获取索引数组中属于当前分组的部分
            else:
                index_array = self.index_array  # 如果索引数组为None，则设为None

            indexer = self.window_indexer(  # 创建窗口索引器对象
                index_array=index_array,
                window_size=self.window_size,
                **self.indexer_kwargs,
            )
            start, end = indexer.get_window_bounds(  # 调用索引器的方法获取窗口边界
                len(indices), min_periods, center, closed, step
            )
            start = start.astype(np.int64)  # 将起始边界转换为np.int64类型
            end = end.astype(np.int64)      # 将结束边界转换为np.int64类型
            assert len(start) == len(
                end
            ), "these should be equal in length from get_window_bounds"  # 检查起始和结束边界数组的长度是否相等

            # 不能使用 groupby_indices，因为它们可能与我们正在遍历的对象不是单调递增的
            window_indices = np.arange(  # 创建窗口索引数组，用于表示当前窗口的索引范围
                window_indices_start, window_indices_start + len(indices)
            )
            window_indices_start += len(indices)  # 更新窗口索引的起始位置

            # 扩展窗口索引数组，以便正确切片窗口范围 [start, end)
            window_indices = np.append(window_indices, [window_indices[-1] + 1]).astype(
                np.int64, copy=False
            )

            start_arrays.append(window_indices.take(ensure_platform_int(start)))  # 将当前分组的起始索引添加到列表中
            end_arrays.append(window_indices.take(ensure_platform_int(end)))      # 将当前分组的结束索引添加到列表中

        if len(start_arrays) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)  # 如果没有窗口范围，则返回空数组

        start = np.concatenate(start_arrays)  # 合并所有分组的起始索引
        end = np.concatenate(end_arrays)      # 合并所有分组的结束索引
        return start, end  # 返回所有分组的窗口起始和结束索引数组
# 定义一个名为ExponentialMovingWindowIndexer的类，继承自BaseIndexer，用于计算指数移动窗口的索引

@Appender(get_window_bounds_doc)
# 使用装饰器@Appender，将get_window_bounds方法附加上get_window_bounds_doc文档字符串

def get_window_bounds(
    self,
    num_values: int = 0,
    min_periods: int | None = None,
    center: bool | None = None,
    closed: str | None = None,
    step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # 定义get_window_bounds方法，用于计算窗口的边界
    # 参数说明：
    # - num_values: 窗口的数值数量，默认为0
    # - min_periods: 最小的周期数，可以为None
    # - center: 是否居中窗口，可以为None
    # - closed: 窗口闭合方式，可以为None
    # - step: 步长，可以为None
    return np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64)
    # 返回一个包含窗口下界和上界的元组，下界为0，上界为num_values，数据类型为np.int64的numpy数组
```