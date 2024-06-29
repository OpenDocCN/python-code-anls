# `D:\src\scipysrc\pandas\pandas\core\methods\selectn.py`

```
"""
Implementation of nlargest and nsmallest.
"""

# 导入必要的模块和类型
from __future__ import annotations

from collections.abc import (
    Hashable,  # 引入 Hashable 类型
    Sequence,  # 引入 Sequence 类型
)
from typing import (
    TYPE_CHECKING,  # 引入 TYPE_CHECKING 类型检查
    Generic,  # 引入泛型 Generic
    cast,  # 引入 cast 类型转换函数
    final,  # 引入 final 装饰器
)

import numpy as np  # 导入 NumPy 库

from pandas._libs import algos as libalgos  # 导入 pandas 库中的算法模块

from pandas.core.dtypes.common import (
    is_bool_dtype,  # 引入 pandas 的数据类型判断函数
    is_complex_dtype,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import BaseMaskedDtype  # 引入 pandas 的基本屏蔽数据类型

from pandas.core.indexes.api import default_index  # 导入 pandas 的索引 API

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,  # 引入 DtypeObj 类型
        IndexLabel,  # 引入 IndexLabel 类型
        NDFrameT,  # 引入 NDFrameT 泛型
    )

    from pandas import (
        DataFrame,  # 引入 DataFrame 类型
        Index,  # 引入 Index 类型
        Series,  # 引入 Series 类型
    )
else:
    # 在运行时，避免循环导入，用一个简单的 TypeVar 代替 Generic[...]
    from pandas._typing import T

    NDFrameT = T  # 将 T 赋给 NDFrameT
    DataFrame = T  # 将 T 赋给 DataFrame
    Series = T  # 将 T 赋给 Series


class SelectN(Generic[NDFrameT]):
    """
    Generic class for selecting n largest/smallest elements from a data structure.

    Parameters
    ----------
    obj : NDFrameT
        The input data structure (e.g., DataFrame or Series).
    n : int
        The number of largest/smallest elements to select.
    keep : str
        One of {'first', 'last', 'all'} specifying how to handle ties.

    Raises
    ------
    ValueError
        If 'keep' is not one of {'first', 'last', 'all'}.

    Methods
    -------
    nlargest() -> NDFrameT:
        Returns the n largest elements from the data structure.
    nsmallest() -> NDFrameT:
        Returns the n smallest elements from the data structure.
    """

    def __init__(self, obj: NDFrameT, n: int, keep: str) -> None:
        """
        Initialize SelectN with input parameters.

        Parameters
        ----------
        obj : NDFrameT
            The input data structure.
        n : int
            The number of elements to select.
        keep : str
            One of {'first', 'last', 'all'} specifying tie handling.
        """
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ("first", "last", "all"):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> NDFrameT:
        """
        Placeholder method for computing n largest or n smallest elements.

        Parameters
        ----------
        method : str
            Either 'nlargest' or 'nsmallest'.

        Returns
        -------
        NDFrameT
            The computed result.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    @final
    def nlargest(self) -> NDFrameT:
        """
        Return the n largest elements from the data structure.

        Returns
        -------
        NDFrameT
            The n largest elements.
        """
        return self.compute("nlargest")

    @final
    def nsmallest(self) -> NDFrameT:
        """
        Return the n smallest elements from the data structure.

        Returns
        -------
        NDFrameT
            The n smallest elements.
        """
        return self.compute("nsmallest")

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods.

        Parameters
        ----------
        dtype : DtypeObj
            The data type to check.

        Returns
        -------
        bool
            True if the dtype is valid, False otherwise.
        """
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)


class SelectNSeries(SelectN[Series]):
    """
    Implements n largest/smallest selection for Series.

    Parameters
    ----------
    obj : Series
        The input Series.
    n : int
        The number of elements to select.
    keep : {'first', 'last'}, default 'first'
        How to handle ties.

    Returns
    -------
    Series
        The n largest/smallest Series.
    """
    def compute(self, method: str) -> Series:
        # 导入 pandas 库中的 concat 函数
        from pandas.core.reshape.concat import concat
        
        # 获取对象中的属性值
        n = self.n
        # 获取对象的数据类型
        dtype = self.obj.dtype
        # 检查数据类型是否有效
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

        # 如果 n 小于等于 0，则返回一个空的 DataFrame
        if n <= 0:
            return self.obj[[]]

        # 删除所有包含 NaN 值的行，得到一个新的 DataFrame
        dropped = self.obj.dropna()
        # 提取原始 DataFrame 中包含 NaN 值的行
        nan_index = self.obj.drop(dropped.index)

        # 使用慢速方法处理情况
        if n >= len(self.obj):
            # 根据排序方法（升序或降序）对 DataFrame 进行排序，并返回前 n 个元素
            ascending = method == "nsmallest"
            return self.obj.sort_values(ascending=ascending).head(n)

        # 使用快速方法处理情况
        new_dtype = dropped.dtype

        # 类似于 algorithms._ensure_data 的操作
        arr = dropped._values
        # 如果需要将 arr 转换为 int64 类型，则进行视图转换
        if needs_i8_conversion(arr.dtype):
            arr = arr.view("i8")
        # 如果 arr 的数据类型是 BaseMaskedDtype 的实例，则获取其 _data 属性
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        # 如果 arr 的数据类型是布尔型，将其视图转换为 uint8
        if arr.dtype.kind == "b":
            arr = arr.view(np.uint8)

        # 如果方法为 "nlargest"，则对 arr 取反
        if method == "nlargest":
            arr = -arr
            # 如果新数据类型是整数型，确保在边界处进行反向排序
            if is_integer_dtype(new_dtype):
                # GH 21426: 确保在边界处进行反向排序
                arr -= 1
            # 如果新数据类型是布尔型，确保 False 比 True 小
            elif is_bool_dtype(new_dtype):
                # GH 26154: 确保 False 比 True 小
                arr = 1 - (-arr)

        # 如果 keep 属性为 "last"，则对 arr 进行反转
        if self.keep == "last":
            arr = arr[::-1]

        # 初始化 nbase 和 narr 变量
        nbase = n
        narr = len(arr)
        # 重新计算 n，取 n 和 narr 中的较小值
        n = min(n, narr)

        # 由于 kth_smallest 函数会修改其输入，因此传入 kth_smallest 的 arr 必须是连续的。这里进行复制操作
        # 当 arr 的长度大于 0 时，复制 arr 并传入 kth_smallest 函数
        if len(arr) > 0:
            kth_val = libalgos.kth_smallest(arr.copy(order="C"), n - 1)
        else:
            kth_val = np.nan
        # 找出 arr 中小于等于 kth_val 的非零元素的索引
        (ns,) = np.nonzero(arr <= kth_val)
        # 根据元素值排序 inds 数组，并返回排序后的索引
        inds = ns[arr[ns].argsort(kind="mergesort")]

        # 如果 keep 属性不是 "all"
        if self.keep != "all":
            # 只保留前 n 个索引
            inds = inds[:n]
            # findex 等于 nbase
            findex = nbase
        else:
            # 如果 inds 的长度在 nbase 和 nan_index 的长度加 inds 的长度之间
            if len(inds) < nbase <= len(nan_index) + len(inds):
                findex = len(nan_index) + len(inds)
            else:
                findex = len(inds)

        # 如果 keep 属性为 "last"，则反转 inds 数组
        if self.keep == "last":
            inds = narr - 1 - inds

        # 将 dropped 中 inds 索引位置的行与 nan_index 连接，然后取前 findex 行
        return concat([dropped.iloc[inds], nan_index]).iloc[:findex]
class SelectNFrame(SelectN[DataFrame]):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
        输入的数据框架对象，用于进行选择操作
    n : int
        要选择的行数或行数的最大数量
    keep : {'first', 'last'}, default 'first'
        决定保留相同大小值时选择的策略：保留第一次出现或最后一次出现
    columns : list or str
        指定用于排序的列名列表或单个列名字符串

    Returns
    -------
    nordered : DataFrame
        返回一个数据框架，包含按指定条件选择的行
    """

    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None:
        super().__init__(obj, n, keep)  # 调用父类的初始化方法
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]

        columns = cast(Sequence[Hashable], columns)  # 确保列名是可散列的序列
        columns = list(columns)  # 转换成列表
        self.columns = columns  # 将处理后的列名赋值给实例变量self.columns
    def compute(self, method: str) -> DataFrame:
        # 获取对象的属性值
        n = self.n
        # 获取对象的数据帧
        frame = self.obj
        # 获取对象的列名列表
        columns = self.columns

        # 遍历每一列
        for column in columns:
            # 获取列的数据类型
            dtype = frame[column].dtype
            # 检查数据类型是否支持指定的方法，否则抛出类型错误
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(
                    f"Column {column!r} has dtype {dtype}, "
                    f"cannot use method {method!r} with this dtype"
                )

        def get_indexer(current_indexer: Index, other_indexer: Index) -> Index:
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            # 根据 method 连接当前索引和其他索引
            if method == "nsmallest":
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # 保存原始索引并重置索引，以防索引包含重复值
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer: Index = default_index(0)

        # 遍历每一列的索引和名称
        for i, column in enumerate(columns):
            # 对当前列的数据应用指定的方法
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            # 获取指定数量的结果值，并根据情况保留全部或者保留部分
            values = getattr(series, method)(
                cur_n, keep=self.keep if is_last_column else "all"
            )

            # 如果是最后一列或者结果值数量小于等于当前要求的数量
            if is_last_column or len(values) <= cur_n:
                # 更新索引器
                indexer = get_indexer(indexer, values.index)
                break

            # 寻找与系列中最小/最大值相等的所有值
            border_value = values == values[values.index[-1]]

            # 一些值是前n个中的一部分，一些不是
            unsafe_values = values[border_value]

            # 这些值肯定是前n个中的一部分
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # 继续处理剩余列中的不安全值
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        # 根据索引器获取数据帧的子集
        frame = frame.take(indexer)

        # 恢复数据帧的原始索引
        frame.index = original_index.take(indexer)

        # 如果只有一列，则数据帧已经排序
        if len(columns) == 1:
            return frame

        # 根据指定的列和升序/降序进行排序，并使用稳定排序算法
        ascending = method == "nsmallest"
        return frame.sort_values(columns, ascending=ascending, kind="mergesort")
```