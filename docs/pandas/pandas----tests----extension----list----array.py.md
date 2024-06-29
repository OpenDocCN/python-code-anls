# `D:\src\scipysrc\pandas\pandas\tests\extension\list\array.py`

```
"""
Test extension array for storing nested data in a pandas container.

The ListArray stores an ndarray of lists.
"""

from __future__ import annotations  # 允许在类型提示中使用当前定义的类

import numbers  # 导入处理数字的模块
import string  # 导入处理字符串的模块
from typing import TYPE_CHECKING  # 导入用于类型检查的模块

import numpy as np  # 导入NumPy库

from pandas.core.dtypes.base import ExtensionDtype  # 导入Pandas扩展类型基类

import pandas as pd  # 导入Pandas库
from pandas.api.types import (  # 从Pandas导入类型检查函数
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.arrays import ExtensionArray  # 导入Pandas扩展数组类

if TYPE_CHECKING:
    from pandas._typing import type_t  # 导入用于类型提示的类型

class ListDtype(ExtensionDtype):
    type = list  # 类型定义为列表
    name = "list"  # 类型名称为"list"
    na_value = np.nan  # 缺失值表示为NaN

    @classmethod
    def construct_array_type(cls) -> type_t[ListArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return ListArray  # 返回与此dtype相关联的数组类型


class ListArray(ExtensionArray):
    dtype = ListDtype()  # 使用ListDtype作为数组的数据类型
    __array_priority__ = 1000  # 设置数组优先级

    def __init__(self, values, dtype=None, copy=False) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array as values")  # 如果不是NumPy数组，抛出类型错误异常
        for val in values:
            if not isinstance(val, self.dtype.type) and not pd.isna(val):
                raise TypeError("All values must be of type " + str(self.dtype.type))  # 如果值不是指定类型或不是NaN，抛出类型错误异常
        self.data = values  # 初始化数组数据

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        data = np.empty(len(scalars), dtype=object)  # 创建一个空的对象数组
        data[:] = scalars  # 将标量填充到数组中
        return cls(data)  # 返回新创建的ListArray对象

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]  # 如果索引是整数，则返回对应的数据项
        else:
            # slice, list-like, mask
            return type(self)(self.data[item])  # 如果是切片、类似列表或掩码，则返回新的ListArray对象

    def __len__(self) -> int:
        return len(self.data)  # 返回数组中的数据项数量

    def isna(self):
        return np.array(
            [not isinstance(x, list) and np.isnan(x) for x in self.data], dtype=bool
        )  # 返回一个布尔数组，指示每个元素是否为NaN或不是列表

    def take(self, indexer, allow_fill=False, fill_value=None):
        # re-implement here, since NumPy has trouble setting
        # sized objects like UserDicts into scalar slots of
        # an ndarary.
        indexer = np.asarray(indexer)  # 将索引器转换为NumPy数组
        msg = (
            "Index is out of bounds or cannot do a "
            "non-empty take from an empty array."
        )  # 索引器超出范围或从空数组中无法执行非空取值时的错误消息

        if allow_fill:
            if fill_value is None:
                fill_value = self.dtype.na_value  # 如果填充值为None，则使用缺失值
            # bounds check
            if (indexer < -1).any():
                raise ValueError  # 如果索引器中有小于-1的索引，引发值错误异常
            try:
                output = [
                    self.data[loc] if loc != -1 else fill_value for loc in indexer
                ]  # 尝试根据索引器取值，如果索引为-1，则使用填充值
            except IndexError as err:
                raise IndexError(msg) from err  # 捕获索引错误异常并抛出详细的错误消息
        else:
            try:
                output = [self.data[loc] for loc in indexer]  # 尝试根据索引器取值
            except IndexError as err:
                raise IndexError(msg) from err  # 捕获索引错误异常并抛出详细的错误消息

        return self._from_sequence(output)  # 返回一个新的ListArray对象，包含取得的值的序列

    def copy(self):
        return type(self)(self.data[:])  # 返回当前ListArray对象的副本
    # 将当前 Series 对象转换为指定的数据类型，可选择是否复制数据
    def astype(self, dtype, copy=True):
        # 如果指定的 dtype 与当前 Series 对象的数据类型相同，并且不需要复制数据，则返回自身的副本或本身
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()  # 返回当前对象的副本
            return self  # 直接返回当前对象
        # 如果指定的 dtype 是字符串类型且不是对象类型，并且当前对象不是对象类型
        elif is_string_dtype(dtype) and not is_object_dtype(dtype):
            # numpy 对于嵌套元素的 astype(str) 存在问题，因此需要逐个转换成字符串数组
            return np.array([str(x) for x in self.data], dtype=dtype)
        # 如果不需要复制数据，则直接将数据转换成指定的 dtype
        elif not copy:
            return np.asarray(self.data, dtype=dtype)
        else:
            # 否则，使用指定的 dtype 和复制标志来创建一个新的 numpy 数组
            return np.array(self.data, dtype=dtype, copy=copy)

    @classmethod
    # 类方法：将同一类型的 Series 对象列表拼接成一个新的 Series 对象
    def _concat_same_type(cls, to_concat):
        # 将所有待拼接的 Series 对象的数据合并成一个大的 numpy 数组
        data = np.concatenate([x.data for x in to_concat])
        # 使用当前类 (cls) 创建一个新的 Series 对象，并将合并后的数据作为参数传入
        return cls(data)
# 创建一个函数 make_data，用于生成数据
def make_data():
    # TODO: 使用普通的字典。参见 _NDFrameIndexer._setitem_with_indexer
    # 使用随机数生成器 np.random.default_rng 创建一个特定种子的随机数生成器 rng
    rng = np.random.default_rng(2)
    # 创建一个长度为 100 的空数组 data，数据类型为 object
    data = np.empty(100, dtype=object)
    # 使用列表推导式为 data 的每个元素赋值
    data[:] = [
        # 生成一个长度在 0 到 10 之间随机整数个数的列表，列表元素是随机选择的字母
        [rng.choice(list(string.ascii_letters)) for _ in range(rng.integers(0, 10))]
        # 生成 100 个这样的列表，作为 data 的每个元素
        for _ in range(100)
    ]
    # 返回生成的数据数组 data
    return data
```