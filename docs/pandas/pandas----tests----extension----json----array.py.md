# `D:\src\scipysrc\pandas\pandas\tests\extension\json\array.py`

```
"""
Test extension array for storing nested data in a pandas container.

The JSONArray stores lists of dictionaries. The storage mechanism is a list,
not an ndarray.

Note
----
We currently store lists of UserDicts. Pandas has a few places
internally that specifically check for dicts, and does non-scalar things
in that case. We *want* the dictionaries to be treated as scalars, so we
hack around pandas by using UserDicts.
"""

from __future__ import annotations

from collections import (
    UserDict,  # 导入 UserDict 类，用于创建用户自定义字典
    abc,       # 导入 abc 模块，支持抽象基类
)
import itertools   # 导入 itertools 模块，提供高效循环操作的函数
import numbers     # 导入 numbers 模块，用于数值类型相关操作
import string      # 导入 string 模块，包含字符串处理相关函数
import sys         # 导入 sys 模块，提供系统相关的功能和变量
from typing import (
    TYPE_CHECKING,  # 导入 TYPE_CHECKING 常量，用于类型检查时的条件判断
    Any,            # 导入 Any 类型，表示任意类型
)

import numpy as np   # 导入 NumPy 库，用于数值计算和数组操作

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,   # 导入 pandas 中的数据类型判断函数，检查是否为布尔类型
    is_list_like,    # 导入 pandas 中的数据类型判断函数，检查是否为类列表类型
    pandas_dtype,    # 导入 pandas 中的数据类型转换函数
)

import pandas as pd   # 导入 pandas 库，用于数据分析和处理
from pandas.api.extensions import (
    ExtensionArray,  # 导入 pandas 扩展数组类
    ExtensionDtype,  # 导入 pandas 扩展数据类型类
)
from pandas.core.indexers import unpack_tuple_and_ellipses

if TYPE_CHECKING:
    from collections.abc import Mapping   # 导入 Mapping 抽象基类，用于映射类型的判断

    from pandas._typing import type_t   # 导入 pandas 内部类型定义

class JSONDtype(ExtensionDtype):
    type = abc.Mapping   # 设置数据类型为映射类型
    name = "json"
    na_value: Mapping[str, Any] = UserDict()   # 设置缺失值为用户字典类型

    @classmethod
    def construct_array_type(cls) -> type_t[JSONArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return JSONArray   # 返回与该数据类型相关联的数组类型


class JSONArray(ExtensionArray):
    dtype = JSONDtype()   # 设置数组的数据类型为 JSONDtype 类型
    __array_priority__ = 1000   # 设置数组的优先级为 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        for val in values:
            if not isinstance(val, self.dtype.type):
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values   # 初始化数组的数据为传入的值列表

        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self._data = self.data   # 设置属性别名以确保 pandas 支持这些属性
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars)   # 从标量序列创建 JSONArray 对象

    @classmethod
    def _from_factorized(cls, values, original):
        return cls([UserDict(x) for x in values if x != ()])   # 从因子化的数据创建 JSONArray 对象
    # 实现索引操作符 [] 的特殊方法，用于获取对象中指定项的值
    def __getitem__(self, item):
        # 如果 item 是元组类型，则解包和展开元素
        if isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        # 如果 item 是整数类型，返回对象中索引为 item 的数据
        if isinstance(item, numbers.Integral):
            return self.data[item]
        # 如果 item 是切片类型且是 slice(None)，确保返回一个视图
        elif isinstance(item, slice) and item == slice(None):
            # 返回一个新的相同类型的对象，拷贝 self.data 数据
            return type(self)(self.data)
        # 如果 item 是切片类型，返回一个新的相同类型的对象，其中包含 self.data[item] 的数据
        elif isinstance(item, slice):
            # 返回一个新的相同类型的对象，其数据是 self.data[item] 的切片
            return type(self)(self.data[item])
        # 如果 item 不是类列表的类型，抛出 IndexError 异常
        elif not is_list_like(item):
            # 例如："foo" 或 2.5
            # 引用自 numpy 的异常消息，说明有效的索引只能是整数、切片、省略号、numpy.newaxis 以及整数或布尔数组
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        else:
            # 检查 item 是否是有效的数组索引器，并将其标准化
            item = pd.api.indexers.check_array_indexer(self, item)
            # 如果 item 的 dtype 是布尔类型，返回一个新的相同类型的对象，包含符合条件的数据
            if is_bool_dtype(item.dtype):
                return type(self)._from_sequence(
                    [x for x, m in zip(self, item) if m], dtype=self.dtype
                )
            # 如果 item 是整数类型，返回一个新的相同类型的对象，包含指定索引的数据
            # 注意：这里的 item 应当是整数类型
            return type(self)([self.data[i] for i in item])

    # 实现赋值操作符 []= 的特殊方法，用于设置对象中指定项的值
    def __setitem__(self, key, value) -> None:
        # 如果 key 是整数类型，设置对象中索引为 key 的数据为 value
        if isinstance(key, numbers.Integral):
            self.data[key] = value
        else:
            # 如果 value 不是当前对象或序列类型的实例，将其广播成一个无限循环的迭代器
            if not isinstance(value, (type(self), abc.Sequence)):
                value = itertools.cycle([value])

            # 如果 key 是布尔类型的 numpy 数组，根据 mask 设置对应位置的数据为 value
            if isinstance(key, np.ndarray) and key.dtype == "bool":
                for i, (k, v) in enumerate(zip(key, value)):
                    if k:
                        assert isinstance(v, self.dtype.type)
                        self.data[i] = v
            else:
                # 否则，根据 key 和 value 的对应关系设置数据
                for k, v in zip(key, value):
                    assert isinstance(v, self.dtype.type)
                    self.data[k] = v

    # 返回对象中数据项的长度，即 len(self.data)
    def __len__(self) -> int:
        return len(self.data)

    # 实现等于操作符 == 的特殊方法，返回 NotImplemented，表示未实现该操作
    def __eq__(self, other):
        return NotImplemented

    # 实现不等于操作符 != 的特殊方法，返回 NotImplemented，表示未实现该操作
    def __ne__(self, other):
        return NotImplemented

    # 实现将对象转换为数组的特殊方法 __array__
    def __array__(self, dtype=None, copy=None):
        # 如果 dtype 为 None，则设置为 object 类型
        if dtype is None:
            dtype = object
        # 如果 dtype 是 object 类型，构造一个 1 维的对象数组
        if dtype == object:
            return construct_1d_object_array_from_listlike(list(self))
        # 否则，使用 numpy 将 self.data 转换为指定 dtype 的数组
        return np.asarray(self.data, dtype=dtype)

    # 返回对象数据所占用的字节数
    @property
    def nbytes(self) -> int:
        return sys.getsizeof(self.data)

    # 返回一个布尔数组，指示对象数据中的缺失值
    def isna(self):
        return np.array([x == self.dtype.na_value for x in self.data], dtype=bool)
    def take(self, indexer, allow_fill=False, fill_value=None):
        # 重新实现此方法，因为 NumPy 在将像 UserDict 这样的大小对象设置为 ndarray 的标量插槽时会出问题。
        # 将 indexer 转换为 NumPy 数组
        indexer = np.asarray(indexer)
        # 出错时的错误信息
        msg = (
            "Index is out of bounds or cannot do a "
            "non-empty take from an empty array."
        )

        if allow_fill:
            # 如果允许填充，则设置填充值
            if fill_value is None:
                fill_value = self.dtype.na_value
            # 检查边界
            if (indexer < -1).any():
                raise ValueError
            try:
                # 根据 indexer 获取 self.data 中的数据，如果 loc 为 -1，则使用 fill_value 填充
                output = [
                    self.data[loc] if loc != -1 else fill_value for loc in indexer
                ]
            except IndexError as err:
                raise IndexError(msg) from err
        else:
            try:
                # 根据 indexer 获取 self.data 中的数据
                output = [self.data[loc] for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err

        # 返回一个新的类实例，从 output 序列构造，使用当前的 dtype
        return type(self)._from_sequence(output, dtype=self.dtype)

    def copy(self):
        # 返回当前对象的一个副本，复制 self.data 列表
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        # NumPy 在所有字典长度相同时存在问题。
        # np.array([UserDict(...), UserDict(...)]) 失败，
        # 但 np.array([{...}, {...}]) 可行，所以进行类型转换。
        from pandas.core.arrays.string_ import StringDtype

        # 转换 dtype 为 pandas 的数据类型
        dtype = pandas_dtype(dtype)
        # 需要为 Series 构造函数添加此检查
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                # 如果需要复制，则返回当前对象的副本
                return self.copy()
            # 否则直接返回当前对象
            return self
        elif isinstance(dtype, StringDtype):
            # 如果 dtype 是字符串类型，将值转换为 str 类型
            value = self.astype(str)  # numpy doesn't like nested dicts
            # 构造 dtype 对应的数组类型
            arr_cls = dtype.construct_array_type()
            # 从 value 序列构造新的数组，使用指定的 dtype，不进行复制
            return arr_cls._from_sequence(value, dtype=dtype, copy=False)
        elif not copy:
            # 如果不需要复制，返回将 self 转换为字典后的数组，使用指定的 dtype
            return np.asarray([dict(x) for x in self], dtype=dtype)
        else:
            # 否则返回将 self 转换为字典后的数组，使用指定的 dtype，并进行复制
            return np.array([dict(x) for x in self], dtype=dtype, copy=copy)

    def unique(self):
        # 父类方法不适用，因为 np.array 会尝试推断一个二维对象。
        # 返回一个新的类实例，包含 self.data 的唯一元素
        return type(self)([dict(x) for x in {tuple(d.items()) for d in self.data}])

    @classmethod
    def _concat_same_type(cls, to_concat):
        # 将 to_concat 中的所有数据连接起来，返回一个新的类实例
        data = list(itertools.chain.from_iterable(x.data for x in to_concat))
        return cls(data)

    def _values_for_factorize(self):
        # 获取用于 factorize 的值，返回一个冻结的数组
        frozen = self._values_for_argsort()
        if len(frozen) == 0:
            # factorize_array 需要一个一维数组，这是一个长度为 0 的二维数组。
            frozen = frozen.ravel()
        return frozen, ()

    def _values_for_argsort(self):
        # 绕过 NumPy 的形状推断，获取一个 (N,) 的元组数组。
        frozen = [tuple(x.items()) for x in self]
        return construct_1d_object_array_from_listlike(frozen)
    # 继承父类方法 _pad_or_backfill，并添加特定参数的调用
    # GH#56616 - 测试不带 limit_area 参数的 EA 方法
    return super()._pad_or_backfill(method=method, limit=limit, copy=copy)
# 定义一个函数，用于生成数据
def make_data():
    # TODO: 使用普通的字典。参见 _NDFrameIndexer._setitem_with_indexer
    # 使用默认种子为2的随机数生成器创建一个实例
    rng = np.random.default_rng(2)
    # 返回一个包含100个元素的列表，每个元素是一个 UserDict 对象
    return [
        UserDict(
            # 对于每个 UserDict 对象，使用随机数生成器创建一个包含随机键值对的字典
            [
                (rng.choice(list(string.ascii_letters)), rng.integers(0, 100))
                # 随机生成包含0到9个元素的字典
                for _ in range(rng.integers(0, 10))
            ]
        )
        # 循环生成100个 UserDict 对象
        for _ in range(100)
    ]
```