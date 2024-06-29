# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_ndarray_backed.py`

```
"""
Tests for subclasses of NDArrayBackedExtensionArray
"""

import numpy as np  # 导入 NumPy 库

from pandas import (
    CategoricalIndex,  # 导入 CategoricalIndex 类
    date_range,  # 导入 date_range 函数
)
from pandas.core.arrays import (
    Categorical,  # 导入 Categorical 类
    DatetimeArray,  # 导入 DatetimeArray 类
    NumpyExtensionArray,  # 导入 NumpyExtensionArray 类
    TimedeltaArray,  # 导入 TimedeltaArray 类
)


class TestEmpty:
    def test_empty_categorical(self):
        ci = CategoricalIndex(["a", "b", "c"], ordered=True)  # 创建有序的 CategoricalIndex 对象
        dtype = ci.dtype  # 获取 CategoricalIndex 对象的数据类型

        # case with int8 codes
        shape = (4,)  # 定义形状为 (4,) 的元组
        result = Categorical._empty(shape, dtype=dtype)  # 使用 Categorical 类的 _empty 方法创建空对象
        assert isinstance(result, Categorical)  # 断言 result 是 Categorical 类的实例
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符
        assert result._ndarray.dtype == np.int8  # 断言 result 的底层 ndarray 的数据类型为 int8

        # case where repr would segfault if we didn't override base implementation
        result = Categorical._empty((4096,), dtype=dtype)  # 使用 Categorical 类的 _empty 方法创建空对象，形状为 (4096,)
        assert isinstance(result, Categorical)  # 断言 result 是 Categorical 类的实例
        assert result.shape == (4096,)  # 断言 result 的形状为 (4096,)
        assert result._ndarray.dtype == np.int8  # 断言 result 的底层 ndarray 的数据类型为 int8
        repr(result)  # 调用 repr 函数以避免段错误

        # case with int16 codes
        ci = CategoricalIndex(list(range(512)) * 4, ordered=False)  # 创建无序的 CategoricalIndex 对象
        dtype = ci.dtype  # 获取 CategoricalIndex 对象的数据类型
        result = Categorical._empty(shape, dtype=dtype)  # 使用 Categorical 类的 _empty 方法创建空对象
        assert isinstance(result, Categorical)  # 断言 result 是 Categorical 类的实例
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符
        assert result._ndarray.dtype == np.int16  # 断言 result 的底层 ndarray 的数据类型为 int16

    def test_empty_dt64tz(self):
        dti = date_range("2016-01-01", periods=2, tz="Asia/Tokyo")  # 创建带时区的日期范围对象
        dtype = dti.dtype  # 获取日期范围对象的数据类型

        shape = (0,)  # 定义形状为 (0,) 的元组
        result = DatetimeArray._empty(shape, dtype=dtype)  # 使用 DatetimeArray 类的 _empty 方法创建空对象
        assert result.dtype == dtype  # 断言 result 的数据类型与定义的数据类型相符
        assert isinstance(result, DatetimeArray)  # 断言 result 是 DatetimeArray 类的实例
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符

    def test_empty_dt64(self):
        shape = (3, 9)  # 定义形状为 (3, 9) 的元组
        result = DatetimeArray._empty(shape, dtype="datetime64[ns]")  # 使用 DatetimeArray 类的 _empty 方法创建空对象，指定数据类型为 datetime64[ns]
        assert isinstance(result, DatetimeArray)  # 断言 result 是 DatetimeArray 类的实例
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符

    def test_empty_td64(self):
        shape = (3, 9)  # 定义形状为 (3, 9) 的元组
        result = TimedeltaArray._empty(shape, dtype="m8[ns]")  # 使用 TimedeltaArray 类的 _empty 方法创建空对象，指定数据类型为 m8[ns]
        assert isinstance(result, TimedeltaArray)  # 断言 result 是 TimedeltaArray 类的实例
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符

    def test_empty_pandas_array(self):
        arr = NumpyExtensionArray(np.array([1, 2]))  # 创建 NumpyExtensionArray 对象
        dtype = arr.dtype  # 获取 NumpyExtensionArray 对象的数据类型

        shape = (3, 9)  # 定义形状为 (3, 9) 的元组
        result = NumpyExtensionArray._empty(shape, dtype=dtype)  # 使用 NumpyExtensionArray 类的 _empty 方法创建空对象
        assert isinstance(result, NumpyExtensionArray)  # 断言 result 是 NumpyExtensionArray 类的实例
        assert result.dtype == dtype  # 断言 result 的数据类型与定义的数据类型相符
        assert result.shape == shape  # 断言 result 的形状与定义的形状相符
```