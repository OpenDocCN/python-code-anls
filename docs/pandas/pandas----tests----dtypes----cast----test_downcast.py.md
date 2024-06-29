# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_downcast.py`

```
import decimal  # 导入 decimal 模块，用于处理精确的十进制运算

import numpy as np  # 导入 numpy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas.core.dtypes.cast import maybe_downcast_to_dtype  # 从 pandas 库中导入数据类型转换函数

from pandas import (  # 从 pandas 库中导入 Series 和 Timedelta 类
    Series,
    Timedelta,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用于多组参数运行同一个测试函数
    "arr,dtype,expected",  # 定义参数化的三个参数名
    [  # 参数化的测试数据列表开始
        (
            np.array([8.5, 8.6, 8.7, 8.8, 8.9999999999995]),  # 第一组参数：浮点数数组
            "infer",  # 推断数据类型
            np.array([8.5, 8.6, 8.7, 8.8, 8.9999999999995]),  # 预期结果为相同的浮点数数组
        ),
        (
            np.array([8.0, 8.0, 8.0, 8.0, 8.9999999999995]),  # 第二组参数：混合浮点数和整数数组
            "infer",  # 推断数据类型
            np.array([8, 8, 8, 8, 9], dtype=np.int64),  # 预期结果为整数数组
        ),
        (
            np.array([8.0, 8.0, 8.0, 8.0, 9.0000000000005]),  # 第三组参数：浮点数数组
            "infer",  # 推断数据类型
            np.array([8, 8, 8, 8, 9], dtype=np.int64),  # 预期结果为整数数组
        ),
        (
            # 这是一个判断，我们不会将 Decimal 对象向下转换
            np.array([decimal.Decimal(0.0)]),  # 第四组参数：Decimal 对象数组
            "int64",  # 数据类型为 int64
            np.array([decimal.Decimal(0.0)]),  # 预期结果为相同的 Decimal 对象数组
        ),
        (
            # GH#45837
            np.array([Timedelta(days=1), Timedelta(days=2)], dtype=object),  # 第五组参数：Timedelta 对象数组
            "infer",  # 推断数据类型
            np.array([1, 2], dtype="m8[D]").astype("m8[ns]"),  # 预期结果为转换为日期时间类型的数组
        ),
        # TODO: similar for dt64, dt64tz, Period, Interval?
    ],  # 参数化的测试数据列表结束
)
def test_downcast(arr, expected, dtype):
    result = maybe_downcast_to_dtype(arr, dtype)  # 调用数据类型转换函数
    tm.assert_numpy_array_equal(result, expected)  # 使用 pandas 的测试工具函数检查结果是否符合预期


def test_downcast_booleans():
    # 参见 gh-16875: 强制转换布尔值。
    ser = Series([True, True, False])  # 创建一个包含布尔值的 Series 对象
    result = maybe_downcast_to_dtype(ser, np.dtype(np.float64))  # 调用数据类型转换函数

    expected = ser.values  # 期望结果为原始布尔值数组
    tm.assert_numpy_array_equal(result, expected)  # 使用 pandas 的测试工具函数检查结果是否符合预期


def test_downcast_conversion_no_nan(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype  # 获取任意实数的 numpy 数据类型
    expected = np.array([1, 2])  # 期望结果为整数数组
    arr = np.array([1.0, 2.0], dtype=dtype)  # 创建一个包含浮点数的 numpy 数组

    result = maybe_downcast_to_dtype(arr, "infer")  # 调用数据类型转换函数
    tm.assert_almost_equal(result, expected, check_dtype=False)  # 使用 pandas 的测试工具函数检查结果是否符合预期


def test_downcast_conversion_nan(float_numpy_dtype):
    dtype = float_numpy_dtype  # 获取浮点数的 numpy 数据类型
    data = [1.0, 2.0, np.nan]  # 包含 NaN 值的数据列表

    expected = np.array(data, dtype=dtype)  # 期望结果为包含 NaN 的浮点数数组
    arr = np.array(data, dtype=dtype)  # 创建一个包含 NaN 的浮点数数组

    result = maybe_downcast_to_dtype(arr, "infer")  # 调用数据类型转换函数
    tm.assert_almost_equal(result, expected)  # 使用 pandas 的测试工具函数检查结果是否符合预期


def test_downcast_conversion_empty(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype  # 获取任意实数的 numpy 数据类型
    arr = np.array([], dtype=dtype)  # 创建一个空的 numpy 数组
    result = maybe_downcast_to_dtype(arr, np.dtype("int64"))  # 调用数据类型转换函数
    tm.assert_numpy_array_equal(result, np.array([], dtype=np.int64))  # 使用 pandas 的测试工具函数检查结果是否符合预期


@pytest.mark.parametrize("klass", [np.datetime64, np.timedelta64])
def test_datetime_likes_nan(klass):
    dtype = klass.__name__ + "[ns]"  # 获取日期时间类的 numpy 数据类型
    arr = np.array([1, 2, np.nan])  # 包含 NaN 值的数据列表

    exp = np.array([1, 2, klass("NaT")], dtype)  # 期望结果为包含 NaN 的日期时间数组
    res = maybe_downcast_to_dtype(arr, dtype)  # 调用数据类型转换函数
    tm.assert_numpy_array_equal(res, exp)  # 使用 pandas 的测试工具函数检查结果是否符合预期
```