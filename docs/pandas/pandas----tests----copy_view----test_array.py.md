# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_array.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

from pandas import (  # 从 Pandas 库中导入以下对象：
    DataFrame,  # 数据框对象
    Series,  # 系列对象
    date_range,  # 日期范围生成器
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.copy_view.util import get_array  # 从复制视图测试工具中导入 get_array 函数

# -----------------------------------------------------------------------------
# Copy/view behaviour for accessing underlying array of Series/DataFrame

# 测试函数，用于测试获取 Series 底层数组的复制和视图行为
@pytest.mark.parametrize(
    "method",
    [lambda ser: ser.values, lambda ser: np.asarray(ser)],  # 参数化测试方法：通过 .values 和 np.asarray 获取数组
    ids=["values", "asarray"],  # 参数化测试方法的标识
)
def test_series_values(method):
    ser = Series([1, 2, 3], name="name")  # 创建一个名为 'name' 的 Series 对象
    ser_orig = ser.copy()  # 备份原始 Series 对象

    arr = method(ser)  # 使用给定的方法获取 Series 底层数组

    # .values 仍然提供视图但只读
    assert np.shares_memory(arr, get_array(ser, "name"))  # 断言底层数组与原始数据共享内存
    assert arr.flags.writeable is False  # 断言底层数组为只读

    # 通过底层数组修改 Series 将会失败
    with pytest.raises(ValueError, match="read-only"):  # 使用 pytest 断言捕获预期的 ValueError 异常
        arr[0] = 0
    tm.assert_series_equal(ser, ser_orig)  # 使用 Pandas 内部测试工具验证 Series 未被修改

    # 直接修改 Series 对象仍然有效
    ser.iloc[0] = 0
    assert ser.values[0] == 0  # 断言 Series 的值已被修改为 0


# 测试函数，用于测试获取 DataFrame 底层数组的复制和视图行为
@pytest.mark.parametrize(
    "method",
    [lambda df: df.values, lambda df: np.asarray(df)],  # 参数化测试方法：通过 .values 和 np.asarray 获取数组
    ids=["values", "asarray"],  # 参数化测试方法的标识
)
def test_dataframe_values(method):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # 创建一个 DataFrame 对象
    df_orig = df.copy()  # 备份原始 DataFrame 对象

    arr = method(df)  # 使用给定的方法获取 DataFrame 底层数组

    # .values 仍然提供视图但只读
    assert np.shares_memory(arr, get_array(df, "a"))  # 断言底层数组与原始数据共享内存
    assert arr.flags.writeable is False  # 断言底层数组为只读

    # 通过底层数组修改 DataFrame 将会失败
    with pytest.raises(ValueError, match="read-only"):  # 使用 pytest 断言捕获预期的 ValueError 异常
        arr[0, 0] = 0
    tm.assert_frame_equal(df, df_orig)  # 使用 Pandas 内部测试工具验证 DataFrame 未被修改

    # 直接修改 DataFrame 对象仍然有效
    df.iloc[0, 0] = 0
    assert df.values[0, 0] == 0  # 断言 DataFrame 的值已被修改为 0


# 测试函数，用于测试将 Series 转换为 NumPy 数组的行为
def test_series_to_numpy():
    ser = Series([1, 2, 3], name="name")  # 创建一个名为 'name' 的 Series 对象
    ser_orig = ser.copy()  # 备份原始 Series 对象

    # 默认情况下：copy=False，无 dtype 或 NA 值
    arr = ser.to_numpy()  # 将 Series 转换为 NumPy 数组
    # to_numpy 仍然提供视图但只读
    assert np.shares_memory(arr, get_array(ser, "name"))  # 断言底层数组与原始数据共享内存
    assert arr.flags.writeable is False  # 断言底层数组为只读

    # 通过底层数组修改 Series 将会失败
    with pytest.raises(ValueError, match="read-only"):  # 使用 pytest 断言捕获预期的 ValueError 异常
        arr[0] = 0
    tm.assert_series_equal(ser, ser_orig)  # 使用 Pandas 内部测试工具验证 Series 未被修改

    # 直接修改 Series 对象仍然有效
    ser.iloc[0] = 0
    assert ser.values[0] == 0  # 断言 Series 的值已被修改为 0

    # 指定 copy=True 将给出可写入的数组
    ser = Series([1, 2, 3], name="name")  # 创建一个名为 'name' 的 Series 对象
    arr = ser.to_numpy(copy=True)  # 将 Series 转换为 NumPy 数组并指定 copy=True
    assert not np.shares_memory(arr, get_array(ser, "name"))  # 断言底层数组与原始数据不共享内存
    assert arr.flags.writeable is True  # 断言底层数组为可写入的

    # 指定会导致复制的 dtype 也会给出可写入的数组
    ser = Series([1, 2, 3], name="name")  # 创建一个名为 'name' 的 Series 对象
    arr = ser.to_numpy(dtype="float64")  # 将 Series 转换为指定 dtype 的 NumPy 数组
    assert not np.shares_memory(arr, get_array(ser, "name"))  # 断言底层数组与原始数据不共享内存
    assert arr.flags.writeable is True  # 断言底层数组为可写入的


# 测试函数，用于测试将 Series 转换为特定 dtype 的 NumPy 数组的行为
def test_series_array_ea_dtypes():
    ser = Series([1, 2, 3], dtype="Int64")  # 创建一个带有特定 dtype 的 Series 对象
    arr = np.asarray(ser, dtype="int64")  # 将 Series 转换为指定 dtype 的 NumPy 数组
    assert np.shares_memory(arr, get_array(ser))  # 断言底层数组与原始数据共享内存
    assert arr.flags.writeable is False  # 断言底层数组为只读

    arr = np.asarray(ser)  # 将 Series 转换为 NumPy 数组
    # 断言：验证 arr 和 get_array(ser) 是否共享内存
    assert np.shares_memory(arr, get_array(ser))
    
    # 断言：验证 arr 的可写标志是否为 False
    assert arr.flags.writeable is False
# 测试DataFrame和NumPy数组之间的互操作性和属性

def test_dataframe_array_ea_dtypes():
    # 创建一个DataFrame对象，其中包含一个列 'a'，其值为 [1, 2, 3]，并指定列的数据类型为 'Int64'
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    # 将DataFrame转换为NumPy数组，指定数组的数据类型为 'int64'
    arr = np.asarray(df, dtype="int64")
    # 断言NumPy数组和由DataFrame列 'a' 返回的数组共享内存
    assert np.shares_memory(arr, get_array(df, "a"))
    # 断言NumPy数组不可写
    assert arr.flags.writeable is False

    # 再次将DataFrame转换为NumPy数组，默认数据类型
    arr = np.asarray(df)
    # 断言NumPy数组和由DataFrame列 'a' 返回的数组共享内存
    assert np.shares_memory(arr, get_array(df, "a"))
    # 断言NumPy数组不可写
    assert arr.flags.writeable is False


def test_dataframe_array_string_dtype():
    # 创建一个DataFrame对象，其中包含一个列 'a'，其值为 ['a', 'b']，并指定列的数据类型为 'string'
    df = DataFrame({"a": ["a", "b"]}, dtype="string")
    # 将DataFrame转换为NumPy数组
    arr = np.asarray(df)
    # 断言NumPy数组和由DataFrame列 'a' 返回的数组共享内存
    assert np.shares_memory(arr, get_array(df, "a"))
    # 断言NumPy数组不可写
    assert arr.flags.writeable is False


def test_dataframe_multiple_numpy_dtypes():
    # 创建一个DataFrame对象，包含两列 'a' 和 'b'，其中 'a' 是整数列 [1, 2, 3]，'b' 是浮点数列 [1.5, 1.5, 1.5]
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    # 将DataFrame转换为NumPy数组
    arr = np.asarray(df)
    # 断言NumPy数组和由DataFrame列 'a' 返回的数组不共享内存
    assert not np.shares_memory(arr, get_array(df, "a"))
    # 断言NumPy数组可写
    assert arr.flags.writeable is True


def test_values_is_ea():
    # 创建一个DataFrame对象，包含一个列 'a'，其值为从 '2012-01-01' 开始的日期范围，共3个日期
    df = DataFrame({"a": date_range("2012-01-01", periods=3)})
    # 将DataFrame转换为NumPy数组
    arr = np.asarray(df)
    # 断言NumPy数组不可写
    assert arr.flags.writeable is False


def test_empty_dataframe():
    # 创建一个空DataFrame对象
    df = DataFrame()
    # 将DataFrame转换为NumPy数组
    arr = np.asarray(df)
    # 断言NumPy数组可写
    assert arr.flags.writeable is True
```