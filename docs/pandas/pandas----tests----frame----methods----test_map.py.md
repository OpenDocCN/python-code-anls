# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_map.py`

```
# 导入所需模块和函数
from datetime import datetime

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入 DataFrame、Series、Timestamp、date_range 等
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部的测试模块

from pandas.tseries.offsets import BDay  # 从 Pandas 时间序列模块中导入 BDay 偏移量


def test_map(float_frame):
    # 对 float_frame 应用 map 方法，lambda 函数将每个元素乘以 2
    result = float_frame.map(lambda x: x * 2)
    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证结果是否与期望相等
    tm.assert_frame_equal(result, float_frame * 2)
    # 对 float_frame 应用 map 方法，lambda 函数获取每个元素的类型
    float_frame.map(type)

    # GH 465: 函数返回元组时的处理
    result = float_frame.map(lambda x: (x, x))["A"].iloc[0]
    # 验证返回值是否为元组类型
    assert isinstance(result, tuple)


@pytest.mark.parametrize("val", [1, 1.0])
def test_map_float_object_conversion(val):
    # GH 2909: 在构造函数中将对象转换为浮点数
    # 创建包含 val 和 "a" 的 DataFrame 对象
    df = DataFrame(data=[val, "a"])
    # 对 DataFrame 应用 map 方法，lambda 函数返回每个元素本身
    result = df.map(lambda x: x).dtypes[0]
    # 验证第一个列的数据类型是否为 object
    assert result == object


def test_map_keeps_dtype(na_action):
    # GH52219
    # 创建包含字符串和 NaN 值的 Series 对象 arr
    arr = Series(["a", np.nan, "b"])
    # 将 arr 转换为稀疏类型对象 sparse_arr
    sparse_arr = arr.astype(pd.SparseDtype(object))
    # 创建包含 arr 和 sparse_arr 的 DataFrame 对象 df
    df = DataFrame(data={"a": arr, "b": sparse_arr})

    # 定义函数 func，将字符串转换为大写，NaN 值保持不变
    def func(x):
        return str.upper(x) if not pd.isna(x) else x

    # 对 DataFrame 应用 map 方法，使用 func 函数处理 NaN 值
    result = df.map(func, na_action=na_action)

    # 创建预期的 DataFrame 对象 expected，包含处理后的数据
    expected_sparse = pd.array(["A", np.nan, "B"], dtype=pd.SparseDtype(object))
    expected_arr = expected_sparse.astype(object)
    expected = DataFrame({"a": expected_arr, "b": expected_sparse})

    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证结果是否与期望相等
    tm.assert_frame_equal(result, expected)

    # 对空 DataFrame 对象 df 应用 map 方法，使用 func 函数处理 NaN 值
    result_empty = df.iloc[:0, :].map(func, na_action=na_action)
    expected_empty = expected.iloc[:0, :]
    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证空 DataFrame 的结果是否与期望相等
    tm.assert_frame_equal(result_empty, expected_empty)


def test_map_str():
    # GH 2786
    # 创建一个随机数据的 DataFrame 对象 df
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    # 复制 df 创建 df2
    df2 = df.copy()
    cols = ["a", "a", "a", "a"]
    df.columns = cols

    # 对 df2 应用 map 方法，将每个元素转换为字符串
    expected = df2.map(str)
    expected.columns = cols
    # 对 df 应用 map 方法，将每个元素转换为字符串
    result = df.map(str)
    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证结果是否与期望相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "col, val",
    [["datetime", Timestamp("20130101")], ["timedelta", pd.Timedelta("1 min")]],
)
def test_map_datetimelike(col, val):
    # datetime/timedelta
    # 创建一个随机数据的 DataFrame 对象 df
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    # 将 val 赋值给 df 的指定列 col
    df[col] = val
    # 对 df 应用 map 方法，将每个元素转换为字符串
    result = df.map(str)
    # 验证第一行指定列 col 的值是否与原始值的字符串表示相等
    assert result.loc[0, col] == str(df.loc[0, col])


@pytest.mark.parametrize(
    "expected",
    [
        DataFrame(),  # 空 DataFrame 对象
        DataFrame(columns=list("ABC")),  # 列为 ABC 的 DataFrame 对象
        DataFrame(index=list("ABC")),  # 索引为 ABC 的 DataFrame 对象
        DataFrame({"A": [], "B": [], "C": []}),  # 指定列为空列表的 DataFrame 对象
    ],
)
@pytest.mark.parametrize("func", [round, lambda x: x])
def test_map_empty(expected, func):
    # GH 8222
    # 对预期结果 expected 应用 map 方法，使用 func 函数处理
    result = expected.map(func)
    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证结果是否与期望相等
    tm.assert_frame_equal(result, expected)


def test_map_kwargs():
    # GH 40652
    # 创建一个包含整数的 DataFrame 对象
    result = DataFrame([[1, 2], [3, 4]]).map(lambda x, y: x + y, y=2)
    expected = DataFrame([[3, 4], [5, 6]])
    # 使用 Pandas 测试模块中的 assert_frame_equal 方法验证结果是否与期望相等
    tm.assert_frame_equal(result, expected)


def test_map_na_ignore(float_frame):
    # GH 23803
    # 对 float_frame 应用 map 方法，lambda 函数计算每个元素的字符串长度
    strlen_frame = float_frame.map(lambda x: len(str(x)))
    # 复制 float_frame 创建 float_frame_with_na
    float_frame_with_na = float_frame.copy()
    # 创建一个布尔掩码，用于 float_frame_with_na
    mask = np.random.default_rng(2).integers(0, 2, size=float_frame.shape, dtype=bool)
    # 将 float_frame_with_na 中符合 mask 条件的元素设为 pd.NA
    float_frame_with_na[mask] = pd.NA
    # 创建一个新的 Series，对 float_frame_with_na 中的每个元素应用 lambda 函数，计算其字符串表示的长度，
    # 并在计算过程中忽略 NA 值
    strlen_frame_na_ignore = float_frame_with_na.map(
        lambda x: len(str(x)), na_action="ignore"
    )
    # 复制 strlen_frame，并将其数据类型转换为 float64，以避免在设置 NA 值时的类型提升
    strlen_frame_with_na = strlen_frame.copy().astype("float64")
    # 将 strlen_frame_with_na 中符合 mask 条件的元素设为 pd.NA
    strlen_frame_with_na[mask] = pd.NA
    # 使用 assert_frame_equal 函数断言两个 DataFrame 是否相等，即验证两者是否完全一致
    tm.assert_frame_equal(strlen_frame_na_ignore, strlen_frame_with_na)
def test_map_box_timestamps():
    # GH 2689, GH 2627
    # 创建一个时间序列，从"2000-01-01"开始，包含10个时间点
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x):
        # 定义一个函数，返回时间戳 x 的小时、日和月份
        return (x.hour, x.day, x.month)

    # 将 func 函数映射到 DataFrame 中的每个元素
    DataFrame(ser).map(func)


def test_map_box():
    # ufunc 不会被封装。与 test_map_box 相同的测试用例
    # 创建一个包含多列时间戳、时间增量和周期的 DataFrame
    df = DataFrame(
        {
            "a": [Timestamp("2011-01-01"), Timestamp("2011-01-02")],
            "b": [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
            ],
            "c": [pd.Timedelta("1 days"), pd.Timedelta("2 days")],
            "d": [
                pd.Period("2011-01-01", freq="M"),
                pd.Period("2011-01-02", freq="M"),
            ],
        }
    )

    # 将 lambda 函数应用于 DataFrame 的每个元素，返回元素的类型名称
    result = df.map(lambda x: type(x).__name__)
    # 期望的 DataFrame 结果，包含每列元素的类型名称
    expected = DataFrame(
        {
            "a": ["Timestamp", "Timestamp"],
            "b": ["Timestamp", "Timestamp"],
            "c": ["Timedelta", "Timedelta"],
            "d": ["Period", "Period"],
        }
    )
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_frame_map_dont_convert_datetime64(unit):
    dtype = f"M8[{unit}]"
    # 创建一个包含一个 datetime 类型列的 DataFrame
    df = DataFrame({"x1": [datetime(1996, 1, 1)]}, dtype=dtype)

    # 将 BDay() 应用于 df.x1 中的每个元素
    df = df.map(lambda x: x + BDay())
    # 再次将 BDay() 应用于 df.x1 中的每个元素
    df = df.map(lambda x: x + BDay())

    # 获取 df.x1 的数据类型
    result = df.x1.dtype
    # 断言结果数据类型与预期的 dtype 是否相等
    assert result == dtype


def test_map_function_runs_once():
    df = DataFrame({"a": [1, 2, 3]})
    values = []  # 保存应用函数的值

    def reducing_function(val):
        # 将值添加到 values 列表中
        values.append(val)

    def non_reducing_function(val):
        # 将值添加到 values 列表中，并返回值本身
        values.append(val)
        return val

    # 对于 reducing_function 和 non_reducing_function，分别执行以下操作
    for func in [reducing_function, non_reducing_function]:
        del values[:]  # 清空 values 列表

        # 将 func 函数映射到 DataFrame 的每个元素
        df.map(func)
        # 断言 values 列表是否与 df.a 的值列表相同
        assert values == df.a.to_list()


def test_map_type():
    # GH 46719
    # 创建一个包含不同类型数据的 DataFrame
    df = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )

    # 将 type 函数映射到 DataFrame 的每个元素
    result = df.map(type)
    # 期望的 DataFrame 结果，包含每列元素的类型
    expected = DataFrame(
        {"col1": [int, str, type], "col2": [float, datetime, float]},
        index=["a", "b", "c"],
    )
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_map_invalid_na_action(float_frame):
    # GH 23803
    # 使用 pytest 断言捕获 ValueError 异常，检查 na_action 参数
    with pytest.raises(ValueError, match="na_action must be .*Got 'abc'"):
        float_frame.map(lambda x: len(str(x)), na_action="abc")
```