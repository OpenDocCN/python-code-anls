# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_groupby_shift_diff.py`

```
import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm

# 定义测试函数，测试 groupby 结果中 shift 方法对缺失键的处理
def test_group_shift_with_null_key():
    # This test is designed to replicate the segfault in issue #13813.
    # 设定行数
    n_rows = 1200

    # 生成一个中等大小的 DataFrame，其中列 `B` 时而有缺失值，
    # 然后按 [`A`, `B`] 分组。这会在 `g._grouper.group_info` 的 `labels` 数组中
    # 正好在部分缺失的分组键位置处强制填入 `-1`。
    df = DataFrame(
        [(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    # 生成预期的 DataFrame，对应每一行的预期结果，若 i 为 3 的倍数且 i 小于 n_rows - 12，则加上 12，否则为 NaN
    expected = DataFrame(
        [(i + 12 if i % 3 and i < n_rows - 12 else np.nan) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    # 使用 shift 方法对分组后的结果进行操作
    result = g.shift(-1)

    # 使用测试模块中的函数来断言结果是否与预期一致
    tm.assert_frame_equal(result, expected)


# 测试 groupby 结果中 shift 方法对填充值的处理
def test_group_shift_with_fill_value():
    # GH #24128
    # 设定行数
    n_rows = 24
    # 生成一个 DataFrame，包含三列，列名分别为 A, B, Z，其中 A 和 B 使用模运算来生成，Z 使用索引值
    df = DataFrame(
        [(i % 12, i % 3, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    # 生成预期的 DataFrame，对应每一行的预期结果，若 i 小于 n_rows - 12，则加上 12，否则为 0
    expected = DataFrame(
        [(i + 12 if i < n_rows - 12 else 0) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    # 使用 shift 方法对分组后的结果进行操作，并使用 0 进行填充
    result = g.shift(-1, fill_value=0)

    # 使用测试模块中的函数来断言结果是否与预期一致
    tm.assert_frame_equal(result, expected)


# 测试 groupby 结果中 shift 方法对时区的处理
def test_group_shift_lose_timezone():
    # GH 30134
    # 获取当前时间戳，以 UTC 时区表示，并转换为纳秒单位
    now_dt = Timestamp.now("UTC").as_unit("ns")
    # 生成一个 DataFrame，包含一列名为 'a'，值为 [1, 1]，一列名为 'date'，值为 now_dt
    df = DataFrame({"a": [1, 1], "date": now_dt})
    # 对 'a' 列进行分组，然后对结果使用 shift(0) 方法，并选取第一行
    result = df.groupby("a").shift(0).iloc[0]
    # 生成预期的 Series，其唯一条目名为 'date'，值为 now_dt
    expected = Series({"date": now_dt}, name=result.name)
    # 使用测试模块中的函数来断言结果是否与预期一致
    tm.assert_series_equal(result, expected)


# 测试 groupby 结果中 diff 方法对真实 Series 的处理
def test_group_diff_real_series(any_real_numpy_dtype):
    # 生成一个 DataFrame，包含两列 'a' 和 'b'，其中 'a' 列值为 [1, 2, 3, 3, 2]，'b' 列值为 [1, 2, 3, 4, 5]
    df = DataFrame(
        {"a": [1, 2, 3, 3, 2], "b": [1, 2, 3, 4, 5]},
        dtype=any_real_numpy_dtype,
    )
    # 对 'a' 列进行分组，然后对 'b' 列应用 diff 方法
    result = df.groupby("a")["b"].diff()
    # 根据 any_real_numpy_dtype 的值选择预期的 dtype
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    # 生成预期的 Series，其值为 [NaN, NaN, NaN, 1.0, 3.0]，dtype 为 exp_dtype，名为 'b'
    expected = Series([np.nan, np.nan, np.nan, 1.0, 3.0], dtype=exp_dtype, name="b")
    # 使用测试模块中的函数来断言结果是否与预期一致
    tm.assert_series_equal(result, expected)


# 测试 groupby 结果中 diff 方法对真实 DataFrame 的处理
def test_group_diff_real_frame(any_real_numpy_dtype):
    # 生成一个 DataFrame，包含三列 'a', 'b', 'c'，其中 'a' 列值为 [1, 2, 3, 3, 2]，'b' 列值为 [1, 2, 3, 4, 5]，'c' 列值为 [1, 2, 3, 4, 6]
    df = DataFrame(
        {
            "a": [1, 2, 3, 3, 2],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 6],
        },
        dtype=any_real_numpy_dtype,
    )
    # 对 'a' 列进行分组，然后对整个 DataFrame 应用 diff 方法
    result = df.groupby("a").diff()
    # 根据 any_real_numpy_dtype 的值选择预期的 dtype
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    # 生成预期的 DataFrame，其中 'b' 列为 [NaN, NaN, NaN, 1.0, 3.0]，'c' 列为 [NaN, NaN, NaN, 1.0, 4.0]，dtype 为 exp_dtype
    expected = DataFrame(
        {
            "b": [np.nan, np.nan, np.nan, 1.0, 3.0],
            "c": [np.nan, np.nan, np.nan, 1.0, 4.0],
        },
        dtype=exp_dtype,
    )
    # 使用测试模块中的函数来断言结果是否与预期一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        # 第一个子列表包含三个时间戳对象，分别对应于 2013 年 1 月 1 日、2013 年 1 月 2 日和 2013 年 1 月 3 日
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            Timestamp("2013-01-03"),
        ],
        # 第二个子列表包含三个时间增量对象，分别表示 5 天、6 天和 7 天
        [Timedelta("5 days"), Timedelta("6 days"), Timedelta("7 days")],
    ],
def test_group_diff_datetimelike(data, unit):
    # 创建 DataFrame，包含列"a"和"b"，其中"b"使用传入的日期时间数据
    df = DataFrame({"a": [1, 2, 2], "b": data})
    # 将"b"列转换为指定单位的日期时间类型
    df["b"] = df["b"].dt.as_unit(unit)
    # 对每个分组的"b"列计算相邻元素的差值
    result = df.groupby("a")["b"].diff()
    # 创建预期的 Series，包含与结果对应的预期差值
    expected = Series([NaT, NaT, Timedelta("1 days")], name="b").dt.as_unit(unit)
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_series_equal(result, expected)


def test_group_diff_bool():
    # 创建 DataFrame，包含列"a"和"b"，其中"b"使用布尔值数据
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    # 对每个分组的"b"列计算相邻元素的差值
    result = df.groupby("a")["b"].diff()
    # 创建预期的 Series，包含与结果对应的预期差值
    expected = Series([np.nan, np.nan, np.nan, False, False], name="b")
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_series_equal(result, expected)


def test_group_diff_object_raises(object_dtype):
    # 创建 DataFrame，包含列"a"和"b"，其中"b"使用对象类型的数据
    df = DataFrame(
        {"a": ["foo", "bar", "bar"], "b": ["baz", "foo", "foo"]}, dtype=object_dtype
    )
    # 使用 pytest 检查对象类型数据在分组操作时是否引发类型错误异常
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -"):
        df.groupby("a")["b"].diff()


def test_empty_shift_with_fill():
    # GH 41264, 单索引检查
    # 创建空列的 DataFrame
    df = DataFrame(columns=["a", "b", "c"])
    # 对每个分组的 DataFrame 进行向前偏移1行
    shifted = df.groupby(["a"]).shift(1)
    # 对每个分组的 DataFrame 进行向前偏移1行，并使用填充值0填充缺失值
    shifted_with_fill = df.groupby(["a"]).shift(1, fill_value=0)
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_frame_equal(shifted, shifted_with_fill)
    # 使用测试工具比较偏移后的索引是否相等
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_multindex_empty_shift_with_fill():
    # GH 41264, 多索引检查
    # 创建空列的 DataFrame
    df = DataFrame(columns=["a", "b", "c"])
    # 对每个分组的 DataFrame 进行向前偏移1行
    shifted = df.groupby(["a", "b"]).shift(1)
    # 对每个分组的 DataFrame 进行向前偏移1行，并使用填充值0填充缺失值
    shifted_with_fill = df.groupby(["a", "b"]).shift(1, fill_value=0)
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_frame_equal(shifted, shifted_with_fill)
    # 使用测试工具比较偏移后的索引是否相等
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_shift_periods_freq():
    # GH 54093
    # 创建包含日期范围和数据的 DataFrame
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    # 对每个日期分组的 DataFrame 进行向前偏移2个周期，并保持日期频率为天
    result = df.groupby(df.index).shift(periods=-2, freq="D")
    # 创建预期的 DataFrame，包含偏移后的预期结果
    expected = DataFrame(data, index=date_range(start="2009-12-30", periods=6))
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_shift_disallow_freq_and_fill_value():
    # GH 53832
    # 创建包含日期范围和数据的 DataFrame
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    msg = "Passing a 'freq' together with a 'fill_value'"
    # 使用 pytest 检查在指定频率的同时传递填充值是否引发值错误异常
    with pytest.raises(ValueError, match=msg):
        df.groupby(df.index).shift(periods=-2, freq="D", fill_value="1")


def test_shift_disallow_suffix_if_periods_is_int():
    # GH#44424
    # 创建包含数据的 DataFrame
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data)
    msg = "Cannot specify `suffix` if `periods` is an int."
    # 使用 pytest 检查在指定整数周期的同时传递后缀是否引发值错误异常
    with pytest.raises(ValueError, match=msg):
        df.groupby("b").shift(1, suffix="fails")


def test_group_shift_with_multiple_periods():
    # GH#44424
    # 创建包含数据的 DataFrame
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    # 对每个分组的 DataFrame 进行向前偏移多个周期
    shifted_df = df.groupby("b")[["a"]].shift([0, 1])
    # 创建预期的 DataFrame，包含偏移后的预期结果
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 3, 2], "a_1": [np.nan, 1.0, np.nan, 3.0, 2.0]}
    )
    # 使用测试工具比较结果和预期结果是否相等
    tm.assert_frame_equal(shifted_df, expected_df)
    # 对 DataFrame df 按列 'b' 进行分组，然后对列 'a' 进行平移操作，平移步长为 [0, 1]。
    shifted_series = df.groupby("b")["a"].shift([0, 1])
    
    # 使用测试工具 tm 对平移后的 Series shifted_series 进行与预期 DataFrame expected_df 的内容比较。
    tm.assert_frame_equal(shifted_series, expected_df)
# 定义一个测试函数，用于测试带有多个周期和频率的组数据位移操作
def test_group_shift_with_multiple_periods_and_freq():
    # GH#44424：GitHub issue 编号
    # 创建一个包含两列 'a' 和 'b' 的数据框 DataFrame
    # 'a' 列包含 [1, 2, 3, 4, 5]，'b' 列包含 [True, True, False, False, True]
    # 索引为从 "1/1/2000" 开始的 5 个小时频率的日期时间索引
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    # 对数据框按 'b' 列进行分组，然后对 'a' 列进行时间频率为小时的位移操作
    shifted_df = df.groupby("b")[["a"]].shift(
        [0, 1],  # 对 'a' 列进行位移的周期列表
        freq="h",  # 指定位移的频率为小时
    )
    # 期望的结果数据框，包含两列 'a_0' 和 'a_1'
    # 'a_0' 列的期望值是 [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]
    # 'a_1' 列的期望值是 [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected_df = DataFrame(
        {
            "a_0": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "a_1": [
                np.nan,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        },
        index=date_range("1/1/2000", periods=6, freq="h"),  # 期望的索引，增加到 6 个小时
    )
    # 使用 pytest 模块的 assert_frame_equal 函数比较位移后的数据框和期望的数据框
    tm.assert_frame_equal(shifted_df, expected_df)


# 定义一个测试函数，用于测试带有多个周期和填充值的组数据位移操作
def test_group_shift_with_multiple_periods_and_fill_value():
    # GH#44424：GitHub issue 编号
    # 创建一个包含两列 'a' 和 'b' 的数据框 DataFrame
    # 'a' 列包含 [1, 2, 3, 4, 5]，'b' 列包含 [True, True, False, False, True]
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
    )
    # 对数据框按 'b' 列进行分组，然后对 'a' 列进行多周期的位移操作，使用 -1 进行填充
    shifted_df = df.groupby("b")[["a"]].shift([0, 1], fill_value=-1)
    # 期望的结果数据框，包含两列 'a_0' 和 'a_1'
    # 'a_0' 列的期望值是 [1, 2, 3, 4, 5]
    # 'a_1' 列的期望值是 [-1, 1, -1, 3, 2]
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 4, 5], "a_1": [-1, 1, -1, 3, 2]},
    )
    # 使用 pytest 模块的 assert_frame_equal 函数比较位移后的数据框和期望的数据框
    tm.assert_frame_equal(shifted_df, expected_df)


# 定义一个测试函数，用于测试带有多个周期、填充值和频率（已弃用）的组数据位移操作
def test_group_shift_with_multiple_periods_and_both_fill_and_freq_deprecated():
    # GH#44424：GitHub issue 编号
    # 创建一个包含两列 'a' 和 'b' 的数据框 DataFrame
    # 'a' 列包含 [1, 2, 3, 4, 5]，'b' 列包含 [True, True, False, False, True]
    # 索引为从 "1/1/2000" 开始的 5 个小时频率的日期时间索引
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    # 准备一个错误信息，用于 pytest 抛出异常时的匹配
    msg = "Passing a 'freq' together with a 'fill_value'"
    # 使用 pytest 的 raises 函数验证在组数据位移操作时传递了已弃用的 'freq' 参数
    with pytest.raises(ValueError, match=msg):
        df.groupby("b")[["a"]].shift([1, 2], fill_value=1, freq="h")
```