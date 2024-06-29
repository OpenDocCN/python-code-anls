# `D:\src\scipysrc\pandas\pandas\tests\window\test_rolling.py`

```
# 从 datetime 模块中导入 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)

# 导入 numpy 库，并用 np 别名表示
import numpy as np

# 导入 pytest 库，用于单元测试
import pytest

# 从 pandas.compat 模块中导入以下函数和布尔值变量
from pandas.compat import (
    IS64,
    is_platform_arm,
    is_platform_power,
    is_platform_riscv64,
)

# 从 pandas 库中导入多个类和函数
from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)

# 导入 pandas._testing 模块，并用 tm 别名表示
import pandas._testing as tm

# 从 pandas.api.indexers 模块中导入 BaseIndexer 类
from pandas.api.indexers import BaseIndexer

# 从 pandas.core.indexers.objects 模块中导入 VariableOffsetWindowIndexer 类
from pandas.core.indexers.objects import VariableOffsetWindowIndexer

# 从 pandas.tseries.offsets 模块中导入 BusinessDay 类
from pandas.tseries.offsets import BusinessDay


# 定义一个单元测试函数，用于测试文档字符串的使用
def test_doc_string():
    # 创建一个 DataFrame 对象，包含列 B，值为 [0, 1, 2, NaN, 4]
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df  # 打印 DataFrame df

    # 对 DataFrame df 应用窗口大小为 2 的滚动求和操作
    df.rolling(2).sum()

    # 对 DataFrame df 应用窗口大小为 2，最小观测数为 1 的滚动求和操作
    df.rolling(2, min_periods=1).sum()


# 定义一个单元测试函数，用于测试构造函数的不同参数
def test_constructor(frame_or_series):
    # GH 12669

    # 获取 frame_or_series 中的前 5 个元素，并创建一个滚动对象 c
    c = frame_or_series(range(5)).rolling

    # 使用不同的参数调用滚动对象 c 的方法，进行有效性测试
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)

    # GH 13383

    # 定义错误消息字符串
    msg = "window must be an integer 0 or greater"

    # 使用 pytest 的 raises 方法，检查是否抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        c(-1)


# 定义一个参数化测试函数，用于测试构造函数中的无效参数
@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    # not valid

    # 获取 frame_or_series 中的前 5 个元素，并创建一个滚动对象 c
    c = frame_or_series(range(5)).rolling

    # 定义错误消息字符串，包含多个可能的错误信息
    msg = "|".join(
        [
            "window must be an integer",
            "passed window foo is not compatible with a datetimelike index",
        ]
    )

    # 使用 pytest 的 raises 方法，检查是否抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        c(window=w)

    # 定义错误消息字符串
    msg = "min_periods must be an integer"

    # 使用 pytest 的 raises 方法，检查是否抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)

    # 定义错误消息字符串
    msg = "center must be a boolean"

    # 使用 pytest 的 raises 方法，检查是否抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)


# 定义一个参数化测试函数，用于测试不支持的频率窗口
@pytest.mark.parametrize(
    "window",
    [
        timedelta(days=3),
        Timedelta(days=3),
        "3D",
        VariableOffsetWindowIndexer(
            index=date_range("2015-12-25", periods=5), offset=BusinessDay(1)
        ),
    ],
)
def test_freq_window_not_implemented(window):
    # GH 15354

    # 创建一个包含索引和数据的 DataFrame 对象
    df = DataFrame(
        np.arange(10),
        index=date_range("2015-12-24", periods=10, freq="D"),
    )

    # 使用 pytest 的 raises 方法，检查是否抛出 NotImplementedError 异常，并匹配指定消息
    with pytest.raises(
        NotImplementedError, match="^step (not implemented|is not supported)"
    ):
        df.rolling(window, step=3).sum()


# 定义一个参数化测试函数，用于测试协方差和相关性函数中不支持的步长参数
@pytest.mark.parametrize("agg", ["cov", "corr"])
def test_step_not_implemented_for_cov_corr(agg):
    # GH 15354

    # 创建一个包含两个元素的 DataFrame 对象，并调用其 rolling 方法
    roll = DataFrame(range(2)).rolling(1, step=2)

    # 使用 pytest 的 raises 方法，检查是否抛出 NotImplementedError 异常，并匹配指定消息
    with pytest.raises(NotImplementedError, match="step not implemented"):
        getattr(roll, agg)()


# 定义一个参数化测试函数，用于测试使用时间增量窗口的构造函数
@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window):
    # GH 15440

    # 定义数据数量 n
    n = 10

    # 创建一个包含值为 0 到 n-1 的列 'value' 的 DataFrame 对象
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )

    # 预期的结果数据，根据窗口大小计算得到
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))

    # 对 DataFrame df 应用指定窗口大小的滚动求和操作，并保存结果
    result = df.rolling(window=window).sum()


这些注释解释了每个代码块的作用，确保代码中的每个步骤和调用都清晰可见。
    # 创建一个 DataFrame 对象，其中包含名为 "value" 的列，列的数据从 expected_data 中获取，
    # 索引为从 "2015-12-24" 开始的 n 个日期，日期频率为每天 ("D")
    expected = DataFrame(
        {"value": expected_data},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    
    # 使用测试模块 tm 来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame df 应用滚动窗口 (rolling window)，窗口大小为 "3D" (3天)，并对每个窗口进行求和
    expected = df.rolling("3D").sum()
    
    # 再次使用 tm 模块来比较 result 和新的 expected DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的参数化装饰器，为 test_constructor_timedelta_window_and_minperiods 函数多次运行测试
@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3), "3D"])
def test_constructor_timedelta_window_and_minperiods(window, raw):
    # GH 15305: GitHub issue编号，标识此测试的背景
    n = 10
    # 创建一个包含数值列的 DataFrame，以日期为索引
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    # 期望的结果 DataFrame，包含预期的数值列及相同的日期索引
    expected = DataFrame(
        {"value": np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    # 使用滚动窗口计算滑动窗口大小为 window 的总和
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    # 使用滚动窗口应用函数 sum 计算结果
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    # 使用 pytest 的测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)


# 测试滚动窗口中不同关闭类型的行为
def test_closed_fixed(closed, arithmetic_win_operators):
    # GH 34315: GitHub issue编号，标识此测试的背景
    func_name = arithmetic_win_operators
    # 创建一个固定索引的 DataFrame，用于测试滚动窗口的行为
    df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
    # 创建一个具有时间索引的 DataFrame，用于预期的结果
    df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

    # 使用 getattr 动态获取指定属性或方法
    result = getattr(
        df_fixed.rolling(2, closed=closed, min_periods=1),
        func_name,
    )()
    # 预期的结果，使用相同的方法和参数应用于 df_time
    expected = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1),
        func_name,
    )().reset_index(drop=True)

    # 使用 pytest 的测试工具比较两个结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试不同的滚动窗口选项和关闭类型
@pytest.mark.parametrize(
    "closed, window_selections",
    [
        (
            "both",
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ],
        ),
        (
            "left",
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
        ),
        (
            "right",
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ],
        ),
        (
            "neither",
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
        ),
    ],
)
def test_datetimelike_centered_selections(
    closed, window_selections, arithmetic_win_operators
):
    # GH 34315: GitHub issue编号，标识此测试的背景
    func_name = arithmetic_win_operators
    # 创建一个时间索引和数值列的 DataFrame，用于测试
    df_time = DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range("2020", periods=5)
    )
    # 创建一个期望的DataFrame对象，包含从df_time中提取每个窗口选择的"A"列的结果，通过调用指定的函数func_name来计算。
    expected = DataFrame(
        {"A": [getattr(df_time["A"].iloc[s], func_name)() for s in window_selections]},
        index=date_range("2020", periods=5),
    )
    
    # 根据函数名称func_name来确定是否需要设置特定的kwargs参数，用于后续函数调用。
    if func_name == "sem":
        kwargs = {"ddof": 0}
    else:
        kwargs = {}
    
    # 调用getattr函数，动态获取df_time对象的rolling方法，通过指定的参数创建一个滚动对象，并应用func_name指定的函数。
    result = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1, center=True),
        func_name,
    )(**kwargs)
    
    # 使用测试模块中的assert_frame_equal函数，比较计算结果result和期望值expected是否相等，忽略数据类型的检查。
    tm.assert_frame_equal(result, expected, check_dtype=False)
@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("3s", "right", [3.0, 3.0, 3.0]),  # 设置滚动窗口大小为3秒，边界关闭方式为右侧，期望结果是每个窗口内的和为3.0
        ("3s", "both", [3.0, 3.0, 3.0]),   # 设置滚动窗口大小为3秒，边界关闭方式为两侧，期望结果是每个窗口内的和为3.0
        ("3s", "left", [3.0, 3.0, 3.0]),   # 设置滚动窗口大小为3秒，边界关闭方式为左侧，期望结果是每个窗口内的和为3.0
        ("3s", "neither", [3.0, 3.0, 3.0]),   # 设置滚动窗口大小为3秒，边界不关闭，期望结果是每个窗口内的和为3.0
        ("2s", "right", [3.0, 2.0, 2.0]),   # 设置滚动窗口大小为2秒，边界关闭方式为右侧，期望结果是每个窗口内的和为3.0、2.0、2.0
        ("2s", "both", [3.0, 3.0, 3.0]),   # 设置滚动窗口大小为2秒，边界关闭方式为两侧，期望结果是每个窗口内的和为3.0
        ("2s", "left", [1.0, 3.0, 3.0]),   # 设置滚动窗口大小为2秒，边界关闭方式为左侧，期望结果是每个窗口内的和为1.0、3.0、3.0
        ("2s", "neither", [1.0, 2.0, 2.0]),   # 设置滚动窗口大小为2秒，边界不关闭，期望结果是每个窗口内的和为1.0、2.0、2.0
    ],
)
def test_datetimelike_centered_offset_covers_all(
    window, closed, expected, frame_or_series
):
    # GH 42753
    # 创建时间戳索引
    index = [
        Timestamp("20130101 09:00:01"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:02"),
    ]
    # 创建DataFrame或Series对象，填充数据为1，使用上述时间戳索引
    df = frame_or_series([1, 1, 1], index=index)

    # 计算滚动窗口内的和，结果存储在result中
    result = df.rolling(window, closed=closed, center=True).sum()
    # 创建预期的DataFrame或Series对象，填充预期结果和使用上述时间戳索引
    expected = frame_or_series(expected, index=index)
    # 断言结果与预期结果相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2D", "right", [4, 4, 4, 4, 4, 4, 2, 2]),   # 设置滚动窗口大小为2天，边界关闭方式为右侧，期望结果是每个窗口内的和
        ("2D", "left", [2, 2, 4, 4, 4, 4, 4, 4]),   # 设置滚动窗口大小为2天，边界关闭方式为左侧，期望结果是每个窗口内的和
        ("2D", "both", [4, 4, 6, 6, 6, 6, 4, 4]),   # 设置滚动窗口大小为2天，边界关闭方式为两侧，期望结果是每个窗口内的和
        ("2D", "neither", [2, 2, 2, 2, 2, 2, 2, 2]),   # 设置滚动窗口大小为2天，边界不关闭，期望结果是每个窗口内的和
    ],
)
def test_datetimelike_nonunique_index_centering(
    window, closed, expected, frame_or_series
):
    # GH 20712
    # 创建日期时间索引
    index = DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
            "2020-01-03",
            "2020-01-04",
            "2020-01-04",
        ]
    )

    # 创建DataFrame或Series对象，填充数据为1.0，使用上述日期时间索引
    df = frame_or_series([1] * 8, index=index, dtype=float)
    # 创建预期的DataFrame或Series对象，填充预期结果和使用上述日期时间索引
    expected = frame_or_series(expected, index=index, dtype=float)

    # 计算滚动窗口内的和，结果存储在result中
    result = df.rolling(window, center=True, closed=closed).sum()

    # 断言结果与预期结果相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("left", [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]),   # 边界关闭方式为左侧，期望结果是每个窗口内的和
        ("neither", [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]),   # 边界不关闭，期望结果是每个窗口内的和
        ("right", [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]),   # 边界关闭方式为右侧，期望结果是每个窗口内的和
        ("both", [0, 1, 3, 6, 10, 15, 20, 27, 26, 30]),   # 边界关闭方式为两侧，期望结果是每个窗口内的和
    ],
)
def test_variable_window_nonunique(closed, expected, frame_or_series):
    # GH 20712
    # 创建日期时间索引
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    # 创建DataFrame或Series对象，填充数据为0到9，使用上述日期时间索引
    df = frame_or_series(range(10), index=index, dtype=float)
    # 创建预期的DataFrame或Series对象，填充预期结果和使用上述日期时间索引
    expected = frame_or_series(expected, index=index, dtype=float)

    # 计算滚动窗口内的和，结果存储在result中
    result = df.rolling("2D", closed=closed).sum()

    # 断言结果与预期结果相等
    tm.assert_equal(result, expected)
# 测试变量偏移窗口非唯一情况
def test_variable_offset_window_nonunique(closed, expected, frame_or_series):
    # GH 20712
    # 创建一个包含重复日期的时间索引
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    # 根据提供的索引创建数据帧或者系列
    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    # 创建一个工作日偏移量为2的变量偏移窗口索引
    offset = BusinessDay(2)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    # 对数据帧进行滚动计算，使用自定义的索引器和其他参数
    result = df.rolling(indexer, closed=closed, min_periods=1).sum()

    # 使用测试工具进行结果验证
    tm.assert_equal(result, expected)


# 测试偶数窗口对齐性
def test_even_number_window_alignment():
    # 参见GH 38780的讨论
    # 创建一个时间序列，频率为每天，持续三天
    s = Series(range(3), index=date_range(start="2020-01-01", freq="D", periods=3))

    # 使用日期为单位的滚动窗口，窗口大小为2天，最小观测期为1，居中对齐
    result = s.rolling(window="2D", min_periods=1, center=True).mean()

    # 创建预期的结果序列
    expected = Series([0.5, 1.5, 2], index=s.index)

    # 使用测试工具验证序列结果
    tm.assert_series_equal(result, expected)


# 测试固定二进制列的闭合运算
def test_closed_fixed_binary_col(center, step):
    # GH 34315
    # 创建一个包含二进制数据的数据帧，索引为每分钟，数据长度为指定步数
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame(
        {"binary_col": data},
        index=date_range(start="2020-01-01", freq="min", periods=len(data)),
    )

    # 根据指定的参数，计算滚动平均值
    rolling = df.rolling(
        window=len(df), closed="left", min_periods=1, center=center, step=step
    )
    result = rolling.mean()

    # 创建预期的数据帧，验证结果
    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]
    expected = DataFrame(
        expected_data,
        columns=["binary_col"],
        index=date_range(start="2020-01-01", freq="min", periods=len(expected_data)),
    )[::step]

    # 使用测试工具验证结果数据帧
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("closed", ["neither", "left"])
def test_closed_empty(closed, arithmetic_win_operators):
    # GH 26005
    # 根据指定的算术窗口操作名创建一个时间序列
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range("2000", periods=5, freq="2D"))
    roll = ser.rolling("1D", closed=closed)

    # 执行指定的滚动窗口操作，生成结果
    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)

    # 使用测试工具验证结果序列
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry(func):
    # GH24718
    # 创建一个仅包含一个数据条目的时间序列
    ser = Series(data=[2], index=date_range("2000", periods=1))
    result = getattr(ser.rolling("10D", closed="left"), func)()

    # 使用测试工具验证结果序列
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry_groupby(func):
    # GH24718
    # 创建一个包含多个列的数据帧，每列都包含多个数据条目
    ser = DataFrame(
        data={"A": [1, 1, 2], "B": [3, 2, 1]},
        index=date_range("2000", periods=3),
    )
    # 调用getattr函数，获取序列ser按"A"分组后的"B"列，执行"10D"滚动窗口操作（左闭右开）
    result = getattr(
        ser.groupby("A", sort=False)["B"].rolling("10D", closed="left"), func
    )()
    
    # 创建一个多级索引对象MultiIndex，其第一级索引由数组[1, 1, 2]构成，第二级索引使用序列ser的索引
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=("A", None))
    
    # 创建一个预期的Series对象，包含数据[NaN, 3, NaN]，使用上面创建的多级索引，列名为"B"
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name="B")
    
    # 使用测试模块tm（通常为pytest或pandas.testing）的assert_series_equal函数，比较result和expected两个Series对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("input_dtype", ["int", "float"])
@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "both", [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("min", "neither", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "left", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", "right", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "both", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "neither", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("max", "left", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_closed_min_max_datetime(input_dtype, func, closed, expected):
    """
    测试函数：test_closed_min_max_datetime
    测试用例：使用不同的数据类型和闭合方式对滚动窗口的最小值和最大值进行测试
    """

    # 创建一个时间序列，数据为0到9，索引为2000年起始的10个日期
    ser = Series(
        data=np.arange(10).astype(input_dtype),
        index=date_range("2000", periods=10),
    )

    # 调用滚动窗口函数，计算指定时间窗口内的最小值或最大值
    result = getattr(ser.rolling("3D", closed=closed), func)()

    # 创建预期结果的时间序列
    expected = Series(expected, index=ser.index)

    # 使用测试工具函数验证计算结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


def test_closed_uneven():
    """
    测试函数：test_closed_uneven
    测试用例：测试不均匀索引情况下的滚动窗口最小值计算
    """

    # 创建一个时间序列，数据为0到9，索引为2000年起始的10个日期
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # 删除索引为1和5的数据点，造成不均匀的时间序列
    ser = ser.drop(index=ser.index[[1, 5]])

    # 调用滚动窗口函数，计算指定时间窗口内的最小值
    result = ser.rolling("3D", closed="left").min()

    # 创建预期结果的时间序列
    expected = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)

    # 使用测试工具函数验证计算结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "both", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("min", "neither", [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "left", [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("max", "right", [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]),
        ("max", "both", [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]),
        ("max", "neither", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]),
        ("max", "left", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan]),
    ],
)
def test_closed_min_max_minp(func, closed, expected):
    """
    测试函数：test_closed_min_max_minp
    测试用例：使用不同的闭合方式对带有缺失值的滚动窗口的最小值和最大值进行测试
    """

    # 创建一个时间序列，数据为0到9，索引为2000年起始的10个日期
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # 将最后三个数据点设置为NaN，模拟数据中的缺失值
    ser[ser.index[-3:]] = np.nan

    # 调用滚动窗口函数，计算指定时间窗口内的最小值或最大值
    result = getattr(ser.rolling("3D", min_periods=2, closed=closed), func)()

    # 创建预期结果的时间序列
    expected = Series(expected, index=ser.index)

    # 使用测试工具函数验证计算结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("right", [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("both", [0, 0.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("neither", [np.nan, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("left", [np.nan, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_closed_median_quantile(closed, expected):
    """
    测试函数：test_closed_median_quantile
    测试用例：测试使用不同的闭合方式对中位数和分位数计算的滚动窗口操作
    """

    # 创建一个时间序列，数据为0到9，索引为2000年起始的10个日期
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # 创建滚动窗口对象，指定闭合方式
    roll = ser.rolling("3D", closed=closed)

    # 创建预期结果的时间序列
    expected = Series(expected, index=ser.index)
    # 计算滚动窗口的中位数，并将结果存储在变量 result 中
    result = roll.median()
    # 使用测试框架中的方法验证 result 是否等于预期结果 expected
    tm.assert_series_equal(result, expected)
    
    # 计算滚动窗口数据的分位数（50%），并将结果存储在变量 result 中
    result = roll.quantile(0.5)
    # 使用测试框架中的方法验证 result 是否等于预期结果 expected
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("roller", ["1s", 1])
# 使用 pytest.mark.parametrize 注释装饰器，为单元测试 tests_empty_df_rolling 提供参数化测试数据
def tests_empty_df_rolling(roller):
    # GH 15819 验证在空的 DataFrame 上应用日期和整数滚动窗口是否有效
    expected = DataFrame()
    # 创建一个空的 DataFrame，并应用滚动窗口求和操作
    result = DataFrame().rolling(roller).sum()
    # 使用 pytest 的 tm.assert_frame_equal 方法验证结果与期望值是否相等
    tm.assert_frame_equal(result, expected)

    # 验证在具有日期索引的空 DataFrame 上应用日期和整数滚动窗口是否有效
    expected = DataFrame(index=DatetimeIndex([]))
    # 创建具有空日期索引的 DataFrame，并应用滚动窗口求和操作
    result = DataFrame(index=DatetimeIndex([])).rolling(roller).sum()
    # 使用 pytest 的 tm.assert_frame_equal 方法验证结果与期望值是否相等
    tm.assert_frame_equal(result, expected)


def test_empty_window_median_quantile():
    # GH 26005
    # 创建一个期望结果包含 NaN 值的 Series
    expected = Series([np.nan, np.nan, np.nan])
    # 创建一个滚动窗口大小为 0 的 Series
    roll = Series(np.arange(3)).rolling(0)

    # 计算滚动窗口的中位数
    result = roll.median()
    # 使用 pytest 的 tm.assert_series_equal 方法验证中位数结果与期望值是否相等
    tm.assert_series_equal(result, expected)

    # 计算滚动窗口的分位数（这里使用了和中位数相同的期望结果）
    result = roll.quantile(0.1)
    # 使用 pytest 的 tm.assert_series_equal 方法验证分位数结果与期望值是否相等
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero():
    # https://github.com/pandas-dev/pandas/pull/18921
    # 在 minp=0 时，创建一个包含 NaN 的 Series
    x = Series([np.nan])
    # 使用滚动窗口大小为 1，并设置 min_periods=0 来计算滚动窗口的和
    result = x.rolling(1, min_periods=0).sum()
    # 创建一个期望结果为 [0.0] 的 Series
    expected = Series([0.0])
    # 使用 pytest 的 tm.assert_series_equal 方法验证结果与期望值是否相等
    tm.assert_series_equal(result, expected)

    # 在 minp=1 时，计算滚动窗口的和
    result = x.rolling(1, min_periods=1).sum()
    # 创建一个期望结果包含 NaN 的 Series
    expected = Series([np.nan])
    # 使用 pytest 的 tm.assert_series_equal 方法验证结果与期望值是否相等
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero_variable():
    # https://github.com/pandas-dev/pandas/pull/18921
    # 创建一个包含 NaN 值的 Series，索引为日期时间索引
    x = Series(
        [np.nan] * 4,
        index=DatetimeIndex(["2017-01-01", "2017-01-04", "2017-01-06", "2017-01-07"]),
    )
    # 使用 Timedelta("2D") 来定义滚动窗口的大小，并设置 min_periods=0 来计算滚动窗口的和
    result = x.rolling(Timedelta("2D"), min_periods=0).sum()
    # 创建一个期望结果为 0.0 的 Series，其索引与 x 相同
    expected = Series(0.0, index=x.index)
    # 使用 pytest 的 tm.assert_series_equal 方法验证结果与期望值是否相等
    tm.assert_series_equal(result, expected)


def test_multi_index_names():
    # GH 16789, 16825
    # 创建一个具有多级索引的 DataFrame
    cols = MultiIndex.from_product([["A", "B"], ["C", "D", "E"]], names=["1", "2"])
    df = DataFrame(np.ones((10, 6)), columns=cols)
    # 对 DataFrame 应用滚动窗口计算协方差
    result = df.rolling(3).cov()

    # 使用 pytest 的 tm.assert_index_equal 方法验证结果的列索引与原始 DataFrame 的列索引是否相等
    tm.assert_index_equal(result.columns, df.columns)
    # 使用 assert 语句验证结果的行索引的名称是否符合预期
    assert result.index.names == [None, "1", "2"]


def test_rolling_axis_sum():
    # see gh-23372.
    # 创建一个数值全为 1 的 DataFrame
    df = DataFrame(np.ones((10, 20)))
    # 创建一个期望结果为 {0: [np.nan] * 2 + [3.0] * 8} 的 DataFrame
    expected = DataFrame({i: [np.nan] * 2 + [3.0] * 8 for i in range(20)})
    # 对 DataFrame 应用滚动窗口计算和
    result = df.rolling(3).sum()
    # 使用 pytest 的 tm.assert_frame_equal 方法验证结果与期望值是否相等
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_count():
    # see gh-26055
    # 创建一个包含两列的 DataFrame
    df = DataFrame({"x": range(3), "y": range(3)})

    # 创建一个期望结果为 {"x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0]} 的 DataFrame
    expected = DataFrame({"x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0]})
    # 对 DataFrame 应用滚动窗口计算计数
    result = df.rolling(2, min_periods=0).count()
    # 使用 pytest 的 tm.assert_frame_equal 方法验证结果与期望值是否相等
    tm.assert_frame_equal(result, expected)


def test_readonly_array():
    # GH-27766
    # 创建一个包含不可写数组的 Series
    arr = np.array([1, 3, np.nan, 3, 5])
    arr.setflags(write=False)
    # 对 Series 应用滚动窗口计算均值
    result = Series(arr).rolling(2).mean()
    # 创建一个期望结果为 [np.nan, 2, np.nan, np.nan, 4] 的 Series
    expected = Series([np.nan, 2, np.nan, np.nan, 4])
    # 使用 pytest 的 tm.assert_series_equal 方法验证结果与期望值是否相等
    tm.assert_series_equal(result, expected)


def test_rolling_datetime(tz_naive_fixture):
    # GH-28192
    tz = tz_naive_fixture
    # 创建一个具有日期时间索引的 DataFrame
    df = DataFrame(
        {i: [1] * 2 for i in date_range("2019-8-01", "2019-08-03", freq="D", tz=tz)}
    )

    # 对 DataFrame 的转置应用滚动窗口计算和，再转置回来
    result = df.T.rolling("2D").sum().T
    # 创建预期的 DataFrame 对象，用于与结果进行比较
    expected = DataFrame(
        {
            **{
                # 使用 date_range 生成从 "2019-8-01" 开始的单日日期，每天值为 [1.0, 1.0]
                i: [1.0] * 2
                for i in date_range("2019-8-01", periods=1, freq="D", tz=tz)
            },
            **{
                # 使用 date_range 生成从 "2019-8-02" 到 "2019-8-03" 的日期范围，每天值为 [2.0, 2.0]
                i: [2.0] * 2
                for i in date_range("2019-8-02", "2019-8-03", freq="D", tz=tz)
            },
        }
    )
    # 使用测试工具 tm.assert_frame_equal 检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于验证滚动窗口作为字符串的效果
def test_rolling_window_as_string(center):
    # 见 GitHub issue #22590

    # 获取当前日期时间
    date_today = datetime.now()
    # 生成一个从当前日期到未来一年的日期范围，每天一个日期
    days = date_range(date_today, date_today + timedelta(365), freq="D")

    # 创建一个长度为日期数的全为1的 NumPy 数组
    data = np.ones(len(days))
    # 创建一个 DataFrame，包含 "DateCol" 列和 "metric" 列，其中 "DateCol" 列为日期，"metric" 列为全为1的数据
    df = DataFrame({"DateCol": days, "metric": data})

    # 将 DataFrame 的索引设为 "DateCol" 列
    df.set_index("DateCol", inplace=True)

    # 对 "metric" 列进行滚动操作，窗口大小为 "21D"，最小周期为2，左闭右开，中心为参数中的 center 变量
    result = df.rolling(window="21D", min_periods=2, closed="left", center=center)["metric"].agg("max")

    # 将日期范围对象重命名为 "DateCol"
    index = days.rename("DateCol")
    # 将日期范围对象的频率设为 None
    index = index._with_freq(None)

    # 创建一个与日期范围长度相同的全为1的浮点数数组，用于作为预期结果
    expected_data = np.ones(len(days), dtype=np.float64)
    # 如果 center 为 False，则将预期结果的前两个值设为 NaN
    if not center:
        expected_data[:2] = np.nan
    # 创建一个 Series 对象作为预期结果，索引为日期范围，数据为预期数据数组，列名为 "metric"
    expected = Series(expected_data, index=index, name="metric")

    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证滚动窗口最小周期为1的效果
def test_min_periods1():
    # GitHub issue #6795

    # 创建一个包含整数的 DataFrame，列名为 "a"
    df = DataFrame([0, 1, 2, 1, 0], columns=["a"])
    # 对 "a" 列进行滚动操作，窗口大小为3，中心为 True，最小周期为1，计算最大值
    result = df["a"].rolling(3, center=True, min_periods=1).max()

    # 创建一个预期结果的 Series 对象，包含预期的最大值结果
    expected = Series([1.0, 2.0, 2.0, 2.0, 1.0], name="a")

    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证滚动计数并设置最小周期为3的效果
def test_rolling_count_with_min_periods(frame_or_series):
    # GitHub issue #26996

    # 对输入的 DataFrame 或 Series 对象进行滚动计数，窗口大小为3，最小周期为3
    result = frame_or_series(range(5)).rolling(3, min_periods=3).count()

    # 创建一个包含预期结果的 Series 对象，前两个值为 NaN，其余为3
    expected = frame_or_series([np.nan, np.nan, 3.0, 3.0, 3.0])

    # 断言结果和预期结果相等
    tm.assert_equal(result, expected)


# 定义一个测试函数，用于验证滚动计数并设置默认最小周期为0时的效果，包括处理空值
def test_rolling_count_default_min_periods_with_null_values(frame_or_series):
    # GitHub issue #26996

    # 创建一个包含整数和 NaN 值的列表
    values = [1, 2, 3, np.nan, 4, 5, 6]
    # 创建一个包含预期计数结果的列表
    expected_counts = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0]

    # GitHub issue #31302
    # 对输入的 DataFrame 或 Series 对象进行滚动计数，窗口大小为3，最小周期为0
    result = frame_or_series(values).rolling(3, min_periods=0).count()

    # 创建一个包含预期计数结果的 Series 对象
    expected = frame_or_series(expected_counts)

    # 断言结果和预期结果相等
    tm.assert_equal(result, expected)
    [
        # 第一个元组
        (
            # 第一个字典
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            # 第一个列表
            [
                # 第一个元组
                ({"A": [1], "B": [4]}, [0]),
                # 第二个元组
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                # 第三个元组
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            # 第一个整数
            3,
            # 第一个空值
            None,
        ),
        # 第二个元组
        (
            # 第二个字典
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            # 第二个列表
            [
                # 第一个元组
                ({"A": [1], "B": [4]}, [0]),
                # 第二个元组
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                # 第三个元组
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            # 第二个整数
            2,
            # 第二个整数
            1,
        ),
        # 第三个元组
        (
            # 第三个字典
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            # 第三个列表
            [
                # 第一个元组
                ({"A": [1], "B": [4]}, [0]),
                # 第二个元组
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                # 第三个元组
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            # 第三个整数
            2,
            # 第三个整数
            2,
        ),
        # 第四个元组
        (
            # 第四个字典
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            # 第四个列表
            [
                # 第一个元组
                ({"A": [1], "B": [4]}, [0]),
                # 第二个元组
                ({"A": [2], "B": [5]}, [1]),
                # 第三个元组
                ({"A": [3], "B": [6]}, [2]),
            ],
            # 第四个整数
            1,
            # 第四个整数
            1,
        ),
        # 第五个元组
        (
            # 第五个字典
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            # 第五个列表
            [
                # 第一个元组
                ({"A": [1], "B": [4]}, [0]),
                # 第二个元组
                ({"A": [2], "B": [5]}, [1]),
                # 第三个元组
                ({"A": [3], "B": [6]}, [2]),
            ],
            # 第五个整数
            1,
            # 第五个整数
            0,
        ),
        # 第六个元组
        ({"A": [1], "B": [4]}, [], 2, None),
        # 第七个元组
        ({"A": [1], "B": [4]}, [], 2, 1),
        # 第八个元组
        (None, [({}, [])], 2, None),
        # 第九个元组
        (
            # 第九个字典
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            # 第九个列表
            [
                # 第一个元组
                ({"A": [1.0], "B": [np.nan]}, [0]),
                # 第二个元组
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                # 第三个元组
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            # 第九个整数
            3,
            # 第九个整数
            2,
        ),
    ],
# 定义测试函数，用于迭代测试 DataFrame 的滚动窗口功能
def test_iter_rolling_dataframe(df, expected, window, min_periods):
    # GH 11704
    # 将输入的 df 转换为 DataFrame 对象
    df = DataFrame(df)
    # 将期望的结果转换为 DataFrame 对象的列表，每个元组(values, index)都转换为一个 DataFrame
    expecteds = [DataFrame(values, index=index) for (values, index) in expected]

    # 使用 zip 函数同时迭代 expecteds 和 df 的滚动窗口对象
    for expected, actual in zip(expecteds, df.rolling(window, min_periods=min_periods)):
        # 断言实际的滚动窗口对象与期望的 DataFrame 相等
        tm.assert_frame_equal(actual, expected)


# 使用 pytest 的 parametrize 装饰器定义多个测试参数
@pytest.mark.parametrize(
    "expected,window",
    [
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            "2D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            "3D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            "1D",
        ),
    ],
)
# 定义测试函数，用于迭代测试 DataFrame 的滚动窗口功能（按日期列 "C" 进行滚动）
def test_iter_rolling_on_dataframe(expected, window):
    # GH 11704, 40373
    # 创建一个包含多列数据的 DataFrame 对象
    df = DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [4, 5, 6, 7, 8],
            "C": date_range(start="2016-01-01", periods=5, freq="D"),
        }
    )

    # 将期望的结果转换为 DataFrame 对象的列表，每个元组(values, index)都转换为一个 DataFrame
    expecteds = [
        DataFrame(values, index=df.loc[index, "C"]) for (values, index) in expected
    ]
    # 使用 zip 函数同时迭代 expecteds 和按日期列 "C" 进行滚动的 df 对象
    for expected, actual in zip(expecteds, df.rolling(window, on="C")):
        # 断言实际的滚动窗口对象与期望的 DataFrame 相等
        tm.assert_frame_equal(actual, expected)


# 定义测试函数，用于测试 DataFrame 按列 "a" 分组后的滚动窗口功能
def test_iter_rolling_on_dataframe_unordered():
    # GH 43386
    # 创建一个包含两列数据的 DataFrame 对象
    df = DataFrame({"a": ["x", "y", "x"], "b": [0, 1, 2]})
    # 对列 "a" 进行分组并应用滚动窗口操作，将结果存储在列表中
    results = list(df.groupby("a").rolling(2))
    # 期望的结果，即分组后的 DataFrame 列表
    expecteds = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    # 使用 zip 函数同时迭代 results 和 expecteds
    for result, expected in zip(results, expecteds):
        # 断言实际的结果与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器定义多个测试参数
@pytest.mark.parametrize(
    "ser,expected,window, min_periods",
    [
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            None,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            2,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([2], [1]), ([3], [2])],
            1,
            0,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([2], [1]), ([3], [2])],
            1,
            1,
        ),
        (
            Series([1, 2]),
            [([1], [0]), ([1, 2], [0, 1])],
            2,
            0,
        ),
        (
            Series([], dtype="int64"),
            [],
            2,
            1,
        ),
    ],
)
# 定义测试函数，用于迭代测试 Series 的滚动窗口功能
def test_iter_rolling_series(ser, expected, window, min_periods):
    # GH 11704
    # 将期望的结果转换为 Series 对象的列表，每个元组(values, index)都转换为一个 Series
    expecteds = [Series(values, index=index) for (values, index) in expected]
    # 使用 zip 函数将 expecteds 和 ser.rolling(window, min_periods=min_periods) 这两个可迭代对象逐个配对
    for expected, actual in zip(
        expecteds, ser.rolling(window, min_periods=min_periods)
    ):
        # 调用 tm.assert_series_equal 函数比较 actual 和 expected 两个序列是否相等
        tm.assert_series_equal(actual, expected)
@pytest.mark.parametrize(
    "expected,expected_index,window",
    [  # 参数化测试，定义了三个参数：expected, expected_index, window
        (
            [[0], [1], [2], [3], [4]],  # 第一个测试用例的预期输出
            [
                date_range("2020-01-01", periods=1, freq="D"),  # 第一个测试用例的日期范围
                date_range("2020-01-02", periods=1, freq="D"),
                date_range("2020-01-03", periods=1, freq="D"),
                date_range("2020-01-04", periods=1, freq="D"),
                date_range("2020-01-05", periods=1, freq="D"),
            ],
            "1D",  # 第一个测试用例的滚动窗口大小
        ),
        (
            [[0], [0, 1], [1, 2], [2, 3], [3, 4]],  # 第二个测试用例的预期输出
            [
                date_range("2020-01-01", periods=1, freq="D"),  # 第二个测试用例的日期范围
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-02", periods=2, freq="D"),
                date_range("2020-01-03", periods=2, freq="D"),
                date_range("2020-01-04", periods=2, freq="D"),
            ],
            "2D",  # 第二个测试用例的滚动窗口大小
        ),
        (
            [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],  # 第三个测试用例的预期输出
            [
                date_range("2020-01-01", periods=1, freq="D"),  # 第三个测试用例的日期范围
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-01", periods=3, freq="D"),
                date_range("2020-01-02", periods=3, freq="D"),
                date_range("2020-01-03", periods=3, freq="D"),
            ],
            "3D",  # 第三个测试用例的滚动窗口大小
        ),
    ],
)
def test_iter_rolling_datetime(expected, expected_index, window):
    # GH 11704
    ser = Series(range(5), index=date_range(start="2020-01-01", periods=5, freq="D"))

    expecteds = [
        Series(values, index=idx) for (values, idx) in zip(expected, expected_index)
    ]

    for expected, actual in zip(expecteds, ser.rolling(window)):
        tm.assert_series_equal(actual, expected)
        # 断言每个滚动窗口计算的实际结果与预期结果是否相等
        

@pytest.mark.parametrize(
    "grouping,_index",
    [  # 参数化测试，定义了两个参数：grouping, _index
        (
            {"level": 0},  # 第一个测试用例的分组条件
            MultiIndex.from_tuples(
                [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None]
            ),  # 第一个测试用例的预期索引
        ),
        (
            {"by": "X"},  # 第二个测试用例的分组条件
            MultiIndex.from_tuples(
                [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=["X", None]
            ),  # 第二个测试用例的预期索引
        ),
    ],
)
def test_rolling_positional_argument(grouping, _index, raw):
    # GH 34605
    # 测试滚动函数的位置参数功能

    def scaled_sum(*args):
        if len(args) < 2:
            raise ValueError("The function needs two arguments")
        array, scale = args
        return array.sum() / scale

    df = DataFrame(data={"X": range(5)}, index=[0, 0, 1, 1, 1])

    expected = DataFrame(data={"X": [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    # GH 40341
    if "by" in grouping:
        expected = expected.drop(columns="X", errors="ignore")
    result = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)
    # 断言滚动函数应用后的结果与预期是否相等


@pytest.mark.parametrize("add", [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add, unit):
    # GH: 36031 implementing kahan summation
    # 测试滚动函数的数值精度，实现卡汉求和算法
    pass
    # 该测试函数目前没有实现具体的测试内容，因此留空
    # 创建 DatetimeIndex 对象，包含指定的时间戳列表，并转换为指定时间单位
    dti = DatetimeIndex(
        [
            Timestamp("19700101 09:00:00"),
            Timestamp("19700101 09:00:03"),
            Timestamp("19700101 09:00:06"),
        ]
    ).as_unit(unit)
    # 创建 DataFrame 对象，包含一列名为 "A" 的数据列，索引为 dti 变量指定的时间戳
    df = DataFrame(
        {"A": [3002399751580331.0 + add, -0.0, -0.0]},
        index=dti,
    )
    # 对 DataFrame 进行重采样，按每秒("1s")进行前向填充(ffill)，再以"3s"为窗口进行左对齐滚动平均
    result = (
        df.resample("1s").ffill().rolling("3s", closed="left", min_periods=3).mean()
    )
    # 创建日期范围，从指定的日期开始，以秒为频率，生成包含指定周期数的日期时间索引
    dates = date_range("19700101 09:00:00", periods=7, freq="s", unit=unit)
    # 创建预期的 DataFrame 对象，包含一列名为 "A" 的数据列，索引为 dates 变量指定的日期时间
    expected = DataFrame(
        {
            "A": [
                np.nan,
                np.nan,
                np.nan,
                3002399751580330.5,
                2001599834386887.25,
                1000799917193443.625,
                0.0,
            ]
        },
        index=dates,
    )
    # 使用 assert_frame_equal 检查 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试滚动数值的精度，特别是Kahan求和算法
def test_rolling_numerical_accuracy_kahan_sum():
    # GH: 13254
    # 创建一个DataFrame，包含一列数值
    df = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=["x"])
    # 对列"x"进行滚动窗口为3的求和操作
    result = df["x"].rolling(3).sum()
    # 期望的结果Series
    expected = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name="x")
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试滚动数值的精度，特别是时间序列的跳跃
def test_rolling_numerical_accuracy_jump():
    # GH: 32761
    # 创建一个时间索引，包含一个额外的时间点，和对应长度的随机数据
    index = date_range(start="2020-01-01", end="2020-01-02", freq="60s").append(
        DatetimeIndex(["2020-01-03"])
    )
    data = np.random.default_rng(2).random(len(index))

    # 创建一个DataFrame，包含随机数据，以时间索引
    df = DataFrame({"data": data}, index=index)
    # 对"data"列进行滚动窗口为"60s"的均值计算
    result = df.rolling("60s").mean()
    # 断言结果DataFrame与原始数据的"data"列相等
    tm.assert_frame_equal(result, df[["data"]])


# 定义一个测试函数，用于测试滚动数值的精度，特别是小数值的处理
def test_rolling_numerical_accuracy_small_values():
    # GH: 10319
    # 创建一个时间序列，包含一些小数值和对应时间索引
    s = Series(
        data=[0.00012456, 0.0003, -0.0, -0.0],
        index=date_range("1999-02-03", "1999-02-06"),
    )
    # 对时间序列进行滚动窗口为1的均值计算
    result = s.rolling(1).mean()
    # 断言结果与原始时间序列相等
    tm.assert_series_equal(result, s)


# 定义一个测试函数，用于测试滚动数值的精度，特别是处理过大数值
def test_rolling_numerical_too_large_numbers():
    # GH: 11645
    # 创建一个时间序列，包含一些浮点数值和对应时间索引
    dates = date_range("2015-01-01", periods=10, freq="D")
    ds = Series(data=range(10), index=dates, dtype=np.float64)
    # 修改第三个索引处的数值为一个极大的负数
    ds.iloc[2] = -9e33
    # 对时间序列进行滚动窗口为5的均值计算
    result = ds.rolling(5).mean()
    # 期望的结果Series
    expected = Series(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.8e33,
            -1.8e33,
            -1.8e33,
            5.0,
            6.0,
            7.0,
        ],
        index=dates,
    )
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试装饰器，定义一个测试函数，用于测试滚动窗口在时间索引上的函数计算
@pytest.mark.parametrize(
    ("index", "window"),
    [
        (
            period_range(start="2020-01-01 08:00", end="2020-01-01 08:08", freq="min"),
            "2min",
        ),
        (
            period_range(
                start="2020-01-01 08:00", end="2020-01-01 12:00", freq="30min"
            ),
            "1h",
        ),
    ],
)
@pytest.mark.parametrize(
    ("func", "values"),
    [
        ("min", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("sum", [np.nan, 0, 1, 3, 5, 7, 9, 11, 13]),
    ],
)
# 定义一个测试函数，用于测试在时间索引上的滚动窗口函数计算
def test_rolling_period_index(index, window, func, values):
    # GH: 34225
    # 创建一个时间序列，以指定的时间索引和数据值
    ds = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    # 调用指定窗口大小的滚动函数，如最小、最大或求和
    result = getattr(ds.rolling(window, closed="left"), func)()
    # 期望的结果Series
    expected = Series(values, index=index)
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试滚动窗口的标准误差计算
def test_rolling_sem(frame_or_series):
    # GH: 26476
    # 从参数中获取一个DataFrame或Series对象
    obj = frame_or_series([0, 1, 2])
    # 对对象进行滚动窗口为2的标准误差计算，最小观测数为1
    result = obj.rolling(2, min_periods=1).sem()
    # 如果结果是一个DataFrame，则转换为Series，取第一列的值
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    # 期望的结果Series
    expected = Series([np.nan] + [0.7071067811865476] * 2)
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 使用xfail装饰器，定义一个参数化测试函数，用于测试在特定平台上的失败情况
@pytest.mark.xfail(
    is_platform_arm() or is_platform_power() or is_platform_riscv64(),
    reason="GH 38921",
)
# 定义一个参数化测试函数，用于测试在特定函数下的第三个数值和结果值
@pytest.mark.parametrize(
    ("func", "third_value", "values"),
    # 创建一个包含多个元组的列表，每个元组包含标识符、版本号和一组数字
    [
        # 第一个元组，标识符为 "var"，版本号为 1，包含以下数字
        ("var", 1, [5e33, 0, 0.5, 0.5, 2, 0]),
        # 第二个元组，标识符为 "std"，版本号为 1，包含以下数字
        ("std", 1, [7.071068e16, 0, 0.7071068, 0.7071068, 1.414214, 0]),
        # 第三个元组，标识符为 "var"，版本号为 2，包含以下数字
        ("var", 2, [5e33, 0.5, 0, 0.5, 2, 0]),
        # 第四个元组，标识符为 "std"，版本号为 2，包含以下数字
        ("std", 2, [7.071068e16, 0.7071068, 0, 0.7071068, 1.414214, 0]),
    ],
# GH: 37051
def test_rolling_var_numerical_issues(func, third_value, values):
    # 创建一个包含大整数的序列，用于测试滚动计算中的数值问题
    ds = Series([99999999999999999, 1, third_value, 2, 3, 1, 1])
    # 对序列应用滚动窗口为2的函数，并保存结果
    result = getattr(ds.rolling(2), func)()
    # 期望的结果序列，用于验证滚动计算的正确性
    expected = Series([np.nan] + values)
    # 断言结果序列与期望序列相等
    tm.assert_series_equal(result, expected)
    # GH 42064
    # 新的 `roll_var` 现在能正确输出0.0
    # 用于验证新功能的断言，确保输出正确处理为0.0
    tm.assert_series_equal(result == 0, expected == 0)


def test_timeoffset_as_window_parameter_for_corr(unit):
    # GH: 28266
    # 创建一个日期时间索引，以指定的单位为时间偏移，用于测试相关系数计算
    dti = DatetimeIndex(
        [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130102 09:00:02"),
            Timestamp("20130103 09:00:03"),
            Timestamp("20130105 09:00:05"),
            Timestamp("20130106 09:00:06"),
        ]
    ).as_unit(unit)
    # 创建一个多级索引，用于测试相关系数计算
    mi = MultiIndex.from_product([dti, ["B", "A"]])

    # 期望的相关系数计算结果数据框
    exp = DataFrame(
        {
            "B": [
                np.nan,
                np.nan,
                0.9999999999999998,
                -1.0,
                1.0,
                -0.3273268353539892,
                0.9999999999999998,
                1.0,
                0.9999999999999998,
                1.0,
            ],
            "A": [
                np.nan,
                np.nan,
                -1.0,
                1.0000000000000002,
                -0.3273268353539892,
                0.9999999999999966,
                1.0,
                1.0000000000000002,
                1.0,
                1.0000000000000002,
            ],
        },
        index=mi,
    )

    # 创建一个数据框，并对其应用3天滚动窗口的相关系数计算
    df = DataFrame(
        {"B": [0, 1, 2, 4, 3], "A": [7, 4, 6, 9, 3]},
        index=dti,
    )

    # 断言计算出的相关系数数据框与期望的数据框相等
    res = df.rolling(window="3d").corr()
    tm.assert_frame_equal(exp, res)


@pytest.mark.parametrize("method", ["var", "sum", "mean", "skew", "kurt", "min", "max"])
def test_rolling_decreasing_indices(method):
    """
    确保递减索引与递增索引给出相同结果。

    GH 36933
    """
    # 创建一个数据框，包含递减索引的平方值
    df = DataFrame({"values": np.arange(-15, 10) ** 2})
    # 创建一个逆序的数据框，用于测试递减索引是否能给出相同结果
    df_reverse = DataFrame({"values": df["values"][::-1]}, index=df.index[::-1])

    # 对递增索引的数据框应用指定方法的滚动窗口计算
    increasing = getattr(df.rolling(window=5), method)()
    # 对递减索引的数据框应用指定方法的滚动窗口计算
    decreasing = getattr(df_reverse.rolling(window=5), method)()

    # 断言递减索引结果的反转与递增索引结果的对应部分之差小于指定阈值
    assert np.abs(decreasing.values[::-1][:-4] - increasing.values[4:]).max() < 1e-12


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2s", "right", [1.0, 3.0, 5.0, 3.0]),
        ("2s", "left", [0.0, 1.0, 3.0, 5.0]),
        ("2s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("2s", "neither", [0.0, 1.0, 2.0, 3.0]),
        ("3s", "right", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "left", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "neither", [1.0, 3.0, 6.0, 5.0]),
    ],
)
def test_rolling_decreasing_indices_centered(window, closed, expected, frame_or_series):
    """
    确保对称反转的索引返回与非反转索引相同的结果。
    """
    # GH 43927

    # 创建一个时间序列索引，以秒为单位，包含4个时间点
    index = date_range("2020", periods=4, freq="1s")
    # 创建一个 DataFrame 或 Series 对象，表示上升阶段的数据，索引为给定的 index
    df_inc = frame_or_series(range(4), index=index)
    
    # 创建一个 DataFrame 或 Series 对象，表示下降阶段的数据，索引为给定的 index 的逆序
    df_dec = frame_or_series(range(4), index=index[::-1])

    # 创建一个 DataFrame 或 Series 对象，表示预期上升阶段的数据，索引为给定的 index
    expected_inc = frame_or_series(expected, index=index)
    
    # 创建一个 DataFrame 或 Series 对象，表示预期下降阶段的数据，索引为给定的 index 的逆序
    expected_dec = frame_or_series(expected, index=index[::-1])

    # 对 df_inc 进行滚动窗口计算，窗口大小为 window，计算结果是每个窗口的和，窗口位于数据的中心，边界为 closed
    result_inc = df_inc.rolling(window, closed=closed, center=True).sum()
    
    # 对 df_dec 进行滚动窗口计算，窗口大小为 window，计算结果是每个窗口的和，窗口位于数据的中心，边界为 closed
    result_dec = df_dec.rolling(window, closed=closed, center=True).sum()

    # 使用测试框架中的 assert_equal 函数检查 result_inc 是否等于 expected_inc
    tm.assert_equal(result_inc, expected_inc)
    
    # 使用测试框架中的 assert_equal 函数检查 result_dec 是否等于 expected_dec
    tm.assert_equal(result_dec, expected_dec)
@pytest.mark.parametrize(
    "window,expected",
    [
        ("1ns", [1.0, 1.0, 1.0, 1.0]),
        ("3ns", [2.0, 3.0, 3.0, 2.0]),
    ],
)
def test_rolling_center_nanosecond_resolution(
    window, closed, expected, frame_or_series
):
    # 创建一个包含4个时间点的时间索引，频率为每纳秒1个时间点
    index = date_range("2020", periods=4, freq="1ns")
    # 使用给定的时间索引创建一个DataFrame或Series，初始值均为1.0，数据类型为浮点数
    df = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    # 期望的结果DataFrame或Series，与输入的时间索引相匹配，初始值根据测试用例中的预期值确定
    expected = frame_or_series(expected, index=index, dtype=float)
    # 对DataFrame或Series执行滚动窗口计算，窗口大小为window参数指定的值，求和操作
    result = df.rolling(window, closed=closed, center=True).sum()
    # 使用测试工具库检查计算结果与期望结果是否相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            "var",
            [
                float("nan"),
                43.0,
                float("nan"),
                136.333333,
                43.5,
                94.966667,
                182.0,
                318.0,
            ],
        ),
        (
            "mean",
            [float("nan"), 7.5, float("nan"), 21.5, 6.0, 9.166667, 13.0, 17.5],
        ),
        (
            "sum",
            [float("nan"), 30.0, float("nan"), 86.0, 30.0, 55.0, 91.0, 140.0],
        ),
        (
            "skew",
            [
                float("nan"),
                0.709296,
                float("nan"),
                0.407073,
                0.984656,
                0.919184,
                0.874674,
                0.842418,
            ],
        ),
        (
            "kurt",
            [
                float("nan"),
                -0.5916711736073559,
                float("nan"),
                -1.0028993131317954,
                -0.06103844629409494,
                -0.254143227116194,
                -0.37362637362637585,
                -0.45439658241367054,
            ],
        ),
    ],
)
def test_rolling_non_monotonic(method, expected):
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    # 创建一个包含整数平方值的DataFrame，用于测试
    use_expanding = [True, False, True, False, True, True, True, True]
    df = DataFrame({"values": np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    # 使用自定义索引器创建一个索引对象，窗口大小为4，使用的扩展模式由use_expanding参数指定
    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)

    # 执行DataFrame的滚动操作，选择对应于method参数指定的统计方法，例如var、mean、sum等
    result = getattr(df.rolling(indexer), method)()
    # 创建一个期望的DataFrame，包含预期的计算结果，用于与实际结果进行比较
    expected = DataFrame({"values": expected})
    # 使用测试工具库检查计算结果DataFrame与期望DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    # 创建包含两个元组的列表，每个元组包含一个列表和一个整数或字符串
    [
        # 第一个元组包含列表 [0, 1, 2, 3, 4] 和整数 2
        ([0, 1, 2, 3, 4], 2),
        # 第二个元组使用 date_range 函数生成的日期范围作为列表，以及字符串 "2D"
        (date_range("2001-01-01", freq="D", periods=5), "2D"),
    ],
# 定义测试函数，计算时间序列滚动相关系数，用于 GH 编号 31286
def test_rolling_corr_timedelta_index(index, window):
    # 创建时间序列 x 和 y，其中 x 的前两个值设为 0
    x = Series([1, 2, 3, 4, 5], index=index)
    y = x.copy()
    x.iloc[0:2] = 0.0
    # 计算滚动窗口为 window 的相关系数
    result = x.rolling(window).corr(y)
    # 期望的相关系数序列
    expected = Series([np.nan, np.nan, 1, 1, 1], index=index)
    # 断言结果与期望相近
    tm.assert_almost_equal(result, expected)


# 定义测试函数，处理包含 NaN 的分组滚动均值计算，用于 GH 编号 35542
def test_groupby_rolling_nan_included():
    # 构造包含 NaN 的数据框
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = DataFrame(data)
    # 对 group 列进行分组，计算滚动窗口为 1 的均值，保留 NaN
    result = df.groupby("group", dropna=False).rolling(1, min_periods=1).mean()
    # 期望的结果数据框
    expected = DataFrame(
        {"B": [0.0, 2.0, 3.0, 1.0, 4.0]},
        # 从元组创建 MultiIndex，NaN 被放在 levels 而非 codes，当前结果预期在这个情况下
        index=MultiIndex(
            [["g1", "g2", np.nan], [0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]],
            names=["group", None],
        ),
    )
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，验证滚动偏度和峰度的数值稳定性，用于 GH 编号 6929
@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_rolling_skew_kurt_numerical_stability(method):
    # 创建随机数序列
    ser = Series(np.random.default_rng(2).random(10))
    ser_copy = ser.copy()
    # 获取滚动窗口为 3 的偏度或峰度结果
    expected = getattr(ser.rolling(3), method)()
    # 断言前后序列相等
    tm.assert_series_equal(ser, ser_copy)
    # 将序列整体增加 50000
    ser = ser + 50000
    # 获取增加后的滚动窗口为 3 的偏度或峰度结果
    result = getattr(ser.rolling(3), method)()
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，验证滚动偏度和峰度对大数值范围的稳定性，用于 GH 编号 37557
@pytest.mark.parametrize(
    ("method", "values"),
    [
        ("skew", [2.0, 0.854563, 0.0, 1.999984]),
        ("kurt", [4.0, -1.289256, -1.2, 3.999946]),
    ],
)
def test_rolling_skew_kurt_large_value_range(method, values):
    # 创建包含大数值范围的序列
    s = Series([3000000, 1, 1, 2, 3, 4, 999])
    # 获取滚动窗口为 4 的偏度或峰度结果
    result = getattr(s.rolling(4), method)()
    # 期望的偏度或峰度结果序列
    expected = Series([np.nan] * 3 + values)
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，验证滚动操作中的无效方法抛出异常，用于 GH 编号 37051
def test_invalid_method():
    # 断言滚动窗口为 1 时使用无效方法会抛出 ValueError 异常
    with pytest.raises(ValueError, match="method must be 'table' or 'single"):
        Series(range(1)).rolling(1, method="foo")


# 定义测试函数，验证逆序时间序列的滚动和日期偏移计算，用于 GH 编号 40002
@pytest.mark.parametrize("window", [1, "1d"])
def test_rolling_descending_date_order_with_offset(window, frame_or_series):
    # 创建日期范围为 2020-01-01 至 2020-01-03 的时间索引
    idx = date_range(start="2020-01-01", end="2020-01-03", freq="1d")
    obj = frame_or_series(range(1, 4), index=idx)
    # 计算滚动窗口为 1 天的左闭合求和
    result = obj.rolling("1d", closed="left").sum()
    # 期望的求和结果序列
    expected = frame_or_series([np.nan, 1, 2], index=idx)
    # 断言结果与期望相等
    tm.assert_equal(result, expected)

    # 对逆序时间序列进行相同操作
    result = obj.iloc[::-1].rolling("1d", closed="left").sum()
    # 创建逆序日期索引，从 2020-01-03 到 2020-01-01
    idx = date_range(start="2020-01-03", end="2020-01-01", freq="-1d")
    # 期望的逆序求和结果序列
    expected = frame_or_series([np.nan, 3, 2], index=idx)
    # 断言结果与期望相等
    tm.assert_equal(result, expected)


# 定义测试函数，验证滚动方差在浮点数精度上的异常情况，用于 GH 编号 37051
def test_rolling_var_floating_artifact_precision():
    # 创建包含整数 7, 5, 5, 5 的序列
    s = Series([7, 5, 5, 5])
    # 计算滚动窗口为 3 的方差
    result = s.rolling(3).var()
    # 期望的方差结果序列
    expected = Series([np.nan, np.nan, 4 / 3, 0])
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)
    # 使用测试工具函数比较结果和期望值的 Series 对象是否近似相等
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)
    # 标识 GitHub 问题号为 42064
    # 当结果为 0 时，确保新的 `roll_var` 正确输出 0.0
    tm.assert_series_equal(result == 0, expected == 0)
def test_rolling_std_small_values():
    # GH 37051
    # 创建一个包含三个浮点数的 Series 对象
    s = Series(
        [
            0.00000054,
            0.00000053,
            0.00000054,
        ]
    )
    # 计算滚动窗口大小为2的标准差
    result = s.rolling(2).std()
    # 预期的结果
    expected = Series([np.nan, 7.071068e-9, 7.071068e-9])
    # 使用近似误差比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)


@pytest.mark.parametrize(
    "start, exp_values",
    [
        (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
        (2, [0.001, 0.001, 0.0015, 0.00366666]),
    ],
)
def test_rolling_mean_all_nan_window_floating_artifacts(start, exp_values):
    # GH#41053
    # 创建包含多个浮点数和 NaN 值的 DataFrame 对象
    df = DataFrame(
        [
            0.03,
            0.03,
            0.001,
            np.nan,
            0.002,
            0.008,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.005,
            0.2,
        ]
    )

    # 预期的浮点数值和 NaN 值序列
    values = exp_values + [
        0.00366666,
        0.005,
        0.005,
        0.008,
        np.nan,
        np.nan,
        0.005,
        0.102500,
    ]
    # 创建预期的 DataFrame 对象
    expected = DataFrame(
        values,
        index=list(range(start, len(values) + start)),
    )
    # 计算滚动窗口大小为5的均值
    result = df.iloc[start:].rolling(5, min_periods=0).mean()
    # 使用近似误差比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)


def test_rolling_sum_all_nan_window_floating_artifacts():
    # GH#41053
    # 创建包含多个浮点数和 NaN 值的 DataFrame 对象
    df = DataFrame([0.002, 0.008, 0.005, np.nan, np.nan, np.nan])
    # 计算滚动窗口大小为3的和
    result = df.rolling(3, min_periods=0).sum()
    # 预期的结果
    expected = DataFrame([0.002, 0.010, 0.015, 0.013, 0.005, 0.0])
    # 使用近似误差比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)


def test_rolling_zero_window():
    # GH 22719
    # 创建一个包含单个整数的 Series 对象
    s = Series(range(1))
    # 计算滚动窗口大小为0的最小值
    result = s.rolling(0).min()
    # 预期的结果
    expected = Series([np.nan])
    # 使用近似误差比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("method", ["min", "max", "average"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans"])
def test_rank(window, method, pct, ascending, test_data):
    length = 20
    if test_data == "default":
        # 创建一个包含随机浮点数的 Series 对象
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        # 创建一个包含随机整数的 Series 对象
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        # 创建一个包含随机浮点数、NaN 和无穷大值的 Series 对象
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )

    # 创建预期的结果，使用滚动窗口大小和指定的方法和参数计算排名
    expected = ser.rolling(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    # 计算滚动窗口大小和指定的方法和参数的排名
    result = ser.rolling(window).rank(method=method, pct=pct, ascending=ascending)

    # 使用近似误差比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_rolling_quantile_np_percentile():
    # #9413: Tests that rolling window's quantile default behavior
    # is analogous to Numpy's percentile
    # 设置行、列和索引的数量
    row = 10
    col = 5
    idx = date_range("20100101", periods=row, freq="B")
    # 使用 NumPy 的随机数生成器生成一个 row 行 col 列的随机数 DataFrame，并以 idx 作为索引
    df = DataFrame(
        np.random.default_rng(2).random(row * col).reshape((row, -1)), index=idx
    )
    
    # 计算 DataFrame 按列的四分位数，返回一个包含 0.25、0.5、0.75 分位数值的 DataFrame
    df_quantile = df.quantile([0.25, 0.5, 0.75], axis=0)
    
    # 使用 NumPy 计算 DataFrame 的按列百分位数，返回一个包含 25%、50%、75% 分位数值的数组
    np_percentile = np.percentile(df, [25, 50, 75], axis=0)
    
    # 使用测试工具包中的函数验证 df_quantile 的值与 np_percentile 数组的近似相等性
    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))
@pytest.mark.parametrize("quantile", [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
        [0.0, np.nan, 0.2, np.nan, 0.4],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5],
        [0.5],
        [np.nan, 0.7, 0.6],
    ],
)
def test_rolling_quantile_interpolation_options(quantile, interpolation, data):
    # Tests that rolling window's quantile behavior is analogous to
    # Series' quantile for each interpolation option

    # 将数据转换为 Series 对象
    s = Series(data)

    # 计算使用指定分位数和插值方法的 Series 的分位数
    q1 = s.quantile(quantile, interpolation)

    # 计算使用扩展窗口的 Series 的分位数，并取最后一个值
    q2 = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]

    # 检查是否存在 NaN 值，并进行相应断言
    if np.isnan(q1):
        assert np.isnan(q2)
    elif not IS64:
        # 在 32 位系统上精度较低
        assert np.allclose([q1], [q2], rtol=1e-07, atol=0)
    else:
        assert q1 == q2


def test_invalid_quantile_value():
    # 准备测试数据
    data = np.arange(5)
    s = Series(data)

    # 准备错误消息
    msg = "Interpolation 'invalid' is not supported"

    # 断言当使用不支持的插值方法时，会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        s.rolling(len(data), min_periods=1).quantile(0.5, interpolation="invalid")


def test_rolling_quantile_param():
    # 准备测试数据
    ser = Series([0.0, 0.1, 0.5, 0.9, 1.0])

    # 准备错误消息
    msg = "quantile value -0.1 not in \\[0, 1\\]"

    # 断言当分位数值小于 0 或大于 1 时，会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(-0.1)

    # 准备错误消息
    msg = "quantile value 10.0 not in \\[0, 1\\]"

    # 断言当分位数值小于 0 或大于 1 时，会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(10.0)

    # 准备错误消息
    msg = "must be real number, not str"

    # 断言当传入非数值类型的分位数值时，会引发 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        ser.rolling(3).quantile("foo")


def test_rolling_std_1obs():
    # 准备测试数据
    vals = Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # 计算滚动窗口大小为 1 时的标准差
    result = vals.rolling(1, min_periods=1).std()

    # 预期结果是 NaN 数组
    expected = Series([np.nan] * 5)

    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected)

    # 计算滚动窗口大小为 1 时的标准差，忽略自由度调整
    result = vals.rolling(1, min_periods=1).std(ddof=0)

    # 预期结果是包含 0.0 的数组
    expected = Series([0.0] * 5)

    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected)

    # 计算滚动窗口大小为 3 时的标准差，并检查第二个位置是否为 NaN
    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])


def test_rolling_std_neg_sqrt():
    # 单元测试来自 Bottleneck

    # 测试负平方根时的 move_nanstd

    # 准备测试数据
    a = Series(
        [
            0.0011448196318903589,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
        ]
    )

    # 使用滚动窗口大小为 3 时的标准差计算
    b = a.rolling(window=3).std()

    # 断言结果中是否有有限的值
    assert np.isfinite(b[2:]).all()

    # 使用指数加权移动平均（span=3）的标准差计算
    b = a.ewm(span=3).std()

    # 断言结果中是否有有限的值
    assert np.isfinite(b[2:]).all()


def test_step_not_integer_raises():
    # 断言当步长为非整数时，会引发 ValueError 异常
    with pytest.raises(ValueError, match="step must be an integer"):
        DataFrame(range(2)).rolling(1, step="foo")


def test_step_not_positive_raises():
    # 断言当步长为负数时，会引发 ValueError 异常
    with pytest.raises(ValueError, match="step must be >= 0"):
        DataFrame(range(2)).rolling(1, step=-1)
@pytest.mark.parametrize(
    ["values", "window", "min_periods", "expected"],
    [
        # 第一个测试用例
        [
            [20, 10, 10, np.inf, 1, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1],
        ],
        # 第二个测试用例
        [
            [20, 10, 10, np.nan, 10, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1],
        ],
        # 第三个测试用例
        [
            [np.nan, 5, 6, 7, 5, 5, 5],
            3,
            3,
            [np.nan] * 3 + [1, 1, 4 / 3, 0],
        ],
        # 第四个测试用例
        [
            [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0],
        ],
        # 第五个测试用例
        [
            [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0],
        ],
        # 第六个测试用例
        [
            [5, 7] * 4,
            3,
            3,
            [np.nan] * 2 + [4 / 3] * 6,
        ],
        # 第七个测试用例
        [
            [5, 7, 5, np.nan, 7, 5, 7],
            3,
            2,
            [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3],
        ],
    ],
)
def test_rolling_var_same_value_count_logic(values, window, min_periods, expected):
    # GH 42064.
    # Rolling variance test for specific logic regarding consecutive identical values.

    expected = Series(expected)
    sr = Series(values)

    # 使用新的算法实现，如果找到足够数量的连续相同值，则结果将设置为 rolling var 的 .0
    result_var = sr.rolling(window, min_periods=min_periods).var()

    # 使用 `assert_series_equal` 两次进行相等性检查，
    # 因为在 32 位测试中 `check_exact=True` 会由于精度丢失而失败。

    # 1. 结果应该接近正确的值
    # 非零值仍然可能与 "真实值" 稍有不同，
    # 因为是在线算法的结果
    tm.assert_series_equal(result_var, expected)
    # 2. 零值应完全相同，因为新算法在这里生效了
    tm.assert_series_equal(expected == 0, result_var == 0)

    # std 应该也能通过，因为它只是 var 的平方根
    result_std = sr.rolling(window, min_periods=min_periods).std()
    tm.assert_series_equal(result_std, np.sqrt(expected))
    tm.assert_series_equal(expected == 0, result_std == 0)


def test_rolling_mean_sum_floating_artifacts():
    # GH 42064.
    # Test case for rolling mean and sum handling floating artifacts.

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(3)
    result = r.mean()
    assert (result[-3:] == 0).all()
    result = r.sum()
    assert (result[-3:] == 0).all()


def test_rolling_skew_kurt_floating_artifacts():
    # GH 42064 46431
    # Test case for handling floating artifacts in rolling skewness and kurtosis.

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(4)
    result = r.skew()
    assert (result[-2:] == 0).all()
    result = r.kurt()
    assert (result[-2:] == -3).all()


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    # Test case for ensuring numeric only frame handling.

    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    rolling = df.rolling(2, min_periods=1)
    ```python`
    # 通过 getattr 函数获取 rolling 对象中的 kernel 属性对应的方法或函数
    op = getattr(rolling, kernel)
    # 使用获取到的方法或函数对数据进行操作，返回结果给 result
    result = op(numeric_only=numeric_only)
    
    # 根据 numeric_only 的值确定要操作的列，如果为 True，则操作列为 ["a", "b"]，否则为 ["a", "b", "c"]
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    # 根据确定的列对 DataFrame df 进行聚合操作，使用 kernel 函数，重置索引并转换为浮点数类型
    expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
    # 断言预期的列与实际计算的列相同
    assert list(expected.columns) == columns
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，为 kernel 参数设置两个值："corr" 和 "cov"
# 以及 use_arg 参数设置两个值：True 和 False，生成多组测试用例
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # GH#46560
    # 创建一个 DataFrame 对象 df，包含三列：a 列为 [1, 2, 3]，b 列为 2，c 列为 3
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    # 将 df 的 c 列转换为对象类型
    df["c"] = df["c"].astype(object)
    # 根据 use_arg 决定是否使用 df 构造一个元组 arg
    arg = (df,) if use_arg else ()
    # 创建一个滚动窗口 rolling，窗口大小为 2，最小观测数为 1
    rolling = df.rolling(2, min_periods=1)
    # 获取 rolling 对象的 kernel 属性（"corr" 或 "cov"）
    op = getattr(rolling, kernel)
    # 调用 op 方法，传入参数 arg 和 numeric_only=numeric_only，计算结果保存在 result 中
    result = op(*arg, numeric_only=numeric_only)

    # 根据 numeric_only 的值选择需要比较的列，如果为 True 则比较 ["a", "b"]，否则比较整个 ["a", "b", "c"]
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    # 将 df 中选择的列转换为 float 类型，保存在 df2 中
    df2 = df[columns].astype(float)
    # 根据 use_arg 决定是否使用 df2 构造一个元组 arg2
    arg2 = (df2,) if use_arg else ()
    # 创建一个滚动窗口 rolling2，窗口大小为 2，最小观测数为 1
    rolling2 = df2.rolling(2, min_periods=1)
    # 获取 rolling2 对象的 kernel 属性（"corr" 或 "cov"）
    op2 = getattr(rolling2, kernel)
    # 调用 op2 方法，传入参数 arg2 和 numeric_only=numeric_only，计算结果保存在 expected 中
    expected = op2(*arg2, numeric_only=numeric_only)
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，为 dtype 参数设置两个值：int 和 object，生成多组测试用例
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560
    # kernel 变量为 arithmetic_win_operators
    kernel = arithmetic_win_operators
    # 创建一个 Series 对象 ser，包含一个元素为 1，数据类型由 dtype 指定
    ser = Series([1], dtype=dtype)
    # 创建一个滚动窗口 rolling，窗口大小为 2，最小观测数为 1
    rolling = ser.rolling(2, min_periods=1)
    # 获取 rolling 对象的 kernel 属性（arithmetic_win_operators）
    op = getattr(rolling, kernel)
    # 如果 numeric_only 为 True 并且 dtype 为 object，则抛出 NotImplementedError 异常
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        # 否则，调用 op 方法，传入 numeric_only=numeric_only，计算结果保存在 result 中
        result = op(numeric_only=numeric_only)
        # 计算 ser 的 kernel 操作的预期结果，转换为 float 类型，保存在 expected 中
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，为 kernel 参数设置两个值："corr" 和 "cov"
# 以及 use_arg 参数设置两个值：True 和 False，dtype 参数设置两个值：int 和 object，生成多组测试用例
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # GH#46560
    # 创建一个 Series 对象 ser，包含三个元素：1, 2, 3，数据类型由 dtype 指定
    ser = Series([1, 2, 3], dtype=dtype)
    # 根据 use_arg 决定是否使用 ser 构造一个元组 arg
    arg = (ser,) if use_arg else ()
    # 创建一个滚动窗口 rolling，窗口大小为 2，最小观测数为 1
    rolling = ser.rolling(2, min_periods=1)
    # 获取 rolling 对象的 kernel 属性（"corr" 或 "cov"）
    op = getattr(rolling, kernel)
    # 如果 numeric_only 为 True 并且 dtype 为 object，则抛出 NotImplementedError 异常
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        # 否则，调用 op 方法，传入参数 arg 和 numeric_only=numeric_only，计算结果保存在 result 中
        result = op(*arg, numeric_only=numeric_only)

        # 将 ser 转换为 float 类型，保存在 ser2 中
        ser2 = ser.astype(float)
        # 根据 use_arg 决定是否使用 ser2 构造一个元组 arg2
        arg2 = (ser2,) if use_arg else ()
        # 创建一个滚动窗口 rolling2，窗口大小为 2，最小观测数为 1
        rolling2 = ser2.rolling(2, min_periods=1)
        # 获取 rolling2 对象的 kernel 属性（"corr" 或 "cov"）
        op2 = getattr(rolling2, kernel)
        # 调用 op2 方法，传入参数 arg2 和 numeric_only=numeric_only，计算结果保存在 expected 中
        expected = op2(*arg2, numeric_only=numeric_only)
        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，为 tz 参数设置三个值：None, "UTC", "Europe/Prague"，生成多组测试用例
def test_rolling_timedelta_window_non_nanoseconds(unit, tz):
    # Test Sum, GH#55106
    # 创建一个 DataFrame 对象 df_time，包含一列 A，索引为每秒的时间戳，共 5 行，时区由 tz 参数指定
    df_time = DataFrame(
        {"A": range(5)}, index=date_range("2013-01-01", freq="1s", periods=5, tz=tz)
    )
    # 对 df_time 应用窗口大小为 1s 的滚动求和操作，结果保存在 sum_in_nanosecs 中
    sum_in_nanosecs = df_time.rolling("1s").sum()
    # 将 df_time 的索引转换为 unit 单位，例如 microseconds 或 milliseconds
    df_time.index = df_time.index.as_unit(unit)
    # 对 df_time 应用窗口大小为 1s 的滚动求和操作，结果保存在 sum_in_microsecs 中
    sum_in_microsecs = df_time.rolling("1s").sum()
    # 将索引单位转换为纳秒
    sum_in_microsecs.index = sum_in_microsecs.index.as_unit("ns")
    # 使用测试工具验证两个数据框是否相等
    tm.assert_frame_equal(sum_in_nanosecs, sum_in_microsecs)

    # 测试滚动窗口最大值，参考 GitHub 问题 #55026
    # 创建参考日期范围，单位为纳秒，带有时区信息
    ref_dates = date_range("2023-01-01", "2023-01-10", unit="ns", tz=tz)
    # 创建初始值为0的时间序列，索引为参考日期
    ref_series = Series(0, index=ref_dates)
    # 设置第一个索引位置的值为1
    ref_series.iloc[0] = 1
    # 计算滚动窗口为4天的最大值时间序列
    ref_max_series = ref_series.rolling(Timedelta(days=4)).max()

    # 创建日期范围，单位由变量 `unit` 决定，带有时区信息
    dates = date_range("2023-01-01", "2023-01-10", unit=unit, tz=tz)
    # 创建初始值为0的时间序列，索引为日期范围
    series = Series(0, index=dates)
    # 设置第一个索引位置的值为1
    series.iloc[0] = 1
    # 计算滚动窗口为4天的最大值时间序列
    max_series = series.rolling(Timedelta(days=4)).max()

    # 创建包含参考最大值时间序列的数据框
    ref_df = DataFrame(ref_max_series)
    # 创建包含最大值时间序列的数据框
    df = DataFrame(max_series)
    # 将数据框的索引单位转换为纳秒
    df.index = df.index.as_unit("ns")

    # 使用测试工具验证两个数据框是否相等
    tm.assert_frame_equal(ref_df, df)
```