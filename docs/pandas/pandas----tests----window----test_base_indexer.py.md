# `D:\src\scipysrc\pandas\pandas\tests\window\test_base_indexer.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下模块：
    DataFrame,  # 数据帧（二维数据结构）
    MultiIndex,  # 多级索引
    Series,  # 数据序列（一维数据结构）
    concat,  # 数据合并函数
    date_range,  # 日期范围生成器
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具
from pandas.api.indexers import (  # 从 Pandas API 中导入以下索引器：
    BaseIndexer,  # 基础索引器
    FixedForwardWindowIndexer,  # 固定前向窗口索引器
)
from pandas.core.indexers.objects import (  # 从 Pandas 核心索引对象中导入以下对象：
    ExpandingIndexer,  # 扩展索引器
    FixedWindowIndexer,  # 固定窗口索引器
    VariableOffsetWindowIndexer,  # 变量偏移窗口索引器
)

from pandas.tseries.offsets import BusinessDay  # 导入工作日偏移量类


def test_bad_get_window_bounds_signature():
    # 定义一个测试函数，检验不良索引器的窗口边界签名
    class BadIndexer(BaseIndexer):
        def get_window_bounds(self):
            return None

    indexer = BadIndexer()
    # 使用 pytest 检查是否引发值错误，且错误消息匹配特定内容
    with pytest.raises(ValueError, match="BadIndexer does not implement"):
        Series(range(5)).rolling(indexer)


def test_expanding_indexer():
    # 测试扩展索引器
    s = Series(range(10))
    indexer = ExpandingIndexer()
    result = s.rolling(indexer).mean()  # 滚动计算均值
    expected = s.expanding().mean()  # 扩展计算均值
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


def test_indexer_constructor_arg():
    # 测试索引器构造函数参数
    use_expanding = [True, False, True, False, True]
    df = DataFrame({"values": range(5)})

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

    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    result = df.rolling(indexer).sum()  # 滚动求和
    expected = DataFrame({"values": [0.0, 1.0, 3.0, 3.0, 10.0]})
    tm.assert_frame_equal(result, expected)  # 断言结果数据帧相等


def test_indexer_accepts_rolling_args():
    # 测试索引器接受滚动参数
    df = DataFrame({"values": range(5)})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if (
                    center
                    and min_periods == 1
                    and closed == "both"
                    and step == 1
                    and i == 2
                ):
                    start[i] = 0
                    end[i] = num_values
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=1)
    result = df.rolling(
        indexer, center=True, min_periods=1, closed="both", step=1
    ).sum()  # 滚动求和
    expected = DataFrame({"values": [0.0, 1.0, 10.0, 3.0, 4.0]})
    tm.assert_frame_equal(result, expected)  # 断言结果数据帧相等


@pytest.mark.parametrize(  # 使用 pytest 的参数化标记
    "func,np_func,expected,np_kwargs",
    [
        # 计算列表中数值的数量
        ("count", len, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, np.nan], {}),
    
        # 计算列表中的最小值
        ("min", np.min, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, np.nan], {}),
    
        # 计算列表中的最大值
        (
            "max",
            np.max,
            [2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 8.0, 9.0, 9.0, np.nan],
            {},
        ),
    
        # 计算列表中数值的标准差
        (
            "std",
            np.std,
            [
                1.0,
                1.0,
                1.0,
                55.71654452,
                54.85739087,
                53.9845657,
                1.0,
                1.0,
                0.70710678,
                np.nan,
            ],
            {"ddof": 1},
        ),
    
        # 计算列表中数值的方差
        (
            "var",
            np.var,
            [
                1.0,
                1.0,
                1.0,
                3104.333333,
                3009.333333,
                2914.333333,
                1.0,
                1.0,
                0.500000,
                np.nan,
            ],
            {"ddof": 1},
        ),
    
        # 计算列表中数值的中位数
        (
            "median",
            np.median,
            [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 8.5, np.nan],
            {},
        ),
    ],
# 定义测试函数，用于测试滚动窗口向前滚动的行为
def test_rolling_forward_window(
    frame_or_series, func, np_func, expected, np_kwargs, step
):
    # 创建一个包含10个浮点数的numpy数组，值从0到9
    values = np.arange(10.0)
    # 将数组中索引为5的值设为100.0
    values[5] = 100.0

    # 创建一个固定向前窗口索引器对象，窗口大小为3
    indexer = FixedForwardWindowIndexer(window_size=3)

    # 测试滚动窗口对象，期望引发值错误异常，消息为"Forward-looking windows can't have center=True"
    match = "Forward-looking windows can't have center=True"
    rolling = frame_or_series(values).rolling(window=indexer, center=True)
    with pytest.raises(ValueError, match=match):
        getattr(rolling, func)()

    # 测试滚动窗口对象，期望引发值错误异常，消息为"Forward-looking windows don't support setting the closed argument"
    match = "Forward-looking windows don't support setting the closed argument"
    rolling = frame_or_series(values).rolling(window=indexer, closed="right")
    with pytest.raises(ValueError, match=match):
        getattr(rolling, func)()

    # 创建滚动窗口对象，设置窗口大小、最小周期和步长
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=2, step=step)
    # 调用指定函数（func参数所指定的函数）并获取结果
    result = getattr(rolling, func)()

    # 检查函数的输出是否与明确提供的数组(expected参数)匹配
    expected = frame_or_series(expected)[::step]
    tm.assert_equal(result, expected)

    # 检查滚动函数的输出是否与应用于滚动窗口对象的另一函数（np_func参数）的结果匹配
    expected2 = frame_or_series(rolling.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result, expected2)

    # 如果未指定最小周期，则检查函数输出是否与应用替代函数（np_func参数）的结果匹配
    # GH 39604: 在最小周期(min_periods)被弃用后，apply(lambda x: len(x)) 等同于设置 min_periods=0 后的 count
    min_periods = 0 if func == "count" else None
    rolling3 = frame_or_series(values).rolling(window=indexer, min_periods=min_periods)
    result3 = getattr(rolling3, func)()
    expected3 = frame_or_series(rolling3.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result3, expected3)


# 定义测试函数，用于测试滚动窗口向前偏斜度的行为
def test_rolling_forward_skewness(frame_or_series, step):
    # 创建一个包含10个浮点数的numpy数组，值从0到9
    values = np.arange(10.0)
    # 将数组中索引为5的值设为100.0
    values[5] = 100.0

    # 创建一个固定向前窗口索引器对象，窗口大小为5
    indexer = FixedForwardWindowIndexer(window_size=5)
    # 创建滚动窗口对象，设置窗口大小、最小周期和步长
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=3, step=step)
    # 调用skew方法计算偏斜度并获取结果
    result = rolling.skew()

    # 创建一个包含期望值的Series对象，期望值是一个列表，步长为step
    expected = frame_or_series(
        [
            0.0,
            2.232396,
            2.229508,
            2.228340,
            2.229091,
            2.231989,
            0.0,
            0.0,
            np.nan,
            np.nan,
        ]
    )[::step]
    # 断言滚动窗口的输出与期望值匹配
    tm.assert_equal(result, expected)


# 使用pytest的parametrize装饰器定义参数化测试
@pytest.mark.parametrize(
    "func,expected",
    [
        # 参数化测试：计算协方差
        ("cov", [2.0, 2.0, 2.0, 97.0, 2.0, -93.0, 2.0, 2.0, np.nan, np.nan]),
        # 参数化测试：计算相关系数
        (
            "corr",
            [
                1.0,
                1.0,
                1.0,
                0.8704775290207161,
                0.018229084250926637,
                -0.861357304646493,
                1.0,
                1.0,
                np.nan,
                np.nan,
            ],
        ),
    ],
)
# 定义测试函数，用于测试滚动窗口向前计算协方差和相关系数的行为
def test_rolling_forward_cov_corr(func, expected):
    # 创建包含10个值的二维numpy数组，值从0到9
    values1 = np.arange(10).reshape(-1, 1)
    # 创建第二个数组，每个值是第一个数组对应位置的两倍
    values2 = values1 * 2
    # 将第一个数组中索引为(5, 0)的值设为100
    values1[5, 0] = 100
    # 将 values1 和 values2 按列拼接成一个 numpy 数组
    values = np.concatenate([values1, values2], axis=1)

    # 创建一个 FixedForwardWindowIndexer 对象，用于定义滚动窗口大小为 3
    indexer = FixedForwardWindowIndexer(window_size=3)

    # 使用 DataFrame 对象包装 values，并以 indexer 定义的滚动窗口和最小期数创建滚动对象
    rolling = DataFrame(values).rolling(window=indexer, min_periods=3)

    # 对滚动对象执行指定的函数（如协方差或相关性计算），并选择特定的行和列
    # 这里只关注成对的协方差或相关性
    result = getattr(rolling, func)().loc[(slice(None), 1), 0]

    # 重置结果的索引，丢弃旧的索引
    result = result.reset_index(drop=True)

    # 将预期结果转换为 Series，并重置其索引，丢弃旧的索引
    expected = Series(expected).reset_index(drop=True)

    # 将预期结果的名称设置为与结果相同的名称
    expected.name = result.name

    # 使用测试框架（可能是 pandas 中的 tm 模块）断言结果与预期结果相等
    tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "closed,expected_data",
    [
        ["right", [0.0, 1.0, 2.0, 3.0, 7.0, 12.0, 6.0, 7.0, 8.0, 9.0]],
        ["left", [0.0, 0.0, 1.0, 2.0, 5.0, 9.0, 5.0, 6.0, 7.0, 8.0]],
    ],
)
# 参数化测试函数，对两种不同的 "closed" 参数进行测试
def test_non_fixed_variable_window_indexer(closed, expected_data):
    # 创建一个包含 10 个日期的时间索引
    index = date_range("2020", periods=10)
    # 创建一个 DataFrame，包含从0到9的整数数据，使用上面的时间索引作为行索引
    df = DataFrame(range(10), index=index)
    # 创建一个 BusinessDay 偏移量对象，表示每次移动一个工作日
    offset = BusinessDay(1)
    # 创建一个 VariableOffsetWindowIndexer 对象，用于在滚动窗口计算中使用
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    # 对 DataFrame 进行滚动计算，使用给定的 indexer 对象和不同的 closed 参数
    result = df.rolling(indexer, closed=closed).sum()
    # 创建一个预期的 DataFrame，包含预期的计算结果数据，使用相同的时间索引
    expected = DataFrame(expected_data, index=index)
    # 断言计算结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_variableoffsetwindowindexer_not_dti():
    # GH 54379
    # 测试当传入的索引不是 DatetimeIndex 时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match="index must be a DatetimeIndex."):
        VariableOffsetWindowIndexer(index="foo", offset=BusinessDay(1))


def test_variableoffsetwindowindexer_not_offset():
    # GH 54379
    # 测试当传入的偏移量不是 DateOffset-like 对象时是否会抛出 ValueError 异常
    idx = date_range("2020", periods=10)
    with pytest.raises(ValueError, match="offset must be a DateOffset-like object."):
        VariableOffsetWindowIndexer(index=idx, offset="foo")


def test_fixed_forward_indexer_count(step):
    # GH: 35579
    # 创建一个包含空值的 DataFrame
    df = DataFrame({"b": [None, None, None, 7]})
    # 创建一个 FixedForwardWindowIndexer 对象，用于在滚动窗口计算中使用
    indexer = FixedForwardWindowIndexer(window_size=2)
    # 对 DataFrame 进行滚动计算，使用给定的 indexer 对象和其他参数
    result = df.rolling(window=indexer, min_periods=0, step=step).count()
    # 创建一个预期的 DataFrame，包含预期的计算结果数据
    expected = DataFrame({"b": [0.0, 0.0, 1.0, 1.0]})[::step]
    # 断言计算结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("end_value", "values"), [(1, [0.0, 1, 1, 3, 2]), (-1, [0.0, 1, 0, 3, 1])]
)
@pytest.mark.parametrize(("func", "args"), [("median", []), ("quantile", [0.5])])
# 参数化测试函数，测试不同的 end_value 和不同的聚合函数
def test_indexer_quantile_sum(end_value, values, func, args):
    # GH 37153
    # 定义一个自定义的索引器类，继承自 BaseIndexer
    class CustomIndexer(BaseIndexer):
        # 实现获取滚动窗口边界的方法
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = max(i + end_value, 1)
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    use_expanding = [True, False, True, False, True]
    # 创建一个包含整数数据的 DataFrame
    df = DataFrame({"values": range(5)})
    # 创建一个 CustomIndexer 对象，用于在滚动窗口计算中使用
    indexer = CustomIndexer(window_size=1, use_expanding=use_expanding)
    # 对 DataFrame 进行滚动计算，并应用指定的聚合函数和参数
    result = getattr(df.rolling(indexer), func)(*args)
    # 创建一个预期的 DataFrame，包含预期的计算结果数据
    expected = DataFrame({"values": values})
    # 断言计算结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer_class", [FixedWindowIndexer, FixedForwardWindowIndexer, ExpandingIndexer]
)
@pytest.mark.parametrize("window_size", [1, 2, 12])
@pytest.mark.parametrize(
    "df_data",
    [
        {"a": [1, 1], "b": [0, 1]},
        {"a": [1, 2], "b": [0, 1]},
        {"a": [1] * 16, "b": [np.nan, 1, 2, np.nan] + list(range(4, 16))},
    ],
)
# 参数化测试函数，测试不同的索引器类、窗口大小和数据框架数据
def test_indexers_are_reusable_after_groupby_rolling(
    indexer_class, window_size, df_data


    # 定义三个变量：indexer_class、window_size、df_data
    # 这些变量通常用于某个程序或函数的参数或配置选项
# GH 43267
# 根据给定的 df_data 创建一个 DataFrame 对象
df = DataFrame(df_data)
# 设置循环试验次数为3次
num_trials = 3
# 使用 indexer_class 创建一个索引器对象，并设定窗口大小为 window_size
indexer = indexer_class(window_size=window_size)
# 保存原始的窗口大小
original_window_size = indexer.window_size
# 进行 num_trials 次循环
for i in range(num_trials):
    # 对 DataFrame 进行分组操作，按照列 "a" 分组后对列 "b" 应用 rolling 窗口函数计算均值
    df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
    # 断言当前的 indexer 窗口大小与原始保存的窗口大小一致
    assert indexer.window_size == original_window_size


@pytest.mark.parametrize(
    "window_size, num_values, expected_start, expected_end",
    [
        # 参数化测试用例，包含多组参数：窗口大小，数值个数，期望的起始和结束值
        (1, 1, [0], [1]),
        (1, 2, [0, 1], [1, 2]),
        (2, 1, [0], [1]),
        (2, 2, [0, 1], [2, 2]),
        (5, 12, range(12), list(range(5, 12)) + [12] * 5),
        (12, 5, range(5), [5] * 5),
        (0, 0, np.array([]), np.array([])),
        (1, 0, np.array([]), np.array([])),
        (0, 1, [0], [0]),
    ],
)
# 定义参数化测试函数，测试固定前向窗口索引器的边界情况
def test_fixed_forward_indexer_bounds(
    window_size, num_values, expected_start, expected_end, step
):
    # GH 43267
    # 使用 FixedForwardWindowIndexer 类创建一个索引器对象，设置窗口大小为 window_size
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    # 调用索引器对象的 get_window_bounds 方法获取窗口的起始和结束位置
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    # 使用 assert_numpy_array_equal 函数断言 start 数组与期望的起始数组一致
    tm.assert_numpy_array_equal(
        start, np.array(expected_start[::step]), check_dtype=False
    )
    # 使用 assert_numpy_array_equal 函数断言 end 数组与期望的结束数组一致
    tm.assert_numpy_array_equal(end, np.array(expected_end[::step]), check_dtype=False)
    # 断言 start 和 end 数组的长度相同
    assert len(start) == len(end)


@pytest.mark.parametrize(
    "df, window_size, expected",
    [
        # 参数化测试用例，包含多组参数：DataFrame 对象，窗口大小，期望的结果
        (
            DataFrame({"b": [0, 1, 2], "a": [1, 2, 2]}),
            2,
            Series(
                [0, 1.5, 2.0],
                index=MultiIndex.from_arrays([[1, 2, 2], range(3)], names=["a", None]),
                name="b",
                dtype=np.float64,
            ),
        ),
        (
            DataFrame(
                {
                    "b": [np.nan, 1, 2, np.nan] + list(range(4, 18)),
                    "a": [1] * 7 + [2] * 11,
                    "c": range(18),
                }
            ),
            12,
            Series(
                [
                    3.6,
                    3.6,
                    4.25,
                    5.0,
                    5.0,
                    5.5,
                    6.0,
                    12.0,
                    12.5,
                    13.0,
                    13.5,
                    14.0,
                    14.5,
                    15.0,
                    15.5,
                    16.0,
                    16.5,
                    17.0,
                ],
                index=MultiIndex.from_arrays(
                    [[1] * 7 + [2] * 11, range(18)], names=["a", None]
                ),
                name="b",
                dtype=np.float64,
            ),
        ),
    ],
)
# 定义参数化测试函数，测试带有固定前向窗口索引器的滚动分组计算
def test_rolling_groupby_with_fixed_forward_specific(df, window_size, expected):
    # GH 43267
    # 使用 FixedForwardWindowIndexer 类创建一个索引器对象，设置窗口大小为 window_size
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    # 对 DataFrame 进行分组操作，按照列 "a" 分组后对列 "b" 应用 rolling 窗口函数计算均值
    result = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).mean()
    # 使用 assert_series_equal 函数断言计算结果与期望结果 expected 相同
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "group_keys",
    # 定义一个包含多个元组的列表，每个元组都是由数字组成
    [
        # 元组 (1,) 包含一个元素 1
        (1,),
        # 元组 (1, 2) 包含两个元素 1 和 2
        (1, 2),
        # 元组 (2, 1) 包含两个元素 2 和 1
        (2, 1),
        # 元组 (1, 1, 2) 包含三个元素 1、1 和 2
        (1, 1, 2),
        # 元组 (1, 2, 1) 包含三个元素 1、2 和 1
        (1, 2, 1),
        # 元组 (1, 1, 2, 2) 包含四个元素 1、1、2 和 2
        (1, 1, 2, 2),
        # 元组 (1, 2, 3, 2, 3) 包含五个元素 1、2、3、2 和 3
        (1, 2, 3, 2, 3),
        # 元组 (1, 1, 2) 重复四次，包含四个元素 1、1 和 2
        (1, 1, 2) * 4,
        # 元组 (1, 2, 3) 重复五次，包含五个元素 1、2 和 3
        (1, 2, 3) * 5,
    ]
# 使用 pytest 的装饰器标记该函数为参数化测试函数，参数为 window_size 分别取 1, 2, 3, 4, 5, 8, 20
@pytest.mark.parametrize("window_size", [1, 2, 3, 4, 5, 8, 20])
def test_rolling_groupby_with_fixed_forward_many(group_keys, window_size):
    # 创建 DataFrame 对象 df，包含三列：a 列为 group_keys 的数组，b 列为从 17 开始的浮点数递增数组，c 列为整数递增数组
    df = DataFrame(
        {
            "a": np.array(list(group_keys)),
            "b": np.arange(len(group_keys), dtype=np.float64) + 17,
            "c": np.arange(len(group_keys), dtype=np.int64),
        }
    )

    # 使用 FixedForwardWindowIndexer 对象创建索引器 indexer
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    # 对 df 按列 "a" 分组，计算每组内 b 列的滚动窗口和，结果为 Series 对象 result
    result = df.groupby("a")["b"].rolling(window=indexer, min_periods=1).sum()
    # 设置结果 result 的索引名称为 ["a", "c"]
    result.index.names = ["a", "c"]

    # 对 df 按列 "a" 分组，选择列 "a", "b", "c" 构成分组对象 groups
    groups = df.groupby("a")[["a", "b", "c"]]
    # 手动计算每组内 b 列的滚动窗口和，结果为 DataFrame 对象 manual
    manual = concat(
        [
            g.assign(
                b=[
                    g["b"].iloc[i : i + window_size].sum(min_count=1)
                    for i in range(len(g))
                ]
            )
            for _, g in groups
        ]
    )
    # 将 manual 的索引设置为 ["a", "c"]，并选择列 "b"
    manual = manual.set_index(["a", "c"])["b"]

    # 使用 assert_series_equal 检查 result 和 manual 是否相等
    tm.assert_series_equal(result, manual)


def test_unequal_start_end_bounds():
    # 定义一个名为 CustomIndexer 的类，继承自 BaseIndexer 类
    class CustomIndexer(BaseIndexer):
        # 定义 get_window_bounds 方法，返回不同的起始和结束边界数组
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return np.array([1]), np.array([1, 2])

    # 创建 CustomIndexer 类的实例 indexer
    indexer = CustomIndexer()
    # 使用 Series 对象创建 rolling 对象 roll，传入 indexer 作为索引器
    roll = Series(1).rolling(indexer)
    # 定义匹配字符串 match 为 "start"
    match = "start"
    # 使用 pytest.raises 检查 roll.mean() 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.mean()

    # 使用 pytest.raises 检查 next(iter(roll)) 是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    # 使用 pytest.raises 检查 roll.corr(pairwise=True) 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    # 使用 pytest.raises 检查 roll.cov(pairwise=True) 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)


def test_unequal_bounds_to_object():
    # GH 44470
    # 定义一个名为 CustomIndexer 的类，继承自 BaseIndexer 类
    class CustomIndexer(BaseIndexer):
        # 定义 get_window_bounds 方法，返回不同的起始和结束边界数组
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return np.array([1]), np.array([2])

    # 创建 CustomIndexer 类的实例 indexer
    indexer = CustomIndexer()
    # 使用 Series 对象创建 rolling 对象 roll，传入 indexer 作为索引器
    roll = Series([1, 1]).rolling(indexer)
    # 定义匹配字符串 match 为 "start and end"
    match = "start and end"
    # 使用 pytest.raises 检查 roll.mean() 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.mean()

    # 使用 pytest.raises 检查 next(iter(roll)) 是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        next(iter(roll))

    # 使用 pytest.raises 检查 roll.corr(pairwise=True) 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)

    # 使用 pytest.raises 检查 roll.cov(pairwise=True) 方法是否抛出 ValueError 异常，且异常信息匹配 match
    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)
```