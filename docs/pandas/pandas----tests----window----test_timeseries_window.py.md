# `D:\src\scipysrc\pandas\pandas\tests\window\test_timeseries_window.py`

```
# 导入NumPy库，用于科学计算
import numpy as np

# 导入pytest库，用于编写和运行测试
import pytest

# 导入pandas.util._test_decorators模块中的_test_decorators别名为td
import pandas.util._test_decorators as td

# 从pandas库中导入DataFrame、DatetimeIndex、Index、MultiIndex、NaT、Series、Timestamp、date_range等类或函数
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)

# 导入pandas._testing模块别名为tm，用于内部测试
import pandas._testing as tm

# 从pandas.tseries模块中导入offsets
from pandas.tseries import offsets

# 使用pytest的fixture装饰器定义regular函数，返回一个DataFrame对象，带有两列'A'和'B'，'A'列包含日期范围数据
@pytest.fixture
def regular():
    return DataFrame(
        {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
    ).set_index("A")

# 使用pytest的fixture装饰器定义ragged函数，返回一个DataFrame对象，带有一列'B'，并将其索引设置为时间戳列表
@pytest.fixture
def ragged():
    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]
    return df

# 定义TestRollingTS类，用于测试时间滚动窗口操作
class TestRollingTS:
    # 测试方法的文档字符串
    # rolling time-series friendly
    # xref GH13327
    def test_doc_string(self):
        # 创建一个包含时间序列数据的DataFrame对象
        df = DataFrame(
            {"B": [0, 1, 2, np.nan, 4]},
            index=[
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ],
        )
        df
        # 对DataFrame对象应用rolling窗口为'2s'的滚动求和操作
        df.rolling("2s").sum()

    # 测试无效的窗口大小非整数情况
    def test_invalid_window_non_int(self, regular):
        # 非有效的频率
        msg = "passed window foobar is not compatible with a datetimelike index"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="foobar")
        # 非日期时间类型的索引
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            regular.reset_index().rolling(window="foobar")

    # 使用pytest.mark.parametrize装饰器参数化测试，测试非固定频率窗口
    @pytest.mark.parametrize("freq", ["2MS", offsets.MonthBegin(2)])
    def test_invalid_window_nonfixed(self, freq, regular):
        # 非固定频率
        msg = "\\<2 \\* MonthBegins\\> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window=freq)

    # 使用pytest.mark.parametrize装饰器参数化测试，测试有效的窗口大小
    @pytest.mark.parametrize("freq", ["1D", offsets.Day(2), "2ms"])
    def test_valid_window(self, freq, regular):
        regular.rolling(window=freq)

    # 使用pytest.mark.parametrize装饰器参数化测试，测试无效的最小期数
    @pytest.mark.parametrize("minp", [1.0, "foo", np.array([1, 2, 3])])
    def test_invalid_minp(self, minp, regular):
        # 非整数的最小期数
        msg = (
            r"local variable 'minp' referenced before assignment|"
            "min_periods must be an integer"
        )
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="1D", min_periods=minp)
    def test_on(self, regular):
        # 将传入的 regular 参数作为 DataFrame df
        df = regular

        # 测试异常情况：指定的列名 "foobar" 不是有效列
        msg = (
            r"invalid on specified as foobar, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2s", on="foobar")

        # 测试正常情况：指定的列名 "C" 是有效的列
        df = df.copy()
        df["C"] = date_range("20130101", periods=len(df))
        # 对指定列 "C" 进行滚动窗口操作，计算窗口为 "2d" 的数据和
        df.rolling(window="2d", on="C").sum()

        # 测试异常情况：window 参数不是整数类型
        msg = "window must be an integer"
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2d", on="B")

        # 测试正常情况：虽然未选择列 "B"，但对列 "C" 进行滚动窗口操作
        df.rolling(window="2d", on="C").B.sum()

    def test_monotonic_on(self):
        # 测试条件：必须选择的列或索引必须单调递增
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )

        # 断言列 "A" 必须是单调递增的
        assert df.A.is_monotonic_increasing
        # 对列 "A" 进行滚动窗口操作，计算窗口为 "2s" 的数据和
        df.rolling("2s", on="A").sum()

        # 将列 "A" 设置为索引
        df = df.set_index("A")
        # 断言索引必须是单调递增的
        assert df.index.is_monotonic_increasing
        # 对索引进行滚动窗口操作，计算窗口为 "2s" 的数据和
        df.rolling("2s").sum()

    def test_non_monotonic_on(self):
        # 测试情况：索引或选择的列不是单调递增的情况
        # GH 19248
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )
        # 将列 "A" 设置为索引
        df = df.set_index("A")
        # 将索引转换为列表，并使其不再单调递增
        non_monotonic_index = df.index.to_list()
        non_monotonic_index[0] = non_monotonic_index[3]
        df.index = non_monotonic_index

        # 断言索引不是单调递增的
        assert not df.index.is_monotonic_increasing

        # 测试异常情况：索引值必须是单调递增的
        msg = "index values must be monotonic"
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s").sum()

        # 重置索引为默认的整数索引
        df = df.reset_index()

        # 测试异常情况：指定的列名 "A" 是索引，而不是列的情况
        msg = (
            r"invalid on specified as A, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s", on="A").sum()
    def test_frame_on(self):
        # 创建一个 DataFrame 对象，包含列'B'和'C'
        df = DataFrame(
            {"B": range(5), "C": date_range("20130101 09:00:00", periods=5, freq="3s")}
        )

        # 向 DataFrame 添加列'A'，包含时间戳数据
        df["A"] = [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130101 09:00:02"),
            Timestamp("20130101 09:00:03"),
            Timestamp("20130101 09:00:05"),
            Timestamp("20130101 09:00:06"),
        ]

        # 创建预期结果，通过将'A'列设置为索引，进行滚动窗口操作（2秒），计算'B'列的和，然后重新设置索引
        expected = df.set_index("A").rolling("2s").B.sum().reset_index(drop=True)

        # 使用'on'参数进行滚动窗口操作，计算'B'列的和
        result = df.rolling("2s", on="A").B.sum()
        tm.assert_series_equal(result, expected)

        # 作为一个数据框进行测试
        # 我们应该忽略'on'作为聚合列
        # 注意，预期结果设置、计算和重置了列的顺序需要交换
        # 而实际结果保持了原始顺序
        expected = (
            df.set_index("A").rolling("2s")[["B"]].sum().reset_index()[["B", "A"]]
        )

        # 使用'on'参数对'B'列进行滚动窗口操作，计算和
        result = df.rolling("2s", on="A")[["B"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_frame_on2(self, unit):
        # 使用多个聚合列
        # 创建一个时间索引对象
        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)
        
        # 创建一个 DataFrame 对象，包含列'A'、'B'、'C'
        df = DataFrame(
            {
                "A": [0, 1, 2, 3, 4],
                "B": [0, 1, 2, np.nan, 4],
                "C": dti,
            },
            columns=["A", "C", "B"],
        )

        # 创建预期结果 DataFrame
        expected1 = DataFrame(
            {"A": [0.0, 1, 3, 3, 7], "B": [0, 1, 3, np.nan, 4], "C": df["C"]},
            columns=["A", "C", "B"],
        )

        # 使用'on'参数对整个 DataFrame 进行滚动窗口操作，计算和
        result = df.rolling("2s", on="C").sum()
        expected = expected1
        tm.assert_frame_equal(result, expected)

        # 创建预期结果 Series
        expected = Series([0, 1, 3, np.nan, 4], name="B")
        # 使用'on'参数对'B'列进行滚动窗口操作，计算和
        result = df.rolling("2s", on="C").B.sum()
        tm.assert_series_equal(result, expected)

        # 创建预期结果 DataFrame，只包含列'A'、'B'、'C'
        expected = expected1[["A", "B", "C"]]
        # 使用'on'参数对整个 DataFrame 的子集进行滚动窗口操作，计算和
        result = df.rolling("2s", on="C")[["A", "B", "C"]].sum()
        tm.assert_frame_equal(result, expected)
    def test_basic_regular(self, regular):
        # 复制传入的 regular 数据框
        df = regular.copy()

        # 将数据框的索引设置为一个日期范围，频率为每天，共5天
        df.index = date_range("20130101", periods=5, freq="D")
        # 计算滚动窗口为1的滑动求和，并与期望结果比较
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="1D").sum()
        tm.assert_frame_equal(result, expected)

        # 将数据框的索引设置为一个日期范围，频率为每两天，共5天
        df.index = date_range("20130101", periods=5, freq="2D")
        # 计算滚动窗口为1，最小期数为1的滑动求和，并与期望结果比较
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        # 计算滚动窗口为1，最小期数为1的滑动求和，并与期望结果比较
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        # 计算滚动窗口为1的滑动求和，并与期望结果比较
        expected = df.rolling(window=1).sum()
        result = df.rolling(window="2D").sum()
        tm.assert_frame_equal(result, expected)

    def test_min_periods(self, regular):
        # 对于最小期数进行比较
        df = regular

        # 这两者略有不同
        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_closed(self, regular, unit):
        # 引用 GH13965

        # 创建一个日期时间索引，作为特定单位的时间戳
        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:01"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:04"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)

        # 创建一个数据框，具有'A'列，索引为上述日期时间索引
        df = DataFrame(
            {"A": [1] * 5},
            index=dti,
        )

        # closed 参数必须是 'right', 'left', 'both', 'neither' 中的一个
        msg = "closed must be 'right', 'left', 'both' or 'neither'"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="2s", closed="blabla")

        # 创建一个期望结果的副本
        expected = df.copy()
        expected["A"] = [1.0, 2, 2, 2, 1]
        # 计算滚动窗口为2秒，closed 参数为'right'的滑动求和，并与期望结果比较
        result = df.rolling("2s", closed="right").sum()
        tm.assert_frame_equal(result, expected)

        # 默认情况应为 'right'
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        # 创建一个期望结果的副本
        expected = df.copy()
        expected["A"] = [1.0, 2, 3, 3, 2]
        # 计算滚动窗口为2秒，closed 参数为'both'的滑动求和，并与期望结果比较
        result = df.rolling("2s", closed="both").sum()
        tm.assert_frame_equal(result, expected)

        # 创建一个期望结果的副本
        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 2, 2, 1]
        # 计算滚动窗口为2秒，closed 参数为'left'的滑动求和，并与期望结果比较
        result = df.rolling("2s", closed="left").sum()
        tm.assert_frame_equal(result, expected)

        # 创建一个期望结果的副本
        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 1, 1, np.nan]
        # 计算滚动窗口为2秒，closed 参数为'neither'的滑动求和，并与期望结果比较
        result = df.rolling("2s", closed="neither").sum()
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试在不同滚动窗口大小和最小有效数据点数下的数据累加和操作
    def test_ragged_sum(self, ragged):
        # 将输入的不规则数据框赋值给变量 df
        df = ragged
        # 对数据框 df 进行滚动窗口为 "1s"，最小有效数据点数为 1 的累加和计算
        result = df.rolling(window="1s", min_periods=1).sum()
        # 创建预期结果的副本，与 df 相同，但 "B" 列预期为 [0.0, 1, 2, 3, 4]
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 同上，对滚动窗口为 "2s"，最小有效数据点数为 1 的累加和操作
        result = df.rolling(window="2s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 3, 7]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "2s"，最小有效数据点数为 2 的累加和操作
        result = df.rolling(window="2s", min_periods=2).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, np.nan, 7]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "3s"，最小有效数据点数为 1 的累加和操作
        result = df.rolling(window="3s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "3s" 的累加和操作，未指定最小有效数据点数，采用默认值
        result = df.rolling(window="3s").sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "4s"，最小有效数据点数为 1 的累加和操作
        result = df.rolling(window="4s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "4s"，最小有效数据点数为 3 的累加和操作
        result = df.rolling(window="4s", min_periods=3).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        # 滚动窗口为 "5s"，最小有效数据点数为 1 的累加和操作
        result = df.rolling(window="5s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 10]
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在不同滚动窗口大小和最小有效数据点数下的数据均值计算
    def test_ragged_mean(self, ragged):
        # 将输入的不规则数据框赋值给变量 df
        df = ragged
        # 对数据框 df 进行滚动窗口为 "1s"，最小有效数据点数为 1 的均值计算
        result = df.rolling(window="1s", min_periods=1).mean()
        # 创建预期结果的副本，与 df 相同，但 "B" 列预期为 [0.0, 1, 2, 3, 4]
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 同上，对滚动窗口为 "2s"，最小有效数据点数为 1 的均值计算
        result = df.rolling(window="2s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在不同滚动窗口大小和最小有效数据点数下的数据中位数计算
    def test_ragged_median(self, ragged):
        # 将输入的不规则数据框赋值给变量 df
        df = ragged
        # 对数据框 df 进行滚动窗口为 "1s"，最小有效数据点数为 1 的中位数计算
        result = df.rolling(window="1s", min_periods=1).median()
        # 创建预期结果的副本，与 df 相同，但 "B" 列预期为 [0.0, 1, 2, 3, 4]
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 同上，对滚动窗口为 "2s"，最小有效数据点数为 1 的中位数计算
        result = df.rolling(window="2s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在不同滚动窗口大小和最小有效数据点数下的数据分位数计算
    def test_ragged_quantile(self, ragged):
        # 将输入的不规则数据框赋值给变量 df
        df = ragged
        # 对数据框 df 进行滚动窗口为 "1s"，最小有效数据点数为 1 的分位数计算（50%分位数）
        result = df.rolling(window="1s", min_periods=1).quantile(0.5)
        # 创建预期结果的副本，与 df 相同，但 "B" 列预期为 [0.0, 1, 2, 3, 4]
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 同上，对滚动窗口为 "2s"，最小有效数据点数为 1 的分位数计算（50%分位数）
        result = df.rolling(window="2s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)
    # 测试函数，计算不同窗口大小下的滚动标准差，并与预期结果进行比较
    def test_ragged_std(self, ragged):
        # 将传入的 ragged 数据框赋值给 df
        df = ragged
        # 计算窗口大小为 "1s"，最小数据点为1的滚动标准差，自由度为0
        result = df.rolling(window="1s", min_periods=1).std(ddof=0)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全0.0
        expected["B"] = [0.0] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "1s"，最小数据点为1的滚动标准差，自由度为1
        result = df.rolling(window="1s", min_periods=1).std(ddof=1)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全 NaN
        expected["B"] = [np.nan] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "3s"，最小数据点为1的滚动标准差，自由度为0
        result = df.rolling(window="3s", min_periods=1).std(ddof=0)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [0.0, 0.5, 0.5, 0.5, 0.5]
        expected["B"] = [0.0] + [0.5] * 4
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "5s"，最小数据点为1的滚动标准差，自由度为1
        result = df.rolling(window="5s", min_periods=1).std(ddof=1)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [NaN, 0.707107, 1.0, 1.0, 1.290994]
        expected["B"] = [np.nan, 0.707107, 1.0, 1.0, 1.290994]
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

    # 测试函数，计算不同窗口大小下的滚动方差，并与预期结果进行比较
    def test_ragged_var(self, ragged):
        # 将传入的 ragged 数据框赋值给 df
        df = ragged
        # 计算窗口大小为 "1s"，最小数据点为1的滚动方差，自由度为0
        result = df.rolling(window="1s", min_periods=1).var(ddof=0)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全0.0
        expected["B"] = [0.0] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "1s"，最小数据点为1的滚动方差，自由度为1
        result = df.rolling(window="1s", min_periods=1).var(ddof=1)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全 NaN
        expected["B"] = [np.nan] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "3s"，最小数据点为1的滚动方差，自由度为0
        result = df.rolling(window="3s", min_periods=1).var(ddof=0)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [0.0, 0.25, 0.25, 0.25, 0.25]
        expected["B"] = [0.0] + [0.25] * 4
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "5s"，最小数据点为1的滚动方差，自由度为1
        result = df.rolling(window="5s", min_periods=1).var(ddof=1)
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [NaN, 0.5, 1.0, 1.0, 1 + 2 / 3.0]
        expected["B"] = [np.nan, 0.5, 1.0, 1.0, 1 + 2 / 3.0]
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

    # 测试函数，计算不同窗口大小下的滚动偏度，并与预期结果进行比较
    def test_ragged_skew(self, ragged):
        # 将传入的 ragged 数据框赋值给 df
        df = ragged
        # 计算窗口大小为 "3s"，最小数据点为1的滚动偏度
        result = df.rolling(window="3s", min_periods=1).skew()
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全 NaN
        expected["B"] = [np.nan] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "5s"，最小数据点为1的滚动偏度
        result = df.rolling(window="5s", min_periods=1).skew()
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [NaN, NaN, 0.0, 0.0, 0.0]
        expected["B"] = [np.nan] * 2 + [0.0] * 3
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

    # 测试函数，计算不同窗口大小下的滚动峰度，并与预期结果进行比较
    def test_ragged_kurt(self, ragged):
        # 将传入的 ragged 数据框赋值给 df
        df = ragged
        # 计算窗口大小为 "3s"，最小数据点为1的滚动峰度
        result = df.rolling(window="3s", min_periods=1).kurt()
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为全 NaN
        expected["B"] = [np.nan] * 5
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal(result, expected)

        # 计算窗口大小为 "5s"，最小数据点为1的滚动峰度
        result = df.rolling(window="5s", min_periods=1).kurt()
        # 复制 df 作为预期结果
        expected = df.copy()
        # 将预期结果中的列 "B" 设为 [NaN, NaN, NaN, NaN, -1.2]
        expected["B"] = [np.nan] * 4 + [-1.2]
        # 使用测试工具比较计算结果与预期结果
        tm.assert_frame_equal
    # 测试处理不规则时间序列的计数功能
    def test_ragged_count(self, ragged):
        # 将输入的不规则时间序列赋给变量 df
        df = ragged
        # 对时间窗口为 "1s" 的滚动窗口应用计数函数，保留每个时间点的计数结果
        result = df.rolling(window="1s", min_periods=1).count()
        # 创建期望的结果 DataFrame，与原始数据框一致
        expected = df.copy()
        expected["B"] = [1.0, 1, 1, 1, 1]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 将输入的不规则时间序列赋给变量 df
        df = ragged
        # 对时间窗口为 "1s" 的滚动窗口应用计数函数，未指定最小观测期数
        result = df.rolling(window="1s").count()
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间窗口为 "2s" 的滚动窗口应用计数函数，保留每个时间点的计数结果
        result = df.rolling(window="2s", min_periods=1).count()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [1.0, 1, 2, 1, 2]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间窗口为 "2s" 的滚动窗口应用计数函数，至少需要2个观测点来计算结果
        result = df.rolling(window="2s", min_periods=2).count()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 2, np.nan, 2]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

    # 测试处理规则时间序列的最小值功能
    def test_regular_min(self):
        # 创建一个包含时间索引和数值列 "B" 的 DataFrame
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [0.0, 1, 2, 3, 4]}
        ).set_index("A")
        # 对时间窗口为 "1s" 的滚动窗口应用最小值函数
        result = df.rolling("1s").min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个包含时间索引和数值列 "B" 的 DataFrame
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [5, 4, 3, 4, 5]}
        ).set_index("A")
        # 对时间窗口为 "2s" 的滚动窗口应用最小值函数
        result = df.rolling("2s").min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 4]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间窗口为 "5s" 的滚动窗口应用最小值函数
        result = df.rolling("5s").min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 3]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

    # 测试处理不规则时间序列的最小值功能
    def test_ragged_min(self, ragged):
        # 将输入的不规则时间序列赋给变量 df
        df = ragged

        # 对时间窗口为 "1s" 的滚动窗口应用最小值函数，保留每个时间点的最小值
        result = df.rolling(window="1s", min_periods=1).min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间窗口为 "2s" 的滚动窗口应用最小值函数，保留每个时间点的最小值
        result = df.rolling(window="2s", min_periods=1).min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [0.0, 1, 1, 3, 3]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间窗口为 "5s" 的滚动窗口应用最小值函数，保留每个时间点的最小值
        result = df.rolling(window="5s", min_periods=1).min()
        # 创建期望的结果 DataFrame，与原始数据框一致，但对列 "B" 进行了更新
        expected = df.copy()
        expected["B"] = [0.0, 0, 0, 1, 1]
        # 使用测试工具比较计算结果和期望结果，确认它们是否相等
        tm.assert_frame_equal(result, expected)

    # 测试性能相关的最小值功能
    def test_perf_min(self):
        # 设置生成随机数据的数量
        N = 10000

        # 创建一个包含随机数列 "B" 的 DataFrame，随机数来自正态分布
        dfp = DataFrame(
            {"B": np.random.default_rng(2).standard_normal(N)},
            index=date_range("20130101", periods=N, freq="s"),
        )
        # 对时间窗口为 2 个单位（观测点）的滚动窗口应用最小值函数，至少需要1个观测点来计算结果
        expected = dfp.rolling(2, min_periods=1).min()
        result = dfp.rolling("2s").min()
        # 使用断言检查计算结果是否在期望结果的误差范围内
        assert ((result - expected) < 0.01).all().all()

        # 对时间窗口为 200 个单位（观测点）的滚动窗口应用最小值函数，至少需要1个观测点来计算结果
        expected = dfp.rolling(200, min_periods=1).min()
        result = dfp.rolling("200s").min()
        # 使用断言检查计算结果是否在期望结果的误差范围内
        assert ((result - expected) < 0.01).all().all()
    # 定义一个测试方法，用于测试不规则时间序列的最大滚动窗口操作
    def test_ragged_max(self, ragged):
        # 将输入的不规则时间序列赋值给变量 df
        df = ragged

        # 对时间序列进行滚动窗口为1秒的最大值计算
        result = df.rolling(window="1s", min_periods=1).max()
        # 创建预期结果，将序列复制到 expected，并将列 'B' 的值替换为 [0.0, 1, 2, 3, 4]
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        # 使用测试工具比较计算结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 对时间序列进行滚动窗口为2秒的最大值计算，与前述过程类似
        result = df.rolling(window="2s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        # 对时间序列进行滚动窗口为5秒的最大值计算，与前述过程类似
        result = df.rolling(window="5s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, op, result_data",
        [
            # 定义一系列参数化测试，涵盖不同的时间频率和操作
            ("ms", "min", [0.0] * 10),
            ("ms", "mean", [0.0] * 9 + [2.0 / 9]),
            ("ms", "max", [0.0] * 9 + [2.0]),
            ("s", "min", [0.0] * 10),
            ("s", "mean", [0.0] * 9 + [2.0 / 9]),
            ("s", "max", [0.0] * 9 + [2.0]),
            ("min", "min", [0.0] * 10),
            ("min", "mean", [0.0] * 9 + [2.0 / 9]),
            ("min", "max", [0.0] * 9 + [2.0]),
            ("h", "min", [0.0] * 10),
            ("h", "mean", [0.0] * 9 + [2.0 / 9]),
            ("h", "max", [0.0] * 9 + [2.0]),
            ("D", "min", [0.0] * 10),
            ("D", "mean", [0.0] * 9 + [2.0 / 9]),
            ("D", "max", [0.0] * 9 + [2.0]),
        ],
    )
    # 定义参数化测试方法，用于测试不同频率和操作下的滚动窗口计算结果
    def test_freqs_ops(self, freq, op, result_data):
        # 创建一个时间索引，从 "2018-1-1 01:00:00" 开始，频率为给定的 freq，共 10 个时间点
        index = date_range(start="2018-1-1 01:00:00", freq=f"1{freq}", periods=10)
        # 创建一个 Series 对象，所有数据初始化为 0，索引使用上述创建的时间索引，数据类型为 float
        # 在 Series 中设置第二个和倒数第一个位置为 NaN 和 2
        s = Series(data=0, index=index, dtype="float")
        s.iloc[1] = np.nan
        s.iloc[-1] = 2
        # 对 Series 进行滚动窗口为 10*freq 的操作 op，并获取结果
        result = getattr(s.rolling(window=f"10{freq}"), op)()
        # 创建预期结果，使用 result_data 中定义的数据和 index
        expected = Series(data=result_data, index=index)
        # 使用测试工具比较计算结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            # 定义一系列参数化测试，涵盖不同的操作
            "sum",
            "mean",
            "count",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "min",
            "max",
        ],
    )
    # 定义参数化测试方法，用于测试不同操作下的滚动窗口计算结果
    def test_all(self, f, regular):
        # 将 regular 扩展成两倍，得到时间序列 df
        df = regular * 2
        # 创建两个滚动窗口对象，一个使用整数窗口大小，一个使用时间窗口大小（1秒）
        er = df.rolling(window=1)
        r = df.rolling(window="1s")

        # 对 r 执行当前操作 f，获取结果
        result = getattr(r, f)()
        # 对 er 执行当前操作 f，获取预期结果
        expected = getattr(er, f)()
        # 使用测试工具比较计算结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

        # 对 r 执行 quantile 操作，获取结果
        result = r.quantile(0.5)
        # 对 er 执行 quantile 操作，获取预期结果
        expected = er.quantile(0.5)
        # 使用测试工具比较计算结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
    def test_all2(self, arithmetic_win_operators):
        f = arithmetic_win_operators
        # 将算术窗口操作符存储在变量 f 中

        # 创建一个 DataFrame，包含一列为 np.arange(50) 的数据，索引为每小时频率的时间序列
        df = DataFrame(
            {"B": np.arange(50)}, index=date_range("20130101", periods=50, freq="h")
        )
        
        # 选择时间范围在 "09:00" 到 "16:00" 之间的数据
        dft = df.between_time("09:00", "16:00")

        # 创建一个滚动窗口对象 r，窗口大小为 "5h"
        r = dft.rolling(window="5h")

        # 对滚动窗口对象 r 执行由变量 f 指定的操作，并存储结果
        result = getattr(r, f)()

        # 需要分别对每一天的数据进行滚动操作，以便与基于时间的滚动进行比较
        
        # 定义一个函数 agg_by_day，对每天的数据在 "09:00" 到 "16:00" 之间执行滚动操作
        def agg_by_day(x):
            x = x.between_time("09:00", "16:00")
            return getattr(x.rolling(5, min_periods=1), f)()

        # 使用 groupby-apply 将数据按照日期分组，应用 agg_by_day 函数，并去除日期作为索引的级别
        expected = (
            df.groupby(df.index.day).apply(agg_by_day).reset_index(level=0, drop=True)
        )

        # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_rolling_cov_offset(self):
        # GH16058

        # 创建一个时间索引 idx，从 "2017-01-01" 开始，周期为 24 小时，频率为每小时
        idx = date_range("2017-01-01", periods=24, freq="1h")
        
        # 创建一个 Series ss，其值为索引的长度，索引为 idx
        ss = Series(np.arange(len(idx)), index=idx)

        # 对 ss 应用窗口为 "2h" 的滚动协方差计算，并存储结果
        result = ss.rolling("2h").cov()
        
        # 创建一个预期结果 Series，首个值为 NaN，其余值为 0.5，索引为 idx
        expected = Series([np.nan] + [0.5] * (len(idx) - 1), index=idx)
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 对 ss 应用窗口大小为 2 的滚动协方差计算，并存储结果
        expected2 = ss.rolling(2, min_periods=1).cov()
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected2 是否相等
        tm.assert_series_equal(result, expected2)

        # 对 ss 应用窗口为 "3h" 的滚动协方差计算，并存储结果
        result = ss.rolling("3h").cov()
        
        # 创建一个预期结果 Series，首两个值为 NaN 和 0.5，其余值为 1.0，索引为 idx
        expected = Series([np.nan, 0.5] + [1.0] * (len(idx) - 2), index=idx)
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 对 ss 应用窗口大小为 3 的滚动协方差计算，并存储结果
        expected2 = ss.rolling(3, min_periods=1).cov()
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected2 是否相等
        tm.assert_series_equal(result, expected2)

    def test_rolling_on_decreasing_index(self, unit):
        # GH-19248, GH-32385

        # 创建一个时间索引 index，包含多个时间戳
        index = DatetimeIndex(
            [
                Timestamp("20190101 09:00:30"),
                Timestamp("20190101 09:00:27"),
                Timestamp("20190101 09:00:20"),
                Timestamp("20190101 09:00:18"),
                Timestamp("20190101 09:00:10"),
            ]
        ).as_unit(unit)

        # 创建一个 DataFrame df，包含一列名为 "column" 的数据，值为 [3, 4, 4, 5, 6]，索引为 index
        df = DataFrame({"column": [3, 4, 4, 5, 6]}, index=index)
        
        # 对 df 应用窗口为 "5s" 的滚动最小值计算，并存储结果
        result = df.rolling("5s").min()
        
        # 创建一个预期结果 DataFrame，包含一列名为 "column" 的数据，值为 [3.0, 3.0, 4.0, 4.0, 6.0]，索引为 index
        expected = DataFrame({"column": [3.0, 3.0, 4.0, 4.0, 6.0]}, index=index)
        
        # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_empty(self):
        # GH-32385

        # 创建一个空的 DataFrame df，不包含任何数据，索引和列均为空
        df = DataFrame({"column": []}, index=[])
        
        # 对空的 df 应用窗口为 "5s" 的滚动最小值计算，并存储结果
        result = df.rolling("5s").min()
        
        # 创建一个预期结果 DataFrame，包含一列名为 "column" 的空数据，索引也为空
        expected = DataFrame({"column": []}, index=[])
        
        # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试多层索引中的滚动操作
    def test_rolling_on_multi_index_level(self):
        # 用于说明 issue GH-15584
        # 创建一个 DataFrame 对象，包含一列数据从 0 到 5，同时设置多层索引
        df = DataFrame(
            {"column": range(6)},
            index=MultiIndex.from_product(
                [date_range("20190101", periods=3), range(2)], names=["date", "seq"]
            ),
        )
        # 对 df 应用滚动窗口为 "10d" 的滚动操作，按照索引中的 "date" 列进行计算求和
        result = df.rolling("10d", on=df.index.get_level_values("date")).sum()
        # 创建一个预期的 DataFrame 对象，包含预期的结果，索引与 df 保持一致
        expected = DataFrame(
            {"column": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]}, index=df.index
        )
        # 使用测试框架中的方法来断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试处理自然轴错误的情况
def test_nat_axis_error():
    # 创建一个索引列表，其中包含一个时间戳和一个 NaT（Not a Time）值
    idx = [Timestamp("2020"), NaT]
    # 创建一个包含单位矩阵的 DataFrame，使用上述索引作为行索引
    df = DataFrame(np.eye(2), index=idx)
    # 使用 pytest 的断言检查，期望引发 ValueError 异常，异常消息应包含 "index values must not have NaT"
    with pytest.raises(ValueError, match="index values must not have NaT"):
        # 对 DataFrame 进行滚动窗口操作，计算每日均值
        df.rolling("D").mean()


# 如果没有安装 pyarrow 库，则跳过以下测试
@td.skip_if_no("pyarrow")
def test_arrow_datetime_axis():
    # GH 55849
    # 创建一个期望的 Series 对象，包含从0到4的浮点数，使用日期范围作为索引，索引类型为 "timestamp[ns][pyarrow]"
    expected = Series(
        np.arange(5, dtype=np.float64),
        index=Index(
            date_range("2020-01-01", periods=5), dtype="timestamp[ns][pyarrow]"
        ),
    )
    # 对期望的 Series 对象进行滚动窗口操作，计算每日总和
    result = expected.rolling("1D").sum()
    # 使用测试框架中的断言检查计算结果与期望值是否相等
    tm.assert_series_equal(result, expected)
```