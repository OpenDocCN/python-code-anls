# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_period_range.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入以下模块
    NaT,  # NaT 表示不可用的时间戳
    Period,  # Period 表示时间段
    PeriodIndex,  # PeriodIndex 表示时间段索引
    date_range,  # date_range 生成日期范围
    period_range,  # period_range 生成时间段范围
)
import pandas._testing as tm  # 导入 pandas 内部的测试模块

class TestPeriodRangeKeywords:
    def test_required_arguments(self):
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range("2011-1-1", "2012-1-1", "B")

    def test_required_arguments2(self):
        start = Period("02-Apr-2005", "D")  # 创建一个日期为 2005 年 4 月 2 日的 Period 对象
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start=start)

    def test_required_arguments3(self):
        # not enough params
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start="2017Q1")

        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(end="2017Q1")

        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(periods=5)

        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range()

    def test_required_arguments_too_many(self):
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start="2017Q1", end="2018Q1", periods=8, freq="Q")

    def test_start_end_non_nat(self):
        # start/end NaT
        msg = "start and end must not be NaT"
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start=NaT, end="2018Q1")
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start=NaT, end="2018Q1", freq="Q")

        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start="2017Q1", end=NaT)
        with pytest.raises(ValueError, match=msg):  # 断言期望捕获 ValueError 异常并匹配指定消息
            period_range(start="2017Q1", end=NaT, freq="Q")

    def test_periods_requires_integer(self):
        # invalid periods param
        msg = "periods must be an integer, got foo"
        with pytest.raises(TypeError, match=msg):  # 断言期望捕获 TypeError 异常并匹配指定消息
            period_range(start="2017Q1", periods="foo")


class TestPeriodRange:
    @pytest.mark.parametrize(  # 使用 pytest 参数化标记来传递不同的参数组合进行测试
        "freq_offset, freq_period",
        [
            ("D", "D"),  # 每日频率测试
            ("W", "W"),  # 每周频率测试
            ("QE", "Q"),  # 每季度末频率测试
            ("YE", "Y"),  # 每年末频率测试
        ],
    )
    # 测试从字符串构建 PeriodIndex 对象的方法，使用给定的频率偏移量和频率周期
    def test_construction_from_string(self, freq_offset, freq_period):
        # 非空情况下的测试
        # 创建预期的日期范围对象，转换为 PeriodIndex 对象
        expected = date_range(
            start="2017-01-01", periods=5, freq=freq_offset, name="foo"
        ).to_period()
        # 提取日期范围的起始日期和结束日期字符串表示
        start, end = str(expected[0]), str(expected[-1])

        # 根据起始日期、结束日期和频率周期创建 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, end=end, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # 根据起始日期、指定周期数和频率周期创建 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期、指定周期数和频率周期创建 PeriodIndex 对象，进行结果比较
        result = period_range(end=end, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # 空情况下的测试
        # 创建空的 PeriodIndex 对象，指定频率周期
        expected = PeriodIndex([], freq=freq_period, name="foo")

        # 根据起始日期、指定周期数和频率周期创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期、指定周期数和频率周期创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(end=end, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期和起始日期相同、频率周期创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(start=end, end=start, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

    # 测试从字符串构建 PeriodIndex 对象的方法，使用月度频率
    def test_construction_from_string_monthly(self):
        # 非空情况下的测试
        # 创建预期的日期范围对象，转换为 PeriodIndex 对象，使用"ME"（MonthEnd）频率
        expected = date_range(
            start="2017-01-01", periods=5, freq="ME", name="foo"
        ).to_period()
        # 提取日期范围的起始日期和结束日期字符串表示
        start, end = str(expected[0]), str(expected[-1])

        # 根据起始日期、结束日期和"M"（月度）频率创建 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # 根据起始日期、指定周期数和"M"（月度）频率创建 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期、指定周期数和"M"（月度）频率创建 PeriodIndex 对象，进行结果比较
        result = period_range(end=end, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # 空情况下的测试
        # 创建空的 PeriodIndex 对象，使用"M"（月度）频率
        expected = PeriodIndex([], freq="M", name="foo")

        # 根据起始日期、指定周期数和"M"（月度）频率创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(start=start, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期、指定周期数和"M"（月度）频率创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(end=end, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # 根据结束日期和起始日期相同、"M"（月度）频率创建空的 PeriodIndex 对象，进行结果比较
        result = period_range(start=end, end=start, freq="M", name="foo")
        tm.assert_index_equal(result, expected)
    def test_construction_from_period(self):
        # 测试从 Period 对象构造 PeriodIndex

        # 设置起始和结束的 Period 对象，频率为季度
        start, end = Period("2017Q1", freq="Q"), Period("2018Q1", freq="Q")
        # 期望的结果是一个日期范围，频率为每月末，起始为"2017-03-31"，结束为"2018-03-31"，名称为"foo"
        expected = date_range(
            start="2017-03-31", end="2018-03-31", freq="ME", name="foo"
        ).to_period()
        # 使用 period_range 函数生成结果，频率为每月，名称为"foo"
        result = period_range(start=start, end=end, freq="M", name="foo")
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 设置起始和结束的 Period 对象，频率为月份
        start = Period("2017-1", freq="M")
        end = Period("2019-12", freq="M")
        # 期望的结果是一个日期范围，频率为每季度末，起始为"2017-01-31"，结束为"2019-12-31"，名称为"foo"
        expected = date_range(
            start="2017-01-31", end="2019-12-31", freq="QE", name="foo"
        ).to_period()
        # 使用 period_range 函数生成结果，频率为每季度，名称为"foo"
        result = period_range(start=start, end=end, freq="Q", name="foo")
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 测试 issue #21793
        start = Period("2017Q1", freq="Q")
        end = Period("2018Q1", freq="Q")
        # 使用 period_range 函数生成 PeriodIndex，频率为季度，名称为"foo"
        idx = period_range(start=start, end=end, freq="Q", name="foo")
        # 检查结果是否与其值相等
        result = idx == idx.values
        # 期望的结果是一个全为 True 的 numpy 数组
        expected = np.array([True, True, True, True, True])
        # 断言 numpy 数组相等
        tm.assert_numpy_array_equal(result, expected)

        # 测试空情况
        # 期望的结果是一个空的 PeriodIndex，频率为周，名称为"foo"
        expected = PeriodIndex([], freq="W", name="foo")

        # 使用 period_range 函数生成结果，从指定起始开始的 0 个 Period，频率为周，名称为"foo"
        result = period_range(start=start, periods=0, freq="W", name="foo")
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 使用 period_range 函数生成结果，到指定结束的 0 个 Period，频率为周，名称为"foo"
        result = period_range(end=end, periods=0, freq="W", name="foo")
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

        # 使用 period_range 函数生成结果，从结束到起始的 0 个 Period，频率为周，名称为"foo"
        result = period_range(start=end, end=start, freq="W", name="foo")
        # 断言结果与期望相等
        tm.assert_index_equal(result, expected)

    def test_mismatched_start_end_freq_raises(self):
        # 测试当起始和结束的频率不匹配时是否会引发异常

        # 设置周末频率的结束 Period 对象
        end_w = Period("2006-12-31", "1W")
        # 引发未来警告，提示 BDay 频率的 Period 已弃用
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            # 设置工作日频率的起始和结束 Period 对象
            start_b = Period("02-Apr-2005", "B")
            end_b = Period("2005-05-01", "B")

        # 设置错误消息
        msg = "start and end must have same freq"
        # 断言引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                # 使用 period_range 函数生成 PeriodIndex，起始和结束频率不匹配
                period_range(start=start_b, end=end_w)

        # 没有频率不匹配时应正常运行
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            # 使用 period_range 函数生成 PeriodIndex，起始和结束频率匹配
            period_range(start=start_b, end=end_b)
class TestPeriodRangeDisallowedFreqs:
    # 定义测试类 TestPeriodRangeDisallowedFreqs

    def test_constructor_U(self):
        # 定义测试方法 test_constructor_U，用于测试未定义期间
        # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match="Invalid frequency: X"):
            period_range("2007-1-1", periods=500, freq="X")

    @pytest.mark.parametrize("freq_depr", ["2H", "2MIN", "2S", "2US", "2NS"])
    # 使用 pytest.mark.parametrize 定义参数化测试，测试频率的不推荐用法
    def test_uppercase_freq_deprecated_from_time_series(self, freq_depr):
        # 定义测试方法 test_uppercase_freq_deprecated_from_time_series，测试大写频率的不推荐用法
        # 设置警告消息，指示将来版本中会删除这些用法
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a " \
              f"future version. Please use '{freq_depr.lower()[1:]}' instead."

        # 使用 assert_produces_warning 检查是否引发 FutureWarning 警告，并匹配特定警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range("2020-01-01 00:00:00 00:00", periods=2, freq=freq_depr)

    @pytest.mark.parametrize("freq", ["2m", "2q-sep", "2y"])
    # 使用 pytest.mark.parametrize 定义参数化测试，测试小写频率的错误用法
    def test_lowercase_freq_from_time_series_raises(self, freq):
        # 定义测试方法 test_lowercase_freq_from_time_series_raises，测试小写频率引发错误的情况
        # 设置错误消息，指示无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")

    @pytest.mark.parametrize("freq", ["2A", "2a", "2A-AUG", "2A-aug"])
    # 使用 pytest.mark.parametrize 定义参数化测试，测试年度频率的错误用法
    def test_A_raises_from_time_series(self, freq):
        # 定义测试方法 test_A_raises_from_time_series，测试年度频率引发错误的情况
        # 设置错误消息，指示无效的频率
        msg = f"Invalid frequency: {freq}"

        # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")

    @pytest.mark.parametrize("freq", ["2w"])
    # 使用 pytest.mark.parametrize 定义参数化测试，测试小写周频率的不推荐用法
    def test_lowercase_freq_from_time_series_deprecated(self, freq):
        # 定义测试方法 test_lowercase_freq_from_time_series_deprecated，测试小写周频率的不推荐用法
        # 设置警告消息，指示将来版本中会删除这些用法
        msg = f"'{freq[1:]}' is deprecated and will be removed in a " \
              f"future version. Please use '{freq.upper()[1:]}' instead."

        # 使用 assert_produces_warning 检查是否引发 FutureWarning 警告，并匹配特定警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range(freq=freq, start="1/1/2001", end="12/1/2009")
```