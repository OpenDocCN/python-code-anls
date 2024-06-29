# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_sort_values.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下类和函数：
    DatetimeIndex,   # 时间日期索引类
    Index,           # 索引类
    NaT,             # Not a Time（时间戳数据中的缺失值）
    PeriodIndex,     # 周期索引类
    TimedeltaIndex,  # 时间增量索引类
    timedelta_range, # 时间增量范围函数
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块作为 tm

def check_freq_ascending(ordered, orig, ascending):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is generated (or generate-able) with
    period_range/date_range/timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):  # 如果 ordered 是 PeriodIndex 类型
        assert ordered.freq == orig.freq  # 断言 ordered 的频率与 orig 的频率相同
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):  # 如果 ordered 是 DatetimeIndex 或 TimedeltaIndex 类型
        if ascending:
            assert ordered.freq.n == orig.freq.n  # 如果是升序，则断言 ordered 的频率 n 值与 orig 的频率 n 值相同
        else:
            assert ordered.freq.n == -1 * orig.freq.n  # 如果是降序，则断言 ordered 的频率 n 值等于 orig 的频率 n 值的负数

def check_freq_nonmonotonic(ordered, orig):
    """
    Check the expected freq on a PeriodIndex/DatetimeIndex/TimedeltaIndex
    when the original index is _not_ generated (or generate-able) with
    period_range/date_range//timedelta_range.
    """
    if isinstance(ordered, PeriodIndex):  # 如果 ordered 是 PeriodIndex 类型
        assert ordered.freq == orig.freq  # 断言 ordered 的频率与 orig 的频率相同
    elif isinstance(ordered, (DatetimeIndex, TimedeltaIndex)):  # 如果 ordered 是 DatetimeIndex 或 TimedeltaIndex 类型
        assert ordered.freq is None  # 断言 ordered 的频率为 None

class TestSortValues:
    @pytest.fixture(params=[DatetimeIndex, TimedeltaIndex, PeriodIndex])
    def non_monotonic_idx(self, request):
        if request.param is DatetimeIndex:
            return DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])  # 如果请求是 DatetimeIndex 类型，返回一个时间日期索引对象
        elif request.param is PeriodIndex:
            dti = DatetimeIndex(["2000-01-04", "2000-01-01", "2000-01-02"])
            return dti.to_period("D")  # 如果请求是 PeriodIndex 类型，返回一个以天为周期的周期索引对象
        else:
            return TimedeltaIndex(
                ["1 day 00:00:05", "1 day 00:00:01", "1 day 00:00:02"]
            )  # 否则，返回一个时间增量索引对象，包含三个时间增量字符串

    def test_argmin_argmax(self, non_monotonic_idx):
        assert non_monotonic_idx.argmin() == 1  # 断言非单调索引的最小值位置为 1
        assert non_monotonic_idx.argmax() == 0  # 断言非单调索引的最大值位置为 0

    def test_sort_values(self, non_monotonic_idx):
        idx = non_monotonic_idx  # 将参数 non_monotonic_idx 赋给变量 idx
        ordered = idx.sort_values()  # 对索引 idx 进行排序，并赋给变量 ordered
        assert ordered.is_monotonic_increasing  # 断言 ordered 是单调递增的
        ordered = idx.sort_values(ascending=False)  # 对索引 idx 进行降序排序，并赋给变量 ordered
        assert ordered[::-1].is_monotonic_increasing  # 断言 ordered 的逆序是单调递增的

        ordered, dexer = idx.sort_values(return_indexer=True)  # 对索引 idx 进行排序，返回排序结果和排序的索引
        assert ordered.is_monotonic_increasing  # 断言 ordered 是单调递增的
        tm.assert_numpy_array_equal(dexer, np.array([1, 2, 0], dtype=np.intp))  # 使用测试工具断言 dexer 与给定的 NumPy 数组相等

        ordered, dexer = idx.sort_values(return_indexer=True, ascending=False)  # 对索引 idx 进行降序排序，返回排序结果和排序的索引
        assert ordered[::-1].is_monotonic_increasing  # 断言 ordered 的逆序是单调递增的
        tm.assert_numpy_array_equal(dexer, np.array([0, 2, 1], dtype=np.intp))  # 使用测试工具断言 dexer 与给定的 NumPy 数组相等
    # 定义一个方法，用于检查排序后的值与频率的相关性
    def check_sort_values_with_freq(self, idx):
        # 对索引 idx 进行升序排序，并验证排序结果与原始索引相等
        ordered = idx.sort_values()
        tm.assert_index_equal(ordered, idx)
        # 检查排序后的频率是否符合预期（升序）
        check_freq_ascending(ordered, idx, True)

        # 对索引 idx 进行降序排序，并生成预期的降序结果
        ordered = idx.sort_values(ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        # 检查排序后的频率是否符合预期（降序）
        check_freq_ascending(ordered, idx, False)

        # 返回排序后的索引以及索引器，用于进一步验证
        ordered, indexer = idx.sort_values(return_indexer=True)
        tm.assert_index_equal(ordered, idx)
        # 验证索引器是否按预期生成（升序）
        tm.assert_numpy_array_equal(indexer, np.array([0, 1, 2], dtype=np.intp))
        # 检查排序后的频率是否符合预期（升序）
        check_freq_ascending(ordered, idx, True)

        # 返回降序排序后的索引以及索引器，用于进一步验证
        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        expected = idx[::-1]
        tm.assert_index_equal(ordered, expected)
        # 验证索引器是否按预期生成（降序）
        tm.assert_numpy_array_equal(indexer, np.array([2, 1, 0], dtype=np.intp))
        # 检查排序后的频率是否符合预期（降序）
        check_freq_ascending(ordered, idx, False)

    @pytest.mark.parametrize("freq", ["D", "h"])
    # 使用 pytest 参数化装饰器，为不同的频率值执行相同的测试函数
    def test_sort_values_with_freq_timedeltaindex(self, freq):
        # 创建一个时间增量索引 idx，从指定频率开始，生成 3 个时间点，并重命名为 "idx"
        idx = timedelta_range(start=f"1{freq}", periods=3, freq=freq).rename("idx")

        # 调用 check_sort_values_with_freq 方法，验证排序和频率是否正确
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            # 创建日期时间索引 idx，包含三个日期时间字符串，频率为 "D"，并命名为 "idx"
            DatetimeIndex(
                ["2011-01-01", "2011-01-02", "2011-01-03"], freq="D", name="idx"
            ),
            # 创建带时区的日期时间索引 idx，包含三个带时区的日期时间字符串，频率为 "h"，时区为 "Asia/Tokyo"，并命名为 "tzidx"
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                name="tzidx",
                tz="Asia/Tokyo",
            ),
        ],
    )
    # 使用 pytest 参数化装饰器，为不同的日期时间索引执行相同的测试函数
    def test_sort_values_with_freq_datetimeindex(self, idx):
        # 调用 check_sort_values_with_freq 方法，验证排序和频率是否正确
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize("freq", ["D", "2D", "4D"])
    # 使用 pytest 参数化装饰器，为不同的周期索引频率执行相同的测试函数
    def test_sort_values_with_freq_periodindex(self, freq):
        # 创建一个周期索引 idx，包含三个日期字符串，使用给定的频率 freq，命名为 "idx"
        idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03"], freq=freq, name="idx"
        )
        # 调用 check_sort_values_with_freq 方法，验证排序和频率是否正确
        self.check_sort_values_with_freq(idx)

    @pytest.mark.parametrize(
        "idx",
        [
            # 创建年份周期索引 idx，包含三个年份，频率为 "Y"，并命名为 "pidx"
            PeriodIndex(["2011", "2012", "2013"], name="pidx", freq="Y"),
            # 创建整数索引 idx，包含三个整数，用于兼容性检查
            Index([2011, 2012, 2013], name="idx"),  # for compatibility check
        ],
    )
    # 使用 pytest 参数化装饰器，为不同的周期索引执行相同的测试函数
    def test_sort_values_with_freq_periodindex2(self, idx):
        # 调用 check_sort_values_with_freq 方法，验证排序和频率是否正确
        self.check_sort_values_with_freq(idx)
    # 定义一个方法用于检查排序后的值（不包括频率信息）是否符合预期
    def check_sort_values_without_freq(self, idx, expected):
        # 对索引 idx 进行排序，并返回排序后的索引
        ordered = idx.sort_values(na_position="first")
        # 使用测试框架验证 ordered 和 expected 是否相等
        tm.assert_index_equal(ordered, expected)
        # 检查排序后的索引是否非单调
        check_freq_nonmonotonic(ordered, idx)

        # 如果 idx 中没有缺失值
        if not idx.isna().any():
            # 对索引 idx 进行升序排序，并返回排序后的索引
            ordered = idx.sort_values()
            # 使用测试框架验证 ordered 和 expected 是否相等
            tm.assert_index_equal(ordered, expected)
            # 检查排序后的索引是否非单调
            check_freq_nonmonotonic(ordered, idx)

        # 对索引 idx 进行降序排序，并返回排序后的索引
        ordered = idx.sort_values(ascending=False)
        # 使用测试框架验证 ordered 和 expected 的逆序是否相等
        tm.assert_index_equal(ordered, expected[::-1])
        # 检查排序后的索引是否非单调
        check_freq_nonmonotonic(ordered, idx)

        # 对索引 idx 进行排序，并返回排序后的索引以及排序的索引器
        ordered, indexer = idx.sort_values(return_indexer=True, na_position="first")
        # 使用测试框架验证 ordered 和 expected 是否相等
        tm.assert_index_equal(ordered, expected)

        # 预期的索引器
        exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
        # 使用测试框架验证 indexer 是否与 exp 数组相等
        tm.assert_numpy_array_equal(indexer, exp)
        # 检查排序后的索引是否非单调
        check_freq_nonmonotonic(ordered, idx)

        # 如果 idx 中没有缺失值
        if not idx.isna().any():
            # 对索引 idx 进行排序，并返回排序后的索引以及排序的索引器
            ordered, indexer = idx.sort_values(return_indexer=True)
            # 使用测试框架验证 ordered 和 expected 是否相等
            tm.assert_index_equal(ordered, expected)

            # 预期的索引器
            exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
            # 使用测试框架验证 indexer 是否与 exp 数组相等
            tm.assert_numpy_array_equal(indexer, exp)
            # 检查排序后的索引是否非单调
            check_freq_nonmonotonic(ordered, idx)

        # 对索引 idx 进行降序排序，并返回排序后的索引以及排序的索引器
        ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
        # 使用测试框架验证 ordered 和 expected 的逆序是否相等
        tm.assert_index_equal(ordered, expected[::-1])

        # 预期的索引器
        exp = np.array([2, 1, 3, 0, 4], dtype=np.intp)
        # 使用测试框架验证 indexer 是否与 exp 数组相等
        tm.assert_numpy_array_equal(indexer, exp)
        # 检查排序后的索引是否非单调

    # 测试对 TimedeltaIndex 进行排序的方法
    def test_sort_values_without_freq_timedeltaindex(self):
        # GH#10295

        # 创建一个 TimedeltaIndex，指定名称为 idx1，用于测试排序
        idx = TimedeltaIndex(
            ["1 hour", "3 hour", "5 hour", "2 hour ", "1 hour"], name="idx1"
        )
        # 创建期望的 TimedeltaIndex，用于验证排序结果
        expected = TimedeltaIndex(
            ["1 hour", "1 hour", "2 hour", "3 hour", "5 hour"], name="idx1"
        )
        # 调用 check_sort_values_without_freq 方法，验证排序结果是否符合预期
        self.check_sort_values_without_freq(idx, expected)

    # 使用参数化测试对 DatetimeIndex 进行排序的方法
    @pytest.mark.parametrize(
        "index_dates,expected_dates",
        [
            (
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-02", "2011-01-01"],
                ["2011-01-01", "2011-01-01", "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
            (
                [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
            ),
        ],
    )
    # 测试对 DatetimeIndex 进行排序的方法
    def test_sort_values_without_freq_datetimeindex(
        self, index_dates, expected_dates, tz_naive_fixture
    ):
        tz = tz_naive_fixture

        # 创建一个 DatetimeIndex，指定时区为 tz，用于测试排序
        idx = DatetimeIndex(index_dates, tz=tz, name="idx")
        # 创建期望的 DatetimeIndex，用于验证排序结果
        expected = DatetimeIndex(expected_dates, tz=tz, name="idx")

        # 调用 check_sort_values_without_freq 方法，验证排序结果是否符合预期
        self.check_sort_values_without_freq(idx, expected)
    # 使用 pytest 的参数化装饰器，为测试方法提供多组参数进行测试
    @pytest.mark.parametrize(
        "idx,expected",
        [
            (
                # 创建 PeriodIndex 对象，指定日期序列及频率，设置名称为 'idx1'
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx1",
                ),
                # 创建期望的 PeriodIndex 对象，指定日期序列及频率，设置名称为 'idx1'
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx1",
                ),
            ),
            (
                # 创建 PeriodIndex 对象，指定日期序列及频率，设置名称为 'idx2'
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-03",
                        "2011-01-05",
                        "2011-01-02",
                        "2011-01-01",
                    ],
                    freq="D",
                    name="idx2",
                ),
                # 创建期望的 PeriodIndex 对象，指定日期序列及频率，设置名称为 'idx2'
                PeriodIndex(
                    [
                        "2011-01-01",
                        "2011-01-01",
                        "2011-01-02",
                        "2011-01-03",
                        "2011-01-05",
                    ],
                    freq="D",
                    name="idx2",
                ),
            ),
            (
                # 创建 PeriodIndex 对象，包含 NaT（Not a Time）值，指定频率和名称为 'idx3'
                PeriodIndex(
                    [NaT, "2011-01-03", "2011-01-05", "2011-01-02", NaT],
                    freq="D",
                    name="idx3",
                ),
                # 创建期望的 PeriodIndex 对象，处理 NaT 值，指定频率和名称为 'idx3'
                PeriodIndex(
                    [NaT, NaT, "2011-01-02", "2011-01-03", "2011-01-05"],
                    freq="D",
                    name="idx3",
                ),
            ),
            (
                # 创建 PeriodIndex 对象，指定年份序列及频率为年，设置名称为 'pidx'
                PeriodIndex(
                    ["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y"
                ),
                # 创建期望的 PeriodIndex 对象，指定年份序列及频率为年，设置名称为 'pidx'
                PeriodIndex(
                    ["2011", "2011", "2012", "2013", "2015"], name="pidx", freq="Y"
                ),
            ),
            (
                # 兼容性检查
                # 创建 Index 对象，包含整数序列，设置名称为 'idx'
                Index([2011, 2013, 2015, 2012, 2011], name="idx"),
                # 创建期望的 Index 对象，包含整数序列，设置名称为 'idx'
                Index([2011, 2011, 2012, 2013, 2015], name="idx"),
            ),
        ],
    )
    # 测试方法：验证在没有指定频率的情况下，使用 check_sort_values_without_freq 方法进行排序
    def test_sort_values_without_freq_periodindex(self, idx, expected):
        self.check_sort_values_without_freq(idx, expected)
    # 定义一个测试方法，用于测试在没有频率和自然时间值的情况下排序值的行为
    def test_sort_values_without_freq_periodindex_nat(self):
        # 创建一个 PeriodIndex 对象，包含字符串日期 "2011", "2013", "NaT", "2011"，命名为 "pidx"，频率为 "D"
        idx = PeriodIndex(["2011", "2013", "NaT", "2011"], name="pidx", freq="D")
        # 预期的排序后的 PeriodIndex 对象，包含 "NaT", "2011", "2011", "2013"，命名为 "pidx"，频率为 "D"
        expected = PeriodIndex(["NaT", "2011", "2011", "2013"], name="pidx", freq="D")

        # 对 idx 进行排序，缺失值（NaT）放在首位，得到排序后的 PeriodIndex 对象
        ordered = idx.sort_values(na_position="first")
        # 断言排序后的结果与预期的结果相等
        tm.assert_index_equal(ordered, expected)
        # 检查排序后的结果是否满足非单调频率的要求
        check_freq_nonmonotonic(ordered, idx)

        # 对 idx 进行降序排序，得到排序后的 PeriodIndex 对象
        ordered = idx.sort_values(ascending=False)
        # 断言降序排序后的结果与预期的结果相等（预期结果倒序）
        tm.assert_index_equal(ordered, expected[::-1])
        # 检查排序后的结果是否满足非单调频率的要求
        check_freq_nonmonotonic(ordered, idx)
def test_order_stability_compat():
    # GH#35922. sort_values is stable both for normal and datetime-like Index
    # 创建一个周期性索引对象，包含特定年份字符串，指定名称和频率为每年一次
    pidx = PeriodIndex(["2011", "2013", "2015", "2012", "2011"], name="pidx", freq="Y")
    # 创建一个普通整数索引对象，包含特定年份整数，指定名称为'idx'
    iidx = Index([2011, 2013, 2015, 2012, 2011], name="idx")
    # 对周期性索引对象进行降序排序，并返回排序后的结果和排序的索引
    ordered1, indexer1 = pidx.sort_values(return_indexer=True, ascending=False)
    # 对普通整数索引对象进行降序排序，并返回排序后的结果和排序的索引
    ordered2, indexer2 = iidx.sort_values(return_indexer=True, ascending=False)
    # 使用测试工具库（tm）来断言两个排序索引数组是否完全相等
    tm.assert_numpy_array_equal(indexer1, indexer2)
```