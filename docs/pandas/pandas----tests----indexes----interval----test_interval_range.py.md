# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_interval_range.py`

```
# 从 datetime 模块中导入 timedelta 类，用于处理时间间隔
from datetime import timedelta

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试
import pytest

# 从 pandas.core.dtypes.common 模块中导入 is_integer 函数，用于检查是否为整数类型
from pandas.core.dtypes.common import is_integer

# 从 pandas 模块中导入多个类和函数
from pandas import (
    DateOffset,         # 处理日期偏移的类
    Interval,           # 表示时间间隔的类
    IntervalIndex,      # 表示时间间隔索引的类
    Timedelta,          # 表示时间差的类
    Timestamp,          # 表示时间戳的类
    date_range,         # 生成日期范围的函数
    interval_range,     # 生成时间间隔范围的函数
    timedelta_range,    # 生成时间差范围的函数
)

# 导入 pandas._testing 库，用于辅助测试
import pandas._testing as tm

# 从 pandas.tseries.offsets 模块中导入 Day 类，表示一天的时间偏移
from pandas.tseries.offsets import Day

# 定义 pytest 的装饰器，用于生成参数化的测试数据，参数为 None 或 "foo"
@pytest.fixture(params=[None, "foo"])
def name(request):
    return request.param

# 定义 TestIntervalRange 测试类
class TestIntervalRange:
    
    # 参数化测试方法，测试 interval_range 构造函数的数值输入
    @pytest.mark.parametrize("freq, periods", [(1, 100), (2.5, 40), (5, 20), (25, 4)])
    def test_constructor_numeric(self, closed, name, freq, periods):
        # 定义起始值和结束值
        start, end = 0, 100
        
        # 生成间隔的断点数组，步长为 freq
        breaks = np.arange(101, step=freq)
        
        # 使用断点数组创建预期的 IntervalIndex 对象
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # 使用 interval_range 函数创建 IntervalIndex 对象，方法：从 start 到 end，使用 freq 作为间隔步长
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

        # 使用 interval_range 函数创建 IntervalIndex 对象，方法：从 start 到 periods，使用 freq 作为间隔步长
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 使用 interval_range 函数创建 IntervalIndex 对象，方法：从 end 到 periods，使用 freq 作为间隔步长
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 使用 interval_range 函数创建 IntervalIndex 对象，方法：从 start 到 end，使用 periods 个数，进行线性插值
        result = interval_range(
            start=start, end=end, periods=periods, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

    # 参数化测试方法，测试 interval_range 构造函数的时间相关输入
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    @pytest.mark.parametrize(
        "freq, periods", [("D", 364), ("2D", 182), ("22D18h", 16), ("ME", 11)]
    )
    def test_constructor_timestamp(self, closed, name, freq, periods, tz):
        # 创建起始和结束时间戳对象
        start, end = Timestamp("20180101", tz=tz), Timestamp("20181231", tz=tz)
        # 根据给定的起始时间、结束时间和频率生成日期范围
        breaks = date_range(start=start, end=end, freq=freq)
        # 根据生成的日期范围创建预期的区间索引对象
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # 根据起始时间、结束时间、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 根据起始时间、数据点数、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 根据结束时间、数据点数、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # GH 20976: 如果日期范围的频率不为1且时区为None，根据起始时间、结束时间、数据点数以及其他参数创建区间索引对象，并进行断言验证
        if not breaks.freq.n == 1 and tz is None:
            result = interval_range(
                start=start, end=end, periods=periods, name=name, closed=closed
            )
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, periods", [("D", 100), ("2D12h", 40), ("5D", 20), ("25D", 4)]
    )
    def test_constructor_timedelta(self, closed, name, freq, periods):
        # 创建起始和结束时间差对象
        start, end = Timedelta("0 days"), Timedelta("100 days")
        # 根据给定的起始时间差、结束时间差和频率生成时间差范围
        breaks = timedelta_range(start=start, end=end, freq=freq)
        # 根据生成的时间差范围创建预期的区间索引对象
        expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)

        # 根据起始时间差、结束时间差、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            start=start, end=end, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 根据起始时间差、数据点数、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            start=start, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # 根据结束时间差、数据点数、频率以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            end=end, periods=periods, freq=freq, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)

        # GH 20976: 根据起始时间差、结束时间差、数据点数以及其他参数创建区间索引对象，并进行断言验证
        result = interval_range(
            start=start, end=end, periods=periods, name=name, closed=closed
        )
        tm.assert_index_equal(result, expected)
    @pytest.mark.parametrize(
        "start, end, freq, expected_endpoint",
        [
            (0, 10, 3, 9),  # 设置起始、结束、频率及预期结果，验证早期截断的情况
            (0, 10, 1.5, 9),  # 同上，使用不同的频率
            (0.5, 10, 3, 9.5),  # 同上，起始包含小数点的情况
            (Timedelta("0D"), Timedelta("10D"), "2D4h", Timedelta("8D16h")),  # 使用时间增量来设置起始、结束及频率，验证预期结果
            (
                Timestamp("2018-01-01"),
                Timestamp("2018-02-09"),
                "MS",
                Timestamp("2018-02-01"),
            ),  # 使用时间戳设置起始、结束及频率，验证预期结果
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-20", tz="US/Eastern"),
                "5D12h",
                Timestamp("2018-01-17 12:00:00", tz="US/Eastern"),
            ),  # 同上，包含时区信息
        ],
    )
    def test_early_truncation(self, start, end, freq, expected_endpoint):
        # 如果频率导致结束点被跳过，则索引会早期截断
        result = interval_range(start=start, end=end, freq=freq)
        result_endpoint = result.right[-1]
        assert result_endpoint == expected_endpoint

    @pytest.mark.parametrize(
        "start, end, freq",
        [(0.5, None, None), (None, 4.5, None), (0.5, None, 1.5), (None, 6.5, 1.5)],
    )
    def test_no_invalid_float_truncation(self, start, end, freq):
        # GH 21161
        # 如果频率为None，则使用固定的浮点数列表，否则使用另一个固定的浮点数列表
        if freq is None:
            breaks = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            breaks = [0.5, 2.0, 3.5, 5.0, 6.5]
        expected = IntervalIndex.from_breaks(breaks)

        result = interval_range(start=start, end=end, periods=4, freq=freq)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "start, mid, end",
        [
            (
                Timestamp("2018-03-10", tz="US/Eastern"),
                Timestamp("2018-03-10 23:30:00", tz="US/Eastern"),
                Timestamp("2018-03-12", tz="US/Eastern"),
            ),  # 设置起始、中间、结束时间戳，验证在夏令时转换期间的线性空间分布行为
            (
                Timestamp("2018-11-03", tz="US/Eastern"),
                Timestamp("2018-11-04 00:30:00", tz="US/Eastern"),
                Timestamp("2018-11-05", tz="US/Eastern"),
            ),  # 同上，不同的日期和时间
        ],
    )
    def test_linspace_dst_transition(self, start, mid, end):
        # GH 20976: linspace 的行为根据起始点、结束点和期间数定义，考虑在夏令时转换期间增加/减少的小时
        start = start.as_unit("ns")
        mid = mid.as_unit("ns")
        end = end.as_unit("ns")
        result = interval_range(start=start, end=end, periods=2)
        expected = IntervalIndex.from_breaks([start, mid, end])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("freq", [2, 2.0])
    @pytest.mark.parametrize("end", [10, 10.0])
    @pytest.mark.parametrize("start", [0, 0.0])
    # 测试函数，用于检查浮点数子类型的情况，根据不同的起始、结束和频率参数进行测试
    def test_float_subtype(self, start, end, freq):
        # 如果 start/end/freq 中有任何一个是浮点数，则结果会有浮点数子类型，即使所有结果端点都能安全地向上转型为整数

        # 根据 start/end/freq 定义索引
        index = interval_range(start=start, end=end, freq=freq)
        # 获取索引的数据类型子类型
        result = index.dtype.subtype
        # 如果 start + end + freq 是整数，则期望结果为 "int64"，否则为 "float64"
        expected = "int64" if is_integer(start + end + freq) else "float64"
        assert result == expected

        # 根据 start/periods/freq 定义索引
        index = interval_range(start=start, periods=5, freq=freq)
        # 获取索引的数据类型子类型
        result = index.dtype.subtype
        # 如果 start + freq 是整数，则期望结果为 "int64"，否则为 "float64"
        expected = "int64" if is_integer(start + freq) else "float64"
        assert result == expected

        # 根据 end/periods/freq 定义索引
        index = interval_range(end=end, periods=5, freq=freq)
        # 获取索引的数据类型子类型
        result = index.dtype.subtype
        # 如果 end + freq 是整数，则期望结果为 "int64"，否则为 "float64"
        expected = "int64" if is_integer(end + freq) else "float64"
        assert result == expected

        # GH 20976: 根据 start/end/periods 定义线性间隔的行为
        index = interval_range(start=start, end=end, periods=5)
        # 获取索引的数据类型子类型
        result = index.dtype.subtype
        # 如果 start + end 是整数，则期望结果为 "int64"，否则为 "float64"
        expected = "int64" if is_integer(start + end) else "float64"
        assert result == expected

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            (np.int8(1), np.int8(10), np.dtype("int8")),
            (np.int8(1), np.float16(10), np.dtype("float64")),
            (np.float32(1), np.float32(10), np.dtype("float32")),
            (1, 10, np.dtype("int64")),
            (1, 10.0, np.dtype("float64")),
        ],
    )
    # 测试函数，用于检查区间的数据类型
    def test_interval_dtype(self, start, end, expected):
        # 调用 interval_range 函数，获取其返回的索引数据类型的子类型
        result = interval_range(start=start, end=end).dtype.subtype
        # 断言结果与预期的数据类型相等
        assert result == expected

    # 测试函数，用于检查带有小数周期的区间范围
    def test_interval_range_fractional_period(self):
        # 周期值为浮点数的情况
        msg = "periods must be an integer, got 10.5"
        # 创建一个时间戳对象
        ts = Timestamp("2024-03-25")
        # 使用 pytest 断言，验证期望的异常信息是否被触发
        with pytest.raises(TypeError, match=msg):
            interval_range(ts, periods=10.5)
    # 测试 interval_range 构造函数的覆盖率
    def test_constructor_coverage(self):
        # 创建等效的时间戳样式的起始和结束时间
        start, end = Timestamp("2017-01-01"), Timestamp("2017-01-15")
        expected = interval_range(start=start, end=end)

        # 使用时间戳样式的起始和结束时间创建 interval_range 对象
        result = interval_range(start=start.to_pydatetime(), end=end.to_pydatetime())
        tm.assert_index_equal(result, expected)

        # 使用时间戳样式的起始和结束时间创建 interval_range 对象
        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)

        # 创建等效的频率与时间戳
        equiv_freq = [
            "D",
            Day(),
            Timedelta(days=1),
            timedelta(days=1),
            DateOffset(days=1),
        ]
        for freq in equiv_freq:
            # 使用不同频率创建 interval_range 对象
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)

        # 创建等效的时间差样式的起始和结束时间
        start, end = Timedelta(days=1), Timedelta(days=10)
        expected = interval_range(start=start, end=end)

        # 使用时间差样式的起始和结束时间创建 interval_range 对象
        result = interval_range(start=start.to_pytimedelta(), end=end.to_pytimedelta())
        tm.assert_index_equal(result, expected)

        # 使用时间差样式的起始和结束时间创建 interval_range 对象
        result = interval_range(start=start.asm8, end=end.asm8)
        tm.assert_index_equal(result, expected)

        # 创建等效的频率与时间差
        equiv_freq = ["D", Day(), Timedelta(days=1), timedelta(days=1)]
        for freq in equiv_freq:
            # 使用不同频率创建 interval_range 对象
            result = interval_range(start=start, end=end, freq=freq)
            tm.assert_index_equal(result, expected)

    # 测试浮点型频率
    def test_float_freq(self):
        # GH 54477
        # 使用浮点型频率创建 interval_range 对象
        result = interval_range(0, 1, freq=0.1)
        expected = IntervalIndex.from_breaks([0 + 0.1 * n for n in range(11)])
        tm.assert_index_equal(result, expected)

        # 使用浮点型频率创建 interval_range 对象
        result = interval_range(0, 1, freq=0.6)
        expected = IntervalIndex.from_breaks([0, 0.6])
        tm.assert_index_equal(result, expected)
```