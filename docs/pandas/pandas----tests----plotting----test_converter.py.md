# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_converter.py`

```
# 导入必要的模块和库
from datetime import (
    date,
    datetime,
)
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import sys  # 导入 sys 模块，用于访问系统相关的参数和功能

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

import pandas._config.config as cf  # 导入 pandas 的配置模块

from pandas._libs.tslibs import to_offset  # 导入 pandas 时间序列相关的模块

from pandas import (  # 导入 pandas 的核心对象和函数
    Index,
    Period,
    PeriodIndex,
    Series,
    Timestamp,
    arrays,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试相关的模块

from pandas.plotting import (  # 导入 pandas 绘图模块中的一些功能
    deregister_matplotlib_converters,
    register_matplotlib_converters,
)
from pandas.tseries.offsets import (  # 导入 pandas 时间序列偏移模块中的一些偏移量
    Day,
    Micro,
    Milli,
    Second,
)

plt = pytest.importorskip("matplotlib.pyplot")  # 导入并检查是否安装了 matplotlib.pyplot
dates = pytest.importorskip("matplotlib.dates")  # 导入并检查是否安装了 matplotlib.dates
units = pytest.importorskip("matplotlib.units")  # 导入并检查是否安装了 matplotlib.units

from pandas.plotting._matplotlib import converter  # 导入 pandas 绘图模块中的转换器

@pytest.mark.single_cpu
def test_registry_mpl_resets():
    # 检查 matplotlib 转换器是否正确重置（参见 issue #27481）
    code = (
        "import matplotlib.units as units; "
        "import matplotlib.dates as mdates; "
        "n_conv = len(units.registry); "
        "import pandas as pd; "
        "pd.plotting.register_matplotlib_converters(); "
        "pd.plotting.deregister_matplotlib_converters(); "
        "assert len(units.registry) == n_conv"
    )
    call = [sys.executable, "-c", code]  # 构造调用子进程的命令
    subprocess.check_output(call)  # 执行命令并获取输出结果


def test_timtetonum_accepts_unicode():
    assert converter.time2num("00:01") == converter.time2num("00:01")


class TestRegistration:
    @pytest.mark.single_cpu
    def test_dont_register_by_default(self):
        # 在子进程中运行以确保干净的状态
        code = (
            "import matplotlib.units; "
            "import pandas as pd; "
            "units = dict(matplotlib.units.registry); "
            "assert pd.Timestamp not in units"
        )
        call = [sys.executable, "-c", code]  # 构造调用子进程的命令
        assert subprocess.check_call(call) == 0  # 执行命令并检查返回状态是否为 0

    def test_registering_no_warning(self):
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # 设置为“warn”状态，以防这不是第一次运行测试
        register_matplotlib_converters()
        ax.plot(s.index, s.values)

    def test_pandas_plots_register(self):
        s = Series(range(12), index=date_range("2017", periods=12))
        # 设置为“warn”状态，以防这不是第一次运行测试
        with tm.assert_produces_warning(None) as w:
            s.plot()

        assert len(w) == 0

    def test_matplotlib_formatters(self):
        # 无法对起始状态做出任何断言。
        # 我们检查关闭转换器后它是否被移除，并且打开后是否被恢复。
        with cf.option_context("plotting.matplotlib.register_converters", True):
            with cf.option_context("plotting.matplotlib.register_converters", False):
                assert Timestamp not in units.registry
            assert Timestamp in units.registry
    def test_option_no_warning(self):
        s = Series(range(12), index=date_range("2017", periods=12))
        _, ax = plt.subplots()

        # Test without registering first, no warning
        # 使用上下文管理器测试不注册的情况，此时不应产生警告
        with cf.option_context("plotting.matplotlib.register_converters", False):
            ax.plot(s.index, s.values)

        # Now test with registering
        # 现在测试注册后的情况
        register_matplotlib_converters()
        with cf.option_context("plotting.matplotlib.register_converters", False):
            ax.plot(s.index, s.values)

    def test_registry_resets(self):
        # make a copy, to reset to
        # 创建一个副本，用于重置
        original = dict(units.registry)

        try:
            # get to a known state
            # 将 registry 清空，设置日期转换器
            units.registry.clear()
            date_converter = dates.DateConverter()
            units.registry[datetime] = date_converter
            units.registry[date] = date_converter

            register_matplotlib_converters()
            # 断言日期的注册状态
            assert units.registry[date] is not date_converter
            deregister_matplotlib_converters()
            assert units.registry[date] is date_converter

        finally:
            # restore original stater
            # 恢复原始状态
            units.registry.clear()
            for k, v in original.items():
                units.registry[k] = v
class TestDateTimeConverter:
    # 创建一个 pytest fixture，返回 DatetimeConverter 的实例 dtc
    @pytest.fixture
    def dtc(self):
        return converter.DatetimeConverter()

    # 测试 DatetimeConverter.convert 是否接受 Unicode 输入
    def test_convert_accepts_unicode(self, dtc):
        r1 = dtc.convert("2000-01-01 12:22", None, None)
        r2 = dtc.convert("2000-01-01 12:22", None, None)
        assert r1 == r2, "DatetimeConverter.convert should accept unicode"

    # 测试不同类型日期转换是否正确
    def test_conversion(self, dtc):
        rs = dtc.convert(["2012-1-1"], None, None)[0]
        xp = dates.date2num(datetime(2012, 1, 1))
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(date(2012, 1, 1), None, None)
        assert rs == xp

        rs = dtc.convert("2012-1-1", None, None)
        assert rs == xp

        rs = dtc.convert(Timestamp("2012-1-1"), None, None)
        assert rs == xp

        # 测试 datetime64 类型的转换 (GH8614)
        rs = dtc.convert("2012-01-01", None, None)
        assert rs == xp

        rs = dtc.convert("2012-01-01 00:00:00+0000", None, None)
        assert rs == xp

        rs = dtc.convert(
            np.array(["2012-01-01 00:00:00+0000", "2012-01-02 00:00:00+0000"]),
            None,
            None,
        )
        assert rs[0] == xp

        # 测试带时区信息的日期转换
        ts = Timestamp("2012-01-01").tz_localize("UTC").tz_convert("US/Eastern")
        rs = dtc.convert(ts, None, None)
        assert rs == xp

        rs = dtc.convert(ts.to_pydatetime(), None, None)
        assert rs == xp

        rs = dtc.convert(Index([ts - Day(1), ts]), None, None)
        assert rs[1] == xp

        rs = dtc.convert(Index([ts - Day(1), ts]).to_pydatetime(), None, None)
        assert rs[1] == xp

    # 测试带有浮点数的日期转换是否正确
    def test_conversion_float(self, dtc):
        rtol = 0.5 * 10**-9

        rs = dtc.convert(Timestamp("2012-1-1 01:02:03", tz="UTC"), None, None)
        xp = dates.date2num(Timestamp("2012-1-1 01:02:03", tz="UTC"))
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(
            Timestamp("2012-1-1 09:02:03", tz="Asia/Hong_Kong"), None, None
        )
        tm.assert_almost_equal(rs, xp, rtol=rtol)

        rs = dtc.convert(datetime(2012, 1, 1, 1, 2, 3), None, None)
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    # 参数化测试超出边界日期时间的转换是否正确
    @pytest.mark.parametrize(
        "values",
        [
            [date(1677, 1, 1), date(1677, 1, 2)],
            [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)],
        ],
    )
    def test_conversion_outofbounds_datetime(self, dtc, values):
        # 用 DatetimeConverter 转换日期值并验证结果是否与期望一致
        rs = dtc.convert(values, None, None)
        xp = dates.date2num(values)
        tm.assert_numpy_array_equal(rs, xp)
        rs = dtc.convert(values[0], None, None)
        xp = dates.date2num(values[0])
        assert rs == xp
    # 使用 pytest 的 @parametrize 装饰器，为 test_time_formatter 方法提供多组参数化测试数据
    @pytest.mark.parametrize(
        "time,format_expected",
        [
            (0, "00:00"),  # 将时间0转换为格式"00:00"
            (86399.999999, "23:59:59.999999"),  # 将最大时间转换为格式"23:59:59.999999"
            (90000, "01:00"),  # 将时间90000转换为格式"01:00"
            (3723, "01:02:03"),  # 将时间3723转换为格式"01:02:03"
            (39723.2, "11:02:03.200"),  # 将时间39723.2转换为格式"11:02:03.200"
        ],
    )
    def test_time_formatter(self, time, format_expected):
        # 调用 TimeFormatter 类进行时间格式化，存储结果
        result = converter.TimeFormatter(None)(time)
        # 断言格式化后的结果与预期格式相符
        assert result == format_expected

    # 使用 pytest 的 @parametrize 装饰器，为 test_dateindex_conversion 方法提供多组参数化测试数据
    @pytest.mark.parametrize("freq", ("B", "ms", "s"))
    def test_dateindex_conversion(self, freq, dtc):
        rtol = 10**-9  # 设置相对误差
        # 创建日期范围对象，从"2020-01-01"开始，包含10个周期，指定频率为参数 freq
        dateindex = date_range("2020-01-01", periods=10, freq=freq)
        # 使用 dtc.convert 方法将日期范围转换为数值表示，存储结果
        rs = dtc.convert(dateindex, None, None)
        # 将日期范围对象的 matplotlib 表示转换为数值，存储期望结果
        xp = dates.date2num(dateindex._mpl_repr())
        # 断言转换结果与期望结果在指定相对误差下几乎相等
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    # 使用 pytest 的 @parametrize 装饰器，为 test_resolution 方法提供多组参数化测试数据
    @pytest.mark.parametrize("offset", [Second(), Milli(), Micro(50)])
    def test_resolution(self, offset, dtc):
        # Matplotlib 的浮点数时间表示在常见的年份范围内无法区分小于约10微秒的间隔。
        # 创建初始时间戳 ts1
        ts1 = Timestamp("2012-1-1")
        # 创建偏移后的时间戳 ts2
        ts2 = ts1 + offset
        # 将 ts1 和 ts2 转换为数值表示，分别存储结果
        val1 = dtc.convert(ts1, None, None)
        val2 = dtc.convert(ts2, None, None)
        # 如果 val1 不小于 val2，则抛出断言错误
        if not val1 < val2:
            raise AssertionError(f"{val1} is not less than {val2}.")

    # 测试嵌套数据转换功能
    def test_convert_nested(self, dtc):
        # 创建内部时间戳列表
        inner = [Timestamp("2017-01-01"), Timestamp("2017-01-02")]
        # 创建包含内部列表的数据列表
        data = [inner, inner]
        # 使用 dtc.convert 方法将数据列表转换为数值表示，存储结果
        result = dtc.convert(data, None, None)
        # 预期结果为将每个内部列表分别转换为数值表示
        expected = [dtc.convert(x, None, None) for x in data]
        # 断言结果数组中的所有元素与期望结果数组相等
        assert (np.array(result) == expected).all()
class TestPeriodConverter:
    # 定义 pytest fixture，返回 PeriodConverter 实例
    @pytest.fixture
    def pc(self):
        return converter.PeriodConverter()

    # 定义 pytest fixture，返回包含 freq 属性的 Axis 类实例
    @pytest.fixture
    def axis(self):
        # 定义 Axis 类
        class Axis:
            pass

        axis = Axis()
        axis.freq = "D"  # 设置频率属性为 "D"
        return axis

    # 测试函数：测试 convert 方法是否接受 Unicode 字符串
    def test_convert_accepts_unicode(self, pc, axis):
        r1 = pc.convert("2012-1-1", None, axis)
        r2 = pc.convert("2012-1-1", None, axis)
        assert r1 == r2

    # 测试函数：测试 convert 方法的日期转换功能
    def test_conversion(self, pc, axis):
        xp = Period("2012-1-1").ordinal  # 获取日期 "2012-1-1" 的序数值
        # 测试 convert 方法对不同类型输入的处理
        rs = pc.convert(["2012-1-1"], None, axis)[0]
        assert rs == xp

        rs = pc.convert("2012-1-1", None, axis)
        assert rs == xp

        rs = pc.convert([date(2012, 1, 1)], None, axis)[0]
        assert rs == xp

        rs = pc.convert(date(2012, 1, 1), None, axis)
        assert rs == xp

        rs = pc.convert([Timestamp("2012-1-1")], None, axis)[0]
        assert rs == xp

        rs = pc.convert(Timestamp("2012-1-1"), None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01", None, axis)
        assert rs == xp

        rs = pc.convert("2012-01-01 00:00:00+0000", None, axis)
        assert rs == xp

        rs = pc.convert(
            np.array(
                ["2012-01-01 00:00:00", "2012-01-02 00:00:00"],
                dtype="datetime64[ns]",
            ),
            None,
            axis,
        )
        assert rs[0] == xp

    # 测试函数：测试 convert 方法对整数列表的传递是否正常
    def test_integer_passthrough(self, pc, axis):
        # GH9012
        rs = pc.convert([0, 1], None, axis)
        xp = [0, 1]
        assert rs == xp

    # 测试函数：测试 convert 方法对嵌套列表的处理是否正常
    def test_convert_nested(self, pc, axis):
        data = ["2012-1-1", "2012-1-2"]
        r1 = pc.convert([data, data], None, axis)
        r2 = [pc.convert(data, None, axis) for _ in range(2)]
        assert r1 == r2


class TestTimeDeltaConverter:
    """Test timedelta converter"""

    # 参数化测试：测试 format_timedelta_ticks 方法的时间间隔格式化功能
    @pytest.mark.parametrize(
        "x, decimal, format_expected",
        [
            (0.0, 0, "00:00:00"),
            (3972320000000, 1, "01:06:12.3"),
            (713233432000000, 2, "8 days 06:07:13.43"),
            (32423432000000, 4, "09:00:23.4320"),
        ],
    )
    def test_format_timedelta_ticks(self, x, decimal, format_expected):
        tdc = converter.TimeSeries_TimedeltaFormatter
        result = tdc.format_timedelta_ticks(x, pos=None, n_decimals=decimal)
        assert result == format_expected

    # 参数化测试：测试 call 方法在不同视图间隔下的调用是否正常
    @pytest.mark.parametrize("view_interval", [(1, 2), (2, 1)])
    def test_call_w_different_view_intervals(self, view_interval, monkeypatch):
        # 为了修复先前在反向限制时出现的问题；见 GH37454
        class mock_axis:
            def get_view_interval(self):
                return view_interval

        tdc = converter.TimeSeries_TimedeltaFormatter()
        monkeypatch.setattr(tdc, "axis", mock_axis())
        tdc(0.0, 0)


# 参数化测试：测试年份跨度范围限制
@pytest.mark.parametrize("year_span", [11.25, 30, 80, 150, 400, 800, 1500, 2500, 3500])
# 定义测试函数 test_quarterly_finder，用于测试 _quarterly_finder() 函数
def test_quarterly_finder(year_span):
    # 设置最小值 vmin 为 -1000
    vmin = -1000
    # 根据年度跨度计算最大值 vmax
    vmax = vmin + year_span * 4
    # 计算数值范围 span
    span = vmax - vmin + 1
    # 如果 span 小于 45，则跳过测试，因为 quarterly finder 只在 span >= 45 时调用
    if span < 45:
        pytest.skip("the quarterly finder is only invoked if the span is >= 45")
    # 计算年数 nyears，即 span 除以 4
    nyears = span / 4
    # 获取默认的年度间隔（最小和最大间隔）
    (min_anndef, maj_anndef) = converter._get_default_annual_spacing(nyears)
    # 调用 _quarterly_finder() 函数，返回结果
    result = converter._quarterly_finder(vmin, vmax, to_offset("QE"))
    # 根据结果创建 PeriodIndex 对象 quarters，表示季度索引
    quarters = PeriodIndex(
        arrays.PeriodArray(np.array([x[0] for x in result]), dtype="period[Q]")
    )
    # 从结果中提取主要和次要索引
    majors = np.array([x[1] for x in result])
    minors = np.array([x[2] for x in result])
    # 根据主要索引获取主要季度
    major_quarters = quarters[majors]
    # 根据次要索引获取次要季度
    minor_quarters = quarters[minors]
    # 检查主要季度的年份是否符合主要年度间隔要求
    check_major_years = major_quarters.year % maj_anndef == 0
    # 检查次要季度的年份是否符合次要年度间隔要求
    check_minor_years = minor_quarters.year % min_anndef == 0
    # 检查主要季度是否为第一季度
    check_major_quarters = major_quarters.quarter == 1
    # 检查次要季度是否为第一季度
    check_minor_quarters = minor_quarters.quarter == 1
    # 断言所有主要年份符合要求
    assert np.all(check_major_years)
    # 断言所有次要年份符合要求
    assert np.all(check_minor_years)
    # 断言所有主要季度为第一季度
    assert np.all(check_major_quarters)
    # 断言所有次要季度为第一季度
    assert np.all(check_minor_quarters)
```