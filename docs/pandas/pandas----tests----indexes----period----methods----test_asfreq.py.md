# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_asfreq.py`

```
# 导入 re 模块，用于正则表达式操作
import re

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 pandas 模块导入 PeriodIndex、Series 和 period_range 函数
from pandas import (
    PeriodIndex,
    Series,
    period_range,
)

# 导入 pandas 内部测试工具，简写为 tm
import pandas._testing as tm

# 从 pandas.tseries 模块导入 offsets，用于时间序列的偏移量操作
from pandas.tseries import offsets


class TestPeriodIndex:
    # 定义 TestPeriodIndex 类，用于测试 PeriodIndex 对象的功能
    # 测试不同频率的时间段创建，并进行频率转换的功能
    def test_asfreq(self):
        # 创建年度频率的时间段对象 pi1，包含2001年整年
        pi1 = period_range(freq="Y", start="1/1/2001", end="1/1/2001")
        # 创建季度频率的时间段对象 pi2，包含2001年第一季度
        pi2 = period_range(freq="Q", start="1/1/2001", end="1/1/2001")
        # 创建月度频率的时间段对象 pi3，包含2001年1月
        pi3 = period_range(freq="M", start="1/1/2001", end="1/1/2001")
        # 创建每日频率的时间段对象 pi4，包含2001年1月1日
        pi4 = period_range(freq="D", start="1/1/2001", end="1/1/2001")
        # 创建每小时频率的时间段对象 pi5，包含2001年1月1日 00:00
        pi5 = period_range(freq="h", start="1/1/2001", end="1/1/2001 00:00")
        # 创建每分钟频率的时间段对象 pi6，包含2001年1月1日 00:00
        pi6 = period_range(freq="Min", start="1/1/2001", end="1/1/2001 00:00")
        # 创建每秒钟频率的时间段对象 pi7，包含2001年1月1日 00:00:00
        pi7 = period_range(freq="s", start="1/1/2001", end="1/1/2001 00:00:00")

        # 检查年度时间段 pi1 转换为季度频率的结果等于 pi2
        assert pi1.asfreq("Q", "s") == pi2
        # 再次检查年度时间段 pi1 转换为季度频率的结果等于 pi2
        assert pi1.asfreq("Q", "s") == pi2
        # 检查年度时间段 pi1 转换为月度频率的结果等于 pi3
        assert pi1.asfreq("M", "start") == pi3
        # 检查年度时间段 pi1 转换为每日频率的结果等于 pi4
        assert pi1.asfreq("D", "StarT") == pi4
        # 检查年度时间段 pi1 转换为每小时频率的结果等于 pi5
        assert pi1.asfreq("h", "beGIN") == pi5
        # 检查年度时间段 pi1 转换为每分钟频率的结果等于 pi6
        assert pi1.asfreq("Min", "s") == pi6
        # 检查年度时间段 pi1 转换为每秒钟频率的结果等于 pi7
        assert pi1.asfreq("s", "s") == pi7

        # 检查季度时间段 pi2 转换为年度频率的结果等于 pi1
        assert pi2.asfreq("Y", "s") == pi1
        # 检查季度时间段 pi2 转换为月度频率的结果等于 pi3
        assert pi2.asfreq("M", "s") == pi3
        # 检查季度时间段 pi2 转换为每日频率的结果等于 pi4
        assert pi2.asfreq("D", "s") == pi4
        # 检查季度时间段 pi2 转换为每小时频率的结果等于 pi5
        assert pi2.asfreq("h", "s") == pi5
        # 检查季度时间段 pi2 转换为每分钟频率的结果等于 pi6
        assert pi2.asfreq("Min", "s") == pi6
        # 检查季度时间段 pi2 转换为每秒钟频率的结果等于 pi7
        assert pi2.asfreq("s", "s") == pi7

        # 检查月度时间段 pi3 转换为年度频率的结果等于 pi1
        assert pi3.asfreq("Y", "s") == pi1
        # 检查月度时间段 pi3 转换为季度频率的结果等于 pi2
        assert pi3.asfreq("Q", "s") == pi2
        # 检查月度时间段 pi3 转换为每日频率的结果等于 pi4
        assert pi3.asfreq("D", "s") == pi4
        # 检查月度时间段 pi3 转换为每小时频率的结果等于 pi5
        assert pi3.asfreq("h", "s") == pi5
        # 检查月度时间段 pi3 转换为每分钟频率的结果等于 pi6
        assert pi3.asfreq("Min", "s") == pi6
        # 检查月度时间段 pi3 转换为每秒钟频率的结果等于 pi7
        assert pi3.asfreq("s", "s") == pi7

        # 检查每日时间段 pi4 转换为年度频率的结果等于 pi1
        assert pi4.asfreq("Y", "s") == pi1
        # 检查每日时间段 pi4 转换为季度频率的结果等于 pi2
        assert pi4.asfreq("Q", "s") == pi2
        # 检查每日时间段 pi4 转换为月度频率的结果等于 pi3
        assert pi4.asfreq("M", "s") == pi3
        # 检查每日时间段 pi4 转换为每小时频率的结果等于 pi5
        assert pi4.asfreq("h", "s") == pi5
        # 检查每日时间段 pi4 转换为每分钟频率的结果等于 pi6
        assert pi4.asfreq("Min", "s") == pi6
        # 检查每日时间段 pi4 转换为每秒钟频率的结果等于 pi7
        assert pi4.asfreq("s", "s") == pi7

        # 检查每小时时间段 pi5 转换为年度频率的结果等于 pi1
        assert pi5.asfreq("Y", "s") == pi1
        # 检查每小时时间段 pi5 转换为季度频率的结果等于 pi2
        assert pi5.asfreq("Q", "s") == pi2
        # 检查每小时时间段 pi5 转换为月度频率的结果等于 pi3
        assert pi5.asfreq("M", "s") == pi3
        # 检查每小时时间段 pi5 转换为每日频率的结果等于 pi4
        assert pi5.asfreq("D", "s") == pi4
        # 检查每小时时间段 pi5 转换为每分钟频率的结果等于 pi6
        assert pi5.asfreq("Min", "s") == pi6
        # 检查每小时时间段 pi5 转换为每秒钟频率的结果等于 pi7
        assert pi5.asfreq("s", "s") == pi7

        # 检查每分钟时间段 pi6 转换为年度频率的结果等于 pi1
        assert pi6.asfreq("Y", "s") == pi1
        # 检查每分钟时间段 pi6 转换为季度频率的结果等于 pi2
        assert pi6.asfreq("Q", "s") == pi2
        # 检查每分钟时间段 pi6 转换为月度频率的结果等于 pi3
        assert pi6.asfreq("M", "s") == pi3
        # 检查每分钟时间段 pi6 转换为每日频率的结果等于 pi4
        assert pi6.asfreq("D", "s") == pi4
        # 检查每分钟时间段 pi6 转换为每小时频率的结果等于 pi5
        assert pi6.asfreq("h", "s") == pi5
        # 检查每分钟时间段 pi6 转换为每秒钟频
    # 测试 PeriodIndex 的 asfreq 方法，使用频率 "M"（月度）
    def test_asfreq_nat(self):
        # 创建一个 PeriodIndex 对象，包含日期 "2011-01", "2011-02", "NaT"（不可用时间），"2011-04"，频率为 "M"（每月）
        idx = PeriodIndex(["2011-01", "2011-02", "NaT", "2011-04"], freq="M")
        # 调用 asfreq 方法，将频率转换为 "Q"（每季度）
        result = idx.asfreq(freq="Q")
        # 期望的结果 PeriodIndex，包含日期 "2011Q1", "2011Q1", "NaT", "2011Q2"，频率为 "Q"（每季度）
        expected = PeriodIndex(["2011Q1", "2011Q1", "NaT", "2011Q2"], freq="Q")
        # 断言结果与期望一致
        tm.assert_index_equal(result, expected)

    # 使用 pytest 参数化装饰器进行多个频率的测试
    @pytest.mark.parametrize("freq", ["D", "3D"])
    def test_asfreq_mult_pi(self, freq):
        # 创建一个 PeriodIndex 对象，包含日期 "2001-01", "2001-02", "NaT"（不可用时间），"2001-03"，频率为 "2M"（两个月）
        pi = PeriodIndex(["2001-01", "2001-02", "NaT", "2001-03"], freq="2M")

        # 调用 asfreq 方法，将频率转换为参数化的 freq
        result = pi.asfreq(freq)
        # 期望的结果 PeriodIndex，包含日期 "2001-02-28", "2001-03-31", "NaT", "2001-04-30"，频率为参数化的 freq
        exp = PeriodIndex(["2001-02-28", "2001-03-31", "NaT", "2001-04-30"], freq=freq)
        # 断言结果与期望一致
        tm.assert_index_equal(result, exp)
        # 断言结果的频率与期望的频率一致
        assert result.freq == exp.freq

        # 调用 asfreq 方法，将频率转换为参数化的 freq，并指定转换方式为 "S"（起始）
        result = pi.asfreq(freq, how="S")
        # 期望的结果 PeriodIndex，包含日期 "2001-01-01", "2001-02-01", "NaT", "2001-03-01"，频率为参数化的 freq
        exp = PeriodIndex(["2001-01-01", "2001-02-01", "NaT", "2001-03-01"], freq=freq)
        # 断言结果与期望一致
        tm.assert_index_equal(result, exp)
        # 断言结果的频率与期望的频率一致
        assert result.freq == exp.freq

    # 测试组合频率的 asfreq 方法
    def test_asfreq_combined_pi(self):
        # 创建一个 PeriodIndex 对象，包含日期 "2001-01-01 00:00", "2001-01-02 02:00", "NaT"（不可用时间），频率为 "h"（小时）
        pi = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="h")
        # 期望的结果 PeriodIndex，包含日期 "2001-01-01 00:00", "2001-01-02 02:00", "NaT"，频率为 "25h"（25小时）
        exp = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="25h")

        # 遍历多个频率和转换方式的组合进行测试
        for freq, how in zip(["1D1h", "1h1D"], ["S", "E"]):
            # 调用 asfreq 方法，将频率转换为 freq，并指定转换方式为 how
            result = pi.asfreq(freq, how=how)
            # 断言结果与期望一致
            tm.assert_index_equal(result, exp)
            # 断言结果的频率与期望的频率一致
            assert result.freq == exp.freq

        # 遍历多个频率进行测试
        for freq in ["1D1h", "1h1D"]:
            # 创建一个 PeriodIndex 对象，包含日期 "2001-01-01 00:00", "2001-01-02 02:00", "NaT"，频率为 freq
            pi = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq=freq)
            # 调用 asfreq 方法，将频率转换为 "h"（小时）
            result = pi.asfreq("h")
            # 期望的结果 PeriodIndex，包含日期 "2001-01-02 00:00", "2001-01-03 02:00", "NaT"，频率为 "h"（小时）
            exp = PeriodIndex(["2001-01-02 00:00", "2001-01-03 02:00", "NaT"], freq="h")
            # 断言结果与期望一致
            tm.assert_index_equal(result, exp)
            # 断言结果的频率与期望的频率一致
            assert result.freq == exp.freq

            # 调用 asfreq 方法，将频率转换为 "h"（小时），并指定转换方式为 "S"（起始）
            result = pi.asfreq("h", how="S")
            # 结果应与原始 PeriodIndex 对象一致，因为转换方式为 "S"（起始）
            exp = PeriodIndex(["2001-01-01 00:00", "2001-01-02 02:00", "NaT"], freq="h")
            # 断言结果与期望一致
            tm.assert_index_equal(result, exp)
            # 断言结果的频率与期望的频率一致
            assert result.freq == exp.freq

    # 测试 PeriodIndex 的 astype 和 asfreq 方法
    def test_astype_asfreq(self):
        # 创建一个 PeriodIndex 对象，包含日期 "2011-01-01", "2011-02-01", "2011-03-01"，频率为 "D"（每天）
        pi1 = PeriodIndex(["2011-01-01", "2011-02-01", "2011-03-01"], freq="D")
        # 期望的结果 PeriodIndex，包含日期 "2011-01", "2011-02", "2011-03"，频率为 "M"（每月）
        exp = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="M")
        # 断言 astype 方法的结果与期望一致
        tm.assert_index_equal(pi1.astype("period[M]"), exp)
        # 断言 asfreq 方法将频率转换为 "M" 的结果与期望一致
        tm.assert_index_equal(pi1.asfreq("M"), exp)

        # 期望的结果 PeriodIndex，包含日期 "2011-01", "2011-02", "2011-03"，频率为 "3M"（每三个月）
        exp = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="3M")
        # 断言 astype 方法的结果与期望一致
        tm.assert_index_equal(pi1.astype("period[3M]"), exp)
        # 断言 asfreq 方法将频率转换为 "3M" 的结果与期望一致
        tm.assert_index_equal(pi1.asfreq("3M"), exp)

    # 测试 Series 对象的 asfreq 方法
    def test_asfreq_with_different_n(self):
        # 创建一个 Series 对象，包含索引为 PeriodIndex 对象，日期为 "2020-01", "2020-03"，频率为 "2M"（两个月）
        ser = Series([1, 2], index=PeriodIndex(["2020-01", "2020-03"], freq="2M"))
        # 调用 asfreq 方法，将索引的频率转换为 "M"（每月
    # 测试函数，用于验证当频率不支持时抛出异常
    def test_pi_asfreq_not_supported_frequency(self, freq):
        # GH#55785, GH#56945
        # 构造错误信息，包括频率无效和不支持的提示信息
        msg = "|".join(
            [
                f"Invalid frequency: {freq}",
                re.escape(f"{freq} is not supported as period frequency"),
                "bh is not supported as period frequency",
            ]
        )

        # 创建一个以月为频率的时间段索引对象
        pi = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        # 使用 pytest 来验证在设定的频率下调用 asfreq 方法是否会抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=msg):
            pi.asfreq(freq=freq)

    # 参数化测试函数，用于验证各种无效频率是否能触发异常
    @pytest.mark.parametrize(
        "freq",
        [
            "2BME",
            "2YE-MAR",
            "2QE",
        ],
    )
    def test_pi_asfreq_invalid_frequency(self, freq):
        # GH#55785
        # 构造错误信息，指明频率无效
        msg = f"Invalid frequency: {freq}"

        # 创建一个以月为频率的时间段索引对象
        pi = PeriodIndex(["2020-01-01", "2021-01-01"], freq="M")
        # 使用 pytest 来验证在设定的频率下调用 asfreq 方法是否会抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=msg):
            pi.asfreq(freq=freq)
```