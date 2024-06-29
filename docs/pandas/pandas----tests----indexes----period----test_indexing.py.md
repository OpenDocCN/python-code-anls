# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_indexing.py`

```
# 导入必要的模块和类
from datetime import datetime  # 导入 datetime 类用于日期操作
import re  # 导入 re 模块用于正则表达式操作

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import period as libperiod  # 导入 pandas 内部的 period 模块
from pandas.errors import InvalidIndexError  # 导入 pandas 的 InvalidIndexError 异常类

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 库中导入多个类和函数
    DatetimeIndex,  # 日期时间索引类
    NaT,  # 表示不可用日期时间的特殊值
    Period,  # 时期类
    PeriodIndex,  # 时期索引类
    Series,  # 系列类
    Timedelta,  # 时间增量类
    date_range,  # 创建日期范围的函数
    notna,  # 检查对象中非缺失值的函数
    period_range,  # 创建时期范围的函数
)
import pandas._testing as tm  # 导入 pandas 测试模块，并使用 tm 别名

# 创建一个日期范围对象 dti4，从 "2016-01-01" 开始，持续 4 个周期
dti4 = date_range("2016-01-01", periods=4)
# 使用 dti4 切片创建一个新的日期时间索引对象 dti，去除最后一个周期
dti = dti4[:-1]
# 创建一个包含整数范围的 pandas 索引对象 rng，从 0 到 2
rng = pd.Index(range(3))

# 定义一个 pytest 的 fixture 函数 non_comparable_idx，参数包括多种类型的索引对象
@pytest.fixture(
    params=[
        dti,  # 日期时间索引对象 dti
        dti.tz_localize("UTC"),  # 将 dti 转换为 UTC 时区的日期时间索引对象
        dti.to_period("W"),  # 将 dti 转换为周期为一周的时期索引对象
        dti - dti[0],  # dti 减去其第一个元素后的结果
        rng,  # 整数范围索引对象 rng
        pd.Index([1, 2, 3]),  # 包含整数的新索引对象
        pd.Index([2.0, 3.0, 4.0]),  # 包含浮点数的新索引对象
        pd.Index([4, 5, 6], dtype="u8"),  # 使用无符号 8 位整数类型的新索引对象
        pd.IntervalIndex.from_breaks(dti4),  # 使用 dti4 的断点创建区间索引对象
    ]
)
def non_comparable_idx(request):
    # 所有索引对象长度均为 3
    return request.param


class TestGetItem:
    def test_getitem_slice_keeps_name(self):
        # 创建一个名称为 'bob' 的每日频率时期范围索引对象 idx
        idx = period_range("20010101", periods=10, freq="D", name="bob")
        # 断言切片后的索引对象名称仍为原索引对象的名称
        assert idx.name == idx[1:].name

    def test_getitem(self):
        # 创建一个从 "2011-01-01" 到 "2011-01-31" 的每日频率时期范围索引对象 idx1，名称为 'idx'
        idx1 = period_range("2011-01-01", "2011-01-31", freq="D", name="idx")

        for idx in [idx1]:
            # 获取索引对象的第一个元素并断言其与预期的日期时期相同
            result = idx[0]
            assert result == Period("2011-01-01", freq="D")

            # 获取索引对象的最后一个元素并断言其与预期的日期时期相同
            result = idx[-1]
            assert result == Period("2011-01-31", freq="D")

            # 获取索引对象的前五个元素并断言与预期的时期范围索引对象相同
            result = idx[0:5]
            expected = period_range("2011-01-01", "2011-01-05", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 获取索引对象的每隔两个元素并断言与预期的时期索引对象相同
            result = idx[0:10:2]
            expected = PeriodIndex(
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-07", "2011-01-09"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 获取索引对象的倒数 20 到倒数 5 的元素并断言与预期的时期索引对象相同
            result = idx[-20:-5:3]
            expected = PeriodIndex(
                ["2011-01-12", "2011-01-15", "2011-01-18", "2011-01-21", "2011-01-24"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 获取索引对象的第五个元素及其之前的所有元素并断言与预期的时期索引对象相同
            result = idx[4::-1]
            expected = PeriodIndex(
                ["2011-01-05", "2011-01-04", "2011-01-03", "2011-01-02", "2011-01-01"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"
    # 定义测试方法，用于测试根据索引获取数据的功能
    def test_getitem_index(self):
        # 创建一个时间段索引对象，从"2007-01"开始，共10个月，频率为每月("M")，命名为"x"
        idx = period_range("2007-01", periods=10, freq="M", name="x")

        # 获取索引为[1, 3, 5]的数据
        result = idx[[1, 3, 5]]
        # 期望结果是一个时间段索引对象，包含["2007-02", "2007-04", "2007-06"]，频率为每月("M")，命名为"x"
        exp = PeriodIndex(["2007-02", "2007-04", "2007-06"], freq="M", name="x")
        # 断言结果与期望值相等
        tm.assert_index_equal(result, exp)

        # 获取索引为[True, True, False, False, False, True, True, False, False, False]的数据
        result = idx[[True, True, False, False, False, True, True, False, False, False]]
        # 期望结果是一个时间段索引对象，包含["2007-01", "2007-02", "2007-06", "2007-07"]，频率为每月("M")，命名为"x"
        exp = PeriodIndex(
            ["2007-01", "2007-02", "2007-06", "2007-07"], freq="M", name="x"
        )
        # 断言结果与期望值相等
        tm.assert_index_equal(result, exp)

    # 定义测试方法，用于测试根据部分索引获取数据的功能
    def test_getitem_partial(self):
        # 创建一个时间段索引对象，从"2007-01"开始，共50个月，频率为每月("M")
        rng = period_range("2007-01", periods=50, freq="M")
        # 创建一个时间序列，索引为rng，值为50个标准正态分布的随机数
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

        # 使用 pytest 断言，检查是否会引发 KeyError，并匹配错误消息 "'2006'"
        with pytest.raises(KeyError, match=r"^'2006'$"):
            ts["2006"]

        # 获取索引为"2008"的数据
        result = ts["2008"]
        # 断言结果的索引年份全部为2008
        assert (result.index.year == 2008).all()

        # 获取索引为"2008"到"2009"之间的数据
        result = ts["2008":"2009"]
        # 断言结果的长度为24
        assert len(result) == 24

        # 获取索引为"2008-1"到"2009-12"之间的数据
        result = ts["2008-1":"2009-12"]
        # 断言结果的长度为24
        assert len(result) == 24

        # 获取索引为"2008Q1"到"2009Q4"之间的数据
        result = ts["2008Q1":"2009Q4"]
        # 断言结果的长度为24
        assert len(result) == 24

        # 获取索引为起始到"2009"之间的数据
        result = ts[:"2009"]
        # 断言结果的长度为36
        assert len(result) == 36

        # 获取索引为"2009"到结束之间的数据
        result = ts["2009":]
        # 断言结果的长度为50 - 24 = 26
        assert len(result) == 50 - 24

        # 将结果赋给exp
        exp = result
        # 获取从索引为24开始的数据
        result = ts[24:]
        # 使用 pandas 测试工具断言 exp 与 result 相等
        tm.assert_series_equal(exp, result)

        # 对时间序列进行切片操作，再次连接切片的结果
        ts = pd.concat([ts[10:], ts[10:]])
        # 定义错误消息
        msg = "left slice bound for non-unique label: '2008'"
        # 使用 pytest 断言，检查是否会引发 KeyError，并匹配错误消息
        with pytest.raises(KeyError, match=msg):
            ts[slice("2008", "2009")]

    # 定义测试方法，用于测试根据日期时间获取数据的功能
    def test_getitem_datetime(self):
        # 创建一个周期范围，从"2012-01-01"开始，共10周，频率为每周的周一("W-MON")
        rng = period_range(start="2012-01-01", periods=10, freq="W-MON")
        # 创建一个时间序列，索引为rng，值为索引的长度
        ts = Series(range(len(rng)), index=rng)

        # 创建两个日期时间对象
        dt1 = datetime(2011, 10, 2)
        dt4 = datetime(2012, 4, 20)

        # 使用日期时间对象dt1和dt4获取时间序列数据
        rs = ts[dt1:dt4]
        # 使用 pandas 测试工具断言 rs 与 ts 相等
        tm.assert_series_equal(rs, ts)

    # 定义测试方法，用于测试根据NaT获取数据的功能
    def test_getitem_nat(self):
        # 创建一个时间段索引对象，包含三个元素："2011-01"、NaT、"2011-02"，频率为每月("M")
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="M")
        # 断言第一个元素等于 Period("2011-01", freq="M")
        assert idx[0] == Period("2011-01", freq="M")
        # 断言第二个元素是 NaT
        assert idx[1] is NaT

        # 创建一个时间序列，索引为[0, 1, 2]，值为[0, 1, 2]，索引为idx
        s = Series([0, 1, 2], index=idx)
        # 断言使用 NaT 作为索引得到的值为1
        assert s[NaT] == 1

        # 创建一个时间序列，索引为idx，值为idx
        s = Series(idx, index=idx)
        # 断言使用 Period("2011-01", freq="M") 作为索引得到的值等于 Period("2011-01", freq="M")
        assert s[Period("2011-01", freq="M")] == Period("2011-01", freq="M")
        # 断言使用 NaT 作为索引得到的值是 NaT
        assert s[NaT] is NaT

    # 定义测试方法，用于测试根据时间段列表获取数据的功能
    @pytest.mark.arm_slow
    def test_getitem_list_periods(self):
        # 创建一个周期范围，从"2012-01-01"开始，共10天，频率为每天("D")
        rng = period_range(start="2012-01-01", periods=10, freq="D")
        # 创建一个时间序列，索引为rng，值为索引的长度
        ts = Series(range(len(rng)), index=rng)
        # 期望结果是一个时间序列，索引为[Period("2012-01-02", freq="D")]，值为索引为1的数据
        exp = ts.iloc[[1]]
        # 使用 pandas 测试工具断言 ts 中索引为Period("2012-01-02", freq="D")的数据等于exp
        tm.assert_series_equal(ts[[Period("2012-01-02", freq="D")]], exp)
    def test_getitem_seconds(self):
        # GH#6716
        # 创建包含秒级频率的日期时间索引
        didx = date_range(start="2013/01/01 09:00:00", freq="s", periods=4000)
        # 创建包含秒级频率的周期索引
        pidx = period_range(start="2013/01/01 09:00:00", freq="s", periods=4000)

        for idx in [didx, pidx]:
            # 对非整数索引进行切片操作应该引发 ValueError 异常
            values = [
                "2014",
                "2013/02",
                "2013/01/02",
                "2013/02/01 9h",
                "2013/02/01 09:00",
            ]
            for val in values:
                # GH7116
                # 这些展示了由于尝试使用非整数索引器而引起的废弃警告
                with pytest.raises(IndexError, match="only integers, slices"):
                    idx[val]

            # 创建随机数据序列，使用日期时间索引或周期索引作为索引
            ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
            # 断言对于指定日期时间索引，序列应与给定范围的子序列相等
            tm.assert_series_equal(ser["2013/01/01 10:00"], ser[3600:3660])
            # 断言对于指定日期时间索引，序列应与从开头到指定范围的子序列相等
            tm.assert_series_equal(ser["2013/01/01 9h"], ser[:3600])
            for d in ["2013/01/01", "2013/01", "2013"]:
                # 断言对于指定日期或日期时间，序列应与自身相等
                tm.assert_series_equal(ser[d], ser)

    @pytest.mark.parametrize(
        "idx_range",
        [
            date_range,
            period_range,
        ],
    )
    def test_getitem_day(self, idx_range):
        # GH#6716
        # 确认日期时间索引和周期索引的工作方式相同
        # 对非整数索引进行切片操作应该引发 ValueError 异常
        idx = idx_range(start="2013/01/01", freq="D", periods=400)
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9h",
            "2013/02/01 09:00",
        ]
        for val in values:
            # GH7116
            # 这些展示了由于尝试使用非整数索引器而引起的废弃警告
            with pytest.raises(IndexError, match="only integers, slices"):
                idx[val]

        # 创建随机数据序列，使用日期时间索引或周期索引作为索引
        ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
        # 断言对于指定年份的序列，序列应与指定范围的子序列相等
        tm.assert_series_equal(ser["2013/01"], ser[0:31])
        # 断言对于指定月份的序列，序列应与指定范围的子序列相等
        tm.assert_series_equal(ser["2013/02"], ser[31:59])
        # 断言对于指定年份的序列，序列应与指定范围的子序列相等
        tm.assert_series_equal(ser["2014"], ser[365:])

        invalid = ["2013/02/01 9h", "2013/02/01 09:00"]
        for val in invalid:
            # 对于无效键的访问应引发 KeyError 异常，匹配特定值
            with pytest.raises(KeyError, match=val):
                ser[val]
class TestGetLoc:
    # 定义测试方法：测试获取位置信息时是否能正确捕获 KeyError 异常
    def test_get_loc_msg(self):
        # 创建一个时间区间索引对象，从 "2000-1-1" 开始，每年频率，共 10 个周期
        idx = period_range("2000-1-1", freq="Y", periods=10)
        # 创建一个错误的时间区间对象，年份为 "2012"，频率为 "Y"
        bad_period = Period("2012", "Y")
        # 使用 pytest 断言捕获期望的 KeyError 异常，其消息匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^Period\('2012', 'Y-DEC'\)$"):
            idx.get_loc(bad_period)

        # 使用 try-except 结构捕获 KeyError 异常，并断言异常对象的第一个参数与 bad_period 相等
        try:
            idx.get_loc(bad_period)
        except KeyError as inst:
            assert inst.args[0] == bad_period

    # 定义测试方法：测试获取位置信息时对 NaT 和 NaN 的处理
    def test_get_loc_nat(self):
        # 创建一个日期时间索引对象，包含日期 "2011-01-01", "NaT", "2011-01-03"
        didx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-03"])
        # 创建一个周期索引对象，包含周期 "2011-01-01", "NaT", "2011-01-03"，频率为 "M"
        pidx = PeriodIndex(["2011-01-01", "NaT", "2011-01-03"], freq="M")

        # 遍历不同类型的索引对象
        for idx in [didx, pidx]:
            # 断言获取 NaT 的位置为 1
            assert idx.get_loc(NaT) == 1
            # 断言获取 None 的位置为 1
            assert idx.get_loc(None) == 1
            # 断言获取 float("nan") 的位置为 1
            assert idx.get_loc(float("nan")) == 1
            # 断言获取 np.nan 的位置为 1
            assert idx.get_loc(np.nan) == 1

    # 定义测试方法：测试获取位置信息的其他情况
    def test_get_loc(self):
        # GH 17717
        # 创建周期对象 p0, p1, p2
        p0 = Period("2017-09-01")
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")

        # 创建一个单调递增的周期索引对象 idx0，包含周期 p0, p1, p2
        idx0 = PeriodIndex([p0, p1, p2])
        expected_idx1_p1 = 1
        expected_idx1_p2 = 2

        # 断言获取 p1 的位置为 expected_idx1_p1
        assert idx0.get_loc(p1) == expected_idx1_p1
        # 断言获取字符串表示的 p1 的位置为 expected_idx1_p1
        assert idx0.get_loc(str(p1)) == expected_idx1_p1
        # 断言获取 p2 的位置为 expected_idx1_p2
        assert idx0.get_loc(p2) == expected_idx1_p2
        # 断言获取字符串表示的 p2 的位置为 expected_idx1_p2
        assert idx0.get_loc(str(p2)) == expected_idx1_p2

        # 定义异常消息
        msg = "Cannot interpret 'foo' as period"
        # 使用 pytest 断言捕获期望的 KeyError 异常，其消息为 msg
        with pytest.raises(KeyError, match=msg):
            idx0.get_loc("foo")
        # 使用 pytest 断言捕获期望的 KeyError 异常，其消息匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^1\.1$"):
            idx0.get_loc(1.1)

        # 使用 pytest 断言捕获期望的 InvalidIndexError 异常，其消息与 idx0 对象的字符串表示形式匹配
        with pytest.raises(InvalidIndexError, match=re.escape(str(idx0))):
            idx0.get_loc(idx0)

        # 创建一个包含重复周期的周期索引对象 idx1，包含周期 p1, p1, p2
        idx1 = PeriodIndex([p1, p1, p2])
        expected_idx1_p1 = slice(0, 2)
        expected_idx1_p2 = 2

        # 断言获取 p1 的位置为 expected_idx1_p1
        assert idx1.get_loc(p1) == expected_idx1_p1
        # 断言获取字符串表示的 p1 的位置为 expected_idx1_p1
        assert idx1.get_loc(str(p1)) == expected_idx1_p1
        # 断言获取 p2 的位置为 expected_idx1_p2
        assert idx1.get_loc(p2) == expected_idx1_p2
        # 断言获取字符串表示的 p2 的位置为 expected_idx1_p2
        assert idx1.get_loc(str(p2)) == expected_idx1_p2

        # 定义异常消息
        msg = "Cannot interpret 'foo' as period"
        # 使用 pytest 断言捕获期望的 KeyError 异常，其消息为 msg
        with pytest.raises(KeyError, match=msg):
            idx1.get_loc("foo")

        # 使用 pytest 断言捕获期望的 KeyError 异常，其消息匹配指定的正则表达式
        with pytest.raises(KeyError, match=r"^1\.1$"):
            idx1.get_loc(1.1)

        # 使用 pytest 断言捕获期望的 InvalidIndexError 异常，其消息与 idx1 对象的字符串表示形式匹配
        with pytest.raises(InvalidIndexError, match=re.escape(str(idx1))):
            idx1.get_loc(idx1)

        # 创建一个包含重复和非单调递增周期的周期索引对象 idx2，包含周期 p2, p1, p2
        idx2 = PeriodIndex([p2, p1, p2])
        expected_idx2_p1 = 1
        expected_idx2_p2 = np.array([True, False, True])

        # 断言获取 p1 的位置为 expected_idx2_p1
        assert idx2.get_loc(p1) == expected_idx2_p1
        # 断言获取字符串表示的 p1 的位置为 expected_idx2_p1
        assert idx2.get_loc(str(p1)) == expected_idx2_p1
        # 使用 pandas.testing 模块断言 numpy 数组相等
        tm.assert_numpy_array_equal(idx2.get_loc(p2), expected_idx2_p2)
        tm.assert_numpy_array_equal(idx2.get_loc(str(p2)), expected_idx2_p2)
    # 测试获取整数位置索引的方法
    def test_get_loc_integer(self):
        # 创建一个日期范围对象，从"2016-01-01"开始，持续3天
        dti = date_range("2016-01-01", periods=3)
        # 将日期范围对象转换为以天为周期的周期对象
        pi = dti.to_period("D")
        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"16801"
        with pytest.raises(KeyError, match="16801"):
            # 尝试获取整数位置索引为16801的元素
            pi.get_loc(16801)

        # 将日期范围对象转换为以年为周期的周期对象，其中存在重复，所有序数都为46
        pi2 = dti.to_period("Y")
        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"46"
        with pytest.raises(KeyError, match="46"):
            # 尝试获取整数位置索引为46的元素
            pi2.get_loc(46)

    # 测试当传入无效字符串时是否会抛出 KeyError 异常
    def test_get_loc_invalid_string_raises_keyerror(self):
        # 创建一个从"2000"开始，持续3个周期的周期范围对象，命名为"A"
        pi = period_range("2000", periods=3, name="A")
        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"A"
        with pytest.raises(KeyError, match="A"):
            # 尝试获取字符串索引为"A"的元素位置
            pi.get_loc("A")

        # 创建一个序列对象，其索引为周期范围对象 pi，数据为 [1, 2, 3]
        ser = Series([1, 2, 3], index=pi)
        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"A"
        with pytest.raises(KeyError, match="A"):
            # 尝试使用 loc 方法获取索引为"A"的元素
            ser.loc["A"]

        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"A"
        with pytest.raises(KeyError, match="A"):
            # 尝试直接通过索引获取序列中索引为"A"的元素
            ser["A"]

        # 断言字符串"A"不在序列对象 ser 的索引中
        assert "A" not in ser
        # 断言字符串"A"不在周期范围对象 pi 中
        assert "A" not in pi

    # 测试当频率不匹配时是否会抛出 KeyError 异常
    def test_get_loc_mismatched_freq(self):
        # 创建一个日期范围对象，从"2016-01-01"开始，持续3天
        dti = date_range("2016-01-01", periods=3)
        # 将日期范围对象转换为以天为周期的周期对象
        pi = dti.to_period("D")
        # 将日期范围对象转换为以周为周期的周期对象
        pi2 = dti.to_period("W")
        # 将 pi 转换为与 pi2 具有相同 i8 表示的周期对象
        pi3 = pi.view(pi2.dtype)

        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"W-SUN"
        with pytest.raises(KeyError, match="W-SUN"):
            # 尝试获取 pi 中 pi2 的第一个元素的位置索引
            pi.get_loc(pi2[0])

        # 使用 pytest 检查是否会抛出 KeyError，并且匹配异常信息为"W-SUN"
        with pytest.raises(KeyError, match="W-SUN"):
            # 即使 pi 和 pi3 有相同的 i8 值，也尝试获取 pi 中 pi3 的第一个元素的位置索引
            pi.get_loc(pi3[0])
class TestGetIndexer:
    # 测试获取索引器功能

    def test_get_indexer(self):
        # 测试用例 GH 17717

        # 创建一些日期周期对象
        p1 = Period("2017-09-01")
        p2 = Period("2017-09-04")
        p3 = Period("2017-09-07")

        # 创建另外一些日期周期对象
        tp0 = Period("2017-08-31")
        tp1 = Period("2017-09-02")
        tp2 = Period("2017-09-05")
        tp3 = Period("2017-09-09")

        # 创建日期周期索引对象
        idx = PeriodIndex([p1, p2, p3])

        # 测试使用相同索引获取索引器
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        # 创建目标日期周期索引对象
        target = PeriodIndex([tp0, tp1, tp2, tp3])

        # 测试使用不同方法获取索引器
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2, -1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 0, 1, 2], dtype=np.intp)
        )

        # 测试使用带有容差的最近方法获取索引器
        res = idx.get_indexer(target, "nearest", tolerance=Timedelta("1 day"))
        tm.assert_numpy_array_equal(res, np.array([0, 0, 1, -1], dtype=np.intp))

    def test_get_indexer_mismatched_dtype(self):
        # 检查返回全部为 -1 并且不会引发或错误地强制转换数据类型

        # 创建日期范围对象和对应的周期对象
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")
        pi2 = dti.to_period("W")

        # 预期结果为全部为 -1 的数组
        expected = np.array([-1, -1, -1], dtype=np.intp)

        # 测试在不同方向上获取索引器
        result = pi.get_indexer(dti)
        tm.assert_numpy_array_equal(result, expected)

        result = dti.get_indexer(pi)
        tm.assert_numpy_array_equal(result, expected)

        result = pi.get_indexer(pi2)
        tm.assert_numpy_array_equal(result, expected)

        # 使用非唯一索引获取索引器，期望结果相同
        result = pi.get_indexer_non_unique(dti)[0]
        tm.assert_numpy_array_equal(result, expected)

        result = dti.get_indexer_non_unique(pi)[0]
        tm.assert_numpy_array_equal(result, expected)

        result = pi.get_indexer_non_unique(pi2)[0]
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_mismatched_dtype_different_length(self, non_comparable_idx):
        # 没有指定方法时，我们不检查不等式，所以全部为缺失，但不会引发异常

        # 创建日期范围对象和对应的周期对象
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")

        # 使用非可比较索引获取索引器
        other = non_comparable_idx

        # 获取索引器并比较结果
        res = pi[:-1].get_indexer(other)
        expected = -np.ones(other.shape, dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    # 测试在索引器方法中处理不兼容的数据类型异常
    def test_get_indexer_mismatched_dtype_with_method(self, non_comparable_idx, method):
        # 创建一个日期范围对象，从"2016-01-01"开始，3个时间点
        dti = date_range("2016-01-01", periods=3)
        # 将日期范围对象转换为周期（每日）的周期索引对象
        pi = dti.to_period("D")

        # 将参数传递的非兼容索引对象存储在变量other中
        other = non_comparable_idx

        # 准备一个正则表达式，用于匹配异常消息，指示无法比较不同数据类型的周期索引和other的数据类型
        msg = re.escape(f"Cannot compare dtypes {pi.dtype} and {other.dtype}")
        # 使用pytest断言检查是否抛出TypeError异常，并匹配预期的消息
        with pytest.raises(TypeError, match=msg):
            pi.get_indexer(other, method=method)

        # 遍历不同的数据类型["object", "category"]
        for dtype in ["object", "category"]:
            # 将other转换为当前遍历的数据类型
            other2 = other.astype(dtype)
            # 如果other的原始数据类型是PeriodIndex，并且当前遍历数据类型是"object"，则跳过后续操作
            if dtype == "object" and isinstance(other, PeriodIndex):
                continue
            # 根据数据类型不同，选择不同的错误消息模式
            # 准备正则表达式，用于匹配不同数据类型导致的TypeError异常消息
            msg = "|".join(
                [
                    re.escape(msg)
                    for msg in (
                        f"Cannot compare dtypes {pi.dtype} and {other.dtype}",
                        " not supported between instances of ",
                    )
                ]
            )
            # 使用pytest断言检查是否抛出TypeError异常，并匹配预期的消息
            with pytest.raises(TypeError, match=msg):
                pi.get_indexer(other2, method=method)

    # 测试处理非唯一索引的情况
    def test_get_indexer_non_unique(self):
        # GH 17717
        # 创建四个周期对象
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")
        p3 = Period("2017-09-04")
        p4 = Period("2017-09-05")

        # 创建两个周期索引对象，其中idx1包含两个p1周期
        idx1 = PeriodIndex([p1, p2, p1])
        # 创建idx2周期索引对象，包含四个不同的周期对象
        idx2 = PeriodIndex([p2, p1, p3, p4])

        # 调用idx1的方法，获取处理非唯一索引情况的结果
        result = idx1.get_indexer_non_unique(idx2)
        # 预期的索引数组，指示idx1中每个元素在idx2中的索引位置，-1表示未匹配到
        expected_indexer = np.array([1, 0, 2, -1, -1], dtype=np.intp)
        # 预期的缺失索引数组，指示idx2中未在idx1中找到匹配的索引位置
        expected_missing = np.array([2, 3], dtype=np.intp)

        # 使用pandas的测试工具函数，断言result的第一个数组与expected_indexer相等
        tm.assert_numpy_array_equal(result[0], expected_indexer)
        # 使用pandas的测试工具函数，断言result的第二个数组与expected_missing相等
        tm.assert_numpy_array_equal(result[1], expected_missing)

    # TODO: This method came from test_period; de-dup with version above
    # 定义单元测试方法 test_get_indexer2，用于测试索引器功能
    def test_get_indexer2(self):
        # 创建时间索引 idx，从 "2000-01-01" 开始，包含 3 个时间点，按小时频率生成，从起始时间开始
        idx = period_range("2000-01-01", periods=3).asfreq("h", how="start")
        # 使用测试工具方法验证 idx.get_indexer(idx) 的结果是否与 np.array([0, 1, 2], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        # 创建目标时间索引 target，包含三个时间点 ["1999-12-31T23", "2000-01-01T12", "2000-01-02T01"]，按小时频率
        target = PeriodIndex(
            ["1999-12-31T23", "2000-01-01T12", "2000-01-02T01"], freq="h"
        )
        # 使用测试工具方法验证 idx.get_indexer(target, "pad") 的结果是否与 np.array([-1, 0, 1], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        # 使用测试工具方法验证 idx.get_indexer(target, "backfill") 的结果是否与 np.array([0, 1, 2], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        # 使用测试工具方法验证 idx.get_indexer(target, "nearest") 的结果是否与 np.array([0, 1, 1], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )
        # 使用测试工具方法验证 idx.get_indexer(target, "nearest", tolerance="1 hour") 的结果是否与 np.array([0, -1, 1], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance="1 hour"),
            np.array([0, -1, 1], dtype=np.intp),
        )

        # 预期抛出 ValueError 异常，消息为 "Input has different freq=None from PeriodArray\\(freq=h\\)"
        msg = "Input has different freq=None from PeriodArray\\(freq=h\\)"
        with pytest.raises(ValueError, match=msg):
            # 验证 idx.get_indexer(target, "nearest", tolerance="1 minute") 抛出预期异常
            idx.get_indexer(target, "nearest", tolerance="1 minute")

        # 使用测试工具方法验证 idx.get_indexer(target, "nearest", tolerance="1 day") 的结果是否与 np.array([0, 1, 1], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance="1 day"),
            np.array([0, 1, 1], dtype=np.intp),
        )
        
        # 定义时间差列表 tol_raw
        tol_raw = [
            Timedelta("1 hour"),
            Timedelta("1 hour"),
            np.timedelta64(1, "D"),
        ]
        # 使用测试工具方法验证 idx.get_indexer(target, "nearest", tolerance=[np.timedelta64(x) for x in tol_raw]) 的结果是否与 np.array([0, -1, 1], dtype=np.intp) 相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(
                target, "nearest", tolerance=[np.timedelta64(x) for x in tol_raw]
            ),
            np.array([0, -1, 1], dtype=np.intp),
        )
        
        # 定义时间差列表 tol_bad
        tol_bad = [
            Timedelta("2 hour").to_timedelta64(),
            Timedelta("1 hour").to_timedelta64(),
            np.timedelta64(1, "M"),
        ]
        # 预期抛出 libperiod.IncompatibleFrequency 异常，消息为 "Input has different freq=None from"
        with pytest.raises(
            libperiod.IncompatibleFrequency, match="Input has different freq=None from"
        ):
            # 验证 idx.get_indexer(target, "nearest", tolerance=tol_bad) 抛出预期异常
            idx.get_indexer(target, "nearest", tolerance=tol_bad)
class TestWhere:
    # 定义测试类 TestWhere

    def test_where(self, listlike_box):
        # 定义测试方法 test_where，接受一个参数 listlike_box

        # 创建一个包含 5 个日期的 PeriodIndex 对象，频率为每天
        i = period_range("20130101", periods=5, freq="D")
        
        # 创建一个长度与 i 相同的全为 True 的条件列表
        cond = [True] * len(i)
        
        # 预期的输出结果是 i 本身
        expected = i
        
        # 使用 listlike_box 处理条件，得到处理后的结果
        result = i.where(listlike_box(cond))
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 创建一个长度为 len(i) 的条件列表，除第一个元素外全为 True
        cond = [False] + [True] * (len(i) - 1)
        
        # 预期的输出是一个新的 PeriodIndex 对象，第一个元素为 NaT，其余与 i[1:] 相同
        expected = PeriodIndex([NaT] + i[1:].tolist(), freq="D")
        
        # 使用 listlike_box 处理条件，得到处理后的结果
        result = i.where(listlike_box(cond))
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

    def test_where_other(self):
        # 定义测试方法 test_where_other

        # 创建一个包含 5 个日期的 PeriodIndex 对象，频率为每天
        i = period_range("20130101", periods=5, freq="D")
        
        # 对于每个数组 arr，在不为 NaN 或 NaT 的条件下使用 i，否则使用 arr
        for arr in [np.nan, NaT]:
            result = i.where(notna(i), other=arr)
            expected = i
            tm.assert_index_equal(result, expected)

        # 复制 i 到 i2
        i2 = i.copy()
        
        # 创建一个新的 PeriodIndex 对象 i2，首两个元素为 NaT，其余与 i[2:] 相同
        i2 = PeriodIndex([NaT, NaT] + i[2:].tolist(), freq="D")
        
        # 使用 notna(i2) 条件，若为 True 使用 i2，否则使用 i
        result = i.where(notna(i2), i2)
        
        # 断言 result 与 i2 相等
        tm.assert_index_equal(result, i2)

        # 复制 i 到 i2
        i2 = i.copy()
        
        # 创建一个新的 PeriodIndex 对象 i2，首两个元素为 NaT，其余与 i[2:] 相同
        i2 = PeriodIndex([NaT, NaT] + i[2:].tolist(), freq="D")
        
        # 使用 notna(i2) 条件，若为 True 使用 i2.values，否则使用 i
        result = i.where(notna(i2), i2.values)
        
        # 断言 result 与 i2 相等
        tm.assert_index_equal(result, i2)

    def test_where_invalid_dtypes(self):
        # 定义测试方法 test_where_invalid_dtypes

        # 创建一个包含 5 个日期的 PeriodIndex 对象，频率为每天
        pi = period_range("20130101", periods=5, freq="D")

        # 获取 pi 从第三个元素开始的列表
        tail = pi[2:].tolist()
        
        # 创建一个新的 PeriodIndex 对象 i2，首两个元素为 NaT，其余与 tail 相同
        i2 = PeriodIndex([NaT, NaT] + tail, freq="D")
        
        # 使用 notna(i2) 条件，若为 True 使用 i2.asi8，否则使用 pi
        mask = notna(i2)
        result = pi.where(mask, i2.asi8)
        
        # 创建一个预期的 Index 对象，首两个元素为 NaT._value，其余与 tail 相同，数据类型为 object
        expected = pd.Index([NaT._value, NaT._value] + tail, dtype=object)
        
        # 断言 expected 的第一个元素为整数类型
        assert isinstance(expected[0], int)
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 将 i2.asi8 视图转换为 timedelta64[ns] 类型
        tdi = i2.asi8.view("timedelta64[ns]")
        
        # 创建一个预期的 Index 对象，首两个元素为 tdi 的前两个元素，其余与 tail 相同，数据类型为 object
        expected = pd.Index([tdi[0], tdi[1]] + tail, dtype=object)
        
        # 断言 expected 的第一个元素为 np.timedelta64 类型
        assert isinstance(expected[0], np.timedelta64)
        
        # 使用 notna(i2) 条件，若为 True 使用 tdi，否则使用 pi
        result = pi.where(mask, tdi)
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 将 i2 转换为 timestamp("s") 类型
        dti = i2.to_timestamp("s")
        
        # 创建一个预期的 Index 对象，首个元素为 NaT，其余与 tail 相同，数据类型为 object
        expected = pd.Index([dti[0], dti[1]] + tail, dtype=object)
        
        # 断言 expected 的第一个元素为 NaT
        assert expected[0] is NaT
        
        # 使用 notna(i2) 条件，若为 True 使用 dti，否则使用 pi
        result = pi.where(mask, dti)
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

        # 创建一个时间增量为 4 天的 Timedelta 对象 td
        td = Timedelta(days=4)
        
        # 创建一个预期的 Index 对象，所有元素为 td，数据类型为 object
        expected = pd.Index([td, td] + tail, dtype=object)
        
        # 断言 expected 的第一个元素与 td 相等
        assert expected[0] == td
        
        # 使用 notna(i2) 条件，若为 True 使用 td，否则使用 pi
        result = pi.where(mask, td)
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self):
        # 定义测试方法 test_where_mismatched_nat

        # 创建一个包含 5 个日期的 PeriodIndex 对象，频率为每天
        pi = period_range("20130101", periods=5, freq="D")
        
        # 创建一个包含 True、False 等值的布尔数组 cond
        cond = np.array([True, False, True, True, False])

        # 创建一个 timedelta64 类型的 NaT
        tdnat = np.timedelta64("NaT", "ns")
        
        # 创建一个预期的 Index 对象，首个元素为 pi[0]，第二个元素为 tdnat，其余与 pi[2:] 相同，数据类型为 object
        expected = pd.Index([pi[0], tdnat, pi[2], pi[3], tdnat], dtype=object)
        
        # 断言 expected 的第二个元素为 tdnat
        assert expected[1] is tdnat
        
        # 使用 cond 条件，若为 True 使用 tdnat，否则使用 pi
        result = pi.where(cond, tdnat)
        
        # 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)
    def test_take(self):
        # 定义时间范围，生成日期索引，频率为每天，起始日期为"2011-01-01"，结束日期为"2011-01-31"，索引名称为"idx"
        idx1 = period_range("2011-01-01", "2011-01-31", freq="D", name="idx")

        # 对于每个日期索引 idx1 中的每个索引 idx 执行以下操作
        for idx in [idx1]:
            # 取出索引位置为0的时间段，并断言结果为 "2011-01-01" 这一天的 Period 对象
            result = idx.take([0])
            assert result == Period("2011-01-01", freq="D")

            # 取出索引位置为5的时间段，并断言结果为 "2011-01-06" 这一天的 Period 对象
            result = idx.take([5])
            assert result == Period("2011-01-06", freq="D")

            # 取出索引位置为0、1、2的时间段，并断言结果与期望的日期索引对象 expected 相等
            result = idx.take([0, 1, 2])
            expected = period_range("2011-01-01", "2011-01-03", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == "D"
            assert result.freq == expected.freq

            # 取出索引位置为0、2、4的时间段，并断言结果与期望的日期索引对象 expected 相等
            result = idx.take([0, 2, 4])
            expected = PeriodIndex(
                ["2011-01-01", "2011-01-03", "2011-01-05"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 取出索引位置为7、4、1的时间段，并断言结果与期望的日期索引对象 expected 相等
            result = idx.take([7, 4, 1])
            expected = PeriodIndex(
                ["2011-01-08", "2011-01-05", "2011-01-02"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 取出索引位置为3、2、5的时间段，并断言结果与期望的日期索引对象 expected 相等
            result = idx.take([3, 2, 5])
            expected = PeriodIndex(
                ["2011-01-04", "2011-01-03", "2011-01-06"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            # 取出索引位置为倒数第3、2、5的时间段，并断言结果与期望的日期索引对象 expected 相等
            result = idx.take([-3, 2, 5])
            expected = PeriodIndex(
                ["2011-01-29", "2011-01-03", "2011-01-06"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

    def test_take_misc(self):
        # 生成时间段索引 index，从"1/1/10"到"12/31/12"，频率为每天，索引名称为"idx"
        index = period_range(start="1/1/10", end="12/31/12", freq="D", name="idx")
        # 期望的时间段索引 expected，包含特定日期的 PeriodIndex 对象
        expected = PeriodIndex(
            [
                datetime(2010, 1, 6),
                datetime(2010, 1, 7),
                datetime(2010, 1, 9),
                datetime(2010, 1, 13),
            ],
            freq="D",
            name="idx",
        )

        # 取出索引位置为5、6、8、12的时间段，分别赋给 taken1 和 taken2
        taken1 = index.take([5, 6, 8, 12])
        taken2 = index[[5, 6, 8, 12]]

        # 对于每个被取出的时间段索引 taken，断言其与期望的时间段索引 expected 相等
        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, PeriodIndex)
            assert taken.freq == index.freq
            assert taken.name == expected.name
    # 定义一个测试函数 test_take_fill_value，用于测试 PeriodIndex 类的 take 方法的不同参数组合
    def test_take_fill_value(self):
        # GH#12631
        # 创建一个 PeriodIndex 对象 idx，包含日期字符串，设置名称为 "xxx"，频率为 "D"
        idx = PeriodIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx", freq="D"
        )
        # 调用 take 方法，传入一个数组 [1, 0, -1]，返回结果保存在 result 中
        result = idx.take(np.array([1, 0, -1]))
        # 期望的 PeriodIndex 对象，与 result 进行比较
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", freq="D"
        )
        # 使用 tm.assert_index_equal 方法断言 result 是否等于 expected

        # fill_value=True
        # 再次调用 take 方法，传入数组 [1, 0, -1] 和 fill_value=True，期望结果保存在 result 中
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        # 期望的 PeriodIndex 对象，包含一个 NaT（Not a Time）值，与 result 进行比较
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "NaT"], name="xxx", freq="D"
        )
        # 使用 tm.assert_index_equal 方法断言 result 是否等于 expected

        # allow_fill=False
        # 再次调用 take 方法，传入数组 [1, 0, -1]、allow_fill=False 和 fill_value=True，期望结果保存在 result 中
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        # 期望的 PeriodIndex 对象，与 result 进行比较，不包含 fill_value 的影响
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", freq="D"
        )
        # 使用 tm.assert_index_equal 方法断言 result 是否等于 expected

        # 设置错误消息，用于下面的异常断言
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        # 使用 pytest.raises 检查是否会抛出 ValueError 异常，匹配 msg 提供的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 take 方法，传入数组 [1, 0, -2] 和 fill_value=True，预期抛出 ValueError 异常
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            # 调用 take 方法，传入数组 [1, 0, -5] 和 fill_value=True，预期抛出 ValueError 异常
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 设置错误消息，用于下面的异常断言
        msg = "index -5 is out of bounds for( axis 0 with)? size 3"
        # 使用 pytest.raises 检查是否会抛出 IndexError 异常，匹配 msg 提供的错误消息
        with pytest.raises(IndexError, match=msg):
            # 调用 take 方法，传入数组 [1, -5]，预期抛出 IndexError 异常
            idx.take(np.array([1, -5]))
class TestGetValue:
    @pytest.mark.parametrize("freq", ["h", "D"])
    def test_get_value_datetime_hourly(self, freq):
        # get_loc and get_value should treat datetime objects symmetrically
        # TODO: this test used to test get_value, which is removed in 2.0.
        #  should this test be moved somewhere, or is what's left redundant?
        # 创建一个日期范围对象，从"2016-01-01"开始，每个月起始日作为时间点，共3个时间点
        dti = date_range("2016-01-01", periods=3, freq="MS")
        # 根据给定频率转换日期范围对象为周期对象
        pi = dti.to_period(freq)
        # 创建一个序列对象，索引为周期对象，数值为7到9的范围
        ser = Series(range(7, 10), index=pi)

        # 获取日期范围对象的第一个日期时间点
        ts = dti[0]

        # 断言：期间索引对象中第一个时间点的位置为0
        assert pi.get_loc(ts) == 0
        # 断言：序列对象中第一个时间点对应的数值为7
        assert ser[ts] == 7
        # 断言：使用loc访问序列对象中第一个时间点对应的数值为7
        assert ser.loc[ts] == 7

        # 创建一个时间点对象，基于第一个时间点加3个小时
        ts2 = ts + Timedelta(hours=3)
        # 如果频率为"h"
        if freq == "h":
            # 断言：期间索引对象中找不到ts2对应的时间点，抛出KeyError异常
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                pi.get_loc(ts2)
            # 断言：序列对象中找不到ts2对应的时间点，抛出KeyError异常
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                ser[ts2]
            # 断言：使用loc访问序列对象中找不到ts2对应的时间点，抛出KeyError异常
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                ser.loc[ts2]
        else:
            # 断言：期间索引对象中ts2对应的时间点位置为0
            assert pi.get_loc(ts2) == 0
            # 断言：序列对象中ts2对应的数值为7
            assert ser[ts2] == 7
            # 断言：使用loc访问序列对象中ts2对应的数值为7
            assert ser.loc[ts2] == 7


class TestContains:
    def test_contains(self):
        # GH 17717
        # 创建四个周期对象，分别表示2017-09-01到2017-09-04
        p0 = Period("2017-09-01")
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")
        p3 = Period("2017-09-04")

        # 将周期对象列表ps0转换为周期索引对象idx0
        ps0 = [p0, p1, p2]
        idx0 = PeriodIndex(ps0)

        # 遍历周期对象列表ps0，断言每个周期对象p在周期索引对象idx0中存在
        for p in ps0:
            assert p in idx0
            assert str(p) in idx0

        # GH#31172
        # 断言高分辨率的周期对象不在周期索引对象idx0中
        key = "2017-09-01 00:00:01"
        assert key not in idx0
        # 断言使用get_loc方法找不到key对应的位置，抛出KeyError异常
        with pytest.raises(KeyError, match=key):
            idx0.get_loc(key)

        # 断言字符串"2017-09"在周期索引对象idx0中存在
        assert "2017-09" in idx0

        # 断言周期对象p3不在周期索引对象idx0中
        assert p3 not in idx0

    def test_contains_freq_mismatch(self):
        # 创建一个周期范围对象rng，从"2007-01"开始，频率为"M"，共10个周期
        rng = period_range("2007-01", freq="M", periods=10)

        # 断言周期对象"2007-01"在周期范围对象rng中存在
        assert Period("2007-01", freq="M") in rng
        # 断言周期对象"2007-01"，但频率为"D"，不在周期范围对象rng中
        assert Period("2007-01", freq="D") not in rng
        # 断言周期对象"2007-01"，但频率为"2M"，不在周期范围对象rng中
        assert Period("2007-01", freq="2M") not in rng

    def test_contains_nat(self):
        # see gh-13582
        # 创建一个周期范围对象idx，从"2007-01"开始，频率为"M"，共10个周期
        idx = period_range("2007-01", freq="M", periods=10)
        # 断言NaT不在周期范围对象idx中
        assert NaT not in idx
        # 断言None不在周期范围对象idx中
        assert None not in idx
        # 断言float("nan")不在周期范围对象idx中
        assert float("nan") not in idx
        # 断言np.nan不在周期范围对象idx中
        assert np.nan not in idx

        # 创建一个周期索引对象idx，包含三个周期对象["2011-01", "NaT", "2011-02"]，频率为"M"
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="M")
        # 断言NaT在周期索引对象idx中
        assert NaT in idx
        # 断言None在周期索引对象idx中
        assert None in idx
        # 断言float("nan")在周期索引对象idx中
        assert float("nan") in idx
        # 断言np.nan在周期索引对象idx中
        assert np.nan in idx
    # 定义一个测试函数，用于测试不匹配类型的情况
    def test_asof_locs_mismatched_type(self):
        # 创建一个日期范围对象，从"2016-01-01"开始，包含3个时间点
        dti = date_range("2016-01-01", periods=3)
        # 将日期范围对象转换为日期周期对象（按天）
        pi = dti.to_period("D")
        # 将日期范围对象转换为日期周期对象（按小时）
        pi2 = dti.to_period("h")

        # 创建一个布尔类型的掩码数组
        mask = np.array([0, 1, 0], dtype=bool)

        # 设置错误消息，用于捕获类型错误异常
        msg = "must be DatetimeIndex or PeriodIndex"
        # 使用 pytest 检查是否会引发类型错误异常，匹配指定错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用 asof_locs 方法，传入一个不匹配的索引类型 pd.Index(pi.asi8, dtype=np.int64)，和掩码数组
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.int64), mask)

        with pytest.raises(TypeError, match=msg):
            # 同上，传入不匹配的索引类型 pd.Index(pi.asi8, dtype=np.float64)
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.float64), mask)

        with pytest.raises(TypeError, match=msg):
            # 使用 TimedeltaIndex 调用 asof_locs 方法，预期引发类型错误异常
            pi.asof_locs(dti - dti, mask)

        # 设置错误消息，用于捕获频率不兼容异常
        msg = "Input has different freq=h"
        # 使用 pytest 检查是否会引发不兼容频率异常，匹配指定错误消息
        with pytest.raises(libperiod.IncompatibleFrequency, match=msg):
            # 调用 asof_locs 方法，传入不兼容频率的日期周期对象 pi2 和掩码数组
            pi.asof_locs(pi2, mask)
```