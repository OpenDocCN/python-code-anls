# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_setops.py`

```
from datetime import (
    datetime,         # 导入 datetime 模块中的 datetime 类
    timedelta,        # 导入 datetime 模块中的 timedelta 类
    timezone,         # 导入 datetime 模块中的 timezone 类
)

import numpy as np                   # 导入 numpy 库，并重命名为 np
import pytest                         # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators 模块，并重命名为 td

import pandas as pd                   # 导入 pandas 库，并重命名为 pd
from pandas import (                 # 从 pandas 库中导入多个类和函数
    DataFrame,                       # 导入 DataFrame 类
    DatetimeIndex,                   # 导入 DatetimeIndex 类
    Index,                           # 导入 Index 类
    Series,                          # 导入 Series 类
    Timestamp,                       # 导入 Timestamp 类
    bdate_range,                     # 导入 bdate_range 函数
    date_range,                      # 导入 date_range 函数
)
import pandas._testing as tm          # 导入 pandas._testing 模块，并重命名为 tm

from pandas.tseries.offsets import (  # 从 pandas.tseries.offsets 模块中导入多个类
    BMonthEnd,                       # 导入 BMonthEnd 类
    Minute,                          # 导入 Minute 类
    MonthEnd,                        # 导入 MonthEnd 类
)

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)  # 定义起始时间和结束时间


class TestDatetimeIndexSetOps:
    tz = [                              # 定义一个时区列表
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Singapore",
        "dateutil/US/Pacific",
    ]

    # TODO: moved from test_datetimelike; dedup with version below
    def test_union2(self, sort):         # 定义测试方法 test_union2，接受 sort 参数
        everything = date_range("2020-01-01", periods=10)  # 创建一个日期范围对象 everything
        first = everything[:5]           # 取前5个日期作为 first
        second = everything[5:]           # 取后5个日期作为 second
        union = first.union(second, sort=sort)  # 对 first 和 second 进行 union 操作，根据 sort 参数排序
        tm.assert_index_equal(union, everything)  # 断言 union 的结果与 everything 相等

    @pytest.mark.parametrize("box", [np.array, Series, list])
    def test_union3(self, sort, box):    # 定义参数化测试方法 test_union3，接受 sort 和 box 参数
        everything = date_range("2020-01-01", periods=10)  # 创建一个日期范围对象 everything
        first = everything[:5]           # 取前5个日期作为 first
        second = everything[5:]           # 取后5个日期作为 second

        # GH 10149 support listlike inputs other than Index objects
        expected = first.union(second, sort=sort)  # 预期的 union 结果
        case = box(second.values)         # 将 second 的值转换为 box 类型
        result = first.union(case, sort=sort)  # 对 first 和 case 进行 union 操作，根据 sort 参数排序
        tm.assert_index_equal(result, expected)  # 断言 result 的结果与 expected 相等

    @pytest.mark.parametrize("tz", tz)
    def test_union(self, tz, sort):      # 定义参数化测试方法 test_union，接受 tz 和 sort 参数
        rng1 = date_range("1/1/2000", freq="D", periods=5, tz=tz)  # 创建一个带时区的日期范围对象 rng1
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)  # 创建一个带时区的日期范围对象 other1
        expected1 = date_range("1/1/2000", freq="D", periods=10, tz=tz)  # 创建预期的带时区的日期范围对象 expected1
        expected1_notsorted = DatetimeIndex(list(other1) + list(rng1))  # 创建未排序的日期索引对象 expected1_notsorted

        rng2 = date_range("1/1/2000", freq="D", periods=5, tz=tz)  # 创建一个带时区的日期范围对象 rng2
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)  # 创建一个带时区的日期范围对象 other2
        expected2 = date_range("1/1/2000", freq="D", periods=8, tz=tz)  # 创建预期的带时区的日期范围对象 expected2
        expected2_notsorted = DatetimeIndex(list(other2) + list(rng2[:3]))  # 创建未排序的日期索引对象 expected2_notsorted

        rng3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)  # 创建一个带时区的日期范围对象 rng3
        other3 = DatetimeIndex([], tz=tz).as_unit("ns")  # 创建一个空的带时区的 DatetimeIndex 对象 other3
        expected3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)  # 创建预期的带时区的日期范围对象 expected3
        expected3_notsorted = rng3       # 未排序的日期索引对象 expected3_notsorted 等于 rng3

        for rng, other, exp, exp_notsorted in [  # 遍历日期范围对象和对应的预期结果
            (rng1, other1, expected1, expected1_notsorted),
            (rng2, other2, expected2, expected2_notsorted),
            (rng3, other3, expected3, expected3_notsorted),
        ]:
            result_union = rng.union(other, sort=sort)  # 对 rng 和 other 进行 union 操作，根据 sort 参数排序
            tm.assert_index_equal(result_union, exp)  # 断言 union 的结果与 exp 相等

            result_union = other.union(rng, sort=sort)  # 对 other 和 rng 进行 union 操作，根据 sort 参数排序
            if sort is None:
                tm.assert_index_equal(result_union, exp)  # 断言 union 的结果与 exp 相等
            else:
                tm.assert_index_equal(result_union, exp_notsorted)  # 断言 union 的结果与 exp_notsorted 相等
    def test_union_coverage(self, sort):
        # 创建一个日期时间索引对象，包含三个日期
        idx = DatetimeIndex(["2000-01-03", "2000-01-01", "2000-01-02"])
        # 对索引进行排序并创建一个新的日期时间索引对象
        ordered = DatetimeIndex(idx.sort_values(), freq="infer")
        # 将排序后的索引与原始索引进行并集操作
        result = ordered.union(idx, sort=sort)
        # 断言结果与排序后的索引相等
        tm.assert_index_equal(result, ordered)

        # 对空的排序后的索引与排序后的索引进行并集操作
        result = ordered[:0].union(ordered, sort=sort)
        # 断言结果与排序后的索引相等
        tm.assert_index_equal(result, ordered)
        # 断言结果的频率与排序后的索引相同
        assert result.freq == ordered.freq

    def test_union_bug_1730(self, sort):
        # 创建两个日期范围对象，每个对象包含四个日期时间点
        rng_a = date_range("1/1/2012", periods=4, freq="3h")
        rng_b = date_range("1/1/2012", periods=4, freq="4h")

        # 对两个日期范围对象进行并集操作
        result = rng_a.union(rng_b, sort=sort)
        # 创建预期的日期时间索引列表，包含了合并后的日期时间点
        exp = list(rng_a) + list(rng_b[1:])
        # 如果没有排序要求，则对预期结果进行排序
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        # 断言结果与预期结果相等
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self, sort):
        # 创建左右两个日期时间索引对象，每个对象包含一个日期时间点
        left = DatetimeIndex(["2012-05-11 15:19:49.695000"])
        right = DatetimeIndex(
            [
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )

        # 对左右两个日期时间索引对象进行并集操作
        result = left.union(right, sort=sort)
        # 创建预期的日期时间索引列表，包含了合并后的日期时间点
        exp = DatetimeIndex(
            [
                "2012-05-11 15:19:49.695000",
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )
        # 如果没有排序要求，则对预期结果进行排序
        if sort is None:
            exp = exp.sort_values()
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self, sort):
        from pandas import DateOffset

        # 创建左右两个日期范围对象，右边的对象日期时间点增加15分钟
        left = date_range("2013-01-01", "2013-02-01")
        right = left + DateOffset(minutes=15)

        # 对左右两个日期范围对象进行并集操作
        result = left.union(right, sort=sort)
        # 创建预期的日期时间索引列表，包含了合并后的日期时间点
        exp = list(left) + list(right)
        # 如果没有排序要求，则对预期结果进行排序
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        # 断言结果与预期结果相等
        tm.assert_index_equal(result, exp)

    def test_union_freq_both_none(self, sort):
        # GH11086
        # 创建一个工作日日期范围对象，包含十个工作日日期
        expected = bdate_range("20150101", periods=10)
        expected._data.freq = None

        # 对工作日日期范围对象进行并集操作
        result = expected.union(expected, sort=sort)
        # 断言结果与预期结果相等
        tm.assert_index_equal(result, expected)
        # 断言结果的频率为None
        assert result.freq is None

    def test_union_freq_infer(self):
        # When taking the union of two DatetimeIndexes, we infer
        #  a freq even if the arguments don't have freq.  This matches
        #  TimedeltaIndex behavior.
        # 创建一个日期范围对象，包含五个日期时间点
        dti = date_range("2016-01-01", periods=5)
        left = dti[[0, 1, 3, 4]]
        right = dti[[2, 3, 1]]

        # 断言左右两个日期时间索引对象的频率为None
        assert left.freq is None
        assert right.freq is None

        # 对左右两个日期时间索引对象进行并集操作
        result = left.union(right)
        # 断言结果与原始日期范围对象相等
        tm.assert_index_equal(result, dti)
        # 断言结果的频率为每天("D")
        assert result.freq == "D"
    # 测试函数：test_union_dataframe_index，用于测试合并数据框架的索引
    def test_union_dataframe_index(self):
        # 创建日期范围 rng1 和相应的随机数据序列 s1
        rng1 = date_range("1/1/1999", "1/1/2012", freq="MS")
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        # 创建日期范围 rng2 和相应的随机数据序列 s2
        rng2 = date_range("1/1/1980", "12/1/2001", freq="MS")
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)

        # 创建包含 s1 和 s2 的数据框 df
        df = DataFrame({"s1": s1, "s2": s2})

        # 期望的合并后的日期范围 exp
        exp = date_range("1/1/1980", "1/1/2012", freq="MS")
        
        # 断言数据框索引与期望的合并后的日期范围相等
        tm.assert_index_equal(df.index, exp)

    # 测试函数：test_union_with_DatetimeIndex，测试 DatetimeIndex 的合并
    def test_union_with_DatetimeIndex(self, sort):
        # 创建索引 i1 和 i2
        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
        i2 = date_range(start="2012-01-03 00:00:00", periods=10, freq="D")

        # 使用 sort 参数合并 i1 和 i2
        i1.union(i2, sort=sort)

        # 尝试使用 sort 参数合并 i2 和 i1，但会导致 "AttributeError: can't set attribute" 错误
        i2.union(i1, sort=sort)

    # 测试函数：test_union_same_timezone_different_units，测试相同时区不同单位的索引合并
    def test_union_same_timezone_different_units(self):
        # 创建具有不同单位的日期范围 idx1 和 idx2
        idx1 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("ms")
        idx2 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")

        # 合并 idx1 和 idx2
        result = idx1.union(idx2)

        # 期望的合并结果 expected
        expected = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")
        
        # 断言合并后的结果与期望的结果相等
        tm.assert_index_equal(result, expected)

    # TODO: moved from test_datetimelike; de-duplicate with version below
    # 测试函数：test_intersection2，测试索引的交集
    def test_intersection2(self):
        # 创建日期范围 first 和截取后的日期范围 second
        first = date_range("2020-01-01", periods=10)
        second = first[5:]

        # 计算 first 和 second 的交集 intersect
        intersect = first.intersection(second)

        # 断言交集 intersect 与 second 相等
        tm.assert_index_equal(intersect, second)

        # GH 10149
        # 针对不同类型的数据结构 case 进行交集操作，验证结果是否与 second 相等
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            tm.assert_index_equal(result, second)

        # 创建第三种类型的索引 third
        third = Index(["a", "b", "c"])

        # 计算 first 和 third 的交集 result
        result = first.intersection(third)

        # 期望的交集结果 expected，这里因为无公共元素，期望结果为空索引
        expected = Index([], dtype=object)

        # 断言交集 result 与期望的结果 expected 相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [None, "Asia/Tokyo", "US/Eastern", "dateutil/US/Pacific"]
    )
    # 定义一个名为 test_intersection 的测试方法，带有两个参数 tz 和 sort
    def test_intersection(self, tz, sort):
        # 创建一个基础的日期范围，从 "6/1/2000" 到 "6/30/2000"，每天频率为 "D"，命名为 "idx"
        base = date_range("6/1/2000", "6/30/2000", freq="D", name="idx")

        # 如果目标日期范围具有相同的名称，则保留该名称
        rng2 = date_range("5/15/2000", "6/20/2000", freq="D", name="idx")
        expected2 = date_range("6/1/2000", "6/20/2000", freq="D", name="idx")

        # 如果目标日期范围具有不同的名称，则重置名称
        rng3 = date_range("5/15/2000", "6/20/2000", freq="D", name="other")
        expected3 = date_range("6/1/2000", "6/20/2000", freq="D", name=None)

        # 创建一个新的日期范围，从 "7/1/2000" 到 "7/31/2000"，每天频率为 "D"，命名为 "idx"
        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = DatetimeIndex([], freq="D", name="idx", dtype="M8[ns]")

        # 遍历不同的日期范围和期望结果的元组列表
        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            # 对基础日期范围和目标日期范围进行交集运算
            result = base.intersection(rng)
            # 断言交集结果与期望结果相等
            tm.assert_index_equal(result, expected)
            # 断言交集结果的频率与期望结果的频率相等
            assert result.freq == expected.freq

        # 创建一个非单调的日期索引，包括时区 tz 和名称 "idx"
        base = DatetimeIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"], tz=tz, name="idx"
        ).as_unit("ns")

        # 创建第二个日期索引，包括时区 tz 和名称 "idx"
        rng2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"], tz=tz, name="idx"
        ).as_unit("ns")
        expected2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name="idx"
        ).as_unit("ns")

        # 创建第三个日期索引，包括时区 tz 和名称 "other"
        rng3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            tz=tz,
            name="other",
        ).as_unit("ns")
        expected3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name=None
        ).as_unit("ns")

        # GH 7880 测试
        # 创建一个日期范围，从 "7/1/2000" 到 "7/31/2000"，每天频率为 "D"，带有时区 tz，命名为 "idx"
        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", tz=tz, name="idx")
        expected4 = DatetimeIndex([], tz=tz, name="idx").as_unit("ns")
        # 断言预期结果的频率为 None
        assert expected4.freq is None

        # 遍历不同的日期范围和期望结果的元组列表
        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            # 对基础日期范围和目标日期范围进行交集运算，如果参数 sort 不为 None，则对期望结果进行排序
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            # 断言交集结果与期望结果相等
            tm.assert_index_equal(result, expected)
            # 断言交集结果的频率与期望结果的频率相等
            assert result.freq == expected.freq
    # 测试空交集情况，针对给定的时区感知日期范围进行测试
    def test_intersection_empty(self, tz_aware_fixture, freq):
        # 将 tz_aware_fixture 赋给 tz
        tz = tz_aware_fixture
        # 创建一个日期范围 rng，从 "6/1/2000" 到 "6/15/2000"，使用给定的频率 freq 和时区 tz
        rng = date_range("6/1/2000", "6/15/2000", freq=freq, tz=tz)
        # 计算 rng[0:0] 和 rng 的交集
        result = rng[0:0].intersection(rng)
        # 断言结果长度为 0
        assert len(result) == 0
        # 断言结果的频率与 rng 的频率相同
        assert result.freq == rng.freq

        # 计算 rng 和 rng[0:0] 的交集
        result = rng.intersection(rng[0:0])
        # 断言结果长度为 0
        assert len(result) == 0
        # 断言结果的频率与 rng 的频率相同
        assert result.freq == rng.freq

        # 没有重叠的情况，GH#33604
        # 检查频率是否不为 "min"，非锚定偏移不保留频率信息
        check_freq = freq != "min"
        # 计算 rng[:3] 和 rng[-3:] 的交集
        result = rng[:3].intersection(rng[-3:])
        # 断言索引结果与 rng[:0] 相等
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # 非锚定偏移不保留频率信息时执行以下断言
            assert result.freq == rng.freq

        # 交换左右操作数
        result = rng[-3:].intersection(rng[:3])
        # 断言索引结果与 rng[:0] 相等
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # 非锚定偏移不保留频率信息时执行以下断言
            assert result.freq == rng.freq

    # 测试 Bug 1708 的情况
    def test_intersection_bug_1708(self):
        from pandas import DateOffset
        
        # 创建一个日期范围 index_1，从 "1/1/2012" 开始，周期为 4，频率为 "12h"
        index_1 = date_range("1/1/2012", periods=4, freq="12h")
        # 创建 index_2，是 index_1 偏移 1 小时后的结果
        index_2 = index_1 + DateOffset(hours=1)
        # 计算 index_1 和 index_2 的交集
        result = index_1.intersection(index_2)
        # 断言结果长度为 0
        assert len(result) == 0

    # 使用 tz 参数作为参数化的测试用例进行测试
    @pytest.mark.parametrize("tz", tz)
    def test_difference(self, tz, sort):
        # 定义一个日期字符串列表 rng_dates
        rng_dates = ["1/2/2000", "1/3/2000", "1/1/2000", "1/4/2000", "1/5/2000"]
        
        # 创建 DatetimeIndex rng1，使用 tz 参数
        rng1 = DatetimeIndex(rng_dates, tz=tz)
        # 创建一个日期范围 other1，从 "1/6/2000" 开始，每天一次，周期为 5，使用 tz 参数
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        # 创建期望结果 DatetimeIndex expected1，使用 tz 参数
        expected1 = DatetimeIndex(rng_dates, tz=tz)

        # 创建 DatetimeIndex rng2，使用 tz 参数
        rng2 = DatetimeIndex(rng_dates, tz=tz)
        # 创建一个日期范围 other2，从 "1/4/2000" 开始，每天一次，周期为 5，使用 tz 参数
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        # 创建期望结果 DatetimeIndex expected2，使用 tz 参数
        expected2 = DatetimeIndex(rng_dates[:3], tz=tz)

        # 创建 DatetimeIndex rng3，使用 tz 参数
        rng3 = DatetimeIndex(rng_dates, tz=tz)
        # 创建一个空的 DatetimeIndex other3，使用 tz 参数
        other3 = DatetimeIndex([], tz=tz)
        # 创建期望结果 DatetimeIndex expected3，使用 tz 参数
        expected3 = DatetimeIndex(rng_dates, tz=tz)

        # 遍历所有 rng、other、expected 组合
        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
        ]:
            # 计算 rng 和 other 的差集，sort 参数决定是否排序
            result_diff = rng.difference(other, sort)
            if sort is None and len(other):
                # 当 other 不为空且 sort 为 None 时，不进行排序 GH#24959
                expected = expected.sort_values()
            # 断言索引结果是否与期望结果 expected 相等
            tm.assert_index_equal(result_diff, expected)
    def test_difference_freq(self, sort):
        # GH14323: difference of DatetimeIndex should not preserve frequency
        
        # 创建一个日期范围，频率为每日，从2016年9月20日到2016年9月25日
        index = date_range("20160920", "20160925", freq="D")
        # 创建另一个日期范围，频率为每日，从2016年9月21日到2016年9月24日
        other = date_range("20160921", "20160924", freq="D")
        # 期望的结果是一个DatetimeIndex，包含日期"20160920"和"20160925"，数据类型为"datetime64[ns]"，没有特定频率
        expected = DatetimeIndex(["20160920", "20160925"], dtype="M8[ns]", freq=None)
        # 计算两个日期范围的差集，按照指定的排序规则
        idx_diff = index.difference(other, sort)
        # 断言计算得到的差集与期望的结果相等
        tm.assert_index_equal(idx_diff, expected)
        # 断言差集的频率属性与期望的结果相等
        tm.assert_attr_equal("freq", idx_diff, expected)

        # 当差集是原始范围的一个连续子集时，保留频率
        # 修改other的范围，使其成为index的一个连续子集
        other = date_range("20160922", "20160925", freq="D")
        # 重新计算差集
        idx_diff = index.difference(other, sort)
        # 期望的结果是一个DatetimeIndex，包含日期"20160920"和"20160921"，数据类型为"datetime64[ns]"，频率为每日
        expected = DatetimeIndex(["20160920", "20160921"], dtype="M8[ns]", freq="D")
        # 断言计算得到的差集与期望的结果相等
        tm.assert_index_equal(idx_diff, expected)


    def test_datetimeindex_diff(self, sort):
        # 创建一个具有季度结束频率的日期范围，从1997年12月31日开始，包含100个时间点
        dti1 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=100)
        # 创建另一个具有季度结束频率的日期范围，从1997年12月31日开始，包含98个时间点
        dti2 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=98)
        # 断言两个日期范围的差集长度为2
        assert len(dti1.difference(dti2, sort)) == 2

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Eastern"])
    def test_setops_preserve_freq(self, tz):
        # 创建一个从"1/1/2000"到"1/1/2002"的日期范围，带有时区信息tz
        rng = date_range("1/1/2000", "1/1/2002", name="idx", tz=tz)

        # 测试union操作，确保结果的名称、频率和时区与原始范围相同
        result = rng[:50].union(rng[50:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[30:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[60:100])
        assert result.name == rng.name
        assert result.freq is None
        assert result.tz == rng.tz

        # 测试intersection操作，确保结果的名称为None，频率与原始范围一致，时区与原始范围相同
        result = rng[:50].intersection(rng[25:75])
        assert result.name == rng.name
        assert result.freqstr == "D"
        assert result.tz == rng.tz

        # 创建一个没有频率的DatetimeIndex，名称为"other"
        nofreq = DatetimeIndex(list(rng[25:75]), name="other")
        # 测试union操作，确保结果的名称为None，频率与原始范围一致，时区与原始范围相同
        result = rng[:50].union(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        # 测试intersection操作，确保结果的名称为None，频率与原始范围一致，时区与原始范围相同
        result = rng[:50].intersection(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

    def test_intersection_non_tick_no_fastpath(self):
        # GH#42104
        # 创建一个具有季度结束频率的DatetimeIndex
        dti = DatetimeIndex(
            [
                "2018-12-31",
                "2019-03-31",
                "2019-06-30",
                "2019-09-30",
                "2019-12-31",
                "2020-03-31",
            ],
            freq="QE-DEC",
        )
        # 计算dti[::2]和dti[1::2]的交集
        result = dti[::2].intersection(dti[1::2])
        # 期望的结果是一个空的DatetimeIndex
        expected = dti[:0]
        # 断言计算得到的交集与期望的结果相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于测试日期时间索引的交集操作
    def test_dti_intersection(self):
        # 创建一个日期范围，从 "1/1/2011" 开始，包含 100 个小时频率的日期时间，时区设为 UTC
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        # 从日期时间索引中选择第 10 到第 89 个位置的时间，并反转顺序
        left = rng[10:90][::-1]
        # 从日期时间索引中选择第 20 到第 79 个位置的时间，并反转顺序
        right = rng[20:80][::-1]

        # 断言左侧时间索引的时区与原始日期时间索引的时区一致
        assert left.tz == rng.tz
        # 对左右两个时间索引进行交集操作
        result = left.intersection(right)
        # 断言交集结果的时区与左侧时间索引的时区一致
        assert result.tz == left.tz

    # 注意：对称性要求不存在，因此不同于差异操作
    @pytest.mark.parametrize("setop", ["union", "intersection", "symmetric_difference"])
    # 定义一个测试方法，测试日期时间索引的集合操作（联合、交集、对称差）
    def test_dti_setop_aware(self, setop):
        # 创建一个非重叠的日期范围，起始于 "2012-11-15 00:00:00"，包含 6 个小时频率的时间，时区为 US/Central
        rng = date_range("2012-11-15 00:00:00", periods=6, freq="h", tz="US/Central")

        # 创建另一个日期范围，起始于 "2012-11-15 12:00:00"，包含 6 个小时频率的时间，时区为 US/Eastern
        rng2 = date_range("2012-11-15 12:00:00", periods=6, freq="h", tz="US/Eastern")

        # 使用getattr函数根据setop参数执行集合操作（联合、交集、对称差）
        result = getattr(rng, setop)(rng2)

        # 将左右两个时间索引转换为 UTC 时区
        left = rng.tz_convert("UTC")
        right = rng2.tz_convert("UTC")
        # 使用getattr函数根据setop参数获取左右时间索引的期望结果
        expected = getattr(left, setop)(right)
        # 断言结果与期望的索引对象相等
        tm.assert_index_equal(result, expected)
        # 断言结果的时区与左侧时间索引的时区一致
        assert result.tz == left.tz
        # 如果结果长度大于0，则断言结果的第一个和最后一个时间对象的时区为 UTC
        if len(result):
            assert result[0].tz is timezone.utc
            assert result[-1].tz is timezone.utc

    # 定义一个测试方法，测试混合类型的日期时间索引的联合操作
    def test_dti_union_mixed(self):
        # 创建一个包含 Timestamp 和 NaT 的日期时间索引
        rng = DatetimeIndex([Timestamp("2011-01-01"), pd.NaT])
        # 创建另一个日期时间索引，包含 "2012-01-01" 和 "2012-01-02" 的时间，时区设为 Asia/Tokyo
        rng2 = DatetimeIndex(["2012-01-01", "2012-01-02"], tz="Asia/Tokyo")
        # 对两个日期时间索引执行联合操作
        result = rng.union(rng2)
        # 构建期望的索引对象
        expected = Index(
            [
                Timestamp("2011-01-01"),
                pd.NaT,
                Timestamp("2012-01-01", tz="Asia/Tokyo"),
                Timestamp("2012-01-02", tz="Asia/Tokyo"),
            ],
            dtype=object,
        )
        # 断言结果与期望的索引对象相等
        tm.assert_index_equal(result, expected)
    # 定义一个测试类 TestBusinessDatetimeIndex，用于测试日期时间索引操作的方法
    class TestBusinessDatetimeIndex:
        
        # 测试 union 方法，合并两个日期时间索引，可能包括排序选项
        def test_union(self, sort):
            # 创建一个工作日范围的日期时间索引 rng，起始和结束时间由外部变量 START 和 END 决定
            rng = bdate_range(START, END)
            
            # 情况1：重叠部分
            left = rng[:10]  # 取前10个日期时间作为左侧索引
            right = rng[5:10]  # 取索引第5到第10个日期时间作为右侧索引

            # 对左右两个索引进行合并操作，可以选择是否排序
            the_union = left.union(right, sort=sort)
            # 断言合并结果的类型为 DatetimeIndex
            assert isinstance(the_union, DatetimeIndex)

            # 情况2：不重叠，中间有间隔
            left = rng[:5]  # 取前5个日期时间作为左侧索引
            right = rng[10:]  # 取从索引第10个到最后一个日期时间作为右侧索引

            # 对左右两个索引进行合并操作，可以选择是否排序
            the_union = left.union(right, sort=sort)
            # 断言合并结果的类型为 Index
            assert isinstance(the_union, Index)

            # 情况3：不重叠，无间隔
            left = rng[:5]  # 取前5个日期时间作为左侧索引
            right = rng[5:10]  # 取索引第5到第10个日期时间作为右侧索引

            # 对左右两个索引进行合并操作，可以选择是否排序
            the_union = left.union(right, sort=sort)
            # 断言合并结果的类型为 DatetimeIndex
            assert isinstance(the_union, DatetimeIndex)

            # 情况4：顺序不影响结果
            if sort is None:
                # 如果未指定排序方式，测试反向合并是否与正向合并结果一致
                tm.assert_index_equal(right.union(left, sort=sort), the_union)
            else:
                # 如果指定了排序方式，预期结果为按照合并后的顺序形成的 DatetimeIndex
                expected = DatetimeIndex(list(right) + list(left))
                tm.assert_index_equal(right.union(left, sort=sort), expected)

            # 情况5：重叠但偏移不同的情况
            rng = date_range(START, END, freq=BMonthEnd())  # 创建一个按月末计算的日期时间索引

            # 对同一个索引进行自身的合并操作，可以选择是否排序
            the_union = rng.union(rng, sort=sort)
            # 断言合并结果的类型为 DatetimeIndex
            assert isinstance(the_union, DatetimeIndex)

        # 测试 union 方法，验证结果不可缓存
        def test_union_not_cacheable(self, sort):
            rng = date_range("1/1/2000", periods=50, freq=Minute())  # 创建一个每分钟的日期时间索引
            rng1 = rng[10:]  # 取从第10个时间点开始到结束的子索引 rng1
            rng2 = rng[:25]  # 取从开始到第25个时间点的子索引 rng2

            # 对两个子索引进行合并操作，可以选择是否排序
            the_union = rng1.union(rng2, sort=sort)
            if sort is None:
                # 如果未指定排序方式，断言合并结果与原索引 rng 相同
                tm.assert_index_equal(the_union, rng)
            else:
                # 如果指定了排序方式，预期结果为按顺序合并后的 DatetimeIndex
                expected = DatetimeIndex(list(rng[10:]) + list(rng[:10]))
                tm.assert_index_equal(the_union, expected)

            rng1 = rng[10:]  # 取从第10个时间点开始到结束的子索引 rng1
            rng2 = rng[15:35]  # 取从第15个时间点到第35个时间点的子索引 rng2

            # 对两个子索引进行合并操作，可以选择是否排序
            the_union = rng1.union(rng2, sort=sort)
            expected = rng[10:]  # 预期结果为从第10个时间点开始到结束的原索引 rng
            tm.assert_index_equal(the_union, expected)

        # 测试 intersection 方法，验证交集操作
        def test_intersection(self):
            rng = date_range("1/1/2000", periods=50, freq=Minute())  # 创建一个每分钟的日期时间索引
            rng1 = rng[10:]  # 取从第10个时间点开始到结束的子索引 rng1
            rng2 = rng[:25]  # 取从开始到第25个时间点的子索引 rng2

            # 对两个子索引进行交集操作
            the_int = rng1.intersection(rng2)
            expected = rng[10:25]  # 预期结果为从第10个到第25个时间点的日期时间索引
            tm.assert_index_equal(the_int, expected)
            # 断言交集结果的类型为 DatetimeIndex
            assert isinstance(the_int, DatetimeIndex)
            # 断言交集结果的频率与原索引 rng 相同
            assert the_int.freq == rng.freq

            # 再次对同一组子索引进行交集操作
            the_int = rng1.intersection(rng2)
            tm.assert_index_equal(the_int, expected)

            # 情况：不重叠的情况
            the_int = rng[:10].intersection(rng[10:])
            expected = DatetimeIndex([]).as_unit("ns")  # 预期结果为空的 DatetimeIndex
            tm.assert_index_equal(the_int, expected)

        # 测试 intersection 方法，验证 bug 修复
        def test_intersection_bug(self):
            # GH #771 bug 的测试案例
            a = bdate_range("11/30/2011", "12/31/2011")  # 创建一个包含 2011 年 11 月到 12 月的工作日索引 a
            b = bdate_range("12/10/2011", "12/20/2011")  # 创建一个包含 2011 年 12 月 10 日到 20 日的工作日索引 b
            result = a.intersection(b)  # 对索引 a 和 b 进行交集操作
            tm.assert_index_equal(result, b)  # 断言交集结果与索引 b 相同
            assert result.freq == b.freq  # 断言交集结果的频率与索引 b 相同

        # 测试 intersection 方法，验证列表输入的情况
        def test_intersection_list(self):
            # GH#35876 情况的测试案例
            # values 是一个时间戳列表，创建一个名称为 "a" 的日期时间索引 idx
            values = [Timestamp("2020-01-01"), Timestamp("2020-02-01")]
            idx = DatetimeIndex(values, name="a")
            res = idx.intersection(values)  # 对 idx 和 values 进行交集操作
            tm.assert_index_equal(res, idx)  # 断言交集结果与 idx 相同
    # 测试函数：使用 pytz 库进行时区处理的月份范围联合测试
    def test_month_range_union_tz_pytz(self, sort):
        # 导入并检查 pytz 库是否可用
        pytz = pytest.importorskip("pytz")
        # 设置时区为 "US/Eastern"
        tz = pytz.timezone("US/Eastern")

        # 设置早期和晚期日期范围
        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        # 创建早期和晚期的日期范围对象，频率为月末，使用指定时区
        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        # 对两个日期范围对象进行联合操作，可以选择是否排序
        early_dr.union(late_dr, sort=sort)

    # 标记为跳过 Windows 平台的测试函数：使用 dateutil 库进行时区处理的月份范围联合测试
    @td.skip_if_windows
    def test_month_range_union_tz_dateutil(self, sort):
        # 导入 dateutil_gettz 函数
        from pandas._libs.tslibs.timezones import dateutil_gettz

        # 设置时区为 "US/Eastern"
        tz = dateutil_gettz("US/Eastern")

        # 设置早期和晚期日期范围
        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        # 创建早期和晚期的日期范围对象，频率为月末，使用指定时区
        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        # 对两个日期范围对象进行联合操作，可以选择是否排序
        early_dr.union(late_dr, sort=sort)

    # 参数化测试函数：测试索引对象的重复元素交集处理
    @pytest.mark.parametrize("sort", [False, None])
    def test_intersection_duplicates(self, sort):
        # GH#38196，创建具有重复时间戳的索引对象
        idx1 = Index(
            [
                Timestamp("2019-12-13"),
                Timestamp("2019-12-12"),
                Timestamp("2019-12-12"),
            ]
        )
        # 执行索引对象的交集操作，可以选择是否排序
        result = idx1.intersection(idx1, sort=sort)
        # 期望的交集结果
        expected = Index([Timestamp("2019-12-13"), Timestamp("2019-12-12")])
        # 断言实际结果与期望结果相等
        tm.assert_index_equal(result, expected)
class TestCustomDatetimeIndex:
    # 测试 datetime 索引对象的 union 方法

    def test_union(self, sort):
        # 测试 union 方法对重叠部分的处理
        rng = bdate_range(START, END, freq="C")
        # 创建两个重叠的时间范围对象
        left = rng[:10]
        right = rng[5:10]

        # 执行 union 操作，将两个时间范围对象合并
        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # 测试 union 方法对非重叠部分的处理（中间有间隙）
        left = rng[:5]
        right = rng[10:]

        # 执行 union 操作，将两个时间范围对象合并
        the_union = left.union(right, sort)
        assert isinstance(the_union, Index)

        # 测试 union 方法对非重叠部分的处理（无间隙）
        left = rng[:5]
        right = rng[5:10]

        # 执行 union 操作，将两个时间范围对象合并
        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # 测试 union 方法在无序的情况下的行为
        if sort is None:
            # 验证无序情况下 union 操作的结果是否正确
            tm.assert_index_equal(right.union(left, sort=sort), the_union)

        # 测试重叠但偏移不同的时间范围对象的 union
        rng = date_range(START, END, freq=BMonthEnd())

        # 执行 union 操作，将同一个时间范围对象合并
        the_union = rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_intersection_bug(self):
        # 测试 intersection 方法修复问题（GH #771）
        a = bdate_range("11/30/2011", "12/31/2011", freq="C")
        b = bdate_range("12/10/2011", "12/20/2011", freq="C")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    @pytest.mark.parametrize(
        "tz", [None, "UTC", "Europe/Berlin", timezone(timedelta(hours=-1))]
    )
    def test_intersection_dst_transition(self, tz):
        # 测试 intersection 方法在夏令时转换时的行为（GH 46702）
        idx1 = date_range("2020-03-27", periods=5, freq="D", tz=tz)
        idx2 = date_range("2020-03-30", periods=5, freq="D", tz=tz)
        result = idx1.intersection(idx2)
        expected = date_range("2020-03-30", periods=2, freq="D", tz=tz)
        tm.assert_index_equal(result, expected)

        # 测试 union 方法在夏令时转换时的行为（GH#45863）
        index1 = date_range("2021-10-28", periods=3, freq="D", tz="Europe/London")
        index2 = date_range("2021-10-30", periods=4, freq="D", tz="Europe/London")
        result = index1.union(index2)
        expected = date_range("2021-10-28", periods=6, freq="D", tz="Europe/London")
        tm.assert_index_equal(result, expected)


def test_union_non_nano_rangelike():
    # 测试非纳秒级的日期时间索引对象的 union 方法（GH 59036）
    l1 = DatetimeIndex(
        ["2024-05-11", "2024-05-12"], dtype="datetime64[us]", name="Date", freq="D"
    )
    l2 = DatetimeIndex(["2024-05-13"], dtype="datetime64[us]", name="Date", freq="D")
    result = l1.union(l2)
    expected = DatetimeIndex(
        ["2024-05-11", "2024-05-12", "2024-05-13"],
        dtype="datetime64[us]",
        name="Date",
        freq="D",
    )
    tm.assert_index_equal(result, expected)
```