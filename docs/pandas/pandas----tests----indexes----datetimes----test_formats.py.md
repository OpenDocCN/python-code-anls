# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_formats.py`

```
# 导入所需模块和库
from datetime import (
    datetime,  # 导入 datetime 对象
    timezone,  # 导入 timezone 对象
)

import dateutil.tz  # 导入 dateutil.tz 模块
import numpy as np  # 导入 numpy 库
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库
from pandas import (
    DatetimeIndex,  # 导入 DatetimeIndex 类
    NaT,  # 导入 NaT（Not a Time）对象
    Series,  # 导入 Series 类
)
import pandas._testing as tm  # 导入 pandas 测试工具模块

# 定义测试函数 test_get_values_for_csv
def test_get_values_for_csv():
    # 创建一个日期范围索引对象 index，频率为每天一次，从指定日期开始，共3天
    index = pd.date_range(freq="1D", periods=3, start="2017-01-01")

    # 第一次调用 _get_values_for_csv 方法，不传入任何参数，期望结果为日期字符串的 numpy 数组
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv()
    tm.assert_numpy_array_equal(result, expected)

    # 再次调用 _get_values_for_csv 方法，传入 na_rep 参数为 "pandas"，验证没有 NaN 值时 na_rep 的效果
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # 验证日期格式化参数 date_format 的影响
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype=object)
    result = index._get_values_for_csv(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # 验证处理含有 NaN 值的索引对象情况
    index = DatetimeIndex(["2017-01-01", NaT, "2017-01-03"])
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv(na_rep="NaT")
    tm.assert_numpy_array_equal(result, expected)

    # 再次验证处理含有 NaN 值的索引对象情况，这次使用 na_rep 参数为 "pandas"
    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # 最后一次验证处理含有 NaN 值的索引对象情况，使用 na_rep 参数为 "NaT" 和自定义 date_format
    expected = np.array(
        ["2017-01-01 00:00:00.000000", "NaT", "2017-01-03 00:00:00.000000"],
        dtype=object,
    )
    result = index._get_values_for_csv(na_rep="NaT", date_format="%Y-%m-%d %H:%M:%S.%f")
    tm.assert_numpy_array_equal(result, expected)

    # 验证无效的 date_format 参数处理情况
    result = index._get_values_for_csv(na_rep="NaT", date_format="foo")
    expected = np.array(["foo", "NaT", "foo"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


# 定义日期索引呈现测试类 TestDatetimeIndexRendering
class TestDatetimeIndexRendering:
    # 参数化测试方法 test_dti_with_timezone_repr，测试不同时区字符串 tzstr
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_with_timezone_repr(self, tzstr):
        # 创建一个日期范围对象 rng，从指定日期范围内生成日期
        rng = pd.date_range("4/13/2010", "5/6/2010")

        # 将日期范围对象 rng 转换为指定时区的日期范围对象 rng_eastern
        rng_eastern = rng.tz_localize(tzstr)

        # 获取 rng_eastern 对象的字符串表示形式 rng_repr，并断言包含特定日期字符串
        rng_repr = repr(rng_eastern)
        assert "2010-04-13 00:00:00" in rng_repr

    # 测试方法 test_dti_repr_dates，验证日期对象的字符串表示形式包含特定日期字符串
    def test_dti_repr_dates(self):
        text = str(pd.to_datetime([datetime(2013, 1, 1), datetime(2014, 1, 1)]))
        assert "['2013-01-01'," in text
        assert ", '2014-01-01']" in text

    # 测试方法 test_dti_repr_mixed，验证混合日期对象的字符串表示形式包含特定日期字符串
    def test_dti_repr_mixed(self):
        text = str(
            pd.to_datetime(
                [datetime(2013, 1, 1), datetime(2014, 1, 1, 12), datetime(2014, 1, 1)]
            )
        )
        assert "'2013-01-01 00:00:00'," in text
        assert "'2014-01-01 00:00:00']" in text

    # 测试方法 test_dti_repr_short，验证日期范围对象的字符串表示形式
    def test_dti_repr_short(self):
        dr = pd.date_range(start="1/1/2012", periods=1)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=2)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=3)
        repr(dr)
    @pytest.mark.parametrize(
        # 参数化测试，提供多组参数进行测试
        "dates, freq, expected_repr",
        [
            (
                ["2012-01-01 00:00:00"],
                "60min",
                (
                    "DatetimeIndex(['2012-01-01 00:00:00'], "
                    "dtype='datetime64[ns]', freq='60min')"
                ),
            ),
            (
                ["2012-01-01 00:00:00", "2012-01-01 01:00:00"],
                "60min",
                "DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 01:00:00'], "
                "dtype='datetime64[ns]', freq='60min')",
            ),
            (
                ["2012-01-01"],
                "24h",
                "DatetimeIndex(['2012-01-01'], dtype='datetime64[ns]', freq='24h')",
            ),
        ],
    )
    # 定义测试方法，测试DatetimeIndex的字符串表示
    def test_dti_repr_time_midnight(self, dates, freq, expected_repr, unit):
        # GH53634
        # 创建DatetimeIndex对象，并设置时间单位
        dti = DatetimeIndex(dates, freq).as_unit(unit)
        # 获取对象的字符串表示形式
        actual_repr = repr(dti)
        # 断言实际的字符串表示形式与期望的相符，替换时间单位的格式
        assert actual_repr == expected_repr.replace("[ns]", f"[{unit}]")
    # 定义一个测试函数，用于测试 DatetimeIndex 对象在不同情况下的表现
    def test_dti_representation(self, unit):
        # 创建一个空的 DatetimeIndex 对象列表
        idxs = []
        # 将空的 DatetimeIndex 对象添加到列表中，频率为每日 ("D")
        idxs.append(DatetimeIndex([], freq="D"))
        # 添加一个包含单个日期 "2011-01-01" 的 DatetimeIndex 对象，频率为每日 ("D")
        idxs.append(DatetimeIndex(["2011-01-01"], freq="D"))
        # 添加一个包含两个日期 "2011-01-01" 和 "2011-01-02" 的 DatetimeIndex 对象，频率为每日 ("D")
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D"))
        # 添加一个包含三个日期 "2011-01-01", "2011-01-02", "2011-01-03" 的 DatetimeIndex 对象，频率为每日 ("D")
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D"))
        # 添加一个包含三个带有时区信息的日期时间 "2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00" 的 DatetimeIndex 对象，频率为每小时 ("h")，时区为 "Asia/Tokyo"
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                tz="Asia/Tokyo",
            )
        )
        # 添加一个包含两个日期时间 "2011-01-01 09:00", "2011-01-01 10:00" 和一个 NaT 的 DatetimeIndex 对象，时区为 "US/Eastern"
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
            )
        )
        # 添加一个包含两个日期时间 "2011-01-01 09:00", "2011-01-01 10:00" 和一个 NaT 的 DatetimeIndex 对象，时区为 "UTC"
        idxs.append(
            DatetimeIndex(["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="UTC")
        )

        # 创建一个预期输出列表
        exp = []
        # 将预期输出字符串 "DatetimeIndex([], dtype='datetime64[ns]', freq='D')" 添加到列表中
        exp.append("DatetimeIndex([], dtype='datetime64[ns]', freq='D')")
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01'], dtype='datetime64[ns]', freq='D')" 添加到列表中
        exp.append("DatetimeIndex(['2011-01-01'], dtype='datetime64[ns]', freq='D')")
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01', '2011-01-02'], dtype='datetime64[ns]', freq='D')" 添加到列表中
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], dtype='datetime64[ns]', freq='D')" 添加到列表中
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01 09:00:00+09:00', '2011-01-01 10:00:00+09:00', '2011-01-01 11:00:00+09:00'], dtype='datetime64[ns, Asia/Tokyo]', freq='h')" 添加到列表中
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+09:00', "
            "'2011-01-01 10:00:00+09:00', '2011-01-01 11:00:00+09:00']"
            ", dtype='datetime64[ns, Asia/Tokyo]', freq='h')"
        )
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01 09:00:00-05:00', '2011-01-01 10:00:00-05:00', 'NaT'], dtype='datetime64[ns, US/Eastern]', freq=None)" 添加到列表中
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00-05:00', "
            "'2011-01-01 10:00:00-05:00', 'NaT'], "
            "dtype='datetime64[ns, US/Eastern]', freq=None)"
        )
        # 将预期输出字符串 "DatetimeIndex(['2011-01-01 09:00:00+00:00', '2011-01-01 10:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]', freq=None)" 添加到列表中
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+00:00', "
            "'2011-01-01 10:00:00+00:00', 'NaT'], "
            "dtype='datetime64[ns, UTC]', freq=None)"
            ""
        )

        # 使用 pd.option_context 设置显示宽度为 300，开始测试循环
        with pd.option_context("display.width", 300):
            # 对于每个 idxs 和 exp 中的对应元素，执行以下操作
            for index, expected in zip(idxs, exp):
                # 将 DatetimeIndex 对象转换为指定的时间单位 (unit)
                index = index.as_unit(unit)
                # 将预期输出中的时间单位 "[ns" 替换为实际的单位字符串
                expected = expected.replace("[ns", f"[{unit}")
                # 获取转换后的 DatetimeIndex 对象的字符串表示形式
                result = repr(index)
                # 断言转换后的字符串与预期输出相同
                assert result == expected
                # 获取转换后的 DatetimeIndex 对象的普通字符串表示形式
                result = str(index)
                # 断言转换后的普通字符串与预期输出相同
                assert result == expected

    # TODO: this is a Series.__repr__ test
    # 定义测试函数，测试日期时间索引对象转换为序列的表示形式
    def test_dti_representation_to_series(self, unit):
        # 创建空的日期时间索引
        idx1 = DatetimeIndex([], freq="D")
        # 创建包含一个日期的日期时间索引
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        # 创建包含两个日期的日期时间索引
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        # 创建包含三个日期的日期时间索引
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        # 创建带有时区和小时频率的日期时间索引
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="h",
            tz="Asia/Tokyo",
        )
        # 创建带有时区的日期时间索引，包含 NaT（不可用时间）值
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
        )
        # 创建不带时区的日期时间索引
        idx7 = DatetimeIndex(["2011-01-01 09:00", "2011-01-02 10:15"])

        # 期望结果：空序列的字符串表示
        exp1 = """Series([], dtype: datetime64[ns])"""

        # 期望结果：包含一个日期的序列的字符串表示
        exp2 = "0   2011-01-01\ndtype: datetime64[ns]"

        # 期望结果：包含两个日期的序列的字符串表示
        exp3 = "0   2011-01-01\n1   2011-01-02\ndtype: datetime64[ns]"

        # 期望结果：包含三个日期的序列的字符串表示
        exp4 = (
            "0   2011-01-01\n"
            "1   2011-01-02\n"
            "2   2011-01-03\n"
            "dtype: datetime64[ns]"
        )

        # 期望结果：带有时区的日期时间序列的字符串表示
        exp5 = (
            "0   2011-01-01 09:00:00+09:00\n"
            "1   2011-01-01 10:00:00+09:00\n"
            "2   2011-01-01 11:00:00+09:00\n"
            "dtype: datetime64[ns, Asia/Tokyo]"
        )

        # 期望结果：带有不同时区和 NaT 的日期时间序列的字符串表示
        exp6 = (
            "0   2011-01-01 09:00:00-05:00\n"
            "1   2011-01-01 10:00:00-05:00\n"
            "2                         NaT\n"
            "dtype: datetime64[ns, US/Eastern]"
        )

        # 期望结果：不带时区的日期时间序列的字符串表示
        exp7 = (
            "0   2011-01-01 09:00:00\n"
            "1   2011-01-02 10:15:00\n"
            "dtype: datetime64[ns]"
        )

        # 设置 Pandas 显示选项，以便输出的宽度不超过 300
        with pd.option_context("display.width", 300):
            # 遍历日期时间索引和期望结果列表
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5, idx6, idx7],
                [exp1, exp2, exp3, exp4, exp5, exp6, exp7],
            ):
                # 将日期时间索引转换为序列，并获取其字符串表示
                ser = Series(idx.as_unit(unit))
                result = repr(ser)
                # 断言结果与期望值相同，用指定单位替换"[ns"
                assert result == expected.replace("[ns", f"[{unit}")
    def test_dti_summary(self):
        # GH#9116
        # 创建空的日期时间索引
        idx1 = DatetimeIndex([], freq="D")
        # 创建包含一个日期的日期时间索引
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        # 创建包含两个日期的日期时间索引
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        # 创建包含三个日期的日期时间索引
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        # 创建带有时区和小时频率的日期时间索引
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="h",
            tz="Asia/Tokyo",
        )
        # 创建带有时区和部分缺失值的日期时间索引
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
        )

        # 预期结果字符串 1
        exp1 = "DatetimeIndex: 0 entries\nFreq: D"

        # 预期结果字符串 2
        exp2 = "DatetimeIndex: 1 entries, 2011-01-01 to 2011-01-01\nFreq: D"

        # 预期结果字符串 3
        exp3 = "DatetimeIndex: 2 entries, 2011-01-01 to 2011-01-02\nFreq: D"

        # 预期结果字符串 4
        exp4 = "DatetimeIndex: 3 entries, 2011-01-01 to 2011-01-03\nFreq: D"

        # 预期结果字符串 5
        exp5 = (
            "DatetimeIndex: 3 entries, 2011-01-01 09:00:00+09:00 "
            "to 2011-01-01 11:00:00+09:00\n"
            "Freq: h"
        )

        # 预期结果字符串 6
        exp6 = """DatetimeIndex: 3 entries, 2011-01-01 09:00:00-05:00 to NaT"""

        # 对于每个日期时间索引和其对应的预期结果字符串，执行以下操作
        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6], [exp1, exp2, exp3, exp4, exp5, exp6]
        ):
            # 调用 _summary 方法，获取结果
            result = idx._summary()
            # 断言结果与预期相符
            assert result == expected

    @pytest.mark.parametrize("tz", [None, timezone.utc, dateutil.tz.tzutc()])
    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_dti_business_repr_etc_smoke(self, tz, freq):
        # 只关心它能否正常工作
        # 创建工作日范围日期时间索引
        dti = pd.bdate_range(
            datetime(2009, 1, 1), datetime(2010, 1, 1), tz=tz, freq=freq
        )
        # 获取日期时间索引的字符串表示形式
        repr(dti)
        # 调用日期时间索引的 _summary 方法
        dti._summary()
        # 调用日期时间索引切片后的 _summary 方法
        dti[2:2]._summary()
```