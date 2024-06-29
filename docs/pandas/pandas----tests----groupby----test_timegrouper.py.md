# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_timegrouper.py`

```
"""
test with the TimeGrouper / grouping with datetimes
"""

# 引入必要的库和模块
from datetime import (
    datetime,       # 日期时间处理模块
    timedelta,      # 时间增量模块
    timezone,       # 时区模块
)

import numpy as np     # 数值计算库
import pytest           # 测试框架

import pandas as pd                    # 数据处理库
from pandas import (                   # 从 pandas 中导入多个子模块和类
    DataFrame,                          # 数据帧类
    DatetimeIndex,                      # 日期时间索引类
    Index,                              # 索引类
    MultiIndex,                         # 多级索引类
    Series,                             # 序列类
    Timestamp,                          # 时间戳类
    date_range,                         # 日期范围生成函数
    offsets,                            # 偏移量类
)
import pandas._testing as tm            # 测试相关的私有模块
from pandas.core.groupby.grouper import Grouper       # 分组器类
from pandas.core.groupby.ops import BinGrouper         # 分组操作类


@pytest.fixture
def frame_for_truncated_bingrouper():
    """
    DataFrame used by groupby_with_truncated_bingrouper, made into
    a separate fixture for easier reuse in
    test_groupby_apply_timegrouper_with_nat_apply_squeeze
    """
    # 创建一个数据帧，用于测试分组器截断的情况
    df = DataFrame(
        {
            "Quantity": [18, 3, 5, 1, 9, 3],
            "Date": [
                Timestamp(2013, 9, 1, 13, 0),
                Timestamp(2013, 9, 1, 13, 5),
                Timestamp(2013, 10, 1, 20, 0),
                Timestamp(2013, 10, 3, 10, 0),
                pd.NaT,
                Timestamp(2013, 9, 2, 14, 0),
            ],
        }
    )
    return df


@pytest.fixture
def groupby_with_truncated_bingrouper(frame_for_truncated_bingrouper):
    """
    GroupBy object such that gb._grouper is a BinGrouper and
    len(gb._grouper.result_index) < len(gb._grouper.group_keys_seq)

    Aggregations on this groupby should have

        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date")

    As either the index or an index level.
    """
    # 使用测试数据帧创建一个分组对象，确保 gb._grouper 是 BinGrouper 类型，
    # 并且 gb._grouper.result_index 的长度小于 gb._grouper.group_keys_seq 的长度
    df = frame_for_truncated_bingrouper

    tdg = Grouper(key="Date", freq="5D")     # 创建时间分组器，按照每5天分组
    gb = df.groupby(tdg)

    # 检查我们正在测试的特定情况
    assert len(gb._grouper.result_index) != len(gb._grouper.codes)

    return gb


class TestGroupBy:
    def test_groupby_with_timegrouper(self):
        # GH 4161
        # TimeGrouper 需要一个排序后的索引
        # 同时验证结果索引是否具有正确的名称
        df_original = DataFrame(
            {
                "Buyer": "Carl Carl Carl Carl Joe Carl".split(),
                "Quantity": [18, 3, 5, 1, 9, 3],
                "Date": [
                    datetime(2013, 9, 1, 13, 0),
                    datetime(2013, 9, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 3, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 9, 2, 14, 0),
                ],
            }
        )

        # GH 6908 改变目标列的顺序
        df_reordered = df_original.sort_values(by="Quantity")

        for df in [df_original, df_reordered]:
            df = df.set_index(["Date"])

            exp_dti = date_range(
                "20130901",
                "20131205",
                freq="5D",
                name="Date",
                inclusive="left",
                unit=df.index.unit,
            )
            expected = DataFrame(
                {"Buyer": 0, "Quantity": 0},
                index=exp_dti,
            )
            # 将 "Buyer" 列强制转换为对象，避免在将条目设置为 "CarlCarlCarl" 时的隐式转换
            expected = expected.astype({"Buyer": object})
            expected.iloc[0, 0] = "CarlCarlCarl"
            expected.iloc[6, 0] = "CarlCarl"
            expected.iloc[18, 0] = "Joe"
            expected.iloc[[0, 6, 18], 1] = np.array([24, 6, 9], dtype="int64")

            result1 = df.resample("5D").sum()
            tm.assert_frame_equal(result1, expected)

            # 按照索引排序后进行分组并求和
            df_sorted = df.sort_index()
            result2 = df_sorted.groupby(Grouper(freq="5D")).sum()
            tm.assert_frame_equal(result2, expected)

            # 按照原始顺序进行分组并求和
            result3 = df.groupby(Grouper(freq="5D")).sum()
            tm.assert_frame_equal(result3, expected)
    def test_groupby_with_timegrouper_methods(self, should_sort):
        # GH 3881
        # 确保 timegrouper 的 API 符合预期

        # 创建包含分组数据的 DataFrame
        df = DataFrame(
            {
                "Branch": "A A A A A B".split(),
                "Buyer": "Carl Mark Carl Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 8, 9, 3],
                "Date": [
                    datetime(2013, 1, 1, 13, 0),
                    datetime(2013, 1, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 12, 2, 14, 0),
                ],
            }
        )

        # 如果需要排序，则按 Quantity 列降序排序 DataFrame
        if should_sort:
            df = df.sort_values(by="Quantity", ascending=False)

        # 将 Date 列设置为索引，保留原始索引列
        df = df.set_index("Date", drop=False)
        
        # 使用频率为 "6ME" 进行分组
        g = df.groupby(Grouper(freq="6ME"))
        
        # 断言分组键存在
        assert g.group_keys

        # 断言内部的 _grouper 对象为 BinGrouper 类型
        assert isinstance(g._grouper, BinGrouper)

        # 获取分组后的结果字典
        groups = g.groups
        
        # 断言 groups 是一个字典且长度为 3
        assert isinstance(groups, dict)
        assert len(groups) == 3

    @pytest.mark.parametrize("freq", ["D", "ME", "YE", "QE-APR"])
    def test_timegrouper_with_reg_groups_freq(self, freq):
        # GH 6764 多个频率下的分组测试，包括排序和不排序

        # 创建包含时间序列数据的 DataFrame
        df = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "20121002",
                        "20121007",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20121002",
                        "20121207",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20130202",
                        "20130305",
                    ]
                ),
                "user_id": [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5],
                "whole_cost": [
                    1790,
                    364,
                    280,
                    259,
                    201,
                    623,
                    90,
                    312,
                    359,
                    301,
                    359,
                    801,
                ],
                "cost1": [12, 15, 10, 24, 39, 1, 0, 90, 45, 34, 1, 12],
            }
        ).set_index("date")

        # 期望的分组汇总结果
        expected = (
            df.groupby("user_id")["whole_cost"]
            .resample(freq)
            .sum(min_count=1)  # XXX
            .dropna()
            .reorder_levels(["date", "user_id"])
            .sort_index()
            .astype("int64")
        )
        expected.name = "whole_cost"

        # 检查排序后的结果
        result1 = (
            df.sort_index().groupby([Grouper(freq=freq), "user_id"])["whole_cost"].sum()
        )
        tm.assert_series_equal(result1, expected)

        # 检查未排序的结果
        result2 = df.groupby([Grouper(freq=freq), "user_id"])["whole_cost"].sum()
        tm.assert_series_equal(result2, expected)
    def test_timegrouper_get_group(self):
        # 定义测试方法：验证时间分组功能是否正常工作
        # GH 6914

        # 创建原始数据框
        df_original = DataFrame(
            {
                "Buyer": "Carl Joe Joe Carl Joe Carl".split(),
                "Quantity": [18, 3, 5, 1, 9, 3],
                "Date": [
                    datetime(2013, 9, 1, 13, 0),
                    datetime(2013, 9, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 3, 10, 0),
                    datetime(2013, 12, 2, 12, 0),
                    datetime(2013, 9, 2, 14, 0),
                ],
            }
        )
        # 根据 Quantity 列对原始数据框进行排序
        df_reordered = df_original.sort_values(by="Quantity")

        # 单一分组测试
        expected_list = [
            df_original.iloc[[0, 1, 5]],  # 预期结果1
            df_original.iloc[[2, 3]],     # 预期结果2
            df_original.iloc[[4]],        # 预期结果3
        ]
        # 时间戳列表
        dt_list = ["2013-09-30", "2013-10-31", "2013-12-31"]

        # 对原始数据框和重新排序后的数据框进行测试
        for df in [df_original, df_reordered]:
            # 根据日期列使用月末频率进行分组
            grouped = df.groupby(Grouper(freq="M", key="Date"))
            # 遍历时间戳列表和预期结果列表
            for t, expected in zip(dt_list, expected_list):
                dt = Timestamp(t)
                # 获取分组后的数据框
                result = grouped.get_group(dt)
                # 断言分组结果与预期结果是否相同
                tm.assert_frame_equal(result, expected)

        # 多重分组测试
        expected_list = [
            df_original.iloc[[1]],  # 预期结果1
            df_original.iloc[[3]],  # 预期结果2
            df_original.iloc[[4]],  # 预期结果3
        ]
        # 分组列表：元组包含买家名称和时间戳
        g_list = [("Joe", "2013-09-30"), ("Carl", "2013-10-31"), ("Joe", "2013-12-31")]

        # 对原始数据框和重新排序后的数据框进行测试
        for df in [df_original, df_reordered]:
            # 根据买家名称和日期列使用月末频率进行分组
            grouped = df.groupby(["Buyer", Grouper(freq="M", key="Date")])
            # 遍历分组列表和预期结果列表
            for (b, t), expected in zip(g_list, expected_list):
                dt = Timestamp(t)
                # 获取分组后的数据框
                result = grouped.get_group((b, dt))
                # 断言分组结果与预期结果是否相同
                tm.assert_frame_equal(result, expected)

        # 带有索引的测试
        # 将原始数据框的日期列设置为索引
        df_original = df_original.set_index("Date")
        # 根据 Quantity 列对重新排序后的数据框进行排序
        df_reordered = df_original.sort_values(by="Quantity")

        # 单一分组测试
        expected_list = [
            df_original.iloc[[0, 1, 5]],  # 预期结果1
            df_original.iloc[[2, 3]],     # 预期结果2
            df_original.iloc[[4]],        # 预期结果3
        ]

        # 对原始数据框和重新排序后的数据框进行测试
        for df in [df_original, df_reordered]:
            # 根据索引日期使用月末频率进行分组
            grouped = df.groupby(Grouper(freq="M"))
            # 遍历时间戳列表和预期结果列表
            for t, expected in zip(dt_list, expected_list):
                dt = Timestamp(t)
                # 获取分组后的数据框
                result = grouped.get_group(dt)
                # 断言分组结果与预期结果是否相同
                tm.assert_frame_equal(result, expected)
    # 测试函数：test_timegrouper_apply_return_type_series
    def test_timegrouper_apply_return_type_series(self):
        # 使用 `apply` 结合 `TimeGrouper` 应该与使用 `apply` 结合 `Grouper` 产生相同的返回类型。
        # Issue #11742

        # 创建一个 DataFrame 包含日期和数值列
        df = DataFrame({"date": ["10/10/2000", "11/10/2000"], "value": [10, 13]})
        df_dt = df.copy()
        # 将日期列转换为 datetime 类型
        df_dt["date"] = pd.to_datetime(df_dt["date"])

        # 定义一个用于返回 Series 的函数
        def sumfunc_series(x):
            return Series([x["value"].sum()], ("sum",))

        # 检查警告消息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 使用 Grouper 对象对日期进行分组，并应用 sumfunc_series 函数
            expected = df.groupby(Grouper(key="date")).apply(sumfunc_series)

        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 使用 TimeGrouper 对日期进行分组（每月末），并应用 sumfunc_series 函数
            result = df_dt.groupby(Grouper(freq="ME", key="date")).apply(sumfunc_series)

        # 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    # 测试函数：test_timegrouper_apply_return_type_value
    def test_timegrouper_apply_return_type_value(self):
        # 使用 `apply` 结合 `TimeGrouper` 应该与使用 `apply` 结合 `Grouper` 产生相同的返回类型。
        # Issue #11742

        # 创建一个 DataFrame 包含日期和数值列
        df = DataFrame({"date": ["10/10/2000", "11/10/2000"], "value": [10, 13]})
        df_dt = df.copy()
        # 将日期列转换为 datetime 类型
        df_dt["date"] = pd.to_datetime(df_dt["date"])

        # 定义一个用于返回数值总和的函数
        def sumfunc_value(x):
            return x.value.sum()

        # 检查警告消息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 使用 Grouper 对象对日期进行分组，并应用 sumfunc_value 函数
            expected = df.groupby(Grouper(key="date")).apply(sumfunc_value)

        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 使用 TimeGrouper 对日期进行分组（每月末），并应用 sumfunc_value 函数
            result = df_dt.groupby(Grouper(freq="ME", key="date")).apply(sumfunc_value)

        # 检查两个 Series 是否相等
        tm.assert_series_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    # 测试函数：test_groupby_groups_datetimeindex
    def test_groupby_groups_datetimeindex(self):
        # GH#1430

        # 创建一个包含时间索引的 DataFrame
        periods = 1000
        ind = date_range(start="2012/1/1", freq="5min", periods=periods)
        df = DataFrame(
            {"high": np.arange(periods), "low": np.arange(periods)}, index=ind
        )

        # 根据日期创建一个分组对象
        grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))

        # 获取分组后的组信息
        groups = grouped.groups

        # 断言第一个组的类型为 datetime
        assert isinstance(next(iter(groups.keys())), datetime)
    def test_groupby_groups_datetimeindex2(self):
        # 测试用例名称：test_groupby_groups_datetimeindex2
        # 测试GH issue #11442

        # 创建一个时间索引，从"2015/01/01"开始，持续5天，索引名为"date"
        index = date_range("2015/01/01", periods=5, name="date")

        # 创建一个数据框，包含两列"A"和"B"，索引为上面创建的时间索引
        df = DataFrame({"A": [5, 6, 7, 8, 9], "B": [1, 2, 3, 4, 5]}, index=index)

        # 对数据框按照"date"索引级别进行分组，并返回分组的组信息
        result = df.groupby(level="date").groups

        # 预期的日期列表
        dates = ["2015-01-05", "2015-01-04", "2015-01-03", "2015-01-02", "2015-01-01"]

        # 生成预期的结果字典，将日期转换为Timestamp对象，并设置为DatetimeIndex类型
        expected = {
            Timestamp(date): DatetimeIndex([date], name="date") for date in dates
        }

        # 使用测试框架检查实际结果和预期结果是否一致
        tm.assert_dict_equal(result, expected)

        # 对数据框按"date"索引级别进行分组
        grouped = df.groupby(level="date")

        # 遍历日期列表，对每个日期获取分组后的数据
        for date in dates:
            result = grouped.get_group(date)

            # 根据日期获取预期的数据，创建一个数据框
            data = [[df.loc[date, "A"], df.loc[date, "B"]]]
            expected_index = DatetimeIndex(
                [date], name="date", freq="D", dtype=index.dtype
            )
            expected = DataFrame(data, columns=list("AB"), index=expected_index)

            # 使用测试框架检查实际结果和预期结果是否一致
            tm.assert_frame_equal(result, expected)

    def test_groupby_groups_datetimeindex_tz(self):
        # 测试用例名称：test_groupby_groups_datetimeindex_tz
        # 测试GH issue #3950

        # 定义日期时间列表
        dates = [
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
        ]

        # 创建数据框，包含"label"、"datetime"、"value1"和"value2"列
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "datetime": dates,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        # 将"datetime"列的值转换为Timestamp对象，并设置时区为"US/Pacific"
        df["datetime"] = df["datetime"].apply(lambda d: Timestamp(d, tz="US/Pacific"))

        # 预期的索引1，包含日期时间和时区信息
        exp_idx1 = DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 09:00:00",
            ],
            tz="US/Pacific",
            name="datetime",
        )

        # 预期的索引2，包含"label"列的值
        exp_idx2 = Index(["a", "b"] * 3, name="label")

        # 组合预期的多级索引
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])

        # 生成预期的数据框，包含"value1"和"value2"列
        expected = DataFrame(
            {"value1": [0, 3, 1, 4, 2, 5], "value2": [1, 2, 2, 1, 1, 2]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        # 对数据框按照["datetime", "label"]列进行分组，并求和
        result = df.groupby(["datetime", "label"]).sum()

        # 使用测试框架检查实际结果和预期结果是否一致
        tm.assert_frame_equal(result, expected)

        # 按照索引级别0进行分组
        didx = DatetimeIndex(dates, tz="Asia/Tokyo")
        df = DataFrame(
            {"value1": np.arange(6, dtype="int64"), "value2": [1, 2, 3, 1, 2, 3]},
            index=didx,
        )

        # 预期的索引，包含日期时间和时区信息
        exp_idx = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            tz="Asia/Tokyo",
        )

        # 生成预期的数据框，包含"value1"和"value2"列
        expected = DataFrame(
            {"value1": [3, 5, 7], "value2": [2, 4, 6]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        # 对数据框按照索引级别0进行分组，并求和
        result = df.groupby(level=0).sum()

        # 使用测试框架检查实际结果和预期结果是否一致
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试处理 datetime64 数据的分组操作
    def test_frame_datetime64_handling_groupby(self):
        # 创建一个 DataFrame 对象，包含两列数据：a 和 date，其中 date 列使用 np.datetime64 表示日期
        df = DataFrame(
            [(3, np.datetime64("2012-07-03")), (3, np.datetime64("2012-07-04"))],
            columns=["a", "date"],
        )
        # 对 DataFrame 进行按列 'a' 分组，并取每组的第一个值
        result = df.groupby("a").first()
        # 断言结果 DataFrame 中索引为 3 的行的 date 列值为 Timestamp("2012-07-03")
        assert result["date"][3] == Timestamp("2012-07-03")

    # 定义另一个测试方法，测试多时区的分组操作
    def test_groupby_multi_timezone(self):
        # 创建一个 DataFrame 对象，包含三列数据：value、date 和 tz，其中 date 是日期字符串，tz 是时区字符串
        df = DataFrame(
            {
                "value": range(5),
                "date": [
                    "2000-01-28 16:47:00",
                    "2000-01-29 16:48:00",
                    "2000-01-30 16:49:00",
                    "2000-01-31 16:50:00",
                    "2000-01-01 16:50:00",
                ],
                "tz": [
                    "America/Chicago",
                    "America/Chicago",
                    "America/Los_Angeles",
                    "America/Chicago",
                    "America/New_York",
                ],
            }
        )

        # 对 DataFrame 按列 'tz' 进行分组，不生成组键，然后对每组的 date 列应用 lambda 函数进行处理
        result = df.groupby("tz", group_keys=False).date.apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(x.name)
        )

        # 预期结果是一个 Series，包含处理后的日期时间对象，带有相应的时区信息
        expected = Series(
            [
                Timestamp("2000-01-28 16:47:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-29 16:48:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-30 16:49:00-0800", tz="America/Los_Angeles"),
                Timestamp("2000-01-31 16:50:00-0600", tz="America/Chicago"),
                Timestamp("2000-01-01 16:50:00-0500", tz="America/New_York"),
            ],
            name="date",
            dtype=object,
        )
        # 断言 result 和 expected Series 相等
        tm.assert_series_equal(result, expected)

        # 选择特定时区 'America/Chicago' 下的日期时间数据，然后将其转换为本地化的日期时间对象
        tz = "America/Chicago"
        res_values = df.groupby("tz").date.get_group(tz)
        result = pd.to_datetime(res_values).dt.tz_localize(tz)
        # 创建预期的 Series，包含指定索引和本地化后的日期时间对象
        exp_values = Series(
            ["2000-01-28 16:47:00", "2000-01-29 16:48:00", "2000-01-31 16:50:00"],
            index=[0, 1, 3],
            name="date",
        )
        expected = pd.to_datetime(exp_values).dt.tz_localize(tz)
        # 断言 result 和 expected Series 相等
        tm.assert_series_equal(result, expected)
    # 定义测试方法，测试按照时间周期和标签进行分组
    def test_groupby_groups_periods(self):
        # 定义日期列表
        dates = [
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
            "2011-07-19 07:00:00",
            "2011-07-19 08:00:00",
            "2011-07-19 09:00:00",
        ]
        # 创建 DataFrame 对象，包括'label'、'period'、'value1'、'value2'列
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "period": [pd.Period(d, freq="h") for d in dates],
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        # 预期的时间周期索引
        exp_idx1 = pd.PeriodIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 09:00:00",
            ],
            freq="h",
            name="period",
        )
        # 预期的标签索引
        exp_idx2 = Index(["a", "b"] * 3, name="label")
        # 创建多重索引对象
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        # 创建预期的 DataFrame 对象
        expected = DataFrame(
            {"value1": [0, 3, 1, 4, 2, 5], "value2": [1, 2, 2, 1, 1, 2]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        # 对 DataFrame 按照['period', 'label']进行分组求和
        result = df.groupby(["period", "label"]).sum()
        # 使用测试工具比较结果与预期是否一致
        tm.assert_frame_equal(result, expected)

        # 按照时间周期索引进行分组
        didx = pd.PeriodIndex(dates, freq="h")
        # 创建新的 DataFrame 对象，只包括'value1'和'value2'列
        df = DataFrame(
            {"value1": np.arange(6, dtype="int64"), "value2": [1, 2, 3, 1, 2, 3]},
            index=didx,
        )

        # 创建预期的时间周期索引
        exp_idx = pd.PeriodIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            freq="h",
        )
        # 创建预期的 DataFrame 对象
        expected = DataFrame(
            {"value1": [3, 5, 7], "value2": [2, 4, 6]},
            index=exp_idx,
            columns=["value1", "value2"],
        )

        # 对 DataFrame 按照第一级索引（时间周期）进行分组求和
        result = df.groupby(level=0).sum()
        # 使用测试工具比较结果与预期是否一致
        tm.assert_frame_equal(result, expected)

    # 测试按照第一个 datetime64 数据进行分组
    def test_groupby_first_datetime64(self):
        # 创建包含 datetime64 数据的 DataFrame 对象
        df = DataFrame([(1, 1351036800000000000), (2, 1351036800000000000)])
        # 将第二列数据转换为 datetime64 类型
        df[1] = df[1].astype("M8[ns]")

        # 断言第二列的数据类型是 np.datetime64
        assert issubclass(df[1].dtype.type, np.datetime64)

        # 按照第一级索引进行分组，获取每组的第一个元素
        result = df.groupby(level=0).first()
        # 获取结果的 datetime64 数据类型
        got_dt = result[1].dtype
        # 断言结果的数据类型是 np.datetime64
        assert issubclass(got_dt.type, np.datetime64)

        # 对第二列按照第一级索引进行分组，获取每组的第一个元素
        result = df[1].groupby(level=0).first()
        # 获取结果的 datetime64 数据类型
        got_dt = result.dtype
        # 断言结果的数据类型是 np.datetime64
        assert issubclass(got_dt.type, np.datetime64)

    # 测试按照最大的 datetime64 数据进行分组
    def test_groupby_max_datetime64(self):
        # 创建包含 Timestamp 和整数列的 DataFrame 对象
        df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
        # 对'A'列进行分组，并将每组的最大值转换为秒级精度的 datetime64 类型
        expected = df.groupby("A")["A"].apply(lambda x: x.max()).astype("M8[s]")
        # 对'A'列进行分组，并获取每组的最大值
        result = df.groupby("A")["A"].max()
        # 使用测试工具比较结果与预期是否一致
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试在 datetime64 32 位下的分组操作
    def test_groupby_datetime64_32_bit(self):
        # GH 6410 / numpy 4328
        # 32-bit under 1.9-dev indexing issue
        # 创建一个包含两列的 DataFrame，其中一列为整数序列，另一列为相同的时间戳 "2000-01-1"
        df = DataFrame({"A": range(2), "B": [Timestamp("2000-01-1")] * 2})
        # 对 DataFrame 进行按列 "A" 分组后，对 "B" 列执行 "min" 变换操作
        result = df.groupby("A")["B"].transform("min")
        # 创建一个预期的 Series，包含两个相同的时间戳 "2000-01-1"
        expected = Series([Timestamp("2000-01-1")] * 2, name="B")
        # 断言两个 Series 相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试在时区选择时的分组操作
    def test_groupby_with_timezone_selection(self):
        # GH 11616
        # Test that column selection returns output in correct timezone.
        # 创建一个包含两列的 DataFrame，其中一列是随机生成的整数列，另一列是带有时区信息的日期时间列
        df = DataFrame(
            {
                "factor": np.random.default_rng(2).integers(0, 3, size=60),
                "time": date_range("01/01/2000 00:00", periods=60, freq="s", tz="UTC"),
            }
        )
        # 对 DataFrame 按 "factor" 列进行分组，然后取每组中 "time" 列的最大值，返回一个 Series
        df1 = df.groupby("factor").max()["time"]
        # 对 DataFrame 按 "factor" 列进行分组，然后取每组中 "time" 列的最大值，返回一个 Series
        df2 = df.groupby("factor")["time"].max()
        # 断言两个 Series 相等
        tm.assert_series_equal(df1, df2)

    # 定义一个测试方法，测试广播时丢失时区信息的问题
    def test_timezone_info(self):
        # see gh-11682: Timezone info lost when broadcasting
        # scalar datetime to DataFrame
        # 创建一个包含一列的 DataFrame，列名为 "a"，列值为整数 1，再加上一列带有当前 UTC 时区信息的当前日期时间
        utc = timezone.utc
        df = DataFrame({"a": [1], "b": [datetime.now(utc)]})
        # 断言 DataFrame 中 "b" 列第一行的时区信息与 UTC 时区相等
        assert df["b"][0].tzinfo == utc
        # 创建一个包含两列的 DataFrame，其中一列为整数序列，另一列为带有当前 UTC 时区信息的当前日期时间
        df = DataFrame({"a": [1, 2, 3]})
        # 在 DataFrame 中添加一列 "b"，列值为带有当前 UTC 时区信息的当前日期时间
        df["b"] = datetime.now(utc)
        # 断言 DataFrame 中 "b" 列第一行的时区信息与 UTC 时区相等
        assert df["b"][0].tzinfo == utc

    # 定义一个测试方法，测试在日期时间数据上进行 max、min、first 和 last 操作时 NaT 是否被正确处理
    def test_first_last_max_min_on_time_data(self):
        # GH 10295
        # Verify that NaT is not in the result of max, min, first and last on
        # Dataframe with datetime or timedelta values.
        # 创建一个包含两列的 DataFrame，其中一列是日期时间或 timedelta 值的序列，另一列是带有 NaT 值的日期时间或 timedelta 值的序列
        df_test = DataFrame(
            {
                "dt": [
                    np.nan,
                    "2015-07-24 10:10",
                    "2015-07-25 11:11",
                    "2015-07-23 12:12",
                    np.nan,
                ],
                "td": [
                    np.nan,
                    timedelta(days=1),
                    timedelta(days=2),
                    timedelta(days=3),
                    np.nan,
                ],
            }
        )
        # 将 "dt" 列转换为 datetime 类型，并赋予 DataFrame 一个新的列 "group"，值为 "A"
        df_test.dt = pd.to_datetime(df_test.dt)
        df_test["group"] = "A"
        # 根据 "group" 列分组并得到分组后的 DataFrame
        df_ref = df_test[df_test.dt.notna()]

        # 对原始和参考的分组后的 DataFrame 分别执行 max、min、first 和 last 操作，并断言结果相等
        grouped_test = df_test.groupby("group")
        grouped_ref = df_ref.groupby("group")
        tm.assert_frame_equal(grouped_ref.max(), grouped_test.max())
        tm.assert_frame_equal(grouped_ref.min(), grouped_test.min())
        tm.assert_frame_equal(grouped_ref.first(), grouped_test.first())
        tm.assert_frame_equal(grouped_ref.last(), grouped_test.last())
    def test_nunique_with_timegrouper_and_nat(self):
        # GH 17575
        # 创建一个包含时间和数据的 DataFrame 对象
        test = DataFrame(
            {
                "time": [
                    Timestamp("2016-06-28 09:35:35"),
                    pd.NaT,  # 插入一个 NaT（Not a Time）对象，表示缺失的时间值
                    Timestamp("2016-06-28 16:46:28"),
                ],
                "data": ["1", "2", "3"],  # 数据列
            }
        )

        # 创建一个时间分组器对象，按小时分组
        grouper = Grouper(key="time", freq="h")
        # 对 DataFrame 进行按组计数唯一值操作
        result = test.groupby(grouper)["data"].nunique()
        # 创建预期的 DataFrame，删除时间列中的 NaT，并按组计数唯一值
        expected = test[test.time.notnull()].groupby(grouper)["data"].nunique()
        expected.index = expected.index._with_freq(None)  # 删除索引的频率信息
        # 使用测试工具比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    def test_scalar_call_versus_list_call(self):
        # Issue: 17530
        # 创建一个包含地点、时间和数值的字典
        data_frame = {
            "location": ["shanghai", "beijing", "shanghai"],
            "time": Series(
                ["2017-08-09 13:32:23", "2017-08-11 23:23:15", "2017-08-11 22:23:15"],
                dtype="datetime64[ns]",  # 指定时间列的数据类型
            ),
            "value": [1, 2, 3],  # 数值列
        }
        # 创建 DataFrame，并将时间列设置为索引
        data_frame = DataFrame(data_frame).set_index("time")
        # 创建一个时间分组器对象，按天分组
        grouper = Grouper(freq="D")

        # 根据分组器对象对 DataFrame 进行分组计数
        grouped = data_frame.groupby(grouper)
        result = grouped.count()
        # 再次根据分组器对象对 DataFrame 进行分组计数
        grouped = data_frame.groupby([grouper])
        expected = grouped.count()
        # 使用测试工具比较两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    def test_grouper_period_index(self):
        # GH 32108
        periods = 2
        # 创建一个时间周期范围对象，每个月为频率，包含指定周期和名称
        index = pd.period_range(
            start="2018-01", periods=periods, freq="M", name="Month"
        )
        # 创建一个 Series 对象，索引为时间周期对象，值为周期内的序号
        period_series = Series(range(periods), index=index)
        # 根据索引中的月份对 Series 对象进行分组并求和
        result = period_series.groupby(period_series.index.month).sum()

        # 创建预期的 Series 对象，索引为从 1 到周期数的整数，名称为索引的名称
        expected = Series(
            range(periods), index=Index(range(1, periods + 1), name=index.name)
        )
        # 使用测试工具比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    def test_groupby_apply_timegrouper_with_nat_dict_returns(
        self, groupby_with_truncated_bingrouper
    ):
        # GH#43500 case where gb._grouper.result_index and gb._grouper.group_keys_seq
        #  have different lengths that goes through the `isinstance(values[0], dict)`
        #  path
        gb = groupby_with_truncated_bingrouper

        # 对分组后的 DataFrame 的 "Quantity" 列应用函数，返回一个字典
        res = gb["Quantity"].apply(lambda x: {"foo": len(x)})

        df = gb.obj
        unit = df["Date"]._values.unit
        # 创建一个日期范围对象，从指定日期开始，每隔5天，包含名称和单位信息
        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date", unit=unit)
        # 创建一个 MultiIndex 对象，包含日期范围和固定字符串作为第二级索引
        mi = MultiIndex.from_arrays([dti, ["foo"] * len(dti)])
        # 创建预期的 Series 对象，包含计数结果，索引为 MultiIndex 对象
        expected = Series([3, 0, 0, 0, 0, 0, 2], index=mi, name="Quantity")
        # 使用测试工具比较两个 Series 对象是否相等
        tm.assert_series_equal(res, expected)

    def test_groupby_apply_timegrouper_with_nat_scalar_returns(
        self, groupby_with_truncated_bingrouper
    ):
        # 这个测试函数尚未完成，没有提供完整的代码段，因此不需要添加注释
    ):
        # GH#43500 Previously raised ValueError bc used index with incorrect
        #  length in wrap_applied_result
        # 定义变量 gb 作为 groupby_with_truncated_bingrouper 的别名
        gb = groupby_with_truncated_bingrouper

        # 应用 lambda 函数计算每个组的第一个非 NaN 值作为结果
        res = gb["Quantity"].apply(lambda x: x.iloc[0] if len(x) else np.nan)

        # 从 gb 对象中获取 DataFrame
        df = gb.obj
        # 从 df 的 "Date" 列中获取时间单位
        unit = df["Date"]._values.unit
        # 生成一个日期范围，从 "2013-09-01" 到 "2013-10-01"，每隔 5 天，频率为 unit
        dti = date_range("2013-09-01", "2013-10-01", freq="5D", name="Date", unit=unit)
        # 创建预期的 Series 对象，包含特定索引和值
        expected = Series(
            [18, np.nan, np.nan, np.nan, np.nan, np.nan, 5],
            index=dti._with_freq(None),
            name="Quantity",
        )

        # 断言 res 和 expected 的 Series 对象相等
        tm.assert_series_equal(res, expected)

    def test_groupby_apply_timegrouper_with_nat_apply_squeeze(
        self, frame_for_truncated_bingrouper
    ):
        # 从参数中获取 DataFrame 对象
        df = frame_for_truncated_bingrouper

        # 使用 Grouper 对象按照 "Date" 列分组，频率为 "100YE"
        tdg = Grouper(key="Date", freq="100YE")
        gb = df.groupby(tdg)

        # 断言 gb 的分组数为 1
        assert gb.ngroups == 1
        # 断言 gb 的选定对象索引层级数为 1
        assert gb._selected_obj.index.nlevels == 1

        # 应用 lambda 函数对每个组执行计算，并返回 Series 对象
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            res = gb.apply(lambda x: x["Quantity"] * 2)

        # 创建特定索引的 Index 对象
        dti = Index([Timestamp("2013-12-31")], dtype=df["Date"].dtype, name="Date")
        # 创建预期的 DataFrame 对象，包含特定索引和列
        expected = DataFrame(
            [[36, 6, 6, 10, 2]],
            index=dti,
            columns=Index([0, 1, 5, 2, 3], name="Quantity"),
        )
        # 断言 res 和 expected 的 DataFrame 对象相等
        tm.assert_frame_equal(res, expected)

    @pytest.mark.single_cpu
    def test_groupby_agg_numba_timegrouper_with_nat(
        self, groupby_with_truncated_bingrouper
    ):
        # 导入 numba 库，如果不存在则跳过测试
        pytest.importorskip("numba")

        # 使用 groupby_with_truncated_bingrouper 对象创建 gb 变量作为别名
        gb = groupby_with_truncated_bingrouper

        # 使用 numba 引擎对 gb["Quantity"] 应用 np.nanmean 函数进行聚合计算
        result = gb["Quantity"].aggregate(
            lambda values, index: np.nanmean(values), engine="numba"
        )

        # 使用内置 mean 函数计算 gb["Quantity"] 的期望值
        expected = gb["Quantity"].aggregate("mean")
        # 断言 result 和 expected 的 Series 对象相等
        tm.assert_series_equal(result, expected)

        # 对 gb[["Quantity"]] 应用 numba 引擎的 np.nanmean 函数进行聚合计算
        result_df = gb[["Quantity"]].aggregate(
            lambda values, index: np.nanmean(values), engine="numba"
        )
        # 使用内置 mean 函数计算 gb[["Quantity"]] 的期望值
        expected_df = gb[["Quantity"]].aggregate("mean")
        # 断言 result_df 和 expected_df 的 DataFrame 对象相等
        tm.assert_frame_equal(result_df, expected_df)
```