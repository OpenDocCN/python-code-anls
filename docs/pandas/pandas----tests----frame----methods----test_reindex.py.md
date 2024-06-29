# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_reindex.py`

```
    # 从 datetime 模块导入 datetime 和 timedelta 类
    datetime,
    timedelta,
)

# 导入 inspect 模块，用于检查对象
import inspect

# 导入 numpy 库，并重命名为 np
import numpy as np

# 导入 pytest 库
import pytest

# 从 pandas._libs.tslibs.timezones 模块中导入 dateutil_gettz 函数并重命名为 gettz
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz

# 从 pandas.compat 模块中导入 IS64 和 is_platform_windows 函数
from pandas.compat import (
    IS64,
    is_platform_windows,
)

# 从 pandas.compat.numpy 模块中导入 np_version_gt2 函数
from pandas.compat.numpy import np_version_gt2

# 导入 pandas 库，并重命名为 pd
import pandas as pd

# 从 pandas 模块中导入多个类和函数
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    isna,
)

# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm

# 从 pandas.api.types 模块中导入 CategoricalDtype 类
from pandas.api.types import CategoricalDtype


# 定义一个测试类 TestReindexSetIndex，用于测试 reindex 和 set_index 方法
class TestReindexSetIndex:
    # 测试在 datetimeindex 上同时使用 set_index 和 reindex 方法
    def test_dti_set_index_reindex_datetimeindex(self):
        # 创建一个 DataFrame，其中的数据是使用随机数生成的
        df = DataFrame(np.random.default_rng(2).random(6))
        # 创建一个日期范围，频率为每月末，时区为 US/Eastern
        idx1 = date_range("2011/01/01", periods=6, freq="ME", tz="US/Eastern")
        # 创建另一个日期范围，频率为每年末，时区为 Asia/Tokyo
        idx2 = date_range("2013", periods=6, freq="YE", tz="Asia/Tokyo")

        # 在 DataFrame 上设置索引为 idx1
        df = df.set_index(idx1)
        # 使用测试模块中的方法验证索引是否与预期的 idx1 相同
        tm.assert_index_equal(df.index, idx1)
        # 在 DataFrame 上重新索引为 idx2
        df = df.reindex(idx2)
        # 使用测试模块中的方法验证索引是否与预期的 idx2 相同
        tm.assert_index_equal(df.index, idx2)

    # 测试在带有时区的频率上同时使用 set_index 和 reindex 方法
    def test_dti_set_index_reindex_freq_with_tz(self):
        # 创建一个时间范围，从 2015 年 10 月 1 日到 2015 年 10 月 1 日 23 点，频率为每小时，时区为 US/Eastern
        index = date_range(
            datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq="h", tz="US/Eastern"
        )
        # 创建一个 DataFrame，其中的数据是使用标准正态分布生成的
        df = DataFrame(
            np.random.default_rng(2).standard_normal((24, 1)),
            columns=["a"],
            index=index,
        )
        # 创建一个新的时间范围，从 2015 年 10 月 2 日到 2015 年 10 月 2 日 23 点，频率为每小时，时区为 US/Eastern
        new_index = date_range(
            datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq="h", tz="US/Eastern"
        )

        # 在 DataFrame 上设置索引为 new_index
        result = df.set_index(new_index)
        # 使用断言语句验证新索引的频率与原始索引的频率相同
        assert result.index.freq == index.freq

    # 测试在 IntervalIndex 上同时使用 set_index 和 reset_index 方法
    def test_set_reset_index_intervalindex(self):
        # 创建一个包含列'A'，数值为 0 到 9 的 DataFrame
        df = DataFrame({"A": range(10)})
        # 使用 pd.cut 方法对列'A'进行分段，并赋值给 ser 变量
        ser = pd.cut(df.A, 5)
        # 将分段后的结果赋值给列'B'，并将 DataFrame 按列'B'设置索引
        df["B"] = ser
        df = df.set_index("B")

        # 使用 reset_index 方法重置索引
        df = df.reset_index()

    # 测试在设定索引后进行 setitem 和 reset_index 方法，验证数据类型
    def test_setitem_reset_index_dtypes(self):
        # 创建一个空 DataFrame，列名为 'a', 'b', 'c'，并指定各列数据类型
        df = DataFrame(columns=["a", "b", "c"]).astype(
            {"a": "datetime64[ns]", "b": np.int64, "c": np.float64}
        )
        # 使用 set_index 方法在列 'a' 上设置索引
        df1 = df.set_index(["a"])
        # 向新 DataFrame 中添加空列 'd'
        df1["d"] = []
        # 对结果进行 reset_index 操作
        result = df1.reset_index()
        # 创建预期的 DataFrame，包含列 'a', 'b', 'c', 'd'，且指定各列数据类型
        expected = DataFrame(columns=["a", "b", "c", "d"], index=range(0)).astype(
            {"a": "datetime64[ns]", "b": np.int64, "c": np.float64, "d": np.float64}
        )
        # 使用测试模块中的方法验证结果与预期是否相同
        tm.assert_frame_equal(result, expected)

        # 使用 set_index 方法在列 'a', 'b' 上设置索引
        df2 = df.set_index(["a", "b"])
        # 向新 DataFrame 中添加空列 'd'
        df2["d"] = []
        # 对结果进行 reset_index 操作
        result = df2.reset_index()
        # 使用测试模块中的方法验证结果与预期是否相同
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器进行参数化测试
    @pytest.mark.parametrize(
        "timezone, year, month, day, hour",
        [["America/Chicago", 2013, 11, 3, 1], ["America/Santiago", 2021, 4, 3, 23]],
    )
    # 定义一个测试方法，测试带有折叠时间的索引重建功能
    def test_reindex_timestamp_with_fold(self, timezone, year, month, day, hour):
        # 见 gh-40817，注释指向相关 GitHub 问题编号
        # 获取指定时区的时区对象
        test_timezone = gettz(timezone)
        # 创建第一个时间戳，带有指定的年、月、日、小时，无折叠
        transition_1 = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=0,
            tzinfo=test_timezone,
        )
        # 创建第二个时间戳，带有指定的年、月、日、小时，有折叠
        transition_2 = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=1,
            tzinfo=test_timezone,
        )
        # 创建包含两个时间戳和对应值的数据框，并以时间戳作为索引
        df = (
            DataFrame({"index": [transition_1, transition_2], "vals": ["a", "b"]})
            .set_index("index")
            .reindex(["1", "2"])
        )
        # 创建预期的数据框，索引为字符串"1"和"2"，值为 NaN
        exp = DataFrame({"index": ["1", "2"], "vals": [np.nan, np.nan]}).set_index(
            "index"
        )
        # 将预期的数据框类型转换为与 df.vals 相同的类型
        exp = exp.astype(df.vals.dtype)
        # 断言 df 和 exp 是否相等
        tm.assert_frame_equal(
            df,
            exp,
        )
class TestDataFrameSelectReindex:
    # 这些是基于重新索引的特定测试；其他索引测试应放在test_indexing中

    @pytest.mark.xfail(
        not IS64 or (is_platform_windows() and not np_version_gt2),
        reason="Passes int32 values to DatetimeArray in make_na_array on "
        "windows, 32bit linux builds",
    )
    # 测试重新索引时的填充值处理，针对GH#52586
    def test_reindex_tzaware_fill_value(self):
        # 创建包含单个值的DataFrame对象
        df = DataFrame([[1]])

        # 创建具有时区信息的时间戳
        ts = pd.Timestamp("2023-04-10 17:32", tz="US/Pacific")
        # 在指定轴上重新索引DataFrame，并用指定值填充缺失位置
        res = df.reindex([0, 1], axis=1, fill_value=ts)
        # 断言第一列的数据类型为带时区信息的时间戳类型
        assert res.dtypes[1] == pd.DatetimeTZDtype(unit="s", tz="US/Pacific")
        # 创建预期的DataFrame，其中第二列转换为与结果DataFrame相同的数据类型
        expected = DataFrame({0: [1], 1: [ts]})
        expected[1] = expected[1].astype(res.dtypes[1])
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(res, expected)

        # 将时间戳转换为不带时区的时期对象
        per = ts.tz_localize(None).to_period("s")
        # 再次进行重新索引，填充值为时期对象
        res = df.reindex([0, 1], axis=1, fill_value=per)
        # 断言第一列的数据类型为时期类型
        assert res.dtypes[1] == pd.PeriodDtype("s")
        # 创建预期的DataFrame
        expected = DataFrame({0: [1], 1: [per]})
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(res, expected)

        # 创建时间间隔对象，起始和结束时间为时间戳及其后1秒
        interval = pd.Interval(ts, ts + pd.Timedelta(seconds=1))
        # 再次进行重新索引，填充值为时间间隔对象
        res = df.reindex([0, 1], axis=1, fill_value=interval)
        # 断言第一列的数据类型为时间间隔类型
        assert res.dtypes[1] == pd.IntervalDtype("datetime64[s, US/Pacific]", "right")
        # 创建预期的DataFrame，其中第二列转换为与结果DataFrame相同的数据类型
        expected = DataFrame({0: [1], 1: [interval]})
        expected[1] = expected[1].astype(res.dtypes[1])
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(res, expected)

    # 测试重新索引时的日期填充值处理
    def test_reindex_date_fill_value(self):
        # 创建日期范围并将其构造为DataFrame对象
        arr = date_range("2016-01-01", periods=6).values.reshape(3, 2)
        df = DataFrame(arr, columns=["A", "B"], index=range(3))

        # 获取DataFrame的第一个元素作为时间戳对象
        ts = df.iloc[0, 0]
        # 将时间戳转换为日期对象
        fv = ts.date()

        # 在指定行索引和列索引上重新索引DataFrame，填充值为日期对象
        res = df.reindex(index=range(4), columns=["A", "B", "C"], fill_value=fv)

        # 创建预期的DataFrame，使用填充值在新增列"C"中填充
        expected = DataFrame(
            {"A": df["A"].tolist() + [fv], "B": df["B"].tolist() + [fv], "C": [fv] * 4},
            dtype=object,
        )
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(res, expected)

        # 只在行上进行重新索引
        res = df.reindex(index=range(4), fill_value=fv)
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等，仅包括"A"和"B"列
        tm.assert_frame_equal(res, expected[["A", "B"]])

        # 使用可转换为日期时间的字符串再次进行重新索引
        res = df.reindex(
            index=range(4), columns=["A", "B", "C"], fill_value="2016-01-01"
        )
        # 创建预期的DataFrame，使用时间戳对象在新增列"C"中填充
        expected = DataFrame(
            {"A": df["A"].tolist() + [ts], "B": df["B"].tolist() + [ts], "C": [ts] * 4},
        )
        # 使用测试工具检查结果DataFrame和预期DataFrame是否相等
        tm.assert_frame_equal(res, expected)
    def test_reindex_with_multi_index(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # 在这个测试中，测试重新索引具有新MultiIndex的多索引DataFrame
        #
        # 确认当使用无填充、向后填充和填充时，我们可以正确地重新索引具有新MultiIndex对象的多索引DataFrame
        #
        # 这个测试中使用的DataFrame `df` 是：
        #       c
        #  a b
        # -1 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #  0 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #  1 0  A
        #    1  B
        #    2  C
        #    3  D
        #    4  E
        #    5  F
        #    6  G
        #
        # 另一个MultiIndex `new_multi_index` 是：
        # 0: 0 0.5
        # 1:   2.0
        # 2:   5.0
        # 3:   5.8
        df = DataFrame(
            {
                "a": [-1] * 7 + [0] * 7 + [1] * 7,
                "b": list(range(7)) * 3,
                "c": ["A", "B", "C", "D", "E", "F", "G"] * 3,
            }
        ).set_index(["a", "b"])
        new_index = [0.5, 2.0, 5.0, 5.8]
        new_multi_index = MultiIndex.from_product([[0], new_index], names=["a", "b"])

        # 重新索引，不使用 `method` 参数
        reindexed = df.reindex(new_multi_index)
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": [np.nan, "C", "F", np.nan]}
        ).set_index(["a", "b"])
        tm.assert_frame_equal(expected, reindexed)

        # 重新索引，使用向后填充
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": ["B", "C", "F", "G"]}
        ).set_index(["a", "b"])
        reindexed_with_backfilling = df.reindex(new_multi_index, method="bfill")
        tm.assert_frame_equal(expected, reindexed_with_backfilling)

        reindexed_with_backfilling = df.reindex(new_multi_index, method="backfill")
        tm.assert_frame_equal(expected, reindexed_with_backfilling)

        # 重新索引，使用填充
        expected = DataFrame(
            {"a": [0] * 4, "b": new_index, "c": ["A", "C", "F", "F"]}
        ).set_index(["a", "b"])
        reindexed_with_padding = df.reindex(new_multi_index, method="pad")
        tm.assert_frame_equal(expected, reindexed_with_padding)

        reindexed_with_padding = df.reindex(new_multi_index, method="ffill")
        tm.assert_frame_equal(expected, reindexed_with_padding)
    # 测试不同的重新索引方法
    def test_reindex_methods(self, method, expected_values):
        # 创建一个包含一列"x"的DataFrame，值为0到4
        df = DataFrame({"x": list(range(5))})
        # 创建目标数组，用于重新索引DataFrame
        target = np.array([-0.1, 0.9, 1.1, 1.5])

        # 创建预期的DataFrame，根据目标数组索引，值为expected_values
        expected = DataFrame({"x": expected_values}, index=target)
        # 使用指定的方法(method)对DataFrame进行重新索引
        actual = df.reindex(target, method=method)
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(expected, actual)

        # 使用指定的方法(method)和容差(tolerance)对DataFrame进行重新索引
        actual = df.reindex(target, method=method, tolerance=1)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=[1, 1, 1, 1])
        tm.assert_frame_equal(expected, actual)

        # 将预期的DataFrame反转
        e2 = expected[::-1]
        # 使用反转后的目标数组对DataFrame进行重新索引，使用指定的方法(method)
        actual = df.reindex(target[::-1], method=method)
        # 断言实际结果与反转后的预期结果相等
        tm.assert_frame_equal(e2, actual)

        # 创建新的顺序列表new_order
        new_order = [3, 0, 2, 1]
        # 使用新顺序的目标数组对DataFrame进行重新索引，使用指定的方法(method)
        e2 = expected.iloc[new_order]
        actual = df.reindex(target[new_order], method=method)
        # 断言实际结果与新顺序的预期结果相等
        tm.assert_frame_equal(e2, actual)

        # 如果方法(method)为"backfill"，则switched_method为"pad"；如果方法为"pad"，则switched_method为"backfill"；否则为method本身
        switched_method = (
            "pad" if method == "backfill" else "backfill" if method == "pad" else method
        )
        # 对DataFrame进行倒序，并使用switched_method方法对目标数组重新索引
        actual = df[::-1].reindex(target, method=switched_method)
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(expected, actual)

    # 测试nearest方法的特殊情况
    def test_reindex_methods_nearest_special(self):
        # 创建一个包含一列"x"的DataFrame，值为0到4
        df = DataFrame({"x": list(range(5))})
        # 创建目标数组，用于重新索引DataFrame
        target = np.array([-0.1, 0.9, 1.1, 1.5])

        # 创建预期的DataFrame，根据目标数组索引，值为特定的nearest方法结果
        expected = DataFrame({"x": [0, 1, 1, np.nan]}, index=target)
        # 使用nearest方法和指定容差对DataFrame进行重新索引
        actual = df.reindex(target, method="nearest", tolerance=0.2)
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(expected, actual)

        # 创建预期的DataFrame，根据目标数组索引，值为特定的nearest方法结果
        expected = DataFrame({"x": [0, np.nan, 1, np.nan]}, index=target)
        # 使用nearest方法和指定容差列表对DataFrame进行重新索引
        actual = df.reindex(target, method="nearest", tolerance=[0.5, 0.01, 0.4, 0.1])
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(expected, actual)

    # 测试带时区信息的nearest方法
    def test_reindex_nearest_tz(self, tz_aware_fixture):
        # GH26683
        # 获取时区信息
        tz = tz_aware_fixture
        # 创建包含时区信息的日期索引
        idx = date_range("2019-01-01", periods=5, tz=tz)
        # 创建一个包含一列"x"的DataFrame，值为0到4，索引为idx
        df = DataFrame({"x": list(range(5))}, index=idx)

        # 获取预期的DataFrame，根据前三个元素的nearest方法结果
        expected = df.head(3)
        # 使用nearest方法对前三个元素的日期索引进行重新索引
        actual = df.reindex(idx[:3], method="nearest")
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(expected, actual)

    # 测试空DataFrame的nearest方法
    def test_reindex_nearest_tz_empty_frame(self):
        # https://github.com/pandas-dev/pandas/issues/31964
        # 创建包含指定时间戳的DatetimeIndex
        dti = pd.DatetimeIndex(["2016-06-26 14:27:26+00:00"])
        # 创建一个空的DataFrame，索引为指定的DatetimeIndex
        df = DataFrame(index=pd.DatetimeIndex(["2016-07-04 14:00:59+00:00"]))
        # 创建预期的DataFrame，索引为dti，使用nearest方法对DataFrame进行重新索引
        expected = DataFrame(index=dti)
        # 使用nearest方法对指定时间戳的DatetimeIndex进行重新索引
        result = df.reindex(dti, method="nearest")
        # 断言实际结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试带NAT值的DataFrame的重新索引
    def test_reindex_frame_add_nat(self):
        # 创建一个时间范围索引
        rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
        # 创建一个包含"A"和"B"两列的DataFrame，其中"A"列为随机数列，"B"列为时间范围索引rng
        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": rng}
        )

        # 使用默认的整数索引对DataFrame进行重新索引
        result = df.reindex(range(15))
        # 断言结果的"B"列的数据类型为日期时间类型
        assert np.issubdtype(result["B"].dtype, np.dtype("M8[ns]"))

        # 检查结果中"B"列中最后5行是否都为缺失值(NAT)
        mask = isna(result)["B"]
        assert mask[-5:].all()
        # 检查结果中"B"列中除了最后5行外是否都不含缺失值(NAT)
        assert not mask[:-5].any()

    @pytest.mark.parametrize(
        "method, exp_values",
        [("ffill", [0, 1, 2, 3]), ("bfill", [1.0, 2.0, 3.0, np.nan])],
    )
    # 定义一个测试方法，用于测试 reindex 函数的时区、前向填充和后向填充功能
    def test_reindex_frame_tz_ffill_bfill(self, frame_or_series, method, exp_values):
        # GH#38566
        # 创建一个 DataFrame 或 Series 对象，指定索引为从 "2020-01-01 00:00:00" 开始的四个时间点，频率为每小时，时区为 UTC
        obj = frame_or_series(
            [0, 1, 2, 3],
            index=date_range("2020-01-01 00:00:00", periods=4, freq="h", tz="UTC"),
        )
        # 创建一个新的索引，从 "2020-01-01 00:01:00" 开始，频率为每小时，时区为 UTC
        new_index = date_range("2020-01-01 00:01:00", periods=4, freq="h", tz="UTC")
        # 使用 reindex 方法重新索引对象 obj，采用指定的方法和容忍度为 1 小时
        result = obj.reindex(new_index, method=method, tolerance=pd.Timedelta("1 hour"))
        # 创建预期的 DataFrame 或 Series 对象，其数据和新索引匹配
        expected = frame_or_series(exp_values, index=new_index)
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于测试 reindex 函数的限制功能
    def test_reindex_limit(self):
        # GH 28631
        # 初始化一个包含数据的列表
        data = [["A", "A", "A"], ["B", "B", "B"], ["C", "C", "C"], ["D", "D", "D"]]
        # 初始化预期结果的列表，增加了额外的行以测试填充方法
        exp_data = [
            ["A", "A", "A"],
            ["B", "B", "B"],
            ["C", "C", "C"],
            ["D", "D", "D"],
            ["D", "D", "D"],
            [np.nan, np.nan, np.nan],
        ]
        # 创建 DataFrame 对象
        df = DataFrame(data)
        # 使用 reindex 方法重新索引 DataFrame，指定填充方法为前向填充，限制为 1
        result = df.reindex([0, 1, 2, 3, 4, 5], method="ffill", limit=1)
        # 创建预期的 DataFrame 对象，其数据和新索引匹配
        expected = DataFrame(exp_data)
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器来定义多个测试参数
    @pytest.mark.parametrize(
        "idx, check_index_type",
        [
            # 不同的索引顺序，以及对结果的类型检查
            [["C", "B", "A"], True],
            [["F", "C", "A", "D"], True],
            [["A"], True],
            [["A", "B", "C"], True],
            [["C", "A", "B"], True],
            [["C", "B"], True],
            [["C", "A"], True],
            [["A", "B"], True],
            [["B", "A", "C"], True],
            # 使用这些索引会导致不同的 MultiIndex 级别
            [["D", "F"], False],
            [["A", "C", "B"], False],
        ],
    )
    # 定义一个测试方法，用于验证 reindex 方法在不同 MultiIndex 级别下的行为
    def test_reindex_level_verify_first_level(self, idx, check_index_type):
        # 创建一个包含随机数据的 DataFrame 对象
        df = DataFrame(
            {
                "jim": list("B" * 4 + "A" * 2 + "C" * 3),
                "joe": list("abcdeabcd")[::-1],
                "jolie": [10, 20, 30] * 3,
                "joline": np.random.default_rng(2).integers(0, 1000, 9),
            }
        )
        # 设置用于索引的列
        icol = ["jim", "joe", "jolie"]

        # 定义一个函数，根据值返回非零索引位置
        def f(val):
            return np.nonzero((df["jim"] == val).to_numpy())[0]

        # 获取所有指定索引的位置
        i = np.concatenate(list(map(f, idx)))
        # 使用 set_index 和 reindex 方法重建左侧 DataFrame，以及基于位置的右侧 DataFrame
        left = df.set_index(icol).reindex(idx, level="jim")
        right = df.iloc[i].set_index(icol)
        # 断言左侧和右侧 DataFrame 是否相等，根据 check_index_type 进行检查
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    # 使用 pytest 的参数化装饰器来定义多个测试参数
    @pytest.mark.parametrize(
        "idx",
        [
            # 不同的索引顺序
            ("mid",),
            ("mid", "btm"),
            ("mid", "btm", "top"),
            ("mid", "top"),
            ("mid", "top", "btm"),
            ("btm",),
            ("btm", "mid"),
            ("btm", "mid", "top"),
            ("btm", "top"),
            ("btm", "top", "mid"),
            ("top",),
            ("top", "mid"),
            ("top", "mid", "btm"),
            ("top", "btm"),
            ("top", "btm", "mid"),
        ],
    )
    # 定义一个测试方法，验证在给定索引下第一级重复出现的情况
    def test_reindex_level_verify_first_level_repeats(self, idx):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "jim": ["mid"] * 5 + ["btm"] * 8 + ["top"] * 7,  # 列 'jim' 包含多个重复值
                "joe": ["3rd"] * 2 + ["1st"] * 3 + ["2nd"] * 3 + ["1st"] * 2
                + ["3rd"] * 3 + ["1st"] * 2 + ["3rd"] * 3 + ["2nd"] * 2,  # 列 'joe' 包含多个重复值
                # 'jolie' 列需要与 'jim' 和 'joe' 列联合唯一，否则重新索引时可能失败
                "jolie": np.concatenate(
                    [
                        np.random.default_rng(2).choice(1000, x, replace=False)
                        for x in [2, 3, 3, 2, 3, 2, 3, 2]
                    ]
                ),
                # 创建一个标准正态分布的随机数据列 'joline'
                "joline": np.random.default_rng(2).standard_normal(20).round(3) * 10,
            }
        )
        # 设置索引列
        icol = ["jim", "joe", "jolie"]

        # 定义一个函数 f，返回指定值在 'jim' 列中出现的索引
        def f(val):
            return np.nonzero((df["jim"] == val).to_numpy())[0]

        # 将所有索引值在 'jim' 列中出现的索引连接成一个数组 i
        i = np.concatenate(list(map(f, idx)))
        # 使用 icol 设置 DataFrame 的多级索引，并重新索引 idx 列，主要操作是在 'jim' 级别上
        left = df.set_index(icol).reindex(idx, level="jim")
        # 使用 i 索引选择 df 的子集，并使用 icol 设置索引
        right = df.iloc[i].set_index(icol)
        # 使用测试工具库 tm 检查 left 和 right 的数据框是否相等
        tm.assert_frame_equal(left, right)

    # 使用 pytest 的参数化装饰器，定义多组参数化测试数据
    @pytest.mark.parametrize(
        "idx, indexer",
        [
            # 第一组参数化数据：idx 和对应的索引器
            [
                ["1st", "2nd", "3rd"],  # idx 列
                [2, 3, 4, 0, 1, 8, 9, 5, 6, 7, 10, 11, 12, 13, 14, 18, 19, 15, 16, 17],  # 索引器
            ],
            # 第二组参数化数据：idx 和对应的索引器
            [
                ["3rd", "2nd", "1st"],  # idx 列
                [0, 1, 2, 3, 4, 10, 11, 12, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 13, 14],  # 索引器
            ],
            # 第三组参数化数据：idx 和对应的索引器
            [["2nd", "3rd"], [0, 1, 5, 6, 7, 10, 11, 12, 18, 19, 15, 16, 17]],  # idx 列和索引器
            # 第四组参数化数据：idx 和对应的索引器
            [["3rd", "1st"], [0, 1, 2, 3, 4, 10, 11, 12, 8, 9, 15, 16, 17, 13, 14]],  # idx 列和索引器
        ],
    )
    # 定义一个测试方法，用于验证重新索引和重复项检查
    def test_reindex_level_verify(self, idx, indexer, check_index_type):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "jim": list("B" * 4 + "A" * 2 + "C" * 3),  # 列 'jim' 包含重复项 'B' 和 'A'
                "joe": list("abcdeabcd")[::-1],  # 列 'joe' 包含字母序列的逆向排列
                "jolie": [10, 20, 30] * 3,  # 列 'jolie' 包含重复的数值序列
                "joline": np.random.default_rng(2).integers(0, 1000, 9),  # 列 'joline' 包含随机整数数据
            }
        )
        icol = ["jim", "joe", "jolie"]  # 定义用于索引的列
        # 使用指定列设置索引，然后根据 'joe' 列重新索引 DataFrame
        left = df.set_index(icol).reindex(idx, level="joe")
        # 根据给定的索引器索引原始 DataFrame，然后设置相同的索引
        right = df.iloc[indexer].set_index(icol)
        # 使用 pytest 的断言方法来比较两个 DataFrame 是否相等，根据 check_index_type 参数决定是否检查索引类型
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    # 使用 pytest 的参数化装饰器标记，定义多组参数进行测试
    @pytest.mark.parametrize(
        "idx, indexer, check_index_type",
        [
            [list("abcde"), [3, 2, 1, 0, 5, 4, 8, 7, 6], True],  # 参数化测试集合 1
            [list("abcd"), [3, 2, 1, 0, 5, 8, 7, 6], True],    # 参数化测试集合 2
            [list("abc"), [3, 2, 1, 8, 7, 6], True],           # 参数化测试集合 3
            [list("eca"), [1, 3, 4, 6, 8], True],              # 参数化测试集合 4
            [list("edc"), [0, 1, 4, 5, 6], True],              # 参数化测试集合 5
            [list("eadbc"), [3, 0, 2, 1, 4, 5, 8, 7, 6], True], # 参数化测试集合 6
            [list("edwq"), [0, 4, 5], True],                   # 参数化测试集合 7
            [list("wq"), [], False],                          # 参数化测试集合 8
        ],
    )
    # 定义另一个测试方法，用于验证重新索引和重复项检查
    def test_reindex_level_verify(self, idx, indexer, check_index_type):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "jim": list("B" * 4 + "A" * 2 + "C" * 3),  # 列 'jim' 包含重复项 'B' 和 'A'
                "joe": list("abcdeabcd")[::-1],  # 列 'joe' 包含字母序列的逆向排列
                "jolie": [10, 20, 30] * 3,  # 列 'jolie' 包含重复的数值序列
                "joline": np.random.default_rng(2).integers(0, 1000, 9),  # 列 'joline' 包含随机整数数据
            }
        )
        icol = ["jim", "joe", "jolie"]  # 定义用于索引的列
        # 使用指定列设置索引，然后根据 'joe' 列重新索引 DataFrame
        left = df.set_index(icol).reindex(idx, level="joe")
        # 根据给定的索引器索引原始 DataFrame，然后设置相同的索引
        right = df.iloc[indexer].set_index(icol)
        # 使用 pytest 的断言方法来比较两个 DataFrame 是否相等，根据 check_index_type 参数决定是否检查索引类型
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)
    # 定义一个测试方法，用于测试非单调重新索引方法
    def test_non_monotonic_reindex_methods(self):
        # 创建一个日期范围，从"2013-08-01"开始，6个工作日，频率为工作日
        dr = date_range("2013-08-01", periods=6, freq="B")
        # 生成一个形状为(6, 1)的随机正态分布数据数组
        data = np.random.default_rng(2).standard_normal((6, 1))
        # 创建一个 DataFrame，使用上述数据，索引为日期范围 dr，列名为"A"
        df = DataFrame(data, index=dr, columns=list("A"))
        # 创建一个索引顺序为[3, 4, 5, 0, 1, 2]的 DataFrame，保持数据和列名不变
        df_rev = DataFrame(data, index=dr[[3, 4, 5] + [0, 1, 2]], columns=list("A"))
        # 抛出 ValueError 异常，消息为"index must be monotonic increasing or decreasing"
        msg = "index must be monotonic increasing or decreasing"
        # 使用 pytest 断言捕获异常，并验证异常消息
        with pytest.raises(ValueError, match=msg):
            # 尝试使用"pad"方法重新索引 df_rev，目标索引为 df.index
            df_rev.reindex(df.index, method="pad")
        with pytest.raises(ValueError, match=msg):
            # 尝试使用"ffill"方法重新索引 df_rev，目标索引为 df.index
            df_rev.reindex(df.index, method="ffill")
        with pytest.raises(ValueError, match=msg):
            # 尝试使用"bfill"方法重新索引 df_rev，目标索引为 df.index
            df_rev.reindex(df.index, method="bfill")
        with pytest.raises(ValueError, match=msg):
            # 尝试使用"nearest"方法重新索引 df_rev，目标索引为 df.index
            df_rev.reindex(df.index, method="nearest")

    # 定义一个测试方法，用于测试稀疏数据的重新索引
    def test_reindex_sparse(self):
        # 创建一个包含稀疏数据的 DataFrame，列"A"包含 [0, 1]，列"B"包含 SparseDtype("int64", 0) 类型的稀疏数组
        df = DataFrame(
            {"A": [0, 1], "B": pd.array([0, 1], dtype=pd.SparseDtype("int64", 0))}
        )
        # 使用给定索引 [0, 2] 对 df 进行重新索引
        result = df.reindex([0, 2])
        # 创建一个期望的 DataFrame，索引为 [0, 2]，列"A"包含 [0.0, np.nan]，列"B"包含 SparseDtype("float64", 0.0) 类型的稀疏数组
        expected = DataFrame(
            {
                "A": [0.0, np.nan],
                "B": pd.array([0.0, np.nan], dtype=pd.SparseDtype("float64", 0.0)),
            },
            index=[0, 2],
        )
        # 使用测试工具比较 result 和 expected 的内容是否一致
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试重新索引的功能，接受一个名为 float_frame 的参数
    def test_reindex(self, float_frame):
        # 创建一个包含 30 个浮点数的 Series 对象，索引为从 "2020-01-01" 开始的 30 天
        datetime_series = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )

        # 使用 datetime_series 的索引重新索引 float_frame，得到一个新的 DataFrame 对象
        newFrame = float_frame.reindex(datetime_series.index)

        # 遍历新 DataFrame 的每一列
        for col in newFrame.columns:
            # 遍历每一列中的每个索引和对应的值
            for idx, val in newFrame[col].items():
                # 如果当前索引 idx 存在于 float_frame 的索引中
                if idx in float_frame.index:
                    # 如果新 DataFrame 中的值是 NaN，则断言 float_frame 中相同索引位置的值也是 NaN
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    # 否则断言新 DataFrame 中的值与 float_frame 中相同索引位置的值相等
                    else:
                        assert val == float_frame[col][idx]
                # 如果当前索引 idx 不在 float_frame 的索引中，则断言新 DataFrame 中的值是 NaN
                else:
                    assert np.isnan(val)

        # 遍历新 DataFrame 的每一列和对应的 Series 对象
        for col, series in newFrame.items():
            # 断言每个 Series 的索引与新 DataFrame 的索引相等
            tm.assert_index_equal(series.index, newFrame.index)

        # 使用空索引重新索引 float_frame，得到一个空的 DataFrame 对象
        emptyFrame = float_frame.reindex(Index([]))
        # 断言空 DataFrame 的索引长度为 0
        assert len(emptyFrame.index) == 0

        # 使用非连续的索引重新索引 float_frame，得到一个新的 DataFrame 对象
        nonContigFrame = float_frame.reindex(datetime_series.index[::2])

        # 遍历非连续索引新 DataFrame 的每一列
        for col in nonContigFrame.columns:
            # 遍历每一列中的每个索引和对应的值
            for idx, val in nonContigFrame[col].items():
                # 如果当前索引 idx 存在于 float_frame 的索引中
                if idx in float_frame.index:
                    # 如果新 DataFrame 中的值是 NaN，则断言 float_frame 中相同索引位置的值也是 NaN
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    # 否则断言新 DataFrame 中的值与 float_frame 中相同索引位置的值相等
                    else:
                        assert val == float_frame[col][idx]
                # 如果当前索引 idx 不在 float_frame 的索引中，则断言新 DataFrame 中的值是 NaN
                else:
                    assert np.isnan(val)

        # 遍历非连续索引新 DataFrame 的每一列和对应的 Series 对象
        for col, series in nonContigFrame.items():
            # 断言每个 Series 的索引与非连续索引新 DataFrame 的索引相等
            tm.assert_index_equal(series.index, nonContigFrame.index)

        # 处理边界情况：使用 float_frame 的索引重新索引 float_frame，得到一个新的 DataFrame 对象
        newFrame = float_frame.reindex(float_frame.index)
        # 断言新 DataFrame 的索引与 float_frame 的索引相同
        assert newFrame.index.is_(float_frame.index)

        # 处理长度为零的情况：使用空索引重新索引 float_frame，得到一个新的 DataFrame 对象
        newFrame = float_frame.reindex([])
        # 断言新 DataFrame 是空的
        assert newFrame.empty
        # 断言新 DataFrame 的列数与 float_frame 的列数相等
        assert len(newFrame.columns) == len(float_frame.columns)

        # 处理长度为零的情况，并使用非空索引重新索引 float_frame，得到一个新的 DataFrame 对象
        newFrame = float_frame.reindex([])
        newFrame = newFrame.reindex(float_frame.index)
        # 断言新 DataFrame 的索引长度与 float_frame 的索引长度相等
        assert len(newFrame.index) == len(float_frame.index)
        # 断言新 DataFrame 的列数与 float_frame 的列数相等
        assert len(newFrame.columns) == len(float_frame.columns)

        # 处理使用非 Index 类型对象重新索引的情况
        newFrame = float_frame.reindex(list(datetime_series.index))
        # 期望的索引是 datetime_series.index 的一个副本
        expected = datetime_series.index._with_freq(None)
        # 断言新 DataFrame 的索引与期望的索引相等
        tm.assert_index_equal(newFrame.index, expected)

        # 处理复制但不包含任何轴的情况
        result = float_frame.reindex()
        # 断言结果 DataFrame 与 float_frame 相等
        tm.assert_frame_equal(result, float_frame)
        # 断言结果对象与 float_frame 对象不是同一个对象
        assert result is not float_frame
    def test_reindex_nan(self):
        # 创建一个包含索引和列的数据框
        df = DataFrame(
            [[1, 2], [3, 5], [7, 11], [9, 23]],
            index=[2, np.nan, 1, 5],  # 设置索引，包括一个 NaN 值
            columns=["joe", "jim"],  # 设置列名
        )

        # 创建两个列表，用于重新索引数据框并进行比较
        i, j = [np.nan, 5, 5, np.nan, 1, 2, np.nan], [1, 3, 3, 1, 2, 0, 1]
        # 使用 tm.assert_frame_equal 检查重新索引后的数据框是否与预期相同
        tm.assert_frame_equal(df.reindex(i), df.iloc[j])

        # 将索引转换为对象类型，并再次进行比较
        df.index = df.index.astype("object")
        tm.assert_frame_equal(df.reindex(i), df.iloc[j], check_index_type=False)

        # GH10388
        # 创建一个包含不同类型数据的数据框
        df = DataFrame(
            {
                "other": ["a", "b", np.nan, "c"],
                "date": ["2015-03-22", np.nan, "2012-01-08", np.nan],
                "amount": [2, 3, 4, 5],
            }
        )

        # 将 "date" 列转换为日期时间格式
        df["date"] = pd.to_datetime(df.date)
        # 计算日期时间差，并将结果作为新列添加到数据框中
        df["delta"] = (pd.to_datetime("2015-06-18") - df["date"]).shift(1)

        # 根据多级索引进行数据框重置，并进行比较
        left = df.set_index(["delta", "other", "date"]).reset_index()
        right = df.reindex(columns=["delta", "other", "date", "amount"])
        tm.assert_frame_equal(left, right)

    def test_reindex_name_remains(self):
        # 创建一个随机序列并将其转换为数据框
        s = Series(np.random.default_rng(2).random(10))
        df = DataFrame(s, index=np.arange(len(s)))
        # 创建一个命名的序列用于重新索引
        i = Series(np.arange(10), name="iname")

        # 根据命名的索引重新索引数据框，并验证索引名保持不变
        df = df.reindex(i)
        assert df.index.name == "iname"

        # 根据新的命名索引重新索引数据框，并验证索引名变为新命名
        df = df.reindex(Index(np.arange(10), name="tmpname"))
        assert df.index.name == "tmpname"

        # 创建一个随机序列并将其转置为数据框
        s = Series(np.random.default_rng(2).random(10))
        df = DataFrame(s.T, index=np.arange(len(s)))
        # 创建一个命名的序列用于重新索引列
        i = Series(np.arange(10), name="iname")
        # 根据命名的列索引重新索引数据框，并验证列名保持不变
        df = df.reindex(columns=i)
        assert df.columns.name == "iname"

    def test_reindex_int(self, int_frame):
        # 根据间隔选择索引，创建一个更小的数据框
        smaller = int_frame.reindex(int_frame.index[::2])

        # 验证选择的列数据类型为 np.int64
        assert smaller["A"].dtype == np.int64

        # 根据原始索引重新索引数据框，验证选择的列数据类型为 np.float64
        bigger = smaller.reindex(int_frame.index)
        assert bigger["A"].dtype == np.float64

        # 根据指定列重新索引数据框，并验证选择的列数据类型为 np.int64
        smaller = int_frame.reindex(columns=["A", "B"])
        assert smaller["A"].dtype == np.int64

    def test_reindex_columns(self, float_frame):
        # 根据指定列重新索引数据框，创建一个新的数据框
        new_frame = float_frame.reindex(columns=["A", "B", "E"])

        # 使用 tm.assert_series_equal 检查新数据框的 "B" 列与原始数据框的相同
        tm.assert_series_equal(new_frame["B"], float_frame["B"])
        # 验证新数据框的 "E" 列全部为 NaN
        assert np.isnan(new_frame["E"]).all()
        # 验证新数据框中不包含 "C" 列
        assert "C" not in new_frame

        # 创建一个空数据框
        new_frame = float_frame.reindex(columns=[])
        # 验证新数据框为空
        assert new_frame.empty
    def test_reindex_columns_method(self):
        # GH 14992, reindexing over columns ignored method
        # 创建一个包含浮点数数据的DataFrame，指定索引和列，并设置数据类型为float
        df = DataFrame(
            data=[[11, 12, 13], [21, 22, 23], [31, 32, 33]],
            index=[1, 2, 4],
            columns=[1, 2, 4],
            dtype=float,
        )

        # 使用默认方法重新索引列
        result = df.reindex(columns=range(6))
        # 期望的DataFrame，包含NaN值，扩展到指定的列范围
        expected = DataFrame(
            data=[
                [np.nan, 11, 12, np.nan, 13, np.nan],
                [np.nan, 21, 22, np.nan, 23, np.nan],
                [np.nan, 31, 32, np.nan, 33, np.nan],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        # 断言重新索引后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 使用前向填充方法重新索引列
        result = df.reindex(columns=range(6), method="ffill")
        # 期望的DataFrame，使用前向填充方法填充NaN值
        expected = DataFrame(
            data=[
                [np.nan, 11, 12, 12, 13, 13],
                [np.nan, 21, 22, 22, 23, 23],
                [np.nan, 31, 32, 32, 33, 33],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        # 断言重新索引后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 使用后向填充方法重新索引列
        result = df.reindex(columns=range(6), method="bfill")
        # 期望的DataFrame，使用后向填充方法填充NaN值
        expected = DataFrame(
            data=[
                [11, 11, 12, 13, 13, np.nan],
                [21, 21, 22, 23, 23, np.nan],
                [31, 31, 32, 33, 33, np.nan],
            ],
            index=[1, 2, 4],
            columns=range(6),
            dtype=float,
        )
        # 断言重新索引后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    def test_reindex_axes(self):
        # GH 3317, reindexing by both axes loses freq of the index
        # 创建一个包含全部元素为1的3x3 DataFrame，指定日期时间索引和列名称
        df = DataFrame(
            np.ones((3, 3)),
            index=[datetime(2012, 1, 1), datetime(2012, 1, 2), datetime(2012, 1, 3)],
            columns=["a", "b", "c"],
        )
        # 创建一个日期范围，以每日频率
        time_freq = date_range("2012-01-01", "2012-01-03", freq="d")
        some_cols = ["a", "b"]

        # 重新索引索引，获取索引频率
        index_freq = df.reindex(index=time_freq).index.freq
        # 重新索引行和列，获取索引频率
        both_freq = df.reindex(index=time_freq, columns=some_cols).index.freq
        # 顺序重新索引，获取索引频率
        seq_freq = df.reindex(index=time_freq).reindex(columns=some_cols).index.freq
        # 断言三种方式的重新索引结果的索引频率相等
        assert index_freq == both_freq
        assert index_freq == seq_freq
    def test_reindex_fill_value(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        # axis=0
        # 在轴0上重新索引DataFrame，将超出范围的行用NaN填充
        result = df.reindex(list(range(15)))
        assert np.isnan(result.values[-5:]).all()

        # 在轴0上重新索引DataFrame，使用fill_value=0来填充缺失值
        result = df.reindex(range(15), fill_value=0)
        expected = df.reindex(range(15)).fillna(0)
        tm.assert_frame_equal(result, expected)

        # axis=1
        # 在轴1上重新索引DataFrame的列，使用fill_value=0.0填充缺失值
        result = df.reindex(columns=range(5), fill_value=0.0)
        expected = df.copy()
        expected[4] = 0.0
        tm.assert_frame_equal(result, expected)

        # 在轴1上重新索引DataFrame的列，使用fill_value=0填充缺失值
        result = df.reindex(columns=range(5), fill_value=0)
        expected = df.copy()
        expected[4] = 0
        tm.assert_frame_equal(result, expected)

        # 在轴1上重新索引DataFrame的列，使用fill_value="foo"填充缺失值
        result = df.reindex(columns=range(5), fill_value="foo")
        expected = df.copy()
        expected[4] = "foo"
        tm.assert_frame_equal(result, expected)

        # other dtypes
        # 添加新列"foo"并设置其所有值为字符串"foo"
        df["foo"] = "foo"
        # 在轴0上重新索引DataFrame，使用fill_value="0"填充缺失值
        result = df.reindex(range(15), fill_value="0")
        expected = df.reindex(range(15)).fillna("0")
        tm.assert_frame_equal(result, expected)

    def test_reindex_uint_dtypes_fill_value(self, any_unsigned_int_numpy_dtype):
        # GH#48184
        # 创建具有指定无符号整数dtype的DataFrame
        df = DataFrame({"a": [1, 2], "b": [1, 2]}, dtype=any_unsigned_int_numpy_dtype)
        # 在轴0和轴1上重新索引DataFrame，使用fill_value=10填充缺失值
        result = df.reindex(columns=list("abcd"), index=[0, 1, 2, 3], fill_value=10)
        expected = DataFrame(
            {"a": [1, 2, 10, 10], "b": [1, 2, 10, 10], "c": 10, "d": 10},
            dtype=any_unsigned_int_numpy_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex_single_column_ea_index_and_columns(self, any_numeric_ea_dtype):
        # GH#48190
        # 创建具有指定扩展精度数值dtype的DataFrame
        df = DataFrame({"a": [1, 2]}, dtype=any_numeric_ea_dtype)
        # 在轴0和轴1上重新索引DataFrame，使用fill_value=10填充缺失值
        result = df.reindex(columns=list("ab"), index=[0, 1, 2], fill_value=10)
        expected = DataFrame(
            {"a": Series([1, 2, 10], dtype=any_numeric_ea_dtype), "b": 10}
        )
        tm.assert_frame_equal(result, expected)

    def test_reindex_dups(self):
        # GH4746, reindex on duplicate index error messages
        arr = np.random.default_rng(2).standard_normal(10)
        # 创建具有重复索引标签的DataFrame
        df = DataFrame(arr, index=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

        # 设置索引是允许的
        result = df.copy()
        result.index = list(range(len(df)))
        expected = DataFrame(arr, index=list(range(len(df))))
        tm.assert_frame_equal(result, expected)

        # 重新索引失败的情况
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            # 尝试在具有重复标签的轴上重新索引会引发值错误异常
            df.reindex(index=list(range(len(df))))
    def test_reindex_with_duplicate_columns(self):
        # 测试处理具有重复列的重新索引情况
        df = DataFrame(
            [[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=["bar", "a", "a"]
        )
        msg = "cannot reindex on an axis with duplicate labels"
        # 确保在存在重复标签的轴上重新索引会引发 ValueError 异常，异常信息为 msg
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=["bar"])
        # 同样，使用包含重复标签和非重复标签的列表进行重新索引，应该引发相同的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=["bar", "foo"])

    def test_reindex_axis_style(self):
        # 测试不同风格的轴重新索引操作
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        expected = DataFrame(
            {"A": [1, 2, np.nan], "B": [4, 5, np.nan]}, index=[0, 1, 3]
        )
        # 使用整数列表进行索引，期望得到指定的索引和 NaN 值的数据框
        result = df.reindex([0, 1, 3])
        tm.assert_frame_equal(result, expected)

        # 使用 axis=0 参数进行索引，效果应该与直接使用整数列表相同
        result = df.reindex([0, 1, 3], axis=0)
        tm.assert_frame_equal(result, expected)

        # 使用 axis="index" 进行索引，效果应该与前面两种方式相同
        result = df.reindex([0, 1, 3], axis="index")
        tm.assert_frame_equal(result, expected)

    def test_reindex_positional_raises(self):
        # 测试位置参数错误引发的异常情况
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        msg = r"reindex\(\) takes from 1 to 2 positional arguments but 3 were given"
        # 传递了多余的位置参数，应当引发 TypeError 异常，异常信息为 msg
        with pytest.raises(TypeError, match=msg):
            df.reindex([0, 1], ["A", "B", "C"])

    def test_reindex_axis_style_raises(self):
        # 测试轴风格参数错误引发的异常情况
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # 尝试同时指定 axis 参数和其他参数，应当引发 TypeError 异常
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=["A"], axis=1)

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=["A"], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(columns=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], columns=[0, 1], axis="columns")

        with pytest.raises(TypeError, match="Cannot specify all"):
            df.reindex(labels=[0, 1], index=[0], columns=["A"])

        # 混合使用风格
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="index")

        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis="columns")

        # 处理重复值
        with pytest.raises(TypeError, match="multiple values"):
            df.reindex([0, 1], labels=[0, 1])
    # 定义单一命名索引器重新索引测试方法
    def test_reindex_single_named_indexer(self):
        # 解决问题12392：https://github.com/pandas-dev/pandas/issues/12392
        # 创建一个包含两列'A'和'B'的DataFrame对象
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})
        # 使用指定索引重新索引DataFrame，仅保留列'A'
        result = df.reindex([0, 1], columns=["A"])
        # 创建预期的DataFrame对象，仅包含列'A'
        expected = DataFrame({"A": [1, 2]})
        # 使用测试工具比较结果和预期DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)

    # API等效性重新索引测试方法
    def test_reindex_api_equivalence(self):
        # 解决问题12392：https://github.com/pandas-dev/pandas/issues/12392
        # 创建一个包含指定索引和列的DataFrame对象
        df = DataFrame(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            index=["a", "b", "c"],
            columns=["d", "e", "f"],
        )

        # 使用不同的索引方法重新索引DataFrame对象，并比较结果的等效性
        res1 = df.reindex(["b", "a"])
        res2 = df.reindex(index=["b", "a"])
        res3 = df.reindex(labels=["b", "a"])
        res4 = df.reindex(labels=["b", "a"], axis=0)
        res5 = df.reindex(["b", "a"], axis=0)
        for res in [res2, res3, res4, res5]:
            tm.assert_frame_equal(res1, res)

        # 使用不同的列名方法重新索引DataFrame对象，并比较结果的等效性
        res1 = df.reindex(columns=["e", "d"])
        res2 = df.reindex(["e", "d"], axis=1)
        res3 = df.reindex(labels=["e", "d"], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)

        # 使用同时指定索引和列名的方法重新索引DataFrame对象，并比较结果的等效性
        res1 = df.reindex(index=["b", "a"], columns=["e", "d"])
        res2 = df.reindex(columns=["e", "d"], index=["b", "a"])
        res3 = df.reindex(labels=["b", "a"], axis=0).reindex(labels=["e", "d"], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)

    # 布尔类型重新索引测试方法
    def test_reindex_boolean(self):
        # 创建一个布尔值类型的DataFrame对象
        frame = DataFrame(
            np.ones((10, 2), dtype=bool), index=np.arange(0, 20, 2), columns=[0, 2]
        )

        # 使用指定索引重新索引DataFrame对象，并进行数据类型和空值的检查
        reindexed = frame.reindex(np.arange(10))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[0][1])

        # 使用指定列名重新索引DataFrame对象，并进行数据类型和空值的检查
        reindexed = frame.reindex(columns=range(3))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[1]).all()

    # 对象类型重新索引测试方法
    def test_reindex_objects(self, float_string_frame):
        # 使用指定列名重新索引DataFrame对象，并检查列是否存在
        reindexed = float_string_frame.reindex(columns=["foo", "A", "B"])
        assert "foo" in reindexed

        # 使用指定列名重新索引DataFrame对象，并检查列是否不存在
        reindexed = float_string_frame.reindex(columns=["A", "B"])
        assert "foo" not in reindexed

    # 边界情况重新索引测试方法
    def test_reindex_corner(self, int_frame):
        # 创建一个Index对象作为索引
        index = Index(["a", "b", "c"])
        # 创建一个空的DataFrame对象，使用指定索引重新索引，并检查列名是否相等
        dm = DataFrame({}).reindex(index=[1, 2, 3])
        reindexed = dm.reindex(columns=index)
        # 使用测试工具比较重新索引后的列名与预期的索引对象是否相等
        tm.assert_index_equal(reindexed.columns, index)

        # 检查特定列的数据类型是否为np.float64
        smaller = int_frame.reindex(columns=["A", "B", "E"])
        assert smaller["E"].dtype == np.float64
    def test_reindex_with_nans(self):
        # 创建一个 DataFrame 对象，包含带有 NaN 值的数据
        df = DataFrame(
            [[1, 2], [3, 4], [np.nan, np.nan], [7, 8], [9, 10]],
            columns=["a", "b"],
            index=[100.0, 101.0, np.nan, 102.0, 103.0],
        )

        # 对 DataFrame 进行重新索引，选取指定索引位置的行
        result = df.reindex(index=[101.0, 102.0, 103.0])
        expected = df.iloc[[1, 3, 4]]
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 再次对 DataFrame 进行重新索引，这次只选取一个索引位置的行
        result = df.reindex(index=[103.0])
        expected = df.iloc[[4]]
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 再次对 DataFrame 进行重新索引，这次只选取一个索引位置的行
        result = df.reindex(index=[101.0])
        expected = df.iloc[[1]]
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_reindex_without_upcasting(self):
        # 创建一个全为 0 的 DataFrame，数据类型为 np.float32
        df = DataFrame(np.zeros((10, 10), dtype=np.float32))
        # 对 DataFrame 进行列的重新索引，使用 np.arange(5, 15)
        result = df.reindex(columns=np.arange(5, 15))
        # 断言重新索引后的所有列的数据类型为 np.float32
        assert result.dtypes.eq(np.float32).all()

    def test_reindex_multi(self):
        # 创建一个随机数填充的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))

        # 对 DataFrame 进行行和列的重新索引，同时使用 range(4)
        result = df.reindex(index=range(4), columns=range(4))
        expected = df.reindex(list(range(4))).reindex(columns=range(4))
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建一个随机整数填充的 DataFrame
        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))

        # 对 DataFrame 进行行和列的重新索引，同时使用 range(4)
        result = df.reindex(index=range(4), columns=range(4))
        expected = df.reindex(list(range(4))).reindex(columns=range(4))
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建一个随机整数填充的 DataFrame
        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))

        # 对 DataFrame 进行行和列的重新索引，同时使用 range(2)
        result = df.reindex(index=range(2), columns=range(2))
        expected = df.reindex(range(2)).reindex(columns=range(2))
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建一个复数随机数填充的 DataFrame，带有指定列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)) + 1j,
            columns=["a", "b", "c"],
        )

        # 对 DataFrame 进行行和列的重新索引，同时使用指定的行索引和列名
        result = df.reindex(index=[0, 1], columns=["a", "b"])
        expected = df.reindex([0, 1]).reindex(columns=["a", "b"])
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_reindex_multi_categorical_time(self):
        # 创建一个 MultiIndex，包含两个分类变量和日期范围
        midx = MultiIndex.from_product(
            [
                Categorical(["a", "b", "c"]),
                Categorical(date_range("2012-01-01", periods=3, freq="h")),
            ]
        )
        # 使用 MultiIndex 创建一个 DataFrame
        df = DataFrame({"a": range(len(midx))}, index=midx)
        # 从 df 中选择指定的行创建一个新的 DataFrame
        df2 = df.iloc[[0, 1, 2, 3, 4, 5, 6, 8]]

        # 对新的 DataFrame 进行重新索引，使用原始的 MultiIndex
        result = df2.reindex(midx)
        expected = DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, np.nan, 8]}, index=midx)
        # 断言重新索引后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_reindex_signature(self):
        # 获取 DataFrame.reindex 方法的签名信息
        sig = inspect.signature(DataFrame.reindex)
        # 提取签名中的所有参数名
        parameters = set(sig.parameters)
        # 断言参数名集合与预期相等
        assert parameters == {
            "self",
            "labels",
            "index",
            "columns",
            "axis",
            "limit",
            "copy",
            "level",
            "method",
            "fill_value",
            "tolerance",
        }
    def test_reindex_multiindex_ffill_added_rows(self):
        # 定义一个测试函数，用于验证在填充方法指定的情况下，即使有NaN值也能重新索引新增的行
        # 创建一个多级索引对象 mi，包含元组 ("a", "b") 和 ("d", "e")
        mi = MultiIndex.from_tuples([("a", "b"), ("d", "e")])
        # 创建一个数据框 df，包含二维数据，使用 mi 作为索引，列名为 ["x", "y"]
        df = DataFrame([[0, 7], [3, 4]], index=mi, columns=["x", "y"])
        # 创建另一个多级索引对象 mi2，包含元组 ("a", "b")、("d", "e") 和 ("h", "i")
        mi2 = MultiIndex.from_tuples([("a", "b"), ("d", "e"), ("h", "i")])
        # 对 df 进行重新索引，指定轴为行（axis=0），使用向前填充方法 ("ffill")
        result = df.reindex(mi2, axis=0, method="ffill")
        # 创建预期结果数据框 expected，包含与 mi2 相同的索引，列名为 ["x", "y"]，未填充的部分使用向前填充
        expected = DataFrame([[0, 7], [3, 4], [3, 4]], index=mi2, columns=["x", "y"])
        # 使用测试框架的断言方法验证结果与预期是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "pad", "tolerance": timedelta(seconds=9)},
            {"method": "backfill", "tolerance": timedelta(seconds=9)},
            {"method": "nearest"},
            {"method": None},
        ],
    )
    def test_reindex_empty_frame(self, kwargs):
        # 定义一个测试函数，用于验证在重新索引空数据框时的不同方法
        # 创建一个日期范围 idx，从 "2020" 开始，频率为每30秒，共3个周期
        idx = date_range(start="2020", freq="30s", periods=3)
        # 创建一个空的数据框 df，行索引为空的时间索引，列名为 ["a"]
        df = DataFrame([], index=Index([], name="time"), columns=["a"])
        # 对 df 进行重新索引，使用传入的参数 kwargs
        result = df.reindex(idx, **kwargs)
        # 创建预期结果数据框 expected，包含一列 "a"，值为三个 NaN 值，使用对象数据类型
        expected = DataFrame({"a": [np.nan] * 3}, index=idx, dtype=object)
        # 使用测试框架的断言方法验证结果与预期是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("src_idx", [Index, CategoricalIndex])
    @pytest.mark.parametrize(
        "cat_idx",
        [
            # 无重复值
            Index([]),
            CategoricalIndex([]),
            Index(["A", "B"]),
            CategoricalIndex(["A", "B"]),
            # 存在重复值，测试 GH#38906
            Index(["A", "A"]),
            CategoricalIndex(["A", "A"]),
        ],
    )
    def test_reindex_empty(self, src_idx, cat_idx):
        # 定义一个测试函数，用于验证在空索引情况下的重新索引
        # 创建一个空的数据框 df，行索引为一个空的 src_idx([])，列索引为 ["K"]，使用浮点数数据类型
        df = DataFrame(columns=src_idx([]), index=["K"], dtype="f8")

        # 对 df 进行重新索引，列索引为 cat_idx
        result = df.reindex(columns=cat_idx)
        # 创建预期结果数据框 expected，行索引为 ["K"]，列索引为 cat_idx，使用浮点数数据类型
        expected = DataFrame(index=["K"], columns=cat_idx, dtype="f8")
        # 使用测试框架的断言方法验证结果与预期是否相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，用于将日期时间索引转换为对象类型，接受参数 dtype 作为数据类型
    def test_reindex_datetimelike_to_object(self, dtype):
        # GH#39755 不将 dt64/td64 强制转换为整数
        # 创建一个多级索引 mi，包含字符列表 "ABCDE" 与范围为 0 到 1 的整数
        mi = MultiIndex.from_product([list("ABCDE"), range(2)])

        # 创建一个日期范围 dti，从 "2016-01-01" 开始，长度为 10
        dti = date_range("2016-01-01", periods=10)
        # 初始化一个 NaT 值，根据参数 dtype 决定是 np.datetime64("NaT", "ns") 还是 np.timedelta64("NaT", "ns")
        fv = np.timedelta64("NaT", "ns")
        if dtype == "m8[ns]":
            # 如果 dtype 是 "m8[ns]"，将日期时间索引 dti 转换为时间差，并设置 fv 为 np.datetime64("NaT", "ns")
            dti = dti - dti[0]
            fv = np.datetime64("NaT", "ns")

        # 创建一个 Series 对象 ser，将 dti 设置为其数据，mi 设置为其索引
        ser = Series(dti, index=mi)
        # 将 ser 中每隔三个位置设置为 pd.NaT
        ser[::3] = pd.NaT

        # 将 Series 对象 ser 转换为 DataFrame 对象 df
        df = ser.unstack()

        # 创建新的索引 index，为 df 的索引加上一个新值 1
        index = df.index.append(Index([1]))
        # 创建新的列名 columns，为 df 的列加上一个新的列名 "foo"
        columns = df.columns.append(Index(["foo"]))

        # 对 DataFrame 对象 df 进行重新索引，设置 fill_value 为 fv
        res = df.reindex(index=index, columns=columns, fill_value=fv)

        # 创建期望的 DataFrame 对象 expected
        expected = DataFrame(
            {
                0: df[0].tolist() + [fv],
                1: df[1].tolist() + [fv],
                "foo": np.array(["NaT"] * 6, dtype=fv.dtype),
            },
            index=index,
        )
        # 断言 res 的指定列的数据类型是否都为对象类型
        assert (res.dtypes[[0, 1]] == object).all()
        # 断言 res 的第一行第一列是否为 pd.NaT
        assert res.iloc[0, 0] is pd.NaT
        # 断言 res 的最后一行第一列是否为 fv
        assert res.iloc[-1, 0] is fv
        # 断言 res 的最后一行第二列是否为 fv
        assert res.iloc[-1, 1] is fv
        # 使用 tm.assert_frame_equal 函数断言 res 与 expected 是否相等
        tm.assert_frame_equal(res, expected)

    # 使用 pytest.mark.parametrize 装饰器参数化 klass 和 data，用于测试不是类别索引的重新索引
    @pytest.mark.parametrize("klass", [Index, CategoricalIndex])
    @pytest.mark.parametrize("data", ["A", "B"])
    def test_reindex_not_category(self, klass, data):
        # GH#28690
        # 创建一个空的 DataFrame 对象 df，设置索引为空的 CategoricalIndex，且类别为 ["A"]
        df = DataFrame(index=CategoricalIndex([], categories=["A"]))
        # 创建一个新的索引 idx，根据参数 klass 和 data 创建相应的索引对象
        idx = klass([data])
        # 对 DataFrame 对象 df 进行重新索引，设置索引为 idx
        result = df.reindex(index=idx)
        # 创建期望的 DataFrame 对象 expected，设置索引为 idx
        expected = DataFrame(index=idx)
        # 使用 tm.assert_frame_equal 函数断言 result 与 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试无效的重新索引方法
    def test_invalid_method(self):
        # 创建一个 DataFrame 对象 df，包含列 "A"，数据为 [1, np.nan, 2]
        df = DataFrame({"A": [1, np.nan, 2]})

        # 设置异常消息的字符串 msg
        msg = "Invalid fill method"
        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证其错误消息是否匹配 msg
        with pytest.raises(ValueError, match=msg):
            # 调用 df 的重新索引方法，使用不支持的方法名 "asfreq"
            df.reindex([1, 0, 2], method="asfreq")
```