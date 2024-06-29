# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_quantile.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas.core.dtypes.common import is_integer  # 导入 Pandas 的数据类型相关模块

import pandas as pd  # 导入 Pandas 库，用于数据处理
from pandas import (  # 从 Pandas 导入多个模块
    Index,  # 索引相关
    Series,  # 序列相关
)
import pandas._testing as tm  # 导入 Pandas 的测试模块
from pandas.core.indexes.datetimes import Timestamp  # 导入 Pandas 的时间戳相关模块


class TestSeriesQuantile:  # 定义测试类 TestSeriesQuantile
    def test_quantile(self, datetime_series):  # 定义测试量化函数
        q = datetime_series.quantile(0.1)  # 计算时间序列的第10%分位数
        assert q == np.percentile(datetime_series.dropna(), 10)  # 使用 NumPy 计算时间序列的第10%分位数并断言相等

        q = datetime_series.quantile(0.9)  # 计算时间序列的第90%分位数
        assert q == np.percentile(datetime_series.dropna(), 90)  # 使用 NumPy 计算时间序列的第90%分位数并断言相等

        # 对象类型
        q = Series(datetime_series, dtype=object).quantile(0.9)  # 将时间序列转换为对象类型后计算第90%分位数
        assert q == np.percentile(datetime_series.dropna(), 90)  # 使用 NumPy 计算时间序列的第90%分位数并断言相等

        # datetime64[ns] 类型
        dts = datetime_series.index.to_series()  # 将时间序列索引转换为序列
        q = dts.quantile(0.2)  # 计算序列的第20%分位数
        assert q == Timestamp("2000-01-10 19:12:00")  # 断言计算结果等于指定的时间戳

        # timedelta64[ns] 类型
        tds = dts.diff()  # 计算序列之间的差异
        q = tds.quantile(0.25)  # 计算差异序列的第25%分位数
        assert q == pd.to_timedelta("24:00:00")  # 断言计算结果等于指定的时间增量

        # GH7661
        result = Series([np.timedelta64("NaT")]).sum()  # 对包含 NaT 的序列进行求和
        assert result == pd.Timedelta(0)  # 断言结果等于零时间增量

        msg = "percentiles should all be in the interval \\[0, 1\\]"  # 定义错误消息
        for invalid in [-1, 2, [0.5, -1], [0.5, 2]]:  # 遍历无效的百分位数列表
            with pytest.raises(ValueError, match=msg):  # 断言引发 ValueError 异常，并匹配错误消息
                datetime_series.quantile(invalid)  # 计算时间序列的无效百分位数

        s = Series(np.random.default_rng(2).standard_normal(100))  # 创建包含随机标准正态分布的序列
        percentile_array = [-0.5, 0.25, 1.5]  # 定义百分位数数组
        with pytest.raises(ValueError, match=msg):  # 断言引发 ValueError 异常，并匹配错误消息
            s.quantile(percentile_array)  # 计算随机序列的百分位数

    def test_quantile_multi(self, datetime_series, unit):  # 定义多重量化测试函数
        datetime_series.index = datetime_series.index.as_unit(unit)  # 将时间序列索引转换为指定单位
        qs = [0.1, 0.9]  # 定义百分位数列表
        result = datetime_series.quantile(qs)  # 计算时间序列的指定百分位数
        expected = Series(  # 创建期望的结果序列
            [
                np.percentile(datetime_series.dropna(), 10),  # 第10%分位数
                np.percentile(datetime_series.dropna(), 90),  # 第90%分位数
            ],
            index=qs,  # 设置索引为百分位数列表
            name=datetime_series.name,  # 设置名称为时间序列的名称
        )
        tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块断言结果序列与期望序列相等

        dts = datetime_series.index.to_series()  # 将时间序列索引转换为序列
        dts.name = "xxx"  # 设置序列名称为 "xxx"
        result = dts.quantile((0.2, 0.2))  # 计算序列的第20%分位数
        expected = Series(  # 创建期望的结果序列
            [Timestamp("2000-01-10 19:12:00"), Timestamp("2000-01-10 19:12:00")],  # 指定时间戳
            index=[0.2, 0.2],  # 设置索引为重复的百分位数
            name="xxx",  # 设置名称为 "xxx"
            dtype=f"M8[{unit}]",  # 设置数据类型为指定单位的日期时间类型
        )
        tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块断言结果序列与期望序列相等

        result = datetime_series.quantile([])  # 计算空百分位数
        expected = Series(  # 创建期望的结果序列
            [], name=datetime_series.name, index=Index([], dtype=float), dtype="float64"  # 设置空序列属性
        )
        tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块断言结果序列与期望序列相等
    # 测试插值和无插值关键字的量化
    def test_quantile_interpolation(self, datetime_series):
        # see gh-10174
        # 使用线性插值（默认情况）
        q = datetime_series.quantile(0.1, interpolation="linear")
        # 断言计算出的分位数与 numpy 计算的百分位数相等
        assert q == np.percentile(datetime_series.dropna(), 10)
        # 不指定插值方式的量化
        q1 = datetime_series.quantile(0.1)
        # 断言计算出的分位数与 numpy 计算的百分位数相等
        assert q1 == np.percentile(datetime_series.dropna(), 10)
        # 检查有无插值方式和无插值方式的量化结果是否一致
        assert q == q1

    # 测试插值类型和数据类型的量化
    def test_quantile_interpolation_dtype(self):
        # GH #10174
        # 使用 lower 插值方式的量化
        q = Series([1, 3, 4]).quantile(0.5, interpolation="lower")
        # 断言计算出的分位数与 numpy 计算的百分位数相等
        assert q == np.percentile(np.array([1, 3, 4]), 50)
        # 断言 q 是整数
        assert is_integer(q)

        # 使用 higher 插值方式的量化
        q = Series([1, 3, 4]).quantile(0.5, interpolation="higher")
        # 断言计算出的分位数与 numpy 计算的百分位数相等
        assert q == np.percentile(np.array([1, 3, 4]), 50)
        # 断言 q 是整数
        assert is_integer(q)

    # 测试含有 NaN 的量化
    def test_quantile_nan(self):
        # GH 13098
        # 创建含有 NaN 的 Series
        ser = Series([1, 2, 3, 4, np.nan])
        # 计算第 50% 分位数
        result = ser.quantile(0.5)
        # 预期的结果是 2.5
        expected = 2.5
        # 断言计算结果与预期结果相等
        assert result == expected

        # 所有值都是 NaN 或空的情况
        s1 = Series([], dtype=object)
        cases = [s1, Series([np.nan, np.nan])]

        # 遍历所有测试案例
        for ser in cases:
            # 计算第 50% 分位数
            res = ser.quantile(0.5)
            # 断言结果是 NaN
            assert np.isnan(res)

            # 计算多个分位数
            res = ser.quantile([0.5])
            # 断言结果是包含 NaN 的 Series
            tm.assert_series_equal(res, Series([np.nan], index=[0.5]))

            # 计算多个分位数
            res = ser.quantile([0.2, 0.3])
            # 断言结果是包含 NaN 的 Series
            tm.assert_series_equal(res, Series([np.nan, np.nan], index=[0.2, 0.3]))

    # 参数化测试箱线图分位数
    @pytest.mark.parametrize(
        "case",
        [
            [
                Timestamp("2011-01-01"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-03"),
            ],
            [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-03", tz="US/Eastern"),
            ],
            [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            # NaT
            [
                Timestamp("2011-01-01"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-03"),
                pd.NaT,
            ],
            [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-03", tz="US/Eastern"),
                pd.NaT,
            ],
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Timedelta("3 days"),
                pd.NaT,
            ],
        ],
    )
    # 测试箱线图的分位数计算
    def test_quantile_box(self, case):
        # 创建 Series，用于测试箱线图分位数
        ser = Series(case, name="XXX")
        # 计算第 50% 分位数
        res = ser.quantile(0.5)
        # 断言结果与 case[1] 相等
        assert res == case[1]

        # 计算多个分位数
        res = ser.quantile([0.5])
        # 构建预期的 Series 结果
        exp = Series([case[1]], index=[0.5], name="XXX")
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(res, exp)
    def test_datetime_timedelta_quantiles(self):
        # 测试空序列的日期时间和时间增量的分位数计算
        assert pd.isna(Series([], dtype="M8[ns]").quantile(0.5))
        assert pd.isna(Series([], dtype="m8[ns]").quantile(0.5))

    def test_quantile_nat(self):
        # 测试包含NaT（Not a Time）值的序列的分位数计算
        res = Series([pd.NaT, pd.NaT]).quantile(0.5)
        assert res is pd.NaT

        # 测试包含NaT值的序列，使用列表形式的分位数计算
        res = Series([pd.NaT, pd.NaT]).quantile([0.5])
        tm.assert_series_equal(res, Series([pd.NaT], index=[0.5]))

    @pytest.mark.parametrize(
        "values, dtype",
        [([0, 0, 0, 1, 2, 3], "Sparse[int]"), ([0.0, None, 1.0, 2.0], "Sparse[float]")],
    )
    def test_quantile_sparse(self, values, dtype):
        # 测试稀疏类型数据的分位数计算
        ser = Series(values, dtype=dtype)
        result = ser.quantile([0.5])
        expected = Series(np.asarray(ser)).quantile([0.5]).astype("Sparse[float]")
        tm.assert_series_equal(result, expected)

    def test_quantile_empty_float64(self):
        # 测试空的float64类型序列的分位数计算
        ser = Series([], dtype="float64")

        res = ser.quantile(0.5)
        assert np.isnan(res)

        res = ser.quantile([0.5])
        exp = Series([np.nan], index=[0.5])
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_int64(self):
        # 测试空的int64类型序列的分位数计算
        ser = Series([], dtype="int64")

        res = ser.quantile(0.5)
        assert np.isnan(res)

        res = ser.quantile([0.5])
        exp = Series([np.nan], index=[0.5])
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_dt64(self):
        # 测试空的datetime64[ns]类型序列的分位数计算
        ser = Series([], dtype="datetime64[ns]")

        res = ser.quantile(0.5)
        assert res is pd.NaT

        res = ser.quantile([0.5])
        exp = Series([pd.NaT], index=[0.5], dtype=ser.dtype)
        tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize("dtype", [int, float, "Int64"])
    def test_quantile_dtypes(self, dtype):
        # 测试不同数据类型（int, float, Int64）的序列的分位数计算
        result = Series([1, 2, 3], dtype=dtype).quantile(np.arange(0, 1, 0.25))
        expected = Series(np.arange(1, 3, 0.5), index=np.arange(0, 1, 0.25))
        if dtype == "Int64":
            expected = expected.astype("Float64")
        tm.assert_series_equal(result, expected)

    def test_quantile_all_na(self, any_int_ea_dtype):
        # 测试包含全部NA值的序列的分位数计算
        ser = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        with tm.assert_produces_warning(None):
            result = ser.quantile([0.1, 0.5])
        expected = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype, index=[0.1, 0.5])
        tm.assert_series_equal(result, expected)

    def test_quantile_dtype_size(self, any_int_ea_dtype):
        # 测试包含NA值和非NA值的序列的分位数计算
        ser = Series([pd.NA, pd.NA, 1], dtype=any_int_ea_dtype)
        result = ser.quantile([0.1, 0.5])
        expected = Series([1, 1], dtype=any_int_ea_dtype, index=[0.1, 0.5])
        tm.assert_series_equal(result, expected)
```