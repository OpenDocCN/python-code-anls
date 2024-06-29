# `D:\src\scipysrc\pandas\pandas\tests\reductions\test_stat_reductions.py`

```
"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""

# 导入 inspect 模块，用于获取对象信息
import inspect

# 导入 numpy 库并重命名为 np
import numpy as np

# 导入 pytest 库
import pytest

# 导入 pandas 库并重命名为 pd
import pandas as pd

# 从 pandas 中导入特定模块和函数
from pandas import (
    DataFrame,
    Series,
    date_range,
)

# 导入 pandas 内部测试模块
import pandas._testing as tm


# 定义日期时间统计缩减测试类
class TestDatetimeLikeStatReductions:
    # 测试方法：测试 datetime64 类型的均值计算
    def test_dt64_mean(self, tz_naive_fixture, index_or_series_or_array):
        # 获取时区无关的 fixture
        tz = tz_naive_fixture

        # 创建日期范围对象，设置时区
        dti = date_range("2001-01-01", periods=11, tz=tz)

        # 打乱顺序以确保不是单调递增
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        # 从日期时间索引中获取内部数据
        dtarr = dti._data

        # 使用 index_or_series_or_array 函数处理内部数据，生成对象
        obj = index_or_series_or_array(dtarr)

        # 断言对象的均值与预期结果相等
        assert obj.mean() == pd.Timestamp("2001-01-06", tz=tz)

        # 断言对象的均值（不包括 NaN 值）与预期结果相等
        assert obj.mean(skipna=False) == pd.Timestamp("2001-01-06", tz=tz)

        # 将倒数第二个元素设为 NaT（Not a Time）
        dtarr[-2] = pd.NaT

        # 使用 index_or_series_or_array 函数处理更新后的内部数据，生成对象
        obj = index_or_series_or_array(dtarr)

        # 断言对象的均值与预期结果相等
        assert obj.mean() == pd.Timestamp("2001-01-06 07:12:00", tz=tz)

        # 断言对象的均值（不包括 NaN 值）为 NaT
        assert obj.mean(skipna=False) is pd.NaT

    # 测试方法：测试 period 类型的均值计算
    @pytest.mark.parametrize("freq", ["s", "h", "D", "W", "B"])
    def test_period_mean(self, index_or_series_or_array, freq):
        # 创建日期范围对象
        dti = date_range("2001-01-01", periods=11)

        # 打乱顺序以确保不是单调递增
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        # 根据频率将日期时间索引转换为 period 类型
        warn = FutureWarning if freq == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)

        # 使用 index_or_series_or_array 函数处理 period 类型数据，生成对象
        obj = index_or_series_or_array(parr)

        # 断言调用均值方法时抛出 TypeError 异常
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()

        # 断言调用均值方法（包括 NaN 值）时抛出 TypeError 异常
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

        # 将倒数第二个元素设为 NaT
        parr[-2] = pd.NaT

        # 断言调用均值方法时抛出 TypeError 异常
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()

        # 断言调用均值方法（包括 NaN 值）时抛出 TypeError 异常
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

    # 测试方法：测试 timedelta64 类型的均值计算
    def test_td64_mean(self, index_or_series_or_array):
        # 创建 timedelta64 类型的索引
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], "m8[D]")
        tdi = pd.TimedeltaIndex(m8values).as_unit("ns")

        # 获取内部数据
        tdarr = tdi._data

        # 使用 index_or_series_or_array 函数处理 timedelta64 类型数据，生成对象
        obj = index_or_series_or_array(tdarr, copy=False)

        # 计算对象的均值
        result = obj.mean()

        # 计算期望的均值
        expected = np.array(tdarr).mean()

        # 断言计算结果与期望结果相等
        assert result == expected

        # 将第一个元素设为 NaT
        tdarr[0] = pd.NaT

        # 断言调用均值方法（不包括 NaN 值）返回 NaT
        assert obj.mean(skipna=False) is pd.NaT

        # 计算调用均值方法（包括 NaN 值）的结果
        result2 = obj.mean(skipna=True)

        # 断言计算结果与期望结果相等，精度为微秒级
        assert result2 == tdi[1:].mean()

        # 断言精确相等性失败不超过 1 纳秒
        assert result2.round("us") == (result * 11.0 / 10).round("us")


# 测试类：Series 统计缩减
class TestSeriesStatReductions:
    # 注释：TestSeriesStatReductions 类名表明这些测试从一个专门针对 Series 的测试文件中移动过来，并非长期意味着仅适用于 Series
    def _check_stat_op(
        self, name, alternate, string_series_, check_objects=False, check_allna=False
    ):
        with pd.option_context("use_bottleneck", False):
            # 获取对应操作名的函数
            f = getattr(Series, name)

            # 向序列中添加一些 NaN 值
            string_series_[5:15] = np.nan

            # 对于日期类型，mean、idxmax、idxmin、min 和 max 是有效的操作
            if name not in ["max", "min", "mean", "median", "std"]:
                # 创建一个日期序列
                ds = Series(date_range("1/1/2001", periods=10))
                msg = f"does not support operation '{name}'"
                # 确保对不支持的操作抛出 TypeError 异常
                with pytest.raises(TypeError, match=msg):
                    f(ds)

            # 测试 skipna 参数的效果
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # 检查结果是否正确
            nona = string_series_.dropna()
            # 使用测试框架的函数检查结果与预期值的接近程度
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            # 创建全部为 NaN 的序列
            allna = string_series_ * np.nan

            if check_allna:
                # 如果需要检查全部为 NaN 的情况，则确认结果是 NaN
                assert np.isnan(f(allna))

            # 对于包含 None 的对象类型序列，确保操作正常运行
            s = Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888 的特定测试情况
            items = [0]
            items.extend(range(2**40, 2**40 + 1000))
            s = Series(items, dtype="int64")
            # 使用测试框架的函数检查结果与预期值的接近程度
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # 检查日期范围
            if check_objects:
                s = Series(pd.bdate_range("1/1/2000", periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # 对字符串类型数据进行测试
            if name not in ["sum", "min", "max"]:
                # 确保对不支持的操作抛出 TypeError 异常
                with pytest.raises(TypeError, match=None):
                    f(Series(list("abc")))

            # 测试无效的轴参数
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)

            if "numeric_only" in inspect.getfullargspec(f).args:
                # 只有索引是字符串类型，数据类型是浮点型
                f(string_series_, numeric_only=True)

    def test_sum(self):
        # 创建一个浮点型的序列用于测试
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("sum", np.sum, string_series, check_allna=False)

    def test_mean(self):
        # 创建一个浮点型的序列用于测试
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("mean", np.mean, string_series)

    def test_median(self):
        # 创建一个浮点型的序列用于测试
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("median", np.median, string_series)

        # 测试整数型数据，预期会失败的情况
        int_ts = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())
    # 测试用例：计算 Series 对象的乘积
    def test_prod(self):
        # 创建一个包含 0 到 19 的浮点数 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")
        # 调用 _check_stat_op 方法，检查 "prod" 操作的结果是否正确
        self._check_stat_op("prod", np.prod, string_series)

    # 测试用例：计算 Series 对象的最小值
    def test_min(self):
        # 创建一个包含 0 到 19 的浮点数 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")
        # 调用 _check_stat_op 方法，检查 "min" 操作的结果是否正确，包括对象检查
        self._check_stat_op("min", np.min, string_series, check_objects=True)

    # 测试用例：计算 Series 对象的最大值
    def test_max(self):
        # 创建一个包含 0 到 19 的浮点数 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")
        # 调用 _check_stat_op 方法，检查 "max" 操作的结果是否正确，包括对象检查
        self._check_stat_op("max", np.max, string_series, check_objects=True)

    # 测试用例：计算 Series 对象和日期 Series 对象的标准差、方差及标准误差
    def test_var_std(self):
        # 创建一个包含 0 到 19 的浮点数 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")
        
        # 创建一个包含 0 到 9 的浮点数 Series 对象，带有日期索引
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # 定义一个 lambda 函数 alt，计算标准差（无偏估计）
        alt = lambda x: np.std(x, ddof=1)
        # 调用 _check_stat_op 方法，检查 "std" 操作的结果是否正确
        self._check_stat_op("std", alt, string_series)

        # 重新定义 lambda 函数 alt，计算方差（无偏估计）
        alt = lambda x: np.var(x, ddof=1)
        # 调用 _check_stat_op 方法，检查 "var" 操作的结果是否正确
        self._check_stat_op("var", alt, string_series)

        # 计算日期 Series 对象的标准差（有偏估计）
        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        # 使用 tm.assert_almost_equal 检查计算结果与预期值的近似程度
        tm.assert_almost_equal(result, expected)

        # 计算日期 Series 对象的方差（有偏估计）
        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        # 使用 tm.assert_almost_equal 检查计算结果与预期值的近似程度
        tm.assert_almost_equal(result, expected)

        # 对仅包含一个元素的日期 Series 对象进行方差计算（无偏估计），预期结果应为 NaN
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        # 使用 assert 语句检查结果是否为 NaN
        assert pd.isna(result)

        # 对仅包含一个元素的日期 Series 对象进行标准差计算（无偏估计），预期结果应为 NaN
        result = s.std(ddof=1)
        # 使用 assert 语句检查结果是否为 NaN
        assert pd.isna(result)

    # 测试用例：计算 Series 对象和日期 Series 对象的标准误差
    def test_sem(self):
        # 创建一个包含 0 到 19 的浮点数 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")
        
        # 创建一个包含 0 到 9 的浮点数 Series 对象，带有日期索引
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # 定义一个 lambda 函数 alt，计算标准差（无偏估计）除以 sqrt(len(x)) 的值，即标准误差
        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        # 调用 _check_stat_op 方法，检查 "sem" 操作的结果是否正确
        self._check_stat_op("sem", alt, string_series)

        # 计算日期 Series 对象的标准误差（有偏估计）
        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(
            len(datetime_series.values)
        )
        # 使用 tm.assert_almost_equal 检查计算结果与预期值的近似程度
        tm.assert_almost_equal(result, expected)

        # 对仅包含一个元素的日期 Series 对象进行标准误差计算（无偏估计），预期结果应为 NaN
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        # 使用 assert 语句检查结果是否为 NaN
        assert pd.isna(result)
    # 定义一个测试方法，用于测试 skew（偏度）统计函数
    def test_skew(self):
        # 导入 scipy.stats 库，如果导入失败则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 创建一个包含 0 到 19 的浮点数的 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")

        # 定义一个 lambda 函数 alt，用于计算非偏态 skew
        alt = lambda x: sp_stats.skew(x, bias=False)
        
        # 调用自定义的 _check_stat_op 方法，测试 skew 统计操作
        self._check_stat_op("skew", alt, string_series)

        # 测试极端情况，当数据点少于 3 个时，skew() 返回 NaN
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                # 断言当数据点少于 min_N 时，skew() 返回 NaN
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                # 断言当数据点至少为 min_N 时，skew() 返回 0
                assert 0 == s.skew()
                assert isinstance(s.skew(), np.float64)  # GH53482
                assert (df.skew() == 0).all()

    # 定义一个测试方法，用于测试 kurtosis（峰度）统计函数
    def test_kurt(self):
        # 导入 scipy.stats 库，如果导入失败则跳过测试
        sp_stats = pytest.importorskip("scipy.stats")

        # 创建一个包含 0 到 19 的浮点数的 Series 对象
        string_series = Series(range(20), dtype=np.float64, name="series")

        # 定义一个 lambda 函数 alt，用于计算非偏态 kurtosis
        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        
        # 调用自定义的 _check_stat_op 方法，测试 kurtosis 统计操作
        self._check_stat_op("kurt", alt, string_series)

    # 定义一个测试方法，用于测试 kurtosis（峰度）的极端情况
    def test_kurt_corner(self):
        # 测试极端情况，当数据点少于 4 个时，kurt() 返回 NaN
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                # 断言当数据点少于 min_N 时，kurt() 返回 NaN
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                # 断言当数据点至少为 min_N 时，kurt() 返回 0
                assert 0 == s.kurt()
                assert isinstance(s.kurt(), np.float64)  # GH53482
                assert (df.kurt() == 0).all()
```