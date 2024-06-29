# `D:\src\scipysrc\pandas\pandas\tests\series\test_cumulative.py`

```
"""
Tests for Series cumulative operations.

See also
--------
tests.frame.test_cumulative
"""

# 导入必要的库
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# 定义可用的累积方法和对应的函数
methods = {
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "cummin": np.minimum.accumulate,
    "cummax": np.maximum.accumulate,
}

# 测试类，用于测试 Series 的累积操作
class TestSeriesCumulativeOps:
    
    # 参数化测试函数，测试 np.cumsum 和 np.cumprod 方法
    @pytest.mark.parametrize("func", [np.cumsum, np.cumprod])
    def test_datetime_series(self, datetime_series, func):
        # 断言 numpy 数组相等，检查数据类型
        tm.assert_numpy_array_equal(
            func(datetime_series).values,
            func(np.array(datetime_series)),
            check_dtype=True,
        )

        # 在包含缺失值的情况下进行测试
        ts = datetime_series.copy()
        ts[::2] = np.nan

        result = func(ts)[1::2]
        expected = func(np.array(ts.dropna()))

        tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)

    # 参数化测试函数，测试 cummin 和 cummax 方法
    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, datetime_series, method):
        # 获取对应的 numpy ufunc 函数
        ufunc = methods[method]

        # 执行 cummin 或 cummax 方法并断言结果与预期相等
        result = getattr(datetime_series, method)().values
        expected = ufunc(np.array(datetime_series))

        tm.assert_numpy_array_equal(result, expected)

        # 在包含缺失值的情况下进行测试
        ts = datetime_series.copy()
        ts[::2] = np.nan
        result = getattr(ts, method)()[1::2]
        expected = ufunc(ts.dropna())

        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    # 参数化测试函数，测试 cummin 和 cummax 方法对于不同的时间序列
    @pytest.mark.parametrize(
        "ts",
        [
            pd.Timedelta(0),
            pd.Timestamp("1999-12-31"),
            pd.Timestamp("1999-12-31").tz_localize("US/Pacific"),
        ],
    )
    @pytest.mark.parametrize(
        "method, skipna, exp_tdi",
        [
            ["cummax", True, ["NaT", "2 days", "NaT", "2 days", "NaT", "3 days"]],
            ["cummin", True, ["NaT", "2 days", "NaT", "1 days", "NaT", "1 days"]],
            [
                "cummax",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
            [
                "cummin",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
        ],
    )
    def test_cummin_cummax_datetimelike(self, ts, method, skipna, exp_tdi):
        # 生成时间增量序列，并将其转换为 Series
        tdi = pd.to_timedelta(["NaT", "2 days", "NaT", "1 days", "NaT", "3 days"])
        ser = pd.Series(tdi + ts)

        # 生成预期的时间增量序列，并将其转换为 Series
        exp_tdi = pd.to_timedelta(exp_tdi)
        expected = pd.Series(exp_tdi + ts)

        # 执行 cummin 或 cummax 方法，并断言结果与预期相等
        result = getattr(ser, method)(skipna=skipna)
        tm.assert_series_equal(expected, result)
    def test_cumsum_datetimelike(self):
        # 测试累积求和对于类日期时间的处理
        # 创建包含类日期时间数据的DataFrame
        df = pd.DataFrame(
            [
                [pd.Timedelta(0), pd.Timedelta(days=1)],
                [pd.Timedelta(days=2), pd.NaT],
                [pd.Timedelta(hours=-6), pd.Timedelta(hours=12)],
            ]
        )
        # 对DataFrame进行累积求和操作
        result = df.cumsum()
        # 期望的累积求和结果
        expected = pd.DataFrame(
            [
                [pd.Timedelta(0), pd.Timedelta(days=1)],
                [pd.Timedelta(days=2), pd.NaT],
                [pd.Timedelta(days=1, hours=18), pd.Timedelta(days=1, hours=12)],
            ]
        )
        # 使用测试工具比较结果和期望值
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "func, exp",
        [
            ("cummin", "2012-1-1"),
            ("cummax", "2012-1-2"),
        ],
    )
    def test_cummin_cummax_period(self, func, exp):
        # 测试周期数据的累积最小和累积最大方法
        # 创建包含周期数据的Series对象
        ser = pd.Series(
            [pd.Period("2012-1-1", freq="D"), pd.NaT, pd.Period("2012-1-2", freq="D")]
        )
        # 执行累积最小或累积最大方法，并获取结果
        result = getattr(ser, func)(skipna=False)
        # 期望的结果
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, pd.NaT])
        # 使用测试工具比较结果和期望值
        tm.assert_series_equal(result, expected)

        # 再次执行累积最小或累积最大方法，但这次跳过缺失值
        result = getattr(ser, func)(skipna=True)
        # 期望的结果是具体的日期周期
        exp = pd.Period(exp, freq="D")
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, exp])
        # 使用测试工具比较结果和期望值
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "arg",
        [
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, False, False],
        ],
    )
    @pytest.mark.parametrize(
        "func", [lambda x: x, lambda x: ~x], ids=["identity", "inverse"]
    )
    @pytest.mark.parametrize("method", methods.keys())
    def test_cummethods_bool(self, arg, func, method):
        # 测试布尔类型的累积方法
        # 根据输入参数创建Series对象
        ser = func(pd.Series(arg))
        # 获取对应的NumPy方法
        ufunc = methods[method]

        # 用NumPy方法处理Series的值，得到期望的结果
        exp_vals = ufunc(ser.values)
        expected = pd.Series(exp_vals)

        # 执行Series对象的累积方法
        result = getattr(ser, method)()

        # 使用测试工具比较结果和期望值
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ["cumsum", pd.Series([0, 1, np.nan, 1], dtype=object)],
            ["cumprod", pd.Series([False, 0, np.nan, 0])],
            ["cummin", pd.Series([False, False, np.nan, False])],
            ["cummax", pd.Series([False, True, np.nan, True])],
        ],
    )
    def test_cummethods_bool_in_object_dtype(self, method, expected):
        # 测试对象类型为布尔型的累积方法
        # 创建包含布尔值的Series对象
        ser = pd.Series([False, True, np.nan, False])
        # 执行指定的累积方法
        result = getattr(ser, method)()
        # 使用测试工具比较结果和期望值
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, order",
        [
            ["cummax", "abc"],
            ["cummin", "cba"],
        ],
    )
    # 测试在有序分类数据上应用累积最大和最小值方法
    def test_cummax_cummin_on_ordered_categorical(self, method, order):
        # GH#52335
        # 创建有序分类数据类型，使用给定的顺序列表
        cat = pd.CategoricalDtype(list(order), ordered=True)
        # 创建包含指定数据和分类类型的序列
        ser = pd.Series(
            list("ababcab"),
            dtype=cat,
        )
        # 调用指定的累积方法（cummax或cummin）
        result = getattr(ser, method)()
        # 创建期望的结果序列，使用相同的分类类型
        expected = pd.Series(
            list("abbbccc"),
            dtype=cat,
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "skip, exp",
        [
            [True, ["a", np.nan, "b", "b", "c"]],
            [False, ["a", np.nan, np.nan, np.nan, np.nan]],
        ],
    )
    @pytest.mark.parametrize(
        "method, order",
        [
            ["cummax", "abc"],
            ["cummin", "cba"],
        ],
    )
    # 测试在有序分类数据上应用累积最大和最小值方法，包括处理NaN值
    def test_cummax_cummin_ordered_categorical_nan(self, skip, exp, method, order):
        # GH#52335
        # 创建有序分类数据类型，使用给定的顺序列表
        cat = pd.CategoricalDtype(list(order), ordered=True)
        # 创建包含指定数据和分类类型的序列，包括NaN值
        ser = pd.Series(
            ["a", np.nan, "b", "a", "c"],
            dtype=cat,
        )
        # 调用指定的累积方法（cummax或cummin），根据skip参数处理NaN值
        result = getattr(ser, method)(skipna=skip)
        # 创建期望的结果序列，使用相同的分类类型
        expected = pd.Series(
            exp,
            dtype=cat,
        )
        # 断言结果序列与期望序列相等
        tm.assert_series_equal(
            result,
            expected,
        )

    # 测试在时间增量数据上应用累积乘积方法
    def test_cumprod_timedelta(self):
        # GH#48111
        # 创建包含时间增量数据的序列
        ser = pd.Series([pd.Timedelta(days=1), pd.Timedelta(days=3)])
        # 断言调用cumprod方法会引发TypeError异常，异常消息包含指定内容
        with pytest.raises(TypeError, match="cumprod not supported for Timedelta"):
            ser.cumprod()
```