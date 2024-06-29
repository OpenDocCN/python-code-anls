# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_diff.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析
from pandas import (  # 从 Pandas 中导入特定模块和函数
    DataFrame,  # 数据框架类
    Series,  # 系列类
    Timestamp,  # 时间戳类
    date_range,  # 日期范围函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestDataFrameDiff:
    def test_diff_requires_integer(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        with pytest.raises(ValueError, match="periods must be an integer"):
            df.diff(1.5)  # 调用 diff 方法，期望抛出值错误，提示周期必须是整数

    # GH#44572 np.int64 is accepted
    @pytest.mark.parametrize("num", [1, np.int64(1)])
    def test_diff(self, datetime_frame, num):
        df = datetime_frame  # 使用传入的 datetime_frame 进行测试
        the_diff = df.diff(num)  # 调用 diff 方法，传入参数 num

        expected = df["A"] - df["A"].shift(num)  # 期望的结果是列"A"的差分
        tm.assert_series_equal(the_diff["A"], expected)  # 断言检查结果与期望是否一致

    def test_diff_int_dtype(self):
        # int dtype
        a = 10_000_000_000_000_000  # 定义一个整数变量 a
        b = a + 1  # 定义一个整数变量 b，比 a 大 1
        ser = Series([a, b])  # 创建一个包含 a 和 b 的系列对象

        rs = DataFrame({"s": ser}).diff()  # 对系列进行差分操作
        assert rs.s[1] == 1  # 断言检查差分结果中 s 列第二行是否为 1

    def test_diff_mixed_numeric(self, datetime_frame):
        # mixed numeric
        tf = datetime_frame.astype("float32")  # 将 datetime_frame 转换为 float32 类型
        the_diff = tf.diff(1)  # 对转换后的数据进行差分操作
        tm.assert_series_equal(the_diff["A"], tf["A"] - tf["A"].shift(1))  # 断言检查差分结果是否符合预期

    def test_diff_axis1_nonconsolidated(self):
        # GH#10907
        df = DataFrame({"y": Series([2]), "z": Series([3])})  # 创建一个包含 "y" 和 "z" 列的数据框架
        df.insert(0, "x", 1)  # 在第一列插入一个名为 "x" 的列，所有值为 1
        result = df.diff(axis=1)  # 沿着列轴进行差分操作
        expected = DataFrame({"x": np.nan, "y": Series(1), "z": Series(1)})  # 期望的结果数据框架
        tm.assert_frame_equal(result, expected)  # 断言检查结果是否与期望一致

    def test_diff_timedelta64_with_nat(self):
        # GH#32441
        arr = np.arange(6).reshape(3, 2).astype("timedelta64[ns]")  # 创建一个时间增量数组
        arr[:, 0] = np.timedelta64("NaT", "ns")  # 将第一列设置为 NaT (Not a Time) 值

        df = DataFrame(arr)  # 创建一个数据框架，使用时间增量数组
        result = df.diff(1, axis=0)  # 在行方向上进行差分操作

        expected = DataFrame({0: df[0], 1: [pd.NaT, pd.Timedelta(2), pd.Timedelta(2)]})  # 期望的结果数据框架
        tm.assert_equal(result, expected)  # 断言检查结果是否与期望一致

        result = df.diff(0)  # 对数据框架进行零差分操作
        expected = df - df  # 创建一个与 df 形状相同且所有值为零的数据框架
        assert expected[0].isna().all()  # 断言检查第一列是否全部为 NaN
        tm.assert_equal(result, expected)  # 断言检查结果是否与期望一致

        result = df.diff(-1, axis=1)  # 在列方向上进行逆差分操作
        expected = df * np.nan  # 创建一个与 df 形状相同且所有值为 NaN 的数据框架
        tm.assert_equal(result, expected)  # 断言检查结果是否与期望一致

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0_with_nat(self, tz, unit):
        # GH#32441
        dti = pd.DatetimeIndex(["NaT", "2019-01-01", "2019-01-02"], tz=tz).as_unit(unit)  # 创建一个日期时间索引
        ser = Series(dti)  # 创建一个包含日期时间索引的系列对象

        df = ser.to_frame()  # 将系列转换为数据框架

        result = df.diff()  # 对数据框架进行默认轴上的差分操作
        ex_index = pd.TimedeltaIndex([pd.NaT, pd.NaT, pd.Timedelta(days=1)]).as_unit(unit)  # 期望的索引
        expected = Series(ex_index).to_frame()  # 将期望的索引转换为数据框架
        tm.assert_frame_equal(result, expected)  # 断言检查结果是否与期望一致

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_with_nat_zero_periods(self, tz):
        # 测试在存在 NaT 值时，diff 应返回 NaT 而不是 timedelta64(0)
        # 创建一个日期范围对象 dti，从 "2016-01-01" 开始，包含4个周期，带有时区 tz
        dti = date_range("2016-01-01", periods=4, tz=tz)
        # 将日期范围对象转换为 Series 对象
        ser = Series(dti)
        # 将 Series 对象转换为 DataFrame 对象 df，进行深拷贝
        df = ser.to_frame().copy()

        # 复制 Series 列到 df 的第二列
        df[1] = ser.copy()

        # 将 df 的第一列所有值设置为 pd.NaT
        df.iloc[:, 0] = pd.NaT

        # 计算 df 与自身的差值，期望结果是所有值都为 NaN 的 DataFrame
        expected = df - df
        assert expected[0].isna().all()

        # 对 df 按照 axis=0 进行差分计算，期望结果与 expected 相等
        result = df.diff(0, axis=0)
        tm.assert_frame_equal(result, expected)

        # 对 df 按照 axis=1 进行差分计算，期望结果与 expected 相等
        result = df.diff(0, axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis0(self, tz):
        # GH#18578 测试 DataFrame 沿着 axis=0 进行差分计算
        # 创建一个 DataFrame df，包含两列，每列为一个日期范围对象，带有时区 tz
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        # 计算 df 按照 axis=0 进行差分计算，期望结果为 TimedeltaIndex 对象
        result = df.diff(axis=0)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "1 days"]),
                1: pd.TimedeltaIndex(["NaT", "1 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_diff_datetime_axis1(self, tz):
        # GH#18578 测试 DataFrame 沿着 axis=1 进行差分计算
        # 创建一个 DataFrame df，包含两列，每列为一个日期范围对象，带有时区 tz
        df = DataFrame(
            {
                0: date_range("2010", freq="D", periods=2, tz=tz),
                1: date_range("2010", freq="D", periods=2, tz=tz),
            }
        )

        # 计算 df 按照 axis=1 进行差分计算，期望结果为 TimedeltaIndex 对象
        result = df.diff(axis=1)
        expected = DataFrame(
            {
                0: pd.TimedeltaIndex(["NaT", "NaT"]),
                1: pd.TimedeltaIndex(["0 days", "0 days"]),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta(self, unit):
        # GH#4533 测试 DataFrame 按时间差分
        # 创建一个 DataFrame df，包含时间戳和数值列
        df = DataFrame(
            {
                "time": [Timestamp("20130101 9:01"), Timestamp("20130101 9:02")],
                "value": [1.0, 2.0],
            }
        )
        # 将时间列 df["time"] 转换为指定时间单位 unit
        df["time"] = df["time"].dt.as_unit(unit)

        # 计算 df 按时间差分，期望结果为指定时间单位的差分
        res = df.diff()
        exp = DataFrame(
            [[pd.NaT, np.nan], [pd.Timedelta("00:01:00"), 1]], columns=["time", "value"]
        )
        exp["time"] = exp["time"].dt.as_unit(unit)
        tm.assert_frame_equal(res, exp)

    def test_diff_mixed_dtype(self):
        # 测试 DataFrame 包含混合数据类型时的差分
        # 创建一个随机数填充的 DataFrame df，其中一列为对象类型的数组
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["A"] = np.array([1, 2, 3, 4, 5], dtype=object)

        # 计算 df 的差分，验证结果的第一列数据类型为 np.float64
        result = df.diff()
        assert result[0].dtype == np.float64

    def test_diff_neg_n(self, datetime_frame):
        # 测试 DataFrame 按负数 n 进行差分计算
        # 计算 DataFrame datetime_frame 的负数 n 差分，与 shift(-1) 结果进行比较
        rs = datetime_frame.diff(-1)
        xp = datetime_frame - datetime_frame.shift(-1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_float_n(self, datetime_frame):
        # 测试 DataFrame 按浮点数 n 进行差分计算
        # 计算 DataFrame datetime_frame 的浮点数 n 差分，与整数 n 差分结果进行比较
        rs = datetime_frame.diff(1.0)
        xp = datetime_frame.diff(1)
        tm.assert_frame_equal(rs, xp)
    # 定义测试方法，用于测试在不同轴上计算数据帧的差分
    def test_diff_axis(self):
        # GH#9727
        # 创建一个包含数值的数据帧
        df = DataFrame([[1.0, 2.0], [3.0, 4.0]])
        # 断言在 axis=1 上计算差分后的数据帧与期望的数据帧相等
        tm.assert_frame_equal(
            df.diff(axis=1), DataFrame([[np.nan, 1.0], [np.nan, 1.0]])
        )
        # 断言在 axis=0 上计算差分后的数据帧与期望的数据帧相等
        tm.assert_frame_equal(
            df.diff(axis=0), DataFrame([[np.nan, np.nan], [2.0, 2.0]])
        )

    # 定义测试方法，用于测试在特定时间周期下的差分计算
    def test_diff_period(self):
        # GH#32995 不要传入不正确的轴参数
        # 创建一个包含日期范围的时间索引
        pi = date_range("2016-01-01", periods=3).to_period("D")
        # 创建一个数据帧，其中列 'A' 包含时间索引对象
        df = DataFrame({"A": pi})

        # 计算 axis=1 方向上的差分
        result = df.diff(1, axis=1)

        # 创建一个期望的数据帧，其中所有值都为 NaN
        expected = (df - pd.NaT).astype(object)
        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在含有混合数据类型且轴为1时的差分计算
    def test_diff_axis1_mixed_dtypes(self):
        # GH#32995 当数据类型混合且轴为1时，应列向操作
        # 创建一个包含混合数据类型的数据帧
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        # 创建一个期望的数据帧，其中'A'列的值为 NaN，'B'列的值为除以2后的结果
        expected = DataFrame({"A": [np.nan, np.nan, np.nan], "B": df["B"] / 2})

        # 计算 axis=1 方向上的差分
        result = df.diff(axis=1)
        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)

        # GH#21437 处理混合浮点数据类型
        # 创建一个包含混合浮点数据类型的数据帧
        df = DataFrame(
            {"a": np.arange(3, dtype="float32"), "b": np.arange(3, dtype="float64")}
        )
        # 计算 axis=1 方向上的差分
        result = df.diff(axis=1)
        # 创建一个期望的数据帧，其中'a'列的值为 NaN，'b'列的值为乘以0后的结果
        expected = DataFrame({"a": df["a"] * np.nan, "b": df["b"] * 0})
        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在含有混合数据类型且轴为1时大周期差分计算
    def test_diff_axis1_mixed_dtypes_large_periods(self):
        # GH#32995 当数据类型混合且轴为1时，应列向操作
        # 创建一个包含混合数据类型的数据帧
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        # 创建一个全部值为 NaN 的期望数据帧
        expected = df * np.nan

        # 计算 axis=1 方向上的大周期差分
        result = df.diff(axis=1, periods=3)
        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在含有混合数据类型且轴为1时负周期差分计算
    def test_diff_axis1_mixed_dtypes_negative_periods(self):
        # GH#32995 当数据类型混合且轴为1时，应列向操作
        # 创建一个包含混合数据类型的数据帧
        df = DataFrame({"A": range(3), "B": 2 * np.arange(3, dtype=np.float64)})

        # 创建一个期望的数据帧，其中'A'列的值为 '-1.0 * df["A"]'，'B'列的值为全部为 NaN
        expected = DataFrame({"A": -1.0 * df["A"], "B": df["B"] * np.nan})

        # 计算 axis=1 方向上的负周期差分
        result = df.diff(axis=1, periods=-1)
        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试稀疏数据帧上的差分计算
    def test_diff_sparse(self):
        # GH#28813 对稀疏数据帧应当正常工作
        # 创建一个稀疏整数数据类型的数据帧
        sparse_df = DataFrame([[0, 1], [1, 0]], dtype="Sparse[int]")

        # 计算差分
        result = sparse_df.diff()

        # 创建一个期望的数据帧，其中包含对应差分后的结果
        expected = DataFrame(
            [[np.nan, np.nan], [1.0, -1.0]], dtype=pd.SparseDtype("float", 0.0)
        )

        # 断言计算结果与期望的数据帧相等
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize(
        "axis,expected",
        [  # 参数化测试的参数定义开始
            (
                0,  # axis = 0，表示沿行进行操作
                DataFrame(  # 期望的 DataFrame 结果，包含四列 'a', 'b', 'c', 'd'
                    {
                        "a": [np.nan, 0, 1, 0, np.nan, np.nan, np.nan, 0],  # 列 'a' 的期望值列表
                        "b": [np.nan, 1, np.nan, np.nan, -2, 1, np.nan, np.nan],  # 列 'b' 的期望值列表
                        "c": np.repeat(np.nan, 8),  # 列 'c' 的期望值，重复 8 次 np.nan
                        "d": [np.nan, 3, 5, 7, 9, 11, 13, 15],  # 列 'd' 的期望值列表
                    },
                    dtype="Int64",  # 数据类型设定为 'Int64'
                ),
            ),
            (
                1,  # axis = 1，表示沿列进行操作
                DataFrame(  # 期望的 DataFrame 结果，包含四列 'a', 'b', 'c', 'd'
                    {
                        "a": np.repeat(np.nan, 8),  # 列 'a' 的期望值，重复 8 次 np.nan
                        "b": [0, 1, np.nan, 1, np.nan, np.nan, np.nan, 0],  # 列 'b' 的期望值列表
                        "c": np.repeat(np.nan, 8),  # 列 'c' 的期望值，重复 8 次 np.nan
                        "d": np.repeat(np.nan, 8),  # 列 'd' 的期望值，重复 8 次 np.nan
                    },
                    dtype="Int64",  # 数据类型设定为 'Int64'
                ),
            ),
        ],  # 参数化测试的参数定义结束
    )
    def test_diff_integer_na(self, axis, expected):
        # GH#24171 IntegerNA Support for DataFrame.diff()
        df = DataFrame(  # 创建一个 DataFrame 对象 df
            {
                "a": np.repeat([0, 1, np.nan, 2], 2),  # 列 'a' 的值重复模式
                "b": np.tile([0, 1, np.nan, 2], 2),  # 列 'b' 的值平铺模式
                "c": np.repeat(np.nan, 8),  # 列 'c' 的值，重复 8 次 np.nan
                "d": np.arange(1, 9) ** 2,  # 列 'd' 的值，1 到 8 的平方
            },
            dtype="Int64",  # 数据类型设定为 'Int64'
        )

        # Test case for default behaviour of diff
        result = df.diff(axis=axis)  # 执行 DataFrame 的差分操作，沿指定轴
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值一致

    def test_diff_readonly(self):
        # https://github.com/pandas-dev/pandas/issues/35559
        arr = np.random.default_rng(2).standard_normal((5, 2))  # 生成一个 5x2 的正态分布随机数组
        arr.flags.writeable = False  # 将数组设为只读
        df = DataFrame(arr)  # 用数组创建 DataFrame 对象 df
        result = df.diff()  # 执行 DataFrame 的差分操作
        expected = DataFrame(np.array(df)).diff()  # 用原始数据创建的 DataFrame 进行差分操作，作为期望值
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值一致

    def test_diff_all_int_dtype(self, any_int_numpy_dtype):
        # GH 14773
        df = DataFrame(range(5))  # 创建一个整数范围的 DataFrame 对象 df
        df = df.astype(any_int_numpy_dtype)  # 将 df 转换为指定的整数数据类型
        result = df.diff()  # 执行 DataFrame 的差分操作
        expected_dtype = (  # 期望的结果数据类型
            "float32" if any_int_numpy_dtype in ("int8", "int16") else "float64"
        )
        expected = DataFrame([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=expected_dtype)  # 期望的 DataFrame 结果
        tm.assert_frame_equal(result, expected)  # 断言结果与期望值一致
```