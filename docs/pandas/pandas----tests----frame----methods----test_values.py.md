# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_values.py`

```
import numpy as np
import pytest

from pandas import (  # 导入 pandas 库中需要的模块
    DataFrame,       # 导入 DataFrame 类
    NaT,             # 导入 NaT（Not a Time）对象
    Series,          # 导入 Series 类
    Timestamp,       # 导入 Timestamp 类
    date_range,      # 导入 date_range 函数
    period_range,    # 导入 period_range 函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestDataFrameValues:
    def test_values(self, float_frame):
        with pytest.raises(ValueError, match="read-only"):  # 检查是否引发 ValueError 异常，并匹配错误消息 "read-only"
            float_frame.values[:, 0] = 5.0  # 尝试修改 float_frame 的第一列的值为 5.0
        assert (float_frame.values[:, 0] != 5).all()  # 断言 float_frame 的第一列的所有值不等于 5

    def test_more_values(self, float_string_frame):
        values = float_string_frame.values  # 获取 float_string_frame 的值
        assert values.shape[1] == len(float_string_frame.columns)  # 断言值的列数等于 float_string_frame 的列数

    def test_values_mixed_dtypes(self, float_frame, float_string_frame):
        frame = float_frame  # 将 float_frame 赋给变量 frame
        arr = frame.values  # 获取 frame 的值数组

        frame_cols = frame.columns  # 获取 frame 的列名
        for i, row in enumerate(arr):  # 遍历值数组的每一行
            for j, value in enumerate(row):  # 遍历每一行的每个值
                col = frame_cols[j]  # 获取当前列名
                if np.isnan(value):  # 如果值为 NaN
                    assert np.isnan(frame[col].iloc[i])  # 断言 frame 的第 i 行、第 col 列的值也为 NaN
                else:
                    assert value == frame[col].iloc[i]  # 否则断言值与 frame 的第 i 行、第 col 列的值相等

        # mixed type
        arr = float_string_frame[["foo", "A"]].values  # 获取 float_string_frame 中 "foo" 和 "A" 列的值数组
        assert arr[0, 0] == "bar"  # 断言第一行第一列的值为 "bar"

        df = DataFrame({"complex": [1j, 2j, 3j], "real": [1, 2, 3]})  # 创建包含复数和实数列的 DataFrame
        arr = df.values  # 获取 DataFrame 的值数组
        assert arr[0, 0] == 1j  # 断言第一行第一列的值为 1j（复数单位）

    def test_values_duplicates(self):
        df = DataFrame(  # 创建包含重复列的 DataFrame
            [[1, 2, "a", "b"], [1, 2, "a", "b"]], columns=["one", "one", "two", "two"]
        )

        result = df.values  # 获取 DataFrame 的值数组
        expected = np.array([[1, 2, "a", "b"], [1, 2, "a", "b"]], dtype=object)  # 创建预期的值数组

        tm.assert_numpy_array_equal(result, expected)  # 使用测试模块 tm 检查 result 是否等于 expected

    def test_values_with_duplicate_columns(self):
        df = DataFrame([[1, 2.5], [3, 4.5]], index=[1, 2], columns=["x", "x"])  # 创建包含重复列名的 DataFrame
        result = df.values  # 获取 DataFrame 的值数组
        expected = np.array([[1, 2.5], [3, 4.5]])  # 创建预期的值数组
        assert (result == expected).all().all()  # 断言所有值与预期值相等

    @pytest.mark.parametrize("constructor", [date_range, period_range])
    def test_values_casts_datetimelike_to_object(self, constructor):
        series = Series(constructor("2000-01-01", periods=10, freq="D"))  # 使用 constructor 创建 Series 对象

        expected = series.astype("object")  # 将 Series 转换为 object 类型

        df = DataFrame(  # 创建包含 Series 和随机数组的 DataFrame
            {"a": series, "b": np.random.default_rng(2).standard_normal(len(series))}
        )

        result = df.values.squeeze()  # 获取 DataFrame 的值数组并去除多余的维度
        assert (result[:, 0] == expected.values).all()  # 断言第一列的值与预期的 Series 值相等

        df = DataFrame({"a": series, "b": ["foo"] * len(series)})  # 创建包含 Series 和重复字符串的 DataFrame

        result = df.values.squeeze()  # 获取 DataFrame 的值数组并去除多余的维度
        assert (result[:, 0] == expected.values).all()  # 断言第一列的值与预期的 Series 值相等
    # 定义一个测试方法，用于测试带有时区信息的数据框的值
    def test_frame_values_with_tz(self):
        # 设置时区为"US/Central"
        tz = "US/Central"
        # 创建一个数据框，其中包含一个列"A"，列值为从"2000-01-01"开始的四个日期，带有指定时区
        df = DataFrame({"A": date_range("2000", periods=4, tz=tz)})
        # 提取数据框的值
        result = df.values
        # 预期的结果，为一个numpy数组，包含四个带有指定时区的时间戳
        expected = np.array(
            [
                [Timestamp("2000-01-01", tz=tz)],
                [Timestamp("2000-01-02", tz=tz)],
                [Timestamp("2000-01-03", tz=tz)],
                [Timestamp("2000-01-04", tz=tz)],
            ]
        )
        # 断言提取的值与预期值相等
        tm.assert_numpy_array_equal(result, expected)

        # 添加第二列"B"，与列"A"的值相同
        df["B"] = df["A"]
        # 更新结果为包含两列的数据框的值
        result = df.values
        # 更新预期结果为将原预期值按列连接得到的numpy数组
        expected = np.concatenate([expected, expected], axis=1)
        # 再次断言提取的值与更新后的预期值相等
        tm.assert_numpy_array_equal(result, expected)

        # 添加第三列"C"，使用列"A"的日期并转换时区为"US/Eastern"
        est = "US/Eastern"
        df["C"] = df["A"].dt.tz_convert(est)
        
        # 创建一个新的numpy数组，包含列"C"的预期时间戳，带有指定时区
        new = np.array(
            [
                [Timestamp("2000-01-01T01:00:00", tz=est)],
                [Timestamp("2000-01-02T01:00:00", tz=est)],
                [Timestamp("2000-01-03T01:00:00", tz=est)],
                [Timestamp("2000-01-04T01:00:00", tz=est)],
            ]
        )
        # 更新预期结果为将前面连接的预期值与新数组按列连接得到的numpy数组
        expected = np.concatenate([expected, new], axis=1)
        # 更新结果为数据框的当前值
        result = df.values
        # 最终断言提取的值与最终预期值相等
        tm.assert_numpy_array_equal(result, expected)
    def test_interleave_with_tzaware(self, timezone_frame):
        # interleave with object
        result = timezone_frame.assign(D="foo").values
        # 创建新的 DataFrame，其中添加了一列 'D'，并获取其值的 NumPy 数组表示
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,  # Not a Time (缺失时间值)
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,  # Not a Time (缺失时间值)
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
                ["foo", "foo", "foo"],  # 字符串数组 'foo'
            ],
            dtype=object,
        ).T  # 转置数组
        tm.assert_numpy_array_equal(result, expected)  # 断言 NumPy 数组相等

        # interleave with only datetime64[ns]
        result = timezone_frame.values
        # 获取 DataFrame 的值的 NumPy 数组表示
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,  # Not a Time (缺失时间值)
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,  # Not a Time (缺失时间值)
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T  # 转置数组
        tm.assert_numpy_array_equal(result, expected)  # 断言 NumPy 数组相等

    def test_values_interleave_non_unique_cols(self):
        df = DataFrame(
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
            columns=["x", "x"],  # 创建具有重复列名的 DataFrame
            index=[1, 2],
        )

        df_unique = df.copy()
        df_unique.columns = ["x", "y"]  # 复制 DataFrame 并重命名列
        assert df_unique.values.shape == df.values.shape  # 断言两个 DataFrame 的值数组形状相同
        tm.assert_numpy_array_equal(df_unique.values[0], df.values[0])  # 断言 NumPy 数组的相等性
        tm.assert_numpy_array_equal(df_unique.values[1], df.values[1])  # 断言 NumPy 数组的相等性

    def test_values_numeric_cols(self, float_frame):
        float_frame["foo"] = "bar"  # 在 DataFrame 中添加一个新列 'foo'，每行值为字符串 "bar"

        values = float_frame[["A", "B", "C", "D"]].values  # 获取特定列的值的 NumPy 数组表示
        assert values.dtype == np.float64  # 断言数组的数据类型为 np.float64
    # 定义一个测试方法，用于测试混合数据帧中特定列的数值类型

    # 对混合浮点数数据帧的列"A", "B", "C", "D"进行值提取，并检查其数据类型是否为np.float64
    values = mixed_float_frame[["A", "B", "C", "D"]].values
    assert values.dtype == np.float64

    # 对混合浮点数数据帧的列"A", "B", "C"进行值提取，并检查其数据类型是否为np.float32
    values = mixed_float_frame[["A", "B", "C"]].values
    assert values.dtype == np.float32

    # 对混合浮点数数据帧的列"C"进行值提取，并检查其数据类型是否为np.float16
    values = mixed_float_frame[["C"]].values
    assert values.dtype == np.float16

    # GH#10364: 对混合整数数据帧的列"A", "B", "C", "D"进行值提取，并检查其数据类型是否为np.float64
    values = mixed_int_frame[["A", "B", "C", "D"]].values
    assert values.dtype == np.float64

    # 对混合整数数据帧的列"A", "D"进行值提取，并检查其数据类型是否为np.int64
    values = mixed_int_frame[["A", "D"]].values
    assert values.dtype == np.int64

    # 对混合整数数据帧的列"A", "B", "C"进行值提取，并检查其数据类型是否为np.float64
    values = mixed_int_frame[["A", "B", "C"]].values
    assert values.dtype == np.float64

    # 因为"B"和"C"都是无符号整数类型，不需要强制转换为float类型
    values = mixed_int_frame[["B", "C"]].values
    assert values.dtype == np.uint64

    # 对混合整数数据帧的列"A", "C"进行值提取，并检查其数据类型是否为np.int32
    values = mixed_int_frame[["A", "C"]].values
    assert values.dtype == np.int32

    # 对混合整数数据帧的列"C", "D"进行值提取，并检查其数据类型是否为np.int64
    values = mixed_int_frame[["C", "D"]].values
    assert values.dtype == np.int64

    # 对混合整数数据帧的列"A"进行值提取，并检查其数据类型是否为np.int32
    values = mixed_int_frame[["A"]].values
    assert values.dtype == np.int32

    # 对混合整数数据帧的列"C"进行值提取，并检查其数据类型是否为np.uint8
    values = mixed_int_frame[["C"]].values
    assert values.dtype == np.uint8
    # 定义一个名为 TestPrivateValues 的测试类，用于测试私有值相关的功能
    class TestPrivateValues:

        # 测试处理带时区信息的日期范围，并使用 reshape 方法重塑数据
        def test_private_values_dt64tz(self):
            # 创建一个带时区信息的日期范围对象，并提取其内部数据并重塑为一列
            dta = date_range("2000", periods=4, tz="US/Central")._data.reshape(-1, 1)

            # 使用提取的数据创建一个 DataFrame，列名为 "A"
            df = DataFrame(dta, columns=["A"])
            # 断言 DataFrame 的内部值与提取的数据相等
            tm.assert_equal(df._values, dta)

            # 断言 df._values 和 dta._ndarray 不共享内存
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)

            # 计算 TimedeltaArray
            tda = dta - dta
            # 计算 DataFrame 的差分，并断言结果与 TimedeltaArray 相等
            df2 = df - df
            tm.assert_equal(df2._values, tda)

        # 测试处理带时区信息的日期范围，并使用 reshape 方法重塑数据为两列
        def test_private_values_dt64tz_multicol(self):
            # 创建一个带时区信息的日期范围对象，并提取其内部数据并重塑为两列
            dta = date_range("2000", periods=8, tz="US/Central")._data.reshape(-1, 2)

            # 使用提取的数据创建一个 DataFrame，列名分别为 "A" 和 "B"
            df = DataFrame(dta, columns=["A", "B"])
            # 断言 DataFrame 的内部值与提取的数据相等
            tm.assert_equal(df._values, dta)

            # 断言 df._values 和 dta._ndarray 不共享内存
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)

            # 计算 TimedeltaArray
            tda = dta - dta
            # 计算 DataFrame 的差分，并断言结果与 TimedeltaArray 相等
            df2 = df - df
            tm.assert_equal(df2._values, tda)

        # 测试处理不带时区信息的日期范围，并使用 reshape 方法重塑数据为多个块
        def test_private_values_dt64_multiblock(self):
            # 创建一个不带时区信息的日期范围对象，并提取其内部数据
            dta = date_range("2000", periods=8)._data

            # 使用提取的数据创建一个 DataFrame，其中 "A" 列使用前四个数据，"B" 列使用后四个数据
            df = DataFrame({"A": dta[:4]}, copy=False)
            df["B"] = dta[4:]

            # 断言 DataFrame 内部数据块的数量为 2
            assert len(df._mgr.blocks) == 2

            # 提取 DataFrame 的内部值
            result = df._values
            # 期望的结果是提取的数据重塑为 2 行 4 列的形式
            expected = dta.reshape(2, 4).T
            # 断言 DataFrame 的内部值与期望的结果相等
            tm.assert_equal(result, expected)
```