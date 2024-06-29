# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_describe.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas.compat.numpy import np_version_gte1p25  # 导入 NumPy 兼容性模块

from pandas.core.dtypes.common import (  # 导入 pandas 核心数据类型常用功能模块
    is_complex_dtype,  # 检查是否为复数类型
    is_extension_array_dtype,  # 检查是否为扩展数组类型
)

from pandas import (  # 导入 pandas 主要模块
    NA,  # pandas 中的缺失值标记
    Period,  # 表示时间周期的数据类型
    Series,  # 表示一维数组和标签的数据类型
    Timedelta,  # 表示时间差的数据类型
    Timestamp,  # 表示时间戳的数据类型
    date_range,  # 生成日期范围的函数
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块


class TestSeriesDescribe:
    def test_describe_ints(self):
        ser = Series([0, 1, 2, 3, 4], name="int_data")  # 创建整数类型的 Series 对象
        result = ser.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [5, 2, ser.std(), 0, 1, 2, 3, 4],
            name="int_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

    def test_describe_bools(self):
        ser = Series([True, True, False, False, False], name="bool_data")  # 创建布尔类型的 Series 对象
        result = ser.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [5, 2, False, 3],
            name="bool_data",
            index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

    def test_describe_strs(self):
        ser = Series(["a", "a", "b", "c", "d"], name="str_data")  # 创建字符串类型的 Series 对象
        result = ser.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [5, 4, "a", 2],
            name="str_data",
            index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

    def test_describe_timedelta64(self):
        ser = Series(  # 创建时间差类型的 Series 对象
            [
                Timedelta("1 days"),
                Timedelta("2 days"),
                Timedelta("3 days"),
                Timedelta("4 days"),
                Timedelta("5 days"),
            ],
            name="timedelta_data",
        )
        result = ser.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [5, ser[2], ser.std(), ser[0], ser[1], ser[2], ser[3], ser[4]],
            name="timedelta_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

    def test_describe_period(self):
        ser = Series(  # 创建时间周期类型的 Series 对象
            [
                Period("2020-01", "M"),
                Period("2020-01", "M"),
                Period("2019-12", "M"),
            ],
            name="period_data",
        )
        result = ser.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [3, 2, ser[0], 2],
            name="period_data",
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

    def test_describe_empty_object(self):
        # https://github.com/pandas-dev/pandas/issues/27183
        s = Series([None, None], dtype=object)  # 创建空对象类型的 Series 对象
        result = s.describe()  # 调用 describe 方法获取描述性统计信息
        expected = Series(  # 期望的结果 Series 对象
            [0, 0, np.nan, np.nan],
            dtype=object,
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致

        result = s[:0].describe()  # 对空对象类型的切片调用 describe 方法获取描述性统计信息
        tm.assert_series_equal(result, expected)  # 使用测试工具验证结果是否与期望一致
        assert np.isnan(result.iloc[2])  # 断言结果的第三个元素是 NaN
        assert np.isnan(result.iloc[3])  # 断言结果的第四个元素是 NaN
    # 定义一个测试函数，测试带有时区信息的描述功能
    def test_describe_with_tz(self, tz_naive_fixture):
        # GH 21332
        # 从测试夹具中获取时区信息
        tz = tz_naive_fixture
        # 将时区信息转换为字符串作为名称
        name = str(tz_naive_fixture)
        # 创建起始时间戳为2018年1月1日的时间戳对象
        start = Timestamp(2018, 1, 1)
        # 创建结束时间戳为2018年1月5日的时间戳对象
        end = Timestamp(2018, 1, 5)
        # 创建一个时间序列，时间范围从start到end，使用给定的时区tz，序列名为name
        s = Series(date_range(start, end, tz=tz), name=name)
        # 对该时间序列执行描述统计，结果保存在result中
        result = s.describe()
        # 创建预期的时间序列，包含统计结果的Series对象
        expected = Series(
            [
                5,
                Timestamp(2018, 1, 3).tz_localize(tz),
                start.tz_localize(tz),
                s[1],
                s[2],
                s[3],
                end.tz_localize(tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        # 使用测试工具库中的函数验证result与expected是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试带有时区信息的数值型描述功能
    def test_describe_with_tz_numeric(self):
        # 设置时区和名称
        name = tz = "CET"
        # 创建起始时间戳为2018年1月1日的时间戳对象
        start = Timestamp(2018, 1, 1)
        # 创建结束时间戳为2018年1月5日的时间戳对象
        end = Timestamp(2018, 1, 5)
        # 创建一个时间序列，时间范围从start到end，使用给定的时区tz，序列名为name
        s = Series(date_range(start, end, tz=tz), name=name)

        # 对该时间序列执行描述统计，结果保存在result中
        result = s.describe()

        # 创建预期的时间序列，包含统计结果的Series对象
        expected = Series(
            [
                5,
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-01 00:00:00", tz=tz),
                Timestamp("2018-01-02 00:00:00", tz=tz),
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-04 00:00:00", tz=tz),
                Timestamp("2018-01-05 00:00:00", tz=tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        # 使用测试工具库中的函数验证result与expected是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试日期时间是数值型并包含日期时间的描述功能
    def test_datetime_is_numeric_includes_datetime(self):
        # 创建一个日期时间范围的时间序列，从2012年开始，持续3个时间点
        s = Series(date_range("2012", periods=3))
        # 对该时间序列执行描述统计，结果保存在result中
        result = s.describe()
        # 创建预期的时间序列，包含统计结果的Series对象
        expected = Series(
            [
                3,
                Timestamp("2012-01-02"),
                Timestamp("2012-01-01"),
                Timestamp("2012-01-01T12:00:00"),
                Timestamp("2012-01-02"),
                Timestamp("2012-01-02T12:00:00"),
                Timestamp("2012-01-03"),
            ],
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        # 使用测试工具库中的函数验证result与expected是否相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的标记，忽略警告："Casting complex values to real discards"
    # 测试函数，验证对于任何数值类型的结果数据类型
    def test_numeric_result_dtype(self, any_numeric_dtype):
        # GH#48340 - describe 应始终在非复数数值输入时返回 float
        # 检查输入数据类型是否为扩展数组类型
        if is_extension_array_dtype(any_numeric_dtype):
            dtype = "Float64"
        else:
            # 如果不是扩展数组类型，检查是否是复数数据类型，决定 dtype
            dtype = "complex128" if is_complex_dtype(any_numeric_dtype) else None

        # 创建一个包含 [0, 1] 的 Series，指定数据类型为 any_numeric_dtype
        ser = Series([0, 1], dtype=any_numeric_dtype)

        # 如果 dtype 是 complex128 并且 NumPy 版本 >= 1.25
        if dtype == "complex128" and np_version_gte1p25:
            # 用 pytest 检查是否会引发 TypeError，匹配错误消息 "^a must be an array of real numbers$"
            with pytest.raises(
                TypeError, match=r"^a must be an array of real numbers$"
            ):
                ser.describe()
            return  # 结束测试函数

        # 获取 Series 的 describe() 方法返回的结果
        result = ser.describe()

        # 期望的结果 Series，包含 count、mean、std、min、25%、50%、75%、max 等指标
        expected = Series(
            [
                2.0,
                0.5,
                ser.std(),
                0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype=dtype,
        )

        # 使用测试模块中的 assert 函数验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 测试函数，验证只包含一个元素的扩展数组的 describe() 方法
    def test_describe_one_element_ea(self):
        # GH#52515
        # 创建一个包含 [0.0] 的 Series，指定数据类型为 "Float64"
        ser = Series([0.0], dtype="Float64")

        # 使用测试模块中的 assert_produces_warning 函数验证是否会产生警告
        with tm.assert_produces_warning(None):
            # 获取 Series 的 describe() 方法返回的结果
            result = ser.describe()

        # 期望的结果 Series，包含 count、mean、std、min、25%、50%、75%、max 等指标
        expected = Series(
            [1, 0, NA, 0, 0, 0, 0, 0],
            dtype="Float64",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )

        # 使用测试模块中的 assert 函数验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
```