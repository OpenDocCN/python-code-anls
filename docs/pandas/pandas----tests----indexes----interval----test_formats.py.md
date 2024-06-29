# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_formats.py`

```
# 导入必要的库
import numpy as np  # 导入 numpy 库
import pytest  # 导入 pytest 库

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 库中的配置项

# 从 pandas 库中导入多个类和函数
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm  # 导入 pandas 库中的测试模块


class TestIntervalIndexRendering:
    # TODO: this is a test for DataFrame/Series, not IntervalIndex
    @pytest.mark.parametrize(
        "constructor,expected",
        [
            (
                Series,
                (
                    "(0.0, 1.0]    a\n"
                    "NaN           b\n"
                    "(2.0, 3.0]    c\n"
                    "dtype: object"
                ),
            ),
            (DataFrame, ("            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c")),
        ],
    )
    def test_repr_missing(self, constructor, expected, using_infer_string, request):
        # GH 25984
        # 如果 using_infer_string 为真且构造函数是 Series，则标记为预期测试失败，原因是表达方式不同
        if using_infer_string and constructor is Series:
            request.applymarker(pytest.mark.xfail(reason="repr different"))
        # 创建一个 IntervalIndex 对象，包含 [(0, 1), np.nan, (2, 3)] 的元组
        index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        # 使用给定的构造函数和索引创建对象 obj
        obj = constructor(list("abc"), index=index)
        # 获取对象 obj 的字符串表示形式
        result = repr(obj)
        # 断言结果与预期相符
        assert result == expected

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="repr different")
    def test_repr_floats(self):
        # GH 32553

        # 创建一个 Series 对象 markers，包含两个元素 "foo" 和 "bar"
        markers = Series(
            ["foo", "bar"],
            # 使用 IntervalIndex 对象作为索引，包含两个 Interval 对象
            index=IntervalIndex(
                [
                    Interval(left, right)
                    for left, right in zip(
                        Index([329.973, 345.137], dtype="float64"),
                        Index([345.137, 360.191], dtype="float64"),
                    )
                ]
            ),
        )
        # 获取 markers 对象的字符串表示形式
        result = str(markers)
        # 预期的字符串表示形式
        expected = "(329.973, 345.137]    foo\n(345.137, 360.191]    bar\ndtype: object"
        # 断言结果与预期相符
        assert result == expected
    @pytest.mark.parametrize(
        "tuples, closed, expected_data",
        [  # 使用 pytest 的参数化装饰器，定义测试用例参数
            ([(0, 1), (1, 2), (2, 3)], "left", ["[0, 1)", "[1, 2)", "[2, 3)"]),  # 第一个测试用例：定义区间、闭合方式和预期输出
            (
                [(0.5, 1.0), np.nan, (2.0, 3.0)],  # 第二个测试用例：包含 NaN 值的区间、闭合方式和预期输出
                "right",
                ["(0.5, 1.0]", "NaN", "(2.0, 3.0]"],
            ),
            (
                [
                    (Timestamp("20180101"), Timestamp("20180102")),  # 第三个测试用例：包含 Timestamp 对象的区间、闭合方式和预期输出
                    np.nan,
                    ((Timestamp("20180102"), Timestamp("20180103"))),
                ],
                "both",
                [
                    "[2018-01-01 00:00:00, 2018-01-02 00:00:00]",  # 预期输出为时间戳范围字符串
                    "NaN",  # 预期输出为 NaN
                    "[2018-01-02 00:00:00, 2018-01-03 00:00:00]",  # 预期输出为时间戳范围字符串
                ],
            ),
            (
                [
                    (Timedelta("0 days"), Timedelta("1 days")),  # 第四个测试用例：包含 Timedelta 对象的区间、闭合方式和预期输出
                    (Timedelta("1 days"), Timedelta("2 days")),
                    np.nan,
                ],
                "neither",
                [
                    "(0 days 00:00:00, 1 days 00:00:00)",  # 预期输出为时间增量范围字符串
                    "(1 days 00:00:00, 2 days 00:00:00)",  # 预期输出为时间增量范围字符串
                    "NaN",  # 预期输出为 NaN
                ],
            ),
        ],
    )
    def test_get_values_for_csv(self, tuples, closed, expected_data):
        # GH 28210
        index = IntervalIndex.from_tuples(tuples, closed=closed)  # 创建区间索引对象
        result = index._get_values_for_csv(na_rep="NaN")  # 调用区间索引对象的方法，获取 CSV 输出值
        expected = np.array(expected_data)  # 将预期输出转换为 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)  # 使用 pandas 测试框架比较结果与预期值的 NumPy 数组是否相等

    def test_timestamp_with_timezone(self, unit):
        # GH 55035
        left = DatetimeIndex(["2020-01-01"], dtype=f"M8[{unit}, UTC]")  # 创建带有时区信息的日期时间索引
        right = DatetimeIndex(["2020-01-02"], dtype=f"M8[{unit}, UTC]")  # 创建带有时区信息的日期时间索引
        index = IntervalIndex.from_arrays(left, right)  # 使用左右日期时间索引创建区间索引对象
        result = repr(index)  # 获取区间索引对象的字符串表示形式
        expected = (
            "IntervalIndex([(2020-01-01 00:00:00+00:00, 2020-01-02 00:00:00+00:00]], "
            f"dtype='interval[datetime64[{unit}, UTC], right]')"
        )  # 预期输出的字符串表示形式
        assert result == expected  # 断言实际输出与预期输出是否相等
```