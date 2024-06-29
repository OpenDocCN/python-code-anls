# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_update.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 导入 pandas 库中的测试用例装饰器
import pandas.util._test_decorators as td

# 从 pandas 库中导入特定模块和类：CategoricalDtype, DataFrame, NaT, Series, Timestamp
from pandas import (
    CategoricalDtype,
    DataFrame,
    NaT,
    Series,
    Timestamp,
)

# 导入 pandas 内部测试工具模块
import pandas._testing as tm

# 定义一个测试类 TestUpdate
class TestUpdate:
    # 定义测试方法 test_update
    def test_update(self):
        # 创建一个包含 NaN 的 Series 对象 s
        s = Series([1.5, np.nan, 3.0, 4.0, np.nan])
        # 创建另一个 Series 对象 s2
        s2 = Series([np.nan, 3.5, np.nan, 5.0])
        # 使用 s2 更新 s
        s.update(s2)

        # 创建期望的 Series 对象 expected
        expected = Series([1.5, 3.5, 3.0, 5.0, np.nan])
        # 断言 s 和 expected 是否相等
        tm.assert_series_equal(s, expected)

        # 创建一个包含字典的 DataFrame 对象 df
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        # 在 df 中新增列 "c"，并赋值为 NaN
        df["c"] = np.nan
        # 将列 "c" 强制转换为 object 类型，以避免设置 "foo" 时的类型提升
        df["c"] = df["c"].astype(object)
        # 复制原始的 df 到 df_orig
        df_orig = df.copy()

        # 使用 assert 引发 chained_assignment_error 异常
        with tm.raises_chained_assignment_error():
            # 使用 Series 对象 ["foo"] 更新 df 中的列 "c"，并指定索引为 [0]
            df["c"].update(Series(["foo"], index=[0]))
        
        # 创建期望的 DataFrame 对象 expected，与 df_orig 相同
        expected = df_orig
        # 断言 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    # 使用 pytest.mark.parametrize 装饰器参数化测试方法
    @pytest.mark.parametrize(
        "other, dtype, expected, warn",
        [
            # other 是 int 类型
            ([61, 63], "int32", Series([10, 61, 12], dtype="int32"), None),
            ([61, 63], "int64", Series([10, 61, 12]), None),
            ([61, 63], float, Series([10.0, 61.0, 12.0]), None),
            ([61, 63], object, Series([10, 61, 12], dtype=object), None),
            # other 是 float 类型，但可以转换为 int
            ([61.0, 63.0], "int32", Series([10, 61, 12], dtype="int32"), None),
            ([61.0, 63.0], "int64", Series([10, 61, 12]), None),
            ([61.0, 63.0], float, Series([10.0, 61.0, 12.0]), None),
            ([61.0, 63.0], object, Series([10, 61.0, 12], dtype=object), None),
            # other 是 float 类型，无法转换为 int
            ([61.1, 63.1], "int32", Series([10.0, 61.1, 12.0]), FutureWarning),
            ([61.1, 63.1], "int64", Series([10.0, 61.1, 12.0]), FutureWarning),
            ([61.1, 63.1], float, Series([10.0, 61.1, 12.0]), None),
            ([61.1, 63.1], object, Series([10, 61.1, 12], dtype=object), None),
            # other 是 object 类型，无法转换为 int
            ([(61,), (63,)], "int32", Series([10, (61,), 12]), FutureWarning),
            ([(61,), (63,)], "int64", Series([10, (61,), 12]), FutureWarning),
            ([(61,), (63,)], float, Series([10.0, (61,), 12.0]), FutureWarning),
            ([(61,), (63,)], object, Series([10, (61,), 12]), None),
        ],
    )
    # 定义参数化测试方法 test_update_dtypes，包含参数 other, dtype, expected, warn
    def test_update_dtypes(self, other, dtype, expected, warn):
        # 创建一个指定 dtype 的 Series 对象 ser
        ser = Series([10, 11, 12], dtype=dtype)
        # 创建另一个 Series 对象 other，带有指定的 other 和 index
        other = Series(other, index=[1, 3])
        # 使用 assert_produces_warning 断言是否引发 warn 警告，匹配 "item of incompatible dtype"
        with tm.assert_produces_warning(warn, match="item of incompatible dtype"):
            # 使用 other 更新 ser
            ser.update(other)

        # 断言 ser 和 expected 是否相等
        tm.assert_series_equal(ser, expected)
    @pytest.mark.parametrize(
        "values, other, expected",
        [
            # update by key
            (
                {"a": 1, "b": 2, "c": 3, "d": 4},
                {"b": 5, "c": np.nan},
                {"a": 1, "b": 5, "c": 3, "d": 4},
            ),
            # update by position
            ([1, 2, 3, 4], [np.nan, 5, 1], [1, 5, 1, 4]),
        ],
    )
    # 定义测试方法，使用参数化测试进行多组测试数据验证
    def test_update_from_non_series(self, values, other, expected):
        # GH 33215
        # 创建 Series 对象，使用传入的 values
        series = Series(values)
        # 调用 Series 的 update 方法，更新数据
        series.update(other)
        # 创建期望的 Series 对象，用于与更新后的结果进行比较
        expected = Series(expected)
        # 使用测试模块中的 assert_series_equal 方法比较两个 Series 对象是否相等
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize(
        "data, other, expected, dtype",
        [
            # 更新字符串数组中的值
            (["a", None], [None, "b"], ["a", "b"], "string[python]"),
            # 使用 pyarrow 更新字符串数组中的值
            pytest.param(
                ["a", None],
                [None, "b"],
                ["a", "b"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            # 更新整数数组中的值
            ([1, None], [None, 2], [1, 2], "Int64"),
            # 更新布尔数组中的值
            ([True, None], [None, False], [True, False], "boolean"),
            # 使用类别数据类型更新数组中的值
            (
                ["a", None],
                [None, "b"],
                ["a", "b"],
                CategoricalDtype(categories=["a", "b"]),
            ),
            # 更新日期时间数组中的值
            (
                [Timestamp(year=2020, month=1, day=1, tz="Europe/London"), NaT],
                [NaT, Timestamp(year=2020, month=1, day=1, tz="Europe/London")],
                [Timestamp(year=2020, month=1, day=1, tz="Europe/London")] * 2,
                "datetime64[ns, Europe/London]",
            ),
        ],
    )
    # 定义测试方法，使用参数化测试进行多组测试数据验证
    def test_update_extension_array_series(self, data, other, expected, dtype):
        # 创建 Series 对象，使用指定的数据类型
        result = Series(data, dtype=dtype)
        # 创建另一个 Series 对象，使用指定的数据类型
        other = Series(other, dtype=dtype)
        # 创建期望的 Series 对象，使用指定的数据类型
        expected = Series(expected, dtype=dtype)

        # 调用 Series 的 update 方法，更新数据
        result.update(other)
        # 使用测试模块中的 assert_series_equal 方法比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    def test_update_with_categorical_type(self):
        # GH 25744
        # 创建类别数据类型对象
        dtype = CategoricalDtype(["a", "b", "c", "d"])
        # 创建 Series 对象，使用指定的数据和索引
        s1 = Series(["a", "b", "c"], index=[1, 2, 3], dtype=dtype)
        # 创建另一个 Series 对象，使用指定的数据和索引
        s2 = Series(["b", "a"], index=[1, 2], dtype=dtype)
        # 调用 Series 的 update 方法，更新数据
        s1.update(s2)
        # 设置结果为 s1
        result = s1
        # 创建期望的 Series 对象，用于与更新后的结果进行比较
        expected = Series(["b", "a", "c"], index=[1, 2, 3], dtype=dtype)
        # 使用测试模块中的 assert_series_equal 方法比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
```