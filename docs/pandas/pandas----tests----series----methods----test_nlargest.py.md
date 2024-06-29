# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_nlargest.py`

```
"""
Note: for naming purposes, most tests are title with as e.g. "test_nlargest_foo"
but are implicitly also testing nsmallest_foo.
"""

# 引入必要的库
import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


def assert_check_nselect_boundary(vals, dtype, method):
    # helper function for 'test_boundary_{dtype}' tests
    # 创建一个 Series 对象，使用给定的值和数据类型
    ser = Series(vals, dtype=dtype)
    # 调用指定的方法（nlargest 或 nsmallest）对 Series 进行操作
    result = getattr(ser, method)(3)
    # 根据方法确定预期的索引顺序
    expected_idxr = [0, 1, 2] if method == "nsmallest" else [3, 2, 1]
    # 根据预期索引顺序获取预期结果的子集
    expected = ser.loc[expected_idxr]
    # 使用测试框架验证结果与预期是否相等
    tm.assert_series_equal(result, expected)


class TestSeriesNLargestNSmallest:
    @pytest.mark.parametrize(
        "r",
        [
            Series([3.0, 2, 1, 2, "5"], dtype="object"),
            Series([3.0, 2, 1, 2, 5], dtype="object"),
            # not supported on some archs
            # Series([3., 2, 1, 2, 5], dtype='complex256'),
            Series([3.0, 2, 1, 2, 5], dtype="complex128"),
            Series(list("abcde")),
            Series(list("abcde"), dtype="category"),
        ],
    )
    @pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
    @pytest.mark.parametrize("arg", [2, 5, 0, -1])
    def test_nlargest_error(self, r, method, arg):
        dt = r.dtype
        # 准备错误消息，指出无法用指定方法处理特定数据类型
        msg = f"Cannot use method 'n(largest|smallest)' with dtype {dt}"
        # 使用 pytest 的异常检测确保调用指定方法时会抛出 TypeError 异常，并且异常消息匹配预期
        with pytest.raises(TypeError, match=msg):
            getattr(r, method)(arg)

    @pytest.mark.parametrize(
        "data",
        [
            pd.to_datetime(["2003", "2002", "2001", "2002", "2005"]),
            pd.to_datetime(["2003", "2002", "2001", "2002", "2005"], utc=True),
            pd.to_timedelta(["3D", "2D", "1D", "2D", "5D"]),
            np.array([3, 2, 1, 2, 5], dtype="int8"),
            np.array([3, 2, 1, 2, 5], dtype="int16"),
            np.array([3, 2, 1, 2, 5], dtype="int32"),
            np.array([3, 2, 1, 2, 5], dtype="int64"),
            np.array([3, 2, 1, 2, 5], dtype="uint8"),
            np.array([3, 2, 1, 2, 5], dtype="uint16"),
            np.array([3, 2, 1, 2, 5], dtype="uint32"),
            np.array([3, 2, 1, 2, 5], dtype="uint64"),
            np.array([3, 2, 1, 2, 5], dtype="float32"),
            np.array([3, 2, 1, 2, 5], dtype="float64"),
        ],
    )
    # 测试 nsmallest 和 nlargest 方法的功能
    def test_nsmallest_nlargest(self, data):
        # 创建 Series 对象，用给定的数据填充
        ser = Series(data)

        # 断言返回 Series 中最小的 2 个元素，按默认方式排序
        tm.assert_series_equal(ser.nsmallest(2), ser.iloc[[2, 1]])
        # 断言返回 Series 中最小的 2 个元素，保留最后一个出现的元素
        tm.assert_series_equal(ser.nsmallest(2, keep="last"), ser.iloc[[2, 3]])

        # 创建空的 Series
        empty = ser.iloc[0:0]
        # 断言返回 Series 中最小的 0 个元素，应该返回空的 Series
        tm.assert_series_equal(ser.nsmallest(0), empty)
        # 断言返回 Series 中最小的 -1 个元素，应该返回空的 Series
        tm.assert_series_equal(ser.nsmallest(-1), empty)
        # 断言返回 Series 中最大的 0 个元素，应该返回空的 Series
        tm.assert_series_equal(ser.nlargest(0), empty)
        # 断言返回 Series 中最大的 -1 个元素，应该返回空的 Series
        tm.assert_series_equal(ser.nlargest(-1), empty)

        # 断言返回 Series 中最小的 len(ser) 个元素，按默认方式排序
        tm.assert_series_equal(ser.nsmallest(len(ser)), ser.sort_values())
        # 断言返回 Series 中最小的 len(ser) + 1 个元素，按默认方式排序
        tm.assert_series_equal(ser.nsmallest(len(ser) + 1), ser.sort_values())
        # 断言返回 Series 中最大的 len(ser) 个元素，按默认方式排序
        tm.assert_series_equal(ser.nlargest(len(ser)), ser.iloc[[4, 0, 1, 3, 2]])
        # 断言返回 Series 中最大的 len(ser) + 1 个元素，按默认方式排序
        tm.assert_series_equal(ser.nlargest(len(ser) + 1), ser.iloc[[4, 0, 1, 3, 2]])

    # 测试 nlargest 方法的其他用例
    def test_nlargest_misc(self):
        # 创建包含 NaN 的 Series
        ser = Series([3.0, np.nan, 1, 2, 5])
        # 断言返回 Series 中最大的元素，默认返回最大的 5 个元素
        result = ser.nlargest()
        expected = ser.iloc[[4, 0, 3, 2, 1]]
        tm.assert_series_equal(result, expected)
        # 断言返回 Series 中最小的元素，默认返回最小的 5 个元素
        result = ser.nsmallest()
        expected = ser.iloc[[2, 3, 0, 4, 1]]
        tm.assert_series_equal(result, expected)

        # 检查 keep 参数必须为 "first" 或 "last"，否则应该引发 ValueError 异常
        msg = 'keep must be either "first", "last"'
        with pytest.raises(ValueError, match=msg):
            ser.nsmallest(keep="invalid")
        with pytest.raises(ValueError, match=msg):
            ser.nlargest(keep="invalid")

        # GH#15297 测试特定索引的情况
        ser = Series([1] * 5, index=[1, 2, 3, 4, 5])
        expected_first = Series([1] * 3, index=[1, 2, 3])
        expected_last = Series([1] * 3, index=[5, 4, 3])

        # 断言返回 Series 中最小的 3 个元素，默认返回最小的 3 个元素
        result = ser.nsmallest(3)
        tm.assert_series_equal(result, expected_first)

        # 断言返回 Series 中最小的 3 个元素，保留最后一个出现的元素
        result = ser.nsmallest(3, keep="last")
        tm.assert_series_equal(result, expected_last)

        # 断言返回 Series 中最大的 3 个元素，默认返回最大的 3 个元素
        result = ser.nlargest(3)
        tm.assert_series_equal(result, expected_first)

        # 断言返回 Series 中最大的 3 个元素，保留最后一个出现的元素
        result = ser.nlargest(3, keep="last")
        tm.assert_series_equal(result, expected_last)

    # 使用参数化测试，测试 nlargest 和 nsmallest 方法的边界情况
    @pytest.mark.parametrize("n", range(1, 5))
    def test_nlargest_n(self, n):
        # GH 13412
        ser = Series([1, 4, 3, 2], index=[0, 0, 1, 1])
        # 断言返回 Series 中最大的 n 个元素，按降序排序
        result = ser.nlargest(n)
        expected = ser.sort_values(ascending=False).head(n)
        tm.assert_series_equal(result, expected)

        # 断言返回 Series 中最小的 n 个元素，按升序排序
        result = ser.nsmallest(n)
        expected = ser.sort_values().head(n)
        tm.assert_series_equal(result, expected)

    # 测试 nlargest 和 nsmallest 方法在整数边界情况下的行为
    def test_nlargest_boundary_integer(self, nselect_method, any_int_numpy_dtype):
        # GH#21426
        # 获取给定整数类型的信息
        dtype_info = np.iinfo(any_int_numpy_dtype)
        # 获取整数类型的最小值和最大值
        min_val, max_val = dtype_info.min, dtype_info.max
        # 构建一个包含整数边界值的列表
        vals = [min_val, min_val + 1, max_val - 1, max_val]
        # 调用函数检查 nlargest 和 nsmallest 方法在给定边界值情况下的行为
        assert_check_nselect_boundary(vals, any_int_numpy_dtype, nselect_method)
    def test_nlargest_boundary_float(self, nselect_method, float_numpy_dtype):
        # GH#21426
        # 获取浮点数类型的信息
        dtype_info = np.finfo(float_numpy_dtype)
        # 获取最小值和最大值
        min_val, max_val = dtype_info.min, dtype_info.max
        # 获取最小值和最小值相邻的浮点数，作为第二小和第二大的值
        min_2nd, max_2nd = np
```