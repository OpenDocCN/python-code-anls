# `D:\src\scipysrc\pandas\pandas\tests\window\test_dtypes.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 错误处理模块中的 DataError 类
from pandas.errors import DataError

# 导入 pandas 中用于处理数据类型的通用函数 pandas_dtype
from pandas.core.dtypes.common import pandas_dtype

# 导入 pandas 库中的 DataFrame 和 Series 类
from pandas import (
    NA,
    DataFrame,
    Series,
)

# 导入 pandas 测试模块
import pandas._testing as tm

# gh-12373 : rolling functions error on float32 data
# make sure rolling functions works for different dtypes
#
# further note that we are only checking rolling for fully dtype
# compliance (though both expanding and ewm inherit)

# 定义一个函数，根据输入的数据类型和是否强制转换为整数来返回对应的 pandas 数据类型
def get_dtype(dtype, coerce_int=None):
    if coerce_int is False and "int" in dtype:
        return None
    return pandas_dtype(dtype)

# 定义一个 pytest fixture，用于在测试中参数化不同的数据类型
@pytest.fixture(
    params=[
        "object",
        "category",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "m8[ns]",
        "M8[ns]",
        "datetime64[ns, UTC]",
    ]
)
def dtypes(request):
    """Dtypes for window tests"""
    return request.param

# 使用 pytest.mark.parametrize 装饰器来参数化测试方法，定义多组测试数据
@pytest.mark.parametrize(
    "method, data, expected_data, coerce_int, min_periods",
    # 参数化测试数据，具体的测试数据将在具体的测试函数中传入
    [
        # 元组1：计算 'count' 函数在 np.arange(5) 上的结果
        ("count", np.arange(5), [1, 2, 2, 2, 2], True, 0),
        # 元组2：计算 'count' 函数在 np.arange(10, 0, -2) 上的结果
        ("count", np.arange(10, 0, -2), [1, 2, 2, 2, 2], True, 0),
        # 元组3：计算 'count' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("count", [0, 1, 2, np.nan, 4], [1, 2, 2, 1, 1], False, 0),
        # 元组4：计算 'max' 函数在 np.arange(5) 上的结果
        ("max", np.arange(5), [np.nan, 1, 2, 3, 4], True, None),
        # 元组5：计算 'max' 函数在 np.arange(10, 0, -2) 上的结果
        ("max", np.arange(10, 0, -2), [np.nan, 10, 8, 6, 4], True, None),
        # 元组6：计算 'max' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("max", [0, 1, 2, np.nan, 4], [np.nan, 1, 2, np.nan, np.nan], False, None),
        # 元组7：计算 'min' 函数在 np.arange(5) 上的结果
        ("min", np.arange(5), [np.nan, 0, 1, 2, 3], True, None),
        # 元组8：计算 'min' 函数在 np.arange(10, 0, -2) 上的结果
        ("min", np.arange(10, 0, -2), [np.nan, 8, 6, 4, 2], True, None),
        # 元组9：计算 'min' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("min", [0, 1, 2, np.nan, 4], [np.nan, 0, 1, np.nan, np.nan], False, None),
        # 元组10：计算 'sum' 函数在 np.arange(5) 上的结果
        ("sum", np.arange(5), [np.nan, 1, 3, 5, 7], True, None),
        # 元组11：计算 'sum' 函数在 np.arange(10, 0, -2) 上的结果
        ("sum", np.arange(10, 0, -2), [np.nan, 18, 14, 10, 6], True, None),
        # 元组12：计算 'sum' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("sum", [0, 1, 2, np.nan, 4], [np.nan, 1, 3, np.nan, np.nan], False, None),
        # 元组13：计算 'mean' 函数在 np.arange(5) 上的结果
        ("mean", np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None),
        # 元组14：计算 'mean' 函数在 np.arange(10, 0, -2) 上的结果
        ("mean", np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None),
        # 元组15：计算 'mean' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("mean", [0, 1, 2, np.nan, 4], [np.nan, 0.5, 1.5, np.nan, np.nan], False, None),
        # 元组16：计算 'std' 函数在 np.arange(5) 上的结果
        ("std", np.arange(5), [np.nan] + [np.sqrt(0.5)] * 4, True, None),
        # 元组17：计算 'std' 函数在 np.arange(10, 0, -2) 上的结果
        ("std", np.arange(10, 0, -2), [np.nan] + [np.sqrt(2)] * 4, True, None),
        # 元组18：计算 'std' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        (
            "std",
            [0, 1, 2, np.nan, 4],
            [np.nan] + [np.sqrt(0.5)] * 2 + [np.nan] * 2,
            False,
            None,
        ),
        # 元组19：计算 'var' 函数在 np.arange(5) 上的结果
        ("var", np.arange(5), [np.nan, 0.5, 0.5, 0.5, 0.5], True, None),
        # 元组20：计算 'var' 函数在 np.arange(10, 0, -2) 上的结果
        ("var", np.arange(10, 0, -2), [np.nan, 2, 2, 2, 2], True, None),
        # 元组21：计算 'var' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("var", [0, 1, 2, np.nan, 4], [np.nan, 0.5, 0.5, np.nan, np.nan], False, None),
        # 元组22：计算 'median' 函数在 np.arange(5) 上的结果
        ("median", np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None),
        # 元组23：计算 'median' 函数在 np.arange(10, 0, -2) 上的结果
        ("median", np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None),
        # 元组24：计算 'median' 函数在 [0, 1, 2, np.nan, 4] 上的结果
        ("median", [0, 1, 2, np.nan, 4], [np.nan, 0.5, 1.5, np.nan, np.nan], False, None),
    ],
# 定义一个用于测试 Series 数据类型和相关方法的函数
def test_series_dtypes(
    method, data, expected_data, coerce_int, dtypes, min_periods, step
):
    # 使用给定的数据和数据类型创建 Series 对象，并根据 coerce_int 参数强制转换数据类型
    ser = Series(data, dtype=get_dtype(dtypes, coerce_int=coerce_int))
    # 对 Series 对象进行滚动窗口处理，设置窗口大小为2，最小周期数为 min_periods，步长为 step
    rolled = ser.rolling(2, min_periods=min_periods, step=step)

    # 如果数据类型为日期时间类型，并且方法不是 "count"，则期望抛出 DataError 异常
    if dtypes in ("m8[ns]", "M8[ns]", "datetime64[ns, UTC]") and method != "count":
        # 准备异常信息
        msg = "No numeric types to aggregate"
        # 使用 pytest 框架断言抛出 DataError 异常，并匹配异常信息
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        # 否则，计算滚动窗口的方法结果
        result = getattr(rolled, method)()
        # 准备预期的结果，使用步长调整期望数据类型为 float64 的 Series 对象
        expected = Series(expected_data, dtype="float64")[::step]
        # 使用 pytest 的 assert_almost_equal 方法比较结果和预期值
        tm.assert_almost_equal(result, expected)


# 定义一个用于测试可空整数类型 Series 的函数
def test_series_nullable_int(any_signed_int_ea_dtype, step):
    # 创建一个包含整数和缺失值的 Series 对象，数据类型为任意可空整数类型
    ser = Series([0, 1, NA], dtype=any_signed_int_ea_dtype)
    # 计算滚动窗口大小为2的均值，使用步长进行结果调整
    result = ser.rolling(2, step=step).mean()
    # 准备预期的结果，使用步长调整期望数据类型为 float64 的 Series 对象
    expected = Series([np.nan, 0.5, np.nan])[::step]
    # 使用 pytest 的 assert_series_equal 方法比较结果和预期值
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化装饰器，定义一个用于测试 DataFrame 数据类型和相关方法的函数
@pytest.mark.parametrize(
    "method, expected_data, min_periods",
    [
        ("count", {0: Series([1, 2, 2, 2, 2]), 1: Series([1, 2, 2, 2, 2])}, 0),
        ("max", {0: Series([np.nan, 2, 4, 6, 8]), 1: Series([np.nan, 3, 5, 7, 9])}, None),
        ("min", {0: Series([np.nan, 0, 2, 4, 6]), 1: Series([np.nan, 1, 3, 5, 7])}, None),
        ("sum", {0: Series([np.nan, 2, 6, 10, 14]), 1: Series([np.nan, 4, 8, 12, 16])}, None),
        ("mean", {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])}, None),
        ("std", {0: Series([np.nan] + [np.sqrt(2)] * 4), 1: Series([np.nan] + [np.sqrt(2)] * 4)}, None),
        ("var", {0: Series([np.nan, 2, 2, 2, 2]), 1: Series([np.nan, 2, 2, 2, 2])}, None),
        ("median", {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])}, None),
    ],
)
def test_dataframe_dtypes(method, expected_data, dtypes, min_periods, step):
    # 创建一个包含 0 到 9 数字的 DataFrame 对象，数据类型由 dtypes 参数指定
    df = DataFrame(np.arange(10).reshape((5, 2)), dtype=get_dtype(dtypes))
    # 对 DataFrame 对象进行滚动窗口处理，设置窗口大小为2，最小周期数为 min_periods，步长为 step
    rolled = df.rolling(2, min_periods=min_periods, step=step)

    # 如果数据类型为日期时间类型，并且方法不是 "count"，则期望抛出 DataError 异常
    if dtypes in ("m8[ns]", "M8[ns]", "datetime64[ns, UTC]") and method != "count":
        # 准备异常信息
        msg = "Cannot aggregate non-numeric type"
        # 使用 pytest 框架断言抛出 DataError 异常，并匹配异常信息
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        # 否则，计算滚动窗口的方法结果
        result = getattr(rolled, method)()
        # 准备预期的结果，使用步长调整期望数据类型为 float64 的 DataFrame 对象
        expected = DataFrame(expected_data, dtype="float64")[::step]
        # 使用 pytest 的 assert_frame_equal 方法比较结果和预期值
        tm.assert_frame_equal(result, expected)
```