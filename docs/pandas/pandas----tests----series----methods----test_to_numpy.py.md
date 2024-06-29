# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_to_numpy.py`

```
# 导入所需的库和模块
import numpy as np
import pytest

# 导入 pandas 库中的测试辅助模块
import pandas.util._test_decorators as td

# 从 pandas 库中导入特定的对象：NA, Series, Timedelta
from pandas import (
    NA,
    Series,
    Timedelta,
)

# 导入 pandas 内部测试工具模块
import pandas._testing as tm


# 使用 pytest 的参数化装饰器，定义测试函数 test_to_numpy_na_value，参数 dtype 可选值为 "int64" 或 "float64"
@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_to_numpy_na_value(dtype):
    # GH#48951
    # 创建一个包含 NA 值的 Series 对象
    ser = Series([1, 2, NA, 4])
    # 调用 Series 对象的 to_numpy 方法，将 NA 值转换为指定的值，并返回结果
    result = ser.to_numpy(dtype=dtype, na_value=0)
    # 创建一个预期的 numpy 数组，将 NA 值替换为指定的值，并指定数据类型为 dtype
    expected = np.array([1, 2, 0, 4], dtype=dtype)
    # 使用测试工具模块中的方法，断言 result 和 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)


# 定义测试函数 test_to_numpy_cast_before_setting_na
def test_to_numpy_cast_before_setting_na():
    # GH#50600
    # 创建一个包含整数的 Series 对象
    ser = Series([1])
    # 调用 Series 对象的 to_numpy 方法，先将数据类型转换为 np.float64，再设置 NA 值为 np.nan，并返回结果
    result = ser.to_numpy(dtype=np.float64, na_value=np.nan)
    # 创建一个预期的 numpy 数组，数据类型为 np.float64
    expected = np.array([1.0])
    # 使用测试工具模块中的方法，断言 result 和 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)


# 使用 pandas 内部测试辅助模块中的条件装饰器，定义测试函数 test_to_numpy_arrow_dtype_given
@td.skip_if_no("pyarrow")
def test_to_numpy_arrow_dtype_given():
    # GH#57121
    # 创建一个包含 NA 值的 Series 对象，数据类型为 "int64[pyarrow]"
    ser = Series([1, NA], dtype="int64[pyarrow]")
    # 调用 Series 对象的 to_numpy 方法，将数据类型转换为 "float64"，并返回结果
    result = ser.to_numpy(dtype="float64")
    # 创建一个预期的 numpy 数组，包含一个 NA 值，数据类型为 "float64"
    expected = np.array([1.0, np.nan])
    # 使用测试工具模块中的方法，断言 result 和 expected 是否相等
    tm.assert_numpy_array_equal(result, expected)


# 定义测试函数 test_astype_ea_int_to_td_ts
def test_astype_ea_int_to_td_ts():
    # GH#57093
    # 创建一个包含整数和 None 值的 Series 对象，数据类型为 "Int64"
    ser = Series([1, None], dtype="Int64")
    # 调用 Series 对象的 astype 方法，将数据类型转换为 "m8[ns]"，并返回结果
    result = ser.astype("m8[ns]")
    # 创建一个预期的 Series 对象，将 None 值转换为 Timedelta("nat")，数据类型为 "m8[ns]"
    expected = Series([1, Timedelta("nat")], dtype="m8[ns]")
    # 使用测试工具模块中的方法，断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 调用 Series 对象的 astype 方法，将数据类型转换为 "M8[ns]"，并返回结果
    result = ser.astype("M8[ns]")
    # 创建一个预期的 Series 对象，将 None 值转换为 Timedelta("nat")，数据类型为 "M8[ns]"
    expected = Series([1, Timedelta("nat")], dtype="M8[ns]")
    # 使用测试工具模块中的方法，断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```