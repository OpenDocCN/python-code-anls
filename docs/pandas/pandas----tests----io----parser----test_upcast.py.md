# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_upcast.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入需要的模块和函数
from pandas._libs.parsers import (
    _maybe_upcast,  # 导入可能需要提升的函数
    na_values,      # 导入缺失值列表
)

import pandas as pd
from pandas import NA  # 导入 pandas 的 NA 常量
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,   # 导入 Arrow 字符串数组
    BooleanArray,       # 导入布尔数组
    FloatingArray,      # 导入浮点数数组
    IntegerArray,       # 导入整数数组
    StringArray,        # 导入字符串数组
)


# 定义测试函数 test_maybe_upcast
def test_maybe_upcast(any_real_numpy_dtype):
    # GH#36712

    # 将任意实际的 NumPy 数据类型转换为 dtype 对象
    dtype = np.dtype(any_real_numpy_dtype)
    # 获取特定 dtype 的缺失值
    na_value = na_values[dtype]
    # 创建包含缺失值的 NumPy 数组
    arr = np.array([1, 2, na_value], dtype=dtype)
    # 调用 _maybe_upcast 函数，尝试提升数据类型
    result = _maybe_upcast(arr, use_dtype_backend=True)

    # 预期的掩码数组，标记缺失值的位置
    expected_mask = np.array([False, False, True])
    # 根据数据类型是否为整数，选择创建 IntegerArray 或 FloatingArray
    if issubclass(dtype.type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    # 使用 pandas 测试工具函数，断言 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_maybe_upcast_no_na
def test_maybe_upcast_no_na(any_real_numpy_dtype):
    # GH#36712
    # 创建不包含缺失值的 NumPy 数组
    arr = np.array([1, 2, 3], dtype=any_real_numpy_dtype)
    # 调用 _maybe_upcast 函数，尝试提升数据类型
    result = _maybe_upcast(arr, use_dtype_backend=True)

    # 预期的掩码数组，没有任何标记
    expected_mask = np.array([False, False, False])
    # 根据数据类型是否为整数，选择创建 IntegerArray 或 FloatingArray
    if issubclass(np.dtype(any_real_numpy_dtype).type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    # 使用 pandas 测试工具函数，断言 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_maybe_upcaste_bool
def test_maybe_upcaste_bool():
    # GH#36712
    # 定义布尔类型的数据类型
    dtype = np.bool_
    # 获取布尔类型的缺失值
    na_value = na_values[dtype]
    # 创建包含布尔类型的 NumPy 数组，使用 uint8 类型作为存储
    arr = np.array([True, False, na_value], dtype="uint8").view(dtype)
    # 调用 _maybe_upcast 函数，尝试提升数据类型
    result = _maybe_upcast(arr, use_dtype_backend=True)

    # 预期的掩码数组，标记缺失值的位置
    expected_mask = np.array([False, False, True])
    # 创建 BooleanArray，作为预期结果
    expected = BooleanArray(arr, mask=expected_mask)

    # 使用 pandas 测试工具函数，断言 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_maybe_upcaste_bool_no_nan
def test_maybe_upcaste_bool_no_nan():
    # GH#36712
    # 定义布尔类型的数据类型
    dtype = np.bool_
    # 创建不包含缺失值的布尔类型 NumPy 数组，使用 uint8 类型作为存储
    arr = np.array([True, False, False], dtype="uint8").view(dtype)
    # 调用 _maybe_upcast 函数，尝试提升数据类型
    result = _maybe_upcast(arr, use_dtype_backend=True)

    # 预期的掩码数组，没有任何标记
    expected_mask = np.array([False, False, False])
    # 创建 BooleanArray，作为预期结果
    expected = BooleanArray(arr, mask=expected_mask)

    # 使用 pandas 测试工具函数，断言 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_maybe_upcaste_all_nan
def test_maybe_upcaste_all_nan():
    # GH#36712
    # 定义整数类型的数据类型
    dtype = np.int64
    # 获取整数类型的缺失值
    na_value = na_values[dtype]
    # 创建包含全部为缺失值的整数类型 NumPy 数组
    arr = np.array([na_value, na_value], dtype=dtype)
    # 调用 _maybe_upcast 函数，尝试提升数据类型
    result = _maybe_upcast(arr, use_dtype_backend=True)

    # 预期的掩码数组，标记全部为缺失值
    expected_mask = np.array([True, True])
    # 创建 IntegerArray，作为预期结果
    expected = IntegerArray(arr, mask=expected_mask)

    # 使用 pandas 测试工具函数，断言 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)


# 定义使用参数化测试的函数 test_maybe_upcast_object
@pytest.mark.parametrize("val", [na_values[np.object_], "c"])
def test_maybe_upcast_object(val, string_storage):
    # GH#36712
    # 导入必要的模块 pyarrow
    pa = pytest.importorskip("pyarrow")
    # 设置 pandas 上下文管理器，配置字符串存储模式为指定的 string_storage
    with pd.option_context("mode.string_storage", string_storage):
        # 创建一个包含字符串数组和指定值的 NumPy 数组，数据类型为 np.object_
        arr = np.array(["a", "b", val], dtype=np.object_)
        # 使用 _maybe_upcast 函数处理数组，可能会提升数组类型，使用 dtype 后端
        result = _maybe_upcast(arr, use_dtype_backend=True)

        # 根据 string_storage 的值进行条件判断
        if string_storage == "python":
            # 如果 string_storage 是 "python"，根据 val 的值设置预期的字符串值或 NA
            exp_val = "c" if val == "c" else NA
            # 创建一个 StringArray 对象，包含指定数组和预期值的字符串数组
            expected = StringArray(np.array(["a", "b", exp_val], dtype=np.object_))
        else:
            # 如果 string_storage 不是 "python"，根据 val 的值设置预期的字符串值或 None
            exp_val = "c" if val == "c" else None
            # 创建一个 ArrowStringArray 对象，包含指定数组和预期值的 Arrow 字符串数组
            expected = ArrowStringArray(pa.array(["a", "b", exp_val]))

        # 使用 tm.assert_extension_array_equal 函数断言 result 和 expected 扩展数组相等
        tm.assert_extension_array_equal(result, expected)
```