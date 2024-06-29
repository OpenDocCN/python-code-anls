# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_construct_ndarray.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库，用于数组操作
import pytest  # 导入 Pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试辅助函数
from pandas.core.construction import sanitize_array  # 导入 Pandas 的数组清理函数


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义多组参数化测试
    "values, dtype, expected",  # 定义测试的参数和期望结果
    [
        ([1, 2, 3], None, np.array([1, 2, 3], dtype=np.int64)),  # 测试整数列表的情况
        (np.array([1, 2, 3]), None, np.array([1, 2, 3])),  # 测试 NumPy 数组的情况
        (["1", "2", None], None, np.array(["1", "2", None])),  # 测试字符串列表的情况
        (["1", "2", None], np.dtype("str"), np.array(["1", "2", None])),  # 测试指定字符串类型的情况
        ([1, 2, None], np.dtype("str"), np.array(["1", "2", None])),  # 测试指定字符串类型的情况
    ],
)
def test_construct_1d_ndarray_preserving_na(
    values, dtype, expected, using_infer_string
):
    # 调用 Pandas 的 sanitize_array 函数，清理输入数组
    result = sanitize_array(values, index=None, dtype=dtype)
    
    # 根据条件选择不同的断言方法进行测试验证
    if using_infer_string and expected.dtype == object and dtype is None:
        # 如果使用了 infer_string 并且期望结果是对象类型且 dtype 为 None，则使用 Pandas 的 assert_extension_array_equal 断言
        tm.assert_extension_array_equal(result, pd.array(expected))
    else:
        # 否则使用 NumPy 的 assert_numpy_array_equal 断言
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]"])
def test_construct_1d_ndarray_preserving_na_datetimelike(dtype):
    # 创建一个包含日期时间的 NumPy 数组
    arr = np.arange(5, dtype=np.int64).view(dtype)
    
    # 创建预期的结果数组，类型为对象类型
    expected = np.array(list(arr), dtype=object)
    
    # 验证预期结果中的每个元素是否与原数组中的类型相同
    assert all(isinstance(x, type(arr[0])) for x in expected)
    
    # 调用 Pandas 的 sanitize_array 函数，清理输入数组
    result = sanitize_array(arr, index=None, dtype=np.dtype(object))
    
    # 使用 NumPy 的 assert_numpy_array_equal 断言验证结果
    tm.assert_numpy_array_equal(result, expected)
```