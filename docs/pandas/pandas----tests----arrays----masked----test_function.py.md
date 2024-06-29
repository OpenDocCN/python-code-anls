# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked\test_function.py`

```
import numpy as np
import pytest

# 从 pandas 库中导入用于判断是否为整数数据类型的函数
from pandas.core.dtypes.common import is_integer_dtype

# 导入 pandas 库并使用别名 pd
import pandas as pd
# 导入 pandas 测试工具模块并使用别名 tm
import pandas._testing as tm
# 从 pandas 核心数组中导入 BaseMaskedArray 类
from pandas.core.arrays import BaseMaskedArray

# 创建包含不同整数和浮点数类型的数组列表
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [
    pd.array([0.141, -0.268, 5.895, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES
]

# 定义名为 'data' 的 pytest fixture，参数化返回不同整数和浮点数类型的数组
@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    """
    Fixture returning parametrized 'data' array with different integer and
    floating point types
    """
    return request.param

# 定义名为 'numpy_dtype' 的 pytest fixture，从 'data' 输入数组返回 numpy 数据类型
@pytest.fixture
def numpy_dtype(data):
    """
    Fixture returning numpy dtype from 'data' input array.
    """
    # 对于整数数据类型，必须转换为浮点数
    if is_integer_dtype(data):
        numpy_dtype = float
    else:
        numpy_dtype = data.dtype.type
    return numpy_dtype

# 定义名为 'test_round' 的测试函数，测试数据的四舍五入功能
def test_round(data, numpy_dtype):
    # 没有参数的情况
    result = data.round()
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None)), dtype=data.dtype
    )
    tm.assert_extension_array_equal(result, expected)

    # 指定小数位数的情况
    result = data.round(decimals=2)
    expected = pd.array(
        np.round(data.to_numpy(dtype=numpy_dtype, na_value=None), decimals=2),
        dtype=data.dtype,
    )
    tm.assert_extension_array_equal(result, expected)

# 定义名为 'test_tolist' 的测试函数，测试数据转换为列表的功能
def test_tolist(data):
    result = data.tolist()
    expected = list(data)
    tm.assert_equal(result, expected)

# 定义名为 'test_to_numpy' 的测试函数，测试自定义字符串数组的 to_numpy 方法
def test_to_numpy():
    # GH#56991

    # 自定义的字符串数组类 MyStringArray，继承自 BaseMaskedArray
    class MyStringArray(BaseMaskedArray):
        dtype = pd.StringDtype()  # 指定数据类型为字符串
        _dtype_cls = pd.StringDtype  # 指定数据类型类为 pd.StringDtype
        _internal_fill_value = pd.NA  # 内部填充值为 pd.NA

    # 创建 MyStringArray 对象，包含值数组和掩码数组
    arr = MyStringArray(
        values=np.array(["a", "b", "c"]), mask=np.array([False, True, False])
    )
    result = arr.to_numpy()  # 调用自定义数组对象的 to_numpy 方法
    expected = np.array(["a", pd.NA, "c"])  # 预期的 numpy 数组结果
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等
```