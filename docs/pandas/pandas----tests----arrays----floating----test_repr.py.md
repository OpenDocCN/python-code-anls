# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_repr.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及其特定的模块
import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


# 测试函数，测试给定 dtype 的类型
def test_dtypes(dtype):
    # smoke tests on auto dtype construction
    # 创建一个 numpy 的 dtype 对象，并检查其类型是否为浮点数类型
    np.dtype(dtype.type).kind == "f"
    # 断言检查 dtype 的名称不为空
    assert dtype.name is not None


# 参数化测试，验证 dtype 对象的字符串表示是否符合预期
@pytest.mark.parametrize(
    "dtype, expected",
    [(Float32Dtype(), "Float32Dtype()"), (Float64Dtype(), "Float64Dtype()")],
)
def test_repr_dtype(dtype, expected):
    # 断言检查 dtype 对象的 repr 字符串是否与期望的字符串一致
    assert repr(dtype) == expected


# 测试 pandas 中数组的字符串表示
def test_repr_array():
    # 创建一个包含浮点数和缺失值的 pandas 数组，并获取其字符串表示
    result = repr(pd.array([1.0, None, 3.0]))
    # 预期的字符串表示
    expected = "<FloatingArray>\n[1.0, <NA>, 3.0]\nLength: 3, dtype: Float64"
    # 断言检查结果是否与预期一致
    assert result == expected


# 测试长数组的字符串表示
def test_repr_array_long():
    # 创建一个包含浮点数和缺失值的长 pandas 数组
    data = pd.array([1.0, 2.0, None] * 1000)
    # 预期的长字符串表示
    expected = """<FloatingArray>
[ 1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,
 ...
 <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>]
Length: 3000, dtype: Float64"""
    # 获取数组的字符串表示
    result = repr(data)
    # 断言检查结果是否与预期一致
    assert result == expected


# 测试数据帧的字符串表示
def test_frame_repr(data_missing):
    # 使用包含缺失值的数据创建 pandas 数据帧
    df = pd.DataFrame({"A": data_missing})
    # 获取数据帧的字符串表示
    result = repr(df)
    # 预期的字符串表示
    expected = "      A\n0  <NA>\n1   0.1"
    # 断言检查结果是否与预期一致
    assert result == expected
```