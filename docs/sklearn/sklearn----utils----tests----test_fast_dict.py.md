# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_fast_dict.py`

```
"""Test fast_dict."""

# 导入必要的库和模块
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# 导入需要测试的类和函数
from sklearn.utils._fast_dict import IntFloatDict, argmin


# 定义测试函数，测试 IntFloatDict 类的基本功能
def test_int_float_dict():
    # 使用随机数生成测试数据
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp))
    values = rng.rand(len(keys))

    # 创建 IntFloatDict 对象并进行基本断言测试
    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value
    assert len(d) == len(keys)

    # 测试在字典中添加新的键值对
    d.append(120, 3.0)
    assert d[120] == 3.0
    assert len(d) == len(keys) + 1

    # 测试大规模添加键值对后的读取
    for i in range(2000):
        d.append(i + 1000, 4.0)
    assert d[1100] == 4.0


# 测试 IntFloatDict 类中的 argmin 函数实现
def test_int_float_dict_argmin():
    # 使用连续的整数和浮点数创建 IntFloatDict 对象
    keys = np.arange(100, dtype=np.intp)
    values = np.arange(100, dtype=np.float64)
    d = IntFloatDict(keys, values)
    
    # 测试 argmin 函数的返回结果
    assert argmin(d) == (0, 0)


# 测试将 IntFloatDict 转换为数组的功能
def test_to_arrays():
    # 创建包含整数键和浮点数值的数组
    keys_in = np.array([1, 2, 3], dtype=np.intp)
    values_in = np.array([4, 5, 6], dtype=np.float64)

    # 创建 IntFloatDict 对象，并将其转换为数组
    d = IntFloatDict(keys_in, values_in)
    keys_out, values_out = d.to_arrays()

    # 进行转换后数组的断言测试
    assert keys_out.dtype == keys_in.dtype
    assert values_in.dtype == values_out.dtype
    assert_array_equal(keys_out, keys_in)
    assert_allclose(values_out, values_in)
```