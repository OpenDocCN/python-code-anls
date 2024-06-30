# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_bitset.py`

```
# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_allclose

# 导入需要测试的函数和类
from sklearn.ensemble._hist_gradient_boosting._bitset import (
    in_bitset_memoryview,
    set_bitset_memoryview,
    set_raw_bitset_from_binned_bitset,
)
from sklearn.ensemble._hist_gradient_boosting.common import X_DTYPE

# 定义测试函数，使用 pytest 的 parametrize 装饰器定义多组参数化测试
@pytest.mark.parametrize(
    "values_to_insert, expected_bitset",
    [
        ([0, 4, 33], np.array([2**0 + 2**4, 2**1, 0], dtype=np.uint32)),
        ([31, 32, 33, 79], np.array([2**31, 2**0 + 2**1, 2**15], dtype=np.uint32)),
    ],
)
def test_set_get_bitset(values_to_insert, expected_bitset):
    # 初始化一个全零的 uint32 数组，长度为3
    n_32bits_ints = 3
    bitset = np.zeros(n_32bits_ints, dtype=np.uint32)
    
    # 设置位集合的值
    for value in values_to_insert:
        set_bitset_memoryview(bitset, value)
    
    # 断言设置后的位集合与预期结果相近
    assert_allclose(expected_bitset, bitset)
    
    # 遍历所有可能的值，验证是否存在于位集合中
    for value in range(32 * n_32bits_ints):
        if value in values_to_insert:
            assert in_bitset_memoryview(bitset, value)
        else:
            assert not in_bitset_memoryview(bitset, value)

# 定义测试函数，测试从分类后的位集合获取原始位集合
@pytest.mark.parametrize(
    "raw_categories, binned_cat_to_insert, expected_raw_bitset",
    [
        ([3, 4, 5, 10, 31, 32, 43], [0, 2, 4, 5, 6], [2**3 + 2**5 + 2**31, 2**0 + 2**11]),
        ([3, 33, 50, 52], [1, 3], [0, 2**1 + 2**20]),
    ],
)
def test_raw_bitset_from_binned_bitset(
    raw_categories, binned_cat_to_insert, expected_raw_bitset
):
    # 初始化一个全零的 uint32 数组，长度为2，用于存储分类后的位集合和原始位集合
    binned_bitset = np.zeros(2, dtype=np.uint32)
    raw_bitset = np.zeros(2, dtype=np.uint32)
    
    # 将原始分类转换为 numpy 数组
    raw_categories = np.asarray(raw_categories, dtype=X_DTYPE)
    
    # 设置分类后的位集合的值
    for val in binned_cat_to_insert:
        set_bitset_memoryview(binned_bitset, val)
    
    # 根据分类后的位集合和原始分类计算原始位集合
    set_raw_bitset_from_binned_bitset(raw_bitset, binned_bitset, raw_categories)
    
    # 断言计算得到的原始位集合与预期结果相近
    assert_allclose(expected_raw_bitset, raw_bitset)
    
    # 遍历所有原始分类值，验证是否存在于计算得到的原始位集合中
    for binned_cat_val, raw_cat_val in enumerate(raw_categories):
        if binned_cat_val in binned_cat_to_insert:
            assert in_bitset_memoryview(raw_bitset, raw_cat_val)
        else:
            assert not in_bitset_memoryview(raw_bitset, raw_cat_val)
```