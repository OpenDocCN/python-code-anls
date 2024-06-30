# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_weight_vector.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 sklearn.utils._weight_vector 模块中导入 WeightVector32 和 WeightVector64 类
from sklearn.utils._weight_vector import (
    WeightVector32,
    WeightVector64,
)

# 使用 pytest 的 parametrize 装饰器，定义参数化测试用例
@pytest.mark.parametrize(
    "dtype, WeightVector",
    [
        # 参数化测试用例，测试 np.float32 类型和 WeightVector32 类
        (np.float32, WeightVector32),
        # 参数化测试用例，测试 np.float64 类型和 WeightVector64 类
        (np.float64, WeightVector64),
    ],
)
def test_type_invariance(dtype, WeightVector):
    """Check the `dtype` consistency of `WeightVector`."""
    
    # 生成一个随机数组，转换成指定的 dtype 类型，作为权重数组
    weights = np.random.rand(100).astype(dtype)
    # 生成另一个随机数组，转换成指定的 dtype 类型，作为平均权重数组
    average_weights = np.random.rand(100).astype(dtype)

    # 使用给定的 WeightVector 类初始化一个权重向量对象
    weight_vector = WeightVector(weights, average_weights)

    # 断言权重向量对象中的权重数组的 dtype 应为指定的 dtype 类型
    assert np.asarray(weight_vector.w).dtype is np.dtype(dtype)
    # 断言权重向量对象中的平均权重数组的 dtype 应为指定的 dtype 类型
    assert np.asarray(weight_vector.aw).dtype is np.dtype(dtype)
```