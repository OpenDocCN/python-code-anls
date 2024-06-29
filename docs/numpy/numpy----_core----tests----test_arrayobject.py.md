# `.\numpy\numpy\_core\tests\test_arrayobject.py`

```py
# 导入 pytest 库，用于测试框架
import pytest

# 导入 numpy 库，并导入 assert_array_equal 函数用于比较数组是否相等
import numpy as np
from numpy.testing import assert_array_equal


# 定义测试函数，验证当数组维度小于 2 时，转置操作会引发 ValueError 异常
def test_matrix_transpose_raises_error_for_1d():
    msg = "matrix transpose with ndim < 2 is undefined"
    arr = np.arange(48)
    # 使用 pytest 的断言来验证是否抛出预期异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        arr.mT


# 定义测试函数，验证二维数组的转置操作
def test_matrix_transpose_equals_transpose_2d():
    arr = np.arange(48).reshape((6, 8))
    # 使用 assert_array_equal 函数验证 arr.T 是否等于 arr.mT
    assert_array_equal(arr.T, arr.mT)


# 定义多个数组形状用于参数化测试
ARRAY_SHAPES_TO_TEST = (
    (5, 2),
    (5, 2, 3),
    (5, 2, 3, 4),
)


# 参数化测试函数，验证 swapaxes 函数的结果与自定义转置操作 mT 相等
@pytest.mark.parametrize("shape", ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    num_of_axes = len(shape)
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    mT = arr.mT  # 执行自定义转置操作 mT
    # 使用 assert_array_equal 函数验证 tgt 是否等于 mT
    assert_array_equal(tgt, mT)
```