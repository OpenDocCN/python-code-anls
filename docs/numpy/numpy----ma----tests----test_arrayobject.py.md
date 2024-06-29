# `.\numpy\numpy\ma\tests\test_arrayobject.py`

```py
# 导入 pytest 库，用于单元测试
import pytest

# 导入 numpy 库，并从中导入所需的模块
import numpy as np
# 从 numpy.ma 模块中导入 masked_array 类
from numpy.ma import masked_array
# 从 numpy.testing 模块中导入 assert_array_equal 函数，用于比较数组是否相等

# 定义一个测试函数，验证在 1 维情况下矩阵转置是否引发错误
def test_matrix_transpose_raises_error_for_1d():
    # 定义错误消息字符串
    msg = "matrix transpose with ndim < 2 is undefined"
    # 创建一个 masked_array 对象，包含数据和掩码
    ma_arr = masked_array(data=[1, 2, 3, 4, 5, 6],
                          mask=[1, 0, 1, 1, 1, 0])
    # 使用 pytest 检查是否引发 ValueError 异常，并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        ma_arr.mT  # 尝试访问矩阵的转置属性


# 定义一个测试函数，验证二维情况下矩阵转置是否正确
def test_matrix_transpose_equals_transpose_2d():
    # 创建一个 masked_array 对象，包含二维数据和掩码
    ma_arr = masked_array(data=[[1, 2, 3], [4, 5, 6]],
                          mask=[[1, 0, 1], [1, 1, 0]])
    # 使用 assert_array_equal 函数验证转置操作是否正确
    assert_array_equal(ma_arr.T, ma_arr.mT)


# 定义要测试的数组形状的元组
ARRAY_SHAPES_TO_TEST = (
    (5, 2),
    (5, 2, 3),
    (5, 2, 3, 4),
)

# 使用 pytest 的 parametrize 装饰器进行参数化测试
@pytest.mark.parametrize("shape", ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    # 获取数组的维度数
    num_of_axes = len(shape)
    # 创建一个连续的向量，并广播到指定的形状
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)

    # 使用随机数生成器创建一个与 arr 形状相同的随机掩码
    rng = np.random.default_rng(42)
    mask = rng.choice([0, 1], size=shape)
    # 创建一个 masked_array 对象，包含数据和随机掩码
    ma_arr = masked_array(data=arr, mask=mask)

    # 执行 np.swapaxes 操作，将倒数第二个轴和最后一个轴交换
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    # 使用 assert_array_equal 函数验证交换轴后的结果是否与 masked_array 的转置相等
    assert_array_equal(tgt, ma_arr.mT)
```