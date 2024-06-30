# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_miobase.py`

```
""" Testing miobase module
"""

# 导入所需模块和函数
import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._miobase import matdims

# 定义测试函数 test_matdims
def test_matdims():
    # 测试 matdims 函数对不同输入的维度识别能力

    # 测试 NumPy 标量
    assert_equal(matdims(np.array(1)), (1, 1))

    # 测试一维数组，1 个元素
    assert_equal(matdims(np.array([1])), (1, 1))

    # 测试一维数组，2 个元素
    assert_equal(matdims(np.array([1,2])), (2, 1))

    # 测试二维数组，列向量
    assert_equal(matdims(np.array([[2],[3]])), (2, 1))

    # 测试二维数组，行向量
    assert_equal(matdims(np.array([[2,3]])), (1, 2))

    # 测试三维数组，行向量
    assert_equal(matdims(np.array([[[2,3]]])), (1, 1, 2))

    # 测试空的一维数组
    assert_equal(matdims(np.array([])), (0, 0))

    # 测试空的二维数组
    assert_equal(matdims(np.array([[]])), (1, 0))

    # 测试空的三维数组
    assert_equal(matdims(np.array([[[]]])), (1, 1, 0))

    # 测试空的三维数组（通过 np.empty 构建）
    assert_equal(matdims(np.empty((1, 0, 1))), (1, 0, 1))

    # 测试可选参数 'row' 对一维数组的影响
    assert_equal(matdims(np.array([1,2]), 'row'), (1, 2))

    # 测试非法参数情况下是否会引发 ValueError 异常
    assert_raises(ValueError, matdims, np.array([1,2]), 'bizarre')

    # 测试空稀疏矩阵的形状
    from scipy.sparse import csr_matrix, csc_matrix
    assert_equal(matdims(csr_matrix(np.zeros((3, 3)))), (3, 3))
    assert_equal(matdims(csc_matrix(np.zeros((2, 2)))), (2, 2))
```