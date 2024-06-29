# `.\numpy\numpy\tests\test_matlib.py`

```
import numpy as np  # 导入 NumPy 库，命名为 np
import numpy.matlib  # 导入 NumPy 的 matlib 模块
from numpy.testing import assert_array_equal, assert_  # 从 NumPy 的 testing 模块导入两个断言函数

def test_empty():
    x = numpy.matlib.empty((2,))  # 创建一个形状为 (2,) 的空矩阵 x
    assert_(isinstance(x, np.matrix))  # 断言 x 是 np.matrix 类型的对象
    assert_(x.shape, (1, 2))  # 断言 x 的形状为 (1, 2)，这里可能有误，应为 assert_array_equal(x.shape, (1, 2))

def test_ones():
    assert_array_equal(numpy.matlib.ones((2, 3)),
                       np.matrix([[ 1.,  1.,  1.],  # 断言 numpy.matlib.ones((2, 3)) 生成的矩阵与给定矩阵相等
                                 [ 1.,  1.,  1.]]))

    assert_array_equal(numpy.matlib.ones(2), np.matrix([[ 1.,  1.]]))  # 断言 numpy.matlib.ones(2) 生成的矩阵与给定矩阵相等

def test_zeros():
    assert_array_equal(numpy.matlib.zeros((2, 3)),
                       np.matrix([[ 0.,  0.,  0.],  # 断言 numpy.matlib.zeros((2, 3)) 生成的矩阵与给定矩阵相等
                                 [ 0.,  0.,  0.]]))

    assert_array_equal(numpy.matlib.zeros(2), np.matrix([[ 0.,  0.]]))  # 断言 numpy.matlib.zeros(2) 生成的矩阵与给定矩阵相等

def test_identity():
    x = numpy.matlib.identity(2, dtype=int)  # 创建一个 2x2 的单位矩阵 x，数据类型为整数
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))  # 断言 x 与给定的单位矩阵相等

def test_eye():
    xc = numpy.matlib.eye(3, k=1, dtype=int)  # 创建一个 3x3 的单位矩阵 xc，主对角线偏移为 1，数据类型为整数
    assert_array_equal(xc, np.matrix([[ 0,  1,  0],  # 断言 xc 与给定的单位矩阵相等
                                      [ 0,  0,  1],
                                      [ 0,  0,  0]]))
    assert xc.flags.c_contiguous  # 断言 xc 是 C 连续的
    assert not xc.flags.f_contiguous  # 断言 xc 不是 Fortran 连续的

    xf = numpy.matlib.eye(3, 4, dtype=int, order='F')  # 创建一个 3x4 的单位矩阵 xf，列序为 Fortran，数据类型为整数
    assert_array_equal(xf, np.matrix([[ 1,  0,  0,  0],  # 断言 xf 与给定的单位矩阵相等
                                      [ 0,  1,  0,  0],
                                      [ 0,  0,  1,  0]]))
    assert not xf.flags.c_contiguous  # 断言 xf 不是 C 连续的
    assert xf.flags.f_contiguous  # 断言 xf 是 Fortran 连续的

def test_rand():
    x = numpy.matlib.rand(3)  # 创建一个形状为 (3,) 的随机矩阵 x
    assert_(x.ndim == 2)  # 断言 x 是二维矩阵

def test_randn():
    x = np.matlib.randn(3)  # 创建一个形状为 (3,) 的标准正态分布随机矩阵 x
    assert_(x.ndim == 2)  # 断言 x 是二维矩阵

def test_repmat():
    a1 = np.arange(4)  # 创建一个从 0 到 3 的一维数组 a1
    x = numpy.matlib.repmat(a1, 2, 2)  # 对数组 a1 进行复制，复制为一个 2x2 的矩阵 x
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3],  # 给定的复制后的矩阵 y
                  [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)  # 断言 x 与给定的复制后的矩阵 y 相等
```