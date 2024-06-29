# `.\numpy\numpy\matrixlib\tests\test_regression.py`

```py
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_, assert_equal, assert_raises  # 导入NumPy测试模块的断言方法

class TestRegression:
    def test_kron_matrix(self):
        # Ticket #71
        x = np.matrix('[1 0; 1 0]')  # 创建一个2x2的矩阵对象x
        assert_equal(type(np.kron(x, x)), type(x))  # 断言np.kron(x, x)的类型与x相同

    def test_matrix_properties(self):
        # Ticket #125
        a = np.matrix([1.0], dtype=float)  # 创建一个浮点数矩阵a，包含单个元素1.0
        assert_(type(a.real) is np.matrix)  # 断言a的实部的类型为np.matrix
        assert_(type(a.imag) is np.matrix)  # 断言a的虚部的类型为np.matrix
        c, d = np.matrix([0.0]).nonzero()  # 获取矩阵[0.0]的非零元素索引，分别赋值给c和d
        assert_(type(c) is np.ndarray)  # 断言c的类型为np.ndarray
        assert_(type(d) is np.ndarray)  # 断言d的类型为np.ndarray

    def test_matrix_multiply_by_1d_vector(self):
        # Ticket #473
        def mul():
            np.asmatrix(np.eye(2))*np.ones(2)  # 对单位矩阵np.eye(2)乘以全1向量np.ones(2)

        assert_raises(ValueError, mul)  # 断言mul函数会引发ValueError异常

    def test_matrix_std_argmax(self):
        # Ticket #83
        x = np.asmatrix(np.random.uniform(0, 1, (3, 3)))  # 创建一个3x3的随机浮点数矩阵x
        assert_equal(x.std().shape, ())  # 断言x的标准差的形状为标量
        assert_equal(x.argmax().shape, ())  # 断言x的最大值索引的形状为标量
```