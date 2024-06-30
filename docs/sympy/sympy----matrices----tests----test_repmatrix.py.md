# `D:\src\scipysrc\sympy\sympy\matrices\tests\test_repmatrix.py`

```
# 导入需要的模块和函数
from sympy.testing.pytest import raises  # 导入raises函数，用于测试异常情况
from sympy.matrices.exceptions import NonSquareMatrixError, NonInvertibleMatrixError  # 导入可能抛出的异常类

from sympy import Matrix, Rational  # 导入Matrix类和Rational类


def test_lll():
    # 创建一个5x5的整数矩阵A
    A = Matrix([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]])
    # 创建一个4x5的整数矩阵L
    L = Matrix([[ 10, -3,  -2,  8,  -4],
                [  3, -9,   8,  1, -11],
                [ -3, 13,  -9, -3,  -9],
                [-12, -7, -11,  9,  -1]])
    # 创建一个4x4的整数矩阵T
    T = Matrix([[ 10, -3,  -2,  8],
                [  3, -9,   8,  1],
                [ -3, 13,  -9, -3],
                [-12, -7, -11,  9]])
    # 断言LLL算法对A的应用结果等于L
    assert A.lll() == L
    # 断言LLL算法返回的变换结果是(L, T)
    assert A.lll_transform() == (L, T)
    # 断言T乘以A等于L
    assert T * A == L


def test_matrix_inv_mod():
    # 创建一个2x1的整数矩阵A
    A = Matrix(2, 1, [1, 0])
    # 测试非方阵求逆的情况，期待引发NonSquareMatrixError异常
    raises(NonSquareMatrixError, lambda: A.inv_mod(2))

    # 创建一个2x2的整数矩阵A
    A = Matrix(2, 2, [1, 0, 0, 0])
    # 测试不可逆矩阵求逆的情况，期待引发NonInvertibleMatrixError异常
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(2))

    # 创建一个2x2的整数矩阵A
    A = Matrix(2, 2, [1, 2, 3, 4])
    # 创建一个2x2的整数矩阵Ai
    Ai = Matrix(2, 2, [1, 1, 0, 1])
    # 断言A模3的逆矩阵等于Ai
    assert A.inv_mod(3) == Ai

    # 创建一个2x2的整数矩阵A
    A = Matrix(2, 2, [1, 0, 0, 1])
    # 断言单位矩阵A模2的逆矩阵等于A
    assert A.inv_mod(2) == A

    # 创建一个3x3的整数矩阵A
    A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 测试不可逆矩阵求逆的情况，期待引发NonInvertibleMatrixError异常
    raises(NonInvertibleMatrixError, lambda: A.inv_mod(5))

    # 创建一个3x3的整数矩阵A
    A = Matrix(3, 3, [5, 1, 3, 2, 6, 0, 2, 1, 1])
    # 创建一个3x3的整数矩阵Ai
    Ai = Matrix(3, 3, [6, 8, 0, 1, 5, 6, 5, 6, 4])
    # 断言A模9的逆矩阵等于Ai
    assert A.inv_mod(9) == Ai

    # 创建一个3x3的整数矩阵A
    A = Matrix(3, 3, [1, 6, -3, 4, 1, -5, 3, -5, 5])
    # 创建一个3x3的整数矩阵Ai
    Ai = Matrix(3, 3, [4, 3, 3, 1, 2, 5, 1, 5, 1])
    # 断言A模6的逆矩阵等于Ai
    assert A.inv_mod(6) == Ai

    # 创建一个3x3的整数矩阵A
    A = Matrix(3, 3, [1, 6, 1, 4, 1, 5, 3, 2, 5])
    # 创建一个3x3的整数矩阵Ai
    Ai = Matrix(3, 3, [6, 0, 3, 6, 6, 4, 1, 6, 1])
    # 断言A模7的逆矩阵等于Ai
    assert A.inv_mod(7) == Ai

    # 创建一个2x2的有理数矩阵A
    A = Matrix([[1, 2], [3, Rational(3,4)]])
    # 测试有理数矩阵求逆的情况，期待引发ValueError异常
    raises(ValueError, lambda: A.inv_mod(2))

    # 创建一个2x2的整数矩阵A
    A = Matrix([[1, 2], [3, 4]])
    # 测试类型错误的情况，期待引发TypeError异常
    raises(TypeError, lambda: A.inv_mod(Rational(1, 2)))
```