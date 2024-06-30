# `D:\src\scipysrc\sympy\sympy\physics\tests\test_physics_matrices.py`

```
from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy


def test_parallel_axis_theorem():
    # 测试平行轴定理矩阵，通过与测试矩阵比较来验证

    # 第一种情况，所有方向上都为1
    mat1 = Matrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
    assert pat_matrix(1, 1, 1, 1) == mat1
    assert pat_matrix(2, 1, 1, 1) == 2*mat1

    # 第二种情况，只有x方向为1，其余为0
    mat2 = Matrix(((0, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert pat_matrix(1, 1, 0, 0) == mat2
    assert pat_matrix(2, 1, 0, 0) == 2*mat2

    # 第三种情况，只有y方向为1，其余为0
    mat3 = Matrix(((1, 0, 0), (0, 0, 0), (0, 0, 1)))
    assert pat_matrix(1, 0, 1, 0) == mat3
    assert pat_matrix(2, 0, 1, 0) == 2*mat3

    # 第四种情况，只有z方向为1，其余为0
    mat4 = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 0)))
    assert pat_matrix(1, 0, 0, 1) == mat4
    assert pat_matrix(2, 0, 0, 1) == 2*mat4


def test_Pauli():
    # 测试Pauli和Dirac矩阵，以及在实际情况下通用的Matrix类的正确性

    sigma1 = msigma(1)
    sigma2 = msigma(2)
    sigma3 = msigma(3)

    assert sigma1 == sigma1
    assert sigma1 != sigma2

    # sigma*I -> I*sigma (参见＃354)
    assert sigma1*sigma2 == sigma3*I
    assert sigma3*sigma1 == sigma2*I
    assert sigma2*sigma3 == sigma1*I

    assert sigma1*sigma1 == eye(2)
    assert sigma2*sigma2 == eye(2)
    assert sigma3*sigma3 == eye(2)

    assert sigma1*2*sigma1 == 2*eye(2)
    assert sigma1*sigma3*sigma1 == -sigma3


def test_Dirac():
    gamma0 = mgamma(0)
    gamma1 = mgamma(1)
    gamma2 = mgamma(2)
    gamma3 = mgamma(3)
    gamma5 = mgamma(5)

    # gamma*I -> I*gamma (参见＃354)
    assert gamma5 == gamma0 * gamma1 * gamma2 * gamma3 * I
    assert gamma1 * gamma2 + gamma2 * gamma1 == zeros(4)
    assert gamma0 * gamma0 == eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 != eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 == eye(4) * minkowski_tensor[2, 2]

    assert mgamma(5, True) == \
        mgamma(0, True)*mgamma(1, True)*mgamma(2, True)*mgamma(3, True)*I


def test_mdft():
    # 使用warns_deprecated_sympy()上下文管理器测试mdft函数

    with warns_deprecated_sympy():
        assert mdft(1) == Matrix([[1]])
    with warns_deprecated_sympy():
        assert mdft(2) == 1/sqrt(2)*Matrix([[1,1],[1,-1]])
    with warns_deprecated_sympy():
        assert mdft(4) == Matrix([[S.Half,  S.Half,  S.Half, S.Half],
                                  [S.Half, -I/2, Rational(-1,2),  I/2],
                                  [S.Half, Rational(-1,2),  S.Half, Rational(-1,2)],
                                  [S.Half,  I/2, Rational(-1,2), -I/2]])
```