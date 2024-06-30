# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_kronecker.py`

```
# 从 sympy 库中导入 Mod 类，用于模运算
from sympy.core.mod import Mod
# 从 sympy 库中导入虚数单位 I
from sympy.core.numbers import I
# 从 sympy 库中导入 symbols 函数，用于创建符号变量
from sympy.core.symbol import symbols
# 从 sympy 库中导入 floor 函数，用于向下取整操作
from sympy.functions.elementary.integers import floor
# 从 sympy 库中导入 Matrix 和 eye 函数，用于创建矩阵和单位矩阵
from sympy.matrices.dense import (Matrix, eye)
# 从 sympy 库中导入 MatrixSymbol 和 Identity 类，用于创建矩阵符号和单位矩阵符号
from sympy.matrices import MatrixSymbol, Identity
# 从 sympy 库中导入 det 和 trace 函数，用于计算矩阵的行列式和迹
from sympy.matrices.expressions import det, trace

# 从 sympy.matrices.expressions.kronecker 模块中导入 KroneckerProduct 类和 kronecker_product 函数
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
                                                  kronecker_product,
                                                  combine_kronecker)

# 创建一个 2x2 的整数矩阵 mat1
mat1 = Matrix([[1, 2 * I], [1 + I, 3]])
# 创建一个 2x2 的整数矩阵 mat2
mat2 = Matrix([[2 * I, 3], [4 * I, 2]])

# 创建符号变量 i, j, k, n, m, o, p, x
i, j, k, n, m, o, p, x = symbols('i,j,k,n,m,o,p,x')
# 创建一个 n x n 的矩阵符号 Z
Z = MatrixSymbol('Z', n, n)
# 创建一个 m x m 的矩阵符号 W
W = MatrixSymbol('W', m, m)
# 创建一个 n x m 的矩阵符号 A
A = MatrixSymbol('A', n, m)
# 创建一个 n x m 的矩阵符号 B
B = MatrixSymbol('B', n, m)
# 创建一个 m x k 的矩阵符号 C
C = MatrixSymbol('C', m, k)


# 定义测试函数 test_KroneckerProduct，用于测试 KroneckerProduct 类的功能
def test_KroneckerProduct():
    # 断言：KroneckerProduct(A, B) 是 KroneckerProduct 类的实例
    assert isinstance(KroneckerProduct(A, B), KroneckerProduct)
    # 断言：将 A 替换为 C 后的 KroneckerProduct(A, B) 等于 KroneckerProduct(C, B)
    assert KroneckerProduct(A, B).subs(A, C) == KroneckerProduct(C, B)
    # 断言：KroneckerProduct(A, C) 的形状为 (n*m, m*k)
    assert KroneckerProduct(A, C).shape == (n*m, m*k)
    # 断言：KroneckerProduct(A, C) + KroneckerProduct(-A, C) 是零矩阵
    assert (KroneckerProduct(A, C) + KroneckerProduct(-A, C)).is_ZeroMatrix
    # 断言：KroneckerProduct(W, Z) * KroneckerProduct(W.I, Z.I) 是单位矩阵
    assert (KroneckerProduct(W, Z) * KroneckerProduct(W.I, Z.I)).is_Identity


# 定义测试函数 test_KroneckerProduct_identity，用于测试单位矩阵的 KroneckerProduct
def test_KroneckerProduct_identity():
    # 断言：KroneckerProduct(Identity(m), Identity(n)) 等于 Identity(m*n)
    assert KroneckerProduct(Identity(m), Identity(n)) == Identity(m*n)
    # 断言：KroneckerProduct(eye(2), eye(3)) 等于 eye(6)
    assert KroneckerProduct(eye(2), eye(3)) == eye(6)


# 定义测试函数 test_KroneckerProduct_explicit，用于测试显式表达的 KroneckerProduct
def test_KroneckerProduct_explicit():
    # 创建符号矩阵符号 X 和 Y，每个为 2x2
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 创建 KroneckerProduct(X, Y)
    kp = KroneckerProduct(X, Y)
    # 断言：kp 的形状为 (4, 4)
    assert kp.shape == (4, 4)
    # 断言：kp 的显式表达式等于特定的 4x4 矩阵
    assert kp.as_explicit() == Matrix(
        [
            [X[0, 0]*Y[0, 0], X[0, 0]*Y[0, 1], X[0, 1]*Y[0, 0], X[0, 1]*Y[0, 1]],
            [X[0, 0]*Y[1, 0], X[0, 0]*Y[1, 1], X[0, 1]*Y[1, 0], X[0, 1]*Y[1, 1]],
            [X[1, 0]*Y[0, 0], X[1, 0]*Y[0, 1], X[1, 1]*Y[0, 0], X[1, 1]*Y[0, 1]],
            [X[1, 0]*Y[1, 0], X[1, 0]*Y[1, 1], X[1, 1]*Y[1, 0], X[1, 1]*Y[1, 1]]
        ]
    )


# 定义测试函数 test_tensor_product_adjoint，用于测试 KroneckerProduct 的共轭转置
def test_tensor_product_adjoint():
    # 断言：KroneckerProduct(I*A, B).adjoint() 等于 -I*KroneckerProduct(A.adjoint(), B.adjoint())
    assert KroneckerProduct(I*A, B).adjoint() == \
        -I*KroneckerProduct(A.adjoint(), B.adjoint())
    # 断言：KroneckerProduct(mat1, mat2).adjoint() 等于 kronecker_product(mat1.adjoint(), mat2.adjoint())
    assert KroneckerProduct(mat1, mat2).adjoint() == \
        kronecker_product(mat1.adjoint(), mat2.adjoint())


# 定义测试函数 test_tensor_product_conjugate，用于测试 KroneckerProduct 的共轭
def test_tensor_product_conjugate():
    # 断言：KroneckerProduct(I*A, B).conjugate() 等于 -I*KroneckerProduct(A.conjugate(), B.conjugate())
    assert KroneckerProduct(I*A, B).conjugate() == \
        -I*KroneckerProduct(A.conjugate(), B.conjugate())
    # 断言：KroneckerProduct(mat1, mat2).conjugate() 等于 kronecker_product(mat1.conjugate(), mat2.conjugate())
    assert KroneckerProduct(mat1, mat2).conjugate() == \
        kronecker_product(mat1.conjugate(), mat2.conjugate())


# 定义测试函数 test_tensor_product_transpose，用于测试 KroneckerProduct 的转置
def test_tensor_product_transpose():
    # 断言：KroneckerProduct(I*A, B).transpose() 等于 I*KroneckerProduct(A.transpose(), B.transpose())
    assert KroneckerProduct(I*A, B).transpose() == \
        I*KroneckerProduct(A.transpose(), B.transpose())
    # 断言：KroneckerProduct(mat1, mat2).transpose() 等于 kronecker_product(mat1.transpose(), mat2.transpose())
    assert KroneckerProduct(mat1, mat2).transpose() == \
        kronecker_product(mat1.transpose(), mat2.transpose())


# 定义测试函数 test_KroneckerProduct_is_associative，用于测试 KroneckerProduct 的结合性
def test_KroneckerProduct_is_associative():
    # 断言：kronecker_product(A, kronecker_product(B, C)) 等于 kronecker_product(kronecker_product(A, B), C)
    assert kronecker_product(A, kronecker_product(
        B, C)) == kronecker_product(kronecker_product(A, B), C)
    # 断言：kronecker_product(A, kronecker_product(B, C)) 等于 Kronecker
    # 断言：Kronecker 乘积具有与数乘交换的性质，左侧为 x 乘以矩阵 A 和 B 的 Kronecker 乘积，应与 x 乘以 A 和 B 的 Kronecker 乘积相等
    assert kronecker_product(x*A, B) == x*kronecker_product(A, B)
    
    # 断言：Kronecker 乘积具有与数乘交换的性质，左侧为 A 和 x 乘以矩阵 B 的 Kronecker 乘积，应与 x 乘以 A 和 B 的 Kronecker 乘积相等
    assert kronecker_product(A, x*B) == x*kronecker_product(A, B)
def test_KroneckerProduct_determinant():
    # 计算 Kronecker 乘积的行列式，应为第一个矩阵的行列式的乘幂乘以第二个矩阵的行列式的乘幂
    kp = kronecker_product(W, Z)
    assert det(kp) == det(W)**n * det(Z)**m


def test_KroneckerProduct_trace():
    # 计算 Kronecker 乘积的迹，应为第一个矩阵的迹乘以第二个矩阵的迹
    kp = kronecker_product(W, Z)
    assert trace(kp) == trace(W)*trace(Z)


def test_KroneckerProduct_isnt_commutative():
    # 检查 Kronecker 乘积的非交换性质
    assert KroneckerProduct(A, B) != KroneckerProduct(B, A)
    # 检查 Kronecker 乘积的交换性属性为 False
    assert KroneckerProduct(A, B).is_commutative is False


def test_KroneckerProduct_extracts_commutative_part():
    # 检查 Kronecker 乘积的提取交换部分
    assert kronecker_product(x * A, 2 * B) == x * \
        2 * KroneckerProduct(A, B)


def test_KroneckerProduct_inverse():
    # 计算 Kronecker 乘积的逆
    kp = kronecker_product(W, Z)
    assert kp.inverse() == kronecker_product(W.inverse(), Z.inverse())


def test_KroneckerProduct_combine_add():
    # 合并 Kronecker 乘积的加法组合
    kp1 = kronecker_product(A, B)
    kp2 = kronecker_product(C, W)
    assert combine_kronecker(kp1*kp2) == kronecker_product(A*C, B*W)


def test_KroneckerProduct_combine_mul():
    # 合并 Kronecker 乘积的乘法组合
    X = MatrixSymbol('X', m, n)
    Y = MatrixSymbol('Y', m, n)
    kp1 = kronecker_product(A, X)
    kp2 = kronecker_product(B, Y)
    assert combine_kronecker(kp1+kp2) == kronecker_product(A+B, X+Y)


def test_KroneckerProduct_combine_pow():
    # 合并 Kronecker 乘积的幂次组合
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert combine_kronecker(KroneckerProduct(
        X, Y)**x) == KroneckerProduct(X**x, Y**x)
    assert combine_kronecker(x * KroneckerProduct(X, Y)
                             ** 2) == x * KroneckerProduct(X**2, Y**2)
    assert combine_kronecker(
        x * (KroneckerProduct(X, Y)**2) * KroneckerProduct(A, B)) == x * KroneckerProduct(X**2 * A, Y**2 * B)
    # 由于 Kronecker 乘积的参数不是方阵，无法简化：
    assert combine_kronecker(KroneckerProduct(A, B.T) ** m) == KroneckerProduct(A, B.T) ** m


def test_KroneckerProduct_expand():
    # 展开 Kronecker 乘积
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)

    assert KroneckerProduct(X + Y, Y + Z).expand(kroneckerproduct=True) == \
        KroneckerProduct(X, Y) + KroneckerProduct(X, Z) + \
        KroneckerProduct(Y, Y) + KroneckerProduct(Y, Z)

def test_KroneckerProduct_entry():
    # 计算 Kronecker 乘积的元素
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', o, p)

    assert KroneckerProduct(A, B)._entry(i, j) == A[Mod(floor(i/o), n), Mod(floor(j/p), m)]*B[Mod(i, o), Mod(j, p)]
```