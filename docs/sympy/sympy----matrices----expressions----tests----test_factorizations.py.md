# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_factorizations.py`

```
# 导入 SymPy 库中的特定模块和函数
from sympy.matrices.expressions.factorizations import lu, LofCholesky, qr, svd
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol

# 定义符号 n 作为 SymPy 符号
n = Symbol('n')
# 定义一个 n × n 的矩阵符号 X
X = MatrixSymbol('X', n, n)

# 测试 LU 分解
def test_LU():
    # 对矩阵 X 进行 LU 分解，返回下三角矩阵 L 和上三角矩阵 U
    L, U = lu(X)
    # 断言 L 和 U 的形状与 X 相同
    assert L.shape == U.shape == X.shape
    # 使用 SymPy 的问询系统检查 L 是否为下三角矩阵
    assert ask(Q.lower_triangular(L))
    # 使用 SymPy 的问询系统检查 U 是否为上三角矩阵
    assert ask(Q.upper_triangular(U))

# 测试 Cholesky 分解
def test_Cholesky():
    # 对矩阵 X 进行 Cholesky 分解，返回下三角矩阵 L
    LofCholesky(X)

# 测试 QR 分解
def test_QR():
    # 对矩阵 X 进行 QR 分解，返回正交矩阵 Q 和上三角矩阵 R
    Q_, R = qr(X)
    # 断言 Q 和 R 的形状与 X 相同
    assert Q_.shape == R.shape == X.shape
    # 使用 SymPy 的问询系统检查 Q 是否为正交矩阵
    assert ask(Q.orthogonal(Q_))
    # 使用 SymPy 的问询系统检查 R 是否为上三角矩阵
    assert ask(Q.upper_triangular(R))

# 测试 SVD 分解
def test_svd():
    # 对矩阵 X 进行 SVD 分解，返回正交矩阵 U、对角矩阵 S 和正交矩阵 V
    U, S, V = svd(X)
    # 断言 U、S 和 V 的形状与 X 相同
    assert U.shape == S.shape == V.shape == X.shape
    # 使用 SymPy 的问询系统检查 U 是否为正交矩阵
    assert ask(Q.orthogonal(U))
    # 使用 SymPy 的问询系统检查 V 是否为正交矩阵
    assert ask(Q.orthogonal(V))
    # 使用 SymPy 的问询系统检查 S 是否为对角矩阵
    assert ask(Q.diagonal(S))
```