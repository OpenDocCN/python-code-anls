# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_transpose.py`

```
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.singleton import S
from sympy.core.symbol import symbols

# 定义符号变量
n, m, l, k, p = symbols('n m l k p', integer=True)

# 定义矩阵符号
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)

# 定义测试函数 test_transpose
def test_transpose():
    # 定义方阵符号
    Sq = MatrixSymbol('Sq', n, n)

    # 断言：A 的转置与 Transpose(A) 相等
    assert transpose(A) == Transpose(A)
    # 断言：Transpose(A) 的形状为 (m, n)
    assert Transpose(A).shape == (m, n)
    # 断言：Transpose(A*B) 的形状为 (l, n)
    assert Transpose(A*B).shape == (l, n)
    # 断言：A 的双重转置与 A 相等
    assert transpose(Transpose(A)) == A
    # 断言：Transpose(Transpose(A)) 的类型为 Transpose
    assert isinstance(Transpose(Transpose(A)), Transpose)

    # 断言：Transpose(A) 的伴随等于 Adjoint(Transpose(A))
    assert adjoint(Transpose(A)) == Adjoint(Transpose(A))
    # 断言：Transpose(A) 的共轭等于 A 的伴随
    assert conjugate(Transpose(A)) == Adjoint(A)

    # 断言：3x3 单位矩阵的转置等于其自身
    assert Transpose(eye(3)).doit() == eye(3)

    # 断言：标量 S(5) 的转置等于其自身
    assert Transpose(S(5)).doit() == S(5)

    # 断言：2x2 矩阵的转置
    assert Transpose(Matrix([[1, 2], [3, 4]])).doit() == Matrix([[1, 3], [2, 4]])

    # 断言：方阵 Sq 的迹的转置等于迹本身
    assert transpose(trace(Sq)) == trace(Sq)
    # 断言：方阵 Sq 的转置的迹等于迹本身
    assert trace(Transpose(Sq)) == trace(Sq)

    # 断言：Sq 的转置的第 (0, 1) 元素等于 Sq 的第 (1, 0) 元素
    assert Transpose(Sq)[0, 1] == Sq[1, 0]

    # 断言：A*B 的转置等于 B 的转置乘以 A 的转置
    assert Transpose(A*B).doit() == Transpose(B) * Transpose(A)

# 定义测试函数 test_transpose_MatAdd_MatMul
def test_transpose_MatAdd_MatMul():
    # 导入三角函数中的 cos 函数
    from sympy.functions.elementary.trigonometric import cos

    # 定义符号变量 x
    x = symbols('x')
    # 定义两个 3x3 矩阵符号
    M = MatrixSymbol('M', 3, 3)
    N = MatrixSymbol('N', 3, 3)

    # 断言：(N + cos(x) * M) 的转置等于 cos(x) * M 的转置加上 N 的转置
    assert (N + (cos(x) * M)).T == cos(x)*M.T + N.T

# 定义测试函数 test_refine
def test_refine():
    # 断言：如果 C 是对称的，则 C 的转置等于 C
    assert refine(C.T, Q.symmetric(C)) == C

# 定义测试函数 test_transpose1x1
def test_transpose1x1():
    # 定义 1x1 矩阵符号 m
    m = MatrixSymbol('m', 1, 1)
    # 断言：m 的转置等于 m 本身
    assert m == refine(m.T)
    # 断言：m 的双重转置等于 m 本身
    assert m == refine(m.T.T)

# 定义测试函数 test_issue_9817
def test_issue_9817():
    # 导入 Identity 矩阵表达式
    from sympy.matrices.expressions import Identity
    # 定义向量符号 v 和 3x3 方阵符号 A
    v = MatrixSymbol('v', 3, 1)
    A = MatrixSymbol('A', 3, 3)
    # 定义向量 x 和 Identity(3) 矩阵 X
    x = Matrix([i + 1 for i in range(3)])
    X = Identity(3)
    # 定义二次型 quadratic
    quadratic = v.T * A * v
    # 替换 quadratic 中的符号 v 和 A，然后断言结果等于预期的矩阵
    subbed = quadratic.xreplace({v:x, A:X})
    assert subbed.as_explicit() == Matrix([[14]])
```