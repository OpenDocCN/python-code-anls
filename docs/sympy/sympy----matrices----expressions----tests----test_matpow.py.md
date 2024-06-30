# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_matpow.py`

```
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement

# 定义整数类型的符号变量
n, m, l, k = symbols('n m l k', integer=True)
# 定义矩阵符号
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)

# 测试函数，测试关于不同矩阵表达式的操作
def test_entry_matrix():
    # 创建一个不可变矩阵对象
    X = ImmutableMatrix([[1, 2], [3, 4]])
    # 断言：X 的零次幂的 (0, 0) 元素应为 1
    assert MatPow(X, 0)[0, 0] == 1
    # 断言：X 的零次幂的 (0, 1) 元素应为 0
    assert MatPow(X, 0)[0, 1] == 0
    # 断言：X 的一次幂的 (0, 0) 元素应为 1
    assert MatPow(X, 1)[0, 0] == 1
    # 断言：X 的一次幂的 (0, 1) 元素应为 2
    assert MatPow(X, 1)[0, 1] == 2
    # 断言：X 的二次幂的 (0, 0) 元素应为 7
    assert MatPow(X, 2)[0, 0] == 7

# 测试函数，测试关于符号矩阵的操作
def test_entry_symbol():
    from sympy.concrete import Sum
    # 断言：C 的零次幂的 (0, 0) 元素应为 1
    assert MatPow(C, 0)[0, 0] == 1
    # 断言：C 的零次幂的 (0, 1) 元素应为 0
    assert MatPow(C, 0)[0, 1] == 0
    # 断言：C 的一次幂的 (0, 0) 元素应与 C[0, 0] 相等
    assert MatPow(C, 1)[0, 0] == C[0, 0]
    # 断言：C 的二次幂的 (0, 0) 元素类型为 Sum 对象
    assert isinstance(MatPow(C, 2)[0, 0], Sum)
    # 断言：C 的 n 次幂的 (0, 0) 元素类型为 MatrixElement 对象
    assert isinstance(MatPow(C, n)[0, 0], MatrixElement)

# 测试函数，测试将矩阵表达式转换为显式矩阵的操作
def test_as_explicit_symbol():
    X = MatrixSymbol('X', 2, 2)
    # 断言：X 的零次幂转换为显式矩阵应为单位矩阵
    assert MatPow(X, 0).as_explicit() == ImmutableMatrix(Identity(2))
    # 断言：X 的一次幂转换为显式矩阵应为 X 的显式表示
    assert MatPow(X, 1).as_explicit() == X.as_explicit()
    # 断言：X 的二次幂转换为显式矩阵应为 X 的显式表示的平方
    assert MatPow(X, 2).as_explicit() == (X.as_explicit())**2
    # 断言：X 的 n 次幂转换为显式矩阵应为指定的显式矩阵形式
    assert MatPow(X, n).as_explicit() == ImmutableMatrix([
        [(X ** n)[0, 0], (X ** n)[0, 1]],
        [(X ** n)[1, 0], (X ** n)[1, 1]],
    ])

    a = MatrixSymbol("a", 3, 1)
    b = MatrixSymbol("b", 3, 1)
    c = MatrixSymbol("c", 3, 1)

    # 断言：表达式 a.T * b 的平方根转换为显式矩阵应为计算结果的矩阵形式
    expr = (a.T*b)**S.Half
    assert expr.as_explicit() == Matrix([[sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])]])

    # 断言：表达式 c * (a.T * b) 的平方根转换为显式矩阵应为计算结果的矩阵形式
    expr = c*(a.T*b)**S.Half
    m = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    assert expr.as_explicit() == Matrix([[c[0, 0]*m], [c[1, 0]*m], [c[2, 0]*m]])

    # 断言：表达式 (a * b.T) 的平方根转换为显式矩阵应为计算结果的矩阵形式
    expr = (a*b.T)**S.Half
    denom = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    expected = (a*b.T).as_explicit()/denom
    assert expr.as_explicit() == expected

    # 断言：表达式 X 的逆矩阵转换为显式矩阵应为计算结果的矩阵形式
    expr = X**-1
    det = X[0, 0]*X[1, 1] - X[1, 0]*X[0, 1]
    expected = Matrix([[X[1, 1], -X[0, 1]], [-X[1, 0], X[0, 0]]])/det
    assert expr.as_explicit() == expected

    # 断言：表达式 X 的 m 次幂转换为显式矩阵应为 X 的 m 次幂的显式表示
    expr = X**m
    assert expr.as_explicit() == X.as_explicit()**m

# 测试函数，测试将矩阵表达式转换为显式矩阵的操作
def test_as_explicit_matrix():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    # 断言：A 的零次幂转换为显式矩阵应为单位矩阵
    assert MatPow(A, 0).as_explicit() == ImmutableMatrix(Identity(2))
    # 断言：A 的一次幂转换为显式矩阵应为 A 的显式表示
    assert MatPow(A, 1).as_explicit() == A
    # 断言：A 的二次幂转换为显式矩阵应为 A 的平方
    assert MatPow(A, 2).as_explicit() == A**2
    # 断言：A 的负一次幂转换为显式矩阵应为 A 的逆矩阵
    assert MatPow(A, -1).as_explicit() == A.inv()
    # 断言：A 的负二次幂转换为显式矩阵应为 A 的逆矩阵的平方
    assert MatPow(A, -2).as_explicit() == (A.inv())**2
    #
    # 断言：计算矩阵 C 的零次幂，并验证其结果是否为 n 阶单位矩阵 Identity(n)
    assert MatPow(C, 0).doit() == Identity(n)
    
    # 断言：计算矩阵 C 的一次幂，并验证其结果是否为矩阵 C 自身
    assert MatPow(C, 1).doit() == C
    
    # 断言：计算矩阵 C 的负一次幂，并验证其结果是否为矩阵 C 的逆矩阵 C.I
    assert MatPow(C, -1).doit() == C.I
    
    # 遍历列表 [2, S.Half, S.Pi, n] 中的每个元素 r
    for r in [2, S.Half, S.Pi, n]:
        # 断言：计算矩阵 C 的 r 次幂，并验证其结果是否等于矩阵 C 的 r 次幂结果
        assert MatPow(C, r).doit() == MatPow(C, r)
def test_doit_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    # 测试 X 的 0 次幂是否为单位矩阵
    assert MatPow(X, 0).doit() == ImmutableMatrix(Identity(2))
    # 测试 X 的 1 次幂是否为 X 本身
    assert MatPow(X, 1).doit() == X
    # 测试 X 的 2 次幂是否为 X 的平方
    assert MatPow(X, 2).doit() == X**2
    # 测试 X 的 -1 次幂是否为 X 的逆矩阵
    assert MatPow(X, -1).doit() == X.inv()
    # 测试 X 的 -2 次幂是否为 X 的逆矩阵的平方
    assert MatPow(X, -2).doit() == (X.inv())**2
    # 在一个更便宜的方式下，测试一个 1x1 矩阵的 1/2 次幂是否为 2
    assert MatPow(ImmutableMatrix([4]), S.Half).doit() == ImmutableMatrix([2])
    X = ImmutableMatrix([[0, 2], [0, 4]]) # det() == 0
    # 测试行列式为零的矩阵的负一次幂是否引发 ValueError
    raises(ValueError, lambda: MatPow(X,-1).doit())
    # 测试行列式为零的矩阵的负二次幂是否引发 ValueError
    raises(ValueError, lambda: MatPow(X,-2).doit())


def test_nonsquare():
    A = MatrixSymbol('A', 2, 3)
    B = ImmutableMatrix([[1, 2, 3], [4, 5, 6]])
    for r in [-1, 0, 1, 2, S.Half, S.Pi, n]:
        # 测试非方阵 A 的任何 r 次幂是否引发 NonSquareMatrixError
        raises(NonSquareMatrixError, lambda: MatPow(A, r))
        # 测试非方阵 B 的任何 r 次幂是否引发 NonSquareMatrixError
        raises(NonSquareMatrixError, lambda: MatPow(B, r))


def test_doit_equals_pow(): #17179
    X = ImmutableMatrix ([[1,0],[0,1]])
    # 测试 X 的 n 次幂是否等于 X**n 是否等于 X
    assert MatPow(X, n).doit() == X**n == X


def test_doit_nested_MatrixExpr():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[2, 3], [4, 5]])
    # 测试复合矩阵表达式 (X*Y)**2 的 doit() 方法是否等于 (X*Y)**2
    assert MatPow(MatMul(X, Y), 2).doit() == (X*Y)**2
    # 测试复合矩阵表达式 (X+Y)**2 的 doit() 方法是否等于 (X+Y)**2
    assert MatPow(MatAdd(X, Y), 2).doit() == (X + Y)**2


def test_identity_power():
    k = Identity(n)
    # 测试单位矩阵 k 的 4 次幂是否为 k 本身
    assert MatPow(k, 4).doit() == k
    # 测试单位矩阵 k 的 n 次幂是否为 k 本身
    assert MatPow(k, n).doit() == k
    # 测试单位矩阵 k 的 -3 次幂是否为 k 本身
    assert MatPow(k, -3).doit() == k
    # 测试单位矩阵 k 的 0 次幂是否为 k 本身
    assert MatPow(k, 0).doit() == k
    l = Identity(3)
    # 测试单位矩阵 l 的 n 次幂是否为 l 本身
    assert MatPow(l, n).doit() == l
    # 测试单位矩阵 l 的 -1 次幂是否为 l 本身
    assert MatPow(l, -1).doit() == l
    # 测试单位矩阵 l 的 0 次幂是否为 l 本身
    assert MatPow(l, 0).doit() == l


def test_zero_power():
    z1 = ZeroMatrix(n, n)
    # 测试零矩阵 z1 的 3 次幂是否为 z1
    assert MatPow(z1, 3).doit() == z1
    # 测试零矩阵 z1 的负一次幂是否引发 ValueError
    raises(ValueError, lambda:MatPow(z1, -1).doit())
    # 测试零矩阵 z1 的 0 次幂是否为 n 阶单位矩阵
    assert MatPow(z1, 0).doit() == Identity(n)
    # 测试零矩阵 z1 的 n 次幂是否为 z1
    assert MatPow(z1, n).doit() == z1
    # 测试零矩阵 z1 的负二次幂是否引发 ValueError
    raises(ValueError, lambda:MatPow(z1, -2).doit())
    z2 = ZeroMatrix(4, 4)
    # 测试零矩阵 z2 的 n 次幂是否为 z2
    assert MatPow(z2, n).doit() == z2
    # 测试零矩阵 z2 的负三次幂是否引发 ValueError
    raises(ValueError, lambda:MatPow(z2, -3).doit())
    # 测试零矩阵 z2 的 2 次幂是否为 z2
    assert MatPow(z2, 2).doit() == z2
    # 测试零矩阵 z2 的 0 次幂是否为 4 阶单位矩阵
    assert MatPow(z2, 0).doit() == Identity(4)
    # 测试零矩阵 z2 的负一次幂是否引发 ValueError
    raises(ValueError, lambda:MatPow(z2, -1).doit())


def test_OneMatrix_power():
    o = OneMatrix(3, 3)
    # 测试单位矩阵 o 的 0 次幂是否为 3 阶单位矩阵
    assert o ** 0 == Identity(3)
    # 测试单位矩阵 o 的 1 次幂是否为 o 本身
    assert o ** 1 == o
    # 测试单位矩阵 o 的平方是否为 3 倍的 o
    assert o * o == o ** 2 == 3 * o
    # 测试单位矩阵 o 的三次幂是否为 9 倍的 o
    assert o * o * o == o ** 3 == 9 * o

    o = OneMatrix(n, n)
    # 测试单位矩阵 o 的 n-1 次幂乘以 o 是否等于 n 的 o
    assert powsimp(o ** (n - 1) * o) == o ** n == n ** (n - 1) * o


def test_transpose_power():
    from sympy.matrices.expressions.transpose import Transpose as TP

    # 测试转置矩阵表达式 (C*D).T**5 是否等于 ((C*D)**5).T 是否等于 (D.T * C.T)**5
    assert (C*D).T**5 == ((C*D)**5).T == (D.T * C.T)**5
    # 测试转置矩阵表达式 ((C*D).T**5).T 是否等于 (C*D)**5
    assert ((C*D).T**5).T == (C*D)**5

    # 测试逆转置矩阵表达式 (C.T.I.T)**7 是否等于 C**-7
    assert (C.T.I.T)**7 == C**-7
    # 测试转置逆矩阵表达式 (C.T**l).T**k 是否等于 C**(l*k)
    assert (C.T**l).T**k == C**(l*k)

    # 测试转置矩阵表达
    # 断言：验证矩阵 C 的零次幂的逆矩阵是否等于单位矩阵
    assert Inverse(MatPow(C, 0)).doit() == Identity(n)
    
    # 断言：验证矩阵 C 的一次幂的逆矩阵是否等于矩阵 C 的逆矩阵
    assert Inverse(MatPow(C, 1)).doit() == Inverse(C)
    
    # 断言：验证矩阵 C 的二次幂的逆矩阵是否等于矩阵 C 的负二次幂
    assert Inverse(MatPow(C, 2)).doit() == MatPow(C, -2)
    
    # 断言：验证矩阵 C 的负一次幂的逆矩阵是否等于矩阵 C 本身
    assert Inverse(MatPow(C, -1)).doit() == C
    
    # 断言：验证矩阵 C 的逆矩阵的零次幂是否等于单位矩阵
    assert MatPow(Inverse(C), 0).doit() == Identity(n)
    
    # 断言：验证矩阵 C 的逆矩阵的一次幂是否等于矩阵 C 的逆矩阵
    assert MatPow(Inverse(C), 1).doit() == Inverse(C)
    
    # 断言：验证矩阵 C 的逆矩阵的二次幂是否等于矩阵 C 的负二次幂
    assert MatPow(Inverse(C), 2).doit() == MatPow(C, -2)
    
    # 断言：验证矩阵 C 的逆矩阵的负一次幂是否等于矩阵 C 本身
    assert MatPow(Inverse(C), -1).doit() == C
# 定义用于测试组合幂操作的函数
def test_combine_powers():
    # 断言：(C ** 1) ** 1 等于 C
    assert (C ** 1) ** 1 == C
    # 断言：(C ** 2) ** 3 等于 MatPow(C, 6)
    assert (C ** 2) ** 3 == MatPow(C, 6)
    # 断言：(C ** -2) ** -3 等于 MatPow(C, 6)
    assert (C ** -2) ** -3 == MatPow(C, 6)
    # 断言：(C ** -1) ** -1 等于 C
    assert (C ** -1) ** -1 == C
    # 断言：((((C ** 2) ** 3) ** 4) ** 5) 等于 MatPow(C, 120)
    assert (((C ** 2) ** 3) ** 4) ** 5 == MatPow(C, 120)
    # 断言：(C ** n) ** n 等于 C ** (n ** 2)
    assert (C ** n) ** n == C ** (n ** 2)


# 定义用于测试未改变操作的函数
def test_unchanged():
    # 断言：对于 MatPow(C, 0)，不发生变化
    assert unchanged(MatPow, C, 0)
    # 断言：对于 MatPow(C, 1)，不发生变化
    assert unchanged(MatPow, C, 1)
    # 断言：对于 Inverse(C)，指数为 -1 时不发生变化
    assert unchanged(MatPow, Inverse(C), -1)
    # 断言：对于 MatPow(C, -1)，指数为 -1 时不发生变化
    assert unchanged(Inverse, MatPow(C, -1), -1)
    # 断言：对于 MatPow(C, -1)，指数为 -1 时不发生变化
    assert unchanged(MatPow, MatPow(C, -1), -1)
    # 断言：对于 MatPow(C, 1)，指数为 1 时不发生变化
    assert unchanged(MatPow, MatPow(C, 1), 1)


# 定义用于测试无指数操作的函数
def test_no_exponentiation():
    # 如果通过，Pow.as_numer_denom 应该能识别 MatAdd 作为指数
    raises(NotImplementedError, lambda: 3**(-2*C))
```