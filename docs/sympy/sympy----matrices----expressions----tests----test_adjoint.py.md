# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_adjoint.py`

```
# 导入必要的符号和函数库
from sympy.core import symbols, S
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix

# 定义符号变量 n, m, l, k, p，并指定其为整数类型
n, m, l, k, p = symbols('n m l k p', integer=True)

# 定义矩阵符号 A, B, C 分别为 n x m, m x l, n x n 的矩阵符号
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)

# 定义测试函数 test_adjoint
def test_adjoint():
    # 定义 n x n 的符号矩阵 Sq
    Sq = MatrixSymbol('Sq', n, n)

    # 断言：Adjoint(A) 的形状应为 (m, n)
    assert Adjoint(A).shape == (m, n)
    
    # 断言：Adjoint(A*B) 的形状应为 (l, n)
    assert Adjoint(A*B).shape == (l, n)
    
    # 断言：adjoint(Adjoint(A)) 应该等于 A
    assert adjoint(Adjoint(A)) == A
    
    # 断言：Adjoint(Adjoint(A)) 应为 Adjoint 类型的对象
    assert isinstance(Adjoint(Adjoint(A)), Adjoint)
    
    # 断言：conjugate(Adjoint(A)) 应等于 Transpose(A)
    assert conjugate(Adjoint(A)) == Transpose(A)
    
    # 断言：transpose(Adjoint(A)) 应等于 Adjoint(Transpose(A))
    assert transpose(Adjoint(A)) == Adjoint(Transpose(A))
    
    # 断言：Adjoint(单位矩阵(3x3)).doit() 应等于 3x3 的单位矩阵
    assert Adjoint(eye(3)).doit() == eye(3)
    
    # 断言：Adjoint(S(5)).doit() 应等于 S(5)
    assert Adjoint(S(5)).doit() == S(5)
    
    # 断言：Adjoint(给定矩阵).doit() 应等于 给定矩阵的转置矩阵
    assert Adjoint(Matrix([[1, 2], [3, 4]])).doit() == Matrix([[1, 3], [2, 4]])
    
    # 断言：adjoint(trace(Sq)) 应等于 conjugate(trace(Sq))
    assert adjoint(trace(Sq)) == conjugate(trace(Sq))
    
    # 断言：trace(adjoint(Sq)) 应等于 conjugate(trace(Sq))
    assert trace(adjoint(Sq)) == conjugate(trace(Sq))
    
    # 断言：Adjoint(Sq) 的第 (0, 1) 个元素应等于 Sq 的转置的 (1, 0) 元素的共轭
    assert Adjoint(Sq)[0, 1] == conjugate(Sq[1, 0])
    
    # 断言：Adjoint(A*B).doit() 应等于 Adjoint(B) * Adjoint(A)
    assert Adjoint(A*B).doit() == Adjoint(B) * Adjoint(A)
```