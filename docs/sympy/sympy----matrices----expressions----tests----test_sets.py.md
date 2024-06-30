# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\test_sets.py`

```
# 导入所需的符号、矩阵和测试模块
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.sets import MatrixSet
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.testing.pytest import raises
from sympy.sets.sets import SetKind
from sympy.matrices.kind import MatrixKind
from sympy.core.kind import NumberKind


# 定义测试函数，测试 MatrixSet 类的功能
def test_MatrixSet():
    # 定义符号 n 和 m，并声明其为整数
    n, m = symbols('n m', integer=True)
    
    # 创建两个矩阵符号对象 A 和 C
    A = MatrixSymbol('A', n, m)
    C = MatrixSymbol('C', n, n)

    # 创建一个实数域上的 2x2 矩阵集合对象 M
    M = MatrixSet(2, 2, set=S.Reals)
    
    # 断言 M 的形状为 (2, 2)
    assert M.shape == (2, 2)
    # 断言 M 的集合为实数域 S.Reals
    assert M.set == S.Reals
    
    # 创建一个具体的 2x2 矩阵 X
    X = Matrix([[1, 2], [3, 4]])
    # 断言 X 属于集合 M
    assert X in M
    
    # 创建一个全零 2x2 矩阵 X
    X = ZeroMatrix(2, 2)
    # 断言 X 属于集合 M
    assert X in M
    
    # 断言 A 不属于集合 M，预期引发 TypeError 异常
    raises(TypeError, lambda: A in M)
    # 断言 1 不属于集合 M，预期引发 TypeError 异常
    raises(TypeError, lambda: 1 in M)
    
    # 创建一个 n x m 矩阵集合对象 M，元素在实数域 S.Reals 上
    M = MatrixSet(n, m, set=S.Reals)
    # 断言 A 属于集合 M
    assert A in M
    
    # 断言 C 不属于集合 M，预期引发 TypeError 异常
    raises(TypeError, lambda: C in M)
    # 断言 X 不属于集合 M，预期引发 TypeError 异常
    raises(TypeError, lambda: X in M)
    
    # 创建一个集合元素为 {1, 2, 3} 的 2x2 矩阵集合对象 M
    M = MatrixSet(2, 2, set={1, 2, 3})
    # 创建两个具体的矩阵 X 和 Y
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    # 断言 X 不属于集合 M，预期结果为 S.false
    assert (X in M) == S.false
    # 断言 Y 不属于集合 M，预期结果为 S.false
    assert (Y in M) == S.false
    
    # 测试创建矩阵集合时的异常情况
    # 预期引发 ValueError 异常，因为第二个参数 -2 不是有效的矩阵尺寸
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    # 预期引发 ValueError 异常，因为第一个参数 2.4 不是整数
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    # 预期引发 TypeError 异常，因为集合参数应为集合对象而非元组
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))


# 定义测试函数，测试 SetKind 类的 MatrixSet 方法
def test_SetKind_MatrixSet():
    # 断言创建一个实数域上的 2x2 矩阵集合对象 M 的类型
    assert MatrixSet(2, 2, set=S.Reals).kind is SetKind(MatrixKind(NumberKind))
```