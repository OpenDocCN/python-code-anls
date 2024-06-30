# `D:\src\scipysrc\sympy\sympy\core\tests\test_kind.py`

```
from sympy.core.add import Add
from sympy.core.kind import NumberKind, UndefinedKind  # 导入 SymPy 中的类型定义：NumberKind 和 UndefinedKind
from sympy.core.mul import Mul  # 导入 SymPy 中的乘法类 Mul
from sympy.core.numbers import pi, zoo, I, AlgebraicNumber  # 导入 SymPy 中的数学常数和代数数类
from sympy.core.singleton import S  # 导入 SymPy 中的单例类 S
from sympy.core.symbol import Symbol  # 导入 SymPy 中的符号类 Symbol
from sympy.integrals.integrals import Integral  # 导入 SymPy 中的积分类 Integral
from sympy.core.function import Derivative  # 导入 SymPy 中的导数类 Derivative
from sympy.matrices import (Matrix, SparseMatrix, ImmutableMatrix,  # 导入 SymPy 中的矩阵类
    ImmutableSparseMatrix, MatrixSymbol, MatrixKind, MatMul)

comm_x = Symbol('x')  # 创建一个可交换的符号 x
noncomm_x = Symbol('x', commutative=False)  # 创建一个不可交换的符号 x

def test_NumberKind():
    # 测试 SymPy 中的 NumberKind 类型判断
    assert S.One.kind is NumberKind
    assert pi.kind is NumberKind
    assert S.NaN.kind is NumberKind
    assert zoo.kind is NumberKind
    assert I.kind is NumberKind
    assert AlgebraicNumber(1).kind is NumberKind

def test_Add_kind():
    # 测试 SymPy 中 Add 类的类型判断
    assert Add(2, 3, evaluate=False).kind is NumberKind
    assert Add(2, comm_x).kind is NumberKind
    assert Add(2, noncomm_x).kind is UndefinedKind

def test_mul_kind():
    # 测试 SymPy 中 Mul 类的类型判断
    assert Mul(2, comm_x, evaluate=False).kind is NumberKind
    assert Mul(2, 3, evaluate=False).kind is NumberKind
    assert Mul(noncomm_x, 2, evaluate=False).kind is UndefinedKind
    assert Mul(2, noncomm_x, evaluate=False).kind is UndefinedKind

def test_Symbol_kind():
    # 测试 SymPy 中 Symbol 类的类型判断
    assert comm_x.kind is NumberKind
    assert noncomm_x.kind is UndefinedKind

def test_Integral_kind():
    # 测试 SymPy 中 Integral 类的类型判断
    A = MatrixSymbol('A', 2, 2)
    assert Integral(comm_x, comm_x).kind is NumberKind
    assert Integral(A, comm_x).kind is MatrixKind(NumberKind)

def test_Derivative_kind():
    # 测试 SymPy 中 Derivative 类的类型判断
    A = MatrixSymbol('A', 2, 2)
    assert Derivative(comm_x, comm_x).kind is NumberKind
    assert Derivative(A, comm_x).kind is MatrixKind(NumberKind)

def test_Matrix_kind():
    # 测试 SymPy 中矩阵类的类型判断
    classes = (Matrix, SparseMatrix, ImmutableMatrix, ImmutableSparseMatrix)
    for cls in classes:
        m = cls.zeros(3, 2)
        assert m.kind is MatrixKind(NumberKind)

def test_MatMul_kind():
    # 测试 SymPy 中 MatMul 类的类型判断
    M = Matrix([[1, 2], [3, 4]])
    assert MatMul(2, M).kind is MatrixKind(NumberKind)
    assert MatMul(comm_x, M).kind is MatrixKind(NumberKind)
```