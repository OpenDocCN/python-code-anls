# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_as_explicit.py`

```
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, \
    ArrayTensorProduct, PermuteDims, ArrayDiagonal, ArrayContraction, ArrayAdd
from sympy.testing.pytest import raises

# 定义测试函数，测试数组表达式的显式调用
def test_array_as_explicit_call():

    # 测试 ZeroArray 类的显式表示
    assert ZeroArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray.zeros(3, 2, 4)
    
    # 测试 OneArray 类的显式表示
    assert OneArray(3, 2, 4).as_explicit() == ImmutableDenseNDimArray([1 for i in range(3*2*4)]).reshape(3, 2, 4)

    # 定义符号 k 和 ArraySymbol X
    k = Symbol("k")
    X = ArraySymbol("X", (k, 3, 2))
    
    # 测试对于符号数组 X 的显式表示会引发 ValueError 异常
    raises(ValueError, lambda: X.as_explicit())
    
    # 测试对于 ZeroArray 类的显式表示会引发 ValueError 异常
    raises(ValueError, lambda: ZeroArray(k, 2, 3).as_explicit())
    
    # 测试对于 OneArray 类的显式表示会引发 ValueError 异常
    raises(ValueError, lambda: OneArray(2, k, 2).as_explicit())

    # 定义符号数组 A 和 B
    A = ArraySymbol("A", (3, 3))
    B = ArraySymbol("B", (3, 3))

    # 测试 tensorproduct 函数生成的对象类型和显式表示
    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    # 测试 tensorcontraction 函数生成的对象类型和显式表示
    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    # 测试 tensordiagonal 函数生成的对象类型和显式表示
    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    # 测试 permutedims 函数生成的对象类型和显式表示
    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])


def test_array_as_explicit_matrix_symbol():

    # 定义矩阵符号 A 和 B
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)

    # 测试 tensorproduct 函数生成的对象类型和显式表示
    texpr = tensorproduct(A, B)
    assert isinstance(texpr, ArrayTensorProduct)
    assert texpr.as_explicit() == tensorproduct(A.as_explicit(), B.as_explicit())

    # 测试 tensorcontraction 函数生成的对象类型和显式表示
    texpr = tensorcontraction(A, (0, 1))
    assert isinstance(texpr, ArrayContraction)
    assert texpr.as_explicit() == A[0, 0] + A[1, 1] + A[2, 2]

    # 测试 tensordiagonal 函数生成的对象类型和显式表示
    texpr = tensordiagonal(A, (0, 1))
    assert isinstance(texpr, ArrayDiagonal)
    assert texpr.as_explicit() == ImmutableDenseNDimArray([A[0, 0], A[1, 1], A[2, 2]])

    # 测试 permutedims 函数生成的对象类型和显式表示
    texpr = permutedims(A, [1, 0])
    assert isinstance(texpr, PermuteDims)
    assert texpr.as_explicit() == permutedims(A.as_explicit(), [1, 0])

    # 定义表达式 expr
    expr = ArrayAdd(ArrayTensorProduct(A, B), ArrayTensorProduct(B, A))
    
    # 测试表达式的显式表示
    assert expr.as_explicit() == expr.args[0].as_explicit() + expr.args[1].as_explicit()
```