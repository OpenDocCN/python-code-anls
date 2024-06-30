# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_array_expressions.py`

```
# 导入 random 模块，用于生成随机数
import random

# 从 sympy 库中导入多个符号和函数
from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
    PermuteDims, ArrayContraction, ArrayTensorProduct, ArrayDiagonal, \
    ArrayAdd, nest_permutation, ArrayElementwiseApplyFunc, _EditArrayContraction, _ArgE, _array_tensor_product, \
    _array_contraction, _array_diagonal, _array_add, _permute_dims, Reshape
from sympy.testing.pytest import raises

# 定义多个符号变量 i, j, k, l, m, n
i, j, k, l, m, n = symbols("i j k l m n")

# 定义多个 ArraySymbol 类对象，用于表示符号数组
M = ArraySymbol("M", (k, k))
N = ArraySymbol("N", (k, k))
P = ArraySymbol("P", (k, k))
Q = ArraySymbol("Q", (k, k))

A = ArraySymbol("A", (k, k))
B = ArraySymbol("B", (k, k))
C = ArraySymbol("C", (k, k))
D = ArraySymbol("D", (k, k))

X = ArraySymbol("X", (k, k))
Y = ArraySymbol("Y", (k, k))

a = ArraySymbol("a", (k, 1))
b = ArraySymbol("b", (k, 1))
c = ArraySymbol("c", (k, 1))
d = ArraySymbol("d", (k, 1))

# 定义一个测试函数 test_array_symbol_and_element
def test_array_symbol_and_element():
    # 定义一个一维 ArraySymbol 对象 A
    A = ArraySymbol("A", (2,))
    # 访问 A 的元素
    A0 = ArrayElement(A, (0,))
    A1 = ArrayElement(A, (1,))
    # 断言 A 的第一个元素等于 A0，第二个元素不等于 A0
    assert A[0] == A0
    assert A[1] != A0
    # 将 A 转换为具体的 DenseNDimArray 对象，包含 A0 和 A1
    assert A.as_explicit() == ImmutableDenseNDimArray([A0, A1])

    # 计算 A 与自身的张量积
    A2 = tensorproduct(A, A)
    assert A2.shape == (2, 2)
    # TODO: not yet supported:
    # assert A2.as_explicit() == Array([[A[0]*A[0], A[1]*A[0]], [A[0]*A[1], A[1]*A[1]]])
    
    # 对 A2 进行张量收缩，指定轴 (0, 1)
    A3 = tensorcontraction(A2, (0, 1))
    assert A3.shape == ()
    # TODO: not yet supported:
    # assert A3.as_explicit() == Array([])

    # 定义一个三维 ArraySymbol 对象 A
    A = ArraySymbol("A", (2, 3, 4))
    # 将 A 转换为具体的 DenseNDimArray 对象
    Ae = A.as_explicit()
    # 断言 Ae 的形状
    assert Ae == ImmutableDenseNDimArray(
        [[[ArrayElement(A, (i, j, k)) for k in range(4)] for j in range(3)] for i in range(2)])

    # 对 A 进行维度置换，使用指定的置换 Permutation(0, 2, 1)
    p = _permute_dims(A, Permutation(0, 2, 1))
    assert isinstance(p, PermuteDims)

    # 对一维 ArraySymbol 对象 A 进行索引操作，应该引发异常
    A = ArraySymbol("A", (2,))
    raises(IndexError, lambda: A[()])
    raises(IndexError, lambda: A[0, 1])
    raises(ValueError, lambda: A[-1])
    raises(ValueError, lambda: A[2])

    # 创建一个尺寸为 (3, 4) 的 OneArray 对象 O 和一个尺寸为 (m, n) 的 ZeroArray 对象 Z
    O = OneArray(3, 4)
    Z = ZeroArray(m, n)

    # 对 OneArray 对象 O 进行索引操作，应该引发异常
    raises(IndexError, lambda: O[()])
    raises(IndexError, lambda: O[1, 2, 3])
    raises(ValueError, lambda: O[3, 0])
    raises(ValueError, lambda: O[0, 4])

    # 断言 O 的某些索引位置的值为 1
    assert O[1, 2] == 1
    # 断言 Z 的某些索引位置的值为 0
    assert Z[1, 2] == 0


# 定义一个测试函数 test_zero_array
def test_zero_array():
    # 断言 ZeroArray() 返回 0，并且是整数类型
    assert ZeroArray() == 0
    assert ZeroArray().is_Integer

    # 创建一个尺寸为 (3, 2, 4) 的 ZeroArray 对象 za
    za = ZeroArray(3, 2, 4)
    # 断言 za 的形状为 (3, 2, 4)
    assert za.shape == (3, 2, 4)
    # 将 za 转换为具体的 DenseNDimArray 对象
    za_e = za.as_explicit()
    # 断言 za_e 的形状为 (3, 2, 4)
    assert za_e.shape == (3, 2, 4)

    # 定义符号变量 m, n, k
    m, n, k = symbols("m n k")
    # 创建一个尺寸为 (m, n, k, 2) 的 ZeroArray 对象 za
    za = ZeroArray(m, n, k, 2)
    # 断言 za 的形状为 (m, n, k, 2)，确保 za 是一个四维数组，最后一维长度为 2
    assert za.shape == (m, n, k, 2)
    # 使用 raises 函数测试 lambda 函数是否会引发 ValueError 异常，lambda 函数调用 za 的 as_explicit() 方法
    raises(ValueError, lambda: za.as_explicit())
# 定义测试函数，用于测试 OneArray 类的不同功能和行为
def test_one_array():
    # 断言默认构造的 OneArray 实例应为 1
    assert OneArray() == 1
    # 断言默认构造的 OneArray 实例具有 is_Integer 特性
    assert OneArray().is_Integer

    # 创建具有指定维度的 OneArray 实例 oa
    oa = OneArray(3, 2, 4)
    # 断言 oa 的形状为 (3, 2, 4)
    assert oa.shape == (3, 2, 4)
    # 将 oa 转换为显式形式，断言其形状不变
    oa_e = oa.as_explicit()
    assert oa_e.shape == (3, 2, 4)

    # 定义符号变量 m, n, k
    m, n, k = symbols("m n k")
    # 创建具有符号维度和额外维度 2 的 OneArray 实例 oa
    oa = OneArray(m, n, k, 2)
    # 断言 oa 的形状为 (m, n, k, 2)
    assert oa.shape == (m, n, k, 2)
    # 断言调用 as_explicit 方法会引发 ValueError 异常
    raises(ValueError, lambda: oa.as_explicit())

# 定义测试函数，用于测试 _array_contraction 函数的构造和行为
def test_arrayexpr_contraction_construction():

    # 测试单个数组 A 的缩并
    cg = _array_contraction(A)
    assert cg == A

    # 测试两个数组 A, B 的张量积缩并，指定缩并的轴
    cg = _array_contraction(_array_tensor_product(A, B), (1, 0))
    assert cg == _array_contraction(_array_tensor_product(A, B), (0, 1))

    # 测试两个矩阵 M, N 的张量积缩并，指定缩并的轴
    cg = _array_contraction(_array_tensor_product(M, N), (0, 1))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (0, 1)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 1)]

    # 测试两个矩阵 M, N 的张量积缩并，指定不同的缩并轴
    cg = _array_contraction(_array_tensor_product(M, N), (1, 2))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 1), (1, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(1, 2)]

    # 测试三个矩阵 M, M, N 的张量积缩并，指定多个不同的缩并轴
    cg = _array_contraction(_array_tensor_product(M, M, N), (1, 4), (2, 5))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (1, 1)], [(0, 1), (2, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 3), (1, 4)]

    # 测试移除 trival 缩并的情况
    assert _array_contraction(a, (1,)) == a
    assert _array_contraction(
        _array_tensor_product(a, b), (0, 2), (1,), (3,)) == _array_contraction(
        _array_tensor_product(a, b), (0, 2))

# 定义测试函数，用于测试 _array_tensor_product 和 _array_contraction 函数的结合使用
def test_arrayexpr_array_flatten():

    # 展开嵌套的 ArrayTensorProduct 对象
    expr1 = _array_tensor_product(M, N)
    expr2 = _array_tensor_product(P, Q)
    expr = _array_tensor_product(expr1, expr2)
    assert expr == _array_tensor_product(M, N, P, Q)
    assert expr.args == (M, N, P, Q)

    # 展开混合的 ArrayTensorProduct 和 ArrayContraction 对象
    cg1 = _array_contraction(expr1, (1, 2))
    cg2 = _array_contraction(expr2, (0, 3))

    expr = _array_tensor_product(cg1, cg2)
    assert expr == _array_contraction(_array_tensor_product(M, N, P, Q), (1, 2), (4, 7))

    expr = _array_tensor_product(M, cg1)
    assert expr == _array_contraction(_array_tensor_product(M, M, N), (3, 4))

    # 展开嵌套的 ArrayContraction 对象
    cgnested = _array_contraction(cg1, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))

    cgnested = _array_contraction(_array_tensor_product(cg1, cg2), (0, 3))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 6), (1, 2), (4, 7))

    cg3 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    cgnested = _array_contraction(cg3, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 5), (1, 3), (2, 4))

    cgnested = _array_contraction(cg3, (0, 3), (1, 2))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 7), (1, 3), (2, 4), (5, 6))
    
    # 调用函数 _array_tensor_product，并对返回结果进行数组收缩操作
    cg4 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))
    # 对收缩后的结果再次进行收缩操作
    cgnested = _array_contraction(cg4, (0, 1))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7))
    
    # 对 cg4 变量进行两次数组收缩操作，并将结果赋给 cgnested
    cgnested = _array_contraction(cg4, (0, 1), (2, 3))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7), (4, 6))
    
    # 对 cg4 变量进行对角线操作，并将结果赋给 cg
    cg = _array_diagonal(cg4)
    # 断言确保变量 cg 等于 cg4
    assert cg == cg4
    # 断言确保 cg 的类型是 cg4 的类型
    assert isinstance(cg, type(cg4))
    
    # 对表达式 expr1 进行对角线操作，得到 cg1
    cg1 = _array_diagonal(expr1, (1, 2))
    # 对表达式 expr2 进行对角线操作，得到 cg2
    cg2 = _array_diagonal(expr2, (0, 3))
    # 对 _array_tensor_product(M, N, P, Q) 的结果进行对角线操作，得到 cg3
    cg3 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    # 对 _array_tensor_product(M, N, P, Q) 的结果进行对角线操作，得到 cg4
    cg4 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))
    
    # 对 cg1 进行数组收缩操作，并将结果赋给 cgnested
    cgnested = _array_diagonal(cg1, (0, 1))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_diagonal(_array_tensor_product(M, N), (1, 2), (0, 3))
    
    # 对 cg3 进行数组收缩操作，并将结果赋给 cgnested
    cgnested = _array_diagonal(cg3, (1, 2))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4), (5, 6))
    
    # 对 cg4 进行数组收缩操作，并将结果赋给 cgnested
    cgnested = _array_diagonal(cg4, (1, 2))
    # 断言确保变量 cgnested 等于函数调用的结果
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7), (2, 4))
    
    # 对 M 和 N 进行数组加法操作，并将结果赋给 cg
    cg = _array_add(M, N)
    # 对 cg 和 P 进行数组加法操作，并将结果赋给 cg2
    cg2 = _array_add(cg, P)
    # 断言确保 cg2 的类型是 ArrayAdd 类型
    assert isinstance(cg2, ArrayAdd)
    # 断言确保 cg2 的参数等于 (M, N, P)
    assert cg2.args == (M, N, P)
    # 断言确保 cg2 的形状等于 (k, k)
    
    # 对 _array_tensor_product(X, A) 的结果进行对角线操作，并将结果赋给 expr
    expr = _array_tensor_product(_array_diagonal(X, (0, 1)), _array_diagonal(A, (0, 1)))
    # 断言确保 expr 的结果等于对 _array_tensor_product(X, A) 进行对角线操作的结果
    assert expr == _array_diagonal(_array_tensor_product(X, A), (0, 1), (2, 3))
    
    # 对 _array_tensor_product(X, A) 的结果进行对角线操作，并将结果赋给 expr1
    expr1 = _array_diagonal(_array_tensor_product(X, A), (1, 2))
    # 对 expr1 和 a 进行张量积操作，并将结果赋给 expr2
    expr2 = _array_tensor_product(expr1, a)
    # 断言确保 expr2 的结果等于对 _array_tensor_product(X, A, a) 进行对角线操作，并进行维度置换的结果
    assert expr2 == _permute_dims(_array_diagonal(_array_tensor_product(X, A, a), (1, 2)), [0, 1, 4, 2, 3])
    
    # 对 _array_tensor_product(X, A) 的结果进行张量积操作，并对结果进行数组收缩操作，并将结果赋给 expr1
    expr1 = _array_contraction(_array_tensor_product(X, A), (1, 2))
    # 对 expr1 和 a 进行张量积操作，并将结果赋给 expr2
    expr2 = _array_tensor_product(expr1, a)
    # 断言确保 expr2 的类型是 ArrayContraction 类型
    assert isinstance(expr2, ArrayContraction)
    # 断言确保 expr2 的表达式是 ArrayTensorProduct 类型
    
    # 对 _array_tensor_product(A, X, Y) 的结果进行对角线操作，并对结果进行张量积操作，并与 a, b 进行张量积操作，并将结果赋给 cg
    cg = _array_tensor_product(_array_diagonal(_array_tensor_product(A, X, Y), (0, 3), (1, 5)), a, b)
    # 断言确保 cg 的结果等于对 _array_tensor_product(A, X, Y, a, b) 进行对角线操作，并进行维度置换的结果
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(A, X, Y, a, b), (0, 3), (1, 5)), [0, 1, 6, 7, 2, 3, 4, 5])
# 定义测试函数，用于测试数组表达式的对角线操作
def test_arrayexpr_array_diagonal():
    # 计算 M 的 (1, 0) 位置处的对角线元素
    cg = _array_diagonal(M, (1, 0))
    # 断言 M 的 (1, 0) 和 (0, 1) 位置处的对角线元素相等
    assert cg == _array_diagonal(M, (0, 1))

    # 计算 M, N, P 的张量积的指定对角线元素
    cg = _array_diagonal(_array_tensor_product(M, N, P), (4, 1), (2, 0))
    # 断言 M, N, P 的张量积的不同对角线位置元素相等
    assert cg == _array_diagonal(_array_tensor_product(M, N, P), (1, 4), (0, 2))

    # 计算 M, N 的张量积的指定对角线元素，允许处理平凡对角线
    cg = _array_diagonal(_array_tensor_product(M, N), (1, 2), (3,), allow_trivial_diags=True)
    # 断言处理后的对角线元素重新排列维度后与指定维度顺序相符
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(M, N), (1, 2)), [0, 2, 1])

    # 创建具有特定形状的数组符号 Ax
    Ax = ArraySymbol("Ax", shape=(1, 2, 3, 4, 3, 5, 6, 2, 7))
    # 计算 Ax 的指定对角线元素，允许处理平凡对角线
    cg = _array_diagonal(Ax, (1, 7), (3,), (2, 4), (6,), allow_trivial_diags=True)
    # 断言处理后的对角线元素重新排列维度后与指定维度顺序相符
    assert cg == _permute_dims(_array_diagonal(Ax, (1, 7), (2, 4)), [0, 2, 4, 5, 1, 6, 3])

    # 计算 M 的指定对角线元素，允许处理平凡对角线
    cg = _array_diagonal(M, (0,), allow_trivial_diags=True)
    # 断言 M 的对角线元素重新排列维度后与指定维度顺序相符
    assert cg == _permute_dims(M, [1, 0])

    # 预期引发 ValueError 异常，因为对角线操作需要相同的维度
    raises(ValueError, lambda: _array_diagonal(M, (0, 0)))


# 定义测试函数，用于测试数组表达式的形状计算
def test_arrayexpr_array_shape():
    # 计算 M, N, P, Q 的张量积的表达式形状
    expr = _array_tensor_product(M, N, P, Q)
    # 断言表达式的形状与指定的形状 (k, k, k, k, k, k, k, k) 相等
    assert expr.shape == (k, k, k, k, k, k, k, k)
    
    # 创建具有特定形状的矩阵符号 Z
    Z = MatrixSymbol("Z", m, n)
    # 计算 M, Z 的张量积的表达式形状
    expr = _array_tensor_product(M, Z)
    # 断言表达式的形状与指定的形状 (k, k, m, n) 相等
    assert expr.shape == (k, k, m, n)
    
    # 对表达式进行张量收缩操作，计算表达式的形状
    expr2 = _array_contraction(expr, (0, 1))
    # 断言表达式的形状与指定的形状 (m, n) 相等
    assert expr2.shape == (m, n)
    
    # 对表达式进行对角线操作，计算表达式的形状
    expr2 = _array_diagonal(expr, (0, 1))
    # 断言表达式的形状与指定的形状 (m, n, k) 相等
    assert expr2.shape == (m, n, k)
    
    # 对表达式进行维度置换操作，计算表达式的形状
    exprp = _permute_dims(expr, [2, 1, 3, 0])
    # 断言表达式的形状与指定的形状 (m, k, n, k) 相等
    assert exprp.shape == (m, k, n, k)
    
    # 计算 N, Z 的张量积的表达式，然后将两个表达式相加
    expr3 = _array_tensor_product(N, Z)
    expr2 = _array_add(expr, expr3)
    # 断言表达式的形状与指定的形状 (k, k, m, n) 相等
    assert expr2.shape == (k, k, m, n)

    # 预期引发 ValueError 异常，因为张量收缩需要一致的维度
    raises(ValueError, lambda: _array_contraction(expr, (1, 2)))
    # 预期引发 ValueError 异常，因为对角线操作需要一致的维度
    raises(ValueError, lambda: _array_diagonal(expr, (1, 2)))
    # 预期引发 ValueError 异常，因为对角线操作至少需要两个轴来计算对角线
    raises(ValueError, lambda: _array_diagonal(expr, (1,)))


# 定义测试函数，用于测试数组表达式的维度置换
def test_arrayexpr_permutedims_sink():
    # 对 M, N 的张量积进行维度置换，关闭嵌套维度置换
    cg = _permute_dims(_array_tensor_product(M, N), [0, 1, 3, 2], nest_permutation=False)
    # 对置换后的结果应用 sink 函数，得到降维结果
    sunk = nest_permutation(cg)
    # 断言降维后的结果与预期结果相等
    assert sunk == _array_tensor_product(M, _permute_dims(N, [1, 0]))

    # 对 M, N 的张量积进行维度置换，关闭嵌套维度置换
    cg = _permute_dims(_array_tensor_product(M, N), [1, 0, 3, 2], nest_permutation=False)
    # 对置换后的结果应用 sink 函数，得到降维结果
    sunk = nest_permutation(cg)
    # 断言降维后的结果与预期结果相等
    assert sunk == _array_tensor_product(_permute_dims(M, [1, 0]), _permute_dims(N, [1, 0]))

    # 对 M, N 的张量积进行维度置换，关闭嵌套维度置换
    cg = _permute_dims(_array_tensor_product(M, N), [3, 2, 1, 0], nest_permutation=False)
    # 对置换后的结果应用 sink 函数，得到降维结果
    sunk = nest_permutation(cg)
    # 断言降维后的结果与预期结果相等
    assert sunk == _array_tensor_product(_permute_dims(N, [1, 0]), _permute_dims(M, [1, 0]))

    # 对 M, N 的张量积进行维度置换和收缩，关闭嵌套维度置换
    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N), (1, 2)), [1, 0], nest_permutation=False)
    # 对置换后的结果应用 sink 函数，得到降维结果
    sunk = nest_permutation(cg)
    # 断言降维后的结果与预期结果相等
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N), [[0, 3]]), (1, 2))

    # 对 M, N 的张量积
    # 对 M, N, P 进行张量乘积，并对结果进行维度收缩操作，调整维度顺序
    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N, P), (1, 2), (3, 4)), [1, 0], nest_permutation=False)
    # 对上一步操作的结果进行巢状置换
    sunk = nest_permutation(cg)
    # 断言验证结果 sunk 应与对 M, N, P 进行张量乘积并按指定维度收缩后再进行维度置换的结果相等
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N, P), [[0, 5]]), (1, 2), (3, 4))
# 定义测试函数 test_arrayexpr_push_indices_up_and_down
def test_arrayexpr_push_indices_up_and_down():

    # 创建一个包含 0 到 11 的整数列表
    indices = list(range(12))

    # 定义对角线收缩的索引列表
    contr_diag_indices = [(0, 6), (2, 8)]
    # 断言调用 ArrayContraction 类的 _push_indices_down 方法，并验证返回结果是否符合预期
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15)
    # 断言调用 ArrayContraction 类的 _push_indices_up 方法，并验证返回结果是否符合预期
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (None, 0, None, 1, 2, 3, None, 4, None, 5, 6, 7)

    # 断言调用 ArrayDiagonal 类的 _push_indices_down 方法，并验证返回结果是否符合预期
    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (1, 3, 4, 5, 7, 9, (0, 6), (2, 8), None, None, None, None)
    # 断言调用 ArrayDiagonal 类的 _push_indices_up 方法，并验证返回结果是否符合预期
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (6, 0, 7, 1, 2, 3, 6, 4, 7, 5, None, None)

    # 重新定义对角线收缩的索引列表
    contr_diag_indices = [(1, 2), (7, 8)]
    # 断言调用 ArrayContraction 类的 _push_indices_down 方法，并验证返回结果是否符合预期
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    # 断言调用 ArrayContraction 类的 _push_indices_up 方法，并验证返回结果是否符合预期
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (0, None, None, 1, 2, 3, 4, None, None, 5, 6, 7)

    # 断言调用 ArrayDiagonal 类的 _push_indices_down 方法，并验证返回结果是否符合预期
    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (0, 3, 4, 5, 6, 9, (1, 2), (7, 8), None, None, None, None)
    # 断言调用 ArrayDiagonal 类的 _push_indices_up 方法，并验证返回结果是否符合预期
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (0, 6, 6, 1, 2, 3, 4, 7, 7, 5, None, None)


# 定义测试函数 test_arrayexpr_split_multiple_contractions
def test_arrayexpr_split_multiple_contractions():

    # 创建符号矩阵变量
    a = MatrixSymbol("a", k, 1)
    b = MatrixSymbol("b", k, 1)
    A = MatrixSymbol("A", k, k)
    B = MatrixSymbol("B", k, k)
    C = MatrixSymbol("C", k, k)
    X = MatrixSymbol("X", k, k)

    # 调用 _array_tensor_product 和 _array_contraction 函数，进行数组操作和收缩
    cg = _array_contraction(_array_tensor_product(A.T, a, b, b.T, (A*X*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    # 预期的数组收缩操作
    expected = _array_contraction(_array_tensor_product(A.T, DiagMatrix(a), OneArray(1), b, b.T, (A*X*b).applyfunc(cos)), (1, 3), (2, 9), (6, 7, 10))
    # 断言调用 split_multiple_contractions 方法后是否符合预期
    assert cg.split_multiple_contractions().dummy_eq(expected)

    # 检查没有线的重叠情况
    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8), (3, 7))
    # 断言没有重叠线时，split_multiple_contractions 方法返回原值
    assert cg.split_multiple_contractions() == cg

    cg = _array_contraction(_array_tensor_product(a, b, A), (0, 2, 4), (1, 3))
    # 断言没有重叠线时，split_multiple_contractions 方法返回原值
    assert cg.split_multiple_contractions() == cg


# 定义测试函数 test_arrayexpr_nested_permutations
def test_arrayexpr_nested_permutations():

    # 调用 _permute_dims 函数，进行维度的置换操作
    cg = _permute_dims(_permute_dims(M, (1, 0)), (1, 0))
    # 断言置换操作后是否等于原始矩阵 M
    assert cg == M

    # 设置置换次数和置换列表
    times = 3
    plist1 = [list(range(6)) for i in range(times)]
    plist2 = [list(range(6)) for i in range(times)]

    # 对置换列表中的每个子列表进行随机置换操作
    for i in range(times):
        random.shuffle(plist1[i])
        random.shuffle(plist2[i])

    # 添加额外的置换列表
    plist1.append([2, 5, 4, 1, 0, 3])
    plist2.append([3, 5, 0, 4, 1, 2])

    plist1.append([2, 5, 4, 0, 3, 1])
    plist2.append([3, 0, 5, 1, 2, 4])

    plist1.append([5, 4, 2, 0, 3, 1])
    plist2.append([4, 5, 0, 2, 3, 1])

    # 使用符号 k 替换矩阵 M、N 和 P，然后将其转换为显式矩阵
    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()
    Pe = P.subs(k, 3).as_explicit()
    # 调用 tensorproduct 函数对显式矩阵进行张量积操作
    cge = tensorproduct(Me, Ne, Pe)
    # 使用 zip 函数同时迭代 plist1 和 plist2 中的每个排列数组
    for permutation_array1, permutation_array2 in zip(plist1, plist2):
        # 创建排列对象 p1 和 p2 分别用 permutation_array1 和 permutation_array2 初始化
        p1 = Permutation(permutation_array1)
        p2 = Permutation(permutation_array2)

        # 计算 cg，对 M, N, P 的张量积进行两次置换操作
        cg = _permute_dims(
            _permute_dims(
                _array_tensor_product(M, N, P),
                p1),
            p2
        )

        # 计算 result，对 M, N, P 的张量积进行 p2*p1 的置换操作
        result = _permute_dims(
            _array_tensor_product(M, N, P),
            p2*p1
        )

        # 断言 cg 和 result 相等
        assert cg == result

        # 检查 `permutedims` 在显式组件数组上的行为是否一致：
        # 计算 result1，对 cge 应用 p1 和 p2 的两次置换操作
        result1 = _permute_dims(_permute_dims(cge, p1), p2)
        # 计算 result2，对 cge 应用 p2*p1 的置换操作
        result2 = _permute_dims(cge, p2*p1)
        # 断言 result1 和 result2 相等
        assert result1 == result2
# 定义测试函数，测试数组表达式中的收缩和排列混合
def test_arrayexpr_contraction_permutation_mix():

    # 使用符号 k 替换 M 和 N 中的 k，并将结果显示为显式形式
    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()

    # 计算 cg1：对 M 和 N 的张量积进行维度排列，再进行维度收缩
    cg1 = _array_contraction(PermuteDims(_array_tensor_product(M, N), Permutation([0, 2, 1, 3])), (2, 3))
    # 计算 cg2：对 M 和 N 的张量积进行维度收缩
    cg2 = _array_contraction(_array_tensor_product(M, N), (1, 3))
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # 计算 cge1：对 Me 和 Ne 的张量积进行维度排列，再进行维度收缩
    cge1 = tensorcontraction(permutedims(tensorproduct(Me, Ne), Permutation([0, 2, 1, 3])), (2, 3))
    # 计算 cge2：对 Me 和 Ne 的张量积进行维度收缩
    cge2 = tensorcontraction(tensorproduct(Me, Ne), (1, 3))
    # 断言 cge1 和 cge2 的相等性
    assert cge1 == cge2

    # 计算 cg1：对 M 和 N 的张量积进行维度排列，再进行维度排列
    cg1 = _permute_dims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    # 计算 cg2：对 M 的张量积和对 N 进行维度排列的结果进行张量积
    cg2 = _array_tensor_product(M, _permute_dims(N, Permutation([1, 0])))
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列和维度收缩
    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    # 计算 cg2：对 M、N、P 的张量积和对 Q 进行维度排列的结果进行维度收缩
    cg2 = _array_contraction(
        _array_tensor_product(M, N, P, _permute_dims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列和维度收缩
    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 2, 7, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    # 计算 cg2：对 M、P、Q、N 的张量积进行维度收缩，并对结果进行维度排列
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(M, P, Q, N),
            (0, 1), (2, 3), (4, 7)),
        [1, 0]
    )
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列和维度收缩
    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 7, 2, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    # 计算 cg2：对 M 进行维度排列后与 N、P、Q 的张量积进行维度收缩，并对结果进行维度排列
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(_permute_dims(M, [1, 0]), N, P, Q),
            (0, 1), (3, 6), (4, 5)
        ),
        Permutation([1, 0])
    )
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2


def test_arrayexpr_permute_tensor_product():
    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 1, 0, 5, 4, 6, 7]))
    # 计算 cg2：对 N、M 进行维度排列后与 P、Q 的张量积进行维度积
    cg2 = _array_tensor_product(N, _permute_dims(M, [1, 0]),
                                    _permute_dims(P, [1, 0]), Q)
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # TODO: reverse operation starting with `PermuteDims` and getting down to `bb`...
    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 5, 0, 1, 6, 7]))
    # 计算 cg2：对 N、P、M、Q 的张量积
    cg2 = _array_tensor_product(N, P, M, Q)
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2

    # 计算 cg1：对 M、N、P、Q 的张量积进行维度排列
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 6, 5, 7, 0, 1]))
    # 断言 cg1 的表达式与对 N、P、Q、M 的张量积的表达式相等
    assert cg1.expr == _array_tensor_product(N, P, Q, M)
    # 断言 cg1 的排列与指定排列相等
    assert cg1.permutation == Permutation([0, 1, 2, 4, 3, 5, 6, 7])

    # 计算 cg1：对 N、Q、Q、M 的张量积进行维度排列和维度收缩
    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(N, Q, Q, M),
            [2, 1, 5, 4, 0, 3, 6, 7]),
        [1, 2, 6])
    # 计算 cg2：对 Q、Q、N、M 的张量积进行维度收缩，然后对结果进行维度排列
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(Q, Q, N, M), (3, 5, 6)), [0, 2, 3, 1, 4])
    # 断言 cg1 和 cg2 的相等性
    assert cg1 == cg2
    ultimate
    # 使用多次数组收缩操作对数组进行变换，以生成cg1
    cg1 = _array_contraction(
        # 第一次数组收缩操作，根据指定的维度顺序进行排列
        _array_contraction(
            # 第二次数组收缩操作，按指定维度顺序排列
            _array_contraction(
                # 第三次数组收缩操作，按指定维度顺序排列
                _array_contraction(
                    # 第四次数组收缩操作，根据指定的维度顺序排列
                    _permute_dims(
                        # 执行数组张量积操作，生成一个新的张量
                        _array_tensor_product(N, Q, Q, M),
                        # 按照给定的维度排列顺序重新排列张量
                        [2, 1, 5, 4, 0, 3, 6, 7]),
                    # 第四次数组收缩操作的维度列表
                    [1, 2, 6]),
                # 第三次数组收缩操作的维度列表
                [1, 3, 4]),
            # 第二次数组收缩操作的维度列表
            [1]),
        # 第一次数组收缩操作的维度列表
        [0])
    # 执行第二种cg2的数组收缩操作
    cg2 = _array_contraction(
        # 数组张量积操作，生成一个新的张量
        _array_tensor_product(M, N, Q, Q),
        # 按照给定的维度列表进行收缩操作
        (0, 3, 5), (1, 4, 7), (2,), (6,))
    # 断言：验证cg1与cg2是否相等
    assert cg1 == cg2
# 定义测试函数，用于验证 array 表达式规范化对角线排列和维度置换的操作
def test_arrayexpr_canonicalize_diagonal__permute_dims():
    # 计算张量积 M, Q, N, P
    tp = _array_tensor_product(M, Q, N, P)
    # 构建表达式 expr，首先对 tp 进行维度置换和对角线操作
    expr = _array_diagonal(
        _permute_dims(tp, [0, 1, 2, 4, 7, 6, 3, 5]), (2, 4, 5), (6, 7), (0, 3))
    # 计算结果表达式，直接对 tp 进行对角线操作
    result = _array_diagonal(tp, (2, 6, 7), (3, 5), (0, 4))
    # 断言两个表达式的结果相等
    assert expr == result

    # 计算张量积 M, N, P, Q
    tp = _array_tensor_product(M, N, P, Q)
    # 构建表达式 expr，对 tp 进行维度置换和对角线操作
    expr = _array_diagonal(_permute_dims(tp, [0, 5, 2, 4, 1, 6, 3, 7]), (1, 2, 6), (3, 4))
    # 计算结果表达式，对 M, P, N, Q 进行张量积后再进行对角线操作
    result = _array_diagonal(_array_tensor_product(M, P, N, Q), (3, 4, 5), (1, 2))
    # 断言两个表达式的结果相等
    assert expr == result


# 定义测试函数，用于验证 array 表达式规范化对角线收缩操作
def test_arrayexpr_canonicalize_diagonal_contraction():
    # 计算张量积 M, N, P, Q
    tp = _array_tensor_product(M, N, P, Q)
    # 构建表达式 expr，对 tp 进行对角线操作后进行收缩操作
    expr = _array_contraction(_array_diagonal(tp, (1, 3, 4)), (0, 3))
    # 计算结果表达式，对 M, N, P, Q 进行张量积后进行收缩和对角线操作
    result = _array_diagonal(_array_contraction(_array_tensor_product(M, N, P, Q), (0, 6)), (0, 2, 3))
    # 断言两个表达式的结果相等
    assert expr == result

    # 构建表达式 expr，对 tp 进行对角线操作后进行多重收缩操作
    expr = _array_contraction(_array_diagonal(tp, (0, 1, 2, 3, 7)), (1, 2, 3))
    # 计算结果表达式，对 M, N, P, Q 进行张量积后进行多重收缩操作
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 2, 3, 5, 6, 7))
    # 断言两个表达式的结果相等
    assert expr == result

    # 构建表达式 expr，对 tp 进行对角线操作后进行多重收缩操作
    expr = _array_contraction(_array_diagonal(tp, (0, 2, 6, 7)), (1, 2, 3))
    # 计算结果表达式，对 tp 进行收缩操作后再进行对角线操作
    result = _array_diagonal(_array_contraction(tp, (3, 4, 5)), (0, 2, 3, 4))
    # 断言两个表达式的结果相等
    assert expr == result

    # 计算张量积 M, N, P, Q 的对角线操作结果
    td = _array_diagonal(_array_tensor_product(M, N, P, Q), (0, 3))
    # 构建表达式 expr，对 td 进行多重收缩操作
    expr = _array_contraction(td, (2, 1), (0, 4, 6, 5, 3))
    # 计算结果表达式，对 M, N, P, Q 进行张量积后进行多重收缩操作
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 3, 5, 6, 7), (2, 4))
    # 断言两个表达式的结果相等
    assert expr == result


# 定义测试函数，用于验证 array 表达式的维度置换操作不正确时抛出异常
def test_arrayexpr_array_wrong_permutation_size():
    # 计算张量积 M, N
    cg = _array_tensor_product(M, N)
    # 断言维度置换 [1, 0] 不合法会抛出 ValueError 异常
    raises(ValueError, lambda: _permute_dims(cg, [1, 0]))
    # 断言维度置换 [1, 0, 2, 3, 5, 4] 不合法会抛出 ValueError 异常
    raises(ValueError, lambda: _permute_dims(cg, [1, 0, 2, 3, 5, 4]))


# 定义测试函数，用于验证 array 表达式的嵌套元素逐元素加法操作
def test_arrayexpr_nested_array_elementwise_add():
    # 计算张量积 M, N 和 N, M 后进行收缩操作
    cg = _array_contraction(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    # 计算结果表达式，分别对 M, N 和 N, M 进行张量积后再进行收缩操作
    result = _array_add(
        _array_contraction(_array_tensor_product(M, N), (1, 2)),
        _array_contraction(_array_tensor_product(N, M), (1, 2))
    )
    # 断言两个表达式的结果相等
    assert cg == result

    # 计算张量积 M, N 和 N, M 后进行对角线操作
    cg = _array_diagonal(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    # 计算结果表达式，分别对 M, N 和 N, M 进行张量积后再进行对角线操作
    result = _array_add(
        _array_diagonal(_array_tensor_product(M, N), (1, 2)),
        _array_diagonal(_array_tensor_product(N, M), (1, 2))
    )
    # 断言两个表达式的结果相等
    assert cg == result


# 定义测试函数，用于验证 array 表达式的零数组操作
def test_arrayexpr_array_expr_zero_array():
    # 创建指定维度的零数组和零矩阵
    za1 = ZeroArray(k, l, m, n)
    zm1 = ZeroMatrix(m, n)

    za2 = ZeroArray(k, m, m, n)
    zm2 = ZeroMatrix(m, m)
    zm3 = ZeroMatrix(k, k)

    # 断言对 M, N, za1 进行张量积后得到指定维度的零数组
    assert _array_tensor_product(M, N, za1) == ZeroArray(k, k, k, k, k, l, m, n)
    # 断言对 M, N, zm1 进行张量积后得到指定维度的零数组
    assert _array_tensor_product(M, N, zm1) == ZeroArray(k, k, k, k, m, n)

    # 断言对 za1 进行指定轴收缩后得到指定维度的零数组
    assert _array_contraction(za1, (3,)) == ZeroArray(k, l, m)
    # 断言对 zm1 进行指定轴收缩后得到指定维度的零数组
    assert _array_contraction(zm1, (1,)) ==
    # 断言：调用 _array_diagonal 函数，传入 za2 和 (1, 2) 作为参数，期望返回 ZeroArray(k, n, m)
    assert _array_diagonal(za2, (1, 2)) == ZeroArray(k, n, m)
    
    # 断言：调用 _array_diagonal 函数，传入 zm2 和 (0, 1) 作为参数，期望返回 ZeroArray(m)
    assert _array_diagonal(zm2, (0, 1)) == ZeroArray(m)
    
    # 断言：调用 _permute_dims 函数，传入 za1 和 [2, 1, 3, 0] 作为参数，期望返回 ZeroArray(m, l, n, k)
    assert _permute_dims(za1, [2, 1, 3, 0]) == ZeroArray(m, l, n, k)
    
    # 断言：调用 _permute_dims 函数，传入 zm1 和 [1, 0] 作为参数，期望返回 ZeroArray(n, m)
    assert _permute_dims(zm1, [1, 0]) == ZeroArray(n, m)
    
    # 断言：调用 _array_add 函数，传入 za1 作为参数，期望返回 za1 本身
    assert _array_add(za1) == za1
    
    # 断言：调用 _array_add 函数，传入 zm1 作为参数，期望返回 ZeroArray(m, n)
    assert _array_add(zm1) == ZeroArray(m, n)
    
    # 创建一个 tensor product 对象 tp1，调用 _array_tensor_product 函数生成，参数为 MatrixSymbol("A", k, l) 和 MatrixSymbol("B", m, n)
    tp1 = _array_tensor_product(MatrixSymbol("A", k, l), MatrixSymbol("B", m, n))
    
    # 断言：调用 _array_add 函数，传入 tp1 和 za1 作为参数，期望返回 tp1 本身
    assert _array_add(tp1, za1) == tp1
    
    # 创建一个 tensor product 对象 tp2，调用 _array_tensor_product 函数生成，参数为 MatrixSymbol("C", k, l) 和 MatrixSymbol("D", m, n)
    tp2 = _array_tensor_product(MatrixSymbol("C", k, l), MatrixSymbol("D", m, n))
    
    # 断言：调用 _array_add 函数，传入 tp1, za1 和 tp2 作为参数，期望返回 _array_add(tp1, tp2) 的结果
    assert _array_add(tp1, za1, tp2) == _array_add(tp1, tp2)
    
    # 断言：调用 _array_add 函数，传入 M 和 zm3 作为参数，期望返回 M 本身
    assert _array_add(M, zm3) == M
    
    # 断言：调用 _array_add 函数，传入 M, N 和 zm3 作为参数，期望返回 _array_add(M, N) 的结果
    assert _array_add(M, N, zm3) == _array_add(M, N)
# 定义一个测试函数，用于测试数组表达式和应用函数的相关功能
def test_arrayexpr_array_expr_applyfunc():

    # 创建一个形状为 (3, k, 2) 的数组符号 A
    A = ArraySymbol("A", (3, k, 2))
    # 创建一个对数组 A 中每个元素应用 sin 函数的元素级应用函数
    aaf = ArrayElementwiseApplyFunc(sin, A)
    # 断言应用函数的形状为 (3, k, 2)
    assert aaf.shape == (3, k, 2)


# 定义一个测试函数，用于测试数组收缩操作的编辑
def test_edit_array_contraction():

    # 对给定数组张量积进行收缩操作，并指定收缩的索引
    cg = _array_contraction(_array_tensor_product(A, B, C, D), (1, 2, 5))
    # 创建一个编辑数组收缩对象
    ecg = _EditArrayContraction(cg)
    # 断言编辑后的对象转换为数组收缩操作与原始操作一致
    assert ecg.to_array_contraction() == cg

    # 交换编辑对象中的索引位置并验证收缩操作是否正确
    ecg.args_with_ind[1], ecg.args_with_ind[2] = ecg.args_with_ind[2], ecg.args_with_ind[1]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, B, D), (1, 3, 4))

    # 获取新的收缩索引并创建新的参数对象
    ci = ecg.get_new_contraction_index()
    new_arg = _ArgE(X)
    new_arg.indices = [ci, ci]
    # 在指定位置插入新的参数并验证收缩操作
    ecg.args_with_ind.insert(2, new_arg)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, X, B, D), (1, 3, 6), (4, 5))

    # 验证获取的收缩索引列表
    assert ecg.get_contraction_indices() == [[1, 3, 6], [4, 5]]
    # 验证获取收缩索引到索引关系的列表
    assert [[tuple(j) for j in i] for i in ecg.get_contraction_indices_to_ind_rel_pos()] == [[(0, 1), (1, 1), (3, 0)], [(2, 0), (2, 1)]]
    # 验证获取特定索引位置映射的列表
    assert [list(i) for i in ecg.get_mapping_for_index(0)] == [[0, 1], [1, 1], [3, 0]]
    assert [list(i) for i in ecg.get_mapping_for_index(1)] == [[2, 0], [2, 1]]
    # 使用 lambda 函数验证异常情况下的映射获取
    raises(ValueError, lambda: ecg.get_mapping_for_index(2))

    # 移除指定位置的参数并验证收缩操作
    ecg.args_with_ind.pop(1)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 4), (2, 3))

    # 更新编辑对象中的参数索引并验证收缩操作
    ecg.args_with_ind[0].indices[1] = ecg.args_with_ind[1].indices[0]
    ecg.args_with_ind[1].indices[1] = ecg.args_with_ind[2].indices[0]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 2), (3, 4))

    # 在指定参数后插入新的参数并验证收缩操作
    ecg.insert_after(ecg.args_with_ind[1], _ArgE(C))
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, C, B, D), (1, 2), (3, 6))


# 定义一个测试函数，用于测试数组表达式不进行规范化的情况
def test_array_expressions_no_canonicalization():

    # 对给定的数组张量积进行操作
    tp = _array_tensor_product(M, N, P)

    # ArrayTensorProduct:

    # 创建一个数组张量积表达式并验证字符串表示和计算结果
    expr = ArrayTensorProduct(tp, N)
    assert str(expr) == "ArrayTensorProduct(ArrayTensorProduct(M, N, P), N)"
    assert expr.doit() == ArrayTensorProduct(M, N, P, N)

    # 创建一个数组收缩表达式并验证字符串表示和计算结果
    expr = ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)"
    assert expr.doit() == ArrayContraction(ArrayTensorProduct(M, N), (0, 1))

    # 创建一个数组对角线表达式并验证字符串表示和计算结果
    expr = ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)"
    assert expr.doit() == PermuteDims(ArrayDiagonal(ArrayTensorProduct(M, N), (0, 1)), [2, 0, 1])

    # 创建一个数组维度置换表达式并验证字符串表示和计算结果
    expr = ArrayTensorProduct(PermuteDims(M, [1, 0]), N)
    assert str(expr) == "ArrayTensorProduct(PermuteDims(M, (0 1)), N)"
    assert expr.doit() == PermuteDims(ArrayTensorProduct(M, N), [1, 0, 2, 3])

    # ArrayContraction:

    # 创建一个数组收缩表达式的嵌套表达式并进行断言验证
    expr = ArrayContraction(_array_contraction(tp, (0, 2)), (0, 1))
    assert isinstance(expr, ArrayContraction)
    assert isinstance(expr.expr, ArrayContraction)
    # 断言表达式是否等于特定字符串，验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的值
    assert expr.doit() == ArrayContraction(tp, (0, 2), (1, 3))

    # 对表达式进行多层 ArrayContraction 操作
    expr = ArrayContraction(ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1)), (0, 1))
    # 验证经过 doit() 计算后，表达式的值是否等于给定的 ArrayContraction 结果
    assert expr.doit() == ArrayContraction(tp, (0, 1), (2, 3), (4, 5))
    # 对于下一行代码的预期功能进行注释
    # assert expr._canonicalize() == ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1), (2, 3))

    # ArrayDiagonal 操作示例
    expr = ArrayContraction(ArrayDiagonal(tp, (0, 1)), (0, 1))
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayContraction(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayDiagonal 结果
    assert expr.doit() == ArrayDiagonal(ArrayContraction(ArrayTensorProduct(N, M, P), (0, 1)), (0, 1))

    expr = ArrayContraction(PermuteDims(M, [1, 0]), (0, 1))
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayContraction(PermuteDims(M, (0 1)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayContraction 结果
    assert expr.doit() == ArrayContraction(M, (0, 1))

    # ArrayDiagonal 操作示例
    expr = ArrayDiagonal(ArrayDiagonal(tp, (0, 2)), (0, 1))
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayDiagonal(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayDiagonal 结果
    assert expr.doit() == ArrayDiagonal(tp, (0, 2), (1, 3))

    expr = ArrayDiagonal(ArrayDiagonal(ArrayDiagonal(tp, (0, 1)), (0, 1)), (0, 1))
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayDiagonal 结果
    assert expr.doit() == ArrayDiagonal(tp, (0, 1), (2, 3), (4, 5))
    # 验证 _canonicalize() 方法是否与 doit() 的结果一致
    assert expr._canonicalize() == expr.doit()

    expr = ArrayDiagonal(ArrayContraction(tp, (0, 1)), (0, 1))
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于表达式本身
    assert expr.doit() == expr

    expr = ArrayDiagonal(PermuteDims(M, [1, 0]), (0, 1))
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayDiagonal(PermuteDims(M, (0 1)), (0, 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayDiagonal 结果
    assert expr.doit() == ArrayDiagonal(M, (0, 1))

    # ArrayAdd 操作示例
    expr = ArrayAdd(M)
    # 验证表达式是否是 ArrayAdd 类型的实例
    assert isinstance(expr, ArrayAdd)
    # 调用 doit() 方法计算表达式的值，并验证其是否等于 M
    assert expr.doit() == M

    expr = ArrayAdd(ArrayAdd(M, N), P)
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayAdd(ArrayAdd(M, N), P)"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayAdd 结果
    assert expr.doit() == ArrayAdd(M, N, P)

    expr = ArrayAdd(M, ArrayAdd(N, ArrayAdd(P, M)))
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayAdd 结果
    assert expr.doit() == ArrayAdd(M, N, P, M)
    # 验证 _canonicalize() 方法是否与 doit() 的结果一致
    assert expr._canonicalize() == ArrayAdd(M, N, ArrayAdd(P, M))

    expr = ArrayAdd(M, ZeroArray(k, k), N)
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "ArrayAdd(M, ZeroArray(k, k), N)"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 ArrayAdd 结果
    assert expr.doit() == ArrayAdd(M, N)

    # PermuteDims 操作示例
    expr = PermuteDims(PermuteDims(M, [1, 0]), [1, 0])
    # 验证表达式的字符串表示是否正确
    assert str(expr) == "PermuteDims(PermuteDims(M, (0 1)), (0 1))"
    # 调用 doit() 方法计算表达式的值，并验证其是否等于 M
    assert expr.doit() == M

    expr = PermuteDims(PermuteDims(PermuteDims(M, [1, 0]), [1, 0]), [1, 0])
    # 调用 doit() 方法计算表达式的值，并验证其是否等于给定的 PermuteDims 结果
    assert expr.doit() == PermuteDims(M, [1, 0])
    # 验证 _canonicalize() 方法是否与 doit() 的结果一致
    assert expr._canonicalize() == expr.doit()

    # Reshape 操作示例
    expr = Reshape(A, (k**2,))
    # 验证表达式的形状是否符合预期
    assert expr.shape == (k**2,)
    # 验证表达式是否是 Reshape 类型的实例
    assert isinstance(expr, Reshape)
def test_array_expr_construction_with_functions():

    tp = tensorproduct(M, N)
    assert tp == ArrayTensorProduct(M, N)

    expr = tensorproduct(A, eye(2))
    assert expr == ArrayTensorProduct(A, eye(2))

    # Contraction:

    expr = tensorcontraction(M, (0, 1))
    assert expr == ArrayContraction(M, (0, 1))

    expr = tensorcontraction(tp, (1, 2))
    assert expr == ArrayContraction(tp, (1, 2))

    expr = tensorcontraction(tensorcontraction(tp, (1, 2)), (0, 1))
    assert expr == ArrayContraction(tp, (0, 3), (1, 2))

    # Diagonalization:

    expr = tensordiagonal(M, (0, 1))
    assert expr == ArrayDiagonal(M, (0, 1))

    expr = tensordiagonal(tensordiagonal(tp, (0, 1)), (0, 1))
    assert expr == ArrayDiagonal(tp, (0, 1), (2, 3))

    # Permutation of dimensions:

    expr = permutedims(M, [1, 0])
    assert expr == PermuteDims(M, [1, 0])

    expr = permutedims(PermuteDims(tp, [1, 0, 2, 3]), [0, 1, 3, 2])
    assert expr == PermuteDims(tp, [1, 0, 3, 2])

    expr = PermuteDims(tp, index_order_new=["a", "b", "c", "d"], index_order_old=["d", "c", "b", "a"])
    assert expr == PermuteDims(tp, [3, 2, 1, 0])

    arr = Array(range(32)).reshape(2, 2, 2, 2, 2)
    expr = PermuteDims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])
    assert expr == PermuteDims(arr, [2, 0, 4, 3, 1])
    assert expr.as_explicit() == permutedims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])


def test_array_element_expressions():
    # Check commutative property:
    assert M[0, 0]*N[0, 0] == N[0, 0]*M[0, 0]

    # Check derivatives:
    assert M[0, 0].diff(M[0, 0]) == 1
    assert M[0, 0].diff(M[1, 0]) == 0
    assert M[0, 0].diff(N[0, 0]) == 0
    assert M[0, 1].diff(M[i, j]) == KroneckerDelta(i, 0)*KroneckerDelta(j, 1)
    assert M[0, 1].diff(N[i, j]) == 0

    K4 = ArraySymbol("K4", shape=(k, k, k, k))

    assert K4[i, j, k, l].diff(K4[1, 2, 3, 4]) == (
        KroneckerDelta(i, 1)*KroneckerDelta(j, 2)*KroneckerDelta(k, 3)*KroneckerDelta(l, 4)
    )


def test_array_expr_reshape():

    A = MatrixSymbol("A", 2, 2)
    B = ArraySymbol("B", (2, 2, 2))
    C = Array([1, 2, 3, 4])

    expr = Reshape(A, (4,))
    assert expr.expr == A
    assert expr.shape == (4,)
    assert expr.as_explicit() == Array([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])

    expr = Reshape(B, (2, 4))
    assert expr.expr == B
    assert expr.shape == (2, 4)
    ee = expr.as_explicit()
    assert isinstance(ee, ImmutableDenseNDimArray)
    assert ee.shape == (2, 4)
    assert ee == Array([[B[0, 0, 0], B[0, 0, 1], B[0, 1, 0], B[0, 1, 1]], [B[1, 0, 0], B[1, 0, 1], B[1, 1, 0], B[1, 1, 1]]])

    expr = Reshape(A, (k, 2))
    assert expr.shape == (k, 2)

    raises(ValueError, lambda: Reshape(A, (2, 3)))
    raises(ValueError, lambda: Reshape(A, (3,)))

    expr = Reshape(C, (2, 2))
    assert expr.expr == C
    assert expr.shape == (2, 2)
    assert expr.doit() == Array([[1, 2], [3, 4]])
# 定义一个测试函数，用于测试在包含显式组件数组的数组表达式中，`.as_explicit()` 方法的工作情况。
def test_array_expr_as_explicit_with_explicit_component_arrays():
    # 导入 sympy.abc 模块中的变量 x, y, z, t
    from sympy.abc import x, y, z, t
    # 创建一个二维数组 A，包含显式组件 [x, y], [z, t]
    A = Array([[x, y], [z, t]])
    # 断言：ArrayTensorProduct(A, A).as_explicit() 等于 tensorproduct(A, A)
    assert ArrayTensorProduct(A, A).as_explicit() == tensorproduct(A, A)
    # 断言：ArrayDiagonal(A, (0, 1)).as_explicit() 等于 tensordiagonal(A, (0, 1))
    assert ArrayDiagonal(A, (0, 1)).as_explicit() == tensordiagonal(A, (0, 1))
    # 断言：ArrayContraction(A, (0, 1)).as_explicit() 等于 tensorcontraction(A, (0, 1))
    assert ArrayContraction(A, (0, 1)).as_explicit() == tensorcontraction(A, (0, 1))
    # 断言：ArrayAdd(A, A).as_explicit() 等于 A + A
    assert ArrayAdd(A, A).as_explicit() == A + A
    # 断言：ArrayElementwiseApplyFunc(sin, A).as_explicit() 等于 A.applyfunc(sin)
    assert ArrayElementwiseApplyFunc(sin, A).as_explicit() == A.applyfunc(sin)
    # 断言：PermuteDims(A, [1, 0]).as_explicit() 等于 permutedims(A, [1, 0])
    assert PermuteDims(A, [1, 0]).as_explicit() == permutedims(A, [1, 0])
    # 断言：Reshape(A, [4]).as_explicit() 等于 A.reshape(4)
    assert Reshape(A, [4]).as_explicit() == A.reshape(4)
```