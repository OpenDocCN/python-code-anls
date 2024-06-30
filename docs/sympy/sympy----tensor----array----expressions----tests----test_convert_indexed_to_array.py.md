# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_convert_indexed_to_array.py`

```
# 导入 sympy 库中的 tanh 函数
from sympy import tanh
# 导入 sympy 库中的 Sum 类
from sympy.concrete.summations import Sum
# 导入 sympy 库中的 symbols 函数
from sympy.core.symbol import symbols
# 导入 sympy 库中的 KroneckerDelta 函数
from sympy.functions.special.tensor_functions import KroneckerDelta
# 导入 sympy 库中的 MatrixSymbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入 sympy 库中的 Identity 类
from sympy.matrices.expressions.special import Identity
# 导入 sympy 库中的 ArrayElementwiseApplyFunc 类
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
# 导入 sympy 库中的 IndexedBase 类
from sympy.tensor.indexed import IndexedBase
# 导入 sympy 库中的 Permutation 类
from sympy.combinatorics import Permutation
# 导入 sympy 库中的数组表达式类
from sympy.tensor.array.expressions.array_expressions import (
    ArrayContraction, ArrayTensorProduct, ArrayDiagonal, ArrayAdd, PermuteDims, ArrayElement,
    _array_tensor_product, _array_contraction, _array_diagonal, _array_add, _permute_dims,
    ArraySymbol, OneArray
)
# 导入 sympy 库中的矩阵和数组转换函数
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array, _convert_indexed_to_array
# 导入 sympy 的测试框架函数 raises
from sympy.testing.pytest import raises

# 定义 IndexedBase 符号 A 和 B
A, B = symbols("A B", cls=IndexedBase)
# 定义符号变量 i, j, k, l, m, n
i, j, k, l, m, n = symbols("i j k l m n")
# 定义符号变量 d0, d1, d2, d3
d0, d1, d2, d3 = symbols("d0:4")

# 创建一个 k x k 的单位矩阵并赋值给符号变量 I
I = Identity(k)

# 创建 MatrixSymbol 类对象 M, N, P, Q，每个对象都是一个 k x k 的矩阵符号
M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

# 创建 MatrixSymbol 类对象 a, b, c, d，每个对象都是一个 k x 1 的列向量符号
a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)

# 定义一个测试函数 test_arrayexpr_convert_index_to_array_support_function
def test_arrayexpr_convert_index_to_array_support_function():
    # 定义表达式 expr 为 M[i, j]
    expr = M[i, j]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (M, (i, j))

    # 定义表达式 expr 为 M[i, j]*N[k, l]
    expr = M[i, j]*N[k, l]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayTensorProduct(M, N), (i, j, k, l))

    # 定义表达式 expr 为 M[i, j]*N[j, k]
    expr = M[i, j]*N[j, k]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2)), (i, k, j))

    # 定义表达式 expr 为 Sum(M[i, j]*N[j, k], (j, 0, k-1))
    expr = Sum(M[i, j]*N[j, k], (j, 0, k-1))
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (i, k))

    # 定义表达式 expr 为 M[i, j] + N[i, j]
    expr = M[i, j] + N[i, j]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, N), (i, j))

    # 定义表达式 expr 为 M[i, j] + N[j, i]
    expr = M[i, j] + N[j, i]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, PermuteDims(N, Permutation([1, 0]))), (i, j))

    # 定义表达式 expr 为 M[i, j] + M[j, i]
    expr = M[i, j] + M[j, i]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, PermuteDims(M, Permutation([1, 0]))), (i, j))

    # 定义表达式 expr 为 (M*N*P)[i, j]
    expr = (M*N*P)[i, j]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (_array_contraction(ArrayTensorProduct(M, N, P), (1, 2), (3, 4)), (i, j))

    # 将 expr 的函数部分提取出来，忽略之前表达式中的求和
    expr = expr.function
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    ret1, ret2 = _convert_indexed_to_array(expr)
    assert ret1 == ArrayDiagonal(ArrayTensorProduct(M, N, P), (1, 2), (3, 4))
    assert str(ret2) == "(i, j, _i_1, _i_2)"

    # 定义表达式 expr 为 KroneckerDelta(i, j)*M[i, k]
    expr = KroneckerDelta(i, j)*M[i, k]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (M, ({i, j}, k))

    # 定义表达式 expr 为 KroneckerDelta(i, j)*KroneckerDelta(j, k)*M[i, l]
    expr = KroneckerDelta(i, j)*KroneckerDelta(j, k)*M[i, l]
    # 断言 _convert_indexed_to_array 函数返回值符合预期
    assert _convert_indexed_to_array(expr) == (M, ({i, j, k}, l))

    # 定义表达式 expr 为 KroneckerDelta(j, k)*(M[i, j]*N[k, l] + N[i, j]*M[k, l])
    # 断言表达式，验证_convert_indexed_to_array函数处理后的结果是否符合预期
    assert _convert_indexed_to_array(expr) == (_array_diagonal(_array_add(
            ArrayTensorProduct(M, N),  # 构建张量积M和N
            _permute_dims(ArrayTensorProduct(M, N), Permutation(0, 2)(1, 3))  # 重排张量积M和N的维度
        ), (1, 2)), (i, l, frozenset({j, k})))  # 预期结果是一个三元组，包含索引i, l和集合{j, k}
    
    # 更新表达式，计算KroneckerDelta函数的结果与矩阵M和N的乘积之和
    expr = KroneckerDelta(j, m)*KroneckerDelta(m, k)*(M[i, j]*N[k, l] + N[i, j]*M[k, l])
    assert _convert_indexed_to_array(expr) == (_array_diagonal(_array_add(
            ArrayTensorProduct(M, N),  # 构建张量积M和N
            _permute_dims(ArrayTensorProduct(M, N), Permutation(0, 2)(1, 3))  # 重排张量积M和N的维度
        ), (1, 2)), (i, l, frozenset({j, m, k})))  # 预期结果是一个三元组，包含索引i, l和集合{j, m, k}
    
    # 更新表达式，计算KroneckerDelta函数的结果与矩阵M的乘积
    expr = KroneckerDelta(i, j)*KroneckerDelta(j, k)*KroneckerDelta(k,m)*M[i, 0]*KroneckerDelta(m, n)
    assert _convert_indexed_to_array(expr) == (M, ({i, j, k, m, n}, 0))  # 预期结果是一个二元组，包含矩阵M和元组({i, j, k, m, n}, 0)
    
    # 更新表达式，计算矩阵M对角线元素M[i, i]
    expr = M[i, i]
    assert _convert_indexed_to_array(expr) == (ArrayDiagonal(M, (0, 1)), (i,))  # 预期结果是一个二元组，包含M的对角线和索引i
# 定义测试函数，用于将索引表达式转换为数组表达式，并进行验证
def test_arrayexpr_convert_indexed_to_array_expression():

    # 创建求和表达式 Sum(A[i]*B[i], (i, 0, 3))
    s = Sum(A[i]*B[i], (i, 0, 3))
    # 转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s)
    # 验证转换后的结果是否与预期的数组收缩操作相等
    assert cg == ArrayContraction(ArrayTensorProduct(A, B), (0, 1))

    # 创建矩阵乘积表达式 expr = M*N
    expr = M*N
    # 预期的数组收缩操作结果
    result = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    # 获取元素 (i, j) 并转换为数组表达式
    elem = expr[i, j]
    # 验证转换后的结果是否与预期的数组收缩操作相等
    assert convert_indexed_to_array(elem) == result

    # 创建多个矩阵乘积的表达式 expr = M*N*M
    expr = M*N*M
    # 获取元素 (i, j)
    elem = expr[i, j]
    # 预期的数组收缩操作结果
    result = _array_contraction(_array_tensor_product(M, M, N), (1, 4), (2, 5))
    # 转换元素为数组表达式
    cg = convert_indexed_to_array(elem)
    # 验证转换后的结果是否与预期的数组收缩操作相等
    assert cg == result

    # 对于复杂的表达式 (M * N * P)[i, j]，进行索引转换为数组表达式的操作
    cg = convert_indexed_to_array((M * N * P)[i, j])
    # 预期的数组收缩操作结果
    assert cg == _array_contraction(ArrayTensorProduct(M, N, P), (1, 2), (3, 4))

    # 对于带转置的表达式 (M * N.T * P)[i, j]，进行索引转换为数组表达式的操作
    cg = convert_indexed_to_array((M * N.T * P)[i, j])
    # 预期的数组收缩操作结果
    assert cg == _array_contraction(ArrayTensorProduct(M, N, P), (1, 3), (2, 4))

    # 创建带负数系数的矩阵乘积表达式 expr = -2*M*N
    expr = -2*M*N
    # 获取元素 (i, j)
    elem = expr[i, j]
    # 转换元素为数组表达式
    cg = convert_indexed_to_array(elem)
    # 验证转换后的结果是否与预期的数组收缩操作相等
    assert cg == ArrayContraction(ArrayTensorProduct(-2, M, N), (1, 2))


# 定义测试函数，用于将数组元素转换为数组表达式，并进行验证
def test_arrayexpr_convert_array_element_to_array_expression():
    # 定义数组符号 A 和 B
    A = ArraySymbol("A", (k,))
    B = ArraySymbol("B", (k,))

    # 创建求和表达式 Sum(A[i]*B[i], (i, 0, k-1))
    s = Sum(A[i]*B[i], (i, 0, k-1))
    # 转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s)
    # 验证转换后的结果是否与预期的数组对角线操作相等
    assert cg == ArrayContraction(ArrayTensorProduct(A, B), (0, 1))

    # 创建表达式 A[i]*B[i]
    s = A[i]*B[i]
    # 转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s)
    # 验证转换后的结果是否与预期的数组对角线操作相等
    assert cg == ArrayDiagonal(ArrayTensorProduct(A, B), (0, 1))

    # 创建表达式 A[i]*B[j]
    s = A[i]*B[j]
    # 指定索引顺序转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s, [i, j])
    # 验证转换后的结果是否与预期的数组张量积操作相等
    assert cg == ArrayTensorProduct(A, B)
    # 指定不同的索引顺序转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s, [j, i])
    # 验证转换后的结果是否与预期的数组张量积操作相等
    assert cg == ArrayTensorProduct(B, A)

    # 创建带双曲正切函数的表达式 tanh(A[i]*B[j])
    s = tanh(A[i]*B[j])
    # 指定索引顺序转换索引表达式为数组表达式
    cg = convert_indexed_to_array(s, [i, j])
    # 验证转换后的结果是否与预期的数组元素逐个应用函数操作相等
    assert cg.dummy_eq(ArrayElementwiseApplyFunc(tanh, ArrayTensorProduct(A, B)))


# 定义测试函数，用于将索引表达式转换为数组表达式，再转换回矩阵，并进行验证
def test_arrayexpr_convert_indexed_to_array_and_back_to_matrix():

    # 创建矩阵乘积表达式 expr = a.T*b
    expr = a.T*b
    # 获取元素 (0, 0)
    elem = expr[0, 0]
    # 转换元素为数组表达式
    cg = convert_indexed_to_array(elem)
    # 验证转换后的结果是否与预期的数组元素操作相等
    assert cg == ArrayElement(ArrayContraction(ArrayTensorProduct(a, b), (0, 2)), [0, 0])

    # 创建矩阵加法表达式 expr = M[i,j] + N[i,j]
    expr = M[i,j] + N[i,j]
    # 执行索引转换为数组表达式，并返回两部分结果
    p1, p2 = _convert_indexed_to_array(expr)
    # 验证数组转换为矩阵后的第一部分结果是否与预期的矩阵加法操作相等
    assert convert_array_to_matrix(p1) == M + N

    # 创建矩阵加法表达式 expr = M[i,j] + N[j,i]
    expr = M[i,j] + N[j,i]
    # 执行索引转换为数组表达式，并返回两部分结果
    p1, p2 = _convert_indexed_to_array(expr)
    # 验证数组转换为矩阵后的第一部分结果是否与预期的矩阵加法操作相等
    assert convert_array_to_matrix(p1) == M + N.T

    # 创建矩阵乘积与加法表达式 expr = M[i,j]*N[k,l] + N[i,j]*M[k,l]
    expr = M[i,j]*N[k,l] + N[i,j]*M[k,l]
    # 执行索引转换为数组表达式，并返回两部分结果
    p1, p2 = _convert_indexed_to_array(expr)
    # 验证数组转换为矩阵后的第一部分结果是否与预期的矩阵加法和张量积操作相等
    assert convert_array_to_matrix(p1) == ArrayAdd(
        ArrayTensorProduct(M, N),
        ArrayTensorProduct(N, M))

    # 创建三个矩阵乘积表达式 expr = (M*N*P)[i, j]
    expr = (M*N*P)[i, j]
    # 执行索引转换为数组表达式，并返回两部分结果
    p1, p2 = _convert_indexed_to_array(expr)
    # 验证数组转换为矩阵后的第一部分结果是否与预期的矩阵乘积操作相等
    assert convert_array_to_matrix(p1) == M * N * P

    # 创建求和表达式 expr = Sum(M[i,j]*(N*P
    # 调用 raises 函数，验证 convert_indexed_to_array 函数在处理表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    # 创建一个求和表达式，对角线元素之和，i 从 0 到 k
    expr = Sum(M[i, i], (i, 0, k))
    # 再次调用 raises 函数，验证 convert_indexed_to_array 函数处理上述表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    # 创建一个求和表达式，对角线元素之和，i 从 1 到 k-1
    expr = Sum(M[i, i], (i, 1, k-1))
    # 再次调用 raises 函数，验证 convert_indexed_to_array 函数处理上述表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    
    # 创建一个求和表达式，M[i, j]*N[j, m] 的总和，其中 j 从 0 到 4
    expr = Sum(M[i, j]*N[j, m], (j, 0, 4))
    # 再次调用 raises 函数，验证 convert_indexed_to_array 函数处理上述表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    # 创建一个求和表达式，M[i, j]*N[j, m] 的总和，其中 j 从 0 到 k
    expr = Sum(M[i, j]*N[j, m], (j, 0, k))
    # 再次调用 raises 函数，验证 convert_indexed_to_array 函数处理上述表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    # 创建一个求和表达式，M[i, j]*N[j, m] 的总和，其中 j 从 1 到 k-1
    expr = Sum(M[i, j]*N[j, m], (j, 1, k-1))
    # 再次调用 raises 函数，验证 convert_indexed_to_array 函数处理上述表达式时是否会引发 ValueError 异常
    raises(ValueError, lambda: convert_indexed_to_array(expr))
# 定义一个测试函数，用于验证将索引表达式转换为数组广播形式的功能
def test_arrayexpr_convert_indexed_to_array_broadcast():
    # 创建一个形状为 (3, 3) 的数组符号 A
    A = ArraySymbol("A", (3, 3))
    # 创建一个形状为 (3, 3) 的数组符号 B
    B = ArraySymbol("B", (3, 3))

    # 构建一个索引表达式 expr，表示 A[i, j] + B[k, l]
    expr = A[i, j] + B[k, l]
    # 创建一个全一数组 OneArray，形状为 (3, 3)
    O2 = OneArray(3, 3)
    # 构建预期的数组表达式 expected，表示 ArrayAdd(ArrayTensorProduct(A, O2), ArrayTensorProduct(O2, B))
    expected = ArrayAdd(ArrayTensorProduct(A, O2), ArrayTensorProduct(O2, B))
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果是否等于 expected
    assert convert_indexed_to_array(expr) == expected
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果（指定索引顺序）是否等于 expected
    assert convert_indexed_to_array(expr, [i, j, k, l]) == expected
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果（不同的索引顺序）是否正确
    assert convert_indexed_to_array(expr, [l, k, i, j]) == ArrayAdd(PermuteDims(ArrayTensorProduct(O2, A), [1, 0, 2, 3]), PermuteDims(ArrayTensorProduct(B, O2), [1, 0, 2, 3]))

    # 更新表达式 expr 为 A[i, j] + B[j, k]
    expr = A[i, j] + B[j, k]
    # 创建一个形状为 (3) 的全一数组 OneArray
    O1 = OneArray(3)
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果（指定索引顺序）是否等于 ArrayAdd(ArrayTensorProduct(A, O1), ArrayTensorProduct(O1, B))
    assert convert_indexed_to_array(expr, [i, j, k]) == ArrayAdd(ArrayTensorProduct(A, O1), ArrayTensorProduct(O1, B))

    # 创建形状为 (d0, d1) 的数组符号 C 和形状为 (d3, d1) 的数组符号 D
    C = ArraySymbol("C", (d0, d1))
    D = ArraySymbol("D", (d3, d1))

    # 更新表达式 expr 为 C[i, j] + D[k, j]
    expr = C[i, j] + D[k, j]
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果（指定索引顺序）是否等于 ArrayAdd(ArrayTensorProduct(C, OneArray(d3)), PermuteDims(ArrayTensorProduct(OneArray(d0), D), [0, 2, 1]))

    # 创建形状为 (5, 3) 的数组符号 X
    X = ArraySymbol("X", (5, 3))

    # 更新表达式 expr 为 X[i, n] - X[j, n]
    expr = X[i, n] - X[j, n]
    # 验证 convert_indexed_to_array 函数对 expr 的转换结果（指定索引顺序）是否等于 ArrayAdd(ArrayTensorProduct(-1, OneArray(5), X), PermuteDims(ArrayTensorProduct(X, OneArray(5)), [0, 2, 1]))

    # 验证 convert_indexed_to_array 函数对于 C[i, j] + D[i, j] 这类不支持的表达式能够引发 ValueError
    raises(ValueError, lambda: convert_indexed_to_array(C[i, j] + D[i, j]))
```