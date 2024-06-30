# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_convert_matrix_to_array.py`

```
# 导入必要的 SymPy 符号和类
from sympy import Lambda, KroneckerProduct
from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import Identity
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayContraction, \
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, _array_contraction, _array_tensor_product, Reshape
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

# 定义符号变量
i, j, k, l, m, n = symbols("i j k l m n")

# 创建单位矩阵和各种矩阵符号
I = Identity(k)

M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

X = MatrixSymbol("X", k, k)
Y = MatrixSymbol("Y", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)

# 定义测试函数 test_arrayexpr_convert_matrix_to_array
def test_arrayexpr_convert_matrix_to_array():

    # 测试矩阵乘法转换为数组表达式
    expr = M*N
    result = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    assert convert_matrix_to_array(expr) == result

    # 测试多个矩阵乘法转换为数组表达式
    expr = M*N*M
    result = _array_contraction(ArrayTensorProduct(M, N, M), (1, 2), (3, 4))
    assert convert_matrix_to_array(expr) == result

    # 测试转置操作转换为数组表达式
    expr = Transpose(M)
    assert convert_matrix_to_array(expr) == PermuteDims(M, [1, 0])

    # 测试矩阵乘以转置的转换为数组表达式
    expr = M*Transpose(N)
    assert convert_matrix_to_array(expr) == _array_contraction(_array_tensor_product(M, PermuteDims(N, [1, 0])), (1, 2))

    # 测试数乘矩阵的转换
    expr = 3*M*N
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    # 复杂表达式的转换测试
    expr = 3*M + N*M.T*M + 4*k*N
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    # 测试矩阵求逆后的转换
    expr = Inverse(M)*N
    rexpr = convert_array_to_matrix(convert_matrix_to_array(expr))
    assert expr == rexpr

    # 测试矩阵的幂运算转换
    expr = M**2
    rexpr = convert_array_to_matrix(convert_matrix_to_array(expr))
    assert expr == rexpr

    # 复杂表达式的转换测试
    expr = M*(2*N + 3*M)
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    # 测试矩阵的迹转换为数组表达式
    expr = Trace(M)
    result = ArrayContraction(M, (0, 1))
    assert convert_matrix_to_array(expr) == result

    # 测试数乘矩阵迹的转换为数组表达式
    expr = 3*Trace(M)
    result = ArrayContraction(ArrayTensorProduct(3, M), (0, 1))
    assert convert_matrix_to_array(expr) == result

    # 复杂表达式的迹的转换测试
    expr = 3*Trace(Trace(M) * M)
    result = ArrayContraction(ArrayTensorProduct(3, M, M), (0, 1), (2, 3))
    assert convert_matrix_to_array(expr) == result

    # 还未完成的测试表达式
    expr = 3*Trace(M)**2
    result = ArrayContraction(ArrayTensorProduct(3, M, M), (0, 1), (2, 3))
    # 计算张量乘积后进行收缩操作，指定收缩的轴，得到结果张量
    assert convert_matrix_to_array(expr) == result

    expr = HadamardProduct(M, N)
    # 计算哈达玛积（逐元素乘积），得到结果张量
    result = ArrayDiagonal(ArrayTensorProduct(M, N), (0, 2), (1, 3))
    # 对张量乘积进行对角提取操作，指定对角线索引，得到结果张量
    assert convert_matrix_to_array(expr) == result

    expr = HadamardProduct(M*N, N*M)
    # 计算两个哈达玛积的乘积，得到结果张量
    result = ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, N, M), (1, 2), (5, 6)), (0, 2), (1, 3))
    # 对四维张量进行收缩和对角提取操作，指定操作的轴，得到结果张量
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M, 2)
    # 计算张量的元素按照指数幂的哈达玛乘积，得到结果张量
    result = ArrayDiagonal(ArrayTensorProduct(M, M), (0, 2), (1, 3))
    # 对张量乘积进行对角提取操作，指定对角线索引，得到结果张量
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M*N, 2)
    # 计算张量乘积的元素按照指数幂的哈达玛乘积，得到结果张量
    result = ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, M, N), (1, 2), (5, 6)), (0, 2), (1, 3))
    # 对四维张量进行收缩和对角提取操作，指定操作的轴，得到结果张量
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M, n)
    d0 = Dummy("d0")
    # 对张量的元素按照给定的 Lambda 函数进行逐元素操作，得到结果张量
    result = ArrayElementwiseApplyFunc(Lambda(d0, d0**n), M)
    assert convert_matrix_to_array(expr).dummy_eq(result)

    expr = M**2
    assert isinstance(expr, MatPow)
    # 将矩阵提升到指数的幂次，得到结果张量
    assert convert_matrix_to_array(expr) == ArrayContraction(ArrayTensorProduct(M, M), (1, 2))

    expr = a.T*b
    cg = convert_matrix_to_array(expr)
    # 计算向量的转置乘积，得到结果张量
    assert cg == ArrayContraction(ArrayTensorProduct(a, b), (0, 2))

    expr = KroneckerProduct(A, B)
    cg = convert_matrix_to_array(expr)
    # 计算 Kronecker 乘积（张量积），得到结果张量
    assert cg == Reshape(PermuteDims(ArrayTensorProduct(A, B), [0, 2, 1, 3]), (k**2, k**2))

    expr = KroneckerProduct(A, B, C, D)
    cg = convert_matrix_to_array(expr)
    # 计算多个 Kronecker 乘积的张量积，得到结果张量
    assert cg == Reshape(PermuteDims(ArrayTensorProduct(A, B, C, D), [0, 2, 4, 6, 1, 3, 5, 7]), (k**4, k**4))
```