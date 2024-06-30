# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_convert_array_to_indexed.py`

```
# 从 sympy 库中导入所需的类和函数
from sympy import Sum, Dummy, sin
from sympy.tensor.array.expressions import ArraySymbol, ArrayTensorProduct, ArrayContraction, PermuteDims, \
    ArrayDiagonal, ArrayAdd, OneArray, ZeroArray, convert_indexed_to_array, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed

# 从 sympy.abc 模块中导入常用的符号
from sympy.abc import i, j, k, l, m, n, o

# 定义测试函数 test_convert_array_to_indexed_main
def test_convert_array_to_indexed_main():
    # 创建三个数组符号 A, B, C，分别表示形状为 (3, 3, 3)、(3, 3) 和 (3, 3) 的数组
    A = ArraySymbol("A", (3, 3, 3))
    B = ArraySymbol("B", (3, 3))
    C = ArraySymbol("C", (3, 3))

    # 创建一个虚拟变量 d_
    d_ = Dummy("d_")

    # 断言：测试 convert_array_to_indexed 函数是否正确将 A 转换为索引形式 A[i, j, k]
    assert convert_array_to_indexed(A, [i, j, k]) == A[i, j, k]

    # 创建一个数组张量乘积表达式 ArrayTensorProduct(A, B, C)
    expr = ArrayTensorProduct(A, B, C)
    # 将数组张量乘积表达式转换为索引形式 A[i, j, k]*B[l, m]*C[n, o]
    conv = convert_array_to_indexed(expr, [i,j,k,l,m,n,o])
    assert conv == A[i,j,k]*B[l,m]*C[n,o]
    # 断言：测试 convert_indexed_to_array 函数是否能将索引形式还原为原始表达式
    assert convert_indexed_to_array(conv, [i,j,k,l,m,n,o]) == expr

    # 创建一个数组收缩表达式 ArrayContraction(A, (0, 2))
    expr = ArrayContraction(A, (0, 2))
    # 将数组收缩表达式转换为索引形式 Sum(A[d_, i, d_], (d_, 0, 2))
    assert convert_array_to_indexed(expr, [i]).dummy_eq(Sum(A[d_, i, d_], (d_, 0, 2)))

    # 创建一个数组对角线表达式 ArrayDiagonal(A, (0, 2))
    expr = ArrayDiagonal(A, (0, 2))
    # 将数组对角线表达式转换为索引形式 A[j, i, j]
    assert convert_array_to_indexed(expr, [i, j]) == A[j, i, j]

    # 创建一个数组维度置换表达式 PermuteDims(A, [1, 2, 0])
    expr = PermuteDims(A, [1, 2, 0])
    # 将数组维度置换表达式转换为索引形式 A[k, i, j]
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == A[k, i, j]
    # 断言：测试 convert_indexed_to_array 函数是否能将索引形式还原为原始表达式
    assert convert_indexed_to_array(conv, [i, j, k]) == expr

    # 创建一个数组加法表达式 ArrayAdd(B, C, PermuteDims(C, [1, 0]))
    expr = ArrayAdd(B, C, PermuteDims(C, [1, 0]))
    # 将数组加法表达式转换为索引形式 B[i, j] + C[i, j] + C[j, i]
    conv = convert_array_to_indexed(expr, [i, j])
    assert conv == B[i, j] + C[i, j] + C[j, i]
    # 断言：测试 convert_indexed_to_array 函数是否能将索引形式还原为原始表达式
    assert convert_indexed_to_array(conv, [i, j]) == expr

    # 创建一个数组逐元素函数应用表达式 ArrayElementwiseApplyFunc(sin, A)
    expr = ArrayElementwiseApplyFunc(sin, A)
    # 将数组逐元素函数应用表达式转换为索引形式 sin(A[i, j, k])
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == sin(A[i, j, k])
    # 断言：测试 convert_indexed_to_array 函数是否能将索引形式还原为原始表达式
    assert convert_indexed_to_array(conv, [i, j, k]).dummy_eq(expr)

    # 断言：测试 OneArray(3, 3) 的索引化结果是否为 1
    assert convert_array_to_indexed(OneArray(3, 3), [i, j]) == 1
    # 断言：测试 ZeroArray(3, 3) 的索引化结果是否为 0
    assert convert_array_to_indexed(ZeroArray(3, 3), [i, j]) == 0

    # 创建一个数组重塑表达式 Reshape(A, (27,))
    expr = Reshape(A, (27,))
    # 将数组重塑表达式转换为索引形式 A[i // 9, i // 3 % 3, i % 3]
    assert convert_array_to_indexed(expr, [i]) == A[i // 9, i // 3 % 3, i % 3]

    # 创建一个数组符号 X，形状为 (2, 3, 4, 5, 6)
    X = ArraySymbol("X", (2, 3, 4, 5, 6))
    # 创建一个数组重塑表达式 Reshape(X, (2*3*4*5*6,))
    expr = Reshape(X, (2*3*4*5*6,))
    # 将数组重塑表达式转换为索引形式 X[i // 360, i // 120 % 3, i // 30 % 4, i // 6 % 5, i % 6]
    assert convert_array_to_indexed(expr, [i]) == X[i // 360, i // 120 % 3, i // 30 % 4, i // 6 % 5, i % 6]

    # 创建一个数组重塑表达式 Reshape(X, (4, 9, 2, 2, 5))
    expr = Reshape(X, (4, 9, 2, 2, 5))
    # 创建一个索引计算 one_index
    one_index = 180*i + 20*j + 10*k + 5*l + m
    # 创建期望的索引结果 expected
    expected = X[one_index // (3*4*5*6), one_index // (4*5*6) % 3, one_index // (5*6) % 4, one_index // 6 % 5, one_index % 6]
    # 将数组重塑表达式转换为索引形式 expected
    assert convert_array_to_indexed(expr, [i, j, k, l, m]) == expected

    # 创建一个数组符号 X，形状为 (2*3*5,)
    X = ArraySymbol("X", (2*3*5,))
    # 创建一个数组重塑表达式 Reshape(X, (2, 3, 5))
    expr = Reshape(X, (2, 3, 5))
    # 将数组重塑表达式转换为索引形式 X[15*i + 5*j + k]
    assert convert_array_to_indexed(expr, [i, j, k]) == X[15*i + 5*j + k]
```