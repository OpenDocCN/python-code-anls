# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\__init__.py`

```
r"""
Array expressions are expressions representing N-dimensional arrays, without
evaluating them. These expressions represent in a certain way abstract syntax
trees of operations on N-dimensional arrays.

Every N-dimensional array operator has a corresponding array expression object.

Table of correspondences:

=============================== =============================
         Array operator           Array expression operator
=============================== =============================
        tensorproduct                 ArrayTensorProduct
        tensorcontraction             ArrayContraction
        tensordiagonal                ArrayDiagonal
        permutedims                   PermuteDims
=============================== =============================

Examples
========

``ArraySymbol`` objects are the N-dimensional equivalent of ``MatrixSymbol``
objects in the matrix module:

>>> from sympy.tensor.array.expressions import ArraySymbol
>>> from sympy.abc import i, j, k
>>> A = ArraySymbol("A", (3, 2, 4))
>>> A.shape
(3, 2, 4)
>>> A[i, j, k]
A[i, j, k]
>>> A.as_explicit()
[[[A[0, 0, 0], A[0, 0, 1], A[0, 0, 2], A[0, 0, 3]],
  [A[0, 1, 0], A[0, 1, 1], A[0, 1, 2], A[0, 1, 3]]],
 [[A[1, 0, 0], A[1, 0, 1], A[1, 0, 2], A[1, 0, 3]],
  [A[1, 1, 0], A[1, 1, 1], A[1, 1, 2], A[1, 1, 3]]],
 [[A[2, 0, 0], A[2, 0, 1], A[2, 0, 2], A[2, 0, 3]],
  [A[2, 1, 0], A[2, 1, 1], A[2, 1, 2], A[2, 1, 3]]]]

Component-explicit arrays can be added inside array expressions:

>>> from sympy import Array
>>> from sympy import tensorproduct
>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> a = Array([1, 2, 3])
>>> b = Array([i, j, k])
>>> expr = ArrayTensorProduct(a, b, b)
>>> expr
ArrayTensorProduct([1, 2, 3], [i, j, k], [i, j, k])
>>> expr.as_explicit() == tensorproduct(a, b, b)
True

Constructing array expressions from index-explicit forms
--------------------------------------------------------

Array expressions are index-implicit. This means they do not use any indices to
represent array operations. The function ``convert_indexed_to_array( ... )``
may be used to convert index-explicit expressions to array expressions.
It takes as input two parameters: the index-explicit expression and the order
of the indices:

>>> from sympy.tensor.array.expressions import convert_indexed_to_array
>>> from sympy import Sum
>>> A = ArraySymbol("A", (3, 3))
>>> B = ArraySymbol("B", (3, 3))
>>> convert_indexed_to_array(A[i, j], [i, j])
A
>>> convert_indexed_to_array(A[i, j], [j, i])
PermuteDims(A, (0 1))
>>> convert_indexed_to_array(A[i, j] + B[j, i], [i, j])
ArrayAdd(A, PermuteDims(B, (0 1)))
>>> convert_indexed_to_array(Sum(A[i, j]*B[j, k], (j, 0, 2)), [i, k])
ArrayContraction(ArrayTensorProduct(A, B), (1, 2))

The diagonal of a matrix in the array expression form:

>>> convert_indexed_to_array(A[i, i], [i])
ArrayDiagonal(A, (0, 1))

The trace of a matrix in the array expression form:

>>> convert_indexed_to_array(Sum(A[i, i], (i, 0, 2)), [i])
"""

注释：
# 导入 MatrixSymbol 类用于创建矩阵符号
>>> from sympy import MatrixSymbol
# 导入 ArrayContraction 类用于数组收缩操作
>>> from sympy.tensor.array.expressions import ArrayContraction
# 创建一个名为 M 的 3x3 矩阵符号
>>> M = MatrixSymbol("M", 3, 3)
# 创建另一个名为 N 的 3x3 矩阵符号
>>> N = MatrixSymbol("N", 3, 3)

# 将矩阵乘积表达为数组表达式的形式
>>> from sympy.tensor.array.expressions import convert_matrix_to_array
>>> expr = convert_matrix_to_array(M*N)
>>> expr
ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

# 将表达式转换回矩阵形式
>>> from sympy.tensor.array.expressions import convert_array_to_matrix
>>> convert_array_to_matrix(expr)
M*N

# 对剩余轴进行第二次收缩以获得 `M \cdot N` 的迹
>>> expr_tr = ArrayContraction(expr, (0, 1))
>>> expr_tr
ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (0, 1))

# 通过调用 `.doit()` 方法展开表达式并移除嵌套的数组收缩操作
>>> expr_tr.doit()
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

# 获取数组表达式的显式形式
>>> expr.as_explicit()
[[M[0, 0]*N[0, 0] + M[0, 1]*N[1, 0] + M[0, 2]*N[2, 0], M[0, 0]*N[0, 1] + M[0, 1]*N[1, 1] + M[0, 2]*N[2, 1], M[0, 0]*N[0, 2] + M[0, 1]*N[1, 2] + M[0, 2]*N[2, 2]],
 [M[1, 0]*N[0, 0] + M[1, 1]*N[1, 0] + M[1, 2]*N[2, 0], M[1, 0]*N[0, 1] + M[1, 1]*N[1, 1] + M[1, 2]*N[2, 1], M[1, 0]*N[0, 2] + M[1, 1]*N[1, 2] + M[1, 2]*N[2, 2]],
 [M[2, 0]*N[0, 0] + M[2, 1]*N[1, 0] + M[2, 2]*N[2, 0], M[2, 0]*N[0, 1] + M[2, 1]*N[1, 1] + M[2, 2]*N[2, 1], M[2, 0]*N[0, 2] + M[2, 1]*N[1, 2] + M[2, 2]*N[2, 2]]]

# 表达矩阵的迹：
>>> from sympy import Trace
>>> convert_matrix_to_array(Trace(M))
ArrayContraction(M, (0, 1))
>>> convert_matrix_to_array(Trace(M*N))
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

# 表达矩阵的转置（将表达为轴的置换操作）：
>>> convert_matrix_to_array(M.T)
PermuteDims(M, (0 1))

# 计算数组表达式的导数：
>>> from sympy.tensor.array.expressions import array_derive
>>> d = array_derive(M, M)
>>> d
PermuteDims(ArrayTensorProduct(I, I), (3)(1 2))

# 验证导数是否与使用显式矩阵计算得到的结果一致：
>>> d.as_explicit()
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]
>>> Me = M.as_explicit()
>>> Me.diff(Me)
# 定义要导出的所有数组表达式类和函数名称的列表
__all__ = [
    "ArraySymbol", "ArrayElement", "ZeroArray", "OneArray",
    "ArrayTensorProduct",
    "ArrayContraction",
    "ArrayDiagonal",
    "PermuteDims",
    "ArrayAdd",
    "ArrayElementwiseApplyFunc",
    "Reshape",
    "convert_array_to_matrix",
    "convert_matrix_to_array",
    "convert_array_to_indexed",
    "convert_indexed_to_array",
    "array_derive",
]

# 导入 SymPy 库中的各种数组表达式类和函数
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, PermuteDims, ArrayDiagonal, \
    ArrayContraction, Reshape, ArraySymbol, ArrayElement, ZeroArray, OneArray, ArrayElementwiseApplyFunc
# 导入数组表达式的导数相关函数
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
# 导入将数组转换为索引形式的函数
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed
# 导入将数组转换为矩阵形式的函数
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
# 导入将索引形式转换为数组形式的函数
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
# 导入将矩阵转换为数组形式的函数
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
```