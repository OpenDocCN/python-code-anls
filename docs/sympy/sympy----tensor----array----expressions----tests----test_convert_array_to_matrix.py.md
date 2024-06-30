# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\tests\test_convert_array_to_matrix.py`

```
# 导入 Sympy 库中的特定模块和函数
from sympy import Lambda, S, Dummy, KroneckerProduct
# 导入符号变量相关的模块
from sympy.core.symbol import symbols
# 导入数学函数，如平方根、三角函数等
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
# 导入矩阵表达式中的哈达玛积和哈达玛幂操作
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
# 导入矩阵表达式中的特殊矩阵，如单位矩阵、全1矩阵、全0矩阵
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
# 导入矩阵元素访问相关的模块
from sympy.matrices.expressions.matexpr import MatrixElement
# 导入将矩阵转换为数组相关的函数
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
# 导入将数组转换为矩阵相关的函数及支持函数
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
    _array_diag2contr_diagmatrix, convert_array_to_matrix, _remove_trivial_dims, _array2matrix, \
    _combine_removed, identify_removable_identity_matrices, _array_contraction_to_diagonal_multiple_identity
# 导入矩阵符号的表示
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入排列组合相关的模块
from sympy.combinatorics import Permutation
# 导入对角矩阵相关的类
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
# 导入矩阵运算相关的类和函数，如迹运算、矩阵乘法、转置
from sympy.matrices import Trace, MatMul, Transpose
# 导入数组表达式相关的类和函数，如全0数组、全1数组、数组元素访问、数组符号等
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
    ArrayElement, ArraySymbol, ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _permute_dims, PermuteDims, ArrayAdd, ArrayDiagonal, ArrayContraction, ArrayTensorProduct
# 导入用于测试的函数和异常处理相关的函数
from sympy.testing.pytest import raises

# 定义符号变量
i, j, k, l, m, n = symbols("i j k l m n")

# 定义单位矩阵及维度为1的单位矩阵
I = Identity(k)
I1 = Identity(1)

# 定义矩阵符号变量
M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

# 定义矩阵符号变量
A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

# 定义矩阵符号变量
X = MatrixSymbol("X", k, k)
Y = MatrixSymbol("Y", k, k)

# 定义维度为 k x 1 的矩阵符号变量
a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)

# 定义维度为 k x 1 的矩阵符号变量
x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)

# 定义测试函数，验证数组到矩阵的转换
def test_arrayexpr_convert_array_to_matrix():

    # 构建数组表达式 _array_tensor_product(M) 并进行数组收缩操作
    cg = _array_contraction(_array_tensor_product(M), (0, 1))
    # 验证转换后的数组表达式等于矩阵 M 的迹
    assert convert_array_to_matrix(cg) == Trace(M)

    # 构建数组表达式 _array_tensor_product(M, N) 并进行多重数组收缩操作
    cg = _array_contraction(_array_tensor_product(M, N), (0, 1), (2, 3))
    # 验证转换后的数组表达式等于矩阵 M 和 N 的迹的乘积
    assert convert_array_to_matrix(cg) == Trace(M) * Trace(N)

    # 构建数组表达式 _array_tensor_product(M, N) 并进行多重数组收缩操作
    cg = _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))
    # 验证转换后的数组表达式等于矩阵 M 和 N 的乘积的迹
    assert convert_array_to_matrix(cg) == Trace(M * N)

    # 构建数组表达式 _array_tensor_product(M, N) 并进行多重数组收缩操作
    cg = _array_contraction(_array_tensor_product(M, N), (0, 2), (1, 3))
    # 验证转换后的数组表达式等于矩阵 M 和 N 转置乘积的迹
    assert convert_array_to_matrix(cg) == Trace(M * N.T)

    # 将矩阵乘积 M * N * P 转换为数组表达式，并将其转换回矩阵
    cg = convert_matrix_to_array(M * N * P)
    # 验证转换后的数组表达式等于原始矩阵乘积 M * N * P
    assert convert_array_to_matrix(cg) == M * N * P

    # 将矩阵乘积 M * N.T * P 转换为数组表达式，并将其转换回矩阵
    cg = convert_matrix_to_array(M * N.T * P)
    # 验证转换后的数组表达式等于原始矩阵乘积 M * N.T * P
    assert convert_array_to_matrix(cg) == M * N.T * P

    # 构建数组表达式 _array_tensor_product(M, N, P, Q) 并进行多重数组收缩操作
    cg = _array_contraction(_array_tensor_product(M,N,P,Q), (1, 2), (5, 6))
    # 验证转换后的数组表达式等于数组 M * N 与 P * Q 的张量积
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * N, P * Q)

    # 构建数组表达式 _array_tensor_product(-2, M, N) 并进行数组收缩操作
    cg = _array_contraction(_array_tensor_product(-2, M, N), (1, 2))
    # 验证转换后的数组表达式等于 -2 乘以矩阵 M 和 N 的乘积
    assert convert_array_to_matrix(cg) == -2 * M * N

    # 重新定义矩阵符号变量 a
    a = MatrixSymbol("a", k, 1)
    # 创建一个 k 行 1 列的符号矩阵 b
    b = MatrixSymbol("b", k, 1)
    # 创建一个 k 行 1 列的符号矩阵 c
    c = MatrixSymbol("c", k, 1)
    # 计算 cg，首先计算张量积 b ⊗ c 和 c ⊗ b 的和，然后对结果进行维度置换
    cg = PermuteDims(
        _array_contraction(
            _array_tensor_product(
                a,
                ArrayAdd(
                    _array_tensor_product(b, c),
                    _array_tensor_product(c, b),
                )
            ), (2, 4)), [0, 1, 3, 2])
    # 断言转换数组到矩阵后的结果与给定表达式 a * (b^T * c + c^T * b) 相等
    assert convert_array_to_matrix(cg) == a * (b.T * c + c.T * b)

    # 创建一个 m 行 n 列的零数组
    za = ZeroArray(m, n)
    # 断言转换数组到矩阵后的结果与给定的 m × n 零矩阵相等
    assert convert_array_to_matrix(za) == ZeroMatrix(m, n)

    # 计算 cg，其中 M 是一个矩阵，结果是 3 × M 的张量积
    cg = _array_tensor_product(3, M)
    # 断言转换数组到矩阵后的结果与给定的 3 × M 相等
    assert convert_array_to_matrix(cg) == 3 * M

    # 部分转换为矩阵乘法的表达式：计算 M × N × P × Q 的张量积，然后按照指定的维度进行缩并
    expr = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 4, 6))
    # 断言转换数组到矩阵后的结果与给定的 M^T * N * P * Q 相等
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(M.T * N, P, Q), (0, 2, 4))

    # 创建一个 k 行 1 列的符号矩阵 x
    x = MatrixSymbol("x", k, 1)
    # 计算 cg，其中包含 OneArray(1)、x、DiagMatrix(Identity(1)) 的张量积，然后进行维度置换
    cg = PermuteDims(
        _array_contraction(_array_tensor_product(OneArray(1), x, OneArray(1), DiagMatrix(Identity(1))),
                                (0, 5)), Permutation(1, 2, 3))
    # 断言转换数组到矩阵后的结果与给定的 x 相等
    assert convert_array_to_matrix(cg) == x

    # 计算表达式 M + PermuteDims(M, [1, 0])
    expr = ArrayAdd(M, PermuteDims(M, [1, 0]))
    # 断言转换数组到矩阵后的结果与给定的 M + Transpose(M) 相等
    assert convert_array_to_matrix(expr) == M + Transpose(M)
def test_arrayexpr_convert_array_to_matrix2():
    # 创建张量收缩后的结果
    cg = _array_contraction(_array_tensor_product(M, N), (1, 3))
    # 断言转换为矩阵后的结果与 M * N 的转置相等
    assert convert_array_to_matrix(cg) == M * N.T

    # 创建维度置换后的张量乘积结果
    cg = PermuteDims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    # 断言转换为矩阵后的结果与 M 与 N 的转置乘积的结果相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    # 创建 M 和 N 维度置换后的张量乘积结果
    cg = _array_tensor_product(M, PermuteDims(N, Permutation([1, 0])))
    # 断言转换为矩阵后的结果与 M 与 N 的转置乘积的结果相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    # 创建多个张量乘积和维度置换后的张量收缩结果
    cg = _array_contraction(
        PermuteDims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    # 断言转换为矩阵后的结果与 M * P 的转置 * Trace(N) * Q 的转置相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    # 创建多个张量乘积和维度置换后的张量收缩结果
    cg = _array_contraction(
        _array_tensor_product(M, N, P, PermuteDims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    # 断言转换为矩阵后的结果与 M * P 的转置 * Trace(N) * Q 的转置相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    # 创建 M 和 N 维度置换后的张量乘积结果
    cg = _array_tensor_product(M, PermuteDims(N, [1, 0]))
    # 断言转换为矩阵后的结果与 M 与 N 的转置乘积的结果相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    # 创建 M 和 N 维度置换后的张量乘积结果
    cg = _array_tensor_product(PermuteDims(M, [1, 0]), PermuteDims(N, [1, 0]))
    # 断言转换为矩阵后的结果与 M 的转置与 N 的转置乘积的结果相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(M.T, N.T)

    # 创建 N 和 M 维度置换后的张量乘积结果
    cg = _array_tensor_product(PermuteDims(N, [1, 0]), PermuteDims(M, [1, 0]))
    # 断言转换为矩阵后的结果与 N 的转置与 M 的转置乘积的结果相等
    assert convert_array_to_matrix(cg) == _array_tensor_product(N.T, M.T)

    # 创建 M 的收缩结果
    cg = _array_contraction(M, (0,), (1,))
    # 断言转换为矩阵后的结果与 OneMatrix(1, k) * M * OneMatrix(k, 1) 相等
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*M*OneMatrix(k, 1)

    # 创建 x 的收缩结果
    cg = _array_contraction(x, (0,), (1,))
    # 断言转换为矩阵后的结果与 OneMatrix(1, k) * x 相等
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*x

    # 创建 MatrixSymbol Xm 的收缩结果
    Xm = MatrixSymbol("Xm", m, n)
    cg = _array_contraction(Xm, (0,), (1,))
    # 断言转换为矩阵后的结果与 OneMatrix(1, m) * Xm * OneMatrix(n, 1) 相等
    assert convert_array_to_matrix(cg) == OneMatrix(1, m)*Xm*OneMatrix(n, 1)


def test_arrayexpr_convert_array_to_diagonalized_vector():

    # 检查在简单维度上的矩阵识别：
    cg = _array_tensor_product(a, b)
    # 断言转换为矩阵后的结果与 a * b 的转置相等
    assert convert_array_to_matrix(cg) == a * b.T

    cg = _array_tensor_product(I1, a, b)
    # 断言转换为矩阵后的结果与 a * b 的转置相等
    assert convert_array_to_matrix(cg) == a * b.T

    # 在张量乘积中识别 Trace 操作：
    cg = _array_contraction(_array_tensor_product(A, B, C), (0, 3), (1, 2))
    # 断言转换为矩阵后的结果与 Trace(A * B) * C 相等
    assert convert_array_to_matrix(cg) == Trace(A * B) * C

    # 将对角算子转换为收缩操作：
    cg = _array_diagonal(_array_tensor_product(A, a), (1, 2))
    # 断言转换为矩阵后的结果与 A * DiagMatrix(a) 相等
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (1, 3))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(a, b), (0, 2))
    # 断言转换为矩阵后的结果与 b 的转置 * DiagMatrix(a) 相等
    assert _array_diag2contr_diagmatrix(cg) == _permute_dims(
        _array_contraction(_array_tensor_product(DiagMatrix(a), OneArray(1), b), (0, 3)), [1, 2, 0]
    )
    assert convert_array_to_matrix(cg) == b.T * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(A, a), (0, 2))
    # 断言：调用 _array_diag2contr_diagmatrix 函数，并检查其返回值是否等于调用 _array_contraction 函数的结果
    # 使用 _array_tensor_product 函数生成数组，传入 A, OneArray(1), DiagMatrix(a)，对其进行 (0, 3) 的缩并操作
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (0, 3))

    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 A.T * DiagMatrix(a) 的结果
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a)

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(I, x, I1), (0, 2), (3, 5)) 的结果
    cg = _array_diagonal(_array_tensor_product(I, x, I1), (0, 2), (3, 5))
    # 断言：调用 _array_diag2contr_diagmatrix 函数，并检查其返回值是否等于调用 _array_contraction 函数的结果
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(I, OneArray(1), I1, DiagMatrix(x)), (0, 5))
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 DiagMatrix(x)
    assert convert_array_to_matrix(cg) == DiagMatrix(x)

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(I, x, A, B), (1, 2), (5, 6)) 的结果
    cg = _array_diagonal(_array_tensor_product(I, x, A, B), (1, 2), (5, 6))
    # 断言：调用 _array_diag2contr_diagmatrix 函数，并检查其返回值是否等于 _array_diagonal(_array_contraction(_array_tensor_product(I, OneArray(1), A, B, DiagMatrix(x)), (1, 7)), (5, 6)) 的结果
    assert _array_diag2contr_diagmatrix(cg) == _array_diagonal(_array_contraction(_array_tensor_product(I, OneArray(1), A, B, DiagMatrix(x)), (1, 7)), (5, 6))
    # TODO: this is returning a wrong result:
    # convert_array_to_matrix(cg)

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(I1, a, b), (1, 3, 5)) 的结果
    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3, 5))
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 a * b.T
    assert convert_array_to_matrix(cg) == a * b.T

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(I1, a, b), (1, 3)) 的结果
    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3))
    # 断言：调用 _array_diag2contr_diagmatrix 函数，并检查其返回值是否等于 _array_contraction(_array_tensor_product(OneArray(1), a, b, I1), (2, 6)) 的结果
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), a, b, I1), (2, 6))
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 a * b.T
    assert convert_array_to_matrix(cg) == a * b.T

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(x, I1), (1, 2)) 的结果
    cg = _array_diagonal(_array_tensor_product(x, I1), (1, 2))
    # 断言：检查 cg 是否是 ArrayDiagonal 的实例
    assert isinstance(cg, ArrayDiagonal)
    # 断言：检查 cg 的对角线索引是否为 ((1, 2),)
    assert cg.diagonal_indices == ((1, 2),)
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 x
    assert convert_array_to_matrix(cg) == x

    # 将 cg 更新为 _array_diagonal(_array_tensor_product(x, I), (0, 2)) 的结果
    cg = _array_diagonal(_array_tensor_product(x, I), (0, 2))
    # 断言：调用 _array_diag2contr_diagmatrix 函数，并检查其返回值是否等于 _array_contraction(_array_tensor_product(OneArray(1), I, DiagMatrix(x)), (1, 3)) 的结果
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), I, DiagMatrix(x)), (1, 3))
    # 断言：调用 convert_array_to_matrix 函数的 doit 方法，并检查其返回值是否等于 DiagMatrix(x)
    assert convert_array_to_matrix(cg).doit() == DiagMatrix(x)

    # 断言：调用 _array_diagonal 函数时会引发 ValueError 异常，使用 lambda 函数进行捕获
    raises(ValueError, lambda: _array_diagonal(x, (1,)))

    # 忽略具有缩并的单位矩阵：

    # 将 cg 更新为 _array_contraction(_array_tensor_product(I, A, I, I), (0, 2), (1, 3), (5, 7)) 的结果
    cg = _array_contraction(_array_tensor_product(I, A, I, I), (0, 2), (1, 3), (5, 7))
    # 断言：调用 cg 的 split_multiple_contractions 方法，并检查其返回值是否等于 cg 本身
    assert cg.split_multiple_contractions() == cg
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 Trace(A) * I
    assert convert_array_to_matrix(cg) == Trace(A) * I

    # 将 cg 更新为 _array_contraction(_array_tensor_product(Trace(A) * I, I, I), (1, 5), (3, 4)) 的结果
    cg = _array_contraction(_array_tensor_product(Trace(A) * I, I, I), (1, 5), (3, 4))
    # 断言：调用 cg 的 split_multiple_contractions 方法，并检查其返回值是否等于 cg 本身
    assert cg.split_multiple_contractions() == cg
    # 断言：调用 convert_array_to_matrix 函数的 doit 方法，并检查其返回值是否等于 Trace(A) * I
    assert convert_array_to_matrix(cg).doit() == Trace(A) * I

    # 在必要时添加 DiagMatrix：

    # 将 cg 更新为 _array_contraction(_array_tensor_product(A, a), (1, 2)) 的结果
    cg = _array_contraction(_array_tensor_product(A, a), (1, 2))
    # 断言：调用 cg 的 split_multiple_contractions 方法，并检查其返回值是否等于 cg 本身
    assert cg.split_multiple_contractions() == cg
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 A * a
    assert convert_array_to_matrix(cg) == A * a

    # 将 cg 更新为 _array_contraction(_array_tensor_product(A, a, B), (1, 2, 4)) 的结果
    cg = _array_contraction(_array_tensor_product(A, a, B), (1, 2, 4))
    # 断言：调用 cg 的 split_multiple_contractions 方法，并检查其返回值是否等于 _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (1, 2), (3, 5)) 的结果
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (1, 2), (3, 5))
    # 断言：调用 convert_array_to_matrix 函数，并检查其返回值是否等于 A * DiagMatrix(a) * B
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * B

    # 将 cg 更新为 _array_contraction(_array_tensor_product(A, a, B), (0, 2, 4)) 的结果
    cg = _array_contraction(_array_tensor_product(A, a, B), (0, 2, 4))
    # 断言：调用 cg 的 split_multiple_contr
    # 断言：检查 `cg` 对象的 `split_multiple_contractions` 方法的返回值是否等于经过 `_array_contraction` 函数处理后的结果
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1),
                                                    DiagMatrix(b), OneArray(1), DiagMatrix(a), OneArray(1), B),
                                                   (0, 2), (3, 5), (6, 9), (8, 12))
    
    # 断言：检查将 `cg` 对象转换为矩阵后是否等于 `A.T * DiagMatrix(a) * DiagMatrix(b) * DiagMatrix(a) * B.T`
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * DiagMatrix(b) * DiagMatrix(a) * B.T
    
    # 将 `_array_tensor_product(I1, I1, I1)` 的结果用 `_array_contraction` 函数处理后赋值给 `cg`
    cg = _array_contraction(_array_tensor_product(I1, I1, I1), (1, 2, 4))
    
    # 断言：检查 `cg` 对象的 `split_multiple_contractions` 方法的返回值是否等于经过 `_array_contraction` 函数处理后的结果
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(I1, I1, OneArray(1), I1), (1, 2), (3, 5))
    
    # 断言：检查将 `cg` 对象转换为矩阵后是否等于 `1`
    assert convert_array_to_matrix(cg) == 1
    
    # 将 `_array_tensor_product(I, I, I, I, A)` 的结果用 `_array_contraction` 函数处理后赋值给 `cg`
    cg = _array_contraction(_array_tensor_product(I, I, I, I, A), (1, 2, 8), (5, 6, 9))
    
    # 断言：检查 `cg` 对象的 `split_multiple_contractions` 方法的返回值经过 `doit` 处理后是否等于 `A`
    assert convert_array_to_matrix(cg.split_multiple_contractions()).doit() == A
    
    # 将 `_array_tensor_product(A, a, C, a, B)` 的结果用 `_array_contraction` 函数处理后赋值给 `cg`
    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8))
    
    # 用 `_array_contraction` 处理 `_array_tensor_product(A, DiagMatrix(a), OneArray(1), C, DiagMatrix(a), OneArray(1), B)` 的结果，赋值给 `expected`
    expected = _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), C, DiagMatrix(a), OneArray(1), B), (1, 3), (2, 5), (6, 7), (8, 10))
    
    # 断言：检查 `cg` 对象的 `split_multiple_contractions` 方法的返回值是否等于 `expected`
    assert cg.split_multiple_contractions() == expected
    
    # 断言：检查将 `cg` 对象转换为矩阵后是否等于 `A * DiagMatrix(a) * C * DiagMatrix(a) * B`
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * C * DiagMatrix(a) * B
    
    # 将 `_array_tensor_product(a, I1, b, I1, (a.T*b).applyfunc(cos))` 的结果用 `_array_contraction` 函数处理后赋值给 `cg`
    cg = _array_contraction(_array_tensor_product(a, I1, b, I1, (a.T*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    
    # 用 `_array_contraction` 处理 `_array_tensor_product(a, I1, OneArray(1), b, I1, OneArray(1), (a.T*b).applyfunc(cos))` 的结果，赋值给 `expected`
    expected = _array_contraction(_array_tensor_product(a, I1, OneArray(1), b, I1, OneArray(1), (a.T*b).applyfunc(cos)),
                                    (1, 3), (2, 10), (6, 8), (7, 11))
    
    # 断言：检查 `cg` 对象的 `split_multiple_contractions` 方法的返回值经过 `dummy_eq` 处理后是否等于 `expected`
    assert cg.split_multiple_contractions().dummy_eq(expected)
    
    # 断言：检查将 `cg` 对象转换为矩阵后经过 `doit` 处理后是否与 `MatMul(a, (a.T * b).applyfunc(cos), b.T)` 相等
    assert convert_array_to_matrix(cg).doit().dummy_eq(MatMul(a, (a.T * b).applyfunc(cos), b.T))
# 定义测试函数，测试将数组表达式转换为数组收缩和张量积的加法形式
def test_arrayexpr_convert_array_contraction_tp_additions():
    # 创建数组加法对象，使用 M 和 N 的张量积
    a = ArrayAdd(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    )
    # 计算 P 和 a 以及 Q 的张量积
    tp = _array_tensor_product(P, a, Q)
    # 对 tp 进行数组收缩，指定收缩的维度 (3, 4)
    expr = _array_contraction(tp, (3, 4))
    # 期望的结果，包含 P、M 和 N 的张量积加法形式，以及 Q
    expected = _array_tensor_product(
        P,
        ArrayAdd(
            _array_contraction(_array_tensor_product(M, N), (1, 2)),
            _array_contraction(_array_tensor_product(N, M), (1, 2)),
        ),
        Q
    )
    # 断言表达式和期望结果相等
    assert expr == expected
    # 断言将表达式转换为矩阵形式后的结果与 P、M*N + N*M、Q 的张量积相等
    assert convert_array_to_matrix(expr) == _array_tensor_product(P, M * N + N * M, Q)

    # 对 tp 进行多次数组收缩，指定多个收缩的维度
    expr = _array_contraction(tp, (1, 2), (3, 4), (5, 6))
    # 计算预期结果，包含 P、M 和 N 的张量积加法形式，以及 Q
    result = _array_contraction(
        _array_tensor_product(
            P,
            ArrayAdd(
                _array_contraction(_array_tensor_product(M, N), (1, 2)),
                _array_contraction(_array_tensor_product(N, M), (1, 2)),
            ),
            Q
        ), (1, 2), (3, 4))
    # 断言表达式和结果相等
    assert expr == result
    # 断言将表达式转换为矩阵形式后的结果与 P、M*N + N*M、Q 的张量积相等
    assert convert_array_to_matrix(expr) == P * (M * N + N * M) * Q


# 定义测试函数，测试将数组表达式转换为隐式矩阵乘法形式
def test_arrayexpr_convert_array_to_implicit_matmul():
    # 简化维度，表达式可以表示为矩阵形式:
    
    # 计算 a 和 b 的张量积，并将其转换为矩阵形式
    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    # 计算 a、b 和 I 的张量积，并将其转换为矩阵形式
    cg = _array_tensor_product(a, b, I)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a*b.T, I)

    # 计算 I、a 和 b 的张量积，并将其转换为矩阵形式
    cg = _array_tensor_product(I, a, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(I, a*b.T)

    # 计算 a、I 和 b 的张量积，并将其转换为矩阵形式
    cg = _array_tensor_product(a, I, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a, I, b)

    # 计算 I 和 I 的张量积，并进行维度置换，将其转换为矩阵形式
    cg = _array_contraction(_array_tensor_product(I, I), (1, 2))
    assert convert_array_to_matrix(cg) == I

    # 计算 I 和单位矩阵的张量积，并进行维度置换，将其转换为矩阵形式
    cg = PermuteDims(_array_tensor_product(I, Identity(1)), [0, 2, 1, 3])
    assert convert_array_to_matrix(cg) == I


# 定义测试函数，测试移除数组表达式中的平凡维度
def test_arrayexpr_convert_array_to_matrix_remove_trivial_dims():

    # 张量积:
    assert _remove_trivial_dims(_array_tensor_product(a, b)) == (a * b.T, [1, 3])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b)) == (a * b.T, [0, 3])
    assert _remove_trivial_dims(_array_tensor_product(a, b.T)) == (a * b.T, [1, 2])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T)) == (a * b.T, [0, 2])

    # 计算 I 和 a.T、b.T 的张量积，并移除平凡维度
    assert _remove_trivial_dims(_array_tensor_product(I, a.T, b.T)) == (_array_tensor_product(I, a * b.T), [2, 4])
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T)) == (_array_tensor_product(a.T, I, b.T), [])

    # 计算 a 和 I 的张量积，并移除平凡维度
    assert _remove_trivial_dims(_array_tensor_product(a, I)) == (_array_tensor_product(a, I), [])
    # 计算 I 和 a 的张量积，并移除平凡维度
    assert _remove_trivial_dims(_array_tensor_product(I, a)) == (_array_tensor_product(I, a), [])

    # 计算 a.T、b.T、c 和 d 的张量积，并移除平凡维度
    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T, c, d)) == (
        _array_tensor_product(a * b.T, c * d.T), [0, 2, 5, 7])
    # 计算 I、a.T、b.T、c、d 和 I 的张量积，并移除平凡维度
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T, c, d, I)) == (
        _array_tensor_product(a.T, I, b*c.T, d, I), [4, 7])
    # Addition:
    cg = ArrayAdd(_array_tensor_product(a, b), _array_tensor_product(c, d))
    # Perform addition of two tensor products: a * b + c * d
    assert _remove_trivial_dims(cg) == (a * b.T + c * d.T, [1, 3])

    # Permute Dims:
    cg = PermuteDims(_array_tensor_product(a, b), Permutation(3)(1, 2))
    # Permute dimensions of the tensor product a * b according to Permutation(3)(1, 2)
    assert _remove_trivial_dims(cg) == (a * b.T, [2, 3])

    cg = PermuteDims(_array_tensor_product(a, I, b), Permutation(5)(1, 2, 3, 4))
    # Permute dimensions of the tensor product a * I * b according to Permutation(5)(1, 2, 3, 4)
    assert _remove_trivial_dims(cg) == (cg, [])

    cg = PermuteDims(_array_tensor_product(I, b, a), Permutation(5)(1, 2, 4, 5, 3))
    # Permute dimensions of the tensor product I * b * a according to Permutation(5)(1, 2, 4, 5, 3)
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_tensor_product(I, b * a.T), [0, 2, 3, 1]), [4, 5])

    # Diagonal:
    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    # Compute diagonal of the tensor product M * a along dimensions (1, 2)
    assert _remove_trivial_dims(cg) == (cg, [])

    # Contraction:
    cg = _array_contraction(_array_tensor_product(M, a), (1, 2))
    # Contract dimensions (1, 2) of the tensor product M * a
    assert _remove_trivial_dims(cg) == (cg, [])

    # A few more cases to test the removal and shift of nested removed axes
    # with array contractions and array diagonals:
    tp = _array_tensor_product(
        OneMatrix(1, 1),
        M,
        x,
        OneMatrix(1, 1),
        Identity(1),
    )

    expr = _array_contraction(tp, (1, 8))
    # Contract dimensions (1, 8) of tensor product tp
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7]

    expr = _array_contraction(tp, (1, 8), (3, 4))
    # Contract dimensions (1, 8) and (3, 4) of tensor product tp
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5]

    expr = _array_diagonal(tp, (1, 8))
    # Compute diagonal of tensor product tp along dimensions (1, 8)
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7, 8]

    expr = _array_diagonal(tp, (1, 8), (3, 4))
    # Compute diagonal of tensor product tp along dimensions (1, 8) and (3, 4)
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5, 6]

    expr = _array_diagonal(_array_contraction(_array_tensor_product(A, x, I, I1), (1, 2, 5)), (1, 4))
    # Compute diagonal of contracted tensor product A * x * I * I1 along dimensions (1, 4)
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [2, 3]

    cg = _array_diagonal(_array_tensor_product(PermuteDims(_array_tensor_product(x, I1), Permutation(1, 2, 3)), (x.T*x).applyfunc(sqrt)), (2, 4), (3, 5))
    # Compute diagonal of tensor product with permuted dimensions and additional diagonal dimensions
    rexpr, removed = _remove_trivial_dims(cg)
    assert removed == [1, 2]

    # Contractions with identity matrices need to be followed by a permutation
    # in order
    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8))
    # Contract dimensions (1, 8) of tensor product A * B * C * M * I
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_tensor_product(A, B, C, M), [0, 2, 3, 4, 5, 6, 7, 1])
    assert removed == []

    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8), (3, 4))
    # Contract dimensions (1, 8) and (3, 4) of tensor product A * B * C * M * I
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_contraction(_array_tensor_product(A, B, C, M), (3, 4)), [0, 2, 3, 4, 5, 1])
    assert removed == []

    # Trivial matrices are sometimes inserted into MatMul expressions:
    cg = _array_tensor_product(b*b.T, a.T*a)
    # Compute tensor product b * b.T and a.T * a, then multiply them
    ret, removed = _remove_trivial_dims(cg)
    assert ret == b*a.T*a*b.T
    assert removed == [2, 3]

    Xs = ArraySymbol("X", (3, 2, k))
    # Define an array symbol X with shape (3, 2, k)
    # 调用函数 _array_tensor_product 计算张量积
    cg = _array_tensor_product(M, Xs, b.T*c, a*a.T, b*b.T, c.T*d)
    # 调用函数 _remove_trivial_dims 处理计算结果，返回结果和移除的维度列表
    ret, removed = _remove_trivial_dims(cg)
    # 使用断言检查计算结果是否符合预期
    assert ret == _array_tensor_product(M, Xs, a*b.T*c*c.T*d*a.T, b*b.T)
    # 使用断言检查移除的维度列表是否符合预期
    assert removed == [5, 6, 11, 12]

    # 调用函数 _array_tensor_product 计算张量积，再调用 _array_diagonal 对结果进行对角化处理
    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    # 使用断言检查计算结果是否符合预期
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_diagonal(_array_tensor_product(I, x), (1, 2)), Permutation(1, 2)), [1])

    # 调用函数 _array_tensor_product 计算张量积，再调用 _array_diagonal 对结果进行对角化处理
    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2))
    # 使用断言检查计算结果是否符合预期
    assert _remove_trivial_dims(expr) == (PermuteDims(_array_tensor_product(DiagMatrix(x), y), [1, 2, 3, 0]), [0])

    # 调用函数 _array_tensor_product 计算张量积，再调用 _array_diagonal 对结果进行对角化处理
    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2), (3, 4))
    # 使用断言检查计算结果是否符合预期
    assert _remove_trivial_dims(expr) == (expr, [])
# 定义测试函数，用于验证数组表达式转换为矩阵的相关功能
def test_arrayexpr_convert_array_to_matrix_diag2contraction_diagmatrix():
    # 计算数组的对角线，并进行指定的张量积和缩并操作
    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    # 将对角线数组转换为缩并对角矩阵
    res = _array_diag2contr_diagmatrix(cg)
    # 断言结果数组的形状与原数组相同
    assert res.shape == cg.shape
    # 断言结果数组等于给定的缩并张量积操作结果
    assert res == _array_contraction(_array_tensor_product(M, OneArray(1), DiagMatrix(a)), (1, 3))

    # 验证当尝试在不支持的维度上进行对角线操作时，抛出值错误异常
    raises(ValueError, lambda: _array_diagonal(_array_tensor_product(a, M), (1, 2)))

    # 继续进行其他类似的操作，并依次验证结果
    cg = _array_diagonal(_array_tensor_product(a.T, M), (1, 2))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(_array_tensor_product(OneArray(1), M, DiagMatrix(a.T)), (1, 4))

    cg = _array_diagonal(_array_tensor_product(a.T, M, N, b.T), (1, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a.T), DiagMatrix(b.T)), (1, 7), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (1, 6), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 4), (3, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (3, 6), (2, 9))

    # 创建单位矩阵和符号矩阵，并进行相关操作
    I1 = Identity(1)
    x = MatrixSymbol("x", k, 1)
    A = MatrixSymbol("A", k, k)
    cg = _array_diagonal(_array_tensor_product(x, A.T, I1), (0, 2))
    assert _array_diag2contr_diagmatrix(cg).shape == cg.shape
    assert _array2matrix(cg).shape == cg.shape


# 定义另一个测试函数，用于验证数组表达式转换为矩阵的支持函数
def test_arrayexpr_convert_array_to_matrix_support_function():
    # 验证支持函数在空列表和乘积为2k时的操作
    assert _support_function_tp1_recognize([], [2 * k]) == 2 * k

    # 验证支持函数在给定元组和列表A, 2*k, B, 3时的操作
    assert _support_function_tp1_recognize([(1, 2)], [A, 2 * k, B, 3]) == 6 * k * A * B

    # 验证支持函数在给定元组和列表A, B时的操作
    assert _support_function_tp1_recognize([(0, 3), (1, 2)], [A, B]) == Trace(A * B)

    # 继续进行其他类似的操作，并依次验证结果
    assert _support_function_tp1_recognize([(1, 2)], [A, B]) == A * B
    assert _support_function_tp1_recognize([(0, 2)], [A, B]) == A.T * B
    assert _support_function_tp1_recognize([(1, 3)], [A, B]) == A * B.T
    assert _support_function_tp1_recognize([(0, 3)], [A, B]) == A.T * B.T

    assert _support_function_tp1_recognize([(1, 2), (5, 6)], [A, B, C, D]) == _array_tensor_product(A * B, C * D)
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(0, 3), (1, 4)], [A, B, C]) == B * A * C

    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4), (7, 8)],
                                           [X, Y, A, B, C, D]) == X * Y * A * B * C * D
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4)],
                                           [X, Y, A, B, C, D]) == _array_tensor_product(X * Y * A * B, C * D)
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，并进行维度置换操作，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(1, 7), (3, 8), (4, 11)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B.T, Y * C, A.T * D.T), [0, 2, 4, 1, 3, 5]
    )
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，并进行维度置换操作，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(0, 1), (3, 6), (5, 8)], [X, A, B, C, D]) == PermuteDims(
        _array_tensor_product(Trace(X) * A * C, B * D), [0, 2, 1, 3]
    )
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [A, A, B, C, D]) == A ** 2 * B * C * D
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [X, A, B, C, D]) == X * A * B * C * D
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，并进行维度置换操作，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(1, 6), (3, 8), (5, 10)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B, Y * C, A * D), [0, 2, 4, 1, 3, 5]
    )
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，并进行维度置换操作，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3]
    )
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C * X.T * B * A * D
    
    # 使用 _support_function_tp1_recognize 函数检验给定的模式是否匹配，返回结果与预期结果进行断言比较
    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C * X.T * B * A * D
# 定义测试函数，用于测试将数组转换为Hadamard积的功能
def test_convert_array_to_hadamard_products():

    # 创建一个Hadamard积表达式
    expr = HadamardProduct(M, N)
    # 将表达式转换为数组表示
    cg = convert_matrix_to_array(expr)
    # 将数组表示转换回矩阵表示
    ret = convert_array_to_matrix(cg)
    # 断言转换后的结果与原始表达式相等
    assert ret == expr

    # 同上，但包含另一个矩阵P
    expr = HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    # 同上，但包含另一个矩阵Q和P
    expr = Q*HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    # 同上，但Hadamard积的第二个矩阵为M的转置
    expr = Q*HadamardProduct(M, N.T)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    # 多个Hadamard积的组合
    expr = HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    # 多个Hadamard积的组合，其中第一个矩阵为P的转置
    expr = P.T*HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    # ArrayDiagonal应当被转换
    cg = _array_diagonal(_array_tensor_product(M, N, Q), (1, 3), (0, 2, 4))
    ret = convert_array_to_matrix(cg)
    # 预期结果
    expected = PermuteDims(_array_diagonal(_array_tensor_product(HadamardProduct(M.T, N.T), Q), (1, 2)), [1, 0, 2])
    # 断言转换后的结果与预期结果相等
    assert expected == ret

    # 特殊情况，应当返回相同的表达式
    cg = _array_diagonal(_array_tensor_product(HadamardProduct(M, N), Q), (0, 2))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    # 带有Trace的Hadamard积
    expr = Trace(HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N.T))

    expr = Trace(A*HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M, N)*A)

    expr = Trace(HadamardProduct(A, M)*N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N)*A)

    # 这些情况不应转换为Hadamard积
    cg = _array_diagonal(_array_tensor_product(M, N), (0, 1, 2, 3))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(A), (0, 1))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 2, 4), (1, 3, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, N, P)

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 3, 4), (1, 2, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, P, N.T)

    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)


# 测试函数，用于测试识别可移除单位矩阵的功能
def test_identify_removable_identity_matrices():

    # 创建一个对角矩阵D
    D = DiagonalMatrix(MatrixSymbol("D", k, k))

    cg = _array_contraction(_array_tensor_product(A, B, I), (1, 2, 4, 5))
    expected = _array_contraction(_array_tensor_product(A, B), (1, 2))
    # 断言调用 identify_removable_identity_matrices 函数，检查结果是否符合期望值
    assert identify_removable_identity_matrices(cg) == expected

    # 使用 _array_tensor_product 函数对数组 A, B, C, I 进行张量积，并对其进行数组收缩操作
    cg = _array_contraction(_array_tensor_product(A, B, C, I), (1, 3, 5, 6, 7))
    # 计算预期的结果，对数组 A, B, C 进行张量积，并进行数组收缩操作
    expected = _array_contraction(_array_tensor_product(A, B, C), (1, 3, 5))
    # 断言调用 identify_removable_identity_matrices 函数，检查结果是否符合期望值
    assert identify_removable_identity_matrices(cg) == expected

    # 测试对角矩阵的情况：

    # 使用 _array_tensor_product 函数对数组 A, B, D 进行张量积，并对其进行数组收缩操作
    cg = _array_contraction(_array_tensor_product(A, B, D), (1, 2, 4, 5))
    # 调用 identify_removable_identity_matrices 函数，获取返回结果
    ret = identify_removable_identity_matrices(cg)
    # 计算预期的结果，对数组 A, B, D 进行张量积，并进行数组收缩操作
    expected = _array_contraction(_array_tensor_product(A, B, D), (1, 4), (2, 5))
    # 断言检查返回结果是否符合预期
    assert ret == expected

    # 使用 _array_tensor_product 函数对数组 A, B, D, M, N 进行张量积，并对其进行数组收缩操作
    cg = _array_contraction(_array_tensor_product(A, B, D, M, N), (1, 2, 4, 5, 6, 8))
    # 调用 identify_removable_identity_matrices 函数，获取返回结果
    ret = identify_removable_identity_matrices(cg)
    # 断言检查返回结果是否与输入相同
    assert ret == cg
# 定义测试函数，测试_combine_removed函数的功能
def test_combine_removed():

    # 断言测试_combine_removed函数对应用例的输出是否符合预期
    assert _combine_removed(6, [0, 1, 2], [0, 1, 2]) == [0, 1, 2, 3, 4, 5]
    assert _combine_removed(8, [2, 5], [1, 3, 4]) == [1, 2, 4, 5, 6]
    assert _combine_removed(8, [7], []) == [7]


# 定义测试函数，测试_array_contraction_to_diagonal_multiple_identities函数的功能
def test_array_contraction_to_diagonal_multiple_identities():

    # 创建表达式expr并应用_array_contraction_to_diagonal_multiple_identity函数，断言结果是否符合预期
    expr = _array_contraction(_array_tensor_product(A, B, I, C), (1, 2, 4), (5, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])
    # 断言convert_array_to_matrix函数是否能正确转换表达式expr为矩阵形式
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(A, B, C), (1, 2, 4))

    expr = _array_contraction(_array_tensor_product(A, I, I), (1, 2, 4))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (A, [2])
    assert convert_array_to_matrix(expr) == A

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 4), (3, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 3, 4, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])


# 定义测试函数，测试convert_array_element_to_matrix函数的功能
def test_convert_array_element_to_matrix():

    # 测试将ArrayElement转换为MatrixElement的功能
    expr = ArrayElement(M, (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M, i, j)

    expr = ArrayElement(_array_contraction(_array_tensor_product(M, N), (1, 3)), (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M*N.T, i, j)

    expr = ArrayElement(_array_tensor_product(M, N), (i, j, m, n))
    assert convert_array_to_matrix(expr) == expr


# 定义测试函数，测试convert_array_elementwise_function_to_matrix函数的功能
def test_convert_array_elementwise_function_to_matrix():

    # 测试将ArrayElementwiseApplyFunc转换为对应的数学表达式
    d = Dummy("d")

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x.T*y)
    assert convert_array_to_matrix(expr) == sin(x.T*y)

    expr = ArrayElementwiseApplyFunc(Lambda(d, d**2), x.T*y)
    assert convert_array_to_matrix(expr) == (x.T*y)**2

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x)
    assert convert_array_to_matrix(expr).dummy_eq(x.applyfunc(sin))

    expr = ArrayElementwiseApplyFunc(Lambda(d, 1 / (2 * sqrt(d))), x)
    assert convert_array_to_matrix(expr) == S.Half * HadamardPower(x, -S.Half)


# 定义测试函数，测试_array2matrix函数的功能
def test_array2matrix():
    # 测试将特定表达式_expr转换为期望的矩阵形式expected
    # 注意此处涉及的数学表达式转换
    expr = PermuteDims(ArrayContraction(ArrayTensorProduct(x, I, I1, x), (0, 3), (1, 7)), Permutation(2, 3))
    expected = PermuteDims(ArrayTensorProduct(x*x.T, I1), Permutation(3)(1, 2))
    assert _array2matrix(expr) == expected


# 定义测试函数，测试_remove_trivial_dims函数的功能
def test_recognize_broadcasting():

    # 测试_remove_trivial_dims函数对ArrayTensorProduct的处理能力
    expr = ArrayTensorProduct(x.T*x, A)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(x.T*x, A), [0, 1])

    expr = ArrayTensorProduct(A, x.T*x)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(A, x.T*x), [2, 3])

    expr = ArrayTensorProduct(A, B, x.T*x, C)
    assert _remove_trivial_dims(expr) == (ArrayTensorProduct(A, KroneckerProduct(B, x.T*x), C), [4, 5])

    # 测试优先选择矩阵乘法而非Kronecker乘积的情况
    expr = ArrayTensorProduct(a, b, x.T*x)
    # 断言语句，用于检查 _remove_trivial_dims 函数的返回结果是否与期望值相等
    assert _remove_trivial_dims(expr) == (a*x.T*x*b.T, [1, 3, 4, 5])
    # 断言检查通过后，表明 _remove_trivial_dims 函数的返回值与期望的元组 (a*x.T*x*b.T, [1, 3, 4, 5]) 相等
```