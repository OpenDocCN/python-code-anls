# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_arrayop.py`

```
import itertools  # 导入 itertools 模块，用于高效的迭代工具
import random  # 导入 random 模块，用于生成随机数

from sympy.combinatorics import Permutation  # 导入 Permutation 类，用于处理排列组合
from sympy.combinatorics.permutations import _af_invert  # 导入 _af_invert 函数，用于处理排列的反转
from sympy.testing.pytest import raises  # 导入 raises 函数，用于测试异常

from sympy.core.function import diff  # 导入 diff 函数，用于求导数
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)  # 导入复数运算相关函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数运算相关函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数相关函数
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray  # 导入数组相关类

from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
    tensordiagonal  # 导入数组操作函数，如张量积、张量收缩、数组导数等


def test_import_NDimArray():
    from sympy.tensor.array import NDimArray  # 导入 NDimArray 类
    del NDimArray  # 删除 NDimArray，用于测试


def test_tensorproduct():
    x,y,z,t = symbols('x y z t')  # 创建符号变量 x, y, z, t
    from sympy.abc import a,b,c,d  # 导入符号变量 a, b, c, d
    assert tensorproduct() == 1  # 确认没有参数时张量积的结果为 1
    assert tensorproduct([x]) == Array([x])  # 测试单个数组的张量积
    assert tensorproduct([x], [y]) == Array([[x*y]])  # 测试两个数组的张量积
    assert tensorproduct([x], [y], [z]) == Array([[[x*y*z]]])  # 测试三个数组的张量积
    assert tensorproduct([x], [y], [z], [t]) == Array([[[[x*y*z*t]]]])  # 测试四个数组的张量积

    assert tensorproduct(x) == x  # 测试单个变量的张量积返回自身
    assert tensorproduct(x, y) == x*y  # 测试两个变量的张量积
    assert tensorproduct(x, y, z) == x*y*z  # 测试三个变量的张量积
    assert tensorproduct(x, y, z, t) == x*y*z*t  # 测试四个变量的张量积

    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        A = ArrayType([x, y])  # 创建指定类型的数组 A
        B = ArrayType([1, 2, 3])  # 创建指定类型的数组 B
        C = ArrayType([a, b, c, d])  # 创建指定类型的数组 C

        assert tensorproduct(A, B, C) == ArrayType([[[a*x, b*x, c*x, d*x], [2*a*x, 2*b*x, 2*c*x, 2*d*x], [3*a*x, 3*b*x, 3*c*x, 3*d*x]],
                                                    [[a*y, b*y, c*y, d*y], [2*a*y, 2*b*y, 2*c*y, 2*d*y], [3*a*y, 3*b*y, 3*c*y, 3*d*y]]])

        assert tensorproduct([x, y], [1, 2, 3]) == tensorproduct(A, B)  # 测试不同形式的张量积等价

        assert tensorproduct(A, 2) == ArrayType([2*x, 2*y])  # 测试数组与标量的张量积
        assert tensorproduct(A, [2]) == ArrayType([[2*x], [2*y]])  # 测试数组与数组的张量积
        assert tensorproduct([2], A) == ArrayType([[2*x, 2*y]])  # 测试标量与数组的张量积
        assert tensorproduct(a, A) == ArrayType([a*x, a*y])  # 测试标量与数组的张量积
        assert tensorproduct(a, A, B) == ArrayType([[a*x, 2*a*x, 3*a*x], [a*y, 2*a*y, 3*a*y]])  # 测试多参数的张量积
        assert tensorproduct(A, B, a) == ArrayType([[a*x, 2*a*x, 3*a*x], [a*y, 2*a*y, 3*a*y]])  # 测试多参数的张量积
        assert tensorproduct(B, a, A) == ArrayType([[a*x, a*y], [2*a*x, 2*a*y], [3*a*x, 3*a*y]])  # 测试多参数的张量积

    # tests for large scale sparse array
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        a = SparseArrayType({1:2, 3:4},(1000, 2000))  # 创建大规模稀疏数组 a
        b = SparseArrayType({1:2, 3:4},(1000, 2000))  # 创建大规模稀疏数组 b
        assert tensorproduct(a, b) == ImmutableSparseNDimArray({2000001: 4, 2000003: 8, 6000001: 8, 6000003: 16}, (1000, 2000, 1000, 2000))  # 测试大规模稀疏数组的张量积


def test_tensorcontraction():
    from sympy.abc import a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x  # 导入符号变量
    B = Array(range(18), (2, 3, 3))  # 创建指定形状的数组 B
    assert tensorcontraction(B, (1, 2)) == Array([12, 39])  # 测试张量收缩操作
    # 创建一个多维数组 C1，其元素是提供的变量 a 到 x 的组合，形状为 (2, 3, 2, 2)
    C1 = Array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x], (2, 3, 2, 2))
    
    # 断言：对数组 C1 进行张量收缩操作，按照轴 (0, 2) 收缩，期望得到的结果是一个形状为 (3, 2) 的数组，
    # 其中每个元素是对应位置上的元素之和
    assert tensorcontraction(C1, (0, 2)) == Array([[a + o, b + p], [e + s, f + t], [i + w, j + x]])
    
    # 断言：对数组 C1 进行张量收缩操作，按照轴 (0, 2, 3) 收缩，期望得到的结果是一个形状为 (3,) 的数组，
    # 其中每个元素是对应位置上的元素之和
    assert tensorcontraction(C1, (0, 2, 3)) == Array([a + p, e + t, i + x])
    
    # 断言：对数组 C1 进行张量收缩操作，按照轴 (2, 3) 收缩，期望得到的结果是一个形状为 (2, 3) 的数组，
    # 其中每个元素是对应位置上的元素之和
    assert tensorcontraction(C1, (2, 3)) == Array([[a + d, e + h, i + l], [m + p, q + t, u + x]])
# 定义一个测试函数，用于测试 derive_by_array 函数的不同用例
def test_derivative_by_array():
    # 从 sympy.abc 模块导入符号变量 i, j, t, x, y, z
    from sympy.abc import i, j, t, x, y, z

    # 定义复杂的数学表达式 bexpr
    bexpr = x*y**2*exp(z)*log(t)
    # 对 bexpr 求 sin 函数
    sexpr = sin(bexpr)
    # 对 bexpr 求 cos 函数
    cexpr = cos(bexpr)

    # 创建包含 sexpr 的 Array 对象 a
    a = Array([sexpr])

    # 断言 derive_by_array 函数对 sexpr 关于 t 的偏导数的计算结果
    assert derive_by_array(sexpr, t) == x*y**2*exp(z)*cos(x*y**2*exp(z)*log(t))/t
    # 断言 derive_by_array 函数对 sexpr 关于 [x, y, z] 的偏导数的计算结果
    assert derive_by_array(sexpr, [x, y, z]) == Array([bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr, bexpr*cexpr])
    # 断言 derive_by_array 函数对 a 关于 [x, y, z] 的偏导数的计算结果
    assert derive_by_array(a, [x, y, z]) == Array([[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr], [bexpr*cexpr]])

    # 断言 derive_by_array 函数对 sexpr 关于 [[x, y], [z, t]] 的偏导数的计算结果
    assert derive_by_array(sexpr, [[x, y], [z, t]]) == Array([[bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr], [bexpr*cexpr, bexpr/log(t)/t*cexpr]])
    # 断言 derive_by_array 函数对 a 关于 [[x, y], [z, t]] 的偏导数的计算结果
    assert derive_by_array(a, [[x, y], [z, t]]) == Array([[[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr]], [[bexpr*cexpr], [bexpr/log(t)/t*cexpr]]])
    # 断言 derive_by_array 函数对 [[x, y], [z, t]] 关于 [x, y] 的偏导数的计算结果
    assert derive_by_array([[x, y], [z, t]], [x, y]) == Array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
    # 断言 derive_by_array 函数对 [[x, y], [z, t]] 关于 [[x, y], [z, t]] 的偏导数的计算结果
    assert derive_by_array([[x, y], [z, t]], [[x, y], [z, t]]) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                                                         [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # 对 sexpr 关于 t 的常规偏导数计算结果进行断言
    assert diff(sexpr, t) == x*y**2*exp(z)*cos(x*y**2*exp(z)*log(t))/t
    # 对 sexpr 关于 Array([x, y, z]) 的常规偏导数计算结果进行断言
    assert diff(sexpr, Array([x, y, z])) == Array([bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr, bexpr*cexpr])
    # 对 a 关于 Array([x, y, z]) 的常规偏导数计算结果进行断言
    assert diff(a, Array([x, y, z])) == Array([[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr], [bexpr*cexpr]])

    # 对 sexpr 关于 Array([[x, y], [z, t]]) 的常规偏导数计算结果进行断言
    assert diff(sexpr, Array([[x, y], [z, t]])) == Array([[bexpr/x*cexpr, 2*y*bexpr/y**2*cexpr], [bexpr*cexpr, bexpr/log(t)/t*cexpr]])
    # 对 a 关于 Array([[x, y], [z, t]]) 的常规偏导数计算结果进行断言
    assert diff(a, Array([[x, y], [z, t]])) == Array([[[bexpr/x*cexpr], [2*y*bexpr/y**2*cexpr]], [[bexpr*cexpr], [bexpr/log(t)/t*cexpr]]])
    # 对 Array([[x, y], [z, t]]) 关于 Array([x, y]) 的常规偏导数计算结果进行断言
    assert diff(Array([[x, y], [z, t]]), Array([x, y])) == Array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
    # 对 Array([[x, y], [z, t]]) 关于 Array([[x, y], [z, t]]) 的常规偏导数计算结果进行断言
    assert diff(Array([[x, y], [z, t]]), Array([[x, y], [z, t]])) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                                                                         [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # 测试大规模稀疏数组的情况
    # 遍历两种 SparseArrayType，ImmutableSparseNDimArray 和 MutableSparseNDimArray
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        # 创建一个大规模稀疏数组 b
        b = MutableSparseNDimArray({0:i, 1:j}, (10000, 20000))
        # 断言 derive_by_array 函数对 b 关于 i 的偏导数的计算结果
        assert derive_by_array(b, i) == ImmutableSparseNDimArray({0: 1}, (10000, 20000))
        # 断言 derive_by_array 函数对 b 关于 (i, j) 的偏导数的计算结果
        assert derive_by_array(b, (i, j)) == ImmutableSparseNDimArray({0: 1, 200000001: 1}, (2, 10000, 20000))

    # https://github.com/sympy/sympy/issues/20655
    # 创建一个包含符号变量 x, y, z 的 Array 对象 U 和常量 E
    U = Array([x, y, z])
    E = 2
    # 断言 derive_by_array 函数对 E 关于 U 的偏导数的计算结果
    assert derive_by_array(E, U) ==  ImmutableDenseNDimArray([0, 0, 0])


# 测试处理 GitHub 问题 #10972 时的问题
def test_issue_emerged_while_discussing_10972():
    # 创建一个包含元素 [-1, 0] 的 Array 对象 ua 和元素 [[0, 1], [-1, 0]] 的 Array 对象 Fa
    ua = Array([-1,0])
    Fa = Array([[0, 1], [-1, 0]])
    # 计算 tensorproduct(Fa, ua, Fa, ua) 并断言结果
    po = tensorproduct(Fa, ua, Fa, ua)
    assert tensorcontraction(po, (1, 2), (4, 5)) == Array([[0, 0], [0, 1]])

    # 创建包含符号 'a0' 到 'a143' 的符号变量数组 sa
    sa = symbols('a0:144')
    # 创建一个包含符号变量 sa 的 Array 对象 po，其形状为 [2, 2, 3, 3, 2, 2]
    po = Array(sa, [2, 2, 3, 3, 2, 2])
    # 对 po 执行 tensorcontraction 操作，并断言结果
    assert tensorcontraction(po, (0, 1), (2, 3), (4, 5)) == sa[0] + sa[108] + sa[111] + sa[124] + sa[127] + sa[140] + sa[143] + sa[16] + sa[19] + sa[3] + sa[32] + sa[35]
    # 断言：验证张量收缩操作的结果是否等于 sa 列表中特定索引位置元素的和
    assert tensorcontraction(po, (0, 1, 4, 5), (2, 3)) == sa[0] + sa[111] + sa[127] + sa[143] + sa[16] + sa[32]
    
    # 断言：验证张量收缩操作的结果是否等于给定的二维数组
    assert tensorcontraction(po, (0, 1), (4, 5)) == Array([[sa[0] + sa[108] + sa[111] + sa[3], sa[112] + sa[115] + sa[4] + sa[7],
                                                             sa[11] + sa[116] + sa[119] + sa[8]],
                                                            [sa[12] + sa[120] + sa[123] + sa[15], sa[124] + sa[127] + sa[16] + sa[19],
                                                             sa[128] + sa[131] + sa[20] + sa[23]],
                                                            [sa[132] + sa[135] + sa[24] + sa[27], sa[136] + sa[139] + sa[28] + sa[31],
                                                             sa[140] + sa[143] + sa[32] + sa[35]]])
    
    # 断言：验证张量收缩操作的结果是否等于给定的二维数组
    assert tensorcontraction(po, (0, 1), (2, 3)) == Array([[sa[0] + sa[108] + sa[124] + sa[140] + sa[16] + sa[32], sa[1] + sa[109] + sa[125] + sa[141] + sa[17] + sa[33]],
                                                           [sa[110] + sa[126] + sa[142] + sa[18] + sa[2] + sa[34], sa[111] + sa[127] + sa[143] + sa[19] + sa[3] + sa[35]]])
def test_tensordiagonal():
    from sympy.matrices.dense import eye
    # 创建一个3x3的数组
    expr = Array(range(9)).reshape(3, 3)
    # 测试表达式的第0和第1维度生成对角元素的函数
    raises(ValueError, lambda: tensordiagonal(expr, [0], [1]))
    # 测试生成对角元素的函数，输入维度列表中有重复项
    raises(ValueError, lambda: tensordiagonal(expr, [0, 0]))
    # 对3x3单位矩阵的对角线元素进行测试
    assert tensordiagonal(eye(3), [0, 1]) == Array([1, 1, 1])
    # 对表达式expr在第0和第1维度上的对角线元素进行测试
    assert tensordiagonal(expr, [0, 1]) == Array([0, 4, 8])
    x, y, z = symbols("x y z")
    # 创建张量积
    expr2 = tensorproduct([x, y, z], expr)
    # 对expr2在第1和第2维度上的对角线元素进行测试
    assert tensordiagonal(expr2, [1, 2]) == Array([[0, 4*x, 8*x], [0, 4*y, 8*y], [0, 4*z, 8*z]])
    # 对expr2在第0和第1维度上的对角线元素进行测试
    assert tensordiagonal(expr2, [0, 1]) == Array([[0, 3*y, 6*z], [x, 4*y, 7*z], [2*x, 5*y, 8*z]])
    # 对expr2在所有维度上的对角线元素进行测试
    assert tensordiagonal(expr2, [0, 1, 2]) == Array([0, 4*y, 8*z])
    # 对expr2在第0维度上的对角线元素进行测试
    # assert tensordiagonal(expr2, [0]) == permutedims(expr2, [1, 2, 0])
    # 断言：对于给定的表达式 expr2，tensordiagonal 函数应返回其根据给定轴进行的对角化结果等于根据指定轴重新排列后的表达式 expr2
    assert tensordiagonal(expr2, [1]) == permutedims(expr2, [0, 2, 1])

    # 断言：对于给定的表达式 expr2，tensordiagonal 函数应返回其在轴 [2] 上进行的对角化结果等于表达式 expr2 自身
    assert tensordiagonal(expr2, [2]) == expr2

    # 断言：对于给定的表达式 expr2，tensordiagonal 函数应返回其在轴 [1] 上进行的对角化结果等于表达式 expr2 自身
    assert tensordiagonal(expr2, [1], [2]) == expr2

    # 断言：对于给定的表达式 expr2，tensordiagonal 函数应返回其在轴 [0] 上进行的对角化结果等于根据指定轴重新排列后的表达式 expr2
    assert tensordiagonal(expr2, [0], [1]) == permutedims(expr2, [2, 0, 1])

    # 定义符号变量
    a, b, c, X, Y, Z = symbols("a b c X Y Z")

    # 创建张量的张量积表达式 expr3
    expr3 = tensorproduct([x, y, z], [1, 2, 3], [a, b, c], [X, Y, Z])

    # 断言：对于表达式 expr3，tensordiagonal 函数应返回其在所有轴上进行的对角化结果等于给定的数组
    assert tensordiagonal(expr3, [0, 1, 2, 3]) == Array([x*a*X, 2*y*b*Y, 3*z*c*Z])

    # 断言：对于表达式 expr3，tensordiagonal 函数应返回其在轴 [0, 1] 上进行的对角化结果等于指定张量积的结果
    assert tensordiagonal(expr3, [0, 1], [2, 3]) == tensorproduct([x, 2*y, 3*z], [a*X, b*Y, c*Z])

    # 断言：对于表达式 expr3，tensordiagonal 函数应返回其在轴 [0] 上进行的对角化结果等于指定张量积的结果
    # 此断言被注释掉，可能因为涉及多维度的操作，需要进一步确认其正确性
    # assert tensordiagonal(expr3, [0], [1, 2], [3]) == tensorproduct([x, y, z], [a, 2*b, 3*c], [X, Y, Z])

    # 断言：对于表达式 expr3 的部分对角化结果，进一步对角化应得到与另一种张量积的结果相同
    assert tensordiagonal(tensordiagonal(expr3, [2, 3]), [0, 1]) == tensorproduct([a*X, b*Y, c*Z], [x, 2*y, 3*z])

    # 断言：对于不符合要求的输入，tensordiagonal 函数应引发 ValueError 异常
    raises(ValueError, lambda: tensordiagonal([[1, 2, 3], [4, 5, 6]], [0, 1]))

    # 断言：对于不符合要求的输入，tensordiagonal 函数应引发 ValueError 异常
    raises(ValueError, lambda: tensordiagonal(expr3.reshape(3, 3, 9), [1, 2]))
```