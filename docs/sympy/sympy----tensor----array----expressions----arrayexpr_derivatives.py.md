# `D:\src\scipysrc\sympy\sympy\tensor\array\expressions\arrayexpr_derivatives.py`

```
# 导入操作符模块
import operator
# 导入 functools 模块中的 reduce 和 singledispatch 函数
from functools import reduce, singledispatch

# 导入 sympy 核心表达式模块中的 Expr 类和 S 单例对象
from sympy.core.expr import Expr
from sympy.core.singleton import S
# 导入 sympy 矩阵表达式模块中的各种类和函数
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import (MatrixExpr, MatrixSymbol)
from sympy.matrices.expressions.special import Identity, OneMatrix
from sympy.matrices.expressions.transpose import Transpose
# 导入 sympy 排列组合模块中的 _af_invert 函数
from sympy.combinatorics.permutations import _af_invert
# 导入 sympy 张量数组表达式模块中的各种类和函数
from sympy.tensor.array.expressions.array_expressions import (
    _ArrayExpr, ZeroArray, ArraySymbol, ArrayTensorProduct, ArrayAdd,
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, get_rank,
    get_shape, ArrayContraction, _array_tensor_product, _array_contraction,
    _array_diagonal, _array_add, _permute_dims, Reshape)
# 导入从矩阵到数组的转换函数
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

# singledispatch 装饰的多态函数，用于处理数组表达式的导数（梯度）
@singledispatch
def array_derive(expr, x):
    """
    数组表达式的导数（梯度）。
    """
    raise NotImplementedError(f"not implemented for type {type(expr)}")

# 处理 Expr 类型的数组表达式的导数，返回相同形状的零数组
@array_derive.register(Expr)
def _(expr: Expr, x: _ArrayExpr):
    return ZeroArray(*x.shape)

# 处理 ArrayTensorProduct 类型的数组表达式的导数
@array_derive.register(ArrayTensorProduct)
def _(expr: ArrayTensorProduct, x: Expr):
    args = expr.args
    addend_list = []
    for i, arg in enumerate(expr.args):
        darg = array_derive(arg, x)
        if darg == 0:
            continue
        args_prev = args[:i]
        args_succ = args[i+1:]
        shape_prev = reduce(operator.add, map(get_shape, args_prev), ())
        shape_succ = reduce(operator.add, map(get_shape, args_succ), ())
        # 计算添加项，包括导数和张量积的乘积
        addend = _array_tensor_product(*args_prev, darg, *args_succ)
        # 计算维度的排列
        tot1 = len(get_shape(x))
        tot2 = tot1 + len(shape_prev)
        tot3 = tot2 + len(get_shape(arg))
        tot4 = tot3 + len(shape_succ)
        perm = list(range(tot1, tot2)) + \
               list(range(tot1)) + list(range(tot2, tot3)) + \
               list(range(tot3, tot4))
        addend = _permute_dims(addend, _af_invert(perm))
        addend_list.append(addend)
    # 如果只有一个添加项，则返回该项
    if len(addend_list) == 1:
        return addend_list[0]
    # 如果没有添加项，则返回零
    elif len(addend_list) == 0:
        return S.Zero
    # 否则，返回所有添加项的和
    else:
        return _array_add(*addend_list)

# 处理 ArraySymbol 类型的数组表达式的导数
@array_derive.register(ArraySymbol)
def _(expr: ArraySymbol, x: _ArrayExpr):
    if expr == x:
        # 如果数组符号与变量相同，返回特定形状的张量积的排列
        return _permute_dims(
            ArrayTensorProduct.fromiter(Identity(i) for i in expr.shape),
            [2*i for i in range(len(expr.shape))] + [2*i+1 for i in range(len(expr.shape))]
        )
    # 否则返回相同形状的零数组
    return ZeroArray(*(x.shape + expr.shape))

# 处理 MatrixSymbol 类型的数组表达式的导数
@array_derive.register(MatrixSymbol)
def _(expr: MatrixSymbol, x: _ArrayExpr):
    m, n = expr.shape
    if expr == x:
        # 如果矩阵符号与变量相同，返回特定形状的张量积的排列
        return _permute_dims(
            _array_tensor_product(Identity(m), Identity(n)),
            [0, 2, 1, 3]
        )
    # 否则返回相同形状的零数组
    return ZeroArray(*(x.shape + expr.shape))
    # 返回一个全零数组，其形状为 x 数组和 expr 数组形状的组合
    return ZeroArray(*(x.shape + expr.shape))
# 注册 Identity 类型的数组求导函数
@array_derive.register(Identity)
def _(expr: Identity, x: _ArrayExpr):
    # 返回一个形状为 (x.shape + expr.shape) 的零数组
    return ZeroArray(*(x.shape + expr.shape))


# 注册 OneMatrix 类型的数组求导函数
@array_derive.register(OneMatrix)
def _(expr: OneMatrix, x: _ArrayExpr):
    # 返回一个形状为 (x.shape + expr.shape) 的零数组
    return ZeroArray(*(x.shape + expr.shape))


# 注册 Transpose 类型的数组求导函数
@array_derive.register(Transpose)
def _(expr: Transpose, x: Expr):
    # 对于 Transpose(A), 返回 D(A, x) 在维度上进行重新排列后的结果
    # 其中表达式 D(A.T, A) ==> (m,n,i,j) ==> D(A_ji, A_mn) = d_mj d_ni
    fd = array_derive(expr.arg, x)
    return _permute_dims(fd, [0, 1, 3, 2])


# 注册 Inverse 类型的数组求导函数
@array_derive.register(Inverse)
def _(expr: Inverse, x: Expr):
    # 对于 Inverse(expr), 计算其导数
    mat = expr.I
    dexpr = array_derive(mat, x)
    tp = _array_tensor_product(-expr, dexpr, expr)
    mp = _array_contraction(tp, (1, 4), (5, 6))
    pp = _permute_dims(mp, [1, 2, 0, 3])
    return pp


# 注册 ElementwiseApplyFunction 类型的数组求导函数
@array_derive.register(ElementwiseApplyFunction)
def _(expr: ElementwiseApplyFunction, x: Expr):
    # 对于 ElementwiseApplyFunction(expr, x), 计算其导数
    assert get_rank(expr) == 2
    assert get_rank(x) == 2
    fdiff = expr._get_function_fdiff()
    dexpr = array_derive(expr.expr, x)
    tp = _array_tensor_product(
        ElementwiseApplyFunction(fdiff, expr.expr),
        dexpr
    )
    td = _array_diagonal(
        tp, (0, 4), (1, 5)
    )
    return td


# 注册 ArrayElementwiseApplyFunc 类型的数组求导函数
@array_derive.register(ArrayElementwiseApplyFunc)
def _(expr: ArrayElementwiseApplyFunc, x: Expr):
    # 对于 ArrayElementwiseApplyFunc(expr, x), 计算其导数
    fdiff = expr._get_function_fdiff()
    subexpr = expr.expr
    dsubexpr = array_derive(subexpr, x)
    tp = _array_tensor_product(
        dsubexpr,
        ArrayElementwiseApplyFunc(fdiff, subexpr)
    )
    b = get_rank(x)
    c = get_rank(expr)
    diag_indices = [(b + i, b + c + i) for i in range(c)]
    return _array_diagonal(tp, *diag_indices)


# 注册 MatrixExpr 类型的数组求导函数
@array_derive.register(MatrixExpr)
def _(expr: MatrixExpr, x: Expr):
    # 将 MatrixExpr 转换为数组表达式后进行求导
    cg = convert_matrix_to_array(expr)
    return array_derive(cg, x)


# 注册 HadamardProduct 类型的数组求导函数
@array_derive.register(HadamardProduct)
def _(expr: HadamardProduct, x: Expr):
    # 抛出未实现错误，表示暂时不支持 HadamardProduct 的求导
    raise NotImplementedError()


# 注册 ArrayContraction 类型的数组求导函数
@array_derive.register(ArrayContraction)
def _(expr: ArrayContraction, x: Expr):
    # 对 ArrayContraction(expr, x), 计算其导数
    fd = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    contraction_indices = expr.contraction_indices
    new_contraction_indices = [tuple(j + rank_x for j in i) for i in contraction_indices]
    return _array_contraction(fd, *new_contraction_indices)


# 注册 ArrayDiagonal 类型的数组求导函数
@array_derive.register(ArrayDiagonal)
def _(expr: ArrayDiagonal, x: Expr):
    # 对 ArrayDiagonal(expr, x), 计算其导数
    dsubexpr = array_derive(expr.expr, x)
    rank_x = len(get_shape(x))
    diag_indices = [[j + rank_x for j in i] for i in expr.diagonal_indices]
    return _array_diagonal(dsubexpr, *diag_indices)


# 注册 ArrayAdd 类型的数组求导函数
@array_derive.register(ArrayAdd)
def _(expr: ArrayAdd, x: Expr):
    # 对 ArrayAdd(expr.args), 计算其导数
    return _array_add(*[array_derive(arg, x) for arg in expr.args])


# 注册 PermuteDims 类型的数组求导函数
@array_derive.register(PermuteDims)
def _(expr: PermuteDims, x: Expr):
    # 对 PermuteDims(expr, x), 计算其导数
    de = array_derive(expr.expr, x)
    perm = [0, 1] + [i + 2 for i in expr.permutation.array_form]
    return _permute_dims(de, perm)


# 注册 Reshape 类型的数组求导函数
@array_derive.register(Reshape)
def _(expr: Reshape, x: Expr):
    # 对 Reshape(expr, x), 计算其导数
    de = array_derive(expr.expr, x)
    # 返回一个通过 Reshape 函数重塑后的张量，其维度为输入张量 x 的维度加上表达式 expr 的维度
    return Reshape(de, get_shape(x) + expr.shape)
# 定义一个函数，用于计算矩阵表达式关于变量 x 的导数
def matrix_derive(expr, x):
    # 导入将数组转换为矩阵的函数
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    # 将矩阵表达式转换为数组表示
    ce = convert_matrix_to_array(expr)
    # 对数组 ce 关于变量 x 求导
    dce = array_derive(ce, x)
    # 将求导后的数组再转换回矩阵，并进行求值
    return convert_array_to_matrix(dce).doit()
```