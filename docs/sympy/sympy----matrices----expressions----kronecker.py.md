# `D:\src\scipysrc\sympy\sympy\matrices\expressions\kronecker.py`

```
"""Implementation of the Kronecker product"""
# 导入必要的库和模块
from functools import reduce  # 导入 reduce 函数
from math import prod  # 导入 prod 函数

# 导入 sympy 相关模块和类
from sympy.core import Mul, sympify
from sympy.functions import adjoint
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.transpose import transpose
from sympy.matrices.expressions.special import Identity
from sympy.matrices.matrixbase import MatrixBase
from sympy.strategies import (
    canon, condition, distribute, do_one, exhaust, flatten, typed, unpack)
from sympy.strategies.traverse import bottom_up
from sympy.utilities import sift

# 导入 Kronecker product 实现相关的子模块
from .matadd import MatAdd
from .matmul import MatMul
from .matpow import MatPow


def kronecker_product(*matrices):
    """
    The Kronecker product of two or more arguments.

    This computes the explicit Kronecker product for subclasses of
    ``MatrixBase`` i.e. explicit matrices. Otherwise, a symbolic
    ``KroneckerProduct`` object is returned.

    Examples
    ========

    For ``MatrixSymbol`` arguments a ``KroneckerProduct`` object is returned.
    Elements of this matrix can be obtained by indexing, or for MatrixSymbols
    with known dimension the explicit matrix can be obtained with
    ``.as_explicit()``

    >>> from sympy import kronecker_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = MatrixSymbol('B', 2, 2)
    >>> kronecker_product(A)
    A
    >>> kronecker_product(A, B)
    KroneckerProduct(A, B)
    >>> kronecker_product(A, B)[0, 1]
    A[0, 0]*B[0, 1]
    >>> kronecker_product(A, B).as_explicit()
    Matrix([
        [A[0, 0]*B[0, 0], A[0, 0]*B[0, 1], A[0, 1]*B[0, 0], A[0, 1]*B[0, 1]],
        [A[0, 0]*B[1, 0], A[0, 0]*B[1, 1], A[0, 1]*B[1, 0], A[0, 1]*B[1, 1]],
        [A[1, 0]*B[0, 0], A[1, 0]*B[0, 1], A[1, 1]*B[0, 0], A[1, 1]*B[0, 1]],
        [A[1, 0]*B[1, 0], A[1, 0]*B[1, 1], A[1, 1]*B[1, 0], A[1, 1]*B[1, 1]]])

    For explicit matrices the Kronecker product is returned as a Matrix

    >>> from sympy import Matrix, kronecker_product
    >>> sigma_x = Matrix([
    ... [0, 1],
    ... [1, 0]])
    ...
    >>> Isigma_y = Matrix([
    ... [0, 1],
    ... [-1, 0]])
    ...
    >>> kronecker_product(sigma_x, Isigma_y)
    Matrix([
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0],
    [ 0, 1,  0, 0],
    [-1, 0,  0, 0]])

    See Also
    ========
        KroneckerProduct

    """
    # 检查是否有输入的矩阵
    if not matrices:
        raise TypeError("Empty Kronecker product is undefined")
    # 如果只有一个矩阵，直接返回该矩阵
    if len(matrices) == 1:
        return matrices[0]
    else:
        # 使用 KroneckerProduct 类处理多个矩阵的 Kronecker 乘积
        return KroneckerProduct(*matrices).doit()


class KroneckerProduct(MatrixExpr):
    """
    The Kronecker product of two or more arguments.

    The Kronecker product is a non-commutative product of matrices.
    Given two matrices of dimension (m, n) and (s, t) it produces a matrix
    of dimension (m s, n t).

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    """

    # 构造函数，接受多个矩阵参数
    def __new__(cls, *args):
        # 检查是否有输入的矩阵
        if not args:
            raise TypeError("Kronecker product requires at least one matrix")
        # 初始化 KroneckerProduct 实例，用于存储参数
        obj = MatrixExpr.__new__(cls, *args)
        return obj

    # doit 方法用于实际计算 Kronecker 乘积
    def doit(self):
        # 初始化首个矩阵
        ret = self.args[0]
        # 遍历剩余的矩阵，依次进行 Kronecker 乘积计算
        for m in self.args[1:]:
            ret = ret.kronecker_product(m)
        return ret

    # 返回表示 Kronecker 乘积的字符串表示形式
    def __str__(self):
        return 'KroneckerProduct(%s)' % ', '.join(str(a) for a in self.args)
    """
    A class representing the Kronecker product of matrices in symbolic algebra.

    >>> from sympy import KroneckerProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(KroneckerProduct(A, B), KroneckerProduct)
    True
    """

    # 设置标志，表示当前类是 KroneckerProduct 类
    is_KroneckerProduct = True

    # 构造函数，创建新的 KroneckerProduct 实例
    def __new__(cls, *args, check=True):
        # 将所有输入参数转换为 SymPy 符号对象
        args = list(map(sympify, args))

        # 如果所有输入矩阵都是单位矩阵，则返回一个单位矩阵
        if all(a.is_Identity for a in args):
            ret = Identity(prod(a.rows for a in args))
            # 如果所有输入矩阵都是 MatrixBase 类型，则返回其显式表示
            if all(isinstance(a, MatrixBase) for a in args):
                return ret.as_explicit()
            else:
                return ret

        # 如果需要进行检查，则验证输入参数
        if check:
            validate(*args)

        # 调用父类的构造函数创建新的实例
        return super().__new__(cls, *args)

    # 获取矩阵乘积的形状
    @property
    def shape(self):
        # 获取第一个矩阵的形状
        rows, cols = self.args[0].shape
        # 遍历剩余的矩阵，计算乘积的总行数和总列数
        for mat in self.args[1:]:
            rows *= mat.rows
            cols *= mat.cols
        return (rows, cols)

    # 计算 Kronecker 乘积的第 (i, j) 元素
    def _entry(self, i, j, **kwargs):
        result = 1
        # 反向遍历所有输入矩阵
        for mat in reversed(self.args):
            # 根据当前矩阵的行数和列数计算当前位置在哪个子矩阵中
            i, m = divmod(i, mat.rows)
            j, n = divmod(j, mat.cols)
            # 计算乘积的累积值
            result *= mat[m, n]
        return result

    # 计算 Kronecker 乘积的伴随
    def _eval_adjoint(self):
        return KroneckerProduct(*list(map(adjoint, self.args))).doit()

    # 计算 Kronecker 乘积的共轭
    def _eval_conjugate(self):
        return KroneckerProduct(*[a.conjugate() for a in self.args]).doit()

    # 计算 Kronecker 乘积的转置
    def _eval_transpose(self):
        return KroneckerProduct(*list(map(transpose, self.args))).doit()

    # 计算 Kronecker 乘积的迹
    def _eval_trace(self):
        from .trace import trace
        return Mul(*[trace(a) for a in self.args])

    # 计算 Kronecker 乘积的行列式
    def _eval_determinant(self):
        from .determinant import det, Determinant
        # 如果所有输入矩阵都是方阵，则计算它们的乘积的行列式
        if not all(a.is_square for a in self.args):
            return Determinant(self)

        # 计算每个输入矩阵的行列式的乘积
        m = self.rows
        return Mul(*[det(a)**(m/a.rows) for a in self.args])

    # 计算 Kronecker 乘积的逆矩阵
    def _eval_inverse(self):
        try:
            # 尝试计算每个输入矩阵的逆矩阵，并返回 Kronecker 乘积
            return KroneckerProduct(*[a.inverse() for a in self.args])
        except ShapeError:
            from sympy.matrices.expressions.inverse import Inverse
            return Inverse(self)
    def structurally_equal(self, other):
        '''判断两个矩阵是否具有相同的克罗内克积结构

        Examples
        ========

        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, m)
        >>> B = MatrixSymbol('B', n, n)
        >>> C = MatrixSymbol('C', m, m)
        >>> D = MatrixSymbol('D', n, n)
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(C, D))
        True
        >>> KroneckerProduct(A, B).structurally_equal(KroneckerProduct(D, C))
        False
        >>> KroneckerProduct(A, B).structurally_equal(C)
        False
        '''
        # 受 BlockMatrix 启发
        return (isinstance(other, KroneckerProduct)
                and self.shape == other.shape
                and len(self.args) == len(other.args)
                and all(a.shape == b.shape for (a, b) in zip(self.args, other.args)))

    def has_matching_shape(self, other):
        '''判断两个矩阵是否具有匹配的结构，以便将矩阵乘法内部化为克罗内克积

        Examples
        ========
        >>> from sympy import KroneckerProduct, MatrixSymbol, symbols
        >>> m, n = symbols(r'm, n', integer=True)
        >>> A = MatrixSymbol('A', m, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(B, A))
        True
        >>> KroneckerProduct(A, B).has_matching_shape(KroneckerProduct(A, B))
        False
        >>> KroneckerProduct(A, B).has_matching_shape(A)
        False
        '''
        return (isinstance(other, KroneckerProduct)
                and self.cols == other.rows
                and len(self.args) == len(other.args)
                and all(a.cols == b.rows for (a, b) in zip(self.args, other.args)))

    def _eval_expand_kroneckerproduct(self, **hints):
        '''扩展克罗内克积，考虑提示

        这个函数扩展克罗内克积并进行规范化处理。

        Parameters
        ==========
        **hints : dict
            可能包含用于扩展的提示信息

        Returns
        =======
        扩展后并经过规范化处理的克罗内克积
        '''
        return flatten(canon(typed({KroneckerProduct: distribute(KroneckerProduct, MatAdd)}))(self))

    def _kronecker_add(self, other):
        '''执行克罗内克积的加法

        如果两个克罗内克积结构相同，则执行元素级别的加法；否则，返回普通加法。

        Parameters
        ==========
        other : KroneckerProduct or MatrixExpr
            另一个克罗内克积或矩阵表达式

        Returns
        =======
        KroneckerProduct
            执行加法后的克罗内克积
        '''
        if self.structurally_equal(other):
            return self.__class__(*[a + b for (a, b) in zip(self.args, other.args)])
        else:
            return self + other

    def _kronecker_mul(self, other):
        '''执行克罗内克积的乘法

        如果两个克罗内克积具有匹配的结构，则执行元素级别的乘法；否则，返回普通乘法。

        Parameters
        ==========
        other : KroneckerProduct or MatrixExpr
            另一个克罗内克积或矩阵表达式

        Returns
        =======
        KroneckerProduct
            执行乘法后的克罗内克积
        '''
        if self.has_matching_shape(other):
            return self.__class__(*[a*b for (a, b) in zip(self.args, other.args)])
        else:
            return self * other

    def doit(self, **hints):
        '''规范化克罗内克积对象

        根据提示规范化克罗内克积对象，可选择进行深度规范化。

        Parameters
        ==========
        **hints : dict
            可能包含规范化过程中的提示信息

        Returns
        =======
        KroneckerProduct
            规范化后的克罗内克积对象
        '''
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(KroneckerProduct(*args))
def validate(*args):
    # 检查所有参数是否都是矩阵，如果有任何一个不是则抛出类型错误
    if not all(arg.is_Matrix for arg in args):
        raise TypeError("Mix of Matrix and Scalar symbols")


# rules

def extract_commutative(kron):
    # 初始化可交换部分和不可交换部分的列表
    c_part = []
    nc_part = []
    # 遍历 KroneckerProduct 对象的每个参数
    for arg in kron.args:
        # 分离每个参数的可交换部分和不可交换部分
        c, nc = arg.args_cnc()
        # 将可交换部分添加到 c_part 列表中
        c_part.extend(c)
        # 将不可交换部分作为 Mul 对象添加到 nc_part 列表中
        nc_part.append(Mul._from_args(nc))

    # 将 c_part 列表中的元素相乘，得到一个 Mul 对象
    c_part = Mul(*c_part)
    # 如果 c_part 不等于 1，则返回 c_part 乘以剩余的不可交换部分的 KroneckerProduct 对象
    if c_part != 1:
        return c_part * KroneckerProduct(*nc_part)
    # 如果 c_part 等于 1，则直接返回原始的 KroneckerProduct 对象
    return kron


def matrix_kronecker_product(*matrices):
    """Compute the Kronecker product of a sequence of SymPy Matrices.

    This is the standard Kronecker product of matrices [1].

    Parameters
    ==========

    matrices : tuple of MatrixBase instances
        The matrices to take the Kronecker product of.

    Returns
    =======

    matrix : MatrixBase
        The Kronecker product matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.expressions.kronecker import (
    ... matrix_kronecker_product)

    >>> m1 = Matrix([[1,2],[3,4]])
    >>> m2 = Matrix([[1,0],[0,1]])
    >>> matrix_kronecker_product(m1, m2)
    Matrix([
    [1, 0, 2, 0],
    [0, 1, 0, 2],
    [3, 0, 4, 0],
    [0, 3, 0, 4]])
    >>> matrix_kronecker_product(m2, m1)
    Matrix([
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_product
    """
    # 确保 matrices 参数中的所有元素都是 MatrixBase 的实例
    if not all(isinstance(m, MatrixBase) for m in matrices):
        raise TypeError(
            'Sequence of Matrices expected, got: %s' % repr(matrices)
        )

    # 将最后一个矩阵作为初始的 matrix_expansion
    matrix_expansion = matrices[-1]
    # 从右向左计算 Kronecker product
    for mat in reversed(matrices[:-1]):
        rows = mat.rows
        cols = mat.cols
        # 遍历每一行，将 Kronecker product 添加到运行中的 matrix_expansion
        for i in range(rows):
            start = matrix_expansion * mat[i * cols]
            # 遍历每一列，将每个项连接起来
            for j in range(cols - 1):
                start = start.row_join(
                    matrix_expansion * mat[i * cols + j + 1]
                )
            # 如果这是第一个元素，则将其作为新行的开始
            if i == 0:
                next = start
            else:
                next = next.col_join(start)
        matrix_expansion = next

    # 根据 matrices 中具有最高 _class_priority 的矩阵确定 MatrixClass
    MatrixClass = max(matrices, key=lambda M: M._class_priority).__class__
    # 如果 matrix_expansion 是 MatrixClass 的实例，则返回 matrix_expansion
    if isinstance(matrix_expansion, MatrixClass):
        return matrix_expansion
    else:
        # 否则，将 matrix_expansion 转换为 MatrixClass 的实例并返回
        return MatrixClass(matrix_expansion)


def explicit_kronecker_product(kron):
    # 确保 kron 中的所有参数都是 MatrixBase 的实例
    if not all(isinstance(m, MatrixBase) for m in kron.args):
        return kron

    # 调用 matrix_kronecker_product 计算显式 Kronecker product
    return matrix_kronecker_product(*kron.args)
rules = (unpack,
         explicit_kronecker_product,
         flatten,
         extract_commutative)

canonicalize = exhaust(condition(lambda x: isinstance(x, KroneckerProduct),
                                 do_one(*rules)))

# 定义规则列表，用于规范化操作
# unpack: 解包操作
# explicit_kronecker_product: 显式克罗内克积操作
# flatten: 扁平化操作
# extract_commutative: 提取可交换项操作

# 使用规则列表中的函数，对表达式进行规范化操作，确保表达式符合一定的规范化形式
canonicalize = exhaust(condition(lambda x: isinstance(x, KroneckerProduct),
                                 do_one(*rules)))


def _kronecker_dims_key(expr):
    if isinstance(expr, KroneckerProduct):
        return tuple(a.shape for a in expr.args)
    else:
        return (0,)

# 根据表达式类型，生成用于排序的关键字，用于后续操作中按照维度排序


def kronecker_mat_add(expr):
    args = sift(expr.args, _kronecker_dims_key)
    nonkrons = args.pop((0,), None)
    if not args:
        return expr

    krons = [reduce(lambda x, y: x._kronecker_add(y), group)
             for group in args.values()]

    if not nonkrons:
        return MatAdd(*krons)
    else:
        return MatAdd(*krons) + nonkrons

# 将输入的矩阵表达式按照克罗内克积进行加法操作，返回合并后的结果


def kronecker_mat_mul(expr):
    # modified from block matrix code
    factor, matrices = expr.as_coeff_matrices()

    i = 0
    while i < len(matrices) - 1:
        A, B = matrices[i:i+2]
        if isinstance(A, KroneckerProduct) and isinstance(B, KroneckerProduct):
            matrices[i] = A._kronecker_mul(B)
            matrices.pop(i+1)
        else:
            i += 1

    return factor*MatMul(*matrices)

# 对输入的矩阵表达式进行乘法操作，如果操作数是克罗内克积，则执行对应的乘法操作，返回结果


def kronecker_mat_pow(expr):
    if isinstance(expr.base, KroneckerProduct) and all(a.is_square for a in expr.base.args):
        return KroneckerProduct(*[MatPow(a, expr.exp) for a in expr.base.args])
    else:
        return expr

# 对输入的矩阵表达式进行幂运算操作，如果基数是克罗内克积且所有矩阵都是方阵，则执行幂运算，返回结果


def combine_kronecker(expr):
    """Combine KronekeckerProduct with expression.

    If possible write operations on KroneckerProducts of compatible shapes
    as a single KroneckerProduct.

    Examples
    ========

    >>> from sympy.matrices.expressions import combine_kronecker
    >>> from sympy import MatrixSymbol, KroneckerProduct, symbols
    >>> m, n = symbols(r'm, n', integer=True)
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', n, m)
    >>> combine_kronecker(KroneckerProduct(A, B)*KroneckerProduct(B, A))
    KroneckerProduct(A*B, B*A)
    >>> combine_kronecker(KroneckerProduct(A, B)+KroneckerProduct(B.T, A.T))
    KroneckerProduct(A + B.T, B + A.T)
    >>> C = MatrixSymbol('C', n, n)
    >>> D = MatrixSymbol('D', m, m)
    >>> combine_kronecker(KroneckerProduct(C, D)**m)
    KroneckerProduct(C**m, D**m)
    """
    def haskron(expr):
        return isinstance(expr, MatrixExpr) and expr.has(KroneckerProduct)

    rule = exhaust(
        bottom_up(exhaust(condition(haskron, typed(
            {MatAdd: kronecker_mat_add,
             MatMul: kronecker_mat_mul,
             MatPow: kronecker_mat_pow})))))

    # 应用一系列规则将表达式中的克罗内克积操作组合成单个克罗内克积操作
    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result

# 将表达式中的克罗内克积操作组合成单个克罗内克积操作，确保表达式在符合条件时返回正确的结果
```