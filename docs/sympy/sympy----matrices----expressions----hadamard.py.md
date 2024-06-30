# `D:\src\scipysrc\sympy\sympy\matrices\expressions\hadamard.py`

```
# 导入 Counter 类，用于计数
from collections import Counter

# 从 sympy.core 模块导入 Mul 和 sympify 函数
from sympy.core import Mul, sympify
# 从 sympy.core.add 模块导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.expr 模块导入 ExprBuilder 类
from sympy.core.expr import ExprBuilder
# 从 sympy.core.sorting 模块导入 default_sort_key 函数
from sympy.core.sorting import default_sort_key
# 从 sympy.functions.elementary.exponential 模块导入 log 函数
from sympy.functions.elementary.exponential import log
# 从 sympy.matrices.expressions.matexpr 模块导入 MatrixExpr 类
from sympy.matrices.expressions.matexpr import MatrixExpr
# 从 sympy.matrices.expressions._shape 模块导入 validate_matadd_integer 函数并重命名为 validate
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
# 从 sympy.matrices.expressions.special 模块导入 ZeroMatrix 和 OneMatrix 类
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
# 从 sympy.strategies 模块导入 unpack、flatten、condition、exhaust、rm_id、sort 函数
from sympy.strategies import (
    unpack, flatten, condition, exhaust, rm_id, sort
)
# 从 sympy.utilities.exceptions 模块导入 sympy_deprecation_warning 函数
from sympy.utilities.exceptions import sympy_deprecation_warning


def hadamard_product(*matrices):
    """
    Return the elementwise (aka Hadamard) product of matrices.

    Examples
    ========

    >>> from sympy import hadamard_product, MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 3)
    >>> B = MatrixSymbol('B', 2, 3)
    >>> hadamard_product(A)
    A
    >>> hadamard_product(A, B)
    HadamardProduct(A, B)
    >>> hadamard_product(A, B)[0, 1]
    A[0, 1]*B[0, 1]
    """
    # 如果没有传入任何矩阵，抛出 TypeError 异常
    if not matrices:
        raise TypeError("Empty Hadamard product is undefined")
    # 如果只有一个矩阵，直接返回该矩阵
    if len(matrices) == 1:
        return matrices[0]
    # 使用 HadamardProduct 类来进行矩阵逐元素乘法并返回结果
    return HadamardProduct(*matrices).doit()


class HadamardProduct(MatrixExpr):
    """
    Elementwise product of matrix expressions

    Examples
    ========

    Hadamard product for matrix symbols:

    >>> from sympy import hadamard_product, HadamardProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(hadamard_product(A, B), HadamardProduct)
    True

    Notes
    =====

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``hadamard_product()`` or ``HadamardProduct.doit``
    """
    # 表明这是一个 HadamardProduct 类的对象
    is_HadamardProduct = True

    def __new__(cls, *args, evaluate=False, check=None):
        # 对传入的参数进行 sympify 处理，确保是符号表达式
        args = list(map(sympify, args))
        # 如果没有参数，则抛出 ValueError 异常
        if len(args) == 0:
            raise ValueError("HadamardProduct needs at least one argument")

        # 检查参数类型，确保所有参数都是 MatrixExpr 类型
        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        # 检查参数是否需要进行验证，默认情况下进行验证
        if check is not None:
            sympy_deprecation_warning(
                "Passing check to HadamardProduct is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        # 如果 check 不是 False，则调用 validate 函数验证参数
        if check is not False:
            validate(*args)

        # 调用父类的 __new__ 方法创建 HadamardProduct 对象
        obj = super().__new__(cls, *args)
        # 如果 evaluate 参数为 True，则立即计算 HadamardProduct 的值并返回
        if evaluate:
            obj = obj.doit(deep=False)
        return obj

    @property
    def shape(self):
        # 返回第一个参数的形状作为 HadamardProduct 对象的形状
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        # 返回 HadamardProduct 对象在位置 (i, j) 处的元素表达式
        return Mul(*[arg._entry(i, j, **kwargs) for arg in self.args])
    # 定义一个方法用于计算矩阵的转置，引入 transpose 函数来处理每个参数并返回 Hadamard 乘积的结果
    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardProduct(*list(map(transpose, self.args)))

    # 执行表达式的操作，并检查是否包含明确的矩阵，将它们重新组织为 Hadamard 乘积的表达式
    def doit(self, **hints):
        # 对每个参数执行 doit 操作，然后重新组合成表达式
        expr = self.func(*(i.doit(**hints) for i in self.args))
        
        # 检查是否有明确的矩阵存在于表达式中
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.immutable import ImmutableMatrix
        explicit = [i for i in expr.args if isinstance(i, MatrixBase)]
        
        # 如果有明确的矩阵，则重新组织这些矩阵和其余项，形成新的 Hadamard 乘积表达式
        if explicit:
            remainder = [i for i in expr.args if i not in explicit]
            expl_mat = ImmutableMatrix([
                Mul.fromiter(i) for i in zip(*explicit)
            ]).reshape(*self.shape)
            expr = HadamardProduct(*([expl_mat] + remainder))
        
        # 规范化并返回表达式
        return canonicalize(expr)

    # 计算对象关于变量 x 的导数，并返回结果
    def _eval_derivative(self, x):
        terms = []
        args = list(self.args)
        # 遍历每个参数，替换其中的一个参数为其关于 x 的导数，并构建 Hadamard 乘积的表达式
        for i in range(len(args)):
            factors = args[:i] + [args[i].diff(x)] + args[i+1:]
            terms.append(hadamard_product(*factors))
        # 从构建的项中创建加法表达式并返回
        return Add.fromiter(terms)

    # 计算对象关于变量 x 的导数，并返回其线性表示的列表
    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.matrices.expressions.matexpr import _make_matrix

        # 找到包含变量 x 的参数的索引
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        
        # 对每个包含 x 的参数进行处理
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            # 计算当前参数关于 x 的导数的线性表示
            d = self.args[ind]._eval_derivative_matrix_lines(x)
            hadam = hadamard_product(*(right_args + left_args))
            diagonal = [(0, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.shape[j] != 1]
            
            # 对于每个导数表示，构建对角线和线性表示的新表达式
            for i in d:
                l1 = i._lines[i._first_line_index]
                l2 = i._lines[i._second_line_index]
                subexpr = ExprBuilder(
                    ArrayDiagonal,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                ExprBuilder(_make_matrix, [l1]),
                                hadam,
                                ExprBuilder(_make_matrix, [l2]),
                            ]
                        ),
                    *diagonal],

                )
                i._first_pointer_parent = subexpr.args[0].args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args[2].args
                i._second_pointer_index = 0
                i._lines = [subexpr]
                lines.append(i)

        # 返回所有处理过的线性表示
        return lines
# TODO Implement algorithm for rewriting Hadamard product as diagonal matrix
# if matmul identy matrix is multiplied.
def canonicalize(x):
    """Canonicalize the Hadamard product ``x`` with mathematical properties.

    Examples
    ========

    >>> from sympy import MatrixSymbol, HadamardProduct
    >>> from sympy import OneMatrix, ZeroMatrix
    >>> from sympy.matrices.expressions.hadamard import canonicalize
    >>> from sympy import init_printing
    >>> init_printing(use_unicode=False)

    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = MatrixSymbol('B', 2, 2)
    >>> C = MatrixSymbol('C', 2, 2)

    Hadamard product associativity:

    >>> X = HadamardProduct(A, HadamardProduct(B, C))
    >>> X
    A.*(B.*C)
    >>> canonicalize(X)
    A.*B.*C

    Hadamard product commutativity:

    >>> X = HadamardProduct(A, B)
    >>> Y = HadamardProduct(B, A)
    >>> X
    A.*B
    >>> Y
    B.*A
    >>> canonicalize(X)
    A.*B
    >>> canonicalize(Y)
    A.*B

    Hadamard product identity:

    >>> X = HadamardProduct(A, OneMatrix(2, 2))
    >>> X
    A.*1
    >>> canonicalize(X)
    A

    Absorbing element of Hadamard product:

    >>> X = HadamardProduct(A, ZeroMatrix(2, 2))
    >>> X
    A.*0
    >>> canonicalize(X)
    0

    Rewriting to Hadamard Power

    >>> X = HadamardProduct(A, A, A)
    >>> X
    A.*A.*A
    >>> canonicalize(X)
    A

    Notes
    =====

    As the Hadamard product is associative, nested products can be flattened.

    The Hadamard product is commutative so that factors can be sorted for
    canonical form.

    A matrix of only ones is an identity for Hadamard product,
    so every matrices of only ones can be removed.

    Any zero matrix will make the whole product a zero matrix.

    Duplicate elements can be collected and rewritten as HadamardPower

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    """
    # Associativity: Flatten nested Hadamard products
    rule = condition(
            lambda x: isinstance(x, HadamardProduct),
            flatten
        )
    fun = exhaust(rule)
    x = fun(x)

    # Identity: Remove identity matrices from Hadamard product
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            rm_id(lambda x: isinstance(x, OneMatrix))
        )
    x = fun(x)

    # Absorbing by Zero Matrix: If any factor is a zero matrix, return a zero matrix
    def absorb(x):
        if any(isinstance(c, ZeroMatrix) for c in x.args):
            return ZeroMatrix(*x.shape)
        else:
            return x
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            absorb
        )
    x = fun(x)

    # Rewriting with HadamardPower: Rewrite duplicate elements as powers
    if isinstance(x, HadamardProduct):
        tally = Counter(x.args)

        new_arg = []
        for base, exp in tally.items():
            if exp == 1:
                new_arg.append(base)
            else:
                new_arg.append(HadamardPower(base, exp))

        x = HadamardProduct(*new_arg)

    # Commutativity: Sort factors in canonical order
    # (The implementation of commutativity is incomplete in the provided code.)
    # 定义一个函数 fun，使用 condition 函数进行筛选和排序
    fun = condition(
            lambda x: isinstance(x, HadamardProduct),
            sort(default_sort_key)
        )
    # 将 x 作为参数传递给函数 fun，对 x 进行处理
    x = fun(x)

    # 对 x 进行解包操作，可能是将 x 拆分成更小的组件或者重新组织数据结构
    x = unpack(x)
    # 返回处理后的结果 x
    return x
# 定义一个函数，用于计算 Hadamard 幂
def hadamard_power(base, exp):
    # 将输入的 base 转换成符号表达式
    base = sympify(base)
    # 将输入的 exp 转换成符号表达式
    exp = sympify(exp)
    
    # 如果指数 exp 等于 1，则直接返回 base
    if exp == 1:
        return base
    
    # 如果 base 不是矩阵，则计算 base 的 exp 次方
    if not base.is_Matrix:
        return base ** exp
    
    # 如果 exp 是矩阵，则抛出异常，因为不能将表达式提升为矩阵的幂
    if exp.is_Matrix:
        raise ValueError("cannot raise expression to a matrix")
    
    # 如果以上条件都不满足，则调用 HadamardPower 类来计算 base 的 exp 次幂
    return HadamardPower(base, exp)


class HadamardPower(MatrixExpr):
    r"""
    Elementwise power of matrix expressions

    Parameters
    ==========

    base : scalar or matrix

    exp : scalar or matrix

    Notes
    =====

    There are four definitions for the hadamard power which can be used.
    Let's consider `A, B` as `(m, n)` matrices, and `a, b` as scalars.

    Matrix raised to a scalar exponent:

    .. math::
        A^{\circ b} = \begin{bmatrix}
        A_{0, 0}^b   & A_{0, 1}^b   & \cdots & A_{0, n-1}^b   \\
        A_{1, 0}^b   & A_{1, 1}^b   & \cdots & A_{1, n-1}^b   \\
        \vdots       & \vdots       & \ddots & \vdots         \\
        A_{m-1, 0}^b & A_{m-1, 1}^b & \cdots & A_{m-1, n-1}^b
        \end{bmatrix}

    Scalar raised to a matrix exponent:

    .. math::
        a^{\circ B} = \begin{bmatrix}
        a^{B_{0, 0}}   & a^{B_{0, 1}}   & \cdots & a^{B_{0, n-1}}   \\
        a^{B_{1, 0}}   & a^{B_{1, 1}}   & \cdots & a^{B_{1, n-1}}   \\
        \vdots         & \vdots         & \ddots & \vdots           \\
        a^{B_{m-1, 0}} & a^{B_{m-1, 1}} & \cdots & a^{B_{m-1, n-1}}
        \end{bmatrix}

    Matrix raised to a matrix exponent:

    .. math::
        A^{\circ B} = \begin{bmatrix}
        A_{0, 0}^{B_{0, 0}}     & A_{0, 1}^{B_{0, 1}}     &
        \cdots & A_{0, n-1}^{B_{0, n-1}}     \\
        A_{1, 0}^{B_{1, 0}}     & A_{1, 1}^{B_{1, 1}}     &
        \cdots & A_{1, n-1}^{B_{1, n-1}}     \\
        \vdots                  & \vdots                  &
        \ddots & \vdots                      \\
        A_{m-1, 0}^{B_{m-1, 0}} & A_{m-1, 1}^{B_{m-1, 1}} &
        \cdots & A_{m-1, n-1}^{B_{m-1, n-1}}
        \end{bmatrix}

    Scalar raised to a scalar exponent:

    .. math::
        a^{\circ b} = a^b
    """

    def __new__(cls, base, exp):
        # 将 base 和 exp 分别转换成符号表达式
        base = sympify(base)
        exp = sympify(exp)

        # 如果 base 和 exp 都是标量，则直接计算它们的幂运算
        if base.is_scalar and exp.is_scalar:
            return base ** exp

        # 如果 base 和 exp 都是矩阵表达式，则验证它们的合法性
        if isinstance(base, MatrixExpr) and isinstance(exp, MatrixExpr):
            validate(base, exp)

        # 调用父类的构造方法来创建 HadamardPower 的实例对象
        obj = super().__new__(cls, base, exp)
        return obj

    @property
    def base(self):
        # 返回 HadamardPower 对象的 base 属性
        return self._args[0]

    @property
    def exp(self):
        # 返回 HadamardPower 对象的 exp 属性
        return self._args[1]

    @property
    def shape(self):
        # 如果 base 是矩阵，则返回 base 的形状；否则返回 exp 的形状
        if self.base.is_Matrix:
            return self.base.shape
        return self.exp.shape
    # 定义一个方法 `_entry`，接受两个索引参数 i 和 j，以及其他关键字参数 kwargs
    def _entry(self, i, j, **kwargs):
        # 将 self.base 赋值给变量 base
        base = self.base
        # 将 self.exp 赋值给变量 exp

        # 如果 base 是一个矩阵
        if base.is_Matrix:
            # 调用 base 对象的 _entry 方法，获取索引为 (i, j) 的元素，传递其他关键字参数
            a = base._entry(i, j, **kwargs)
        # 如果 base 是一个标量（非矩阵）
        elif base.is_scalar:
            # 直接将 base 赋值给 a
            a = base
        else:
            # 抛出 ValueError 异常，指示 base 必须是标量或矩阵
            raise ValueError(
                'The base {} must be a scalar or a matrix.'.format(base))

        # 如果 exp 是一个矩阵
        if exp.is_Matrix:
            # 调用 exp 对象的 _entry 方法，获取索引为 (i, j) 的元素，传递其他关键字参数
            b = exp._entry(i, j, **kwargs)
        # 如果 exp 是一个标量（非矩阵）
        elif exp.is_scalar:
            # 直接将 exp 赋值给 b
            b = exp
        else:
            # 抛出 ValueError 异常，指示 exp 必须是标量或矩阵
            raise ValueError(
                'The exponent {} must be a scalar or a matrix.'.format(exp))

        # 返回 a 的 b 次幂
        return a ** b

    # 定义一个方法 `_eval_transpose`
    def _eval_transpose(self):
        # 导入 transpose 函数
        from sympy.matrices.expressions.transpose import transpose
        # 返回 self.base 的转置矩阵与 self.exp 的 Hadamard 幂运算结果
        return HadamardPower(transpose(self.base), self.exp)

    # 定义一个方法 `_eval_derivative`，接受一个变量 x 作为参数
    def _eval_derivative(self, x):
        # 计算指数的导数 dexp
        dexp = self.exp.diff(x)
        # 对 base 元素应用对数函数，得到 logbase
        logbase = self.base.applyfunc(log)
        # 计算 logbase 对 x 的导数 dlbase
        dlbase = logbase.diff(x)
        # 返回哈达玛积结果，其中包含 dexp*logbase 和 self.exp*dlbase
        return hadamard_product(
            dexp*logbase + self.exp*dlbase,
            self
        )

    # 定义一个方法 `_eval_derivative_matrix_lines`，接受一个变量 x 作为参数
    def _eval_derivative_matrix_lines(self, x):
        # 导入必要的类和函数
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.matrices.expressions.matexpr import _make_matrix

        # 调用 base 对象的 _eval_derivative_matrix_lines 方法，得到结果 lr
        lr = self.base._eval_derivative_matrix_lines(x)
        # 对 lr 中的每个元素 i 进行迭代
        for i in lr:
            # 定义对角线的位置数组
            diagonal = [(1, 2), (3, 4)]
            # 根据 self.base 的形状，过滤掉对角线上长度为 1 的维度
            diagonal = [e for j, e in enumerate(diagonal) if self.base.shape[j] != 1]
            # 获取 i 的第一行和第二行
            l1 = i._lines[i._first_line_index]
            l2 = i._lines[i._second_line_index]
            # 构建子表达式 subexpr
            subexpr = ExprBuilder(
                # 使用 ArrayTensorProduct 构建张量积表达式
                ArrayTensorProduct,
                [
                    # 第一个参数是 _make_matrix 函数应用于 l1 的结果
                    ExprBuilder(_make_matrix, [l1]),
                    # 第二个参数是 self.exp 乘以 self.base 的 self.exp-1 次幂的哈达玛幂运算结果
                    self.exp*hadamard_power(self.base, self.exp-1),
                    # 第三个参数是 _make_matrix 函数应用于 l2 的结果
                    ExprBuilder(_make_matrix, [l2]),
                ]
            )
            # 将 diagonal 添加到 subexpr 的参数中
            subexpr = ExprBuilder(
                ArrayDiagonal,
                [subexpr, *diagonal],
                validator=ArrayDiagonal._validate
            )
            # 调整 i 对象的指针和行索引等属性
            i._first_pointer_parent = subexpr.args[0].args[0].args
            i._first_pointer_index = 0
            i._first_line_index = 0
            i._second_pointer_parent = subexpr.args[0].args[2].args
            i._second_pointer_index = 0
            i._second_line_index = 0
            i._lines = [subexpr]
        # 返回处理后的 lr 结果
        return lr
```