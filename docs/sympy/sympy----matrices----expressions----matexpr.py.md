# `D:\src\scipysrc\sympy\sympy\matrices\expressions\matexpr.py`

```
# 导入未来版本的注解语法支持
from __future__ import annotations
# 导入函数装饰器
from functools import wraps

# 导入 sympy 核心模块的特定类和函数
from sympy.core import S, Integer, Basic, Mul, Add
# 导入符号表达式的断言模块
from sympy.core.assumptions import check_assumptions
# 导入调用优先级相关的装饰器
from sympy.core.decorators import call_highest_priority
# 导入表达式和表达式构建器
from sympy.core.expr import Expr, ExprBuilder
# 导入模糊布尔逻辑模块
from sympy.core.logic import FuzzyBool
# 导入符号和虚拟符号类
from sympy.core.symbol import Str, Dummy, symbols, Symbol
# 导入 sympify 异常和函数
from sympy.core.sympify import SympifyError, _sympify
# 导入 gmpy 的整数支持
from sympy.external.gmpy import SYMPY_INTS
# 导入共轭和伴随函数
from sympy.functions import conjugate, adjoint
# 导入克罗内克 delta 函数
from sympy.functions.special.tensor_functions import KroneckerDelta
# 导入矩阵异常和矩阵类型
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.kind import MatrixKind
# 导入矩阵基类
from sympy.matrices.matrixbase import MatrixBase
# 导入多分派函数的装饰器
from sympy.multipledispatch import dispatch
# 导入实用工具中的填充缩进函数
from sympy.utilities.misc import filldedent


def _sympifyit(arg, retval=None):
    # 这个版本的 _sympifyit 函数用于 sympify MutableMatrix 对象
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                # 尝试对第二个参数进行 sympify 处理
                b = _sympify(b)
                # 调用原始函数处理参数
                return func(a, b)
            except SympifyError:
                # 处理 sympify 异常，返回指定的默认值
                return retval

        return __sympifyit_wrapper

    return deco


class MatrixExpr(Expr):
    """矩阵表达式的超类

    MatrixExpr 表示抽象矩阵，线性变换在特定基础下的表示。

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    # 禁止增加额外的实例属性
    __slots__: tuple[str, ...] = ()

    # 应该不被 sympy.utilities.iterables.iterable 函数视为可迭代的标记
    _iterable = False

    # 操作的优先级
    _op_priority = 11.0

    # 类型标记
    is_Matrix: bool = True
    is_MatrixExpr: bool = True
    is_Identity: FuzzyBool = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False

    # 不是可交换的
    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False

    # 矩阵类型
    kind: MatrixKind = MatrixKind()

    def __new__(cls, *args, **kwargs):
        # 将所有参数进行 sympify 处理
        args = map(_sympify, args)
        # 调用基类的构造函数创建实例
        return Basic.__new__(cls, *args, **kwargs)

    # 以下部分改编自核心 Expr 对象

    @property
    def shape(self) -> tuple[Expr, Expr]:
        # 返回矩阵的形状
        raise NotImplementedError

    @property
    def _add_handler(self):
        # 返回加法操作的处理类
        return MatAdd

    @property
    def _mul_handler(self):
        # 返回乘法操作的处理类
        return MatMul

    def __neg__(self):
        # 返回矩阵取负结果
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        # 抽象方法，返回矩阵的绝对值
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        # 加法运算的魔法方法重载，返回加法结果
        return MatAdd(self, other).doit()

    @_sympifyit('other', NotImplemented)
    # 使用装饰器调用 `call_highest_priority`，注册 `__add__` 为最高优先级的加法操作
    @call_highest_priority('__add__')
    def __radd__(self, other):
        # 创建并返回 `MatAdd` 对象，将 `other` 与 `self` 相加并执行
        return MatAdd(other, self).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__rsub__` 为最高优先级的右减法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        # 创建并返回 `MatAdd` 对象，将 `self` 与 `-other` 相加并执行
        return MatAdd(self, -other).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__sub__` 为最高优先级的减法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        # 创建并返回 `MatAdd` 对象，将 `other` 与 `-self` 相加并执行
        return MatAdd(other, -self).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__rmul__` 为最高优先级的右乘法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        # 创建并返回 `MatMul` 对象，将 `self` 与 `other` 相乘并执行
        return MatMul(self, other).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__rmul__` 为最高优先级的右矩阵乘法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        # 创建并返回 `MatMul` 对象，将 `self` 与 `other` 进行矩阵乘法并执行
        return MatMul(self, other).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__mul__` 为最高优先级的乘法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        # 创建并返回 `MatMul` 对象，将 `other` 与 `self` 相乘并执行
        return MatMul(other, self).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__mul__` 为最高优先级的右矩阵乘法操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        # 创建并返回 `MatMul` 对象，将 `other` 与 `self` 进行矩阵乘法并执行
        return MatMul(other, self).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__rpow__` 为最高优先级的右幂运算操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        # 创建并返回 `MatPow` 对象，将 `self` 和 `other` 进行幂运算并执行
        return MatPow(self, other).doit()

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__pow__` 为最高优先级的幂运算操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        # 抛出未实现错误，提示暂不支持矩阵幂运算的右操作数
        raise NotImplementedError("Matrix Power not defined")

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__rtruediv__` 为最高优先级的右真除操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        # 返回 `self` 乘以 `other` 的负一次幂
        return self * other**S.NegativeOne

    # 使用装饰器 `_sympifyit` 将 `other` 转换为合适类型，注册 `__truediv__` 为最高优先级的真除操作
    @ _sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        # 抛出未实现错误，提示暂不支持右真除操作
        raise NotImplementedError()
        # return MatMul(other, Pow(self, S.NegativeOne))

    # 返回矩阵的行数
    @property
    def rows(self):
        return self.shape[0]

    # 返回矩阵的列数
    @property
    def cols(self):
        return self.shape[1]

    # 返回矩阵是否为方阵，返回值为布尔类型或 `None`
    @property
    def is_square(self) -> bool | None:
        rows, cols = self.shape
        if isinstance(rows, Integer) and isinstance(cols, Integer):
            return rows == cols
        if rows == cols:
            return True
        return None

    # 返回矩阵的共轭转置
    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(Transpose(self))

    # 返回矩阵的实部和虚部，可以选择是否进行深度转换及其他提示参数
    def as_real_imag(self, deep=True, **hints):
        return self._eval_as_real_imag()

    # 返回矩阵的实部和虚部
    def _eval_as_real_imag(self):
        # 计算并返回矩阵的实部和虚部
        real = S.Half * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate())/(2*S.ImaginaryUnit)
        return (real, im)

    # 返回矩阵的逆矩阵
    def _eval_inverse(self):
        return Inverse(self)

    # 返回矩阵的行列式
    def _eval_determinant(self):
        return Determinant(self)

    # 返回矩阵的转置
    def _eval_transpose(self):
        return Transpose(self)

    # 返回矩阵的迹，暂返回 `None`
    def _eval_trace(self):
        return None
    # 用于子类中实现幂运算的简化。在 MatPow.doit() 已经处理了指数为 -1、0、1 的情况，因此这些情况不需要在此实现中处理。
    def _eval_power(self, exp):
        return MatPow(self, exp)

    # 简化矩阵表达式。如果矩阵是原子的，则直接返回自身；否则调用 sympy.simplify 中的 simplify 函数处理其参数后重新构造表达式。
    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            from sympy.simplify import simplify
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    # 返回矩阵的伴随矩阵。
    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    # 计算矩阵对指定变量 x 的 n 阶导数。
    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    # 计算矩阵对指定变量 x 的一阶导数。如果表达式中包含 x，则调用超类的对应方法；否则返回相同形状的零矩阵。
    def _eval_derivative(self, x):
        # `x` 是一个标量：
        if self.has(x):
            # 检查是否有其他使用它的方法：
            return super()._eval_derivative(x)
        else:
            return ZeroMatrix(*self.shape)

    # 类方法，用于检查矩阵维度的有效性，确保维度是非负整数。
    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))

    # 抛出未实现异常，表示矩阵类型不支持索引操作。
    def _entry(self, i, j, **kwargs):
        raise NotImplementedError(
            "Indexing not implemented for %s" % self.__class__.__name__)

    # 返回伴随矩阵。
    def adjoint(self):
        return adjoint(self)

    # 返回矩阵作为乘积的系数。在这里默认返回单位矩阵和自身的乘积。
    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return S.One, self

    # 返回矩阵的共轭矩阵。
    def conjugate(self):
        return conjugate(self)

    # 返回矩阵的转置。
    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    # 属性方法，返回矩阵的转置，等同于调用 transpose() 方法。
    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    # 返回矩阵的逆矩阵。如果矩阵不是方阵，则抛出 NonSquareMatrixError 异常。
    def inverse(self):
        if self.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix')
        return self._eval_inverse()

    # 返回矩阵的逆矩阵，等同于调用 inverse() 方法。
    def inv(self):
        return self.inverse()

    # 返回矩阵的行列式。
    def det(self):
        from sympy.matrices.expressions.determinant import det
        return det(self)

    # 属性方法，返回矩阵的逆矩阵，等同于调用 inverse() 方法。
    @property
    def I(self):
        return self.inverse()

    # 检查指定的行列索引是否有效，确保其为整数、符号或表达式，并且在矩阵的有效行列范围内。
    def valid_index(self, i, j):
        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return (is_valid(i) and is_valid(j) and
                (self.rows is None or
                 (i >= -self.rows) != False and (i < self.rows) != False) and
                (j >= -self.cols) != False and (j < self.cols) != False)
    # 定义特殊方法 __getitem__，处理对象的索引操作
    def __getitem__(self, key):
        # 检查 key 是否为单个切片，创建并返回 MatrixSlice 对象
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        
        # 检查 key 是否为二元组
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            # 如果 i 或 j 是切片对象，则创建并返回 MatrixSlice 对象
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            
            # 将 i 和 j 转换为符号表达式，并检查索引的有效性
            i, j = _sympify(i), _sympify(j)
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid indices (%s, %s)" % (i, j))
        
        # 如果 key 是整数或者 SymPy 的整数类型
        elif isinstance(key, (SYMPY_INTS, Integer)):
            # 获取矩阵的行数和列数
            rows, cols = self.shape
            # 如果列数不是整数类型，则抛出异常
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('''
                    Single indexing is only supported when the number
                    of columns is known.'''))
            
            # 将 key 转换为符号表达式，计算其在矩阵中的行列索引
            key = _sympify(key)
            i = key // cols
            j = key % cols
            # 检查索引的有效性，并返回对应元素
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid index %s" % key)
        
        # 如果 key 是符号或者表达式类型，则抛出异常
        elif isinstance(key, (Symbol, Expr)):
            raise IndexError(filldedent('''
                Only integers may be used when addressing the matrix
                with a single index.'''))
        
        # 如果 key 类型不符合预期，则抛出异常
        raise IndexError("Invalid index, wanted %s[i,j]" % self)
    
    # 判断矩阵的行数和列数是否为符号表达式
    def _is_shape_symbolic(self) -> bool:
        return (not isinstance(self.rows, (SYMPY_INTS, Integer))
            or not isinstance(self.cols, (SYMPY_INTS, Integer)))
    
    # 将矩阵表示为显式矩阵，返回 ImmutableDenseMatrix 类型的对象
    def as_explicit(self):
        """
        Returns a dense Matrix with elements represented explicitly

        Returns an object of type ImmutableDenseMatrix.

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.as_explicit()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_mutable: returns mutable Matrix type

        """
        # 如果矩阵的行数或列数是符号表达式，则抛出异常
        if self._is_shape_symbolic():
            raise ValueError(
                'Matrix with symbolic shape '
                'cannot be represented explicitly.')
        
        # 导入 ImmutableDenseMatrix 类，并使用推导式创建其对象
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix([[self[i, j]
                            for j in range(self.cols)]
                            for i in range(self.rows)])
    def as_mutable(self):
        """
        Returns a dense, mutable matrix with elements represented explicitly

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.shape
        (3, 3)
        >>> I.as_mutable()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_explicit: returns ImmutableDenseMatrix
        """
        # 转换为显式表示的不可变密集矩阵，然后将其转换为可变形式
        return self.as_explicit().as_mutable()

    def __array__(self, dtype=object, copy=None):
        # 如果指定了不复制 (copy=False)，则抛出类型错误
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        # 导入numpy库中的empty函数
        from numpy import empty
        # 创建一个空的对象类型的数组，形状与矩阵相同
        a = empty(self.shape, dtype=object)
        # 遍历矩阵的每个元素，将其赋值给数组
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = self[i, j]
        # 返回生成的数组
        return a

    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
        # 比较当前矩阵与另一个矩阵在元素级别上是否相等
        return self.as_explicit().equals(other)

    def canonicalize(self):
        # 返回当前矩阵本身，表示矩阵的规范形式
        return self

    def as_coeff_mmul(self):
        # 返回一个元组 (S.One, MatMul(self))，表示当前矩阵可以看作是与S.One相乘的MatMul对象
        return S.One, MatMul(self)

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        r"""
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        """
        # 导入所需模块来进行张量和数组表示之间的转换
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
        # 初始化一个空列表用于存放首要索引
        first_indices = []
        # 如果给定了 first_index 参数，则将其添加到首要索引列表中
        if first_index is not None:
            first_indices.append(first_index)
        # 如果给定了 last_index 参数，则将其添加到首要索引列表中
        if last_index is not None:
            first_indices.append(last_index)
        # 将表达式转换为数组表示形式
        arr = convert_indexed_to_array(expr, first_indices=first_indices)
        # 将数组表示形式转换为矩阵表达形式并返回
        return convert_array_to_matrix(arr)

    def applyfunc(self, func):
        # 导入 ElementwiseApplyFunction 类并应用给定的函数到当前对象上
        from .applyfunc import ElementwiseApplyFunction
        return ElementwiseApplyFunction(func, self)
@dispatch(MatrixExpr, Expr)
def _eval_is_eq(lhs, rhs): # noqa:F811
    # 默认返回 False，因为不同类型的表达式不可能相等
    return False

@dispatch(MatrixExpr, MatrixExpr)  # type: ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # 如果两个矩阵形状不同，则它们不相等
    if lhs.shape != rhs.shape:
        return False
    # 如果两个矩阵相减的结果是零矩阵，则它们相等
    if (lhs - rhs).is_ZeroMatrix:
        return True

def get_postprocessor(cls):
    def _postprocessor(expr):
        # 避免循环导入，顶层不能直接引用 MatMul 或 MatAdd
        mat_class = {Mul: MatMul, Add: MatAdd}[cls]
        nonmatrices = []
        matrices = []
        # 遍历表达式的每个项，将矩阵和非矩阵分开存储
        for term in expr.args:
            if isinstance(term, MatrixExpr):
                matrices.append(term)
            else:
                nonmatrices.append(term)

        # 如果没有矩阵项，直接返回原始表达式
        if not matrices:
            return cls._from_args(nonmatrices)

        # 如果有非矩阵项，处理乘法的情况
        if nonmatrices:
            if cls == Mul:
                for i in range(len(matrices)):
                    if not matrices[i].is_MatrixExpr:
                        # 如果其中一个矩阵是显式给定的，将标量吸收到它内部
                        # (doit 将所有显式矩阵合并为一个，所以吸收的顺序不重要)
                        matrices[i] = matrices[i].__mul__(cls._from_args(nonmatrices))
                        nonmatrices = []
                        break

            else:
                # 维持创建 Add(scalar, matrix) 的能力，以避免引发异常
                # 这样不同算法可以用非交换符号替换矩阵表达式来操作它们
                return cls._from_args(nonmatrices + [mat_class(*matrices).doit(deep=False)])

        # 如果 mat_class 是 MatAdd，则返回相加后的矩阵
        if mat_class == MatAdd:
            return mat_class(*matrices).doit(deep=False)
        # 否则返回乘法处理后的结果
        return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
    return _postprocessor

# 将后处理器映射到 MatrixExpr 的构造函数
Basic._constructor_postprocessor_mapping[MatrixExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}

def _matrix_derivative(expr, x, old_algorithm=False):

    if isinstance(expr, MatrixBase) or isinstance(x, MatrixBase):
        # 对于显式矩阵不使用数组表达式：
        old_algorithm = True

    # 如果采用旧算法，则调用旧算法的矩阵导数计算函数
    if old_algorithm:
        return _matrix_derivative_old_algorithm(expr, x)

    # 否则，使用新算法转换为数组表达式，计算导数后再转换回矩阵表达式
    from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    array_expr = convert_matrix_to_array(expr)
    diff_array_expr = array_derive(array_expr, x)
    diff_matrix_expr = convert_array_to_matrix(diff_array_expr)
    return diff_matrix_expr

def _matrix_derivative_old_algorithm(expr, x):
    # 使用旧算法的矩阵导数计算函数
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)

    # 构建导数计算结果的各部分
    parts = [i.build() for i in lines]
    # 导入从数组到矩阵的转换函数
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    # 使用列表推导式将每个元素转换为矩阵表达式
    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

    # 定义一个函数，用于获取元素的形状
    def _get_shape(elem):
        # 如果元素是 MatrixExpr 类型，返回其形状
        if isinstance(elem, MatrixExpr):
            return elem.shape
        # 否则返回默认形状 (1, 1)
        return 1, 1

    # 定义一个函数，用于计算 parts 中所有元素的秩非1的个数
    def get_rank(parts):
        # 统计所有元素中秩不为1或None的个数
        return sum(j not in (1, None) for i in parts for j in _get_shape(i))

    # 计算每个部分的秩
    ranks = [get_rank(i) for i in parts]
    # 取第一个部分的秩作为整体的秩
    rank = ranks[0]

    # 定义一个函数，用于合并单维度
    def contract_one_dims(parts):
        # 如果只有一个部分，则直接返回该部分
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            # 如果 p2 是矩阵，则转置 p2
            if p2.is_Matrix:
                p2 = p2.T
            # 如果 p1 是单位矩阵 Identity(1)，则以 p2 作为基础
            if p1 == Identity(1):
                pbase = p2
            # 如果 p2 是单位矩阵 Identity(1)，则以 p1 作为基础
            elif p2 == Identity(1):
                pbase = p1
            else:
                # 否则，以 p1 乘以 p2 作为基础
                pbase = p1 * p2
            # 如果 parts 只剩下两个部分，则返回 pbase
            if len(parts) == 2:
                return pbase
            else:  # len(parts) > 2
                # 如果 pbase 是矩阵，则抛出 ValueError
                if pbase.is_Matrix:
                    raise ValueError("")
                # 否则，以 Mul.fromiter(parts[2:]) 作为基础的 pbase
                return pbase * Mul.fromiter(parts[2:])

    # 如果整体秩小于等于2，则返回所有部分的合并结果
    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])

    # 否则，返回 ArrayDerivative(expr, x)
    return ArrayDerivative(expr, x)
class MatrixElement(Expr):
    # MatrixElement 类继承自 Expr 类
    parent = property(lambda self: self.args[0])
    # 获取父级对象的属性，这里是第一个参数
    i = property(lambda self: self.args[1])
    # 获取行索引 i 的属性，这里是第二个参数
    j = property(lambda self: self.args[2])
    # 获取列索引 j 的属性，这里是第三个参数
    _diff_wrt = True
    # 指示该表达式可以进行微分
    is_symbol = True
    # 标记这是一个符号对象
    is_commutative = True
    # 标记这是一个可交换对象

    def __new__(cls, name, n, m):
        # 创建一个新的 MatrixElement 对象
        n, m = map(_sympify, (n, m))
        # 将 n 和 m 转换为符号对象
        if isinstance(name, str):
            name = Symbol(name)
            # 如果 name 是字符串，则转换为符号对象
        else:
            if isinstance(name, MatrixBase):
                # 如果 name 是矩阵对象
                if n.is_Integer and m.is_Integer:
                    return name[n, m]
                    # 如果 n 和 m 都是整数，则返回矩阵 name 的子元素
                name = _sympify(name)  # change mutable into immutable
                # 将可变的对象转换为不可变对象
            else:
                name = _sympify(name)
                # 将 name 转换为符号对象
                if not isinstance(name.kind, MatrixKind):
                    raise TypeError("First argument of MatrixElement should be a matrix")
                    # 如果 name 不是矩阵类型，则抛出类型错误异常
            if not getattr(name, 'valid_index', lambda n, m: True)(n, m):
                raise IndexError('indices out of range')
                # 如果索引 n, m 超出范围，则抛出索引错误异常
        obj = Expr.__new__(cls, name, n, m)
        # 使用父类的构造方法创建对象
        return obj

    @property
    def symbol(self):
        # 返回 MatrixElement 对象的第一个参数，即符号对象
        return self.args[0]

    def doit(self, **hints):
        # 对 MatrixElement 对象执行计算
        deep = hints.get('deep', True)
        # 获取深度计算标志，默认为 True
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
            # 如果需要深度计算，则递归地计算每个参数
        else:
            args = self.args
            # 否则直接使用参数列表
        return args[0][args[1], args[2]]
        # 返回索引为 args[1], args[2] 的矩阵元素值

    @property
    def indices(self):
        # 返回 MatrixElement 对象的索引，即第二个和第三个参数
        return self.args[1:]

    def _eval_derivative(self, v):
        # 计算 MatrixElement 对象对变量 v 的导数

        if not isinstance(v, MatrixElement):
            # 如果 v 不是 MatrixElement 类型
            return self.parent.diff(v)[self.i, self.j]
            # 返回父级对象相对于 v 的偏导数的第 i 行、第 j 列元素

        M = self.args[0]
        # 获取 MatrixElement 对象的第一个参数

        m, n = self.parent.shape
        # 获取父级对象的形状信息

        if M == v.args[0]:
            # 如果 MatrixElement 对象的参数与 v 的参数相同
            return KroneckerDelta(self.args[1], v.args[1], (0, m-1)) * \
                   KroneckerDelta(self.args[2], v.args[2], (0, n-1))
            # 返回 KroneckerDelta 函数的结果，用于描述矩阵元素的 delta 函数

        if isinstance(M, Inverse):
            from sympy.concrete.summations import Sum
            # 如果 M 是 Inverse 类型，则导入 Sum 类
            i, j = self.args[1:]
            # 获取 MatrixElement 对象的行和列索引
            i1, i2 = symbols("z1, z2", cls=Dummy)
            # 创建虚拟符号 i1 和 i2
            Y = M.args[0]
            # 获取 Inverse 对象的参数 Y
            r1, r2 = Y.shape
            # 获取 Y 的形状信息
            return -Sum(M[i, i1]*Y[i1, i2].diff(v)*M[i2, j], (i1, 0, r1-1), (i2, 0, r2-1))
            # 返回 Sum 函数的结果，用于描述矩阵元素的偏导数

        if self.has(v.args[0]):
            # 如果 MatrixElement 对象包含变量 v 的参数
            return None
            # 返回空值

        return S.Zero
        # 返回零值


class MatrixSymbol(MatrixExpr):
    # MatrixSymbol 类继承自 MatrixExpr 类
    """Symbolic representation of a Matrix object

    创建一个符号表示的矩阵对象
    """

    is_commutative = False
    # 标记这是一个非可交换对象
    is_symbol = True
    # 标记这是一个符号对象
    _diff_wrt = True
    # 指示该表达式可以进行微分

    def __new__(cls, name, n, m):
        # 创建一个新的 MatrixSymbol 对象
        n, m = _sympify(n), _sympify(m)
        # 将 n 和 m 转换为符号对象

        cls._check_dim(m)
        # 检查 m 的维度
        cls._check_dim(n)
        # 检查 n 的维度

        if isinstance(name, str):
            name = Str(name)
            # 如果 name 是字符串，则转换为字符串对象
        obj = Basic.__new__(cls, name, n, m)
        # 使用父类的构造方法创建对象
        return obj

    @property
    def shape(self):
        # 返回 MatrixSymbol 对象的形状，即第二个和第三个参数
        return self.args[1], self.args[2]

    @property
    def```
    # 返回对象的第一个参数的名称
    def name(self):
        return self.args[0].name

    # 返回一个 MatrixElement 对象，表示矩阵中的元素
    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    # 返回自由符号的集合，这里只包含自身
    @property
    def free_symbols(self):
        return {self}

    # 简化表达式的求值，直接返回自身，因为这个类表示一个抽象的矩阵
    def _eval_simplify(self, **kwargs):
        return self

    # 计算关于变量 x 的导数，返回一个与原矩阵形状相同的零矩阵
    def _eval_derivative(self, x):
        # x 是一个标量:
        return ZeroMatrix(self.shape[0], self.shape[1])

    # 计算矩阵关于变量 x 的导数，返回一个包含两个矩阵的列表
    def _eval_derivative_matrix_lines(self, x):
        if self != x:
            # 如果 self 不等于 x，则返回第一个矩阵为零矩阵，第二个矩阵为单位矩阵或标量 1
            first = ZeroMatrix(x.shape[0], self.shape[0]) if self.shape[0] != 1 else S.Zero
            second = ZeroMatrix(x.shape[1], self.shape[1]) if self.shape[1] != 1 else S.Zero
            return [_LeftRightArgs(
                [first, second],
            )]
        else:
            # 如果 self 等于 x，则返回第一个矩阵为单位矩阵或标量 1，第二个矩阵为单位矩阵或标量 1
            first = Identity(self.shape[0]) if self.shape[0] != 1 else S.One
            second = Identity(self.shape[1]) if self.shape[1] != 1 else S.One
            return [_LeftRightArgs(
                [first, second],
            )]
# 定义一个函数 `matrix_symbols`，用于从表达式中提取所有的矩阵符号
def matrix_symbols(expr):
    # 使用列表推导式，遍历表达式中的自由符号集合，筛选出其中的矩阵符号
    return [sym for sym in expr.free_symbols if sym.is_Matrix]


# 定义一个帮助类 `_LeftRightArgs`，用于计算矩阵导数
class _LeftRightArgs:
    r"""
    辅助类，用于计算矩阵导数。

    逻辑：当表达式通过矩阵 `X_{mn}` 导数时，创建两行矩阵乘积：
    第一行与 `m` 缩并，第二行与 `n` 缩并。

    转置通过连接新矩阵的方式翻转这两行。

    迹连接两行的末端。
    """

    # 初始化方法，接受 lines 和 higher 两个参数
    def __init__(self, lines, higher=S.One):
        self._lines = list(lines)  # 将 lines 转换为列表
        self._first_pointer_parent = self._lines  # 第一个指针的父对象为 lines
        self._first_pointer_index = 0  # 第一个指针的索引为 0
        self._first_line_index = 0  # 第一行的索引为 0
        self._second_pointer_parent = self._lines  # 第二个指针的父对象为 lines
        self._second_pointer_index = 1  # 第二个指针的索引为 1
        self._second_line_index = 1  # 第二行的索引为 1
        self.higher = higher  # 初始化 higher 属性为给定的 higher 参数

    # 属性方法，返回第一个指针的当前值
    @property
    def first_pointer(self):
        return self._first_pointer_parent[self._first_pointer_index]

    # 属性方法，设置第一个指针的当前值
    @first_pointer.setter
    def first_pointer(self, value):
        self._first_pointer_parent[self._first_pointer_index] = value

    # 属性方法，返回第二个指针的当前值
    @property
    def second_pointer(self):
        return self._second_pointer_parent[self._second_pointer_index]

    # 属性方法，设置第二个指针的当前值
    @second_pointer.setter
    def second_pointer(self, value):
        self._second_pointer_parent[self._second_pointer_index] = value

    # repr 方法，返回对象的字符串表示
    def __repr__(self):
        built = [self._build(i) for i in self._lines]  # 构建 lines 中每个元素的表示形式
        return "_LeftRightArgs(lines=%s, higher=%s)" % (
            built,
            self.higher,
        )

    # 转置方法，翻转指针和行的连接关系
    def transpose(self):
        self._first_pointer_parent, self._second_pointer_parent = self._second_pointer_parent, self._first_pointer_parent
        self._first_pointer_index, self._second_pointer_index = self._second_pointer_index, self._first_pointer_index
        self._first_line_index, self._second_line_index = self._second_line_index, self._first_line_index
        return self

    # 静态方法，用于构建表达式
    @staticmethod
    def _build(expr):
        if isinstance(expr, ExprBuilder):
            return expr.build()  # 如果 expr 是 ExprBuilder 类型，则调用其 build 方法
        if isinstance(expr, list):
            if len(expr) == 1:
                return expr[0]  # 如果 expr 是长度为 1 的列表，则返回其第一个元素
            else:
                return expr[0](*[_LeftRightArgs._build(i) for i in expr[1]])  # 否则，构建表达式的其余部分
        else:
            return expr  # 如果 expr 是其他类型，则直接返回它

    # 构建方法，构建 lines 中的所有元素
    def build(self):
        data = [self._build(i) for i in self._lines]  # 使用 _build 方法构建 lines 中的每个元素
        if self.higher != 1:
            data += [self._build(self.higher)]  # 如果 higher 不等于 1，则构建 higher 并添加到 data 中
        data = list(data)  # 将 data 转换为列表
        return data  # 返回构建后的 data
    def matrix_form(self):
        # 检查是否为一维数组，如果是则无法表示更高维度的数组
        if self.first != 1 and self.higher != 1:
            raise ValueError("higher dimensional array cannot be represented")

        def _get_shape(elem):
            # 返回矩阵表达式的形状
            if isinstance(elem, MatrixExpr):
                return elem.shape
            return (None, None)

        # 检查第一个矩阵和第二个矩阵的列数是否相等
        if _get_shape(self.first)[1] != _get_shape(self.second)[1]:
            # 移除一维单位矩阵：在 `a.diff(a)` 中需要（其中 `a` 是一个向量）
            if _get_shape(self.second) == (1, 1):
                return self.first * self.second[0, 0]
            if _get_shape(self.first) == (1, 1):
                return self.first[1, 1] * self.second.T
            raise ValueError("incompatible shapes")
        
        # 如果第一个矩阵不为一维，则返回第一个矩阵乘以第二个矩阵的转置
        if self.first != 1:
            return self.first * self.second.T
        else:
            return self.higher

    def rank(self):
        """
        Number of dimensions different from trivial (warning: not related to
        matrix rank).
        """
        rank = 0
        # 计算第一个矩阵的维数不为一的数量
        if self.first != 1:
            rank += sum(i != 1 for i in self.first.shape)
        # 计算第二个矩阵的维数不为一的数量
        if self.second != 1:
            rank += sum(i != 1 for i in self.second.shape)
        # 如果 higher 不为一，则增加维数 2
        if self.higher != 1:
            rank += 2
        return rank

    def _multiply_pointer(self, pointer, other):
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from ...tensor.array.expressions.array_expressions import ArrayContraction

        # 构建数组收缩的表达式
        subexpr = ExprBuilder(
            ArrayContraction,
            [
                ExprBuilder(
                    ArrayTensorProduct,
                    [
                        pointer,
                        other
                    ]
                ),
                (1, 2)
            ],
            validator=ArrayContraction._validate
        )

        return subexpr

    def append_first(self, other):
        # 将 other 乘到 self 的第一个指针上
        self.first_pointer *= other

    def append_second(self, other):
        # 将 other 乘到 self 的第二个指针上
        self.second_pointer *= other
# 定义一个函数 _make_matrix，接受一个参数 x
def _make_matrix(x):
    # 导入 ImmutableDenseMatrix 类
    from sympy.matrices.immutable import ImmutableDenseMatrix
    # 如果 x 是 MatrixExpr 类型的对象，则直接返回 x，不进行处理
    if isinstance(x, MatrixExpr):
        return x
    # 否则，将 x 包装成一个二维不可变矩阵（ImmutableDenseMatrix）
    return ImmutableDenseMatrix([[x]])


# 导入矩阵乘法相关模块
from .matmul import MatMul
# 导入矩阵加法相关模块
from .matadd import MatAdd
# 导入矩阵幂相关模块
from .matpow import MatPow
# 导入矩阵转置相关模块
from .transpose import Transpose
# 导入矩阵求逆相关模块
from .inverse import Inverse
# 导入特殊矩阵模块，包括零矩阵和单位矩阵
from .special import ZeroMatrix, Identity
# 导入矩阵行列式相关模块
from .determinant import Determinant
```