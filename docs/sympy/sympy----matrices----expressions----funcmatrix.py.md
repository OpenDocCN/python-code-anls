# `D:\src\scipysrc\sympy\sympy\matrices\expressions\funcmatrix.py`

```
from .matexpr import MatrixExpr
from sympy.core.function import FunctionClass, Lambda
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify, sympify
from sympy.matrices import Matrix
from sympy.functions.elementary.complexes import re, im

class FunctionMatrix(MatrixExpr):
    """Represents a matrix using a function (``Lambda``) which gives
    outputs according to the coordinates of each matrix entries.

    Parameters
    ==========

    rows : nonnegative integer. Can be symbolic.

    cols : nonnegative integer. Can be symbolic.

    lamda : Function, Lambda or str
        If it is a SymPy ``Function`` or ``Lambda`` instance,
        it should be able to accept two arguments which represents the
        matrix coordinates.

        If it is a pure string containing Python ``lambda`` semantics,
        it is interpreted by the SymPy parser and casted into a SymPy
        ``Lambda`` instance.

    Examples
    ========

    Creating a ``FunctionMatrix`` from ``Lambda``:

    >>> from sympy import FunctionMatrix, symbols, Lambda, MatPow
    >>> i, j, n, m = symbols('i,j,n,m')
    >>> FunctionMatrix(n, m, Lambda((i, j), i + j))
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Creating a ``FunctionMatrix`` from a SymPy function:

    >>> from sympy import KroneckerDelta
    >>> X = FunctionMatrix(3, 3, KroneckerDelta)
    >>> X.as_explicit()
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    Creating a ``FunctionMatrix`` from a SymPy undefined function:

    >>> from sympy import Function
    >>> f = Function('f')
    >>> X = FunctionMatrix(3, 3, f)
    >>> X.as_explicit()
    Matrix([
    [f(0, 0), f(0, 1), f(0, 2)],
    [f(1, 0), f(1, 1), f(1, 2)],
    [f(2, 0), f(2, 1), f(2, 2)]])

    Creating a ``FunctionMatrix`` from Python ``lambda``:

    >>> FunctionMatrix(n, m, 'lambda i, j: i + j')
    FunctionMatrix(n, m, Lambda((i, j), i + j))

    Example of lazy evaluation of matrix product:

    >>> Y = FunctionMatrix(1000, 1000, Lambda((i, j), i + j))
    >>> isinstance(Y*Y, MatPow) # this is an expression object
    True
    >>> (Y**2)[10,10] # So this is evaluated lazily
    342923500

    Notes
    =====

    This class provides an alternative way to represent an extremely
    dense matrix with entries in some form of a sequence, in a most
    sparse way.
    """
    # 定义一个新的类构造方法 __new__
    def __new__(cls, rows, cols, lamda):
        # 将 rows 和 cols 转换为符号表达式
        rows, cols = _sympify(rows), _sympify(cols)
        # 检查行维度是否符合要求
        cls._check_dim(rows)
        # 检查列维度是否符合要求
        cls._check_dim(cols)

        # 将 lamda 转换为符号表达式
        lamda = sympify(lamda)
        # 如果 lamda 不是 SymPy 函数类或 Lambda 类的实例，则引发 ValueError 异常
        if not isinstance(lamda, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(lamda))

        # 如果 lamda 的参数个数不包含 2，则引发 ValueError 异常
        if 2 not in lamda.nargs:
            raise ValueError(
                '{} should be able to accept 2 arguments.'.format(lamda))

        # 如果 lamda 不是 Lambda 类的实例，则创建两个虚拟变量 i 和 j，并将 lamda 转换为 Lambda 类
        if not isinstance(lamda, Lambda):
            i, j = Dummy('i'), Dummy('j')
            lamda = Lambda((i, j), lamda(i, j))

        # 调用父类的 __new__ 方法创建实例并返回
        return super().__new__(cls, rows, cols, lamda)

    # 返回当前矩阵的形状，即前两个参数
    @property
    def shape(self):
        return self.args[0:2]

    # 返回当前矩阵的 lambda 函数
    @property
    def lamda(self):
        return self.args[2]

    # 返回 lambda 函数在位置 (i, j) 处的值
    def _entry(self, i, j, **kwargs):
        return self.lamda(i, j)

    # 计算矩阵的迹（Trace），使用 SymPy 中的 Trace 函数并执行求值
    def _eval_trace(self):
        from sympy.matrices.expressions.trace import Trace
        from sympy.concrete.summations import Sum
        return Trace(self).rewrite(Sum).doit()

    # 将矩阵表示为实部和虚部的形式返回
    def _eval_as_real_imag(self):
        return (re(Matrix(self)), im(Matrix(self)))
```