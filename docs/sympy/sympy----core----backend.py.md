# `D:\src\scipysrc\sympy\sympy\core\backend.py`

```
import os
# 从环境变量中获取是否使用 SymEngine 的配置，将其转换为小写字符串后判断是否为 True
USE_SYMENGINE = os.getenv('USE_SYMENGINE', '0')
USE_SYMENGINE = USE_SYMENGINE.lower() in ('1', 't', 'true')  # type: ignore

if USE_SYMENGINE:
    # 若使用 SymEngine，则导入以下符号和函数
    from symengine import (Symbol, Integer, sympify as sympify_symengine, S,
        SympifyError, exp, log, gamma, sqrt, I, E, pi, Matrix,
        sin, cos, tan, cot, csc, sec, asin, acos, atan, acot, acsc, asec,
        sinh, cosh, tanh, coth, asinh, acosh, atanh, acoth,
        lambdify, symarray, diff, zeros, eye, diag, ones,
        expand, Function, symbols, var, Add, Mul, Derivative,
        ImmutableMatrix, MatrixBase, Rational, Basic)
    from symengine.lib.symengine_wrapper import gcd as igcd
    from symengine import AppliedUndef

    def sympify(a, *, strict=False):
        """
        Notes
        =====

        SymEngine's ``sympify`` does not accept keyword arguments and is
        therefore not compatible with SymPy's ``sympify`` with ``strict=True``
        (which ensures that only the types for which an explicit conversion has
        been defined are converted). This wrapper adds an addiotional parameter
        ``strict`` (with default ``False``) that will raise a ``SympifyError``
        if ``strict=True`` and the argument passed to the parameter ``a`` is a
        string.

        See Also
        ========

        sympify: Converts an arbitrary expression to a type that can be used
            inside SymPy.

        """
        # 若 strict=True 且参数 a 是字符串，则抛出 SympifyError 异常
        if strict and isinstance(a, str):
            raise SympifyError(a)
        # 否则调用 SymEngine 的 sympify 函数进行转换
        return sympify_symengine(a)

    # 将 SymEngine 的 sympify 函数的文档字符串与 sympify 函数的文档字符串结合，并修正缩进
    sympify.__doc__ = (
        sympify_symengine.__doc__
        + sympify.__doc__.replace('        ', '    ')  # type: ignore
    )
else:
    # 若不使用 SymEngine，则导入以下符号和函数
    from sympy.core.add import Add
    from sympy.core.basic import Basic
    from sympy.core.function import (diff, Function, AppliedUndef,
        expand, Derivative)
    from sympy.core.mul import Mul
    from sympy.core.intfunc import igcd
    from sympy.core.numbers import pi, I, Integer, Rational, E
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol, var, symbols
    from sympy.core.sympify import SympifyError, sympify
    from sympy.functions.elementary.exponential import log, exp
    from sympy.functions.elementary.hyperbolic import (coth, sinh,
        acosh, acoth, tanh, asinh, atanh, cosh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (csc,
        asec, cos, atan, sec, acot, asin, tan, sin, cot, acsc, acos)
    from sympy.functions.special.gamma_functions import gamma
    from sympy.matrices.dense import (eye, zeros, diag, Matrix,
        ones, symarray)
    # 导入 sympy 库中的 ImmutableMatrix 类
    from sympy.matrices.immutable import ImmutableMatrix
    # 导入 sympy 库中的 MatrixBase 类
    from sympy.matrices.matrixbase import MatrixBase
    # 导入 sympy 库中的 lambdify 函数，用于将 sympy 表达式转换为可计算的函数
    from sympy.utilities.lambdify import lambdify
#
# XXX: Handling of immutable and mutable matrices in SymEngine is inconsistent
# with SymPy's matrix classes in at least SymEngine version 0.7.0. Until that
# is fixed the function below is needed for consistent behaviour when
# attempting to simplify a matrix.
#
# Expected behaviour of a SymPy mutable/immutable matrix .simplify() method:
#
#   Matrix.simplify() : works in place, returns None
#   ImmutableMatrix.simplify() : returns a simplified copy
#
# In SymEngine both mutable and immutable matrices simplify in place and return
# None. This is inconsistent with the matrix being "immutable" and also the
# returned None leads to problems in the mechanics module.
#
# The simplify function should not be used because simplify(M) sympifies the
# matrix M and the SymEngine matrices all sympify to SymPy matrices. If we want
# to work with SymEngine matrices then we need to use their .simplify() method
# but that method does not work correctly with immutable matrices.
#
# The _simplify_matrix function can be removed when the SymEngine bug is fixed.
# Since this should be a temporary problem we do not make this function part of
# the public API.
#

def _simplify_matrix(M):
    """Return a simplified copy of the matrix M"""
    # 检查输入的矩阵 M 是否是 Matrix 或 ImmutableMatrix 的实例，否则抛出类型错误异常
    if not isinstance(M, (Matrix, ImmutableMatrix)):
        raise TypeError("The matrix M must be an instance of Matrix or ImmutableMatrix")
    # 创建矩阵 M 的可变副本
    Mnew = M.as_mutable() # makes a copy if mutable
    # 对副本进行简化操作
    Mnew.simplify()
    # 如果原始矩阵 M 是不可变矩阵 ImmutableMatrix，则将副本 Mnew 转换回不可变状态
    if isinstance(M, ImmutableMatrix):
        Mnew = Mnew.as_immutable()
    # 返回简化后的矩阵副本
    return Mnew


__all__ = [
    'Symbol', 'Integer', 'sympify', 'S', 'SympifyError', 'exp', 'log',
    'gamma', 'sqrt', 'I', 'E', 'pi', 'Matrix', 'sin', 'cos', 'tan', 'cot',
    'csc', 'sec', 'asin', 'acos', 'atan', 'acot', 'acsc', 'asec', 'sinh',
    'cosh', 'tanh', 'coth', 'asinh', 'acosh', 'atanh', 'acoth', 'lambdify',
    'symarray', 'diff', 'zeros', 'eye', 'diag', 'ones', 'expand', 'Function',
    'symbols', 'var', 'Add', 'Mul', 'Derivative', 'ImmutableMatrix',
    'MatrixBase', 'Rational', 'Basic', 'igcd', 'AppliedUndef',
]
```