# `D:\src\scipysrc\sympy\sympy\matrices\expressions\_shape.py`

```
# 导入必要的符号和表达式类别，用于矩阵运算的形状验证和逻辑操作
from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.exceptions import ShapeError
from typing import Union


def is_matadd_valid(*args: MatrixExpr) -> Boolean:
    """Return the symbolic condition how ``MatAdd``, ``HadamardProduct``
    makes sense.

    Parameters
    ==========

    args
        The list of arguments of matrices to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matadd_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matadd_valid(A, B)
    Eq(m, p) & Eq(n, q)
    """
    # 提取每个矩阵参数的行和列的形状
    rows, cols = zip(*(arg.shape for arg in args))
    # 返回一个逻辑 AND 条件，检查相邻矩阵参数的行和列形状是否相等
    return And(
        *(Eq(i, j) for i, j in zip(rows[:-1], rows[1:])),
        *(Eq(i, j) for i, j in zip(cols[:-1], cols[1:])),
    )


def is_matmul_valid(*args: Union[MatrixExpr, Expr]) -> Boolean:
    """Return the symbolic condition how ``MatMul`` makes sense

    Parameters
    ==========

    args
        The list of arguments of matrices and scalar expressions to be tested
        for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_matmul_valid

    >>> m, n, p, q = symbols('m n p q')
    >>> A = MatrixSymbol('A', m, n)
    >>> B = MatrixSymbol('B', p, q)
    >>> is_matmul_valid(A, B)
    Eq(n, p)
    """
    # 提取每个矩阵参数（非标量表达式）的行和列的形状
    rows, cols = zip(*(arg.shape for arg in args if isinstance(arg, MatrixExpr)))
    # 返回一个逻辑 AND 条件，检查相邻矩阵参数的列数和下一个矩阵参数的行数是否相等
    return And(*(Eq(i, j) for i, j in zip(cols[:-1], rows[1:])))


def is_square(arg: MatrixExpr, /) -> Boolean:
    """Return the symbolic condition how the matrix is assumed to be square

    Parameters
    ==========

    arg
        The matrix to be tested for.

    Examples
    ========

    >>> from sympy import MatrixSymbol, symbols
    >>> from sympy.matrices.expressions._shape import is_square

    >>> m, n = symbols('m n')
    >>> A = MatrixSymbol('A', m, n)
    >>> is_square(A)
    Eq(m, n)
    """
    # 返回一个逻辑条件，检查矩阵参数的行数是否等于列数，即矩阵是否为方阵
    return Eq(arg.rows, arg.cols)


def validate_matadd_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for addition only for integer values"""
    # 提取每个矩阵参数的行和列的形状，并检查是否包含整数类型
    rows, cols = zip(*(x.shape for x in args))
    # 如果参数中的行数包含不同的整数值，抛出形状错误异常
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), rows))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {rows}")
    # 如果参数中的列数包含不同的整数值，抛出形状错误异常
    if len(set(filter(lambda x: isinstance(x, (int, Integer)), cols))) > 1:
        raise ShapeError(f"Matrices have mismatching shape: {cols}")


def validate_matmul_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for multiplication only for integer values"""

    # 这里的函数暂时没有实现，需要根据需求补充其功能和注释
    pass
    # 对传入的参数进行迭代，每次迭代同时考虑相邻的两个元素 A 和 B
    for A, B in zip(args[:-1], args[1:]):
        # 从 A 和 B 中分别获取 cols 和 rows 属性值，并赋值给 i 和 j
        i, j = A.cols, B.rows
        # 检查 i 和 j 是否都是整数类型或者 Integer 类型，并且二者不相等
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and i != j:
            # 如果 i 和 j 的类型正确且不相等，抛出形状错误异常，指示矩阵尺寸不一致
            raise ShapeError("Matrices are not aligned", i, j)
```