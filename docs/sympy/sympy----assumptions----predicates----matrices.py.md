# `D:\src\scipysrc\sympy\sympy\assumptions\predicates\matrices.py`

```
# 导入 Predicate 类，用于创建谓词（predicate）
from sympy.assumptions import Predicate
# 导入 Dispatcher 类，用于创建分派器（dispatcher）
from sympy.multipledispatch import Dispatcher

# 创建 SquarePredicate 类，继承自 Predicate 类
class SquarePredicate(Predicate):
    """
    Square matrix predicate.

    Explanation
    ===========

    ``Q.square(x)`` is true iff ``x`` is a square matrix. A square matrix
    is a matrix with the same number of rows and columns.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('X', 2, 3)
    >>> ask(Q.square(X))
    True
    >>> ask(Q.square(Y))
    False
    >>> ask(Q.square(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.square(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_matrix

    """
    # 设置谓词名称为 'square'
    name = 'square'
    # 创建一个名为 'SquareHandler' 的分派器（dispatcher），用于处理 Q.square
    handler = Dispatcher("SquareHandler", doc="Handler for Q.square.")


# 创建 SymmetricPredicate 类，继承自 Predicate 类
class SymmetricPredicate(Predicate):
    """
    Symmetric matrix predicate.

    Explanation
    ===========

    ``Q.symmetric(x)`` is true iff ``x`` is a square matrix and is equal to
    its transpose. Every square diagonal matrix is a symmetric matrix.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.symmetric(X*Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(X + Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(Y))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_matrix

    """
    # 设置谓词名称为 'symmetric'
    name = 'symmetric'
    # 创建一个名为 'SymmetricHandler' 的分派器（dispatcher），用于处理 Q.symmetric
    handler = Dispatcher("SymmetricHandler", doc="Handler for Q.symmetric.")


# 创建 InvertiblePredicate 类，继承自 Predicate 类
class InvertiblePredicate(Predicate):
    """
    Invertible matrix predicate.

    Explanation
    ===========

    ``Q.invertible(x)`` is true iff ``x`` is an invertible matrix.
    A square matrix is called invertible only if its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.invertible(X*Y), Q.invertible(X))
    False
    >>> ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z))
    True
    >>> ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Invertible_matrix

    """
    # 设置谓词名称为 'invertible'
    name = 'invertible'
    # 创建一个名为 'InvertibleHandler' 的分派器（dispatcher），用于处理 Q.invertible
    handler = Dispatcher("InvertibleHandler", doc="Handler for Q.invertible.")


# 创建 OrthogonalPredicate 类，继承自 Predicate 类
class OrthogonalPredicate(Predicate):
    """
    Orthogonal matrix predicate.

    Explanation
    ===========

    ``Q.orthogonal(x)`` is true iff ``x`` is an orthogonal matrix.
    A square matrix ``M`` is an orthogonal matrix if it satisfies
    ``M^TM = MM^T = I`` where ``M^T`` is the transpose matrix of ``M``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_matrix

    """
    # 设置谓词名称为 'orthogonal'
    name = 'orthogonal'
    # TODO: 添加处理器以使这些键适用于实际矩阵，并在文档字符串中添加更多示例。
    # 创建一个名为 'OrthogonalHandler' 的分派器（dispatcher），用于处理 Q.orthogonal
    handler = Dispatcher("OrthogonalHandler", doc="Handler for Q.orthogonal.")
    # 设置变量 name 为 'orthogonal'
    name = 'orthogonal'
    # 创建一个名为 "OrthogonalHandler" 的调度器对象，并提供文档说明
    handler = Dispatcher("OrthogonalHandler", doc="Handler for key 'orthogonal'.")
class UnitaryPredicate(Predicate):
    """
    Unitary matrix predicate.

    Explanation
    ===========

    ``Q.unitary(x)`` is true iff ``x`` is a unitary matrix.
    Unitary matrix is an analogue to orthogonal matrix. A square
    matrix ``M`` with complex elements is unitary if :math:``M^TM = MM^T= I``
    where :math:``M^T`` is the conjugate transpose matrix of ``M``.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.unitary(Y))
    False
    >>> ask(Q.unitary(X*Z*X), Q.unitary(X) & Q.unitary(Z))
    True
    >>> ask(Q.unitary(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_matrix

    """
    name = 'unitary'
    handler = Dispatcher("UnitaryHandler", doc="Handler for key 'unitary'.")


class FullRankPredicate(Predicate):
    """
    Fullrank matrix predicate.

    Explanation
    ===========

    ``Q.fullrank(x)`` is true iff ``x`` is a full rank matrix.
    A matrix is full rank if all rows and columns of the matrix
    are linearly independent. A square matrix is full rank iff
    its determinant is nonzero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.fullrank(X.T), Q.fullrank(X))
    True
    >>> ask(Q.fullrank(ZeroMatrix(3, 3)))
    False
    >>> ask(Q.fullrank(Identity(3)))
    True

    """
    name = 'fullrank'
    handler = Dispatcher("FullRankHandler", doc="Handler for key 'fullrank'.")


class PositiveDefinitePredicate(Predicate):
    r"""
    Positive definite matrix predicate.

    Explanation
    ===========

    If $M$ is a :math:`n \times n` symmetric real matrix, it is said
    to be positive definite if :math:`Z^TMZ` is positive for
    every non-zero column vector $Z$ of $n$ real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.positive_definite(Y))
    False
    >>> ask(Q.positive_definite(Identity(3)))
    True
    >>> ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
    ...     Q.positive_definite(Z))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Positive-definite_matrix

    """
    name = "positive_definite"
    handler = Dispatcher("PositiveDefiniteHandler", doc="Handler for key 'positive_definite'.")


class UpperTriangularPredicate(Predicate):
    """
    Upper triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called upper triangular matrix if :math:`M_{ij}=0`
    for :math:`i<j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.upper_triangular(Identity(3)))
    True

    """
    # 使用 ask 函数查询 Q 对象的 upper_triangular 方法对 ZeroMatrix(3, 3) 所返回的结果
    >>> ask(Q.upper_triangular(ZeroMatrix(3, 3)))
    # 返回 True，表示 ZeroMatrix(3, 3) 是一个上三角矩阵
    True

    # 引用了数学世界 Wolfram 网站关于上三角矩阵的定义和性质的参考文献
    References
    ==========

    .. [1] https://mathworld.wolfram.com/UpperTriangularMatrix.html

    """
    # 定义变量 name，赋值为字符串 "upper_triangular"
    name = "upper_triangular"
    # 创建一个 Dispatcher 对象，用于处理键 'upper_triangular' 的操作，包含文档字符串描述
    handler = Dispatcher("UpperTriangularHandler", doc="Handler for key 'upper_triangular'.")
class LowerTriangularPredicate(Predicate):
    """
    Lower triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called lower triangular matrix if :math:`M_{ij}=0`
    for :math:`i>j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.lower_triangular(Identity(3)))
    True
    >>> ask(Q.lower_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LowerTriangularMatrix.html

    """
    # 设置谓词名为 'lower_triangular'
    name = "lower_triangular"
    # 创建一个处理程序分发器，用于处理键 'lower_triangular'
    handler = Dispatcher("LowerTriangularHandler", doc="Handler for key 'lower_triangular'.")


class DiagonalPredicate(Predicate):
    """
    Diagonal matrix predicate.

    Explanation
    ===========

    ``Q.diagonal(x)`` is true iff ``x`` is a diagonal matrix. A diagonal
    matrix is a matrix in which the entries outside the main diagonal
    are all zero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.diagonal(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.diagonal(X), Q.lower_triangular(X) &
    ...     Q.upper_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Diagonal_matrix

    """
    # 设置谓词名为 'diagonal'
    name = "diagonal"
    # 创建一个处理程序分发器，用于处理键 'diagonal'
    handler = Dispatcher("DiagonalHandler", doc="Handler for key 'diagonal'.")


class IntegerElementsPredicate(Predicate):
    """
    Integer elements matrix predicate.

    Explanation
    ===========

    ``Q.integer_elements(x)`` is true iff all the elements of ``x``
    are integers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    True

    """
    # 设置谓词名为 'integer_elements'
    name = "integer_elements"
    # 创建一个处理程序分发器，用于处理键 'integer_elements'
    handler = Dispatcher("IntegerElementsHandler", doc="Handler for key 'integer_elements'.")


class RealElementsPredicate(Predicate):
    """
    Real elements matrix predicate.

    Explanation
    ===========

    ``Q.real_elements(x)`` is true iff all the elements of ``x``
    are real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
    True

    """
    # 设置谓词名为 'real_elements'
    name = "real_elements"
    # 创建一个处理程序分发器，用于处理键 'real_elements'
    handler = Dispatcher("RealElementsHandler", doc="Handler for key 'real_elements'.")


class ComplexElementsPredicate(Predicate):
    """
    Complex elements matrix predicate.

    Explanation
    ===========

    ``Q.complex_elements(x)`` is true iff all the elements of ``x``
    are complex numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    True
    >>> ask(Q.complex_elements(X), Q.integer_elements(X))
    True

    """
    # 设置谓词名为 'complex_elements'
    name = "complex_elements"
    # 创建一个处理程序分发器，用于处理键 'complex_elements'
    handler = Dispatcher("ComplexElementsHandler", doc="Handler for key 'complex_elements'.")
    # 创建名为 "ComplexElementsHandler" 的调度程序对象，并设置其文档字符串为 "Handler for key 'complex_elements'."
    handler = Dispatcher("ComplexElementsHandler", doc="Handler for key 'complex_elements'.")
# 定义一个名为 SingularPredicate 的类，继承自 Predicate 类
class SingularPredicate(Predicate):
    """
    Singular matrix predicate.

    A matrix is singular iff the value of its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.singular(X), Q.invertible(X))
    False
    >>> ask(Q.singular(X), ~Q.invertible(X))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/SingularMatrix.html

    """
    # 设置名称属性为 "singular"
    name = "singular"
    # 创建一个名为 handler 的 Dispatcher 对象，用于处理 "singular" 键的操作
    handler = Dispatcher("SingularHandler", doc="Predicate fore key 'singular'.")


# 定义一个名为 NormalPredicate 的类，继承自 Predicate 类
class NormalPredicate(Predicate):
    """
    Normal matrix predicate.

    A matrix is normal if it commutes with its conjugate transpose.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.normal(X), Q.unitary(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_matrix

    """
    # 设置名称属性为 "normal"
    name = "normal"
    # 创建一个名为 handler 的 Dispatcher 对象，用于处理 "normal" 键的操作
    handler = Dispatcher("NormalHandler", doc="Predicate fore key 'normal'.")


# 定义一个名为 TriangularPredicate 的类，继承自 Predicate 类
class TriangularPredicate(Predicate):
    """
    Triangular matrix predicate.

    Explanation
    ===========

    ``Q.triangular(X)`` is true if ``X`` is one that is either lower
    triangular or upper triangular.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.upper_triangular(X))
    True
    >>> ask(Q.triangular(X), Q.lower_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_matrix

    """
    # 设置名称属性为 "triangular"
    name = "triangular"
    # 创建一个名为 handler 的 Dispatcher 对象，用于处理 "triangular" 键的操作
    handler = Dispatcher("TriangularHandler", doc="Predicate fore key 'triangular'.")


# 定义一个名为 UnitTriangularPredicate 的类，继承自 Predicate 类
class UnitTriangularPredicate(Predicate):
    """
    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    """
    # 设置名称属性为 "unit_triangular"
    name = "unit_triangular"
    # 创建一个名为 handler 的 Dispatcher 对象，用于处理 "unit_triangular" 键的操作
    handler = Dispatcher("UnitTriangularHandler", doc="Predicate fore key 'unit_triangular'.")
```