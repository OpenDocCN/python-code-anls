# `D:\src\scipysrc\sympy\sympy\matrices\dense.py`

```
import random  # 导入 random 模块

from sympy.core.basic import Basic  # 导入 sympy 核心基础类 Basic
from sympy.core.singleton import S  # 导入 sympy 单例类 S
from sympy.core.symbol import Symbol  # 导入 sympy 符号类 Symbol
from sympy.core.sympify import sympify  # 导入 sympy 的 sympify 函数
from sympy.functions.elementary.trigonometric import cos, sin  # 导入 sympy 的三角函数 cos 和 sin
from sympy.utilities.decorator import doctest_depends_on  # 导入 sympy 的装饰器 doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 sympy 的异常类 sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence  # 导入 sympy 的迭代工具函数 is_sequence

from .exceptions import ShapeError  # 导入当前包内的异常类 ShapeError
from .decompositions import _cholesky, _LDLdecomposition  # 导入当前包内的分解函数 _cholesky 和 _LDLdecomposition
from .matrixbase import MatrixBase  # 导入当前包内的矩阵基类 MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix  # 导入当前包内的可变矩阵类 MutableRepMatrix 和 不可变矩阵类 RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve  # 导入当前包内的解法函数 _lower_triangular_solve 和 _upper_triangular_solve


__doctest_requires__ = {('symarray',): ['numpy']}  # 设置特定文档测试所需的依赖项为 numpy


def _iszero(x):
    """Returns True if x is zero."""
    return x.is_zero  # 返回 x 是否为零的判断结果


class DenseMatrix(RepMatrix):
    """Matrix implementation based on DomainMatrix as the internal representation"""

    #
    # DenseMatrix is a superclass for both MutableDenseMatrix and
    # ImmutableDenseMatrix. Methods shared by both classes but not for the
    # Sparse classes should be implemented here.
    #

    is_MatrixExpr = False  # 设置类属性 is_MatrixExpr 为 False，表示此类不是 MatrixExpr 类型

    _op_priority = 10.01  # 设置运算优先级为 10.01
    _class_priority = 4  # 设置类优先级为 4

    @property
    def _mat(self):
        sympy_deprecation_warning(
            """
            The private _mat attribute of Matrix is deprecated. Use the
            .flat() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )  # 发出 sympy 弃用警告，指示使用 .flat() 方法代替私有属性 _mat

        return self.flat()  # 返回调用 self.flat() 方法的结果

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'GE'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))  # 返回矩阵的逆，可以指定方法和零判断函数等参数

    def as_immutable(self):
        """Returns an Immutable version of this Matrix
        """
        from .immutable import ImmutableDenseMatrix as cls
        return cls._fromrep(self._rep.copy())  # 返回此矩阵的不可变版本

    def as_mutable(self):
        """Returns a mutable version of this matrix

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return Matrix(self)  # 返回此矩阵的可变版本

    def cholesky(self, hermitian=True):
        return _cholesky(self, hermitian=hermitian)  # 调用 _cholesky 函数进行 Cholesky 分解

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition(self, hermitian=hermitian)  # 调用 _LDLdecomposition 函数进行 LDL 分解

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve(self, rhs)  # 调用 _lower_triangular_solve 函数解下三角线性方程组

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve(self, rhs)  # 调用 _upper_triangular_solve 函数解上三角线性方程组

    cholesky.__doc__ = _cholesky.__doc__  # 将 cholesky 方法的文档字符串设置为 _cholesky 函数的文档字符串
    LDLdecomposition.__doc__ = _LDLdecomposition.__doc__  # 将 LDLdecomposition 方法的文档字符串设置为 _LDLdecomposition 函数的文档字符串
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__  # 将 lower_triangular_solve 方法的文档字符串设置为 _lower_triangular_solve 函数的文档字符串
    # 将函数的文档字符串设置为另一个函数的文档字符串
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
def _force_mutable(x):
    """Return a matrix as a Matrix, otherwise return x."""
    # Check if x has an attribute 'is_Matrix' and if it's True, return its mutable version
    if getattr(x, 'is_Matrix', False):
        return x.as_mutable()
    # If x is an instance of Basic (from SymPy), return x itself
    elif isinstance(x, Basic):
        return x
    # If x has a '__array__' attribute (likely a NumPy array or similar), convert to SymPy Matrix
    elif hasattr(x, '__array__'):
        a = x.__array__()
        # If the array has zero dimensions, convert its scalar value to SymPy
        if len(a.shape) == 0:
            return sympify(a)
        # Otherwise, convert the array to SymPy Matrix
        return Matrix(x)
    # If none of the above conditions match, return x unchanged
    return x


class MutableDenseMatrix(DenseMatrix, MutableRepMatrix):

    def simplify(self, **kwargs):
        """Applies simplify to the elements of a matrix in place.

        This is a shortcut for M.applyfunc(lambda x: simplify(x, ratio, measure))

        See Also
        ========

        sympy.simplify.simplify.simplify
        """
        # Import the simplify function from sympy.simplify.simplify module
        from sympy.simplify.simplify import simplify as _simplify
        # Iterate through the dictionary of keys/values in self.todok()
        for (i, j), element in self.todok().items():
            # Apply _simplify function to each element of the matrix in place
            self[i, j] = _simplify(element, **kwargs)


MutableMatrix = Matrix = MutableDenseMatrix

###########
# Numpy Utility Functions:
# list2numpy, matrix2numpy, symmarray
###########


def list2numpy(l, dtype=object):  # pragma: no cover
    """Converts Python list of SymPy expressions to a NumPy array.

    See Also
    ========

    matrix2numpy
    """
    # Import numpy's empty function
    from numpy import empty
    # Create an empty numpy array 'a' with length equal to the input list 'l' and specified dtype
    a = empty(len(l), dtype)
    # Iterate through the index and elements of list 'l'
    for i, s in enumerate(l):
        # Assign each element 's' of 'l' to corresponding index 'i' in numpy array 'a'
        a[i] = s
    # Return the populated numpy array 'a'
    return a


def matrix2numpy(m, dtype=object):  # pragma: no cover
    """Converts SymPy's matrix to a NumPy array.

    See Also
    ========

    list2numpy
    """
    # Import numpy's empty function
    from numpy import empty
    # Create an empty numpy array 'a' with shape matching the dimensions of SymPy matrix 'm' and specified dtype
    a = empty(m.shape, dtype)
    # Iterate through the rows and columns of SymPy matrix 'm'
    for i in range(m.rows):
        for j in range(m.cols):
            # Assign each element 'm[i, j]' to corresponding index '(i, j)' in numpy array 'a'
            a[i, j] = m[i, j]
    # Return the populated numpy array 'a'
    return a


###########
# Rotation matrices:
# rot_givens, rot_axis[123], rot_ccw_axis[123]
###########


def rot_givens(i, j, theta, dim=3):
    r"""Returns a a Givens rotation matrix, a a rotation in the
    plane spanned by two coordinates axes.

    Explanation
    ===========

    The Givens rotation corresponds to a generalization of rotation
    matrices to any number of dimensions, given by:

    .. math::
        G(i, j, \theta) =
            \begin{bmatrix}
                1   & \cdots &    0   & \cdots &    0   & \cdots &    0   \\
                \vdots & \ddots & \vdots &        & \vdots &        & \vdots \\
                0   & \cdots &    c   & \cdots &   -s   & \cdots &    0   \\
                \vdots &        & \vdots & \ddots & \vdots &        & \vdots \\
                0   & \cdots &    s   & \cdots &    c   & \cdots &    0   \\
                \vdots &        & \vdots &        & \vdots & \ddots & \vdots \\
                0   & \cdots &    0   & \cdots &    0   & \cdots &    1
            \end{bmatrix}

    Where $c = \cos(\theta)$ and $s = \sin(\theta)$ appear at the intersections
    ``i``\th and ``j``\th rows and columns.

    For fixed ``i > j``\, the non-zero elements of a Givens matrix are
    given by:

    - $g_{kk} = 1$ for $k \ne i,\,j$
    - $g_{kk} = c$ for $k = i,\,j$
    - $g_{ji} = -g_{ij} = -s$

    Parameters
    ==========

    """
    # Placeholder function for a Givens rotation matrix
    pass
    def rot_givens(i, j, theta, dim=3):
        """
        Generate a Givens rotation matrix for given indices and angle.
    
        Parameters
        ==========
        i : int between 0 and dim - 1
            Represents the first axis.
        j : int between 0 and dim - 1
            Represents the second axis.
        theta : symbolic or numeric angle
            Angle of rotation in radians.
        dim : int bigger than 1
            Number of dimensions. Defaults to 3.
    
        Raises
        ======
        ValueError
            If dim is not an integer greater than one.
            If i equals j, indicating the same axis for rotation.
            If i or j are not integers between 0 and dim - 1.
    
        Returns
        =======
        M : sympy Matrix
            Givens rotation matrix of size dim x dim.
    
        Examples
        ========
        A counterclockwise rotation of pi/3 (60 degrees) around
        the third axis (z-axis):
    
        >>> rot_givens(1, 0, pi/3)
        Matrix([
        [      1/2, -sqrt(3)/2, 0],
        [sqrt(3)/2,        1/2, 0],
        [        0,          0, 1]])
    
        If we rotate by pi/2 (90 degrees):
    
        >>> rot_givens(1, 0, pi/2)
        Matrix([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]])
    
        This can be generalized to any number
        of dimensions:
    
        >>> rot_givens(1, 0, pi/2, dim=4)
        Matrix([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]])
    
        References
        ==========
        [1] https://en.wikipedia.org/wiki/Givens_rotation
    
        See Also
        ========
        rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
            about the 1-axis (clockwise around the x axis)
        rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
            about the 2-axis (clockwise around the y axis)
        rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
            about the 3-axis (clockwise around the z axis)
        rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
            about the 1-axis (counterclockwise around the x axis)
        rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
            about the 2-axis (counterclockwise around the y axis)
        rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
            about the 3-axis (counterclockwise around the z axis)
        """
        # Check if dim is valid
        if not isinstance(dim, int) or dim < 2:
            raise ValueError('dim must be an integer bigger than one, '
                             'got {}.'.format(dim))
    
        # Check if i and j are different
        if i == j:
            raise ValueError('i and j must be different, '
                             'got ({}, {})'.format(i, j))
    
        # Validate i and j within the range [0, dim-1]
        for ij in [i, j]:
            if not isinstance(ij, int) or ij < 0 or ij > dim - 1:
                raise ValueError('i and j must be integers between 0 and '
                                 '{}, got i={} and j={}.'.format(dim-1, i, j))
    
        # Calculate cosine and sine of the rotation angle
        theta = sympify(theta)
        c = cos(theta)
        s = sin(theta)
    
        # Initialize the identity matrix of size dim x dim
        M = eye(dim)
    
        # Fill the rotation matrix M according to Givens rotation formula
        M[i, i] = c
        M[j, j] = c
        M[i, j] = s
        M[j, i] = -s
    
        return M
def rot_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &           0 &            0 \\
                0 &  \cos(\theta) & \sin(\theta) \\
                0 & -\sin(\theta) & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis1(theta)
    Matrix([
    [1,         0,          0],
    [0,       1/2, sqrt(3)/2],
    [0, -sqrt(3)/2,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis1(pi/2)
    Matrix([
    [1,  0, 0],
    [0,  0, 1],
    [0, -1, 0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(1, 2, theta, dim=3)
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &             0 &            0 \\
                0 &  \cos(\theta) & \sin(\theta) \\
                0 & -\sin(\theta) & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis1(theta)
    Matrix([
    [1,          0,         0],
    [0,        1/2, sqrt(3)/2],
    [0, -sqrt(3)/2,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis1(pi/2)
    Matrix([
    [1,  0, 0],
    [0,  0, 1],
    [0, -1, 0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    """
    # 调用 rot_givens 函数生成一个绕第一轴旋转 theta 弧度的旋转矩阵
    return rot_givens(1, 2, theta, dim=3)
# 定义一个函数，返回绕着第三轴旋转 theta 弧度（逆时针方向）的旋转矩阵

def rot_ccw_axis3(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                \cos(\theta) & -\sin(\theta) & 0 \\
                \sin(\theta) &  \cos(\theta) & 0 \\
                           0 &             0 & 1
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis3(theta)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis3(pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    """
    
    # 调用通用的 Givens 旋转矩阵函数，维度为 3，生成绕 z 轴逆时针旋转 theta 的矩阵
    return rot_givens(1, 0, theta, dim=3)


# 定义一个函数，返回绕着第二轴旋转 theta 弧度（逆时针方向）的旋转矩阵

def rot_ccw_axis2(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 2-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `y`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                 \cos(\theta) & 0 & \sin(\theta) \\
                            0 & 1 &            0 \\
                -\sin(\theta) & 0 & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis2

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis2(theta)
    Matrix([
    [       1/2, 0, sqrt(3)/2],
    [         0, 1,         0],
    [-sqrt(3)/2, 0,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis2(pi/2)
    Matrix([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    # 调用名为 rot_givens 的函数，并传入参数 0, 2, theta, dim=3，返回其计算结果
    return rot_givens(0, 2, theta, dim=3)
# 定义函数 rot_ccw_axis1，返回绕第1轴（x轴）旋转 theta 弧度的旋转矩阵
def rot_ccw_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========
    
    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &            0 &             0 \\
                0 & \cos(\theta) & -\sin(\theta) \\
                0 & \sin(\theta) &  \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis1(theta)
    Matrix([
    [1,         0,          0],
    [0,       1/2, -sqrt(3)/2],
    [0, sqrt(3)/2,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis1(pi/2)
    Matrix([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    # 调用 rot_givens 函数，返回维度为 3 的 theta 弧度的第 2 和第 1 轴的 Givens 旋转矩阵
    return rot_givens(2, 1, theta, dim=3)
    # 导入必要的库和模块
    from numpy import empty, ndindex
    # 定义一个函数，用于创建一个由符号对象组成的多维数组
    def symarray(prefix, shape, **kwargs):
        # 创建一个空的多维数组，每个元素是一个符号对象
        arr = empty(shape, dtype=object)
        # 使用 ndindex 遍历多维数组的每个索引
        for index in ndindex(shape):
            # 根据给定的前缀和索引创建符号对象，并将其存储在数组中
            arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))),
                                **kwargs)
        # 返回填充好符号对象的多维数组
        return arr
# 定义函数 casoratian，计算线性差分方程组的 Casoratian 行列式
def casoratian(seqs, n, zero=True):
    """Given linear difference operator L of order 'k' and homogeneous
       equation Ly = 0 we want to compute kernel of L, which is a set
       of 'k' sequences: a(n), b(n), ... z(n).

       Solutions of L are linearly independent iff their Casoratian,
       denoted as C(a, b, ..., z), do not vanish for n = 0.

       Casoratian is defined by k x k determinant::

                  +  a(n)     b(n)     . . . z(n)     +
                  |  a(n+1)   b(n+1)   . . . z(n+1)   |
                  |    .         .     .        .     |
                  |    .         .       .      .     |
                  |    .         .         .    .     |
                  +  a(n+k-1) b(n+k-1) . . . z(n+k-1) +

       It proves very useful in rsolve_hyper() where it is applied
       to a generating set of a recurrence to factor out linearly
       dependent solutions and return a basis:

       >>> from sympy import Symbol, casoratian, factorial
       >>> n = Symbol('n', integer=True)

       Exponential and factorial are linearly independent:

       >>> casoratian([2**n, factorial(n)], n) != 0
       True

    """

    # 将输入的序列列表转换为符号表达式列表
    seqs = list(map(sympify, seqs))

    # 定义函数 f，根据 zero 参数选择不同的偏移方式
    if not zero:
        f = lambda i, j: seqs[j].subs(n, n + i)
    else:
        f = lambda i, j: seqs[j].subs(n, i)

    # 获取序列的数量 k
    k = len(seqs)

    # 返回由 f 函数计算的 k x k 矩阵的行列式
    return Matrix(k, k, f).det()


# 定义函数 eye，创建一个 n x n 的单位矩阵
def eye(*args, **kwargs):
    """Create square identity matrix n x n

    See Also
    ========

    diag
    zeros
    ones
    """

    # 调用 Matrix.eye 函数创建单位矩阵并返回
    return Matrix.eye(*args, **kwargs)


# 定义函数 diag，返回以给定值为对角线元素的矩阵
def diag(*values, strict=True, unpack=False, **kwargs):
    """Returns a matrix with the provided values placed on the
    diagonal. If non-square matrices are included, they will
    produce a block-diagonal matrix.

    Examples
    ========

    This version of diag is a thin wrapper to Matrix.diag that differs
    in that it treats all lists like matrices -- even when a single list
    is given. If this is not desired, either put a `*` before the list or
    set `unpack=True`.

    >>> from sympy import diag

    >>> diag([1, 2, 3], unpack=True)  # = diag(1,2,3) or diag(*[1,2,3])
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    >>> diag([1, 2, 3])  # a column vector
    Matrix([
    [1],
    [2],
    [3]])

    See Also
    ========
    .matrixbase.MatrixBase.eye
    .matrixbase.MatrixBase.diagonal
    .matrixbase.MatrixBase.diag
    .expressions.blockmatrix.BlockMatrix
    """
    # 调用 Matrix.diag 函数创建并返回对角矩阵
    return Matrix.diag(*values, strict=strict, unpack=unpack, **kwargs)


# 定义函数 GramSchmidt，对一组向量应用 Gram-Schmidt 过程
def GramSchmidt(vlist, orthonormal=False):
    """Apply the Gram-Schmidt process to a set of vectors.

    Parameters
    ==========

    vlist : List of Matrix
        Vectors to be orthogonalized for.

    orthonormal : Bool, optional
        If true, return an orthonormal basis.

    Returns
    =======

    vlist : List of Matrix
        Orthogonalized vectors

    Notes
    =====
    """
    # 返回经过 Gram-Schmidt 过程处理后的向量列表
    return vlist  # 返回原始向量列表，Gram-Schmidt 过程的实现未给出
    This routine is mostly duplicate from ``Matrix.orthogonalize``,
    except for some difference that this always raises error when
    linearly dependent vectors are found, and the keyword ``normalize``
    has been named as ``orthonormal`` in this function.

    See Also
    ========

    .matrixbase.MatrixBase.orthogonalize

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    使用 MutableDenseMatrix 类的 orthogonalize 方法对向量列表进行正交化处理，
    *vlist 是可变长度参数，传递给 orthogonalize 方法。
    关键字参数 normalize 被重命名为 orthonormal，并设置为 True，表示需要正交化。
    同时进行秩检查（rankcheck=True），以检测线性相关的向量并抛出错误。

    See Also
    ========

    .matrixbase.MatrixBase.orthogonalize
    参见 .matrixbase.MatrixBase.orthogonalize 方法的详细说明。

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    参考 Gram-Schmidt 过程的维基百科页面。
    """
    返回 MutableDenseMatrix.orthogonalize 方法的调用结果，即处理后的正交化向量。
# 计算给定函数 f 关于变量列表 varlist 的 Hessian 矩阵
def hessian(f, varlist, constraints=()):
    """Compute Hessian matrix for a function f wrt parameters in varlist
    which may be given as a sequence or a row/column vector. A list of
    constraints may optionally be given.

    Examples
    ========

    >>> from sympy import Function, hessian, pprint
    >>> from sympy.abc import x, y
    >>> f = Function('f')(x, y)
    >>> g1 = Function('g')(x, y)
    >>> g2 = x**2 + 3*y
    >>> pprint(hessian(f, (x, y), [g1, g2]))
    [                   d               d            ]
    [     0        0    --(g(x, y))     --(g(x, y))  ]
    [                   dx              dy           ]
    [                                                ]
    [     0        0        2*x              3       ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))  2*x   ---(f(x, y))   -----(f(x, y))]
    [dx                   2            dy dx         ]
    [                   dx                           ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))   3   -----(f(x, y))   ---(f(x, y)) ]
    [dy                dy dx              2          ]
    [                                   dy           ]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hessian_matrix

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    wronskian
    """
    # 如果 varlist 是 MatrixBase 类型的对象，则将其转换为列表形式
    if isinstance(varlist, MatrixBase):
        # 如果 varlist 不是一维的，则抛出形状错误异常
        if 1 not in varlist.shape:
            raise ShapeError("`varlist` must be a column or row vector.")
        # 如果 varlist 是列向量，则转置为行向量
        if varlist.cols == 1:
            varlist = varlist.T
        # 将 MatrixBase 对象转换为列表
        varlist = varlist.tolist()[0]
    # 如果 varlist 是可迭代对象，则获取其长度并进行检查
    if is_sequence(varlist):
        n = len(varlist)
        # 如果 varlist 长度为零，则抛出形状错误异常
        if not n:
            raise ShapeError("`len(varlist)` must not be zero.")
    else:
        # 如果 varlist 不是可迭代对象，则抛出值错误异常
        raise ValueError("Improper variable list in hessian function")
    # 如果函数 f 没有差分属性，则抛出值错误异常，指出 f 不可微分
    if not getattr(f, 'diff'):
        raise ValueError("Function `f` (%s) is not differentiable" % f)
    # 计算约束条件的数量 m
    m = len(constraints)
    # 总变量数量 N
    N = m + n
    # 初始化一个零矩阵 out，形状为 N x N
    out = zeros(N)
    # 遍历约束条件
    for k, g in enumerate(constraints):
        # 如果约束条件 g 没有差分属性，则抛出值错误异常，指出 g 不可微分
        if not getattr(g, 'diff'):
            raise ValueError("Function `f` (%s) is not differentiable" % f)
        # 对于每个变量进行偏导数计算，填充 out 矩阵的对应位置
        for i in range(n):
            out[k, i + m] = g.diff(varlist[i])
    # 计算函数 f 对每对变量的二阶偏导数，填充 out 矩阵的对应位置
    for i in range(n):
        for j in range(i, n):
            out[i + m, j + m] = f.diff(varlist[i]).diff(varlist[j])
    # 将 out 矩阵转换为对称矩阵
    for i in range(N):
        for j in range(i + 1, N):
            out[j, i] = out[i, j]
    # 返回计算得到的 Hessian 矩阵
    return out


def jordan_cell(eigenval, n):
    """
    Create a Jordan block:

    Examples
    ========

    >>> from sympy import jordan_cell
    >>> from sympy.abc import x
    >>> jordan_cell(x, 4)
    # 调用 jordan_cell 函数，生成一个 Jordan 细胞矩阵，以 x 为特征值，大小为 4x4
    Matrix([
    [x, 1, 0, 0],   # 第一行：对角线上为 x，其余为 1 的矩阵
    [0, x, 1, 0],   # 第二行：对角线上为 x，其余为 1 的矩阵
    [0, 0, x, 1],   # 第三行：对角线上为 x，其余为 1 的矩阵
    [0, 0, 0, x]])  # 第四行：对角线上为 x，其余为 1 的矩阵
    """

    return Matrix.jordan_block(size=n, eigenvalue=eigenval)
    # 调用 Matrix 类的 jordan_block 方法，生成一个大小为 n 的 Jordan 块矩阵，特征值为 eigenval，并返回该矩阵
# 返回矩阵 A 和 B 的哈达玛积（元素对应相乘）。
def matrix_multiply_elementwise(A, B):
    """Return the Hadamard product (elementwise product) of A and B

    >>> from sympy import Matrix, matrix_multiply_elementwise
    >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
    >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
    >>> matrix_multiply_elementwise(A, B)
    Matrix([
    [  0, 10, 200],
    [300, 40,   5]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.__mul__
    """
    return A.multiply_elementwise(B)


# 返回一个指定大小的全1矩阵。如果省略 cols 参数，则返回一个方阵。
def ones(*args, **kwargs):
    """Returns a matrix of ones with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    zeros
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    return Matrix.ones(*args, **kwargs)


# 创建一个随机矩阵，其维度为 r 行 c 列。如果省略 c，则返回一个方阵。
# 如果 symmetric 为 True，则矩阵必须是方阵。
# 如果 percent 小于 100，则只有大约给定百分比的元素将非零。
# 用于生成矩阵的伪随机数生成器根据提供的 prng 参数选择。
def randMatrix(r, c=None, min=0, max=99, seed=None, symmetric=False,
               percent=100, prng=None):
    """Create random matrix with dimensions ``r`` x ``c``. If ``c`` is omitted
    the matrix will be square. If ``symmetric`` is True the matrix must be
    square. If ``percent`` is less than 100 then only approximately the given
    percentage of elements will be non-zero.

    The pseudo-random number generator used to generate matrix is chosen in the
    following way.

    * If ``prng`` is supplied, it will be used as random number generator.
      It should be an instance of ``random.Random``, or at least have
      ``randint`` and ``shuffle`` methods with same signatures.
    * if ``prng`` is not supplied but ``seed`` is supplied, then new
      ``random.Random`` with given ``seed`` will be created;
    * otherwise, a new ``random.Random`` with default seed will be used.

    Examples
    ========

    >>> from sympy import randMatrix
    >>> randMatrix(3) # doctest:+SKIP
    [25, 45, 27]
    [44, 54,  9]
    [23, 96, 46]
    >>> randMatrix(3, 2) # doctest:+SKIP
    [87, 29]
    [23, 37]
    [90, 26]
    >>> randMatrix(3, 3, 0, 2) # doctest:+SKIP
    [0, 2, 0]
    [2, 0, 1]
    [0, 0, 1]
    >>> randMatrix(3, symmetric=True) # doctest:+SKIP
    [85, 26, 29]
    [26, 71, 43]
    [29, 43, 57]
    >>> A = randMatrix(3, seed=1)
    >>> B = randMatrix(3, seed=2)
    >>> A == B
    False
    >>> A == randMatrix(3, seed=1)
    True
    >>> randMatrix(3, symmetric=True, percent=50) # doctest:+SKIP
    [77, 70,  0],
    [70,  0,  0],
    [ 0,  0, 88]
    """
    # 如果 prng 未提供，则使用给定的 seed 创建一个新的随机数生成器
    prng = prng or random.Random(seed)

    if c is None:
        c = r

    # 如果 symmetric 为 True 并且 r 不等于 c，则引发 ValueError
    if symmetric and r != c:
        raise ValueError('For symmetric matrices, r must equal c, but %i != %i' % (r, c))

    # 创建一个包含 r 行 c 列的全零矩阵
    ij = range(r * c)

    # 如果 percent 不等于 100，则从 ij 中随机采样，使得仅约定百分比的元素非零
    if percent != 100:
        ij = prng.sample(ij, int(len(ij)*percent // 100))

    # 创建一个大小为 r 行 c 列的零矩阵
    m = zeros(r, c)

    # 如果 symmetric 为 False，则为矩阵填充随机值
    if not symmetric:
        for ijk in ij:
            i, j = divmod(ijk, c)
            m[i, j] = prng.randint(min, max)
    else:
        # 如果不满足上述条件，则执行以下代码块
        for ijk in ij:
            # 将 ijk 按照 c 进行整除和取余，分别赋值给 i 和 j
            i, j = divmod(ijk, c)
            # 如果 i 小于等于 j，则执行以下操作
            if i <= j:
                # 在 m[i, j] 和 m[j, i] 处填入随机整数，范围在 min 到 max 之间
                m[i, j] = m[j, i] = prng.randint(min, max)

    # 返回填充完毕的矩阵 m
    return m
# 定义一个函数用于计算给定函数列表的 Wronskian 行列式
def wronskian(functions, var, method='bareiss'):
    """
    Compute Wronskian for [] of functions

    计算函数列表的 Wronskian 行列式

    ::

                         | f1       f2        ...   fn      |
                         | f1'      f2'       ...   fn'     |
                         |  .        .        .      .      |
        W(f1, ..., fn) = |  .        .         .     .      |
                         |  .        .          .    .      |
                         |  (n)      (n)            (n)     |
                         | D   (f1) D   (f2)  ...  D   (fn) |

    see: https://en.wikipedia.org/wiki/Wronskian

    查看更多详细信息：https://en.wikipedia.org/wiki/Wronskian

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    hessian
    """

    # 将函数列表中的每个函数符号化
    functions = [sympify(f) for f in functions]
    # 获取函数列表的长度
    n = len(functions)
    # 如果函数列表为空，返回单位矩阵
    if n == 0:
        return S.One
    # 创建一个 n x n 的矩阵 W，其元素为各函数对给定变量 var 的导数
    W = Matrix(n, n, lambda i, j: functions[i].diff(var, j))
    # 返回计算得到的 Wronskian 行列式，使用指定的方法（默认为 'bareiss'）
    return W.det(method)


# 定义一个函数用于生成指定维度的零矩阵
def zeros(*args, **kwargs):
    """Returns a matrix of zeros with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    返回一个具有指定行数和列数的零矩阵；
    如果省略了 ``cols`` 参数，则返回一个方阵。

    See Also
    ========

    ones
    eye
    diag
    """

    # 如果 kwargs 中包含 'c' 键，则将其替换为 'cols' 键
    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    # 调用 SymPy 中 Matrix 类的 zeros 方法生成零矩阵并返回
    return Matrix.zeros(*args, **kwargs)
```