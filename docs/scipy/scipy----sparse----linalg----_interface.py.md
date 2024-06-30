# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_interface.py`

```
# 导入警告模块
import warnings

# 导入科学计算库numpy
import numpy as np

# 导入稀疏矩阵相关的函数和类
from scipy.sparse import issparse
from scipy.sparse._sputils import isshape, isintlike, asmatrix, is_pydata_spmatrix

# 定义模块中对外公开的类名列表
__all__ = ['LinearOperator', 'aslinearoperator']

# 定义线性操作符类
class LinearOperator:
    """用于执行矩阵向量乘法的通用接口类

    许多迭代方法（例如cg、gmres）在解线性系统A*x=b时不需要知道矩阵的各个条目。
    这些求解器只需要计算矩阵向量乘积A*v，其中v是一个密集向量。
    这个类作为迭代求解器和类似矩阵的对象之间的抽象接口。

    要构建一个具体的LinearOperator，可以将适当的可调用对象传递给此类的构造函数，或者对其进行子类化。

    子类必须实现以下方法之一：_matvec或者_matmat，并且必须实现属性/属性shape（一对整数）和dtype（可以为None）。
    可以调用此类上的__init__来验证这些属性。实现

    """
    
    def __init__(self, dtype=None, shape=None):
        # 初始化方法，用于设定dtype和shape属性
        pass

    def _matvec(self, x):
        # _matvec方法，用于定义此线性操作符如何与向量x相乘
        return np.repeat(x.sum(), self.shape[0])

# 导入线性操作符的函数
def aslinearoperator(A):
    pass

# 示例：定义一个线性操作符类，模拟np.ones(shape)但使用恒定的存储量
class Ones(LinearOperator):
    def __init__(self, shape):
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        return np.repeat(x.sum(), self.shape[0])

# 示例：创建一个稀疏矩阵，存储偏移量，并将其与Ones操作符相加
offsets = csr_matrix([[1, 0, 2], [0, -1, 0], [0, 0, 3]])
A = aslinearoperator(offsets) + Ones(offsets.shape)

# 示例：对新矩阵A与向量[1, 2, 3]进行矩阵向量乘法
A.dot([1, 2, 3])

# 示例：使用密集表示计算A与向量[1, 2, 3]的乘积，结果与稀疏矩阵相同
(np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])
    """
    _matvec automatically implements _matmat (using a naive
    algorithm) and vice-versa.

    Optionally, a subclass may implement _rmatvec or _adjoint
    to implement the Hermitian adjoint (conjugate transpose). As with
    _matvec and _matmat, implementing either _rmatvec or
    _adjoint implements the other automatically. Implementing
    _adjoint is preferable; _rmatvec is mostly there for
    backwards compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A * v.
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (M, K).

    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.

    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])

    """

    # Necessary for right matmul with numpy arrays.
    ndim = 2
    __array_ufunc__ = None
    def __new__(cls, *args, **kwargs):
        # 如果 cls 是 LinearOperator 类本身，则返回 _CustomLinearOperator 类的实例，
        # 作为 _CustomLinearOperator 的工厂方法。
        if cls is LinearOperator:
            return super().__new__(_CustomLinearOperator)
        else:
            # 否则，创建当前类的实例对象
            obj = super().__new__(cls)

            # 检查新创建对象的 _matvec 和 _matmat 方法是否与 LinearOperator 的相同，
            # 如果是，则发出运行时警告，要求子类至少实现其中之一。
            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                warnings.warn("LinearOperator subclass should implement"
                              " at least one of _matvec and _matmat.",
                              category=RuntimeWarning, stacklevel=2)

            return obj

    def __init__(self, dtype, shape):
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        # 如果 dtype 不为 None，则转换为 NumPy 的数据类型对象
        if dtype is not None:
            dtype = np.dtype(dtype)

        # 将 shape 转换为元组形式
        shape = tuple(shape)
        # 检查 shape 是否为有效的二维形状，否则引发 ValueError 异常
        if not isshape(shape):
            raise ValueError(f"invalid shape {shape!r} (must be 2-d)")

        # 将 dtype 和 shape 赋值给对象的实例变量
        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Called from subclasses at the end of the __init__ routine.
        """
        # 如果 dtype 为 None，则通过 matvec 方法计算出一个零向量的数据类型，并赋值给 dtype
        if self.dtype is None:
            v = np.zeros(self.shape[-1])
            self.dtype = np.asarray(self.matvec(v)).dtype

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """

        # 对输入矩阵 X 的每一列调用 matvec 方法，然后将结果在水平方向上堆叠起来
        return np.hstack([self.matvec(col.reshape(-1,1)) for col in X.T])

    def _matvec(self, x):
        """Default matrix-vector multiplication handler.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        # 如果 self 是一个形状为 (M, N) 的线性操作对象，则该方法将在形状为 (N,) 或 (N, 1) 的 ndarray 上调用，
        # 并应返回一个形状为 (M,) 或 (M, 1) 的 ndarray。
        # 此默认实现会回退到 _matmat 方法，因此定义 _matmat 方法也将定义矩阵-向量乘法。
        return self.matmat(x.reshape(-1, 1))
    def matvec(self, x):
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)  # 将输入向量 x 转换为 NumPy 数组

        M,N = self.shape  # 获取 LinearOperator 对象的形状 M 和 N

        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')  # 如果 x 的形状与 N 不匹配，抛出异常

        y = self._matvec(x)  # 调用 _matvec 方法执行矩阵向量乘法操作，计算 y = A*x

        if isinstance(x, np.matrix):  # 如果 x 是 NumPy 矩阵类型
            y = asmatrix(y)  # 将 y 转换为矩阵
        else:
            y = np.asarray(y)  # 将 y 转换为 ndarray 类型

        if x.ndim == 1:  # 如果 x 是一维数组
            y = y.reshape(M)  # 将 y 重新调整为形状 (M,)
        elif x.ndim == 2:  # 如果 x 是二维数组
            y = y.reshape(M,1)  # 将 y 重新调整为形状 (M,1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')  # 如果 x 的维度不合法，抛出异常

        return y  # 返回计算结果 y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)  # 将输入向量 x 转换为 NumPy 数组

        M,N = self.shape  # 获取 LinearOperator 对象的形状 M 和 N

        if x.shape != (M,) and x.shape != (M,1):
            raise ValueError('dimension mismatch')  # 如果 x 的形状与 M 不匹配，抛出异常

        y = self._rmatvec(x)  # 调用 _rmatvec 方法执行共轭转置矩阵向量乘法操作，计算 y = A^H * x

        if isinstance(x, np.matrix):  # 如果 x 是 NumPy 矩阵类型
            y = asmatrix(y)  # 将 y 转换为矩阵
        else:
            y = np.asarray(y)  # 将 y 转换为 ndarray 类型

        if x.ndim == 1:  # 如果 x 是一维数组
            y = y.reshape(N)  # 将 y 重新调整为形状 (N,)
        elif x.ndim == 2:  # 如果 x 是二维数组
            y = y.reshape(N,1)  # 将 y 重新调整为形状 (N,1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')  # 如果 x 的维度不合法，抛出异常

        return y  # 返回计算结果 y

    def _rmatvec(self, x):
        """Default implementation of _rmatvec; defers to adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            # 如果 _adjoint 没有被重写，则防止无限递归，抛出未实现异常
            raise NotImplementedError
        else:
            return self.H.matvec(x)  # 否则，调用共轭转置操作 H.matvec(x) 返回结果
    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        # 如果输入的 X 不是稀疏矩阵或 PyData 稀疏矩阵，则将其转换为 ndarray
        if not (issparse(X) or is_pydata_spmatrix(X)):
            X = np.asanyarray(X)

        # 检查 X 的维度是否为 2
        if X.ndim != 2:
            raise ValueError(f'expected 2-d ndarray or matrix, not {X.ndim}-d')

        # 检查 X 的第一维度是否与 self 的第二维度相匹配
        if X.shape[0] != self.shape[1]:
            raise ValueError(f'dimension mismatch: {self.shape}, {X.shape}')

        try:
            # 调用 _matmat 方法计算结果 Y
            Y = self._matmat(X)
        except Exception as e:
            # 如果 X 是稀疏矩阵或 PyData 稀疏矩阵，抛出类型错误
            if issparse(X) or is_pydata_spmatrix(X):
                raise TypeError(
                    "Unable to multiply a LinearOperator with a sparse matrix."
                    " Wrap the matrix in aslinearoperator first."
                ) from e
            raise

        # 如果 Y 是 np.matrix 类型，则转换为 np.matrix
        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)

        return Y

    def rmatmat(self, X):
        """Adjoint matrix-matrix multiplication.

        Performs the operation y = A^H * x where A is an MxN linear
        operator and x is a column vector or 1-d array, or 2-d array.
        The default implementation defers to the adjoint.

        Parameters
        ----------
        X : {matrix, ndarray}
            A matrix or 2D array.

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or 2D array depending on the type of the input.

        Notes
        -----
        This rmatmat wraps the user-specified rmatmat routine.

        """
        # 如果输入的 X 不是稀疏矩阵或 PyData 稀疏矩阵，则将其转换为 ndarray
        if not (issparse(X) or is_pydata_spmatrix(X)):
            X = np.asanyarray(X)

        # 检查 X 的维度是否为 2
        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        # 检查 X 的第一维度是否与 self 的第一维度相匹配
        if X.shape[0] != self.shape[0]:
            raise ValueError(f'dimension mismatch: {self.shape}, {X.shape}')

        try:
            # 调用 _rmatmat 方法计算结果 Y
            Y = self._rmatmat(X)
        except Exception as e:
            # 如果 X 是稀疏矩阵或 PyData 稀疏矩阵，抛出类型错误
            if issparse(X) or is_pydata_spmatrix(X):
                raise TypeError(
                    "Unable to multiply a LinearOperator with a sparse matrix."
                    " Wrap the matrix in aslinearoperator() first."
                ) from e
            raise

        # 如果 Y 是 np.matrix 类型，则转换为 np.matrix
        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)

        return Y
    def _rmatmat(self, X):
        """Default implementation of _rmatmat defers to rmatvec or adjoint."""
        # 检查 LinearOperator 的子类是否使用默认的 adjoint 方法
        if type(self)._adjoint == LinearOperator._adjoint:
            # 如果是，则对 X 的每一列调用 rmatvec 方法，并将结果水平堆叠起来
            return np.hstack([self.rmatvec(col.reshape(-1, 1)) for col in X.T])
        else:
            # 否则，调用 H 的 matmat 方法处理 X
            return self.H.matmat(X)

    def __call__(self, x):
        # 重载 () 运算符，调用 __mul__ 方法
        return self * x

    def __mul__(self, x):
        # 重载 * 运算符，调用 dot 方法
        return self.dot(x)

    def __truediv__(self, other):
        # 重载 / 运算符，用于将 LinearOperator 对象除以标量
        if not np.isscalar(other):
            raise ValueError("Can only divide a linear operator by a scalar.")
        
        # 返回一个 _ScaledLinearOperator 对象，其作用是将 self 缩放为 1/other
        return _ScaledLinearOperator(self, 1.0 / other)

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        # 如果 x 是 LinearOperator 对象，则返回一个 _ProductLinearOperator 对象
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            # 如果 x 是标量，则返回一个 _ScaledLinearOperator 对象
            return _ScaledLinearOperator(self, x)
        else:
            # 将 x 转换为 numpy 数组（如果它不是稀疏矩阵）
            if not issparse(x) and not is_pydata_spmatrix(x):
                x = np.asarray(x)

            # 根据 x 的维度调用不同的方法
            if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
                return self.matvec(x)  # 调用 matvec 方法
            elif x.ndim == 2:
                return self.matmat(x)  # 调用 matmat 方法
            else:
                raise ValueError('expected 1-d or 2-d array or matrix, got %r'
                                 % x)

    def __matmul__(self, other):
        # 重载 @ 运算符，调用 __mul__ 方法
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        # 重载 @ 运算符的反向操作，调用 __rmul__ 方法
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        # 重载右乘运算符，如果 x 是标量，则返回一个 _ScaledLinearOperator 对象
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return self._rdot(x)  # 否则调用 _rdot 方法处理 x
    def __pow__(self, p):
        # 实现线性算子的乘幂操作

        if np.isscalar(p):
            # 如果 p 是标量（数值），返回一个乘幂线性算子对象
            return _PowerLinearOperator(self, p)
        else:
            # 如果 p 不是标量，返回 Not Implemented，表示不支持当前操作
            return NotImplemented

    def __add__(self, x):
        # 实现线性算子与另一个线性算子的加法操作

        if isinstance(x, LinearOperator):
            # 如果 x 是线性算子对象，返回一个表示两个算子相加的线性算子对象
            return _SumLinearOperator(self, x)
        else:
            # 如果 x 不是线性算子对象，返回 Not Implemented，表示不支持当前操作
            return NotImplemented

    def __neg__(self):
        # 实现线性算子的负操作，即乘以 -1

        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        # 实现线性算子与另一个线性算子的减法操作，等价于加上该算子的负数

        return self.__add__(-x)

    def __repr__(self):
        # 返回线性算子对象的字符串表示形式，包括其维度和数据类型信息

        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        """返回自身的共轭转置，也称为伴随算子。

        Returns
        -------
        A_H : LinearOperator
            自身的共轭转置。
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """返回该线性算子的转置。

        Returns
        -------
        T : LinearOperator
            表示该算子转置的线性算子对象。
        """
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """_adjoint 的默认实现，内部委托给 rmatvec。"""
        return _AdjointLinearOperator(self)
    # 定义一个方法 `_transpose`，用于返回一个 `_TransposedLinearOperator` 对象
    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        # 返回一个 `_TransposedLinearOperator` 对象，作用类似于对 `self` 执行转置操作
        return _TransposedLinearOperator(self)
class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super().__init__(dtype, shape)  # 调用父类的构造函数，初始化线性操作符的形状和数据类型

        self.args = ()  # 初始化参数元组为空

        self.__matvec_impl = matvec  # 设置矩阵向量乘法的实现函数
        self.__rmatvec_impl = rmatvec  # 设置反向矩阵向量乘法的实现函数
        self.__rmatmat_impl = rmatmat  # 设置反向矩阵乘法的实现函数
        self.__matmat_impl = matmat  # 设置矩阵乘法的实现函数

        self._init_dtype()  # 初始化数据类型

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)  # 如果定义了矩阵乘法实现函数，则调用它
        else:
            return super()._matmat(X)  # 否则调用父类的默认矩阵乘法实现函数

    def _matvec(self, x):
        return self.__matvec_impl(x)  # 调用矩阵向量乘法实现函数

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")  # 如果未定义反向矩阵向量乘法实现函数，则抛出异常
        return self.__rmatvec_impl(x)  # 否则调用反向矩阵向量乘法实现函数

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)  # 如果定义了反向矩阵乘法实现函数，则调用它
        else:
            return super()._rmatmat(X)  # 否则调用父类的默认反向矩阵乘法实现函数

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype)  # 返回该线性操作符的共轭转置操作符


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)  # 调用父类的构造函数，初始化共轭转置操作符的形状和数据类型
        self.A = A  # 存储原始线性操作符对象
        self.args = (A,)  # 存储操作符的参数元组

    def _matvec(self, x):
        return self.A._rmatvec(x)  # 调用原始操作符的反向矩阵向量乘法

    def _rmatvec(self, x):
        return self.A._matvec(x)  # 调用原始操作符的矩阵向量乘法

    def _matmat(self, x):
        return self.A._rmatmat(x)  # 调用原始操作符的反向矩阵乘法

    def _rmatmat(self, x):
        return self.A._matmat(x)  # 调用原始操作符的矩阵乘法


class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)  # 调用父类的构造函数，初始化转置操作符的形状和数据类型
        self.A = A  # 存储原始线性操作符对象
        self.args = (A,)  # 存储操作符的参数元组

    def _matvec(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self.A._rmatvec(np.conj(x)))  # 执行转置操作的矩阵向量乘法

    def _rmatvec(self, x):
        return np.conj(self.A._matvec(np.conj(x)))  # 执行转置操作的反向矩阵向量乘法

    def _matmat(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self.A._rmatmat(np.conj(x)))  # 执行转置操作的矩阵乘法

    def _rmatmat(self, x):
        return np.conj(self.A._matmat(np.conj(x)))  # 执行转置操作的反向矩阵乘法


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)  # 如果对象不为空且具有数据类型属性，则将其数据类型添加到列表中
    return np.result_type(*dtypes)  # 返回所有数据类型的统一结果类型


class _SumLinearOperator(LinearOperator):
    """Sum of arbitrary Linear Operators"""
    # 初始化函数，接受两个参数 A 和 B，确保它们都是 LinearOperator 类的实例
    def __init__(self, A, B):
        # 如果 A 或者 B 不是 LinearOperator 类的实例，则抛出数值错误异常
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        # 如果 A 和 B 的形状不匹配，则抛出数值错误异常，显示形状不匹配的具体信息
        if A.shape != B.shape:
            raise ValueError(f'cannot add {A} and {B}: shape mismatch')
        # 将 A 和 B 存储为成员变量 args 的元组形式
        self.args = (A, B)
        # 调用父类的初始化方法，确定线性操作符的数据类型并使用 A 的形状
        super().__init__(_get_dtype([A, B]), A.shape)

    # 定义矩阵向量乘法方法，返回两个 LinearOperator 对象对输入向量 x 的乘积之和
    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    # 定义右侧矩阵向量乘法方法，返回两个 LinearOperator 对象右乘输入向量 x 的乘积之和
    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    # 定义右侧矩阵乘法方法，返回两个 LinearOperator 对象右乘输入矩阵 x 的乘积之和
    def _rmatmat(self, x):
        return self.args[0].rmatmat(x) + self.args[1].rmatmat(x)

    # 定义矩阵乘法方法，返回两个 LinearOperator 对象乘以输入矩阵 x 的乘积之和
    def _matmat(self, x):
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    # 定义共轭转置方法，返回两个 LinearOperator 对象的共轭转置之和
    def _adjoint(self):
        # 分别获取存储在 args 元组中的 A 和 B
        A, B = self.args
        # 返回 A 和 B 的共轭转置之和
        return A.H + B.H
class _ProductLinearOperator(LinearOperator):
    # 自定义的线性操作符类，继承自LinearOperator

    def __init__(self, A, B):
        # 初始化方法，接受两个参数A和B

        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            # 检查A和B是否都是LinearOperator的实例，如果不是则抛出数值错误异常
            raise ValueError('both operands have to be a LinearOperator')

        if A.shape[1] != B.shape[0]:
            # 检查A的列数是否等于B的行数，如果不等则抛出形状不匹配的数值错误异常
            raise ValueError(f'cannot multiply {A} and {B}: shape mismatch')

        super().__init__(_get_dtype([A, B]),
                         (A.shape[0], B.shape[1]))
        # 调用父类的初始化方法，设置数据类型和形状（矩阵乘积的结果形状）
        self.args = (A, B)
        # 设置实例变量args为元组(A, B)，用于存储操作数A和B

    def _matvec(self, x):
        # 定义矩阵向量乘法方法，接受向量x作为参数
        return self.args[0].matvec(self.args[1].matvec(x))
        # 返回A乘以(B乘以x)的结果

    def _rmatvec(self, x):
        # 定义右矩阵向量乘法方法，接受向量x作为参数
        return self.args[1].rmatvec(self.args[0].rmatvec(x))
        # 返回B乘以(A乘以x)的结果

    def _rmatmat(self, x):
        # 定义右矩阵乘法方法，接受矩阵x作为参数
        return self.args[1].rmatmat(self.args[0].rmatmat(x))
        # 返回B乘以(A乘以x)的结果

    def _matmat(self, x):
        # 定义矩阵乘法方法，接受矩阵x作为参数
        return self.args[0].matmat(self.args[1].matmat(x))
        # 返回A乘以(B乘以x)的结果

    def _adjoint(self):
        # 定义伴随（共轭转置）方法
        A, B = self.args
        # 解构实例变量args为A和B
        return B.H * A.H
        # 返回B的共轭转置乘以A的共轭转置的结果


class _ScaledLinearOperator(LinearOperator):
    # 自定义的按比例缩放线性操作符类，继承自LinearOperator

    def __init__(self, A, alpha):
        # 初始化方法，接受线性操作符A和标量alpha作为参数

        if not isinstance(A, LinearOperator):
            # 检查A是否为LinearOperator的实例，如果不是则抛出数值错误异常
            raise ValueError('LinearOperator expected as A')

        if not np.isscalar(alpha):
            # 检查alpha是否为标量，如果不是则抛出数值错误异常
            raise ValueError('scalar expected as alpha')

        if isinstance(A, _ScaledLinearOperator):
            # 如果A是_ScaledLinearOperator的实例，则解构A并避免原地乘法以防意外更改原始比例因子
            A, alpha_original = A.args
            alpha = alpha * alpha_original
            # 使用新的alpha乘以原始alpha，避免意外变更原始比例因子

        dtype = _get_dtype([A], [type(alpha)])
        # 获取数据类型，考虑A和alpha的类型
        super().__init__(dtype, A.shape)
        # 调用父类的初始化方法，设置数据类型和形状
        self.args = (A, alpha)
        # 设置实例变量args为元组(A, alpha)，用于存储操作数A和比例因子alpha

    def _matvec(self, x):
        # 定义矩阵向量乘法方法，接受向量x作为参数
        return self.args[1] * self.args[0].matvec(x)
        # 返回alpha乘以A乘以x的结果

    def _rmatvec(self, x):
        # 定义右矩阵向量乘法方法，接受向量x作为参数
        return np.conj(self.args[1]) * self.args[0].rmatvec(x)
        # 返回alpha的共轭乘以A的右矩阵向量乘以x的结果

    def _rmatmat(self, x):
        # 定义右矩阵乘法方法，接受矩阵x作为参数
        return np.conj(self.args[1]) * self.args[0].rmatmat(x)
        # 返回alpha的共轭乘以A的右矩阵乘以x的结果

    def _matmat(self, x):
        # 定义矩阵乘法方法，接受矩阵x作为参数
        return self.args[1] * self.args[0].matmat(x)
        # 返回alpha乘以A乘以x的结果

    def _adjoint(self):
        # 定义伴随（共轭转置）方法
        A, alpha = self.args
        # 解构实例变量args为A和alpha
        return A.H * np.conj(alpha)
        # 返回A的共轭转置乘以alpha的共轭的结果


class _PowerLinearOperator(LinearOperator):
    # 自定义的幂次线性操作符类，继承自LinearOperator

    def __init__(self, A, p):
        # 初始化方法，接受线性操作符A和非负整数p作为参数

        if not isinstance(A, LinearOperator):
            # 检查A是否为LinearOperator的实例，如果不是则抛出数值错误异常
            raise ValueError('LinearOperator expected as A')

        if A.shape[0] != A.shape[1]:
            # 检查A是否为方阵，如果不是则抛出数值错误异常
            raise ValueError('square LinearOperator expected, got %r' % A)

        if not isintlike(p) or p < 0:
            # 检查p是否为非负整数，如果不是则抛出数值错误异常
            raise ValueError('non-negative integer expected as p')

        super().__init__(_get_dtype([A]), A.shape)
        # 调用父类的初始化方法，设置数据类型和形状
        self.args = (A, p)
        # 设置实例变量args为元组(A, p)，用于存储操作数A和幂次p

    def _power(self, fun, x):
        # 定义幂函数，接受函数fun和输入x作为参数
        res = np.array(x, copy=True)
        # 复制输入x生成结果数组res
        for i in range(self.args[1]):
            # 对于范围在0到args[1]的所有整数i
            res = fun(res)
            # 计算fun(res)并赋值给res
        return res
        # 返回结果res

    def _matvec(self, x):
        # 定义矩阵向量乘法方法，接受向量x作为参数
        return self._power(self.args[0].matvec, x)
        # 返回A的matvec函数应用于x的args[1]次幂的结果

    def _rmatvec(self, x):
        # 定义右矩阵向量乘法方法，接受向量x作为参数
        return self._power(self.args[0].rmatvec, x)
        # 返回A的rmatvec函数应用于x的args[1]次幂的结果

    def _rmatmat(self, x):
        # 定义右矩阵乘法方法，接受矩阵x作为参数
        return self._power(self.args[0].rmatmat, x)
        # 返回A的rmatmat函数应用于x的args[1]次幂的结果

    def _matmat(self, x):
        # 定义矩阵乘法方法，接
    # 定义一个方法 `_adjoint`，用于计算自身对象的共轭转置幂次
    def _adjoint(self):
        # 从对象的参数中获取矩阵 A 和整数 p
        A, p = self.args
        # 返回矩阵 A 的共轭转置的 p 次幂
        return A.H ** p
class MatrixLinearOperator(LinearOperator):
    # MatrixLinearOperator 类，继承自 LinearOperator
    def __init__(self, A):
        # 构造函数，初始化对象
        super().__init__(A.dtype, A.shape)
        self.A = A  # 设置成员变量 A，表示线性操作的矩阵
        self.__adj = None  # 初始化私有变量 __adj 为 None
        self.args = (A,)  # 设置参数为 A 的元组

    def _matmat(self, X):
        # 矩阵乘法运算，返回 A 乘以 X 的结果
        return self.A.dot(X)

    def _adjoint(self):
        # 返回伴随操作的对象
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj

class _AdjointMatrixOperator(MatrixLinearOperator):
    # _AdjointMatrixOperator 类，继承自 MatrixLinearOperator
    def __init__(self, adjoint):
        # 构造函数，接受 adjoint 参数作为参数
        self.A = adjoint.A.T.conj()  # 设置 A 为 adjoint.A 的共轭转置
        self.__adjoint = adjoint  # 设置私有变量 __adjoint 为 adjoint
        self.args = (adjoint,)  # 设置参数为 adjoint 的元组
        self.shape = adjoint.shape[1], adjoint.shape[0]  # 设置形状为 adjoint 的转置形状

    @property
    def dtype(self):
        # 返回 __adjoint 的数据类型
        return self.__adjoint.dtype

    def _adjoint(self):
        # 返回 __adjoint 对象
        return self.__adjoint


class IdentityOperator(LinearOperator):
    # IdentityOperator 类，继承自 LinearOperator
    def __init__(self, shape, dtype=None):
        # 构造函数，初始化对象
        super().__init__(dtype, shape)

    def _matvec(self, x):
        # 矩阵向量乘法运算，返回输入向量 x
        return x

    def _rmatvec(self, x):
        # 右矩阵向量乘法运算，返回输入向量 x
        return x

    def _rmatmat(self, x):
        # 右矩阵矩阵乘法运算，返回输入矩阵 x
        return x

    def _matmat(self, x):
        # 矩阵矩阵乘法运算，返回输入矩阵 x
        return x

    def _adjoint(self):
        # 返回自身，作为伴随操作
        return self


def aslinearoperator(A):
    """Return A as a LinearOperator.

    'A' may be any of the following types:
     - ndarray
     - matrix
     - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
     - LinearOperator
     - An object with .shape and .matvec attributes

    See the LinearOperator documentation for additional information.

    Notes
    -----
    If 'A' has no .dtype attribute, the data type is determined by calling
    :func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
    call upon the linear operator creation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinearoperator(M)
    <2x3 MatrixLinearOperator with dtype=int32>
    """
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = np.atleast_2d(np.asarray(A))
        return MatrixLinearOperator(A)

    elif issparse(A) or is_pydata_spmatrix(A):
        return MatrixLinearOperator(A)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = None
            rmatmat = None
            dtype = None

            if hasattr(A, 'rmatvec'):
                rmatvec = A.rmatvec
            if hasattr(A, 'rmatmat'):
                rmatmat = A.rmatmat
            if hasattr(A, 'dtype'):
                dtype = A.dtype
            return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec,
                                  rmatmat=rmatmat, dtype=dtype)

        else:
            raise TypeError('type not understood')
```