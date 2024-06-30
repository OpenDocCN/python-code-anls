# `D:\src\scipysrc\scipy\scipy\stats\_covariance.py`

```
# 导入 functools 模块中的 cached_property 装饰器
from functools import cached_property

# 导入 numpy 库，并使用别名 np
import numpy as np

# 从 scipy 库中导入 linalg 模块
from scipy import linalg

# 从 scipy.stats 中导入 _multivariate 模块
from scipy.stats import _multivariate

# 定义了一个公开可用的类列表，只包含字符串 "Covariance"
__all__ = ["Covariance"]

# 定义了一个 Covariance 类，表示协方差矩阵的表示
class Covariance:
    """
    Representation of a covariance matrix

    Calculations involving covariance matrices (e.g. data whitening,
    multivariate normal function evaluation) are often performed more
    efficiently using a decomposition of the covariance matrix instead of the
    covariance matrix itself. This class allows the user to construct an
    object representing a covariance matrix using any of several
    decompositions and perform calculations using a common interface.

    .. note::

        The `Covariance` class cannot be instantiated directly. Instead, use
        one of the factory methods (e.g. `Covariance.from_diagonal`).

    Examples
    --------
    The `Covariance` class is used by calling one of its
    factory methods to create a `Covariance` object, then pass that
    representation of the `Covariance` matrix as a shape parameter of a
    multivariate distribution.

    For instance, the multivariate normal distribution can accept an array
    representing a covariance matrix:

    >>> from scipy import stats
    >>> import numpy as np
    >>> d = [1, 2, 3]
    >>> A = np.diag(d)  # a diagonal covariance matrix
    >>> x = [4, -2, 5]  # a point of interest
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=A)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    but the calculations are performed in a very generic way that does not
    take advantage of any special properties of the covariance matrix. Because
    our covariance matrix is diagonal, we can use ``Covariance.from_diagonal``
    to create an object representing the covariance matrix, and
    `multivariate_normal` can use this to compute the probability density
    function more efficiently.

    >>> cov = stats.Covariance.from_diagonal(d)
    >>> dist = stats.multivariate_normal(mean=[0, 0, 0], cov=cov)
    >>> dist.pdf(x)
    4.9595685102808205e-08

    """
    # 构造函数，抛出一个未实现错误，并显示给定消息
    def __init__(self):
        message = ("The `Covariance` class cannot be instantiated directly. "
                   "Please use one of the factory methods "
                   "(e.g. `Covariance.from_diagonal`).")
        raise NotImplementedError(message)

    # 静态方法声明，用于创建对象的工厂方法
    @staticmethod
    def from_diagonal(diagonal):
        r"""
        Return a representation of a covariance matrix from its diagonal.

        Parameters
        ----------
        diagonal : array_like
            The diagonal elements of a diagonal matrix.

        Notes
        -----
        Let the diagonal elements of a diagonal covariance matrix :math:`D` be
        stored in the vector :math:`d`.

        When all elements of :math:`d` are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x \cdot d^{-1/2}`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{D}` is calculated as :math:`-2 \sum(\log{d})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive elements of :math:`d` are
        ignored. Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive elements of :math:`d` as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = np.diag(rng.random(n))
        >>> x = rng.random(size=n)

        Extract the diagonal from ``A`` and create the `Covariance` object.

        >>> d = np.diag(A)
        >>> cov = stats.Covariance.from_diagonal(d)

        Compare the functionality of the `Covariance` object against a
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = np.diag(d**-0.5) @ x
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        # 返回一个使用对角元素创建的 CovViaDiagonal 对象
        return CovViaDiagonal(diagonal)

    @staticmethod
    # 使用 precision 矩阵生成一个协方差的表示形式

    def from_precision(precision, covariance=None):
        r"""
        Return a representation of a covariance from its precision matrix.

        Parameters
        ----------
        precision : array_like
            The precision matrix; that is, the inverse of a square, symmetric,
            positive definite covariance matrix.
        covariance : array_like, optional
            The square, symmetric, positive definite covariance matrix. If not
            provided, this may need to be calculated (e.g. to evaluate the
            cumulative distribution function of
            `scipy.stats.multivariate_normal`) by inverting `precision`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, its precision matrix be
        :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such
        that :math:`L L^T = P`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`x^T L`. :math:`\log\det{A}` is calculated as
        :math:`-2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the precision matrix does not exist for a singular covariance
        matrix.

        Examples
        --------
        Prepare a symmetric positive definite precision matrix ``P`` and a
        data point ``x``. (If the precision matrix is not already available,
        consider the other factory methods of the `Covariance` class.)

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> P = rng.random(size=(n, n))
        >>> P = P @ P.T  # a precision matrix must be positive definite
        >>> x = rng.random(size=n)

        Create the `Covariance` object.

        >>> cov = stats.Covariance.from_precision(P)

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ np.linalg.cholesky(P)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = -np.linalg.slogdet(P)[-1]
        >>> np.allclose(res, ref)
        True

        """
        # 返回一个使用 precision 和 covariance 创建的 CovViaPrecision 对象
        return CovViaPrecision(precision, covariance)

    @staticmethod
    def from_cholesky(cholesky):
        r"""
        Representation of a covariance provided via the (lower) Cholesky factor

        Parameters
        ----------
        cholesky : array_like
            The lower triangular Cholesky factor of the covariance matrix.

        Notes
        -----
        Let the covariance matrix be :math:`A` and :math:`L` be the lower
        Cholesky factor such that :math:`L L^T = A`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`L^{-1} x`. :math:`\log\det{A}` is calculated as
        :math:`2tr(\log{L})`, where the :math:`\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the Cholesky decomposition does not exist for a singular
        covariance matrix.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the Cholesky decomposition of ``A`` and create the
        `Covariance` object.

        >>> L = np.linalg.cholesky(A)
        >>> cov = stats.Covariance.from_cholesky(L)

        Compare the functionality of the `Covariance` object against
        reference implementation.

        >>> from scipy.linalg import solve_triangular
        >>> res = cov.whiten(x)
        >>> ref = solve_triangular(L, x, lower=True)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        return CovViaCholesky(cholesky)
    # 定义一个方法，用于根据特征分解表示协方差矩阵
    def from_eigendecomposition(eigendecomposition):
        """
        Representation of a covariance provided via eigendecomposition

        Parameters
        ----------
        eigendecomposition : sequence
            A sequence (nominally a tuple) containing the eigenvalue and
            eigenvector arrays as computed by `scipy.linalg.eigh` or
            `numpy.linalg.eigh`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, let :math:`V` be matrix of
        eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues
        such that `V W V^T = A`.

        When all of the eigenvalues are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken
        element-wise.
        :math:`\log\det{A}` is calculated as  :math:`tr(\log{W})`,
        where the :math:`\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive eigenvalues are ignored.
        Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive eigenvalues as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the eigendecomposition of ``A`` and create the `Covariance`
        object.

        >>> w, v = np.linalg.eigh(A)
        >>> cov = stats.Covariance.from_eigendecomposition((w, v))

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ (v @ np.diag(w**-0.5))
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
        # 返回一个通过特征分解构造的 CovViaEigendecomposition 对象
        return CovViaEigendecomposition(eigendecomposition)
    def whiten(self, x):
        """
        Perform a whitening transformation on data.

        "Whitening" ("white" as in "white noise", in which each frequency has
        equal magnitude) transforms a set of random variables into a new set of
        random variables with unit-diagonal covariance. When a whitening
        transform is applied to a sample of points distributed according to
        a multivariate normal distribution with zero mean, the covariance of
        the transformed sample is approximately the identity matrix.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> precision = np.linalg.inv(cov_array)
        >>> cov_object = stats.Covariance.from_precision(precision)
        >>> x = rng.multivariate_normal(np.zeros(n), cov_array, size=(10000))
        >>> x_ = cov_object.whiten(x)
        >>> np.cov(x_, rowvar=False)  # near-identity covariance
        array([[0.97862122, 0.00893147, 0.02430451],
               [0.00893147, 0.96719062, 0.02201312],
               [0.02430451, 0.02201312, 0.99206881]])

        """
        # 调用内部方法 _whiten 对输入数据进行白化处理并返回结果
        return self._whiten(np.asarray(x))
    def colorize(self, x):
        """
        Perform a colorizing transformation on data.

        "Colorizing" ("color" as in "colored noise", in which different
        frequencies may have different magnitudes) transforms a set of
        uncorrelated random variables into a new set of random variables with
        the desired covariance. When a coloring transform is applied to a
        sample of points distributed according to a multivariate normal
        distribution with identity covariance and zero mean, the covariance of
        the transformed sample is approximately the covariance matrix used
        in the coloring transform.

        Parameters
        ----------
        x : array_like
            An array of points. The last dimension must correspond with the
            dimensionality of the space, i.e., the number of columns in the
            covariance matrix.

        Returns
        -------
        x_ : array_like
            The transformed array of points.

        References
        ----------
        .. [1] "Whitening Transformation". Wikipedia.
               https://en.wikipedia.org/wiki/Whitening_transformation
        .. [2] Novak, Lukas, and Miroslav Vorechovsky. "Generalization of
               coloring linear transformation". Transactions of VSB 18.2
               (2018): 31-35. :doi:`10.31490/tces-2018-0013`

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng(1638083107694713882823079058616272161)
        >>> n = 3
        >>> A = rng.random(size=(n, n))
        >>> cov_array = A @ A.T  # make matrix symmetric positive definite
        >>> cholesky = np.linalg.cholesky(cov_array)
        >>> cov_object = stats.Covariance.from_cholesky(cholesky)
        >>> x = rng.multivariate_normal(np.zeros(n), np.eye(n), size=(10000))
        >>> x_ = cov_object.colorize(x)
        >>> cov_data = np.cov(x_, rowvar=False)
        >>> np.allclose(cov_data, cov_array, rtol=3e-2)
        True
        """
        # 调用内部方法 _colorize 处理输入数据 x，并返回结果
        return self._colorize(np.asarray(x))

    @property
    def log_pdet(self):
        """
        Log of the pseudo-determinant of the covariance matrix
        """
        # 返回私有属性 _log_pdet 的值，强制转换为浮点数类型
        return np.array(self._log_pdet, dtype=float)[()]

    @property
    def rank(self):
        """
        Rank of the covariance matrix
        """
        # 返回私有属性 _rank 的值，强制转换为整数类型
        return np.array(self._rank, dtype=int)[()]

    @property
    def covariance(self):
        """
        Explicit representation of the covariance matrix
        """
        # 返回私有属性 _covariance 的值
        return self._covariance

    @property
    def shape(self):
        """
        Shape of the covariance array
        """
        # 返回私有属性 _shape 的值
        return self._shape
    # 对输入的矩阵进行验证，确保其为二维数组，并且是实数类型（整数或浮点数）
    def _validate_matrix(self, A, name):
        # 将输入数组至少转换为二维数组
        A = np.atleast_2d(A)
        # 获取数组的行数 m 和列数 n
        m, n = A.shape[-2:]
        # 检查是否为方阵，且确保数组是二维的，并且元素类型是整数或浮点数
        if m != n or A.ndim != 2 or not (np.issubdtype(A.dtype, np.integer) or
                                         np.issubdtype(A.dtype, np.floating)):
            # 若验证失败，则抛出 ValueError 异常
            message = (f"The input `{name}` must be a square, "
                       "two-dimensional array of real numbers.")
            raise ValueError(message)
        # 返回验证后的数组 A
        return A

    # 对输入的向量进行验证，确保其为一维数组，并且是实数类型（整数或浮点数）
    def _validate_vector(self, A, name):
        # 将输入数组至少转换为一维数组
        A = np.atleast_1d(A)
        # 检查数组是否为一维的，并且元素类型是整数或浮点数
        if A.ndim != 1 or not (np.issubdtype(A.dtype, np.integer) or
                               np.issubdtype(A.dtype, np.floating)):
            # 若验证失败，则抛出 ValueError 异常
            message = (f"The input `{name}` must be a one-dimensional array "
                       "of real numbers.")
            raise ValueError(message)
        # 返回验证后的数组 A
        return A
class CovViaPrecision(Covariance):

    def __init__(self, precision, covariance=None):
        # Validate precision matrix input
        precision = self._validate_matrix(precision, 'precision')
        # Check and validate covariance matrix if provided
        if covariance is not None:
            covariance = self._validate_matrix(covariance, 'covariance')
            # Ensure shapes of precision and covariance matrices match
            message = "`precision.shape` must equal `covariance.shape`."
            if precision.shape != covariance.shape:
                raise ValueError(message)

        # Compute Cholesky decomposition of precision matrix
        self._chol_P = np.linalg.cholesky(precision)
        # Compute log determinant of precision matrix
        self._log_pdet = -2*np.log(np.diag(self._chol_P)).sum(axis=-1)
        # Determine the rank of the precision matrix
        self._rank = precision.shape[-1]  # must be full rank if invertible
        # Store precision and covariance matrices
        self._precision = precision
        self._cov_matrix = covariance
        # Store shape of precision matrix
        self._shape = precision.shape
        # Flag indicating whether singular matrices are allowed
        self._allow_singular = False

    def _whiten(self, x):
        # Whitening transformation using Cholesky factor of precision matrix
        return x @ self._chol_P

    @cached_property
    def _covariance(self):
        # Compute covariance matrix using the Cholesky factor of precision matrix
        n = self._shape[-1]
        return (linalg.cho_solve((self._chol_P, True), np.eye(n))
                if self._cov_matrix is None else self._cov_matrix)

    def _colorize(self, x):
        # Colorization transformation using the transpose of Cholesky factor
        return linalg.solve_triangular(self._chol_P.T, x.T, lower=False).T


def _dot_diag(x, d):
    # Element-wise multiplication of matrix `x` with diagonal matrix `d`
    # Handles special cases where `d` is an n-dimensional diagonal matrix
    return x * d if x.ndim < 2 else x * np.expand_dims(d, -2)


class CovViaDiagonal(Covariance):

    def __init__(self, diagonal):
        # Validate diagonal vector input
        diagonal = self._validate_vector(diagonal, 'diagonal')

        # Identify and handle elements of diagonal that are <= 0
        i_zero = diagonal <= 0
        positive_diagonal = np.array(diagonal, dtype=np.float64)

        # Compute log determinant based on positive diagonal elements
        positive_diagonal[i_zero] = 1  # ones don't affect determinant
        self._log_pdet = np.sum(np.log(positive_diagonal), axis=-1)

        # Compute pseudo-reciprocals for whitening
        psuedo_reciprocals = 1 / np.sqrt(positive_diagonal)
        psuedo_reciprocals[i_zero] = 0

        # Store square root of diagonal for colorization
        self._sqrt_diagonal = np.sqrt(diagonal)
        # Store pseudo-reciprocals for whitening
        self._LP = psuedo_reciprocals
        # Determine rank based on positive diagonal elements
        self._rank = positive_diagonal.shape[-1] - i_zero.sum(axis=-1)
        # Create covariance matrix using diagonal elements
        self._covariance = np.apply_along_axis(np.diag, -1, diagonal)
        # Store locations where diagonal elements are zero or negative
        self._i_zero = i_zero
        # Store shape of covariance matrix
        self._shape = self._covariance.shape
        # Flag indicating whether singular matrices are allowed
        self._allow_singular = True

    def _whiten(self, x):
        # Whitening transformation using pseudo-reciprocals
        return _dot_diag(x, self._LP)

    def _colorize(self, x):
        # Colorization transformation using square root of diagonal
        return _dot_diag(x, self._sqrt_diagonal)

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        # Determine support mask based on locations of zero elements in diagonal
        return ~np.any(_dot_diag(x, self._i_zero), axis=-1)


class CovViaCholesky(Covariance):

    def __init__(self, cholesky):
        # Validate Cholesky factor input
        L = self._validate_matrix(cholesky, 'cholesky')

        # Store Cholesky factor
        self._factor = L
        # Compute log determinant using Cholesky factor
        self._log_pdet = 2*np.log(np.diag(self._factor)).sum(axis=-1)
        # Determine the rank of Cholesky factor
        self._rank = L.shape[-1]  # must be full rank for cholesky
        # Store shape of Cholesky factor
        self._shape = L.shape
        # Flag indicating whether singular matrices are allowed
        self._allow_singular = False

    @cached_property
    # 计算因子矩阵的协方差矩阵
    def _covariance(self):
        return self._factor @ self._factor.T
    
    # 对输入数据进行白化处理
    def _whiten(self, x):
        # 使用三角解法求解线性方程组，对输入数据进行白化处理
        res = linalg.solve_triangular(self._factor, x.T, lower=True).T
        return res
    
    # 对输入数据进行着色处理
    def _colorize(self, x):
        # 将输入数据与因子矩阵的转置相乘，进行着色处理
        return x @ self._factor.T
class CovViaEigendecomposition(Covariance):
    # 继承自 Covariance 类，表示通过特征分解获得协方差信息的类

    def __init__(self, eigendecomposition):
        # 初始化方法，接受特征分解作为参数

        eigenvalues, eigenvectors = eigendecomposition
        # 从特征分解元组中提取特征值和特征向量

        eigenvalues = self._validate_vector(eigenvalues, 'eigenvalues')
        # 调用内部方法验证特征值的向量格式

        eigenvectors = self._validate_matrix(eigenvectors, 'eigenvectors')
        # 调用内部方法验证特征向量的矩阵格式

        message = ("The shapes of `eigenvalues` and `eigenvectors` "
                   "must be compatible.")
        # 定义异常信息，用于错误处理

        try:
            eigenvalues = np.expand_dims(eigenvalues, -2)
            # 在特征值的倒数第二个维度上扩展维度

            eigenvectors, eigenvalues = np.broadcast_arrays(eigenvectors,
                                                            eigenvalues)
            # 使用广播操作，使特征向量和特征值的维度兼容

            eigenvalues = eigenvalues[..., 0, :]
            # 选择特征值的第一个维度作为有效维度
        except ValueError:
            raise ValueError(message)
        # 捕获值异常，抛出自定义错误信息

        i_zero = eigenvalues <= 0
        # 生成一个布尔数组，标记非正特征值的位置

        positive_eigenvalues = np.array(eigenvalues, dtype=np.float64)
        # 将特征值转换为浮点数类型的数组

        positive_eigenvalues[i_zero] = 1  # ones don't affect determinant
        # 将非正特征值替换为1，因为这些特征值不影响行列式的计算

        self._log_pdet = np.sum(np.log(positive_eigenvalues), axis=-1)
        # 计算正特征值的对数，并在最后一个维度上求和，存储在实例变量中

        psuedo_reciprocals = 1 / np.sqrt(positive_eigenvalues)
        # 计算正特征值的平方根的倒数

        psuedo_reciprocals[i_zero] = 0
        # 将非正特征值对应的倒数设置为0，避免除零错误

        self._LP = eigenvectors * psuedo_reciprocals
        # 计算白化矩阵 LP，为特征向量乘以正特征值的平方根的倒数

        self._LA = eigenvectors * np.sqrt(eigenvalues)
        # 计算染色矩阵 LA，为特征向量乘以特征值的平方根

        self._rank = positive_eigenvalues.shape[-1] - i_zero.sum(axis=-1)
        # 计算秩，排除非正特征值的数量

        self._w = eigenvalues
        # 存储特征值

        self._v = eigenvectors
        # 存储特征向量

        self._shape = eigenvectors.shape
        # 存储特征向量的形状信息

        self._null_basis = eigenvectors * i_zero
        # 计算零空间基，即特征向量乘以非正特征值的位置

        # This is only used for `_support_mask`, not to decide whether
        # the covariance is singular or not.
        self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10**3
        # 计算一个小量 eps，用于支持掩码函数而不是判断协方差是否奇异

        self._allow_singular = True
        # 设置允许奇异协方差

    def _whiten(self, x):
        # 白化操作，对输入数据 x 进行白化处理
        return x @ self._LP

    def _colorize(self, x):
        # 染色操作，对输入数据 x 进行染色处理
        return x @ self._LA.T

    @cached_property
    def _covariance(self):
        # 缓存属性，计算并返回协方差矩阵
        return (self._v * self._w) @ self._v.T

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        # 支持掩码函数，检查向量 x 是否在分布的支持域内部
        residual = np.linalg.norm(x @ self._null_basis, axis=-1)
        in_support = residual < self._eps
        return in_support


class CovViaPSD(Covariance):
    # 继承自 Covariance 类，表示通过正定对称矩阵 (PSD) 提供的协方差信息的类

    def __init__(self, psd):
        # 初始化方法，接受一个正定对称矩阵 (PSD) 实例作为参数

        self._LP = psd.U
        # 从 PSD 对象中获取白化矩阵 U

        self._log_pdet = psd.log_pdet
        # 从 PSD 对象中获取对数行列式值

        self._rank = psd.rank
        # 从 PSD 对象中获取秩信息

        self._covariance = psd._M
        # 从 PSD 对象中获取协方差矩阵信息

        self._shape = psd._M.shape
        # 存储协方差矩阵的形状信息

        self._psd = psd
        # 存储 PSD 对象的引用

        self._allow_singular = False  # by default
        # 默认情况下不允许奇异协方差

    def _whiten(self, x):
        # 白化操作，对输入数据 x 进行白化处理
        return x @ self._LP

    def _support_mask(self, x):
        # 支持掩码函数，调用 PSD 对象的支持掩码函数检查向量 x 是否在支持域内部
        return self._psd._support_mask(x)
```