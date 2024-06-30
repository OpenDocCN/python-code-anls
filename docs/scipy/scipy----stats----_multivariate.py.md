# `D:\src\scipysrc\scipy\scipy\stats\_multivariate.py`

```
# 作者信息
# Joris Vankerschaver 2013

# 导入所需的库
import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
                           ive, loggamma)
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar

# 暴露给外部的函数和类的列表
__all__ = ['multivariate_normal',
           'matrix_normal',
           'dirichlet',
           'dirichlet_multinomial',
           'wishart',
           'invwishart',
           'multinomial',
           'special_ortho_group',
           'ortho_group',
           'random_correlation',
           'unitary_group',
           'multivariate_t',
           'multivariate_hypergeom',
           'random_table',
           'uniform_direction',
           'vonmises_fisher']

# 常数定义
_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)

# 随机数种子的文档字符串
_doc_random_state = """\
seed : {None, int, np.random.RandomState, np.random.Generator}, optional
    Used for drawing random variates.
    If `seed` is `None`, the `~np.random.RandomState` singleton is used.
    If `seed` is an int, a new ``RandomState`` instance is used, seeded
    with seed.
    If `seed` is already a ``RandomState`` or ``Generator`` instance,
    then that object is used.
    Default is `None`.
"""


def _squeeze_output(out):
    """
    从数组中去除单维度条目并将其转换为标量，如果必要的话。
    """
    out = out.squeeze()  # 去除数组中的单维度条目
    if out.ndim == 0:
        out = out[()]  # 如果数组维度为零，则转换为标量
    return out


def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """确定给定频谱中哪些特征值“很小”。

    这是为了确保各种线性代数函数在判断一个Hermitian矩阵是否数值上奇异以及其数值矩阵秩时达成一致。
    这个函数设计用于与scipy.linalg.pinvh兼容。

    Parameters
    ----------
    spectrum : 1d ndarray
        Hermitian矩阵的特征值数组。
    cond, rcond : float, optional
        小特征值的截断。
        如果奇异值小于rcond * 最大特征值，则被视为零。
        如果为None或-1，则使用合适的机器精度。

    Returns
    -------
    eps : float
        数值上可忽略的量级截断值。

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


def _pinv_1d(v, eps=1e-5):
    """计算伪逆的辅助函数。

    Parameters
    ----------
    v : array_like
        输入向量。
    eps : float, optional
        截断误差，默认为1e-5。
    # 将输入的可迭代数v中的每个元素x进行处理，生成一个新的numpy数组
    # 如果x的绝对值小于等于eps，则返回0；否则返回1/x。这里使用了列表推导式。
    return np.array([0 if abs(x) <= eps else 1/x for x in v], dtype=float)
class _PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.

    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().

    Parameters
    ----------
    M : array_like
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)

    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().
    """

    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        # Convert the input matrix to a NumPy array
        self._M = np.asarray(M)

        # Compute the symmetric eigendecomposition using eigh
        # eigh handles array conversion, checks for finiteness, and ensures the matrix is square
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

        # Compute the threshold epsilon for small eigenvalues
        eps = _eigvalsh_to_eps(s, cond, rcond)

        # Check if the smallest eigenvalue is negative, which is not allowed for positive semidefinite matrices
        if np.min(s) < -eps:
            msg = "The input matrix must be symmetric positive semidefinite."
            raise ValueError(msg)

        # Filter eigenvalues smaller than eps and check for singularity if allow_singular is False
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            msg = ("When `allow_singular is False`, the input matrix must be "
                   "symmetric positive definite.")
            raise np.linalg.LinAlgError(msg)

        # Compute the pseudoinverse of the eigenvalues and adjust eigenvectors
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Store the tolerance epsilon scaled for use in comparisons
        self.eps = 1e3 * eps

        # Save eigenvectors corresponding to eigenvalues <= eps as V
        self.V = u[:, s <= eps]

        # Initialize precomputed attributes
        self.rank = len(d)  # Rank of the matrix
        self.U = U  # Adjusted eigenvectors
        self.log_pdet = np.sum(np.log(d))  # Logarithm of the pseudo-determinant

        # Placeholder for lazily computed attribute
        self._pinv = None
    # 定义一个方法 `_support_mask`，用于检查向量 x 是否位于分布的支持集内部
    def _support_mask(self, x):
        # 计算向量 x 与矩阵 self.V 的乘积后的 L2 范数
        residual = np.linalg.norm(x @ self.V, axis=-1)
        # 检查 residual 是否小于阈值 self.eps，返回布尔数组表示是否在支持集内
        in_support = residual < self.eps
        return in_support

    # 定义一个属性 `pinv`，返回矩阵 U 的伪逆
    @property
    def pinv(self):
        # 如果 `_pinv` 属性尚未计算，则计算 U 乘以其转置的结果作为 `_pinv` 的值
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        # 返回已计算好的 `_pinv` 值
        return self._pinv
class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.
    """
    
    def __init__(self, seed=None):
        super().__init__()  # 调用父类的初始化方法
        self._random_state = check_random_state(seed)  # 初始化随机数生成器的状态

    @property
    def random_state(self):
        """ Get or set the Generator object for generating random variates.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        """
        return self._random_state  # 返回当前随机数生成器的状态

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)  # 设置随机数生成器的状态

    def _get_random_state(self, random_state):
        if random_state is not None:
            return check_random_state(random_state)  # 如果传入了随机数生成器的状态，则返回检查后的状态
        else:
            return self._random_state  # 否则返回当前对象的随机数生成器的状态


class multi_rv_frozen:
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """
    
    @property
    def random_state(self):
        return self._dist._random_state  # 返回与分布相关联的随机数生成器的状态

    @random_state.setter
    def random_state(self, seed):
        self._dist._random_state = check_random_state(seed)  # 设置与分布相关联的随机数生成器的状态


_mvn_doc_default_callparams = """\
mean : array_like, default: ``[0]``
    Mean of the distribution.
cov : array_like or `Covariance`, default: ``[1]``
    Symmetric positive (semi)definite covariance matrix of the distribution.
allow_singular : bool, default: ``False``
    Whether to allow a singular covariance matrix. This is ignored if `cov` is
    a `Covariance` object.
"""

_mvn_doc_callparams_note = """\
Setting the parameter `mean` to `None` is equivalent to having `mean`
be the zero-vector. The parameter `cov` can be a scalar, in which case
the covariance matrix is the identity times that value, a vector of
diagonal entries for the covariance matrix, a two-dimensional array_like,
or a `Covariance` object.
"""

_mvn_doc_frozen_callparams = ""

_mvn_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

mvn_docdict_params = {
    '_mvn_doc_default_callparams': _mvn_doc_default_callparams,  # 多元正态分布的默认参数描述字典
    '_mvn_doc_callparams_note': _mvn_doc_callparams_note,  # 多元正态分布的参数注释说明字典
    '_doc_random_state': _doc_random_state  # 随机数生成器状态的文档描述字典
}

mvn_docdict_noparams = {
    '_mvn_doc_default_callparams': _mvn_doc_frozen_callparams,  # 冻结多元正态分布的默认参数描述字典（空）
    '_mvn_doc_callparams_note': _mvn_doc_frozen_callparams_note,  # 冻结多元正态分布的参数注释说明字典
    '_doc_random_state': _doc_random_state  # 随机数生成器状态的文档描述字典
}


class multivariate_normal_gen(multi_rv_generic):
    r"""A multivariate normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.

    Methods
    -------
    pdf(x, mean=None, cov=1, allow_singular=False)
        Probability density function.

    """
    def logpdf(x, mean=None, cov=1, allow_singular=False):
        """
        Calculate the logarithm of the probability density function (PDF) of
        the multivariate normal distribution.
    
        Parameters
        ----------
        x : array_like
            The quantile, at which to compute the log-pdf.
        mean : array_like, optional
            The mean vector of the distribution. Default is None (zero mean).
        cov : array_like, optional
            The covariance matrix of the distribution. Default is 1.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix `cov`. Default is False.
    
        Returns
        -------
        float
            The logarithm of the PDF evaluated at `x`.
    
        Notes
        -----
        The logarithm of the PDF for the multivariate normal distribution is
        computed using the formula:
    
        .. math::
    
            \log f(x) = -\frac{k}{2} \log(2 \pi) - \frac{1}{2} \log|\Sigma|
                        -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu),
    
        where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
        and :math:`k` the dimensionality of the distribution.
    
        The parameter `cov` can be a subclass of `Covariance`, in which case
        `allow_singular` is ignored. If `cov` is not a subclass, it must be
        symmetric positive semidefinite when `allow_singular` is True, and
        strictly positive definite when `allow_singular` is False.
        """
        pass
    
    
    def cdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5, lower_limit=None):
        """
        Calculate the cumulative distribution function (CDF) of the multivariate
        normal distribution.
    
        Parameters
        ----------
        x : array_like
            The quantile, at which to compute the CDF.
        mean : array_like, optional
            The mean vector of the distribution. Default is None (zero mean).
        cov : array_like, optional
            The covariance matrix of the distribution. Default is 1.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix `cov`. Default is False.
        maxpts : int, optional
            Maximum number of points to use for integration. Default is 1000000 * dim,
            where `dim` is the dimensionality of the distribution.
        abseps : float, optional
            Absolute error tolerance. Default is 1e-5.
        releps : float, optional
            Relative error tolerance. Default is 1e-5.
        lower_limit : float or array_like, optional
            Lower integration limit. Default is None.
    
        Returns
        -------
        float
            The value of the CDF evaluated at `x`.
        """
        pass
    
    
    def logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5):
        """
        Calculate the logarithm of the cumulative distribution function (CDF)
        of the multivariate normal distribution.
    
        Parameters
        ----------
        x : array_like
            The quantile, at which to compute the log-CDF.
        mean : array_like, optional
            The mean vector of the distribution. Default is None (zero mean).
        cov : array_like, optional
            The covariance matrix of the distribution. Default is 1.
        allow_singular : bool, optional
            Whether to allow a singular covariance matrix `cov`. Default is False.
        maxpts : int, optional
            Maximum number of points to use for integration. Default is 1000000 * dim,
            where `dim` is the dimensionality of the distribution.
        abseps : float, optional
            Absolute error tolerance. Default is 1e-5.
        releps : float, optional
            Relative error tolerance. Default is 1e-5.
    
        Returns
        -------
        float
            The logarithm of the value of the CDF evaluated at `x`.
        """
        pass
    
    
    def rvs(mean=None, cov=1, size=1, random_state=None):
        """
        Generate random samples from the multivariate normal distribution.
    
        Parameters
        ----------
        mean : array_like, optional
            The mean vector of the distribution. Default is None (zero mean).
        cov : array_like, optional
            The covariance matrix of the distribution. Default is 1.
        size : int or tuple of ints, optional
            Number of samples to draw. Default is 1.
        random_state : int or RandomState, optional
            Random seed or RandomState instance. Default is None.
    
        Returns
        -------
        ndarray
            Random samples from the multivariate normal distribution.
        """
        pass
    
    
    def entropy(mean=None, cov=1):
        """
        Compute the differential entropy of the multivariate normal distribution.
    
        Parameters
        ----------
        mean : array_like, optional
            The mean vector of the distribution. Default is None (zero mean).
        cov : array_like, optional
            The covariance matrix of the distribution. Default is 1.
    
        Returns
        -------
        float
            The differential entropy of the multivariate normal distribution.
        """
        pass
    
    
    def fit(x, fix_mean=None, fix_cov=None):
        """
        Fit a multivariate normal distribution to data.
    
        Parameters
        ----------
        x : array_like
            Data array to fit the distribution to.
        fix_mean : array_like, optional
            Fixed mean vector for the distribution. Default is None (estimate from data).
        fix_cov : array_like, optional
            Fixed covariance matrix for the distribution. Default is None (estimate from data).
    
        Returns
        -------
        multivariate_normal_frozen
            A frozen object representing the fitted multivariate normal distribution.
            This object has methods for PDF, CDF, etc., with fixed mean and covariance.
        """
        pass
    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.dstack((x, y))
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    """  # noqa: E501

    # 初始化方法，设置种子参数（可选）
    def __init__(self, seed=None):
        # 调用父类的初始化方法
        super().__init__(seed)
        # 根据 mvn_docdict_params 格式化文档字符串
        self.__doc__ = doccer.docformat(self.__doc__, mvn_docdict_params)

    # 实例调用方法，创建一个冻结的多变量正态分布
    def __call__(self, mean=None, cov=1, allow_singular=False, seed=None):
        """Create a frozen multivariate normal distribution.

        See `multivariate_normal_frozen` for more information.
        """
        # 返回一个冻结的多变量正态分布对象
        return multivariate_normal_frozen(mean, cov,
                                          allow_singular=allow_singular,
                                          seed=seed)

    # 处理参数的内部方法，根据输入的 mean 和 cov 推断维度，确保它们是完整的向量或矩阵
    def _process_parameters(self, mean, cov, allow_singular=True):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.
        """
        # 如果 cov 是 Covariance 类的实例，则调用 _process_parameters_Covariance 处理
        if isinstance(cov, _covariance.Covariance):
            return self._process_parameters_Covariance(mean, cov)
        else:
            # 在引入 Covariance 类之前，multivariate_normal 接受普通数组作为 cov，并进行以下输入验证
            dim, mean, cov = self._process_parameters_psd(None, mean, cov)
            # 进行输入验证后，某些方法使用 _PSD 对象处理数组，并用它来执行计算
            # 为了避免在每个方法中使用分支语句依赖于 cov 是数组还是 Covariance 对象，
            # 我们总是使用 _PSD 处理数组，然后使用满足 Covariance 接口的包装器 CovViaPSD。
            psd = _PSD(cov, allow_singular=allow_singular)
            cov_object = _covariance.CovViaPSD(psd)
            return dim, mean, cov_object

    # 处理 Covariance 类型参数的内部方法
    def _process_parameters_Covariance(self, mean, cov):
        dim = cov.shape[-1]
        mean = np.array([0.]) if mean is None else mean
        message = (f"`cov` represents a covariance matrix in {dim} dimensions,"
                   f"and so `mean` must be broadcastable to shape {(dim,)}")
        try:
            mean = np.broadcast_to(mean, dim)
        except ValueError as e:
            raise ValueError(message) from e
        return dim, mean, cov
    # 处理 PSD 参数的方法，用于推断维度、检查并返回合适的均值和协方差数组
    def _process_parameters_psd(self, dim, mean, cov):
        # 尝试推断维度
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            # 如果维度不是标量，引发值错误异常
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be "
                                 "a scalar.")

        # 检查输入的大小并为均值和协方差返回完整的数组
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)

        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean = mean.reshape(1)
            cov = cov.reshape(1, 1)

        # 检查均值数组是否是一维的且长度为 dim
        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." %
                             dim)
        
        # 根据协方差数组的维度进行相应的处理
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(cov.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % cov.ndim)

        # 返回处理后的维度、均值和协方差
        return dim, mean, cov

    # 处理量化数列的方法，调整数组以使最后一个轴标记每个数据点的组件
    def _process_quantiles(self, x, dim):
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        # 返回调整后的量化数列数组
        return x
    # 计算多变量正态分布概率密度函数的对数值。

    # Parameters
    # ----------
    # x : ndarray
    #     要评估概率密度函数对数的点
    # mean : ndarray
    #     分布的均值
    # cov_object : Covariance
    #     表示协方差矩阵的对象

    # Notes
    # -----
    # 由于此函数不执行参数检查，不应直接调用；应使用 'logpdf' 方法。

    def _logpdf(self, x, mean, cov_object):
        log_det_cov, rank = cov_object.log_pdet, cov_object.rank
        dev = x - mean
        if dev.ndim > 1:
            log_det_cov = log_det_cov[..., np.newaxis]
            rank = rank[..., np.newaxis]
        maha = np.sum(np.square(cov_object.whiten(dev)), axis=-1)
        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    # 计算多变量正态分布概率密度函数的对数值。

    # Parameters
    # ----------
    # x : array_like
    #     量化值，其中 `x` 的最后一个轴表示组件。
    # %(_mvn_doc_default_callparams)s

    # Returns
    # -------
    # pdf : ndarray or scalar
    #     在 `x` 处评估的概率密度函数的对数值

    # Notes
    # -----
    # %(_mvn_doc_callparams_note)s

    def logpdf(self, x, mean=None, cov=1, allow_singular=False):
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = self._logpdf(x, mean, cov_object)
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x-mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    # 计算多变量正态分布概率密度函数。

    # Parameters
    # ----------
    # x : array_like
    #     量化值，其中 `x` 的最后一个轴表示组件。
    # %(_mvn_doc_default_callparams)s

    # Returns
    # -------
    # pdf : ndarray or scalar
    #     在 `x` 处评估的概率密度函数值

    # Notes
    # -----
    # %(_mvn_doc_callparams_note)s

    def pdf(self, x, mean=None, cov=1, allow_singular=False):
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        x = self._process_quantiles(x, dim)
        out = np.exp(self._logpdf(x, mean, cov_object))
        if np.any(cov_object.rank < dim):
            out_of_bounds = ~cov_object._support_mask(x-mean)
            out[out_of_bounds] = 0.0
        return _squeeze_output(out)
    def _cdf(self, x, mean, cov, maxpts, abseps, releps, lower_limit):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the cumulative distribution function.
        mean : ndarray
            Mean of the distribution
        cov : array_like
            Covariance matrix of the distribution
        maxpts : integer
            The maximum number of points to use for integration
        abseps : float
            Absolute error tolerance
        releps : float
            Relative error tolerance
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'cdf' instead.


        .. versionadded:: 1.0.0

        """
        # Initialize lower bound for integration; use negative infinity if `lower_limit` is None
        lower = (np.full(mean.shape, -np.inf)
                 if lower_limit is None else lower_limit)

        # Ensure that lower bounds are less than or equal to corresponding `x` elements
        # to fix issues in dimensions where lower bounds may be incorrectly higher than `x`.
        b, a = np.broadcast_arrays(x, lower)
        i_swap = b < a
        # Determine signs of resulting CDF values based on whether there were odd/even swaps
        signs = (-1)**(i_swap.sum(axis=-1))  # odd # of swaps -> negative
        a, b = a.copy(), b.copy()
        a[i_swap], b[i_swap] = b[i_swap], a[i_swap]
        n = x.shape[-1]
        # Concatenate lower and upper bounds for integration
        limits = np.concatenate((a, b), axis=-1)

        # mvnun expects 1-d arguments, so process points sequentially
        def func1d(limits):
            return _mvn.mvnun(limits[:n], limits[n:], mean, cov,
                              maxpts, abseps, releps)[0]

        # Evaluate the multivariate normal CDF for each set of limits, adjusting by signs
        out = np.apply_along_axis(func1d, -1, limits) * signs
        # Squeeze the output array to remove single-dimensional entries
        return _squeeze_output(out)
    def logcdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
               abseps=1e-5, releps=1e-5, *, lower_limit=None):
        """Log of the multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
            根据传入的参数设定默认的调用参数，具体内容未在此处展示
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default ``1000000*dim``)
            积分时使用的最大点数，默认为 ``1000000*dim``
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
            绝对误差容忍度（默认为 1e-5）
        releps : float, optional
            Relative error tolerance (default 1e-5)
            相对误差容忍度（默认为 1e-5）
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.
            累积分布函数的积分下限，默认为负无穷大。必须能够广播到 `x` 的形状。

        Returns
        -------
        cdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at `x`
            在 `x` 处评估的累积分布函数的对数值

        Notes
        -----
        %(_mvn_doc_callparams_note)s
            提示：根据调用参数的说明注意事项，具体内容未在此处展示

        .. versionadded:: 1.0.0
            版本新增功能：1.0.0 版本

        """
        params = self._process_parameters(mean, cov, allow_singular)
        # 处理参数，返回维度、均值和协方差对象
        dim, mean, cov_object = params
        cov = cov_object.covariance
        x = self._process_quantiles(x, dim)
        if not maxpts:
            maxpts = 1000000 * dim
        cdf = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        # the log of a negative real is complex, and cdf can be negative
        # if lower limit is greater than upper limit
        # 如果 cdf 小于 0，则其对数是复数；如果下限大于上限，则 cdf 可能为负数
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        out = np.log(cdf)
        return out
    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5, *, lower_limit=None):
        """Multivariate normal cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_mvn_doc_default_callparams)s
        maxpts : integer, optional
            The maximum number of points to use for integration
            (default ``1000000*dim``)
        abseps : float, optional
            Absolute error tolerance (default 1e-5)
        releps : float, optional
            Relative error tolerance (default 1e-5)
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        .. versionadded:: 1.0.0

        """
        # 处理输入参数，获取维度、均值和协方差对象
        params = self._process_parameters(mean, cov, allow_singular)
        dim, mean, cov_object = params
        # 获取协方差矩阵
        cov = cov_object.covariance
        # 处理输入的分位数，确保与维度匹配
        x = self._process_quantiles(x, dim)
        # 如果未指定最大点数，则默认为维度乘以1000000
        if not maxpts:
            maxpts = 1000000 * dim
        # 调用内部的累积分布函数计算
        out = self._cdf(x, mean, cov, maxpts, abseps, releps, lower_limit)
        return out

    def rvs(self, mean=None, cov=1, size=1, random_state=None):
        """Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        %(_mvn_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Notes
        -----
        %(_mvn_doc_callparams_note)s

        """
        # 处理输入参数，获取维度、均值和协方差对象
        dim, mean, cov_object = self._process_parameters(mean, cov)
        # 获取随机状态对象
        random_state = self._get_random_state(random_state)

        # 如果协方差对象是通过PSD对象获取的
        if isinstance(cov_object, _covariance.CovViaPSD):
            # 获取协方差矩阵
            cov = cov_object.covariance
            # 使用多元正态分布生成随机样本
            out = random_state.multivariate_normal(mean, cov, size)
            # 压缩输出的维度
            out = _squeeze_output(out)
        else:
            size = size or tuple()
            if not np.iterable(size):
                size = (size,)
            # 确定生成随机样本的形状
            shape = tuple(size) + (cov_object.shape[-1],)
            # 生成标准正态分布的随机样本
            x = random_state.normal(size=shape)
            # 对随机样本进行处理，生成符合多元正态分布的随机样本
            out = mean + cov_object.colorize(x)
        return out
    def entropy(self, mean=None, cov=1):
        """计算多变量正态分布的差分熵。

        Parameters
        ----------
        mean : array_like, optional
            正态分布的均值。如果未提供，则使用默认值。
        cov : array_like, optional
            正态分布的协方差矩阵或标量。默认为单位矩阵。

        Returns
        -------
        h : scalar
            多变量正态分布的差分熵

        Notes
        -----
        此方法计算多变量正态分布的差分熵，基于给定的均值和协方差矩阵（或标量）。

        """
        # 处理参数，获取维度、均值和协方差对象
        dim, mean, cov_object = self._process_parameters(mean, cov)
        # 计算差分熵的公式
        return 0.5 * (cov_object.rank * (_LOG_2PI + 1) + cov_object.log_pdet)
    def fit(self, x, fix_mean=None, fix_cov=None):
        """Fit a multivariate normal distribution to data.

        Parameters
        ----------
        x : ndarray (m, n)
            Data the distribution is fitted to. Must have two axes.
            The first axis of length `m` represents the number of vectors
            the distribution is fitted to. The second axis of length `n`
            determines the dimensionality of the fitted distribution.
        fix_mean : ndarray(n, )
            Fixed mean vector. Must have length `n`.
        fix_cov: ndarray (n, n)
            Fixed covariance matrix. Must have shape ``(n, n)``.

        Returns
        -------
        mean : ndarray (n, )
            Maximum likelihood estimate of the mean vector
        cov : ndarray (n, n)
            Maximum likelihood estimate of the covariance matrix

        """
        # input validation for data to be fitted
        # 将输入数据转换为 NumPy 数组
        x = np.asarray(x)
        # 检查输入数据的维度是否为二维
        if x.ndim != 2:
            raise ValueError("`x` must be two-dimensional.")

        # 获取输入数据的向量数和向量维度
        n_vectors, dim = x.shape

        # parameter estimation
        # 参数估计过程参考文献：https://home.ttic.edu/~shubhendu/Slides/Estimation.pdf

        # 处理固定均值的情况
        if fix_mean is not None:
            # input validation for `fix_mean`
            # 将 `fix_mean` 转换为至少一维的数组
            fix_mean = np.atleast_1d(fix_mean)
            # 检查 `fix_mean` 的形状是否与向量维度一致
            if fix_mean.shape != (dim, ):
                msg = ("`fix_mean` must be a one-dimensional array the same "
                       "length as the dimensionality of the vectors `x`.")
                raise ValueError(msg)
            mean = fix_mean
        else:
            # 计算数据的均值作为估计的均值向量
            mean = x.mean(axis=0)

        # 处理固定协方差矩阵的情况
        if fix_cov is not None:
            # input validation for `fix_cov`
            # 将 `fix_cov` 转换为至少二维的数组
            fix_cov = np.atleast_2d(fix_cov)
            # 检查 `fix_cov` 的形状是否为方阵且与向量维度一致
            if fix_cov.shape != (dim, dim):
                msg = ("`fix_cov` must be a two-dimensional square array "
                       "of same side length as the dimensionality of the "
                       "vectors `x`.")
                raise ValueError(msg)
            # 验证协方差矩阵是否为对称正半定
            # 参考 _PSD 的简化副本
            s, u = scipy.linalg.eigh(fix_cov, lower=True, check_finite=True)
            eps = _eigvalsh_to_eps(s)
            if np.min(s) < -eps:
                msg = "`fix_cov` must be symmetric positive semidefinite."
                raise ValueError(msg)
            cov = fix_cov
        else:
            # 计算数据的协方差矩阵作为估计的协方差矩阵
            centered_data = x - mean
            cov = centered_data.T @ centered_data / n_vectors
        
        # 返回估计的均值向量和协方差矩阵
        return mean, cov
# 创建一个冻结的多元正态分布类，继承自多变量随机变量冻结基类
class multivariate_normal_frozen(multi_rv_frozen):
    def __init__(self, mean=None, cov=1, allow_singular=False, seed=None,
                 maxpts=None, abseps=1e-5, releps=1e-5):
        """Create a frozen multivariate normal distribution.

        Parameters
        ----------
        mean : array_like, default: ``[0]``
            Mean of the distribution.
        cov : array_like, default: ``[1]``
            Symmetric positive (semi)definite covariance matrix of the
            distribution.
        allow_singular : bool, default: ``False``
            Whether to allow a singular covariance matrix.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        maxpts : integer, optional
            The maximum number of points to use for integration of the
            cumulative distribution function (default ``1000000*dim``)
        abseps : float, optional
            Absolute error tolerance for the cumulative distribution function
            (default 1e-5)
        releps : float, optional
            Relative error tolerance for the cumulative distribution function
            (default 1e-5)

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from scipy.stats import multivariate_normal
        >>> r = multivariate_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """ # numpy/numpydoc#87  # noqa: E501
        # 使用 multivariate_normal_gen 创建一个多元正态分布的实例
        self._dist = multivariate_normal_gen(seed)
        # 处理参数，得到维度、均值、协方差对象
        self.dim, self.mean, self.cov_object = (
            self._dist._process_parameters(mean, cov, allow_singular))
        # 是否允许奇异协方差矩阵
        self.allow_singular = allow_singular or self.cov_object._allow_singular
        # 如果没有设置 maxpts，则默认为 1000000 * 维度
        if not maxpts:
            maxpts = 1000000 * self.dim
        self.maxpts = maxpts
        self.abseps = abseps
        self.releps = releps

    @property
    def cov(self):
        # 返回协方差矩阵
        return self.cov_object.covariance

    def logpdf(self, x):
        # 处理输入量 x，调用 _logpdf 方法计算对数概率密度
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.cov_object)
        # 如果协方差矩阵的秩小于维度，则对超出边界的值设为负无穷
        if np.any(self.cov_object.rank < self.dim):
            out_of_bounds = ~self.cov_object._support_mask(x-self.mean)
            out[out_of_bounds] = -np.inf
        return _squeeze_output(out)

    def pdf(self, x):
        # 计算概率密度函数，返回指数值
        return np.exp(self.logpdf(x))
    # 计算对数累积分布函数 (log CDF) 的方法
    def logcdf(self, x, *, lower_limit=None):
        # 调用累积分布函数 (CDF)，计算给定 x 的累积分布函数值
        cdf = self.cdf(x, lower_limit=lower_limit)
        # 如果任何 cdf 值小于零，则将 cdf 转换为复数以避免取对数时出现负数
        # 这是因为对负实数取对数会得到复数结果，而累积分布函数可能为负数
        cdf = cdf + 0j if np.any(cdf < 0) else cdf
        # 计算对数累积分布函数值
        out = np.log(cdf)
        return out

    # 计算累积分布函数 (CDF) 的方法
    def cdf(self, x, *, lower_limit=None):
        # 处理输入的 x 值，确保其格式符合期望
        x = self._dist._process_quantiles(x, self.dim)
        # 调用底层的累积分布函数计算方法，返回累积分布函数值
        out = self._dist._cdf(x, self.mean, self.cov_object.covariance,
                              self.maxpts, self.abseps, self.releps,
                              lower_limit)
        # 对输出进行压缩处理，确保输出的维度符合预期
        return _squeeze_output(out)

    # 生成随机变量 (Random Variates) 的方法
    def rvs(self, size=1, random_state=None):
        # 调用底层分布对象的 rvs 方法，生成指定数量的随机变量
        return self._dist.rvs(self.mean, self.cov_object, size, random_state)

    # 计算多元正态分布的微分熵 (Differential Entropy) 的方法
    def entropy(self):
        """Computes the differential entropy of the multivariate normal.

        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution

        """
        # 获取协方差对象的对数行列式值和秩
        log_pdet = self.cov_object.log_pdet
        rank = self.cov_object.rank
        # 计算多元正态分布的微分熵
        return 0.5 * (rank * (_LOG_2PI + 1) + log_pdet)
# 使用循环遍历列表 ['logpdf', 'pdf', 'logcdf', 'cdf', 'rvs']
for name in ['logpdf', 'pdf', 'logcdf', 'cdf', 'rvs']:
    # 获取 multivariate_normal_gen 类中名为 name 的方法对象
    method = multivariate_normal_gen.__dict__[name]
    # 获取 multivariate_normal_frozen 类中名为 name 的方法对象
    method_frozen = multivariate_normal_frozen.__dict__[name]
    # 为 multivariate_normal_frozen 类中的方法添加文档字符串，使用 doccer.docformat 处理
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             mvn_docdict_noparams)
    # 为 multivariate_normal_gen 类中的方法添加文档字符串，使用 doccer.docformat 处理
    method.__doc__ = doccer.docformat(method.__doc__, mvn_docdict_params)

# 默认参数字符串，用于 matrix_normal_gen 类的文档字符串
_matnorm_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default: `None`)
rowcov : array_like, optional
    Among-row covariance matrix of the distribution (default: ``1``)
colcov : array_like, optional
    Among-column covariance matrix of the distribution (default: ``1``)
"""

# 调用参数说明字符串，用于 matrix_normal_gen 类的文档字符串
_matnorm_doc_callparams_note = """\
If `mean` is set to `None` then a matrix of zeros is used for the mean.
The dimensions of this matrix are inferred from the shape of `rowcov` and
`colcov`, if these are provided, or set to ``1`` if ambiguous.

`rowcov` and `colcov` can be two-dimensional array_likes specifying the
covariance matrices directly. Alternatively, a one-dimensional array will
be be interpreted as the entries of a diagonal matrix, and a scalar or
zero-dimensional array will be interpreted as this value times the
identity matrix.
"""

# 冻结参数说明字符串，用于 matrix_normal_gen 类的文档字符串
_matnorm_doc_frozen_callparams = ""

# 冻结参数说明备注字符串，用于 matrix_normal_gen 类的文档字符串
_matnorm_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

# 参数字典，用于包含 matrix_normal_gen 类文档字符串中的特定参数说明和备注
matnorm_docdict_params = {
    '_matnorm_doc_default_callparams': _matnorm_doc_default_callparams,
    '_matnorm_doc_callparams_note': _matnorm_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

# 无参数字典，用于包含 matrix_normal_gen 类文档字符串中冻结状态下的参数说明和备注
matnorm_docdict_noparams = {
    '_matnorm_doc_default_callparams': _matnorm_doc_frozen_callparams,
    '_matnorm_doc_callparams_note': _matnorm_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class matrix_normal_gen(multi_rv_generic):
    r"""A matrix normal random variable.

    The `mean` keyword specifies the mean. The `rowcov` keyword specifies the
    among-row covariance matrix. The 'colcov' keyword specifies the
    among-column covariance matrix.

    Methods
    -------
    pdf(X, mean=None, rowcov=1, colcov=1)
        Probability density function.
    logpdf(X, mean=None, rowcov=1, colcov=1)
        Log of the probability density function.
    rvs(mean=None, rowcov=1, colcov=1, size=1, random_state=None)
        Draw random samples.
    entropy(rowcol=1, colcov=1)
        Differential entropy.

    Parameters
    ----------
    %(_matnorm_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_matnorm_doc_callparams_note)s

    The covariance matrices specified by `rowcov` and `colcov` must be
    (symmetric) positive definite. If the samples in `X` are
    :math:`m \times n`, then `rowcov` must be :math:`m \times m` and
    """
    `colcov` must be :math:`n \times n`. `mean` must be the same shape as `X`.

    The probability density function for `matrix_normal` is

    .. math::

        f(X) = (2 \pi)^{-\frac{mn}{2}}|U|^{-\frac{n}{2}} |V|^{-\frac{m}{2}}
               \exp\left( -\frac{1}{2} \mathrm{Tr}\left[ U^{-1} (X-M) V^{-1}
               (X-M)^T \right] \right),

    where :math:`M` is the mean, :math:`U` the among-row covariance matrix,
    :math:`V` the among-column covariance matrix.

    The `allow_singular` behaviour of the `multivariate_normal`
    distribution is not currently supported. Covariance matrices must be
    full rank.

    The `matrix_normal` distribution is closely related to the
    `multivariate_normal` distribution. Specifically, :math:`\mathrm{Vec}(X)`
    (the vector formed by concatenating the columns  of :math:`X`) has a
    multivariate normal distribution with mean :math:`\mathrm{Vec}(M)`
    and covariance :math:`V \otimes U` (where :math:`\otimes` is the Kronecker
    product). Sampling and pdf evaluation are
    :math:`\mathcal{O}(m^3 + n^3 + m^2 n + m n^2)` for the matrix normal, but
    :math:`\mathcal{O}(m^3 n^3)` for the equivalent multivariate normal,
    making this equivalent form algorithmically inefficient.

    .. versionadded:: 0.17.0

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.stats import matrix_normal

    >>> M = np.arange(6).reshape(3,2); M
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> U = np.diag([1,2,3]); U
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    >>> V = 0.3*np.identity(2); V
    array([[ 0.3,  0. ],
           [ 0. ,  0.3]])
    >>> X = M + 0.1; X
    array([[ 0.1,  1.1],
           [ 2.1,  3.1],
           [ 4.1,  5.1]])
    >>> matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
    0.023410202050005054

    >>> # Equivalent multivariate normal
    >>> from scipy.stats import multivariate_normal
    >>> vectorised_X = X.T.flatten()
    >>> equiv_mean = M.T.flatten()
    >>> equiv_cov = np.kron(V,U)
    >>> multivariate_normal.pdf(vectorised_X, mean=equiv_mean, cov=equiv_cov)
    0.023410202050005054

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" matrix normal
    random variable:

    >>> rv = matrix_normal(mean=None, rowcov=1, colcov=1)
    >>> # Frozen object with the same methods but holding the given
    >>> # mean and covariance fixed.

    """

    def __init__(self, seed=None):
        # 调用父类的构造方法，传递种子参数
        super().__init__(seed)
        # 使用文档字符串格式化工具对当前对象的文档字符串进行格式化，应用matnorm_docdict_params字典
        self.__doc__ = doccer.docformat(self.__doc__, matnorm_docdict_params)

    def __call__(self, mean=None, rowcov=1, colcov=1, seed=None):
        """Create a frozen matrix normal distribution.

        See `matrix_normal_frozen` for more information.

        """
        # 返回一个冻结的矩阵正态分布对象，传递给定的均值、行协方差、列协方差以及种子参数
        return matrix_normal_frozen(mean, rowcov, colcov, seed=seed)
    def _process_parameters(self, mean, rowcov, colcov):
        """
        Infer dimensionality from mean or covariance matrices. Handle
        defaults. Ensure compatible dimensions.
        """

        # Process mean parameter
        if mean is not None:
            # Convert mean into a NumPy array of type float
            mean = np.asarray(mean, dtype=float)
            # Determine the shape of the mean array
            meanshape = mean.shape
            # Check if mean array is not two-dimensional
            if len(meanshape) != 2:
                raise ValueError("Array `mean` must be two dimensional.")
            # Check for any zero dimension in mean array
            if np.any(meanshape == 0):
                raise ValueError("Array `mean` has invalid shape.")

        # Process among-row covariance parameter
        rowcov = np.asarray(rowcov, dtype=float)
        # Handle scalar rowcov input
        if rowcov.ndim == 0:
            if mean is not None:
                rowcov = rowcov * np.identity(meanshape[0])
            else:
                rowcov = rowcov * np.identity(1)
        # Handle 1D rowcov input
        elif rowcov.ndim == 1:
            rowcov = np.diag(rowcov)
        # Check if rowcov is not a 2D array
        rowshape = rowcov.shape
        if len(rowshape) != 2:
            raise ValueError("`rowcov` must be a scalar or a 2D array.")
        # Check if rowcov is not square
        if rowshape[0] != rowshape[1]:
            raise ValueError("Array `rowcov` must be square.")
        # Check for zero dimension in rowcov
        if rowshape[0] == 0:
            raise ValueError("Array `rowcov` has invalid shape.")
        # Determine number of rows in rowcov
        numrows = rowshape[0]

        # Process among-column covariance parameter
        colcov = np.asarray(colcov, dtype=float)
        # Handle scalar colcov input
        if colcov.ndim == 0:
            if mean is not None:
                colcov = colcov * np.identity(meanshape[1])
            else:
                colcov = colcov * np.identity(1)
        # Handle 1D colcov input
        elif colcov.ndim == 1:
            colcov = np.diag(colcov)
        # Check if colcov is not a 2D array
        colshape = colcov.shape
        if len(colshape) != 2:
            raise ValueError("`colcov` must be a scalar or a 2D array.")
        # Check if colcov is not square
        if colshape[0] != colshape[1]:
            raise ValueError("Array `colcov` must be square.")
        # Check for zero dimension in colcov
        if colshape[0] == 0:
            raise ValueError("Array `colcov` has invalid shape.")
        # Determine number of columns in colcov
        numcols = colshape[0]

        # Ensure mean and covariances are compatible
        if mean is not None:
            # Check compatibility between mean and row covariance dimensions
            if meanshape[0] != numrows:
                raise ValueError("Arrays `mean` and `rowcov` must have the "
                                 "same number of rows.")
            # Check compatibility between mean and column covariance dimensions
            if meanshape[1] != numcols:
                raise ValueError("Arrays `mean` and `colcov` must have the "
                                 "same number of columns.")
        else:
            # If mean is None, create a zero matrix with dimensions matching rowcov and colcov
            mean = np.zeros((numrows, numcols))

        # Return dimensions tuple and processed parameters
        dims = (numrows, numcols)
        return dims, mean, rowcov, colcov
    # 调整量化数列，使得最后两个轴表示每个数据点的组成部分。
    def _process_quantiles(self, X, dims):
        """
        Adjust quantiles array so that last two axes labels the components of
        each data point.
        """
        # 将输入的X转换为numpy数组，数据类型为float
        X = np.asarray(X, dtype=float)
        # 如果X的维度为2，则增加一个新的维度
        if X.ndim == 2:
            X = X[np.newaxis, :]
        # 如果X的最后两个维度不等于给定的dims，则引发值错误
        if X.shape[-2:] != dims:
            raise ValueError("The shape of array `X` is not compatible "
                             "with the distribution parameters.")
        # 返回调整后的X数组
        return X

    # 对数矩阵正态概率密度函数的对数值。
    def _logpdf(self, dims, X, mean, row_prec_rt, log_det_rowcov,
                col_prec_rt, log_det_colcov):
        """Log of the matrix normal probability density function.

        Parameters
        ----------
        dims : tuple
            Dimensions of the matrix variates
        X : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution
        row_prec_rt : ndarray
            A decomposition such that np.dot(row_prec_rt, row_prec_rt.T)
            is the inverse of the among-row covariance matrix
        log_det_rowcov : float
            Logarithm of the determinant of the among-row covariance matrix
        col_prec_rt : ndarray
            A decomposition such that np.dot(col_prec_rt, col_prec_rt.T)
            is the inverse of the among-column covariance matrix
        log_det_colcov : float
            Logarithm of the determinant of the among-column covariance matrix

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # 获取矩阵的行数和列数
        numrows, numcols = dims
        # 将X减去均值，并调整维度，使得最后一个轴移到第一个位置
        roll_dev = np.moveaxis(X-mean, -1, 0)
        # 计算缩放后的偏差，使用列的精确平方根和行的精确平方根的乘积
        scale_dev = np.tensordot(col_prec_rt.T,
                                 np.dot(roll_dev, row_prec_rt), 1)
        # 计算马氏距离
        maha = np.sum(np.sum(np.square(scale_dev), axis=-1), axis=0)
        # 返回计算的对数概率密度函数的值
        return -0.5 * (numrows*numcols*_LOG_2PI + numcols*log_det_rowcov
                       + numrows*log_det_colcov + maha)

    # 矩阵正态概率密度函数的对数值。
    def logpdf(self, X, mean=None, rowcov=1, colcov=1):
        """Log of the matrix normal probability density function.

        Parameters
        ----------
        X : array_like
            Quantiles, with the last two axes of `X` denoting the components.
        %(_matnorm_doc_default_callparams)s

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at `X`

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        # 处理参数，包括维度、均值、行协方差、列协方差
        dims, mean, rowcov, colcov = self._process_parameters(mean, rowcov,
                                                              colcov)
        # 处理量化数列X，使得最后两个轴表示组成部分
        X = self._process_quantiles(X, dims)
        # 对行协方差和列协方差进行PSD（正定性）处理
        rowpsd = _PSD(rowcov, allow_singular=False)
        colpsd = _PSD(colcov, allow_singular=False)
        # 调用_logpdf函数计算对数概率密度函数的值
        out = self._logpdf(dims, X, mean, rowpsd.U, rowpsd.log_pdet, colpsd.U,
                           colpsd.log_pdet)
        # 压缩输出结果并返回
        return _squeeze_output(out)
    def pdf(self, X, mean=None, rowcov=1, colcov=1):
        """Matrix normal probability density function.

        Parameters
        ----------
        X : array_like
            Quantiles, with the last two axes of `X` denoting the components.
        %(_matnorm_doc_default_callparams)s
            定义了默认的调用参数，这些参数将在文档字符串的末尾展开

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `X`
            在 `X` 处评估的概率密度函数值

        Notes
        -----
        %(_matnorm_doc_callparams_note)s
            给出了一些函数调用的附加说明

        """
        # 返回矩阵正态分布的概率密度函数，使用 logpdf 方法计算并取指数
        return np.exp(self.logpdf(X, mean, rowcov, colcov))

    def rvs(self, mean=None, rowcov=1, colcov=1, size=1, random_state=None):
        """Draw random samples from a matrix normal distribution.

        Parameters
        ----------
        %(_matnorm_doc_default_callparams)s
            定义了默认的调用参数，这些参数将在文档字符串的末尾展开
        size : integer, optional
            Number of samples to draw (default 1).
            要抽取的样本数（默认为 1）
        %(_doc_random_state)s
            随机数生成器的说明，控制随机数生成的状态

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `dims`), where `dims` is the
            dimension of the random matrices.
            尺寸为 (`size`, `dims`) 的随机变量，其中 `dims` 是随机矩阵的维度

        Notes
        -----
        %(_matnorm_doc_callparams_note)s
            给出了一些函数调用的附加说明

        """
        # 将 size 转换为整数
        size = int(size)
        # 处理参数，返回维度、均值、行协方差和列协方差
        dims, mean, rowcov, colcov = self._process_parameters(mean, rowcov,
                                                              colcov)
        # 计算行协方差的 Cholesky 分解
        rowchol = scipy.linalg.cholesky(rowcov, lower=True)
        # 计算列协方差的 Cholesky 分解
        colchol = scipy.linalg.cholesky(colcov, lower=True)
        # 获取随机数生成器状态
        random_state = self._get_random_state(random_state)
        # 为了确保随机变量的向后兼容性，我们不直接生成尺寸为 (size, dims[0], dims[1]) 的标准正态变量
        # 参见 https://github.com/scipy/scipy/pull/12312 获取更多详细信息
        std_norm = random_state.standard_normal(
            size=(dims[1], size, dims[0])
        ).transpose(1, 2, 0)
        # 使用 Einstein 求和约定计算随机变量
        out = mean + np.einsum('jp,ipq,kq->ijk',
                               rowchol, std_norm, colchol,
                               optimize=True)
        # 如果 size 为 1，则将输出形状重塑为均值的形状
        if size == 1:
            out = out.reshape(mean.shape)
        # 返回随机变量
        return out
    def entropy(self, rowcov=1, colcov=1):
        """计算矩阵正态分布的对数概率密度函数的熵。

        Parameters
        ----------
        rowcov : array_like, optional
            行之间协方差矩阵（默认为 ``1``）
        colcov : array_like, optional
            列之间协方差矩阵（默认为 ``1``）

        Returns
        -------
        entropy : float
            分布的熵

        Notes
        -----
        %(_matnorm_doc_callparams_note)s

        """
        dummy_mean = np.zeros((rowcov.shape[0], colcov.shape[0]))
        # 处理参数并返回相关维度及协方差对
        dims, _, rowcov, colcov = self._process_parameters(dummy_mean,
                                                           rowcov,
                                                           colcov)
        # 确保行协方差矩阵为半正定矩阵
        rowpsd = _PSD(rowcov, allow_singular=False)
        # 确保列协方差矩阵为半正定矩阵
        colpsd = _PSD(colcov, allow_singular=False)

        # 返回熵值计算结果
        return self._entropy(dims, rowpsd.log_pdet, colpsd.log_pdet)

    def _entropy(self, dims, row_cov_logdet, col_cov_logdet):
        n, p = dims
        # 计算熵值
        return (0.5 * n * p * (1 + _LOG_2PI) + 0.5 * p * row_cov_logdet +
                0.5 * n * col_cov_logdet)
matrix_normal = matrix_normal_gen()

class matrix_normal_frozen(multi_rv_frozen):
    """
    Create a frozen matrix normal distribution.

    Parameters
    ----------
    %(_matnorm_doc_default_callparams)s
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is `None` the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import matrix_normal

    >>> distn = matrix_normal(mean=np.zeros((3,3)))
    >>> X = distn.rvs(); X
    array([[-0.02976962,  0.93339138, -0.09663178],
           [ 0.67405524,  0.28250467, -0.93308929],
           [-0.31144782,  0.74535536,  1.30412916]])
    >>> distn.pdf(X)
    2.5160642368346784e-05
    >>> distn.logpdf(X)
    -10.590229595124615
    """

    def __init__(self, mean=None, rowcov=1, colcov=1, seed=None):
        # 初始化 matrix_normal_gen 对象
        self._dist = matrix_normal_gen(seed)
        # 处理参数并设置对象的属性
        self.dims, self.mean, self.rowcov, self.colcov = \
            self._dist._process_parameters(mean, rowcov, colcov)
        # 计算行和列的协方差的平方根，确保它们是半正定的
        self.rowpsd = _PSD(self.rowcov, allow_singular=False)
        self.colpsd = _PSD(self.colcov, allow_singular=False)

    def logpdf(self, X):
        # 处理输入数据 X，确保符合分布的要求
        X = self._dist._process_quantiles(X, self.dims)
        # 调用 matrix_normal_gen 的 logpdf 方法计算对数概率密度
        out = self._dist._logpdf(self.dims, X, self.mean, self.rowpsd.U,
                                 self.rowpsd.log_pdet, self.colpsd.U,
                                 self.colpsd.log_pdet)
        return _squeeze_output(out)

    def pdf(self, X):
        # 计算给定数据 X 的概率密度函数值
        return np.exp(self.logpdf(X))

    def rvs(self, size=1, random_state=None):
        # 生成符合该分布的随机变量
        return self._dist.rvs(self.mean, self.rowcov, self.colcov, size,
                              random_state)

    def entropy(self):
        # 计算分布的熵
        return self._dist._entropy(self.dims, self.rowpsd.log_pdet,
                                   self.colpsd.log_pdet)


# Set frozen generator docstrings from corresponding docstrings in
# matrix_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'entropy']:
    # 获取 matrix_normal_gen 对象的方法
    method = matrix_normal_gen.__dict__[name]
    # 获取 matrix_normal_frozen 对象的同名方法
    method_frozen = matrix_normal_frozen.__dict__[name]
    # 将方法的文档字符串格式化填充到 matrix_normal_frozen 对象的同名方法上
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             matnorm_docdict_noparams)
    # 将方法的文档字符串格式化填充到 matrix_normal_gen 对象的同名方法上
    method.__doc__ = doccer.docformat(method.__doc__, matnorm_docdict_params)

_dirichlet_doc_default_callparams = """\
alpha : array_like
    The concentration parameters. The number of entries determines the
    dimensionality of the distribution.
"""
_dirichlet_doc_frozen_callparams = ""

_dirichlet_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

# 定义 dirichlet_docdict_params 字典，用于存储参数文档信息
dirichlet_docdict_params = {
    # 将 '_dirichlet_doc_default_callparams' 键指向 _dirichlet_doc_default_callparams 变量的值
    '_dirichlet_doc_default_callparams': _dirichlet_doc_default_callparams,
    # 将 '_doc_random_state' 键指向 _doc_random_state 变量的值
    '_doc_random_state': _doc_random_state
}

dirichlet_docdict_noparams = {
    '_dirichlet_doc_default_callparams': _dirichlet_doc_frozen_callparams,
    '_doc_random_state': _doc_random_state
}

# 定义一个函数 _dirichlet_check_parameters，用于检查 Dirichlet 分布的参数 alpha
def _dirichlet_check_parameters(alpha):
    # 将参数 alpha 转换为 NumPy 数组
    alpha = np.asarray(alpha)
    # 检查参数中是否有小于等于 0 的值，如果有则引发 ValueError 异常
    if np.min(alpha) <= 0:
        raise ValueError("All parameters must be greater than 0")
    # 检查 alpha 的维度是否为 1，如果不是则引发 ValueError 异常
    elif alpha.ndim != 1:
        raise ValueError("Parameter vector 'a' must be one dimensional, "
                         f"but a.shape = {alpha.shape}.")
    # 返回处理后的 alpha
    return alpha

# 定义一个函数 _dirichlet_check_input，用于检查 Dirichlet 分布的输入参数 alpha 和 x
def _dirichlet_check_input(alpha, x):
    # 将输入的 x 转换为 NumPy 数组
    x = np.asarray(x)

    # 检查 x 的长度是否与 alpha 的长度相等或者比 alpha 的长度少 1
    if x.shape[0] + 1 != alpha.shape[0] and x.shape[0] != alpha.shape[0]:
        raise ValueError("Vector 'x' must have either the same number "
                         "of entries as, or one entry fewer than, "
                         f"parameter vector 'a', but alpha.shape = {alpha.shape} "
                         f"and x.shape = {x.shape}.")

    # 如果 x 的长度不等于 alpha 的长度，添加一个补充的值使得它们相等
    if x.shape[0] != alpha.shape[0]:
        xk = np.array([1 - np.sum(x, 0)])
        if xk.ndim == 1:
            x = np.append(x, xk)
        elif xk.ndim == 2:
            x = np.vstack((x, xk))
        else:
            raise ValueError("The input must be one dimensional or a two "
                             "dimensional matrix containing the entries.")

    # 检查 x 中是否有小于 0 的值，如果有则引发 ValueError 异常
    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal "
                         "to zero.")

    # 检查 x 中是否有大于 1 的值，如果有则引发 ValueError 异常
    if np.max(x) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")

    # 检查 x_i 是否为 0 时 alpha_i 是否小于 1，如果满足则引发 ValueError 异常
    xeq0 = (x == 0)
    alphalt1 = (alpha < 1)
    if x.shape != alpha.shape:
        alphalt1 = np.repeat(alphalt1, x.shape[-1], axis=-1).reshape(x.shape)
    chk = np.logical_and(xeq0, alphalt1)
    if np.sum(chk):
        raise ValueError("Each entry in 'x' must be greater than zero if its "
                         "alpha is less than one.")

    # 检查输入向量 x 的总和是否接近 1，如果不是则引发 ValueError 异常
    if (np.abs(np.sum(x, 0) - 1.0) > 10e-10).any():
        raise ValueError("The input vector 'x' must lie within the normal "
                         "simplex. but np.sum(x, 0) = %s." % np.sum(x, 0))

    # 返回处理后的 x
    return x

# 定义一个函数 _lnB，用于计算 Dirichlet 分布的辅助函数
def _lnB(alpha):
    r"""Internal helper function to compute the log of the useful quotient.

    .. math::

        B(\alpha) = \frac{\prod_{i=1}{K}\Gamma(\alpha_i)}
                         {\Gamma\left(\sum_{i=1}^{K} \alpha_i \right)}

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s

    Returns
    -------
    B : scalar
        Helper quotient, internal use only

    """
    # 计算 B(\alpha) 的对数，用于内部使用
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

# 定义一个类 dirichlet_gen，用于生成 Dirichlet 分布的随机变量
class dirichlet_gen(multi_rv_generic):
    r"""A Dirichlet random variable.

    The ``alpha`` keyword specifies the concentration parameters of the
    distribution.

    .. versionadded:: 0.15.0

    Methods
    -------
    pdf(x, alpha)
        Probability density function.
    logpdf(x, alpha)
        Log of the probability density function.

    """
    rvs(alpha, size=1, random_state=None)
        从狄利克雷分布中抽取随机样本。
    mean(alpha)
        狄利克雷分布的均值。
    var(alpha)
        狄利克雷分布的方差。
    cov(alpha)
        狄利克雷分布的协方差。
    entropy(alpha)
        计算狄利克雷分布的微分熵。

    Parameters
    ----------
    %(_dirichlet_doc_default_callparams)s
        狄利克雷分布函数的默认调用参数。
    %(_doc_random_state)s
        随机状态的文档说明。

    Notes
    -----
    每个 :math:`\alpha` 值必须为正数。该分布仅支持以下凸集：

    .. math::
        \sum_{i=1}^{K} x_i = 1

    其中 :math:`0 < x_i < 1`。

    如果分位数不在凸集内，会引发 ValueError。

    `dirichlet` 的概率密度函数为：

    .. math::

        f(x) = \frac{1}{\mathrm{B}(\boldsymbol\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}

    其中

    .. math::

        \mathrm{B}(\boldsymbol\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}
                                     {\Gamma\bigl(\sum_{i=1}^K \alpha_i\bigr)}

    :math:`\boldsymbol\alpha=(\alpha_1,\ldots,\alpha_K)` 为浓度参数，:math:`K` 为取值空间的维数。

    需要注意 `dirichlet` 接口有些不一致。
    rvs 函数返回的数组与 pdf 和 logpdf 期望的格式相反。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import dirichlet

    生成一个狄利克雷随机变量

    >>> quantiles = np.array([0.2, 0.2, 0.6])  # 指定分位数
    >>> alpha = np.array([0.4, 5, 15])  # 指定浓度参数
    >>> dirichlet.pdf(quantiles, alpha)
    0.2843831684937255

    同样的 PDF 但是使用对数刻度

    >>> dirichlet.logpdf(quantiles, alpha)
    -1.2574327653159187

    一旦我们指定了狄利克雷分布
    我们可以计算感兴趣的量

    >>> dirichlet.mean(alpha)  # 获取分布的均值
    array([0.01960784, 0.24509804, 0.73529412])
    >>> dirichlet.var(alpha) # 获取方差
    array([0.00089829, 0.00864603, 0.00909517])
    >>> dirichlet.entropy(alpha)  # 计算微分熵
    -4.3280162474082715

    我们也可以从分布中返回随机样本

    >>> dirichlet.rvs(alpha, size=1, random_state=1)
    array([[0.00766178, 0.24670518, 0.74563305]])
    >>> dirichlet.rvs(alpha, size=2, random_state=2)
    array([[0.01639427, 0.1292273 , 0.85437844],
           [0.00156917, 0.19033695, 0.80809388]])

    或者，可以将对象作为函数调用，固定浓度参数，返回一个“冻结”的狄利克雷随机变量：

    >>> rv = dirichlet(alpha)
    >>> # 冻结的对象具有相同的方法，但保持给定的浓度参数不变。
    def __init__(self, seed=None):
        # 调用父类的初始化方法，设置随机数种子
        super().__init__(seed)
        # 根据给定的文档格式化类的文档字符串
        self.__doc__ = doccer.docformat(self.__doc__, dirichlet_docdict_params)

    def __call__(self, alpha, seed=None):
        # 返回一个冻结的 Dirichlet 分布实例
        return dirichlet_frozen(alpha, seed=seed)

    def _logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        %(_dirichlet_doc_default_callparams)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # 计算 Dirichlet 概率密度函数的对数
        lnB = _lnB(alpha)
        return - lnB + np.sum((xlogy(alpha - 1, x.T)).T, 0)

    def logpdf(self, x, alpha):
        """Log of the Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`.

        """
        # 检查参数 alpha 是否合法
        alpha = _dirichlet_check_parameters(alpha)
        # 检查输入 x 是否合法
        x = _dirichlet_check_input(alpha, x)

        # 调用内部方法计算对数概率密度函数
        out = self._logpdf(x, alpha)
        return _squeeze_output(out)

    def pdf(self, x, alpha):
        """The Dirichlet probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray or scalar
            The probability density function evaluated at `x`.

        """
        # 检查参数 alpha 是否合法
        alpha = _dirichlet_check_parameters(alpha)
        # 检查输入 x 是否合法
        x = _dirichlet_check_input(alpha, x)

        # 调用内部方法计算概率密度函数的指数
        out = np.exp(self._logpdf(x, alpha))
        return _squeeze_output(out)

    def mean(self, alpha):
        """Mean of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        mu : ndarray or scalar
            Mean of the Dirichlet distribution.

        """
        # 检查参数 alpha 是否合法
        alpha = _dirichlet_check_parameters(alpha)

        # 计算 Dirichlet 分布的均值
        out = alpha / (np.sum(alpha))
        return _squeeze_output(out)

    def var(self, alpha):
        """Variance of the Dirichlet distribution.

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s

        Returns
        -------
        v : ndarray or scalar
            Variance of the Dirichlet distribution.

        """
        # 检查参数 alpha 是否合法
        alpha = _dirichlet_check_parameters(alpha)

        # 计算 Dirichlet 分布的方差
        alpha0 = np.sum(alpha)
        out = (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))
        return _squeeze_output(out)
    def cov(self, alpha):
        """计算Dirichlet分布的协方差矩阵。

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
            _dirichlet_doc_default_callparams是一个文档字符串占位符，描述了函数的参数。

        Returns
        -------
        cov : ndarray
            分布的协方差矩阵。
        """

        alpha = _dirichlet_check_parameters(alpha)  # 检查并规范化参数alpha
        alpha0 = np.sum(alpha)  # 计算参数alpha的总和
        a = alpha / alpha0  # 计算归一化的alpha值

        cov = (np.diag(a) - np.outer(a, a)) / (alpha0 + 1)  # 计算Dirichlet分布的协方差矩阵
        return _squeeze_output(cov)  # 返回压缩后的协方差矩阵结果

    def entropy(self, alpha):
        """
        计算Dirichlet分布的微分熵（Differential entropy）。

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
            _dirichlet_doc_default_callparams是一个文档字符串占位符，描述了函数的参数。

        Returns
        -------
        h : scalar
            Dirichlet分布的熵

        """

        alpha = _dirichlet_check_parameters(alpha)  # 检查并规范化参数alpha

        alpha0 = np.sum(alpha)  # 计算参数alpha的总和
        lnB = _lnB(alpha)  # 计算对数Beta函数值
        K = alpha.shape[0]  # 获取参数alpha的维度

        out = lnB + (alpha0 - K) * scipy.special.psi(alpha0) - np.sum(
            (alpha - 1) * scipy.special.psi(alpha))  # 计算Dirichlet分布的熵
        return _squeeze_output(out)  # 返回压缩后的熵结果

    def rvs(self, alpha, size=1, random_state=None):
        """
        从Dirichlet分布中抽取随机样本。

        Parameters
        ----------
        %(_dirichlet_doc_default_callparams)s
            _dirichlet_doc_default_callparams是一个文档字符串占位符，描述了函数的参数。
        size : int, optional
            要抽取的样本数量（默认为1）。
        %(_doc_random_state)s
            _doc_random_state是一个文档字符串占位符，描述了随机状态参数的作用。

        Returns
        -------
        rvs : ndarray or scalar
            大小为(`size`, `N`)的随机变量，其中`N`是随机变量的维度。

        """
        alpha = _dirichlet_check_parameters(alpha)  # 检查并规范化参数alpha
        random_state = self._get_random_state(random_state)  # 获取随机状态对象
        return random_state.dirichlet(alpha, size=size)  # 使用给定参数从Dirichlet分布中抽取随机样本
dirichlet = dirichlet_gen()

class dirichlet_frozen(multi_rv_frozen):
    def __init__(self, alpha, seed=None):
        # 初始化冻结的Dirichlet分布实例
        self.alpha = _dirichlet_check_parameters(alpha)
        # 使用alpha参数检查并设置Dirichlet分布的参数
        self._dist = dirichlet_gen(seed)
        # 创建一个基于种子(seed)的Dirichlet分布生成器实例

    def logpdf(self, x):
        # 返回Dirichlet分布的对数概率密度函数值
        return self._dist.logpdf(x, self.alpha)

    def pdf(self, x):
        # 返回Dirichlet分布的概率密度函数值
        return self._dist.pdf(x, self.alpha)

    def mean(self):
        # 返回Dirichlet分布的期望值
        return self._dist.mean(self.alpha)

    def var(self):
        # 返回Dirichlet分布的方差
        return self._dist.var(self.alpha)

    def cov(self):
        # 返回Dirichlet分布的协方差
        return self._dist.cov(self.alpha)

    def entropy(self):
        # 返回Dirichlet分布的熵
        return self._dist.entropy(self.alpha)

    def rvs(self, size=1, random_state=None):
        # 从Dirichlet分布中抽取随机样本
        return self._dist.rvs(self.alpha, size, random_state)


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_normal_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'mean', 'var', 'cov', 'entropy']:
    method = dirichlet_gen.__dict__[name]
    method_frozen = dirichlet_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, dirichlet_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, dirichlet_docdict_params)


_wishart_doc_default_callparams = """\
df : int
    Degrees of freedom, must be greater than or equal to dimension of the
    scale matrix
scale : array_like
    Symmetric positive definite scale matrix of the distribution
"""

_wishart_doc_callparams_note = ""

_wishart_doc_frozen_callparams = ""

_wishart_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

wishart_docdict_params = {
    '_doc_default_callparams': _wishart_doc_default_callparams,
    '_doc_callparams_note': _wishart_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

wishart_docdict_noparams = {
    '_doc_default_callparams': _wishart_doc_frozen_callparams,
    '_doc_callparams_note': _wishart_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class wishart_gen(multi_rv_generic):
    r"""A Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal precision matrix (the inverse of the covariance
    matrix). These arguments must satisfy the relationship
    ``df > scale.ndim - 1``, but see notes on using the `rvs` method with
    ``df < scale.ndim``.

    Methods
    -------
    pdf(x, df, scale)
        Probability density function.
    logpdf(x, df, scale)
        Log of the probability density function.
    rvs(df, scale, size=1, random_state=None)
        Draw random samples from a Wishart distribution.
    entropy()
        Compute the differential entropy of the Wishart distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Raises
    """
        ------
        scipy.linalg.LinAlgError
            If the scale matrix `scale` is not positive definite.
    
        See Also
        --------
        invwishart, chi2
    
        Notes
        -----
        %(_doc_callparams_note)s
    
        The scale matrix `scale` must be a symmetric positive definite
        matrix. Singular matrices, including the symmetric positive semi-definite
        case, are not supported. Symmetry is not checked; only the lower triangular
        portion is used.
    
        The Wishart distribution is often denoted
    
        .. math::
    
            W_p(\nu, \Sigma)
    
        where :math:`\nu` is the degrees of freedom and :math:`\Sigma` is the
        :math:`p \times p` scale matrix.
    
        The probability density function for `wishart` has support over positive
        definite matrices :math:`S`; if :math:`S \sim W_p(\nu, \Sigma)`, then
        its PDF is given by:
    
        .. math::
    
            f(S) = \frac{|S|^{\frac{\nu - p - 1}{2}}}{2^{ \frac{\nu p}{2} }
                   |\Sigma|^\frac{\nu}{2} \Gamma_p \left ( \frac{\nu}{2} \right )}
                   \exp\left( -tr(\Sigma^{-1} S) / 2 \right)
    
        If :math:`S \sim W_p(\nu, \Sigma)` (Wishart) then
        :math:`S^{-1} \sim W_p^{-1}(\nu, \Sigma^{-1})` (inverse Wishart).
    
        If the scale matrix is 1-dimensional and equal to one, then the Wishart
        distribution :math:`W_1(\nu, 1)` collapses to the :math:`\chi^2(\nu)`
        distribution.
    
        The algorithm [2]_ implemented by the `rvs` method may
        produce numerically singular matrices with :math:`p - 1 < \nu < p`; the
        user may wish to check for this condition and generate replacement samples
        as necessary.
    
    
        .. versionadded:: 0.16.0
    
        References
        ----------
        .. [1] M.L. Eaton, "Multivariate Statistics: A Vector Space Approach",
               Wiley, 1983.
        .. [2] W.B. Smith and R.R. Hocking, "Algorithm AS 53: Wishart Variate
               Generator", Applied Statistics, vol. 21, pp. 341-345, 1972.
    
        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy.stats import wishart, chi2
        >>> x = np.linspace(1e-5, 8, 100)
        >>> w = wishart.pdf(x, df=3, scale=1); w[:5]
        array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
        >>> c = chi2.pdf(x, 3); c[:5]
        array([ 0.00126156,  0.10892176,  0.14793434,  0.17400548,  0.1929669 ])
        >>> plt.plot(x, w)
        >>> plt.show()
    
        The input quantiles can be any shape of array, as long as the last
        axis labels the components.
    
        Alternatively, the object may be called (as a function) to fix the degrees
        of freedom and scale parameters, returning a "frozen" Wishart random
        variable:
    
        >>> rv = wishart(df=1, scale=1)
        >>> # Frozen object with the same methods but holding the given
        >>> # degrees of freedom and scale fixed.
    
        """
    
        # 初始化函数，用于设置 Wishart 分布的种子
        def __init__(self, seed=None):
            # 调用父类的初始化函数
            super().__init__(seed)
            # 使用文档字符串格式化工具，将文档字符串根据给定的参数格式化
            self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)
    def __call__(self, df=None, scale=None, seed=None):
        """
        创建一个冻结的 Wishart 分布。

        查看 `wishart_frozen` 获取更多信息。
        """
        # 调用 wishart_frozen 函数并返回结果
        return wishart_frozen(df, scale, seed)

    def _process_parameters(self, df, scale):
        # 如果 scale 为 None，则设置为默认值 1.0
        if scale is None:
            scale = 1.0
        # 将 scale 转换为浮点型数组
        scale = np.asarray(scale, dtype=float)

        # 处理 scale 数组的维度
        if scale.ndim == 0:
            scale = scale[np.newaxis, np.newaxis]
        elif scale.ndim == 1:
            # 如果是一维数组，则将其作为对角矩阵处理
            scale = np.diag(scale)
        elif scale.ndim == 2 and not scale.shape[0] == scale.shape[1]:
            # 如果是二维数组但不是方阵，则引发错误
            raise ValueError("Array 'scale' must be square if it is two"
                             " dimensional, but scale.shape = %s."
                             % str(scale.shape))
        elif scale.ndim > 2:
            # 如果维度大于二，则引发错误
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)

        # 获取 scale 的维度
        dim = scale.shape[0]

        # 处理 df 参数
        if df is None:
            df = dim
        elif not np.isscalar(df):
            # 如果 df 不是标量，则引发错误
            raise ValueError("Degrees of freedom must be a scalar.")
        elif df <= dim - 1:
            # 如果 df 小于等于 scale 矩阵维度减一，则引发错误
            raise ValueError("Degrees of freedom must be greater than the "
                             "dimension of scale matrix minus 1.")

        # 返回处理后的 dim, df, scale
        return dim, df, scale

    def _process_quantiles(self, x, dim):
        """
        调整分位数数组，使得最后一个轴标记每个数据点的组成部分。
        """
        # 将 x 转换为浮点型数组
        x = np.asarray(x, dtype=float)

        # 处理 x 数组的维度
        if x.ndim == 0:
            x = x * np.eye(dim)[:, :, np.newaxis]
        if x.ndim == 1:
            if dim == 1:
                x = x[np.newaxis, np.newaxis, :]
            else:
                x = np.diag(x)[:, :, np.newaxis]
        elif x.ndim == 2:
            if not x.shape[0] == x.shape[1]:
                # 如果是二维数组但不是方阵，则引发错误
                raise ValueError("Quantiles must be square if they are two"
                                 " dimensional, but x.shape = %s."
                                 % str(x.shape))
            x = x[:, :, np.newaxis]
        elif x.ndim == 3:
            if not x.shape[0] == x.shape[1]:
                # 如果是三维数组且前两个维度不是方阵，则引发错误
                raise ValueError("Quantiles must be square in the first two"
                                 " dimensions if they are three dimensional"
                                 ", but x.shape = %s." % str(x.shape))
        elif x.ndim > 3:
            # 如果维度大于三，则引发错误
            raise ValueError("Quantiles must be at most two-dimensional with"
                             " an additional dimension for multiple"
                             "components, but x.ndim = %d" % x.ndim)

        # 确保 x 数组的形状为 [dim, dim, *]
        if not x.shape[0:2] == (dim, dim):
            # 如果形状不符合预期，则引发错误
            raise ValueError('Quantiles have incompatible dimensions: should'
                             f' be {(dim, dim)}, got {x.shape[0:2]}.')

        # 返回处理后的 x 数组
        return x
    def _process_size(self, size):
        # 将size转换为NumPy数组
        size = np.asarray(size)

        # 如果size的维度为0，添加一个新轴
        if size.ndim == 0:
            size = size[np.newaxis]
        # 如果size的维度大于1，抛出数值错误
        elif size.ndim > 1:
            raise ValueError('Size must be an integer or tuple of integers;'
                             ' thus must have dimension <= 1.'
                             ' Got size.ndim = %s' % str(tuple(size)))
        # 计算元素个数n和形状shape
        n = size.prod()
        shape = tuple(size)

        return n, shape

    def _logpdf(self, x, dim, df, scale, log_det_scale, C):
        """Log of the Wishart probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        scale : ndarray
            Scale matrix
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triangular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # 计算x的对数行列式
        # 注意：x沿最后一个轴有分量，因此x.T沿第0轴有分量。由于det(A) = det(A')，这给出了行列式的1维向量

        # 检索tr(scale^{-1} x)
        log_det_x = np.empty(x.shape[-1])
        scale_inv_x = np.empty(x.shape)
        tr_scale_inv_x = np.empty(x.shape[-1])
        for i in range(x.shape[-1]):
            _, log_det_x[i] = self._cholesky_logdet(x[:, :, i])
            scale_inv_x[:, :, i] = scipy.linalg.cho_solve((C, True), x[:, :, i])
            tr_scale_inv_x[i] = scale_inv_x[:, :, i].trace()

        # 计算对数概率密度函数值
        out = ((0.5 * (df - dim - 1) * log_det_x - 0.5 * tr_scale_inv_x) -
               (0.5 * df * dim * _LOG_2 + 0.5 * df * log_det_scale +
                multigammaln(0.5*df, dim)))

        return out

    def logpdf(self, x, df, scale):
        """Log of the Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s

        """
        # 处理参数并确保它们符合要求
        dim, df, scale = self._process_parameters(df, scale)
        # 处理输入的量化数值x，使其符合维度要求
        x = self._process_quantiles(x, dim)

        # 对scale进行Cholesky分解，获取其对数行列式
        C, log_det_scale = self._cholesky_logdet(scale)

        # 计算对数概率密度函数的输出值
        out = self._logpdf(x, dim, df, scale, log_det_scale, C)
        return _squeeze_output(out)
    def pdf(self, x, df, scale):
        """Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
            Each quantile must be a symmetric positive definite matrix.
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
            Additional notes about the parameters and usage.
        """
        # 返回通过对数概率密度函数计算得到的指数函数结果
        return np.exp(self.logpdf(x, df, scale))

    def _mean(self, dim, df, scale):
        """Mean of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mean' instead.
            Note warning about direct usage.
        """
        # 返回均值，即自由度乘以规模矩阵
        return df * scale

    def mean(self, df, scale):
        """Mean of the Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Returns
        -------
        mean : float
            The mean of the distribution
        """
        # 处理参数并调用 _mean 方法计算均值，然后压缩输出
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        return _squeeze_output(out)

    def _mode(self, dim, df, scale):
        """Mode of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'mode' instead.
            Note warning about direct usage.
        """
        # 如果自由度大于等于维度加一，则返回众数，否则返回 None
        if df >= dim + 1:
            out = (df-dim-1) * scale
        else:
            out = None
        return out

    def mode(self, df, scale):
        """Mode of the Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix.

        Parameters
        ----------
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Returns
        -------
        mode : float or None
            The Mode of the distribution
        """
        # 处理参数并调用 _mode 方法计算众数，然后压缩输出（如果有的话）
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        return _squeeze_output(out) if out is not None else out

    def _var(self, dim, df, scale):
        """Variance of the Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
        %(_doc_default_callparams)s
            Additional parameters, typically `df` and `scale`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.
            Note warning about direct usage.
        """
        # 计算方差，包括规模矩阵的平方和外积，并乘以自由度
        var = scale**2
        diag = scale.diagonal()  # 1 x dim array
        var += np.outer(diag, diag)
        var *= df
        return var
    # 计算 Wishart 分布的方差。

    # Parameters
    # ----------
    # df : int
    #     自由度
    # scale : array-like
    #     比例矩阵的维度

    # Returns
    # -------
    # var : float
    #     分布的方差
    def var(self, df, scale):
        # 处理输入参数，获取维度、自由度和比例矩阵
        dim, df, scale = self._process_parameters(df, scale)
        # 调用 _var 方法计算方差
        out = self._var(dim, df, scale)
        # 压缩输出结果
        return _squeeze_output(out)

    # Parameters
    # ----------
    # n : integer
    #     要生成的变量数
    # shape : iterable
    #     要生成的变量形状
    # dim : int
    #     比例矩阵的维度
    # df : int
    #     自由度
    # random_state : {None, int, `numpy.random.Generator`,
    #                 `numpy.random.RandomState`}, optional
    #     随机数种子，控制随机数生成的状态

    # Notes
    # -----
    # 由于此函数不进行参数检查，不应直接调用；应使用 'rvs' 方法代替。
    def _standard_rvs(self, n, shape, dim, df, random_state):
        # 生成非对角线元素的随机正态变量
        n_tril = dim * (dim-1) // 2
        covariances = random_state.normal(
            size=n*n_tril).reshape(shape+(n_tril,))

        # 生成对角线元素的随机卡方变量
        variances = (np.r_[[random_state.chisquare(df-(i+1)+1, size=n)**0.5
                            for i in range(dim)]].reshape((dim,) +
                                                          shape[::-1]).T)

        # 创建 A 矩阵（矩阵）- 下三角形式
        A = np.zeros(shape + (dim, dim))

        # 输入非对角线元素的协方差
        size_idx = tuple([slice(None, None, None)]*len(shape))
        tril_idx = np.tril_indices(dim, k=-1)
        A[size_idx + tril_idx] = covariances

        # 输入对角线元素的方差
        diag_idx = np.diag_indices(dim)
        A[size_idx + diag_idx] = variances

        return A
    def _rvs(self, n, shape, dim, df, C, random_state):
        """Draw random samples from a Wishart distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triangular.
        %(_doc_random_state)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        # 计算矩阵 A，实际上是矩阵 B 的下三角 Cholesky 分解，其中 B ~ W(df, I)
        A = self._standard_rvs(n, shape, dim, df, random_state)

        # 计算 SA = C A A' C', 其中 SA ~ W(df, scale)
        # 注意：这是 (下) (下) (下)' (下)' 的乘积，或者用 B = AA' 表示，
        #       它是 C B C'，其中 C 是 scale 矩阵的下三角 Cholesky 分解。
        #       这似乎与 [1]_ 中的说明相冲突，后者建议应该是 D' B D，
        #       其中 D 是 scale 矩阵的下三角因子分解。然而，它意味着 Bartlett (1933)
        #       表示 Wishart 随机变量为 L A A' L'，其中 L 是下三角的，
        #       因此将 D' 视为上三角可能是 [1]_ 中的一个打字错误或误读。
        for index in np.ndindex(shape):
            CA = np.dot(C, A[index])
            A[index] = np.dot(CA, CA.T)

        return A

    def rvs(self, df, scale, size=1, random_state=None):
        """Draw random samples from a Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (``dim``, ``dim``), where
            ``dim`` is the dimension of the scale matrix.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # 对 scale 进行 Cholesky 分解
        C = scipy.linalg.cholesky(scale, lower=True)

        # 调用 _rvs 方法来生成 Wishart 分布的随机样本
        out = self._rvs(n, shape, dim, df, C, random_state)

        return _squeeze_output(out)
    # 计算 Wishart 分布的差分熵

    def _entropy(self, dim, df, log_det_scale):
        """Compute the differential entropy of the Wishart.

        Parameters
        ----------
        dim : int
            Scale 矩阵的维度
        df : int
            自由度
        log_det_scale : float
            Scale 矩阵的对数行列式

        Notes
        -----
        由于此函数不进行参数检查，不应直接调用；应使用 'entropy' 方法代替。
        """
        return (
            0.5 * (dim+1) * log_det_scale +    # Wishart 分布的熵的一部分
            0.5 * dim * (dim+1) * _LOG_2 +    # 常数项与维度相关
            multigammaln(0.5*df, dim) -       # 多维 Gamma 函数的自然对数
            0.5 * (df - dim - 1) * np.sum(
                [psi(0.5*(df + 1 - (i+1))) for i in range(dim)]    # psi 函数的求和
            ) +
            0.5 * df * dim                     # 常数项与自由度、维度相关
        )

    # 计算 Wishart 分布的熵

    def entropy(self, df, scale):
        """Compute the differential entropy of the Wishart.

        Parameters
        ----------
        %(_doc_default_callparams)s
            默认调用参数

        Returns
        -------
        h : scalar
            Wishart 分布的熵

        Notes
        -----
        %(_doc_callparams_note)s
            参数调用的注释说明
        """
        dim, df, scale = self._process_parameters(df, scale)
        _, log_det_scale = self._cholesky_logdet(scale)   # 计算 scale 的 Cholesky 分解和对数行列式
        return self._entropy(dim, df, log_det_scale)      # 调用 _entropy 方法计算熵

    # 计算 Cholesky 分解和对数行列式

    def _cholesky_logdet(self, scale):
        """Compute Cholesky decomposition and determine (log(det(scale)).

        Parameters
        ----------
        scale : ndarray
            Scale matrix. 尺度矩阵

        Returns
        -------
        c_decomp : ndarray
            `scale` 的 Cholesky 分解
        logdet : scalar
            `scale` 的对数行列式

        Notes
        -----
        此处计算 `logdet` 等价于 `np.linalg.slogdet(scale)`。速度大约快两倍。
        """
        c_decomp = scipy.linalg.cholesky(scale, lower=True)   # 使用 Cholesky 分解计算尺度矩阵的下三角部分
        logdet = 2 * np.sum(np.log(c_decomp.diagonal()))     # 计算尺度矩阵的对数行列式
        return c_decomp, logdet
# 创建一个 Wishart 分布的随机数生成器对象
wishart = wishart_gen()


class wishart_frozen(multi_rv_frozen):
    """创建一个冻结的 Wishart 分布。

    Parameters
    ----------
    df : array_like
        分布的自由度
    scale : array_like
        分布的规模矩阵
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        如果 `seed` 是 None (或 `np.random`)，则使用 `numpy.random.RandomState` 单例。
        如果 `seed` 是一个整数，则使用一个新的 ``RandomState`` 实例，并用 `seed` 初始化。
        如果 `seed` 已经是一个 ``Generator`` 或 ``RandomState`` 实例，则直接使用该实例。
    """
    def __init__(self, df, scale, seed=None):
        self._dist = wishart_gen(seed)
        # 处理参数并设置维度、自由度和规模矩阵
        self.dim, self.df, self.scale = self._dist._process_parameters(
            df, scale)
        # 计算规模矩阵的 Cholesky 分解和对数行列式
        self.C, self.log_det_scale = self._dist._cholesky_logdet(self.scale)

    def logpdf(self, x):
        # 处理输入的量化值并计算对数概率密度函数
        x = self._dist._process_quantiles(x, self.dim)

        out = self._dist._logpdf(x, self.dim, self.df, self.scale,
                                 self.log_det_scale, self.C)
        return _squeeze_output(out)

    def pdf(self, x):
        # 计算概率密度函数
        return np.exp(self.logpdf(x))

    def mean(self):
        # 计算均值
        out = self._dist._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def mode(self):
        # 计算众数
        out = self._dist._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def var(self):
        # 计算方差
        out = self._dist._var(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        # 处理样本大小并生成随机样本
        n, shape = self._dist._process_size(size)
        out = self._dist._rvs(n, shape, self.dim, self.df,
                              self.C, random_state)
        return _squeeze_output(out)

    def entropy(self):
        # 计算熵
        return self._dist._entropy(self.dim, self.df, self.log_det_scale)


# 从 Wishart 分布中设置冻结生成器的文档字符串，并在类文档字符串中填充默认字符串
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs', 'entropy']:
    method = wishart_gen.__dict__[name]
    method_frozen = wishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)


class invwishart_gen(wishart_gen):
    r"""一个逆 Wishart 分布的随机变量。

    关键字 `df` 指定自由度。关键字 `scale` 指定规模矩阵，必须是对称且正定的。
    在这个上下文中，规模矩阵通常被解释为多变量正态分布的协方差矩阵。

    Methods
    -------
    pdf(x, df, scale)
        概率密度函数。
    logpdf(x, df, scale)
        概率密度函数的对数。
    """
    rvs(df, scale, size=1, random_state=None)
        从逆 Wishart 分布中抽取随机样本。

    entropy(df, scale)
        分布的微分熵。

    Parameters
    ----------
    %(_doc_default_callparams)s
        默认参数设置，通常指示必需的输入参数。
    %(_doc_random_state)s
        随机数生成器的状态。

    Raises
    ------
    scipy.linalg.LinAlgError
        如果规模矩阵 `scale` 不是正定的。

    See Also
    --------
    wishart
        与 Wishart 分布相关的函数。

    Notes
    -----
    %(_doc_callparams_note)s
        关于函数调用参数的额外说明。

    The scale matrix `scale` must be a symmetric positive definite
    matrix. Singular matrices, including the symmetric positive semi-definite
    case, are not supported. Symmetry is not checked; only the lower triangular
    portion is used.
        规模矩阵 `scale` 必须是对称正定的矩阵。不支持奇异矩阵，包括对称正半定情况。不检查对称性，只使用下三角部分。

    The inverse Wishart distribution is often denoted
        逆 Wishart 分布通常表示为

    .. math::

        W_p^{-1}(\nu, \Psi)

    where :math:`\nu` is the degrees of freedom and :math:`\Psi` is the
    :math:`p \times p` scale matrix.
        其中 :math:`\nu` 是自由度，:math:`\Psi` 是 :math:`p \times p` 规模矩阵。

    The probability density function for `invwishart` has support over positive
    definite matrices :math:`S`; if :math:`S \sim W^{-1}_p(\nu, \Sigma)`,
    then its PDF is given by:

    .. math::

        f(S) = \frac{|\Sigma|^\frac{\nu}{2}}{2^{ \frac{\nu p}{2} }
               |S|^{\frac{\nu + p + 1}{2}} \Gamma_p \left(\frac{\nu}{2} \right)}
               \exp\left( -tr(\Sigma S^{-1}) / 2 \right)
        `invwishart` 的概率密度函数支持正定矩阵 :math:`S`；如果 :math:`S \sim W^{-1}_p(\nu, \Sigma)`，
        则其概率密度函数为：

    If :math:`S \sim W_p^{-1}(\nu, \Psi)` (inverse Wishart) then
    :math:`S^{-1} \sim W_p(\nu, \Psi^{-1})` (Wishart).
        如果 :math:`S \sim W_p^{-1}(\nu, \Psi)`（逆 Wishart 分布），则 :math:`S^{-1} \sim W_p(\nu, \Psi^{-1})`（Wishart 分布）。

    If the scale matrix is 1-dimensional and equal to one, then the inverse
    Wishart distribution :math:`W_1(\nu, 1)` collapses to the
    inverse Gamma distribution with parameters shape = :math:`\frac{\nu}{2}`
    and scale = :math:`\frac{1}{2}`.
        如果规模矩阵是一维并且等于一，那么逆 Wishart 分布 :math:`W_1(\nu, 1)` 会退化为参数形状为 :math:`\frac{\nu}{2}` 和尺度为 :math:`\frac{1}{2}` 的逆 Gamma 分布。

    Instead of inverting a randomly generated Wishart matrix as described in [2],
    here the algorithm in [4] is used to directly generate a random inverse-Wishart
    matrix without inversion.
        不同于在文献 [2] 中描述的随机生成 Wishart 矩阵再求逆，这里使用文献 [4] 中的算法直接生成随机的逆 Wishart 矩阵。

    .. versionadded:: 0.16.0
        添加于版本 0.16.0。

    References
    ----------
    .. [1] M.L. Eaton, "Multivariate Statistics: A Vector Space Approach",
           Wiley, 1983.
        M.L. Eaton,《多元统计：向量空间方法》，Wiley 出版社，1983 年。
    .. [2] M.C. Jones, "Generating Inverse Wishart Matrices", Communications
           in Statistics - Simulation and Computation, vol. 14.2, pp.511-514,
           1985.
        M.C. Jones,《生成逆 Wishart 矩阵》，《统计学与计算模拟通信》，第 14 卷第 2 期，1985 年，页码 511-514。
    .. [3] Gupta, M. and Srivastava, S. "Parametric Bayesian Estimation of
           Differential Entropy and Relative Entropy". Entropy 12, 818 - 843.
           2010.
        Gupta, M. 和 Srivastava, S.，《微分熵和相对熵的参数贝叶斯估计》，《熵》，第 12 卷，818 - 843 页，2010 年。
    .. [4] S.D. Axen, "Efficiently generating inverse-Wishart matrices and
           their Cholesky factors", :arXiv:`2310.15884v1`. 2023.
        S.D. Axen，《高效生成逆 Wishart 矩阵及其 Cholesky 因子》，:arXiv:`2310.15884v1`，2023 年。

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import invwishart, invgamma
    >>> x = np.linspace(0.01, 1, 100)
    >>> iw = invwishart.pdf(x, df=6, scale=1)
    >>> iw[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    >>> ig = invgamma.pdf(x, 6/2., scale=1./2)
    >>> ig[:3]
    array([  1.20546865e-15,   5.42497807e-06,   4.45813929e-03])
    # 绘制 x 和 iw 的图形
    >>> plt.plot(x, iw)
    # 显示图形
    >>> plt.show()

    # 输入分位数可以是任意形状的数组，只要最后一个轴标记组件。
    # Alternatively, the object may be called (as a function) to fix the degrees
    # of freedom and scale parameters, returning a "frozen" inverse Wishart
    # random variable:
    # 或者，可以将对象作为函数调用来固定自由度和比例参数，返回一个“冻结”的逆 Wishart 随机变量：

    """

    # 初始化方法，设置随机数种子
    def __init__(self, seed=None):
        super().__init__(seed)
        # 格式化文档字符串，根据 wishart_docdict_params 字典
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    # 调用对象时创建一个冻结的逆 Wishart 分布
    def __call__(self, df=None, scale=None, seed=None):
        """Create a frozen inverse Wishart distribution.

        See `invwishart_frozen` for more information.

        """
        return invwishart_frozen(df, scale, seed)

    # 计算逆 Wishart 概率密度函数的对数
    def _logpdf(self, x, dim, df, log_det_scale, C):
        """Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        log_det_scale : float
            Logarithm of the determinant of the scale matrix
        C : ndarray
            Cholesky factorization of the scale matrix, lower triangular.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        # 检索 tr(scale x^{-1})
        log_det_x = np.empty(x.shape[-1])
        tr_scale_x_inv = np.empty(x.shape[-1])
        # 获取 BLAS 函数，例如 'trsm'
        trsm = get_blas_funcs(('trsm'), (x,))
        # 如果维度大于1，逐个计算
        if dim > 1:
            for i in range(x.shape[-1]):
                # 计算 x 的 Cholesky 分解和对数行列式
                Cx, log_det_x[i] = self._cholesky_logdet(x[:, :, i])
                # 使用 BLAS 函数计算矩阵乘法的下三角矩阵
                A = trsm(1., Cx, C, side=0, lower=True)
                # 计算 tr(scale x^{-1}) 的范数的平方
                tr_scale_x_inv[i] = np.linalg.norm(A)**2
        else:
            # 对于维度为1的情况，简化计算
            log_det_x[:] = np.log(x[0, 0])
            tr_scale_x_inv[:] = C[0, 0]**2 / x[0, 0]

        # 计算逆 Wishart 概率密度函数的对数
        out = ((0.5 * df * log_det_scale - 0.5 * tr_scale_x_inv) -
               (0.5 * df * dim * _LOG_2 + 0.5 * (df + dim + 1) * log_det_x) -
               multigammaln(0.5*df, dim))

        return out
    # 计算逆 Wishart 分布的对数概率密度函数（PDF）。

    Parameters
    ----------
    x : array_like
        分位数，其中 `x` 的最后一个轴表示组成部分。
        每个分位数必须是对称正定矩阵。
    %(_doc_default_callparams)s

    Returns
    -------
    pdf : ndarray
        在 `x` 处评估的对数概率密度函数

    Notes
    -----
    %(_doc_callparams_note)s
    """
    # 处理参数，获取维度、自由度（df）、比例尺度（scale）
    dim, df, scale = self._process_parameters(df, scale)
    # 处理分位数 `x`，确保其符合要求
    x = self._process_quantiles(x, dim)
    # 对比例尺度矩阵进行 Cholesky 分解，同时计算其对数行列式
    C, log_det_scale = self._cholesky_logdet(scale)
    # 调用内部方法计算对数概率密度函数
    out = self._logpdf(x, dim, df, log_det_scale, C)
    # 压缩输出结果，去除多余的维度
    return _squeeze_output(out)

    # 计算逆 Wishart 分布的概率密度函数。

    Parameters
    ----------
    x : array_like
        分位数，其中 `x` 的最后一个轴表示组成部分。
        每个分位数必须是对称正定矩阵。
    %(_doc_default_callparams)s

    Returns
    -------
    pdf : ndarray
        在 `x` 处评估的概率密度函数

    Notes
    -----
    %(_doc_callparams_note)s
    """
    # 返回逆 Wishart 分布的对数概率密度函数的指数形式，即概率密度函数
    return np.exp(self.logpdf(x, df, scale))

    # 计算逆 Wishart 分布的均值。

    Parameters
    ----------
    dim : int
        比例矩阵的维度
    %(_doc_default_callparams)s

    Notes
    -----
    由于此函数不进行参数检查，不应直接调用；应使用 'mean'。

    """
    # 若自由度（df）大于维度加一，计算均值，否则返回 None
    if df > dim + 1:
        out = scale / (df - dim - 1)
    else:
        out = None
    return out

    # 计算逆 Wishart 分布的均值。

    Parameters
    ----------
    %(_doc_default_callparams)s

    Returns
    -------
    mean : float or None
        分布的均值

    """
    # 处理参数，获取维度、自由度（df）、比例尺度（scale）
    dim, df, scale = self._process_parameters(df, scale)
    # 调用内部方法计算均值
    out = self._mean(dim, df, scale)
    # 压缩输出结果，去除多余的维度
    return _squeeze_output(out) if out is not None else out

    # 计算逆 Wishart 分布的众数。

    Parameters
    ----------
    dim : int
        比例矩阵的维度
    %(_doc_default_callparams)s

    Notes
    -----
    由于此函数不进行参数检查，不应直接调用；应使用 'mode'。

    """
    # 计算逆 Wishart 分布的众数
    return scale / (df + dim + 1)
    def mode(self, df, scale):
        """Mode of the inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            模式函数的默认调用参数

        Returns
        -------
        mode : float
            The Mode of the distribution
            分布的模式值

        """
        dim, df, scale = self._process_parameters(df, scale)
        # 调用内部方法处理参数，获取维度、自由度和缩放矩阵
        out = self._mode(dim, df, scale)
        # 调用内部方法计算模式值
        return _squeeze_output(out)

    def _var(self, dim, df, scale):
        """Variance of the inverse Wishart distribution.

        Parameters
        ----------
        dim : int
            Dimension of the scale matrix
            缩放矩阵的维度
        %(_doc_default_callparams)s
            模式函数的默认调用参数

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'var' instead.
        由于此函数不执行参数检查，不应直接调用；应使用 'var' 替代。

        """
        if df > dim + 3:
            var = (df - dim + 1) * scale**2
            diag = scale.diagonal()  # 1 x dim array
            var += (df - dim - 1) * np.outer(diag, diag)
            var /= (df - dim) * (df - dim - 1)**2 * (df - dim - 3)
        else:
            var = None
        return var

    def var(self, df, scale):
        """Variance of the inverse Wishart distribution.

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus three.

        Parameters
        ----------
        %(_doc_default_callparams)s
            模式函数的默认调用参数

        Returns
        -------
        var : float
            The variance of the distribution
            分布的方差
        """
        dim, df, scale = self._process_parameters(df, scale)
        # 调用内部方法处理参数，获取维度、自由度和缩放矩阵
        out = self._var(dim, df, scale)
        # 返回计算得到的方差值，如果为 None 则返回 None
        return _squeeze_output(out) if out is not None else out
    # 生成 n 个标准化反变量，返回形状为 `shape` + (`dim`, `dim`) 的随机变量数组
    def _inv_standard_rvs(self, n, shape, dim, df, random_state):
        """
        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        A : ndarray
            Random variates of shape (`shape`) + (``dim``, ``dim``).
            Each slice `A[..., :, :]` is lower-triangular, and its
            inverse is the lower Cholesky factor of a draw from
            `invwishart(df, np.eye(dim))`.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        # 初始化一个全零数组，形状为 `shape` + (`dim`, `dim`)
        A = np.zeros(shape + (dim, dim))

        # 对角线以下的元素使用标准正态分布生成随机数
        tri_rows, tri_cols = np.tril_indices(dim, k=-1)
        n_tril = dim * (dim-1) // 2
        A[..., tri_rows, tri_cols] = random_state.normal(
            size=(*shape, n_tril),
        )

        # 对角线上的元素使用卡方分布生成随机数的平方根
        rows = np.arange(dim)
        chi_dfs = (df - dim + 1) + rows
        A[..., rows, rows] = random_state.chisquare(
            df=chi_dfs, size=(*shape, dim),
        )**0.5

        return A
    def _rvs(self, n, shape, dim, df, C, random_state):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        n : integer
            Number of variates to generate
        shape : iterable
            Shape of the variates to generate
        dim : int
            Dimension of the scale matrix
        df : int
            Degrees of freedom
        C : ndarray
            Cholesky factorization of the scale matrix, lower triangular.
        %(_doc_random_state)s

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        # Get random draws A such that inv(A) ~ iW(df, I)
        A = self._inv_standard_rvs(n, shape, dim, df, random_state)

        # Calculate SA = (CA)'^{-1} (CA)^{-1} ~ iW(df, scale)
        trsm = get_blas_funcs(('trsm'), (A,))
        trmm = get_blas_funcs(('trmm'), (A,))

        for index in np.ndindex(A.shape[:-2]):
            if dim > 1:
                # Calculate CA
                # Get CA = C A^{-1} via triangular solver
                CA = trsm(1., A[index], C, side=1, lower=True)
                # get SA
                A[index] = trmm(1., CA, CA, side=1, lower=True, trans_a=True)
            else:
                A[index][0, 0] = (C[0, 0] / A[index][0, 0])**2
                
        return A



    def rvs(self, df, scale, size=1, random_state=None):
        """Draw random samples from an inverse Wishart distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`) + (``dim``, ``dim``), where
            ``dim`` is the dimension of the scale matrix.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        n, shape = self._process_size(size)
        dim, df, scale = self._process_parameters(df, scale)

        # Cholesky decomposition of scale
        C = scipy.linalg.cholesky(scale, lower=True)

        out = self._rvs(n, shape, dim, df, C, random_state)

        return _squeeze_output(out)



    def _entropy(self, dim, df, log_det_scale):
        # reference: eq. (17) from ref. 3
        psi_eval_points = [0.5 * (df - dim + i) for i in range(1, dim + 1)]
        psi_eval_points = np.asarray(psi_eval_points)
        return multigammaln(0.5 * df, dim) + 0.5 * dim * df + \
            0.5 * (dim + 1) * (log_det_scale - _LOG_2) - \
            0.5 * (df + dim + 1) * \
            psi(psi_eval_points, out=psi_eval_points).sum()
    # 定义一个方法 `entropy`，接受参数 `df` 和 `scale`
    def entropy(self, df, scale):
        # 调用 `_process_parameters` 方法处理参数 `df` 和 `scale`，获取返回的结果 (`dim`, `df`, `scale`)
        dim, df, scale = self._process_parameters(df, scale)
        # 调用 `_cholesky_logdet` 方法计算 `scale` 的 Cholesky 分解和对数行列式，并分别赋值给 `_` 和 `log_det_scale`
        _, log_det_scale = self._cholesky_logdet(scale)
        # 调用 `_entropy` 方法计算熵，并返回结果
        return self._entropy(dim, df, log_det_scale)
invwishart = invwishart_gen()

class invwishart_frozen(multi_rv_frozen):
    def __init__(self, df, scale, seed=None):
        """Create a frozen inverse Wishart distribution.

        Parameters
        ----------
        df : array_like
            Degrees of freedom of the distribution
        scale : array_like
            Scale matrix of the distribution
        seed : {None, int, `numpy.random.Generator`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.

        """
        # Initialize with inverse Wishart generator
        self._dist = invwishart_gen(seed)
        # Process and assign parameters for dimension, degrees of freedom, and scale
        self.dim, self.df, self.scale = self._dist._process_parameters(
            df, scale
        )
        
        # Calculate Cholesky decomposition to get determinant of scale matrix
        self.C = scipy.linalg.cholesky(self.scale, lower=True)
        self.log_det_scale = 2 * np.sum(np.log(self.C.diagonal()))

    def logpdf(self, x):
        # Process quantiles and compute log probability density function
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.dim, self.df,
                                 self.log_det_scale, self.C)
        return _squeeze_output(out)

    def pdf(self, x):
        # Compute probability density function by exponentiating logpdf
        return np.exp(self.logpdf(x))

    def mean(self):
        # Compute mean of the distribution
        out = self._dist._mean(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def mode(self):
        # Compute mode of the distribution
        out = self._dist._mode(self.dim, self.df, self.scale)
        return _squeeze_output(out)

    def var(self):
        # Compute variance of the distribution
        out = self._dist._var(self.dim, self.df, self.scale)
        return _squeeze_output(out) if out is not None else out

    def rvs(self, size=1, random_state=None):
        # Generate random variates from the distribution
        n, shape = self._dist._process_size(size)

        out = self._dist._rvs(n, shape, self.dim, self.df,
                              self.C, random_state)

        return _squeeze_output(out)

    def entropy(self):
        # Compute entropy of the distribution
        return self._dist._entropy(self.dim, self.df, self.log_det_scale)


# Set frozen generator docstrings from corresponding docstrings in
# inverse Wishart and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'mean', 'mode', 'var', 'rvs']:
    method = invwishart_gen.__dict__[name]
    method_frozen = invwishart_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, wishart_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, wishart_docdict_params)

_multinomial_doc_default_callparams = """\
n : int
    Number of trials
p : array_like
    Probability of a trial falling into each category; should sum to 1
"""

_multinomial_doc_callparams_note = """\
`n` should be a nonnegative integer. Each element of `p` should be in the
interval :math:`[0,1]` and the elements should sum to 1. If they do not sum to
1, the last element of the `p` array is not used and is replaced with the
ultimate
# 离散多项分布的文档字符串，用于描述多项分布的调用参数和方法
_multinomial_doc_frozen_callparams = ""

# 冻结多项分布参数的文档字符串注释
_multinomial_doc_frozen_callparams_note = """\
查看类定义以获取参数的详细描述。"""

# 多项分布参数字典，包含默认调用参数、调用参数注释和随机状态信息
multinomial_docdict_params = {
    '_doc_default_callparams': _multinomial_doc_default_callparams,
    '_doc_callparams_note': _multinomial_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

# 不带参数的多项分布字典，包含默认调用参数（空字符串）、冻结参数的注释和随机状态信息
multinomial_docdict_noparams = {
    '_doc_default_callparams': _multinomial_doc_frozen_callparams,
    '_doc_callparams_note': _multinomial_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}

# 多项分布的生成器类，继承自多变量随机变量的通用类
class multinomial_gen(multi_rv_generic):
    r"""A multinomial random variable.

    Methods
    -------
    pmf(x, n, p)
        Probability mass function.
    logpmf(x, n, p)
        Log of the probability mass function.
    rvs(n, p, size=1, random_state=None)
        Draw random samples from a multinomial distribution.
    entropy(n, p)
        Compute the entropy of the multinomial distribution.
    cov(n, p)
        Compute the covariance matrix of the multinomial distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_callparams_note)s

    The probability mass function for `multinomial` is

    .. math::

        f(x) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k},

    supported on :math:`x=(x_1, \ldots, x_k)` where each :math:`x_i` is a
    nonnegative integer and their sum is :math:`n`.

    .. versionadded:: 0.19.0

    Examples
    --------

    >>> from scipy.stats import multinomial
    >>> rv = multinomial(8, [0.3, 0.2, 0.5])
    >>> rv.pmf([1, 3, 4])
    0.042000000000000072

    The multinomial distribution for :math:`k=2` is identical to the
    corresponding binomial distribution (tiny numerical differences
    notwithstanding):

    >>> from scipy.stats import binom
    >>> multinomial.pmf([3, 4], n=7, p=[0.4, 0.6])
    0.29030399999999973
    >>> binom.pmf(3, 7, 0.4)
    0.29030400000000012

    The functions ``pmf``, ``logpmf``, ``entropy``, and ``cov`` support
    broadcasting, under the convention that the vector parameters (``x`` and
    ``p``) are interpreted as if each row along the last axis is a single
    object. For instance:

    >>> multinomial.pmf([[3, 4], [3, 5]], n=[7, 8], p=[.3, .7])
    array([0.2268945,  0.25412184])

    Here, ``x.shape == (2, 2)``, ``n.shape == (2,)``, and ``p.shape == (2,)``,
    but following the rules mentioned above they behave as if the rows
    ``[3, 4]`` and ``[3, 5]`` in ``x`` and ``[.3, .7]`` in ``p`` were a single
    object, and as if we had ``x.shape = (2,)``, ``n.shape = (2,)``, and
    ``p.shape = ()``. To obtain the individual elements without broadcasting,
    we would do this:

    >>> multinomial.pmf([3, 4], n=7, p=[.3, .7])
    0.2268945
    >>> multinomial.pmf([3, 5], 8, p=[.3, .7])
    0.25412184
    """
    This broadcasting also works for ``cov``, where the output objects are
    square matrices of size ``p.shape[-1]``. For example:

    >>> multinomial.cov([4, 5], [[.3, .7], [.4, .6]])
    array([[[ 0.84, -0.84],
            [-0.84,  0.84]],
           [[ 1.2 , -1.2 ],
            [-1.2 ,  1.2 ]]])

    In this example, ``n.shape == (2,)`` and ``p.shape == (2, 2)``, and
    following the rules above, these broadcast as if ``p.shape == (2,)``.
    Thus the result should also be of shape ``(2,)``, but since each output is
    a :math:`2 \times 2` matrix, the result in fact has shape ``(2, 2, 2)``,
    where ``result[0]`` is equal to ``multinomial.cov(n=4, p=[.3, .7])`` and
    ``result[1]`` is equal to ``multinomial.cov(n=5, p=[.4, .6])``.

    Alternatively, the object may be called (as a function) to fix the `n` and
    `p` parameters, returning a "frozen" multinomial random variable:

    >>> rv = multinomial(n=7, p=[.3, .7])
    >>> # Frozen object with the same methods but holding the given
    >>> # degrees of freedom and scale fixed.

    See also
    --------
    scipy.stats.binom : The binomial distribution.
    numpy.random.Generator.multinomial : Sampling from the multinomial distribution.
    scipy.stats.multivariate_hypergeom :
        The multivariate hypergeometric distribution.
    """

    def __init__(self, seed=None):
        """
        Initialize the multinomial distribution with an optional seed.

        Parameters
        ----------
        seed : int or None, optional
            Seed value for random number generation.

        Returns
        -------
        None
        """
        super().__init__(seed)
        # Set the documentation string using a formatted docstring
        self.__doc__ = \
            doccer.docformat(self.__doc__, multinomial_docdict_params)

    def __call__(self, n, p, seed=None):
        """
        Create a frozen multinomial distribution.

        This method fixes the `n` and `p` parameters, returning a frozen
        multinomial random variable.

        Parameters
        ----------
        n : int or array_like
            Number of trials.
        p : float or array_like
            Probabilities of each trial.
        seed : int or None, optional
            Seed value for random number generation.

        Returns
        -------
        rv : multinomial_frozen object
            Frozen multinomial random variable with fixed parameters `n` and `p`.
        """
        return multinomial_frozen(n, p, seed)

    def _process_parameters(self, n, p, eps=1e-15):
        """
        Process and validate the parameters `n` and `p` for the multinomial distribution.

        Parameters
        ----------
        n : int or array_like
            Number of trials.
        p : float or array_like
            Probabilities of each trial.
        eps : float, optional
            Small value to check adjustments.

        Returns
        -------
        n_ : ndarray
            Validated and processed `n` array.
        p_ : ndarray
            Validated and processed `p` array.
        npcond : ndarray of bool
            Conditions array indicating invalid parameter values.
        """
        # Convert `p` to a numpy array of float64 type
        p = np.array(p, dtype=np.float64, copy=True)
        # Adjust probabilities to ensure they sum up to 1
        p_adjusted = 1. - p[..., :-1].sum(axis=-1)
        i_adjusted = np.abs(p_adjusted) > eps
        p[i_adjusted, -1] = p_adjusted[i_adjusted]

        # Check conditions for bad probabilities (`p` values)
        pcond = np.any(p < 0, axis=-1)
        pcond |= np.any(p > 1, axis=-1)

        # Convert `n` to an integer numpy array
        n = np.array(n, dtype=int, copy=True)

        # Check conditions for bad number of trials (`n` values)
        ncond = n < 0

        # Combine conditions for `n` and `p`
        return n, p, ncond | pcond
    def _process_quantiles(self, x, n, p):
        """Returns: x_, xcond.

        x_ is an int array; xcond is a boolean array flagging values out of the
        domain.
        """
        # 将输入的 x 转换为整数类型的数组 xx
        xx = np.asarray(x, dtype=int)

        # 检查 xx 的维度是否为 0，如果是，则抛出数值错误异常
        if xx.ndim == 0:
            raise ValueError("x must be an array.")

        # 检查 xx 的大小是否与 p 的大小一致，如果不一致，则抛出数值错误异常
        if xx.size != 0 and not xx.shape[-1] == p.shape[-1]:
            raise ValueError("Size of each quantile should be size of p: "
                             "received %d, but expected %d." %
                             (xx.shape[-1], p.shape[-1]))

        # 对超出定义域的 x 值进行标记，返回布尔类型的条件数组
        cond = np.any(xx != x, axis=-1)
        cond |= np.any(xx < 0, axis=-1)
        cond = cond | (np.sum(xx, axis=-1) != n)

        # 返回处理后的 xx 数组和条件数组 cond
        return xx, cond

    def _checkresult(self, result, cond, bad_value):
        # 将 result 转换为 numpy 数组
        result = np.asarray(result)

        # 根据 cond 的维度进行条件判断和赋值
        if cond.ndim != 0:
            result[cond] = bad_value
        elif cond:
            if result.ndim == 0:
                return bad_value
            result[...] = bad_value
        return result

    def _logpmf(self, x, n, p):
        # 计算多项式分布的对数概率质量函数
        return gammaln(n+1) + np.sum(xlogy(x, p) - gammaln(x+1), axis=-1)

    def logpmf(self, x, n, p):
        """Log of the Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        # 处理参数 n 和 p，得到处理后的参数和条件数组
        n, p, npcond = self._process_parameters(n, p)
        x, xcond = self._process_quantiles(x, n, p)

        # 计算对数概率质量函数的结果
        result = self._logpmf(x, n, p)

        # 替换超出定义域的 x 值对应的结果为 -∞，使用广播使 xcond 与结果的形状相匹配
        xcond_ = xcond | np.zeros(npcond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, -np.inf)

        # 替换对 n 或 p 不合适的值对应的结果为 NaN，使用广播使 npcond 与 xcond 的形状相匹配
        npcond_ = npcond | np.zeros(xcond.shape, dtype=np.bool_)
        return self._checkresult(result, npcond_, np.nan)

    def pmf(self, x, n, p):
        """Multinomial probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability density function evaluated at `x`

        Notes
        -----
        %(_doc_callparams_note)s
        """
        # 返回多项式分布的概率质量函数，即通过 logpmf 计算后取指数得到
        return np.exp(self.logpmf(x, n, p))
    def mean(self, n, p):
        """Mean of the Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            参数说明: 从 _doc_default_callparams 中获取的参数说明

        Returns
        -------
        mean : float
            The mean of the distribution
            分布的均值
        """
        # 处理参数并返回处理后的参数及相应条件
        n, p, npcond = self._process_parameters(n, p)
        # 计算均值结果
        result = n[..., np.newaxis]*p
        # 检查结果并返回，处理 NaN 值
        return self._checkresult(result, npcond, np.nan)

    def cov(self, n, p):
        """Covariance matrix of the multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            参数说明: 从 _doc_default_callparams 中获取的参数说明

        Returns
        -------
        cov : ndarray
            The covariance matrix of the distribution
            分布的协方差矩阵
        """
        # 处理参数并返回处理后的参数及相应条件
        n, p, npcond = self._process_parameters(n, p)

        # 构造 n 的新形状，以进行后续计算
        nn = n[..., np.newaxis, np.newaxis]
        # 计算协方差矩阵结果
        result = nn * np.einsum('...j,...k->...jk', -p, p)

        # 调整对角线元素
        for i in range(p.shape[-1]):
            result[..., i, i] += n*p[..., i]

        # 检查结果并返回，处理 NaN 值
        return self._checkresult(result, npcond, np.nan)

    def entropy(self, n, p):
        r"""Compute the entropy of the multinomial distribution.

        The entropy is computed using this expression:

        .. math::

            f(x) = - \log n! - n\sum_{i=1}^k p_i \log p_i +
            \sum_{i=1}^k \sum_{x=0}^n \binom n x p_i^x(1-p_i)^{n-x} \log x!

        Parameters
        ----------
        %(_doc_default_callparams)s
            参数说明: 从 _doc_default_callparams 中获取的参数说明

        Returns
        -------
        h : scalar
            Entropy of the Multinomial distribution
            多项式分布的熵

        Notes
        -----
        %(_doc_callparams_note)s
            注意事项: 从 _doc_callparams_note 中获取的注意事项
        """
        # 处理参数并返回处理后的参数及相应条件
        n, p, npcond = self._process_parameters(n, p)

        # 构造 x 的取值范围
        x = np.r_[1:np.max(n)+1]

        # 计算熵的第一项
        term1 = n*np.sum(entr(p), axis=-1)
        term1 -= gammaln(n+1)

        # 调整 n 的形状以便进行计算
        n = n[..., np.newaxis]
        new_axes_needed = max(p.ndim, n.ndim) - x.ndim + 1
        x.shape += (1,)*new_axes_needed

        # 计算熵的第二项
        term2 = np.sum(binom.pmf(x, n, p)*gammaln(x+1),
                       axis=(-1, -1-new_axes_needed))

        # 检查结果并返回，处理 NaN 值
        return self._checkresult(term1 + term2, npcond, np.nan)

    def rvs(self, n, p, size=None, random_state=None):
        """Draw random samples from a Multinomial distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            参数说明: 从 _doc_default_callparams 中获取的参数说明
        size : integer or iterable of integers, optional
            Number of samples to draw (default 1).
            抽样数量：要抽样的样本数量（默认为1）
        %(_doc_random_state)s
            随机状态: 从 _doc_random_state 中获取的随机状态说明

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of shape (`size`, `len(p)`)
            形状为 (`size`, `len(p)`) 的随机变量

        Notes
        -----
        %(_doc_callparams_note)s
            注意事项: 从 _doc_callparams_note 中获取的注意事项
        """
        # 处理参数并返回处理后的参数及相应条件
        n, p, npcond = self._process_parameters(n, p)
        # 获取随机状态
        random_state = self._get_random_state(random_state)
        # 从多项式分布中抽取随机样本
        return random_state.multinomial(n, p, size)
# 生成一个多项式分布的随机变量生成器对象
multinomial = multinomial_gen()

# 定义一个冻结的多项式分布类，继承自多维随机变量的冻结类
class multinomial_frozen(multi_rv_frozen):
    r"""Create a frozen Multinomial distribution.

    Parameters
    ----------
    n : int
        number of trials
    p: array_like
        probability of a trial falling into each category; should sum to 1
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    """
    
    # 初始化方法，创建一个多项式分布的冻结实例
    def __init__(self, n, p, seed=None):
        # 使用多项式分布生成器创建一个实例，保存在 self._dist 属性中
        self._dist = multinomial_gen(seed)
        # 对 n 和 p 进行处理，返回处理后的参数
        self.n, self.p, self.npcond = self._dist._process_parameters(n, p)

        # 通过 monkey patching 修改 self._dist 的 _process_parameters 方法
        def _process_parameters(n, p):
            return self.n, self.p, self.npcond

        self._dist._process_parameters = _process_parameters

    # 返回给定值 x 的对数概率质量函数值
    def logpmf(self, x):
        return self._dist.logpmf(x, self.n, self.p)

    # 返回给定值 x 的概率质量函数值
    def pmf(self, x):
        return self._dist.pmf(x, self.n, self.p)

    # 返回分布的均值
    def mean(self):
        return self._dist.mean(self.n, self.p)

    # 返回分布的协方差矩阵
    def cov(self):
        return self._dist.cov(self.n, self.p)

    # 返回分布的熵
    def entropy(self):
        return self._dist.entropy(self.n, self.p)

    # 从分布中生成随机样本
    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.n, self.p, size, random_state)


# 设置冻结生成器的文档字符串，从多项式分布中相应的文档字符串中获取信息，并填充类文档字符串中的默认字符串
for name in ['logpmf', 'pmf', 'mean', 'cov', 'rvs']:
    method = multinomial_gen.__dict__[name]
    method_frozen = multinomial_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, multinomial_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__,
                                      multinomial_docdict_params)


# 定义一个特殊正交群随机变量生成器类，继承自多维随机变量的通用生成器类
class special_ortho_group_gen(multi_rv_generic):
    r"""A Special Orthogonal matrix (SO(N)) random variable.

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(N)) with a determinant of +1.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from SO(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    # 这个类封装了 MDP Toolkit 中的 random_rot 代码，参见 https://github.com/mdp-toolkit/mdp-toolkit

    class special_ortho_group:
        """
        Return a random rotation matrix, drawn from the Haar distribution
        (the only uniform distribution on SO(N)).
        The algorithm is described in the paper
        Stewart, G.W., "The efficient generation of random orthogonal
        matrices with an application to condition estimators", SIAM Journal
        on Numerical Analysis, 17(3), pp. 403-409, 1980.
        For more information see
        https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization

        See also the similar `ortho_group`. For a random rotation in three
        dimensions, see `scipy.spatial.transform.Rotation.random`.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import special_ortho_group
        >>> x = special_ortho_group.rvs(3)

        >>> np.dot(x, x.T)
        array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
               [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
               [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

        >>> import scipy.linalg
        >>> scipy.linalg.det(x)
        1.0

        This generates one random matrix from SO(3). It is orthogonal and
        has a determinant of 1.

        Alternatively, the object may be called (as a function) to fix the `dim`
        parameter, returning a "frozen" special_ortho_group random variable:

        >>> rv = special_ortho_group(5)
        >>> # Frozen object with the same methods but holding the
        >>> # dimension parameter fixed.

        See Also
        --------
        ortho_group, scipy.spatial.transform.Rotation.random
        """

        def __init__(self, seed=None):
            super().__init__(seed)
            # 使用 doccer.docformat 方法格式化文档字符串
            self.__doc__ = doccer.docformat(self.__doc__)

        def __call__(self, dim=None, seed=None):
            """Create a frozen SO(N) distribution.

            See `special_ortho_group_frozen` for more information.
            """
            # 返回一个冻结的 special_ortho_group 随机变量，固定维度参数 dim
            return special_ortho_group_frozen(dim, seed=seed)

        def _process_parameters(self, dim):
            """Dimension N must be specified; it cannot be inferred."""
            # 处理参数函数，确保维度 dim 被指定，且为大于1的标量整数
            if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
                raise ValueError("""Dimension of rotation must be specified,
                                    and must be a scalar greater than 1.""")

            return dim
    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from SO(N).

        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        # 获取随机数生成器对象
        random_state = self._get_random_state(random_state)

        # 确保 size 是整数，并且转换成元组形式
        size = int(size)
        size = (size,) if size > 1 else ()

        # 处理 dim 参数，确保其有效性
        dim = self._process_parameters(dim)

        # 初始化一个空的数组 H，其形状为 (size, dim, dim)，并将其设为单位矩阵
        H = np.empty(size + (dim, dim))
        H[..., :, :] = np.eye(dim)

        # 初始化一个空的数组 D，其形状为 (size, dim)，用于存放对角元素
        D = np.empty(size + (dim,))

        # 循环生成 Householder 变换的参数
        for n in range(dim-1):

            # 生成随机向量 x，长度为 dim-n
            x = random_state.normal(size=size + (dim-n,))
            xrow = x[..., None, :]  # 将 x 视为行向量
            xcol = x[..., :, None]  # 将 x 视为列向量

            # 计算向量 x 的二范数的平方
            norm2 = np.matmul(xrow, xcol).squeeze((-2, -1))

            # 处理向量 x 的首元素，用于生成对角矩阵 D
            x0 = x[..., 0].copy()
            D[..., n] = np.where(x0 != 0, np.sign(x0), 1)
            x[..., 0] += D[..., n]*np.sqrt(norm2)

            # 对向量 x 进行归一化处理
            x /= np.sqrt((norm2 - x0**2 + x[..., 0]**2) / 2.)[..., None]

            # 对 H 进行 Householder 变换的更新
            H[..., :, n:] -= np.matmul(H[..., :, n:], xcol) * xrow

        # 完成对角矩阵 D 的最后一个元素的赋值
        D[..., -1] = (-1)**(dim-1)*D[..., :-1].prod(axis=-1)

        # 将 H 乘以对角矩阵 D，完成最终的旋转矩阵生成
        H *= D[..., :, None]
        return H
# 创建一个特殊正交群的生成器对象，用于生成特殊正交群中的随机矩阵
special_ortho_group = special_ortho_group_gen()

# 定义一个特殊正交群的冻结分布类，继承自多变量随机变量的冻结类
class special_ortho_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen SO(N) distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import special_ortho_group
        >>> g = special_ortho_group(5)
        >>> x = g.rvs()

        """ # numpy/numpydoc#87  # noqa: E501
        # 初始化特殊正交群生成器对象，并设定随机数种子
        self._dist = special_ortho_group_gen(seed)
        # 处理并保存维度参数
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        # 调用特殊正交群生成器对象的随机变量生成方法
        return self._dist.rvs(self.dim, size, random_state)


class ortho_group_gen(multi_rv_generic):
    r"""An Orthogonal matrix (O(N)) random variable.

    Return a random orthogonal matrix, drawn from the O(N) Haar
    distribution (the only uniform distribution on O(N)).

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from O(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is closely related to `special_ortho_group`.

    Some care is taken to avoid numerical error, as per the paper by Mezzadri.

    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
           compact groups", :arXiv:`math-ph/0609050v2`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> x = ortho_group.rvs(3)

    >>> np.dot(x, x.T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    >>> import scipy.linalg
    >>> np.fabs(scipy.linalg.det(x))
    1.0

    This generates one random matrix from O(3). It is orthogonal and
    has a determinant of +1 or -1.

    Alternatively, the object may be called (as a function) to fix the `dim`
    """
    pass  # 这里是正交矩阵生成器类的定义，暂无额外的代码逻辑
    def __init__(self, seed=None):
        super().__init__(seed)
        # 将对象的文档字符串格式化为适当的格式
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen O(N) distribution.

        See `ortho_group_frozen` for more information.
        """
        # 返回一个冻结的 O(N) 分布对象
        return ortho_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        # 检查并处理维度参数
        if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
            raise ValueError("Dimension of rotation must be specified,"
                             "and must be a scalar greater than 1.")

        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from O(N).

        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        # 获取随机状态对象
        random_state = self._get_random_state(random_state)

        size = int(size)

        # 处理维度参数
        dim = self._process_parameters(dim)

        # 设置返回值的形状
        size = (size,) if size > 1 else ()

        # 生成正态分布随机矩阵
        z = random_state.normal(size=size + (dim, dim))

        # 计算 QR 分解
        q, r = np.linalg.qr(z)

        # 最后两个维度是 R 矩阵的行和列，提取对角线元素
        d = r.diagonal(offset=0, axis1=-2, axis2=-1)

        # 通过广播将每个 R 矩阵的每一行除以 R 矩阵的对角线
        q *= (d/abs(d))[..., np.newaxis, :]  # 以正确的方式广播

        return q
ortho_group = ortho_group_gen()

class ortho_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen O(N) distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import ortho_group
        >>> g = ortho_group(5)
        >>> x = g.rvs()

        """
        # 使用 ortho_group_gen 创建一个冻结的 O(N) 分布对象
        self._dist = ortho_group_gen(seed)
        # 处理并设置参数 dim
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        # 调用 _dist 对象的 rvs 方法生成随机变量
        return self._dist.rvs(self.dim, size, random_state)


class random_correlation_gen(multi_rv_generic):
    r"""A random correlation matrix.

    Return a random correlation matrix, given a vector of eigenvalues.

    The `eigs` keyword specifies the eigenvalues of the correlation matrix,
    and implies the dimension.

    Methods
    -------
    rvs(eigs=None, random_state=None)
        Draw random correlation matrices, all with eigenvalues eigs.

    Parameters
    ----------
    eigs : 1d ndarray
        Eigenvalues of correlation matrix
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance
        then that instance is used.
    tol : float, optional
        Tolerance for input parameter checks
    diag_tol : float, optional
        Tolerance for deviation of the diagonal of the resulting
        matrix. Default: 1e-7

    Raises
    ------
    RuntimeError
        Floating point error prevented generating a valid correlation
        matrix.

    Returns
    -------
    rvs : ndarray or scalar
        Random size N-dimensional matrices, dimension (size, dim, dim),
        each having eigenvalues eigs.

    Notes
    -----

    Generates a random correlation matrix following a numerically stable
    algorithm spelled out by Davies & Higham. This algorithm uses a single O(N)
    similarity transformation to construct a symmetric positive semi-definite
    matrix, and applies a series of Givens rotations to scale it to have ones
    on the diagonal.

    References
    ----------
    """
    Computes a 2x2 Givens matrix to put 1's on the diagonal.

    The input matrix is a 2x2 symmetric matrix M = [ aii aij ; aij ajj ].

    The output matrix g is a 2x2 anti-symmetric matrix of the form
    [ c s ; -s c ];  the elements c and s are returned.

    Applying the output matrix to the input matrix (as b=g.T M g)
    results in a matrix with bii=1, provided tr(M) - det(M) >= 1
    and floating point issues do not occur. Otherwise, some other
    valid rotation is returned. When tr(M)==2, also bjj=1.
    """
def _givens_to_1(self, aii, ajj, aij):
    # 对称矩阵 M = [ aii aij ; aij ajj ] 的特征值做处理，如果 ajj==1，则交换 aii 和 ajj 以避免除零错误
    aiid = aii - 1.
    ajjd = ajj - 1.

    if ajjd == 0:
        # 如果 ajj==1，则交换 aii 和 ajj 以避免除零错误
        return 0., 1.

    # 计算对角线元素之间的差值
    dd = math.sqrt(max(aij**2 - aiid*ajjd, 0))

    # 选择 t 的值以避免数值问题 [1]
    t = (aij + math.copysign(dd, aij)) / ajjd
    # 计算旋转矩阵的元素 c 和 s
    c = 1. / math.sqrt(1. + t*t)
    if c == 0:
        # 避免下溢
        s = 1.0
    else:
        s = c*t
    # 返回计算出的旋转矩阵的元素 c 和 s
    return c, s
    def _to_corr(self, m):
        """
        Given a psd matrix m, rotate to put one's on the diagonal, turning it
        into a correlation matrix.  This also requires the trace equal the
        dimensionality. Note: modifies input matrix
        """
        # 检查是否符合进行原地 Givens 变换的要求
        if not (m.flags.c_contiguous and m.dtype == np.float64 and
                m.shape[0] == m.shape[1]):
            raise ValueError()

        d = m.shape[0]  # 获取矩阵的维度
        for i in range(d-1):
            if m[i, i] == 1:
                continue  # 如果对角线元素已经是1，跳过
            elif m[i, i] > 1:
                for j in range(i+1, d):
                    if m[j, j] < 1:
                        break  # 找到第一个小于1的对角元素
            else:
                for j in range(i+1, d):
                    if m[j, j] > 1:
                        break  # 找到第一个大于1的对角元素

            c, s = self._givens_to_1(m[i, i], m[j, j], m[i, j])

            # 使用 BLAS 库在原地应用 Givens 旋转。等价于以下注释中的代码：
            # g = np.eye(d)
            # g[i, i] = g[j,j] = c
            # g[j, i] = -s; g[i, j] = s
            # m = np.dot(g.T, np.dot(m, g))
            mv = m.ravel()
            drot(mv, mv, c, -s, n=d,
                 offx=i*d, incx=1, offy=j*d, incy=1,
                 overwrite_x=True, overwrite_y=True)
            drot(mv, mv, c, -s, n=d,
                 offx=i, incx=d, offy=j, incy=d,
                 overwrite_x=True, overwrite_y=True)

        return m

    def rvs(self, eigs, random_state=None, tol=1e-13, diag_tol=1e-7):
        """Draw random correlation matrices.

        Parameters
        ----------
        eigs : 1d ndarray
            Eigenvalues of correlation matrix
        tol : float, optional
            Tolerance for input parameter checks
        diag_tol : float, optional
            Tolerance for deviation of the diagonal of the resulting
            matrix. Default: 1e-7

        Raises
        ------
        RuntimeError
            Floating point error prevented generating a valid correlation
            matrix.

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim),
            each having eigenvalues eigs.

        """
        dim, eigs = self._process_parameters(eigs, tol=tol)

        random_state = self._get_random_state(random_state)

        m = ortho_group.rvs(dim, random_state=random_state)
        m = np.dot(np.dot(m, np.diag(eigs)), m.T)  # 设置 m 的迹
        m = self._to_corr(m)  # 将 m 调整为单位对角线的相关矩阵

        # 检查对角线元素
        if abs(m.diagonal() - 1).max() > diag_tol:
            raise RuntimeError("Failed to generate a valid correlation matrix")

        return m
random_correlation = random_correlation_gen()
# 调用 random_correlation_gen() 函数生成随机相关矩阵

class random_correlation_frozen(multi_rv_frozen):
    def __init__(self, eigs, seed=None, tol=1e-13, diag_tol=1e-7):
        """Create a frozen random correlation matrix distribution.

        Parameters
        ----------
        eigs : 1d ndarray
            Eigenvalues of correlation matrix
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.
        tol : float, optional
            Tolerance for input parameter checks
        diag_tol : float, optional
            Tolerance for deviation of the diagonal of the resulting
            matrix. Default: 1e-7

        Raises
        ------
        RuntimeError
            Floating point error prevented generating a valid correlation
            matrix.

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim),
            each having eigenvalues eigs.
        """ # numpy/numpydoc#87  # noqa: E501
        # 初始化一个随机相关矩阵分布对象

        self._dist = random_correlation_gen(seed)
        # 使用 random_correlation_gen(seed) 创建随机相关矩阵生成器对象并保存在 self._dist 中
        self.tol = tol
        # 设置容差值 tol
        self.diag_tol = diag_tol
        # 设置对角线容差值 diag_tol
        _, self.eigs = self._dist._process_parameters(eigs, tol=self.tol)
        # 调用 self._dist._process_parameters(eigs, tol=self.tol) 处理参数 eigs 并保存结果到 self.eigs

    def rvs(self, random_state=None):
        # 定义随机变量样本生成方法 rvs
        return self._dist.rvs(self.eigs, random_state=random_state,
                              tol=self.tol, diag_tol=self.diag_tol)


class unitary_group_gen(multi_rv_generic):
    r"""A matrix-valued U(N) random variable.

    Return a random unitary matrix.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from U(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices, must be greater than 1.
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is similar to `ortho_group`.

    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
           compact groups", :arXiv:`math-ph/0609050v2`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(3)

    >>> np.dot(x, x.conj().T)
    """
    # 定义一个返回随机酉矩阵的类 unitary_group_gen，继承自 multi_rv_generic
    # 提供方法 rvs(dim=None, size=1, random_state=None)，用于从 U(N) 中抽取随机样本
    # 包含参数 dim 和 seed，以及类的说明和示例
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])
    
    This represents a 3x3 array, likely a unitary matrix generated randomly from U(3).

    The dot product confirms its unitarity up to machine precision.

    Alternatively, this object can be called as a function with the `dim` parameter fixed,
    returning a frozen random variable from the unitary_group:

    >>> rv = unitary_group(5)

    See Also
    --------
    ortho_group
    """
    
    class unitary_group(object):
        """Class for generating random unitary matrices.
    
        Parameters
        ----------
        seed : int or None, optional
            Seed for random number generator.
        """
    
        def __init__(self, seed=None):
            super().__init__(seed)
            self.__doc__ = doccer.docformat(self.__doc__)
    
        def __call__(self, dim=None, seed=None):
            """Create a frozen (U(N)) n-dimensional unitary matrix distribution.
    
            See `unitary_group_frozen` for more information.
            """
            return unitary_group_frozen(dim, seed=seed)
    
        def _process_parameters(self, dim):
            """Dimension N must be specified; it cannot be inferred."""
            if dim is None or not np.isscalar(dim) or dim <= 1 or dim != int(dim):
                raise ValueError("Dimension of rotation must be specified,"
                                 "and must be a scalar greater than 1.")
    
            return dim
    
        def rvs(self, dim, size=1, random_state=None):
            """Draw random samples from U(N).
    
            Parameters
            ----------
            dim : integer
                Dimension of space (N).
            size : integer, optional
                Number of samples to draw (default 1).
    
            Returns
            -------
            rvs : ndarray or scalar
                Random size N-dimensional matrices, dimension (size, dim, dim)
    
            """
            random_state = self._get_random_state(random_state)
    
            size = int(size)
    
            dim = self._process_parameters(dim)
    
            size = (size,) if size > 1 else ()
            z = 1/math.sqrt(2)*(random_state.normal(size=size + (dim, dim)) +
                                1j*random_state.normal(size=size + (dim, dim)))
            q, r = np.linalg.qr(z)
            # The last two dimensions are the rows and columns of R matrices.
            # Extract the diagonals. Note that this eliminates a dimension.
            d = r.diagonal(offset=0, axis1=-2, axis2=-1)
            # Add back a dimension for proper broadcasting: we're dividing
            # each row of each R matrix by the diagonal of the R matrix.
            q *= (d/abs(d))[..., np.newaxis, :]  # to broadcast properly
            return q
# 使用 unitary_group_gen() 函数生成一个单射群对象
unitary_group = unitary_group_gen()

# 定义 unitary_group_frozen 类，继承自 multi_rv_frozen 类
class unitary_group_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen (U(N)) n-dimensional unitary matrix distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import unitary_group
        >>> x = unitary_group(3)
        >>> x.rvs()

        """ # numpy/numpydoc#87  # noqa: E501
        # 使用 unitary_group_gen() 函数创建一个冻结的单射群分布对象
        self._dist = unitary_group_gen(seed)
        # 处理维度参数并赋值给实例变量 dim
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        # 调用 _dist 对象的 rvs 方法生成随机样本数据
        return self._dist.rvs(self.dim, size, random_state)


# 多变量 t 分布的默认调用参数字符串文档
_mvt_doc_default_callparams = """\
loc : array_like, optional
    Location of the distribution. (default ``0``)
shape : array_like, optional
    Positive semidefinite matrix of the distribution. (default ``1``)
df : float, optional
    Degrees of freedom of the distribution; must be greater than zero.
    If ``np.inf`` then results are multivariate normal. The default is ``1``.
allow_singular : bool, optional
    Whether to allow a singular matrix. (default ``False``)
"""

# 多变量 t 分布调用参数的注意事项字符串文档
_mvt_doc_callparams_note = """\
Setting the parameter `loc` to ``None`` is equivalent to having `loc`
be the zero-vector. The parameter `shape` can be a scalar, in which case
the shape matrix is the identity times that value, a vector of
diagonal entries for the shape matrix, or a two-dimensional array_like.
"""

# 冻结多变量 t 分布参数的注意事项字符串文档
_mvt_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

# 多变量 t 分布的参数文档字典
mvt_docdict_params = {
    '_mvt_doc_default_callparams': _mvt_doc_default_callparams,
    '_mvt_doc_callparams_note': _mvt_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

# 没有参数的多变量 t 分布的参数文档字典
mvt_docdict_noparams = {
    '_mvt_doc_default_callparams': "",
    '_mvt_doc_callparams_note': _mvt_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


# 定义 multivariate_t_gen 类，继承自 multi_rv_generic 类
class multivariate_t_gen(multi_rv_generic):
    r"""A multivariate t-distributed random variable.

    The `loc` parameter specifies the location. The `shape` parameter specifies
    the positive semidefinite shape matrix. The `df` parameter specifies the
    degrees of freedom.

    In addition to calling the methods below, the object itself may be called
    as a function to fix the location, shape matrix, and degrees of freedom
    parameters, returning a "frozen" multivariate t-distribution random.

    Methods
    -------
    # 定义一个多变量 t-分布的随机变量类
    def pdf(x, loc=None, shape=1, df=1, allow_singular=False)
        # 概率密度函数
    logpdf(x, loc=None, shape=1, df=1, allow_singular=False)
        # 概率密度函数的对数
    cdf(x, loc=None, shape=1, df=1, allow_singular=False, *,
        maxpts=None, lower_limit=None, random_state=None)
        # 累积分布函数
    rvs(loc=None, shape=1, df=1, size=1, random_state=None)
        # 从多变量 t-分布中抽取随机样本
    entropy(loc=None, shape=1, df=1)
        # 多变量 t-分布的微分熵

    Parameters
    ----------
    %(_mvt_doc_default_callparams)s
        # 默认调用参数
    %(_doc_random_state)s
        # 随机状态的文档说明

    Notes
    -----
    %(_mvt_doc_callparams_note)s
        # 调用参数的注意事项说明
    The matrix `shape` must be a (symmetric) positive semidefinite matrix. The
    determinant and inverse of `shape` are computed as the pseudo-determinant
    and pseudo-inverse, respectively, so that `shape` does not need to have
    full rank.
        # shape 矩阵必须是（对称的）半正定矩阵。shape 的行列式和逆被计算为伪行列式和伪逆，因此 shape 不需要是满秩的。

    The probability density function for `multivariate_t` is

    .. math::

        f(x) = \frac{\Gamma((\nu + p)/2)}{\Gamma(\nu/2)\nu^{p/2}\pi^{p/2}|\Sigma|^{1/2}}
               \left[1 + \frac{1}{\nu} (\mathbf{x} - \boldsymbol{\mu})^{\top}
               \boldsymbol{\Sigma}^{-1}
               (\mathbf{x} - \boldsymbol{\mu}) \right]^{-(\nu + p)/2},

    where :math:`p` is the dimension of :math:`\mathbf{x}`,
    :math:`\boldsymbol{\mu}` is the :math:`p`-dimensional location,
    :math:`\boldsymbol{\Sigma}` the :math:`p \times p`-dimensional shape
    matrix, and :math:`\nu` is the degrees of freedom.
        # 多变量 t-分布的概率密度函数公式及其参数说明

    .. versionadded:: 1.6.0
        # 添加于版本 1.6.0

    References
    ----------
    .. [1] Arellano-Valle et al. "Shannon Entropy and Mutual Information for
           Multivariate Skew-Elliptical Distributions". Scandinavian Journal
           of Statistics. Vol. 40, issue 1.
        # 参考文献

    Examples
    --------
    The object may be called (as a function) to fix the `loc`, `shape`,
    `df`, and `allow_singular` parameters, returning a "frozen"
    multivariate_t random variable:

    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> rv = multivariate_t([1.0, -0.5], [[2.1, 0.3], [0.3, 1.5]], df=2)
    >>> # Frozen object with the same methods but holding the given location,
    >>> # scale, and degrees of freedom fixed.
        # 创建冻结的多变量 t-分布随机变量对象，并固定给定的 loc、shape、df 和 allow_singular 参数

    Create a contour plot of the PDF.

    >>> import matplotlib.pyplot as plt
    >>> x, y = np.mgrid[-1:3:.01, -2:1.5:.01]
    >>> pos = np.dstack((x, y))
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.set_aspect('equal')
    >>> plt.contourf(x, y, rv.pdf(pos))
        # 创建概率密度函数的等高线图示例
    """

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.
            # 随机种子

        """
        super().__init__(seed)
        # 调用父类初始化方法
        self.__doc__ = doccer.docformat(self.__doc__, mvt_docdict_params)
        # 格式化文档字符串
        self._random_state = check_random_state(seed)
        # 检查和设置随机状态
    def __call__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        """Create a frozen multivariate t-distribution.

        See `multivariate_t_frozen` for parameters.
        """
        # 如果自由度 df 是无穷大，则返回一个多变量正态分布的冻结对象
        if df == np.inf:
            return multivariate_normal_frozen(mean=loc, cov=shape,
                                              allow_singular=allow_singular,
                                              seed=seed)
        # 否则返回一个多变量 t 分布的冻结对象
        return multivariate_t_frozen(loc=loc, shape=shape, df=df,
                                     allow_singular=allow_singular, seed=seed)

    def pdf(self, x, loc=None, shape=1, df=1, allow_singular=False):
        """Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        pdf : Probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.pdf(x, loc, shape, df)
        0.00075713

        """
        # 处理参数并返回维度、位置、形状和自由度
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        # 处理量化点 x
        x = self._process_quantiles(x, dim)
        # 处理形状信息并返回一个半正定矩阵
        shape_info = _PSD(shape, allow_singular=allow_singular)
        # 计算多变量 t 分布的对数概率密度函数并返回其指数
        logpdf = self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df,
                              dim, shape_info.rank)
        return np.exp(logpdf)

    def logpdf(self, x, loc=None, shape=1, df=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        %(_mvt_doc_default_callparams)s

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.logpdf(x, loc, shape, df)
        -7.1859802

        See Also
        --------
        pdf : Probability density function.

        """
        # 处理参数并返回维度、位置、形状和自由度
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        # 处理量化点 x
        x = self._process_quantiles(x, dim)
        # 处理形状信息并返回一个半正定矩阵
        shape_info = _PSD(shape)
        # 返回多变量 t 分布的对数概率密度函数
        return self._logpdf(x, loc, shape_info.U, shape_info.log_pdet, df, dim,
                            shape_info.rank)
    def _logpdf(self, x, loc, prec_U, log_pdet, df, dim, rank):
        """Utility method `pdf`, `logpdf` for parameters.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability density
            function.
        loc : ndarray
            Location of the distribution.
        prec_U : ndarray
            A decomposition such that `np.dot(prec_U, prec_U.T)` is the inverse
            of the shape matrix.
        log_pdet : float
            Logarithm of the determinant of the shape matrix.
        df : float
            Degrees of freedom of the distribution.
        dim : int
            Dimension of the quantiles x.
        rank : int
            Rank of the shape matrix.

        Notes
        -----
        As this function does no argument checking, it should not be called
        directly; use 'logpdf' instead.

        """
        # Check if degrees of freedom is infinite, then delegate to multivariate_normal's _logpdf
        if df == np.inf:
            return multivariate_normal._logpdf(x, loc, prec_U, log_pdet, rank)

        # Calculate deviation from the location
        dev = x - loc
        # Calculate squared Mahalanobis distance
        maha = np.square(np.dot(dev, prec_U)).sum(axis=-1)

        # Calculate components of the log probability density function
        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim / 2.0 * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1.0 / df) * maha)

        # Return the squeezed output of the computed log probability density
        return _squeeze_output(A - B - C - D + E)

    def _cdf(self, x, loc, shape, df, dim, maxpts=None, lower_limit=None,
             random_state=None):

        # Handle random state initialization
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        # Set default max points for integration
        if not maxpts:
            maxpts = 1000 * dim

        # Process quantiles and handle lower limit
        x = self._process_quantiles(x, dim)
        lower_limit = (np.full(loc.shape, -np.inf)
                       if lower_limit is None else lower_limit)

        # Center the inputs by removing the mean
        x, lower_limit = x - loc, lower_limit - loc

        # Broadcast arrays and handle swapping
        b, a = np.broadcast_arrays(x, lower_limit)
        i_swap = b < a
        signs = (-1) ** (i_swap.sum(axis=-1))  # odd # of swaps -> negative
        a, b = a.copy(), b.copy()
        a[i_swap], b[i_swap] = b[i_swap], a[i_swap]
        n = x.shape[-1]
        limits = np.concatenate((a, b), axis=-1)

        # Define a function to apply along the last axis
        def func1d(limits):
            a, b = limits[:n], limits[n:]
            return _qmvt(maxpts, df, shape, a, b, rng)[0]

        # Apply the function to each 1-D slice of `limits`
        res = np.apply_along_axis(func1d, -1, limits) * signs

        # Squeeze the output to ensure correct shape
        return _squeeze_output(res)
    def cdf(self, x, loc=None, shape=1, df=1, allow_singular=False, *,
            maxpts=None, lower_limit=None, random_state=None):
        """Multivariate t-distribution cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        %(_mvt_doc_default_callparams)s
            默认参数和关键字参数的文档字符串，由外部源（可能是模块级别的设置）提供。
        maxpts : int, optional
            Maximum number of points to use for integration. The default is
            1000 times the number of dimensions.
        lower_limit : array_like, optional
            Lower limit of integration of the cumulative distribution function.
            Default is negative infinity. Must be broadcastable with `x`.
        %(_doc_random_state)s
            随机数生成器的文档字符串，描述如何处理随机状态参数。

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at `x`.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.cdf(x, loc, shape, df)
        0.64798491

        """
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        # 处理参数并确保正确的维度、位置、形状和自由度
        shape = _PSD(shape, allow_singular=allow_singular)._M
        # 将形状参数传递给一个处理函数并返回修正后的形状

        return self._cdf(x, loc, shape, df, dim, maxpts,
                         lower_limit, random_state)

    def _entropy(self, dim, df=1, shape=1):
        if df == np.inf:
            return multivariate_normal(None, cov=shape).entropy()
        # 如果自由度是无穷大，则返回多变量正态分布的熵

        shape_info = _PSD(shape)
        shape_term = 0.5 * shape_info.log_pdet
        # 计算形状信息的半对数行列式，并存储为术语

        def regular(dim, df):
            halfsum = 0.5 * (dim + df)
            half_df = 0.5 * df
            return (
                -gammaln(halfsum) + gammaln(half_df)
                + 0.5 * dim * np.log(df * np.pi) + halfsum
                * (psi(halfsum) - psi(half_df))
                + shape_term
            )
            # 计算正常情况下的熵，涉及伽马函数、对数和Ψ（Γ）函数

        def asymptotic(dim, df):
            # Wolfram Alpha的渐近扩展公式：
            # "-gammaln((m+d)/2) + gammaln(d/2) + (m*log(d*pi))/2
            #  + ((m+d)/2) * (digamma((m+d)/2) - digamma(d/2))"
            return (
                dim * norm._entropy() + dim / df
                - dim * (dim - 2) * df**-2.0 / 4
                + dim**2 * (dim - 2) * df**-3.0 / 6
                + dim * (-3 * dim**3 + 8 * dim**2 - 8) * df**-4.0 / 24
                + dim**2 * (3 * dim**3 - 10 * dim**2 + 16) * df**-5.0 / 30
                + shape_term
            )[()]
            # 计算渐近情况下的熵，包含多项式级数和形状术语

        # 保留至少 `dim=1e5` 时大约12位数字的精度。参见 gh-18465。
        threshold = dim * 100 * 4 / (np.log(dim) + 1)
        return _lazywhere(df >= threshold, (dim, df), f=asymptotic, f2=regular)
        # 根据阈值选择使用渐近或正常情况下的熵函数
    def entropy(self, loc=None, shape=1, df=1):
        """Calculate the differential entropy of a multivariate
        t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
            Parameters for the multivariate t-distribution.

        Returns
        -------
        h : float
            Differential entropy of the distribution.

        """
        # Process the distribution parameters and return the calculated entropy
        dim, loc, shape, df = self._process_parameters(None, shape, df)
        return self._entropy(dim, df, shape)

    def rvs(self, loc=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        %(_mvt_doc_default_callparams)s
            Parameters for the multivariate t-distribution.
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
            Random state information for reproducibility.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `P`), where `P` is the
            dimension of the random variable.

        Examples
        --------
        >>> from scipy.stats import multivariate_t
        >>> x = [0.4, 5]
        >>> loc = [0, 1]
        >>> shape = [[1, 0.1], [0.1, 1]]
        >>> df = 7
        >>> multivariate_t.rvs(loc, shape, df)
        array([[0.93477495, 3.00408716]])

        """
        # Reference for the sampling method:
        #
        #    Hofert, "On Sampling from the Multivariate t Distribution", 2013
        #    http://rjournal.github.io/archive/2013-2/hofert.pdf
        #
        # Process the distribution parameters and initialize the random number generator
        dim, loc, shape, df = self._process_parameters(loc, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        # Generate samples according to the multivariate t-distribution properties
        if np.isinf(df):
            x = np.ones(size)
        else:
            x = rng.chisquare(df, size=size) / df

        z = rng.multivariate_normal(np.zeros(dim), shape, size=size)
        samples = loc + z / np.sqrt(x)[..., None]
        return _squeeze_output(samples)

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        # Convert input to float array and adjust dimensions accordingly
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x
    def _process_parameters(self, loc, shape, df):
        """
        Infer dimensionality from location array and shape matrix, handle
        defaults, and ensure compatible dimensions.
        """
        # 如果 loc 和 shape 都为 None，则设定默认值
        if loc is None and shape is None:
            loc = np.asarray(0, dtype=float)   # 默认 loc 为 0
            shape = np.asarray(1, dtype=float)  # 默认 shape 为 1
            dim = 1  # 维度设定为 1
        elif loc is None:
            shape = np.asarray(shape, dtype=float)
            if shape.ndim < 2:
                dim = 1  # 如果 shape 的维度小于 2，则维度设定为 1
            else:
                dim = shape.shape[0]  # 否则维度为 shape 的第一维度长度
            loc = np.zeros(dim)  # loc 初始化为全零向量
        elif shape is None:
            loc = np.asarray(loc, dtype=float)  # loc 转换为浮点数数组
            dim = loc.size  # 维度为 loc 的大小
            shape = np.eye(dim)  # shape 初始化为单位矩阵
        else:
            shape = np.asarray(shape, dtype=float)
            loc = np.asarray(loc, dtype=float)
            dim = loc.size  # 维度为 loc 的大小

        if dim == 1:
            loc = loc.reshape(1)  # 如果维度为 1，则将 loc reshape 成为一维数组
            shape = shape.reshape(1, 1)  # 将 shape reshape 成为 1x1 的矩阵

        if loc.ndim != 1 or loc.shape[0] != dim:
            raise ValueError("Array 'loc' must be a vector of length %d." %
                             dim)  # 如果 loc 不是一维数组或者长度不等于 dim，则抛出 ValueError

        if shape.ndim == 0:
            shape = shape * np.eye(dim)  # 如果 shape 是零维，则用单位矩阵乘以 shape
        elif shape.ndim == 1:
            shape = np.diag(shape)  # 如果 shape 是一维，则转换为对角矩阵
        elif shape.ndim == 2 and shape.shape != (dim, dim):
            rows, cols = shape.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(shape.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'loc' is a vector of length %d.")
                msg = msg % (str(shape.shape), len(loc))
            raise ValueError(msg)  # 如果 shape 是二维但不是 dim x dim，则抛出 ValueError
        elif shape.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % shape.ndim)  # 如果 shape 的维度大于 2，则抛出 ValueError

        # 处理自由度参数 df
        if df is None:
            df = 1  # 如果 df 为 None，则设为默认值 1
        elif df <= 0:
            raise ValueError("'df' must be greater than zero.")  # 如果 df 小于等于 0，则抛出 ValueError
        elif np.isnan(df):
            raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")  # 如果 df 是 NaN，则抛出 ValueError

        return dim, loc, shape, df  # 返回处理后的 dim, loc, shape, df
class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, loc=None, shape=1, df=1, allow_singular=False,
                 seed=None):
        """Create a frozen multivariate t distribution.

        Parameters
        ----------
        loc : array_like, optional
            Mean of the distribution (default is zero vector).
        shape : array_like, optional
            Covariance matrix of the distribution (default is identity matrix).
        df : float, optional
            Degrees of freedom (default is 1).
        allow_singular : bool, optional
            Whether to allow singular covariance matrices (default is False).
        seed : {None, int, numpy.random.Generator}, optional
            Random seed or Generator object (default is None).

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import multivariate_t
        >>> loc = np.zeros(3)
        >>> shape = np.eye(3)
        >>> df = 10
        >>> dist = multivariate_t(loc, shape, df)
        >>> dist.rvs()
        array([[ 0.81412036, -1.53612361,  0.42199647]])
        >>> dist.pdf([1, 1, 1])
        array([0.01237803])

        """
        self._dist = multivariate_t_gen(seed)
        # 处理参数并返回维度、均值、协方差矩阵和自由度
        dim, loc, shape, df = self._dist._process_parameters(loc, shape, df)
        self.dim, self.loc, self.shape, self.df = dim, loc, shape, df
        # 计算半正定矩阵信息
        self.shape_info = _PSD(shape, allow_singular=allow_singular)

    def logpdf(self, x):
        # 处理输入量化参数并返回对数概率密度
        x = self._dist._process_quantiles(x, self.dim)
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.loc, U, log_pdet, self.df, self.dim,
                                  self.shape_info.rank)

    def cdf(self, x, *, maxpts=None, lower_limit=None, random_state=None):
        # 处理输入量化参数并返回累积分布函数值
        x = self._dist._process_quantiles(x, self.dim)
        return self._dist._cdf(x, self.loc, self.shape, self.df, self.dim,
                               maxpts, lower_limit, random_state)

    def pdf(self, x):
        # 返回概率密度函数值
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        # 生成随机变量样本并返回
        return self._dist.rvs(loc=self.loc,
                              shape=self.shape,
                              df=self.df,
                              size=size,
                              random_state=random_state)

    def entropy(self):
        # 返回熵值
        return self._dist._entropy(self.dim, self.df, self.shape)


multivariate_t = multivariate_t_gen()


# Set frozen generator docstrings from corresponding docstrings in
# multivariate_t_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs', 'cdf', 'entropy']:
    method = multivariate_t_gen.__dict__[name]
    method_frozen = multivariate_t_frozen.__dict__[name]
    # 使用doccer工具格式化文档字符串
    method_frozen.__doc__ = doccer.docformat(method.__doc__,
                                             mvt_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__, mvt_docdict_params)


_mhg_doc_default_callparams = """\
m : array_like
    The number of each type of object in the population.
    That is, :math:`m[i]` is the number of objects of
    type :math:`i`.
n : array_like
    The number of samples taken from the population.
"""

_mhg_doc_callparams_note = """\
`m` must be an array of positive integers. If the quantile
:math:`i` contains values out of the range :math:`[0, m_i]`
where :math:`m_i` is the number of objects of type :math:`i`

"""
# 确保冻结的调用参数文档为空字符串
_mhg_doc_frozen_callparams = ""

# 冻结调用参数的注释说明
_mhg_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

# 多元超几何分布的文档字典，包含默认调用参数、调用参数注释和随机状态的文档说明
mhg_docdict_params = {
    '_doc_default_callparams': _mhg_doc_default_callparams,
    '_doc_callparams_note': _mhg_doc_callparams_note,
    '_doc_random_state': _doc_random_state
}

# 多元超几何分布的无参数文档字典，包含冻结的调用参数文档和随机状态的文档说明
mhg_docdict_noparams = {
    '_doc_default_callparams': _mhg_doc_frozen_callparams,
    '_doc_callparams_note': _mhg_doc_frozen_callparams_note,
    '_doc_random_state': _doc_random_state
}


class multivariate_hypergeom_gen(multi_rv_generic):
    r"""A multivariate hypergeometric random variable.

    Methods
    -------
    pmf(x, m, n)
        Probability mass function.
    logpmf(x, m, n)
        Log of the probability mass function.
    rvs(m, n, size=1, random_state=None)
        Draw random samples from a multivariate hypergeometric
        distribution.
    mean(m, n)
        Mean of the multivariate hypergeometric distribution.
    var(m, n)
        Variance of the multivariate hypergeometric distribution.
    cov(m, n)
        Compute the covariance matrix of the multivariate
        hypergeometric distribution.

    Parameters
    ----------
    %(_doc_default_callparams)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_callparams_note)s

    The probability mass function for `multivariate_hypergeom` is

    .. math::

        P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{\binom{m_1}{x_1}
        \binom{m_2}{x_2} \cdots \binom{m_k}{x_k}}{\binom{M}{n}}, \\ \quad
        (x_1, x_2, \ldots, x_k) \in \mathbb{N}^k \text{ with }
        \sum_{i=1}^k x_i = n

    where :math:`m_i` are the number of objects of type :math:`i`, :math:`M`
    is the total number of objects in the population (sum of all the
    :math:`m_i`), and :math:`n` is the size of the sample to be taken
    from the population.

    .. versionadded:: 1.6.0

    Examples
    --------
    To evaluate the probability mass function of the multivariate
    hypergeometric distribution, with a dichotomous population of size
    :math:`10` and :math:`20`, at a sample of size :math:`12` with
    :math:`8` objects of the first type and :math:`4` objects of the
    second type, use:

    >>> from scipy.stats import multivariate_hypergeom
    >>> multivariate_hypergeom.pmf(x=[8, 4], m=[10, 20], n=12)
    0.0025207176631464523

    The `multivariate_hypergeom` distribution is identical to the
    corresponding `hypergeom` distribution (tiny numerical differences
    notwithstanding) when only two types (good and bad) of objects
    are present in the population as in the example above. Consider
    another example for a comparison with the hypergeometric distribution:

    >>> from scipy.stats import hypergeom
    # 初始化函数，用于设置对象的初始状态
    def __init__(self, seed=None):
        # 调用父类的初始化方法
        super().__init__(seed)
        # 使用文档格式化工具，将对象文档格式化为特定参数的格式
        self.__doc__ = doccer.docformat(self.__doc__, mhg_docdict_params)

    # 对象可调用方法，创建一个冻结的多元超几何分布
    def __call__(self, m, n, seed=None):
        """Create a frozen multivariate_hypergeom distribution.

        See `multivariate_hypergeom_frozen` for more information.
        """
        # 返回冻结的多元超几何分布对象，使用给定的参数m、n和seed
        return multivariate_hypergeom_frozen(m, n, seed=seed)
    # 处理参数 m 和 n，将它们转换为 NumPy 数组
    def _process_parameters(self, m, n):
        m = np.asarray(m)
        n = np.asarray(n)

        # 如果 m 是空数组，则将其类型转换为整数类型
        if m.size == 0:
            m = m.astype(int)

        # 如果 n 是空数组，则将其类型转换为整数类型
        if n.size == 0:
            n = n.astype(int)

        # 检查 m 数组的数据类型是否为整数类型
        if not np.issubdtype(m.dtype, np.integer):
            raise TypeError("'m' must an array of integers.")

        # 检查 n 数组的数据类型是否为整数类型
        if not np.issubdtype(n.dtype, np.integer):
            raise TypeError("'n' must an array of integers.")

        # 检查 m 数组是否至少是一维的
        if m.ndim == 0:
            raise ValueError("'m' must be an array with at least one dimension.")

        # 检查是否 m 数组不为空，如果不为空则扩展 n 数组的维度
        if m.size != 0:
            n = n[..., np.newaxis]

        # 使用广播将 m 和 n 数组扩展为相同的形状
        m, n = np.broadcast_arrays(m, n)

        # 如果 m 数组不为空，则将 n 数组降低一个维度
        if m.size != 0:
            n = n[..., 0]

        # 计算 m 中小于 0 的条件
        mcond = m < 0

        # 计算 m 数组的和，沿着最后一个轴
        M = m.sum(axis=-1)

        # 计算 n 中小于 0 或大于 M 的条件
        ncond = (n < 0) | (n > M)

        # 返回处理后的结果：总和 M，处理后的 m 和 n，以及它们的条件
        return M, m, n, mcond, ncond, np.any(mcond, axis=-1) | ncond

    # 处理分位数参数 x，与前一函数返回的 M、m、n 相关
    def _process_quantiles(self, x, M, m, n):
        x = np.asarray(x)

        # 检查 x 数组的数据类型是否为整数类型
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError("'x' must an array of integers.")

        # 检查 x 数组是否至少是一维的
        if x.ndim == 0:
            raise ValueError("'x' must be an array with at least one dimension.")

        # 检查 x 数组与 m 数组最后一个维度的大小是否一致
        if not x.shape[-1] == m.shape[-1]:
            raise ValueError(f"Size of each quantile must be size of 'm': received {x.shape[-1]}, but expected {m.shape[-1]}.")

        # 如果 m 数组不为空，则扩展 n 和 M 数组的维度
        if m.size != 0:
            n = n[..., np.newaxis]
            M = M[..., np.newaxis]

        # 使用广播将 x、m、n、M 扩展为相同的形状
        x, m, n, M = np.broadcast_arrays(x, m, n, M)

        # 如果 m 数组不为空，则将 n 和 M 数组降低一个维度
        if m.size != 0:
            n, M = n[..., 0], M[..., 0]

        # 计算 x 中小于 0 或大于对应 m 中的条件
        xcond = (x < 0) | (x > m)

        # 返回处理后的结果：x、M、m、n，以及 xcond 条件
        return (x, M, m, n, xcond, np.any(xcond, axis=-1) | (x.sum(axis=-1) != n))

    # 检查计算结果的有效性，并根据条件替换为指定的错误值
    def _checkresult(self, result, cond, bad_value):
        result = np.asarray(result)

        # 如果 cond 的维度不为 0，则将 result 中 cond 位置的值替换为 bad_value
        if cond.ndim != 0:
            result[cond] = bad_value

        # 如果 cond 为 True，则直接返回 bad_value
        elif cond:
            return bad_value

        # 如果 result 的维度为 0，则返回 result 的第一个元素
        if result.ndim == 0:
            return result[()]

        # 返回处理后的 result
        return result

    # 计算对数概率质量函数，使用给定的参数和条件
    def _logpmf(self, x, M, m, n, mxcond, ncond):
        # 这个概率质量函数的公式来自于组合数的关系
        num = np.zeros_like(m, dtype=np.float64)
        den = np.zeros_like(n, dtype=np.float64)

        # 根据条件过滤 m 和 x 数组中的值
        m, x = m[~mxcond], x[~mxcond]
        M, n = M[~ncond], n[~ncond]

        # 计算 num 和 den 的值
        num[~mxcond] = (betaln(m+1, 1) - betaln(x+1, m-x+1))
        den[~ncond] = (betaln(M+1, 1) - betaln(n+1, M-n+1))

        # 将满足条件的位置设为 NaN
        num[mxcond] = np.nan
        den[ncond] = np.nan

        # 对 num 沿最后一个轴求和，并返回结果
        num = num.sum(axis=-1)
        return num - den
    def logpmf(self, x, m, n):
        """Log of the multivariate hypergeometric probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s
            默认调用参数，用于替换在函数调用中的默认参数说明

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`
            在 `x` 处评估的概率质量函数的对数值

        Notes
        -----
        %(_doc_callparams_note)s
            调用参数的注意事项说明
        """
        # 处理参数，获得 M, m, n, mcond, ncond, mncond
        M, m, n, mcond, ncond, mncond = self._process_parameters(m, n)
        # 处理量化数据 x，获取 (x, M, m, n, xcond, xcond_reduced)
        (x, M, m, n, xcond,
         xcond_reduced) = self._process_quantiles(x, M, m, n)
        # 计算 mxcond，用于指示 mcond 或 xcond
        mxcond = mcond | xcond
        # 将 ncond 初始化为与 n 形状相同的布尔数组
        ncond = ncond | np.zeros(n.shape, dtype=np.bool_)

        # 计算概率质量函数的对数值
        result = self._logpmf(x, M, m, n, mxcond, ncond)

        # 替换 x 超出定义域的值，使用广播将 xcond 扩展到合适的形状
        xcond_ = xcond_reduced | np.zeros(mncond.shape, dtype=np.bool_)
        result = self._checkresult(result, xcond_, -np.inf)

        # 替换不适合 n 或 m 的值，使用广播将 mncond 扩展到合适的形状
        mncond_ = mncond | np.zeros(xcond_reduced.shape, dtype=np.bool_)
        return self._checkresult(result, mncond_, np.nan)

    def pmf(self, x, m, n):
        """Multivariate hypergeometric probability mass function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s
            默认调用参数，用于替换在函数调用中的默认参数说明

        Returns
        -------
        pmf : ndarray or scalar
            Probability density function evaluated at `x`
            在 `x` 处评估的概率密度函数

        Notes
        -----
        %(_doc_callparams_note)s
            调用参数的注意事项说明
        """
        # 计算概率质量函数的指数值，即 exp(logpmf(x, m, n))
        out = np.exp(self.logpmf(x, m, n))
        return out

    def mean(self, m, n):
        """Mean of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            默认调用参数，用于替换在函数调用中的默认参数说明

        Returns
        -------
        mean : array_like or scalar
            The mean of the distribution
            分布的均值
        """
        # 处理参数，获取 M, m, n, _, _, mncond
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        # 检查空数组
        if m.size != 0:
            M, n = M[..., np.newaxis], n[..., np.newaxis]
        # 检查 M 中是否有零值，用掩码数组处理
        cond = (M == 0)
        M = np.ma.masked_array(M, mask=cond)
        # 计算分布的均值 mu
        mu = n*(m/M)
        # 如果 m 不为空数组，则扩展 mncond 到 mu 的形状
        if m.size != 0:
            mncond = (mncond[..., np.newaxis] |
                      np.zeros(mu.shape, dtype=np.bool_))
        return self._checkresult(mu, mncond, np.nan)
    def var(self, m, n):
        """Variance of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            默认调用参数的描述信息，通常会在文档字符串的 %(_doc_default_callparams)s 中替换成实际内容。

        Returns
        -------
        array_like
            组件的方差。这是分布的协方差矩阵的对角线。

        """
        # 处理参数，并返回所需的参数和条件
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        
        # 检查空数组
        if m.size != 0:
            M, n = M[..., np.newaxis], n[..., np.newaxis]
        
        # 创建掩码数组以处理 M 中为零的情况
        cond = (M == 0) & (M-1 == 0)
        M = np.ma.masked_array(M, mask=cond)
        
        # 计算输出，即分布组件的方差
        output = n * m/M * (M-m)/M * (M-n)/(M-1)
        
        # 进一步处理空数组
        if m.size != 0:
            mncond = (mncond[..., np.newaxis] |
                      np.zeros(output.shape, dtype=np.bool_))
        
        # 检查并返回结果，确保结果的一致性
        return self._checkresult(output, mncond, np.nan)

    def cov(self, m, n):
        """Covariance matrix of the multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
            默认调用参数的描述信息，通常会在文档字符串的 %(_doc_default_callparams)s 中替换成实际内容。

        Returns
        -------
        cov : array_like
            分布的协方差矩阵。

        """
        # 参见 [1]_ 中的公式和 [2]_ 中的实现方法
        # 计算协方差矩阵的公式为 cov( x_i,x_j ) = -n * (M-n)/(M-1) * (K_i*K_j) / (M**2)
        
        # 处理参数，并返回所需的参数和条件
        M, m, n, _, _, mncond = self._process_parameters(m, n)
        
        # 检查空数组
        if m.size != 0:
            M = M[..., np.newaxis, np.newaxis]
            n = n[..., np.newaxis, np.newaxis]
        
        # 创建掩码数组以处理 M 中为零的情况
        cond = (M == 0) & (M-1 == 0)
        M = np.ma.masked_array(M, mask=cond)
        
        # 计算输出，即协方差矩阵
        output = (-n * (M-n)/(M-1) *
                  np.einsum("...i,...j->...ij", m, m) / (M**2))
        
        # 进一步处理空数组
        if m.size != 0:
            M, n = M[..., 0, 0], n[..., 0, 0]
            cond = cond[..., 0, 0]
        
        dim = m.shape[-1]
        
        # 对角线条目需要单独计算
        for i in range(dim):
            output[..., i, i] = (n * (M-n) * m[..., i]*(M-m[..., i]))
            output[..., i, i] = output[..., i, i] / (M-1)
            output[..., i, i] = output[..., i, i] / (M**2)
        
        # 进一步处理空数组
        if m.size != 0:
            mncond = (mncond[..., np.newaxis, np.newaxis] |
                      np.zeros(output.shape, dtype=np.bool_))
        
        # 检查并返回结果，确保结果的一致性
        return self._checkresult(output, mncond, np.nan)
    def rvs(self, m, n, size=None, random_state=None):
        """Draw random samples from a multivariate hypergeometric distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        size : integer or iterable of integers, optional
            Number of samples to draw. Default is ``None``, in which case a
            single variate is returned as an array with shape ``m.shape``.
        %(_doc_random_state)s

        Returns
        -------
        rvs : array_like
            Random variates of shape ``size`` or ``m.shape``
            (if ``size=None``).

        Notes
        -----
        %(_doc_callparams_note)s

        Also note that NumPy's `multivariate_hypergeometric` sampler is not
        used as it doesn't support broadcasting.
        """
        # 解析参数并返回处理后的参数
        M, m, n, _, _, _ = self._process_parameters(m, n)

        # 获取随机数生成器的状态
        random_state = self._get_random_state(random_state)

        # 如果 size 不为 None 且为整数，则转换为元组
        if size is not None and isinstance(size, int):
            size = (size, )

        # 初始化结果变量 rvs
        if size is None:
            # 如果 size 为 None，则返回单个变量的形状为 m.shape 的数组
            rvs = np.empty(m.shape, dtype=m.dtype)
        else:
            # 否则，返回形状为 size + (m.shape[-1], ) 的数组
            rvs = np.empty(size + (m.shape[-1], ), dtype=m.dtype)
        # 初始化剩余元素 rem 为总体 M
        rem = M

        # 这个采样器来自于 numpy 的 gh-13794
        # https://github.com/numpy/numpy/pull/13794
        # 对每个维度进行采样
        for c in range(m.shape[-1] - 1):
            # 计算剩余元素
            rem = rem - m[..., c]
            # 生成 n0mask，用于处理 n 为 0 的情况
            n0mask = n == 0
            # 使用超几何分布生成随机变量并存入 rvs
            rvs[..., c] = (~n0mask *
                           random_state.hypergeometric(m[..., c],
                                                       rem + n0mask,
                                                       n + n0mask,
                                                       size=size))
            # 更新 n
            n = n - rvs[..., c]
        # 存储最后一个维度的随机变量
        rvs[..., m.shape[-1] - 1] = n

        # 返回生成的随机变量数组
        return rvs
# 创建多元超几何分布生成器对象
multivariate_hypergeom = multivariate_hypergeom_gen()

# 定义多元超几何冻结分布类，继承自多元随机变量的冻结分布类
class multivariate_hypergeom_frozen(multi_rv_frozen):
    def __init__(self, m, n, seed=None):
        # 使用多元超几何分布生成器创建分布对象
        self._dist = multivariate_hypergeom_gen(seed)
        # 处理参数并保存到实例变量中
        (self.M, self.m, self.n,
         self.mcond, self.ncond,
         self.mncond) = self._dist._process_parameters(m, n)

        # monkey patch self._dist，替换 _process_parameters 方法
        def _process_parameters(m, n):
            return (self.M, self.m, self.n,
                    self.mcond, self.ncond,
                    self.mncond)
        self._dist._process_parameters = _process_parameters

    # 返回给定观测值 x 的对数概率质量函数值
    def logpmf(self, x):
        return self._dist.logpmf(x, self.m, self.n)

    # 返回给定观测值 x 的概率质量函数值
    def pmf(self, x):
        return self._dist.pmf(x, self.m, self.n)

    # 返回分布的均值
    def mean(self):
        return self._dist.mean(self.m, self.n)

    # 返回分布的方差
    def var(self):
        return self._dist.var(self.m, self.n)

    # 返回分布的协方差
    def cov(self):
        return self._dist.cov(self.m, self.n)

    # 返回随机抽样值，可指定抽样个数和随机数种子
    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.m, self.n,
                              size=size,
                              random_state=random_state)


# 将多元超几何分布的冻结生成器的文档字符串设置为相应的文档字符串，并填充类文档字符串中的默认字符串
for name in ['logpmf', 'pmf', 'mean', 'var', 'cov', 'rvs']:
    method = multivariate_hypergeom_gen.__dict__[name]
    method_frozen = multivariate_hypergeom_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, mhg_docdict_noparams)
    method.__doc__ = doccer.docformat(method.__doc__,
                                      mhg_docdict_params)


# 定义随机表格生成器类，继承自多元随机变量的通用生成器类
class random_table_gen(multi_rv_generic):
    r"""Contingency tables from independent samples with fixed marginal sums.

    This is the distribution of random tables with given row and column vector
    sums. This distribution represents the set of random tables under the null
    hypothesis that rows and columns are independent. It is used in hypothesis
    tests of independence.

    Because of assumed independence, the expected frequency of each table
    element can be computed from the row and column sums, so that the
    distribution is completely determined by these two vectors.

    Methods
    -------
    logpmf(x)
        Log-probability of table `x` to occur in the distribution.
    pmf(x)
        Probability of table `x` to occur in the distribution.
    mean(row, col)
        Mean table.
    rvs(row, col, size=None, method=None, random_state=None)
        Draw random tables with given row and column vector sums.

    Parameters
    ----------
    %(_doc_row_col)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_row_col_note)s

    Random elements from the distribution are generated either with Boyett's
    [1]_ or Patefield's algorithm [2]_. Boyett's algorithm has
    O(N) time and space complexity, where N is the total sum of entries in the
    # 这部分是文档字符串，描述了 `random_table` 类的作用、使用方法、示例和参考文献。

    def __init__(self, seed=None):
        # 构造函数，初始化随机数生成器对象，可以指定种子 `seed`
        super().__init__(seed)

    def __call__(self, row, col, *, seed=None):
        """Create a frozen distribution of tables with given marginals.

        See `random_table_frozen` for more information.
        """
        # 调用对象时，创建一个具有给定边际分布的冻结分布，返回 `random_table_frozen` 对象
        return random_table_frozen(row, col, seed=seed)
    # 计算给定分布下，表格 `x` 出现的对数概率质量函数。

    # 参数说明：
    # %(_doc_x)s：表格 `x`，通常是一个二维数组。
    # %(_doc_row_col)s：行和列的边际总和，用于指定分布的期望行和列总和。

    # 返回值：
    # logpmf : ndarray 或者标量
    #     在 `x` 处评估的对数概率质量函数的对数值。

    # 注意事项：
    # %(_doc_row_col_note)s
    # 如果 `x` 的行和列边际与 `row` 和 `col` 不匹配，则返回负无穷。

    r, c, n = self._process_parameters(row, col)
    # 将 `row` 和 `col` 参数处理为期望的形式，并分别赋值给 `r`、`c` 和 `n`。

    x = np.asarray(x)
    # 将输入的 `x` 转换为 NumPy 数组。

    if x.ndim < 2:
        raise ValueError("`x` 必须至少是二维的")

    dtype_is_int = np.issubdtype(x.dtype, np.integer)
    # 检查 `x` 的数据类型是否为整数类型。

    with np.errstate(invalid='ignore'):
        if not dtype_is_int and not np.all(x.astype(int) == x):
            raise ValueError("`x` 必须仅包含整数值")

    # 到达这里时，如果 `x` 不包含 NaN，则无需处理。

    if np.any(x < 0):
        raise ValueError("`x` 必须仅包含非负值")

    r2 = np.sum(x, axis=-1)
    c2 = np.sum(x, axis=-2)
    # 计算 `x` 在最后两个轴上的和，分别存储在 `r2` 和 `c2` 中。

    if r2.shape[-1] != len(r):
        raise ValueError("`x` 的形状必须与 `row` 一致")

    if c2.shape[-1] != len(c):
        raise ValueError("`x` 的形状必须与 `col` 一致")

    res = np.empty(x.shape[:-2])
    # 创建一个空数组 `res`，其形状与 `x` 的前两个轴相同。

    mask = np.all(r2 == r, axis=-1) & np.all(c2 == c, axis=-1)
    # 创建布尔掩码，指示 `x` 的每个表格是否与期望的行和列边际匹配。

    def lnfac(x):
        return gammaln(x + 1)
    # 定义一个函数 `lnfac`，用于计算对数阶乘 `ln(x!)`。

    res[mask] = (np.sum(lnfac(r), axis=-1) + np.sum(lnfac(c), axis=-1)
                 - lnfac(n) - np.sum(lnfac(x[mask]), axis=(-1, -2)))
    # 对于匹配的表格，计算其对数概率质量函数值并存储在 `res` 中。

    res[~mask] = -np.inf
    # 对于不匹配的表格，将对数概率质量函数值设置为负无穷。

    return res[()]
    # 返回计算得到的对数概率质量函数值数组。
    def pmf(self, x, row, col):
        """Probability mass function of the distribution.

        Parameters
        ----------
        x : array_like
            The table whose probability mass function is to be computed.
        row : array_like
            The vector specifying the marginal sums of rows in `x`.
        col : array_like
            The vector specifying the marginal sums of columns in `x`.

        Returns
        -------
        pmf : ndarray or scalar
            Probability mass function evaluated at `x`.

        Notes
        -----
        The function computes the probability of the table `x` occurring under
        the given row and column marginals. If these do not match `row` and `col`,
        the function returns zero.

        Examples
        --------
        >>> from scipy.stats import random_table
        >>> import numpy as np

        >>> x = [[1, 5, 1], [2, 3, 1]]
        >>> row = np.sum(x, axis=1)
        >>> col = np.sum(x, axis=0)
        >>> random_table.pmf(x, row, col)
        0.19580419580419592

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.pmf(x)
        0.19580419580419592
        """
        # Compute the probability mass function using the logarithmic approach
        return np.exp(self.logpmf(x, row, col))

    def mean(self, row, col):
        """Mean of the conditional distribution of tables.

        Parameters
        ----------
        row : array_like
            The vector specifying the marginal sums of rows.
        col : array_like
            The vector specifying the marginal sums of columns.

        Returns
        -------
        mean: ndarray
            Mean of the distribution, representing the expected value of the table.

        Notes
        -----
        This function computes the mean (expected value) of the distribution of
        conditional tables, given the marginal sums of rows and columns `row` and `col`.

        Examples
        --------
        >>> from scipy.stats import random_table

        >>> row = [1, 5]
        >>> col = [2, 3, 1]
        >>> random_table.mean(row, col)
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.mean()
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])
        """
        # Process parameters and compute the mean using outer product and normalization
        r, c, n = self._process_parameters(row, col)
        return np.outer(r, c) / n
    @staticmethod
    def _process_parameters(row, col):
        """
        Check that row and column vectors are one-dimensional, that they do
        not contain negative or non-integer entries, and that the sums over
        both vectors are equal.

        Parameters
        ----------
        row : array-like
            Vector representing row marginals.
        col : array-like
            Vector representing column marginals.

        Returns
        -------
        r : ndarray
            Processed row vector after validation.
        c : ndarray
            Processed column vector after validation.
        n : int
            Sum of the row vector, equivalent to sum of column vector.

        Raises
        ------
        ValueError
            If any validation checks fail.

        Notes
        -----
        This static method ensures that both row and column vectors are valid
        for generating random tables with fixed marginals.
        """
        # Convert row and column vectors to numpy arrays with int64 type
        r = np.array(row, dtype=np.int64, copy=True)
        c = np.array(col, dtype=np.int64, copy=True)

        # Check dimensions of row and column vectors
        if np.ndim(r) != 1:
            raise ValueError("`row` must be one-dimensional")
        if np.ndim(c) != 1:
            raise ValueError("`col` must be one-dimensional")

        # Check for negative entries in row and column vectors
        if np.any(r < 0):
            raise ValueError("each element of `row` must be non-negative")
        if np.any(c < 0):
            raise ValueError("each element of `col` must be non-negative")

        # Calculate sum of elements in row and column vectors
        n = np.sum(r)
        if n != np.sum(c):
            raise ValueError("sums over `row` and `col` must be equal")

        # Check if input vectors were correctly converted to int64 type
        if not np.all(r == np.asarray(row)):
            raise ValueError("each element of `row` must be an integer")
        if not np.all(c == np.asarray(col)):
            raise ValueError("each element of `col` must be an integer")

        # Return validated row and column vectors, and their sum
        return r, c, n
    def _process_size_shape(size, r, c):
        """
        Compute the number of samples to be drawn and the shape of the output
        """
        # 计算行数和列数，以元组形式保存在 shape 中
        shape = (len(r), len(c))

        if size is None:
            # 如果 size 参数为 None，则返回一个样本和形状 shape
            return 1, shape

        size = np.atleast_1d(size)
        # 确保 size 至少是一维的数组
        if not np.issubdtype(size.dtype, np.integer) or np.any(size < 0):
            # 如果 size 不是整数类型或者有任何负值，则抛出 ValueError 异常
            raise ValueError("`size` must be a non-negative integer or `None`")

        # 计算 size 数组所有元素的乘积，以及 size 和 shape 的元组连接
        return np.prod(size), tuple(size) + shape

    @classmethod
    def _process_rvs_method(cls, method, r, c, n):
        known_methods = {
            None: cls._rvs_select(r, c, n),
            "boyett": cls._rvs_boyett,
            "patefield": cls._rvs_patefield,
        }
        try:
            # 根据 method 从 known_methods 字典中选择相应的方法并返回
            return known_methods[method]
        except KeyError:
            # 如果 method 不在 known_methods 中，则抛出 ValueError 异常
            raise ValueError(f"'{method}' not recognized, "
                             f"must be one of {set(known_methods)}")

    @classmethod
    def _rvs_select(cls, r, c, n):
        fac = 1.0  # benchmarks show that this value is about 1
        k = len(r) * len(c)  # number of cells
        # n + 1 guards against failure if n == 0
        # 如果 n 大于 fac * np.log(n + 1) * k，则返回 _rvs_patefield 方法
        if n > fac * np.log(n + 1) * k:
            return cls._rvs_patefield
        # 否则返回 _rvs_boyett 方法
        return cls._rvs_boyett

    @staticmethod
    def _rvs_boyett(row, col, ntot, size, random_state):
        # 调用 _rcont 模块的 rvs_rcont1 方法进行随机变量采样
        return _rcont.rvs_rcont1(row, col, ntot, size, random_state)

    @staticmethod
    def _rvs_patefield(row, col, ntot, size, random_state):
        # 调用 _rcont 模块的 rvs_rcont2 方法进行随机变量采样
        return _rcont.rvs_rcont2(row, col, ntot, size, random_state)
# 生成随机表格
random_table = random_table_gen()

# 创建一个冻结的随机表格类，继承自 multi_rv_frozen
class random_table_frozen(multi_rv_frozen):
    def __init__(self, row, col, *, seed=None):
        # 使用随机种子生成随机表格
        self._dist = random_table_gen(seed)
        # 处理参数，确定行和列的数量
        self._params = self._dist._process_parameters(row, col)

        # monkey patch self._dist
        def _process_parameters(r, c):
            return self._params
        self._dist._process_parameters = _process_parameters

    # 返回对数概率质量函数
    def logpmf(self, x):
        return self._dist.logpmf(x, None, None)

    # 返回概率质量函数
    def pmf(self, x):
        return self._dist.pmf(x, None, None)

    # 返回期望值
    def mean(self):
        return self._dist.mean(None, None)

    # 生成随机变量样本
    def rvs(self, size=None, method=None, random_state=None):
        # 可以在这里进行优化
        return self._dist.rvs(None, None, size=size, method=method,
                              random_state=random_state)


# 关于行和列的文档说明
_ctab_doc_row_col = """\
row : array_like
    Sum of table entries in each row.
col : array_like
    Sum of table entries in each column."""

# 关于 x 的文档说明
_ctab_doc_x = """\
x : array-like
   Two-dimensional table of non-negative integers, or a
   multi-dimensional array with the last two dimensions
   corresponding with the tables."""

# 关于行和列的注意事项文档说明
_ctab_doc_row_col_note = """\
The row and column vectors must be one-dimensional, not empty,
and each sum up to the same value. They cannot contain negative
or noninteger entries."""

# 关于均值参数的文档说明
_ctab_doc_mean_params = f"""
Parameters
----------
{_ctab_doc_row_col}"""

# 冻结状态下的行和列注意事项文档说明
_ctab_doc_row_col_note_frozen = """\
See class definition for a detailed description of parameters."""

# 文档字典，包含各种文档说明
_ctab_docdict = {
    "_doc_random_state": _doc_random_state,
    "_doc_row_col": _ctab_doc_row_col,
    "_doc_x": _ctab_doc_x,
    "_doc_mean_params": _ctab_doc_mean_params,
    "_doc_row_col_note": _ctab_doc_row_col_note,
}

# 冻结状态下的文档字典，从 _ctab_docdict 复制并更新特定键的内容
_ctab_docdict_frozen = _ctab_docdict.copy()
_ctab_docdict_frozen.update({
    "_doc_row_col": "",
    "_doc_mean_params": "",
    "_doc_row_col_note": _ctab_doc_row_col_note_frozen,
})


# 填充对象的文档字符串，使用给定的模板
def _docfill(obj, docdict, template=None):
    obj.__doc__ = doccer.docformat(template or obj.__doc__, docdict)


# 从 random_table 中设置冻结生成器的文档字符串，并在类文档字符串中填入默认字符串
_docfill(random_table_gen, _ctab_docdict)

# 遍历方法列表，设置冻结版本的文档字符串
for name in ['logpmf', 'pmf', 'mean', 'rvs']:
    method = random_table_gen.__dict__[name]
    method_frozen = random_table_frozen.__dict__[name]
    _docfill(method_frozen, _ctab_docdict_frozen, method.__doc__)
    _docfill(method, _ctab_docdict)


# 均匀方向生成器类，继承自 multi_rv_generic
class uniform_direction_gen(multi_rv_generic):
    r"""A vector-valued uniform direction.

    Return a random direction (unit vector). The `dim` keyword specifies
    the dimensionality of the space.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random directions.

    Parameters
    ----------
    dim : scalar
        Dimension of directions.
"""
    `
        # 参数 seed，可以是 None、int、numpy.random.Generator 或 numpy.random.RandomState
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional
    
            # 用于生成随机变量
            Used for drawing random variates.
            # 如果 seed 为 None，则使用 numpy.random.RandomState 的单例
            If `seed` is `None`, the `~np.random.RandomState` singleton is used.
            # 如果 seed 为整数，则使用一个新的 RandomState 实例，使用 seed 进行种子初始化
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            # 如果 seed 已经是 RandomState 或 Generator 实例，则使用该对象
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            # 默认值为 None
            Default is `None`.
    
        # 说明部分
        Notes
        -----
        # 该分布生成均匀分布在高维球面上的单位向量，可以解释为随机方向
        This distribution generates unit vectors uniformly distributed on
        the surface of a hypersphere. These can be interpreted as random
        directions.
        # 例如，如果维度 dim 为 3，则会从 S^2 的表面采样 3D 向量
        For example, if `dim` is 3, 3D vectors from the surface of :math:`S^2`
        will be sampled.
    
        # 参考文献部分
        References
        ----------
        .. [1] Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    
        # 示例部分
        Examples
        --------
        >>> import numpy as np
        >>> from scipy.stats import uniform_direction
        # 生成一个随机方向，位于 S^2 表面上
        >>> x = uniform_direction.rvs(3)
        >>> np.linalg.norm(x)
        1.
    
        # 这生成一个随机方向，位于 S^2 表面上。
    
        # 另一种方法是调用该对象（作为函数），返回一个固定维度参数的冻结分布。
        # 这里，我们创建一个 dim 为 3 的 uniform_direction，并抽取 5 个样本。
        # 样本随后按形状为 5x3 的数组排列。
        >>> rng = np.random.default_rng()
        >>> uniform_sphere_dist = uniform_direction(3)
        >>> unit_vectors = uniform_sphere_dist.rvs(5, random_state=rng)
        >>> unit_vectors
        array([[ 0.56688642, -0.1332634 , -0.81294566],
               [-0.427126  , -0.74779278,  0.50830044],
               [ 0.3793989 ,  0.92346629,  0.05715323],
               [ 0.36428383, -0.92449076, -0.11231259],
               [-0.27733285,  0.94410968, -0.17816678]])
    def rvs(self, dim, size=None, random_state=None):
        """Draw random samples from S(N-1).

        Parameters
        ----------
        dim : integer
            Dimension of space (N).

        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.

        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
            Pseudorandom number generator state used to generate resamples.
            
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

        Returns
        -------
        rvs : ndarray
            Random direction vectors
        """
        # 获取有效的随机数生成器状态
        random_state = self._get_random_state(random_state)
        
        # 如果没有指定 size，则设为空数组
        if size is None:
            size = np.array([], dtype=int)
        
        # 至少将 size 转换为一维数组
        size = np.atleast_1d(size)

        # 处理维度参数，确保其有效
        dim = self._process_parameters(dim)

        # 调用函数 _sample_uniform_direction 生成均匀分布的方向向量样本
        samples = _sample_uniform_direction(dim, size, random_state)
        
        # 返回生成的样本
        return samples
# 使用 uniform_direction_gen() 函数生成一个统一方向分布对象
uniform_direction = uniform_direction_gen()

# 定义一个冻结的 uniform_direction 分布类，继承自 multi_rv_frozen 类
class uniform_direction_frozen(multi_rv_frozen):
    def __init__(self, dim=None, seed=None):
        """Create a frozen n-dimensional uniform direction distribution.

        Parameters
        ----------
        dim : int
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import uniform_direction
        >>> x = uniform_direction(3)
        >>> x.rvs()

        """
        # 创建一个冻结的 n 维统一方向分布对象
        self._dist = uniform_direction_gen(seed)
        # 处理参数 dim，将其保存到实例的 dim 属性中
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=None, random_state=None):
        # 调用 _dist 对象的 rvs 方法生成随机样本
        return self._dist.rvs(self.dim, size, random_state)


def _sample_uniform_direction(dim, size, random_state):
    """
    Private method to generate uniform directions
    Reference: Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    """
    # 生成符合统一方向分布的样本
    samples_shape = np.append(size, dim)
    samples = random_state.standard_normal(samples_shape)
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples

# _dirichlet_mn_doc_default_callparams 文档字符串，描述了 Dirichlet-Multinomial 分布的默认调用参数
_dirichlet_mn_doc_default_callparams = """\
alpha : array_like
    The concentration parameters. The number of entries along the last axis
    determines the dimensionality of the distribution. Each entry must be
    strictly positive.
n : int or array_like
    The number of trials. Each element must be a strictly positive integer.
"""

# _dirichlet_mn_doc_frozen_callparams 空字符串，作为冻结的 Dirichlet-Multinomial 分布的调用参数说明
_dirichlet_mn_doc_frozen_callparams = ""

# _dirichlet_mn_doc_frozen_callparams_note 文档字符串，提醒查看类定义获取详细的参数说明
_dirichlet_mn_doc_frozen_callparams_note = """\
See class definition for a detailed description of parameters."""

# dirichlet_mn_docdict_params 字典，包含默认参数相关的文档字符串
dirichlet_mn_docdict_params = {
    '_dirichlet_mn_doc_default_callparams': _dirichlet_mn_doc_default_callparams,
    '_doc_random_state': _doc_random_state
}

# dirichlet_mn_docdict_noparams 字典，不包含参数的文档字符串
dirichlet_mn_docdict_noparams = {
    '_dirichlet_mn_doc_default_callparams': _dirichlet_mn_doc_frozen_callparams,
    '_doc_random_state': _doc_random_state
}

def _dirichlet_multinomial_check_parameters(alpha, n, x=None):
    # 将 alpha 和 n 转换为 numpy 数组
    alpha = np.asarray(alpha)
    n = np.asarray(n)
    # 如果 `x` 不是 None，则进行以下处理
    if x is not None:
        # 确保 `x` 和 `alpha` 是数组。如果它们的形状不兼容，NumPy 会抛出适当的错误。
        try:
            # 尝试将 `x` 和 `alpha` 广播为兼容形状的数组
            x, alpha = np.broadcast_arrays(x, alpha)
        except ValueError as e:
            # 如果广播失败，抛出自定义的错误信息
            msg = "`x` and `alpha` must be broadcastable."
            raise ValueError(msg) from e

        # 将 `x` 转换为整数类型并向下取整
        x_int = np.floor(x)
        # 如果 `x` 中存在负数或者不是整数，抛出错误
        if np.any(x < 0) or np.any(x != x_int):
            raise ValueError("`x` must contain only non-negative integers.")
        # 更新 `x` 为整数类型的 `x_int`
        x = x_int

    # 检查 `alpha` 中是否存在非正数的值，若存在则抛出错误
    if np.any(alpha <= 0):
        raise ValueError("`alpha` must contain only positive values.")

    # 将 `n` 转换为整数类型并向下取整
    n_int = np.floor(n)
    # 如果 `n` 中存在非正数或者不是整数，抛出错误
    if np.any(n <= 0) or np.any(n != n_int):
        raise ValueError("`n` must be a positive integer.")
    # 更新 `n` 为整数类型的 `n_int`
    n = n_int

    # 计算 `alpha` 在最后一个维度上的和，并将其与 `n` 广播为兼容形状的数组
    sum_alpha = np.sum(alpha, axis=-1)
    sum_alpha, n = np.broadcast_arrays(sum_alpha, n)

    # 如果 `x` 是 None，则返回 `(alpha, sum_alpha, n)`，否则返回 `(alpha, sum_alpha, n, x)`
    return (alpha, sum_alpha, n) if x is None else (alpha, sum_alpha, n, x)
# 定义一个 Dirichlet multinomial 随机变量类，继承自 multi_rv_generic 类
class dirichlet_multinomial_gen(multi_rv_generic):
    r"""A Dirichlet multinomial random variable.

    The Dirichlet multinomial distribution is a compound probability
    distribution: it is the multinomial distribution with number of trials
    `n` and class probabilities ``p`` randomly sampled from a Dirichlet
    distribution with concentration parameters ``alpha``.

    Methods
    -------
    logpmf(x, alpha, n):
        Log of the probability mass function.
    pmf(x, alpha, n):
        Probability mass function.
    mean(alpha, n):
        Mean of the Dirichlet multinomial distribution.
    var(alpha, n):
        Variance of the Dirichlet multinomial distribution.
    cov(alpha, n):
        The covariance of the Dirichlet multinomial distribution.

    Parameters
    ----------
    %(_dirichlet_mn_doc_default_callparams)s
    %(_doc_random_state)s

    See Also
    --------
    scipy.stats.dirichlet : The dirichlet distribution.
    scipy.stats.multinomial : The multinomial distribution.

    References
    ----------
    .. [1] Dirichlet-multinomial distribution, Wikipedia,
           https://www.wikipedia.org/wiki/Dirichlet-multinomial_distribution

    Examples
    --------
    >>> from scipy.stats import dirichlet_multinomial

    Get the PMF

    >>> n = 6  # number of trials
    >>> alpha = [3, 4, 5]  # concentration parameters
    >>> x = [1, 2, 3]  # counts
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    0.08484162895927604

    If the sum of category counts does not equal the number of trials,
    the probability mass is zero.

    >>> dirichlet_multinomial.pmf(x, alpha, n=7)
    0.0

    Get the log of the PMF

    >>> dirichlet_multinomial.logpmf(x, alpha, n)
    -2.4669689491013327

    Get the mean

    >>> dirichlet_multinomial.mean(alpha, n)
    array([1.5, 2. , 2.5])

    Get the variance

    >>> dirichlet_multinomial.var(alpha, n)
    array([1.55769231, 1.84615385, 2.01923077])

    Get the covariance

    >>> dirichlet_multinomial.cov(alpha, n)
    array([[ 1.55769231, -0.69230769, -0.86538462],
           [-0.69230769,  1.84615385, -1.15384615],
           [-0.86538462, -1.15384615,  2.01923077]])

    Alternatively, the object may be called (as a function) to fix the
    `alpha` and `n` parameters, returning a "frozen" Dirichlet multinomial
    random variable.

    >>> dm = dirichlet_multinomial(alpha, n)
    >>> dm.pmf(x)
    0.08484162895927579

    All methods are fully vectorized. Each element of `x` and `alpha` is
    a vector (along the last axis), each element of `n` is an
    integer (scalar), and the result is computed element-wise.

    >>> x = [[1, 2, 3], [4, 5, 6]]
    >>> alpha = [[1, 2, 3], [4, 5, 6]]
    >>> n = [6, 15]
    >>> dirichlet_multinomial.pmf(x, alpha, n)
    array([0.06493506, 0.02626937])

    >>> dirichlet_multinomial.cov(alpha, n).shape  # both covariance matrices
    (2, 3, 3)

    Broadcasting according to standard NumPy conventions is supported. Here,
    we have four sets of concentration parameters (each a two element vector)
    for each of three numbers of trials (each a scalar).

    >>> alpha = [[3, 4], [4, 5], [5, 6], [6, 7]]
    >>> n = [[6], [7], [8]]
    >>> dirichlet_multinomial.mean(alpha, n).shape
    (3, 4, 2)

    """
    # 定义一个类，继承自父类
    def __init__(self, seed=None):
        # 调用父类的初始化方法
        super().__init__(seed)
        # 根据提供的参数格式化文档字符串
        self.__doc__ = doccer.docformat(self.__doc__,
                                        dirichlet_mn_docdict_params)

    # 定义类的调用方法
    def __call__(self, alpha, n, seed=None):
        # 调用底层函数 dirichlet_multinomial_frozen，返回结果
        return dirichlet_multinomial_frozen(alpha, n, seed=seed)

    # 定义对数概率质量函数方法
    def logpmf(self, x, alpha, n):
        """The log of the probability mass function.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Log of the probability mass function.

        """
        # 检查参数，并返回合适的参数
        a, Sa, n, x = _dirichlet_multinomial_check_parameters(alpha, n, x)

        # 计算对数概率质量函数的结果
        out = np.asarray(loggamma(Sa) + loggamma(n + 1) - loggamma(n + Sa))
        out += (loggamma(x + a) - (loggamma(a) + loggamma(x + 1))).sum(axis=-1)
        np.place(out, n != x.sum(axis=-1), -np.inf)
        return out[()]

    # 定义概率质量函数方法
    def pmf(self, x, alpha, n):
        """Probability mass function for a Dirichlet multinomial distribution.

        Parameters
        ----------
        x: ndarray
            Category counts (non-negative integers). Must be broadcastable
            with shape parameter ``alpha``. If multidimensional, the last axis
            must correspond with the categories.
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray or scalar
            Probability mass function.

        """
        # 返回概率质量函数的结果，通过调用 logpmf 方法
        return np.exp(self.logpmf(x, alpha, n))

    # 定义均值方法
    def mean(self, alpha, n):
        """Mean of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s

        Returns
        -------
        out: ndarray
            Mean of a Dirichlet multinomial distribution.

        """
        # 检查参数，并返回合适的参数
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = n[..., np.newaxis], Sa[..., np.newaxis]
        # 计算并返回均值
        return n * a / Sa
    def var(self, alpha, n):
        """The variance of the Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s
            传入参数说明字符串的格式化输出

        Returns
        -------
        out: array_like
            组件的方差。这是分布的协方差矩阵的对角线。

        """
        # 检查和规范化参数
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        n, Sa = n[..., np.newaxis], Sa[..., np.newaxis]
        # 计算方差
        return n * a / Sa * (1 - a/Sa) * (n + Sa) / (1 + Sa)

    def cov(self, alpha, n):
        """Covariance matrix of a Dirichlet multinomial distribution.

        Parameters
        ----------
        %(_dirichlet_mn_doc_default_callparams)s
            传入参数说明字符串的格式化输出

        Returns
        -------
        out : array_like
            分布的协方差矩阵。

        """
        # 检查和规范化参数
        a, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        # 计算方差
        var = dirichlet_multinomial.var(a, n)

        n, Sa = n[..., np.newaxis, np.newaxis], Sa[..., np.newaxis, np.newaxis]
        aiaj = a[..., :, np.newaxis] * a[..., np.newaxis, :]
        # 计算协方差矩阵
        cov = -n * aiaj / Sa ** 2 * (n + Sa) / (1 + Sa)

        ii = np.arange(cov.shape[-1])
        cov[..., ii, ii] = var
        return cov
# 创建一个 Dirichlet-Multinomial 分布的生成器对象
dirichlet_multinomial = dirichlet_multinomial_gen()

# 定义一个冻结的 Dirichlet-Multinomial 分布类，继承自 multi_rv_frozen 类
class dirichlet_multinomial_frozen(multi_rv_frozen):
    def __init__(self, alpha, n, seed=None):
        # 检查和设置参数 alpha, n，并确保其符合 Dirichlet-Multinomial 分布的要求
        alpha, Sa, n = _dirichlet_multinomial_check_parameters(alpha, n)
        self.alpha = alpha  # 存储参数 alpha
        self.n = n  # 存储参数 n
        self._dist = dirichlet_multinomial_gen(seed)  # 创建一个新的 Dirichlet-Multinomial 分布生成器对象

    def logpmf(self, x):
        return self._dist.logpmf(x, self.alpha, self.n)  # 返回给定 x 的对数概率质量函数值

    def pmf(self, x):
        return self._dist.pmf(x, self.alpha, self.n)  # 返回给定 x 的概率质量函数值

    def mean(self):
        return self._dist.mean(self.alpha, self.n)  # 返回分布的均值

    def var(self):
        return self._dist.var(self.alpha, self.n)  # 返回分布的方差

    def cov(self):
        return self._dist.cov(self.alpha, self.n)  # 返回分布的协方差

# 为 dirichlet_multinomial 和 dirichlet_multinomial_frozen 中的方法设置文档字符串，
# 并填充类文档字符串中的默认字符串。
for name in ['logpmf', 'pmf', 'mean', 'var', 'cov']:
    method = dirichlet_multinomial_gen.__dict__[name]
    method_frozen = dirichlet_multinomial_frozen.__dict__[name]
    method_frozen.__doc__ = doccer.docformat(
        method.__doc__, dirichlet_mn_docdict_noparams)  # 使用 doccer 将无参数版本的文档字符串格式化并赋值给冻结方法
    method.__doc__ = doccer.docformat(method.__doc__,
                                      dirichlet_mn_docdict_params)  # 使用 doccer 将带参数版本的文档字符串格式化并赋值给原始方法

# 定义一个 Von Mises-Fisher 分布的生成器类，继承自 multi_rv_generic 类
class vonmises_fisher_gen(multi_rv_generic):
    r"""A von Mises-Fisher variable.

    The `mu` keyword specifies the mean direction vector. The `kappa` keyword
    specifies the concentration parameter.

    Methods
    -------
    pdf(x, mu=None, kappa=1)
        Probability density function.
    logpdf(x, mu=None, kappa=1)
        Log of the probability density function.
    rvs(mu=None, kappa=1, size=1, random_state=None)
        Draw random samples from a von Mises-Fisher distribution.
    entropy(mu=None, kappa=1)
        Compute the differential entropy of the von Mises-Fisher distribution.
    fit(data)
        Fit a von Mises-Fisher distribution to data.

    Parameters
    ----------
    mu : array_like
        Mean direction of the distribution. Must be a one-dimensional unit
        vector of norm 1.
    kappa : float
        Concentration parameter. Must be positive.
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    See Also
    --------
    scipy.stats.vonmises : Von-Mises Fisher distribution in 2D on a circle
    uniform_direction : uniform distribution on the surface of a hypersphere

    Notes
    -----
    The von Mises-Fisher distribution is a directional distribution on the
    surface of the unit hypersphere. The probability density
    function of a unit vector :math:`\mathbf{x}` is
    """
    .. math::

        f(\mathbf{x}) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2}I_{d/2-1}(\kappa)}
               \exp\left(\kappa \mathbf{\mu}^T\mathbf{x}\right),

    where :math:`\mathbf{\mu}` is the mean direction, :math:`\kappa` the
    concentration parameter, :math:`d` the dimension and :math:`I` the
    modified Bessel function of the first kind. As :math:`\mu` represents
    a direction, it must be a unit vector or in other words, a point
    on the hypersphere: :math:`\mathbf{\mu}\in S^{d-1}`. :math:`\kappa` is a
    concentration parameter, which means that it must be positive
    (:math:`\kappa>0`) and that the distribution becomes more narrow with
    increasing :math:`\kappa`. In that sense, the reciprocal value
    :math:`1/\kappa` resembles the variance parameter of the normal
    distribution.

    The von Mises-Fisher distribution often serves as an analogue of the
    normal distribution on the sphere. Intuitively, for unit vectors, a
    useful distance measure is given by the angle :math:`\alpha` between
    them. This is exactly what the scalar product
    :math:`\mathbf{\mu}^T\mathbf{x}=\cos(\alpha)` in the
    von Mises-Fisher probability density function describes: the angle
    between the mean direction :math:`\mathbf{\mu}` and the vector
    :math:`\mathbf{x}`. The larger the angle between them, the smaller the
    probability to observe :math:`\mathbf{x}` for this particular mean
    direction :math:`\mathbf{\mu}`.

    In dimensions 2 and 3, specialized algorithms are used for fast sampling
    [2]_, [3]_. For dimensions of 4 or higher the rejection sampling algorithm
    described in [4]_ is utilized. This implementation is partially based on
    the geomstats package [5]_, [6]_.

    .. versionadded:: 1.11

    References
    ----------
    .. [1] Von Mises-Fisher distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    .. [2] Mardia, K., and Jupp, P. Directional statistics. Wiley, 2000.
    .. [3] J. Wenzel. Numerically stable sampling of the von Mises Fisher
           distribution on S2.
           https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    .. [4] Wood, A. Simulation of the von mises fisher distribution.
           Communications in statistics-simulation and computation 23,
           1 (1994), 157-164. https://doi.org/10.1080/03610919408813161
    .. [5] geomstats, Github. MIT License. Accessed: 06.01.2023.
           https://github.com/geomstats/geomstats
    .. [6] Miolane, N. et al. Geomstats:  A Python Package for Riemannian
           Geometry in Machine Learning. Journal of Machine Learning Research
           21 (2020). http://jmlr.org/papers/v21/19-027.html

    Examples
    --------
    **Visualization of the probability density**

    Plot the probability density in three dimensions for increasing
    concentration parameter. The density is calculated by the ``pdf``
    method.

    >>> import numpy as np


注释：
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import vonmises_fisher
    >>> from matplotlib.colors import Normalize
    >>> n_grid = 100
    >>> u = np.linspace(0, np.pi, n_grid)
    >>> v = np.linspace(0, 2 * np.pi, n_grid)
    >>> u_grid, v_grid = np.meshgrid(u, v)
    >>> vertices = np.stack([np.cos(v_grid) * np.sin(u_grid),
    ...                      np.sin(v_grid) * np.sin(u_grid),
    ...                      np.cos(u_grid)],
    ...                     axis=2)
    >>> x = np.outer(np.cos(v), np.sin(u))
    >>> y = np.outer(np.sin(v), np.sin(u))
    >>> z = np.outer(np.ones_like(u), np.cos(u))
    
    # 定义函数 `plot_vmf_density` 用于绘制 von Mises-Fisher 分布的密度图
    >>> def plot_vmf_density(ax, x, y, z, vertices, mu, kappa):
    ...     # 创建 von Mises-Fisher 分布对象
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     # 计算顶点处的概率密度值
    ...     pdf_values = vmf.pdf(vertices)
    ...     # 根据密度值范围设置颜色映射规范化
    ...     pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    ...     # 绘制三维曲面图，使用 viridis 调色板根据概率密度着色
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1,
    ...                     facecolors=plt.cm.viridis(pdfnorm(pdf_values)),
    ...                     linewidth=0)
    ...     # 设置坐标轴等比例显示
    ...     ax.set_aspect('equal')
    ...     # 设置视角方位角为 -130 度，仰角为 0 度
    ...     ax.view_init(azim=-130, elev=0)
    ...     # 关闭坐标轴显示
    ...     ax.axis('off')
    ...     # 设置图表标题，显示当前的 kappa 值
    ...     ax.set_title(rf"$\kappa={kappa}$")
    
    >>> fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4),
    ...                          subplot_kw={"projection": "3d"})
    >>> left, middle, right = axes
    >>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
    
    # 在三个子图中分别绘制不同 kappa 值下的 von Mises-Fisher 分布密度图
    >>> plot_vmf_density(left, x, y, z, vertices, mu, 5)
    >>> plot_vmf_density(middle, x, y, z, vertices, mu, 20)
    >>> plot_vmf_density(right, x, y, z, vertices, mu, 100)
    
    # 调整子图之间的布局，使其填充整个图像空间，并显示图像
    >>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, wspace=0.)
    >>> plt.show()
    
    # 增加分隔行，进行下一部分的代码注释
    
    # 从 von Mises-Fisher 分布中抽取样本，生成 5x3 的样本数组
    >>> rng = np.random.default_rng()
    >>> mu = np.array([0, 0, 1])
    >>> samples = vonmises_fisher(mu, 20).rvs(5, random_state=rng)
    >>> samples
    array([[ 0.3884594 , -0.32482588,  0.86231516],
           [ 0.00611366, -0.09878289,  0.99509023],
           [-0.04154772, -0.01637135,  0.99900239],
           [-0.14613735,  0.12553507,  0.98126695],
           [-0.04429884, -0.23474054,  0.97104814]])
    
    # 这些样本是单位球面上的单位向量，验证其欧几里得范数是否为 1
    >>> np.linalg.norm(samples, axis=1)
    array([1., 1., 1., 1., 1.])
    
    # 增加分隔行，进行下一部分的代码注释
    
    # 定义函数 `plot_vmf_samples` 用于绘制从 von Mises-Fisher 分布抽取的样本点
    >>> def plot_vmf_samples(ax, x, y, z, mu, kappa):
    ...     # 创建 von Mises-Fisher 分布对象
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     # 从分布中抽取 20 个样本点
    ...     samples = vmf.rvs(20)
    ...     # 绘制三维曲面图，用于显示分布的轮廓，线宽设为 0，透明度设为 0.2
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
    ...                     alpha=0.2)
    # 在 3D 散点图中绘制样本点，使用黑色表示，大小为 5
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5)
    # 在 3D 散点图中绘制平均方向 mu 的点，使用红色表示，大小为 30
    ax.scatter(mu[0], mu[1], mu[2], c='r', s=30)
    # 设置坐标轴比例为相等
    ax.set_aspect('equal')
    # 设置视角，azim 表示方位角，elev 表示仰角
    ax.view_init(azim=-130, elev=0)
    # 关闭坐标轴
    ax.axis('off')
    # 设置标题，使用 LaTeX 表达式显示 κ 的值
    ax.set_title(rf"$\kappa={kappa}$")

>>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
>>> fig, axes = plt.subplots(nrows=1, ncols=3,
...                          subplot_kw={"projection": "3d"},
...                          figsize=(9, 4))
>>> left, middle, right = axes
>>> plot_vmf_samples(left, x, y, z, mu, 5)
>>> plot_vmf_samples(middle, x, y, z, mu, 20)
>>> plot_vmf_samples(right, x, y, z, mu, 100)
>>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0,
...                     right=1.0, wspace=0.)
>>> plt.show()

The plots show that with increasing concentration :math:`\kappa` the
resulting samples are centered more closely around the mean direction.

**Fitting the distribution parameters**

The distribution can be fitted to data using the ``fit`` method returning
the estimated parameters. As a toy example let's fit the distribution to
samples drawn from a known von Mises-Fisher distribution.

>>> mu, kappa = np.array([0, 0, 1]), 20
>>> samples = vonmises_fisher(mu, kappa).rvs(1000, random_state=rng)
>>> mu_fit, kappa_fit = vonmises_fisher.fit(samples)
>>> mu_fit, kappa_fit
(array([0.01126519, 0.01044501, 0.99988199]), 19.306398751730995)

We see that the estimated parameters `mu_fit` and `kappa_fit` are
very close to the ground truth parameters.

"""
# 构造函数，初始化随机种子
def __init__(self, seed=None):
    super().__init__(seed)

# 实例调用方法，创建一个冻结的 von Mises-Fisher 分布
def __call__(self, mu=None, kappa=1, seed=None):
    """Create a frozen von Mises-Fisher distribution.

    See `vonmises_fisher_frozen` for more information.
    """
    return vonmises_fisher_frozen(mu, kappa, seed=seed)

# 内部方法，处理分布参数，确保 mu 是单位向量，kappa 是正数
def _process_parameters(self, mu, kappa):
    """
    Infer dimensionality from mu and ensure that mu is a one-dimensional
    unit vector and kappa positive.
    """
    mu = np.asarray(mu)
    if mu.ndim > 1:
        raise ValueError("'mu' must have one-dimensional shape.")
    if not np.allclose(np.linalg.norm(mu), 1.):
        raise ValueError("'mu' must be a unit vector of norm 1.")
    if not mu.size > 1:
        raise ValueError("'mu' must have at least two entries.")
    kappa_error_msg = "'kappa' must be a positive scalar."
    if not np.isscalar(kappa) or kappa < 0:
        raise ValueError(kappa_error_msg)
    if float(kappa) == 0.:
        raise ValueError("For 'kappa=0' the von Mises-Fisher distribution "
                         "becomes the uniform distribution on the sphere "
                         "surface. Consider using "
                         "'scipy.stats.uniform_direction' instead.")
    dim = mu.size

    return dim, mu, kappa
   `
    def _check_data_vs_dist(self, x, dim):
        # 检查输入向量 x 的最后一个维度是否与给定的维度 dim 匹配
        if x.shape[-1] != dim:
            raise ValueError("The dimensionality of the last axis of 'x' must "
                             "match the dimensionality of the "
                             "von Mises Fisher distribution.")
        # 检查所有向量 x 是否都是单位向量（最后一个维度）
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)

    def _log_norm_factor(self, dim, kappa):
        # 计算 von Mises-Fisher 分布的归一化系数的对数值
        halfdim = 0.5 * dim
        return (0.5 * (dim - 2) * np.log(kappa) - halfdim * _LOG_2PI -
                np.log(ive(halfdim - 1, kappa)) - kappa)

    def _logpdf(self, x, dim, mu, kappa):
        """von Mises-Fisher 概率密度函数的对数值。

        由于此函数不做参数检查，不应直接调用；请使用 'logpdf' 方法。

        """
        x = np.asarray(x)
        self._check_data_vs_dist(x, dim)
        dotproducts = np.einsum('i,...i->...', mu, x)
        return self._log_norm_factor(dim, kappa) + kappa * dotproducts

    def logpdf(self, x, mu=None, kappa=1):
        """von Mises-Fisher 概率密度函数的对数值。

        Parameters
        ----------
        x : array_like
            用于计算概率密度函数对数值的点。`x` 的最后一个轴必须与分布的维度匹配，
            是相同维度的单位向量。
        mu : array_like, 默认值: None
            分布的平均方向。必须是一维单位向量，范数为1。
        kappa : float, 默认值: 1
            集中参数。必须为正数。

        Returns
        -------
        logpdf : ndarray 或标量
            在 `x` 处计算的概率密度函数的对数值。

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        return self._logpdf(x, dim, mu, kappa)
    def pdf(self, x, mu=None, kappa=1):
        """
        Von Mises-Fisher probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.
        mu : array_like, optional
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1. Defaults to None.
        kappa : float, optional
            Concentration parameter. Must be positive. Defaults to 1.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.
        """
        # 处理参数 `mu` 和 `kappa`，确保它们符合要求
        dim, mu, kappa = self._process_parameters(mu, kappa)
        # 返回计算出的概率密度函数值的指数
        return np.exp(self._logpdf(x, dim, mu, kappa))

    def _rvs_2d(self, mu, kappa, size, random_state):
        """
        In 2D, the von Mises-Fisher distribution reduces to the
        von Mises distribution which can be efficiently sampled by numpy.
        This method is much faster than the general rejection
        sampling based algorithm.

        Parameters
        ----------
        mu : array_like
            Mean direction of the distribution.
        kappa : float
            Concentration parameter. Must be positive.
        size : int or tuple
            Number of samples to generate.
        random_state : np.random.RandomState
            Random state object for reproducible random numbers.

        Returns
        -------
        samples : ndarray
            Samples from the von Mises-Fisher distribution in 2D.
        """
        # 计算平均角度
        mean_angle = np.arctan2(mu[1], mu[0])
        # 使用 numpy 的 von Mises 分布生成角度样本
        angle_samples = random_state.vonmises(mean_angle, kappa, size=size)
        # 将角度样本转换为二维平面上的样本点
        samples = np.stack([np.cos(angle_samples), np.sin(angle_samples)],
                           axis=-1)
        return samples

    def _rvs_3d(self, kappa, size, random_state):
        """
        Generate samples from a von Mises-Fisher distribution
        with mu = [1, 0, 0] and kappa. Samples then have to be
        rotated towards the desired mean direction mu.
        This method is much faster than the general rejection
        sampling based algorithm.
        Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

        Parameters
        ----------
        kappa : float
            Concentration parameter. Must be positive.
        size : int or tuple
            Number of samples to generate.
        random_state : np.random.RandomState
            Random state object for reproducible random numbers.

        Returns
        -------
        samples : ndarray
            Samples from the von Mises-Fisher distribution in 3D.
        """
        if size is None:
            sample_size = 1
        else:
            sample_size = size

        # 根据公式计算 x 坐标
        x = random_state.random(sample_size)
        x = 1. + np.log(x + (1. - x) * np.exp(-2 * kappa))/kappa

        # (y, z) 是随机的二维向量，需要按比例归一化
        temp = np.sqrt(1. - np.square(x))
        uniformcircle = _sample_uniform_direction(2, sample_size, random_state)
        # 组合成最终的样本点
        samples = np.stack([x, temp * uniformcircle[..., 0],
                            temp * uniformcircle[..., 1]],
                           axis=-1)
        if size is None:
            samples = np.squeeze(samples)
        return samples
    def _rotate_samples(self, samples, mu, dim):
        """使用QR分解找到将北极（1, 0,...,0）映射到向量mu的旋转矩阵。
        然后将这个旋转应用到所有的样本中。

        Parameters
        ----------
        samples: array_like, shape = [..., n]
            待旋转的样本数据
        mu : array-like, shape=[n, ]
            用于参数化旋转的点。

        Returns
        -------
        samples : rotated samples
            旋转后的样本数据

        """
        base_point = np.zeros((dim, ))
        base_point[0] = 1.  # 设置北极点为(1, 0,...,0)
        embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])
        rotmatrix, _ = np.linalg.qr(np.transpose(embedded))  # 对嵌入矩阵进行QR分解得到旋转矩阵
        if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], mu):
            rotsign = 1  # 确定旋转的符号
        else:
            rotsign = -1

        # 应用旋转
        samples = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign
        return samples

    def _rvs(self, dim, mu, kappa, size, random_state):
        if dim == 2:
            samples = self._rvs_2d(mu, kappa, size, random_state)  # 对于二维情况，使用专门的方法生成样本
        elif dim == 3:
            samples = self._rvs_3d(kappa, size, random_state)  # 对于三维情况，使用专门的方法生成样本
        else:
            samples = self._rejection_sampling(dim, kappa, size,
                                               random_state)  # 对于其他维度情况，使用拒绝采样方法生成样本

        if dim != 2:
            samples = self._rotate_samples(samples, mu, dim)  # 如果维度不是二维，则进行样本的旋转操作
        return samples
    # 定义一个方法，用于从 von Mises-Fisher 分布中抽取随机样本
    def rvs(self, mu=None, kappa=1, size=1, random_state=None):
        """Draw random samples from a von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array_like
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float
            Concentration parameter. Must be positive.
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, np.random.RandomState, np.random.Generator},
                        optional
            Used for drawing random variates.
            If `seed` is `None`, the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is `None`.

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`, `N`), where `N` is the
            dimension of the distribution.

        """
        # 处理参数，并获取维度、均值方向向量和集中参数 kappa
        dim, mu, kappa = self._process_parameters(mu, kappa)
        # 获取随机数生成器的实例
        random_state = self._get_random_state(random_state)
        # 使用私有方法 _rvs 生成指定维度、均值方向向量和集中参数 kappa 的随机样本
        samples = self._rvs(dim, mu, kappa, size, random_state)
        # 返回生成的随机样本
        return samples

    def _entropy(self, dim, kappa):
        # 计算 von Mises-Fisher 分布的差分熵
        halfdim = 0.5 * dim
        # 使用修正 Bessel 函数计算 von Mises-Fisher 分布的差分熵
        return (-self._log_norm_factor(dim, kappa) - kappa *
                ive(halfdim, kappa) / ive(halfdim - 1, kappa))

    def entropy(self, mu=None, kappa=1):
        """Compute the differential entropy of the von Mises-Fisher
        distribution.

        Parameters
        ----------
        mu : array_like, default: None
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float, default: 1
            Concentration parameter. Must be positive.

        Returns
        -------
        h : scalar
            Entropy of the von Mises-Fisher distribution.

        """
        # 处理参数，并获取维度和集中参数 kappa
        dim, _, kappa = self._process_parameters(mu, kappa)
        # 调用 _entropy 方法计算 von Mises-Fisher 分布的差分熵
        return self._entropy(dim, kappa)
    def fit(self, x):
        """Fit the von Mises-Fisher distribution to data.

        Parameters
        ----------
        x : array-like
            Data the distribution is fitted to. Must be two dimensional.
            The second axis of `x` must be unit vectors of norm 1 and
            determine the dimensionality of the fitted
            von Mises-Fisher distribution.

        Returns
        -------
        mu : ndarray
            Estimated mean direction.
        kappa : float
            Estimated concentration parameter.

        """
        # 将输入数据转换为 NumPy 数组
        x = np.asarray(x)
        # 检查数据维度是否为二维
        if x.ndim != 2:
            raise ValueError("'x' must be two dimensional.")
        # 检查是否所有向量的范数为 1
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)
        # 确定数据维度
        dim = x.shape[-1]

        # 计算方向统计量，mu 是方向的均值
        dirstats = directional_stats(x)
        mu = dirstats.mean_direction
        r = dirstats.mean_resultant_length

        # 求解 kappa 的方程：
        # r = I[dim/2](kappa) / I[dim/2 -1](kappa)
        #   = I[dim/2](kappa) * exp(-kappa) / I[dim/2 -1](kappa) * exp(-kappa)
        #   = ive(dim/2, kappa) / ive(dim/2 -1, kappa)

        halfdim = 0.5 * dim

        def solve_for_kappa(kappa):
            # 计算修正的贝塞尔函数值
            bessel_vals = ive([halfdim, halfdim - 1], kappa)
            return bessel_vals[0] / bessel_vals[1] - r

        # 使用 Brent 方法求解 kappa
        root_res = root_scalar(solve_for_kappa, method="brentq",
                               bracket=(1e-8, 1e9))
        kappa = root_res.root
        return mu, kappa
vonmises_fisher = vonmises_fisher_gen()

# 创建一个 von Mises-Fisher 分布生成器对象并赋值给变量 vonmises_fisher


class vonmises_fisher_frozen(multi_rv_frozen):

# 定义一个名为 vonmises_fisher_frozen 的类，继承自 multi_rv_frozen


def __init__(self, mu=None, kappa=1, seed=None):

# 类的初始化方法，用于创建一个冻结的 von Mises-Fisher 分布对象


"""Create a frozen von Mises-Fisher distribution.

Parameters
----------
mu : array_like, default: None
    Mean direction of the distribution.
kappa : float, default: 1
    Concentration parameter. Must be positive.
seed : {None, int, `numpy.random.Generator`,
        `numpy.random.RandomState`}, optional
    If `seed` is None (or `np.random`), the `numpy.random.RandomState`
    singleton is used.
    If `seed` is an int, a new ``RandomState`` instance is used,
    seeded with `seed`.
    If `seed` is already a ``Generator`` or ``RandomState`` instance
    then that instance is used.

"""

# 初始化方法的文档字符串，描述了如何创建一个冻结的 von Mises-Fisher 分布对象，以及可接受的参数说明


self._dist = vonmises_fisher_gen(seed)

# 使用给定的种子创建一个 von Mises-Fisher 分布生成器对象，并赋值给 self._dist


self.dim, self.mu, self.kappa = (
    self._dist._process_parameters(mu, kappa)
)

# 调用 von Mises-Fisher 分布生成器对象的 _process_parameters 方法，处理参数 mu 和 kappa，并将结果分别赋值给 self.dim、self.mu 和 self.kappa


def logpdf(self, x):

# 定义计算对数概率密度函数的方法 logpdf，接受参数 x


"""
Parameters
----------
x : array_like
    Points at which to evaluate the log of the probability
    density function. The last axis of `x` must correspond
    to unit vectors of the same dimensionality as the distribution.

Returns
-------
logpdf : ndarray or scalar
    Log of probability density function evaluated at `x`.

"""

# 方法 logpdf 的文档字符串，描述了参数 x 的含义以及返回值的描述


return self._dist._logpdf(x, self.dim, self.mu, self.kappa)

# 调用 von Mises-Fisher 分布生成器对象的 _logpdf 方法，计算在点 x 处概率密度函数的对数值，并返回结果


def pdf(self, x):

# 定义计算概率密度函数的方法 pdf，接受参数 x


"""
Parameters
----------
x : array_like
    Points at which to evaluate the log of the probability
    density function. The last axis of `x` must correspond
    to unit vectors of the same dimensionality as the distribution.

Returns
-------
pdf : ndarray or scalar
    Probability density function evaluated at `x`.

"""

# 方法 pdf 的文档字符串，描述了参数 x 的含义以及返回值的描述


return np.exp(self.logpdf(x))

# 调用当前对象的 logpdf 方法计算在点 x 处概率密度函数的对数值，并使用 np.exp 计算其指数，返回结果作为概率密度函数在点 x 处的值
    def rvs(self, size=1, random_state=None):
        """
        Draw random variates from the Von Mises-Fisher distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Specifies the shape of the output array. If `size` is (m,n,k),
            then m*n*k samples are generated, each of dimension `N`,
            resulting in an array of shape (m,n,k,N). If not specified,
            a single sample of dimension `N` is returned.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
            Seed for random number generation. If None, the global random
            state is used. If int, a new random state is initialized with
            the given seed. If `numpy.random.Generator` or
            `numpy.random.RandomState`, it directly uses the provided
            instance for random number generation.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the dimension
            of the Von Mises-Fisher distribution.
        """
        # Obtain the random state object for generating random numbers
        random_state = self._dist._get_random_state(random_state)
        # Draw random variates from the distribution using internal method
        return self._dist._rvs(self.dim, self.mu, self.kappa, size,
                               random_state)

    def entropy(self):
        """
        Calculate the differential entropy of the Von Mises-Fisher
        distribution.

        Returns
        -------
        h: float
            Entropy of the Von Mises-Fisher distribution.
        """
        # Calculate and return the entropy using internal method
        return self._dist._entropy(self.dim, self.kappa)
```