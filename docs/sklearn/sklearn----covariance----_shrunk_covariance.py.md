# `D:\src\scipysrc\scikit-learn\sklearn\covariance\_shrunk_covariance.py`

```
"""
Covariance estimators using shrinkage.

Shrinkage corresponds to regularising `cov` using a convex combination:
shrunk_cov = (1-shrinkage)*cov + shrinkage*structured_estimate.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# avoid division truncation
import warnings
from numbers import Integral, Real

import numpy as np

from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance


def _ledoit_wolf(X, *, assume_centered, block_size):
    """Estimate the shrunk Ledoit-Wolf covariance matrix."""
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        # Return the variance of the single feature and a shrinkage of 0.0
        return np.atleast_2d((X**2).mean()), 0.0
    n_features = X.shape[1]

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, block_size=block_size
    )
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp_cov)) / n_features
    # Compute the shrunk covariance matrix
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage


def _oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(X.shape) == 2 and X.shape[1] == 1:
        # for only one feature, the result is the same whatever the shrinkage
        if not assume_centered:
            X = X - X.mean()
        # Return the variance of the single feature and a shrinkage of 0.0
        return np.atleast_2d((X**2).mean()), 0.0

    n_samples, n_features = X.shape

    emp_cov = empirical_covariance(X, assume_centered=assume_centered)

    # The shrinkage is defined as:
    # shrinkage = min(
    # trace(S @ S.T) + trace(S)**2) / ((n + 1) (trace(S @ S.T) - trace(S)**2 / p), 1
    # )
    # where n and p are n_samples and n_features, respectively (cf. Eq. 23 in [1]).
    # The factor 2 / p is omitted since it does not impact the value of the estimator
    # for large p.

    # Instead of computing trace(S)**2, we can compute the average of the squared
    # elements of S that is equal to trace(S)**2 / p**2.
    # See the definition of the Frobenius norm:
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    alpha = np.mean(emp_cov**2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu**2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    # 如果分母 den 为零，则收缩系数 shrinkage 设为 1.0；否则为 num / den 和 1.0 中较小的值
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    # 计算收缩后的协方差矩阵 shrunk_cov，其定义为：
    # (1 - shrinkage) * S + shrinkage * F （参见文献 [1] 中的方程 4）
    # 其中 S 是经验协方差，F 是收缩目标，定义为
    # F = trace(S) / n_features * np.identity(n_features) （参见文献 [1] 中的方程 3）
    shrunk_cov = (1.0 - shrinkage) * emp_cov

    # 在 shrunk_cov 的对角线上加上收缩系数 shrinkage 乘以 mu 的值
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    # 返回收缩后的协方差矩阵 shrunk_cov 和收缩系数 shrinkage
    return shrunk_cov, shrinkage
###############################################################################
# Public API
# ShrunkCovariance estimator

# 导入验证参数的装饰器和相关的数据类型约束
@validate_params(
    {
        "emp_cov": ["array-like"],  # 参数emp_cov的类型约束为array-like
        "shrinkage": [Interval(Real, 0, 1, closed="both")],  # 参数shrinkage的类型约束为0到1之间的闭区间的实数
    },
    prefer_skip_nested_validation=True,
)
# 定义函数shrunk_covariance，用于计算经过对角线收缩的协方差矩阵
def shrunk_covariance(emp_cov, shrinkage=0.1):
    """Calculate covariance matrices shrunk on the diagonal.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    emp_cov : array-like of shape (..., n_features, n_features)
        Covariance matrices to be shrunk, at least 2D ndarray.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Returns
    -------
    shrunk_cov : ndarray of shape (..., n_features, n_features)
        Shrunk covariance matrices.

    Notes
    -----
    The regularized (shrunk) covariance is given by::

        (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where `mu = trace(cov) / n_features`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> from sklearn.covariance import empirical_covariance, shrunk_covariance
    >>> real_cov = np.array([[.8, .3], [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=500)
    >>> shrunk_covariance(empirical_covariance(X))
    array([[0.73..., 0.25...],
           [0.25..., 0.41...]])
    """
    # 检查并将emp_cov转换为合适的数组格式，允许多维数组
    emp_cov = check_array(emp_cov, allow_nd=True)
    # 获取特征数目
    n_features = emp_cov.shape[-1]

    # 计算经过收缩处理后的协方差矩阵
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mu = np.trace(emp_cov, axis1=-2, axis2=-1) / n_features
    mu = np.expand_dims(mu, axis=tuple(range(mu.ndim, emp_cov.ndim)))
    shrunk_cov += shrinkage * mu * np.eye(n_features)

    return shrunk_cov


# 定义ShrunkCovariance类，继承自EmpiricalCovariance类
class ShrunkCovariance(EmpiricalCovariance):
    """Covariance estimator with shrinkage.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)
    """
    #
    def fit(self, X, y=None):
        """Fit the shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 验证并确保输入数据 X 符合要求
        X = self._validate_data(X)
        
        # 如果 assume_centered 为 True，则设置 location_ 为零向量
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            # 否则，计算 X 的每列均值，作为 location_
            self.location_ = X.mean(0)
        
        # 计算 X 的经验协方差矩阵
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        
        # 使用收缩方法对协方差矩阵进行收缩处理
        covariance = shrunk_covariance(covariance, self.shrinkage)
        
        # 将收缩后的协方差矩阵设置给对象的内部变量
        self._set_covariance(covariance)

        # 返回对象实例本身
        return self
# Ledoit-Wolf estimator

# 使用装饰器验证参数，确保参数的类型和范围符合要求
@validate_params(
    {
        "X": ["array-like"],  # 参数X是一个类数组对象，用于计算Ledoit-Wolf收缩协方差矩阵
        "assume_centered": ["boolean"],  # 布尔型参数，指示是否在计算前假定数据已居中
        "block_size": [Interval(Integral, 1, None, closed="left")],  # 整数型参数，用于指定分割协方差矩阵的块大小
    },
    prefer_skip_nested_validation=True,  # 设置偏好跳过嵌套验证
)
def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import ledoit_wolf_shrinkage
    >>> real_cov = np.array([[.4, .2], [.2, .8]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=50)
    >>> shrinkage_coefficient = ledoit_wolf_shrinkage(X)
    >>> shrinkage_coefficient
    0.23...
    """
    X = check_array(X)  # 使用sklearn工具函数确保X是一个数组

    # 如果X的维度是2且特征数量为1，直接返回0.0
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    
    # 如果X的维度是1，将X重新形状为(1, -1)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    # 如果只有一个样本，发出警告建议重新构造数据数组
    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )

    n_samples, n_features = X.shape  # 获取样本数和特征数

    # 如果assume_centered为False，则对数据进行居中处理
    if not assume_centered:
        X = X - X.mean(0)

    # 计算协方差矩阵分割成块的数量
    n_splits = int(n_features / block_size)

    X2 = X**2  # 计算X的平方
    emp_cov_trace = np.sum(X2, axis=0) / n_samples  # 计算每个特征的经验协方差的迹
    mu = np.sum(emp_cov_trace) / n_features  # 计算mu，即协方差的迹除以特征数的平均值
    beta_ = 0.0  # X2.T和X2的系数之和
    delta_ = 0.0  # X.T和X的平方系数之和

    # 开始块计算
    # 循环遍历分割的块数，计算 beta_ 和 delta_
    for i in range(n_splits):
        for j in range(n_splits):
            # 计算当前块的行范围和列范围
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            # 更新 beta_，使用 X2 的部分转置与切片的乘积的和
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            # 更新 delta_，使用 X 的部分转置与切片的乘积的和的平方
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        # 处理当前行范围和最后一个列块之间的乘积和
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits :]))
        delta_ += np.sum(np.dot(X.T[rows], X[:, block_size * n_splits :]) ** 2)

    # 处理剩余的列块与当前行块之间的乘积和
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits :], X2[:, cols]))
        delta_ += np.sum(np.dot(X.T[block_size * n_splits :], X[:, cols]) ** 2)

    # 处理剩余的行块与剩余的列块之间的乘积和
    delta_ += np.sum(
        np.dot(X.T[block_size * n_splits :], X[:, block_size * n_splits :]) ** 2
    )

    # 计算 delta_ 的均值并归一化
    delta_ /= n_samples**2

    # 处理剩余的行块与剩余的列块之间的乘积和
    beta_ += np.sum(
        np.dot(X2.T[block_size * n_splits :], X2[:, block_size * n_splits :])
    )

    # 使用 delta_ 计算 beta
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)

    # 计算 delta，表示 (<X.T,X> - mu*Id) / p 的平方系数和
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features

    # 获取最终的 beta，取 beta 和 delta 的较小值
    # 这样做是为了防止收缩超过 "1"，这会颠倒协方差的值
    beta = min(beta, delta)

    # 最终计算收缩系数 shrinkage
    shrinkage = 0 if beta == 0 else beta / delta

    # 返回收缩系数
    return shrinkage
# 定义 ledoit_wolf 函数，用于估计 Ledoit-Wolf 收缩协方差矩阵
@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
def ledoit_wolf(X, *, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import empirical_covariance, ledoit_wolf
    >>> real_cov = np.array([[.4, .2], [.2, .8]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=50)
    >>> covariance, shrinkage = ledoit_wolf(X)
    >>> covariance
    array([[0.44..., 0.16...],
           [0.16..., 0.80...]])
    >>> shrinkage
    0.23...
    """
    # 创建 LedoitWolf 实例，用于估计协方差和收缩系数
    estimator = LedoitWolf(
        assume_centered=assume_centered,
        block_size=block_size,
        store_precision=False,
    ).fit(X)

    # 返回估计的协方差矩阵和收缩系数
    return estimator.covariance_, estimator.shrinkage_


# LedoitWolf 类，继承自 EmpiricalCovariance 类，用于实现 Ledoit-Wolf 估计器
class LedoitWolf(EmpiricalCovariance):
    """LedoitWolf Estimator.

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.
    """
    # block_size：int，默认为1000
    #   在Ledoit-Wolf估计协方差矩阵时，用于分割协方差矩阵的块的大小。这仅仅是一个内存优化措施，不影响结果。
    
    Attributes
    ----------
    covariance_：shape为(n_features, n_features)的ndarray
    #   估计的协方差矩阵。
    
    location_：shape为(n_features,)的ndarray
    #   估计的位置，即估计的均值。
    
    precision_：shape为(n_features, n_features)的ndarray
    #   估计的伪逆矩阵。仅在store_precision为True时存储。
    
    shrinkage_：float
    #   在计算收缩估计时使用的凸组合中的系数。范围为[0, 1]。
    
    n_features_in_：int
    #   在拟合期间观察到的特征数。
    
        .. versionadded:: 0.24
    
    feature_names_in_：shape为(n_features_in_,)的ndarray
    #   在拟合期间观察到的特征名称。仅当X具有全为字符串的特征名称时定义。
    
        .. versionadded:: 1.0
    
    See Also
    --------
    EllipticEnvelope：用于检测高斯分布数据集中异常值的对象。
    EmpiricalCovariance：最大似然协方差估计器。
    GraphicalLasso：带有l1惩罚估计器的稀疏逆协方差估计。
    GraphicalLassoCV：稀疏逆协方差，带有交叉验证的l1惩罚的选择。
    MinCovDet：最小协方差行列式（协方差的鲁棒估计器）。
    OAS：Oracle近似收缩估计器。
    ShrunkCovariance：带有收缩的协方差估计器。
    
    Notes
    -----
    正则化的协方差是：
    
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    
    其中 mu = trace(cov) / n_features
    shrinkage由Ledoit和Wolf的公式给出（参见References）。
    
    References
    ----------
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import LedoitWolf
    >>> real_cov = np.array([[.4, .2],
    ...                      [.2, .8]])
    >>> np.random.seed(0)
    >>> X = np.random.multivariate_normal(mean=[0, 0],
    ...                                   cov=real_cov,
    ...                                   size=50)
    >>> cov = LedoitWolf().fit(X)
    >>> cov.covariance_
    array([[0.4406..., 0.1616...],
           [0.1616..., 0.8022...]])
    >>> cov.location_
    array([ 0.0595... , -0.0075...])
    
    _parameter_constraints：dict = {
    #   参数约束字典，继承自EmpiricalCovariance的参数约束，并包含新的键值对"block_size"。
    
        **EmpiricalCovariance._parameter_constraints,
        "block_size": [Interval(Integral, 1, None, closed="left")],
    }
    def __init__(self, *, store_precision=True, assume_centered=False, block_size=1000):
        # 调用父类的初始化方法，设置是否存储精度和是否假设数据已居中
        super().__init__(
            store_precision=store_precision, assume_centered=assume_centered
        )
        # 设置对象的块大小属性
        self.block_size = block_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 不调用父对象的fit方法，以避免计算协方差矩阵（以及可能的精度计算）
        X = self._validate_data(X)
        # 如果假设数据已居中，则设置位置属性为零向量
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            # 否则，设置位置属性为数据X每列的均值
            self.location_ = X.mean(0)
        # 使用Ledoit-Wolf方法计算收缩协方差和收缩系数
        covariance, shrinkage = _ledoit_wolf(
            X - self.location_, assume_centered=True, block_size=self.block_size
        )
        # 将计算得到的收缩系数存储在对象的属性中
        self.shrinkage_ = shrinkage
        # 设置对象的协方差矩阵属性
        self._set_covariance(covariance)

        return self
# OAS estimator
@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
# 定义了一个函数 oas，用于使用 Oracle Approximating Shrinkage 方法估计协方差矩阵
def oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.

    Returns
    -------
    shrunk_cov : array-like of shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import oas
    >>> rng = np.random.RandomState(0)
    >>> real_cov = [[.8, .3], [.3, .4]]
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=500)
    >>> shrunk_cov, shrinkage = oas(X)
    >>> shrunk_cov
    array([[0.7533..., 0.2763...],
           [0.2763..., 0.3964...]])
    >>> shrinkage
    0.0195...
    """
    # 创建 OAS 估计器对象，并拟合数据 X
    estimator = OAS(
        assume_centered=assume_centered,
    ).fit(X)
    # 返回估计的协方差矩阵和收缩系数
    return estimator.covariance_, estimator.shrinkage_


# 定义一个类 OAS，继承自 EmpiricalCovariance 类
class OAS(EmpiricalCovariance):
    """Oracle Approximating Shrinkage Estimator.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    Attributes
    ----------
    """
    # 估计的协方差矩阵，形状为 (n_features, n_features)
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    # 估计的位置，即估计的均值，形状为 (n_features,)
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    # 估计的伪逆矩阵，形状为 (n_features, n_features)
    # （仅在 store_precision=True 时存储）
    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    # 用于计算收缩估计中的凸组合中的系数。范围为 [0, 1]。
    shrinkage_ : float
        coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    # 在拟合期间看到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在拟合期间看到的特征的名称。仅当 `X` 的特征名称全为字符串时定义。
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 另请参阅
    # --------
    # EllipticEnvelope : 用于检测高斯分布数据集中异常值的对象。
    # EmpiricalCovariance : 最大似然协方差估计器。
    # GraphicalLasso : 具有 l1 惩罚估计器的稀疏逆协方差估计。
    # GraphicalLassoCV : 具有交叉验证选择 l1 惩罚的稀疏逆协方差。
    # LedoitWolf : LedoitWolf 估计器。
    # MinCovDet : 最小协方差确定（协方差的鲁棒估计器）。
    # ShrunkCovariance : 具有收缩的协方差估计。

    # 注释
    # -----
    # 正则化协方差为：
    #
    # (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),
    #
    # 其中 mu = trace(cov) / n_features，shrinkage 由 OAS 公式给出
    # （参见 [1]_）。
    #
    # 此处实现的收缩形式与 [1]_ 中的公式（23）不同。在原始文章中，公式（23）
    # 指出，2/p（p 为特征数）在分子和分母中均乘以 Trace(cov*cov)，但此操作被省略，
    # 因为对于较大的 p，2/p 的值太小，不会影响估计器的值。

    # 参考文献
    # ----------
    # .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
    #        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
    #        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
    #        <0907.4698>`

    # 示例
    # --------
    # >>> import numpy as np
    # >>> from sklearn.covariance import OAS
    # >>> from sklearn.datasets import make_gaussian_quantiles
    # >>> real_cov = np.array([[.8, .3],
    # ...                      [.3, .4]])
    # >>> rng = np.random.RandomState(0)
    # >>> X = rng.multivariate_normal(mean=[0, 0],
    # ...                             cov=real_cov,
    # ...                             size=500)
    # >>> oas = OAS().fit(X)
    # >>> oas.covariance_
    # array([[0.7533..., 0.2763...],
    #        [0.2763..., 0.3964...]])
    # >>> oas.precision_
    # array([[ 1.7833..., -1.2431... ],
    #        [-1.2431...,  3.3889...]])
    # >>> oas.shrinkage_
    # 0.0195...
    """
    # 应用装饰器 @_fit_context，设置参数 prefer_skip_nested_validation=True
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义 fit 方法，用于训练 Oracle Approximating Shrinkage 协方差模型，拟合输入数据 X
    def fit(self, X, y=None):
        """Fit the Oracle Approximating Shrinkage covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 验证输入数据 X，并确保数据的合法性
        X = self._validate_data(X)
        
        # 如果 assume_centered 为 True，则设置数据中心位置为零向量
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            # 否则，计算数据 X 每列的均值作为中心位置
            self.location_ = X.mean(0)

        # 计算协方差矩阵和收缩系数，使用 Oracle Approximating Shrinkage 方法
        covariance, shrinkage = _oas(X - self.location_, assume_centered=True)
        
        # 将计算得到的收缩系数 shrinkage 存储在实例变量 shrinkage_ 中
        self.shrinkage_ = shrinkage
        
        # 调用 _set_covariance 方法，将计算得到的协方差矩阵 covariance 设置给实例
        self._set_covariance(covariance)

        # 返回实例本身，表示训练过程完成并返回训练后的模型实例
        return self
```