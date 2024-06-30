# `D:\src\scipysrc\scikit-learn\sklearn\covariance\_empirical_covariance.py`

```
"""
Maximum likelihood covariance estimator.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 避免除法截断警告
import warnings

# 导入必要的库
import numpy as np
from scipy import linalg

# 导入配置上下文，基本估计器和拟合上下文
from .. import config_context
from ..base import BaseEstimator, _fit_context

# 导入计算成对距离的函数
from ..metrics.pairwise import pairwise_distances

# 导入数据验证和处理工具函数
from ..utils import check_array

# 导入参数验证函数
from ..utils._param_validation import validate_params

# 导入快速计算对数行列式的函数
from ..utils.extmath import fast_logdet


@validate_params(
    {
        "emp_cov": [np.ndarray],
        "precision": [np.ndarray],
    },
    prefer_skip_nested_validation=True,
)
def log_likelihood(emp_cov, precision):
    """Compute the sample mean of the log_likelihood under a covariance model.

    Computes the empirical expected log-likelihood, allowing for universal
    comparison (beyond this software package), and accounts for normalization
    terms and scaling.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Maximum Likelihood Estimator of covariance.

    precision : ndarray of shape (n_features, n_features)
        The precision matrix of the covariance model to be tested.

    Returns
    -------
    log_likelihood_ : float
        Sample mean of the log-likelihood.
    """
    # 计算特征数量
    p = precision.shape[0]

    # 计算对数似然值
    log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_


@validate_params(
    {
        "X": ["array-like"],
        "assume_centered": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def empirical_covariance(X, *, assume_centered=False):
    """Compute the Maximum likelihood covariance estimator.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : bool, default=False
        If `True`, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If `False`, data will be centered before computation.

    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).

    Examples
    --------
    >>> from sklearn.covariance import empirical_covariance
    >>> X = [[1,1,1],[1,1,1],[1,1,1],
    ...      [0,0,0],[0,0,0],[0,0,0]]
    >>> empirical_covariance(X)
    array([[0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25]])
    """
    # 检查并转换输入数据为数组形式
    X = check_array(X, ensure_2d=False, force_all_finite=False)

    # 如果输入是一维数组，则转换为二维数组
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    # 如果只有一行样本数据，则发出警告建议重塑数据数组
    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )

    # 根据是否假定数据已中心化选择计算协方差的方法
    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)
    # 如果协方差矩阵是标量（ndim为0），则将其转换为一个形状为(1, 1)的二维NumPy数组
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    # 返回处理后的协方差矩阵（可能已经转换为二维数组）
    return covariance
class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator.

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EmpiricalCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7569..., 0.2818...],
           [0.2818..., 0.3928...]])
    >>> cov.location_
    array([0.0622..., 0.0193...])
    """

    _parameter_constraints: dict = {
        "store_precision": ["boolean"],
        "assume_centered": ["boolean"],
    }

    def __init__(self, *, store_precision=True, assume_centered=False):
        # 初始化方法，设置对象的属性
        self.store_precision = store_precision
        self.assume_centered = assume_centered
    def _set_covariance(self, covariance):
        """
        Saves the covariance and precision estimates

        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.

        Parameters
        ----------
        covariance : array-like of shape (n_features, n_features)
            Estimated covariance matrix to be stored, and from which precision
            is computed.
        """
        # 将传入的协方差矩阵转换成数组，并进行数据验证
        covariance = check_array(covariance)
        # 将协方差矩阵保存到实例变量 covariance_ 中
        self.covariance_ = covariance
        # 如果需要存储精度矩阵
        if self.store_precision:
            # 计算协方差矩阵的伪逆作为精度矩阵，忽略无限的检查
            self.precision_ = linalg.pinvh(covariance, check_finite=False)
        else:
            # 否则将精度矩阵设为 None
            self.precision_ = None

    def get_precision(self):
        """
        Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        # 如果需要存储精度矩阵，则返回已存储的精度矩阵
        if self.store_precision:
            precision = self.precision_
        else:
            # 否则重新计算协方差矩阵的伪逆作为精度矩阵，忽略无限的检查
            precision = linalg.pinvh(self.covariance_, check_finite=False)
        return precision

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the maximum likelihood covariance estimator to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where `n_samples` is the number of samples and
          `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 验证并转换输入数据 X
        X = self._validate_data(X)
        # 如果假设数据已居中
        if self.assume_centered:
            # 设置位置参数为特征维度上的全零向量
            self.location_ = np.zeros(X.shape[1])
        else:
            # 否则计算数据 X 在特征维度上的均值作为位置参数
            self.location_ = X.mean(0)
        # 计算数据 X 的经验协方差矩阵
        covariance = empirical_covariance(X, assume_centered=self.assume_centered)
        # 将计算得到的经验协方差矩阵设置为实例的协方差矩阵
        self._set_covariance(covariance)

        return self
    # 定义一个方法用于计算 `X_test` 数据在估计的高斯模型下的对数似然度。
    # 高斯模型由其均值和协方差矩阵定义，分别由 `self.location_` 和 `self.covariance_` 表示。
    def score(self, X_test, y=None):
        """
        Compute the log-likelihood of `X_test` under the estimated Gaussian model.

        The Gaussian model is defined by its mean and covariance matrix which are
        represented respectively by `self.location_` and `self.covariance_`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where `n_samples` is
            the number of samples and `n_features` is the number of features.
            `X_test` is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        res : float
            The log-likelihood of `X_test` with `self.location_` and `self.covariance_`
            as estimators of the Gaussian model mean and covariance matrix respectively.
        """
        # 验证并处理测试数据 `X_test`
        X_test = self._validate_data(X_test, reset=False)
        
        # 计算测试集的经验协方差
        test_cov = empirical_covariance(X_test - self.location_, assume_centered=True)
        
        # 计算对数似然度
        res = log_likelihood(test_cov, self.get_precision())

        # 返回计算结果
        return res
    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.

        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool, default=True
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool, default=True
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        result : float
            The Mean Squared Error (in the sense of the Frobenius norm) between
            `self` and `comp_cov` covariance estimators.
        """
        # compute the error between comp_cov and the stored covariance
        error = comp_cov - self.covariance_
        
        # compute the error norm based on the specified norm type
        if norm == "frobenius":
            squared_norm = np.sum(error**2)  # Frobenius norm: sum of squared differences
        elif norm == "spectral":
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
            # Spectral norm: maximum singular value of the error matrix
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented"
            )
        
        # optionally scale the error norm by dividing by the number of features
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        
        # determine whether to return the squared norm or its square root
        if squared:
            result = squared_norm  # return the squared norm
        else:
            result = np.sqrt(squared_norm)  # return the square root of the norm
        
        return result

    def mahalanobis(self, X):
        """Compute the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of which we compute.
            Observations are assumed to be drawn from the same distribution as
            the data used in fit.

        Returns
        -------
        dist : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        X = self._validate_data(X, reset=False)  # validate input data
        
        precision = self.get_precision()  # retrieve precision matrix
        
        with config_context(assume_finite=True):
            # compute Mahalanobis distances using pairwise_distances function
            dist = pairwise_distances(
                X, self.location_[np.newaxis, :], metric="mahalanobis", VI=precision
            )
        
        return np.reshape(dist, (len(X),)) ** 2  # return squared Mahalanobis distances
```