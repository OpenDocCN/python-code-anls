# `D:\src\scipysrc\scikit-learn\sklearn\mixture\_gaussian_mixture.py`

```
"""Gaussian Mixture Model."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import linalg

from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape

###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class


def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    # 将weights转换为合适的数组类型，并确保是一维数组
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    # 检查weights数组的形状是否符合预期
    _check_shape(weights, (n_components,), "weights")

    # 检查权重值的范围是否在 [0, 1] 之间
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # 检查权重值是否归一化
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    # 将means转换为合适的数组类型，并确保是二维数组
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    # 检查means数组的形状是否符合预期
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    # 检查精度向量中的所有元素是否大于零
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    # 检查精度矩阵是否对称并且所有特征值是否大于零
    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, positive-definite" % covariance_type
        )


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    # 遍历检查每个精度矩阵是否对称并且所有特征值是否大于零
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Check the precision matrices are of the correct shape."""
    pass  # Placeholder function, not implemented here
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : str

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
        Validated and formatted precision matrix based on user input.
    """
    # 使用 check_array 函数验证并转换输入的 precisions
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )

    # 定义不同 covariance_type 下的预期形状
    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    # 根据 covariance_type 检查 precisions 的形状是否符合预期
    _check_shape(
        precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
    )

    # 定义不同 covariance_type 下的精度检查函数
    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    # 调用相应 covariance_type 的精度检查函数
    _check_precisions[covariance_type](precisions, covariance_type)
    
    # 返回经过验证和处理的 precisions 数组
    return precisions
###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

# 估计完整协方差矩阵
def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
        每个样本对每个组件的响应度量

    X : array-like of shape (n_samples, n_features)
        数据集，每行表示一个样本

    nk : array-like of shape (n_components,)
        每个组件的样本权重

    means : array-like of shape (n_components, n_features)
        每个组件的均值向量

    reg_covar : float
        协方差矩阵的正则化项

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        当前组件的协方差矩阵
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances


# 估计相同协方差矩阵
def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
        每个样本对每个组件的响应度量

    X : array-like of shape (n_samples, n_features)
        数据集，每行表示一个样本

    nk : array-like of shape (n_components,)
        每个组件的样本权重

    means : array-like of shape (n_components, n_features)
        每个组件的均值向量

    reg_covar : float
        协方差矩阵的正则化项

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        组件的相同协方差矩阵
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


# 估计对角线协方差向量
def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)
        每个样本对每个组件的响应度量

    X : array-like of shape (n_samples, n_features)
        数据集，每行表示一个样本

    nk : array-like of shape (n_components,)
        每个组件的样本权重

    means : array-like of shape (n_components, n_features)
        每个组件的均值向量

    reg_covar : float
        协方差矩阵的正则化项

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        组件的对角线协方差向量
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


# 估计球面方差值
def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)
        每个样本对每个组件的响应度量

    X : array-like of shape (n_samples, n_features)
        数据集，每行表示一个样本

    nk : array-like of shape (n_components,)
        每个组件的样本权重

    means : array-like of shape (n_components, n_features)
        每个组件的均值向量

    reg_covar : float
        协方差矩阵的正则化项

    Returns
    -------
    """
    variances : array, shape (n_components,)
        组件的每个方差值。
    """
    使用 `_estimate_gaussian_covariances_diag` 函数计算每个高斯分布组件的对角协方差矩阵。
    返回这些方差值的均值，即每个组件的平均方差。
# 估计高斯分布的参数。

def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array. 输入数据数组，形状为 (样本数, 特征数)

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X. 对每个数据样本在 X 中的责任度量，形状为 (样本数, 组件数)

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices. 添加到协方差矩阵对角线的正则化参数

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices. 协方差矩阵的类型

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components. 当前组件中的数据样本数量

    means : array-like of shape (n_components, n_features)
        The centers of the current components. 当前组件的中心

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
        当前组件的协方差矩阵，其形状取决于 covariance_type
    """
    # 计算每个组件中的数据样本数量
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    # 计算每个组件的中心点坐标
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # 根据 covariance_type 选择对应的协方差估计方法
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type. 当前组件的协方差矩阵，其形状取决于 covariance_type

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices. 协方差矩阵的类型

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
        当前组件样本精度的 Cholesky 分解，其形状取决于 covariance_type
    """
    # 估计精度错误信息提示
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    if covariance_type == "full":
        # 获取协方差矩阵的维度信息
        n_components, n_features, _ = covariances.shape
        # 初始化精度的 Cholesky 分解数组
        precisions_chol = np.empty((n_components, n_features, n_features))
        # 对每个组件的协方差进行 Cholesky 分解
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                # 如果 Cholesky 分解失败，则抛出 ValueError
                raise ValueError(estimate_precision_error_message)
            # 计算 Cholesky 分解后的精度矩阵
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    # 如果协方差类型为 "tied"
    elif covariance_type == "tied":
        # 获取协方差矩阵的行数和特征数目
        _, n_features = covariances.shape
        try:
            # 尝试使用 Cholesky 分解计算协方差的下三角矩阵
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            # 如果 Cholesky 分解失败，抛出值错误并附带错误消息
            raise ValueError(estimate_precision_error_message)
        # 计算精度矩阵的 Cholesky 分解的转置
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    # 如果协方差类型不是 "tied"
    else:
        # 如果协方差矩阵中有任何元素小于等于零，抛出值错误并附带错误消息
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        # 计算对角元素为协方差的逆矩阵的 Cholesky 分解
        precisions_chol = 1.0 / np.sqrt(covariances)
    # 返回计算得到的精度矩阵的 Cholesky 分解
    return precisions_chol
def _flipudlr(array):
    """Reverse the rows and columns of an array."""
    # 使用 NumPy 函数 np.fliplr 对数组进行左右翻转，再对结果使用 np.flipud 进行上下翻转
    return np.flipud(np.fliplr(array))


def _compute_precision_cholesky_from_precisions(precisions, covariance_type):
    r"""Compute the Cholesky decomposition of precisions using precisions themselves.

    As implemented in :func:`_compute_precision_cholesky`, the `precisions_cholesky_` is
    an upper-triangular matrix for each Gaussian component, which can be expressed as
    the $UU^T$ factorization of the precision matrix for each Gaussian component, where
    $U$ is an upper-triangular matrix.

    In order to use the Cholesky decomposition to get $UU^T$, the precision matrix
    $\Lambda$ needs to be permutated such that its rows and columns are reversed, which
    can be done by applying a similarity transformation with an exchange matrix $J$,
    where the 1 elements reside on the anti-diagonal and all other elements are 0. In
    particular, the Cholesky decomposition of the transformed precision matrix is
    $J\Lambda J=LL^T$, where $L$ is a lower-triangular matrix. Because $\Lambda=UU^T$
    and $J=J^{-1}=J^T$, the `precisions_cholesky_` for each Gaussian component can be
    expressed as $JLJ$.

    Refer to #26415 for details.

    Parameters
    ----------
    precisions : array-like
        The precision matrix of the current components.
        The shape depends on the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends on the covariance_type.
    """
    if covariance_type == "full":
        # 对于 'full' 协方差类型，对每个精度矩阵应用 _flipudlr 函数并进行 Cholesky 分解
        precisions_cholesky = np.array(
            [
                _flipudlr(linalg.cholesky(_flipudlr(precision), lower=True))
                for precision in precisions
            ]
        )
    elif covariance_type == "tied":
        # 对于 'tied' 协方差类型，直接对精度矩阵应用 _flipudlr 函数并进行 Cholesky 分解
        precisions_cholesky = _flipudlr(
            linalg.cholesky(_flipudlr(precisions), lower=True)
        )
    else:
        # 对于 'diag' 和 'spherical' 协方差类型，直接对精度矩阵应用平方根运算
        precisions_cholesky = np.sqrt(precisions)
    return precisions_cholesky


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_chol : array-like
        The log-determinant of the cholesky decomposition of matrices.
        The shape depends on the covariance_type.
    """
    # 计算每个混合成分的精度矩阵的行列式的对数值
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    # 如果协方差类型为 "full"
    if covariance_type == "full":
        # 获取矩阵 matrix_chol 的维度信息
        n_components, _, _ = matrix_chol.shape
        # 计算每个混合成分的对数行列式
        log_det_chol = np.sum(
            # 对矩阵重新形状，并按照指定步长选取元素，然后计算对数
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    # 如果协方差类型为 "tied"
    elif covariance_type == "tied":
        # 计算矩阵的对角线元素的对数，并求和
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    # 如果协方差类型为 "diag"
    elif covariance_type == "diag":
        # 计算每个混合成分的对数行列式
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    # 如果协方差类型未知或未列出
    else:
        # 计算矩阵的每个元素的对数，然后乘以特征数
        log_det_chol = n_features * (np.log(matrix_chol))

    # 返回每个混合成分的对数行列式
    return log_det_chol
def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        输入数据集，每行表示一个样本，每列表示一个特征。

    means : array-like of shape (n_components, n_features)
        高斯分布的均值向量。

    precisions_chol : array-like
        精度矩阵的Cholesky分解。
        'full' : 形状为 (n_components, n_features, n_features)
        'tied' : 形状为 (n_features, n_features)
        'diag' : 形状为 (n_components, n_features)
        'spherical' : 形状为 (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        协方差类型，决定了如何计算精度矩阵。

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
        对数概率值数组，每行对应一个样本，每列对应一个混合成分。
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    # 从Cholesky分解的精度矩阵计算对数行列式
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )

    # 使用Cholesky分解的精度矩阵，因此 `- 0.5 * log_det_precision` 相当于 `+ log_det_precision_chol`
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each component has its own diagonal covariance matrix.
        - 'spherical': each component has its own single variance.
        描述协方差参数类型的字符串，可以是以下之一：
        - 'full': 每个组件有自己的一般协方差矩阵。
        - 'tied': 所有组件共享相同的一般协方差矩阵。
        - 'diag': 每个组件有自己的对角线协方差矩阵。
        - 'spherical': 每个组件有自己的单一方差。

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
        收敛阈值。当下限平均增益低于此阈值时，EM 迭代将停止。

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.
        添加到协方差对角线的非负正则化项，确保所有协方差矩阵都是正的。

    max_iter : int, default=100
        The number of EM iterations to perform.
        执行的EM迭代次数。

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.
        执行的初始化次数。保留最佳结果。

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.
        初始化权重、均值和精度的方法。
        字符串必须是以下之一：
        - 'kmeans' : 使用kmeans初始化责任。
        - 'k-means++' : 使用k-means++方法进行初始化。
        - 'random' : 随机初始化责任。
        - 'random_from_data' : 从随机选择的数据点初始化均值。

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.
        版本变更说明：`init_params` 现在接受 'random_from_data' 和 'k-means++' 作为初始化方法。

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.
        用户提供的初始权重。
        如果为None，则使用 `init_params` 方法初始化权重。

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.
        用户提供的初始均值。
        如果为None，则使用 `init_params` 方法初始化均值。

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
        用户提供的初始精度（协方差矩阵的逆）。
        如果为None，则使用 'init_params' 方法初始化精度。
        形状取决于 'covariance_type'：
            (n_components,)                        如果 'spherical',
            (n_features, n_features)               如果 'tied',
            (n_components, n_features)             如果 'diag',
            (n_components, n_features, n_features) 如果 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
        控制用于初始化参数的方法选择的随机种子（参见 `init_params`）。
        此外，它还控制从拟合分布生成随机样本（参见方法 `sample`）。
        传递一个整数以在多次函数调用中生成可重复的输出。
        参见 :term:`术语表 <random_state>`。
    # 如果 'warm_start' 为 True，则在下一次调用 fit() 时使用上次拟合的解作为初始化值。
    # 这可以加快在相似问题上多次调用 fit() 时的收敛速度。
    # 在这种情况下，'n_init' 被忽略，仅在第一次调用时发生初始化。
    # 参见术语表中的 'warm_start' 。
    warm_start : bool, default=False

    # verbose 控制详细输出的级别。如果为 1，则打印当前初始化和每次迭代步骤。
    # 如果大于 1，则还打印每步的对数概率和所需时间。
    verbose : int, default=0

    # 每次打印输出之间的迭代次数。
    verbose_interval : int, default=10

    # weights_ 是每个混合组件的权重数组，形状为 (n_components,)
    Attributes
    ----------
    weights_ : array-like of shape (n_components,)

    # means_ 是每个混合组件的均值数组，形状为 (n_components, n_features)
    means_ : array-like of shape (n_components, n_features)

    # covariances_ 是每个混合组件的协方差数组。
    # 其形状取决于 'covariance_type'：
    #   - (n_components,)                        如果 'spherical',
    #   - (n_features, n_features)               如果 'tied',
    #   - (n_components, n_features)             如果 'diag',
    #   - (n_components, n_features, n_features) 如果 'full'
    covariances_ : array-like

    # precisions_ 是每个混合组件的精度矩阵数组。
    # 精度矩阵是协方差矩阵的逆矩阵。由于协方差矩阵是对称正定的，因此高斯混合可以等效地由精度矩阵参数化。
    # 形状取决于 'covariance_type'：
    #   - (n_components,)                        如果 'spherical',
    #   - (n_features, n_features)               如果 'tied',
    #   - (n_components, n_features)             如果 'diag',
    #   - (n_components, n_features, n_features) 如果 'full'
    precisions_ : array-like

    # precisions_cholesky_ 是每个混合组件的精度矩阵的Cholesky分解数组。
    # 精度矩阵是协方差矩阵的逆矩阵。由于协方差矩阵是对称正定的，因此高斯混合可以等效地由精度矩阵参数化。
    # 存储精度矩阵而不是协方差矩阵使得在测试时更高效地计算新样本的对数似然。
    # 形状取决于 'covariance_type'：
    #   - (n_components,)                        如果 'spherical',
    #   - (n_features, n_features)               如果 'tied',
    #   - (n_components, n_features)             如果 'diag',
    #   - (n_components, n_features, n_features) 如果 'full'
    precisions_cholesky_ : array-like
    # 表示EM算法的最佳拟合是否收敛的布尔值
    converged_ : bool
    # 最佳拟合的EM算法达到收敛时使用的步数
    n_iter_ : int
    # 最佳拟合的EM算法的对数似然的下界值（与模型相关的训练数据的对数似然）
    lower_bound_ : float
    # 在“拟合”期间看到的特征数量
    n_features_in_ : int
    # 在“拟合”期间看到的特征的名称数组。仅在X具有全为字符串的特征名称时定义。
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
    
    See Also
    --------
    BayesianGaussianMixture : 使用变分推断拟合的高斯混合模型。
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])
    
    
    
    # _parameter_constraints: dict 类型，包含参数的约束条件
    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "covariance_type": [StrOptions({"full", "tied", "diag", "spherical"})],
        "weights_init": ["array-like", None],
        "means_init": ["array-like", None],
        "precisions_init": ["array-like", None],
    }
    
    
    
    # GaussianMixture类的构造函数，初始化高斯混合模型
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        # 调用父类BaseMixture的构造函数，设置初始参数
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        
        # 设置高斯混合模型的特定参数
        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        # 获取输入数据 X 的形状信息
        _, n_features = X.shape

        # 如果给定了初始权重参数，则检查并调整为合适的形式
        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        # 如果给定了初始均值参数，则检查并调整为合适的形式
        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        # 如果给定了初始精度参数，则检查并调整为合适的形式
        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize_parameters(self, X, random_state):
        """Initialize parameters for Gaussian mixture model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        random_state : RandomState instance or int, optional
            RandomState instance or seed used to initialize the centers.
        """
        # 如果任一初始参数未提供，则进行初始化
        compute_resp = (
            self.weights_init is None
            or self.means_init is None
            or self.precisions_init is None
        )
        if compute_resp:
            # 调用父类方法进行参数初始化
            super()._initialize_parameters(X, random_state)
        else:
            # 否则直接使用给定的初始参数进行初始化
            self._initialize(X, None)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        resp : array-like of shape (n_samples, n_components), optional
            Responsibilities for each sample.
        """
        # 获取样本数和特征数
        n_samples, _ = X.shape
        # 初始化权重、均值、协方差矩阵为 None
        weights, means, covariances = None, None, None
        # 如果给定了 resp，则根据 resp 估计高斯分布的参数
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, self.reg_covar, self.covariance_type
            )
            # 如果未指定初始权重，则归一化权重
            if self.weights_init is None:
                weights /= n_samples

        # 根据初始参数设置模型的权重、均值
        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        # 如果未指定初始精度参数，则计算协方差矩阵及其 Cholesky 分解
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        else:
            # 否则根据给定的精度参数计算 Cholesky 分解
            self.precisions_cholesky_ = _compute_precision_cholesky_from_precisions(
                self.precisions_init, self.covariance_type
            )

    def _m_step(self, X, log_resp):
        """M step of the expectation-maximization algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities)
            of each sample in X.
        """
        # 根据给定的 resp 估计高斯混合模型的参数：权重、均值、协方差矩阵
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        # 归一化权重
        self.weights_ /= self.weights_.sum()
        # 根据估计的协方差矩阵计算其 Cholesky 分解
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
    # 使用内部方法 _estimate_log_gaussian_prob 估计给定数据 X 的对数概率
    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    # 计算当前模型的对数权重
    def _estimate_log_weights(self):
        return np.log(self.weights_)

    # 返回对数概率的规范化形式
    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    # 获取当前模型的参数元组
    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    # 设置当前模型的参数
    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # 计算附加属性
        _, n_features = self.means_.shape

        # 根据协方差类型计算精度矩阵
        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)
        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    # 返回模型中的自由参数数量
    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    # 计算当前模型在输入数据 X 上的贝叶斯信息准则
    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )
    # 计算当前模型在输入数据 X 上的赤池信息准则（AIC）。

    # 可以参考 :ref:`mathematical section <aic_bic>` 查看关于使用 AIC 的详细数学公式。

    # Parameters 参数:
    # ----------
    # X : array of shape (n_samples, n_dimensions)
    #     输入样本数据。
    
    # Returns 返回:
    # -------
    # aic : float
    #     AIC 值，数值越小越好。

    return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
```