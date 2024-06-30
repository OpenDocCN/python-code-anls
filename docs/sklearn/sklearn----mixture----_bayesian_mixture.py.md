# `D:\src\scipysrc\scikit-learn\sklearn\mixture\_bayesian_mixture.py`

```
"""Bayesian Gaussian Mixture Model."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math                              # 导入数学库，用于数学计算
from numbers import Real                 # 从numbers模块导入Real类，用于验证实数类型

import numpy as np                       # 导入NumPy库，用于数值计算
from scipy.special import betaln, digamma, gammaln   # 从SciPy库中导入特定函数

from ..utils import check_array          # 从当前目录上级的utils模块中导入check_array函数
from ..utils._param_validation import Interval, StrOptions   # 从当前目录上级的utils/_param_validation模块导入Interval和StrOptions类
from ._base import BaseMixture, _check_shape   # 从当前目录的_base模块中导入BaseMixture类和_check_shape函数
from ._gaussian_mixture import (
    _check_precision_matrix,            # 从当前目录的_gaussian_mixture模块导入_check_precision_matrix函数
    _check_precision_positivity,         # 从当前目录的_gaussian_mixture模块导入_check_precision_positivity函数
    _compute_log_det_cholesky,           # 从当前目录的_gaussian_mixture模块导入_compute_log_det_cholesky函数
    _compute_precision_cholesky,         # 从当前目录的_gaussian_mixture模块导入_compute_precision_cholesky函数
    _estimate_gaussian_parameters,       # 从当前目录的_gaussian_mixture模块导入_estimate_gaussian_parameters函数
    _estimate_log_gaussian_prob,         # 从当前目录的_gaussian_mixture模块导入_estimate_log_gaussian_prob函数
)


def _log_dirichlet_norm(dirichlet_concentration):
    """Compute the log of the Dirichlet distribution normalization term.

    Parameters
    ----------
    dirichlet_concentration : array-like of shape (n_samples,)
        The parameters values of the Dirichlet distribution.

    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return gammaln(np.sum(dirichlet_concentration)) - np.sum(
        gammaln(dirichlet_concentration)
    )


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precision_chol : array-like of shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),
            0,
        )
    )


class BayesianGaussianMixture(BaseMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    # 混合高斯模型的组件数量，默认为1
    n_components : int, default=1
        # 混合模型的组件数量。根据数据和weight_concentration_prior的值，
        # 模型可以决定通过将某些组件的权重设置为接近零的值来不使用所有组件。
        # 因此，有效组件的数量可能小于n_components。

    # 协方差类型，可以是'full'、'tied'、'diag'、'spherical'之一，默认为'full'
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        # 描述要使用的协方差参数类型的字符串。
        # 必须是以下之一：
        #   'full'（每个组件有自己的通用协方差矩阵），
        #   'tied'（所有组件共享相同的通用协方差矩阵），
        #   'diag'（每个组件有自己的对角协方差矩阵），
        #   'spherical'（每个组件有自己的单一方差）。

    # 收敛阈值，默认为1e-3
    tol : float, default=1e-3
        # 收敛阈值。当对数似然（关于模型的训练数据的）的下界的平均增益低于此阈值时，EM迭代将停止。

    # 协方差矩阵的正则化项，默认为1e-6
    reg_covar : float, default=1e-6
        # 非负正则化项，添加到协方差的对角线上。
        # 可以确保所有协方差矩阵都是正定的。

    # EM迭代的最大次数，默认为100
    max_iter : int, default=100
        # 执行EM算法的最大迭代次数。

    # 初始化次数，默认为1
    n_init : int, default=1
        # 执行初始化的次数。保留对似然性下界值最高的结果。

    # 初始化参数的方法，默认为'kmeans'
    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        # 用于初始化权重、均值和协方差的方法。
        # 字符串必须是以下之一：
        #   'kmeans'：使用k均值算法初始化责任。
        #   'k-means++'：使用k均值++方法初始化。
        #   'random'：随机初始化责任。
        #   'random_from_data'：从数据中随机选择初始均值点。

        # .. versionchanged:: v1.1
        #    `init_params`现在接受'random_from_data'和'k-means++'作为初始化方法。

    # 权重集中先验类型，默认为'dirichlet_process'
    weight_concentration_prior_type : {'dirichlet_process', 'dirichlet_distribution'}, \
            default='dirichlet_process'
        # 描述权重集中先验类型的字符串。

    # 每个组件在权重分布（Dirichlet分布）上的Dirichlet浓度，默认为None
    weight_concentration_prior : float or None, default=None
        # 每个组件在权重分布（Dirichlet分布）上的Dirichlet浓度。
        # 在文献中通常称为gamma。较高的浓度将在中心放置更多的质量，
        # 并导致更多的组件处于活跃状态，而较低的浓度参数将导致在混合权重单纯体的边缘上有更多的质量。
        # 参数的值必须大于0。如果为None，则设置为``1. / n_components``。
    mean_precision_prior : float or None, default=None
        # 均值分布（高斯分布）的精度先验。
        # 控制均值可能放置的范围。较大的值使得聚类均值集中在 `mean_prior` 的周围。
        # 参数的值必须大于0。如果为None，则设置为1。

    mean_prior : array-like, shape (n_features,), default=None
        # 均值分布（高斯分布）的先验。
        # 如果为None，则设置为 X 的均值。

    degrees_of_freedom_prior : float or None, default=None
        # 协方差分布（Wishart分布）的自由度先验。
        # 如果为None，则设置为 `n_features`。

    covariance_prior : float or array-like, default=None
        # 协方差分布（Wishart分布）的先验。
        # 如果为None，则使用 X 的经验协方差作为初始化。
        # 形状取决于 `covariance_type`::
        #     (n_features, n_features) 如果是 'full',
        #     (n_features, n_features) 如果是 'tied',
        #     (n_features)             如果是 'diag',
        #     float                    如果是 'spherical'

    random_state : int, RandomState instance or None, default=None
        # 控制用于初始化参数的方法（见 `init_params`）的随机种子。
        # 另外，它控制从拟合分布中生成随机样本（见 `sample` 方法）。
        # 传递一个整数以在多次函数调用中产生可重现的输出。
        # 参见 :term:`Glossary <random_state>`。

    warm_start : bool, default=False
        # 如果 'warm_start' 为 True，则使用上次拟合的解作为下一次调用 `fit()` 的初始化。
        # 当在类似问题上多次调用 `fit` 时，这可以加快收敛速度。
        # 参见 :term:`the Glossary <warm_start>`。

    verbose : int, default=0
        # 启用详细输出。
        # 如果为1，则打印当前初始化和每个迭代步骤。
        # 如果大于1，则还打印对数概率和每步所需的时间。

    verbose_interval : int, default=10
        # 每次打印输出之前的迭代次数。

Attributes
----------
    weights_ : array-like of shape (n_components,)
        # 每个混合成分的权重。

    means_ : array-like of shape (n_components, n_features)
        # 每个混合成分的均值。

    covariances_ : array-like
        # 每个混合成分的协方差。
        # 形状取决于 `covariance_type`::
        #     (n_components,)                        如果是 'spherical',
        #     (n_features, n_features)               如果是 'tied',
        #     (n_components, n_features)             如果是 'diag',
        #     (n_components, n_features, n_features) 如果是 'full'
    # 精度矩阵，用于混合高斯模型的每个成分。精度矩阵是协方差矩阵的逆矩阵。协方差矩阵是对称正定的，因此可以等价地使用精度矩阵参数化混合高斯模型。
    # 存储精度矩阵而不是协方差矩阵，在测试时计算新样本的对数似然更有效率。形状依赖于 `covariance_type` 参数：
    #   - (n_components,)                        如果是 'spherical'
    #   - (n_features, n_features)               如果是 'tied'
    #   - (n_components, n_features)             如果是 'diag'
    #   - (n_components, n_features, n_features) 如果是 'full'
    precisions_ : array-like

    # 每个混合成分的精度矩阵的Cholesky分解。精度矩阵是协方差矩阵的逆矩阵。协方差矩阵是对称正定的，因此可以等价地使用精度矩阵参数化混合高斯模型。
    # 存储精度矩阵而不是协方差矩阵，在测试时计算新样本的对数似然更有效率。形状依赖于 `covariance_type` 参数：
    #   - (n_components,)                        如果是 'spherical'
    #   - (n_features, n_features)               如果是 'tied'
    #   - (n_components, n_features)             如果是 'diag'
    #   - (n_components, n_features, n_features) 如果是 'full'
    precisions_cholesky_ : array-like

    # 指示是否达到了最佳拟合推断的收敛性的布尔值。如果为True，则达到了收敛；否则为False。
    converged_ : bool

    # 达到收敛所用的步数，由最佳拟合推断使用。
    n_iter_ : int

    # 最佳拟合推断模型证据（训练数据的）的下界值。
    lower_bound_ : float

    # 权重分布（Dirichlet分布）中每个成分的Dirichlet集中度。类型取决于 `weight_concentration_prior_type` 参数：
    #   - (float, float) 如果是 'dirichlet_process'（Beta参数）
    #   - float          如果是 'dirichlet_distribution'（Dirichlet参数）
    # 集中度较高会在中心投放更多质量，导致更多成分活跃；而集中度较低则在单纯体边缘投放更多质量。
    weight_concentration_prior_ : tuple or float

    # 权重分布（Dirichlet分布）中每个成分的Dirichlet集中度的数组形式，形状为 (n_components,)。
    weight_concentration_ : array-like
    # GaussianMixture模型中的参数，控制均值分布（高斯分布）的精度先验
    mean_precision_prior_ : float
        # 均值可以放置的范围的控制器
        Controls the extent of where means can be placed.
        # 如果mean_precision_prior为None，则将mean_precision_prior_设置为1
        Larger values concentrate the cluster means around `mean_prior`.
        If mean_precision_prior is set to None, `mean_precision_prior_` is set
        to 1.

    # GaussianMixture模型中每个组件的均值分布（高斯分布）的精度
    mean_precision_ : array-like of shape (n_components,)
    
    # GaussianMixture模型中均值分布（高斯分布）的先验
    mean_prior_ : array-like of shape (n_features,)
    
    # covariance分布（Wishart分布）中自由度的先验
    degrees_of_freedom_prior_ : float
    
    # GaussianMixture模型中每个组件的协方差分布（Wishart分布）的自由度
    degrees_of_freedom_ : array-like of shape (n_components,)
    
    # covariance分布（Wishart分布）中的先验，根据covariance_type的不同而不同
    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    # 训练期间观察到的特征数目
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 训练期间观察到的特征名称数组
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <https://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <https://citeseerx.ist.psu.edu/doc_view/pid/ee844fd96db7041a9681b5a18bff008912052c7e>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import BayesianGaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [12, 4], [10, 7]])
    >>> bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(X)
    >>> bgm.means_
    array([[2.49... , 2.29...],
           [8.45..., 4.52... ]])
    >>> bgm.predict([[0, 0], [9, 3]])
    array([0, 1])
    """
    # 定义参数约束字典，继承自基类的参数约束
    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        # 协方差类型约束为字符串选项，包括{"spherical", "tied", "diag", "full"}
        "covariance_type": [StrOptions({"spherical", "tied", "diag", "full"})],
        # 权重集中先验类型约束为字符串选项，包括{"dirichlet_process", "dirichlet_distribution"}
        "weight_concentration_prior_type": [
            StrOptions({"dirichlet_process", "dirichlet_distribution"})
        ],
        # 权重集中先验的值约束为实数区间 [0.0, +∞)，左闭右开
        "weight_concentration_prior": [
            None,
            Interval(Real, 0.0, None, closed="neither"),
        ],
        # 均值精度先验的值约束为实数区间 [0.0, +∞)，左闭右开
        "mean_precision_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        # 均值先验的值约束为数组形式
        "mean_prior": [None, "array-like"],
        # 自由度先验的值约束为实数区间 [0.0, +∞)，左闭右开
        "degrees_of_freedom_prior": [None, Interval(Real, 0.0, None, closed="neither")],
        # 协方差先验的值约束为数组形式或实数区间 [0.0, +∞)，左闭右开
        "covariance_prior": [
            None,
            "array-like",
            Interval(Real, 0.0, None, closed="neither"),
        ],
    }

    # 初始化函数，设置混合模型的参数
    def __init__(
        self,
        *,
        n_components=1,  # 组件数，默认为1
        covariance_type="full",  # 协方差类型，默认为"full"
        tol=1e-3,  # 收敛容限，默认为1e-3
        reg_covar=1e-6,  # 协方差矩阵的正则化项，默认为1e-6
        max_iter=100,  # 最大迭代次数，默认为100
        n_init=1,  # 初始化次数，默认为1
        init_params="kmeans",  # 初始化参数方法，默认为"kmeans"
        weight_concentration_prior_type="dirichlet_process",  # 权重集中先验类型，默认为"dirichlet_process"
        weight_concentration_prior=None,  # 权重集中先验的值，默认为None
        mean_precision_prior=None,  # 均值精度先验的值，默认为None
        mean_prior=None,  # 均值先验的值，默认为None
        degrees_of_freedom_prior=None,  # 自由度先验的值，默认为None
        covariance_prior=None,  # 协方差先验的值，默认为None
        random_state=None,  # 随机数种子，默认为None
        warm_start=False,  # 是否启用热启动，默认为False
        verbose=0,  # 冗余级别，默认为0
        verbose_interval=10,  # 冗余输出间隔，默认为10
    ):
        # 调用父类的初始化方法，设置通用的参数
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

        # 设置特定于该类的参数
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior

    # 检查模型参数是否合理的内部方法
    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        """
        # 检查权重参数的有效性
        self._check_weights_parameters()
        # 检查均值参数的有效性
        self._check_means_parameters(X)
        # 检查精度参数的有效性
        self._check_precision_parameters(X)
        # 检查协方差先验参数的有效性
        self._checkcovariance_prior_parameter(X)

    # 检查权重参数的有效性的内部方法
    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""
        # 如果权重集中先验为None，则设定默认值
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1.0 / self.n_components
        else:
            # 否则使用给定的权重集中先验值
            self.weight_concentration_prior_ = self.weight_concentration_prior
    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data array containing samples and features.
        """
        # 获取数据集的特征数
        _, n_features = X.shape

        # 如果未指定均值精度的先验值，则设定为默认值 1.0
        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.0
        else:
            # 否则使用指定的均值精度先验值
            self.mean_precision_prior_ = self.mean_precision_prior

        # 如果未指定均值的先验值，则设定为输入数据每列的均值
        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            # 否则检查并转换均值先验值的数据类型和维度
            self.mean_prior_ = check_array(
                self.mean_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(self.mean_prior_, (n_features,), "means")

    def _check_precision_parameters(self, X):
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data array containing samples and features.
        """
        # 获取数据集的特征数
        _, n_features = X.shape

        # 如果未指定自由度的先验值，则设定为特征数
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.degrees_of_freedom_prior > n_features - 1.0:
            # 如果指定的自由度先验值大于特征数减一，则使用指定值
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            # 否则抛出异常，自由度先验值应大于特征数减一
            raise ValueError(
                "The parameter 'degrees_of_freedom_prior' "
                "should be greater than %d, but got %.3f."
                % (n_features - 1, self.degrees_of_freedom_prior)
            )
    def _checkcovariance_prior_parameter(self, X):
        """Check and set the `covariance_prior_` parameter based on given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data array.

        """
        # Determine number of features from input data shape
        _, n_features = X.shape

        # If covariance_prior is not provided, initialize it based on covariance_type
        if self.covariance_prior is None:
            self.covariance_prior_ = {
                "full": np.atleast_2d(np.cov(X.T)),
                "tied": np.atleast_2d(np.cov(X.T)),
                "diag": np.var(X, axis=0, ddof=1),
                "spherical": np.var(X, axis=0, ddof=1).mean(),
            }[self.covariance_type]

        # If covariance_prior is provided, validate its shape and properties based on covariance_type
        elif self.covariance_type in ["full", "tied"]:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features, n_features),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_matrix(self.covariance_prior_, self.covariance_type)
        elif self.covariance_type == "diag":
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features,),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_positivity(self.covariance_prior_, self.covariance_type)
        # If covariance_type is "spherical", use covariance_prior directly
        else:
            self.covariance_prior_ = self.covariance_prior

    def _initialize(self, X, resp):
        """Initialize the parameters of the mixture model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data array.

        resp : array-like of shape (n_samples, n_components)
            Responsibilities of the samples for each component.
        """
        # Estimate Gaussian parameters nk, xk, sk from input data and responsibilities
        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )

        # Estimate weights of the mixture model
        self._estimate_weights(nk)

        # Estimate means of the mixture components
        self._estimate_means(nk, xk)

        # Estimate precisions (inverse covariance matrices) of the mixture components
        self._estimate_precisions(nk, xk, sk)

    def _estimate_weights(self, nk):
        """Estimate the weights of the mixture components.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Number of samples assigned to each component.
        """
        # Adjust weight_concentration_ based on the prior type
        if self.weight_concentration_prior_type == "dirichlet_process":
            # For dirichlet process, weight_concentration_ is adjusted using a beta distribution
            self.weight_concentration_ = (
                1.0 + nk,
                (
                    self.weight_concentration_prior_
                    + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
                ),
            )
        else:
            # For other types, adjust weight_concentration_ using the prior directly
            self.weight_concentration_ = self.weight_concentration_prior_ + nk
    def _estimate_means(self, nk, xk):
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Array containing the number of observations for each component.

        xk : array-like of shape (n_components, n_features)
            Array containing the observations for each component and feature.
        """
        # Update mean precision using prior and counts
        self.mean_precision_ = self.mean_precision_prior_ + nk

        # Estimate means using weighted combination of prior mean and new observations
        self.means_ = (
            self.mean_precision_prior_ * self.mean_prior_
            + nk[:, np.newaxis] * xk
        ) / self.mean_precision_[:, np.newaxis]

    def _estimate_precisions(self, nk, xk, sk):
        """Estimate the precision parameters of the precision distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Array containing the number of observations for each component.

        xk : array-like of shape (n_components, n_features)
            Array containing the observations for each component and feature.

        sk : array-like
            The shape depends on `covariance_type`:
            'full' : (n_components, n_features, n_features)
            'tied' : (n_features, n_features)
            'diag' : (n_components, n_features)
            'spherical' : (n_components,)
            Precision matrix(s) related to the covariance structure.
        """
        # Dispatch estimation based on covariance type
        {
            "full": self._estimate_wishart_full,
            "tied": self._estimate_wishart_tied,
            "diag": self._estimate_wishart_diag,
            "spherical": self._estimate_wishart_spherical,
        }[self.covariance_type](nk, xk, sk)

        # Compute Cholesky decomposition of precision matrices
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_wishart_full(self, nk, xk, sk):
        """Estimate the full Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Array containing the number of observations for each component.

        xk : array-like of shape (n_components, n_features)
            Array containing the observations for each component and feature.

        sk : array-like of shape (n_components, n_features, n_features)
            Array containing the precision matrices for each component.
        """
        _, n_features = xk.shape

        # Update degrees of freedom using prior and counts
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        # Initialize empty array for covariances
        self.covariances_ = np.empty((self.n_components, n_features, n_features))

        # Compute covariances for each component
        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covariances_[k] = (
                self.covariance_prior_
                + nk[k] * sk[k]
                + nk[k]
                * self.mean_precision_prior_
                / self.mean_precision_[k]
                * np.outer(diff, diff)
            )

        # Normalize covariances by degrees of freedom
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis, np.newaxis]
    def _estimate_wishart_tied(self, nk, xk, sk):
        """Estimate the tied Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_features, n_features)
        """
        # 获取特征数量
        _, n_features = xk.shape

        # 警告：在一些Bishop的书中，公式10.63有一个错字
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # 是正确的公式
        # 更新自由度参数
        self.degrees_of_freedom_ = (
            self.degrees_of_freedom_prior_ + nk.sum() / self.n_components
        )

        # 计算均值差异
        diff = xk - self.mean_prior_

        # 更新协方差矩阵
        self.covariances_ = (
            self.covariance_prior_
            + sk * nk.sum() / self.n_components
            + self.mean_precision_prior_
            / self.n_components
            * np.dot((nk / self.mean_precision_) * diff.T, diff)
        )

        # 与原始的Bishop书中相反，我们对协方差矩阵进行归一化处理
        self.covariances_ /= self.degrees_of_freedom_

    def _estimate_wishart_diag(self, nk, xk, sk):
        """Estimate the diag Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features)
        """
        # 获取特征数量
        _, n_features = xk.shape

        # 警告：在一些Bishop的书中，公式10.63有一个错字
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # 是正确的公式
        # 更新自由度参数
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        # 计算均值差异
        diff = xk - self.mean_prior_

        # 更新协方差矩阵
        self.covariances_ = self.covariance_prior_ + nk[:, np.newaxis] * (
            sk
            + (self.mean_precision_prior_ / self.mean_precision_)[:, np.newaxis]
            * np.square(diff)
        )

        # 与原始的Bishop书中相反，我们对协方差矩阵进行归一化处理
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis]
    def _estimate_wishart_spherical(self, nk, xk, sk):
        """Estimate the spherical Wishart distribution parameters.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
            Array containing the number of observations for each component.

        xk : array-like of shape (n_components, n_features)
            Array containing the mean parameters for each component.

        sk : array-like of shape (n_components,)
            Array containing scale parameters for each component.
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        # Calculate degrees of freedom for the Wishart distribution
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        # Calculate covariance matrices for each component
        diff = xk - self.mean_prior_
        self.covariances_ = self.covariance_prior_ + nk * (
            sk
            + self.mean_precision_prior_
            / self.mean_precision_
            * np.mean(np.square(diff), 1)
        )

        # Contrary to the original bishop book, we normalize the covariances
        # Normalize the estimated covariances by dividing by the degrees of freedom
        self.covariances_ /= self.degrees_of_freedom_

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (responsibilities) of
            each sample in X.
        """
        n_samples, _ = X.shape

        # Estimate Gaussian parameters nk, xk, sk using log_resp
        nk, xk, sk = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )

        # Update weights, means, and precisions based on the estimated parameters
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_weights(self):
        """Estimate log weights.

        Returns
        -------
        log_weights : array-like of shape (n_components,)
            Logarithm of the weights for each component.
        """
        if self.weight_concentration_prior_type == "dirichlet_process":
            # Calculate log weights for Dirichlet process mixture
            digamma_sum = digamma(
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (
                digamma_a
                - digamma_sum
                + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
            )
        else:
            # Calculate log weights for Variational Gaussian mixture with dirichlet distribution
            return digamma(self.weight_concentration_) - digamma(
                np.sum(self.weight_concentration_)
            )
    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # 计算高斯分布的对数概率，去除 `n_features * np.log(self.degrees_of_freedom_)` 是因为精度矩阵已经归一化
        log_gauss = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        # 计算 lambda 的对数值
        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5
                * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )

        # 返回估计的对数概率
        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to increase at
        each iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # 与原始公式相反，我们进行了简化，并移除了所有常数项。
        (n_features,) = self.mean_prior_.shape

        # 移除 `.5 * n_features * np.log(self.degrees_of_freedom_)` 是因为精度矩阵已经归一化。
        log_det_precisions_chol = _compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        # 根据协方差类型计算对数 Wishart 分布值
        if self.covariance_type == "tied":
            log_wishart = self.n_components * np.float64(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )
        else:
            log_wishart = np.sum(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )

        # 根据权重集中先验类型计算对数权重值
        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -np.sum(
                betaln(self.weight_concentration_[0], self.weight_concentration_[1])
            )
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        # 返回模型的下界估计值
        return (
            -np.sum(np.exp(log_resp) * log_resp)
            - log_wishart
            - log_norm_weight
            - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )
    # 获取当前模型的参数并返回一个元组
    def _get_parameters(self):
        return (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    # 根据给定的参数设置当前模型的参数
    def _set_parameters(self, params):
        (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # 计算权重（混合系数）
        if self.weight_concentration_prior_type == "dirichlet_process":
            # 对于狄利克雷过程，计算权重的分量和
            weight_dirichlet_sum = (
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            # 计算权重的每个成分
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0]
                / weight_dirichlet_sum
                * np.hstack((1, np.cumprod(tmp[:-1])))
            )
            # 归一化权重
            self.weights_ /= np.sum(self.weights_)
        else:
            # 对于其他先验类型，直接归一化权重
            self.weights_ = self.weight_concentration_ / np.sum(
                self.weight_concentration_
            )

        # 计算精度矩阵
        if self.covariance_type == "full":
            # 对于完全协方差类型，通过Cholesky分解计算精度矩阵
            self.precisions_ = np.array(
                [
                    np.dot(prec_chol, prec_chol.T)
                    for prec_chol in self.precisions_cholesky_
                ]
            )

        elif self.covariance_type == "tied":
            # 对于相同协方差类型，通过Cholesky分解计算精度矩阵
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            # 对于对角协方差类型，直接平方Cholesky分解得到精度矩阵
            self.precisions_ = self.precisions_cholesky_**2
```