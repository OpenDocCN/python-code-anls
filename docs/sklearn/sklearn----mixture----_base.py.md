# `D:\src\scipysrc\scikit-learn\sklearn\mixture\_base.py`

```
"""Base class for mixture models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入抽象基类元类和抽象方法装饰器
from abc import ABCMeta, abstractmethod
# 导入整数和实数类型
from numbers import Integral, Real
# 导入时间模块中的时间函数
from time import time

# 导入NumPy库并指定别名np
import numpy as np
# 从SciPy库中导入logsumexp函数
from scipy.special import logsumexp

# 导入相关的聚类模块
from .. import cluster
# 从基本估计器模块中导入基本估计器类、密度混合类和拟合上下文
from ..base import BaseEstimator, DensityMixin, _fit_context
# 从聚类模块中导入kmeans_plusplus方法
from ..cluster import kmeans_plusplus
# 导入收敛警告异常类
from ..exceptions import ConvergenceWarning
# 从工具模块中导入随机状态检查函数
from ..utils import check_random_state
# 从参数验证模块中导入区间和字符串选项验证类
from ..utils._param_validation import Interval, StrOptions
# 从验证模块中导入检查是否拟合函数
from ..utils.validation import check_is_fitted


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array
        Input parameter to validate.

    param_shape : tuple
        Expected shape of the parameter.

    name : str
        Name of the parameter.
    """
    # 将参数转换为NumPy数组
    param = np.array(param)
    # 检查参数的形状是否与预期形状一致
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s"
            % (name, param_shape, param.shape)
        )


class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    # 定义参数约束字典
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init_params": [
            StrOptions({"kmeans", "random", "random_from_data", "k-means++"})
        ],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
        "verbose_interval": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        n_init,
        init_params,
        random_state,
        warm_start,
        verbose,
        verbose_interval,
    ):
        """Initialize a BaseMixture instance.

        Parameters
        ----------
        n_components : int
            Number of mixture components.

        tol : float
            Convergence threshold. EM iterations will stop when the lower bound
            average gain is below this threshold.

        reg_covar : float
            Non-negative regularization added to the diagonal of covariance.
            Allows to assure that the covariance matrices are always positive.

        max_iter : int
            Maximum number of EM iterations.

        n_init : int
            Number of initializations to perform. The best results across
            initializations are kept.

        init_params : str
            Method for initialization:
            - 'kmeans': Initialize using k-means.
            - 'random': Random initialization.
            - 'random_from_data': Random initialization from the data.
            - 'k-means++': Improved k-means++ initialization.

        random_state : int, RandomState instance, or None
            Controls the random seed given at initialization.

        warm_start : bool
            If True, reuse the solution of the previous fit as initialization.

        verbose : bool
            Enable verbose output during fitting.

        verbose_interval : int
            Interval in number of iterations between verbose output.

        """
        # 初始化各参数
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
            Training data.

        """
        pass
    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape  # 获取输入数据的样本数量

        if self.init_params == "kmeans":
            # 初始化响应矩阵为全零矩阵
            resp = np.zeros((n_samples, self.n_components))
            # 使用 K-means 算法进行聚类，得到每个样本点的聚类标签
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            # 将响应矩阵中对应的标签位置置为1
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            # 从给定的随机状态生成均匀分布的随机数，初始化响应矩阵
            resp = random_state.uniform(size=(n_samples, self.n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]  # 归一化响应矩阵
        elif self.init_params == "random_from_data":
            # 初始化响应矩阵为全零矩阵
            resp = np.zeros((n_samples, self.n_components))
            # 从数据中随机选择样本索引，将响应矩阵中对应位置置为1
            indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
            resp[indices, np.arange(self.n_components)] = 1
        elif self.init_params == "k-means++":
            # 初始化响应矩阵为全零矩阵
            resp = np.zeros((n_samples, self.n_components))
            # 使用 K-means++ 算法初始化聚类中心，得到每个样本点的聚类标签
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            # 将响应矩阵中对应的标签位置置为1
            resp[indices, np.arange(self.n_components)] = 1

        self._initialize(X, resp)  # 调用初始化函数进行后续操作

    @abstractmethod
    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        resp : array-like of shape (n_samples, n_components)
            Matrix of responsibilities.

        This method is meant to be overridden by derived classes to
        initialize model-specific parameters.
        """
        pass
    def fit(self, X, y=None):
        """
        Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # 调用 fit_predict 方法进行参数估计
        self.fit_predict(X, y)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def _e_step(self, X):
        """
        E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # 调用 _estimate_log_prob_resp 方法获取对数概率和对数后验概率
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    @abstractmethod
    def _m_step(self, X, log_resp):
        """
        M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        """
        Abstract method for getting parameters.
        """
        pass

    @abstractmethod
    def _set_parameters(self, params):
        """
        Abstract method for setting parameters.

        Parameters
        ----------
        params : object
            Parameters to set.
        """
        pass

    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 验证数据 X，并且不重置数据验证状态
        X = self._validate_data(X, reset=False)

        # 返回每个样本的对数似然
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    def score(self, X, y=None):
        """
        计算给定数据 X 的每个样本的平均对数似然。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            包含 n_features 维度数据点的列表。每一行对应一个单独的数据点。

        y : Ignored
            不使用，仅出于 API 一致性而保留。

        Returns
        -------
        log_likelihood : float
            在高斯混合模型下，X 的对数似然。
        """
        return self.score_samples(X).mean()

    def predict(self, X):
        """
        使用训练好的模型预测数据样本 X 的标签。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            包含 n_features 维度数据点的列表。每一行对应一个单独的数据点。

        Returns
        -------
        labels : array, shape (n_samples,)
            组件标签。
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        评估每个样本的组件密度。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            包含 n_features 维度数据点的列表。每一行对应一个单独的数据点。

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            X 中每个样本的每个高斯分量的密度。
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)
    # 定义一个方法用于从拟合的高斯分布中生成随机样本
    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 如果样本数小于1，则抛出值错误
        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        # 获取均值矩阵的形状，即 (n_components, n_features)
        _, n_features = self.means_.shape
        # 获取随机数生成器，根据给定的随机状态
        rng = check_random_state(self.random_state)
        # 使用多项式分布生成各高斯分布组件的样本数
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        # 根据协方差类型生成随机样本
        if self.covariance_type == "full":
            # 对每个高斯分布组件生成多元正态分布样本，并将结果堆叠成数组 X
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )
        elif self.covariance_type == "tied":
            # 对每个高斯分布组件生成多元正态分布样本，并将结果堆叠成数组 X
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            # 对每个高斯分布组件生成独立同分布样本，并将结果堆叠成数组 X
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )

        # 生成组件标签 y，用于表示每个样本属于哪个高斯分布组件
        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        # 返回生成的样本数据 X 和对应的组件标签 y
        return (X, y)

    # 抽象方法：估计加权对数概率，即估计 log P(X | Z) + log weights
    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        # 调用 _estimate_log_prob 方法计算对数概率，并加上 _estimate_log_weights 方法计算的对数权重
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    # 抽象方法：估计对数权重，在 EM 算法中使用，或在 VB 算法中估计 E[ log pi ]
    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass
    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
            Array containing log-probabilities of each sample under each component.
        """
        pass



    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            Array of log-likelihoods for each sample.
        
        log_responsibilities : array, shape (n_samples, n_components)
            Logarithm of the responsibilities matrix.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp



    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time



    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time



    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        """Print verbose message on the end of iteration."""
        converged_msg = "converged" if init_has_converged else "did not converge"
        if self.verbose == 1:
            print(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            print(
                f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound"
                f" {lb:.5f}."
            )
```