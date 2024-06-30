# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\_gpc.py`

```
"""Gaussian processes classification."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的模块和类
from numbers import Integral
from operator import itemgetter

# 导入科学计算库
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit

# 导入scikit-learn中的基本类和函数
from ..base import BaseEstimator, ClassifierMixin, _fit_context, clone
from ..multiclass import OneVsOneClassifier, OneVsRestClassifier
from ..preprocessing import LabelEncoder
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted

# 导入核函数相关模块
from .kernels import RBF, CompoundKernel, Kernel
from .kernels import ConstantKernel as C

# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
# 预先计算的lambda和coef用于逼近逻辑sigmoid函数
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, np.newaxis]


class _BinaryGaussianProcessClassifierLaplace(BaseEstimator):
    """Binary Gaussian process classification based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
"""
    optimizer : 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
        # 优化器选择，默认为 'fmin_l_bfgs_b' 字符串或可调用对象，用于优化核函数的参数。
        可以传入一个可调用对象作为外部定义的优化器。如果传入可调用对象，必须具有以下签名：

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' 是待最大化的目标函数，接受超参数 theta 和一个
                #   可选的 eval_gradient 标志，用于确定是否额外返回梯度。
                # * 'initial_theta'：theta 的初始值，可以被局部优化器使用。
                # * 'bounds'：theta 值的范围限制。
                ....
                # 返回找到的最佳超参数 theta 和对应的目标函数值。
                return theta_opt, func_min

        # 默认情况下，使用 scipy.optimize.minimize 中的 'L-BFGS-B' 算法。
        如果传入 None，则保持核函数的参数固定。
        内部支持的优化器有：

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        # 用于找到最大化对数边际似然的核参数的优化器重新启动次数。
        第一次优化从核函数的初始参数开始，剩余的次数（如果有）从允许的 theta 值空间中
        对数均匀采样的 theta 开始。如果大于 0，则所有边界必须是有限的。
        注意，n_restarts_optimizer=0 表示只运行一次优化。

    max_iter_predict : int, default=100
        # 在预测期间用于近似后验的牛顿方法的最大迭代次数。
        较小的值会减少计算时间，但会牺牲结果的准确性。

    warm_start : bool, default=False
        # 如果启用热启动，则会使用后验模式的拉普拉斯近似的最后一次牛顿迭代的解
        作为下一次调用 _posterior_mode() 的初始化。当像超参数优化中多次调用
        _posterior_mode 解决类似问题时，这可以加快收敛速度。参见“术语表”中的“热启动”。

    copy_X_train : bool, default=True
        # 如果为 True，则在对象中存储训练数据的持久副本。否则，仅存储对训练数据的引用，
        这可能导致在外部修改数据时预测结果发生变化。
    random_state : int, RandomState instance or None, default=None
        # 控制随机数生成，用于初始化中心点。
        # 设为整数以便在多次函数调用中获得可重现的结果。
        # 参见“术语表”中的“随机状态”。

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        # 训练数据的特征向量或其他表示形式（预测时也需要）。

    y_train_ : array-like of shape (n_samples,)
        # 训练数据中的目标值（预测时也需要）。

    classes_ : array-like of shape (n_classes,)
        # 唯一的类标签。

    kernel_ : kernl instance
        # 用于预测的核函数。
        # 核函数的结构与传入的参数相同，但具有经过优化的超参数。

    L_ : array-like of shape (n_samples, n_samples)
        # X_train_ 的核矩阵的下三角 Cholesky 分解。

    pi_ : array-like of shape (n_samples,)
        # 正类别的概率，针对训练点 X_train_。

    W_sr_ : array-like of shape (n_samples,)
        # W 的平方根，即观测标签的对数似然函数的 Hessian 的对角线。
        # 由于 W 是对角的，因此仅存储 sqrt(W) 的对角线。

    log_marginal_likelihood_value_ : float
        # self.kernel_.theta 的对数边际似然值。

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    """

    def __init__(
        self,
        kernel=None,
        *,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
    ):
        # 初始化方法，设置对象的各种参数
        self.kernel = kernel  # 设置核函数
        self.optimizer = optimizer  # 优化器的选择，默认为"fmin_l_bfgs_b"
        self.n_restarts_optimizer = n_restarts_optimizer  # 优化器重启次数，默认为0
        self.max_iter_predict = max_iter_predict  # 预测的最大迭代次数，默认为100
        self.warm_start = warm_start  # 是否热启动，默认为False
        self.copy_X_train = copy_X_train  # 是否复制训练数据，默认为True
        self.random_state = random_state  # 随机数种子，用于初始化中心点，控制随机性
    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 根据 Gaussian Process for Machine Learning 第 3.4.2 节讨论，
        # 对于做出硬性二进制决策，只需计算后验的最大后验概率，并通过链接函数传递
        K_star = self.kernel_(self.X_train_, X)  # 计算测试点与训练点的核函数
        f_star = K_star.T.dot(self.y_train_ - self.pi_)  # 算法 3.2，第 4 行

        # 根据 f_star 的正负决定预测类别
        return np.where(f_star > 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 基于 Gaussian Process for Machine Learning 算法 3.2
        K_star = self.kernel_(self.X_train_, X)  # 计算测试点与训练点的核函数
        f_star = K_star.T.dot(self.y_train_ - self.pi_)  # 第 4 行
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # 第 5 行
        # 第 6 行 (通过 einsum 计算 np.diag(v.T.dot(v)))
        var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)

        # 第 7 行:
        # 近似计算 \int log(z) * N(z | f_star, var_f_star)
        # 近似方法来自 Williams & Barber 的 "Bayesian Classification
        # with Gaussian Processes"，附录 A：通过线性组合 5 个误差函数近似逻辑 S 型函数。
        # 有关如何计算此积分的信息，请参见
        # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = (
            np.sqrt(np.pi / alpha)
            * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2)))
            / (2 * np.sqrt(var_f_star * 2 * np.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        return np.vstack((1 - pi_star, pi_star)).T

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,), default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,), \
                optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when `eval_gradient` is True.
        """
        # If theta is None, return precomputed log marginal likelihood if eval_gradient is False
        if theta is None:
            if eval_gradient:
                # Raise error if eval_gradient is True and theta is None
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        # Clone or modify the kernel object based on clone_kernel flag
        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        # Evaluate the kernel function with or without gradient
        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Compute the posterior mode and store temporaries for computing gradient
        Z, (pi, W_sr, L, b, a) = self._posterior_mode(K, return_temporaries=True)

        # If eval_gradient is False, return only the log-marginal likelihood Z
        if not eval_gradient:
            return Z

        # Compute the gradient of log-marginal likelihood with respect to theta
        d_Z = np.empty(theta.shape[0])
        # Compute R using Woodbury matrix identity and other temporaries
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))  # Line 7
        C = solve(L, W_sr[:, np.newaxis] * K)  # Line 8
        # Compute s_2 which involves third derivatives
        s_2 = (
            -0.5
            * (np.diag(K) - np.einsum("ij, ij -> j", C, C))
            * (pi * (1 - pi) * (1 - 2 * pi))
        )  # third derivative

        # Compute gradient components for each parameter in theta
        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]  # Line 11
            # Compute s_1 which is a component of the gradient
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * R.T.ravel().dot(C.ravel())  # Line 12

            # Compute b and s_3 which are components of the gradient
            b = C.dot(self.y_train_ - pi)  # Line 13
            s_3 = b - K.dot(R.dot(b))  # Line 14

            # Combine s_1, s_2, and s_3 to compute d_Z[j]
            d_Z[j] = s_1 + s_2.T.dot(s_3)  # Line 15

        # Return log-marginal likelihood Z and its gradient d_Z
        return Z, d_Z
    def _posterior_mode(self, K, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.

        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if (
            self.warm_start
            and hasattr(self, "f_cached")
            and self.f_cached.shape == self.y_train_.shape
        ):
            f = self.f_cached  # Reuse cached solution
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64)  # Initialize f to zeros

        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            # Line 4: Compute sigmoid function applied to f
            pi = expit(f)
            # Line 5: Compute W matrix for Laplace approximation
            W = pi * (1 - pi)
            # Line 6: Compute square root of W
            W_sr = np.sqrt(W)
            # Construct matrix B for Laplace approximation
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            # Perform Cholesky decomposition on B
            L = cholesky(B, lower=True)
            # Line 7: Compute b vector
            b = W * f + (self.y_train_ - pi)
            # Line 8: Solve for a vector using Cholesky decomposition
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 9: Update f using kernel matrix K and a vector
            f = K.dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            lml = (
                -0.5 * a.T.dot(f)
                - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum()
                - np.log(np.diag(L)).sum()
            )
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        if return_temporaries:
            return log_marginal_likelihood, (pi, W_sr, L, b, a)
        else:
            return log_marginal_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min
class GaussianProcessClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 from [RW2006]_.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting. Also kernel
        cannot be a `CompoundKernel`.

    optimizer : 'fmin_l_bfgs_b', callable or None, default='fmin_l_bfgs_b'
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.
    """
    # 最大迭代次数，用于在预测过程中使用牛顿法近似后验分布。较小的值可以减少计算时间，但可能导致较差的结果。
    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    # 是否启用热启动。如果启用，会使用后验模式的拉普拉斯近似的最后牛顿迭代的解作为下一次调用_posterior_mode()的初始化值。
    # 这可以加速在类似问题上多次调用_posterior_mode，例如超参数优化。参见“术语表”中的“热启动”。
    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.

    # 如果为True，训练数据的持久副本将存储在对象中。否则，只存储对训练数据的引用，这可能导致在外部修改数据时预测结果发生变化。
    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    # 随机数生成器的种子，用于初始化中心点。传递一个整数以确保多次函数调用生成可重复的结果。参见“术语表”中的“随机状态”。
    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    # 多类分类问题的处理方式。支持'one_vs_rest'和'one_vs_one'两种方式。
    # 在'one_vs_rest'中，为每个类别拟合一个二元高斯过程分类器，用于将该类与其余类别分开。
    # 在'one_vs_one'中，为每对类别拟合一个二元高斯过程分类器，用于将这两个类别分开。
    # 这些二元预测器的预测结果被合并为多类预测。注意，'one_vs_one'不支持预测概率估计。
    multi_class : {'one_vs_rest', 'one_vs_one'}, default='one_vs_rest'
        Specifies how multi-class classification problems are handled.
        Supported are 'one_vs_rest' and 'one_vs_one'. In 'one_vs_rest',
        one binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest. In 'one_vs_one', one
        binary Gaussian process classifier is fitted for each pair of classes,
        which is trained to separate these two classes. The predictions of
        these binary predictors are combined into multi-class predictions.
        Note that 'one_vs_one' does not support predicting probability
        estimates.

    # 计算过程中使用的作业数。指定多类问题并行计算的作业数。
    # ``None``表示默认为1，除非在``joblib.parallel_backend``上下文中。
    # ``-1``表示使用所有处理器。参见“术语表”中的“作业数”以获取更多详情。
    n_jobs : int, default=None
        The number of jobs to use for the computation: the specified
        multiclass problems are computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    # 属性
    # --------------
    # 定义似然函数的估计器实例，使用观测数据。
    base_estimator_ : ``Estimator`` instance
        The estimator instance that defines the likelihood function
        using the observed data.

    # 用于预测的核函数。在二元分类中，核函数的结构与传递的参数相同，但超参数已经优化。
    # 在多类分类中，返回一个CompoundKernel，其中包含在一个对多个分类器中使用的不同核函数。
    kernel_ : kernel instance
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.

    # ``self.kernel_.theta``的对数边缘似然值。
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    # 定义一个成员变量，用于存储唯一的类标签数组
    classes_ : array-like of shape (n_classes,)
    
    # 记录训练数据中类别的数量
    n_classes_ : int
    
    # 记录在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    # 如果输入数据有特征名称且全为字符串，则记录这些特征名称
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    # 相关链接，指向高斯过程回归器的文档
    See Also
    --------
    GaussianProcessRegressor : Gaussian process regression (GPR).
    
    # 参考文献，引用高斯过程在机器学习中的应用
    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    
    # 使用示例，展示如何使用高斯过程分类器进行分类
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.83548752, 0.03228706, 0.13222543],
           [0.79064206, 0.06525643, 0.14410151]])
    
    # 参数约束字典，定义了高斯过程分类器接受的参数类型和取值范围
    _parameter_constraints: dict = {
        "kernel": [Kernel, None],
        "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "max_iter_predict": [Interval(Integral, 1, None, closed="left")],
        "warm_start": ["boolean"],
        "copy_X_train": ["boolean"],
        "random_state": ["random_state"],
        "multi_class": [StrOptions({"one_vs_rest", "one_vs_one"})],
        "n_jobs": [Integral, None],
    }
    
    # 初始化方法，初始化高斯过程分类器的各个参数
    def __init__(
        self,
        kernel=None,
        *,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
    ):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
    
    # 带有装饰器的私有方法，用于上下文中的拟合过程，优先跳过嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit Gaussian process classification model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,)
            Target values, must be binary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # 如果 kernel 是 CompoundKernel 类型，则抛出数值错误异常
        if isinstance(self.kernel, CompoundKernel):
            raise ValueError("kernel cannot be a CompoundKernel")

        # 检查 kernel 是否为空或者需要向量输入，根据情况验证数据类型
        if self.kernel is None or self.kernel.requires_vector_input:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=True, dtype="numeric"
            )
        else:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=False, dtype=None
            )

        # 初始化基础估计器，根据参数配置创建 _BinaryGaussianProcessClassifierLaplace 对象
        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )

        # 计算类别的唯一值并确定类别数量
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size

        # 如果类别数量为1，则抛出数值错误异常
        if self.n_classes_ == 1:
            raise ValueError(
                "GaussianProcessClassifier requires 2 or more "
                "distinct classes; got %d class (only class %s "
                "is present)" % (self.n_classes_, self.classes_[0])
            )

        # 如果类别数量大于2，根据 multi_class 参数选择适当的多类别分类器
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        # 使用基础估计器拟合数据
        self.base_estimator_.fit(X, y)

        # 如果类别数量大于2，计算对数边际似然值的平均值
        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [
                    estimator.log_marginal_likelihood()
                    for estimator in self.base_estimator_.estimators_
                ]
            )
        else:
            # 如果类别数量为2，则直接计算对数边际似然值
            self.log_marginal_likelihood_value_ = (
                self.base_estimator_.log_marginal_likelihood()
            )

        # 返回当前对象实例
        return self
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``.
        """
        # 检查模型是否已经拟合，如果未拟合则引发异常
        check_is_fitted(self)

        # 根据核函数属性，确定是否需要对输入数据 X 进行验证和转换
        if self.kernel is None or self.kernel.requires_vector_input:
            X = self._validate_data(X, ensure_2d=True, dtype="numeric", reset=False)
        else:
            X = self._validate_data(X, ensure_2d=False, dtype=None, reset=False)

        # 调用基础估计器的 predict 方法进行预测
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        # 检查模型是否已经拟合，如果未拟合则引发异常
        check_is_fitted(self)

        # 如果是多类别问题且使用 one_vs_one 模式，则抛出异常
        if self.n_classes_ > 2 and self.multi_class == "one_vs_one":
            raise ValueError(
                "one_vs_one multi-class mode does not support "
                "predicting probability estimates. Use "
                "one_vs_rest mode instead."
            )

        # 根据核函数属性，确定是否需要对输入数据 X 进行验证和转换
        if self.kernel is None or self.kernel.requires_vector_input:
            X = self._validate_data(X, ensure_2d=True, dtype="numeric", reset=False)
        else:
            X = self._validate_data(X, ensure_2d=False, dtype=None, reset=False)

        # 调用基础估计器的 predict_proba 方法返回概率估计
        return self.base_estimator_.predict_proba(X)

    @property
    def kernel_(self):
        """
        Return the kernel of the base estimator.
        """
        # 如果是二元分类，则返回基础估计器的核函数属性
        if self.n_classes_ == 2:
            return self.base_estimator_.kernel_
        else:
            # 否则返回由基础估计器的所有子估计器的核函数组成的复合核函数
            return CompoundKernel(
                [estimator.kernel_ for estimator in self.base_estimator_.estimators_]
            )

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        # 这里省略 log_marginal_likelihood 方法的注释，根据规定不需要总结代码整个含义
```