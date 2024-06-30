# `D:\src\scipysrc\scikit-learn\sklearn\kernel_ridge.py`

```
# 引入Real类，用于验证alpha参数是否为实数类型
from numbers import Real

# 引入NumPy库，用于数学运算和数组操作
import numpy as np

# 从当前目录的base模块中引入BaseEstimator、MultiOutputMixin和RegressorMixin类及_fit_context函数
from .base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context

# 从linear_model._ridge模块中引入_solve_cholesky_kernel函数
from .linear_model._ridge import _solve_cholesky_kernel

# 从metrics.pairwise模块中引入PAIRWISE_KERNEL_FUNCTIONS和pairwise_kernels函数
from .metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels

# 从utils._param_validation模块中引入Interval和StrOptions类
from .utils._param_validation import Interval, StrOptions

# 从utils.validation模块中引入_check_sample_weight和check_is_fitted函数
from .utils.validation import _check_sample_weight, check_is_fitted


class KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Kernel ridge regression.

    Kernel ridge regression (KRR) combines ridge regression (linear least
    squares with l2-norm regularization) with the kernel trick. It thus
    learns a linear function in the space induced by the respective kernel and
    the data. For non-linear kernels, this corresponds to a non-linear
    function in the original space.

    The form of the model learned by KRR is identical to support vector
    regression (SVR). However, different loss functions are used: KRR uses
    squared error loss while support vector regression uses epsilon-insensitive
    loss, both combined with l2 regularization. In contrast to SVR, fitting a
    KRR model can be done in closed-form and is typically faster for
    medium-sized datasets. On the other hand, the learned model is non-sparse
    and thus slower than SVR, which learns a sparse model for epsilon > 0, at
    prediction-time.

    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

    Read more in the :ref:`User Guide <kernel_ridge>`.

    Parameters
    ----------
    alpha : float or array-like of shape (n_targets,), default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number. See :ref:`ridge_regression` for formula.
    # kernel参数指定内部使用的核映射。该参数直接传递给pairwise_kernels类。
    # 如果kernel是字符串，则必须是pairwise.PAIRWISE_KERNEL_FUNCTIONS中的度量之一或者是"precomputed"。
    # 如果kernel是"precomputed"，则假定X是一个核矩阵。
    # 如果kernel是一个可调用函数，则会对每对实例（行）调用该函数，并记录结果值。
    # 可调用函数应该接受X的两行作为输入，并返回相应的核值作为单个数字。
    # 这意味着不允许使用来自sklearn.metrics.pairwise的可调用函数，因为它们操作的是矩阵而不是单个样本。
    # 应该使用标识核的字符串来设置该参数。

    # gamma参数用于RBF、拉普拉斯、多项式、指数卡方和sigmoid核函数。
    # 默认值的解释取决于具体的核函数；请参阅sklearn.metrics.pairwise的文档。
    # 其他核函数会忽略该参数。

    # degree参数用于多项式核函数的阶数。其他核函数会忽略该参数。

    # coef0参数用于多项式和sigmoid核函数的零系数。
    # 其他核函数会忽略该参数。

    # kernel_params参数是传递给核函数的可调用对象的额外参数（关键字参数）。

    # dual_coef_属性：在核空间中表示权重向量或向量的表示。

    # X_fit_属性：训练数据，也是预测所需的数据。
    # 如果kernel == "precomputed"，则这是预先计算的训练矩阵，形状为(n_samples, n_samples)。

    # n_features_in_属性：在拟合过程中观察到的特征数。

    # feature_names_in_属性：在拟合过程中观察到的特征名称数组。
    # 仅在X具有所有字符串类型特征名称时才会定义。

    # 参见：
    # sklearn.gaussian_process.GaussianProcessRegressor：提供自动核超参数调整和预测不确定性的高斯过程回归器。
    # sklearn.linear_model.Ridge：线性岭回归。
    # sklearn.linear_model.RidgeCV：带内置交叉验证的岭回归。
    # sklearn.svm.SVR：支持多种核函数的支持向量回归器。

    # 参考文献：
    # Kevin P. Murphy，《机器学习：概率透视》，麻省理工学院出版社，第14.4.3章，492-493页。
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> krr = KernelRidge(alpha=1.0)
    >>> krr.fit(X, y)
    KernelRidge(alpha=1.0)
    """

    # 定义参数的约束条件字典
    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS.keys()) | {"precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
    }

    # 定义 KernelRidge 类
    def __init__(
        self,
        alpha=1,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
    ):
        # 初始化类的各个参数
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    # 获取核矩阵的方法
    def _get_kernel(self, X, Y=None):
        # 根据 kernel 参数决定是否使用 kernel_params
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        # 调用 pairwise_kernels 函数生成核矩阵
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    # 返回更多标签信息的方法
    def _more_tags(self):
        return {"pairwise": self.kernel == "precomputed"}

    # 使用装饰器定义的 fit 方法
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Kernel Ridge regression model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. If kernel == "precomputed" this is instead
            a precomputed kernel matrix, of shape (n_samples, n_samples).

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 验证和转换输入数据
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        # 如果 sample_weight 不为 None 且不是 float 类型，则进行检查和调整
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = _check_sample_weight(sample_weight, X)

        # 获取核矩阵
        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)

        # 如果 y 的维度是 1，则进行 reshape
        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        # 如果 kernel 参数为 "precomputed"，则设置 copy 标志为 True
        copy = self.kernel == "precomputed"
        # 使用 _solve_cholesky_kernel 解决对偶问题
        self.dual_coef_ = _solve_cholesky_kernel(K, y, alpha, sample_weight, copy)
        # 如果之前进行了 reshape，现在将 dual_coef_ 进行 ravel 操作
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        # 记录输入数据 X
        self.X_fit_ = X

        # 返回自身实例
        return self
    def predict(self, X):
        """Predict using the kernel ridge model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples. If kernel == "precomputed" this is instead a
            precomputed kernel matrix, shape = [n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for this estimator.

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        # 检查模型是否已经拟合，确保模型已经训练过
        check_is_fitted(self)
        # 验证输入数据 X 的格式，并确保接受稀疏矩阵格式，不重置数据
        X = self._validate_data(X, accept_sparse=("csr", "csc"), reset=False)
        # 根据输入数据 X 和模型内保存的训练数据 self.X_fit_ 计算核矩阵 K
        K = self._get_kernel(X, self.X_fit_)
        # 返回预测值，使用核矩阵 K 和模型的 dual_coef_ 进行点乘操作
        return np.dot(K, self.dual_coef_)
```