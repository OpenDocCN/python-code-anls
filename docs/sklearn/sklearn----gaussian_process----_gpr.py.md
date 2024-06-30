# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\_gpr.py`

```
# 导入警告模块，用于处理可能的警告信息
import warnings
# 导入整数和实数类型，用于参数验证
from numbers import Integral, Real
# 导入操作符模块中的itemgetter函数，用于数据操作
from operator import itemgetter

# 导入科学计算库NumPy
import numpy as np
# 导入SciPy中的优化函数
import scipy.optimize
# 导入SciPy中的线性代数函数
from scipy.linalg import cho_solve, cholesky, solve_triangular

# 导入基本估计器类、多输出混合类、回归器混合类、拟合上下文和克隆函数
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
# 导入数据预处理模块中的函数，处理比例尺中的零值
from ..preprocessing._data import _handle_zeros_in_scale
# 导入工具函数，检查随机状态
from ..utils import check_random_state
# 导入参数验证模块中的Interval类和StrOptions类
from ..utils._param_validation import Interval, StrOptions
# 导入优化模块中的函数，检查优化结果
from ..utils.optimize import _check_optimize_result
# 导入内置的核函数RBF和通用核函数C
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C

# 定义常量，指示Cholesky分解是否采用下三角矩阵形式
GPR_CHOLESKY_LOWER = True

# 定义高斯过程回归器类，继承自多输出混合类、回归器混合类和基本估计器类
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """高斯过程回归 (GPR)。

    实现基于[RW2006]_的算法2.1。

    除了标准的scikit-learn估计器API外，
    :class:`GaussianProcessRegressor`：

       * 允许在没有先验拟合的情况下进行预测（基于GP先验）
       * 提供一个额外的方法 `sample_y(X)`，用于在给定输入处评估从GPR（先验或后验）中抽取的样本
       * 暴露一个方法 `log_marginal_likelihood(theta)`，可用于外部选择超参数的其他方式，例如通过马尔可夫链蒙特卡罗。

    要了解点估计方法与更贝叶斯建模方法之间的区别，请参阅名为
    :ref:`sphx_glr_auto_examples_gaussian_process_plot_compare_gpr_krr.py`的示例。

    在 :ref:`User Guide <gaussian_process>` 中阅读更多。

    .. versionadded:: 0.18

    参数
    ----------
    kernel : 核实例，默认为None
        指定GP的协方差函数的核。如果传入None，则使用默认的核
        ``ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")``。注意，
        在拟合期间将优化核的超参数，除非边界标记为“fixed”。

    alpha : 浮点数或形状为(n_samples,)的ndarray，默认为1e-10
        在拟合期间添加到核矩阵对角线上的值。这可以防止拟合过程中可能的数值问题，
        确保计算出的值形成一个正定矩阵。也可以解释为训练观察值上的额外高斯测量噪声的方差。
        请注意，这与使用`WhiteKernel`是不同的。如果传入数组，则必须与用于拟合的数据具有相同数量的条目，
        并且用作数据点相关噪声水平。直接指定噪声水平作为参数主要是为了方便和与:class:`~sklearn.linear_model.Ridge`的一致性。
    optimizer : "fmin_l_bfgs_b", callable or None, default="fmin_l_bfgs_b"
        # 优化器类型，可以是字符串指定的内部支持的优化器，也可以是可调用的外部定义的优化器。
        # 如果传入可调用对象，则必须具有以下签名：
        # def optimizer(obj_func, initial_theta, bounds):
        # * 'obj_func': 要最小化的目标函数，接受超参数 theta 作为参数，还可以接受一个可选的 eval_gradient 标志，
        #   用于确定是否额外返回梯度。
        # * 'initial_theta': theta 的初始值，可供局部优化器使用。
        # * 'bounds': theta 值的边界。
        # 返回找到的最佳超参数 theta 和相应的目标函数值。
        # 默认情况下，使用 `scipy.optimize.minimize` 中的 L-BFGS-B 算法。
        # 如果传入 None，则保持核的参数不变。
        # 可用的内部优化器包括：`{'fmin_l_bfgs_b'}`。

    n_restarts_optimizer : int, default=0
        # 优化器重新启动的次数，用于找到最大化对数边际似然的核参数。
        # 优化器的第一次运行从核的初始参数开始，如果有剩余的运行，则从从允许的 theta 值空间中对数均匀采样的 thetas 开始。
        # 如果大于 0，则所有边界必须是有限的。
        # 注意，`n_restarts_optimizer == 0` 意味着只运行一次。

    normalize_y : bool, default=False
        # 是否对目标值 `y` 进行归一化，通过去除均值并缩放为单位方差。
        # 推荐在使用零均值、单位方差先验的情况下使用此选项。
        # 在此实现中，在报告 GP 预测之前会反转归一化操作。

        .. versionchanged:: 0.23
            # 从版本 0.23 开始更改

    copy_X_train : bool, default=True
        # 如果为 True，则在对象中存储训练数据的持久副本。
        # 否则，只存储对训练数据的引用，如果外部修改数据可能会导致预测结果变化。

    n_targets : int, default=None
        # 目标值的维度数量。用于从先验分布中采样时（即在调用 :meth:`sample_y` 之前），决定输出的数量。
        # 一旦调用了 :meth:`fit`，此参数将被忽略。

        .. versionadded:: 1.3
            # 添加于版本 1.3
    # random_state : int, RandomState instance or None, default=None
    # 初始化聚类中心时使用的随机数生成器。可以传入一个整数以确保多次函数调用时生成相同的结果。参见“术语表”中的“随机状态”。
    random_state : int, RandomState instance or None, default=None

    # Attributes
    # ----------
    # X_train_ : array-like of shape (n_samples, n_features) or list of object
    # 训练数据的特征向量或其他表示形式（预测时也需要）。
    X_train_ : array-like of shape (n_samples, n_features) or list of object

    # y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
    # 训练数据中的目标值（预测时也需要）。
    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)

    # kernel_ : kernel instance
    # 用于预测的核函数。核函数的结构与传递的参数相同，但具有经过优化的超参数。
    kernel_ : kernel instance

    # L_ : array-like of shape (n_samples, n_samples)
    # ``X_train_`` 中核函数的下三角 Cholesky 分解。
    L_ : array-like of shape (n_samples, n_samples)

    # alpha_ : array-like of shape (n_samples,)
    # 核空间中训练数据点的对偶系数。
    alpha_ : array-like of shape (n_samples,)

    # log_marginal_likelihood_value_ : float
    # ``self.kernel_.theta`` 的对数边缘似然值。
    log_marginal_likelihood_value_ : float

    # n_features_in_ : int
    # 在“拟合”期间看到的特征数。
    # .. versionadded:: 0.24
    n_features_in_ : int

    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    # 在“拟合”期间看到的特征名称。仅当 `X` 中的特征名称全部为字符串时定义。
    # .. versionadded:: 1.0
    feature_names_in_ : ndarray of shape (`n_features_in_`,)

    # See Also
    # --------
    # GaussianProcessClassifier : 基于拉普拉斯近似的高斯过程分类（GPC）。
    # 参考文献
    # ----------
    # .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
    #    "Gaussian Processes for Machine Learning",
    #    MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.

    # Examples
    # --------
    # >>> from sklearn.datasets import make_friedman2
    # >>> from sklearn.gaussian_process import GaussianProcessRegressor
    # >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    # >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    # >>> kernel = DotProduct() + WhiteKernel()
    # >>> gpr = GaussianProcessRegressor(kernel=kernel,
    # ...         random_state=0).fit(X, y)
    # >>> gpr.score(X, y)
    # 0.3680...
    # >>> gpr.predict(X[:2,:], return_std=True)
    # (array([653.0..., 592.1...]), array([316.6..., 316.6...]))
    """

    # _parameter_constraints: dict = {
    #     "kernel": [None, Kernel],
    #     "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
    #     "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
    #     "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
    #     "normalize_y": ["boolean"],
    #     "copy_X_train": ["boolean"],
    #     "n_targets": [Interval(Integral, 1, None, closed="left"), None],
    #     "random_state": ["random_state"],
    # }
    _parameter_constraints: dict = {
        "kernel": [None, Kernel],
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "normalize_y": ["boolean"],
        "copy_X_train": ["boolean"],
        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }
    # 初始化方法，设置高斯过程的参数和配置
    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,  # 正则化参数，默认值为极小值
        optimizer="fmin_l_bfgs_b",  # 优化器，默认使用L-BFGS-B方法
        n_restarts_optimizer=0,  # 优化器重新启动次数，默认为0
        normalize_y=False,  # 是否对目标值进行归一化，默认不归一化
        copy_X_train=True,  # 是否复制训练集数据，默认复制
        n_targets=None,  # 目标数量，默认为None
        random_state=None,  # 随机种子，默认为None
    ):
        self.kernel = kernel  # 设置内核函数
        self.alpha = alpha  # 设置正则化参数
        self.optimizer = optimizer  # 设置优化器
        self.n_restarts_optimizer = n_restarts_optimizer  # 设置优化器重新启动次数
        self.normalize_y = normalize_y  # 设置是否归一化目标值
        self.copy_X_train = copy_X_train  # 设置是否复制训练集数据
        self.n_targets = n_targets  # 设置目标数量
        self.random_state = random_state  # 设置随机种子

    # 用于在高斯过程中从多维正态分布中抽取样本，并在给定的查询点处进行评估
    @_fit_context(prefer_skip_nested_validation=True)
    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)  # 确定随机数生成器

        y_mean, y_cov = self.predict(X, return_cov=True)  # 获取预测均值和协方差矩阵
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T  # 抽取多维正态分布的样本
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)  # 按指定轴堆叠数组
        return y_samples  # 返回抽样结果数组

    # 计算对数边缘似然函数
    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        # 未提供具体的实现，仅声明方法，后续需要实现具体逻辑

    # 高斯过程的参数优化函数，根据选择的优化器进行不同的优化操作
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":  # 如果选择的优化器是L-BFGS-B
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )  # 使用L-BFGS-B方法进行优化
            _check_optimize_result("lbfgs", opt_res)  # 检查优化结果
            theta_opt, func_min = opt_res.x, opt_res.fun  # 获取优化后的参数和最小值
        elif callable(self.optimizer):  # 如果优化器是可调用的函数
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)  # 调用优化器函数
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")  # 抛出未知优化器异常

        return theta_opt, func_min  # 返回优化后的参数和最小值

    # 返回更多的标签信息，表明此方法不需要拟合过程
    def _more_tags(self):
        return {"requires_fit": False}  # 返回不需要拟合的标签信息
```