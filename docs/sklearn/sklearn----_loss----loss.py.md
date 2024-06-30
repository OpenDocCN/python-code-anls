# `D:\src\scipysrc\scikit-learn\sklearn\_loss\loss.py`

```
# 该模块包含适用于拟合的损失类。

# 它不是公共 API 的一部分。
# 具体损失用于回归、二分类或多分类。
"""

# Goals:
# - 提供损失函数/类的通用私有模块。
# - 用于以下模型：
#   - LogisticRegression
#   - PoissonRegressor, GammaRegressor, TweedieRegressor
#   - HistGradientBoostingRegressor, HistGradientBoostingClassifier
#   - GradientBoostingRegressor, GradientBoostingClassifier
#   - SGDRegressor, SGDClassifier
# - 替代 GLMs 的链接模块。

import numbers  # 导入数字模块，用于处理数字相关的操作

import numpy as np  # 导入 NumPy 库
from scipy.special import xlogy  # 从 SciPy 库中导入 xlogy 函数

from ..utils import check_scalar  # 从上层目录的 utils 模块导入 check_scalar 函数
from ..utils.stats import _weighted_percentile  # 从上层目录的 utils.stats 模块导入 _weighted_percentile 函数
from ._loss import (  # 从当前目录的 _loss 模块导入以下损失类
    CyAbsoluteError,
    CyExponentialLoss,
    CyHalfBinomialLoss,
    CyHalfGammaLoss,
    CyHalfMultinomialLoss,
    CyHalfPoissonLoss,
    CyHalfSquaredError,
    CyHalfTweedieLoss,
    CyHalfTweedieLossIdentity,
    CyHuberLoss,
    CyPinballLoss,
)
from .link import (  # 从当前目录的 link 模块导入以下链接类
    HalfLogitLink,
    IdentityLink,
    Interval,
    LogitLink,
    LogLink,
    MultinomialLogit,
)

# Note: 对于多类别分类，raw_prediction 的形状如下
# - GradientBoostingClassifier: (n_samples, n_classes)
# - HistGradientBoostingClassifier: (n_classes, n_samples)
#
# Note: 我们使用组合而非继承来改进可维护性，避免了上述 Cython 边缘情况，并使代码更易于理解（哪个方法调用哪段代码）。

class BaseLoss:
    """一维目标损失函数的基类。

    约定：

        - y_true.shape = sample_weight.shape = (n_samples,)
        - y_pred.shape = raw_prediction.shape = (n_samples,)
        - 如果 is_multiclass 为真（多类别分类），则
          y_pred.shape = raw_prediction.shape = (n_samples, n_classes)
          注意这对应于 decision_function 的返回值。

    y_true、y_pred、sample_weight 和 raw_prediction 必须全部为 float64 或全部为 float32。
    gradient 和 hessian 必须全部为 float64 或全部为 float32。

    注意 y_pred = link.inverse(raw_prediction)。

    特定的损失类可以继承特定的链接类以满足 BaseLink 的抽象方法。

    Parameters
    ----------
    sample_weight : {None, ndarray}
        如果 sample_weight 为 None，则 hessian 可能是常数。
    n_classes : {None, int}
        分类的类别数，否则为 None。

    Attributes
    ----------
    closs: CyLossFunction
    # link : BaseLink
    # interval_y_true : Interval
    #     y_true 的有效区间
    # interval_y_pred : Interval
    #     y_pred 的有效区间
    # differentiable : bool
    #     表示损失函数在 raw_prediction 的所有位置是否可导
    # need_update_leaves_values : bool
    #     表示在梯度提升决策树中，是否需要在拟合（负）梯度后更新叶节点的值
    # approx_hessian : bool
    #     表示是否使用近似的 Hessian 矩阵，若使用近似的，则应大于或等于精确值
    # constant_hessian : bool
    #     表示该损失函数的 Hessian 矩阵是否恒为一
    # is_multiclass : bool
    #     表示是否允许 n_classes > 2，即是否为多分类问题

    # For gradient boosted decision trees:
    # This variable indicates whether the loss requires the leaves values to
    # be updated once the tree has been trained. The trees are trained to
    # predict a Newton-Raphson step (see grower._finalize_leaf()). But for
    # some losses (e.g. least absolute deviation) we need to adjust the tree
    # values to account for the "line search" of the gradient descent
    # procedure. See the original paper Greedy Function Approximation: A
    # Gradient Boosting Machine by Friedman
    # (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) for the theory.
    differentiable = True
    need_update_leaves_values = False
    is_multiclass = False

    def __init__(self, closs, link, n_classes=None):
        self.closs = closs
        self.link = link
        self.approx_hessian = False
        self.constant_hessian = False
        self.n_classes = n_classes
        self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        self.interval_y_pred = self.link.interval_y_pred

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y)

    def in_y_pred_range(self, y):
        """Return True if y is in the valid range of y_pred.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_pred.includes(y)

    def loss(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        n_threads=1,
    ):
        """计算每个输入点的逐点损失值。

        Parameters
        ----------
        y_true : 形状为 (n_samples,) 的 C 连续数组
            观察到的真实目标值。
        raw_prediction : 形状为 (n_samples,) 或 (n_samples, n_classes) 的 C 连续数组
            原始预测值（在链接空间中）。
        sample_weight : None 或 形状为 (n_samples,) 的 C 连续数组
            样本权重。
        loss_out : None 或 形状为 (n_samples,) 的 C 连续数组
            结果存放位置。如果为 None，则可能会创建一个新数组。
        n_threads : int，默认为 1
            可能使用的 OpenMP 线程并行度。

        Returns
        -------
        loss : 形状为 (n_samples,) 的数组
            逐元素损失函数值。
        """
        if loss_out is None:
            loss_out = np.empty_like(y_true)
        # 对形状为 (n_samples, 1) 的情况进行处理，将其转换为形状 (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)

        self.closs.loss(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            n_threads=n_threads,
        )
        return loss_out

    def loss_gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        loss_out=None,
        gradient_out=None,
        n_threads=1,
    ):
        """
        Compute loss and gradient w.r.t. raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        loss_out : None or C-contiguous array of shape (n_samples,)
            A location into which the loss is stored. If None, a new array
            might be created.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : array of shape (n_samples,)
            Element-wise loss function.

        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        # If loss_out is not provided, initialize it based on gradient_out or y_true
        if loss_out is None:
            if gradient_out is None:
                loss_out = np.empty_like(y_true)
                gradient_out = np.empty_like(raw_prediction)
            else:
                loss_out = np.empty_like(y_true, dtype=gradient_out.dtype)
        # If gradient_out is not provided, initialize it based on loss_out or raw_prediction
        elif gradient_out is None:
            gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)

        # Ensure raw_prediction and gradient_out are shaped correctly
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        # Call the loss and gradient computation method from self.closs
        self.closs.loss_gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        # Return computed loss and gradient
        return loss_out, gradient_out

    def gradient(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        n_threads=1,
    ):
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the result is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.
        """
        # 如果 gradient_out 为 None，则创建一个与 raw_prediction 形状相同的空数组
        if gradient_out is None:
            gradient_out = np.empty_like(raw_prediction)

        # 如果 raw_prediction 的维度为 2 并且第二维的大小为 1，则将其压缩为一维数组
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        # 如果 gradient_out 的维度为 2 并且第二维的大小为 1，则将其压缩为一维数组
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        # 调用 self.closs 对象的 gradient 方法计算梯度
        self.closs.gradient(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        # 返回计算后的梯度数组 gradient_out
        return gradient_out

    # 定义一个方法 gradient_hessian，计算 loss 对 raw_prediction 的梯度和 Hessian 矩阵
    def gradient_hessian(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        hessian_out=None,
        n_threads=1,
    ):
        """
        Compute gradient and hessian of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        hessian_out : None or C-contiguous array of shape (n_samples,) or array \
            of shape (n_samples, n_classes)
            A location into which the hessian is stored. If None, a new array
            might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise gradients.

        hessian : arrays of shape (n_samples,) or (n_samples, n_classes)
            Element-wise hessians.
        """
        if gradient_out is None:
            # 如果 gradient_out 为 None，则根据 hessian_out 是否为 None 决定创建新的数组
            if hessian_out is None:
                gradient_out = np.empty_like(raw_prediction)
                hessian_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(hessian_out)
        elif hessian_out is None:
            # 如果 hessian_out 为 None，则创建与 gradient_out 形状相同的新数组
            hessian_out = np.empty_like(gradient_out)

        # 处理 raw_prediction, gradient_out 和 hessian_out 形状为 (n_samples, 1) 的情况，将其压缩为一维数组 (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)
        if hessian_out.ndim == 2 and hessian_out.shape[1] == 1:
            hessian_out = hessian_out.squeeze(1)

        # 调用 self.closs 对象的 gradient_hessian 方法，计算梯度和海森矩阵
        self.closs.gradient_hessian(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            hessian_out=hessian_out,
            n_threads=n_threads,
        )
        # 返回计算得到的 gradient_out 和 hessian_out
        return gradient_out, hessian_out
    def __call__(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        """Compute the weighted average loss.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        loss : float
            Mean or averaged loss function.
        """
        return np.average(
            self.loss(
                y_true=y_true,
                raw_prediction=raw_prediction,
                sample_weight=None,  # 使用给定的样本权重（默认为None）
                loss_out=None,  # 输出的损失结果（默认为None）
                n_threads=n_threads,  # 线程数，可能使用OpenMP线程并行化（默认为1）
            ),
            weights=sample_weight,  # 根据样本权重进行加权平均
        )

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This can be used as initial estimates of predictions, i.e. before the
        first iteration in fit.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or array of shape (n_samples,)
            Sample weights.

        Returns
        -------
        raw_prediction : numpy scalar or array of shape (n_classes,)
            Raw predictions of an intercept-only model.
        """
        # As default, take weighted average of the target over the samples
        # axis=0 and then transform into link-scale (raw_prediction).
        y_pred = np.average(y_true, weights=sample_weight, axis=0)  # 对目标值进行加权平均
        eps = 10 * np.finfo(y_pred.dtype).eps  # 获取目标值数据类型的机器精度

        if self.interval_y_pred.low == -np.inf:
            a_min = None  # 如果下限为负无穷，则设为None
        elif self.interval_y_pred.low_inclusive:
            a_min = self.interval_y_pred.low  # 如果下限包含，则设为具体值
        else:
            a_min = self.interval_y_pred.low + eps  # 否则，在下限上加上一个小的机器精度

        if self.interval_y_pred.high == np.inf:
            a_max = None  # 如果上限为正无穷，则设为None
        elif self.interval_y_pred.high_inclusive:
            a_max = self.interval_y_pred.high  # 如果上限包含，则设为具体值
        else:
            a_max = self.interval_y_pred.high - eps  # 否则，在上限上减去一个小的机器精度

        if a_min is None and a_max is None:
            return self.link.link(y_pred)  # 如果没有下限和上限，则直接应用链接函数得到结果
        else:
            return self.link.link(np.clip(y_pred, a_min, a_max))  # 否则，对预测值进行裁剪后再应用链接函数

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        """Calculate term dropped in loss.

        With this term added, the loss of perfect predictions is zero.
        """
        return np.zeros_like(y_true)  # 返回与y_true相同形状的全零数组作为损失中被省略的项
    def init_gradient_and_hessian(self, n_samples, dtype=np.float64, order="F"):
        """Initialize arrays for gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined values.

        Parameters
        ----------
        n_samples : int
            The number of samples, usually passed to `fit()`.
        dtype : {np.float64, np.float32}, default=np.float64
            The dtype of the arrays gradient and hessian.
        order : {'C', 'F'}, default='F'
            Order of the arrays gradient and hessian. The default 'F' makes the arrays
            contiguous along samples.

        Returns
        -------
        gradient : C-contiguous array of shape (n_samples,) or array of shape \
            (n_samples, n_classes)
            Empty array (allocated but not initialized) to be used as argument
            gradient_out.
        hessian : C-contiguous array of shape (n_samples,), array of shape
            (n_samples, n_classes) or shape (1,)
            Empty (allocated but not initialized) array to be used as argument
            hessian_out.
            If constant_hessian is True (e.g. `HalfSquaredError`), the array is
            initialized to ``1``.
        """
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                "Valid options for 'dtype' are np.float32 and np.float64. "
                f"Got dtype={dtype} instead."
            )

        if self.is_multiclass:
            # Determine the shape of the gradient array based on multiclass or binary classification
            shape = (n_samples, self.n_classes)
        else:
            # For binary classification, shape is simply (n_samples,)
            shape = (n_samples,)
        # Create an uninitialized array for gradients
        gradient = np.empty(shape=shape, dtype=dtype, order=order)

        if self.constant_hessian:
            # If hessians are constant across samples (e.g., for certain loss functions),
            # initialize hessian array with ones
            hessian = np.ones(shape=(1,), dtype=dtype)
        else:
            # Create an uninitialized array for hessians
            hessian = np.empty(shape=shape, dtype=dtype, order=order)

        return gradient, hessian
# 注意：自然情况下，应按以下顺序继承：
#         class HalfSquaredError(IdentityLink, CyHalfSquaredError, BaseLoss)
#       但由于 https://github.com/cython/cython/issues/4350 的问题，
#       我们将 BaseLoss 设置为最后一个。这当然改变了方法解析顺序（MRO）。

class HalfSquaredError(BaseLoss):
    """Half squared error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half squared error is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_prediction_i)**2

    The factor of 0.5 simplifies the computation of gradients and results in a
    unit hessian (and is consistent with what is done in LightGBM). It is also
    half the Normal distribution deviance.
    """

    def __init__(self, sample_weight=None):
        # 调用父类构造函数，使用 CyHalfSquaredError 作为损失函数，IdentityLink 作为链接函数
        super().__init__(closs=CyHalfSquaredError(), link=IdentityLink())
        # 若样本权重未提供，则设置常数 Hessian 为 True
        self.constant_hessian = sample_weight is None


class AbsoluteError(BaseLoss):
    """Absolute error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the absolute error is defined as::

        loss(x_i) = |y_true_i - raw_prediction_i|

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.
    """

    # 损失函数几乎处处不可导（只在一个点可导），因此设为不可导
    differentiable = False
    # 需要更新叶子节点的值
    need_update_leaves_values = True

    def __init__(self, sample_weight=None):
        # 调用父类构造函数，使用 CyAbsoluteError 作为损失函数，IdentityLink 作为链接函数
        super().__init__(closs=CyAbsoluteError(), link=IdentityLink())
        # 使用近似 Hessian
        self.approx_hessian = True
        # 若样本权重未提供，则设置常数 Hessian 为 True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            # 返回目标的加权中位数，即沿样本轴（axis=0）计算
            return np.median(y_true, axis=0)
        else:
            # 使用加权百分位数计算目标的预测值
            return _weighted_percentile(y_true, sample_weight, 50)


class PinballLoss(BaseLoss):
    """Quantile loss aka pinball loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the pinball loss is defined as::

        loss(x_i) = rho_{quantile}(y_true_i - raw_prediction_i)

        rho_{quantile}(u) = u * (quantile - 1_{u<0})
                          = -u *(1 - quantile)  if u < 0
                             u * quantile       if u >= 0

    Note: 2 * PinballLoss(quantile=0.5) equals AbsoluteError().

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.

    Additional Attributes
    ---------------------
    """
    # 损失函数几乎处处不可导（只在一个点可导），因此设为不可导
    differentiable = False
    # 需要更新叶子节点的值
    need_update_leaves_values = True

    def __init__(self, sample_weight=None):
        # 调用父类构造函数，使用 CyPinballLoss 作为损失函数，IdentityLink 作为链接函数
        super().__init__(closs=CyPinballLoss(), link=IdentityLink())
        # 使用近似 Hessian
        self.approx_hessian = True
        # 若样本权重未提供，则设置常数 Hessian 为 True
        self.constant_hessian = sample_weight is None
    quantile : float
        要估计的分位数水平。必须在 (0, 1) 的范围内。
    """

    # 初始化两个属性
    differentiable = False  # 不可微
    need_update_leaves_values = True  # 需要更新叶子节点的值

    def __init__(self, sample_weight=None, quantile=0.5):
        # 检查 quantile 参数是否为实数，并在指定范围内
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        # 调用父类的初始化方法
        super().__init__(
            closs=CyPinballLoss(quantile=float(quantile)),  # 设置损失函数的分位数
            link=IdentityLink(),  # 设置链接函数为恒等映射
        )
        self.approx_hessian = True  # 启用近似的 Hessian 矩阵
        self.constant_hessian = sample_weight is None  # 根据 sample_weight 是否为 None 设置常数 Hessian 属性

    def fit_intercept_only(self, y_true, sample_weight=None):
        """计算只有截距的模型的原始预测值。

        这是目标的加权中位数，即在样本轴上计算。
        """
        if sample_weight is None:
            return np.percentile(y_true, 100 * self.closs.quantile, axis=0)
        else:
            return _weighted_percentile(
                y_true, sample_weight, 100 * self.closs.quantile
            )
class HuberLoss(BaseLoss):
    """Huber loss, for regression.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the Huber loss is defined as::

        loss(x_i) = 1/2 * abserr**2            if abserr <= delta
                    delta * (abserr - delta/2) if abserr > delta

        abserr = |y_true_i - raw_prediction_i|
        delta = quantile(abserr, self.quantile)

    Note: HuberLoss(quantile=1) equals HalfSquaredError and HuberLoss(quantile=0)
    equals delta * (AbsoluteError() - delta/2).

    Additional Attributes
    ---------------------
    quantile : float
        The quantile level which defines the breaking point `delta` to distinguish
        between absolute error and squared error. Must be in range (0, 1).

     Reference
    ---------
    .. [1] Friedman, J.H. (2001). :doi:`Greedy function approximation: A gradient
      boosting machine <10.1214/aos/1013203451>`.
      Annals of Statistics, 29, 1189-1232.
    """

    differentiable = False  # 标记此损失函数不可导
    need_update_leaves_values = True  # 标记此损失函数需要更新叶子节点的值

    def __init__(self, sample_weight=None, quantile=0.9, delta=0.5):
        check_scalar(
            quantile,
            "quantile",
            target_type=numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="neither",
        )
        self.quantile = quantile  # 存储分位数值，用于定义 Huber 损失的分界点
        super().__init__(
            closs=CyHuberLoss(delta=float(delta)),  # 初始化 CyHuberLoss 类型的损失函数
            link=IdentityLink(),  # 使用恒等连接函数
        )
        self.approx_hessian = True  # 允许近似的 Hessian 矩阵
        self.constant_hessian = False  # 非常数 Hessian 矩阵

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        # See formula before algo 4 in Friedman (2001), but we apply it to y_true,
        # not to the residual y_true - raw_prediction. An estimator like
        # HistGradientBoostingRegressor might then call it on the residual, e.g.
        # fit_intercept_only(y_true - raw_prediction).
        if sample_weight is None:
            median = np.percentile(y_true, 50, axis=0)  # 计算 y_true 的加权中位数
        else:
            median = _weighted_percentile(y_true, sample_weight, 50)  # 计算加权百分位数
        diff = y_true - median  # 计算 y_true 与中位数的差异
        term = np.sign(diff) * np.minimum(self.closs.delta, np.abs(diff))  # 根据 Huber 损失计算术语
        return median + np.average(term, weights=sample_weight)  # 返回预测结果


class HalfPoissonLoss(BaseLoss):
    """Half Poisson deviance loss with log-link, for regression.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half the Poisson deviance is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_prediction_i))
                    - y_true_i + exp(raw_prediction_i)
    """
    # 这段代码片段定义了一个特定的损失函数类，用于半泊松损失函数的计算。
    # 半泊松损失函数是泊松偏差的一半，实际上是负对数似然，不包括与原始预测无关的常数项，简化了梯度计算。
    # 该函数还跳过了常数项 `y_true_i * log(y_true_i) - y_true_i` 的计算。

    def __init__(self, sample_weight=None):
        # 调用父类构造函数，初始化损失函数为半泊松损失函数(closs=CyHalfPoissonLoss())，链接函数为对数链接(LogLink())
        super().__init__(closs=CyHalfPoissonLoss(), link=LogLink())
        # 创建一个区间对象，表示目标变量的取值范围为 [0, ∞)
        self.interval_y_true = Interval(0, np.inf, True, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # 计算常数项 `y_true * log(y_true) - y_true`，称为 term
        term = xlogy(y_true, y_true) - y_true
        # 如果提供了样本权重，则将 term 乘以样本权重
        if sample_weight is not None:
            term *= sample_weight
        # 返回计算得到的 term
        return term
class HalfGammaLoss(BaseLoss):
    """Half Gamma deviance loss with log-link, for regression.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Gamma deviance loss is defined as::

        loss(x_i) = log(exp(raw_prediction_i)/y_true_i)
                    + y_true/exp(raw_prediction_i) - 1

    Half the Gamma deviance is actually proportional to the negative log-
    likelihood up to constant terms (not involving raw_prediction) and
    simplifies the computation of the gradients.
    We also skip the constant term `-log(y_true_i) - 1`.
    """

    def __init__(self, sample_weight=None):
        # 调用父类构造函数，初始化使用 CyHalfGammaLoss 和 LogLink
        super().__init__(closs=CyHalfGammaLoss(), link=LogLink())
        # 设置 y_true 的取值范围区间
        self.interval_y_true = Interval(0, np.inf, False, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # 计算常数项 `-log(y_true_i) - 1`
        term = -np.log(y_true) - 1
        if sample_weight is not None:
            term *= sample_weight
        return term


class HalfTweedieLoss(BaseLoss):
    """Half Tweedie deviance loss with log-link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers
    power in real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * exp(raw_prediction_i)**(1-p) / (1-p)
                    + exp(raw_prediction_i)**(2-p) / (2-p)

    Taking the limits for p=0, 1, 2 gives HalfSquaredError with a log link,
    HalfPoissonLoss and HalfGammaLoss.

    We also skip constant terms, but those are different for p=0, 1, 2.
    Therefore, the loss is not continuous in `power`.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5):
        # 调用父类构造函数，初始化使用 CyHalfTweedieLoss 和 LogLink，设置 power 参数
        super().__init__(
            closs=CyHalfTweedieLoss(power=float(power)),
            link=LogLink(),
        )
        # 根据 power 参数设置 y_true 的取值范围区间
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)
    # 如果损失函数的幂为 0，则调用 HalfSquaredError 类的 constant_to_optimal_zero 方法
    if self.closs.power == 0:
        return HalfSquaredError().constant_to_optimal_zero(
            y_true=y_true, sample_weight=sample_weight
        )
    # 如果损失函数的幂为 1，则调用 HalfPoissonLoss 类的 constant_to_optimal_zero 方法
    elif self.closs.power == 1:
        return HalfPoissonLoss().constant_to_optimal_zero(
            y_true=y_true, sample_weight=sample_weight
        )
    # 如果损失函数的幂为 2，则调用 HalfGammaLoss 类的 constant_to_optimal_zero 方法
    elif self.closs.power == 2:
        return HalfGammaLoss().constant_to_optimal_zero(
            y_true=y_true, sample_weight=sample_weight
        )
    else:
        # 否则，计算一个项 term，用于损失函数的优化
        p = self.closs.power
        term = np.power(np.maximum(y_true, 0), 2 - p) / (1 - p) / (2 - p)
        # 如果给定了样本权重，则将 term 乘以样本权重
        if sample_weight is not None:
            term *= sample_weight
        # 返回计算出的 term 作为结果
        return term
class HalfTweedieLossIdentity(BaseLoss):
    """Half Tweedie deviance loss with identity link, for regression.

    Domain:
    y_true in real numbers for power <= 0
    y_true in non-negative real numbers for 0 < power < 2
    y_true in positive real numbers for 2 <= power
    y_pred in positive real numbers for power != 0
    y_pred in real numbers for power = 0
    power in real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, half Tweedie deviance loss with p=power is defined
    as::

        loss(x_i) = max(y_true_i, 0)**(2-p) / (1-p) / (2-p)
                    - y_true_i * raw_prediction_i**(1-p) / (1-p)
                    + raw_prediction_i**(2-p) / (2-p)

    Note that the minimum value of this loss is 0.

    Note furthermore that although no Tweedie distribution exists for
    0 < power < 1, it still gives a strictly consistent scoring function for
    the expectation.
    """

    def __init__(self, sample_weight=None, power=1.5):
        super().__init__(
            closs=CyHalfTweedieLossIdentity(power=float(power)),
            link=IdentityLink(),
        )
        # Define the interval for y_true based on the value of power
        if self.closs.power <= 0:
            self.interval_y_true = Interval(-np.inf, np.inf, False, False)
        elif self.closs.power < 2:
            self.interval_y_true = Interval(0, np.inf, True, False)
        else:
            self.interval_y_true = Interval(0, np.inf, False, False)

        # Define the interval for y_pred based on the value of power
        if self.closs.power == 0:
            self.interval_y_pred = Interval(-np.inf, np.inf, False, False)
        else:
            self.interval_y_pred = Interval(0, np.inf, False, False)


class HalfBinomialLoss(BaseLoss):
    """Half Binomial deviance loss with logit link, for binary classification.

    This is also know as binary cross entropy, log-loss and logistic loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)

    For a given sample x_i, half Binomial deviance is defined as the negative
    log-likelihood of the Binomial/Bernoulli distribution and can be expressed
    as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).

    Note that the formulation works for classification, y = {0, 1}, as well as
    logistic regression, y = [0, 1].
    If you add `constant_to_optimal_zero` to the loss, you get half the
    Bernoulli/binomial deviance.

    More details: Inserting the predicted probability y_pred = expit(raw_prediction)
    in the loss gives the well known::

        loss(x_i) = - y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i)
    """
    def __init__(self, sample_weight=None):
        # 调用父类的初始化方法，并设置相关参数
        super().__init__(
            closs=CyHalfBinomialLoss(),  # 使用 CyHalfBinomialLoss 作为损失函数
            link=LogitLink(),  # 使用 LogitLink 作为链接函数
            n_classes=2,  # 设置类别数为 2
        )
        # 创建一个区间对象，表示预测值的有效范围为 [0, 1]
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # 如果 y_true 不是 0 或 1，则这个 term 非零
        term = xlogy(y_true, y_true) + xlogy(1 - y_true, 1 - y_true)
        if sample_weight is not None:
            # 如果有样本权重，则将 term 乘以样本权重
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # 如果 raw_prediction 的维度是 (n_samples, 1)，将其压缩为一维数组 (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        # 创建一个空的概率数组，形状为 (n_samples, 2)，数据类型与 raw_prediction 相同
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        # 将类别 1 的概率计算为 raw_prediction 经链接函数逆变换后的结果
        proba[:, 1] = self.link.inverse(raw_prediction)
        # 将类别 0 的概率计算为 1 减去类别 1 的概率
        proba[:, 0] = 1 - proba[:, 1]
        return proba
class HalfMultinomialLoss(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred has n_classes elements, each element in (0, 1)

    Link:
    y_pred = softmax(raw_prediction)

    Note: We assume y_true to be already label encoded. The inverse link is
    softmax. But the full link function is the symmetric multinomial logit
    function.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the multinomial distribution, it
    generalizes the binary cross-entropy to more than 2 classes::

        loss_i = log(sum(exp(raw_pred_{i, k}), k=0..n_classes-1))
                - sum(y_true_{i, k} * raw_pred_{i, k}, k=0..n_classes-1)

    See [1].

    Note that for the hessian, we calculate only the diagonal part in the
    classes: If the full hessian for classes k and l and sample i is H_i_k_l,
    we calculate H_i_k_k, i.e. k=l.

    Reference
    ---------
    .. [1] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
        "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
        Multinomial Regression".
        <1311.6529>`
    """

    # Indicate that this loss function is for multiclass problems
    is_multiclass = True

    def __init__(self, sample_weight=None, n_classes=3):
        # Initialize the loss function with a Cython implementation, link function,
        # and number of classes
        super().__init__(
            closs=CyHalfMultinomialLoss(),
            link=MultinomialLogit(),
            n_classes=n_classes,
        )
        # Define the interval for valid values of y_true
        self.interval_y_true = Interval(0, np.inf, True, False)
        # Define the interval for valid values of y_pred
        self.interval_y_pred = Interval(0, 1, False, False)

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
            Array of y_true values
        """
        # Check if each value in y is within the defined interval and is integer
        return self.interval_y_true.includes(y) and np.all(y.astype(int) == y)

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.

        Parameters
        ----------
        y_true : ndarray
            True labels
        sample_weight : ndarray or None
            Weights for each sample
        """
        # Initialize an array for the output raw predictions
        out = np.zeros(self.n_classes, dtype=y_true.dtype)
        # Machine epsilon for dtype of y_true
        eps = np.finfo(y_true.dtype).eps
        # Calculate average of y_true values for each class k, weighted by sample_weight
        for k in range(self.n_classes):
            out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)  # Clip values to avoid extreme probabilities
        # Apply the link function to convert to probabilities and reshape
        return self.link.link(out[None, :]).reshape(-1)

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        # Inverse of the link function to get predicted probabilities
        return self.link.inverse(raw_prediction)

    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        proba_out=None,
        n_threads=1,
    ):
        # Method for computing the gradient of the predicted probabilities
        # (Not fully annotated here as it continues beyond this snippet)
    ):
        """
        Compute gradient and class probabilities for raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        # Check if gradient_out is None; if so, initialize gradient_out and proba_out
        if gradient_out is None:
            if proba_out is None:
                gradient_out = np.empty_like(raw_prediction)
                proba_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(proba_out)
        # If gradient_out is not None but proba_out is None, initialize proba_out
        elif proba_out is None:
            proba_out = np.empty_like(gradient_out)

        # Compute gradients and class probabilities using self.closs.gradient_proba method
        self.closs.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            proba_out=proba_out,
            n_threads=n_threads,
        )
        # Return computed gradient and probabilities
        return gradient_out, proba_out
class ExponentialLoss(BaseLoss):
    """Exponential loss with (half) logit link, for binary classification.

    This is also know as boosting loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(2 * raw_prediction)

    For a given sample x_i, the exponential loss is defined as::

        loss(x_i) = y_true_i * exp(-raw_pred_i)) + (1 - y_true_i) * exp(raw_pred_i)

    See:
    - J. Friedman, T. Hastie, R. Tibshirani.
      "Additive logistic regression: a statistical view of boosting (With discussion
      and a rejoinder by the authors)." Ann. Statist. 28 (2) 337 - 407, April 2000.
      https://doi.org/10.1214/aos/1016218223
    - A. Buja, W. Stuetzle, Y. Shen. (2005).
      "Loss Functions for Binary Class Probability Estimation and Classification:
      Structure and Applications."

    Note that the formulation works for classification, y = {0, 1}, as well as
    "exponential logistic" regression, y = [0, 1].
    Note that this is a proper scoring rule, but without it's canonical link.

    More details: Inserting the predicted probability
    y_pred = expit(2 * raw_prediction) in the loss gives::

        loss(x_i) = y_true_i * sqrt((1 - y_pred_i) / y_pred_i)
            + (1 - y_true_i) * sqrt(y_pred_i / (1 - y_pred_i))
    """

    def __init__(self, sample_weight=None):
        super().__init__(
            closs=CyExponentialLoss(),
            link=HalfLogitLink(),
            n_classes=2,
        )
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        # This is non-zero only if y_true is neither 0 nor 1.
        term = -2 * np.sqrt(y_true * (1 - y_true))
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


_LOSSES = {
    "squared_error": HalfSquaredError,  # Loss function for squared error regression.
    "absolute_error": AbsoluteError,  # Loss function for absolute error regression.
    "pinball_loss": PinballLoss,  # Loss function for pinball regression.
    "huber_loss": HuberLoss,  # Loss function for Huber regression.
    "poisson_loss": HalfPoissonLoss,  # Loss function for Poisson regression.
    "gamma_loss": HalfGammaLoss,  # Loss function for Gamma regression.
    "tweedie_loss": HalfTweedieLoss,  # Loss function for Tweedie regression.
    "binomial_loss": HalfBinomialLoss,  # Loss function for binomial (logistic) regression.
    "multinomial_loss": HalfMultinomialLoss,  # Loss function for multinomial (softmax) regression.
}
    "exponential_loss": ExponentialLoss,


# 将字符串"exponential_loss"作为键，关联到 ExponentialLoss 类的引用作为值，添加到字典或映射中使用
}


注释：


# 这行代码关闭了一个代码块，通常用于结束一个函数、循环或条件语句的定义。
```