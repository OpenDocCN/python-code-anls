# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_gb.py`

```
# Gradient Boosted Regression Trees 梯度提升回归树

# 该模块包含了用于分类和回归的梯度提升回归树拟合方法。

# 模块结构如下：

# - BaseGradientBoosting 基类实现了模块中所有估计器的通用拟合方法。回归和分类的区别仅在于具体使用的 LossFunction。

# - GradientBoostingClassifier 实现了分类问题的梯度提升。

# - GradientBoostingRegressor 实现了回归问题的梯度提升。
"""

# 作者：scikit-learn 开发者
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
import math  # 导入数学库
import warnings  # 导入警告处理模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类元类和抽象方法装饰器
from numbers import Integral, Real  # 导入数字类型模块中的整数和实数类型
from time import time  # 导入时间模块中的时间函数

import numpy as np  # 导入NumPy库
from scipy.sparse import csc_matrix, csr_matrix, issparse  # 导入稀疏矩阵相关函数

# 从相关的内部模块导入所需的损失函数类
from .._loss.loss import (
    _LOSSES,
    AbsoluteError,
    ExponentialLoss,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    HuberLoss,
    PinballLoss,
)

# 从scikit-learn基础模块导入分类器和回归器的Mixin类，以及相关辅助函数和异常类
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from ..dummy import DummyClassifier, DummyRegressor  # 导入虚拟分类器和虚拟回归器
from ..exceptions import NotFittedError  # 导入未拟合错误类
from ..model_selection import train_test_split  # 导入数据集分割函数
from ..preprocessing import LabelEncoder  # 导入标签编码器
from ..tree import DecisionTreeRegressor  # 导入决策树回归器
from ..tree._tree import DOUBLE, DTYPE, TREE_LEAF  # 导入决策树相关类型和常量
from ..utils import check_array, check_random_state, column_or_1d  # 导入数据验证和处理函数
from ..utils._param_validation import HasMethods, Interval, StrOptions  # 导入参数验证函数
from ..utils.multiclass import check_classification_targets  # 导入多类别分类目标验证函数
from ..utils.stats import _weighted_percentile  # 导入加权百分位数计算函数
from ..utils.validation import _check_sample_weight, check_is_fitted  # 导入样本权重验证和拟合检查函数
from ._base import BaseEnsemble  # 从基础集成模块导入基础集成类
from ._gradient_boosting import _random_sample_mask, predict_stage, predict_stages  # 导入梯度提升相关函数

# 将 _LOSSES 字典复制一份并添加了两个额外的损失函数："quantile" 和 "huber"
_LOSSES = _LOSSES.copy()
_LOSSES.update(
    {
        "quantile": PinballLoss,
        "huber": HuberLoss,
    }
)


def _safe_divide(numerator, denominator):
    """防止溢出和除零错误。

    这个函数用于分类器中，其中分母可能恰好变为零。
    例如对于对数损失（log loss），HalfBinomialLoss，
    如果概率 proba=0 或者 proba=1，那么分母 = hessian = 0，
    我们应该将线搜索中的节点值设为零，因为损失无法再有所改善。
    为了数值安全，我们已经针对极小的值执行了此操作。
    """
    if abs(denominator) < 1e-150:
        return 0.0
    else:
        # 将分子和分母转换为 Python 浮点数，触发可能的 Python 错误，如 ZeroDivisionError，
        # 而不依赖于 Pyodide 不支持的 `np.errstate`。
        result = float(numerator) / float(denominator)
        # 再次将分子和分母转换为 Python 浮点数，触发 ZeroDivisionError，不依赖于
        # Pyodide 不支持的 `np.errstate`。
        result = float(numerator) / float(denominator)
        # 如果结果为无穷大，发出运行时警告
        if math.isinf(result):
            warnings.warn("overflow encountered in _safe_divide", RuntimeWarning)
        # 返回计算结果
        return result
# 返回初始的原始预测值。

# Parameters参数说明:
# X : ndarray of shape (n_samples, n_features)
#     数据数组。
# estimator : object
#     用于计算预测值的估计器对象。
# loss : BaseLoss
#     损失函数类的实例。
# use_predict_proba : bool
#     是否使用 estimator.predict_proba 而不是 estimator.predict。

# Returns返回:
# raw_predictions : ndarray of shape (n_samples, K)
#     初始的原始预测值。对于二元分类和回归问题，K 等于 1；对于多类分类，K 等于类别数。
#     raw_predictions 被转换为 float64 类型。

def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    if use_predict_proba:
        # 我们通过 _fit_context 和 _parameter_constraints 设置了参数验证，
        # 确保 estimator 具有 predict_proba 方法。
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # 正类别的概率
        eps = np.finfo(np.float32).eps  # FIXME: 这个值有点大！
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)


# 更新树的叶节点值以及 raw_prediction 的值。

# 更新当前模型的原始预测值（当前阶段的预测值）。

# 另外，也会更新给定树的终端区域（叶节点）。这相当于“贪婪函数逼近”中的线性搜索步骤，
# 参考 Friedman 的算法1步骤5。

# 更新操作等同于：
#     argmin_{x} loss(y_true, raw_prediction_old + x * tree.value)

# 对于像二项损失这样的非平凡情况，更新没有闭式公式，只能进行近似计算，参见 Friedman 的论文。

# 还要注意，对于平方误差损失（SquaredError），更新公式是恒等式。因此，在这种情况下，
# 叶节点值不需要更新，只更新 raw_predictions（包含学习率）。

# Parameters参数说明:
# loss : BaseLoss
#     损失函数类。
# tree : tree.Tree
#     树对象。
# X : ndarray of shape (n_samples, n_features)
#     数据数组。
# y : ndarray of shape (n_samples,)
#     目标标签。
# neg_gradient : ndarray of shape (n_samples,)
#     负梯度。

def _update_terminal_regions(
    loss,
    tree,
    X,
    y,
    neg_gradient,
    raw_prediction,
    sample_weight,
    sample_mask,
    learning_rate=0.1,
    k=0,
):
    pass  # 这个函数还没有实现具体内容，所以暂时没有操作
    # raw_prediction : ndarray of shape (n_samples, n_trees_per_iteration)
    #     存储着树集合在第 ``i - 1`` 次迭代中每棵树的原始预测值（即叶子节点的值）。
    # sample_weight : ndarray of shape (n_samples,)
    #     每个样本的权重。
    # sample_mask : ndarray of shape (n_samples,)
    #     将要使用的样本掩码。
    # learning_rate : float, default=0.1
    #     学习率，通过 ``learning_rate`` 缩减每棵树的贡献。
    # k : int, default=0
    #     正在更新的估计器的索引。
    """
    # 对 ``X`` 中的每个样本计算其所在叶子节点。
    terminal_regions = tree.apply(X)
    
    # 更新预测值（包括袋内和袋外）
    raw_prediction[:, k] += learning_rate * tree.value[:, 0, 0].take(
        terminal_regions, axis=0
    )
def set_huber_delta(loss, y_true, raw_prediction, sample_weight=None):
    """Calculate and set self.closs.delta based on self.quantile."""
    # 计算绝对误差
    abserr = np.abs(y_true - raw_prediction.squeeze())
    # sample_weight 总是一个 ndarray，从不是 None。
    # 使用加权百分位数计算 delta 值
    delta = _weighted_percentile(abserr, sample_weight, 100 * loss.quantile)
    # 将计算得到的 delta 值赋给 loss.closs.delta
    loss.closs.delta = float(delta)


class VerboseReporter:
    """Reports verbose output to stdout.

    Parameters
    ----------
    verbose : int
        Verbosity level. If ``verbose==1`` output is printed once in a while
        (when iteration mod verbose_mod is zero).; if larger than 1 then output
        is printed for each update.
    """

    def __init__(self, verbose):
        # 设置输出详细程度
        self.verbose = verbose

    def init(self, est, begin_at_stage=0):
        """Initialize reporter

        Parameters
        ----------
        est : Estimator
            The estimator

        begin_at_stage : int, default=0
            stage at which to begin reporting
        """
        # 头部字段和行格式字符串
        header_fields = ["Iter", "Train Loss"]
        verbose_fmt = ["{iter:>10d}", "{train_score:>16.4f}"]
        # 是否进行 oob 计算？
        if est.subsample < 1:
            header_fields.append("OOB Improve")
            verbose_fmt.append("{oob_impr:>16.4f}")
        header_fields.append("Remaining Time")
        verbose_fmt.append("{remaining_time:>16s}")

        # 打印头部行
        print(("%10s " + "%16s " * (len(header_fields) - 1)) % tuple(header_fields))

        # 设置详细信息格式字符串
        self.verbose_fmt = " ".join(verbose_fmt)
        # 每次迭代 i % verbose_mod == 0 时绘制详细信息
        self.verbose_mod = 1
        self.start_time = time()
        self.begin_at_stage = begin_at_stage
    def update(self, j, est):
        """Update reporter with new iteration.

        Parameters
        ----------
        j : int
            新的迭代次数。
        est : Estimator
            用于估计的对象。

        """
        # 如果估计器使用了子采样，则需要考虑这一点
        do_oob = est.subsample < 1
        # 计算相对于开始迭代的索引
        i = j - self.begin_at_stage  # iteration relative to the start iter
        # 如果满足每个 verbose_mod 进行一次输出
        if (i + 1) % self.verbose_mod == 0:
            # 如果使用 out-of-bag 样本，则获取当前迭代的改善值
            oob_impr = est.oob_improvement_[j] if do_oob else 0
            # 计算剩余的训练时间估计
            remaining_time = (
                (est.n_estimators - (j + 1)) * (time() - self.start_time) / float(i + 1)
            )
            # 根据时间长度格式化剩余时间
            if remaining_time > 60:
                remaining_time = "{0:.2f}m".format(remaining_time / 60.0)
            else:
                remaining_time = "{0:.2f}s".format(remaining_time)
            # 打印格式化后的输出信息
            print(
                self.verbose_fmt.format(
                    iter=j + 1,
                    train_score=est.train_score_[j],
                    oob_impr=oob_impr,
                    remaining_time=remaining_time,
                )
            )
            # 如果 verbose 设置为 1，并且迭代次数达到了 verbose_mod 的多个整数倍时，调整 verbose_mod 的频率
            if self.verbose == 1 and ((i + 1) // (self.verbose_mod * 10) > 0):
                self.verbose_mod *= 10
# 定义一个名为 BaseGradientBoosting 的抽象基类，继承自 BaseEnsemble，使用 ABCMeta 作为元类
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""

    # 定义一个类属性 _parameter_constraints，是一个字典，包含一系列参数约束
    _parameter_constraints: dict = {
        **DecisionTreeRegressor._parameter_constraints,  # 继承决策树回归器的参数约束
        "learning_rate": [Interval(Real, 0.0, None, closed="left")],  # 学习率的约束条件
        "n_estimators": [Interval(Integral, 1, None, closed="left")],  # 基础估计器数量的约束条件
        "criterion": [StrOptions({"friedman_mse", "squared_error"})],  # 分裂标准的约束条件
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],  # 子样本比例的约束条件
        "verbose": ["verbose"],  # 是否输出详细信息的约束条件
        "warm_start": ["boolean"],  # 是否启用热启动的约束条件
        "validation_fraction": [Interval(Real, 0.0, 1.0, closed="neither")],  # 验证集比例的约束条件
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left"), None],  # 迭代无变化时停止的约束条件
        "tol": [Interval(Real, 0.0, None, closed="left")],  # 容忍度的约束条件
    }
    # 从参数约束中移除 "splitter" 和 "monotonic_cst" 两个约束条件
    _parameter_constraints.pop("splitter")
    _parameter_constraints.pop("monotonic_cst")

    # 定义抽象方法 __init__，用于初始化梯度提升算法的参数
    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        criterion,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_depth,
        min_impurity_decrease,
        init,
        subsample,
        max_features,
        ccp_alpha,
        random_state,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
    ):
        # 初始化各个参数
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    # 定义抽象方法 _encode_y，用于验证和编码目标变量 y
    @abstractmethod
    def _encode_y(self, y=None, sample_weight=None):
        """Called by fit to validate and encode y."""

    # 定义抽象方法 _get_loss，用于获取损失函数对象
    @abstractmethod
    def _get_loss(self, sample_weight):
        """Get loss object from sklearn._loss.loss."""

    # 定义方法 _fit_stage，用于执行单个训练阶段
    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
        """Fit another stage of ``n_trees_per_iteration_`` trees."""
        # 定义函数，用于拟合另一阶段的 `n_trees_per_iteration_` 棵树。

        original_y = y
        # 备份原始的 y 值，用于多分类问题中的处理。

        if isinstance(self._loss, HuberLoss):
            # 如果损失函数是 HuberLoss 类型，则设置 Huber 损失的 delta 值。
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight,
            )

        # TODO: Without oob, i.e. with self.subsample = 1.0, we could call
        # self._loss.loss_gradient and use it to set train_score_.
        # But note that train_score_[i] is the score AFTER fitting the i-th tree.
        # Note: We need the negative gradient!
        # 计算负梯度，用于树的训练。
        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,  # We pass sample_weights to the tree directly.
        )

        # 2-d views of shape (n_samples, n_trees_per_iteration_) or (n_samples, 1)
        # on neg_gradient to simplify the loop over n_trees_per_iteration_.
        # 将负梯度视图变为二维数组形式，便于对 n_trees_per_iteration_ 进行循环操作。
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                # 如果是多分类问题，则将原始 y 转换为浮点数数组。
                y = np.array(original_y == k, dtype=np.float64)

            # induce regression tree on the negative gradient
            # 在负梯度上训练回归树模型。
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter="best",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha,
            )

            if self.subsample < 1.0:
                # 如果采样率小于 1.0，则更新样本权重。
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            # 根据输入数据类型选择适当的 X
            X = X_csc if X_csc is not None else X
            # 使用样本权重和负梯度的第 k 列，拟合回归树模型。
            tree.fit(
                X, neg_g_view[:, k], sample_weight=sample_weight, check_input=False
            )

            # update tree leaves
            # 更新树的叶节点。
            X_for_tree_update = X_csr if X_csr is not None else X
            _update_terminal_regions(
                self._loss,
                tree.tree_,
                X_for_tree_update,
                y,
                neg_g_view[:, k],
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            # add tree to ensemble
            # 将训练好的树添加到集成模型中。
            self.estimators_[i, k] = tree

        return raw_predictions
    def _set_max_features(self):
        """Set self.max_features_."""
        # 如果 max_features 是字符串类型
        if isinstance(self.max_features, str):
            # 如果 max_features 是 "auto"
            if self.max_features == "auto":
                # 如果当前对象是分类器
                if is_classifier(self):
                    # 计算 max_features 为大于等于1的平方根整数
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    # 否则 max_features 等于输入特征数
                    max_features = self.n_features_in_
            # 如果 max_features 是 "sqrt"
            elif self.max_features == "sqrt":
                # 计算 max_features 为大于等于1的平方根整数
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:  # 如果 max_features 是 "log2"
                # 计算 max_features 为大于等于1的以2为底的对数整数
                max_features = max(1, int(np.log2(self.n_features_in_)))
        # 如果 max_features 是 None
        elif self.max_features is None:
            # max_features 等于输入特征数
            max_features = self.n_features_in_
        # 如果 max_features 是整数类型
        elif isinstance(self.max_features, Integral):
            # max_features 直接等于 self.max_features
            max_features = self.max_features
        else:  # 如果 max_features 是浮点数类型
            # 计算 max_features 为大于等于1的浮点数乘以输入特征数后的整数值
            max_features = max(1, int(self.max_features * self.n_features_in_))

        # 设置 self.max_features_ 为计算得到的 max_features
        self.max_features_ = max_features

    def _init_state(self):
        """Initialize model state and allocate model state data structures."""
        # 初始化 self.init_ 为 self.init，如果 self.init 为 None
        self.init_ = self.init
        if self.init_ is None:
            # 如果当前对象是分类器
            if is_classifier(self):
                # 使用 DummyClassifier 来初始化 self.init_
                self.init_ = DummyClassifier(strategy="prior")
            # 如果损失函数是绝对误差或者 Huber 损失
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                # 使用 DummyRegressor 来初始化 self.init_
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            # 如果损失函数是分位数损失
            elif isinstance(self._loss, PinballLoss):
                # 使用 DummyRegressor 来初始化 self.init_，设置分位数为 self.alpha
                self.init_ = DummyRegressor(strategy="quantile", quantile=self.alpha)
            else:
                # 否则使用 DummyRegressor 来初始化 self.init_
                self.init_ = DummyRegressor(strategy="mean")

        # 分配空的 numpy 数组来存储估计器和训练分数
        self.estimators_ = np.empty(
            (self.n_estimators, self.n_trees_per_iteration_), dtype=object
        )
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)

        # 如果 subsample 小于 1.0，需要做 out-of-bag 计算
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_scores_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_score_ = np.nan

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        # 清除模型状态中的相关属性
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
        if hasattr(self, "oob_improvement_"):
            del self.oob_improvement_
        if hasattr(self, "oob_scores_"):
            del self.oob_scores_
        if hasattr(self, "oob_score_"):
            del self.oob_score_
        if hasattr(self, "init_"):
            del self.init_
        if hasattr(self, "_rng"):
            del self._rng
    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes."""
        # 获取要添加的总的estimator数量
        total_n_estimators = self.n_estimators
        # 如果要添加的estimator数量小于当前已有的estimator数量，抛出数值错误异常
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError(
                "resize with smaller n_estimators %d < %d"
                % (total_n_estimators, self.estimators_[0])
            )

        # 调整self.estimators_的大小以容纳更多的estimator
        self.estimators_ = np.resize(
            self.estimators_, (total_n_estimators, self.n_trees_per_iteration_)
        )
        # 调整self.train_score_的大小以容纳更多的训练得分
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        # 如果subsample小于1或者存在oob_improvement_属性，则调整oob相关的数组大小或者创建新的数组
        if self.subsample < 1 or hasattr(self, "oob_improvement_"):
            # 如果存在oob_improvement_属性，则调整self.oob_improvement_和self.oob_scores_的大小
            if hasattr(self, "oob_improvement_"):
                self.oob_improvement_ = np.resize(
                    self.oob_improvement_, total_n_estimators
                )
                self.oob_scores_ = np.resize(self.oob_scores_, total_n_estimators)
                self.oob_score_ = np.nan
            # 否则，创建大小为total_n_estimators的零数组，并将self.oob_score_设置为NaN
            else:
                self.oob_improvement_ = np.zeros(
                    (total_n_estimators,), dtype=np.float64
                )
                self.oob_scores_ = np.zeros((total_n_estimators,), dtype=np.float64)
                self.oob_score_ = np.nan

    def _is_fitted(self):
        """Check if the estimator is fitted."""
        return len(getattr(self, "estimators_", [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        # 使用check_is_fitted函数检查模型是否已经拟合
        check_is_fitted(self)

    @_fit_context(
        # GradientBoosting*.init is not validated yet
        prefer_skip_nested_validation=False
    )
    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        random_state,
        X_val,
        y_val,
        sample_weight_val,
        begin_at_stage=0,
        monitor=None,
    ):
        """Fit stages of the model."""
        # 这个方法用于训练模型的各个阶段，在_fit_context装饰器下执行
        pass

    def _make_estimator(self, append=True):
        """Make an estimator object."""
        # 暂时不需要实现_make_estimator方法，因此抛出未实现错误
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        # 检查模型是否已经初始化
        self._check_initialized()
        # 使用第一个estimator验证输入数据X，并生成初始预测raw_predictions
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        # 如果初始化方式为"zero"，则生成全零的raw_predictions数组；否则调用_init_raw_predictions生成初始预测
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, is_classifier(self)
            )
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 初始化raw_predictions
        raw_predictions = self._raw_predict_init(X)
        # 对raw_predictions进行预测累加，得到最终的预测结果
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions
    def _staged_raw_predict(self, X, check_input=True):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            If False, the input arrays X will not be checked.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        if check_input:
            # Validate and preprocess input data X
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
        # Initialize raw predictions
        raw_predictions = self._raw_predict_init(X)
        # Iterate through each estimator stage
        for i in range(self.estimators_.shape[0]):
            # Predict using the current stage estimator
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
            # Yield a copy of raw predictions after each stage
            yield raw_predictions.copy()

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        # Ensure the estimator is initialized
        self._check_initialized()

        # Select relevant trees (those with more than one node)
        relevant_trees = [
            tree
            for stage in self.estimators_
            for tree in stage
            if tree.tree_.node_count > 1
        ]

        if not relevant_trees:
            # Handle the case where all trees are single node trees
            return np.zeros(shape=self.n_features_in_, dtype=np.float64)

        # Compute feature importances for relevant trees
        relevant_feature_importances = [
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in relevant_trees
        ]

        # Average feature importances across relevant trees
        avg_feature_importances = np.mean(
            relevant_feature_importances, axis=0, dtype=np.float64
        )

        # Normalize feature importances to sum up to 1
        return avg_feature_importances / np.sum(avg_feature_importances)
    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features), dtype=np.float32
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features,), dtype=np.intp
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape \
                (n_trees_per_iteration_, n_samples)
            The value of the partial dependence function on each grid point.
        """
        # 如果初始化不是None，则发出警告
        if self.init is not None:
            warnings.warn(
                "Using recursion method with a non-constant init predictor "
                "will lead to incorrect partial dependence values. "
                "Got init=%s." % self.init,
                UserWarning,
            )
        # 将grid转换为dtype为DTYPE（可能为float32）的ndarray，使用C顺序存储
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        # 获取estimators_的维度信息
        n_estimators, n_trees_per_stage = self.estimators_.shape
        # 初始化一个dtype为np.float64的零数组，用于存储平均预测值，使用C顺序存储
        averaged_predictions = np.zeros(
            (n_trees_per_stage, grid.shape[0]), dtype=np.float64, order="C"
        )
        # 将target_features转换为dtype为np.intp的ndarray，使用C顺序存储
        target_features = np.asarray(target_features, dtype=np.intp, order="C")

        # 遍历每个estimator的每个阶段
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                # 获取当前树的引用
                tree = self.estimators_[stage, k].tree_
                # 计算当前树对于给定grid和target_features的部分依赖，并更新averaged_predictions
                tree.compute_partial_dependence(
                    grid, target_features, averaged_predictions[k]
                )
        # 乘以学习率，以得到最终的平均预测值
        averaged_predictions *= self.learning_rate

        return averaged_predictions
    def apply(self, X):
        """
        将集成中的每棵树应用到 X 上，并返回叶子节点索引。

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            输入样本。其数据类型将被转换为 ``dtype=np.float32``。如果提供了稀疏矩阵，
            将会转换为稀疏的 ``csr_matrix``。

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            对于 X 中的每个数据点 x 和集成中的每棵树，返回 x 所在叶子节点的索引。
            在二元分类情况下，n_classes 为 1。
        """

        self._check_initialized()  # 检查集成是否已经初始化

        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        # 验证输入 X 并转换其数据类型，确保输入合法性

        # n_classes 在二元分类或回归案例中将等于 1。
        n_estimators, n_classes = self.estimators_.shape

        # 初始化一个全零数组，用于存储每个样本在每个估计器中的叶子节点索引
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        # 遍历每个估计器和每个类别
        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                # 应用当前估计器到输入 X 上，获取叶子节点索引，不检查输入
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves
class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.

    This algorithm builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage ``n_classes_`` regression trees are fit on the negative gradient
    of the loss function, e.g. binary or multiclass log loss. Binary
    classification is a special case where only a single regression tree is
    induced.

    :class:`sklearn.ensemble.HistGradientBoostingClassifier` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'log_loss', 'exponential'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to binomial and
        multinomial deviance, the same as used in logistic regression.
        It is a good choice for classification with probabilistic outputs.
        For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        'friedman_mse' for the mean squared error with improvement score by
        Friedman, 'squared_error' for mean squared error. The default value of
        'friedman_mse' is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.
"""
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.
           
        最小的叶子节点所需的样本数。在任何深度上只有当左右分支中的每个分支至少留下``min_samples_leaf``个训练样本时才会考虑分割点。这可能会平滑模型，尤其是在回归问题中。

        - 如果是整数，则值必须在 `[1, inf)` 范围内。
        - 如果是浮点数，则值必须在 `(0.0, 1.0)` 范围内，并且 `min_samples_leaf` 将会是 `ceil(min_samples_leaf * n_samples)`。

        .. versionchanged:: 0.18
           添加了浮点数用于表示分数。

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

        叶节点的最小加权样本权重和占总样本权重之比。当未提供 `sample_weight` 时，样本权重相等。
        值必须在 `[0.0, 0.5]` 范围内。

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

        个体回归估算器的最大深度。最大深度限制树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的交互作用。
        如果为 None，则节点会一直扩展，直到所有叶子节点都是纯净的或者所有叶子节点包含的样本数少于 `min_samples_split`。
        如果是整数，则值必须在 `[1, inf)` 范围内。

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        如果此次分割导致不纯度减少大于或等于此值，则会分裂节点。
        值必须在 `[0.0, inf)` 范围内。

        加权不纯度减少的方程如下：

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中，`N` 是总样本数，`N_t` 是当前节点的样本数，`N_t_L` 是左子节点的样本数，`N_t_R` 是右子节点的样本数。

        如果传递了 `sample_weight`，则所有这些数都是加权和。

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

        用于计算初始预测的估算器对象。`init` 必须提供 `fit` 和 `predict_proba` 方法。如果设为 'zero'，则初始原始预测设为零。默认情况下，使用预测类先验的 `DummyEstimator`。
    # 控制每棵树的随机种子，影响每次 boosting 迭代中 Tree 估计器的随机性
    # 此外，控制每次分裂时特征的随机排列（详见备注）
    # 如果 `n_iter_no_change` 不为 None，还控制训练数据的随机分割以获取验证集
    # 设置为 int 可实现跨多次函数调用时的可重复输出
    # 参见“术语表 <random_state>”
    random_state : int, RandomState instance or None, default=None

    # 在寻找最佳分割时考虑的特征数量：
    # - 如果是 int，则值必须在区间 `[1, inf)`
    # - 如果是 float，则值必须在区间 `(0.0, 1.0]`，每次分割考虑的特征数量为 `max(1, int(max_features * n_features_in_))`
    # - 如果是 'sqrt'，则 `max_features=sqrt(n_features)`
    # - 如果是 'log2'，则 `max_features=log2(n_features)`
    # - 如果是 None，则 `max_features=n_features`
    # 选择 `max_features < n_features` 可降低方差，增加偏差
    # 注意：即使需要检查超过 `max_features` 个特征，寻找分割时也不会停止
    max_features : {'sqrt', 'log2'}, int or float, default=None

    # 控制输出详细程度
    # - 如果为 1，则定期打印进度和性能（树越多，频率越低）
    # - 如果大于 1，则每棵树都打印进度和性能
    # 值必须在区间 `[0, inf)`
    verbose : int, default=0

    # 以最佳优先方式生长具有 `max_leaf_nodes` 的树
    # 最佳节点定义为杂质的相对减少
    # 值必须在区间 `[2, inf)`
    # 如果为 `None`，则叶节点数量不受限制
    max_leaf_nodes : int, default=None

    # 当设置为 `True` 时，重复使用前一次拟合调用的解决方案，并向集成中添加更多的估计器
    # 否则，只是擦除前一个解决方案
    # 参见“术语表 <warm_start>”
    warm_start : bool, default=False

    # 用作早期停止的验证集的训练数据比例
    # 值必须在区间 `(0.0, 1.0)`
    # 仅在 `n_iter_no_change` 设置为整数时使用
    # 版本新增于 0.20
    validation_fraction : float, default=0.1
    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range `[1, inf)`.
        See
        :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    n_trees_per_iteration_ : int
        The number of trees that are built at each iteration. For binary classifiers,
        this is always 1.

        .. versionadded:: 1.4.0

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3
    oob_score_ : float
        # out-of-bag 样本上的最后一个损失值，等同于 `oob_scores_[-1]`。仅在 `subsample < 1.0` 时可用。
        # 自版本 1.3 起添加

    train_score_ : ndarray of shape (n_estimators,)
        # 第 i 个分数 `train_score_[i]` 是模型在第 i 次迭代时在 in-bag 样本上的损失。
        # 如果 `subsample == 1`，则这是在训练数据上的损失。

    init_ : estimator
        # 提供初始预测的估算器。通过 `init` 参数设置。

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, ``n_trees_per_iteration_``)
        # 拟合的子估算器的集合。对于二元分类，`n_trees_per_iteration_` 为 1，否则为 `n_classes`。

    classes_ : ndarray of shape (n_classes,)
        # 类别标签。

    n_features_in_ : int
        # 在 `fit` 过程中看到的特征数量。
        # 自版本 0.24 起添加

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中看到的特征名称。仅在 `X` 中的特征名称全部为字符串时定义。
        # 自版本 1.0 起添加

    n_classes_ : int
        # 类别的数量。

    max_features_ : int
        # 推断得到的 `max_features` 值。

    See Also
    --------
    HistGradientBoostingClassifier : 基于直方图的梯度提升分类树。
    sklearn.tree.DecisionTreeClassifier : 决策树分类器。
    RandomForestClassifier : 元估计器，对数据集的多个子样本拟合决策树分类器，并使用平均化来提高预测准确性并控制过拟合。
    AdaBoostClassifier : 元估计器，首先在原始数据集上拟合分类器，然后在权重调整过的相同数据集上拟合额外的分类器，使得后续的分类器更专注于难例。

    Notes
    -----
    每次分裂时特征总是随机排列的。因此，即使对于相同的训练数据和 `max_features=n_features`，如果准则的改进在搜索最佳分裂期间对几个分裂枚举是相同的，最佳找到的分裂也可能不同。
    要在拟合过程中获得确定性行为，必须固定 `random_state`。

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    下面的示例展示了如何使用 100 个决策桩作为弱学习器拟合梯度提升分类器。
    """
    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }
    """

    # 定义一个字典，用于参数约束，继承自 BaseGradientBoosting 的参数约束
    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        # 损失函数可选项，可以是 'log_loss' 或 'exponential'
        "loss": [StrOptions({"log_loss", "exponential"})],
        # 初始化方法可选项，可以是 'zero'，或者 None，或者具有 'fit' 和 'predict_proba' 方法的对象
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        # 损失函数，默认为 'log_loss'
        loss="log_loss",
        # 学习率，默认为 0.1
        learning_rate=0.1,
        # 弱学习器数量，默认为 100
        n_estimators=100,
        # 子采样比例，默认为 1.0（不进行子采样）
        subsample=1.0,
        # 划分标准，默认为 'friedman_mse'
        criterion="friedman_mse",
        # 最小分割样本数，默认为 2
        min_samples_split=2,
        # 最小叶子节点样本数，默认为 1
        min_samples_leaf=1,
        # 最小叶子节点权重和的最小加权分数，默认为 0.0
        min_weight_fraction_leaf=0.0,
        # 最大树深度，默认为 3
        max_depth=3,
        # 最小不纯度减少量，默认为 0.0
        min_impurity_decrease=0.0,
        # 初始化方法，默认为 None
        init=None,
        # 随机数种子，默认为 None
        random_state=None,
        # 最大特征数，默认为 None（即所有特征）
        max_features=None,
        # 是否打印详细信息，默认为 0（不打印）
        verbose=0,
        # 最大叶子节点数，默认为 None（不限制）
        max_leaf_nodes=None,
        # 温和启动，默认为 False
        warm_start=False,
        # 验证集分数所占比例，默认为 0.1
        validation_fraction=0.1,
        # 连续多少次迭代不改善验证集分数停止训练，默认为 None（不停止）
        n_iter_no_change=None,
        # 公差，默认为 1e-4
        tol=1e-4,
        # CCP α 参数，默认为 0.0
        ccp_alpha=0.0,
    ):
        # 调用父类 BaseGradientBoosting 的构造函数，初始化梯度提升分类器
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
    # 将目标变量 y 编码为 0 到 n_classes - 1，并设置类属性 classes_ 和 n_trees_per_iteration_
    def _encode_y(self, y, sample_weight):
        # 检查分类目标 y 的有效性
        check_classification_targets(y)

        # 使用 LabelEncoder 对象对目标变量 y 进行编码
        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        # 设置类属性 classes_
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # 根据类别数量确定每次迭代中的树的数量
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        # 将编码后的 y 转换为浮点数类型
        encoded_y = encoded_y_int.astype(float, copy=False)

        # 以下内容与 HGBT 案例相关
        # 暴露 n_classes_ 属性
        self.n_classes_ = n_classes

        # 根据样本权重确定有效的类别数量 n_trim_classes
        if sample_weight is None:
            n_trim_classes = n_classes
        else:
            n_trim_classes = np.count_nonzero(np.bincount(encoded_y_int, sample_weight))

        # 如果 n_trim_classes 小于 2，则抛出数值错误
        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        # 返回编码后的 y
        return encoded_y

    # 获取损失函数对象根据样本权重的设置
    def _get_loss(self, sample_weight):
        # 如果损失函数为 "log_loss"
        if self.loss == "log_loss":
            # 如果类别数量为 2，返回 HalfBinomialLoss 对象
            if self.n_classes_ == 2:
                return HalfBinomialLoss(sample_weight=sample_weight)
            # 如果类别数量大于 2，返回 HalfMultinomialLoss 对象
            else:
                return HalfMultinomialLoss(
                    sample_weight=sample_weight, n_classes=self.n_classes_
                )
        # 如果损失函数为 "exponential"
        elif self.loss == "exponential":
            # 如果类别数量大于 2，抛出数值错误
            if self.n_classes_ > 2:
                raise ValueError(
                    f"loss='{self.loss}' is only suitable for a binary classification "
                    f"problem, you have n_classes={self.n_classes_}. "
                    "Please use loss='log_loss' instead."
                )
            # 否则返回 ExponentialLoss 对象
            else:
                return ExponentialLoss(sample_weight=sample_weight)
    def decision_function(self, X):
        """
        Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble. The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        # Validate and convert input data to dtype=np.float32 and csr_matrix format if sparse
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # Get raw predictions from the underlying estimator
        raw_predictions = self._raw_predict(X)
        # If there is only one class, return flattened raw predictions
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        # Otherwise, return the raw predictions
        return raw_predictions

    def staged_decision_function(self, X):
        """
        Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble. The
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        # Yield staged raw predictions from the underlying estimator
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        # Get raw predictions using the decision_function method
        raw_predictions = self.decision_function(X)
        # If raw_predictions is 1-dimensional, directly encode classes based on threshold
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            # Otherwise, find the class index with the highest score for each sample
            encoded_classes = np.argmax(raw_predictions, axis=1)
        # Return the predicted classes corresponding to the encoded class indices
        return self.classes_[encoded_classes]
    def staged_predict(self, X):
        """
        Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        if self.n_classes_ == 2:  # Check if number of classes is 2
            # Iterate over staged raw predictions
            for raw_predictions in self._staged_raw_predict(X):
                # Encode classes based on raw predictions
                encoded_classes = (raw_predictions.squeeze() >= 0).astype(int)
                # Yield predicted classes based on encoding
                yield self.classes_.take(encoded_classes, axis=0)
        else:
            # Iterate over staged raw predictions
            for raw_predictions in self._staged_raw_predict(X):
                # Find class with maximum probability for each sample
                encoded_classes = np.argmax(raw_predictions, axis=1)
                # Yield predicted classes based on encoding
                yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        # Compute raw predictions using decision function
        raw_predictions = self.decision_function(X)
        # Return predicted probabilities using loss function
        return self._loss.predict_proba(raw_predictions)

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        # Compute probabilities using predict_proba method
        proba = self.predict_proba(X)
        # Compute log of probabilities
        return np.log(proba)
    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        try:
            # Iterate over raw predictions from each stage
            for raw_predictions in self._staged_raw_predict(X):
                # Predict probabilities using the loss function's predict_proba method
                yield self._loss.predict_proba(raw_predictions)
        except NotFittedError:
            # Propagate the exception if the model is not fitted
            raise
        except AttributeError as e:
            # Raise a detailed AttributeError if the loss function does not support predict_proba
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e
class Gradient`
class GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting):
    """Gradient Boosting for regression.

    This estimator builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage a regression tree is fit on the negative gradient of the given
    loss function.

    :class:`sklearn.ensemble.HistGradientBoostingRegressor` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'huber', 'quantile'}, \
            default='squared_error'
        Loss function to be optimized. 'squared_error' refers to the squared
        error for regression. 'absolute_error' refers to the absolute error of
        regression and is a robust loss function. 'huber' is a
        combination of the two. 'quantile' allows quantile regression (use
        `alpha` to specify the quantile).

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        "friedman_mse" for the mean squared error with improvement score by
        Friedman, "squared_error" for mean squared error. The default value of
        "friedman_mse" is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        叶节点所需的最小样本数。
        在任何深度上的分割点只有在左右分支中至少留下 ``min_samples_leaf`` 训练样本时才会考虑。
        这可能会使模型平滑化，特别是在回归问题中。

        - 如果是整数，则值必须在 `[1, inf)` 范围内。
        - 如果是浮点数，则值必须在 `(0.0, 1.0)` 范围内，且 `min_samples_leaf`
          将被设置为 `ceil(min_samples_leaf * n_samples)`。

        .. versionchanged:: 0.18
           添加了浮点数值以表示比例。

    min_weight_fraction_leaf : float, default=0.0
        叶节点的最小加权样本总权重比例。当未提供 sample_weight 时，样本权重相等。
        值必须在 `[0.0, 0.5]` 范围内。

    max_depth : int or None, default=3
        每个回归估算器的最大深度。最大深度限制树中的节点数量。
        调整此参数以获得最佳性能；最佳值取决于输入变量的交互作用。
        如果为 None，则节点将展开直到所有叶子节点均为纯净节点或所有叶子节点包含的样本数少于
        min_samples_split。

        如果是整数，则值必须在 `[1, inf)` 范围内。

    min_impurity_decrease : float, default=0.0
        如果此分割导致不纯度减少大于或等于此值，则将分裂节点。
        值必须在 `[0.0, inf)` 范围内。

        加权不纯度减少公式如下::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中 ``N`` 是总样本数，``N_t`` 是当前节点的样本数，``N_t_L`` 是左子节点的样本数，
        ``N_t_R`` 是右子节点的样本数。

        如果传递了 ``sample_weight``，则 ``N``、``N_t``、``N_t_R`` 和 ``N_t_L`` 都是加权和。

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        用于计算初始预测的估算器对象。
        ``init`` 必须提供 :term:`fit` 和 :term:`predict` 方法。
        如果设为 'zero'，则初始原始预测值被设置为零。
        默认情况下使用 ``DummyEstimator``，对于损失函数为 'squared_error'，预测平均目标值，
        其他损失函数则预测一个分位数。
    # 控制每个 Tree 估计器在每次 boosting 迭代时的随机种子
    # 此外，它控制每次分裂时特征的随机排列（详见 Notes）
    # 如果 `n_iter_no_change` 不为 None，则它还控制训练数据的随机分割以获取验证集
    # 传入一个整数以确保在多次函数调用时输出一致的结果
    # 参见 :term:`Glossary <random_state>`
    random_state : int, RandomState instance or None, default=None
    
    # 在寻找最佳分裂时考虑的特征数量：
    # - 如果是整数，则值必须在 `[1, inf)` 范围内
    # - 如果是浮点数，则值必须在 `(0.0, 1.0]` 范围内，每次分裂考虑的特征将是 `max(1, int(max_features * n_features_in_))`
    # - 如果是 "sqrt"，则 `max_features=sqrt(n_features)`
    # - 如果是 "log2"，则 `max_features=log2(n_features)`
    # - 如果是 None，则 `max_features=n_features`
    # 选择 `max_features < n_features` 可以减少方差但会增加偏差
    # 注意：寻找分裂不会停止，直到找到至少一个有效的节点样本分区，即使需要实际检查超过 `max_features` 个特征
    max_features : {'sqrt', 'log2'}, int or float, default=None
    
    # huber 损失函数和分位数损失函数的 alpha 分位数
    # 仅在 `loss='huber'` 或 `loss='quantile'` 时有效
    # 值必须在 `(0.0, 1.0)` 范围内
    alpha : float, default=0.9
    
    # 启用详细输出
    # 如果为 1，则偶尔打印进度和性能（树越多，频率越低）
    # 如果大于 1，则为每棵树打印进度和性能
    # 值必须在 `[0, inf)` 范围内
    verbose : int, default=0
    
    # 以最佳优先方式生成带有 `max_leaf_nodes` 的树
    # 最佳节点定义为不纯度的相对减少
    # 值必须在 `[2, inf)` 范围内
    # 如果是 None，则叶节点数目不受限制
    max_leaf_nodes : int, default=None
    
    # 当设置为 True 时，重复使用前一次调用 `fit` 的解决方案，并向集成中添加更多估计器
    # 否则，只擦除先前的解决方案
    # 参见 :term:`the Glossary <warm_start>`
    warm_start : bool, default=False
    
    # 用作提前停止的验证集的训练数据的比例
    # 值必须在 `(0.0, 1.0)` 范围内
    # 仅在 `n_iter_no_change` 设置为整数时使用
    validation_fraction : float, default=0.1
    n_iter_no_change : int, default=None
        ``n_iter_no_change``用于决定是否启用早停机制来终止训练，当验证集分数不再改善时。默认情况下为None，禁用早停机功能。如果设置为一个数字，它将保留训练数据大小为``validation_fraction``的一部分作为验证集，并在前``n_iter_no_change``次迭代中验证集分数未改善时终止训练。
        取值范围必须在 `[1, inf)`之间。
        参见: :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        早停机制的容忍度。当损失在``n_iter_no_change``次迭代中未至少改善tol时（如果设置为一个数字），则停止训练。
        取值范围必须在 `[0.0, inf)`之间。

        .. versionadded:: 0.20

    ccp_alpha : 非负浮点数, default=0.0
        用于最小成本复杂性修剪的复杂性参数。选择成本复杂度大于``ccp_alpha``但最小的子树。默认情况下，不执行修剪。
        取值范围必须在 `[0.0, inf)`之间。
        详细信息请参见: :ref:`minimal_cost_complexity_pruning`.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        通过早停机（如果指定``n_iter_no_change``）选择的估计器数量。否则将设置为``n_estimators``。

    n_trees_per_iteration_ : int
        每次迭代构建的树的数量。对于回归器，这始终为1。

        .. versionadded:: 1.4.0

    feature_importances_ : 形状为 (n_features,) 的 ndarray
        基于杂质的特征重要性。
        数值越高，特征越重要。
        特征重要性计算为由该特征带来的（标准化的）准则总减少量。也称为基尼重要性。

        警告: 基于杂质的特征重要性对于基数较高的特征（具有许多唯一值）可能具有误导性。请参见: :func:`sklearn.inspection.permutation_importance` 作为替代方法。

    oob_improvement_ : 形状为 (n_estimators,) 的 ndarray
        相对于上一次迭代，对袋外样本损失的改善。
        ``oob_improvement_[0]``是第一个阶段相对于``init``估计器的损失改善量。
        仅当``subsample < 1.0``时可用。

    oob_scores_ : 形状为 (n_estimators,) 的 ndarray
        对袋外样本的损失值完整历史记录。仅当`subsample < 1.0`时可用。

        .. versionadded:: 1.3
    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3
        # oob_score_ 表示袋外样本的损失的最后一个值。如果 `subsample < 1.0`，则与 `oob_scores_[-1]` 相同。

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.
        # train_score_ 是一个形状为 (n_estimators,) 的 ndarray，表示模型在第 i 次迭代时对应的损失值。
        # 如果 `subsample == 1`，则这是在训练数据上的损失值。

    init_ : estimator
        The estimator that provides the initial predictions. Set via the ``init``
        argument.
        # init_ 是一个估计器，用于提供初始预测值，通过参数 `init` 设置。

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.
        # estimators_ 是一个形状为 (n_estimators, 1) 的 ndarray，表示拟合的子估计器的集合。

    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
        # n_features_in_ 表示在拟合过程中观察到的特征数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
        # feature_names_in_ 是一个形状为 (`n_features_in_`,) 的 ndarray，
        # 表示在拟合过程中观察到的特征名称。仅在 `X` 具有所有字符串特征名称时定义。

    max_features_ : int
        The inferred value of max_features.
        # max_features_ 表示推断出的 `max_features` 的值。

    See Also
    --------
    HistGradientBoostingRegressor : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.
    # 参见项，列出了与 GradientBoostingRegressor 相关的其他类和模型。

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    # 注意事项，说明了特征在每次分割时都会随机排列。因此，即使使用相同的训练数据和 `max_features=n_features`，
    # 如果准则的改进对于搜索最佳分割过程中列举的多个分割是相同的，最佳找到的分割可能会有所不同。
    # 为了在拟合过程中获得确定性行为，必须固定 `random_state`。

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    # 引用文献，列出了与 Gradient Boosting 相关的参考文献。

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = GradientBoostingRegressor(random_state=0)
    >>> reg.fit(X_train, y_train)
    GradientBoostingRegressor(random_state=0)
    >>> reg.predict(X_test[1:2])
    array([-61...])
    >>> reg.score(X_test, y_test)
    0.4...

    For a detailed example of utilizing
    :class:`~sklearn.ensemble.GradientBoostingRegressor`
    to fit an ensemble of weak predictive models, please refer to
    :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_regression.py`.
    # 示例，展示了如何使用 GradientBoostingRegressor 拟合弱预测模型的集合。
    # 定义参数约束字典，继承基类的参数约束，并添加特定于梯度提升树的约束条件
    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"squared_error", "absolute_error", "huber", "quantile"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict"])],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
    }

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        # 调用基类构造函数，初始化梯度提升回归树的参数
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )

    def _encode_y(self, y=None, sample_weight=None):
        # 将y转换为预期的数据类型
        self.n_trees_per_iteration_ = 1
        y = y.astype(DOUBLE, copy=False)
        return y

    def _get_loss(self, sample_weight):
        # 根据损失函数类型返回相应的损失函数对象，并传递相应的参数
        if self.loss in ("quantile", "huber"):
            return _LOSSES[self.loss](sample_weight=sample_weight, quantile=self.alpha)
        else:
            return _LOSSES[self.loss](sample_weight=sample_weight)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        # 验证输入数据X的格式和类型，并按需转换
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # 在回归中，直接返回树模型的原始预测值
        return self._raw_predict(X).ravel()
    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        # Iterate over raw predictions generated by _staged_raw_predict method
        for raw_predictions in self._staged_raw_predict(X):
            # Yield flattened predictions for each stage
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """
        # Apply the base class method to get the leaf indices for input samples
        leaves = super().apply(X)
        # Reshape the leaf indices to match the shape (n_samples, n_estimators)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        # Return the reshaped leaf indices
        return leaves
```