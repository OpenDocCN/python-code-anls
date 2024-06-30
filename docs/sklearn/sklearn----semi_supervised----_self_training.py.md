# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\_self_training.py`

```
# 引入警告模块，用于管理警告信息的显示
import warnings
# 从 numbers 模块中引入 Integral（整数）和 Real（实数）类型，用于参数类型验证
from numbers import Integral, Real
# 引入 numpy 库，并将其命名为 np，用于数值计算
import numpy as np
# 从当前包的 base 模块中引入 BaseEstimator（基本估计器）、MetaEstimatorMixin（元估计器混合类）、_fit_context（适配上下文）、clone（克隆函数）
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
# 从 utils 模块中引入 safe_mask 函数，用于安全地获取掩码
from ..utils import safe_mask
# 从 utils._param_validation 模块中引入 HasMethods（具有方法）、Interval（区间验证）、StrOptions（字符串选项验证），用于参数验证
from ..utils._param_validation import HasMethods, Interval, StrOptions
# 从 utils.metadata_routing 模块中引入 _RoutingNotSupportedMixin（路由不支持混合类），用于元数据路由管理
from ..utils.metadata_routing import _RoutingNotSupportedMixin
# 从 utils.metaestimators 模块中引入 available_if 函数，用于条件性可用性检查
from ..utils.metaestimators import available_if
# 从 utils.validation 模块中引入 check_is_fitted 函数，用于验证估计器是否已经适配数据
from ..utils.validation import check_is_fitted

# 定义一个全局变量 __all__，包含当前模块中需要导出的公共接口
__all__ = ["SelfTrainingClassifier"]

# 作者信息：scikit-learn 开发团队
# 许可证信息：BSD-3-Clause


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted `base_estimator_` if available, otherwise we check
    the unfitted `base_estimator`. We raise the original `AttributeError` if
    `attr` does not exist. This function is used together with `available_if`.
    """
    # 定义一个内部函数 check，用于检查是否可以将方法委托给基础估计器
    def check(self):
        # 如果存在已适配的 base_estimator_ 属性，则调用其 attr 方法
        if hasattr(self, "base_estimator_"):
            getattr(self.base_estimator_, attr)
        else:
            # 否则，调用未适配的 base_estimator 的 attr 方法
            getattr(self.base_estimator, attr)

        return True

    return check


class SelfTrainingClassifier(
    _RoutingNotSupportedMixin, MetaEstimatorMixin, BaseEstimator
):
    """Self-training classifier.

    This :term:`metaestimator` allows a given supervised classifier to function as a
    semi-supervised classifier, allowing it to learn from unlabeled data. It
    does this by iteratively predicting pseudo-labels for the unlabeled data
    and adding them to the training set.

    The classifier will continue iterating until either max_iter is reached, or
    no pseudo-labels were added to the training set in the previous iteration.

    Read more in the :ref:`User Guide <self_training>`.

    Parameters
    ----------
    base_estimator : estimator object
        An estimator object implementing `fit` and `predict_proba`.
        Invoking the `fit` method will fit a clone of the passed estimator,
        which will be stored in the `base_estimator_` attribute.

    threshold : float, default=0.75
        The decision threshold for use with `criterion='threshold'`.
        Should be in [0, 1). When using the `'threshold'` criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    criterion : {'threshold', 'k_best'}, default='threshold'
        The selection criterion used to select which labels to add to the
        training set. If `'threshold'`, pseudo-labels with prediction
        probabilities above `threshold` are added to the dataset. If `'k_best'`,
        the `k_best` pseudo-labels with highest prediction probabilities are
        added to the dataset. When using the 'threshold' criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    k_best : int, default=10
        The amount of samples to add in each iteration. Only used when
        `criterion='k_best'`.
    """
    # 自我训练分类器类，继承自 _RoutingNotSupportedMixin（路由不支持混合类）、MetaEstimatorMixin（元估计器混合类）、BaseEstimator（基本估计器）

    # 构造函数的文档字符串，介绍了自我训练分类器的功能、用法和参数说明
    def __init__(
        self,
        base_estimator,
        threshold=0.75,
        criterion="threshold",
        k_best=10,
    ):
        # 初始化函数，接受以下参数：
        # - base_estimator: 实现了 `fit` 和 `predict_proba` 方法的估计器对象
        # - threshold: 决策阈值，用于 `criterion='threshold'` 时的使用，默认为 0.75
        # - criterion: 选择添加到训练集的标签的标准，可选值为 {'threshold', 'k_best'}，默认为 'threshold'
        # - k_best: 每次迭代中要添加的样本数量，仅在 `criterion='k_best'` 时使用，默认为 10
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
    max_iter : int or None, default=10
        # 最大迭代次数，允许的最大迭代次数。应大于或等于0。如果为 `None`，分类器将继续预测标签，直到没有添加新的伪标签，或所有未标记样本都已标记。

    verbose : bool, default=False
        # 启用详细输出。

    Attributes
    ----------
    base_estimator_ : estimator object
        # 已拟合的估计器。

    classes_ : ndarray or list of ndarray of shape (n_classes,)
        # 每个输出的类标签。 (从训练过的 `base_estimator_` 获取)。

    transduction_ : ndarray of shape (n_samples,)
        # 分类器最终拟合时使用的标签，包括拟合过程中添加的伪标签。

    labeled_iter_ : ndarray of shape (n_samples,)
        # 每个样本被标记的迭代次数。当一个样本具有迭代次数 0 时，该样本已在原始数据集中标记。当一个样本具有迭代次数 -1 时，该样本在任何迭代中均未标记。

    n_features_in_ : int
        # 在 `fit` 过程中看到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中看到的特征名称。仅在 `X` 具有全部字符串特征名时定义。

        .. versionadded:: 1.0

    n_iter_ : int
        # 自训练的轮数，即基础估计器在训练集的重新标记变体上拟合的次数。

    termination_condition_ : {'max_iter', 'no_change', 'all_labeled'}
        # 停止拟合的原因。

        - `'max_iter'`: `n_iter_` 达到了 `max_iter`。
        - `'no_change'`: 没有预测到新的标签。
        - `'all_labeled'`: 所有未标记样本在达到 `max_iter` 之前都已被标记。

    See Also
    --------
    LabelPropagation : 标签传播分类器。
    LabelSpreading : 半监督学习的标签传播模型。

    References
    ----------
    :doi:`David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling
    supervised methods. In Proceedings of the 33rd annual meeting on
    Association for Computational Linguistics (ACL '95). Association for
    Computational Linguistics, Stroudsburg, PA, USA, 189-196.
    <10.3115/981658.981684>`
    # 参考文献，引用了 David Yarowsky 的研究论文，讨论了无监督词义消歧与监督方法的竞争性。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import SelfTrainingClassifier
    >>> from sklearn.svm import SVC
    >>> rng = np.random.RandomState(42)
    >>> iris = datasets.load_iris()
    >>> random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
    >>> iris.target[random_unlabeled_points] = -1
    >>> svc = SVC(probability=True, gamma="auto")
    >>> self_training_model = SelfTrainingClassifier(svc)
    >>> self_training_model.fit(iris.data, iris.target)
    SelfTrainingClassifier(...)
    ```
    # 设定模型的类型为分类器
    _estimator_type = "classifier"

    # 定义参数的约束条件字典
    _parameter_constraints: dict = {
        # 不要求 `predic_proba`，以允许传递一个仅在拟合后才公开 `predict_proba` 的元估计器
        "base_estimator": [HasMethods(["fit"])],
        # 阈值必须在 [0.0, 1.0) 的实数范围内
        "threshold": [Interval(Real, 0.0, 1.0, closed="left")],
        # 准则只能是 "threshold" 或 "k_best" 中的一个字符串
        "criterion": [StrOptions({"threshold", "k_best"})],
        # k_best 必须是大于等于 1 的整数
        "k_best": [Interval(Integral, 1, None, closed="left")],
        # 最大迭代次数必须是大于等于 0 的整数，或者可以是 None
        "max_iter": [Interval(Integral, 0, None, closed="left"), None],
        # 是否显示详细信息，可为布尔值
        "verbose": ["verbose"],
    }

    # 初始化方法，设置模型的各个参数
    def __init__(
        self,
        base_estimator,
        threshold=0.75,
        criterion="threshold",
        k_best=10,
        max_iter=10,
        verbose=False,
    ):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose

    # 装饰器函数，用于预测类别
    @_fit_context(
        # SelfTrainingClassifier.base_estimator 尚未验证
        prefer_skip_nested_validation=False
    )
    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Predict the classes of `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 验证数据格式
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        # 返回预测的结果标签
        return self.base_estimator_.predict(X)

    # 装饰器函数，用于预测每个可能结果的概率
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Array with prediction probabilities.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 验证数据格式
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        # 返回预测的概率结果
        return self.base_estimator_.predict_proba(X)

    # 装饰器函数，用于决策函数
    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """
        Call decision function of the `base_estimator`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Result of the decision function of the `base_estimator`.
        """
        # 检查当前对象是否已经拟合
        check_is_fitted(self)
        # 验证输入数据，并允许稀疏矩阵，不强制所有元素有限，保持不重置
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        # 调用基础估算器的决策函数并返回结果
        return self.base_estimator_.decision_function(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """
        Predict log probability for each possible outcome.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Array with log prediction probabilities.
        """
        # 检查当前对象是否已经拟合
        check_is_fitted(self)
        # 验证输入数据，并允许稀疏矩阵，不强制所有元素有限，保持不重置
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        # 调用基础估算器的预测对数概率函数并返回结果
        return self.base_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("score"))
    def score(self, X, y):
        """
        Call score on the `base_estimator`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        y : array-like of shape (n_samples,)
            Array representing the labels.

        Returns
        -------
        score : float
            Result of calling score on the `base_estimator`.
        """
        # 检查当前对象是否已经拟合
        check_is_fitted(self)
        # 验证输入数据，并允许稀疏矩阵，不强制所有元素有限，保持不重置
        X = self._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        # 调用基础估算器的得分函数并返回结果
        return self.base_estimator_.score(X, y)
```