# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_sequential.py`

```
"""
Sequential feature selection
"""

from numbers import Integral, Real  # 导入需要用到的模块：Integral 和 Real，用于数值类型判断

import numpy as np  # 导入 NumPy 库，用于数值计算

from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier  # 导入基础模型和元估计器相关的类和函数
from ..metrics import get_scorer_names  # 导入评分函数名称获取函数
from ..model_selection import check_cv, cross_val_score  # 导入交叉验证相关的函数
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions  # 导入参数验证相关的类
from ..utils._tags import _safe_tags  # 导入安全标签相关函数
from ..utils.metadata_routing import _RoutingNotSupportedMixin  # 导入元数据路由不支持相关的类
from ..utils.validation import check_is_fitted  # 导入检查模型是否已拟合的函数
from ._base import SelectorMixin  # 导入特征选择基类


class SequentialFeatureSelector(
    _RoutingNotSupportedMixin, SelectorMixin, MetaEstimatorMixin, BaseEstimator
):
    """Transformer that performs Sequential Feature Selection.

    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator. In
    the case of unsupervised learning, this Sequential Feature Selector
    looks only at the features (X), not the desired outputs (y).

    Read more in the :ref:`User Guide <sequential_feature_selection>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    n_features_to_select : "auto", int or float, default="auto"
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected while the score
          change does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

        .. versionadded:: 1.1
           The option `"auto"` was added in version 1.1.

        .. versionchanged:: 1.3
           The default changed from `"warn"` to `"auto"` in 1.3.

    tol : float, default=None
        If the score is not incremented by at least `tol` between two
        consecutive feature additions or removals, stop adding or removing.

        `tol` can be negative when removing features using `direction="backward"`.
        It can be useful to reduce the number of features at the cost of a small
        decrease in the score.

        `tol` is enabled only when `n_features_to_select` is `"auto"`.

        .. versionadded:: 1.1

    direction : {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.

    scoring : str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        NOTE that when using a custom scorer, it should return a single
        value.

        If None, the estimator's score method is used.
    """
    """顺序特征选择器，用于贪婪方式添加（前向选择）或移除（后向选择）特征以形成特征子集。在每个阶段，该估计器根据估计器的交叉验证分数选择最佳特征进行添加或移除。对于无监督学习，该顺序特征选择器仅考虑特征（X），而不考虑期望的输出（y）。

    详细内容请参阅：:ref:`用户指南 <sequential_feature_selection>`。

    .. versionadded:: 0.24
    """
    cv : int, cross-validation generator or an iterable, default=None
        确定交叉验证的分割策略。
        可用的 cv 输入有：

        - None，使用默认的 5 折交叉验证，
        - 整数，指定在 `(Stratified)KFold` 中的折数，
        - :term:`CV splitter`，
        - 生成 (train, test) 分割索引数组的可迭代对象。

        对于整数/None 输入，如果估计器是分类器且 ``y`` 是二元或多类别的，
        将使用 :class:`~sklearn.model_selection.StratifiedKFold`。在其他所有情况下，
        将使用 :class:`~sklearn.model_selection.KFold`。这些分割器被实例化时 `shuffle=False`，
        因此每次调用时分割将保持相同。

        参见 :ref:`用户指南 <cross_validation>`，了解可用的各种交叉验证策略。

    n_jobs : int, default=None
        并行运行的作业数。在评估要添加或删除的新特征时，交叉验证过程在折叠之间是并行的。
        ``None`` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为 1。
        ``-1`` 表示使用所有处理器。详见 :term:`术语表 <n_jobs>`。

    Attributes
    ----------
    n_features_in_ : int
        在 :term:`fit` 过程中看到的特征数。仅当基础估计器在 `fit` 时公开此类属性时才定义。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在 :term:`fit` 过程中看到的特征名称。仅当 `X` 的特征名称都是字符串时才定义。

        .. versionadded:: 1.0

    n_features_to_select_ : int
        选择的特征数量。

    support_ : ndarray of shape (n_features,), dtype=bool
        所选特征的掩码。

    See Also
    --------
    GenericUnivariateSelect : 可配置策略的单变量特征选择器。
    RFE : 基于重要性权重的递归特征消除。
    RFECV : 基于重要性权重的递归特征消除，自动选择特征数量。
    SelectFromModel : 基于重要性权重阈值的特征选择。

    Examples
    --------
    >>> from sklearn.feature_selection import SequentialFeatureSelector
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    >>> sfs.fit(X, y)
    SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                              n_features_to_select=3)
    >>> sfs.get_support()
    array([ True, False,  True,  True])
    >>> sfs.transform(X).shape
    (150, 3)
    # 定义参数的约束条件字典，包含了各个参数的类型和取值约束
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],  # estimator 参数应具有 "fit" 方法
        "n_features_to_select": [
            StrOptions({"auto"}),  # n_features_to_select 可以为字符串 "auto"
            Interval(RealNotInt, 0, 1, closed="right"),  # 或者为实数区间 [0, 1)
            Interval(Integral, 0, None, closed="neither"),  # 或者为整数区间 (0, ∞)
        ],
        "tol": [None, Interval(Real, None, None, closed="neither")],  # tol 可以为 None 或者为实数区间 (None, ∞)
        "direction": [StrOptions({"forward", "backward"})],  # direction 可以是字符串 "forward" 或 "backward"
        "scoring": [None, StrOptions(set(get_scorer_names())), callable],  # scoring 可以为 None、已定义的评分器名称集合、或者是可调用对象
        "cv": ["cv_object"],  # cv 通常是交叉验证对象
        "n_jobs": [None, Integral],  # n_jobs 可以为 None 或整数
    }

    # 初始化方法，接受多个参数并将其存储在对象中
    def __init__(
        self,
        estimator,
        *,
        n_features_to_select="auto",
        tol=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
    ):
        self.estimator = estimator  # 将传入的 estimator 参数存储在实例变量中
        self.n_features_to_select = n_features_to_select  # 将传入的 n_features_to_select 参数存储在实例变量中
        self.tol = tol  # 将传入的 tol 参数存储在实例变量中
        self.direction = direction  # 将传入的 direction 参数存储在实例变量中
        self.scoring = scoring  # 将传入的 scoring 参数存储在实例变量中
        self.cv = cv  # 将传入的 cv 参数存储在实例变量中
        self.n_jobs = n_jobs  # 将传入的 n_jobs 参数存储在实例变量中

    # 使用装饰器 _fit_context 进行修饰，可能影响其内部逻辑
    @_fit_context(
        prefer_skip_nested_validation=False  # 设置 prefer_skip_nested_validation 参数为 False
    )
    # 返回在当前状态下，要添加到 current_mask 的最佳新特征及其得分
    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask):
        # candidate_feature_indices 中存储候选特征的索引，这些特征当前未被选择
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        # 遍历候选特征索引，计算每个特征的得分
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            # 如果选择的方向是 "backward"，则取反候选掩码
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            X_new = X[:, candidate_mask]  # 根据候选掩码选择数据集的子集
            # 计算交叉验证的分数，并取其平均值作为当前特征组合的得分
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()
        # 返回具有最高得分的新特征索引及其得分
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        return new_feature_idx, scores[new_feature_idx]

    # 返回支持掩码，即哪些特征被选择了
    def _get_support_mask(self):
        check_is_fitted(self)  # 检查对象是否已经拟合
        return self.support_

    # 返回更多的标签信息
    def _more_tags(self):
        return {
            "allow_nan": _safe_tags(self.estimator, key="allow_nan"),  # 返回允许 NaN 值的标签信息
        }
```