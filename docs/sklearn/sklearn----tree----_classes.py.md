# `D:\src\scipysrc\scikit-learn\sklearn\tree\_classes.py`

```
"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import copy                               # 导入 copy 模块，用于对象的浅复制和深复制操作
import numbers                            # 导入 numbers 模块，用于数值相关的基本数据类型检查
from abc import ABCMeta, abstractmethod   # 从 abc 模块导入 ABCMeta 类和 abstractmethod 装饰器
from math import ceil                    # 导入 ceil 函数，用于向上取整

import numpy as np                       # 导入 NumPy 库，用于数值计算
from scipy.sparse import issparse        # 导入 issparse 函数，用于判断对象是否为稀疏矩阵

from ..base import (                     # 导入自定义模块中的基础组件
    BaseEstimator,                       # 导入基础估计器类
    ClassifierMixin,                     # 导入分类器混合类
    MultiOutputMixin,                    # 导入多输出混合类
    RegressorMixin,                      # 导入回归器混合类
    _fit_context,                        # 导入拟合上下文
    clone,                               # 导入克隆函数，用于对象的深复制
    is_classifier,                       # 导入分类器检查函数
)
from ..utils import Bunch, check_random_state, compute_sample_weight  # 导入自定义工具函数
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions  # 导入参数验证类
from ..utils.multiclass import check_classification_targets  # 导入多类别分类目标检查函数
from ..utils.validation import (         # 导入验证函数
    _assert_all_finite_element_wise,     # 导入逐元素有限性检查函数
    _check_sample_weight,                # 导入样本权重检查函数
    assert_all_finite,                   # 导入全部有限性检查函数
    check_is_fitted,                     # 导入已拟合性检查函数
)
from . import _criterion, _splitter, _tree  # 导入决策树相关模块
from ._criterion import Criterion         # 导入决策树准则类
from ._splitter import Splitter           # 导入决策树分裂器类
from ._tree import (                      # 导入决策树构建和修剪相关函数和类
    BestFirstTreeBuilder,                 # 导入最佳优先树构建器
    DepthFirstTreeBuilder,                # 导入深度优先树构建器
    Tree,                                 # 导入决策树类
    _build_pruned_tree_ccp,               # 导入基于CCP修剪的决策树构建函数
    ccp_pruning_path,                     # 导入CCP修剪路径函数
)
from ._utils import _any_isnan_axis0      # 导入轴0上是否有NaN的函数

__all__ = [                               # 模块公开的所有符号列表
    "DecisionTreeClassifier",             # 决策树分类器
    "DecisionTreeRegressor",              # 决策树回归器
    "ExtraTreeClassifier",                # 额外树分类器
    "ExtraTreeRegressor",                 # 额外树回归器
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE                        # 决策树中使用的数据类型
DOUBLE = _tree.DOUBLE                      # 决策树中使用的双精度数据类型

CRITERIA_CLF = {                           # 分类器的准则字典
    "gini": _criterion.Gini,               # 基尼系数
    "log_loss": _criterion.Entropy,        # 对数损失（交叉熵）
    "entropy": _criterion.Entropy,         # 熵
}
CRITERIA_REG = {                           # 回归器的准则字典
    "squared_error": _criterion.MSE,       # 均方误差
    "friedman_mse": _criterion.FriedmanMSE, # Friedman均方误差
    "absolute_error": _criterion.MAE,      # 绝对误差
    "poisson": _criterion.Poisson,         # 泊松分布
}

DENSE_SPLITTERS = {                        # 密集数据的分裂器字典
    "best": _splitter.BestSplitter,        # 最佳分裂器
    "random": _splitter.RandomSplitter,    # 随机分裂器
}

SPARSE_SPLITTERS = {                       # 稀疏数据的分裂器字典
    "best": _splitter.BestSparseSplitter,  # 最佳稀疏分裂器
    "random": _splitter.RandomSparseSplitter,  # 随机稀疏分裂器
}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    # 定义参数约束字典，用于验证决策树模型的参数是否符合预期类型和范围
    _parameter_constraints: dict = {
        "splitter": [StrOptions({"best", "random"})],  # splitter参数必须是{"best", "random"}中的一个字符串
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],  # max_depth参数为大于等于1的整数或None
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),  # min_samples_split参数为大于等于2的整数
            Interval(RealNotInt, 0.0, 1.0, closed="right"),  # 或者为0到1之间的实数，包括0但不包括1
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),  # min_samples_leaf参数为大于等于1的整数
            Interval(RealNotInt, 0.0, 1.0, closed="neither"),  # 或者为0到1之间的实数，不包括0和1
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],  # min_weight_fraction_leaf参数为0到0.5之间的实数，包括0和0.5
        "max_features": [
            Interval(Integral, 1, None, closed="left"),  # max_features参数为大于等于1的整数
            Interval(RealNotInt, 0.0, 1.0, closed="right"),  # 或者为0到1之间的实数，包括0但不包括1
            StrOptions({"sqrt", "log2"}),  # 或者为"sqrt"或"log2"中的一个字符串
            None,  # 或者为None
        ],
        "random_state": ["random_state"],  # random_state参数为"random_state"字符串
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],  # max_leaf_nodes参数为大于等于2的整数或None
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],  # min_impurity_decrease参数为大于等于0的实数
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],  # ccp_alpha参数为大于等于0的实数
        "monotonic_cst": ["array-like", None],  # monotonic_cst参数为"array-like"字符串或者为None
    }

    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        # 初始化决策树模型的参数
        self.criterion = criterion  # 模型评估标准
        self.splitter = splitter  # 决策树节点分裂策略
        self.max_depth = max_depth  # 决策树的最大深度
        self.min_samples_split = min_samples_split  # 节点分裂所需的最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶子节点所需的最小样本数
        self.min_weight_fraction_leaf = min_weight_fraction_leaf  # 叶子节点样本权重和的最小加权分数
        self.max_features = max_features  # 寻找最佳分割时考虑的特征数量
        self.max_leaf_nodes = max_leaf_nodes  # 树上的最大叶子节点数
        self.random_state = random_state  # 控制随机性的种子值
        self.min_impurity_decrease = min_impurity_decrease  # 停止树增长的最小不纯减少量
        self.class_weight = class_weight  # 类别权重
        self.ccp_alpha = ccp_alpha  # 剪枝参数
        self.monotonic_cst = monotonic_cst  # 约束模型的单调性

    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def _support_missing_values(self, X):
        # 检查是否支持缺失值，并且不是稀疏矩阵，并且允许nan值，并且没有单调性约束
        return (
            not issparse(X)
            and self._get_tags()["allow_nan"]
            and self.monotonic_cst is None
        )
    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        """Return boolean mask denoting if there are missing values for each feature.

        This method also ensures that X is finite.

        Parameter
        ---------
        X : array-like of shape (n_samples, n_features), dtype=DOUBLE
            Input data.

        estimator_name : str or None, default=None
            Name to use when raising an error. Defaults to the class name.

        Returns
        -------
        missing_values_in_feature_mask : ndarray of shape (n_features,), or None
            Missing value mask. If missing values are not supported or there
            are no missing values, return None.
        """
        estimator_name = estimator_name or self.__class__.__name__  # 设置估计器的名称，若未指定则使用类名

        common_kwargs = dict(estimator_name=estimator_name, input_name="X")  # 常用参数字典

        if not self._support_missing_values(X):  # 如果模型不支持缺失值
            assert_all_finite(X, **common_kwargs)  # 确保所有元素都是有限的
            return None  # 返回空值表示没有缺失值

        with np.errstate(over="ignore"):  # 设置忽略溢出警告
            overall_sum = np.sum(X)  # 计算所有元素的总和

        if not np.isfinite(overall_sum):  # 如果总和不是有限数
            # 在存在无限元素时引发 ValueError
            _assert_all_finite_element_wise(X, xp=np, allow_nan=True, **common_kwargs)

        # 如果总和不是 NaN，则表示没有缺失值
        if not np.isnan(overall_sum):
            return None  # 返回空值表示没有缺失值

        missing_values_in_feature_mask = _any_isnan_axis0(X)  # 检查每个特征是否存在缺失值
        return missing_values_in_feature_mask  # 返回缺失值的布尔掩码数组

    def _fit(
        self,
        X,
        y,
        sample_weight=None,
        check_input=True,
        missing_values_in_feature_mask=None,
    ):
        """Fit the model to the training data."""
        # 这是一个 _fit 方法的定义，用于拟合模型到训练数据，具体操作不在此处展开讨论

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:  # 如果需要验证输入数据
            if self._support_missing_values(X):  # 如果模型支持缺失值
                force_all_finite = "allow-nan"  # 允许 NaN
            else:
                force_all_finite = True  # 强制所有元素为有限数
            X = self._validate_data(
                X,
                dtype=DTYPE,
                accept_sparse="csr",
                reset=False,
                force_all_finite=force_all_finite,
            )  # 验证数据的格式和内容是否符合要求

            if issparse(X) and (
                X.indices.dtype != np.intc or X.indptr.dtype != np.intc
            ):  # 如果数据是稀疏矩阵且索引不是 np.intc 类型
                raise ValueError("No support for np.int64 index based sparse matrices")  # 抛出错误，不支持基于 np.int64 的索引稀疏矩阵
        else:
            # 不论 `check_input` 是否为 True，都需要检查特征数量
            self._check_n_features(X, reset=False)  # 检查特征数量是否符合要求

        return X  # 返回验证后的数据
    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        # 检查模型是否已经拟合，即是否已经训练
        check_is_fitted(self)
        # 验证并预处理输入数据 X
        X = self._validate_X_predict(X, check_input)
        # 使用模型的树结构进行预测
        proba = self.tree_.predict(X)
        # 获取样本数量
        n_samples = X.shape[0]

        # 分类模型
        if is_classifier(self):
            # 单输出的情况下，返回预测类别
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)
            # 多输出的情况下，为每个输出类别创建预测结果
            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )
                return predictions

        # 回归模型
        else:
            # 单输出的情况下，返回预测值
            if self.n_outputs_ == 1:
                return proba[:, 0]
            # 多输出的情况下，返回所有输出的预测值
            else:
                return proba[:, :, 0]

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        # 检查模型是否已经拟合，即是否已经训练
        check_is_fitted(self)
        # 验证并预处理输入数据 X
        X = self._validate_X_predict(X, check_input)
        # 返回每个样本预测的叶子节点索引
        return self.tree_.apply(X)
    # 返回决策树中样本通过的节点指示器的稀疏矩阵
    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        # 确保输入数据 X 符合预测要求
        X = self._validate_X_predict(X, check_input)
        # 返回决策树对象的决策路径
        return self.tree_.decision_path(X)

    # 使用最小代价复杂度剪枝来剪枝决策树
    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        # 检查是否已经拟合过模型
        check_is_fitted(self)

        # 如果 ccp_alpha 参数为 0.0，则不进行剪枝
        if self.ccp_alpha == 0.0:
            return

        # 构建剪枝后的树
        # 如果是分类器，则获取类别数
        if is_classifier(self):
            n_classes = np.atleast_1d(self.n_classes_)
            pruned_tree = Tree(self.n_features_in_, n_classes, self.n_outputs_)
        else:
            # 否则创建一个适当大小的数组
            pruned_tree = Tree(
                self.n_features_in_,
                # TODO: the tree shouldn't need this param
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )
        # 使用 Minimal Cost-Complexity Pruning 算法构建剪枝后的树
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        # 将剪枝后的树赋值给当前对象的 tree_ 属性
        self.tree_ = pruned_tree
    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """
        计算在最小成本复杂性修剪期间的修剪路径。

        查看 :ref:`minimal_cost_complexity_pruning` 获取有关修剪过程的详细信息。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            训练输入样本。将被内部转换为 ``dtype=np.float32``，
            如果提供了稀疏矩阵，则转换为稀疏 ``csc_matrix``。

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            目标值（类标签），可以是整数或字符串。

        sample_weight : array-like of shape (n_samples,), default=None
            样本权重。如果为 None，则样本被等权重。在每个节点搜索分割时，
            将忽略创建子节点带有净零或负权重的分割。如果分割导致任何单个类在任一
            子节点中承载负权重，则也将忽略该分割。

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            类似字典的对象，具有以下属性。

            ccp_alphas : ndarray
                修剪期间子树的有效 alpha 值。

            impurities : ndarray
                对应于 ``ccp_alphas`` 中的 alpha 值的子树叶子节点的杂质总和。
        """
        est = clone(self).set_params(ccp_alpha=0.0)
        est.fit(X, y, sample_weight=sample_weight)
        return Bunch(**ccp_pruning_path(est.tree_))

    @property
    def feature_importances_(self):
        """
        返回特征重要性。

        特征重要性计算为该特征带来的准则总减少量的（归一化）值。
        这也称为基尼重要性。

        警告：基于杂质的特征重要性可能对高基数特征（具有许多唯一值）具有误导性。
        可以参考 :func:`sklearn.inspection.permutation_importance` 作为替代方法。

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            每个特征的归一化准则总减少量（基尼重要性）。
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_importances()
# =============================================================================
# Public estimators
# =============================================================================

# 定义决策树分类器类，继承自ClassifierMixin和BaseDecisionTree类
class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
        
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
          
        .. versionchanged:: 0.18
           Added float values for fractions.
           
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
          
        .. versionchanged:: 0.18
           Added float values for fractions.
           
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float or {"sqrt", "log2"}, default=None
        # 决定在寻找最佳分割时考虑的特征数量：

            # 如果是整数，则每次分割考虑 `max_features` 个特征。
            # 如果是浮点数，则 `max_features` 是一个分数，每次分割考虑 `max(1, int(max_features * n_features_in_))` 个特征。
            # 如果是 "sqrt"，则 `max_features=sqrt(n_features)`。
            # 如果是 "log2"，则 `max_features=log2(n_features)`。
            # 如果是 None，则 `max_features=n_features`。

        # 注意：即使需要检查超过 `max_features` 个特征才能找到有效的节点样本分区，寻找分割也不会停止。

    random_state : int, RandomState instance or None, default=None
        # 控制估算器的随机性。每次分割时，特征总是随机排列，即使 `splitter` 设置为 `"best"` 也是如此。
        # 当 `max_features < n_features` 时，算法会在每次分割之前随机选择 `max_features` 个特征，然后在它们中找到最佳分割。
        # 但是，即使 `max_features=n_features`，不同运行中找到的最佳分割可能会有所不同。这种情况发生在多个分割的标准改善相同且需要随机选择一个分割时。
        # 若要在拟合过程中获得确定性行为，请将 `random_state` 固定为整数。
        # 详细信息请参见术语表中的 "random_state"。

    max_leaf_nodes : int, default=None
        # 以最佳优先方式生长一个具有 `max_leaf_nodes` 个叶节点的树。
        # 最佳节点定义为杂质的相对减少。
        # 如果为 None，则叶节点数量不受限制。

    min_impurity_decrease : float, default=0.0
        # 如果此分割导致杂质减少大于或等于此值，则将分割节点。

        # 加权杂质减少方程如下所示：

            # N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        # 其中 `N` 是样本的总数，`N_t` 是当前节点的样本数，`N_t_L` 是左子节点中的样本数，`N_t_R` 是右子节点中的样本数。

        # 如果传递了 `sample_weight`，则所有 `N`，`N_t`，`N_t_R` 和 `N_t_L` 都是加权和。

        # .. versionadded:: 0.19
    # class_weight : dict, list of dict or "balanced", default=None
    #     用于指定类别权重的参数，可以是字典形式 `{类标签: 权重}`，也可以是列表形式的字典集合，按照 y 的列顺序提供。
    #     如果为 None，则假定所有类别权重为一。对于多输出问题，可以提供与 y 列相同顺序的字典列表。
    #
    #     注意，对于多输出（包括多标签）问题，应该为每列的每个类别定义权重。例如，对于四类多标签分类，权重应为
    #     `[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]` 而不是 `[{1:1}, {2:5}, {3:1}, {4:1}]`。
    #
    #     "balanced" 模式根据输入数据中类别频率的倒数自动调整权重，计算方式为 `n_samples / (n_classes * np.bincount(y))`。
    #     对于多输出问题，将对每列 y 的权重进行乘法运算。
    #
    #     注意，如果指定了 sample_weight（通过 fit 方法传递），这些权重将与 sample_weight 相乘。
    #
    ccp_alpha : non-negative float, default=0.0
    #     最小成本复杂性剪枝所使用的复杂性参数。选择小于 `ccp_alpha` 的最大成本复杂性子树。
    #     默认情况下，不执行剪枝。详细信息请参阅 :ref:`minimal_cost_complexity_pruning`。
    #     
    #     .. versionadded:: 0.22
    #
    monotonic_cst : array-like of int of shape (n_features), default=None
    #     指示要强制执行的每个特征的单调性约束。
    #       - 1: 单调增加
    #       - 0: 无约束
    #       - -1: 单调减少
    #
    #     如果 monotonic_cst 为 None，则不应用约束。
    #
    #     不支持以下情况的单调性约束：
    #       - 多类分类（即当 `n_classes > 2` 时），
    #       - 多输出分类（即当 `n_outputs_ > 1` 时），
    #       - 在包含缺失值的数据上训练的分类。
    #
    #     约束适用于正类的概率。
    #
    #     详细信息请参阅 :ref:`User Guide <monotonic_cst_gbdt>`。
    #
    #     .. versionadded:: 1.4
    #
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
    #     类别标签（单输出问题），或者类别标签数组的列表（多输出问题）。
    #
    feature_importances_ : ndarray of shape (n_features,)
    #     基于不纯度的特征重要性。
    #     数值越高，特征越重要。
    #     特征重要性的计算方式为特征带来的（归一化的）准则总减少量。也称为基尼重要性 [4]_。
    #
    #     警告：基于不纯度的特征重要性对于高基数特征（具有许多唯一值）可能具有误导性。请参阅 :func:`sklearn.inspection.permutation_importance` 作为替代方法。
    max_features_ : int
        # 存储推断出的 max_features 的值，即特征的最大数量。

    n_classes_ : int or list of int
        # 单输出问题中的类别数，或者多输出问题中每个输出的类别数的列表。

    n_features_in_ : int
        # 在拟合过程中观察到的特征数量。

        .. versionadded:: 0.24
        # 版本 0.24 中添加的功能说明。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在拟合过程中观察到的特征名称。仅当 `X` 中的特征名称全部为字符串时定义。

        .. versionadded:: 1.0
        # 版本 1.0 中添加的功能说明。

    n_outputs_ : int
        # 在执行 `fit` 时的输出数量。

    tree_ : Tree instance
        # 底层的树对象。请参考 `help(sklearn.tree._tree.Tree)` 获取树对象的属性，
        # 以及 :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        # 获取这些属性的基本用法示例。

    See Also
    --------
    DecisionTreeRegressor : 决策树回归器。

    Notes
    -----
    # 控制树大小的参数的默认值（例如 `max_depth`、`min_samples_leaf` 等）会导致
    # 完全生长且未修剪的树，在某些数据集上可能非常大。为了减少内存消耗，应通过
    # 设置这些参数值来控制树的复杂性和大小。

    # `predict` 方法在 `predict_proba` 的输出上使用 `numpy.argmax` 函数。
    # 这意味着如果最高预测概率相同，则分类器将预测具有最低索引的“类别”。

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        # 继承自 BaseDecisionTree 的参数约束字典的扩展。

        "criterion": [StrOptions({"gini", "entropy", "log_loss"}), Hidden(Criterion)],
        # "criterion" 参数的约束条件，包括可接受的字符串选项和 Criterion 的隐藏类。

        "class_weight": [dict, list, StrOptions({"balanced"}), None],
        # "class_weight" 参数的约束条件，包括字典、列表、字符串选项 {"balanced"} 和 None。
    }
    # 初始化决策树分类器对象，设置各种参数
    def __init__(
        self,
        *,
        criterion="gini",  # 划分标准，默认为基尼系数
        splitter="best",    # 划分策略，默认为最佳划分
        max_depth=None,     # 树的最大深度，默认为不限制
        min_samples_split=2,    # 内部节点再划分所需最小样本数，默认为2
        min_samples_leaf=1,     # 叶节点最少样本数，默认为1
        min_weight_fraction_leaf=0.0,   # 叶节点样本权重最小总和的分数，默认为0.0
        max_features=None,      # 在寻找最佳分割时考虑的特征数或者特征比例，默认为所有特征
        random_state=None,      # 随机数种子，用于随机化划分时的控制
        max_leaf_nodes=None,    # 最大叶节点数，默认为不限制
        min_impurity_decrease=0.0,  # 分割节点的最小不纯度减少量，默认为0.0
        class_weight=None,      # 类别权重设置，默认为无权重
        ccp_alpha=0.0,          # 剪枝参数，默认为0.0，不剪枝
        monotonic_cst=None,     # 单调约束，用于处理特征的单调性约束，默认为无约束
    ):
        # 调用父类的初始化方法，设置决策树分类器的各种参数
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
        )

    # 使用装饰器定义拟合方法，并在拟合前做一些上下文设置
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """

        # 调用父类的_fit方法，进行决策树分类器的拟合
        super()._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        # 返回拟合后的分类器实例
        return self
    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        # Validate and preprocess input data X
        X = self._validate_X_predict(X, check_input)
        # Use the decision tree to predict probabilities
        proba = self.tree_.predict(X)

        if self.n_outputs_ == 1:
            # For single output, return probabilities directly
            return proba[:, : self.n_classes_]
        else:
            # For multiple outputs, prepare probabilities for each output
            all_proba = []
            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                all_proba.append(proba_k)
            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Predict class probabilities using predict_proba method
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            # For single output, return logarithm of probabilities
            return np.log(proba)
        else:
            # For multiple outputs, return logarithm of probabilities for each output
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def _more_tags(self):
        # XXX: nan is only support for dense arrays, but we set this for common test to
        # pass, specifically: check_estimators_nan_inf
        # Determine if NaN values are allowed based on splitter and criterion
        allow_nan = self.splitter == "best" and self.criterion in {
            "gini",
            "log_loss",
            "entropy",
        }
        return {"multilabel": True, "allow_nan": allow_nan}
# 定义一个决策树回归器类 DecisionTreeRegressor，继承自 RegressorMixin 和 BaseDecisionTree
class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float or {"sqrt", "log2"}, default=None
        # 决定最佳分裂时考虑的特征数量：

        - If int, then consider `max_features` features at each split.
        # 如果是整数，则在每次分裂时考虑 `max_features` 个特征。

        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        # 如果是浮点数，则 `max_features` 是一个分数，每次分裂时考虑 `max(1, int(max_features * n_features_in_))` 个特征。

        - If "sqrt", then `max_features=sqrt(n_features)`.
        # 如果是 "sqrt"，则 `max_features=sqrt(n_features)`。

        - If "log2", then `max_features=log2(n_features)`.
        # 如果是 "log2"，则 `max_features=log2(n_features)`。

        - If None, then `max_features=n_features`.
        # 如果是 None，则 `max_features=n_features`。

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        # 注意：寻找分裂点并不会在找到至少一个有效的节点样本分区之前停止，即使这需要检查超过 ``max_features`` 个特征。

    random_state : int, RandomState instance or None, default=None
        # 控制估计器的随机性。特征总是在每次分裂时随机排列，即使 ``splitter`` 设置为 ``"best"`` 也是如此。
        当 ``max_features < n_features`` 时，算法会在每次分裂前随机选择 ``max_features`` 个特征，然后在它们中找到最佳分裂点。
        但是，即使 ``max_features=n_features``，不同运行时找到的最佳分裂点也可能不同。
        这种情况发生在，如果几个分裂点的准则改善相同，并且需要随机选择一个分裂点时。
        在拟合过程中要获得确定性行为，需要将 ``random_state`` 固定为一个整数。
        参见 :term:`Glossary <random_state>` 获取详细信息。

    max_leaf_nodes : int, default=None
        # 以最佳优先方式增长一个具有 ``max_leaf_nodes`` 的树。
        最佳节点定义为杂质的相对减少。
        如果为 None，则叶节点数量不受限制。

    min_impurity_decrease : float, default=0.0
        # 如果此分裂导致杂质减少大于或等于此值，则节点将被分裂。

        加权杂质减少方程如下::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        其中 ``N`` 是总样本数，``N_t`` 是当前节点的样本数，``N_t_L`` 是左子节点的样本数，``N_t_R`` 是右子节点的样本数。

        如果传递了 ``sample_weight``，则所有的 ``N``、``N_t``、``N_t_R`` 和 ``N_t_L`` 都是加权总和。

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        # 用于最小成本复杂度剪枝的复杂度参数。选择成本复杂度大于 ``ccp_alpha`` 的子树。默认情况下不执行剪枝。
        参见 :ref:`minimal_cost_complexity_pruning` 获取详细信息。

        .. versionadded:: 0.22
    # 定义一个数组，用于指定每个特征的单调性约束
    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase   # 1 表示单调增加
          - 0: no constraint        # 0 表示无约束
          - -1: monotonic decrease  # -1 表示单调减少

        If monotonic_cst is None, no constraints are applied.
        如果 monotonic_cst 为 None，则不应用约束。

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),  # 不支持多输出回归
          - regressions trained on data with missing values.       # 在有缺失值的数据上训练的回归模型不支持单调性约束

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4  # 添加版本说明

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        特征重要性，值越高表示特征越重要。
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.
        特征的重要性是通过特征带来的标准化总减少量来计算的，也称为基尼重要性。

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        警告：基于不纯度的特征重要性可能对高基数特征（具有许多唯一值）产生误导性。建议查看 :func:`sklearn.inspection.permutation_importance` 作为替代方法。

    max_features_ : int
        The inferred value of max_features.
        推断得出的 max_features 的值。

    n_features_in_ : int
        Number of features seen during :term:`fit`.
        在拟合过程中看到的特征数量。

        .. versionadded:: 0.24  # 添加版本说明

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        在拟合过程中看到的特征名称数组。仅在 `X` 具有所有字符串特征名称时定义。

        .. versionadded:: 1.0  # 添加版本说明

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
        在执行 `fit` 过程时的输出数量。

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.
        底层的树对象。有关树对象的属性，请参阅 ``help(sklearn.tree._tree.Tree)``，以及 :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py` 中对这些属性的基本用法。

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    控制树大小的参数的默认值（例如 `max_depth`，`min_samples_leaf` 等）会导致完全生长且未剪枝的树，在某些数据集上可能非常大。为了减少内存消耗，应通过设置这些参数值来控制树的复杂性和大小。

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    # 导入决策树回归器模型
    from sklearn.tree import DecisionTreeRegressor
    # 加载糖尿病数据集，并返回特征 X 和目标 y
    X, y = load_diabetes(return_X_y=True)
    # 创建一个决策树回归器对象，设置随机种子为 0
    regressor = DecisionTreeRegressor(random_state=0)
    # 对决策树回归器进行交叉验证，返回十折交叉验证的得分数组
    cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    """
    # 数组中包含的是十折交叉验证的分数结果，每个分数表示每一折的模型性能
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    # 决策树回归器参数的约束条件
    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [
            StrOptions({"squared_error", "friedman_mse", "absolute_error", "poisson"}),
            Hidden(Criterion),
        ],
    }

    # 决策树回归器的初始化方法
    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        # 调用父类 BaseDecisionTree 的初始化方法，设置决策树回归器的参数
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )

    # 决策树回归器的拟合方法装饰器，用于进行拟合过程的上下文管理
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """

        # 调用父类的 _fit 方法，对训练集 (X, y) 进行拟合
        super()._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        # 返回拟合后的决策树回归器对象
        return self
    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features), dtype=np.float32
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features), dtype=np.intp
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,), dtype=np.float64
            The value of the partial dependence function on each grid point.
        """
        # 将输入的网格点转换为指定数据类型的多维数组
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        # 创建一个用于存储平均预测结果的数组，初始值为零
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )
        # 将目标特征转换为指定数据类型的一维数组
        target_features = np.asarray(target_features, dtype=np.intp, order="C")

        # 调用决策树对象的方法计算部分依赖值
        self.tree_.compute_partial_dependence(
            grid, target_features, averaged_predictions
        )
        # 返回计算得到的部分依赖值数组
        return averaged_predictions

    def _more_tags(self):
        # XXX: nan is only support for dense arrays, but we set this for common test to
        # pass, specifically: check_estimators_nan_inf
        # 根据条件设置是否允许 NaN 值，主要用于测试中的通用情况检查
        allow_nan = self.splitter == "best" and self.criterion in {
            "squared_error",
            "friedman_mse",
            "poisson",
        }
        # 返回包含 allow_nan 属性的字典
        return {"allow_nan": allow_nan}
# 定义一个基于决策树的极端随机森林分类器类 ExtraTreeClassifier
class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
    
    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
    
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    
        .. versionchanged:: 0.18
           Added float values for fractions.
    
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
    
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    
        .. versionchanged:: 0.18
           Added float values for fractions.
    
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    """
    
    # 初始化方法，设置分类器的参数
    def __init__(self, criterion="gini", splitter="random",
                 max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0):
        # 调用父类 DecisionTreeClassifier 的初始化方法，传递参数
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf
        )
    # max_features 参数可以是整数、浮点数、字符串或 None，默认为 "sqrt"
    # 控制每次寻找最佳分割时考虑的特征数：
    # - 如果是整数，则每次考虑 max_features 个特征。
    # - 如果是浮点数，则 max_features 是一个分数，考虑 max(1, int(max_features * n_features_in_)) 个特征。
    # - 如果是 "sqrt"，则 max_features=sqrt(n_features)。
    # - 如果是 "log2"，则 max_features=log2(n_features)。
    # - 如果是 None，则 max_features=n_features。
    # 注意：即使需要检查超过 max_features 个特征才能找到有效的节点样本分区，寻找分割也不会停止。
    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        
    # random_state 参数用于随机选择每次分割时使用的 max_features。
    # 详见术语表中的 "随机状态"。
    random_state : int, RandomState instance or None, default=None

    # max_leaf_nodes 参数用于以最佳优先方式增长树。
    # 最佳节点定义为杂质的相对减少。
    # 如果为 None，则叶节点数量不受限制。
    max_leaf_nodes : int, default=None

    # min_impurity_decrease 参数表示如果这次分裂引起杂质减少大于或等于此值，则将节点分裂。
    # 加权杂质减少的计算公式如下：
    # N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
    # 其中，N 是总样本数，N_t 是当前节点的样本数，N_t_L 是左子节点的样本数，N_t_R 是右子节点的样本数。
    # 如果传递了 sample_weight，则所有数量都是加权总和。
    # 详见 0.19 版本新增内容。
    min_impurity_decrease : float, default=0.0
    class_weight : dict, list of dict or "balanced", default=None
        # 定义类别权重，可以是字典形式 `{类标签: 权重}`，或者是字符串 "balanced"。
        # 如果为 None，则所有类别权重默认为1。对于多输出问题，可以提供一个与 y 列相同顺序的字典列表。
        # 对于多输出问题（包括多标签问题），每列的类别权重应该在其自己的字典中定义。
        # 例如，对于四类多标签分类，权重应该是 [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]，
        # 而不是 [{1:1}, {2:5}, {3:1}, {4:1}]。
        # "balanced" 模式根据输入数据中每个类别的频率自动调整权重，与类别频率成反比。
        # 公式为 `n_samples / (n_classes * np.bincount(y))`。
        # 对于多输出问题，将会对每列 y 的权重进行乘法运算。
        # 注意，如果指定了 sample_weight（通过 fit 方法传递），这些权重将与 sample_weight 相乘。

    ccp_alpha : non-negative float, default=0.0
        # 用于最小代价复杂度剪枝的复杂度参数。选择比 `ccp_alpha` 小的最大成本复杂度子树。
        # 默认情况下，不执行剪枝。详细信息请参阅 :ref:`minimal_cost_complexity_pruning`。

        .. versionadded:: 0.22
        # 添加版本说明：从版本0.22开始添加了此参数。

    monotonic_cst : array-like of int of shape (n_features), default=None
        # 指示在每个特征上强制执行的单调性约束。
        # - 1：单调增加
        # - 0：无约束
        # - -1：单调减少
        # 如果 monotonic_cst 为 None，则不应用约束。
        # 单调性约束不支持以下情况：
        # - 多类别分类（即 `n_classes > 2`），
        # - 多输出分类（即 `n_outputs_ > 1`），
        # - 在具有缺失值的数据上训练的分类问题。
        # 约束适用于正类的概率。

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
        # 详细信息请阅读用户指南中的 :ref:`monotonic_cst_gbdt` 部分。

        .. versionadded:: 1.4
        # 添加版本说明：从版本1.4开始添加了此参数。

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        # 类别标签（单输出问题）的数组，
        # 或者是包含每个输出类别数的数组列表（多输出问题）。

    max_features_ : int
        # 推断得出的 max_features 的值。

    n_classes_ : int or list of int
        # 类别数量（单输出问题）的整数，
        # 或者是包含每个输出类别数量的整数列表（多输出问题）。
    feature_importances_ : ndarray of shape (n_features,)
        特征重要性数组，形状为 (n_features,)，基于不纯度计算得出。
        数值越高表示特征越重要，特征重要性的计算是通过特征带来的准则的总减少量（归一化）。
        也称为基尼重要性。

        警告：基于不纯度的特征重要性对于高基数特征（具有许多唯一值）可能具有误导性。
        可以参考 :func:`sklearn.inspection.permutation_importance` 进行替代计算。

    n_features_in_ : int
        在拟合过程中观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在拟合过程中观察到的特征名称数组。仅在 `X` 具有全部为字符串的特征名称时定义。

        .. versionadded:: 1.0

    n_outputs_ : int
        执行 `fit` 过程时的输出数量。

    tree_ : Tree instance
        底层的树对象。请参阅 ``help(sklearn.tree._tree.Tree)`` 获取树对象的属性，
        以及 :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        了解这些属性的基本用法。

    See Also
    --------
    ExtraTreeRegressor : 极端随机树回归器。
    sklearn.ensemble.ExtraTreesClassifier : 极端随机树分类器。
    sklearn.ensemble.ExtraTreesRegressor : 极端随机树回归器。
    sklearn.ensemble.RandomForestClassifier : 随机森林分类器。
    sklearn.ensemble.RandomForestRegressor : 随机森林回归器。
    sklearn.ensemble.RandomTreesEmbedding : 完全随机树的集成方法。

    Notes
    -----
    控制树的大小的参数的默认值（如 ``max_depth``、``min_samples_leaf`` 等）
    会导致完全生长且未剪枝的树，这在某些数据集上可能会非常大。为了减少内存消耗，
    应通过设置这些参数值来控制树的复杂度和大小。

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.tree import ExtraTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, random_state=0)
    >>> extra_tree = ExtraTreeClassifier(random_state=0)
    >>> cls = BaggingClassifier(extra_tree, random_state=0).fit(
    ...    X_train, y_train)
    >>> cls.score(X_test, y_test)
    0.8947...
    # 初始化方法，用于初始化决策树分类器的各种参数
    def __init__(
        self,
        *,
        criterion="gini",  # 划分标准，默认为基尼系数
        splitter="random",  # 划分策略，默认为随机划分
        max_depth=None,  # 树的最大深度，默认为无限制
        min_samples_split=2,  # 内部节点再划分所需最小样本数，默认为2
        min_samples_leaf=1,  # 叶子节点最少样本数，默认为1
        min_weight_fraction_leaf=0.0,  # 叶子节点最小加权分数，默认为0.0
        max_features="sqrt",  # 搜索划分时考虑的最大特征数，默认为"sqrt"
        random_state=None,  # 随机数种子，控制随机性，默认为None
        max_leaf_nodes=None,  # 最大叶子节点数，默认为无限制
        min_impurity_decrease=0.0,  # 分裂节点的不纯度减少阈值，默认为0.0
        class_weight=None,  # 类别权重，默认为None
        ccp_alpha=0.0,  # 剪枝参数，默认为0.0
        monotonic_cst=None,  # 约束单调性，默认为None
    ):
        # 调用父类的初始化方法，传入各个参数
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
# 继承自 DecisionTreeRegressor 的极端随机树回归器
class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.
    # min_samples_leaf : int or float, default=1
    #     叶节点所需的最小样本数。
    #     在任何深度上考虑分割点时，只有当左右分支中至少有 `min_samples_leaf` 训练样本时才会考虑。
    #     这可能会使模型平滑化，特别是在回归问题中。
    # 
    #     - 如果是 int，则将 `min_samples_leaf` 视为最小数量。
    #     - 如果是 float，则 `min_samples_leaf` 是一个分数，需要至少 `ceil(min_samples_leaf * n_samples)` 个样本。
    # 
    #     .. versionchanged:: 0.18
    #        添加了分数值的支持。
    
    # min_weight_fraction_leaf : float, default=0.0
    #     叶节点所需的加权样本总权重的最小分数。当未提供 sample_weight 时，样本权重相等。
    
    # max_features : int, float, {"sqrt", "log2"} or None, default=1.0
    #     在寻找最佳分割时考虑的特征数：
    # 
    #     - 如果是 int，则在每次分割时考虑 `max_features` 个特征。
    #     - 如果是 float，则 `max_features` 是一个分数，考虑每次分割时最多 `max(1, int(max_features * n_features_in_))` 个特征。
    #     - 如果是 "sqrt"，则 `max_features=sqrt(n_features)`。
    #     - 如果是 "log2"，则 `max_features=log2(n_features)`。
    #     - 如果是 None，则 `max_features=n_features`。
    # 
    #     .. versionchanged:: 1.1
    #         默认的 `max_features` 从 `"auto"` 更改为 `1.0`。
    # 
    #     注意：即使需要实际检查超过 `max_features` 个特征，寻找分割也不会在未找到有效的节点样本分区之前停止。
    
    # random_state : int, RandomState instance or None, default=None
    #     用于在每次分割时随机选择使用的 `max_features`。
    #     有关详细信息，请参见 :term:`Glossary <random_state>`。
    
    # min_impurity_decrease : float, default=0.0
    #     如果分割导致的不纯度减少大于或等于此值，则将节点分割。
    # 
    #     加权不纯度减少方程如下：
    # 
    #         N_t / N * (impurity - N_t_R / N_t * right_impurity
    #                             - N_t_L / N_t * left_impurity)
    # 
    #     其中 ``N`` 是总样本数，``N_t`` 是当前节点的样本数，``N_t_L`` 是左子节点中的样本数，``N_t_R`` 是右子节点中的样本数。
    #     如果传递了 `sample_weight`，则所有这些都是加权和。
    # 
    #     .. versionadded:: 0.19
    
    # max_leaf_nodes : int, default=None
    #     以最佳优先方式增长具有 `max_leaf_nodes` 个叶节点的树。
    #     最佳节点定义为不纯度的相对减少。
    #     如果为 None，则叶节点数量不受限制。
    ccp_alpha : non-negative float, default=0.0
        Minimal Cost-Complexity Pruning的复杂度参数。选择小于“ccp_alpha”的最大成本复杂度的子树。默认情况下不执行修剪。详细信息请参见:ref:`minimal_cost_complexity_pruning`。

        .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        指示对每个特征施加的单调性约束。
          - 1: 单调增加
          - 0: 无约束
          - -1: 单调减少

        如果monotonic_cst为None，则不应用约束。

        不支持以下情况的单调性约束:
          - 多输出回归（即`n_outputs_ > 1`），
          - 在具有缺失值的数据上训练的回归。

        详细信息请参阅:ref:`User Guide <monotonic_cst_gbdt>`。

        .. versionadded:: 1.4

    Attributes
    ----------
    max_features_ : int
        推断出的max_features的值。

    n_features_in_ : int
        在“fit”期间看到的特征数。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        在“fit”期间看到的特征名称。仅当`X`具有所有字符串类型的特征名称时定义。

        .. versionadded:: 1.0

    feature_importances_ : ndarray of shape (n_features,)
        基于杂质的特征重要性（值越高，特征越重要）。

        警告：基于杂质的特征重要性对于高基数特征（具有许多唯一值）可能具有误导性。作为替代，请参阅:func:`sklearn.inspection.permutation_importance`。

    n_outputs_ : int
        执行“fit”时的输出数量。

    tree_ : Tree instance
        底层的Tree对象。请参阅“help(sklearn.tree._tree.Tree)”了解Tree对象的属性，以及:ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        了解这些属性的基本用法。

    See Also
    --------
    ExtraTreeClassifier : 极其随机的树分类器。
    sklearn.ensemble.ExtraTreesClassifier : 额外的树分类器。
    sklearn.ensemble.ExtraTreesRegressor : 额外的树回归器。

    Notes
    -----
    控制树大小的参数的默认值（例如“max_depth”，“min_samples_leaf”等）会导致完全生长且未修剪的树，这可能在某些数据集上非常大。为了减少内存消耗，应通过设置这些参数值来控制树的复杂性和大小。

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    # 导入 train_test_split 函数用于数据集划分
    >>> from sklearn.model_selection import train_test_split
    # 导入 BaggingRegressor 用于集成学习
    >>> from sklearn.ensemble import BaggingRegressor
    # 导入 ExtraTreeRegressor 用于构建额外树回归模型
    >>> from sklearn.tree import ExtraTreeRegressor
    # 加载糖尿病数据集，返回特征 X 和目标 y
    >>> X, y = load_diabetes(return_X_y=True)
    # 将数据集划分为训练集和测试集，使用随机种子 0
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    # 创建额外树回归模型对象，使用随机种子 0
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    # 创建 BaggingRegressor 对象，使用额外树回归模型作为基础估计器，随机种子 0
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    # 在测试集上评估 BaggingRegressor 模型的性能得分
    >>> reg.score(X_test, y_test)
    0.33...
    """

    # 初始化函数，继承父类的参数，并设置默认值
    def __init__(
        self,
        *,
        criterion="squared_error",                      # 划分准则，默认为平方误差
        splitter="random",                              # 划分策略，默认为随机
        max_depth=None,                                 # 树的最大深度，默认为无限制
        min_samples_split=2,                            # 分裂内部节点所需的最小样本数，默认为2
        min_samples_leaf=1,                             # 叶子节点所需的最小样本数，默认为1
        min_weight_fraction_leaf=0.0,                   # 叶子节点的最小权重，可用于加权实例，默认为0.0
        max_features=1.0,                               # 在寻找最佳分割时考虑的特征数，默认为1.0（所有特征）
        random_state=None,                              # 随机数种子，控制每次分裂的随机性，默认为None
        min_impurity_decrease=0.0,                      # 如果分裂导致杂质减少大于或等于此值，则分裂节点，默认为0.0
        max_leaf_nodes=None,                            # 叶子节点的最大数量，默认为无限制
        ccp_alpha=0.0,                                  # 最小成本复杂度剪枝的复杂度参数，默认为0.0
        monotonic_cst=None,                             # 每个特征的单调性约束，默认为None
    ):
        # 调用父类的初始化方法，设置决策树回归器的参数
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
```