# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\gradient_boosting.py`

```
"""Fast Gradient Boosting decision trees for classification and regression."""

# Author: Nicolas Hug

# 导入所需的库和模块
import itertools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
from numbers import Integral, Real
from time import time

# 导入 NumPy 库，并重命名为 np
import numpy as np

# 导入损失函数相关模块
from ..._loss.loss import (
    _LOSSES,
    BaseLoss,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    PinballLoss,
)

# 导入基础模块和类
from ...base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
)

# 导入数据处理相关模块和函数
from ...compose import ColumnTransformer
from ...metrics import check_scoring
from ...metrics._scorer import _SCORERS
from ...model_selection import train_test_split
from ...preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from ...utils import check_random_state, compute_sample_weight, resample
from ...utils._missing import is_scalar_nan
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ...utils.multiclass import check_classification_targets
from ...utils.validation import (
    _check_monotonic_cst,
    _check_sample_weight,
    _check_y,
    _is_pandas_df,
    check_array,
    check_consistent_length,
    check_is_fitted,
)

# 导入梯度提升相关模块和函数
from ._gradient_boosting import _update_raw_predictions
from .binning import _BinMapper
from .common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from .grower import TreeGrower

# 复制损失函数字典并更新其中的部分损失函数
_LOSSES = _LOSSES.copy()
_LOSSES.update(
    {
        "poisson": HalfPoissonLoss,
        "gamma": HalfGammaLoss,
        "quantile": PinballLoss,
    }
)


def _update_leaves_values(loss, grower, y_true, raw_prediction, sample_weight):
    """Update the leaf values to be predicted by the tree.

    Update equals:
        loss.fit_intercept_only(y_true - raw_prediction)

    This is only applied if loss.differentiable is False.
    Note: It only works, if the loss is a function of the residual, as is the
    case for AbsoluteError and PinballLoss. Otherwise, one would need to get
    the minimum of loss(y_true, raw_prediction + x) in x. A few examples:
      - AbsoluteError: median(y_true - raw_prediction).
      - PinballLoss: quantile(y_true - raw_prediction).

    More background:
    For the standard gradient descent method according to "Greedy Function
    Approximation: A Gradient Boosting Machine" by Friedman, all loss functions but the
    squared loss need a line search step. BaseHistGradientBoosting, however, implements
    a so called Newton boosting where the trees are fitted to a 2nd order
    approximations of the loss in terms of gradients and hessians. In this case, the
    line search step is only necessary if the loss is not smooth, i.e. not
    differentiable, which renders the 2nd order approximation invalid. In fact,
    non-smooth losses arbitrarily set hessians to 1 and effectively use the standard
    gradient descent method with line search.
    """
    # 循环遍历最终的叶子节点列表
    for leaf in grower.finalized_leaves:
        # 获取当前叶子节点的样本索引
        indices = leaf.sample_indices
        # 如果未提供样本权重，则设置为None
        if sample_weight is None:
            sw = None
        else:
            # 否则，从样本权重中获取当前叶子节点的权重
            sw = sample_weight[indices]
        # 计算更新量，使用损失函数的fit_intercept_only方法
        update = loss.fit_intercept_only(
            y_true=y_true[indices] - raw_prediction[indices],
            sample_weight=sw,
        )
        # 更新当前叶子节点的值，乘以生长器的收缩率
        leaf.value = grower.shrinkage * update
        # 注意，这里忽略了正则化过程
# 定义一个上下文管理器，用于临时替换 `_raw_predict` 方法返回预测的原始值
@contextmanager
def _patch_raw_predict(estimator, raw_predictions):
    """Context manager that patches _raw_predict to return raw_predictions.

    `raw_predictions` is typically a precomputed array to avoid redundant
    state-wise computations fitting with early stopping enabled: in this case
    `raw_predictions` is incrementally updated whenever we add a tree to the
    boosted ensemble.

    Note: this makes fitting HistGradientBoosting* models inherently non thread
    safe at fit time. However thread-safety at fit time was never guaranteed nor
    enforced for scikit-learn estimators in general.

    Thread-safety at prediction/transform time is another matter as those
    operations are typically side-effect free and therefore often thread-safe by
    default for most scikit-learn models and would like to keep it that way.
    Therefore this context manager should only be used at fit time.

    TODO: in the future, we could explore the possibility to extend the scorer
    public API to expose a way to compute vales from raw predictions. That would
    probably require also making the scorer aware of the inverse link function
    used by the estimator which is typically private API for now, hence the need
    for this patching mechanism.
    """
    # 保存原始的 `_raw_predict` 方法
    orig_raw_predict = estimator._raw_predict

    # 定义一个新的 `_raw_predict` 方法，始终返回预先计算好的 `raw_predictions`
    def _patched_raw_predicts(*args, **kwargs):
        return raw_predictions

    # 替换 estimator 的 `_raw_predict` 方法为新定义的 `_patched_raw_predicts`
    estimator._raw_predict = _patched_raw_predicts
    
    # 执行上下文管理器代码块
    yield estimator
    
    # 恢复 estimator 的 `_raw_predict` 方法为原始的 `_raw_predict`
    estimator._raw_predict = orig_raw_predict


# 定义一个抽象基类，用于基于直方图的梯度提升估算器
class BaseHistGradientBoosting(BaseEstimator, ABC):
    """Base class for histogram-based gradient boosting estimators."""
    # 定义参数约束字典，指定每个参数的类型和取值范围
    _parameter_constraints: dict = {
        "loss": [BaseLoss],  # 损失函数类型约束为BaseLoss类
        "learning_rate": [Interval(Real, 0, None, closed="neither")],  # 学习率约束为大于0的实数
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # 最大迭代次数约束为大于等于1的整数
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],  # 最大叶子节点数约束为大于等于2的整数或None
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],  # 最大深度约束为大于等于1的整数或None
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],  # 最小叶子节点样本数约束为大于等于1的整数
        "l2_regularization": [Interval(Real, 0, None, closed="left")],  # L2正则化约束为大于等于0的实数
        "max_features": [Interval(RealNotInt, 0, 1, closed="right")],  # 最大特征数约束为0到1之间的实数
        "monotonic_cst": ["array-like", dict, None],  # 单调性约束为数组、字典或None
        "interaction_cst": [  # 交互约束为列表、元组、指定字符串集合或None
            list,
            tuple,
            StrOptions({"pairwise", "no_interactions"}),
            None,
        ],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],  # 迭代不变次数约束为大于等于1的整数
        "validation_fraction": [  # 验证集比例约束为0到1之间的实数或大于等于1的整数或None
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "tol": [Interval(Real, 0, None, closed="left")],  # 容忍度约束为大于等于0的实数
        "max_bins": [Interval(Integral, 2, 255, closed="both")],  # 最大箱数约束为2到255之间的整数
        "categorical_features": [  # 类别特征约束为数组、指定字符串集合或None
            "array-like",
            StrOptions({"from_dtype"}),
            Hidden(StrOptions({"warn"})),
            None,
        ],
        "warm_start": ["boolean"],  # 是否热启动约束为布尔值
        "early_stopping": [StrOptions({"auto"}), "boolean"],  # 提前停止约束为指定字符串或布尔值
        "scoring": [str, callable, None],  # 评分函数约束为字符串、可调用对象或None
        "verbose": ["verbose"],  # 详细程度约束为指定字符串
        "random_state": ["random_state"],  # 随机种子约束为指定字符串
    }

    @abstractmethod
    def __init__(
        self,
        loss,
        *,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        l2_regularization,
        max_features,
        max_bins,
        categorical_features,
        monotonic_cst,
        interaction_cst,
        warm_start,
        early_stopping,
        scoring,
        validation_fraction,
        n_iter_no_change,
        tol,
        verbose,
        random_state,
    ):
        # 初始化模型参数
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.categorical_features = categorical_features
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
    # 验证传递给 __init__ 的参数的有效性
    def _validate_parameters(self):
        """Validate parameters passed to __init__.

        The parameters that are directly passed to the grower are checked in
        TreeGrower."""
        
        # 如果 monotonic_cst 不为 None，并且 n_trees_per_iteration_ 不等于 1，则抛出数值错误
        if self.monotonic_cst is not None and self.n_trees_per_iteration_ != 1:
            raise ValueError(
                "monotonic constraints are not supported for multiclass classification."
            )

    # 最终化样本权重
    def _finalize_sample_weight(self, sample_weight, y):
        """Finalize sample weight.

        Used by subclasses to adjust sample_weights. This is useful for implementing
        class weights.
        """
        
        # 返回样本权重 sample_weight，用于调整类权重的子类使用
        return sample_weight
    # 对输入数据 X 进行预处理和验证
    def _preprocess_X(self, X, *, reset):
        """Preprocess and validate X.

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        reset : bool
            Whether to reset the `n_features_in_` and `feature_names_in_` attributes.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        known_categories : list of ndarray of shape (n_categories,)
            List of known categories for each categorical feature.
        """
        # 如果存在预处理器，则让预处理器处理验证
        # 否则，我们自行验证数据
        check_X_kwargs = dict(dtype=[X_DTYPE], force_all_finite=False)
        if not reset:
            if self._preprocessor is None:
                # 返回验证后的数据
                return self._validate_data(X, reset=False, **check_X_kwargs)
            # 使用预处理器对数据进行转换
            return self._preprocessor.transform(X)

        # 到此处，reset 为 False，即在 `fit` 运行期间
        # 检查数据中的分类特征
        self.is_categorical_ = self._check_categorical_features(X)

        if self.is_categorical_ is None:
            self._preprocessor = None
            self._is_categorical_remapped = None

            # 验证数据并返回
            X = self._validate_data(X, **check_X_kwargs)
            return X, None

        # 获取特征数量
        n_features = X.shape[1]
        # 创建 OrdinalEncoder 对象
        ordinal_encoder = OrdinalEncoder(
            categories="auto",
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            dtype=X_DTYPE,
        )

        # 设置验证函数的参数
        check_X = partial(check_array, **check_X_kwargs)
        numerical_preprocessor = FunctionTransformer(check_X)
        # 创建 ColumnTransformer 对象，处理分类和数值特征
        self._preprocessor = ColumnTransformer(
            [
                ("encoder", ordinal_encoder, self.is_categorical_),
                ("numerical", numerical_preprocessor, ~self.is_categorical_),
            ]
        )
        self._preprocessor.set_output(transform="default")
        # 使用 ColumnTransformer 对数据进行拟合和转换
        X = self._preprocessor.fit_transform(X)
        
        # 检查 OrdinalEncoder 找到的类别并获取其编码值
        known_categories = self._check_categories()
        self.n_features_in_ = self._preprocessor.n_features_in_
        # 尝试获取特征名称
        with suppress(AttributeError):
            self.feature_names_in_ = self._preprocessor.feature_names_in_

        # ColumnTransformer 输出将分类特征放在前面
        categorical_remapped = np.zeros(n_features, dtype=bool)
        categorical_remapped[self._preprocessor.output_indices_["encoder"]] = True
        self._is_categorical_remapped = categorical_remapped

        # 返回处理后的数据 X 和已知的分类特征列表
        return X, known_categories
    # 检查预处理器找到的类别，并返回它们的编码值。
    #
    # 返回长度为 `self.n_features_in_` 的列表，每个输入特征对应一个条目。
    #
    # 对于非分类特征，相应的条目是 `None`。
    #
    # 对于分类特征，相应的条目是一个数组，其中包含由预处理器（即 `OrdinalEncoder`）编码的类别，不包括缺失值。
    # 条目因此为 `np.arange(n_categories)`，其中 `n_categories` 是考虑特征列后剩余的唯一值数量，去除缺失值后计算。
    #
    # 如果对于任何特征，`n_categories > self.max_bins`，则会引发 `ValueError`。
    def _check_categories(self):
        # 获取预处理器中的编码器（encoder）
        encoder = self._preprocessor.named_transformers_["encoder"]
        # 初始化已知类别列表，每个特征初始化为 `None`
        known_categories = [None] * self._preprocessor.n_features_in_
        # 获取所有分类特征的列索引
        categorical_column_indices = np.arange(self._preprocessor.n_features_in_)[
            self._preprocessor.output_indices_["encoder"]
        ]
        # 遍历分类特征的索引和其对应的类别
        for feature_idx, categories in zip(
            categorical_column_indices, encoder.categories_
        ):
            # 如果最后一个类别是 np.nan（表示训练数据中存在缺失值），则移除它
            if len(categories) and is_scalar_nan(categories[-1]):
                categories = categories[:-1]
            # 如果类别数量超过了设定的最大分箱数 `self.max_bins`
            if categories.size > self.max_bins:
                try:
                    # 尝试获取特征名，用于错误消息
                    feature_name = repr(encoder.feature_names_in_[feature_idx])
                except AttributeError:
                    feature_name = f"at index {feature_idx}"
                # 抛出值错误，说明该分类特征的基数超过了允许的最大分箱数
                raise ValueError(
                    f"Categorical feature {feature_name} is expected to "
                    f"have a cardinality <= {self.max_bins} but actually "
                    f"has a cardinality of {categories.size}."
                )
            # 将已知类别列表中的当前特征索引位置设置为该特征的编码数组
            known_categories[feature_idx] = np.arange(len(categories), dtype=X_DTYPE)
        # 返回已知类别列表
        return known_categories
    # 检查和验证交互约束条件
    def _check_interaction_cst(self, n_features):
        """Check and validation for interaction constraints."""
        # 如果没有定义交互约束，则返回 None
        if self.interaction_cst is None:
            return None
        
        # 如果交互约束为 "no_interactions"，则每个特征作为自己的独立组
        if self.interaction_cst == "no_interactions":
            interaction_cst = [[i] for i in range(n_features)]
        # 如果交互约束为 "pairwise"，则生成所有特征的两两组合
        elif self.interaction_cst == "pairwise":
            interaction_cst = itertools.combinations(range(n_features), 2)
        else:
            interaction_cst = self.interaction_cst
        
        try:
            # 将交互约束转换为集合的列表
            constraints = [set(group) for group in interaction_cst]
        except TypeError:
            # 如果交互约束不是序列的元组或列表，则引发值错误
            raise ValueError(
                "Interaction constraints must be a sequence of tuples or lists, got:"
                f" {self.interaction_cst!r}."
            )
        
        # 验证每个组中的索引是否有效
        for group in constraints:
            for x in group:
                if not (isinstance(x, Integral) and 0 <= x < n_features):
                    raise ValueError(
                        "Interaction constraints must consist of integer indices in"
                        f" [0, n_features - 1] = [0, {n_features - 1}], specifying the"
                        " position of features, got invalid indices:"
                        f" {group!r}"
                    )
        
        # 将未列出的特征作为默认组添加到约束中
        rest = set(range(n_features)) - set().union(*constraints)
        if len(rest) > 0:
            constraints.append(rest)
        
        # 返回所有有效的交互约束条件
        return constraints

    # 使用装饰器检查模型是否已拟合
    @_fit_context(prefer_skip_nested_validation=True)
    def _is_fitted(self):
        return len(getattr(self, "_predictors", [])) > 0

    # 清除梯度提升模型的状态
    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        # 删除训练分数和验证分数的属性状态
        for var in ("train_score_", "validation_score_"):
            if hasattr(self, var):
                delattr(self, var)
    def _get_small_trainset(self, X_binned_train, y_train, sample_weight_train, seed):
        """Compute the indices of the subsample set and return this set.

        For efficiency, we need to subsample the training set to compute scores
        with scorers.
        """
        # 如果训练集大小超过10000个样本，则进行子采样
        subsample_size = 10000
        if X_binned_train.shape[0] > subsample_size:
            # 创建索引数组，包含训练集样本的所有索引
            indices = np.arange(X_binned_train.shape[0])
            # 如果是分类器，使用分层采样策略
            stratify = y_train if is_classifier(self) else None
            # 对索引数组进行无放回采样，保证子采样集大小为subsample_size
            indices = resample(
                indices,
                n_samples=subsample_size,
                replace=False,
                random_state=seed,
                stratify=stratify,
            )
            # 根据子采样后的索引，获取对应的训练集特征数据和标签
            X_binned_small_train = X_binned_train[indices]
            y_small_train = y_train[indices]
            # 如果存在样本权重，则也对样本权重进行子采样
            if sample_weight_train is not None:
                sample_weight_small_train = sample_weight_train[indices]
            else:
                sample_weight_small_train = None
            # 将子采样后的特征数据转换为连续内存的数组
            X_binned_small_train = np.ascontiguousarray(X_binned_small_train)
            # 返回子采样后的特征数据、标签、样本权重和索引数组
            return (
                X_binned_small_train,
                y_small_train,
                sample_weight_small_train,
                indices,
            )
        else:
            # 如果训练集大小不超过10000个样本，则直接返回原始数据
            return X_binned_train, y_train, sample_weight_train, slice(None)

    def _check_early_stopping_scorer(
        self,
        X_binned_small_train,
        y_small_train,
        sample_weight_small_train,
        X_binned_val,
        y_val,
        sample_weight_val,
        raw_predictions_small_train=None,
        raw_predictions_val=None,
    ):
        """Check if fitting should be early-stopped based on scorer.

        Scores are computed on validation data or on training data.
        """
        # 如果是分类器，将标签转换为类别数组
        if is_classifier(self):
            y_small_train = self.classes_[y_small_train.astype(int)]

        # 计算在小训练集上的得分并添加到训练得分列表中
        self.train_score_.append(
            self._score_with_raw_predictions(
                X_binned_small_train,
                y_small_train,
                sample_weight_small_train,
                raw_predictions_small_train,
            )
        )

        # 如果使用验证数据
        if self._use_validation_data:
            # 如果是分类器，将验证集标签转换为类别数组
            if is_classifier(self):
                y_val = self.classes_[y_val.astype(int)]
            # 计算在验证集上的得分并添加到验证得分列表中
            self.validation_score_.append(
                self._score_with_raw_predictions(
                    X_binned_val, y_val, sample_weight_val, raw_predictions_val
                )
            )
            # 根据验证得分判断是否应该提前停止训练
            return self._should_stop(self.validation_score_)
        else:
            # 根据训练得分判断是否应该提前停止训练
            return self._should_stop(self.train_score_)
    # 如果未提供原始预测值，则创建一个空的上下文管理器作为替代
    if raw_predictions is None:
        patcher_raw_predict = nullcontext()
    else:
        # 使用提供的原始预测值创建一个上下文管理器
        patcher_raw_predict = _patch_raw_predict(self, raw_predictions)

    # 执行上下文管理器，确保在其范围内运行的代码块按预期工作
    with patcher_raw_predict:
        # 如果没有提供样本权重，则计算评分（分数）使用的函数
        if sample_weight is None:
            return self._scorer(self, X, y)
        else:
            # 否则，使用提供的样本权重计算评分（分数）
            return self._scorer(self, X, y, sample_weight=sample_weight)

    def _check_early_stopping_loss(
        self,
        raw_predictions,
        y_train,
        sample_weight_train,
        raw_predictions_val,
        y_val,
        sample_weight_val,
        n_threads=1,
    ):
        """Check if fitting should be early-stopped based on loss.

        Scores are computed on validation data or on training data.
        """
        # 计算训练数据的损失得分，并将其添加到训练得分列表中
        self.train_score_.append(
            -self._loss(
                y_true=y_train,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight_train,
                n_threads=n_threads,
            )
        )

        # 如果要使用验证数据，计算验证数据的损失得分，并将其添加到验证得分列表中
        if self._use_validation_data:
            self.validation_score_.append(
                -self._loss(
                    y_true=y_val,
                    raw_prediction=raw_predictions_val,
                    sample_weight=sample_weight_val,
                    n_threads=n_threads,
                )
            )
            # 根据验证得分判断是否应该早停
            return self._should_stop(self.validation_score_)
        else:
            # 否则，根据训练得分判断是否应该早停
            return self._should_stop(self.train_score_)

    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        # 确定参考位置，即评分列表中要比较的位置
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # 更高的得分总是更好的。较高的tol值意味着后续的迭代更难被视为对参考得分的改进，
        # 因此更可能因为缺乏显著改进而进行早停。
        reference_score = scores[-reference_position] + self.tol
        # 获取最近的得分，以确定是否有显著改进
        recent_scores = scores[-reference_position + 1 :]
        recent_improvements = [score > reference_score for score in recent_scores]
        # 如果没有最近的显著改进，则返回True，表示应该早停
        return not any(recent_improvements)
    # Bin data X according to the specified training or validation context.
    # If is_training_data is True, fit the _bin_mapper to X and transform it.
    # If False, only transform X using the already fitted _bin_mapper.
    def _bin_data(self, X, is_training_data):
        """Bin data X.

        If is_training_data, then fit the _bin_mapper attribute.
        Else, the binned data is converted to a C-contiguous array.
        """

        description = "training" if is_training_data else "validation"
        # Print information about the amount of data being binned if verbose mode is on
        if self.verbose:
            print(
                "Binning {:.3f} GB of {} data: ".format(X.nbytes / 1e9, description),
                end="",
                flush=True,
            )
        tic = time()
        if is_training_data:
            # Fit and transform X to a F-aligned array using _bin_mapper
            X_binned = self._bin_mapper.fit_transform(X)  # F-aligned array
        else:
            # Transform X to a F-aligned array using the already fitted _bin_mapper
            X_binned = self._bin_mapper.transform(X)  # F-aligned array
            # Convert the array to C-contiguous format for faster prediction
            # Training is faster on F-arrays, but prediction benefits from C-contiguous layout
            X_binned = np.ascontiguousarray(X_binned)
        toc = time()
        if self.verbose:
            duration = toc - tic
            print("{:.3f} s".format(duration))

        return X_binned

    # Print statistics about the current fitting iteration.
    def _print_iteration_stats(self, iteration_start_time):
        """Print info about the current fitting iteration."""
        log_msg = ""

        # Extract predictors of the current iteration and calculate related statistics
        predictors_of_ith_iteration = [
            predictors_list
            for predictors_list in self._predictors[-1]
            if predictors_list
        ]
        n_trees = len(predictors_of_ith_iteration)
        max_depth = max(
            predictor.get_max_depth() for predictor in predictors_of_ith_iteration
        )
        n_leaves = sum(
            predictor.get_n_leaf_nodes() for predictor in predictors_of_ith_iteration
        )

        # Format message with number of trees, leaves, and max depth
        if n_trees == 1:
            log_msg += "{} tree, {} leaves, ".format(n_trees, n_leaves)
        else:
            log_msg += "{} trees, {} leaves ".format(n_trees, n_leaves)
            log_msg += "({} on avg), ".format(int(n_leaves / n_trees))

        log_msg += "max depth = {}, ".format(max_depth)

        # Include training and validation scores if early stopping is enabled
        if self.do_early_stopping_:
            if self.scoring == "loss":
                factor = -1  # score_ arrays contain the negative loss
                name = "loss"
            else:
                factor = 1
                name = "score"
            log_msg += "train {}: {:.5f}, ".format(name, factor * self.train_score_[-1])
            if self._use_validation_data:
                log_msg += "val {}: {:.5f}, ".format(
                    name, factor * self.validation_score_[-1]
                )

        iteration_time = time() - iteration_start_time
        log_msg += "in {:0.3f}s".format(iteration_time)

        # Print the constructed log message
        print(log_msg)
    def _raw_predict(self, X, n_threads=None):
        """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        n_threads : int, default=None
            Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
            to determine the effective number of threads use, which takes cgroups CPU
            quotes into account. See the docstring of `_openmp_effective_n_threads`
            for details.

        Returns
        -------
        raw_predictions : array, shape (n_samples, n_trees_per_iteration)
            The raw predicted values.
        """
        # Ensure the model is fitted before making predictions
        check_is_fitted(self)
        
        # Check if the model has been binned during training; if not, preprocess input X
        is_binned = getattr(self, "_in_fit", False)
        if not is_binned:
            X = self._preprocess_X(X, reset=False)

        # Initialize an array to store the raw predictions
        n_samples = X.shape[0]
        raw_predictions = np.zeros(
            shape=(n_samples, self.n_trees_per_iteration_),
            dtype=self._baseline_prediction.dtype,
            order="F",
        )
        raw_predictions += self._baseline_prediction  # Add baseline predictions

        # Determine the effective number of threads to use for prediction
        n_threads = _openmp_effective_n_threads(n_threads)

        # Perform predictions using the stored predictors
        self._predict_iterations(
            X, self._predictors, raw_predictions, is_binned, n_threads
        )
        return raw_predictions

    def _predict_iterations(self, X, predictors, raw_predictions, is_binned, n_threads):
        """Add the predictions of the predictors to raw_predictions."""
        # Prepare additional data structures if the model was not binned during training
        if not is_binned:
            (
                known_cat_bitsets,
                f_idx_map,
            ) = self._bin_mapper.make_known_categories_bitsets()

        # Iterate through predictors and make predictions
        for predictors_of_ith_iteration in predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                # Choose the appropriate prediction method based on whether the model is binned
                if is_binned:
                    predict = partial(
                        predictor.predict_binned,
                        missing_values_bin_idx=self._bin_mapper.missing_values_bin_idx_,
                        n_threads=n_threads,
                    )
                else:
                    predict = partial(
                        predictor.predict,
                        known_cat_bitsets=known_cat_bitsets,
                        f_idx_map=f_idx_map,
                        n_threads=n_threads,
                    )
                # Add predictions to raw_predictions
                raw_predictions[:, k] += predict(X)
    # 定义一个方法，用于在每个迭代阶段计算输入数据集 X 的原始预测值
    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        raw_predictions : generator of ndarray of shape \
            (n_samples, n_trees_per_iteration)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)
        
        # 预处理输入数据 X，确保特征处理的一致性
        X = self._preprocess_X(X, reset=False)
        
        # 检查输入数据的特征维度是否与模型训练时一致
        if X.shape[1] != self._n_features:
            raise ValueError(
                "X has {} features but this estimator was trained with "
                "{} features.".format(X.shape[1], self._n_features)
            )
        
        # 获取输入数据的样本数
        n_samples = X.shape[0]
        
        # 初始化原始预测值矩阵，使用基准预测值进行初始化
        raw_predictions = np.zeros(
            shape=(n_samples, self.n_trees_per_iteration_),
            dtype=self._baseline_prediction.dtype,
            order="F",
        )
        raw_predictions += self._baseline_prediction

        # 在预测时故意将使用的线程数与拟合时使用的线程数分开，
        # 因为模型可能在不同的机器上进行预测
        n_threads = _openmp_effective_n_threads()
        
        # 遍历每个迭代阶段的预测器
        for iteration in range(len(self._predictors)):
            # 对当前迭代的预测器进行预测
            self._predict_iterations(
                X,
                self._predictors[iteration : iteration + 1],
                raw_predictions,
                is_binned=False,
                n_threads=n_threads,
            )
            # 生成当前迭代阶段的原始预测值的副本，并通过生成器返回
            yield raw_predictions.copy()
    def _compute_partial_dependence_recursion(self, grid, target_features):
        """快速计算部分依赖。

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features), dtype=np.float32
            要评估部分依赖的网格点。
        target_features : ndarray, shape (n_target_features), dtype=np.intp
            要评估部分依赖的目标特征集合。

        Returns
        -------
        averaged_predictions : ndarray, shape (n_trees_per_iteration, n_samples)
            每个网格点的部分依赖函数值。
        """

        if getattr(self, "_fitted_with_sw", False):
            raise NotImplementedError(
                "{} does not support partial dependence "
                "plots with the 'recursion' method when "
                "sample weights were given during fit "
                "time.".format(self.__class__.__name__)
            )

        # 将 grid 转换为特定类型 X_DTYPE 的 ndarray
        grid = np.asarray(grid, dtype=X_DTYPE, order="C")
        # 初始化 averaged_predictions 数组，用于存储计算结果
        averaged_predictions = np.zeros(
            (self.n_trees_per_iteration_, grid.shape[0]), dtype=Y_DTYPE
        )
        # 将 target_features 转换为特定类型 np.intp 的 ndarray
        target_features = np.asarray(target_features, dtype=np.intp, order="C")

        # 遍历 self._predictors 中的每个迭代器的预测器
        for predictors_of_ith_iteration in self._predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                # 调用预测器的 compute_partial_dependence 方法计算部分依赖
                predictor.compute_partial_dependence(
                    grid, target_features, averaged_predictions[k]
                )
        # 注意：学习率已经在叶子节点的值中考虑过了。

        return averaged_predictions

    def _more_tags(self):
        """返回额外的标签字典，允许 NaN 值。"""
        return {"allow_nan": True}

    @abstractmethod
    def _get_loss(self, sample_weight):
        """抽象方法：获取损失函数值。"""
        pass

    @abstractmethod
    def _encode_y(self, y=None):
        """抽象方法：编码目标变量 y。"""
        pass

    @property
    def n_iter_(self):
        """提供属性：返回提升过程的迭代次数。"""
        check_is_fitted(self)
        return len(self._predictors)
# 定义一个新的回归器类，继承自`RegressorMixin`和`BaseHistGradientBoosting`。
class HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting):
    """Histogram-based Gradient Boosting Regression Tree.
    
    这个估计器比大数据集（n_samples >= 10 000）下的:class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`快得多。
    
    这个估计器原生支持缺失值（NaN）。在训练时，树生成器会根据潜在增益学习每个分裂点上缺失值样本应该进入左子树还是右子树。
    在预测时，缺失值样本会相应地被分配到左子树或右子树。如果在训练过程中对于某个特征没有遇到缺失值，则在预测时缺失值样本会被映射到样本数最多的子树。
    参见 :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` 以查看这一特性的用例示例。
    
    这个实现受到`LightGBM <https://github.com/Microsoft/LightGBM>`_的启发。
    
    详细内容请参阅 :ref:`User Guide <histogram_based_gradient_boosting>`。
    
    .. versionadded:: 0.21  添加版本：0.21
    
    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'}, \
            default='squared_error'
        在提升过程中使用的损失函数。注意，“squared_error”、“gamma”和“poisson”损失实际上实现了“半最小二乘损失”、“半Gamma偏差”和“半Poisson偏差”，以简化梯度计算。
        此外，“gamma”和“poisson”损失内部使用对数链接，“gamma”要求``y > 0``和“poisson”要求``y >= 0``。
        “quantile”使用Pinball损失。
        
        .. versionchanged:: 0.23  添加选项 'poisson'。
        .. versionchanged:: 1.1   添加选项 'quantile'。
        .. versionchanged:: 1.3   添加选项 'gamma'。
    
    quantile : float, default=None
        如果损失是“quantile”，此参数指定要估计的分位数，必须介于0和1之间。
    
    learning_rate : float, default=0.1
        学习率，也称为*缩减*。这是用于叶值的乘法因子。使用``1``表示没有缩减。
    
    max_iter : int, default=100
        提升过程的最大迭代次数，即树的最大数量。
    
    max_leaf_nodes : int or None, default=31
        每棵树的最大叶子数。必须严格大于1。如果为None，则没有最大限制。
    
    max_depth : int or None, default=None
        每棵树的最大深度。树的深度是从根到最深叶子的边数。
        默认情况下深度不受限制。
    
    """
    # 最小叶子节点样本数，默认为20。
    min_samples_leaf : int, default=20
        # 对于样本数少于几百个的小数据集，建议降低此值，
        # 因为只会构建非常浅的树。
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.

    # L2 正则化参数，惩罚具有较小 hessian 的叶子。
    l2_regularization : float, default=0
        # 使用 ``0`` 表示没有正则化（默认）。
        The L2 regularization parameter penalizing leaves with small hessians.
        Use ``0`` for no regularization (default).

    # 每个节点分裂时随机选择的特征比例。
    max_features : float, default=1.0
        # 这是一种正则化形式，较小的值会使得树的学习能力更弱，
        # 可能可以防止过拟合。
        # 如果存在来自 `interaction_cst` 的交互约束，则只允许
        # 指定的特征用于子采样。
        Proportion of randomly chosen features in each and every node split.
        This is a form of regularization, smaller values make the trees weaker
        learners and might prevent overfitting.
        If interaction constraints from `interaction_cst` are present, only allowed
        features are taken into account for the subsampling.

        .. versionadded:: 1.4

    # 非缺失值使用的最大箱数。
    max_bins : int, default=255
        # 在训练之前，将输入数组 `X` 的每个特征进行分箱，
        # 分箱后可以大大加快训练速度。
        # 具有少量唯一值的特征可能使用少于 ``max_bins`` 个箱。
        # 除了 ``max_bins`` 个箱外，还始终保留一个用于缺失值。
        # 必须不大于 255。
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.

    # 指示分类特征的数组形状。
    categorical_features : array-like of {bool, int, str} of shape (n_features) \
            or shape (n_categorical_features,), default=None
        # 表示分类特征。

        # - None: 不考虑任何特征为分类特征。
        # - 布尔数组: 表示分类特征的布尔掩码。
        # - 整数数组: 表示分类特征的整数索引。
        # - 字符串数组: 表示分类特征的名称（假定训练数据具有特征名称）。
        # - `"from_dtype"`: 具有 dtype 为 "category" 的 dataframe 列被视为分类特征。
        #   输入必须是具有 ``__dataframe__`` 方法的对象，例如 pandas 或 polars
        #   的 DataFrame 才能使用此特性。

        # 每个分类特征最多可以有 `max_bins` 个唯一类别。
        # 作为数字类型编码的分类特征的负值将视为缺失值。
        # 所有分类值都将转换为浮点数。这意味着分类值 1.0 和 1 被视为相同类别。

        # 详细内容请参阅 :ref:`用户指南 <categorical_support_gbdt>`。

        .. versionadded:: 0.24

        .. versionchanged:: 1.2
           增加了对特征名称的支持。

        .. versionchanged:: 1.4
           增加了 `"from_dtype"` 选项。默认将在 v1.6 中更改为 `"from_dtype"`。
    monotonic_cst : array-like of int of shape (n_features) or dict, default=None
        # 定义变量 `monotonic_cst`，用于指定每个特征的单调性约束。
        # 可以是一个整数数组或字典：
        # - 整数含义：
        #   - 1：单调递增
        #   - 0：无约束
        #   - -1：单调递减
        # - 如果是字典，以特征名为键，指定特征对应的单调性约束。
        #   如果是数组，则按位置映射到约束。
        # 查看更多信息请参考用户指南中的 `monotonic_cst_gbdt` 章节。
        .. versionadded:: 0.23

        .. versionchanged:: 1.2
           支持以特征名为键的字典形式定义约束。
    
    interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets \
            of int, default=None
        # 定义变量 `interaction_cst`，用于指定特征之间的交互约束。
        # 每个项目指定哪些特征索引可以在子节点分裂时相互交互。
        # 如果特征数超出约束指定的范围，则视为额外的集合。
        # 字符串 "pairwise" 和 "no_interactions" 是只允许成对交互或不允许交互的简写形式。
        # 例如，对于总共5个特征，在 `interaction_cst=[{0, 1}]` 的情况下，
        # 等同于 `interaction_cst=[{0, 1}, {2, 3, 4}]`，
        # 指定树的每个分支只能在特征0和1上进行分裂，或者在特征2、3、4上进行分裂。
        .. versionadded:: 1.2

    warm_start : bool, default=False
        # 当设置为 `True` 时，重新使用上一次拟合的解决方案，并向集成中添加更多的估算器。
        # 为了结果有效，估算器应仅在相同数据上重新训练。
        # 请参阅术语表中的 `warm_start`。
    
    early_stopping : 'auto' or bool, default='auto'
        # 如果设置为 `'auto'`，并且样本大小大于10000，则启用早停。
        # 如果为 `True`，则启用早停，否则禁用早停。
        # 请参阅版本0.23中添加的说明。
    
    scoring : str or callable or None, default='loss'
        # 用于早停的评分参数。可以是一个字符串（参见 `scoring_parameter`）或可调用对象（参见 `scoring`）。
        # 如果为 `None`，则使用估算器的默认评分器。
        # 如果 `scoring='loss'`，则相对于损失值进行早停检查。
        # 仅在执行早停时使用。
    
    validation_fraction : int or float or None, default=0.1
        # 作为验证数据的比例（或绝对大小），用于早停。
        # 如果为 `None`，则在训练数据上进行早停。
        # 仅在执行早停时使用。
    # 最大迭代次数，用于确定何时进行“提前停止”。当最后的 ``n_iter_no_change`` 分数都没有比倒数第 ``n_iter_no_change - 1`` 个更好时，停止拟合过程，直到某个容差。仅在执行提前停止时使用。
    n_iter_no_change : int, default=10

    # 在早期停止期间比较分数时使用的绝对容差。容差越高，越可能提前停止：更高的容差意味着后续迭代更难被视为对参考分数的改进。
    tol : float, default=1e-7

    # 决定拟合过程中是否输出信息的详细程度。如果不为零，打印有关拟合过程的一些信息。
    verbose : int, default=0

    # 伪随机数生成器，用于控制分箱过程中的子采样以及启用早期停止时的训练/验证数据分割。传递一个整数以确保多次函数调用产生可重现的输出。参见“术语表”中的“随机状态”。
    random_state : int, RandomState instance or None, default=None

    # 指示是否在训练过程中使用早期停止。
    Attributes
    ----------
    do_early_stopping_ : bool

    # 通过早期停止选择的迭代次数，取决于 `early_stopping` 参数。否则，它对应于 `max_iter`。
    n_iter_ : int

    # 每次迭代构建的树的数量。对于回归器来说，这总是 1。
    n_trees_per_iteration_ : int

    # 每个迭代在训练数据上的得分。第一个条目是在第一次迭代之前集合的得分。根据 ``scoring`` 参数计算得分。如果 ``scoring`` 不是 'loss'，则在最多 10,000 个样本的子集上计算得分。如果没有启用早期停止，则为空。
    train_score_ : ndarray, shape (n_iter_+1,)

    # 在验证数据上每次迭代的得分。第一个条目是在第一次迭代之前集合的得分。根据 ``scoring`` 参数计算得分。如果没有启用早期停止或者 ``validation_fraction`` 是 None，则为空。
    validation_score_ : ndarray, shape (n_iter_+1,)

    # 分类特征的布尔掩码。如果没有分类特征，则为 ``None``。
    is_categorical_ : ndarray, shape (n_features, ) or None

    # 在 :term:`fit` 过程中看到的特征数。
    n_features_in_ : int

    # 在 :term:`fit` 过程中看到的特征的名称。仅当 `X` 具有全为字符串的特征名称时定义。
    feature_names_in_ : ndarray of shape (`n_features_in_`,)

    # 参见
    # GradientBoostingRegressor：确切的梯度提升方法，在具有大量样本的数据集上效果不佳。
    See Also
    --------
    GradientBoostingRegressor
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    RandomForestRegressor : A meta-estimator that fits a number of decision
        tree regressors on various sub-samples of the dataset and uses
        averaging to improve the statistical performance and control
        over-fitting.
    AdaBoostRegressor : A meta-estimator that begins by fitting a regressor
        on the original dataset and then fits additional copies of the
        regressor on the same dataset but where the weights of instances are
        adjusted according to the error of the current prediction. As such,
        subsequent regressors focus more on difficult cases.

    Examples
    --------
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> est = HistGradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.92...
    """

    # 定义参数约束字典，继承自基类 BaseHistGradientBoosting 的参数约束
    _parameter_constraints: dict = {
        **BaseHistGradientBoosting._parameter_constraints,
        "loss": [
            StrOptions(
                {
                    "squared_error",
                    "absolute_error",
                    "poisson",
                    "gamma",
                    "quantile",
                }
            ),
            BaseLoss,
        ],
        "quantile": [Interval(Real, 0, 1, closed="both"), None],
    }

    # 初始化方法，设置各种参数来配置 HistGradientBoostingRegressor 实例
    def __init__(
        self,
        loss="squared_error",  # 损失函数，默认为平方误差
        *,
        quantile=None,  # 分位数参数，默认为None
        learning_rate=0.1,  # 学习率，默认为0.1
        max_iter=100,  # 最大迭代次数，默认为100
        max_leaf_nodes=31,  # 叶子节点最大数目，默认为31
        max_depth=None,  # 树的最大深度，默认为None（不限制深度）
        min_samples_leaf=20,  # 叶子节点最小样本数，默认为20
        l2_regularization=0.0,  # L2 正则化参数，默认为0.0
        max_features=1.0,  # 最大特征数，默认为1.0
        max_bins=255,  # 直方图梯度提升中的最大箱数，默认为255
        categorical_features="warn",  # 类别特征处理策略，默认警告
        monotonic_cst=None,  # 单调性约束，默认为None
        interaction_cst=None,  # 交互约束，默认为None
        warm_start=False,  # 是否热启动，默认为False
        early_stopping="auto",  # 是否提前停止，默认为"auto"
        scoring="loss",  # 评分方法，默认为"loss"
        validation_fraction=0.1,  # 验证集比例，默认为0.1
        n_iter_no_change=10,  # 连续迭代次数不改善时停止，默认为10
        tol=1e-7,  # 迭代收敛容忍度，默认为1e-7
        verbose=0,  # 是否输出详细信息，默认为0（不输出）
        random_state=None,  # 随机数种子，默认为None
    ):
        # 调用父类的初始化方法，设置 HistGradientBoostingRegressor 的参数
        super(HistGradientBoostingRegressor, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            categorical_features=categorical_features,
            early_stopping=early_stopping,
            warm_start=warm_start,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        # 设置 quantile 属性
        self.quantile = quantile
    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """
        # 确保模型已经拟合（适用于监督学习模型）
        check_is_fitted(self)
        # 返回经过原始预测的反函数的结果，将形状从 (n_samples, 1) 转换为 (n_samples,)
        return self._loss.link.inverse(self._raw_predict(X).ravel())

    def staged_predict(self, X):
        """Predict regression target for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted values of the input samples, for each iteration.
        """
        # 针对每个迭代阶段预测回归目标
        for raw_predictions in self._staged_raw_predict(X):
            yield self._loss.link.inverse(raw_predictions.ravel())

    def _encode_y(self, y):
        # 将 y 转换为期望的数据类型
        self.n_trees_per_iteration_ = 1
        y = y.astype(Y_DTYPE, copy=False)
        if self.loss == "gamma":
            # 确保 y > 0
            if not np.all(y > 0):
                raise ValueError("loss='gamma' requires strictly positive y.")
        elif self.loss == "poisson":
            # 确保 y >= 0 并且 sum(y) > 0
            if not (np.all(y >= 0) and np.sum(y) > 0):
                raise ValueError(
                    "loss='poisson' requires non-negative y and sum(y) > 0."
                )
        return y

    def _get_loss(self, sample_weight):
        if self.loss == "quantile":
            # 返回相应损失函数的实例，用于特定的加权样本和分位数
            return _LOSSES[self.loss](
                sample_weight=sample_weight, quantile=self.quantile
            )
        else:
            # 返回相应损失函数的实例，用于特定的加权样本
            return _LOSSES[self.loss](sample_weight=sample_weight)
# 定义一个类 HistGradientBoostingClassifier，继承自 ClassifierMixin 和 BaseHistGradientBoosting
class HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting):
    """Histogram-based Gradient Boosting Classification Tree.

    This estimator is much faster than
    :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
    for big datasets (n_samples >= 10 000).

    This estimator has native support for missing values (NaNs). During
    training, the tree grower learns at each split point whether samples
    with missing values should go to the left or right child, based on the
    potential gain. When predicting, samples with missing values are
    assigned to the left or right child consequently. If no missing values
    were encountered for a given feature during training, then samples with
    missing values are mapped to whichever child has the most samples.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    loss : {'log_loss'}, default='log_loss'
        The loss function to use in the boosting process.

        For binary classification problems, 'log_loss' is also known as logistic loss,
        binomial deviance or binary crossentropy. Internally, the model fits one tree
        per boosting iteration and uses the logistic sigmoid function (expit) as
        inverse link function to compute the predicted positive class probability.

        For multiclass classification problems, 'log_loss' is also known as multinomial
        deviance or categorical crossentropy. Internally, the model fits one tree per
        boosting iteration and per class and uses the softmax function as inverse link
        function to compute the predicted probabilities of the classes.

    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    # L2 正则化参数，惩罚具有较小 hessian 的叶子节点。
    # 使用 ``0`` 表示无正则化（默认）。
    l2_regularization : float, default=0

    # 每个节点分裂时随机选择特征的比例。
    # 这是一种正则化方法，较小的值使得树的学习能力较弱，可能有助于防止过拟合。
    # 如果存在来自 `interaction_cst` 的交互约束，则仅允许使用指定的特征进行子抽样。
    # 
    # .. versionadded:: 1.4
    max_features : float, default=1.0

    # 针对非缺失值使用的最大 bin 数量。
    # 在训练之前，输入数组 `X` 的每个特征都会被分成整数值 bin，这样可以大大加快训练阶段。
    # 具有少量唯一值的特征可能会使用少于 ``max_bins`` 个 bin。
    # 除了 ``max_bins`` 个 bin 外，还会额外保留一个 bin 用于缺失值。
    # 必须不大于 255。
    max_bins : int, default=255

    # 表示分类特征的数组。
    # 
    # - None: 表示没有特征被视为分类特征。
    # - 布尔数组: 表示分类特征的布尔掩码。
    # - 整数数组: 表示分类特征的整数索引。
    # - 字符串数组: 表示分类特征的名称（假设训练数据有特征名称）。
    # - `"from_dtype"`: 表示数据类型为 "category" 的数据帧列被视为分类特征。
    #   输入必须是具有 `__dataframe__` 方法的对象，如 pandas 或 polars DataFrames。
    # 
    # 每个分类特征最多可以有 `max_bins` 个唯一的类别。
    # 对于以数值类型编码的分类特征中的负值会被视为缺失值。
    # 所有分类值都会被转换为浮点数。这意味着分类值 1.0 和 1 被视为同一类别。
    # 
    # 在 :ref:`用户指南 <categorical_support_gbdt>` 中阅读更多详细信息。
    # 
    # .. versionadded:: 0.24
    # 
    # .. versionchanged:: 1.2
    #    增加了对特征名称的支持。
    # 
    # .. versionchanged:: 1.4
    #    增加了 `"from_dtype"` 选项。默认将在 v1.6 中更改为 `"from_dtype"`。
    categorical_features : array-like of {bool, int, str} of shape (n_features) \
            or shape (n_categorical_features,), default=None
    monotonic_cst : array-like of int of shape (n_features) or dict, default=None
        # 定义单调约束，可以是一个数组或字典：
        # - 数组情况下，每个元素代表对应特征的单调性约束：
        #   - 1：单调增加
        #   - 0：无约束
        #   - -1：单调减少
        # - 字典情况下，特征名映射到对应的单调性约束
        # 只在二分类问题中生效，且约束适用于正类的概率
        # 更多信息请参考用户指南中的相关部分

        If a dict with str keys, map feature to monotonic constraints by name.
        If an array, the features are mapped to constraints by position. See
        :ref:`monotonic_cst_features_names` for a usage example.

        The constraints are only valid for binary classifications and hold
        over the probability of the positive class.
        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 0.23

        .. versionchanged:: 1.2
           Accept dict of constraints with feature names as keys.

    interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets \
            of int, default=None
        # 定义交互约束，即哪些特征可以在子节点分裂时相互作用：
        # 每个条目指定允许相互作用的特征索引集合
        # 如果指定的特征少于总特征数，剩余特征将视为另一组集合
        # 字符串 "pairwise" 和 "no_interactions" 分别代表仅允许成对交互和禁止交互
        # 例如，对于总共5个特征，`interaction_cst=[{0, 1}]` 等同于 `interaction_cst=[{0, 1}, {2, 3, 4}]`
        # 指定每棵树分支只能在特征0和1或特征2、3和4上分裂

        .. versionadded:: 1.2

    warm_start : bool, default=False
        # 当设置为True时，重复使用上次拟合的解决方案，并向集成中添加更多的估算器
        # 结果要有效，估算器应仅在相同数据上重新训练
        # 参见术语表中的“温启动”条目

    early_stopping : 'auto' or bool, default='auto'
        # 如果设置为 'auto'，且样本大小大于10000时启用早停
        # 如果设置为 True，启用早停，否则禁用早停

        .. versionadded:: 0.23

    scoring : str or callable or None, default='loss'
        # 用于早停的评分参数。可以是单个字符串或可调用对象
        # 如果为None，则使用估算器的默认评分器
        # 如果 `scoring='loss'`，则根据损失值进行早停检查
        # 仅在执行早停时使用
    # 验证集的比例或绝对大小，用于提前停止时作为验证数据。如果为 None，则在训练数据上进行提前停止。
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.
    
    # 用于确定何时进行“提前停止”。当最后的 n_iter_no_change 个得分都没有比倒数第二个更好时，停止拟合过程，直到达到某个容差。只在执行提前停止时使用。
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.
    
    # 在比较得分时使用的绝对容差。容差越高，越可能提前停止：更高的容差意味着后续迭代很难被认为是对参考得分的改进。
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    
    # 详细程度的级别。如果不为零，打印关于拟合过程的一些信息。
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    
    # 伪随机数生成器，控制子采样过程中的分箱和启用提前停止时的训练/验证数据拆分。
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    
    # 类别权重的字典或'balanced'。格式为 `{class_label: weight}`。
    # 如果未提供，则假定所有类别权重均为一。
    # “balanced”模式根据输入数据中每个类别的频率自动调整权重，与类别频率成反比。
    # 注意，如果指定了 `sample_weight`，这些权重将与 `sample_weight` 相乘（通过 `fit` 方法传递）。
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form `{class_label: weight}`.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as `n_samples / (n_classes * np.bincount(y))`.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if `sample_weight` is specified.

        .. versionadded:: 1.2

    # 类标签的数组，形状为 (n_classes,)
    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        Class labels.
    
    # 是否在训练过程中使用了提前停止。
    do_early_stopping_ : bool
        Indicates whether early stopping is used during training.
    
    # 由提前停止选择的迭代次数。否则，它对应于 max_iter。
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    
    # 每次迭代构建的树的数量。对于二元分类，它等于 1；对于多类分类，它等于 `n_classes`。
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. This is equal to 1
        for binary classification, and to ``n_classes`` for multiclass
        classification.
    
    # 训练数据每次迭代的得分数组，形状为 (n_iter_+1,)
    # 第一个条目是第一次迭代之前的集成得分。
    # 根据 `scoring` 参数计算得分。如果 `scoring` 不是 'loss'，则在最多 10,000 个样本的子集上计算得分。如果没有提前停止，为空。
    train_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    _parameter_constraints: dict = {
        **BaseHistGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"log_loss"}), BaseLoss],
        "class_weight": [dict, StrOptions({"balanced"}), None],
    }


# 定义参数约束字典，继承自基类 BaseHistGradientBoosting 的约束，并添加新的约束条件。



    def __init__(
        self,
        loss="log_loss",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="warn",
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        class_weight=None,


# 初始化方法，设置 HistGradientBoostingClassifier 的参数及其默认值。
# 参数列表包括：
# - loss: 损失函数，默认为 "log_loss"
# - learning_rate: 学习率，默认为 0.1
# - max_iter: 最大迭代次数，默认为 100
# - max_leaf_nodes: 单棵树最大叶子节点数，默认为 31
# - max_depth: 单棵树最大深度，默认为 None
# - min_samples_leaf: 叶子节点最小样本数，默认为 20
# - l2_regularization: L2 正则化参数，默认为 0.0
# - max_features: 最大特征数的比例，默认为 1.0
# - max_bins: 直方图中的最大箱数，默认为 255
# - categorical_features: 类别特征处理方式，默认为 "warn"
# - monotonic_cst: 单调性约束，默认为 None
# - interaction_cst: 交互约束，默认为 None
# - warm_start: 是否启用热启动，默认为 False
# - early_stopping: 是否启用早停，默认为 "auto"
# - scoring: 评分方法，默认为 "loss"
# - validation_fraction: 验证集比例，默认为 0.1
# - n_iter_no_change: 连续迭代次数无改善时停止，默认为 10
# - tol: 收敛容忍度，默认为 1e-7
# - verbose: 冗余模式，默认为 0
# - random_state: 随机数种子，默认为 None
# - class_weight: 类别权重，默认为 None
    ):
        # 调用父类的初始化方法，传入各种参数来初始化梯度提升分类器
        super(HistGradientBoostingClassifier, self).__init__(
            loss=loss,  # 损失函数的类型
            learning_rate=learning_rate,  # 学习率
            max_iter=max_iter,  # 最大迭代次数
            max_leaf_nodes=max_leaf_nodes,  # 最大叶节点数
            max_depth=max_depth,  # 树的最大深度
            min_samples_leaf=min_samples_leaf,  # 叶节点最小样本数
            l2_regularization=l2_regularization,  # L2 正则化参数
            max_features=max_features,  # 最大特征数
            max_bins=max_bins,  # 最大箱数
            categorical_features=categorical_features,  # 类别特征
            monotonic_cst=monotonic_cst,  # 单调约束
            interaction_cst=interaction_cst,  # 交互约束
            warm_start=warm_start,  # 是否热启动
            early_stopping=early_stopping,  # 是否早停
            scoring=scoring,  # 评分方法
            validation_fraction=validation_fraction,  # 验证集比例
            n_iter_no_change=n_iter_no_change,  # 迭代无改变次数
            tol=tol,  # 容忍度
            verbose=verbose,  # 是否输出详细信息
            random_state=random_state,  # 随机种子
        )
        # 设置分类器的类别权重
        self.class_weight = class_weight

    def _finalize_sample_weight(self, sample_weight, y):
        """Adjust sample_weights with class_weights."""
        # 如果未设置类别权重，则直接返回样本权重
        if self.class_weight is None:
            return sample_weight

        # 根据类别权重调整样本权重
        expanded_class_weight = compute_sample_weight(self.class_weight, y)

        # 如果样本权重不为 None，则返回调整后的样本权重
        if sample_weight is not None:
            return sample_weight * expanded_class_weight
        else:
            # 否则返回扩展后的类别权重
            return expanded_class_weight

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        # TODO: This could be done in parallel
        # 获取原始预测结果
        raw_predictions = self._raw_predict(X)
        
        # 如果只有一个类别，则使用大于 0 的判断，否则使用 argmax
        if raw_predictions.shape[1] == 1:
            # 对于二分类情况，大于 0 的值即为预测结果
            encoded_classes = (raw_predictions.ravel() > 0).astype(int)
        else:
            # 对于多分类情况，取每行最大值的索引作为预测结果
            encoded_classes = np.argmax(raw_predictions, axis=1)
        
        # 返回预测类别对应的类别标签
        return self.classes_[encoded_classes]

    def staged_predict(self, X):
        """Predict classes at each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes of the input samples, for each iteration.
        """
        # 生成器函数，用于逐步预测每个迭代阶段的类别
        for raw_predictions in self._staged_raw_predict(X):
            # 如果只有一个类别，则使用大于 0 的判断，否则使用 argmax
            if raw_predictions.shape[1] == 1:
                # 对于二分类情况，大于 0 的值即为预测结果
                encoded_classes = (raw_predictions.ravel() > 0).astype(int)
            else:
                # 对于多分类情况，取每行最大值的索引作为预测结果
                encoded_classes = np.argmax(raw_predictions, axis=1)
            
            # 生成当前迭代阶段的预测结果
            yield self.classes_.take(encoded_classes, axis=0)
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # 获取原始的预测结果
        raw_predictions = self._raw_predict(X)
        # 使用损失函数对象进行预测类别概率
        return self._loss.predict_proba(raw_predictions)

    def staged_predict_proba(self, X):
        """
        Predict class probabilities at each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted class probabilities of the input samples,
            for each iteration.
        """
        # 在每个阶段预测类别概率
        for raw_predictions in self._staged_raw_predict(X):
            yield self._loss.predict_proba(raw_predictions)

    def decision_function(self, X):
        """
        Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,) or \
                (n_samples, n_trees_per_iteration)
            The raw predicted values (i.e. the sum of the trees leaves) for
            each sample. n_trees_per_iteration is equal to the number of
            classes in multiclass classification.
        """
        # 计算输入样本的决策函数值
        decision = self._raw_predict(X)
        # 如果是二分类，则将结果展平为一维数组
        if decision.shape[1] == 1:
            decision = decision.ravel()
        return decision

    def staged_decision_function(self, X):
        """
        Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        decision : generator of ndarray of shape (n_samples,) or \
                (n_samples, n_trees_per_iteration)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # 在每个阶段计算输入样本的决策函数值
        for staged_decision in self._staged_raw_predict(X):
            # 如果是二分类，则将结果展平为一维数组
            if staged_decision.shape[1] == 1:
                staged_decision = staged_decision.ravel()
            yield staged_decision
    # 将目标变量 y 编码为 0 到 n_classes - 1，并设置 classes_ 和 n_trees_per_iteration_ 属性
    check_classification_targets(y)

    # 创建一个 LabelEncoder 对象，用于对目标变量 y 进行编码
    label_encoder = LabelEncoder()

    # 对目标变量 y 进行编码，返回编码后的结果
    encoded_y = label_encoder.fit_transform(y)

    # 将 label_encoder 对象中的 classes_ 属性赋给模型的 classes_ 属性
    self.classes_ = label_encoder.classes_

    # 计算类别的数量
    n_classes = self.classes_.shape[0]

    # 对于二分类问题，每次迭代只建立一棵树；对于多类分类问题，每个类别建立一棵树
    self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes

    # 将编码后的 y 转换为指定的数据类型 Y_DTYPE，并确保不进行复制
    encoded_y = encoded_y.astype(Y_DTYPE, copy=False)

    # 返回编码后的目标变量 encoded_y
    return encoded_y

    # 根据样本权重返回损失函数对象，如果 self.n_trees_per_iteration_ 为 1，则使用二项损失函数 HalfBinomialLoss
    # 否则使用多项式损失函数 HalfMultinomialLoss，同时传入样本权重和类别数量 self.n_trees_per_iteration_
```