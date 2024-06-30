# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_partial_dependence.py`

```
"""Partial dependence plots for regression and classification models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
from collections.abc import Iterable

import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles

# 导入 scikit-learn 相关模块
from ..base import is_classifier, is_regressor
from ..ensemble import RandomForestRegressor
from ..ensemble._gb import BaseGradientBoosting
from ..ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting,
)
from ..exceptions import NotFittedError
from ..tree import DecisionTreeRegressor
from ..utils import Bunch, _safe_indexing, check_array
from ..utils._indexing import _determine_key_type, _get_column_indices, _safe_assign
from ..utils._optional_dependencies import check_matplotlib_support  # noqa
from ..utils._param_validation import (
    HasMethods,
    Integral,
    Interval,
    StrOptions,
    validate_params,
)
from ..utils.extmath import cartesian
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._pd_utils import _check_feature_names, _get_feature_index

# 公开的函数和类
__all__ = [
    "partial_dependence",
]

# 根据 X 的分位数生成网格点
def _grid_from_X(X, percentiles, is_categorical, grid_resolution):
    """Generate a grid of points based on the percentiles of X.

    The grid is a cartesian product between the columns of ``values``. The
    ith column of ``values`` consists in ``grid_resolution`` equally-spaced
    points between the percentiles of the jth column of X.

    If ``grid_resolution`` is bigger than the number of unique values in the
    j-th column of X or if the feature is a categorical feature (by inspecting
    `is_categorical`) , then those unique values will be used instead.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_target_features)
        The data.

    percentiles : tuple of float
        The percentiles which are used to construct the extreme values of
        the grid. Must be in [0, 1].

    is_categorical : list of bool
        For each feature, tells whether it is categorical or not. If a feature
        is categorical, then the values used will be the unique ones
        (i.e. categories) instead of the percentiles.

    grid_resolution : int
        The number of equally spaced points to be placed on the grid for each
        feature.

    Returns
    -------
    grid : ndarray of shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= grid_resolution ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``grid_resolution``, or the number of
        unique values in ``X[:, j]``, whichever is smaller.
    """
    # 检查 percentiles 是否为可迭代对象且长度为2
    if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
        raise ValueError("'percentiles' must be a sequence of 2 elements.")
    # 检查所有百分位数是否在 [0, 1] 范围内，如果不是则抛出数值错误异常
    if not all(0 <= x <= 1 for x in percentiles):
        raise ValueError("'percentiles' values must be in [0, 1].")
    
    # 检查第一个百分位数是否严格小于第二个百分位数，如果不是则抛出数值错误异常
    if percentiles[0] >= percentiles[1]:
        raise ValueError("percentiles[0] must be strictly less than percentiles[1].")
    
    # 检查网格分辨率是否严格大于1，如果不是则抛出数值错误异常
    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    # 初始化一个空列表 values 来存储处理后的轴数组
    values = []
    
    # 遍历 is_categorical 中的特征及其是否为分类变量的布尔值
    for feature, is_cat in enumerate(is_categorical):
        try:
            # 从 X 中获取第 feature 列的唯一值
            uniques = np.unique(_safe_indexing(X, feature, axis=1))
        except TypeError as exc:
            # 如果遇到 TypeError 异常，通常是由于列包含 `np.nan` 和字符串类别导致的，抛出详细的数值错误异常
            raise ValueError(
                f"The column #{feature} contains mixed data types. Finding unique "
                "categories fail due to sorting. It usually means that the column "
                "contains `np.nan` values together with `str` categories. Such use "
                "case is not yet supported in scikit-learn."
            ) from exc
        
        # 如果当前特征是分类变量，或者唯一值数量小于网格分辨率，则将唯一值作为轴
        if is_cat or uniques.shape[0] < grid_resolution:
            axis = uniques
        else:
            # 否则，根据给定的百分位数和网格分辨率创建轴
            emp_percentiles = mquantiles(
                _safe_indexing(X, feature, axis=1), prob=percentiles, axis=0
            )
            # 如果百分位数过于接近，则抛出数值错误异常
            if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                raise ValueError(
                    "percentiles are too close to each other, "
                    "unable to build the grid. Please choose percentiles "
                    "that are further apart."
                )
            # 使用 linspace 创建轴
            axis = np.linspace(
                emp_percentiles[0],
                emp_percentiles[1],
                num=grid_resolution,
                endpoint=True,
            )
        
        # 将生成的轴添加到 values 列表中
        values.append(axis)

    # 返回 values 的笛卡尔乘积和 values 本身
    return cartesian(values), values
# 计算部分依赖通过蛮力方法。

def _partial_dependence_brute(
    est, grid, features, X, response_method, sample_weight=None
):
    # 计算部分依赖通过蛮力方法。
    The brute method explicitly averages the predictions of an estimator over a
    grid of feature values.

    For each `grid` value, all the samples from `X` have their variables of
    interest replaced by that specific `grid` value. The predictions are then made
    and averaged across the samples.

    This method is slower than the `'recursion'`
    (:func:`~sklearn.inspection._partial_dependence._partial_dependence_recursion`)
    version for estimators with this second option. However, with the `'brute'`
    force method, the average will be done with the given `X` and not the `X`
    used during training, as it is done in the `'recursion'` version. Therefore
    the average can always accept `sample_weight` (even when the estimator was
    fitted without).

    Parameters
    ----------
    est : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    grid : array-like of shape (n_points, n_target_features)
        The grid of feature values for which the partial dependence is calculated.
        Note that `n_points` is the number of points in the grid and `n_target_features`
        is the number of features you are doing partial dependence at.

    features : array-like of {int, str}
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    X : array-like of shape (n_samples, n_features)
        `X` is used to generate values for the complement features. That is, for
        each value in `grid`, the method will average the prediction of each
        sample from `X` having that grid value for `features`.

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights are used to calculate weighted means when averaging the
        model output. If `None`, then samples are equally weighted. Note that
        `sample_weight` does not change the individual predictions.

    Returns
    -------
    averaged_predictions : array-like of shape (n_targets, n_points)
        The averaged predictions for the given `grid` of features values.
        Note that `n_targets` is the number of targets (e.g. 1 for binary
        classification, `n_tasks` for multi-output regression, and `n_classes` for
        multiclass classification) and `n_points` is the number of points in the `grid`.
    # 初始化空列表以存储预测结果
    predictions = []
    # 初始化空列表以存储平均预测结果
    averaged_predictions = []

    # 确定预测方法（predict, predict_proba, decision_function）
    if is_regressor(est):
        # 如果是回归器，则使用 predict 方法进行预测
        prediction_method = est.predict
    else:
        # 否则，尝试获取 predict_proba 和 decision_function 方法
        predict_proba = getattr(est, "predict_proba", None)
        decision_function = getattr(est, "decision_function", None)
        if response_method == "auto":
            # 根据 response_method 自动选择预测方法
            prediction_method = predict_proba or decision_function
        else:
            # 根据 response_method 明确选择预测方法
            prediction_method = (
                predict_proba
                if response_method == "predict_proba"
                else decision_function
            )
        # 如果未能获取有效的预测方法，则抛出相应的错误
        if prediction_method is None:
            if response_method == "auto":
                raise ValueError(
                    "The estimator has no predict_proba and no "
                    "decision_function method."
                )
            elif response_method == "predict_proba":
                raise ValueError("The estimator has no predict_proba method.")
            else:
                raise ValueError("The estimator has no decision_function method.")

    # 复制输入特征 X，用于评估不同的网格点
    X_eval = X.copy()
    # 遍历每个网格点
    for new_values in grid:
        # 针对每个特征变量，将新值安全地分配到 X_eval 的相应列
        for i, variable in enumerate(features):
            _safe_assign(X_eval, new_values[i], column_indexer=variable)

        try:
            # 尝试使用预测方法进行预测
            pred = prediction_method(X_eval)

            # 将预测结果添加到 predictions 列表中
            predictions.append(pred)
            # 对样本进行加权平均，将结果添加到 averaged_predictions 列表中
            averaged_predictions.append(np.average(pred, axis=0, weights=sample_weight))
        except NotFittedError as e:
            # 如果估计器未被拟合，则抛出异常
            raise ValueError("'estimator' parameter must be a fitted estimator") from e

    # 获取输入数据 X 的样本数
    n_samples = X.shape[0]

    # 调整预测结果的形状为 (n_targets, n_instances, n_points)，其中 n_targets 取决于：
    # - 非多输出回归和二元分类器时为 1（此时形状已正确）
    # - 多输出回归时为 n_tasks
    # - 多类分类时为 n_classes
    # 将 predictions 转换为 NumPy 数组，并转置
    predictions = np.array(predictions).T
    # 如果估计器是回归器并且 predictions 的维度为 2
    if is_regressor(est) and predictions.ndim == 2:
        # 非多输出回归，形状为 (n_instances, n_points,)
        # 将 predictions 重新reshape为 (n_samples, -1)
        predictions = predictions.reshape(n_samples, -1)
    # 如果估计器是分类器并且 predictions 的第一维为 2
    elif is_classifier(est) and predictions.shape[0] == 2:
        # 二元分类，形状为 (2, n_instances, n_points)。
        # 输出**正类**的效果
        # 只保留 predictions[1]，即正类的预测结果
        predictions = predictions[1]
        # 将 predictions 重新reshape为 (n_samples, -1)
        predictions = predictions.reshape(n_samples, -1)

    # 将 averaged_predictions 转换为 NumPy 数组，并转置
    averaged_predictions = np.array(averaged_predictions).T
    # 如果估计器是回归器并且 averaged_predictions 的维度为 1
    if is_regressor(est) and averaged_predictions.ndim == 1:
        # 非多输出回归，形状为 (n_points,)
        # 将 averaged_predictions 重新reshape为 (1, -1)
        averaged_predictions = averaged_predictions.reshape(1, -1)
    # 如果估计器是分类器并且 averaged_predictions 的第一维为 2
    elif is_classifier(est) and averaged_predictions.shape[0] == 2:
        # 二元分类，形状为 (2, n_points)。
        # 输出**正类**的效果
        # 只保留 averaged_predictions[1]，即正类的平均预测结果
        averaged_predictions = averaged_predictions[1]
        # 将 averaged_predictions 重新reshape为 (1, -1)
        averaged_predictions = averaged_predictions.reshape(1, -1)

    # 返回重新调整后的 averaged_predictions 和 predictions
    return averaged_predictions, predictions
# 使用 @validate_params 装饰器验证输入参数的合法性，确保以下参数满足特定的类型和方法要求
@validate_params(
    {
        "estimator": [
            HasMethods(["fit", "predict"]),  # estimator 参数必须具有 fit 和 predict 方法
            HasMethods(["fit", "predict_proba"]),  # 或者具有 fit 和 predict_proba 方法
            HasMethods(["fit", "decision_function"]),  # 或者具有 fit 和 decision_function 方法
        ],
        "X": ["array-like", "sparse matrix"],  # X 参数可以是数组形式或稀疏矩阵
        "features": ["array-like", Integral, str],  # features 参数可以是数组形式、整数或字符串
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是数组形式或空值
        "categorical_features": ["array-like", None],  # categorical_features 参数可以是数组形式或空值
        "feature_names": ["array-like", None],  # feature_names 参数可以是数组形式或空值
        "response_method": [StrOptions({"auto", "predict_proba", "decision_function"})],  # response_method 参数必须是 "auto", "predict_proba" 或 "decision_function" 中的一个
        "percentiles": [tuple],  # percentiles 参数必须是元组形式
        "grid_resolution": [Interval(Integral, 1, None, closed="left")],  # grid_resolution 参数必须是大于等于1的整数
        "method": [StrOptions({"auto", "recursion", "brute"})],  # method 参数必须是 "auto", "recursion" 或 "brute" 中的一个
        "kind": [StrOptions({"average", "individual", "both"})],  # kind 参数必须是 "average", "individual" 或 "both" 中的一个
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def partial_dependence(
    estimator,  # 模型估计器对象，必须已经拟合
    X,  # 特征数据，形状为 (n_samples, n_features)
    features,  # 感兴趣的特征或特征集合，可以是数组形式、整数或字符串
    *,  # 后续参数为关键字参数
    sample_weight=None,  # 样本权重，可以是数组形式或空值
    categorical_features=None,  # 类别特征，可以是数组形式或空值
    feature_names=None,  # 特征名称，可以是数组形式或空值
    response_method="auto",  # 响应方法，默认为 "auto"，可以是 "auto", "predict_proba" 或 "decision_function"
    percentiles=(0.05, 0.95),  # 百分位数，必须是元组形式
    grid_resolution=100,  # 网格分辨率，默认为100，必须是大于等于1的整数
    method="auto",  # 方法，默认为 "auto"，可以是 "auto", "recursion" 或 "brute"
    kind="average",  # 类型，默认为 "average"，可以是 "average", "individual" 或 "both"
):
    """Partial dependence of ``features``.

    Partial dependence of a feature (or a set of features) corresponds to
    the average response of an estimator for each possible value of the
    feature.

    Read more in the :ref:`User Guide <partial_dependence>`.

    .. warning::

        For :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, the
        `'recursion'` method (used by default) will not account for the `init`
        predictor of the boosting process. In practice, this will produce
        the same values as `'brute'` up to a constant offset in the target
        response, provided that `init` is a constant estimator (which is the
        default). However, if `init` is not a constant estimator, the
        partial dependence values are incorrect for `'recursion'` because the
        offset will be sample-dependent. It is preferable to use the `'brute'`
        method. Note that this only applies to
        :class:`~sklearn.ensemble.GradientBoostingClassifier` and
        :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like, sparse matrix or dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the partial dependence will be evaluated), and
        also to generate values for the complement features when the
        `method` is 'brute'.

    features : array-like, Integral, str
        The feature or features for which the partial dependence should be computed.

    sample_weight : array-like, None
        Sample weights. If None, samples are equally weighted.

    categorical_features : array-like, None
        Indices or mask indicating categorical features. If provided, these features are treated as categorical.

    feature_names : array-like, None
        Names of features. If None, feature names are automatically generated.

    response_method : {'auto', 'predict_proba', 'decision_function'}, default='auto'
        Method to compute the response of the estimator. 'auto' will infer the appropriate method based on the estimator type.

    percentiles : tuple, default=(0.05, 0.95)
        The percentiles to use for generating grid points.

    grid_resolution : int, default=100
        Number of grid points to use when generating the grid.

    method : {'auto', 'recursion', 'brute'}, default='auto'
        Method used to calculate the partial dependence. 'auto' will select the method automatically based on the estimator type.

    kind : {'average', 'individual', 'both'}, default='average'
        The kind of partial dependence calculation to perform. 'average' computes the average partial dependence across all features, 'individual' computes partial dependence for each feature independently, and 'both' computes both.

    """
    features : array-like of {int, str, bool} or int or str
        # 定义了要计算偏依赖的特征或交互特征对
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    sample_weight : array-like of shape (n_samples,), default=None
        # 样本权重用于计算加权平均值，以平均模型输出。如果为 `None`，则所有样本权重相等。
        # 如果 `sample_weight` 不为 `None`，则 `method` 将被设置为 `'brute'`。
        # 注意，对于 `kind='individual'`，将忽略 `sample_weight`。
        Sample weights are used to calculate weighted means when averaging the
        model output. If `None`, then samples are equally weighted. If
        `sample_weight` is not `None`, then `method` will be set to `'brute'`.
        Note that `sample_weight` is ignored for `kind='individual'`.

        .. versionadded:: 1.3

    categorical_features : array-like of shape (n_features,) or shape \
            (n_categorical_features,), dtype={bool, int, str}, default=None
        # 指定了分类特征。
        # - `None`: 没有特征被视为分类特征；
        # - 布尔数组: 形状为 `(n_features,)` 的布尔掩码，指示哪些特征是分类的。
        #   因此，此数组的形状与 `X.shape[1]` 相同；
        # - 整数或字符串数组: 整数索引或字符串，指示分类特征。
        Indicates the categorical features.

        .. versionadded:: 1.2

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        # 每个特征的名称；`feature_names[i]` 包含索引为 `i` 的特征的名称。
        # 默认情况下，特征的名称对应于它们在 NumPy 数组中的数值索引，以及在 pandas 数据框中的列名。
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.

        .. versionadded:: 1.2

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        # 指定使用 `predict_proba` 还是 `decision_function` 作为目标响应。
        # 对于回归器，此参数将被忽略，响应总是 `predict` 的输出。
        # 默认情况下，首先尝试 `predict_proba`，如果不存在，则返回 `decision_function`。
        # 如果 `method` 是 'recursion'，则响应总是 `decision_function` 的输出。
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.

    percentiles : tuple of float, default=(0.05, 0.95)
        # 用于创建网格极值的下限和上限百分位数。必须在 [0, 1] 内。
        The lower and upper percentile used to create the extreme values
        for the grid. Must be in [0, 1].

    grid_resolution : int, default=100
        # 每个目标特征上网格的等间距点数。
        The number of equally spaced points on the grid, for each target
        feature.
    method : {'auto', 'recursion', 'brute'}, default='auto'
        # 方法选择参数，控制计算平均预测的方法：

        - `'recursion'` 仅支持某些基于树的估计器
          （例如
          :class:`~sklearn.ensemble.GradientBoostingClassifier`,
          :class:`~sklearn.ensemble.GradientBoostingRegressor`,
          :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
          :class:`~sklearn.tree.DecisionTreeRegressor`,
          :class:`~sklearn.ensemble.RandomForestRegressor`）
          当 `kind='average'` 时。
          这种方法在速度上更高效。
          使用此方法时，分类器的目标响应始终是决策函数，而不是预测的概率。
          因为 `'recursion'` 方法通过设计隐式计算个体条件期望（ICE）的平均值，
          所以它与ICE不兼容，因此 `kind` 必须为 `'average'`。

        - `'brute'` 支持任何估计器，但计算密集度更高。

        - `'auto'`：对于支持的估计器，使用 `'recursion'` 方法；否则使用 `'brute'` 方法。
          如果 `sample_weight` 不为 `None`，则无论估计器如何都使用 `'brute'` 方法。

        请参阅 :ref:`this note <pdp_method_differences>` 了解 `'brute'` 和 `'recursion'` 方法的差异。

    kind : {'average', 'individual', 'both'}, default='average'
        # 返回的部分依赖是平均化整个数据集中样本的部分依赖，还是每个样本一个值或两者都有。

        注意，快速的 `method='recursion'` 选项仅适用于 `kind='average'` 和 `sample_weights=None` 的情况。
        计算个体依赖并进行加权平均需要使用较慢的 `method='brute'` 方法。

        .. versionadded:: 0.24

    Returns
    -------
    predictions : :class:`~sklearn.utils.Bunch`
        预测结果对象，类似字典，具有以下属性。

        individual : ndarray of shape (n_outputs, n_instances, \
                len(values[0]), len(values[1]), ...)
            网格中所有点的预测结果，针对X中的所有样本。也称为个体条件期望（ICE）。
            仅在 `kind='individual'` 或 `kind='both'` 时可用。

        average : ndarray of shape (n_outputs, len(values[0]), \
                len(values[1]), ...)
            网格中所有点的预测结果的平均值，对所有X中的样本进行平均（如果 `method` 是 'recursion'，则对训练数据进行平均）。
            仅在 `kind='average'` 或 `kind='both'` 时可用。

        grid_values : seq of 1d ndarrays
            创建网格时使用的值。生成的网格是 `grid_values` 中数组的笛卡尔积，其中 `len(grid_values) == len(features)`。
            每个数组 `grid_values[j]` 的大小为 `grid_resolution` 或 `X[:, j]` 中的唯一值数量中较小的一个。

            .. versionadded:: 1.3

        `n_outputs` 对于多类别设置表示类的数量，对于多输出回归表示任务的数量。
        对于经典回归和二元分类，`n_outputs==1`。
        `n_values_feature_j` 表示 `grid_values[j]` 的大小。

    See Also
    --------
    PartialDependenceDisplay.from_estimator : 绘制偏依赖图。
    PartialDependenceDisplay : 偏依赖可视化。

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])

    """
    # 检查估计器是否已拟合
    check_is_fitted(estimator)

    # 如果估计器不是分类器或回归器，则引发值错误
    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError("'estimator' must be a fitted regressor or classifier.")

    # 如果估计器是分类器并且第一个类别是 NumPy 数组，则引发值错误
    if is_classifier(estimator) and isinstance(estimator.classes_[0], np.ndarray):
        raise ValueError("Multiclass-multioutput estimators are not supported")

    # 仅对列表和其他非数组类型/稀疏数据使用 check_array。不将 DataFrame 转换为 NumPy 数组。
    if not (hasattr(X, "__array__") or sparse.issparse(X)):
        X = check_array(X, force_all_finite="allow-nan", dtype=object)

    # 如果估计器是回归器并且 response_method 不是 "auto"，则引发值错误
    if is_regressor(estimator) and response_method != "auto":
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )
    # 如果 kind 不是 "average"，则根据 method 设置合适的值
    if kind != "average":
        if method == "recursion":
            # 如果 method 是 "recursion" 而 kind 不是 "average"，抛出错误
            raise ValueError(
                "The 'recursion' method only applies when 'kind' is set to 'average'"
            )
        # 否则将 method 设置为 "brute"
        method = "brute"

    # 如果 method 是 "recursion" 且 sample_weight 不为 None，则抛出错误
    if method == "recursion" and sample_weight is not None:
        raise ValueError(
            "The 'recursion' method can only be applied when sample_weight is None."
        )

    # 如果 method 是 "auto"
    if method == "auto":
        # 根据条件设置合适的 method
        if sample_weight is not None:
            method = "brute"
        elif isinstance(estimator, BaseGradientBoosting) and estimator.init is None:
            method = "recursion"
        elif isinstance(
            estimator,
            (BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor),
        ):
            method = "recursion"
        else:
            method = "brute"

    # 如果 method 是 "recursion"
    if method == "recursion":
        # 检查 estimator 是否属于支持的类型，如果不是则抛出错误
        if not isinstance(
            estimator,
            (
                BaseGradientBoosting,
                BaseHistGradientBoosting,
                DecisionTreeRegressor,
                RandomForestRegressor,
            ),
        ):
            # 支持 'recursion' 方法的 estimator 类型列表
            supported_classes_recursion = (
                "GradientBoostingClassifier",
                "GradientBoostingRegressor",
                "HistGradientBoostingClassifier",
                "HistGradientBoostingRegressor",
                "HistGradientBoostingRegressor",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
            )
            raise ValueError(
                "Only the following estimators support the 'recursion' "
                "method: {}. Try using method='brute'.".format(
                    ", ".join(supported_classes_recursion)
                )
            )
        # 如果 response_method 是 "auto"，则设置为 "decision_function"
        if response_method == "auto":
            response_method = "decision_function"

        # 如果 response_method 不是 "decision_function"，则抛出错误
        if response_method != "decision_function":
            raise ValueError(
                "With the 'recursion' method, the response_method must be "
                "'decision_function'. Got {}.".format(response_method)
            )

    # 如果 sample_weight 不为 None，则检查并返回符合要求的 sample_weight
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    # 确定 features 的类型，如果是整数型则进行检查
    if _determine_key_type(features, accept_slice=False) == "int":
        # _get_column_indices() 支持负索引，但这里限制索引必须是正数，并由 _get_column_indices() 进一步检查上限
        if np.any(np.less(features, 0)):
            raise ValueError("all features must be in [0, {}]".format(X.shape[1] - 1))

    # 获取 features 对应的索引数组，并转换为一维数组
    features_indices = np.asarray(
        _get_column_indices(X, features), dtype=np.intp, order="C"
    ).ravel()

    # 检查并返回符合要求的 feature_names
    feature_names = _check_feature_names(X, feature_names)

    # 获取 X 的特征数
    n_features = X.shape[1]

    # 如果 categorical_features 为 None，则初始化 is_categorical 为全 False 数组
    if categorical_features is None:
        is_categorical = [False] * len(features_indices)
    else:
        # 转换为 NumPy 数组，确保是 ndarray 类型
        categorical_features = np.asarray(categorical_features)
        # 检查数组元素类型是否为布尔类型
        if categorical_features.dtype.kind == "b":
            # 当 categorical_features 是布尔数组时
            if categorical_features.size != n_features:
                # 检查布尔数组长度是否与特征数相符合
                raise ValueError(
                    "When `categorical_features` is a boolean array-like, "
                    "the array should be of shape (n_features,). Got "
                    f"{categorical_features.size} elements while `X` contains "
                    f"{n_features} features."
                )
            # 获取与要分析特征对应的布尔值列表
            is_categorical = [categorical_features[idx] for idx in features_indices]
        elif categorical_features.dtype.kind in ("i", "O", "U"):
            # 当 categorical_features 是索引或特征名列表时
            categorical_features_idx = [
                _get_feature_index(cat, feature_names=feature_names)
                for cat in categorical_features
            ]
            # 判断特征是否在分类特征索引中
            is_categorical = [
                idx in categorical_features_idx for idx in features_indices
            ]
        else:
            # 若 categorical_features 类型不符合预期，抛出异常
            raise ValueError(
                "Expected `categorical_features` to be an array-like of boolean,"
                f" integer, or string. Got {categorical_features.dtype} instead."
            )

    # 根据特征索引和百分位数组创建网格
    grid, values = _grid_from_X(
        _safe_indexing(X, features_indices, axis=1),
        percentiles,
        is_categorical,
        grid_resolution,
    )

    if method == "brute":
        # 使用 brute 方法计算偏依赖
        averaged_predictions, predictions = _partial_dependence_brute(
            estimator, grid, features_indices, X, response_method, sample_weight
        )

        # 调整预测结果形状为 (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
        predictions = predictions.reshape(
            -1, X.shape[0], *[val.shape[0] for val in values]
        )
    else:
        # 使用递归方法计算偏依赖
        averaged_predictions = _partial_dependence_recursion(
            estimator, grid, features_indices
        )

    # 调整平均预测结果的形状为 (n_outputs, n_values_feature_0, n_values_feature_1, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )
    # 创建包含网格值的 Bunch 对象
    pdp_results = Bunch(grid_values=values)

    if kind == "average":
        # 若 kind 为 average，存储平均预测结果
        pdp_results["average"] = averaged_predictions
    elif kind == "individual":
        # 若 kind 为 individual，存储每个实例的预测结果
        pdp_results["individual"] = predictions
    else:  # kind='both'
        # 若 kind 为 both，同时存储平均预测结果和每个实例的预测结果
        pdp_results["average"] = averaged_predictions
        pdp_results["individual"] = predictions

    # 返回偏依赖分析结果
    return pdp_results
```