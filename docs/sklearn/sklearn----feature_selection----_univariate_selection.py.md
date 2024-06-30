# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_univariate_selection.py`

```
"""Univariate features selection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入整数和实数类型
from numbers import Integral, Real

# 导入科学计算库 NumPy 和 SciPy 的特殊函数和统计模块
import numpy as np
from scipy import special, stats
# 导入处理稀疏矩阵的方法
from scipy.sparse import issparse

# 导入基础估计器、拟合上下文、标签二值化器、数组类型转换、数组检查、数据集检查、安全遮罩、安全平方方法、参数验证方法、行向量范数计算、稀疏矩阵乘法、检查拟合情况的方法和特征选择混合器
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin


# 定义一个函数来清除 NaN 值
def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # 将 scores 转换为浮点数数组，并复制一份以防止原数组改变
    scores = as_float_array(scores, copy=True)
    # 将 scores 中的 NaN 值替换为该类型中的最小值
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


######################################################################
# Scoring functions


# 下面的函数是对 scipy.stats.f_oneway 的重新实现
# 与 scipy.stats.f_oneway 不同的是，它在不改变输入的情况下不复制数据
def f_oneway(*args):
    """Perform a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    *args : {array-like, sparse matrix}
        Sample1, sample2... The sample measurements should be given as
        arguments.

    Returns
    -------
    f_statistic : float
        The computed F-value of the test.
    p_value : float
        The associated p-value from the F-distribution.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal. This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`_) although
    with some loss of power.

    The algorithm is from Heiman[2], pp.394-7.

    See ``scipy.stats.f_oneway`` that should give the same results while
    being less efficient.

    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 14.
           http://vassarstats.net/textbook

    .. [2] Heiman, G.W.  Research Methods in Statistics. 2002.
    """
    # 获取样本数
    n_classes = len(args)
    # 将输入的参数转换为浮点数数组
    args = [as_float_array(a) for a in args]
    # 计算每个类别样本数，并创建一个包含这些数目的 NumPy 数组
    n_samples_per_class = np.array([a.shape[0] for a in args])
    
    # 计算总样本数
    n_samples = np.sum(n_samples_per_class)
    
    # 计算所有类别样本各特征平方和的总和
    ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
    
    # 计算每个类别样本各特征的和并转换为 NumPy 数组
    sums_args = [np.asarray(a.sum(axis=0)) for a in args]
    
    # 计算所有类别样本各特征的和的平方和
    square_of_sums_alldata = sum(sums_args) ** 2
    
    # 计算总体总离差平方和
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    
    # 初始化类别内离差平方和
    ssbn = 0.0
    
    # 计算类别内离差平方和
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    
    # 计算类别间离差平方和
    sswn = sstot - ssbn
    
    # 计算类别间自由度
    dfbn = n_classes - 1
    
    # 计算类别内自由度
    dfwn = n_samples - n_classes
    
    # 计算类别间均方（mean square between）
    msb = ssbn / float(dfbn)
    
    # 计算类别内均方（mean square within）
    msw = sswn / float(dfwn)
    
    # 找出均方内为零的常数特征索引
    constant_features_idx = np.where(msw == 0.0)[0]
    
    # 如果类别间均方不为零且存在常数特征，则发出警告
    if np.nonzero(msb)[0].size != msb.size and constant_features_idx.size:
        warnings.warn("Features %s are constant." % constant_features_idx, UserWarning)
    
    # 计算 F 统计量
    f = msb / msw
    
    # 将 F 统计量展平为向量，适用于稀疏情况
    f = np.asarray(f).ravel()
    
    # 计算 F 分布的累积分布函数值
    prob = special.fdtrc(dfbn, dfwn, f)
    
    # 返回 F 统计量向量和其对应的概率值
    return f, prob
# 使用 @validate_params 装饰器对函数进行参数验证，确保输入的 X 和 y 符合要求
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
# 定义函数 f_classif，计算给定样本的ANOVA F值
def f_classif(X, y):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : array-like of shape (n_samples,)
        The target vector.

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import f_classif
    >>> X, y = make_classification(
    ...     n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
    ...     shuffle=False, random_state=42
    ... )
    >>> f_statistic, p_values = f_classif(X, y)
    >>> f_statistic
    array([2.2...e+02, 7.0...e-01, 1.6...e+00, 9.3...e-01,
           5.4...e+00, 3.2...e-01, 4.7...e-02, 5.7...e-01,
           7.5...e-01, 8.9...e-02])
    >>> p_values
    array([7.1...e-27, 4.0...e-01, 1.9...e-01, 3.3...e-01,
           2.2...e-02, 5.7...e-01, 8.2...e-01, 4.5...e-01,
           3.8...e-01, 7.6...e-01])
    """
    # 调用 sklearn.utils.validation 中的 check_X_y 函数验证输入的 X 和 y，并接受稀疏矩阵格式
    X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"])
    # 对每个类别 k，使用 safe_mask 函数获取安全掩码，然后从 X 中选择相应的数据，组成 args 列表
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    # 调用 scipy.stats.f_oneway 对 args 列表中的数据执行单因素方差分析，并返回 F 值和对应的 p 值
    return f_oneway(*args)
    most likely to be independent of class and therefore irrelevant for
    classification.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.

    y : array-like of shape (n_samples,)
        Target vector (class labels).

    Returns
    -------
    chi2 : ndarray of shape (n_features,)
        Chi2 statistics for each feature.

    p_values : ndarray of shape (n_features,)
        P-values for each feature.

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    f_regression : F-value between label/feature for regression tasks.

    Notes
    -----
    Complexity of this algorithm is O(n_classes * n_features).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_selection import chi2
    >>> X = np.array([[1, 1, 3],
    ...               [0, 1, 5],
    ...               [5, 4, 1],
    ...               [6, 6, 2],
    ...               [1, 4, 0],
    ...               [0, 0, 0]])
    >>> y = np.array([1, 1, 0, 0, 2, 2])
    >>> chi2_stats, p_values = chi2(X, y)
    >>> chi2_stats
    array([15.3...,  6.5       ,  8.9...])
    >>> p_values
    array([0.0004..., 0.0387..., 0.0116... ])
    """

    # XXX: we might want to do some of the following in logspace instead for
    # numerical stability.
    # Converting X to float allows getting better performance for the
    # safe_sparse_dot call made below.
    X = check_array(X, accept_sparse="csr", dtype=(np.float64, np.float32))
    # 检查并转换 X 为浮点数类型，以便在下面的 safe_sparse_dot 调用中获得更好的性能
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")
    # 如果 X 中有任何负数，则抛出异常

    # Use a sparse representation for Y by default to reduce memory usage when
    # y has many unique classes.
    Y = LabelBinarizer(sparse_output=True).fit_transform(y)
    # 使用稀疏表示 Y，默认情况下以减少内存使用，当 y 有许多唯一类别时
    if Y.shape[1] == 1:
        Y = Y.toarray()
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features
    # 计算观察值，使用 safe_sparse_dot 进行矩阵乘法

    if issparse(observed):
        # convert back to a dense array before calling _chisquare
        # XXX: could _chisquare be reimplement to accept sparse matrices for
        # cases where both n_classes and n_features are large (and X is
        # sparse)?
        observed = observed.toarray()
    # 如果 observed 是稀疏矩阵，则转换为密集数组

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return _chisquare(observed, expected)
# 使用装饰器验证函数参数，确保参数 X 和 y 符合指定的数据类型要求，接受稀疏矩阵格式
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "y": ["array-like"],  # 参数 y 应为数组
        "center": ["boolean"],  # 参数 center 应为布尔类型
        "force_finite": ["boolean"],  # 参数 force_finite 应为布尔类型
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
def r_regression(X, y, *, center=True, force_finite=True):
    """Compute Pearson's r for each features and the target.

    Pearson's r is also known as the Pearson correlation coefficient.

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature selection procedure, not
    a free standing feature selection procedure.

    The cross correlation between each regressor and the target is computed
    as::

        E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    force_finite : bool, default=True
        Whether or not to force the Pearson's R correlation to be finite.
        In the particular case where some features in `X` or the target `y`
        are constant, the Pearson's R correlation is not defined. When
        `force_finite=False`, a correlation of `np.nan` is returned to
        acknowledge this case. When `force_finite=True`, this value will be
        forced to a minimal correlation of `0.0`.

        .. versionadded:: 1.1

    Returns
    -------
    correlation_coefficient : ndarray of shape (n_features,)
        Pearson's R correlation coefficients of features.

    See Also
    --------
    f_regression: Univariate linear regression tests returning f-statistic
        and p-values.
    mutual_info_regression: Mutual information for a continuous target.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.feature_selection import r_regression
    >>> X, y = make_regression(
    ...     n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
    ... )
    >>> r_regression(X, y)
    array([-0.15...,  1.        , -0.22...])
    """
    # 检查并确保输入的 X 和 y 符合要求，将它们转换为特定的稀疏矩阵格式（csr/csc/coo），并且数据类型为 np.float64
    X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"], dtype=np.float64)
    n_samples = X.shape[0]

    # Compute centered values
    # Note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    # 计算中心化后的数值
    # 注意到 E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))]，因此我们不需要对 X 进行中心化处理
    # 如果需要对数据进行居中处理
    if center:
        # 将目标变量 y 居中化（减去均值）
        y = y - np.mean(y)
        
        # TODO: 对于 Scipy <= 1.10，`isspmatrix(X)` 对稀疏数组返回 `True`。
        # 这里我们检查 `.mean` 操作的输出，对稀疏矩阵返回 `np.matrix`，
        # 而对密集和稀疏数组返回 `np.array`。
        # 当 SciPy >= 1.11 时，可以重新考虑使用 `isspmatrix`。
        
        # 计算 X 的列均值
        X_means = X.mean(axis=0)
        
        # 如果 X_means 是 np.matrix 类型，则转换为 1 维数组
        X_means = X_means.getA1() if isinstance(X_means, np.matrix) else X_means
        
        # 通过矩阵的方法计算标准差的缩放值
        X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means**2)
    else:
        # 如果不需要居中处理，则直接计算 X 的行标准差
        X_norms = row_norms(X.T)

    # 计算 y 和 X 之间的相关系数
    correlation_coefficient = safe_sparse_dot(y, X)
    
    # 忽略计算过程中的除零和无效值警告
    with np.errstate(divide="ignore", invalid="ignore"):
        # 将相关系数除以 X 的标准差
        correlation_coefficient /= X_norms
        # 将相关系数除以 y 的 L2 范数
        correlation_coefficient /= np.linalg.norm(y)

    # 如果 force_finite 为 True，并且相关系数中有非有限的值
    if force_finite and not np.isfinite(correlation_coefficient).all():
        # 处理目标变量或某些特征恒定的情况
        # 将相关系数中的 NaN 值设为最小值（即 0.0）
        nan_mask = np.isnan(correlation_coefficient)
        correlation_coefficient[nan_mask] = 0.0
    
    # 返回计算得到的相关系数
    return correlation_coefficient
# 使用装饰器 @validate_params 对函数进行参数验证，确保输入参数符合指定的类型和条件
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为类数组或稀疏矩阵类型
        "y": ["array-like"],                  # 参数 y 应为类数组类型
        "center": ["boolean"],                # 参数 center 应为布尔类型
        "force_finite": ["boolean"],          # 参数 force_finite 应为布尔类型
    },
    prefer_skip_nested_validation=True,       # 设置为 True，优先跳过嵌套验证
)
# 定义函数 f_regression，执行单变量线性回归检验，返回 F 统计量和 p 值
def f_regression(X, y, *, center=True, force_finite=True):
    """Univariate linear regression tests returning F-statistic and p-values.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 2 steps:

    1. The cross correlation between each regressor and the target is computed
       using :func:`r_regression` as::

           E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

    2. It is converted to an F score and then to a p-value.

    :func:`f_regression` is derived from :func:`r_regression` and will rank
    features in the same order if all the features are positively correlated
    with the target.

    Note however that contrary to :func:`f_regression`, :func:`r_regression`
    values lie in [-1, 1] and can thus be negative. :func:`f_regression` is
    therefore recommended as a feature selection criterion to identify
    potentially predictive feature for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Furthermore :func:`f_regression` returns p-values while
    :func:`r_regression` does not.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    force_finite : bool, default=True
        Whether or not to force the F-statistics and associated p-values to
        be finite. There are two cases where the F-statistic is expected to not
        be finite:

        - when the target `y` or some features in `X` are constant. In this
          case, the Pearson's R correlation is not defined leading to obtain
          `np.nan` values in the F-statistic and p-value. When
          `force_finite=True`, the F-statistic is set to `0.0` and the
          associated p-value is set to `1.0`.
        - when a feature in `X` is perfectly correlated (or
          anti-correlated) with the target `y`. In this case, the F-statistic
          is expected to be `np.inf`. When `force_finite=True`, the F-statistic
          is set to `np.finfo(dtype).max` and the associated p-value is set to
          `0.0`.

        .. versionadded:: 1.1

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    """
    # 计算回归任务中标签和特征之间的Pearson相关系数R
    r_regression: Pearson's R between label/feature for regression tasks.
    # 计算分类任务中标签和特征之间的ANOVA F值
    f_classif: ANOVA F-value between label/feature for classification tasks.
    # 计算非负特征在分类任务中的卡方统计量
    chi2: Chi-squared stats of non-negative features for classification tasks.
    # 根据最高分数选择特征的类，k是指定的数量
    SelectKBest: Select features based on the k highest scores.
    # 根据假阳性率测试选择特征
    SelectFpr: Select features based on a false positive rate test.
    # 根据估计的假发现率选择特征
    SelectFdr: Select features based on an estimated false discovery rate.
    # 根据家庭误差率选择特征
    SelectFwe: Select features based on family-wise error rate.
    # 根据最高分数的百分位数选择特征
    SelectPercentile: Select features based on percentile of the highest
        scores.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.feature_selection import f_regression
    >>> X, y = make_regression(
    ...     n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
    ... )
    >>> f_statistic, p_values = f_regression(X, y)
    >>> f_statistic
    array([1.2...+00, 2.6...+13, 2.6...+00])
    >>> p_values
    array([2.7..., 1.5..., 1.0...])
    """
    # 计算Pearson相关系数的平方
    correlation_coefficient = r_regression(
        X, y, center=center, force_finite=force_finite
    )
    # 自由度的度数，y的大小减去1或2（如果有中心值）
    deg_of_freedom = y.size - (2 if center else 1)

    # 相关系数的平方
    corr_coef_squared = correlation_coefficient**2

    # 处理除法和无效错误，计算F统计量和p值
    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
        p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)

    # 如果强制有限，并且F统计量有非有限值
    if force_finite and not np.isfinite(f_statistic).all():
        # 处理完全（反）相关的情况
        # 将F统计量设置为最大值，p值设置为零
        mask_inf = np.isinf(f_statistic)
        f_statistic[mask_inf] = np.finfo(f_statistic.dtype).max
        # 处理目标或某些特征是常数的情况
        # 将F统计量设置为最小值，p值设置为一
        mask_nan = np.isnan(f_statistic)
        f_statistic[mask_nan] = 0.0
        p_values[mask_nan] = 1.0
    # 返回计算的F统计量和p值
    return f_statistic, p_values
######################################################################
# Base classes


class _BaseFilter(SelectorMixin, BaseEstimator):
    """初始化单变量特征选择器。

    Parameters
    ----------
    score_func : callable
        接受两个数组 X 和 y 作为参数，并返回一对数组(scores, pvalues)或只返回分数数组的函数。
    """

    _parameter_constraints: dict = {"score_func": [callable]}

    def __init__(self, score_func):
        self.score_func = score_func

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """对 (X, y) 运行评分函数，并获取适当的特征。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练输入样本。

        y : array-like of shape (n_samples,) or None
            目标值（分类任务中的类标签，回归任务中的实数）。如果选择器是无监督的，则可以将 y 设置为 None。

        Returns
        -------
        self : object
            返回实例本身。
        """
        if y is None:
            X = self._validate_data(X, accept_sparse=["csr", "csc"])
        else:
            X, y = self._validate_data(
                X, y, accept_sparse=["csr", "csc"], multi_output=True
            )

        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        self.scores_ = np.asarray(self.scores_)

        return self

    def _check_params(self, X, y):
        pass

    def _more_tags(self):
        return {"requires_y": True}


######################################################################
# Specific filters
######################################################################
class SelectPercentile(_BaseFilter):
    """根据最高分数的百分位数选择特征。

    详细信息请参阅 :ref:`用户指南 <univariate_feature_selection>`。

    Parameters
    ----------
    score_func : callable, default=f_classif
        接受两个数组 X 和 y 作为参数，并返回一对数组(scores, pvalues)或只返回分数数组的函数。
        默认为 f_classif（参见下文中的“参见”部分）。默认函数仅适用于分类任务。

        .. versionadded:: 0.18

    percentile : int, default=10
        要保留的特征百分比。

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        特征的分数。

    pvalues_ : array-like of shape (n_features,)
        特征分数的 p 值，如果 `score_func` 仅返回分数，则为 None。
    """
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    This filter supports unsupervised feature selection that only requests `X` for
    computing the scores.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectPercentile, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    >>> X_new.shape
    (1797, 7)
    """

    # 定义参数约束字典，扩展自_BaseFilter._parameter_constraints
    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "percentile": [Interval(Real, 0, 100, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, percentile=10):
        # 调用父类的初始化方法，设定评分函数和百分位数
        super().__init__(score_func=score_func)
        self.percentile = percentile

    def _get_support_mask(self):
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 处理百分位数为100%时的情况
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=bool)
        # 处理百分位数为0%时的情况
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=bool)

        # 清理NaN值后的得分
        scores = _clean_nans(self.scores_)
        # 根据百分位数计算阈值
        threshold = np.percentile(scores, 100 - self.percentile)
        # 创建布尔掩码，选择高于阈值的特征
        mask = scores > threshold
        # 处理得分相同的特征
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[: max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask

    def _more_tags(self):
        # 返回附加标签，指定此方法不需要y（目标值）
        return {"requires_y": False}
class SelectKBest(_BaseFilter):
    """Select features according to the k highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

        .. versionadded:: 0.18

    k : int or "all", default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    This filter supports unsupervised feature selection that only requests `X` for
    computing the scores.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    >>> X_new.shape
    (1797, 20)
    """

    # 设置参数约束字典，扩展自基类的参数约束
    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "k": [StrOptions({"all"}), Interval(Integral, 0, None, closed="left")],
    }

    def __init__(self, score_func=f_classif, *, k=10):
        # 调用基类的构造函数，传入评分函数作为参数
        super().__init__(score_func=score_func)
        # 初始化 k 属性
        self.k = k
    # 检查参数是否有效，确保 k 是字符串类型或者小于等于特征数目
    def _check_params(self, X, y):
        if not isinstance(self.k, str) and self.k > X.shape[1]:
            # 如果 k 大于特征数目，发出警告并返回所有特征
            warnings.warn(
                f"k={self.k} is greater than n_features={X.shape[1]}. "
                "All the features will be returned."
            )
    
    # 获取支持掩码
    def _get_support_mask(self):
        # 检查模型是否已拟合
        check_is_fitted(self)
    
        if self.k == "all":
            # 如果 k 等于 "all"，返回一个全为真的掩码
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            # 如果 k 等于 0，返回一个全为假的掩码
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)
    
            # 请求稳定排序。在 x86-64 平台上，mergesort 需要更多内存（大约每兆特征需要 40MB）。
            # 对分数进行排序，并选取分数最高的 k 个特征，将相应位置的掩码设为真
            mask[np.argsort(scores, kind="mergesort")[-self.k :]] = 1
            return mask
    
    # 返回更多的标签信息，表明模型不需要 y（目标变量）
    def _more_tags(self):
        return {"requires_y": False}
# 继承自 _BaseFilter 类的 SelectFpr 特征选择器类，用于基于 FPR 测试选择 alpha 以下的 p 值特征

class SelectFpr(_BaseFilter):
    """Filter: Select the pvalues below alpha based on a FPR test.

    FPR test stands for False Positive Rate test. It controls the total
    amount of false detections.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        Features with p-values less than `alpha` are selected.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFpr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)
    """

    # 参数约束字典，包含 alpha 参数的约束范围 [0, 1]
    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "alpha": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        # 调用父类构造函数，初始化 score_func
        super().__init__(score_func=score_func)
        # 设定 alpha 值
        self.alpha = alpha

    def _get_support_mask(self):
        # 检查是否已拟合模型
        check_is_fitted(self)

        # 返回 p 值小于 alpha 的布尔掩码
        return self.pvalues_ < self.alpha


# SelectFdr 类的定义未完整提供，可能包含后续的属性和方法定义
class SelectFdr(_BaseFilter):
    """Filter: Select the p-values for an estimated false discovery rate.

    This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
    on the expected false discovery rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.
    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        The highest uncorrected p-value for features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    References
    ----------
    https://en.wikipedia.org/wiki/False_discovery_rate

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFdr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)
    """

    # Define parameter constraints, extending those from _BaseFilter class
    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "alpha": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        # Initialize the SelectFdr object with the specified score function and alpha value
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        # Check if the estimator has been fitted
        check_is_fitted(self)

        # Determine the number of features
        n_features = len(self.pvalues_)

        # Sort p-values and compute the threshold for feature selection
        sv = np.sort(self.pvalues_)
        selected = sv[
            sv <= float(self.alpha) / n_features * np.arange(1, n_features + 1)
        ]

        # If no features are selected, return an array of False values
        if selected.size == 0:
            return np.zeros_like(self.pvalues_, dtype=bool)

        # Return a mask indicating which features are selected based on p-value threshold
        return self.pvalues_ <= selected.max()
class SelectFwe(_BaseFilter):
    """Filter: Select the p-values corresponding to Family-wise error rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        The highest uncorrected p-value for features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFwe, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 15)
    """

    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "alpha": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        # Ensure the estimator is fitted
        check_is_fitted(self)
        
        # Return a boolean mask indicating which features to keep based on the p-values
        return self.pvalues_ < self.alpha / len(self.pvalues_)


######################################################################
# Generic filter
######################################################################

# TODO this class should fit on either p-values or scores,
# depending on the mode.
class GenericUnivariateSelect(_BaseFilter):
    """Univariate feature selector with configurable strategy.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    # score_func : callable, default=f_classif
    #     定义评分函数，可调用对象，默认为 f_classif，用于评估特征重要性并返回得分和 p 值。
    # mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}, default='percentile'
    #     特征选择模式。注意，'percentile' 和 'k_best' 模式支持无监督特征选择（当 `y` 为 `None` 时）。
    # param : "all", float or int, default=1e-5
    #     对应模式的参数值。可以是 "all"，浮点数或整数，默认为 1e-5。
    # 
    # Attributes
    # ----------
    # scores_ : array-like of shape (n_features,)
    #     特征的得分。
    # 
    # pvalues_ : array-like of shape (n_features,)
    #     特征得分的 p 值，如果 `score_func` 只返回得分则为 None。
    # 
    # n_features_in_ : int
    #     在拟合期间看到的特征数量。
    # 
    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在拟合期间看到的特征名称。仅当 `X` 的特征名称全部为字符串时定义。
    # 
    # See Also
    # --------
    # f_classif : 用于分类任务中标签/特征的ANOVA F值。
    # mutual_info_classif : 用于离散目标的互信息。
    # chi2 : 用于分类任务中非负特征的卡方统计量。
    # f_regression : 用于回归任务中标签/特征的F值。
    # mutual_info_regression : 用于连续目标的互信息。
    # SelectPercentile : 基于最高分的百分位数选择特征。
    # SelectKBest : 基于最高分的 k 个选择特征。
    # SelectFpr : 基于假阳率测试选择特征。
    # SelectFdr : 基于估计的虚假发现率选择特征。
    # SelectFwe : 基于家族错误率选择特征。
    # 
    # Examples
    # --------
    # >>> from sklearn.datasets import load_breast_cancer
    # >>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
    # >>> X, y = load_breast_cancer(return_X_y=True)
    # >>> X.shape
    # (569, 30)
    # >>> transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
    # >>> X_new = transformer.fit_transform(X, y)
    # >>> X_new.shape
    # (569, 20)
    ```
    # 创建并返回一个选择器对象，根据当前的模式和评分函数初始化
    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)

        # 现在执行一些操作以设置选择器中正确的命名参数
        # 获取选择器可能的参数名列表
        possible_params = selector._get_param_names()
        # 移除掉不需要设置的参数名"score_func"
        possible_params.remove("score_func")
        # 使用self.param设置剩余参数的值，通过关键字参数方式
        selector.set_params(**{possible_params[0]: self.param})

        # 返回配置好的选择器对象
        return selector

    # 返回一个字典，指定数据类型的保留情况为浮点数（np.float64和np.float32）
    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

    # 检查参数的有效性，委托给内部生成的选择器对象来进行参数检查
    def _check_params(self, X, y):
        self._make_selector()._check_params(X, y)

    # 返回一个布尔掩码数组，指示哪些特征被选择器选择为重要特征
    def _get_support_mask(self):
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 创建选择器对象
        selector = self._make_selector()
        # 将预先计算的p值和得分设置到选择器对象中
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        # 返回选择器对象计算的重要特征掩码
        return selector._get_support_mask()
```