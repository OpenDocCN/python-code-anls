# `D:\src\scipysrc\scikit-learn\sklearn\impute\_iterative.py`

```
import warnings
from collections import namedtuple  # 导入namedtuple模块，用于创建命名元组
from numbers import Integral, Real  # 导入Integral和Real类，用于数值类型检查
from time import time  # 导入time函数，用于时间操作

import numpy as np  # 导入NumPy库，用于数值计算
from scipy import stats  # 导入SciPy的统计模块

from ..base import _fit_context, clone  # 从当前包的base模块导入_fit_context和clone函数
from ..exceptions import ConvergenceWarning  # 从当前包的exceptions模块导入ConvergenceWarning异常
from ..preprocessing import normalize  # 从当前包的preprocessing模块导入normalize函数
from ..utils import _safe_indexing, check_array, check_random_state  # 从当前包的utils模块导入_safe_indexing, check_array, check_random_state函数
from ..utils._indexing import _safe_assign  # 从当前包的utils._indexing模块导入_safe_assign函数
from ..utils._mask import _get_mask  # 从当前包的utils._mask模块导入_get_mask函数
from ..utils._missing import is_scalar_nan  # 从当前包的utils._missing模块导入is_scalar_nan函数
from ..utils._param_validation import HasMethods, Interval, StrOptions  # 从当前包的utils._param_validation模块导入HasMethods, Interval, StrOptions类
from ..utils.metadata_routing import (  # 从当前包的utils.metadata_routing模块导入以下内容：
    MetadataRouter,  # MetadataRouter类
    MethodMapping,  # MethodMapping类
    _raise_for_params,  # _raise_for_params函数
    process_routing,  # process_routing函数
)
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted  # 从当前包的utils.validation模块导入FLOAT_DTYPES, _check_feature_names_in, check_is_fitted函数
from ._base import SimpleImputer, _BaseImputer, _check_inputs_dtype  # 从当前包的_base模块导入SimpleImputer, _BaseImputer, _check_inputs_dtype类和函数

_ImputerTriplet = namedtuple(  # 创建命名元组 _ImputerTriplet，包含 feat_idx, neighbor_feat_idx, estimator 字段
    "_ImputerTriplet", ["feat_idx", "neighbor_feat_idx", "estimator"]
)


def _assign_where(X1, X2, cond):
    """Assign X2 to X1 where cond is True.

    Parameters
    ----------
    X1 : ndarray or dataframe of shape (n_samples, n_features)
        Data.

    X2 : ndarray of shape (n_samples, n_features)
        Data to be assigned.

    cond : ndarray of shape (n_samples, n_features)
        Boolean mask to assign data.
    """
    if hasattr(X1, "mask"):  # 如果X1对象具有mask属性，适用于pandas数据帧
        X1.mask(cond=cond, other=X2, inplace=True)
    else:  # 如果是ndarray类型
        X1[cond] = X2[cond]


class IterativeImputer(_BaseImputer):
    """Multivariate imputer that estimates each feature from all the others.

    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.

    Read more in the :ref:`User Guide <iterative_imputer>`.

    .. versionadded:: 0.21

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import `enable_iterative_imputer`::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer  # noqa
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    """
    # 是否从拟合估计器的（高斯）预测后验中采样来进行每次插补
    # 如果设置为 True，则估计器的 `predict` 方法必须支持 `return_std`。如果使用 `IterativeImputer` 进行多次插补，则应设置为 `True`。
    sample_posterior : bool, default=False
    
    # 在返回最终轮次计算的插补之前，执行的最大插补轮次数
    # 一轮是对每个具有缺失值的特征进行一次插补。停止标准是 `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`，其中 `X_t` 是第 `t` 次迭代时的 `X`。注意，仅当 `sample_posterior=False` 时才应用早期停止。
    max_iter : int, default=10
    
    # 停止条件的容差
    tol : float, default=1e-3
    
    # 用于估计每个特征列的缺失值的其他特征数
    # 特征之间的接近性使用初始插补后的每对特征的绝对相关系数来衡量。为确保在整个插补过程中覆盖特征，邻居特征不一定是最接近的，而是按比例于每个被插补目标特征的相关性来抽取。
    # 当特征数量巨大时，可以显著加快速度。如果为 `None`，则使用所有特征。
    n_nearest_features : int, default=None
    
    # 初始化缺失值的策略
    # 与 :class:`~sklearn.impute.SimpleImputer` 中的 `strategy` 参数相同。
    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
    
    # 当 `strategy="constant"` 时，用于替换所有缺失值的值
    # 对于字符串或对象数据类型，`fill_value` 必须是一个字符串。
    # 如果为 `None`，当插补数值数据时，`fill_value` 将为 0，对于字符串或对象数据类型将为 "missing_value"。
    fill_value : str or numerical value, default=None
    
    # 插补特征的顺序
    # 可能的值有：
    # - `'ascending'`：从缺失值最少的特征到最多的特征。
    # - `'descending'`：从缺失值最多的特征到最少的特征。
    # - `'roman'`：从左到右。
    # - `'arabic'`：从右到左。
    # - `'random'`：每轮随机顺序。
    imputation_order : {'ascending', 'descending', 'roman', 'arabic', 'random'}, default='ascending'
    # 是否跳过在 `transform` 时有缺失值但在 `fit` 时没有缺失值的特征，仅使用初始填充方法进行填充。
    # 如果在 `fit` 和 `transform` 时都有许多没有缺失值的特征，则设置为 `True` 可以节省计算资源。
    skip_complete : bool, default=False

    # 最小可能的填充值。如果是标量，则广播到 `(n_features,)` 形状。
    # 如果是数组形式，则期望形状为 `(n_features,)`，每个特征一个最小值。
    # 默认为 `-np.inf`。
    # 
    # .. versionchanged:: 0.23
    #    增加了对数组形式的支持。
    min_value : float or array-like of shape (n_features,), default=-np.inf

    # 最大可能的填充值。如果是标量，则广播到 `(n_features,)` 形状。
    # 如果是数组形式，则期望形状为 `(n_features,)`，每个特征一个最大值。
    # 默认为 `np.inf`。
    #
    # .. versionchanged:: 0.23
    #    增加了对数组形式的支持。
    max_value : float or array-like of shape (n_features,), default=np.inf

    # 冗长模式标志，控制评估函数时发出的调试消息。
    # 数字越高，越冗长。可以为 0、1 或 2。
    verbose : int, default=0

    # 伪随机数生成器的种子，用于控制随机性。
    # 如果 `n_nearest_features` 不为 `None`，则随机选择估算器特征。
    # 如果 `imputation_order` 是 `random`，或者 `sample_posterior=True` 时，从后验中抽样。
    # 使用整数以确保确定性。
    # 详见 :term:`术语表 <random_state>`。
    random_state : int, RandomState instance or None, default=None

    # 如果为 `True`，则在填充器的转换输出上叠加一个 :class:`MissingIndicator` 转换。
    # 这允许预测估算器在填充的同时考虑缺失情况。
    # 如果一个特征在 `fit/train` 时没有缺失值，则即使在 `transform/test` 时存在缺失值，该特征也不会出现在缺失指示器上。
    add_indicator : bool, default=False

    # 如果为 True，在调用 `transform` 时返回在 `fit` 时完全由缺失值组成的特征。
    # 填充值始终为 `0`，除非 `initial_strategy="constant"`，在这种情况下将使用 `fill_value`。
    #
    # .. versionadded:: 1.2
    keep_empty_features : bool, default=False

Attributes
----------
    # 用于初始化缺失值的填充器的类型为 :class:`~sklearn.impute.SimpleImputer` 的对象。
    initial_imputer_ : object of type :class:`~sklearn.impute.SimpleImputer`
    imputation_sequence_ : list of tuples
        # 用于存储每个特征的填充顺序和相关信息的列表
        Each tuple has `(feat_idx, neighbor_feat_idx, estimator)`, where
        # 每个元组包含 `(feat_idx, neighbor_feat_idx, estimator)`，其中
        `feat_idx` is the current feature to be imputed,
        # `feat_idx` 是当前需要填充的特征索引
        `neighbor_feat_idx` is the array of other features used to impute the
        # `neighbor_feat_idx` 是用于填充当前特征的其他特征数组
        current feature, and `estimator` is the trained estimator used for
        # `estimator` 是用于填充的训练好的估计器
        the imputation. Length is `self.n_features_with_missing_ *
        # 长度为 `self.n_features_with_missing_ * self.n_iter_`
        self.n_iter_`.

    n_iter_ : int
        # 迭代的次数
        Number of iteration rounds that occurred. Will be less than
        # 已经进行的迭代轮数。如果达到了早停止条件，将少于 `self.max_iter`
        `self.max_iter` if early stopping criterion was reached.

    n_features_in_ : int
        # 在 `fit` 过程中观察到的特征数量
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        # 在 `fit` 过程中观察到的特征的名称数组，仅当 `X` 的特征名称全部为字符串时定义

        .. versionadded:: 1.0

    n_features_with_missing_ : int
        # 具有缺失值的特征的数量

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        # 用于添加缺失值二进制指示器的指示器对象。如果 `add_indicator=False`，则为 `None`

    random_state_ : RandomState instance
        # 生成的随机状态实例，可以是从种子、随机数生成器或者 `np.random` 生成的

    See Also
    --------
    SimpleImputer : Univariate imputer for completing missing values
        with simple strategies.
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    Notes
    -----
    # 支持归纳模式中的填充，我们在 `fit` 阶段存储每个特征的估计器，并在 `transform` 阶段无需重新拟合按顺序预测
    To support imputation in inductive mode we store each feature's estimator
    # 当 `fit` 阶段存在所有缺失值的特征时，在 `transform` 阶段将被丢弃

    # 使用默认设置，该填充器的缩放为 `O(knp^3 * min(n,p))`，其中 `k` = `max_iter`，`n` 是样本数，`p` 是特征数。
    # 因此在特征数量增加时成本极高。可以通过设置 `n_nearest_features << n_features`、`skip_complete=True` 或增加 `tol` 来减少计算成本。

    Depending on the nature of missing values, simple imputers can be
    preferable in a prediction context.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_

    .. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
        Multivariate Data Suitable for use with an Electronic Computer".
        Journal of the Royal Statistical Society 22(2): 302-306.
        <https://www.jstor.org/stable/2984099>`_

    Examples
    --------
    >>> import numpy as np
    # 启用 sklearn 中的迭代式填补器功能
    from sklearn.experimental import enable_iterative_imputer
    # 导入迭代式填补器类
    from sklearn.impute import IterativeImputer

    # 创建一个迭代式填补器对象，并设定随机种子为 0
    imp_mean = IterativeImputer(random_state=0)

    # 对给定的样本数据进行拟合，用于生成填补器的内部模型
    imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

    # 定义一个输入特征数组 X，其中包含缺失值 np.nan
    X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]

    # 使用已经拟合好的填补器对象 imp_mean 对 X 进行填补操作，返回填补后的数组
    imp_mean.transform(X)

    # 更详细的示例请参见相关链接
    # :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py` 或
    # :ref:`sphx_glr_auto_examples_impute_plot_iterative_imputer_variants_comparison.py`.
    """

    # 定义参数约束字典，继承了 _BaseImputer 的参数约束
    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        # 估计器（estimator）参数可以为 None 或具有 fit 和 predict 方法的对象
        "estimator": [None, HasMethods(["fit", "predict"])],
        # 是否对后验样本进行采样的标志，应为布尔值
        "sample_posterior": ["boolean"],
        # 最大迭代次数，应为大于等于 0 的整数
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        # 公差，应为大于 0 的实数
        "tol": [Interval(Real, 0, None, closed="left")],
        # 最近特征的数量，可以为 None 或大于等于 1 的整数
        "n_nearest_features": [None, Interval(Integral, 1, None, closed="left")],
        # 初始策略，应为 {"mean", "median", "most_frequent", "constant"} 中的一种
        "initial_strategy": [
            StrOptions({"mean", "median", "most_frequent", "constant"})
        ],
        # 填补值，任何对象都是有效的
        "fill_value": "no_validation",
        # 填补顺序，应为 {"ascending", "descending", "roman", "arabic", "random"} 中的一种
        "imputation_order": [
            StrOptions({"ascending", "descending", "roman", "arabic", "random"})
        ],
        # 是否跳过已完整的数据项，应为布尔值
        "skip_complete": ["boolean"],
        # 最小值，可以为 None、实数区间或数组形式的约束
        "min_value": [None, Interval(Real, None, None, closed="both"), "array-like"],
        # 最大值，可以为 None、实数区间或数组形式的约束
        "max_value": [None, Interval(Real, None, None, closed="both"), "array-like"],
        # 是否输出详细信息的标志，应为 "verbose"
        "verbose": ["verbose"],
        # 随机种子，应为 "random_state"
        "random_state": ["random_state"],
    }

    # 初始化函数，设定迭代式填补器的各项参数
    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        fill_value=None,
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        keep_empty_features=False,
    ):
        # 调用父类的初始化方法，设定缺失值和输出指示器参数
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )

        # 设置各项参数值
        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
        params=None,
    ):
        """
        Impute missing values for a single feature.

        Parameters
        ----------
        X_filled : ndarray, shape (n_samples, n_features)
            Input data matrix with missing values filled.

        mask_missing_values : ndarray, shape (n_samples, n_features)
            Boolean mask array indicating missing values.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : array-like
            List of indices of other features used to impute `feat_idx`.

        estimator : object or None, default=None
            Estimator object used for imputation. If None, a default estimator will be used.

        fit_mode : bool, default=True
            Whether to fit the estimator if it's not already fitted.

        params : dict or None, default=None
            Additional parameters passed to the estimator.

        Returns
        -------
        X_filled : ndarray, shape (n_samples, n_features)
            Data matrix with imputed values for the specified feature.
        """
        if estimator is None:
            estimator = self.estimator

        # If fit_mode is True, fit the estimator if it's not already fitted
        if fit_mode and not estimator.is_fitted:
            estimator.fit(X_filled[:, neighbor_feat_idx], X_filled[:, feat_idx], **params)

        # Impute missing values for the feature using the fitted estimator
        X_filled[mask_missing_values[:, feat_idx], feat_idx] = estimator.predict(
            X_filled[mask_missing_values[:, feat_idx], neighbor_feat_idx], **params
        )

        return X_filled


    def _get_neighbor_feat_idx(self, n_features, feat_idx, abs_corr_mat):
        """
        Get a list of other features to predict `feat_idx`.

        If `self.n_nearest_features` is less than or equal to the total
        number of features, then use a probability proportional to the absolute
        correlation between `feat_idx` and each other feature to randomly
        choose a subsample of the other features (without replacement).

        Parameters
        ----------
        n_features : int
            Number of features in `X`.

        feat_idx : int
            Index of the feature currently being imputed.

        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X`. The diagonal has been zeroed
            out and each feature has been normalized to sum to 1. Can be None.

        Returns
        -------
        neighbor_feat_idx : array-like
            The features to use to impute `feat_idx`.
        """
        if self.n_nearest_features is not None and self.n_nearest_features < n_features:
            # Probability vector based on absolute correlation with feat_idx
            p = abs_corr_mat[:, feat_idx]

            # Randomly choose self.n_nearest_features features based on p
            neighbor_feat_idx = self.random_state_.choice(
                np.arange(n_features), self.n_nearest_features, replace=False, p=p
            )
        else:
            # Include all features except feat_idx
            inds_left = np.arange(feat_idx)
            inds_right = np.arange(feat_idx + 1, n_features)
            neighbor_feat_idx = np.concatenate((inds_left, inds_right))

        return neighbor_feat_idx
    def _get_ordered_idx(self, mask_missing_values):
        """
        Decide in what order we will update the features.

        As a homage to the MICE R package, we will have 4 main options of
        how to order the updates, and use a random order if anything else
        is specified.

        Also, this function skips features which have no missing values.

        Parameters
        ----------
        mask_missing_values : array-like, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        Returns
        -------
        ordered_idx : ndarray, shape (n_features,)
            The order in which to impute the features.
        """
        # 计算每个特征缺失值的比例
        frac_of_missing_values = mask_missing_values.mean(axis=0)
        
        # 如果设定跳过完整的特征（没有缺失值的特征），则找到具有缺失值的特征的索引
        if self.skip_complete:
            missing_values_idx = np.flatnonzero(frac_of_missing_values)
        else:
            missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
        
        # 根据不同的 imputation_order 参数确定特征更新的顺序
        if self.imputation_order == "roman":
            ordered_idx = missing_values_idx
        elif self.imputation_order == "arabic":
            ordered_idx = missing_values_idx[::-1]
        elif self.imputation_order == "ascending":
            # 按照缺失值比例升序排序特征索引
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
        elif self.imputation_order == "descending":
            # 按照缺失值比例降序排序特征索引
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
        elif self.imputation_order == "random":
            # 随机打乱特征索引顺序
            ordered_idx = missing_values_idx
            self.random_state_.shuffle(ordered_idx)
        
        # 返回确定的特征更新顺序
        return ordered_idx
    # 获取特征之间的绝对相关性矩阵

    Parameters
    ----------
    X_filled : ndarray, shape (n_samples, n_features)
        包含最新填充值的输入数据。

    tolerance : float, default=1e-6
        `abs_corr_mat` 可能包含 NaN 值，这些值将被替换为 `tolerance`。

    Returns
    -------
    abs_corr_mat : ndarray, shape (n_features, n_features)
        当前轮次开始时 `X` 的绝对相关性矩阵。对角线已经清零，每个特征与所有其他特征的绝对相关性已经归一化为和为1。
    """
    n_features = X_filled.shape[1]
    if self.n_nearest_features is None or self.n_nearest_features >= n_features:
        return None
    with np.errstate(invalid="ignore"):
        # 如果邻域中的特征仅有一个值（例如分类特征），标准差将为 null，
        # np.corrcoef 会由于除以零而引发警告
        abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
    # np.corrcoef 对于标准差为零的特征未定义
    abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
    # 确保探索性，即至少有一定的概率可以进行采样
    np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
    # 特征与自身不相关
    np.fill_diagonal(abs_corr_mat, 0)
    # 为了使用 np.random.choice 进行采样，需要归一化为和为1
    abs_corr_mat = normalize(abs_corr_mat, norm="l1", axis=0, copy=False)
    return abs_corr_mat
    def _initial_imputation(self, X, in_fit=False):
        """Perform initial imputation for input `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        in_fit : bool, default=False
            Whether function is called in :meth:`fit`.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Input data after initial imputation.

        X_filled : ndarray of shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray of shape (n_samples, n_features)
            Missing indicator matrix, indicating where values were missing
            in the input data.

        X_missing_mask : ndarray, shape (n_samples, n_features)
            Mask matrix indicating missing datapoints in the input data.
        """
        # Determine how to handle nan values based on the `missing_values` parameter
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True
        
        # Validate the input data `X` and handle its characteristics
        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )
        # Check and enforce consistency of input data types
        _check_inputs_dtype(X, self.missing_values)

        # Create a mask for missing values in the input data `X`
        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()

        # Perform initial imputation if no imputer has been initialized
        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                missing_values=self.missing_values,
                strategy=self.initial_strategy,
                fill_value=self.fill_value,
                keep_empty_features=self.keep_empty_features,
            ).set_output(transform="default")
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            # Use existing imputer to transform `X`
            X_filled = self.initial_imputer_.transform(X)

        # Identify valid columns that are not completely empty after imputation
        valid_mask = np.flatnonzero(
            np.logical_not(np.isnan(self.initial_imputer_.statistics_))
        )

        # Handle whether to keep or drop empty features
        if not self.keep_empty_features:
            # Drop columns with empty features from `X`
            Xt = X[:, valid_mask]
            mask_missing_values = mask_missing_values[:, valid_mask]
        else:
            # Mark empty columns as not missing and keep the original imputation
            mask_missing_values[:, valid_mask] = True
            Xt = X

        return Xt, X_filled, mask_missing_values, X_missing_mask
    # 验证特征值的上下限（最小值或最大值）的有效性
    def _validate_limit(limit, limit_type, n_features):
        """Validate the limits (min/max) of the feature values.

        Converts scalar min/max limits to vectors of shape `(n_features,)`.

        Parameters
        ----------
        limit: scalar or array-like
            The user-specified limit (i.e, min_value or max_value).
        limit_type: {'max', 'min'}
            Type of limit to validate.
        n_features: int
            Number of features in the dataset.

        Returns
        -------
        limit: ndarray, shape(n_features,)
            Array of limits, one for each feature.
        """
        # 根据限制类型设置限制边界，如果是最大值类型，使用正无穷，否则使用负无穷
        limit_bound = np.inf if limit_type == "max" else -np.inf
        # 如果限制为None，则使用限制边界
        limit = limit_bound if limit is None else limit
        # 如果限制是标量，则将其转换为形状为(n_features,)的向量
        if np.isscalar(limit):
            limit = np.full(n_features, limit)
        # 检查并确保限制数组中的所有元素都是有限的
        limit = check_array(limit, force_all_finite=False, copy=False, ensure_2d=False)
        # 如果限制数组的长度与特征数不匹配，则引发 ValueError 异常
        if not limit.shape[0] == n_features:
            raise ValueError(
                f"'{limit_type}_value' should be of "
                f"shape ({n_features},) when an array-like "
                f"is provided. Got {limit.shape}, instead."
            )
        # 返回有效的限制数组
        return limit

    @_fit_context(
        # IterativeImputer.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def transform(self, X):
        """Impute all missing values in `X`.

        Note that this is stochastic, and that if `random_state` is not fixed,
        repeated calls, or permuted input, results will differ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        # 确保模型已经拟合，即检查是否已经调用了 fit 方法
        check_is_fitted(self)

        # 初始化缺失值填充前的数据和掩码
        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, in_fit=False
        )

        # 生成完整性指示器矩阵
        X_indicator = super()._transform_indicator(complete_mask)

        # 如果是第一轮迭代或所有值都是缺失值，则直接返回填充后的数据和指示器
        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return super()._concatenate_indicator(Xt, X_indicator)

        # 计算每轮迭代需要进行的填充次数
        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        # 如果设置了输出详细信息，打印正在完成矩阵的信息
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        # 对每个估算器三元组执行多轮迭代填充
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False,
            )
            # 如果达到了一轮迭代的结束点，根据输出级别打印信息
            if not (it + 1) % imputations_per_round:
                if self.verbose > 1:
                    print(
                        "[IterativeImputer] Ending imputation round "
                        "%d/%d, elapsed time %0.2f"
                        % (i_rnd + 1, self.n_iter_, time() - start_t)
                    )
                i_rnd += 1

        # 将填充后的数据赋值回原始数据中未缺失的位置
        _assign_where(Xt, X, cond=~mask_missing_values)

        # 返回填充后的数据和完整性指示器的拼接结果
        return super()._concatenate_indicator(Xt, X_indicator)

    def fit(self, X, y=None, **fit_params):
        """Fit the imputer on `X` and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **fit_params : dict
            Parameters routed to the `fit` method of the sub-estimator via the
            metadata routing API.

            .. versionadded:: 1.5
              Only available if
              `sklearn.set_config(enable_metadata_routing=True)` is set. See
              :ref:`Metadata Routing User Guide <metadata_routing>` for more
              details.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 调用 fit_transform 方法进行拟合
        self.fit_transform(X, **fit_params)
        # 返回已拟合的估算器对象
        return self
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # 确保模型已经拟合，获取输入特征的名称
        check_is_fitted(self, "n_features_in_")
        # 根据输入特征名称进行验证和处理
        input_features = _check_feature_names_in(self, input_features)
        # 获取初始填充器对象的输出特征名称
        names = self.initial_imputer_.get_feature_names_out(input_features)
        # 返回连接了指示器特征名称后的最终输出特征名称
        return self._concatenate_indicator_feature_names_out(names, input_features)

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.5

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        # 创建一个 MetadataRouter 对象，设置拥有者和估计器以及方法映射
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(callee="fit", caller="fit"),
        )
        # 返回包含路由信息的 MetadataRouter 对象
        return router
```