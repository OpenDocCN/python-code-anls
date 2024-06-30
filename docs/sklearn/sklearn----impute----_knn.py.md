# `D:\src\scipysrc\scikit-learn\sklearn\impute\_knn.py`

```
# 从 numbers 模块导入 Integral 类型，用于后续参数类型验证
from numbers import Integral

# 导入 numpy 库并使用 np 别名
import numpy as np

# 从 base 模块导入 _fit_context 函数
from ..base import _fit_context

# 从 metrics 模块导入 pairwise_distances_chunked 函数
from ..metrics import pairwise_distances_chunked

# 从 metrics.pairwise 模块导入 _NAN_METRICS 常量
from ..metrics.pairwise import _NAN_METRICS

# 从 neighbors._base 模块导入 _get_weights 函数
from ..neighbors._base import _get_weights

# 从 utils._mask 模块导入 _get_mask 函数
from ..utils._mask import _get_mask

# 从 utils._missing 模块导入 is_scalar_nan 函数
from ..utils._missing import is_scalar_nan

# 从 utils._param_validation 模块导入 Hidden、Interval、StrOptions 类
from ..utils._param_validation import Hidden, Interval, StrOptions

# 从 utils.validation 模块导入 FLOAT_DTYPES、_check_feature_names_in、check_is_fitted 函数
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted

# 从 _base 模块导入 _BaseImputer 类
from ._base import _BaseImputer


class KNNImputer(_BaseImputer):
    """Imputation for completing missing values using k-Nearest Neighbors.

    Each sample's missing values are imputed using the mean value from
    `n_neighbors` nearest neighbors found in the training set. Two samples are
    close if the features that neither is missing are close.

    Read more in the :ref:`User Guide <knnimpute>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - callable : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : {'nan_euclidean'} or callable, default='nan_euclidean'
        Distance metric for searching neighbors. Possible values:

        - 'nan_euclidean'
        - callable : a user-defined function which conforms to the definition
          of ``_pairwise_callable(X, Y, metric, **kwds)``. The function
          accepts two arrays, X and Y, and a `missing_values` keyword in
          `kwds` and returns a scalar distance value.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
"""
    # 控制是否添加缺失指示器的布尔标志，默认为 False
    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto the
        output of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on the
        missing indicator even if there are missing values at transform/test
        time.

    # 控制是否保留空特征的布尔标志，默认为 False
    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0`.

        .. versionadded:: 1.2

    # 类属性：用于添加缺失值二进制指示器的指示器对象，如果 add_indicator 为 False，则为 None
    Attributes
    ----------
    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    # 在 fit 过程中看到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 在 fit 过程中看到的特征名称的数组，仅当 `X` 的特征名称都是字符串时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 参见
    # --------
    # SimpleImputer : 使用简单策略完成缺失值的单变量填充
    # IterativeImputer : 使用其他特征的值估计每个特征的缺失值的多变量填充
    See Also
    --------
    SimpleImputer : Univariate imputer for completing missing values
        with simple strategies.
    IterativeImputer : Multivariate imputer that estimates values to impute for
        each feature with missing values from all the others.

    # 参考文献
    # ----------
    # 缺失值估计方法在 DNA 微阵列中的应用，提供了详细的说明
    References
    ----------
    * `Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
      Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
      value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
      no. 6, 2001 Pages 520-525.
      <https://academic.oup.com/bioinformatics/article/17/6/520/272365>`_

    # 示例
    # --------
    # 使用 KNNImputer 对象对输入的 X 进行填充处理
    # >>> import numpy as np
    # >>> from sklearn.impute import KNNImputer
    # >>> X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
    # >>> imputer = KNNImputer(n_neighbors=2)
    # >>> imputer.fit_transform(X)
    # array([[1. , 2. , 4. ],
    #        [3. , 4. , 3. ],
    #        [5.5, 6. , 5. ],
    #        [8. , 8. , 7. ]])
    #
    # 更详细的示例请参见 :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

    # 私有成员变量：用于约束参数的字典，继承自 _BaseImputer 的约束
    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "weights": [StrOptions({"uniform", "distance"}), callable, Hidden(None)],
        "metric": [StrOptions(set(_NAN_METRICS)), callable],
        "copy": ["boolean"],
    }

    # 初始化方法，设置缺失值、邻居数、权重、距离度量、拷贝标志、缺失指示器和保留空特征标志等参数
    def __init__(
        self,
        *,
        missing_values=np.nan,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
    ):
        # 调用父类的初始化方法，设置缺失值、指示器添加、保留空特征等参数
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        # 设置近邻数目、权重方式和距离度量方法
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """Helper function to impute a single column.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # 获取最近的邻居索引
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # 从距离矩阵中获取权重矩阵
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # 将 NaN 值填充为零
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0
        else:
            weight_matrix = np.ones_like(donors_dist)
            weight_matrix[np.isnan(donors_dist)] = 0.0

        # 获取捐赠者的值并计算 kNN 平均值
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        return np.ma.average(donors, axis=1, weights=weight_matrix).data

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : array-like shape of (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            The fitted `KNNImputer` class instance.
        """
        # 检查数据的完整性和调用参数
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        # 对输入数据进行验证和转换
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        # 将数据保存在实例中作为拟合数据
        self._fit_X = X
        # 获取数据中缺失值的掩码
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)
        # 计算有效特征的掩码
        self._valid_mask = ~np.all(self._mask_fit_X, axis=0)

        # 调用父类的方法来处理拟合指示器
        super()._fit_indicator(self._mask_fit_X)

        # 返回拟合后的实例自身
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

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
        # 检查模型是否已经拟合
        check_is_fitted(self, "n_features_in_")
        # 检查输入特征的有效性和一致性
        input_features = _check_feature_names_in(self, input_features)
        # 根据有效掩码获取有效特征的名称
        names = input_features[self._valid_mask]
        # 返回拼接后的输出特征名称
        return self._concatenate_indicator_feature_names_out(names, input_features)
```