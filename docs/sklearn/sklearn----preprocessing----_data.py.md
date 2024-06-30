# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_data.py`

```
# 导入警告模块，用于管理警告信息
import warnings
# 导入用于数值类型检查的模块
from numbers import Integral, Real

# 导入常用的科学计算库和工具函数
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox

# 导入基础模块和工具函数
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
# 导入数组处理和验证函数
from ..utils import _array_api, check_array, resample
# 导入数组API相关的命名空间函数
from ..utils._array_api import get_namespace
# 导入参数验证相关的类和函数
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
# 导入数学函数和算法工具
from ..utils.extmath import _incremental_mean_and_var, row_norms
# 导入稀疏矩阵处理函数
from ..utils.sparsefuncs import (
    incr_mean_variance_axis,
    inplace_column_scale,
    mean_variance_axis,
    min_max_axis,
)
# 导入快速稀疏矩阵处理函数
from ..utils.sparsefuncs_fast import (
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)
# 导入验证相关的函数和数据类型
from ..utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
)
# 导入编码器相关的类
from ._encoders import OneHotEncoder

# 设定用于判断常数特征的阈值
BOUNDS_THRESHOLD = 1e-7

# 定义公开的类和函数列表，这些是在模块外可见的
__all__ = [
    "Binarizer",
    "KernelCenterer",
    "MinMaxScaler",
    "MaxAbsScaler",
    "Normalizer",
    "OneHotEncoder",
    "RobustScaler",
    "StandardScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "add_dummy_feature",
    "binarize",
    "normalize",
    "scale",
    "robust_scale",
    "maxabs_scale",
    "minmax_scale",
    "quantile_transform",
    "power_transform",
]

# 检测一个特征是否与常数特征难以区分
def _is_constant_feature(var, mean, n_samples):
    """Detect if a feature is indistinguishable from a constant feature.

    The detection is based on its computed variance and on the theoretical
    error bounds of the '2 pass algorithm' for variance computation.

    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    """
    # 在scikit-learn中，方差总是使用float64类型的累加器计算
    eps = np.finfo(np.float64).eps

    # 计算上界，用于判断是否是常数特征
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound

# 处理标度中接近常数特征的零值
def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # 如果标度是一维数组，可能scale是标量
    if np.isscalar(scale):
        # 如果scale为0，将其设为1
        if scale == 0.0:
            scale = 1.0
        return scale
    # 如果scale是数组，处理每个元素
    else:
        # 获取与 `scale` 数组相同命名空间的 `xp` 和 `_`
        xp, _ = get_namespace(scale)
        # 如果 `constant_mask` 为 None，则检测接近常数值的元素
        # 避免除以一个非常小的值，可能导致意外结果和数值稳定性问题
        if constant_mask is None:
            constant_mask = scale < 10 * xp.finfo(scale.dtype).eps

        # 如果需要复制数组以避免副作用，则创建一个新的数组
        if copy:
            scale = xp.asarray(scale, copy=True)
        # 将 `scale` 中被标记为常数的元素设置为 1.0
        scale[constant_mask] = 1.0
        # 返回修改后的 `scale` 数组
        return scale
# 使用 @validate_params 装饰器对 scale 函数进行参数验证，确保参数符合指定的类型和取值范围
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数可以是 array-like 或 sparse matrix 类型
        "axis": [Options(Integral, {0, 1})],  # axis 参数必须是整数类型，且取值为 {0, 1}
        "with_mean": ["boolean"],  # with_mean 参数必须是布尔类型
        "with_std": ["boolean"],  # with_std 参数必须是布尔类型
        "copy": ["boolean"],  # copy 参数必须是布尔类型
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
# scale 函数用于标准化数据集沿任意轴
def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    StandardScaler : Performs scaling to unit variance using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.
    """
    """
    X = check_array(
        X,  # 输入数据，将进行检查和转换
        accept_sparse="csc",  # 允许稀疏矩阵格式为压缩列格式
        copy=copy,  # 指定是否复制输入数据，默认为True
        ensure_2d=False,  # 确保输入至少为二维数组，如果为False，则不强制要求
        estimator="the scale function",  # 估计器名称，用于错误消息
        dtype=FLOAT_DTYPES,  # 指定期望的数据类型
        force_all_finite="allow-nan",  # 指定如何处理非有限的数据点，允许NaN值
    )
    if sparse.issparse(X):  # 如果输入数据是稀疏矩阵
        if with_mean:  # 如果要进行均值中心化
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives."
            )
        if axis != 0:  # 如果指定的轴不是0
            raise ValueError(
                "Can only scale sparse matrix on axis=0,  got axis=%d" % axis
            )
        if with_std:  # 如果要进行标准化
            _, var = mean_variance_axis(X, axis=0)  # 计算方差
            var = _handle_zeros_in_scale(var, copy=False)  # 处理方差中的零值
            inplace_column_scale(X, 1 / np.sqrt(var))  # 对每列进行标准化处理
    """
    else:
        # 如果输入 X 不是 NaN 数组，则将其转换为 NumPy 数组
        X = np.asarray(X)
        
        # 如果指定需要进行均值中心化操作
        if with_mean:
            # 计算 X 在指定轴上的均值，忽略 NaN 值
            mean_ = np.nanmean(X, axis)
        
        # 如果指定需要进行标准化操作
        if with_std:
            # 计算 X 在指定轴上的标准差，忽略 NaN 值
            scale_ = np.nanstd(X, axis)
        
        # Xr 是对原始数组的视图，便于在指定轴上进行广播操作
        Xr = np.rollaxis(X, axis)
        
        # 如果指定需要进行均值中心化操作
        if with_mean:
            # 将 Xr 中的数据减去均值 mean_
            Xr -= mean_
            # 计算经过均值中心化后的 Xr 在每列上的均值
            mean_1 = np.nanmean(Xr, axis=0)
            
            # 如果均值 mean_1 不接近零，可能是由于均值 mean_ 的精度不足导致
            if not np.allclose(mean_1, 0):
                warnings.warn(
                    "Numerical issues were encountered "
                    "when centering the data "
                    "and might not be solved. Dataset may "
                    "contain too large values. You may need "
                    "to prescale your features."
                )
                # 再次将 Xr 中的数据减去均值 mean_1
                Xr -= mean_1
        
        # 如果指定需要进行标准化操作
        if with_std:
            # 处理标准差 scale_ 中的零值
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            # 将 Xr 中的数据除以标准差 scale_
            Xr /= scale_
            
            # 如果同时指定需要进行均值中心化操作
            if with_mean:
                # 计算经过标准化后的 Xr 在每列上的均值
                mean_2 = np.nanmean(Xr, axis=0)
                
                # 如果均值 mean_2 不接近零，可能是由于标准差 scale_ 很小导致
                if not np.allclose(mean_2, 0):
                    warnings.warn(
                        "Numerical issues were encountered "
                        "when scaling the data "
                        "and might not be solved. The standard "
                        "deviation of the data is probably "
                        "very close to 0. "
                    )
                    # 再次将 Xr 中的数据减去均值 mean_2
                    Xr -= mean_2
    
    # 返回处理后的数组 X
    return X
class MinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    `MinMaxScaler` doesn't reduce the effect of outliers, but it linearly
    scales them down into a fixed range, where the largest occurring data point
    corresponds to the maximum value and the smallest one corresponds to the
    minimum value. For an example visualization, refer to :ref:`Compare
    MinMaxScaler with other scalers <plot_all_scaling_minmax_scaler_section>`.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.

        .. versionadded:: 0.24

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

        .. versionadded:: 0.17
           *scale_* attribute.

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data

        .. versionadded:: 0.17
           *data_min_*

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data

        .. versionadded:: 0.17
           *data_max_*

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

        .. versionadded:: 0.17
           *data_range_*

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    minmax_scale : Equivalent function without the estimator API.

    Notes
    -----
    """
    # 定义MinMaxScaler类，继承自OneToOneFeatureMixin, TransformerMixin, BaseEstimator
    # 实现了特征缩放，将每个特征缩放到指定的范围内
    # 用于将数据特征缩放到训练集上的给定范围，例如在0到1之间
    # 其转换方式为使用(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))进行标准化，再乘以(max - min)并加上min
    # 其中min, max由feature_range指定
    # MinMaxScaler不减少异常值的影响，但会线性地缩小它们到一个固定的范围，其中最大出现的数据点对应最大值，最小点对应最小值
    # 参考 :ref:`Compare MinMaxScaler with other scalers <plot_all_scaling_minmax_scaler_section>` 进行示例可视化
    # 更多信息请参考 :ref:`User Guide <preprocessing_scaler>`

    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        # 初始化时设置feature_range, copy, clip属性

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # 计算最小值和最大值用于后续的特征缩放
        # X为数据，形状为(n_samples, n_features)，计算每个特征沿特征轴的最小值和最大值
        # 返回已拟合的缩放器对象
        self.n_samples_seen_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        # 设置n_samples_seen_为样本数量，n_features_in_为特征数量
        if sparse.issparse(X):
            data_min = []
            data_max = []
            data_range = []
            for i in range(X.shape[1]):
                # 循环计算稀疏矩阵的每一列的最小值和最大值
                column_min = X.data[X.indptr[i]:X.indptr[i + 1]].min()
                column_max = X.data[X.indptr[i]:X.indptr[i + 1]].max()
                data_min.append(column_min)
                data_max.append(column_max)
                data_range.append(column_max - column_min)
            # 设置data_min_, data_max_, data_range_为每列的最小值、最大值、范围
            self.data_min_ = np.array(data_min)
            self.data_max_ = np.array(data_max)
            self.data_range_ = np.array(data_range)
        else:
            # 计算非稀疏矩阵的每列的最小值、最大值、范围
            self.data_min_ = np.min(X, axis=0)
            self.data_max_ = np.max(X, axis=0)
            self.data_range_ = self.data_max_ - self.data_min_
        # 计算scale_，每个特征的相对缩放数据
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        # 返回已拟合的缩放器对象
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        X_transformed : ndarray array-like or sparse matrix, shape (n_samples, n_features)
            Transformed array.
        """
        # 根据feature_range缩放X的特征
        # 返回经过转换后的数组
        X = check_array(X, accept_sparse='csr', copy=self.copy, estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')
        # 检查并转换X为数组形式，接受稀疏矩阵
        if sparse.issparse(X):
            if self.clip:
                warn("Compressed sparse input with clip=True is not supported. "
                     "Proceeding without clipping.")
            data_min = self.data_min_.reshape((1, -1))
            scale = self.scale_.reshape((1, -1))
            # 对稀疏矩阵的每个特征进行缩放处理
            X.data = X.data * scale + self.min_
            if self.clip:
                X.data = np.clip(X.data, self.feature_range[0], self.feature_range[1])
        else:
            # 对非稀疏矩阵的每个特征进行缩放处理
            X = X * self.scale_ + self.min_
            if self.clip:
                np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        # 返回缩放后的数组
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data that will
    """
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]
    """

    # 定义参数的约束字典，指定了各参数的类型
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        # 初始化MinMaxScaler对象的参数
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # 重置Scaler的内部状态，删除与数据相关的属性
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # 在拟合之前重置内部状态
        self._reset()
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # 获取特征范围
        feature_range = self.feature_range
        # 检查特征范围是否合法
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        # 检查输入是否为稀疏矩阵，MinMaxScaler 不支持稀疏输入
        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        # 获取数据的命名空间并进行初始验证
        xp, _ = get_namespace(X)

        # 第一次通过检查
        first_pass = not hasattr(self, "n_samples_seen_")
        # 验证数据，并根据需要重置数据
        X = self._validate_data(
            X,
            reset=first_pass,
            dtype=_array_api.supported_float_dtypes(xp),
            force_all_finite="allow-nan",
        )

        # 计算数据的最小值和最大值
        data_min = _array_api._nanmin(X, axis=0, xp=xp)
        data_max = _array_api._nanmax(X, axis=0, xp=xp)

        # 如果是第一次通过，则初始化 n_samples_seen_
        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            # 否则，更新最小值和最大值
            data_min = xp.minimum(self.data_min_, data_min)
            data_max = xp.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        # 计算数据的范围
        data_range = data_max - data_min
        # 计算缩放比例
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        # 计算最小值
        self.min_ = feature_range[0] - data_min * self.scale_
        # 存储数据的最小值、最大值、数据范围
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        # 确保模型已经拟合
        check_is_fitted(self)

        # 获取数据的命名空间并验证输入数据
        xp, _ = get_namespace(X)

        # 验证输入数据，并根据需要复制数据
        X = self._validate_data(
            X,
            copy=self.copy,
            dtype=_array_api.supported_float_dtypes(xp),
            force_writeable=True,
            force_all_finite="allow-nan",
            reset=False,
        )

        # 对数据进行缩放操作
        X *= self.scale_
        X += self.min_

        # 如果设置了裁剪选项，则对数据进行裁剪
        if self.clip:
            xp.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        
        return X
    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)

        # 获取输入数据的命名空间
        xp, _ = get_namespace(X)

        # 检查并转换输入数据 X
        X = check_array(
            X,
            copy=self.copy,
            dtype=_array_api.supported_float_dtypes(xp),
            force_writeable=True,
            force_all_finite="allow-nan",
        )

        # 根据最小值和缩放比例进行数据逆转换（反标准化）
        X -= self.min_
        X /= self.scale_

        # 返回转换后的数据
        return X

    # 返回附加标签信息，允许 NaN 值
    def _more_tags(self):
        return {"allow_nan": True}
# 使用装饰器 validate_params 进行参数验证，确保函数参数满足指定类型和选项
@validate_params(
    {
        "X": ["array-like"],    # X 参数类型为 array-like
        "axis": [Options(Integral, {0, 1})],   # axis 参数类型为 Integral，取值为 0 或 1
    },
    prefer_skip_nested_validation=False,
)
def minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`~sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : {0, 1}, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : ndarray of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.minmax_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MinMaxScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MinMaxScaler(), LogisticRegression())`.

    See Also
    --------
    MinMaxScaler : Performs scaling to a given range using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    Examples
    --------
    >>> from sklearn.preprocessing import minmax_scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    """
    # 对输入的数据进行最小-最大缩放，使得数据在指定的特征范围内
    def minmax_scale(X, axis=0):
        # 检查输入数据 X，确保其为数组形式，不复制原数据，不强制要求二维，数据类型为浮点型，允许存在 NaN 值
        X = check_array(
            X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )
        # 记录原始数据的维度
        original_ndim = X.ndim
    
        # 如果原始数据是一维的，将其转换为二维的形式，列数为1
        if original_ndim == 1:
            X = X.reshape(X.shape[0], 1)
    
        # 创建最小-最大缩放器对象，指定特征范围和是否复制数据
        s = MinMaxScaler(feature_range=feature_range, copy=copy)
    
        # 根据指定的轴进行数据缩放
        if axis == 0:
            X = s.fit_transform(X)  # 按列进行缩放
        else:
            X = s.fit_transform(X.T).T  # 按行进行缩放，先转置后再转置回来
    
        # 如果原始数据是一维的，将结果展平为一维数组
        if original_ndim == 1:
            X = X.ravel()
    
        # 返回缩放后的数据
        return X
# 定义一个名为 StandardScaler 的类，它继承自 OneToOneFeatureMixin、TransformerMixin 和 BaseEstimator。
# StandardScaler 类用于标准化特征，即去除均值并缩放到单位方差。

class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    `StandardScaler` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers. For an example
    visualization, refer to :ref:`Compare StandardScaler with other scalers
    <plot_all_scaling_standard_scaler_section>`.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    # scale_ : ndarray of shape (n_features,) or None
    #     每个特征的相对缩放比例，用于实现零均值和单位方差。通常使用 `np.sqrt(var_)` 计算。
    #     如果方差为零，则无法实现单位方差，数据保持不变，缩放因子为 1。
    #     当 `with_std=False` 时，`scale_` 等于 `None`。
    #
    # mean_ : ndarray of shape (n_features,) or None
    #     训练集中每个特征的均值。当 `with_mean=False` 和 `with_std=False` 时等于 ``None``。
    #
    # var_ : ndarray of shape (n_features,) or None
    #     训练集中每个特征的方差。用于计算 `scale_`。当 `with_mean=False` 和 `with_std=False` 时等于 ``None``。
    #
    # n_features_in_ : int
    #     在 `fit` 过程中看到的特征数量。
    #
    # feature_names_in_ : ndarray of shape (`n_features_in_`,)
    #     在 `fit` 过程中看到的特征名称。仅当 `X` 的特征名都是字符串时定义。
    #
    # n_samples_seen_ : int or ndarray of shape (n_features,)
    #     估计器处理每个特征的样本数量。如果没有丢失的样本，则 `n_samples_seen_` 将是整数，
    #     否则将是一个 `dtype` 为 `int` 的数组。如果使用 `sample_weights`，则它将是一个浮点数
    #     （如果没有丢失的数据）或一个 `dtype` 为 `float` 的数组，其中包含到目前为止看到的权重总和。
    #     将在新的 `fit` 调用中重置，但在 `partial_fit` 调用之间递增。
    #
    # See Also
    # --------
    # scale : 没有估计器 API 的等效函数。
    #
    # :class:`~sklearn.decomposition.PCA` : 使用 'whiten=True' 进一步消除特征之间的线性相关性。
    #
    # Notes
    # -----
    # NaN 被视为缺失值：在 `fit` 中被忽略，在 `transform` 中保留。
    #
    # 我们使用偏差估计器来计算标准差，等效于 `numpy.std(x, ddof=0)`。注意 `ddof` 的选择不太可能影响模型性能。
    #
    # Examples
    # --------
    # >>> from sklearn.preprocessing import StandardScaler
    # >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    # >>> scaler = StandardScaler()
    # >>> print(scaler.fit(data))
    # StandardScaler()
    # >>> print(scaler.mean_)
    # [0.5 0.5]
    # >>> print(scaler.transform(data))
    # [[-1. -1.]
    #  [-1. -1.]
    #  [ 1.  1.]
    #  [ 1.  1.]]
    # >>> print(scaler.transform([[2, 2]]))
    # [[3. 3.]]
    #
    # _parameter_constraints: dict
    #     用于存储参数约束的字典，指定每个参数的类型。
    #
    # __init__(self, *, copy=True, with_mean=True, with_std=True)
    #     初始化函数，设置 `with_mean`、`with_std` 和 `copy`。
    #
    #     Parameters:
    #     -----------
    #     copy : bool, optional, default=True
    #         是否复制输入数据，默认为 `True`。
    #     with_mean : bool, optional, default=True
    #         是否在缩放前将数据居中，默认为 `True`。
    #     with_std : bool, optional, default=True
    #         是否在缩放前将数据标准化到单位方差，默认为 `True`。
    def _reset(self):
        """重置标量器的内部数据相关状态，如果需要的话。

        不会影响 __init__ 方法中的参数设定。
        """
        # 检查一个属性就足够了，因为它们都在 partial_fit 中一起设置
        # 检查是否存在 scale_ 属性，如果存在则删除与标量器状态相关的所有属性
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """计算用于后续标准化的均值和标准差。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            用于计算后续沿特征轴标准化的数据。

        y : None
            忽略。

        sample_weight : array-like of shape (n_samples,), default=None
            每个样本的个体权重。

            .. versionadded:: 0.24
               支持 StandardScaler 的 *sample_weight* 参数。

        Returns
        -------
        self : object
            拟合后的标量器。
        """
        # 在拟合之前重置内部状态
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    @_fit_context(prefer_skip_nested_validation=True)
    def transform(self, X, copy=None):
        """通过中心化和缩放执行标准化。

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)}
            用于沿特征轴进行缩放的数据。
        copy : bool, default=None
            是否复制输入 X。

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            转换后的数组。
        """
        # 检查是否已拟合
        check_is_fitted(self)

        # 确定是否复制输入 X
        copy = copy if copy is not None else self.copy

        # 验证输入数据 X
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csr",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            force_all_finite="allow-nan",
        )

        # 处理稀疏矩阵的情况
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            # 处理非稀疏矩阵的情况
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X
    # 使用逆变换将数据重新缩放为原始表示
    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 确保模型已拟合
        check_is_fitted(self)

        # 确定是否复制输入的 X
        copy = copy if copy is not None else self.copy
        # 检查并转换输入的 X，确保其接受稀疏矩阵格式，且类型为 FLOAT_DTYPES 中的一种
        X = check_array(
            X,
            accept_sparse="csr",  # 仅接受稀疏矩阵的 CSR 格式
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_writeable=True,  # 确保 X 可写
            force_all_finite="allow-nan",  # 允许包含 NaN 值
        )

        # 如果 X 是稀疏矩阵
        if sparse.issparse(X):
            # 如果需要均值调整，则抛出错误，因为无法对稀疏矩阵进行均值调整
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives."
                )
            # 如果有缩放因子，对 X 进行原地列缩放
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            # 如果需要标准化，则对 X 应用缩放因子
            if self.with_std:
                X *= self.scale_
            # 如果需要均值调整，则对 X 添加均值
            if self.with_mean:
                X += self.mean_
        # 返回转换后的 X
        return X

    # 返回关于估算器的额外标签信息，表示允许包含 NaN 值，且保持数据类型为 np.float64 或 np.float32
    def _more_tags(self):
        return {"allow_nan": True, "preserves_dtype": [np.float64, np.float32]}
class MaxAbsScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.

    This scaler can also be applied to sparse CSR or CSC matrices.

    `MaxAbsScaler` doesn't reduce the effect of outliers; it only linearly
    scales them down. For an example visualization, refer to :ref:`Compare
    MaxAbsScaler with other scalers <plot_all_scaling_max_abs_scaler_section>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.

        .. versionadded:: 0.17
           *scale_* attribute.

    max_abs_ : ndarray of shape (n_features,)
        Per feature maximum absolute value.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    See Also
    --------
    maxabs_scale : Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    Examples
    --------
    >>> from sklearn.preprocessing import MaxAbsScaler
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = MaxAbsScaler().fit(X)
    >>> transformer
    MaxAbsScaler()
    >>> transformer.transform(X)
    array([[ 0.5, -1. ,  1. ],
           [ 1. ,  0. ,  0. ],
           [ 0. ,  1. , -0.5]])
    """

    _parameter_constraints: dict = {"copy": ["boolean"]}

    def __init__(self, *, copy=True):
        # Initialize MaxAbsScaler with optional parameter 'copy'
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Check if the scaler has been fitted (attributes exist) and reset them
        if hasattr(self, "scale_"):
            del self.scale_  # Delete scaling factors
            del self.n_samples_seen_  # Delete number of samples seen
            del self.max_abs_  # Delete maximum absolute values
    def fit(self, X, y=None):
        """
        计算后续缩放所需的最大绝对值。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            用于计算每个特征的最小值和最大值，以便后续沿特征轴进行缩放的数据。

        y : None
            忽略。

        Returns
        -------
        self : object
            已拟合的缩放器。
        """
        # 在拟合之前重置内部状态
        self._reset()
        # 调用 partial_fit 方法进行拟合
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """
        在线计算 X 的最大绝对值，用于后续缩放。

        所有的 X 将作为单个批次进行处理。适用于由于非常大的 `n_samples` 数量或者 X 是从连续流中读取的情况。

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            用于计算均值和标准差，以便后续沿特征轴进行缩放的数据。

        y : None
            忽略。

        Returns
        -------
        self : object
            已拟合的缩放器。
        """
        xp, _ = get_namespace(X)

        first_pass = not hasattr(self, "n_samples_seen_")
        # 对数据进行验证和处理
        X = self._validate_data(
            X,
            reset=first_pass,
            accept_sparse=("csr", "csc"),
            dtype=_array_api.supported_float_dtypes(xp),
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            # 对稀疏数据进行轴向最小最大值计算
            mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
        else:
            # 对非稀疏数据使用指定的数组 API 计算最大绝对值
            max_abs = _array_api._nanmax(xp.abs(X), axis=0, xp=xp)

        if first_pass:
            # 如果是第一次拟合，则记录样本数
            self.n_samples_seen_ = X.shape[0]
        else:
            # 否则更新最大绝对值并增加样本数计数
            max_abs = xp.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        # 处理缩放时的零值情况
        self.scale_ = _handle_zeros_in_scale(max_abs, copy=True)
        return self
    def transform(self, X):
        """Scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data that should be scaled.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 检查模型是否已拟合（适配）当前数据集
        check_is_fitted(self)

        # 获取数据 X 所属的命名空间
        xp, _ = get_namespace(X)

        # 验证数据 X 是否符合要求，可能会进行类型转换和复制
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),  # 接受稀疏矩阵格式为 csr 或 csc
            copy=self.copy,  # 是否复制数据
            reset=False,  # 是否重置数据（未使用）
            dtype=_array_api.supported_float_dtypes(xp),  # 数据类型为支持的浮点数类型
            force_writeable=True,  # 强制数据可写
            force_all_finite="allow-nan",  # 允许数据包含 NaN
        )

        # 如果数据 X 是稀疏矩阵，则原地按列缩放
        if sparse.issparse(X):
            inplace_column_scale(X, 1.0 / self.scale_)
        else:
            # 否则，按 self.scale_ 缩放数据 X
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data that should be transformed back.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 检查模型是否已拟合（适配）当前数据集
        check_is_fitted(self)

        # 获取数据 X 所属的命名空间
        xp, _ = get_namespace(X)

        # 验证数据 X 是否符合要求，可能会进行类型转换和复制
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),  # 接受稀疏矩阵格式为 csr 或 csc
            copy=self.copy,  # 是否复制数据
            dtype=_array_api.supported_float_dtypes(xp),  # 数据类型为支持的浮点数类型
            force_writeable=True,  # 强制数据可写
            force_all_finite="allow-nan",  # 允许数据包含 NaN
        )

        # 如果数据 X 是稀疏矩阵，则原地按列缩放
        if sparse.issparse(X):
            inplace_column_scale(X, self.scale_)
        else:
            # 否则，按 self.scale_ 还原数据 X
            X *= self.scale_
        return X

    def _more_tags(self):
        # 返回更多标签，表明模型允许处理 NaN 值
        return {"allow_nan": True}
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数X可以是类数组或稀疏矩阵
        "axis": [Options(Integral, {0, 1})],  # 参数axis必须是整数0或1
    },
    prefer_skip_nested_validation=False,
)
def maxabs_scale(X, *, axis=0, copy=True):
    """Scale each feature to the [-1, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data.

    axis : {0, 1}, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.maxabs_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MaxAbsScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MaxAbsScaler(), LogisticRegression())`.

    See Also
    --------
    MaxAbsScaler : Performs scaling to the [-1, 1] range using
        the Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    Examples
    --------
    >>> from sklearn.preprocessing import maxabs_scale
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> maxabs_scale(X, axis=0)  # scale each column independently
    array([[-1. ,  1. ,  1. ],
           [-0.5,  0. ,  0.5]])
    >>> maxabs_scale(X, axis=1)  # scale each row independently
    array([[-1. ,  0.5,  1. ],
           [-1. ,  0. ,  1. ]])
    """
    # Unlike the scaler object, this function allows 1d input.
    # 与缩放器对象不同，此函数允许一维输入。

    # If copy is required, it will be done inside the scaler object.
    X = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )
    # 使用check_array函数验证和转换输入X，确保接受稀疏矩阵(csr或csc)，不进行复制，
    # 确保不是二维数据，使用浮点数数据类型，允许NaN存在。
    # 获取输入数组 X 的原始维度
    original_ndim = X.ndim
    
    # 如果 X 是一维数组，将其重塑为二维数组，列数为1
    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)
    
    # 创建一个 MaxAbsScaler 对象 s，用于最大绝对值缩放，可选择复制数据
    s = MaxAbsScaler(copy=copy)
    
    # 如果指定 axis 为 0，对 X 进行最大绝对值缩放并替换原数组
    if axis == 0:
        X = s.fit_transform(X)
    # 否则，对 X 的转置进行最大绝对值缩放，并将其再次转置回原来的形状
    else:
        X = s.fit_transform(X.T).T
    
    # 如果输入 X 原始是一维数组，将最终结果 X 恢复为一维
    if original_ndim == 1:
        X = X.ravel()
    
    # 返回经过最大绝对值缩放处理后的数组 X
    return X
class RobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the :meth:`transform` method.

    Standardization of a dataset is a common preprocessing for many machine
    learning estimators. Typically this is done by removing the mean and
    scaling to unit variance. However, outliers can often influence the sample
    mean / variance in a negative way. In such cases, using the median and the
    interquartile range often give better results. For an example visualization
    and comparison to other scalers, refer to :ref:`Compare RobustScaler with
    other scalers <plot_all_scaling_robust_scaler_section>`.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    with_centering : bool, default=True
        If `True`, center the data before scaling.
        This will cause :meth:`transform` to raise an exception when attempted
        on sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_scaling : bool, default=True
        If `True`, scale the data to interquartile range.

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, \
        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`. By default this is equal to
        the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
        quantile.

        .. versionadded:: 0.18

    copy : bool, default=True
        If `False`, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    unit_variance : bool, default=False
        If `True`, scale data so that normally distributed features have a
        variance of 1. In general, if the difference between the x-values of
        `q_max` and `q_min` for a standard normal distribution is greater
        than 1, the dataset will be scaled down. If less than 1, the dataset
        will be scaled up.

        .. versionadded:: 0.24

    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.
    """
    # 用于存储每个特征在训练集中的（缩放后的）四分位距。
    #
    # .. versionadded:: 0.17
    #    *scale_* 属性。
    
    scale_ : array of floats
    
    # 记录在 `fit` 过程中观察到的特征数量。
    #
    # .. versionadded:: 0.24
    
    n_features_in_ : int
    
    # 在 `fit` 过程中看到的特征的名称数组。仅当 `X` 的特征名全为字符串时才有定义。
    #
    # .. versionadded:: 1.0
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
    
    # 相关链接
    #
    # - https://en.wikipedia.org/wiki/Median
    # - https://en.wikipedia.org/wiki/Interquartile_range
    
    See Also
    --------
    robust_scale : 没有估计器 API 的等效函数。
    sklearn.decomposition.PCA : 通过 `whiten=True` 进一步消除特征之间的线性相关性。
    
    Notes
    -----
    
    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> transformer = RobustScaler().fit(X)
    >>> transformer
    RobustScaler()
    >>> transformer.transform(X)
    array([[ 0. , -2. ,  0. ],
           [-1. ,  0. ,  0.4],
           [ 1. ,  0. , -1.6]])
    
    """
    
    # 参数约束字典，用于存储 `_fit_context` 方法的参数约束
    _parameter_constraints: dict = {
        "with_centering": ["boolean"],
        "with_scaling": ["boolean"],
        "quantile_range": [tuple],
        "copy": ["boolean"],
        "unit_variance": ["boolean"],
    }
    
    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):
        # 初始化 RobustScaler 对象时的参数设置
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy
    
    @_fit_context(prefer_skip_nested_validation=True)
    # 计算中位数和分位数以用于后续特征轴上的标准化

    # 在拟合阶段，将稀疏矩阵转换为 csc 格式，以优化分位数计算
    X = self._validate_data(
        X,
        accept_sparse="csc",
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )

    # 获取分位数范围
    q_min, q_max = self.quantile_range
    # 检查分位数范围的有效性
    if not 0 <= q_min <= q_max <= 100:
        raise ValueError("Invalid quantile range: %s" % str(self.quantile_range))

    # 如果需要中心化
    if self.with_centering:
        # 如果输入数据是稀疏矩阵，抛出异常
        if sparse.issparse(X):
            raise ValueError(
                "Cannot center sparse matrices: use `with_centering=False`"
                " instead. See docstring for motivation and alternatives."
            )
        # 计算每个特征的中位数，并赋给 center_
        self.center_ = np.nanmedian(X, axis=0)
    else:
        self.center_ = None

    # 如果需要缩放
    if self.with_scaling:
        quantiles = []
        # 遍历每个特征的索引
        for feature_idx in range(X.shape[1]):
            # 如果输入数据是稀疏矩阵
            if sparse.issparse(X):
                # 获取非零数据的列
                column_nnz_data = X.data[
                    X.indptr[feature_idx] : X.indptr[feature_idx + 1]
                ]
                # 创建特征列数据
                column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
                column_data[: len(column_nnz_data)] = column_nnz_data
            else:
                # 否则直接获取特征列数据
                column_data = X[:, feature_idx]

            # 计算特征列数据的分位数，并存储在 quantiles 中
            quantiles.append(np.nanpercentile(column_data, self.quantile_range))

        # 转置 quantiles 数组
        quantiles = np.transpose(quantiles)

        # 计算缩放比例
        self.scale_ = quantiles[1] - quantiles[0]
        # 处理缩放比例中的零值
        self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        
        # 如果需要单位方差
        if self.unit_variance:
            # 根据分位数范围调整缩放比例
            adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
            self.scale_ = self.scale_ / adjust
    else:
        self.scale_ = None

    # 返回已拟合的标量对象
    return self
    # 数据转换方法，用于中心化和缩放数据

    def transform(self, X):
        """Center and scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 检查模型是否已拟合，确保可以进行数据转换
        check_is_fitted(self)
        # 验证并返回有效数据，根据需要接受稀疏矩阵
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            reset=False,
            force_all_finite="allow-nan",
        )

        # 如果输入数据是稀疏矩阵
        if sparse.issparse(X):
            # 如果需要进行缩放，则原地按列缩放
            if self.with_scaling:
                inplace_column_scale(X, 1.0 / self.scale_)
        else:
            # 如果需要中心化，则减去中心化的值
            if self.with_centering:
                X -= self.center_
            # 如果需要缩放，则除以缩放因子
            if self.with_scaling:
                X /= self.scale_
        # 返回转换后的数据数组
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The rescaled data to be transformed back.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 检查模型是否已拟合，确保可以进行逆转换
        check_is_fitted(self)
        # 验证并返回有效数据，根据需要接受稀疏矩阵
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            force_all_finite="allow-nan",
        )

        # 如果输入数据是稀疏矩阵
        if sparse.issparse(X):
            # 如果需要进行缩放，则原地按列缩放
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            # 如果需要缩放，则乘以缩放因子
            if self.with_scaling:
                X *= self.scale_
            # 如果需要中心化，则加上中心化的值
            if self.with_centering:
                X += self.center_
        # 返回逆转换后的数据数组
        return X

    def _more_tags(self):
        # 返回额外的标签，允许 NaN 值存在
        return {"allow_nan": True}
# 使用装饰器 validate_params 验证函数参数，确保参数 X 是 array-like 或 sparse matrix，axis 是整数 0 或 1
@validate_params(
    {"X": ["array-like", "sparse matrix"], "axis": [Options(Integral, {0, 1})]},
    prefer_skip_nested_validation=False,
)
# 定义 robust_scale 函数，用于数据标准化
def robust_scale(
    X,
    *,
    axis=0,  # 指定计算中位数和四分位数范围的轴，默认为 0，表示每列独立标准化
    with_centering=True,  # 是否对数据进行中心化，默认为 True
    with_scaling=True,  # 是否对数据进行缩放到单位方差，默认为 True
    quantile_range=(25.0, 75.0),  # 用于计算缩放的四分位数范围，默认为 (25.0, 75.0)
    copy=True,  # 是否复制数据，默认为 True
    unit_variance=False,  # 是否将数据缩放到单位方差，默认为 False
):
    """Standardize a dataset along any axis.

    Center to the median and component wise scale
    according to the interquartile range.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_sample, n_features)
        The data to center and scale.

    axis : int, default=0
        Axis used to compute the medians and IQR along. If 0,
        independently scale each feature, otherwise (if 1) scale
        each sample.

    with_centering : bool, default=True
        If `True`, center the data before scaling.

    with_scaling : bool, default=True
        If `True`, scale the data to unit variance (or equivalently,
        unit standard deviation).

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,\
        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`. By default this is equal to
        the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
        quantile.

        .. versionadded:: 0.18

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    unit_variance : bool, default=False
        If `True`, scale data so that normally distributed features have a
        variance of 1. In general, if the difference between the x-values of
        `q_max` and `q_min` for a standard normal distribution is greater
        than 1, the dataset will be scaled down. If less than 1, the dataset
        will be scaled up.

        .. versionadded:: 0.24

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    RobustScaler : Performs centering and scaling using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_centering=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    For a comparison of the different scalers, transformers, and normalizers,
    """
    # 函数体中的具体实现被省略，但是根据文档字符串，函数实现了数据的中心化和标准化操作
    """
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),  # 接受稀疏矩阵的类型：压缩行稀疏矩阵和压缩列稀疏矩阵
            copy=False,  # 不复制输入数据以节省内存
            ensure_2d=False,  # 不要求输入是二维数组
            dtype=FLOAT_DTYPES,  # 指定数据类型为浮点型
            force_all_finite="allow-nan",  # 允许数据中包含NaN值
        )
        original_ndim = X.ndim  # 记录原始输入数据的维度
    
        if original_ndim == 1:
            X = X.reshape(X.shape[0], 1)  # 如果原始数据是一维的，将其转换为二维的列向量
    
        s = RobustScaler(
            with_centering=with_centering,  # 是否进行中心化处理
            with_scaling=with_scaling,  # 是否进行缩放处理
            quantile_range=quantile_range,  # 分位数范围
            unit_variance=unit_variance,  # 是否使得方差为单位方差
            copy=copy,  # 是否复制输入数据
        )
        if axis == 0:
            X = s.fit_transform(X)  # 如果按列进行缩放，对X进行拟合和转换
        else:
            X = s.fit_transform(X.T).T  # 如果按行进行缩放，先对X进行转置，然后拟合和转换后再转置回来
    
        if original_ndim == 1:
            X = X.ravel()  # 如果原始数据是一维的，将结果展平成一维数组
    
        return X  # 返回缩放后的数据
    """
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X 参数接受 array-like 或稀疏矩阵类型的输入
        "norm": [StrOptions({"l1", "l2", "max"})],  # norm 参数必须是 {'l1', 'l2', 'max'} 中的一个字符串
        "axis": [Options(Integral, {0, 1})],  # axis 参数必须是整数类型，且取值为 {0, 1} 中的一个
        "copy": ["boolean"],  # copy 参数必须是布尔类型
        "return_norm": ["boolean"],  # return_norm 参数必须是布尔类型
    },
    prefer_skip_nested_validation=True,  # 设置验证参数时优先跳过嵌套验证
)
def normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : {0, 1}, default=1
        Define axis used to normalize the data along. If 1, independently
        normalize each sample, otherwise (if 0) normalize each feature.

    copy : bool, default=True
        If False, try to avoid a copy and normalize in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    return_norm : bool, default=False
        Whether to return the computed norms.

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Normalized input X.

    norms : ndarray of shape (n_samples, ) if axis=1 else (n_features, )
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See Also
    --------
    Normalizer : Performs normalization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    Examples
    --------
    >>> from sklearn.preprocessing import normalize
    >>> X = [[-2, 1, 2], [-1, 0, 1]]
    >>> normalize(X, norm="l1")  # L1 normalization each row independently
    array([[-0.4,  0.2,  0.4],
           [-0.5,  0. ,  0.5]])
    >>> normalize(X, norm="l2")  # L2 normalization each row independently
    array([[-0.66...,  0.33...,  0.66...],
           [-0.70...,  0.     ,  0.70...]])
    """
    if axis == 0:
        sparse_format = "csc"  # 如果 axis 为 0，则选择稀疏矩阵格式为 "csc"
    else:  # axis == 1:
        sparse_format = "csr"  # 否则选择稀疏矩阵格式为 "csr"

    xp, _ = get_namespace(X)  # 获取 X 的命名空间信息

    X = check_array(
        X,
        accept_sparse=sparse_format,  # 接受稀疏矩阵格式为 sparse_format
        copy=copy,  # 是否复制输入数据
        estimator="the normalize function",  # 估计器名称
        dtype=_array_api.supported_float_dtypes(xp),  # 支持的浮点数数据类型
        force_writeable=True,  # 是否强制可写
    )
    if axis == 0:
        X = X.T  # 如果 axis 为 0，则对 X 进行转置操作
    # 如果输入的矩阵 X 是稀疏矩阵
    if sparse.issparse(X):
        # 如果需要返回归一化后的矩阵并且指定了 'l1' 或者 'l2' 范数，则抛出未实现的异常
        if return_norm and norm in ("l1", "l2"):
            raise NotImplementedError(
                "return_norm=True is not implemented "
                "for sparse matrices with norm 'l1' "
                "or norm 'l2'"
            )
        # 如果指定了 'l1' 范数，就原地对稀疏矩阵 X 进行 'l1' 范数归一化
        if norm == "l1":
            inplace_csr_row_normalize_l1(X)
        # 如果指定了 'l2' 范数，就原地对稀疏矩阵 X 进行 'l2' 范数归一化
        elif norm == "l2":
            inplace_csr_row_normalize_l2(X)
        # 如果指定了 'max' 范数
        elif norm == "max":
            # 分别计算每行的最小值和最大值
            mins, maxes = min_max_axis(X, 1)
            # 计算每行的范数，取绝对值最大值
            norms = np.maximum(abs(mins), maxes)
            # 重复每行的范数值，以便与数据值进行逐元素操作
            norms_elementwise = norms.repeat(np.diff(X.indptr))
            # 创建一个掩码，用于排除范数为零的元素，避免除以零的情况
            mask = norms_elementwise != 0
            # 对非零范数的数据元素进行归一化处理
            X.data[mask] /= norms_elementwise[mask]
    else:
        # 如果输入的矩阵 X 不是稀疏矩阵
        # 根据指定的范数类型对每行进行归一化处理
        if norm == "l1":
            norms = xp.sum(xp.abs(X), axis=1)
        elif norm == "l2":
            norms = row_norms(X)
        elif norm == "max":
            norms = xp.max(xp.abs(X), axis=1)
        # 处理归一化过程中出现的零值，确保不会出现除以零的情况
        norms = _handle_zeros_in_scale(norms, copy=False)
        # 对矩阵 X 的每行数据进行归一化处理
        X /= norms[:, None]

    # 如果指定了 axis=0，则返回矩阵 X 的转置
    if axis == 0:
        X = X.T

    # 如果需要返回归一化后的矩阵及其范数值，则返回 X 和 norms
    if return_norm:
        return X, norms
    # 否则，仅返回归一化后的矩阵 X
    else:
        return X
class Normalizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    For an example visualization, refer to :ref:`Compare Normalizer with other
    scalers <plot_all_scaling_normalizer_section>`.

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    normalize : Equivalent function without the estimator API.

    Notes
    -----
    This estimator is :term:`stateless` and does not need to be fitted.
    However, we recommend to call :meth:`fit_transform` instead of
    :meth:`transform`, as parameter validation is only performed in
    :meth:`fit`.

    Examples
    --------
    >>> from sklearn.preprocessing import Normalizer
    >>> X = [[4, 1, 2, 2],
    ...      [1, 3, 9, 3],
    ...      [5, 7, 5, 1]]
    >>> transformer = Normalizer().fit(X)  # fit does nothing.
    >>> transformer
    Normalizer()
    >>> transformer.transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])
    """

    _parameter_constraints: dict = {
        "norm": [StrOptions({"l1", "l2", "max"})],  # 确定参数 `norm` 只能为 'l1', 'l2', 'max' 中的一种
        "copy": ["boolean"],  # 确定参数 `copy` 必须为布尔值
    }

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm  # 初始化对象的归一化方式参数
        self.copy = copy  # 初始化对象的复制方式参数

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to estimate the normalization parameters.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # 调用 _validate_data 方法验证输入数据 X 的格式，接受 CSR 格式的稀疏矩阵
        self._validate_data(X, accept_sparse="csr")
        # 返回已拟合的转换器对象 self
        return self

    def transform(self, X, copy=None):
        """
        Scale each non zero row of X to unit norm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.

        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 如果未指定 copy 参数，则使用 self.copy 的值
        copy = copy if copy is not None else self.copy
        # 调用 _validate_data 方法验证输入数据 X 的格式，强制可写，并按照 copy 参数进行处理
        X = self._validate_data(
            X, accept_sparse="csr", force_writeable=True, copy=copy, reset=False
        )
        # 对输入数据 X 进行归一化处理，沿行（axis=1）归一化，不复制数据
        return normalize(X, norm=self.norm, axis=1, copy=False)

    def _more_tags(self):
        """
        Return additional tags for the estimator.

        Returns
        -------
        dict
            Dictionary with additional tags.
        """
        # 返回一个字典，包含额外的标签信息，表示该估算器是无状态的并支持数组 API
        return {"stateless": True, "array_api_support": True}
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "threshold": [Interval(Real, None, None, closed="neither")],
        "copy": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def binarize(X, *, threshold=0.0, copy=True):
    """Boolean thresholding of array-like or scipy.sparse matrix.

    Read more in the :ref:`User Guide <preprocessing_binarization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to binarize, element by element.
        scipy.sparse matrices should be in CSR or CSC format to avoid an
        un-necessary copy.

    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : bool, default=True
        If False, try to avoid a copy and binarize in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an object dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    Binarizer : Performs binarization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).

    Examples
    --------
    >>> from sklearn.preprocessing import binarize
    >>> X = [[0.4, 0.6, 0.5], [0.6, 0.1, 0.2]]
    >>> binarize(X, threshold=0.5)
    array([[0., 1., 0.],
           [1., 0., 0.]])
    """
    # 将输入数据 X 转换为数组或稀疏矩阵格式，确保支持写操作和指定的复制策略
    X = check_array(X, accept_sparse=["csr", "csc"], force_writeable=True, copy=copy)
    
    # 如果 X 是稀疏矩阵
    if sparse.issparse(X):
        # 如果阈值 threshold 小于 0，则抛出异常，因为稀疏矩阵不支持负阈值的二值化
        if threshold < 0:
            raise ValueError("Cannot binarize a sparse matrix with threshold < 0")
        
        # 生成条件掩码，大于阈值的元素设为1，小于等于阈值的元素设为0
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        
        # 根据条件修改稀疏矩阵的元素值
        X.data[cond] = 1
        X.data[not_cond] = 0
        
        # 删除值为0的元素，优化稀疏矩阵结构
        X.eliminate_zeros()
    else:
        # 如果 X 是密集矩阵，生成条件掩码，大于阈值的元素设为1，小于等于阈值的元素设为0
        cond = X > threshold
        not_cond = np.logical_not(cond)
        
        # 根据条件修改密集矩阵的元素值
        X[cond] = 1
        X[not_cond] = 0
    
    # 返回二值化后的数据 X
    return X
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.
        特征值小于等于此阈值将被替换为0，大于阈值将被替换为1。
        在稀疏矩阵操作中，阈值不能小于0。

    copy : bool, default=True
        Set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).
        设置为False以执行原地二值化并避免复制（如果输入已经是numpy数组或scipy.sparse CSR矩阵）。

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        拟合期间观察到的特征数量。

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        拟合期间观察到的特征名称数组。仅在`X`具有全为字符串的特征名称时定义。

        .. versionadded:: 1.0

    See Also
    --------
    binarize : Equivalent function without the estimator API.
    KBinsDiscretizer : Bin continuous data into intervals.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the :class:`Binarizer` class.
    如果输入是稀疏矩阵，则只有非零值会被:class:`Binarizer`类更新。

    This estimator is :term:`stateless` and does not need to be fitted.
    However, we recommend to call :meth:`fit_transform` instead of
    :meth:`transform`, as parameter validation is only performed in
    :meth:`fit`.
    此估算器是无状态的，不需要拟合。但建议使用:meth:`fit_transform`而不是:meth:`transform`，因为参数验证仅在:meth:`fit`中执行。

    Examples
    --------
    >>> from sklearn.preprocessing import Binarizer
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = Binarizer().fit(X)  # fit does nothing.
    >>> transformer
    Binarizer()
    >>> transformer.transform(X)
    array([[1., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """

    _parameter_constraints: dict = {
        "threshold": [Real],
        "copy": ["boolean"],
    }

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_data(X, accept_sparse="csr")
        return self
    # 定义一个方法 `transform`，用于对输入的数据进行二值化处理
    def transform(self, X, copy=None):
        """Binarize each element of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.

        copy : bool
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # 如果未指定 copy 参数，则使用类的默认值 self.copy
        copy = copy if copy is not None else self.copy
        # 调用 self._validate_data 方法，验证并准备输入数据 X
        # 使用 CSR 或 CSC 格式的稀疏矩阵，强制可写，根据参数决定是否拷贝数据
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            force_writeable=True,
            copy=copy,
            reset=False,
        )
        # 调用 binarize 函数，对输入的 X 进行二值化处理，使用预设的阈值 self.threshold，且不进行拷贝操作
        return binarize(X, threshold=self.threshold, copy=False)

    # 定义一个私有方法 `_more_tags`，返回一个包含 "stateless": True 的字典
    def _more_tags(self):
        return {"stateless": True}
# 定义一个用于中心化任意核矩阵的类 KernelCenterer，继承了 ClassNamePrefixFeaturesOutMixin、TransformerMixin 和 BaseEstimator。
class KernelCenterer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    r"""Center an arbitrary kernel matrix :math:`K`.

    # 文档字符串（docstring）说明了该类的目的是用于中心化任意核矩阵 K。

    Let define a kernel :math:`K` such that:

    .. math::
        K(X, Y) = \phi(X) . \phi(Y)^{T}

    # 核函数 K 的定义，其中 :math:`K(X, Y)` 是通过映射函数 :math:`\phi` 计算得到的。

    :math:`\phi(X)` is a function mapping of rows of :math:`X` to a
    Hilbert space and :math:`K` is of shape `(n_samples, n_samples)`.

    # :math:`\phi(X)` 将 :math:`X` 的行映射到 Hilbert 空间，而核矩阵 K 的形状为 `(n_samples, n_samples)`。

    This class allows to compute :math:`\tilde{K}(X, Y)` such that:

    .. math::
        \tilde{K(X, Y)} = \tilde{\phi}(X) . \tilde{\phi}(Y)^{T}

    # 该类允许计算中心化的核矩阵 :math:`\tilde{K}(X, Y)`，其中 :math:`\tilde{\phi}(X)` 是映射到 Hilbert 空间中的中心化数据。

    :math:`\tilde{\phi}(X)` is the centered mapped data in the Hilbert
    space.

    # :math:`\tilde{\phi}(X)` 是 Hilbert 空间中的中心化映射数据。

    `KernelCenterer` centers the features without explicitly computing the
    mapping :math:`\phi(\cdot)`.

    # `KernelCenterer` 类在不显式计算映射 :math:`\phi(\cdot)` 的情况下进行特征中心化操作。

    Working with centered kernels is sometime expected when dealing with algebra computation such as eigendecomposition
    for :class:`~sklearn.decomposition.KernelPCA` for instance.

    # 在涉及到代数计算（例如对于 :class:`~sklearn.decomposition.KernelPCA` 的特征分解）时，通常希望使用中心化的核。

    Read more in the :ref:`User Guide <kernel_centering>`.

    # 详细信息请参阅用户指南中的 `kernel_centering` 部分。

    Attributes
    ----------
    K_fit_rows_ : ndarray of shape (n_samples,)
        Average of each column of kernel matrix.

    # `K_fit_rows_`：核矩阵每列的平均值。

    K_fit_all_ : float
        Average of kernel matrix.

    # `K_fit_all_`：核矩阵的平均值。

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # `n_features_in_`：在 `fit` 过程中看到的特征数目。

        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # `feature_names_in_`：在 `fit` 过程中看到的特征名称。仅当 `X` 具有所有字符串名称的特征时定义。

    See Also
    --------
    sklearn.kernel_approximation.Nystroem : Approximate a kernel map
        using a subset of the training data.

    # 参见 `sklearn.kernel_approximation.Nystroem`：使用训练数据子集近似核映射。

    References
    ----------
    .. [1] `Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller.
       "Nonlinear component analysis as a kernel eigenvalue problem."
       Neural computation 10.5 (1998): 1299-1319.
       <https://www.mlpack.org/papers/kpca.pdf>`_

    # 参考文献，引用了关于核特征分析的论文。

    Examples
    --------
    >>> from sklearn.preprocessing import KernelCenterer
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelCenterer().fit(K)
    >>> transformer
    KernelCenterer()
    >>> transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    """
    def fit(self, K, y=None):
        """Fit KernelCenterer.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        xp, _ = get_namespace(K)  # 获取数据的命名空间（可能是cupy或numpy）

        K = self._validate_data(K, dtype=_array_api.supported_float_dtypes(xp))  # 验证数据类型，并转换为支持的浮点数类型

        if K.shape[0] != K.shape[1]:
            raise ValueError(
                "Kernel matrix must be a square matrix."
                " Input is a {}x{} matrix.".format(K.shape[0], K.shape[1])
            )  # 检查核矩阵是否为方阵，抛出错误如果不是

        n_samples = K.shape[0]
        self.K_fit_rows_ = xp.sum(K, axis=0) / n_samples  # 计算每列的均值作为拟合的行
        self.K_fit_all_ = xp.sum(self.K_fit_rows_) / n_samples  # 计算所有行的均值作为总体拟合值
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
            Returns the instance itself.
        """
        check_is_fitted(self)  # 检查模型是否已拟合

        xp, _ = get_namespace(K)  # 获取数据的命名空间（可能是cupy或numpy）

        K = self._validate_data(
            K,
            copy=copy,
            force_writeable=True,
            dtype=_array_api.supported_float_dtypes(xp),
            reset=False,
        )  # 验证数据类型，并根据需要复制数据以进行修改

        K_pred_cols = (xp.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, None]  # 计算预测列的均值

        K -= self.K_fit_rows_  # 中心化每行
        K -= K_pred_cols  # 中心化每列
        K += self.K_fit_all_  # 添加总体拟合值

        return K

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Used by ClassNamePrefixFeaturesOutMixin. This model preserves the
        # number of input features but this is not a one-to-one mapping in the
        # usual sense. Hence the choice not to use OneToOneFeatureMixin to
        # implement get_feature_names_out for this class.
        return self.n_features_in_  # 返回输出特征的数量（与输入特征数量相同）

    def _more_tags(self):
        return {"pairwise": True, "array_api_support": True}
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # X参数需要是array-like或sparse matrix类型
        "value": [Interval(Real, None, None, closed="neither")],  # value参数需要是一个实数类型，且不能等于无穷，且不包含在闭区间内
    },
    prefer_skip_nested_validation=True,
)
def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features + 1)
        Same data with dummy feature added as first column.

    Examples
    --------
    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    """
    X = check_array(X, accept_sparse=["csc", "csr", "coo"], dtype=FLOAT_DTYPES)
    # 检查并转换X为指定格式的数组，接受稀疏矩阵格式，数据类型为FLOAT_DTYPES
    n_samples, n_features = X.shape
    # 获取数据集的样本数和特征数
    shape = (n_samples, n_features + 1)
    # 定义输出数据的形状，增加一个特征列
    if sparse.issparse(X):
        # 如果X是稀疏矩阵
        if X.format == "coo":
            # 如果X的格式是COO
            # 将列索引向右移动一位
            col = X.col + 1
            # 虚拟特征的列索引处处为0
            col = np.concatenate((np.zeros(n_samples), col))
            # 虚拟特征的行索引从0到n_samples-1
            row = np.concatenate((np.arange(n_samples), X.row))
            # 在数据中插入虚拟特征值n_samples次
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.coo_matrix((data, (row, col)), shape)
        elif X.format == "csc":
            # 如果X的格式是CSC
            # 由于需要添加n_samples个元素，因此移动索引指针
            indptr = X.indptr + n_samples
            # indptr[0]必须为0
            indptr = np.concatenate((np.array([0]), indptr))
            # 虚拟特征的行索引从0到n_samples-1
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # 在数据中插入虚拟特征值n_samples次
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.csc_matrix((data, indices, indptr), shape)
        else:
            # 对于其他格式的稀疏矩阵，递归调用add_dummy_feature函数
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        # 如果X不是稀疏矩阵，则在数据的左侧插入虚拟特征列
        return np.hstack((np.full((n_samples, 1), value), X))


class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    """
    # QuantileTransformer 类实现了一个转换器，用于将数据映射到一个均匀或正态分布的输出分布中。它利用分位数函数将原始值映射到一个均匀分布或正态分布。
    class QuantileTransformer(BaseEstimator, TransformerMixin):
        """
        used to map the original values to a uniform distribution. The obtained
        values are then mapped to the desired output distribution using the
        associated quantile function. Features values of new/unseen data that fall
        below or above the fitted range will be mapped to the bounds of the output
        distribution. Note that this transform is non-linear. It may distort linear
        correlations between variables measured at the same scale but renders
        variables measured at different scales more directly comparable.
    
        For example visualizations, refer to :ref:`Compare QuantileTransformer with
        other scalers <plot_all_scaling_quantile_transformer_section>`.
    
        Read more in the :ref:`User Guide <preprocessing_transformer>`.
    
        .. versionadded:: 0.19
        """
    
        # 初始化方法，设置转换器的参数
        def __init__(self, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                     subsample=10_000, random_state=None, copy=True):
            """
            Parameters
            ----------
            n_quantiles : int, default=1000 or n_samples
                Number of quantiles to be computed. It corresponds to the number
                of landmarks used to discretize the cumulative distribution function.
                If n_quantiles is larger than the number of samples, n_quantiles is set
                to the number of samples as a larger number of quantiles does not give
                a better approximation of the cumulative distribution function
                estimator.
    
            output_distribution : {'uniform', 'normal'}, default='uniform'
                Marginal distribution for the transformed data. The choices are
                'uniform' (default) or 'normal'.
    
            ignore_implicit_zeros : bool, default=False
                Only applies to sparse matrices. If True, the sparse entries of the
                matrix are discarded to compute the quantile statistics. If False,
                these entries are treated as zeros.
    
            subsample : int or None, default=10_000
                Maximum number of samples used to estimate the quantiles for
                computational efficiency. Note that the subsampling procedure may
                differ for value-identical sparse and dense matrices.
                Disable subsampling by setting `subsample=None`.
    
                .. versionadded:: 1.5
                   The option `None` to disable subsampling was added.
    
            random_state : int, RandomState instance or None, default=None
                Determines random number generation for subsampling and smoothing
                noise.
                Please see ``subsample`` for more details.
                Pass an int for reproducible results across multiple function calls.
                See :term:`Glossary <random_state>`.
    
            copy : bool, default=True
                Set to False to perform inplace transformation and avoid a copy (if the
                input is already a numpy array).
            """
            self.n_quantiles = n_quantiles
            self.output_distribution = output_distribution
            self.ignore_implicit_zeros = ignore_implicit_zeros
            self.subsample = subsample
            self.random_state = random_state
            self.copy = copy
    
        # 实现 QuantileTransformer 转换的方法
        def fit(self, X, y=None):
            """
            Fit the quantile transformer to X.
    
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data used to compute the quantile statistics.
    
            Returns
            -------
            self : object
                Fitted transformer.
            """
            # Fit method implementation
            return self
    
        # 使用已拟合的 QuantileTransformer 对 X 进行转换的方法
        def transform(self, X):
            """
            Transform X using the quantile transformation.
    
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data used to compute the quantile statistics.
    
            Returns
            -------
            X_transformed : ndarray of shape (n_samples, n_features)
                The transformed data.
            """
            # Transform method implementation
            return X
    
        # 返回 QuantileTransformer 的参数设置和状态的方法
        def get_params(self, deep=True):
            """
            Get parameters for this estimator.
    
            Parameters
            ----------
            deep : bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
    
            Returns
            -------
            params : dict
                Parameter names mapped to their values.
            """
            return {
                'n_quantiles': self.n_quantiles,
                'output_distribution': self.output_distribution,
                'ignore_implicit_zeros': self.ignore_implicit_zeros,
                'subsample': self.subsample,
                'random_state': self.random_state,
                'copy': self.copy
            }
    
        # 设置 QuantileTransformer 参数的方法
        def set_params(self, **params):
            """
            Set the parameters of this estimator.
    
            Returns
            -------
            self : object
                Estimator instance.
            """
            for parameter, value in params.items():
                setattr(self, parameter, value)
            return self
    # _parameter_constraints 是一个字典，用于定义 QuantileTransformer 的参数约束。
    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],  # n_quantiles 参数的约束条件，必须为整数且大于等于1
        "output_distribution": [StrOptions({"uniform", "normal"})],  # output_distribution 参数的约束条件，必须为 "uniform" 或 "normal" 中的一个字符串
        "ignore_implicit_zeros": ["boolean"],  # ignore_implicit_zeros 参数的约束条件，必须为布尔值
        "subsample": [Interval(Integral, 1, None, closed="left"), None],  # subsample 参数的约束条件，必须为大于等于1的整数或者为 None
        "random_state": ["random_state"],  # random_state 参数的约束条件，可以为任意类型，但通常用于随机数生成器的种子
        "copy": ["boolean"],  # copy 参数的约束条件，必须为布尔值
    }
    
    def __init__(
        self,
        *,
        n_quantiles=1000,  # 初始化函数，设置 n_quantiles 参数，默认为 1000
        output_distribution="uniform",  # 初始化函数，设置 output_distribution 参数，默认为 "uniform"
        ignore_implicit_zeros=False,  # 初始化函数，设置 ignore_implicit_zeros 参数，默认为 False
        subsample=10_000,  # 初始化函数，设置 subsample 参数，默认为 10000
        random_state=None,  # 初始化函数，设置 random_state 参数，默认为 None
        copy=True,  # 初始化函数，设置 copy 参数，默认为 True
    ):
        self.n_quantiles = n_quantiles  # 将初始化函数中的 n_quantiles 参数赋值给实例变量 self.n_quantiles
        self.output_distribution = output_distribution  # 将初始化函数中的 output_distribution 参数赋值给实例变量 self.output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros  # 将初始化函数中的 ignore_implicit_zeros 参数赋值给实例变量 self.ignore_implicit_zeros
        self.subsample = subsample  # 将初始化函数中的 subsample 参数赋值给实例变量 self.subsample
        self.random_state = random_state  # 将初始化函数中的 random_state 参数赋值给实例变量 self.random_state
        self.copy = copy  # 将初始化函数中的 copy 参数赋值给实例变量 self.copy
    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            # 如果忽略隐式零值，则发出警告，这个参数只在稀疏矩阵中有效，对稠密矩阵无影响。
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        # 获取样本数和特征数
        n_samples, n_features = X.shape
        # 将参考值乘以100，用于计算百分位数
        references = self.references_ * 100

        if self.subsample is not None and self.subsample < n_samples:
            # 如果指定了子样本数，并且小于总样本数，从 `X` 中取一个子样本
            X = resample(
                X, replace=False, n_samples=self.subsample, random_state=random_state
            )

        # 计算百分位数，并存储在 `quantiles_` 中
        self.quantiles_ = np.nanpercentile(X, references, axis=0)
        
        # 由于 `np.nanpercentile` 存在浮点精度问题，确保百分位数单调递增。
        # 这是 numpy 中的一个上游问题，详情见：
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)
    # 定义一个方法，用于在稀疏矩阵上计算百分位数

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative. If a sparse matrix is provided,
            it will be converted into a sparse ``csc_matrix``.
        """
        
        # 获取稀疏矩阵的样本数和特征数
        n_samples, n_features = X.shape
        
        # 将参考值扩展为百分比
        references = self.references_ * 100

        # 初始化一个空列表来存储百分位数
        self.quantiles_ = []

        # 遍历每一个特征
        for feature_idx in range(n_features):
            # 获取当前特征列的非零元素数据
            column_nnz_data = X.data[X.indptr[feature_idx] : X.indptr[feature_idx + 1]]
            
            # 如果指定了子样本大小且当前列非零元素数大于子样本大小
            if self.subsample is not None and len(column_nnz_data) > self.subsample:
                # 计算要抽样的子样本大小
                column_subsample = self.subsample * len(column_nnz_data) // n_samples
                
                # 如果忽略隐式零值，则创建指定形状的零数组
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample, dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                
                # 从当前列非零数据中无放回抽样填充列数据
                column_data[:column_subsample] = random_state.choice(
                    column_nnz_data, size=column_subsample, replace=False
                )
            else:
                # 如果忽略隐式零值，则创建指定形状的零数组
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data), dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                
                # 将当前列非零数据填充到列数据中
                column_data[: len(column_nnz_data)] = column_nnz_data

            # 如果列数据为空
            if not column_data.size:
                # 如果没有非零元素，则强制将百分位数设置为零数组
                self.quantiles_.append([0] * len(references))
            else:
                # 否则，计算列数据的百分位数并添加到 quantiles_
                self.quantiles_.append(np.nanpercentile(column_data, references))
        
        # 将 quantiles_ 转置，使得每行对应一个特征的所有百分位数
        self.quantiles_ = np.transpose(self.quantiles_)
        
        # 由于 `np.nanpercentile` 存在浮点精度错误，确保百分位数单调递增
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        # 检查是否设置了子样本数量，并且确保 quantiles 数量不超过子样本数量
        if self.subsample is not None and self.n_quantiles > self.subsample:
            raise ValueError(
                "The number of quantiles cannot be greater than"
                " the number of samples used. Got {} quantiles"
                " and {} samples.".format(self.n_quantiles, self.subsample)
            )

        # 检查并准备输入数据 X，确保格式正确
        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        # 如果 quantiles 数量大于样本数量，发出警告并将 quantiles 数量设置为样本数量
        if self.n_quantiles > n_samples:
            warnings.warn(
                "n_quantiles (%s) is greater than the total number "
                "of samples (%s). n_quantiles is set to "
                "n_samples." % (self.n_quantiles, n_samples)
            )
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        # 检查随机数生成器并初始化
        rng = check_random_state(self.random_state)

        # 创建作为参考的分位数
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)

        # 根据输入的稀疏性调用相应的拟合方法
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        # 返回拟合后的转换器对象
        return self

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        # 验证数据 X 的合法性，包括类型、稀疏性等
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse="csc",
            copy=copy,
            dtype=FLOAT_DTYPES,
            # 仅在转换时进行强制可写检查，因为 QuantileTransformer 只在此时执行原地操作
            force_writeable=True if not in_fit else None,
            force_all_finite="allow-nan",
        )
        # 在忽略隐式零值为 false 且调用 fit 或 transform 时，仅接受正的稀疏矩阵
        with np.errstate(invalid="ignore"):  # 隐藏 NaN 比较警告
            if (
                not accept_sparse_negative
                and not self.ignore_implicit_zeros
                and (sparse.issparse(X) and np.any(X.data < 0))
            ):
                raise ValueError(
                    "QuantileTransformer only accepts non-negative sparse matrices."
                )

        # 返回验证后的数据 X
        return X
    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        # 检查输入数据是否为稀疏矩阵，如果是，按列进行变换
        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                # 获取当前特征列的切片范围
                column_slice = slice(X.indptr[feature_idx], X.indptr[feature_idx + 1])
                # 对当前特征列进行变换
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx], inverse
                )
        else:
            # 对每个特征列进行变换
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
                )

        # 返回变换后的数据
        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 检查并准备输入数据
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        # 应用正向变换
        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of (n_samples, n_features)
            The projected data.
        """
        # 确保模型已拟合
        check_is_fitted(self)
        # 检查并准备输入数据，允许稀疏矩阵中的负值
        X = self._check_inputs(
            X, in_fit=False, accept_sparse_negative=True, copy=self.copy
        )

        # 应用反向变换
        return self._transform(X, inverse=True)

    def _more_tags(self):
        # 返回更多标签信息，允许 NaN 值
        return {"allow_nan": True}
# 使用装饰器 @validate_params 对 quantile_transform 函数进行参数验证和类型检查
@validate_params(
    # 定义参数 X 的类型为 array-like 或 sparse matrix，axis 的类型为整数，并且只能取值 0 或 1
    {"X": ["array-like", "sparse matrix"], "axis": [Options(Integral, {0, 1})]},
    # 设置是否跳过嵌套验证为 False，即对嵌套参数也进行验证
    prefer_skip_nested_validation=False,
)
# 定义 quantile_transform 函数，用于特征转换基于分位数信息
def quantile_transform(
    X,
    *,
    axis=0,  # 默认轴为 0，表示按列进行处理
    n_quantiles=1000,  # 默认使用 1000 个分位数
    output_distribution="uniform",  # 默认输出均匀分布
    ignore_implicit_zeros=False,  # 默认不忽略稀疏矩阵中的隐式零值
    subsample=int(1e5),  # 默认子采样数量为 100000
    random_state=None,  # 默认随机状态为 None
    copy=True,  # 默认复制数据
):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to transform.

    axis : int, default=0
        Axis used to compute the means and standard deviations along. If 0,
        transform each feature, otherwise (if 1) transform each sample.

    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.
    subsample : int or None, default=1e5
        # 用于估计分位数的最大样本数，以提高计算效率。
        # 注意，稀疏矩阵和密集矩阵的值相同时，子采样过程可能不同。
        # 通过将 `subsample=None` 来禁用子采样。
        .. versionadded:: 1.5
           添加了 `None` 选项以禁用子采样。

    random_state : int, RandomState instance or None, default=None
        # 控制子采样和平滑噪声的随机数生成。
        # 详细信息请参见 `subsample`。
        # 通过传递一个整数可以获得可重复的结果。
        # 参见 :term:`Glossary <random_state>`。
        
    copy : bool, default=True
        # 如果为 False，则尝试避免复制并原地转换。
        # 但不能保证始终在原地操作；例如，如果数据是具有整数dtype的numpy数组，则即使 copy=False 也会返回副本。
        
        .. versionchanged:: 0.23
            `copy` 的默认值从 0.23 版本更改为 False。

    Returns
    -------
    Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
        # 转换后的数据。

    See Also
    --------
    QuantileTransformer : 使用 Transformer API（例如作为预处理的一部分:class:`~sklearn.pipeline.Pipeline`）执行基于分位数的缩放。
    power_transform : 使用功率变换将数据映射到正态分布。
    scale : 执行标准化，速度更快，但对异常值不太鲁棒。
    robust_scale : 执行健壮的标准化，消除异常值的影响，但不将异常值和内囊者放在同一比例上。

    Notes
    -----
    NaNs 被视为缺失值：在拟合过程中忽略，在转换过程中保留。

    .. warning:: 数据泄漏的风险

        不要在分割为训练集和测试集之前对整个数据集应用 :func:`~sklearn.preprocessing.quantile_transform`。
        这将导致模型评估的偏见，因为信息会从测试集泄漏到训练集。
        通常建议在 :ref:`Pipeline <pipeline>` 中使用 :class:`~sklearn.preprocessing.QuantileTransformer`，
        以防止大多数数据泄漏风险：`pipe = make_pipeline(QuantileTransformer(), LogisticRegression())`。

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    # 导入数组处理函数，可能是NumPy的函数
    array([...])
    """
    # 使用QuantileTransformer对数据进行分位数转换
    n = QuantileTransformer(
        n_quantiles=n_quantiles,                    # 指定分位数的数量
        output_distribution=output_distribution,    # 指定输出分布的类型
        subsample=subsample,                        # 子样本大小，用于估计密度和累积分布
        ignore_implicit_zeros=ignore_implicit_zeros,  # 是否忽略隐式的零值
        random_state=random_state,                  # 随机数种子
        copy=copy,                                  # 是否复制输入数据
    )
    if axis == 0:
        # 如果axis为0，对原始数据X进行拟合和转换
        X = n.fit_transform(X)
    else:  # axis == 1
        # 如果axis为1，对原始数据X的转置进行拟合和转换，然后再转置回来
        X = n.fit_transform(X.T).T
    # 返回转换后的数据X
    return X
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    For an example visualization, refer to :ref:`Compare PowerTransformer with
    other scalers <plot_all_scaling_power_transformer_section>`. To see the
    effect of Box-Cox and Yeo-Johnson transformations on different
    distributions, see:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_map_data_to_normal.py`.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : ndarray of float of shape (n_features,)
        The parameters of the power transformation for the selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    References
    ----------

    .. [1] :doi:`I.K. Yeo and R.A. Johnson, "A new family of power
           transformations to improve normality or symmetry." Biometrika,
           87(4), pp.954-959, (2000). <10.1093/biomet/87.4.954>`
    """

    def __init__(self, method='yeo-johnson', standardize=True, copy=True):
        """Initialize PowerTransformer with specified parameters.

        Parameters
        ----------
        method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
            The power transform method.

        standardize : bool, default=True
            Whether to apply zero-mean, unit-variance normalization.

        copy : bool, default=True
            Whether to perform inplace computation during transformation.
        """
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the PowerTransformer to X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to learn the power transformation parameters.

        y : None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Fit the PowerTransformer to X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to learn the power transformation parameters.
        """
        X = self._check_input(X)
        self.lambdas_ = np.zeros(X.shape[1])  # Initialize lambdas array

        # Loop through each feature to estimate lambda
        for feature_idx in range(X.shape[1]):
            feature = X[:, feature_idx]

            # Calculate optimal lambda for the current feature
            if self.method == 'box-cox':
                _, self.lambdas_[feature_idx] = stats.boxcox(feature)
            elif self.method == 'yeo-johnson':
                _, self.lambdas_[feature_idx] = stats.yeojohnson(feature)

    def transform(self, X):
        """Apply the power transformation to X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        X = self._check_input(X)
        X_trans = np.copy(X) if self.copy else X

        # Transform each feature using pre-computed lambdas
        for feature_idx in range(X.shape[1]):
            if self.method == 'box-cox':
                X_trans[:, feature_idx] = stats.boxcox(
                    X_trans[:, feature_idx], lmbda=self.lambdas_[feature_idx]
                )
            elif self.method == 'yeo-johnson':
                X_trans[:, feature_idx] = stats.yeojohnson(
                    X_trans[:, feature_idx], lmbda=self.lambdas_[feature_idx]
                )

        if self.standardize:
            X_trans = self._standardize(X_trans)

        return X_trans

    def _check_input(self, X):
        """Validate and convert input to ndarray if necessary.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The validated and possibly converted input.
        """
        X = check_array(X, accept_sparse='csc')
        return X

    def _standardize(self, X):
        """Standardize the transformed data to zero mean and unit variance.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X_standardized : ndarray of shape (n_samples, n_features)
            The standardized data.
        """
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        X_standardized = (X - X_mean) / X_std
        return X_standardized
    """
    .. [2] :doi:`G.E.P. Box and D.R. Cox, "An Analysis of Transformations",
           Journal of the Royal Statistical Society B, 26, 211-252 (1964).
           <10.1111/j.2517-6161.1964.tb00553.x>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer()
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]
    """

    # 定义参数约束字典，指定各参数的允许取值
    _parameter_constraints: dict = {
        "method": [StrOptions({"yeo-johnson", "box-cox"})],  # method 参数可选值为 "yeo-johnson" 或 "box-cox"
        "standardize": ["boolean"],  # standardize 参数应为布尔值
        "copy": ["boolean"],  # copy 参数应为布尔值
    }

    # PowerTransformer 类的初始化方法
    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method  # 初始化 method 属性
        self.standardize = standardize  # 初始化 standardize 属性
        self.copy = copy  # 初始化 copy 属性

    # 装饰器，用于 fit 方法，用于估计每个特征的最优参数 lambda
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Estimate the optimal parameter lambda for each feature.

        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._fit(X, y=y, force_transform=False)  # 调用 _fit 方法进行拟合，不强制转换
        return self  # 返回 self 对象

    # 装饰器，用于 fit_transform 方法，用于拟合数据并进行转换
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Fit `PowerTransformer` to `X`, then transform `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters
            and to be transformed using a power transformation.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self._fit(X, y, force_transform=True)  # 调用 _fit 方法进行拟合并强制转换，返回转换后的数据
    # 根据输入的数据 X 进行拟合操作，可能会进行数据变换和标准化
    def _fit(self, X, y=None, force_transform=False):
        # 检查输入数据 X，确保在拟合过程中进行了必要的验证和处理
        X = self._check_input(X, in_fit=True, check_positive=True)

        # 如果不复制数据且不强制转换，说明是从 fit() 调用过来的
        if not self.copy and not force_transform:
            X = X.copy()  # 强制复制数据，以防 fit() 改变原始数据 X

        # 计算样本数
        n_samples = X.shape[0]
        # 计算每个特征的均值和方差
        mean = np.mean(X, axis=0, dtype=np.float64)
        var = np.var(X, axis=0, dtype=np.float64)

        # 根据选定的变换方法选择优化函数
        optim_function = {
            "box-cox": self._box_cox_optimize,
            "yeo-johnson": self._yeo_johnson_optimize,
        }[self.method]

        # 根据选定的变换方法选择变换函数
        transform_function = {
            "box-cox": boxcox,
            "yeo-johnson": self._yeo_johnson_transform,
        }[self.method]

        # 忽略无效值警告，例如 NaN
        with np.errstate(invalid="ignore"):  # 隐藏 NaN 警告
            # 初始化 lambdas_ 属性为一个空数组，用于存储每个特征的变换参数
            self.lambdas_ = np.empty(X.shape[1], dtype=X.dtype)
            # 对每列特征进行遍历
            for i, col in enumerate(X.T):
                # 对于 yeo-johnson 方法，保持常数特征不变
                # lambda=1 对应于恒等变换
                is_constant_feature = _is_constant_feature(var[i], mean[i], n_samples)
                if self.method == "yeo-johnson" and is_constant_feature:
                    self.lambdas_[i] = 1.0  # 将 lambda 设置为 1.0
                    continue

                # 计算当前特征的最优 lambda 值
                self.lambdas_[i] = optim_function(col)

                # 如果需要标准化或强制转换，则对当前特征进行变换
                if self.standardize or force_transform:
                    X[:, i] = transform_function(X[:, i], self.lambdas_[i])

        # 如果需要标准化，则初始化 _scaler 属性为一个标准化器对象，并进行相应的操作
        if self.standardize:
            self._scaler = StandardScaler(copy=False).set_output(transform="default")
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

        # 返回拟合（可能变换）后的数据 X
        return X

    # 对数据 X 应用已拟合的 lambda 参数进行变换
    def transform(self, X):
        """Apply the power transform to each feature using the fitted lambdas.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed using a power transformation.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        # 检查是否已拟合，如果未拟合则抛出异常
        check_is_fitted(self)
        # 检查输入数据 X，并确保在转换过程中进行了必要的验证和处理
        X = self._check_input(X, in_fit=False, check_positive=True, check_shape=True)

        # 根据选定的变换方法选择变换函数
        transform_function = {
            "box-cox": boxcox,
            "yeo-johnson": self._yeo_johnson_transform,
        }[self.method]

        # 对每个特征按照已拟合的 lambda 参数进行变换
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid="ignore"):  # 隐藏 NaN 警告
                X[:, i] = transform_function(X[:, i], lmbda)

        # 如果需要标准化，则使用已保存的标准化器对数据 X 进行变换
        if self.standardize:
            X = self._scaler.transform(X)

        # 返回变换后的数据 X
        return X
    def inverse_transform(self, X):
        """Apply the inverse power transformation using the fitted lambdas.

        The inverse of the Box-Cox transformation is given by::

            if lambda_ == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_)

        The inverse of the Yeo-Johnson transformation is given by::

            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The original data.
        """
        # 检查模型是否已拟合
        check_is_fitted(self)
        # 检查输入数据的格式和形状是否符合要求
        X = self._check_input(X, in_fit=False, check_shape=True)

        # 如果需要标准化，对数据进行逆标准化
        if self.standardize:
            X = self._scaler.inverse_transform(X)

        # 根据选择的方法调用相应的逆变换函数
        inv_fun = {
            "box-cox": self._box_cox_inverse_tranform,
            "yeo-johnson": self._yeo_johnson_inverse_transform,
        }[self.method]
        # 对每个特征应用逆变换
        for i, lmbda in enumerate(self.lambdas_):
            # 忽略计算过程中可能产生的 NaN 警告
            with np.errstate(invalid="ignore"):
                X[:, i] = inv_fun(X[:, i], lmbda)

        # 返回原始数据
        return X

    def _box_cox_inverse_tranform(self, x, lmbda):
        """Return inverse-transformed input x following Box-Cox inverse
        transform with parameter lambda.
        """
        # 根据 Box-Cox 变换的逆公式进行逆变换
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        """
        # 初始化逆变换结果的数组
        x_inv = np.zeros_like(x)
        # 根据 x 的正负情况进行不同的逆变换计算
        pos = x >= 0

        # 当 x >= 0 时的逆变换计算
        if abs(lmbda) < np.spacing(1.0):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # 当 x < 0 时的逆变换计算
        if abs(lmbda - 2) > np.spacing(1.0):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-x[~pos])

        return x_inv
    def _yeo_johnson_transform(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """
        # 创建一个与输入x形状相同的全零数组
        out = np.zeros_like(x)
        # 创建一个布尔掩码，标记x中大于等于0的位置
        pos = x >= 0  # binary mask

        # 当 x >= 0 时的转换
        if abs(lmbda) < np.spacing(1.0):
            # 当 lambda 接近于0时，使用稳定的log1p函数进行转换
            out[pos] = np.log1p(x[pos])
        else:  # lmbda != 0
            # 否则根据 Yeo-Johnson 变换公式进行转换
            out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

        # 当 x < 0 时的转换
        if abs(lmbda - 2) > np.spacing(1.0):
            # 当 lambda 与2的差值足够大时，使用稳定的log1p函数进行负值转换
            out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            # 否则直接使用-log1p函数进行负值转换
            out[~pos] = -np.log1p(-x[~pos])

        return out

    def _box_cox_optimize(self, x):
        """Find and return optimal lambda parameter of the Box-Cox transform by
        MLE, for observed data x.

        We here use scipy builtins which uses the brent optimizer.
        """
        # 创建一个布尔掩码，标记x中的NaN值
        mask = np.isnan(x)
        if np.all(mask):
            raise ValueError("Column must not be all nan.")

        # 由于NaN值的存在，需要将其剔除后计算lambda值
        _, lmbda = stats.boxcox(x[~mask], lmbda=None)

        return lmbda

    def _yeo_johnson_optimize(self, x):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.

        Like for Box-Cox, MLE is done via the brent optimizer.
        """
        # 定义一个极小值，避免浮点数运算中出现除以零的错误
        x_tiny = np.finfo(np.float64).tiny

        def _neg_log_likelihood(lmbda):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""
            # 使用当前类的_yeo_johnson_transform方法对x进行转换
            x_trans = self._yeo_johnson_transform(x, lmbda)
            n_samples = x.shape[0]
            x_trans_var = x_trans.var()

            # 拒绝方差小于极小值的转换数据，避免np.log引发RuntimeWarning
            if x_trans_var < x_tiny:
                return np.inf

            # 计算对数似然函数
            log_var = np.log(x_trans_var)
            loglike = -n_samples / 2 * log_var
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # 由于NaN值的存在，需要将其剔除后再进行优化计算lambda
        x = x[~np.isnan(x)]
        # 选择区间[-2, 2]进行lambda的优化计算，类似于Box-Cox方法
        return optimize.brent(_neg_log_likelihood, brack=(-2, 2))
    # 验证在拟合和转换之前的输入数据是否有效

    # 参数说明：
    # X : 形状为 (n_samples, n_features) 的类数组对象
    # in_fit : bool
    #     指示 `_check_input` 方法是否从 `fit` 或其他方法（如 `predict`, `transform` 等）中调用
    # check_positive : bool, default=False
    #     如果为 True，并且 ``self.method=='box-cox'``，则检查所有数据是否为正且非零
    # check_shape : bool, default=False
    #     如果为 True，则检查 n_features 是否与 self.lambdas_ 的长度匹配

    X = self._validate_data(
        X,
        ensure_2d=True,            # 确保输入数据是二维的
        dtype=FLOAT_DTYPES,        # 指定数据类型为 FLOAT_DTYPES 中定义的类型
        force_writeable=True,      # 强制数据可写
        copy=self.copy,            # 根据 self.copy 指示是否复制输入数据
        force_all_finite="allow-nan",  # 检查数据中的非有限数值，允许 NaN
        reset=in_fit,              # 根据 in_fit 参数重置验证器的状态
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        # 如果 check_positive 为 True，并且使用 Box-Cox 方法，并且 X 中存在小于等于零的值
        if check_positive and self.method == "box-cox" and np.nanmin(X) <= 0:
            raise ValueError(
                "The Box-Cox transformation can only be "
                "applied to strictly positive data"
            )

    # 如果 check_shape 为 True，并且输入数据 X 的特征数量与 self.lambdas_ 的长度不匹配
    if check_shape and not X.shape[1] == len(self.lambdas_):
        raise ValueError(
            "Input data has a different number of features "
            "than fitting data. Should have {n}, data has {m}".format(
                n=len(self.lambdas_), m=X.shape[1]
            )
        )

    # 返回验证后的输入数据 X
    return X

# 返回一个字典，指定了更多的标签信息，此处为 {"allow_nan": True}
def _more_tags(self):
    return {"allow_nan": True}
@validate_params(
    {"X": ["array-like"]},  # 使用装饰器 @validate_params 对函数参数进行验证，确保 X 是一个类数组对象
    prefer_skip_nested_validation=False,  # 设置装饰器的参数 prefer_skip_nested_validation 为 False
)
def power_transform(X, method="yeo-johnson", *, standardize=True, copy=True):
    """Parametric, monotonic transformation to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        .. versionchanged:: 0.23
            The default value of the `method` parameter changed from
            'box-cox' to 'yeo-johnson' in 0.23.

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        If False, try to avoid a copy and transform in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    Returns
    -------
    X_trans : ndarray of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    PowerTransformer : Equivalent transformation with the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see: :ref:`sphx_glr_auto_examples_preprocessing_plot_all_scaling.py`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    """
    # 创建一个 PowerTransformer 对象，用于对数据进行幂变换
    pt = PowerTransformer(method=method, standardize=standardize, copy=copy)
    # 对输入的数据 X 进行拟合和变换，返回转换后的数据
    return pt.fit_transform(X)
```