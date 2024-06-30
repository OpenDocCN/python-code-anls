# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_discretization.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入 Integral 类型，用于数值判断
from numbers import Integral

# 导入 NumPy 库，并简称为 np
import numpy as np

# 从当前包的 base 模块中导入基础估计器、转换器混合类和内部 _fit_context
from ..base import BaseEstimator, TransformerMixin, _fit_context
# 从 utils 模块中导入重新采样函数 resample
from ..utils import resample
# 从 utils 模块的 _param_validation 子模块中导入 Interval、Options 和 StrOptions 类
from ..utils._param_validation import Interval, Options, StrOptions
# 从 utils 模块的 deprecation 子模块中导入 _deprecate_Xt_in_inverse_transform 函数
from ..utils.deprecation import _deprecate_Xt_in_inverse_transform
# 从 utils 模块的 stats 子模块中导入 _weighted_percentile 函数
from ..utils.stats import _weighted_percentile
# 从 utils 模块的 validation 子模块中导入数据校验相关函数
from ..utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    check_array,
    check_is_fitted,
)
# 从当前模块的 _encoders 子模块中导入 OneHotEncoder 类
from ._encoders import OneHotEncoder


class KBinsDiscretizer(TransformerMixin, BaseEstimator):
    """
    将连续数据分成区间。

    详细信息请参阅：:ref:`用户指南 <preprocessing_discretization>`。

    .. versionadded:: 0.20

    Parameters
    ----------
    n_bins : int or array-like of shape (n_features,), default=5
        要生成的区间数。如果 ``n_bins < 2``，则引发 ValueError。

    encode : {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
        用于编码转换结果的方法。

        - 'onehot': 使用独热编码对转换结果进行编码，并返回稀疏矩阵。被忽略的特征始终堆叠在右侧。
        - 'onehot-dense': 使用独热编码对转换结果进行编码，并返回密集数组。被忽略的特征始终堆叠在右侧。
        - 'ordinal': 返回编码为整数值的区间标识符。

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        用于定义区间宽度的策略。

        - 'uniform': 每个特征中的所有区间具有相同的宽度。
        - 'quantile': 每个特征中的所有区间具有相同数量的点。
        - 'kmeans': 每个区间中的值具有相同的最近 1D k-means 簇中心。

        有关不同策略的示例，请参见：
        :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_strategies.py`。

    dtype : {np.float32, np.float64}, default=None
        输出的所需数据类型。如果为 None，则输出类型与输入类型一致。仅支持 np.float32 和 np.float64。

        .. versionadded:: 0.24
    # subsample 参数控制用于拟合模型的最大样本数，以提高计算效率。
    # `subsample=None` 表示在计算确定分箱阈值的分位数时使用所有训练样本。
    # 由于分位数计算依赖于对 `X` 的每一列进行排序，而排序的时间复杂度为 `n log(n)`，
    # 建议在样本数非常大的数据集上使用子采样。

    # .. versionchanged:: 1.3
    #     当 `strategy="quantile"` 时，subsample 的默认值从 `None` 更改为 `200_000`。

    # .. versionchanged:: 1.5
    #     当 `strategy="uniform"` 或 `strategy="kmeans"` 时，subsample 的默认值从 `None` 更改为 `200_000`。

    # random_state 参数确定用于子采样的随机数生成，可通过传递一个整数实现跨多次函数调用的可重现结果。
    # 更多详细信息请参见 `subsample` 参数。
    # 参见 :term:`Glossary <random_state>`。

    # .. versionadded:: 1.1

Attributes
----------
bin_edges_ : ndarray of ndarray of shape (n_features,)
    每个特征的分箱边界。包含不同形状的数组 ``(n_bins_, )``。
    忽略的特征将具有空数组。

n_bins_ : ndarray of shape (n_features,), dtype=np.int64
    每个特征的分箱数量。宽度太小的分箱（即 <= 1e-8）将被移除，并显示警告信息。

n_features_in_ : int
    在 :term:`fit` 过程中观察到的特征数。

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    在 :term:`fit` 过程中观察到的特征名称。仅在 `X` 具有全部为字符串的特征名称时定义。

    .. versionadded:: 1.0

See Also
--------
Binarizer : 用于基于参数 `threshold` 将值分箱为 ``0`` 或 ``1`` 的类。

Notes
-----

有关在不同数据集上可视化离散化效果，请参阅
:ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_classification.py`。

关于离散化对线性模型影响的详细信息，请参阅：
:ref:`sphx_glr_auto_examples_preprocessing_plot_discretization.py`。

在特征 `i` 的分箱边界中，仅在 `inverse_transform` 时使用第一个和最后一个值。
在 transform 过程中，分箱边界扩展为::

  np.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])

如果您只想预处理部分特征，可以将 ``KBinsDiscretizer`` 与 :class:`~sklearn.compose.ColumnTransformer` 结合使用。

当 `KBinsDiscretizer` 可能生成常数特征（例如当 `encode='onehot'` 且某些分箱不包含任何数据时）时，这些特征可以使用特征选择算法（例如 :class:`~sklearn.feature_selection.VarianceThreshold`）来移除。
    # 参数约束字典，指定了每个参数的类型和取值范围或选项
    _parameter_constraints: dict = {
        "n_bins": [Interval(Integral, 2, None, closed="left"), "array-like"],
        "encode": [StrOptions({"onehot", "onehot-dense", "ordinal"})],
        "strategy": [StrOptions({"uniform", "quantile", "kmeans"})],
        "dtype": [Options(type, {np.float64, np.float32}), None],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_bins=5,
        *,
        encode="onehot",
        strategy="quantile",
        dtype=None,
        subsample=200_000,
        random_state=None,
    ):
        # 初始化方法，设置离散化的参数
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature."""
        # 验证并返回每个特征的分箱数 n_bins_
        orig_bins = self.n_bins
        if isinstance(orig_bins, Integral):
            # 如果 n_bins 是整数，则对所有特征返回相同的分箱数数组
            return np.full(n_features, orig_bins, dtype=int)

        # 将原始的 n_bins 转换为整数数组
        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

        # 检查 n_bins 的维度和形状是否合法
        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array of shape (n_features,).")

        # 检查每个分箱数是否符合要求
        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            # 如果存在不符合要求的分箱数，则抛出异常
            raise ValueError(
                "{} received an invalid number "
                "of bins at indices {}. Number of bins "
                "must be at least 2, and must be an int.".format(
                    KBinsDiscretizer.__name__, indices
                )
            )
        # 返回验证后的分箱数数组
        return n_bins
    # 将数据离散化处理。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        待离散化处理的数据。

    Returns
    -------
    Xt : {ndarray, sparse matrix}, dtype={np.float32, np.float64}
        处于分箱空间中的数据。如果 `self.encode='onehot'` 则为稀疏矩阵，否则为 ndarray。
    """
    # 检查模型是否已经拟合
    check_is_fitted(self)

    # 检查输入数据和属性的数据类型
    dtype = (np.float64, np.float32) if self.dtype is None else self.dtype
    Xt = self._validate_data(X, copy=True, dtype=dtype, reset=False)

    # 获取分箱的边界
    bin_edges = self.bin_edges_

    # 对每个特征进行处理
    for jj in range(Xt.shape[1]):
        # 使用二分查找将数据映射到分箱的右侧边界
        Xt[:, jj] = np.searchsorted(bin_edges[jj][1:-1], Xt[:, jj], side="right")

    # 如果采用序数编码，直接返回处理后的数据
    if self.encode == "ordinal":
        return Xt

    # 初始化一个变量来保存初始的 dtype
    dtype_init = None

    # 如果使用 onehot 编码，调整编码器的 dtype
    if "onehot" in self.encode:
        dtype_init = self._encoder.dtype
        self._encoder.dtype = Xt.dtype

    try:
        # 使用编码器对数据进行转换
        Xt_enc = self._encoder.transform(Xt)
    finally:
        # 恢复初始的 dtype 以避免修改 self
        self._encoder.dtype = dtype_init

    # 返回转换后的数据
    return Xt_enc

def inverse_transform(self, X=None, *, Xt=None):
    """
    将离散化后的数据转换回原始特征空间。

    注意，由于离散化的舍入，此函数不会完全恢复原始数据。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        在分箱空间中的转换后的数据。

    Xt : array-like of shape (n_samples, n_features)
        在分箱空间中的转换后的数据。

        .. deprecated:: 1.5
            `Xt` 在 1.5 版本中已经弃用，并将在 1.7 版本中移除。请使用 `X` 替代。

    Returns
    -------
    Xinv : ndarray, dtype={np.float32, np.float64}
        原始特征空间中的数据。
    """
    # 使用 _deprecate_Xt_in_inverse_transform 函数将 Xt 转换为 X
    X = _deprecate_Xt_in_inverse_transform(X, Xt)

    # 检查模型是否已经拟合
    check_is_fitted(self)

    # 如果使用 onehot 编码，逆转换数据
    if "onehot" in self.encode:
        X = self._encoder.inverse_transform(X)

    # 检查并复制输入数据，确保数据类型正确
    Xinv = check_array(X, copy=True, dtype=(np.float64, np.float32))

    # 检查特征的数量是否正确
    n_features = self.n_bins_.shape[0]
    if Xinv.shape[1] != n_features:
        raise ValueError(
            "特征数量不正确。期望 {}, 收到 {}.".format(
                n_features, Xinv.shape[1]
            )
        )

    # 对每个特征进行处理，根据分箱边界计算反转后的数据
    for jj in range(n_features):
        bin_edges = self.bin_edges_[jj]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        Xinv[:, jj] = bin_centers[(Xinv[:, jj]).astype(np.int64)]

    # 返回反转后的数据
    return Xinv
    def get_feature_names_out(self, input_features=None):
        """Get output feature names.
        
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
        # 检查模型是否已拟合，确保已定义 n_features_in_
        check_is_fitted(self, "n_features_in_")
        # 根据 input_features 和 feature_names_in_ 进行合法性检查和处理
        input_features = _check_feature_names_in(self, input_features)
        
        # 如果模型有 _encoder 属性，调用 _encoder 的 get_feature_names_out 方法
        if hasattr(self, "_encoder"):
            return self._encoder.get_feature_names_out(input_features)
        
        # 如果没有 _encoder 属性，假设使用序数编码，直接返回 input_features
        # 这里可能会涉及到具体的序数编码逻辑
        
        # ordinal encoding
        return input_features
```