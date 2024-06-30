# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\binning.py`

```
"""
This module contains the BinMapper class.

BinMapper is used for mapping a real-valued dataset into integer-valued bins.
Bin thresholds are computed with the quantiles so that each bin contains
approximately the same number of samples.
"""

# Author: Nicolas Hug
import concurrent.futures  # 导入并发处理模块

import numpy as np  # 导入NumPy库

from ...base import BaseEstimator, TransformerMixin  # 导入基础估计器和转换器混合类
from ...utils import check_array, check_random_state  # 导入数组检查和随机状态检查函数
from ...utils._openmp_helpers import _openmp_effective_n_threads  # 导入OpenMP相关函数
from ...utils.fixes import percentile  # 导入修复函数，包括百分位数计算修复
from ...utils.validation import check_is_fitted  # 导入验证模型是否拟合的函数
from ._binning import _map_to_bins  # 导入映射到箱体的函数
from ._bitset import set_bitset_memoryview  # 导入设置位集合内存视图的函数
from .common import ALMOST_INF, X_BINNED_DTYPE, X_BITSET_INNER_DTYPE, X_DTYPE  # 导入常用常量和数据类型

def _find_binning_thresholds(col_data, max_bins):
    """Extract quantiles from a continuous feature.

    Missing values are ignored for finding the thresholds.

    Parameters
    ----------
    col_data : array-like, shape (n_samples,)
        The continuous feature to bin.
    max_bins: int
        The maximum number of bins to use for non-missing values. If for a
        given feature the number of unique values is less than ``max_bins``,
        then those unique values will be used to compute the bin thresholds,
        instead of the quantiles

    Return
    ------
    binning_thresholds : ndarray of shape(min(max_bins, n_unique_values) - 1,)
        The increasing numeric values that can be used to separate the bins.
        A given value x will be mapped into bin value i iff
        bining_thresholds[i - 1] < x <= binning_thresholds[i]
    """
    # ignore missing values when computing bin thresholds
    missing_mask = np.isnan(col_data)  # 创建一个布尔掩码，标记缺失值
    if missing_mask.any():  # 如果有缺失值
        col_data = col_data[~missing_mask]  # 从数据中删除缺失值

    # 数据将在 np.unique 和 percentile 中排序，所以这里进行排序，同时返回一个连续数组
    col_data = np.sort(col_data)

    distinct_values = np.unique(col_data).astype(X_DTYPE)  # 获取唯一值并转换为指定数据类型
    if len(distinct_values) <= max_bins:  # 如果唯一值的数量小于等于最大箱数
        midpoints = distinct_values[:-1] + distinct_values[1:]  # 计算每对相邻值的中点
        midpoints *= 0.5  # 每个中点乘以0.5
    else:
        # 否则，使用百分位数计算中点的近似值
        percentiles = np.linspace(0, 100, num=max_bins + 1)
        percentiles = percentiles[1:-1]
        midpoints = percentile(col_data, percentiles, method="midpoint").astype(X_DTYPE)
        assert midpoints.shape[0] == max_bins - 1

    # 避免出现+inf阈值：+inf阈值仅在“拆分NaN”情况下允许
    np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
    return midpoints  # 返回计算出的分箱阈值数组


class _BinMapper(TransformerMixin, BaseEstimator):
    """Transformer that maps a dataset into integer-valued bins.

    This class implements the transformation of a dataset into integer-valued
    bins using quantile-based binning.

    Attributes
    ----------
    binning_thresholds_ : ndarray of shape (n_features,)
        The binning thresholds computed for each feature.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    >>> est.fit(X)
    KBinsDiscretizer(encode='ordinal', n_bins=3, strategy='quantile')
    >>> est.transform(X)
    array([[0., 0.],
           [1., 1.],
           [2., 2.],
           [2., 2.]])

    Notes
    -----
    The transformation is applied feature-wise independently.

    """
    # 对于连续特征，以特征为单位使用分位数来创建分箱，确保每个分箱包含大致相同数量的样本。对于大型数据集，
    # 会在数据的子集上计算分位数以加快分箱速度，但分位数应保持稳定。

    # 对于分类特征，假设原始分类值在 [0, 254] 范围内（这里不进行验证），每个分类对应一个分箱。
    # 所有分类值必须在初始化时已知：transform() 不知道如何对未知的分类值进行分箱。
    # 注意，transform() 仅在早停的情况下用于非训练数据。

    # 具有少量值的特征可能会被分箱为少于 ``n_bins`` 个分箱。最后一个分箱（索引为 ``n_bins - 1``）始终保留用于缺失值。

    Parameters
    ----------
    n_bins : int, default=256
        要使用的最大分箱数（包括缺失值的分箱）。应在 [3, 256] 范围内。非缺失值将分箱为 ``max_bins = n_bins - 1`` 个分箱。
        最后一个分箱始终保留用于缺失值。对于给定特征，如果唯一值的数量小于 ``max_bins``，那么将使用这些唯一值来计算分箱阈值，
        而不是分位数。关于是否为分类特征的详细信息，请参见 ``is_categorical`` 的文档字符串。

    subsample : int or None, default=2e5
        如果 ``n_samples > subsample``，则将随机选择 ``sub_samples`` 个样本来计算分位数。如果为 ``None``，则使用所有数据。

    is_categorical : ndarray of bool of shape (n_features,), default=None
        指示分类特征的布尔型数组。默认情况下，所有特征都被视为连续的。

    known_categories : list of {ndarray, None} of shape (n_features,), default=None
        每个分类特征的列表，其中数组指示唯一的分类值集合。这些应该是所有数据的可能值，而不仅仅是训练数据。
        对于连续特征，相应的条目应为 None。

    random_state : int, RandomState instance or None, default=None
        伪随机数生成器，用于控制随机子采样。传递一个整数以确保多次函数调用时输出可复现。
        参见 :term:`Glossary <random_state>`。

    n_threads : int, default=None
        要使用的 OpenMP 线程数。将调用 `_openmp_effective_n_threads` 来确定实际使用的线程数，
        它考虑了 cgroups CPU 配额。有关详细信息，请参阅 `_openmp_effective_n_threads` 的文档字符串。

    Attributes
    ----------
    # 用于存储每个特征的阈值数组列表，每个数组指示如何将特征映射到分箱特征。其语义和大小取决于特征的性质：
    # - 对于实值特征，数组对应于实值分箱阈值（每个分箱的上界）。有 `max_bins - 1` 个阈值，其中 `max_bins = n_bins - 1` 是非缺失值使用的分箱数。
    # - 对于分类特征，数组是从分箱类别值到原始类别值的映射。数组的大小等于 `min(max_bins, category_cardinality)`，其中我们忽略基数中的缺失值。
    bin_thresholds_ : list of ndarray
    
    # 每个特征实际使用的非缺失值分箱数。对于具有大量唯一值的特征，这等于 `n_bins - 1`。
    n_bins_non_missing_ : ndarray, dtype=np.uint32
    
    # 分类特征的指示器。形状为 (n_features,) 的 ndarray，数据类型为 np.uint8。
    is_categorical_ : ndarray of shape (n_features,), dtype=np.uint8
    
    # 缺失值映射到的分箱索引。这是所有特征中常量的索引。对应于最后一个分箱，始终等于 `n_bins - 1`。
    # 注意，如果对于给定特征，`n_bins_non_missing_` 小于 `n_bins - 1`，则存在空的（未使用的）分箱。
    missing_values_bin_idx_ : np.uint8
    
    # 构造函数，初始化分箱器对象
    def __init__(
        self,
        n_bins=256,                    # 分箱数，默认为 256
        subsample=int(2e5),            # 子样本大小，默认为 200,000
        is_categorical=None,           # 分类特征指示器，默认为 None
        known_categories=None,         # 已知分类，默认为 None
        random_state=None,             # 随机数种子，默认为 None
        n_threads=None,                # 线程数，默认为 None
    ):
        self.n_bins = n_bins           # 将传入的 n_bins 分配给对象属性 n_bins
        self.subsample = subsample     # 将传入的 subsample 分配给对象属性 subsample
        self.is_categorical = is_categorical  # 将传入的 is_categorical 分配给对象属性 is_categorical
        self.known_categories = known_categories  # 将传入的 known_categories 分配给对象属性 known_categories
        self.random_state = random_state  # 将传入的 random_state 分配给对象属性 random_state
        self.n_threads = n_threads     # 将传入的 n_threads 分配给对象属性 n_threads
    def transform(self, X):
        """Bin data X.

        Missing values will be mapped to the last bin.

        For categorical features, the mapping will be incorrect for unknown
        categories. Since the BinMapper is given known_categories of the
        entire training data (i.e. before the call to train_test_split() in
        case of early-stopping), this never happens.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to bin.

        Returns
        -------
        X_binned : array-like of shape (n_samples, n_features)
            The binned data (fortran-aligned).
        """
        # 检查输入数据 X 的类型和是否包含无限值
        X = check_array(X, dtype=[X_DTYPE], force_all_finite=False)
        
        # 检查当前对象是否已经拟合过（即已训练过）
        check_is_fitted(self)
        
        # 如果输入数据 X 的列数与模型拟合时的列数不一致，则抛出异常
        if X.shape[1] != self.n_bins_non_missing_.shape[0]:
            raise ValueError(
                "This estimator was fitted with {} features but {} got passed "
                "to transform()".format(self.n_bins_non_missing_.shape[0], X.shape[1])
            )

        # 计算有效的线程数
        n_threads = _openmp_effective_n_threads(self.n_threads)
        
        # 创建一个与输入数据 X 类型和形状相同的零数组，用于存放 bin 后的数据
        binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order="F")
        
        # 调用 C 扩展函数 _map_to_bins 将数据 X 映射到相应的 bin 中
        _map_to_bins(
            X,
            self.bin_thresholds_,
            self.is_categorical_,
            self.missing_values_bin_idx_,
            n_threads,
            binned,
        )
        
        # 返回经过 bin 处理后的数据
        return binned

    def make_known_categories_bitsets(self):
        """Create bitsets of known categories.

        Returns
        -------
        - known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.
        - f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.
        """

        # 获取所有的分类特征的索引
        categorical_features_indices = np.flatnonzero(self.is_categorical_)

        # 获取总特征数和分类特征数
        n_features = self.is_categorical_.size
        n_categorical_features = categorical_features_indices.size

        # 创建一个特征索引映射数组，用于映射原始特征索引到 bitsets 数组中的索引
        f_idx_map = np.zeros(n_features, dtype=np.uint32)
        f_idx_map[categorical_features_indices] = np.arange(
            n_categorical_features, dtype=np.uint32
        )

        # 获取已知类别的阈值数组
        known_categories = self.bin_thresholds_

        # 创建一个存储已知类别 bitsets 的数组
        known_cat_bitsets = np.zeros(
            (n_categorical_features, 8), dtype=X_BITSET_INNER_DTYPE
        )

        # 遍历每个分类特征的索引和其对应的原始特征索引
        # 将已知类别的原始值转换为 bitsets 存储到 known_cat_bitsets 中
        for mapped_f_idx, f_idx in enumerate(categorical_features_indices):
            for raw_cat_val in known_categories[f_idx]:
                set_bitset_memoryview(known_cat_bitsets[mapped_f_idx], raw_cat_val)

        # 返回已知类别的 bitsets 数组和特征索引映射数组
        return known_cat_bitsets, f_idx_map
```