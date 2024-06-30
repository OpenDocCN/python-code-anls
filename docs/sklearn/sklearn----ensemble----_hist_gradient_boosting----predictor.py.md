# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\predictor.py`

```
"""
This module contains the TreePredictor class which is used for prediction.
"""

# Author: Nicolas Hug

import numpy as np

from ._predictor import (
    _compute_partial_dependence,
    _predict_from_binned_data,
    _predict_from_raw_data,
)
from .common import PREDICTOR_RECORD_DTYPE, Y_DTYPE


class TreePredictor:
    """Tree class used for predictions.

    Parameters
    ----------
    nodes : ndarray of PREDICTOR_RECORD_DTYPE
        The nodes of the tree.
    binned_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for binned categories used in predict_binned when a
        split is categorical.
    raw_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for raw categories used in predict when a split is
        categorical.
    """

    def __init__(self, nodes, binned_left_cat_bitsets, raw_left_cat_bitsets):
        self.nodes = nodes
        self.binned_left_cat_bitsets = binned_left_cat_bitsets
        self.raw_left_cat_bitsets = raw_left_cat_bitsets

    def get_n_leaf_nodes(self):
        """Return number of leaves."""
        return int(self.nodes["is_leaf"].sum())

    def get_max_depth(self):
        """Return maximum depth among all leaves."""
        return int(self.nodes["depth"].max())

    def predict(self, X, known_cat_bitsets, f_idx_map, n_threads):
        """Predict raw values for non-binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.

        known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.

        f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.

        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        # Initialize an empty array to store the predicted values
        out = np.empty(X.shape[0], dtype=Y_DTYPE)

        # Call the internal function _predict_from_raw_data to perform prediction
        _predict_from_raw_data(
            self.nodes,                     # Tree nodes
            X,                              # Input samples
            self.raw_left_cat_bitsets,      # Bitsets for raw categories
            known_cat_bitsets,              # Bitsets for known categories
            f_idx_map,                      # Feature index mapping
            n_threads,                      # Number of threads to use
            out                             # Output array for predictions
        )
        
        # Return the array of predicted values
        return out
    # 使用该方法对经过分箱处理的数据进行原始值的预测。

    # Parameters 参数说明：
    # X : ndarray, shape (n_samples, n_features)
    #     输入样本数据。
    # missing_values_bin_idx : uint8
    #     用于表示缺失值的分箱索引。这个索引对应于最后一个分箱，总是等于 max_bins（传递给 GBDT 类的值），或等同于 n_bins - 1。
    # n_threads : int
    #     要使用的 OpenMP 线程数。

    # Returns 返回值：
    # y : ndarray, shape (n_samples,)
    #     原始预测值数组。
    def predict_binned(self, X, missing_values_bin_idx, n_threads):
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_binned_data(
            self.nodes,
            X,
            self.binned_left_cat_bitsets,
            missing_values_bin_idx,
            n_threads,
            out,
        )
        return out

    # 使用快速方法计算偏依赖值。

    # Parameters 参数说明：
    # grid : ndarray, shape (n_samples, n_target_features)
    #     应该评估偏依赖性的网格点。
    # target_features : ndarray, shape (n_target_features)
    #     要评估偏依赖性的目标特征集合。
    # out : ndarray, shape (n_samples)
    #     每个网格点上偏依赖函数的值。
    def compute_partial_dependence(self, grid, target_features, out):
        _compute_partial_dependence(self.nodes, grid, target_features, out)

    # 重新定义 __setstate__ 方法以支持对象状态的反序列化。

    # Parameters 参数说明：
    # state : dict
    #     包含对象状态的字典。

    # Notes 注意事项：
    # feature_idx 的数据类型为 np.intp，这是依赖平台的。在此处，我们确保在不同位数系统上的保存和加载工作正常无误。
    # 例如，在 64 位 Python 运行时，np.intp = np.int64，而在 32 位上 np.intp = np.int32。

    # TODO: 考虑总是使用平台无关的数据类型来保存已拟合的估计器属性。对于这个特定的估计器，可以将 PREDICTOR_RECORD_DTYPE 的 intp 字段替换为 int32 字段。
    # 理想情况下，应该在整个 scikit-learn 中一致地完成这一操作，并伴有共同的测试。
    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

        if self.nodes.dtype != PREDICTOR_RECORD_DTYPE:
            # 将 self.nodes 转换为与 PREDICTOR_RECORD_DTYPE 相同类型，确保类型匹配
            self.nodes = self.nodes.astype(PREDICTOR_RECORD_DTYPE, casting="same_kind")
```