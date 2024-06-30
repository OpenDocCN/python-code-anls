# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_feature_agglomeration.py`

```
"""
Feature agglomeration. Base classes and functions for performing feature
agglomeration.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from scipy.sparse import issparse

from ..base import TransformerMixin
from ..utils import metadata_routing
from ..utils.deprecation import _deprecate_Xt_in_inverse_transform
from ..utils.validation import check_is_fitted

###############################################################################
# Mixin class for feature agglomeration.


class AgglomerationTransform(TransformerMixin):
    """
    A class for feature agglomeration via the transform interface.
    """

    # This prevents ``set_split_inverse_transform`` to be generated for the
    # non-standard ``Xt`` arg on ``inverse_transform``.
    # TODO(1.7): remove when Xt is removed for inverse_transform.
    __metadata_request__inverse_transform = {"Xt": metadata_routing.UNUSED}

    def transform(self, X):
        """
        Transform a new matrix using the built clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            A M by N array of M observations in N dimensions or a length
            M array of M one-dimensional observations.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_clusters) or (n_clusters,)
            The pooled values for each feature cluster.
        """
        # 检查模型是否已经拟合，如果没有拟合则引发异常
        check_is_fitted(self)

        # 验证输入数据并保持数据格式不变
        X = self._validate_data(X, reset=False)
        
        # 根据聚合函数选择不同的数据池化方法
        if self.pooling_func == np.mean and not issparse(X):
            # 如果聚合函数是均值且X不是稀疏矩阵
            # 计算每个簇的样本数
            size = np.bincount(self.labels_)
            n_samples = X.shape[0]
            # 一种快速计算特征组均值的方法
            nX = np.array(
                [np.bincount(self.labels_, X[i, :]) / size for i in range(n_samples)]
            )
        else:
            # 对于其他聚合函数或者稀疏矩阵
            # 对每个标签对应的特征子集应用聚合函数
            nX = [
                self.pooling_func(X[:, self.labels_ == l], axis=1)
                for l in np.unique(self.labels_)
            ]
            # 转换结果为NumPy数组并转置
            nX = np.array(nX).T
        # 返回聚合后的结果
        return nX
    def inverse_transform(self, X=None, *, Xt=None):
        """
        Inverse the transformation and return a vector of size `n_features`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

        Xt : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

            .. deprecated:: 1.5
                `Xt` was deprecated in 1.5 and will be removed in 1.7. Use `X` instead.
        
        Returns
        -------
        X : ndarray of shape (n_samples, n_features) or (n_features,)
            A vector of size `n_samples` with the values of `Xred` assigned to
            each of the cluster of samples.
        """
        # 使用 _deprecate_Xt_in_inverse_transform 函数处理 X 和 Xt 参数的兼容性问题
        X = _deprecate_Xt_in_inverse_transform(X, Xt)
        
        # 检查当前对象是否已经适配（即是否已经进行过拟合）
        check_is_fitted(self)
        
        # 使用 numpy 的 unique 函数获取 self.labels_ 数组的唯一值和反向索引
        unil, inverse = np.unique(self.labels_, return_inverse=True)
        
        # 返回根据反向索引 inverse 重组后的 X 数组
        return X[..., inverse]
```