# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\_base.py`

```
"""Principal Component Analysis Base Classes"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod  # 导入抽象基类元类和抽象方法装饰器

import numpy as np  # 导入NumPy库
from scipy import linalg  # 导入SciPy的线性代数模块

from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin  # 导入基础估计器和特征输出混合类、转换器混合类
from ..utils._array_api import _add_to_diagonal, device, get_namespace  # 导入数组API中的函数：向对角线添加值、设备信息、获取命名空间
from ..utils.validation import check_is_fitted  # 导入验证模块中的检查是否已拟合函数


class _BasePCA(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def get_covariance(self):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array of shape=(n_features, n_features)
            Estimated covariance of data.
        """
        xp, _ = get_namespace(self.components_)  # 获取命名空间信息和设备信息

        components_ = self.components_  # 获取主成分
        exp_var = self.explained_variance_  # 获取解释方差
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])  # 若进行白化，调整主成分的尺度
        exp_var_diff = exp_var - self.noise_variance_  # 计算解释方差和噪声方差的差异
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,  # 条件：解释方差大于噪声方差
            exp_var_diff,  # 符合条件时的结果
            xp.asarray(0.0, device=device(exp_var)),  # 不符合条件时的结果，设备信息由exp_var确定
        )
        cov = (components_.T * exp_var_diff) @ components_  # 计算数据的协方差矩阵
        _add_to_diagonal(cov, self.noise_variance_, xp)  # 向协方差矩阵的对角线添加噪声方差
        return cov  # 返回计算得到的协方差矩阵
    def get_precision(self):
        """
        Compute data precision matrix with the generative model.
        
        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.
        
        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        xp, is_array_api_compliant = get_namespace(self.components_)

        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return xp.eye(n_features) / self.noise_variance_

        if is_array_api_compliant:
            linalg_inv = xp.linalg.inv
        else:
            linalg_inv = linalg.inv

        if self.noise_variance_ == 0.0:
            return linalg_inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        
        # Scale components if whitening is enabled
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        
        # Calculate variance differences and handle zero cases
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            xp.asarray(0.0, device=device(exp_var)),
        )
        
        # Compute initial precision matrix
        precision = components_ @ components_.T / self.noise_variance_
        
        # Update diagonal elements of precision matrix
        _add_to_diagonal(precision, 1.0 / exp_var_diff, xp)
        
        # Further refine precision matrix
        precision = components_.T @ linalg_inv(precision) @ components_
        precision /= -(self.noise_variance_**2)
        
        # Adjust diagonal elements again
        _add_to_diagonal(precision, 1.0 / self.noise_variance_, xp)
        
        return precision

    @abstractmethod
    def fit(self, X, y=None):
        """
        Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        xp, _ = get_namespace(X, self.components_, self.explained_variance_)

        # 检查模型是否已拟合
        check_is_fitted(self)

        # 校验数据格式，确保数据类型和稀疏性符合要求
        X = self._validate_data(
            X, dtype=[xp.float64, xp.float32], accept_sparse=("csr", "csc"), reset=False
        )

        # 执行数据变换
        return self._transform(X, xp=xp, x_is_centered=False)

    def _transform(self, X, xp, x_is_centered=False):
        # 使用主成分对数据进行投影
        X_transformed = X @ self.components_.T

        if not x_is_centered:
            # 如果未中心化，对投影后的数据进行中心化处理
            # 对于密集矩阵，这样做可以避免复制或者改变调用者传递的数据
            # 对于稀疏矩阵，保持稀疏性并避免将 X 包装成线性操作器
            X_transformed -= xp.reshape(self.mean_, (1, -1)) @ self.components_.T

        if self.whiten:
            # 如果启用白化，对投影后的数据进行白化处理
            # 在某些求解器（如 "arpack" 和 "covariance_eigh"）中，当数据秩不足时，
            # 某些成分的方差可能接近零，导致白化时出现非有限结果。为了避免这个问题，
            # 我们将方差剪辑到一个非常小的正数以下。
            scale = xp.sqrt(self.explained_variance_)
            min_scale = xp.finfo(scale.dtype).eps
            scale[scale < min_scale] = min_scale
            X_transformed /= scale

        return X_transformed
    # 将数据反向转换回原始空间。

    # 获取输入数据 X 的命名空间和数据类型
    xp, _ = get_namespace(X)

    # 如果启用了白化（whitening），则计算精确的逆操作，包括反转白化过程
    if self.whiten:
        # 缩放成分：乘以标准差的平方根与主成分矩阵的点乘
        scaled_components = (
            xp.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_
        )
        # 返回原始数据 X_original，经过逆白化处理
        return X @ scaled_components + self.mean_
    else:
        # 返回原始数据 X_original，直接通过主成分矩阵和均值进行转换
        return X @ self.components_ + self.mean_

    # 属性方法：输出转换后的特征数
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]
```