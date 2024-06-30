# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_nearest_centroid.py`

```
"""
Nearest Centroid Classification
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 numbers 模块中导入 Real 类型，用于验证参数是否为实数
from numbers import Real

# 导入必要的库和模块
import numpy as np
from scipy import sparse as sp

# 从 scikit-learn 中导入基础类和函数
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..preprocessing import LabelEncoder
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.sparsefuncs import csc_median_axis_0
from ..utils.validation import check_is_fitted

# 定义最近质心分类器类 NearestCentroid，继承自 ClassifierMixin 和 BaseEstimator
class NearestCentroid(ClassifierMixin, BaseEstimator):
    """Nearest centroid classifier.

    Each class is represented by its centroid, with test samples classified to
    the class with the nearest centroid.

    Read more in the :ref:`User Guide <nearest_centroid_classifier>`.

    Parameters
    ----------
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Metric to use for distance computation.

        If `metric="euclidean"`, the centroid for the samples corresponding to each
        class is the arithmetic mean, which minimizes the sum of squared L1 distances.
        If `metric="manhattan"`, the centroid is the feature-wise median, which
        minimizes the sum of L1 distances.

        .. versionchanged:: 1.5
            All metrics but `"euclidean"` and `"manhattan"` were deprecated and
            now raise an error.

        .. versionchanged:: 0.19
            `metric='precomputed'` was deprecated and now raises an error

    shrink_threshold : float, default=None
        Threshold for shrinking centroids to remove features.

    Attributes
    ----------
    centroids_ : array-like of shape (n_classes, n_features)
        Centroid of each class.

    classes_ : array of shape (n_classes,)
        The unique classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KNeighborsClassifier : Nearest neighbors classifier.

    Notes
    -----
    When used for text classification with tf-idf vectors, this classifier is
    also known as the Rocchio classifier.

    References
    ----------
    Tibshirani, R., Hastie, T., Narasimhan, B., & Chu, G. (2002). Diagnosis of
    multiple cancer types by shrunken centroids of gene expression. Proceedings
    of the National Academy of Sciences of the United States of America,
    99(10), 6567-6572. The National Academy of Sciences.

    Examples
    --------
    >>> from sklearn.neighbors import NearestCentroid
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = NearestCentroid()
    >>> clf.fit(X, y)
    NearestCentroid()


    # 使用训练数据 X 和标签 y 来训练分类器 clf
    >>> clf.fit(X, y)
    # 输出 NearestCentroid() 表示使用最近质心分类器
    NearestCentroid()


    >>> print(clf.predict([[-0.8, -1]]))
    [1]


    # 打印预测结果，对输入的测试向量[[-0.8, -1]]进行预测，返回预测的类别[1]
    >>> print(clf.predict([[-0.8, -1]]))
    [1]


    For a more detailed example see:
    :ref:`sphx_glr_auto_examples_neighbors_plot_nearest_centroid.py`
    """


    # 提供更详细的示例参考链接，指向 sphx_glr_auto_examples_neighbors_plot_nearest_centroid.py
    # """


    _parameter_constraints: dict = {
        "metric": [StrOptions({"manhattan", "euclidean"})],
        "shrink_threshold": [Interval(Real, 0, None, closed="neither"), None],
    }


    # 定义参数约束字典 _parameter_constraints，包含两个键值对
    _parameter_constraints: dict = {
        # metric 键对应的值为包含 {"manhattan", "euclidean"} 的字符串选项
        "metric": [StrOptions({"manhattan", "euclidean"})],
        # shrink_threshold 键对应的值为一个包含实数类型，取值大于0的区间，不包括0，无上界的间隔，以及 None
        "shrink_threshold": [Interval(Real, 0, None, closed="neither"), None],
    }


    def __init__(self, metric="euclidean", *, shrink_threshold=None):
        # 初始化方法，设定分类器的 metric 和 shrink_threshold 参数
        self.metric = metric
        self.shrink_threshold = shrink_threshold


    @_fit_context(prefer_skip_nested_validation=True)
    # 使用装饰器 _fit_context 标记下面的 predict 方法，prefer_skip_nested_validation 参数设为 True
    def predict(self, X):
        """Perform classification on an array of test vectors `X`.
        
        对数组 `X` 进行分类预测。
        
        The predicted class `C` for each sample in `X` is returned.
        
        返回 `X` 中每个样本的预测类别 `C`。
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.
            
            测试样本，形状为 (n_samples, n_features) 的数组或稀疏矩阵。

        Returns
        -------
        C : ndarray of shape (n_samples,)
            The predicted classes.
            
            预测的类别数组，形状为 (n_samples,) 的 ndarray。
        """
        # 检查分类器是否已经拟合过数据
        check_is_fitted(self)
        
        # 验证输入数据 X，并按需接受稀疏矩阵 csr 格式，不重置数据
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        
        # 返回每个样本的预测类别，使用最近质心法寻找最近的质心索引
        return self.classes_[
            pairwise_distances_argmin(X, self.centroids_, metric=self.metric)
        ]


        # 执行分类预测操作，输入数据 X
        @_fit_context(prefer_skip_nested_validation=True)
        def predict(self, X):
            # 执行分类预测操作
            """Perform classification on an array of test vectors `X`.
            
            对数组 `X` 进行分类预测。
            
            The predicted class `C` for each sample in `X` is returned.
            
            返回 `X` 中每个样本的预测类别 `C`。
            
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Test samples.
                
                测试样本，形状为 (n_samples, n_features) 的数组或稀疏矩阵。
    
            Returns
            -------
            C : ndarray of shape (n_samples,)
                The predicted classes.
                
                预测的类别数组，形状为 (n_samples,) 的 ndarray。
            """
            # 检查分类器是否已经拟合
            check_is_fitted(self)
    
            # 验证输入数据 X，接受稀疏矩阵 csr 格式，不重置数据
            X = self._validate_data(X, accept_sparse="csr", reset=False)
    
            # 返回每个样本的预测类别，使用最近质心法寻找最近的质心索引
            return self.classes_[
                pairwise_distances_argmin(X, self.centroids_, metric=self.metric)
            ]
```