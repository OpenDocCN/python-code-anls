# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_variance_threshold.py`

```
# 导入必要的模块和类
from numbers import Real  # 导入 Real 类型，用于参数验证
import numpy as np  # 导入 NumPy 库
from ..base import BaseEstimator, _fit_context  # 导入基础估计器和拟合上下文
from ..utils._param_validation import Interval  # 导入参数验证工具中的 Interval 类
from ..utils.sparsefuncs import mean_variance_axis, min_max_axis  # 导入稀疏函数中的均值方差和最小最大函数
from ..utils.validation import check_is_fitted  # 导入验证工具中的检查是否拟合函数
from ._base import SelectorMixin  # 从当前目录中导入 SelectorMixin 类


class VarianceThreshold(SelectorMixin, BaseEstimator):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SelectFromModel: Meta-transformer for selecting features based on
        importance weights.
    SelectPercentile : Select features according to a percentile of the highest
        scores.
    SequentialFeatureSelector : Transformer that performs Sequential Feature
        Selection.

    Notes
    -----
    Allows NaN in the input.
    Raises ValueError if no feature in X meets the variance threshold.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> from sklearn.feature_selection import VarianceThreshold
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    _parameter_constraints: dict = {
        "threshold": [Interval(Real, 0, None, closed="left")]
    }

    def __init__(self, threshold=0.0):
        self.threshold = threshold  # 初始化阈值参数

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input data X and convert to float64 if needed, allowing NaN values
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),  # Specify accepted sparse formats
            dtype=np.float64,  # Convert to float64
            force_all_finite="allow-nan",  # Allow NaN values
        )

        if hasattr(X, "toarray"):  # Check if X is a sparse matrix
            # Compute mean and variance along axis 0 for sparse matrix
            _, self.variances_ = mean_variance_axis(X, axis=0)
            if self.threshold == 0:
                # Compute min and max along axis 0 for sparse matrix
                mins, maxes = min_max_axis(X, axis=0)
                peak_to_peaks = maxes - mins  # Compute peak-to-peak differences
        else:
            # Compute variance along axis 0 for dense matrix
            self.variances_ = np.nanvar(X, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(X, axis=0)  # Compute peak-to-peak differences

        if self.threshold == 0:
            # Use peak-to-peak differences to handle constant features and avoid precision issues
            compare_arr = np.array([self.variances_, peak_to_peaks])
            self.variances_ = np.nanmin(compare_arr, axis=0)  # Choose minimum values

        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            # Raise error if no feature in X meets the variance threshold
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self  # Return fitted instance of the class

    def _get_support_mask(self):
        check_is_fitted(self)  # Ensure that the estimator has been fitted

        return self.variances_ > self.threshold  # Return mask of features based on variance threshold

    def _more_tags(self):
        return {"allow_nan": True}  # Additional tags indicating NaN values are allowed
```