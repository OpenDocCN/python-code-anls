# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\_base.py`

```
"""Generic feature selection mixin"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings  # 导入警告模块，用于发出警告信息
from abc import ABCMeta, abstractmethod  # 导入抽象基类相关模块
from operator import attrgetter  # 导入属性获取器模块

import numpy as np  # 导入NumPy库
from scipy.sparse import csc_matrix, issparse  # 导入稀疏矩阵相关模块

from ..base import TransformerMixin  # 导入变换器Mixin类
from ..utils import _safe_indexing, check_array, safe_sqr  # 导入工具函数
from ..utils._set_output import _get_output_config  # 导入设置输出配置的函数
from ..utils._tags import _safe_tags  # 导入安全标签函数
from ..utils.validation import _check_feature_names_in, _is_pandas_df, check_is_fitted  # 导入验证函数


class SelectorMixin(TransformerMixin, metaclass=ABCMeta):
    """
    Transformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.base import BaseEstimator
    >>> from sklearn.feature_selection import SelectorMixin
    >>> class FeatureSelector(SelectorMixin, BaseEstimator):
    ...    def fit(self, X, y=None):
    ...        self.n_features_in_ = X.shape[1]
    ...        return self
    ...    def _get_support_mask(self):
    ...        mask = np.zeros(self.n_features_in_, dtype=bool)
    ...        mask[:2] = True  # select the first two features
    ...        return mask
    >>> X, y = load_iris(return_X_y=True)
    >>> FeatureSelector().fit_transform(X, y).shape
    (150, 2)
    """

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask()  # 调用抽象方法获取特征选择的布尔掩码
        return mask if not indices else np.where(mask)[0]

    @abstractmethod
    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
    # 使用 Transformer 类的方法，对输入数据 X 进行特征选择处理
    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        # 根据输出配置确定是否保留输入数据 X，当 X 是 DataFrame 且输出是 pandas 时
        output_config_dense = _get_output_config("transform", estimator=self)["dense"]
        preserve_X = output_config_dense != "default" and _is_pandas_df(X)

        # 注意：我们使用 _safe_tags 而不是 _get_tags，因为这是一个公共的 Mixin。
        # 验证输入数据 X，确保其符合要求
        X = self._validate_data(
            X,
            dtype=None,
            accept_sparse="csr",
            force_all_finite=not _safe_tags(self, key="allow_nan"),
            cast_to_ndarray=not preserve_X,
            reset=False,
        )
        # 调用内部方法 _transform 进行实际的特征选择操作
        return self._transform(X)

    # 内部方法，实现将输入数据 X 降维到选定的特征集合
    def _transform(self, X):
        """Reduce X to the selected features."""
        # 获取特征选择的掩码
        mask = self.get_support()
        # 如果没有选择任何特征，发出警告并返回空数据
        if not mask.any():
            warnings.warn(
                (
                    "No features were selected: either the data is"
                    " too noisy or the selection test too strict."
                ),
                UserWarning,
            )
            # 如果 X 是一个 DataFrame，返回空的 DataFrame
            if hasattr(X, "iloc"):
                return X.iloc[:, :0]
            # 否则返回空的 NumPy 数组
            return np.empty(0, dtype=X.dtype).reshape((X.shape[0], 0))
        # 使用掩码对 X 进行安全索引，仅保留选定的特征
        return _safe_indexing(X, mask, axis=1)
    def inverse_transform(self, X):
        """Reverse the transformation operation.

        Parameters
        ----------
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_original_features]
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        # 检查输入的稀疏性，如果是稀疏矩阵，则转换为压缩稀疏列格式
        if issparse(X):
            X = X.tocsc()
            # 插入额外的条目到indptr中：
            # 例如，如果transform方法改变了indptr从[0 2 6 7]到[0 2 3]，
            # col_nonzeros将会是[2 0 1]，所以indptr变成了[0 2 2 3]
            it = self.inverse_transform(np.diff(X.indptr).reshape(1, -1))
            col_nonzeros = it.ravel()
            indptr = np.concatenate([[0], np.cumsum(col_nonzeros)])
            # 创建压缩稀疏列矩阵对象Xt
            Xt = csc_matrix(
                (X.data, X.indices, indptr),
                shape=(X.shape[0], len(indptr) - 1),
                dtype=X.dtype,
            )
            return Xt

        # 获取支持的特征索引
        support = self.get_support()
        # 检查并转换输入数据X为数组
        X = check_array(X, dtype=None)
        # 如果支持的特征数不等于输入X的特征数，抛出数值错误
        if support.sum() != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        # 如果X是一维数组，则转换为二维数组
        if X.ndim == 1:
            X = X[None, :]
        # 创建一个全零矩阵Xt，将输入X中支持的特征值复制到对应位置
        Xt = np.zeros((X.shape[0], support.size), dtype=X.dtype)
        Xt[:, support] = X
        return Xt

    def get_feature_names_out(self, input_features=None):
        """Mask feature names according to selected features.

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
        # 确保模型已经拟合
        check_is_fitted(self)
        # 检查和获取输入的特征名列表
        input_features = _check_feature_names_in(self, input_features)
        # 返回支持的特征名列表
        return input_features[self.get_support()]
# 定义函数 _get_feature_importances，用于从估算器中提取和聚合特征重要性，还可以选择应用变换。
def _get_feature_importances(estimator, getter, transform_func=None, norm_order=1):
    """
    Retrieve and aggregate (ndim > 1) the feature importances
    from an estimator. Also optionally applies transformation.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator from which we want to get the feature
        importances.

    getter : "auto", str or callable
        An attribute or a callable to get the feature importance. If `"auto"`,
        `estimator` is expected to expose `coef_` or `feature_importances`.

    transform_func : {"norm", "square"}, default=None
        The transform to apply to the feature importances. By default (`None`)
        no transformation is applied.

    norm_order : int, default=1
        The norm order to apply when `transform_func="norm"`. Only applied
        when `importances.ndim > 1`.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        The features importances, optionally transformed.
    """
    # 如果 getter 是字符串
    if isinstance(getter, str):
        # 如果 getter 是 "auto"
        if getter == "auto":
            # 如果估算器具有 coef_ 属性，则使用 attrgetter("coef_")
            if hasattr(estimator, "coef_"):
                getter = attrgetter("coef_")
            # 如果估算器具有 feature_importances_ 属性，则使用 attrgetter("feature_importances_")
            elif hasattr(estimator, "feature_importances_"):
                getter = attrgetter("feature_importances_")
            # 否则抛出 ValueError 异常
            else:
                raise ValueError(
                    "when `importance_getter=='auto'`, the underlying "
                    f"estimator {estimator.__class__.__name__} should have "
                    "`coef_` or `feature_importances_` attribute. Either "
                    "pass a fitted estimator to feature selector or call fit "
                    "before calling transform."
                )
        else:
            # 否则将 getter 转换为 attrgetter(getter)
            getter = attrgetter(getter)
    # 如果 getter 不是 callable，抛出 ValueError 异常
    elif not callable(getter):
        raise ValueError("`importance_getter` has to be a string or `callable`")

    # 调用 getter 获取特征重要性
    importances = getter(estimator)

    # 如果 transform_func 为 None，直接返回 importances
    if transform_func is None:
        return importances
    # 如果 transform_func 是 "norm"
    elif transform_func == "norm":
        # 如果 importances 的维度为 1，则对其取绝对值
        if importances.ndim == 1:
            importances = np.abs(importances)
        # 否则，使用指定的 norm_order 对 importances 应用范数
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    # 如果 transform_func 是 "square"
    elif transform_func == "square":
        # 如果 importances 的维度为 1，则对其进行平方
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        # 否则，对 importances 进行平方后按列求和
        else:
            importances = safe_sqr(importances).sum(axis=0)
    # 否则，抛出 ValueError 异常，提示 transform_func 的值无效
    else:
        raise ValueError(
            "Valid values for `transform_func` are "
            + "None, 'norm' and 'square'. Those two "
            + "transformation are only supported now"
        )

    # 返回经过处理后的 importances
    return importances
```