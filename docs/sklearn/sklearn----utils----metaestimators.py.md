# `D:\src\scipysrc\scikit-learn\sklearn\utils\metaestimators.py`

```
"""Utilities for meta-estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的模块和类
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import Any, List

import numpy as np

# 导入基类 BaseEstimator 和工具函数 _safe_indexing
from ..base import BaseEstimator
from ..utils import _safe_indexing
# 导入安全标签函数 _safe_tags
from ..utils._tags import _safe_tags
# 导入条件导入函数 available_if
from ._available_if import available_if

# __all__ 变量定义了这个模块导出的公共接口
__all__ = ["available_if"]


class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for classifiers composed of named estimators."""

    # 定义一个列表 steps，用来保存组成分类器的步骤
    steps: List[Any]

    @abstractmethod
    def __init__(self):
        pass

    # 获取参数的方法，支持深度查询
    def _get_params(self, attr, deep=True):
        # 调用父类的 get_params 方法获取基础参数
        out = super().get_params(deep=deep)
        if not deep:
            return out

        # 获取指定属性（如 steps）中的估算器列表
        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            # 忽略 TypeError 和 ValueError 异常，防止在调用 set_params 时出错
            return out

        # 遍历每个估算器，如果估算器有 get_params 方法，则将其参数加入到 out 中
        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value
        return out

    # 设置参数的方法
    def _set_params(self, attr, **params):
        # 保证参数设置的严格顺序：
        # 1. 设置所有步骤
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. 替换参数中的估算器项
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            with suppress(TypeError):
                item_names, _ = zip(*items)
                # 遍历 params 中的键，如果键是估算器的名称，则替换对应的估算器
                for name in list(params.keys()):
                    if "__" not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))

        # 3. 设置步骤参数和其他初始化参数
        super().set_params(**params)
        return self

    # 替换估算器的方法
    def _replace_estimator(self, attr, name, new_val):
        # 假设 name 是一个有效的估算器名称
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)
    # 验证给定的名称列表是否都是唯一的，如果有重复的名称将引发 ValueError 异常
    def _validate_names(self, names):
        # 检查名称列表中是否有重复的名称
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        
        # 检查给定的名称列表是否与构造函数参数冲突，如果有冲突将引发 ValueError 异常
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: {0!r}".format(
                    sorted(invalid_names)
                )
            )
        
        # 检查名称列表中是否有包含双下划线 '__' 的名称，如果有将引发 ValueError 异常
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got {0!r}".format(invalid_names)
            )
def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be used to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.
    """
    # Check if the estimator requires pairwise operations
    if _safe_tags(estimator, key="pairwise"):
        # Check if X has a 'shape' attribute (indicating it's an array or sparse matrix)
        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have "
                "to be passed as arrays or sparse matrices."
            )
        # Check if X is a square matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        # Select the subset of X based on indices and optionally train_indices
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]  # Slice rows and columns by indices
        else:
            X_subset = X[np.ix_(indices, train_indices)]  # Slice rows by indices and columns by train_indices
    else:
        # For non-pairwise estimators, simply index X by indices
        X_subset = _safe_indexing(X, indices)

    # If y is provided, index y by indices to get the subset
    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset
```