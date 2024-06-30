# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_split.py`

```
"""
The :mod:`sklearn.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers  # 导入用于数值类型判断的模块
import warnings  # 导入警告模块
from abc import ABCMeta, abstractmethod  # 导入抽象基类相关模块
from collections import defaultdict  # 导入默认字典模块
from collections.abc import Iterable  # 导入Iterable抽象基类
from inspect import signature  # 导入函数签名获取相关模块
from itertools import chain, combinations  # 导入迭代器工具模块
from math import ceil, floor  # 导入数学函数ceil和floor

import numpy as np  # 导入NumPy数值计算库
from scipy.special import comb  # 导入组合函数

from ..utils import (  # 导入工具函数
    _safe_indexing,
    check_random_state,
    indexable,
    metadata_routing,
)
from ..utils._array_api import (  # 导入数组API相关模块
    _convert_to_numpy,
    ensure_common_namespace_device,
    get_namespace,
)
from ..utils._param_validation import Interval, RealNotInt, validate_params  # 导入参数验证相关模块
from ..utils.extmath import _approximate_mode  # 导入数学函数_approximate_mode
from ..utils.metadata_routing import _MetadataRequester  # 导入元数据请求器
from ..utils.multiclass import type_of_target  # 导入多类别类型判断函数
from ..utils.validation import _num_samples, check_array, column_or_1d  # 导入验证函数

__all__ = [  # 定义可导出的模块成员列表
    "BaseCrossValidator",
    "KFold",
    "GroupKFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "RepeatedStratifiedKFold",
    "RepeatedKFold",
    "ShuffleSplit",
    "GroupShuffleSplit",
    "StratifiedKFold",
    "StratifiedGroupKFold",
    "StratifiedShuffleSplit",
    "PredefinedSplit",
    "train_test_split",
    "check_cv",
]


class _UnsupportedGroupCVMixin:
    """Mixin for splitters that do not support Groups."""

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return super().split(X, y, groups=groups)


class GroupsConsumerMixin(_MetadataRequester):
    """A Mixin to ``groups`` by default.

    This Mixin makes the object to request ``groups`` by default as ``True``.

    .. versionadded:: 1.3
    """

    __metadata_request__split = {"groups": True}


class BaseCrossValidator(_MetadataRequester, metaclass=ABCMeta):
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # 定义一个类变量，用于标识“分割请求”，指示不使用任何元数据分组
    # 也防止对不支持“groups”的分隔器生成“set_split_request”
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def split(self, X, y=None, groups=None):
        """生成用于将数据拆分为训练集和测试集的索引。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据，其中 `n_samples` 是样本数，`n_features` 是特征数。

        y : array-like of shape (n_samples,)
            监督学习问题的目标变量。

        groups : array-like of shape (n_samples,), default=None
            用于在拆分数据集成训练集/测试集时指定样本的分组标签。

        Yields
        ------
        train : ndarray
            训练集的索引数组。

        test : ndarray
            测试集的索引数组。
        """
        # 调用 indexable 函数确保 X, y, groups 是可索引的
        X, y, groups = indexable(X, y, groups)
        # 生成索引范围数组
        indices = np.arange(_num_samples(X))
        # 对每个测试集掩码进行迭代
        for test_index in self._iter_test_masks(X, y, groups):
            # 获取训练集索引，即不在测试集中的索引
            train_index = indices[np.logical_not(test_index)]
            # 获取测试集索引
            test_index = indices[test_index]
            yield train_index, test_index

    # 由于子类必须实现 _iter_test_masks 或 _iter_test_indices 其中之一，因此不能是抽象的。
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """生成与测试集对应的布尔掩码。

        默认情况下，委托给 _iter_test_indices(X, y, groups)。
        """
        for test_index in self._iter_test_indices(X, y, groups):
            # 创建与样本数相同的布尔掩码
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            # 将测试集索引位置置为 True
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """生成与测试集对应的整数索引。"""
        # 抛出未实现错误，要求子类实现该方法
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回交叉验证器中的分割迭代次数。"""

    def __repr__(self):
        return _build_repr(self)
class LeaveOneOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
    """Leave-One-Out cross-validator.

    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.

    Note: ``LeaveOneOut()`` is equivalent to ``KFold(n_splits=n)`` and
    ``LeavePOut(p=1)`` where ``n`` is the number of samples.

    Due to the high number of test sets (which is the same as the
    number of samples) this cross-validation method can be very costly.
    For large datasets one should favor :class:`KFold`, :class:`ShuffleSplit`
    or :class:`StratifiedKFold`.

    Read more in the :ref:`User Guide <leave_one_out>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([1, 2])
    >>> loo = LeaveOneOut()
    >>> loo.get_n_splits(X)
    2
    >>> print(loo)
    LeaveOneOut()
    >>> for i, (train_index, test_index) in enumerate(loo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1]
      Test:  index=[0]
    Fold 1:
      Train: index=[0]
      Test:  index=[1]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit,
        domain-specific stratification of the dataset.
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_indices(self, X, y=None, groups=None):
        # 获取样本数量
        n_samples = _num_samples(X)
        # 如果样本数量小于等于1，抛出异常
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
            )
        # 返回一个迭代器，迭代范围为样本数量
        return range(n_samples)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 如果 X 为 None，则抛出异常
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        # 返回样本数量
        return _num_samples(X)


class LeavePOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
    """Leave-P-Out cross-validator.

    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.

    Note: ``LeavePOut(p)`` is NOT equivalent to
    ```
    """
    Implement the Leave-P-Out cross-validator for splitting the data into
    non-overlapping test sets of size p.

    Due to the combinatorial growth of iterations with the number of samples,
    this cross-validation method can be computationally expensive. For larger
    datasets, consider alternatives like KFold, StratifiedKFold, or ShuffleSplit.

    Read more in the User Guide on Leave-P-Out cross-validation.

    Parameters
    ----------
    p : int
        Size of the test sets. Must be strictly less than the number of
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> lpo = LeavePOut(2)
    >>> lpo.get_n_splits(X)
    6
    >>> print(lpo)
    LeavePOut(p=2)
    >>> for i, (train_index, test_index) in enumerate(lpo.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 2:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 3:
      Train: index=[0 3]
      Test:  index=[1 2]
    Fold 4:
      Train: index=[0 2]
      Test:  index=[1 3]
    Fold 5:
      Train: index=[0 1]
      Test:  index=[2 3]
    """

    # 初始化 Leave-P-Out 分割器，设定测试集大小为 p
    def __init__(self, p):
        self.p = p

    # 生成测试集索引的迭代器
    def _iter_test_indices(self, X, y=None, groups=None):
        # 获取样本数
        n_samples = _num_samples(X)
        # 若样本数小于等于 p，则抛出异常
        if n_samples <= self.p:
            raise ValueError(
                "p={} must be strictly less than the number of samples={}".format(
                    self.p, n_samples
                )
            )
        # 生成所有可能的组合，每个组合包含 p 个索引
        for combination in combinations(range(n_samples), self.p):
            yield np.array(combination)

    # 返回交叉验证器中的拆分迭代次数
    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.
        """
        # 检查 X 是否为 None
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        # 返回拆分次数，使用组合数计算方式确保精确性
        return int(comb(_num_samples(X), self.p, exact=True))
class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for K-Fold cross-validators and TimeSeriesSplit."""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        # 确保 n_splits 是整数类型
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        # 确保 n_splits 至少为 2
        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        # 确保 shuffle 是布尔类型
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        # 如果 shuffle=False，确保 random_state 为 None
        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        # 初始化对象的属性
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 确保输入数据可以索引化
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        # 确保 n_splits 不超过样本数目
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        # 调用父类的 split 方法生成训练集和测试集的索引
        for train, test in super().split(X, y, groups):
            yield train, test
    # 返回交叉验证器中的分割迭代次数
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 返回保存在对象中的分割次数
        return self.n_splits
class KFold(_UnsupportedGroupCVMixin, _BaseKFold):
    """K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Read more in the :ref:`User Guide <k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(kf.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2 3]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).

    GroupKFold : K-fold iterator variant with non-overlapping groups.

    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # 调用父类的构造方法，初始化 KFold 对象
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    # 定义一个方法，用于生成交叉验证的折叠索引
    def _iter_test_indices(self, X, y=None, groups=None):
        # 获取样本数
        n_samples = _num_samples(X)
        # 创建一个包含所有样本索引的数组
        indices = np.arange(n_samples)
        # 如果设置了shuffle标志，打乱索引数组
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        # 确定交叉验证折叠的数量
        n_splits = self.n_splits
        # 计算每个折叠的大小，尽可能均匀地分配样本
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        # 初始化当前处理的起始索引位置
        current = 0
        # 遍历每个折叠的大小
        for fold_size in fold_sizes:
            # 确定当前折叠的起始和结束索引
            start, stop = current, current + fold_size
            # 生成当前折叠的测试索引，使用yield实现生成器功能
            yield indices[start:stop]
            # 更新当前处理的位置到下一个折叠的起始位置
            current = stop
# 定义一个自定义的 K 折交叉验证类 GroupKFold，继承自 GroupsConsumerMixin 和 _BaseKFold
class GroupKFold(GroupsConsumerMixin, _BaseKFold):
    """K-fold iterator variant with non-overlapping groups.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of
    samples is approximately the same in each test fold.

    Read more in the :ref:`User Guide <group_k_fold>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    Notes
    -----
    Groups appear in an arbitrary order throughout the folds.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([0, 0, 2, 2, 3, 3])
    >>> group_kfold = GroupKFold(n_splits=2)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2)
    >>> for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1 4 5], group=[0 0 3 3]
    Fold 1:
      Train: index=[0 1 4 5], group=[0 0 3 3]
      Test:  index=[2 3], group=[2 2]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit
        domain-specific stratification of the dataset.

    StratifiedKFold : Takes class information into account to avoid building
        folds with imbalanced class proportions (for binary or multiclass
        classification tasks).
    """

    # 初始化函数，设定 K 折交叉验证的参数
    def __init__(self, n_splits=5):
        # 调用父类 _BaseKFold 的初始化函数，传入 n_splits，同时设置不进行数据集的随机洗牌和随机状态为 None
        super().__init__(n_splits, shuffle=False, random_state=None)
    def _iter_test_indices(self, X, y, groups):
        # 如果未提供 groups 参数，则抛出数值错误异常
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        
        # 检查并转换 groups 参数为数组，确保其维度为2
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        # 返回 groups 数组中唯一值和对应的反向索引
        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        # 如果分组数量小于分割数目，则抛出数值错误异常
        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # 计算每个分组的样本数目
        n_samples_per_group = np.bincount(groups)

        # 将分组按样本数目从大到小排序的索引
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # 每个折叠的总样本数
        n_samples_per_fold = np.zeros(self.n_splits)

        # 将分组索引映射到折叠索引
        group_to_fold = np.zeros(len(unique_groups))

        # 分配样本到折叠中，将最大权重加到最轻的折叠中
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        # 根据分组索引重置 indices 变量
        indices = group_to_fold[groups]

        # 生成每个折叠的训练集和测试集索引
        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 调用父类的 split 方法来生成训练集和测试集的索引
        return super().split(X, y, groups)
# 定义了一个继承自 _BaseKFold 的类 StratifiedKFold，用于分层 K 折交叉验证。
class StratifiedKFold(_BaseKFold):
    """Stratified K-Fold cross-validator.

    提供训练/测试索引以将数据拆分为训练集和测试集。

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    这是一个返回分层折叠的 K 折交叉验证对象。通过保持每个类别样本的百分比来生成折叠。

    Read more in the :ref:`User Guide <stratified_k_fold>`.

    详细信息请参阅用户指南中的 :ref:`Stratified K-Fold <stratified_k_fold>`。

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    有关交叉验证行为的可视化和常见 scikit-learn 分割方法之间的比较，请参阅 :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        折叠的数量。必须至少为 2。

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

        是否在拆分成批之前对每个类别的样本进行洗牌。请注意，每个拆分内部的样本不会被洗牌。

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

        当 `shuffle` 为 True 时，`random_state` 影响索引的排序，从而控制每个类别每个折叠的随机性。
        否则，将 `random_state` 设置为 `None`。
        传入一个整数以在多次函数调用中获得可重复的输出。
        参见 :term:`Glossary <random_state>`。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.

    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

# 这里不需要额外的注释，因为函数和类的定义本身已经提供了详细的解释。
    # 初始化方法，设置交叉验证的参数
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        # 调用父类的初始化方法，设定交叉验证的参数
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # 生成测试集掩码的迭代器
    def _iter_test_masks(self, X, y=None, groups=None):
        # 使用_make_test_folds方法生成测试折叠
        test_folds = self._make_test_folds(X, y)
        # 遍历生成器，产生每个分割的测试集掩码
        for i in range(self.n_splits):
            yield test_folds == i

    # 分割数据集为训练集和测试集的索引
    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        # 如果提供了groups参数，则发出警告，但不会使用它
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # 对y进行检查，确保其为数组形式，不强制为2维
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        # 调用父类的split方法，生成训练集和测试集的索引
        return super().split(X, y, groups)
class StratifiedGroupKFold(GroupsConsumerMixin, _BaseKFold):
    """Stratified K-Fold iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold attempts to
    return stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The difference between :class:`~sklearn.model_selection.GroupKFold`
    and :class:`~sklearn.model_selection.StratifiedGroupKFold` is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class as much as possible given the
    constraint of non-overlapping groups between splits.

    Read more in the :ref:`User Guide <cross_validation>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> sgkf = StratifiedGroupKFold(n_splits=3)
    >>> sgkf.get_n_splits(X, y)
    3
    >>> print(sgkf)
    StratifiedGroupKFold(n_splits=3, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"         group={groups[train_index]}")  # 输出训练集的组标识
    ...     print(f"  Test:  index={test_index}")           # 输出测试集的索引
    ...     print(f"         group={groups[test_index]}")  # 输出测试集的组标识
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


# 初始化函数，用于初始化 KFold 对象
def __init__(self, n_splits=5, shuffle=False, random_state=None):
    # 调用父类的初始化方法，设置折叠数、是否洗牌、随机种子
    super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)



    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval
                or np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold


# 选择最佳的折叠方法函数
def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
    # 初始化最佳折叠、评估值最小、最小折叠样本数
    best_fold = None
    min_eval = np.inf
    min_samples_in_fold = np.inf
    # 遍历所有折叠
    for i in range(self.n_splits):
        # 将当前组的类别计数加到第 i 个折叠中
        y_counts_per_fold[i] += group_y_counts
        # 计算每个建议折叠中类别分布的总结
        std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
        # 恢复第 i 个折叠的类别计数
        y_counts_per_fold[i] -= group_y_counts
        # 计算当前折叠的评估值
        fold_eval = np.mean(std_per_class)
        # 计算当前折叠的样本数
        samples_in_fold = np.sum(y_counts_per_fold[i])
        # 判断当前折叠是否更好
        is_current_fold_better = (
            fold_eval < min_eval
            or np.isclose(fold_eval, min_eval)
            and samples_in_fold < min_samples_in_fold
        )
        # 如果当前折叠更好，则更新最佳折叠、最小评估值、最小折叠样本数
        if is_current_fold_better:
            min_eval = fold_eval
            min_samples_in_fold = samples_in_fold
            best_fold = i
    # 返回最佳折叠的索引
    return best_fold
class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    .. versionadded:: 0.18

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

        .. versionadded:: 0.24

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

        .. versionadded:: 0.24

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0]
      Test:  index=[1]
    Fold 1:
      Train: index=[0 1]
      Test:  index=[2]
    Fold 2:
      Train: index=[0 1 2]
      Test:  index=[3]
    Fold 3:
      Train: index=[0 1 2 3]
      Test:  index=[4]
    Fold 4:
      Train: index=[0 1 2 3 4]
      Test:  index=[5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for i, (train_index, test_index) in enumerate(tscv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2 3 4 5]
      Test:  index=[6 7]
    Fold 1:
      Train: index=[0 1 2 3 4 5 6 7]
      Test:  index=[8 9]
    """
    初始化一个时间序列分割器，继承自基类 TimeSeriesSplit。

    Parameters
    ----------
    n_splits : int, 默认为 5
        分割的次数，即折数。
    max_train_size : int or None, 可选
        训练集的最大大小。如果为 None，则不限制。
    test_size : int or None, 可选
        测试集的大小。如果为 None，则采用默认值。
    gap : int, 默认为 0
        训练集和测试集之间的间隔大小。

    Attributes
    ----------
    max_train_size : int or None
        训练集的最大大小。
    test_size : int or None
        测试集的大小。
    gap : int
        训练集和测试集之间的间隔大小。

    Methods
    -------
    split(X, y=None, groups=None)
        生成训练集和测试集的索引。

    Notes
    -----
    根据给定的参数和默认值计算训练集和测试集的大小。
    """
    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        # 调用父类的初始化方法，设置分割次数、是否随机洗牌、随机数种子等
        super().__init__(n_splits, shuffle=False, random_state=None)
        # 设置最大训练集大小
        self.max_train_size = max_train_size
        # 设置测试集大小
        self.test_size = test_size
        # 设置训练集和测试集之间的间隔大小
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """生成用于训练集和测试集的索引。

        Parameters
        ----------
        X : 形状为 (n_samples, n_features) 的数组
            训练数据，其中 `n_samples` 是样本数，`n_features` 是特征数。
        y : 形状为 (n_samples,) 的数组，可选
            总是被忽略，仅存在以保持兼容性。
        groups : 形状为 (n_samples,) 的数组，可选
            总是被忽略，仅存在以保持兼容性。

        Yields
        ------
        train : ndarray
            该次分割的训练集索引。
        test : ndarray
            该次分割的测试集索引。
        """
        # 如果存在 groups 参数，则发出警告
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # 调用内部方法 _split 生成训练集和测试集索引
        return self._split(X)
    # 定义一个方法用于生成将数据分割为训练集和测试集的索引

    def _split(self, X):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 调用 indexable 函数，确保 X 可索引
        (X,) = indexable(X)
        # 获取样本数
        n_samples = _num_samples(X)
        # 获取切分数
        n_splits = self.n_splits
        # 计算折数
        n_folds = n_splits + 1
        # 获取间隔大小
        gap = self.gap
        # 获取测试集大小
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # 确保给定的切分参数下有足够的样本数
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        # 确保样本数、测试集大小和间隔能够支持给定的切分数
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        # 创建索引数组，从 0 到 n_samples-1
        indices = np.arange(n_samples)
        # 计算每个测试集的起始索引
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        # 遍历每个测试集的起始索引，生成训练集和测试集的索引对
        for test_start in test_starts:
            train_end = test_start - gap
            # 如果设置了最大训练集大小，并且超出了这个大小，则截取最大训练集大小
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size : train_end],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )
class LeaveOneGroupOut(GroupsConsumerMixin, BaseCrossValidator):
    """Leave One Group Out cross-validator.

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group.
    Arbitrary domain specific group information is provided an array integers
    that encodes the group of each sample.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    Read more in the :ref:`User Guide <leave_one_group_out>`.

    Notes
    -----
    Splits are ordered according to the index of the group left out. The first
    split has testing set consisting of the group whose index in `groups` is
    lowest, and so on.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneGroupOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([1, 1, 2, 2])
    >>> logo = LeaveOneGroupOut()
    >>> logo.get_n_splits(X, y, groups)
    2
    >>> logo.get_n_splits(groups=groups)  # 'groups' is always required
    2
    >>> print(logo)
    LeaveOneGroupOut()
    >>> for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1], group=[1 1]
      Test:  index=[2 3], group=[2 2]

    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_masks(self, X, y, groups):
        """Generates boolean masks corresponding to each unique group in 'groups'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target values.

        groups : array-like of shape (n_samples,)
            Group labels for each sample.

        Yields
        ------
        mask : boolean array of shape (n_samples,)
            Boolean mask indicating which samples are in the test set.

        Raises
        ------
        ValueError
            If 'groups' is None or contains fewer than 2 unique groups.

        Notes
        -----
        Each yielded mask corresponds to one unique group in 'groups', with
        'True' indicating the samples in the test set for that group.

        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        for i in unique_groups:
            yield groups == i
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 如果 'groups' 参数为 None，则抛出值错误异常
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # 将 groups 转换成数组，并进行验证
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        # 返回唯一的 groups 数量作为分割次数
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 调用父类的 split 方法，生成训练集和测试集的索引
        return super().split(X, y, groups)
# 定义 LeavePGroupsOut 类，继承自 GroupsConsumerMixin 和 BaseCrossValidator
class LeavePGroupsOut(GroupsConsumerMixin, BaseCrossValidator):
    """Leave P Group(s) Out cross-validator.

    提供一个用于交叉验证的 Leave P Group(s) Out 方法。

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.

    提供训练/测试索引，根据第三方提供的组来划分数据。此组信息可用于将样本的任意领域特定分层编码为整数。

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    例如，组可以是样本收集的年份，因此允许根据基于时间的分割进行交叉验证。

    The difference between LeavePGroupsOut and LeaveOneGroupOut is that
    the former builds the test sets with all the samples assigned to
    ``p`` different values of the groups while the latter uses samples
    all assigned the same groups.

    LeavePGroupsOut 与 LeaveOneGroupOut 的区别在于前者使用所有分配给 'p' 个不同组值的样本构建测试集，而后者使用分配给同一组的所有样本。

    Read more in the :ref:`User Guide <leave_p_groups_out>`.

    在 :ref:`User Guide <leave_p_groups_out>` 中阅读更多内容。

    Parameters
    ----------
    n_groups : int
        Number of groups (``p``) to leave out in the test split.

    n_groups: int
        在测试拆分中要留出的组数（``p``）。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePGroupsOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1])
    >>> groups = np.array([1, 2, 3])
    >>> lpgo = LeavePGroupsOut(n_groups=2)
    >>> lpgo.get_n_splits(X, y, groups)
    3
    >>> lpgo.get_n_splits(groups=groups)  # 'groups' is always required
    3
    >>> print(lpgo)
    LeavePGroupsOut(n_groups=2)
    >>> for i, (train_index, test_index) in enumerate(lpgo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2], group=[3]
      Test:  index=[0 1], group=[1 2]
    Fold 1:
      Train: index=[1], group=[2]
      Test:  index=[0 2], group=[1 3]
    Fold 2:
      Train: index=[0], group=[1]
      Test:  index=[1 2], group=[2 3]

    See Also
    --------
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    """

    # 初始化方法，设置要留出的组数 n_groups
    def __init__(self, n_groups):
        self.n_groups = n_groups
    def _iter_test_masks(self, X, y, groups):
        # 如果没有提供 groups 参数，抛出数值错误异常
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # 检查并复制 groups 参数，确保是一维数组
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        # 获取唯一的分组标签
        unique_groups = np.unique(groups)
        # 如果 n_groups 大于等于唯一分组标签的数量，抛出数值错误异常
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1)
            )
        # 生成唯一分组标签的组合，长度为 self.n_groups
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            # 创建一个布尔数组，用于标记测试集索引
            test_index = np.zeros(_num_samples(X), dtype=bool)
            # 根据当前组合的分组标签设置对应的测试集索引为 True
            for l in unique_groups[np.array(indices)]:
                test_index[groups == l] = True
            # 生成当前测试集索引
            yield test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 如果没有提供 groups 参数，抛出数值错误异常
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # 检查并返回 groups 参数，确保是一维数组
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        # 返回计算得到的分裂次数
        return int(comb(len(np.unique(groups)), self.n_groups, exact=True))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 调用父类的 split 方法，返回分裂后的训练集和测试集索引
        return super().split(X, y, groups)
# 定义一个带有元类ABCMeta的类_RepeatedSplits，继承自_MetadataRequester类
class _RepeatedSplits(_MetadataRequester, metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.

    Repeats splits for cross-validators n times with different randomization
    in each repetition.

    Parameters
    ----------
    cv : callable
        Cross-validator class.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Passes `random_state` to the arbitrary repeating cross validator.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    # 这表明默认情况下，CV分离器不具有 "groups" 关键字参数，
    # 除非从 "GroupsConsumerMixin" 继承。
    # 这也防止为不支持 "groups" 的分离器生成 "set_split_request"。
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(self, cv, *, n_repeats=10, random_state=None, **cvargs):
        # 检查 n_repeats 是否为整数类型
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")

        # 检查 n_repeats 是否大于 0
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        # 检查 cvargs 中是否包含 "random_state" 或 "shuffle"
        if any(key in cvargs for key in ("random_state", "shuffle")):
            raise ValueError("cvargs must not contain random_state or shuffle.")

        # 将参数保存为实例变量
        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 获取重复次数和随机数生成器
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        # 迭代生成不同随机化下的分割索引
        for idx in range(n_repeats):
            # 使用给定参数实例化交叉验证器
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            # 遍历交叉验证器生成的每个分割
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index
    # 返回交叉验证器中的分割迭代次数

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.

        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 检查随机状态，并生成一个随机数生成器对象
        rng = check_random_state(self.random_state)
        # 使用给定的随机状态、打乱数据的方式和其他参数创建交叉验证对象
        cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
        # 返回交叉验证对象的分割迭代次数乘以重复次数
        return cv.get_n_splits(X, y, groups) * self.n_repeats

    def __repr__(self):
        # 调用私有函数 `_build_repr` 来生成对象的字符串表示形式
        return _build_repr(self)
class RepeatedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    """Repeated K-Fold cross validator.

    Repeats K-Fold n times with different randomization in each repetition.

    Read more in the :ref:`User Guide <repeated_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> rkf.get_n_splits(X, y)
    4
    >>> print(rkf)
    RepeatedKFold(n_repeats=2, n_splits=2, random_state=2652124)
    >>> for i, (train_index, test_index) in enumerate(rkf.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    ...
    Fold 0:
      Train: index=[0 1]
      Test:  index=[2 3]
    Fold 1:
      Train: index=[2 3]
      Test:  index=[0 1]
    Fold 2:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 3:
      Train: index=[0 3]
      Test:  index=[1 2]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        # 调用父类的初始化方法，使用 KFold 作为基类，设置重复次数、随机状态和折数
        super().__init__(
            KFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits
        )


class RepeatedStratifiedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    """Repeated Stratified K-Fold cross validator.

    Repeats Stratified K-Fold n times with different randomization in each
    repetition.

    Read more in the :ref:`User Guide <repeated_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    # 创建一个重复的分层 K 折交叉验证对象，设置为每次分割包含 2 折，重复 2 次，使用固定的随机种子 36851234
    >>> rskf.get_n_splits(X, y)
    # 获取数据集 X 和标签 y 的分割次数，这里返回结果为 4，因为有 2 折重复 2 次
    4
    >>> print(rskf)
    # 打印输出重复的分层 K 折交叉验证对象的描述信息
    RepeatedStratifiedKFold(n_repeats=2, n_splits=2, random_state=36851234)
    >>> for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    ...
    # 遍历每个交叉验证折叠，打印每折的训练集和测试集的索引
    Fold 0:
      Train: index=[1 2]
      Test:  index=[0 3]
    Fold 1:
      Train: index=[0 3]
      Test:  index=[1 2]
    Fold 2:
      Train: index=[1 3]
      Test:  index=[0 2]
    Fold 3:
      Train: index=[0 2]
      Test:  index=[1 3]

    Notes
    -----
    # 说明随机化的交叉验证分割器可能在每次调用时返回不同的结果。可以通过设置 `random_state` 参数为整数来使结果保持一致。
    
    See Also
    --------
    RepeatedKFold : 重复 K 折交叉验证 n 次。
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        # 初始化方法，继承自父类，使用 StratifiedKFold 作为基础分割器
        super().__init__(
            StratifiedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
class BaseShuffleSplit(_MetadataRequester, metaclass=ABCMeta):
    """Base class for *ShuffleSplit.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        # Initialize the BaseShuffleSplit instance with provided parameters.
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        # Ensure X, y, groups are indexable
        X, y, groups = indexable(X, y, groups)
        # Iterate over the indices yielding train/test splits
        for train, test in self._iter_indices(X, y, groups):
            yield train, test
    # 生成 (train, test) 索引的迭代器方法
    def _iter_indices(self, X, y=None, groups=None):
        # 获取样本数量
        n_samples = _num_samples(X)
        # 根据设定的参数验证并计算训练集和测试集的大小
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        # 使用给定的随机状态创建随机数生成器
        rng = check_random_state(self.random_state)
        # 生成指定次数的循环，每次迭代生成一个(train, test)索引对
        for i in range(self.n_splits):
            # 随机打乱样本索引
            permutation = rng.permutation(n_samples)
            # 获取测试集的索引
            ind_test = permutation[:n_test]
            # 获取训练集的索引
            ind_train = permutation[n_test : (n_test + n_train)]
            # 生成训练集和测试集的索引对
            yield ind_train, ind_test

    # 返回交叉验证器的分裂次数
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    # 返回交叉验证器的字符串表示形式
    def __repr__(self):
        return _build_repr(self)
class ShuffleSplit(_UnsupportedGroupCVMixin, BaseShuffleSplit):
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <ShuffleSplit>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1, 2, 1, 2])
    >>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    >>> rs.get_n_splits(X)
    5
    >>> print(rs)
    ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3 0 4]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2 5]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4 0]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1 0]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1 0]
      Test:  index=[2 4]
    >>> # Specify train and test size
    >>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    """

    def __init__(self, n_splits=10, test_size=None, train_size=None, random_state=None):
        # 继承 _UnsupportedGroupCVMixin 和 BaseShuffleSplit，构造随机排列交叉验证器
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        # 初始化随机排列交叉验证器，设置参数 n_splits, test_size, train_size, random_state
        # 调用父类的初始化方法

    def split(self, X, y=None, groups=None):
        # 使用父类的 split 方法进行数据集拆分
        return super().split(X, y=y, groups=groups)
        # 返回拆分后的索引，用于训练和测试数据集
    """
    Fold 0:
      Train: index=[1 3 0]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1]
      Test:  index=[2 4]
    """

    # 初始化方法，用于创建一个交叉验证对象
    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        # 调用父类的初始化方法，设置交叉验证的参数
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        # 设置默认的测试集大小为 0.1
        self._default_test_size = 0.1
class GroupShuffleSplit(GroupsConsumerMixin, BaseShuffleSplit):
    """Shuffle-Group(s)-Out cross-validation iterator.

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    The difference between LeavePGroupsOut and GroupShuffleSplit is that
    the former generates splits using all subsets of size ``p`` unique groups,
    whereas GroupShuffleSplit generates a user-determined number of random
    test splits, each with a user-determined fraction of unique groups.

    For example, a less computationally intensive alternative to
    ``LeavePGroupsOut(p=10)`` would be
    ``GroupShuffleSplit(test_size=10, n_splits=100)``.

    Note: The parameters ``test_size`` and ``train_size`` refer to groups, and
    not to samples, as in ShuffleSplit.

    Read more in the :ref:`User Guide <group_shuffle_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    test_size : float, int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size.
        The default will change in version 0.21. It will remain 0.2 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupShuffleSplit
    >>> X = np.ones(shape=(8, 2))
    >>> y = np.ones(shape=(8, 1))
    >>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
    >>> print(groups.shape)
    (8,)
    >>> gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
    >>> gss.get_n_splits()
    2
    >>> print(gss)
    """

    def __init__(self, n_splits=5, test_size=0.2, train_size=None, random_state=None):
        """
        Initialize a GroupShuffleSplit instance.

        Parameters
        ----------
        n_splits : int, default=5
            Number of re-shuffling & splitting iterations.

        test_size : float, int, default=0.2
            Proportion of groups to include in the test split or absolute number
            of test groups if an integer. If None, it complements `train_size`.

        train_size : float or int, default=None
            Proportion of groups to include in the train split or absolute number
            of train groups if an integer. If None, it complements `test_size`.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the training and testing indices produced.

        Notes
        -----
        The test_size and train_size parameters refer to groups, not individual samples.
        """
        super().__init__(n_splits=n_splits, random_state=random_state)
        self.test_size = test_size
        self.train_size = train_size

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        The yield consists of indices to the original dataset. These indices are
        useful for splitting the dataset using different cross-validation strategies.
        """
        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(X, y, groups)
    GroupShuffleSplit(n_splits=2, random_state=42, test_size=None, train_size=0.7)
    # 创建一个GroupShuffleSplit对象，用于将数据集分成两个部分，保证每个部分都包含不同的组，随机性由random_state确定，不设定测试集大小，训练集大小设定为0.7

    >>> for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    # 使用GroupShuffleSplit对象gss对数据集(X, y)进行分割，同时遍历每次分割的索引和组信息
    ...     print(f"Fold {i}:")
    # 打印当前折叠（fold）的索引
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    # 打印当前训练集的索引和对应的组信息
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    # 打印当前测试集的索引和对应的组信息
    Fold 0:
      Train: index=[2 3 4 5 6 7], group=[2 2 2 3 3 3]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1 5 6 7], group=[1 1 3 3 3]
      Test:  index=[2 3 4], group=[2 2 2]

    See Also
    --------
    ShuffleSplit : Shuffles samples to create independent test/train sets.

    LeavePGroupsOut : Train set leaves out all possible subsets of `p` groups.
    """

    def __init__(
        self, n_splits=5, *, test_size=None, train_size=None, random_state=None
    ):
        # 初始化函数，设置分割次数n_splits，默认测试集大小test_size和训练集大小train_size，随机种子random_state
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2
        # 设置默认的测试集大小为0.2

    def _iter_indices(self, X, y, groups):
        # 生成器函数，用于生成每次分割的训练集和测试集的索引

        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # 如果groups为None，抛出数值错误异常

        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        # 检查并转换groups为数组形式，确保不是2维数组

        classes, group_indices = np.unique(groups, return_inverse=True)
        # 获取唯一的组类别和组的反向索引

        for group_train, group_test in super()._iter_indices(X=classes):
            # 对每个分组的训练集和测试集进行遍历
            # 这些是分区中类的索引，将它们反转为数据索引

            train = np.flatnonzero(np.isin(group_indices, group_train))
            # 在组索引中找到属于训练集的非零平坦索引

            test = np.flatnonzero(np.isin(group_indices, group_test))
            # 在组索引中找到属于测试集的非零平坦索引

            yield train, test
            # 生成训练集和测试集的索引

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)
        # 调用父类的split方法，生成数据集的训练集和测试集的索引
# StratifiedShuffleSplit 类定义，继承自 BaseShuffleSplit
class StratifiedShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross-validator.

    提供用于分割数据为训练集和测试集的训练/测试索引。

    该交叉验证对象是 StratifiedKFold 和 ShuffleSplit 的结合体，
    返回分层随机化的折叠。这些折叠通过保持每个类别样本的百分比来生成。

    注意：像 ShuffleSplit 策略一样，分层随机分割不能保证所有折叠都不同，但对于大型数据集仍然非常可能。

    详细信息请参阅 :ref:`User Guide <stratified_shuffle_split>`。

    若要查看交叉验证行为的可视化，并比较常见的 scikit-learn 分割方法，
    可参考 :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`。

    Parameters
    ----------
    n_splits : int, default=10
        重洗和分割迭代的次数。

    test_size : float or int, default=None
        如果是 float，应在 0.0 到 1.0 之间，表示测试集的比例。
        如果是 int，表示测试样本的绝对数量。
        如果是 None，则设置为训练集的补充值。
        如果 ``train_size`` 也是 None，则默认为 0.1。

    train_size : float or int, default=None
        如果是 float，应在 0.0 到 1.0 之间，表示训练集的比例。
        如果是 int，表示训练样本的绝对数量。
        如果是 None，则自动设置为测试集的补充值。

    random_state : int, RandomState instance or None, default=None
        控制生成的训练和测试索引的随机性。
        传入一个 int 可以实现多次函数调用的可复现输出。
        参见 :term:`Glossary <random_state>`。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    >>> sss.get_n_splits(X, y)
    5
    >>> print(sss)
    StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    >>> for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[5 2 3]
      Test:  index=[4 1 0]
    Fold 1:
      Train: index=[5 1 4]
      Test:  index=[0 2 3]
    Fold 2:
      Train: index=[5 0 2]
      Test:  index=[4 3 1]
    Fold 3:
      Train: index=[4 1 0]
      Test:  index=[2 3 5]
    Fold 4:
      Train: index=[0 5 1]
      Test:  index=[3 4 2]
    """

    # 构造函数，初始化 StratifiedShuffleSplit 对象
    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
        # 调用父类的构造函数，初始化交叉验证分割器
        super().__init__(
            n_splits=n_splits,  # 设置分割的份数
            test_size=test_size,  # 设置测试集大小
            train_size=train_size,  # 设置训练集大小
            random_state=random_state,  # 设置随机数种子
        )
        # 设置默认的测试集大小为 0.1
        self._default_test_size = 0.1
    # 定义一个方法 `_iter_indices`，用于生成训练集和测试集的索引
    def _iter_indices(self, X, y, groups=None):
        # 获取样本数量
        n_samples = _num_samples(X)
        
        # 检查并确保 `y` 是一个合法的数组，且不是二维的，可以是任意数据类型
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        
        # 验证并获取训练集和测试集的大小
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        # 将 `y` 转换为 numpy 数组，因为不是所有操作都支持 Array API
        xp, _ = get_namespace(y)
        y = _convert_to_numpy(y, xp=xp)

        # 如果 `y` 是二维的，对于多标签的情况，将每行转换为字符串表示
        if y.ndim == 2:
            y = np.array([" ".join(row.astype("str")) for row in y])

        # 获取类别列表和 `y` 的反向索引
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        # 统计每个类别的样本数量
        class_counts = np.bincount(y_indices)
        
        # 如果最少类别的样本数量小于 2，则抛出 ValueError
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class in y has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2."
            )

        # 如果训练集大小小于类别数目，则抛出 ValueError
        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )

        # 如果测试集大小小于类别数目，则抛出 ValueError
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # 根据类别索引排序样本索引，以便为每个类别找到实例列表
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        # 检查随机状态并初始化随机数生成器
        rng = check_random_state(self.random_state)

        # 生成指定数量的迭代次数，用于创建交叉验证的训练集和测试集索引
        for _ in range(self.n_splits):
            # 通过 `_approximate_mode` 函数估计每个类别的训练集和测试集大小
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            # 对每个类别执行随机排列，并根据排列生成训练集和测试集索引
            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

            # 对训练集和测试集索引进行随机排列
            train = rng.permutation(train)
            test = rng.permutation(test)

            # 生成当前迭代的训练集和测试集索引
            yield train, test
    # 定义一个方法用于生成数据集拆分的索引，将数据分为训练集和测试集

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        # 如果提供了 groups 参数，则发出警告并忽略它，这是为了兼容性而存在的
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # 检查并转换 y，确保其为数组格式，用于后续处理
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        # 调用父类的 split 方法进行数据集拆分，返回拆分后的训练集和测试集索引
        return super().split(X, y, groups)
# 定义一个辅助函数，用于验证分割数据时的大小参数是否合理，与数据集大小(n_samples)相关
def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful w.r.t. the
    size of the data (n_samples).
    """
    # 如果未指定 test_size 和 train_size，则使用默认的 test_size
    if test_size is None and train_size is None:
        test_size = default_test_size

    # 确定 test_size 和 train_size 的数据类型
    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    # 检查 test_size 是否合理，根据其数据类型进行不同的验证
    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    # 检查 train_size 是否合理，根据其数据类型进行不同的验证
    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    # 如果 train_size 或 test_size 的数据类型不是整数或浮点数，抛出异常
    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    # 如果 train_size 和 test_size 都是浮点数，且它们的和大于1，抛出异常
    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )

    # 根据 test_size 的类型计算测试集的样本数
    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    # 根据 train_size 的类型计算训练集的样本数
    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    # 如果未指定 train_size，则根据计算得到的 n_test 计算 n_train
    if train_size is None:
        n_train = n_samples - n_test
    # 如果未指定 test_size，则根据计算得到的 n_train 计算 n_test
    elif test_size is None:
        n_test = n_samples - n_train

    # 如果计算得到的 train_size 和 test_size 的总和大于 n_samples，抛出异常
    if n_train + n_test > n_samples:
        raise ValueError(
            "The sum of train_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "train_size." % (n_train + n_test, n_samples)
        )

    # 将 n_train 和 n_test 转换为整数类型
    n_train, n_test = int(n_train), int(n_test)

    # 如果计算得到的 n_train 为 0，抛出异常，因为训练集将为空集
    if n_train == 0:
        raise ValueError(
            "With n_samples={}, test_size={} and train_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, test_size, train_size)
        )

    # 返回经验证后的 n_train 和 n_test
    return n_train, n_test


class PredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator.
    
    Placeholder class definition for a predefined split cross-validator.
    """
    """
    Provides train/test indices to split data into train/test sets using a
    predefined scheme specified by the user with the ``test_fold`` parameter.

    Read more in the :ref:`User Guide <predefined_split>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set (i.e. include sample ``i`` in every training set) by
        setting ``test_fold[i]`` equal to -1.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import PredefinedSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> test_fold = [0, 1, -1, 1]
    >>> ps = PredefinedSplit(test_fold)
    >>> ps.get_n_splits()
    2
    >>> print(ps)
    PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))
    >>> for i, (train_index, test_index) in enumerate(ps.split()):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 2 3]
      Test:  index=[0]
    Fold 1:
      Train: index=[0 2]
      Test:  index=[1 3]
    """

    # 初始化函数，接受一个测试折叠参数 test_fold
    def __init__(self, test_fold):
        # 将 test_fold 转换为 numpy 数组，强制数据类型为整数
        self.test_fold = np.array(test_fold, dtype=int)
        # 调用 column_or_1d 函数确保 test_fold 是一维数组
        self.test_fold = column_or_1d(self.test_fold)
        # 找出所有不重复的折叠数值（不包括 -1），存储在 unique_folds 中
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

    # 分割数据集的方法，返回训练集和测试集的索引
    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 如果 groups 参数不为空，发出警告
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        # 调用内部方法 _split 来生成训练集和测试集的索引
        return self._split()

    # 内部方法，生成训练集和测试集的索引
    def _split(self):
        """Generate indices to split data into training and test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 创建索引数组 ind，范围为测试折叠参数的长度
        ind = np.arange(len(self.test_fold))
        # 遍历 _iter_test_masks 方法生成的测试集索引掩码
        for test_index in self._iter_test_masks():
            # 通过逻辑非操作得到训练集索引
            train_index = ind[np.logical_not(test_index)]
            # 通过测试集索引掩码得到测试集索引
            test_index = ind[test_index]
            # 生成并返回训练集和测试集的索引
            yield train_index, test_index
    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        # 遍历每个唯一的折叠编号
        for f in self.unique_folds:
            # 找到测试集中与当前折叠编号匹配的索引
            test_index = np.where(self.test_fold == f)[0]
            # 创建一个与测试折叠长度相同的布尔类型的零数组
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            # 将测试索引对应位置设置为 True
            test_mask[test_index] = True
            # 返回当前生成的测试集布尔掩码
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 返回唯一折叠编号列表的长度，即拆分迭代次数
        return len(self.unique_folds)
class _CVIterableWrapper(BaseCrossValidator):
    """Wrapper class for old style cv objects and iterables."""

    def __init__(self, cv):
        # 将传入的交叉验证对象转换为列表形式
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        # 返回交叉验证对象的分割次数，即列表长度
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # 遍历交叉验证对象的每个分割，生成训练集和测试集的索引
        for train, test in self.cv:
            yield train, test


def check_cv(cv=5, y=None, *, classifier=False):
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator, iterable or None, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For integer/None inputs, if classifier is True and ``y`` is either
        binary or multiclass, :class:`StratifiedKFold` is used. In all other
        cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value changed from 3-fold to 5-fold.

    y : array-like, default=None
        The target variable for supervised learning problems.

    classifier : bool, default=False
        Whether the task is a classification task, in which case
        stratified KFold will be used.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    Examples
    --------
    >>> from sklearn.model_selection import check_cv
    >>> check_cv(cv=5, y=None, classifier=False)
    KFold(...)
    >>> check_cv(cv=5, y=[1, 1, 0, 0, 0, 0], classifier=True)
    StratifiedKFold(...)
    """
    # 根据输入参数构建和返回合适的交叉验证器实例
    # 具体使用 StratifiedKFold 或 KFold 取决于 classifier 和 y 的类型
    pass
    """
    根据条件返回适当的交叉验证对象。
    cv 如果为 None，则默认为 5。
    如果 cv 是整数类型：
        - 如果 classifier 存在，并且 y 不为 None，并且 y 的类型是二元或多类别的，则返回 StratifiedKFold(cv)。
        - 否则返回 KFold(cv)。
    如果 cv 不具有 "split" 属性或者是字符串类型：
        - 如果 cv 不是 Iterable 或者是字符串，则引发 ValueError。
        返回 _CVIterableWrapper(cv)。
    否则直接返回 cv，表示新风格的 cv 对象被原样传递。
    """
    cv = 5 if cv is None else cv  # 如果 cv 为 None，则默认为 5
    if isinstance(cv, numbers.Integral):  # 如果 cv 是整数类型
        if (
            classifier  # 如果 classifier 存在
            and (y is not None)  # 并且 y 不为 None
            and (type_of_target(y, input_name="y") in ("binary", "multiclass"))  # 并且 y 的类型是二元或多类别的
        ):
            return StratifiedKFold(cv)  # 返回 StratifiedKFold(cv)
        else:
            return KFold(cv)  # 返回 KFold(cv)
    
    if not hasattr(cv, "split") or isinstance(cv, str):  # 如果 cv 不具有 "split" 属性或者是字符串类型
        if not isinstance(cv, Iterable) or isinstance(cv, str):  # 如果 cv 不是 Iterable 或者是字符串
            raise ValueError(
                "Expected cv as an integer, cross-validation "
                "object (from sklearn.model_selection) "
                "or an iterable. Got %s." % cv
            )  # 抛出 ValueError 异常，提示错误信息
        return _CVIterableWrapper(cv)  # 返回 _CVIterableWrapper(cv)
    
    return cv  # 返回 cv，表示新风格的 cv 对象被原样传递
# 使用装饰器 validate_params 进行参数验证，确保 train_test_split 函数的参数符合规范
@validate_params(
    {
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),  # 验证 test_size 参数是否为非整数实数，且在 (0, 1) 范围内
            Interval(numbers.Integral, 1, None, closed="left"),  # 验证 test_size 参数是否为整数，且大于等于 1
            None,  # 允许 test_size 参数为 None
        ],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),  # 验证 train_size 参数是否为非整数实数，且在 (0, 1) 范围内
            Interval(numbers.Integral, 1, None, closed="left"),  # 验证 train_size 参数是否为整数，且大于等于 1
            None,  # 允许 train_size 参数为 None
        ],
        "random_state": ["random_state"],  # 验证 random_state 参数是否为整数或 RandomState 实例，或者为 None
        "shuffle": ["boolean"],  # 验证 shuffle 参数是否为布尔值
        "stratify": ["array-like", None],  # 验证 stratify 参数是否为类数组或 None
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# 定义 train_test_split 函数，用于将输入的数组或矩阵随机分割为训练集和测试集
def train_test_split(
    *arrays,  # 接受可变数量的参数，每个参数是一个可索引对象，长度或形状相同
    test_size=None,  # 测试集大小，默认为 None
    train_size=None,  # 训练集大小，默认为 None
    random_state=None,  # 控制数据分割前的随机排列，默认为 None
    shuffle=True,  # 是否在分割前对数据进行随机排列，默认为 True
    stratify=None,  # 如果不为 None，则按指定的类标签进行分层分割，默认为 None
):
    """Split arrays or matrices into random train and test subsets.

    Quick utility that wraps input validation,
    ``next(ShuffleSplit().split(X, y))``, and application to input data
    into a single call for splitting (and optionally subsampling) data into a
    one-liner.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]

    >>> train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]
    """
    n_arrays = len(arrays)  # 计算输入数组的数量

    # 如果没有输入数组，则抛出数值错误
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    # 将输入的多个数组转换为可索引的形式
    arrays = indexable(*arrays)

    # 获取第一个数组中的样本数量
    n_samples = _num_samples(arrays[0])

    # 根据参数验证并获取训练集和测试集的样本数量
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    # 如果不进行打乱(shuffle=False)处理
    if shuffle is False:
        # 如果指定了分层(stratify)，则抛出值错误
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        # 创建训练集和测试集的索引
        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        # 如果指定了分层(stratify)，选择适当的交叉验证类
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        # 使用指定参数创建交叉验证对象
        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        # 获取交叉验证的训练集和测试集索引
        train, test = next(cv.split(X=arrays[0], y=stratify))

    # 确保训练集和测试集的数据处于共同的命名空间和设备上
    train, test = ensure_common_namespace_device(arrays[0], train, test)

    # 返回所有数组按照训练集和测试集索引划分后的结果
    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )
# 设置 train_test_split 函数的 "__test__" 属性为 False，告诉 nose 它不是测试函数。
# 这是为了避免当 monkeypatching 时，mypy 报错。
setattr(train_test_split, "__test__", False)


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int, default=0
        The offset in characters to add at the begin of each line.

    printer : callable, default=repr
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    # 获取当前的打印选项
    options = np.get_printoptions()
    # 临时修改打印选项，设置精度、阈值和边缘项
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    # 初始化参数列表和当前行长度
    params_list = list()
    this_line_length = offset
    # 定义行分隔符，带有偏移量的空格
    line_sep = ",\n" + (1 + offset // 2) * " "
    # 遍历并排序参数字典中的键值对
    for i, (k, v) in enumerate(sorted(params.items())):
        # 根据值的类型选择合适的表示方法
        if isinstance(v, float):
            # 对于浮点数，使用 str 表示，确保跨不同架构和版本保持一致
            this_repr = "%s=%s" % (k, str(v))
        else:
            # 对于其他类型，使用指定的 printer 转换为字符串表示
            this_repr = "%s=%s" % (k, printer(v))
        # 如果字符串超过 500 字符，进行截断处理
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + "..." + this_repr[-100:]
        # 根据当前行长度和新加入的字符串长度判断是否需要换行
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or "\n" in this_repr:
                # 需要换行时，添加行分隔符
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                # 否则添加逗号和空格
                params_list.append(", ")
                this_line_length += 2
        # 添加当前参数表示到参数列表，并更新当前行长度
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    # 恢复之前的打印选项
    np.set_printoptions(**options)
    # 将参数列表拼接成最终的输出字符串
    lines = "".join(params_list)
    # 去除每行末尾的空格，避免在 doctest 中引起问题
    lines = "\n".join(l.rstrip(" ") for l in lines.split("\n"))
    return lines


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    # 获取当前对象的类
    cls = self.__class__
    # 获取类的初始化方法，如果有替代的方法则使用替代方法
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # 获取初始化方法的签名信息
    init_signature = signature(init)
    # 排除 'self' 参数外，考虑构造函数的其余参数
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    # 获取当前对象的类名
    class_name = self.__class__.__name__
    # 初始化参数字典
    params = dict()
    # 遍历参数列表中的每个键（可能是属性名）
    for key in args:
        # 我们需要始终打开过时警告，以便捕获已弃用的参数值。
        # 这在 utils/__init__.py 中设置，但在某种情况下在 python3 下会被覆盖。
        warnings.simplefilter("always", FutureWarning)
        try:
            # 尝试获取当前对象 self 中键名为 key 的属性值
            value = getattr(self, key, None)
            # 如果值为 None 并且 self 中有属性 "cvargs"，则尝试从中获取 key 的值
            if value is None and hasattr(self, "cvargs"):
                value = self.cvargs.get(key, None)
            # 如果有警告记录，则捕获所有警告
            with warnings.catch_warnings(record=True) as w:
                # 如果参数值为已弃用，不显示它
                if len(w) and w[0].category == FutureWarning:
                    continue
        finally:
            # 移除最早添加的警告过滤器
            warnings.filters.pop(0)
        # 将参数及其对应的值存入 params 字典中
        params[key] = value

    # 返回格式化后的字符串，显示类名及其参数
    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))
# 判断是否生成常数分割的交叉验证对象
def _yields_constant_splits(cv):
    # 如果调用 cv.split() 总是返回相同的分割，则返回 True
    # 假设如果一个交叉验证对象没有 shuffle 参数，它默认是会打乱顺序的（例如 ShuffleSplit）。
    # 如果实际上不会打乱顺序（例如 LeaveOneOut），那么它也不会有 random_state 参数，默认为 0，导致输出为 True。
    
    # 获取 cv 对象的 shuffle 属性，如果不存在则默认为 True
    shuffle = getattr(cv, "shuffle", True)
    # 获取 cv 对象的 random_state 属性，如果不存在则默认为 0
    random_state = getattr(cv, "random_state", 0)
    
    # 返回条件：random_state 是整数或者 shuffle 不为 True
    return isinstance(random_state, numbers.Integral) or not shuffle
```