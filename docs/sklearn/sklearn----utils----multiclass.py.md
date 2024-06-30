# `D:\src\scipysrc\scikit-learn\sklearn\utils\multiclass.py`

```
"""Utilities to handle multiclass/multioutput target in classifiers."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings  # 导入警告模块
from collections.abc import Sequence  # 从标准库的collections.abc模块导入Sequence类
from itertools import chain  # 从标准库的itertools模块导入chain函数

import numpy as np  # 导入NumPy库，重命名为np
from scipy.sparse import issparse  # 从SciPy稀疏矩阵库导入issparse函数

from ..utils._array_api import get_namespace  # 从当前包的utils._array_api模块导入get_namespace函数
from ..utils.fixes import VisibleDeprecationWarning  # 从当前包的utils.fixes模块导入VisibleDeprecationWarning警告类
from .validation import _assert_all_finite, check_array  # 从当前包的validation模块导入_assert_all_finite函数和check_array函数


def _unique_multiclass(y):
    xp, is_array_api_compliant = get_namespace(y)  # 获取命名空间和是否符合数组API规范的信息
    if hasattr(y, "__array__") or is_array_api_compliant:
        return xp.unique_values(xp.asarray(y))  # 返回数组中的唯一值
    else:
        return set(y)  # 返回集合中的唯一元素


def _unique_indicator(y):
    xp, _ = get_namespace(y)  # 获取命名空间
    return xp.arange(
        check_array(y, input_name="y", accept_sparse=["csr", "csc", "coo"]).shape[1]
    )  # 返回一个从0开始到数组y列数-1的数组


_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,  # 标签类型为二元，使用_unique_multiclass函数
    "multiclass": _unique_multiclass,  # 标签类型为多类，使用_unique_multiclass函数
    "multilabel-indicator": _unique_indicator,  # 标签类型为多标签指示器，使用_unique_indicator函数
}


def unique_labels(*ys):
    """Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes
        Label values.

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    xp, is_array_api_compliant = get_namespace(*ys)  # 获取命名空间和是否符合数组API规范的信息
    if not ys:
        raise ValueError("No argument has been passed.")  # 如果没有传入参数，则引发值错误

    # Check that we don't mix label format
    ys_types = set(type_of_target(x) for x in ys)  # 获取所有参数ys的类型集合
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)  # 如果标签类型混合，引发值错误

    label_type = ys_types.pop()  # 弹出集合中的唯一标签类型

    # Check consistency for the indicator format
    if (
        label_type == "multilabel-indicator"
        and len(
            set(
                check_array(y, accept_sparse=["csr", "csc", "coo"]).shape[1] for y in ys
            )
        )
        > 1
    ):
        raise ValueError(
            "Multi-label binary indicator input with different numbers of labels"
        )  # 如果多标签二元指示器的输入标签数不一致，则引发值错误

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)  # 根据标签类型获取对应的唯一标签函数
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))  # 如果标签类型未知，则引发值错误
    if is_array_api_compliant:
        # 如果符合数组 API 要求
        # array_api 不允许混合数据类型
        unique_ys = xp.concat([_unique_labels(y) for y in ys])
        # 返回所有唯一的值
        return xp.unique_values(unique_ys)

    ys_labels = set(chain.from_iterable((i for i in _unique_labels(y)) for y in ys))
    # 获取所有标签的集合
    # 检查我们不混合字符串类型和数值类型
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        # 如果有混合的标签输入类型（字符串和数值），则抛出 ValueError
        raise ValueError("Mix of label input types (string and number)")

    # 将标签集合排序并转换为数组返回
    return xp.asarray(sorted(ys_labels))
# 检查输入的数组或序列是否是整数类型的浮点数
def _is_integral_float(y):
    # 获取数组库的命名空间和数组是否符合API规范的标志
    xp, is_array_api_compliant = get_namespace(y)
    # 返回是否是实数浮点型并且通过整型转换后等于原始数组的布尔值
    return xp.isdtype(y.dtype, "real floating") and bool(
        xp.all(xp.astype((xp.astype(y, xp.int64)), y.dtype) == y)
    )


def is_multilabel(y):
    """检查 ``y`` 是否处于多标签格式。

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        目标值。

    Returns
    -------
    out : bool
        如果 ``y`` 处于多标签格式，则返回 ``True``，否则返回 ``False``。

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    # 获取数组库的命名空间和数组是否符合API规范的标志
    xp, is_array_api_compliant = get_namespace(y)
    if hasattr(y, "__array__") or isinstance(y, Sequence) or is_array_api_compliant:
        # 将数组作为输入的参数进行检查，接受稀疏矩阵，不强制所有有限，不确保2D，最小样本和特征数量都可以为0
        check_y_kwargs = dict(
            accept_sparse=True,
            allow_nd=True,
            force_all_finite=False,
            ensure_2d=False,
            ensure_min_samples=0,
            ensure_min_features=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            try:
                # 尝试使用给定的参数检查数组
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (VisibleDeprecationWarning, ValueError) as e:
                if str(e).startswith("Complex data not supported"):
                    raise

                # 对于不规则数组，应显式提供 dtype=object，参见 NEP 34
                y = check_array(y, dtype=object, **check_y_kwargs)

    # 如果不是二维且不是多标签格式，则返回 False
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    # 如果是稀疏矩阵
    if issparse(y):
        if y.format in ("dok", "lil"):
            y = y.tocsr()
        # 获取唯一的标签值
        labels = xp.unique_values(y.data)
        # 返回条件判断结果，判断是否是空数据，标签数为1或者标签数为2且包含0，并且数据类型为布尔型、整数型或无符号整数型
        return (
            len(y.data) == 0
            or (labels.size == 1 or (labels.size == 2) and (0 in labels))
            and (y.dtype.kind in "biu" or _is_integral_float(labels))  # bool, int, uint
        )
    else:
        # 获取唯一的标签值
        labels = xp.unique_values(y)

        # 返回条件判断结果，判断标签数是否小于3且数据类型为布尔型、有符号整数型或无符号整数型，或者标签是否为整数型浮点数
        return labels.shape[0] < 3 and (
            xp.isdtype(y.dtype, ("bool", "signed integer", "unsigned integer"))
            or _is_integral_float(labels)
        )


def check_classification_targets(y):
    """确保目标 ``y`` 是非回归类型。

    只允许以下目标类型（如 type_of_target 中定义的）：
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        目标值。
    """
    # 使用 type_of_target 函数获取目标值的类型，并将其命名为 y_type
    y_type = type_of_target(y, input_name="y")
    # 如果 y_type 不在以下几种类型中：
    # "binary", "multiclass", "multiclass-multioutput",
    # "multilabel-indicator", "multilabel-sequences"
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        # 抛出 ValueError 异常，指示未知的标签类型，并提供可能的解释
        raise ValueError(
            f"Unknown label type: {y_type}. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a "
            "regression target with continuous values."
        )
# 确定目标数据的类型，返回最具体的类型信息。
def type_of_target(y, input_name=""):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : {array-like, sparse matrix}
        Target values. If a sparse matrix, `y` is expected to be a
        CSR/CSC matrix.

    input_name : str, default=""
        The data name used to construct the error message.

        .. versionadded:: 1.1.0

    Returns
    -------
    target_type : str
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    # 获取目标数据的命名空间和其是否符合数组API的要求
    xp, is_array_api_compliant = get_namespace(y)
    # 验证目标数据是否为序列、稀疏矩阵或具有__array__属性的非字符串数组，同时不是字符串，或符合数组API的要求
    valid = (
        (isinstance(y, Sequence) or issparse(y) or hasattr(y, "__array__"))
        and not isinstance(y, str)
        or is_array_api_compliant
    )

    # 如果验证不通过，则抛出值错误异常，显示期望的数据类型
    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), got %r" % y
        )
    # 检查 y 是否是 SparseSeries 或 SparseArray 类型
    sparse_pandas = y.__class__.__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:
        # 如果是 SparseSeries 或 SparseArray 类型，则抛出数值错误
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    # 检查 y 是否为多标签数据集
    if is_multilabel(y):
        # 如果是多标签数据集，返回字符串 "multilabel-indicator"
        return "multilabel-indicator"

    # 以下代码段捕获 DeprecationWarning 和 ValueError 异常，用于处理 NumPy < 1.24 和 >= 1.24 版本的情况
    # 参考 NEP 34：https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    check_y_kwargs = dict(
        accept_sparse=True,
        allow_nd=True,
        force_all_finite=False,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_min_features=0,
    )

    # 使用警告捕获器捕获 VisibleDeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", VisibleDeprecationWarning)
        
        # 如果 y 不是稀疏矩阵，则尝试对 y 进行检查和转换
        if not issparse(y):
            try:
                # 尝试检查数组 y，不指定数据类型，使用预定义的检查参数
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (VisibleDeprecationWarning, ValueError) as e:
                # 处理特定异常信息，如 "Complex data not supported"
                if str(e).startswith("Complex data not supported"):
                    raise

                # 对于异常，要求明确指定 dtype=object，用于处理不规则数组情况
                y = check_array(y, dtype=object, **check_y_kwargs)

    try:
        # TODO(1.7): 当 byte 格式标签被废弃后，修改为 ValueError
        # 检查第一行或值是否是 bytes 类型
        first_row_or_val = y[[0], :] if issparse(y) else y[0]
        if isinstance(first_row_or_val, bytes):
            # 发出警告，标签以字节格式表示已过时，将在 v1.7 中报错
            warnings.warn(
                (
                    "Support for labels represented as bytes is deprecated in v1.5 and"
                    " will error in v1.7. Convert the labels to a string or integer"
                    " format."
                ),
                FutureWarning,
            )

        # 检查是否为旧的序列格式
        if (
            not hasattr(first_row_or_val, "__array__")
            and isinstance(first_row_or_val, Sequence)
            and not isinstance(first_row_or_val, str)
        ):
            # 抛出值错误，不再支持旧的多标签数据表示形式
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # 检查输入是否有效
    if y.ndim not in (1, 2):
        # 如果数组维度大于2，返回 "unknown"
        return "unknown"
    if not min(y.shape):
        # 如果数组形状最小值为0
        if y.ndim == 1:
            # 如果是一维空数组，返回 "binary"
            return "binary"  # []
        # 如果是二维空数组，返回 "unknown"
        return "unknown"
    if not issparse(y) and y.dtype == object and not isinstance(y.flat[0], str):
        # 如果 y 不是稀疏矩阵，数据类型为 object，并且第一个元素不是字符串类型
        # 返回 "unknown"
        return "unknown"

    # 检查是否为多输出
    # 如果 y 的维度是二维且第二维度大于1
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # 设置后缀为 "-multioutput"，表示多输出情况
    else:
        suffix = ""  # 否则后缀为空字符串，表示单输出情况

    # 检查是否为浮点数并且包含非整数浮点数值
    if xp.isdtype(y.dtype, "real floating"):
        # 如果 y 是稀疏矩阵，则使用其数据部分，否则直接使用 y
        data = y.data if issparse(y) else y
        # 如果数据中存在不等于其整数类型转换后的值，则进行有限性检查
        if xp.any(data != xp.astype(data, int)):
            _assert_all_finite(data, input_name=input_name)
            return "continuous" + suffix  # 返回 "continuous" 加上相应的后缀

    # 检查是否为多分类问题
    if issparse(first_row_or_val):
        first_row_or_val = first_row_or_val.data
    # 如果 y 的唯一值数量大于2，或者 y 是二维且第一行或第一个值的长度大于1
    if xp.unique_values(y).shape[0] > 2 or (y.ndim == 2 and len(first_row_or_val) > 1):
        return "multiclass" + suffix  # 返回 "multiclass" 加上相应的后缀
    else:
        return "binary"  # 否则返回 "binary"，表示二分类问题
# 检查是否是 clf 的首次调用 partial_fit 函数的私有辅助函数
def _check_partial_fit_first_call(clf, classes=None):
    """Private helper function for factorizing common classes param logic.

    Estimators that implement the ``partial_fit`` API need to be provided with
    the list of possible classes at the first call to partial_fit.

    Subsequent calls to partial_fit should check that ``classes`` is still
    consistent with a previous value of ``clf.classes_`` when provided.

    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.
    """
    # 如果 clf 没有已经设置的 classes_ 属性，并且在首次调用 partial_fit 时未传入 classes，则抛出异常
    if getattr(clf, "classes_", None) is None and classes is None:
        raise ValueError("classes must be passed on the first call to partial_fit.")

    # 如果 classes 不为 None，则进行以下逻辑
    elif classes is not None:
        # 如果 clf 已经有设置 classes_ 属性，并且传入的 classes 与 clf.classes_ 不一致，则抛出异常
        if getattr(clf, "classes_", None) is not None:
            if not np.array_equal(clf.classes_, unique_labels(classes)):
                raise ValueError(
                    "`classes=%r` is not the same as on last call "
                    "to partial_fit, was: %r" % (classes, clf.classes_)
                )
        # 如果 clf 首次调用 partial_fit，则设置 clf.classes_ 属性为唯一的 classes 值，并返回 True
        else:
            clf.classes_ = unique_labels(classes)
            return True

    # 当 classes 为 None 且 clf.classes_ 已经被设置过时，返回 False，无需执行其他操作
    return False


def class_distribution(y, sample_weight=None):
    """Compute class priors from multioutput-multiclass target data.

    Parameters
    ----------
    y : {array-like, sparse matrix} of size (n_samples, n_outputs)
        The labels for each example.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    classes : list of size n_outputs of ndarray of size (n_classes,)
        List of classes for each column.

    n_classes : list of int of size n_outputs
        Number of classes in each column.

    class_prior : list of size n_outputs of ndarray of size (n_classes,)
        Class distribution of each column.
    """
    classes = []      # 存储每列的类别
    n_classes = []    # 存储每列中的类别数量
    class_prior = []  # 存储每列的类别分布

    n_samples, n_outputs = y.shape  # 获取样本数和输出列数

    # 如果存在 sample_weight，则转换为 numpy 数组
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
    # 如果 y 是稀疏矩阵
    if issparse(y):
        # 将稀疏矩阵转换为 CSC 格式以便于按列访问
        y = y.tocsc()
        # 计算每列中非零元素的数量
        y_nnz = np.diff(y.indptr)

        # 遍历输出的个数（通常是类别的数量）
        for k in range(n_outputs):
            # 获取第 k 列中非零元素的索引
            col_nonzero = y.indices[y.indptr[k] : y.indptr[k + 1]]

            # 如果指定了样本权重
            if sample_weight is not None:
                # 获取非零元素的样本权重
                nz_samp_weight = sample_weight[col_nonzero]
                # 计算零元素的样本权重总和
                zeros_samp_weight_sum = np.sum(sample_weight) - np.sum(nz_samp_weight)
            else:
                # 如果没有指定样本权重，则置为 None
                nz_samp_weight = None
                # 计算零元素的样本权重总和，即样本总数减去非零元素个数
                zeros_samp_weight_sum = y.shape[0] - y_nnz[k]

            # 获取第 k 列中不重复的类别和类别的索引
            classes_k, y_k = np.unique(
                y.data[y.indptr[k] : y.indptr[k + 1]], return_inverse=True
            )
            # 计算每个类别的先验概率，考虑非零元素的权重
            class_prior_k = np.bincount(y_k, weights=nz_samp_weight)

            # 如果类别中包含显式的零元素，则将其权重与隐式零元素的权重合并
            if 0 in classes_k:
                class_prior_k[classes_k == 0] += zeros_samp_weight_sum

            # 如果类别中不包含零元素，并且存在隐式的零元素，则添加一个条目
            if 0 not in classes_k and y_nnz[k] < y.shape[0]:
                classes_k = np.insert(classes_k, 0, 0)
                class_prior_k = np.insert(class_prior_k, 0, zeros_samp_weight_sum)

            # 将当前输出的类别、类别数和先验概率添加到对应的列表中
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])
            class_prior.append(class_prior_k / class_prior_k.sum())
    else:
        # 如果 y 不是稀疏矩阵，处理稠密矩阵的情况
        for k in range(n_outputs):
            # 获取第 k 列中不重复的类别和类别的索引
            classes_k, y_k = np.unique(y[:, k], return_inverse=True)
            # 将当前输出的类别添加到列表中
            classes.append(classes_k)
            # 记录当前输出的类别数量
            n_classes.append(classes_k.shape[0])
            # 计算每个类别的先验概率，考虑样本权重
            class_prior_k = np.bincount(y_k, weights=sample_weight)
            # 将计算得到的先验概率归一化并添加到列表中
            class_prior.append(class_prior_k / class_prior_k.sum())

    # 返回所有输出的类别、类别数和归一化后的先验概率
    return (classes, n_classes, class_prior)
def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking OvR decision function from OvO.

    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.

    confidences : array-like of shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.

    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``.
    """
    # 获取样本数量
    n_samples = predictions.shape[0]
    # 初始化一个用于计数投票的零数组
    votes = np.zeros((n_samples, n_classes))
    # 初始化一个用于累加置信度的零数组
    sum_of_confidences = np.zeros((n_samples, n_classes))

    # 循环遍历所有可能的类别对
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # 更新累加置信度，按照 OvR 策略
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            # 更新投票数，根据预测结果来确定投票给哪个类别
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    # 对 sum_of_confidences 进行单调变换，将其映射到 (-1/3, 1/3) 区间
    # 并与投票数相加。单调变换函数为 f: x -> x / (3 * (|x| + 1))，
    # 使用 1/3 而非 1/2 是为了确保不会触及边界值并改变投票顺序。
    # 这样做的动机是利用置信度水平来打破投票平局，而不改变基于一票差异做出的决策。
    transformed_confidences = sum_of_confidences / (
        3 * (np.abs(sum_of_confidences) + 1)
    )
    # 返回最终的投票数加上转换后的置信度
    return votes + transformed_confidences
```