# `D:\src\scipysrc\scikit-learn\sklearn\svm\_bounds.py`

```
"""Determination of parameter bounds"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real  # 导入 Real 类型，用于定义实数类型的参数

import numpy as np  # 导入 NumPy 库，用于数值计算

from ..preprocessing import LabelBinarizer  # 导入标签二值化预处理器
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证相关函数和类
from ..utils.extmath import safe_sparse_dot  # 导入稀疏矩阵乘法函数
from ..utils.validation import check_array, check_consistent_length  # 导入数组验证和长度一致性检查函数


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],  # 参数 X 应为数组或稀疏矩阵
        "y": ["array-like"],  # 参数 y 应为数组
        "loss": [StrOptions({"squared_hinge", "log"})],  # 参数 loss 应为 "squared_hinge" 或 "log"
        "fit_intercept": ["boolean"],  # 参数 fit_intercept 应为布尔类型
        "intercept_scaling": [Interval(Real, 0, None, closed="neither")],  # 参数 intercept_scaling 应为大于0的实数
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def l1_min_c(X, y, *, loss="squared_hinge", fit_intercept=True, intercept_scaling=1.0):
    """Return the lowest bound for C.

    The lower bound for C is computed such that for C in (l1_min_C, infinity)
    the model is guaranteed not to be empty. This applies to l1 penalized
    classifiers, such as LinearSVC with penalty='l1' and
    linear_model.LogisticRegression with penalty='l1'.

    This value is valid if class_weight parameter in fit() is not set.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X.

    loss : {'squared_hinge', 'log'}, default='squared_hinge'
        Specifies the loss function.
        With 'squared_hinge' it is the squared hinge loss (a.k.a. L2 loss).
        With 'log' it is the loss of logistic regression models.

    fit_intercept : bool, default=True
        Specifies if the intercept should be fitted by the model.
        It must match the fit() method parameter.

    intercept_scaling : float, default=1.0
        When fit_intercept is True, instance vector x becomes
        [x, intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        It must match the fit() method parameter.

    Returns
    -------
    l1_min_c : float
        Minimum value for C.

    Examples
    --------
    >>> from sklearn.svm import l1_min_c
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> print(f"{l1_min_c(X, y, loss='squared_hinge', fit_intercept=True):.4f}")
    0.0044
    """

    X = check_array(X, accept_sparse="csc")  # 验证并转换输入的特征数据 X
    check_consistent_length(X, y)  # 检查 X 和 y 的长度是否一致

    Y = LabelBinarizer(neg_label=-1).fit_transform(y).T  # 对目标标签 y 进行二值化处理，并转置
    # 计算 Y 和 X 的稀疏矩阵乘积后的最大绝对值
    den = np.max(np.abs(safe_sparse_dot(Y, X)))
    # 如果需要拟合截距项
    if fit_intercept:
        # 创建一个形状为 (y 的大小, 1) 的数组，填充值为 intercept_scaling，数据类型与 intercept_scaling 的数组类型相同
        bias = np.full(
            (np.size(y), 1), intercept_scaling, dtype=np.array(intercept_scaling).dtype
        )
        # 计算 Y 与 bias 的点乘结果的绝对值的最大值，并与当前 den 的值取较大者
        den = max(den, abs(np.dot(Y, bias)).max())

    # 如果 den 的值为 0，则抛出数值错误异常
    if den == 0.0:
        raise ValueError(
            "Ill-posed l1_min_c calculation: l1 will always "
            "select zero coefficients for this data"
        )
    
    # 如果损失函数为 "squared_hinge"，返回 0.5 / den
    if loss == "squared_hinge":
        return 0.5 / den
    else:  # 如果损失函数为 "log"，返回 2.0 / den
        return 2.0 / den
```