# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_sag.py`

```
# Solvers for Ridge and LogisticRegression using SAG algorithm

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 引入警告模块，用于处理警告信息
import warnings

# 引入 NumPy 库，用于数值计算
import numpy as np

# 引入自定义异常模块
from ..exceptions import ConvergenceWarning

# 引入数据验证和处理工具函数
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight

# 引入数据集生成函数
from ._base import make_dataset

# 引入特定位数的SAG算法实现
from ._sag_fast import sag32, sag64


def get_auto_step_size(
    max_squared_sum, alpha_scaled, loss, fit_intercept, n_samples=None, is_saga=False
):
    """Compute automatic step size for SAG solver.

    The step size is set to 1 / (alpha_scaled + L + fit_intercept) where L is
    the max sum of squares for over all samples.

    Parameters
    ----------
    max_squared_sum : float
        Maximum squared sum of X over samples.

    alpha_scaled : float
        Constant that multiplies the regularization term, scaled by
        1. / n_samples, the number of samples.

    loss : {'log', 'squared', 'multinomial'}
        The loss function used in SAG solver.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) will be
        added to the decision function.

    n_samples : int, default=None
        Number of rows in X. Useful if is_saga=True.

    is_saga : bool, default=False
        Whether to return step size for the SAGA algorithm or the SAG
        algorithm.

    Returns
    -------
    step_size : float
        Step size used in SAG solver.

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    "SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives" <1407.0202>
    """
    # 根据损失函数类型计算 L
    if loss in ("log", "multinomial"):
        L = 0.25 * (max_squared_sum + int(fit_intercept)) + alpha_scaled
    elif loss == "squared":
        # 对于平方损失函数，计算其逆李普希茨常数
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
    else:
        # 如果损失函数未知，则引发错误
        raise ValueError(
            "Unknown loss function for SAG solver, got %s instead of 'log' or 'squared'"
            % loss
        )
    
    # 根据算法类型选择步长大小
    if is_saga:
        # 对于 SAGA 算法，计算其理论步长
        # 参考 Defazio et al. 2014 的建议
        mun = min(2 * n_samples * alpha_scaled, L)
        step = 1.0 / (2 * L + mun)
    else:
        # 对于普通的 SAG 算法，计算其理论步长
        # 参考 Schmidt et al. 2013 的建议
        step = 1.0 / L
    
    # 返回计算得到的步长大小
    return step


def sag_solver(
    X,
    y,
    sample_weight=None,
    loss="log",
    alpha=1.0,
    beta=0.0,
    max_iter=1000,
    tol=0.001,
    verbose=0,
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    warm_start_mem=None,
    is_saga=False,
):
    """SAG solver for Ridge and LogisticRegression.

    SAG stands for Stochastic Average Gradient: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a constant learning rate.

    IMPORTANT NOTE: 'sag' solver converges faster on columns that are on the
    same scale. You can normalize the data by using
    sklearn.preprocessing.StandardScaler on your data before passing it to the
    fit method.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values for the features. It will
    fit the data according to squared loss or log loss.

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using the squared euclidean norm L2.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values. With loss='multinomial', y must be label encoded
        (see preprocessing.LabelEncoder).

    sample_weight : array-like of shape (n_samples,), default=None
        Weights applied to individual samples (1. for unweighted).

    loss : {'log', 'squared', 'multinomial'}, default='log'
        Loss function that will be optimized:
        -'log' is the binary logistic loss, as used in LogisticRegression.
        -'squared' is the squared loss, as used in Ridge.
        -'multinomial' is the multinomial logistic loss, as used in
         LogisticRegression.

        .. versionadded:: 0.18
           *loss='multinomial'*

    alpha : float, default=1.
        L2 regularization term in the objective function
        ``(0.5 * alpha * || W ||_F^2)``.

    beta : float, default=0.
        L1 regularization term in the objective function
        ``(beta * || W ||_1)``. Only applied if ``is_saga`` is set to True.

    max_iter : int, default=1000
        The max number of passes over the training data if the stopping
        criteria is not reached.

    tol : float, default=0.001
        The stopping criteria for the weights. The iterations will stop when
        max(change in weights) / max(weights) < tol.

    verbose : int, default=0
        The verbosity level.

    random_state : int, RandomState instance or None, default=None
        Used when shuffling the data. Pass an int for reproducible output
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. If None, it will be computed,
        going through all the samples. The value should be precomputed
        to speed up cross validation.
    """
    # 如果 warm_start_mem 参数为 None，则初始化为空字典
    if warm_start_mem is None:
        warm_start_mem = {}

    # 如果 max_iter 参数为 None，则设置默认值为 1000
    # Ridge 默认的 max_iter 参数是 None
    if max_iter is None:
        max_iter = 1000

    # 如果需要进行输入检查，则进行以下操作：
    if check_input:
        # 定义数据类型列表，接受 np.float64 和 np.float32 类型的数据
        _dtype = [np.float64, np.float32]
        # 检查并转换 X 到指定数据类型，接受稀疏矩阵格式，按行优先存储
        X = check_array(X, dtype=_dtype, accept_sparse="csr", order="C")
        # 检查并转换 y 到指定数据类型，确保为二维数组格式，按行优先存储
        y = check_array(y, dtype=_dtype, ensure_2d=False, order="C")

    # 获取样本数和特征数
    n_samples, n_features = X.shape[0], X.shape[1]

    # 根据样本数对 alpha 和 beta 进行缩放处理
    alpha_scaled = float(alpha) / n_samples
    beta_scaled = float(beta) / n_samples

    # 如果 loss 函数为 'multinomial'，则 y 应为标签编码后的数据
    # 根据损失函数类型确定类别数目
    n_classes = int(y.max()) + 1 if loss == "multinomial" else 1

    # 初始化：检查和调整样本权重，保证与输入数据类型一致
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    # 如果warm_start_mem中有"coef"，则使用它作为初始化的系数
    if "coef" in warm_start_mem.keys():
        coef_init = warm_start_mem["coef"]
    else:
        # 如果没有，则假设fit_intercept为False，初始化为零矩阵
        coef_init = np.zeros((n_features, n_classes), dtype=X.dtype, order="C")

    # coef_init可能包含截距项，根据fit_intercept的情况进行处理
    fit_intercept = coef_init.shape[0] == (n_features + 1)
    if fit_intercept:
        # 提取并移除coef_init中的截距项
        intercept_init = coef_init[-1, :]
        coef_init = coef_init[:-1, :]
    else:
        # 如果没有截距项，则初始化截距为零
        intercept_init = np.zeros(n_classes, dtype=X.dtype)

    # 如果warm_start_mem中有"intercept_sum_gradient"，则使用它作为初始化的截距梯度和
    if "intercept_sum_gradient" in warm_start_mem.keys():
        intercept_sum_gradient = warm_start_mem["intercept_sum_gradient"]
    else:
        # 否则初始化截距梯度和为零向量
        intercept_sum_gradient = np.zeros(n_classes, dtype=X.dtype)

    # 如果warm_start_mem中有"gradient_memory"，则使用它作为初始化的梯度内存
    if "gradient_memory" in warm_start_mem.keys():
        gradient_memory_init = warm_start_mem["gradient_memory"]
    else:
        # 否则初始化梯度内存为零矩阵
        gradient_memory_init = np.zeros(
            (n_samples, n_classes), dtype=X.dtype, order="C"
        )

    # 如果warm_start_mem中有"sum_gradient"，则使用它作为初始化的梯度总和
    if "sum_gradient" in warm_start_mem.keys():
        sum_gradient_init = warm_start_mem["sum_gradient"]
    else:
        # 否则初始化梯度总和为零矩阵
        sum_gradient_init = np.zeros((n_features, n_classes), dtype=X.dtype, order="C")

    # 如果warm_start_mem中有"seen"，则使用它作为初始化的样本观察标记
    if "seen" in warm_start_mem.keys():
        seen_init = warm_start_mem["seen"]
    else:
        # 否则初始化样本观察标记为零向量
        seen_init = np.zeros(n_samples, dtype=np.int32, order="C")

    # 如果warm_start_mem中有"num_seen"，则使用它作为初始化的样本观察次数
    if "num_seen" in warm_start_mem.keys():
        num_seen_init = warm_start_mem["num_seen"]
    else:
        # 否则初始化样本观察次数为零
        num_seen_init = 0

    # 创建数据集和截距衰减参数
    dataset, intercept_decay = make_dataset(X, y, sample_weight, random_state)

    # 如果max_squared_sum为None，则计算X的行范数的最大平方和
    if max_squared_sum is None:
        max_squared_sum = row_norms(X, squared=True).max()

    # 获取自动步长大小
    step_size = get_auto_step_size(
        max_squared_sum,
        alpha_scaled,
        loss,
        fit_intercept,
        n_samples=n_samples,
        is_saga=is_saga,
    )

    # 如果step_size * alpha_scaled等于1，则抛出ZeroDivisionError异常
    if step_size * alpha_scaled == 1:
        raise ZeroDivisionError(
            "Current sag implementation does not handle "
            "the case step_size * alpha_scaled == 1"
        )

    # 根据输入数据类型选择SAG算法的具体实现（64位或32位）
    sag = sag64 if X.dtype == np.float64 else sag32

    # 运行SAG优化算法，返回更新后的参数和迭代次数
    num_seen, n_iter_ = sag(
        dataset,
        coef_init,
        intercept_init,
        n_samples,
        n_features,
        n_classes,
        tol,
        max_iter,
        loss,
        step_size,
        alpha_scaled,
        beta_scaled,
        sum_gradient_init,
        gradient_memory_init,
        seen_init,
        num_seen_init,
        fit_intercept,
        intercept_sum_gradient,
        intercept_decay,
        is_saga,
        verbose,
    )

    # 如果迭代次数达到最大值，则发出警告，表明coef_未收敛
    if n_iter_ == max_iter:
        warnings.warn(
            "The max_iter was reached which means the coef_ did not converge",
            ConvergenceWarning,
        )
    # 如果需要拟合截距项，将截距初始化值添加到系数初始化值中
    if fit_intercept:
        coef_init = np.vstack((coef_init, intercept_init))

    # 构建用于存储模型参数和梯度信息的字典
    warm_start_mem = {
        "coef": coef_init,                          # 系数的初始值
        "sum_gradient": sum_gradient_init,          # 梯度总和的初始值
        "intercept_sum_gradient": intercept_sum_gradient,  # 截距梯度总和的初始值
        "gradient_memory": gradient_memory_init,    # 梯度记忆
        "seen": seen_init,                          # 记录已经观察到的样本
        "num_seen": num_seen,                       # 已观察到的样本数
    }

    # 如果损失函数为多项逻辑回归，则将系数初始化值转置后赋值给 coef_
    if loss == "multinomial":
        coef_ = coef_init.T
    else:
        coef_ = coef_init[:, 0]  # 否则直接取系数初始化值的第一列

    # 返回最终的模型参数 coef_，迭代次数 n_iter_，以及存储的参数和梯度信息 warm_start_mem
    return coef_, n_iter_, warm_start_mem
```