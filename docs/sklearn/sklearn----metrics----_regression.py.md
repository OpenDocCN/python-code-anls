# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_regression.py`

```
"""
用于评估回归任务性能的指标。

以 ``*_score`` 命名的函数返回一个标量值以最大化：值越高越好。

以 ``*_error`` 或 ``*_loss`` 命名的函数返回一个标量值以最小化：值越低越好。
"""

# 作者：scikit-learn 开发人员
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入 Real 类型
from numbers import Real

# 导入 numpy 库
import numpy as np
# 导入 scipy 库中的 xlogy 函数
from scipy.special import xlogy

# 导入异常处理模块
from ..exceptions import UndefinedMetricWarning
# 导入数组操作相关函数
from ..utils._array_api import (
    _average,
    _find_matching_floating_dtype,
    get_namespace,
    get_namespace_and_device,
    size,
)
# 导入参数验证相关函数
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
# 导入统计相关函数
from ..utils.stats import _weighted_percentile
# 导入验证相关函数
from ..utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    column_or_1d,
)

# 定义导出的函数列表
__ALL__ = [
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "mean_absolute_percentage_error",
    "mean_pinball_loss",
    "r2_score",
    "root_mean_squared_log_error",
    "root_mean_squared_error",
    "explained_variance_score",
    "mean_tweedie_deviance",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "d2_tweedie_score",
    "d2_pinball_score",
    "d2_absolute_error_score",
]

# 检查回归任务的目标值是否属于同一任务
def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric", xp=None):
    """
    检查 y_true 和 y_pred 是否属于同一回归任务。

    参数
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like 或字符串 ['raw_values', uniform_average', 'variance_weighted'] 或 None
        由于 r2_score() 的向后兼容性，接受 None。

    dtype : str 或列表，默认为 "numeric"
        传递给 check_array 的 dtype 参数。

    返回
    -------
    type_true : {'continuous', continuous-multioutput'} 中的一个
        真实目标数据的类型，由 'utils.multiclass.type_of_target' 输出。

    y_true : 形状为 (n_samples, n_outputs) 的 array-like
        真实目标值。

    y_pred : 形状为 (n_samples, n_outputs) 的 array-like
        预测目标值。

    multioutput : 形状为 (n_outputs) 的 array-like 或字符串 ['raw_values',
        uniform_average', 'variance_weighted'] 或 None
        如果 ``multioutput`` 是 array-like，则为自定义输出权重，
        如果 ``multioutput`` 是正确的关键字，则为相应的参数。
    """
    # 获取命名空间
    xp, _ = get_namespace(y_true, y_pred, multioutput, xp=xp)

    # 检查 y_true 和 y_pred 的长度是否一致
    check_consistent_length(y_true, y_pred)
    # 检查 y_true 和 y_pred 是否为数组，并指定 dtype
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    # 如果 y_true 是一维数组，则将其重塑为二维数组
    if y_true.ndim == 1:
        y_true = xp.reshape(y_true, (-1, 1))

    # 如果 y_pred 是一维数组，则将其重塑为二维数组
    if y_pred.ndim == 1:
        y_pred = xp.reshape(y_pred, (-1, 1))
    # 检查 y_true 和 y_pred 的列数是否相同，如果不同则抛出数值错误异常
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output ({0}!={1})".format(
                y_true.shape[1], y_pred.shape[1]
            )
        )

    # 获取 y_true 的输出列数
    n_outputs = y_true.shape[1]

    # 定义允许的 multioutput 字符串值
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")

    # 如果 multioutput 是字符串类型，则检查其是否在允许的字符串值中，否则抛出数值错误异常
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    # 如果 multioutput 不为 None，则检查其是否为合法的数组（非二维），并根据输出列数进行进一步验证
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != multioutput.shape[0]:
            raise ValueError(
                "There must be equally many custom weights "
                f"({multioutput.shape[0]}) as outputs ({n_outputs})."
            )

    # 根据输出列数确定 y_type 类型，是单输出还是多输出
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    # 返回结果：y_type 类型、y_true、y_pred 和 multioutput
    return y_type, y_true, y_pred, multioutput
# 使用装饰器validate_params对mean_absolute_error函数进行参数验证，确保输入参数类型和取值符合预期
@validate_params(
    {
        "y_true": ["array-like"],  # y_true参数应为类数组对象
        "y_pred": ["array-like"],  # y_pred参数应为类数组对象
        "sample_weight": ["array-like", None],  # sample_weight参数可以是类数组对象或None
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput参数可选项为'raw_values'或'uniform_average'，或者是类数组对象
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
# 定义计算均值绝对误差（MAE）的函数
def mean_absolute_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """Mean absolute error regression loss.

    Read more in the :ref:`User Guide <mean_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or array of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        MAE output is non-negative floating point. The best value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_error(y_true, y_pred)
    0.75
    >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85...
    """
    # 将输入参数组成列表
    input_arrays = [y_true, y_pred, sample_weight, multioutput]
    # 获取命名空间xp和未使用的变量_
    xp, _ = get_namespace(*input_arrays)

    # 确定匹配的浮点数数据类型
    dtype = _find_matching_floating_dtype(y_true, y_pred, sample_weight, xp=xp)

    # 检查并返回处理后的目标值、预测值和multioutput参数
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )

    # 检查y_true、y_pred和sample_weight的长度是否一致
    check_consistent_length(y_true, y_pred, sample_weight)

    # 计算输出误差，使用平均值函数_average计算绝对值误差
    output_errors = _average(
        xp.abs(y_pred - y_true), weights=sample_weight, axis=0, xp=xp
    )

    # 如果multioutput参数是字符串类型
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors  # 返回每个输出的绝对误差值

        elif multioutput == "uniform_average":
            # 作为np.average的权重传入None，表示均匀平均
            multioutput = None

    # 如果需要，对输出进行平均处理
    # 计算平均绝对误差，调用 _average 函数
    mean_absolute_error = _average(output_errors, weights=multioutput)

    # 由于 `y_pred.ndim <= 2` 和 `y_true.ndim <= 2`，第二次调用 _average 函数应始终返回一个标量数组，
    # 我们将其转换为 Python 浮点数，以确保始终返回相同的急切评估值，不受数组 API 实现的影响。
    # 断言确保平均绝对误差的形状是一个空元组，即标量值。
    assert mean_absolute_error.shape == ()

    # 返回平均绝对误差的 Python 浮点数值
    return float(mean_absolute_error)
# 使用装饰器验证函数参数，确保参数的类型和取值符合指定的要求
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数必须是类数组的数据类型
        "y_pred": ["array-like"],  # y_pred 参数必须是类数组的数据类型
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组的数据类型或者 None
        "alpha": [Interval(Real, 0, 1, closed="both")],  # alpha 参数必须是介于 [0, 1] 闭区间内的实数
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput 参数可以是预设字符串集合中的值或者类数组的数据类型
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
# 定义一个计算 Pinball 损失函数的函数，用于分位数回归
def mean_pinball_loss(
    y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"
):
    """Pinball loss for quantile regression.

    Read more in the :ref:`User Guide <pinball_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    alpha : float, slope of the pinball loss, default=0.5,
        This loss is equivalent to :ref:`mean_absolute_error` when `alpha=0.5`,
        `alpha=0.95` is minimized by estimators of the 95th percentile.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        The pinball loss output is a non-negative floating point. The best
        value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_pinball_loss
    >>> y_true = [1, 2, 3]
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    0.03...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.1)
    0.3...
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.9)
    0.3...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.9)
    0.03...
    >>> mean_pinball_loss(y_true, y_true, alpha=0.1)
    0.0
    >>> mean_pinball_loss(y_true, y_true, alpha=0.9)
    0.0
    """
    # 检查目标值的类型和形状，并确保适当的多输出格式
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 检查目标值、预测值和样本权重的长度一致性
    check_consistent_length(y_true, y_pred, sample_weight)
    # 计算预测值与目标值的差异
    diff = y_true - y_pred
    # 根据差异的正负情况确定符号
    sign = (diff >= 0).astype(diff.dtype)
    # 计算 Pinball 损失函数
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    # 根据样本权重计算加权平均的损失
    output_errors = np.average(loss, weights=sample_weight, axis=0)

    # 如果 multioutput 参数为 'raw_values'，则返回每个输出单独的均值绝对误差
    if isinstance(multioutput, str) and multioutput == "raw_values":
        return output_errors
    # 检查 multioutput 是否是字符串类型并且其值为 "uniform_average"
    # 如果是，则将 multioutput 设置为 None，表示在 np.average 中使用均匀平均的权重
    if isinstance(multioutput, str) and multioutput == "uniform_average":
        multioutput = None
    
    # 返回 output_errors 的加权平均值
    return np.average(output_errors, weights=multioutput)
# 使用 @validate_params 装饰器对下面定义的函数进行参数验证，确保参数的类型和取值符合要求
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数类型应为 array-like（类数组）
        "y_pred": ["array-like"],  # y_pred 参数类型应为 array-like（类数组）
        "sample_weight": ["array-like", None],  # sample_weight 参数类型应为 array-like 或者 None
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput 参数类型应为 'raw_values' 或 'uniform_average' 或者 array-like
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
def mean_absolute_percentage_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """Mean absolute percentage error (MAPE) regression loss.

    Note here that the output is not a percentage in the range [0, 100]
    and a value of 100 does not mean 100% but 1e2. Furthermore, the output
    can be arbitrarily high when `y_true` is small (which is specific to the
    metric) or when `abs(y_true - y_pred)` is large (which is common for most
    regression metrics). Read more in the
    :ref:`User Guide <mean_absolute_percentage_error>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.

        MAPE output is non-negative floating point. The best value is 0.0.
        But note that bad predictions can lead to arbitrarily large
        MAPE values, especially if some `y_true` values are very close to zero.
        Note that we return a large value instead of `inf` when `y_true` is zero.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3273...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5515...
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.6198...
    >>> # the value when some element of the y_true is zero is arbitrarily high because
    >>> # of the division by epsilon
    >>> y_true = [1., 0., 2.4, 7.]
    >>> y_pred = [1.2, 0.1, 2.4, 8.]
    # 计算平均绝对百分比误差（MAPE）并返回结果
    >>> mean_absolute_percentage_error(y_true, y_pred)
    # 检查目标数据的类型，调整格式并返回相关信息
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 检查目标数据和预测数据的长度是否一致，以及样本权重是否合理
    check_consistent_length(y_true, y_pred, sample_weight)
    # 设置一个极小值，防止除数为零
    epsilon = np.finfo(np.float64).eps
    # 计算每个样本的绝对百分比误差
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    # 计算加权平均的绝对百分比误差，作为输出误差
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    # 如果multioutput是字符串类型
    if isinstance(multioutput, str):
        # 如果multioutput为"raw_values"，直接返回输出误差
        if multioutput == "raw_values":
            return output_errors
        # 如果multioutput为"uniform_average"，使用均匀权重（None）计算平均值
        elif multioutput == "uniform_average":
            # 将multioutput设为None，表示均匀平均
            multioutput = None

    # 返回加权平均的输出误差
    return np.average(output_errors, weights=multioutput)
# 使用装饰器验证参数有效性，并指定预期参数的类型和选项
@validate_params(
    {
        "y_true": ["array-like"],  # 真实目标值，可以是数组形式
        "y_pred": ["array-like"],  # 预测目标值，可以是数组形式
        "sample_weight": ["array-like", None],  # 样本权重，可以是数组形式或者为空
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # 多输出定义，可以是特定字符串或者数组形式
        "squared": [Hidden(StrOptions({"deprecated"})), "boolean"],  # 方形标志，可以是布尔值，但已废弃
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
# 均方误差函数定义，用于回归损失计算
def mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,  # 样本权重，默认为空
    multioutput="uniform_average",  # 多输出处理方式，默认统一平均
    squared="deprecated",  # 是否使用方形计算，默认已废弃
):
    """均方误差回归损失函数。

    更多详细信息请参阅 :ref:`用户指南 <mean_squared_error>`。

    参数
    ----------
    y_true : 形状为 (n_samples,) 或 (n_samples, n_outputs) 的 array-like
        真实目标值（正确的）。

    y_pred : 形状为 (n_samples,) 或 (n_samples, n_outputs) 的 array-like
        预测目标值。

    sample_weight : 形状为 (n_samples,) 的 array-like，默认为 None
        样本权重。

    multioutput : {'raw_values', 'uniform_average'} 或形状为 (n_outputs,) 的 array-like，默认为 'uniform_average'
        定义多个输出值的聚合方式。
        如果是数组形式，定义了用于平均错误的权重。

        'raw_values' :
            在多输出输入情况下返回完整的错误集合。

        'uniform_average' :
            所有输出的错误均匀加权平均。

    squared : bool，默认为 True
        如果为 True，则返回 MSE 值，如果为 False，则返回 RMSE 值。

        .. deprecated:: 1.4
           `squared` 在版本 1.4 中已废弃，并将在 1.6 中删除。
           请改用 :func:`~sklearn.metrics.root_mean_squared_error`
           计算均方根误差。

    返回
    -------
    loss : float 或浮点数数组
        非负浮点数值（最佳值为 0.0），或每个单独目标的浮点数值数组。

    示例
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)
    0.708...
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825...

    """
    # TODO(1.6): remove
    # 如果 squared 参数不是 'deprecated'，则发出警告，说明该参数已废弃
    if squared != "deprecated":
        warnings.warn(
            (
                "'squared' is deprecated in version 1.4 and "
                "will be removed in 1.6. To calculate the "
                "root mean squared error, use the function"
                "'root_mean_squared_error'."
            ),
            FutureWarning,
        )
        # 如果 squared 不为真，则调用 root_mean_squared_error 函数计算 RMSE，并返回结果
        if not squared:
            return root_mean_squared_error(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )
    # 调用 get_namespace 函数获取预测和真实值的命名空间
    xp, _ = get_namespace(y_true, y_pred, sample_weight, multioutput)
    # 使用 _find_matching_floating_dtype 函数找到与 y_true 和 y_pred 匹配的浮点类型
    dtype = _find_matching_floating_dtype(y_true, y_pred, xp=xp)

    # 检查并返回经过处理后的目标值、真实值、预测值和多输出标志
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )
    # 检查真实值和预测值的长度是否一致，如果不一致则引发异常
    check_consistent_length(y_true, y_pred, sample_weight)
    # 计算平均误差，即 (y_true - y_pred) ** 2 的加权平均值
    output_errors = _average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)

    # 如果 multioutput 是字符串类型
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # 如果 multioutput 为 "raw_values"，直接返回 output_errors
            return output_errors
        elif multioutput == "uniform_average":
            # 如果 multioutput 为 "uniform_average"，将 multioutput 设置为 None，表示均匀平均
            # 将在 np.average 中将权重设置为 None：均匀平均
            multioutput = None

    # 计算均方误差，即 _average(output_errors, weights=multioutput) 的结果
    mean_squared_error = _average(output_errors, weights=multioutput)
    # 断言均方误差的形状应为一个标量
    assert mean_squared_error.shape == ()
    # 返回均方误差的浮点值
    return float(mean_squared_error)
@validate_params(
    {
        "y_true": ["array-like"],  # y_true参数应为类数组对象
        "y_pred": ["array-like"],  # y_pred参数应为类数组对象
        "sample_weight": ["array-like", None],  # sample_weight参数可以是类数组对象或None
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput参数可以是指定字符串集合或类数组对象
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def root_mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """Root mean squared error regression loss.

    Read more in the :ref:`User Guide <mean_squared_error>`.

    .. versionadded:: 1.4  # 版本1.4中新增的功能

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.  # 真实的目标值（正确值）

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.  # 预测的目标值

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.  # 样本权重

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.  # 定义多个输出值的聚合方式
        Array-like value defines weights used to average errors.  # 类数组值定义用于平均误差的权重。

        'raw_values' :
            Returns a full set of errors in case of multioutput input.  # 在多输出情况下返回完整的误差集合。

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.  # 所有输出的误差均使用均匀权重平均。

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.  # 非负浮点数值（最佳值为0.0），或者每个单独目标的浮点数数组。

    Examples
    --------
    >>> from sklearn.metrics import root_mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> root_mean_squared_error(y_true, y_pred)
    0.612...
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> root_mean_squared_error(y_true, y_pred)
    0.822...
    """
    output_errors = np.sqrt(
        mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values"
        )
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors  # 如果multioutput为'raw_values'，返回完整的误差集合
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)  # 返回加权平均的输出误差


@validate_params(
    {
        "y_true": ["array-like"],  # y_true参数应为类数组对象
        "y_pred": ["array-like"],  # y_pred参数应为类数组对象
        "sample_weight": ["array-like", None],  # sample_weight参数可以是类数组对象或None
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput参数可以是指定字符串集合或类数组对象
        "squared": [Hidden(StrOptions({"deprecated"})), "boolean"],  # squared参数为隐藏的布尔类型，已被废弃
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def mean_squared_log_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    squared="deprecated",
):
    """Mean squared logarithmic error regression loss.

    Read more in the :ref:`User Guide <mean_squared_log_error>`.

    Parameters
    ----------
    # y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
    #     Ground truth (correct) target values.
    #
    # y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
    #     Estimated target values.
    #
    # sample_weight : array-like of shape (n_samples,), default=None
    #     Sample weights.
    #
    # multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
    #         (n_outputs,), default='uniform_average'
    #
    #         Defines aggregating of multiple output values.
    #         Array-like value defines weights used to average errors.
    #
    #         'raw_values' :
    #             Returns a full set of errors when the input is of multioutput
    #             format.
    #
    #         'uniform_average' :
    #             Errors of all outputs are averaged with uniform weight.
    #
    # squared : bool, default=True
    #     If True returns MSLE (mean squared log error) value.
    #     If False returns RMSLE (root mean squared log error) value.
    #
    #     .. deprecated:: 1.4
    #        `squared` is deprecated in 1.4 and will be removed in 1.6.
    #        Use :func:`~sklearn.metrics.root_mean_squared_log_error`
    #        instead to calculate the root mean squared logarithmic error.
    #
    # Returns
    # -------
    # loss : float or ndarray of floats
    #     A non-negative floating point value (the best value is 0.0), or an
    #     array of floating point values, one for each individual target.
    #
    # Examples
    # --------
    # >>> from sklearn.metrics import mean_squared_log_error
    # >>> y_true = [3, 5, 2.5, 7]
    # >>> y_pred = [2.5, 5, 4, 8]
    # >>> mean_squared_log_error(y_true, y_pred)
    # 0.039...
    # >>> y_true = [[0.5, 1], [1, 2], [7, 6]]
    # >>> y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
    # >>> mean_squared_log_error(y_true, y_pred)
    # 0.044...
    # >>> mean_squared_log_error(y_true, y_pred, multioutput='raw_values')
    # array([0.00462428, 0.08377444])
    # >>> mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
    # 0.060...
    """
    # TODO(1.6): remove
    # 如果 squared 不是 "deprecated"，则发出警告
    if squared != "deprecated":
        warnings.warn(
            (
                "'squared' is deprecated in version 1.4 and "
                "will be removed in 1.6. To calculate the "
                "root mean squared logarithmic error, use the function "
                "'root_mean_squared_log_error'."
            ),
            FutureWarning,
        )
        # 如果 squared 是 False，则调用 root_mean_squared_log_error 函数
        if not squared:
            return root_mean_squared_log_error(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )

    # 检查目标 y_true 和 y_pred 的类型及多输出情况，并更新 multioutput
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 检查 y_true, y_pred, sample_weight 的长度是否一致
    check_consistent_length(y_true, y_pred, sample_weight)

    # 如果 y_true 或 y_pred 中有任何值为负数，则引发 ValueError
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError(
            "Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values."
        )
    # 计算均方误差（Mean Squared Error，MSE），衡量对数变换后的预测值与真实值之间的平均平方差
    return mean_squared_error(
        np.log1p(y_true),    # 对真实值应用对数变换后的结果
        np.log1p(y_pred),    # 对预测值应用对数变换后的结果
        sample_weight=sample_weight,   # 样本权重，用于加权计算误差
        multioutput=multioutput,       # 是否计算多输出的平均误差，如 'raw_values' 或 'uniform_average'
    )
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为 array-like 类型
        "y_pred": ["array-like"],  # y_pred 参数应为 array-like 类型
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是 array-like 类型或者 None
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput 参数可以是 'raw_values' 或 'uniform_average' 字符串集合中的一个，或者是 array-like 类型
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def root_mean_squared_log_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """Root mean squared logarithmic error regression loss.

    Read more in the :ref:`User Guide <mean_squared_log_error>`.

    .. versionadded:: 1.4

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'

        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import root_mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> root_mean_squared_log_error(y_true, y_pred)
    0.199...
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)  # 检查 y_true, y_pred 和 sample_weight 的长度是否一致

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError(
            "Root Mean Squared Logarithmic Error cannot be used when "
            "targets contain negative values."
        )

    return root_mean_squared_error(
        np.log1p(y_true),
        np.log1p(y_pred),
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为 array-like 类型
        "y_pred": ["array-like"],  # y_pred 参数应为 array-like 类型
        "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],  # multioutput 参数可以是 'raw_values' 或 'uniform_average' 字符串集合中的一个，或者是 array-like 类型
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是 array-like 类型或者 None
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def median_absolute_error(
    y_true, y_pred, *, multioutput="uniform_average", sample_weight=None
):
    """Median absolute error regression loss.

    Median absolute error output is non-negative floating point. The best value
    is 0.0. Read more in the :ref:`User Guide <median_absolute_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'

        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    """
    # 定义函数参数 y_true 和 y_pred，分别表示真实目标值和预测目标值
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        定义多输出值的聚合方式。如果是数组形式，则定义用于平均误差的权重。

        'raw_values' :
            如果输入是多输出，返回完整的误差集合。

        'uniform_average' :
            所有输出的误差均匀加权平均。

    sample_weight : array-like of shape (n_samples,), default=None
        样本权重。

        .. versionadded:: 0.24

    Returns
    -------
    loss : float or ndarray of floats
        如果 multioutput 是 'raw_values'，则分别返回每个输出的平均绝对误差。
        如果 multioutput 是 'uniform_average' 或权重数组，则返回所有输出误差的加权平均值。

    Examples
    --------
    >>> from sklearn.metrics import median_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> median_absolute_error(y_true, y_pred)
    0.75
    >>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    # 检查目标值类型，并进行必要的预处理
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 如果没有指定样本权重，则计算不同输出的中位数绝对误差
    if sample_weight is None:
        output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    else:
        # 检查并调整样本权重的形状以匹配预测目标值的形状
        sample_weight = _check_sample_weight(sample_weight, y_pred)
        # 使用加权百分位数函数计算加权误差
        output_errors = _weighted_percentile(
            np.abs(y_pred - y_true), sample_weight=sample_weight
        )
    # 如果 multioutput 是字符串类型
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # 返回每个输出的误差数组
            return output_errors
        elif multioutput == "uniform_average":
            # 将权重设置为 None，以便于 np.average 函数进行均匀平均
            multioutput = None

    # 返回加权平均误差值
    return np.average(output_errors, weights=multioutput)
# 用于计算解释方差分数和 R^2 分数的共同部分
def _assemble_r2_explained_variance(
    numerator, denominator, n_outputs, multioutput, force_finite, xp, device
):
    """Common part used by explained variance score and :math:`R^2` score."""
    # 确定 numerator 的数据类型
    dtype = numerator.dtype

    # 计算非零的 denominator 的布尔掩码
    nonzero_denominator = denominator != 0

    if not force_finite:
        # 标准公式，可能导致 NaN 或 -Inf
        output_scores = 1 - (numerator / denominator)
    else:
        # 计算非零的 numerator 的布尔掩码
        nonzero_numerator = numerator != 0

        # 如果 force_finite 为 True，则设定默认值为 1.0（完美预测的情况）
        output_scores = xp.ones([n_outputs], device=device, dtype=dtype)

        # 对于非零的 numerator 和 denominator，使用标准公式
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )

        # 对于非零的 numerator 但零的 denominator，设定为 0.0 避免 -Inf 分数
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # 返回单独的分数数组
            return output_scores
        elif multioutput == "uniform_average":
            # 将 None 作为权重传递给 np.average，结果是均匀平均
            avg_weights = None
        elif multioutput == "variance_weighted":
            # 使用 denominator 作为权重
            avg_weights = denominator
            if not xp.any(nonzero_denominator):
                # 如果所有权重都为零，np.average 将会抛出 ZeroDiv 错误
                # 这仅在所有 y 都是常数时发生（或者只有一个元素）
                # 因为权重都相等，回退到均匀权重
                avg_weights = None
    else:
        avg_weights = multioutput

    # 计算加权平均分数
    result = _average(output_scores, weights=avg_weights)
    if size(result) == 1:
        return float(result)
    return result


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
        "multioutput": [
            StrOptions({"raw_values", "uniform_average", "variance_weighted"}),
            "array-like",
        ],
        "force_finite": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def explained_variance_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
):
    """Explained variance regression score function.

    Best possible score is 1.0, lower values are worse.

    In the particular case when ``y_true`` is constant, the explained variance
    score is not finite: it is either ``NaN`` (perfect predictions) or
    ``-Inf`` (imperfect predictions). To prevent such non-finite numbers to
    pollute higher-level experiments such as a grid search cross-validation,
    by default these cases are replaced with 1.0 (perfect predictions) or 0.0
    """
        (imperfect predictions) respectively. If ``force_finite``
        is set to ``False``, this score falls back on the original :math:`R^2`
        definition.
    
        .. note::
           The Explained Variance score is similar to the
           :func:`R^2 score <r2_score>`, with the notable difference that it
           does not account for systematic offsets in the prediction. Most often
           the :func:`R^2 score <r2_score>` should be preferred.
    
        Read more in the :ref:`User Guide <explained_variance_score>`.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
    
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
    
        multioutput : {'raw_values', 'uniform_average', 'variance_weighted'} or \
                array-like of shape (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output scores.
            Array-like value defines weights used to average scores.
    
            'raw_values' :
                Returns a full set of scores in case of multioutput input.
    
            'uniform_average' :
                Scores of all outputs are averaged with uniform weight.
    
            'variance_weighted' :
                Scores of all outputs are averaged, weighted by the variances
                of each individual output.
    
        force_finite : bool, default=True
            Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
            data should be replaced with real numbers (``1.0`` if prediction is
            perfect, ``0.0`` otherwise). Default is ``True``, a convenient setting
            for hyperparameters' search procedures (e.g. grid search
            cross-validation).
    
            .. versionadded:: 1.1
    
        Returns
        -------
        score : float or ndarray of floats
            The explained variance or ndarray if 'multioutput' is 'raw_values'.
    
        See Also
        --------
        r2_score :
            Similar metric, but accounting for systematic offsets in
            prediction.
    
        Notes
        -----
        This is not a symmetric function.
    
        Examples
        --------
        >>> from sklearn.metrics import explained_variance_score
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> explained_variance_score(y_true, y_pred)
        0.957...
        >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
        >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
        >>> explained_variance_score(y_true, y_pred, multioutput='uniform_average')
        0.983...
        >>> y_true = [-2, -2, -2]
        >>> y_pred = [-2, -2, -2]
        >>> explained_variance_score(y_true, y_pred)
        1.0
        >>> explained_variance_score(y_true, y_pred, force_finite=False)
        nan
        >>> y_true = [-2, -2, -2]
        >>> y_pred = [-2, -2, -2 + 1e-8]
        >>> explained_variance_score(y_true, y_pred)
        0.0
        >>> explained_variance_score(y_true, y_pred, force_finite=False)
        -inf
    """
    # 检查回归目标的类型，确保与多输出模式兼容
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 检查真实值和预测值的长度是否一致，同时考虑样本权重
    check_consistent_length(y_true, y_pred, sample_weight)

    # 计算真实值与预测值的加权平均差异
    y_diff_avg = np.average(y_true - y_pred, weights=sample_weight, axis=0)
    # 计算解释方差的分子部分，考虑加权
    numerator = np.average(
        (y_true - y_pred - y_diff_avg) ** 2, weights=sample_weight, axis=0
    )

    # 计算真实值的加权平均值
    y_true_avg = np.average(y_true, weights=sample_weight, axis=0)
    # 计算解释方差的分母部分，考虑加权
    denominator = np.average((y_true - y_true_avg) ** 2, weights=sample_weight, axis=0)

    # 调用函数计算 R^2 和解释方差，并返回结果
    return _assemble_r2_explained_variance(
        numerator=numerator,
        denominator=denominator,
        n_outputs=y_true.shape[1],
        multioutput=multioutput,
        force_finite=force_finite,
        xp=get_namespace(y_true)[0],
        # TODO: 添加对 Array API 的支持后更新 explained_variance_score
        device=None,
    )
# 使用装饰器 validate_params 对 r2_score 函数的参数进行验证和类型检查
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应为类数组对象
        "y_pred": ["array-like"],  # y_pred 参数应为类数组对象
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组对象或 None
        "multioutput": [  # multioutput 参数可以是预定义字符串选项集合，类数组对象或 None
            StrOptions({"raw_values", "uniform_average", "variance_weighted"}),
            "array-like",
            None,
        ],
        "force_finite": ["boolean"],  # force_finite 参数应为布尔类型
    },
    prefer_skip_nested_validation=True,  # 设置 prefer_skip_nested_validation 参数为 True
)
def r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,  # sample_weight 参数，默认为 None
    multioutput="uniform_average",  # multioutput 参数，默认为 "uniform_average"
    force_finite=True,  # force_finite 参数，默认为 True
):
    """:math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    In the particular case when ``y_true`` is constant, the :math:`R^2` score
    is not finite: it is either ``NaN`` (perfect predictions) or ``-Inf``
    (imperfect predictions). To prevent such non-finite numbers to pollute
    higher-level experiments such as a grid search cross-validation, by default
    these cases are replaced with 1.0 (perfect predictions) or 0.0 (imperfect
    predictions) respectively. You can set ``force_finite`` to ``False`` to
    prevent this fix from happening.

    Note: when the prediction residuals have zero mean, the :math:`R^2` score
    is identical to the
    :func:`Explained Variance score <explained_variance_score>`.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'}, \
            array-like of shape (n_outputs,) or None, default='uniform_average'

        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".

        'raw_values' :
            Returns a full set of scores in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.
    # 调用函数获取命名空间和设备信息，用于后续计算
    xp, _, device_ = get_namespace_and_device(
        y_true, y_pred, sample_weight, multioutput
    )

    # 查找匹配的浮点数数据类型，用于处理真实值、预测值和样本权重
    dtype = _find_matching_floating_dtype(y_true, y_pred, sample_weight, xp=xp)

    # 检查和调整真实值和预测值的格式，确保能够进行回归评估
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput, dtype=dtype, xp=xp
    )

    # 检查真实值和预测值的长度是否一致，如果使用了样本权重，则也要检查其长度
    check_consistent_length(y_true, y_pred, sample_weight)

    # 如果样本数量小于2，则R^2分数无法定义，返回NaN
    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    # 如果存在样本权重，则将其转换为列向量；否则权重默认为1.0
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight, dtype=dtype)
        weight = sample_weight[:, None]
    else:
        weight = 1.0

    # 计算R^2分数的分子部分：加权平方误差的和
    numerator = xp.sum(weight * (y_true - y_pred) ** 2, axis=0)

    # 计算R^2分数的分母部分：加权真实值和加权平均真实值之间的差的平方和
    denominator = xp.sum(
        weight * (y_true - _average(y_true, axis=0, weights=sample_weight, xp=xp)) ** 2,
        axis=0,
    )
    # 调用函数 _assemble_r2_explained_variance，返回其计算的解释方差值
    return _assemble_r2_explained_variance(
        # 参数：分子（numerator），用于计算解释方差的值
        numerator=numerator,
        # 参数：分母（denominator），用于计算解释方差的值
        denominator=denominator,
        # 参数：输出的数量，即 y_true 的列数，用于多输出情况
        n_outputs=y_true.shape[1],
        # 参数：多输出模式的设置
        multioutput=multioutput,
        # 参数：是否强制使用有限值
        force_finite=force_finite,
        # 参数：用于计算的数值运算库，可能是 NumPy 等
        xp=xp,
        # 参数：指定的设备，例如 CPU 或 GPU
        device=device_,
    )
@validate_params(
    {
        "y_true": ["array-like"],  # 验证参数装饰器，确保y_true是一个类数组对象
        "y_pred": ["array-like"],  # 验证参数装饰器，确保y_pred是一个类数组对象
    },
    prefer_skip_nested_validation=True,
)
def max_error(y_true, y_pred):
    """
    The max_error metric calculates the maximum residual error.

    Read more in the :ref:`User Guide <max_error>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    max_error : float
        A positive floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import max_error
    >>> y_true = [3, 2, 7, 1]
    >>> y_pred = [4, 2, 7, 1]
    >>> max_error(y_true, y_pred)
    1
    """
    xp, _ = get_namespace(y_true, y_pred)  # 获取适当的数学运算库，并可能执行命名空间转换
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)  # 检查回归目标值的类型和数据
    if y_type == "continuous-multioutput":  # 如果目标类型是"continuous-multioutput"，则抛出异常
        raise ValueError("Multioutput not supported in max_error")
    return xp.max(xp.abs(y_true - y_pred))  # 返回预测值与真实值之差的绝对值的最大值


def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    """Mean Tweedie deviance regression loss."""
    xp, _ = get_namespace(y_true, y_pred)  # 获取适当的数学运算库，并可能执行命名空间转换
    p = power
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        dev = 2 * (
            xp.pow(xp.where(y_true > 0, y_true, 0), 2 - p) / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, 1 - p) / (1 - p)
            + xp.pow(y_pred, 2 - p) / (2 - p)
        )
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif p == 2:
        # Gamma distribution
        dev = 2 * (xp.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        dev = 2 * (
            xp.pow(y_true, 2 - p) / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, 1 - p) / (1 - p)
            + xp.pow(y_pred, 2 - p) / (2 - p)
        )
    return float(_average(dev, weights=sample_weight))  # 返回加权平均的Tweedie偏差


@validate_params(
    {
        "y_true": ["array-like"],  # 验证参数装饰器，确保y_true是一个类数组对象
        "y_pred": ["array-like"],  # 验证参数装饰器，确保y_pred是一个类数组对象
        "sample_weight": ["array-like", None],  # 验证参数装饰器，确保sample_weight是一个类数组对象或者None
        "power": [
            Interval(Real, None, 0, closed="right"),  # power参数的范围为(负无穷, 0]
            Interval(Real, 1, None, closed="left"),  # 或者范围为[1, 正无穷)
        ],
    },
    prefer_skip_nested_validation=True,
)
def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    """
    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.

        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.

        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to
          mean_squared_error. y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_tweedie_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_tweedie_deviance(y_true, y_pred, power=1)
    1.4260...
    """
    # 获取命名空间，可能是cupy或numpy的数组操作
    xp, _ = get_namespace(y_true, y_pred)
    # 检查和转换目标值为合适的数据类型
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[xp.float64, xp.float32]
    )
    # 如果目标类型是多输出连续型，抛出错误
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")
    # 检查目标值和预测值的长度是否一致
    check_consistent_length(y_true, y_pred, sample_weight)

    # 如果有样本权重，则处理样本权重
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]

    # 构建错误信息字符串，指定使用的 Tweedie deviance 函数及其参数
    message = f"Mean Tweedie deviance error with power={power} can only be used on "
    if power < 0:
        # 如果 power < 0，采用极稳定分布，要求 y_pred > 0
        if (y_pred <= 0).any():
            raise ValueError(message + "strictly positive y_pred.")
    elif power == 0:
        # 如果 power = 0，采用正态分布，y 和 y_pred 可以是任意实数
        pass
    elif 1 <= power < 2:
        # 如果 1 <= power < 2，采用泊松或复合泊松分布，要求 y_true >= 0 和 y_pred > 0
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(message + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        # 如果 power >= 2，采用伽马分布或极稳定分布，要求 y_true > 0 和 y_pred > 0
        if xp.any(y_true <= 0) or xp.any(y_pred <= 0):
            raise ValueError(message + "strictly positive y and y_pred.")
    else:  # pragma: nocover
        # 对于不可达的情况，抛出错误
        raise ValueError

    # 调用实际的 Tweedie deviance 计算函数，并返回结果
    return _mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )
# 定义一个装饰器函数，用于验证输入参数的类型和条件
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应当是类数组类型
        "y_pred": ["array-like"],  # y_pred 参数应当是类数组类型
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组类型或者 None
        "power": [
            Interval(Real, None, 0, closed="right"),  # power 参数应当为实数类型，且大于等于 0
            Interval(Real, 1, None, closed="left"),  # power 参数应当为实数类型，且大于 1
        ],
    },
    prefer_skip_nested_validation=True,  # 设置跳过嵌套验证，提高效率
)
def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0):
    """
    """
    :math:`D^2` regression score function, fraction of Tweedie deviance explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical mean of `y_true` as
    constant prediction, disregarding the input features, gets a D^2 score of 0.0.

    Read more in the :ref:`User Guide <d2_score>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.

        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.

        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to r2_score.
          y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.

    Returns
    -------
    z : float or ndarray of floats
        The D^2 score.

    Notes
    -----
    This is not a symmetric function.

    Like R^2, D^2 score may be negative (it need not actually be the square of
    a quantity D).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] Eq. (3.11) of Hastie, Trevor J., Robert Tibshirani and Martin J.
           Wainwright. "Statistical Learning with Sparsity: The Lasso and
           Generalizations." (2015). https://hastie.su.domains/StatLearnSparsity/

    Examples
    --------
    >>> from sklearn.metrics import d2_tweedie_score
    >>> y_true = [0.5, 1, 2.5, 7]
    >>> y_pred = [1, 1, 5, 3.5]
    >>> d2_tweedie_score(y_true, y_pred)
    0.285...
    >>> d2_tweedie_score(y_true, y_pred, power=1)
    0.487...
    >>> d2_tweedie_score(y_true, y_pred, power=2)
    0.630...
    >>> d2_tweedie_score(y_true, y_true, power=2)
    1.0
    """

    # 获取数值计算库及其命名空间
    xp, _ = get_namespace(y_true, y_pred)

    # 检查并验证目标值的类型和数据
    y_type, y_true, y_pred, _ = _check_reg_targets(
        y_true, y_pred, None, dtype=[xp.float64, xp.float32], xp=xp
    )

    # 如果目标值类型为多输出连续型，则抛出异常
    if y_type == "continuous-multioutput":
        raise ValueError("Multioutput not supported in d2_tweedie_score")
    # 如果预测值中的样本数小于2，则无法定义 D^2 分数
    if _num_samples(y_pred) < 2:
        # 提示警告信息
        msg = "D^2 score is not well-defined with less than two samples."
        # 发出警告
        warnings.warn(msg, UndefinedMetricWarning)
        # 返回一个 NaN 值
        return float("nan")

    # 将真实值和预测值按指定轴进行压缩，确保为一维数组
    y_true, y_pred = xp.squeeze(y_true, axis=1), xp.squeeze(y_pred, axis=1)
    # 计算 Tweedie 偏差的分子部分
    numerator = mean_tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=power
    )

    # 计算加权平均的真实值
    y_avg = _average(y_true, weights=sample_weight, xp=xp)
    # 计算 Tweedie 偏差的分母部分
    denominator = _mean_tweedie_deviance(
        y_true, y_avg, sample_weight=sample_weight, power=power
    )

    # 返回 D^2 统计量的值
    return 1 - numerator / denominator
# 使用装饰器 @validate_params 进行参数验证，确保输入参数符合指定的类型和要求
@validate_params(
    {
        "y_true": ["array-like"],  # y_true 参数应该是类数组结构
        "y_pred": ["array-like"],  # y_pred 参数应该是类数组结构
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组结构或者 None
        "alpha": [Interval(Real, 0, 1, closed="both")],  # alpha 参数是一个介于 0 和 1 之间的实数，包含边界值
        "multioutput": [
            StrOptions({"raw_values", "uniform_average"}),  # multioutput 参数可以是 'raw_values' 或 'uniform_average'
            "array-like",  # 或者是类数组结构
        ],
    },
    prefer_skip_nested_validation=True,  # 优先跳过嵌套验证
)
def d2_pinball_score(
    y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"
):
    """
    :math:`D^2` regression score function, fraction of pinball loss explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical alpha-quantile of
    `y_true` as constant prediction, disregarding the input features,
    gets a :math:`D^2` score of 0.0.

    Read more in the :ref:`User Guide <d2_score>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    alpha : float, default=0.5
        Slope of the pinball deviance. It determines the quantile level alpha
        for which the pinball deviance and also D2 are optimal.
        The default `alpha=0.5` is equivalent to `d2_absolute_error_score`.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average scores.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray of floats
        The :math:`D^2` score with a pinball deviance
        or ndarray of scores if `multioutput='raw_values'`.

    Notes
    -----
    Like :math:`R^2`, :math:`D^2` score may be negative
    (it need not actually be the square of a quantity D).

    This metric is not well-defined for a single point and will return a NaN
    value if n_samples is less than two.

     References
    ----------
    .. [1] Eq. (7) of `Koenker, Roger; Machado, José A. F. (1999).
           "Goodness of Fit and Related Inference Processes for Quantile Regression"
           <https://doi.org/10.1080/01621459.1999.10473882>`_
    .. [2] Eq. (3.11) of Hastie, Trevor J., Robert Tibshirani and Martin J.
           Wainwright. "Statistical Learning with Sparsity: The Lasso and
           Generalizations." (2015). https://hastie.su.domains/StatLearnSparsity/

    Examples
    --------
    >>> from sklearn.metrics import d2_pinball_score
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 3, 3]
    >>> d2_pinball_score(y_true, y_pred)
    0.5
    >>> d2_pinball_score(y_true, y_pred, alpha=0.9)
    0.772...
    >>> d2_pinball_score(y_true, y_pred, alpha=0.1)
    -1.045...
    >>> d2_pinball_score(y_true, y_true, alpha=0.1)
    1.0
    """
    # 检查目标数据类型及其一致性，返回相应的处理后的变量
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    # 检查目标值和预测值的长度一致性，并检查样本权重的一致性
    check_consistent_length(y_true, y_pred, sample_weight)

    # 如果预测值样本数小于2，则D^2分数未定义
    if _num_samples(y_pred) < 2:
        msg = "D^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float("nan")

    # 计算分子，即平均分位损失
    numerator = mean_pinball_loss(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        alpha=alpha,
        multioutput="raw_values",
    )

    # 根据是否有样本权重，计算目标分位数值
    if sample_weight is None:
        y_quantile = np.tile(
            np.percentile(y_true, q=alpha * 100, axis=0), (len(y_true), 1)
        )
    else:
        sample_weight = _check_sample_weight(sample_weight, y_true)
        y_quantile = np.tile(
            _weighted_percentile(
                y_true, sample_weight=sample_weight, percentile=alpha * 100
            ),
            (len(y_true), 1),
        )

    # 计算分母，即平均分位损失
    denominator = mean_pinball_loss(
        y_true,
        y_quantile,
        sample_weight=sample_weight,
        alpha=alpha,
        multioutput="raw_values",
    )

    # 确定非零的分子和分母，以计算有效得分
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = np.ones(y_true.shape[1])

    # 计算有效得分并赋值给输出得分数组
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    # 处理多输出情况下的均值计算权重
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # 返回各自的得分
            return output_scores
        else:  # multioutput == "uniform_average"
            # 通过将权重设置为None，使用np.average进行均匀平均
            avg_weights = None
    else:
        avg_weights = multioutput

    # 返回加权平均得分
    return np.average(output_scores, weights=avg_weights)
@validate_params(
    {  # 使用 validate_params 装饰器，验证输入参数的类型和约束条件
        "y_true": ["array-like"],  # y_true 参数应为类数组类型
        "y_pred": ["array-like"],  # y_pred 参数应为类数组类型
        "sample_weight": ["array-like", None],  # sample_weight 参数可以是类数组类型或 None
        "multioutput": [  # multioutput 参数有两种可能的类型选择
            StrOptions({"raw_values", "uniform_average"}),  # 可以是预定义的字符串选项集合
            "array-like",  # 或者可以是类数组类型
        ],
    },
    prefer_skip_nested_validation=True,  # 偏好跳过嵌套验证
)
def d2_absolute_error_score(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    """
    :math:`D^2` regression score function, fraction of absolute error explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical median of `y_true`
    as constant prediction, disregarding the input features,
    gets a :math:`D^2` score of 0.0.

    Read more in the :ref:`User Guide <d2_score>`.

    .. versionadded:: 1.1  # 引入版本说明，表示从版本 1.1 开始添加

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average scores.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score : float or ndarray of floats
        The :math:`D^2` score with an absolute error deviance
        or ndarray of scores if 'multioutput' is 'raw_values'.

    Notes
    -----
    Like :math:`R^2`, :math:`D^2` score may be negative
    (it need not actually be the square of a quantity D).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

     References
    ----------
    .. [1] Eq. (3.11) of Hastie, Trevor J., Robert Tibshirani and Martin J.
           Wainwright. "Statistical Learning with Sparsity: The Lasso and
           Generalizations." (2015). https://hastie.su.domains/StatLearnSparsity/

    Examples
    --------
    >>> from sklearn.metrics import d2_absolute_error_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> d2_absolute_error_score(y_true, y_pred)
    0.764...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> d2_absolute_error_score(y_true, y_pred, multioutput='uniform_average')
    0.691...
    >>> d2_absolute_error_score(y_true, y_pred, multioutput='raw_values')
    array([0.8125    , 0.57142857])
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> d2_absolute_error_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    """
    # 定义一个示例的预测结果列表
    >>> y_pred = [2, 2, 2]
    # 调用函数 d2_absolute_error_score 计算真实值 y_true 和预测值 y_pred 的 D2 绝对误差得分，预期返回值为 0.0
    >>> d2_absolute_error_score(y_true, y_pred)
    0.0
    # 更新真实值 y_true 的列表
    >>> y_true = [1, 2, 3]
    # 更新预测值 y_pred 的列表
    >>> y_pred = [3, 2, 1]
    # 调用函数 d2_absolute_error_score 计算更新后的真实值 y_true 和预测值 y_pred 的 D2 绝对误差得分，预期返回值为 -1.0
    >>> d2_absolute_error_score(y_true, y_pred)
    -1.0
    """
    # 返回使用 d2_pinball_score 函数计算的 D2 乒乓球损失得分，参数包括真实值 y_true、预测值 y_pred、样本权重 sample_weight、alpha 参数设置为 0.5、multioutput 参数
    return d2_pinball_score(
        y_true, y_pred, sample_weight=sample_weight, alpha=0.5, multioutput=multioutput
    )
```