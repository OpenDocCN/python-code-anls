# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_regression.py`

```
# 导入必要的模块和函数
from itertools import product  # 导入 itertools 模块中的 product 函数

import numpy as np  # 导入 NumPy 库并使用别名 np
import pytest  # 导入 pytest 库，用于单元测试
from numpy.testing import assert_allclose  # 从 NumPy 测试模块中导入 assert_allclose 函数
from scipy import optimize  # 导入 scipy 库中的 optimize 模块
from scipy.special import factorial, xlogy  # 从 scipy.special 导入 factorial 和 xlogy 函数

# 导入 sklearn 相关模块和函数
from sklearn.dummy import DummyRegressor  # 导入 sklearn 中的 DummyRegressor 类
from sklearn.exceptions import UndefinedMetricWarning  # 导入 sklearn 中的 UndefinedMetricWarning 异常类
from sklearn.metrics import (  # 导入 sklearn 中的多个评估指标函数
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.metrics._regression import _check_reg_targets  # 导入 sklearn 中的 _check_reg_targets 函数
from sklearn.model_selection import GridSearchCV  # 导入 sklearn 中的 GridSearchCV 类
from sklearn.utils._testing import (  # 导入 sklearn 内部测试模块中的多个断言函数
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

# 定义回归指标测试函数
def test_regression_metrics(n_samples=50):
    y_true = np.arange(n_samples)  # 创建真实值数组，从 0 到 n_samples-1
    y_pred = y_true + 1  # 创建预测值数组，比真实值数组每个元素都大 1
    y_pred_2 = y_true - 1  # 创建另一个预测值数组，比真实值数组每个元素都小 1

    # 断言预测值与真实值之间的均方误差接近 1.0
    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.0)
    # 断言预测值与真实值之间的均方对数误差接近
    assert_almost_equal(
        mean_squared_log_error(y_true, y_pred),
        mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)),
    )
    # 断言预测值与真实值之间的平均绝对误差接近 1.0
    assert_almost_equal(mean_absolute_error(y_true, y_pred), 1.0)
    # 断言预测值与真实值之间的平均分位损失接近 0.5
    assert_almost_equal(mean_pinball_loss(y_true, y_pred), 0.5)
    # 断言另一个预测值与真实值之间的平均分位损失接近 0.5
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2), 0.5)
    # 断言预测值与真实值之间的平均分位损失（带 alpha 参数）接近 0.6
    assert_almost_equal(mean_pinball_loss(y_true, y_pred, alpha=0.4), 0.6)
    # 断言另一个预测值与真实值之间的平均分位损失（带 alpha 参数）接近 0.4
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2, alpha=0.4), 0.4)
    # 断言预测值与真实值之间的中位数绝对误差接近 1.0
    assert_almost_equal(median_absolute_error(y_true, y_pred), 1.0)
    # 计算预测值与真实值之间的平均绝对百分比误差
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # 断言平均绝对百分比误差是有限的并且大于 1e6
    assert np.isfinite(mape)
    assert mape > 1e6
    # 断言预测值与真实值之间的最大误差接近 1.0
    assert_almost_equal(max_error(y_true, y_pred), 1.0)
    # 断言预测值与真实值之间的 R^2 分数接近 0.995（精确到小数点后两位）
    assert_almost_equal(r2_score(y_true, y_pred), 0.995, 2)
    # 断言预测值与真实值之间的 R^2 分数接近 0.995（精确到小数点后两位），并且不强制要求有限性
    assert_almost_equal(r2_score(y_true, y_pred, force_finite=False), 0.995, 2)
    # 断言解释方差得分接近 1.0
    assert_almost_equal(explained_variance_score(y_true, y_pred), 1.0)
    # 断言解释方差得分接近 1.0，并且不强制要求有限性
    assert_almost_equal(
        explained_variance_score(y_true, y_pred, force_finite=False), 1.0
    )
    # 断言预测值与真实值之间的 Tweedie 损失的均值接近均方误差
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=0),
        mean_squared_error(y_true, y_pred),
    )
    # 断言预测值与真实值之间的 Tweedie 分数接近 R^2 分数
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=0), r2_score(y_true, y_pred)
    )
    # 计算真实值与其中位数的绝对差值的和
    dev_median = np.abs(y_true - np.median(y_true)).sum()
    # 断言预测值与真实值之间的绝对误差分数
    assert_array_almost_equal(
        d2_absolute_error_score(y_true, y_pred),
        1 - np.abs(y_true - y_pred).sum() / dev_median,
    )
    # 定义分位损失函数
    alpha = 0.2
    pinball_loss = lambda y_true, y_pred, alpha: alpha * np.maximum(
        y_true - y_pred, 0
    ) + (1 - alpha) * np.maximum(y_pred - y_true, 0)
    # 计算真实值的 alpha 分位数
    y_quantile = np.percentile(y_true, q=alpha * 100)
    assert_almost_equal(
        d2_pinball_score(y_true, y_pred, alpha=alpha),
        1
        - pinball_loss(y_true, y_pred, alpha).sum()
        / pinball_loss(y_true, y_quantile, alpha).sum(),
    )
    # 断言：计算 D2 pinball 评分，与计算损失函数之间的关系
    assert_almost_equal(
        d2_absolute_error_score(y_true, y_pred),
        d2_pinball_score(y_true, y_pred, alpha=0.5),
    )
    # 断言：计算 D2 绝对误差评分与 D2 pinball 评分(alpha=0.5)的关系

    # Tweedie deviance 需要 y_pred 为正数，除非 p=0，
    # 对于 p>=2，需要 y_true 为正数
    # 结果通过 sympy 进行评估
    y_true = np.arange(1, 1 + n_samples)
    y_pred = 2 * y_true
    n = n_samples
    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=-1),
        5 / 12 * n * (n**2 + 2 * n + 1),
    )
    # 断言：计算 Tweedie 偏差的平均值，当 power=-1 时的期望值

    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=1), (n + 1) * (1 - np.log(2))
    )
    # 断言：计算 Tweedie 偏差的平均值，当 power=1 时的期望值

    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=2), 2 * np.log(2) - 1
    )
    # 断言：计算 Tweedie 偏差的平均值，当 power=2 时的期望值

    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=3 / 2),
        ((6 * np.sqrt(2) - 8) / n) * np.sqrt(y_true).sum(),
    )
    # 断言：计算 Tweedie 偏差的平均值，当 power=3/2 时的期望值

    assert_almost_equal(
        mean_tweedie_deviance(y_true, y_pred, power=3), np.sum(1 / y_true) / (4 * n)
    )
    # 断言：计算 Tweedie 偏差的平均值，当 power=3 时的期望值

    dev_mean = 2 * np.mean(xlogy(y_true, 2 * y_true / (n + 1)))
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=1),
        1 - (n + 1) * (1 - np.log(2)) / dev_mean,
    )
    # 计算 Tweedie 分数 D2，与 Tweedie 偏差平均值之间的关系

    dev_mean = 2 * np.log((n + 1) / 2) - 2 / n * np.log(factorial(n))
    assert_almost_equal(
        d2_tweedie_score(y_true, y_pred, power=2), 1 - (2 * np.log(2) - 1) / dev_mean
    )
    # 计算 Tweedie 分数 D2，与 Tweedie 偏差平均值之间的关系
# 测试 root_mean_squared_error 函数对于 multioutput="raw_values" 参数的非回归测试
def test_root_mean_squared_error_multioutput_raw_value():
    # 使用 mean_squared_error 计算均方误差，传入两个 1x1 的数组作为参数
    mse = mean_squared_error([[1]], [[10]], multioutput="raw_values")
    # 使用 root_mean_squared_error 计算均方根误差，传入两个 1x1 的数组作为参数
    rmse = root_mean_squared_error([[1]], [[10]], multioutput="raw_values")
    # 断言均方误差的平方根等于 root_mean_squared_error 计算结果的近似值
    assert np.sqrt(mse) == pytest.approx(rmse)


# 测试多输出回归的不同评估指标
def test_multioutput_regression():
    # 定义真实值和预测值的 NumPy 数组
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    # 计算均方误差并断言结果近似于指定值
    error = mean_squared_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 / 3 + 2.0 / 3 + 2.0 / 3) / 4.0)

    # 计算均方根误差并断言结果近似于指定值
    error = root_mean_squared_error(y_true, y_pred)
    assert_almost_equal(error, 0.454, decimal=2)

    # 计算均方对数误差并断言结果近似于指定值
    error = mean_squared_log_error(y_true, y_pred)
    assert_almost_equal(error, 0.200, decimal=2)

    # 计算均方根对数误差并断言结果近似于指定值
    error = root_mean_squared_log_error(y_true, y_pred)
    assert_almost_equal(error, 0.315, decimal=2)

    # 平均绝对误差和均方误差相等，因为这是一个二元问题
    error = mean_absolute_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 2.0 / 3) / 4.0)

    # 计算平均 Pinball 损失并断言结果近似于指定值
    error = mean_pinball_loss(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 2.0 / 3) / 8.0)

    # 计算平均绝对百分比误差并断言结果是有限的且大于 1e6
    error = np.around(mean_absolute_percentage_error(y_true, y_pred), decimals=2)
    assert np.isfinite(error)
    assert error > 1e6

    # 计算中位数绝对误差并断言结果近似于指定值
    error = median_absolute_error(y_true, y_pred)
    assert_almost_equal(error, (1.0 + 1.0) / 4.0)

    # 计算 R² 分数，并使用 multioutput="variance_weighted" 指定多输出策略
    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    assert_almost_equal(error, 1.0 - 5.0 / 2)

    # 计算 R² 分数，并使用 multioutput="uniform_average" 指定多输出策略
    error = r2_score(y_true, y_pred, multioutput="uniform_average")
    assert_almost_equal(error, -0.875)

    # 计算 D² Pinball 分数，并使用 multioutput="raw_values" 指定多输出策略
    score = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")
    # 预期的原始分数为数组，计算方式在注释中描述了分子和分母的情况
    raw_expected_score = [
        1
        - np.abs(y_true[:, i] - y_pred[:, i]).sum()
        / np.abs(y_true[:, i] - np.median(y_true[:, i])).sum()
        for i in range(y_true.shape[1])
    ]
    # 对于最后一种情况，分母为零，因此结果为 nan，但由于分子也为零，预期得分是 1.0
    raw_expected_score = np.where(np.isnan(raw_expected_score), 1, raw_expected_score)
    # 断言数组几乎等于预期的原始分数
    assert_array_almost_equal(score, raw_expected_score)

    # 计算 D² Pinball 分数，并使用 multioutput="uniform_average" 指定多输出策略
    score = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="uniform_average")
    # 断言分数几乎等于预期的原始分数的平均值
    assert_almost_equal(score, raw_expected_score.mean())

    # 当 force_finite=True 时，常量 y_true 导致 1.0 或 0.0
    yc = [5.0, 5.0]
    error = r2_score(yc, [5.0, 5.0], multioutput="variance_weighted")
    assert_almost_equal(error, 1.0)
    error = r2_score(yc, [5.0, 5.1], multioutput="variance_weighted")
    assert_almost_equal(error, 0.0)

    # 当 force_finite=False 时，第四个输出的 nan 会传播
    error = r2_score(
        y_true, y_pred, multioutput="variance_weighted", force_finite=False
    )
    assert_almost_equal(error, np.nan)
    # 使用 r2_score 函数计算预测值与真实值之间的 R^2 分数，采用 uniform_average 多输出参数，不强制有限数值计算
    error = r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False)
    # 断言检查计算的误差是否为 NaN
    assert_almost_equal(error, np.nan)

    # 删除第四个输出以检查 `force_finite=False` 是否有效
    y_true = y_true[:, :-1]
    y_pred = y_pred[:, :-1]
    
    # 使用 r2_score 函数计算新的 R^2 分数，采用 variance_weighted 多输出参数
    error = r2_score(y_true, y_pred, multioutput="variance_weighted")
    # 再次计算相同条件下的 R^2 分数，但是这次强制不使用有限数值计算
    error2 = r2_score(
        y_true, y_pred, multioutput="variance_weighted", force_finite=False
    )
    # 断言检查两次计算的误差是否相等
    assert_almost_equal(error, error2)

    # 使用 r2_score 函数计算 R^2 分数，采用 uniform_average 多输出参数
    error = r2_score(y_true, y_pred, multioutput="uniform_average")
    # 再次计算相同条件下的 R^2 分数，但是这次不强制使用有限数值计算
    error2 = r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False)
    # 断言检查两次计算的误差是否相等
    assert_almost_equal(error, error2)

    # 对于 `y_true` 是常数且 force_finite=False，可能导致结果是 NaN 或者 -Inf
    error = r2_score(
        yc, [5.0, 5.0], multioutput="variance_weighted", force_finite=False
    )
    # 断言检查计算的误差是否为 NaN
    assert_almost_equal(error, np.nan)
    error = r2_score(
        yc, [5.0, 6.0], multioutput="variance_weighted", force_finite=False
    )
    # 断言检查计算的误差是否为 -Inf
    assert_almost_equal(error, -np.inf)
# 定义用于测试回归指标在极限情况下的函数
def test_regression_metrics_at_limits():
    # 单样本情况

    # 断言均方误差（mean_squared_error）接近于零
    assert_almost_equal(mean_squared_error([0.0], [0.0]), 0.0)

    # 断言均方根误差（root_mean_squared_error）接近于零
    assert_almost_equal(root_mean_squared_error([0.0], [0.0]), 0.0)

    # 断言均方对数误差（mean_squared_log_error）接近于零
    assert_almost_equal(mean_squared_log_error([0.0], [0.0]), 0.0)

    # 断言平均绝对误差（mean_absolute_error）接近于零
    assert_almost_equal(mean_absolute_error([0.0], [0.0]), 0.0)

    # 断言平均分位数损失（mean_pinball_loss）接近于零
    assert_almost_equal(mean_pinball_loss([0.0], [0.0]), 0.0)

    # 断言平均绝对百分比误差（mean_absolute_percentage_error）接近于零
    assert_almost_equal(mean_absolute_percentage_error([0.0], [0.0]), 0.0)

    # 断言中位数绝对误差（median_absolute_error）接近于零
    assert_almost_equal(median_absolute_error([0.0], [0.0]), 0.0)

    # 断言最大误差（max_error）接近于零
    assert_almost_equal(max_error([0.0], [0.0]), 0.0)

    # 断言可解释方差得分（explained_variance_score）接近于1.0
    assert_almost_equal(explained_variance_score([0.0], [0.0]), 1.0)

    # 完美情况

    # 断言R²得分（r2_score）接近于1.0
    assert_almost_equal(r2_score([0.0, 1], [0.0, 1]), 1.0)

    # 断言Tweedie分位损失（d2_pinball_score）接近于1.0
    assert_almost_equal(d2_pinball_score([0.0, 1], [0.0, 1]), 1.0)

    # 非有限情况

    # 对于R²和可解释方差得分，非有限情况下的断言
    for s in (r2_score, explained_variance_score):
        assert_almost_equal(s([0, 0], [1, -1]), 0.0)
        assert_almost_equal(s([0, 0], [1, -1], force_finite=False), -np.inf)
        assert_almost_equal(s([1, 1], [1, 1]), 1.0)
        assert_almost_equal(s([1, 1], [1, 1], force_finite=False), np.nan)

    # 断言在目标值包含负值时抛出值错误异常，针对均方对数误差（mean_squared_log_error）
    msg = (
        "Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([-1.0], [-1.0])
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])
    with pytest.raises(ValueError, match=msg):
        mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])

    # 断言在目标值包含负值时抛出值错误异常，针对均方根对数误差（root_mean_squared_log_error）
    msg = (
        "Root Mean Squared Logarithmic Error cannot be used when targets "
        "contain negative values."
    )
    with pytest.raises(ValueError, match=msg):
        root_mean_squared_log_error([1.0, -2.0, 3.0], [1.0, 2.0, 3.0])

    # Tweedie偏差误差

    # 定义Tweedie偏差的功率值
    power = -1.2

    # 断言均值Tweedie偏差（mean_tweedie_deviance）接近于2 / (2 - power)
    assert_allclose(
        mean_tweedie_deviance([0], [1.0], power=power), 2 / (2 - power), rtol=1e-3
    )

    # 断言在y_pred严格为正值时可用于Tweedie偏差，否则抛出值错误异常
    msg = "can only be used on strictly positive y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)

    # 断言均值Tweedie偏差（mean_tweedie_deviance）接近于零，用于power等于0的情况
    assert_almost_equal(mean_tweedie_deviance([0.0], [0.0], power=0), 0.0, 2)

    # 重新定义power值为1.0

    power = 1.0

    # 断言在y为非负且y_pred严格为正时可用于Tweedie偏差，否则抛出值错误异常
    msg = "only be used on non-negative y and strictly positive y_pred."
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    # 使用 pytest 框架验证 d2_tweedie_score 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    
    # 设置 tweedie 指数 power 为 1.5
    power = 1.5
    # 使用 assert_allclose 验证 mean_tweedie_deviance 函数计算结果是否接近预期值 2 / (2 - power)
    assert_allclose(mean_tweedie_deviance([0.0], [1.0], power=power), 2 / (2 - power))
    # 更新异常消息字符串 msg
    msg = "only be used on non-negative y and strictly positive y_pred."
    # 使用 pytest 框架验证 mean_tweedie_deviance 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    # 使用 pytest 框架验证 d2_tweedie_score 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    
    # 设置 tweedie 指数 power 为 2.0
    power = 2.0
    # 使用 assert_allclose 验证 mean_tweedie_deviance 函数计算结果是否接近预期值 0.00，允许的误差为 1e-8
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.00, atol=1e-8)
    # 更新异常消息字符串 msg
    msg = "can only be used on strictly positive y and y_pred."
    # 使用 pytest 框架验证 mean_tweedie_deviance 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    # 使用 pytest 框架验证 d2_tweedie_score 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
    
    # 设置 tweedie 指数 power 为 3.0
    power = 3.0
    # 使用 assert_allclose 验证 mean_tweedie_deviance 函数计算结果是否接近预期值 0.00，允许的误差为 1e-8
    assert_allclose(mean_tweedie_deviance([1.0], [1.0], power=power), 0.00, atol=1e-8)
    # 更新异常消息字符串 msg
    msg = "can only be used on strictly positive y and y_pred."
    # 使用 pytest 框架验证 mean_tweedie_deviance 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        mean_tweedie_deviance([0.0], [0.0], power=power)
    # 使用 pytest 框架验证 d2_tweedie_score 函数在给定异常条件时是否引发 ValueError 异常，异常消息需要匹配特定的字符串 msg
    with pytest.raises(ValueError, match=msg):
        d2_tweedie_score([0.0] * 2, [0.0] * 2, power=power)
def test__check_reg_targets():
    # 定义一个示例列表，每个元素是一个三元组，包含类型、目标数据、输出数
    EXAMPLES = [
        ("continuous", [1, 2, 3], 1),  # 连续型数据，目标是一维数组，输出数为1
        ("continuous", [[1], [2], [3]], 1),  # 连续型数据，目标是二维数组，输出数为1
        ("continuous-multioutput", [[1, 1], [2, 2], [3, 1]], 2),  # 多输出连续型数据，目标是二维数组，输出数为2
        ("continuous-multioutput", [[5, 1], [4, 2], [3, 1]], 2),  # 多输出连续型数据，目标是二维数组，输出数为2
        ("continuous-multioutput", [[1, 3, 4], [2, 2, 2], [3, 1, 1]], 3),  # 多输出连续型数据，目标是二维数组，输出数为3
    ]

    # 使用product函数生成EXAMPLES的所有排列组合，每个元素是一个元组
    for (type1, y1, n_out1), (type2, y2, n_out2) in product(EXAMPLES, repeat=2):
        # 如果类型和输出数都相同
        if type1 == type2 and n_out1 == n_out2:
            # 调用_check_reg_targets函数，返回的是类型、检查后的y1和y2、multioutput值
            y_type, y_check1, y_check2, multioutput = _check_reg_targets(y1, y2, None)
            # 断言类型相同
            assert type1 == y_type
            # 如果类型是"continuous"
            if type1 == "continuous":
                # 断言y_check1和y1的形状一致
                assert_array_equal(y_check1, np.reshape(y1, (-1, 1)))
                # 断言y_check2和y2的形状一致
                assert_array_equal(y_check2, np.reshape(y2, (-1, 1)))
            else:
                # 断言y_check1和y1相等
                assert_array_equal(y_check1, y1)
                # 断言y_check2和y2相等
                assert_array_equal(y_check2, y2)
        else:
            # 如果类型或输出数不同，预期触发ValueError异常
            with pytest.raises(ValueError):
                _check_reg_targets(y1, y2, None)


def test__check_reg_targets_exception():
    # 定义无效的multioutput值
    invalid_multioutput = "this_value_is_not_valid"
    # 期望的异常消息字符串格式
    expected_message = (
        "Allowed 'multioutput' string values are.+You provided multioutput={!r}".format(
            invalid_multioutput
        )
    )
    # 使用pytest检查是否触发预期异常消息的ValueError异常
    with pytest.raises(ValueError, match=expected_message):
        _check_reg_targets([1, 2, 3], [[1], [2], [3]], invalid_multioutput)


def test_regression_multioutput_array():
    # 定义真实值和预测值的二维数组
    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    # 计算均方误差（MSE），返回多输出的原始值
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    # 计算平均绝对误差（MAE），返回多输出的原始值
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")

    # 计算平均分位数损失（PBL），返回多输出的原始值
    pbl = mean_pinball_loss(y_true, y_pred, multioutput="raw_values")
    # 计算平均绝对百分比误差（MAPE），返回多输出的原始值
    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    # 计算R^2分数，返回多输出的原始值
    r = r2_score(y_true, y_pred, multioutput="raw_values")
    # 计算解释方差得分，返回多输出的原始值
    evs = explained_variance_score(y_true, y_pred, multioutput="raw_values")
    # 计算D2分位数损失，返回多输出的原始值
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")
    # 计算解释方差得分，返回多输出的原始值，强制finite
    evs2 = explained_variance_score(
        y_true, y_pred, multioutput="raw_values", force_finite=False
    )

    # 断言均方误差（MSE）的结果与预期结果非常接近
    assert_array_almost_equal(mse, [0.125, 0.5625], decimal=2)
    # 断言平均绝对误差（MAE）的结果与预期结果非常接近
    assert_array_almost_equal(mae, [0.25, 0.625], decimal=2)
    # 断言平均分位数损失（PBL）的结果与预期结果非常接近
    assert_array_almost_equal(pbl, [0.25 / 2, 0.625 / 2], decimal=2)
    # 断言平均绝对百分比误差（MAPE）的结果与预期结果非常接近
    assert_array_almost_equal(mape, [0.0778, 0.2262], decimal=2)
    # 断言R^2分数的结果与预期结果非常接近
    assert_array_almost_equal(r, [0.95, 0.93], decimal=2)
    # 断言解释方差得分（EVS）的结果与预期结果非常接近
    assert_array_almost_equal(evs, [0.95, 0.93], decimal=2)
    # 断言D2分位数损失（D2PS）的结果与预期结果非常接近
    assert_array_almost_equal(d2ps, [0.833, 0.722], decimal=2)
    # 断言解释方差得分（EVS2）的结果与预期结果非常接近
    assert_array_almost_equal(evs2, [0.95, 0.93], decimal=2)

    # 由于是二进制问题，均方误差（MSE）和平均绝对误差（MAE）相等
    y_true = [[0, 0]] * 4
    y_pred = [[1, 1]] * 4
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    # 计算平均损失分位数损失
    pbl = mean_pinball_loss(y_true, y_pred, multioutput="raw_values")

    # 计算R²分数
    r = r2_score(y_true, y_pred, multioutput="raw_values")

    # 计算D²损失分数
    d2ps = d2_pinball_score(y_true, y_pred, multioutput="raw_values")

    # 断言均方误差接近给定值 [1.0, 1.0]
    assert_array_almost_equal(mse, [1.0, 1.0], decimal=2)

    # 断言平均绝对误差接近给定值 [1.0, 1.0]
    assert_array_almost_equal(mae, [1.0, 1.0], decimal=2)

    # 断言平均损失分位数损失接近给定值 [0.5, 0.5]
    assert_array_almost_equal(pbl, [0.5, 0.5], decimal=2)

    # 断言R²分数接近给定值 [0.0, 0.0]
    assert_array_almost_equal(r, [0.0, 0.0], decimal=2)

    # 断言D²损失分数接近给定值 [0.0, 0.0]
    assert_array_almost_equal(d2ps, [0.0, 0.0], decimal=2)

    # 使用其他数据进行R²分数的计算，并进行断言验证
    r = r2_score([[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="raw_values")
    assert_array_almost_equal(r, [0, -3.5], decimal=2)

    # 断言R²分数的均值等于指定的统一平均值
    assert np.mean(r) == r2_score(
        [[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="uniform_average"
    )

    # 计算解释方差分数
    evs = explained_variance_score(
        [[0, -1], [0, 1]], [[2, 2], [1, 1]], multioutput="raw_values"
    )

    # 断言解释方差分数接近给定值 [0, -1.25]
    assert_array_almost_equal(evs, [0, -1.25], decimal=2)

    # 使用force_finite=False参数进行解释方差分数计算
    evs2 = explained_variance_score(
        [[0, -1], [0, 1]],
        [[2, 2], [1, 1]],
        multioutput="raw_values",
        force_finite=False,
    )

    # 断言解释方差分数接近给定值 [-∞, -1.25]
    assert_array_almost_equal(evs2, [-np.inf, -1.25], decimal=2)

    # 处理分母和分子均为零的条件
    y_true = [[1, 3], [1, 2]]
    y_pred = [[1, 4], [1, 1]]

    # 计算R²分数
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    # 断言R²分数接近给定值 [1.0, -3.0]
    assert_array_almost_equal(r2, [1.0, -3.0], decimal=2)

    # 断言R²分数的均值等于指定的统一平均值
    assert np.mean(r2) == r2_score(y_true, y_pred, multioutput="uniform_average")

    # 使用force_finite=False参数进行R²分数计算
    r22 = r2_score(y_true, y_pred, multioutput="raw_values", force_finite=False)

    # 断言R²分数接近给定值 [NaN, -3.0]
    assert_array_almost_equal(r22, [np.nan, -3.0], decimal=2)

    # 断言R²分数的均值接近指定的统一平均值
    assert_almost_equal(
        np.mean(r22),
        r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=False),
    )

    # 计算解释方差分数
    evs = explained_variance_score(y_true, y_pred, multioutput="raw_values")

    # 断言解释方差分数接近给定值 [1.0, -3.0]
    assert_array_almost_equal(evs, [1.0, -3.0], decimal=2)

    # 断言解释方差分数的均值等于解释方差分数
    assert np.mean(evs) == explained_variance_score(y_true, y_pred)

    # 计算D²损失分数
    d2ps = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput="raw_values")

    # 断言D²损失分数接近给定值 [1.0, -1.0]
    assert_array_almost_equal(d2ps, [1.0, -1.0], decimal=2)

    # 使用force_finite=False参数进行解释方差分数计算
    evs2 = explained_variance_score(
        y_true, y_pred, multioutput="raw_values", force_finite=False
    )

    # 断言解释方差分数接近给定值 [NaN, -3.0]
    assert_array_almost_equal(evs2, [np.nan, -3.0], decimal=2)

    # 断言解释方差分数的均值接近解释方差分数
    assert_almost_equal(
        np.mean(evs2), explained_variance_score(y_true, y_pred, force_finite=False)
    )

    # 单独处理均方对数误差，因其不接受负输入
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])

    # 计算均方对数误差
    msle = mean_squared_log_error(y_true, y_pred, multioutput="raw_values")

    # 通过均方误差计算对数的方式计算均方对数误差
    msle2 = mean_squared_error(
        np.log(1 + y_true), np.log(1 + y_pred), multioutput="raw_values"
    )

    # 断言均方对数误差接近通过均方误差计算对数得到的值
    assert_array_almost_equal(msle, msle2, decimal=2)
# 定义一个用于测试回归指标的函数，使用自定义的权重来评估不同的指标
def test_regression_custom_weights():
    # 真实值
    y_true = [[1, 2], [2.5, -1], [4.5, 3], [5, 7]]
    # 预测值
    y_pred = [[1, 1], [2, -1], [5, 4], [5, 6.5]]

    # 计算带权重的均方误差
    msew = mean_squared_error(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的均方根误差
    rmsew = root_mean_squared_error(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的平均绝对误差
    maew = mean_absolute_error(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的平均绝对百分比误差
    mapew = mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的R^2分数
    rw = r2_score(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的解释方差分数
    evsw = explained_variance_score(y_true, y_pred, multioutput=[0.4, 0.6])
    # 计算带权重的D2 pinball评分
    d2psw = d2_pinball_score(y_true, y_pred, alpha=0.5, multioutput=[0.4, 0.6])
    # 计算带权重的解释方差分数（允许非有限数值）
    evsw2 = explained_variance_score(
        y_true, y_pred, multioutput=[0.4, 0.6], force_finite=False
    )

    # 断言近似相等，验证计算结果是否符合预期
    assert_almost_equal(msew, 0.39, decimal=2)
    assert_almost_equal(rmsew, 0.59, decimal=2)
    assert_almost_equal(maew, 0.475, decimal=3)
    assert_almost_equal(mapew, 0.1668, decimal=2)
    assert_almost_equal(rw, 0.94, decimal=2)
    assert_almost_equal(evsw, 0.94, decimal=2)
    assert_almost_equal(d2psw, 0.766, decimal=2)
    assert_almost_equal(evsw2, 0.94, decimal=2)

    # 单独处理MSLE，因为它不接受负数输入
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    # 计算带权重的均方对数误差
    msle = mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7])
    # 使用转换后的数据计算带权重的均方误差
    msle2 = mean_squared_error(
        np.log(1 + y_true), np.log(1 + y_pred), multioutput=[0.3, 0.7]
    )
    # 断言近似相等，验证计算结果是否符合预期
    assert_almost_equal(msle, msle2, decimal=2)


# 使用pytest的参数化测试来测试单一样本的回归指标
@pytest.mark.parametrize("metric", [r2_score, d2_tweedie_score, d2_pinball_score])
def test_regression_single_sample(metric):
    y_true = [0]
    y_pred = [1]
    warning_msg = "not well-defined with less than two samples."

    # 触发警告
    with pytest.warns(UndefinedMetricWarning, match=warning_msg):
        # 计算指定的回归指标
        score = metric(y_true, y_pred)
        # 断言结果为NaN
        assert np.isnan(score)


# 测试Tweedie偏差连续性
def test_tweedie_deviance_continuity():
    n_samples = 100

    # 随机生成样本数据
    y_true = np.random.RandomState(0).rand(n_samples) + 0.1
    y_pred = np.random.RandomState(1).rand(n_samples) + 0.1

    # 验证当幂接近0时，Tweedie偏差的连续性
    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=0 - 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=0),
    )

    # 对于幂接近1的情况，增加绝对容差，因为边缘处可能存在数值精度问题
    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=1 + 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=1),
        atol=1e-6,
    )

    # 对于幂接近2的情况，增加绝对容差，因为边缘处可能存在数值精度问题
    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=2 - 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=2),
        atol=1e-6,
    )

    # 对于幂接近2的情况，增加绝对容差，因为边缘处可能存在数值精度问题
    assert_allclose(
        mean_tweedie_deviance(y_true, y_pred, power=2 + 1e-10),
        mean_tweedie_deviance(y_true, y_pred, power=2),
        atol=1e-6,
    )
# 定义测试函数，用于测试平均绝对百分比误差
def test_mean_absolute_percentage_error():
    # 创建随机数生成器对象，并设置种子为42
    random_number_generator = np.random.RandomState(42)
    # 生成服从指数分布的随机样本作为真实值
    y_true = random_number_generator.exponential(size=100)
    # 用1.2倍的真实值作为预测值
    y_pred = 1.2 * y_true
    # 断言计算得到的平均绝对百分比误差等于预期值，使用 pytest 的近似比较
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(0.2)


# 使用参数化装饰器，测试在常量预测下的平均分位损失
@pytest.mark.parametrize(
    "distribution", ["normal", "lognormal", "exponential", "uniform"]
)
@pytest.mark.parametrize("target_quantile", [0.05, 0.5, 0.75])
def test_mean_pinball_loss_on_constant_predictions(distribution, target_quantile):
    # 检查 numpy 是否支持 np.quantile，否则跳过测试
    if not hasattr(np, "quantile"):
        pytest.skip(
            "This test requires a more recent version of numpy "
            "with support for np.quantile."
        )

    # 指定样本数目
    n_samples = 3000
    # 创建随机数生成器对象，并设置种子为42
    rng = np.random.RandomState(42)
    # 生成指定分布的随机样本数据
    data = getattr(rng, distribution)(size=n_samples)

    # 计算任意常量预测值下的最佳分位损失
    best_pred = np.quantile(data, target_quantile)
    best_constant_pred = np.full(n_samples, fill_value=best_pred)
    best_pbl = mean_pinball_loss(data, best_constant_pred, alpha=target_quantile)

    # 在一系列分位数上评估损失
    candidate_predictions = np.quantile(data, np.linspace(0, 1, 100))
    for pred in candidate_predictions:
        # 计算常量预测值下的分位损失
        constant_pred = np.full(n_samples, fill_value=pred)
        pbl = mean_pinball_loss(data, constant_pred, alpha=target_quantile)

        # 断言当前常量预测值的损失大于等于最佳分位损失，考虑到机器精度
        assert pbl >= best_pbl - np.finfo(best_pbl.dtype).eps

        # 检查分位损失值与分析公式计算的预期分位损失值的接近程度
        expected_pbl = (pred - data[data < pred]).sum() * (1 - target_quantile) + (
            data[data >= pred] - pred
        ).sum() * target_quantile
        expected_pbl /= n_samples
        assert_almost_equal(expected_pbl, pbl)

    # 检查通过最小化常量预测分位数来恢复目标分位数的可行性
    def objective_func(x):
        constant_pred = np.full(n_samples, fill_value=x)
        return mean_pinball_loss(data, constant_pred, alpha=target_quantile)

    # 使用 Nelder-Mead 方法最小化目标函数
    result = optimize.minimize(objective_func, data.mean(), method="Nelder-Mead")
    assert result.success
    # 由于数据有限，最小值可能不唯一，因此设置较大的容差
    assert result.x == pytest.approx(best_pred, rel=1e-2)
    assert result.fun == pytest.approx(best_pbl)


# 测试虚拟的分位数参数调整
def test_dummy_quantile_parameter_tuning():
    # 集成测试，检查是否可以使用分位损失来调整分位数回归器的超参数
    # 这与前一个测试在概念上类似，但使用 scikit-learn 的估计器和评分 API
    n_samples = 1000
    # 使用种子 0 初始化一个随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个形状为 (n_samples, 5) 的正态分布随机数矩阵 X
    X = rng.normal(size=(n_samples, 5))  # Ignored
    # 生成一个包含 n_samples 个指数分布随机数的数组 y
    y = rng.exponential(size=n_samples)
    
    # 定义所有要计算的分位数列表
    all_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for alpha in all_quantiles:
        # 创建一个评分函数，用于计算负的均值分位数损失
        neg_mean_pinball_loss = make_scorer(
            mean_pinball_loss,
            alpha=alpha,
            greater_is_better=False,
        )
        # 使用 DummyRegressor 创建一个回归器，策略为分位数，分位数为 0.25
        regressor = DummyRegressor(strategy="quantile", quantile=0.25)
        # 使用 GridSearchCV 对 regressor 进行网格搜索，寻找最佳参数
        grid_search = GridSearchCV(
            regressor,
            param_grid=dict(quantile=all_quantiles),
            scoring=neg_mean_pinball_loss,
        ).fit(X, y)
    
        # 确保找到的最佳分位数参数与当前 alpha 接近
        assert grid_search.best_params_["quantile"] == pytest.approx(alpha)
# 定义测试函数，验证 mean_pinball_loss 和 mean_absolute_error 之间的关系
def test_pinball_loss_relation_with_mae():
    # 使用种子为714的随机数生成器
    rng = np.random.RandomState(714)
    # 设定样本数量
    n = 100
    # 生成真实值，符合正态分布
    y_true = rng.normal(size=n)
    # 生成预测值，为真实值的复制加上从均匀分布中随机采样的偏移量
    y_pred = y_true.copy() + rng.uniform(n)
    # 断言：mean_absolute_error 的两倍等于 mean_pinball_loss(alpha=0.5)
    assert (
        mean_absolute_error(y_true, y_pred)
        == mean_pinball_loss(y_true, y_pred, alpha=0.5) * 2
    )


# TODO(1.6): remove this test
# 参数化测试函数，用于测试 mean_squared_error 和 mean_squared_log_error 的特定情况
@pytest.mark.parametrize("metric", [mean_squared_error, mean_squared_log_error])
def test_mean_squared_deprecation_squared(metric):
    """Check the deprecation warning of the squared parameter"""
    # 提示消息："'squared' 在版本 1.4 中已弃用，并将在 1.6 中移除。"
    depr_msg = "'squared' is deprecated in version 1.4 and will be removed in 1.6."
    # 设置真实值和预测值
    y_true, y_pred = np.arange(10), np.arange(1, 11)
    # 使用 pytest 的 warn 机制捕获 FutureWarning，确保匹配到指定的提示消息
    with pytest.warns(FutureWarning, match=depr_msg):
        # 调用指定的 metric 函数，将 squared 参数设为 False，检查是否触发警告
        metric(y_true, y_pred, squared=False)


# TODO(1.6): remove this test
# 参数化测试函数，用于测试新旧函数之间的等价性
@pytest.mark.filterwarnings("ignore:'squared' is deprecated")
@pytest.mark.parametrize(
    "old_func, new_func",
    [
        (mean_squared_error, root_mean_squared_error),
        (mean_squared_log_error, root_mean_squared_log_error),
    ],
)
def test_rmse_rmsle_parameter(old_func, new_func):
    # 检查新的 rmse/rmsle 函数是否等价于旧的 mse/msle 函数加上 squared=False 参数
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    sw = np.arange(len(y_true))

    # 断言：新旧函数的结果应当接近
    expected = old_func(y_true, y_pred, squared=False)
    actual = new_func(y_true, y_pred)
    assert_allclose(expected, actual)

    # 断言：带有 sample_weight 参数的新旧函数的结果应当接近
    expected = old_func(y_true, y_pred, sample_weight=sw, squared=False)
    actual = new_func(y_true, y_pred, sample_weight=sw)
    assert_allclose(expected, actual)

    # 断言：带有 multioutput="raw_values" 参数的新旧函数的结果应当接近
    expected = old_func(y_true, y_pred, multioutput="raw_values", squared=False)
    actual = new_func(y_true, y_pred, multioutput="raw_values")
    assert_allclose(expected, actual)

    # 断言：带有 sample_weight 和 multioutput="raw_values" 参数的新旧函数的结果应当接近
    expected = old_func(
        y_true, y_pred, sample_weight=sw, multioutput="raw_values", squared=False
    )
    actual = new_func(y_true, y_pred, sample_weight=sw, multioutput="raw_values")
    assert_allclose(expected, actual)
```