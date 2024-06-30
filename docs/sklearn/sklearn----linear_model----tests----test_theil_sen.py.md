# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_theil_sen.py`

```
"""
Testing for Theil-Sen module (sklearn.linear_model.theil_sen)
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
import os                           # 导入操作系统相关模块
import re                           # 导入正则表达式模块
import sys                          # 导入系统相关模块
from contextlib import contextmanager  # 导入上下文管理器相关模块

import numpy as np                  # 导入NumPy库，并使用别名np
import pytest                       # 导入pytest测试框架
from numpy.testing import (         # 导入NumPy测试模块的多个函数
    assert_array_almost_equal,      # 导入数组几乎相等的断言函数
    assert_array_equal,             # 导入数组相等的断言函数
    assert_array_less,              # 导入数组小于的断言函数
)
from scipy.linalg import norm       # 从SciPy线性代数模块导入norm函数
from scipy.optimize import fmin_bfgs  # 从SciPy优化模块导入fmin_bfgs函数

from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.linear_model import LinearRegression, TheilSenRegressor  # 导入线性回归和Theil-Sen回归类
from sklearn.linear_model._theil_sen import (  # 导入Theil-Sen回归的一些内部函数
    _breakdown_point,               # 导入_breakdown_point函数
    _modified_weiszfeld_step,       # 导入_modified_weiszfeld_step函数
    _spatial_median,                # 导入_spatial_median函数
)
from sklearn.utils._testing import assert_almost_equal  # 导入近似相等的断言函数


@contextmanager
def no_stdout_stderr():             # 定义一个上下文管理器，用于禁止标准输出和标准错误输出
    old_stdout = sys.stdout         # 保存旧的标准输出
    old_stderr = sys.stderr         # 保存旧的标准错误输出
    with open(os.devnull, "w") as devnull:  # 打开/dev/null，用于写操作
        sys.stdout = devnull        # 将标准输出重定向到/dev/null
        sys.stderr = devnull        # 将标准错误输出重定向到/dev/null
        yield                       # 返回控制权给调用者
        devnull.flush()             # 刷新/dev/null
        sys.stdout = old_stdout     # 恢复旧的标准输出
        sys.stderr = old_stderr     # 恢复旧的标准错误输出


def gen_toy_problem_1d(intercept=True):  # 定义生成一维示例问题的函数，可选包含截距
    random_state = np.random.RandomState(0)  # 使用种子0创建随机数生成器
    # 线性模型 y = 3*x + N(2, 0.1**2)
    w = 3.0                          # 设置斜率
    if intercept:
        c = 2.0                      # 设置截距
        n_samples = 50               # 样本数量
    else:
        c = 0.1                      # 设置截距
        n_samples = 100              # 样本数量
    x = random_state.normal(size=n_samples)  # 生成正态分布的随机数作为自变量
    noise = 0.1 * random_state.normal(size=n_samples)  # 生成正态分布的噪声
    y = w * x + c + noise           # 根据线性关系生成因变量
    # 添加一些异常值
    if intercept:
        x[42], y[42] = (-2, 4)       # 添加特定位置的异常值
        x[43], y[43] = (-2.5, 8)     # 添加特定位置的异常值
        x[33], y[33] = (2.5, 1)      # 添加特定位置的异常值
        x[49], y[49] = (2.1, 2)      # 添加特定位置的异常值
    else:
        x[42], y[42] = (-2, 4)       # 添加特定位置的异常值
        x[43], y[43] = (-2.5, 8)     # 添加特定位置的异常值
        x[53], y[53] = (2.5, 1)      # 添加特定位置的异常值
        x[60], y[60] = (2.1, 2)      # 添加特定位置的异常值
        x[72], y[72] = (1.8, -7)     # 添加特定位置的异常值
    return x[:, np.newaxis], y, w, c  # 返回自变量、因变量、斜率和截距


def gen_toy_problem_2d():           # 定义生成二维示例问题的函数
    random_state = np.random.RandomState(0)  # 使用种子0创建随机数生成器
    n_samples = 100                  # 样本数量
    # 线性模型 y = 5*x_1 + 10*x_2 + N(1, 0.1**2)
    X = random_state.normal(size=(n_samples, 2))  # 生成正态分布的二维随机数作为自变量
    w = np.array([5.0, 10.0])        # 设置斜率向量
    c = 1.0                          # 设置截距
    noise = 0.1 * random_state.normal(size=n_samples)  # 生成正态分布的噪声
    y = np.dot(X, w) + c + noise     # 根据线性关系生成因变量
    # 添加一些异常值
    n_outliers = n_samples // 10     # 异常值数量
    ix = random_state.randint(0, n_samples, size=n_outliers)  # 生成随机索引
    y[ix] = 50 * random_state.normal(size=n_outliers)  # 替换部分因变量为大异常值
    return X, y, w, c                # 返回自变量矩阵、因变量向量、斜率向量和截距


def gen_toy_problem_4d():           # 定义生成四维示例问题的函数
    random_state = np.random.RandomState(0)  # 使用种子0创建随机数生成器
    n_samples = 10000                # 样本数量
    # 线性模型 y = 5*x_1 + 10*x_2 + 42*x_3 + 7*x_4 + N(1, 0.1**2)
    X = random_state.normal(size=(n_samples, 4))  # 生成正态分布的四维随机数作为自变量
    w = np.array([5.0, 10.0, 42.0, 7.0])  # 设置斜率向量
    c = 1.0                          # 设置截距
    noise = 0.1 * random_state.normal(size=n_samples)  # 生成正态分布的噪声
    y = np.dot(X, w) + c + noise     # 根据线性关系生成因变量
    # 添加一些异常值
    n_outliers = n_samples // 10     # 异常值数量
    ix = random_state.randint(0, n_samples, size=n_outliers)  # 生成随机索引
    y[ix] = 50 * random_state.normal(size=n_outliers)  # 替换部分因变量为大异常值
    return X, y, w, c                # 返回自变量矩阵、因变量向量、斜率向量和截距


def test_modweiszfeld_step_1d():    # 定义测试一维Modified Weiszfeld步骤的函数
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)  # 创建一个三行一列的数组作为自变量
    # 检查起始值是否是X的元素并且是解
    median = 2.0                    # 设置中位数初始值
    # 使用 _modified_weiszfeld_step 函数计算给定输入 X 和中值 median 的修改韦伊茨费尔德步骤的结果
    new_y = _modified_weiszfeld_step(X, median)
    # 断言新计算得到的 new_y 与中值 median 几乎相等
    assert_array_almost_equal(new_y, median)
    
    # 检查起始值 y 不是解
    y = 2.5
    # 使用 _modified_weiszfeld_step 函数计算给定输入 X 和起始值 y 的修改韦伊茨费尔德步骤的结果
    new_y = _modified_weiszfeld_step(X, y)
    # 断言中值 median 小于新计算得到的 new_y 的每个元素
    assert_array_less(median, new_y)
    # 断言新计算得到的 new_y 的每个元素都小于起始值 y
    assert_array_less(new_y, y)
    
    # 再次检查起始值 y 不是解，但是是输入 X 的一个元素
    y = 3.0
    # 使用 _modified_weiszfeld_step 函数计算给定输入 X 和起始值 y 的修改韦伊茨费尔德步骤的结果
    new_y = _modified_weiszfeld_step(X, y)
    # 断言中值 median 小于新计算得到的 new_y 的每个元素
    assert_array_less(median, new_y)
    # 断言新计算得到的 new_y 的每个元素都小于起始值 y
    assert_array_less(new_y, y)
    
    # 检查单个向量是否是单位向量
    # 创建一个形状为 (1, 3) 的 NumPy 数组 X，包含元素 [1.0, 2.0, 3.0]
    X = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    # 将 y 设为数组 X 的第一个元素，即 [1.0, 2.0, 3.0] 中的 [1.0, 2.0, 3.0]
    y = X[0]
    # 使用 _modified_weiszfeld_step 函数计算给定输入 X 和起始值 y 的修改韦伊茨费尔德步骤的结果
    new_y = _modified_weiszfeld_step(X, y)
    # 断言原始向量 y 等于新计算得到的 new_y
    assert_array_equal(y, new_y)
# 定义测试函数，用于测试二维情况下的修改后的 Weiszfeld 算法
def test_modweiszfeld_step_2d():
    # 创建一个二维 NumPy 数组 X，表示三个点的坐标
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    # 创建一个一维 NumPy 数组 y，表示初始点的坐标
    y = np.array([0.5, 0.5])
    # 调用 _modified_weiszfeld_step 函数进行第一次迭代计算新的坐标
    new_y = _modified_weiszfeld_step(X, y)
    # 断言新计算的坐标与预期的坐标接近
    assert_array_almost_equal(new_y, np.array([1 / 3, 2 / 3]))
    # 再次调用 _modified_weiszfeld_step 函数进行第二次迭代计算新的坐标
    new_y = _modified_weiszfeld_step(X, new_y)
    # 断言新计算的坐标与预期的坐标接近
    assert_array_almost_equal(new_y, np.array([0.2792408, 0.7207592]))
    # 设置 y 为一个已知的不动点坐标
    y = np.array([0.21132505, 0.78867497])
    # 再次调用 _modified_weiszfeld_step 函数进行计算，应返回与输入 y 接近的坐标
    new_y = _modified_weiszfeld_step(X, y)
    # 断言新计算的坐标与输入 y 接近
    assert_array_almost_equal(new_y, y)


# 定义测试函数，用于测试一维情况下的空间中位数计算
def test_spatial_median_1d():
    # 创建一个一维 NumPy 数组 X，表示三个点的坐标
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    # 设置真实的中位数
    true_median = 2.0
    # 调用 _spatial_median 函数计算空间中位数，并获取结果
    _, median = _spatial_median(X)
    # 断言计算得到的中位数与真实中位数接近
    assert_array_almost_equal(median, true_median)
    # 生成一个较大的问题实例，并检查在一维情况下是否能得到精确解
    random_state = np.random.RandomState(0)
    X = random_state.randint(100, size=(1000, 1))
    true_median = np.median(X.ravel())
    # 再次调用 _spatial_median 函数计算空间中位数，并获取结果
    _, median = _spatial_median(X)
    # 断言计算得到的中位数与真实中位数接近
    assert_array_equal(median, true_median)


# 定义测试函数，用于测试二维情况下的空间中位数计算
def test_spatial_median_2d():
    # 创建一个二维 NumPy 数组 X，表示三个点的坐标
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    # 调用 _spatial_median 函数计算空间中位数，并获取结果中的中位数
    _, median = _spatial_median(X, max_iter=100, tol=1.0e-6)

    # 定义成本函数，用于计算给定点 y 的总距离
    def cost_func(y):
        dists = np.array([norm(x - y) for x in X])
        return np.sum(dists)

    # 使用 BFGS 方法寻找 Fermat-Weber 位置问题的解
    fermat_weber = fmin_bfgs(cost_func, median, disp=False)
    # 断言计算得到的中位数与 Fermat-Weber 解接近
    assert_array_almost_equal(median, fermat_weber)
    # 检查当超过最大迭代次数时是否会发出警告信息
    warning_message = "Maximum number of iterations 30 reached in spatial median."
    with pytest.warns(ConvergenceWarning, match=warning_message):
        _spatial_median(X, max_iter=30, tol=0.0)


# 定义测试函数，用于测试一维情况下的 Theil-Sen 回归
def test_theil_sen_1d():
    # 生成一个简单的一维问题，包括输入 X, y, 真实权重 w 和截距 c
    X, y, w, c = gen_toy_problem_1d()
    # 使用最小二乘法拟合，检查其是否失败
    lstq = LinearRegression().fit(X, y)
    assert np.abs(lstq.coef_ - w) > 0.9
    # 使用 Theil-Sen 回归拟合，检查其是否成功
    theil_sen = TheilSenRegressor(random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


# 定义测试函数，用于测试一维情况下没有截距的 Theil-Sen 回归
def test_theil_sen_1d_no_intercept():
    # 生成一个简单的一维问题，包括输入 X, y, 真实权重 w 和截距 c（截距设置为 False）
    X, y, w, c = gen_toy_problem_1d(intercept=False)
    # 使用没有截距的最小二乘法拟合，检查其是否失败
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.abs(lstq.coef_ - w - c) > 0.5
    # 使用没有截距的 Theil-Sen 回归拟合，检查其是否成功
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w + c, 1)
    assert_almost_equal(theil_sen.intercept_, 0.0)
    # 针对 issue #18104 进行非回归测试
    theil_sen.score(X, y)


# 定义测试函数，用于测试二维情况下的 Theil-Sen 回归
def test_theil_sen_2d():
    # 生成一个简单的二维问题，包括输入 X, y, 真实权重 w 和截距 c
    X, y, w, c = gen_toy_problem_2d()
    # 使用最小二乘法拟合，检查其是否失败
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    # 使用 Theil-Sen 回归拟合，检查其是否成功
    theil_sen = TheilSenRegressor(max_subpopulation=1e3, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    # 使用 assert_array_almost_equal 函数来比较 Theil-Sen 回归的截距(intercept_)与 c 是否在精度为 1 的范围内接近
    assert_array_almost_equal(theil_sen.intercept_, c, 1)
# 测试计算中断点函数的正确性
def test_calc_breakdown_point():
    # 调用内部函数 _breakdown_point 计算中断点，期望值与理论值比较小于给定的误差容限
    bp = _breakdown_point(1e10, 2)
    assert np.abs(bp - 1 + 1 / (np.sqrt(2))) < 1.0e-6


# 参数化测试，验证异常情况下的输入参数是否能正确抛出异常
@pytest.mark.parametrize(
    "param, ExceptionCls, match",
    [
        (
            {"n_subsamples": 1},
            ValueError,
            re.escape("Invalid parameter since n_features+1 > n_subsamples (2 > 1)"),
        ),
        (
            {"n_subsamples": 101},
            ValueError,
            re.escape("Invalid parameter since n_subsamples > n_samples (101 > 50)"),
        ),
    ],
)
def test_checksubparams_invalid_input(param, ExceptionCls, match):
    # 生成 1 维的示例问题数据，初始化 TheilSenRegressor 模型
    X, y, w, c = gen_toy_problem_1d()
    theil_sen = TheilSenRegressor(**param, random_state=0)
    # 使用 pytest 检测是否抛出指定的异常类型和匹配的异常信息
    with pytest.raises(ExceptionCls, match=match):
        theil_sen.fit(X, y)


# 测试当 n_subsamples 小于样本数时的行为
def test_checksubparams_n_subsamples_if_less_samples_than_features():
    random_state = np.random.RandomState(0)
    n_samples, n_features = 10, 20
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    # 使用 TheilSenRegressor 模型，期望抛出 ValueError 异常
    theil_sen = TheilSenRegressor(n_subsamples=9, random_state=0)
    with pytest.raises(ValueError):
        theil_sen.fit(X, y)


# 测试子样本情况
def test_subpopulation():
    # 生成 4 维的示例问题数据，拟合 TheilSenRegressor 模型并检查系数和截距的近似相等性
    X, y, w, c = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(max_subpopulation=250, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


# 测试子样本数等于样本数的情况
def test_subsamples():
    # 生成 4 维的示例问题数据，使用 TheilSenRegressor 和 LinearRegression 模型进行拟合，并验证系数的精确性
    X, y, w, c = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(n_subsamples=X.shape[0], random_state=0).fit(X, y)
    lstq = LinearRegression().fit(X, y)
    # 检查 TheilSenRegressor 模型的系数与 LinearRegression 模型的系数是否几乎相等
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 9)


# 测试详细程度设置
def test_verbosity():
    # 生成 1 维的示例问题数据，验证 TheilSenRegressor 在详细模式下是否能正常工作
    X, y, w, c = gen_toy_problem_1d()
    # 使用 no_stdout_stderr 上下文管理器确保不输出任何标准输出和错误信息
    with no_stdout_stderr():
        TheilSenRegressor(verbose=True, random_state=0).fit(X, y)
        TheilSenRegressor(verbose=True, max_subpopulation=10, random_state=0).fit(X, y)


# 测试 Theil-Sen 回归在并行计算下的表现
def test_theil_sen_parallel():
    # 生成 2 维的示例问题数据，验证 TheilSenRegressor 在并行计算下的系数和截距是否几乎相等
    X, y, w, c = gen_toy_problem_2d()
    # 拟合 LinearRegression 模型并检查系数的欧几里得范数是否大于给定阈值
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    # 使用 TheilSenRegressor 拟合模型，并验证系数和截距的近似相等性
    theil_sen = TheilSenRegressor(n_jobs=2, random_state=0, max_subpopulation=2e3).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)


# 测试样本数小于特征数的情况
def test_less_samples_than_features():
    random_state = np.random.RandomState(0)
    n_samples, n_features = 10, 20
    X = random_state.normal(size=(n_samples, n_features))
    y = random_state.normal(size=n_samples)
    # 验证 TheilSenRegressor 在 fit_intercept=False 时是否回退到 LinearRegression
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 12)
    # 使用 fit_intercept=True 的情况进行检查。由于截距(intercept)的计算方式不同，
    # 这个结果不会等于最小二乘法的解。
    theil_sen = TheilSenRegressor(fit_intercept=True, random_state=0).fit(X, y)
    # 使用 Theil-Sen 回归模型对输入数据 X 进行预测
    y_pred = theil_sen.predict(X)
    # 检查预测结果 y_pred 是否与真实值 y 几乎相等，精确到小数点后12位
    assert_array_almost_equal(y_pred, y, 12)
# TODO(1.8): Remove
# 定义一个测试函数 test_copy_X_deprecated，用于测试在复制数据（X 变量）时的过时功能
def test_copy_X_deprecated():
    # 调用 gen_toy_problem_1d 函数生成一个简单的数据集 X, y，并忽略其余两个返回值
    X, y, _, _ = gen_toy_problem_1d()
    # 创建 TheilSenRegressor 对象，设置 copy_X 参数为 True，并指定随机种子为 0
    theil_sen = TheilSenRegressor(copy_X=True, random_state=0)
    # 使用 pytest.warns 来捕获未来警告 FutureWarning，并匹配字符串 "`copy_X` was deprecated"
    with pytest.warns(FutureWarning, match="`copy_X` was deprecated"):
        # 对 TheilSenRegressor 对象进行拟合，传入数据集 X 和 y
        theil_sen.fit(X, y)
```