# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\tests\test_gpr.py`

```
"""Testing for Gaussian process regression"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的模块和库
import re  # 导入正则表达式模块
import sys  # 导入系统相关模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from scipy.optimize import approx_fprime  # 导入求近似梯度的函数

from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.gaussian_process import GaussianProcessRegressor  # 导入高斯过程回归模型
from sklearn.gaussian_process.kernels import (  # 导入高斯过程回归所需的核函数
    RBF,
    DotProduct,
    ExpSineSquared,
    WhiteKernel,
)
from sklearn.gaussian_process.kernels import (  # 导入常数核函数，并重命名为C
    ConstantKernel as C,
)
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel  # 导入自定义的MiniSeqKernel类
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_less,
)


def f(x):
    return x * np.sin(x)  # 定义一个函数f(x) = x * sin(x)


X = np.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T  # 创建输入数据X
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T  # 创建另一个输入数据X2
y = f(X).ravel()  # 计算目标值y，将其展平为一维数组

fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")  # 创建一个固定长度尺度的RBF核函数
kernels = [  # 创建多个核函数对象组成的列表
    RBF(length_scale=1.0),  # 基本RBF核函数
    fixed_kernel,  # 固定长度尺度的RBF核函数
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),  # 指定长度尺度范围的RBF核函数
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),  # 常数核函数乘以RBF核函数的组合
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    + C(1e-5, (1e-5, 1e2)),  # 复合核函数，包括常数核函数和RBF核函数以及白噪声核函数
    C(0.1, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    + C(1e-5, (1e-5, 1e2)),  # 另一种复合核函数
]
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]  # 选择非固定长度尺度的核函数


@pytest.mark.parametrize("kernel", kernels)
def test_gpr_interpolation(kernel):
    if sys.maxsize <= 2**32:
        pytest.xfail("This test may fail on 32 bit Python")  # 如果是32位Python环境，则标记为预期失败

    # Test the interpolating property for different kernels.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)  # 创建高斯过程回归对象并拟合数据
    y_pred, y_cov = gpr.predict(X, return_cov=True)  # 预测均值和协方差

    assert_almost_equal(y_pred, y)  # 断言预测均值与真实值的近似相等
    assert_almost_equal(np.diag(y_cov), 0.0)  # 断言协方差矩阵的对角线元素接近于零


def test_gpr_interpolation_structured():
    # Test the interpolating property for different kernels.
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")  # 创建自定义MiniSeqKernel对象
    X = ["A", "B", "C"]  # 创建结构化输入数据X
    y = np.array([1, 2, 3])  # 创建目标值y
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)  # 创建高斯过程回归对象并拟合数据
    y_pred, y_cov = gpr.predict(X, return_cov=True)  # 预测均值和协方差

    assert_almost_equal(
        kernel(X, eval_gradient=True)[1].ravel(), (1 - np.eye(len(X))).ravel()
    )  # 断言核函数的评估梯度与指定的值接近
    assert_almost_equal(y_pred, y)  # 断言预测均值与真实值的近似相等
    assert_almost_equal(np.diag(y_cov), 0.0)  # 断言协方差矩阵的对角线元素接近于零


@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_lml_improving(kernel):
    if sys.maxsize <= 2**32:
        pytest.xfail("This test may fail on 32 bit Python")  # 如果是32位Python环境，则标记为预期失败

    # Test that hyperparameter-tuning improves log-marginal likelihood.
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)  # 创建高斯过程回归对象并拟合数据
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
        kernel.theta
    )  # 断言超参数调优后的对数边际似然大于初始核函数的对数边际似然


@pytest.mark.parametrize("kernel", kernels)
def test_lml_precomputed(kernel):
    # Test that lml of optimized kernel is stored correctly.
    # 使用指定的核函数训练高斯过程回归模型，将模型拟合到数据集 (X, y)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    # 断言模型在指定参数下的对数边际似然等于默认参数下的对数边际似然（使用 pytest 的近似匹配检查）
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) == pytest.approx(
        gpr.log_marginal_likelihood()
    )
@pytest.mark.parametrize("kernel", kernels)
# 使用pytest的参数化标记，对每个参数化的kernel执行测试
def test_lml_without_cloning_kernel(kernel):
    # 测试优化后的kernel的lml是否正确存储
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    # 创建一个与gpr.kernel_.theta相同形状的全1数组作为输入theta
    input_theta = np.ones(gpr.kernel_.theta.shape, dtype=np.float64)

    # 调用log_marginal_likelihood方法，不克隆kernel，计算lml
    gpr.log_marginal_likelihood(input_theta, clone_kernel=False)
    # 断言kernel_.theta与输入theta几乎相等，精度为7位小数
    assert_almost_equal(gpr.kernel_.theta, input_theta, 7)


@pytest.mark.parametrize("kernel", non_fixed_kernels)
# 使用pytest的参数化标记，对每个非固定的kernel执行测试
def test_converged_to_local_maximum(kernel):
    # 测试超参数优化后是否处于局部最大值
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    # 计算lml和lml的梯度
    lml, lml_gradient = gpr.log_marginal_likelihood(gpr.kernel_.theta, True)

    # 断言lml的梯度的绝对值小于1e-4，或者theta处于边界值
    assert np.all(
        (np.abs(lml_gradient) < 1e-4)
        | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 0])
        | (gpr.kernel_.theta == gpr.kernel_.bounds[:, 1])
    )


@pytest.mark.parametrize("kernel", non_fixed_kernels)
# 使用pytest的参数化标记，对每个非固定的kernel执行测试
def test_solution_inside_bounds(kernel):
    # 测试超参数优化后是否仍在边界内
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    # 获取kernel的边界
    bounds = gpr.kernel_.bounds
    max_ = np.finfo(gpr.kernel_.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_

    # 断言kernel.theta比边界的下限稍大
    assert_array_less(bounds[:, 0], gpr.kernel_.theta + tiny)
    # 断言kernel.theta比边界的上限稍小
    assert_array_less(gpr.kernel_.theta, bounds[:, 1] + tiny)


@pytest.mark.parametrize("kernel", kernels)
# 使用pytest的参数化标记，对每个kernel执行测试
def test_lml_gradient(kernel):
    # 比较lml的解析梯度和数值梯度
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    # 计算lml和lml的梯度
    lml, lml_gradient = gpr.log_marginal_likelihood(kernel.theta, True)
    lml_gradient_approx = approx_fprime(
        kernel.theta, lambda theta: gpr.log_marginal_likelihood(theta, False), 1e-10
    )

    # 断言解析梯度与数值梯度几乎相等，精度为3位小数
    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)


@pytest.mark.parametrize("kernel", kernels)
# 使用pytest的参数化标记，对每个kernel执行测试
def test_prior(kernel):
    # 测试GP先验是否具有均值为0和相同方差
    gpr = GaussianProcessRegressor(kernel=kernel)

    # 进行预测，返回均值和协方差
    y_mean, y_cov = gpr.predict(X, return_cov=True)

    # 断言均值几乎为0，精度为5位小数
    assert_almost_equal(y_mean, 0, 5)
    if len(gpr.kernel.theta) > 1:
        # XXX: quite hacky, works only for current kernels
        # 断言协方差的对角线元素几乎等于指数形式的第一个theta元素，精度为5位小数
        assert_almost_equal(np.diag(y_cov), np.exp(kernel.theta[0]), 5)
    else:
        # 断言协方差的对角线元素几乎等于1，精度为5位小数
        assert_almost_equal(np.diag(y_cov), 1, 5)


@pytest.mark.parametrize("kernel", kernels)
# 使用pytest的参数化标记，对每个kernel执行测试
def test_sample_statistics(kernel):
    # 测试从GP抽样得到的样本统计量是否正确
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    # 进行预测，返回均值和协方差
    y_mean, y_cov = gpr.predict(X2, return_cov=True)

    # 从GP中抽取300000个样本
    samples = gpr.sample_y(X2, 300000)

    # 断言均值几乎等于样本的均值，精度为1
    assert_almost_equal(y_mean, np.mean(samples, 1), 1)
    # 断言协方差的对角线元素归一化后几乎等于样本方差的归一化，精度为1
    assert_almost_equal(
        np.diag(y_cov) / np.diag(y_cov).max(),
        np.var(samples, 1) / np.diag(y_cov).max(),
        1,
    )


def test_no_optimizer():
    # 空测试函数，用于测试不使用优化器的情况
    pass
    # 创建 RBF 核函数对象，设定参数为 1.0
    kernel = RBF(1.0)
    # 创建高斯过程回归器对象，使用上面创建的 RBF 核函数，优化器设为 None
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(X, y)
    # 断言检查：验证高斯过程回归器对象中的核函数参数 theta 的指数函数值是否等于 1.0
    assert np.exp(gpr.kernel_.theta) == 1.0
@pytest.mark.parametrize("kernel", kernels)
@pytest.mark.parametrize("target", [y, np.ones(X.shape[0], dtype=np.float64)])
def test_predict_cov_vs_std(kernel, target):
    if sys.maxsize <= 2**32:
        # 如果 Python 是 32 位版本，则标记此测试为预期失败
        pytest.xfail("This test may fail on 32 bit Python")

    # 使用指定的核函数训练高斯过程回归模型
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    # 预测输出的均值和协方差
    y_mean, y_cov = gpr.predict(X2, return_cov=True)
    # 预测输出的均值和标准差
    y_mean, y_std = gpr.predict(X2, return_std=True)
    # 断言预测的标准差是否与协方差对角线一致
    assert_almost_equal(np.sqrt(np.diag(y_cov)), y_std)


def test_anisotropic_kernel():
    # 测试高斯过程回归能否识别有意义的各向异性长度尺度
    # 学习一个函数，在一个维度上变化的速度比另一个维度慢十倍
    # 相应的长度尺度应该至少相差 5 倍
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (50, 2))
    y = X[:, 0] + 0.1 * X[:, 1]

    # 使用指定的 RBF 核函数训练高斯过程回归模型
    kernel = RBF([1.0, 1.0])
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    # 断言第二个维度的长度尺度大于第一个维度的长度尺度的 5 倍
    assert np.exp(gpr.kernel_.theta[1]) > np.exp(gpr.kernel_.theta[0]) * 5


def test_random_starts():
    # 测试增加 GP 拟合的随机启动次数是否仅增加所选择的 theta 的对数边际似然
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (
        np.sin(X).sum(axis=1)
        + np.sin(3 * X).sum(axis=1)
        + rng.normal(scale=0.1, size=n_samples)
    )

    # 定义复合核函数，包含 C、RBF 和 WhiteKernel 组合
    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1.0] * n_features, length_scale_bounds=[(1e-4, 1e2)] * n_features
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e1))
    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        # 使用指定的核函数和随机种子训练高斯过程回归模型
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=0,
        ).fit(X, y)
        # 计算当前模型的对数边际似然
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        # 断言当前的对数边际似然大于上一次迭代的对数边际似然
        assert lml > last_lml - np.finfo(np.float32).eps
        last_lml = lml


@pytest.mark.parametrize("kernel", kernels)
def test_y_normalization(kernel):
    """
    Test normalization of the target values in GP

    Fitting non-normalizing GP on normalized y and fitting normalizing GP
    on unnormalized y should yield identical results. Note that, here,
    'normalized y' refers to y that has been made zero mean and unit
    variance.

    """

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    # 在标准化后的 y 上拟合非标准化的 GP
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y_norm)

    # 在未标准化的 y 上拟合标准化的 GP
    gpr_norm = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr_norm.fit(X, y)

    # 比较预测的均值、标准差和协方差
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)
    y_pred = y_pred * y_std + y_mean
    y_pred_std = y_pred_std * y_std
    # 使用高斯过程回归模型 gpr_norm 对输入数据 X2 进行预测，返回预测值和标准化的预测标准差
    y_pred_norm, y_pred_std_norm = gpr_norm.predict(X2, return_std=True)

    # 断言：验证非标准化预测值 y_pred 与标准化预测值 y_pred_norm 几乎相等
    assert_almost_equal(y_pred, y_pred_norm)

    # 断言：验证非标准化预测标准差 y_pred_std 与标准化预测标准差 y_pred_std_norm 几乎相等
    assert_almost_equal(y_pred_std, y_pred_std_norm)

    # 使用高斯过程回归模型 gpr 对输入数据 X2 进行预测，返回预测值和协方差矩阵
    _, y_cov = gpr.predict(X2, return_cov=True)

    # 将协方差矩阵 y_cov 乘以非标准化的预测标准差的平方 y_std**2，得到调整后的协方差矩阵
    y_cov = y_cov * y_std**2

    # 使用高斯过程回归模型 gpr_norm 对输入数据 X2 进行预测，返回预测值和标准化的协方差矩阵
    _, y_cov_norm = gpr_norm.predict(X2, return_cov=True)

    # 断言：验证调整后的协方差矩阵 y_cov 与标准化的协方差矩阵 y_cov_norm 几乎相等
    assert_almost_equal(y_cov, y_cov_norm)
# 定义一个测试函数，用于验证当 normalize_y=True 时，高斯过程能否对方差显著大于单位的训练数据进行合理拟合。
def test_large_variance_y():
    """
    Here we test that, when noramlize_y=True, our GP can produce a
    sensible fit to training data whose variance is significantly
    larger than unity. This test was made in response to issue #15612.

    GP predictions are verified against predictions that were made
    using GPy which, here, is treated as the 'gold standard'. Note that we
    only investigate the RBF kernel here, as that is what was used in the
    GPy implementation.

    The following code can be used to recreate the GPy data:

    --------------------------------------------------------------------------
    import GPy

    kernel_gpy = GPy.kern.RBF(input_dim=1, lengthscale=1.)
    gpy = GPy.models.GPRegression(X, np.vstack(y_large), kernel_gpy)
    gpy.optimize()
    y_pred_gpy, y_var_gpy = gpy.predict(X2)
    y_pred_std_gpy = np.sqrt(y_var_gpy)
    --------------------------------------------------------------------------
    """

    # 将训练数据的方差放大十倍，以便进行测试
    y_large = 10 * y

    # 使用 RBF 核函数和 normalize_y=True 创建高斯过程回归器对象
    RBF_params = {"length_scale": 1.0}
    kernel = RBF(**RBF_params)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_large)
    y_pred, y_pred_std = gpr.predict(X2, return_std=True)

    # 'Gold standard' 均值预测值来自于 GPy
    y_pred_gpy = np.array(
        [15.16918303, -27.98707845, -39.31636019, 14.52605515, 69.18503589]
    )

    # 'Gold standard' 标准差预测值来自于 GPy
    y_pred_std_gpy = np.array(
        [7.78860962, 3.83179178, 0.63149951, 0.52745188, 0.86170042]
    )

    # 基于数值实验，合理预期我们的高斯过程的均值预测能够与 GPy 的预测结果在 7% 的相对容差内
    assert_allclose(y_pred, y_pred_gpy, rtol=0.07, atol=0)

    # 基于数值实验，合理预期我们的高斯过程的标准差预测能够与 GPy 的预测结果在 15% 的相对容差内
    assert_allclose(y_pred_std, y_pred_std_gpy, rtol=0.15, atol=0)


# 定义一个测试函数，验证高斯过程回归器能处理多维目标值
def test_y_multioutput():
    # 将 y 扩展为二维数组，包含原始 y 和 y 的两倍
    y_2d = np.vstack((y, y * 2)).T

    # 使用 RBF 核函数和 normalize_y=False 创建高斯过程回归器对象，优化器设置为 None
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr.fit(X, y)

    # 对于二维目标值，创建另一个高斯过程回归器对象，与上面保持一致
    gpr_2d = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
    gpr_2d.fit(X, y_2d)

    # 预测均值和标准差，同时返回协方差
    y_pred_1d, y_std_1d = gpr.predict(X2, return_std=True)
    y_pred_2d, y_std_2d = gpr_2d.predict(X2, return_std=True)
    _, y_cov_1d = gpr.predict(X2, return_cov=True)
    _, y_cov_2d = gpr_2d.predict(X2, return_cov=True)

    # 验证一维和二维预测均值的一致性
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 0])
    # 验证一维和二维预测均值的一致性，第二维的预测应为第一维的两倍
    assert_almost_equal(y_pred_1d, y_pred_2d[:, 1] / 2)

    # 标准差和协方差不依赖于输出维度
    # 对于每个目标变量，检查标准化后的结果是否准确
    for target in range(y_2d.shape[1]):
        assert_almost_equal(y_std_1d, y_std_2d[..., target])
        assert_almost_equal(y_cov_1d, y_cov_2d[..., target])

    # 从单变量的高斯过程模型中抽样得到一维输出
    y_sample_1d = gpr.sample_y(X2, n_samples=10)
    # 从多变量的高斯过程模型中抽样得到二维输出
    y_sample_2d = gpr_2d.sample_y(X2, n_samples=10)

    # 检查抽样结果的形状是否正确
    assert y_sample_1d.shape == (5, 10)
    assert y_sample_2d.shape == (5, 2, 10)
    # 只有第一个目标变量的抽样结果是相等的
    assert_almost_equal(y_sample_1d, y_sample_2d[:, 0, :])

    # 测试超参数优化
    for kernel in kernels:
        # 创建单变量高斯过程回归器，使用指定的核函数并标准化目标值
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        # 对数据进行拟合
        gpr.fit(X, y)

        # 创建多变量高斯过程回归器，使用相同的核函数并标准化目标值
        gpr_2d = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        # 对数据进行拟合，将目标值堆叠成二维
        gpr_2d.fit(X, np.vstack((y, y)).T)

        # 检查两个回归器使用的核函数的超参数是否接近（精确到小数点后四位）
        assert_almost_equal(gpr.kernel_.theta, gpr_2d.kernel_.theta, 4)
# 使用 pytest 的参数化功能，依次测试非固定的内核列表中的每个内核
@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_custom_optimizer(kernel):
    # 测试高斯过程回归器能够使用外部定义的优化器

    # 定义一个虚拟的优化器，简单地测试 50 个随机超参数
    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(0)
        theta_opt, func_min = initial_theta, obj_func(
            initial_theta, eval_gradient=False
        )
        for _ in range(50):
            # 生成随机的超参数 theta，确保在给定的边界范围内
            theta = np.atleast_1d(
                rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
            )
            # 计算目标函数在当前超参数下的值
            f = obj_func(theta, eval_gradient=False)
            # 如果当前值比记录的最小值还小，则更新最优超参数和最小值
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    # 创建一个带有自定义优化器的高斯过程回归器对象
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
    # 使用数据 X, y 进行拟合
    gpr.fit(X, y)

    # 断言：检查优化器是否改进了边际似然
    assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
        gpr.kernel.theta
    )


def test_gpr_correct_error_message():
    # 测试：确保高斯过程回归器能够正确处理错误消息

    # 定义输入数据 X 和目标值 y
    X = np.arange(12).reshape(6, -1)
    y = np.ones(6)
    # 使用 DotProduct 内核创建高斯过程回归器
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    # 创建期望的错误消息
    message = (
        "The kernel, %s, is not returning a "
        "positive definite matrix. Try gradually increasing "
        "the 'alpha' parameter of your "
        "GaussianProcessRegressor estimator." % kernel
    )
    # 使用 pytest 来断言引发特定类型的异常，并且错误消息与预期的一致
    with pytest.raises(np.linalg.LinAlgError, match=re.escape(message)):
        gpr.fit(X, y)


@pytest.mark.parametrize("kernel", kernels)
def test_duplicate_input(kernel):
    # 测试：确保高斯过程回归器能够处理相同输入不同输出的情况

    # 创建两个不同的高斯过程回归器，使用相同的内核和 alpha 值
    gpr_equal_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
    gpr_similar_inputs = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)

    # 扩展输入数据 X 和目标值 y，确保存在相同的输入
    X_ = np.vstack((X, X[0]))
    y_ = np.hstack((y, y[0] + 1))
    gpr_equal_inputs.fit(X_, y_)

    # 对输入数据 X 添加微小扰动，以确保输入几乎相似但输出不同
    X_ = np.vstack((X, X[0] + 1e-15))
    y_ = np.hstack((y, y[0] + 1))
    gpr_similar_inputs.fit(X_, y_)

    # 使用新的测试数据 X_test 进行预测，并获取预测结果和标准差
    X_test = np.linspace(0, 10, 100)[:, None]
    y_pred_equal, y_std_equal = gpr_equal_inputs.predict(X_test, return_std=True)
    y_pred_similar, y_std_similar = gpr_similar_inputs.predict(X_test, return_std=True)

    # 断言：检查两个模型的预测值和标准差是否几乎相等
    assert_almost_equal(y_pred_equal, y_pred_similar)
    assert_almost_equal(y_std_equal, y_std_similar)


def test_no_fit_default_predict():
    # 测试：确保在没有拟合的情况下，默认的高斯过程回归器预测不会出错

    # 定义默认的内核
    default_kernel = C(1.0, constant_value_bounds="fixed") * RBF(
        1.0, length_scale_bounds="fixed"
    )
    # 创建两个高斯过程回归器对象，一个默认内核，一个指定内核
    gpr1 = GaussianProcessRegressor()
    _, y_std1 = gpr1.predict(X, return_std=True)
    _, y_cov1 = gpr1.predict(X, return_cov=True)

    gpr2 = GaussianProcessRegressor(kernel=default_kernel)
    _, y_std2 = gpr2.predict(X, return_std=True)
    _, y_cov2 = gpr2.predict(X, return_cov=True)

    # 断言：检查默认内核下两个模型的预测标准差和协方差是否几乎相等
    assert_array_almost_equal(y_std1, y_std2)
    assert_array_almost_equal(y_cov1, y_cov2)


def test_warning_bounds():
    # 待实现的测试函数，目前没有内容，仅用作占位符
    pass
    # 创建一个 RBF 核函数，指定长度尺度的范围为 [1e-5, 1e-3]
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    
    # 创建一个高斯过程回归器，使用上述定义的核函数
    gpr = GaussianProcessRegressor(kernel=kernel)
    
    # 定义警告信息，用于匹配收敛警告
    warning_message = (
        "The optimal value found for dimension 0 of parameter "
        "length_scale is close to the specified upper bound "
        "0.001. Increasing the bound and calling fit again may "
        "find a better value."
    )
    
    # 使用 pytest 模块捕获 ConvergenceWarning 类型的警告信息，并检查是否匹配特定的警告消息
    with pytest.warns(ConvergenceWarning, match=warning_message):
        # 使用给定的数据 X 和 y 训练高斯过程回归器
        gpr.fit(X, y)
    
    # 创建一个包含白噪声核和 RBF 核的和核函数，各自指定参数范围
    kernel_sum = WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) + RBF(
        length_scale_bounds=[1e3, 1e5]
    )
    
    # 创建一个高斯过程回归器，使用和核函数
    gpr_sum = GaussianProcessRegressor(kernel=kernel_sum)
    
    # 使用 warnings 模块捕获所有警告信息
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # 设置警告过滤器，始终捕获警告信息
        gpr_sum.fit(X, y)  # 使用给定的数据 X 和 y 训练高斯过程回归器
    
        # 断言捕获到两条警告信息
        assert len(record) == 2
    
        # 断言第一条警告信息的类是 ConvergenceWarning
        assert issubclass(record[0].category, ConvergenceWarning)
        # 断言第一条警告信息的消息内容匹配特定字符串
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k1__noise_level is close to the "
            "specified upper bound 0.001. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )
    
        # 断言第二条警告信息的类是 ConvergenceWarning
        assert issubclass(record[1].category, ConvergenceWarning)
        # 断言第二条警告信息的消息内容匹配特定字符串
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k2__length_scale is close to the "
            "specified lower bound 1000.0. "
            "Decreasing the bound and calling "
            "fit again may find a better value."
        )
    
    # 使用 np.tile 函数对数据 X 进行复制扩展
    X_tile = np.tile(X, 2)
    
    # 创建一个 RBF 核函数，指定长度尺度为 [1.0, 2.0] 的范围为 [1e1, 1e2]
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[1e1, 1e2])
    
    # 创建一个高斯过程回归器，使用上述定义的核函数
    gpr_dims = GaussianProcessRegressor(kernel=kernel_dims)
    
    # 使用 warnings 模块捕获所有警告信息
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # 设置警告过滤器，始终捕获警告信息
        gpr_dims.fit(X_tile, y)  # 使用给定的数据 X_tile 和 y 训练高斯过程回归器
    
        # 断言捕获到两条警告信息
        assert len(record) == 2
    
        # 断言第一条警告信息的类是 ConvergenceWarning
        assert issubclass(record[0].category, ConvergenceWarning)
        # 断言第一条警告信息的消息内容匹配特定字符串
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "length_scale is close to the "
            "specified lower bound 10.0. "
            "Decreasing the bound and calling "
            "fit again may find a better value."
        )
    
        # 断言第二条警告信息的类是 ConvergenceWarning
        assert issubclass(record[1].category, ConvergenceWarning)
        # 断言第二条警告信息的消息内容匹配特定字符串
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 1 of parameter "
            "length_scale is close to the "
            "specified lower bound 10.0. "
            "Decreasing the bound and calling "
            "fit again may find a better value."
        )
def test_bound_check_fixed_hyperparameter():
    # Regression test for issue #17943
    # Check that having a hyperparameter with fixed bounds doesn't cause an
    # error
    
    # 定义一个长期平滑上升趋势的核函数 k1
    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    
    # 定义一个季节性成分的核函数 k2，其中周期性边界设置为 "fixed"
    k2 = ExpSineSquared(
        length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed"
    )  # seasonal component
    
    # 将两个核函数相加形成最终的核函数
    kernel = k1 + k2
    
    # 使用 GaussianProcessRegressor 拟合数据
    GaussianProcessRegressor(kernel=kernel).fit(X, y)


@pytest.mark.parametrize("kernel", kernels)
def test_constant_target(kernel):
    """Check that the std. dev. is affected to 1 when normalizing a constant
    feature.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/18318
    NaN where affected to the target when scaling due to null std. dev. with
    constant target.
    """
    
    # 创建一个全为 1 的常数目标值 y_constant
    y_constant = np.ones(X.shape[0], dtype=np.float64)
    
    # 使用指定的核函数和标准化目标值的参数创建 GaussianProcessRegressor 对象
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    
    # 使用 y_constant 拟合模型
    gpr.fit(X, y_constant)
    
    # 断言拟合后的训练集标准差为 1.0
    assert gpr._y_train_std == pytest.approx(1.0)
    
    # 使用拟合后的模型预测结果和协方差
    y_pred, y_cov = gpr.predict(X, return_cov=True)
    
    # 断言预测值与 y_constant 相近
    assert_allclose(y_pred, y_constant)
    
    # 设置 atol 因为我们要与零比较
    assert_allclose(np.diag(y_cov), 0.0, atol=1e-9)
    
    # 测试多目标数据
    n_samples, n_targets = X.shape[0], 2
    rng = np.random.RandomState(0)
    
    # 创建包含非常数目标和常数目标的合成数据集 y
    y = np.concatenate(
        [
            rng.normal(size=(n_samples, 1)),  # non-constant target
            np.full(shape=(n_samples, 1), fill_value=2),  # constant target
        ],
        axis=1,
    )
    
    # 使用合成数据集 y 拟合模型
    gpr.fit(X, y)
    
    # 使用拟合后的模型预测结果和协方差
    Y_pred, Y_cov = gpr.predict(X, return_cov=True)
    
    # 断言第二个目标的预测值为 2
    assert_allclose(Y_pred[:, 1], 2)
    
    # 设置 atol 因为我们要与零比较
    assert_allclose(np.diag(Y_cov[..., 1]), 0.0, atol=1e-9)
    
    # 断言预测结果的形状符合预期
    assert Y_pred.shape == (n_samples, n_targets)
    assert Y_cov.shape == (n_samples, n_samples, n_targets)


def test_gpr_consistency_std_cov_non_invertible_kernel():
    """Check the consistency between the returned std. dev. and the covariance.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19936
    Inconsistencies were observed when the kernel cannot be inverted (or
    numerically stable).
    """
    
    # 创建一个复杂的核函数，包含常数项和 RBF 核以及 WhiteKernel 噪声
    kernel = C(8.98576054e05, (1e-12, 1e12)) * RBF(
        [5.91326520e02, 1.32584051e03], (1e-12, 1e12)
    ) + WhiteKernel(noise_level=1e-5)
    
    # 使用指定的核函数和其他参数创建 GaussianProcessRegressor 对象
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None)
    
    # 定义训练数据 X_train 和目标值 y_train
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.54919334, -0.77459667],
            [-1.54919334, 0.0],
            [0.0, -1.54919334],
            [0.77459667, 0.77459667],
            [-0.77459667, 1.54919334],
        ]
    )
    y_train = np.array(
        [
            [-2.14882017e-10],
            [-4.66975823e00],
            [4.01823986e00],
            [-1.30303674e00],
            [-1.35760156e00],
            [3.31215668e00],
        ]
    )
    
    # 使用训练数据拟合模型
    gpr.fit(X_train, y_train)
    # 创建一个包含四个数据点的二维数组 X_test，用于进行高斯过程回归的预测
    X_test = np.array(
        [
            [-1.93649167, -1.93649167],   # 第一个数据点的坐标
            [1.93649167, -1.93649167],    # 第二个数据点的坐标
            [-1.93649167, 1.93649167],    # 第三个数据点的坐标
            [1.93649167, 1.93649167],     # 第四个数据点的坐标
        ]
    )
    
    # 使用高斯过程回归模型 gpr 对 X_test 进行预测，并返回预测值 pred1 和标准差 std
    pred1, std = gpr.predict(X_test, return_std=True)
    
    # 使用高斯过程回归模型 gpr 对 X_test 进行预测，并返回预测值 pred2 和协方差矩阵 cov
    pred2, cov = gpr.predict(X_test, return_cov=True)
    
    # 使用断言检查 std 是否与协方差矩阵 cov 的对角线元素的平方根相近，相对误差不超过 1e-5
    assert_allclose(std, np.sqrt(np.diagonal(cov)), rtol=1e-5)
# 使用 pytest.mark.parametrize 装饰器为 test_gpr_fit_error 函数添加参数化测试
@pytest.mark.parametrize(
    "params, TypeError, err_msg",
    [
        (
            {"alpha": np.zeros(100)},
            ValueError,
            "alpha must be a scalar or an array with same number of entries as y",
        ),
        (
            {
                "kernel": WhiteKernel(noise_level_bounds=(-np.inf, np.inf)),
                "n_restarts_optimizer": 2,
            },
            ValueError,
            "requires that all bounds are finite",
        ),
    ],
)
# 定义测试函数 test_gpr_fit_error，检查是否会引发预期的错误
def test_gpr_fit_error(params, TypeError, err_msg):
    """Check that expected error are raised during fit."""
    # 创建 GaussianProcessRegressor 对象
    gpr = GaussianProcessRegressor(**params)
    # 使用 pytest.raises 检查是否引发指定类型的异常，并匹配错误消息
    with pytest.raises(TypeError, match=err_msg):
        gpr.fit(X, y)


# 定义测试函数 test_gpr_lml_error，检查在 LML 方法中是否引发正确的错误
def test_gpr_lml_error():
    """Check that we raise the proper error in the LML method."""
    # 创建 GaussianProcessRegressor 对象并拟合数据
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)

    # 定义错误消息
    err_msg = "Gradient can only be evaluated for theta!=None"
    # 使用 pytest.raises 检查是否引发指定类型的异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        gpr.log_marginal_likelihood(eval_gradient=True)


# 定义测试函数 test_gpr_predict_error，检查在预测过程中是否引发正确的错误
def test_gpr_predict_error():
    """Check that we raise the proper error during predict."""
    # 创建 GaussianProcessRegressor 对象并拟合数据
    gpr = GaussianProcessRegressor(kernel=RBF()).fit(X, y)

    # 定义错误消息
    err_msg = "At most one of return_std or return_cov can be requested."
    # 使用 pytest.raises 检查是否引发指定类型的异常，并匹配错误消息
    with pytest.raises(RuntimeError, match=err_msg):
        gpr.predict(X, return_cov=True, return_std=True)


# 使用 pytest.mark.parametrize 装饰器为 test_predict_shapes 函数添加参数化测试
@pytest.mark.parametrize("normalize_y", [True, False])
@pytest.mark.parametrize("n_targets", [None, 1, 10])
# 定义测试函数 test_predict_shapes，检查在单输出和多输出设置中 y_mean、y_std 和 y_cov 的形状
def test_predict_shapes(normalize_y, n_targets):
    """Check the shapes of y_mean, y_std, and y_cov in single-output
    (n_targets=None) and multi-output settings, including the edge case when
    n_targets=1, where the sklearn convention is to squeeze the predictions.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/17394
    https://github.com/scikit-learn/scikit-learn/issues/18065
    https://github.com/scikit-learn/scikit-learn/issues/22174
    """
    # 创建随机数生成器
    rng = np.random.RandomState(1234)

    n_features, n_samples_train, n_samples_test = 6, 9, 7

    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)

    # 根据 sklearn 约定，在预测时会压缩单输出数据
    y_test_shape = (n_samples_test,)
    if n_targets is not None and n_targets > 1:
        y_test_shape = y_test_shape + (n_targets,)

    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_test, n_features)
    y_train = rng.randn(*y_train_shape)

    # 创建 GaussianProcessRegressor 对象
    model = GaussianProcessRegressor(normalize_y=normalize_y)
    model.fit(X_train, y_train)

    # 进行预测并获取预测结果和标准差
    y_pred, y_std = model.predict(X_test, return_std=True)
    _, y_cov = model.predict(X_test, return_cov=True)

    # 检查预测结果、标准差和协方差的形状是否符合预期
    assert y_pred.shape == y_test_shape
    assert y_std.shape == y_test_shape
    assert y_cov.shape == (n_samples_test,) + y_test_shape


# 使用 pytest.mark.parametrize 装饰器为 test_predict_shapes 函数添加参数化测试
@pytest.mark.parametrize("normalize_y", [True, False])
@pytest.mark.parametrize("n_targets", [None, 1, 10])
def test_sample_y_shapes(normalize_y, n_targets):
    """Check the shapes of y_samples in single-output (n_targets=0) and
    multi-output settings, including the edge case when n_targets=1, where the
    sklearn convention is to squeeze the predictions.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/22175
    """
    rng = np.random.RandomState(1234)

    n_features, n_samples_train = 6, 9
    # Number of spatial locations to predict at
    n_samples_X_test = 7
    # Number of sample predictions per test point
    n_samples_y_test = 5

    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)

    # By convention single-output data is squeezed upon prediction
    if n_targets is not None and n_targets > 1:
        y_test_shape = (n_samples_X_test, n_targets, n_samples_y_test)
    else:
        y_test_shape = (n_samples_X_test, n_samples_y_test)

    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_X_test, n_features)
    y_train = rng.randn(*y_train_shape)

    model = GaussianProcessRegressor(normalize_y=normalize_y)

    # FIXME: before fitting, the estimator does not have information regarding
    # the number of targets and default to 1. This is inconsistent with the shape
    # provided after `fit`. This assert should be made once the following issue
    # is fixed:
    # https://github.com/scikit-learn/scikit-learn/issues/22430
    # y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    # assert y_samples.shape == y_test_shape

    model.fit(X_train, y_train)

    y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    assert y_samples.shape == y_test_shape


@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
@pytest.mark.parametrize("n_samples", [1, 5])
def test_sample_y_shape_with_prior(n_targets, n_samples):
    """Check the output shape of `sample_y` is consistent before and after `fit`."""
    rng = np.random.RandomState(1024)

    X = rng.randn(10, 3)
    y = rng.randn(10, n_targets if n_targets is not None else 1)

    model = GaussianProcessRegressor(n_targets=n_targets)
    shape_before_fit = model.sample_y(X, n_samples=n_samples).shape
    model.fit(X, y)
    shape_after_fit = model.sample_y(X, n_samples=n_samples).shape
    assert shape_before_fit == shape_after_fit


@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
def test_predict_shape_with_prior(n_targets):
    """Check the output shape of `predict` with prior distribution."""
    rng = np.random.RandomState(1024)

    n_sample = 10
    X = rng.randn(n_sample, 3)
    y = rng.randn(n_sample, n_targets if n_targets is not None else 1)

    model = GaussianProcessRegressor(n_targets=n_targets)
    # Obtain mean and covariance matrix from prior distribution
    mean_prior, cov_prior = model.predict(X, return_cov=True)
    # Obtain mean and standard deviation from prior distribution
    _, std_prior = model.predict(X, return_std=True)

    model.fit(X, y)


注释：

# 参数化测试函数，测试不同的 n_targets 值
@pytest.mark.parametrize("n_targets", [None, 1, 10])
def test_sample_y_shapes(normalize_y, n_targets):
    """检查 y_samples 的形状，包括单输出情况 (n_targets=0) 和多输出设置，
    包括 n_targets=1 的边缘情况，此时 sklearn 的惯例是压缩预测结果。

    非回归测试用例：
    https://github.com/scikit-learn/scikit-learn/issues/22175
    """
    rng = np.random.RandomState(1234)

    n_features, n_samples_train = 6, 9
    # 预测的空间位置数量
    n_samples_X_test = 7
    # 每个测试点的样本预测数量
    n_samples_y_test = 5

    y_train_shape = (n_samples_train,)
    if n_targets is not None:
        y_train_shape = y_train_shape + (n_targets,)

    # 根据惯例，单输出数据在预测时会被压缩
    if n_targets is not None and n_targets > 1:
        y_test_shape = (n_samples_X_test, n_targets, n_samples_y_test)
    else:
        y_test_shape = (n_samples_X_test, n_samples_y_test)

    X_train = rng.randn(n_samples_train, n_features)
    X_test = rng.randn(n_samples_X_test, n_features)
    y_train = rng.randn(*y_train_shape)

    model = GaussianProcessRegressor(normalize_y=normalize_y)

    # FIXME: 在拟合之前，估计器对于目标数量没有信息，默认为 1。这与拟合后提供的形状不一致，
    # 应在修复以下问题后断言：
    # https://github.com/scikit-learn/scikit-learn/issues/22430
    # y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    # assert y_samples.shape == y_test_shape

    model.fit(X_train, y_train)

    y_samples = model.sample_y(X_test, n_samples=n_samples_y_test)
    assert y_samples.shape == y_test_shape


# 参数化测试函数，测试不同的 n_targets 和 n_samples 值
@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
@pytest.mark.parametrize("n_samples", [1, 5])
def test_sample_y_shape_with_prior(n_targets, n_samples):
    """检查 `sample_y` 的输出形状在 `fit` 前后是否一致。"""
    rng = np.random.RandomState(1024)

    X = rng.randn(10, 3)
    y = rng.randn(10, n_targets if n_targets is not None else 1)

    model = GaussianProcessRegressor(n_targets=n_targets)
    shape_before_fit = model.sample_y(X, n_samples=n_samples).shape
    model.fit(X, y)
    shape_after_fit = model.sample_y(X, n_samples=n_samples).shape
    assert shape_before_fit == shape_after_fit


# 参数化测试函数，测试不同的 n_targets 值
@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
def test_predict_shape_with_prior(n_targets):
    """检查带有先验分布的 `predict` 的输出形状。"""
    rng = np.random.RandomState(1024)

    n_sample = 10
    X = rng.randn(n_sample, 3)
    y = rng.randn(n_sample, n_targets if n_targets is not None else 1)

    model = GaussianProcessRegressor(n_targets=n_targets)
    # 获取先验分布的均值和协方差矩阵
    mean_prior, cov_prior = model.predict(X, return_cov=True)
    # 获取先验分布的均值和标准差
    _, std_prior = model.predict(X, return_std=True)

    model.fit(X, y)
    # 使用训练好的模型对输入数据 X 进行预测，返回后验均值和协方差矩阵
    mean_post, cov_post = model.predict(X, return_cov=True)
    
    # 使用训练好的模型再次对输入数据 X 进行预测，返回后验标准差
    _, std_post = model.predict(X, return_std=True)
    
    # 断言语句，验证先验均值的形状与后验均值的形状相同
    assert mean_prior.shape == mean_post.shape
    
    # 断言语句，验证先验协方差矩阵的形状与后验协方差矩阵的形状相同
    assert cov_prior.shape == cov_post.shape
    
    # 断言语句，验证先验标准差的形状与后验标准差的形状相同
    assert std_prior.shape == std_post.shape
# 确保在调用 fit 方法时，当目标数量与 n_targets 参数不一致时抛出错误
def test_n_targets_error():
    """Check that an error is raised when the number of targets seen at fit is
    inconsistent with n_targets.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 3)  # 生成一个 10x3 的随机数组 X
    y = rng.randn(10, 2)  # 生成一个 10x2 的随机数组 y

    model = GaussianProcessRegressor(n_targets=1)  # 创建一个具有指定目标数量的高斯过程回归模型
    with pytest.raises(ValueError, match="The number of targets seen in `y`"):
        model.fit(X, y)  # 调用模型的 fit 方法，预期会引发值错误异常


class CustomKernel(C):
    """
    A custom kernel that has a diag method that returns the first column of the
    input matrix X. This is a helper for the test to check that the input
    matrix X is not mutated.
    """

    def diag(self, X):
        return X[:, 0]  # 返回输入矩阵 X 的第一列


# 检查高斯过程回归器的预测方法在设置 return_std=True 时不修改输入 X 的行为
def test_gpr_predict_input_not_modified():
    """
    Check that the input X is not modified by the predict method of the
    GaussianProcessRegressor when setting return_std=True.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24340
    """
    gpr = GaussianProcessRegressor(kernel=CustomKernel()).fit(X, y)  # 使用自定义内核创建和拟合高斯过程回归器

    X2_copy = np.copy(X2)  # 复制输入 X2 到 X2_copy
    _, _ = gpr.predict(X2, return_std=True)  # 调用高斯过程回归器的预测方法，设置返回标准差为真

    assert_allclose(X2, X2_copy)  # 断言输入 X2 和复制后的 X2_copy 相等
```