# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\tests\test_gpc.py`

```
# 导入警告模块，用于管理警告信息
import warnings

# 导入科学计算库 NumPy
import numpy as np

# 导入 pytest 用于测试
import pytest

# 导入 scipy 中的近似梯度函数
from scipy.optimize import approx_fprime

# 导入 sklearn 中的异常类 ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning

# 导入 sklearn 中的高斯过程分类器
from sklearn.gaussian_process import GaussianProcessClassifier

# 导入 sklearn 中的高斯过程核函数：RBF, CompoundKernel, WhiteKernel
from sklearn.gaussian_process.kernels import (
    RBF,
    CompoundKernel,
    WhiteKernel,
)

# 导入 sklearn 中的高斯过程核函数，将 ConstantKernel 简称为 C
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
)

# 导入 sklearn 中的测试用 MiniSeqKernel 和测试工具函数
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import assert_almost_equal, assert_array_equal


# 定义函数 f，计算输入 x 的正弦值
def f(x):
    return np.sin(x)


# 生成至少二维的数组 X，包含在 0 到 10 之间的 30 个等间距数值
X = np.atleast_2d(np.linspace(0, 10, 30)).T

# 生成二维数组 X2，包含值为 [2.0, 4.0, 5.5, 6.5, 7.5]
X2 = np.atleast_2d([2.0, 4.0, 5.5, 6.5, 7.5]).T

# 根据函数 f(X) 大于 0 的结果生成二进制数组 y
y = np.array(f(X).ravel() > 0, dtype=int)

# 计算 f(X) 的一维数组 fX
fX = f(X).ravel()

# 初始化多类别数组 y_mc
y_mc = np.empty(y.shape, dtype=int)

# 根据 fX 的值填充多类别数组 y_mc
y_mc[fX < -0.35] = 0
y_mc[(fX >= -0.35) & (fX < 0.35)] = 1
y_mc[fX > 0.35] = 2

# 定义固定长度尺度为 1.0 的 RBF 核函数
fixed_kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")

# 定义核函数列表 kernels
kernels = [
    RBF(length_scale=0.1),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
]

# 非固定尺度核函数的列表 non_fixed_kernels
non_fixed_kernels = [kernel for kernel in kernels if kernel != fixed_kernel]


# 使用 pytest 的参数化装饰器，测试预测结果一致性
@pytest.mark.parametrize("kernel", kernels)
def test_predict_consistent(kernel):
    # 创建高斯过程分类器对象 gpc，使用指定的核函数 kernel 拟合数据 X, y
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 断言预测结果与预测概率的一致性，预测概率是否大于等于 0.5
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


# 测试预测结果一致性（结构化数据）
def test_predict_consistent_structured():
    # 定义结构化数据 X 和二分类标签 y
    X = ["A", "AB", "B"]
    y = np.array([True, False, True])
    # 使用 MiniSeqKernel 拟合数据 X, y 的高斯过程分类器对象 gpc
    kernel = MiniSeqKernel(baseline_similarity_bounds="fixed")
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 断言预测结果与预测概率的一致性，预测概率是否大于等于 0.5
    assert_array_equal(gpc.predict(X), gpc.predict_proba(X)[:, 1] >= 0.5)


# 使用非固定核函数列表 non_fixed_kernels 参数化装饰器，测试对数边际似然改善
@pytest.mark.parametrize("kernel", non_fixed_kernels)
def test_lml_improving(kernel):
    # 创建高斯过程分类器对象 gpc，使用指定的核函数 kernel 拟合数据 X, y
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 断言超参数调优是否改善了对数边际似然
    assert gpc.log_marginal_likelihood(gpc.kernel_.theta) > gpc.log_marginal_likelihood(
        kernel.theta
    )


# 使用 kernels 参数化装饰器，测试优化核函数的 lml 是否正确存储
@pytest.mark.parametrize("kernel", kernels)
def test_lml_precomputed(kernel):
    # 创建高斯过程分类器对象 gpc，使用指定的核函数 kernel 拟合数据 X, y
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 断言优化核函数的 lml 是否正确存储，精确到小数点后 7 位
    assert_almost_equal(
        gpc.log_marginal_likelihood(gpc.kernel_.theta), gpc.log_marginal_likelihood(), 7
    )


# 使用 kernels 参数化装饰器，测试 clone_kernel=False 对核函数.theta 的影响
@pytest.mark.parametrize("kernel", kernels)
def test_lml_without_cloning_kernel(kernel):
    # 创建高斯过程分类器对象 gpc，使用指定的核函数 kernel 拟合数据 X, y
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 初始化输入 theta 为全为 1 的浮点数组
    input_theta = np.ones(gpc.kernel_.theta.shape, dtype=np.float64)
    # 计算高斯过程分类器的边际似然并记录日志
    gpc.log_marginal_likelihood(input_theta, clone_kernel=False)
    # 断言检查高斯过程分类器的核参数是否与输入的参数几乎相等，精度为7位小数
    assert_almost_equal(gpc.kernel_.theta, input_theta, 7)
@pytest.mark.parametrize("kernel", non_fixed_kernels)
# 使用 pytest 的参数化测试，遍历非固定的核函数列表
def test_converged_to_local_maximum(kernel):
    # 测试在超参数优化后是否处于局部最大值
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 使用指定核函数训练高斯过程分类器

    lml, lml_gradient = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)
    # 计算对数边际似然及其梯度

    assert np.all(
        (np.abs(lml_gradient) < 1e-4)
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 0])
        | (gpc.kernel_.theta == gpc.kernel_.bounds[:, 1])
    )
    # 断言条件：梯度绝对值小于1e-4或者参数等于边界值


@pytest.mark.parametrize("kernel", kernels)
# 使用 pytest 的参数化测试，遍历核函数列表
def test_lml_gradient(kernel):
    # 比较对数边际似然的解析梯度和数值梯度
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
    # 使用指定核函数训练高斯过程分类器

    lml, lml_gradient = gpc.log_marginal_likelihood(kernel.theta, True)
    # 计算对数边际似然及其梯度
    lml_gradient_approx = approx_fprime(
        kernel.theta, lambda theta: gpc.log_marginal_likelihood(theta, False), 1e-10
    )
    # 使用数值方法计算梯度的近似值

    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)
    # 断言条件：解析梯度与数值梯度的近似值在小数点后三位精度上相等


def test_random_starts(global_random_seed):
    # 测试增加的随机启动次数是否只会增加所选参数的对数边际似然
    n_samples, n_features = 25, 2
    rng = np.random.RandomState(global_random_seed)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1)) > 0
    # 创建随机数据集 X 和二分类标签 y

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1e-3] * n_features, length_scale_bounds=[(1e-4, 1e2)] * n_features
    )
    # 定义核函数

    last_lml = -np.inf
    for n_restarts_optimizer in range(5):
        gp = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=global_random_seed,
        ).fit(X, y)
        # 使用不同的随机启动次数训练高斯过程分类器

        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        # 计算对数边际似然

        assert lml > last_lml - np.finfo(np.float32).eps
        # 断言条件：当前对数边际似然大于上一次的似然值减去一个极小的浮点数
        last_lml = lml


@pytest.mark.parametrize("kernel", non_fixed_kernels)
# 使用 pytest 的参数化测试，遍历非固定的核函数列表
def test_custom_optimizer(kernel, global_random_seed):
    # 测试 GPC 是否可以使用外部定义的优化器
    # 定义一个简单的优化器，简单测试10个随机超参数
    def optimizer(obj_func, initial_theta, bounds):
        rng = np.random.RandomState(global_random_seed)
        theta_opt, func_min = initial_theta, obj_func(
            initial_theta, eval_gradient=False
        )
        for _ in range(10):
            theta = np.atleast_1d(
                rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
            )
            f = obj_func(theta, eval_gradient=False)
            if f < func_min:
                theta_opt, func_min = theta, f
        return theta_opt, func_min

    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(X, y_mc)
    # 使用自定义优化器训练高斯过程分类器

    assert gpc.log_marginal_likelihood(
        gpc.kernel_.theta
    ) >= gpc.log_marginal_likelihood(kernel.theta)
    # 断言条件：优化器改进了边际似然
# 测试多类分类问题的高斯过程分类器（GPC）。
def test_multi_class(kernel):
    # 创建一个带有给定内核的高斯过程分类器对象
    gpc = GaussianProcessClassifier(kernel=kernel)
    # 使用训练数据 X 和多类别标签 y_mc 进行模型拟合
    gpc.fit(X, y_mc)

    # 对测试数据 X2 进行预测，返回预测类别的概率
    y_prob = gpc.predict_proba(X2)
    # 断言每个样本的预测概率之和接近于 1
    assert_almost_equal(y_prob.sum(1), 1)

    # 对测试数据 X2 进行预测，返回预测的类别标签
    y_pred = gpc.predict(X2)
    # 断言预测的类别标签与预测概率中概率最大的类别索引一致
    assert_array_equal(np.argmax(y_prob, 1), y_pred)


@pytest.mark.parametrize("kernel", kernels)
# 测试在多线程工作情况下，多类别高斯过程分类器（GPC）是否能够产生相同的结果。
def test_multi_class_n_jobs(kernel):
    # 创建一个带有给定内核的高斯过程分类器对象
    gpc = GaussianProcessClassifier(kernel=kernel)
    # 使用训练数据 X 和多类别标签 y_mc 进行模型拟合
    gpc.fit(X, y_mc)

    # 创建一个带有给定内核和多线程参数的高斯过程分类器对象
    gpc_2 = GaussianProcessClassifier(kernel=kernel, n_jobs=2)
    # 使用训练数据 X 和多类别标签 y_mc 进行模型拟合
    gpc_2.fit(X, y_mc)

    # 对测试数据 X2 进行预测，返回预测类别的概率
    y_prob = gpc.predict_proba(X2)
    # 对测试数据 X2 进行预测，返回预测类别的概率（使用多线程的模型）
    y_prob_2 = gpc_2.predict_proba(X2)
    # 断言两种模型的预测概率相等
    assert_almost_equal(y_prob, y_prob_2)


# 测试警告信息：核的长度尺度接近于指定的上限值时的警告
def test_warning_bounds():
    # 创建一个带有指定长度尺度上下限的 RBF 内核对象
    kernel = RBF(length_scale_bounds=[1e-5, 1e-3])
    # 创建一个带有指定内核的高斯过程分类器对象
    gpc = GaussianProcessClassifier(kernel=kernel)
    # 设置警告信息文本内容
    warning_message = (
        "The optimal value found for dimension 0 of parameter "
        "length_scale is close to the specified upper bound "
        "0.001. Increasing the bound and calling fit again may "
        "find a better value."
    )
    # 断言在拟合过程中会产生收敛警告，并且警告信息符合预期
    with pytest.warns(ConvergenceWarning, match=warning_message):
        gpc.fit(X, y)

    # 创建一个包含白噪声和 RBF 内核的核对象
    kernel_sum = WhiteKernel(noise_level_bounds=[1e-5, 1e-3]) + RBF(
        length_scale_bounds=[1e3, 1e5]
    )
    # 创建一个带有指定内核的高斯过程分类器对象
    gpc_sum = GaussianProcessClassifier(kernel=kernel_sum)
    # 使用警告记录上下文捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # 使用训练数据 X 和标签 y 进行模型拟合
        gpc_sum.fit(X, y)

        # 断言捕获的警告数量为 2
        assert len(record) == 2

        # 断言第一个捕获的警告属于收敛警告类别，并且警告信息符合预期
        assert issubclass(record[0].category, ConvergenceWarning)
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k1__noise_level is close to the "
            "specified upper bound 0.001. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )

        # 断言第二个捕获的警告属于收敛警告类别，并且警告信息符合预期
        assert issubclass(record[1].category, ConvergenceWarning)
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "k2__length_scale is close to the "
            "specified lower bound 1000.0. "
            "Decreasing the bound and calling "
            "fit again may find a better value."
        )

    # 创建一个重复拼接 X 数据两次的新数据集
    X_tile = np.tile(X, 2)
    # 创建一个带有指定长度尺度和长度尺度上下限的 RBF 内核对象
    kernel_dims = RBF(length_scale=[1.0, 2.0], length_scale_bounds=[1e1, 1e2])
    # 创建一个带有指定内核的高斯过程分类器对象
    gpc_dims = GaussianProcessClassifier(kernel=kernel_dims)
    # 使用 `warnings` 模块捕获警告信息，并将其记录在 `record` 列表中
    with warnings.catch_warnings(record=True) as record:
        # 设置警告过滤器，使得所有警告都被记录下来
        warnings.simplefilter("always")
        # 使用 `X_tile` 和 `y` 来拟合 `gpc_dims` 模型
        gpc_dims.fit(X_tile, y)

        # 断言捕获的警告记录数为2，确保两个特定的警告被捕获
        assert len(record) == 2

        # 断言第一个记录的警告属于 `ConvergenceWarning` 类
        assert issubclass(record[0].category, ConvergenceWarning)
        # 断言第一个记录的警告消息内容符合预期的文本
        assert (
            record[0].message.args[0] == "The optimal value found for "
            "dimension 0 of parameter "
            "length_scale is close to the "
            "specified upper bound 100.0. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )

        # 断言第二个记录的警告属于 `ConvergenceWarning` 类
        assert issubclass(record[1].category, ConvergenceWarning)
        # 断言第二个记录的警告消息内容符合预期的文本
        assert (
            record[1].message.args[0] == "The optimal value found for "
            "dimension 1 of parameter "
            "length_scale is close to the "
            "specified upper bound 100.0. "
            "Increasing the bound and calling "
            "fit again may find a better value."
        )
@pytest.mark.parametrize(
    # 参数化测试，指定测试参数和预期的错误类型以及错误消息
    "params, error_type, err_msg",
    [
        (
            {"kernel": CompoundKernel(0)},  # 设置参数字典，包括一个错误的 kernel 类型
            ValueError,                     # 预期引发 ValueError 异常
            "kernel cannot be a CompoundKernel",  # 预期的错误消息
        )
    ],
)
def test_gpc_fit_error(params, error_type, err_msg):
    """Check that expected error are raised during fit."""
    gpc = GaussianProcessClassifier(**params)  # 创建高斯过程分类器对象，传入参数字典
    with pytest.raises(error_type, match=err_msg):  # 断言期望引发指定类型异常，并匹配错误消息
        gpc.fit(X, y)  # 调用 fit 方法，期望在此处引发异常
```