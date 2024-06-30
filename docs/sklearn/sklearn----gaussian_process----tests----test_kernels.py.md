# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\tests\test_kernels.py`

```
"""Testing for kernels for Gaussian processes."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从inspect模块中导入signature函数，用于获取函数的签名信息
from inspect import signature

# 导入必要的库
import numpy as np
import pytest

# 导入需要测试的类和函数
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
    RBF,
    CompoundKernel,
    ConstantKernel,
    DotProduct,
    Exponentiation,
    ExpSineSquared,
    KernelOperator,
    Matern,
    PairwiseKernel,
    RationalQuadratic,
    WhiteKernel,
    _approx_fprime,
)
# 导入用于计算核函数的工具函数和度量函数
from sklearn.metrics.pairwise import (
    PAIRWISE_KERNEL_FUNCTIONS,
    euclidean_distances,
    pairwise_kernels,
)
# 导入用于测试的工具函数和断言函数
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

# 生成随机数据
X = np.random.RandomState(0).normal(0, 1, (5, 2))
Y = np.random.RandomState(0).normal(0, 1, (6, 2))

# 创建复合核函数，由RBF核函数和WhiteKernel核函数组成
kernel_rbf_plus_white = RBF(length_scale=2.0) + WhiteKernel(noise_level=3.0)

# 定义多个核函数的列表
kernels = [
    RBF(length_scale=2.0),
    RBF(length_scale_bounds=(0.5, 2.0)),
    ConstantKernel(constant_value=10.0),
    2.0 * RBF(length_scale=0.33, length_scale_bounds="fixed"),
    2.0 * RBF(length_scale=0.5),
    kernel_rbf_plus_white,
    2.0 * RBF(length_scale=[0.5, 2.0]),
    2.0 * Matern(length_scale=0.33, length_scale_bounds="fixed"),
    2.0 * Matern(length_scale=0.5, nu=0.5),
    2.0 * Matern(length_scale=1.5, nu=1.5),
    2.0 * Matern(length_scale=2.5, nu=2.5),
    2.0 * Matern(length_scale=[0.5, 2.0], nu=0.5),
    3.0 * Matern(length_scale=[2.0, 0.5], nu=1.5),
    4.0 * Matern(length_scale=[0.5, 0.5], nu=2.5),
    RationalQuadratic(length_scale=0.5, alpha=1.5),
    ExpSineSquared(length_scale=0.5, periodicity=1.5),
    DotProduct(sigma_0=2.0),
    DotProduct(sigma_0=2.0) ** 2,
    RBF(length_scale=[2.0]),
    Matern(length_scale=[2.0]),
]

# 将内置的核函数添加到测试列表中
for metric in PAIRWISE_KERNEL_FUNCTIONS:
    if metric in ["additive_chi2", "chi2"]:
        continue
    kernels.append(PairwiseKernel(gamma=1.0, metric=metric))

# 使用@pytest.mark.parametrize装饰器，对每个核函数进行梯度测试
@pytest.mark.parametrize("kernel", kernels)
def test_kernel_gradient(kernel):
    # 比较核函数的解析梯度和数值梯度
    K, K_gradient = kernel(X, eval_gradient=True)

    assert K_gradient.shape[0] == X.shape[0]
    assert K_gradient.shape[1] == X.shape[0]
    assert K_gradient.shape[2] == kernel.theta.shape[0]

    # 定义一个函数，用于根据参数theta计算核函数的值
    def eval_kernel_for_theta(theta):
        kernel_clone = kernel.clone_with_theta(theta)
        K = kernel_clone(X, eval_gradient=False)
        return K

    # 使用数值方法计算梯度的近似值
    K_gradient_approx = _approx_fprime(kernel.theta, eval_kernel_for_theta, 1e-10)

    # 断言解析梯度与数值梯度的近似值在指定精度内相等
    assert_almost_equal(K_gradient, K_gradient_approx, 4)


# 使用@pytest.mark.parametrize装饰器，对每个核函数的参数向量进行测试
@pytest.mark.parametrize(
    "kernel",
    [
        kernel
        for kernel in kernels
        # 跳过非基本核函数
        if not (isinstance(kernel, (KernelOperator, Exponentiation)))
    ],
)
def test_kernel_theta(kernel):
    # 检查核函数的参数向量theta是否被正确设置
    theta = kernel.theta
    _, K_gradient = kernel(X, eval_gradient=True)
    # 确定对于 theta 起作用的核参数
    init_sign = signature(kernel.__class__.__init__).parameters.values()
    # 提取参数名列表，排除 "self"
    args = [p.name for p in init_sign if p.name != "self"]
    # 找到以 "_bounds" 结尾的参数名，并去除 "_bounds" 后缀，形成 theta 变量名列表
    theta_vars = map(
        lambda s: s[0 : -len("_bounds")], filter(lambda s: s.endswith("_bounds"), args)
    )
    # 断言核的超参数名集合与 theta 变量名集合一致
    assert set(hyperparameter.name for hyperparameter in kernel.hyperparameters) == set(
        theta_vars
    )

    # 检查 theta 中的值是否与超参数值（对数值）一致
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        # 断言 theta[i] 等于对应超参数的自然对数
        assert theta[i] == np.log(getattr(kernel, hyperparameter.name))

    # 固定的核参数必须从 theta 和梯度中排除
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        # 创建一个固定某个超参数的副本
        params = kernel.get_params()
        params[hyperparameter.name + "_bounds"] = "fixed"
        kernel_class = kernel.__class__
        new_kernel = kernel_class(**params)
        # 检查 theta 和 K_gradient 是否与排除了固定维度的 new_kernel 相同
        _, K_gradient_new = new_kernel(X, eval_gradient=True)
        assert theta.shape[0] == new_kernel.theta.shape[0] + 1
        assert K_gradient.shape[2] == K_gradient_new.shape[2] + 1
        if i > 0:
            assert theta[:i] == new_kernel.theta[:i]
            assert_array_equal(K_gradient[..., :i], K_gradient_new[..., :i])
        if i + 1 < len(kernel.hyperparameters):
            assert theta[i + 1 :] == new_kernel.theta[i:]
            assert_array_equal(K_gradient[..., i + 1 :], K_gradient_new[..., i:])

    # 检查 theta 的值是否被正确修改
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        # 将 theta[i] 设为对数值 42
        theta[i] = np.log(42)
        kernel.theta = theta
        # 断言相应超参数的值接近 42
        assert_almost_equal(getattr(kernel, hyperparameter.name), 42)

        # 将核的超参数设为 43
        setattr(kernel, hyperparameter.name, 43)
        # 断言 kernel.theta[i] 接近 np.log(43)
        assert_almost_equal(kernel.theta[i], np.log(43))
@pytest.mark.parametrize(
    "kernel",
    [
        kernel
        for kernel in kernels
        # Identity is not satisfied on diagonal
        if kernel != kernel_rbf_plus_white
    ],
)
# 定义测试函数，参数化测试不同的核函数
def test_auto_vs_cross(kernel):
    # Auto-correlation and cross-correlation should be consistent.
    # 计算自相关和交叉相关应保持一致
    K_auto = kernel(X)
    K_cross = kernel(X, X)
    assert_almost_equal(K_auto, K_cross, 5)


@pytest.mark.parametrize("kernel", kernels)
# 定义测试函数，参数化测试不同的核函数
def test_kernel_diag(kernel):
    # Test that diag method of kernel returns consistent results.
    # 测试核函数的 diag 方法返回一致的结果
    K_call_diag = np.diag(kernel(X))
    K_diag = kernel.diag(X)
    assert_almost_equal(K_call_diag, K_diag, 5)


def test_kernel_operator_commutative():
    # Adding kernels and multiplying kernels should be commutative.
    # Check addition
    # 校验核函数的加法是否符合交换律
    assert_almost_equal((RBF(2.0) + 1.0)(X), (1.0 + RBF(2.0))(X))

    # Check multiplication
    # 校验核函数的乘法是否符合交换律
    assert_almost_equal((3.0 * RBF(2.0))(X), (RBF(2.0) * 3.0)(X))


def test_kernel_anisotropic():
    # Anisotropic kernel should be consistent with isotropic kernels.
    # 测试各向异性核函数与各向同性核函数的一致性
    kernel = 3.0 * RBF([0.5, 2.0])

    K = kernel(X)
    X1 = np.array(X)
    X1[:, 0] *= 4
    K1 = 3.0 * RBF(2.0)(X1)
    assert_almost_equal(K, K1)

    X2 = np.array(X)
    X2[:, 1] /= 4
    K2 = 3.0 * RBF(0.5)(X2)
    assert_almost_equal(K, K2)

    # Check getting and setting via theta
    # 通过 theta 检查获取和设置
    kernel.theta = kernel.theta + np.log(2)
    assert_array_equal(kernel.theta, np.log([6.0, 1.0, 4.0]))
    assert_array_equal(kernel.k2.length_scale, [1.0, 4.0])


@pytest.mark.parametrize(
    "kernel", [kernel for kernel in kernels if kernel.is_stationary()]
)
# 参数化测试静止核函数
def test_kernel_stationary(kernel):
    # Test stationarity of kernels.
    # 测试核函数的平稳性
    K = kernel(X, X + 1)
    assert_almost_equal(K[0, 0], np.diag(K))


@pytest.mark.parametrize("kernel", kernels)
# 参数化测试核函数
def test_kernel_input_type(kernel):
    # Test whether kernels is for vectors or structured data
    # 测试核函数是用于向量还是结构化数据
    if isinstance(kernel, Exponentiation):
        assert kernel.requires_vector_input == kernel.kernel.requires_vector_input
    if isinstance(kernel, KernelOperator):
        assert kernel.requires_vector_input == (
            kernel.k1.requires_vector_input or kernel.k2.requires_vector_input
        )


def test_compound_kernel_input_type():
    kernel = CompoundKernel([WhiteKernel(noise_level=3.0)])
    assert not kernel.requires_vector_input

    kernel = CompoundKernel([WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    assert kernel.requires_vector_input


def check_hyperparameters_equal(kernel1, kernel2):
    # Check that hyperparameters of two kernels are equal
    # 检查两个核函数的超参数是否相等
    for attr in set(dir(kernel1) + dir(kernel2)):
        if attr.startswith("hyperparameter_"):
            attr_value1 = getattr(kernel1, attr)
            attr_value2 = getattr(kernel2, attr)
            assert attr_value1 == attr_value2


@pytest.mark.parametrize("kernel", kernels)
# 参数化测试核函数
def test_kernel_clone(kernel):
    # Test that sklearn's clone works correctly on kernels.
    # 测试 sklearn 的克隆函数在核函数上的正确性
    kernel_cloned = clone(kernel)
    # XXX: Should this be fixed?
    # 这里使用了一个标记 XXX，通常用于指出需要注意或修复的问题

    # This differs from the sklearn's estimators equality check.
    # 这里进行了一项断言，验证 kernel 和 kernel_cloned 是否相等
    assert kernel == kernel_cloned

    # Check that all constructor parameters are equal.
    # 检查两个对象的构造函数参数是否完全相等
    assert kernel.get_params() == kernel_cloned.get_params()

    # Check that all hyperparameters are equal.
    # 调用自定义函数 check_hyperparameters_equal 来验证所有超参数是否相等
    check_hyperparameters_equal(kernel, kernel_cloned)
# 使用@pytest.mark.parametrize装饰器来参数化测试函数，测试不同的内核
@pytest.mark.parametrize("kernel", kernels)
def test_kernel_clone_after_set_params(kernel):
    # 此测试用于验证使用set_params不会破坏内核的克隆。
    # 之前这会出问题，因为在一些内核（如RBF）中，修改长度尺度的非平凡逻辑曾在构造函数中。
    # 参见https://github.com/scikit-learn/scikit-learn/issues/6961获取更多详情。

    # 设定边界范围
    bounds = (1e-5, 1e5)
    # 克隆内核对象
    kernel_cloned = clone(kernel)
    # 获取内核当前的参数
    params = kernel.get_params()

    # 对于RationalQuadratic核，长度尺度是各向同性的
    isotropic_kernels = (ExpSineSquared, RationalQuadratic)
    if "length_scale" in params and not isinstance(kernel, isotropic_kernels):
        length_scale = params["length_scale"]
        if np.iterable(length_scale):
            # XXX 作为v0.22之后的版本不可达的代码
            params["length_scale"] = length_scale[0]
            params["length_scale_bounds"] = bounds
        else:
            params["length_scale"] = [length_scale] * 2
            params["length_scale_bounds"] = bounds * 2

        # 使用更新后的参数设置内核克隆对象的参数
        kernel_cloned.set_params(**params)
        # 克隆内核的克隆对象
        kernel_cloned_clone = clone(kernel_cloned)

        # 断言克隆后的内核对象参数与原始克隆对象参数相同
        assert kernel_cloned_clone.get_params() == kernel_cloned.get_params()
        # 断言克隆后的内核对象与原始克隆对象不是同一个对象
        assert id(kernel_cloned_clone) != id(kernel_cloned)
        # 检查内核超参数是否相等
        check_hyperparameters_equal(kernel_cloned, kernel_cloned_clone)


def test_matern_kernel():
    # 测试Matern核对nu的特殊值的一致性
    K = Matern(nu=1.5, length_scale=1.0)(X)
    # Matern核的对角线元素为1
    assert_array_almost_equal(np.diag(K), np.ones(X.shape[0]))

    # 对于coef0==0.5的Matern核等于绝对指数核
    K_absexp = np.exp(-euclidean_distances(X, X, squared=False))
    K = Matern(nu=0.5, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_absexp)

    # 对于coef0==inf的Matern核等于RBF核
    K_rbf = RBF(length_scale=1.0)(X)
    K = Matern(nu=np.inf, length_scale=1.0)(X)
    assert_array_almost_equal(K, K_rbf)
    assert_allclose(K, K_rbf)

    # 测试Matern核的特殊情况（coef0在[0.5, 1.5, 2.5]）是否与coef0在[0.5 + tiny, 1.5 + tiny, 2.5 + tiny]的一般情况几乎相同
    tiny = 1e-10
    for nu in [0.5, 1.5, 2.5]:
        K1 = Matern(nu=nu, length_scale=1.0)(X)
        K2 = Matern(nu=nu + tiny, length_scale=1.0)(X)
        assert_array_almost_equal(K1, K2)

    # 测试coef0==large时，Matern核接近RBF核
    large = 100
    K1 = Matern(nu=large, length_scale=1.0)(X)
    K2 = RBF(length_scale=1.0)(X)
    assert_array_almost_equal(K1, K2, decimal=2)


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_versus_pairwise(kernel):
    # 检查GP内核是否也可以用作成对内核
    pass
    # 如果 kernel 不等于 kernel_rbf_plus_white
    if kernel != kernel_rbf_plus_white:
        # 对于 WhiteKernel：k(X) != k(X,X)，这是 pairwise_kernels 假设的情况
        # 计算 kernel(X) 的结果
        K1 = kernel(X)
        # 使用 pairwise_kernels 计算指定 metric 下的核矩阵
        K2 = pairwise_kernels(X, metric=kernel)
        # 断言 K1 和 K2 几乎相等
        assert_array_almost_equal(K1, K2)

    # 测试不同核之间的交叉核
    # 计算 kernel(X, Y) 的结果
    K1 = kernel(X, Y)
    # 使用 pairwise_kernels 计算指定 metric 下的核矩阵
    K2 = pairwise_kernels(X, Y, metric=kernel)
    # 断言 K1 和 K2 几乎相等
    assert_array_almost_equal(K1, K2)
@pytest.mark.parametrize("kernel", kernels)
def test_set_get_params(kernel):
    # 参数化测试：使用不同的内核进行测试

    # 测试 get_params() 方法
    index = 0
    params = kernel.get_params()
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # 对于各向异性内核
            assert_almost_equal(
                np.exp(kernel.theta[index : index + size]), params[hyperparameter.name]
            )
            index += size
        else:
            assert_almost_equal(
                np.exp(kernel.theta[index]), params[hyperparameter.name]
            )
            index += 1

    # 测试 set_params() 方法
    index = 0
    value = 10  # 任意值
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # 对于各向异性内核
            kernel.set_params(**{hyperparameter.name: [value] * size})
            assert_almost_equal(
                np.exp(kernel.theta[index : index + size]), [value] * size
            )
            index += size
        else:
            kernel.set_params(**{hyperparameter.name: value})
            assert_almost_equal(np.exp(kernel.theta[index]), value)
            index += 1


@pytest.mark.parametrize("kernel", kernels)
def test_repr_kernels(kernel):
    # 内核的 __repr__() 方法的烟雾测试

    repr(kernel)


def test_rational_quadratic_kernel():
    # 测试 RationalQuadratic 内核

    kernel = RationalQuadratic(length_scale=[1.0, 1.0])
    message = (
        "RationalQuadratic kernel only supports isotropic "
        "version, please use a single "
        "scalar for length_scale"
    )
    # 断言引发 AttributeError 异常，并匹配指定消息
    with pytest.raises(AttributeError, match=message):
        kernel(X)
```