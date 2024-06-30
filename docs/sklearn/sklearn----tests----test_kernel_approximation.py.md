# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_kernel_approximation.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from sklearn.datasets import make_classification  # 导入生成分类数据的函数
from sklearn.kernel_approximation import (  # 导入核近似方法
    AdditiveChi2Sampler,
    Nystroem,
    PolynomialCountSketch,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.metrics.pairwise import (  # 导入核函数计算方法
    chi2_kernel,
    kernel_metrics,  # 未使用的导入，可能是错误
    polynomial_kernel,
    rbf_kernel,
)
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入稀疏矩阵容器的修复

# 生成数据
rng = np.random.RandomState(0)  # 使用种子0生成随机数种子
X = rng.random_sample(size=(300, 50))  # 生成300x50的随机数数组X
Y = rng.random_sample(size=(300, 50))  # 生成300x50的随机数数组Y
X /= X.sum(axis=1)[:, np.newaxis]  # 对X的每一行进行归一化处理
Y /= Y.sum(axis=1)[:, np.newaxis]  # 对Y的每一行进行归一化处理


@pytest.mark.parametrize("gamma", [0.1, 1, 2.5])  # 参数化测试，gamma取值0.1, 1, 2.5
@pytest.mark.parametrize("degree, n_components", [(1, 500), (2, 500), (3, 5000)])  # 参数化测试，degree和n_components不同组合
@pytest.mark.parametrize("coef0", [0, 2.5])  # 参数化测试，coef0取值0或2.5
def test_polynomial_count_sketch(gamma, degree, coef0, n_components):
    # 测试PolynomialCountSketch是否逼近多项式核函数在随机数据上的表现

    # 计算精确的核矩阵
    kernel = polynomial_kernel(X, Y, gamma=gamma, degree=degree, coef0=coef0)

    # 进行多项式核近似映射
    ps_transform = PolynomialCountSketch(
        n_components=n_components,
        gamma=gamma,
        coef0=coef0,
        degree=degree,
        random_state=42,
    )
    X_trans = ps_transform.fit_transform(X)  # 对X进行变换
    Y_trans = ps_transform.transform(Y)  # 对Y进行变换
    kernel_approx = np.dot(X_trans, Y_trans.T)  # 计算近似核矩阵

    error = kernel - kernel_approx  # 计算误差
    assert np.abs(np.mean(error)) <= 0.05  # 断言平均误差小于等于0.05，接近无偏差
    np.abs(error, out=error)  # 求误差的绝对值
    assert np.max(error) <= 0.1  # 断言最大误差小于等于0.1，没有太大偏差
    assert np.mean(error) <= 0.05  # 断言平均误差小于等于0.05，平均值相对较接近


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化测试，使用不同的稀疏矩阵容器
@pytest.mark.parametrize("gamma", [0.1, 1.0])  # 参数化测试，gamma取值0.1或1.0
@pytest.mark.parametrize("degree", [1, 2, 3])  # 参数化测试，degree取值1, 2, 3
@pytest.mark.parametrize("coef0", [0, 2.5])  # 参数化测试，coef0取值0或2.5
def test_polynomial_count_sketch_dense_sparse(gamma, degree, coef0, csr_container):
    """检查PolynomialCountSketch对于密集和稀疏输入的结果是否相同。"""
    ps_dense = PolynomialCountSketch(
        n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42
    )
    Xt_dense = ps_dense.fit_transform(X)  # 对密集数据X进行变换
    Yt_dense = ps_dense.transform(Y)  # 对密集数据Y进行变换

    ps_sparse = PolynomialCountSketch(
        n_components=500, gamma=gamma, degree=degree, coef0=coef0, random_state=42
    )
    Xt_sparse = ps_sparse.fit_transform(csr_container(X))  # 对稀疏数据X进行变换
    Yt_sparse = ps_sparse.transform(csr_container(Y))  # 对稀疏数据Y进行变换

    assert_allclose(Xt_dense, Xt_sparse)  # 断言密集数据和稀疏数据的变换结果近似相等
    assert_allclose(Yt_dense, Yt_sparse)  # 断言密集数据和稀疏数据的变换结果近似相等


def _linear_kernel(X, Y):
    return np.dot(X, Y.T)  # 计算线性核矩阵


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化测试，使用不同的稀疏矩阵容器
def test_additive_chi2_sampler(csr_container):
    # 测试AdditiveChi2Sampler是否在随机数据上逼近核函数

    # 计算精确核矩阵
    # 方便计算公式的缩写
    X_ = X[:, np.newaxis, :]
    # 将 Y 扩展为一个新的维度，以便与 X 的维度匹配
    Y_ = Y[np.newaxis, :, :]

    # 计算大核矩阵，通过元素级运算实现
    large_kernel = 2 * X_ * Y_ / (X_ + Y_)

    # 对大核矩阵按照特征维度求和，将结果降维为 n_samples_x x n_samples_y 的矩阵
    kernel = large_kernel.sum(axis=2)

    # 进行核映射的近似处理，使用 AdditiveChi2Sampler 进行变换
    transform = AdditiveChi2Sampler(sample_steps=3)

    # 对 X 应用变换并拟合数据
    X_trans = transform.fit_transform(X)

    # 对 Y 应用变换
    Y_trans = transform.transform(Y)

    # 使用变换后的数据计算近似核矩阵
    kernel_approx = np.dot(X_trans, Y_trans.T)

    # 检查近似的核矩阵与原核矩阵的数值是否几乎相等，精度为小数点后 1 位
    assert_array_almost_equal(kernel, kernel_approx, 1)

    # 对稀疏矩阵数据进行相同的变换和比较
    X_sp_trans = transform.fit_transform(csr_container(X))
    Y_sp_trans = transform.transform(csr_container(Y))

    # 断言稀疏矩阵经过变换后的结果与密集矩阵转换为稀疏后的结果相等
    assert_array_equal(X_trans, X_sp_trans.toarray())
    assert_array_equal(Y_trans, Y_sp_trans.toarray())

    # 测试在输入包含负值时是否会触发 ValueError 异常
    Y_neg = Y.copy()
    Y_neg[0, 0] = -1
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        transform.fit(Y_neg)
@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
@pytest.mark.parametrize("sample_steps", range(1, 4))
def test_additive_chi2_sampler_sample_steps(method, sample_steps):
    """Check that the input sample step doesn't raise an error
    and that sample interval doesn't change after fit.
    """
    # 创建 AdditiveChi2Sampler 的实例，设置 sample_steps 参数
    transformer = AdditiveChi2Sampler(sample_steps=sample_steps)
    # 调用相应的方法（fit、fit_transform 或 transform）处理输入数据 X
    getattr(transformer, method)(X)

    # 设置 sample_interval 的值为 0.5
    sample_interval = 0.5
    # 创建 AdditiveChi2Sampler 的另一个实例，设置 sample_steps 和 sample_interval 参数
    transformer = AdditiveChi2Sampler(
        sample_steps=sample_steps,
        sample_interval=sample_interval,
    )
    # 再次调用相应的方法处理输入数据 X
    getattr(transformer, method)(X)
    # 断言 transformer 的 sample_interval 属性与设置的值 sample_interval 相等
    assert transformer.sample_interval == sample_interval


@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
def test_additive_chi2_sampler_wrong_sample_steps(method):
    """Check that we raise a ValueError on invalid sample_steps"""
    # 创建 AdditiveChi2Sampler 的实例，设置一个无效的 sample_steps 值为 4
    transformer = AdditiveChi2Sampler(sample_steps=4)
    # 准备错误消息的正则表达式匹配字符串
    msg = re.escape(
        "If sample_steps is not in [1, 2, 3], you need to provide sample_interval"
    )
    # 使用 pytest 的 raises 断言检查是否会抛出 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        getattr(transformer, method)(X)


def test_skewed_chi2_sampler():
    # test that RBFSampler approximates kernel on random data

    # 计算精确的核函数值
    c = 0.03
    # 设置 Y 的一个元素为一个负值，但大于 c，确保核函数逼近在 (-c; +\infty) 上的有效性
    Y[0, 0] = -c / 2.0

    # 为了简化公式，定义一些缩写
    X_c = (X + c)[:, np.newaxis, :]
    Y_c = (Y + c)[np.newaxis, :, :]

    # 希望在对数空间中进行计算以提高稳定性
    log_kernel = (
        (np.log(X_c) / 2.0) + (np.log(Y_c) / 2.0) + np.log(2.0) - np.log(X_c + Y_c)
    )
    # 在对数空间中对特征求和，减少到 n_samples_x x n_samples_y 大小的核函数
    kernel = np.exp(log_kernel.sum(axis=2))

    # 近似核映射
    transform = SkewedChi2Sampler(skewedness=c, n_components=1000, random_state=42)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)
    # 断言核函数与近似核函数在精度 1 下相等
    assert_array_almost_equal(kernel, kernel_approx, 1)
    # 断言 Gram 矩阵中没有 NaN 值
    assert np.isfinite(kernel).all(), "NaNs found in the Gram matrix"
    assert np.isfinite(kernel_approx).all(), "NaNs found in the approximate Gram matrix"

    # 测试当输入包含小于 -c 的值时是否会抛出错误
    Y_neg = Y.copy()
    Y_neg[0, 0] = -c * 2.0
    msg = "X may not contain entries smaller than -skewedness"
    with pytest.raises(ValueError, match=msg):
        transform.transform(Y_neg)


def test_additive_chi2_sampler_exceptions():
    """Ensures correct error message"""
    # 创建 AdditiveChi2Sampler 的实例
    transformer = AdditiveChi2Sampler()
    # 复制 X 数据并将第一个元素设为 -1
    X_neg = X.copy()
    X_neg[0, 0] = -1
    # 使用 pytest 的 raises 断言检查是否会抛出 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.fit"):
        transformer.fit(X_neg)
    # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，并且异常信息中包含特定字符串 "X in AdditiveChi2Sampler.transform"
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.transform"):
        # 对 transformer 对象进行拟合操作，使用输入数据 X
        transformer.fit(X)
        # 对 transformer 对象进行转换操作，使用输入数据 X_neg
        transformer.transform(X_neg)
# 定义测试函数，用于测试 RBFSampler 是否能够在随机数据上近似核函数
def test_rbf_sampler():
    # 计算精确的核函数
    gamma = 10.0
    kernel = rbf_kernel(X, Y, gamma=gamma)

    # 创建 RBF 转换器
    rbf_transform = RBFSampler(gamma=gamma, n_components=1000, random_state=42)
    # 对输入数据 X 进行变换
    X_trans = rbf_transform.fit_transform(X)
    # 对输入数据 Y 进行变换
    Y_trans = rbf_transform.transform(Y)
    # 近似核函数的计算结果
    kernel_approx = np.dot(X_trans, Y_trans.T)

    # 计算误差
    error = kernel - kernel_approx
    # 断言误差的平均值接近于零
    assert np.abs(np.mean(error)) <= 0.01  # 接近无偏差
    # 将误差取绝对值
    np.abs(error, out=error)
    # 断言误差的最大值不超过 0.1
    assert np.max(error) <= 0.1  # 没有太大偏差
    # 断言误差的平均值不超过 0.05
    assert np.mean(error) <= 0.05  # 平均值相当接近


# 检查 RBFSampler 在不同数据类型 X 下的拟合属性数据类型是否正确
def test_rbf_sampler_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    rbf = RBFSampler()

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    rbf.fit(X)

    # 断言随机偏移量的数据类型正确
    assert rbf.random_offset_.dtype == global_dtype
    # 断言随机权重的数据类型正确
    assert rbf.random_weights_.dtype == global_dtype


# 检查在不同位数（32位和64位）输入时结果的等价性
def test_rbf_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    rbf32 = RBFSampler(random_state=42)
    X32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    rbf32.fit(X32)

    rbf64 = RBFSampler(random_state=42)
    X64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    rbf64.fit(X64)

    # 断言随机偏移量的结果在32位和64位输入下近似
    assert_allclose(rbf32.random_offset_, rbf64.random_offset_)
    # 断言随机权重的结果在32位和64位输入下近似
    assert_allclose(rbf32.random_weights_, rbf64.random_weights_)


# 检查当 `gamma='scale'` 时内部计算的值
def test_rbf_sampler_gamma_scale():
    """Check the inner value computed when `gamma='scale'`."""
    X, y = [[0.0], [1.0]], [0, 1]
    rbf = RBFSampler(gamma="scale")
    rbf.fit(X, y)
    # 断言计算出的内部 gamma 值近似于 4
    assert rbf._gamma == pytest.approx(4)


# 检查 SkewedChi2Sampler 在不同数据类型 X 下的拟合属性数据类型是否正确
def test_skewed_chi2_sampler_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    skewed_chi2_sampler = SkewedChi2Sampler()

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)

    skewed_chi2_sampler.fit(X)

    # 断言随机偏移量的数据类型正确
    assert skewed_chi2_sampler.random_offset_.dtype == global_dtype
    # 断言随机权重的数据类型正确
    assert skewed_chi2_sampler.random_weights_.dtype == global_dtype


# 检查在不同位数（32位和64位）输入时 SkewedChi2Sampler 结果的等价性
def test_skewed_chi2_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    skewed_chi2_sampler_32 = SkewedChi2Sampler(random_state=42)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    skewed_chi2_sampler_32.fit(X_32)

    skewed_chi2_sampler_64 = SkewedChi2Sampler(random_state=42)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    skewed_chi2_sampler_64.fit(X_64)

    # 断言随机偏移量的结果在32位和64位输入下近似
    assert_allclose(
        skewed_chi2_sampler_32.random_offset_, skewed_chi2_sampler_64.random_offset_
    )
    # 断言随机权重的结果在32位和64位输入下近似
    assert_allclose(
        skewed_chi2_sampler_32.random_weights_, skewed_chi2_sampler_64.random_weights_
    )


# 使用参数化测试来验证输入验证功能
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_input_validation(csr_container):
    # 回归测试：核近似变换器应该能够处理列表输入
    # 没有断言语句；旧版本可能会直接崩溃
    
    # 创建输入数据 X，这里是一个包含三个子列表的列表
    X = [[1, 2], [3, 4], [5, 6]]
    
    # 使用 AdditiveChi2Sampler 对象对 X 进行拟合和转换
    AdditiveChi2Sampler().fit(X).transform(X)
    
    # 使用 SkewedChi2Sampler 对象对 X 进行拟合和转换
    SkewedChi2Sampler().fit(X).transform(X)
    
    # 使用 RBFSampler 对象对 X 进行拟合和转换
    RBFSampler().fit(X).transform(X)
    
    # 将 X 转换为 CSR 格式的稀疏矩阵容器
    X = csr_container(X)
    
    # 使用 RBFSampler 对象对 CSR 格式的 X 进行拟合和转换
    RBFSampler().fit(X).transform(X)
def test_nystroem_approximation():
    # 定义一个测试函数，用于测试 Nystroem 近似方法的功能

    # 使用随机数生成器创建一个 10x4 的随机矩阵 X
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 4))

    # 当 n_components = n_samples 时，这是一个精确的近似
    # 使用 Nystroem 进行转换，并计算变换后的特征矩阵 X_transformed
    X_transformed = Nystroem(n_components=X.shape[0]).fit_transform(X)
    # 计算原始数据 X 的 RBF 核矩阵 K
    K = rbf_kernel(X)
    # 断言 X_transformed 与自身转置的点积近似等于 K
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    # 使用 Nystroem 转换数据，设定 n_components=2
    trans = Nystroem(n_components=2, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    # 断言变换后的特征矩阵形状为 (样本数, 2)
    assert X_transformed.shape == (X.shape[0], 2)

    # 测试可调用的核函数
    trans = Nystroem(n_components=2, kernel=_linear_kernel, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    # 断言变换后的特征矩阵形状为 (样本数, 2)
    assert X_transformed.shape == (X.shape[0], 2)

    # 测试可用的核函数是否能够适配和转换
    kernels_available = kernel_metrics()
    for kern in kernels_available:
        trans = Nystroem(n_components=2, kernel=kern, random_state=rnd)
        X_transformed = trans.fit(X).transform(X)
        # 断言变换后的特征矩阵形状为 (样本数, 2)
        assert X_transformed.shape == (X.shape[0], 2)


def test_nystroem_default_parameters():
    # 测试 Nystroem 默认参数的函数

    rnd = np.random.RandomState(42)
    X = rnd.uniform(size=(10, 4))

    # rbf 核函数默认情况下应当表现为 gamma=None
    # 即 gamma = 1 / 特征数
    nystroem = Nystroem(n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = rbf_kernel(X, gamma=None)
    K2 = np.dot(X_transformed, X_transformed.T)
    # 断言 K 和 K2 近似相等
    assert_array_almost_equal(K, K2)

    # chi2 核函数默认情况下应当表现为 gamma=1
    nystroem = Nystroem(kernel="chi2", n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = chi2_kernel(X, gamma=1)
    K2 = np.dot(X_transformed, X_transformed.T)
    # 断言 K 和 K2 近似相等
    assert_array_almost_equal(K, K2)


def test_nystroem_singular_kernel():
    # 测试 Nystroem 在奇异核矩阵上的工作情况

    # 使用随机数生成器创建一个 10x20 的随机矩阵 X
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    # 将样本复制一份，使得 X 变成 20 行的矩阵
    X = np.vstack([X] * 2)

    gamma = 100
    # 使用 Nystroem 进行转换，设定 gamma 和 n_components
    N = Nystroem(gamma=gamma, n_components=X.shape[0]).fit(X)
    X_transformed = N.transform(X)

    # 计算原始数据 X 的 RBF 核矩阵 K
    K = rbf_kernel(X, gamma=gamma)

    # 断言变换后的特征矩阵与自身转置的点积近似等于 K
    assert_array_almost_equal(K, np.dot(X_transformed, X_transformed.T))
    # 断言 Y 是有限的
    assert np.all(np.isfinite(Y))


def test_nystroem_poly_kernel_params():
    # 测试 Nystroem 在多项式核的参数传递上的情况

    rnd = np.random.RandomState(37)
    X = rnd.uniform(size=(10, 4))

    # 使用多项式核函数，设定 degree 和 coef0 参数
    K = polynomial_kernel(X, degree=3.1, coef0=0.1)
    nystroem = Nystroem(
        kernel="polynomial", n_components=X.shape[0], degree=3.1, coef0=0.1
    )
    X_transformed = nystroem.fit_transform(X)
    # 断言 X_transformed 与自身转置的点积近似等于 K
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)


def test_nystroem_callable():
    # 测试 Nystroem 在可调用核函数上的表现

    rnd = np.random.RandomState(42)
    n_samples = 10
    X = rnd.uniform(size=(n_samples, 4))

    def logging_histogram_kernel(x, y, log):
        """Histogram kernel that writes to a log."""
        log.append(1)
        return np.minimum(x, y).sum()

    kernel_log = []
    X = list(X)  # 测试输入验证
    # 使用 logging_histogram_kernel 作为核函数创建 Nystroem 核近似对象，使用 n_samples - 1 作为近似的组件数
    Nystroem(
        kernel=logging_histogram_kernel,
        n_components=(n_samples - 1),
        kernel_params={"log": kernel_log},
    ).fit(X)
    # 断言核函数日志的长度符合预期的数据点对数
    assert len(kernel_log) == n_samples * (n_samples - 1) / 2

    # 如果传递了 degree、gamma 或 coef0 参数，将引发 ValueError 错误
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    # 对于每个参数组合，使用 _linear_kernel 作为核函数创建 Nystroem 核近似对象，使用 n_samples - 1 作为近似的组件数
    for param in params:
        ny = Nystroem(kernel=_linear_kernel, n_components=(n_samples - 1), **param)
        # 使用 pytest 检查是否会引发特定的 ValueError 错误，错误消息应与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            ny.fit(X)
def test_nystroem_precomputed_kernel():
    """Non-regression: test Nystroem on precomputed kernel.
    PR - 14706"""
    # 创建一个随机数生成器对象，种子为12
    rnd = np.random.RandomState(12)
    # 生成一个大小为(10, 4)的均匀分布的随机矩阵
    X = rnd.uniform(size=(10, 4))

    # 使用二次多项式核函数计算核矩阵K
    K = polynomial_kernel(X, degree=2, coef0=0.1)
    # 初始化 Nystroem 类，指定使用预计算的核矩阵作为核函数，以及输出的特征数量
    nystroem = Nystroem(kernel="precomputed", n_components=X.shape[0])
    # 对核矩阵K进行拟合和转换
    X_transformed = nystroem.fit_transform(K)
    # 断言转换后的特征矩阵与原核矩阵K的内积近似相等
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    # 如果传入了degree、gamma或coef0参数，则引发 ValueError
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    for param in params:
        # 初始化 Nystroem 对象，传入预计算的核矩阵、输出的特征数量以及其他参数
        ny = Nystroem(kernel="precomputed", n_components=X.shape[0], **param)
        # 使用 pytest 检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            ny.fit(K)


def test_nystroem_component_indices():
    """Check that `component_indices_` corresponds to the subset of
    training points used to construct the feature map.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20474
    """
    # 生成100个样本和20个特征的分类数据集
    X, _ = make_classification(n_samples=100, n_features=20)
    # 初始化 Nystroem 类，指定输出的特征数量和随机状态
    feature_map_nystroem = Nystroem(
        n_components=10,
        random_state=0,
    )
    # 对数据集X进行拟合，构建特征映射
    feature_map_nystroem.fit(X)
    # 断言组件索引数组的形状为(10,)
    assert feature_map_nystroem.component_indices_.shape == (10,)


@pytest.mark.parametrize(
    "Estimator", [PolynomialCountSketch, RBFSampler, SkewedChi2Sampler, Nystroem]
)
def test_get_feature_names_out(Estimator):
    """Check get_feature_names_out"""
    # 初始化指定的 Estimator 类，并使用数据集X进行拟合
    est = Estimator().fit(X)
    # 对数据集X进行转换，生成转换后的特征表示
    X_trans = est.transform(X)

    # 获取转换后的特征名称
    names_out = est.get_feature_names_out()
    # 获取类名的小写形式
    class_name = Estimator.__name__.lower()
    # 生成预期的特征名称列表
    expected_names = [f"{class_name}{i}" for i in range(X_trans.shape[1])]
    # 断言转换后的特征名称数组与预期的特征名称列表相等
    assert_array_equal(names_out, expected_names)


def test_additivechi2sampler_get_feature_names_out():
    """Check get_feature_names_out for AdditiveChi2Sampler."""
    # 创建随机数生成器对象，种子为0
    rng = np.random.RandomState(0)
    # 生成一个大小为(300, 3)的随机样本矩阵
    X = rng.random_sample(size=(300, 3))

    # 初始化 AdditiveChi2Sampler 类，指定样本步数
    chi2_sampler = AdditiveChi2Sampler(sample_steps=3).fit(X)
    # 指定输入特征的名称列表
    input_names = ["f0", "f1", "f2"]
    # 定义特征名称的后缀列表
    suffixes = [
        "f0_sqrt",
        "f1_sqrt",
        "f2_sqrt",
        "f0_cos1",
        "f1_cos1",
        "f2_cos1",
        "f0_sin1",
        "f1_sin1",
        "f2_sin1",
        "f0_cos2",
        "f1_cos2",
        "f2_cos2",
        "f0_sin2",
        "f1_sin2",
        "f2_sin2",
    ]

    # 获取转换后的特征名称
    names_out = chi2_sampler.get_feature_names_out(input_features=input_names)
    # 生成预期的特征名称列表
    expected_names = [f"additivechi2sampler_{suffix}" for suffix in suffixes]
    # 断言转换后的特征名称数组与预期的特征名称列表相等
    assert_array_equal(names_out, expected_names)
```