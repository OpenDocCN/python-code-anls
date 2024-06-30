# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_data.py`

```
# 导入所需的库和模块
import re  # 导入正则表达式模块
import warnings  # 导入警告处理模块

import numpy as np  # 导入NumPy库并使用别名np
import numpy.linalg as la  # 导入NumPy线性代数模块并使用别名la
import pytest  # 导入pytest测试框架
from scipy import sparse, stats  # 导入SciPy稀疏矩阵和统计模块

from sklearn import datasets  # 导入scikit-learn的数据集模块
from sklearn.base import clone  # 导入scikit-learn的克隆函数
from sklearn.exceptions import NotFittedError  # 导入scikit-learn的未拟合错误
from sklearn.metrics.pairwise import linear_kernel  # 导入线性核函数
from sklearn.model_selection import cross_val_predict  # 导入交叉验证预测函数
from sklearn.pipeline import Pipeline  # 导入scikit-learn的管道模块
from sklearn.preprocessing import (  # 导入数据预处理模块
    Binarizer,
    KernelCenterer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    add_dummy_feature,
    maxabs_scale,
    minmax_scale,
    normalize,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale  # 导入数据预处理内部函数和常量
from sklearn.svm import SVR  # 导入支持向量机回归模型
from sklearn.utils import gen_batches, shuffle  # 导入生成批次函数和数据随机化函数
from sklearn.utils._array_api import (  # 导入数组API函数
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._testing import (  # 导入测试辅助函数
    _convert_container,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    skip_if_32bit,
)
from sklearn.utils.estimator_checks import (  # 导入评估器检查函数
    _get_check_estimator_ids,
    check_array_api_input_and_values,
)
from sklearn.utils.fixes import (  # 导入兼容性修复模块
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    LIL_CONTAINERS,
)
from sklearn.utils.sparsefuncs import mean_variance_axis  # 导入稀疏矩阵函数

iris = datasets.load_iris()  # 加载鸢尾花数据集

# 创建随机数生成器并生成数据用于多次使用
rng = np.random.RandomState(0)  # 创建随机数种子
n_features = 30  # 特征数量
n_samples = 1000  # 样本数量
offsets = rng.uniform(-1, 1, size=n_features)  # 特征偏移量数组
scales = rng.uniform(1, 10, size=n_features)  # 特征缩放系数数组
X_2d = rng.randn(n_samples, n_features) * scales + offsets  # 生成二维随机数据集
X_1row = X_2d[0, :].reshape(1, n_features)  # 从二维数据集中提取一行并转换为1行n_features列的形式
X_1col = X_2d[:, 0].reshape(n_samples, 1)  # 从二维数据集中提取一列并转换为n_samples行1列的形式
X_list_1row = X_1row.tolist()  # 将1行n_features列的数据转换为列表
X_list_1col = X_1col.tolist()  # 将n_samples行1列的数据转换为列表


def toarray(a):
    # 如果a具有toarray方法，则调用并返回其转换后的数组形式；否则直接返回a
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def _check_dim_1axis(a):
    # 返回数组a的第一维度的长度
    return np.asarray(a).shape[0]


def assert_correct_incr(i, batch_start, batch_stop, n, chunk_size, n_samples_seen):
    # 断言确保增量的正确性
    if batch_stop != n:
        # 如果批次结束不是最后一个批次，则检查已观察的样本数是否等于期望的增量
        assert (i + 1) * chunk_size == n_samples_seen
    else:
        # 如果批次结束是最后一个批次，则检查已观察的样本数是否等于期望的增量
        assert i * chunk_size + (batch_stop - batch_start) == n_samples_seen


def test_raises_value_error_if_sample_weights_greater_than_1d():
    # 测试函数：如果样本权重大于1维，则抛出值错误异常

    n_samples = [2, 3]  # 不同样本数量列表
    n_features = [3, 2]  # 不同特征数量列表

    for n_samples, n_features in zip(n_samples, n_features):
        # 遍历不同的样本数量和特征数量组合

        X = rng.randn(n_samples, n_features)  # 生成随机样本数据
        y = rng.randn(n_samples)  # 生成随机目标数据

        scaler = StandardScaler()  # 创建StandardScaler对象

        # 确保抛出异常：样本权重大于1维
        sample_weight_notOK = rng.randn(n_samples, 1) ** 2  # 生成不符合要求的样本权重
        with pytest.raises(ValueError):
            scaler.fit(X, y, sample_weight=sample_weight_notOK)
# 使用 pytest 的 parametrize 装饰器为 test_standard_scaler_sample_weight 函数提供多组参数化输入
@pytest.mark.parametrize(
    ["Xw", "X", "sample_weight"],
    [
        ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [1, 2, 3], [4, 5, 6]], [2.0, 1.0]),
        ([[1, 0, 1], [0, 0, 1]], [[1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], np.array([1, 3])),
        ([[1, np.nan, 1], [np.nan, np.nan, 1]],
         [[1, np.nan, 1], [np.nan, np.nan, 1], [np.nan, np.nan, 1], [np.nan, np.nan, 1]],
         np.array([1, 3])),
    ],
)
# 使用 pytest 的 parametrize 装饰器为 test_standard_scaler_sample_weight 函数提供多组参数化输入
@pytest.mark.parametrize("array_constructor", ["array", "sparse_csr", "sparse_csc"])
def test_standard_scaler_sample_weight(Xw, X, sample_weight, array_constructor):
    # 根据 array_constructor 判断是否使用稀疏矩阵容器
    with_mean = not array_constructor.startswith("sparse")
    # 将 X 和 Xw 转换为指定容器类型
    X = _convert_container(X, array_constructor)
    Xw = _convert_container(Xw, array_constructor)

    # 加权的标准化器
    yw = np.ones(Xw.shape[0])
    scaler_w = StandardScaler(with_mean=with_mean)
    scaler_w.fit(Xw, yw, sample_weight=sample_weight)

    # 无权重的标准化器，但具有重复样本
    y = np.ones(X.shape[0])
    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(X, y)

    # 测试数据
    X_test = [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]

    # 断言均值和方差的近似相等
    assert_almost_equal(scaler.mean_, scaler_w.mean_)
    assert_almost_equal(scaler.var_, scaler_w.var_)
    # 断言转换结果的近似相等
    assert_almost_equal(scaler.transform(X_test), scaler_w.transform(X_test))


# 单轴数据集的标准化测试
def test_standard_scaler_1d():
    # 测试不同的 X 输入数据
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:
        # 创建标准化器对象
        scaler = StandardScaler()
        # 对 X 进行标准化并转换
        X_scaled = scaler.fit(X).transform(X, copy=True)

        # 如果 X 是列表，仅在完成标准化后进行类型转换
        if isinstance(X, list):
            X = np.array(X)

        # 检查 X 的维度是否为 1
        if _check_dim_1axis(X) == 1:
            # 断言均值和标准差的近似相等
            assert_almost_equal(scaler.mean_, X.ravel())
            assert_almost_equal(scaler.scale_, np.ones(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.std(axis=0), np.zeros_like(n_features))
        else:
            # 断言均值和标准差的近似相等
            assert_almost_equal(scaler.mean_, X.mean())
            assert_almost_equal(scaler.scale_, X.std())
            assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros_like(n_features))
            assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
            assert_array_almost_equal(X_scaled.std(axis=0), 1.0)
        # 断言 n_samples_seen_ 等于 X 的行数
        assert scaler.n_samples_seen_ == X.shape[0]

        # 检查逆转换
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # 常数特征数据集
    X = np.ones((5, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_almost_equal(scaler.mean_, 1.0)
    assert_almost_equal(scaler.scale_, 1.0)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 0.0)
    # 确保scaler对象中记录的样本数与输入数据X的行数相等
    assert scaler.n_samples_seen_ == X.shape[0]
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
# 使用pytest的参数化标记，为sparse_container参数指定多个测试参数：None，CSC_CONTAINERS，CSR_CONTAINERS
@pytest.mark.parametrize("add_sample_weight", [False, True])
# 使用pytest的参数化标记，为add_sample_weight参数指定两个测试参数：False和True
def test_standard_scaler_dtype(add_sample_weight, sparse_container):
    # 测试函数：test_standard_scaler_dtype，用于验证标准缩放器（StandardScaler）对数据类型的影响

    # 设定随机数种子，确保结果可复现
    rng = np.random.RandomState(0)
    n_samples = 10  # 样本数为10
    n_features = 3   # 特征数为3

    if add_sample_weight:
        sample_weight = np.ones(n_samples)  # 如果add_sample_weight为True，设置样本权重为全1数组
    else:
        sample_weight = None  # 否则样本权重为None

    with_mean = True  # 是否中心化，默认为True

    if sparse_container is not None:
        # 如果sparse_container不为None，即使用稀疏容器时
        # scipy的稀疏容器不支持float16类型，参见https://github.com/scipy/scipy/issues/7408
        supported_dtype = [np.float64, np.float32]  # 支持的数据类型为np.float64和np.float32
    else:
        supported_dtype = [np.float64, np.float32, np.float16]  # 否则支持的数据类型为np.float64, np.float32, np.float16

    for dtype in supported_dtype:
        # 遍历每种支持的数据类型

        X = rng.randn(n_samples, n_features).astype(dtype)  # 生成随机数据，并转换为指定数据类型

        if sparse_container is not None:
            X = sparse_container(X)  # 如果使用稀疏容器，将数据转换为稀疏格式
            with_mean = False  # 不进行均值中心化

        scaler = StandardScaler(with_mean=with_mean)  # 创建标准缩放器对象
        X_scaled = scaler.fit(X, sample_weight=sample_weight).transform(X)  # 对数据进行拟合和转换

        assert X.dtype == X_scaled.dtype  # 断言原始数据和缩放后数据的数据类型相同
        assert scaler.mean_.dtype == np.float64  # 断言均值的数据类型为np.float64
        assert scaler.scale_.dtype == np.float64  # 断言标准差的数据类型为np.float64


@pytest.mark.parametrize(
    "scaler",
    [
        StandardScaler(with_mean=False),  # 创建不进行均值中心化的标准缩放器对象
        RobustScaler(with_centering=False),  # 创建不进行中心化的RobustScaler对象
    ],
)
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
# 使用pytest的参数化标记，为sparse_container参数指定多个测试参数：None，CSC_CONTAINERS，CSR_CONTAINERS
@pytest.mark.parametrize("add_sample_weight", [False, True])
# 使用pytest的参数化标记，为add_sample_weight参数指定两个测试参数：False和True
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
# 使用pytest的参数化标记，为dtype参数指定两个测试参数：np.float32和np.float64
@pytest.mark.parametrize("constant", [0, 1.0, 100.0])
# 使用pytest的参数化标记，为constant参数指定多个测试参数：0, 1.0, 100.0
def test_standard_scaler_constant_features(
    scaler, add_sample_weight, sparse_container, dtype, constant
):
    # 测试函数：test_standard_scaler_constant_features，用于验证标准缩放器（StandardScaler）对常数特征的处理

    if isinstance(scaler, RobustScaler) and add_sample_weight:
        pytest.skip(f"{scaler.__class__.__name__} does not yet support sample_weight")
        # 如果是RobustScaler且add_sample_weight为True，则跳过该测试，因为RobustScaler暂不支持sample_weight

    rng = np.random.RandomState(0)
    n_samples = 100  # 样本数为100
    n_features = 1    # 特征数为1

    if add_sample_weight:
        fit_params = dict(sample_weight=rng.uniform(size=n_samples) * 2)  # 如果add_sample_weight为True，设置样本权重
    else:
        fit_params = {}  # 否则fit_params为空字典

    X_array = np.full(shape=(n_samples, n_features), fill_value=constant, dtype=dtype)
    # 创建全为constant值的数据数组X_array

    X = X_array if sparse_container is None else sparse_container(X_array)
    # 如果使用稀疏容器，将X_array转换为稀疏格式，否则不变

    X_scaled = scaler.fit(X, **fit_params).transform(X)  # 对数据进行拟合和转换

    if isinstance(scaler, StandardScaler):
        # 如果是标准缩放器（StandardScaler）
        # 常数特征的方差信息应接近于零
        assert_allclose(scaler.var_, np.zeros(X.shape[1]), atol=1e-7)

    # 常数特征的缩放比例应为1.0
    assert_allclose(scaler.scale_, np.ones(X.shape[1]))

    assert X_scaled is not X  # 确保进行了数据复制
    assert_allclose_dense_sparse(X_scaled, X)
    # 断言稠密和稀疏数据在数值上接近
    # 检查scaler是否为StandardScaler类型，并且不添加样本权重
    if isinstance(scaler, StandardScaler) and not add_sample_weight:
        # 同时检查与标准缩放函数的一致性。
        X_scaled_2 = scale(X, with_mean=scaler.with_mean)
        # 确保我们进行了复制操作，而不是引用相同对象。
        assert X_scaled_2 is not X
        # 检查X_scaled_2和X是否在数值上非常接近，用于验证缩放的正确性。
        assert_allclose_dense_sparse(X_scaled_2, X)
@pytest.mark.parametrize("n_samples", [10, 100, 10_000])
@pytest.mark.parametrize("average", [1e-10, 1, 1e10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_standard_scaler_near_constant_features(
    n_samples, sparse_container, average, dtype
):
    # Check that when the variance is too small (var << mean**2) the feature
    # is considered constant and not scaled.

    scale_min, scale_max = -30, 19
    scales = np.array([10**i for i in range(scale_min, scale_max + 1)], dtype=dtype)

    n_features = scales.shape[0]
    X = np.empty((n_samples, n_features), dtype=dtype)
    # Make a dataset of known var = scales**2 and mean = average
    X[: n_samples // 2, :] = average + scales
    X[n_samples // 2 :, :] = average - scales
    X_array = X if sparse_container is None else sparse_container(X)

    scaler = StandardScaler(with_mean=False).fit(X_array)

    # StandardScaler uses float64 accumulators even if the data has a float32
    # dtype.
    eps = np.finfo(np.float64).eps

    # if var < bound = N.eps.var + N².eps².mean², the feature is considered
    # constant and the scale_ attribute is set to 1.
    bounds = n_samples * eps * scales**2 + n_samples**2 * eps**2 * average**2
    within_bounds = scales**2 <= bounds

    # Check that scale_min is small enough to have some scales below the
    # bound and therefore detected as constant:
    assert np.any(within_bounds)

    # Check that such features are actually treated as constant by the scaler:
    assert all(scaler.var_[within_bounds] <= bounds[within_bounds])
    assert_allclose(scaler.scale_[within_bounds], 1.0)

    # Depending the on the dtype of X, some features might not actually be
    # representable as non constant for small scales (even if above the
    # precision bound of the float64 variance estimate). Such feature should
    # be correctly detected as constants with 0 variance by StandardScaler.
    representable_diff = X[0, :] - X[-1, :] != 0
    assert_allclose(scaler.var_[np.logical_not(representable_diff)], 0)
    assert_allclose(scaler.scale_[np.logical_not(representable_diff)], 1)

    # The other features are scaled and scale_ is equal to sqrt(var_) assuming
    # that scales are large enough for average + scale and average - scale to
    # be distinct in X (depending on X's dtype).
    common_mask = np.logical_and(scales**2 > bounds, representable_diff)
    assert_allclose(scaler.scale_[common_mask], np.sqrt(scaler.var_)[common_mask])


def test_scale_1d():
    # 1-d inputs
    X_list = [1.0, 3.0, 5.0, 0.0]
    X_arr = np.array(X_list)

    for X in [X_list, X_arr]:
        X_scaled = scale(X)
        assert_array_almost_equal(X_scaled.mean(), 0.0)
        assert_array_almost_equal(X_scaled.std(), 1.0)
        assert_array_equal(scale(X, with_mean=False, with_std=False), X)


@skip_if_32bit
def test_standard_scaler_numerical_stability():
    # This test is skipped if the platform is 32-bit.
    # Test numerical stability of scaling
    # 测试缩放的数值稳定性

    # np.log(1e-5) is taken because of its floating point representation
    # was empirically found to cause numerical problems with np.mean & np.std.
    # 使用 np.log(1e-5) 是因为其浮点表示在实际使用中被发现会导致 np.mean 和 np.std 函数出现数值问题。

    x = np.full(8, np.log(1e-5), dtype=np.float64)
    # 创建一个包含 8 个元素的数组，每个元素的值为 np.log(1e-5)，数据类型为 np.float64

    # This does not raise a warning as the number of samples is too low
    # to trigger the problem in recent numpy
    # 这不会引发警告，因为样本数量太少，无法在最近的 numpy 中触发问题。

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 捕获 UserWarning 类型的警告，并将其设为错误级别
        scale(x)
    assert_array_almost_equal(scale(x), np.zeros(8))
    # 断言 scale 函数处理后的结果与全零数组 np.zeros(8) 几乎相等

    # with 2 more samples, the std computation run into numerical issues:
    # 加上两个样本，标准差计算会遇到数值问题：

    x = np.full(10, np.log(1e-5), dtype=np.float64)
    warning_message = "standard deviation of the data is probably very close to 0"
    # 设置警告信息字符串，指出数据的标准差可能非常接近 0

    with pytest.warns(UserWarning, match=warning_message):
        # 使用 pytest 的 warn 函数捕获 UserWarning，并匹配警告信息字符串
        x_scaled = scale(x)
    assert_array_almost_equal(x_scaled, np.zeros(10))
    # 断言 scale 函数处理后的结果与全零数组 np.zeros(10) 几乎相等

    x = np.full(10, 1e-100, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 捕获 UserWarning 类型的警告，并将其设为错误级别
        x_small_scaled = scale(x)
    assert_array_almost_equal(x_small_scaled, np.zeros(10))
    # 断言 scale 函数处理后的结果与全零数组 np.zeros(10) 几乎相等

    # Large values can cause (often recoverable) numerical stability issues:
    # 大数值可能导致（通常是可恢复的）数值稳定性问题：

    x_big = np.full(10, 1e100, dtype=np.float64)
    warning_message = "Dataset may contain too large values"
    # 设置警告信息字符串，指出数据集可能包含过大的值

    with pytest.warns(UserWarning, match=warning_message):
        # 使用 pytest 的 warn 函数捕获 UserWarning，并匹配警告信息字符串
        x_big_scaled = scale(x_big)
    assert_array_almost_equal(x_big_scaled, np.zeros(10))
    # 断言 scale 函数处理后的结果与全零数组 np.zeros(10) 几乎相等
    assert_array_almost_equal(x_big_scaled, x_small_scaled)
    # 断言大数值和小数值经过 scale 函数处理后结果几乎相等

    with pytest.warns(UserWarning, match=warning_message):
        # 使用 pytest 的 warn 函数捕获 UserWarning，并匹配警告信息字符串
        x_big_centered = scale(x_big, with_std=False)
    assert_array_almost_equal(x_big_centered, np.zeros(10))
    # 断言 scale 函数处理后的结果与全零数组 np.zeros(10) 几乎相等
    assert_array_almost_equal(x_big_centered, x_small_scaled)
    # 断言大数值和小数值经过 scale 函数处理后结果几乎相等
# 定义用于测试二维数组缩放的函数
def test_scaler_2d_arrays():
    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)
    # 设定特征数和样本数
    n_features = 5
    n_samples = 4
    # 生成一个形状为(n_samples, n_features)的随机标准正态分布数组
    X = rng.randn(n_samples, n_features)
    # 将所有样本的第一个特征设置为0
    X[:, 0] = 0.0  

    # 创建一个StandardScaler对象
    scaler = StandardScaler()
    # 对数组进行拟合和转换，返回经过缩放后的数组
    X_scaled = scaler.fit(X).transform(X, copy=True)
    # 断言没有任何元素为NaN
    assert not np.any(np.isnan(X_scaled))
    # 断言scaler对象观察到的样本数等于n_samples
    assert scaler.n_samples_seen_ == n_samples

    # 断言经过缩放后的均值在每列上接近0
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    # 断言经过缩放后的标准差在每列上接近[0.0, 1.0, 1.0, 1.0, 1.0]
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # 检查X是否被复制
    assert X_scaled is not X

    # 检查逆转换
    X_scaled_back = scaler.inverse_transform(X_scaled)
    # 断言X_scaled_back不是X
    assert X_scaled_back is not X
    # 断言X_scaled_back不是X_scaled
    assert X_scaled_back is not X_scaled
    # 断言X_scaled_back接近于原始数组X
    assert_array_almost_equal(X_scaled_back, X)

    # 对X进行无标准差缩放，按行缩放
    X_scaled = scale(X, axis=1, with_std=False)
    # 断言没有任何元素为NaN
    assert not np.any(np.isnan(X_scaled))
    # 断言按行缩放后的均值为每个样本数乘以0
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    
    # 对X进行有标准差缩放，按行缩放
    X_scaled = scale(X, axis=1, with_std=True)
    # 断言没有任何元素为NaN
    assert not np.any(np.isnan(X_scaled))
    # 断言按行缩放后的均值为每个样本数乘以0
    assert_array_almost_equal(X_scaled.mean(axis=1), n_samples * [0.0])
    # 断言按行缩放后的标准差为每个样本数乘以1
    assert_array_almost_equal(X_scaled.std(axis=1), n_samples * [1.0])
    # 检查数据是否被修改
    assert X_scaled is not X

    # 对X进行拟合和转换，不复制数组
    X_scaled = scaler.fit(X).transform(X, copy=False)
    # 断言没有任何元素为NaN
    assert not np.any(np.isnan(X_scaled))
    # 断言拟合和转换后的均值在每列上接近0
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    # 断言拟合和转换后的标准差在每列上接近[0.0, 1.0, 1.0, 1.0, 1.0]
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # 检查X是否未被复制
    assert X_scaled is X

    # 重新生成X数组，形状为(4, 5)
    X = rng.randn(4, 5)
    # 将所有样本的第一个特征设置为1.0，非零特征
    X[:, 0] = 1.0  
    # 创建一个StandardScaler对象
    scaler = StandardScaler()
    # 对数组进行拟合和转换，返回经过缩放后的数组
    X_scaled = scaler.fit(X).transform(X, copy=True)
    # 断言没有任何元素为NaN
    assert not np.any(np.isnan(X_scaled))
    # 断言经过缩放后的均值在每列上接近0
    assert_array_almost_equal(X_scaled.mean(axis=0), n_features * [0.0])
    # 断言经过缩放后的标准差在每列上接近[0.0, 1.0, 1.0, 1.0, 1.0]
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # 检查X是否未被复制
    assert X_scaled is not X
    # 使用 assert_array_almost_equal 函数进行断言，比较两个数组 X_scaled 和 X_scaled_f64 是否几乎相等
    # decimal=2 表示检查精度，保留两位小数以考虑精度差异
    assert_array_almost_equal(X_scaled, X_scaled_f64, decimal=2)
# 测试函数：处理标尺中的零值情况
def test_handle_zeros_in_scale():
    # 创建一个 NumPy 数组，包含数值为 [0, 1e-16, 1, 2, 3]
    s1 = np.array([0, 1e-16, 1, 2, 3])
    # 调用 _handle_zeros_in_scale 函数处理 s1 数组，使用复制副本的方式
    s2 = _handle_zeros_in_scale(s1, copy=True)

    # 断言 s1 数组的值接近给定的数组 [0, 1e-16, 1, 2, 3]
    assert_allclose(s1, np.array([0, 1e-16, 1, 2, 3]))
    # 断言 s2 数组的值接近给定的数组 [1, 1, 1, 2, 3]
    assert_allclose(s2, np.array([1, 1, 1, 2, 3]))


# 测试函数：部分拟合最小最大值缩放器
def test_minmax_scaler_partial_fit():
    # 测试部分拟合在大小为 1 和 50 的多个批次上是否与完全拟合的结果相同
    X = X_2d
    n = X.shape[0]

    # 遍历不同的批次大小进行测试
    for chunk_size in [1, 2, 50, n, n + 42]:
        # 测试过程末尾的均值
        scaler_batch = MinMaxScaler().fit(X)

        scaler_incr = MinMaxScaler()
        # 针对每个批次进行部分拟合
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])

        # 断言 scaler_batch 的最小数据等于 scaler_incr 的最小数据
        assert_array_almost_equal(scaler_batch.data_min_, scaler_incr.data_min_)
        # 断言 scaler_batch 的最大数据等于 scaler_incr 的最大数据
        assert_array_almost_equal(scaler_batch.data_max_, scaler_incr.data_max_)
        # 断言 scaler_batch 的样本数等于 scaler_incr 的样本数
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        # 断言 scaler_batch 的数据范围等于 scaler_incr 的数据范围
        assert_array_almost_equal(scaler_batch.data_range_, scaler_incr.data_range_)
        # 断言 scaler_batch 的比例等于 scaler_incr 的比例
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        # 断言 scaler_batch 的最小值等于 scaler_incr 的最小值
        assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

        # 测试第一步后的标准差
        batch0 = slice(0, chunk_size)
        scaler_batch = MinMaxScaler().fit(X[batch0])
        scaler_incr = MinMaxScaler().partial_fit(X[batch0])

        # 断言 scaler_batch 的最小数据等于 scaler_incr 的最小数据
        assert_array_almost_equal(scaler_batch.data_min_, scaler_incr.data_min_)
        # 断言 scaler_batch 的最大数据等于 scaler_incr 的最大数据
        assert_array_almost_equal(scaler_batch.data_max_, scaler_incr.data_max_)
        # 断言 scaler_batch 的样本数等于 scaler_incr 的样本数
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        # 断言 scaler_batch 的数据范围等于 scaler_incr 的数据范围
        assert_array_almost_equal(scaler_batch.data_range_, scaler_incr.data_range_)
        # 断言 scaler_batch 的比例等于 scaler_incr 的比例
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        # 断言 scaler_batch 的最小值等于 scaler_incr 的最小值
        assert_array_almost_equal(scaler_batch.min_, scaler_incr.min_)

        # 测试直到部分拟合结束后的标准差，以及
        scaler_batch = MinMaxScaler().fit(X)
        scaler_incr = MinMaxScaler()  # 清除估计器状态
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            # 断言递增部分中的正确性
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )


# 测试函数：部分拟合标准缩放器
def test_standard_scaler_partial_fit():
    # 测试部分拟合在大小为 1 和 50 的多个批次上是否与完全拟合的结果相同
    X = X_2d
    n = X.shape[0]
    for chunk_size in [1, 2, 50, n, n + 42]:
        # 使用不同的 chunk_size 迭代：1, 2, 50, n, n + 42

        # 在整个处理过程结束时测试均值
        scaler_batch = StandardScaler(with_std=False).fit(X)

        # 创建一个新的 StandardScaler 对象，用于增量学习
        scaler_incr = StandardScaler(with_std=False)
        for batch in gen_batches(n_samples, chunk_size):
            # 对当前批次的数据进行增量学习
            scaler_incr = scaler_incr.partial_fit(X[batch])
        
        # 断言批处理后的均值应该一致
        assert_array_almost_equal(scaler_batch.mean_, scaler_incr.mean_)
        
        # 断言批处理后的方差应该一致（注意：可能是 None）
        assert scaler_batch.var_ == scaler_incr.var_
        
        # 断言批处理后的样本数应该一致
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_

        # 在第一个批次之后测试标准差
        batch0 = slice(0, chunk_size)
        scaler_incr = StandardScaler().partial_fit(X[batch0])
        if chunk_size == 1:
            # 当 chunk_size 为 1 时，断言方差为零，标准差为一
            assert_array_almost_equal(
                np.zeros(n_features, dtype=np.float64), scaler_incr.var_
            )
            assert_array_almost_equal(
                np.ones(n_features, dtype=np.float64), scaler_incr.scale_
            )
        else:
            # 否则，使用批次数据计算方差和标准差
            assert_array_almost_equal(np.var(X[batch0], axis=0), scaler_incr.var_)
            assert_array_almost_equal(
                np.std(X[batch0], axis=0), scaler_incr.scale_
            )

        # 在所有增量学习步骤结束后测试标准差和方差
        scaler_batch = StandardScaler().fit(X)
        scaler_incr = StandardScaler()  # 清空评估器
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            # 断言增量学习后的各种属性正确性
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )

        # 断言整体数据的方差应该一致
        assert_array_almost_equal(scaler_batch.var_, scaler_incr.var_)
        
        # 断言整体数据的样本数应该一致
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
# 使用 pytest 的 parametrize 装饰器，测试函数参数化，对每种稀疏容器分别执行测试
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_standard_scaler_partial_fit_numerical_stability(sparse_container):
    # 测试增量计算在大数据集和大幅度数值下是否引入显著误差
    rng = np.random.RandomState(0)
    n_features = 2
    n_samples = 100
    # 生成偏移量和缩放比例数组
    offsets = rng.uniform(-1e15, 1e15, size=n_features)
    scales = rng.uniform(1e3, 1e6, size=n_features)
    # 生成数据矩阵 X
    X = rng.randn(n_samples, n_features) * scales + offsets

    # 使用全数据进行标准化拟合
    scaler_batch = StandardScaler().fit(X)
    # 使用增量方式初始化标准化器
    scaler_incr = StandardScaler()
    for chunk in X:
        # 逐个样本块进行增量拟合
        scaler_incr = scaler_incr.partial_fit(chunk.reshape(1, n_features))

    # 断言增量拟合结果的均值、方差和标准差与全数据拟合结果在指定精度内相等
    tol = 10 ** (-6)
    assert_allclose(scaler_incr.mean_, scaler_batch.mean_, rtol=tol)
    assert_allclose(scaler_incr.var_, scaler_batch.var_, rtol=tol)
    assert_allclose(scaler_incr.scale_, scaler_batch.scale_, rtol=tol)
    # 注意：对于更大的偏移量，标准差非常不稳定（最后一个断言），而均值没有问题。

    # 稀疏输入
    size = (100, 3)
    scale = 1e20
    X = sparse_container(rng.randint(0, 2, size).astype(np.float64) * scale)

    # 对于稀疏输入，需要设置 with_mean=False
    scaler = StandardScaler(with_mean=False).fit(X)
    scaler_incr = StandardScaler(with_mean=False)

    for chunk in X:
        if chunk.ndim == 1:
            # 稀疏数组可以是1维（在 scipy 1.14及更高版本中），而旧的稀疏矩阵实例总是2维。
            chunk = chunk.reshape(1, -1)
        # 使用增量方式拟合
        scaler_incr = scaler_incr.partial_fit(chunk)

    # 断言增量拟合结果的方差和标准差与全数据拟合结果在指定精度内相等
    tol = 10 ** (-6)
    assert scaler.mean_ is not None
    assert_allclose(scaler_incr.var_, scaler.var_, rtol=tol)
    assert_allclose(scaler_incr.scale_, scaler.scale_, rtol=tol)


# 对于不同的 sample_weight 参数，以及各种稀疏容器，测试部分拟合对稀疏输入的影响
@pytest.mark.parametrize("sample_weight", [True, None])
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_partial_fit_sparse_input(sample_weight, sparse_container):
    # 检查稀疏性是否被破坏
    X = sparse_container(np.array([[1.0], [0.0], [0.0], [5.0]]))

    if sample_weight:
        sample_weight = rng.rand(X.shape[0])

    # 创建一个不执行任何转换的 StandardScaler 对象
    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    # 对 X 执行部分拟合，然后转换
    X_null = null_transform.partial_fit(X, sample_weight=sample_weight).transform(X)
    # 断言转换后的稀疏数组与原始稀疏数组完全相同
    assert_array_equal(X_null.toarray(), X.toarray())
    # 对转换后的稀疏数组执行逆转换，再次断言结果与原始稀疏数组相同
    X_orig = null_transform.inverse_transform(X_null)
    assert_array_equal(X_orig.toarray(), X_null.toarray())
    assert_array_equal(X_orig.toarray(), X.toarray())


# 对于不同的 sample_weight 参数，测试部分拟合和转换后的标准化结果
@pytest.mark.parametrize("sample_weight", [True, None])
def test_standard_scaler_trasform_with_partial_fit(sample_weight):
    # 在应用部分拟合和转换后，检查一些后置条件
    X = X_2d[:100, :]

    if sample_weight:
        sample_weight = rng.rand(X.shape[0])
    # 创建一个标准化缩放器对象，用于处理数据批次
    scaler_incr = StandardScaler()
    # 遍历生成器产生的每个数据批次
    for i, batch in enumerate(gen_batches(X.shape[0], 1)):
        # 从数据集中获取当前批次的数据
        X_sofar = X[: (i + 1), :]
        # 复制当前数据批次，用于后续断言比较
        chunks_copy = X_sofar.copy()
        # 如果没有样本权重，使用标准化缩放器对当前批次数据进行标准化
        if sample_weight is None:
            scaled_batch = StandardScaler().fit_transform(X_sofar)
            # 对当前批次数据进行增量拟合到全局缩放器
            scaler_incr = scaler_incr.partial_fit(X[batch])
        else:
            # 使用带有样本权重的标准化缩放器对当前批次数据进行标准化
            scaled_batch = StandardScaler().fit_transform(
                X_sofar, sample_weight=sample_weight[: i + 1]
            )
            # 对当前批次数据及其权重进行增量拟合到全局缩放器
            scaler_incr = scaler_incr.partial_fit(
                X[batch], sample_weight=sample_weight[batch]
            )
        # 使用增量缩放器对当前数据批次进行转换
        scaled_incr = scaler_incr.transform(X_sofar)

        # 断言当前批次经过标准化后的数据与增量缩放器转换的结果相等
        assert_array_almost_equal(scaled_batch, scaled_incr)
        # 断言当前数据批次未被修改
        assert_array_almost_equal(X_sofar, chunks_copy)  # No change
        # 使用增量缩放器对标准化后的数据进行逆转换
        right_input = scaler_incr.inverse_transform(scaled_incr)
        # 断言逆转换后的数据与原始数据相等
        assert_array_almost_equal(X_sofar, right_input)

        # 创建一个全零数组，用于检查增量缩放器的方差和标准差是否大于零
        zero = np.zeros(X.shape[1])
        epsilon = np.finfo(float).eps
        # 断言增量缩放器的方差和标准差均大于零
        assert_array_less(zero, scaler_incr.var_ + epsilon)  # as less or equal
        assert_array_less(zero, scaler_incr.scale_ + epsilon)
        # 如果没有样本权重，断言增量缩放器已经观察到的样本数等于 (i+1)
        if sample_weight is None:
            assert (i + 1) == scaler_incr.n_samples_seen_
        else:
            # 如果有样本权重，断言样本权重的累加和等于增量缩放器已观察到的样本数
            assert np.sum(sample_weight[: i + 1]) == pytest.approx(
                scaler_incr.n_samples_seen_
            )
# 测试标准化器的逆转换功能
def test_standard_check_array_of_inverse_transform():
    # 创建一个整数类型的 NumPy 数组作为输入数据
    x = np.array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 8, 0, 1, 0, 0],
            [1, 4, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 4, 0, 1, 0, 1],
        ],
        dtype=np.int32,
    )

    # 实例化标准化器对象
    scaler = StandardScaler()
    # 使用输入数据 x 对标准化器进行拟合
    scaler.fit(x)

    # 对 x 进行逆转换，期望结果为浮点数数组
    scaler.inverse_transform(x)


# 使用参数化测试框架对多个估算器进行 API 一致性测试
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "check",
    [check_array_api_input_and_values],
    ids=_get_check_estimator_ids,
)
@pytest.mark.parametrize(
    "estimator",
    [
        MaxAbsScaler(),
        MinMaxScaler(),
        KernelCenterer(),
        Normalizer(norm="l1"),
        Normalizer(norm="l2"),
        Normalizer(norm="max"),
    ],
    ids=_get_check_estimator_ids,
)
def test_scaler_array_api_compliance(
    estimator, check, array_namespace, device, dtype_name
):
    # 获取估算器的类名
    name = estimator.__class__.__name__
    # 调用检查函数，测试估算器的 API 一致性
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)


# 测试 MinMaxScaler 在鸢尾花数据集上的表现
def test_min_max_scaler_iris():
    # 加载鸢尾花数据集
    X = iris.data
    # 实例化 MinMaxScaler，默认参数
    scaler = MinMaxScaler()
    # 对数据 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)
    # 断言转换后数据的最小值列为 0
    assert_array_almost_equal(X_trans.min(axis=0), 0)
    # 断言转换后数据的最大值列为 1
    assert_array_almost_equal(X_trans.max(axis=0), 1)
    # 对转换后的数据进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言逆转换后的数据与原始数据 X 相等
    assert_array_almost_equal(X, X_trans_inv)

    # 使用非默认参数：min=1, max=2
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), 1)
    assert_array_almost_equal(X_trans.max(axis=0), 2)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # 使用参数：min=-0.5, max=0.6
    scaler = MinMaxScaler(feature_range=(-0.5, 0.6))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), -0.5)
    assert_array_almost_equal(X_trans.max(axis=0), 0.6)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # 测试无效范围时是否引发异常
    scaler = MinMaxScaler(feature_range=(2, 1))
    with pytest.raises(ValueError):
        scaler.fit(X)


# 测试 MinMaxScaler 在具有零方差特征的玩具数据上的表现
def test_min_max_scaler_zero_variance_features():
    # 使用具有零方差特征的玩具数据进行测试
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]]
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]

    # 默认参数下实例化 MinMaxScaler
    scaler = MinMaxScaler()
    # 对数据 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)
    # 期望的转换后的数据
    X_expected_0_1 = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # 断言转换后的数据与期望数据一致
    assert_array_almost_equal(X_trans, X_expected_0_1)
    # 对转换后的数据进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言：验证 X 与 X_trans_inv 的近似相等性
    assert_array_almost_equal(X, X_trans_inv)

    # 使用缩放器对新数据 X_new 进行转换
    X_trans_new = scaler.transform(X_new)
    # 预期的经过缩放后的数据 X_expected_0_1_new
    X_expected_0_1_new = [[+0.0, 1.0, 0.500], [-1.0, 0.0, 0.083], [+0.0, 0.0, 1.333]]
    # 断言：验证 X_trans_new 与 X_expected_0_1_new 的近似相等性，精度为小数点后两位
    assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)

    # 使用非默认参数创建 MinMaxScaler 缩放器
    scaler = MinMaxScaler(feature_range=(1, 2))
    # 对 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)
    # 预期的经过缩放后的数据 X_expected_1_2
    X_expected_1_2 = [[1.0, 1.0, 1.5], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]
    # 断言：验证 X_trans 与 X_expected_1_2 的近似相等性
    assert_array_almost_equal(X_trans, X_expected_1_2)

    # 使用函数接口进行最小-最大缩放
    X_trans = minmax_scale(X)
    # 断言：验证 X_trans 与 X_expected_0_1 的近似相等性
    assert_array_almost_equal(X_trans, X_expected_0_1)
    # 使用函数接口进行最小-最大缩放，并指定特征范围为 (1, 2)
    X_trans = minmax_scale(X, feature_range=(1, 2))
    # 断言：验证 X_trans 与 X_expected_1_2 的近似相等性
    assert_array_almost_equal(X_trans, X_expected_1_2)
def test_minmax_scale_axis1():
    # 获取 Iris 数据集中的特征数据
    X = iris.data
    # 对数据集 X 进行按行（axis=1）的最小-最大缩放
    X_trans = minmax_scale(X, axis=1)
    # 断言按行缩放后的最小值接近于 0
    assert_array_almost_equal(np.min(X_trans, axis=1), 0)
    # 断言按行缩放后的最大值接近于 1
    assert_array_almost_equal(np.max(X_trans, axis=1), 1)


def test_min_max_scaler_1d():
    # 测试沿单个轴向缩放数据集
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:
        # 创建 MinMaxScaler 对象
        scaler = MinMaxScaler(copy=True)
        # 对数据集 X 进行拟合和转换
        X_scaled = scaler.fit(X).transform(X)

        if isinstance(X, list):
            X = np.array(X)  # 仅在缩放完成后进行类型转换

        # 如果数据 X 的维度为 1，则断言按轴 0 缩放后的最小值接近于零向量
        if _check_dim_1axis(X) == 1:
            assert_array_almost_equal(X_scaled.min(axis=0), np.zeros(n_features))
            assert_array_almost_equal(X_scaled.max(axis=0), np.zeros(n_features))
        else:
            # 否则断言按轴 0 缩放后的最小值接近于 0.0
            assert_array_almost_equal(X_scaled.min(axis=0), 0.0)
            # 断言按轴 0 缩放后的最大值接近于 1.0
            assert_array_almost_equal(X_scaled.max(axis=0), 1.0)
        # 断言 scaler 记录的样本数量与数据集 X 的行数相同
        assert scaler.n_samples_seen_ == X.shape[0]

        # 检查逆变换
        X_scaled_back = scaler.inverse_transform(X_scaled)
        # 断言逆变换后的数据接近于原始数据 X
        assert_array_almost_equal(X_scaled_back, X)

    # 常数特征
    X = np.ones((5, 1))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit(X).transform(X)
    # 断言缩放后的数据最小值大于等于 0.0
    assert X_scaled.min() >= 0.0
    # 断言缩放后的数据最大值小于等于 1.0
    assert X_scaled.max() <= 1.0
    # 断言 scaler 记录的样本数量与数据集 X 的行数相同
    assert scaler.n_samples_seen_ == X.shape[0]

    # 函数接口
    X_1d = X_1row.ravel()
    min_ = X_1d.min()
    max_ = X_1d.max()
    # 断言按最小-最大缩放公式计算的结果与手动调用 minmax_scale 函数的结果接近
    assert_array_almost_equal(
        (X_1d - min_) / (max_ - min_), minmax_scale(X_1d, copy=True)
    )


@pytest.mark.parametrize("sample_weight", [True, None])
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_without_centering(sample_weight, sparse_container):
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # 第一个特征始终为零
    X_sparse = sparse_container(X)

    if sample_weight:
        sample_weight = rng.rand(X.shape[0])

    with pytest.raises(ValueError):
        StandardScaler().fit(X_sparse)

    # 创建 StandardScaler 对象，关闭均值中心化，拟合数据集 X
    scaler = StandardScaler(with_mean=False).fit(X, sample_weight=sample_weight)
    # 对数据集 X 进行转换，保留副本
    X_scaled = scaler.transform(X, copy=True)
    # 断言转换后的数据中没有 NaN 值
    assert not np.any(np.isnan(X_scaled))

    # 创建 StandardScaler 对象，关闭均值中心化，拟合稀疏数据 X_sparse
    scaler_sparse = StandardScaler(with_mean=False).fit(
        X_sparse, sample_weight=sample_weight
    )
    # 对稀疏数据 X_sparse 进行转换，保留副本
    X_sparse_scaled = scaler_sparse.transform(X_sparse, copy=True)
    # 断言转换后的稀疏数据中没有 NaN 值
    assert not np.any(np.isnan(X_sparse_scaled.data))

    # 断言 scaler 和 scaler_sparse 的均值、方差、缩放系数和样本数量等属性近似相等
    assert_array_almost_equal(scaler.mean_, scaler_sparse.mean_)
    assert_array_almost_equal(scaler.var_, scaler_sparse.var_)
    assert_array_almost_equal(scaler.scale_, scaler_sparse.scale_)
    assert_array_almost_equal(scaler.n_samples_seen_, scaler_sparse.n_samples_seen_)

    if sample_weight is None:
        # 断言按轴 0 计算的缩放后数据的平均值接近于指定值
        assert_array_almost_equal(
            X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2
        )
        # 断言按轴 0 计算的缩放后数据的标准差为 1.0
        assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])

    # 计算稀疏数据 X_sparse_scaled 按轴 0 的均值和方差
    X_sparse_scaled_mean, X_sparse_scaled_var = mean_variance_axis(X_sparse_scaled, 0)
    # 使用断言检查稀疏矩阵经过缩放和均值计算后的近似值是否与原始矩阵的均值轴相等
    assert_array_almost_equal(X_sparse_scaled_mean, X_scaled.mean(axis=0))
    # 使用断言检查稀疏矩阵经过缩放和方差计算后的近似值是否与原始矩阵的方差轴相等

    assert_array_almost_equal(X_sparse_scaled_var, X_scaled.var(axis=0))

    # 检查 X 是否未被修改（即是否为副本）
    assert X_scaled is not X
    # 检查 X_sparse 是否未被修改（即是否为副本）
    assert X_sparse_scaled is not X_sparse

    # 使用缩放器的逆变换将 X_scaled 还原为原始数据 X，并使用断言检查其是否不是 X 的引用
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    # 再次检查 X_scaled_back 是否不是 X_scaled 的引用
    assert X_scaled_back is not X_scaled
    # 使用断言检查 X_scaled_back 是否与原始数据 X 近似相等
    assert_array_almost_equal(X_scaled_back, X)

    # 使用稀疏矩阵缩放器的逆变换将 X_sparse_scaled 还原为原始数据 X，并使用断言检查其是否不是 X_sparse 的引用
    X_sparse_scaled_back = scaler_sparse.inverse_transform(X_sparse_scaled)
    assert X_sparse_scaled_back is not X_sparse
    # 再次检查 X_sparse_scaled_back 是否不是 X_sparse_scaled 的引用
    assert X_sparse_scaled_back is not X_sparse_scaled
    # 使用断言检查稀疏矩阵的逆变换后的数据是否与原始数据 X 的密集表示相等
    assert_array_almost_equal(X_sparse_scaled_back.toarray(), X)

    # 如果稀疏容器属于 CSR_CONTAINERS，执行以下操作
    if sparse_container in CSR_CONTAINERS:
        # 创建一个不做任何转换的标准缩放器实例
        null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
        # 对稀疏矩阵 X_sparse 进行拟合和转换，得到 X_null
        X_null = null_transform.fit_transform(X_sparse)
        # 使用断言检查 X_null 的数据部分是否与 X_sparse 的数据部分相等
        assert_array_equal(X_null.data, X_sparse.data)
        # 使用逆变换将 X_null 还原为 X_orig
        X_orig = null_transform.inverse_transform(X_null)
        # 使用断言检查 X_orig 的数据部分是否与 X_sparse 的数据部分相等
        assert_array_equal(X_orig.data, X_sparse.data)
# 使用 pytest 的参数化功能，为 test_scaler_n_samples_seen_with_nan 函数生成多个测试用例
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_n_samples_seen_with_nan(with_mean, with_std, sparse_container):
    # 创建一个包含 NaN 值的 numpy 数组 X
    X = np.array(
        [[0, 1, 3], [np.nan, 6, 10], [5, 4, np.nan], [8, 0, np.nan]], dtype=np.float64
    )
    # 如果 sparse_container 不为 None，则将 X 转换为稀疏格式
    if sparse_container is not None:
        X = sparse_container(X)

    # 如果 X 是稀疏矩阵并且 with_mean 为 True，则跳过测试并提示异常信息
    if sparse.issparse(X) and with_mean:
        pytest.skip("'with_mean=True' cannot be used with sparse matrix.")

    # 创建 StandardScaler 转换器实例，设置 with_mean 和 with_std 参数
    transformer = StandardScaler(with_mean=with_mean, with_std=with_std)
    # 对数据 X 进行拟合
    transformer.fit(X)

    # 断言 transformer 的 n_samples_seen_ 属性与预期结果一致
    assert_array_equal(transformer.n_samples_seen_, np.array([3, 4, 2]))


# 检查两个标准化转换器实例的属性是否相等
def _check_identity_scalers_attributes(scaler_1, scaler_2):
    assert scaler_1.mean_ is scaler_2.mean_ is None
    assert scaler_1.var_ is scaler_2.var_ is None
    assert scaler_1.scale_ is scaler_2.scale_ is None
    assert scaler_1.n_samples_seen_ == scaler_2.n_samples_seen_


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_return_identity(sparse_container):
    # 测试当 with_mean 和 with_std 均为 False 时，标准化转换器返回恒等变换
    X_dense = np.array([[0, 1, 3], [5, 6, 0], [8, 0, 10]], dtype=np.float64)
    X_sparse = sparse_container(X_dense)

    # 创建 with_mean 和 with_std 均为 False 的标准化转换器实例，对稠密数据进行拟合和转换
    transformer_dense = StandardScaler(with_mean=False, with_std=False)
    X_trans_dense = transformer_dense.fit_transform(X_dense)
    assert_allclose(X_trans_dense, X_dense)

    # 克隆 transformer_dense 并对稀疏数据进行拟合和转换，检查转换后结果的一致性
    transformer_sparse = clone(transformer_dense)
    X_trans_sparse = transformer_sparse.fit_transform(X_sparse)
    assert_allclose_dense_sparse(X_trans_sparse, X_sparse)

    # 检查两个转换器的属性是否相等
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)

    # 使用 partial_fit 方法对稠密和稀疏数据分别进行部分拟合，并检查转换器的属性是否保持一致
    transformer_dense.partial_fit(X_dense)
    transformer_sparse.partial_fit(X_sparse)
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)

    # 使用 fit 方法对稠密和稀疏数据重新拟合，并再次检查转换器的属性是否保持一致
    transformer_dense.fit(X_dense)
    transformer_sparse.fit(X_sparse)
    _check_identity_scalers_attributes(transformer_dense, transformer_sparse)


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_int(sparse_container):
    # 测试标准化转换器能否将整数输入转换为浮点数，包括稀疏和稠密矩阵
    rng = np.random.RandomState(42)
    X = rng.randint(20, size=(4, 5))
    X[:, 0] = 0  # 将第一列特征设为零

    X_sparse = sparse_container(X)

    # 使用标准化转换器对稠密数据拟合和转换，检查转换后数据是否包含 NaN 值
    with warnings.catch_warnings(record=True):
        scaler = StandardScaler(with_mean=False).fit(X)
        X_scaled = scaler.transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))

    # 使用标准化转换器对稀疏数据拟合和转换，检查转换后数据是否包含 NaN 值
    with warnings.catch_warnings(record=True):
        scaler_sparse = StandardScaler(with_mean=False).fit(X_sparse)
        X_sparse_scaled = scaler_sparse.transform(X_sparse, copy=True)
    assert not np.any(np.isnan(X_sparse_scaled.data))

    # 断言 scaler 和 scaler_sparse 的均值相等
    assert_array_almost_equal(scaler.mean_, scaler_sparse.mean_)
    # 检查两个数组的元素是否近似相等
    assert_array_almost_equal(scaler.var_, scaler_sparse.var_)
    assert_array_almost_equal(scaler.scale_, scaler_sparse.scale_)

    # 检查经过缩放的特征矩阵的均值是否近似等于指定值
    assert_array_almost_equal(
        X_scaled.mean(axis=0), [0.0, 1.109, 1.856, 21.0, 1.559], 2
    )
    # 检查经过缩放的特征矩阵的标准差是否近似等于指定值
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])

    # 计算稀疏矩阵经过缩放后的均值和标准差
    X_sparse_scaled_mean, X_sparse_scaled_std = mean_variance_axis(
        X_sparse_scaled.astype(float), 0
    )
    # 检查稀疏矩阵经过缩放后的均值是否与原始特征矩阵的均值近似相等
    assert_array_almost_equal(X_sparse_scaled_mean, X_scaled.mean(axis=0))
    # 检查稀疏矩阵经过缩放后的标准差是否与原始特征矩阵的标准差近似相等
    assert_array_almost_equal(X_sparse_scaled_std, X_scaled.std(axis=0))

    # 检查原始特征矩阵 X 是否被修改（是否为深拷贝）
    assert X_scaled is not X
    assert X_sparse_scaled is not X_sparse

    # 使用逆变换恢复经过缩放的特征矩阵 X_scaled
    X_scaled_back = scaler.inverse_transform(X_scaled)
    # 检查恢复的特征矩阵是否不是原始特征矩阵 X 的引用
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    # 检查恢复的特征矩阵是否近似等于原始特征矩阵 X
    assert_array_almost_equal(X_scaled_back, X)

    # 使用逆变换恢复经过缩放的稀疏矩阵 X_sparse_scaled
    X_sparse_scaled_back = scaler_sparse.inverse_transform(X_sparse_scaled)
    # 检查恢复的稀疏矩阵是否不是原始稀疏矩阵 X_sparse 的引用
    assert X_sparse_scaled_back is not X_sparse
    assert X_sparse_scaled_back is not X_sparse_scaled
    # 检查恢复的稀疏矩阵（转换为密集数组后）是否近似等于原始特征矩阵 X
    assert_array_almost_equal(X_sparse_scaled_back.toarray(), X)

    # 如果稀疏容器为 CSR 类型，则进行以下检查
    if sparse_container in CSR_CONTAINERS:
        # 创建一个空的标准缩放器对象，禁用均值和标准差的调整，进行深拷贝
        null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
        # 忽略警告，执行稀疏矩阵 X_sparse 的标准缩放并获取结果 X_null
        with warnings.catch_warnings(record=True):
            X_null = null_transform.fit_transform(X_sparse)
        # 检查处理后稀疏矩阵 X_null 的数据部分是否与原始稀疏矩阵 X_sparse 的数据部分相等
        assert_array_equal(X_null.data, X_sparse.data)
        # 使用逆变换恢复处理后的稀疏矩阵 X_null 并获取结果 X_orig
        X_orig = null_transform.inverse_transform(X_null)
        # 检查逆变换后稀疏矩阵 X_orig 的数据部分是否与原始稀疏矩阵 X_sparse 的数据部分相等
        assert_array_equal(X_orig.data, X_sparse.data)
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_scaler_without_copy(sparse_container):
    # 使用参数化测试，循环稀疏容器类型进行测试

    # 设定随机数种子为42
    rng = np.random.RandomState(42)
    # 创建一个4x5的随机数组，其中第一列被设为0
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # 第一个特征总是0
    # 将稀疏数据容器应用于X数组
    X_sparse = sparse_container(X)

    # 复制X数组
    X_copy = X.copy()
    # 使用copy=False创建StandardScaler对象，并对X进行拟合
    StandardScaler(copy=False).fit(X)
    # 断言X与X_copy相等，即fit操作没有改变X的值
    assert_array_equal(X, X_copy)

    # 复制稀疏矩阵X_sparse
    X_sparse_copy = X_sparse.copy()
    # 使用with_mean=False和copy=False创建StandardScaler对象，并对X_sparse进行拟合
    StandardScaler(with_mean=False, copy=False).fit(X_sparse)
    # 断言X_sparse的数组表示与X_sparse_copy的数组表示相等，即fit操作没有改变X_sparse的值
    assert_array_equal(X_sparse.toarray(), X_sparse_copy.toarray())


@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_scale_sparse_with_mean_raise_exception(sparse_container):
    # 使用参数化测试，循环稀疏容器类型进行测试
    rng = np.random.RandomState(42)
    # 创建一个4x5的随机数组
    X = rng.randn(4, 5)
    # 将稀疏数据容器应用于X数组
    X_sparse = sparse_container(X)

    # 检查对稀疏数据的缩放和拟合直接调用
    with pytest.raises(ValueError):
        scale(X_sparse, with_mean=True)
    with pytest.raises(ValueError):
        StandardScaler(with_mean=True).fit(X_sparse)

    # 在密集数组上拟合后，检查transform和inverse_transform
    scaler = StandardScaler(with_mean=True).fit(X)
    with pytest.raises(ValueError):
        scaler.transform(X_sparse)

    # 将scaler.transform(X)后的结果转换为稀疏数据并断言引发值错误
    X_transformed_sparse = sparse_container(scaler.transform(X))
    with pytest.raises(ValueError):
        scaler.inverse_transform(X_transformed_sparse)


def test_scale_input_finiteness_validation():
    # 检查非有限输入是否引发值错误
    X = [[np.inf, 5, 6, 7, 8]]
    with pytest.raises(
        ValueError, match="Input contains infinity or a value too large"
    ):
        scale(X)


def test_robust_scaler_error_sparse():
    # 创建一个稀疏矩阵X_sparse，形状为1000x10
    X_sparse = sparse.rand(1000, 10)
    # 创建RobustScaler对象，指定with_centering=True
    scaler = RobustScaler(with_centering=True)
    # 设置错误消息
    err_msg = "Cannot center sparse matrices"
    # 断言使用RobustScaler拟合稀疏矩阵X_sparse会引发值错误，并且错误消息匹配err_msg
    with pytest.raises(ValueError, match=err_msg):
        scaler.fit(X_sparse)


@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("X", [np.random.randn(10, 3), sparse.rand(10, 3, density=0.5)])
def test_robust_scaler_attributes(X, with_centering, with_scaling):
    # 检查属性的类型是否一致
    if with_centering and sparse.issparse(X):
        pytest.skip("RobustScaler cannot center sparse matrix")

    # 创建RobustScaler对象，根据参数with_centering和with_scaling设置
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    # 对X进行拟合
    scaler.fit(X)

    # 如果with_centering为True，则断言scaler.center_为np.ndarray类型；否则为None
    if with_centering:
        assert isinstance(scaler.center_, np.ndarray)
    else:
        assert scaler.center_ is None
    # 如果with_scaling为True，则断言scaler.scale_为np.ndarray类型；否则为None
    if with_scaling:
        assert isinstance(scaler.scale_, np.ndarray)
    else:
        assert scaler.scale_ is None


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_robust_scaler_col_zero_sparse(csr_container):
    # 检查稀疏矩阵列中没有数据时，标量是否正常工作
    X = np.random.randn(10, 5)
    X[:, 0] = 0
    # 将X数组转换为CSR格式的稀疏矩阵
    X = csr_container(X)
    # 创建一个 RobustScaler 对象，用于对数据进行鲁棒缩放，不进行中心化处理
    scaler = RobustScaler(with_centering=False)
    # 使用给定的数据 X 对 scaler 进行拟合，计算相关的缩放参数
    scaler.fit(X)
    # 使用断言确保第一个特征的缩放参数接近于 1，使用 pytest 的 approx 函数进行近似比较
    assert scaler.scale_[0] == pytest.approx(1)
    
    # 使用已拟合好的 scaler 对数据 X 进行转换
    X_trans = scaler.transform(X)
    # 使用断言确保转换后的第一个特征列（以稀疏矩阵形式表示）与原始数据第一个特征列的值非常接近
    assert_allclose(X[:, [0]].toarray(), X_trans[:, [0]].toarray())
# 测试 RobustScaler 对 2 维数组的稳健缩放
def test_robust_scaler_2d_arrays():
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 生成一个大小为 4x5 的随机正态分布数组 X
    X = rng.randn(4, 5)
    # 将 X 的第一列设为 0.0，即第一个特征总是为零
    X[:, 0] = 0.0

    # 创建 RobustScaler 实例
    scaler = RobustScaler()
    # 对 X 进行拟合并进行转换
    X_scaled = scaler.fit(X).transform(X)

    # 断言 X_scaled 每列的中位数近似为 0.0
    assert_array_almost_equal(np.median(X_scaled, axis=0), 5 * [0.0])
    # 断言 X_scaled 第一列的标准差近似为 0
    assert_array_almost_equal(X_scaled.std(axis=0)[0], 0)


# 使用参数化测试装饰器，测试 RobustScaler 对密集和稀疏矩阵的等效性
@pytest.mark.parametrize("density", [0, 0.05, 0.1, 0.5, 1])
@pytest.mark.parametrize("strictly_signed", ["positive", "negative", "zeros", None])
def test_robust_scaler_equivalence_dense_sparse(density, strictly_signed):
    # 创建一个稀疏矩阵 X_sparse，密度为 density
    X_sparse = sparse.rand(1000, 5, density=density).tocsc()
    # 根据 strictly_signed 的值调整 X_sparse 的数据
    if strictly_signed == "positive":
        X_sparse.data = np.abs(X_sparse.data)
    elif strictly_signed == "negative":
        X_sparse.data = -np.abs(X_sparse.data)
    elif strictly_signed == "zeros":
        X_sparse.data = np.zeros(X_sparse.data.shape, dtype=np.float64)
    # 将 X_sparse 转换为密集矩阵 X_dense
    X_dense = X_sparse.toarray()

    # 创建两个 RobustScaler 实例，都不进行中心化
    scaler_sparse = RobustScaler(with_centering=False)
    scaler_dense = RobustScaler(with_centering=False)

    # 对稀疏矩阵 X_sparse 和密集矩阵 X_dense 进行拟合
    scaler_sparse.fit(X_sparse)
    scaler_dense.fit(X_dense)

    # 断言两个 scaler 的 scale_ 属性近似相等
    assert_allclose(scaler_sparse.scale_, scaler_dense.scale_)


# 使用参数化测试装饰器，测试 RobustScaler 对 csr 格式包装器进行转换的行为
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_robust_scaler_transform_one_row_csr(csr_container):
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 生成一个大小为 4x5 的随机正态分布数组 X
    X = rng.randn(4, 5)
    # 创建一个包含单行数据的数组 single_row
    single_row = np.array([[0.1, 1.0, 2.0, 0.0, -1.0]])
    # 创建 RobustScaler 实例，不进行中心化
    scaler = RobustScaler(with_centering=False)
    # 对 X 进行拟合
    scaler = scaler.fit(X)
    # 对 csr 格式的单行数据进行转换
    row_trans = scaler.transform(csr_container(single_row))
    # 计算预期的转换后的单行数据
    row_expected = single_row / scaler.scale_
    # 断言转换后的数据与预期的单行数据近似相等
    assert_array_almost_equal(row_trans.toarray(), row_expected)
    # 对转换后的数据再进行逆转换
    row_scaled_back = scaler.inverse_transform(row_trans)
    # 断言逆转换后的数据与原始的单行数据近似相等
    assert_array_almost_equal(single_row, row_scaled_back.toarray())


# 测试 RobustScaler 在鸢尾花数据集上的表现
def test_robust_scaler_iris():
    # 载入鸢尾花数据集的特征部分 X
    X = iris.data
    # 创建 RobustScaler 实例
    scaler = RobustScaler()
    # 对 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)
    # 断言转换后的 X 每列的中位数近似为 0
    assert_array_almost_equal(np.median(X_trans, axis=0), 0)
    # 对转换后的 X 再进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言逆转换后的数据与原始数据 X 近似相等
    assert_array_almost_equal(X, X_trans_inv)
    # 计算转换后的 X 每列的 25% 和 75% 分位数
    q = np.percentile(X_trans, q=(25, 75), axis=0)
    # 计算 IQR (Interquartile Range)
    iqr = q[1] - q[0]
    # 断言 IQR 近似为 1
    assert_array_almost_equal(iqr, 1)


# 测试 RobustScaler 在鸢尾花数据集上，并设置分位数范围
def test_robust_scaler_iris_quantiles():
    # 载入鸢尾花数据集的特征部分 X
    X = iris.data
    # 创建 RobustScaler 实例，设置分位数范围为 (10, 90)
    scaler = RobustScaler(quantile_range=(10, 90))
    # 对 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)
    # 断言转换后的 X 每列的中位数近似为 0
    assert_array_almost_equal(np.median(X_trans, axis=0), 0)
    # 对转换后的 X 再进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言逆转换后的数据与原始数据 X 近似相等
    assert_array_almost_equal(X, X_trans_inv)
    # 计算转换后的 X 每列的 10% 和 90% 分位数
    q = np.percentile(X_trans, q=(10, 90), axis=0)
    # 计算分位数范围
    q_range = q[1] - q[0]
    # 断言分位数范围近似为 1
    assert_array_almost_equal(q_range, 1)


# 使用参数化测试装饰器，测试 QuantileTransformer 在鸢尾花数据集上的行为
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_iris(csc_container):
    # 载入鸢尾花数据集的特征部分 X
    X = iris.data
    # uniform output distribution
    # 创建一个分位数转换器对象，设定分位数数量为30
    transformer = QuantileTransformer(n_quantiles=30)
    
    # 使用分位数转换器对数据集 X 进行拟合和转换
    X_trans = transformer.fit_transform(X)
    
    # 对转换后的数据再次进行逆转换，得到原始数据的近似值
    X_trans_inv = transformer.inverse_transform(X_trans)
    
    # 断言转换前后的数据应该几乎相等
    assert_array_almost_equal(X, X_trans_inv)
    
    # 创建一个分位数转换器对象，设定分位数数量为30，并设定输出分布为正态分布
    transformer = QuantileTransformer(n_quantiles=30, output_distribution="normal")
    
    # 使用分位数转换器对数据集 X 进行拟合和转换
    X_trans = transformer.fit_transform(X)
    
    # 对转换后的数据再次进行逆转换，得到原始数据的近似值
    X_trans_inv = transformer.inverse_transform(X_trans)
    
    # 断言转换前后的数据应该几乎相等
    assert_array_almost_equal(X, X_trans_inv)
    
    # 确保能对稀疏矩阵进行逆转换，这在鸢尾花数据集中包含负值时尤为重要
    # 将密集矩阵 X 转换为稀疏矩阵 X_sparse
    X_sparse = csc_container(X)
    
    # 使用分位数转换器对稀疏矩阵 X_sparse 进行拟合和转换
    X_sparse_tran = transformer.fit_transform(X_sparse)
    
    # 对转换后的稀疏矩阵数据再次进行逆转换，得到原始数据的近似值
    X_sparse_tran_inv = transformer.inverse_transform(X_sparse_tran)
    
    # 断言转换前后的稀疏矩阵数据应该几乎相等，需要将稀疏矩阵转换为密集矩阵进行比较
    assert_array_almost_equal(X_sparse.toarray(), X_sparse_tran_inv.toarray())
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数，每个参数会作为单独的测试运行
def test_quantile_transform_check_error(csc_container):
    X = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    # 转置输入矩阵 X
    X = csc_container(X)
    # 使用 csc_container 转换 X 成稀疏格式

    X_neg = np.transpose(
        [
            [0, 25, 50, 0, 0, 0, 75, 0, 0, 100],
            [-2, 4, 0, 0, 6, 8, 0, 10, 0, 0],
            [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1],
        ]
    )
    # 转置输入矩阵 X_neg
    X_neg = csc_container(X_neg)
    # 使用 csc_container 转换 X_neg 成稀疏格式

    err_msg = (
        "The number of quantiles cannot be greater than "
        "the number of samples used. Got 1000 quantiles "
        "and 10 samples."
    )
    # 定义错误消息字符串

    with pytest.raises(ValueError, match=err_msg):
        # 断言会抛出 ValueError 异常，并匹配指定的错误消息
        QuantileTransformer(subsample=10).fit(X)

    transformer = QuantileTransformer(n_quantiles=10)
    # 初始化 QuantileTransformer 对象，设置 n_quantiles=10

    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    # 更新错误消息字符串

    with pytest.raises(ValueError, match=err_msg):
        # 断言会抛出 ValueError 异常，并匹配指定的错误消息
        transformer.fit(X_neg)

    transformer.fit(X)
    # 对 X 进行拟合

    err_msg = "QuantileTransformer only accepts non-negative sparse matrices."
    # 更新错误消息字符串

    with pytest.raises(ValueError, match=err_msg):
        # 断言会抛出 ValueError 异常，并匹配指定的错误消息
        transformer.transform(X_neg)

    X_bad_feat = np.transpose(
        [[0, 25, 50, 0, 0, 0, 75, 0, 0, 100], [0, 0, 2.6, 4.1, 0, 0, 2.3, 0, 9.5, 0.1]]
    )
    # 转置输入矩阵 X_bad_feat，其中包含 2 个特征

    err_msg = (
        "X has 2 features, but QuantileTransformer is expecting 3 features as input."
    )
    # 定义错误消息字符串

    with pytest.raises(ValueError, match=err_msg):
        # 断言会抛出 ValueError 异常，并匹配指定的错误消息
        transformer.inverse_transform(X_bad_feat)

    transformer = QuantileTransformer(n_quantiles=10).fit(X)
    # 初始化 QuantileTransformer 对象，设置 n_quantiles=10，并对 X 进行拟合

    # 检查输入为标量时是否引发错误
    with pytest.raises(ValueError, match="Expected 2D array, got scalar array instead"):
        transformer.transform(10)

    transformer = QuantileTransformer(n_quantiles=100)
    # 初始化 QuantileTransformer 对象，设置 n_quantiles=100

    warn_msg = "n_quantiles is set to n_samples"
    # 定义警告消息字符串

    with pytest.warns(UserWarning, match=warn_msg) as record:
        # 断言会产生 UserWarning 警告，并匹配指定的警告消息
        transformer.fit(X)
    assert len(record) == 1
    assert transformer.n_quantiles_ == X.shape[0]
    # 断言 transformer 的 n_quantiles_ 属性与 X 的样本数相等
    # 创建稀疏矩阵的数据部分，包含非零值
    X_data = np.array([0, 0, 1, 0, 2, 2, 1, 0, 1, 2, 0])
    # 创建稀疏矩阵的列索引
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # 创建稀疏矩阵的行偏移索引
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    # 使用稀疏矩阵容器创建稀疏矩阵
    X_sparse = csc_container((X_data, (X_row, X_col)))
    # 使用转换器对象对稀疏矩阵进行转换
    X_trans = transformer.fit_transform(X_sparse)
    # 预期的稀疏矩阵转换结果
    X_expected = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    # 断言转换后的稀疏矩阵与预期结果的近似相等性
    assert_almost_equal(X_expected, X_trans.toarray())

    # 创建新的量化转换器对象，忽略隐式零值，设置量化分位数为5
    transformer = QuantileTransformer(ignore_implicit_zeros=True, n_quantiles=5)
    # 创建新的稀疏矩阵的数据部分，包含负值
    X_data = np.array([-1, -1, 1, 0, 0, 0, 1, -1, 1])
    # 创建新的稀疏矩阵的列索引
    X_col = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    # 创建新的稀疏矩阵的行偏移索引
    X_row = np.array([0, 4, 0, 1, 2, 3, 4, 5, 6])
    # 使用新的稀疏矩阵容器创建稀疏矩阵
    X_sparse = csc_container((X_data, (X_row, X_col)))
    # 使用量化转换器对新的稀疏矩阵进行转换
    X_trans = transformer.fit_transform(X_sparse)
    # 新的预期稀疏矩阵转换结果
    X_expected = np.array(
        [[0, 1], [0, 0.375], [0, 0.375], [0, 0.375], [0, 1], [0, 0], [0, 1]]
    )
    # 断言新的稀疏矩阵转换后与预期结果的近似相等性
    assert_almost_equal(X_expected, X_trans.toarray())
    # 断言逆转换后的稀疏矩阵与原始稀疏矩阵的近似相等性
    assert_almost_equal(
        X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray()
    )

    # 创建带有子采样参数的新的量化转换器对象
    transformer = QuantileTransformer(
        ignore_implicit_zeros=True, n_quantiles=5, subsample=8, random_state=0
    )
    # 使用新的量化转换器对象对原始稀疏矩阵进行转换
    X_trans = transformer.fit_transform(X_sparse)
    # 断言带有子采样的稀疏矩阵转换后与预期结果的近似相等性
    assert_almost_equal(X_expected, X_trans.toarray())
    # 断言带有子采样的逆转换后的稀疏矩阵与原始稀疏矩阵的近似相等性
    assert_almost_equal(
        X_sparse.toarray(), transformer.inverse_transform(X_trans).toarray()
    )
# 定义一个测试函数，用于测试 QuantileTransformer 在密集数据集上的转换行为
def test_quantile_transform_dense_toy():
    # 创建一个包含数值的 NumPy 数组 X，每行代表一个样本，每列代表一个特征
    X = np.array(
        [[0, 2, 2.6], [25, 4, 4.1], [50, 6, 2.3], [75, 8, 9.5], [100, 10, 0.1]]
    )

    # 初始化 QuantileTransformer 对象，指定分位数的数量为 5
    transformer = QuantileTransformer(n_quantiles=5)
    # 对 X 进行拟合，使得 transformer 能够对 X 进行变换
    transformer.fit(X)

    # 使用均匀输出，期望每个 X 的条目被映射到 0 到 1 之间，并且等间距
    X_trans = transformer.fit_transform(X)
    # 生成期望的结果 X_expected，将 0 到 1 等分成 5 份，并在行上复制三次，然后转置
    X_expected = np.tile(np.linspace(0, 1, num=5), (3, 1)).T
    # 断言排序后的 X_trans 等于 X_expected
    assert_almost_equal(np.sort(X_trans, axis=0), X_expected)

    # 定义测试集 X_test，包含两个样本
    X_test = np.array(
        [
            [-1, 1, 0],
            [101, 11, 10],
        ]
    )
    # 定义期望的转换结果 X_expected
    X_expected = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    # 断言 transformer 对 X_test 的转换结果等于 X_expected
    assert_array_almost_equal(transformer.transform(X_test), X_expected)

    # 对 X_trans 进行逆变换，期望结果等于原始的 X
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)


# 定义测试函数，测试 QuantileTransformer 在子采样输入时的行为
def test_quantile_transform_subsampling():
    # 测试当对输入进行子采样时，确保得到一致的结果。检查计算的分位数几乎被映射到 [0, 1] 向量，其中值是等间距的。
    # 使用无穷范数检查其是否小于给定的阈值。此过程重复 5 次。

    # 密集支持
    n_samples = 1000000
    n_quantiles = 1000
    # 创建一个排序后的随机样本 X，形状为 (n_samples, 1)
    X = np.sort(np.random.sample((n_samples, 1)), axis=0)
    ROUND = 5
    inf_norm_arr = []
    for random_state in range(ROUND):
        # 初始化 QuantileTransformer 对象，设置随机种子和子采样参数
        transformer = QuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        # 对 X 进行拟合
        transformer.fit(X)
        # 计算预期分位数与实际分位数之间的差异，并计算其无穷范数
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        # 断言无穷范数小于 0.01
        assert inf_norm < 1e-2
        inf_norm_arr.append(inf_norm)
    # 每个随机子采样产生一个唯一的近似预期的线性空间 CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)

    # 稀疏支持

    # 创建一个稀疏矩阵 X，密度为 0.99，形式为 "csc"
    X = sparse.rand(n_samples, 1, density=0.99, format="csc", random_state=0)
    inf_norm_arr = []
    for random_state in range(ROUND):
        # 初始化 QuantileTransformer 对象，设置随机种子和子采样参数
        transformer = QuantileTransformer(
            random_state=random_state,
            n_quantiles=n_quantiles,
            subsample=n_samples // 10,
        )
        # 对 X 进行拟合
        transformer.fit(X)
        # 计算预期分位数与实际分位数之间的差异，并计算其无穷范数
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        # 断言无穷范数小于 0.1
        assert inf_norm < 1e-1
        inf_norm_arr.append(inf_norm)
    # 每个随机子采样产生一个唯一的近似预期的线性空间 CDF
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)


# 定义测试函数，测试当 `subsample=None` 时 `QuantileTransformer` 的行为
def test_quantile_transform_subsampling_disabled():
    """检查 `QuantileTransformer` 在 `subsample=None` 时的行为。"""
    # 创建一个随机状态为 0 的正态分布样本 X，形状为 (200, 1)
    X = np.random.RandomState(0).normal(size=(200, 1))

    n_quantiles = 5
    # 初始化 QuantileTransformer 对象，设置分位数数量和子采样参数为 None，并对 X 进行拟合
    transformer = QuantileTransformer(n_quantiles=n_quantiles, subsample=None).fit(X)

    # 期望的参考值是等间距的 0 到 1
    expected_references = np.linspace(0, 1, n_quantiles)
    # 使用 assert_allclose 函数检查 transformer.references_ 是否与 expected_references 接近
    assert_allclose(transformer.references_, expected_references)
    
    # 计算数组 X 展开后的分位数，存储在 expected_quantiles 中
    expected_quantiles = np.quantile(X.ravel(), expected_references)
    
    # 使用 assert_allclose 函数检查 transformer.quantiles_.ravel() 是否与 expected_quantiles 接近
    assert_allclose(transformer.quantiles_.ravel(), expected_quantiles)
# 使用参数化测试，循环遍历 CSC_CONTAINERS 中的每个容器
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_sparse_toy(csc_container):
    # 创建一个三列十行的 numpy 数组 X，用于测试
    X = np.array(
        [
            [0.0, 2.0, 0.0],
            [25.0, 4.0, 0.0],
            [50.0, 0.0, 2.6],
            [0.0, 0.0, 4.1],
            [0.0, 6.0, 0.0],
            [0.0, 8.0, 0.0],
            [75.0, 0.0, 2.3],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 9.5],
            [100.0, 0.0, 0.1],
        ]
    )

    # 使用当前的 csc_container 转换 X 为稀疏格式
    X = csc_container(X)

    # 创建一个 QuantileTransformer 对象，指定分位数为 10，并对 X 进行拟合
    transformer = QuantileTransformer(n_quantiles=10)
    transformer.fit(X)

    # 对 X 进行转换，并断言转换后的最小值接近 0，最大值接近 1
    X_trans = transformer.fit_transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    # 对 X_trans 进行逆转换，并断言逆转换后的结果与原始 X 相近
    X_trans_inv = transformer.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())

    # 使用稠密格式的 QuantileTransformer 对象对 X 进行拟合和转换
    transformer_dense = QuantileTransformer(n_quantiles=10).fit(X.toarray())

    # 再次对 X 进行转换，并断言转换后的最小值接近 0，最大值接近 1
    X_trans = transformer_dense.transform(X)
    assert_array_almost_equal(np.min(X_trans.toarray(), axis=0), 0.0)
    assert_array_almost_equal(np.max(X_trans.toarray(), axis=0), 1.0)

    # 再次对 X_trans 进行逆转换，并断言逆转换后的结果与原始 X 相近
    X_trans_inv = transformer_dense.inverse_transform(X_trans)
    assert_array_almost_equal(X.toarray(), X_trans_inv.toarray())


# 测试 QuantileTransformer 在 axis=1 上的转换
def test_quantile_transform_axis1():
    # 创建一个 3x5 的 numpy 数组 X
    X = np.array([[0, 25, 50, 75, 100], [2, 4, 6, 8, 10], [2.6, 4.1, 2.3, 9.5, 0.1]])

    # 对 X 进行 axis=0 和 axis=1 方向上的 Quantile 转换，并断言转换后的结果相等
    X_trans_a0 = quantile_transform(X.T, axis=0, n_quantiles=5)
    X_trans_a1 = quantile_transform(X, axis=1, n_quantiles=5)
    assert_array_almost_equal(X_trans_a0, X_trans_a1.T)


# 使用参数化测试，测试 QuantileTransformer 对边界情况的处理
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_quantile_transform_bounds(csc_container):
    # 创建一个稠密的 3x2 numpy 数组 X_dense
    X_dense = np.array([[0, 0], [0, 0], [1, 0]])
    # 将 X_dense 转换为稀疏格式 X_sparse
    X_sparse = csc_container(X_dense)

    # 使用 QuantileTransformer 对 X_dense 进行拟合和转换，断言转换后的稠密和稀疏结果一致
    X_trans = QuantileTransformer(n_quantiles=3, random_state=0).fit_transform(X_dense)
    assert_array_almost_equal(X_trans, X_dense)
    X_trans_sp = QuantileTransformer(n_quantiles=3, random_state=0).fit_transform(
        X_sparse
    )
    assert_array_almost_equal(X_trans_sp.toarray(), X_dense)
    assert_array_almost_equal(X_trans, X_trans_sp.toarray())

    # 创建两个 3x2 的 numpy 数组 X 和 X1，分别用于拟合和转换
    X = np.array([[0, 1], [0, 0.5], [1, 0]])
    X1 = np.array([[0, 0.1], [0, 0.5], [1, 0.1]])

    # 使用 QuantileTransformer 对 X 进行拟合，再对 X1 进行转换，并断言转换结果相等
    transformer = QuantileTransformer(n_quantiles=3).fit(X)
    X_trans = transformer.transform(X1)
    assert_array_almost_equal(X_trans, X1)

    # 创建一个包含 1000 个随机数的 numpy 数组 X，并使用 QuantileTransformer 对其进行拟合
    X = np.random.random((1000, 1))
    transformer = QuantileTransformer()
    transformer.fit(X)

    # 断言在学习范围之外的值经过转换后仍保持一致
    assert transformer.transform([[-10]]) == transformer.transform([[np.min(X)]])
    # 断言：使用 transformer 对象进行转换，并验证两个输入值的转换结果相等
    assert transformer.transform([[10]]) == transformer.transform([[np.max(X)]])
    # 断言：使用 transformer 对象进行逆转换，并验证两个输入值的逆转换结果相等
    assert transformer.inverse_transform([[-10]]) == transformer.inverse_transform(
        [[np.min(transformer.references_)]]
    )
    # 断言：使用 transformer 对象进行逆转换，并验证两个输入值的逆转换结果相等
    assert transformer.inverse_transform([[10]]) == transformer.inverse_transform(
        [[np.max(transformer.references_)]]
    )
# 定义用于测试 QuantileTransformer 的函数
def test_quantile_transform_and_inverse():
    # 使用 iris 数据集的特征数据作为 X_1
    X_1 = iris.data
    # 创建一个包含特定值的新数组 X_2
    X_2 = np.array([[0.0], [BOUNDS_THRESHOLD / 10], [1.5], [2], [3], [3], [4]])
    # 对每个数据集 X 进行迭代
    for X in [X_1, X_2]:
        # 创建 QuantileTransformer 对象，指定参数 n_quantiles=1000 和 random_state=0
        transformer = QuantileTransformer(n_quantiles=1000, random_state=0)
        # 对数据集 X 进行拟合转换
        X_trans = transformer.fit_transform(X)
        # 对转换后的数据进行逆转换
        X_trans_inv = transformer.inverse_transform(X_trans)
        # 断言转换前后的数据应该几乎相等，精确到小数点后 9 位
        assert_array_almost_equal(X, X_trans_inv, decimal=9)


# 定义用于测试带有 NaN 值的 QuantileTransformer 的函数
def test_quantile_transform_nan():
    # 创建包含 NaN 值的特定数据集 X
    X = np.array([[np.nan, 0, 0, 1], [np.nan, np.nan, 0, 0.5], [np.nan, 1, 1, 0]])
    # 创建 QuantileTransformer 对象，指定参数 n_quantiles=10 和 random_state=42，对 X 进行拟合转换
    transformer = QuantileTransformer(n_quantiles=10, random_state=42)
    transformer.fit_transform(X)
    
    # 检查第一列的分位数是否全部为 NaN
    assert np.isnan(transformer.quantiles_[:, 0]).all()
    # 检查除第一列外的所有列是否不包含 NaN 值
    assert not np.isnan(transformer.quantiles_[:, 1:]).any()


# 使用参数化测试的方式，测试排序后的 QuantileTransformer
@pytest.mark.parametrize("array_type", ["array", "sparse"])
def test_quantile_transformer_sorted_quantiles(array_type):
    # 创建特定数据集 X
    X = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7] * 10)
    X = 0.1 * X.reshape(-1, 1)
    X = _convert_container(X, array_type)
    
    # 指定 n_quantiles=100 创建 QuantileTransformer 对象 qt，并对 X 进行拟合
    qt = QuantileTransformer(n_quantiles=100).fit(X)

    # 检查估计的分位数阈值是否单调递增
    quantiles = qt.quantiles_[:, 0]
    assert len(quantiles) == 100
    assert all(np.diff(quantiles) >= 0)


# 定义测试 RobustScaler 的非法范围输入的函数
def test_robust_scaler_invalid_range():
    # 遍历多个范围的输入
    for range_ in [
        (-1, 90),
        (-2, -3),
        (10, 101),
        (100.5, 101),
        (90, 50),
    ]:
        # 创建 RobustScaler 对象，使用指定的 quantile_range 进行初始化
        scaler = RobustScaler(quantile_range=range_)
        
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，异常消息应包含 "Invalid quantile range: ("
        with pytest.raises(ValueError, match=r"Invalid quantile range: \("):
            scaler.fit(iris.data)


# 使用参数化测试的方式，测试 scale 函数在不进行均值中心化时的表现
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_scale_function_without_centering(csr_container):
    # 创建随机数生成器 rng 和特定的数据集 X
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # 将第一列特征设置为零
    X_csr = csr_container(X)

    # 对 X 进行无均值中心化的缩放，得到 X_scaled
    X_scaled = scale(X, with_mean=False)
    assert not np.any(np.isnan(X_scaled))

    # 对 X_csr 进行无均值中心化的缩放，得到 X_csr_scaled
    X_csr_scaled = scale(X_csr, with_mean=False)
    assert not np.any(np.isnan(X_csr_scaled.data))

    # 检查使用 csc 格式的 X_csr_scaled 是否与 X_scaled 的结果几乎相等
    X_csc_scaled = scale(X_csr.tocsc(), with_mean=False)
    assert_array_almost_equal(X_scaled, X_csc_scaled.toarray())

    # 对 axis != 0 的情况引发 ValueError 异常
    with pytest.raises(ValueError):
        scale(X_csr, with_mean=False, axis=1)

    # 检查 X_scaled 的均值是否与预期值几乎相等，精确到小数点后 2 位
    assert_array_almost_equal(
        X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2
    )
    # 检查 X_scaled 的标准差是否都为 1
    assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    # 检查 X 是否未被复制
    assert X_scaled is not X

    # 获取经均值和方差缩放后的 X_csr_scaled 的均值和方差
    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis(X_csr_scaled, 0)
    # 对比稀疏矩阵经过缩放后的均值是否接近于原始矩阵按列计算的均值
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    # 对比稀疏矩阵经过缩放后的标准差是否接近于原始矩阵按列计算的标准差
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))
    
    # 对稀疏矩阵进行空缩放（不进行均值和标准差的缩放）
    X_csr_scaled = scale(X_csr, with_mean=False, with_std=False, copy=True)
    # 断言空缩放后的稀疏矩阵与原始稀疏矩阵在转换为密集数组后的数值几乎相等
    assert_array_almost_equal(X_csr.toarray(), X_csr_scaled.toarray())
def test_robust_scale_axis1():
    # 使用 iris 数据集中的全部数据
    X = iris.data
    # 对数据进行沿 axis=1 的 RobustScaler 转换
    X_trans = robust_scale(X, axis=1)
    # 断言：沿 axis=1 的中位数应该接近于 0
    assert_array_almost_equal(np.median(X_trans, axis=1), 0)
    # 计算沿 axis=1 的分位数 q=(25, 75)
    q = np.percentile(X_trans, q=(25, 75), axis=1)
    # 计算 IQR（四分位距）
    iqr = q[1] - q[0]
    # 断言：IQR 应该接近于 1
    assert_array_almost_equal(iqr, 1)


def test_robust_scale_1d_array():
    # 使用 iris 数据集中第二列作为一维数据
    X = iris.data[:, 1]
    # 对一维数据进行 RobustScaler 转换
    X_trans = robust_scale(X)
    # 断言：中位数应该接近于 0
    assert_array_almost_equal(np.median(X_trans), 0)
    # 计算数据的分位数 q=(25, 75)
    q = np.percentile(X_trans, q=(25, 75))
    # 计算 IQR（四分位距）
    iqr = q[1] - q[0]
    # 断言：IQR 应该接近于 1
    assert_array_almost_equal(iqr, 1)


def test_robust_scaler_zero_variance_features():
    # 检查 RobustScaler 在具有零方差特征的示例数据上的表现
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]]

    # 创建 RobustScaler 对象
    scaler = RobustScaler()
    # 对数据 X 进行拟合并转换
    X_trans = scaler.fit_transform(X)

    # 注意：对于这么小的样本大小，第三列的预期值非常依赖于用来计算分位数的方法。
    # 这里的值是使用 numpy 1.9 中 np.percentile 产生的分位数来计算的。
    # 使用 scipy.stats.mstats.scoreatquantile 或 scipy.stats.mstats.mquantiles
    # 计算分位数会得到非常不同的结果！

    # 预期的转换后的数据
    X_expected = [[0.0, 0.0, +0.0], [0.0, 0.0, -1.0], [0.0, 0.0, +1.0]]
    # 断言：转换后的数据应该与预期的数据 X_expected 接近
    assert_array_almost_equal(X_trans, X_expected)
    
    # 对转换后的数据进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言：逆转换后的数据应该与原始数据 X 接近
    assert_array_almost_equal(X, X_trans_inv)

    # 确保新数据能够正确转换
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    X_trans_new = scaler.transform(X_new)
    X_expected_new = [[+0.0, 1.0, +0.0], [-1.0, 0.0, -0.83333], [+0.0, 0.0, +1.66667]]
    # 断言：新数据的转换结果应该与预期的数据 X_expected_new 接近
    assert_array_almost_equal(X_trans_new, X_expected_new, decimal=3)


def test_robust_scaler_unit_variance():
    # 使用具有离群值的标准正态分布数据来检查 RobustScaler 在 unit_variance=True 时的表现
    rng = np.random.RandomState(42)
    X = rng.randn(1000000, 1)
    X_with_outliers = np.vstack([X, np.ones((100, 1)) * 100, np.ones((100, 1)) * -100])

    quantile_range = (1, 99)
    # 创建 RobustScaler 对象，设置 quantile_range 和 unit_variance=True，并对数据进行拟合
    robust_scaler = RobustScaler(quantile_range=quantile_range, unit_variance=True).fit(
        X_with_outliers
    )
    # 对数据 X 进行转换
    X_trans = robust_scaler.transform(X)

    # 断言：期望 RobustScaler 的中心（center_）接近 0
    assert robust_scaler.center_ == pytest.approx(0, abs=1e-3)
    # 断言：期望 RobustScaler 的尺度（scale_）接近 1
    assert robust_scaler.scale_ == pytest.approx(1, abs=1e-2)
    # 断言：转换后的数据 X_trans 的标准差应接近 1
    assert X_trans.std() == pytest.approx(1, abs=1e-2)


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_maxabs_scaler_zero_variance_features(sparse_container):
    # 检查 MaxAbsScaler 在具有零方差特征的示例数据上的表现
    X = [[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]]

    # 创建 MaxAbsScaler 对象
    scaler = MaxAbsScaler()
    # 对数据 X 进行拟合并转换
    X_trans = scaler.fit_transform(X)

    # 预期的转换后的数据
    X_expected = [
        [0.0, 1.0, 1.0 / 3.0],
        [0.0, 1.0, -0.2],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    # 断言：转换后的数据应该与预期的数据 X_expected 接近
    assert_array_almost_equal(X_trans, X_expected)
    
    # 对转换后的数据进行逆转换
    X_trans_inv = scaler.inverse_transform(X_trans)
    # 断言：逆转换后的数据应该与原始数据 X 接近
    assert_array_almost_equal(X, X_trans_inv)
    # 确保新数据正确转换
    X_new = [[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]]
    # 使用预先定义的缩放器对新数据进行转换
    X_trans_new = scaler.transform(X_new)
    # 预期的转换后数据
    X_expected_new = [[+0.0, 2.0, 1.0 / 3.0], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.0]]
    
    # 使用断言确保转换后的数据与预期一致，允许小数点后两位的误差
    assert_array_almost_equal(X_trans_new, X_expected_new, decimal=2)

    # 函数接口测试
    X_trans = maxabs_scale(X)
    # 使用断言确保转换后的数据与预期一致
    assert_array_almost_equal(X_trans, X_expected)

    # 稀疏数据测试
    X_sparse = sparse_container(X)
    # 使用缩放器对稀疏数据进行拟合和转换
    X_trans_sparse = scaler.fit_transform(X_sparse)
    # 预期的转换后数据
    X_expected = [
        [0.0, 1.0, 1.0 / 3.0],
        [0.0, 1.0, -0.2],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    # 使用断言确保稀疏数据转换后的结果与预期一致，转换为密集数组进行比较
    assert_array_almost_equal(X_trans_sparse.toarray(), X_expected)
    # 对转换后的稀疏数据进行逆转换
    X_trans_sparse_inv = scaler.inverse_transform(X_trans_sparse)
    # 使用断言确保逆转换后的数据与原始数据一致，转换为密集数组进行比较
    assert_array_almost_equal(X, X_trans_sparse_inv.toarray())
def test_maxabs_scaler_large_negative_value():
    # Check MaxAbsScaler on toy data with a large negative value

    # 定义输入数据 X，包含四个样本，每个样本有四个特征
    X = [
        [0.0, 1.0, +0.5, -1.0],
        [0.0, 1.0, -0.3, -0.5],
        [0.0, 1.0, -100.0, 0.0],
        [0.0, 0.0, +0.0, -2.0],
    ]

    # 创建 MaxAbsScaler 的实例
    scaler = MaxAbsScaler()

    # 对数据 X 进行拟合和转换
    X_trans = scaler.fit_transform(X)

    # 预期的转换后数据 X_expected
    X_expected = [
        [0.0, 1.0, 0.005, -0.5],
        [0.0, 1.0, -0.003, -0.25],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ]

    # 断言转换后的数据与预期数据 X_expected 几乎相等
    assert_array_almost_equal(X_trans, X_expected)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_maxabs_scaler_transform_one_row_csr(csr_container):
    # Check MaxAbsScaler on transforming csr matrix with one row

    # 创建一个只有一行的 CSR 矩阵 X
    X = csr_container([[0.5, 1.0, 1.0]])

    # 创建 MaxAbsScaler 的实例
    scaler = MaxAbsScaler()

    # 对 X 进行拟合
    scaler = scaler.fit(X)

    # 对 X 进行转换
    X_trans = scaler.transform(X)

    # 预期的转换后的数据 X_expected
    X_expected = csr_container([[1.0, 1.0, 1.0]])

    # 断言转换后的数据 X_trans 与预期数据 X_expected 的数组表示几乎相等
    assert_array_almost_equal(X_trans.toarray(), X_expected.toarray())

    # 对转换后的数据进行逆转换
    X_scaled_back = scaler.inverse_transform(X_trans)

    # 断言逆转换后的数据与原始数据 X 几乎相等
    assert_array_almost_equal(X.toarray(), X_scaled_back.toarray())


def test_maxabs_scaler_1d():
    # Test scaling of dataset along single axis

    # 遍历不同的输入数据 X 进行测试
    for X in [X_1row, X_1col, X_list_1row, X_list_1row]:
        # 创建 MaxAbsScaler 的实例
        scaler = MaxAbsScaler(copy=True)

        # 对数据 X 进行拟合和转换
        X_scaled = scaler.fit(X).transform(X)

        # 如果 X 是列表，则在进行缩放后转换为 NumPy 数组
        if isinstance(X, list):
            X = np.array(X)  # cast only after scaling done

        # 检查数据 X 的维度
        if _check_dim_1axis(X) == 1:
            assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), np.ones(n_features))
        else:
            assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), 1.0)

        # 断言 scaler.n_samples_seen_ 等于 X 的样本数量
        assert scaler.n_samples_seen_ == X.shape[0]

        # 检查逆转换
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_almost_equal(X_scaled_back, X)

    # 创建一个全为常数的特征矩阵 X
    X = np.ones((5, 1))

    # 创建 MaxAbsScaler 的实例
    scaler = MaxAbsScaler()

    # 对数据 X 进行拟合和转换
    X_scaled = scaler.fit(X).transform(X)

    # 断言经过缩放后的数据 X_scaled 的最大绝对值等于 1.0
    assert_array_almost_equal(np.abs(X_scaled.max(axis=0)), 1.0)

    # 断言 scaler.n_samples_seen_ 等于 X 的样本数量
    assert scaler.n_samples_seen_ == X.shape[0]

    # 函数接口测试
    X_1d = X_1row.ravel()
    max_abs = np.abs(X_1d).max()

    # 断言经过 maxabs_scale 函数处理后的结果与手动计算的结果几乎相等
    assert_array_almost_equal(X_1d / max_abs, maxabs_scale(X_1d, copy=True))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_maxabs_scaler_partial_fit(csr_container):
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit

    # 从二维数据 X_2d 中取出前 100 行作为测试数据 X
    X = X_2d[:100, :]

    # 数据 X 的样本数量 n
    n = X.shape[0]
    for chunk_size in [1, 2, 50, n, n + 42]:
        # 使用不同的chunk_size进行循环测试

        # 创建一个新的MaxAbsScaler对象，拟合整个数据集X，并生成批处理的标量转换器
        scaler_batch = MaxAbsScaler().fit(X)

        # 创建三个新的MaxAbsScaler对象，分别用于增量拟合原始数据，稀疏矩阵数据（CSR格式），稀疏矩阵数据（CSC格式）
        scaler_incr = MaxAbsScaler()
        scaler_incr_csr = MaxAbsScaler()
        scaler_incr_csc = MaxAbsScaler()

        # 对于通过gen_batches生成的每个批次，分别使用增量拟合方法进行拟合
        for batch in gen_batches(n, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            
            # 将批次数据X[batch]转换为CSR格式，并使用增量拟合方法拟合稀疏矩阵数据
            X_csr = csr_container(X[batch])
            scaler_incr_csr = scaler_incr_csr.partial_fit(X_csr)
            
            # 将批次数据X[batch]转换为CSC格式，并使用增量拟合方法拟合稀疏矩阵数据
            X_csc = csc_container(X[batch])
            scaler_incr_csc = scaler_incr_csc.partial_fit(X_csc)

        # 断言：验证批处理对象与增量拟合对象的最大值绝对值、样本数是否一致
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csr.max_abs_)
        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr_csc.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csr.n_samples_seen_
        assert scaler_batch.n_samples_seen_ == scaler_incr_csc.n_samples_seen_

        # 断言：验证批处理对象与增量拟合对象的标量转换比例是否一致
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csr.scale_)
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr_csc.scale_)
        
        # 断言：验证批处理对象与增量拟合对象的数据转换结果是否一致
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # 测试在进行一步增量拟合后的标准差
        batch0 = slice(0, chunk_size)
        scaler_batch = MaxAbsScaler().fit(X[batch0])
        scaler_incr = MaxAbsScaler().partial_fit(X[batch0])

        assert_array_almost_equal(scaler_batch.max_abs_, scaler_incr.max_abs_)
        assert scaler_batch.n_samples_seen_ == scaler_incr.n_samples_seen_
        assert_array_almost_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_almost_equal(scaler_batch.transform(X), scaler_incr.transform(X))

        # 测试直到增量拟合的末尾的标准差，并验证assert_correct_incr函数的调用
        scaler_batch = MaxAbsScaler().fit(X)
        scaler_incr = MaxAbsScaler()  # 清除先前的估算器
        for i, batch in enumerate(gen_batches(n, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(
                i,
                batch_start=batch.start,
                batch_stop=batch.stop,
                n=n,
                chunk_size=chunk_size,
                n_samples_seen=scaler_incr.n_samples_seen_,
            )
# 定义用于检查标准化器的方便函数，用于 `test_normalizer_l1_l2_max` 和 `test_normalizer_l1_l2_max_non_csr`
def check_normalizer(norm, X_norm):
    if norm == "l1":
        # 计算每行绝对值的和
        row_sums = np.abs(X_norm).sum(axis=1)
        # 断言前三行的和接近1.0
        for i in range(3):
            assert_almost_equal(row_sums[i], 1.0)
        # 断言第四行的和接近0.0
        assert_almost_equal(row_sums[3], 0.0)
    elif norm == "l2":
        # 断言前三行的 L2 范数接近1.0
        for i in range(3):
            assert_almost_equal(la.norm(X_norm[i]), 1.0)
        # 断言第四行的 L2 范数接近0.0
        assert_almost_equal(la.norm(X_norm[3]), 0.0)
    elif norm == "max":
        # 计算每行绝对值的最大值
        row_maxs = abs(X_norm).max(axis=1)
        # 断言前三行的最大值接近1.0
        for i in range(3):
            assert_almost_equal(row_maxs[i], 1.0)
        # 断言第四行的最大值接近0.0
        assert_almost_equal(row_maxs[3], 0.0)


# 使用参数化测试对 "l1", "l2", "max" 进行测试
@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_normalizer_l1_l2_max(norm, csr_container):
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个 4x5 的随机数组
    X_dense = rng.randn(4, 5)
    # 将稠密数组转换为 CSR 格式的稀疏矩阵
    X_sparse_unpruned = csr_container(X_dense)

    # 将第三行置为零
    X_dense[3, :] = 0.0

    # 在不修剪的情况下将第三行置为零（在实际情况中可能发生）
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # 使用正常构造函数构建修剪后的稀疏矩阵
    X_sparse_pruned = csr_container(X_dense)

    # 检查支持无复制优化的输入
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):
        # 创建标准化器对象，拷贝输入数据
        normalizer = Normalizer(norm=norm, copy=True)
        # 对数据进行标准化转换，得到标准化后的数组
        X_norm1 = normalizer.transform(X)
        # 断言标准化后的数组不是原始输入的引用
        assert X_norm1 is not X
        # 将稀疏矩阵转换为稠密数组
        X_norm1 = toarray(X_norm1)

        # 创建标准化器对象，不拷贝输入数据
        normalizer = Normalizer(norm=norm, copy=False)
        # 对数据进行标准化转换，得到标准化后的数组
        X_norm2 = normalizer.transform(X)
        # 断言标准化后的数组是原始输入的引用
        assert X_norm2 is X
        # 将稀疏矩阵转换为稠密数组
        X_norm2 = toarray(X_norm2)

        # 对于每个标准化后的数组，调用检查函数检查其正确性
        for X_norm in (X_norm1, X_norm2):
            check_normalizer(norm, X_norm)


# 使用参数化测试对 "l1", "l2", "max" 进行测试（非 CSR 格式）
@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + LIL_CONTAINERS
)
def test_normalizer_l1_l2_max_non_csr(norm, sparse_container):
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个 4x5 的随机数组
    X_dense = rng.randn(4, 5)

    # 将第三行置为零
    X_dense[3, :] = 0.0

    # 将稠密数组转换为指定稀疏矩阵格式
    X = sparse_container(X_dense)
    # 使用指定的标准化方式对稀疏矩阵进行标准化转换
    X_norm = Normalizer(norm=norm, copy=False).transform(X)

    # 断言标准化后的数组不是原始输入的引用
    assert X_norm is not X
    # 断言标准化后的数组是稀疏格式且格式为 "csr"
    assert sparse.issparse(X_norm) and X_norm.format == "csr"

    # 将稀疏矩阵转换为稠密数组，并调用检查函数检查其正确性
    X_norm = toarray(X_norm)
    check_normalizer(norm, X_norm)


# 使用参数化测试对 CSR 格式的输入进行最大值标准化测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_normalizer_max_sign(csr_container):
    # 检查在负数据情况下我们会标准化为正数
    rng = np.random.RandomState(0)
    # 生成一个 4x5 的随机数组
    X_dense = rng.randn(4, 5)
    # 将第三行置为零
    X_dense[3, :] = 0.0
    # 在混合数据中检查具有最大幅度的值为负数的情况
    X_dense[2, abs(X_dense[2, :]).argmax()] *= -1
    # 将所有值取负
    X_all_neg = -np.abs(X_dense)
    # 将取负后的稠密数组转换为 CSR 格式的稀疏矩阵
    X_all_neg_sparse = csr_container(X_all_neg)
    # 对于每个数据集 X_dense、X_all_neg、X_all_neg_sparse，依次执行以下操作：

        # 创建一个最大值归一化器
        normalizer = Normalizer(norm="max")

        # 对当前数据集 X 进行最大值归一化，得到归一化后的数据集 X_norm
        X_norm = normalizer.transform(X)

        # 断言归一化后的数据集 X_norm 与原始数据集 X 不是同一个对象
        assert X_norm is not X

        # 将归一化后的稠密数据集 X_norm 转换为稀疏表示
        X_norm = toarray(X_norm)

        # 断言归一化后的数据集 X_norm 的符号与原始数据集 X 转换为稀疏表示后的符号相同
        assert_array_equal(np.sign(X_norm), np.sign(toarray(X)))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest.mark.parametrize 装饰器，对测试函数 test_normalize 参数化，csr_container 为参数
def test_normalize(csr_container):
    # 测试 normalize 函数
    # 仅测试 Normalizer 测试用例未使用的功能
    X = np.random.RandomState(37).randn(3, 2)
    # 使用随机种子为 37 的随机数生成器生成随机数组 X
    assert_array_equal(normalize(X, copy=False), normalize(X.T, axis=0, copy=False).T)
    # 断言 normalize 函数的结果是否符合预期

    rs = np.random.RandomState(0)
    # 使用随机种子为 0 的随机数生成器创建 rs
    X_dense = rs.randn(10, 5)
    # 使用 rs 生成 10x5 的随机浮点数数组 X_dense
    X_sparse = csr_container(X_dense)
    # 使用 csr_container 将 X_dense 转换为稀疏矩阵 X_sparse
    ones = np.ones((10))
    # 创建元素全为 1 的长度为 10 的数组 ones
    for X in (X_dense, X_sparse):
        # 遍历 X_dense 和 X_sparse
        for dtype in (np.float32, np.float64):
            # 遍历数据类型 np.float32 和 np.float64
            for norm in ("l1", "l2"):
                # 遍历规范化方式 "l1" 和 "l2"
                X = X.astype(dtype)
                # 将 X 转换为指定的数据类型 dtype
                X_norm = normalize(X, norm=norm)
                # 使用指定的规范化方式对 X 进行规范化并赋值给 X_norm
                assert X_norm.dtype == dtype
                # 断言 X_norm 的数据类型是否为 dtype

                X_norm = toarray(X_norm)
                # 将 X_norm 转换为密集数组
                if norm == "l1":
                    row_sums = np.abs(X_norm).sum(axis=1)
                    # 如果规范化方式是 "l1"，计算每行绝对值之和
                else:
                    X_norm_squared = X_norm**2
                    row_sums = X_norm_squared.sum(axis=1)
                    # 否则，计算每行平方和

                assert_array_almost_equal(row_sums, ones)
                # 断言 row_sums 是否几乎等于 ones

    # 测试 return_norm
    X_dense = np.array([[3.0, 0, 4.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    # 创建一个 3x3 的浮点数数组 X_dense
    for norm in ("l1", "l2", "max"):
        # 遍历规范化方式 "l1"、"l2" 和 "max"
        _, norms = normalize(X_dense, norm=norm, return_norm=True)
        # 对 X_dense 使用指定的规范化方式进行规范化，并返回规范化后的数组和 norms
        if norm == "l1":
            assert_array_almost_equal(norms, np.array([7.0, 1.0, 5.0]))
            # 如果规范化方式是 "l1"，断言 norms 是否几乎等于指定的数组
        elif norm == "l2":
            assert_array_almost_equal(norms, np.array([5.0, 1.0, 3.60555127]))
            # 如果规范化方式是 "l2"，断言 norms 是否几乎等于指定的数组
        else:
            assert_array_almost_equal(norms, np.array([4.0, 1.0, 3.0]))
            # 否则，断言 norms 是否几乎等于指定的数组

    X_sparse = csr_container(X_dense)
    # 将 X_dense 转换为稀疏矩阵 X_sparse
    for norm in ("l1", "l2"):
        # 遍历规范化方式 "l1" 和 "l2"
        with pytest.raises(NotImplementedError):
            normalize(X_sparse, norm=norm, return_norm=True)
            # 断言对于稀疏矩阵 X_sparse，使用规范化方式 "l1" 和 "l2" 会引发 NotImplementedError 异常
    _, norms = normalize(X_sparse, norm="max", return_norm=True)
    # 对稀疏矩阵 X_sparse 使用规范化方式 "max" 进行规范化，并返回规范化后的数组和 norms
    assert_array_almost_equal(norms, np.array([4.0, 1.0, 3.0]))
    # 断言 norms 是否几乎等于指定的数组
    # 检查构造函数是否为列表类型，如果不是则断言 X_bin 应该等于 X_float
    if constructor is not list:
        assert X_bin is X_float

    # 将 X_bin 转换为稀疏数组表示
    X_bin = toarray(X_bin)
    # 断言 X_bin 中值为 0 的元素个数为 2
    assert np.sum(X_bin == 0) == 2
    # 断言 X_bin 中值为 1 的元素个数为 4
    assert np.sum(X_bin == 1) == 4

    # 创建 Binarizer 对象，使用阈值 -0.5 进行二值化，复制原始数据
    binarizer = Binarizer(threshold=-0.5, copy=True)
    # 如果构造函数为 np.array 或 list
    if constructor in (np.array, list):
        # 复制并转换 X_ 到 X
        X = constructor(X_.copy())

        # 对 X 进行二值化并转换为稀疏数组表示
        X_bin = toarray(binarizer.transform(X))
        # 断言 X_bin 中值为 0 的元素个数为 1
        assert np.sum(X_bin == 0) == 1
        # 断言 X_bin 中值为 1 的元素个数为 5
        assert np.sum(X_bin == 1) == 5
        # 再次使用 binarizer 对象对 X 进行二值化
        X_bin = binarizer.transform(X)

    # 如果构造函数为 CSC_CONTAINERS 中的一种
    # 使用 pytest 模块断言调用 binarizer.transform(X) 时会抛出 ValueError 异常
    if constructor in CSC_CONTAINERS:
        with pytest.raises(ValueError):
            binarizer.transform(constructor(X))
# 测试 KernelCenterer 是否等同于 StandardScaler 在特征空间中的表现
def test_center_kernel():
    # 创建一个随机数生成器对象
    rng = np.random.RandomState(0)
    # 生成一个 5x4 的随机矩阵作为 X_fit
    X_fit = rng.random_sample((5, 4))
    # 使用 StandardScaler 进行无标准差缩放的拟合
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_fit)
    # 将 X_fit 进行中心化
    X_fit_centered = scaler.transform(X_fit)
    # 计算 X_fit 的核矩阵 K_fit
    K_fit = np.dot(X_fit, X_fit.T)

    # 创建一个 KernelCenterer 对象
    centerer = KernelCenterer()
    # 将 X_fit_centered 进行中心化得到核矩阵 K_fit_centered
    K_fit_centered = np.dot(X_fit_centered, X_fit_centered.T)
    # 使用 centerer 对 K_fit 进行中心化得到 K_fit_centered2
    K_fit_centered2 = centerer.fit_transform(K_fit)
    # 断言 K_fit_centered 和 K_fit_centered2 的近似相等性
    assert_array_almost_equal(K_fit_centered, K_fit_centered2)

    # 预测数据集 X_pred
    X_pred = rng.random_sample((2, 4))
    # 计算预测数据 X_pred 与训练数据 X_fit 的核矩阵 K_pred
    K_pred = np.dot(X_pred, X_fit.T)
    # 将 X_pred 进行中心化
    X_pred_centered = scaler.transform(X_pred)
    # 计算预测数据的中心化核矩阵 K_pred_centered
    K_pred_centered = np.dot(X_pred_centered, X_fit_centered.T)
    # 使用 centerer 对预测核矩阵 K_pred 进行中心化得到 K_pred_centered2
    K_pred_centered2 = centerer.transform(K_pred)
    # 断言 K_pred_centered 和 K_pred_centered2 的近似相等性
    assert_array_almost_equal(K_pred_centered, K_pred_centered2)

    # 使用下列方法验证结果与文献中方法的一致性：
    # B. Schölkopf, A. Smola, and K.R. Müller,
    # "Nonlinear component analysis as a kernel eigenvalue problem"
    # 方程 (B.3)

    # 计算中心化后的核矩阵 K_fit_centered3
    ones_M = np.ones_like(K_fit) / K_fit.shape[0]
    K_fit_centered3 = K_fit - ones_M @ K_fit - K_fit @ ones_M + ones_M @ K_fit @ ones_M
    # 断言 K_fit_centered 和 K_fit_centered3 的近似相等性
    assert_allclose(K_fit_centered, K_fit_centered3)

    # 计算预测核矩阵 K_pred_centered3
    ones_prime_M = np.ones_like(K_pred) / K_fit.shape[0]
    K_pred_centered3 = (
        K_pred - ones_prime_M @ K_fit - K_pred @ ones_M + ones_prime_M @ K_fit @ ones_M
    )
    # 断言 K_pred_centered 和 K_pred_centered3 的近似相等性
    assert_allclose(K_pred_centered, K_pred_centered3)


# 非线性核的 KernelCenterer 中心化检查
def test_kernelcenterer_non_linear_kernel():
    # 创建一个随机数生成器对象
    rng = np.random.RandomState(0)
    # 生成大小为 (100, 50) 和 (20, 50) 的随机数据集 X 和 X_test
    X, X_test = rng.randn(100, 50), rng.randn(20, 50)

    # 定义映射函数 phi
    def phi(X):
        """我们的映射函数 phi."""
        return np.vstack(
            [
                np.clip(X, a_min=0, a_max=None),
                -np.clip(X, a_min=None, a_max=0),
            ]
        )

    # 对数据集 X 和 X_test 进行映射得到 phi_X 和 phi_X_test
    phi_X = phi(X)
    phi_X_test = phi(X_test)

    # 使用 StandardScaler 对映射后的数据进行中心化
    scaler = StandardScaler(with_std=False)
    phi_X_center = scaler.fit_transform(phi_X)
    phi_X_test_center = scaler.transform(phi_X_test)

    # 创建不同的核矩阵
    K = phi_X @ phi_X.T
    K_test = phi_X_test @ phi_X.T
    K_center = phi_X_center @ phi_X_center.T
    K_test_center = phi_X_test_center @ phi_X_center.T

    # 创建 KernelCenterer 对象并对 K 进行拟合
    kernel_centerer = KernelCenterer()
    kernel_centerer.fit(K)

    # 断言 KernelCenterer 对 K 和 K_test 的中心化结果与 K_center 和 K_test_center 的近似相等性
    assert_allclose(kernel_centerer.transform(K), K_center)
    assert_allclose(kernel_centerer.transform(K_test), K_test_center)

    # 使用下列方法验证结果与文献中方法的一致性：
    # B. Schölkopf, A. Smola, and K.R. Müller,
    # "Nonlinear component analysis as a kernel eigenvalue problem"
    # 方程 (B.3)
    # 计算中心化的核矩阵 K_centered = (I - 1_M) K (I - 1_M)
    # 其中 I 是单位矩阵，1_M 是元素均为 1 的矩阵
    ones_M = np.ones_like(K) / K.shape[0]  # 创建一个与 K 形状相同、元素为均匀分布的 1/M 的数组
    K_centered = (
        K - ones_M @ K - K @ ones_M + ones_M @ K @ ones_M
    )  # 计算中心化的核矩阵 K_centered
    
    assert_allclose(kernel_centerer.transform(K), K_centered)  # 断言转换后的核矩阵与计算的中心化核矩阵 K_centered 相近
    
    # 计算中心化的测试核矩阵 K_test_centered = (K_test - 1'_M K)(I - 1_M)
    # 其中 K_test 是测试用的核矩阵，1'_M 是测试集上元素均为 1 的矩阵，1_M 是训练集上元素均为 1 的矩阵
    ones_prime_M = np.ones_like(K_test) / K.shape[0]  # 创建一个与 K_test 形状相同、元素为均匀分布的 1/M 的数组
    K_test_centered = (
        K_test - ones_prime_M @ K - K_test @ ones_M + ones_prime_M @ K @ ones_M
    )  # 计算中心化的测试核矩阵 K_test_centered
    
    assert_allclose(kernel_centerer.transform(K_test), K_test_centered)  # 断言转换后的测试核矩阵与计算的中心化测试核矩阵 K_test_centered 相近
def test_cv_pipeline_precomputed():
    # 对四个共面点执行回归交叉验证，这些点具有相同的值。
    # 使用预计算的核函数确保 Pipeline 中的 KernelCenterer 被视为成对操作。
    X = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]])
    y_true = np.ones((4,))
    # 计算输入矩阵 X 的核矩阵 K
    K = X.dot(X.T)
    # 创建一个 KernelCenterer 对象
    kcent = KernelCenterer()
    # 创建一个 Pipeline 对象，包括 KernelCenterer 和 SVR (支持向量回归)
    pipeline = Pipeline([("kernel_centerer", kcent), ("svr", SVR())])

    # 检查 Pipeline 是否设置了成对属性
    assert pipeline._get_tags()["pairwise"]

    # 测试交叉验证，预测值应该几乎完美匹配真实值
    # 注意：这个测试相当空洞 -- 主要是为了测试 Pipeline 和 KernelCenterer 的集成
    y_pred = cross_val_predict(pipeline, K, y_true, cv=2)
    assert_array_almost_equal(y_true, y_pred)


def test_fit_transform():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    # 对标准化、归一化和二值化器依次进行拟合和转换操作，并比较两种方式的结果
    for obj in (StandardScaler(), Normalizer(), Binarizer()):
        X_transformed = obj.fit(X).transform(X)
        X_transformed2 = obj.fit_transform(X)
        assert_array_equal(X_transformed, X_transformed2)


def test_add_dummy_feature():
    # 添加虚拟特征到输入矩阵 X
    X = [[1, 0], [0, 1], [0, 1]]
    X = add_dummy_feature(X)
    assert_array_equal(X, [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_add_dummy_feature_sparse(sparse_container):
    # 对稀疏矩阵容器进行参数化测试，确保添加虚拟特征后格式不变
    X = sparse_container([[1, 0], [0, 1], [0, 1]])
    desired_format = X.format
    X = add_dummy_feature(X)
    assert sparse.issparse(X) and X.format == desired_format, X
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_fit_cold_start():
    X = iris.data
    X_2d = X[:, :2]

    # 具有 partial_fit 方法的缩放器
    scalers = [
        StandardScaler(with_mean=False, with_std=False),
        MinMaxScaler(),
        MaxAbsScaler(),
    ]

    for scaler in scalers:
        # 对完整数据集 X 进行拟合和转换
        scaler.fit_transform(X)
        # 由于形状不同，这可能会破坏缩放器，除非内部状态被重置
        scaler.fit_transform(X_2d)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_notfitted(method):
    # 测试 PowerTransformer 对象在未拟合状态下对数据的转换行为
    pt = PowerTransformer(method=method)
    X = np.abs(X_1col)
    with pytest.raises(NotFittedError):
        pt.transform(X)
    with pytest.raises(NotFittedError):
        pt.inverse_transform(X)


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("X", [X_1col, X_2d])
def test_power_transformer_inverse(method, standardize, X):
    # 确保在应用变换和反向变换时能够得到原始输入
    X = np.abs(X) if method == "box-cox" else X
    pt = PowerTransformer(method=method, standardize=standardize)
    X_trans = pt.fit_transform(X)
    assert_almost_equal(X, pt.inverse_transform(X_trans))


def test_power_transformer_1d():
    # 确保 PowerTransformer 在处理一维数据时的行为正确
    pass  # Placeholder for future implementation
    # 计算 X 矩阵每个元素的绝对值
    X = np.abs(X_1col)

    # 对于每个 standardize 值为 True 和 False，分别进行以下操作
    for standardize in [True, False]:
        # 创建 PowerTransformer 对象，使用 Box-Cox 方法，并指定是否标准化
        pt = PowerTransformer(method="box-cox", standardize=standardize)

        # 使用 PowerTransformer 对象拟合并转换 X 矩阵
        X_trans = pt.fit_transform(X)
        # 使用 power_transform 函数进行 Box-Cox 变换
        X_trans_func = power_transform(X, method="box-cox", standardize=standardize)

        # 使用 scipy 的 stats 模块计算 X 矩阵的 Box-Cox 变换后的期望值及 lambda 值
        X_expected, lambda_expected = stats.boxcox(X.flatten())

        # 如果 standardize 为 True，则对 X_expected 进行标准化处理
        if standardize:
            X_expected = scale(X_expected)

        # 断言 X_trans 与 X_expected 形状一致且数值接近
        assert_almost_equal(X_expected.reshape(-1, 1), X_trans)
        # 断言 X_trans_func 与 X_expected 形状一致且数值接近
        assert_almost_equal(X_expected.reshape(-1, 1), X_trans_func)

        # 断言使用 PowerTransformer 对象的反向转换结果与原始 X 矩阵形状一致且数值接近
        assert_almost_equal(X, pt.inverse_transform(X_trans))
        # 断言 PowerTransformer 对象的 lambda 值接近预期的 lambda_expected
        assert_almost_equal(lambda_expected, pt.lambdas_[0])

        # 断言 PowerTransformer 对象的 lambda_ 数组长度与 X 矩阵的列数一致
        assert len(pt.lambdas_) == X.shape[1]
        # 断言 PowerTransformer 对象的 lambda_ 是 NumPy 数组类型
        assert isinstance(pt.lambdas_, np.ndarray)
def test_power_transformer_2d():
    # 使用 X_2d 的绝对值作为输入数据 X
    X = np.abs(X_2d)

    # 对于 standardize 参数分别进行 True 和 False 的测试
    for standardize in [True, False]:
        # 创建 PowerTransformer 对象，使用 Box-Cox 方法，并指定 standardize 参数
        pt = PowerTransformer(method="box-cox", standardize=standardize)

        # 使用对象的 fit_transform 方法和 power_transform 函数进行数据转换
        X_trans_class = pt.fit_transform(X)
        X_trans_func = power_transform(X, method="box-cox", standardize=standardize)

        # 遍历 X_trans_class 和 X_trans_func 进行进一步的检查
        for X_trans in [X_trans_class, X_trans_func]:
            # 针对每列数据执行 Box-Cox 变换，并与预期结果进行比较
            for j in range(X_trans.shape[1]):
                X_expected, lmbda = stats.boxcox(X[:, j].flatten())

                # 如果 standardize 为 True，则对预期结果进行缩放处理
                if standardize:
                    X_expected = scale(X_expected)

                # 使用 assert_almost_equal 断言实际结果与预期结果的近似程度
                assert_almost_equal(X_trans[:, j], X_expected)
                assert_almost_equal(lmbda, pt.lambdas_[j])

            # 测试逆转换的正确性
            X_inv = pt.inverse_transform(X_trans)
            assert_array_almost_equal(X_inv, X)

        # 断言对象的 lambdas_ 属性长度与输入数据 X 的列数相同
        assert len(pt.lambdas_) == X.shape[1]
        assert isinstance(pt.lambdas_, np.ndarray)


def test_power_transformer_boxcox_strictly_positive_exception():
    # 当方法为 box-cox 时，应当对负数数组和零数组引发异常

    # 创建 PowerTransformer 对象，指定方法为 box-cox
    pt = PowerTransformer(method="box-cox")
    pt.fit(np.abs(X_2d))
    X_with_negatives = X_2d
    not_positive_message = "strictly positive"

    # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(X_with_negatives)

    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(X_with_negatives)

    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(X_with_negatives, method="box-cox")

    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(np.zeros(X_2d.shape))

    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(np.zeros(X_2d.shape))

    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(np.zeros(X_2d.shape), method="box-cox")


@pytest.mark.parametrize("X", [X_2d, np.abs(X_2d), -np.abs(X_2d), np.zeros(X_2d.shape)])
def test_power_transformer_yeojohnson_any_input(X):
    # Yeo-Johnson 方法应当支持任何类型的输入
    power_transform(X, method="yeo-johnson")


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_shape_exception(method):
    # 对于不同方法应当引发形状异常

    # 创建 PowerTransformer 对象，指定方法为 method
    pt = PowerTransformer(method=method)
    X = np.abs(X_2d)
    pt.fit(X)

    # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配错误信息
    wrong_shape_message = (
        r"X has \d+ features, but PowerTransformer is " r"expecting \d+ features"
    )

    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.transform(X[:, 0:1])

    with pytest.raises(ValueError, match=wrong_shape_message):
        pt.inverse_transform(X[:, 0:1])


def test_power_transformer_lambda_zero():
    # 测试 lambda = 0 的情况

    # 创建 PowerTransformer 对象，使用 Box-Cox 方法，并指定 standardize 为 False
    pt = PowerTransformer(method="box-cox", standardize=False)
    X = np.abs(X_2d)[:, 0:1]

    # 设置 lambdas_ 属性为 [0]
    pt.lambdas_ = np.array([0])

    # 执行数据变换
    X_trans = pt.transform(X)
    # 使用逆变换函数 inverse_transform 对转换后的数据 X_trans 进行逆变换，并断言其与原始数据 X 几乎相等
    assert_array_almost_equal(pt.inverse_transform(X_trans), X)
def test_power_transformer_lambda_one():
    # 确保 lambda = 1 对于 yeo-johnson 方法相当于恒等变换
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X = np.abs(X_2d)[:, 0:1]  # 取二维数组 X_2d 的第一列的绝对值作为 X

    pt.lambdas_ = np.array([1])  # 设置 PowerTransformer 对象的 lambda 参数为 [1]
    X_trans = pt.transform(X)  # 对 X 进行变换
    assert_array_almost_equal(X_trans, X)  # 断言变换后的 X 与原始 X 大致相等


@pytest.mark.parametrize(
    "method, lmbda",
    [
        ("box-cox", 0.1),
        ("box-cox", 0.5),
        ("yeo-johnson", 0.1),
        ("yeo-johnson", 0.5),
        ("yeo-johnson", 1.0),
    ],
)
def test_optimization_power_transformer(method, lmbda):
    # 测试优化过程：
    # - 设置预定义的 lambda 值
    # - 对正态分布应用逆变换（得到 X_inv）
    # - 对 X_inv 应用 fit_transform（得到 X_inv_trans）
    # - 检查 X_inv_trans 是否与 X 大致相等

    rng = np.random.RandomState(0)
    n_samples = 20000
    X = rng.normal(loc=0, scale=1, size=(n_samples, 1))  # 从正态分布中生成样本数据 X

    pt = PowerTransformer(method=method, standardize=False)
    pt.lambdas_ = [lmbda]  # 设置 PowerTransformer 对象的 lambda 参数为 lmbda
    X_inv = pt.inverse_transform(X)  # 对 X 进行逆变换

    pt = PowerTransformer(method=method, standardize=False)
    X_inv_trans = pt.fit_transform(X_inv)  # 对 X_inv 进行 fit_transform

    assert_almost_equal(0, np.linalg.norm(X - X_inv_trans) / n_samples, decimal=2)  # 断言 X 与 X_inv_trans 的差异足够小
    assert_almost_equal(0, X_inv_trans.mean(), decimal=1)  # 断言 X_inv_trans 的均值接近 0
    assert_almost_equal(1, X_inv_trans.std(), decimal=1)  # 断言 X_inv_trans 的标准差接近 1


def test_yeo_johnson_darwin_example():
    # 测试原始论文 "A new family of power transformations to improve normality or symmetry" 中的示例
    X = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0, 3.0, 9.3, 7.5, -6.0]
    X = np.array(X).reshape(-1, 1)
    lmbda = PowerTransformer(method="yeo-johnson").fit(X).lambdas_
    assert np.allclose(lmbda, 1.305, atol=1e-3)  # 断言估计的 lambda 值接近 1.305


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_power_transformer_nans(method):
    # 确保 lambda 的估计不受 NaN 值的影响，并且 transform() 方法支持 NaN 值
    X = np.abs(X_1col)
    pt = PowerTransformer(method=method)
    pt.fit(X)
    lmbda_no_nans = pt.lambdas_[0]

    # 在末尾添加 NaN 值，检查 lambda 是否保持不变
    X = np.concatenate([X, np.full_like(X, np.nan)])
    X = shuffle(X, random_state=0)

    pt.fit(X)
    lmbda_nans = pt.lambdas_[0]

    assert_almost_equal(lmbda_no_nans, lmbda_nans, decimal=5)

    X_trans = pt.transform(X)
    assert_array_equal(np.isnan(X_trans), np.isnan(X))


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_fit_transform(method, standardize):
    # 检查 fit_transform() 和 fit().transform() 返回相同的值
    X = X_1col
    if method == "box-cox":
        X = np.abs(X)

    pt = PowerTransformer(method, standardize=standardize)
    assert_array_almost_equal(pt.fit(X).transform(X), pt.fit_transform(X))
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_copy_True(method, standardize):
    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试参数为 standardize，取值为 True 和 False
    # 定义测试函数 test_power_transformer_copy_True，测试 PowerTransformer 类的行为

    X = X_1col
    # 将变量 X 设置为 X_1col 的引用

    if method == "box-cox":
        X = np.abs(X)
        # 如果 method 参数为 "box-cox"，则对 X 取绝对值

    X_original = X.copy()
    # 复制 X 并将其赋值给 X_original，用于后续的比较

    assert X is not X_original  # sanity checks
    # 断言 X 和 X_original 不是同一个对象，作为健全性检查的一部分
    assert_array_almost_equal(X, X_original)
    # 断言 X 和 X_original 数组几乎相等

    pt = PowerTransformer(method, standardize=standardize, copy=True)
    # 创建 PowerTransformer 对象 pt，使用给定的 method 和 standardize 参数，copy=True

    pt.fit(X)
    # 对 X 进行拟合

    assert_array_almost_equal(X, X_original)
    # 断言 X 和 X_original 数组几乎相等，fit 方法不应该改变 X

    X_trans = pt.transform(X)
    # 使用拟合后的 pt 对象对 X 进行变换
    assert X_trans is not X
    # 断言 X_trans 不是 X 的引用

    X_trans = pt.fit_transform(X)
    # 使用 pt 对象对 X 进行拟合并变换
    assert_array_almost_equal(X, X_original)
    # 断言 X 和 X_original 数组几乎相等，fit_transform 方法不应该改变 X
    assert X_trans is not X
    # 断言 X_trans 不是 X 的引用

    X_inv_trans = pt.inverse_transform(X_trans)
    # 使用 pt 对象对 X_trans 进行逆变换
    assert X_trans is not X_inv_trans
    # 断言 X_trans 不是 X_inv_trans 的引用


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [True, False])
def test_power_transformer_copy_False(method, standardize):
    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试参数为 method 和 standardize
    # method 参数取值为 "box-cox" 和 "yeo-johnson"，standardize 参数取值为 True 和 False
    # 定义测试函数 test_power_transformer_copy_False，测试 PowerTransformer 类的行为

    X = X_1col
    # 将变量 X 设置为 X_1col 的引用

    if method == "box-cox":
        X = np.abs(X)
        # 如果 method 参数为 "box-cox"，则对 X 取绝对值

    X_original = X.copy()
    # 复制 X 并将其赋值给 X_original，用于后续的比较

    assert X is not X_original  # sanity checks
    # 断言 X 和 X_original 不是同一个对象，作为健全性检查的一部分
    assert_array_almost_equal(X, X_original)
    # 断言 X 和 X_original 数组几乎相等

    pt = PowerTransformer(method, standardize=standardize, copy=False)
    # 创建 PowerTransformer 对象 pt，使用给定的 method 和 standardize 参数，copy=False

    pt.fit(X)
    # 对 X 进行拟合
    assert_array_almost_equal(X, X_original)
    # 断言 X 和 X_original 数组几乎相等，fit 方法不应该改变 X

    X_trans = pt.transform(X)
    # 使用拟合后的 pt 对象对 X 进行变换
    assert X_trans is X
    # 断言 X_trans 是 X 的引用，因为 copy=False

    if method == "box-cox":
        X = np.abs(X)
        # 如果 method 参数为 "box-cox"，则对 X 取绝对值

    X_trans = pt.fit_transform(X)
    # 使用 pt 对象对 X 进行拟合并变换
    assert X_trans is X
    # 断言 X_trans 是 X 的引用，因为 copy=False

    X_inv_trans = pt.inverse_transform(X_trans)
    # 使用 pt 对象对 X_trans 进行逆变换
    assert X_trans is X_inv_trans
    # 断言 X_trans 是 X_inv_trans 的引用


def test_power_transformer_box_cox_raise_all_nans_col():
    """Check that box-cox raises informative when a column contains all nans.

    Non-regression test for gh-26303
    """
    # 检查当列包含所有 NaN 时，box-cox 是否会引发相关的错误信息
    # 非回归测试，用于检查是否修复了 gh-26303 的问题

    X = rng.random_sample((4, 5))
    # 创建一个 4x5 的随机数组 X
    X[:, 0] = np.nan
    # 将 X 的第一列设置为 NaN

    err_msg = "Column must not be all nan."
    # 错误信息，用于断言异常信息的内容

    pt = PowerTransformer(method="box-cox")
    # 创建 PowerTransformer 对象 pt，使用 "box-cox" 方法

    with pytest.raises(ValueError, match=err_msg):
        # 使用 pytest 的 raises 断言，期望引发 ValueError 异常，且异常信息匹配 err_msg
        pt.fit_transform(X)
        # 对 X 进行拟合并变换


@pytest.mark.parametrize(
    "X_2",
    [sparse.random(10, 1, density=0.8, random_state=0)]
    + [
        csr_container(np.full((10, 1), fill_value=np.nan))
        for csr_container in CSR_CONTAINERS
    ],
)
def test_standard_scaler_sparse_partial_fit_finite_variance(X_2):
    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试参数为 X_2
    # X_2 参数包含稀疏数组和 CSR 容器，用于测试 StandardScaler 的行为

    # 非回归测试，用于检查是否修复了 https://github.com/scikit-learn/scikit-learn/issues/16448

    X_1 = sparse.random(5, 1, density=0.8)
    # 创建一个 5x1 的稀疏随机数组 X_1

    scaler = StandardScaler(with_mean=False)
    # 创建 StandardScaler 对象 scaler，设置 with_mean=False

    scaler.fit(X_1).partial_fit(X_2)
    # 对 X_1 进行拟合并使用 partial_fit 对 X_2 进行部分拟合

    assert np.isfinite(scaler.var_[0])
    # 断言 scaler.var_[0] 是有限的，检查方差是否有限


@pytest.mark.parametrize("feature_range", [(0, 1), (-10, 10)])
def test_minmax_scaler_clip(feature_range):
    # 使用 pytest.mark.parametrize 装饰器标记参数化测试，测试参数为 feature_range
    # feature_range 参数为 MinMaxScaler 的 feature_range 参数取值范围

    # 测试 MinMaxScaler 中参数 'clip' 的行为

    X = iris.data
    # 使用 iris 数据集中的数据 X

    scaler = MinMaxScaler(feature_range=feature_range, clip=True).fit(X)
    # 创建 MinMaxScaler 对象 scaler，设置 feature_range 和 clip=True，并对 X 进行拟合

    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    # 计算 X 的每列的最小值和最大值
    # 创建一个包含单个测试样本的列表，该样本是根据 X_min 和 X_max 扩展后的范围
    X_test = [np.r_[X_min[:2] - 10, X_max[2:] + 10]]
    
    # 使用预先定义的 scaler 对 X_test 进行数据转换
    X_transformed = scaler.transform(X_test)
    
    # 使用 assert_allclose 函数验证 X_transformed 是否接近于预期的值
    assert_allclose(
        X_transformed,
        [[feature_range[0], feature_range[0], feature_range[1], feature_range[1]]],
    )
def test_standard_scaler_raise_error_for_1d_input():
    """Check that `inverse_transform` from `StandardScaler` raises an error
    with 1D array.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19518
    """
    # 使用标准化器 `StandardScaler` 对 2D 数据 `X_2d` 进行拟合
    scaler = StandardScaler().fit(X_2d)
    # 预期的错误信息字符串
    err_msg = "Expected 2D array, got 1D array instead"
    # 使用 pytest 来检测是否抛出 ValueError 异常，并验证错误信息是否匹配
    with pytest.raises(ValueError, match=err_msg):
        scaler.inverse_transform(X_2d[:, 0])


def test_power_transformer_significantly_non_gaussian():
    """Check that significantly non-Gaussian data before transforms correctly.

    For some explored lambdas, the transformed data may be constant and will
    be rejected. Non-regression test for
    https://github.com/scikit-learn/scikit-learn/issues/14959
    """
    # 创建一个非常非高斯分布的数据 `X_non_gaussian`
    X_non_gaussian = 1e6 * np.array(
        [0.6, 2.0, 3.0, 4.0] * 4 + [11, 12, 12, 16, 17, 20, 85, 90], dtype=np.float64
    ).reshape(-1, 1)
    # 初始化一个 PowerTransformer 对象
    pt = PowerTransformer()

    # 使用警告模块捕获 RuntimeWarning 警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 对非高斯数据进行拟合和转换
        X_trans = pt.fit_transform(X_non_gaussian)

    # 断言转换后的数据中不包含 NaN 值
    assert not np.any(np.isnan(X_trans))
    # 断言转换后的数据均值近似为 0.0
    assert X_trans.mean() == pytest.approx(0.0)
    # 断言转换后的数据标准差近似为 1.0
    assert X_trans.std() == pytest.approx(1.0)
    # 断言转换后的数据最小值大于 -2
    assert X_trans.min() > -2
    # 断言转换后的数据最大值小于 2


@pytest.mark.parametrize(
    "Transformer",
    [
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        StandardScaler,
        QuantileTransformer,
        PowerTransformer,
    ],
)
def test_one_to_one_features(Transformer):
    """Check one-to-one transformers give correct feature names."""
    # 使用给定的 Transformer 对象对 iris 数据集进行拟合
    tr = Transformer().fit(iris.data)
    # 获取转换后的特征名称列表
    names_out = tr.get_feature_names_out(iris.feature_names)
    # 断言转换后的特征名称列表与原始特征名称列表相等
    assert_array_equal(names_out, iris.feature_names)


@pytest.mark.parametrize(
    "Transformer",
    [
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        StandardScaler,
        QuantileTransformer,
        PowerTransformer,
        Normalizer,
        Binarizer,
    ],
)
def test_one_to_one_features_pandas(Transformer):
    """Check one-to-one transformers give correct feature names."""
    # 导入 pandas 库，如果不存在则跳过该测试
    pd = pytest.importorskip("pandas")

    # 创建包含 iris 数据的 DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 使用给定的 Transformer 对象对 DataFrame 进行拟合
    tr = Transformer().fit(df)

    # 获取默认的输出特征名称列表
    names_out_df_default = tr.get_feature_names_out()
    # 断言转换后的特征名称列表与原始特征名称列表相等
    assert_array_equal(names_out_df_default, iris.feature_names)

    # 获取指定输入特征名称后的输出特征名称列表
    names_out_df_valid_in = tr.get_feature_names_out(iris.feature_names)
    # 断言转换后的特征名称列表与原始特征名称列表相等
    assert_array_equal(names_out_df_valid_in, iris.feature_names)

    # 断言传入无效输入特征名称时抛出 ValueError 异常，错误信息匹配
    msg = re.escape("input_features is not equal to feature_names_in_")
    with pytest.raises(ValueError, match=msg):
        invalid_names = list("abcd")
        tr.get_feature_names_out(invalid_names)


def test_kernel_centerer_feature_names_out():
    """Test that kernel centerer `feature_names_out`."""

    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成随机样本数据矩阵 X
    X = rng.random_sample((6, 4))
    # 计算 X 的线性核
    X_pairwise = linear_kernel(X)
    # 初始化一个 KernelCenterer 对象并对 X_pairwise 进行拟合
    centerer = KernelCenterer().fit(X_pairwise)
    # 获取经过中心化处理后的特征名称列表
    names_out = centerer.get_feature_names_out()
    # 获取输入数据 X_pairwise 的列数，即样本数量
    samples_out2 = X_pairwise.shape[1]
    # 使用断言验证经过中心化后的特征名称列表是否与预期的列表一致
    assert_array_equal(names_out, [f"kernelcenterer{i}" for i in range(samples_out2)])
# 使用 pytest 的参数化装饰器，指定测试函数的标准化参数为 True 和 False 两种情况
@pytest.mark.parametrize("standardize", [True, False])
# 测试函数，检查 PowerTransformer 对常量特征的处理是否正确
def test_power_transformer_constant_feature(standardize):
    """Check that PowerTransfomer leaves constant features unchanged."""
    # 待转换的特征矩阵 X
    X = [[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]]

    # 创建 PowerTransformer 对象，使用 Yeo-Johnson 方法，并根据参数决定是否标准化
    pt = PowerTransformer(method="yeo-johnson", standardize=standardize).fit(X)

    # 检查每个特征的 Lambda 值是否为 1
    assert_allclose(pt.lambdas_, [1, 1, 1])

    # 对 X 进行拟合转换，获取转换后的结果 Xft
    Xft = pt.fit_transform(X)
    # 对 X 进行转换，获取转换后的结果 Xt
    Xt = pt.transform(X)

    # 遍历处理后的结果，根据标准化参数进行断言
    for Xt_ in [Xft, Xt]:
        if standardize:
            # 如果标准化为 True，则断言结果应接近全零矩阵
            assert_allclose(Xt_, np.zeros_like(X))
        else:
            # 如果标准化为 False，则断言结果应与原始输入 X 接近
            assert_allclose(Xt_, X)
```