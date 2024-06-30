# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_discretization.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架
import scipy.sparse as sp  # 导入 SciPy 稀疏矩阵模块

from sklearn import clone  # 导入 sklearn 中的 clone 函数
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder  # 导入 sklearn 中的数据预处理模块 KBinsDiscretizer 和 OneHotEncoder
from sklearn.utils._testing import (  # 导入 sklearn 内部测试工具中的多个断言函数
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
)

X = [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]

@pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器，定义参数化测试
    "strategy, expected, sample_weight",  # 参数化的参数和期望值
    [
        ("uniform", [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 1], [2, 2, 2, 2]], None),  # 参数化测试用例：uniform 策略
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], None),  # 参数化测试用例：kmeans 策略
        ("quantile", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], None),  # 参数化测试用例：quantile 策略
        ("quantile", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], [1, 1, 2, 1]),  # 参数化测试用例：quantile 策略，带样本权重
        ("quantile", [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], [1, 1, 1, 1]),  # 参数化测试用例：quantile 策略，带相同样本权重
        ("quantile", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], [0, 1, 1, 1]),  # 参数化测试用例：quantile 策略，带不同样本权重
        ("kmeans", [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 2]], [1, 0, 3, 1]),  # 参数化测试用例：kmeans 策略，带样本权重
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], [1, 1, 1, 1]),  # 参数化测试用例：kmeans 策略，带相同样本权重
    ],
)
def test_fit_transform(strategy, expected, sample_weight):
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)  # 初始化 KBinsDiscretizer 对象
    est.fit(X, sample_weight=sample_weight)  # 对数据 X 进行拟合，支持样本权重
    assert_array_equal(expected, est.transform(X))  # 断言预期输出与实际输出是否相等


def test_valid_n_bins():
    KBinsDiscretizer(n_bins=2).fit_transform(X)  # 测试 n_bins 参数为整数的情况
    KBinsDiscretizer(n_bins=np.array([2])[0]).fit_transform(X)  # 测试 n_bins 参数为 NumPy 数组的情况
    assert KBinsDiscretizer(n_bins=2).fit(X).n_bins_.dtype == np.dtype(int)  # 断言 n_bins_ 的数据类型为整数


@pytest.mark.parametrize("strategy", ["uniform"])  # 参数化测试，仅使用 uniform 策略
def test_kbinsdiscretizer_wrong_strategy_with_weights(strategy):
    """Check that we raise an error when the wrong strategy is used."""
    sample_weight = np.ones(shape=(len(X)))  # 创建全为 1 的样本权重数组
    est = KBinsDiscretizer(n_bins=3, strategy=strategy)  # 初始化 KBinsDiscretizer 对象
    err_msg = (
        "`sample_weight` was provided but it cannot be used with strategy='uniform'."
    )  # 错误消息字符串
    with pytest.raises(ValueError, match=err_msg):  # 使用 pytest 断言引发 ValueError 异常，且异常消息匹配指定字符串
        est.fit(X, sample_weight=sample_weight)  # 对数据 X 进行拟合，传入样本权重


def test_invalid_n_bins_array():
    # Bad shape
    n_bins = np.full((2, 4), 2.0)  # 创建形状不正确的 n_bins 数组
    est = KBinsDiscretizer(n_bins=n_bins)  # 初始化 KBinsDiscretizer 对象
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."  # 错误消息字符串
    with pytest.raises(ValueError, match=err_msg):  # 使用 pytest 断言引发 ValueError 异常，且异常消息匹配指定字符串
        est.fit_transform(X)  # 对数据 X 进行拟合

    # Incorrect number of features
    n_bins = [1, 2, 2]  # 创建数量不匹配的 n_bins 列表
    est = KBinsDiscretizer(n_bins=n_bins)  # 初始化 KBinsDiscretizer 对象
    err_msg = r"n_bins must be a scalar or array of shape \(n_features,\)."  # 错误消息字符串
    with pytest.raises(ValueError, match=err_msg):  # 使用 pytest 断言引发 ValueError 异常，且异常消息匹配指定字符串
        est.fit_transform(X)  # 对数据 X 进行拟合

    # Bad bin values
    n_bins = [1, 2, 2, 1]  # 创建包含非法 bin 值的 n_bins 列表
    # 创建一个 KBinsDiscretizer 对象，指定分箱的数量为 n_bins
    est = KBinsDiscretizer(n_bins=n_bins)
    
    # 设置错误消息，用于验证异常情况时的匹配信息
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 3. Number of bins must be at least 2, "
        "and must be an int."
    )
    
    # 使用 pytest 的 raises 方法检查是否会抛出 ValueError 异常，并验证异常消息是否匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 对给定的数据 X 进行拟合和转换，此处预期会抛出异常
        est.fit_transform(X)
    
    # Float 类型的分箱值
    n_bins = [2.1, 2, 2.1, 2]
    
    # 创建另一个 KBinsDiscretizer 对象，使用浮点数作为分箱的数量
    est = KBinsDiscretizer(n_bins=n_bins)
    
    # 更新错误消息，针对浮点数分箱值的异常情况
    err_msg = (
        "KBinsDiscretizer received an invalid number of bins "
        "at indices 0, 2. Number of bins must be at least 2, "
        "and must be an int."
    )
    
    # 使用 pytest 的 raises 方法再次检查是否会抛出 ValueError 异常，并验证异常消息是否匹配更新后的 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 对给定的数据 X 进行拟合和转换，此处预期会抛出异常
        est.fit_transform(X)
# 使用 pytest 的 parametrize 装饰器为 test_fit_transform_n_bins_array 函数提供多组参数化测试数据
@pytest.mark.parametrize(
    "strategy, expected, sample_weight",
    [
        ("uniform", [[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]], None),
        ("kmeans", [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2]], None),
        ("quantile", [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]], None),
        (
            "quantile",
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]],
            [1, 1, 3, 1],
        ),
        (
            "quantile",
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
            [0, 1, 3, 1],
        ),
        # 下面是一个被注释掉的测试用例，用于说明一个问题，这个问题已经在 GitHub 上有相关的讨论和记录
        # TODO: This test case above aims to test if the case where an array of
        #       ones passed in sample_weight parameter is equal to the case when
        #       sample_weight is None.
        #       Unfortunately, the behavior of `_weighted_percentile` when
        #       `sample_weight = [1, 1, 1, 1]` are currently not equivalent.
        #       This problem has been addressed in issue :
        #       https://github.com/scikit-learn/scikit-learn/issues/17370
        (
            "kmeans",
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 2]],
            [1, 0, 3, 1],
        ),
    ],
)
def test_fit_transform_n_bins_array(strategy, expected, sample_weight):
    # 使用 KBinsDiscretizer 对象进行初始化，设置参数：分箱数、编码方式和策略
    est = KBinsDiscretizer(
        n_bins=[2, 3, 3, 3], encode="ordinal", strategy=strategy
    ).fit(X, sample_weight=sample_weight)
    # 断言转换后的结果与期望结果相等
    assert_array_equal(expected, est.transform(X))

    # 测试 bin_edges_ 的形状
    n_features = np.array(X).shape[1]
    assert est.bin_edges_.shape == (n_features,)
    # 遍历每个特征的 bin_edges_，断言其形状为 (n_bins + 1,)
    for bin_edges, n_bins in zip(est.bin_edges_, est.n_bins_):
        assert bin_edges.shape == (n_bins + 1,)


# 使用 pytest 的 filterwarnings 装饰器忽略特定的警告信息
@pytest.mark.filterwarnings("ignore: Bins whose width are too small")
def test_kbinsdiscretizer_effect_sample_weight():
    """Check the impact of `sample_weight` on computed quantiles."""
    X = np.array([[-2], [-1], [1], [3], [500], [1000]])
    # 设置 KBinsDiscretizer 对象，定义参数：分箱数、编码方式和策略
    # 使用 sample_weight=[1, 1, 1, 1, 0, 0] 来训练模型
    est = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    est.fit(X, sample_weight=[1, 1, 1, 1, 0, 0])
    # 断言第一个特征的 bin_edges_ 与预期值相近
    assert_allclose(est.bin_edges_[0], [-2, -1, 1, 3])
    # 断言转换后的结果与预期值相等
    assert_allclose(est.transform(X), [[0.0], [1.0], [2.0], [2.0], [2.0], [2.0]])


# 使用 pytest 的 parametrize 装饰器为 test_kbinsdiscretizer_no_mutating_sample_weight 函数提供多组参数化测试数据
@pytest.mark.parametrize("strategy", ["kmeans", "quantile"])
def test_kbinsdiscretizer_no_mutating_sample_weight(strategy):
    """Make sure that `sample_weight` is not changed in place."""
    # 初始化 KBinsDiscretizer 对象，设置参数：分箱数、编码方式和策略
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy=strategy)
    # 创建样本权重数组，并进行深拷贝备份
    sample_weight = np.array([1, 3, 1, 2], dtype=np.float64)
    sample_weight_copy = np.copy(sample_weight)
    # 使用样本权重来训练模型
    est.fit(X, sample_weight=sample_weight)
    # 断言样本权重未被修改
    assert_allclose(sample_weight, sample_weight_copy)
@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
# 参数化测试，对于每种策略分别运行测试
def test_same_min_max(strategy):
    warnings.simplefilter("always")
    # 设置警告过滤器，始终显示警告
    X = np.array([[1, -2], [1, -1], [1, 0], [1, 1]])
    # 创建一个包含特征值的 NumPy 数组
    est = KBinsDiscretizer(strategy=strategy, n_bins=3, encode="ordinal")
    # 初始化 KBinsDiscretizer 类，指定策略、箱数和编码方式
    warning_message = "Feature 0 is constant and will be replaced with 0."
    # 设置预期的警告消息
    with pytest.warns(UserWarning, match=warning_message):
        # 检查是否会引发特定警告消息
        est.fit(X)
    assert est.n_bins_[0] == 1
    # 断言第一个特征的箱数为1
    # 替换特征为零
    Xt = est.transform(X)
    # 对数据集进行转换
    assert_array_equal(Xt[:, 0], np.zeros(X.shape[0]))


def test_transform_1d_behavior():
    X = np.arange(4)
    # 创建一个包含连续数字的 NumPy 数组
    est = KBinsDiscretizer(n_bins=2)
    # 初始化 KBinsDiscretizer 类，指定箱数
    with pytest.raises(ValueError):
        # 检查是否引发 ValueError 异常
        est.fit(X)

    est = KBinsDiscretizer(n_bins=2)
    # 初始化 KBinsDiscretizer 类，指定箱数
    est.fit(X.reshape(-1, 1))
    # 对数据进行reshape以符合模型的输入要求
    with pytest.raises(ValueError):
        # 检查是否引发 ValueError 异常
        est.transform(X)


@pytest.mark.parametrize("i", range(1, 9))
# 参数化测试，对于每个 i 在范围1到8内分别运行测试
def test_numeric_stability(i):
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    # 创建一个初始数据集，将其reshape以符合模型的输入要求
    Xt_expected = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)

    # Test up to discretizing nano units
    # 测试直到将数据离散成纳米单位
    X = X_init / 10**i
    # 对初始数据集进行归一化处理
    Xt = KBinsDiscretizer(n_bins=2, encode="ordinal").fit_transform(X)
    # 初始化 KBinsDiscretizer 类，指定箱数和编码方式，并对数据进行转换
    assert_array_equal(Xt_expected, Xt)


def test_encode_options():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="ordinal").fit(X)
    # 初始化 KBinsDiscretizer 类，指定箱数和编码方式，并对数据进行拟合
    Xt_1 = est.transform(X)
    # 对数据进行转换
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot-dense").fit(X)
    # 初始化 KBinsDiscretizer 类，指定箱数和编码方式，并对数据进行拟合
    Xt_2 = est.transform(X)
    # 对数据进行转换
    assert not sp.issparse(Xt_2)
    # 断言输出的矩阵不是稀疏矩阵
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=False
        ).fit_transform(Xt_1),
        Xt_2,
    )
    # 断言独热编码器对输出进行了适当的变换
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode="onehot").fit(X)
    # 初始化 KBinsDiscretizer 类，指定箱数和编码方式，并对数据进行拟合
    Xt_3 = est.transform(X)
    # 对数据进行转换
    assert sp.issparse(Xt_3)
    # 断言输出的矩阵是稀疏矩阵
    assert_array_equal(
        OneHotEncoder(
            categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=True
        )
        .fit_transform(Xt_1)
        .toarray(),
        Xt_3.toarray(),
    )


@pytest.mark.parametrize(
    "strategy, expected_2bins, expected_3bins, expected_5bins",
    [
        ("uniform", [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 2, 2], [0, 0, 1, 1, 4, 4]),
        ("kmeans", [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 3, 4]),
        ("quantile", [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 4]),
    ],
)
# 参数化测试，对于每种策略分别运行测试，每个测试包含多个预期的输出
def test_nonuniform_strategies(
    strategy, expected_2bins, expected_3bins, expected_5bins
):
    X = np.array([0, 0.5, 2, 3, 9, 10]).reshape(-1, 1)

    # with 2 bins
    est = KBinsDiscretizer(n_bins=2, strategy=strategy, encode="ordinal")
    # 初始化 KBinsDiscretizer 类，指定箱数、策略和编码方式
    Xt = est.fit_transform(X)
    # 对数据进行拟合和转换
    assert_array_equal(expected_2bins, Xt.ravel())
    # 断言实际输出与预期输出一致

    # with 3 bins
    est = KBinsDiscretizer(n_bins=3, strategy=strategy, encode="ordinal")
    # 初始化 KBinsDiscretizer 类，指定箱数、策略和编码方式
    Xt = est.fit_transform(X)
    # 对数据进行拟合和转换
    assert_array_equal(expected_3bins, Xt.ravel())
    # 断言实际输出与预期输出一致

    # with 5 bins
    est = KBinsDiscretizer(n_bins=5, strategy=strategy, encode="ordinal")
    # 初始化 KBinsDiscretizer 类，指定箱数、策略和编码方式
    Xt = est.fit_transform(X)
    # 对数据进行拟合和转换
    assert_array_equal(expected_5bins, Xt.ravel())
    # 断言实际输出与预期输出一致
    # 使用断言检查预期结果列表 expected_5bins 与变量 Xt 扁平化后的值是否相等
    assert_array_equal(expected_5bins, Xt.ravel())
@pytest.mark.parametrize(
    "strategy, expected_inv",
    [  # 参数化测试参数，包括三种策略和预期的逆转换结果
        (
            "uniform",
            [  # 使用均匀策略时的预期逆转换结果列表
                [-1.5, 2.0, -3.5, -0.5],
                [-0.5, 3.0, -2.5, -0.5],
                [0.5, 4.0, -1.5, 0.5],
                [0.5, 4.0, -1.5, 1.5],
            ],
        ),
        (
            "kmeans",
            [  # 使用k均值策略时的预期逆转换结果列表
                [-1.375, 2.125, -3.375, -0.5625],
                [-1.375, 2.125, -3.375, -0.5625],
                [-0.125, 3.375, -2.125, 0.5625],
                [0.75, 4.25, -1.25, 1.625],
            ],
        ),
        (
            "quantile",
            [  # 使用分位数策略时的预期逆转换结果列表
                [-1.5, 2.0, -3.5, -0.75],
                [-0.5, 3.0, -2.5, 0.0],
                [0.5, 4.0, -1.5, 1.25],
                [0.5, 4.0, -1.5, 1.25],
            ],
        ),
    ],
)
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_inverse_transform(strategy, encode, expected_inv):
    # 创建KBinsDiscretizer对象，指定分箱数为3，策略和编码方式为参数化提供的值
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy, encode=encode)
    # 对数据集X进行拟合转换
    Xt = kbd.fit_transform(X)
    # 对转换后的数据进行逆转换
    Xinv = kbd.inverse_transform(Xt)
    # 断言逆转换结果与预期结果相等
    assert_array_almost_equal(expected_inv, Xinv)


@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
def test_transform_outside_fit_range(strategy):
    # 创建包含单列数据的数组X
    X = np.array([0, 1, 2, 3])[:, None]
    # 创建KBinsDiscretizer对象，指定分箱数为4，策略为参数化提供的值，编码方式为ordinal
    kbd = KBinsDiscretizer(n_bins=4, strategy=strategy, encode="ordinal")
    # 对数据集X进行拟合
    kbd.fit(X)

    # 创建新的单列数据集X2
    X2 = np.array([-2, 5])[:, None]
    # 对数据集X2进行转换
    X2t = kbd.transform(X2)
    # 断言转换后的数据集每列的最大值加1等于分箱数
    assert_array_equal(X2t.max(axis=0) + 1, kbd.n_bins_)
    # 断言转换后的数据集每列的最小值等于0
    assert_array_equal(X2t.min(axis=0), [0])


def test_overwrite():
    # 创建包含单列数据的数组X
    X = np.array([0, 1, 2, 3])[:, None]
    # 备份原始数据集X
    X_before = X.copy()

    # 创建KBinsDiscretizer对象，指定分箱数为3，编码方式为ordinal
    est = KBinsDiscretizer(n_bins=3, encode="ordinal")
    # 对数据集X进行拟合转换
    Xt = est.fit_transform(X)
    # 断言拟合转换前后数据集X保持不变
    assert_array_equal(X, X_before)

    # 备份转换后的数据集Xt
    Xt_before = Xt.copy()
    # 对转换后的数据集进行逆转换
    Xinv = est.inverse_transform(Xt)
    # 断言逆转换前后数据集Xt保持不变
    assert_array_equal(Xt, Xt_before)
    # 断言逆转换结果与预期结果相等
    assert_array_equal(Xinv, np.array([[0.5], [1.5], [2.5], [2.5]]))


@pytest.mark.parametrize(
    "strategy, expected_bin_edges", [("quantile", [0, 1, 3]), ("kmeans", [0, 1.5, 3])]
)
def test_redundant_bins(strategy, expected_bin_edges):
    # 创建包含单列数据的列表X
    X = [[0], [0], [0], [0], [3], [3]]
    # 创建KBinsDiscretizer对象，指定分箱数为3，策略为参数化提供的值，子样本为空
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy, subsample=None)
    # 准备警告消息
    warning_message = "Consider decreasing the number of bins."
    # 使用pytest的warns断言捕获UserWarning并匹配警告消息
    with pytest.warns(UserWarning, match=warning_message):
        # 对数据集X进行拟合
        kbd.fit(X)
    # 断言分箱边界与预期边界相等
    assert_array_almost_equal(kbd.bin_edges_[0], expected_bin_edges)


def test_percentile_numeric_stability():
    # 创建包含单列数据的数组X
    X = np.array([0.05, 0.05, 0.95]).reshape(-1, 1)
    # 准备分箱边界数组
    bin_edges = np.array([0.05, 0.23, 0.41, 0.59, 0.77, 0.95])
    # 准备转换后的数据集Xt
    Xt = np.array([0, 0, 4]).reshape(-1, 1)
    # 创建KBinsDiscretizer对象，指定分箱数为10，编码方式为ordinal，策略为quantile
    kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    # 准备警告消息
    warning_message = "Consider decreasing the number of bins."
    # 使用pytest的warns断言捕获UserWarning并匹配警告消息
    with pytest.warns(UserWarning, match=warning_message):
        # 对数据集X进行拟合
        kbd.fit(X)

    # 断言分箱边界与预期边界相等
    assert_array_almost_equal(kbd.bin_edges_[0], bin_edges)
    # 断言转换后的数据集与预期结果相等
    assert_array_almost_equal(kbd.transform(X), Xt)
@pytest.mark.parametrize("in_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("out_dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_consistent_dtype(in_dtype, out_dtype, encode):
    # 创建一个测试函数，用于验证 KBinsDiscretizer 的数据类型一致性
    X_input = np.array(X, dtype=in_dtype)  # 使用给定的输入数据类型创建 NumPy 数组 X_input
    kbd = KBinsDiscretizer(n_bins=3, encode=encode, dtype=out_dtype)  # 创建 KBinsDiscretizer 对象 kbd
    kbd.fit(X_input)  # 对输入数据进行拟合

    # 测试输出数据类型的一致性
    if out_dtype is not None:
        expected_dtype = out_dtype  # 如果指定了输出数据类型，则使用指定的类型作为预期数据类型
    elif out_dtype is None and X_input.dtype == np.float16:
        # 如果未指定输出数据类型且输入数据类型为 np.float16，则预期输出数据类型为 np.float64
        expected_dtype = np.float64
    else:
        expected_dtype = X_input.dtype  # 其他情况下，预期输出数据类型与输入数据类型一致
    Xt = kbd.transform(X_input)  # 对输入数据进行转换
    assert Xt.dtype == expected_dtype  # 断言转换后的数据类型与预期一致


@pytest.mark.parametrize("input_dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("encode", ["ordinal", "onehot", "onehot-dense"])
def test_32_equal_64(input_dtype, encode):
    # TODO this check is redundant with common checks and can be removed
    #  once #16290 is merged
    X_input = np.array(X, dtype=input_dtype)  # 使用给定的输入数据类型创建 NumPy 数组 X_input

    # 创建 KBinsDiscretizer 对象，输出数据类型为 np.float32
    kbd_32 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float32)
    kbd_32.fit(X_input)  # 对输入数据进行拟合
    Xt_32 = kbd_32.transform(X_input)  # 对输入数据进行转换

    # 创建 KBinsDiscretizer 对象，输出数据类型为 np.float64
    kbd_64 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float64)
    kbd_64.fit(X_input)  # 对输入数据进行拟合
    Xt_64 = kbd_64.transform(X_input)  # 对输入数据进行转换

    assert_allclose_dense_sparse(Xt_32, Xt_64)  # 断言两个转换结果的密集表示和稀疏表示接近


def test_kbinsdiscretizer_subsample_default():
    # 由于 X 的大小较小（< 2e5），不会进行子采样
    X = np.array([-2, 1.5, -4, -1]).reshape(-1, 1)
    kbd_default = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    kbd_default.fit(X)  # 对输入数据进行拟合

    kbd_without_subsampling = clone(kbd_default)
    kbd_without_subsampling.set_params(subsample=None)
    kbd_without_subsampling.fit(X)  # 对输入数据进行拟合（无子采样设置）

    for bin_kbd_default, bin_kbd_with_subsampling in zip(
        kbd_default.bin_edges_[0], kbd_without_subsampling.bin_edges_[0]
    ):
        np.testing.assert_allclose(bin_kbd_default, bin_kbd_with_subsampling)
    assert kbd_default.bin_edges_.shape == kbd_without_subsampling.bin_edges_.shape


@pytest.mark.parametrize(
    "encode, expected_names",
    [
        (
            "onehot",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        (
            "onehot-dense",
            [
                f"feat{col_id}_{float(bin_id)}"
                for col_id in range(3)
                for bin_id in range(4)
            ],
        ),
        ("ordinal", [f"feat{col_id}" for col_id in range(3)]),
    ],
)
def test_kbinsdiscrtizer_get_feature_names_out(encode, expected_names):
    """Check get_feature_names_out for different settings.
    Non-regression test for #22731
    """
    X = [[-2, 1, -4], [-1, 2, -3], [0, 3, -2], [1, 4, -1]]
    # 测试 get_feature_names_out 方法在不同设置下的输出结果是否正确
    # 使用 KBinsDiscretizer 对象对输入数据 X 进行离散化处理，分成4个区间，并进行拟合
    kbd = KBinsDiscretizer(n_bins=4, encode=encode).fit(X)
    # 将输入数据 X 进行离散化转换，得到转换后的数据 Xt
    Xt = kbd.transform(X)

    # 创建包含3个特征名称的列表，用于获取特征转换后的输出名称
    input_features = [f"feat{i}" for i in range(3)]
    # 获取特征转换后的输出名称
    output_names = kbd.get_feature_names_out(input_features)
    # 断言转换后的数据 Xt 的列数与输出名称数组的长度相等
    assert Xt.shape[1] == output_names.shape[0]

    # 使用断言检查转换后的输出名称是否与预期的名称数组 expected_names 相等
    assert_array_equal(output_names, expected_names)
# 使用 pytest.mark.parametrize 装饰器来定义参数化测试，测试不同的策略：uniform、kmeans、quantile
@pytest.mark.parametrize("strategy", ["uniform", "kmeans", "quantile"])
# 定义测试函数 test_kbinsdiscretizer_subsample，验证在使用子采样时分箱边界几乎相同
def test_kbinsdiscretizer_subsample(strategy, global_random_seed):
    # 生成一个形状为 (100000, 1) 的随机数数组 X，使用给定的全局随机种子 global_random_seed
    X = np.random.RandomState(global_random_seed).random_sample((100000, 1)) + 1

    # 使用 KBinsDiscretizer 创建实例 kbd_subsampling，使用给定的策略和子采样数目 50000
    kbd_subsampling = KBinsDiscretizer(
        strategy=strategy, subsample=50000, random_state=global_random_seed
    )
    # 对 X 进行拟合
    kbd_subsampling.fit(X)

    # 克隆 kbd_subsampling 为 kbd_no_subsampling，并将其 subsample 参数设置为 None
    kbd_no_subsampling = clone(kbd_subsampling)
    kbd_no_subsampling.set_params(subsample=None)
    # 对 X 进行拟合
    kbd_no_subsampling.fit(X)

    # 使用较大的容差值，因为在使用子采样时不能期望分箱边界完全相同
    assert_allclose(
        kbd_subsampling.bin_edges_[0], kbd_no_subsampling.bin_edges_[0], rtol=1e-2
    )


# TODO(1.7): remove this test
# 定义测试函数 test_KBD_inverse_transform_Xt_deprecation，测试逆变换方法在使用过程中的警告和异常情况
def test_KBD_inverse_transform_Xt_deprecation():
    # 创建一个形状为 (10, 1) 的数组 X，包含 0 到 9 的整数
    X = np.arange(10)[:, None]
    # 创建 KBinsDiscretizer 实例 kbd
    kbd = KBinsDiscretizer()
    # 对 X 进行拟合和转换
    X = kbd.fit_transform(X)

    # 验证在没有提供所需位置参数时是否引发 TypeError 异常
    with pytest.raises(TypeError, match="Missing required positional argument"):
        kbd.inverse_transform()

    # 验证在同时使用 X 和 Xt 时是否引发 TypeError 异常
    with pytest.raises(TypeError, match="Cannot use both X and Xt. Use X only"):
        kbd.inverse_transform(X=X, Xt=X)

    # 验证在使用 warnings 模块捕获异常并转换为错误时是否正常工作
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        kbd.inverse_transform(X)

    # 验证在使用过时警告时是否发出 FutureWarning
    with pytest.warns(FutureWarning, match="Xt was renamed X in version 1.5"):
        kbd.inverse_transform(Xt=X)
```