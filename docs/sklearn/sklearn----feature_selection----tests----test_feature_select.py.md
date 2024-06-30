# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_feature_select.py`

```
"""
Todo: cross-check the F-value with stats model
"""

import itertools  # 导入 itertools 模块，用于高效循环和迭代操作
import warnings  # 导入 warnings 模块，用于处理警告信息

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库，用于单元测试
from numpy.testing import assert_allclose  # 导入 NumPy 测试模块中的断言函数
from scipy import sparse, stats  # 导入 SciPy 库中的 sparse 和 stats 模块

from sklearn.datasets import load_iris, make_classification, make_regression  # 导入用于生成数据集的函数
from sklearn.feature_selection import (  # 导入特征选择相关的类和函数
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_oneway,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.utils import safe_mask  # 导入用于安全掩码的工具函数
from sklearn.utils._testing import (  # 导入用于测试的私有测试函数
    _convert_container,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入修复函数，用于处理 CSR 容器的问题

##############################################################################
# Test the score functions


def test_f_oneway_vs_scipy_stats():
    # Test that our f_oneway gives the same result as scipy.stats
    rng = np.random.RandomState(0)  # 创建随机数生成器对象 rng
    X1 = rng.randn(10, 3)  # 生成一个 10x3 的随机数组 X1
    X2 = 1 + rng.randn(10, 3)  # 生成一个偏移后的 10x3 的随机数组 X2
    f, pv = stats.f_oneway(X1, X2)  # 使用 scipy.stats 中的 f_oneway 计算 F 值和 p 值
    f2, pv2 = f_oneway(X1, X2)  # 使用 sklearn 中的 f_oneway 计算 F 值和 p 值
    assert np.allclose(f, f2)  # 断言两个 F 值近似相等
    assert np.allclose(pv, pv2)  # 断言两个 p 值近似相等


def test_f_oneway_ints():
    # Smoke test f_oneway on integers: that it does raise casting errors
    # with recent numpys
    rng = np.random.RandomState(0)  # 创建随机数生成器对象 rng
    X = rng.randint(10, size=(10, 10))  # 生成一个 10x10 的随机整数数组 X
    y = np.arange(10)  # 创建一个长度为 10 的整数数组 y
    fint, pint = f_oneway(X, y)  # 使用 f_oneway 对 X 和 y 进行方差分析

    # test that is gives the same result as with float
    f, p = f_oneway(X.astype(float), y)  # 将 X 转换为浮点数后再次使用 f_oneway 进行方差分析
    assert_array_almost_equal(f, fint, decimal=4)  # 断言两个 F 值近似相等
    assert_array_almost_equal(p, pint, decimal=4)  # 断言两个 p 值近似相等


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_f_classif(csr_container):
    # Test whether the F test yields meaningful results
    # on a simple simulated classification problem
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    F, pv = f_classif(X, y)  # 使用 f_classif 进行特征选择和 F 检验
    F_sparse, pv_sparse = f_classif(csr_container(X), y)  # 对 CSR 容器的 X 进行特征选择和 F 检验
    assert (F > 0).all()  # 断言所有的 F 值都大于 0
    assert (pv > 0).all()  # 断言所有的 p 值都大于 0
    assert (pv < 1).all()  # 断言所有的 p 值都小于 1
    assert (pv[:5] < 0.05).all()  # 断言前 5 个 p 值小于 0.05
    assert (pv[5:] > 1.0e-4).all()  # 断言后面的 p 值大于 0.0001
    assert_array_almost_equal(F_sparse, F)  # 断言稀疏矩阵版本的 F 值与普通版本近似相等
    assert_array_almost_equal(pv_sparse, pv)  # 断言稀疏矩阵版本的 p 值与普通版本近似相等


@pytest.mark.parametrize("center", [True, False])
def test_r_regression(center):
    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=5, shuffle=False, random_state=0
    )

    corr_coeffs = r_regression(X, y, center=center)  # 使用 r_regression 进行回归系数计算
    assert (-1 < corr_coeffs).all()  # 断言所有回归系数大于 -1
    assert (corr_coeffs < 1).all()  # 断言所有回归系数小于 1

    sparse_X = _convert_container(X, "sparse")  # 将 X 转换为稀疏格式

    sparse_corr_coeffs = r_regression(sparse_X, y, center=center)  # 对稀疏矩阵版本的 X 进行回归系数计算
    assert_allclose(sparse_corr_coeffs, corr_coeffs)  # 断言稀疏矩阵版本的回归系数与普通版本近似相等
    # 对 numpy 进行测试，作为参考基准
    Z = np.hstack((X, y[:, np.newaxis]))
    # 计算 Z 的相关系数矩阵，其中 rowvar=False 表示 Z 的每一列是一个变量
    correlation_matrix = np.corrcoef(Z, rowvar=False)
    # 提取出相关系数矩阵中除了最后一列外的所有行（即 X 和 y 之间的相关系数）
    np_corr_coeffs = correlation_matrix[:-1, -1]
    # 断言 numpy 计算得到的相关系数与预期的 corr_coeffs 非常接近，精度为小数点后三位
    assert_array_almost_equal(np_corr_coeffs, corr_coeffs, decimal=3)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用pytest的parametrize装饰器，为csr_container参数进行参数化测试

def test_f_regression(csr_container):
    # 测试F检验在简单的模拟回归问题上是否产生有意义的结果

    # 创建一个简单的模拟回归问题数据集
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0
    )

    # 进行F检验，计算F统计量和对应的p值
    F, pv = f_regression(X, y)
    # 断言F统计量的所有值大于0
    assert (F > 0).all()
    # 断言p值的所有值大于0
    assert (pv > 0).all()
    # 断言p值的所有值小于1
    assert (pv < 1).all()
    # 断言前5个p值小于0.05
    assert (pv[:5] < 0.05).all()
    # 断言第5个之后的p值大于1.0e-4
    assert (pv[5:] > 1.0e-4).all()

    # 使用中心化参数进行比较，与稀疏矩阵进行比较
    F, pv = f_regression(X, y, center=True)
    F_sparse, pv_sparse = f_regression(csr_container(X), y, center=True)
    # 断言稀疏矩阵得到的F统计量与非稀疏情况下的一致
    assert_allclose(F_sparse, F)
    # 断言稀疏矩阵得到的p值与非稀疏情况下的一致
    assert_allclose(pv_sparse, pv)

    # 再次使用非中心化参数进行比较，与稀疏矩阵进行比较
    F, pv = f_regression(X, y, center=False)
    F_sparse, pv_sparse = f_regression(csr_container(X), y, center=False)
    # 断言稀疏矩阵得到的F统计量与非稀疏情况下的一致
    assert_allclose(F_sparse, F)
    # 断言稀疏矩阵得到的p值与非稀疏情况下的一致
    assert_allclose(pv_sparse, pv)


def test_f_regression_input_dtype():
    # 测试f_regression对于任何数值数据类型是否返回相同的值

    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    y = np.arange(10).astype(int)

    # 测试不同数据类型的X对应的F统计量和p值是否相同
    F1, pv1 = f_regression(X, y)
    F2, pv2 = f_regression(X, y.astype(float))
    # 断言不同数据类型下的F统计量近似相等
    assert_allclose(F1, F2, 5)
    # 断言不同数据类型下的p值近似相等
    assert_allclose(pv1, pv2, 5)


def test_f_regression_center():
    # 测试f_regression根据'center'参数是否保留自由度

    # 创建一个简单的示例数据集，使得X具有零均值
    X = np.arange(-5, 6).reshape(-1, 1)
    n_samples = X.size
    Y = np.ones(n_samples)
    Y[::2] *= -1.0
    Y[0] = 0.0  # 使得Y的均值为零

    # 测试使用中心化和非中心化参数计算的F统计量是否保持一致
    F1, _ = f_regression(X, Y, center=True)
    F2, _ = f_regression(X, Y, center=False)
    # 断言在不使用中心化时，F统计量乘以自由度调整因子与使用中心化时的F统计量近似相等
    assert_allclose(F1 * (n_samples - 1.0) / (n_samples - 2.0), F2)
    # 使用statsmodels OLS得到的值，与第一个F统计量的特定值进行比较
    assert_almost_equal(F2[0], 0.232558139)  # value from statsmodels OLS


@pytest.mark.parametrize(
    "X, y, expected_corr_coef, force_finite",
    [
        (
            # 特征 X 中的一个特征是常数 - 强制有限
            np.array([[2, 1], [2, 0], [2, 10], [2, 4]]),
            # 目标 y 是常数 - 强制有限
            np.array([0, 1, 1, 0]),
            # 样本权重
            np.array([0.0, 0.32075]),
            # 是否强制有限
            True,
        ),
        (
            # 目标 y 是常数 - 强制有限
            np.array([[5, 1], [3, 0], [2, 10], [8, 4]]),
            # 目标 y 是常数 - 强制有限
            np.array([0, 0, 0, 0]),
            # 样本权重
            np.array([0.0, 0.0]),
            # 是否强制有限
            True,
        ),
        (
            # 特征 X 中的一个特征是常数 - 不强制有限
            np.array([[2, 1], [2, 0], [2, 10], [2, 4]]),
            # 目标 y 是常数 - 不强制有限
            np.array([0, 1, 1, 0]),
            # 样本权重
            np.array([np.nan, 0.32075]),
            # 是否强制有限
            False,
        ),
        (
            # 目标 y 是常数 - 不强制有限
            np.array([[5, 1], [3, 0], [2, 10], [8, 4]]),
            # 目标 y 是常数 - 不强制有限
            np.array([0, 0, 0, 0]),
            # 样本权重
            np.array([np.nan, np.nan]),
            # 是否强制有限
            False,
        ),
    ],
# 定义一个测试函数，用于测试在r_regression函数中使用force_finite参数的行为，尤其是对一些边界情况的处理。
def test_r_regression_force_finite(X, y, expected_corr_coef, force_finite):
    """Check the behaviour of `force_finite` for some corner cases with `r_regression`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15672
    """
    # 捕获 RuntimeWarning 类型的警告，并将其作为异常处理
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # 调用 r_regression 函数，传入参数 X, y, force_finite，并获取相关系数
        corr_coef = r_regression(X, y, force_finite=force_finite)
    # 使用 NumPy 测试工具断言 corr_coef 与期望的相关系数几乎相等
    np.testing.assert_array_almost_equal(corr_coef, expected_corr_coef)

# 使用 pytest 的参数化装饰器，对以下多组输入进行参数化测试
@pytest.mark.parametrize(
    "X, y, expected_f_statistic, expected_p_values, force_finite",
    [
        (
            # 当 X 中的一个特征是常数时，强制为有限值
            np.array([[2, 1], [2, 0], [2, 10], [2, 4]]),
            np.array([0, 1, 1, 0]),
            np.array([0.0, 0.2293578]),
            np.array([1.0, 0.67924985]),
            True,
        ),
        (
            # 目标 y 是常数时，强制为有限值
            np.array([[5, 1], [3, 0], [2, 10], [8, 4]]),
            np.array([0, 0, 0, 0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            True,
        ),
        (
            # X 中的特征与 y 相关时，强制为有限值
            np.array([[0, 1], [1, 0], [2, 10], [3, 4]]),
            np.array([0, 1, 2, 3]),
            np.array([np.finfo(np.float64).max, 0.845433]),
            np.array([0.0, 0.454913]),
            True,
        ),
        (
            # X 中的特征与 y 反相关时，强制为有限值
            np.array([[3, 1], [2, 0], [1, 10], [0, 4]]),
            np.array([0, 1, 2, 3]),
            np.array([np.finfo(np.float64).max, 0.845433]),
            np.array([0.0, 0.454913]),
            True,
        ),
        (
            # 当 X 中的一个特征是常数时，不强制为有限值
            np.array([[2, 1], [2, 0], [2, 10], [2, 4]]),
            np.array([0, 1, 1, 0]),
            np.array([np.nan, 0.2293578]),
            np.array([np.nan, 0.67924985]),
            False,
        ),
        (
            # 目标 y 是常数时，不强制为有限值
            np.array([[5, 1], [3, 0], [2, 10], [8, 4]]),
            np.array([0, 0, 0, 0]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            False,
        ),
        (
            # X 中的特征与 y 相关时，不强制为有限值
            np.array([[0, 1], [1, 0], [2, 10], [3, 4]]),
            np.array([0, 1, 2, 3]),
            np.array([np.inf, 0.845433]),
            np.array([0.0, 0.454913]),
            False,
        ),
        (
            # X 中的特征与 y 反相关时，不强制为有限值
            np.array([[3, 1], [2, 0], [1, 10], [0, 4]]),
            np.array([0, 1, 2, 3]),
            np.array([np.inf, 0.845433]),
            np.array([0.0, 0.454913]),
            False,
        ),
    ],
)
# 定义测试函数，对 r_regression 函数在边界情况下的行为进行测试
def test_f_regression_corner_case(
    X, y, expected_f_statistic, expected_p_values, force_finite
):
    """Check the behaviour of `force_finite` for some corner cases with `f_regression`.
    
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15672
    """
    # 使用警告捕获器来捕获运行时警告
    with warnings.catch_warnings():
        # 设置简单过滤器，捕获 RuntimeWarning 类型的警告
        warnings.simplefilter("error", RuntimeWarning)
        # 调用 f_regression 函数，传入参数 X, y 和 force_finite，并获取返回的 f_statistic 和 p_values
        f_statistic, p_values = f_regression(X, y, force_finite=force_finite)
    # 使用 NumPy 的测试函数，断言 f_statistic 应近似等于预期的 f_statistic
    np.testing.assert_array_almost_equal(f_statistic, expected_f_statistic)
    # 使用 NumPy 的测试函数，断言 p_values 应近似等于预期的 p_values
    np.testing.assert_array_almost_equal(p_values, expected_p_values)
def test_f_classif_multi_class():
    # 测试 F 检验在简单的模拟分类问题上是否产生有意义的结果

    # 创建一个简单的模拟分类问题的数据集
    X, y = make_classification(
        n_samples=200,                  # 样本数
        n_features=20,                  # 特征数
        n_informative=3,                # 有信息特征数
        n_redundant=2,                  # 冗余特征数
        n_repeated=0,                   # 重复特征数
        n_classes=8,                    # 类别数
        n_clusters_per_class=1,         # 每个类别的簇数
        flip_y=0.0,                     # 标签反转概率
        class_sep=10,                   # 类别间距
        shuffle=False,                  # 是否打乱顺序
        random_state=0,                 # 随机种子
    )

    # 应用 F 检验计算特征的 F 值和 p 值
    F, pv = f_classif(X, y)

    # 断言所有的 F 值大于 0
    assert (F > 0).all()

    # 断言所有的 p 值大于 0
    assert (pv > 0).all()

    # 断言所有的 p 值小于 1
    assert (pv < 1).all()

    # 断言前五个特征的 p 值都小于 0.05
    assert (pv[:5] < 0.05).all()

    # 断言后面的特征的 p 值都大于 1.0e-4
    assert (pv[5:] > 1.0e-4).all()


def test_select_percentile_classif():
    # 测试相对单变量特征选择在简单分类问题中是否能正确选择特征，
    # 使用百分位数启发式方法

    # 创建一个简单的模拟分类问题的数据集
    X, y = make_classification(
        n_samples=200,                  # 样本数
        n_features=20,                  # 特征数
        n_informative=3,                # 有信息特征数
        n_redundant=2,                  # 冗余特征数
        n_repeated=0,                   # 重复特征数
        n_classes=8,                    # 类别数
        n_clusters_per_class=1,         # 每个类别的簇数
        flip_y=0.0,                     # 标签反转概率
        class_sep=10,                   # 类别间距
        shuffle=False,                  # 是否打乱顺序
        random_state=0,                 # 随机种子
    )

    # 使用选择百分位数的单变量特征选择器，基于 F 检验
    univariate_filter = SelectPercentile(f_classif, percentile=25)

    # 对特征进行选择和转换
    X_r = univariate_filter.fit(X, y).transform(X)

    # 使用通用的单变量特征选择器，同样基于百分位数
    X_r2 = (
        GenericUnivariateSelect(f_classif, mode="percentile", param=25)
        .fit(X, y)
        .transform(X)
    )

    # 断言两种方法得到的特征选择结果相等
    assert_array_equal(X_r, X_r2)

    # 获取特征选择器的支持向量
    support = univariate_filter.get_support()

    # 生成真实的支持向量标记
    gtruth = np.zeros(20)
    gtruth[:5] = 1

    # 断言支持向量与真实标记相等
    assert_array_equal(support, gtruth)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_select_percentile_classif_sparse(csr_container):
    # 测试相对单变量特征选择在简单分类问题中是否能正确选择特征，
    # 使用百分位数启发式方法，处理稀疏数据

    # 创建一个简单的模拟分类问题的数据集
    X, y = make_classification(
        n_samples=200,                  # 样本数
        n_features=20,                  # 特征数
        n_informative=3,                # 有信息特征数
        n_redundant=2,                  # 冗余特征数
        n_repeated=0,                   # 重复特征数
        n_classes=8,                    # 类别数
        n_clusters_per_class=1,         # 每个类别的簇数
        flip_y=0.0,                     # 标签反转概率
        class_sep=10,                   # 类别间距
        shuffle=False,                  # 是否打乱顺序
        random_state=0,                 # 随机种子
    )

    # 将数据转换为稀疏矩阵格式
    X = csr_container(X)

    # 使用选择百分位数的单变量特征选择器，基于 F 检验
    univariate_filter = SelectPercentile(f_classif, percentile=25)

    # 对特征进行选择和转换
    X_r = univariate_filter.fit(X, y).transform(X)

    # 使用通用的单变量特征选择器，同样基于百分位数
    X_r2 = (
        GenericUnivariateSelect(f_classif, mode="percentile", param=25)
        .fit(X, y)
        .transform(X)
    )

    # 断言两种方法得到的特征选择结果相等
    assert_array_equal(X_r.toarray(), X_r2.toarray())

    # 获取特征选择器的支持向量
    support = univariate_filter.get_support()

    # 生成真实的支持向量标记
    gtruth = np.zeros(20)
    gtruth[:5] = 1

    # 断言支持向量与真实标记相等
    assert_array_equal(support, gtruth)

    # 反转变换并检查稀疏性质
    X_r2inv = univariate_filter.inverse_transform(X_r2)
    assert sparse.issparse(X_r2inv)

    # 使用支持向量创建安全掩码
    support_mask = safe_mask(X_r2inv, support)

    # 断言反转变换后的形状与原始数据相同
    assert X_r2inv.shape == X.shape

    # 断言特定支持向量的稀疏数组值与原始数据的稀疏数组值相同
    assert_array_equal(X_r2inv[:, support_mask].toarray(), X.toarray())

    # 检查其他列是否为空
    assert X_r2inv.nnz == X_r.nnz
# 在分类设置中测试单变量特征选择

def test_select_kbest_classif():
    # 测试是否能在简单的分类问题中，使用 k 最佳启发式方法
    # 正确获取相应的特征项
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    # 使用 f_classif 作为评分函数，选择 k 个最佳特征
    univariate_filter = SelectKBest(f_classif, k=5)
    # 进行拟合和转换操作，将数据集 X 转换为 X_r
    X_r = univariate_filter.fit(X, y).transform(X)
    # 使用 GenericUnivariateSelect 进行相同操作，并转换为 X_r2
    X_r2 = (
        GenericUnivariateSelect(f_classif, mode="k_best", param=5)
        .fit(X, y)
        .transform(X)
    )
    # 断言转换后的结果 X_r 与 X_r2 相等
    assert_array_equal(X_r, X_r2)
    # 获取特征选择器的支持向量
    support = univariate_filter.get_support()
    # 创建一个长度为 20 的零向量，前 5 个元素置为 1，作为真实的支持向量
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    # 断言支持向量与真实支持向量相等
    assert_array_equal(support, gtruth)


def test_select_kbest_all():
    # 测试当 k="all" 时是否能正确返回所有特征。
    X, y = make_classification(
        n_samples=20, n_features=10, shuffle=False, random_state=0
    )

    # 使用 f_classif 作为评分函数，选择所有特征
    univariate_filter = SelectKBest(f_classif, k="all")
    # 进行拟合和转换操作，将数据集 X 转换为 X_r
    X_r = univariate_filter.fit(X, y).transform(X)
    # 断言转换后的结果 X 与 X_r 相等
    assert_array_equal(X, X_r)
    # 针对非回归测试：
    # https://github.com/scikit-learn/scikit-learn/issues/24949
    # 使用 GenericUnivariateSelect 进行相同操作，并转换为 X_r2
    X_r2 = (
        GenericUnivariateSelect(f_classif, mode="k_best", param="all")
        .fit(X, y)
        .transform(X)
    )
    # 断言转换后的结果 X_r 与 X_r2 相等
    assert_array_equal(X_r, X_r2)


@pytest.mark.parametrize("dtype_in", [np.float32, np.float64])
def test_select_kbest_zero(dtype_in):
    # 测试当 k=0 时是否能正确返回零个特征。
    X, y = make_classification(
        n_samples=20, n_features=10, shuffle=False, random_state=0
    )
    X = X.astype(dtype_in)

    # 使用 f_classif 作为评分函数，选择零个特征
    univariate_filter = SelectKBest(f_classif, k=0)
    # 进行拟合操作
    univariate_filter.fit(X, y)
    # 获取特征选择器的支持向量
    support = univariate_filter.get_support()
    # 创建一个长度为 10 的布尔零向量，作为真实的支持向量
    gtruth = np.zeros(10, dtype=bool)
    # 断言支持向量与真实支持向量相等
    assert_array_equal(support, gtruth)
    # 断言会产生 UserWarning，指示未选择任何特征
    with pytest.warns(UserWarning, match="No features were selected"):
        # 转换数据集 X，预期的 X_selected 形状为 (20, 0)
        X_selected = univariate_filter.transform(X)
    # 断言转换后的数据集形状和类型符合预期
    assert X_selected.shape == (20, 0)
    assert X_selected.dtype == dtype_in


def test_select_heuristics_classif():
    # 测试是否能在简单的分类问题中，使用 fdr、fwe 和 fpr 启发式方法
    # 正确获取相应的特征项
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    # 使用 f_classif 作为评分函数，选择 fwe 方法，alpha=0.01
    univariate_filter = SelectFwe(f_classif, alpha=0.01)
    # 进行拟合和转换操作，将数据集 X 转换为 X_r
    X_r = univariate_filter.fit(X, y).transform(X)
    # 创建一个长度为 20 的零向量，前 5 个元素置为 1，作为真实的支持向量
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    for mode in ["fdr", "fpr", "fwe"]:
        # 遍历三种统计模式：False Discovery Rate (fdr), False Positive Rate (fpr), Family Wise Error (fwe)
        
        # 使用 GenericUnivariateSelect 进行特征选择，基于 f_classif 统计方法，设置模式为当前循环的 mode，参数为 0.01
        X_r2 = (
            GenericUnivariateSelect(f_classif, mode=mode, param=0.01)
            .fit(X, y)  # 在数据集 X, y 上进行拟合（特征选择）
            .transform(X)  # 将 X 转换为所选特征的子集 X_r2
        )
        
        # 断言 X_r 和 X_r2 数组相等
        assert_array_equal(X_r, X_r2)
        
        # 获取特征选择器 univariate_filter 的支持信息（即被选中的特征）
        support = univariate_filter.get_support()
        
        # 断言 support 数组与真实的支持信息 gtruth 数组在数值上是近似相等的
        assert_allclose(support, gtruth)
##############################################################################
# Test univariate selection in regression settings

# 定义一个函数，用于验证分数过滤器是否保留了最佳分数
def assert_best_scores_kept(score_filter):
    # 获取特征评分
    scores = score_filter.scores_
    # 获取支持的特征
    support = score_filter.get_support()
    # 断言支持的特征的分数数组按排序后应与所有分数的排序后的前部分一致
    assert_allclose(np.sort(scores[support]), np.sort(scores)[-support.sum():])

# 定义一个测试函数，测试基于百分比的单变量特征选择在回归问题中的表现
def test_select_percentile_regression():
    # 生成一个简单的回归问题数据集
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0
    )

    # 创建基于百分比的单变量特征选择对象
    univariate_filter = SelectPercentile(f_regression, percentile=25)
    # 使用该对象拟合数据并转换数据集
    X_r = univariate_filter.fit(X, y).transform(X)
    # 断言是否保留了最佳分数
    assert_best_scores_kept(univariate_filter)
    
    # 创建另一个基于百分比的单变量特征选择对象
    X_r2 = (
        GenericUnivariateSelect(f_regression, mode="percentile", param=25)
        .fit(X, y)
        .transform(X)
    )
    # 断言两种方法的转换结果是否一致
    assert_array_equal(X_r, X_r2)
    
    # 获取特征选择的支持情况
    support = univariate_filter.get_support()
    # 创建一个理想的支持向量
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    # 断言实际支持与理想支持一致
    assert_array_equal(support, gtruth)
    
    # 创建一个新的数据集，并将不支持的特征置为零
    X_2 = X.copy()
    X_2[:, np.logical_not(support)] = 0
    # 断言逆转换后的数据是否与预期相符
    assert_array_equal(X_2, univariate_filter.inverse_transform(X_r))
    
    # 检查逆转换是否保持了数据类型
    assert_array_equal(
        X_2.astype(bool), univariate_filter.inverse_transform(X_r.astype(bool))
    )

# 测试百分比为100%时是否选择了所有特征
def test_select_percentile_regression_full():
    # 生成一个简单的回归问题数据集
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0
    )

    # 创建基于百分比的单变量特征选择对象，选择百分比为100%
    univariate_filter = SelectPercentile(f_regression, percentile=100)
    # 使用该对象拟合数据并转换数据集
    X_r = univariate_filter.fit(X, y).transform(X)
    # 断言是否保留了最佳分数
    assert_best_scores_kept(univariate_filter)
    
    # 创建另一个基于百分比的单变量特征选择对象，选择百分比为100%
    X_r2 = (
        GenericUnivariateSelect(f_regression, mode="percentile", param=100)
        .fit(X, y)
        .transform(X)
    )
    # 断言两种方法的转换结果是否一致
    assert_array_equal(X_r, X_r2)
    
    # 获取特征选择的支持情况
    support = univariate_filter.get_support()
    # 创建一个全部为1的理想支持向量
    gtruth = np.ones(20)
    # 断言实际支持与理想支持一致
    assert_array_equal(support, gtruth)

# 测试基于k个最佳特征的单变量特征选择在回归问题中的表现
def test_select_kbest_regression():
    # 生成一个简单的回归问题数据集
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=5,
        shuffle=False,
        random_state=0,
        noise=10,
    )

    # 创建基于k个最佳特征的单变量特征选择对象
    univariate_filter = SelectKBest(f_regression, k=5)
    # 使用该对象拟合数据并转换数据集
    X_r = univariate_filter.fit(X, y).transform(X)
    # 断言是否保留了最佳分数
    assert_best_scores_kept(univariate_filter)
    
    # 创建另一个基于k个最佳特征的单变量特征选择对象
    X_r2 = (
        GenericUnivariateSelect(f_regression, mode="k_best", param=5)
        .fit(X, y)
        .transform(X)
    )
    # 断言两种方法的转换结果是否一致
    assert_array_equal(X_r, X_r2)
    
    # 获取特征选择的支持情况
    support = univariate_filter.get_support()
    # 创建一个理想的支持向量
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    # 断言实际支持与理想支持一致
    assert_array_equal(support, gtruth)

# 测试各种启发式方法在回归问题中的表现
def test_select_heuristics_regression():
    # 创建一个具有特定特征的回归问题的样本数据集 X 和目标值 y
    X, y = make_regression(
        n_samples=200,          # 样本数为200
        n_features=20,          # 特征数为20
        n_informative=5,        # 其中有信息的特征数为5
        shuffle=False,          # 不对样本进行洗牌
        random_state=0,         # 随机数种子设为0，以确保可重复性
        noise=10,               # 添加高斯噪声，标准差为10
    )

    # 创建一个使用 F 检验（f_regression）进行特征选择的过滤器对象
    univariate_filter = SelectFpr(f_regression, alpha=0.01)
    # 对数据集 X 进行拟合并转换，得到选择后的特征数据集 X_r
    X_r = univariate_filter.fit(X, y).transform(X)

    # 创建一个真值数组，表示哪些特征是相关的
    gtruth = np.zeros(20)
    gtruth[:5] = 1

    # 针对不同的模式（"fdr", "fpr", "fwe"），进行泛化单变量特征选择的测试
    for mode in ["fdr", "fpr", "fwe"]:
        # 使用指定模式和参数进行泛化单变量特征选择，得到选定的特征数据集 X_r2
        X_r2 = (
            GenericUnivariateSelect(f_regression, mode=mode, param=0.01)
            .fit(X, y)
            .transform(X)
        )
        
        # 断言两种特征选择方法得到的结果应该一致
        assert_array_equal(X_r, X_r2)
        
        # 获取使用 FPR 方法选择的特征支持数组
        support = univariate_filter.get_support()
        
        # 断言前5个特征确实被选择
        assert_array_equal(support[:5], np.ones((5,), dtype=bool))
        
        # 断言后15个特征中最多有3个被选择
        assert np.sum(support[5:] == 1) < 3
# 定义测试边界情况，始终选择一个特征进行测试。
def test_boundary_case_ch2():
    # 创建包含特征和标签的 NumPy 数组 X 和 y
    X = np.array([[10, 20], [20, 20], [20, 30]])
    y = np.array([[1], [0], [0]])
    
    # 使用卡方检验计算特征的得分和 p 值
    scores, pvalues = chi2(X, y)
    # 断言计算得到的卡方分数和 p 值与预期值相近
    assert_array_almost_equal(scores, np.array([4.0, 0.71428571]))
    assert_array_almost_equal(pvalues, np.array([0.04550026, 0.39802472]))

    # 使用 SelectFdr 进行特征选择，设置 alpha 为 0.1
    filter_fdr = SelectFdr(chi2, alpha=0.1)
    filter_fdr.fit(X, y)
    # 获取 SelectFdr 选择的特征支持情况
    support_fdr = filter_fdr.get_support()
    # 断言 SelectFdr 选择的特征支持情况与预期相等
    assert_array_equal(support_fdr, np.array([True, False]))

    # 使用 SelectKBest 进行特征选择，选择 k=1 个特征
    filter_kbest = SelectKBest(chi2, k=1)
    filter_kbest.fit(X, y)
    # 获取 SelectKBest 选择的特征支持情况
    support_kbest = filter_kbest.get_support()
    # 断言 SelectKBest 选择的特征支持情况与预期相等
    assert_array_equal(support_kbest, np.array([True, False]))

    # 使用 SelectPercentile 进行特征选择，选择前 50% 的特征
    filter_percentile = SelectPercentile(chi2, percentile=50)
    filter_percentile.fit(X, y)
    # 获取 SelectPercentile 选择的特征支持情况
    support_percentile = filter_percentile.get_support()
    # 断言 SelectPercentile 选择的特征支持情况与预期相等
    assert_array_equal(support_percentile, np.array([True, False]))

    # 使用 SelectFpr 进行特征选择，设置 alpha 为 0.1
    filter_fpr = SelectFpr(chi2, alpha=0.1)
    filter_fpr.fit(X, y)
    # 获取 SelectFpr 选择的特征支持情况
    support_fpr = filter_fpr.get_support()
    # 断言 SelectFpr 选择的特征支持情况与预期相等
    assert_array_equal(support_fpr, np.array([True, False]))

    # 使用 SelectFwe 进行特征选择，设置 alpha 为 0.1
    filter_fwe = SelectFwe(chi2, alpha=0.1)
    filter_fwe.fit(X, y)
    # 获取 SelectFwe 选择的特征支持情况
    support_fwe = filter_fwe.get_support()
    # 断言 SelectFwe 选择的特征支持情况与预期相等
    assert_array_equal(support_fwe, np.array([True, False]))


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("n_informative", [1, 5, 10])
def test_select_fdr_regression(alpha, n_informative):
    # 测试 FDR 策略确实具有较低的 FDR。
    def single_fdr(alpha, n_informative, random_state):
        # 生成回归数据集 X, y
        X, y = make_regression(
            n_samples=150,
            n_features=20,
            n_informative=n_informative,
            shuffle=False,
            random_state=random_state,
            noise=10,
        )

        with warnings.catch_warnings(record=True):
            # 当未选择特征时可能会引发警告（低 alpha 或数据噪声较大）
            univariate_filter = SelectFdr(f_regression, alpha=alpha)
            # 使用 SelectFdr 进行特征选择并转换数据集 X
            X_r = univariate_filter.fit(X, y).transform(X)
            # 使用 GenericUnivariateSelect 进行相同的操作
            X_r2 = (
                GenericUnivariateSelect(f_regression, mode="fdr", param=alpha)
                .fit(X, y)
                .transform(X)
            )

        # 断言两种方法得到的结果一致
        assert_array_equal(X_r, X_r2)
        # 获取 SelectFdr 选择的特征支持情况
        support = univariate_filter.get_support()
        # 计算假阳性的数量
        num_false_positives = np.sum(support[n_informative:] == 1)
        # 计算真阳性的数量
        num_true_positives = np.sum(support[:n_informative] == 1)

        if num_false_positives == 0:
            return 0.0
        # 计算假发现率（FDR）
        false_discovery_rate = num_false_positives / (
            num_true_positives + num_false_positives
        )
        return false_discovery_rate

    # 根据 Benjamini-Hochberg 方法，预期的假发现率应低于 alpha：
    # FDR = E(FP / (TP + FP)) <= alpha
    # 计算多个随机状态下的平均假发现率
    false_discovery_rate = np.mean(
        [single_fdr(alpha, n_informative, random_state) for random_state in range(100)]
    )
    # 断言 alpha 大于等于平均假发现率
    assert alpha >= false_discovery_rate
    # 确保经验误发现率随着 alpha 的增加而增加：
    if false_discovery_rate != 0:
        # 断言：经验误发现率应大于 alpha 的十分之一
        assert false_discovery_rate > alpha / 10
def test_select_fwe_regression():
    # 测试相对单变量特征选择在简单回归问题中使用 fwe 启发式方法是否能正确获取项目
    # 创建具有 200 个样本、20 个特征和 5 个信息特征的回归数据集，不打乱顺序
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=0
    )

    # 使用 f_regression 函数创建选择 FWE 方法的单变量特征选择器
    univariate_filter = SelectFwe(f_regression, alpha=0.01)
    # 对数据集 X, y 进行拟合并转换为新的特征集 X_r
    X_r = univariate_filter.fit(X, y).transform(X)
    
    # 使用 f_regression 函数创建通用单变量选择器，使用 FWE 模式和参数 0.01
    X_r2 = (
        GenericUnivariateSelect(f_regression, mode="fwe", param=0.01)
        .fit(X, y)
        .transform(X)
    )
    # 断言两个转换后的特征集是否相等
    assert_array_equal(X_r, X_r2)
    
    # 获取特征选择器的支持向量（选择的特征的布尔掩码）
    support = univariate_filter.get_support()
    # 创建一个真值数组，前 5 个为 1，其余为 0
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    # 断言支持向量的前 5 个元素是否为 True
    assert_array_equal(support[:5], np.ones((5,), dtype=bool))
    # 断言支持向量中为 1 的元素数量小于 2
    assert np.sum(support[5:] == 1) < 2


def test_selectkbest_tiebreaking():
    # 测试 SelectKBest 在存在并列情况下是否确实选择 k 个特征
    # 在 0.11 版本之前，SelectKBest 可能会返回比请求的特征更多的情况
    Xs = [[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]]
    y = [1]
    # 定义一个虚拟评分函数，返回元素自身作为评分
    dummy_score = lambda X, y: (X[0], X[0])
    for X in Xs:
        # 使用 dummy_score 函数创建 SelectKBest 选择器，选择 1 个特征
        sel = SelectKBest(dummy_score, k=1)
        # 忽略警告并对数据进行拟合和转换
        X1 = ignore_warnings(sel.fit_transform)([X], y)
        # 断言转换后的特征集 X1 的特征数是否为 1
        assert X1.shape[1] == 1
        # 断言最佳得分是否得到保留
        assert_best_scores_kept(sel)

        # 使用 dummy_score 函数创建 SelectKBest 选择器，选择 2 个特征
        sel = SelectKBest(dummy_score, k=2)
        # 忽略警告并对数据进行拟合和转换
        X2 = ignore_warnings(sel.fit_transform)([X], y)
        # 断言转换后的特征集 X2 的特征数是否为 2
        assert X2.shape[1] == 2
        # 断言最佳得分是否得到保留
        assert_best_scores_kept(sel)


def test_selectpercentile_tiebreaking():
    # 测试 SelectPercentile 在存在并列情况下是否选择正确的 n_features
    Xs = [[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]]
    y = [1]
    # 定义一个虚拟评分函数，返回元素自身作为评分
    dummy_score = lambda X, y: (X[0], X[0])
    for X in Xs:
        # 使用 dummy_score 函数创建 SelectPercentile 选择器，选择 34% 的特征
        sel = SelectPercentile(dummy_score, percentile=34)
        # 忽略警告并对数据进行拟合和转换
        X1 = ignore_warnings(sel.fit_transform)([X], y)
        # 断言转换后的特征集 X1 的特征数是否为 1
        assert X1.shape[1] == 1
        # 断言最佳得分是否得到保留
        assert_best_scores_kept(sel)

        # 使用 dummy_score 函数创建 SelectPercentile 选择器，选择 67% 的特征
        sel = SelectPercentile(dummy_score, percentile=67)
        # 忽略警告并对数据进行拟合和转换
        X2 = ignore_warnings(sel.fit_transform)([X], y)
        # 断言转换后的特征集 X2 的特征数是否为 2
        assert X2.shape[1] == 2
        # 断言最佳得分是否得到保留
        assert_best_scores_kept(sel)


def test_tied_pvalues():
    # 测试在 chi2 返回并列 p 值的情况下，k-best 和 percentiles 是否工作正常
    # chi2 将返回以下特征的相同 p 值，但会返回不同的分数。
    X0 = np.array([[10000, 9999, 9998], [1, 1, 1]])
    y = [0, 1]

    # 对特征排列的所有排列进行迭代
    for perm in itertools.permutations((0, 1, 2)):
        X = X0[:, perm]
        # 使用 chi2 创建 SelectKBest 选择器，选择 2 个特征
        Xt = SelectKBest(chi2, k=2).fit_transform(X, y)
        # 断言转换后的特征集 Xt 的形状是否为 (2, 2)
        assert Xt.shape == (2, 2)
        # 断言 9998 不在 Xt 中
        assert 9998 not in Xt

        # 使用 chi2 创建 SelectPercentile 选择器，选择 67% 的特征
        Xt = SelectPercentile(chi2, percentile=67).fit_transform(X, y)
        # 断言转换后的特征集 Xt 的形状是否为 (2, 2)
        assert Xt.shape == (2, 2)
        # 断言 9998 不在 Xt 中
        assert 9998 not in Xt


def test_scorefunc_multilabel():
    # 测试多标签情况下，k-best 和 percentiles 是否与 chi2 正常工作
    X = np.array([[10000, 9999, 0], [100, 9999, 0], [1000, 99, 0]])
    y = [[1, 1], [0, 1], [1, 0]]

    # 使用 chi2 创建 SelectKBest 选择器，选择 2 个特征
    Xt = SelectKBest(chi2, k=2).fit_transform(X, y)
    # 断言转换后的特征集 Xt 的形状是否为 (3, 2)
    assert Xt.shape == (3, 2)
    # 断言 0 不在 Xt 中
    assert 0 not in Xt
    # 使用卡方检验选择最相关的特征，并转换原始特征矩阵 X
    Xt = SelectPercentile(chi2, percentile=67).fit_transform(X, y)
    # 断言转换后的特征矩阵 Xt 的形状是否为 (3, 2)，即行数为 3，列数为 2
    assert Xt.shape == (3, 2)
    # 断言转换后的特征矩阵 Xt 中不包含值为 0 的元素
    assert 0 not in Xt
def test_tied_scores():
    # 测试在有并列分数的情况下 k-best 稳定排序。
    X_train = np.array([[0, 0, 0], [1, 1, 1]])
    y_train = [0, 1]

    for n_features in [1, 2, 3]:
        # 使用 chi2 方法选择 k 个最佳特征，拟合并转换训练数据
        sel = SelectKBest(chi2, k=n_features).fit(X_train, y_train)
        # 对新数据集应用选择的特征集合
        X_test = sel.transform([[0, 1, 2]])
        # 断言转换后的数据与预期结果一致
        assert_array_equal(X_test[0], np.arange(3)[-n_features:])


def test_nans():
    # 断言 SelectKBest 和 SelectPercentile 能够处理 NaN 值。
    # 第一个特征方差为零，可能会导致 f_classif 返回 NaN。
    X = [[0, 1, 0], [0, -1, -1], [0, 0.5, 0.5]]
    y = [1, 0, 1]

    for select in (
        SelectKBest(f_classif, k=2),
        SelectPercentile(f_classif, percentile=67),
    ):
        # 忽略警告并拟合数据
        ignore_warnings(select.fit)(X, y)
        # 断言选定的特征索引与预期结果一致
        assert_array_equal(select.get_support(indices=True), np.array([1, 2]))


def test_invalid_k():
    X = [[0, 1, 0], [0, -1, -1], [0, 0.5, 0.5]]
    y = [1, 0, 1]

    # 断言当 k 大于特征数时会产生 UserWarning 提示消息
    msg = "k=4 is greater than n_features=3. All the features will be returned."
    with pytest.warns(UserWarning, match=msg):
        SelectKBest(k=4).fit(X, y)
    with pytest.warns(UserWarning, match=msg):
        GenericUnivariateSelect(mode="k_best", param=4).fit(X, y)


def test_f_classif_constant_feature():
    # 测试当特征全为常数时，f_classif 是否会发出警告。
    X, y = make_classification(n_samples=10, n_features=5)
    X[:, 0] = 2.0
    with pytest.warns(UserWarning):
        f_classif(X, y)


def test_no_feature_selected():
    rng = np.random.RandomState(0)

    # 生成随机的不相关数据集：严格的单变量测试应该拒绝所有特征
    X = rng.rand(40, 10)
    y = rng.randint(0, 4, size=40)
    strict_selectors = [
        SelectFwe(alpha=0.01).fit(X, y),
        SelectFdr(alpha=0.01).fit(X, y),
        SelectFpr(alpha=0.01).fit(X, y),
        SelectPercentile(percentile=0).fit(X, y),
        SelectKBest(k=0).fit(X, y),
    ]
    for selector in strict_selectors:
        # 断言没有特征被选择，并且会产生警告消息
        assert_array_equal(selector.get_support(), np.zeros(10))
        with pytest.warns(UserWarning, match="No features were selected"):
            X_selected = selector.transform(X)
        assert X_selected.shape == (40, 0)


def test_mutual_info_classif():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=1,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=10,
        shuffle=False,
        random_state=0,
    )

    # 在 KBest 模式下测试 mutual_info_classif
    univariate_filter = SelectKBest(mutual_info_classif, k=2)
    X_r = univariate_filter.fit(X, y).transform(X)
    X_r2 = (
        GenericUnivariateSelect(mutual_info_classif, mode="k_best", param=2)
        .fit(X, y)
        .transform(X)
    )
    assert_array_equal(X_r, X_r2)
    # 断言支持的特征与预期结果一致
    support = univariate_filter.get_support()
    gtruth = np.zeros(5)
    gtruth[:2] = 1
    assert_array_equal(support, gtruth)
    # 在百分位模式下进行测试。

    # 创建一个基于互信息分类的 SelectPercentile 对象，选择排名在前 40% 的特征
    univariate_filter = SelectPercentile(mutual_info_classif, percentile=40)

    # 使用 SelectPercentile 对象对特征进行拟合和转换
    X_r = univariate_filter.fit(X, y).transform(X)

    # 使用 GenericUnivariateSelect 创建另一个基于百分位模式和互信息分类的对象，选择排名在前 40% 的特征并进行拟合和转换
    X_r2 = (
        GenericUnivariateSelect(mutual_info_classif, mode="percentile", param=40)
        .fit(X, y)
        .transform(X)
    )

    # 断言两个转换后的特征数组 X_r 和 X_r2 相等
    assert_array_equal(X_r, X_r2)

    # 获取特征选择后的支持情况（哪些特征被选中）
    support = univariate_filter.get_support()

    # 创建一个全零的目标数组 gtruth，前两个位置置为 1，表示预期的特征选择情况
    gtruth = np.zeros(5)
    gtruth[:2] = 1

    # 断言实际支持情况与预期相同
    assert_array_equal(support, gtruth)
def test_mutual_info_regression():
    # 创建具有特定特征和噪声的合成数据集
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=2,
        shuffle=False,
        random_state=0,
        noise=10,
    )

    # 在 KBest 模式下进行测试。
    # 使用 mutual_info_regression 作为评分函数创建 SelectKBest 对象
    univariate_filter = SelectKBest(mutual_info_regression, k=2)
    # 对数据集 X 进行拟合和转换
    X_r = univariate_filter.fit(X, y).transform(X)
    # 确保最佳分数被保留
    assert_best_scores_kept(univariate_filter)
    # 使用 GenericUnivariateSelect 创建对象，并进行拟合和转换
    X_r2 = (
        GenericUnivariateSelect(mutual_info_regression, mode="k_best", param=2)
        .fit(X, y)
        .transform(X)
    )
    # 确保两种方法的转换结果一致
    assert_array_equal(X_r, X_r2)
    # 获取特征选择的支持向量
    support = univariate_filter.get_support()
    # 创建一个真值数组，前两个特征被选择
    gtruth = np.zeros(10)
    gtruth[:2] = 1
    # 确保支持向量与真值数组一致
    assert_array_equal(support, gtruth)

    # 在 Percentile 模式下进行测试。
    # 使用 mutual_info_regression 作为评分函数创建 SelectPercentile 对象
    univariate_filter = SelectPercentile(mutual_info_regression, percentile=20)
    # 对数据集 X 进行拟合和转换
    X_r = univariate_filter.fit(X, y).transform(X)
    # 使用 GenericUnivariateSelect 创建对象，并进行拟合和转换
    X_r2 = (
        GenericUnivariateSelect(mutual_info_regression, mode="percentile", param=20)
        .fit(X, y)
        .transform(X)
    )
    # 确保两种方法的转换结果一致
    assert_array_equal(X_r, X_r2)
    # 获取特征选择的支持向量
    support = univariate_filter.get_support()
    # 创建一个真值数组，前两个特征被选择
    gtruth = np.zeros(10)
    gtruth[:2] = 1
    # 确保支持向量与真值数组一致
    assert_array_equal(support, gtruth)


def test_dataframe_output_dtypes():
    """Check that the output datafarme dtypes are the same as the input.

    Non-regression test for gh-24860.
    """
    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 加载鸢尾花数据集，并将其转换为 DataFrame
    X, y = load_iris(return_X_y=True, as_frame=True)
    # 将数据集 X 的某些列转换为指定的数据类型
    X = X.astype(
        {
            "petal length (cm)": np.float32,
            "petal width (cm)": np.float64,
        }
    )
    # 创建一个新的列，将 X 中的某列分为 10 个箱子
    X["petal_width_binned"] = pd.cut(X["petal width (cm)"], bins=10)

    # 获取列的顺序
    column_order = X.columns

    # 定义一个选择器函数，返回特征名称对应的排名
    def selector(X, y):
        ranking = {
            "sepal length (cm)": 1,
            "sepal width (cm)": 2,
            "petal length (cm)": 3,
            "petal width (cm)": 4,
            "petal_width_binned": 5,
        }
        return np.asarray([ranking[name] for name in column_order])

    # 使用 selector 函数创建 SelectKBest 对象，并指定输出为 DataFrame
    univariate_filter = SelectKBest(selector, k=3).set_output(transform="pandas")
    # 对数据集 X, y 进行拟合和转换
    output = univariate_filter.fit_transform(X, y)

    # 确保输出 DataFrame 的列与指定的列名一致
    assert_array_equal(
        output.columns, ["petal length (cm)", "petal width (cm)", "petal_width_binned"]
    )
    # 确保输出 DataFrame 的数据类型与输入 X 的数据类型一致
    for name, dtype in output.dtypes.items():
        assert dtype == X.dtypes[name]


@pytest.mark.parametrize(
    "selector",
    [
        SelectKBest(k=4),
        SelectPercentile(percentile=80),
        GenericUnivariateSelect(mode="k_best", param=4),
        GenericUnivariateSelect(mode="percentile", param=80),
    ],
)
def test_unsupervised_filter(selector):
    """Check support for unsupervised feature selection for the filter that could
    require only `X`.
    """
    # 使用指定随机种子创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个随机数据集 X，大小为 (10, 5)
    X = rng.randn(10, 5)

    # 定义一个评分函数，返回固定的特征重要性评分
    def score_func(X, y=None):
        return np.array([1, 1, 1, 1, 0])

    # 设置选择器的评分函数为 score_func，并对数据集 X 进行拟合
    selector.set_params(score_func=score_func)
    selector.fit(X)
    # 对数据集 X 进行转换，并与原始数据的前四列进行比较
    X_trans = selector.transform(X)
    # 确保转换结果与预期的前四列数据一致
    assert_allclose(X_trans, X[:, :4])
    # 使用选择器对象对输入数据 X 进行拟合和转换
    X_trans = selector.fit_transform(X)
    # 断言函数，验证 X_trans 和 X 的前四列数据是否在数值上非常接近
    assert_allclose(X_trans, X[:, :4])
```