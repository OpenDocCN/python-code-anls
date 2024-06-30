# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_random_projection.py`

```
import functools  # 导入 functools 模块，用于高阶函数的操作
import warnings  # 导入 warnings 模块，用于管理警告信息
from typing import Any, List  # 导入类型提示相关的模块和类

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 PyTest 测试框架
import scipy.sparse as sp  # 导入 SciPy 稀疏矩阵模块

from sklearn.exceptions import DataDimensionalityWarning, NotFittedError  # 导入异常类
from sklearn.metrics import euclidean_distances  # 导入 Euclidean 距离计算函数
from sklearn.random_projection import (  # 导入随机投影相关模块和函数
    GaussianRandomProjection,
    SparseRandomProjection,
    _gaussian_random_matrix,
    _sparse_random_matrix,
    johnson_lindenstrauss_min_dim,
)
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import COO_CONTAINERS  # 导入 COO 格式相关的修复函数和变量

all_sparse_random_matrix: List[Any] = [_sparse_random_matrix]  # 初始化稀疏随机矩阵列表
all_dense_random_matrix: List[Any] = [_gaussian_random_matrix]  # 初始化密集随机矩阵列表
all_random_matrix = all_sparse_random_matrix + all_dense_random_matrix  # 组合所有随机矩阵类型列表

all_SparseRandomProjection: List[Any] = [SparseRandomProjection]  # 初始化稀疏随机投影列表
all_DenseRandomProjection: List[Any] = [GaussianRandomProjection]  # 初始化密集随机投影列表
all_RandomProjection = all_SparseRandomProjection + all_DenseRandomProjection  # 组合所有随机投影类型列表


def make_sparse_random_data(  # 定义生成稀疏随机数据的函数
    coo_container,
    n_samples,
    n_features,
    n_nonzeros,
    random_state=None,
    sparse_format="csr",
):
    """Make some random data with uniformly located non zero entries with
    Gaussian distributed values; `sparse_format` can be `"csr"` (default) or
    `None` (in which case a dense array is returned).
    """
    rng = np.random.RandomState(random_state)  # 使用给定的随机种子创建随机数生成器
    data_coo = coo_container(  # 使用给定的 COO 容器创建稀疏矩阵
        (
            rng.randn(n_nonzeros),  # 生成服从标准正态分布的随机数作为数据
            (
                rng.randint(n_samples, size=n_nonzeros),  # 生成随机行索引
                rng.randint(n_features, size=n_nonzeros),  # 生成随机列索引
            ),
        ),
        shape=(n_samples, n_features),  # 指定稀疏矩阵的形状
    )
    if sparse_format is not None:  # 如果指定了稀疏格式
        return data_coo.asformat(sparse_format)  # 返回按格式转换后的稀疏矩阵
    else:
        return data_coo.toarray()  # 否则返回稠密数组


def densify(matrix):
    if not sp.issparse(matrix):  # 如果输入矩阵不是稀疏矩阵
        return matrix  # 直接返回原始矩阵
    else:
        return matrix.toarray()  # 否则将稀疏矩阵转换为稠密数组


n_samples, n_features = (10, 1000)  # 定义样本数和特征数
n_nonzeros = int(n_samples * n_features / 100.0)  # 计算非零元素的数量


###############################################################################
# test on JL lemma
###############################################################################


@pytest.mark.parametrize(  # 使用 PyTest 的参数化装饰器，定义参数化测试
    "n_samples, eps",  # 参数名
    [  # 参数组合列表
        ([100, 110], [0.9, 1.1]),  # 第一组参数
        ([90, 100], [0.1, 0.0]),  # 第二组参数
        ([50, -40], [0.1, 0.2]),  # 第三组参数
    ],
)
def test_invalid_jl_domain(n_samples, eps):
    with pytest.raises(ValueError):  # 检查是否抛出 ValueError 异常
        johnson_lindenstrauss_min_dim(n_samples, eps=eps)  # 调用 Johnson-Lindenstrauss 最小维度函数


def test_input_size_jl_min_dim():
    with pytest.raises(ValueError):  # 检查是否抛出 ValueError 异常
        johnson_lindenstrauss_min_dim(3 * [100], eps=2 * [0.9])  # 调用 Johnson-Lindenstrauss 最小维度函数

    johnson_lindenstrauss_min_dim(
        np.random.randint(1, 10, size=(10, 10)), eps=np.full((10, 10), 0.5)
    )  # 调用 Johnson-Lindenstrauss 最小维度函数，输入随机生成的数组


###############################################################################
# tests random matrix generation
###############################################################################
# 检查输入大小是否符合随机矩阵生成函数的要求
def check_input_size_random_matrix(random_matrix):
    # 定义不同的输入组合：(n_components, n_features)
    inputs = [(0, 0), (-1, 1), (1, -1), (1, 0), (-1, 0)]
    # 遍历每个输入组合
    for n_components, n_features in inputs:
        # 使用 pytest 检查调用 random_matrix(n_components, n_features) 是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features)


# 检查生成的随机矩阵的大小是否与期望一致
def check_size_generated(random_matrix):
    # 定义不同的输入组合：(n_components, n_features)
    inputs = [(1, 5), (5, 1), (5, 5), (1, 1)]
    # 遍历每个输入组合
    for n_components, n_features in inputs:
        # 断言生成的随机矩阵的形状是否符合期望 (n_components, n_features)
        assert random_matrix(n_components, n_features).shape == (
            n_components,
            n_features,
        )


# 检查生成的随机矩阵是否具有零均值和单位范数
def check_zero_mean_and_unit_norm(random_matrix):
    # 所有随机矩阵应生成具有零均值和单位范数的转换矩阵
    A = densify(random_matrix(10000, 1, random_state=0))
    # 断言矩阵 A 的平均值近似为 0，精确到小数点后三位
    assert_array_almost_equal(0, np.mean(A), 3)
    # 断言矩阵 A 的范数近似为 1.0，精确到小数点后一位
    assert_array_almost_equal(1.0, np.linalg.norm(A), 1)


# 检查稀疏随机矩阵生成函数对输入的处理
def check_input_with_sparse_random_matrix(random_matrix):
    n_components, n_features = 5, 10
    # 对于不同的密度值，检查是否会抛出 ValueError 异常
    for density in [-1.0, 0.0, 1.1]:
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features, density=density)


# 测试随机矩阵生成函数的基本属性
@pytest.mark.parametrize("random_matrix", all_random_matrix)
def test_basic_property_of_random_matrix(random_matrix):
    # 检查随机矩阵生成函数的基本属性
    check_input_size_random_matrix(random_matrix)
    check_size_generated(random_matrix)
    check_zero_mean_and_unit_norm(random_matrix)


# 测试稀疏随机矩阵生成函数的基本属性
@pytest.mark.parametrize("random_matrix", all_sparse_random_matrix)
def test_basic_property_of_sparse_random_matrix(random_matrix):
    # 检查稀疏随机矩阵生成函数对输入的处理
    check_input_with_sparse_random_matrix(random_matrix)
    
    # 创建稠密随机矩阵生成函数
    random_matrix_dense = functools.partial(random_matrix, density=1.0)
    
    # 检查生成的稠密随机矩阵是否具有零均值和单位范数
    check_zero_mean_and_unit_norm(random_matrix_dense)


# 测试高斯随机矩阵生成函数
def test_gaussian_random_matrix():
    # 检查高斯随机矩阵生成函数的统计属性
    n_components = 100
    n_features = 1000
    # 生成高斯随机矩阵 A
    A = _gaussian_random_matrix(n_components, n_features, random_state=0)
    # 断言矩阵 A 的平均值近似为 0.0，精确到小数点后两位
    assert_array_almost_equal(0.0, np.mean(A), 2)
    # 断言矩阵 A 的方差近似为 1 / n_components，精确到小数点后一位
    assert_array_almost_equal(np.var(A, ddof=1), 1 / n_components, 1)


# 测试稀疏随机矩阵生成函数
def test_sparse_random_matrix():
    # 检查稀疏随机矩阵生成函数的统计属性
    n_components = 100
    n_features = 500
    for density in [0.3, 1.0]:
        # 计算稀疏度对应的稠密化因子
        s = 1 / density

        # 生成稀疏随机矩阵 A
        A = _sparse_random_matrix(
            n_components, n_features, density=density, random_state=0
        )
        # 将稀疏矩阵 A 转换为稠密矩阵
        A = densify(A)

        # 检查可能的数值集合
        values = np.unique(A)
        # 断言 -sqrt(s) / sqrt(n_components) 在 values 中
        assert np.sqrt(s) / np.sqrt(n_components) in values
        # 断言 +sqrt(s) / sqrt(n_components) 在 values 中
        assert -np.sqrt(s) / np.sqrt(n_components) in values

        if density == 1.0:
            # 断言 values 中元素数为 2
            assert np.size(values) == 2
        else:
            # 断言 values 中包含 0.0
            assert 0.0 in values
            # 断言 values 中元素数为 3
            assert np.size(values) == 3

        # 检查随机矩阵是否遵循正确的分布
        # 每个元素 a_{ij} 可能取自以下分布：
        #
        # - -sqrt(s) / sqrt(n_components)   概率为 1 / 2s
        # -  0                              概率为 1 - 1 / s
        # - +sqrt(s) / sqrt(n_components)   概率为 1 / 2s
        #
        assert_almost_equal(np.mean(A == 0.0), 1 - 1 / s, decimal=2)
        assert_almost_equal(
            np.mean(A == np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2
        )
        assert_almost_equal(
            np.mean(A == -np.sqrt(s) / np.sqrt(n_components)), 1 / (2 * s), decimal=2
        )

        # 检查方差是否正确
        assert_almost_equal(np.var(A == 0.0, ddof=1), (1 - 1 / s) * 1 / s, decimal=2)
        assert_almost_equal(
            np.var(A == np.sqrt(s) / np.sqrt(n_components), ddof=1),
            (1 - 1 / (2 * s)) * 1 / (2 * s),
            decimal=2,
        )
        assert_almost_equal(
            np.var(A == -np.sqrt(s) / np.sqrt(n_components), ddof=1),
            (1 - 1 / (2 * s)) * 1 / (2 * s),
            decimal=2,
        )
###############################################################################
# tests on random projection transformer
###############################################################################


# 定义测试函数，检查随机投影转换器在无效输入下的行为
def test_random_projection_transformer_invalid_input():
    # 设置自动计算主成分数（无效输入）
    n_components = "auto"
    # 创建适合拟合的数据
    fit_data = [[0, 1, 2]]
    # 遍历所有的随机投影器类
    for RandomProjection in all_RandomProjection:
        # 断言应该抛出值错误异常
        with pytest.raises(ValueError):
            # 使用自动主成分数进行拟合
            RandomProjection(n_components=n_components).fit(fit_data)


# 使用参数化测试装饰器，测试在未拟合之前尝试转换的行为
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_try_to_transform_before_fit(coo_container, global_random_seed):
    # 生成稀疏随机数据
    data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format=None,
    )
    # 遍历所有的随机投影器类
    for RandomProjection in all_RandomProjection:
        # 断言应该抛出未拟合错误异常
        with pytest.raises(NotFittedError):
            # 尝试使用自动主成分数进行转换
            RandomProjection(n_components="auto").transform(data)


# 使用参数化测试装饰器，测试尝试在给定条件下找到安全嵌入的行为
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_too_many_samples_to_find_a_safe_embedding(coo_container, global_random_seed):
    # 生成稀疏随机数据，样本数较多
    data = make_sparse_random_data(
        coo_container,
        n_samples=1000,
        n_features=100,
        n_nonzeros=1000,
        random_state=global_random_seed,
        sparse_format=None,
    )

    # 遍历所有的随机投影器类
    for RandomProjection in all_RandomProjection:
        # 创建随机投影器对象，设置自动主成分数和容差
        rp = RandomProjection(n_components="auto", eps=0.1)
        # 预期的错误信息
        expected_msg = (
            "eps=0.100000 and n_samples=1000 lead to a target dimension"
            " of 5920 which is larger than the original space with"
            " n_features=100"
        )
        # 断言应该抛出值错误异常，并匹配预期的错误信息
        with pytest.raises(ValueError, match=expected_msg):
            # 对数据进行拟合
            rp.fit(data)


# 使用参数化测试装饰器，测试随机投影嵌入质量的行为
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_random_projection_embedding_quality(coo_container):
    # 生成稀疏随机数据，设置样本数和特征数
    data = make_sparse_random_data(
        coo_container,
        n_samples=8,
        n_features=5000,
        n_nonzeros=15000,
        random_state=0,
        sparse_format=None,
    )
    # 设置容差
    eps = 0.2

    # 计算原始数据间的欧氏距离的平方
    original_distances = euclidean_distances(data, squared=True)
    # 将距离扁平化
    original_distances = original_distances.ravel()
    # 非零距离的索引
    non_identical = original_distances != 0.0

    # 移除零距离，避免除以零
    original_distances = original_distances[non_identical]
    # 对每个随机投影对象进行迭代
    for RandomProjection in all_RandomProjection:
        # 使用自动计算的组件数创建随机投影对象，并设置参数 eps 和 random_state
        rp = RandomProjection(n_components="auto", eps=eps, random_state=0)
        # 对数据进行随机投影变换
        projected = rp.fit_transform(data)

        # 计算投影后数据点之间的欧氏距离的平方
        projected_distances = euclidean_distances(projected, squared=True)
        # 将距离展平为一维数组
        projected_distances = projected_distances.ravel()

        # 从投影后的距离中移除值为 0 的距离，以避免出现除以 0 的情况
        projected_distances = projected_distances[non_identical]

        # 计算投影后距离与原始数据点距离的比率
        distances_ratio = projected_distances / original_distances

        # 检查自动调整的密度值是否符合 eps 的要求：
        # 根据Johnson-Lindenstrauss引理，保持成对距离不变
        assert distances_ratio.max() < 1 + eps
        assert 1 - eps < distances_ratio.min()
# 使用参数化测试框架pytest.mark.parametrize，对每个COO_CONTAINERS中的元素进行单独测试
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_SparseRandomProj_output_representation(coo_container):
    # 使用make_sparse_random_data函数生成稀疏随机数据，格式为稠密数组（sparse_format=None）
    dense_data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=0,
        sparse_format=None,
    )
    # 使用make_sparse_random_data函数生成稀疏随机数据，格式为CSR稀疏数组（sparse_format="csr"）
    sparse_data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=0,
        sparse_format="csr",
    )
    # 对all_SparseRandomProjection中的每个SparseRandomProj进行测试
    for SparseRandomProj in all_SparseRandomProjection:
        # 使用SparseRandomProj构造函数创建SparseRandomProj对象rp，将稠密数据dense_data拟合进去
        rp = SparseRandomProj(n_components=10, dense_output=True, random_state=0)
        rp.fit(dense_data)
        # 断言transform后的输出类型为numpy数组
        assert isinstance(rp.transform(dense_data), np.ndarray)
        # 断言transform后的稀疏数据输出类型为numpy数组
        assert isinstance(rp.transform(sparse_data), np.ndarray)

        # 使用SparseRandomProj构造函数创建SparseRandomProj对象rp，将稠密数据dense_data拟合进去
        rp = SparseRandomProj(n_components=10, dense_output=False, random_state=0)
        rp = rp.fit(dense_data)
        # 断言transform后的稠密输入保持为稠密输出类型为numpy数组
        assert isinstance(rp.transform(dense_data), np.ndarray)

        # 断言transform后的稀疏输入保持为稀疏输出类型
        assert sp.issparse(rp.transform(sparse_data))


# 使用参数化测试框架pytest.mark.parametrize，对每个COO_CONTAINERS中的元素进行单独测试
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_correct_RandomProjection_dimensions_embedding(
    coo_container, global_random_seed
):
    # 使用make_sparse_random_data函数生成稀疏随机数据，格式为稠密数组（sparse_format=None）
    data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format=None,
    )
    # 遍历所有的随机投影器对象
    for RandomProjection in all_RandomProjection:
        # 使用"auto"作为组件数量自动调整，随机种子为0，阈值为0.5，对数据进行拟合
        rp = RandomProjection(n_components="auto", random_state=0, eps=0.5).fit(data)

        # 断言自动确定的组件数量为"auto"
        assert rp.n_components == "auto"
        # 断言实际确定的组件数量为110
        assert rp.n_components_ == 110

        # 如果当前随机投影器是稀疏随机投影器的一种
        if RandomProjection in all_SparseRandomProjection:
            # 断言稀疏度为"auto"
            assert rp.density == "auto"
            # 断言实际的稀疏度约为0.03（精确到小数点后两位）
            assert_almost_equal(rp.density_, 0.03, 2)

        # 断言投影后的组件形状为(110, n_features)
        assert rp.components_.shape == (110, n_features)

        # 对数据进行投影转换，断言投影后的形状为(n_samples, 110)
        projected_1 = rp.transform(data)
        assert projected_1.shape == (n_samples, 110)

        # 再次对同一数据进行投影转换，断言结果与第一次投影结果相同
        projected_2 = rp.transform(data)
        assert_array_equal(projected_1, projected_2)

        # 使用相同的随机种子进行拟合和转换，断言结果与之前的投影结果相同
        rp2 = RandomProjection(random_state=0, eps=0.5)
        projected_3 = rp2.fit_transform(data)
        assert_array_equal(projected_1, projected_3)

        # 尝试对与拟合数据大小不同的数据进行转换，断言会抛出 ValueError 异常
        with pytest.raises(ValueError):
            rp.transform(data[:, 1:5])

        # 还可以固定组件数量和稀疏度级别进行拟合和转换
        if RandomProjection in all_SparseRandomProjection:
            # 使用指定的组件数量、稀疏度和随机种子进行拟合
            rp = RandomProjection(n_components=100, density=0.001, random_state=0)
            projected = rp.fit_transform(data)
            # 断言投影后的形状为(n_samples, 100)
            assert projected.shape == (n_samples, 100)
            # 断言组件的形状为(100, n_features)
            assert rp.components_.shape == (100, n_features)
            # 断言组件的非零元素个数小于115（接近1%的稀疏度）
            assert rp.components_.nnz < 115
            # 断言组件的非零元素个数大于85（接近1%的稀疏度）
            assert 85 < rp.components_.nnz
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 使用pytest的parametrize装饰器，为COO_CONTAINERS中的每个元素分别运行测试函数
def test_warning_n_components_greater_than_n_features(
    coo_container, global_random_seed
):
    # 设置测试数据的特征数和样本数
    n_features = 20
    n_samples = 5
    # 计算稀疏数据中的非零元素个数
    n_nonzeros = int(n_features / 4)
    # 创建稀疏随机数据，使用make_sparse_random_data函数
    data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format=None,
    )

    # 对所有的RandomProjection类进行迭代
    for RandomProjection in all_RandomProjection:
        # 断言会触发DataDimensionalityWarning警告
        with pytest.warns(DataDimensionalityWarning):
            # 使用RandomProjection拟合数据，设置n_components为n_features + 1
            RandomProjection(n_components=n_features + 1).fit(data)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 使用pytest的parametrize装饰器，为COO_CONTAINERS中的每个元素分别运行测试函数
def test_works_with_sparse_data(coo_container, global_random_seed):
    # 设置测试数据的特征数和样本数
    n_features = 20
    n_samples = 5
    # 计算稀疏数据中的非零元素个数
    n_nonzeros = int(n_features / 4)
    # 创建稠密和稀疏随机数据，使用make_sparse_random_data函数
    dense_data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format=None,
    )
    sparse_data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format="csr",
    )

    # 对所有的RandomProjection类进行迭代
    for RandomProjection in all_RandomProjection:
        # 创建RandomProjection对象，分别拟合稠密和稀疏数据
        rp_dense = RandomProjection(n_components=3, random_state=1).fit(dense_data)
        rp_sparse = RandomProjection(n_components=3, random_state=1).fit(sparse_data)
        # 断言两者的结果在数值上几乎相等
        assert_array_almost_equal(
            densify(rp_dense.components_), densify(rp_sparse.components_)
        )


def test_johnson_lindenstrauss_min_dim():
    """Test Johnson-Lindenstrauss for small eps.

    Regression test for #17111: before #19374, 32-bit systems would fail.
    """
    # 测试Johnson-Lindenstrauss最小维度，用于小的eps值
    assert johnson_lindenstrauss_min_dim(100, eps=1e-5) == 368416070986


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 使用pytest的parametrize装饰器，为COO_CONTAINERS中的每个元素分别运行测试函数
@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
# 使用pytest的parametrize装饰器，为all_RandomProjection中的每个类分别运行测试函数
def test_random_projection_feature_names_out(
    coo_container, random_projection_cls, global_random_seed
):
    # 创建稀疏随机数据，使用make_sparse_random_data函数
    data = make_sparse_random_data(
        coo_container,
        n_samples,
        n_features,
        n_nonzeros,
        random_state=global_random_seed,
        sparse_format=None,
    )
    # 创建random_projection_cls类的实例，设置n_components为2
    random_projection = random_projection_cls(n_components=2)
    # 使用数据拟合random_projection对象
    random_projection.fit(data)
    # 获取生成的特征名列表
    names_out = random_projection.get_feature_names_out()
    # 获取random_projection_cls类的小写类名
    class_name_lower = random_projection_cls.__name__.lower()
    # 生成预期的特征名列表
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(random_projection.n_components_)],
        dtype=object,
    )

    # 断言生成的特征名列表与预期的特征名列表相等
    assert_array_equal(names_out, expected_names_out)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 使用pytest的parametrize装饰器，为COO_CONTAINERS中的每个元素分别运行测试函数
@pytest.mark.parametrize("n_samples", (2, 9, 10, 11, 1000))
# 使用pytest的parametrize装饰器，为n_samples中的每个元素分别运行测试函数
@pytest.mark.parametrize("n_features", (2, 9, 10, 11, 1000))
# 使用pytest的parametrize装饰器，为n_features中的每个元素分别运行测试函数
@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
# 使用pytest的parametrize装饰器，为all_RandomProjection中的每个类分别运行测试函数
@pytest.mark.parametrize("compute_inverse_components", [True, False])
# 使用pytest的parametrize装饰器，为compute_inverse_components中的每个元素分别运行测试函数
def test_inverse_transform(
    coo_container,                   # 定义变量 coo_container，通常用于存储稀疏矩阵的坐标形式数据
    n_samples,                       # 定义变量 n_samples，表示样本数量
    n_features,                      # 定义变量 n_features，表示特征数量
    random_projection_cls,           # 定义变量 random_projection_cls，表示随机投影的类或函数
    compute_inverse_components,      # 定义变量 compute_inverse_components，表示计算逆组件的函数或标志
    global_random_seed,              # 定义变量 global_random_seed，表示全局随机种子
# 设置随机投影的组件数量为10
n_components = 10

# 创建一个随机投影对象，设置参数包括组件数量、是否计算逆组件以及随机种子
random_projection = random_projection_cls(
    n_components=n_components,
    compute_inverse_components=compute_inverse_components,
    random_state=global_random_seed,
)

# 使用稠密格式生成稀疏随机数据
X_dense = make_sparse_random_data(
    coo_container,
    n_samples,
    n_features,
    n_nonzeros=n_samples * n_features // 100 + 1,
    random_state=global_random_seed,
    sparse_format=None,
)

# 使用CSR格式生成稀疏随机数据
X_csr = make_sparse_random_data(
    coo_container,
    n_samples,
    n_features,
    n_nonzeros=n_samples * n_features // 100 + 1,
    random_state=global_random_seed,
    sparse_format="csr",
)

# 对稠密和CSR格式的数据进行迭代处理
for X in [X_dense, X_csr]:
    # 忽略特定警告类别的警告信息
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The number of components is higher than the number of features"
            ),
            category=DataDimensionalityWarning,
        )
        # 对数据进行随机投影并获取投影结果
        projected = random_projection.fit_transform(X)

    # 如果需要计算逆组件，则进行以下断言
    if compute_inverse_components:
        assert hasattr(random_projection, "inverse_components_")
        inv_components = random_projection.inverse_components_
        assert inv_components.shape == (n_features, n_components)

    # 对投影结果进行逆变换，并断言逆变换后的形状与原始数据一致
    projected_back = random_projection.inverse_transform(projected)
    assert projected_back.shape == X.shape

    # 再次对逆变换后的数据进行投影，并进行数据一致性断言
    projected_again = random_projection.transform(projected_back)
    if hasattr(projected, "toarray"):
        projected = projected.toarray()
    assert_allclose(projected, projected_again, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
@pytest.mark.parametrize(
    "input_dtype, expected_dtype",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
# 测试随机投影类的数据类型匹配情况
def test_random_projection_dtype_match(
    random_projection_cls, input_dtype, expected_dtype
):
    # 验证输出矩阵的数据类型
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp = random_projection_cls(random_state=0)
    transformed = rp.fit_transform(X.astype(input_dtype))

    assert rp.components_.dtype == expected_dtype
    assert transformed.dtype == expected_dtype


@pytest.mark.parametrize("random_projection_cls", all_RandomProjection)
# 测试随机投影的数值一致性
def test_random_projection_numerical_consistency(random_projection_cls):
    # 验证在np.float32和np.float64之间的数值一致性
    atol = 1e-5
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp_32 = random_projection_cls(random_state=0)
    rp_64 = random_projection_cls(random_state=0)

    projection_32 = rp_32.fit_transform(X.astype(np.float32))
    projection_64 = rp_64.fit_transform(X.astype(np.float64))

    assert_allclose(projection_64, projection_32, atol=atol)

    # 对密集和稀疏投影组件进行数值一致性断言
    assert_allclose_dense_sparse(rp_32.components_, rp_64.components_)
```