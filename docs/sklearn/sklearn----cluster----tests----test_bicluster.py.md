# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_bicluster.py`

```
# 导入必要的库和模块
import numpy as np
import pytest
from scipy.sparse import issparse

# 导入所需的类和函数
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
    _bistochastic_normalize,
    _log_normalize,
    _scale_normalize,
)
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 创建一个模拟的双向聚类对象，用于测试 get_submatrix 方法
class MockBiclustering(BiclusterMixin, BaseEstimator):
    def __init__(self):
        pass

    # 重写 get_indices 方法以重现旧的 get_submatrix 测试
    def get_indices(self, i):
        return (
            np.where([True, True, False, False, True])[0],  # 返回行索引
            np.where([False, False, True, True])[0],        # 返回列索引
        )

# 使用 CSR 容器作为参数化测试的参数，测试 get_submatrix 方法
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_get_submatrix(csr_container):
    # 创建一个数据集
    data = np.arange(20).reshape(5, 4)
    model = MockBiclustering()  # 创建模拟的双向聚类对象

    # 对不同类型的输入进行测试
    for X in (data, csr_container(data), data.tolist()):
        submatrix = model.get_submatrix(0, X)  # 获取子矩阵
        if issparse(submatrix):
            submatrix = submatrix.toarray()  # 如果是稀疏矩阵，转换为密集矩阵
        assert_array_equal(submatrix, [[2, 3], [6, 7], [18, 19]])  # 断言子矩阵的值符合预期
        submatrix[:] = -1  # 修改子矩阵的值为 -1
        if issparse(X):
            X = X.toarray()  # 如果输入是稀疏矩阵，转换为密集矩阵
        assert np.all(X != -1)  # 断言原始数据中没有 -1 的值

# 测试已拟合模型上的 get_shape 和 get_indices 方法
def _test_shape_indices(model):
    for i in range(model.n_clusters):
        m, n = model.get_shape(i)  # 获取聚类 i 的形状信息
        i_ind, j_ind = model.get_indices(i)  # 获取聚类 i 的行列索引
        assert len(i_ind) == m  # 断言行索引的长度与 m 相符
        assert len(j_ind) == n  # 断言列索引的长度与 n 相符

# 使用 CSR 容器作为参数化测试的参数，测试 Dhillon 的谱共聚类算法
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_coclustering(global_random_seed, csr_container):
    # 定义参数网格
    param_grid = {
        "svd_method": ["randomized", "arpack"],
        "n_svd_vecs": [None, 20],
        "mini_batch": [False, True],
        "init": ["k-means++"],
        "n_init": [10],
    }
    # 创建双向聚类数据集
    S, rows, cols = make_biclusters(
        (30, 30), 3, noise=0.1, random_state=global_random_seed
    )
    S -= S.min()  # 确保非负性
    S = np.where(S < 1, 0, S)  # 对某些值进行阈值处理
    # 对于每个输入矩阵 S 及其压缩稀疏行表示 (csr_container(S)) 进行迭代
    for mat in (S, csr_container(S)):
        # 遍历参数网格中的所有组合
        for kwargs in ParameterGrid(param_grid):
            # 创建 SpectralCoclustering 模型实例，设定簇数为 3，随机种子为全局设定值，同时传入参数 kwargs
            model = SpectralCoclustering(
                n_clusters=3, random_state=global_random_seed, **kwargs
            )
            # 使用当前矩阵 mat 对模型进行拟合
            model.fit(mat)

            # 断言模型的行聚类形状为 (3, 30)
            assert model.rows_.shape == (3, 30)
            # 断言模型的行聚类的每列和为全为 1
            assert_array_equal(model.rows_.sum(axis=0), np.ones(30))
            # 断言模型的列聚类的每列和为全为 1
            assert_array_equal(model.columns_.sum(axis=0), np.ones(30))
            # 断言模型的双向聚类的共识分数为 1
            assert consensus_score(model.biclusters_, (rows, cols)) == 1

            # 调用 _test_shape_indices 函数，传入当前模型进行额外的形状索引测试
            _test_shape_indices(model)
# 使用参数化测试，对 CSR_CONTAINERS 中的每个容器执行测试函数 test_spectral_biclustering
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_spectral_biclustering(global_random_seed, csr_container):
    # 在一个棋盘数据集上测试 Kluger 方法
    S, rows, cols = make_checkerboard(
        (30, 30), 3, noise=0.5, random_state=global_random_seed
    )

    # 定义非默认参数字典
    non_default_params = {
        "method": ["scale", "log"],
        "svd_method": ["arpack"],
        "n_svd_vecs": [20],
        "mini_batch": [True],
    }

    # 遍历数据集 S 和 csr_container(S)
    for mat in (S, csr_container(S)):
        # 遍历非默认参数字典中的参数名和对应的值列表
        for param_name, param_values in non_default_params.items():
            for param_value in param_values:
                # 创建 SpectralBiclustering 模型对象
                model = SpectralBiclustering(
                    n_clusters=3,
                    n_init=3,
                    init="k-means++",
                    random_state=global_random_seed,
                )
                # 设置模型参数为当前参数名和值
                model.set_params(**dict([(param_name, param_value)]))

                # 如果 mat 是稀疏矩阵，并且模型的 method 参数为 "log"
                if issparse(mat) and model.get_params().get("method") == "log":
                    # 无法对稀疏矩阵取对数，预期会抛出 ValueError 异常
                    with pytest.raises(ValueError):
                        model.fit(mat)
                    continue
                else:
                    # 否则，对模型进行拟合
                    model.fit(mat)

                # 断言模型的行形状和列形状
                assert model.rows_.shape == (9, 30)
                assert model.columns_.shape == (9, 30)
                # 断言模型的行和列的和
                assert_array_equal(model.rows_.sum(axis=0), np.repeat(3, 30))
                assert_array_equal(model.columns_.sum(axis=0), np.repeat(3, 30))
                # 断言模型的一致性分数
                assert consensus_score(model.biclusters_, (rows, cols)) == 1

                # 调用 _test_shape_indices 函数对模型进行额外的测试
                _test_shape_indices(model)


# 定义函数 _do_scale_test，用于检查行和列的和是否为常数
def _do_scale_test(scaled):
    """Check that rows sum to one constant, and columns to another."""
    row_sum = scaled.sum(axis=1)
    col_sum = scaled.sum(axis=0)
    if issparse(scaled):
        row_sum = np.asarray(row_sum).squeeze()
        col_sum = np.asarray(col_sum).squeeze()
    assert_array_almost_equal(row_sum, np.tile(row_sum.mean(), 100), decimal=1)
    assert_array_almost_equal(col_sum, np.tile(col_sum.mean(), 100), decimal=1)


# 定义函数 _do_bistochastic_test，用于检查行和列的和是否相等
def _do_bistochastic_test(scaled):
    """Check that rows and columns sum to the same constant."""
    _do_scale_test(scaled)
    assert_almost_equal(scaled.sum(axis=0).mean(), scaled.sum(axis=1).mean(), decimal=1)


# 使用参数化测试，对 CSR_CONTAINERS 中的每个容器执行测试函数 test_scale_normalize
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_scale_normalize(global_random_seed, csr_container):
    generator = np.random.RandomState(global_random_seed)
    X = generator.rand(100, 100)
    # 遍历数据集 X 和 csr_container(X)
    for mat in (X, csr_container(X)):
        # 对 mat 进行尺度归一化，并获取结果
        scaled, _, _ = _scale_normalize(mat)
        # 调用 _do_scale_test 函数对尺度归一化后的结果进行测试
        _do_scale_test(scaled)
        # 如果 mat 是稀疏矩阵，则断言 scaled 也是稀疏矩阵
        if issparse(mat):
            assert issparse(scaled)


# 使用参数化测试，对 CSR_CONTAINERS 中的每个容器执行测试函数 test_bistochastic_normalize
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_bistochastic_normalize(global_random_seed, csr_container):
    generator = np.random.RandomState(global_random_seed)
    X = generator.rand(100, 100)
    # 对于输入的每个对象 mat 和 csr_container(X)，依次执行以下操作：
    
    # 调用 _bistochastic_normalize 函数，对 mat 进行双随机规范化，并将结果赋给 scaled 变量
    scaled = _bistochastic_normalize(mat)
    
    # 调用 _do_bistochastic_test 函数，对 scaled 进行双随机性测试
    _do_bistochastic_test(scaled)
    
    # 如果 mat 是稀疏矩阵（sparse matrix），则进行以下断言检查：
    if issparse(mat):
        # 确保 scaled 也是稀疏矩阵
        assert issparse(scaled)
def test_log_normalize(global_random_seed):
    # 使用全局随机种子创建随机数生成器
    generator = np.random.RandomState(global_random_seed)
    # 生成一个 100x100 的随机矩阵
    mat = generator.rand(100, 100)
    # 对矩阵进行对数归一化，并加上常数 1，使其成为双随机矩阵
    scaled = _log_normalize(mat) + 1
    # 对归一化后的矩阵进行双随机性测试
    _do_bistochastic_test(scaled)


def test_fit_best_piecewise(global_random_seed):
    # 使用全局随机种子创建 SpectralBiclustering 模型实例
    model = SpectralBiclustering(random_state=global_random_seed)
    # 创建一个示例向量数组
    vectors = np.array([[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5]])
    # 使用模型的内部方法 _fit_best_piecewise 对示例向量数组进行分段拟合
    best = model._fit_best_piecewise(vectors, n_best=2, n_clusters=2)
    # 断言拟合结果与预期的前两个向量相等
    assert_array_equal(best, vectors[:2])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_project_and_cluster(global_random_seed, csr_container):
    # 使用全局随机种子创建 SpectralBiclustering 模型实例
    model = SpectralBiclustering(random_state=global_random_seed)
    # 创建示例数据数组
    data = np.array([[1, 1, 1], [1, 1, 1], [3, 6, 3], [3, 6, 3]])
    # 创建示例向量数组
    vectors = np.array([[1, 0], [0, 1], [0, 0]])
    # 遍历数据数组及其对应的 csr_container（稀疏矩阵表示），并使用模型的内部方法 _project_and_cluster 进行投影聚类
    for mat in (data, csr_container(data)):
        labels = model._project_and_cluster(mat, vectors, n_clusters=2)
        # 断言聚类结果的 v_measure_score 与预期的相似度评分为 1.0
        assert_almost_equal(v_measure_score(labels, [0, 0, 1, 1]), 1.0)


def test_perfect_checkerboard(global_random_seed):
    # XXX 先前在构建机器人上失败（不可重现）
    # 使用全局随机种子创建 SpectralBiclustering 模型实例
    model = SpectralBiclustering(
        3, svd_method="arpack", random_state=global_random_seed
    )

    # 创建完美棋盘模式的数据 S，以及行和列索引
    S, rows, cols = make_checkerboard(
        (30, 30), 3, noise=0, random_state=global_random_seed
    )
    # 使用数据 S 拟合模型
    model.fit(S)
    # 断言模型的双向聚类结果与预期的行列索引完全一致
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    # 创建不同参数下的完美棋盘模式数据 S，以及行和列索引
    S, rows, cols = make_checkerboard(
        (40, 30), 3, noise=0, random_state=global_random_seed
    )
    # 使用数据 S 拟合模型
    model.fit(S)
    # 断言模型的双向聚类结果与预期的行列索引完全一致
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    # 创建不同参数下的完美棋盘模式数据 S，以及行和列索引
    S, rows, cols = make_checkerboard(
        (30, 40), 3, noise=0, random_state=global_random_seed
    )
    # 使用数据 S 拟合模型
    model.fit(S)
    # 断言模型的双向聚类结果与预期的行列索引完全一致
    assert consensus_score(model.biclusters_, (rows, cols)) == 1


@pytest.mark.parametrize(
    "params, type_err, err_msg",
    [
        (
            {"n_clusters": 6},
            ValueError,
            "n_clusters should be <= n_samples=5",
        ),
        (
            {"n_clusters": (3, 3, 3)},
            ValueError,
            "Incorrect parameter n_clusters",
        ),
        (
            {"n_clusters": (3, 6)},
            ValueError,
            "Incorrect parameter n_clusters",
        ),
        (
            {"n_components": 3, "n_best": 4},
            ValueError,
            "n_best=4 must be <= n_components=3",
        ),
    ],
)
def test_spectralbiclustering_parameter_validation(params, type_err, err_msg):
    """检查 SpectralBiClustering 中的参数验证"""
    # 创建一个简单的数据数组
    data = np.arange(25).reshape((5, 5))
    # 使用给定的参数创建 SpectralBiClustering 模型实例
    model = SpectralBiclustering(**params)
    # 断言在模型拟合数据时抛出指定类型的异常，并匹配给定的错误信息
    with pytest.raises(type_err, match=err_msg):
        model.fit(data)


@pytest.mark.parametrize("est", (SpectralBiclustering(), SpectralCoclustering()))
def test_n_features_in_(est):
    # 创建一个模拟的特征数据 X，以及对应的行和列索引
    X, _, _ = make_biclusters((3, 3), 3, random_state=0)
    # 检查 `est` 对象是否没有属性 "n_features_in_"，断言应为 True
    assert not hasattr(est, "n_features_in_")
    # 使用输入数据 `X` 对估计器 `est` 进行拟合
    est.fit(X)
    # 断言 `est` 对象现在应该具有属性 "n_features_in_"，并且其值应为 3
    assert est.n_features_in_ == 3
```