# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_kd_tree.py`

```
# 导入必要的库和模块
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

# 从 sklearn.neighbors._kd_tree 模块中导入 KDTree 相关类
from sklearn.neighbors._kd_tree import KDTree, KDTree32, KDTree64
# 从 sklearn.neighbors.tests.test_ball_tree 模块中导入用于二叉树测试的数据集生成函数
from sklearn.neighbors.tests.test_ball_tree import get_dataset_for_binary_tree
# 从 sklearn.utils.parallel 中导入并行处理相关模块
from sklearn.utils.parallel import Parallel, delayed

# 设置数据维度常量
DIMENSION = 3

# 定义不同距离度量的字典
METRICS = {"euclidean": {}, "manhattan": {}, "chebyshev": {}, "minkowski": dict(p=3)}

# 定义要测试的 KDTree 类列表
KD_TREE_CLASSES = [
    KDTree64,
    KDTree32,
]

# 测试 KDTree 是否是 KDTree64 的子类
def test_KDTree_is_KDTree64_subclass():
    assert issubclass(KDTree, KDTree64)

# 使用参数化测试对 KDTree 类型进行测试
@pytest.mark.parametrize("BinarySearchTree", KD_TREE_CLASSES)
def test_array_object_type(BinarySearchTree):
    """Check that we do not accept object dtype array."""
    # 创建一个对象类型为 object 的 NumPy 数组，验证不能接受这种类型
    X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        BinarySearchTree(X)

# 使用参数化测试检查 KDTree 是否可以与 joblib 进行序列化
@pytest.mark.parametrize("BinarySearchTree", KD_TREE_CLASSES)
def test_kdtree_picklable_with_joblib(BinarySearchTree):
    """Make sure that KDTree queries work when joblib memmaps.

    Non-regression test for #21685 and #21228."""
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3))
    # 创建 KDTree 对象
    tree = BinarySearchTree(X, leaf_size=2)

    # 使用 Parallel 并行处理，设置 max_nbytes=1 触发只读内存映射
    # 这个过程曾在之前的 Cython 代码版本中引发 "ValueError: buffer source array is read-only" 错误
    Parallel(n_jobs=2, max_nbytes=1)(delayed(tree.query)(data) for data in 2 * [X])

# 使用参数化测试检查 KDTree 在不同数据类型和度量下的数值一致性
@pytest.mark.parametrize("metric", METRICS)
def test_kd_tree_numerical_consistency(global_random_seed, metric):
    # 获取用于二叉树测试的数据集，包括 float64 和 float32 版本的特征数据和标签数据
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(
        random_seed=global_random_seed, features=50
    )

    # 根据指定度量创建 KDTree64 和 KDTree32 对象
    metric_params = METRICS.get(metric, {})
    kd_64 = KDTree64(X_64, leaf_size=2, metric=metric, **metric_params)
    kd_32 = KDTree32(X_32, leaf_size=2, metric=metric, **metric_params)

    # 测试与 query 方法相关的一致性
    k = 4
    dist_64, ind_64 = kd_64.query(Y_64, k=k)
    dist_32, ind_32 = kd_32.query(Y_32, k=k)
    assert_allclose(dist_64, dist_32, rtol=1e-5)  # 检查距离数组是否数值上接近
    assert_equal(ind_64, ind_32)  # 检查索引数组是否完全一致
    assert dist_64.dtype == np.float64  # 检查距离数组的数据类型是否为 float64
    assert dist_32.dtype == np.float32  # 检查距离数组的数据类型是否为 float32

    # 测试与 query_radius 方法相关的一致性
    r = 2.38
    ind_64 = kd_64.query_radius(Y_64, r=r)
    ind_32 = kd_32.query_radius(Y_32, r=r)
    for _ind64, _ind32 in zip(ind_64, ind_32):
        assert_equal(_ind64, _ind32)

    # 测试与 query_radius 方法相关的一致性，同时返回距离
    ind_64, dist_64 = kd_64.query_radius(Y_64, r=r, return_distance=True)
    ind_32, dist_32 = kd_32.query_radius(Y_32, r=r, return_distance=True)
    # 使用 zip 函数同时迭代四个列表 ind_64, ind_32, dist_64, dist_32 中的对应元素
    for _ind64, _ind32, _dist_64, _dist_32 in zip(ind_64, ind_32, dist_64, dist_32):
        # 断言 _ind64 和 _ind32 的值相等，如果不相等则引发 AssertionError
        assert_equal(_ind64, _ind32)
        # 断言 _dist_64 和 _dist_32 的值在给定的相对容差 1e-5 范围内近似相等
        assert_allclose(_dist_64, _dist_32, rtol=1e-5)
        # 断言 _dist_64 的数据类型是 np.float64
        assert _dist_64.dtype == np.float64
        # 断言 _dist_32 的数据类型是 np.float32
        assert _dist_32.dtype == np.float32
@pytest.mark.parametrize("metric", METRICS)
def test_kernel_density_numerical_consistency(global_random_seed, metric):
    # 使用参数化测试，对每个指标(metric)执行以下测试
    # 测试与 `kernel_density` 方法的一致性

    # 从数据集中获取 64 位和 32 位版本的数据
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)

    # 根据指定的指标(metric)和参数，创建 KD 树的 64 位版本
    metric_params = METRICS.get(metric, {})
    kd_64 = KDTree64(X_64, leaf_size=2, metric=metric, **metric_params)

    # 根据指定的指标(metric)和参数，创建 KD 树的 32 位版本
    kd_32 = KDTree32(X_32, leaf_size=2, metric=metric, **metric_params)

    # 设置核函数为高斯核，带宽为 h
    kernel = "gaussian"
    h = 0.1

    # 使用 KD 树的 64 位版本计算密度估计
    density64 = kd_64.kernel_density(Y_64, h=h, kernel=kernel, breadth_first=True)

    # 使用 KD 树的 32 位版本计算密度估计
    density32 = kd_32.kernel_density(Y_32, h=h, kernel=kernel, breadth_first=True)

    # 断言两个密度估计结果应该非常接近，相对误差小于 1e-5
    assert_allclose(density64, density32, rtol=1e-5)

    # 断言密度估计的数据类型应该分别为 np.float64 和 np.float32
    assert density64.dtype == np.float64
    assert density32.dtype == np.float32
```