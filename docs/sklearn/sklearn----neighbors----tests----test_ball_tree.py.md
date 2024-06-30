# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_ball_tree.py`

```
import itertools  # 导入 itertools 库，用于生成迭代器的工具函数

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal  # 导入 NumPy 测试相关的断言函数

from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64  # 导入 BallTree 相关类
from sklearn.utils import check_random_state  # 导入随机状态检查函数
from sklearn.utils._testing import _convert_container  # 导入转换容器的辅助函数
from sklearn.utils.validation import check_array  # 导入数组验证函数

rng = np.random.RandomState(10)  # 创建一个随机种子为10的随机数生成器对象
V_mahalanobis = rng.rand(3, 3)  # 生成一个3x3的随机数矩阵
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)  # 计算矩阵的转置乘积，生成协方差矩阵

DIMENSION = 3  # 设置维度为3

METRICS = {  # 定义一个包含多种距离度量的字典
    "euclidean": {},  # 欧氏距离
    "manhattan": {},  # 曼哈顿距离
    "minkowski": dict(p=3),  # 闵可夫斯基距离，参数p=3
    "chebyshev": {},  # 切比雪夫距离
}

DISCRETE_METRICS = ["hamming", "canberra", "braycurtis"]  # 定义一组离散距离度量

BOOLEAN_METRICS = [  # 定义一组布尔距离度量
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
]

BALL_TREE_CLASSES = [  # 定义一个包含 BallTree 类型的列表
    BallTree64,  # 64位 BallTree 类
    BallTree32,  # 32位 BallTree 类
]


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    """使用暴力方法计算最近邻。

    Args:
        X (array-like): 第一个数组
        Y (array-like): 第二个数组
        k (int): 最近邻的数量
        metric (str): 距离度量名称
        **kwargs: 其他参数传递给距离度量函数

    Returns:
        dist (ndarray): 计算得到的距离数组
        ind (ndarray): 最近邻的索引数组
    """
    from sklearn.metrics import DistanceMetric  # 导入距离度量函数

    X, Y = check_array(X), check_array(Y)  # 确保输入数组是合法的 NumPy 数组
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)  # 使用指定的距离度量计算 Y 和 X 之间的距离矩阵
    ind = np.argsort(D, axis=1)[:, :k]  # 对距离矩阵按照距离大小进行排序，获取最近的 k 个邻居的索引
    dist = D[np.arange(Y.shape[0])[:, None], ind]  # 根据索引获取对应的距离值
    return dist, ind  # 返回计算得到的距离和索引数组


def test_BallTree_is_BallTree64_subclass():
    """测试 BallTree 是否是 BallTree64 的子类。"""
    assert issubclass(BallTree, BallTree64)


@pytest.mark.parametrize("metric", itertools.chain(BOOLEAN_METRICS, DISCRETE_METRICS))
@pytest.mark.parametrize("array_type", ["list", "array"])
@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_ball_tree_query_metrics(metric, array_type, BallTreeImplementation):
    """测试 BallTree 查询功能的距离度量。

    Args:
        metric (str): 距离度量名称
        array_type (str): 数组类型，可以是 'list' 或 'array'
        BallTreeImplementation (class): BallTree 类的具体实现
    """
    rng = check_random_state(0)  # 创建随机状态对象

    if metric in BOOLEAN_METRICS:
        X = rng.random_sample((40, 10)).round(0)  # 生成随机二进制数组 X
        Y = rng.random_sample((10, 10)).round(0)  # 生成随机二进制数组 Y
    elif metric in DISCRETE_METRICS:
        X = (4 * rng.random_sample((40, 10))).round(0)  # 生成随机离散数组 X
        Y = (4 * rng.random_sample((10, 10))).round(0)  # 生成随机离散数组 Y

    X = _convert_container(X, array_type)  # 转换 X 到指定的数组类型
    Y = _convert_container(Y, array_type)  # 转换 Y 到指定的数组类型

    k = 5  # 设置最近邻的数量为5

    bt = BallTreeImplementation(X, leaf_size=1, metric=metric)  # 创建 BallTree 实例
    dist1, ind1 = bt.query(Y, k)  # 使用 BallTree 查询 Y 中每个点的最近邻
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric)  # 使用暴力方法计算最近邻

    assert_array_almost_equal(dist1, dist2)  # 断言 BallTree 查询结果和暴力计算结果的距离数组近似相等


@pytest.mark.parametrize(
    "BallTreeImplementation, decimal_tol", zip(BALL_TREE_CLASSES, [6, 5])
)
def test_query_haversine(BallTreeImplementation, decimal_tol):
    """测试 BallTree 在球面距离度量（haversine）下的查询功能。

    Args:
        BallTreeImplementation (class): BallTree 类的具体实现
        decimal_tol (int): 比较结果时的小数位数精度
    """
    rng = check_random_state(0)  # 创建随机状态对象
    X = 2 * np.pi * rng.random_sample((40, 2))  # 生成随机的球面坐标数组 X

    bt = BallTreeImplementation(X, leaf_size=1, metric="haversine")  # 创建 BallTree 实例
    dist1, ind1 = bt.query(X, k=5)  # 使用 BallTree 查询球面坐标 X 的最近邻
    dist2, ind2 = brute_force_neighbors(X, X, k=5, metric="haversine")  # 使用暴力方法计算最近邻

    assert_array_almost_equal(dist1, dist2, decimal=decimal_tol)  # 断言 BallTree 查询结果和暴力计算结果的距离数组近似相等
    assert_array_almost_equal(ind1, ind2)  # 断言 BallTree 查询结果和暴力计算结果的索引数组相等


@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_array_object_type(BallTreeImplementation):
    """检查是否不接受对象类型的数组。"""
    X = np.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)  # 创建包含对象的数组 X
    # 使用 pytest 模块进行测试，期望捕获 ValueError 异常，并匹配指定的错误信息字符串 "setting an array element with a sequence"
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        # 调用 BallTreeImplementation 类的构造函数，传入参数 X，测试是否会引发期望的异常
        BallTreeImplementation(X)
@pytest.mark.parametrize("BallTreeImplementation", BALL_TREE_CLASSES)
def test_bad_pyfunc_metric(BallTreeImplementation):
    # 定义一个错误的距离函数，总是返回字符串 "1"
    def wrong_returned_value(x, y):
        return "1"

    # 定义一个只接受一个参数的函数，但是被期望接受两个参数
    def one_arg_func(x):
        return 1.0  # pragma: no cover

    # 创建一个形状为 (5, 2) 的全为 1 的 NumPy 数组
    X = np.ones((5, 2))
    # 设置错误消息，用于检查是否抛出 TypeError 异常
    msg = "Custom distance function must accept two vectors and return a float."
    # 检查是否在使用错误距离函数时抛出了 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=wrong_returned_value)

    # 设置错误消息，用于检查是否抛出 TypeError 异常
    msg = "takes 1 positional argument but 2 were given"
    # 检查是否在使用只接受一个参数的函数时抛出了 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=one_arg_func)


@pytest.mark.parametrize("metric", itertools.chain(METRICS, BOOLEAN_METRICS))
def test_ball_tree_numerical_consistency(global_random_seed, metric):
    # 对于二叉树的 float64 和 float32 版本，结果必须在数值上非常接近
    # 获取用于二叉树的数据集，其中包含 float64 和 float32 版本
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(
        random_seed=global_random_seed, features=50
    )

    # 获取指定距离度量的参数
    metric_params = METRICS.get(metric, {})
    # 创建一个使用 float64 的 BallTree 对象
    bt_64 = BallTree64(X_64, leaf_size=1, metric=metric, **metric_params)
    # 创建一个使用 float32 的 BallTree 对象
    bt_32 = BallTree32(X_32, leaf_size=1, metric=metric, **metric_params)

    # 测试在 `query` 方法上的一致性
    k = 5
    dist_64, ind_64 = bt_64.query(Y_64, k=k)
    dist_32, ind_32 = bt_32.query(Y_32, k=k)
    assert_allclose(dist_64, dist_32, rtol=1e-5)
    assert_equal(ind_64, ind_32)
    assert dist_64.dtype == np.float64
    assert dist_32.dtype == np.float32

    # 测试在 `query_radius` 方法上的一致性
    r = 2.38
    ind_64 = bt_64.query_radius(Y_64, r=r)
    ind_32 = bt_32.query_radius(Y_32, r=r)
    for _ind64, _ind32 in zip(ind_64, ind_32):
        assert_equal(_ind64, _ind32)

    # 测试在 `query_radius` 方法上的一致性，且返回距离信息
    ind_64, dist_64 = bt_64.query_radius(Y_64, r=r, return_distance=True)
    ind_32, dist_32 = bt_32.query_radius(Y_32, r=r, return_distance=True)
    for _ind64, _ind32, _dist_64, _dist_32 in zip(ind_64, ind_32, dist_64, dist_32):
        assert_equal(_ind64, _ind32)
        assert_allclose(_dist_64, _dist_32, rtol=1e-5)
        assert _dist_64.dtype == np.float64
        assert _dist_32.dtype == np.float32


@pytest.mark.parametrize("metric", itertools.chain(METRICS, BOOLEAN_METRICS))
def test_kernel_density_numerical_consistency(global_random_seed, metric):
    # 测试在 `kernel_density` 方法上的一致性
    # 获取用于二叉树的数据集，其中包含 float64 和 float32 版本
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)

    # 获取指定距离度量的参数
    metric_params = METRICS.get(metric, {})
    # 创建一个使用 float64 的 BallTree 对象
    bt_64 = BallTree64(X_64, leaf_size=1, metric=metric, **metric_params)
    # 创建一个使用 float32 的 BallTree 对象
    bt_32 = BallTree32(X_32, leaf_size=1, metric=metric, **metric_params)

    # 设置核函数和带宽参数
    kernel = "gaussian"
    h = 0.1
    # 计算使用 float64 BallTree 对象的核密度估计
    density64 = bt_64.kernel_density(Y_64, h=h, kernel=kernel, breadth_first=True)
    # 计算使用 float32 BallTree 对象的核密度估计
    density32 = bt_32.kernel_density(Y_32, h=h, kernel=kernel, breadth_first=True)
    # 使用 assert_allclose 函数检查两个密度数组 density64 和 density32 是否在相对容差 1e-5 内接近
    assert_allclose(density64, density32, rtol=1e-5)
    
    # 检查 density64 的数据类型是否为 np.float64
    assert density64.dtype == np.float64
    
    # 检查 density32 的数据类型是否为 np.float32
    assert density32.dtype == np.float32
# 测试数值一致性，使用 `two_point_correlation` 方法验证
def test_two_point_correlation_numerical_consistency(global_random_seed):
    # 获取二叉树数据集，使用全局随机种子
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)

    # 创建使用 BallTree64 类型的 BallTree 对象，叶子大小为 10
    bt_64 = BallTree64(X_64, leaf_size=10)
    # 创建使用 BallTree32 类型的 BallTree 对象，叶子大小为 10
    bt_32 = BallTree32(X_32, leaf_size=10)

    # 在 [0, 1] 范围内生成 10 个等间距的值作为半径 r
    r = np.linspace(0, 1, 10)

    # 计算 Y_64 对 X_64 的两点相关性，使用双树方法
    counts_64 = bt_64.two_point_correlation(Y_64, r=r, dualtree=True)
    # 计算 Y_32 对 X_32 的两点相关性，使用双树方法
    counts_32 = bt_32.two_point_correlation(Y_32, r=r, dualtree=True)

    # 断言两者计算结果的数值应该非常接近
    assert_allclose(counts_64, counts_32)


def get_dataset_for_binary_tree(random_seed, features=3):
    # 使用给定的随机种子创建随机数生成器对象
    rng = np.random.RandomState(random_seed)
    # 生成一个 shape 为 (100, features) 的随机浮点数数组 _X
    _X = rng.rand(100, features)
    # 生成一个 shape 为 (5, features) 的随机浮点数数组 _Y
    _Y = rng.rand(5, features)

    # 将 _X 转换为 float64 类型，共享存储以节省内存
    X_64 = _X.astype(dtype=np.float64, copy=False)
    # 将 _Y 转换为 float64 类型，共享存储以节省内存
    Y_64 = _Y.astype(dtype=np.float64, copy=False)

    # 将 _X 转换为 float32 类型，共享存储以节省内存
    X_32 = _X.astype(dtype=np.float32, copy=False)
    # 将 _Y 转换为 float32 类型，共享存储以节省内存
    Y_32 = _Y.astype(dtype=np.float32, copy=False)

    # 返回四个数据集：X_64, X_32, Y_64, Y_32
    return X_64, X_32, Y_64, Y_32
```