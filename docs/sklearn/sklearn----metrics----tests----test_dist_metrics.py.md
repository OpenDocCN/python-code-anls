# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_dist_metrics.py`

```
# 导入所需的库
import copy
import itertools
import pickle

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from scipy.spatial.distance import cdist  # 从 SciPy 库中导入 cdist 函数，用于计算距离

from sklearn.metrics import DistanceMetric  # 导入距离度量类
from sklearn.metrics._dist_metrics import (  # 导入距离度量相关模块
    BOOL_METRICS,
    DistanceMetric32,
    DistanceMetric64,
)
from sklearn.utils import check_random_state  # 导入用于检查随机状态的工具函数
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data  # 导入用于测试和创建内存映射数据的函数
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version  # 导入用于兼容性修复和版本解析的函数

# 定义计算距离的函数
def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1.0 / p)

# 设置随机数种子
rng = check_random_state(0)
d = 4  # 数据维度
n1 = 20  # 第一个数据集样本数
n2 = 25  # 第二个数据集样本数

# 创建浮点型数据集 X64 和 Y64，数据类型为 float64
X64 = rng.random_sample((n1, d))
Y64 = rng.random_sample((n2, d))

# 将 X64 和 Y64 转换为 float32 类型的数据集 X32 和 Y32
X32 = X64.astype("float32")
Y32 = Y64.astype("float32")

# 使用 create_memmap_backed_data 函数创建内存映射数据 X_mmap 和 Y_mmap
[X_mmap, Y_mmap] = create_memmap_backed_data([X64, Y64])

# 根据阈值创建布尔型数据集 X_bool 和 Y_bool，数据类型为 float64
X_bool = (X64 < 0.3).astype(np.float64)  # 数据比较稀疏
Y_bool = (Y64 < 0.7).astype(np.float64)  # 数据不太稀疏

# 使用 create_memmap_backed_data 函数创建布尔型内存映射数据 X_bool_mmap 和 Y_bool_mmap
[X_bool_mmap, Y_bool_mmap] = create_memmap_backed_data([X_bool, Y_bool])

# 生成随机矩阵 V，计算其转置乘积得到 VI
V = rng.random_sample((d, d))
VI = np.dot(V, V.T)

# 设置默认参数的距离度量列表 METRICS_DEFAULT_PARAMS
METRICS_DEFAULT_PARAMS = [
    ("euclidean", {}),
    ("cityblock", {}),
    ("minkowski", dict(p=(0.5, 1, 1.5, 2, 3))),
    ("chebyshev", {}),
    ("seuclidean", dict(V=(rng.random_sample(d),))),
    ("mahalanobis", dict(VI=(VI,))),
    ("hamming", {}),
    ("canberra", {}),
    ("braycurtis", {}),
    ("minkowski", dict(p=(0.5, 1, 1.5, 3), w=(rng.random_sample(d),))),
]

# 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数
@pytest.mark.parametrize(
    "metric_param_grid", METRICS_DEFAULT_PARAMS, ids=lambda params: params[0]
)
@pytest.mark.parametrize("X, Y", [(X64, Y64), (X32, Y32), (X_mmap, Y_mmap)])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数 test_cdist，接收距离度量参数、数据集 X、Y 和 CSR 容器
def test_cdist(metric_param_grid, X, Y, csr_container):
    metric, param_grid = metric_param_grid  # 解包距离度量参数和其参数网格
    keys = param_grid.keys()  # 获取参数网格的键

    X_csr, Y_csr = csr_container(X), csr_container(Y)  # 将 X 和 Y 转换为 CSR 格式
    # 使用 itertools.product 生成 param_grid 中所有参数组合的迭代器
    for vals in itertools.product(*param_grid.values()):
        # 将参数组合与参数名称对应，创建关键字参数字典
        kwargs = dict(zip(keys, vals))
        # 初始化相对容差字典
        rtol_dict = {}
        
        # 如果距离度量为 "mahalanobis" 并且 X 的数据类型为 np.float32
        if metric == "mahalanobis" and X.dtype == np.float32:
            # 计算马哈拉诺比斯距离在 scipy 和 scikit-learn 实现之间的差异
            # 因此，增加相对容差
            # TODO: 检查 scipy 中轻微的数值差异
            rtol_dict = {"rtol": 1e-6}

        # TODO: 当 scipy 最小版本 >= 1.7.0 时移除
        # 对于 minkowski 距离度量
        if metric == "minkowski":
            p = kwargs["p"]
            if sp_version < parse_version("1.7.0") and p < 1:
                # 如果 scipy 不支持 0 < p < 1 的 minkowski 距离度量
                pytest.skip("scipy does not support 0<p<1 for minkowski metric < 1.7.0")

        # 使用 cdist 计算 X 和 Y 之间的距离矩阵 D_scipy_cdist
        D_scipy_cdist = cdist(X, Y, metric, **kwargs)

        # 使用 DistanceMetric.get_metric 创建距离度量对象 dm
        dm = DistanceMetric.get_metric(metric, X.dtype, **kwargs)

        # 使用 dm.pairwise 计算稠密格式下 X 和 Y 的距离矩阵 D_sklearn
        D_sklearn = dm.pairwise(X, Y)
        # 断言确保 D_sklearn 是 C 连续的
        assert D_sklearn.flags.c_contiguous
        # 断言确保 D_sklearn 与 D_scipy_cdist 的值接近，根据 rtol_dict 指定的容差
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)

        # 使用 dm.pairwise 计算稀疏格式下 X_csr 和 Y_csr 的距离矩阵 D_sklearn
        D_sklearn = dm.pairwise(X_csr, Y_csr)
        # 断言确保 D_sklearn 是 C 连续的
        assert D_sklearn.flags.c_contiguous
        # 断言确保 D_sklearn 与 D_scipy_cdist 的值接近，根据 rtol_dict 指定的容差
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)

        # 使用 dm.pairwise 计算稀疏格式下 X_csr 和 稠密格式下 Y 的距离矩阵 D_sklearn
        D_sklearn = dm.pairwise(X_csr, Y)
        # 断言确保 D_sklearn 是 C 连续的
        assert D_sklearn.flags.c_contiguous
        # 断言确保 D_sklearn 与 D_scipy_cdist 的值接近，根据 rtol_dict 指定的容差
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)

        # 使用 dm.pairwise 计算稠密格式下 X 和 稀疏格式下 Y_csr 的距离矩阵 D_sklearn
        D_sklearn = dm.pairwise(X, Y_csr)
        # 断言确保 D_sklearn 是 C 连续的
        assert D_sklearn.flags.c_contiguous
        # 断言确保 D_sklearn 与 D_scipy_cdist 的值接近，根据 rtol_dict 指定的容差
        assert_allclose(D_sklearn, D_scipy_cdist, **rtol_dict)
@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize(
    "X_bool, Y_bool", [(X_bool, Y_bool), (X_bool_mmap, Y_bool_mmap)]
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数，使用参数化测试对布尔型数据进行距离计算的测试
def test_cdist_bool_metric(metric, X_bool, Y_bool, csr_container):
    # 使用scipy中的cdist计算布尔型数据的距离矩阵
    D_scipy_cdist = cdist(X_bool, Y_bool, metric)

    # 获取指定距离度量的距离计算器对象
    dm = DistanceMetric.get_metric(metric)
    # 使用scikit-learn中的pairwise方法计算布尔型数据的距离矩阵
    D_sklearn = dm.pairwise(X_bool, Y_bool)
    # 断言scikit-learn计算结果与scipy计算结果的近似性
    assert_allclose(D_sklearn, D_scipy_cdist)

    # DistanceMetric.pairwise 在 {sparse, dense}² 的所有组合中必须保持一致
    # 对输入数据进行稀疏格式和密集格式的组合测试
    X_bool_csr, Y_bool_csr = csr_container(X_bool), csr_container(Y_bool)

    # 使用scikit-learn计算布尔型数据的距离矩阵
    D_sklearn = dm.pairwise(X_bool, Y_bool)
    # 断言结果矩阵是C连续存储的
    assert D_sklearn.flags.c_contiguous
    # 断言scikit-learn计算结果与scipy计算结果的近似性
    assert_allclose(D_sklearn, D_scipy_cdist)

    # 使用scikit-learn计算稀疏格式数据的距离矩阵
    D_sklearn = dm.pairwise(X_bool_csr, Y_bool_csr)
    # 断言结果矩阵是C连续存储的
    assert D_sklearn.flags.c_contiguous
    # 断言scikit-learn计算结果与scipy计算结果的近似性
    assert_allclose(D_sklearn, D_scipy_cdist)

    # 使用scikit-learn计算布尔型输入和稀疏格式输出的距离矩阵
    D_sklearn = dm.pairwise(X_bool, Y_bool_csr)
    # 断言结果矩阵是C连续存储的
    assert D_sklearn.flags.c_contiguous
    # 断言scikit-learn计算结果与scipy计算结果的近似性
    assert_allclose(D_sklearn, D_scipy_cdist)

    # 使用scikit-learn计算稀疏格式输入和布尔型输出的距离矩阵
    D_sklearn = dm.pairwise(X_bool_csr, Y_bool)
    # 断言结果矩阵是C连续存储的
    assert D_sklearn.flags.c_contiguous
    # 断言scikit-learn计算结果与scipy计算结果的近似性
    assert_allclose(D_sklearn, D_scipy_cdist)


@pytest.mark.parametrize(
    "metric_param_grid", METRICS_DEFAULT_PARAMS, ids=lambda params: params[0]
)
@pytest.mark.parametrize("X", [X64, X32, X_mmap])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数，使用参数化测试对给定数据进行距离计算的测试
def test_pdist(metric_param_grid, X, csr_container):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    X_csr = csr_container(X)
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        rtol_dict = {}
        if metric == "mahalanobis" and X.dtype == np.float32:
            # mahalanobis距离的计算在scipy和scikit-learn实现中略有差异，需要增加相对容差
            rtol_dict = {"rtol": 1e-6}

        # TODO: 当scipy最小版本 >= 1.7.0 时移除
        # 对于minkowski距离，在scipy 1.7.0及以上版本支持0<p<1
        if metric == "minkowski":
            p = kwargs["p"]
            if sp_version < parse_version("1.7.0") and p < 1:
                pytest.skip("scipy does not support 0<p<1 for minkowski metric < 1.7.0")
        
        # 使用scipy中的cdist计算给定数据的距离矩阵
        D_scipy_pdist = cdist(X, X, metric, **kwargs)

        # 获取指定距离度量和参数的距离计算器对象
        dm = DistanceMetric.get_metric(metric, X.dtype, **kwargs)
        
        # 使用scikit-learn计算数据的距离矩阵
        D_sklearn = dm.pairwise(X)
        # 断言结果矩阵是C连续存储的
        assert D_sklearn.flags.c_contiguous
        # 断言scikit-learn计算结果与scipy计算结果的近似性
        assert_allclose(D_sklearn, D_scipy_pdist, **rtol_dict)

        # 使用scikit-learn计算稀疏格式数据的距离矩阵
        D_sklearn_csr = dm.pairwise(X_csr)
        # 断言结果矩阵是C连续存储的
        assert D_sklearn.flags.c_contiguous
        # 断言scikit-learn计算结果与scipy计算结果的近似性
        assert_allclose(D_sklearn_csr, D_scipy_pdist, **rtol_dict)

        # 使用scikit-learn计算稀疏格式数据对自身的距离矩阵
        D_sklearn_csr = dm.pairwise(X_csr, X_csr)
        # 断言结果矩阵是C连续存储的
        assert D_sklearn.flags.c_contiguous
        # 断言scikit-learn计算结果与scipy计算结果的近似性
        assert_allclose(D_sklearn_csr, D_scipy_pdist, **rtol_dict)
    # 使用pytest提供的`parametrize`装饰器，定义一个参数化测试
    "metric_param_grid", METRICS_DEFAULT_PARAMS, ids=lambda params: params[0]
)
def test_distance_metrics_dtype_consistency(metric_param_grid):
    # DistanceMetric must return similar distances for both float32 and float64
    # input data.
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()

    # Choose rtol to make sure that this test is robust to changes in the random
    # seed in the module-level test data generation code.
    rtol = 1e-5

    # Iterate over all combinations of parameter values
    for vals in itertools.product(*param_grid.values()):
        # Create a dictionary of keyword arguments
        kwargs = dict(zip(keys, vals))
        
        # Obtain DistanceMetric objects for both np.float64 and np.float32
        dm64 = DistanceMetric.get_metric(metric, np.float64, **kwargs)
        dm32 = DistanceMetric.get_metric(metric, np.float32, **kwargs)

        # Compute pairwise distances using np.float64 and np.float32 data types
        D64 = dm64.pairwise(X64)
        D32 = dm32.pairwise(X32)

        # Assert that the computed distances have the expected data types
        assert D64.dtype == np.float64
        assert D32.dtype == np.float32

        # Use assert_allclose to compare D64 and D32 with a specified relative tolerance
        # due to potential precision differences between np.float64 and np.float32
        assert_allclose(D64, D32, rtol=rtol)

        # Compute pairwise distances when two datasets (X64, Y64) and (X32, Y32) are provided
        D64 = dm64.pairwise(X64, Y64)
        D32 = dm32.pairwise(X32, Y32)
        
        # Assert that the pairwise distances for the different data types are consistent
        assert_allclose(D64, D32, rtol=rtol)


@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize("X_bool", [X_bool, X_bool_mmap])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pdist_bool_metrics(metric, X_bool, csr_container):
    # Compute pairwise distances using scipy's cdist function for boolean arrays
    D_scipy_pdist = cdist(X_bool, X_bool, metric)
    
    # Obtain DistanceMetric object for the specified metric
    dm = DistanceMetric.get_metric(metric)
    
    # Compute pairwise distances using sklearn's DistanceMetric
    D_sklearn = dm.pairwise(X_bool)
    
    # Assert that the pairwise distances computed by sklearn and scipy are close
    assert_allclose(D_sklearn, D_scipy_pdist)

    # Convert X_bool to a CSR container and compute distances again
    X_bool_csr = csr_container(X_bool)
    D_sklearn = dm.pairwise(X_bool_csr)
    
    # Assert that the pairwise distances are consistent after conversion to CSR format
    assert_allclose(D_sklearn, D_scipy_pdist)


@pytest.mark.parametrize("writable_kwargs", [True, False])
@pytest.mark.parametrize(
    "metric_param_grid", METRICS_DEFAULT_PARAMS, ids=lambda params: params[0]
)
@pytest.mark.parametrize("X", [X64, X32])
def test_pickle(writable_kwargs, metric_param_grid, X):
    # Unpack metric and parameter grid
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    
    # Iterate over all combinations of parameter values
    for vals in itertools.product(*param_grid.values()):
        # Deep copy the values if any of them are numpy arrays and set writable flag
        if any(isinstance(val, np.ndarray) for val in vals):
            vals = copy.deepcopy(vals)
            for val in vals:
                if isinstance(val, np.ndarray):
                    val.setflags(write=writable_kwargs)
        
        # Create a dictionary of keyword arguments
        kwargs = dict(zip(keys, vals))
        
        # Obtain DistanceMetric object for the specified metric and data type X.dtype
        dm = DistanceMetric.get_metric(metric, X.dtype, **kwargs)
        
        # Compute pairwise distances using the specified DistanceMetric object
        D1 = dm.pairwise(X)
        
        # Serialize and deserialize the DistanceMetric object using pickle
        dm2 = pickle.loads(pickle.dumps(dm))
        
        # Compute pairwise distances again with the deserialized DistanceMetric object
        D2 = dm2.pairwise(X)
        
        # Assert that the pairwise distances before and after pickling are close
        assert_allclose(D1, D2)


@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize("X_bool", [X_bool, X_bool_mmap])
def test_pickle_bool_metrics(metric, X_bool):
    # Obtain DistanceMetric object for the specified boolean metric
    dm = DistanceMetric.get_metric(metric)
    
    # Compute pairwise distances using the DistanceMetric object
    D1 = dm.pairwise(X_bool)
    
    # Serialize and deserialize the DistanceMetric object using pickle
    dm2 = pickle.loads(pickle.dumps(dm))
    
    # Compute pairwise distances again with the deserialized DistanceMetric object
    D2 = dm2.pairwise(X_bool)
    
    # Assert that the pairwise distances before and after pickling are close
    assert_allclose(D1, D2)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_haversine_metric(X, Y, csr_container):
    # Haversine DistanceMetric only operates on latitude and longitude, so reduce X and Y to 2 features.
    X = np.asarray(X[:, :2])
    Y = np.asarray(Y[:, :2])

    X_csr, Y_csr = csr_container(X), csr_container(Y)

    # Define a slow implementation of Haversine distance for reference.
    def haversine_slow(x1, x2):
        return 2 * np.arcsin(
            np.sqrt(
                np.sin(0.5 * (x1[0] - x2[0])) ** 2
                + np.cos(x1[0]) * np.cos(x2[0]) * np.sin(0.5 * (x1[1] - x2[1])) ** 2
            )
        )

    # Initialize a matrix to store pairwise distances calculated using the slow implementation.
    D_reference = np.zeros((X_csr.shape[0], Y_csr.shape[0]))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            D_reference[i, j] = haversine_slow(xi, yj)

    # Obtain the Haversine DistanceMetric object.
    haversine = DistanceMetric.get_metric("haversine", X.dtype)

    # Calculate pairwise distances using the Haversine DistanceMetric and assert closeness to the reference.
    D_sklearn = haversine.pairwise(X, Y)
    assert_allclose(
        haversine.dist_to_rdist(D_sklearn), np.sin(0.5 * D_reference) ** 2, rtol=1e-6
    )

    # Assert that distances calculated by the Haversine metric match the reference distances.
    assert_allclose(D_sklearn, D_reference)

    # Calculate pairwise distances using CSR representations of X and Y.
    D_sklearn = haversine.pairwise(X_csr, Y_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)

    # Calculate pairwise distances using CSR representation of X and full Y.
    D_sklearn = haversine.pairwise(X_csr, Y)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)

    # Calculate pairwise distances using full X and CSR representation of Y.
    D_sklearn = haversine.pairwise(X, Y_csr)
    assert D_sklearn.flags.c_contiguous
    assert_allclose(D_sklearn, D_reference)


def test_pyfunc_metric():
    # Generate random data.
    X = np.random.random((10, 3))

    # Obtain DistanceMetric objects for Euclidean and a custom callable metric (pyfunc).
    euclidean = DistanceMetric.get_metric("euclidean")
    pyfunc = DistanceMetric.get_metric("pyfunc", func=dist_func, p=2)

    # Ensure that the pickled versions of the metrics are equivalent.
    euclidean_pkl = pickle.loads(pickle.dumps(euclidean))
    pyfunc_pkl = pickle.loads(pickle.dumps(pyfunc))

    # Calculate pairwise distances using both metric objects and assert equality.
    D1 = euclidean.pairwise(X)
    D2 = pyfunc.pairwise(X)

    D1_pkl = euclidean_pkl.pairwise(X)
    D2_pkl = pyfunc_pkl.pairwise(X)

    assert_allclose(D1, D2)
    assert_allclose(D1_pkl, D2_pkl)


def test_input_data_size():
    # Regression test for issue #6288
    # Ensure that custom metrics requiring specific input dimensions do not fail.

    def custom_metric(x, y):
        # Ensure input dimension is as expected.
        assert x.shape[0] == 3
        return np.sum((x - y) ** 2)

    # Generate random data.
    rng = check_random_state(0)
    X = rng.rand(10, 3)

    # Obtain DistanceMetric objects for custom callable metric and Euclidean metric.
    pyfunc = DistanceMetric.get_metric("pyfunc", func=custom_metric)
    eucl = DistanceMetric.get_metric("euclidean")

    # Assert that pairwise distances calculated using both metrics are approximately equal.
    assert_allclose(pyfunc.pairwise(X), eucl.pairwise(X) ** 2)


def test_readonly_kwargs():
    # Non-regression test for issue #21685
    # Ensure that metrics supporting readonly buffers work correctly.

    # Generate random data.
    rng = check_random_state(0)

    # Create readonly buffers for weights and VI.
    weights = rng.rand(100)
    VI = rng.rand(10, 10)
    weights.setflags(write=False)
    VI.setflags(write=False)

    # Obtain DistanceMetric object for a specific metric (seuclidean) using readonly buffers.
    DistanceMetric.get_metric("seuclidean", V=weights)
    # 使用 DistanceMetric 类的 get_metric 方法来获取马哈拉诺比斯距离的度量对象，传入协方差矩阵 VI 作为参数
    DistanceMetric.get_metric("mahalanobis", VI=VI)
# 使用 pytest 的 parametrize 装饰器，对 test_minkowski_metric_validate_weights_values 函数进行参数化测试
@pytest.mark.parametrize(
    "w, err_type, err_msg",
    [
        # 第一个参数组：包含一个非法权重的 NumPy 数组，期望引发 ValueError 异常，错误信息为 "w cannot contain negative weights"
        (np.array([1, 1.5, -13]), ValueError, "w cannot contain negative weights"),
        # 第二个参数组：包含 NaN 值的 NumPy 数组，期望引发 ValueError 异常，错误信息为 "w contains NaN"
        (np.array([1, 1.5, np.nan]), ValueError, "w contains NaN"),
        # 使用 CSR_CONTAINERS 中的每个类型进行参数化
        *[
            (
                csr_container([[1, 1.5, 1]]),
                TypeError,
                "Sparse data was passed for w, but dense data is required",
            )
            for csr_container in CSR_CONTAINERS
        ],
        # 最后一个参数组：包含字符串的 NumPy 数组，期望引发 ValueError 异常，错误信息为 "could not convert string to float"
        (np.array(["a", "b", "c"]), ValueError, "could not convert string to float"),
        # 空数组的参数组，期望引发 ValueError 异常，错误信息为 "a minimum of 1 is required"
        (np.array([]), ValueError, "a minimum of 1 is required"),
    ],
)
# 定义测试函数 test_minkowski_metric_validate_weights_values
def test_minkowski_metric_validate_weights_values(w, err_type, err_msg):
    # 使用 pytest 的 raises 上下文管理器来检查是否引发指定类型和错误消息的异常
    with pytest.raises(err_type, match=err_msg):
        # 调用 DistanceMetric 类的 get_metric 方法，验证对 "minkowski" 距离度量的权重 w 的验证
        DistanceMetric.get_metric("minkowski", p=3, w=w)


# 定义测试函数 test_minkowski_metric_validate_weights_size
def test_minkowski_metric_validate_weights_size():
    # 使用随机数生成器 rng 生成大小为 d+1 的随机数组 w2
    w2 = rng.random_sample(d + 1)
    # 调用 DistanceMetric 类的 get_metric 方法，创建 Minkowski 距离度量对象 dm，使用参数 p=3 和 w=w2
    dm = DistanceMetric.get_metric("minkowski", p=3, w=w2)
    # 构建错误消息，用于验证 w 的大小必须与特征数 X64.shape[1] 相匹配
    msg = (
        "MinkowskiDistance: the size of w must match "
        f"the number of features \\({X64.shape[1]}\\). "
        f"Currently len\\(w\\)={w2.shape[0]}."
    )
    # 使用 pytest 的 raises 上下文管理器来检查是否引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 dm 对象的 pairwise 方法，传递参数 X64 和 Y64，验证异常处理机制
        dm.pairwise(X64, Y64)


# 使用 pytest 的 parametrize 装饰器，对 test_get_metric_dtype 函数进行参数化测试
@pytest.mark.parametrize("metric, metric_kwargs", METRICS_DEFAULT_PARAMS)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
# 定义测试函数 test_get_metric_dtype
def test_get_metric_dtype(metric, metric_kwargs, dtype):
    # 根据 dtype 选择 DistanceMetric32 或 DistanceMetric64 类
    specialized_cls = {
        np.float32: DistanceMetric32,
        np.float64: DistanceMetric64,
    }[dtype]
    
    # 缩小 metric_kwargs 的范围，只保留第一个值以进行基本检查
    metric_kwargs = {k: v[0] for k, v in metric_kwargs.items()}
    # 使用 DistanceMetric 类的 get_metric 方法，获取通用类型的距离度量对象
    generic_type = type(DistanceMetric.get_metric(metric, dtype, **metric_kwargs))
    # 使用 specialized_cls 对象的 get_metric 方法，获取特定类型的距离度量对象
    specialized_type = type(specialized_cls.get_metric(metric, **metric_kwargs))
    
    # 断言通用类型与特定类型应当相等
    assert generic_type is specialized_type


# 定义测试函数 test_get_metric_bad_dtype
def test_get_metric_bad_dtype():
    # 定义不支持的 dtype
    dtype = np.int32
    # 构建错误消息的正则表达式，验证给定的 dtype 是否为预期之外的值
    msg = r"Unexpected dtype .* provided. Please select a dtype from"
    # 使用 pytest 的 raises 上下文管理器来检查是否引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 DistanceMetric 类的 get_metric 方法，验证异常处理机制
        DistanceMetric.get_metric("manhattan", dtype)


# 定义测试函数 test_minkowski_metric_validate_bad_p_parameter
def test_minkowski_metric_validate_bad_p_parameter():
    # 构建错误消息，验证 p 参数必须大于 0
    msg = "p must be greater than 0"
    # 使用 pytest 的 raises 上下文管理器来检查是否引发 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 DistanceMetric 类的 get_metric 方法，验证异常处理机制
        DistanceMetric.get_metric("minkowski", p=0)
```