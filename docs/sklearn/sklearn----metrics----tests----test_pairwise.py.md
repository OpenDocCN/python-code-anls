# `D:\src\scipysrc\scikit-learn\sklearn\metrics\tests\test_pairwise.py`

```
# 引入警告模块，用于忽略特定的警告信息
import warnings
# 引入生成器类型，用于检查对象是否为生成器类型
from types import GeneratorType

# 引入 NumPy 库，并从中导入线性代数模块
import numpy as np
from numpy import linalg

# 引入 SciPy 稀疏矩阵模块，用于检查对象是否为稀疏矩阵
from scipy.sparse import issparse
# 引入 SciPy 空间距离模块，并从中导入多个距离度量函数
from scipy.spatial.distance import (
    cdist,
    cityblock,
    cosine,
    minkowski,
    pdist,
    squareform,
)

# 尝试引入 SciPy 空间距离模块中的 wminkowski 函数
try:
    from scipy.spatial.distance import wminkowski
# 如果引入失败，则将 minkowski 函数赋值给 wminkowski
except ImportError:
    # 在 SciPy 1.6.0 版本中，wminkowski 已被弃用，应使用 minkowski 函数替代
    from scipy.spatial.distance import minkowski as wminkowski

# 引入 pytest 库，用于编写和运行测试用例
import pytest

# 引入 scikit-learn 中的配置上下文
from sklearn import config_context
# 引入 scikit-learn 中的数据转换警告异常
from sklearn.exceptions import DataConversionWarning
# 引入 scikit-learn 中的距离度量计算模块，并从中导入多个距离度量函数
from sklearn.metrics.pairwise import (
    PAIRED_DISTANCES,
    PAIRWISE_BOOLEAN_FUNCTIONS,
    PAIRWISE_DISTANCE_FUNCTIONS,
    PAIRWISE_KERNEL_FUNCTIONS,
    _euclidean_distances_upcast,
    additive_chi2_kernel,
    check_paired_arrays,
    check_pairwise_arrays,
    chi2_kernel,
    cosine_distances,
    cosine_similarity,
    euclidean_distances,
    haversine_distances,
    laplacian_kernel,
    linear_kernel,
    manhattan_distances,
    nan_euclidean_distances,
    paired_cosine_distances,
    paired_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
    pairwise_distances,
    pairwise_distances_argmin,
    pairwise_distances_argmin_min,
    pairwise_distances_chunked,
    pairwise_kernels,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
# 引入 scikit-learn 中的数据标准化模块
from sklearn.preprocessing import normalize
# 引入 scikit-learn 中的测试工具，用于编写测试断言
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
# 引入 scikit-learn 中的修复工具，包括容器类型和版本解析
from sklearn.utils.fixes import (
    BSR_CONTAINERS,
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    parse_version,
    sp_version,
)
# 引入 scikit-learn 中的并行计算模块
from sklearn.utils.parallel import Parallel, delayed


# 定义测试函数：测试稠密数据的成对距离计算
def test_pairwise_distances_for_dense_data(global_dtype):
    # 创建随机数发生器对象
    rng = np.random.RandomState(0)

    # 使用随机数发生器创建随机浮点数矩阵 X，形状为 (5, 4)，并指定数据类型
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # 计算 X 中样本点的成对欧氏距离，并赋值给 S
    S = pairwise_distances(X, metric="euclidean")
    # 调用单独的欧氏距离计算函数，计算 X 中样本点的成对欧氏距离，并赋值给 S2
    S2 = euclidean_distances(X)
    # 断言 S 和 S2 相等
    assert_allclose(S, S2)
    # 断言 S 和 S2 的数据类型与全局数据类型相同
    assert S.dtype == S2.dtype == global_dtype

    # 计算 X 和 Y 中样本点之间的成对欧氏距离，其中 Y 的形状为 (2, 4)
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    # 调用成对距离计算函数，计算 X 和 Y 中样本点之间的成对欧氏距离，并赋值给 S
    S = pairwise_distances(X, Y, metric="euclidean")
    # 调用单独的欧氏距离计算函数，计算 X 和 Y 中样本点之间的成对欧氏距离，并赋值给 S2
    S2 = euclidean_distances(X, Y)
    # 断言 S 和 S2 相等
    assert_allclose(S, S2)
    # 断言 S 和 S2 的数据类型与全局数据类型相同
    assert S.dtype == S2.dtype == global_dtype

    # 检查 NaN 值在成对距离计算中的处理情况
    # 创建带有 NaN 值的随机浮点数矩阵 X 和 Y
    X_masked = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    Y_masked = rng.random_sample((2, 4)).astype(global_dtype, copy=False)
    # 将 X 和 Y 中的某些元素设为 NaN
    X_masked[0, 0] = np.nan
    Y_masked[0, 0] = np.nan
    # 调用带有 NaN 值的成对距离计算函数，计算 X 和 Y 中样本点的成对 NaN 欧氏距离，并赋值给 S_masked
    S_masked = pairwise_distances(X_masked, Y_masked, metric="nan_euclidean")
    # 调用单独的带有 NaN 值的欧氏距离计算函数，计算 X 和 Y 中样本点的成对 NaN 欧氏距离，并赋值给 S2_masked
    S2_masked = nan_euclidean_distances(X_masked, Y_masked)
    # 断言 S_masked 和 S2_masked 相等
    assert_allclose(S_masked, S2_masked)
    # 断言 S_masked 和 S2_masked 的数据类型与全局数据类型相同
    assert S_masked.dtype == S2_masked.dtype == global_dtype
    # Test with tuples as X and Y
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    # Calculate pairwise Euclidean distances between X_tuples and Y_tuples
    S2 = pairwise_distances(X_tuples, Y_tuples, metric="euclidean")
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)
    # Assert the data types of S and S2 are equal to global_dtype
    assert S.dtype == S2.dtype == global_dtype

    # Test haversine distance
    # X contains valid latitude and longitude data
    # Convert X to global_dtype and adjust values for haversine calculation
    X = rng.random_sample((5, 2)).astype(global_dtype, copy=False)
    X[:, 0] = (X[:, 0] - 0.5) * 2 * np.pi / 2
    X[:, 1] = (X[:, 1] - 0.5) * 2 * np.pi
    # Calculate pairwise haversine distances for X
    S = pairwise_distances(X, metric="haversine")
    # Calculate haversine distances using a different method
    S2 = haversine_distances(X)
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)

    # Test haversine distance, with Y != X
    # Y contains valid latitude and longitude data
    Y = rng.random_sample((2, 2)).astype(global_dtype, copy=False)
    Y[:, 0] = (Y[:, 0] - 0.5) * 2 * np.pi / 2
    Y[:, 1] = (Y[:, 1] - 0.5) * 2 * np.pi
    # Calculate pairwise haversine distances between X and Y
    S = pairwise_distances(X, Y, metric="haversine")
    # Calculate haversine distances using a different method for X and Y
    S2 = haversine_distances(X, Y)
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)

    # "cityblock" uses scikit-learn metric, cityblock (function) is
    # scipy.spatial.
    # Calculate pairwise cityblock distances for X
    S = pairwise_distances(X, metric="cityblock")
    # Calculate cityblock distances using a function from scipy for X
    S2 = pairwise_distances(X, metric=cityblock)
    # Assert that both resulting matrices have the same shape
    assert S.shape[0] == S.shape[1]
    assert S.shape[0] == X.shape[0]
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)

    # The manhattan metric should be equivalent to cityblock.
    # Calculate pairwise Manhattan distances between X and Y
    S = pairwise_distances(X, Y, metric="manhattan")
    # Calculate cityblock distances using a function from scipy for X and Y
    S2 = pairwise_distances(X, Y, metric=cityblock)
    # Assert that S has the correct shape relative to X and Y
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)

    # Test cosine as a string metric versus cosine callable
    # Calculate pairwise cosine distances between X and Y using scikit-learn
    S = pairwise_distances(X, Y, metric="cosine")
    # Calculate cosine distances using a function from scipy for X and Y
    S2 = pairwise_distances(X, Y, metric=cosine)
    # Assert that S has the correct shape relative to X and Y
    assert S.shape[0] == X.shape[0]
    assert S.shape[1] == Y.shape[0]
    # Assert that S and S2 are almost equal
    assert_allclose(S, S2)
# 使用参数化测试，依次对COO、CSC、BSR和CSR稀疏矩阵容器进行测试
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize("bsr_container", BSR_CONTAINERS)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_distances_for_sparse_data(
    coo_container, csc_container, bsr_container, csr_container, global_dtype
):
    # 测试pairwise_distance辅助函数
    rng = np.random.RandomState(0)
    # 创建一个5x4的随机浮点数矩阵X，并转换为指定的全局数据类型
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # 创建一个2x4的随机浮点数矩阵Y，并转换为指定的全局数据类型
    Y = rng.random_sample((2, 4)).astype(global_dtype, copy=False)

    # 测试稀疏矩阵X和Y，目前仅支持欧氏距离、L1距离和余弦距离
    X_sparse = csr_container(X)  # 将稀疏矩阵X转换为指定类型的CSR格式
    Y_sparse = csr_container(Y)  # 将稀疏矩阵Y转换为指定类型的CSR格式

    # 计算稀疏矩阵X_sparse和Y_sparse的欧氏距离，并进行比较
    S = pairwise_distances(X_sparse, Y_sparse, metric="euclidean")
    S2 = euclidean_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # 计算稀疏矩阵X_sparse和Y_sparse的余弦距离，并进行比较
    S = pairwise_distances(X_sparse, Y_sparse, metric="cosine")
    S2 = cosine_distances(X_sparse, Y_sparse)
    assert_allclose(S, S2)
    assert S.dtype == S2.dtype == global_dtype

    # 使用稀疏矩阵X_sparse和转换后的稀疏矩阵csc_container(Y)计算曼哈顿距离，并进行比较
    S = pairwise_distances(X_sparse, csc_container(Y), metric="manhattan")
    S2 = manhattan_distances(bsr_container(X), coo_container(Y))
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO 修复manhattan_distances以保持数据类型
        # 目前pairwise_distances使用manhattan_distances但将结果转换回输入的数据类型
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    # 计算非稀疏矩阵X和Y的曼哈顿距离，并进行比较
    S2 = manhattan_distances(X, Y)
    assert_allclose(S, S2)
    if global_dtype == np.float64:
        assert S.dtype == S2.dtype == global_dtype
    else:
        # TODO 修复manhattan_distances以保持数据类型
        # 目前pairwise_distances使用manhattan_distances但将结果转换回输入的数据类型
        with pytest.raises(AssertionError):
            assert S.dtype == S2.dtype == global_dtype

    # 使用带有kwd参数的scipy.spatial.distance度量函数测试
    kwds = {"p": 2.0}
    S = pairwise_distances(X, Y, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, Y, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # 当Y=None时，同样测试
    kwds = {"p": 2.0}
    S = pairwise_distances(X, metric="minkowski", **kwds)
    S2 = pairwise_distances(X, metric=minkowski, **kwds)
    assert_allclose(S, S2)

    # 测试如果给定稀疏矩阵时，scipy距离度量函数是否会抛出TypeError错误
    with pytest.raises(TypeError):
        pairwise_distances(X_sparse, metric="minkowski")
    with pytest.raises(TypeError):
        pairwise_distances(X, Y_sparse, metric="minkowski")
    # 生成一个5行4列的随机数组X
    X = rng.randn(5, 4)
    # 复制数组X并赋值给数组Y
    Y = X.copy()
    # 修改Y的第一个元素，使其不等于原始数组X的第一个元素
    Y[0, 0] = 1 - Y[0, 0]

    # 忽略在pairwise_distances函数中将数据转换为布尔值时的警告
    with ignore_warnings(category=DataConversionWarning):
        # 对数组X和None（空值）分别计算使用给定度量(metric)的距离
        for Z in [Y, None]:
            # 计算X和Z之间的距离，并将结果存储在res中
            res = pairwise_distances(X, Z, metric=metric)
            # 将res中的NaN值替换为0，不生成副本
            np.nan_to_num(res, nan=0, posinf=0, neginf=0, copy=False)
            # 断言所有res中不为0的元素数量为0
            assert np.sum(res != 0) == 0

    # 如果数据不是布尔类型，在使用布尔距离度量时，会发出数据转换警告
    msg = "Data was converted to boolean for metric %s" % metric
    with pytest.warns(DataConversionWarning, match=msg):
        # 使用指定度量(metric)计算数组X之间的距离
        pairwise_distances(X, metric=metric)

    # 检查如果X是布尔类型而Y不是布尔类型，则会发出警告
    with pytest.warns(DataConversionWarning, match=msg):
        # 使用指定度量(metric)计算布尔类型的数组X与非布尔类型的Y之间的距离
        pairwise_distances(X.astype(bool), Y=Y, metric=metric)

    # 检查如果X已经是布尔类型且Y是None，则不会发出警告
    with warnings.catch_warnings():
        # 设置只抛出DataConversionWarning类型的警告
        warnings.simplefilter("error", DataConversionWarning)
        # 使用指定度量(metric)计算布尔类型的数组X之间的距离
        pairwise_distances(X.astype(bool), metric=metric)
def test_no_data_conversion_warning():
    # No warnings issued if metric is not a boolean distance function
    # 使用指定种子创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 生成一个5行4列的随机数矩阵
    X = rng.randn(5, 4)
    # 捕获警告并设置警告过滤器，当遇到DataConversionWarning时抛出异常
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        # 调用pairwise_distances函数计算X中数据的Minkowski距离
        pairwise_distances(X, metric="minkowski")


@pytest.mark.parametrize("func", [pairwise_distances, pairwise_kernels])
def test_pairwise_precomputed(func):
    # Test correct shape
    # 使用pytest断言测试函数参数错误异常是否匹配指定的正则表达式匹配模式
    with pytest.raises(ValueError, match=".* shape .*"):
        # 调用func函数并传入一个预计算的距离矩阵，验证其形状是否引发异常
        func(np.zeros((5, 3)), metric="precomputed")
    # with two args
    with pytest.raises(ValueError, match=".* shape .*"):
        # 传入两个预计算距离矩阵，验证其形状是否引发异常
        func(np.zeros((5, 3)), np.zeros((4, 4)), metric="precomputed")
    # even if shape[1] agrees (although thus second arg is spurious)
    with pytest.raises(ValueError, match=".* shape .*"):
        # 传入两个预计算距离矩阵，验证其形状是否引发异常（即使第二个参数的列数匹配）
        func(np.zeros((5, 3)), np.zeros((4, 3)), metric="precomputed")

    # Test not copied (if appropriate dtype)
    # 创建一个5行5列全零矩阵S
    S = np.zeros((5, 5))
    # 调用func函数计算矩阵S的预计算距离矩阵，并验证S与返回值是否为同一个对象
    S2 = func(S, metric="precomputed")
    assert S is S2
    # with two args
    # 创建一个5行3列全零矩阵S
    S = np.zeros((5, 3))
    # 调用func函数计算两个矩阵的预计算距离矩阵，并验证S与返回值是否为同一个对象
    S2 = func(S, np.zeros((3, 3)), metric="precomputed")
    assert S is S2

    # Test always returns float dtype
    # 调用func函数计算一个整数类型的矩阵的预计算距离矩阵，验证返回结果是否为浮点类型
    S = func(np.array([[1]], dtype="int"), metric="precomputed")
    assert "f" == S.dtype.kind

    # Test converts list to array-like
    # 调用func函数计算一个列表形式的矩阵的预计算距离矩阵，验证返回结果是否为numpy数组类型
    S = func([[1.0]], metric="precomputed")
    assert isinstance(S, np.ndarray)


def test_pairwise_precomputed_non_negative():
    # Test non-negative values
    # 使用pytest断言测试函数参数错误异常是否匹配指定的正则表达式匹配模式
    with pytest.raises(ValueError, match=".* non-negative values.*"):
        # 调用pairwise_distances函数计算一个全为负数的5行5列矩阵的距离矩阵，验证是否引发异常
        pairwise_distances(np.full((5, 5), -1), metric="precomputed")


_minkowski_kwds = {"w": np.arange(1, 5).astype("double", copy=False), "p": 1}
_wminkowski_kwds = {"w": np.arange(1, 5).astype("double", copy=False), "p": 1}


def callable_rbf_kernel(x, y, **kwds):
    # Callable version of pairwise.rbf_kernel.
    # 调用pairwise.rbf_kernel的可调用版本，计算输入x和y的径向基函数核矩阵K
    K = rbf_kernel(np.atleast_2d(x), np.atleast_2d(y), **kwds)
    # 解包K的输出，因为它是封装在0维数组中的标量
    return K.item()


@pytest.mark.parametrize(
    "func, metric, kwds",
    [
        # 调用 pairwise_distances 函数，使用欧氏距离计算，参数为空字典
        (pairwise_distances, "euclidean", {}),
        # 使用 pytest.param 传递 pairwise_distances 函数，使用 Minkowski 距离计算，传递 _minkowski_kwds 参数
        pytest.param(
            pairwise_distances,
            minkowski,
            _minkowski_kwds,
        ),
        # 使用 pytest.param 传递 pairwise_distances 函数，使用字符串 "minkowski" 计算，传递 _minkowski_kwds 参数
        pytest.param(
            pairwise_distances,
            "minkowski",
            _minkowski_kwds,
        ),
        # 使用 pytest.param 传递 pairwise_distances 函数，使用 wminkowski 距离计算，传递 _wminkowski_kwds 参数，并标记为跳过条件
        pytest.param(
            pairwise_distances,
            wminkowski,
            _wminkowski_kwds,
            marks=pytest.mark.skipif(
                sp_version >= parse_version("1.6.0"),
                reason="wminkowski is now minkowski and it has been already tested.",
            ),
        ),
        # 使用 pytest.param 传递 pairwise_distances 函数，使用字符串 "wminkowski" 计算，传递 _wminkowski_kwds 参数，并标记为跳过条件
        pytest.param(
            pairwise_distances,
            "wminkowski",
            _wminkowski_kwds,
            marks=pytest.mark.skipif(
                sp_version >= parse_version("1.6.0"),
                reason="wminkowski is now minkowski and it has been already tested.",
            ),
        ),
        # 调用 pairwise_kernels 函数，使用多项式核，指定 degree 参数为 1
        (pairwise_kernels, "polynomial", {"degree": 1}),
        # 调用 pairwise_kernels 函数，使用 callable_rbf_kernel 函数作为核函数，指定 gamma 参数为 0.1
        (pairwise_kernels, callable_rbf_kernel, {"gamma": 0.1}),
    ],
@pytest.mark.parametrize("dtype", [np.float64, np.float32, int])
def test_pairwise_parallel(func, metric, kwds, dtype):
    # 创建一个随机数生成器实例
    rng = np.random.RandomState(0)
    # 生成指定类型和形状的随机数组 X 和 Y
    X = np.array(5 * rng.random_sample((5, 4)), dtype=dtype)
    Y = np.array(5 * rng.random_sample((3, 4)), dtype=dtype)

    # 使用单线程计算 func(X) 的结果 S 和 func(X, Y) 的结果 S2
    S = func(X, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, metric=metric, n_jobs=2, **kwds)
    # 断言单线程和双线程结果的近似性
    assert_allclose(S, S2)

    # 再次使用单线程计算 func(X, Y) 的结果 S 和 func(X, Y) 的结果 S2
    S = func(X, Y, metric=metric, n_jobs=1, **kwds)
    S2 = func(X, Y, metric=metric, n_jobs=2, **kwds)
    # 断言单线程和双线程结果的近似性
    assert_allclose(S, S2)


def test_pairwise_callable_nonstrict_metric():
    # 测试 pairwise_distances 函数是否允许具有非严格度量的可调用度量函数
    # 确定可调用函数是严格度量时，对角线可不计算并设为0
    assert pairwise_distances([[1.0]], metric=lambda x, y: 5)[0, 0] == 5


# 测试所有应在 PAIRWISE_KERNEL_FUNCTIONS 中的度量函数
@pytest.mark.parametrize(
    "metric",
    ["rbf", "laplacian", "sigmoid", "polynomial", "linear", "chi2", "additive_chi2"],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pairwise_kernels(metric, csr_container):
    # 测试 pairwise_kernels 辅助函数

    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    function = PAIRWISE_KERNEL_FUNCTIONS[metric]
    # 测试当 Y=None 时的 pairwise_kernels
    K1 = pairwise_kernels(X, metric=metric)
    K2 = function(X)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)
    # 测试当 Y=Y 时的 pairwise_kernels
    K1 = pairwise_kernels(X, Y=Y, metric=metric)
    K2 = function(X, Y=Y)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)
    # 测试以元组形式输入 X 和 Y 的情况
    X_tuples = tuple([tuple([v for v in row]) for row in X])
    Y_tuples = tuple([tuple([v for v in row]) for row in Y])
    K2 = pairwise_kernels(X_tuples, Y_tuples, metric=metric)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)

    # 测试稀疏矩阵 X 和 Y 的情况
    X_sparse = csr_container(X)
    Y_sparse = csr_container(Y)
    if metric in ["chi2", "additive_chi2"]:
        # 这些度量不支持稀疏矩阵
        return
    K1 = pairwise_kernels(X_sparse, Y=Y_sparse, metric=metric)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)


def test_pairwise_kernels_callable():
    # 测试带有可调用函数的 pairwise_kernels 辅助函数
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))

    metric = callable_rbf_kernel
    kwds = {"gamma": 0.1}
    K1 = pairwise_kernels(X, Y=Y, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=Y, **kwds)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)

    # 使用 X=Y 的情况测试可调用函数
    K1 = pairwise_kernels(X, Y=X, metric=metric, **kwds)
    K2 = rbf_kernel(X, Y=X, **kwds)
    # 断言两种计算方式的结果近似
    assert_allclose(K1, K2)


def test_pairwise_kernels_filter_param():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((2, 4))
    K = rbf_kernel(X, Y, gamma=0.1)
    params = {"gamma": 0.1, "blabla": ":)"}
    # 使用径向基函数(RBF)作为核函数计算输入数据集 X 和 Y 之间的核矩阵 K2
    K2 = pairwise_kernels(X, Y, metric="rbf", filter_params=True, **params)
    
    # 断言核矩阵 K 和计算得到的 K2 在数值上相等，如果不相等则会引发 AssertionError
    assert_allclose(K, K2)
    
    # 使用 pytest 的上下文管理器 pytest.raises 检查是否会抛出 TypeError 异常
    # 当使用径向基函数(RBF)作为核函数计算时，不应该传递额外的未知参数，这里预期会抛出 TypeError
    with pytest.raises(TypeError):
        pairwise_kernels(X, Y, metric="rbf", **params)
@pytest.mark.parametrize("metric, func", PAIRED_DISTANCES.items())
# 使用参数化测试，对每一对 metric 和 func 进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用参数化测试，对每一种 CSR 容器进行测试
def test_paired_distances(metric, func, csr_container):
    # 测试 paired_distances 辅助函数
    rng = np.random.RandomState(0)
    # 创建一个随机数生成器实例
    X = rng.random_sample((5, 4))
    # 创建大小为 (5, 4) 的随机样本矩阵 X
    Y = rng.random_sample((5, 4))
    # 创建大小为 (5, 4) 的随机样本矩阵 Y

    S = paired_distances(X, Y, metric=metric)
    # 计算 X 和 Y 之间的距离，使用给定的 metric
    S2 = func(X, Y)
    # 使用 func 计算 X 和 Y 之间的距离
    assert_allclose(S, S2)
    S3 = func(csr_container(X), csr_container(Y))
    # 使用 csr_container 处理 X 和 Y，并使用 func 计算它们之间的距离
    assert_allclose(S, S3)
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        # 检查 pairwise_distances 实现是否给出相同的值
        distances = PAIRWISE_DISTANCE_FUNCTIONS[metric](X, Y)
        # 计算 X 和 Y 之间的距离，使用给定的 metric
        distances = np.diag(distances)
        # 提取距离矩阵的对角线元素
        assert_allclose(distances, S)


def test_paired_distances_callable(global_dtype):
    # 测试 paired_distances 辅助函数，使用可调用实现
    rng = np.random.RandomState(0)
    # 创建一个随机数生成器实例
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # 创建大小为 (5, 4) 的随机样本矩阵 X，并指定数据类型
    Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
    # 创建大小为 (5, 4) 的随机样本矩阵 Y，并指定数据类型

    S = paired_distances(X, Y, metric="manhattan")
    # 计算 X 和 Y 之间的 Manhattan 距离
    S2 = paired_distances(X, Y, metric=lambda x, y: np.abs(x - y).sum(axis=0))
    # 使用 lambda 函数计算 X 和 Y 之间的距离
    assert_allclose(S, S2)

    # 测试当 X 和 Y 的长度不同时是否会引发 ValueError
    Y = rng.random_sample((3, 4))
    with pytest.raises(ValueError):
        paired_distances(X, Y)


@pytest.mark.parametrize("dok_container", DOK_CONTAINERS)
# 使用参数化测试，对每一种 DOK 容器进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用参数化测试，对每一种 CSR 容器进行测试
def test_pairwise_distances_argmin_min(dok_container, csr_container, global_dtype):
    # 检查任意 metric 的最小距离计算
    X = np.asarray([[0], [1]], dtype=global_dtype)
    # 创建一个大小为 (2, 1) 的数组 X，指定数据类型
    Y = np.asarray([[-2], [3]], dtype=global_dtype)
    # 创建一个大小为 (2, 1) 的数组 Y，指定数据类型

    Xsp = dok_container(X)
    # 使用 dok_container 处理 X，生成稀疏矩阵
    Ysp = csr_container(Y, dtype=global_dtype)
    # 使用 csr_container 处理 Y，生成稀疏矩阵

    expected_idx = [0, 1]
    # 预期的索引数组
    expected_vals = [2, 2]
    # 预期的值数组
    expected_vals_sq = [4, 4]
    # 预期的平方值数组

    # 欧氏距离 metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="euclidean")
    # 计算 X 和 Y 之间的最小距离索引和值，使用欧氏距离
    idx2 = pairwise_distances_argmin(X, Y, metric="euclidean")
    # 计算 X 和 Y 之间的最小距离索引，使用欧氏距离
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)

    # 稀疏矩阵情况
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="euclidean")
    # 计算稀疏矩阵 Xsp 和 Ysp 之间的最小距离索引和值，使用欧氏距离
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="euclidean")
    # 计算稀疏矩阵 Xsp 和 Ysp 之间的最小距离索引，使用欧氏距离
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)
    # 确保不要出现 np.matrix 类型
    assert type(idxsp) == np.ndarray
    assert type(valssp) == np.ndarray

    # 平方欧氏距离 metric
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="sqeuclidean")
    # 使用 scikit-learn 的函数计算 X 和 Y 之间的欧几里德距离的最小值的索引和距离的平方
    idx2, vals2 = pairwise_distances_argmin_min(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的平方欧几里德距离的最小值的索引
    idx3 = pairwise_distances_argmin(X, Y, metric="sqeuclidean")

    # 再次使用 scikit-learn 的函数计算 X 和 Y 之间的欧几里德距离的最小值的索引和距离的平方
    idx4 = pairwise_distances_argmin(
        X, Y, metric="euclidean", metric_kwargs={"squared": True}
    )

    # 使用 numpy.testing.assert_allclose 检查 vals 和期望的平方欧几里德距离是否几乎相等
    assert_allclose(vals, expected_vals_sq)
    assert_allclose(vals2, expected_vals_sq)

    # 使用 numpy.testing.assert_allclose 检查 idx 和期望的最小索引是否几乎相等
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(idx3, expected_idx)
    assert_allclose(idx4, expected_idx)

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的曼哈顿距离的最小值的索引和距离
    idx, vals = pairwise_distances_argmin_min(X, Y, metric="manhattan")
    idx2 = pairwise_distances_argmin(X, Y, metric="manhattan")

    # 使用 numpy.testing.assert_allclose 检查曼哈顿距离的索引和值是否几乎相等
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)

    # 对稀疏矩阵的情况进行测试，计算稀疏矩阵 Xsp 和 Ysp 之间的曼哈顿距离的最小值的索引和距离
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric="manhattan")
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric="manhattan")

    # 使用 numpy.testing.assert_allclose 检查稀疏矩阵曼哈顿距离的索引和值是否几乎相等
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的 Minkowski 距离（通过可调用对象 minkowski）的最小值的索引和距离
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric=minkowski, metric_kwargs={"p": 2}
    )

    # 使用 numpy.testing.assert_allclose 检查 Minkowski 距离的索引和值是否几乎相等
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的 Minkowski 距离（通过字符串 "minkowski"）的最小值的索引和距离
    idx, vals = pairwise_distances_argmin_min(
        X, Y, metric="minkowski", metric_kwargs={"p": 2}
    )

    # 使用 numpy.testing.assert_allclose 检查 Minkowski 距离的索引和值是否几乎相等
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)

    # 比较计算结果与朴素实现的结果是否接近
    rng = np.random.RandomState(0)
    X = rng.randn(97, 149)
    Y = rng.randn(111, 149)

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的曼哈顿距离矩阵
    dist = pairwise_distances(X, Y, metric="manhattan")

    # 计算最小距离的原始索引和值
    dist_orig_ind = dist.argmin(axis=0)
    dist_orig_val = dist[dist_orig_ind, range(len(dist_orig_ind))]

    # 使用 scikit-learn 的函数计算 X 和 Y 之间的曼哈顿距离的最小值的索引和距离（分块方式）
    dist_chunked_ind, dist_chunked_val = pairwise_distances_argmin_min(
        X, Y, axis=0, metric="manhattan"
    )

    # 使用 numpy.testing.assert_allclose 检查分块方式计算的结果与原始计算结果是否几乎相等
    assert_allclose(dist_orig_ind, dist_chunked_ind, rtol=1e-7)
    assert_allclose(dist_orig_val, dist_chunked_val, rtol=1e-7)

    # 改变计算轴和排列数据集应该给出相同的结果
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, Y, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(Y, X, axis=1)

    # 使用 numpy.testing.assert_allclose 检查不同轴上计算的结果是否几乎相等
    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    # 改变计算轴和排列数据集应该给出相同的结果
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, X, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(X, X, axis=1)

    # 使用 numpy.testing.assert_allclose 检查相同数据集不同轴上计算的结果是否几乎相等
    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)

    # 改变计算轴和排列数据集应该给出相同的结果
    argmin_0 = pairwise_distances_argmin(X, Y, axis=0)
    argmin_1 = pairwise_distances_argmin(Y, X, axis=1)

    # 使用 numpy.testing.assert_array_equal 检查不同轴上计算的结果是否相等
    assert_array_equal(argmin_0, argmin_1)

    # 改变计算轴和排列数据集应该给出相同的结果
    argmin_0 = pairwise_distances_argmin(X, X, axis=0)
    argmin_1 = pairwise_distances_argmin(X, X, axis=1)
    # 断言两个数组 argmin_0 和 argmin_1 必须相等
    assert_array_equal(argmin_0, argmin_1)
    
    # 对于 F-contiguous（列优先存储）的数组必须支持，并且应返回相同的结果。
    # 计算 C-contiguous（行优先存储）数组 X 和 Y 之间的最小距离的索引
    argmin_C_contiguous = pairwise_distances_argmin(X, Y)
    # 使用 np.asfortranarray 将数组 X 和 Y 转换为 F-contiguous 格式，然后计算它们之间的最小距离的索引
    argmin_F_contiguous = pairwise_distances_argmin(
        np.asfortranarray(X), np.asfortranarray(Y)
    )
    
    # 断言计算出的 C-contiguous 和 F-contiguous 数组的最小距离索引结果必须相等
    assert_array_equal(argmin_C_contiguous, argmin_F_contiguous)
# 定义一个函数 `_reduce_func`，接受两个参数 `dist` 和 `start`，返回 `dist` 的前 100 列
def _reduce_func(dist, start):
    return dist[:, :100]


# 测试函数 `test_pairwise_distances_chunked_reduce`，接受全局数据类型 `global_dtype` 作为参数
def test_pairwise_distances_chunked_reduce(global_dtype):
    # 使用种子为 0 的随机数生成器创建随机数组 `X`，形状为 (400, 4)，并转换为 `global_dtype` 类型
    rng = np.random.RandomState(0)
    X = rng.random_sample((400, 4)).astype(global_dtype, copy=False)
    
    # 计算 `X` 中样本间的欧氏距离，并取每行的前 100 列，存储在 `S` 中
    S = pairwise_distances(X)[:, :100]
    
    # 使用 `pairwise_distances_chunked` 函数计算 `X` 的距离，指定 `reduce_func` 为 `_reduce_func` 函数，
    # 并设置工作内存大小为 2^(-16)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=_reduce_func, working_memory=2**-16
    )
    
    # 断言 `S_chunks` 是生成器类型
    assert isinstance(S_chunks, GeneratorType)
    
    # 将生成器转换为列表
    S_chunks = list(S_chunks)
    
    # 断言生成的距离块数量大于 1
    assert len(S_chunks) > 1
    
    # 断言第一个距离块的数据类型与 `X` 相同
    assert S_chunks[0].dtype == X.dtype
    
    # 使用 `assert_allclose` 函数检查所有距离块连接后的结果与 `S` 的吻合度，允许误差为 1e-7
    assert_allclose(np.vstack(S_chunks), S, atol=1e-7)


# 测试函数 `test_pairwise_distances_chunked_reduce_none`，接受全局数据类型 `global_dtype` 作为参数
def test_pairwise_distances_chunked_reduce_none(global_dtype):
    # 检查 reduce 函数允许返回 None 的情况
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    
    # 使用 `pairwise_distances_chunked` 函数计算 `X` 的距离，指定 reduce_func 为 lambda 表达式，始终返回 None，
    # 并设置工作内存大小为 2^(-16)
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=lambda dist, start: None, working_memory=2**-16
    )
    
    # 断言 `S_chunks` 是生成器类型
    assert isinstance(S_chunks, GeneratorType)
    
    # 将生成器转换为列表
    S_chunks = list(S_chunks)
    
    # 断言生成的所有距离块都为 None
    assert all(chunk is None for chunk in S_chunks)


# 使用 `pytest.mark.parametrize` 标记的测试函数 `test_pairwise_distances_chunked_reduce_valid`
# 接受一个名为 `good_reduce` 的参数，它是各种可接受的 reduce 函数的 lambda 函数列表和一些特定的 CSR 容器和 DOK 容器
@pytest.mark.parametrize(
    "good_reduce",
    [
        lambda D, start: list(D),  # 返回列表
        lambda D, start: np.array(D),  # 返回 NumPy 数组
        lambda D, start: (list(D), list(D)),  # 返回元组，包含两个列表
    ]
    + [
        lambda D, start, scipy_csr_type=scipy_csr_type: scipy_csr_type(D)
        for scipy_csr_type in CSR_CONTAINERS
    ]
    + [
        lambda D, start, scipy_dok_type=scipy_dok_type: (
            scipy_dok_type(D),
            np.array(D),
            list(D),
        )
        for scipy_dok_type in DOK_CONTAINERS
    ],
)
def test_pairwise_distances_chunked_reduce_valid(good_reduce):
    # 创建一个简单的数组 `X`，形状为 (10, 1)
    X = np.arange(10).reshape(-1, 1)
    
    # 使用 `pairwise_distances_chunked` 函数计算 `X` 的距离，指定 reduce_func 为 `good_reduce`，
    # 并设置工作内存大小为 64
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=good_reduce, working_memory=64
    )
    
    # 调用生成器的 `next` 方法
    next(S_chunks)


# 使用 `pytest.mark.parametrize` 标记的测试函数 `test_pairwise_distances_chunked_reduce_invalid`
# 接受三个参数：`global_dtype`、`bad_reduce` 和两个字符串 `err_type`、`message`
@pytest.mark.parametrize(
    ("bad_reduce", "err_type", "message"),
    [
        (
            lambda D, s: np.concatenate([D, D[-1:]]),
            ValueError,
            r"length 11\..* input: 10\.",
        ),
        (
            lambda D, s: (D, np.concatenate([D, D[-1:]])),
            ValueError,
            r"length \(10, 11\)\..* input: 10\.",
        ),
        (lambda D, s: (D[:9], D), ValueError, r"length \(9, 10\)\..* input: 10\."),
        (
            lambda D, s: 7,
            TypeError,
            r"returned 7\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (7, 8),
            TypeError,
            r"returned \(7, 8\)\. Expected sequence\(s\) of length 10\.",
        ),
        (
            lambda D, s: (np.arange(10), 9),
            TypeError,
            r", 9\)\. Expected sequence\(s\) of length 10\.",
        ),
    ],
)
def test_pairwise_distances_chunked_reduce_invalid(
    global_dtype, bad_reduce, err_type, message
):
    # 使用 NumPy 创建一个包含数字0到9的数组，然后将其转换为一列，数据类型与全局变量 global_dtype 一致，并且不进行复制
    X = np.arange(10).reshape(-1, 1).astype(global_dtype, copy=False)
    
    # 使用 pairwise_distances_chunked 函数计算输入数据 X 的成对距离，但不提供第二个数据集 (None)，使用 bad_reduce 函数进行数据减少，工作内存大小为64
    S_chunks = pairwise_distances_chunked(
        X, None, reduce_func=bad_reduce, working_memory=64
    )
    
    # 使用 pytest 模块中的 pytest.raises 上下文管理器，验证下一个操作会引发指定类型的异常 err_type，且异常消息与给定的 message 匹配
    with pytest.raises(err_type, match=message):
        # 获取 S_chunks 的下一个元素，预期会引发异常
        next(S_chunks)
# 定义一个函数，用于按块处理两个数据集的成对距离，检查内存使用情况和距离计算正确性
def check_pairwise_distances_chunked(X, Y, working_memory, metric="euclidean"):
    # 生成器对象，按块生成 X 和 Y 之间的成对距离
    gen = pairwise_distances_chunked(X, Y, working_memory=working_memory, metric=metric)
    # 断言 gen 是一个生成器对象
    assert isinstance(gen, GeneratorType)
    # 将生成器转换为列表，获取所有块的距离
    blockwise_distances = list(gen)
    # 如果 Y 是 None，则将 Y 设置为 X
    Y = X if Y is None else Y
    # 计算最小块大小（单位：MiB），确保内存充足
    min_block_mib = len(Y) * 8 * 2**-20

    # 遍历每个块的距离
    for block in blockwise_distances:
        # 计算当前块占用的内存字节数
        memory_used = block.nbytes
        # 断言当前块的内存使用不超过设定的工作内存或最小块大小的上限
        assert memory_used <= max(working_memory, min_block_mib) * 2**20

    # 将所有块的距离堆叠为一个 numpy 数组
    blockwise_distances = np.vstack(blockwise_distances)
    # 计算完全距离矩阵 S
    S = pairwise_distances(X, Y, metric=metric)
    # 断言块矩阵和完全距离矩阵 S 在指定容差下非常接近
    assert_allclose(blockwise_distances, S, atol=1e-7)


# 使用参数化装饰器定义测试函数，测试按块计算成对距离是否正确
@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_pairwise_distances_chunked_diagonal(metric, global_dtype):
    # 创建随机数发生器对象
    rng = np.random.RandomState(0)
    # 生成具有大尺度浮点数随机值的数据集 X
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    # 获取 X 的成对距离块列表
    chunks = list(pairwise_distances_chunked(X, working_memory=1, metric=metric))
    # 断言成对距离块数量大于 1
    assert len(chunks) > 1
    # 断言所有对角线元素非常接近 0
    assert_allclose(np.diag(np.vstack(chunks)), 0, rtol=1e-10)


# 使用参数化装饰器定义测试函数，测试并行计算成对距离的对角线元素是否正确
@pytest.mark.parametrize("metric", ("euclidean", "l2", "sqeuclidean"))
def test_parallel_pairwise_distances_diagonal(metric, global_dtype):
    # 创建随机数发生器对象
    rng = np.random.RandomState(0)
    # 生成具有大尺度浮点数随机值的数据集 X
    X = rng.normal(size=(1000, 10), scale=1e10).astype(global_dtype, copy=False)
    # 计算 X 的完全距离矩阵
    distances = pairwise_distances(X, metric=metric, n_jobs=2)
    # 断言距离矩阵的对角线元素非常接近 0
    assert_allclose(np.diag(distances), 0, atol=1e-10)


# 使用忽略警告装饰器定义测试函数，测试按块计算成对距离的辅助函数
@ignore_warnings
def test_pairwise_distances_chunked(global_dtype):
    # 测试按块计算成对距离的辅助函数
    rng = np.random.RandomState(0)
    # 创建随机数据集 X
    X = rng.random_sample((200, 4)).astype(global_dtype, copy=False)
    # 测试欧几里得距离是否与函数调用等效
    check_pairwise_distances_chunked(X, None, working_memory=1, metric="euclidean")
    # 测试小内存量
    for power in range(-16, 0):
        check_pairwise_distances_chunked(
            X, None, working_memory=2**power, metric="euclidean"
        )
    # 将 X 转换为列表形式
    check_pairwise_distances_chunked(
        X.tolist(), None, working_memory=1, metric="euclidean"
    )
    # 测试与 Y 不同的情况下的欧几里得距离
    Y = rng.random_sample((100, 4)).astype(global_dtype, copy=False)
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="euclidean")
    check_pairwise_distances_chunked(
        X.tolist(), Y.tolist(), working_memory=1, metric="euclidean"
    )
    # 测试极大工作内存
    check_pairwise_distances_chunked(X, Y, working_memory=10000, metric="euclidean")
    # "cityblock" 使用 scikit-learn 的度量，而 cityblock 函数属于 scipy.spatial 库
    check_pairwise_distances_chunked(X, Y, working_memory=1, metric="cityblock")

    # 测试预计算的距离矩阵一次性返回
    D = pairwise_distances(X)
    gen = pairwise_distances_chunked(D, working_memory=2**-16, metric="precomputed")
    # 断言 gen 是一个生成器对象
    assert isinstance(gen, GeneratorType)
    # 断言生成器的第一个元素是完全距离矩阵 D
    assert next(gen) is D
    # 检查生成器抛出 StopIteration 异常
    with pytest.raises(StopIteration):
        next(gen)
# 使用 pytest.mark.parametrize 装饰器定义测试函数 test_euclidean_distances_known_result，参数 x_array_constr 和 y_array_constr 分别为 np.array 和 CSR_CONTAINERS 中的各项。
# 为每个参数设置 ID，第一个参数的 ID 是 "dense"，后续参数的 ID 分别为 CSR 容器的类名。
@pytest.mark.parametrize(
    "x_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_known_result(x_array_constr, y_array_constr):
    # 检查对已知结果计算两两欧氏距离的正确性
    X = x_array_constr([[0]])  # 使用 x_array_constr 创建数组 X
    Y = y_array_constr([[1], [2]])  # 使用 y_array_constr 创建数组 Y
    D = euclidean_distances(X, Y)  # 计算 X 和 Y 之间的欧氏距离
    assert_allclose(D, [[1.0, 2.0]])  # 断言 D 的值与预期结果 [[1.0, 2.0]] 相近


@pytest.mark.parametrize(
    "y_array_constr",
    [np.array] + CSR_CONTAINERS,
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
)
def test_euclidean_distances_with_norms(global_dtype, y_array_constr):
    # 检查在使用正确的 {X,Y}_norm_squared 时能得到正确答案，以及在使用错误的 {X,Y}_norm_squared 时能得到错误答案
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10)).astype(global_dtype, copy=False)  # 生成随机数组 X
    Y = rng.random_sample((20, 10)).astype(global_dtype, copy=False)  # 生成随机数组 Y

    # 只有当数据类型为 float64 时才会使用 norms
    X_norm_sq = (X.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)  # 计算 X 的每行平方和的 float64 形式
    Y_norm_sq = (Y.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)  # 计算 Y 的每行平方和的 float64 形式

    Y = y_array_constr(Y)  # 使用 y_array_constr 转换 Y

    # 分别计算不同参数下的欧氏距离
    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)

    # 断言使用不同 norms 计算得到的结果都与 D1 相近
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)

    # 断言使用错误的 norms 计算会引发 AssertionError
    wrong_D = euclidean_distances(
        X,
        Y,
        X_norm_squared=np.zeros_like(X_norm_sq),
        Y_norm_squared=np.zeros_like(Y_norm_sq),
    )
    with pytest.raises(AssertionError):
        assert_allclose(wrong_D, D1)


@pytest.mark.parametrize("symmetric", [True, False])
def test_euclidean_distances_float32_norms(global_random_seed, symmetric):
    # 针对 #27621 的非回归测试
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((10, 10))  # 生成随机数组 X
    Y = X if symmetric else rng.random_sample((20, 10))  # 根据 symmetric 决定生成随机数组 Y 或复制 X
    X_norm_sq = (X.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)  # 计算 X 的每行平方和的 float32 形式
    Y_norm_sq = (Y.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)  # 计算 Y 的每行平方和的 float32 形式
    D1 = euclidean_distances(X, Y)  # 计算 X 和 Y 之间的欧氏距离
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)  # 使用 X_norm_squared 计算欧氏距离
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)  # 使用 Y_norm_squared 计算欧氏距离
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)  # 同时使用 X_norm_squared 和 Y_norm_squared 计算欧氏距离
    assert_allclose(D2, D1)  # 断言 D2 与 D1 相近
    assert_allclose(D3, D1)  # 断言 D3 与 D1 相近
    assert_allclose(D4, D1)  # 断言 D4 与 D1 相近


def test_euclidean_distances_norm_shapes():
    # 检查 norms 的所有可接受形状或适当的错误消息
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10))  # 生成随机数组 X
    # 生成一个 20x10 的随机数组 Y
    Y = rng.random_sample((20, 10))
    
    # 计算 X 中每个样本的平方和，沿着 axis=1（即每行）的方向求和
    X_norm_squared = (X**2).sum(axis=1)
    # 计算 Y 中每个样本的平方和，沿着 axis=1（即每行）的方向求和
    Y_norm_squared = (Y**2).sum(axis=1)
    
    # 计算 X 和 Y 之间的欧氏距离，使用事先计算好的 X_norm_squared 和 Y_norm_squared
    D1 = euclidean_distances(
        X, Y, X_norm_squared=X_norm_squared, Y_norm_squared=Y_norm_squared
    )
    
    # 将 X_norm_squared 和 Y_norm_squared 调整为列向量后再计算欧氏距离
    D2 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(-1, 1),
        Y_norm_squared=Y_norm_squared.reshape(-1, 1),
    )
    
    # 将 X_norm_squared 和 Y_norm_squared 调整为行向量后再计算欧氏距离
    D3 = euclidean_distances(
        X,
        Y,
        X_norm_squared=X_norm_squared.reshape(1, -1),
        Y_norm_squared=Y_norm_squared.reshape(1, -1),
    )
    
    # 断言 D2 和 D1 的值近似相等
    assert_allclose(D2, D1)
    # 断言 D3 和 D1 的值近似相等
    assert_allclose(D3, D1)
    
    # 使用 pytest 的断言，检查传递给 euclidean_distances 的 X_norm_squared 维度是否兼容
    with pytest.raises(ValueError, match="Incompatible dimensions for X"):
        euclidean_distances(X, Y, X_norm_squared=X_norm_squared[:5])
    
    # 使用 pytest 的断言，检查传递给 euclidean_distances 的 Y_norm_squared 维度是否兼容
    with pytest.raises(ValueError, match="Incompatible dimensions for Y"):
        euclidean_distances(X, Y, Y_norm_squared=Y_norm_squared[:5])
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试函数提供多组参数
    "x_array_constr",  # 参数名为 x_array_constr
    [np.array] + CSR_CONTAINERS,  # 参数值为 np.array 和 CSR_CONTAINERS 列表中的元素
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],  # 参数化标识符，对应每组参数的名称
)
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试函数提供多组参数
    "y_array_constr",  # 参数名为 y_array_constr
    [np.array] + CSR_CONTAINERS,  # 参数值为 np.array 和 CSR_CONTAINERS 列表中的元素
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],  # 参数化标识符，对应每组参数的名称
)
def test_euclidean_distances(global_dtype, x_array_constr, y_array_constr):
    # 检查欧几里得距离函数是否与 scipy 的 cdist 函数在提供 X 和 Y != X 时给出相同结果
    rng = np.random.RandomState(0)  # 使用种子为 0 的随机数生成器创建随机状态对象
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)  # 生成随机样本矩阵 X，并指定数据类型
    X[X < 0.8] = 0  # 将 X 中小于 0.8 的元素设为 0
    Y = rng.random_sample((10, 10)).astype(global_dtype, copy=False)  # 生成随机样本矩阵 Y，并指定数据类型
    Y[Y < 0.8] = 0  # 将 Y 中小于 0.8 的元素设为 0

    expected = cdist(X, Y)  # 计算 X 和 Y 之间的欧几里得距离作为期望结果

    X = x_array_constr(X)  # 使用 x_array_constr 构造 X 的数组形式
    Y = y_array_constr(Y)  # 使用 y_array_constr 构造 Y 的数组形式
    distances = euclidean_distances(X, Y)  # 计算 X 和 Y 之间的欧几里得距离

    # 默认的相对容差 rtol=1e-7 太接近 float32 的精度，因舍入误差而失败
    assert_allclose(distances, expected, rtol=1e-6)  # 使用 assert_allclose 断言距离数组与期望值的接近程度
    assert distances.dtype == global_dtype  # 断言距离数组的数据类型与全局数据类型一致


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试函数提供多组参数
    "x_array_constr",  # 参数名为 x_array_constr
    [np.array] + CSR_CONTAINERS,  # 参数值为 np.array 和 CSR_CONTAINERS 列表中的元素
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],  # 参数化标识符，对应每组参数的名称
)
def test_euclidean_distances_sym(global_dtype, x_array_constr):
    # 检查欧几里得距离函数是否与 scipy 的 pdist 函数在只提供 X 时给出相同结果
    rng = np.random.RandomState(0)  # 使用种子为 0 的随机数生成器创建随机状态对象
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)  # 生成随机样本矩阵 X，并指定数据类型
    X[X < 0.8] = 0  # 将 X 中小于 0.8 的元素设为 0

    expected = squareform(pdist(X))  # 使用 pdist 计算 X 的距离矩阵，并将结果转换为方阵形式

    X = x_array_constr(X)  # 使用 x_array_constr 构造 X 的数组形式
    distances = euclidean_distances(X)  # 计算 X 之间的欧几里得距离

    # 默认的相对容差 rtol=1e-7 太接近 float32 的精度，因舍入误差而失败
    assert_allclose(distances, expected, rtol=1e-6)  # 使用 assert_allclose 断言距离数组与期望值的接近程度
    assert distances.dtype == global_dtype  # 断言距离数组的数据类型与全局数据类型一致


@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])  # 使用 pytest 的参数化装饰器，为测试函数提供多组 batch_size 参数
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试函数提供多组参数
    "x_array_constr",  # 参数名为 x_array_constr
    [np.array] + CSR_CONTAINERS,  # 参数值为 np.array 和 CSR_CONTAINERS 列表中的元素
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],  # 参数化标识符，对应每组参数的名称
)
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，为测试函数提供多组参数
    "y_array_constr",  # 参数名为 y_array_constr
    [np.array] + CSR_CONTAINERS,  # 参数值为 np.array 和 CSR_CONTAINERS 列表中的元素
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],  # 参数化标识符，对应每组参数的名称
)
def test_euclidean_distances_upcast(batch_size, x_array_constr, y_array_constr):
    # 检查当 Y != X 时的批处理处理 (#13910)
    rng = np.random.RandomState(0)  # 使用种子为 0 的随机数生成器创建随机状态对象
    X = rng.random_sample((100, 10)).astype(np.float32)  # 生成随机样本矩阵 X，并指定数据类型为 float32
    X[X < 0.8] = 0  # 将 X 中小于 0.8 的元素设为 0
    Y = rng.random_sample((10, 10)).astype(np.float32)  # 生成随机样本矩阵 Y，并指定数据类型为 float32
    Y[Y < 0.8] = 0  # 将 Y 中小于 0.8 的元素设为 0

    expected = cdist(X, Y)  # 计算 X 和 Y 之间的欧几里得距离作为期望结果

    X = x_array_constr(X)  # 使用 x_array_constr 构造 X 的数组形式
    Y = y_array_constr(Y)  # 使用 y_array_constr 构造 Y 的数组形式
    distances = _euclidean_distances_upcast(X, Y=Y, batch_size=batch_size)  # 调用 _euclidean_distances_upcast 处理批次距离计算，并进行类型提升
    distances = np.sqrt(np.maximum(distances, 0))  # 对距离数组进行元素级的平方根处理和截断处理

    # 默认的相对容差 rtol=1e-7 太接近 float32 的精度，因舍入误差而失败
    assert_allclose(distances, expected, rtol=1e-6)  # 使用 assert_allclose 断言距离数组与期望值的接近程度
    # 创建一个包含字符串"dense"和CSR_CONTAINERS中每个容器类名的列表
    ids=["dense"] + [container.__name__ for container in CSR_CONTAINERS],
# 定义一个测试函数，用于验证在向上转型时欧几里得距离的计算是否正确
def test_euclidean_distances_upcast_sym(batch_size, x_array_constr):
    # 使用种子为0的随机数生成器创建一个形状为(100, 10)的随机浮点数数组 X
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    # 将 X 中小于0.8的元素置为0
    X[X < 0.8] = 0

    # 计算 X 中点与点之间的欧几里得距离的平方，并将其转换为成对距离形式
    expected = squareform(pdist(X))

    # 使用 x_array_constr 函数处理 X，然后计算 X 与自身的欧几里得距离
    distances = _euclidean_distances_upcast(X, Y=X, batch_size=batch_size)
    # 将距离矩阵中的负数取平方根并确保非负
    distances = np.sqrt(np.maximum(distances, 0))

    # 默认的相对误差容差 rtol=1e-7 对于 float32 精度过于严格，因而使用 rtol=1e-6
    assert_allclose(distances, expected, rtol=1e-6)


# 使用参数化测试，分别测试不同的数据类型（dtype）、eps 和相对误差容差（rtol）
@pytest.mark.parametrize(
    "dtype, eps, rtol",
    [
        (np.float32, 1e-4, 1e-5),
        pytest.param(
            np.float64,
            1e-8,
            0.99,
            marks=pytest.mark.xfail(reason="failing due to lack of precision"),
        ),
    ],
)
# 参数化测试，测试不同的维度（dim）
@pytest.mark.parametrize("dim", [1, 1000000])
def test_euclidean_distances_extreme_values(dtype, eps, rtol, dim):
    # 检查在 float32 输入情况下由于向上转型，欧几里得距离的正确性。在 float64 下仍存在精度问题。
    X = np.array([[1.0] * dim], dtype=dtype)
    Y = np.array([[1.0 + eps] * dim], dtype=dtype)

    # 计算 X 和 Y 之间的欧几里得距离
    distances = euclidean_distances(X, Y)
    # 计算 X 和 Y 之间的真实距离（使用 cdist 函数）
    expected = cdist(X, Y)

    # 使用相对误差容差 rtol=1e-5 来验证距离矩阵的近似性
    assert_allclose(distances, expected, rtol=1e-5)


# 参数化测试，测试是否有 NaN 值时的欧几里得距离计算是否正确
@pytest.mark.parametrize("squared", [True, False])
def test_nan_euclidean_distances_equal_to_euclidean_distance(squared):
    # 生成一个形状为 (3, 4) 的随机数矩阵 X 和一个形状为 (4, 4) 的随机数矩阵 Y
    rng = np.random.RandomState(1337)
    X = rng.randn(3, 4)
    Y = rng.randn(4, 4)

    # 计算 X 和 Y 之间的普通欧几里得距离和带有 NaN 的欧几里得距离
    normal_distance = euclidean_distances(X, Y=Y, squared=squared)
    nan_distance = nan_euclidean_distances(X, Y=Y, squared=squared)
    # 使用相对误差容差来验证两者的近似性
    assert_allclose(normal_distance, nan_distance)


# 参数化测试，测试当输入包含无穷大值时，是否会抛出 ValueError 异常
@pytest.mark.parametrize("X", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]])])
@pytest.mark.parametrize("Y", [np.array([[np.inf, 0]]), np.array([[0, -np.inf]]), None])
def test_nan_euclidean_distances_infinite_values(X, Y):
    # 确保当输入包含无穷大值时会引发 ValueError 异常
    with pytest.raises(ValueError) as excinfo:
        nan_euclidean_distances(X, Y=Y)

    exp_msg = "Input contains infinity or a value too large for dtype('float64')."
    assert exp_msg == str(excinfo.value)


# 参数化测试，测试当输入包含 NaN 值时的 2x2 情况下的欧几里得距离计算是否正确
@pytest.mark.parametrize(
    "X, X_diag, missing_value",
    [
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[np.nan, 1], [1, np.nan]]), np.nan, np.nan),
        (np.array([[np.nan, 1], [np.nan, 0]]), np.sqrt(2), np.nan),
        (np.array([[0, np.nan], [1, np.nan]]), np.sqrt(2), np.nan),
        (np.array([[0, 1], [1, 0]]), np.sqrt(2), -1),
        (np.array([[0, 1], [1, -1]]), np.sqrt(2), -1),
        (np.array([[-1, 1], [1, -1]]), np.nan, -1),
        (np.array([[-1, 1], [-1, 0]]), np.sqrt(2), -1),
        (np.array([[0, -1], [1, -1]]), np.sqrt(2), -1),
    ],
)
def test_nan_euclidean_distances_2x2(X, X_diag, missing_value):
    # 预期的 2x2 情况下的欧几里得距离矩阵
    exp_dist = np.array([[0.0, X_diag], [X_diag, 0]])
    # 计算给定数据集 X 的欧几里得距离矩阵，其中可能包含缺失值，使用默认设置
    dist = nan_euclidean_distances(X, missing_values=missing_value)
    # 断言计算得到的距离矩阵与预期的距离矩阵 exp_dist 很接近（几乎相等）
    assert_allclose(exp_dist, dist)

    # 计算给定数据集 X 的欧几里得距离的平方，同样可能包含缺失值
    dist_sq = nan_euclidean_distances(X, squared=True, missing_values=missing_value)
    # 断言计算得到的距离矩阵的平方与预期的距离矩阵 exp_dist 的平方很接近
    assert_allclose(exp_dist**2, dist_sq)

    # 计算数据集 X 与其自身之间的欧几里得距离矩阵，处理可能的缺失值
    dist_two = nan_euclidean_distances(X, X, missing_values=missing_value)
    # 断言计算得到的自身距离矩阵与预期的距离矩阵 exp_dist 很接近
    assert_allclose(exp_dist, dist_two)

    # 计算数据集 X 与其深拷贝之间的欧几里得距离矩阵，处理可能的缺失值
    dist_two_copy = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    # 断言计算得到的深拷贝距离矩阵与预期的距离矩阵 exp_dist 很接近
    assert_allclose(exp_dist, dist_two_copy)
@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_complete_nan(missing_value):
    # 创建一个包含缺失值的测试输入矩阵 X
    X = np.array([[missing_value, missing_value], [0, 1]])

    # 预期的欧氏距离矩阵，其中缺失值位置为 NaN
    exp_dist = np.array([[np.nan, np.nan], [np.nan, 0]])

    # 调用 nan_euclidean_distances 函数计算距离矩阵 dist，并断言其与 exp_dist 的吻合性
    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)

    # 再次调用 nan_euclidean_distances 函数，传入 X 的副本，进行距离计算，并断言其与 exp_dist 的吻合性
    dist = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_not_trival(missing_value):
    # 创建包含多个缺失值的输入矩阵 X 和 Y
    X = np.array(
        [
            [1.0, missing_value, 3.0, 4.0, 2.0],
            [missing_value, 4.0, 6.0, 1.0, missing_value],
            [3.0, missing_value, missing_value, missing_value, 1.0],
        ]
    )

    Y = np.array(
        [
            [missing_value, 7.0, 7.0, missing_value, 2.0],
            [missing_value, missing_value, 5.0, 4.0, 7.0],
            [missing_value, missing_value, missing_value, 4.0, 5.0],
        ]
    )

    # 检查距离矩阵的对称性
    D1 = nan_euclidean_distances(X, Y, missing_values=missing_value)
    D2 = nan_euclidean_distances(Y, X, missing_values=missing_value)
    assert_almost_equal(D1, D2.T)

    # 使用显式公式和 squared=True 进行距离计算，进行断言验证
    assert_allclose(
        nan_euclidean_distances(
            X[:1], Y[:1], squared=True, missing_values=missing_value
        ),
        [[5.0 / 2.0 * ((7 - 3) ** 2 + (2 - 2) ** 2)]],
    )

    # 使用显式公式和 squared=False 进行距离计算，进行断言验证
    assert_allclose(
        nan_euclidean_distances(
            X[1:2], Y[1:2], squared=False, missing_values=missing_value
        ),
        [[np.sqrt(5.0 / 2.0 * ((6 - 5) ** 2 + (1 - 4) ** 2))]],
    )

    # 检查当 Y = X 或其副本被显式传递时的距离矩阵，并进行断言验证
    D3 = nan_euclidean_distances(X, missing_values=missing_value)
    D4 = nan_euclidean_distances(X, X, missing_values=missing_value)
    D5 = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(D3, D4)
    assert_allclose(D4, D5)

    # 检查 copy=True 和 copy=False 的情况下的距离矩阵，并进行断言验证
    D6 = nan_euclidean_distances(X, Y, copy=True)
    D7 = nan_euclidean_distances(X, Y, copy=False)
    assert_allclose(D6, D7)


@pytest.mark.parametrize("missing_value", [np.nan, -1])
def test_nan_euclidean_distances_one_feature_match_positive(missing_value):
    # X 包含两个样本，第一个特征是唯一一个在两个样本中都不是 NaN 的特征。
    # 使用 squared=True 的情况下，距离矩阵应为非负数；非平方的版本应该接近于 0。
    X = np.array(
        [
            [-122.27, 648.0, missing_value, 37.85],
            [-122.27, missing_value, 2.34701493, missing_value],
        ]
    )

    # 计算距离的平方，并断言所有元素都大于等于 0
    dist_squared = nan_euclidean_distances(
        X, missing_values=missing_value, squared=True
    )
    assert np.all(dist_squared >= 0)

    # 计算距离，并断言其接近于 0
    dist = nan_euclidean_distances(X, missing_values=missing_value, squared=False)
    assert_allclose(dist, 0.0)
def test_cosine_distances():
    # 检查余弦距离的成对计算
    rng = np.random.RandomState(1337)  # 使用种子1337初始化随机数生成器
    x = np.abs(rng.rand(910))  # 生成910个随机数并取绝对值
    XA = np.vstack([x, x])  # 垂直堆叠x两次，构成一个2行910列的数组
    D = cosine_distances(XA)  # 计算X的余弦距离矩阵
    assert_allclose(D, [[0.0, 0.0], [0.0, 0.0]], atol=1e-10)  # 断言D的值接近[[0.0, 0.0], [0.0, 0.0]]，容差为1e-10
    # 检查所有元素都在区间[0, 2]内
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)
    # 检查对角线元素为0
    assert_allclose(D[np.diag_indices_from(D)], [0.0, 0.0])

    XB = np.vstack([x, -x])  # 垂直堆叠x和-x，构成一个2行910列的数组
    D2 = cosine_distances(XB)  # 计算XB的余弦距离矩阵
    # 检查所有元素都在区间[0, 2]内
    assert np.all(D2 >= 0.0)
    assert np.all(D2 <= 2.0)
    # 检查对角线元素为0，非对角线元素为2
    assert_allclose(D2, [[0.0, 2.0], [2.0, 0.0]])

    # 检查大型随机矩阵
    X = np.abs(rng.rand(1000, 5000))  # 生成1000行5000列的随机数并取绝对值
    D = cosine_distances(X)  # 计算X的余弦距离矩阵
    # 检查对角线元素为0
    assert_allclose(D[np.diag_indices_from(D)], [0.0] * D.shape[0])
    # 检查所有元素都在区间[0, 2]内
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)


def test_haversine_distances():
    # 检查球面距离的哈文赛因计算
    def slow_haversine_distances(x, y):
        # 计算慢速哈文赛因距离
        diff_lat = y[0] - x[0]  # 计算纬度差
        diff_lon = y[1] - x[1]  # 计算经度差
        a = np.sin(diff_lat / 2) ** 2 + (
            np.cos(x[0]) * np.cos(y[0]) * np.sin(diff_lon / 2) ** 2
        )  # 计算哈文赛因距离公式中的a部分
        c = 2 * np.arcsin(np.sqrt(a))  # 计算球面距离
        return c

    rng = np.random.RandomState(0)  # 使用种子0初始化随机数生成器
    X = rng.random_sample((5, 2))  # 生成5行2列的随机数
    Y = rng.random_sample((10, 2))  # 生成10行2列的随机数
    D1 = np.array([[slow_haversine_distances(x, y) for y in Y] for x in X])  # 使用慢速函数计算哈文赛因距离矩阵D1
    D2 = haversine_distances(X, Y)  # 使用快速函数计算哈文赛因距离矩阵D2
    assert_allclose(D1, D2)  # 断言D1和D2的值接近
    # 测试当X的特征数量不等于2时，哈文赛因距离函数不接受X
    X = rng.random_sample((10, 3))  # 生成10行3列的随机数
    err_msg = "Haversine distance only valid in 2 dimensions"  # 错误信息字符串
    with pytest.raises(ValueError, match=err_msg):  # 断言抛出值错误且错误信息符合预期
        haversine_distances(X)


# 成对距离


def test_paired_euclidean_distances():
    # 检查成对欧氏距离的计算
    X = [[0], [0]]  # 两个点的欧氏距离
    Y = [[1], [2]]  # 两个点的欧氏距离
    D = paired_euclidean_distances(X, Y)  # 计算X和Y的成对欧氏距离
    assert_allclose(D, [1.0, 2.0])  # 断言D的值接近[1.0, 2.0]


def test_paired_manhattan_distances():
    # 检查成对曼哈顿距离的计算
    X = [[0], [0]]  # 两个点的曼哈顿距离
    Y = [[1], [2]]  # 两个点的曼哈顿距离
    D = paired_manhattan_distances(X, Y)  # 计算X和Y的成对曼哈顿距离
    assert_allclose(D, [1.0, 2.0])  # 断言D的值接近[1.0, 2.0]


def test_paired_cosine_distances():
    # 检查成对余弦距离的计算
    X = [[0], [0]]  # 两个点的余弦距离
    Y = [[1], [2]]  # 两个点的余弦距离
    D = paired_cosine_distances(X, Y)  # 计算X和Y的成对余弦距离
    assert_allclose(D, [0.5, 0.5])  # 断言D的值接近[0.5, 0.5]


def test_chi_square_kernel():
    rng = np.random.RandomState(0)  # 使用种子0初始化随机数生成器
    X = rng.random_sample((5, 4))  # 生成5行4列的随机数
    Y = rng.random_sample((10, 4))  # 生成10行4列的随机数
    K_add = additive_chi2_kernel(X, Y)  # 计算加性卡方核矩阵K_add
    gamma = 0.1  # 设置gamma值
    K = chi2_kernel(X, Y, gamma=gamma)  # 计算卡方核矩阵K
    assert K.dtype == float  # 断言K的数据类型为float
    # 对 X 和 Y 中的每对数据计算 Chi-squared kernel
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            # 计算 Chi-squared kernel 的值
            chi2 = -np.sum((x - y) ** 2 / (x + y))
            # 计算 Chi-squared kernel 的指数形式
            chi2_exp = np.exp(gamma * chi2)
            # 断言 K_add[i, j] 等于计算得到的 Chi-squared kernel 值
            assert_almost_equal(K_add[i, j], chi2)
            # 断言 K[i, j] 等于计算得到的 Chi-squared kernel 的指数形式
            assert_almost_equal(K[i, j], chi2_exp)

    # 检查对角线上的元素是否为 1（数据与自身的核应为 1）
    K = chi2_kernel(Y)
    assert_array_equal(np.diag(K), 1)
    # 检查非对角线上的元素是否大于 0
    assert np.all(K > 0)
    # 检查非对角线上的元素是否小于 1
    assert np.all(K - np.diag(np.diag(K)) < 1)

    # 检查保持 float32 类型不变
    X = rng.random_sample((5, 4)).astype(np.float32)
    Y = rng.random_sample((10, 4)).astype(np.float32)
    K = chi2_kernel(X, Y)
    assert K.dtype == np.float32

    # 检查整数类型是否被转换为 float 类型
    X = rng.random_sample((10, 4)).astype(np.int32)
    K = chi2_kernel(X, X)
    assert np.isfinite(K).all()
    assert K.dtype == float

    # 检查相似数据的核大于不相似数据的核
    X = [[0.3, 0.7], [1.0, 0]]
    Y = [[0, 1], [0.9, 0.1]]
    K = chi2_kernel(X, Y)
    assert K[0, 0] > K[0, 1]
    assert K[1, 1] > K[1, 0]

    # 测试负输入值引发 ValueError 异常
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, -1]], [[-1, -1]])
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[-1, -1]])

    # 检查 X 和 Y 中的特征数不同引发 ValueError 异常
    with pytest.raises(ValueError):
        chi2_kernel([[0, 1]], [[0.2, 0.2, 0.6]])
@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
# 定义一个参数化测试，测试各种核函数的对称性
def test_kernel_symmetry(kernel):
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X
    X = rng.random_sample((5, 4))
    # 计算核函数在 X 上的输出矩阵 K
    K = kernel(X, X)
    # 断言核函数的输出矩阵 K 应该是对称的
    assert_allclose(K, K.T, 15)


@pytest.mark.parametrize(
    "kernel",
    (
        linear_kernel,
        polynomial_kernel,
        rbf_kernel,
        laplacian_kernel,
        sigmoid_kernel,
        cosine_similarity,
    ),
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 参数化测试，测试稀疏输入下各种核函数的行为
def test_kernel_sparse(kernel, csr_container):
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X
    X = rng.random_sample((5, 4))
    # 将 X 转换为稀疏格式
    X_sparse = csr_container(X)
    # 计算核函数在稠密输入 X 上的输出矩阵 K
    K = kernel(X, X)
    # 计算核函数在稀疏输入 X_sparse 上的输出矩阵 K2
    K2 = kernel(X_sparse, X_sparse)
    # 断言两个输出矩阵 K 和 K2 应该非常接近
    assert_allclose(K, K2)


def test_linear_kernel():
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X
    X = rng.random_sample((5, 4))
    # 计算线性核函数在输入 X 上的输出矩阵 K
    K = linear_kernel(X, X)
    # 断言线性核函数的对角线元素应该等于每个向量的平方范数
    assert_allclose(K.flat[::6], [linalg.norm(x) ** 2 for x in X])


def test_rbf_kernel():
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X
    X = rng.random_sample((5, 4))
    # 计算径向基函数（RBF）核在输入 X 上的输出矩阵 K
    K = rbf_kernel(X, X)
    # 断言径向基函数核的对角线元素应该都是 1
    assert_allclose(K.flat[::6], np.ones(5))


def test_laplacian_kernel():
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X
    X = rng.random_sample((5, 4))
    # 计算拉普拉斯核在输入 X 上的输出矩阵 K
    K = laplacian_kernel(X, X)
    # 断言拉普拉斯核的对角线元素应该都是 1
    assert_allclose(np.diag(K), np.ones(5))
    # 断言非对角线元素应该大于 0 且小于 1
    assert np.all(K > 0)
    assert np.all(K - np.diag(np.diag(K)) < 1)


@pytest.mark.parametrize(
    "metric, pairwise_func",
    [("linear", linear_kernel), ("cosine", cosine_similarity)],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 参数化测试，测试成对相似度计算的稀疏输出
def test_pairwise_similarity_sparse_output(metric, pairwise_func, csr_container):
    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X 和 3x4 的随机数组 Y
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    # 将 X 和 Y 转换为稀疏格式
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)

    # 计算稀疏输入 Xcsr 和 Ycsr 上的成对相似度，期望稀疏输出
    K1 = pairwise_func(Xcsr, Ycsr, dense_output=False)
    assert issparse(K1)

    # 计算稠密输入 X 和 Y 上的成对相似度，期望稠密输出，并且等于 K1
    K2 = pairwise_func(X, Y, dense_output=True)
    assert not issparse(K2)
    assert_allclose(K1.toarray(), K2)

    # 使用 pairwise_kernels 函数计算 X 和 Y 的核输出，应与稀疏输出 K1 相等
    K3 = pairwise_kernels(X, Y=Y, metric=metric)
    assert_allclose(K1.toarray(), K3)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 参数化测试，测试余弦相似度计算
def test_cosine_similarity(csr_container):
    # 测试余弦相似度计算

    # 使用随机种子初始化随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个 5x4 的随机数组 X 和 3x4 的随机数组 Y
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    # 将 X 和 Y 转换为稀疏格式
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)
    for X_, Y_ in ((X, None), (X, Y), (Xcsr, None), (Xcsr, Ycsr)):
        # 遍历四种不同的数据对(X_, Y_)，其中可能包括稠密矩阵X和Y，或稀疏矩阵Xcsr和Ycsr

        # 测试当数据已经通过L2范数归一化后，余弦核函数是否等同于线性核函数
        K1 = pairwise_kernels(X_, Y=Y_, metric="cosine")

        # 对X_进行L2范数归一化
        X_ = normalize(X_)

        # 如果Y_不为None，对Y_也进行L2范数归一化
        if Y_ is not None:
            Y_ = normalize(Y_)

        # 计算经过归一化后的数据对(X_, Y_)的线性核函数
        K2 = pairwise_kernels(X_, Y=Y_, metric="linear")

        # 使用 assert_allclose 函数检查K1和K2是否近似相等
        assert_allclose(K1, K2)
def test_check_dense_matrices():
    # 确保对稠密矩阵的成对数组检查工作正常。
    # 检查如果XB为None，则XA_checked和XB_checked引用相同对象。
    XA = np.resize(np.arange(40), (5, 8))  # 创建一个5x8的矩阵XA
    XA_checked, XB_checked = check_pairwise_arrays(XA, None)  # 调用函数检查XA和None
    assert XA_checked is XB_checked  # 断言XA_checked和XB_checked是同一个对象
    assert_array_equal(XA, XA_checked)  # 断言XA和XA_checked相等


def test_check_XB_returned():
    # 确保如果XA和XB都正确给定，它们返回相等。
    # 检查如果XB不为None，则返回相等。
    # 注意XB的第二维度与XA相同。
    XA = np.resize(np.arange(40), (5, 8))  # 创建一个5x8的矩阵XA
    XB = np.resize(np.arange(32), (4, 8))  # 创建一个4x8的矩阵XB
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)  # 调用函数检查XA和XB
    assert_array_equal(XA, XA_checked)  # 断言XA和XA_checked相等
    assert_array_equal(XB, XB_checked)  # 断言XB和XB_checked相等

    XB = np.resize(np.arange(40), (5, 8))  # 更新XB为一个5x8的矩阵
    XA_checked, XB_checked = check_paired_arrays(XA, XB)  # 调用函数检查XA和XB
    assert_array_equal(XA, XA_checked)  # 断言XA和XA_checked相等
    assert_array_equal(XB, XB_checked)  # 断言XB和XB_checked相等


def test_check_different_dimensions():
    # 确保如果维度不同则引发错误。
    XA = np.resize(np.arange(45), (5, 9))  # 创建一个5x9的矩阵XA
    XB = np.resize(np.arange(32), (4, 8))  # 创建一个4x8的矩阵XB
    with pytest.raises(ValueError):  # 断言会引发ValueError异常
        check_pairwise_arrays(XA, XB)  # 调用函数检查XA和XB

    XB = np.resize(np.arange(4 * 9), (4, 9))  # 创建一个4x9的矩阵XB
    with pytest.raises(ValueError):  # 断言会引发ValueError异常
        check_paired_arrays(XA, XB)  # 调用函数检查XA和XB


def test_check_invalid_dimensions():
    # 确保在输入为1维数组时引发错误。
    # 修改的测试不是1维的。在旧的测试中，数组在内部总是转换为2维。
    XA = np.arange(45).reshape(9, 5)  # 创建一个9x5的矩阵XA
    XB = np.arange(32).reshape(4, 8)  # 创建一个4x8的矩阵XB
    with pytest.raises(ValueError):  # 断言会引发ValueError异常
        check_pairwise_arrays(XA, XB)  # 调用函数检查XA和XB
    XA = np.arange(45).reshape(9, 5)  # 创建一个9x5的矩阵XA
    XB = np.arange(32).reshape(4, 8)  # 创建一个4x8的矩阵XB
    with pytest.raises(ValueError):  # 断言会引发ValueError异常
        check_pairwise_arrays(XA, XB)  # 调用函数检查XA和XB


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_check_sparse_arrays(csr_container):
    # 确保检查返回有效的稀疏矩阵。
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))  # 创建一个5x4的随机矩阵XA
    XA_sparse = csr_container(XA)  # 将XA转换为稀疏矩阵
    XB = rng.random_sample((5, 4))  # 创建一个5x4的随机矩阵XB
    XB_sparse = csr_container(XB)  # 将XB转换为稀疏矩阵
    XA_checked, XB_checked = check_pairwise_arrays(XA_sparse, XB_sparse)  # 调用函数检查稀疏矩阵XA和XB
    # 比较它们的差异，因为使用'=='测试csr矩阵的相等性不起作用。
    assert issparse(XA_checked)  # 断言XA_checked是稀疏矩阵
    assert abs(XA_sparse - XA_checked).sum() == 0  # 断言XA_sparse和XA_checked的差异为0
    assert issparse(XB_checked)  # 断言XB_checked是稀疏矩阵
    assert abs(XB_sparse - XB_checked).sum() == 0  # 断言XB_sparse和XB_checked的差异为0

    XA_checked, XA_2_checked = check_pairwise_arrays(XA_sparse, XA_sparse)  # 再次调用函数检查稀疏矩阵XA
    assert issparse(XA_checked)  # 断言XA_checked是稀疏矩阵
    assert abs(XA_sparse - XA_checked).sum() == 0  # 断言XA_sparse和XA_checked的差异为0
    assert issparse(XA_2_checked)  # 断言XA_2_checked是稀疏矩阵
    assert abs(XA_2_checked - XA_checked).sum() == 0  # 断言XA_2_checked和XA_checked的差异为0


def tuplify(X):
    # 将numpy矩阵（任何n维数组）转换为元组。
    s = X.shape  # 获取矩阵的形状
    if len(s) > 1:  # 如果维度大于1
        # 对输入中的每个子数组进行元组化。
        return tuple(tuplify(row) for row in X)  # 递归调用tuplify函数对每一行进行元组化
    else:
        # 如果输入是单维数组，则返回其内容的元组。
        # 生成器表达式，用于将单维数组 X 中的每个元素 r 放入元组中
        return tuple(r for r in X)
# 定义一个测试函数，用于检查输入是否为元组
def test_check_tuple_input():
    # 确保检查函数返回有效的元组。
    rng = np.random.RandomState(0)
    XA = rng.random_sample((5, 4))  # 生成一个随机的5x4的数组
    XA_tuples = tuplify(XA)  # 将数组转换为元组
    XB = rng.random_sample((5, 4))  # 生成另一个随机的5x4的数组
    XB_tuples = tuplify(XB)  # 将数组转换为元组
    XA_checked, XB_checked = check_pairwise_arrays(XA_tuples, XB_tuples)  # 检查并返回校验后的元组
    assert_array_equal(XA_tuples, XA_checked)  # 断言两个元组是否相等
    assert_array_equal(XB_tuples, XB_checked)  # 断言两个元组是否相等


def test_check_preserve_type():
    # 确保类型 float32 被保留。
    XA = np.resize(np.arange(40), (5, 8)).astype(np.float32)  # 创建一个大小为5x8的float32类型的数组
    XB = np.resize(np.arange(40), (5, 8)).astype(np.float32)  # 创建另一个大小为5x8的float32类型的数组

    XA_checked, XB_checked = check_pairwise_arrays(XA, None)  # 检查并返回校验后的数组
    assert XA_checked.dtype == np.float32  # 断言数组的数据类型为float32

    # both float32
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB)  # 检查并返回校验后的数组
    assert XA_checked.dtype == np.float32  # 断言数组的数据类型为float32
    assert XB_checked.dtype == np.float32  # 断言数组的数据类型为float32

    # mismatched A
    XA_checked, XB_checked = check_pairwise_arrays(XA.astype(float), XB)  # 检查并返回校验后的数组
    assert XA_checked.dtype == float  # 断言数组的数据类型为float
    assert XB_checked.dtype == float  # 断言数组的数据类型为float

    # mismatched B
    XA_checked, XB_checked = check_pairwise_arrays(XA, XB.astype(float))  # 检查并返回校验后的数组
    assert XA_checked.dtype == float  # 断言数组的数据类型为float
    assert XB_checked.dtype == float  # 断言数组的数据类型为float


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
@pytest.mark.parametrize(
    "dist_function", [pairwise_distances, pairwise_distances_chunked]
)
def test_pairwise_distances_data_derived_params(n_jobs, metric, dist_function):
    # 检查在度量有数据导出参数时，pairwise_distances在顺序和并行运行时是否给出相同的结果。
    with config_context(working_memory=0.1):  # 使用配置上下文确保有超过1个块
        rng = np.random.RandomState(0)
        X = rng.random_sample((100, 10))  # 生成一个大小为100x10的随机数组

        expected_dist = squareform(pdist(X, metric=metric))  # 计算预期的距离矩阵
        dist = np.vstack(tuple(dist_function(X, metric=metric, n_jobs=n_jobs)))  # 调用距离函数计算距离矩阵

        assert_allclose(dist, expected_dist)  # 断言计算得到的距离矩阵与预期的距离矩阵是否相近


@pytest.mark.parametrize("metric", ["seuclidean", "mahalanobis"])
def test_pairwise_distances_data_derived_params_error(metric):
    # 检查当Y被传入但度量有未由用户提供的数据导出参数时，pairwise_distances是否会引发错误。
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))  # 生成一个大小为100x10的随机数组
    Y = rng.random_sample((100, 10))  # 生成另一个大小为100x10的随机数组

    with pytest.raises(
        ValueError,
        match=rf"The '(V|VI)' parameter is required for the " rf"{metric} metric",
    ):
        pairwise_distances(X, Y, metric=metric)  # 调用距离计算函数，预期引发特定错误


@pytest.mark.parametrize(
    "metric",
    [
        "braycurtis",
        "canberra",
        "chebyshev",
        "correlation",
        "hamming",
        "mahalanobis",
        "minkowski",
        "seuclidean",
        "sqeuclidean",
        "cityblock",
        "cosine",
        "euclidean",
    ],
)
@pytest.mark.parametrize("y_is_x", [True, False], ids=["Y is X", "Y is not X"])
# 检查在使用任何 scipy 度量标准比较数值向量时，pairwise_distances 是否与 pdist 和 cdist 给出相同结果
# 不考虑输入数据类型时
def test_numeric_pairwise_distances_datatypes(metric, global_dtype, y_is_x):
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)

    # 生成一个形状为 (5, 4) 的随机数数组 X，并将其类型转换为 global_dtype
    X = rng.random_sample((5, 4)).astype(global_dtype, copy=False)

    # 初始化空字典 params
    params = {}

    # 如果 y_is_x 为 True，则将 Y 设置为 X，并使用 pdist 计算期望距离
    if y_is_x:
        Y = X
        expected_dist = squareform(pdist(X, metric=metric))
    else:
        # 否则，生成一个形状为 (5, 4) 的随机数数组 Y，并将其类型转换为 global_dtype
        Y = rng.random_sample((5, 4)).astype(global_dtype, copy=False)
        # 使用 cdist 计算 X 和 Y 之间的期望距离
        expected_dist = cdist(X, Y, metric=metric)
        # 当 metric 为 "seuclidean" 时，预先计算 seuclidean 和 mahalanobis 的参数
        if metric == "seuclidean":
            params = {"V": np.var(np.vstack([X, Y]), axis=0, ddof=1, dtype=np.float64)}
        elif metric == "mahalanobis":
            params = {"VI": np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T}

    # 使用 pairwise_distances 计算 X 和 Y 之间的距离，使用给定的 metric 和预计的参数
    dist = pairwise_distances(X, Y, metric=metric, **params)

    # 使用 assert_allclose 检查计算得到的距离和期望的距离是否接近
    assert_allclose(dist, expected_dist)


# 使用参数化测试框架，检查以字符串列表作为输入时的 pairwise_distances 函数
@pytest.mark.parametrize(
    "X,Y,expected_distance",
    [
        (
            ["a", "ab", "abc"],
            None,
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        ),
        (
            ["a", "ab", "abc"],
            ["a", "ab"],
            [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]],
        ),
    ],
)
def test_pairwise_dist_custom_metric_for_string(X, Y, expected_distance):
    """检查使用字符串列表作为输入时的 pairwise_distances 函数。"""

    # 定义一个虚拟的字符串相似度函数
    def dummy_string_similarity(x, y):
        return np.abs(len(x) - len(y))

    # 使用 dummy_string_similarity 计算实际的距离矩阵
    actual_distance = pairwise_distances(X=X, Y=Y, metric=dummy_string_similarity)

    # 使用 assert_allclose 检查实际计算得到的距离和期望的距离是否接近
    assert_allclose(actual_distance, expected_distance)


# 检查在使用自定义度量时，pairwise_distances 是否将布尔值输入保持为布尔值
def test_pairwise_dist_custom_metric_for_bool():
    """检查在使用自定义度量时，pairwise_distances 是否将布尔值输入保持为布尔值。"""

    # 定义一个虚拟的布尔距离函数
    def dummy_bool_dist(v1, v2):
        # 使用位运算计算布尔值的距离
        return 1 - (v1 & v2).sum() / (v1 | v2).sum()

    # 创建一个布尔值数组 X
    X = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool)

    # 创建预期的布尔值距离矩阵
    expected_distance = np.array(
        [
            [0.0, 0.5, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.5, 0.0],
        ]
    )

    # 使用 dummy_bool_dist 计算实际的布尔值距离矩阵
    actual_distance = pairwise_distances(X=X, metric=dummy_bool_dist)

    # 使用 assert_allclose 检查实际计算得到的距离和期望的距离是否接近
    assert_allclose(actual_distance, expected_distance)


# 使用参数化测试框架，检查在稀疏矩阵容器中使用 Manhattan 距离时的 pairwise_distances 函数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_manhattan_readonly_dataset(csr_container):
    """检查在稀疏矩阵容器中使用 Manhattan 距离时的 pairwise_distances 函数。"""

    # 创建包含稀疏矩阵的列表 matrices1 和 matrices2
    matrices1 = [csr_container(np.ones((5, 5)))]
    matrices2 = [csr_container(np.ones((5, 5)))]

    # Joblib 将数据集存储为只读，这里进行非回归测试
    # 使用并行处理库Parallel来并行执行任务，指定同时执行2个任务，并且内存限制为0字节。
    # 对于每一对matrices1和matrices2中的矩阵m1和m2，调用manhattan_distances函数计算它们之间的曼哈顿距离。
    Parallel(n_jobs=2, max_nbytes=0)(
        delayed(manhattan_distances)(m1, m2) for m1, m2 in zip(matrices1, matrices2)
    )
    # 注：这行代码的目的是为了解决某个问题（#7981），并确保其执行成功。
```