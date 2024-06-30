# `D:\src\scipysrc\scipy\scipy\cluster\tests\test_vq.py`

```
# 导入警告模块，用于管理警告信息的显示
import warnings
# 导入系统模块，用于访问系统相关功能
import sys
# 从copy模块中导入deepcopy函数，用于深拷贝对象
from copy import deepcopy

# 导入numpy库，并将其命名为np，用于数值计算
import numpy as np
# 从numpy.testing模块中导入用于数组测试的函数和类
from numpy.testing import (
    assert_array_equal, assert_equal, assert_, suppress_warnings
)
# 导入pytest库，用于编写和运行测试
import pytest
# 从pytest模块中导入raises函数，并将其命名为assert_raises，用于断言异常抛出
from pytest import raises as assert_raises

# 导入scipy.cluster.vq模块中的相关函数和类，用于向量量化和聚类
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
                              ClusterError, _krandinit)
# 导入scipy.cluster模块中的_vq对象，用于向量量化和聚类
from scipy.cluster import _vq
# 从scipy.conftest模块中导入array_api_compatible装饰器，用于检查数组API兼容性
from scipy.conftest import array_api_compatible
# 导入scipy.sparse._sputils模块中的matrix类，用于稀疏矩阵操作
from scipy.sparse._sputils import matrix

# 导入scipy._lib._array_api模块中的相关函数和类，用于数组操作和API支持
from scipy._lib._array_api import (
    SCIPY_ARRAY_API, copy, cov, xp_assert_close, xp_assert_equal
)

# 定义pytest标记，指示本测试用例与数组API兼容，并使用skip_xp_backends修饰器跳过某些后端
pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
# 定义skip_xp_backends装饰器，用于跳过不兼容的数组后端

# 定义二维测试数据集TESTDATA_2D，包含多个浮点数，用于聚类和向量量化的测试
TESTDATA_2D = np.array([
    -2.2, 1.17, -1.63, 1.69, -2.04, 4.38, -3.09, 0.95, -1.7, 4.79, -1.68, 0.68,
    -2.26, 3.34, -2.29, 2.55, -1.72, -0.72, -1.99, 2.34, -2.75, 3.43, -2.45,
    2.41, -4.26, 3.65, -1.57, 1.87, -1.96, 4.03, -3.01, 3.86, -2.53, 1.28,
    -4.0, 3.95, -1.62, 1.25, -3.42, 3.17, -1.17, 0.12, -3.03, -0.27, -2.07,
    -0.55, -1.17, 1.34, -2.82, 3.08, -2.44, 0.24, -1.71, 2.48, -5.23, 4.29,
    -2.08, 3.69, -1.89, 3.62, -2.09, 0.26, -0.92, 1.07, -2.25, 0.88, -2.25,
    2.02, -4.31, 3.86, -2.03, 3.42, -2.76, 0.3, -2.48, -0.29, -3.42, 3.21,
    -2.3, 1.73, -2.84, 0.69, -1.81, 2.48, -5.24, 4.52, -2.8, 1.31, -1.67,
    -2.34, -1.18, 2.17, -2.17, 2.82, -1.85, 2.25, -2.45, 1.86, -6.79, 3.94,
    -2.33, 1.89, -1.55, 2.08, -1.36, 0.93, -2.51, 2.74, -2.39, 3.92, -3.33,
    2.99, -2.06, -0.9, -2.83, 3.35, -2.59, 3.05, -2.36, 1.85, -1.69, 1.8,
    -1.39, 0.66, -2.06, 0.38, -1.47, 0.44, -4.68, 3.77, -5.58, 3.44, -2.29,
    2.24, -1.04, -0.38, -1.85, 4.23, -2.88, 0.73, -2.59, 1.39, -1.34, 1.75,
    -1.95, 1.3, -2.45, 3.09, -1.99, 3.41, -5.55, 5.21, -1.73, 2.52, -2.17,
    0.85, -2.06, 0.49, -2.54, 2.07, -2.03, 1.3, -3.23, 3.09, -1.55, 1.44,
    -0.81, 1.1, -2.99, 2.92, -1.59, 2.18, -2.45, -0.73, -3.12, -1.3, -2.83,
    0.2, -2.77, 3.24, -1.98, 1.6, -4.59, 3.39, -4.85, 3.75, -2.25, 1.71, -3.28,
    3.38, -1.74, 0.88, -2.41, 1.92, -2.24, 1.19, -2.48, 1.06, -1.68, -0.62,
    -1.3, 0.39, -1.78, 2.35, -3.54, 2.44, -1.32, 0.66, -2.38, 2.76, -2.35,
    3.95, -1.86, 4.32, -2.01, -1.23, -1.79, 2.76, -2.13, -0.13, -5.25, 3.84,
    -2.24, 1.59, -4.85, 2.96, -2.41, 0.01, -0.43, 0.13, -3.92, 2.91, -1.75,
    -0.53, -1.69, 1.69, -1.09, 0.15, -2.11, 2.17, -1.53, 1.22, -2.1, -0.86,
    -2.56, 2.28, -3.02, 3.33, -1.12, 3.86, -2.18, -1.19, -3.03, 0.79, -0.83,
    0.97, -3.19, 1.45, -1.34, 1.28, -2.52, 4.22, -4.53, 3.22, -1.97, 1.75,
    -2.36, 3.19, -0.83, 1.53, -1.59, 1.86, -2.17, 2.3, -1.63, 2.71, -2.03,
    3.75, -2.57, -0.6, -1.47, 1.33, -1.95, 0.7, -1.65, 1.27, -1.42, 1.09,
    # 创建一个包含 200 行、每行包含两个元素的二维数组
    data = np.array([
        4.17, -1.29, 2.99, -5.92, 3.43, -1.83, 1.23, -1.24, -1.04, -2.56, 2.37,
        -3.26, 0.39, -4.63, 2.51, -4.52, 3.04, -1.7, 0.36, -1.41, 0.04, -2.1, 1.0,
        -1.87, 3.78, -4.32, 3.59, -2.24, 1.38, -1.99, -0.22, -1.87, 1.95, -0.84,
        2.17, -5.38, 3.56, -1.27, 2.9, -1.79, 3.31, -5.47, 3.85, -1.44, 3.69,
        -2.02, 0.37, -1.29, 0.33, -2.34, 2.56, -1.74, -1.27, -1.97, 1.22, -2.51,
        -0.16, -1.64, -0.96, -2.99, 1.4, -1.53, 3.31, -2.24, 0.45, -2.46, 1.71,
        -2.88, 1.56, -1.63, 1.46, -1.41, 0.68, -1.96, 2.76, -1.61,
        2.11
    ]).reshape((200, 2))
# 全局数据定义
X = np.array([[3.0, 3], [4, 3], [4, 2],
              [9, 2], [5, 1], [6, 2], [9, 4],
              [5, 2], [5, 4], [7, 4], [6, 5]])

# CODET1 数据定义，包含浮点数的二维数组
CODET1 = np.array([[3.0000, 3.0000],
                   [6.2000, 4.0000],
                   [5.8000, 1.8000]])

# CODET2 数据定义，包含浮点数的二维数组
CODET2 = np.array([[11.0/3, 8.0/3],
                   [6.7500, 4.2500],
                   [6.2500, 1.7500]])

# LABEL1 数据定义，包含整数的一维数组
LABEL1 = np.array([0, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1])


class TestWhiten:
    # 测试类 TestWhiten

    def test_whiten(self, xp):
        # 测试方法，使用参数 xp
        desired = xp.asarray([[5.08738849, 2.97091878],
                            [3.19909255, 0.69660580],
                            [4.51041982, 0.02640918],
                            [4.38567074, 0.95120889],
                            [2.32191480, 1.63195503]])

        obs = xp.asarray([[0.98744510, 0.82766775],
                          [0.62093317, 0.19406729],
                          [0.87545741, 0.00735733],
                          [0.85124403, 0.26499712],
                          [0.45067590, 0.45464607]])

        # 调用 xp_assert_close 函数，验证 whiten 函数的输出与期望值的接近程度
        xp_assert_close(whiten(obs), desired, rtol=1e-5)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'])
    def test_whiten_zero_std(self, xp):
        # 测试方法，使用参数 xp，在某些背景下跳过
        desired = xp.asarray([[0., 1.0, 2.86666544],
                              [0., 1.0, 1.32460034],
                              [0., 1.0, 3.74382172]])

        obs = xp.asarray([[0., 1., 0.74109533],
                          [0., 1., 0.34243798],
                          [0., 1., 0.96785929]])

        # 设置警告的处理方式，并调用 xp_assert_close 函数验证结果
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            xp_assert_close(whiten(obs), desired, rtol=1e-5)

            # 断言捕获的警告数量为1
            assert_equal(len(w), 1)
            # 断言捕获的最后一个警告类别是 RuntimeWarning 的子类
            assert_(issubclass(w[-1].category, RuntimeWarning))

    def test_whiten_not_finite(self, xp):
        # 测试方法，用于处理 xp 中的非有限值异常
        for bad_value in xp.nan, xp.inf, -xp.inf:
            obs = xp.asarray([[0.98744510, bad_value],
                              [0.62093317, 0.19406729],
                              [0.87545741, 0.00735733],
                              [0.85124403, 0.26499712],
                              [0.45067590, 0.45464607]])
            # 断言调用 whiten 函数时，对非有限值抛出 ValueError 异常
            assert_raises(ValueError, whiten, obs)

    @pytest.mark.skipif(SCIPY_ARRAY_API,
                        reason='`np.matrix` unsupported in array API mode')
    def test_whiten_not_finite_matrix(self, xp):
        # 测试方法，在 SCIPY_ARRAY_API 模式下跳过，处理 xp 中的非有限值异常
        for bad_value in np.nan, np.inf, -np.inf:
            obs = matrix([[0.98744510, bad_value],
                          [0.62093317, 0.19406729],
                          [0.87545741, 0.00735733],
                          [0.85124403, 0.26499712],
                          [0.45067590, 0.45464607]])
            # 断言调用 whiten 函数时，对非有限值抛出 ValueError 异常
            assert_raises(ValueError, whiten, obs)


class TestVq:

    @skip_xp_backends(cpu_only=True)
    # 定义一个测试方法，用于测试 py_vq 函数在给定的平台上运行是否正确
    def test_py_vq(self, xp):
        # 初始化聚类中心，使用前三个样本点的值
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 调用 py_vq 函数进行向量量化，将结果的第一个元素赋给 label1
        label1 = py_vq(xp.asarray(X), xp.asarray(initc))[0]
        # 使用 xp_assert_equal 断言函数检查 label1 是否等于 LABEL1，忽略数据类型的检查
        xp_assert_equal(label1, xp.asarray(LABEL1, dtype=xp.int64),
                        check_dtype=False)

    # 在满足条件时跳过此测试方法，条件是 SCIPY_ARRAY_API 为真，因为在数组API模式下不支持 np.matrix
    @pytest.mark.skipif(SCIPY_ARRAY_API,
                        reason='`np.matrix` unsupported in array API mode')
    def test_py_vq_matrix(self, xp):
        # 初始化聚类中心，使用前三个样本点的值
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 调用 py_vq 函数进行向量量化，将结果的第一个元素赋给 label1，使用 np.matrix 包装输入
        label1 = py_vq(matrix(X), matrix(initc))[0]
        # 使用 assert_array_equal 断言函数检查 label1 是否等于 LABEL1
        assert_array_equal(label1, LABEL1)

    # 在满足条件时跳过此测试方法，条件是 SCIPY_ARRAY_API 为真，因为在数组API模式下不支持 np.matrix
    @skip_xp_backends(np_only=True, reasons=['`_vq` only supports NumPy backend'])
    def test_vq(self, xp):
        # 初始化聚类中心，使用前三个样本点的值
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 调用 _vq.vq 函数进行向量量化，将结果的第一个元素赋给 label1
        label1, _ = _vq.vq(xp.asarray(X), xp.asarray(initc))
        # 使用 assert_array_equal 断言函数检查 label1 是否等于 LABEL1
        assert_array_equal(label1, LABEL1)
        # 调用 vq 函数，不关心其返回结果

    # 在满足条件时跳过此测试方法，条件是 SCIPY_ARRAY_API 为真，因为在数组API模式下不支持 np.matrix
    @pytest.mark.skipif(SCIPY_ARRAY_API,
                        reason='`np.matrix` unsupported in array API mode')
    def test_vq_matrix(self, xp):
        # 初始化聚类中心，使用前三个样本点的值
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 调用 _vq.vq 函数进行向量量化，将结果的第一个元素赋给 label1，使用 np.matrix 包装输入
        label1, _ = _vq.vq(matrix(X), matrix(initc))
        # 使用 assert_array_equal 断言函数检查 label1 是否等于 LABEL1
        assert_array_equal(label1, LABEL1)
        # 调用 vq 函数，不关心其返回结果

    # 在满足条件时跳过此测试方法，条件是 cpu_only 为真，因为 _vq 只支持 NumPy 后端
    @skip_xp_backends(cpu_only=True)
    def test_vq_1d(self, xp):
        # 测试特殊的一维向量量化算法，使用 Python 实现
        data = X[:, 0]
        # 初始化聚类中心，使用 data 的前三个值
        initc = data[:3]
        # 调用 _vq.vq 函数进行向量量化，将结果分别赋给 a 和 b
        a, b = _vq.vq(data, initc)
        # 将 data 和 initc 转换为 xp 的数组类型
        data = xp.asarray(data)
        initc = xp.asarray(initc)
        # 调用 py_vq 函数进行向量量化，将结果的第一个元素赋给 ta，忽略数据类型的检查
        ta, tb = py_vq(data[:, np.newaxis], initc[:, np.newaxis])
        # 使用 xp_assert_equal 断言函数检查 ta 是否等于 a，忽略数据类型的检查
        xp_assert_equal(ta, xp.asarray(a, dtype=xp.int64), check_dtype=False)
        # 使用 xp_assert_equal 断言函数检查 tb 是否等于 b
        xp_assert_equal(tb, xp.asarray(b))

    # 在满足条件时跳过此测试方法，条件是 _vq 只支持 NumPy 后端
    @skip_xp_backends(np_only=True, reasons=['`_vq` only supports NumPy backend'])
    def test__vq_sametype(self, xp):
        # 创建一个浮点64位的数组 a
        a = xp.asarray([1.0, 2.0], dtype=xp.float64)
        # 将数组 a 转换为浮点32位，并赋值给 b
        b = a.astype(xp.float32)
        # 断言函数，预期会抛出 TypeError 异常，因为 _vq.vq 不支持不同类型的输入
        assert_raises(TypeError, _vq.vq, a, b)

    # 在满足条件时跳过此测试方法，条件是 _vq 只支持 NumPy 后端
    @skip_xp_backends(np_only=True, reasons=['`_vq` only supports NumPy backend'])
    def test__vq_invalid_type(self, xp):
        # 创建一个整数数组 a
        a = xp.asarray([1, 2], dtype=int)
        # 断言函数，预期会抛出 TypeError 异常，因为 _vq.vq 不支持不同类型的输入
        assert_raises(TypeError, _vq.vq, a, a)

    # 在满足条件时跳过此测试方法，条件是 cpu_only 为真，因为 _vq 只支持 NumPy 后端
    @skip_xp_backends(cpu_only=True)
    # 定义一个测试方法，用于验证大特征数量情况下的向量量化
    def test_vq_large_nfeat(self, xp):
        # 生成一个 20x20 的随机数组作为输入数据 X
        X = np.random.rand(20, 20)
        # 生成一个 3x20 的随机数组作为码书 code_book
        code_book = np.random.rand(3, 20)

        # 使用自定义的向量量化函数 _vq.vq 对 X 和 code_book 进行量化，得到 codes0 和 dis0
        codes0, dis0 = _vq.vq(X, code_book)
        # 使用外部函数 py_vq 对 X 和 code_book 进行量化，得到 codes1 和 dis1
        codes1, dis1 = py_vq(
            xp.asarray(X), xp.asarray(code_book)
        )
        # 断言 dis1 和 dis0 在相对误差不超过 1e-5 的情况下相等
        xp_assert_close(dis1, xp.asarray(dis0), rtol=1e-5)
        # 断言 codes1 和 codes0 在数据类型不考虑的情况下相等
        # （因为 codes1 的数据类型在不同平台可能是 int32 或 int64）
        xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)

        # 将 X 和 code_book 转换为 np.float32 类型
        X = X.astype(np.float32)
        code_book = code_book.astype(np.float32)

        # 重新使用向量量化函数 _vq.vq 对 X 和 code_book 进行量化，得到 codes0 和 dis0
        codes0, dis0 = _vq.vq(X, code_book)
        # 重新使用外部函数 py_vq 对 X 和 code_book 进行量化，得到 codes1 和 dis1
        codes1, dis1 = py_vq(
            xp.asarray(X), xp.asarray(code_book)
        )
        # 断言 dis1 和 dis0 在相对误差不超过 1e-5 的情况下相等
        xp_assert_close(dis1, xp.asarray(dis0), rtol=1e-5)
        # 断言 codes1 和 codes0 在数据类型不考虑的情况下相等
        # （因为 codes1 的数据类型在不同平台可能是 int32 或 int64）
        xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)

    # 跳过支持 CPU 的后端，定义一个测试方法，用于验证大特征数情况下的向量量化
    @skip_xp_backends(cpu_only=True)
    def test_vq_large_features(self, xp):
        # 生成一个大小为 10x5 的随机数组 X，并乘以 1000000
        X = np.random.rand(10, 5) * 1000000
        # 生成一个大小为 2x5 的随机数组 code_book，并乘以 1000000
        code_book = np.random.rand(2, 5) * 1000000

        # 使用自定义的向量量化函数 _vq.vq 对 X 和 code_book 进行量化，得到 codes0 和 dis0
        codes0, dis0 = _vq.vq(X, code_book)
        # 使用外部函数 py_vq 对 X 和 code_book 进行量化，得到 codes1 和 dis1
        codes1, dis1 = py_vq(
            xp.asarray(X), xp.asarray(code_book)
        )
        # 断言 dis1 和 dis0 在相对误差不超过 1e-5 的情况下相等
        xp_assert_close(dis1, xp.asarray(dis0), rtol=1e-5)
        # 断言 codes1 和 codes0 在数据类型不考虑的情况下相等
        # （因为 codes1 的数据类型在不同平台可能是 int32 或 int64）
        xp_assert_equal(codes1, xp.asarray(codes0, dtype=xp.int64), check_dtype=False)
# 跳过在 GPU 上的测试，目前只在 CPU 上进行测试
# 一旦 pdist/cdist 和 CuPy 集成，更多的测试将会生效
@skip_xp_backends(cpu_only=True)
# 定义一个测试类 TestKMean
class TestKMean:

    # 测试处理大特征集的情况
    def test_large_features(self, xp):
        # 设置数据维度和样本数
        d = 300
        n = 100

        # 生成两组随机数据集 x 和 y
        m1 = np.random.randn(d)
        m2 = np.random.randn(d)
        x = 10000 * np.random.randn(n, d) - 20000 * m1
        y = 10000 * np.random.randn(n, d) + 20000 * m2

        # 合并数据集 x 和 y
        data = np.empty((x.shape[0] + y.shape[0], d), np.float64)
        data[:x.shape[0]] = x
        data[x.shape[0]:] = y

        # 调用 kmeans 算法处理合并后的数据
        kmeans(xp.asarray(data), 2)

    # 测试简单的 kmeans 算法
    def test_kmeans_simple(self, xp):
        np.random.seed(54321)
        # 初始化聚类中心
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 运行一次迭代的 kmeans 算法，并获取聚类编码
        code1 = kmeans(xp.asarray(X), xp.asarray(initc), iter=1)[0]
        # 检查结果是否接近预期值 CODET2
        xp_assert_close(code1, xp.asarray(CODET2))

    # 如果使用数组 API 模式，跳过测试；原因是不支持 np.matrix
    @pytest.mark.skipif(SCIPY_ARRAY_API,
                        reason='`np.matrix` unsupported in array API mode')
    def test_kmeans_simple_matrix(self, xp):
        np.random.seed(54321)
        # 初始化聚类中心
        initc = np.concatenate([[X[0]], [X[1]], [X[2]]])
        # 运行一次迭代的 kmeans 算法，并获取聚类编码
        code1 = kmeans(matrix(X), matrix(initc), iter=1)[0]
        # 检查结果是否接近预期值 CODET2
        xp_assert_close(code1, CODET2)

    # 测试 kmeans 算法中出现丢失聚类的情况
    def test_kmeans_lost_cluster(self, xp):
        # 导致 kmeans 中一个聚类没有数据点的情况
        data = xp.asarray(TESTDATA_2D)
        initk = xp.asarray([[-1.8127404, -0.67128041],
                            [2.04621601, 0.07401111],
                            [-2.31149087, -0.05160469]])

        # 运行 kmeans 算法
        kmeans(data, initk)
        # 使用 suppress_warnings 捕捉 UserWarning
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "One of the clusters is empty. Re-run kmeans with a "
                       "different initialization")
            # 调用 kmeans2 函数并警告丢失的聚类
            kmeans2(data, initk, missing='warn')

        # 断言会抛出 ClusterError 异常
        assert_raises(ClusterError, kmeans2, data, initk, missing='raise')

    # 测试 kmeans2 算法的简单情况
    def test_kmeans2_simple(self, xp):
        np.random.seed(12345678)
        # 初始化聚类中心
        initc = xp.asarray(np.concatenate([[X[0]], [X[1]], [X[2]]]))
        # 根据不同的数组 API 运行 kmeans2 算法，进行一次和两次迭代，并获取聚类编码
        arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
        for tp in arrays:
            code1 = kmeans2(tp(X), tp(initc), iter=1)[0]
            code2 = kmeans2(tp(X), tp(initc), iter=2)[0]

            # 检查结果是否接近预期值 CODET1 和 CODET2
            xp_assert_close(code1, xp.asarray(CODET1))
            xp_assert_close(code2, xp.asarray(CODET2))

    # 如果使用数组 API 模式，跳过测试；原因是不支持 np.matrix
    @pytest.mark.skipif(SCIPY_ARRAY_API,
                        reason='`np.matrix` unsupported in array API mode')
    def test_kmeans2_simple_matrix(self, xp):
        np.random.seed(12345678)
        # 初始化聚类中心
        initc = xp.asarray(np.concatenate([[X[0]], [X[1]], [X[2]]]))
        # 运行一次和两次迭代的 kmeans2 算法，并获取聚类编码
        code1 = kmeans2(matrix(X), matrix(initc), iter=1)[0]
        code2 = kmeans2(matrix(X), matrix(initc), iter=2)[0]

        # 检查结果是否接近预期值 CODET1 和 CODET2
        xp_assert_close(code1, CODET1)
        xp_assert_close(code2, CODET2)
    def test_kmeans2_rank1(self, xp):
        # 将测试数据转换为对应的数组
        data = xp.asarray(TESTDATA_2D)
        # 从二维数据中选择第一列作为单独的一维数据
        data1 = data[:, 0]

        # 从第一维数据中选择前三个作为初始聚类中心
        initc = data1[:3]
        # 复制初始聚类中心数组，使用特定的数学库（如numpy或jax）
        code = copy(initc, xp=xp)
        # 运行一次迭代的kmeans2算法，并返回聚类结果的第一个值
        kmeans2(data1, code, iter=1)[0]
        # 运行两次迭代的kmeans2算法，并返回聚类结果的第一个值
        kmeans2(data1, code, iter=2)[0]

    def test_kmeans2_rank1_2(self, xp):
        # 将测试数据转换为对应的数组
        data = xp.asarray(TESTDATA_2D)
        # 从二维数据中选择第一列作为单独的一维数据
        data1 = data[:, 0]
        # 运行一次迭代的kmeans2算法，并设定聚类数为2
        kmeans2(data1, 2, iter=1)

    def test_kmeans2_high_dim(self, xp):
        # 测试高维数据的kmeans2算法，当维度数超过输入点数时
        data = xp.asarray(TESTDATA_2D)
        # 将二维数据重新形状为20x20，然后选择前10行作为新的数据
        data = xp.reshape(data, (20, 20))[:10, :]
        # 运行kmeans2算法，设定聚类数为2
        kmeans2(data, 2)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_kmeans2_init(self, xp):
        # 设定随机种子
        np.random.seed(12345)
        # 将测试数据转换为对应的数组
        data = xp.asarray(TESTDATA_2D)
        # 设定聚类数为3

        # 使用points方法进行初始化聚类中心的kmeans2算法
        kmeans2(data, k, minit='points')
        # 对数据的第二列进行聚类
        kmeans2(data[:, 1], k, minit='points')  # 特殊情况（一维）

        # 使用++方法进行初始化聚类中心的kmeans2算法
        kmeans2(data, k, minit='++')
        # 对数据的第二列进行聚类
        kmeans2(data[:, 1], k, minit='++')  # 特殊情况（一维）

        # 使用random方法进行初始化聚类中心的kmeans2算法，可能会产生警告，过滤警告信息
        with suppress_warnings() as sup:
            sup.filter(message="One of the clusters is empty. Re-run.")
            kmeans2(data, k, minit='random')
            # 对数据的第二列进行聚类
            kmeans2(data[:, 1], k, minit='random')  # 特殊情况（一维）

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MemoryError in Wine.')
    def test_krandinit(self, xp):
        # 将测试数据转换为对应的数组
        data = xp.asarray(TESTDATA_2D)
        # 将数据分别重塑为200x2和20x20的形状，并选择前10行作为新的数据
        datas = [xp.reshape(data, (200, 2)),
                 xp.reshape(data, (20, 20))[:10, :]]
        # 设定聚类数为100万
        k = int(1e6)
        for data in datas:
            # 使用特定的随机数生成器对数据进行初始化
            rng = np.random.default_rng(1234)
            # 使用_krandinit函数对数据进行聚类初始化，并返回初始值
            init = _krandinit(data, k, rng, xp)
            # 计算数据的原始协方差
            orig_cov = cov(data.T)
            # 计算初始化后的聚类中心的协方差
            init_cov = cov(init.T)
            # 断言原始协方差与初始化后协方差的接近程度
            xp_assert_close(orig_cov, init_cov, atol=1.1e-2)

    def test_kmeans2_empty(self, xp):
        # 对空数据运行kmeans2算法，预期会引发值错误
        assert_raises(ValueError, kmeans2, xp.asarray([]), 2)

    def test_kmeans_0k(self, xp):
        # 当k参数为0时，运行kmeans和kmeans2算法，预期会引发值错误
        assert_raises(ValueError, kmeans, xp.asarray(X), 0)
        assert_raises(ValueError, kmeans2, xp.asarray(X), 0)
        assert_raises(ValueError, kmeans2, xp.asarray(X), xp.asarray([]))

    def test_kmeans_large_thres(self, xp):
        # 对具有大阈值的数据运行kmeans算法，进行回归测试
        x = xp.asarray([1, 2, 3, 4, 10], dtype=xp.float64)
        # 设定阈值为1e16运行kmeans算法，并返回结果
        res = kmeans(x, 1, thresh=1e16)
        # 断言聚类结果与预期结果的接近程度
        xp_assert_close(res[0], xp.asarray([4.], dtype=xp.float64))
        xp_assert_close(res[1], xp.asarray(2.3999999999999999, dtype=xp.float64)[()])

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_kmeans2_kpp_low_dim(self, xp):
        # 回归测试 gh-11462
        prev_res = xp.asarray([[-1.95266667, 0.898],
                               [-3.153375, 3.3945]], dtype=xp.float64)
        # 设定随机种子
        np.random.seed(42)
        # 运行 kmeans2 算法，使用 k-means++ 初始化方式
        res, _ = kmeans2(xp.asarray(TESTDATA_2D), 2, minit='++')
        # 验证结果是否与预期一致
        xp_assert_close(res, prev_res)

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_kmeans2_kpp_high_dim(self, xp):
        # 回归测试 gh-11462
        n_dim = 100
        size = 10
        # 设置中心点
        centers = np.vstack([5 * np.ones(n_dim),
                             -5 * np.ones(n_dim)])
        np.random.seed(42)
        # 生成高维数据
        data = np.vstack([
            np.random.multivariate_normal(centers[0], np.eye(n_dim), size=size),
            np.random.multivariate_normal(centers[1], np.eye(n_dim), size=size)
        ])

        # 将数据转换为 xp 数组（可能是 numpy 或其他数组）
        data = xp.asarray(data)
        # 运行 kmeans2 算法，使用 k-means++ 初始化方式
        res, _ = kmeans2(data, 2, minit='++')
        # 验证结果与中心点的符号是否一致
        xp_assert_equal(xp.sign(res), xp.sign(xp.asarray(centers)))

    def test_kmeans_diff_convergence(self, xp):
        # 回归测试 gh-8727
        obs = xp.asarray([-3, -1, 0, 1, 1, 8], dtype=xp.float64)
        # 运行 kmeans 算法
        res = kmeans(obs, xp.asarray([-3., 0.99]))
        # 验证聚类结果的接近程度
        xp_assert_close(res[0], xp.asarray([-0.4,  8.], dtype=xp.float64))
        # 验证聚类结果的接近程度
        xp_assert_close(res[1], xp.asarray(1.0666666666666667, dtype=xp.float64)[()])

    @skip_xp_backends('jax.numpy',
                      reasons=['jax arrays do not support item assignment'],
                      cpu_only=True)
    def test_kmeans_and_kmeans2_random_seed(self, xp):
        # 测试不同的随机种子
        seed_list = [
            1234, np.random.RandomState(1234), np.random.default_rng(1234)
        ]

        for seed in seed_list:
            seed1 = deepcopy(seed)
            seed2 = deepcopy(seed)
            data = xp.asarray(TESTDATA_2D)
            # 测试 kmeans 算法
            res1, _ = kmeans(data, 2, seed=seed1)
            res2, _ = kmeans(data, 2, seed=seed2)
            # 验证两次运行结果是否一致
            xp_assert_close(res1, res2)  # 应该得到相同的结果
            # 测试 kmeans2 算法的不同初始化方式
            for minit in ["random", "points", "++"]:
                res1, _ = kmeans2(data, 2, minit=minit, seed=seed1)
                res2, _ = kmeans2(data, 2, minit=minit, seed=seed2)
                # 验证两次运行结果是否一致
                xp_assert_close(res1, res2)  # 应该得到相同的结果
```