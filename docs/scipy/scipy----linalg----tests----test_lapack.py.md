# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_lapack.py`

```
# Created by: Pearu Peterson, September 2002
#

# 从 functools 库中导入 reduce 函数
from functools import reduce

# 从 numpy.testing 模块中导入多个断言函数
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
                           assert_allclose, assert_almost_equal,
                           assert_array_equal)

# 导入 pytest 测试框架，并将 raises 函数重命名为 assert_raises
import pytest
from pytest import raises as assert_raises

# 导入 numpy 库并将其命名为 np，同时从 numpy 中导入多个函数和对象
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
                   triu_indices)

# 从 numpy.random 模块导入 rand, randint, seed 函数
from numpy.random import rand, randint, seed

# 从 scipy.linalg 模块导入多个线性代数相关的函数
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
                          solve, ldl, norm, block_diag, qr, eigh, qz)

# 从 scipy.linalg.lapack 模块导入 _compute_lwork 函数
from scipy.linalg.lapack import _compute_lwork

# 从 scipy.stats 模块导入 ortho_group 和 unitary_group 函数
from scipy.stats import ortho_group, unitary_group

# 导入 scipy.sparse 并将其命名为 sps
import scipy.sparse as sps

# 尝试导入 scipy.__config__ 中的 CONFIG 对象，如果失败则将 CONFIG 设置为 None
try:
    from scipy.__config__ import CONFIG
except ImportError:
    CONFIG = None

# 尝试导入 scipy.linalg 中的 _clapack 模块，如果失败则将 clapack 设置为 None
try:
    from scipy.linalg import _clapack as clapack
except ImportError:
    clapack = None

# 从 scipy.linalg.lapack 模块导入 get_lapack_funcs 函数
from scipy.linalg.lapack import get_lapack_funcs

# 从 scipy.linalg.blas 模块导入 get_blas_funcs 函数
from scipy.linalg.blas import get_blas_funcs

# 定义一个包含所有实数数据类型和复数数据类型的列表
REAL_DTYPES = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

# 初始化 blas_provider 和 blas_version 为 None
blas_provider = blas_version = None

# 如果 CONFIG 不为 None，则从 CONFIG 中获取 blas 的名称和版本信息
if CONFIG is not None:
    blas_provider = CONFIG['Build Dependencies']['blas']['name']
    blas_version = CONFIG['Build Dependencies']['blas']['version']


def generate_random_dtype_array(shape, dtype):
    # 生成指定数据类型和形状的随机矩阵
    if dtype in COMPLEX_DTYPES:
        return (np.random.rand(*shape)
                + np.random.rand(*shape)*1.0j).astype(dtype)
    return np.random.rand(*shape).astype(dtype)


def test_lapack_documented():
    """Test that all entries are in the doc."""
    # 如果 lapack 模块的 __doc__ 属性为 None，则跳过测试
    if lapack.__doc__ is None:  # just in case there is a python -OO
        pytest.skip('lapack.__doc__ is None')
    
    # 将 lapack 模块的文档字符串拆分为单词集合
    names = set(lapack.__doc__.split())
    
    # 定义一个忽略列表，包含不需要检查的单词
    ignore_list = {
        "absolute_import",
        "clapack",
        "division",
        "find_best_lapack_type",
        "flapack",
        "print_function",
        "HAS_ILP64",
        "np",
    }
    
    # 初始化一个列表，用于存储未在文档字符串中找到的单词
    missing = list()
    
    # 遍历 lapack 模块中的所有名称
    for name in dir(lapack):
        # 检查名称不以 '_' 开头，并且不在忽略列表中，且不在文档字符串单词集合中
        if (not name.startswith('_') and name not in ignore_list and
                name not in names):
            missing.append(name)
    
    # 使用断言检查未找到的单词列表是否为空
    assert missing == [], 'Name(s) missing from lapack.__doc__ or ignore_list'


class TestFlapackSimple:
    # 定义一个测试函数 test_gebal
    def test_gebal(self):
        # 初始化一个二维列表 a
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # 初始化另一个二维列表 a1，包含浮点数和整数
        a1 = [[1, 0, 0, 3e-4],
              [4, 0, 0, 2e-3],
              [7, 1, 0, 0],
              [0, 1, 0, 0]]
        # 遍历字符串 'sdzc'
        for p in 'sdzc':
            # 获取名为 p+'gebal' 的函数引用，若不存在则跳过
            f = getattr(flapack, p+'gebal', None)
            if f is None:
                continue
            # 调用 f 函数，返回 ba, lo, hi, pivscale, info 五个变量
            ba, lo, hi, pivscale, info = f(a)
            # 断言 info 为假，否则输出 info 的字符串表示
            assert_(not info, repr(info))
            # 断言 ba 与 a 几乎相等
            assert_array_almost_equal(ba, a)
            # 断言 (lo, hi) 等于 (0, len(a[0])-1)
            assert_equal((lo, hi), (0, len(a[0])-1))
            # 断言 pivscale 为长度为 len(a) 的全1数组
            assert_array_almost_equal(pivscale, np.ones(len(a)))

            # 再次调用 f 函数，传入 a1, permute=1, scale=1 三个参数
            ba, lo, hi, pivscale, info = f(a1, permute=1, scale=1)
            # 断言 info 为假，否则输出 info 的字符串表示
            assert_(not info, repr(info))
            # 输出 a1 的内容（已注释）
            # 输出 ba, lo, hi, pivscale 的内容（已注释）

    # 定义一个测试函数 test_gehrd
    def test_gehrd(self):
        # 初始化一个二维列表 a，包含整数
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        # 遍历字符 'd'
        for p in 'd':
            # 获取名为 p+'gehrd' 的函数引用，若不存在则跳过
            f = getattr(flapack, p+'gehrd', None)
            if f is None:
                continue
            # 调用 f 函数，返回 ht, tau, info 三个变量
            ht, tau, info = f(a)
            # 断言 info 为假，否则输出 info 的字符串表示
            assert_(not info, repr(info))

    # 定义一个测试函数 test_trsyl
    def test_trsyl(self):
        # 初始化三个 numpy 数组 a, b, c
        a = np.array([[1, 2], [0, 4]])
        b = np.array([[5, 6], [0, 8]])
        c = np.array([[9, 10], [11, 12]])
        # 初始化字符串 trans 为 'T'
        trans = 'T'

        # 遍历字符串 'fdFD'
        # 测试单精度、双精度、单复数、双复数实现，包括大部分选项
        for dtype in 'fdFD':
            # 将 a, b, c 转换为 dtype 类型的数组 a1, b1, c1
            a1, b1, c1 = a.astype(dtype), b.astype(dtype), c.astype(dtype)
            # 获取 trsyl 函数的引用
            trsyl, = get_lapack_funcs(('trsyl',), (a1,))
            # 若 dtype 是大写字母，则为复数 dtype
            if dtype.isupper():
                a1[0] += 1j
                trans = 'C'

            # 调用 trsyl 函数，返回 x, scale, info 三个变量
            x, scale, info = trsyl(a1, b1, c1)
            # 断言 np.dot(a1, x) + np.dot(x, b1) 几乎等于 scale * c1
            assert_array_almost_equal(np.dot(a1, x) + np.dot(x, b1),
                                      scale * c1)

            # 再次调用 trsyl 函数，传入 a1, b1, c1, trana=trans, tranb=trans 四个参数
            x, scale, info = trsyl(a1, b1, c1, trana=trans, tranb=trans)
            # 断言 np.dot(a1.conjugate().T, x) + np.dot(x, b1.conjugate().T) 几乎等于 scale * c1
            assert_array_almost_equal(
                    np.dot(a1.conjugate().T, x) + np.dot(x, b1.conjugate().T),
                    scale * c1, decimal=4)

            # 再次调用 trsyl 函数，传入 a1, b1, c1, isgn=-1 三个参数
            x, scale, info = trsyl(a1, b1, c1, isgn=-1)
            # 断言 np.dot(a1, x) - np.dot(x, b1) 几乎等于 scale * c1
            assert_array_almost_equal(np.dot(a1, x) - np.dot(x, b1),
                                      scale * c1, decimal=4)
    # 定义一个测试函数，用于测试不同的数据类型和范数计算
    def test_lange(self):
        # 创建一个3x3的NumPy数组，包含整数值
        a = np.array([
            [-149, -50, -154],
            [537, 180, 546],
            [-27, -9, -25]])

        # 循环遍历数据类型和范数字符串
        for dtype in 'fdFD':
            for norm_str in 'Mm1OoIiFfEe':
                # 将数组a转换为指定的数据类型
                a1 = a.astype(dtype)

                # 如果数据类型是复数类型
                if dtype.isupper():
                    # 给数组中的第一个元素添加虚部1j（复数单位）
                    a1[0, 0] += 1j

                # 获取用于计算范数的LAPACK函数
                lange, = get_lapack_funcs(('lange',), (a1,))
                
                # 调用LAPACK函数计算指定范数的值
                value = lange(norm_str, a1)

                # 根据范数类型进行不同的断言
                if norm_str in 'FfEe':
                    # 如果范数是Frobenius范数或2-范数
                    if dtype in 'Ff':
                        # 对于单精度或双精度浮点数，设定小数点精度为3
                        decimal = 3
                    else:
                        # 对于双精度或复数，设定小数点精度为7
                        decimal = 7

                    # 计算参考值：对数组元素取绝对值后平方，再开平方根
                    ref = np.sqrt(np.sum(np.square(np.abs(a1))))
                    # 断言计算得到的值与参考值的接近程度
                    assert_almost_equal(value, ref, decimal)
                else:
                    # 如果范数是其他类型（最大范数、1-范数、无穷范数）
                    if norm_str in 'Mm':
                        # 最大范数：数组元素的最大绝对值
                        ref = np.max(np.abs(a1))
                    elif norm_str in '1Oo':
                        # 1-范数和无穷范数：按行或按列求和后取最大值
                        ref = np.max(np.sum(np.abs(a1), axis=0 if norm_str in '1Ii' else 1))

                    # 断言计算得到的值与参考值相等
                    assert_equal(value, ref)
class TestLapack:

    def test_flapack(self):
        # 检查是否存在名为 'empty_module' 的属性，用以判断 flapack 模块是否为空
        if hasattr(flapack, 'empty_module'):
            # flapack 模块为空时不执行任何操作
            pass

    def test_clapack(self):
        # 检查是否存在名为 'empty_module' 的属性，用以判断 clapack 模块是否为空
        if hasattr(clapack, 'empty_module'):
            # clapack 模块为空时不执行任何操作
            pass


class TestLeastSquaresSolvers:

    def test_gels(self):
        seed(1234)
        # 测试矩阵的处理能力，包括 gh-issue #8329 中提到的对于“矮胖”矩阵和“瘦高”矩阵的处理
        for ind, dtype in enumerate(DTYPES):
            m = 10
            n = 20
            nrhs = 1
            a1 = rand(m, n).astype(dtype)
            b1 = rand(n).astype(dtype)
            # 获取 LAPACK 函数 'gels' 和 'gels_lwork'
            gls, glslw = get_lapack_funcs(('gels', 'gels_lwork'), dtype=dtype)

            # 计算所需的工作空间大小
            lwork = _compute_lwork(glslw, m, n, nrhs)
            # 调用 LAPACK 的 gels 函数进行求解
            _, _, info = gls(a1, b1, lwork=lwork)
            # 确保 LAPACK 返回的信息码大于等于 0
            assert_(info >= 0)
            # 使用不同的转置选项再次调用 gels 函数
            _, _, info = gls(a1, b1, trans='TTCC'[ind], lwork=lwork)
            # 确保 LAPACK 返回的信息码大于等于 0
            assert_(info >= 0)

        # 针对实数类型的数据类型进行进一步的测试
        for dtype in REAL_DTYPES:
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            # 获取 LAPACK 函数 'gels'、'gels_lwork' 和 'geqrf'
            gels, gels_lwork, geqrf = get_lapack_funcs(
                    ('gels', 'gels_lwork', 'geqrf'), (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 计算所需的工作空间大小
            lwork = _compute_lwork(gels_lwork, m, n, nrhs)

            # 调用 LAPACK 的 gels 函数进行求解
            lqr, x, info = gels(a1, b1, lwork=lwork)
            # 断言检查计算结果 x 的精确性
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            # 调用 LAPACK 的 geqrf 函数获取 QR 分解结果
            lqr_truth, _, _, _ = geqrf(a1)
            # 断言检查计算结果 lqr 的精确性
            assert_array_equal(lqr, lqr_truth)

        # 针对复数类型的数据类型进行进一步的测试
        for dtype in COMPLEX_DTYPES:
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            # 获取 LAPACK 函数 'gels'、'gels_lwork' 和 'geqrf'
            gels, gels_lwork, geqrf = get_lapack_funcs(
                    ('gels', 'gels_lwork', 'geqrf'), (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 计算所需的工作空间大小
            lwork = _compute_lwork(gels_lwork, m, n, nrhs)

            # 调用 LAPACK 的 gels 函数进行求解
            lqr, x, info = gels(a1, b1, lwork=lwork)
            # 断言检查计算结果 x 的精确性
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)
            # 调用 LAPACK 的 geqrf 函数获取 QR 分解结果
            lqr_truth, _, _, _ = geqrf(a1)
            # 断言检查计算结果 lqr 的精确性
            assert_array_equal(lqr, lqr_truth)
    # 定义一个测试方法 test_gelsd，用于测试 gelsd 函数的功能
    def test_gelsd(self):
        # 遍历所有的实数数据类型
        for dtype in REAL_DTYPES:
            # 创建实数类型的矩阵 a1
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            # 创建实数类型的向量 b1
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            # 获取 LAPACK 函数 gelsd 和 gelsd_lwork
            gelsd, gelsd_lwork = get_lapack_funcs(('gelsd', 'gelsd_lwork'),
                                                  (a1, b1))

            # 获取矩阵 a1 的行数 m 和列数 n
            m, n = a1.shape
            # 确定右侧向量 b1 的列数 nrhs
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求工作区大小
            work, iwork, info = gelsd_lwork(m, n, nrhs, -1)
            # 将工作区大小转换为整数 lwork
            lwork = int(np.real(work))
            # 记录 iwork 的大小
            iwork_size = iwork

            # 调用 gelsd 函数求解线性最小二乘问题
            x, s, rank, info = gelsd(a1, b1, lwork, iwork_size,
                                     -1, False, False)
            # 使用 assert_allclose 断言近似相等性
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            assert_allclose(s, np.array([12.596017180511966,
                                         0.583396253199685], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        # 遍历所有的复数数据类型
        for dtype in COMPLEX_DTYPES:
            # 创建复数类型的矩阵 a1
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            # 创建复数类型的向量 b1
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            # 获取 LAPACK 函数 gelsd 和 gelsd_lwork
            gelsd, gelsd_lwork = get_lapack_funcs(('gelsd', 'gelsd_lwork'),
                                                  (a1, b1))

            # 获取矩阵 a1 的行数 m 和列数 n
            m, n = a1.shape
            # 确定右侧向量 b1 的列数 nrhs
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求工作区大小
            work, rwork, iwork, info = gelsd_lwork(m, n, nrhs, -1)
            # 将工作区大小转换为整数 lwork 和 rwork 的大小为整数
            lwork = int(np.real(work))
            rwork_size = int(rwork)
            iwork_size = iwork

            # 调用 gelsd 函数求解线性最小二乘问题
            x, s, rank, info = gelsd(a1, b1, lwork, rwork_size, iwork_size,
                                     -1, False, False)
            # 使用 assert_allclose 断言近似相等性
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)
            assert_allclose(s,
                            np.array([13.035514762572043, 4.337666985231382],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)
    # 定义一个测试函数 test_gelss，用于测试 gelss 函数的行为
    def test_gelss(self):

        # 对于实数数据类型，循环测试
        for dtype in REAL_DTYPES:
            # 创建一个实数类型的 numpy 数组 a1，包含指定的数据
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            # 创建一个实数类型的 numpy 数组 b1，包含指定的数据
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            # 获取 LAPACK 函数 gelss 和 gelss_lwork
            gelss, gelss_lwork = get_lapack_funcs(('gelss', 'gelss_lwork'),
                                                  (a1, b1))

            # 获取数组 a1 的形状 m 和 n
            m, n = a1.shape
            # 确定 nrhs（右手边的列数），如果 b1 是二维数组则为列数，否则为 1
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求计算所需的工作空间大小
            work, info = gelss_lwork(m, n, nrhs, -1)
            # 将工作空间大小转换为整数
            lwork = int(np.real(work))

            # 调用 gelss 函数执行最小二乘解法，计算出 v, x, s, rank, work, info
            v, x, s, rank, work, info = gelss(a1, b1, -1, lwork, False, False)
            # 断言 x 的前 n-1 个元素与预期值相近
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            # 断言 s 与预期值相近
            assert_allclose(s, np.array([12.596017180511966,
                                         0.583396253199685], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        # 对于复数数据类型，循环测试
        for dtype in COMPLEX_DTYPES:
            # 创建一个复数类型的 numpy 数组 a1，包含指定的数据
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            # 创建一个复数类型的 numpy 数组 b1，包含指定的数据
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            # 获取 LAPACK 函数 gelss 和 gelss_lwork
            gelss, gelss_lwork = get_lapack_funcs(('gelss', 'gelss_lwork'),
                                                  (a1, b1))

            # 获取数组 a1 的形状 m 和 n
            m, n = a1.shape
            # 确定 nrhs（右手边的列数），如果 b1 是二维数组则为列数，否则为 1
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求计算所需的工作空间大小
            work, info = gelss_lwork(m, n, nrhs, -1)
            # 将工作空间大小转换为整数
            lwork = int(np.real(work))

            # 调用 gelss 函数执行最小二乘解法，计算出 v, x, s, rank, work, info
            v, x, s, rank, work, info = gelss(a1, b1, -1, lwork, False, False)
            # 断言 x 的前 n-1 个元素与预期值相近
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            # 断言 s 与预期值相近
            assert_allclose(s, np.array([13.035514762572043,
                                         4.337666985231382], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
    # 定义一个测试函数 test_gelsy，用于测试不同数据类型的线性方程组求解
    def test_gelsy(self):

        # 遍历所有的实数数据类型
        for dtype in REAL_DTYPES:
            # 创建一个实数类型的二维数组 a1，表示系数矩阵
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            # 创建一个实数类型的一维数组 b1，表示常数向量
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            # 获取 LAPACK 函数 gelsy 和 gelss_lwork
            gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'),
                                                  (a1, b1))

            # 获取矩阵 a1 的行数 m 和列数 n
            m, n = a1.shape
            # 判断 b1 是否为二维数组，确定 nrhs 的值
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求计算工作空间和信息
            work, info = gelsy_lwork(m, n, nrhs, 10*np.finfo(dtype).eps)
            # 将工作空间的实部取整得到 lwork
            lwork = int(np.real(work))

            # 初始化 jptv，一个形状为 (a1.shape[1], 1) 的零矩阵，数据类型为 np.int32
            jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
            # 调用 gelsy 函数求解线性方程组，并获取返回值 v, x, j, rank, info
            v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps,
                                        lwork, False, False)
            # 使用 assert_allclose 断言函数检查 x[:-1] 是否接近于预期值
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        # 遍历所有的复数数据类型
        for dtype in COMPLEX_DTYPES:
            # 创建一个复数类型的二维数组 a1，表示系数矩阵
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            # 创建一个复数类型的一维数组 b1，表示常数向量
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            # 获取 LAPACK 函数 gelsy 和 gelss_lwork
            gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'),
                                                  (a1, b1))

            # 获取矩阵 a1 的行数 m 和列数 n
            m, n = a1.shape
            # 判断 b1 是否为二维数组，确定 nrhs 的值
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # 请求计算工作空间和信息
            work, info = gelsy_lwork(m, n, nrhs, 10*np.finfo(dtype).eps)
            # 将工作空间的实部取整得到 lwork
            lwork = int(np.real(work))

            # 初始化 jptv，一个形状为 (a1.shape[1], 1) 的零矩阵，数据类型为 np.int32
            jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
            # 调用 gelsy 函数求解线性方程组，并获取返回值 v, x, j, rank, info
            v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps,
                                        lwork, False, False)
            # 使用 assert_allclose 断言函数检查 x[:-1] 是否接近于预期值
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3, 4), (5, 2), (2**18, 2**18)])
def test_geqrf_lwork(dtype, shape):
    # 使用 pytest 的参数化装饰器，对 dtype 和 shape 进行参数化测试
    geqrf_lwork = get_lapack_funcs(('geqrf_lwork'), dtype=dtype)
    m, n = shape
    # 调用 LAPACK 函数 geqrf_lwork，获取计算所需工作空间大小 lwork 和返回信息 info
    lwork, info = geqrf_lwork(m=m, n=n)
    # 断言 LAPACK 返回信息 info 为 0，表示函数调用成功
    assert_equal(info, 0)


class TestRegression:

    def test_ticket_1645(self):
        # 检查 RQ 分解函数的 lwork 参数是否正确
        for dtype in DTYPES:
            a = np.zeros((300, 2), dtype=dtype)

            # 获取 LAPACK 函数 gerqf
            gerqf, = get_lapack_funcs(['gerqf'], [a])
            # 断言 gerqf 函数在 lwork=2 的情况下抛出异常
            assert_raises(Exception, gerqf, a, lwork=2)
            # 执行 gerqf 函数，获取 RQ 分解结果
            rq, tau, work, info = gerqf(a)

            if dtype in REAL_DTYPES:
                # 对于实数类型，获取 orgrq 函数
                orgrq, = get_lapack_funcs(['orgrq'], [a])
                # 断言 orgrq 函数在 lwork=1 的情况下抛出异常
                assert_raises(Exception, orgrq, rq[-2:], tau, lwork=1)
                # 执行 orgrq 函数，重构 RQ 分解结果
                orgrq(rq[-2:], tau, lwork=2)
            elif dtype in COMPLEX_DTYPES:
                # 对于复数类型，获取 ungrq 函数
                ungrq, = get_lapack_funcs(['ungrq'], [a])
                # 断言 ungrq 函数在 lwork=1 的情况下抛出异常
                assert_raises(Exception, ungrq, rq[-2:], tau, lwork=1)
                # 执行 ungrq 函数，重构 RQ 分解结果
                ungrq(rq[-2:], tau, lwork=2)


class TestDpotr:
    def test_gh_2691(self):
        # 测试 dportf/dpotri 函数的 'lower' 参数
        for lower in [True, False]:
            for clean in [True, False]:
                np.random.seed(42)
                x = np.random.normal(size=(3, 3))
                a = x.dot(x.T)

                # 获取 LAPACK 函数 dpotrf 和 dpotri
                dpotrf, dpotri = get_lapack_funcs(("potrf", "potri"), (a, ))

                # 调用 dpotrf 函数进行 Cholesky 分解
                c, info = dpotrf(a, lower, clean=clean)
                # 调用 dpotri 函数进行 Cholesky 逆运算
                dpt = dpotri(c, lower)[0]

                if lower:
                    # 对于 lower=True，比较 dpt 的下三角部分与逆矩阵的下三角部分
                    assert_allclose(np.tril(dpt), np.tril(inv(a)))
                else:
                    # 对于 lower=False，比较 dpt 的上三角部分与逆矩阵的上三角部分
                    assert_allclose(np.triu(dpt), np.triu(inv(a)))


class TestDlasd4:
    def test_sing_val_update(self):
        # 检查 singular value 更新的 LAPACK 函数 dlasd4
        sigmas = np.array([4., 3., 2., 0])
        m_vec = np.array([3.12, 5.7, -4.8, -2.2])

        M = np.hstack((np.vstack((np.diag(sigmas[0:-1]),
                                  np.zeros((1, len(m_vec) - 1)))),
                       m_vec[:, np.newaxis]))
        SM = svd(M, full_matrices=False, compute_uv=False, overwrite_a=False,
                 check_finite=False)

        it_len = len(sigmas)
        sgm = np.concatenate((sigmas[::-1], [sigmas[0] + it_len*norm(m_vec)]))
        mvc = np.concatenate((m_vec[::-1], (0,)))

        # 获取 LAPACK 函数 lasd4
        lasd4 = get_lapack_funcs('lasd4', (sigmas,))

        roots = []
        for i in range(0, it_len):
            # 调用 lasd4 函数，获取 singular value 更新后的结果
            res = lasd4(i, sgm, mvc)
            roots.append(res[1])

            # 断言 LAPACK root 查找函数 dlasd4 能够找到非正根
            assert_((res[3] <= 0), "LAPACK root finding dlasd4 failed to find \
                                    the singular value %i" % i)
        roots = np.array(roots)[::-1]

        # 断言 roots 中没有 NaN 值
        assert_((not np.any(np.isnan(roots))), "There are NaN roots")
        # 比较 SM 和 roots 的值，使用绝对容差和相对容差
        assert_allclose(SM, roots, atol=100*np.finfo(np.float64).eps,
                        rtol=100*np.finfo(np.float64).eps)


class TestTbtrs:

    @pytest.mark.parametrize('dtype', DTYPES)
    # 定义测试函数，用于测试 NAG 提供的实例 f07vef 和复数示例 f07vsf
    def test_nag_example_f07vef_f07vsf(self, dtype):
        """Test real (f07vef) and complex (f07vsf) examples from NAG
        
        Examples available from:
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vef.html
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vsf.html
        
        """
        # 根据 dtype 类型选择不同的示例数据
        if dtype in REAL_DTYPES:
            # 实数类型的示例数据 ab, b, x_out
            ab = np.array([[-4.16, 4.78, 6.32, 0.16],
                           [-2.25, 5.86, -4.82, 0]],
                          dtype=dtype)
            b = np.array([[-16.64, -4.16],
                          [-13.78, -16.59],
                          [13.10, -4.94],
                          [-14.14, -9.96]],
                         dtype=dtype)
            x_out = np.array([[4, 1],
                              [-1, -3],
                              [3, 2],
                              [2, -2]],
                             dtype=dtype)
        elif dtype in COMPLEX_DTYPES:
            # 复数类型的示例数据 ab, b, x_out
            ab = np.array([[-1.94+4.43j, 4.12-4.27j, 0.43-2.66j, 0.44+0.1j],
                           [-3.39+3.44j, -1.84+5.52j, 1.74 - 0.04j, 0],
                           [1.62+3.68j, -2.77-1.93j, 0, 0]],
                          dtype=dtype)
            b = np.array([[-8.86 - 3.88j, -24.09 - 5.27j],
                          [-15.57 - 23.41j, -57.97 + 8.14j],
                          [-7.63 + 22.78j, 19.09 - 29.51j],
                          [-14.74 - 2.40j, 19.17 + 21.33j]],
                         dtype=dtype)
            x_out = np.array([[2j, 1 + 5j],
                              [1 - 3j, -7 - 2j],
                              [-4.001887 - 4.988417j, 3.026830 + 4.003182j],
                              [1.996158 - 1.045105j, -6.103357 - 8.986653j]],
                             dtype=dtype)
        else:
            # 抛出异常，当 dtype 既不是实数也不是复数类型时
            raise ValueError(f"Datatype {dtype} not understood.")

        # 获取 LAPACK 函数 tbtrs 并传入参数 ab, b, uplo='L'
        tbtrs = get_lapack_funcs(('tbtrs'), dtype=dtype)
        # 调用 tbtrs 函数计算结果 x 和信息 info
        x, info = tbtrs(ab=ab, b=b, uplo='L')
        # 断言 info 的值为 0，表示调用成功
        assert_equal(info, 0)
        # 断言计算得到的 x 与预期的 x_out 在给定的相对容差和绝对容差范围内相等
        assert_allclose(x, x_out, rtol=0, atol=1e-5)
    def test_random_matrices(self, dtype, trans, uplo, diag):
        # 设置随机数种子为1724，确保结果可重复
        seed(1724)
        
        # 定义矩阵和向量的尺寸及带宽
        n, nrhs, kd = 4, 3, 2
        
        # 获取 LAPACK 函数 tbtrs，根据数据类型选择对应函数
        tbtrs = get_lapack_funcs('tbtrs', dtype=dtype)

        # 根据是否上三角来确定带宽
        is_upper = (uplo == 'U')
        ku = kd * is_upper
        kl = kd - ku

        # 构造矩阵 A 的对角线和 kd 个超/次对角线
        band_offsets = range(ku, -kl - 1, -1)
        band_widths = [n - abs(x) for x in band_offsets]
        bands = [generate_random_dtype_array((width,), dtype)
                 for width in band_widths]

        # 如果要求 A 是单位上三角矩阵，则将对角线设置为全1
        if diag == 'U':
            bands[ku] = np.ones(n, dtype=dtype)

        # 使用 bands 和 band_offsets 构造带状矩阵 A
        a = sps.diags(bands, band_offsets, format='dia')

        # 将 A 转换为带状存储形式
        ab = np.zeros((kd + 1, n), dtype)
        for row, k in enumerate(band_offsets):
            ab[row, max(k, 0):min(n+k, n)] = a.diagonal(k)

        # 生成右侧向量 b
        b = generate_random_dtype_array((n, nrhs), dtype)

        # 调用 LAPACK 函数 tbtrs 求解线性方程组 ab * x = b
        x, info = tbtrs(ab=ab, b=b, uplo=uplo, trans=trans, diag=diag)
        
        # 检查解的有效性，info 应为 0
        assert_equal(info, 0)

        # 根据 trans 参数检查解的正确性
        if trans == 'N':
            assert_allclose(a @ x, b, rtol=5e-5)
        elif trans == 'T':
            assert_allclose(a.T @ x, b, rtol=5e-5)
        elif trans == 'C':
            assert_allclose(a.T.conjugate() @ x, b, rtol=5e-5)
        else:
            raise ValueError('Invalid trans argument')

    @pytest.mark.parametrize('uplo,trans,diag',
                             [['U', 'N', 'Invalid'],
                              ['U', 'Invalid', 'N'],
                              ['Invalid', 'N', 'N']])
    def test_invalid_argument_raises_exception(self, uplo, trans, diag):
        """测试当提供无效的 uplo、trans 和 diag 值时是否引发异常"""
        # 参数检查与使用的数据类型无关
        tbtrs = get_lapack_funcs('tbtrs', dtype=np.float64)
        
        # 随机生成 ab 和 b 矩阵
        ab = rand(4, 2)
        b = rand(2, 4)
        
        # 断言调用 tbtrs 函数时是否抛出异常
        assert_raises(Exception, tbtrs, ab, b, uplo, trans, diag)

    def test_zero_element_in_diagonal(self):
        """测试当矩阵对角线元素为零时，矩阵是否奇异

        如果矩阵 A 的第 i 个对角线元素为零，?tbtrs 应该在 info 中返回 `i`，
        表示提供的矩阵是奇异的。

        注意，?tbtrs 要求矩阵 A 以带状形式存储，此形式中对角线对应于最后一行。
        """
        # 创建全1矩阵 ab 和向量 b
        ab = np.ones((3, 4), dtype=float)
        b = np.ones(4, dtype=float)
        
        # 获取 tbtrs 函数
        tbtrs = get_lapack_funcs('tbtrs', dtype=float)

        # 将 ab 的最后一行最后一个元素置为 0
        ab[-1, 3] = 0
        
        # 调用 tbtrs 函数求解线性方程组 ab * x = b
        _, info = tbtrs(ab=ab, b=b, uplo='U')
        
        # 检查返回的 info 值是否为 4，表示矩阵奇异
        assert_equal(info, 4)
    # 使用 pytest 模块的 parametrize 装饰器为测试函数 test_invalid_matrix_shapes 添加多组参数化测试数据
    @pytest.mark.parametrize('ldab,n,ldb,nrhs', [
                              (5, 5, 0, 5),    # 参数组1：ldab=5, n=5, ldb=0, nrhs=5
                              (5, 5, 3, 5)     # 参数组2：ldab=5, n=5, ldb=3, nrhs=5
    ])
    # 定义测试函数，测试当矩阵形状无效时是否正确抛出异常
    def test_invalid_matrix_shapes(self, ldab, n, ldb, nrhs):
        """Test ?tbtrs fails correctly if shapes are invalid."""
        # 创建大小为 (ldab, n) 的全一矩阵 ab 和大小为 (ldb, nrhs) 的全一矩阵 b，数据类型为 float
        ab = np.ones((ldab, n), dtype=float)
        b = np.ones((ldb, nrhs), dtype=float)
        # 使用 NumPy 的 get_lapack_funcs 函数获取 tbtrs LAPACK 函数的实现，数据类型为 float
        tbtrs = get_lapack_funcs('tbtrs', dtype=float)
        # 断言测试过程中会抛出异常，即调用 tbtrs(ab, b) 时应该抛出 Exception
        assert_raises(Exception, tbtrs, ab, b)
def test_lartg():
    # 循环遍历数据类型 'fdFD'
    for dtype in 'fdFD':
        # 获取 'lartg' 函数对应的 LAPACK 或 BLAS 实现
        lartg = get_lapack_funcs('lartg', dtype=dtype)

        # 创建数组 f 和 g，其中 f = 3，g = 4，数据类型由 dtype 决定
        f = np.array(3, dtype)
        g = np.array(4, dtype)

        # 如果 g 是复数对象，则将其乘以虚数单位 1j
        if np.iscomplexobj(g):
            g *= 1j

        # 调用 lartg 函数，计算 cs、sn 和 r
        cs, sn, r = lartg(f, g)

        # 断言 cs 的值接近于 3.0/5.0
        assert_allclose(cs, 3.0/5.0)
        # 断言 r 的值接近于 5.0
        assert_allclose(r, 5.0)

        # 如果 g 是复数对象
        if np.iscomplexobj(g):
            # 断言 sn 的值接近于 -4.0j/5.0
            assert_allclose(sn, -4.0j/5.0)
            # 断言 r 是复数类型
            assert_(isinstance(r, complex))
            # 断言 cs 是浮点数类型
            assert_(isinstance(cs, float))
        else:
            # 断言 sn 的值接近于 4.0/5.0
            assert_allclose(sn, 4.0/5.0)


def test_rot():
    # srot, drot from blas and crot and zrot from lapack.

    # 循环遍历数据类型 'fdFD'
    for dtype in 'fdFD':
        # 初始化变量 c 和 s
        c = 0.6
        s = 0.8

        # 创建数组 u 和 v，元素均为 3 和 4，数据类型由 dtype 决定
        u = np.full(4, 3, dtype)
        v = np.full(4, 4, dtype)
        
        # 计算公差 atol，精度由数据类型 dtype 的精度决定
        atol = 10**-(np.finfo(dtype).precision-1)

        # 根据数据类型选择 BLAS 或 LAPACK 中的 'rot' 函数，并设置 f 的值
        if dtype in 'fd':
            rot = get_blas_funcs('rot', dtype=dtype)
            f = 4
        else:
            rot = get_lapack_funcs('rot', dtype=dtype)
            s *= -1j
            v *= 1j
            f = 4j

        # 断言调用 rot 函数的结果与预期值接近，使用给定的 atol 容差
        assert_allclose(rot(u, v, c, s), [[5, 5, 5, 5],
                                          [0, 0, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, n=2), [[5, 5, 3, 3],
                                               [0, 0, f, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, offy=2),
                        [[3, 3, 5, 5], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=2, offy=2, n=2),
                        [[5, 3, 5, 3], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incy=2, n=2),
                        [[3, 3, 5, 5], [0, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incx=2, offy=2, incy=2, n=1),
                        [[3, 3, 5, 3], [f, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=-2, incy=-2, n=2),
                        [[5, 3, 5, 3], [0, f, 0, f]], atol=atol)

        # 测试 overwrite_x 和 overwrite_y 参数为 True 时的结果
        a, b = rot(u, v, c, s, overwrite_x=1, overwrite_y=1)
        assert_(a is u)
        assert_(b is v)
        assert_allclose(a, [5, 5, 5, 5], atol=atol)
        assert_allclose(b, [0, 0, 0, 0], atol=atol)


def test_larfg_larf():
    # 设置随机种子
    np.random.seed(1234)
    # 创建 4x4 的随机矩阵 a0
    a0 = np.random.random((4, 4))
    a0 = a0.T.dot(a0)

    # 创建 4x4 的随机复数矩阵 a0j
    a0j = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    a0j = a0j.T.conj().dot(a0j)

    # 我们的测试目标是使用 Householder 变换将 Hermite 矩阵减少到三对角形式的一步操作。
    # 遍历数据类型列表，包括 'f', 'd', 'F', 'D'
    for dtype in 'fdFD':
        # 获得特定数据类型（dtype）下的 LAPACK 函数 'larfg' 和 'larf'
        larfg, larf = get_lapack_funcs(['larfg', 'larf'], dtype=dtype)

        # 根据数据类型选择复制相应的矩阵 a0j 或者 a0
        if dtype in 'FD':
            a = a0j.copy()
        else:
            a = a0.copy()

        # 生成一个 Householder 变换，用于清除 a[2:,0] 元素
        alpha, x, tau = larfg(a.shape[0]-1, a[1, 0], a[2:, 0])

        # 创建预期输出数组
        expected = np.zeros_like(a[:, 0])
        expected[0] = a[0, 0]
        expected[1] = alpha

        # 组装 Householder 向量
        v = np.zeros_like(a[1:, 0])
        v[0] = 1.0
        v[1:] = x

        # 从左侧应用变换
        a[1:, :] = larf(v, tau.conjugate(), a[1:, :], np.zeros(a.shape[1]))

        # 从右侧应用变换
        a[:, 1:] = larf(v, tau, a[:, 1:], np.zeros(a.shape[0]), side='R')

        # 断言检查列向量和行向量的变换结果是否接近预期值
        assert_allclose(a[:, 0], expected, atol=1e-5)
        assert_allclose(a[0, :], expected, atol=1e-5)
# 测试 SGESDD 的 lwork 是否足够大以支持 LAPACK
#
# 这里检查 _compute_lwork() 是否正确处理 LAPACK 旧版本（小于 3.10.1）中的一个 bug。

def test_sgesdd_lwork_bug_workaround():
    sgesdd_lwork = get_lapack_funcs('gesdd_lwork', dtype=np.float32,
                                    ilp64='preferred')
    # 获取测试中使用的矩阵维度
    n = 9537
    # 计算所需的工作空间大小
    lwork = _compute_lwork(sgesdd_lwork, n, n,
                           compute_uv=True, full_matrices=True)
    
    # 如果直接调用 Fortran 函数 SGESDD，且使用 IWORK=-1，LAPACK 的 bug 会导致 lwork 为 272929856，这个值过小。
    # （结果以单精度浮点数返回，不足以精确表示内部计算的整数值。）
    # _compute_lwork() 中的 workaround 会将其修正为 272929888。
    # 如果使用的是 LAPACK 3.10.1 或更新版本（例如 OpenBLAS 0.3.21 或更新版本），work-around 将返回 272929920，
    # 因为它无法知道正在使用的 LAPACK 版本，所以总是对其给定的值应用修正。
    # 我们接受 272929888 或 272929920 作为合理的值。
    # 注意，这些值是 LAPACK 实现的细节。
    # 如果将来的 LAPACK 版本更改了 SGESDD 的工作方式，从而改变了所需的 LWORK 大小，可能需要更新可接受的值。
    assert lwork == 272929888 or lwork == 272929920


class TestSytrd:
    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    def test_sytrd_with_zero_dim_array(self, dtype):
        # 断言对于一个 0x0 的矩阵会触发错误
        A = np.zeros((0, 0), dtype=dtype)
        # 获取 'sytrd' 函数对象
        sytrd = get_lapack_funcs('sytrd', (A,))
        # 断言会引发 ValueError 异常
        assert_raises(ValueError, sytrd, A)

    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    @pytest.mark.parametrize('n', (1, 3))
    # 定义一个测试函数，用于测试对称矩阵的三对角化过程
    def test_sytrd(self, dtype, n):
        # 创建一个大小为 n x n 的零矩阵，并指定数据类型为 dtype
        A = np.zeros((n, n), dtype=dtype)

        # 获取 LAPACK 函数 sytrd 和 sytrd_lwork
        sytrd, sytrd_lwork = \
            get_lapack_funcs(('sytrd', 'sytrd_lwork'), (A,))

        # 在 A 的上三角部分填入从 1 开始递增的数列
        A[np.triu_indices_from(A)] = \
            np.arange(1, n*(n+1)//2+1, dtype=dtype)

        # 查询所需的工作空间大小 lwork，并检查返回信息是否为 0
        lwork, info = sytrd_lwork(n)
        assert_equal(info, 0)

        # 调用 sytrd 函数进行对称矩阵的三对角化，lower=1 表示矩阵是下三角的情况，但由于 A 是上三角，效果不大
        data, d, e, tau, info = sytrd(A, lower=1, lwork=lwork)
        assert_equal(info, 0)

        # 断言三对角化后的矩阵 data 应该与 A 非常接近，atol 设置为数值 dtype 的机器精度的 5 倍，rtol 设置为 1.0
        assert_allclose(data, A, atol=5*np.finfo(dtype).eps, rtol=1.0)

        # 断言对角线元素 d 应该与 A 的对角线元素一致
        assert_allclose(d, np.diag(A))

        # 断言副对角线元素 e 应该接近于 0.0
        assert_allclose(e, 0.0)

        # 断言 tau 应该接近于 0.0
        assert_allclose(tau, 0.0)

        # 使用默认参数 lower=0，进行对称矩阵的三对角化
        data, d, e, tau, info = sytrd(A, lwork=lwork)
        assert_equal(info, 0)

        # 构建三对角矩阵 T，其中对角线元素为 d，副对角线元素为 e
        T = np.zeros_like(A, dtype=dtype)
        k = np.arange(A.shape[0])
        T[k, k] = d
        k2 = np.arange(A.shape[0]-1)
        T[k2+1, k2] = e
        T[k2, k2+1] = e

        # 构建正交矩阵 Q
        Q = np.eye(n, n, dtype=dtype)
        for i in range(n-1):
            v = np.zeros(n, dtype=dtype)
            v[:i] = data[:i, i+1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=dtype) - tau[i] * np.outer(v, v)
            Q = np.dot(H, Q)

        # 将 A 变为完全对称矩阵
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = A.T[i_lower]

        # 计算 Q^T * A * Q，得到的结果应当接近于 T
        QTAQ = np.dot(Q.T, np.dot(A, Q))

        # 由于 QTAQ 和 T 中的一些值非常接近于 0，这里禁用 rtol 设置
        assert_allclose(QTAQ, T, atol=5*np.finfo(dtype).eps, rtol=1.0)
# 定义一个测试类 TestHetrd
class TestHetrd:
    # 使用 pytest 的参数化装饰器，对 COMPLEX_DTYPES 中的每种复杂数据类型进行测试
    @pytest.mark.parametrize('complex_dtype', COMPLEX_DTYPES)
    def test_hetrd_with_zero_dim_array(self, complex_dtype):
        # 断言对一个 0x0 的矩阵会触发 ValueError 异常
        A = np.zeros((0, 0), dtype=complex_dtype)
        # 获取 LAPACK 函数 'hetrd' 的实现
        hetrd = get_lapack_funcs('hetrd', (A,))
        # 断言调用 hetrd(A) 会引发 ValueError 异常
        assert_raises(ValueError, hetrd, A)

    # 使用 pytest 的参数化装饰器，对 REAL_DTYPES 和 COMPLEX_DTYPES 中的每种实数和复数数据类型进行测试
    @pytest.mark.parametrize('real_dtype,complex_dtype',
                             zip(REAL_DTYPES, COMPLEX_DTYPES))
    # 对 n 取 1 和 3 进行测试
    @pytest.mark.parametrize('n', (1, 3))
    def test_hetrd(self, n, real_dtype, complex_dtype):
        # 创建一个 n x n 的零矩阵，数据类型为 complex_dtype
        A = np.zeros((n, n), dtype=complex_dtype)
        # 获取 'hetrd' 和 'hetrd_lwork' LAPACK 函数
        hetrd, hetrd_lwork = \
            get_lapack_funcs(('hetrd', 'hetrd_lwork'), (A,))

        # 创建一个上三角矩阵 A
        A[np.triu_indices_from(A)] = (
            np.arange(1, n*(n+1)//2+1, dtype=real_dtype)
            + 1j * np.arange(1, n*(n+1)//2+1, dtype=real_dtype)
            )
        np.fill_diagonal(A, np.real(np.diag(A)))

        # 查询 lwork 的正确性
        for x in [0, 1]:
            # 调用 hetrd_lwork(n, lower=x)，获取 lwork 和 info
            _, info = hetrd_lwork(n, lower=x)
            # 断言 info 的值为 0
            assert_equal(info, 0)
        
        # 计算 lwork 的值，避免复杂数据类型的 lwork 引起 'hetrd' 调用段错误
        lwork = _compute_lwork(hetrd_lwork, n)

        # 测试 lower=1 的行为（由于矩阵是上三角的，不应该有太大影响）
        data, d, e, tau, info = hetrd(A, lower=1, lwork=lwork)
        # 断言 info 的值为 0
        assert_equal(info, 0)

        # 断言 data 矩阵与 A 矩阵在一定误差范围内相等
        assert_allclose(data, A, atol=5*np.finfo(real_dtype).eps, rtol=1.0)

        # 断言 d 与 A 的实部对角线相等
        assert_allclose(d, np.real(np.diag(A)))
        # 断言 e 全为 0
        assert_allclose(e, 0.0)
        # 断言 tau 全为 0
        assert_allclose(tau, 0.0)

        # 使用默认参数 lower=0 进行测试
        data, d, e, tau, info = hetrd(A, lwork=lwork)
        # 断言 info 的值为 0
        assert_equal(info, 0)

        # 构建三对角矩阵 T
        T = np.zeros_like(A, dtype=real_dtype)
        k = np.arange(A.shape[0], dtype=int)
        T[k, k] = d
        k2 = np.arange(A.shape[0]-1, dtype=int)
        T[k2+1, k2] = e
        T[k2, k2+1] = e

        # 构建单位矩阵 Q
        Q = np.eye(n, n, dtype=complex_dtype)
        for i in range(n-1):
            v = np.zeros(n, dtype=complex_dtype)
            v[:i] = data[:i, i+1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=complex_dtype) \
                - tau[i] * np.outer(v, np.conj(v))
            Q = np.dot(H, Q)

        # 将 A 矩阵变为完全共轭对称矩阵
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = np.conj(A.T[i_lower])

        # 计算 Q^H * A * Q
        QHAQ = np.dot(np.conj(Q.T), np.dot(A, Q))

        # 由于 QTAQ 和 T 的某些值非常接近 0，关闭 rtol 检查
        assert_allclose(
            QHAQ, T, atol=10*np.finfo(real_dtype).eps, rtol=1.0
            )
    # 遍历 DTYPES 列表中的每个元素和对应的索引值
    for ind, dtype in enumerate(DTYPES):
        # 根据 dtype 获取 LAPACK 函数 gglse 和 gglse_lwork
        func, func_lwork = get_lapack_funcs(('gglse', 'gglse_lwork'),
                                            dtype=dtype)
        # 计算所需的工作空间大小
        lwork = _compute_lwork(func_lwork, m=6, n=4, p=2)
        
        # 根据索引值判断使用哪组测试数据
        if ind < 2:
            # 对于 <s,d> 类型，定义系数矩阵 a
            a = np.array([[-0.57, -1.28, -0.39, 0.25],
                          [-1.93, 1.08, -0.31, -2.14],
                          [2.30, 0.24, 0.40, -0.35],
                          [-1.93, 0.64, -0.66, 0.08],
                          [0.15, 0.30, 0.15, -2.13],
                          [-0.02, 1.03, -1.43, 0.50]], dtype=dtype)
            # 定义向量 c 和 d
            c = np.array([-1.50, -2.14, 1.23, -0.54, -1.68, 0.82], dtype=dtype)
            d = np.array([0., 0.], dtype=dtype)
        else:
            # 对于 <c,z> 类型，定义复数类型的系数矩阵 a
            a = np.array([[0.96-0.81j, -0.03+0.96j, -0.91+2.06j, -0.05+0.41j],
                          [-0.98+1.98j, -1.20+0.19j, -0.66+0.42j, -0.81+0.56j],
                          [0.62-0.46j, 1.01+0.02j, 0.63-0.17j, -1.11+0.60j],
                          [0.37+0.38j, 0.19-0.54j, -0.98-0.36j, 0.22-0.20j],
                          [0.83+0.51j, 0.20+0.01j, -0.17-0.46j, 1.47+1.59j],
                          [1.08-0.28j, 0.20-0.12j, -0.07+1.23j, 0.26+0.26j]])
            # 定义复数类型的向量 c 和 d
            c = np.array([[-2.54+0.09j],
                          [1.65-2.26j],
                          [-2.11-3.96j],
                          [1.82+3.30j],
                          [-6.41+3.77j],
                          [2.07+0.66j]])
            d = np.zeros(2, dtype=dtype)

        # 定义矩阵 b
        b = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]], dtype=dtype)

        # 调用 LAPACK 函数 func 进行计算，获取结果
        _, _, _, result, _ = func(a, b, c, d, lwork=lwork)
        
        # 根据索引值判断预期的结果，并进行断言
        if ind < 2:
            expected = np.array([0.48904455,
                                 0.99754786,
                                 0.48904455,
                                 0.99754786])
        else:
            expected = np.array([1.08742917-1.96205783j,
                                 -0.74093902+3.72973919j,
                                 1.08742917-1.96205759j,
                                 -0.74093896+3.72973895j])
        assert_array_almost_equal(result, expected, decimal=4)
def test_sycon_hecon():
    # 设定随机种子为1234，确保结果可重复性
    seed(1234)
    # 对于每个数据类型，依次执行以下操作
    for ind, dtype in enumerate(DTYPES+COMPLEX_DTYPES):
        # DTYPES + COMPLEX DTYPES = <s,d,c,z> sycon + <c,z>hecon
        n = 10
        # 对于 <s,d,c,z>sycon
        if ind < 4:
            # 获取 LAPACK 函数 sytrf_lwork 的函数对象
            func_lwork = get_lapack_funcs('sytrf_lwork', dtype=dtype)
            # 获取 LAPACK 函数 sycon 和 sytrf 的函数对象
            funcon, functrf = get_lapack_funcs(('sycon', 'sytrf'), dtype=dtype)
            # 生成一个随机矩阵 A，并转换为指定数据类型
            A = (rand(n, n)).astype(dtype)
        # 对于 <c,z>hecon
        else:
            # 获取 LAPACK 函数 hetrf_lwork 的函数对象
            func_lwork = get_lapack_funcs('hetrf_lwork', dtype=dtype)
            # 获取 LAPACK 函数 hecon 和 hetrf 的函数对象
            funcon, functrf = get_lapack_funcs(('hecon', 'hetrf'), dtype=dtype)
            # 生成一个随机复数矩阵 A，并转换为指定数据类型
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)

        # 将 A 转换为 Hermitian 矩阵，处理共轭部分
        A = (A + A.conj().T)/2 + 2*np.eye(n, dtype=dtype)

        # 计算 A 的 1-范数
        anorm = norm(A, 1)
        # 计算 LAPACK 函数 func_lwork 的工作空间
        lwork = _compute_lwork(func_lwork, n)
        # 对 A 进行 LU 分解，返回 LDU 分解、置换向量 ipiv 和信息
        ldu, ipiv, _ = functrf(A, lwork=lwork, lower=1)
        # 计算矩阵 A 的条件数的估计值 rcond
        rcond, _ = funcon(a=ldu, ipiv=ipiv, anorm=anorm, lower=1)
        # 断言条件数的误差最多为原条件数的一倍
        assert_(abs(1/rcond - np.linalg.cond(A, p=1))*rcond < 1)


def test_sygst():
    # 设定随机种子为1234，确保结果可重复性
    seed(1234)
    # 对于每个实数数据类型，依次执行以下操作
    for ind, dtype in enumerate(REAL_DTYPES):
        # DTYPES = <s,d> sygst
        n = 10

        # 获取 LAPACK 函数 potrf, sygst, syevd, sygvd 的函数对象
        potrf, sygst, syevd, sygvd = get_lapack_funcs(('potrf', 'sygst',
                                                       'syevd', 'sygvd'),
                                                      dtype=dtype)

        # 生成一个随机对称矩阵 A，并转换为指定数据类型
        A = rand(n, n).astype(dtype)
        A = (A + A.T)/2
        # 生成一个随机对称正定矩阵 B，并转换为指定数据类型
        B = rand(n, n).astype(dtype)
        B = (B + B.T)/2 + 2 * np.eye(n, dtype=dtype)

        # 执行广义特征值问题的求解 (sygvd)
        eig_gvd, _, info = sygvd(A, B)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)

        # 将广义特征值问题转换为标准特征值问题 (potrf)
        b, info = potrf(B)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)
        a, info = sygst(A, b)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)

        # 执行特征值分解 (syevd)
        eig, _, info = syevd(a)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)
        # 断言特征值解符合预期，相对误差不超过 1.2e-4
        assert_allclose(eig, eig_gvd, rtol=1.2e-4)


def test_hegst():
    # 设定随机种子为1234，确保结果可重复性
    seed(1234)
    # 对于每个复数数据类型，依次执行以下操作
    for ind, dtype in enumerate(COMPLEX_DTYPES):
        # DTYPES = <c,z> hegst
        n = 10

        # 获取 LAPACK 函数 potrf, hegst, heevd, hegvd 的函数对象
        potrf, hegst, heevd, hegvd = get_lapack_funcs(('potrf', 'hegst',
                                                       'heevd', 'hegvd'),
                                                      dtype=dtype)

        # 生成一个随机 Hermite 矩阵 A，并转换为指定数据类型
        A = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        A = (A + A.conj().T)/2
        # 生成一个随机 Hermite 正定矩阵 B，并转换为指定数据类型
        B = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        B = (B + B.conj().T)/2 + 2 * np.eye(n, dtype=dtype)

        # 执行广义特征值问题的求解 (hegvd)
        eig_gvd, _, info = hegvd(A, B)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)

        # 将广义特征值问题转换为标准特征值问题 (potrf)
        b, info = potrf(B)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)
        a, info = hegst(A, b)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)

        # 执行特征值分解 (heevd)
        eig, _, info = heevd(a)
        # 断言 LAPACK 操作返回状态为成功 (info = 0)
        assert_(info == 0)
        # 断言特征值解符合预期，相对误差不超过 1e-4
        assert_allclose(eig, eig_gvd, rtol=1e-4)


def test_tzrzf():
    """
    This test performs an RZ decomposition in which an m x n upper trapezoidal
    array M (m <= n) is factorized as M = [R 0] * Z where R is upper triangular
    and Z is unitary.
    """
    # 设置随机种子为1234，用于使每次运行时的随机数结果可复现
    seed(1234)
    # 设置矩阵的行数m和列数n
    m, n = 10, 15
    # 对于提供的数据类型列表DTYPES中的每一种数据类型，依次执行以下操作
    for ind, dtype in enumerate(DTYPES):
        # 获取 LAPACK 函数 'tzrzf' 和 'tzrzf_lwork'，适用于当前数据类型dtype
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'),
                                           dtype=dtype)
        # 计算所需的工作空间大小
        lwork = _compute_lwork(tzrzf_lw, m, n)

        # 根据不同情况生成随机上三角矩阵A
        if ind < 2:
            A = triu(rand(m, n).astype(dtype))
        else:
            A = triu((rand(m, n) + rand(m, n)*1j).astype(dtype))

        # 断言：传递错误形状的参数时会抛出异常，f2py 会返回通用错误
        assert_raises(Exception, tzrzf, A.T)
        # 执行 RZ 分解，得到 R、tau 和 info
        rz, tau, info = tzrzf(A, lwork=lwork)
        # 检查分解是否成功
        assert_(info == 0)

        # 手动获取 Z 用于比较
        R = np.hstack((rz[:, :m], np.zeros((m, n-m), dtype=dtype)))
        V = np.hstack((np.eye(m, dtype=dtype), rz[:, m:]))
        Id = np.eye(n, dtype=dtype)
        # 计算参考值 ref
        ref = [Id-tau[x]*V[[x], :].T.dot(V[[x], :].conj()) for x in range(m)]
        Z = reduce(np.dot, ref)
        # 断言 RZ 分解的结果是否与原始矩阵 A 相近
        assert_allclose(R.dot(Z) - A, zeros_like(A, dtype=dtype),
                        atol=10*np.spacing(dtype(1.0).real), rtol=0.)
# 定义测试函数 test_tfsm，用于测试解决系数矩阵为 Full Packed (RFP) 格式的三角数组的线性系统求解问题。
def test_tfsm():
    """
    Test for solving a linear system with the coefficient matrix is a
    triangular array stored in Full Packed (RFP) format.
    """
    # 设置随机种子为 1234
    seed(1234)
    # 遍历 DTYPES 中的数据类型
    for ind, dtype in enumerate(DTYPES):
        # 设置矩阵维度为 n = 20
        n = 20
        # 根据索引 ind 和数据类型选择 A 的构造方式
        if ind > 1:
            # A 是一个上三角矩阵，元素为随机实部和虚部的和加上单位阵的 dtype 版本
            A = triu(rand(n, n) + rand(n, n)*1j + eye(n)).astype(dtype)
            trans = 'C'
        else:
            # A 是一个上三角矩阵，元素为随机数加上单位阵的 dtype 版本
            A = triu(rand(n, n) + eye(n)).astype(dtype)
            trans = 'T'

        # 获取 LAPACK 函数 trttf, tfttr, tfsm
        trttf, tfttr, tfsm = get_lapack_funcs(('trttf', 'tfttr', 'tfsm'),
                                              dtype=dtype)

        # 对 A 应用 trttf 函数，返回 Afp 和 _
        Afp, _ = trttf(A)
        # 创建一个随机的 B 矩阵，dtype 与 A 相同
        B = rand(n, 2).astype(dtype)
        # 使用 tfsm 函数求解线性系统 -A * X = B
        soln = tfsm(-1, Afp, B)
        # 断言求解结果与 solve(-A, B) 的近似相等性，精度为 4 或 6 位小数
        assert_array_almost_equal(soln, solve(-A, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        # 使用 tfsm 函数求解线性系统 -A^H * X = B
        soln = tfsm(-1, Afp, B, trans=trans)
        # 断言求解结果与 solve(-A.conj().T, B) 的近似相等性，精度为 4 或 6 位小数
        assert_array_almost_equal(soln, solve(-A.conj().T, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        # 将 A 的对角线元素设置为单位元素，单位阵化 A
        A[np.arange(n), np.arange(n)] = dtype(1.)
        # 使用 tfsm 函数求解单位阵化后的线性系统 -A^H * X = B
        soln = tfsm(-1, Afp, B, trans=trans, diag='U')
        # 断言求解结果与 solve(-A.conj().T, B) 的近似相等性，精度为 4 或 6 位小数
        assert_array_almost_equal(soln, solve(-A.conj().T, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        # 改变 B 的形状，转置 B 为 B2
        B2 = rand(3, n).astype(dtype)
        # 使用 tfsm 函数求解改变形状后的线性系统 -A * X = B2^T
        soln = tfsm(-1, Afp, B2, trans=trans, diag='U', side='R')
        # 断言求解结果与 solve(-A, B2.T).conj().T 的近似相等性，精度为 4 或 6 位小数
        assert_array_almost_equal(soln, solve(-A, B2.T).conj().T,
                                  decimal=4 if ind % 2 == 0 else 6)


# 定义测试函数 test_ormrz_unmrz，用于测试通过单位矩阵 Q 执行矩阵乘法而无需显式形成数组。
def test_ormrz_unmrz():
    """
    This test performs a matrix multiplication with an arbitrary m x n matrix C
    and a unitary matrix Q without explicitly forming the array. The array data
    is encoded in the rectangular part of A which is obtained from ?TZRZF. Q
    size is inferred by m, n, side keywords.
    """
    # 设置随机种子为 1234
    seed(1234)
    # 定义 Q 的行数 qm，列数 qn，C 的列数 cn
    qm, qn, cn = 10, 15, 15
    # 对于给定的数据类型列表DTYPES中的每个数据类型执行以下操作
    for ind, dtype in enumerate(DTYPES):
        # 获取 LAPACK 库函数 tzrzf 和 tzrzf_lwork 对应的函数对象
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'),
                                           dtype=dtype)
        # 计算所需的工作空间大小 lwork_rz
        lwork_rz = _compute_lwork(tzrzf_lw, qm, qn)

        # 根据不同的索引条件，初始化矩阵 A 和矩阵 C
        if ind < 2:
            # 对于索引小于2的情况，A 是一个上三角矩阵，C 是一个随机矩阵
            A = triu(rand(qm, qn).astype(dtype))
            C = rand(cn, cn).astype(dtype)
            # 获取 LAPACK 库函数 ormz 和 ormz_lwork 对应的函数对象
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('ormrz', 'ormrz_lwork'),
                                                     dtype=dtype)
        else:
            # 对于索引大于等于2的情况，A 是一个包含复数的上三角矩阵，C 是一个包含复数的随机矩阵
            A = triu((rand(qm, qn) + rand(qm, qn)*1j).astype(dtype))
            C = (rand(cn, cn) + rand(cn, cn)*1j).astype(dtype)
            # 获取 LAPACK 库函数 unmrz 和 unmrz_lwork 对应的函数对象
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('unmrz', 'unmrz_lwork'),
                                                     dtype=dtype)

        # 计算所需的工作空间大小 lwork_mrz
        lwork_mrz = _compute_lwork(orun_mrz_lw, cn, cn)
        
        # 对矩阵 A 进行 QR 分解，得到 rz（QR 分解后的上三角矩阵）、tau 和 info
        rz, tau, info = tzrzf(A, lwork=lwork_rz)

        # 手动计算 Q 以便后续比较
        V = np.hstack((np.eye(qm, dtype=dtype), rz[:, qm:]))
        Id = np.eye(qn, dtype=dtype)
        ref = [Id - tau[x]*V[[x], :].T.dot(V[[x], :].conj()) for x in range(qm)]
        Q = reduce(np.dot, ref)

        # 使用 lapack 函数 orun_mrz 计算 CQ、CQ^H、QC、QC^H，并进行比较
        trans = 'T' if ind < 2 else 'C'  # 根据索引选择转置类型
        tol = 10*np.spacing(dtype(1.0).real)  # 容忍度设置为极小值的10倍

        # 计算 CQ，并断言结果的信息值为0
        cq, info = orun_mrz(rz, tau, C, lwork=lwork_mrz)
        assert_(info == 0)
        # 断言 CQ 与预期值 Q.dot(C) 非常接近
        assert_allclose(cq - Q.dot(C), zeros_like(C), atol=tol, rtol=0.)

        # 计算 CQ^H，并断言结果的信息值为0
        cq, info = orun_mrz(rz, tau, C, trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        # 断言 CQ^H 与预期值 Q^H.dot(C) 非常接近
        assert_allclose(cq - Q.conj().T.dot(C), zeros_like(C), atol=tol,
                        rtol=0.)

        # 计算 QC，并断言结果的信息值为0
        cq, info = orun_mrz(rz, tau, C, side='R', lwork=lwork_mrz)
        assert_(info == 0)
        # 断言 QC 与预期值 C.dot(Q) 非常接近
        assert_allclose(cq - C.dot(Q), zeros_like(C), atol=tol, rtol=0.)

        # 计算 QC^H，并断言结果的信息值为0
        cq, info = orun_mrz(rz, tau, C, side='R', trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        # 断言 QC^H 与预期值 C.dot(Q^H) 非常接近
        assert_allclose(cq - C.dot(Q.conj().T), zeros_like(C), atol=tol,
                        rtol=0.)
# 定义一个用于测试 Rectengular Full Packed (RFP) 格式和 Standard Triangular Array (TR) 之间转换的函数
def test_tfttr_trttf():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    # 设定随机数种子为1234，确保可重复性
    seed(1234)
    
    # 遍历不同的数据类型
    for ind, dtype in enumerate(DTYPES):
        # 定义矩阵大小为20
        n = 20
        
        # 根据数据类型不同，生成不同的 A_full 矩阵
        if ind > 1:
            # 生成复数类型的 A_full
            A_full = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            transr = 'C'  # 设置转置标志为 'C'
        else:
            # 生成实数类型的 A_full
            A_full = (rand(n, n)).astype(dtype)
            transr = 'T'  # 设置转置标志为 'T'
        
        # 获取 LAPACK 函数 trttf 和 tfttr
        trttf, tfttr = get_lapack_funcs(('trttf', 'tfttr'), dtype=dtype)
        
        # 调用 trttf 函数进行 RFP 到 TR 的转换，上三角部分
        A_tf_U, info = trttf(A_full)
        assert_(info == 0)  # 确保转换成功
        
        # 调用 trttf 函数进行 RFP 到 TR 的转换，下三角部分
        A_tf_L, info = trttf(A_full, uplo='L')
        assert_(info == 0)  # 确保转换成功
        
        # 调用 trttf 函数进行 RFP 到 TR 的转换，上三角部分，转置
        A_tf_U_T, info = trttf(A_full, transr=transr, uplo='U')
        assert_(info == 0)  # 确保转换成功
        
        # 调用 trttf 函数进行 RFP 到 TR 的转换，下三角部分，转置
        A_tf_L_T, info = trttf(A_full, transr=transr, uplo='L')
        assert_(info == 0)  # 确保转换成功
        
        # 手动创建 RFP 数组 A_tf_U_m（偶数情况）
        A_tf_U_m = zeros((n+1, n//2), dtype=dtype)
        A_tf_U_m[:-1, :] = triu(A_full)[:, n//2:]
        A_tf_U_m[n//2+1:, :] += triu(A_full)[:n//2, :n//2].conj().T
        
        # 手动创建 RFP 数组 A_tf_L_m（偶数情况）
        A_tf_L_m = zeros((n+1, n//2), dtype=dtype)
        A_tf_L_m[1:, :] = tril(A_full)[:, :n//2]
        A_tf_L_m[:n//2, :] += tril(A_full)[n//2:, n//2:].conj().T
        
        # 检查转换后的结果与手动创建的 RFP 数组是否一致
        assert_array_almost_equal(A_tf_U, A_tf_U_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_U_T,
                                  A_tf_U_m.conj().T.reshape(-1, order='F'))
        
        assert_array_almost_equal(A_tf_L, A_tf_L_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_L_T,
                                  A_tf_L_m.conj().T.reshape(-1, order='F'))
        
        # 从 RFP 数组获取原始数组
        A_tr_U, info = tfttr(n, A_tf_U)
        assert_(info == 0)  # 确保转换成功
        
        A_tr_L, info = tfttr(n, A_tf_L, uplo='L')
        assert_(info == 0)  # 确保转换成功
        
        A_tr_U_T, info = tfttr(n, A_tf_U_T, transr=transr, uplo='U')
        assert_(info == 0)  # 确保转换成功
        
        A_tr_L_T, info = tfttr(n, A_tf_L_T, transr=transr, uplo='L')
        assert_(info == 0)  # 确保转换成功
        
        # 检查转换后的结果与原始数组是否一致
        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_U_T, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))
        assert_array_almost_equal(A_tr_L_T, tril(A_full))


def test_tpttr_trttp():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    seed(1234)
    # 遍历 DTYPES 列表，同时获取索引和数据类型
    for ind, dtype in enumerate(DTYPES):
        # 设置矩阵维度为20
        n = 20
        # 如果索引大于1，则创建复数类型的随机实数矩阵，并转换为指定的数据类型
        if ind > 1:
            A_full = (rand(n, n) + rand(n, n)*1j).astype(dtype)
        else:
            # 否则创建实数类型的随机矩阵，并转换为指定的数据类型
            A_full = (rand(n, n)).astype(dtype)

        # 获取 LAPACK 函数 trttp 和 tpttr，用于处理特殊矩阵的转换
        trttp, tpttr = get_lapack_funcs(('trttp', 'tpttr'), dtype=dtype)
        
        # 对称矩阵的上三角部分转换为 TP 格式，info 返回状态信息
        A_tp_U, info = trttp(A_full)
        assert_(info == 0)  # 断言确保操作成功完成
        # 对称矩阵的下三角部分转换为 TP 格式，info 返回状态信息
        A_tp_L, info = trttp(A_full, uplo='L')
        assert_(info == 0)  # 断言确保操作成功完成

        # 手动创建 TP 格式的数组 A_tp_U_m
        inds = tril_indices(n)
        A_tp_U_m = zeros(n*(n+1)//2, dtype=dtype)
        A_tp_U_m[:] = (triu(A_full).T)[inds]

        # 手动创建 TP 格式的数组 A_tp_L_m
        inds = triu_indices(n)
        A_tp_L_m = zeros(n*(n+1)//2, dtype=dtype)
        A_tp_L_m[:] = (tril(A_full).T)[inds]

        # 断言确保计算得到的 TP 格式数组与手动创建的一致
        assert_array_almost_equal(A_tp_U, A_tp_U_m)
        assert_array_almost_equal(A_tp_L, A_tp_L_m)

        # 从 TP 格式数组还原得到原始的对称矩阵 A_tr_U 和 A_tr_L
        A_tr_U, info = tpttr(n, A_tp_U)
        assert_(info == 0)  # 断言确保操作成功完成
        A_tr_L, info = tpttr(n, A_tp_L, uplo='L')
        assert_(info == 0)  # 断言确保操作成功完成

        # 断言确保从 TP 格式数组还原的对称矩阵与原始矩阵的上下三角部分一致
        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))
# 定义测试函数，用于测试正定矩形全打包（RFP）格式数组的Cholesky分解
def test_pftrf():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array
    """
    # 设置随机数种子为1234，确保可重复性
    seed(1234)
    # 遍历不同的数据类型
    for ind, dtype in enumerate(DTYPES):
        # 设置矩阵的大小为20
        n = 20
        # 根据索引选择不同的矩阵构造方式
        if ind > 1:
            # 构造包含虚部的复数矩阵，并使其正定
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            # 构造实数矩阵，并使其正定
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        # 获取 LAPACK 库中的函数：pftrf, trttf, tfttr
        pftrf, trttf, tfttr = get_lapack_funcs(('pftrf', 'trttf', 'tfttr'),
                                               dtype=dtype)

        # 从矩阵A中获取原始数组Afp，并返回信息info
        Afp, info = trttf(A)
        # 对Afp进行Cholesky分解，得到Achol_rfp和信息info
        Achol_rfp, info = pftrf(n, Afp)
        # 断言info为0，表示Cholesky分解成功
        assert_(info == 0)
        # 从Achol_rfp中获取原始数组A_chol_r，并返回信息_
        A_chol_r, _ = tfttr(n, Achol_rfp)
        # 使用标准库中的Cholesky分解函数对矩阵A进行分解，得到Achol
        Achol = cholesky(A)
        # 断言A_chol_r与Achol几乎完全相等
        assert_array_almost_equal(A_chol_r, Achol)


# 定义测试函数，用于测试正定矩形全打包（RFP）格式数组的Cholesky分解并找到其逆矩阵
def test_pftri():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array to find its inverse
    """
    # 设置随机数种子为1234，确保可重复性
    seed(1234)
    # 遍历不同的数据类型
    for ind, dtype in enumerate(DTYPES):
        # 设置矩阵的大小为20
        n = 20
        # 根据索引选择不同的矩阵构造方式
        if ind > 1:
            # 构造包含虚部的复数矩阵，并使其正定
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            # 构造实数矩阵，并使其正定
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        # 获取 LAPACK 库中的函数：pftri, pftrf, trttf, tfttr
        pftri, pftrf, trttf, tfttr = get_lapack_funcs(('pftri',
                                                       'pftrf',
                                                       'trttf',
                                                       'tfttr'),
                                                      dtype=dtype)

        # 从矩阵A中获取原始数组Afp，并返回信息info
        Afp, info = trttf(A)
        # 对Afp进行Cholesky分解，得到A_chol_rfp和信息info
        A_chol_rfp, info = pftrf(n, Afp)
        # 对A_chol_rfp进行逆Cholesky分解，得到A_inv_rfp和信息info
        A_inv_rfp, info = pftri(n, A_chol_rfp)
        # 断言info为0，表示逆Cholesky分解成功
        assert_(info == 0)
        # 从A_inv_rfp中获取原始数组A_inv_r，并返回信息_
        A_inv_r, _ = tfttr(n, A_inv_rfp)
        # 使用标准库中的逆函数计算矩阵A的逆矩阵Ainv，并与A_inv_r近似比较
        Ainv = inv(A)
        assert_array_almost_equal(A_inv_r, triu(Ainv),
                                  decimal=4 if ind % 2 == 0 else 6)


# 定义测试函数，用于测试正定矩形全打包（RFP）格式数组的Cholesky分解并解线性系统
def test_pftrs():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array and solve a linear system
    """
    # 设置随机数种子为1234，确保可重复性
    seed(1234)
    # 遍历 DTYPES 列表中的数据类型和它们的索引
    for ind, dtype in enumerate(DTYPES):
        # 设置矩阵的大小
        n = 20
        # 根据索引选择不同的分支，生成不同类型的复数矩阵 A
        if ind > 1:
            # 生成复数矩阵 A，包含随机实部和虚部，转换为指定的数据类型
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            # 对称化 A 并加上对角矩阵
            A = A + A.conj().T + n * eye(n)
        else:
            # 生成实数矩阵 A，转换为指定的数据类型
            A = (rand(n, n)).astype(dtype)
            # 对称化 A 并加上对角矩阵
            A = A + A.T + n * eye(n)

        # 生成矩阵 B 和两个不同尺寸的矩阵 Bf1、Bf2，数据类型为 dtype
        B = ones((n, 3), dtype=dtype)
        Bf1 = ones((n + 2, 3), dtype=dtype)
        Bf2 = ones((n - 2, 3), dtype=dtype)

        # 获取指定 LAPACK 函数的函数句柄
        pftrs, pftrf, trttf, tfttr = get_lapack_funcs(('pftrs',
                                                       'pftrf',
                                                       'trttf',
                                                       'tfttr'),
                                                      dtype=dtype)

        # 从 TP 中获取原始数组 Afp，并返回额外的信息
        Afp, info = trttf(A)
        # 对 Afp 进行 Cholesky 分解，返回 Cholesky 因子和信息
        A_chol_rfp, info = pftrf(n, Afp)
        
        # 使用 Cholesky 分解的结果解线性系统 A_chol_rfp * soln = Bf1，返回解 soln 和信息
        soln, info = pftrs(n, A_chol_rfp, Bf1)
        # 断言信息 info 为 0，表示解成功
        assert_(info == 0)
        # 断言对于更大的 Bf2 矩阵，调用 pftrs 应该引发异常
        assert_raises(Exception, pftrs, n, A_chol_rfp, Bf2)
        # 使用矩阵 B 解线性系统 A_chol_rfp * soln = B，返回解 soln 和信息
        soln, info = pftrs(n, A_chol_rfp, B)
        # 断言信息 info 为 0，表示解成功
        assert_(info == 0)
        
        # 断言解 soln 和 solve(A, B) 函数返回的结果在一定的数值精度内相等
        assert_array_almost_equal(solve(A, B), soln,
                                  decimal=4 if ind % 2 == 0 else 6)
def test_sfrk_hfrk():
    """
    Test for performing a symmetric rank-k operation for matrix in RFP format.
    """
    # 设置随机种子以确保可复现性
    seed(1234)
    
    # 遍历不同的数据类型
    for ind, dtype in enumerate(DTYPES):
        n = 20
        
        # 根据索引选择不同的矩阵构造方式
        if ind > 1:
            # 创建复数类型的随机矩阵，并使其对称
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n * eye(n)
        else:
            # 创建实数类型的随机矩阵，并使其对称
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n * eye(n)

        # 根据索引选择不同的函数前缀
        prefix = 's' if ind < 2 else 'h'
        
        # 获取 LAPACK 函数 trttf, tfttr, 和相应的 frk 函数
        trttf, tfttr, shfrk = get_lapack_funcs(('trttf', 'tfttr', f'{prefix}frk'),
                                               dtype=dtype)

        # 对 A 进行操作，获取 Afp
        Afp, _ = trttf(A)
        
        # 创建随机矩阵 C
        C = np.random.rand(n, 2).astype(dtype)
        
        # 调用特定的 frk 函数，进行操作并获取输出 Afp_out
        Afp_out = shfrk(n, 2, -1, C, 2, Afp)
        
        # 使用 tfttr 函数对 Afp_out 进行逆操作，得到 A_out
        A_out, _ = tfttr(n, Afp_out)
        
        # 断言 A_out 与预期值的接近程度，根据不同的 ind 确定精度
        assert_array_almost_equal(A_out, triu(-C.dot(C.conj().T) + 2 * A),
                                  decimal=4 if ind % 2 == 0 else 6)


def test_syconv():
    """
    Test for going back and forth between the returned format of he/sytrf to
    L and D factors/permutations.
    """
    # 设置随机种子以确保可复现性
    seed(1234)
    
    # 遍历不同的数据类型
    for ind, dtype in enumerate(DTYPES):
        n = 10

        # 根据索引选择不同的矩阵构造方式
        if ind > 1:
            # 创建复数类型的随机矩阵，并使其对称
            A = (randint(-30, 30, (n, n)) +
                 randint(-30, 30, (n, n)) * 1j).astype(dtype)

            A = A + A.conj().T
        else:
            # 创建实数类型的随机矩阵，并使其对称
            A = randint(-30, 30, (n, n)).astype(dtype)
            A = A + A.T + n * eye(n)

        # 设置容差
        tol = 100 * np.spacing(dtype(1.0).real)
        
        # 获取 LAPACK 函数 syconv, sytrf, sytrf_lwork
        syconv, trf, trf_lwork = get_lapack_funcs(('syconv', 'sytrf',
                                                   'sytrf_lwork'), dtype=dtype)
        
        # 计算所需的工作空间大小
        lw = _compute_lwork(trf_lwork, n, lower=1)
        
        # 对 A 进行 LDL 分解，得到 L, D 和 perm
        L, D, perm = ldl(A, lower=1, hermitian=False)
        
        # 执行 LAPACK 中的 trf 操作，得到 ldu, ipiv 和 info
        ldu, ipiv, info = trf(A, lower=1, lwork=lw)
        
        # 使用 syconv 函数进行操作，得到 a, e 和 info
        a, e, info = syconv(ldu, ipiv, lower=1)
        
        # 断言 a 与 L[perm, :] 的下三角部分接近
        assert_allclose(tril(a, -1), tril(L[perm, :], -1), atol=tol, rtol=0.)

        # 测试上三角情况
        U, D, perm = ldl(A, lower=0, hermitian=False)
        ldu, ipiv, info = trf(A, lower=0)
        a, e, info = syconv(ldu, ipiv, lower=0)
        assert_allclose(triu(a, 1), triu(U[perm, :], 1), atol=tol, rtol=0.)


class TestBlockedQR:
    """
    Tests for the blocked QR factorization, namely through geqrt, gemqrt, tpqrt
    and tpmqr.
    """
    # 定义测试函数 test_geqrt_gemqrt
    def test_geqrt_gemqrt(self):
        # 设置随机数种子为1234
        seed(1234)
        # 遍历不同的数据类型
        for ind, dtype in enumerate(DTYPES):
            n = 20

            # 根据索引选择不同的数据类型创建矩阵A
            if ind > 1:
                # 如果索引大于1，则创建复数类型的随机矩阵
                A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            else:
                # 否则创建实数类型的随机矩阵
                A = (rand(n, n)).astype(dtype)

            # 设置误差容差
            tol = 100*np.spacing(dtype(1.0).real)
            # 获取 LAPACK 函数 geqrt 和 gemqrt
            geqrt, gemqrt = get_lapack_funcs(('geqrt', 'gemqrt'), dtype=dtype)

            # 调用 geqrt 函数进行 QR 分解
            a, t, info = geqrt(n, A)
            # 断言 info 等于 0，表示成功完成 QR 分解
            assert info == 0

            # 从下三角矩阵a中提取初等反射器，并加上主对角线的1
            v = np.tril(a, -1) + np.eye(n, dtype=dtype)
            # 生成分块 Householder 变换 I - VTV^H
            Q = np.eye(n, dtype=dtype) - v @ t @ v.T.conj()
            R = np.triu(a)

            # 断言 Q 的转置共轭乘以 Q 等于单位矩阵，检验 Q 的列是否正交
            assert_allclose(Q.T.conj() @ Q, np.eye(n, dtype=dtype), atol=tol, rtol=0.)
            # 断言 QR 分解是否还原原始矩阵 A
            assert_allclose(Q @ R, A, atol=tol, rtol=0.)

            # 根据索引选择不同的数据类型创建矩阵 C 和 transpose 类型
            if ind > 1:
                C = (rand(n, n) + rand(n, n)*1j).astype(dtype)
                transpose = 'C'
            else:
                C = (rand(n, n)).astype(dtype)
                transpose = 'T'

            # 遍历不同的 side 和 trans 参数组合
            for side in ('L', 'R'):
                for trans in ('N', transpose):
                    # 调用 gemqrt 函数进行 gemm-based QR 变换
                    c, info = gemqrt(a, t, C, side=side, trans=trans)
                    # 断言 info 等于 0，表示成功完成 gemqrt 变换
                    assert info == 0

                    # 根据 trans 参数选择 Q 或者 Q 的转置共轭
                    if trans == transpose:
                        q = Q.T.conj()
                    else:
                        q = Q

                    # 根据 side 参数计算 qC
                    if side == 'L':
                        qC = q @ C
                    else:
                        qC = C @ q

                    # 断言 gemqrt 变换后的结果 c 等于 qC
                    assert_allclose(c, qC, atol=tol, rtol=0.)

                    # 测试默认参数下 gemqrt 的行为
                    if (side, trans) == ('L', 'N'):
                        c_default, info = gemqrt(a, t, C)
                        # 断言 info 等于 0，表示成功完成 gemqrt 变换
                        assert info == 0
                        # 断言使用默认参数得到的结果与指定参数一致
                        assert_equal(c_default, c)

            # 测试不支持的 side/trans 参数组合是否会抛出异常
            assert_raises(Exception, gemqrt, a, t, C, side='A')
            assert_raises(Exception, gemqrt, a, t, C, trans='A')
def test_pstrf():
    # 设置随机种子以保证可重复性
    seed(1234)
    # 对于每种数据类型进行迭代
    for ind, dtype in enumerate(DTYPES):
        # DTYPES = <s, d, c, z> pstrf
        n = 10  # 矩阵的大小为 10x10
        r = 2   # 前 r 个主子阵的秩
        pstrf = get_lapack_funcs('pstrf', dtype=dtype)

        # 创建正半定矩阵 A
        if ind > 1:
            A = rand(n, n-r).astype(dtype) + 1j * rand(n, n-r).astype(dtype)
            A = A @ A.conj().T  # A = A @ A的共轭转置
        else:
            A = rand(n, n-r).astype(dtype)
            A = A @ A.T  # A = A @ A的转置

        # 执行 pstrf 分解
        c, piv, r_c, info = pstrf(A)
        U = triu(c)  # 提取 c 的上三角部分
        U[r_c - n:, r_c - n:] = 0.  # 将 U 的后 r 行列置零

        # 断言确保返回的信息是正常的
        assert_equal(info, 1)
        # 设置浮点数误差的绝对容忍度
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        # 断言近似相等性，检查分解结果的正确性
        assert_allclose(A[piv-1][:, piv-1], U.conj().T @ U, rtol=0., atol=atol)

        # 以 lower=1 参数再次执行 pstrf 分解
        c, piv, r_c, info = pstrf(A, lower=1)
        L = tril(c)  # 提取 c 的下三角部分
        L[r_c - n:, r_c - n:] = 0.  # 将 L 的后 r 行列置零

        # 断言确保返回的信息是正常的
        assert_equal(info, 1)
        # 设置浮点数误差的绝对容忍度
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        # 断言近似相等性，检查分解结果的正确性
        assert_allclose(A[piv-1][:, piv-1], L @ L.conj().T, rtol=0., atol=atol)


def test_pstf2():
    # 设置随机种子以保证可重复性
    seed(1234)
    # 对于每种数据类型进行迭代
    for ind, dtype in enumerate(DTYPES):
        # DTYPES = <s, d, c, z> pstf2
        n = 10  # 矩阵的大小为 10x10
        r = 2   # 前 r 个主子阵的秩
        pstf2 = get_lapack_funcs('pstf2', dtype=dtype)

        # 创建正半定矩阵 A
        if ind > 1:
            A = rand(n, n-r).astype(dtype) + 1j * rand(n, n-r).astype(dtype)
            A = A @ A.conj().T  # A = A @ A的共轭转置
        else:
            A = rand(n, n-r).astype(dtype)
            A = A @ A.T  # A = A @ A的转置

        # 执行 pstf2 分解
        c, piv, r_c, info = pstf2(A)
        U = triu(c)  # 提取 c 的上三角部分
        U[r_c - n:, r_c - n:] = 0.  # 将 U 的后 r 行列置零

        # 断言确保返回的信息是正常的
        assert_equal(info, 1)
        # 设置浮点数误差的绝对容忍度
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        # 断言近似相等性，检查分解结果的正确性
        assert_allclose(A[piv-1][:, piv-1], U.conj().T @ U, rtol=0., atol=atol)

        # 以 lower=1 参数再次执行 pstf2 分解
        c, piv, r_c, info = pstf2(A, lower=1)
        L = tril(c)  # 提取 c 的下三角部分
        L[r_c - n:, r_c - n:] = 0.  # 将 L 的后 r 行列置零

        # 断言确保返回的信息是正常的
        assert_equal(info, 1)
        # 设置浮点数误差的绝对容忍度
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        # 断言近似相等性，检查分解结果的正确性
        assert_allclose(A[piv-1][:, piv-1], L @ L.conj().T, rtol=0., atol=atol)


def test_geequ():
    desired_real = np.array([[0.6250, 1.0000, 0.0393, -0.4269],
                             [1.0000, -0.5619, -1.0000, -1.0000],
                             [0.5874, -1.0000, -0.0596, -0.5341],
                             [-1.0000, -0.5946, -0.0294, 0.9957]])
    # 定义复杂的期望值矩阵，包含复数元素
    desired_cplx = np.array([[-0.2816+0.5359*1j,
                              0.0812+0.9188*1j,
                              -0.7439-0.2561*1j],
                             [-0.3562-0.2954*1j,
                              0.9566-0.0434*1j,
                              -0.0174+0.1555*1j],
                             [0.8607+0.1393*1j,
                              -0.2759+0.7241*1j,
                              -0.1642-0.1365*1j]])

    # 遍历数据类型列表 DTYPES 中的索引和数据类型
    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            # 使用 NAG 文档中的示例创建实数矩阵 A
            A = np.array([[1.80e+10, 2.88e+10, 2.05e+00, -8.90e+09],
                          [5.25e+00, -2.95e+00, -9.50e-09, -3.80e+00],
                          [1.58e+00, -2.69e+00, -2.90e-10, -1.04e+00],
                          [-1.11e+00, -6.60e-01, -5.90e-11, 8.00e-01]])
            # 将矩阵 A 转换为指定数据类型 dtype
            A = A.astype(dtype)
        else:
            # 使用 NAG 文档中的示例创建复数矩阵 A
            A = np.array([[-1.34e+00, 0.28e+10, -6.39e+00],
                          [-1.70e+00, 3.31e+10, -0.15e+00],
                          [2.41e-10, -0.56e+00, -0.83e-10]], dtype=dtype)
            # 添加虚数部分到复数矩阵 A
            A += np.array([[2.55e+00, 3.17e+10, -2.20e+00],
                           [-1.41e+00, -0.15e+10, 1.34e+00],
                           [0.39e-10, 1.47e+00, -0.69e-10]])*1j

            # 将矩阵 A 转换为指定数据类型 dtype
            A = A.astype(dtype)

        # 获取 LAPACK 函数 'geequ'，使用指定的数据类型 dtype
        geequ = get_lapack_funcs('geequ', dtype=dtype)
        # 调用 geequ 函数计算矩阵 A 的一些指标
        r, c, rowcnd, colcnd, amax, info = geequ(A)

        if ind < 2:
            # 对于前两个数据类型，断言期望的实数结果与计算结果的近似程度
            assert_allclose(desired_real.astype(dtype), r[:, None]*A*c,
                            rtol=0, atol=1e-4)
        else:
            # 对于其他数据类型，断言期望的复数结果与计算结果的近似程度
            assert_allclose(desired_cplx.astype(dtype), r[:, None]*A*c,
                            rtol=0, atol=1e-4)
def test_syequb():
    # 创建期望的 s 对数值，这是一个 numpy 数组
    desired_log2s = np.array([0, 0, 0, 0, 0, 0, -1, -1, -2, -3])

    # 对于每种数据类型，在指定的 DTYPES 列表中进行循环
    for ind, dtype in enumerate(DTYPES):
        # 创建一个 10x10 的单位矩阵 A，使用指定的数据类型
        A = np.eye(10, dtype=dtype)
        # 根据索引值 ind 来决定 alpha 的值，如果 ind 小于 2，则 alpha 是实数 1.0，否则是复数 1.j
        alpha = dtype(1. if ind < 2 else 1.j)
        # 创建长度为 10 的数组 d，包含 alpha 乘以 2 的负幂次方值
        d = np.array([alpha * 2.**x for x in range(-5, 5)], dtype=dtype)
        # 将对角线上的元素用 d 的旋转90度的对角矩阵进行更新
        A += np.rot90(np.diag(d))

        # 获取 lapack 中的 syequb 函数，针对当前数据类型 dtype
        syequb = get_lapack_funcs('syequb', dtype=dtype)
        # 调用 syequb 函数计算 A 的特征值、条件数、最大绝对值、返回码
        s, scond, amax, info = syequb(A)

        # 断言 s 的对数转换为整数应该等于预期的对数值
        assert_equal(np.log2(s).astype(int), desired_log2s)


@pytest.mark.skipif(True,
                    reason="Failing on some OpenBLAS version, see gh-12276")
def test_heequb():
    # 由于 zheequb 在 LAPACK 3.9.0 及以下版本存在 bug
    # 参考 LAPACK gh-61 和 gh-408
    # 因此 zheequb 测试根据情况进行自定义，避免工作缩放问题
    A = np.diag([2]*5 + [1002]*5) + np.diag(np.ones(9), k=1)*1j
    # 调用 lapack 中的 zheequb 函数计算 A 的特征值、条件数、最大绝对值、返回码
    s, scond, amax, info = lapack.zheequb(A)
    # 断言返回码应为 0
    assert_equal(info, 0)
    # 断言 s 的对数应接近指定的值数组
    assert_allclose(np.log2(s), [0., -1.]*2 + [0.] + [-4]*5)

    A = np.diag(2**np.abs(np.arange(-5, 6)) + 0j)
    A[5, 5] = 1024
    A[5, 0] = 16j
    # 调用 lapack 中的 cheequb 函数计算 A 的特征值、条件数、最大绝对值、返回码
    s, scond, amax, info = lapack.cheequb(A.astype(np.complex64), lower=1)
    # 断言返回码应为 0
    assert_equal(info, 0)
    # 断言 s 的对数应接近指定的值数组
    assert_allclose(np.log2(s), [-2, -1, -1, 0, 0, -5, 0, -1, -1, -2, -2])


def test_getc2_gesc2():
    np.random.seed(42)
    n = 10
    desired_real = np.random.rand(n)
    desired_cplx = np.random.rand(n) + np.random.rand(n)*1j

    # 对于每种数据类型，在指定的 DTYPES 列表中进行循环
    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            # 创建一个随机的 n x n 实数矩阵 A
            A = np.random.rand(n, n)
            A = A.astype(dtype)
            # 计算 b，即 A 乘以期望的实数数组 desired_real
            b = A @ desired_real
            b = b.astype(dtype)
        else:
            # 创建一个随机的 n x n 复数矩阵 A
            A = np.random.rand(n, n) + np.random.rand(n, n)*1j
            A = A.astype(dtype)
            # 计算 b，即 A 乘以期望的复数数组 desired_cplx
            b = A @ desired_cplx
            b = b.astype(dtype)

        # 获取 lapack 中的 getc2 和 gesc2 函数，针对当前数据类型 dtype
        getc2 = get_lapack_funcs('getc2', dtype=dtype)
        gesc2 = get_lapack_funcs('gesc2', dtype=dtype)
        # 调用 getc2 函数，分解 A 并返回 LU 分解、置换向量、返回码
        lu, ipiv, jpiv, info = getc2(A, overwrite_a=0)
        # 调用 gesc2 函数，求解 LU 分解得到的方程组，并返回解 x 和缩放因子 scale
        x, scale = gesc2(lu, b, ipiv, jpiv, overwrite_rhs=0)

        if ind < 2:
            # 断言实数解 x/scale 应接近期望的实数数组 desired_real
            assert_array_almost_equal(desired_real.astype(dtype),
                                      x/scale, decimal=4)
        else:
            # 断言复数解 x/scale 应接近期望的复数数组 desired_cplx
            assert_array_almost_equal(desired_cplx.astype(dtype),
                                      x/scale, decimal=4)


@pytest.mark.parametrize('size', [(6, 5), (5, 5)])
@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('joba', range(6))  # 'C', 'E', 'F', 'G', 'A', 'R'
@pytest.mark.parametrize('jobu', range(4))  # 'U', 'F', 'W', 'N'
@pytest.mark.parametrize('jobv', range(4))  # 'V', 'J', 'W', 'N'
@pytest.mark.parametrize('jobr', [0, 1])
@pytest.mark.parametrize('jobp', [0, 1])
def test_gejsv_general(size, dtype, joba, jobu, jobv, jobr, jobp, jobt=0):
    """Test the lapack routine ?gejsv.

    This function tests that a singular value decomposition can be performed
    on the random M-by-N matrix A. The test performs the SVD using ?gejsv
    then performs the following checks:
    ```
    # 在随机的 MxN 矩阵 A 上测试 lapack 函数 ?gejsv
    # 确保可以执行奇异值分解（SVD），然后进行以下检查
    ```    
    # 设定随机数种子为42，确保结果可重复性
    seed(42)
    
    # 定义一些后续使用的常量：
    m, n = size  # 获取 size 元组中的两个值，分别赋给 m 和 n
    atol = 100 * np.finfo(dtype).eps  # 根据数据类型的机器精度计算绝对容差
    A = generate_random_dtype_array(size, dtype)  # 生成指定大小和数据类型的随机数组 A
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)  # 获取指定数据类型的 LAPACK 函数 gejsv
    
    # 设置检查无效 jobu 和 jobv 组合的条件
    lsvec = jobu < 2  # 是否计算左奇异向量
    rsvec = jobv < 2  # 是否计算右奇异向量
    l2tran = (jobt == 1) and (m == n)  # 是否需要进行 L2 转置操作
    is_complex = np.iscomplexobj(A)  # A 是否包含复数元素
    
    # 检查无效的实数 jobv 组合
    invalid_real_jobv = (jobv == 1) and (not lsvec) and (not is_complex)
    # 检查无效的复数 jobu 组合
    invalid_cplx_jobu = (jobu == 2) and not (rsvec and l2tran) and is_complex
    # 检查无效的复数 jobv 组合
    invalid_cplx_jobv = (jobv == 2) and not (lsvec and l2tran) and is_complex
    
    # 根据无效组合设置退出状态
    if invalid_cplx_jobu:
        exit_status = -2  # 设置退出状态为 -2，表示无效的复数 jobu 组合
    elif invalid_real_jobv or invalid_cplx_jobv:
        exit_status = -3  # 设置退出状态为 -3，表示无效的实数 jobv 或复数 jobv 组合
    else:
        exit_status = 0  # 设置退出状态为 0，表示一切正常
    
    # 如果需要计算左奇异向量且同时需要计算右奇异向量，则断言 gejsv 函数会引发异常
    if (jobu > 1) and (jobv == 1):
        assert_raises(Exception, gejsv, A, joba, jobu, jobv, jobr, jobt, jobp)
    # 调用 gejsv 函数，计算广义奇异值分解，返回结果包括奇异值、左右奇异向量等
    sva, u, v, work, iwork, info = gejsv(A,
                                         joba=joba,
                                         jobu=jobu,
                                         jobv=jobv,
                                         jobr=jobr,
                                         jobt=jobt,
                                         jobp=jobp)

    # 检查 gejsv 函数是否成功完成，返回的 info 应与预期的 exit_status 相等
    assert_equal(info, exit_status)

    # 如果 exit_status 是非零值，则说明输入的 job 组合无效，不进行计算
    if not exit_status:

        # 检查返回的奇异值是否正确
        sigma = (work[0] / work[1]) * sva[:n]
        assert_allclose(sigma, svd(A, compute_uv=False), atol=atol)

        if jobu == 1:
            # 如果 jobu = 1，表示需要左奇异向量，截取前 n 列作为有效部分
            u = u[:, :n]

        # 如果同时需要左奇异向量和右奇异向量
        if lsvec and rsvec:
            # 检查计算的近似重构是否与原始矩阵 A 非常接近
            assert_allclose(u @ np.diag(sigma) @ v.conj().T, A, atol=atol)
        if lsvec:
            # 如果需要左奇异向量，检查其是否满足单位正交性
            assert_allclose(u.conj().T @ u, np.identity(n), atol=atol)
        if rsvec:
            # 如果需要右奇异向量，检查其是否满足单位正交性
            assert_allclose(v.conj().T @ v, np.identity(n), atol=atol)

        # 检查工作数组 iwork 的第一项是否与 A 的秩相等
        assert_equal(iwork[0], np.linalg.matrix_rank(A))
        # 检查奇异值向量 sigma 中非零元素的数量是否与 A 中的奇异值数量相等
        assert_equal(iwork[1], np.count_nonzero(sigma))
        # 如果 iwork 的第三项非零，表示数据的精度可能不足，这在测试中不应发生
        assert_equal(iwork[2], 0)
@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_gejsv_edge_arguments(dtype):
    """Test edge arguments return expected status"""
    # 获取指定数据类型的 LAPACK 函数 gejsv
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)

    # scalar A
    # 调用 gejsv 函数，传入标量值 1.0 作为参数
    sva, u, v, work, iwork, info = gejsv(1.)
    # 断言返回的 info 值为 0
    assert_equal(info, 0)
    # 断言返回的 u 和 v 的形状为 (1, 1)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    # 断言返回的 sva 为包含单个元素 1.0 的 numpy 数组，数据类型为 dtype
    assert_equal(sva, np.array([1.], dtype=dtype))

    # 1d A
    A = np.ones((1,), dtype=dtype)
    # 调用 gejsv 函数，传入 1 维数组 A 作为参数
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    assert_equal(sva, np.array([1.], dtype=dtype))

    # 2d empty A
    A = np.ones((1, 0), dtype=dtype)
    # 调用 gejsv 函数，传入 2 维空数组 A 作为参数
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 0))
    assert_equal(v.shape, (1, 0))
    assert_equal(sva, np.array([], dtype=dtype))

    # make sure "overwrite_a" is respected - user reported in gh-13191
    A = np.sin(np.arange(100).reshape(10, 10)).astype(dtype)
    A = np.asfortranarray(A + A.T)  # make it symmetric and column major
    Ac = A.copy('A')
    _ = gejsv(A)
    # 断言经过 gejsv 处理后 A 的值保持不变
    assert_allclose(A, Ac)


@pytest.mark.parametrize(('kwargs'),
                         ({'joba': 9},
                          {'jobu': 9},
                          {'jobv': 9},
                          {'jobr': 9},
                          {'jobt': 9},
                          {'jobp': 9})
                         )
def test_gejsv_invalid_job_arguments(kwargs):
    """Test invalid job arguments raise an Exception"""
    A = np.ones((2, 2), dtype=float)
    gejsv = get_lapack_funcs('gejsv', dtype=float)
    # 断言传入具有无效 job 参数的 kwargs 会引发异常
    assert_raises(Exception, gejsv, A, **kwargs)


@pytest.mark.parametrize("A,sva_expect,u_expect,v_expect",
                         [(np.array([[2.27, -1.54, 1.15, -1.94],
                                     [0.28, -1.67, 0.94, -0.78],
                                     [-0.48, -3.09, 0.99, -0.21],
                                     [1.07, 1.22, 0.79, 0.63],
                                     [-2.35, 2.93, -1.45, 2.30],
                                     [0.62, -7.39, 1.03, -2.57]]),
                           np.array([9.9966, 3.6831, 1.3569, 0.5000]),
                           np.array([[0.2774, -0.6003, -0.1277, 0.1323],
                                     [0.2020, -0.0301, 0.2805, 0.7034],
                                     [0.2918, 0.3348, 0.6453, 0.1906],
                                     [-0.0938, -0.3699, 0.6781, -0.5399],
                                     [-0.4213, 0.5266, 0.0413, -0.0575],
                                     [0.7816, 0.3353, -0.1645, -0.3957]]),
                           np.array([[0.1921, -0.8030, 0.0041, -0.5642],
                                     [-0.8794, -0.3926, -0.0752, 0.2587],
                                     [0.2140, -0.2980, 0.7827, 0.5027],
                                     [-0.3795, 0.3351, 0.6178, -0.6017]]))])
def test_gejsv_NAG(A, sva_expect, u_expect, v_expect):
    """Test gejsv with NAG-like example matrices"""
    """
    This test implements the example found in the NAG manual, f08khf.
    An example was not found for the complex case.
    """
    # NAG manual provides accuracy up to 4 decimals
    atol = 1e-4  # 设置允许误差的阈值为 0.0001
    
    # 获取 LAPACK 函数库中的 gejsv 函数，并根据数组 A 的数据类型选择适当的函数
    gejsv = get_lapack_funcs('gejsv', dtype=A.dtype)
    
    # 调用 gejsv 函数，对矩阵 A 进行奇异值分解，并返回结果 sva, u, v, work, iwork, info
    sva, u, v, work, iwork, info = gejsv(A)
    
    # 使用 assert_allclose 函数检查计算得到的 sva 是否与预期的 sva_expect 在给定的阈值 atol 内接近
    assert_allclose(sva_expect, sva, atol=atol)
    
    # 使用 assert_allclose 函数检查计算得到的 u 是否与预期的 u_expect 在给定的阈值 atol 内接近
    assert_allclose(u_expect, u, atol=atol)
    
    # 使用 assert_allclose 函数检查计算得到的 v 是否与预期的 v_expect 在给定的阈值 atol 内接近
    assert_allclose(v_expect, v, atol=atol)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gttrf_gttrs(dtype):
    # 使用参数化测试，对每种数据类型进行测试
    # 测试使用 ?gttrf 和 ?gttrs 来解决随机系统，测试 ?gttrf 的输出定义了 LU 矩阵，输入参数未被修改，转置选项功能正确，不兼容的矩阵形状会引发错误，奇异矩阵返回非零信息。

    seed(42)
    n = 10
    atol = 100 * np.finfo(dtype).eps

    # 根据数据类型创建矩阵
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    dl = generate_random_dtype_array((n-1,), dtype=dtype)

    diag_cpy = [dl.copy(), d.copy(), du.copy()]

    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    x = np.random.rand(n)
    b = A @ x

    gttrf, gttrs = get_lapack_funcs(('gttrf', 'gttrs'), dtype=dtype)

    _dl, _d, _du, du2, ipiv, info = gttrf(dl, d, du)
    # 测试确保 ?gttrf 的输入未被修改
    assert_array_equal(dl, diag_cpy[0])
    assert_array_equal(d, diag_cpy[1])
    assert_array_equal(du, diag_cpy[2])

    # 从 ?gttrf 返回值生成 L 和 U 因子
    # L/U 是通过构造（初始和最终）下三角/上三角矩阵
    U = np.diag(_d, 0) + np.diag(_du, 1) + np.diag(du2, 2)
    L = np.eye(n, dtype=dtype)

    for i, m in enumerate(_dl):
        # L 以分解形式给出
        piv = ipiv[i] - 1
        # 右乘置换矩阵
        L[:, [i, piv]] = L[:, [piv, i]]
        # 右乘 Li，单位矩阵的秩一修改
        L[:, i] += L[:, i+1]*m

    # 最后一次置换
    i, piv = -1, ipiv[-1] - 1
    # 右乘最终置换矩阵
    L[:, [i, piv]] = L[:, [piv, i]]

    # 检查 ?gttrf 的输出是否定义了 A 的 LU 分解
    assert_allclose(A, L @ U, atol=atol)

    b_cpy = b.copy()
    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b)
    # 测试 ?gttrs 的输入未被修改
    assert_array_equal(b, b_cpy)
    # 测试 ?gttrs 的结果是否与预期输入匹配
    assert_allclose(x, x_gttrs, atol=atol)

    # 测试 ?gttrf 和 ?gttrs 是否支持转置选项
    if dtype in REAL_DTYPES:
        trans = "T"
        b_trans = A.T @ x
    else:
        trans = "C"
        b_trans = A.conj().T @ x

    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b_trans, trans=trans)
    assert_allclose(x, x_gttrs, atol=atol)

    # 测试不兼容的矩阵形状是否会引发 ValueError
    with assert_raises(ValueError):
        gttrf(dl[:-1], d, du)
    with assert_raises(ValueError):
        gttrf(dl, d[:-1], du)
    with assert_raises(ValueError):
        gttrf(dl, d, du[:-1])

    # 测试大小为 n=2 的矩阵是否会引发异常
    with assert_raises(Exception):
        gttrf(dl[0], d[:1], du[0])
    # 将矩阵 du 的第一行置为零，测试是否通过 info 报错
    du[0] = 0
    # 将矩阵 d 的第一行置为零，测试是否通过 info 报错
    d[0] = 0
    # 调用 gttrf 函数进行矩阵三对角分解，返回结果分别赋值给 __dl, __d, __du, _du2, _ipiv, _info
    __dl, __d, __du, _du2, _ipiv, _info = gttrf(dl, d, du)
    # 使用 np.testing.assert_ 进行断言，检查 __d 数组中的特定位置是否为零，以验证是否触发了异常
    np.testing.assert_(__d[info - 1] == 0, (f"?gttrf: _d[info-1] is {__d[info - 1]},"
                                            " not the illegal value :0."))
@pytest.mark.parametrize("du, d, dl, du_exp, d_exp, du2_exp, ipiv_exp, b, x",
                         [(np.array([2.1, -1.0, 1.9, 8.0]),  # Upper band of the tridiagonal matrix
                           np.array([3.0, 2.3, -5.0, -.9, 7.1]),  # Diagonal of the tridiagonal matrix
                           np.array([3.4, 3.6, 7.0, -6.0]),  # Lower band of the tridiagonal matrix
                           np.array([2.3, -5, -.9, 7.1]),  # Expected upper band after factorization
                           np.array([3.4, 3.6, 7, -6, -1.015373]),  # Expected diagonal after factorization
                           np.array([-1, 1.9, 8]),  # Expected secondary diagonal after factorization
                           np.array([2, 3, 4, 5, 5]),  # Pivot indices after factorization
                           np.array([[2.7, 6.6],  # Right-hand side matrix for the system of equations
                                     [-0.5, 10.8],
                                     [2.6, -3.2],
                                     [0.6, -11.2],
                                     [2.7, 19.1]
                                     ]),
                           np.array([[-4, 5],  # Expected solution matrix
                                     [7, -4],
                                     [3, -3],
                                     [-4, -2],
                                     [-3, 1]])),
                          (
                           np.array([2 - 1j, 2 + 1j, -1 + 1j, 1 - 1j]),  # Upper band for complex matrix
                           np.array([-1.3 + 1.3j, -1.3 + 1.3j,  # Diagonal for complex matrix
                                     -1.3 + 3.3j, - .3 + 4.3j,
                                     -3.3 + 1.3j]),
                           np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j]),  # Lower band for complex matrix
                           np.array([-1.3 + 1.3j, -1.3 + 3.3j,  # Expected upper band after factorization (complex)
                                     -0.3 + 4.3j, -3.3 + 1.3j]),
                           np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j,  # Expected diagonal after factorization (complex)
                                     -1.3399 + 0.2875j]),
                           np.array([2 + 1j, -1 + 1j, 1 - 1j]),  # Expected secondary diagonal after factorization (complex)
                           np.array([2, 3, 4, 5, 5]),  # Pivot indices after factorization (complex)
                           np.array([[2.4 - 5j, 2.7 + 6.9j],  # Right-hand side matrix for complex equations
                                     [3.4 + 18.2j, - 6.9 - 5.3j],
                                     [-14.7 + 9.7j, - 6 - .6j],
                                     [31.9 - 7.7j, -3.9 + 9.3j],
                                     [-1 + 1.6j, -3 + 12.2j]]),
                           np.array([[1 + 1j, 2 - 1j],  # Expected solution matrix for complex equations
                                     [3 - 1j, 1 + 2j],
                                     [4 + 5j, -1 + 1j],
                                     [-1 - 2j, 2 + 1j],
                                     [1 - 1j, 2 - 2j]])
                            )])
def test_gttrf_gttrs_NAG_f07cdf_f07cef_f07crf_f07csf(du, d, dl, du_exp, d_exp,
                                                     du2_exp, ipiv_exp, b, x):
    """
    Test function for verifying consistency with NAG Library Manual Mark 26
    example problems: f07cdf and f07cef (real)
    examples: f07crf and f07csf (complex)
    (Links may expire, so search for "NAG Library Manual Mark 26" online)
    """
    # Obtain LAPACK functions for tridiagonal factorization and solution
    gttrf, gttrs = get_lapack_funcs(('gttrf', "gttrs"), (du[0], du[0]))
    # 调用名为gttrf的函数，进行三对角矩阵分解，返回_dl, _d, _du, du2, ipiv, info
    _dl, _d, _du, du2, ipiv, info = gttrf(dl, d, du)
    
    # 使用assert_allclose函数检查du2与du2_exp是否近似相等
    assert_allclose(du2, du2_exp)
    
    # 使用assert_allclose函数检查_d与du_exp是否近似相等，给定允许误差atol=1e-4
    assert_allclose(_du, du_exp, atol=1e-4)  # NAG examples provide 4 decimals.
    
    # 使用assert_allclose函数检查_ipiv与ipiv_exp是否近似相等
    assert_allclose(ipiv, ipiv_exp)
    
    # 使用_gttrs函数求解三对角线性方程组，返回解x_gttrs和info
    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b)
    
    # 使用assert_allclose函数检查x_gttrs与x是否近似相等
    assert_allclose(x_gttrs, x)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3, 7), (7, 3), (2**18, 2**18)])
# 使用参数化装饰器指定多个数据类型和形状的测试用例
def test_geqrfp_lwork(dtype, shape):
    # 获取 LAPACK 函数 geqrfp_lwork 对应的函数对象
    geqrfp_lwork = get_lapack_funcs(('geqrfp_lwork'), dtype=dtype)
    m, n = shape
    # 调用 geqrfp_lwork 函数计算所需的工作空间大小和返回信息
    lwork, info = geqrfp_lwork(m=m, n=n)
    # 断言 LAPACK 函数执行结果的信息码为 0
    assert_equal(info, 0)


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
# 使用参数化装饰器指定多个实数数据类型和复数数据类型的测试用例
def test_pttrf_pttrs(ddtype, dtype):
    seed(42)
    # 设置适用于指定数据类型的测试容差值
    atol = 100*np.finfo(dtype).eps
    # 矩阵 A 对角线长度为 n
    n = 10
    # 生成大小为 n 的随机实数或复数数组 d 和 e
    d = generate_random_dtype_array((n,), ddtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)

    # 组装 d 和 e 成矩阵 A
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    # 存储对角线副本，用于后续验证
    diag_cpy = [d.copy(), e.copy()]

    # 获取 LAPACK 函数 pttrf 对应的函数对象
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)

    _d, _e, info = pttrf(d, e)
    # 断言 ?pttrf 函数的输入参数未被修改
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    # 断言 ?pttrf 函数的执行结果信息码为 0
    assert_equal(info, 0, err_msg=f"pttrf: info = {info}, should be 0")

    # 验证 pttrf 函数的因子能够重组成矩阵 A
    L = np.diag(_e, -1) + np.diag(np.ones(n))
    D = np.diag(_d)

    assert_allclose(A, L@D@L.conjugate().T, atol=atol)

    # 生成随机解 x
    x = generate_random_dtype_array((n,), dtype)
    # 确定相应的 b 以获取解 x
    b = A@x

    # 通过 pttrs 函数确定 _x
    pttrs = get_lapack_funcs('pttrs', dtype=dtype)
    _x, info = pttrs(_d, _e.conj(), b)
    # 断言 pttrs 函数的执行结果信息码为 0
    assert_equal(info, 0, err_msg=f"pttrs: info = {info}, should be 0")

    # 断言 pttrs 函数的解 _x 与期望解 x 相匹配
    assert_allclose(x, _x, atol=atol)


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
# 使用参数化装饰器指定多个实数数据类型和复数数据类型的测试用例
def test_pttrf_pttrs_errors_incompatible_shape(ddtype, dtype):
    n = 10
    # 获取 LAPACK 函数 pttrf 对应的函数对象
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    d = generate_random_dtype_array((n,), ddtype) + 2
    e = generate_random_dtype_array((n-1,), dtype)
    # 断言当矩阵形状不兼容时，会触发 ValueError 异常
    assert_raises(ValueError, pttrf, d[:-1], e)
    assert_raises(ValueError, pttrf, d, e[:-1])


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
# 使用参数化装饰器指定多个实数数据类型和复数数据类型的测试用例
def test_pttrf_pttrs_errors_singular_nonSPD(ddtype, dtype):
    n = 10
    # 获取 LAPACK 函数 pttrf 对应的函数对象
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    d = generate_random_dtype_array((n,), ddtype) + 2
    e = generate_random_dtype_array((n-1,), dtype)
    # 设置为奇异矩阵（所有元素为零的行）以验证异常情况
    d[0] = 0
    e[0] = 0
    _d, _e, info = pttrf(d, e)
    # 断言检查：验证 _d[info - 1] 的值是否等于 0
    assert_equal(_d[info - 1], 0,
                 f"?pttrf: _d[info-1] is {_d[info - 1]}, not the illegal value :0.")

    # 使用非对称正定矩阵进行测试
    # 生成一个随机数据类型数组 d，长度为 n，数据类型为 ddtype
    d = generate_random_dtype_array((n,), ddtype)
    # 对 d, e 进行 Cholesky 分解，返回分解后的结果 _d, _e，以及 info 表示状态信息
    _d, _e, info = pttrf(d, e)
    # 断言检查：验证 pttrf 函数是否在处理非对称正定矩阵时失败（info != 0 表示失败）
    assert_(info != 0, "?pttrf should fail with non-spd matrix, but didn't")
@pytest.mark.parametrize(("d, e, d_expect, e_expect, b, x_expect"), [
                         # 第一个参数组
                         (np.array([4, 10, 29, 25, 5]),  # d 数组
                          np.array([-2, -6, 15, 8]),    # e 数组
                          np.array([4, 9, 25, 16, 1]),  # d_expect 数组
                          np.array([-.5, -.6667, .6, .5]),  # e_expect 数组
                          np.array([[6, 10], [9, 4], [2, 9], [14, 65], [7, 23]]),  # b 数组
                          np.array([[2.5, 2], [2, -1], [1, -3], [-1, 6], [3, -5]])  # x_expect 数组
                          ), (
                          # 第二个参数组
                          np.array([16, 41, 46, 21]),    # d 数组
                          np.array([16 + 16j, 18 - 9j, 1 - 4j]),  # e 数组
                          np.array([16, 9, 1, 4]),       # d_expect 数组
                          np.array([1+1j, 2-1j, 1-4j]),  # e_expect 数组
                          np.array([[64+16j, -16-32j], [93+62j, 61-66j],
                                    [78-80j, 71-74j], [14-27j, 35+15j]]),  # b 数组
                          np.array([[2+1j, -3-2j], [1+1j, 1+1j], [1-2j, 1-2j],
                                    [1-1j, 2+1j]])  # x_expect 数组
                         )])
def test_pttrf_pttrs_NAG(d, e, d_expect, e_expect, b, x_expect):
    # 用于确保包装器与 NAG Manual Mark 26 中的示例问题一致的测试
    # 示例问题：f07jdf 和 f07jef (实数)
    # 示例：f07jrf 和 f07csf (复数)
    # NAG 示例提供 4 位小数精度。
    # （链接会过期，请搜索在线查找 "NAG Library Manual Mark 26"）

    atol = 1e-4  # 允许的绝对误差
    pttrf = get_lapack_funcs('pttrf', dtype=e[0])
    _d, _e, info = pttrf(d, e)  # 执行 pttrf 分解
    assert_allclose(_d, d_expect, atol=atol)  # 断言 _d 与 d_expect 接近
    assert_allclose(_e, e_expect, atol=atol)  # 断言 _e 与 e_expect 接近

    pttrs = get_lapack_funcs('pttrs', dtype=e[0])
    _x, info = pttrs(_d, _e.conj(), b)  # 解线性方程组
    assert_allclose(_x, x_expect, atol=atol)  # 断言 _x 与 x_expect 接近

    # 还测试选项 `lower`
    if e.dtype in COMPLEX_DTYPES:
        _x, info = pttrs(_d, _e, b, lower=1)  # 使用 lower 选项解线性方程组
        assert_allclose(_x, x_expect, atol=atol)


def pteqr_get_d_e_A_z(dtype, realtype, n, compute_z):
    # ?pteqr 测试中用于构建参数的函数
    # 返回 (d, e, A, z) 的元组
    if compute_z == 1:
        # 从 Q**T * tri * Q = A 构建 Hermitian A，通过创建 Q 和 tri
        A_eig = generate_random_dtype_array((n, n), dtype)  # 生成随机的 A_eig 数组
        A_eig = A_eig + np.diag(np.zeros(n) + 4*n)  # 对角线加上 4*n
        A_eig = (A_eig + A_eig.conj().T) / 2  # 取复共轭转置的平均，确保 Hermitian 性质
        # 获取右特征向量（正交的）
        vr = eigh(A_eig)[1]
        # 创建三对角矩阵
        d = generate_random_dtype_array((n,), realtype) + 4  # 生成随机的 d 数组
        e = generate_random_dtype_array((n-1,), realtype)  # 生成随机的 e 数组
        tri = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)  # 构建三对角矩阵 tri
        # 使用这些因子构建 A，即 sytrd 会得到的 (Q**T * tri * Q = A)
        A = vr @ tri @ vr.conj().T
        # vr 是正交的
        z = vr
    else:
        # 根据 LAPACK 文档，d 和 e 总是实数。
        # 生成长度为 n 的随机数组 d，元素类型为 realtype
        d = generate_random_dtype_array((n,), realtype)
        # 生成长度为 n-1 的随机数组 e，元素类型为 realtype
        e = generate_random_dtype_array((n-1,), realtype)

        # 使得 A 成为对称正定矩阵（Symmetric Positive Definite, SPD）
        # 将 d 中每个元素增加 4
        d = d + 4
        # 构造对称正定矩阵 A
        A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        # z 与 A 结构相同，但不一定是对称正定的
        z = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
    # 返回生成的数组和矩阵 (d, e, A, z)
    return (d, e, A, z)
# 使用 pytest 的 parametrize 装饰器，为 test_pteqr 函数传入参数组合，dtype 和 realtype 由 DTYPES 和 REAL_DTYPES 组成
@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr(dtype, realtype, compute_z):
    '''
    Tests the ?pteqr lapack routine for all dtypes and compute_z parameters.
    It generates random SPD matrix diagonals d and e, and then confirms
    correct eigenvalues with scipy.linalg.eig. With applicable compute_z=2 it
    tests that z can reform A.
    '''
    # 设置随机数生成种子
    seed(42)
    # 计算每个数据类型的浮点数精度的 atol
    atol = 1000*np.finfo(dtype).eps
    # 获取指定数据类型的 pteqr LAPACK 函数
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)

    # 设置矩阵维度
    n = 10

    # 调用辅助函数 pteqr_get_d_e_A_z 获取随机生成的对角线向量 d、次对角线向量 e、矩阵 A 和矩阵 z
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)

    # 调用 pteqr 函数进行计算，返回更新后的对角线向量 d、次对角线向量 e、矩阵 z 和返回码 info
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    # 使用 assert_equal 断言，检查返回码 info 是否为 0
    assert_equal(info, 0, f"info = {info}, should be 0.")

    # 使用 eigh 函数计算矩阵 A 的特征值，并使用 assert_allclose 断言，检查计算得到的特征值与 d_pteqr 的排序后是否一致
    assert_allclose(np.sort(eigh(A)[0]), np.sort(d_pteqr), atol=atol)

    if compute_z:
        # 使用 assert_allclose 断言，检查 z_pteqr 是否正交
        assert_allclose(z_pteqr @ np.conj(z_pteqr).T, np.identity(n),
                        atol=atol)
        # 使用 assert_allclose 断言，检查 z_pteqr 是否能够重新组合成矩阵 A
        assert_allclose(z_pteqr @ np.diag(d_pteqr) @ np.conj(z_pteqr).T,
                        A, atol=atol)


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_error_non_spd(dtype, realtype, compute_z):
    # 设置随机数生成种子
    seed(42)
    # 获取指定数据类型的 pteqr LAPACK 函数
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)

    # 设置矩阵维度
    n = 10
    # 调用辅助函数 pteqr_get_d_e_A_z 获取随机生成的对角线向量 d、次对角线向量 e、矩阵 A 和矩阵 z
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)

    # 使用 assert_raises 断言，检查当矩阵非正定时是否抛出 ValueError
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d - 4, e, z=z, compute_z=compute_z)
    assert info > 0


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_raise_error_wrong_shape(dtype, realtype, compute_z):
    # 设置随机数生成种子
    seed(42)
    # 获取指定数据类型的 pteqr LAPACK 函数
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)
    # 设置矩阵维度
    n = 10
    # 调用辅助函数 pteqr_get_d_e_A_z 获取随机生成的对角线向量 d、次对角线向量 e、矩阵 A 和矩阵 z
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    # 使用 assert_raises 断言，检查当数组大小不匹配时是否抛出 ValueError
    assert_raises(ValueError, pteqr, d[:-1], e, z=z, compute_z=compute_z)
    assert_raises(ValueError, pteqr, d, e[:-1], z=z, compute_z=compute_z)
    if compute_z:
        assert_raises(ValueError, pteqr, d, e, z=z[:-1], compute_z=compute_z)


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_error_singular(dtype, realtype, compute_z):
    # 设置随机数生成种子
    seed(42)
    # 获取指定数据类型的 pteqr LAPACK 函数
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)
    # 设置矩阵维度
    n = 10
    # 调用辅助函数 pteqr_get_d_e_A_z 获取随机生成的对角线向量 d、次对角线向量 e、矩阵 A 和矩阵 z
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    # 将矩阵 A 变为奇异矩阵
    d[0] = 0
    e[0] = 0
    # 使用 assert_raises 断言，检查当矩阵奇异时是否抛出 ValueError
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d, e, z=z, compute_z=compute_z)
    assert info > 0
@pytest.mark.parametrize("compute_z,d,e,d_expect,z_expect",
                         [(2,  # "I"
                           np.array([4.16, 5.25, 1.09, .62]),  # Array containing eigenvalues d
                           np.array([3.17, -.97, .55]),  # Array containing off-diagonal elements e
                           np.array([8.0023, 1.9926, 1.0014, 0.1237]),  # Expected eigenvalues _d
                           np.array([[0.6326, 0.6245, -0.4191, 0.1847],  # Expected eigenvectors _z
                                     [0.7668, -0.4270, 0.4176, -0.2352],
                                     [-0.1082, 0.6071, 0.4594, -0.6393],
                                     [-0.0081, 0.2432, 0.6625, 0.7084]])),
                          ])
def test_pteqr_NAG_f08jgf(compute_z, d, e, d_expect, z_expect):
    '''
    Implements real (f08jgf) example from NAG Manual Mark 26.
    Tests for correct outputs.
    '''
    # the NAG manual has 4 decimals accuracy
    atol = 1e-4
    # Obtain the LAPACK function 'pteqr' appropriate for the dtype of d
    pteqr = get_lapack_funcs(('pteqr'), dtype=d.dtype)

    # Construct the tridiagonal matrix z
    z = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    
    # Call the LAPACK routine pteqr to compute eigenvalues and eigenvectors
    _d, _e, _z, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    
    # Assert that computed eigenvalues _d are close to expected d_expect
    assert_allclose(_d, d_expect, atol=atol)
    # Assert that computed eigenvectors _z are close in absolute value to expected z_expect
    assert_allclose(np.abs(_z), np.abs(z_expect), atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('matrix_size', [(3, 4), (7, 6), (6, 6)])
def test_geqrfp(dtype, matrix_size):
    # Tests for all dytpes, tall, wide, and square matrices.
    # Using the routine with random matrix A, Q and R are obtained and then
    # tested such that R is upper triangular and non-negative on the diagonal,
    # and Q is an orthogonal matrix. Verifies that A=Q@R. It also
    # tests against a matrix that for which the  linalg.qr method returns
    # negative diagonals, and for error messaging.

    # set test tolerance appropriate for dtype
    np.random.seed(42)
    rtol = 250*np.finfo(dtype).eps
    atol = 100*np.finfo(dtype).eps
    
    # Obtain LAPACK function 'geqrfp' appropriate for the dtype of A
    geqrfp = get_lapack_funcs(('geqrfp'), dtype=dtype)
    # Obtain LAPACK function 'orgqr' appropriate for the dtype of A
    gqr = get_lapack_funcs(("orgqr"), dtype=dtype)

    m, n = matrix_size

    # create random matrix of dimensions m x n
    A = generate_random_dtype_array((m, n), dtype=dtype)
    # create qr matrix using geqrfp
    qr_A, tau, info = geqrfp(A)

    # obtain r from the upper triangular area
    r = np.triu(qr_A)

    # obtain q from the orgqr lapack routine
    # based on linalg.qr's extraction strategy of q with orgqr
    if m > n:
        # this adds an extra column to the end of qr_A
        # let qqr be an empty m x m matrix
        qqr = np.zeros((m, m), dtype=dtype)
        # set first n columns of qqr to qr_A
        qqr[:, :n] = qr_A
        # determine q from this qqr
        # note that m is a sufficient for lwork based on LAPACK documentation
        q = gqr(qqr, tau=tau, lwork=m)[0]
    else:
        q = gqr(qr_A[:, :m], tau=tau, lwork=m)[0]

    # test that q and r still make A
    assert_allclose(q@r, A, rtol=rtol)
    # ensure that q is orthogonal (that q @ transposed q is the identity)
    # 确保正交矩阵 q 的转置乘积与单位矩阵非常接近，给定相对和绝对容忍度
    assert_allclose(np.eye(q.shape[0]), q@(q.conj().T), rtol=rtol,
                    atol=atol)
    # 确保矩阵 r 是上三角矩阵，通过将原始 r 与其上三角形式进行比较来实现
    assert_allclose(r, np.triu(r), rtol=rtol)
    # 确保 r 的对角线元素都是正数，对于这个随机解决方案
    assert_(np.all(np.diag(r) > np.zeros(len(np.diag(r)))))
    # 确保 info 变量为零，表示成功完成操作
    assert_(info == 0)

    # 测试确保此函数生成的 r 对角线元素为正数，即使对于返回对角线元素为负数的矩阵，例如使用 scipy.linalg.rq
    A_negative = generate_random_dtype_array((n, m), dtype=dtype) * -1
    r_rq_neg, q_rq_neg = qr(A_negative)
    rq_A_neg, tau_neg, info_neg = geqrfp(A_negative)
    # 断言确保从 linalg.qr 返回的对角线元素中至少有一个是负数，并且 geqrfp 返回的对角线元素全部为正数
    assert_(np.any(np.diag(r_rq_neg) < 0) and
            np.all(np.diag(r) > 0))
def test_geqrfp_errors_with_empty_array():
    # 定义测试函数，验证空数组是否会引发异常
    A_empty = np.array([])  # 创建一个空的 NumPy 数组
    geqrfp = get_lapack_funcs('geqrfp', dtype=A_empty.dtype)  # 获取 LAPACK 函数 'geqrfp'，使用与 A_empty 相同的数据类型
    assert_raises(Exception, geqrfp, A_empty)  # 断言调用 geqrfp 函数时会引发异常


@pytest.mark.parametrize("driver", ['ev', 'evd', 'evr', 'evx'])
@pytest.mark.parametrize("pfx", ['sy', 'he'])
def test_standard_eigh_lworks(pfx, driver):
    n = 1200  # 一个足够大的任意数值
    dtype = REAL_DTYPES if pfx == 'sy' else COMPLEX_DTYPES  # 根据 pfx 的值选择相应的数据类型
    sc_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[0])  # 获取 LAPACK 函数，对称或厄米矩阵的特征值计算
    dz_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[1])  # 获取 LAPACK 函数，复数对称或复厄米矩阵的特征值计算
    try:
        _compute_lwork(sc_dlw, n, lower=1)  # 计算特征值计算所需的工作空间大小（对称/厄米）
        _compute_lwork(dz_dlw, n, lower=1)  # 计算特征值计算所需的工作空间大小（复数对称/复厄米）
    except Exception as e:
        pytest.fail(f"{pfx+driver}_lwork raised unexpected exception: {e}")  # 如果出现异常，测试失败并输出异常信息


@pytest.mark.parametrize("driver", ['gv', 'gvx'])
@pytest.mark.parametrize("pfx", ['sy', 'he'])
def test_generalized_eigh_lworks(pfx, driver):
    n = 1200  # 一个足够大的任意数值
    dtype = REAL_DTYPES if pfx == 'sy' else COMPLEX_DTYPES  # 根据 pfx 的值选择相应的数据类型
    sc_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[0])  # 获取 LAPACK 函数，对称或厄米矩阵的广义特征值计算
    dz_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[1])  # 获取 LAPACK 函数，复数对称或复厄米矩阵的广义特征值计算
    # 应该不会引发任何异常
    try:
        _compute_lwork(sc_dlw, n, uplo="L")  # 计算广义特征值计算所需的工作空间大小（对称/厄米，下三角）
        _compute_lwork(dz_dlw, n, uplo="L")  # 计算广义特征值计算所需的工作空间大小（复数对称/复厄米，下三角）
    except Exception as e:
        pytest.fail(f"{pfx+driver}_lwork raised unexpected exception: {e}")  # 如果出现异常，测试失败并输出异常信息


@pytest.mark.parametrize("dtype_", DTYPES)
@pytest.mark.parametrize("m", [1, 10, 100, 1000])
def test_orcsd_uncsd_lwork(dtype_, m):
    seed(1234)  # 设置随机数种子
    p = randint(0, m)  # 随机生成一个介于 0 到 m 之间的整数 p
    q = m - p  # 计算 q，使得 p + q = m
    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'  # 根据 dtype_ 的类型选择相应的前缀
    dlw = pfx + 'csd_lwork'  # 组合出 LAPACK 函数名称
    lw = get_lapack_funcs(dlw, dtype=dtype_)  # 获取 LAPACK 函数，用于奇异值分解的工作空间计算
    lwval = _compute_lwork(lw, m, p, q)  # 计算工作空间的大小
    lwval = lwval if pfx == 'un' else (lwval,)  # 如果是非厄米矩阵，工作空间大小是一个元组
    assert all([x > 0 for x in lwval])  # 断言工作空间的大小都大于零


@pytest.mark.parametrize("dtype_", DTYPES)
def test_orcsd_uncsd(dtype_):
    m, p, q = 250, 80, 170  # 设置参数

    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'  # 根据 dtype_ 的类型选择相应的前缀
    X = ortho_group.rvs(m) if pfx == 'or' else unitary_group.rvs(m)  # 生成正交或酉矩阵 X

    drv, dlw = get_lapack_funcs((pfx + 'csd', pfx + 'csd_lwork'), dtype=dtype_)  # 获取奇异值分解及其工作空间计算的 LAPACK 函数
    lwval = _compute_lwork(dlw, m, p, q)  # 计算工作空间的大小
    lwvals = {'lwork': lwval} if pfx == 'or' else dict(zip(['lwork',
                                                            'lrwork'], lwval))  # 构建参数字典

    cs11, cs12, cs21, cs22, theta, u1, u2, v1t, v2t, info =\
        drv(X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:], **lwvals)  # 调用奇异值分解 LAPACK 函数

    assert info == 0  # 断言返回信息为 0

    U = block_diag(u1, u2)  # 构建 U 矩阵
    VH = block_diag(v1t, v2t)  # 构建 VH 矩阵
    r = min(min(p, q), min(m-p, m-q))  # 计算 r 的值
    n11 = min(p, q) - r  # 计算 n11 的值
    n12 = min(p, m-q) - r  # 计算 n12 的值
    n21 = min(m-p, q) - r  # 计算 n21 的值
    n22 = min(m-p, m-q) - r  # 计算 n22 的值

    S = np.zeros((m, m), dtype=dtype_)  # 创建零矩阵 S
    one = dtype_(1.)  # 创建数据类型为 dtype_ 的值 1
    for i in range(n11):
        S[i, i] = one  # 设置 S 的对角线元素为 1
    for i in range(n22):
        S[p+i, q+i] = one  # 设置 S 的对角线元素为 1
    for i in range(n12):
        S[i+n11+r, i+n11+r+n21+n22+r] = -one  # 设置 S 的非对角线元素为 -1
    # 循环操作，为矩阵 S 的特定行和列赋值为 one
    for i in range(n21):
        S[p+n22+r+i, n11+r+i] = one

    # 循环操作，设置矩阵 S 的对角线和特定位置的值，使用 np.cos 和 np.sin 函数
    for i in range(r):
        # 设置矩阵 S 对角线上的值为 np.cos(theta[i])
        S[i+n11, i+n11] = np.cos(theta[i])
        # 设置矩阵 S 中的特定位置的值为 np.cos(theta[i])
        S[p+n22+i, i+r+n21+n22] = np.cos(theta[i])

        # 设置矩阵 S 中的特定位置的值为 -np.sin(theta[i])
        S[i+n11, i+n11+n21+n22+r] = -np.sin(theta[i])
        # 设置矩阵 S 中的特定位置的值为 np.sin(theta[i])
        S[p+n22+i, i+n11] = np.sin(theta[i])

    # 计算矩阵 Xc，使用矩阵 U、S、VH 的乘积
    Xc = U @ S @ VH
    # 断言 X 和 Xc 的近似程度，使用 assert_allclose 函数
    assert_allclose(X, Xc, rtol=0., atol=1e4*np.finfo(dtype_).eps)
@pytest.mark.parametrize("dtype", DTYPES)
# 参数化测试用例，对每种数据类型进行测试
@pytest.mark.parametrize("trans_bool", [False, True])
# 参数化测试用例，测试是否进行转置操作
@pytest.mark.parametrize("fact", ["F", "N"])
# 参数化测试用例，测试不同的'fact'值

def test_gtsvx(dtype, trans_bool, fact):
    """
    These tests uses ?gtsvx to solve a random Ax=b system for each dtype.
    It tests that the outputs define an LU matrix, that inputs are unmodified,
    transposal options, incompatible shapes, singular matrices, and
    singular factorizations. It parametrizes DTYPES and the 'fact' value along
    with the fact related inputs.
    """
    seed(42)
    # 设置与数据类型相适应的测试容差
    atol = 100 * np.finfo(dtype).eps
    # 获取 LAPACK 求解函数 gtsvx 和 gttrf
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    
    # 生成随机的三对角矩阵 A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    
    # 生成随机解向量 x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    # 根据 x 生成方程组 Ax=b 的右端向量 b
    trans = ("T" if dtype in REAL_DTYPES else "C") if trans_bool else "N"
    b = (A.conj().T if trans_bool else A) @ x

    # 存储输入的副本，以便后续检查它们是否被修改
    inputs_cpy = [dl.copy(), d.copy(), du.copy(), b.copy()]

    # 如果 fact = 'F'，则将返回的值赋给变量，否则置为 None
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6
    
    # 使用 gtsvx 求解方程组
    gtsvx_out = gtsvx(dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    
    # 断言求解信息为零，表示求解成功
    assert_(info == 0, f"?gtsvx info = {info}, should be zero")

    # 确保输入参数未被修改
    assert_array_equal(dl, inputs_cpy[0])
    assert_array_equal(d, inputs_cpy[1])
    assert_array_equal(du, inputs_cpy[2])
    assert_array_equal(b, inputs_cpy[3])

    # 检查 x_soln 是否接近于期望的 x
    assert_allclose(x, x_soln, atol=atol)

    # 断言输出结果的类型或形状正确
    # rcond 应为标量
    assert_(hasattr(rcond, "__len__") is not True,
            f"rcond should be scalar but is {rcond}")
    # ferr 的长度应与 x 的列数相同
    assert_(ferr.shape[0] == b.shape[1], (f"ferr.shape is {ferr.shape[0]} but should"
                                          f" be {b.shape[1]}"))
    # berr 的长度应与 x 的列数相同
    assert_(berr.shape[0] == b.shape[1], (f"berr.shape is {berr.shape[0]} but should"
                                          f" be {b.shape[1]}"))


@pytest.mark.parametrize("dtype", DTYPES)
# 参数化测试用例，对每种数据类型进行测试
@pytest.mark.parametrize("trans_bool", [0, 1])
# 参数化测试用例，测试是否进行转置操作
@pytest.mark.parametrize("fact", ["F", "N"])
# 参数化测试用例，测试不同的'fact'值

def test_gtsvx_error_singular(dtype, trans_bool, fact):
    seed(42)
    # 获取 LAPACK 求解函数 gtsvx 和 gttrf
    # 获取 LAPACK 中的函数 gtsvx 和 gttrf，返回两个函数对象
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    
    # 生成一个随机的三对角矩阵 A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)  # 下对角线元素数组
    d = generate_random_dtype_array((n,), dtype=dtype)     # 对角线元素数组
    du = generate_random_dtype_array((n-1,), dtype=dtype)  # 上对角线元素数组
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)       # 组合成三对角矩阵 A
    
    # 生成一个随机解 x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    
    # 根据 A 和转置类型创建方程 Ax=b 的右侧向量 b
    trans = "T" if dtype in REAL_DTYPES else "C"
    b = (A.conj().T if trans_bool else A) @ x
    
    # 如果 fact = 'N'，则将以下变量设置为 None；如果 fact = 'F'，则通过 gttrf 获取这些变量的值
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6
    
    # 调用 gtsvx 解方程组 dl, d, du, b，根据 fact 和 trans 的设置，以及 gttrf 的输出
    gtsvx_out = gtsvx(dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    
    # 将 gtsvx 的输出解包到各个变量中
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    
    # 对于 fact = "N" 的情况，测试一个奇异矩阵的情况
    if fact == "N":
        # 手动构造一个奇异的例子
        d[-1] = 0
        dl[-1] = 0
        
        # 使用 gtsvx 解方程组 dl, d, du, b
        gtsvx_out = gtsvx(dl, d, du, b)
        
        # 将 gtsvx 的输出解包到各个变量中
        dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
        
        # 断言：对于奇异矩阵，info 应大于 0
        assert info > 0, "info should be > 0 for singular matrix"
    
    elif fact == 'F':
        # 假设输入是一个奇异因子分解
        df_[-1] = 0
        duf_[-1] = 0
        du2f_[-1] = 0
        
        # 使用 gtsvx 解方程组 dl, d, du, b，根据 fact 和先前计算的因子
        gtsvx_out = gtsvx(dl, d, du, b, fact=fact, dlf=dlf_, df=df_, duf=duf_,
                          du2=du2f_, ipiv=ipiv_)
        
        # 将 gtsvx 的输出解包到各个变量中
        dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
        
        # 断言：info 应大于 0，表明奇异因子分解存在问题
        assert info > 0, "info should be > 0 for singular matrix"
@pytest.mark.parametrize("dtype", DTYPES*2)
# 使用参数化测试，循环测试所有数据类型两次
@pytest.mark.parametrize("trans_bool", [False, True])
# 使用参数化测试，测试转置标志为False和True两种情况
@pytest.mark.parametrize("fact", ["F", "N"])
# 使用参数化测试，测试因子化选项为'F'和'N'两种情况
def test_gtsvx_error_incompatible_size(dtype, trans_bool, fact):
    # 设定随机数生成器种子为42
    seed(42)
    
    # 获得 LAPACK 函数 gtsvx 和 gttrf
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    
    # 生成随机的三对角矩阵 A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    
    # 生成随机解 x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    
    # 根据转置标志创建方程 Ax=b 的右端向量 b
    trans = "T" if dtype in REAL_DTYPES else "C"
    b = (A.conj().T if trans_bool else A) @ x

    # 如果 fact = 'F'，通过 gttrf 计算因子化结果，否则设为 None
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6

    # 根据不同的 fact 值进行测试
    if fact == "N":
        # 如果 fact = 'N'，预期引发 ValueError 异常
        assert_raises(ValueError, gtsvx, dl[:-1], d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d[:-1], du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du[:-1], b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(Exception, gtsvx, dl, d, du, b[:-1],
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    else:
        # 如果 fact != 'N'，预期引发 ValueError 异常
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_[:-1], df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_[:-1],
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_[:-1], du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_[:-1], ipiv=ipiv_)
@pytest.mark.parametrize("du,d,dl,b,x",
                         [(np.array([2.1, -1.0, 1.9, 8.0]),  # 定义测试参数 du，为包含浮点数的数组
                           np.array([3.0, 2.3, -5.0, -0.9, 7.1]),  # 定义测试参数 d，为包含浮点数的数组
                           np.array([3.4, 3.6, 7.0, -6.0]),  # 定义测试参数 dl，为包含浮点数的数组
                           np.array([[2.7, 6.6], [-.5, 10.8], [2.6, -3.2],  # 定义测试参数 b，为包含浮点数的二维数组
                                     [.6, -11.2], [2.7, 19.1]]),
                           np.array([[-4, 5], [7, -4], [3, -3], [-4, -2],  # 定义期望结果 x，为包含整数的二维数组
                                     [-3, 1]])),
                          (np.array([2 - 1j, 2 + 1j, -1 + 1j, 1 - 1j]),  # 定义测试参数 du，为包含复数的数组
                           np.array([-1.3 + 1.3j, -1.3 + 1.3j, -1.3 + 3.3j,  # 定义测试参数 d，为包含复数的数组
                                     -.3 + 4.3j, -3.3 + 1.3j]),
                           np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j]),  # 定义测试参数 dl，为包含复数的数组
                           np.array([[2.4 - 5j, 2.7 + 6.9j],  # 定义测试参数 b，为包含复数的二维数组
                                     [3.4 + 18.2j, -6.9 - 5.3j],
                                     [-14.7 + 9.7j, -6 - .6j],
                                     [31.9 - 7.7j, -3.9 + 9.3j],
                                     [-1 + 1.6j, -3 + 12.2j]]),
                           np.array([[1 + 1j, 2 - 1j], [3 - 1j, 1 + 2j],  # 定义期望结果 x，为包含复数的二维数组
                                     [4 + 5j, -1 + 1j], [-1 - 2j, 2 + 1j],
                                     [1 - 1j, 2 - 2j]]))])
def test_gtsvx_NAG(du, d, dl, b, x):
    # Test to ensure wrapper is consistent with NAG Manual Mark 26
    # example problems: real (f07cbf) and complex (f07cpf)
    gtsvx = get_lapack_funcs('gtsvx', dtype=d.dtype)  # 获取 LAPACK 函数 gtsvx 对应指定数据类型的函数

    gtsvx_out = gtsvx(dl, d, du, b)  # 调用 gtsvx 函数进行求解
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out  # 解析 gtsvx 函数返回的结果元组

    assert_array_almost_equal(x, x_soln)  # 断言计算得到的解 x_soln 与预期结果 x 接近


@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES
                                               + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e: get_lapack_funcs('pttrf',  # 获取 LAPACK 函数 pttrf 对应指定数据类型的函数
                                                         dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx(dtype, realtype, fact, df_de_lambda):
    '''
    This tests the ?ptsvx lapack routine wrapper to solve a random system
    Ax = b for all dtypes and input variations. Tests for: unmodified
    input parameters, fact options, incompatible matrix shapes raise an error,
    and singular matrices return info of illegal value.
    '''
    seed(42)
    # set test tolerance appropriate for dtype
    atol = 100 * np.finfo(dtype).eps  # 设置测试精度公差，适合于指定数据类型
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)  # 获取 LAPACK 函数 ptsvx 对应指定数据类型的函数
    n = 5
    # create diagonals according to size and dtype
    d = generate_random_dtype_array((n,), realtype) + 4  # 根据大小和数据类型生成对角线元素 d
    e = generate_random_dtype_array((n-1,), dtype)  # 根据大小和数据类型生成次对角线元素 e
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)  # 创建对称带状矩阵 A
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)  # 生成解的预期结果 x_soln
    b = A @ x_soln  # 计算右侧向量 b

    # use lambda to determine what df, ef are
    df, ef, info = df_de_lambda(d, e)  # 使用 lambda 函数确定 df, ef 的值
    # 创建副本以便稍后测试它们是否未被修改
    diag_cpy = [d.copy(), e.copy(), b.copy()]

    # 使用 ptsvx 函数求解
    # df, ef, b 是输入参数，fact 是可选参数
    # ptsvx 返回 df, ef, x, rcond, ferr, berr, info 这些变量
    df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b, fact=fact,
                                               df=df, ef=ef)
    
    # 检查 d, e, b 是否未被修改
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    assert_array_equal(b, diag_cpy[2])
    # 检查 info 是否为 0
    assert_(info == 0, f"info 应为 0，实际为 {info}.")
    # 检查 x_soln 和 x 是否近似相等
    assert_array_almost_equal(x_soln, x)

    # 测试 ptsvx 返回的因子是否能够重新组合成矩阵 A
    L = np.diag(ef, -1) + np.diag(np.ones(n))
    D = np.diag(df)
    # 检查 A 是否可以由 L, D, L 的共轭转置重组成
    assert_allclose(A, L @ D @ (np.conj(L).T), atol=atol)

    # 断言输出变量的类型或形状是否正确
    # 检查 rcond 是否为标量
    assert not hasattr(rcond, "__len__"), \
        f"rcond 应为标量，实际为 {rcond}"
    # 检查 ferr 的形状是否与 x 的列数匹配
    assert_(ferr.shape == (2,), (f"ferr.shape 为 {ferr.shape}，应为 "
                                 f"({x_soln.shape[1]},)"))
    # 检查 berr 的形状是否与 x 的列数匹配
    assert_(berr.shape == (2,), (f"berr.shape 为 {berr.shape}，应为 "
                                 f"({x_soln.shape[1]},)"))
# 使用 pytest 的 mark.parametrize 装饰器，为测试函数 test_ptsvx_error_raise_errors 和 test_ptsvx_non_SPD_singular 提供参数化测试数据
@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e: get_lapack_funcs('pttrf',
                                                         dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx_error_raise_errors(dtype, realtype, fact, df_de_lambda):
    # 设定随机数种子
    seed(42)
    # 获取特定数据类型的 LAPACK 算法函数
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    # 创建大小为 n 的对角数组 d 和大小为 n-1 的副对角数组 e
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)
    # 构建矩阵 A，包括主对角线 d、副对角线 e 和共轭副对角线
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    # 生成随机数组 x_soln，并计算右侧向量 b
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln

    # 使用 lambda 函数确定 df 和 ef 的值
    df, ef, info = df_de_lambda(d, e)

    # 测试输入数组大小不匹配时是否引发 ValueError 异常
    assert_raises(ValueError, ptsvx, d[:-1], e, b, fact=fact, df=df, ef=ef)
    assert_raises(ValueError, ptsvx, d, e[:-1], b, fact=fact, df=df, ef=ef)
    assert_raises(Exception, ptsvx, d, e, b[:-1], fact=fact, df=df, ef=ef)


# 同上一个测试函数，但这里测试的是非正定和奇异情况
@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e: get_lapack_funcs('pttrf',
                                                         dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx_non_SPD_singular(dtype, realtype, fact, df_de_lambda):
    seed(42)
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln

    df, ef, info = df_de_lambda(d, e)

    if fact == "N":
        # 引入一个非正定情况，修改 d 使其第三个元素为 0
        d[3] = 0
        df, ef, info = df_de_lambda(d, e)
        # 使用 ptsvx 求解
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b)
        # 检查是否因为矩阵奇异引发了 info > 0 and info <= n 的情况
        assert info > 0 and info <= n

        # 针对非正定矩阵的情况
        d = generate_random_dtype_array((n,), realtype)
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b)
        assert info > 0 and info <= n
    else:
        # 假设某人使用了奇异因子分解，此时将 df[0] 和 ef[0] 设置为 0
        df, ef, info = df_de_lambda(d, e)
        df[0] = 0
        ef[0] = 0
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b, fact=fact,
                                                   df=df, ef=ef)
        assert info > 0
@pytest.mark.parametrize('d,e,b,x',
                         [(np.array([4, 10, 29, 25, 5]),   # 定义测试参数 d，表示问题的维度
                           np.array([-2, -6, 15, 8]),    # 定义测试参数 e，表示问题的维度
                           np.array([[6, 10], [9, 4], [2, 9], [14, 65], [7, 23]]),  # 定义测试参数 b，表示问题的维度
                           np.array([[2.5, 2], [2, -1], [1, -3], [-1, 6], [3, -5]])),  # 定义测试参数 x，表示问题的维度
                          (np.array([16, 41, 46, 21]),   # 定义测试参数 d，表示问题的维度
                           np.array([16 + 16j, 18 - 9j, 1 - 4j]),   # 定义测试参数 e，表示问题的维度
                           np.array([[64 + 16j, -16 - 32j], [93 + 62j, 61 - 66j], [78 - 80j, 71 - 74j], [14 - 27j, 35 + 15j]]),  # 定义测试参数 b，表示问题的维度
                           np.array([[2 + 1j, -3 - 2j], [1 + 1j, 1 + 1j], [1 - 2j, 1 - 2j], [1 - 1j, 2 + 1j]]))])  # 定义测试参数 x，表示问题的维度
def test_ptsvx_NAG(d, e, b, x):
    # 确保封装函数与 NAG 手册第 26 版中示例问题 f07jbf、f07jpf 保持一致
    # （链接可能过期，请搜索在线查找"NAG Library Manual Mark 26"）
    
    # 根据 e 的数据类型获取适当类型的 Lapack 函数
    ptsvx = get_lapack_funcs('ptsvx', dtype=e.dtype)
    # 使用 Lapack 函数求解
    df, ef, x_ptsvx, rcond, ferr, berr, info = ptsvx(d, e, b)
    # 断言 ptsvx 的解 x 与预期解 x_ptsvx 相同
    assert_array_almost_equal(x, x_ptsvx)


@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('dtype', DTYPES)
def test_pptrs_pptri_pptrf_ppsv_ppcon(dtype, lower):
    seed(1234)
    atol = np.finfo(dtype).eps*100
    # 在此处可以进行手动将/从压缩格式转换
    n, nrhs = 10, 4
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, nrhs], dtype=dtype)

    a = a.conj().T + a + np.eye(n, dtype=dtype) * dtype(5.)
    if lower:
        inds = ([x for y in range(n) for x in range(y, n)],   # 确定上/下三角矩阵的索引
                [y for y in range(n) for x in range(y, n)])
    else:
        inds = ([x for y in range(1, n+1) for x in range(y)],   # 确定上/下三角矩阵的索引
                [y-1 for y in range(1, n+1) for x in range(y)])
    ap = a[inds]
    ppsv, pptrf, pptrs, pptri, ppcon = get_lapack_funcs(
        ('ppsv', 'pptrf', 'pptrs', 'pptri', 'ppcon'),
        dtype=dtype,
        ilp64="preferred")

    ul, info = pptrf(n, ap, lower=lower)
    assert_equal(info, 0)
    aul = cholesky(a, lower=lower)[inds]
    assert_allclose(ul, aul, rtol=0, atol=atol)

    uli, info = pptri(n, ul, lower=lower)
    assert_equal(info, 0)
    auli = inv(a)[inds]
    assert_allclose(uli, auli, rtol=0, atol=atol)

    x, info = pptrs(n, ul, b, lower=lower)
    assert_equal(info, 0)
    bx = solve(a, b)
    assert_allclose(x, bx, rtol=0, atol=atol)

    xv, info = ppsv(n, ap, b, lower=lower)
    assert_equal(info, 0)
    assert_allclose(xv, bx, rtol=0, atol=atol)

    anorm = np.linalg.norm(a, 1)
    # 调用 ppcon 函数，计算给定参数 n, ap 的条件数和信息
    rcond, info = ppcon(n, ap, anorm=anorm, lower=lower)
    # 使用断言确保返回的 info 值为 0，表示计算正常
    assert_equal(info, 0)
    # 使用断言检查条件数的计算结果，确保其相对误差小于 1
    assert_(abs(1/rcond - np.linalg.cond(a, p=1))*rcond < 1)
@pytest.mark.parametrize('dtype', DTYPES)
def test_gees_trexc(dtype):
    # 设置随机数种子，以确保可重复性
    seed(1234)
    # 计算所需的容差值，用于数值比较
    atol = np.finfo(dtype).eps * 100

    n = 10
    # 生成指定类型和形状的随机数组
    a = generate_random_dtype_array([n, n], dtype=dtype)

    # 获取 LAPACK 函数 gees 和 trexc
    gees, trexc = get_lapack_funcs(('gees', 'trexc'), dtype=dtype)

    # 调用 gees 函数，并传入 lambda 函数作为空回调函数
    result = gees(lambda x: None, a, overwrite_a=False)
    # 断言返回结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取 gees 返回结果中的 t 和 z
    t = result[0]
    z = result[-3]

    # 获取 t 矩阵中特定位置的值
    d2 = t[6, 6]

    # 如果数据类型是复数类型，则检查 t 是否为上三角矩阵
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言 z @ t @ z^H 结果与 a 矩阵相近
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)

    # 调用 trexc 函数
    result = trexc(t, z, 7, 1)
    # 断言返回结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取 trexc 返回结果中的 t 和 z
    t = result[0]
    z = result[-2]

    # 如果数据类型是复数类型，则检查 t 是否为上三角矩阵
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言 z @ t @ z^H 结果与 a 矩阵相近
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)

    # 断言 t 矩阵特定位置的值与 d2 相等
    assert_allclose(t[0, 0], d2, rtol=0, atol=atol)


@pytest.mark.parametrize(
    "t, expect, ifst, ilst",
    [(np.array([[0.80, -0.11, 0.01, 0.03],
                [0.00, -0.10, 0.25, 0.35],
                [0.00, -0.65, -0.10, 0.20],
                [0.00, 0.00, 0.00, -0.10]]),
      np.array([[-0.1000, -0.6463, 0.0874, 0.2010],
                [0.2514, -0.1000, 0.0927, 0.3505],
                [0.0000, 0.0000, 0.8000, -0.0117],
                [0.0000, 0.0000, 0.0000, -0.1000]]),
      2, 1),
     (np.array([[-6.00 - 7.00j, 0.36 - 0.36j, -0.19 + 0.48j, 0.88 - 0.25j],
                [0.00 + 0.00j, -5.00 + 2.00j, -0.03 - 0.72j, -0.23 + 0.13j],
                [0.00 + 0.00j, 0.00 + 0.00j, 8.00 - 1.00j, 0.94 + 0.53j],
                [0.00 + 0.00j, 0.00 + 0.00j, 0.00 + 0.00j, 3.00 - 4.00j]]),
      np.array([[-5.0000 + 2.0000j, -0.1574 + 0.7143j,
                 0.1781 - 0.1913j, 0.3950 + 0.3861j],
                [0.0000 + 0.0000j, 8.0000 - 1.0000j,
                 1.0742 + 0.1447j, 0.2515 - 0.3397j],
                [0.0000 + 0.0000j, 0.0000 + 0.0000j,
                 3.0000 - 4.0000j, 0.2264 + 0.8962j],
                [0.0000 + 0.0000j, 0.0000 + 0.0000j,
                 0.0000 + 0.0000j, -6.0000 - 7.0000j]]),
      1, 4)])
def test_trexc_NAG(t, ifst, ilst, expect):
    """
    This test implements the example found in the NAG manual,
    f08qfc, f08qtc, f08qgc, f08quc.
    """
    # NAG 手册提供的精度为小数点后四位
    atol = 1e-4
    # 获取 trexc LAPACK 函数
    trexc = get_lapack_funcs('trexc', dtype=t.dtype)

    # 调用 trexc 函数
    result = trexc(t, t, ifst, ilst, wantq=0)
    # 断言返回结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取 trexc 返回结果中的 t 矩阵
    t = result[0]
    # 断言 t 矩阵与期望值相近
    assert_allclose(expect, t, atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
def test_gges_tgexc(dtype):
    # 设置随机数种子，以确保可重复性
    seed(1234)
    # 计算所需的容差值，用于数值比较
    atol = np.finfo(dtype).eps * 100

    n = 10
    # 生成指定类型和形状的随机数组
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)

    # 获取 LAPACK 函数 gges 和 tgexc
    gges, tgexc = get_lapack_funcs(('gges', 'tgexc'), dtype=dtype)

    # 调用 gges 函数，并传入 lambda 函数作为空回调函数
    result = gges(lambda x: None, a, b, overwrite_a=False, overwrite_b=False)
    # 断言返回结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取 gges 返回结果中的 s、t、q 和 z
    s = result[0]
    t = result[1]
    q = result[-4]
    z = result[-3]
    # 计算变量 d1，等于矩阵 s 的第一行第一列元素除以矩阵 t 的第一行第一列元素
    d1 = s[0, 0] / t[0, 0]
    
    # 计算变量 d2，等于矩阵 s 的第七行第七列元素除以矩阵 t 的第七行第七列元素
    d2 = s[6, 6] / t[6, 6]
    
    # 如果数据类型为复数类型列表中的一种
    if dtype in COMPLEX_DTYPES:
        # 断言 s 等于 s 的上三角矩阵，使用指定的绝对误差值
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        # 断言 t 等于 t 的上三角矩阵，使用指定的绝对误差值
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    
    # 断言矩阵乘积 q @ s @ z 的共轭转置等于 a，使用指定的绝对误差值
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    # 断言矩阵乘积 q @ t @ z 的共轭转置等于 b，使用指定的绝对误差值
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
    
    # 调用 tgexc 函数处理 s, t, q, z，返回结果
    result = tgexc(s, t, q, z, 7, 1)
    # 断言结果的最后一个元素为 0
    assert_equal(result[-1], 0)
    
    # 更新变量 s, t, q, z 为 tgexc 函数返回结果的前四个元素
    s = result[0]
    t = result[1]
    q = result[2]
    z = result[3]
    
    # 如果数据类型为复数类型列表中的一种
    if dtype in COMPLEX_DTYPES:
        # 断言 s 等于 s 的上三角矩阵，使用指定的绝对误差值
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        # 断言 t 等于 t 的上三角矩阵，使用指定的绝对误差值
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)
    
    # 断言矩阵乘积 q @ s @ z 的共轭转置等于 a，使用指定的绝对误差值
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    # 断言矩阵乘积 q @ t @ z 的共轭转置等于 b，使用指定的绝对误差值
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)
    
    # 断言 s 的第一行第一列元素除以 t 的第一行第一列元素等于变量 d2，使用指定的绝对误差值
    assert_allclose(s[0, 0] / t[0, 0], d2, rtol=0, atol=atol)
    # 断言 s 的第二行第二列元素除以 t 的第二行第二列元素等于变量 d1，使用指定的绝对误差值
    assert_allclose(s[1, 1] / t[1, 1], d1, rtol=0, atol=atol)
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_gees_trsen 创建多个参数化的测试实例
@pytest.mark.parametrize('dtype', DTYPES)
def test_gees_trsen(dtype):
    # 设置随机数种子为 1234
    seed(1234)
    # 计算当前数据类型的机器精度乘以 100作为误差容限
    atol = np.finfo(dtype).eps * 100

    # 设定数组大小为 n x n，并生成随机数据类型数组 a
    n = 10
    a = generate_random_dtype_array([n, n], dtype=dtype)

    # 获取 Lapack 函数 gees, trsen, trsen_lwork 对应当前数据类型 dtype
    gees, trsen, trsen_lwork = get_lapack_funcs(('gees', 'trsen', 'trsen_lwork'), dtype=dtype)

    # 调用 gees 函数，传入 lambda 函数和数组 a，不覆盖输入数组 a
    result = gees(lambda x: None, a, overwrite_a=False)
    # 断言结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取结果中的矩阵 t 和矩阵 z
    t = result[0]
    z = result[-3]

    # 从 t 中提取特定位置的元素 d2
    d2 = t[6, 6]

    # 如果数据类型为复数类型，则断言 t 是上三角矩阵
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言 z @ t @ z.conj().T 等于输入数组 a
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)

    # 创建一个长度为 n 的全零数组 select，并将第 7 个位置设为 1
    select = np.zeros(n)
    select[6] = 1

    # 计算传递给 trsen_lwork 函数的工作空间长度
    lwork = _compute_lwork(trsen_lwork, select, t)

    # 如果数据类型为复数类型，则调用 trsen 函数，传入 select, t, z 和 lwork
    if dtype in COMPLEX_DTYPES:
        result = trsen(select, t, z, lwork=lwork)
    else:
        # 否则，传入额外的整数工作空间长度 liwork=lwork[1]
        result = trsen(select, t, z, lwork=lwork, liwork=lwork[1])
    # 断言结果的最后一个元素为 0
    assert_equal(result[-1], 0)

    # 获取结果中的矩阵 t 和矩阵 z
    t = result[0]
    z = result[1]

    # 如果数据类型为复数类型，则断言 t 是上三角矩阵
    if dtype in COMPLEX_DTYPES:
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言 z @ t @ z.conj().T 等于输入数组 a
    assert_allclose(z @ t @ z.conj().T, a, rtol=0, atol=atol)

    # 断言 t 中第一个元素 t[0, 0] 等于之前保存的 d2
    assert_allclose(t[0, 0], d2, rtol=0, atol=atol)


# 设置参数化测试的参数，用于输入多个测试实例
@pytest.mark.parametrize(
    "t, q, expect, select, expect_s, expect_sep",
    # 定义一个包含多个元组的列表，每个元组包含以下内容：
    #   - 4x4 复数类型的 NumPy 数组
    #   - 4x4 复数类型的 NumPy 数组
    #   - 4x4 复数类型的 NumPy 数组
    #   - 长度为 4 的整数 NumPy 数组
    #   - 浮点数，代表某个数值
    #   - 浮点数，代表另一个数值
    [
        (
            np.array([[0.7995, -0.1144, 0.0060, 0.0336],
                      [0.0000, -0.0994, 0.2478, 0.3474],
                      [0.0000, -0.6483, -0.0994, 0.2026],
                      [0.0000, 0.0000, 0.0000, -0.1007]]),
            np.array([[0.6551, 0.1037, 0.3450, 0.6641],
                      [0.5236, -0.5807, -0.6141, -0.1068],
                      [-0.5362, -0.3073, -0.2935, 0.7293],
                      [0.0956, 0.7467, -0.6463, 0.1249]]),
            np.array([[0.3500, 0.4500, -0.1400, -0.1700],
                      [0.0900, 0.0700, -0.5399, 0.3500],
                      [-0.4400, -0.3300, -0.0300, 0.1700],
                      [0.2500, -0.3200, -0.1300, 0.1100]]),
            np.array([1, 0, 0, 1]),
            1.75e+00, 3.22e+00
        ),
        (
            np.array([[-6.0004 - 6.9999j, 0.3637 - 0.3656j,
                       -0.1880 + 0.4787j, 0.8785 - 0.2539j],
                      [0.0000 + 0.0000j, -5.0000 + 2.0060j,
                       -0.0307 - 0.7217j, -0.2290 + 0.1313j],
                      [0.0000 + 0.0000j, 0.0000 + 0.0000j,
                       7.9982 - 0.9964j, 0.9357 + 0.5359j],
                      [0.0000 + 0.0000j, 0.0000 + 0.0000j,
                       0.0000 + 0.0000j, 3.0023 - 3.9998j]]),
            np.array([[-0.8347 - 0.1364j, -0.0628 + 0.3806j,
                       0.2765 - 0.0846j, 0.0633 - 0.2199j],
                      [0.0664 - 0.2968j, 0.2365 + 0.5240j,
                       -0.5877 - 0.4208j, 0.0835 + 0.2183j],
                      [-0.0362 - 0.3215j, 0.3143 - 0.5473j,
                       0.0576 - 0.5736j, 0.0057 - 0.4058j],
                      [0.0086 + 0.2958j, -0.3416 - 0.0757j,
                       -0.1900 - 0.1600j, 0.8327 - 0.1868j]]),
            np.array([[-3.9702 - 5.0406j, -4.1108 + 3.7002j,
                       -0.3403 + 1.0098j, 1.2899 - 0.8590j],
                      [0.3397 - 1.5006j, 1.5201 - 0.4301j,
                       1.8797 - 5.3804j, 3.3606 + 0.6498j],
                      [3.3101 - 3.8506j, 2.4996 + 3.4504j,
                       0.8802 - 1.0802j, 0.6401 - 1.4800j],
                      [-1.0999 + 0.8199j, 1.8103 - 1.5905j,
                       3.2502 + 1.3297j, 1.5701 - 3.4397j]]),
            np.array([1, 0, 0, 1]),
            1.02e+00, 1.82e-01
        )
    ]
# 定义一个测试函数，用于测试 trsen 和 tgsen 函数在 NAG 手册中的示例中的实现
def test_trsen_NAG(t, q, select, expect, expect_s, expect_sep):
    """
    This test implements the example found in the NAG manual,
    f08qgc, f08quc.
    """
    # 设置精度容差为 1e-4 和 1e-2，以匹配 NAG 手册中提供的精度要求
    atol = 1e-4
    atol2 = 1e-2

    # 获取 LAPACK 函数 trsen 和 trsen_lwork
    trsen, trsen_lwork = get_lapack_funcs(('trsen', 'trsen_lwork'), dtype=t.dtype)

    # 计算所需的工作空间大小
    lwork = _compute_lwork(trsen_lwork, select, t)

    # 根据数据类型调用不同的 LAPACK 函数
    if t.dtype in COMPLEX_DTYPES:
        result = trsen(select, t, q, lwork=lwork)
    else:
        result = trsen(select, t, q, lwork=lwork, liwork=lwork[1])

    # 断言最后一个返回值为 0，表示函数执行成功
    assert_equal(result[-1], 0)

    # 更新 t 和 q 的值为函数返回结果中的新值
    t = result[0]
    q = result[1]

    # 根据数据类型选择相应的变量来存储返回结果中的特征值和分离因子
    if t.dtype in COMPLEX_DTYPES:
        s = result[4]
        sep = result[5]
    else:
        s = result[5]
        sep = result[6]

    # 断言计算得到的矩阵乘积与期望值在给定的容差范围内接近
    assert_allclose(expect, q @ t @ q.conj().T, atol=atol)
    assert_allclose(expect_s, 1 / s, atol=atol2)
    assert_allclose(expect_sep, 1 / sep, atol=atol2)


# 使用给定的数据类型在参数化测试中测试 gges_tgsen 函数
@pytest.mark.parametrize('dtype', DTYPES)
def test_gges_tgsen(dtype):
    seed(1234)
    # 设置浮点数精度容差为当前数据类型的 epsilon 的 100 倍
    atol = np.finfo(dtype).eps * 100

    # 设置矩阵维度为 10，并生成随机数组 a 和 b
    n = 10
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)

    # 获取 LAPACK 函数 gges, tgsen, tgsen_lwork
    gges, tgsen, tgsen_lwork = get_lapack_funcs(('gges', 'tgsen', 'tgsen_lwork'), dtype=dtype)

    # 调用 gges 函数并获取返回结果
    result = gges(lambda x: None, a, b, overwrite_a=False, overwrite_b=False)

    # 断言最后一个返回值为 0，表示函数执行成功
    assert_equal(result[-1], 0)

    # 从返回结果中提取特征值、特征向量矩阵、Q 矩阵和 Z 矩阵
    s = result[0]
    t = result[1]
    q = result[-4]
    z = result[-3]

    # 计算特定位置上的比值 d1 和 d2
    d1 = s[0, 0] / t[0, 0]
    d2 = s[6, 6] / t[6, 6]

    # 如果数据类型是复数类型，额外断言返回的特征值和特征向量的上三角部分在容差范围内接近
    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言计算得到的矩阵乘积与输入矩阵 a 和 b 在容差范围内接近
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

    # 创建选择数组 select，并计算 tgsen 函数所需的工作空间大小
    select = np.zeros(n)
    select[6] = 1
    lwork = _compute_lwork(tgsen_lwork, select, s, t)

    # 由于 LAPACK 存在的一个偏差，调整工作空间大小
    lwork = (lwork[0] + 1, lwork[1])

    # 调用 tgsen 函数并获取返回结果
    result = tgsen(select, s, t, q, z, lwork=lwork)

    # 断言最后一个返回值为 0，表示函数执行成功
    assert_equal(result[-1], 0)

    # 从返回结果中提取特征值、特征向量矩阵、Q 矩阵和 Z 矩阵
    s = result[0]
    t = result[1]
    q = result[-7]
    z = result[-6]

    # 如果数据类型是复数类型，额外断言返回的特征值和特征向量的上三角部分在容差范围内接近
    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    # 断言计算得到的矩阵乘积与输入矩阵 a 和 b 在容差范围内接近
    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

    # 断言特定位置上的比值与预期值在容差范围内接近
    assert_allclose(s[0, 0] / t[0, 0], d2, rtol=0, atol=atol)
    assert_allclose(s[1, 1] / t[1, 1], d1, rtol=0, atol=atol)
    # 创建一个包含多个 4x4 的 NumPy 数组的列表
    [
        # 第一个 NumPy 数组
        np.array([[4.0,   1.0,  1.0,  2.0],
                  [0.0,   3.0,  4.0,  1.0],
                  [0.0,   1.0,  3.0,  1.0],
                  [0.0,   0.0,  0.0,  6.0]]),
        # 第二个 NumPy 数组
        np.array([[1.0,   1.0,  1.0,  1.0],
                  [0.0,   3.0,  4.0,  1.0],
                  [0.0,   1.0,  3.0,  1.0],
                  [0.0,   0.0,  0.0,  4.0]]),
        # 第三个 NumPy 数组
        np.array([[-4.0,  7.0,  1.0, 12.0],
                  [-9.0,  2.0, -2.0, -2.0],
                  [-4.0,  2.0, -2.0,  8.0],
                  [-7.0,  7.0, -6.0, 19.0]]),
        # 第四个 NumPy 数组
        np.array([[2.0,   1.0,  1.0,  3.0],
                  [0.0,   1.0,  2.0,  1.0],
                  [0.0,   0.0,  1.0,  1.0],
                  [0.0,   0.0,  0.0,  2.0]]),
        # 第五个 NumPy 数组
        np.array([[1.0,   1.0,  1.0,  2.0],
                  [0.0,   1.0,  4.0,  1.0],
                  [0.0,   0.0,  1.0,  1.0],
                  [0.0,   0.0,  0.0,  1.0]]),
        # 第六个 NumPy 数组
        np.array([[-7.0,  5.0,  0.0,  7.0],
                  [-5.0,  1.0, -8.0,  0.0],
                  [-1.0,  2.0, -3.0,  5.0],
                  [-3.0,  2.0,  0.0,  5.0]]),
        # 第七个 NumPy 数组
        np.array([[1.0,   1.0,  1.0,  1.0],
                  [-1.0,  2.0, -1.0, -1.0],
                  [-1.0,  1.0,  3.0,  1.0],
                  [-1.0,  1.0, -1.0,  4.0]]),
        # 第八个 NumPy 数组
        np.array([[4.0,  -1.0,  1.0, -1.0],
                  [1.0,   3.0, -1.0,  1.0],
                  [-1.0,  1.0,  2.0, -1.0],
                  [1.0,  -1.0,  1.0,  1.0]])
    ]
# 使用参数化测试对 REAL_DTYPES 中的每种数据类型执行测试
@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_tgsyl_NAG(a, b, c, d, e, f, rans, lans, dtype):
    # 设置测试中的误差容限
    atol = 1e-4

    # 获取 LAPACK 中 tgsyl 函数对应的函数对象
    tgsyl = get_lapack_funcs(('tgsyl'), dtype=dtype)
    # 调用 tgsyl 函数进行计算
    rout, lout, scale, dif, info = tgsyl(a, b, c, d, e, f)

    # 断言 INFO 值为 0，表示计算无误
    assert_equal(info, 0)
    # 断言 SCALE 值接近 1.0，误差容限为 np.finfo(dtype).eps*100
    assert_allclose(scale, 1.0, rtol=0, atol=np.finfo(dtype).eps*100,
                    err_msg="SCALE must be 1.0")
    # 断言 DIF 值接近 0.0，误差容限为 np.finfo(dtype).eps*100
    assert_allclose(dif, 0.0, rtol=0, atol=np.finfo(dtype).eps*100,
                    err_msg="DIF must be nearly 0")
    # 断言 rout 的计算结果与预期结果 rans 接近，误差容限为 atol
    assert_allclose(rout, rans, atol=atol,
                    err_msg="Solution for R is incorrect")
    # 断言 lout 的计算结果与预期结果 lans 接近，误差容限为 atol
    assert_allclose(lout, lans, atol=atol,
                    err_msg="Solution for L is incorrect")


# 参数化测试函数，测试不同参数下的 tgsyl 函数行为
@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('trans', ('N', 'T'))
@pytest.mark.parametrize('ijob', [0, 1, 2, 3, 4])
def test_tgsyl(dtype, trans, ijob):
    # 根据数据类型设置不同的误差容限
    atol = 1e-3 if dtype == np.float32 else 1e-10
    # 设置随机数生成器的种子
    rng = np.random.default_rng(1685779866898198)
    # 定义矩阵维度
    m, n = 10, 15

    # 使用 QZ 分解生成矩阵 a 和 d
    a, d, *_ = qz(rng.uniform(-10, 10, [m, m]).astype(dtype),
                  rng.uniform(-10, 10, [m, m]).astype(dtype),
                  output='real')

    # 使用 QZ 分解生成矩阵 b 和 e
    b, e, *_ = qz(rng.uniform(-10, 10, [n, n]).astype(dtype),
                  rng.uniform(-10, 10, [n, n]).astype(dtype),
                  output='real')

    # 生成随机矩阵 c 和 f
    c = rng.uniform(-2, 2, [m, n]).astype(dtype)
    f = rng.uniform(-2, 2, [m, n]).astype(dtype)

    # 获取 LAPACK 中 tgsyl 函数对应的函数对象
    tgsyl = get_lapack_funcs(('tgsyl'), dtype=dtype)
    # 调用 tgsyl 函数进行计算
    rout, lout, scale, dif, info = tgsyl(a, b, c, d, e, f,
                                         trans=trans, ijob=ijob)

    # 断言 INFO 值为 0，表示计算无误
    assert info == 0, "INFO is non-zero"
    # 断言 SCALE 值为非负数
    assert scale >= 0.0, "SCALE must be non-negative"
    # 根据不同的 ijob 设置 DIF 的断言条件
    if ijob == 0:
        assert_allclose(dif, 0.0, rtol=0, atol=np.finfo(dtype).eps*100,
                        err_msg="DIF must be 0 for ijob =0")
    else:
        assert dif >= 0.0, "DIF must be non-negative"

    # 根据 trans 和 ijob 的值，对不同的计算方式进行断言
    # 只有在 ijob = 3 或 4 时计算 lhs1 和 rhs1
    if ijob <= 2:
        if trans == 'N':
            lhs1 = a @ rout - lout @ b
            rhs1 = scale*c
            lhs2 = d @ rout - lout @ e
            rhs2 = scale*f
        elif trans == 'T':
            lhs1 = np.transpose(a) @ rout + np.transpose(d) @ lout
            rhs1 = scale*c
            lhs2 = rout @ np.transpose(b) + lout @ np.transpose(e)
            rhs2 = -1.0*scale*f

        # 断言 lhs1 和 rhs1 接近，误差容限为 atol
        assert_allclose(lhs1, rhs1, atol=atol, rtol=0.,
                        err_msg='lhs1 and rhs1 do not match')
        # 断言 lhs2 和 rhs2 接近，误差容限为 atol
        assert_allclose(lhs2, rhs2, atol=atol, rtol=0.,
                        err_msg='lhs2 and rhs2 do not match')
```