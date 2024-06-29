# `.\numpy\numpy\linalg\tests\test_regression.py`

```py
# 导入警告模块，用于处理警告信息
import warnings

# 导入 pytest 模块，用于测试
import pytest

# 导入 numpy 库并指定别名 np
import numpy as np

# 从 numpy 库中导入线性代数相关的函数和对象
from numpy import linalg, arange, float64, array, dot, transpose

# 从 numpy.testing 模块中导入用于测试的函数和断言方法
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_array_equal,
    assert_array_almost_equal, assert_array_less
)

# 定义一个测试类 TestRegression，用于测试线性回归相关功能
class TestRegression:

    # 定义测试方法 test_eig_build，用于测试特征值的构建
    def test_eig_build(self):
        # Ticket #652
        # 定义一个复数数组 rva，表示预期的特征值
        rva = array([1.03221168e+02 + 0.j,
                     -1.91843603e+01 + 0.j,
                     -6.04004526e-01 + 15.84422474j,
                     -6.04004526e-01 - 15.84422474j,
                     -1.13692929e+01 + 0.j,
                     -6.57612485e-01 + 10.41755503j,
                     -6.57612485e-01 - 10.41755503j,
                     1.82126812e+01 + 0.j,
                     1.06011014e+01 + 0.j,
                     7.80732773e+00 + 0.j,
                     -7.65390898e-01 + 0.j,
                     1.51971555e-15 + 0.j,
                     -1.51308713e-15 + 0.j])
        
        # 创建一个 13x13 的浮点型数组 a，并进行初始化
        a = arange(13 * 13, dtype=float64)
        a.shape = (13, 13)
        a = a % 17
        
        # 使用 numpy.linalg.eig 函数计算矩阵 a 的特征值和特征向量
        va, ve = linalg.eig(a)
        
        # 对计算得到的特征值进行排序
        va.sort()
        rva.sort()
        
        # 使用断言方法验证计算得到的特征值数组 va 与预期的 rva 数组近似相等
        assert_array_almost_equal(va, rva)

    # 定义测试方法 test_eigh_build，用于测试对称矩阵的特征值和特征向量的构建
    def test_eigh_build(self):
        # Ticket 662.
        # 定义一个预期的特征值数组 rvals
        rvals = [68.60568999, 89.57756725, 106.67185574]
        
        # 创建一个对称矩阵 cov
        cov = array([[77.70273908,   3.51489954,  15.64602427],
                     [3.51489954,  88.97013878,  -1.07431931],
                     [15.64602427,  -1.07431931,  98.18223512]])
        
        # 使用 numpy.linalg.eigh 函数计算对称矩阵 cov 的特征值和特征向量
        vals, vecs = linalg.eigh(cov)
        
        # 使用断言方法验证计算得到的特征值数组 vals 与预期的 rvals 数组近似相等
        assert_array_almost_equal(vals, rvals)

    # 定义测试方法 test_svd_build，用于测试奇异值分解（SVD）
    def test_svd_build(self):
        # Ticket 627.
        # 创建一个数组 a
        a = array([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
        
        # 获取数组 a 的行数 m 和列数 n
        m, n = a.shape
        
        # 使用 numpy.linalg.svd 函数计算数组 a 的奇异值分解
        u, s, vh = linalg.svd(a)
        
        # 计算 dot(transpose(u[:, n:]), a) 并赋值给 b
        b = dot(transpose(u[:, n:]), a)
        
        # 使用断言方法验证计算得到的 b 数组与预期的零数组大小相同
        assert_array_almost_equal(b, np.zeros((2, 2)))

    # 定义测试方法 test_norm_vector_badarg，用于测试向量的范数
    def test_norm_vector_badarg(self):
        # Regression for #786: Frobenius norm for vectors raises
        # ValueError.
        # 使用断言方法验证计算向量的 Frobenius 范数会抛出 ValueError 异常
        assert_raises(ValueError, linalg.norm, array([1., 2., 3.]), 'fro')

    # 定义测试方法 test_lapack_endian，用于测试 LAPACK 函数在不同字节序下的一致性
    def test_lapack_endian(self):
        # For bug #1482
        # 创建一个以大端字节序存储的数组 a
        a = array([[5.7998084,  -2.1825367],
                   [-2.1825367,   9.85910595]], dtype='>f8')
        
        # 将数组 a 转换成小端字节序，并赋值给数组 b
        b = array(a, dtype='<f8')
        
        # 使用 numpy.linalg.cholesky 函数计算大端和小端字节序数组 a 和 b 的 Cholesky 分解
        ap = linalg.cholesky(a)
        bp = linalg.cholesky(b)
        
        # 使用断言方法验证大端和小端字节序数组的 Cholesky 分解结果相等
        assert_array_equal(ap, bp)

    # 定义测试方法 test_large_svd_32bit，用于测试大型矩阵的奇异值分解
    def test_large_svd_32bit(self):
        # See gh-4442, 64bit would require very large/slow matrices.
        # 创建一个单位矩阵，其大小为 1000x66
        x = np.eye(1000, 66)
        
        # 使用 numpy.linalg.svd 函数计算单位矩阵 x 的奇异值分解
        np.linalg.svd(x)
    def test_svd_no_uv(self):
        # 对于 issue 4733 进行测试

        # 遍历不同形状的矩阵：(3, 4), (4, 4), (4, 3)
        for shape in (3, 4), (4, 4), (4, 3):
            # 对于 float 和 complex 两种数据类型
            for t in float, complex:
                # 创建一个全为 1 的指定形状和数据类型的数组 a
                a = np.ones(shape, dtype=t)
                # 计算 a 的奇异值分解，但不计算 U 和 V 矩阵
                w = linalg.svd(a, compute_uv=False)
                # 计算绝对值大于 0.5 的元素个数
                c = np.count_nonzero(np.absolute(w) > 0.5)
                # 断言 c 的值为 1
                assert_equal(c, 1)
                # 断言 a 的秩为 1
                assert_equal(np.linalg.matrix_rank(a), 1)
                # 断言 a 的二范数大于 1
                assert_array_less(1, np.linalg.norm(a, ord=2))

                # 使用 svdvals 函数计算 a 的奇异值
                w_svdvals = linalg.svdvals(a)
                # 断言 w 和 w_svdvals 几乎相等
                assert_array_almost_equal(w, w_svdvals)

    def test_norm_object_array(self):
        # 对于 issue 7575 进行测试

        # 创建一个包含对象数组的 numpy 数组 testvector
        testvector = np.array([np.array([0, 1]), 0, 0], dtype=object)

        # 计算 testvector 的默认 L2 范数
        norm = linalg.norm(testvector)
        # 断言计算的范数值与预期相等
        assert_array_equal(norm, [0, 1])
        # 断言范数的数据类型为 float64
        assert_(norm.dtype == np.dtype('float64'))

        # 计算 testvector 的 L1 范数
        norm = linalg.norm(testvector, ord=1)
        # 断言计算的范数值与预期相等
        assert_array_equal(norm, [0, 1])
        # 断言范数的数据类型不是 float64
        assert_(norm.dtype != np.dtype('float64'))

        # 计算 testvector 的 L2 范数
        norm = linalg.norm(testvector, ord=2)
        # 断言计算的范数值与预期相等
        assert_array_equal(norm, [0, 1])
        # 断言范数的数据类型为 float64
        assert_(norm.dtype == np.dtype('float64'))

        # 测试异常情况：使用不支持的范数类型参数
        assert_raises(ValueError, linalg.norm, testvector, ord='fro')
        assert_raises(ValueError, linalg.norm, testvector, ord='nuc')
        assert_raises(ValueError, linalg.norm, testvector, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testvector, ord=-np.inf)
        assert_raises(ValueError, linalg.norm, testvector, ord=0)
        assert_raises(ValueError, linalg.norm, testvector, ord=-1)
        assert_raises(ValueError, linalg.norm, testvector, ord=-2)

        # 创建一个包含对象数组的二维 numpy 数组 testmatrix
        testmatrix = np.array([[np.array([0, 1]), 0, 0],
                               [0,                0, 0]], dtype=object)

        # 计算 testmatrix 的默认 L2 范数
        norm = linalg.norm(testmatrix)
        # 断言计算的范数值与预期相等
        assert_array_equal(norm, [0, 1])
        # 断言范数的数据类型为 float64
        assert_(norm.dtype == np.dtype('float64'))

        # 计算 testmatrix 的 Frobenius 范数
        norm = linalg.norm(testmatrix, ord='fro')
        # 断言计算的范数值与预期相等
        assert_array_equal(norm, [0, 1])
        # 断言范数的数据类型为 float64
        assert_(norm.dtype == np.dtype('float64'))

        # 测试异常情况：使用不支持的范数类型参数
        assert_raises(TypeError, linalg.norm, testmatrix, ord='nuc')
        assert_raises(ValueError, linalg.norm, testmatrix, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=0)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=1)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-1)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=2)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=-2)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=3)
    # 定义测试方法，用于测试解线性方程组时，当右侧矩阵较大时的情况
    def test_lstsq_complex_larger_rhs(self):
        # 标识issue编号为gh-9891，说明这段代码解决了该问题
        size = 20  # 设定矩阵大小为20x20
        n_rhs = 70  # 设定右侧矩阵的列数为70
        # 生成复数随机矩阵G，其实部和虚部都是从标准正态分布中生成的
        G = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        # 生成复数随机矩阵u，其实部和虚部都是从标准正态分布中生成的
        u = np.random.randn(size, n_rhs) + 1j * np.random.randn(size, n_rhs)
        # 计算线性方程组的右侧向量b，即G.dot(u)
        b = G.dot(u)
        # 使用最小二乘法求解线性方程组Gx = b，rcond=None表示使用默认条件
        # 返回结果包括解向量u_lstsq、残差res、秩rank和奇异值sv
        u_lstsq, res, rank, sv = linalg.lstsq(G, b, rcond=None)
        # 检查最小二乘法计算的解u_lstsq是否与原始u接近
        assert_array_almost_equal(u_lstsq, u)

    # 使用pytest的参数化装饰器，测试Cholesky分解对空数组的处理
    @pytest.mark.parametrize("upper", [True, False])
    def test_cholesky_empty_array(self, upper):
        # 标识issue编号为gh-25840，说明这段代码解决了该问题
        # 对空的0x0数组进行Cholesky分解，根据upper参数选择返回上三角或下三角矩阵
        res = np.linalg.cholesky(np.zeros((0, 0)), upper=upper)
        # 检查结果矩阵的大小是否为0
        assert res.size == 0

    # 使用pytest的参数化装饰器，测试matrix_rank函数对rtol参数的处理
    @pytest.mark.parametrize("rtol", [0.0, [0.0] * 4, np.zeros((4,))])
    def test_matrix_rank_rtol_argument(self, rtol):
        # 标识issue编号为gh-25877，说明这段代码解决了该问题
        # 创建一个4x3x2的全零数组x
        x = np.zeros((4, 3, 2))
        # 使用rtol参数调用matrix_rank函数计算数组x的秩
        res = np.linalg.matrix_rank(x, rtol=rtol)
        # 检查返回的秩数组的形状是否为(4,)
        assert res.shape == (4,)
```