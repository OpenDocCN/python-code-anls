# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_lu.py`

```
import pytest  # 导入 pytest 模块
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 方法，重命名为 assert_raises

import numpy as np  # 导入 NumPy 库并重命名为 np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve  # 从 SciPy 的 linalg 模块导入 LU 分解相关函数
from numpy.testing import assert_allclose, assert_array_equal, assert_equal  # 从 NumPy 的 testing 模块导入数组相等性断言方法

REAL_DTYPES = [np.float32, np.float64]  # 定义实数类型的 NumPy 数据类型列表
COMPLEX_DTYPES = [np.complex64, np.complex128]  # 定义复数类型的 NumPy 数据类型列表
DTYPES = REAL_DTYPES + COMPLEX_DTYPES  # 将实数和复数类型列表合并成一个总的数据类型列表


class TestLU:  # 定义测试类 TestLU

    def setup_method(self):  # 定义初始化方法
        self.rng = np.random.default_rng(1682281250228846)  # 创建一个 NumPy 随机数生成器对象

    @pytest.mark.parametrize('shape', [[2, 2], [2, 4], [4, 2], [20, 20],  # 参数化装饰器，定义测试方法的参数 shape
                                       [20, 4], [4, 20], [3, 2, 9, 9],
                                       [2, 2, 17, 5], [2, 2, 11, 7]])
    def test_simple_lu_shapes_real_complex(self, shape):  # 定义测试方法 test_simple_lu_shapes_real_complex
        a = self.rng.uniform(-10., 10., size=shape)  # 生成指定形状的随机数组 a
        p, l, u = lu(a)  # 对数组 a 进行 LU 分解，得到置换矩阵 p、下三角矩阵 l、上三角矩阵 u
        assert_allclose(a, p @ l @ u)  # 断言原始数组 a 与重构的乘积 p @ l @ u 很接近
        pl, u = lu(a, permute_l=True)  # 对数组 a 进行 LU 分解，并对下三角矩阵进行置换
        assert_allclose(a, pl @ u)  # 断言原始数组 a 与重构的乘积 pl @ u 很接近

        b = self.rng.uniform(-10., 10., size=shape)*1j  # 生成复数类型的随机数组 b
        b += self.rng.uniform(-10, 10, size=shape)  # 将实数随机数组加到复数数组 b 上
        pl, u = lu(b, permute_l=True)  # 对复数数组 b 进行 LU 分解，并对下三角矩阵进行置换
        assert_allclose(b, pl @ u)  # 断言原始数组 b 与重构的乘积 pl @ u 很接近

    @pytest.mark.parametrize('shape', [[2, 2], [2, 4], [4, 2], [20, 20],  # 参数化装饰器，定义测试方法的参数 shape
                                       [20, 4], [4, 20]])
    def test_simple_lu_shapes_real_complex_2d_indices(self, shape):  # 定义测试方法 test_simple_lu_shapes_real_complex_2d_indices
        a = self.rng.uniform(-10., 10., size=shape)  # 生成指定形状的随机数组 a
        p, l, u = lu(a, p_indices=True)  # 对数组 a 进行 LU 分解，返回置换矩阵 p、下三角矩阵 l、上三角矩阵 u，并返回置换索引
        assert_allclose(a, l[p, :] @ u)  # 断言原始数组 a 与重构的乘积 l[p, :] @ u 很接近

    def test_1by1_input_output(self):  # 定义测试方法 test_1by1_input_output
        a = self.rng.random([4, 5, 1, 1], dtype=np.float32)  # 生成指定形状的随机数组 a，数据类型为 np.float32
        p, l, u = lu(a, p_indices=True)  # 对数组 a 进行 LU 分解，返回置换矩阵 p、下三角矩阵 l、上三角矩阵 u，并返回置换索引
        assert_allclose(p, np.zeros(shape=(4, 5, 1), dtype=int))  # 断言置换矩阵 p 的值与全零数组的形状相同
        assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))  # 断言下三角矩阵 l 的值与全一数组的形状相同
        assert_allclose(u, a)  # 断言上三角矩阵 u 的值与数组 a 的值相同

        a = self.rng.random([4, 5, 1, 1], dtype=np.float32)  # 生成指定形状的随机数组 a，数据类型为 np.float32
        p, l, u = lu(a)  # 对数组 a 进行 LU 分解，返回置换矩阵 p、下三角矩阵 l、上三角矩阵 u
        assert_allclose(p, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))  # 断言置换矩阵 p 的值与全一数组的形状相同
        assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))  # 断言下三角矩阵 l 的值与全一数组的形状相同
        assert_allclose(u, a)  # 断言上三角矩阵 u 的值与数组 a 的值相同

        pl, u = lu(a, permute_l=True)  # 对数组 a 进行 LU 分解，并对下三角矩阵进行置换
        assert_allclose(pl, np.ones(shape=(4, 5, 1, 1), dtype=np.float32))  # 断言置换下三角矩阵 pl 的值与全一数组的形状相同
        assert_allclose(u, a)  # 断言上三角矩阵 u 的值与数组 a 的值相同

        a = self.rng.random([4, 5, 1, 1], dtype=np.float32)*np.complex64(1.j)  # 生成指定形状的随机数组 a，数据类型为 np.float32 的复数类型
        p, l, u = lu(a)  # 对数组 a 进行 LU 分解，返回置换矩阵 p、下三角矩阵 l、上三角矩阵 u
        assert_allclose(p, np.ones(shape=(4, 5, 1, 1), dtype=np.complex64))  # 断言置换矩阵 p 的值与全一数组的形状相同，数据类型为 np.complex64
        assert_allclose(l, np.ones(shape=(4, 5, 1, 1), dtype=np.complex64))  # 断言下三角矩阵 l 的值与全一数组的形状相同，数据类型为 np.complex64
        assert_allclose(u, a)  # 断言上三角矩阵 u 的值与数组 a 的值相同
    # 测试空数组的边缘情况

    # 创建一个空的2维数组
    a = np.empty([0, 0])
    # 对空数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(p, np.empty(shape=(0, 0), dtype=np.float64))
    assert_allclose(l, np.empty(shape=(0, 0), dtype=np.float64))
    assert_allclose(u, np.empty(shape=(0, 0), dtype=np.float64))

    # 创建一个空的2维数组，指定dtype为float16
    a = np.empty([0, 3], dtype=np.float16)
    # 对空数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(p, np.empty(shape=(0, 0), dtype=np.float32))
    assert_allclose(l, np.empty(shape=(0, 0), dtype=np.float32))
    assert_allclose(u, np.empty(shape=(0, 3), dtype=np.float32))

    # 创建一个空的2维数组，指定dtype为complex64
    a = np.empty([3, 0], dtype=np.complex64)
    # 对空数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(p, np.empty(shape=(0,), dtype=int))
    assert_allclose(l, np.empty(shape=(3, 0), dtype=np.complex64))
    assert_allclose(u, np.empty(shape=(0, 0), dtype=np.complex64))

    # 对空数组进行LU分解，指定不同的选项
    p, l, u = lu(a, p_indices=True)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(p, np.empty(shape=(0,), dtype=int))
    assert_allclose(l, np.empty(shape=(3, 0), dtype=np.complex64))
    assert_allclose(u, np.empty(shape=(0, 0), dtype=np.complex64))

    # 对空数组进行LU分解，指定不同的选项
    pl, u = lu(a, permute_l=True)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(pl, np.empty(shape=(3, 0), dtype=np.complex64))
    assert_allclose(u, np.empty(shape=(0, 0), dtype=np.complex64))

    # 创建一个空的3维数组，指定dtype为complex64
    a = np.empty([3, 0, 0], dtype=np.complex64)
    # 对空数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空3维数组一致
    assert_allclose(p, np.empty(shape=(3, 0, 0), dtype=np.float32))
    assert_allclose(l, np.empty(shape=(3, 0, 0), dtype=np.complex64))
    assert_allclose(u, np.empty(shape=(3, 0, 0), dtype=np.complex64))

    # 创建一个空的3维数组
    a = np.empty([0, 0, 3])
    # 对空数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空3维数组一致
    assert_allclose(p, np.empty(shape=(0, 0, 0)))
    assert_allclose(l, np.empty(shape=(0, 0, 0)))
    assert_allclose(u, np.empty(shape=(0, 0, 3)))

    # 使用空数组调用LU分解，预期抛出ValueError异常，且异常信息包含'at least two-dimensional'
    with assert_raises(ValueError, match='at least two-dimensional'):
        lu(np.array([]))

    # 创建一个包含单个空子数组的2维数组
    a = np.array([[]])
    # 对数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空2维数组一致
    assert_allclose(p, np.empty(shape=(0, 0)))
    assert_allclose(l, np.empty(shape=(1, 0)))
    assert_allclose(u, np.empty(shape=(0, 0)))

    # 创建一个包含单个空子数组的3维数组
    a = np.array([[[]]])
    # 对数组进行LU分解
    p, l, u = lu(a)
    # 断言LU分解后的结果与预期的空3维数组一致
    assert_allclose(p, np.empty(shape=(1, 0, 0)))
    assert_allclose(l, np.empty(shape=(1, 1, 0)))
    assert_allclose(u, np.empty(shape=(1, 0, 0)))
class TestLUFactor:
    # 设置测试方法的初始化
    def setup_method(self):
        # 初始化随机数生成器
        self.rng = np.random.default_rng(1682281250228846)

        # 定义各种测试矩阵
        self.a = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        self.ca = np.array([[1, 2, 3], [1, 2, 3], [2, 5j, 6]])
        # 以下矩阵对于检测置换矩阵问题更为稳健
        self.b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.cb = np.array([[1j, 2j, 3j], [4j, 5j, 6j], [7j, 8j, 9j]])

        # 长方形矩阵
        self.hrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]])
        self.chrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8],
                                [9, 10, 12, 12]]) * 1.j

        self.vrect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]])
        self.cvrect = 1.j * np.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9],
                                      [10, 12, 12]])

        # 中等大小的矩阵
        self.med = self.rng.random((30, 40))
        self.cmed = self.rng.random((30, 40)) + 1.j*self.rng.random((30, 40))

    # 辅助方法，测试 lu_factor 的公共功能
    def _test_common_lu_factor(self, data):
        # 执行 lu 分解
        l_and_u1, piv1 = lu_factor(data)
        # 调用 LAPACK 函数获取 lu 分解
        (getrf,) = get_lapack_funcs(("getrf",), (data,))
        l_and_u2, piv2, _ = getrf(data, overwrite_a=False)
        # 检查两种方法得到的结果是否接近
        assert_allclose(l_and_u1, l_and_u2)
        assert_allclose(piv1, piv2)

    # 测试长方形矩阵的 lu 分解
    def test_hrectangular(self):
        self._test_common_lu_factor(self.hrect)

    # 测试竖直长方形矩阵的 lu 分解
    def test_vrectangular(self):
        self._test_common_lu_factor(self.vrect)

    # 测试复数长方形矩阵的 lu 分解
    def test_hrectangular_complex(self):
        self._test_common_lu_factor(self.chrect)

    # 测试复数竖直长方形矩阵的 lu 分解
    def test_vrectangular_complex(self):
        self._test_common_lu_factor(self.cvrect)

    # 测试中等大小矩阵的 lu 分解
    def test_medium1(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common_lu_factor(self.med)

    # 测试复数中等大小矩阵的 lu 分解
    def test_medium1_complex(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common_lu_factor(self.cmed)

    # 检查 lu 分解对于 self.a 的有限性
    def test_check_finite(self):
        p, l, u = lu(self.a, check_finite=False)
        assert_allclose(p @ l @ u, self.a)

    # 简单已知测试
    # Ticket #1458
    def test_simple_known(self):
        # 针对 'C' 和 'F' 两种存储顺序进行测试
        for order in ['C', 'F']:
            A = np.array([[2, 1], [0, 1.]], order=order)
            LU, P = lu_factor(A)
            # 检查 LU 分解结果是否符合预期
            assert_allclose(LU, np.array([[2, 1], [0, 1]]))
            assert_array_equal(P, np.array([0, 1]))

    # 参数化测试，对不同的 m、n 和 dtype 进行组合测试
    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("n", [0, 1, 2])
    @pytest.mark.parametrize('dtype', DTYPES)
    # 定义一个测试方法，用于验证矩阵的形状和数据类型
    def test_shape_dtype(self, m, n,  dtype):
        # 计算 m 和 n 的最小值
        k = min(m, n)

        # 创建一个 m 行 n 列的单位矩阵，指定数据类型为 dtype
        a = np.eye(m, n, dtype=dtype)
        
        # 对 a 进行 LU 分解，lu 是分解后的下三角部分，p 是置换矩阵
        lu, p = lu_factor(a)
        
        # 断言 LU 分解后的下三角矩阵 lu 的形状应为 (m, n)
        assert_equal(lu.shape, (m, n))
        
        # 断言 lu 的数据类型应为指定的 dtype
        assert_equal(lu.dtype, dtype)
        
        # 断言置换矩阵 p 的形状应为 (k,)，其中 k=min(m, n)
        assert_equal(p.shape, (k,))
        
        # 断言置换矩阵 p 的数据类型应为 np.int32
        assert_equal(p.dtype, np.int32)

    # 使用 pytest 的参数化装饰器，定义一个测试空矩阵的方法
    @pytest.mark.parametrize(("m", "n"), [(0, 0), (0, 2), (2, 0)])
    def test_empty(self, m, n):
        # 创建一个 m 行 n 列的全零矩阵 a
        a = np.zeros((m, n))
        
        # 对 a 进行 LU 分解，lu 是分解后的下三角部分，p 是置换矩阵
        lu, p = lu_factor(a)
        
        # 使用 assert_allclose 断言 lu 应为空数组，形状为 (m, n)
        assert_allclose(lu, np.empty((m, n)))
        
        # 使用 assert_allclose 断言置换矩阵 p 应为空数组，长度为 0
        assert_allclose(p, np.arange(0))
# 定义一个名为 TestLUSolve 的测试类，用于测试 LU 分解和解方程的功能
class TestLUSolve:
    
    # 设置每个测试方法执行前的初始化方法
    def setup_method(self):
        # 使用指定种子初始化随机数生成器 rng
        self.rng = np.random.default_rng(1682281250228846)

    # 测试 LU 分解和解方程的功能
    def test_lu(self):
        # 生成一个 10x10 的随机数组 a0
        a0 = self.rng.random((10, 10))
        # 生成一个长度为 10 的随机数组 b
        b = self.rng.random((10,))

        # 对于两种不同的存储顺序 ['C', 'F']
        for order in ['C', 'F']:
            # 根据指定的存储顺序创建数组 a
            a = np.array(a0, order=order)
            # 使用 solve 函数解方程 a*x = b，返回解 x1
            x1 = solve(a, b)
            # 对数组 a 进行 LU 分解，返回 LU 分解结果 lu_a
            lu_a = lu_factor(a)
            # 使用 lu_solve 函数根据 LU 分解结果解方程，返回解 x2
            x2 = lu_solve(lu_a, b)
            # 断言 x1 和 x2 在数值上相近
            assert_allclose(x1, x2)

    # 测试在非有限检查下的 LU 分解和解方程的功能
    def test_check_finite(self):
        # 生成一个 10x10 的随机数组 a
        a = self.rng.random((10, 10))
        # 生成一个长度为 10 的随机数组 b
        b = self.rng.random((10,))
        # 使用 solve 函数解方程 a*x = b，返回解 x1
        x1 = solve(a, b)
        # 对数组 a 进行不进行有限性检查的 LU 分解，返回 LU 分解结果 lu_a
        lu_a = lu_factor(a, check_finite=False)
        # 使用不进行有限性检查的 lu_solve 函数根据 LU 分解结果解方程，返回解 x2
        x2 = lu_solve(lu_a, b, check_finite=False)
        # 断言 x1 和 x2 在数值上相近
        assert_allclose(x1, x2)

    # 使用参数化测试框架对空输入的 LU 分解和解方程进行测试
    @pytest.mark.parametrize('dt', [int, float, np.float32, complex, np.complex64])
    @pytest.mark.parametrize('dt_b', [int, float, np.float32, complex, np.complex64])
    def test_empty(self, dt, dt_b):
        # 生成一个空的 LU 分解和置换元组 lu_and_piv
        lu_and_piv = (np.empty((0, 0), dtype=dt), np.array([]))
        # 生成一个空的数组 b，并指定数据类型为 dt_b
        b = np.asarray([], dtype=dt_b)
        # 使用 lu_solve 函数对空的 LU 分解和置换 lu_and_piv 解方程，返回解 x
        x = lu_solve(lu_and_piv, b)
        # 断言解 x 的形状为 (0,)
        assert x.shape == (0,)
        
        # 对单位矩阵的 LU 分解结果进行解方程，返回解 m
        m = lu_solve((np.eye(2, dtype=dt), [0, 1]), np.ones(2, dtype=dt_b))
        # 断言解 x 的数据类型与解 m 相同
        assert x.dtype == m.dtype

        # 生成一个空的数组 b，并指定数据类型为 dt_b
        b = np.empty((0, 0), dtype=dt_b)
        # 使用 lu_solve 函数对空的 LU 分解和置换 lu_and_piv 解方程，返回解 x
        x = lu_solve(lu_and_piv, b)
        # 断言解 x 的形状为 (0, 0)
        assert x.shape == (0, 0)
        # 断言解 x 的数据类型与解 m 相同
        assert x.dtype == m.dtype
```