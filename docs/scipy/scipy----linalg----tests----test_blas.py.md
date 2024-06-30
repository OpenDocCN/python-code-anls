# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_blas.py`

```
# 导入所需的模块和库
import math  # 导入数学模块
import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import (assert_equal, assert_almost_equal, assert_,  # 从 NumPy 测试模块导入断言方法
                           assert_array_almost_equal, assert_allclose)
from pytest import raises as assert_raises  # 从 pytest 导入 raises 方法，并使用 assert_raises 别名

from numpy import float32, float64, complex64, complex128, arange, triu,  # 导入多个 NumPy 对象和函数
                  tril, zeros, tril_indices, ones, mod, diag, append, eye, \
                  nonzero

from numpy.random import rand, seed  # 从 NumPy 随机模块导入 rand 和 seed 方法
import scipy  # 导入 SciPy 库
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve  # 从 SciPy linalg 模块导入指定函数

try:
    from scipy.linalg import _cblas as cblas  # 尝试导入 _cblas 模块
except ImportError:
    cblas = None  # 如果导入失败，设定 cblas 为 None

REAL_DTYPES = [float32, float64]  # 定义实数类型的列表
COMPLEX_DTYPES = [complex64, complex128]  # 定义复数类型的列表
DTYPES = REAL_DTYPES + COMPLEX_DTYPES  # 合并实数和复数类型的列表


def test_get_blas_funcs():
    # 测试 get_blas_funcs 函数的返回结果

    # 检查它对于 Fortran 排序的数组返回 Fortran 代码
    f1, f2, f3 = get_blas_funcs(
        ('axpy', 'axpy', 'axpy'),
        (np.empty((2, 2), dtype=np.complex64, order='F'),
         np.empty((2, 2), dtype=np.complex128, order='C'))
        )

    # 根据最通用的数组，get_blas_funcs 将选择不同的库
    assert_equal(f1.typecode, 'z')  # 检查返回结果的类型码
    assert_equal(f2.typecode, 'z')  # 检查返回结果的类型码
    if cblas is not None:
        assert_equal(f1.module_name, 'cblas')  # 如果 cblas 可用，检查模块名称
        assert_equal(f2.module_name, 'cblas')  # 如果 cblas 可用，检查模块名称

    # 检查默认情况
    f1 = get_blas_funcs('rotg')
    assert_equal(f1.typecode, 'd')  # 检查默认情况下返回结果的类型码

    # 检查 dtype 接口
    f1 = get_blas_funcs('gemm', dtype=np.complex64)
    assert_equal(f1.typecode, 'c')  # 检查指定 dtype 后返回结果的类型码
    f1 = get_blas_funcs('gemm', dtype='F')
    assert_equal(f1.typecode, 'c')  # 检查指定 dtype 后返回结果的类型码

    # 扩展精度复数
    f1 = get_blas_funcs('gemm', dtype=np.clongdouble)
    assert_equal(f1.typecode, 'z')  # 检查指定 dtype 后返回结果的类型码

    # 检查安全的复数类型转换
    f1 = get_blas_funcs('axpy',
                        (np.empty((2, 2), dtype=np.float64),
                         np.empty((2, 2), dtype=np.complex64))
                        )
    assert_equal(f1.typecode, 'z')  # 检查复数类型转换后返回结果的类型码


def test_get_blas_funcs_alias():
    # 检查 get_blas_funcs 的别名

    # 检查复数64位别名
    f, g = get_blas_funcs(('nrm2', 'dot'), dtype=np.complex64)
    assert f.typecode == 'c'  # 检查返回结果的类型码
    assert g.typecode == 'c'  # 检查返回结果的类型码

    # 检查64位浮点数别名
    f, g, h = get_blas_funcs(('dot', 'dotc', 'dotu'), dtype=np.float64)
    assert f is g  # 检查两个结果对象是否相同
    assert f is h  # 检查两个结果对象是否相同


class TestCBLAS1Simple:

    def test_axpy(self):
        # 测试 cblas 中的 axpy 函数

        # 测试单精度和双精度实数类型
        for p in 'sd':
            f = getattr(cblas, p+'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2, 3], [2, -1, 3], a=5),
                                      [7, 9, 18])  # 检查 axpy 函数的结果是否准确

        # 测试单精度和双精度复数类型
        for p in 'cz':
            f = getattr(cblas, p+'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2j, 3], [2, -1, 3], a=5),
                                      [7, 10j-1, 18])  # 检查 axpy 函数的结果是否准确


class TestFBLAS1Simple:

    # TestFBLAS1Simple 类的定义，用于测试 FBLAS1 相关功能，暂无代码
    pass
    # 对于每个数据类型（单精度浮点数、双精度浮点数），执行axpy（a * x + y）操作的测试
    def test_axpy(self):
        for p in 'sd':
            # 获取对应数据类型的axpy函数，如果不存在则跳过
            f = getattr(fblas, p+'axpy', None)
            if f is None:
                continue
            # 断言axpy函数的返回结果与预期结果的数组几乎相等
            assert_array_almost_equal(f([1, 2, 3], [2, -1, 3], a=5),
                                      [7, 9, 18])
        for p in 'cz':
            # 获取对应数据类型的axpy函数，如果不存在则跳过
            f = getattr(fblas, p+'axpy', None)
            if f is None:
                continue
            # 断言axpy函数的返回结果与预期结果的数组几乎相等
            assert_array_almost_equal(f([1, 2j, 3], [2, -1, 3], a=5),
                                      [7, 10j-1, 18])

    # 测试复制操作的函数
    def test_copy(self):
        for p in 'sd':
            # 获取对应数据类型的copy函数，如果不存在则跳过
            f = getattr(fblas, p+'copy', None)
            if f is None:
                continue
            # 断言copy函数的返回结果与预期结果的数组几乎相等
            assert_array_almost_equal(f([3, 4, 5], [8]*3), [3, 4, 5])
        for p in 'cz':
            # 获取对应数据类型的copy函数，如果不存在则跳过
            f = getattr(fblas, p+'copy', None)
            if f is None:
                continue
            # 断言copy函数的返回结果与预期结果的数组几乎相等
            assert_array_almost_equal(f([3, 4j, 5+3j], [8]*3), [3, 4j, 5+3j])

    # 测试向量的绝对值和操作的函数
    def test_asum(self):
        for p in 'sd':
            # 获取对应数据类型的asum函数，如果不存在则跳过
            f = getattr(fblas, p+'asum', None)
            if f is None:
                continue
            # 断言asum函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3, -4, 5]), 12)
        for p in ['sc', 'dz']:
            # 获取对应数据类型的asum函数，如果不存在则跳过
            f = getattr(fblas, p+'asum', None)
            if f is None:
                continue
            # 断言asum函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3j, -4, 3-4j]), 14)

    # 测试向量点积（内积）操作的函数
    def test_dot(self):
        for p in 'sd':
            # 获取对应数据类型的dot函数，如果不存在则跳过
            f = getattr(fblas, p+'dot', None)
            if f is None:
                continue
            # 断言dot函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3, -4, 5], [2, 5, 1]), -9)

    # 测试复数向量的dotu（未共轭点积）操作的函数
    def test_complex_dotu(self):
        for p in 'cz':
            # 获取对应数据类型的dotu函数，如果不存在则跳过
            f = getattr(fblas, p+'dotu', None)
            if f is None:
                continue
            # 断言dotu函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3j, -4, 3-4j], [2, 3, 1]), -9+2j)

    # 测试复数向量的dotc（共轭点积）操作的函数
    def test_complex_dotc(self):
        for p in 'cz':
            # 获取对应数据类型的dotc函数，如果不存在则跳过
            f = getattr(fblas, p+'dotc', None)
            if f is None:
                continue
            # 断言dotc函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3j, -4, 3-4j], [2, 3j, 1]), 3-14j)

    # 测试向量的二范数（欧几里得范数）操作的函数
    def test_nrm2(self):
        for p in 'sd':
            # 获取对应数据类型的nrm2函数，如果不存在则跳过
            f = getattr(fblas, p+'nrm2', None)
            if f is None:
                continue
            # 断言nrm2函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3, -4, 5]), math.sqrt(50))
        for p in ['c', 'z', 'sc', 'dz']:
            # 获取对应数据类型的nrm2函数，如果不存在则跳过
            f = getattr(fblas, p+'nrm2', None)
            if f is None:
                continue
            # 断言nrm2函数的返回结果与预期结果几乎相等
            assert_almost_equal(f([3j, -4, 3-4j]), math.sqrt(50))
    # 测试函数，用于检查不同类型（'sd', 'cz', 'cs', 'zd'）的BLAS函数是否正常工作
    def test_scal(self):
        # 对于'sd'类型，获取对应的函数并进行测试
        for p in 'sd':
            f = getattr(fblas, p+'scal', None)
            if f is None:
                continue
            # 断言对于给定的参数，函数执行后的结果与预期结果相近
            assert_array_almost_equal(f(2, [3, -4, 5]), [6, -8, 10])
        # 对于'cz'类型，获取对应的函数并进行测试
        for p in 'cz':
            f = getattr(fblas, p+'scal', None)
            if f is None:
                continue
            # 断言对于给定的参数，函数执行后的结果与预期结果相近
            assert_array_almost_equal(f(3j, [3j, -4, 3-4j]), [-9, -12j, 12+9j])
        # 对于'cs'和'zd'类型，获取对应的函数并进行测试
        for p in ['cs', 'zd']:
            f = getattr(fblas, p+'scal', None)
            if f is None:
                continue
            # 断言对于给定的参数，函数执行后的结果与预期结果相近
            assert_array_almost_equal(f(3, [3j, -4, 3-4j]), [9j, -12, 9-12j])

    # 测试函数，用于检查不同类型（'sd', 'cz'）的BLAS函数是否正常工作
    def test_swap(self):
        # 对于'sd'类型，获取对应的函数并进行测试
        for p in 'sd':
            f = getattr(fblas, p+'swap', None)
            if f is None:
                continue
            # 设置初始数组，并使用函数进行交换操作
            x, y = [2, 3, 1], [-2, 3, 7]
            x1, y1 = f(x, y)
            # 断言交换后的结果与预期结果相近
            assert_array_almost_equal(x1, y)
            assert_array_almost_equal(y1, x)
        # 对于'cz'类型，获取对应的函数并进行测试
        for p in 'cz':
            f = getattr(fblas, p+'swap', None)
            if f is None:
                continue
            # 设置初始数组，并使用函数进行交换操作
            x, y = [2, 3j, 1], [-2, 3, 7-3j]
            x1, y1 = f(x, y)
            # 断言交换后的结果与预期结果相近
            assert_array_almost_equal(x1, y)
            assert_array_almost_equal(y1, x)

    # 测试函数，用于检查不同类型（'sd', 'cz'）的BLAS函数是否正常工作
    def test_amax(self):
        # 对于'sd'类型，获取对应的函数并进行测试
        for p in 'sd':
            f = getattr(fblas, 'i'+p+'amax')
            # 断言对于给定的数组，函数执行后返回的最大元素的索引与预期索引相等
            assert_equal(f([-2, 4, 3]), 1)
        # 对于'cz'类型，获取对应的函数并进行测试
        for p in 'cz':
            f = getattr(fblas, 'i'+p+'amax')
            # 断言对于给定的数组，函数执行后返回的最大元素的索引与预期索引相等
            assert_equal(f([-5, 4+3j, 6]), 1)
    # XXX: need tests for rot,rotm,rotg,rotmg
class TestFBLAS2Simple:

    def test_gemv(self):
        # 遍历字符 's' 和 'd'
        for p in 'sd':
            # 获取 fblas 模块中名为 p+'gemv' 的函数，如果不存在则跳过
            f = getattr(fblas, p+'gemv', None)
            if f is None:
                continue
            # 断言调用 f 函数返回的结果与预期值几乎相等
            assert_array_almost_equal(f(3, [[3]], [-4]), [-36])
            assert_array_almost_equal(f(3, [[3]], [-4], 3, [5]), [-21])
        
        # 遍历字符 'c' 和 'z'
        for p in 'cz':
            # 获取 fblas 模块中名为 p+'gemv' 的函数，如果不存在则跳过
            f = getattr(fblas, p+'gemv', None)
            if f is None:
                continue
            # 断言调用 f 函数返回的结果与预期值几乎相等
            assert_array_almost_equal(f(3j, [[3-4j]], [-4]), [-48-36j])
            assert_array_almost_equal(f(3j, [[3-4j]], [-4], 3, [5j]),
                                      [-48-21j])

    def test_ger(self):
        # 遍历字符 's' 和 'd'
        for p in 'sd':
            # 获取 fblas 模块中名为 p+'ger' 的函数，如果不存在则跳过
            f = getattr(fblas, p+'ger', None)
            if f is None:
                continue
            # 断言调用 f 函数返回的结果与预期值几乎相等
            assert_array_almost_equal(f(1, [1, 2], [3, 4]), [[3, 4], [6, 8]])
            assert_array_almost_equal(f(2, [1, 2, 3], [3, 4]),
                                      [[6, 8], [12, 16], [18, 24]])

            # 断言调用 f 函数返回的结果与预期值几乎相等，同时传递额外参数 a
            assert_array_almost_equal(f(1, [1, 2], [3, 4],
                                        a=[[1, 2], [3, 4]]), [[4, 6], [9, 12]])

        # 遍历字符 'c' 和 'z'
        for p in 'cz':
            # 获取 fblas 模块中名为 p+'geru' 的函数，如果不存在则跳过
            f = getattr(fblas, p+'geru', None)
            if f is None:
                continue
            # 断言调用 f 函数返回的结果与预期值几乎相等
            assert_array_almost_equal(f(1, [1j, 2], [3, 4]),
                                      [[3j, 4j], [6, 8]])
            assert_array_almost_equal(f(-2, [1j, 2j, 3j], [3j, 4j]),
                                      [[6, 8], [12, 16], [18, 24]])

        # 再次遍历字符 'c' 和 'z'，以及名称 'ger' 和 'gerc'
        for p in 'cz':
            for name in ('ger', 'gerc'):
                # 获取 fblas 模块中名为 p+name 的函数，如果不存在则跳过
                f = getattr(fblas, p+name, None)
                if f is None:
                    continue
                # 断言调用 f 函数返回的结果与预期值几乎相等
                assert_array_almost_equal(f(1, [1j, 2], [3, 4]),
                                          [[3j, 4j], [6, 8]])
                assert_array_almost_equal(f(2, [1j, 2j, 3j], [3j, 4j]),
                                          [[6, 8], [12, 16], [18, 24]])
    # 定义一个测试方法 test_syr2，用于测试特定的数学运算
    def test_syr2(self):
        # 创建一个包含1到4的浮点数数组 x
        x = np.arange(1, 5, dtype='d')
        # 创建一个包含5到8的浮点数数组 y
        y = np.arange(5, 9, dtype='d')
        # 计算 resxy，它是 x 和 y 的外积的上三角部分
        resxy = np.triu(x[:, np.newaxis] * y + y[:, np.newaxis] * x)
        # 计算 resxy_reverse，它是 x 和 y 反向数组的外积的上三角部分
        resxy_reverse = np.triu(x[::-1, np.newaxis] * y[::-1]
                                + y[::-1, np.newaxis] * x[::-1])

        # 创建一个从0到8.5的间隔为0.5的数组 q
        q = np.linspace(0, 8.5, 17, endpoint=False)

        # 迭代遍历字符 's' 和 'd' 以及它们对应的相对误差 rtol
        for p, rtol in zip('sd', [1e-7, 1e-14]):
            # 获取 fblas 模块中名为 psyr2 或 dsyr2 的函数，若不存在则跳过
            f = getattr(fblas, p+'syr2', None)
            if f is None:
                continue
            # 断言调用 f 函数计算结果与 resxy 在相对误差 rtol 范围内相等
            assert_allclose(f(1.0, x, y), resxy, rtol=rtol)
            # 断言调用 f 函数计算结果的前3x3部分与 resxy 的前3x3部分在相对误差 rtol 范围内相等
            assert_allclose(f(1.0, x, y, n=3), resxy[:3, :3], rtol=rtol)
            # 断言调用 f 函数计算结果的下三角部分转置与 resxy 在相对误差 rtol 范围内相等
            assert_allclose(f(1.0, x, y, lower=True), resxy.T, rtol=rtol)

            # 断言调用 f 函数计算结果与 resxy 在相对误差 rtol 范围内相等，
            # 使用自定义的增量和偏移参数 incx、offx、incy、offy
            assert_allclose(f(1.0, q, q, incx=2, offx=2, incy=2, offy=10),
                            resxy, rtol=rtol)
            # 断言调用 f 函数计算结果的前3x3部分与 resxy 的前3x3部分在相对误差 rtol 范围内相等，
            # 使用自定义的增量和偏移参数 incx、offx、incy、offy
            assert_allclose(f(1.0, q, q, incx=2, offx=2, incy=2, offy=10, n=3),
                            resxy[:3, :3], rtol=rtol)
            # 断言调用 f 函数计算结果与 resxy_reverse 在相对误差 rtol 范围内相等，
            # 使用负增量参数导致反向向量计算
            assert_allclose(f(1.0, q, q, incx=-2, offx=2, incy=-2, offy=10),
                            resxy_reverse, rtol=rtol)

            # 创建一个具有0填充的4x4浮点数数组 a，其类型根据 p 的值确定是 'f' 还是 'd'
            a = np.zeros((4, 4), 'f' if p == 's' else 'd', 'F')
            # 调用 f 函数，计算结果存储在数组 a 中，覆盖原有内容
            b = f(1.0, x, y, a=a, overwrite_a=True)
            # 断言数组 a 的值与 resxy 在相对误差 rtol 范围内相等
            assert_allclose(a, resxy, rtol=rtol)

            # 调用 f 函数，计算结果存储在数组 a 中，不覆盖原有内容
            b = f(2.0, x, y, a=a)
            # 断言数组 a 与 b 不是同一个对象
            assert_(a is not b)
            # 断言数组 b 的值与 3*resxy 在相对误差 rtol 范围内相等
            assert_allclose(b, 3*resxy, rtol=rtol)

            # 断言调用 f 函数时传入的参数 incx=0 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, incx=0)
            # 断言调用 f 函数时传入的参数 offx=5 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, offx=5)
            # 断言调用 f 函数时传入的参数 offx=-2 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, offx=-2)
            # 断言调用 f 函数时传入的参数 incy=0 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, incy=0)
            # 断言调用 f 函数时传入的参数 offy=5 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, offy=5)
            # 断言调用 f 函数时传入的参数 offy=-2 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, offy=-2)
            # 断言调用 f 函数时传入的参数 n=-2 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, n=-2)
            # 断言调用 f 函数时传入的参数 n=5 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, n=5)
            # 断言调用 f 函数时传入的参数 lower=2 会抛出异常
            assert_raises(Exception, f, 1.0, x, y, lower=2)
            # 断言调用 f 函数时传入的参数 a 的形状不匹配会抛出异常
            assert_raises(Exception, f, 1.0, x, y,
                          a=np.zeros((2, 2), 'd', 'F'))
    # 定义一个测试方法，用于测试特定的数学运算
    def test_her2(self):
        # 创建一个包含复数的NumPy数组x，包含1到8的元素
        x = np.arange(1, 9, dtype='d').view('D')
        # 创建另一个包含复数的NumPy数组y，包含9到16的元素
        y = np.arange(9, 17, dtype='d').view('D')
        # 计算resxy矩阵，包含x和y的乘积及共轭乘积的和，然后取上三角部分
        resxy = x[:, np.newaxis] * y.conj() + y[:, np.newaxis] * x.conj()
        resxy = np.triu(resxy)

        # 计算resxy_reverse矩阵，包含x和y的倒序乘积及共轭乘积的和，然后取上三角部分
        resxy_reverse = x[::-1, np.newaxis] * y[::-1].conj()
        resxy_reverse += y[::-1, np.newaxis] * x[::-1].conj()
        resxy_reverse = np.triu(resxy_reverse)

        # 创建向量u和v，分别由x和y组成，并在中间插入四个零
        u = np.c_[np.zeros(4), x, np.zeros(4)].ravel()
        v = np.c_[np.zeros(4), y, np.zeros(4)].ravel()

        # 遍历字符串'cz'，并依次计算对应的fblas函数
        for p, rtol in zip('cz', [1e-7, 1e-14]):
            # 获取名为p+'her2'的函数对象，如果不存在则继续下一个循环
            f = getattr(fblas, p+'her2', None)
            if f is None:
                continue
            # 检查函数f对x和y的计算结果是否与resxy矩阵接近，指定允许误差rtol
            assert_allclose(f(1.0, x, y), resxy, rtol=rtol)
            # 再次检查f对x和y的计算结果是否与resxy矩阵的前3x3部分接近，指定允许误差rtol
            assert_allclose(f(1.0, x, y, n=3), resxy[:3, :3], rtol=rtol)
            # 检查f对x和y的计算结果是否与resxy转置后的共轭矩阵接近，指定允许误差rtol
            assert_allclose(f(1.0, x, y, lower=True), resxy.T.conj(),
                            rtol=rtol)

            # 检查f对u和v的计算结果是否与resxy矩阵接近，使用指定的偏移和增量
            assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1),
                            resxy, rtol=rtol)
            # 再次检查f对u和v的计算结果是否与resxy矩阵的前3x3部分接近，使用指定的偏移和增量
            assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1, n=3),
                            resxy[:3, :3], rtol=rtol)
            # 检查f对u和v的计算结果是否与resxy_reverse矩阵接近，使用负增量和指定的偏移
            assert_allclose(f(1.0, u, v, incx=-3, offx=1, incy=-3, offy=1),
                            resxy_reverse, rtol=rtol)

            # 创建一个4x4的零矩阵a，数据类型为'F'或者'D'，具体类型由p决定，然后调用f函数
            a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
            # 调用f函数，将结果存入矩阵a中，并检查a与resxy矩阵的接近程度
            b = f(1.0, x, y, a=a, overwrite_a=True)
            assert_allclose(a, resxy, rtol=rtol)

            # 调用f函数，使用不同的参数，并检查返回的矩阵b与a不同
            b = f(2.0, x, y, a=a)
            assert_(a is not b)
            # 再次检查b与3倍的resxy矩阵的接近程度
            assert_allclose(b, 3*resxy, rtol=rtol)

            # 检查调用f函数时传入不合法参数是否会引发异常
            assert_raises(Exception, f, 1.0, x, y, incx=0)
            assert_raises(Exception, f, 1.0, x, y, offx=5)
            assert_raises(Exception, f, 1.0, x, y, offx=-2)
            assert_raises(Exception, f, 1.0, x, y, incy=0)
            assert_raises(Exception, f, 1.0, x, y, offy=5)
            assert_raises(Exception, f, 1.0, x, y, offy=-2)
            assert_raises(Exception, f, 1.0, x, y, n=-2)
            assert_raises(Exception, f, 1.0, x, y, n=5)
            assert_raises(Exception, f, 1.0, x, y, lower=2)
            assert_raises(Exception, f, 1.0, x, y,
                          a=np.zeros((2, 2), 'd', 'F'))
    def test_gbmv(self):
        # 设置随机数种子，确保结果可重现性
        seed(1234)
        # 遍历不同数据类型
        for ind, dtype in enumerate(DTYPES):
            # 定义矩阵的行数和列数
            n = 7
            m = 5
            # 上下带宽
            kl = 1
            ku = 2
            
            # 通过Toeplitz矩阵创建带状矩阵
            A = toeplitz(append(rand(kl+1), zeros(m-kl-1)),
                         append(rand(ku+1), zeros(n-ku-1)))
            A = A.astype(dtype)
            # 初始化带状存储的数组
            Ab = zeros((kl+ku+1, n), dtype=dtype)

            # 填充带状存储数组
            Ab[2, :5] = A[0, 0]  # 对角线
            Ab[1, 1:6] = A[0, 1]  # 上对角线1
            Ab[0, 2:7] = A[0, 2]  # 上对角线2
            Ab[3, :4] = A[1, 0]  # 下对角线1

            # 随机生成向量
            x = rand(n).astype(dtype)
            y = rand(m).astype(dtype)
            # 设置alpha和beta的值
            alpha, beta = dtype(3), dtype(-5)

            # 获取BLAS库中的gbmv函数
            func, = get_blas_funcs(('gbmv',), dtype=dtype)
            # 调用gbmv函数计算结果y1
            y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab,
                      x=x, y=y, beta=beta)
            # 计算期望结果y2
            y2 = alpha * A.dot(x) + beta * y
            # 使用assert检查结果的准确性
            assert_array_almost_equal(y1, y2)

            # 另一种调用gbmv函数的方式，计算结果y1
            y1 = func(m=m, n=n, ku=ku, kl=kl, alpha=alpha, a=Ab,
                      x=y, y=x, beta=beta, trans=1)
            # 计算期望结果y2
            y2 = alpha * A.T.dot(y) + beta * x
            # 使用assert检查结果的准确性
            assert_array_almost_equal(y1, y2)

    def test_sbmv_hbmv(self):
        # 设置随机数种子，确保结果可重现性
        seed(1234)
        # 遍历不同数据类型
        for ind, dtype in enumerate(DTYPES):
            # 矩阵的维度
            n = 6
            # 带宽
            k = 2
            # 初始化矩阵A和带状存储的数组Ab
            A = zeros((n, n), dtype=dtype)
            Ab = zeros((k+1, n), dtype=dtype)

            # 填充数组A及其带状存储的Ab
            A[arange(n), arange(n)] = rand(n)
            for ind2 in range(1, k+1):
                temp = rand(n-ind2)
                A[arange(n-ind2), arange(ind2, n)] = temp
                Ab[-1-ind2, ind2:] = temp
            A = A.astype(dtype)
            # 根据情况处理A的共轭转置或转置
            A = A + A.T if ind < 2 else A + A.conj().T
            Ab[-1, :] = diag(A)

            # 随机生成向量
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            # 设置alpha和beta的值
            alpha, beta = dtype(1.25), dtype(3)

            # 根据ind的值选择使用hbmv或sbmv函数
            if ind > 1:
                func, = get_blas_funcs(('hbmv',), dtype=dtype)
            else:
                func, = get_blas_funcs(('sbmv',), dtype=dtype)
            # 调用函数计算结果y1
            y1 = func(k=k, alpha=alpha, a=Ab, x=x, y=y, beta=beta)
            # 计算期望结果y2
            y2 = alpha * A.dot(x) + beta * y
            # 使用assert检查结果的准确性
            assert_array_almost_equal(y1, y2)
    # 定义名为 test_spmv_hpmv 的测试函数，用于测试稠密和稀疏 BLAS 级数乘法的正确性
    def test_spmv_hpmv(self):
        # 设置随机种子为 1234，确保测试结果可重现
        seed(1234)
        # 对于 DTYPES 和 COMPLEX_DTYPES 中的每个数据类型进行测试
        for ind, dtype in enumerate(DTYPES+COMPLEX_DTYPES):
            n = 3
            # 生成一个 n x n 的随机数组 A，并将其转换为指定数据类型 dtype
            A = rand(n, n).astype(dtype)
            # 如果数据类型索引大于 1，将 A 的虚部设为随机数
            if ind > 1:
                A += rand(n, n)*1j
            # 再次确保 A 的数据类型为 dtype
            A = A.astype(dtype)
            # 根据索引值选择加法类型：对称或共轭对称加法
            A = A + A.T if ind < 4 else A + A.conj().T
            # 生成下三角矩阵的行列索引
            c, r = tril_indices(n)
            # 提取下三角部分形成 Ap
            Ap = A[r, c]
            # 生成长度为 n 的随机数组 x 和 y，并将它们转换为指定数据类型 dtype
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            # 生成长度为 2*n 的数组 xlong 和 ylong，并将其转换为指定数据类型 dtype
            xlong = arange(2*n).astype(dtype)
            ylong = ones(2*n).astype(dtype)
            # 设定 alpha 和 beta 的值为指定数据类型的常数
            alpha, beta = dtype(1.25), dtype(2)

            # 根据索引值选择 BLAS 函数：hpmv 或 spmv，并赋值给 func
            if ind > 3:
                func, = get_blas_funcs(('hpmv',), dtype=dtype)
            else:
                func, = get_blas_funcs(('spmv',), dtype=dtype)
            # 计算 BLAS 函数的结果 y1
            y1 = func(n=n, alpha=alpha, ap=Ap, x=x, y=y, beta=beta)
            # 计算标准结果 y2
            y2 = alpha * A.dot(x) + beta * y
            # 断言 y1 与 y2 的值近似相等
            assert_array_almost_equal(y1, y2)

            # 测试增量和偏移量功能
            # 计算 BLAS 函数的结果 y1，带有自定义的增量和偏移量
            y1 = func(n=n-1, alpha=alpha, beta=beta, x=xlong, y=ylong, ap=Ap,
                      incx=2, incy=2, offx=n, offy=n)
            # 计算标准结果 y2
            y2 = (alpha * A[:-1, :-1]).dot(xlong[3::2]) + beta * ylong[3::2]
            # 断言 y1 的部分值与 y2 的部分值近似相等
            assert_array_almost_equal(y1[3::2], y2)
            # 断言 y1 的另一个特定索引处的值近似相等
            assert_almost_equal(y1[4], ylong[4])

    # 定义名为 test_spr_hpr 的测试函数，用于测试稠密和稀疏 BLAS 矩阵向量乘法的正确性
    def test_spr_hpr(self):
        # 设置随机种子为 1234，确保测试结果可重现
        seed(1234)
        # 对于 DTYPES 和 COMPLEX_DTYPES 中的每个数据类型进行测试
        for ind, dtype in enumerate(DTYPES+COMPLEX_DTYPES):
            n = 3
            # 生成一个 n x n 的随机数组 A，并将其转换为指定数据类型 dtype
            A = rand(n, n).astype(dtype)
            # 如果数据类型索引大于 1，将 A 的虚部设为随机数
            if ind > 1:
                A += rand(n, n)*1j
            # 再次确保 A 的数据类型为 dtype
            A = A.astype(dtype)
            # 根据索引值选择加法类型：对称或共轭对称加法
            A = A + A.T if ind < 4 else A + A.conj().T
            # 生成下三角矩阵的行列索引
            c, r = tril_indices(n)
            # 提取下三角部分形成 Ap
            Ap = A[r, c]
            # 生成长度为 n 的随机数组 x，并将其转换为指定数据类型 dtype
            x = rand(n).astype(dtype)
            # 从 DTYPES 和 COMPLEX_DTYPES 中选择一个数据类型作为 alpha 的值
            alpha = (DTYPES+COMPLEX_DTYPES)[mod(ind, 4)](2.5)

            # 根据索引值选择 BLAS 函数：hpr 或 spr，并赋值给 func
            if ind > 3:
                func, = get_blas_funcs(('hpr',), dtype=dtype)
                # 计算标准结果 y2
                y2 = alpha * x[:, None].dot(x[None, :].conj()) + A
            else:
                func, = get_blas_funcs(('spr',), dtype=dtype)
                # 计算标准结果 y2
                y2 = alpha * x[:, None].dot(x[None, :]) + A

            # 计算 BLAS 函数的结果 y1
            y1 = func(n=n, alpha=alpha, ap=Ap, x=x)
            # 创建一个全零数组 y1f，将 y1 的值填充到对称位置上
            y1f = zeros((3, 3), dtype=dtype)
            y1f[r, c] = y1
            # 如果数据类型索引大于 3，则将 y1 的共轭值填充到对称位置上，否则直接填充 y1
            y1f[c, r] = y1.conj() if ind > 3 else y1
            # 断言 y1f 与 y2 的值近似相等
            assert_array_almost_equal(y1f, y2)
    # 定义测试函数 test_spr2_hpr2，用于测试特定的线性代数函数
    def test_spr2_hpr2(self):
        # 设置随机种子以确保每次测试结果一致
        seed(1234)
        # 遍历指定的数据类型列表 DTYPES
        for ind, dtype in enumerate(DTYPES):
            n = 3
            # 生成一个 n x n 的随机数组 A，并将其类型转换为指定的数据类型 dtype
            A = rand(n, n).astype(dtype)
            # 若索引值大于 1，则将 A 的虚部加到实部上
            if ind > 1:
                A += rand(n, n) * 1j
            # 再次将 A 转换为指定的数据类型 dtype
            A = A.astype(dtype)
            # 根据索引值选择不同的操作：若小于 2，则 A 与其转置矩阵相加；否则 A 与其共轭转置矩阵相加
            A = A + A.T if ind < 2 else A + A.conj().T
            # 获取下三角矩阵的索引
            c, r = tril_indices(n)
            # 根据索引提取 A 的下三角部分形成 Ap
            Ap = A[r, c]
            # 生成随机向量 x 和 y，并将其类型转换为指定的数据类型 dtype
            x = rand(n).astype(dtype)
            y = rand(n).astype(dtype)
            # 设置 alpha 为指定的数据类型 dtype，并赋值为 2
            alpha = dtype(2)

            # 根据索引值选择不同的 BLAS 函数：若大于 1，则选择 'hpr2' 函数；否则选择 'spr2' 函数
            if ind > 1:
                func, = get_blas_funcs(('hpr2',), dtype=dtype)
            else:
                func, = get_blas_funcs(('spr2',), dtype=dtype)

            # 计算 u，其中 u = alpha 的共轭乘积与 x 的转置矩阵乘以 y 的共轭
            u = alpha.conj() * x[:, None].dot(y[None, :].conj())
            # 计算 y2，其中 y2 = A + u + u 的共轭转置
            y2 = A + u + u.conj().T
            # 计算 y1，调用 BLAS 函数计算得到结果
            y1 = func(n=n, alpha=alpha, x=x, y=y, ap=Ap)
            # 创建一个全零矩阵 y1f，其类型为指定的数据类型 dtype，形状为 (3, 3)
            y1f = zeros((3, 3), dtype=dtype)
            # 根据索引 r, c 将 y1 的值填充到 y1f 的对应位置
            y1f[r, c] = y1
            # 将 y1 的部分值（索引 [1, 3, 4]）的共轭填充到 y1f 的指定位置（索引 [1, 2, 2], [0, 0, 1]）
            y1f[[1, 2, 2], [0, 0, 1]] = y1[[1, 3, 4]].conj()
            # 断言 y1f 与 y2 几乎相等
            assert_array_almost_equal(y1f, y2)

    # 定义测试函数 test_tbmv，用于测试特定的 BLAS 函数
    def test_tbmv(self):
        # 设置随机种子以确保每次测试结果一致
        seed(1234)
        # 遍历指定的数据类型列表 DTYPES
        for ind, dtype in enumerate(DTYPES):
            n = 10
            k = 3
            # 生成一个随机向量 x，并将其类型转换为指定的数据类型 dtype
            x = rand(n).astype(dtype)
            # 生成一个全零矩阵 A，其类型为指定的数据类型 dtype，形状为 (n, n)
            A = zeros((n, n), dtype=dtype)
            
            # 创建带状上三角矩阵 A
            for sup in range(k+1):
                A[arange(n-sup), arange(sup, n)] = rand(n-sup)

            # 若索引值大于 1，则给 A 的非零元素添加虚部
            if ind > 1:
                A[nonzero(A)] += 1j * rand((k+1)*n-(k*(k+1)//2)).astype(dtype)

            # 生成带状存储 Ab
            Ab = zeros((k+1, n), dtype=dtype)
            for row in range(k+1):
                Ab[-row-1, row:] = diag(A, k=row)

            # 获取 BLAS 函数 'tbmv'
            func, = get_blas_funcs(('tbmv',), dtype=dtype)

            # 计算 y1，调用 BLAS 函数计算得到结果
            y1 = func(k=k, a=Ab, x=x)
            # 计算 y2，直接用 A 点乘 x 得到结果
            y2 = A.dot(x)
            # 断言 y1 与 y2 几乎相等
            assert_array_almost_equal(y1, y2)

            # 计算 y1，调用 BLAS 函数计算得到结果，同时指定 diag=1
            y1 = func(k=k, a=Ab, x=x, diag=1)
            # 将 A 的对角线元素设置为 1
            A[arange(n), arange(n)] = dtype(1)
            # 计算 y2，直接用 A 点乘 x 得到结果
            y2 = A.dot(x)
            # 断言 y1 与 y2 几乎相等
            assert_array_almost_equal(y1, y2)

            # 计算 y1，调用 BLAS 函数计算得到结果，同时指定 diag=1 和 trans=1
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=1)
            # 计算 y2，直接用 A 的转置点乘 x 得到结果
            y2 = A.T.dot(x)
            # 断言 y1 与 y2 几乎相等
            assert_array_almost_equal(y1, y2)

            # 计算 y1，调用 BLAS 函数计算得到结果，同时指定 diag=1 和 trans=2
            y1 = func(k=k, a=Ab, x=x, diag=1, trans=2)
            # 计算 y2，直接用 A 的共轭转置点乘 x 得到结果
            y2 = A.conj().T.dot(x)
            # 断言 y1 与 y2 几乎相等
            assert_array_almost_equal(y1, y2)
    # 定义一个用于测试 tbsv 函数的测试方法
    def test_tbsv(self):
        # 设置随机数种子
        seed(1234)
        # 遍历不同的数据类型
        for ind, dtype in enumerate(DTYPES):
            n = 6
            k = 3
            # 生成长度为 n 的随机数组并转换为指定数据类型
            x = rand(n).astype(dtype)
            # 创建一个元素全为零的 n x n 的数组，指定数据类型
            A = zeros((n, n), dtype=dtype)
            
            # 创建带状上三角阵列
            for sup in range(k+1):
                A[arange(n-sup), arange(sup, n)] = rand(n-sup)
            
            # 对于复数类型 c,z，添加复数部分
            if ind > 1:
                A[nonzero(A)] += 1j * rand((k+1)*n-(k*(k+1)//2)).astype(dtype)
            
            # 创建带状存储
            Ab = zeros((k+1, n), dtype=dtype)
            for row in range(k+1):
                Ab[-row-1, row:] = diag(A, k=row)
            
            # 获取 BLAS 函数 tbsv
            func, = get_blas_funcs(('tbsv',), dtype=dtype)
            
            # 测试不同参数组合下的 tbsv 函数
            y1 = func(k=k, a=Ab, x=x)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)

            y1 = func(k=k, a=Ab, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)

            y1 = func(k=k, a=Ab, x=x, diag=1, trans=1)
            y2 = solve(A.T, x)
            assert_array_almost_equal(y1, y2)

            y1 = func(k=k, a=Ab, x=x, diag=1, trans=2)
            y2 = solve(A.conj().T, x)
            assert_array_almost_equal(y1, y2)

    # 定义一个用于测试 tpmv 函数的测试方法
    def test_tpmv(self):
        # 设置随机数种子
        seed(1234)
        # 遍历不同的数据类型
        for ind, dtype in enumerate(DTYPES):
            n = 10
            # 生成长度为 n 的随机数组并转换为指定数据类型
            x = rand(n).astype(dtype)
            # 生成上三角阵列，如果数据类型索引小于 2 则为实数，否则为复数
            A = triu(rand(n, n)) if ind < 2 else triu(rand(n, n)+rand(n, n)*1j)
            # 创建压缩存储
            c, r = tril_indices(n)
            Ap = A[r, c]
            
            # 获取 BLAS 函数 tpmv
            func, = get_blas_funcs(('tpmv',), dtype=dtype)
            
            # 测试不同参数组合下的 tpmv 函数
            y1 = func(n=n, ap=Ap, x=x)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)

            y1 = func(n=n, ap=Ap, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = A.dot(x)
            assert_array_almost_equal(y1, y2)

            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=1)
            y2 = A.T.dot(x)
            assert_array_almost_equal(y1, y2)

            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=2)
            y2 = A.conj().T.dot(x)
            assert_array_almost_equal(y1, y2)
    # 定义测试函数 test_tpsv，用于测试 tpsv 相关功能
    def test_tpsv(self):
        # 设定随机数种子为 1234
        seed(1234)
        # 遍历数据类型列表 DTYPES 中的元素，ind 为索引，dtype 为数据类型
        for ind, dtype in enumerate(DTYPES):
            # 设定数组长度 n 为 10
            n = 10
            # 生成长度为 n 的随机数组 x，并转换为指定数据类型 dtype
            x = rand(n).astype(dtype)
            # 如果索引 ind 小于 2，则生成实数上三角阵 A，否则生成复数上三角阵 A
            A = triu(rand(n, n)) if ind < 2 else triu(rand(n, n)+rand(n, n)*1j)
            # 将 A 的对角线元素加 1
            A += eye(n)
            # 构造压缩存储格式的数组 Ap
            c, r = tril_indices(n)
            Ap = A[r, c]
            # 获取 BLAS 函数中的 tpsv 函数，并赋值给 func
            func, = get_blas_funcs(('tpsv',), dtype=dtype)

            # 调用 tpsv 函数计算结果 y1，与用 solve 函数计算的结果 y2 进行近似相等性断言
            y1 = func(n=n, ap=Ap, x=x)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)

            # 在对角线处加入值为 1 的元素，再次调用 tpsv 函数计算结果 y1，并与 solve 函数计算的结果 y2 进行比较
            y1 = func(n=n, ap=Ap, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(A, x)
            assert_array_almost_equal(y1, y2)

            # 使用 tpsv 函数进行转置运算，计算结果 y1，并与 solve 函数计算的结果 y2 进行比较
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=1)
            y2 = solve(A.T, x)
            assert_array_almost_equal(y1, y2)

            # 使用 tpsv 函数进行共轭转置运算，计算结果 y1，并与 solve 函数计算的结果 y2 进行比较
            y1 = func(n=n, ap=Ap, x=x, diag=1, trans=2)
            y2 = solve(A.conj().T, x)
            assert_array_almost_equal(y1, y2)

    # 定义测试函数 test_trmv，用于测试 trmv 相关功能
    def test_trmv(self):
        # 设定随机数种子为 1234
        seed(1234)
        # 遍历数据类型列表 DTYPES 中的元素，ind 为索引，dtype 为数据类型
        for ind, dtype in enumerate(DTYPES):
            # 设定矩阵大小为 3x3，生成随机矩阵 A，并加上单位矩阵，然后转换为指定数据类型 dtype
            n = 3
            A = (rand(n, n)+eye(n)).astype(dtype)
            # 生成长度为 3 的随机数组 x，并转换为指定数据类型 dtype
            x = rand(3).astype(dtype)
            # 获取 BLAS 函数中的 trmv 函数，并赋值给 func
            func, = get_blas_funcs(('trmv',), dtype=dtype)

            # 调用 trmv 函数计算结果 y1，与用 triu(A).dot(x) 计算的结果 y2 进行近似相等性断言
            y1 = func(a=A, x=x)
            y2 = triu(A).dot(x)
            assert_array_almost_equal(y1, y2)

            # 在对角线处加入值为 1 的元素，再次调用 trmv 函数计算结果 y1，并与 triu(A).dot(x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = triu(A).dot(x)
            assert_array_almost_equal(y1, y2)

            # 使用 trmv 函数进行转置运算，计算结果 y1，并与 triu(A).T.dot(x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1, trans=1)
            y2 = triu(A).T.dot(x)
            assert_array_almost_equal(y1, y2)

            # 使用 trmv 函数进行共轭转置运算，计算结果 y1，并与 triu(A).conj().T.dot(x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1, trans=2)
            y2 = triu(A).conj().T.dot(x)
            assert_array_almost_equal(y1, y2)

    # 定义测试函数 test_trsv，用于测试 trsv 相关功能
    def test_trsv(self):
        # 设定随机数种子为 1234
        seed(1234)
        # 遍历数据类型列表 DTYPES 中的元素，ind 为索引，dtype 为数据类型
        for ind, dtype in enumerate(DTYPES):
            # 设定矩阵大小为 15x15，生成随机矩阵 A，并加上单位矩阵，然后转换为指定数据类型 dtype
            n = 15
            A = (rand(n, n)+eye(n)).astype(dtype)
            # 生成长度为 15 的随机数组 x，并转换为指定数据类型 dtype
            x = rand(n).astype(dtype)
            # 获取 BLAS 函数中的 trsv 函数，并赋值给 func
            func, = get_blas_funcs(('trsv',), dtype=dtype)

            # 调用 trsv 函数计算结果 y1，与用 solve(triu(A), x) 计算的结果 y2 进行近似相等性断言
            y1 = func(a=A, x=x)
            y2 = solve(triu(A), x)
            assert_array_almost_equal(y1, y2)

            # 使用 trsv 函数在下三角部分计算结果 y1，与用 solve(tril(A), x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, lower=1)
            y2 = solve(tril(A), x)
            assert_array_almost_equal(y1, y2)

            # 在对角线处加入值为 1 的元素，再次调用 trsv 函数计算结果 y1，并与 solve(triu(A), x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1)
            A[arange(n), arange(n)] = dtype(1)
            y2 = solve(triu(A), x)
            assert_array_almost_equal(y1, y2)

            # 使用 trsv 函数进行转置运算，计算结果 y1，并与 solve(triu(A).T, x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1, trans=1)
            y2 = solve(triu(A).T, x)
            assert_array_almost_equal(y1, y2)

            # 使用 trsv 函数进行共轭转置运算，计算结果 y1，并与 solve(triu(A).conj().T, x) 计算的结果 y2 进行比较
            y1 = func(a=A, x=x, diag=1, trans=2)
            y2 = solve(triu(A).conj().T, x)
            assert_array_almost_equal(y1, y2)
class TestFBLAS3Simple:

    def test_gemm(self):
        # 对于每种类型的矩阵（'s'和'd'），获取对应的BLAS函数，执行矩阵乘法并断言结果接近预期
        for p in 'sd':
            f = getattr(fblas, p+'gemm', None)
            if f is None:
                continue
            # 断言矩阵乘法结果与预期结果近似相等
            assert_array_almost_equal(f(3, [3], [-4]), [[-36]])
            assert_array_almost_equal(f(3, [3], [-4], 3, [5]), [-21])
        
        # 对于复数类型的矩阵（'c'和'z'），获取对应的BLAS函数，执行复数矩阵乘法并断言结果接近预期
        for p in 'cz':
            f = getattr(fblas, p+'gemm', None)
            if f is None:
                continue
            # 断言复数矩阵乘法结果与预期结果近似相等
            assert_array_almost_equal(f(3j, [3-4j], [-4]), [[-48-36j]])
            assert_array_almost_equal(f(3j, [3-4j], [-4], 3, [5j]), [-48-21j])


def _get_func(func, ps='sdzc'):
    """Just a helper: return a specified BLAS function w/typecode."""
    # 根据指定的BLAS函数和类型码生成对应的函数生成器
    for p in ps:
        f = getattr(fblas, p+func, None)
        if f is None:
            continue
        yield f


class TestBLAS3Symm:

    def setup_method(self):
        # 设置测试方法的初始化数据
        self.a = np.array([[1., 2.],
                           [0., 1.]])
        self.b = np.array([[1., 0., 3.],
                           [0., -1., 2.]])
        self.c = np.ones((2, 3))
        self.t = np.array([[2., -1., 8.],
                           [3., 0., 9.]])

    def test_symm(self):
        # 测试SYMM函数的不同参数组合下的表现
        for f in _get_func('symm'):
            # 测试正常情况下的SYMM调用
            res = f(a=self.a, b=self.b, c=self.c, alpha=1., beta=1.)
            assert_array_almost_equal(res, self.t)

            # 测试transpose(a)情况下的SYMM调用
            res = f(a=self.a.T, b=self.b, lower=1, c=self.c, alpha=1., beta=1.)
            assert_array_almost_equal(res, self.t)

            # 测试transpose(b)和transpose(c)情况下的SYMM调用
            res = f(a=self.a, b=self.b.T, side=1, c=self.c.T,
                    alpha=1., beta=1.)
            assert_array_almost_equal(res, self.t.T)

    def test_summ_wrong_side(self):
        # 测试SYMM函数在错误的side参数下是否会引发异常
        f = getattr(fblas, 'dsymm', None)
        if f is not None:
            assert_raises(Exception, f, **{'a': self.a, 'b': self.b,
                                           'alpha': 1, 'side': 1})
            # `side=1` 表示C <- B*A，因此A和B的形状必须兼容，否则将引发f2py异常

    def test_symm_wrong_uplo(self):
        """SYMM仅考虑矩阵A的上/下三角部分。因此设置错误的lower值（默认lower=0，表示上三角）
        将导致错误的结果。
        """
        # 测试SYMM函数在错误的lower参数下是否会返回不正确的结果
        f = getattr(fblas, 'dsymm', None)
        if f is not None:
            res = f(a=self.a, b=self.b, c=self.c, alpha=1., beta=1.)
            assert np.allclose(res, self.t)

            res = f(a=self.a, b=self.b, lower=1, c=self.c, alpha=1., beta=1.)
            assert not np.allclose(res, self.t)


class TestBLAS3Syrk:
    def setup_method(self):
        # 设置测试方法的初始化数据
        self.a = np.array([[1., 0.],
                           [0., -2.],
                           [2., 3.]])
        self.t = np.array([[1., 0., 2.],
                           [0., 4., -6.],
                           [2., -6., 13.]])
        self.tt = np.array([[5., 6.],
                            [6., 13.]])
    # 定义测试函数 test_syrk，用于测试针对'syrk'操作的函数
    def test_syrk(self):
        # 遍历获取'syrk'函数的不同版本
        for f in _get_func('syrk'):
            # 调用函数 f，计算结果 c，并检查其上三角部分是否与预期的 self.t 相似
            c = f(a=self.a, alpha=1.)
            assert_array_almost_equal(np.triu(c), np.triu(self.t))

            # 再次调用函数 f，传递额外的参数 lower=1，检查其下三角部分是否与预期的 self.t 相似
            c = f(a=self.a, alpha=1., lower=1)
            assert_array_almost_equal(np.tril(c), np.tril(self.t))

            # 准备一个全为 1 的数组 c0，调用函数 f，传递 alpha=1., beta=1., c=c0，检查结果与 self.t+c0 的上三角部分是否相似
            c0 = np.ones(self.t.shape)
            c = f(a=self.a, alpha=1., beta=1., c=c0)
            assert_array_almost_equal(np.triu(c), np.triu(self.t+c0))

            # 再次调用函数 f，传递参数 alpha=1., trans=1，检查结果的上三角部分是否与预期的 self.tt 相似
            c = f(a=self.a, alpha=1., trans=1)
            assert_array_almost_equal(np.triu(c), np.triu(self.tt))

    # 打印错误信息 '0-th dimension must be fixed to 3 but got 5'，
    # FIXME: 是否需要抑制？
    # FIXME: 如何捕获 _fblas.error？
    # 定义测试函数 test_syrk_wrong_c，用于测试当传递错误的 c 参数时的情况
    def test_syrk_wrong_c(self):
        # 获取 fblas 模块中的 'dsyrk' 函数，如果不存在则为 None
        f = getattr(fblas, 'dsyrk', None)
        if f is not None:
            # 断言调用函数 f 时会抛出异常 Exception，传递参数 {'a': self.a, 'alpha': 1., 'c': np.ones((5, 8))}
            assert_raises(Exception, f, **{'a': self.a, 'alpha': 1.,
                                           'c': np.ones((5, 8))})
        # 如果提供了参数 C，则其必须具有兼容的维度
class TestBLAS3Syr2k:
    # 设置测试方法的初始化
    def setup_method(self):
        # 初始化数组 a，包含浮点数
        self.a = np.array([[1., 0.],
                           [0., -2.],
                           [2., 3.]])
        # 初始化数组 b，包含浮点数
        self.b = np.array([[0., 1.],
                           [1., 0.],
                           [0, 1.]])
        # 初始化数组 t，包含浮点数
        self.t = np.array([[0., -1., 3.],
                           [-1., 0., 0.],
                           [3., 0., 6.]])
        # 初始化数组 tt，包含浮点数
        self.tt = np.array([[0., 1.],
                            [1., 6]])

    # 测试 syr2k 函数
    def test_syr2k(self):
        # 对于 _get_func('syr2k') 返回的每个函数 f
        for f in _get_func('syr2k'):
            # 调用 f 函数，计算 c，要求满足 np.triu(c) 约等于 np.triu(self.t)
            c = f(a=self.a, b=self.b, alpha=1.)
            assert_array_almost_equal(np.triu(c), np.triu(self.t))

            # 再次调用 f 函数，设置 lower=1，确保 np.tril(c) 约等于 np.tril(self.t)
            c = f(a=self.a, b=self.b, alpha=1., lower=1)
            assert_array_almost_equal(np.tril(c), np.tril(self.t))

            # 初始化 c0 为全为 1 的与 self.t 相同形状的数组，调用 f 函数，确保 np.triu(c) 约等于 np.triu(self.t+c0)
            c0 = np.ones(self.t.shape)
            c = f(a=self.a, b=self.b, alpha=1., beta=1., c=c0)
            assert_array_almost_equal(np.triu(c), np.triu(self.t+c0))

            # 再次调用 f 函数，设置 trans=1，确保 np.triu(c) 约等于 np.triu(self.tt)
            c = f(a=self.a, b=self.b, alpha=1., trans=1)
            assert_array_almost_equal(np.triu(c), np.triu(self.tt))

    # 打印 '0-th dimension must be fixed to 3 but got 5', FIXME: suppress?
    def test_syr2k_wrong_c(self):
        # 获取 fblas 模块中的 'dsyr2k' 函数，并赋给变量 f
        f = getattr(fblas, 'dsyr2k', None)
        # 如果 f 不为 None
        if f is not None:
            # 断言异常，调用 f 函数，传入参数 {'a': self.a, 'b': self.b, 'alpha': 1., 'c': np.zeros((15, 8))}
            assert_raises(Exception, f, **{'a': self.a,
                                           'b': self.b,
                                           'alpha': 1.,
                                           'c': np.zeros((15, 8))})
            # 如果提供了 C，则其必须具有兼容的维度

class TestSyHe:
    """Quick and simple tests for (zc)-symm, syrk, syr2k."""

    # 设置测试方法的初始化
    def setup_method(self):
        # 初始化 sigma_y，包含复数
        self.sigma_y = np.array([[0., -1.j],
                                 [1.j, 0.]])

    # 测试 symm_zc 函数
    def test_symm_zc(self):
        # 对于 _get_func('symm', 'zc') 返回的每个函数 f
        for f in _get_func('symm', 'zc'):
            # 调用 f 函数，计算 res，确保 np.triu(res) 约等于 np.diag([1, -1])
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.)
            assert_array_almost_equal(np.triu(res), np.diag([1, -1]))

    # 测试 hemm_zc 函数
    def test_hemm_zc(self):
        # 对于 _get_func('hemm', 'zc') 返回的每个函数 f
        for f in _get_func('hemm', 'zc'):
            # 调用 f 函数，计算 res，确保 np.triu(res) 约等于 np.diag([1, 1])
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.)
            assert_array_almost_equal(np.triu(res), np.diag([1, 1]))

    # 测试 syrk_zr 函数
    def test_syrk_zr(self):
        # 对于 _get_func('syrk', 'zc') 返回的每个函数 f
        for f in _get_func('syrk', 'zc'):
            # 调用 f 函数，计算 res，确保 np.triu(res) 约等于 np.diag([-1, -1])
            res = f(a=self.sigma_y, alpha=1.)
            assert_array_almost_equal(np.triu(res), np.diag([-1, -1]))

    # 测试 herk_zr 函数
    def test_herk_zr(self):
        # 对于 _get_func('herk', 'zc') 返回的每个函数 f
        for f in _get_func('herk', 'zc'):
            # 调用 f 函数，计算 res，确保 np.triu(res) 约等于 np.diag([1, 1])
            res = f(a=self.sigma_y, alpha=1.)
            assert_array_almost_equal(np.triu(res), np.diag([1, 1]))

    # 测试 syr2k_zr 函数
    def test_syr2k_zr(self):
        # 对于 _get_func('syr2k', 'zc') 返回的每个函数 f
        for f in _get_func('syr2k', 'zc'):
            # 调用 f 函数，计算 res，确保 np.triu(res) 约等于 2.*np.diag([-1, -1])
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.)
            assert_array_almost_equal(np.triu(res), 2.*np.diag([-1, -1]))
    # 定义一个测试方法 `test_her2k_zr`，用于测试 her2k 和 zc 函数的各种变体
    def test_her2k_zr(self):
        # 遍历获取函数列表中的每个函数对象
        for f in _get_func('her2k', 'zc'):
            # 调用当前函数对象 f，传入参数 a=self.sigma_y, b=self.sigma_y, alpha=1.，获取结果
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.)
            # 断言结果 res 的上三角部分近似等于 2 倍的对角矩阵 [1, 1]
            assert_array_almost_equal(np.triu(res), 2.*np.diag([1, 1]))
class TestTRMM:
    """Quick and simple tests for dtrmm."""

    def setup_method(self):
        # 初始化测试所需的矩阵 a 和 b
        self.a = np.array([[1., 2., ],
                           [-2., 1.]])
        self.b = np.array([[3., 4., -1.],
                           [5., 6., -2.]])

        # 初始化另外两个测试用例所需的矩阵 a2 和 b2，使用指定的存储顺序
        self.a2 = np.array([[1, 1, 2, 3],
                            [0, 1, 4, 5],
                            [0, 0, 1, 6],
                            [0, 0, 0, 1]], order="f")
        self.b2 = np.array([[1, 4], [2, 5], [3, 6], [7, 8], [9, 10]],
                           order="f")

    @pytest.mark.parametrize("dtype_", DTYPES)
    def test_side(self, dtype_):
        trmm = get_blas_funcs("trmm", dtype=dtype_)
        # 对于 side=1，提供一个较大的 A 数组，应该成功；而 side=0 会失败 (见 gh-10841)
        assert_raises(Exception, trmm, 1.0, self.a2, self.b2)
        # 执行 TRMM 操作，验证结果是否与预期接近
        res = trmm(1.0, self.a2.astype(dtype_), self.b2.astype(dtype_),
                   side=1)
        k = self.b2.shape[1]
        assert_allclose(res, self.b2 @ self.a2[:k, :k], rtol=0.,
                        atol=100*np.finfo(dtype_).eps)

    def test_ab(self):
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            # 执行 dtrmm 操作，验证结果是否与预期数组 expected 接近
            result = f(1., self.a, self.b)
            # 默认情况下 a 是上三角的
            expected = np.array([[13., 16., -5.],
                                 [5., 6., -2.]])
            assert_array_almost_equal(result, expected)

    def test_ab_lower(self):
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            # 执行 dtrmm 操作，使用 lower=True，验证结果是否与预期数组 expected 接近
            result = f(1., self.a, self.b, lower=True)
            expected = np.array([[3., 4., -1.],
                                 [-1., -2., 0.]])  # 现在 a 是下三角的
            assert_array_almost_equal(result, expected)

    def test_b_overwrites(self):
        # BLAS 的 dtrmm 在原地修改 B 参数。
        # 默认情况下是复制，但可以通过 overwrite_b 参数来覆盖
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            for overwr in [True, False]:
                bcopy = self.b.copy()
                # 执行 dtrmm 操作，验证是否复制或原地修改了 B 参数
                result = f(1., self.a, bcopy, overwrite_b=overwr)
                # 检查是否复制了 C 连续的数组
                assert_(bcopy.flags.f_contiguous is False and
                        np.may_share_memory(bcopy, result) is False)
                assert_equal(bcopy, self.b)

            bcopy = np.asfortranarray(self.b.copy())  # 或者直接转置它
            # 执行 dtrmm 操作，验证是否原地修改了 B 参数
            result = f(1., self.a, bcopy, overwrite_b=True)
            assert_(bcopy.flags.f_contiguous is True and
                    np.may_share_memory(bcopy, result) is True)
            assert_array_almost_equal(bcopy, result)


def test_trsm():
    seed(1234)
    # 对于给定的每个数据类型，执行以下操作
    for ind, dtype in enumerate(DTYPES):
        # 计算当前数据类型的机器精度的1000倍作为容差
        tol = np.finfo(dtype).eps * 1000
        # 获取当前数据类型的 BLAS 函数中的 'trsm' 函数
        func, = get_blas_funcs(('trsm',), dtype=dtype)

        # 测试保护机制，防止大小不匹配
        # 创建随机矩阵 A 和 B，要求它们的维度不匹配，断言应该抛出异常
        A = rand(4, 5).astype(dtype)
        B = rand(4, 4).astype(dtype)
        alpha = dtype(1)
        assert_raises(Exception, func, alpha, A, B)
        assert_raises(Exception, func, alpha, A.T, B)

        # 设置矩阵 A 的大小和复数部分，计算结果 x1 和 x2 并比较它们的形状和数值接近程度
        n = 8
        m = 7
        alpha = dtype(-2.5)
        A = (rand(m, m) if ind < 2 else rand(m, m) + rand(m, m) * 1j) + eye(m)
        A = A.astype(dtype)
        Au = triu(A)
        Al = tril(A)
        B1 = rand(m, n).astype(dtype)
        B2 = rand(n, m).astype(dtype)

        # 计算函数 func 的结果 x1，并断言其形状与 B1 相同
        x1 = func(alpha=alpha, a=A, b=B1)
        assert_equal(B1.shape, x1.shape)
        # 使用 scipy 中的 solve 函数计算 x2，比较 x1 和 x2 的数值接近程度
        x2 = solve(Au, alpha * B1)
        assert_allclose(x1, x2, atol=tol)

        # 测试矩阵转置时的函数 func 的结果
        x1 = func(alpha=alpha, a=A, b=B1, trans_a=1)
        x2 = solve(Au.T, alpha * B1)
        assert_allclose(x1, x2, atol=tol)

        # 测试共轭转置时的函数 func 的结果
        x1 = func(alpha=alpha, a=A, b=B1, trans_a=2)
        x2 = solve(Au.conj().T, alpha * B1)
        assert_allclose(x1, x2, atol=tol)

        # 测试主对角线设置为单位矩阵时的函数 func 的结果
        x1 = func(alpha=alpha, a=A, b=B1, diag=1)
        Au[arange(m), arange(m)] = dtype(1)
        x2 = solve(Au, alpha * B1)
        assert_allclose(x1, x2, atol=tol)

        # 测试设置 side=1 时的函数 func 的结果
        x1 = func(alpha=alpha, a=A, b=B2, diag=1, side=1)
        x2 = solve(Au.conj().T, alpha * B2.conj().T)
        assert_allclose(x1, x2.conj().T, atol=tol)

        # 测试设置 side=1 和 lower=1 时的函数 func 的结果
        Al[arange(m), arange(m)] = dtype(1)
        x1 = func(alpha=alpha, a=A, b=B2, diag=1, side=1, lower=1)
        x2 = solve(Al.conj().T, alpha * B2.conj().T)
        assert_allclose(x1, x2.conj().T, atol=tol)
@pytest.mark.xfail(run=False,
                   reason="gh-16930")
# 标记为预期失败的测试用例，原因是问题编号为gh-16930
def test_gh_169309():
    # 创建一个长度为9的数组，每个元素值为10
    x = np.repeat(10, 9)
    # 使用BLAS库中的dnrm2函数计算向量x的2-范数
    actual = scipy.linalg.blas.dnrm2(x, 5, 3, -1)
    # 预期的向量x的2-范数的计算结果
    expected = math.sqrt(500)
    # 断言实际计算结果与预期结果的接近程度
    assert_allclose(actual, expected)


def test_dnrm2_neg_incx():
    # 检查当incx < 0时，dnrm2函数是否会引发异常
    # XXX: 在最低支持的BLAS实现（LAPACK 3.10新增特性）之后，移除此测试
    # 创建一个长度为9的数组，每个元素值为10
    x = np.repeat(10, 9)
    # 设置incx为负数
    incx = -1
    # 使用BLAS库中的dnrm2函数尝试计算向量x的2-范数，预期会引发异常
    with assert_raises(fblas.__fblas_error):
        scipy.linalg.blas.dnrm2(x, 5, 3, incx)
```