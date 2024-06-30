# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_ldl.py`

```
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_
# 导入必要的测试断言函数

from numpy import (array, eye, zeros, empty_like, empty, tril_indices_from,
                   tril, triu_indices_from, spacing, float32, float64,
                   complex64, complex128)
# 导入 numpy 模块中的各种函数和类，用于数组操作和数值计算

from numpy.random import rand, randint, seed
# 导入随机数生成相关函数

from scipy.linalg import ldl
# 导入 scipy 中的 ldl 分解函数

from scipy._lib._util import ComplexWarning
# 导入复杂数警告类

import pytest
# 导入 pytest 测试框架

from pytest import raises as assert_raises, warns
# 导入 pytest 的异常断言和警告断言函数


def test_args():
    A = eye(3)
    # 创建一个 3x3 的单位矩阵 A

    # 测试对非方阵进行 ldl 分解是否抛出 ValueError 异常
    assert_raises(ValueError, ldl, A[:, :2])

    # 测试带有虚数对角元的复杂矩阵是否会触发 ComplexWarning 警告
    with warns(ComplexWarning):
        ldl(A*1j)


def test_empty_array():
    a = empty((0, 0), dtype=complex)
    # 创建一个空的复数类型的 0x0 数组 a

    l, d, p = ldl(empty((0, 0)))
    # 对空的 0x0 数组进行 ldl 分解，得到 l、d、p

    assert_array_almost_equal(l, empty_like(a))
    # 断言 l 的结果与空数组 a 的结构相同

    assert_array_almost_equal(d, empty_like(a))
    # 断言 d 的结果与空数组 a 的结构相同

    assert_array_almost_equal(p, array([], dtype=int))
    # 断言 p 是一个空的整数数组


def test_simple():
    a = array([[-0.39-0.71j, 5.14-0.64j, -7.86-2.96j, 3.80+0.92j],
               [5.14-0.64j, 8.86+1.81j, -3.52+0.58j, 5.32-1.59j],
               [-7.86-2.96j, -3.52+0.58j, -2.83-0.03j, -1.54-2.86j],
               [3.80+0.92j, 5.32-1.59j, -1.54-2.86j, -0.56+0.12j]])
    # 创建一个复数类型的 4x4 数组 a

    b = array([[5., 10, 1, 18],
               [10., 2, 11, 1],
               [1., 11, 19, 9],
               [18., 1, 9, 0]])
    # 创建一个 4x4 的实数数组 b

    c = array([[52., 97, 112, 107, 50],
               [97., 114, 89, 98, 13],
               [112., 89, 64, 33, 6],
               [107., 98, 33, 60, 73],
               [50., 13, 6, 73, 77]])
    # 创建一个 5x5 的实数数组 c

    d = array([[2., 2, -4, 0, 4],
               [2., -2, -2, 10, -8],
               [-4., -2, 6, -8, -4],
               [0., 10, -8, 6, -6],
               [4., -8, -4, -6, 10]])
    # 创建一个 5x5 的实数数组 d

    e = array([[-1.36+0.00j, 0+0j, 0+0j, 0+0j],
               [1.58-0.90j, -8.87+0j, 0+0j, 0+0j],
               [2.21+0.21j, -1.84+0.03j, -4.63+0j, 0+0j],
               [3.91-1.50j, -1.78-1.18j, 0.11-0.11j, -1.84+0.00j]])
    # 创建一个复数类型的 4x4 数组 e

    for x in (b, c, d):
        l, d, p = ldl(x)
        # 对每个数组 x 进行 ldl 分解，得到 l、d、p

        assert_allclose(l.dot(d).dot(l.T), x, atol=spacing(1000.), rtol=0)
        # 断言 l、d 的乘积再乘以 l 的转置等于原始数组 x

        u, d, p = ldl(x, lower=False)
        # 对每个数组 x 进行 ldl 分解，指定 lower=False 得到上三角矩阵 u

        assert_allclose(u.dot(d).dot(u.T), x, atol=spacing(1000.), rtol=0)
        # 断言 u、d 的乘积再乘以 u 的转置等于原始数组 x

    l, d, p = ldl(a, hermitian=False)
    # 对复数数组 a 进行 ldl 分解，指定 hermitian=False

    assert_allclose(l.dot(d).dot(l.T), a, atol=spacing(1000.), rtol=0)
    # 断言 l、d 的乘积再乘以 l 的转置等于原始复数数组 a

    u, d, p = ldl(a, lower=False, hermitian=False)
    # 对复数数组 a 进行 ldl 分解，指定 lower=False 和 hermitian=False

    assert_allclose(u.dot(d).dot(u.T), a, atol=spacing(1000.), rtol=0)
    # 断言 u、d 的乘积再乘以 u 的转置等于原始复数数组 a

    # 使用共轭转置的上三角部分进行计算，并使用下三角部分进行比较
    l, d, p = ldl(e.conj().T, lower=0)
    # 对共轭转置后的 e 的上三角部分进行 ldl 分解，指定 lower=0 得到下三角矩阵 l

    assert_allclose(tril(l.dot(d).dot(l.conj().T)-e), zeros((4, 4)),
                    atol=spacing(1000.), rtol=0)
    # 断言 l、d 的乘积再乘以 l 的共轭转置减去原始数组 e 的下三角部分接近零矩阵


def test_permutations():
    seed(1234)
    # 设置随机数种子为 1234
    # 进行10次循环，生成随机数n，范围在1到100之间
    for _ in range(10):
        n = randint(1, 100)
        # 随机生成实数或复数数组x
        x = rand(n, n) if randint(2) else rand(n, n) + rand(n, n)*1j
        # 计算x与其共轭转置的和，即Hermitian对称化
        x = x + x.conj().T
        # 将x的对角线元素加上随机数乘以单位矩阵的值
        x += eye(n)*randint(5, 1e6)
        # 获取下三角部分的索引
        l_ind = tril_indices_from(x, k=-1)
        # 获取上三角部分的索引
        u_ind = triu_indices_from(x, k=1)

        # 使用ldl分解对x进行分解，lower=0表示返回不含下三角的部分
        u, d, p = ldl(x, lower=0)
        # 检验排列是否导致了三角形数组，下三角部分应为零
        assert_(not any(u[p, :][l_ind]), f'Spin {_} failed')

        # 使用ldl分解对x进行分解，lower=1表示返回不含上三角的部分
        l, d, p = ldl(x, lower=1)
        # 检验排列是否导致了三角形数组，上三角部分应为零
        assert_(not any(l[p, :][u_ind]), f'Spin {_} failed')
# 使用 pytest 的 parametrize 装饰器，为 test_ldl_type_size_combinations_real 函数指定多组参数进行测试
@pytest.mark.parametrize("dtype", [float32, float64])
@pytest.mark.parametrize("n", [30, 150])
def test_ldl_type_size_combinations_real(n, dtype):
    # 设置随机数种子
    seed(1234)
    # 构建错误消息字符串，描述当前测试失败时的测试条件
    msg = (f"Failed for size: {n}, dtype: {dtype}")

    # 生成 n x n 的随机矩阵 x，并转换为指定的数据类型 dtype
    x = rand(n, n).astype(dtype)
    # 将 x 与其转置相加，使其成为对称矩阵
    x = x + x.T
    # 对角线加上一个随机数，确保 x 是正定的
    x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

    # 对矩阵 x 进行 LDL 分解
    l, d1, p = ldl(x)
    # 对称分解，要求 l 是下三角矩阵
    u, d2, p = ldl(x, lower=0)
    # 设置相对误差容限 rtol，根据 dtype 类型选择不同的值
    rtol = 1e-4 if dtype is float32 else 1e-10
    # 断言 LDL 分解的正确性：l * d1 * l^T 应当接近 x
    assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg)
    # 断言对称的 LDL 分解的正确性：u * d2 * u^T 应当接近 x
    assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg)


# 使用 pytest 的 parametrize 装饰器，为 test_ldl_type_size_combinations_complex 函数指定多组参数进行测试
@pytest.mark.parametrize("dtype", [complex64, complex128])
@pytest.mark.parametrize("n", [30, 150])
def test_ldl_type_size_combinations_complex(n, dtype):
    # 设置随机数种子
    seed(1234)
    # 构建错误消息字符串，描述当前测试失败时的测试条件
    msg1 = (f"Her failed for size: {n}, dtype: {dtype}")
    msg2 = (f"Sym failed for size: {n}, dtype: {dtype}")

    # 复数域 Hermitian 上/下三角矩阵
    x = (rand(n, n)+1j*rand(n, n)).astype(dtype)
    x = x + x.conj().T
    x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

    # 对矩阵 x 进行 LDL 分解
    l, d1, p = ldl(x)
    # 对称分解，要求 l 是下三角矩阵
    u, d2, p = ldl(x, lower=0)
    # 设置相对误差容限 rtol，根据 dtype 类型选择不同的值
    rtol = 2e-4 if dtype is complex64 else 1e-10
    # 断言 LDL 分解的正确性：l * d1 * l^H 应当接近 x
    assert_allclose(l.dot(d1).dot(l.conj().T), x, rtol=rtol, err_msg=msg1)
    # 断言对称的 LDL 分解的正确性：u * d2 * u^H 应当接近 x
    assert_allclose(u.dot(d2).dot(u.conj().T), x, rtol=rtol, err_msg=msg1)

    # 复数域 Symmetric 上/下三角矩阵
    x = (rand(n, n)+1j*rand(n, n)).astype(dtype)
    x = x + x.T
    x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

    # 对矩阵 x 进行非 Hermitian LDL 分解
    l, d1, p = ldl(x, hermitian=0)
    # 对称分解，要求 l 是下三角矩阵
    u, d2, p = ldl(x, lower=0, hermitian=0)
    # 断言非 Hermitian LDL 分解的正确性：l * d1 * l^T 应当接近 x
    assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg2)
    # 断言对称的 LDL 分解的正确性：u * d2 * u^T 应当接近 x
    assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg2)
```