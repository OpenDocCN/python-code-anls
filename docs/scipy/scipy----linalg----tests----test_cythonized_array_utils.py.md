# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_cythonized_array_utils.py`

```
# 导入 numpy 库，命名为 np
import numpy as np
# 从 scipy.linalg 中导入 bandwidth, issymmetric, ishermitian 函数
from scipy.linalg import bandwidth, issymmetric, ishermitian
# 导入 pytest 库
import pytest
# 从 pytest 中导入 raises 函数
from pytest import raises


# 定义测试函数，验证 bandwidth 函数处理不同数据类型的行为
def test_bandwidth_dtypes():
    # 设置矩阵维度为 5
    n = 5
    # 遍历 numpy 的所有数据类型
    for t in np.typecodes['All']:
        # 创建一个全零矩阵 A，数据类型为 t
        A = np.zeros([n, n], dtype=t)
        # 如果数据类型 t 在 'eUVOMm' 中
        if t in 'eUVOMm':
            # 断言调用 bandwidth(A) 会引发 TypeError 异常
            raises(TypeError, bandwidth, A)
        # 如果数据类型 t 为 'G'
        elif t == 'G':  # No-op test. On win these pass on others fail.
            # 不执行任何操作，仅在 Windows 上测试通过，其他情况下失败
            pass
        else:
            # 调用 bandwidth(A) 函数
            _ = bandwidth(A)


# 定义测试函数，验证 bandwidth 函数处理非二维输入的行为
def test_bandwidth_non2d_input():
    # 创建一个一维数组 A
    A = np.array([1, 2, 3])
    # 断言调用 bandwidth(A) 会引发 ValueError 异常
    raises(ValueError, bandwidth, A)
    # 创建一个三维数组 A
    A = np.array([[[1, 2, 3], [4, 5, 6]]])
    # 断言调用 bandwidth(A) 会引发 ValueError 异常
    raises(ValueError, bandwidth, A)


# 使用 pytest 的参数化装饰器，对 bandwidth 函数进行测试，排除特定数据类型
@pytest.mark.parametrize('T', [x for x in np.typecodes['All']
                               if x not in 'eGUVOMm'])
def test_bandwidth_square_inputs(T):
    # 设置矩阵维度为 20，带宽为 4
    n = 20
    k = 4
    # 创建一个全零矩阵 R，数据类型为 T，列序为 Fortran
    R = np.zeros([n, n], dtype=T, order='F')
    # 在原地创建一个带状矩阵
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n-k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n-1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n-k)]] = 1
    # 断言调用 bandwidth(R) 返回的结果为 (k, k)
    assert bandwidth(R) == (k, k)
    # 创建一个二维数组 A，用于额外的测试
    A = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ])
    # 断言调用 bandwidth(A) 返回的结果为 (2, 2)
    assert bandwidth(A) == (2, 2)


# 使用 pytest 的参数化装饰器，对 bandwidth 函数进行测试，排除特定数据类型
@pytest.mark.parametrize('T', [x for x in np.typecodes['All']
                               if x not in 'eGUVOMm'])
def test_bandwidth_rect_inputs(T):
    # 设置矩阵维度为 10x20，带宽为 5
    n, m = 10, 20
    k = 5
    # 创建一个全零矩阵 R，数据类型为 T，列序为 Fortran
    R = np.zeros([n, m], dtype=T, order='F')
    # 在原地创建一个带状矩阵
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n-k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n-1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n-k)]] = 1
    # 断言调用 bandwidth(R) 返回的结果为 (k, k)
    assert bandwidth(R) == (k, k)


# 定义测试函数，验证 issymmetric 和 ishermitian 函数处理不同数据类型的行为
def test_issymetric_ishermitian_dtypes():
    # 设置矩阵维度为 5
    n = 5
    # 遍历 numpy 的所有数据类型
    for t in np.typecodes['All']:
        # 创建一个全零矩阵 A，数据类型为 t
        A = np.zeros([n, n], dtype=t)
        # 如果数据类型 t 在 'eUVOMm' 中
        if t in 'eUVOMm':
            # 断言调用 issymmetric(A) 和 ishermitian(A) 会引发 TypeError 异常
            raises(TypeError, issymmetric, A)
            raises(TypeError, ishermitian, A)
        # 如果数据类型 t 为 'G'
        elif t == 'G':  # No-op test. On win these pass on others fail.
            # 不执行任何操作，仅在 Windows 上测试通过，其他情况下失败
            pass
        else:
            # 断言调用 issymmetric(A) 和 ishermitian(A) 返回 True
            assert issymmetric(A)
            assert ishermitian(A)


# 定义测试函数，验证 issymmetric 和 ishermitian 函数处理非法输入的行为
def test_issymmetric_ishermitian_invalid_input():
    # 创建一个一维数组 A
    A = np.array([1, 2, 3])
    # 断言调用 issymmetric(A) 和 ishermitian(A) 会引发 ValueError 异常
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    # 创建一个三维数组 A
    A = np.array([[[1, 2, 3], [4, 5, 6]]])
    # 断言调用 issymmetric(A) 和 ishermitian(A) 会引发 ValueError 异常
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    # 创建一个二维数组 A
    A = np.array([[1, 2, 3], [4, 5, 6]])
    # 断言调用 issymmetric(A) 和 ishermitian(A) 会引发 ValueError 异常
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)


# 定义测试函数，验证 issymmetric 函数处理复数和小数的行为
def test_issymetric_complex_decimals():
    # 创建一个复数类型的 3x3 数组 A
    A = np.arange(1, 10).astype(complex).reshape(3, 3)
    A += np.arange(-4, 5).astype(complex).reshape(3, 3)*1j
    # 将矩阵 A 中的每个元素除以 π
    A /= np.pi
    # 将矩阵 A 与其转置相加，得到对称矩阵
    A = A + A.T
    # 断言矩阵 A 是对称的，如果不是则会抛出异常
    assert issymmetric(A)
# 测试是否为厄米矩阵（复数和小数）
def test_ishermitian_complex_decimals():
    # 创建一个复数类型的3x3矩阵，元素从1到9
    A = np.arange(1, 10).astype(complex).reshape(3, 3)
    # 将矩阵A的每个元素加上虚数单位乘以从-4到4的复数矩阵
    A += np.arange(-4, 5).astype(complex).reshape(3, 3)*1j
    # 将矩阵A的所有元素除以π
    A /= np.pi
    # 将矩阵A与其共轭转置的和赋给A，使其成为厄米矩阵
    A = A + A.T.conj()
    # 断言矩阵A是厄米矩阵
    assert ishermitian(A)


# 测试是否为对称矩阵（近似结果）
def test_issymmetric_approximate_results():
    n = 20
    rng = np.random.RandomState(123456789)
    # 创建一个大小为[n, n]的均匀分布随机矩阵x
    x = rng.uniform(high=5., size=[n, n])
    # 计算x与其转置的乘积y，得到对称矩阵
    y = x @ x.T  # symmetric
    # 创建一个标准正态分布随机矩阵p
    p = rng.standard_normal([n, n])
    # 计算p、y和p转置的乘积z
    z = p @ y @ p.T
    # 断言z是一个对称矩阵，允许误差为1e-10
    assert issymmetric(z, atol=1e-10)
    # 再次断言z是一个对称矩阵，精确到误差为1e-10和相对误差为0
    assert issymmetric(z, atol=1e-10, rtol=0.)
    # 再次断言z是一个对称矩阵，允许绝对误差为0和相对误差为1e-12
    assert issymmetric(z, atol=0., rtol=1e-12)
    # 再次断言z是一个对称矩阵，允许绝对误差为1e-13和相对误差为1e-12
    assert issymmetric(z, atol=1e-13, rtol=1e-12)


# 测试是否为厄米矩阵（近似结果）
def test_ishermitian_approximate_results():
    n = 20
    rng = np.random.RandomState(987654321)
    # 创建一个大小为[n, n]的均匀分布随机矩阵x
    x = rng.uniform(high=5., size=[n, n])
    # 计算x与其转置的乘积y，得到对称矩阵
    y = x @ x.T  # symmetric
    # 创建一个具有实部和虚部的标准正态分布随机矩阵p
    p = rng.standard_normal([n, n]) + rng.standard_normal([n, n])*1j
    # 计算p、y和p的共轭转置的乘积z
    z = p @ y @ p.conj().T
    # 断言z是一个厄米矩阵，允许误差为1e-10
    assert ishermitian(z, atol=1e-10)
    # 再次断言z是一个厄米矩阵，精确到误差为1e-10和相对误差为0
    assert ishermitian(z, atol=1e-10, rtol=0.)
    # 再次断言z是一个厄米矩阵，允许绝对误差为0和相对误差为1e-12
    assert ishermitian(z, atol=0., rtol=1e-12)
    # 再次断言z是一个厄米矩阵，允许绝对误差为1e-13和相对误差为1e-12
    assert ishermitian(z, atol=1e-13, rtol=1e-12)
```