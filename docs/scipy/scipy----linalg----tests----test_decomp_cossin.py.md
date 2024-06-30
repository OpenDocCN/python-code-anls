# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_cossin.py`

```
# 导入 pytest 库，用于编写和运行测试
import pytest
# 导入 numpy 库，并从中导入 default_rng 和 assert_allclose 函数
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_allclose
# 从 scipy.linalg.lapack 中导入 _compute_lwork 函数
from scipy.linalg.lapack import _compute_lwork
# 从 scipy.stats 中导入 ortho_group 和 unitary_group 函数
from scipy.stats import ortho_group, unitary_group
# 从 scipy.linalg 中导入 cossin 和 get_lapack_funcs 函数
from scipy.linalg import cossin, get_lapack_funcs

# 定义浮点数和复数类型的数据类型元组
REAL_DTYPES = (np.float32, np.float64)
COMPLEX_DTYPES = (np.complex64, np.complex128)
DTYPES = REAL_DTYPES + COMPLEX_DTYPES

# 使用 pytest 的 parametrize 装饰器对 test_cossin 函数进行参数化测试
@pytest.mark.parametrize('dtype_', DTYPES)
@pytest.mark.parametrize('m, p, q',
                         [
                             (2, 1, 1),
                             (3, 2, 1),
                             (3, 1, 2),
                             (4, 2, 2),
                             (4, 1, 2),
                             (40, 12, 20),
                             (40, 30, 1),
                             (40, 1, 30),
                             (100, 50, 1),
                             (100, 50, 50),
                         ])
@pytest.mark.parametrize('swap_sign', [True, False])
def test_cossin(dtype_, m, p, q, swap_sign):
    # 创建默认随机数生成器对象 rng
    rng = default_rng(1708093570726217)
    # 根据数据类型选择生成正交矩阵或酉矩阵 x
    if dtype_ in COMPLEX_DTYPES:
        x = np.array(unitary_group.rvs(m, random_state=rng), dtype=dtype_)
    else:
        x = np.array(ortho_group.rvs(m, random_state=rng), dtype=dtype_)

    # 调用 cossin 函数进行余弦正弦分解，获取 u, cs, vh
    u, cs, vh = cossin(x, p, q,
                       swap_sign=swap_sign)
    # 断言余弦正弦分解结果的近似性
    assert_allclose(x, u @ cs @ vh, rtol=0., atol=m*1e3*np.finfo(dtype_).eps)
    # 断言 u 的数据类型与输入的数据类型一致
    assert u.dtype == dtype_
    # 断言 cs 的数据类型与 u 的实部的数据类型一致
    assert cs.dtype == np.real(u).dtype
    # 断言 vh 的数据类型与输入的数据类型一致
    assert vh.dtype == dtype_

    # 对 x 的不同划分进行余弦正弦分解，获取 u, cs, vh
    u, cs, vh = cossin([x[:p, :q], x[:p, q:], x[p:, :q], x[p:, q:]],
                       swap_sign=swap_sign)
    # 断言划分后的余弦正弦分解结果的近似性
    assert_allclose(x, u @ cs @ vh, rtol=0., atol=m*1e3*np.finfo(dtype_).eps)
    # 断言 u 的数据类型与输入的数据类型一致
    assert u.dtype == dtype_
    # 断言 cs 的数据类型与 u 的实部的数据类型一致
    assert cs.dtype == np.real(u).dtype
    # 断言 vh 的数据类型与输入的数据类型一致
    assert vh.dtype == dtype_

    # 调用 cossin 函数，仅计算 cs 和 vh，获取 cs2, vh2
    _, cs2, vh2 = cossin(x, p, q,
                         compute_u=False,
                         swap_sign=swap_sign)
    # 断言 cs 和 cs2 的近似性
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)
    # 断言 vh 和 vh2 的近似性
    assert_allclose(vh, vh2, rtol=0., atol=10*np.finfo(dtype_).eps)

    # 调用 cossin 函数，仅计算 u 和 cs，获取 u2, cs2, _
    u2, cs2, _ = cossin(x, p, q,
                        compute_vh=False,
                        swap_sign=swap_sign)
    # 断言 u 和 u2 的近似性
    assert_allclose(u, u2, rtol=0., atol=10*np.finfo(dtype_).eps)
    # 断言 cs 和 cs2 的近似性
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)

    # 调用 cossin 函数，不计算 u 和 vh，获取 cs2, _
    _, cs2, _ = cossin(x, p, q,
                       compute_u=False,
                       compute_vh=False,
                       swap_sign=swap_sign)
    # 断言 cs 和 cs2 的近似性
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)


# 定义测试混合类型数据的函数 test_cossin_mixed_types
def test_cossin_mixed_types():
    # 创建默认随机数生成器对象 rng
    rng = default_rng(1708093736390459)
    # 生成一个实数类型的正交矩阵 x
    x = np.array(ortho_group.rvs(4, random_state=rng), dtype=np.float64)
    # 对 x 的不同划分进行余弦正弦分解，获取 u, cs, vh
    u, cs, vh = cossin([x[:2, :2],
                        np.array(x[:2, 2:], dtype=np.complex128),
                        x[2:, :2],
                        x[2:, 2:]])

    # 断言 u 的数据类型为复数类型
    assert u.dtype == np.complex128
    # 断言 cs 的数据类型为实数类型
    assert cs.dtype == np.float64
    # 断言，验证 vh 的数据类型是否为复数类型（np.complex128）
    assert vh.dtype == np.complex128
    
    # 断言，验证 x 是否接近于 u @ cs @ vh 的结果，相对误差不超过 0，绝对误差为 1e4 倍的 np.complex128 类型的机器精度
    assert_allclose(x, u @ cs @ vh, rtol=0., atol=1e4 * np.finfo(np.complex128).eps)
# 测试函数：测试 cossin 函数在不正确的子块情况下是否会引发 ValueError 异常
def test_cossin_error_incorrect_subblocks():
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match="be due to missing p, q arguments."):
        # 调用 cossin 函数，传入一个不正确的子块列表作为参数
        cossin(([1, 2], [3, 4, 5], [6, 7], [8, 9, 10]))


# 测试函数：测试 cossin 函数在空子块情况下是否会引发 ValueError 异常
def test_cossin_error_empty_subblocks():
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match="x11.*empty"):
        # 调用 cossin 函数，传入全部子块为空的参数组合
        cossin(([], [], [], []))
    with pytest.raises(ValueError, match="x12.*empty"):
        # 调用 cossin 函数，传入部分子块为空的参数组合
        cossin(([1, 2], [], [6, 7], [8, 9, 10]))
    with pytest.raises(ValueError, match="x21.*empty"):
        # 调用 cossin 函数，传入部分子块为空的参数组合
        cossin(([1, 2], [3, 4, 5], [], [8, 9, 10]))
    with pytest.raises(ValueError, match="x22.*empty"):
        # 调用 cossin 函数，传入部分子块为空的参数组合
        cossin(([1, 2], [3, 4, 5], [2], []))


# 测试函数：测试 cossin 函数在缺少分块信息时是否会引发 ValueError 异常
def test_cossin_error_missing_partitioning():
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=".*exactly four arrays.* got 2"):
        # 调用 cossin 函数，传入一个只有两个数组的参数
        cossin(unitary_group.rvs(2))

    with pytest.raises(ValueError, match=".*might be due to missing p, q"):
        # 调用 cossin 函数，传入一个只有四个数组但缺少分块信息的参数
        cossin(unitary_group.rvs(4))


# 测试函数：测试 cossin 函数在非可迭代参数情况下是否会引发 ValueError 异常
def test_cossin_error_non_iterable():
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match="containing the subblocks of X"):
        # 调用 cossin 函数，传入一个非可迭代对象作为参数
        cossin(12j)


# 测试函数：测试 cossin 函数在非方阵参数情况下是否会引发 ValueError 异常
def test_cossin_error_non_square():
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match="only supports square"):
        # 调用 cossin 函数，传入一个非方阵参数
        cossin(np.array([[1, 2]]), 1, 1)


# 测试函数：测试 cossin 函数在分块参数不合法情况下是否会引发 ValueError 异常
def test_cossin_error_partitioning():
    # 创建一个随机正交矩阵 X
    x = np.array(ortho_group.rvs(4), dtype=np.float64)
    # 使用 pytest 来检查是否引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match="invalid p=0.*0<p<4.*"):
        # 调用 cossin 函数，传入 p=0 的参数
        cossin(x, 0, 1)
    with pytest.raises(ValueError, match="invalid p=4.*0<p<4.*"):
        # 调用 cossin 函数，传入 p=4 的参数
        cossin(x, 4, 1)
    with pytest.raises(ValueError, match="invalid q=-2.*0<q<4.*"):
        # 调用 cossin 函数，传入 q=-2 的参数
        cossin(x, 1, -2)
    with pytest.raises(ValueError, match="invalid q=5.*0<q<4.*"):
        # 调用 cossin 函数，传入 q=5 的参数
        cossin(x, 1, 5)


# 参数化测试函数：测试 cossin 函数在不同数据类型下的分离情况
@pytest.mark.parametrize("dtype_", DTYPES)
def test_cossin_separate(dtype_):
    # 设定随机数生成器和参数
    rng = default_rng(1708093590167096)
    m, p, q = 98, 37, 61

    # 根据数据类型选择不同的随机正交或酉矩阵
    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'
    X = (ortho_group.rvs(m, random_state=rng) if pfx == 'or'
         else unitary_group.rvs(m, random_state=rng))
    X = np.array(X, dtype=dtype_)

    # 获取 LAPACK 函数以及计算所需的工作空间大小
    drv, dlw = get_lapack_funcs((pfx + 'csd', pfx + 'csd_lwork'), [X])
    lwval = _compute_lwork(dlw, m, p, q)
    lwvals = {'lwork': lwval} if pfx == 'or' else dict(zip(['lwork',
                                                            'lrwork'],
                                                           lwval))

    # 调用 LAPACK 函数获取返回值
    *_, theta, u1, u2, v1t, v2t, _ = \
        drv(X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:], **lwvals)

    # 调用 cossin 函数进行分离计算
    (u1_2, u2_2), theta2, (v1t_2, v2t_2) = cossin(X, p, q, separate=True)

    # 使用 assert_allclose 断言函数比较计算结果的近似性
    assert_allclose(u1_2, u1, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(u2_2, u2, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(v1t_2, v1t, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(v2t_2, v2t, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(theta2, theta, rtol=0., atol=10*np.finfo(dtype_).eps)
```