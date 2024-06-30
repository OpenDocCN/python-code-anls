# `D:\src\scipysrc\scipy\scipy\special\tests\test_wrightomega.py`

```
# 导入 pytest 库，用于测试
# 导入 numpy 库并使用 np 别名
# 从 numpy.testing 导入断言函数 assert_, assert_equal, assert_allclose

import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose

# 导入 scipy.special 库并使用 sc 别名
# 从 scipy.special._testutils 导入 assert_func_equal 函数
import scipy.special as sc
from scipy.special._testutils import assert_func_equal

# 定义测试函数 test_wrightomega_nan，测试 wrightomega 函数处理 NaN 的情况
def test_wrightomega_nan():
    # 定义包含各种 NaN 值的复数列表
    pts = [complex(np.nan, 0),
           complex(0, np.nan),
           complex(np.nan, np.nan),
           complex(np.nan, 1),
           complex(1, np.nan)]
    # 遍历列表中的每个复数
    for p in pts:
        # 调用 wrightomega 函数
        res = sc.wrightomega(p)
        # 断言结果的实部和虚部均为 NaN
        assert_(np.isnan(res.real))
        assert_(np.isnan(res.imag))

# 定义测试函数 test_wrightomega_inf_branch，测试 wrightomega 函数处理无穷分支的情况
def test_wrightomega_inf_branch():
    # 定义包含各种无穷分支的复数列表
    pts = [complex(-np.inf, np.pi/4),
           complex(-np.inf, -np.pi/4),
           complex(-np.inf, 3*np.pi/4),
           complex(-np.inf, -3*np.pi/4)]
    # 预期的结果列表
    expected_results = [complex(0.0, 0.0),
                        complex(0.0, -0.0),
                        complex(-0.0, 0.0),
                        complex(-0.0, -0.0)]
    # 遍历列表中的每个复数及其对应的预期结果
    for p, expected in zip(pts, expected_results):
        # 调用 wrightomega 函数
        res = sc.wrightomega(p)
        # 使用 assert_equal 检查结果的实部和虚部与预期结果相等
        # 注意：此处解释为无法使用 assert_equal(res, expected)，因为在旧版本的 numpy 中，
        # assert_equal 不会检查复数零的实部和虚部的符号。但它在比较实标量时会检查符号。
        assert_equal(res.real, expected.real)
        assert_equal(res.imag, expected.imag)

# 定义测试函数 test_wrightomega_inf，测试 wrightomega 函数处理无穷的情况
def test_wrightomega_inf():
    # 定义包含各种无穷值的复数列表
    pts = [complex(np.inf, 10),
           complex(-np.inf, 10),
           complex(10, np.inf),
           complex(10, -np.inf)]
    # 遍历列表中的每个复数
    for p in pts:
        # 使用 assert_equal 检查 wrightomega 函数的结果等于输入值
        assert_equal(sc.wrightomega(p), p)

# 定义测试函数 test_wrightomega_singular，测试 wrightomega 函数处理奇点的情况
def test_wrightomega_singular():
    # 定义包含奇点值的复数列表
    pts = [complex(-1.0, np.pi),
           complex(-1.0, -np.pi)]
    # 遍历列表中的每个复数
    for p in pts:
        # 调用 wrightomega 函数
        res = sc.wrightomega(p)
        # 使用 assert_equal 检查结果等于 -1.0
        assert_equal(res, -1.0)
        # 使用 assert_ 检查结果的虚部不是负数
        assert_(np.signbit(res.imag) == np.bool_(False))

# 使用 pytest.mark.parametrize 标记测试函数，测试 wrightomega 函数处理特定参数的情况
@pytest.mark.parametrize('x, desired', [
    (-np.inf, 0),
    (np.inf, np.inf),
])
def test_wrightomega_real_infinities(x, desired):
    # 使用 assert 检查 wrightomega 函数的结果等于预期值
    assert sc.wrightomega(x) == desired

# 定义测试函数 test_wrightomega_real_nan，测试 wrightomega 函数处理实数 NaN 的情况
def test_wrightomega_real_nan():
    # 使用 assert 检查 wrightomega 函数返回值是 NaN
    assert np.isnan(sc.wrightomega(np.nan))

# 定义测试函数 test_wrightomega_real_series_crossover，测试 wrightomega 函数在实数序列交叉点的近似情况
def test_wrightomega_real_series_crossover():
    # 定义所需的误差范围
    desired_error = 2 * np.finfo(float).eps
    crossover = 1e20
    # 计算交叉点之前和之后的值
    x_before_crossover = np.nextafter(crossover, -np.inf)
    x_after_crossover = np.nextafter(crossover, np.inf)
    # 使用 Mpmath 计算得到的预期值
    desired_before_crossover = 99999999999999983569.948
    desired_after_crossover = 100000000000000016337.948
    # 使用 assert_allclose 检查 wrightomega 函数的结果与预期值的接近程度
    assert_allclose(
        sc.wrightomega(x_before_crossover),
        desired_before_crossover,
        atol=0,
        rtol=desired_error,
    )
    assert_allclose(
        sc.wrightomega(x_after_crossover),
        desired_after_crossover,
        atol=0,
        rtol=desired_error,
    )
    # 设定交叉点前的期望值
    desired_before_crossover = 1.9287498479639314876e-22
    # 设定交叉点后的期望值
    desired_after_crossover = 1.9287498479639040784e-22
    # 使用 assert_allclose 函数验证在给定误差范围内，调用 sc.wrightomega 函数计算的结果与预期值是否接近
    assert_allclose(
        sc.wrightomega(x_before_crossover),  # 调用 wrightomega 函数计算交叉点前的值
        desired_before_crossover,  # 期望的交叉点前的值
        atol=0,  # 允许的绝对误差
        rtol=desired_error,  # 允许的相对误差
    )
    # 使用 assert_allclose 函数验证在给定误差范围内，调用 sc.wrightomega 函数计算的结果与预期值是否接近
    assert_allclose(
        sc.wrightomega(x_after_crossover),  # 调用 wrightomega 函数计算交叉点后的值
        desired_after_crossover,  # 期望的交叉点后的值
        atol=0,  # 允许的绝对误差
        rtol=desired_error,  # 允许的相对误差
    )
# 定义一个函数来测试 Wright Ω 函数在实数和复数输入情况下的表现差异
def test_wrightomega_real_versus_complex():
    # 生成一个包含 1001 个元素的 numpy 数组，范围从 -500 到 500
    x = np.linspace(-500, 500, 1001)
    # 调用 scipy 的 wrightomega 函数计算给定复数 x + 0j 的实部结果
    results = sc.wrightomega(x + 0j).real
    # 使用自定义的 assert_func_equal 函数检查 wrightomega 的实现是否与给定的结果匹配，
    # 其中 atol 表示绝对误差的允许范围，rtol 表示相对误差的允许范围
    assert_func_equal(sc.wrightomega, results, x, atol=0, rtol=1e-14)
```