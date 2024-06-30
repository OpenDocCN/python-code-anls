# `D:\src\scipysrc\scipy\scipy\special\tests\test_owens_t.py`

```
# 导入 NumPy 库，并使用 np 别名
import numpy as np
# 从 numpy.testing 中导入 assert_equal 和 assert_allclose 函数
from numpy.testing import assert_equal, assert_allclose
# 导入 SciPy 库中的 scipy.special 模块，并使用 sc 别名
import scipy.special as sc


# 定义测试函数 test_symmetries，用于测试 Owens T 函数的对称性
def test_symmetries():
    # 设置随机种子为 1234
    np.random.seed(1234)
    # 创建长度为 100 的随机数组 a 和 h
    a, h = np.random.rand(100), np.random.rand(100)
    # 断言 Owens T 函数满足 h 和 -h 对称性
    assert_equal(sc.owens_t(h, a), sc.owens_t(-h, a))
    # 断言 Owens T 函数满足 h 和 -a 对称性
    assert_equal(sc.owens_t(h, a), -sc.owens_t(h, -a))


# 定义测试函数 test_special_cases，用于测试 Owens T 函数的特殊情况
def test_special_cases():
    # 断言 Owens T 函数的特殊情况 Owens T(5, 0) = 0
    assert_equal(sc.owens_t(5, 0), 0)
    # 使用 assert_allclose 断言 Owens T(0, 5) 的近似值
    assert_allclose(sc.owens_t(0, 5), 0.5*np.arctan(5)/np.pi,
                    rtol=5e-14)
    # 使用 assert_allclose 断言 Owens T(5, 1) 的近似值
    # 目标值是 0.5*Phi(5)*(1 - Phi(5))，其中 Phi 是标准正态分布的累积分布函数
    assert_allclose(sc.owens_t(5, 1), 1.4332574485503512543e-07,
                    rtol=5e-14)


# 定义测试函数 test_nans，用于测试 Owens T 函数处理 NaN 的情况
def test_nans():
    # 断言 Owens T 函数处理输入为 NaN 的情况返回 NaN
    assert_equal(sc.owens_t(20, np.nan), np.nan)
    assert_equal(sc.owens_t(np.nan, 20), np.nan)
    assert_equal(sc.owens_t(np.nan, np.nan), np.nan)


# 定义测试函数 test_infs，用于测试 Owens T 函数处理无穷大的情况
def test_infs():
    # 设置 h=0, a=inf
    h, a = 0, np.inf
    # 计算目标值 T(0, a) = 1/2π * arctan(a)
    res = 1/(2*np.pi) * np.arctan(a)
    # 使用 assert_allclose 断言 Owens T(0, a) 的近似值
    assert_allclose(sc.owens_t(h, a), res, rtol=5e-14)
    # 断言 Owens T(0, -a) 的近似值
    assert_allclose(sc.owens_t(h, -a), -res, rtol=5e-14)

    # 设置 h=1
    h = 1
    # 计算 Owens T(1, inf) 的近似值，参考维基百科中 Owens T 函数的定义
    # 使用数值积分计算
    res = 0.07932762696572854
    # 使用 assert_allclose 断言 Owens T(1, inf) 的近似值
    assert_allclose(sc.owens_t(h, np.inf), res, rtol=5e-14)
    # 断言 Owens T(1, -inf) 的近似值
    assert_allclose(sc.owens_t(h, -np.inf), -res, rtol=5e-14)

    # 断言 Owens T(inf, 1) 的值为 0
    assert_equal(sc.owens_t(np.inf, 1), 0)
    # 断言 Owens T(-inf, 1) 的值为 0
    assert_equal(sc.owens_t(-np.inf, 1), 0)

    # 断言 Owens T(inf, inf) 的值为 0
    assert_equal(sc.owens_t(np.inf, np.inf), 0)
    # 断言 Owens T(-inf, inf) 的值为 0
    assert_equal(sc.owens_t(-np.inf, np.inf), 0)
    # 断言 Owens T(inf, -inf) 的值为 -0.0
    assert_equal(sc.owens_t(np.inf, -np.inf), -0.0)
    # 断言 Owens T(-inf, -inf) 的值为 -0.0
    assert_equal(sc.owens_t(-np.inf, -np.inf), -0.0)
```