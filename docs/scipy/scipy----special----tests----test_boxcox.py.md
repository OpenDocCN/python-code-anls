# `D:\src\scipysrc\scipy\scipy\special\tests\test_boxcox.py`

```
# 导入numpy库，简写为np
import numpy as np
# 从numpy.testing模块导入用于测试的函数和方法：assert_equal, assert_almost_equal, assert_allclose
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
# 从scipy.special库中导入boxcox, boxcox1p, inv_boxcox, inv_boxcox1p函数
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p
# 导入pytest测试框架
import pytest

# "test_mpmath.py"中还有更多关于boxcox和boxcox1p的测试。

# 定义测试函数test_boxcox_basic，测试boxcox函数的基本功能
def test_boxcox_basic():
    # 创建一个包含浮点数的numpy数组
    x = np.array([0.5, 1, 2, 4])

    # lambda = 0  =>  y = log(x)，测试lambda=0时的输出
    y = boxcox(x, 0)
    assert_almost_equal(y, np.log(x))

    # lambda = 1  =>  y = x - 1，测试lambda=1时的输出
    y = boxcox(x, 1)
    assert_almost_equal(y, x - 1)

    # lambda = 2  =>  y = 0.5*(x**2 - 1)，测试lambda=2时的输出
    y = boxcox(x, 2)
    assert_almost_equal(y, 0.5*(x**2 - 1))

    # x = 0 and lambda > 0  =>  y = -1 / lambda，测试x=0且lambda>0时的输出
    lam = np.array([0.5, 1, 2])
    y = boxcox(0, lam)
    assert_almost_equal(y, -1.0 / lam)

# 定义测试函数test_boxcox_underflow，测试boxcox函数在下溢时的行为
def test_boxcox_underflow():
    # 创建一个接近1的浮点数
    x = 1 + 1e-15
    # 设置一个接近0的lambda值
    lmbda = 1e-306
    # 调用boxcox函数
    y = boxcox(x, lmbda)
    # 断言y与np.log(x)接近
    assert_allclose(y, np.log(x), rtol=1e-14)

# 定义测试函数test_boxcox_nonfinite，测试boxcox函数处理非有限值时的行为
def test_boxcox_nonfinite():
    # 创建包含负数的numpy数组
    x = np.array([-1, -1, -0.5])
    # 调用boxcox函数，传入lambda值的数组
    y = boxcox(x, [0.5, 2.0, -1.5])
    # 断言y与包含NaN的numpy数组相等
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))

    # 设置x为0，测试x=0且lambda<=0时的输出
    x = 0
    y = boxcox(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))

# 定义测试函数test_boxcox1p_basic，测试boxcox1p函数的基本功能
def test_boxcox1p_basic():
    # 创建包含浮点数的numpy数组
    x = np.array([-0.25, -1e-20, 0, 1e-20, 0.25, 1, 3])

    # lambda = 0  =>  y = log(1+x)，测试lambda=0时的输出
    y = boxcox1p(x, 0)
    assert_almost_equal(y, np.log1p(x))

    # lambda = 1  =>  y = x，测试lambda=1时的输出
    y = boxcox1p(x, 1)
    assert_almost_equal(y, x)

    # lambda = 2  =>  y = 0.5*((1+x)**2 - 1)，测试lambda=2时的输出
    y = boxcox1p(x, 2)
    assert_almost_equal(y, 0.5*x*(2 + x))

    # x = -1 and lambda > 0  =>  y = -1 / lambda，测试x=-1且lambda>0时的输出
    lam = np.array([0.5, 1, 2])
    y = boxcox1p(-1, lam)
    assert_almost_equal(y, -1.0 / lam)

# 定义测试函数test_boxcox1p_underflow，测试boxcox1p函数在下溢时的行为
def test_boxcox1p_underflow():
    # 创建包含浮点数的numpy数组
    x = np.array([1e-15, 1e-306])
    lmbda = np.array([1e-306, 1e-18])
    y = boxcox1p(x, lmbda)
    assert_allclose(y, np.log1p(x), rtol=1e-14)

# 定义测试函数test_boxcox1p_nonfinite，测试boxcox1p函数处理非有限值时的行为
def test_boxcox1p_nonfinite():
    # 创建包含负数的numpy数组
    x = np.array([-2, -2, -1.5])
    y = boxcox1p(x, [0.5, 2.0, -1.5])
    # 断言y与包含NaN的numpy数组相等
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))

    # 设置x为-1，测试x=-1且lambda<=0时的输出
    x = -1
    y = boxcox1p(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))

# 定义测试函数test_inv_boxcox，测试inv_boxcox函数的功能
def test_inv_boxcox():
    # 创建包含浮点数的numpy数组
    x = np.array([0., 1., 2.])
    lam = np.array([0., 1., 2.])
    # 调用boxcox函数得到y
    y = boxcox(x, lam)
    # 调用inv_boxcox函数还原x2
    x2 = inv_boxcox(y, lam)
    # 断言x与x2接近
    assert_almost_equal(x, x2)

    # 创建包含浮点数的numpy数组
    x = np.array([0., 1., 2.])
    lam = np.array([0., 1., 2.])
    # 调用boxcox1p函数得到y
    y = boxcox1p(x, lam)
    # 调用inv_boxcox1p函数还原x2
    x2 = inv_boxcox1p(y, lam)
    # 断言x与x2接近
    assert_almost_equal(x, x2)

# 定义测试函数test_inv_boxcox1p_underflow，测试inv_boxcox1p函数在下溢时的行为
def test_inv_boxcox1p_underflow():
    # 创建一个接近1e-15的浮点数
    x = 1e-15
    # 设置一个接近1e-306的lambda值
    lam = 1e-306
    # 调用inv_boxcox1p函数
    y = inv_boxcox1p(x, lam)
    # 断言y与x接近
    assert_allclose(y, x, rtol=1e-14)

# 使用pytest.mark.parametrize装饰器定义参数化测试函数test_boxcox_premature_overflow
@pytest.mark.parametrize(
    "x, lmb",
    [[100, 155],  # 测试x=100, lmb=155时的输出
     [0.01, -155]]  # 测试x=0.01, lmb=-155时的输出
)
def test_boxcox_premature_overflow(x, lmb):
    # 测试boxcox和inv_boxcox
    y = boxcox(x, lmb)
    # 断言y是有限值
    assert np.isfinite(y)
    # 调用inv_boxcox还原x_inv
    x_inv = inv_boxcox(y, lmb)
    # 断言x与x_inv接近
    assert_allclose(x, x_inv)

    # 测试boxcox1p和inv_boxcox1p
    # 使用 Box-Cox 变换对 x 进行转换，参数为 lmb
    y1p = boxcox1p(x-1, lmb)
    
    # 断言确保 y1p 中的所有值都是有限的（finite）
    assert np.isfinite(y1p)
    
    # 对 y1p 进行逆 Box-Cox 变换，使用相同的 lmb 参数
    x1p_inv = inv_boxcox1p(y1p, lmb)
    
    # 断言确保逆变换后的结果 x1p_inv 与原始 x-1 几乎相等
    assert_allclose(x-1, x1p_inv)
```