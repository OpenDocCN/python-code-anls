# `D:\src\scipysrc\scipy\scipy\special\tests\test_trig.py`

```
# 导入pytest库，用于编写和运行测试
import pytest
# 导入numpy库，并使用np作为别名
import numpy as np
# 从numpy.testing模块中导入断言函数，用于测试数值是否相等
from numpy.testing import assert_equal, assert_allclose, suppress_warnings
# 从scipy.special._ufuncs模块中导入_sinpi和_cospi函数
from scipy.special._ufuncs import _sinpi as sinpi
from scipy.special._ufuncs import _cospi as cospi


# 定义测试函数，用于测试sinpi和cospi函数对整数实部和半整数实部的处理
def test_integer_real_part():
    # 创建整数范围的numpy数组
    x = np.arange(-100, 101)
    # 创建包含负线性间隔和正线性间隔的堆叠数组
    y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
    # 创建x和y的网格
    x, y = np.meshgrid(x, y)
    # 创建复数数组z
    z = x + 1j*y
    # 测试sinpi函数的实部是否为0
    res = sinpi(z)
    assert_equal(res.real, 0.0)
    # 测试cospi函数的虚部是否为0
    res = cospi(z)
    assert_equal(res.imag, 0.0)


# 定义测试函数，用于测试sinpi和cospi函数对半整数实部的处理
def test_half_integer_real_part():
    # 创建半整数范围的numpy数组
    x = np.arange(-100, 101) + 0.5
    # 创建包含负线性间隔和正线性间隔的堆叠数组
    y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
    # 创建x和y的网格
    x, y = np.meshgrid(x, y)
    # 创建复数数组z
    z = x + 1j*y
    # 测试sinpi函数的虚部是否为0
    res = sinpi(z)
    assert_equal(res.imag, 0.0)
    # 测试cospi函数的实部是否为0
    res = cospi(z)
    assert_equal(res.real, 0.0)


# 标记为暂时跳过的测试函数，直至问题gh-19526解决
@pytest.mark.skip("Temporary skip while gh-19526 is being resolved")
def test_intermediate_overlow():
    # 确保在cosh/sinh会溢出而与sin/cos的乘积不会溢出的情况下避免溢出
    # sinpi_pts列表，包含复数点
    sinpi_pts = [complex(1 + 1e-14, 227),
                 complex(1e-35, 250),
                 complex(1e-301, 445)]
    # sinpi_std列表，包含用mpmath生成的数据
    sinpi_std = [complex(-8.113438309924894e+295, -np.inf),
                 complex(1.9507801934611995e+306, np.inf),
                 complex(2.205958493464539e+306, np.inf)]
    # 使用suppress_warnings上下文管理器，忽略特定警告
    with suppress_warnings() as sup:
        # 过滤RuntimeWarning警告，指定在"invalid value encountered in multiply"时忽略
        sup.filter(RuntimeWarning, "invalid value encountered in multiply")
        # 遍历sinpi_pts和sinpi_std，对每个点p调用sinpi函数，断言结果的实部和虚部近似相等于标准值std
        for p, std in zip(sinpi_pts, sinpi_std):
            res = sinpi(p)
            assert_allclose(res.real, std.real)
            assert_allclose(res.imag, std.imag)

    # 对cospi函数进行测试，较不重要，因为cos(0) = 1
    p = complex(0.5 + 1e-14, 227)
    std = complex(-8.113438309924894e+295, -np.inf)
    # 使用suppress_warnings上下文管理器，忽略特定警告
    with suppress_warnings() as sup:
        # 过滤RuntimeWarning警告，指定在"invalid value encountered in multiply"时忽略
        sup.filter(RuntimeWarning, "invalid value encountered in multiply")
        # 调用cospi函数，断言结果的实部近似相等于标准值std的实部
        assert_allclose(res.real, std.real)
        # 断言结果的虚部近似相等于标准值std的虚部
        assert_allclose(res.imag, std.imag)


# 定义测试函数，用于测试sinpi和cospi函数对0的处理
def test_zero_sign():
    # 测试sinpi函数对-0.0的返回值是否为0.0，并且是否为负数
    y = sinpi(-0.0)
    assert y == 0.0
    assert np.signbit(y)

    # 测试sinpi函数对+0.0的返回值是否为0.0，并且是否为非负数
    y = sinpi(0.0)
    assert y == 0.0
    assert not np.signbit(y)

    # 测试cospi函数对0.5的返回值是否为0.0，并且是否为非负数
    y = cospi(0.5)
    assert y == 0.0
    assert not np.signbit(y)
```