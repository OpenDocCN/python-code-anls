# `D:\src\scipysrc\scipy\scipy\special\tests\test_zeta.py`

```
# 导入scipy.special模块中的sc别名，用于特殊函数的计算
import scipy.special as sc
# 导入numpy模块，并将其重命名为np，用于数值计算
import numpy as np
# 从numpy.testing模块中导入断言函数assert_equal和assert_allclose，用于测试时的结果验证

# 测试函数，验证ζ(2, 2)的近似值是否符合预期
def test_zeta():
    assert_allclose(sc.zeta(2, 2), np.pi**2 / 6 - 1, rtol=1e-12)

# 测试函数，验证ζ的补函数在特定输入下的计算结果是否符合预期
def test_zetac():
    # 以下期望值是使用Wolfram Alpha计算得出的`Zeta[x] - 1`
    x = [-2.1, 0.8, 0.9999, 9, 50, 75]
    desired = [
        -0.9972705002153750,
        -5.437538415895550,
        -10000.42279161673,
        0.002008392826082214,
        8.881784210930816e-16,
        2.646977960169853e-23,
    ]
    assert_allclose(sc.zetac(x), desired, rtol=1e-12)

# 测试函数，验证ζ的补函数在特殊情况下的计算结果是否符合预期
def test_zetac_special_cases():
    assert sc.zetac(np.inf) == 0
    assert np.isnan(sc.zetac(-np.inf))
    assert sc.zetac(0) == -1.5
    assert sc.zetac(1.0) == np.inf

    assert_equal(sc.zetac([-2, -50, -100]), -1)

# 测试函数，验证黎曼ζ函数在特殊情况下的计算结果是否符合预期
def test_riemann_zeta_special_cases():
    assert np.isnan(sc.zeta(np.nan))
    assert sc.zeta(np.inf) == 1
    assert sc.zeta(0) == -0.5

    # 黎曼ζ函数在负偶整数处结果为零。
    assert_equal(sc.zeta([-2, -4, -6, -8, -10]), 0)

    assert_allclose(sc.zeta(2), np.pi**2 / 6, rtol=1e-12)
    assert_allclose(sc.zeta(4), np.pi**4 / 90, rtol=1e-12)

# 测试函数，验证黎曼ζ函数在避免溢出情况下的计算结果是否符合预期
def test_riemann_zeta_avoid_overflow():
    s = -260.00000000001
    desired = -5.6966307844402683127e+297  # 使用Mpmath计算得出的结果
    assert_allclose(sc.zeta(s), desired, atol=0, rtol=5e-14)
```