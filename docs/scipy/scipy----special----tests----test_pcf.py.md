# `D:\src\scipysrc\scipy\scipy\special\tests\test_pcf.py`

```
"""Tests for parabolic cylinder functions.

"""
# 导入 NumPy 库，并重命名为 np
import numpy as np
# 从 NumPy 测试工具中导入 assert_allclose 和 assert_equal 函数
from numpy.testing import assert_allclose, assert_equal
# 导入 SciPy 库中的 special 模块，并重命名为 sc
import scipy.special as sc


# 定义测试函数 test_pbwa_segfault
def test_pbwa_segfault():
    # 回归测试，检查 https://github.com/scipy/scipy/issues/6208 的问题
    #
    # 使用 mpmath 生成的数据
    #
    w = 1.02276567211316867161
    wp = -0.48887053372346189882
    # 断言 sc.pbwa(0, 0) 的结果接近于 (w, wp)，相对误差不超过 1e-13，绝对误差为 0
    assert_allclose(sc.pbwa(0, 0), (w, wp), rtol=1e-13, atol=0)


# 定义测试函数 test_pbwa_nan
def test_pbwa_nan():
    # 检查在实现准确范围之外是否返回 NaN
    pts = [(-6, -6), (-6, 6), (6, -6), (6, 6)]
    # 对于每个点 p，断言 sc.pbwa(*p) 的结果为 (np.nan, np.nan)
    for p in pts:
        assert_equal(sc.pbwa(*p), (np.nan, np.nan))
```