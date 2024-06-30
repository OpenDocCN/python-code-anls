# `D:\src\scipysrc\sympy\sympy\series\tests\test_kauers.py`

```
# 导入从 sympy.series.kauers 模块中的 finite_diff 函数
# 以及 finite_diff_kauers 函数
from sympy.series.kauers import finite_diff
from sympy.series.kauers import finite_diff_kauers
# 从 sympy.abc 模块中导入符号 x, y, z, m, n, w
from sympy.abc import x, y, z, m, n, w
# 从 sympy.core.numbers 模块中导入 pi 常数
from sympy.core.numbers import pi
# 从 sympy.functions.elementary.trigonometric 模块中导入 cos 和 sin 函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.concrete.summations 模块中导入 Sum 函数

from sympy.concrete.summations import Sum


# 定义测试 finite_diff 函数的测试用例
def test_finite_diff():
    # 断言计算 x**2 + 2*x + 1 对 x 的有限差分应为 2*x + 3
    assert finite_diff(x**2 + 2*x + 1, x) == 2*x + 3
    # 断言计算 y**3 + 2*y**2 + 3*y + 5 对 y 的有限差分应为 3*y**2 + 7*y + 6
    assert finite_diff(y**3 + 2*y**2 + 3*y + 5, y) == 3*y**2 + 7*y + 6
    # 断言计算 z**2 - 2*z + 3 对 z 的有限差分应为 2*z - 1
    assert finite_diff(z**2 - 2*z + 3, z) == 2*z - 1
    # 断言计算 w**2 + 3*w - 2 对 w 的有限差分应为 2*w + 4
    assert finite_diff(w**2 + 3*w - 2, w) == 2*w + 4
    # 断言计算 sin(x) 对 x 的有限差分在 x = pi/6 处的值应为 -sin(x) + sin(x + pi/6)
    assert finite_diff(sin(x), x, pi/6) == -sin(x) + sin(x + pi/6)
    # 断言计算 cos(y) 对 y 的有限差分在 y = pi/3 处的值应为 -cos(y) + cos(y + pi/3)
    assert finite_diff(cos(y), y, pi/3) == -cos(y) + cos(y + pi/3)
    # 断言计算 x**2 - 2*x + 3 对 x 的有限差分在 x = 2 处的值应为 4*x
    assert finite_diff(x**2 - 2*x + 3, x, 2) == 4*x
    # 断言计算 n**2 - 2*n + 3 对 n 的有限差分在 n = 3 处的值应为 6*n + 3
    assert finite_diff(n**2 - 2*n + 3, n, 3) == 6*n + 3


# 定义测试 finite_diff_kauers 函数的测试用例
def test_finite_diff_kauers():
    # 断言对 Sum(x**2, (x, 1, n)) 使用 finite_diff_kauers 应返回 (n + 1)**2
    assert finite_diff_kauers(Sum(x**2, (x, 1, n))) == (n + 1)**2
    # 断言对 Sum(y, (y, 1, m)) 使用 finite_diff_kauers 应返回 (m + 1)
    assert finite_diff_kauers(Sum(y, (y, 1, m))) == (m + 1)
    # 断言对 Sum(x*y, (x, 1, m), (y, 1, n)) 使用 finite_diff_kauers 应返回 (m + 1)*(n + 1)
    assert finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n))) == (m + 1)*(n + 1)
    # 断言对 Sum(x*y**2, (x, 1, m), (y, 1, n)) 使用 finite_diff_kauers 应返回 (n + 1)**2*(m + 1)
    assert finite_diff_kauers(Sum((x*y**2), (x, 1, m), (y, 1, n))) == (n + 1)**2*(m + 1)
```