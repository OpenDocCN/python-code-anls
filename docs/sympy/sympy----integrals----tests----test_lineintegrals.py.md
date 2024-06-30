# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_lineintegrals.py`

```
# 从 sympy 库中导入必要的符号、常数和函数
from sympy.core.numbers import E
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry.curve import Curve
from sympy.integrals.integrals import line_integrate

# 定义符号变量 s, t, x, y, z
s, t, x, y, z = symbols('s,t,x,y,z')

# 定义测试函数 test_lineintegral
def test_lineintegral():
    # 创建一个曲线对象 c，这条曲线由参数 t 表示，其 x 坐标是 E**t + 1，y 坐标是 E**t - 1
    # 参数 t 的范围是从 0 到 log(2)
    c = Curve([E**t + 1, E**t - 1], (t, 0, log(2)))
    # 断言线积分 x + y 关于曲线 c 的值等于 3*sqrt(2)
    assert line_integrate(x + y, c, [x, y]) == 3*sqrt(2)
```