# `D:\src\scipysrc\scipy\scipy\special\tests\test_dd.py`

```
# 导入必要的库和模块
import pytest
from numpy.testing import assert_allclose
from scipy.special._test_internal import _dd_exp, _dd_log

# 定义测试数据列表，每个元组包含以下内容：
#   dd_func: 待测试的双精度函数
#   xhi, xlo: 输入参数的高位和低位部分
#   expected_yhi, expected_ylo: 预期的输出结果的高位和低位部分
# 预期值是使用 mpmath 计算得到的，例如：
#   import mpmath
#   mpmath.mp.dps = 100
#   xhi = 10.0
#   xlo = 0.0
#   x = mpmath.mpf(xhi) + mpmath.mpf(xlo)
#   y = mpmath.log(x)
#   expected_yhi = float(y)
#   expected_ylo = float(y - expected_yhi)
test_data = [
    (_dd_exp, -0.3333333333333333, -1.850371707708594e-17,
     0.7165313105737893, -2.0286948382455594e-17),
    (_dd_exp, 0.0, 0.0, 1.0, 0.0),
    (_dd_exp, 10.0, 0.0, 22026.465794806718, -1.3780134700517372e-12),
    (_dd_log, 0.03125, 0.0, -3.4657359027997265, -4.930038229799327e-18),
    (_dd_log, 10.0, 0.0, 2.302585092994046, -2.1707562233822494e-16),
]

# 使用 pytest 的 parametrize 装饰器，为每组测试数据执行测试函数
@pytest.mark.parametrize('dd_func, xhi, xlo, expected_yhi, expected_ylo',
                         test_data)
def test_dd(dd_func, xhi, xlo, expected_yhi, expected_ylo):
    # 调用双精度函数计算结果
    yhi, ylo = dd_func(xhi, xlo)
    # 断言高位结果与预期值相等
    assert yhi == expected_yhi, (f"high double ({yhi}) does not equal the "
                                 f"expected value {expected_yhi}")
    # 使用 assert_allclose 断言低位结果与预期值相近（相对容差设为 5e-15）
    assert_allclose(ylo, expected_ylo, rtol=5e-15)
```