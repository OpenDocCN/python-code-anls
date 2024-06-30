# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\test_waves.py`

```
# 导入从sympy库中的core.function模块的Derivative和Function类
# 导入Derivative和Function类
from sympy.core.function import (Derivative, Function)
# 导入sympy库中core.numbers模块的I和pi
from sympy.core.numbers import (I, pi)
# 导入sympy库中core.symbol模块的Symbol和symbols函数
from sympy.core.symbol import (Symbol, symbols)
# 导入sympy库中functions.elementary.miscellaneous模块的sqrt函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入sympy库中functions.elementary.trigonometric模块的atan2、cos、sin函数
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
# 导入sympy库中simplify.simplify模块的simplify函数
from sympy.simplify.simplify import simplify
# 导入sympy库中abc模块中的epsilon和mu
from sympy.abc import epsilon, mu
# 导入sympy库中functions.elementary.exponential模块的exp函数
from sympy.functions.elementary.exponential import exp
# 导入sympy.physics.units模块中的speed_of_light、m、s
from sympy.physics.units import speed_of_light, m, s
# 导入sympy.physics.optics模块的TWave类
from sympy.physics.optics import TWave
# 导入sympy.testing.pytest模块的raises函数，用于测试框架

# 将光速常数speed_of_light转换为单位m/s
c = speed_of_light.convert_to(m/s)

# 定义测试函数test_twave
def test_twave():
    # 定义符号变量A1、phi1、A2、phi2、f
    A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    # 定义折射率的符号变量n
    n = Symbol('n')  # Refractive index
    # 定义时间的符号变量t
    t = Symbol('t')  # Time
    # 定义空间变量的符号变量x
    x = Symbol('x')  # Spatial variable
    # 定义函数E(x, t)
    E = Function('E')
    
    # 创建TWave对象w1，其振幅为A1、频率为f、相位为phi1
    w1 = TWave(A1, f, phi1)
    # 创建TWave对象w2，其振幅为A2、频率为f、相位为phi2
    w2 = TWave(A2, f, phi2)
    
    # 断言语句，验证w1对象的属性
    assert w1.amplitude == A1
    assert w1.frequency == f
    assert w1.phase == phi1
    assert w1.wavelength == c/(f*n)
    assert w1.time_period == 1/f
    assert w1.angular_velocity == 2*pi*f
    assert w1.wavenumber == 2*pi*f*n/c
    assert w1.speed == c/n

    # 创建TWave对象w3，其为w1和w2的叠加
    w3 = w1 + w2
    # 断言语句，验证w3对象的属性
    assert w3.amplitude == sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    assert w3.frequency == f
    assert w3.phase == atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    assert w3.wavelength == c/(f*n)
    assert w3.time_period == 1/f
    assert w3.angular_velocity == 2*pi*f
    assert w3.wavenumber == 2*pi*f*n/c
    assert w3.speed == c/n
    assert simplify(w3.rewrite(sin) - w2.rewrite(sin) - w1.rewrite(sin)) == 0
    assert w3.rewrite('pde') == epsilon*mu*Derivative(E(x, t), t, t) + Derivative(E(x, t), x, x)
    assert w3.rewrite(cos) == sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*cos(pi*f*n*x*s/(149896229*m) - 2*pi*f*t + atan2(A1*sin(phi1)
        + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)))
    
    # 创建TWave对象w4，其振幅为A1、时间周期为1/f
    w4 = TWave(A1, None, 0, 1/f)
    # 断言语句，验证w4对象的频率属性
    assert w4.frequency == f

    # 创建TWave对象w5，其为w1和w2的差
    w5 = w1 - w2
    # 断言语句，验证w5对象的属性
    assert w5.amplitude == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2) + A2**2)
    assert w5.frequency == f
    assert w5.phase == atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1) - A2*cos(phi2))
    assert w5.wavelength == c/(f*n)
    assert w5.time_period == 1/f
    assert w5.angular_velocity == 2*pi*f
    assert w5.wavenumber == 2*pi*f*n/c
    assert w5.speed == c/n
    assert simplify(w5.rewrite(sin) - w1.rewrite(sin) + w2.rewrite(sin)) == 0
    assert w5.rewrite('pde') == epsilon*mu*Derivative(E(x, t), t, t) + Derivative(E(x, t), x, x)
    assert w5.rewrite(cos) == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*cos(-2*pi*f*t + atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1)
        - A2*cos(phi2)) + pi*s*f*n*x/(149896229*m))
    # 断言：使用 w5 对象的 rewrite 方法重写表达式 exp，并进行断言验证
    assert w5.rewrite(exp) == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*exp(I*(-2*pi*f*t + atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1)
        - A2*cos(phi2)) + pi*s*f*n*x/(149896229*m)))

    # 创建 w6 对象作为 w1 的两倍
    w6 = 2*w1
    # 断言：验证 w6 的振幅是否为 2*A1
    assert w6.amplitude == 2*A1
    # 断言：验证 w6 的频率是否为 f
    assert w6.frequency == f
    # 断言：验证 w6 的相位是否为 phi1
    assert w6.phase == phi1

    # 创建 w7 对象作为 w6 的负值
    w7 = -w6
    # 断言：验证 w7 的振幅是否为 -2*A1
    assert w7.amplitude == -2*A1
    # 断言：验证 w7 的频率是否为 f
    assert w7.frequency == f
    # 断言：验证 w7 的相位是否为 phi1
    assert w7.phase == phi1

    # 断言：验证 TWave(A1) 会引发 ValueError 异常
    raises(ValueError, lambda:TWave(A1))
    # 断言：验证 TWave(A1, f, phi1, t) 会引发 ValueError 异常
    raises(ValueError, lambda:TWave(A1, f, phi1, t))
```