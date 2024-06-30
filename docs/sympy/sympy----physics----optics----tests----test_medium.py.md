# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\test_medium.py`

```
# 导入必要的模块和函数
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.optics import Medium
from sympy.abc import epsilon, mu, n
from sympy.physics.units import speed_of_light, u0, e0, m, kg, s, A

from sympy.testing.pytest import raises

# 将光速单位转换为米每秒
c = speed_of_light.convert_to(m/s)
# 将电场常数单位转换为安培的平方秒的四次方除以千克米的三次方
e0 = e0.convert_to(A**2*s**4/(kg*m**3))
# 将磁场常数单位转换为米千克除以安培的平方秒的平方
u0 = u0.convert_to(m*kg/(A**2*s**2))

# 定义测试函数 test_medium
def test_medium():
    # 创建介质对象 m1，不指定电介质常数和磁导率
    m1 = Medium('m1')
    # 断言 m1 的本征阻抗等于电场常数和磁场常数的平方根
    assert m1.intrinsic_impedance == sqrt(u0/e0)
    # 断言 m1 的速度等于电场常数和磁场常数的乘积的平方根的倒数
    assert m1.speed == 1/sqrt(e0*u0)
    # 断言 m1 的折射率等于光速乘以电场常数和磁场常数的平方根
    assert m1.refractive_index == c*sqrt(e0*u0)
    # 断言 m1 的电介质常数等于电场常数
    assert m1.permittivity == e0
    # 断言 m1 的磁导率等于磁场常数
    assert m1.permeability == u0
    
    # 创建介质对象 m2，指定电介质常数和磁导率
    m2 = Medium('m2', epsilon, mu)
    # 断言 m2 的本征阻抗等于磁导率除以电介质常数的平方根
    assert m2.intrinsic_impedance == sqrt(mu/epsilon)
    # 断言 m2 的速度等于电介质常数和磁导率的乘积的平方根的倒数
    assert m2.speed == 1/sqrt(epsilon*mu)
    # 断言 m2 的折射率等于光速乘以电介质常数和磁导率的平方根
    assert m2.refractive_index == c*sqrt(epsilon*mu)
    # 断言 m2 的电介质常数等于给定的电介质常数
    assert m2.permittivity == epsilon
    # 断言 m2 的磁导率等于给定的磁导率
    assert m2.permeability == mu
    
    # 创建介质对象 m3，增加其电场常数和磁导率的微小值（相对于真空中的值）
    m3 = Medium('m3', 9.0*10**(-12)*s**4*A**2/(m**3*kg), 1.45*10**(-6)*kg*m/(A**2*s**2))
    # 断言 m3 的折射率大于 m1 的折射率
    assert m3.refractive_index > m1.refractive_index
    # 断言 m3 不等于 m1
    assert m3 != m1
    
    # 创建介质对象 m4，减少其电场常数和磁导率的微小值（相对于真空中的值）
    m4 = Medium('m4', 7.0*10**(-12)*s**4*A**2/(m**3*kg), 1.15*10**(-6)*kg*m/(A**2*s**2))
    # 断言 m4 的折射率小于 m1 的折射率
    assert m4.refractive_index < m1.refractive_index
    
    # 创建介质对象 m5，指定电场常数和折射率，其他参数使用默认值
    m5 = Medium('m5', permittivity=710*10**(-12)*s**4*A**2/(m**3*kg), n=1.33)
    # 断言 m5 的本征阻抗与给定值之间的差小于给定的容差
    assert abs(m5.intrinsic_impedance - 6.24845417765552*kg*m**2/(A**2*s**3)) \
                < 1e-12*kg*m**2/(A**2*s**3)
    # 断言 m5 的速度与给定值之间的差小于给定的容差
    assert abs(m5.speed - 225407863.157895*m/s) < 1e-6*m/s
    # 断言 m5 的折射率与给定值之间的差小于给定的容差
    assert abs(m5.refractive_index - 1.33000000000000) < 1e-12
    # 断言 m5 的电场常数与给定值之间的差小于给定的容差
    assert abs(m5.permittivity - 7.1e-10*A**2*s**4/(kg*m**3)) \
                < 1e-20*A**2*s**4/(kg*m**3)
    # 断言 m5 的磁导率与给定值之间的差小于给定的容差
    assert abs(m5.permeability - 2.77206575232851e-8*kg*m/(A**2*s**2)) \
                < 1e-20*kg*m/(A**2*s**2)
    
    # 创建介质对象 m6，只指定磁导率和折射率，电场常数使用默认值
    m6 = Medium('m6', None, mu, n)
    # 断言 m6 的电场常数等于折射率的平方除以光速的平方乘以磁导率
    assert m6.permittivity == n**2/(c**2*mu)
    
    # 断言介质对象 m7 和 m8 的折射率相等
    assert Medium('m7').refractive_index == Medium('m8', e0, u0).refractive_index
    
    # 断言创建介质对象 m9 时会引发 ValueError 异常
    raises(ValueError, lambda:Medium('m9', e0, u0, 2))
```