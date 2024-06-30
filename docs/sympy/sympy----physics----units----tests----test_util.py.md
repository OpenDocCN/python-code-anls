# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_util.py`

```
from sympy.core.containers import Tuple  # 导入 Tuple 类，用于处理元组
from sympy.core.numbers import pi  # 导入 pi 常数
from sympy.core.power import Pow  # 导入 Pow 类，用于表示指数运算
from sympy.core.symbol import symbols  # 导入 symbols 函数，用于创建符号变量
from sympy.core.sympify import sympify  # 导入 sympify 函数，用于将字符串转换为 SymPy 表达式
from sympy.printing.str import sstr  # 导入 sstr 函数，用于将 SymPy 表达式转换为字符串
from sympy.physics.units import (
    G, centimeter, coulomb, day, degree, gram, hbar, hour, inch, joule, kelvin,
    kilogram, kilometer, length, meter, mile, minute, newton, planck,
    planck_length, planck_mass, planck_temperature, planck_time, radians,
    second, speed_of_light, steradian, time, km)  # 导入物理单位和常数
from sympy.physics.units.util import convert_to, check_dimensions  # 导入单位转换和维度检查函数
from sympy.testing.pytest import raises  # 导入 raises 函数，用于测试异常情况
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数，用于计算平方根


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)  # 定义 NS 函数，用于精确打印数值表达式


L = length  # 设置 L 为 length 单位的别名
T = time  # 设置 T 为 time 单位的别名


def test_dim_simplify_add():
    # 测试：L + L 应该等于 L
    assert L + L == L


def test_dim_simplify_mul():
    # 测试：L * T 应该等于 L * T
    assert L*T == L*T


def test_dim_simplify_pow():
    # 测试：L ** 2 应该等于 L 的平方
    assert Pow(L, 2) == L**2


def test_dim_simplify_rec():
    # 测试：(L + L) * T 应该等于 L * T
    assert (L + L) * T == L*T


def test_convert_to_quantities():
    # 测试单位转换函数 convert_to

    # 基本单位转换
    assert convert_to(3, meter) == 3

    # 复杂单位转换
    assert convert_to(mile, kilometer) == 25146*kilometer/15625
    assert convert_to(meter/second, speed_of_light) == speed_of_light/299792458
    assert convert_to(299792458*meter/second, speed_of_light) == speed_of_light
    assert convert_to(2*299792458*meter/second, speed_of_light) == 2*speed_of_light
    assert convert_to(speed_of_light, meter/second) == 299792458*meter/second
    assert convert_to(2*speed_of_light, meter/second) == 599584916*meter/second
    assert convert_to(day, second) == 86400*second
    assert convert_to(2*hour, minute) == 120*minute
    assert convert_to(mile, meter) == 201168*meter/125
    assert convert_to(mile/hour, kilometer/hour) == 25146*kilometer/(15625*hour)
    assert convert_to(3*newton, meter/second) == 3*newton
    assert convert_to(3*newton, kilogram*meter/second**2) == 3*meter*kilogram/second**2
    assert convert_to(kilometer + mile, meter) == 326168*meter/125
    assert convert_to(2*kilometer + 3*mile, meter) == 853504*meter/125
    assert convert_to(inch**2, meter**2) == 16129*meter**2/25000000
    assert convert_to(3*inch**2, meter) == 48387*meter**2/25000000
    assert convert_to(2*kilometer/hour + 3*mile/hour, meter/second) == 53344*meter/(28125*second)
    assert convert_to(2*kilometer/hour + 3*mile/hour, centimeter/second) == 213376*centimeter/(1125*second)
    assert convert_to(kilometer * (mile + kilometer), meter) == 2609344 * meter ** 2

    # 角度和立体角单位转换
    assert convert_to(steradian, coulomb) == steradian
    assert convert_to(radians, degree) == 180*degree/pi
    assert convert_to(radians, [meter, degree]) == 180*degree/pi
    assert convert_to(pi*radians, degree) == 180*degree
    assert convert_to(pi, degree) == 180*degree

    # 特殊情况测试链接
    # https://github.com/sympy/sympy/issues/26263
    # 断言语句：将计算结果转换为米单位后，应该与原始表达式的计算结果相等
    assert convert_to(sqrt(meter**2 + meter**2.0), meter) == sqrt(meter**2 + meter**2.0)
    
    # 断言语句：将计算结果转换为米单位后，应该与原始表达式的计算结果相等
    assert convert_to((meter**2 + meter**2.0)**2, meter) == (meter**2 + meter**2.0)**2
# 定义测试函数，用于测试 convert_to 函数的不同输入和输出
def test_convert_to_tuples_of_quantities():
    # 从 sympy.core.symbol 模块导入 symbols 函数
    from sympy.core.symbol import symbols

    # 创建两个符号变量 alpha 和 beta
    alpha, beta = symbols('alpha beta')

    # 断言，将 speed_of_light 转换为 [meter, second] 单位，预期结果为 299792458 * meter / second
    assert convert_to(speed_of_light, [meter, second]) == 299792458 * meter / second
    # 断言，将 speed_of_light 转换为 (meter, second) 单位，预期结果为 299792458 * meter / second
    assert convert_to(speed_of_light, (meter, second)) == 299792458 * meter / second
    # 断言，将 speed_of_light 转换为 Tuple(meter, second) 单位，预期结果为 299792458 * meter / second
    assert convert_to(speed_of_light, Tuple(meter, second)) == 299792458 * meter / second
    # 断言，将 joule 转换为 [meter, kilogram, second] 单位，预期结果为 kilogram*meter**2/second**2
    assert convert_to(joule, [meter, kilogram, second]) == kilogram*meter**2/second**2
    # 断言，将 joule 转换为 [centimeter, gram, second] 单位，预期结果为 10000000*centimeter**2*gram/second**2
    assert convert_to(joule, [centimeter, gram, second]) == 10000000*centimeter**2*gram/second**2
    # 断言，将 299792458*meter/second 转换为 [speed_of_light] 单位，预期结果为 speed_of_light
    assert convert_to(299792458*meter/second, [speed_of_light]) == speed_of_light
    # 断言，将 speed_of_light / 2 转换为 [meter, second, kilogram] 单位，预期结果为 meter/second*299792458 / 2
    assert convert_to(speed_of_light / 2, [meter, second, kilogram]) == meter/second*299792458 / 2
    # 断言，将 2 * speed_of_light 转换为 [meter, second, kilogram] 单位，预期结果为 2 * 299792458 * meter / second
    # 这里是一种转换测试，实际物理意义上不成立，但保留作为转换测试
    assert convert_to(2 * speed_of_light, [meter, second, kilogram]) == 2 * 299792458 * meter / second
    # 断言，将 G 转换为 [G, speed_of_light, planck] 单位，预期结果为 1.0*G
    assert convert_to(G, [G, speed_of_light, planck]) == 1.0*G

    # 断言，将 meter 转换为 [G, speed_of_light, hbar] 单位，并保留7位有效数字
    assert NS(convert_to(meter, [G, speed_of_light, hbar]), n=7) == '6.187142e+34*gravitational_constant**0.5000000*hbar**0.5000000/speed_of_light**1.500000'
    # 断言，将 planck_mass 转换为 kilogram 单位，并保留7位有效数字
    assert NS(convert_to(planck_mass, kilogram), n=7) == '2.176434e-8*kilogram'
    # 断言，将 planck_length 转换为 meter 单位，并保留7位有效数字
    assert NS(convert_to(planck_length, meter), n=7) == '1.616255e-35*meter'
    # 断言，将 planck_time 转换为 second 单位，并保留6位有效数字
    assert NS(convert_to(planck_time, second), n=6) == '5.39125e-44*second'
    # 断言，将 planck_temperature 转换为 kelvin 单位，并保留7位有效数字
    assert NS(convert_to(planck_temperature, kelvin), n=7) == '1.416784e+32*kelvin'
    # 断言，将 meter 转换为 [G, speed_of_light, planck] 单位，再转换回 meter 单位，并保留10位有效数字
    assert NS(convert_to(convert_to(meter, [G, speed_of_light, planck]), meter), n=10) == '1.000000000*meter'

    # 类似于 https://github.com/sympy/sympy/issues/26263 的情况
    # 断言，将 sqrt(meter**2 + second**2.0) 转换为 [meter, second] 单位，预期结果为 sqrt(meter**2 + second**2.0)
    assert convert_to(sqrt(meter**2 + second**2.0), [meter, second]) == sqrt(meter**2 + second**2.0)
    # 断言，将 (meter**2 + second**2.0)**2 转换为 [meter, second] 单位，预期结果为 (meter**2 + second**2.0)**2

    # 类似于 https://github.com/sympy/sympy/issues/21463 的情况
    assert convert_to((meter**2 + second**2.0)**2, [meter, second]) == (meter**2 + second**2.0)**2


# 定义测试函数，用于测试 evaluate 和 simplify 操作
def test_eval_simplify():
    # 从 sympy.physics.units 模块导入 cm, mm, km, m, K, kilo 等单位和 symbols 函数
    from sympy.physics.units import cm, mm, km, m, K, kilo
    # 从 sympy.core.symbol 模块导入 symbols 函数
    from sympy.core.symbol import symbols

    # 创建两个符号变量 x 和 y
    x, y = symbols('x y')

    # 断言，厘米与毫米的比较，预期结果为 10
    assert (cm/mm).simplify() == 10
    # 断言，千米与米的比较，预期结果为 1000
    assert (km/m).simplify() == 1000
    # 断言，千米与厘米的比较，预期结果为 100000
    assert (km/cm).simplify() == 100000
    # 断言，10*x*K*km**2/m/cm 的简化结果，预期结果为 1000000000*x*kelvin
    assert (10*x*K*km**2/m/cm).simplify() == 1000000000*x*kelvin
    # 断言，厘米与千米与米的比较，预期结果为 1/(10000000*centimeter)
    assert (cm/km/m).simplify() == 1/(10000000*centimeter)

    # 断言，3*kilo*meter 的简化结果，预期结果为 3000*meter
    assert (3*kilo*meter).simplify() == 3000*meter
    # 断言，4*kilo*meter/(2*kilometer) 的简化结果，预期结果为 2
    assert (4*kilo*meter/(2*kilometer)).simplify() == 2
    # 断言，4*kilometer**2/(kilo*meter)**2 的简化结果，预期结果为 4
    assert (4*kilometer**2/(kilo*meter)**2).simplify() == 4


# 定义测试函数，用于测试 quantity_simplify 函数
def test_quantity_simplify():
    # 从 sympy.physics.units.util 模块导入 quantity_simplify 函数，从 sympy.physics.units 模块导入 kilo, foot 单位，从 sympy.core.symbol 模块导入 symbols 函数
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import kilo, foot
    from sympy.core.symbol import symbols

    # 创建两个符号变量 x 和 y
    x, y = symbols('x y')

    # 断言，对 x*(8*kilo*newton*meter + y) 进行 quantity_simplify 操作，预期结果为 x*(8000*meter*newton + y)
    assert quantity_simplify(x*(8*kilo*newton*m
    # 断言：简化量纲表达式，并验证结果是否等于规定的量纲表达式
    assert quantity_simplify(foot*inch*(foot + inch)) == foot**2*(foot + foot/12)/12
    
    # 断言：简化量纲表达式，并验证结果是否等于规定的量纲表达式
    assert quantity_simplify(foot*inch*(foot*foot + inch*(foot + inch))) == foot**2*(foot**2 + foot/12*(foot + foot/12))/12
    
    # 断言：简化量纲表达式，并验证结果是否等于规定的量纲表达式
    assert quantity_simplify(2**(foot/inch*kilo/1000)*inch) == 4096*foot/12
    
    # 断言：简化量纲表达式，并验证结果是否等于规定的量纲表达式
    assert quantity_simplify(foot**2*inch + inch**2*foot) == 13*foot**3/144
# 定义测试函数，用于简化跨维度的量纲
def test_quantity_simplify_across_dimensions():
    # 导入量纲简化函数和各种物理单位
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import ampere, ohm, volt, joule, pascal, farad, second, watt, siemens, henry, tesla, weber, hour, newton

    # 断言语句，验证量纲简化函数的正确性
    assert quantity_simplify(ampere*ohm, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(6*ampere*ohm, across_dimensions=True, unit_system="SI") == 6*volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(volt/ohm, across_dimensions=True, unit_system="SI") == ampere
    assert quantity_simplify(joule/meter**3, across_dimensions=True, unit_system="SI") == pascal
    assert quantity_simplify(farad*ohm, across_dimensions=True, unit_system="SI") == second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt
    assert quantity_simplify(meter**3/second, across_dimensions=True, unit_system="SI") == meter**3/second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt

    # 更多断言语句，验证量纲简化在不同单位组合下的行为
    assert quantity_simplify(joule/coulomb, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(ampere/volt, across_dimensions=True, unit_system="SI") == siemens
    assert quantity_simplify(coulomb/volt, across_dimensions=True, unit_system="SI") == farad
    assert quantity_simplify(volt*second/ampere, across_dimensions=True, unit_system="SI") == henry
    assert quantity_simplify(volt*second/meter**2, across_dimensions=True, unit_system="SI") == tesla
    assert quantity_simplify(joule/ampere, across_dimensions=True, unit_system="SI") == weber

    # 单位组合中包含数值的情况
    assert quantity_simplify(5*kilometer/hour, across_dimensions=True, unit_system="SI") == 25*meter/(18*second)
    assert quantity_simplify(5*kilogram*meter/second**2, across_dimensions=True, unit_system="SI") == 5*newton

# 定义测试函数，用于检查量纲
def test_check_dimensions():
    # 导入符号 'x'
    x = symbols('x')
    
    # 断言语句，验证检查量纲函数的行为
    assert check_dimensions(inch + x) == inch + x
    assert check_dimensions(length + x) == length + x
    
    # 替换后得到2*length；检查将清除常数
    assert check_dimensions((length + x).subs(x, length)) == length
    assert check_dimensions(newton*meter + joule) == joule + meter*newton
    
    # 引发异常，测试不合法的量纲组合
    raises(ValueError, lambda: check_dimensions(inch + 1))
    raises(ValueError, lambda: check_dimensions(length + 1))
    raises(ValueError, lambda: check_dimensions(length + time))
    raises(ValueError, lambda: check_dimensions(meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + 3 * second))
    raises(ValueError, lambda: check_dimensions(1 / second + 1 / meter))
    raises(ValueError, lambda: check_dimensions(2 * meter*(mile + centimeter) + km))
```