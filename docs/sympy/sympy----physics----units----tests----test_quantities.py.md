# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_quantities.py`

```
import warnings  # 导入警告模块

from sympy.core.add import Add  # 导入 SymPy 中的加法操作
from sympy.core.function import (Function, diff)  # 导入 SymPy 中的函数和求导函数
from sympy.core.numbers import (Number, Rational)  # 导入 SymPy 中的数值和有理数
from sympy.core.singleton import S  # 导入 SymPy 中的单例对象 S
from sympy.core.symbol import (Symbol, symbols)  # 导入 SymPy 中的符号和符号集合
from sympy.functions.elementary.complexes import Abs  # 导入 SymPy 中的复数相关函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入 SymPy 中的指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 SymPy 中的平方根函数
from sympy.functions.elementary.trigonometric import sin  # 导入 SymPy 中的正弦函数
from sympy.integrals.integrals import integrate  # 导入 SymPy 中的积分函数
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,  # 导入 SymPy 中的单位
                                 volume, kilometer, joule, molar_gas_constant,
                                 vacuum_permittivity, elementary_charge, volt,
                                 ohm)
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,  # 导入 SymPy 中的单位定义
    day, foot, grams, hour, inch, kg, km, m, meter, millimeter,
    minute, quart, s, second, speed_of_light, bit,
    byte, kibibyte, mebibyte, gibibyte, tebibyte, pebibyte, exbibyte,
    kilogram, gravitational_constant, electron_rest_mass)

from sympy.physics.units.definitions.dimension_definitions import (  # 导入 SymPy 中的维度定义
    Dimension, charge, length, time, temperature, pressure,
    energy, mass
)
from sympy.physics.units.prefixes import PREFIXES, kilo  # 导入 SymPy 中的单位前缀和千分位前缀
from sympy.physics.units.quantities import PhysicalConstant, Quantity  # 导入 SymPy 中的物理常数和物理量
from sympy.physics.units.systems import SI  # 导入 SymPy 中的国际单位制
from sympy.testing.pytest import raises  # 导入 SymPy 测试模块中的 raises 函数

k = PREFIXES["k"]  # 获取单位前缀中的千分位前缀

def test_str_repr():
    assert str(kg) == "kilogram"  # 断言 kilogram 的字符串表示为 "kilogram"

def test_eq():
    # 简单的相等性测试
    assert 10*m == 10*m  # 断言 10 米等于 10 米
    assert 10*m != 10*s  # 断言 10 米不等于 10 秒

def test_convert_to():
    q = Quantity("q1")  # 创建一个量 q1
    q.set_global_relative_scale_factor(S(5000), meter)  # 设置量 q1 的全局相对比例因子为 5000，单位为米

    assert q.convert_to(m) == 5000*m  # 断言将 q1 转换为米后的结果为 5000 米

    assert speed_of_light.convert_to(m / s) == 299792458 * m / s  # 断言将光速转换为米每秒后的结果为 299792458 米每秒
    assert day.convert_to(s) == 86400*s  # 断言将天转换为秒后的结果为 86400 秒

    # 错误的转换维度:
    assert q.convert_to(s) == q  # 断言将 q1 转换为秒后仍然是 q1
    assert speed_of_light.convert_to(m) == speed_of_light  # 断言将光速转换为米后仍然是光速

    expr = joule*second  # 创建能量乘以时间的表达式
    conv = convert_to(expr, joule)  # 将表达式转换为焦耳单位
    assert conv == joule*second  # 断言转换后的结果为焦耳乘以秒

def test_Quantity_definition():
    q = Quantity("s10", abbrev="sabbr")  # 创建一个名为 s10 的量，缩写为 sabbr
    q.set_global_relative_scale_factor(10, second)  # 设置量 s10 的全局相对比例因子为 10，单位为秒
    u = Quantity("u", abbrev="dam")  # 创建一个名为 u 的量，缩写为 dam
    u.set_global_relative_scale_factor(10, meter)  # 设置量 u 的全局相对比例因子为 10，单位为米
    km = Quantity("km")  # 创建一个名为 km 的量
    km.set_global_relative_scale_factor(kilo, meter)  # 设置量 km 的全局相对比例因子为千乘以米
    v = Quantity("u")  # 创建一个名为 u 的量
    v.set_global_relative_scale_factor(5*kilo, meter)  # 设置量 u 的全局相对比例因子为 5 千乘以米

    assert q.scale_factor == 10  # 断言量 s10 的比例因子为 10
    assert q.dimension == time  # 断言量 s10 的维度为时间
    assert q.abbrev == Symbol("sabbr")  # 断言量 s10 的缩写为 sabbr

    assert u.dimension == length  # 断言量 u 的维度为长度
    assert u.scale_factor == 10  # 断言量 u 的比例因子为 10
    assert u.abbrev == Symbol("dam")  # 断言量 u 的缩写为 dam

    assert km.scale_factor == 1000  # 断言量 km 的比例因子为 1000
    assert km.func(*km.args) == km  # 断言量 km 的函数形式等于自身
    assert km.func(*km.args).args == km.args  # 断言量 km 的函数形式的参数等于自身的参数

    assert v.dimension == length  # 断言量 u 的维度为长度
    assert v.scale_factor == 5000  # 断言量 u 的比例因子为 5000

def test_abbrev():
    u = Quantity("u")  # 创建一个名为 u 的量
    u.set_global_relative_scale_factor(S.One, meter)  # 设置量 u 的全局相对比例因子为 1，单位为米

    assert u.name == Symbol("u")  # 断言量 u 的名称为 u
    # 断言：验证对象 u 的缩写属性等于符号 "u"
    assert u.abbrev == Symbol("u")
    
    # 创建一个名为 "u" 的 Quantity 对象，设置其缩写为 "om"
    u = Quantity("u", abbrev="om")
    # 设置全局相对比例因子为 2，基于米的单位
    u.set_global_relative_scale_factor(S(2), meter)
    
    # 断言：验证对象 u 的名称属性等于符号 "u"
    assert u.name == Symbol("u")
    # 断言：验证对象 u 的缩写属性等于符号 "om"
    assert u.abbrev == Symbol("om")
    # 断言：验证对象 u 的比例因子等于 2
    assert u.scale_factor == 2
    # 断言：验证对象 u 的比例因子是 Number 类型的实例
    assert isinstance(u.scale_factor, Number)
    
    # 创建一个名为 "u" 的 Quantity 对象，设置其缩写为 "ikm"
    u = Quantity("u", abbrev="ikm")
    # 设置全局相对比例因子为 3000，基于千米和米的单位
    u.set_global_relative_scale_factor(3*kilo, meter)
    
    # 断言：验证对象 u 的缩写属性等于符号 "ikm"
    assert u.abbrev == Symbol("ikm")
    # 断言：验证对象 u 的比例因子等于 3000
    assert u.scale_factor == 3000
def test_print():
    # 创建一个名为 "unitname" 的 Quantity 对象，缩写为 "dam"
    u = Quantity("unitname", abbrev="dam")
    # 断言该对象的字符串表示为 "unitname"
    assert repr(u) == "unitname"
    # 断言该对象的字符串表示为 "unitname"
    assert str(u) == "unitname"


def test_Quantity_eq():
    # 创建两个 Quantity 对象，分别为 u 和 v
    u = Quantity("u", abbrev="dam")
    v = Quantity("v1")
    # 断言 u 不等于 v
    assert u != v
    # 修改 v 的属性，再次断言 u 不等于 v
    v = Quantity("v2", abbrev="ds")
    assert u != v
    # 再次修改 v 的属性，再次断言 u 不等于 v
    v = Quantity("v3", abbrev="dm")
    assert u != v


def test_add_sub():
    # 创建三个 Quantity 对象，u、v、w
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    # 为 u、v、w 设置全局的相对比例因子
    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    # 断言 u + v 返回一个 Add 对象
    assert isinstance(u + v, Add)
    # 断言 (u + v 转换到 u 的单位) 等于 (1 + S.Half)*u
    assert (u + v.convert_to(u)) == (1 + S.Half)*u
    # 断言 u - v 返回一个 Add 对象
    assert isinstance(u - v, Add)
    # 断言 (u - v 转换到 u 的单位) 等于 S.Half*u
    assert (u - v.convert_to(u)) == S.Half*u


def test_quantity_abs():
    # 创建三个 Quantity 对象，v_w1、v_w2、v_w3
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w3 = Quantity('v_w3')

    # 为 v_w1、v_w2、v_w3 设置全局的相对比例因子
    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)
    v_w3.set_global_relative_scale_factor(1, meter/second)

    # 构造表达式 expr
    expr = v_w3 - Abs(v_w1 - v_w2)

    # 断言 SI.get_dimensional_expr(v_w1) 等于 (length/time).name
    assert SI.get_dimensional_expr(v_w1) == (length/time).name

    # 创建 Dimension 对象 Dq
    Dq = Dimension(SI.get_dimensional_expr(expr))

    # 断言 SI.get_dimension_system().get_dimensional_dependencies(Dq) 等于指定的维度依赖关系
    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: 1,
        time: -1,
    }
    # 断言 meter 等于 sqrt(meter**2)
    assert meter == sqrt(meter**2)


def test_check_unit_consistency():
    # 创建三个 Quantity 对象，u、v、w
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    # 为 u、v、w 设置全局的相对比例因子
    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    # 定义函数 check_unit_consistency，并尝试在它上面引发异常
    def check_unit_consistency(expr):
        SI._collect_factor_and_dimension(expr)

    # 断言调用 check_unit_consistency(u + w) 会引发 ValueError 异常
    raises(ValueError, lambda: check_unit_consistency(u + w))
    # 断言调用 check_unit_consistency(u - w) 会引发 ValueError 异常
    raises(ValueError, lambda: check_unit_consistency(u - w))
    # 断言调用 check_unit_consistency(u + 1) 会引发 ValueError 异常
    raises(ValueError, lambda: check_unit_consistency(u + 1))
    # 断言调用 check_unit_consistency(u - 1) 会引发 ValueError 异常
    raises(ValueError, lambda: check_unit_consistency(u - 1))
    # 断言调用 check_unit_consistency(1 - exp(u / w)) 会引发 ValueError 异常
    raises(ValueError, lambda: check_unit_consistency(1 - exp(u / w)))


def test_mul_div():
    # 创建五个 Quantity 对象，u、v、t、ut、v2
    u = Quantity("u")
    v = Quantity("v")
    t = Quantity("t")
    ut = Quantity("ut")
    v2 = Quantity("v")

    # 为 u、v、t、ut、v2 设置全局的相对比例因子
    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    t.set_global_relative_scale_factor(S(2), second)
    ut.set_global_relative_scale_factor(S(20), meter*second)
    v2.set_global_relative_scale_factor(S(5), meter/second)

    # 断言 1 / u 等于 u**(-1)
    assert 1 / u == u**(-1)
    # 断言 u / 1 等于 u
    assert u / 1 == u

    # 创建 v1 为 u / t，v2 为 v
    v1 = u / t
    v2 = v

    # 断言 v1 不等于 v2
    assert v1 != v2
    # 断言 v1 等于 v2 转换到 v1 的单位
    assert v1 == v2.convert_to(v1)

    # TODO: decide whether to allow such expression in the future
    # (requires somehow manipulating the core).
    # 断言 u / Quantity('l2', dimension=length, scale_factor=2) == 5

    # 断言 u * 1 等于 u
    assert u * 1 == u

    # 创建 ut1 为 u * t，ut2 为 ut
    ut1 = u * t
    ut2 = ut

    # 断言 ut1 不等于 ut2
    assert ut1 != ut2
    # 断言 ut1 等于 ut2 转换到 ut1 的单位
    assert ut1 == ut2.convert_to(ut1)

    # Mul only supports structural equality:
    # Mul 只支持结构上的相等性
    # 创建名为 lp1 的 Quantity 对象
    lp1 = Quantity("lp1")
    # 设置 lp1 对象的全局相对比例因子为 2，单位是每米（1/meter）
    lp1.set_global_relative_scale_factor(S(2), 1/meter)
    # 断言 u 乘以 lp1 不等于 20

    assert u * lp1 != 20

    # 断言 u 的 0 次方等于 1
    assert u**0 == 1
    # 断言 u 的 1 次方等于 u

    assert u**1 == u

    # TODO: Pow 只支持结构相等性:
    # 创建名为 u2 和 u3 的 Quantity 对象
    u2 = Quantity("u2")
    u3 = Quantity("u3")
    # 设置 u2 对象的全局相对比例因子为 100，单位是平方米（meter**2）
    u2.set_global_relative_scale_factor(S(100), meter**2)
    # 设置 u3 对象的全局相对比例因子为 1/10，单位是每米（1/meter）
    u3.set_global_relative_scale_factor(Rational(1, 10), 1/meter)

    # 断言 u 的平方不等于 u2
    assert u ** 2 != u2
    # 断言 u 的负一次方不等于 u3
    assert u ** -1 != u3

    # 断言 u 的平方等于 u2 转换为 u 的结果
    assert u ** 2 == u2.convert_to(u)
    # 断言 u 的负一次方等于 u3 转换为 u 的结果
    assert u ** -1 == u3.convert_to(u)
# 定义测试函数，用于验证单位转换和数学运算的正确性
def test_units():
    # 断言：将速度、时间和距离单位转换为相同单位，并验证结果是否为432
    assert convert_to((5*m/s * day) / km, 1) == 432
    # 断言：将英尺与米单位进行换算，验证结果是否为3048/10000
    assert convert_to(foot / meter, meter) == Rational(3048, 10000)
    
    # 由于amu是纯质量单位，质量除以质量得到数量而不是摩尔量，验证结果是否为6.0e+23
    # TODO: 需要更好的简化例程：
    assert str(convert_to(grams/amu, grams).n(2)) == '6.0e+23'

    # 太阳光需要大约8.3分钟才能到达地球
    t = (1*au / speed_of_light) / minute
    # TODO: 需要一种更好的方法来简化包含单位的表达式：
    t = convert_to(convert_to(t, meter / minute), meter)
    # 断言：将时间单位换算为秒，验证结果是否等于49865956897/5995849160
    assert t.simplify() == Rational(49865956897, 5995849160)

    # TODO: 修复此处应该返回没有Abs的m
    # 断言：验证平方根m的平方是否等于m
    assert sqrt(m**2) == m
    # 断言：验证平方根m的平方是否等于m
    assert (sqrt(m))**2 == m

    # 定义符号t
    t = Symbol('t')
    # 断言：对t*m/s进行积分，从1秒到5秒，验证结果是否为12*m*s
    assert integrate(t*m/s, (t, 1*s, 5*s)) == 12*m*s
    # 断言：对t*m/s进行积分，从1秒到5秒，验证结果是否为12*m*s
    assert (t * m/s).integrate((t, 1*s, 5*s)) == 12*m*s


# 定义测试函数，验证在特定情况下的单位转换问题
def test_issue_quart():
    # 断言：将4夸脱/立方英寸转换为米，验证结果是否为231
    assert convert_to(4 * quart / inch ** 3, meter) == 231
    # 断言：将4夸脱/立方英寸转换为毫米，验证结果是否为231
    assert convert_to(4 * quart / inch ** 3, millimeter) == 231


# 定义测试函数，验证电子静止质量的单位转换
def test_electron_rest_mass():
    # 断言：将电子的静止质量转换为千克，验证结果是否为9.1093837015e-31*kilogram
    assert convert_to(electron_rest_mass, kilogram) == 9.1093837015e-31*kilogram
    # 断言：将电子的静止质量转换为克，验证结果是否为9.1093837015e-28*grams
    assert convert_to(electron_rest_mass, grams) == 9.1093837015e-28*grams


# 定义测试函数，验证问题编号5565的单位关系
def test_issue_5565():
    # 断言：验证m是否小于s的关系是否是一个关系型
    assert (m < s).is_Relational


# 定义测试函数，验证单位的查找功能
def test_find_unit():
    # 断言：查找'coulomb'相关的单位，验证结果是否为['coulomb', 'coulombs', 'coulomb_constant']
    assert find_unit('coulomb') == ['coulomb', 'coulombs', 'coulomb_constant']
    # 断言：查找coulomb单位相关的单位，验证结果是否为['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(coulomb) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    # 断言：查找charge单位相关的单位，验证结果是否为['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(charge) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    # 断言：查找inch单位相关的单位，验证结果是否为一系列长度单位列表
    assert find_unit(inch) == [
        'm', 'au', 'cm', 'dm', 'ft', 'km', 'ly', 'mi', 'mm', 'nm', 'pm', 'um', 'yd',
        'nmi', 'feet', 'foot', 'inch', 'mile', 'yard', 'meter', 'miles', 'yards',
        'inches', 'meters', 'micron', 'microns', 'angstrom', 'angstroms', 'decimeter',
        'kilometer', 'lightyear', 'nanometer', 'picometer', 'centimeter', 'decimeters',
        'kilometers', 'lightyears', 'micrometer', 'millimeter', 'nanometers', 'picometers',
        'centimeters', 'micrometers', 'millimeters', 'nautical_mile', 'planck_length',
        'nautical_miles', 'astronomical_unit', 'astronomical_units']
    # 断言：查找inch的倒数单位相关的单位，验证结果是否为['D', 'dioptre', 'optical_power']
    assert find_unit(inch**-1) == ['D', 'dioptre', 'optical_power']
    # 断言：查找长度的倒数单位相关的单位，验证结果是否为['D', 'dioptre', 'optical_power']
    assert find_unit(length**-1) == ['D', 'dioptre', 'optical_power']
    # 断言：查找inch的平方单位相关的单位，验证结果是否为['ha', 'hectare', 'planck_area']
    assert find_unit(inch ** 2) == ['ha', 'hectare', 'planck_area']
    # 断言：查找inch的立方单位相关的单位，验证结果是否为一系列体积单位列表
    assert find_unit(inch ** 3) == [
        'L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter', 'quart', 'liters', 'quarts',
        'deciliter', 'centiliter', 'deciliters', 'milliliter',
        'centiliters', 'milliliters', 'planck_volume']
    # 断言：查找'voltage'相关的单位，验证结果是否为['V', 'v', 'volt', 'volts', 'planck_voltage']
    assert find_unit('voltage') == ['V', 'v', 'volt', 'volts', 'planck_voltage']
    # 断言：确认函数 find_unit(grams) 的返回值与预期的列表相等
    assert find_unit(grams) == ['g', 't', 'Da', 'kg', 'me', 'mg', 'ug', 'amu', 'mmu', 'amus',
                                'gram', 'mmus', 'grams', 'pound', 'tonne', 'dalton', 'pounds',
                                'kilogram', 'kilograms', 'microgram', 'milligram', 'metric_ton',
                                'micrograms', 'milligrams', 'planck_mass', 'milli_mass_unit', 'atomic_mass_unit',
                                'electron_rest_mass', 'atomic_mass_constant']
# 定义测试函数 test_Quantity_derivative
def test_Quantity_derivative():
    # 定义符号变量 x
    x = symbols("x")
    # 断言求导结果，对常量倍数的长度单位进行求导应该得到长度单位
    assert diff(x*meter, x) == meter
    # 断言对 x**3*meter**2 求导结果应为 3*x**2*meter**2
    assert diff(x**3*meter**2, x) == 3*x**2*meter**2
    # 断言对长度单位 meter 求自身的导数应为 1
    assert diff(meter, meter) == 1
    # 断言对长度单位 meter**2 求导结果应为 2*meter
    assert diff(meter**2, meter) == 2*meter

# 定义测试函数 test_quantity_postprocessing
def test_quantity_postprocessing():
    # 创建两个量 q1 和 q2
    q1 = Quantity('q1')
    q2 = Quantity('q2')
    # 设置 q1 和 q2 的量纲
    SI.set_quantity_dimension(q1, length*pressure**2*temperature/time)
    SI.set_quantity_dimension(q2, energy*pressure*temperature/(length**2*time))
    # 断言 q1 + q2 的结果
    assert q1 + q2
    # 计算 q1 + q2 的维度表达式，并检查其维度依赖关系
    q = q1 + q2
    Dq = Dimension(SI.get_dimensional_expr(q))
    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: -1,
        mass: 2,
        temperature: 1,
        time: -5,
    }

# 定义测试函数 test_factor_and_dimension
def test_factor_and_dimension():
    # 断言 SI._collect_factor_and_dimension 对常量的处理结果
    assert (3000, Dimension(1)) == SI._collect_factor_and_dimension(3000)
    # 断言 SI._collect_factor_and_dimension 对长度单位表达式 meter + km 的处理结果
    assert (1001, length) == SI._collect_factor_and_dimension(meter + km)
    # 断言 SI._collect_factor_and_dimension 对复杂单位表达式的处理结果
    assert (2, length/time) == SI._collect_factor_and_dimension(
        meter/second + 36*km/(10*hour))
    # 定义符号变量 x, y
    x, y = symbols('x y')
    # 断言 SI._collect_factor_and_dimension 对混合单位表达式的处理结果
    assert (x + y/100, length) == SI._collect_factor_and_dimension(
        x*m + y*centimeter)
    # 创建量 cH 并设置其量纲
    cH = Quantity('cH')
    SI.set_quantity_dimension(cH, amount_of_substance/volume)
    # 计算 pH，并断言 SI._collect_factor_and_dimension 对其处理结果
    pH = -log(cH)
    assert (1, volume/amount_of_substance) == SI._collect_factor_and_dimension(
        exp(pH))
    # 创建量 v_w1 和 v_w2，并设置其相对比例因子
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w1.set_global_relative_scale_factor(Rational(3, 2), meter/second)
    v_w2.set_global_relative_scale_factor(2, meter/second)
    # 计算表达式 expr，并断言 SI._collect_factor_and_dimension 对其处理结果
    expr = Abs(v_w1/2 - v_w2)
    assert (Rational(5, 4), length/time) == \
        SI._collect_factor_and_dimension(expr)
    # 计算表达式 expr，并断言 SI._collect_factor_and_dimension 对其处理结果
    expr = Rational(5, 2)*second/meter*v_w1 - 3000
    assert (-(2996 + Rational(1, 4)), Dimension(1)) == \
        SI._collect_factor_and_dimension(expr)
    # 计算表达式 expr，并断言 SI._collect_factor_and_dimension 对其处理结果
    expr = v_w1**(v_w2/v_w1)
    assert ((Rational(3, 2))**Rational(4, 3), (length/time)**Rational(4, 3)) == \
        SI._collect_factor_and_dimension(expr)

# 定义测试函数 test_dimensional_expr_of_derivative
def test_dimensional_expr_of_derivative():
    # 创建量 l, t, t1，并设置它们的相对比例因子
    l = Quantity('l')
    t = Quantity('t')
    t1 = Quantity('t1')
    l.set_global_relative_scale_factor(36, km)
    t.set_global_relative_scale_factor(1, hour)
    t1.set_global_relative_scale_factor(1, second)
    # 创建符号变量 x, y 和函数 f
    x = Symbol('x')
    y = Symbol('y')
    f = Function('f')
    # 计算函数 f(x, y) 对 x, y 的偏导数，并赋值给 dfdx
    dfdx = f(x, y).diff(x, y)
    # 计算 dl_dt，并断言 SI.get_dimensional_expr 对其的处理结果
    dl_dt = dfdx.subs({f(x, y): l, x: t, y: t1})
    assert SI.get_dimensional_expr(dl_dt) ==\
        SI.get_dimensional_expr(l / t / t1) ==\
        Symbol("length")/Symbol("time")**2
    # 断言 SI._collect_factor_and_dimension 对 dl_dt 的处理结果
    assert SI._collect_factor_and_dimension(dl_dt) ==\
        SI._collect_factor_and_dimension(l / t / t1) ==\
        (10, length/time**2)

# 定义测试函数 test_get_dimensional_expr_with_function
def test_get_dimensional_expr_with_function():
    # 创建量 v_w1 和 v_w2，并设置其相对比例因子
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)
    # 断言 SI.get_dimensional_expr 对 sin(v_w1) 的处理结果
    assert SI.get_dimensional_expr(sin(v_w1)) == \
        sin(SI.get_dimensional_expr(v_w1))
    # 断言 SI.get_dimensional_expr 对 sin(v_w1/v_w2) 的处理结果
    assert SI.get_dimensional_expr(sin(v_w1/v_w2)) == 1
# 测试函数，验证单位转换函数的正确性
def test_binary_information():
    # 断言：将 kibibyte 转换为 byte 应为 1024 字节
    assert convert_to(kibibyte, byte) == 1024*byte
    # 断言：将 mebibyte 转换为 byte 应为 1024^2 字节
    assert convert_to(mebibyte, byte) == 1024**2*byte
    # 断言：将 gibibyte 转换为 byte 应为 1024^3 字节
    assert convert_to(gibibyte, byte) == 1024**3*byte
    # 断言：将 tebibyte 转换为 byte 应为 1024^4 字节
    assert convert_to(tebibyte, byte) == 1024**4*byte
    # 断言：将 pebibyte 转换为 byte 应为 1024^5 字节
    assert convert_to(pebibyte, byte) == 1024**5*byte
    # 断言：将 exbibyte 转换为 byte 应为 1024^6 字节
    assert convert_to(exbibyte, byte) == 1024**6*byte

    # 断言：将 kibibyte 转换为 bit 应为 8*1024 位
    assert kibibyte.convert_to(bit) == 8*1024*bit
    # 断言：将 byte 转换为 bit 应为 8 位
    assert byte.convert_to(bit) == 8*bit

    # 计算 a 的值，10 kibibyte * hour
    a = 10*kibibyte*hour

    # 断言：将 a 转换为 byte 应为 10240 字节 * 小时
    assert convert_to(a, byte) == 10240*byte*hour
    # 断言：将 a 转换为 minute 应为 600 kibibyte * minute
    assert convert_to(a, minute) == 600*kibibyte*minute
    # 断言：将 a 转换为 [byte, minute] 应为 614400 字节 * 分钟
    assert convert_to(a, [byte, minute]) == 614400*byte*minute


# 测试函数，验证使用两个非标准维度的单位转换
def test_conversion_with_2_nonstandard_dimensions():
    # 创建量纲对象 good_grade, kilo_good_grade, centi_good_grade
    good_grade = Quantity("good_grade")
    kilo_good_grade = Quantity("kilo_good_grade")
    centi_good_grade = Quantity("centi_good_grade")

    # 设置 kilo_good_grade 相对于 good_grade 的全局比例因子为 1000
    kilo_good_grade.set_global_relative_scale_factor(1000, good_grade)
    # 设置 centi_good_grade 相对于 kilo_good_grade 的全局比例因子为 1/10**5
    centi_good_grade.set_global_relative_scale_factor(S.One/10**5, kilo_good_grade)

    # 创建量纲对象 charity_points, milli_charity_points, missions
    charity_points = Quantity("charity_points")
    milli_charity_points = Quantity("milli_charity_points")
    missions = Quantity("missions")

    # 设置 milli_charity_points 相对于 charity_points 的全局比例因子为 1/1000
    milli_charity_points.set_global_relative_scale_factor(S.One/1000, charity_points)
    # 设置 missions 相对于 charity_points 的全局比例因子为 251
    missions.set_global_relative_scale_factor(251, charity_points)

    # 断言：将 kilo_good_grade * milli_charity_points * millimeter 转换为 [centi_good_grade, missions, centimeter] 应为 10**5 / (251*1000) / 10 * centi_good_grade * missions * centimeter
    assert convert_to(
        kilo_good_grade*milli_charity_points*millimeter,
        [centi_good_grade, missions, centimeter]
    ) == S.One * 10**5 / (251*1000) / 10 * centi_good_grade*missions*centimeter


# 测试函数，验证使用表达式替换进行单位计算
def test_eval_subs():
    # 定义符号 energy, mass, force
    energy, mass, force = symbols('energy mass force')
    # 计算表达式 expr1 = energy / mass
    expr1 = energy/mass
    # 设置单位替换字典 units
    units = {energy: kilogram*meter**2/second**2, mass: kilogram}
    # 断言：将 expr1 替换单位为 units 后应为 meter**2/second**2
    assert expr1.subs(units) == meter**2/second**2

    # 计算表达式 expr2 = force / mass
    expr2 = force/mass
    # 设置单位替换字典 units
    units = {force: gravitational_constant*kilogram**2/meter**2, mass: kilogram}
    # 断言：将 expr2 替换单位为 units 后应为 gravitational_constant * kilogram / meter**2
    assert expr2.subs(units) == gravitational_constant*kilogram/meter**2


# 测试函数，验证 issue 14932 的问题
def test_issue_14932():
    # 断言：log(inch) - log(2) 简化后应为 log(inch/2)
    assert (log(inch) - log(2)).simplify() == log(inch/2)
    # 断言：log(inch) - log(foot) 简化后应为 -log(12)
    assert (log(inch) - log(foot)).simplify() == -log(12)
    # 定义正数符号 p
    p = symbols('p', positive=True)
    # 断言：log(inch) - log(p) 简化后应为 log(inch/p)
    assert (log(inch) - log(p)).simplify() == log(inch/p)


# 测试函数，验证 issue 14547 的问题
def test_issue_14547():
    # the root issue is that an argument with dimensions should
    # not raise an error when the `arg - 1` calculation is
    # performed in the assumptions system
    from sympy.physics.units import foot, inch
    from sympy.core.relational import Eq
    # 断言：log(foot).is_zero 应为 None
    assert log(foot).is_zero is None
    # 断言：log(foot).is_positive 应为 None
    assert log(foot).is_positive is None
    # 断言：log(foot).is_nonnegative 应为 None
    assert log(foot).is_nonnegative is None
    # 断言：log(foot).is_negative 应为 None
    assert log(foot).is_negative is None
    # 断言：log(foot).is_algebraic 应为 None
    assert log(foot).is_algebraic is None
    # 断言：log(foot).is_rational 应为 None
    assert log(foot).is_rational is None
    # 断言：log(foot) == log(inch) 不应引发错误，可能为 False 或未评估
    assert Eq(log(foot), log(inch)) is not None  # might be False or unevaluated

    # 定义符号 x
    x = Symbol('x')
    # 计算表达式 e = foot + x
    e = foot + x
    # 断言：e 是 Add 类型且其参数集合应为 {foot, x}
    assert e.is_Add and set(e.args) == {foot, x}
    # 计算表达式 e = foot + 1
    e = foot + 1
    # 断言：e 是 Add 类型且其参数集合应为 {foot, 1}
    assert e.is_Add and set(e.args) == {foot, 1}


# 测试函数，验证 issue 22164 的问题
def test_issue_22164():
    # 设置警告过滤器为 "error"
    warnings.simplefilter("error")
    # 创建一个表示长度单位的 Quantity 对象，命名为 "dm"
    dm = Quantity("dm")
    # 将长度维度关联到 "dm" 单位
    SI.set_quantity_dimension(dm, length)
    # 设置 "dm" 单位的比例因子为 1
    SI.set_quantity_scale_factor(dm, 1)

    # 创建一个表示长度单位的 Quantity 对象，命名为 "bad_exp"
    bad_exp = Quantity("bad_exp")
    # 将长度维度关联到 "bad_exp" 单位
    SI.set_quantity_dimension(bad_exp, length)
    # 设置 "bad_exp" 单位的比例因子为 1
    SI.set_quantity_scale_factor(bad_exp, 1)

    # 构建表达式 dm 的 bad_exp 次方
    expr = dm ** bad_exp

    # 在此处不应该期望到废弃警告
    # 调用 SI 对象的方法 _collect_factor_and_dimension，用于收集表达式的因子和维度
    SI._collect_factor_and_dimension(expr)
def test_issue_22819():
    # 导入必要的单位和量纲系统
    from sympy.physics.units import tonne, gram, Da
    from sympy.physics.units.systems.si import dimsys_SI
    # 断言：吨转换为克的结果应为 1000000 克
    assert tonne.convert_to(gram) == 1000000*gram
    # 断言：面积的量纲依赖应为长度的平方
    assert dimsys_SI.get_dimensional_dependencies(area) == {length: 2}
    # 断言：达尔顿的比例因子应为 1.66053906660000e-24
    assert Da.scale_factor == 1.66053906660000e-24


def test_issue_20288():
    # 导入必要的常数和单位
    from sympy.core.numbers import E
    from sympy.physics.units import energy
    # 创建量纲对象
    u = Quantity('u')
    v = Quantity('v')
    # 设置量纲系统中的能量维度
    SI.set_quantity_dimension(u, energy)
    SI.set_quantity_dimension(v, energy)
    # 设置全局相对比例因子为焦耳
    u.set_global_relative_scale_factor(1, joule)
    v.set_global_relative_scale_factor(1, joule)
    # 表达式：1 + exp(u^2/v^2) 的收集因子和量纲维度
    expr = 1 + exp(u**2/v**2)
    assert SI._collect_factor_and_dimension(expr) == (1 + E, Dimension(1))


def test_issue_24062():
    # 导入必要的常数和单位
    from sympy.core.numbers import E
    from sympy.physics.units import impedance, capacitance, time, ohm, farad, second

    # 创建量纲对象
    R = Quantity('R')
    C = Quantity('C')
    T = Quantity('T')
    # 设置量纲系统中的阻抗、电容和时间维度
    SI.set_quantity_dimension(R, impedance)
    SI.set_quantity_dimension(C, capacitance)
    SI.set_quantity_dimension(T, time)
    # 设置全局相对比例因子为欧姆、法拉和秒
    R.set_global_relative_scale_factor(1, ohm)
    C.set_global_relative_scale_factor(1, farad)
    T.set_global_relative_scale_factor(1, second)
    # 表达式：T / (R * C) 的收集因子和量纲维度
    expr = T / (R * C)
    dim = SI._collect_factor_and_dimension(expr)[1]
    assert SI.get_dimension_system().is_dimensionless(dim)

    # 表达式：1 + exp(expr) 的收集因子和量纲维度
    exp_expr = 1 + exp(expr)
    assert SI._collect_factor_and_dimension(exp_expr) == (1 + E, Dimension(1))


def test_issue_24211():
    # 导入必要的单位和量纲系统
    from sympy.physics.units import time, velocity, acceleration, second, meter
    # 创建速度、加速度和时间量纲对象
    V1 = Quantity('V1')
    SI.set_quantity_dimension(V1, velocity)
    SI.set_quantity_scale_factor(V1, 1 * meter / second)
    A1 = Quantity('A1')
    SI.set_quantity_dimension(A1, acceleration)
    SI.set_quantity_scale_factor(A1, 1 * meter / second**2)
    T1 = Quantity('T1')
    SI.set_quantity_dimension(T1, time)
    SI.set_quantity_scale_factor(T1, 1 * second)

    # 表达式：A1*T1 + V1 的收集因子和量纲维度
    expr = A1*T1 + V1
    # 不应该在这里引发 ValueError
    SI._collect_factor_and_dimension(expr)


def test_prefixed_property():
    # 断言：这些单位不应带有前缀
    assert not meter.is_prefixed
    assert not joule.is_prefixed
    assert not day.is_prefixed
    assert not second.is_prefixed
    assert not volt.is_prefixed
    assert not ohm.is_prefixed
    # 断言：这些单位应带有前缀
    assert centimeter.is_prefixed
    assert kilometer.is_prefixed
    assert kilogram.is_prefixed
    assert pebibyte.is_prefixed


def test_physics_constant():
    # 导入物理常数定义
    from sympy.physics.units import definitions

    # 遍历物理常数定义
    for name in dir(definitions):
        quantity = getattr(definitions, name)
        # 跳过非 Quantity 类型的物理常数
        if not isinstance(quantity, Quantity):
            continue
        # 断言：以 "_constant" 结尾的物理常数应为 PhysicalConstant 类型
        if name.endswith('_constant'):
            assert isinstance(quantity, PhysicalConstant), f"{quantity} must be PhysicalConstant, but is {type(quantity)}"
            assert quantity.is_physical_constant, f"{name} is not marked as physics constant when it should be"
    # 对于给定的物理常数列表，逐个检查其类型是否为 PhysicalConstant 类型
    for const in [gravitational_constant, molar_gas_constant, vacuum_permittivity, speed_of_light, elementary_charge]:
        # 使用断言确保每个常数都是 PhysicalConstant 类型，如果不是则触发断言异常
        assert isinstance(const, PhysicalConstant), f"{const} must be PhysicalConstant, but is {type(const)}"
        # 使用断言确保每个常数都标记为物理常数，如果没有标记则触发断言异常
        assert const.is_physical_constant, f"{const} is not marked as physics constant when it should be"

    # 使用断言确保 meter 不是物理常数，如果不是则触发断言异常
    assert not meter.is_physical_constant
    # 使用断言确保 joule 不是物理常数，如果不是则触发断言异常
    assert not joule.is_physical_constant
```