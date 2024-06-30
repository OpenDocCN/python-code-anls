# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_prefixes.py`

```
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.physics.units import Quantity, length, meter, W
from sympy.physics.units.prefixes import PREFIXES, Prefix, prefix_unit, kilo, \
    kibi
from sympy.physics.units.systems import SI

# 定义一个符号变量 x
x = Symbol('x')

# 定义一个测试函数，测试单位前缀的操作
def test_prefix_operations():
    # 获取不同单位前缀
    m = PREFIXES['m']
    k = PREFIXES['k']
    M = PREFIXES['M']

    # 定义一个自定义前缀对象
    dodeca = Prefix('dodeca', 'dd', 1, base=12)

    # 断言不同单位前缀的乘法运算
    assert m * k is S.One
    # 断言单位前缀与物理量 W 的乘法运算
    assert m * W == W / 1000
    # 断言单位前缀的乘法运算
    assert k * k == M
    # 断言单位前缀的除法运算
    assert 1 / m == k
    # 断言单位前缀之间的除法运算
    assert k / m == M

    # 断言自定义前缀对象的乘法运算
    assert dodeca * dodeca == 144
    # 断言自定义前缀对象的除法运算
    assert 1 / dodeca == S.One / 12
    # 断言单位前缀与自定义前缀对象的除法运算
    assert k / dodeca == S(1000) / 12
    # 断言自定义前缀对象与自身的除法运算
    assert dodeca / dodeca is S.One

    # 创建一个假的长度单位 Quantity 对象
    m = Quantity("fake_meter")
    # 设置其维度
    SI.set_quantity_dimension(m, S.One)
    # 设置其比例因子
    SI.set_quantity_scale_factor(m, S.One)

    # 断言自定义前缀对象与假长度单位对象的乘法运算
    assert dodeca * m == 12 * m
    # 断言自定义前缀对象与假长度单位对象的除法运算
    assert dodeca / m == 12 / m

    # 创建一个 kilo 与数字 3 的乘法表达式
    expr1 = kilo * 3
    assert isinstance(expr1, Mul)
    assert expr1.args == (3, kilo)

    # 创建一个 kilo 与符号 x 的乘法表达式
    expr2 = kilo * x
    assert isinstance(expr2, Mul)
    assert expr2.args == (x, kilo)

    # 创建一个 kilo 与数字 3 的除法表达式
    expr3 = kilo / 3
    assert isinstance(expr3, Mul)
    assert expr3.args == (Rational(1, 3), kilo)
    assert expr3.args == (S.One/3, kilo)

    # 创建一个 kilo 与符号 x 的除法表达式
    expr4 = kilo / x
    assert isinstance(expr4, Mul)
    assert expr4.args == (1/x, kilo)


# 定义一个测试函数，测试单位前缀的单位转换功能
def test_prefix_unit():
    # 创建一个假的长度单位 Quantity 对象
    m = Quantity("fake_meter", abbrev="m")
    # 设置其全局相对比例因子
    m.set_global_relative_scale_factor(1, meter)

    # 定义单位前缀字典
    pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}

    # 创建三个假的长度单位 Quantity 对象
    q1 = Quantity("millifake_meter", abbrev="mm")
    q2 = Quantity("centifake_meter", abbrev="cm")
    q3 = Quantity("decifake_meter", abbrev="dm")

    # 设置第一个假长度单位对象的维度
    SI.set_quantity_dimension(q1, length)

    # 设置三个假长度单位对象的比例因子
    SI.set_quantity_scale_factor(q1, PREFIXES["m"])
    SI.set_quantity_scale_factor(q1, PREFIXES["c"])
    SI.set_quantity_scale_factor(q1, PREFIXES["d"])

    # 构建结果列表
    res = [q1, q2, q3]

    # 调用函数进行单位前缀转换
    prefs = prefix_unit(m, pref)
    # 断言转换结果与预期结果一致
    assert set(prefs) == set(res)
    # 断言转换后的单位前缀对象的缩写符号符合预期
    assert {v.abbrev for v in prefs} == set(symbols("mm,cm,dm"))


# 定义一个测试函数，测试单位前缀的基数
def test_bases():
    # 断言 kilo 的基数为 10
    assert kilo.base == 10
    # 断言 kibi 的基数为 2
    assert kibi.base == 2


# 定义一个测试函数，测试单位前缀对象的表达式表示
def test_repr():
    # 断言通过表达式求值后得到的对象与原对象一致
    assert eval(repr(kilo)) == kilo
    assert eval(repr(kibi)) == kibi
```