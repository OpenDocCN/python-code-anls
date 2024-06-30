# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_dimensions.py`

```
# 从 sympy.physics.units.systems.si 模块中导入 dimsys_SI 单位系统对象
from sympy.physics.units.systems.si import dimsys_SI

# 从 sympy.core.numbers 模块中导入 pi 常数
from sympy.core.numbers import pi
# 从 sympy.core.singleton 模块中导入 S 单例对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块中导入 Symbol 符号对象
from sympy.core.symbol import Symbol
# 从 sympy.functions.elementary.complexes 模块中导入 Abs 绝对值函数
from sympy.functions.elementary.complexes import Abs
# 从 sympy.functions.elementary.exponential 模块中导入 log 对数函数
from sympy.functions.elementary.exponential import log
# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt 平方根函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.trigonometric 模块中导入 acos, atan2, cos 三角函数
from sympy.functions.elementary.trigonometric import (acos, atan2, cos)
# 从 sympy.physics.units.dimensions 模块中导入 Dimension 维度对象
from sympy.physics.units.dimensions import Dimension
# 从 sympy.physics.units.definitions.dimension_definitions 模块中导入 length, time, mass, force, pressure, angle 等维度定义
from sympy.physics.units.definitions.dimension_definitions import (
    length, time, mass, force, pressure, angle
)
# 从 sympy.physics.units 模块中导入 foot 单位对象
from sympy.physics.units import foot
# 从 sympy.testing.pytest 模块中导入 raises 函数，用于测试异常
from sympy.testing.pytest import raises


def test_Dimension_definition():
    # 断言获取 length 维度对象的依赖维度为 {length: 1}
    assert dimsys_SI.get_dimensional_dependencies(length) == {length: 1}
    # 断言 length 维度对象的名称为 Symbol("length")
    assert length.name == Symbol("length")
    # 断言 length 维度对象的符号为 Symbol("L")
    assert length.symbol == Symbol("L")

    # 创建 halflength 为 length 的平方根
    halflength = sqrt(length)
    # 断言获取 halflength 维度对象的依赖维度为 {length: S.Half}
    assert dimsys_SI.get_dimensional_dependencies(halflength) == {length: S.Half}


def test_Dimension_error_definition():
    # 测试 Dimension 构造函数参数不是两个元素的元组引发 TypeError 异常
    raises(TypeError, lambda: Dimension(("length", 1, 2)))
    # 测试 Dimension 构造函数参数不是列表引发 TypeError 异常
    raises(TypeError, lambda: Dimension(["length"]))

    # 测试 Dimension 构造函数中维度字典的值不是数字引发 TypeError 异常
    raises(TypeError, lambda: Dimension({"length": "a"}))

    # 测试 Dimension 构造函数中维度字典的值为元组（非数字）引发 TypeError 异常
    raises(TypeError, lambda: Dimension({"length": (1, 2)}))

    # 测试 Dimension 构造函数中的符号参数不是 Symbol 或 str 引发 AssertionError 异常
    raises(AssertionError, lambda: Dimension("length", symbol=1))


def test_str():
    # 断言 Dimension("length") 的字符串表示为 "Dimension(length)"
    assert str(Dimension("length")) == "Dimension(length)"
    # 断言 Dimension("length", "L") 的字符串表示为 "Dimension(length, L)"
    assert str(Dimension("length", "L")) == "Dimension(length, L)"


def test_Dimension_properties():
    # 断言 length 维度对象不是无量纲的
    assert dimsys_SI.is_dimensionless(length) is False
    # 断言 length/length 维度对象是无量纲的
    assert dimsys_SI.is_dimensionless(length/length) is True
    # 断言 Dimension("undefined") 维度对象不是无量纲的
    assert dimsys_SI.is_dimensionless(Dimension("undefined")) is False

    # 断言 length 维度对象在 dimsys_SI 单位系统中具有整数次幂
    assert length.has_integer_powers(dimsys_SI) is True
    # 断言 length**(-1) 维度对象在 dimsys_SI 单位系统中具有整数次幂
    assert (length**(-1)).has_integer_powers(dimsys_SI) is True
    # 断言 length**1.5 维度对象在 dimsys_SI 单位系统中不具有整数次幂
    assert (length**1.5).has_integer_powers(dimsys_SI) is False


def test_Dimension_add_sub():
    # 断言 length + length 等于 length
    assert length + length == length
    # 断言 length - length 等于 length
    assert length - length == length
    # 断言 -length 等于 length
    assert -length == length

    # 断言 length 与 foot 相加引发 TypeError 异常
    raises(TypeError, lambda: length + foot)
    # 断言 foot 与 length 相加引发 TypeError 异常
    raises(TypeError, lambda: foot + length)
    # 断言 length 与 foot 相减引发 TypeError 异常
    raises(TypeError, lambda: length - foot)
    # 断言 foot 与 length 相减引发 TypeError 异常
    raises(TypeError, lambda: foot - length)

    # 测试符号 x 与 length 相加的结果，长度维度与符号的加法
    x = Symbol('x')
    e = length + x
    assert e == x + length and e.is_Add and set(e.args) == {length, x}
    # 测试 length + 1 的结果
    e = length + 1
    assert e == 1 + length == 1 - length and e.is_Add and set(e.args) == {length, 1}

    # 断言获取质量乘以长度除以时间平方再加力的维度依赖为 {length: 1, mass: 1, time: -2}
    assert dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + force) == \
            {length: 1, mass: 1, time: -2}
    # 断言获取质量乘以长度除以时间平方再加力再减压力乘以长度平方的维度依赖为 {length: 1, mass: 1, time: -2}
    assert dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + force -
                                                   pressure * length**2) == \
            {length: 1, mass: 1, time: -2}
    # 调用 raises 函数，预期会抛出 TypeError 异常，使用 lambda 函数测试 dimsys_SI.get_dimensional_dependencies 的行为
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + pressure))
def test_Dimension_mul_div_exp():
    # 断言长度乘法和除法的基本性质
    assert 2*length == length*2 == length/2 == length
    assert 2/length == 1/length
    # 创建符号变量 x
    x = Symbol('x')
    # 测试乘法的性质和符号是否正确
    m = x*length
    assert m == length*x and m.is_Mul and set(m.args) == {x, length}
    # 测试除法的性质和符号是否正确
    d = x/length
    assert d == x*length**-1 and d.is_Mul and set(d.args) == {x, 1/length}
    d = length/x
    assert d == length*x**-1 and d.is_Mul and set(d.args) == {1/x, length}

    # 计算速度的维度
    velo = length / time

    # 断言长度的平方
    assert (length * length) == length ** 2

    # 使用维度系统获取长度平方的维度依赖
    assert dimsys_SI.get_dimensional_dependencies(length * length) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length ** 2) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length * time) == {length: 1, time: 1}
    assert dimsys_SI.get_dimensional_dependencies(velo) == {length: 1, time: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** 2) == {length: 2, time: -2}

    # 断言长度的除法结果
    assert dimsys_SI.get_dimensional_dependencies(length / length) == {}
    assert dimsys_SI.get_dimensional_dependencies(velo / length * time) == {}
    assert dimsys_SI.get_dimensional_dependencies(length ** -1) == {length: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** -1.5) == {length: -1.5, time: 1.5}

    # 测试长度的幂运算
    length_a = length**"a"
    assert dimsys_SI.get_dimensional_dependencies(length_a) == {length: Symbol("a")}

    assert dimsys_SI.get_dimensional_dependencies(length**pi) == {length: pi}
    assert dimsys_SI.get_dimensional_dependencies(length**(length/length)) == {length: Dimension(1)}

    # 测试不支持的长度幂运算，应引发 TypeError
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(length**length))

    # 断言长度不等于 1
    assert length != 1
    assert length / length != 1

    # 测试长度的零次幂
    length_0 = length ** 0
    assert dimsys_SI.get_dimensional_dependencies(length_0) == {}

    # issue 18738
    # 创建符号变量 a 和 b
    a = Symbol('a')
    b = Symbol('b')
    # 计算直角三角形的斜边长度
    c = sqrt(a**2 + b**2)
    # 替换变量并检查维度是否等效
    c_dim = c.subs({a: length, b: length})
    assert dimsys_SI.equivalent_dims(c_dim, length)

def test_Dimension_functions():
    # 测试不支持的函数应引发 TypeError
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(cos(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(acos(angle)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(atan2(length, time)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(100, length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length, 10)))

    # 断言数学常数 pi 没有维度依赖
    assert dimsys_SI.get_dimensional_dependencies(pi) == {}

    # 断言 cos 函数的维度依赖
    assert dimsys_SI.get_dimensional_dependencies(cos(1)) == {}
    assert dimsys_SI.get_dimensional_dependencies(cos(angle)) == {}

    # 断言 atan2 函数的维度依赖
    assert dimsys_SI.get_dimensional_dependencies(atan2(length, length)) == {}

    # 断言 log 函数的维度依赖
    assert dimsys_SI.get_dimensional_dependencies(log(length / length, length / length)) == {}

    # 断言绝对值函数的维度依赖
    assert dimsys_SI.get_dimensional_dependencies(Abs(length)) == {length: 1}
    # 使用 dimsys_SI.get_dimensional_dependencies 函数计算绝对值的长度与长度的绝对值的 SI 单位的维度依赖关系，应返回空字典
    assert dimsys_SI.get_dimensional_dependencies(Abs(length / length)) == {}
    
    # 使用 dimsys_SI.get_dimensional_dependencies 函数计算虚数单位的平方根的 SI 单位的维度依赖关系，应返回空字典
    assert dimsys_SI.get_dimensional_dependencies(sqrt(-1)) == {}
```