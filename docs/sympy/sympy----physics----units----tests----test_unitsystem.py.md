# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_unitsystem.py`

```
# 导入必要的模块和类
from sympy.physics.units import DimensionSystem, joule, second, ampere
# 导入有理数和单例类
from sympy.core.numbers import Rational
from sympy.core.singleton import S
# 导入物理单位的定义
from sympy.physics.units.definitions import c, kg, m, s
# 导入维度定义
from sympy.physics.units.definitions.dimension_definitions import length, time
# 导入量和单位系统相关的类和函数
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.physics.units.util import convert_to


def test_definition():
    # 创建自定义单位"dm"的量
    dm = Quantity("dm")
    # 基本单位是米和秒
    base = (m, s)
    # 创建一个单位系统"MS"，包括米、秒、光速常数c和单位"dm"
    ms = UnitSystem(base, (c, dm), "MS", "MS system")
    # 设置单位"dm"的维度为长度
    ms.set_quantity_dimension(dm, length)
    # 设置单位"dm"的比例因子为1/10
    ms.set_quantity_scale_factor(dm, Rational(1, 10))

    # 断言检查基本单位和所有单位是否正确设置
    assert set(ms._base_units) == set(base)
    assert set(ms._units) == {m, s, c, dm}
    # 断言检查名称和描述是否正确设置
    assert ms.name == "MS"
    assert ms.descr == "MS system"


def test_str_repr():
    # 测试单位系统的字符串表示是否正确
    assert str(UnitSystem((m, s), name="MS")) == "MS"
    assert str(UnitSystem((m, s))) == "UnitSystem((meter, second))"

    # 测试单位系统的详细表示是否正确
    assert repr(UnitSystem((m, s))) == "<UnitSystem: (%s, %s)>" % (m, s)


def test_convert_to():
    # 创建量"A"和"Js"
    A = Quantity("A")
    Js = Quantity("Js")
    # 设置全局相对比例因子
    A.set_global_relative_scale_factor(S.One, ampere)
    Js.set_global_relative_scale_factor(S.One, joule*second)

    # 创建包含米、千克、秒和安培的单位系统mksa
    mksa = UnitSystem((m, kg, s, A), (Js,))
    # 断言检查转换函数的正确性
    assert convert_to(Js, mksa._base_units) == m**2*kg*s**-1/1000


def test_extend():
    # 创建单位系统ms，并扩展其包含千克和"Js"
    ms = UnitSystem((m, s), (c,))
    Js = Quantity("Js")
    Js.set_global_relative_scale_factor(1, joule*second)
    mks = ms.extend((kg,), (Js,))

    # 创建期望的结果单位系统res
    res = UnitSystem((m, s, kg), (c, Js))
    # 断言检查扩展后的单位系统是否正确设置
    assert set(mks._base_units) == set(res._base_units)
    assert set(mks._units) == set(res._units)


def test_dim():
    # 创建包含米、千克、秒和光速常数c的单位系统dimsys
    dimsys = UnitSystem((m, kg, s), (c,))
    # 断言检查单位系统的维度是否正确
    assert dimsys.dim == 3


def test_is_consistent():
    # 创建维度系统dimension_system，包含长度和时间维度
    dimension_system = DimensionSystem([length, time])
    # 创建单位系统us，包含米和秒，并关联到维度系统dimension_system
    us = UnitSystem([m, s], dimension_system=dimension_system)
    # 断言检查单位系统的一致性
    assert us.is_consistent == True


def test_get_units_non_prefixed():
    # 导入电压和电阻单位
    from sympy.physics.units import volt, ohm
    # 获取"SI"单位系统
    unit_system = UnitSystem.get_unit_system("SI")
    # 获取非有前缀的单位列表
    units = unit_system.get_units_non_prefixed()
    # 逐个检查非有前缀单位的属性和命名规则
    for prefix in ["giga", "tera", "peta", "exa", "zetta", "yotta", "kilo", "hecto", "deca", "deci", "centi", "milli", "micro", "nano", "pico", "femto", "atto", "zepto", "yocto"]:
        for unit in units:
            assert isinstance(unit, Quantity), f"{unit} must be a Quantity, not {type(unit)}"
            assert not unit.is_prefixed, f"{unit} is marked as prefixed"
            assert not unit.is_physical_constant, f"{unit} is marked as physics constant"
            assert not unit.name.name.startswith(prefix), f"Unit {unit.name} has prefix {prefix}"
    # 断言检查特定单位是否在单位列表中
    assert volt in units
    assert ohm in units
    # 遍历 UnitSystem 类中所有注册的单位系统
    for unit_system in UnitSystem._unit_systems.values():
        # 遍历每个单位系统中的衍生单位
        for preferred_unit in unit_system.derived_units.values():
            # 获取每个衍生单位中的基本单位，并作为 Quantity 对象存储在 units 中
            units = preferred_unit.atoms(Quantity)
            # 遍历每个基本单位
            for unit in units:
                # 检查单位是否存在于单位系统的已注册单位中，如果不存在则触发断言错误
                assert unit in unit_system._units, f"Unit {unit} is not in unit system {unit_system}"
```