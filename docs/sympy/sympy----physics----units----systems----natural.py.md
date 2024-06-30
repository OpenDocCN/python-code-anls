# `D:\src\scipysrc\sympy\sympy\physics\units\systems\natural.py`

```
"""
Natural unit system.

The natural system comes from "setting c = 1, hbar = 1". From the computer
point of view it means that we use velocity and action instead of length and
time. Moreover, instead of mass we use energy.
"""

# 导入所需模块和类
from sympy.physics.units import DimensionSystem  # 导入维度系统相关模块
from sympy.physics.units.definitions import c, eV, hbar  # 导入物理常数 c, eV, hbar
from sympy.physics.units.definitions.dimension_definitions import (
    action, energy, force, frequency, length, mass, momentum,
    power, time, velocity)  # 导入各物理量的维度定义
from sympy.physics.units.prefixes import PREFIXES, prefix_unit  # 导入单位前缀相关模块
from sympy.physics.units.unitsystem import UnitSystem  # 导入单位系统类


# 维度系统
_natural_dim = DimensionSystem(
    base_dims=(action, energy, velocity),  # 基本维度包括行动、能量、速度
    derived_dims=(length, mass, time, momentum, force, power, frequency)  # 派生维度包括长度、质量、时间等
)

# 创建带有 eV 单位前缀的单位
units = prefix_unit(eV, PREFIXES)

# 单位系统
natural = UnitSystem(base_units=(hbar, eV, c), units=units, name="Natural system")
# 使用 hbar, eV, c 作为基本单位，创建名为 "Natural system" 的自然单位系统
```