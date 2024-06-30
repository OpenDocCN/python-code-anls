# `D:\src\scipysrc\sympy\sympy\physics\units\systems\mks.py`

```
"""
MKS unit system.

MKS stands for "meter, kilogram, second".
"""

# 导入单位系统相关模块
from sympy.physics.units import UnitSystem
from sympy.physics.units.definitions import gravitational_constant, hertz, joule, newton, pascal, watt, speed_of_light, gram, kilogram, meter, second
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, energy, force, frequency, momentum,
    power, pressure, velocity, length, mass, time)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.length_weight_time import dimsys_length_weight_time

# 定义基本维度
dims = (velocity, acceleration, momentum, force, energy, power, pressure,
        frequency, action)

# 定义基本单位列表
units = [meter, gram, second, joule, newton, watt, pascal, hertz]
all_units = []

# 将单位的前缀（如千克、焦耳、牛顿等）添加到单位列表中
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend([gravitational_constant, speed_of_light])

# 创建 MKS 单位系统对象
MKS = UnitSystem(base_units=(meter, kilogram, second),  # 基本单位为米、千克、秒
                units=all_units,                     # 包含所有定义的单位
                name="MKS",                          # 单位系统名称为 MKS
                dimension_system=dimsys_length_weight_time,  # 使用长度-重量-时间维度系统
                derived_units={                      # 定义派生单位
                    power: watt,
                    time: second,
                    pressure: pascal,
                    length: meter,
                    frequency: hertz,
                    mass: kilogram,
                    force: newton,
                    energy: joule,
                    velocity: meter/second,
                    acceleration: meter/(second**2),
                })

# 暴露的对象列表
__all__ = [
    'MKS', 'units', 'all_units', 'dims',
]
```