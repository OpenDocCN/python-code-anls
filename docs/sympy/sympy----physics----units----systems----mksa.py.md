# `D:\src\scipysrc\sympy\sympy\physics\units\systems\mksa.py`

```
"""
MKS unit system.

MKS stands for "meter, kilogram, second, ampere".
"""

# 从 __future__ 模块导入 annotations 功能，支持类型提示中的类型变量
from __future__ import annotations

# 导入物理单位定义模块中所需的物理量和单位
from sympy.physics.units.definitions import Z0, ampere, coulomb, farad, henry, siemens, tesla, volt, weber, ohm
# 导入物理单位定义模块中的维度定义
from sympy.physics.units.definitions.dimension_definitions import (
    capacitance, charge, conductance, current, impedance, inductance,
    magnetic_density, magnetic_flux, voltage)
# 导入物理单位前缀和单位的关联
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
# 导入 M, K, S 单位系统
from sympy.physics.units.systems.mks import MKS, dimsys_length_weight_time
# 导入物理量对象
from sympy.physics.units.quantities import Quantity

# 定义需要考虑的物理量维度列表
dims = (voltage, impedance, conductance, current, capacitance, inductance, charge,
        magnetic_density, magnetic_flux)

# 定义单位列表
units = [ampere, volt, ohm, siemens, farad, henry, coulomb, tesla, weber]

# 定义所有单位的列表
all_units: list[Quantity] = []

# 使用单位对象和前缀列表扩展所有单位列表
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend(units)

# 添加特殊的阻抗 Z0 到所有单位列表中
all_units.append(Z0)

# 扩展 MKS 系统的维度系统，增加电流作为新的维度
dimsys_MKSA = dimsys_length_weight_time.extend([
    current,
], new_dim_deps={
    # 衍生维度的维度依赖定义
    "voltage": {"mass": 1, "length": 2, "current": -1, "time": -3},
    "impedance": {"mass": 1, "length": 2, "current": -2, "time": -3},
    "conductance": {"mass": -1, "length": -2, "current": 2, "time": 3},
    "capacitance": {"mass": -1, "length": -2, "current": 2, "time": 4},
    "inductance": {"mass": 1, "length": 2, "current": -2, "time": -2},
    "charge": {"current": 1, "time": 1},
    "magnetic_density": {"mass": 1, "current": -1, "time": -2},
    "magnetic_flux": {"length": 2, "mass": 1, "current": -1, "time": -2},
})

# 创建 MKSA 单位系统，扩展自 MKS 系统，并定义导出单位
MKSA = MKS.extend(base=(ampere,), units=all_units, name='MKSA', dimension_system=dimsys_MKSA, derived_units={
    magnetic_flux: weber,
    impedance: ohm,
    current: ampere,
    voltage: volt,
    inductance: henry,
    conductance: siemens,
    magnetic_density: tesla,
    charge: coulomb,
    capacitance: farad,
})
```