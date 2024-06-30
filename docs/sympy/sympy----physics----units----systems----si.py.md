# `D:\src\scipysrc\sympy\sympy\physics\units\systems\si.py`

```
"""
SI unit system.
Based on MKSA, which stands for "meter, kilogram, second, ampere".
Added kelvin, candela and mole.
"""

# 导入必要的模块和类
from __future__ import annotations  # 允许使用类型注解

from sympy.physics.units import DimensionSystem, Dimension, dHg0  # 导入单位相关的类和函数

from sympy.physics.units.quantities import Quantity  # 导入量纲相关的类

from sympy.core.numbers import (Rational, pi)  # 导入有理数和圆周率等数学常数
from sympy.core.singleton import S  # 导入SymPy的单例对象S
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, current, impedance, length, mass, time, velocity,
    amount_of_substance, temperature, information, frequency, force, pressure,
    energy, power, charge, voltage, capacitance, conductance, magnetic_flux,
    magnetic_density, inductance, luminous_intensity
)  # 导入各种量纲的定义

from sympy.physics.units.definitions import (
    kilogram, newton, second, meter, gram, cd, K, joule, watt, pascal, hertz,
    coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre, lux,
    katal, gray, becquerel, inch, liter, julian_year, gravitational_constant,
    speed_of_light, elementary_charge, planck, hbar, electronvolt,
    avogadro_number, avogadro_constant, boltzmann_constant, electron_rest_mass,
    stefan_boltzmann_constant, Da, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge, planck_area,
    planck_volume, planck_momentum, planck_energy, planck_force, planck_power,
    planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, c, kelvin,
    mol, mole, candela, m, kg, s, electric_constant, G, boltzmann
)  # 导入各种物理常数、单位和量纲

from sympy.physics.units.prefixes import PREFIXES, prefix_unit  # 导入单位前缀相关的类和函数
from sympy.physics.units.systems.mksa import MKSA, dimsys_MKSA  # 导入MKSA单位系统

# 定义派生量纲和基本量纲
derived_dims = (frequency, force, pressure, energy, power, charge, voltage,
                capacitance, conductance, magnetic_flux,
                magnetic_density, inductance, luminous_intensity)
base_dims = (amount_of_substance, luminous_intensity, temperature)

# 定义单位列表
units = [mol, cd, K, lux, hertz, newton, pascal, joule, watt, coulomb, volt,
        farad, ohm, siemens, weber, tesla, henry, candela, lux, becquerel,
        gray, katal]

# 初始化所有单位列表
all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))  # 添加各种单位的带有前缀的变体

all_units.extend(units)  # 添加未带前缀的单位
all_units.extend([mol, cd, K, lux])  # 添加额外的特定单位

# 扩展MKSA单位系统到SI单位系统，并命名为dimsys_SI
dimsys_SI = dimsys_MKSA.extend(
    [
        # 这里列出了其他基本维度的依赖关系：
        temperature,
        amount_of_substance,
        luminous_intensity,
    ])
# 将 information 添加到 dimsys_SI 中，并将结果赋给 dimsys_default
dimsys_default = dimsys_SI.extend(
    [information],
)

# 创建 SI 系统，继承 MKSA 系统，并指定一些基本单位和所有单位，命名为 'SI'，使用 dimsys_SI 作为维度系统，
# 同时定义派生单位的映射关系
SI = MKSA.extend(base=(mol, cd, K), units=all_units, name='SI', dimension_system=dimsys_SI, derived_units={
    power: watt,
    magnetic_flux: weber,
    time: second,
    impedance: ohm,
    pressure: pascal,
    current: ampere,
    voltage: volt,
    length: meter,
    frequency: hertz,
    inductance: henry,
    temperature: kelvin,
    amount_of_substance: mole,
    luminous_intensity: candela,
    conductance: siemens,
    mass: kilogram,
    magnetic_density: tesla,
    charge: coulomb,
    force: newton,
    capacitance: farad,
    energy: joule,
    velocity: meter/second,
})

# 设置量纲为弧度的数量的比例因子为 1
One = S.One

# 设置量纲为 ampere 的数量的比例因子为 1
SI.set_quantity_scale_factor(ampere, One)

# 设置量纲为 kelvin 的数量的比例因子为 1
SI.set_quantity_scale_factor(kelvin, One)

# 设置量纲为 mole 的数量的比例因子为 1
SI.set_quantity_scale_factor(mole, One)

# 设置量纲为 candela 的数量的比例因子为 1
SI.set_quantity_scale_factor(candela, One)

# 设置量纲为 coulomb 的数量的比例因子为 1
SI.set_quantity_scale_factor(coulomb, One)

# 设置量纲为 volt 的数量的比例因子为 joule/coulomb
SI.set_quantity_scale_factor(volt, joule/coulomb)

# 设置量纲为 ohm 的数量的比例因子为 volt/ampere
SI.set_quantity_scale_factor(ohm, volt/ampere)

# 设置量纲为 siemens 的数量的比例因子为 ampere/volt
SI.set_quantity_scale_factor(siemens, ampere/volt)

# 设置量纲为 farad 的数量的比例因子为 coulomb/volt
SI.set_quantity_scale_factor(farad, coulomb/volt)

# 设置量纲为 henry 的数量的比例因子为 volt*second/ampere
SI.set_quantity_scale_factor(henry, volt*second/ampere)

# 设置量纲为 tesla 的数量的比例因子为 volt*second/meter^2
SI.set_quantity_scale_factor(tesla, volt*second/meter**2)

# 设置量纲为 weber 的数量的比例因子为 joule/ampere
SI.set_quantity_scale_factor(weber, joule/ampere)

# 设置量纲为 lux 的数量的量纲为 luminous_intensity / length^2，比例因子为 steradian*candela/meter^2
SI.set_quantity_dimension(lux, luminous_intensity / length ** 2)
SI.set_quantity_scale_factor(lux, steradian*candela/meter**2)

# 设置量纲为 katal 的数量的量纲为 amount_of_substance / time，比例因子为 mol/second
SI.set_quantity_dimension(katal, amount_of_substance / time)
SI.set_quantity_scale_factor(katal, mol/second)

# 设置量纲为 gray 的数量的量纲为 energy / mass，比例因子为 meter^2/second^2
SI.set_quantity_dimension(gray, energy / mass)
SI.set_quantity_scale_factor(gray, meter**2/second**2)

# 设置量纲为 becquerel 的数量的量纲为 1 / time，比例因子为 1/second
SI.set_quantity_dimension(becquerel, 1 / time)
SI.set_quantity_scale_factor(becquerel, 1/second)

#### CONSTANTS ####

# 设置量纲为 elementary_charge 的数量的量纲为 charge，比例因子为 1.602176634e-19*coulomb
SI.set_quantity_dimension(elementary_charge, charge)
SI.set_quantity_scale_factor(elementary_charge, 1.602176634e-19*coulomb)

# 设置量纲为 electronvolt 的数量的量纲为 energy，比例因子为 1.602176634e-19*joule
SI.set_quantity_dimension(electronvolt, energy)
SI.set_quantity_scale_factor(electronvolt, 1.602176634e-19*joule)

# 设置量纲为 avogadro_number 的数量的量纲为 One，比例因子为 6.02214076e23
SI.set_quantity_dimension(avogadro_number, One)
SI.set_quantity_scale_factor(avogadro_number, 6.02214076e23)

# 设置量纲为 avogadro_constant 的数量的量纲为 amount_of_substance ** -1，比例因子为 avogadro_number / mol
SI.set_quantity_dimension(avogadro_constant, amount_of_substance ** -1)
SI.set_quantity_scale_factor(avogadro_constant, avogadro_number / mol)

# 设置量纲为 boltzmann_constant 的数量的量纲为 energy / temperature，比例因子为 1.380649e-23*joule/kelvin
SI.set_quantity_dimension(boltzmann_constant, energy / temperature)
SI.set_quantity_scale_factor(boltzmann_constant, 1.380649e-23*joule/kelvin)

# 设置量纲为 stefan_boltzmann_constant 的数量的量纲为 energy * time ** -1 * length ** -2 * temperature ** -4
# (斯特藩-玻尔兹曼常数)
# 设置斯蒂芬-玻尔兹曼常数的量纲和比例因子
SI.set_quantity_scale_factor(stefan_boltzmann_constant, pi**2 * boltzmann_constant**4 / (60 * hbar**3 * speed_of_light ** 2))

# 原子质量常数
# 参考：NIST SP 959 (2019年6月)
SI.set_quantity_dimension(atomic_mass_constant, mass)
SI.set_quantity_scale_factor(atomic_mass_constant, 1.66053906660e-24*gram)

# 摩尔气体常数
# 参考：NIST SP 959 (2019年6月)
SI.set_quantity_dimension(molar_gas_constant, energy / (temperature * amount_of_substance))
SI.set_quantity_scale_factor(molar_gas_constant, boltzmann_constant * avogadro_constant)

# 法拉第常数
SI.set_quantity_dimension(faraday_constant, charge / amount_of_substance)
SI.set_quantity_scale_factor(faraday_constant, elementary_charge * avogadro_constant)

# 约瑟夫逊常数
SI.set_quantity_dimension(josephson_constant, frequency / voltage)
SI.set_quantity_scale_factor(josephson_constant, 0.5 * planck / elementary_charge)

# 冯·克利青常数
SI.set_quantity_dimension(von_klitzing_constant, voltage / current)
SI.set_quantity_scale_factor(von_klitzing_constant, hbar / elementary_charge ** 2)

# 地球表面的重力加速度
SI.set_quantity_dimension(acceleration_due_to_gravity, acceleration)
SI.set_quantity_scale_factor(acceleration_due_to_gravity, 9.80665*meter/second**2)

# 磁常数
SI.set_quantity_dimension(magnetic_constant, force / current ** 2)
SI.set_quantity_scale_factor(magnetic_constant, 4*pi/10**7 * newton/ampere**2)

# 电常数（真空介电常数）
SI.set_quantity_dimension(vacuum_permittivity, capacitance / length)
SI.set_quantity_scale_factor(vacuum_permittivity, 1/(u0 * c**2))

# 真空阻抗
SI.set_quantity_dimension(vacuum_impedance, impedance)
SI.set_quantity_scale_factor(vacuum_impedance, u0 * c)

# 电子静止质量
SI.set_quantity_dimension(electron_rest_mass, mass)
SI.set_quantity_scale_factor(electron_rest_mass, 9.1093837015e-31*kilogram)

# 库仑常数
SI.set_quantity_dimension(coulomb_constant, force * length ** 2 / charge ** 2)
SI.set_quantity_scale_factor(coulomb_constant, 1/(4*pi*vacuum_permittivity))

# 磅力每平方英寸（psi）
SI.set_quantity_dimension(psi, pressure)
SI.set_quantity_scale_factor(psi, pound * gee / inch ** 2)

# 毫米汞柱（mmHg）
SI.set_quantity_dimension(mmHg, pressure)
SI.set_quantity_scale_factor(mmHg, dHg0 * acceleration_due_to_gravity * kilogram / meter**2)

# 毫质量单位
SI.set_quantity_dimension(milli_mass_unit, mass)
SI.set_quantity_scale_factor(milli_mass_unit, atomic_mass_unit/1000)

# 夸脱（quart）
SI.set_quantity_dimension(quart, length ** 3)
SI.set_quantity_scale_factor(quart, Rational(231, 4) * inch**3)

# 光年（lightyear）
SI.set_quantity_dimension(lightyear, length)
SI.set_quantity_scale_factor(lightyear, speed_of_light*julian_year)

# 天文单位（astronomical unit）
SI.set_quantity_dimension(astronomical_unit, length)
SI.set_quantity_scale_factor(astronomical_unit, 149597870691*meter)

# 基本普朗克单位：

# 普朗克质量
SI.set_quantity_dimension(planck_mass, mass)
SI.set_quantity_scale_factor(planck_mass, sqrt(hbar*speed_of_light/G))

# 普朗克时间
SI.set_quantity_dimension(planck_time, time)
# 设置 Planck 时间单位的数量尺度因子，基于普朗克常数的平方根与引力常数、光速的五次方根
SI.set_quantity_scale_factor(planck_time, sqrt(hbar*G/speed_of_light**5))

# 设置 Planck 温度单位的数量尺度因子，基于普朗克常数的平方根与光速的五次方、玻尔兹曼常数的平方根、引力常数的倒数
SI.set_quantity_dimension(planck_temperature, temperature)
SI.set_quantity_scale_factor(planck_temperature, sqrt(hbar*speed_of_light**5/G/boltzmann**2))

# 设置 Planck 长度单位的数量尺度因子，基于普朗克常数的平方根与引力常数、光速的立方根
SI.set_quantity_dimension(planck_length, length)
SI.set_quantity_scale_factor(planck_length, sqrt(hbar*G/speed_of_light**3))

# 设置 Planck 电荷单位的数量尺度因子，基于电场常数、普朗克常数、光速的平方根、4π 的平方根
SI.set_quantity_dimension(planck_charge, charge)
SI.set_quantity_scale_factor(planck_charge, sqrt(4*pi*electric_constant*hbar*speed_of_light))

# 派生的 Planck 单位:

# 设置 Planck 面积单位的数量尺度因子，基于 Planck 长度的平方
SI.set_quantity_dimension(planck_area, length ** 2)
SI.set_quantity_scale_factor(planck_area, planck_length**2)

# 设置 Planck 体积单位的数量尺度因子，基于 Planck 长度的立方
SI.set_quantity_dimension(planck_volume, length ** 3)
SI.set_quantity_scale_factor(planck_volume, planck_length**3)

# 设置 Planck 动量单位的数量尺度因子，基于 Planck 质量与光速
SI.set_quantity_dimension(planck_momentum, mass * velocity)
SI.set_quantity_scale_factor(planck_momentum, planck_mass * speed_of_light)

# 设置 Planck 能量单位的数量尺度因子，基于 Planck 质量与光速的平方
SI.set_quantity_dimension(planck_energy, energy)
SI.set_quantity_scale_factor(planck_energy, planck_mass * speed_of_light**2)

# 设置 Planck 力单位的数量尺度因子，基于 Planck 能量与 Planck 长度的倒数
SI.set_quantity_dimension(planck_force, force)
SI.set_quantity_scale_factor(planck_force, planck_energy / planck_length)

# 设置 Planck 功率单位的数量尺度因子，基于 Planck 能量与 Planck 时间的倒数
SI.set_quantity_dimension(planck_power, power)
SI.set_quantity_scale_factor(planck_power, planck_energy / planck_time)

# 设置 Planck 密度单位的数量尺度因子，基于 Planck 质量与 Planck 长度的立方的倒数
SI.set_quantity_dimension(planck_density, mass / length ** 3)
SI.set_quantity_scale_factor(planck_density, planck_mass / planck_length**3)

# 设置 Planck 能量密度单位的数量尺度因子，基于 Planck 能量与 Planck 长度的立方的倒数
SI.set_quantity_dimension(planck_energy_density, energy / length ** 3)
SI.set_quantity_scale_factor(planck_energy_density, planck_energy / planck_length**3)

# 设置 Planck 强度单位的数量尺度因子，基于 Planck 能量密度与光速
SI.set_quantity_dimension(planck_intensity, mass * time ** (-3))
SI.set_quantity_scale_factor(planck_intensity, planck_energy_density * speed_of_light)

# 设置 Planck 角频率单位的数量尺度因子，基于 Planck 时间的倒数
SI.set_quantity_dimension(planck_angular_frequency, 1 / time)
SI.set_quantity_scale_factor(planck_angular_frequency, 1 / planck_time)

# 设置 Planck 压强单位的数量尺度因子，基于 Planck 力与 Planck 长度的平方的倒数
SI.set_quantity_dimension(planck_pressure, pressure)
SI.set_quantity_scale_factor(planck_pressure, planck_force / planck_length**2)

# 设置 Planck 电流单位的数量尺度因子，基于 Planck 电荷与 Planck 时间的倒数
SI.set_quantity_dimension(planck_current, current)
SI.set_quantity_scale_factor(planck_current, planck_charge / planck_time)

# 设置 Planck 电压单位的数量尺度因子，基于 Planck 能量与 Planck 电荷的倒数
SI.set_quantity_dimension(planck_voltage, voltage)
SI.set_quantity_scale_factor(planck_voltage, planck_energy / planck_charge)

# 设置 Planck 阻抗单位的数量尺度因子，基于 Planck 电压与 Planck 电流的倒数
SI.set_quantity_dimension(planck_impedance, impedance)
SI.set_quantity_scale_factor(planck_impedance, planck_voltage / planck_current)

# 设置 Planck 加速度单位的数量尺度因子，基于光速与 Planck 时间的倒数
SI.set_quantity_dimension(planck_acceleration, acceleration)
SI.set_quantity_scale_factor(planck_acceleration, speed_of_light / planck_time)

# 旧的放射性单位:

# 设置居里单位的数量尺度因子，基于贝克勒尔的数量
SI.set_quantity_dimension(curie, 1 / time)
SI.set_quantity_scale_factor(curie, 37000000000*becquerel)

# 设置卢瑟福单位的数量尺度因子，基于贝克勒尔的数量
SI.set_quantity_dimension(rutherford, 1 / time)
SI.set_quantity_scale_factor(rutherford, 1000000*becquerel)

# 检查数量尺度因子是否符合正确的国际单位制维度：
for _scale_factor, _dimension in zip(
    SI._quantity_scale_factors.values(),
    SI._quantity_dimension_map.values()
):
    dimex = SI.get_dimensional_expr(_scale_factor)
    # 检查 dimex 是否不等于 1
    if dimex != 1:
        # 如果 dimension system 的 equivalent_dims 方法无法接受 self 以外的两个参数，
        # 下面的代码将无法正常工作：
        if not DimensionSystem.equivalent_dims(_dimension, Dimension(dimex)):  # type: ignore
            # 抛出数值与维度不匹配的异常
            raise ValueError("quantity value and dimension mismatch")
# 删除 _scale_factor 和 _dimension 变量，这可能是为了清理或重置相关的全局状态

__all__ = [
    # 定义了一个列表 __all__，包含了可以从当前模块导出的公共符号

    'mmHg', 'atmosphere', 'inductance', 'newton', 'meter',
    'vacuum_permittivity', 'pascal', 'magnetic_constant', 'voltage',
    'angular_mil', 'luminous_intensity', 'all_units',
    'julian_year', 'weber', 'exbibyte', 'liter',
    'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'lightyear', 'planck_density', 'gee', 'mol', 'bit', 'gray',
    'planck_momentum', 'bar', 'magnetic_density', 'prefix_unit', 'PREFIXES',
    'planck_time', 'dimex', 'gram', 'candela', 'force', 'planck_intensity',
    'energy', 'becquerel', 'planck_acceleration', 'speed_of_light',
    'conductance', 'frequency', 'coulomb_constant', 'degree', 'lux', 'planck',
    'current', 'planck_current', 'tebibyte', 'planck_power', 'MKSA', 'power',
    'K', 'planck_volume', 'quart', 'pressure', 'amount_of_substance',
    'joule', 'boltzmann_constant', 'Dimension', 'c', 'planck_force', 'length',
    'watt', 'action', 'hbar', 'gibibyte', 'DimensionSystem', 'cd', 'volt',
    'planck_charge', 'dioptre', 'vacuum_impedance', 'dimsys_default', 'farad',
    'charge', 'gravitational_constant', 'temperature', 'u0', 'hertz',
    'capacitance', 'tesla', 'steradian', 'planck_mass', 'josephson_constant',
    'planck_area', 'stefan_boltzmann_constant', 'base_dims',
    'astronomical_unit', 'radian', 'planck_voltage', 'impedance',
    'planck_energy', 'Da', 'atomic_mass_constant', 'rutherford', 'second', 'inch',
    'elementary_charge', 'SI', 'electronvolt', 'dimsys_SI', 'henry',
    'planck_angular_frequency', 'ohm', 'pound', 'planck_pressure', 'G', 'psi',
    'dHg0', 'von_klitzing_constant', 'planck_length', 'avogadro_number',
    'mole', 'acceleration', 'information', 'planck_energy_density',
    'mebibyte', 's', 'acceleration_due_to_gravity', 'electron_rest_mass',
    'planck_temperature', 'units', 'mass', 'dimsys_MKSA', 'kelvin', 'kPa',
    'boltzmann', 'milli_mass_unit', 'planck_impedance', 'electric_constant',
    'derived_dims', 'kg', 'coulomb', 'siemens', 'byte', 'magnetic_flux',
    'atomic_mass_unit', 'm', 'kibibyte', 'kilogram', 'One', 'curie', 'u',
    'time', 'pebibyte', 'velocity', 'ampere', 'katal',
]
# 列出了大量物理和计量单位，这些都是可以从当前模块导出的符号
```