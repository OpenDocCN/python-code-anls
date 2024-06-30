# `D:\src\scipysrc\sympy\sympy\physics\units\systems\length_weight_time.py`

```
# 导入 sympy 库中的一些特定模块和函数
from sympy.core.singleton import S

# 导入 sympy 库中的数学常数 pi
from sympy.core.numbers import pi

# 导入 sympy 物理单位相关模块和特定单位
from sympy.physics.units import DimensionSystem, hertz, kilogram
from sympy.physics.units.definitions import (
    G, Hz, J, N, Pa, W, c, g, kg, m, s, meter, gram, second, newton,
    joule, watt, pascal)
    
# 导入 sympy 物理单位定义中的维度定义
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, energy, force, frequency, momentum,
    power, pressure, velocity, length, mass, time)

# 导入 sympy 物理单位前缀相关模块和函数
from sympy.physics.units.prefixes import PREFIXES, prefix_unit

# 导入 sympy 物理单位前缀定义
from sympy.physics.units.prefixes import (
    kibi, mebi, gibi, tebi, pebi, exbi
)

# 导入 sympy 物理单位具体的物理常数和单位
from sympy.physics.units.definitions import (
    cd, K, coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre,
    lux, katal, gray, becquerel, inch, hectare, liter, julian_year,
    gravitational_constant, speed_of_light, elementary_charge, planck, hbar,
    electronvolt, avogadro_number, avogadro_constant, boltzmann_constant,
    stefan_boltzmann_constant, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge,
    planck_area, planck_volume, planck_momentum, planck_energy, planck_force,
    planck_power, planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, kelvin,
    mol, mole, candela, electric_constant, boltzmann, angstrom
)

# 创建一个维度系统，包含长度、质量和时间
dimsys_length_weight_time = DimensionSystem([
    length,
    mass,
    time,
], dimensional_dependencies={
    # 派生维度的依赖关系
    "velocity": {"length": 1, "time": -1},
    "acceleration": {"length": 1, "time": -2},
    "momentum": {"mass": 1, "length": 1, "time": -1},
    "force": {"mass": 1, "length": 1, "time": -2},
    "energy": {"mass": 1, "length": 2, "time": -2},
    "power": {"length": 2, "mass": 1, "time": -3},
    "pressure": {"mass": 1, "length": -1, "time": -2},
    "frequency": {"time": -1},
    "action": {"length": 2, "mass": 1, "time": -1},
    "area": {"length": 2},
    "volume": {"length": 3},
})

# 设置长度单位 meter 的维度和比例因子
One = S.One  # 设置比例因子为 1
dimsys_length_weight_time.set_quantity_dimension(meter, length)
dimsys_length_weight_time.set_quantity_scale_factor(meter, One)

# 设置质量单位 gram 的维度和比例因子
dimsys_length_weight_time.set_quantity_dimension(gram, mass)
dimsys_length_weight_time.set_quantity_scale_factor(gram, One)

# 设置时间单位 second 的维度
dimsys_length_weight_time.set_quantity_dimension(second, time)
# 设置时间单位的数量标度因子为秒
dimsys_length_weight_time.set_quantity_scale_factor(second, One)

# 设置牛顿单位的数量维度为力量
dimsys_length_weight_time.set_quantity_dimension(newton, force)
# 设置牛顿单位的数量标度因子为千克·米/秒²
dimsys_length_weight_time.set_quantity_scale_factor(newton, kilogram*meter/second**2)

# 设置焦耳单位的数量维度为能量
dimsys_length_weight_time.set_quantity_dimension(joule, energy)
# 设置焦耳单位的数量标度因子为牛顿·米
dimsys_length_weight_time.set_quantity_scale_factor(joule, newton*meter)

# 设置瓦特单位的数量维度为功率
dimsys_length_weight_time.set_quantity_dimension(watt, power)
# 设置瓦特单位的数量标度因子为焦耳/秒
dimsys_length_weight_time.set_quantity_scale_factor(watt, joule/second)

# 设置帕斯卡单位的数量维度为压力
dimsys_length_weight_time.set_quantity_dimension(pascal, pressure)
# 设置帕斯卡单位的数量标度因子为牛顿/米²
dimsys_length_weight_time.set_quantity_scale_factor(pascal, newton/meter**2)

# 设置赫兹单位的数量维度为频率
dimsys_length_weight_time.set_quantity_dimension(hertz, frequency)
# 设置赫兹单位的数量标度因子为无量纲
dimsys_length_weight_time.set_quantity_scale_factor(hertz, One)

# 设置屈光度单位的数量维度为长度的倒数
dimsys_length_weight_time.set_quantity_dimension(dioptre, 1 / length)
# 设置屈光度单位的数量标度因子为米的倒数
dimsys_length_weight_time.set_quantity_scale_factor(dioptre, 1/meter)

# 设置公顷单位的数量维度为长度的平方
dimsys_length_weight_time.set_quantity_dimension(hectare, length**2)
# 设置公顷单位的数量标度因子为平方米乘以10000
dimsys_length_weight_time.set_quantity_scale_factor(hectare, (meter**2)*(10000))

# 设置升单位的数量维度为长度的立方
dimsys_length_weight_time.set_quantity_dimension(liter, length**3)
# 设置升单位的数量标度因子为立方米除以1000
dimsys_length_weight_time.set_quantity_scale_factor(liter, meter**3/1000)

# 牛顿常数
# 参考：NIST SP 959 (2019年6月)
# 设置牛顿常数的数量维度为长度的立方乘以质量的倒数乘以时间的负二次方
dimsys_length_weight_time.set_quantity_dimension(gravitational_constant, length ** 3 * mass ** -1 * time ** -2)
# 设置牛顿常数的数量标度因子为6.67430e-11立方米除以千克·秒²
dimsys_length_weight_time.set_quantity_scale_factor(gravitational_constant, 6.67430e-11*m**3/(kg*s**2))

# 光速
# 参考：NIST SP 959 (2019年6月)
# 设置光速的数量维度为速度
dimsys_length_weight_time.set_quantity_dimension(speed_of_light, velocity)
# 设置光速的数量标度因子为299792458米/秒
dimsys_length_weight_time.set_quantity_scale_factor(speed_of_light, 299792458*meter/second)

# 普朗克常数
# 参考：NIST SP 959 (2019年6月)
# 设置普朗克常数的数量维度为作用量
dimsys_length_weight_time.set_quantity_dimension(planck, action)
# 设置普朗克常数的数量标度因子为6.62607015e-34焦耳·秒
dimsys_length_weight_time.set_quantity_scale_factor(planck, 6.62607015e-34*joule*second)

# 缩减普朗克常数
# 参考：NIST SP 959 (2019年6月)
# 设置缩减普朗克常数的数量维度为作用量
dimsys_length_weight_time.set_quantity_dimension(hbar, action)
# 设置缩减普朗克常数的数量标度因子为普朗克常数除以(2π)
dimsys_length_weight_time.set_quantity_scale_factor(hbar, planck / (2 * pi))

# 导出的量纲符号列表
__all__ = [
    'mmHg', 'atmosphere', 'newton', 'meter', 'vacuum_permittivity', 'pascal',
    'magnetic_constant', 'angular_mil', 'julian_year', 'weber', 'exbibyte',
    'liter', 'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'planck_momentum', 'planck_density', 'gee', 'mol', 'bit', 'gray', 'kibi',
    'bar', 'curie', 'prefix_unit', 'PREFIXES', 'planck_time', 'gram',
    'candela', 'force', 'planck_intensity', 'energy', 'becquerel',
    'planck_acceleration', 'speed_of_light', 'dioptre', 'second', 'frequency',
    'Hz', 'power', 'lux', 'planck_current', 'momentum', 'tebibyte',
    'planck_power', 'degree', 'mebi', 'K', 'planck_volume',
    'quart', 'pressure', 'W', 'joule', 'boltzmann_constant', 'c', 'g',
    'planck_force', 'exbi', 's', 'watt', 'action', 'hbar', 'gibibyte',
]
    # 定义一个包含物理常数名称的元组
    phys_constants = (
        'DimensionSystem', 'cd', 'volt', 'planck_charge', 'angstrom',
        'dimsys_length_weight_time', 'pebi', 'vacuum_impedance', 'planck',
        'farad', 'gravitational_constant', 'u0', 'hertz', 'tesla', 'steradian',
        'josephson_constant', 'planck_area', 'stefan_boltzmann_constant',
        'astronomical_unit', 'J', 'N', 'planck_voltage', 'planck_energy',
        'atomic_mass_constant', 'rutherford', 'elementary_charge', 'Pa',
        'planck_mass', 'henry', 'planck_angular_frequency', 'ohm', 'pound',
        'planck_pressure', 'G', 'avogadro_number', 'psi', 'von_klitzing_constant',
        'planck_length', 'radian', 'mole', 'acceleration',
        'planck_energy_density', 'mebibyte', 'length',
        'acceleration_due_to_gravity', 'planck_temperature', 'tebi', 'inch',
        'electronvolt', 'coulomb_constant', 'kelvin', 'kPa', 'boltzmann',
        'milli_mass_unit', 'gibi', 'planck_impedance', 'electric_constant', 'kg',
        'coulomb', 'siemens', 'byte', 'atomic_mass_unit', 'm', 'kibibyte',
        'kilogram', 'lightyear', 'mass', 'time', 'pebibyte', 'velocity',
        'ampere', 'katal',
    )
    # 输出元组中的第一个物理常数名称
    print(phys_constants[0])
# 定义一个空列表，用于存储符合条件的整数
numbers = []

# 迭代整数范围从1到100（不包括101）
for i in range(1, 101):
    # 如果当前整数可以同时被3和5整除
    if i % 3 == 0 and i % 5 == 0:
        # 将当前整数添加到列表中
        numbers.append(i)

# 输出存储了能同时被3和5整除的整数的列表
print(numbers)
```