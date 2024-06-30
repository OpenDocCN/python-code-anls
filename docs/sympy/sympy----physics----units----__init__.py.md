# `D:\src\scipysrc\sympy\sympy\physics\units\__init__.py`

```
# isort:skip_file
"""
Dimensional analysis and unit systems.

This module defines dimension/unit systems and physical quantities. It is
based on a group-theoretical construction where dimensions are represented as
vectors (coefficients being the exponents), and units are defined as a dimension
to which we added a scale.

Quantities are built from a factor and a unit, and are the basic objects that
one will use when doing computations.

All objects except systems and prefixes can be used in SymPy expressions.
Note that as part of a CAS, various objects do not combine automatically
under operations.

Details about the implementation can be found in the documentation, and we
will not repeat all the explanations we gave there concerning our approach.
Ideas about future developments can be found on the `Github wiki
<https://github.com/sympy/sympy/wiki/Unit-systems>`_, and you should consult
this page if you are willing to help.

Useful functions:

- ``find_unit``: easily lookup pre-defined units.
- ``convert_to(expr, newunit)``: converts an expression into the same
    expression expressed in another unit.
"""

# 导入定义的维度和单位系统
from .dimensions import Dimension, DimensionSystem
# 导入单位系统
from .unitsystem import UnitSystem
# 导入工具函数 convert_to
from .util import convert_to
# 导入物理量 Quantity
from .quantities import Quantity

# 导入各种物理量的定义，如物质量、速度、能量等
from .definitions.dimension_definitions import (
    amount_of_substance, acceleration, action, area,
    capacitance, charge, conductance, current, energy,
    force, frequency, impedance, inductance, length,
    luminous_intensity, magnetic_density,
    magnetic_flux, mass, momentum, power, pressure, temperature, time,
    velocity, voltage, volume
)

# 将 Quantity 作为 Unit 的别名
Unit = Quantity

# 定义几个别名
speed = velocity
luminosity = luminous_intensity
magnetic_flux_density = magnetic_density
amount = amount_of_substance

# 导入单位前缀，分为 10 的幂和 2 的幂两部分
from .prefixes import (
    # 10 的幂
    yotta,
    zetta,
    exa,
    peta,
    tera,
    giga,
    mega,
    kilo,
    hecto,
    deca,
    deci,
    centi,
    milli,
    micro,
    nano,
    pico,
    femto,
    atto,
    zepto,
    yocto,
    # 2 的幂
    kibi,
    mebi,
    gibi,
    tebi,
    pebi,
    exbi,
)

# 导入其他定义，包括百分比、弧度、米制单位等
from .definitions import (
    percent, percents,
    permille,
    rad, radian, radians,
    deg, degree, degrees,
    sr, steradian, steradians,
    mil, angular_mil, angular_mils,
    m, meter, meters,
    kg, kilogram, kilograms,
    s, second, seconds,
    A, ampere, amperes,
    K, kelvin, kelvins,
    mol, mole, moles,
    cd, candela, candelas,
    g, gram, grams,
    mg, milligram, milligrams,
    ug, microgram, micrograms,
    t, tonne, metric_ton,
    newton, newtons, N,
    joule, joules, J,
    watt, watts, W,
    pascal, pascals, Pa, pa,
    hertz, hz, Hz,
    coulomb, coulombs, C,
    volt, volts, v, V,
    ohm, ohms,
    siemens, S, mho, mhos,
    farad, farads, F,
    henry, henrys, H,
    tesla, teslas, T,
    weber, webers, Wb, wb,
    optical_power, dioptre, D,
    lux, lx,
    katal, kat,
    gray, Gy,
    becquerel, Bq,
    km, kilometer, kilometers,
)
    dm, decimeter, decimeters,  # 十分米
    cm, centimeter, centimeters,  # 厘米
    mm, millimeter, millimeters,  # 毫米
    um, micrometer, micrometers, micron, microns,  # 微米
    nm, nanometer, nanometers,  # 纳米
    pm, picometer, picometers,  # 皮米
    ft, foot, feet,  # 英尺
    inch, inches,  # 英寸
    yd, yard, yards,  # 码
    mi, mile, miles,  # 英里
    nmi, nautical_mile, nautical_miles,  # 海里
    angstrom, angstroms,  # 埃
    ha, hectare,  # 公顷
    l, L, liter, liters,  # 升
    dl, dL, deciliter, deciliters,  # 分升
    cl, cL, centiliter, centiliters,  # 厘升
    ml, mL, milliliter, milliliters,  # 毫升
    ms, millisecond, milliseconds,  # 毫秒
    us, microsecond, microseconds,  # 微秒
    ns, nanosecond, nanoseconds,  # 纳秒
    ps, picosecond, picoseconds,  # 皮秒
    minute, minutes,  # 分钟
    h, hour, hours,  # 小时
    day, days,  # 天
    anomalistic_year, anomalistic_years,  # 异常年
    sidereal_year, sidereal_years,  # 恒星年
    tropical_year, tropical_years,  # 热带年
    common_year, common_years,  # 平年
    julian_year, julian_years,  # 朱利安年
    draconic_year, draconic_years,  # 龙年
    gaussian_year, gaussian_years,  # 高斯年
    full_moon_cycle, full_moon_cycles,  # 满月周期
    year, years,  # 年
    G, gravitational_constant,  # 重力常数
    c, speed_of_light,  # 光速
    elementary_charge,  # 元电荷
    hbar,  # 约化普朗克常数
    planck,  # 普朗克常数
    eV, electronvolt, electronvolts,  # 电子伏特
    avogadro_number,  # 阿伏伽德罗常数
    avogadro, avogadro_constant,  # 阿伏伽德罗常数
    boltzmann, boltzmann_constant,  # 玻尔兹曼常数
    stefan, stefan_boltzmann_constant,  # 斯特藩-玻尔兹曼常数
    R, molar_gas_constant,  # 摩尔气体常数
    faraday_constant,  # 法拉第常数
    josephson_constant,  # 约瑟夫森常数
    von_klitzing_constant,  # 冯·克里青常数
    Da, dalton, amu, amus, atomic_mass_unit, atomic_mass_constant,  # 原子质量单位
    me, electron_rest_mass,  # 电子静止质量
    gee, gees, acceleration_due_to_gravity,  # 重力加速度
    u0, magnetic_constant, vacuum_permeability,  # 真空磁导率
    e0, electric_constant, vacuum_permittivity,  # 真空介电常数
    Z0, vacuum_impedance,  # 真空阻抗
    coulomb_constant, electric_force_constant,  # 库仑常数
    atmosphere, atmospheres, atm,  # 大气压
    kPa,  # 千帕
    bar, bars,  # 巴
    pound, pounds,  # 磅
    psi,  # 磅力每平方英寸
    dHg0,  # 零摄氏度汞柱
    mmHg, torr,  # 毫米汞柱、托
    mmu, mmus, milli_mass_unit,  # 毫质量单位
    quart, quarts,  # 夸脱
    ly, lightyear, lightyears,  # 光年
    au, astronomical_unit, astronomical_units,  # 天文单位
    planck_mass,  # 普朗克质量
    planck_time,  # 普朗克时间
    planck_temperature,  # 普朗克温度
    planck_length,  # 普朗克长度
    planck_charge,  # 普朗克电荷
    planck_area,  # 普朗克面积
    planck_volume,  # 普朗克体积
    planck_momentum,  # 普朗克动量
    planck_energy,  # 普朗克能量
    planck_force,  # 普朗克力量
    planck_power,  # 普朗克功率
    planck_density,  # 普朗克密度
    planck_energy_density,  # 普朗克能量密度
    planck_intensity,  # 普朗克强度
    planck_angular_frequency,  # 普朗克角频率
    planck_pressure,  # 普朗克压力
    planck_current,  # 普朗克电流
    planck_voltage,  # 普朗克电压
    planck_impedance,  # 普朗克阻抗
    planck_acceleration,  # 普朗克加速度
    bit, bits,  # 比特
    byte,  # 字节
    kibibyte, kibibytes,  # 奇比字节
    mebibyte, mebibytes,  # 梅比字节
    gibibyte, gibibytes,  # 吉比字节
    tebibyte, tebibytes,  # 泰比字节
    pebibyte, pebibytes,  # 拍比字节
    exbibyte, exbibytes,  # 艾比字节
# 导入systems模块中的mks, mksa, si三个符号集合
from .systems import (
    mks, mksa, si
)


def find_unit(quantity, unit_system="SI"):
    """
    Return a list of matching units or dimension names.

    - If ``quantity`` is a string -- units/dimensions containing the string
    `quantity`.
    - If ``quantity`` is a unit or dimension -- units having matching base
    units or dimensions.

    Examples
    ========

    >>> from sympy.physics import units as u
    >>> u.find_unit('charge')
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit(u.charge)
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit("ampere")
    ['ampere', 'amperes']
    >>> u.find_unit('angstrom')
    ['angstrom', 'angstroms']
    >>> u.find_unit('volt')
    ['volt', 'volts', 'electronvolt', 'electronvolts', 'planck_voltage']
    >>> u.find_unit(u.inch**3)[:9]
    ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter']
    """
    # 获取指定单位系统的单位系统对象
    unit_system = UnitSystem.get_unit_system(unit_system)

    # 导入sympy.physics.units作为u
    import sympy.physics.units as u
    # 初始化返回值列表
    rv = []
    # 如果quantity是字符串
    if isinstance(quantity, str):
        # 使用列表推导式找到所有包含quantity字符串并且是Quantity类型的属性
        rv = [i for i in dir(u) if quantity in i and isinstance(getattr(u, i), Quantity)]
        # 获取quantity对应的Dimension对象
        dim = getattr(u, quantity)
        # 如果dim是Dimension类型
        if isinstance(dim, Dimension):
            # 扩展rv列表，包含与dim相关的单位
            rv.extend(find_unit(dim))
    else:
        # 遍历u中所有属性名排序后的列表
        for i in sorted(dir(u)):
            # 获取当前属性对象
            other = getattr(u, i)
            # 如果不是Quantity类型，则跳过
            if not isinstance(other, Quantity):
                continue
            # 如果quantity也是Quantity类型
            if isinstance(quantity, Quantity):
                # 如果quantity的维度与other的维度相同，将i添加到rv中
                if quantity.dimension == other.dimension:
                    rv.append(str(i))
            elif isinstance(quantity, Dimension):
                # 如果other的维度与quantity相同，将i添加到rv中
                if other.dimension == quantity:
                    rv.append(str(i))
            elif other.dimension == Dimension(unit_system.get_dimensional_expr(quantity)):
                # 如果other的维度与unit_system中的维度表达式相同，将i添加到rv中
                rv.append(str(i))
    # 返回排序后去重的rv列表，按照字符串长度和字母顺序排序
    return sorted(set(rv), key=lambda x: (len(x), x))

# NOTE: the old units module had additional variables:
# 'density', 'illuminance', 'resistance'.
# They were not dimensions, but units (old Unit class).

# 定义__all__列表，包含导出的模块和类名
__all__ = [
    'Dimension', 'DimensionSystem',
    'UnitSystem',
    'convert_to',
    'Quantity',

    'amount_of_substance', 'acceleration', 'action', 'area',
    'capacitance', 'charge', 'conductance', 'current', 'energy',
    'force', 'frequency', 'impedance', 'inductance', 'length',
    'luminous_intensity', 'magnetic_density',
    'magnetic_flux', 'mass', 'momentum', 'power', 'pressure', 'temperature', 'time',
    'velocity', 'voltage', 'volume',

    'Unit',

    'speed',
    'luminosity',
    'magnetic_flux_density',
    'amount',

    'yotta',
    'zetta',
    'exa',
    'peta',
    'tera',
    'giga',
    'mega',
    'kilo',
    'hecto',
    'deca',
    'deci',
    'centi',
    'milli',
    'micro',
    'nano',
    'pico',
    'femto',
    'atto',
    'zepto',
    'yocto',

    'kibi',
    'mebi',
    'gibi',
    'tebi',
    'pebi',
    'exbi',

    'percent', 'percents',
    'permille',
    'rad', 'radian', 'radians',
]
    # 单位和常量的简写与完整写法的对应关系
        'deg', 'degree', 'degrees',
        'sr', 'steradian', 'steradians',
        'mil', 'angular_mil', 'angular_mils',
        'm', 'meter', 'meters',
        'kg', 'kilogram', 'kilograms',
        's', 'second', 'seconds',
        'A', 'ampere', 'amperes',
        'K', 'kelvin', 'kelvins',
        'mol', 'mole', 'moles',
        'cd', 'candela', 'candelas',
        'g', 'gram', 'grams',
        'mg', 'milligram', 'milligrams',
        'ug', 'microgram', 'micrograms',
        't', 'tonne', 'metric_ton',
        'newton', 'newtons', 'N',
        'joule', 'joules', 'J',
        'watt', 'watts', 'W',
        'pascal', 'pascals', 'Pa', 'pa',
        'hertz', 'hz', 'Hz',
        'coulomb', 'coulombs', 'C',
        'volt', 'volts', 'v', 'V',
        'ohm', 'ohms',
        'siemens', 'S', 'mho', 'mhos',
        'farad', 'farads', 'F',
        'henry', 'henrys', 'H',
        'tesla', 'teslas', 'T',
        'weber', 'webers', 'Wb', 'wb',
        'optical_power', 'dioptre', 'D',
        'lux', 'lx',
        'katal', 'kat',
        'gray', 'Gy',
        'becquerel', 'Bq',
        'km', 'kilometer', 'kilometers',
        'dm', 'decimeter', 'decimeters',
        'cm', 'centimeter', 'centimeters',
        'mm', 'millimeter', 'millimeters',
        'um', 'micrometer', 'micrometers', 'micron', 'microns',
        'nm', 'nanometer', 'nanometers',
        'pm', 'picometer', 'picometers',
        'ft', 'foot', 'feet',
        'inch', 'inches',
        'yd', 'yard', 'yards',
        'mi', 'mile', 'miles',
        'nmi', 'nautical_mile', 'nautical_miles',
        'angstrom', 'angstroms',
        'ha', 'hectare',
        'l', 'L', 'liter', 'liters',
        'dl', 'dL', 'deciliter', 'deciliters',
        'cl', 'cL', 'centiliter', 'centiliters',
        'ml', 'mL', 'milliliter', 'milliliters',
        'ms', 'millisecond', 'milliseconds',
        'us', 'microsecond', 'microseconds',
        'ns', 'nanosecond', 'nanoseconds',
        'ps', 'picosecond', 'picoseconds',
        'minute', 'minutes',
        'h', 'hour', 'hours',
        'day', 'days',
        'anomalistic_year', 'anomalistic_years',
        'sidereal_year', 'sidereal_years',
        'tropical_year', 'tropical_years',
        'common_year', 'common_years',
        'julian_year', 'julian_years',
        'draconic_year', 'draconic_years',
        'gaussian_year', 'gaussian_years',
        'full_moon_cycle', 'full_moon_cycles',
        'year', 'years',
        'G', 'gravitational_constant',
        'c', 'speed_of_light',
        'elementary_charge',
        'hbar',
        'planck',
        'eV', 'electronvolt', 'electronvolts',
        'avogadro_number',
        'avogadro', 'avogadro_constant',
        'boltzmann', 'boltzmann_constant',
        'stefan', 'stefan_boltzmann_constant',
        'R', 'molar_gas_constant',
        'faraday_constant',
        'josephson_constant',
        'von_klitzing_constant',
        'Da', 'dalton', 'amu', 'amus', 'atomic_mass_unit', 'atomic_mass_constant',
        'me', 'electron_rest_mass',
        'gee', 'gees', 'acceleration_due_to_gravity',
        'u0', 'magnetic_constant', 'vacuum_permeability',
        'e0', 'electric_constant', 'vacuum_permittivity',
        'Z0', 'vacuum_impedance',
        'coulomb_constant', 'electric_force_constant',
        'atmosphere', 'atmospheres', 'atm',
        'kPa',
        'bar', 'bars',
    # 定义了一系列单位和常量的名称，用于物理量和计算机存储量的表示
    'pound', 'pounds',          # 磅
    'psi',                      # 磅力/平方英寸
    'dHg0',                     # 汞柱高度的标准状态
    'mmHg', 'torr',             # 毫米汞柱，托
    'mmu', 'mmus', 'milli_mass_unit',  # 毫质量单位
    'quart', 'quarts',          # 夸脱
    'ly', 'lightyear', 'lightyears',     # 光年
    'au', 'astronomical_unit', 'astronomical_units',  # 天文单位
    'planck_mass',              # 普朗克质量
    'planck_time',              # 普朗克时间
    'planck_temperature',       # 普朗克温度
    'planck_length',            # 普朗克长度
    'planck_charge',            # 普朗克电荷
    'planck_area',              # 普朗克面积
    'planck_volume',            # 普朗克体积
    'planck_momentum',          # 普朗克动量
    'planck_energy',            # 普朗克能量
    'planck_force',             # 普朗克力
    'planck_power',             # 普朗克功率
    'planck_density',           # 普朗克密度
    'planck_energy_density',    # 普朗克能量密度
    'planck_intensity',         # 普朗克强度
    'planck_angular_frequency', # 普朗克角频率
    'planck_pressure',          # 普朗克压强
    'planck_current',           # 普朗克电流
    'planck_voltage',           # 普朗克电压
    'planck_impedance',         # 普朗克阻抗
    'planck_acceleration',      # 普朗克加速度
    'bit', 'bits',              # 位
    'byte',                     # 字节
    'kibibyte', 'kibibytes',    # 奇比字节（1024字节）
    'mebibyte', 'mebibytes',    # 梅比字节（1024 * 1024字节）
    'gibibyte', 'gibibytes',    # 吉比字节（1024 * 1024 * 1024字节）
    'tebibyte', 'tebibytes',    # 太比字节（1024 * 1024 * 1024 * 1024字节）
    'pebibyte', 'pebibytes',    # 拍比字节（1024 * 1024 * 1024 * 1024 * 1024字节）
    'exbibyte', 'exbibytes',    # 艾比字节（1024 * 1024 * 1024 * 1024 * 1024 * 1024字节）
    
    'mks', 'mksa', 'si',        # 国际单位制（米-千克-秒制）
]



# 这行代码似乎是意外出现的右方括号，可能是拼写错误或代码错误。
# 它没有明确的语义或功能，因为单独的右方括号不能独立存在或完成任何操作。
# 可能需要检查代码逻辑，确认这个括号是否应该与其他代码结合使用。
```