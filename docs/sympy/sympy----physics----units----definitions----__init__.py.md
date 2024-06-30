# `D:\src\scipysrc\sympy\sympy\physics\units\definitions\__init__.py`

```
# 从.unit_definitions模块中导入一系列单位定义，包括百分比、千分比、弧度、角度、米、千克、秒等
from .unit_definitions import (
    percent, percents,             # 百分比相关
    permille,                      # 千分比
    rad, radian, radians,         # 弧度相关
    deg, degree, degrees,         # 角度相关
    sr, steradian, steradians,     # 球面度相关
    mil, angular_mil, angular_mils,# 角毫相关
    m, meter, meters,             # 米相关
    kg, kilogram, kilograms,      # 千克相关
    s, second, seconds,           # 秒相关
    A, ampere, amperes,           # 安培相关
    K, kelvin, kelvins,           # 开尔文相关
    mol, mole, moles,             # 摩尔相关
    cd, candela, candelas,        # 坎德拉相关
    g, gram, grams,               # 克相关
    mg, milligram, milligrams,    # 毫克相关
    ug, microgram, micrograms,    # 微克相关
    t, tonne, metric_ton,         # 吨相关
    newton, newtons, N,           # 牛顿相关
    joule, joules, J,             # 焦耳相关
    watt, watts, W,               # 瓦特相关
    pascal, pascals, Pa, pa,      # 帕斯卡尔相关
    hertz, hz, Hz,                # 赫兹相关
    coulomb, coulombs, C,         # 库仑相关
    volt, volts, v, V,            # 伏特相关
    ohm, ohms,                    # 欧姆相关
    siemens, S, mho, mhos,        # 西门子相关
    farad, farads, F,             # 法拉相关
    henry, henrys, H,             # 亨利相关
    tesla, teslas, T,             # 特斯拉相关
    weber, webers, Wb, wb,        # 韦伯相关
    optical_power, dioptre, D,    # 光学功率相关
    lux, lx,                      # 勒克斯相关
    katal, kat,                   # 卡特相关
    gray, Gy,                     # 格雷相关
    becquerel, Bq,                # 贝克勒尔相关
    km, kilometer, kilometers,    # 千米相关
    dm, decimeter, decimeters,    # 分米相关
    cm, centimeter, centimeters,  # 厘米相关
    mm, millimeter, millimeters,  # 毫米相关
    um, micrometer, micrometers, micron, microns,  # 微米相关
    nm, nanometer, nanometers,    # 纳米相关
    pm, picometer, picometers,    # 皮米相关
    ft, foot, feet,               # 英尺相关
    inch, inches,                 # 英寸相关
    yd, yard, yards,              # 码相关
    mi, mile, miles,              # 英里相关
    nmi, nautical_mile, nautical_miles,  # 海里相关
    ha, hectare,                  # 公顷相关
    l, L, liter, liters,          # 升相关
    dl, dL, deciliter, deciliters, # 分升相关
    cl, cL, centiliter, centiliters, # 厘升相关
    ml, mL, milliliter, milliliters, # 毫升相关
    ms, millisecond, milliseconds,# 毫秒相关
    us, microsecond, microseconds,# 微秒相关
    ns, nanosecond, nanoseconds,  # 纳秒相关
    ps, picosecond, picoseconds,  # 皮秒相关
    minute, minutes,              # 分钟相关
    h, hour, hours,               # 小时相关
    day, days,                    # 天相关
    anomalistic_year, anomalistic_years,  # 周年差相关
    sidereal_year, sidereal_years,# 恒星年相关
    tropical_year, tropical_years,# 热带年相关
    common_year, common_years,    # 公历年相关
    julian_year, julian_years,    # 朱利安年相关
    draconic_year, draconic_years,# 龙年相关
    gaussian_year, gaussian_years,# 高斯年相关
    full_moon_cycle, full_moon_cycles,  # 满月周期相关
    year, years,                  # 年相关
    G, gravitational_constant,    # 万有引力常数相关
    c, speed_of_light,            # 光速相关
    elementary_charge,            # 元电荷相关
    hbar,                         # 约化普朗克常数相关
    planck,                       # 普朗克常数相关
    eV, electronvolt, electronvolts,  # 电子伏特相关
    avogadro_number,              # 阿伏伽德罗常数相关
    avogadro, avogadro_constant,  # 阿伏伽德罗常数相关
    boltzmann, boltzmann_constant,# 玻尔兹曼常数相关
    stefan, stefan_boltzmann_constant,  # 斯特凡·玻尔兹曼常数相关
    R, molar_gas_constant,        # 摩尔气体常数相关
    faraday_constant,             # 法拉第常数相关
    josephson_constant,           # 约瑟夫逊常数相关
    von_klitzing_constant,        # 冯·克里兹因常数相关
    Da, dalton, amu, amus, atomic_mass_unit, atomic_mass_constant,  # 原子质量单位相关
    me, electron_rest_mass,       # 电子静止质量相关
    gee, gees, acceleration_due_to_gravity,  # 重力加速度相关
    u0, magnetic_constant, vacuum_permeability,  # 真空磁导率相关
    e0, electric_constant, vacuum_permittivity,  # 真空介电常数相关
    Z0, vacuum_impedance,        # 真空阻抗相关
    coulomb_constant, coulombs_constant, electric_force_constant,  # 库仑常数相关
    atmosphere, atmospheres, atm, # 大气压相关
    kPa, kilopascal,              # 千帕相关
    bar, bars,                    # 巴相关
    pound, pounds,                # 磅相关
    psi,                          # 磅力/平方英寸相关
    dHg0,                         # 标准汞柱高度相关
    mmHg, torr,                   # 毫米汞柱相关
    mmu, mmus, milli_mass_unit,   # 毫质量单位相关
    quart, quarts,                # 夸脱相关
    angstrom, angstroms,          # 埃相关
    ly, lightyear, lightyears,    # 光年相关
    au, astronomical_unit, astronomical_units,  # 天文单位相关
    planck_mass,                  # 普朗克质量相关
    planck_time,                  # 普朗克时间相关
    planck_temperature,           # 普朗克温度相关
    planck_length,                # 普朗克长度相关
    planck_charge,                # 普朗克电荷相关
    planck_area,                  # 普朗克面积相关
    planck_volume,                # 普朗克体积相关
    planck_momentum,              # 普朗克动量相关
    planck_energy,          # Planck energy (derived unit in quantum mechanics)
    planck_force,           # Planck force (derived unit in quantum mechanics)
    planck_power,           # Planck power (derived unit in quantum mechanics)
    planck_density,         # Planck density (derived unit in quantum mechanics)
    planck_energy_density,  # Planck energy density (derived unit in quantum mechanics)
    planck_intensity,       # Planck intensity (derived unit in quantum mechanics)
    planck_angular_frequency,  # Planck angular frequency (derived unit in quantum mechanics)
    planck_pressure,        # Planck pressure (derived unit in quantum mechanics)
    planck_current,         # Planck current (derived unit in quantum mechanics)
    planck_voltage,         # Planck voltage (derived unit in quantum mechanics)
    planck_impedance,       # Planck impedance (derived unit in quantum mechanics)
    planck_acceleration,    # Planck acceleration (derived unit in quantum mechanics)
    bit,                    # A binary digit, the smallest unit of data in computing
    bits,                   # Plural of bit, multiple binary digits
    byte,                   # Unit of digital information typically consisting of 8 bits
    kibibyte,               # 1024 bytes, a unit of digital information based on binary multiples
    kibibytes,              # Plural of kibibyte
    mebibyte,               # 1024 kibibytes, a unit of digital information based on binary multiples
    mebibytes,              # Plural of mebibyte
    gibibyte,               # 1024 mebibytes, a unit of digital information based on binary multiples
    gibibytes,              # Plural of gibibyte
    tebibyte,               # 1024 gibibytes, a unit of digital information based on binary multiples
    tebibytes,              # Plural of tebibyte
    pebibyte,               # 1024 tebibytes, a unit of digital information based on binary multiples
    pebibytes,              # Plural of pebibyte
    exbibyte,               # 1024 pebibytes, a unit of digital information based on binary multiples
    exbibytes,              # Plural of exbibyte
    curie,                  # Unit of radioactivity, approximately 3.7 × 10^10 disintegrations per second
    rutherford              # Unit of radioactivity, approximately 10^6 disintegrations per second
# __all__ 列表定义了模块中可以被导出的公共符号
__all__ = [
    'percent', 'percents',              # 百分比
    'permille',                         # 千分比
    'rad', 'radian', 'radians',         # 弧度
    'deg', 'degree', 'degrees',         # 角度
    'sr', 'steradian', 'steradians',    # 固体角度
    'mil', 'angular_mil', 'angular_mils',   # 角毫
    'm', 'meter', 'meters',             # 米
    'kg', 'kilogram', 'kilograms',      # 千克
    's', 'second', 'seconds',           # 秒
    'A', 'ampere', 'amperes',           # 安培
    'K', 'kelvin', 'kelvins',           # 开尔文
    'mol', 'mole', 'moles',             # 摩尔
    'cd', 'candela', 'candelas',        # 坎德拉
    'g', 'gram', 'grams',               # 克
    'mg', 'milligram', 'milligrams',    # 毫克
    'ug', 'microgram', 'micrograms',    # 微克
    't', 'tonne', 'metric_ton',         # 公吨
    'newton', 'newtons', 'N',           # 牛顿
    'joule', 'joules', 'J',             # 焦耳
    'watt', 'watts', 'W',               # 瓦特
    'pascal', 'pascals', 'Pa', 'pa',    # 帕斯卡
    'hertz', 'hz', 'Hz',                # 赫兹
    'coulomb', 'coulombs', 'C',         # 库仑
    'volt', 'volts', 'v', 'V',          # 伏特
    'ohm', 'ohms',                      # 欧姆
    'siemens', 'S', 'mho', 'mhos',      # 西门子
    'farad', 'farads', 'F',             # 法拉
    'henry', 'henrys', 'H',             # 亨利
    'tesla', 'teslas', 'T',             # 特斯拉
    'weber', 'webers', 'Wb', 'wb',      # 韦伯
    'optical_power', 'dioptre', 'D',    # 光度
    'lux', 'lx',                        # 勒克斯
    'katal', 'kat',                     # 卡塔尔
    'gray', 'Gy',                       # 格雷
    'becquerel', 'Bq',                  # 贝克勒尔
    'km', 'kilometer', 'kilometers',    # 千米
    'dm', 'decimeter', 'decimeters',    # 分米
    'cm', 'centimeter', 'centimeters',  # 厘米
    'mm', 'millimeter', 'millimeters',  # 毫米
    'um', 'micrometer', 'micrometers', 'micron', 'microns',   # 微米
    'nm', 'nanometer', 'nanometers',    # 纳米
    'pm', 'picometer', 'picometers',    # 皮米
    'ft', 'foot', 'feet',               # 英尺
    'inch', 'inches',                   # 英寸
    'yd', 'yard', 'yards',              # 码
    'mi', 'mile', 'miles',              # 英里
    'nmi', 'nautical_mile', 'nautical_miles',   # 海里
    'ha', 'hectare',                    # 公顷
    'l', 'L', 'liter', 'liters',         # 升
    'dl', 'dL', 'deciliter', 'deciliters',    # 分升
    'cl', 'cL', 'centiliter', 'centiliters',  # 厘升
    'ml', 'mL', 'milliliter', 'milliliters',  # 毫升
    'ms', 'millisecond', 'milliseconds',      # 毫秒
    'us', 'microsecond', 'microseconds',      # 微秒
    'ns', 'nanosecond', 'nanoseconds',        # 纳秒
    'ps', 'picosecond', 'picoseconds',        # 皮秒
    'minute', 'minutes',                # 分钟
    'h', 'hour', 'hours',               # 小时
    'day', 'days',                      # 天
    'anomalistic_year', 'anomalistic_years',   # 异常年
    'sidereal_year', 'sidereal_years',   # 恒星年
    'tropical_year', 'tropical_years',   # 热带年
    'common_year', 'common_years',       # 公历年
    'julian_year', 'julian_years',       # 朱利安年
    'draconic_year', 'draconic_years',   # 龙年
    'gaussian_year', 'gaussian_years',   # 高斯年
    'full_moon_cycle', 'full_moon_cycles',   # 满月周期
    'year', 'years',                    # 年
    'G', 'gravitational_constant',      # 万有引力常数
    'c', 'speed_of_light',              # 光速
    'elementary_charge',                # 元电荷
    'hbar',                             # 约化普朗克常数
    'planck',                           # 普朗克常数
    'eV', 'electronvolt', 'electronvolts',   # 电子伏特
    'avogadro_number',                  # 阿伏伽德罗常数
    'avogadro', 'avogadro_constant',    # 阿伏伽德罗常数
    'boltzmann', 'boltzmann_constant',  # 玻尔兹曼常数
    'stefan', 'stefan_boltzmann_constant',   # 斯特藩-玻尔兹曼常数
    'R', 'molar_gas_constant',          # 摩尔气体常数
    'faraday_constant',                 # 法拉第常数
    'josephson_constant',               # 约瑟夫森常数
    'von_klitzing_constant',            # 冯·克里青常数
    'Da', 'dalton', 'amu', 'amus', 'atomic_mass_unit', 'atomic_mass_constant',   # 道尔顿/原子质量单位
    'me', 'electron_rest_mass',         # 电子静止质量
    'gee', 'gees', 'acceleration_due_to_gravity',   # 重力加速度
    'u0', 'magnetic_constant', 'vacuum_permeability',   # 真空磁导率
    'e0', 'electric_constant', 'vacuum_permittivity',   # 真空介电常数
    'Z0', 'vacuum_impedance',           # 真空阻抗
]
    # 创建包含物理常数和单位的元组
    constants_and_units = (
        'coulomb_constant', 'coulombs_constant', 'electric_force_constant',
        'atmosphere', 'atmospheres', 'atm',
        'kPa', 'kilopascal',
        'bar', 'bars',
        'pound', 'pounds',
        'psi',
        'dHg0',
        'mmHg', 'torr',
        'mmu', 'mmus', 'milli_mass_unit',
        'quart', 'quarts',
        'angstrom', 'angstroms',
        'ly', 'lightyear', 'lightyears',
        'au', 'astronomical_unit', 'astronomical_units',
        'planck_mass',
        'planck_time',
        'planck_temperature',
        'planck_length',
        'planck_charge',
        'planck_area',
        'planck_volume',
        'planck_momentum',
        'planck_energy',
        'planck_force',
        'planck_power',
        'planck_density',
        'planck_energy_density',
        'planck_intensity',
        'planck_angular_frequency',
        'planck_pressure',
        'planck_current',
        'planck_voltage',
        'planck_impedance',
        'planck_acceleration',
        'bit', 'bits',
        'byte',
        'kibibyte', 'kibibytes',
        'mebibyte', 'mebibytes',
        'gibibyte', 'gibibytes',
        'tebibyte', 'tebibytes',
        'pebibyte', 'pebibytes',
        'exbibyte', 'exbibytes',
        'curie', 'rutherford',
    )
]



# 这行代码似乎是不完整的，缺少了它所属的上下文。可能是一个语法错误或者不完整的表达式。
```