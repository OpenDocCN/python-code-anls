# `D:\src\scipysrc\sympy\sympy\physics\units\definitions\unit_definitions.py`

```
# 从 sympy.physics.units.definitions.dimension_definitions 模块导入单位定义
# 导入电流、温度、物质的量、光强、角度、电荷、电压、阻抗、电导、电容、电感、磁密度、磁通量、信息的定义
from sympy.physics.units.definitions.dimension_definitions import current, temperature, amount_of_substance, \
    luminous_intensity, angle, charge, voltage, impedance, conductance, capacitance, inductance, magnetic_density, \
    magnetic_flux, information

# 从 sympy.core.numbers 模块导入 Rational, pi
from sympy.core.numbers import (Rational, pi)
# 从 sympy.core.singleton 模块导入 S_singleton 作为 One
from sympy.core.singleton import S as S_singleton
# 从 sympy.physics.units.prefixes 模块导入各种单位前缀
from sympy.physics.units.prefixes import kilo, mega, milli, micro, deci, centi, nano, pico, kibi, mebi, gibi, tebi, pebi, exbi
# 从 sympy.physics.units.quantities 模块导入 PhysicalConstant, Quantity
from sympy.physics.units.quantities import PhysicalConstant, Quantity

# 将 S_singleton.One 赋值给 One
One = S_singleton.One

#### UNITS ####

# 无量纲单位:
# 创建名为 percent 的 Quantity 实例，代表百分比，设置其全局相对比例因子为 1/100
percent = percents = Quantity("percent", latex_repr=r"\%")
percent.set_global_relative_scale_factor(Rational(1, 100), One)

# 创建名为 permille 的 Quantity 实例，代表千分比，设置其全局相对比例因子为 1/1000
permille = Quantity("permille")
permille.set_global_relative_scale_factor(Rational(1, 1000), One)


# 角度单位（无量纲）
# 创建名为 radian 的 Quantity 实例，代表弧度，设置其全局维度为角度
rad = radian = radians = Quantity("radian", abbrev="rad")
radian.set_global_dimension(angle)

# 创建名为 degree 的 Quantity 实例，代表角度，设置其全局相对比例因子为 π/180 弧度
deg = degree = degrees = Quantity("degree", abbrev="deg", latex_repr=r"^\circ")
degree.set_global_relative_scale_factor(pi/180, radian)

# 创建名为 steradian 的 Quantity 实例，代表球面弧度，设置其全局维度为角度
sr = steradian = steradians = Quantity("steradian", abbrev="sr")

# 创建名为 angular_mil 的 Quantity 实例，代表角米，设置其全局维度为角度
mil = angular_mil = angular_mils = Quantity("angular_mil", abbrev="mil")

# 基本单位:
# 创建名为 meter 的 Quantity 实例，代表米，设置其缩写为 m
m = meter = meters = Quantity("meter", abbrev="m")

# 创建名为 gram 的 Quantity 实例，代表克，设置其缩写为 g
g = gram = grams = Quantity("gram", abbrev="g")

# 注意：千克（kilogram）的比例因子为 1000，尽管在国际单位制中千克是一个基本单位，但我们保持与 kilo 前缀兼容。
# 在未来，模块将被修改以支持所有种类的单位制（即支持所有的单位制）。
# 创建名为 kilogram 的 Quantity 实例，代表千克，设置其全局相对比例因子为 1000 克
kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
kg.set_global_relative_scale_factor(kilo, gram)

# 创建名为 second 的 Quantity 实例，代表秒，设置其缩写为 s
s = second = seconds = Quantity("second", abbrev="s")

# 创建名为 ampere 的 Quantity 实例，代表安培，设置其缩写为 A，并设置其全局维度为电流
A = ampere = amperes = Quantity("ampere", abbrev='A')
ampere.set_global_dimension(current)

# 创建名为 kelvin 的 Quantity 实例，代表开尔文，设置其缩写为 K，并设置其全局维度为温度
K = kelvin = kelvins = Quantity("kelvin", abbrev='K')
kelvin.set_global_dimension(temperature)

# 创建名为 mole 的 Quantity 实例，代表摩尔，设置其缩写为 mol，并设置其全局维度为物质的量
mol = mole = moles = Quantity("mole", abbrev="mol")
mole.set_global_dimension(amount_of_substance)

# 创建名为 candela 的 Quantity 实例，代表坎德拉，设置其缩写为 cd，并设置其全局维度为光强
cd = candela = candelas = Quantity("candela", abbrev="cd")
candela.set_global_dimension(luminous_intensity)

# 派生单位
# 创建名为 newton 的 Quantity 实例，代表牛顿，设置其缩写为 N
newton = newtons = N = Quantity("newton", abbrev="N")

# 创建名为 kilonewton 的 Quantity 实例，代表千牛顿，设置其缩写为 kN，并设置其全局相对比例因子为 1000 牛顿
kilonewton = kilonewtons = kN = Quantity("kilonewton", abbrev="kN")
kilonewton.set_global_relative_scale_factor(kilo, newton)

# 创建名为 meganewton 的 Quantity 实例，代表兆牛顿，设置其缩写为 MN，并设置其全局相对比例因子为 10^6 牛顿
meganewton = meganewtons = MN = Quantity("meganewton", abbrev="MN")
meganewton.set_global_relative_scale_factor(mega, newton)

# 创建名为 joule 的 Quantity 实例，代表焦耳，设置其缩写为 J
joule = joules = J = Quantity("joule", abbrev="J")

# 创建名为 watt 的 Quantity 实例，代表瓦特，设置其缩写为 W
watt = watts = W = Quantity("watt", abbrev="W")
# 定义几个单位对象，分别表示帕斯卡（压强单位）和赫兹（频率单位）
pascal = pascals = Pa = pa = Quantity("pascal", abbrev="Pa")
hertz = hz = Hz = Quantity("hertz", abbrev="Hz")

# CGS衍生单位：
# 定义代表达因（力单位）的对象，并设置相对于牛顿单位的比例因子为1/10^5
dyne = Quantity("dyne")
dyne.set_global_relative_scale_factor(One/10**5, newton)
# 定义代表厘尔（能量单位）的对象，并设置相对于焦耳单位的比例因子为1/10^7
erg = Quantity("erg")
erg.set_global_relative_scale_factor(One/10**7, joule)

# MKSA扩展到MKS的衍生单位：
coulomb = coulombs = C = Quantity("coulomb", abbrev='C')
coulomb.set_global_dimension(charge)
volt = volts = v = V = Quantity("volt", abbrev='V')
volt.set_global_dimension(voltage)
ohm = ohms = Quantity("ohm", abbrev='ohm', latex_repr=r"\Omega")
ohm.set_global_dimension(impedance)
siemens = S = mho = mhos = Quantity("siemens", abbrev='S')
siemens.set_global_dimension(conductance)
farad = farads = F = Quantity("farad", abbrev='F')
farad.set_global_dimension(capacitance)
henry = henrys = H = Quantity("henry", abbrev='H')
henry.set_global_dimension(inductance)
tesla = teslas = T = Quantity("tesla", abbrev='T')
tesla.set_global_dimension(magnetic_density)
weber = webers = Wb = wb = Quantity("weber", abbrev='Wb')
weber.set_global_dimension(magnetic_flux)

# CGS单位，用于电磁量：
statampere = Quantity("statampere")
statcoulomb = statC = franklin = Quantity("statcoulomb", abbrev="statC")
statvolt = Quantity("statvolt")
gauss = Quantity("gauss")
maxwell = Quantity("maxwell")
debye = Quantity("debye")
oersted = Quantity("oersted")

# 其他衍生单位：
optical_power = dioptre = diopter = D = Quantity("dioptre")
lux = lx = Quantity("lux", abbrev="lx")

# 喀特尔（催化活性单位）
katal = kat = Quantity("katal", abbrev="kat")

# 格雷（吸收剂量单位）
gray = Gy = Quantity("gray")

# 贝克勒尔（放射性单位）
becquerel = Bq = Quantity("becquerel", abbrev="Bq")


# 常见质量单位

mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
mg.set_global_relative_scale_factor(milli, gram)

ug = microgram = micrograms = Quantity("microgram", abbrev="ug", latex_repr=r"\mu\text{g}")
ug.set_global_relative_scale_factor(micro, gram)

# 原子质量单位
Da = dalton = amu = amus = atomic_mass_unit = atomic_mass_constant = PhysicalConstant("atomic_mass_constant")

t = metric_ton = tonne = Quantity("tonne", abbrev="t")
tonne.set_global_relative_scale_factor(mega, gram)

# 电子静止质量
me = electron_rest_mass = Quantity("electron_rest_mass", abbrev="me")


# 常见长度单位

km = kilometer = kilometers = Quantity("kilometer", abbrev="km")
km.set_global_relative_scale_factor(kilo, meter)

dm = decimeter = decimeters = Quantity("decimeter", abbrev="dm")
dm.set_global_relative_scale_factor(deci, meter)

cm = centimeter = centimeters = Quantity("centimeter", abbrev="cm")
cm.set_global_relative_scale_factor(centi, meter)

mm = millimeter = millimeters = Quantity("millimeter", abbrev="mm")
mm.set_global_relative_scale_factor(milli, meter)

um = micrometer = micrometers = micron = microns = \
    Quantity("micrometer", abbrev="um", latex_repr=r'\mu\text{m}')
um.set_global_relative_scale_factor(micro, meter)
# 设置微米与米的全局相对比例因子

nm = nanometer = nanometers = Quantity("nanometer", abbrev="nm")
# 定义纳米及其缩写，作为长度单位

nm.set_global_relative_scale_factor(nano, meter)
# 设置纳米与米的全局相对比例因子

pm = picometer = picometers = Quantity("picometer", abbrev="pm")
# 定义皮米及其缩写，作为长度单位

pm.set_global_relative_scale_factor(pico, meter)
# 设置皮米与米的全局相对比例因子

ft = foot = feet = Quantity("foot", abbrev="ft")
# 定义英尺及其缩写，作为长度单位

ft.set_global_relative_scale_factor(Rational(3048, 10000), meter)
# 设置英尺与米的全局相对比例因子，3048/10000是英尺到米的换算比例

inch = inches = Quantity("inch")
# 定义英寸及其复数形式，作为长度单位

inch.set_global_relative_scale_factor(Rational(1, 12), foot)
# 设置英寸与英尺的全局相对比例因子，1/12是英寸到英尺的换算比例

yd = yard = yards = Quantity("yard", abbrev="yd")
# 定义码及其缩写，作为长度单位

yd.set_global_relative_scale_factor(3, feet)
# 设置码与英尺的全局相对比例因子，1码等于3英尺

mi = mile = miles = Quantity("mile")
# 定义英里及其复数形式，作为长度单位

mi.set_global_relative_scale_factor(5280, feet)
# 设置英里与英尺的全局相对比例因子，1英里等于5280英尺

nmi = nautical_mile = nautical_miles = Quantity("nautical_mile")
# 定义海里及其复数形式，作为长度单位

nmi.set_global_relative_scale_factor(6076, feet)
# 设置海里与英尺的全局相对比例因子，1海里等于6076英尺

angstrom = angstroms = Quantity("angstrom", latex_repr=r'\r{A}')
# 定义埃及及其复数形式，作为长度单位，并指定LaTeX表示

angstrom.set_global_relative_scale_factor(Rational(1, 10**10), meter)
# 设置埃及与米的全局相对比例因子，1埃等于10^(-10)米

ha = hectare = Quantity("hectare", abbrev="ha")
# 定义公顷及其缩写，作为面积单位

l = L = liter = liters = Quantity("liter", abbrev="l")
# 定义升及其复数形式，作为体积单位

dl = dL = deciliter = deciliters = Quantity("deciliter", abbrev="dl")
# 定义分升及其复数形式，作为体积单位

dl.set_global_relative_scale_factor(Rational(1, 10), liter)
# 设置分升与升的全局相对比例因子，1分升等于0.1升

cl = cL = centiliter = centiliters = Quantity("centiliter", abbrev="cl")
# 定义厘升及其复数形式，作为体积单位

cl.set_global_relative_scale_factor(Rational(1, 100), liter)
# 设置厘升与升的全局相对比例因子，1厘升等于0.01升

ml = mL = milliliter = milliliters = Quantity("milliliter", abbrev="ml")
# 定义毫升及其复数形式，作为体积单位

ml.set_global_relative_scale_factor(Rational(1, 1000), liter)
# 设置毫升与升的全局相对比例因子，1毫升等于0.001升

ms = millisecond = milliseconds = Quantity("millisecond", abbrev="ms")
# 定义毫秒及其复数形式，作为时间单位

millisecond.set_global_relative_scale_factor(milli, second)
# 设置毫秒与秒的全局相对比例因子，milli是指10^(-3)

us = microsecond = microseconds = Quantity("microsecond", abbrev="us", latex_repr=r'\mu\text{s}')
# 定义微秒及其复数形式，作为时间单位，并指定LaTeX表示

microsecond.set_global_relative_scale_factor(micro, second)
# 设置微秒与秒的全局相对比例因子，micro是指10^(-6)

ns = nanosecond = nanoseconds = Quantity("nanosecond", abbrev="ns")
# 定义纳秒及其复数形式，作为时间单位

nanosecond.set_global_relative_scale_factor(nano, second)
# 设置纳秒与秒的全局相对比例因子，nano是指10^(-9)

ps = picosecond = picoseconds = Quantity("picosecond", abbrev="ps")
# 定义皮秒及其复数形式，作为时间单位

picosecond.set_global_relative_scale_factor(pico, second)
# 设置皮秒与秒的全局相对比例因子，pico是指10^(-12)

minute = minutes = Quantity("minute")
# 定义分钟及其复数形式，作为时间单位

minute.set_global_relative_scale_factor(60, second)
# 设置分钟与秒的全局相对比例因子，1分钟等于60秒

h = hour = hours = Quantity("hour")
# 定义小时及其复数形式，作为时间单位

hour.set_global_relative_scale_factor(60, minute)
# 设置小时与分钟的全局相对比例因子，1小时等于60分钟

day = days = Quantity("day")
# 定义日及其复数形式，作为时间单位

day.set_global_relative_scale_factor(24, hour)
# 设置日与小时的全局相对比例因子，1日等于24小时

anomalistic_year = anomalistic_years = Quantity("anomalistic_year")
# 定义异常年及其复数形式，作为时间单位

anomalistic_year.set_global_relative_scale_factor(365.259636, day)
# 设置异常年与日的全局相对比例因子

sidereal_year = sidereal_years = Quantity("sidereal_year")
# 定义恒星年及其复数形式，作为时间单位

sidereal_year.set_global_relative_scale_factor(31558149.540, seconds)
# 设置恒星年与秒的全局相对比例因子

tropical_year = tropical_years = Quantity("tropical_year")
# 定义回归年及其复数形式，作为时间单位

tropical_year.set_global_relative_scale_factor(365.24219, day)
# 设置回归年与日的全局相对比例因子

common_year = common_years = Quantity("common_year")
# 定义公历年及其复数形式，作为时间单位

common_year.set_global_relative_scale_factor(365, day)
# 设置公历年与日的全局相对比例因子，普通年有365天

julian_year = julian_years = Quantity("julian_year")
# 定义儒略年及其复数形式，作为时间单位

julian_year.set_global_relative_scale_factor((365 + One/4), day)
# 设置儒略年与日的全局相对比例因子，儒略年为365.25天

draconic_year = draconic_years = Quantity("draconic_year")
# 定义龙年及其复数形式，作为时间单位
# 设置龙年的全局相对比例因子为346.62，基于天的计量单位
draconic_year.set_global_relative_scale_factor(346.62, day)

# 定义高斯年的量及其别名
gaussian_year = gaussian_years = Quantity("gaussian_year")
# 设置高斯年的全局相对比例因子为365.2568983，基于天的计量单位
gaussian_year.set_global_relative_scale_factor(365.2568983, day)

# 定义满月周期的量及其别名
full_moon_cycle = full_moon_cycles = Quantity("full_moon_cycle")
# 设置满月周期的全局相对比例因子为411.78443029，基于天的计量单位
full_moon_cycle.set_global_relative_scale_factor(411.78443029, day)

# 年的量等于热带年的量
year = years = tropical_year


#### CONSTANTS ####

# 牛顿引力常数
G = gravitational_constant = PhysicalConstant("gravitational_constant", abbrev="G")

# 光速
c = speed_of_light = PhysicalConstant("speed_of_light", abbrev="c")

# 元电荷
elementary_charge = PhysicalConstant("elementary_charge", abbrev="e")

# 普朗克常数
planck = PhysicalConstant("planck", abbrev="h")

# 约化普朗克常数
hbar = PhysicalConstant("hbar", abbrev="hbar")

# 电子伏特
eV = electronvolt = electronvolts = PhysicalConstant("electronvolt", abbrev="eV")

# 阿伏伽德罗常数
avogadro_number = PhysicalConstant("avogadro_number")

# 阿伏伽德罗常数
avogadro = avogadro_constant = PhysicalConstant("avogadro_constant")

# 玻尔兹曼常数
boltzmann = boltzmann_constant = PhysicalConstant("boltzmann_constant")

# 斯特藩-玻尔兹曼常数
stefan = stefan_boltzmann_constant = PhysicalConstant("stefan_boltzmann_constant")

# 摩尔气体常数
R = molar_gas_constant = PhysicalConstant("molar_gas_constant", abbrev="R")

# 法拉第常数
faraday_constant = PhysicalConstant("faraday_constant")

# 约瑟夫森常数
josephson_constant = PhysicalConstant("josephson_constant", abbrev="K_j")

# 冯·克里兹常数
von_klitzing_constant = PhysicalConstant("von_klitzing_constant", abbrev="R_k")

# 地球表面重力加速度
gee = gees = acceleration_due_to_gravity = PhysicalConstant("acceleration_due_to_gravity", abbrev="g")

# 磁常数：真空磁导率
u0 = magnetic_constant = vacuum_permeability = PhysicalConstant("magnetic_constant")

# 电常数：真空介电常数
e0 = electric_constant = vacuum_permittivity = PhysicalConstant("vacuum_permittivity")

# 真空阻抗
Z0 = vacuum_impedance = PhysicalConstant("vacuum_impedance", abbrev='Z_0', latex_repr=r'Z_{0}')

# 库仑常数
coulomb_constant = coulombs_constant = electric_force_constant = \
    PhysicalConstant("coulomb_constant", abbrev="k_e")


# 大气压单位
atmosphere = atmospheres = atm = Quantity("atmosphere", abbrev="atm")

# 千帕斯卡单位
kPa = kilopascal = Quantity("kilopascal", abbrev="kPa")
# 设置千帕斯卡的全局相对比例因子为千和帕斯卡的比例
kilopascal.set_global_relative_scale_factor(kilo, Pa)

# 巴单位
bar = bars = Quantity("bar", abbrev="bar")

# 磅单位
pound = pounds = Quantity("pound")  # 精确值

# 磅力/平方英寸单位
psi = Quantity("psi")

# 汞柱单位
dHg0 = 13.5951  # 近似值，摄氏零度
mmHg = torr = Quantity("mmHg")

# 设置大气压的全局相对比例因子为101325帕斯卡
atmosphere.set_global_relative_scale_factor(101325, pascal)
# 设置巴的全局相对比例因子为100千帕斯卡
bar.set_global_relative_scale_factor(100, kPa)
# 设置磅的全局相对比例因子为45359237/100000000千克
pound.set_global_relative_scale_factor(Rational(45359237, 100000000), kg)

# 毫质单位
mmu = mmus = milli_mass_unit = Quantity("milli_mass_unit")

# 夸脱单位
quart = quarts = Quantity("quart")


# 其他方便的单位和量级
# 定义光年单位，包括多个变量名以及其缩写
ly = lightyear = lightyears = Quantity("lightyear", abbrev="ly")

# 定义天文单位，包括多个变量名以及其缩写
au = astronomical_unit = astronomical_units = Quantity("astronomical_unit", abbrev="AU")


# 定义基本的普朗克单位：
# 普朗克质量，使用缩写 m_P
planck_mass = Quantity("planck_mass", abbrev="m_P", latex_repr=r'm_\text{P}')

# 普朗克时间，使用缩写 t_P
planck_time = Quantity("planck_time", abbrev="t_P", latex_repr=r't_\text{P}')

# 普朗克温度，使用缩写 T_P
planck_temperature = Quantity("planck_temperature", abbrev="T_P",
                              latex_repr=r'T_\text{P}')

# 普朗克长度，使用缩写 l_P
planck_length = Quantity("planck_length", abbrev="l_P", latex_repr=r'l_\text{P}')

# 普朗克电荷，使用缩写 q_P
planck_charge = Quantity("planck_charge", abbrev="q_P", latex_repr=r'q_\text{P}')


# 派生的普朗克单位：
# 普朗克面积
planck_area = Quantity("planck_area")

# 普朗克体积
planck_volume = Quantity("planck_volume")

# 普朗克动量
planck_momentum = Quantity("planck_momentum")

# 普朗克能量，使用缩写 E_P
planck_energy = Quantity("planck_energy", abbrev="E_P", latex_repr=r'E_\text{P}')

# 普朗克力，使用缩写 F_P
planck_force = Quantity("planck_force", abbrev="F_P", latex_repr=r'F_\text{P}')

# 普朗克功率，使用缩写 P_P
planck_power = Quantity("planck_power", abbrev="P_P", latex_repr=r'P_\text{P}')

# 普朗克密度，使用缩写 rho_P
planck_density = Quantity("planck_density", abbrev="rho_P", latex_repr=r'\rho_\text{P}')

# 普朗克能量密度
planck_energy_density = Quantity("planck_energy_density", abbrev="rho^E_P")

# 普朗克强度，使用缩写 I_P
planck_intensity = Quantity("planck_intensity", abbrev="I_P", latex_repr=r'I_\text{P}')

# 普朗克角频率，使用缩写 omega_P
planck_angular_frequency = Quantity("planck_angular_frequency", abbrev="omega_P",
                                    latex_repr=r'\omega_\text{P}')

# 普朗克压强，使用缩写 p_P
planck_pressure = Quantity("planck_pressure", abbrev="p_P", latex_repr=r'p_\text{P}')

# 普朗克电流，使用缩写 I_P
planck_current = Quantity("planck_current", abbrev="I_P", latex_repr=r'I_\text{P}')

# 普朗克电压，使用缩写 V_P
planck_voltage = Quantity("planck_voltage", abbrev="V_P", latex_repr=r'V_\text{P}')

# 普朗克阻抗，使用缩写 Z_P
planck_impedance = Quantity("planck_impedance", abbrev="Z_P", latex_repr=r'Z_\text{P}')

# 普朗克加速度，使用缩写 a_P
planck_acceleration = Quantity("planck_acceleration", abbrev="a_P",
                               latex_repr=r'a_\text{P}')


# 信息理论单位：
# 比特
bit = bits = Quantity("bit")
# 将比特的全局维度设置为信息
bit.set_global_dimension(information)

# 字节
byte = bytes = Quantity("byte")

# kibibyte（1024字节）
kibibyte = kibibytes = Quantity("kibibyte")
# mebibyte（1024 kibibyte）
mebibyte = mebibytes = Quantity("mebibyte")
# gibibyte（1024 mebibyte）
gibibyte = gibibytes = Quantity("gibibyte")
# tebibyte（1024 gibibyte）
tebibyte = tebibytes = Quantity("tebibyte")
# pebibyte（1024 tebibyte）
pebibyte = pebibytes = Quantity("pebibyte")
# exbibyte（1024 pebibyte）
exbibyte = exbibytes = Quantity("exbibyte")

# 将字节转换为比特的全局相对比例因子设为8
byte.set_global_relative_scale_factor(8, bit)
# 将 kibibyte 转换为字节的全局相对比例因子设为 kibi（1024）
kibibyte.set_global_relative_scale_factor(kibi, byte)
# 将 mebibyte 转换为字节的全局相对比例因子设为 mebi（1024）
mebibyte.set_global_relative_scale_factor(mebi, byte)
# 将 gibibyte 转换为字节的全局相对比例因子设为 gibi（1024）
gibibyte.set_global_relative_scale_factor(gibi, byte)
# 将 tebibyte 转换为字节的全局相对比例因子设为 tebi（1024）
tebibyte.set_global_relative_scale_factor(tebi, byte)
# 将 pebibyte 转换为字节的全局相对比例因子设为 pebi（1024）
pebibyte.set_global_relative_scale_factor(pebi, byte)
# 将 exbibyte 转换为字节的全局相对比例因子设为 exbi（1024）
exbibyte.set_global_relative_scale_factor(exbi, byte)


# 较早的放射性单位
# 居里（Ci）
curie = Ci = Quantity("curie", abbrev="Ci")

# 卢瑟福（Rd）
rutherford = Rd = Quantity("rutherford", abbrev="Rd")
```