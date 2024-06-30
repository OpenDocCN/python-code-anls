# `D:\src\scipysrc\scipy\scipy\constants\_constants.py`

```
"""
Collection of physical constants and conversion factors.

Most constants are in SI units, so you can do
print '10 mile per minute is', 10*mile/minute, 'm/s or', 10*mile/(minute*knot), 'knots'

The list is not meant to be comprehensive, but just convenient for everyday use.
"""

# 导入未来的类型注解支持
from __future__ import annotations

# 导入数学库并重命名为 _math
import math as _math

# 导入 CODATA 模块的 value 属性
from ._codata import value as _cd

# 如果当前环境支持类型检查，导入 numpy 的类型注解
if TYPE_CHECKING:
    import numpy.typing as npt

# 从 scipy 库中导入 array_namespace 和 _asarray 函数
from scipy._lib._array_api import array_namespace, _asarray


"""
BasSw 2006
physical constants: imported from CODATA
unit conversion: see e.g., NIST special publication 811
Use at own risk: double-check values before calculating your Mars orbit-insertion burn.
Some constants exist in a few variants, which are marked with suffixes.
The ones without any suffix should be the most common ones.
"""

# 定义 __all__ 列表，包含导出的常量和单位名称
__all__ = [
    'Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'G',
    'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg',
    'Stefan_Boltzmann', 'Wien', 'acre', 'alpha',
    'angstrom', 'arcmin', 'arcminute', 'arcsec',
    'arcsecond', 'astronomical_unit', 'atm',
    'atmosphere', 'atomic_mass', 'atto', 'au', 'bar',
    'barrel', 'bbl', 'blob', 'c', 'calorie',
    'calorie_IT', 'calorie_th', 'carat', 'centi',
    'convert_temperature', 'day', 'deci', 'degree',
    'degree_Fahrenheit', 'deka', 'dyn', 'dyne', 'e',
    'eV', 'electron_mass', 'electron_volt',
    'elementary_charge', 'epsilon_0', 'erg',
    'exa', 'exbi', 'femto', 'fermi', 'fine_structure',
    'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp',
    'foot', 'g', 'gallon', 'gallon_US', 'gallon_imp',
    'gas_constant', 'gibi', 'giga', 'golden', 'golden_ratio',
    'grain', 'gram', 'gravitational_constant', 'h', 'hbar',
    'hectare', 'hecto', 'horsepower', 'hour', 'hp',
    'inch', 'k', 'kgf', 'kibi', 'kilo', 'kilogram_force',
    'kmh', 'knot', 'lambda2nu', 'lb', 'lbf',
    'light_year', 'liter', 'litre', 'long_ton', 'm_e',
    'm_n', 'm_p', 'm_u', 'mach', 'mebi', 'mega',
    'metric_ton', 'micro', 'micron', 'mil', 'mile',
    'milli', 'minute', 'mmHg', 'mph', 'mu_0', 'nano',
    'nautical_mile', 'neutron_mass', 'nu2lambda',
    'ounce', 'oz', 'parsec', 'pebi', 'peta',
    'pi', 'pico', 'point', 'pound', 'pound_force',
    'proton_mass', 'psi', 'pt', 'quecto', 'quetta', 'ronna', 'ronto',
    'short_ton', 'sigma', 'slinch', 'slug', 'speed_of_light',
    'speed_of_sound', 'stone', 'survey_foot',
    'survey_mile', 'tebi', 'tera', 'ton_TNT',
    'torr', 'troy_ounce', 'troy_pound', 'u',
    'week', 'yard', 'year', 'yobi', 'yocto',
    'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta'
]


# 数学常数
# 定义圆周率 pi
pi = _math.pi
# 定义黄金比例及其别名
golden = golden_ratio = (1 + _math.sqrt(5)) / 2

# SI 单位前缀定义
quetta = 1e30
ronna = 1e27
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21  # 定义zepto为10的负21次方，用于表示极小的数值

yocto = 1e-24  # 定义yocto为10的负24次方，更小的数值单位

ronto = 1e-27  # 定义ronto为10的负27次方，再小一些的数值单位

quecto = 1e-30  # 定义quecto为10的负30次方，非常小的数值单位

# binary prefixes
kibi = 2**10  # 定义kibi为2的10次方，即1024，用于二进制单位前缀

mebi = 2**20  # 定义mebi为2的20次方，即1048576，更大的二进制单位前缀

gibi = 2**30  # 定义gibi为2的30次方，即1073741824，更大的二进制单位前缀

tebi = 2**40  # 定义tebi为2的40次方，更大的二进制单位前缀

pebi = 2**50  # 定义pebi为2的50次方，更大的二进制单位前缀

exbi = 2**60  # 定义exbi为2的60次方，更大的二进制单位前缀

zebi = 2**70  # 定义zebi为2的70次方，极大的二进制单位前缀

yobi = 2**80  # 定义yobi为2的80次方，极大的二进制单位前缀

# physical constants
c = speed_of_light = _cd('speed of light in vacuum')  # 光速，在真空中的速度

mu_0 = _cd('vacuum mag. permeability')  # 真空中的磁导率

epsilon_0 = _cd('vacuum electric permittivity')  # 真空中的电容率

h = Planck = _cd('Planck constant')  # 普朗克常数

hbar = h / (2 * pi)  # 常数h除以2π，得到约化普朗克常数

G = gravitational_constant = _cd('Newtonian constant of gravitation')  # 牛顿引力常数

g = _cd('standard acceleration of gravity')  # 标准重力加速度

e = elementary_charge = _cd('elementary charge')  # 元电荷

R = gas_constant = _cd('molar gas constant')  # 气体常数，摩尔气体常数

alpha = fine_structure = _cd('fine-structure constant')  # 微结构常数，也称为精细结构常数

N_A = Avogadro = _cd('Avogadro constant')  # 阿伏伽德罗常数

k = Boltzmann = _cd('Boltzmann constant')  # 玻尔兹曼常数

sigma = Stefan_Boltzmann = _cd('Stefan-Boltzmann constant')  # 斯特藩-玻尔兹曼常数

Wien = _cd('Wien wavelength displacement law constant')  # 维恩位移定律常数

Rydberg = _cd('Rydberg constant')  # 雷德堡常数

# mass in kg
gram = 1e-3  # 克转换为千克

metric_ton = 1e3  # 公吨，即千克

grain = 64.79891e-6  # 格令，常用于质量测量

lb = pound = 7000 * grain  # 磅，常用于质量测量（阿沃伊德皮斯）

blob = slinch = pound * g / 0.0254  # 斯林奇，常用于惯性测量

slug = blob / 12  # 重量单位，常用于工程中的质量和加速度

oz = ounce = pound / 16  # 盎司，常用于质量测量

stone = 14 * pound  # 英国常用的质量单位

long_ton = 2240 * pound  # 长吨，英国常用的质量单位

short_ton = 2000 * pound  # 短吨，常用于质量测量

troy_ounce = 480 * grain  # 特洛伊盎司，用于金属和宝石的质量

troy_pound = 12 * troy_ounce  # 特洛伊磅，用于金属和宝石的质量

carat = 200e-6  # 克拉，用于宝石的质量

m_e = electron_mass = _cd('electron mass')  # 电子质量

m_p = proton_mass = _cd('proton mass')  # 质子质量

m_n = neutron_mass = _cd('neutron mass')  # 中子质量

m_u = u = atomic_mass = _cd('atomic mass constant')  # 原子质量单位

# angle in rad
degree = pi / 180  # 角度单位，弧度

arcmin = arcminute = degree / 60  # 弧度的分之一，常用于角度测量

arcsec = arcsecond = arcmin / 60  # 弧度的秒，角度测量的一部分

# time in second
minute = 60.0  # 分钟到秒的转换

hour = 60 * minute  # 小时到秒的转换

day = 24 * hour  # 天到秒的转换

week = 7 * day  # 周到秒的转换

year = 365 * day  # 年到秒的转换

Julian_year = 365.25 * day  # 儒略年到秒的转换

# length in meter
inch = 0.0254  # 英寸到米的转换

foot = 12 * inch  # 英尺到米的转换

yard = 3 * foot  # 码到米的转换

mile = 1760 * yard  # 英里到米的转换

mil = inch / 1000  # 千分英寸到米的转换

pt = point = inch / 72  # 印刷点到米的转换

survey_foot = 1200.0 / 3937  # 测量英尺到米的转换

survey_mile = 5280 * survey_foot  # 测量英里到米的转换

nautical_mile = 1852.0  # 海里到米的转换

fermi = 1e-15  # 飞米，极小的长度单位

angstrom = 1e-10  # 埃，用于测量原子尺寸

micron = 1e-6  # 微米，也称为百万分之一米

au = astronomical_unit = 149597870700.0  # 天文单位，用于天文距离测量

light_year = Julian_year * c  # 光年，以光速和儒略年为基准

parsec = au / arcsec  # 帕秒，天文学的距离单位

# pressure in pascal
atm = atmosphere = _cd('standard atmosphere')  # 标准大气压

bar = 1e5  # 巴，用于压力单位

torr = mmHg = atm / 760  # 托，汞柱压力单位

psi = pound * g / (inch * inch)  # 磅力每平方英寸，用于压力单位

# area in meter**2
hectare = 1e4  # 公顷，用于面积单位

acre = 43560 * foot**2  # 英亩，用于面积单位

# volume in meter**3
litre = liter = 1e-3  # 升，用于体积单位

gallon = gallon_US = 231 * inch**3  # 加仑（美制），用于体积单位

fluid
Btu = Btu_IT = pound * degree_Fahrenheit * calorie_IT / gram
ton_TNT = 1e9 * calorie_th
# Wh = watt_hour

# power in watt
hp = horsepower = 550 * foot * pound * g
# 定义并计算马力（horsepower），公式为 550 * foot * pound * g

# force in newton
dyn = dyne = 1e-5
lbf = pound_force = pound * g
kgf = kilogram_force = g  # * 1 kg
# 定义并计算力的单位换算，包括 dyn, lbf, kgf 对应的单位换算公式

# functions for conversions that are not linear

def convert_temperature(
    val: npt.ArrayLike,
    old_scale: str,
    new_scale: str,
) -> Any:
    """
    Convert from a temperature scale to another one among Celsius, Kelvin,
    Fahrenheit, and Rankine scales.

    Parameters
    ----------
    val : array_like
        Value(s) of the temperature(s) to be converted expressed in the
        original scale.
    old_scale : str
        Specifies as a string the original scale from which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').
    new_scale : str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').

    Returns
    -------
    res : float or array of floats
        Value(s) of the converted temperature(s) expressed in the new scale.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    >>> from scipy.constants import convert_temperature
    >>> import numpy as np
    >>> convert_temperature(np.array([-40, 40]), 'Celsius', 'Kelvin')
    array([ 233.15,  313.15])

    """
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp, subok=True)
    # 根据旧温度标度转换为开尔文（Kelvin）
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _val * 5 / 9
    else:
        raise NotImplementedError(f"{old_scale=} is unsupported: supported scales "
                                   "are Celsius, Kelvin, Fahrenheit, and "
                                   "Rankine")
    # 根据开尔文转换为新温度标度
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        # 如果新的温度标度不是支持的几种类型之一，抛出未实现错误，指明支持的标度类型
        raise NotImplementedError(f"{new_scale=} is unsupported: supported "
                                   "scales are 'Celsius', 'Kelvin', "
                                   "'Fahrenheit', and 'Rankine'")
    
    # 返回转换后的温度结果
    return res
# 定义函数 lambda2nu，将波长转换为光学频率

def lambda2nu(lambda_: npt.ArrayLike) -> Any:
    """
    Convert wavelength to optical frequency

    Parameters
    ----------
    lambda_ : array_like
        Wavelength(s) to be converted.

    Returns
    -------
    nu : float or array of floats
        Equivalent optical frequency.

    Notes
    -----
    Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    Examples
    --------
    >>> from scipy.constants import lambda2nu, speed_of_light
    >>> import numpy as np
    >>> lambda2nu(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    """
    # 使用 array_namespace 函数处理 lambda_，返回合适的数组对象
    xp = array_namespace(lambda_)
    # 返回光学频率 nu，计算公式为 c / lambda_
    return c / _asarray(lambda_, xp=xp, subok=True)


# 定义函数 nu2lambda，将光学频率转换为波长

def nu2lambda(nu: npt.ArrayLike) -> Any:
    """
    Convert optical frequency to wavelength.

    Parameters
    ----------
    nu : array_like
        Optical frequency to be converted.

    Returns
    -------
    lambda : float or array of floats
        Equivalent wavelength(s).

    Notes
    -----
    Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    Examples
    --------
    >>> from scipy.constants import nu2lambda, speed_of_light
    >>> import numpy as np
    >>> nu2lambda(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    """
    # 使用 array_namespace 函数处理 nu，返回合适的数组对象
    xp = array_namespace(nu)
    # 返回波长 lambda，计算公式为 c / nu
    return c / _asarray(nu, xp=xp, subok=True)
```