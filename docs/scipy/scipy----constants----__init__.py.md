# `D:\src\scipysrc\scipy\scipy\constants\__init__.py`

```
r"""
==================================
Constants (:mod:`scipy.constants`)
==================================

.. currentmodule:: scipy.constants

Physical and mathematical constants and units.


Mathematical constants
======================

================  =================================================================
``pi``            Pi
``golden``        Golden ratio
``golden_ratio``  Golden ratio
================  =================================================================


Physical constants
==================

===========================  =================================================================
``c``                        speed of light in vacuum
``speed_of_light``           speed of light in vacuum
``mu_0``                     the magnetic constant :math:`\mu_0`
``epsilon_0``                the electric constant (vacuum permittivity), :math:`\epsilon_0`
``h``                        the Planck constant :math:`h`
``Planck``                   the Planck constant :math:`h`
``hbar``                     :math:`\hbar = h/(2\pi)`
``G``                        Newtonian constant of gravitation
``gravitational_constant``   Newtonian constant of gravitation
``g``                        standard acceleration of gravity
``e``                        elementary charge
``elementary_charge``        elementary charge
``R``                        molar gas constant
``gas_constant``             molar gas constant
``alpha``                    fine-structure constant
``fine_structure``           fine-structure constant
``N_A``                      Avogadro constant
``Avogadro``                 Avogadro constant
``k``                        Boltzmann constant
``Boltzmann``                Boltzmann constant
``sigma``                    Stefan-Boltzmann constant :math:`\sigma`
``Stefan_Boltzmann``         Stefan-Boltzmann constant :math:`\sigma`
``Wien``                     Wien displacement law constant
``Rydberg``                  Rydberg constant
``m_e``                      electron mass
``electron_mass``            electron mass
``m_p``                      proton mass
``proton_mass``              proton mass
``m_n``                      neutron mass
``neutron_mass``             neutron mass
===========================  =================================================================


Constants database
------------------

In addition to the above variables, :mod:`scipy.constants` also contains the
2018 CODATA recommended values [CODATA2018]_ database containing more physical
constants.

.. autosummary::
   :toctree: generated/

   value      -- Value in physical_constants indexed by key
   unit       -- Unit in physical_constants indexed by key
   precision  -- Relative precision in physical_constants indexed by key
   find       -- Return list of physical_constant keys with a given string
   ConstantWarning -- Constant sought not in newest CODATA data set


"""
# 物理常数字典，格式为 `physical_constants[name] = (value, unit, uncertainty)`
.. data:: physical_constants

   Dictionary of physical constants, of the format
   ``physical_constants[name] = (value, unit, uncertainty)``.

   This dictionary stores various physical constants with their values, units, and uncertainties.

Available constants:

======================================================================  ====
%(constant_names)s
======================================================================  ====


Units
=====

SI prefixes
-----------

============  =================================================================
``quetta``    :math:`10^{30}`
``ronna``     :math:`10^{27}`
``yotta``     :math:`10^{24}`
``zetta``     :math:`10^{21}`
``exa``       :math:`10^{18}`
``peta``      :math:`10^{15}`
``tera``      :math:`10^{12}`
``giga``      :math:`10^{9}`
``mega``      :math:`10^{6}`
``kilo``      :math:`10^{3}`
``hecto``     :math:`10^{2}`
``deka``      :math:`10^{1}`
``deci``      :math:`10^{-1}`
``centi``     :math:`10^{-2}`
``milli``     :math:`10^{-3}`
``micro``     :math:`10^{-6}`
``nano``      :math:`10^{-9}`
``pico``      :math:`10^{-12}`
``femto``     :math:`10^{-15}`
``atto``      :math:`10^{-18}`
``zepto``     :math:`10^{-21}`
``yocto``     :math:`10^{-24}`
``ronto``     :math:`10^{-27}`
``quecto``    :math:`10^{-30}`
============  =================================================================

   This section lists SI prefixes with their corresponding powers of ten for unit conversion.

Binary prefixes
---------------

============  =================================================================
``kibi``      :math:`2^{10}`
``mebi``      :math:`2^{20}`
``gibi``      :math:`2^{30}`
``tebi``      :math:`2^{40}`
``pebi``      :math:`2^{50}`
``exbi``      :math:`2^{60}`
``zebi``      :math:`2^{70}`
``yobi``      :math:`2^{80}`
============  =================================================================

   This section lists binary prefixes with their corresponding powers of two.

Mass
----

=================  ============================================================
``gram``           :math:`10^{-3}` kg
``metric_ton``     :math:`10^{3}` kg
``grain``          one grain in kg
``lb``             one pound (avoirdupous) in kg
``pound``          one pound (avoirdupous) in kg
``blob``           one inch version of a slug in kg (added in 1.0.0)
``slinch``         one inch version of a slug in kg (added in 1.0.0)
``slug``           one slug in kg (added in 1.0.0)
``oz``             one ounce in kg
``ounce``          one ounce in kg
``stone``          one stone in kg
``grain``          one grain in kg
``long_ton``       one long ton in kg
``short_ton``      one short ton in kg
``troy_ounce``     one Troy ounce in kg
``troy_pound``     one Troy pound in kg
``carat``          one carat in kg
``m_u``            atomic mass constant (in kg)
``u``              atomic mass constant (in kg)
``atomic_mass``    atomic mass constant (in kg)
=================  ============================================================

   This section lists various mass units and their conversions to kilograms.

Angle
-----

=================  ============================================================
``degree``         degree in radians
``arcmin``         arc minute in radians
``arcminute``      arc minute in radians
``arcsec``         arc second in radians

   This section lists angle units and their conversions to radians.
# 弧度中的角秒
=================  ============================================================


Time
----

=================  ============================================================
# 秒中的一分钟
``minute``         one minute in seconds
# 小时中的一小时
``hour``           one hour in seconds
# 一天中的一天
``day``            one day in seconds
# 一周中的一周
``week``           one week in seconds
# 一年中的一年（365天）
``year``           one year (365 days) in seconds
# 一个儒略年（365.25天）中的一年
``Julian_year``    one Julian year (365.25 days) in seconds
=================  ============================================================


Length
------

=====================  ============================================================
# 米中的一英寸
``inch``               one inch in meters
# 米中的一英尺
``foot``               one foot in meters
# 米中的一码
``yard``               one yard in meters
# 米中的一英里
``mile``               one mile in meters
# 米中的一千分之一英寸
``mil``                one mil in meters
# 米中的一点（磅）
``pt``                 one point in meters
# 米中的一点（磅）
``point``              one point in meters
# 米中的一英尺（测量用）
``survey_foot``        one survey foot in meters
# 米中的一英里（测量用）
``survey_mile``        one survey mile in meters
# 米中的一海里
``nautical_mile``      one nautical mile in meters
# 米中的一费米
``fermi``              one Fermi in meters
# 米中的一埃
``angstrom``           one Angstrom in meters
# 米中的一微米
``micron``             one micron in meters
# 米中的一天文单位
``au``                 one astronomical unit in meters
# 米中的一天文单位
``astronomical_unit``  one astronomical unit in meters
# 米中的一光年
``light_year``         one light year in meters
# 米中的一秒差距
``parsec``             one parsec in meters
=====================  ============================================================

Pressure
--------

=================  ============================================================
# 标准大气压（帕斯卡）
``atm``            standard atmosphere in pascals
# 标准大气压（帕斯卡）
``atmosphere``     standard atmosphere in pascals
# 一巴（帕斯卡）
``bar``            one bar in pascals
# 一托（毫米汞柱）（帕斯卡）
``torr``           one torr (mmHg) in pascals
# 一托（毫米汞柱）（帕斯卡）
``mmHg``           one torr (mmHg) in pascals
# 一磅力/平方英寸（帕斯卡）
``psi``            one psi in pascals
=================  ============================================================

Area
----

=================  ============================================================
# 米中的一公顷
``hectare``        one hectare in square meters
# 米中的一英亩
``acre``           one acre in square meters
=================  ============================================================


Volume
------

===================    ========================================================
# 立方米中的一升
``liter``              one liter in cubic meters
# 立方米中的一升
``litre``              one liter in cubic meters
# 立方米中的一美制加仑
``gallon``             one gallon (US) in cubic meters
# 立方米中的一美制加仑
``gallon_US``          one gallon (US) in cubic meters
# 立方米中的一英制加仑
``gallon_imp``         one gallon (UK) in cubic meters
# 立方米中的一美制液体盎司
``fluid_ounce``        one fluid ounce (US) in cubic meters
# 立方米中的一美制液体盎司
``fluid_ounce_US``     one fluid ounce (US) in cubic meters
# 立方米中的一英制液体盎司
``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters
# 立方米中的一桶
``bbl``                one barrel in cubic meters
# 立方米中的一桶
``barrel``             one barrel in cubic meters
===================    ========================================================

Speed
-----
# 导入模块，包括基本的物理常数和数据
from ._codata import *
from ._constants import *
from ._codata import _obsolete_constants, physical_constants

# Deprecated namespaces, to be removed in v2.0.0
# 引入被弃用的命名空间，将在 v2.0.0 版本中移除
from . import codata, constants
# 创建一个常量名称列表，列表元素为元组(_k.lower(), _k, _v)，其中_k是物理常数的名称（小写形式），_v是其值，
# physical_constants 是一个包含物理常数及其值的字典。不包含已过时常数 _obsolete_constants。
_constant_names_list = [(_k.lower(), _k, _v)
                        for _k, _v in physical_constants.items()
                        if _k not in _obsolete_constants]

# 将常量名称列表转换为一个字符串，每个常量名格式化为 ``{}``格式，后面是常量的值和单位，每行长度为66个字符。
_constant_names = "\n".join(["``{}``{}  {} {}".format(_x[1], " "*(66-len(_x[1])),
                                                  _x[2][0], _x[2][1])
                             for _x in sorted(_constant_names_list)])

# 如果模块有文档字符串，则使用常量名称字符串来格式化文档字符串，替换其中的占位符 %s。
if __doc__:
    __doc__ = __doc__ % dict(constant_names=_constant_names)

# 删除不再需要的变量 _constant_names 和 _constant_names_list，释放内存。
del _constant_names
del _constant_names_list

# 设置模块的公开接口，__all__ 列表包含所有不以下划线开头的变量名。
__all__ = [s for s in dir() if not s.startswith('_')]

# 从 scipy._lib._testutils 模块导入 PytestTester 类，并创建一个测试对象 test。
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)

# 删除不再需要的 PytestTester 类，释放内存。
del PytestTester
```