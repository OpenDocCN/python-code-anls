# `D:\src\scipysrc\sympy\sympy\physics\units\tests\test_unit_system_cgs_gauss.py`

```
# 导入需要的模块和函数
from sympy.concrete.tests.test_sums_products import NS
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import convert_to, coulomb_constant, elementary_charge, gravitational_constant, planck
from sympy.physics.units.definitions.unit_definitions import angstrom, statcoulomb, coulomb, second, gram, centimeter, erg, \
    newton, joule, dyne, speed_of_light, meter, farad, henry, statvolt, volt, ohm
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.cgs import cgs_gauss

# 测试函数：将静电单位转换为国际单位制中库仑单位
def test_conversion_to_from_si():
    assert convert_to(statcoulomb, coulomb, cgs_gauss) == coulomb/2997924580
    assert convert_to(coulomb, statcoulomb, cgs_gauss) == 2997924580*statcoulomb
    assert convert_to(statcoulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == centimeter**(S(3)/2)*sqrt(gram)/second
    assert convert_to(coulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == 2997924580*centimeter**(S(3)/2)*sqrt(gram)/second

    # SI单位中电磁学部分不进行转换：
    assert convert_to(coulomb, statcoulomb, SI) == coulomb
    assert convert_to(statcoulomb, coulomb, SI) == statcoulomb

    # SI单位中不涉及电磁学：
    assert convert_to(erg, joule, SI) == joule/10**7
    assert convert_to(erg, joule, cgs_gauss) == joule/10**7
    assert convert_to(joule, erg, SI) == 10**7*erg
    assert convert_to(joule, erg, cgs_gauss) == 10**7*erg

    assert convert_to(dyne, newton, SI) == newton/10**5
    assert convert_to(dyne, newton, cgs_gauss) == newton/10**5
    assert convert_to(newton, dyne, SI) == 10**5*dyne
    assert convert_to(newton, dyne, cgs_gauss) == 10**5*dyne


# 测试函数：转换到CGS高斯单位制的物理常数
def test_cgs_gauss_convert_constants():

    assert convert_to(speed_of_light, centimeter/second, cgs_gauss) == 29979245800*centimeter/second

    assert convert_to(coulomb_constant, 1, cgs_gauss) == 1
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, cgs_gauss) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, dyne*centimeter**2/statcoulomb**2, cgs_gauss) == centimeter**2*dyne/statcoulomb**2
    assert convert_to(coulomb_constant, 1, SI) == coulomb_constant
    assert NS(convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI)) == '8987551787.36818*meter**2*newton/coulomb**2'

    assert convert_to(elementary_charge, statcoulomb, cgs_gauss)
    assert convert_to(angstrom, centimeter, cgs_gauss) == 1*centimeter/10**8
    assert convert_to(gravitational_constant, dyne*centimeter**2/gram**2, cgs_gauss)
    assert NS(convert_to(planck, erg*second, cgs_gauss)) == '6.62607015e-27*erg*second'

    spc = 25000*second/(22468879468420441*centimeter)
    assert convert_to(ohm, second/centimeter, cgs_gauss) == spc


这段代码包含了两个测试函数，分别用于测试单位转换和物理常数在不同单位制（CGS高斯单位制和国际单位制）中的转换。
    # 断言：将亨利单位转换为每平方厘米的秒平方单位，并与 spc*second 比较是否相等
    assert convert_to(henry, second**2/centimeter, cgs_gauss) == spc*second
    
    # 断言：将伏特单位转换为静电单位伏特，并与 10**6*statvolt/299792458 比较是否相等
    assert convert_to(volt, statvolt, cgs_gauss) == 10**6*statvolt/299792458
    
    # 断言：将法拉单位转换为厘米单位，并与 299792458**2*centimeter/10**5 比较是否相等
    assert convert_to(farad, centimeter, cgs_gauss) == 299792458**2*centimeter/10**5
```